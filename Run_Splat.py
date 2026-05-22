"""Pass 3: 3D Gaussian Splatting refinement on top of ORB-SLAM3 + MiDaS outputs.

Inputs:
    - TUM-style frames directory (rgb/ + rgb.txt) — same one Pass 1 ingested.
    - ORB-SLAM3 CameraTrajectory.txt — per-frame camera-to-world poses (Twc).
    - ORB-SLAM3 calibration YAML — fx/fy/cx/cy + distortion coefficients.
    - Init point cloud (default: scene_pointcloud.ply from Pass 2), voxel-
      downsampled to seed Gaussian positions/colors.

Outputs (in --out):
    - splats.ply         — 3DGS-format point cloud (load in SuperSplat etc.).
    - eval/*.png         — renders of held-out frames every --eval-every iters.
    - eval/gt_*.png      — GT versions of those frames for side-by-side.

Pose convention: ORB-SLAM3 writes Twc (camera-to-world). gsplat wants Tcw
(world-to-camera) view matrices, so we invert.

Coordinate frame: ORB-SLAM3 uses OpenCV (z-forward, y-down). gsplat defaults
to the same — no axis flips needed.
"""

import argparse
import math
import os
import random

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from torch import nn

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--init-cloud", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--voxel-size", type=float, default=0.02,
                   help="Voxel size (m) for downsampling the init cloud.")
    p.add_argument("--iters", type=int, default=7000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-stride", type=int, default=8,
                   help="Every Nth frame is held out for eval.")
    p.add_argument("--pose-tol", type=float, default=0.005)
    p.add_argument("--image-scale", type=float, default=1.0,
                   help="Downscale image side by this factor (>1 = smaller).")
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--no-densify", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_calib(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")

    def first(*keys, required=True, default=0.0):
        for k in keys:
            node = fs.getNode(k)
            if not node.empty():
                return float(node.real())
        if required:
            raise KeyError(f"None of {keys} in {path}")
        return default

    fx = first("Camera.fx", "Camera1.fx")
    fy = first("Camera.fy", "Camera1.fy")
    cx = first("Camera.cx", "Camera1.cx")
    cy = first("Camera.cy", "Camera1.cy")
    k1 = first("Camera.k1", "Camera1.k1", required=False)
    k2 = first("Camera.k2", "Camera1.k2", required=False)
    p1 = first("Camera.p1", "Camera1.p1", required=False)
    p2 = first("Camera.p2", "Camera1.p2", required=False)
    k3 = first("Camera.k3", "Camera1.k3", required=False)
    fs.release()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def load_trajectory(path):
    times, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            T = np.eye(4)
            T[:3, :3] = quat_to_R(qx, qy, qz, qw)
            T[:3, 3] = (tx, ty, tz)
            times.append(t)
            poses.append(T)
    return np.asarray(times), np.stack(poses, 0)


def load_frames_index(frames_dir):
    with open(os.path.join(frames_dir, "rgb.txt")) as f:
        items = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            items.append((float(parts[0]), os.path.join(frames_dir, parts[1])))
    return items


def match_frames_to_poses(frame_index, traj_t, traj_T, tol):
    out = []
    for t, path in frame_index:
        i = int(np.argmin(np.abs(traj_t - t)))
        if abs(traj_t[i] - t) <= tol:
            out.append((t, path, traj_T[i]))
    return out


def init_gaussians(cloud_path, voxel, device):
    pcd = o3d.io.read_point_cloud(cloud_path)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() \
        else np.full((len(pts), 3), 0.5, dtype=np.float32)
    print(f"Init from {cloud_path}: {len(pts):,} gaussians "
          f"(voxel={voxel} m)")

    # Scale init: log of distance to nearest neighbor (standard 3DGS init).
    tree = o3d.geometry.KDTreeFlann(pcd)
    nn_dists = np.empty(len(pts), dtype=np.float32)
    for i, p in enumerate(pts):
        _, idx, d2 = tree.search_knn_vector_3d(p, 2)
        nn_dists[i] = math.sqrt(d2[1]) if len(d2) > 1 else voxel
    nn_dists = np.clip(nn_dists, 1e-4, 1.0)
    log_scales = np.log(nn_dists)[:, None].repeat(3, axis=1)

    quats = np.zeros((len(pts), 4), dtype=np.float32)
    quats[:, 0] = 1.0  # gsplat convention: (w, x, y, z) with w in slot 0

    # Inverse sigmoid of 0.1 ~ -2.197
    opacities_logit = np.full(len(pts), -2.197, dtype=np.float32)

    # Store raw (pre-sigmoid) colors. We'll sigmoid at render time so they stay in [0,1].
    colors_logit = np.log(np.clip(cols, 1e-4, 1 - 1e-4) /
                          (1 - np.clip(cols, 1e-4, 1 - 1e-4))).astype(np.float32)

    params = nn.ParameterDict({
        "means":     nn.Parameter(torch.from_numpy(pts).to(device)),
        "scales":    nn.Parameter(torch.from_numpy(log_scales).to(device)),
        "quats":     nn.Parameter(torch.from_numpy(quats).to(device)),
        "opacities": nn.Parameter(torch.from_numpy(opacities_logit).to(device)),
        "colors":    nn.Parameter(torch.from_numpy(colors_logit).to(device)),
    })
    return params


def build_optimizers(params, scene_scale):
    # LRs from the 3DGS paper, scaled by scene extent for the means.
    lrs = {
        "means":     1.6e-4 * scene_scale,
        "scales":    5e-3,
        "quats":     1e-3,
        "opacities": 5e-2,
        "colors":    2.5e-3,
    }
    return {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15)
            for k, lr in lrs.items()}


def render(params, viewmat, K, w, h, near=0.01, far=1e10):
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.sigmoid(params["colors"]),
        viewmats=viewmat,
        Ks=K,
        width=w,
        height=h,
        near_plane=near,
        far_plane=far,
        packed=False,
        sh_degree=None,
        render_mode="RGB",
    )
    return out, alpha, info


def ssim(a, b, window=11):
    # Lightweight Gaussian-window SSIM. a, b: (1, 3, H, W) in [0, 1].
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5
    coords = torch.arange(window, dtype=a.dtype, device=a.device) - (window - 1) / 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    kx = g.view(1, 1, 1, window).expand(3, 1, 1, window)
    ky = g.view(1, 1, window, 1).expand(3, 1, window, 1)

    def blur(x):
        x = F.conv2d(x, kx, padding=(0, window // 2), groups=3)
        return F.conv2d(x, ky, padding=(window // 2, 0), groups=3)

    mu_a, mu_b = blur(a), blur(b)
    sa = blur(a * a) - mu_a ** 2
    sb = blur(b * b) - mu_b ** 2
    sab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2)
    return (num / den).mean()


def save_splats_ply(params, out_path):
    # Inria-3DGS-compatible PLY: one float property per channel, SH degree 0
    # (we store DC only, derived from sigmoided RGB).
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    opacities = params["opacities"].detach().cpu().numpy()
    rgb = torch.sigmoid(params["colors"]).detach().cpu().numpy()
    # SH(0) coefficient that yields the given RGB when evaluated.
    SH_C0 = 0.28209479177387814
    f_dc = (rgb - 0.5) / SH_C0

    n = means.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )
    zeros = np.zeros((n, 3), dtype=np.float32)
    data = np.concatenate([
        means.astype(np.float32),
        zeros,                              # normals (unused)
        f_dc.astype(np.float32),
        opacities[:, None].astype(np.float32),
        scales.astype(np.float32),
        quats.astype(np.float32),
    ], axis=1)
    with open(out_path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("gsplat requires CUDA.")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)

    K_np, dist = load_calib(args.calib)
    print(f"K =\n{K_np}\ndist = {dist}")

    traj_t, traj_T = load_trajectory(args.trajectory)
    print(f"Loaded {len(traj_t)} poses")

    frame_index = load_frames_index(args.frames_dir)
    print(f"Indexed {len(frame_index)} frames")

    matched = match_frames_to_poses(frame_index, traj_t, traj_T, args.pose_tol)
    print(f"{len(matched)} frames matched to a pose within {args.pose_tol}s")
    if not matched:
        raise SystemExit("No frame-pose matches.")

    # Probe image size + adjust K for image_scale.
    first_img = cv2.imread(matched[0][1], cv2.IMREAD_COLOR)
    h0, w0 = first_img.shape[:2]
    scale = args.image_scale
    h, w = int(h0 / scale), int(w0 / scale)
    K_scaled = K_np.copy()
    K_scaled[0, 0] /= scale; K_scaled[1, 1] /= scale
    K_scaled[0, 2] /= scale; K_scaled[1, 2] /= scale
    print(f"Image size: {w0}x{h0} -> {w}x{h} (scale={scale})")

    # Pre-load + undistort + resize all images. fr1/xyz at 640x480 is fine in
    # RAM; for larger sets we'd stream from disk.
    print("Loading and undistorting frames...")
    images = []
    poses_wc = []
    for t, path, Twc in matched:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.undistort(rgb_img, K_np, dist)
        if scale != 1.0:
            rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_AREA)
        images.append(torch.from_numpy(rgb_img).to(device).float() / 255.0)
        poses_wc.append(Twc)
    poses_wc = np.stack(poses_wc, 0)
    poses_cw = np.linalg.inv(poses_wc).astype(np.float32)

    # Train / eval split.
    indices = list(range(len(matched)))
    eval_idx = indices[::args.eval_stride]
    train_idx = [i for i in indices if i not in set(eval_idx)]
    print(f"Train: {len(train_idx)} | Eval: {len(eval_idx)}")

    # Scene scale = norm of camera centers spread, used to scale means LR.
    cam_centers = poses_wc[:, :3, 3]
    scene_scale = float(np.linalg.norm(cam_centers - cam_centers.mean(0), axis=1).max())
    scene_scale = max(scene_scale, 0.1)
    print(f"scene_scale = {scene_scale:.3f} m")

    params = init_gaussians(args.init_cloud, args.voxel_size, device)
    optimizers = build_optimizers(params, scene_scale)

    if args.no_densify:
        strategy = None
        strategy_state = None
    else:
        strategy = DefaultStrategy(
            refine_start_iter=500,
            refine_stop_iter=int(args.iters * 0.7),
            reset_every=3000,
            refine_every=100,
            prune_opa=0.005,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            verbose=False,
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    K_t = torch.from_numpy(K_scaled.astype(np.float32))[None].to(device)
    poses_cw_t = torch.from_numpy(poses_cw).to(device)

    print(f"Training for {args.iters} iters...")
    for step in range(args.iters):
        i = random.choice(train_idx)
        gt = images[i].permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        viewmat = poses_cw_t[i:i + 1]

        out, alpha, info = render(params, viewmat, K_t, w, h)
        # out: (1, H, W, 3)
        pred = out.permute(0, 3, 1, 2)
        loss_l1 = F.l1_loss(pred, gt)
        loss_ssim = 1.0 - ssim(pred, gt)
        loss = (1 - args.ssim_weight) * loss_l1 + args.ssim_weight * loss_ssim

        if strategy is not None:
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if strategy is not None:
            strategy.step_post_backward(params, optimizers, strategy_state, step, info,
                                        packed=False)

        if step % 100 == 0 or step == args.iters - 1:
            n_gs = params["means"].shape[0]
            print(f"  step {step:5d}  loss={loss.item():.4f}  "
                  f"(l1={loss_l1.item():.4f}, ssim={1 - loss_ssim.item():.3f})  "
                  f"#gaussians={n_gs:,}")

        if (step + 1) % args.eval_every == 0 or step == args.iters - 1:
            with torch.no_grad():
                for j in eval_idx[:4]:
                    out, _, _ = render(params, poses_cw_t[j:j + 1], K_t, w, h)
                    img = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(args.out, "eval",
                                             f"step{step + 1:05d}_view{j:03d}.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if step == args.iters - 1:
                        gt_img = (images[j].cpu().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(args.out, "eval",
                                                 f"gt_view{j:03d}.png"),
                                    cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))

    out_ply = os.path.join(args.out, "splats.ply")
    save_splats_ply(params, out_ply)
    print(f"Saved {params['means'].shape[0]:,} gaussians to {out_ply}")


if __name__ == "__main__":
    main()
