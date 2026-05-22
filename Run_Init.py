import os
import math
import argparse

import cv2
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract frames from a video, run MiDaS depth, and fuse with "
                    "ORB-SLAM3 poses into a single point cloud / mesh.")
    p.add_argument("--video", default=None,
                   help="Input video. Mutually exclusive with --frames-dir.")
    p.add_argument("--frames-dir", default=None,
                   help="Directory containing rgb/ + rgb.txt (TUM layout). "
                        "Skips frame extraction and preserves original timestamps.")
    p.add_argument("--out", default="output")
    p.add_argument("--fps", type=float, default=2.0,
                   help="target sampling rate for frame extraction (--video mode only)")
    p.add_argument("--extract-only", action="store_true",
                   help="In --video mode, write rgb/ + rgb.txt and exit (skip MiDaS / fusion). "
                        "Use this to prepare a TUM-layout folder for ORB-SLAM3 mono_tum.")
    p.add_argument("--trajectory", default=None,
                   help="TUM-format trajectory from ORB-SLAM3 (camera-to-world).")
    p.add_argument("--pose-tol", type=float, default=None,
                   help="Max |frame_t - keyframe_t| in seconds for nearest_pose. "
                        "Default: half the median keyframe interval (auto).")
    p.add_argument("--map-points", default=None,
                   help="MapPoints.txt from ORB-SLAM3 (one world-frame xyz per line). "
                        "Enables per-frame MiDaS scale alignment to SLAM geometry.")
    p.add_argument("--min-points-per-frame", type=int, default=20,
                   help="Skip a frame's scale fit if fewer map points project into it.")
    p.add_argument("--calib", default=None,
                   help="ORB-SLAM3 YAML with Camera.fx/.fy/.cx/.cy.")
    p.add_argument("--midas-model", default="DPT_Hybrid",
                   choices=["DPT_Hybrid", "DPT_Large", "MiDaS_small"])
    p.add_argument("--depth-min", type=float, default=0.2)
    p.add_argument("--depth-max", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=4,
                   help="pixel stride for backprojection")
    p.add_argument("--skip-mesh", action="store_true")
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()


def load_calib(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration: {path}")

    def first_real(*keys):
        for k in keys:
            node = fs.getNode(k)
            if not node.empty():
                return float(node.real())
        raise KeyError(f"None of {keys} found in {path}")

    fx = first_real("Camera.fx", "Camera1.fx")
    fy = first_real("Camera.fy", "Camera1.fy")
    cx = first_real("Camera.cx", "Camera1.cx")
    cy = first_real("Camera.cy", "Camera1.cy")
    fs.release()
    return fx, fy, cx, cy


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def load_trajectory(path):
    # TUM format: `timestamp tx ty tz qx qy qz qw`, camera-to-world (ORB-SLAM3
    # SaveTrajectoryTUM convention).
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
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quat_to_R(qx, qy, qz, qw)
            T[:3, 3] = (tx, ty, tz)
            times.append(t)
            poses.append(T)
    return np.asarray(times), np.stack(poses, 0) if poses else np.empty((0, 4, 4))


def nearest_pose(t, times, poses, tol):
    if len(times) == 0:
        return None
    i = int(np.argmin(np.abs(times - t)))
    if abs(times[i] - t) > tol:
        return None
    return poses[i]


def load_map_points(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float64)


def fit_disparity_affine(disp_midas, depth_slam):
    # Robust LS fit of: disp_slam = alpha * disp_midas + beta, where
    # disp_slam = 1 / depth_slam. Returns (alpha, beta) or None on degeneracy.
    # Reweighted least squares with a Cauchy-like kernel (~3 iterations is enough).
    disp_slam = 1.0 / np.maximum(depth_slam, 1e-6)
    x = disp_midas.astype(np.float64)
    y = disp_slam.astype(np.float64)
    if x.size < 3:
        return None

    w = np.ones_like(x)
    for _ in range(3):
        A = np.stack([x, np.ones_like(x)], axis=1) * w[:, None]
        b = y * w
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        alpha, beta = sol
        r = (alpha * x + beta) - y
        s = np.median(np.abs(r)) + 1e-6
        w = 1.0 / (1.0 + (r / (1.4826 * s)) ** 2)

    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    return float(alpha), float(beta)


def extract_frames(video_path, out_dir, target_fps):
    frames_dir = os.path.join(out_dir, "rgb")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = max(1, int(round(src_fps / target_fps)))
    frames, timestamps = [], []
    rgb_lines = ["# color images\n", "# timestamp filename\n"]
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            t = idx / src_fps
            fname = f"{t:.6f}.png"
            cv2.imwrite(os.path.join(frames_dir, fname), frame_bgr)
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            timestamps.append(t)
            rgb_lines.append(f"{t:.6f} rgb/{fname}\n")
        idx += 1
    cap.release()

    with open(os.path.join(out_dir, "rgb.txt"), "w") as f:
        f.writelines(rgb_lines)

    return frames, timestamps, w, h, src_fps, step


def load_frames_dir(path):
    rgb_txt = os.path.join(path, "rgb.txt")
    if not os.path.isfile(rgb_txt):
        raise RuntimeError(f"Missing rgb.txt in {path}")
    frames, timestamps = [], []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            img_path = os.path.join(path, parts[1])
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Failed to read {img_path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            timestamps.append(t)
    if not frames:
        raise RuntimeError(f"No frames listed in {rgb_txt}")
    h, w = frames[0].shape[:2]
    if len(timestamps) > 1:
        dts = np.diff(np.asarray(timestamps))
        src_fps = 1.0 / float(np.median(dts))
    else:
        src_fps = 30.0
    return frames, timestamps, w, h, src_fps, 1


def main():
    args = parse_args()
    if bool(args.video) == bool(args.frames_dir):
        raise SystemExit("Specify exactly one of --video or --frames-dir.")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.frames_dir:
        frames, timestamps, w, h, src_fps, step = load_frames_dir(args.frames_dir)
        print(f"Loaded {len(frames)} frames from {args.frames_dir} "
              f"(inferred src_fps={src_fps:.2f})")
    else:
        frames, timestamps, w, h, src_fps, step = extract_frames(
            args.video, args.out, args.fps)
        print(f"Extracted {len(frames)} frames -> {args.out}/rgb/ + {args.out}/rgb.txt")
        if args.extract_only:
            print("--extract-only set; skipping MiDaS / fusion.")
            return

    if args.calib:
        fx, fy, cx, cy = load_calib(args.calib)
        print(f"Calibration: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    else:
        fx = fy = 0.9 * max(w, h)
        cx, cy = w / 2.0, h / 2.0
        print(f"WARNING: no --calib; using heuristic fx={fx:.2f}")

    if args.trajectory:
        traj_t, traj_T = load_trajectory(args.trajectory)
        if len(traj_t) == 0:
            raise RuntimeError(f"Empty trajectory: {args.trajectory}")
        if args.pose_tol is not None:
            pose_tol = float(args.pose_tol)
        elif len(traj_t) > 1:
            # 0.5 * 90th-percentile keyframe gap, capped at 0.5s. Median is too
            # small on bimodal KF distributions (dense clusters + occasional
            # wide gaps); p90 better reflects the gap most frames sit in.
            p90 = float(np.percentile(np.diff(traj_t), 90))
            pose_tol = max(0.1, min(0.5, 0.5 * p90))
        else:
            pose_tol = max(0.1, step / src_fps)
        print(f"Loaded {len(traj_t)} poses from {args.trajectory} "
              f"(pose_tol={pose_tol:.3f}s)")
    else:
        traj_t = traj_T = pose_tol = None
        print("WARNING: no --trajectory; falling back to fake forward translation.")

    if args.map_points:
        if traj_T is None:
            raise RuntimeError("--map-points requires --trajectory.")
        map_pts_w = load_map_points(args.map_points)
        print(f"Loaded {len(map_pts_w)} map points from {args.map_points}")
        if len(map_pts_w) == 0:
            map_pts_w = None
    else:
        map_pts_w = None

    midas = torch.hub.load("intel-isl/MiDaS", args.midas_model).to(device).eval()
    tfs = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = (tfs.dpt_transform
                 if args.midas_model in ("DPT_Hybrid", "DPT_Large")
                 else tfs.small_transform)

    all_points, all_colors = [], []
    used = skipped = unscaled = 0
    for i, img in enumerate(frames):
        if traj_T is not None:
            T_wc = nearest_pose(timestamps[i], traj_t, traj_T, pose_tol)
            if T_wc is None:
                skipped += 1
                continue
        else:
            T_wc = np.eye(4)
            T_wc[2, 3] = -0.05 * i

        input_batch = transform(img).to(device)
        with torch.no_grad():
            pred = midas(input_batch)
            pred = F.interpolate(pred.unsqueeze(1), size=img.shape[:2],
                                 mode="bicubic", align_corners=False).squeeze()
        pred_np = pred.detach().cpu().numpy()

        scaled = False
        if map_pts_w is not None:
            # World -> camera frame: p_c = R_wc^T (p_w - t_wc)
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            p_cam = (map_pts_w - t_wc) @ R_wc
            z = p_cam[:, 2]
            ok = z > 1e-3
            u = fx * p_cam[ok, 0] / z[ok] + cx
            v = fy * p_cam[ok, 1] / z[ok] + cy
            in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            uu = u[in_img].astype(np.int32)
            vv = v[in_img].astype(np.int32)
            z_sel = z[ok][in_img]

            if z_sel.size >= args.min_points_per_frame:
                disp_m = pred_np[vv, uu]
                fit = fit_disparity_affine(disp_m, z_sel)
                if fit is not None:
                    alpha, beta = fit
                    disp_dense = alpha * pred_np + beta
                    depth = 1.0 / np.maximum(disp_dense, 1e-3)
                    depth = np.clip(depth, args.depth_min, args.depth_max)
                    scaled = True

        if not scaled:
            unscaled += 1
            depth = pred_np - pred_np.min()
            depth = depth / (depth.max() + 1e-8)
            depth = 1.0 / (depth + 0.05)
            depth = np.clip(depth, args.depth_min, args.depth_max)

        s = args.stride
        ys, xs = np.mgrid[0:h:s, 0:w:s]
        z = depth[0:h:s, 0:w:s]
        cols = img[0:h:s, 0:w:s].astype(np.float32) / 255.0

        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        pts_cam = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
        pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
        pts_world = (T_wc @ pts_h.T).T[:, :3]

        all_points.append(pts_world)
        all_colors.append(cols.reshape(-1, 3))
        used += 1

    print(f"Backprojected {used} frames; skipped {skipped} (no pose); "
          f"{unscaled} fell back to heuristic depth (no scale fit).")
    if not all_points:
        raise RuntimeError("No frames had usable poses. Aborting.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    mask = np.isfinite(points).all(axis=1)
    points, colors = points[mask], colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    pcd_path = os.path.join(args.out, "scene_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud to {pcd_path}")

    if not args.skip_mesh:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(50)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        densities = np.asarray(densities)
        keep = densities > np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(~keep)
        mesh_path = os.path.join(args.out, "scene_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Saved mesh to {mesh_path}")

    if args.visualize:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
