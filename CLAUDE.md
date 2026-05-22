# CLAUDE.md

Project for turning a monocular indoor video into a 3D reconstruction by combining
ORB-SLAM3 (camera poses + sparse map) with MiDaS (dense per-frame depth) and
fusing into a single point cloud / mesh with Open3D.

See `README.md` for the user-facing pipeline description and end-to-end commands.

## Layout

- `Run_Init.py` — Python driver. Input is either `--video <path>` (Pass 1
  extracts frames and writes a TUM-style `rgb.txt`) or `--frames-dir <path>`
  (already-extracted `rgb/` + `rgb.txt`, preserves original timestamps —
  required when validating against TUM-style datasets so timestamps match
  the trajectory file). `--extract-only` exits after writing frames, useful
  when feeding the resulting TUM-layout folder to `mono_tum`. Pass 2 reads
  the ORB-SLAM3 trajectory + map points, runs MiDaS, scale-aligns disparity
  per frame (`fit_disparity_affine`), backprojects, and writes
  `scene_pointcloud.ply` + `scene_mesh.ply`. Pose-to-frame matching uses
  `--pose-tol` seconds (default: half the median keyframe interval).
- `Run_Splat.py` — Pass 3: 3D Gaussian Splatting refinement (gsplat 1.5+).
  Seeds Gaussians from a voxel-downsampled `scene_pointcloud.ply`, undistorts
  frames using the calib YAML, runs L1+SSIM training with the DefaultStrategy
  densifier, and writes `splats.ply` (Inria-3DGS-compatible) plus held-out
  eval PNGs every `--eval-every` iters. Pose convention is the same as
  Pass 2 (ORB-SLAM3 Twc; the script inverts to Tcw for gsplat viewmats).
- `ORB_SLAM3/` — vendored ORB-SLAM3, already built (build dir intact, lib at
  `ORB_SLAM3/lib/libORB_SLAM3.so`, binaries under `ORB_SLAM3/Examples/`).
- `Pangolin/` — viewer dependency, built.
- `Data/RGBD/rgbd_dataset_freiburg1_xyz/` — extracted TUM fr1/xyz sequence used
  as the ORB-SLAM3 smoke test (tgz still present alongside).

## Conda environment: `slam-recon`

Always activate before running anything Python or ORB-SLAM3 — ORB-SLAM3 also
needs `LD_LIBRARY_PATH` pointed at the env's libs (libstdc++ mismatch
otherwise).

```bash
source /home/jn48tivu/miniconda3/etc/profile.d/conda.sh
conda activate slam-recon
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Versions in the env (probed 2026-05-20): python 3.10.20, opencv 4.13.0,
torch 2.4.1+cu121 (CUDA available), numpy 2.2.6, open3d 0.19.0,
gsplat 1.5.3+pt24cu121 (prebuilt wheel — no nvcc on this box). MiDaS is
loaded via `torch.hub.load("intel-isl/MiDaS", ...)`.

gsplat install (if you ever recreate the env):

```bash
pip install ninja
pip install gsplat --extra-index-url https://docs.gsplat.studio/whl/pt24cu121
```

Do NOT use `--index-url` (only `--extra-index-url`) — the gsplat index does
not host `ninja`, so a plain `--index-url` breaks resolution.

## Headless server

No `$DISPLAY`, no Xvfb. Pangolin viewer is disabled in the patched
`ORB_SLAM3/Examples/Monocular/mono_tum.cc` (the 4th arg of the `System`
constructor is `false`, not vanilla `true`). If you edit other example
binaries (`mono_euroc`, etc.) and want to run them here, apply the same
patch and rebuild — `make -C ORB_SLAM3/build <target>`.

## Local patches to ORB-SLAM3 vs upstream

- `mono_tum.cc`: viewer disabled (above) **and** calls
  `SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt")`,
  `SLAM.SaveTrajectoryTUM("CameraTrajectory.txt")`, and
  `SLAM.SaveMapPoints("MapPoints.txt")` after shutdown. The three files land
  in the cwd from which mono_tum was launched. `CameraTrajectory.txt` is
  what Pass 2 of `Run_Init.py` consumes — it has per-frame poses, not just
  keyframes, so dense MiDaS depths get back-projected at the actual frame
  pose. Using `KeyFrameTrajectory.txt` causes visible smearing (frames
  snapped to the nearest keyframe several cm / a few degrees away).
- `System::SaveTrajectoryTUM`: upstream early-returns for `MONOCULAR`. We
  removed that guard — the relative-pose accumulation underneath works fine
  for monocular and we need it for Pass 2.
- `System::SaveMapPoints` exists in the modified ORB-SLAM3 source (added
  alongside the existing `SaveTrajectoryTUM` / `SaveKeyFrameTrajectoryTUM`).

## Smoke test (validated)

```bash
cd ORB_SLAM3
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml \
    ../Data/RGBD/rgbd_dataset_freiburg1_xyz
```

Expected on fr1/xyz: 798 frames in, ~52–59 keyframes out, ~3400–3600 map
points, ~796 per-frame poses in `CameraTrajectory.txt`, no tracking loss,
~31 ms median tracking time. KF count varies slightly run-to-run; the
per-frame pose count is the stable signal.

## Known caveat

Monocular SLAM is up-to-scale, so the final reconstruction is metric
*relative to the SLAM map* but not in absolute meters. Need a known scene
reference (or IMU) to recover real scale.
