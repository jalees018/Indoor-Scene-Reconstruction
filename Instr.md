# Instructions for 3D reconstruction given a video of a small indoor scene

After installing ORB-SLAM3, the next step is to use it to get camera poses, then feed those poses into your reconstruction script instead of the fake motion. ORB-SLAM3 is a visual SLAM system, so its main job is tracking the camera trajectory and producing a consistent world coordinate system for your video.

What to do next
Verify ORB-SLAM3 runs on a test sequence first.
Before using your phone video, run one of the provided monocular examples and confirm that you get a live tracking window and a trajectory output.

Extract your phone video into frames.
Your reconstruction script works on frames, so save the video as an image sequence or load it frame-by-frame in Python.

Run ORB-SLAM3 on the same frames.
For a monocular indoor video, you want the monocular pipeline and a matching camera calibration file. ORB-SLAM3 Python wrappers exist, and the typical setup uses the vocabulary file plus a YAML file with intrinsics.

Save the camera trajectory.
ORB-SLAM3 outputs poses in a trajectory format, often TUM-style or a keyframe trajectory. That pose stream is what your PyTorch script should use for backprojecting depth into a common 3D frame.

Replace the fake translation in your script.
In the Run_Init.py script, remove the line that adds a made-up camera shift and instead transform each frame’s 3D points using the real ORB-SLAM3 pose for that frame.

Your pipeline should become:

video frame 

ORB-SLAM3 pose 


MiDaS depth for frame 

backproject pixels into camera coordinates,

transform points into world coordinates with the ORB pose,

fuse all frames into one cloud or mesh.

That is the key change. Without real poses, the points from different frames cannot align correctly, so the room will look stretched or duplicated.

Pose format
ORB-SLAM-style outputs are commonly interpreted as camera pose in a world frame or a world-to-camera transform, so you must check the exact convention before using them. One common issue is quaternion ordering and whether the file stores camera-to-world or world-to-camera transforms, so you should verify that carefully before multiplying matrices.

Practical step-by-step plan
Run ORB-SLAM3 on a small test video until it works.

Export the trajectory file.

Parse the trajectory in Python.

For each frame, compute depth with MiDaS.

Backproject depth to 3D using intrinsics.

Apply the pose transform from ORB-SLAM3.

Merge all points into one cloud.

Optional: run mesh reconstruction or Gaussian splatting afterward.

# Current State

End-to-end flow (Pass 1 → SLAM → Pass 2 → Pass 3):

  1. Pass 1 — extract frames from the video (writes rgb/ + rgb.txt):
  python Run_Init.py --video input.mp4 --out output --extract-only

  2. Run mono_tum on those frames (writes KeyFrameTrajectory.txt + CameraTrajectory.txt + MapPoints.txt to ORB_SLAM3/):
  cd ORB_SLAM3
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt /path/to/calib.yaml ../output

  3. Pass 2 — fuse with per-frame poses + scale-aligned depth:
  python Run_Init.py --frames-dir output --out output --calib calib.yaml \
      --trajectory ORB_SLAM3/CameraTrajectory.txt \
      --map-points ORB_SLAM3/MapPoints.txt \
      --pose-tol 0.005

  Use CameraTrajectory.txt (per-frame poses), NOT KeyFrameTrajectory.txt.
  The keyframe-only file has ~50 poses for ~800 frames and causes visible
  smearing / ghost-layer artifacts when each non-keyframe gets snapped to a
  neighboring keyframe's pose.

  What the scale alignment does (fit_disparity_affine in Run_Init.py):
  - Projects all SLAM map points into the frame using T_cw = inv(T_wc) + intrinsics, keeps the ones in front of the camera and inside the image.
  - Samples raw MiDaS disparity at those pixels.
  - Robust IRLS fit (Cauchy-style weights, 3 iterations) of α·midas + β = 1/Z_slam, giving a 2-parameter affine in disparity space — the recommended MiDaS calibration model.
  - Inverts α·pred + β to get metric(ish) dense depth, then backprojects.
  - If <--min-points-per-frame (default 20) map points land in the image, falls back to the old heuristic depth for that frame and increments unscaled so you can see how many frames missed alignment.

  4. Pass 3 — Gaussian Splatting refinement (gsplat):
  python Run_Splat.py --frames-dir output \
      --trajectory ORB_SLAM3/CameraTrajectory.txt \
      --calib calib.yaml \
      --init-cloud output/scene_pointcloud.ply \
      --out output/splat \
      --voxel-size 0.02 --iters 7000

  Seeds Gaussians from a voxel-downsampled Pass-2 cloud, undistorts frames
  with the calib YAML, runs L1+SSIM training with the gsplat DefaultStrategy
  densifier, and writes splats.ply (Inria-3DGS-compatible, loadable in
  SuperSplat / antimatter15's viewer) plus held-out eval PNGs.

  Caveat that's still there: monocular SLAM itself is up-to-scale, so the whole reconstruction is metric relative to the SLAM map — i.e. consistent across frames, but you'd still need a known reference (e.g. one known distance in the scene, or IMU) to recover absolute meters. For non-metric multi-view
  fusion this is exactly the level of consistency you want.
