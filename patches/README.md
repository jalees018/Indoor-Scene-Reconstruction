# Patches

## `orbslam3.patch`

Applies on top of upstream ORB-SLAM3 at commit
`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` ("Update Dependencies.md").

Changes:

1. **`Examples/Monocular/mono_tum.cc`** — disables the Pangolin viewer
   (headless server compatibility) and adds calls to `SaveTrajectoryTUM` and
   `SaveMapPoints` after shutdown so Pass 2 has per-frame poses + the sparse
   map.
2. **`src/System.cc::SaveTrajectoryTUM`** — removes the upstream early-return
   that blocks monocular sensors. The relative-pose accumulation below the
   guard works fine for monocular and is required for per-frame (not
   keyframe-only) pose export.
3. **`src/System.cc::SaveMapPoints` + `include/System.h`** — new method that
   dumps the active map's world-frame landmarks as `x y z` text.
4. **CMake / OpenCV 4 fixes** — bumps `cmake_minimum_required` to 3.5, switches
   to C++14, and replaces deprecated OpenCV 3 constants (`CV_LOAD_IMAGE_*`,
   `CV_GRAY2BGR`, `CV_FONT_HERSHEY_DUPLEX`) in `LoopClosing.cc`. These are
   build-only changes; behavior is unchanged.

Applied automatically by `setup.sh`. To apply manually:

```bash
cd ORB_SLAM3
git apply ../patches/orbslam3.patch
```
