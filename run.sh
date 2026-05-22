#!/usr/bin/env bash
# run.sh — end-to-end pipeline: video -> Gaussian-splat scene.
#
# Usage:
#   ./run.sh <video> <calib.yaml> [options]
#
# Stages (in order):
#   1a. Extract frames at --fps into <out>/rgb/ + <out>/rgb.txt.
#   1b. ORB-SLAM3 (mono_tum) -> CameraTrajectory.txt + MapPoints.txt in <out>/.
#   2.  MiDaS depth + scale fit + fusion -> <out>/scene_pointcloud.ply (+ mesh).
#   3.  Gaussian Splatting refinement     -> <out>/splat/splats.ply.
#
# Options:
#   --out DIR          Output dir (default: output_<video basename>).
#   --fps N            Frame extraction rate (default: 5).
#   --pose-tol S       Frame<->pose match tolerance in seconds (default: 0.1).
#   --depth-min M      Pass 2 near clip in meters (default: 0.2).
#   --depth-max M      Pass 2 far clip  in meters (default: 5.0).
#   --stride N         Pass 2 pixel stride (default: 4).
#   --voxel-size M     Pass 3 init voxel size in meters (default: 0.02).
#   --iters N          Pass 3 training iterations (default: 7000).
#   --image-scale S    Pass 3 train-time downscale factor (default: 1.0).
#   --skip-splat       Stop after Pass 2 (point cloud + mesh).
#   --skip-mesh        Pass 2 only: skip Poisson mesh generation.
#   --resume           Reuse existing frames / SLAM outputs if present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SLAM_RECON_ENV:-slam-recon}"

log()  { printf "\033[1;34m[run]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[run]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[run]\033[0m %s\n" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
run.sh — end-to-end pipeline: video -> Gaussian-splat scene.

Usage:
  ./run.sh <video> <calib.yaml> [options]

Stages (in order):
  1a. Extract frames at --fps into <out>/rgb/ + <out>/rgb.txt.
  1b. ORB-SLAM3 (mono_tum) -> CameraTrajectory.txt + MapPoints.txt in <out>/.
  2.  MiDaS depth + scale fit + fusion -> <out>/scene_pointcloud.ply (+ mesh).
  3.  Gaussian Splatting refinement     -> <out>/splat/splats.ply.

Options:
  --out DIR          Output dir (default: output_<video basename>).
  --fps N            Frame extraction rate (default: 5).
  --pose-tol S       Frame<->pose match tolerance in seconds (default: 0.1).
  --depth-min M      Pass 2 near clip in meters (default: 0.2).
  --depth-max M      Pass 2 far clip  in meters (default: 5.0).
  --stride N         Pass 2 pixel stride (default: 4).
  --voxel-size M     Pass 3 init voxel size in meters (default: 0.02).
  --iters N          Pass 3 training iterations (default: 7000).
  --image-scale S    Pass 3 train-time downscale factor (default: 1.0).
  --skip-splat       Stop after Pass 2 (point cloud + mesh).
  --skip-mesh        Pass 2 only: skip Poisson mesh generation.
  --resume           Reuse existing frames / SLAM outputs if present.
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
VIDEO=""; CALIB=""; OUT=""
FPS=5
POSE_TOL=0.1
DEPTH_MIN=0.2
DEPTH_MAX=5.0
STRIDE=4
VOXEL=0.02
ITERS=7000
IMAGE_SCALE=1.0
SKIP_SPLAT=0
SKIP_MESH=0
RESUME=0

positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --out)         OUT="$2"; shift 2 ;;
        --fps)         FPS="$2"; shift 2 ;;
        --pose-tol)    POSE_TOL="$2"; shift 2 ;;
        --depth-min)   DEPTH_MIN="$2"; shift 2 ;;
        --depth-max)   DEPTH_MAX="$2"; shift 2 ;;
        --stride)      STRIDE="$2"; shift 2 ;;
        --voxel-size)  VOXEL="$2"; shift 2 ;;
        --iters)       ITERS="$2"; shift 2 ;;
        --image-scale) IMAGE_SCALE="$2"; shift 2 ;;
        --skip-splat)  SKIP_SPLAT=1; shift ;;
        --skip-mesh)   SKIP_MESH=1; shift ;;
        --resume)      RESUME=1; shift ;;
        --) shift; positional+=("$@"); break ;;
        -*) die "Unknown flag: $1 (run --help)" ;;
        *) positional+=("$1"); shift ;;
    esac
done
[[ ${#positional[@]} -ge 2 ]] || usage 1
VIDEO="${positional[0]}"
CALIB="${positional[1]}"

[[ -f "$VIDEO" ]] || die "Video not found: $VIDEO"
[[ -f "$CALIB" ]] || die "Calibration YAML not found: $CALIB"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
VIDEO_ABS="$(cd "$(dirname "$VIDEO")" && pwd)/$(basename "$VIDEO")"
CALIB_ABS="$(cd "$(dirname "$CALIB")" && pwd)/$(basename "$CALIB")"
if [[ -z "$OUT" ]]; then
    base="$(basename "${VIDEO%.*}")"
    OUT="$REPO_ROOT/output_$base"
fi
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

VOCAB="$REPO_ROOT/ORB_SLAM3/Vocabulary/ORBvoc.txt"
MONO_TUM="$REPO_ROOT/ORB_SLAM3/Examples/Monocular/mono_tum"
[[ -s "$VOCAB"    ]] || die "Vocabulary missing: $VOCAB (run ./setup.sh first)"
[[ -x "$MONO_TUM" ]] || die "mono_tum binary missing: $MONO_TUM (run ./setup.sh first)"

# ---------------------------------------------------------------------------
# Conda env
# ---------------------------------------------------------------------------
command -v conda >/dev/null 2>&1 || die "conda not in PATH"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate touches unbound variables in its own deactivate scripts.
set +u
conda activate "$ENV_NAME" || { set -u; die "Conda env '$ENV_NAME' not found. Run ./setup.sh first."; }
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Pass 1a: extract frames
# ---------------------------------------------------------------------------
if [[ "$RESUME" == 1 && -s "$OUT_ABS/rgb.txt" ]]; then
    log "[1a] Resume: reusing existing $OUT_ABS/rgb.txt"
else
    log "[1a] Extracting frames from $VIDEO_ABS at ${FPS} fps -> $OUT_ABS"
    python "$REPO_ROOT/Run_Init.py" \
        --video "$VIDEO_ABS" \
        --out   "$OUT_ABS" \
        --fps   "$FPS" \
        --extract-only
fi

# ---------------------------------------------------------------------------
# Pass 1b: ORB-SLAM3 (writes CameraTrajectory.txt + MapPoints.txt to cwd)
# ---------------------------------------------------------------------------
TRAJ="$OUT_ABS/CameraTrajectory.txt"
MAPPTS="$OUT_ABS/MapPoints.txt"
if [[ "$RESUME" == 1 && -s "$TRAJ" && -s "$MAPPTS" ]]; then
    log "[1b] Resume: reusing existing $TRAJ + $MAPPTS"
else
    log "[1b] Running ORB-SLAM3 monocular SLAM..."
    pushd "$OUT_ABS" >/dev/null
    "$MONO_TUM" "$VOCAB" "$CALIB_ABS" "$OUT_ABS"
    popd >/dev/null
    [[ -s "$TRAJ"   ]] || die "ORB-SLAM3 did not produce CameraTrajectory.txt — likely tracking loss. Check intrinsics + ensure the first seconds of the video contain translation, not just rotation."
    [[ -s "$MAPPTS" ]] || die "ORB-SLAM3 did not produce MapPoints.txt."
    n_poses=$(grep -vc '^#' "$TRAJ" || true)
    log "[1b] Got $n_poses per-frame poses."
fi

# ---------------------------------------------------------------------------
# Pass 2: MiDaS + scale fit + fusion
# ---------------------------------------------------------------------------
PCD="$OUT_ABS/scene_pointcloud.ply"
if [[ "$RESUME" == 1 && -s "$PCD" ]]; then
    log "[2]  Resume: reusing existing $PCD"
else
    log "[2]  MiDaS depth + scale fit + dense fusion..."
    mesh_arg=()
    [[ "$SKIP_MESH" == 1 ]] && mesh_arg=(--skip-mesh)
    python "$REPO_ROOT/Run_Init.py" \
        --frames-dir "$OUT_ABS" \
        --trajectory "$TRAJ" \
        --map-points "$MAPPTS" \
        --calib      "$CALIB_ABS" \
        --pose-tol   "$POSE_TOL" \
        --depth-min  "$DEPTH_MIN" \
        --depth-max  "$DEPTH_MAX" \
        --stride     "$STRIDE" \
        --out        "$OUT_ABS" \
        "${mesh_arg[@]}"
fi

if [[ "$SKIP_SPLAT" == 1 ]]; then
    log "Done (--skip-splat). Point cloud: $PCD"
    exit 0
fi

# ---------------------------------------------------------------------------
# Pass 3: Gaussian Splatting
# ---------------------------------------------------------------------------
log "[3]  3D Gaussian Splatting refinement..."
python "$REPO_ROOT/Run_Splat.py" \
    --frames-dir  "$OUT_ABS" \
    --trajectory  "$TRAJ" \
    --calib       "$CALIB_ABS" \
    --init-cloud  "$PCD" \
    --out         "$OUT_ABS/splat" \
    --voxel-size  "$VOXEL" \
    --iters       "$ITERS" \
    --image-scale "$IMAGE_SCALE"

log "Done."
log "  Point cloud: $PCD"
log "  Mesh:        $OUT_ABS/scene_mesh.ply"
log "  Splats:      $OUT_ABS/splat/splats.ply"
log "  Eval views:  $OUT_ABS/splat/eval/"
