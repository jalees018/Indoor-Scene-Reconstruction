#!/usr/bin/env bash
# setup.sh — one-shot setup for the monocular indoor 3D reconstruction pipeline.
#
# What this does:
#   1. Checks for required system packages (cmake, eigen, opencv, glew, etc.).
#   2. Initializes ORB-SLAM3 + Pangolin git submodules.
#   3. Applies patches/orbslam3.patch to the ORB-SLAM3 submodule.
#   4. Creates / reuses the `slam-recon` conda env and installs Python deps.
#   5. Builds Pangolin and ORB-SLAM3 (including DBoW2 / g2o / Sophus thirdparty).
#   6. Extracts ORBvoc.txt from the bundled tarball.
#
# Idempotent: re-run after pulling new commits — already-done steps are skipped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${SLAM_RECON_ENV:-slam-recon}"
PY_VERSION="3.10"

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[setup]\033[0m %s\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
log "Checking system dependencies..."
missing=()
for cmd in cmake make g++ git pkg-config; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
for pc in eigen3 opencv4 glew; do
    pkg-config --exists "$pc" 2>/dev/null || missing+=("pkg-config:$pc")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    warn "Missing system packages: ${missing[*]}"
    warn "On Debian/Ubuntu, install with:"
    warn "  sudo apt-get install -y build-essential cmake git pkg-config \\"
    warn "      libeigen3-dev libopencv-dev libglew-dev libssl-dev \\"
    warn "      libboost-all-dev libpython3-dev python3-pip"
    die  "Re-run setup.sh after installing the above."
fi

# ---------------------------------------------------------------------------
# 2. Submodules
# ---------------------------------------------------------------------------
if [[ -f .gitmodules ]] && [[ ! -f ORB_SLAM3/CMakeLists.txt || ! -f Pangolin/CMakeLists.txt ]]; then
    log "Initializing git submodules..."
    git submodule update --init --recursive
fi
[[ -f ORB_SLAM3/CMakeLists.txt ]] || die "ORB_SLAM3/ not populated. Run: git submodule update --init --recursive"
[[ -f Pangolin/CMakeLists.txt   ]] || die "Pangolin/ not populated. Run: git submodule update --init --recursive"

# ---------------------------------------------------------------------------
# 3. Apply ORB-SLAM3 patches (idempotent — check with --reverse --check first)
# ---------------------------------------------------------------------------
PATCH="$REPO_ROOT/patches/orbslam3.patch"
if [[ -f "$PATCH" ]]; then
    pushd ORB_SLAM3 >/dev/null
    if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        log "ORB-SLAM3 patch already applied; skipping."
    elif git apply --check "$PATCH" >/dev/null 2>&1; then
        log "Applying patches/orbslam3.patch to ORB-SLAM3..."
        git apply "$PATCH"
    else
        warn "Patch neither cleanly applies nor cleanly reverses against current ORB-SLAM3 tree."
        warn "Manual inspection required: git -C ORB_SLAM3 status"
    fi
    popd >/dev/null
fi

# ---------------------------------------------------------------------------
# 4. Conda env + Python deps
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    die "conda not found in PATH. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating conda env '$ENV_NAME' (Python $PY_VERSION)..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
else
    log "Conda env '$ENV_NAME' already exists; reusing."
fi
# conda activate touches unbound variables in its own deactivate scripts.
set +u
conda activate "$ENV_NAME"
set -u

log "Installing Python dependencies into '$ENV_NAME'..."
python -m pip install --upgrade pip
python -m pip install \
    "torch==2.4.1" "torchvision==0.19.1" \
    --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install \
    "numpy>=2.0,<3" \
    "opencv-python>=4.10" \
    "open3d>=0.19" \
    "timm>=0.9" \
    "ninja"
# gsplat: must use --extra-index-url (the gsplat index does not host `ninja`).
python -m pip install gsplat==1.5.3 \
    --extra-index-url https://docs.gsplat.studio/whl/pt24cu121

# ---------------------------------------------------------------------------
# 5. Build Pangolin
# ---------------------------------------------------------------------------
if [[ ! -f Pangolin/build/src/libpango_core.so && ! -f Pangolin/build/libpango_core.so ]]; then
    log "Building Pangolin..."
    cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release
    cmake --build Pangolin/build -- -j"$(nproc)"
else
    log "Pangolin already built; skipping."
fi

# ---------------------------------------------------------------------------
# 6. Build ORB-SLAM3 (uses its own build.sh which also builds DBoW2 / g2o / Sophus)
# ---------------------------------------------------------------------------
if [[ ! -f ORB_SLAM3/lib/libORB_SLAM3.so ]]; then
    log "Building ORB-SLAM3 (this takes ~5-10 minutes)..."
    pushd ORB_SLAM3 >/dev/null
    chmod +x build.sh
    ./build.sh
    popd >/dev/null
else
    log "ORB-SLAM3 already built (lib/libORB_SLAM3.so present); skipping."
fi

# ---------------------------------------------------------------------------
# 7. Vocabulary file
# ---------------------------------------------------------------------------
if [[ ! -s ORB_SLAM3/Vocabulary/ORBvoc.txt && -s ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz ]]; then
    log "Extracting ORBvoc.txt..."
    tar -xzf ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C ORB_SLAM3/Vocabulary
fi
[[ -s ORB_SLAM3/Vocabulary/ORBvoc.txt ]] || die "ORBvoc.txt missing. Expected at ORB_SLAM3/Vocabulary/ORBvoc.txt"

# ---------------------------------------------------------------------------
log "Setup complete."
log "Next:  conda activate $ENV_NAME  &&  ./run.sh <video> <calib.yaml>"
