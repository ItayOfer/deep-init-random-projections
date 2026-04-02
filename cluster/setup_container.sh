#!/bin/bash
# ============================================================
# setup_container.sh — One-time container setup on the DLC
# ============================================================
# Run this ONCE after uploading your code to the cluster.
# It pulls an NGC PyTorch container, installs your extra
# dependencies, and exports a ready-to-use squash file.
#
# Usage:  bash ~/thesis/cluster/setup_container.sh
# ============================================================

set -euo pipefail

# --- Configuration ---
CONTAINER_NAME="nvidia_pt"
NGC_VERSION="${NGC_VERSION:-24.07}"   # Override with: NGC_VERSION=24.12 bash setup_container.sh
SQSH_FILE="${CONTAINER_NAME}.sqsh"
CODE_DIR="$HOME/thesis"

echo "=== Step 1: Pull NGC PyTorch container (version ${NGC_VERSION}) ==="
if [ ! -f "nvidia+pytorch+${NGC_VERSION}-py3.sqsh" ]; then
    enroot import "docker://nvcr.io#nvidia/pytorch:${NGC_VERSION}-py3"
else
    echo "Squash file already exists, skipping pull."
fi

echo ""
echo "=== Step 2: Create container ==="
# Remove existing container if it exists
enroot remove -f "${CONTAINER_NAME}" 2>/dev/null || true
enroot create --name "${CONTAINER_NAME}" "nvidia+pytorch+${NGC_VERSION}-py3.sqsh"

echo ""
echo "=== Step 3: Install dependencies inside container ==="
enroot start --root --rw --mount "${CODE_DIR}:/mount" "${CONTAINER_NAME}" \
    /bin/bash -c "
        pip install -r /mount/requirements.txt && \
        echo '--- Verifying imports ---' && \
        python -c \"
import sys
sys.path.insert(0, '/mount/src')
import torch
import numpy
import matplotlib
import sklearn
import rp_study.models.initializers
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('NumPy:', numpy.__version__)
print('ALL OK')
\"
    "

echo ""
echo "=== Step 4: Export modified container ==="
enroot export --output "${SQSH_FILE}" "${CONTAINER_NAME}"

echo ""
echo "=== Step 5: Cleanup ==="
enroot remove -f "${CONTAINER_NAME}" 2>/dev/null || true
# Keep the original sqsh in case we need to redo setup
# rm -f "nvidia+pytorch+${NGC_VERSION}-py3.sqsh"

echo ""
echo "============================================"
echo "Container ready: ~/${SQSH_FILE}"
echo "You can now submit jobs with:"
echo "  sbatch ~/thesis/cluster/test_job.sub"
echo "============================================"
