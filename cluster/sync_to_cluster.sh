#!/bin/bash
# ============================================================
# sync_to_cluster.sh — Upload/sync code to the DLC
# ============================================================
# Usage:  bash cluster/sync_to_cluster.sh
#
# Run from your local machine (project root) whenever you
# want to push code changes to the cluster.
# ============================================================

set -euo pipefail

REMOTE="user@cluster"
REMOTE_DIR="~/thesis"

echo "Syncing code to ${REMOTE}:${REMOTE_DIR} ..."

rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='/data/' \
    --exclude='*.sqsh' \
    --exclude='logs/' \
    /Users/itayofer/Thesis/Thesis/ \
    "${REMOTE}:${REMOTE_DIR}/"

echo ""
echo "Done. Code synced to ${REMOTE}:${REMOTE_DIR}"
