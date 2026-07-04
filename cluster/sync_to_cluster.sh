#!/bin/bash
# ============================================================
# sync_to_cluster.sh — Upload/sync code to the cluster
# ============================================================
# Usage:  bash cluster/sync_to_cluster.sh [--dry-run] [--print-dest]
#
# Run from your local machine (project root) whenever you
# want to push code changes to the cluster.
#
# Connection settings come from cluster/cluster.env (gitignored):
#   cp cluster/cluster.env.example cluster/cluster.env   # then edit
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  echo "Create it from the template:" >&2
  echo "  cp ${SCRIPT_DIR}/cluster.env.example ${SCRIPT_DIR}/cluster.env   # then edit" >&2
  exit 1
fi
# shellcheck source=cluster.env.example
source "${ENV_FILE}"
: "${CLUSTER_USER:?CLUSTER_USER not set in ${ENV_FILE}}"
: "${CLUSTER_HOST:?CLUSTER_HOST not set in ${ENV_FILE}}"
: "${REMOTE_DIR:?REMOTE_DIR not set in ${ENV_FILE}}"

REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"
RSYNC_FLAGS=(-avz --progress)

for arg in "$@"; do
  case "${arg}" in
    --print-dest) echo "${REMOTE}:${REMOTE_DIR}/"; exit 0 ;;
    --dry-run)    RSYNC_FLAGS+=(-n) ;;
    *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

echo "Syncing code to ${REMOTE}:${REMOTE_DIR} ..."

rsync "${RSYNC_FLAGS[@]}" \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='/data/' \
    --exclude='notebooks/data/' \
    --exclude='*.sqsh' \
    --exclude='logs/' \
    --exclude='docs/scratch/' \
    --exclude='reports/figures/' \
    --exclude='cluster/cluster.env' \
    "${PROJECT_ROOT}/" \
    "${REMOTE}:${REMOTE_DIR}/"

echo ""
echo "Done. Code synced to ${REMOTE}:${REMOTE_DIR}"
