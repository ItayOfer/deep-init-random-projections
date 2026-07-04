#!/bin/bash
# ============================================================
# pull_results.sh — Fetch result JSONs and SLURM logs back
# ============================================================
# Usage:  bash cluster/pull_results.sh <label-glob> [campaign-dir]
#
#   bash cluster/pull_results.sh 'rcfwd_rescale_smoke_*' 09_rcfwd_rescale
#     -> reports/results/rcfwd_rescale_smoke_*.json
#     -> logs/slurm/09_rcfwd_rescale/rcfwd_rescale_smoke_*-<JOBID>.out
#
# Without a campaign dir, logs go to logs/slurm/.
# Connection settings come from cluster/cluster.env (gitignored).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${CLUSTER_ENV:-${SCRIPT_DIR}/cluster.env}"

if [[ $# -lt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | sed -n '2,13p'
  exit 0
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  echo "  cp ${SCRIPT_DIR}/cluster.env.example ${SCRIPT_DIR}/cluster.env   # then edit" >&2
  exit 1
fi
source "${ENV_FILE}"
: "${CLUSTER_USER:?CLUSTER_USER not set in ${ENV_FILE}}"
: "${CLUSTER_HOST:?CLUSTER_HOST not set in ${ENV_FILE}}"
: "${REMOTE_DIR:?REMOTE_DIR not set in ${ENV_FILE}}"

LABEL="$1"
CAMPAIGN="${2:-}"
REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"
LOG_DIR="${PROJECT_ROOT}/logs/slurm${CAMPAIGN:+/${CAMPAIGN}}"
mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/reports/results"

# Quote remote globs — they must expand on the cluster, not in local zsh.
echo "Pulling results for '${LABEL}' from ${REMOTE} ..."
scp "${REMOTE}:${REMOTE_DIR}/reports/results/${LABEL}.json" "${PROJECT_ROOT}/reports/results/" || \
  echo "  (no matching result JSONs)"
scp "${REMOTE}:${REMOTE_DIR}/${LABEL}-*.out" "${LOG_DIR}/" || \
  echo "  (no matching .out logs)"

echo "Done."
