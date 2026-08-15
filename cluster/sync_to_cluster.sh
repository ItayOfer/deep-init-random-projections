#!/bin/bash
# ============================================================
# sync_to_cluster.sh — Upload/sync code to the cluster
# ============================================================
# Usage:  bash cluster/sync_to_cluster.sh [--dry-run] [--print-dest] [--tar]
#
# Run from your local machine (project root) whenever you
# want to push code changes to the cluster.
#
# Connection settings come from cluster/cluster.env (gitignored):
#   cp cluster/cluster.env.example cluster/cluster.env   # then edit
# Point at a different node with CLUSTER_ENV=cluster/cluster.env.<name>.
#
# If the remote reports "rsync: command not found" (seen on the Aug 2026
# re-imaged DLC login node), you have two options:
#   * rsync exists but is off the non-interactive PATH — rsync-over-ssh runs a
#     NON-login shell, so module inits and ~/.bash_profile do not apply. Give
#     the absolute path:   RSYNC_PATH=/usr/local/bin/rsync bash cluster/sync_to_cluster.sh
#   * rsync is genuinely absent — use the tar fallback, which needs only tar
#     and ssh on the remote and uses ONE connection (one password prompt):
#                          bash cluster/sync_to_cluster.sh --tar
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
USE_TAR=0
DRY_RUN=0

# One exclude list, shared by both transports. rsync and tar agree on these
# patterns; the leading-slash form ('/data/') is rsync-anchored, so the tar
# path uses its own equivalent below.
EXCLUDES=(
  '__pycache__' '.git' '*.pyc' '.ipynb_checkpoints'
  'notebooks/data' '*.sqsh' 'logs' 'docs/scratch'
  'reports/figures' 'cluster/cluster.env'
)

for arg in "$@"; do
  case "${arg}" in
    --print-dest) echo "${REMOTE}:${REMOTE_DIR}/"; exit 0 ;;
    --dry-run)    DRY_RUN=1; RSYNC_FLAGS+=(-n) ;;
    --tar)        USE_TAR=1 ;;
    *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

echo "Syncing code to ${REMOTE}:${REMOTE_DIR} ..."

if [[ "${USE_TAR}" -eq 1 ]]; then
  # Fallback transport: needs only tar + ssh on the remote. One connection, so
  # one password prompt. Like the rsync call below it does NOT delete remote
  # files that vanished locally -- it overlays.
  TAR_EXCLUDES=()
  for pattern in "${EXCLUDES[@]}"; do TAR_EXCLUDES+=(--exclude="${pattern}"); done
  TAR_EXCLUDES+=(--exclude='./data')          # top-level dataset dir only

  # macOS bsdtar embeds Apple metadata (com.apple.provenance, FinderInfo, BSD
  # fflags) that GNU tar on the cluster does not understand, producing one
  # "Ignoring unknown extended header keyword" warning PER FILE -- thousands of
  # lines that bury the real output. None of it is meaningful for source code on
  # Linux. Probe for each flag rather than assuming, so this stays portable to a
  # GNU-tar client. COPYFILE_DISABLE additionally suppresses AppleDouble ._*.
  export COPYFILE_DISABLE=1
  TAR_META_FLAGS=()
  for flag in --no-mac-metadata --no-xattrs --no-fflags; do
    if tar czf /dev/null "${flag}" -T /dev/null >/dev/null 2>&1; then
      TAR_META_FLAGS+=("${flag}")
    fi
  done
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry run: listing what would be sent, not transferring)"
    # bsdtar (macOS) writes the -v listing to stderr; fold it into stdout so the
    # list is greppable/pipeable the way a dry run should be.
    tar czf /dev/null -v "${TAR_META_FLAGS[@]}" "${TAR_EXCLUDES[@]}" -C "${PROJECT_ROOT}" . 2>&1
    exit 0
  fi
  tar czf - "${TAR_META_FLAGS[@]}" "${TAR_EXCLUDES[@]}" -C "${PROJECT_ROOT}" . \
    | ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR} && tar xzf - -C ${REMOTE_DIR}"
else
  RSYNC_EXCLUDES=()
  for pattern in "${EXCLUDES[@]}"; do RSYNC_EXCLUDES+=(--exclude="${pattern}"); done
  RSYNC_EXCLUDES+=(--exclude='/data/')        # top-level dataset dir only
  # RSYNC_PATH lets you name the remote rsync binary when it is installed but
  # not on the non-interactive PATH (rsync-over-ssh runs a non-login shell).
  RSYNC_PATH_FLAG=()
  if [[ -n "${RSYNC_PATH:-}" ]]; then RSYNC_PATH_FLAG=(--rsync-path="${RSYNC_PATH}"); fi

  if ! rsync "${RSYNC_FLAGS[@]}" "${RSYNC_PATH_FLAG[@]}" "${RSYNC_EXCLUDES[@]}" \
        "${PROJECT_ROOT}/" "${REMOTE}:${REMOTE_DIR}/"; then
    status=$?
    echo "" >&2
    echo "rsync failed (exit ${status}). If the remote said 'rsync: command not found':" >&2
    echo "  * rsync installed but off the non-interactive PATH:" >&2
    echo "      RSYNC_PATH=/path/to/rsync bash cluster/sync_to_cluster.sh" >&2
    echo "  * rsync genuinely absent — use the tar fallback (tar+ssh only):" >&2
    echo "      bash cluster/sync_to_cluster.sh --tar" >&2
    exit "${status}"
  fi
fi

echo ""
echo "Done. Code synced to ${REMOTE}:${REMOTE_DIR}"
