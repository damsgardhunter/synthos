#!/usr/bin/env bash
# ============================================================
#  Deploy local GNN Python files to the GCP training worker
#
#  Usage:
#    bash scripts/deploy-gnn-to-gcp.sh            # deploy + restart + tail
#    bash scripts/deploy-gnn-to-gcp.sh --no-tail  # deploy + restart, no tail
#    bash scripts/deploy-gnn-to-gcp.sh --dry-run  # print actions, do nothing
#
#  What it does:
#    1. SCPs gnn/superconductor_gnn.py + gnn/server.py to /tmp on the box
#    2. SSHs in, moves them into /opt/quantum-alchemy/gnn/
#    3. Force-removes any cached __pycache__ so Python re-imports the new
#       source instead of an out-of-date .pyc
#    4. Restarts quantum-alchemy-gcp.service via systemctl
#    5. Tails journald for 60 seconds so you can see the new training
#       member fire and the FE pretrain logs start streaming
#
#  Why this exists: the GCP worker keeps its own copy of the GNN Python
#  files at /opt/quantum-alchemy/gnn/ — they are NOT mounted from the
#  repo. Until you push, the GCP service is running stale code regardless
#  of what's in your local working tree.
#
#  Override defaults via env vars before invoking:
#    GCP_PROJECT, GCP_ZONE, GCP_HOST, GCP_REMOTE_GNN_DIR, GCP_SERVICE_NAME
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_GNN_DIR="$PROJECT_ROOT/gnn"

# Defaults match the values from the working Colab notebook (Cell 15).
# Override by exporting these before running the script.
GCP_PROJECT="${GCP_PROJECT:-project-590ec869-5404-4bdc-9e8}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_HOST="${GCP_HOST:-damsgardhunter@instance-20260319-200127}"
GCP_REMOTE_GNN_DIR="${GCP_REMOTE_GNN_DIR:-/opt/quantum-alchemy/gnn}"
GCP_SERVICE_NAME="${GCP_SERVICE_NAME:-quantum-alchemy-gcp.service}"

# Files to sync. Add more here if you start touching graph_builder.py too.
# training_data.py and mp_fetch.py are imported by server.py at module load —
# they MUST land on the box together or the service will crash on restart.
FILES=(
  "superconductor_gnn.py"
  "server.py"
  "training_data.py"
  "mp_fetch.py"
)

DRY_RUN=0
TAIL=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-tail) TAIL=0 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      echo "Usage: $0 [--dry-run] [--no-tail]" >&2
      exit 2
      ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

# ── 0. Sanity check: do the local files exist? ──────────────────────────
echo "── Pre-flight ──────────────────────────────────────────"
for f in "${FILES[@]}"; do
  local_path="$LOCAL_GNN_DIR/$f"
  if [[ ! -f "$local_path" ]]; then
    echo "ERROR: $local_path not found" >&2
    exit 1
  fi
  size=$(wc -c < "$local_path" | tr -d ' ')
  echo "  $f  ($size bytes)"
done

# Quick syntax check so we don't ship a broken file across the network.
# `command -v` is not enough on Windows (the python3 shim returns success
# but execution hits the Microsoft Store stub). Actually invoke --version
# and check the exit code.
PY_BIN=""
for candidate in python3 python py; do
  if "$candidate" --version >/dev/null 2>&1; then
    PY_BIN="$candidate"
    break
  fi
done
if [[ -n "$PY_BIN" ]]; then
  for f in "${FILES[@]}"; do
    if ! "$PY_BIN" -m py_compile "$LOCAL_GNN_DIR/$f"; then
      echo "ERROR: $f failed py_compile — refusing to deploy broken code" >&2
      exit 1
    fi
  done
  echo "  Syntax check: OK ($PY_BIN)"
else
  echo "  Syntax check: SKIPPED (no working python on PATH)"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not on PATH — install the Google Cloud SDK first" >&2
  exit 1
fi

echo
echo "── Target ──────────────────────────────────────────────"
echo "  project : $GCP_PROJECT"
echo "  zone    : $GCP_ZONE"
echo "  host    : $GCP_HOST"
echo "  dir     : $GCP_REMOTE_GNN_DIR"
echo "  service : $GCP_SERVICE_NAME"
[[ "$DRY_RUN" -eq 1 ]] && echo "  mode    : DRY RUN (no actions performed)"
echo

# ── 1. SCP files to /tmp on the box ─────────────────────────────────────
echo "── Step 1/4: SCP files to remote /tmp ──────────────────"
SCP_ARGS=()
for f in "${FILES[@]}"; do
  SCP_ARGS+=("$LOCAL_GNN_DIR/$f")
done
SCP_ARGS+=("$GCP_HOST:/tmp/")

run gcloud compute scp \
  "${SCP_ARGS[@]}" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT"

# ── 2. Move into place + clear pyc cache + restart ──────────────────────
echo
echo "── Step 2/4: Move into $GCP_REMOTE_GNN_DIR + clear cache ──"

# Build a single remote command. We deliberately use sudo for the moves
# and the cache clear since the directory is owned by root in most setups.
# Each step is && so the next only runs if the previous succeeds.
REMOTE_CMD=""
for f in "${FILES[@]}"; do
  REMOTE_CMD+="sudo mv -f /tmp/$f $GCP_REMOTE_GNN_DIR/$f && "
done
REMOTE_CMD+="sudo find $GCP_REMOTE_GNN_DIR -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; "
REMOTE_CMD+="echo '[deploy] files in place at $(date -Iseconds)' && "
# Verify every file we shipped actually landed — catches silent SCP drops
# and tells you the size + mtime so you can sanity-check the deploy.
LS_TARGETS=""
for f in "${FILES[@]}"; do
  LS_TARGETS+="$GCP_REMOTE_GNN_DIR/$f "
done
REMOTE_CMD+="ls -la $LS_TARGETS"

run gcloud compute ssh "$GCP_HOST" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT" \
  --command="$REMOTE_CMD"

# ── 3. Restart the systemd service ──────────────────────────────────────
echo
echo "── Step 3/4: Restart $GCP_SERVICE_NAME ─────────────────"
run gcloud compute ssh "$GCP_HOST" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT" \
  --command="sudo systemctl restart $GCP_SERVICE_NAME && echo '[deploy] $GCP_SERVICE_NAME restarted at $(date -Iseconds)' && sudo systemctl is-active $GCP_SERVICE_NAME"

# ── 4. Tail journald so you can see the new training fire ──────────────
if [[ "$TAIL" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  echo
  echo "── Step 4/4: Tailing journald for 60s ─────────────────"
  echo "  Look for these lines to confirm the new code is live:"
  echo "    [ensemble] Training member 1/1 on N graphs…"
  echo "    Pre-training formation energy for 15 epochs…"
  echo "    FE pretrain 1/15  loss=…"
  echo "    Epoch 1/80  loss=…"
  echo "  If you see 'member 1/5' instead of 'member 1/1', the .pyc cache"
  echo "  wasn't fully cleared — re-run this script and it will retry."
  echo
  # `timeout` may not be available on some hosts; fall back to background+sleep+kill.
  if command -v timeout >/dev/null 2>&1; then
    timeout 60 gcloud compute ssh "$GCP_HOST" \
      --zone="$GCP_ZONE" \
      --project="$GCP_PROJECT" \
      --command="sudo journalctl -u $GCP_SERVICE_NAME -f -n 50 --no-pager" || true
  else
    gcloud compute ssh "$GCP_HOST" \
      --zone="$GCP_ZONE" \
      --project="$GCP_PROJECT" \
      --command="sudo journalctl -u $GCP_SERVICE_NAME -f -n 50 --no-pager" &
    TAIL_PID=$!
    sleep 60
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
  fi
else
  echo
  echo "── Step 4/4: Tail skipped ───────────────────────────────"
  echo "  To watch live logs:"
  echo "    gcloud compute ssh $GCP_HOST --zone=$GCP_ZONE --project=$GCP_PROJECT \\"
  echo "      --command='sudo journalctl -u $GCP_SERVICE_NAME -f -n 50'"
fi

echo
echo "── Deploy complete ─────────────────────────────────────"
echo "  Files synced : ${#FILES[@]}"
echo "  Service      : $GCP_SERVICE_NAME (restarted)"
echo "  Next training run will use the Colab-parity config:"
echo "    ENSEMBLE_SIZE=1, train_set (no leakage), no curriculum, verbose epochs"
