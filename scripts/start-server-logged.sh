#!/usr/bin/env bash
# ============================================================
#  QAE Server Startup with Log Capture
#  Usage: bash scripts/start-server-logged.sh
#
#  Starts the server if not already running, tees all stdout/stderr
#  to logs/server-latest.log so the monitor loop can read it.
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/server-latest.log"
PID_FILE="$LOG_DIR/server.pid"
# Read PORT from .env if present, default to 5000
_ENV_PORT=$(grep -E "^PORT=" "$PROJECT_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '[:space:]')
HOST="${QAE_HOST:-http://localhost:${_ENV_PORT:-5000}}"

mkdir -p "$LOG_DIR"

# ── Helper: is server already responding? ────────────────────
server_running() {
  # 10s timeout: GradientBoost training can block the Node.js event loop for
  # several seconds; 3s was causing false-positive "server unresponsive" kills.
  curl -sf --max-time 10 "$HOST/api/health" > /dev/null 2>&1
}

# ── Helper: is our tracked PID still alive? ─────────────────
# On Windows, `kill -0 PID` only works for PIDs that bash itself spawned.
# After the PID file is updated to the netstat-discovered Windows process PID
# (e.g. the actual tsx/node child), bash has no record of it and kill -0 returns
# false even though the process is healthy. Use `tasklist` for a reliable
# Windows-native liveness check.
pid_alive() {
  local _pid
  _pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [[ -n "$_pid" ]] && tasklist //FI "PID eq ${_pid}" 2>/dev/null | grep -q "${_pid}"
}

# ── Check if already up ──────────────────────────────────────
if server_running; then
  echo "[start] QAE server already running at $HOST"
  echo "[start] Log: $LOG_FILE"
  # Refresh PID file to the actual listening process so the next cycle's
  # pid_alive check doesn't see a stale npm wrapper PID and fall through
  # to the orphan-cleanup block (which would kill the running server).
  _ENV_PORT_EARLY=$(grep -E "^PORT=" "$PROJECT_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '[:space:]')
  _PORT_EARLY="${_ENV_PORT_EARLY:-5000}"
  _LIVE_PID=$(netstat -ano 2>/dev/null \
    | grep ":${_PORT_EARLY}[[:space:]].*LISTEN" \
    | awk '{print $NF}' | tr -d '[:space:][:cntrl:]' | head -1)
  if [[ -n "$_LIVE_PID" ]]; then
    _OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "none")
    if [[ "$_LIVE_PID" != "$_OLD_PID" ]]; then
      echo "[start] Refreshing PID file: $_OLD_PID → $_LIVE_PID"
      echo "$_LIVE_PID" > "$PID_FILE"
    fi
  fi
  export LOG_FILE
  exit 0
fi

# ── If tracked PID is still alive, server is busy (not crashed) ──────────
# GradientBoost training blocks the Node.js event loop for up to 30s, causing
# HTTP health checks to time out even though the process is healthy.  If the
# tracked PID (real node process, not npm wrapper) is still alive, trust it and
# exit without killing anything.  HTTP unresponsiveness during training is
# transient; the monitor-check script will detect any actual stuck state.
if pid_alive; then
  _BUSY_PID=$(cat "$PID_FILE")
  echo "[start] Process PID $_BUSY_PID is alive (HTTP may be blocked by training). Skipping restart."
  export LOG_FILE
  exit 0
fi

# ── Kill any untracked node process holding the server port ──
# (handles orphans left when the PID file is out of sync)
_ENV_PORT_VAL=$(grep -E "^PORT=" "$PROJECT_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '[:space:]')
_SERVER_PORT="${_ENV_PORT_VAL:-5000}"
ORPHAN_PIDS=$(netstat -ano 2>/dev/null | grep ":${_SERVER_PORT}[[:space:]].*LISTEN" \
  | awk '{print $NF}' | tr -d '[:space:][:cntrl:]' || true)
if [[ -n "$ORPHAN_PIDS" ]]; then
  for OPID in $ORPHAN_PIDS; do
    echo "[start] Killing orphan process PID $OPID holding port ${_SERVER_PORT} ..."
    taskkill //F //PID "$OPID" 2>/dev/null || kill "$OPID" 2>/dev/null || true
  done
  sleep 2
fi

# ── Archive previous log ──────────────────────────────────────
if [[ -f "$LOG_FILE" && -s "$LOG_FILE" ]]; then
  ARCHIVE="$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
  cp "$LOG_FILE" "$ARCHIVE"
  echo "[start] Archived previous log → $ARCHIVE"
fi

# ── Start server ──────────────────────────────────────────────
echo "[start] Starting QAE server (logging to $LOG_FILE) ..."
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== QAE Server Start: $START_TS ===" > "$LOG_FILE"
# Record start time so the monitor can filter out stale pre-restart client errors
echo "$START_TS" > "$LOG_DIR/server-start-time.txt"

cd "$PROJECT_ROOT"
# Run in background, capture both stdout and stderr
npm run dev >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "[start] Server process PID: $SERVER_PID"

# ── Wait for readiness (up to 60s) ────────────────────────────
echo "[start] Waiting for server to be ready..."
READY=0
for i in $(seq 1 30); do
  sleep 2
  if server_running; then
    echo "[start] Server ready after $((i * 2))s"
    READY=1
    break
  fi
  # Check if process died
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[start] ERROR: Server process died. Last log lines:"
    tail -20 "$LOG_FILE"
    exit 1
  fi
  echo "[start] ... waiting (${i}/30)"
done

if [[ $READY -eq 0 ]]; then
  echo "[start] WARNING: Server not responding after 60s. Check $LOG_FILE"
  tail -30 "$LOG_FILE"
fi

# ── Update PID file to actual listening process ───────────────
# npm run dev spawns: npm → cross-env → tsx → node.  The npm wrapper ($!)
# exits shortly after spawning its children, so the saved PID quickly becomes
# stale.  The next monitor cycle then falls through to the orphan-cleanup block
# and kills the real running server.  Fix: after readiness is confirmed, replace
# the npm wrapper PID with the PID of the process actually listening on the port.
if [[ $READY -eq 1 ]]; then
  ACTUAL_PID=$(netstat -ano 2>/dev/null \
    | grep ":${_SERVER_PORT}[[:space:]].*LISTEN" \
    | awk '{print $NF}' | tr -d '[:space:][:cntrl:]' | head -1)
  if [[ -n "$ACTUAL_PID" && "$ACTUAL_PID" != "$SERVER_PID" ]]; then
    echo "[start] Updating PID file: npm wrapper $SERVER_PID → actual server PID $ACTUAL_PID"
    echo "$ACTUAL_PID" > "$PID_FILE"
  fi
fi

export LOG_FILE
echo "[start] LOG_FILE=$LOG_FILE"
