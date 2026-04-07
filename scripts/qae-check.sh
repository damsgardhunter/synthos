#!/usr/bin/env bash
# ============================================================
#  QAE Diagnostic Commands
#  Usage: source scripts/qae-check.sh
#         Then call any function below, e.g.:  qae-cycles
#
#  Set LOG_FILE to point at your server log for file-based checks:
#    export LOG_FILE=/tmp/server-myrun.log
# ============================================================

HOST="${QAE_HOST:-http://localhost:5000}"
LOG_FILE="${LOG_FILE:-}"

_qae_api() { curl -sf "${HOST}$1"; }
_needs_log() {
  if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
    echo "Set LOG_FILE=/path/to/server.log first (e.g. /tmp/server-fix6.log)"
    return 1
  fi
}

# ── Engine Status ────────────────────────────────────────────
# Quick snapshot: state, tempo, active tasks, status message
qae-status() {
  echo "=== Engine Status ==="
  _qae_api "/api/engine/status" | python3 -m json.tool 2>/dev/null \
    || _qae_api "/api/engine/status"
}

# ── Cycle Health ─────────────────────────────────────────────
# Show last N cycle start/end events from DB logs
qae-cycles() {
  local n="${1:-20}"
  echo "=== Last $n Cycle Events ==="
  _qae_api "/api/research-logs?limit=$n&event=cycleStart,cycleEnd" 2>/dev/null \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
for l in logs:
    print(l.get('timestamp','')[:19], l.get('phase','').ljust(12), l.get('event','').ljust(12), l.get('detail','') or '')
" 2>/dev/null || echo "(API unavailable)"

  # Also scan log file if set
  if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    echo ""
    echo "--- From log file ---"
    grep -E "\[Engine\] Cycle #[0-9]+ (START|END)" "$LOG_FILE" | tail -"$n"
  fi
}

# ── Cycle Timing (log file) ──────────────────────────────────
# Parse cycle START/END pairs and show duration
qae-cycle-timing() {
  _needs_log || return 1
  echo "=== Cycle Timing ==="
  grep -E "\[Engine\] Cycle #[0-9]+ (START|END)" "$LOG_FILE" \
    | awk '
      /START/ { match($0, /Cycle #([0-9]+)/, a); n=a[1]; match($0, /T\+([0-9.]+)s/, b); start[n]=b[1] }
      /END/   { match($0, /Cycle #([0-9]+)/, a); n=a[1]; match($0, /T\+([0-9.]+)s/, b);
                if (start[n]) printf "Cycle %4d  start=%7.1fs  end=%7.1fs  duration=%5.1fs\n", n, start[n]+0, b[1]+0, b[1]-start[n]
              }
    ' | tail -20
}

# ── Phase Activity ───────────────────────────────────────────
# Show recent phase events from DB
qae-phases() {
  local n="${1:-30}"
  echo "=== Recent Phase Events (last $n) ==="
  _qae_api "/api/research-logs?limit=$n" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
for l in logs:
    ts  = l.get('timestamp','')[:19]
    ph  = (l.get('phase') or '').ljust(20)
    ev  = (l.get('event') or '').ljust(28)
    det = (l.get('detail') or '')[:80]
    print(ts, ph, ev, det)
" 2>/dev/null || echo "(API unavailable)"
}

# ── Filter by Phase ──────────────────────────────────────────
# qae-phase "inverse-optimizer" 20
qae-phase() {
  local phase="${1:-engine}" n="${2:-20}"
  echo "=== Phase: $phase (last $n) ==="
  _qae_api "/api/research-logs?limit=200" \
    | python3 -c "
import sys, json
phase = sys.argv[1]; n = int(sys.argv[2])
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
hits = [l for l in logs if (l.get('phase') or '').lower().startswith(phase.lower())][-n:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('event') or '').ljust(30), (l.get('detail') or '')[:90])
" "$phase" "$n" 2>/dev/null
}

# ── Deliberation Results ─────────────────────────────────────
# Show accept/reject verdicts from the deliberative evaluator
qae-deliberations() {
  local n="${1:-20}"
  echo "=== Deliberation Verdicts (last $n) ==="
  _needs_log && grep -E "verdict=(accept|reject|borderline)" "$LOG_FILE" | tail -"$n" \
    | sed 's/.*\(verdict=[a-z]*\)/\1/' | awk '{print NR, $0}'
  echo ""
  echo "--- From DB logs (Deliberative Evaluator) ---"
  _qae_api "/api/research-logs?limit=200" \
    | python3 -c "
import sys, json
n = int(sys.argv[1])
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
hits = [l for l in logs if 'deliberat' in (l.get('phase') or '').lower()
        or 'verdict' in (l.get('event') or '').lower()
        or 'deliberat' in (l.get('dataSource') or '').lower()][-n:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('event') or '').ljust(25), (l.get('detail') or '')[:90])
" "$n" 2>/dev/null
}

# ── Inverse Design / Campaign Loop ───────────────────────────
# Show Phase 7 inverse design activity
qae-inverse() {
  local n="${1:-30}"
  echo "=== Inverse Design / Campaign Activity (last $n) ==="
  _qae_api "/api/research-logs?limit=300" \
    | python3 -c "
import sys, json
n = int(sys.argv[1])
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
keywords = ['inverse', 'campaign', 'phase-7', 'optimizer']
hits = [l for l in logs if any(k in (l.get('phase') or '').lower()
        or k in (l.get('event') or '').lower()
        or k in (l.get('dataSource') or '').lower() for k in keywords)][-n:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('phase') or '').ljust(18), (l.get('event') or '').ljust(28), (l.get('detail') or '')[:70])
" "$n" 2>/dev/null

  if _needs_log 2>/dev/null; then
    echo ""
    echo "--- Campaign lines from log file ---"
    grep -iE "(campaign|inverse|phase.?7)" "$LOG_FILE" | grep -v "^$" | tail -"$n"
  fi
}

# ── SC Candidates ─────────────────────────────────────────────
# Show superconductor candidates and their Tc predictions
qae-candidates() {
  local n="${1:-20}"
  echo "=== Top SC Candidates (last $n discovered) ==="
  _qae_api "/api/superconductor-candidates?limit=$n" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
cands = d.get('candidates', d) if isinstance(d, dict) else d
if not isinstance(cands, list): print('unexpected response'); sys.exit()
for c in cands[:$n]:
    formula = (c.get('formula') or c.get('name') or '?').ljust(20)
    tc      = str(c.get('predictedTc') or c.get('tc') or '?').ljust(8)
    score   = str(c.get('score') or c.get('mlScore') or '?').ljust(8)
    status  = c.get('status') or c.get('phase') or ''
    print(formula, 'Tc='+tc, 'score='+score, status)
" 2>/dev/null || echo "(API unavailable)"
}

# ── DFT Queue ────────────────────────────────────────────────
# Show DFT job queue status
qae-dft() {
  echo "=== DFT Queue ==="
  _qae_api "/api/dft-status" | python3 -m json.tool 2>/dev/null \
    || _qae_api "/api/dft-status"
  echo ""
  echo "--- Recent DFT log events ---"
  _qae_api "/api/research-logs?limit=200" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
hits = [l for l in logs if 'dft' in (l.get('phase') or '').lower()
        or 'dft' in (l.get('dataSource') or '').lower()
        or 'qe' in (l.get('phase') or '').lower()][-15:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('event') or '').ljust(30), (l.get('detail') or '')[:80])
" 2>/dev/null
}

# ── Errors & Failures ────────────────────────────────────────
# Show error/failure log lines
qae-errors() {
  local n="${1:-30}"
  echo "=== Errors & Failures ==="
  if _needs_log 2>/dev/null; then
    grep -iE "(error|fail|crash|exception|unhandled|ECONNREFUSED|timeout)" "$LOG_FILE" \
      | grep -viE "(no error|0 errors|without error)" \
      | tail -"$n"
  fi
  echo ""
  echo "--- Error events from DB ---"
  _qae_api "/api/research-logs?limit=300" \
    | python3 -c "
import sys, json
n = int(sys.argv[1])
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
hits = [l for l in logs if any(k in (l.get('event') or '').lower()
        or k in (l.get('detail') or '').lower() for k in ['error','fail','crash','timeout'])][-n:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('phase') or '').ljust(18), (l.get('event') or '').ljust(25), (l.get('detail') or '')[:80])
" "$n" 2>/dev/null
}

# ── ML / Prediction Activity ─────────────────────────────────
qae-ml() {
  local n="${1:-20}"
  echo "=== ML Prediction Activity (last $n) ==="
  _qae_api "/api/research-logs?limit=300" \
    | python3 -c "
import sys, json
n = int(sys.argv[1])
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
keywords = ['ml', 'prediction', 'model', 'calibrat', 'xgb', 'gnn', 'feature']
hits = [l for l in logs if any(k in (l.get('phase') or '').lower()
        or k in (l.get('event') or '').lower()
        or k in (l.get('dataSource') or '').lower() for k in keywords)][-n:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('phase') or '').ljust(18), (l.get('event') or '').ljust(28), (l.get('detail') or '')[:70])
" "$n" 2>/dev/null
  echo ""
  echo "--- ML Calibration ---"
  _qae_api "/api/ml-calibration" | python3 -m json.tool 2>/dev/null \
    || _qae_api "/api/ml-calibration"
}

# ── Synthesis Discovery ──────────────────────────────────────
qae-synthesis() {
  local n="${1:-20}"
  echo "=== Synthesis Discovery (last $n) ==="
  _qae_api "/api/research-logs?limit=300" \
    | python3 -c "
import sys, json
n = int(sys.argv[1])
d = json.load(sys.stdin)
logs = d.get('logs', d) if isinstance(d, dict) else d
hits = [l for l in logs if 'synthesis' in (l.get('phase') or '').lower()
        or 'synthesis' in (l.get('dataSource') or '').lower()][-n:]
for l in hits:
    print(l.get('timestamp','')[:19], (l.get('event') or '').ljust(28), (l.get('detail') or '')[:80])
" "$n" 2>/dev/null
}

# ── Live Tail (log file) ─────────────────────────────────────
# Live-tail the log file filtering by keyword
# Usage: qae-tail [keyword]   e.g. qae-tail "Cycle #"
qae-tail() {
  _needs_log || return 1
  local filter="${1:-}"
  if [[ -n "$filter" ]]; then
    echo "Tailing $LOG_FILE | grep '$filter'  (Ctrl+C to stop)"
    tail -f "$LOG_FILE" | grep --line-buffered -i "$filter"
  else
    echo "Tailing $LOG_FILE  (Ctrl+C to stop)"
    tail -f "$LOG_FILE"
  fi
}

# ── Full Health Summary ──────────────────────────────────────
# One-shot overview of everything
qae-health() {
  echo "╔══════════════════════════════════════════════════╗"
  echo "║           QAE Health Check                       ║"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""
  qae-status
  echo ""
  echo "--- Recent Cycles ---"
  qae-cycles 6
  echo ""
  echo "--- Recent Errors ---"
  qae-errors 10
  echo ""
  echo "--- DFT Queue ---"
  _qae_api "/api/dft-status" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('  queued:', d.get('queued', '?'), '  running:', d.get('running', '?'), '  completed:', d.get('completed', '?'))
" 2>/dev/null || echo "  (unavailable)"
  echo ""
  echo "--- Top Candidates ---"
  qae-candidates 5
}

# ── Help ─────────────────────────────────────────────────────
qae-help() {
  cat <<'EOF'
QAE Diagnostic Commands
─────────────────────────────────────────────────────────
qae-health              Full health snapshot (status + cycles + errors + DFT + candidates)
qae-status              Engine state, tempo, active tasks
qae-cycles [n]          Last n cycle start/end events
qae-cycle-timing        Parse log file for cycle durations (needs LOG_FILE)
qae-phases [n]          Last n phase events from DB
qae-phase <name> [n]    Filter DB logs by phase name (e.g. "inverse-optimizer")
qae-deliberations [n]   Recent accept/reject verdicts
qae-inverse [n]         Phase 7 inverse design / campaign activity
qae-candidates [n]      Top SC candidates with Tc predictions
qae-dft                 DFT queue status + recent DFT logs
qae-ml [n]              ML prediction activity + calibration
qae-synthesis [n]       Synthesis discovery events
qae-errors [n]          Error and failure lines
qae-tail [keyword]      Live-tail log file (needs LOG_FILE), optional grep filter

Log file commands need:  export LOG_FILE=/tmp/server-yourlog.log
Server host (default localhost:5000):  export QAE_HOST=http://localhost:5000
EOF
}

echo "QAE diagnostics loaded. Run 'qae-help' for all commands."
