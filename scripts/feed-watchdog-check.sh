#!/usr/bin/env bash
# ============================================================
#  QAE Feed Watchdog — Fast Check (runs every 2 min)
#  Outputs a JSON status block so the feed-watchdog skill can
#  read it without needing a separate Python analysis step.
#
#  Exit codes:  0 = feed healthy
#               1 = feed frozen/slowing (needs intervention)
#               2 = server down
# ============================================================
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/server-latest.log}"
HOST="http://localhost:4000"
REPORT_FILE="$LOG_DIR/feed-watchdog-report.json"
if command -v cygpath &>/dev/null; then
  REPORT_FILE_NATIVE=$(cygpath -w "$REPORT_FILE")
  _TMPDIR_NATIVE=""  # set after tmpdir creation
else
  REPORT_FILE_NATIVE="$REPORT_FILE"
fi
PYTHON_CMD="python3"
if ! python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
  if python -c "import sys; sys.exit(0)" 2>/dev/null; then
    PYTHON_CMD="python"
  fi
fi

# ── Temp dir (needed before any curl output) ─────────────────
_TMPDIR_WD=$(mktemp -d 2>/dev/null || echo "/tmp/qae-wd-$$")
mkdir -p "$_TMPDIR_WD"
if command -v cygpath &>/dev/null; then
  _TMPDIR_WD_NATIVE=$(cygpath -w "$_TMPDIR_WD")
  REPORT_FILE_NATIVE=$(cygpath -w "$REPORT_FILE")
else
  _TMPDIR_WD_NATIVE="$_TMPDIR_WD"
  REPORT_FILE_NATIVE="$REPORT_FILE"
fi

# ── Quick HTTP check ─────────────────────────────────────────
_http_code=$(curl -4 -o /dev/null -sf -w "%{http_code}" --max-time 8 "$HOST/api/health" 2>/dev/null || echo "000")
_server_up=false
[[ "$_http_code" == "200" ]] && _server_up=true

# ── Fetch last 50 research log entries (errors + timestamp) ──
echo "null" > "$_TMPDIR_WD/logs.json"
if [[ "$_server_up" == "true" ]]; then
  curl -4 -sf --max-time 8 "$HOST/api/research-logs?limit=50" 2>/dev/null > "$_TMPDIR_WD/logs.json" || echo "null" > "$_TMPDIR_WD/logs.json"
fi
if [[ "$_server_up" == "true" ]]; then
  _since=$(( $(date +%s%3N 2>/dev/null || $PYTHON_CMD -c "import time;print(int(time.time()*1000))") - 180000 ))
  curl -4 -sf --max-time 8 "$HOST/api/network-metrics?limit=100&since=${_since}" 2>/dev/null > "$_TMPDIR_WD/net.json" || echo "null" > "$_TMPDIR_WD/net.json"
  _net_max_ms=$($PYTHON_CMD -c "
import json
try:
    with open('$_TMPDIR_WD/net.json') as f: d = json.load(f)
    entries = d.get('entries', []) if isinstance(d, dict) else []
    if entries:
        worst = max(entries, key=lambda e: e.get('durationMs',0))
        print(worst.get('durationMs',0))
    else: print(0)
except: print(0)
" 2>/dev/null || echo 0)
  _net_max_ep=$($PYTHON_CMD -c "
import json
try:
    with open('$_TMPDIR_WD/net.json') as f: d = json.load(f)
    entries = d.get('entries', []) if isinstance(d, dict) else []
    if entries:
        worst = max(entries, key=lambda e: e.get('durationMs',0))
        print(worst.get('endpoint',''))
    else: print('')
except: print('')
" 2>/dev/null || echo "")
fi

# ── Check log for GB training activity ───────────────────────
_gb_in_log=false
if [[ -f "$LOG_FILE" ]]; then
  tail -200 "$LOG_FILE" 2>/dev/null | grep -qE "\[GradientBoost\].*(training|retrain|cool.down|trainEnsemble|trainVariance)" && _gb_in_log=true || true
fi


# ── Python analysis ──────────────────────────────────────────
PYTHONIOENCODING=utf-8 $PYTHON_CMD - <<PYEOF
import json, sys, os, re
from datetime import datetime, timezone

tmpdir      = r"$_TMPDIR_WD_NATIVE"
report_file = r"$REPORT_FILE_NATIVE"
gb_in_log   = "$_gb_in_log" == "true"
server_up   = "$_server_up" == "true"
now         = datetime.now(timezone.utc)

def rf(name):
    try:
        with open(os.path.join(tmpdir, name), encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except: return ""

def load_json(name):
    try:
        with open(os.path.join(tmpdir, name), encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except: return None

# ── Load research logs ────────────────────────────────────────
logs_raw = load_json("logs.json")
research_logs = []
if isinstance(logs_raw, dict):
    research_logs = logs_raw.get("logs", [])
elif isinstance(logs_raw, list):
    research_logs = logs_raw

# ── Last log timestamp and event ─────────────────────────────
last_ts = None
last_event = ""
for _l in research_logs[:3]:
    _raw = (_l.get("timestamp") or "")
    if _raw:
        try:
            _t = datetime.fromisoformat(_raw.replace("Z", "+00:00"))
            if _t.tzinfo is None: _t = _t.replace(tzinfo=timezone.utc)
            last_ts = _t
            last_event = (_l.get("event") or "")[:80]
            break
        except: pass

feed_age_s = (now - last_ts).total_seconds() if last_ts else None

# ── Load network metrics ──────────────────────────────────────
net_raw = load_json("net.json")
net_entries = (net_raw or {}).get("entries", []) if isinstance(net_raw, dict) else []
net_max_ms = 0
net_max_ep = ""
if net_entries:
    _w = max(net_entries, key=lambda e: e.get("durationMs", 0))
    net_max_ms = _w.get("durationMs", 0)
    net_max_ep = _w.get("endpoint", "")

# ── Error detection in research logs ─────────────────────────
# Patterns that mean "something is broken and needs a code fix"
ERROR_EVENT_PATTERNS = [
    re.compile(r'\berror\b',         re.IGNORECASE),
    re.compile(r'\bfailed\b',        re.IGNORECASE),
    re.compile(r'\bcrash\b',         re.IGNORECASE),
    re.compile(r'\bexception\b',     re.IGNORECASE),
    re.compile(r'timed? out',        re.IGNORECASE),
    re.compile(r'connection term',   re.IGNORECASE),
    re.compile(r'cannot read prop',  re.IGNORECASE),
    re.compile(r'is not a function', re.IGNORECASE),
    re.compile(r'undefined.*null|null.*undefined', re.IGNORECASE),
    re.compile(r'typeerror|referenceerror|syntaxerror', re.IGNORECASE),
]
# Events that look like errors but are normal/expected — skip them
ERROR_IGNORE_PATTERNS = [
    re.compile(r'duplicate skipped',              re.IGNORECASE),
    re.compile(r'rejected.*validation',           re.IGNORECASE),
    re.compile(r'skipped.*circuit',               re.IGNORECASE),
    re.compile(r'material rejected',              re.IGNORECASE),
    re.compile(r'phonon.*unstable|unstable.*phonon', re.IGNORECASE),
    re.compile(r'candidate.*rejected',            re.IGNORECASE),
    re.compile(r'0 passed.*filter|filter.*0 pass',re.IGNORECASE),
    re.compile(r'GCP dispatch failed',            re.IGNORECASE),  # fire-and-forget, non-fatal
    re.compile(r'discovery efficiency',           re.IGNORECASE),  # stats log; "crash=N" is a counter not an error
    re.compile(r'(synthesis|reaction) discovery error.*request timed out', re.IGNORECASE),  # OpenAI first-call timeout; circuit breaker handles backoff
    re.compile(r'known materials import error.*request timed out', re.IGNORECASE),  # Materials Project API transient timeout; self-recovers on next cycle
    re.compile(r'feedback loop calibration',      re.IGNORECASE),  # stats log; "mean abs error" in detail is a metric not an error
    re.compile(r'structure learning loop cycle',  re.IGNORECASE),  # stats log; "failed=N" is a counter not an error
    re.compile(r'sg sweep progress',             re.IGNORECASE),  # SG sweep stats; "valence-filter-failed" is a rejection category name, not an error
    re.compile(r'sg-sweep-guard',               re.IGNORECASE),  # sweep guard throw leaking into Cycle error catch — not a real error
    re.compile(r'signal rejected',              re.IGNORECASE),  # signal-scanner AI rejection of non-matching materials — expected behavior, "failed AI verification" is normal
    re.compile(r'GB score backfill complete',   re.IGNORECASE),  # stats log; "N failed (set to default)" is a count of unscoreable candidates, not an error
    re.compile(r'OQMD fetch failed.*HTTP 5\d\d', re.IGNORECASE),  # transient OQMD API server error (502/503/504); self-recovers on next cycle
    re.compile(r'model improvement.*executing experiment', re.IGNORECASE),  # RL agent hyperparameter experiment log; "average error" in detail is MAE metric not a runtime error
]

detected_errors = []  # list of {event, detail, phase, timestamp}
_error_event_set = set()  # deduplicate by event+detail combo

for _l in research_logs[:50]:
    _ev  = (_l.get("event")  or "")
    _det = (_l.get("detail") or "")
    _ph  = (_l.get("phase")  or "")
    _ts  = (_l.get("timestamp") or "")
    _combined = f"{_ev} {_det}"

    # Skip if matches ignore patterns
    if any(p.search(_combined) for p in ERROR_IGNORE_PATTERNS):
        continue

    # Check if matches error patterns
    if any(p.search(_ev) or p.search(_det) for p in ERROR_EVENT_PATTERNS):
        _dedup_key = f"{_ev[:60]}|{_det[:60]}"
        if _dedup_key not in _error_event_set:
            _error_event_set.add(_dedup_key)
            detected_errors.append({
                "event":     _ev,
                "detail":    _det[:300],
                "phase":     _ph,
                "timestamp": _ts,
            })

# ── Classify errors by type ───────────────────────────────────
def classify_error(ev, det):
    combined = f"{ev} {det}".lower()
    if "cannot read properties" in combined or "is not a function" in combined or "typeerror" in combined:
        return "null_reference", "free_investigation_null_guard"
    if "referenceerror" in combined:
        return "reference_error", "free_investigation_code_bug"
    if "timed out" in combined or "timeout" in combined:
        return "timeout", "free_investigation_timeout"
    if "connection term" in combined or "connection timeout" in combined or "pool" in combined:
        return "db_connection", "free_investigation_db"
    if "request timed out" in combined:
        return "openai_timeout", "free_investigation_openai_circuit"
    return "generic_error", "free_investigation_code_bug"

classified_errors = []
for e in detected_errors:
    err_type, err_action = classify_error(e["event"], e["detail"])
    classified_errors.append({**e, "errorType": err_type, "suggestedAction": err_action})

# ── Determine overall status ──────────────────────────────────
status = "healthy"
issue  = None
action = None

# Priority 1: Event loop blocked / slow endpoint
net_max_min = net_max_ms / 60000
if net_max_min >= 3.0:
    status = "event_loop_blocked"
    cause  = "local GB training" if gb_in_log else "CPU-intensive operation"
    issue  = (
        f"EVENT LOOP BLOCKED {net_max_min:.1f} min on '{net_max_ep}'. "
        f"Cause: {cause}. Server frozen — no feed entries written."
    )
    action = "free_investigation_gb_training" if gb_in_log else "free_investigation_event_loop"
elif net_max_min >= 2.0:
    status = "event_loop_blocked"
    cause  = "local GB training" if gb_in_log else "blocking operation"
    issue  = (
        f"SLOW ENDPOINT {net_max_min:.1f} min on '{net_max_ep}'. "
        f"Cause: {cause}. Event loop severely degraded."
    )
    action = "free_investigation_gb_training" if gb_in_log else "free_investigation_slow_endpoint"
elif net_max_min >= 1.0:
    status = "event_loop_slow"
    issue  = f"Event loop slow: '{net_max_ep}' took {net_max_min:.1f} min."
    action = "monitor"

# Priority 2: Server down / feed frozen
if status in ("healthy", "event_loop_slow"):
    if not server_up:
        status = "server_down"
        issue  = "Server not responding."
        action = "restart_server"
    elif feed_age_s is None:
        status = "feed_empty"
        issue  = "No research log entries found."
        action = "start_engine"
    elif feed_age_s > 1200:
        status = "feed_frozen"
        cause  = "GB training" if gb_in_log else "unknown"
        issue  = f"Feed FROZEN {feed_age_s/60:.0f} min. Last: '{last_event}'. Cause: {cause}."
        action = "free_investigation_gb_training" if gb_in_log else "restart_engine_then_investigate"
    elif feed_age_s > 600:
        status = "feed_frozen"
        issue  = f"Feed FROZEN {feed_age_s/60:.0f} min. Last: '{last_event}'."
        action = "restart_engine_then_investigate"
    elif feed_age_s > 300:
        status = "feed_slow"
        issue  = f"Feed quiet {feed_age_s/60:.1f} min."
        action = "monitor"

# Priority 3: Errors in feed (even if feed is flowing)
# If there are recurring errors, flag them regardless of status
if classified_errors and status == "healthy":
    status = "feed_errors"
    first  = classified_errors[0]
    issue  = (
        f"{len(classified_errors)} error(s) in last 50 feed entries. "
        f"Most recent: [{first['phase']}] '{first['event']}' — {first['detail'][:150]}"
    )
    action = first["suggestedAction"]
elif classified_errors and status not in ("event_loop_blocked", "server_down", "feed_frozen"):
    # Errors alongside a slow/quiet feed — still needs investigation
    first  = classified_errors[0]
    issue  = (issue or "") + (
        f" ALSO: {len(classified_errors)} error(s) in feed. "
        f"Most recent: '{first['event']}' — {first['detail'][:120]}"
    )
    if action == "monitor":
        action = first["suggestedAction"]

report = {
    "timestamp":       now.isoformat(),
    "status":          status,
    "issue":           issue,
    "action":          action,
    "feedAgeSeconds":  feed_age_s,
    "lastLogEvent":    last_event,
    "netMaxMs":        net_max_ms,
    "netMaxEndpoint":  net_max_ep,
    "gbInLog":         gb_in_log,
    "serverUp":        server_up,
    "detectedErrors":  classified_errors[:10],
    "errorCount":      len(classified_errors),
}

with open(report_file, "w") as f:
    json.dump(report, f, indent=2, default=str)

# Print summary
print(f"[watchdog] status={status} feed_age={int(feed_age_s or 0)}s errors={len(classified_errors)} net_max={net_max_ms}ms")
if classified_errors:
    for e in classified_errors[:3]:
        print(f"  [!] [{e['phase']}] {e['event']}: {e['detail'][:100]}")

if status in ("event_loop_blocked", "feed_frozen", "server_down", "feed_empty", "feed_errors"):
    sys.exit(1)
sys.exit(0)
PYEOF

rm -rf "$_TMPDIR_WD" 2>/dev/null || true
