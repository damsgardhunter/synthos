#!/usr/bin/env bash
# ============================================================
#  QAE Monitor Health Check
#  Usage: bash scripts/qae-monitor-check.sh
#

# Resolve python3 on Windows where 'python3' may be a Store alias
PYTHON_CMD="python3"
if ! python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
  if python -c "import sys; sys.exit(0)" 2>/dev/null; then
    PYTHON_CMD="python"
  fi
fi

#
#  Queries the API + tails the server log, then writes a
#  structured JSON report to logs/health-report.json.
#  The qae-monitor slash command reads this report each cycle.
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Read PORT from .env if present, default to 5000
_ENV_PORT=$(grep -E "^PORT=" "$PROJECT_ROOT/.env" 2>/dev/null | head -1 | cut -d= -f2 | tr -d '[:space:]')
HOST="${QAE_HOST:-http://localhost:${_ENV_PORT:-5000}}"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/server-latest.log}"
REPORT_FILE="$LOG_DIR/health-report.json"

mkdir -p "$LOG_DIR"

# Convert POSIX paths to native paths for Python on Windows
if command -v cygpath &>/dev/null; then
  REPORT_FILE_NATIVE=$(cygpath -w "$REPORT_FILE")
else
  REPORT_FILE_NATIVE="$REPORT_FILE"
fi

# ── Fetch with timeout and graceful failure ───────────────────
# 4s timeout (down from 8s) — if the server is responsive it answers in <1s;
# the extra headroom is for slow DB-backed routes under moderate load.
api_get() {
  curl -sf --max-time 4 "$HOST$1" 2>/dev/null || echo "null"
}

# ── Server reachability ───────────────────────────────────────
# Use a 5s health-check to detect unresponsive servers quickly.
# If HTTP times out, retry once after 2s — a single transient spike on the
# event loop (e.g. pareto-frontier computing) shouldn't cause a cascade of
# false-positive "HTTP blocked" warnings. Only declare HTTP_BLOCKED if BOTH
# probes fail.
# If both probes time out but the PID is alive → event loop is genuinely
# saturated (SG sweep / GB training). Set HTTP_BLOCKED=True and skip ALL
# subsequent curl calls so a blocked server cannot make the monitor hang.
SERVER_RUNNING=False
HTTP_BLOCKED=False
PID_FILE="logs/server.pid"
if curl -sf --max-time 5 "$HOST/api/health" > /dev/null 2>&1; then
  SERVER_RUNNING=True
else
  sleep 2
  if curl -sf --max-time 5 "$HOST/api/health" > /dev/null 2>&1; then
    SERVER_RUNNING=True   # second probe succeeded — first was a transient spike
  elif [[ -f "$PID_FILE" ]] && tasklist //FI "PID eq $(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null | grep -q "$(cat "$PID_FILE" 2>/dev/null)"; then
    SERVER_RUNNING=True   # HTTP blocked (event loop saturated), process alive — not crashed
    HTTP_BLOCKED=True     # skip all further curl calls to avoid 100s hang
  fi
fi

# ── Fetch API data (skipped when HTTP_BLOCKED to prevent 100s hang) ──────────
SERVER_START_TIME=$(cat "$LOG_DIR/server-start-time.txt" 2>/dev/null || echo "")
if [[ "$HTTP_BLOCKED" == "True" ]]; then
  ENGINE_STATUS="null"
  DFT_STATUS="null"
  RESEARCH_LOGS="null"
  ML_CALIBRATION="null"
  DASHBOARD="null"
  CLIENT_ERRORS="null"
  CANDIDATES="null"
else
  ENGINE_STATUS=$(api_get "/api/engine/status")
  DFT_STATUS=$(api_get "/api/dft-status")
  # Double-fetch research-logs to defeat stale cache: cache.getOrSetStale returns
  # stale data immediately and triggers a background refresh. If no client has
  # queried this key for >TTL, the first call returns a stale entry from minutes
  # or hours ago. Sleep briefly so the background refresh completes, then re-fetch
  # to get the fresh data. Without this, the freeze detector reports 20-min false
  # positives any time the cache hasn't been warmed by browser traffic.
  api_get "/api/research-logs?limit=100" > /dev/null 2>&1
  sleep 1
  RESEARCH_LOGS=$(api_get "/api/research-logs?limit=100")
  ML_CALIBRATION=$(api_get "/api/ml-calibration")
  DASHBOARD=$(api_get "/api/dashboard")
  CLIENT_ERRORS=$(api_get "/api/client-errors?limit=50")
  CANDIDATES=$(api_get "/api/superconductor-candidates?limit=20&sort=newest")
  # Fetch last 2 minutes of network metrics (since=now-120s) for the table display
  _NET_SINCE=$(( $(date +%s%3N 2>/dev/null || python3 -c "import time;print(int(time.time()*1000))") - 120000 ))
  NETWORK_METRICS=$(api_get "/api/network-metrics?limit=200&since=${_NET_SINCE}")
fi

# ── Page endpoint probes (skipped when HTTP_BLOCKED) ─────────
# Each probe records: HTTP status + response time in ms
probe_endpoint() {
  local path="$1"
  local start_ms end_ms duration_ms http_code
  start_ms=$(date +%s%3N 2>/dev/null || python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || echo 0)
  http_code=$(curl -o /dev/null -sf -w "%{http_code}" --max-time 4 "$HOST$path" 2>/dev/null || echo "000")
  end_ms=$(date +%s%3N 2>/dev/null || python3 -c "import time; print(int(time.time()*1000))" 2>/dev/null || echo 0)
  duration_ms=$((end_ms - start_ms))
  echo "${http_code}:${duration_ms}"
}

if [[ "$HTTP_BLOCKED" == "True" ]]; then
  PROBE_DASHBOARD="000:0"
  PROBE_PIPELINE="000:0"
  PROBE_PHYSICS="000:0"
  PROBE_LAB="000:0"
  PROBE_MATERIALS="000:0"
  PROBE_DISCOVERY="000:0"
  PROBE_DFT="000:0"
  PROBE_ELEMENTS="000:0"
  PROBE_PARETO="000:0"
else
  PROBE_DASHBOARD=$(probe_endpoint "/api/dashboard")
  PROBE_PIPELINE=$(probe_endpoint "/api/research-logs?limit=10")
  PROBE_PHYSICS=$(probe_endpoint "/api/pipeline-stats")
  PROBE_LAB=$(probe_endpoint "/api/superconductor-candidates?limit=5")
  PROBE_MATERIALS=$(probe_endpoint "/api/materials?limit=5")
  PROBE_DISCOVERY=$(probe_endpoint "/api/novel-predictions?limit=5")
  PROBE_DFT=$(probe_endpoint "/api/dft-status")
  PROBE_ELEMENTS=$(probe_endpoint "/api/elements")
  PROBE_PARETO=$(probe_endpoint "/api/pareto-frontier?limit=50")
fi

# ── Write API data and log tail to temp files (avoids heredoc injection bugs) ──
TMPDIR_QAE=$(mktemp -d 2>/dev/null || echo "/tmp/qae-monitor-$$")
mkdir -p "$TMPDIR_QAE"
echo "$ENGINE_STATUS"   > "$TMPDIR_QAE/engine_status.json"
echo "$DFT_STATUS"      > "$TMPDIR_QAE/dft_status.json"
echo "$RESEARCH_LOGS"   > "$TMPDIR_QAE/research_logs.json"
echo "$ML_CALIBRATION"  > "$TMPDIR_QAE/ml_calibration.json"
echo "$DASHBOARD"       > "$TMPDIR_QAE/dashboard.json"
echo "$CLIENT_ERRORS"      > "$TMPDIR_QAE/client_errors.json"
echo "$CANDIDATES"         > "$TMPDIR_QAE/candidates.json"
echo "${NETWORK_METRICS:-null}" > "$TMPDIR_QAE/network_metrics.json"
echo "$SERVER_START_TIME" > "$TMPDIR_QAE/server_start_time.txt"

# Write probe results as simple key=value
cat > "$TMPDIR_QAE/page_probes.txt" <<PROBES
dashboard=$PROBE_DASHBOARD
research_pipeline=$PROBE_PIPELINE
computational_physics=$PROBE_PHYSICS
superconductor_lab=$PROBE_LAB
materials_database=$PROBE_MATERIALS
novel_discovery=$PROBE_DISCOVERY
dft_queue=$PROBE_DFT
atomic_explorer=$PROBE_ELEMENTS
pareto_frontier=$PROBE_PARETO
PROBES
if [[ -f "$LOG_FILE" ]]; then
  tail -200 "$LOG_FILE" 2>/dev/null > "$TMPDIR_QAE/log_tail.txt" || touch "$TMPDIR_QAE/log_tail.txt"
else
  touch "$TMPDIR_QAE/log_tail.txt"
fi

# Convert TMPDIR path for Python on Windows
if command -v cygpath &>/dev/null; then
  TMPDIR_QAE_NATIVE=$(cygpath -w "$TMPDIR_QAE")
else
  TMPDIR_QAE_NATIVE="$TMPDIR_QAE"
fi

# ── Python analysis ───────────────────────────────────────────
PYTHONIOENCODING=utf-8 $PYTHON_CMD - <<PYEOF
import json, re, sys, os
from datetime import datetime, timezone
# Ensure stdout can handle Unicode on Windows (cp1252 → utf-8)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

server_running = $SERVER_RUNNING
http_blocked = $HTTP_BLOCKED   # True when HTTP timed out but PID alive (training block)
report_file = r"$REPORT_FILE_NATIVE"
tmpdir = r"$TMPDIR_QAE_NATIVE"

# ── Load data from temp files ─────────────────────────────────
def read_file(name):
    try:
        with open(os.path.join(tmpdir, name), encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return ""

log_tail = read_file("log_tail.txt")

# ── Load API responses ────────────────────────────────────────
def safe_json(s):
    try:
        return json.loads(s) if s and s != "null" else None
    except Exception:
        return None

engine_status   = safe_json(read_file("engine_status.json"))
dft_status      = safe_json(read_file("dft_status.json"))
research_logs_r = safe_json(read_file("research_logs.json"))
ml_calibration  = safe_json(read_file("ml_calibration.json"))
dashboard       = safe_json(read_file("dashboard.json"))
_net_raw        = safe_json(read_file("network_metrics.json"))
net_metrics     = (_net_raw or {}).get("entries", []) if isinstance(_net_raw, dict) else []

research_logs = []
if isinstance(research_logs_r, dict):
    research_logs = research_logs_r.get("logs", [])
elif isinstance(research_logs_r, list):
    research_logs = research_logs_r

# Parse server start time early — used in frozen-feed check to suppress false
# positives right after a restart (engine needs time to write first DB entry).
server_start_ts = None
_ss_raw_early = read_file("server_start_time.txt").strip()
if _ss_raw_early:
    try:
        server_start_ts = datetime.fromisoformat(_ss_raw_early.replace("Z", "+00:00"))
        if server_start_ts.tzinfo is None:
            server_start_ts = server_start_ts.replace(tzinfo=timezone.utc)
    except Exception:
        server_start_ts = None

issues   = []
warnings = []

def add_issue(category, message, source, severity="error"):
    issues.append({"category": category, "message": message, "source": source, "severity": severity})

def add_warning(category, message, source):
    warnings.append({"category": category, "message": message, "source": source})

# ═══════════════════════════════════════════════════════════════
# 1. SERVER / INFRASTRUCTURE CHECKS
# ═══════════════════════════════════════════════════════════════
if not server_running:
    add_issue("infrastructure", "Server is not responding", "healthcheck")
elif http_blocked:
    # PID alive but HTTP unresponsive — event loop is saturated.
    # Count [research-logs] ERROR lines in log to distinguish SG sweep DB storm
    # from GB training (which blocks for 30s then recovers without DB errors).
    _blocked_reslog_errors = sum(1 for _l in (log_tail or "").split("\n") if re.search(r"\[research-logs\]\s+ERROR:", _l))
    if _blocked_reslog_errors > 8:
        add_issue("infrastructure",
            f"Server HTTP BLOCKED and DB writes failing ({_blocked_reslog_errors} [research-logs] ERROR lines) — "
            f"event loop saturated, likely SG sweep with DB contention. "
            f"Fix: POST /api/engine/stop then /start to interrupt sweep. If errors persist, restart server.",
            "healthcheck")
    else:
        add_warning("infrastructure",
            "Server HTTP blocked (event loop busy — GradientBoost training or brief SG sweep yield gap). "
            "Process is alive; should recover within 30s.",
            "healthcheck")

# ═══════════════════════════════════════════════════════════════
# 2. LOG FILE — DB / CONNECTION ERRORS
# ═══════════════════════════════════════════════════════════════
if log_tail:
    lines = log_tail.split("\n")

    db_error_patterns = [
        (r"ETIMEDOUT",                    "Database connection timed out (ETIMEDOUT)"),
        (r"connection terminated",        "DB connection terminated unexpectedly"),
        (r"ECONNRESET",                   "DB connection reset (ECONNRESET)"),
        (r"socket hang up",               "DB socket hang up"),
        (r"pool.*exhaust|max clients",    "Connection pool exhausted / max clients reached"),
        (r"query timeout",                "Database query timed out"),
        (r"connection ended unexpectedly","DB connection ended unexpectedly"),
        (r"ECONNREFUSED.*5432",           "PostgreSQL port 5432 refused — DB may be down"),
    ]
    # These prefixes indicate the DB error was caught by a try/catch handler and the
    # server recovered gracefully — downgrade to warning rather than issue.
    db_caught_patterns = [
        r"Queue refill error:",       # DFT queue catch block — retries next cycle
        r"refill error:",             # generic queue catch
        r"\[cache\].*error:",         # cache layer catch
        r"\[\w[\w-]*\] ERROR:",       # route handler catch block (e.g. [research-logs] ERROR:, [dft-status] ERROR:)
        r"\[express\]",               # express HTTP access log — error was already logged above; this is just the response summary
        r"\[Engine\] bg ",            # engine background operations (topology/Fermi, etc.) catch errors gracefully
        r"\[Engine\] .*(?:failed|write failed|error):",  # engine catches DB write failures (research log, etc.) and continues
        r"updatePhaseStatus failed:",  # engine.ts:1110 catch block — phase status update failure, engine continues
        r"\[cause\]:",                  # Node.js error [cause] sub-entry in stack trace — the outer error was caught by Promise.allSettled or try/catch
        r"\[MP API\]",                 # Materials Project external API errors — outbound HTTP, not DB; has retry logic (attempt N)
        r"\[Storage\] Bulk insert chunk failed",  # storage.ts catch block — bulk insert catches per-chunk failures and continues; inserted count returned
        r"\[InsightDetector\] Bootstrap failed:",  # insight-detector.ts catch block — embedding cache bootstrap failure, engine continues without embeddings
        r"\[Pareto\] Scheduled recompute failed:",  # pareto-optimizer.ts scheduleParetoRecompute try/catch — DB failure during background recompute; next scheduled run will retry
        r"\[SC Research\] Novel SC generation error:",  # superconductor-research.ts catch block (line 718) — caught, logs error, function returns gracefully
        r"\[XGBoost\] Failed to persist surrogate metrics:",  # gradient-boost.ts:2504 catch block — surrogate metric persistence failure caught with console.warn, function returns gracefully
        r"\[tsc-candidates\] DB timeout with cold cache",  # routes.ts tsc-candidates handler — DB timeout caught, returns session-only data gracefully
        r"\[learning-phases\] DB timeout with cold cache",  # routes.ts learning-phases handler — DB timeout caught, returns empty state gracefully
        r"\[engine-memory\] DB timeout with cold cache",   # routes.ts engine-memory handler — DB timeout caught, returns empty state gracefully
        r"\[MatDB\] Failed to insert ",                   # data-fetcher.ts catch block (line 423) — insert failure caught with console.warn, loop continues to next material
        r"\[XGBoost\] GCP dispatch failed:",              # gradient-boost.ts:2165 .catch(console.warn) — fire-and-forget GCP dispatch; failure is non-fatal, GB training continues
        r"\[KB\] Failed to insert ",                      # data-fetcher.ts:632 catch block — KB insert failure caught with console.warn, loop continues to next material
        r"^\s+code:\s+'ECONNRESET'",                      # Node.js error object dump — indented `code: 'ECONNRESET'` means the error was caught and printed via console.error(e); the catch context is on the preceding line (e.g. "updatePhaseStatus failed:")
        r"\[process\] unhandledRejection:.*(?:Connection terminated|ETIMEDOUT|pool)",  # process-level unhandledRejection for DB connection errors — Node.js logged it but server did not crash; cascading effect of DB storm
        r"\[Engine\] Campaign restoration failed:",        # engine.ts startEngine catch block — DB error during startup campaign restore; engine continues without restored campaigns
        r"\[Engine\] Stats restore from DB failed:",       # engine.ts startEngine catch block — DB error during startup stats restore; engine continues with default stats
        r"\[3DSC\] Batch insert failed:",                  # threedsc-ingestion.ts:280 console.warn catch — individual batch failure; ingestion continues and completes
        r"\[ActiveLearning\] Failed to dispatch GNN job to GCP:",  # active-learning.ts:1556 console.warn catch — GNN dispatch failure; GNN inference already disabled, active learning continues
        r"\[OQMD\] Failed to insert ",                            # data-fetcher.ts:295 console.warn catch — OQMD insert failure; loop continues to next entry
        r"DB socket timeout",                                     # server/db.ts socket.on('timeout') handler — zombie connection destroyed by socket timeout fix; error propagates to caller's catch block
    ]
    for line in lines:
        for pattern, msg in db_error_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_caught = any(re.search(cp, line, re.IGNORECASE) for cp in db_caught_patterns)
                if is_caught:
                    # Caught/handled — emit warning, not issue
                    warn_msg = msg + " (caught — server recovered)"
                    if not any(w["message"] == warn_msg for w in warnings):
                        add_warning("database", warn_msg, f"log: {line.strip()[:120]}")
                else:
                    # Deduplicate unhandled DB errors
                    if not any(i["message"] == msg for i in issues):
                        add_issue("database", msg, f"log: {line.strip()[:120]}")

    # DB write error rate — if [research-logs] ERROR appears >8 times in the log
    # tail it means the circuit breaker is cycling open/closed: writes are failing
    # persistently, not just transiently. Escalate to issue so the monitor fixes it.
    _reslog_error_count = sum(
        1 for line in lines
        if re.search(r"\[research-logs\]\s+ERROR:", line)
    )
    if _reslog_error_count > 8:
        add_issue("database",
            f"Research log DB writes failing at high rate ({_reslog_error_count} errors in log tail) — "
            f"connection pool may have stale connections; activity feed will freeze within minutes. "
            f"Fix: restart engine (POST /api/engine/stop then /start). If errors persist, restart server.",
            "log: [research-logs] ERROR burst")

    # Feature extraction / ML timeouts
    feat_patterns = [
        (r"feature extraction took\s+(\d+)ms",  5000, "Feature extraction"),
        (r"xgboost.*?(\d+)ms",                  8000, "XGBoost prediction"),
        (r"gnn.*?(\d+)ms",                      8000, "GNN inference"),
        (r"ml.*?timeout",                          0, "ML model timeout"),
    ]
    _extract_skip_count = 0
    for line in lines:
        # Skip [express] HTTP access log lines — their JSON response bodies contain
        # URL paths and client-reported messages that can falsely match timing patterns
        # (e.g. /api/gnn/version-history in a client-errors response body).
        if re.search(r"\[express\]", line):
            continue
        for pattern, threshold, label in feat_patterns:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                if threshold == 0:
                    # extractFeatures timeout lines ending in "— skipping" are gracefully-handled
                    # circuit-breaker events (GradientBoost blocks event loop → 8s cap fires).
                    # Count them and emit one warning, not one issue per formula.
                    if re.search(r"extractFeatures timeout.*skipping", line, re.IGNORECASE):
                        _extract_skip_count += 1
                    else:
                        # Deduplicate genuine ML service timeouts
                        msg = f"{label} timed out"
                        if not any(i["message"] == msg for i in issues):
                            add_issue("ml", msg, f"log: {line.strip()[:120]}")
                else:
                    val = int(m.group(1))
                    if val > threshold:
                        add_issue("ml", f"{label} took {val}ms (threshold {threshold}ms)", f"log: {line.strip()[:120]}")
    if _extract_skip_count > 0:
        add_warning("ml", f"extractFeatures circuit-breaker fired {_extract_skip_count}x (event-loop congestion during GradientBoost training — materials skipped gracefully)", "log scan")

    # Cycle 1367 fix verification: extractFeatures wall-clock TOTAL > 10s.
    # Pre-fix this commonly hit 50-100s due to many concurrent compute bodies stacking
    # on the event loop. Post-fix (global FEATURE_COMPUTE_MAX_CONCURRENT=2 semaphore in
    # ml-predictor.ts), TOTAL should be 1-3s actual CPU. Anything still >10s means the
    # semaphore isn't bounding contention enough, OR a single sub-call is genuinely
    # slow — surface as a deduped warning so it's visible without spamming issues.
    _extract_total_max = 0
    _extract_total_count = 0
    _extract_total_formula_max = ""
    for line in lines:
        m = re.search(r"\[extractFeatures\]\s+TOTAL\s+(\d+)ms\s+for\s+(\S+)", line)
        if m:
            ms = int(m.group(1))
            if ms > 10000:
                _extract_total_count += 1
                if ms > _extract_total_max:
                    _extract_total_max = ms
                    _extract_total_formula_max = m.group(2)
    if _extract_total_count > 0:
        add_warning(
            "ml",
            f"extractFeatures wall-clock TOTAL >10s fired {_extract_total_count}x "
            f"(max {_extract_total_max}ms for {_extract_total_formula_max}) — "
            f"post-cycle-1367 fix in place (FEATURE_COMPUTE_MAX_CONCURRENT=2 semaphore); "
            f"if persistent, semaphore may need lowering to 1 or a sub-call regressed",
            "log scan",
        )

    # OpenAI API errors — treated as errors, not warnings, because they block
    # formula generation, NLP analysis, structure prediction, and synthesis tracking.
    openai_timeout_seen = False
    openai_ratelimit_seen = False
    for line in lines:
        if re.search(r"openai|gpt-4|gpt-3|text-embedding", line, re.IGNORECASE):
            if re.search(r"timeout|timed out|ETIMEDOUT|ECONNRESET|socket hang|504|ENOTFOUND", line, re.IGNORECASE):
                if not openai_timeout_seen:
                    openai_timeout_seen = True
                    add_issue("openai", "OpenAI API request timed out — queries may need batching or spacing", f"log: {line.strip()[:120]}")
            if re.search(r"429|rate.?limit|too many request", line, re.IGNORECASE):
                if not openai_ratelimit_seen:
                    openai_ratelimit_seen = True
                    add_issue("openai", "OpenAI rate limit hit (429) — too many concurrent requests", f"log: {line.strip()[:120]}")
            if re.search(r"insufficient_quota|billing|exceeded.*quota", line, re.IGNORECASE):
                add_issue("openai", "OpenAI quota exceeded — API key billing issue", f"log: {line.strip()[:120]}")

    # Frontend / HTTP errors
    http_patterns = [
        (r"\b(4[0-9]{2}|5[0-9]{2})\b.*(?:GET|POST|PUT|DELETE|PATCH)", "HTTP error response"),
        (r"Error: Cannot find module",   "Missing module / import error"),
        (r"Unhandled.*rejection",        "Unhandled promise rejection"),
        (r"TypeError|ReferenceError",    "JavaScript runtime error"),
        (r"Cannot read propert",         "Null/undefined access error"),
        (r"req.*\d{4,}ms",               None),  # slow route — extracted below
    ]
    for line in lines:
        # HTTP 4xx/5xx
        m = re.search(r"(4[0-9]{2}|5[0-9]{2})\s+(GET|POST|PUT|DELETE|PATCH)\s+(\S+)\s+(\d+)ms", line, re.IGNORECASE)
        if m:
            code, method, path, ms = m.groups()
            add_issue("frontend", f"HTTP {code} {method} {path}", f"log: {line.strip()[:120]}")

        # Slow routes > 5000ms
        m2 = re.search(r"(GET|POST|PUT|DELETE|PATCH)\s+(\S+)\s+(\d+)ms", line, re.IGNORECASE)
        if m2:
            method, path, ms = m2.groups()
            if int(ms) > 5000:
                msg = f"Slow route: {method} {path} took {ms}ms (>5s)"
                if not any(w["message"] == msg for w in warnings):
                    add_warning("frontend", msg, f"log: {line.strip()[:120]}")

        # JS runtime errors
        for pattern, label in [
            (r"Unhandled.*rejection", "Unhandled promise rejection"),
            (r"TypeError", "TypeError in server"),
            (r"Cannot read propert", "Null/undefined access"),
        ]:
            if re.search(pattern, line) and "node_modules" not in line and "[express]" not in line:
                if label and not any(i["message"].startswith(label) for i in issues):
                    add_issue("runtime", label, f"log: {line.strip()[:120]}")

    # ── OOM / heap growth detection ──────────────────────────────
    heap_readings = []
    for line in lines:
        m = re.search(r'Cycle #\d+ START.*heap=(\d+)MB', line)
        if m:
            heap_readings.append(int(m.group(1)))
    if len(heap_readings) >= 3:
        # Warn if heap grew by more than 100 MB across observed cycles
        heap_delta = heap_readings[-1] - heap_readings[0]
        if heap_delta > 100:
            add_warning("infrastructure",
                f"Heap growing: {heap_readings[0]}MB → {heap_readings[-1]}MB (+{heap_delta}MB) across {len(heap_readings)} observed cycles — "
                f"likely memory leak in TrainingPool or module-level Maps. "
                f"Check gradient-boost.ts TrainingPool.MAX_SIZE cap and engine.ts module-level Maps (dftEnrichmentTracker, stagnationReanalyzedIds).",
                "log scan")
        if heap_readings[-1] > 2400:
            add_issue("infrastructure",
                f"Heap at {heap_readings[-1]}MB — approaching 3072MB limit (--max-old-space-size). "
                f"OOM crash likely within next few cycles. Investigate TrainingPool size or add tighter memory caps.",
                "log scan")

    # Physics / computation issues
    physics_patterns = [
        (r"bandGap.*null.*->.*0.*MP|bandgap corrected to 0 from null", "MP cross-validation zeroed out a bandgap — null should stay null (insulator misclassified as metal)"),
        (r"imaginary freq|negative phonon|imaginary mode",     "Imaginary phonon frequencies detected (dynamic instability)"),
        (r"dynamically unstable",                               "Dynamically unstable structure in pipeline"),
        (r"phonon calculation failed|phonon.*error",            "Phonon calculation failure"),
        (r"frequencies.*\[\]|phonon\.frequencies.*empty|parsePhononOutput.*0 freq", "QE phonon job completed but frequencies list is empty — regex may not match QE output format (check omega(N-M) range notation in parsePhononOutput)"),
        (r"band structure.*failed|bandstructure.*error",        "Band structure calculation failure"),
        (r"electronic structure.*error|estructure.*fail",       "Electronic structure calculation error"),
        (r"kinetic.*lifetime=0(?![.\d])|kinetic.*unstable",      "Kinetic stability check: zero lifetime or unstable"),
        # Exclude pipeline's own rejection messages ("below -5.0 — likely numerical garbage", "too unstable to synthesize")
        # which already mean the bad data was handled — only flag truly unhandled anomalies.
        # Also exclude [express] HTTP access log lines — these are JSON API responses, not processing issues.
        # Also exclude "outside synthesizable range ... rejecting" — these are correct Stage 4 filter behavior.
        # The [1-9][0-9] alternative can false-positive on decimal substrings (e.g. "71" in "1.071"), so
        # adding "rejecting|outside synthesizable range" to the negative lookahead prevents false positives
        # from correctly-handled out-of-range formations.
        # Anchored with ^ so all lookaheads evaluate from position 0 (re.search without ^ can advance past
        # [express] and then the negative lookahead no longer sees it, causing false positives).
        (r"^(?=.*formation energy)(?=.*(?:[5-9]\.[0-9]|[1-9][0-9]))(?!.*(?:below -5\.0|likely numerical|too unstable to synthesize|data error|\[express\]|outside synthesizable range|rejecting))", "Suspiciously high/low formation energy"),
        (r'"predictedTc":\s*(?:NaN|null|undefined)|predicted_tc.*\bNaN\b|\bNaN\b.*predicted.?[Tt]c', "NaN/undefined Tc prediction"),
        (r"Eliashberg.*error|eliashberg.*fail",                 "Eliashberg solver failure"),
        (r"convex hull.*error|hull.*fail",                      "Convex hull stability computation error"),
        (r"distortion.*error|(?:crystal|distortion).*detector.*fail|detector.*fail.*crystal",  "Distortion/crystal detector failure"),
        (r"synthesis.*error.*plan|planner.*fail",               "Synthesis planner failure"),
    ]
    for line in lines:
        for pattern, msg in physics_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                if not any(i["message"] == msg for i in issues):
                    add_issue("physics", msg, f"log: {line.strip()[:120]}")

# ═══════════════════════════════════════════════════════════════
# 3. ACTIVITY FEED — PIPELINE HEALTH
# ═══════════════════════════════════════════════════════════════
pipeline_all_filtered_count = 0
recent_cycle_ends = 0
recent_cycle_starts = 0
tc_anomalies = []

if research_logs:
    # Count pipeline all-filtered events
    for log in research_logs:
        detail = (log.get("detail") or "").lower()
        event  = (log.get("event")  or "").lower()

        # All filtered: "0 passed all"
        if "0 passed all" in detail or ("stage4=0" in detail and "passed all" in detail):
            pipeline_all_filtered_count += 1

        if event in ("cycleend", "cycle-end", "cycleend"):
            recent_cycle_ends += 1
        if event in ("cyclestart", "cycle-start", "cyclestart"):
            recent_cycle_starts += 1

        # Physics anomaly: Tc > 550K with no pressure context at ambient
        m = re.search(r"tc\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*k", detail, re.IGNORECASE)
        if m:
            tc = float(m.group(1))
            pressure = re.search(r"(\d+)\s*gpa", detail, re.IGNORECASE)
            pressure_val = int(pressure.group(1)) if pressure else 0
            # Flag Tc > 550K at ambient (< 10 GPa) as unphysical
            if tc > 550 and pressure_val < 10:
                tc_anomalies.append(f"Tc={tc}K at ~{pressure_val}GPa: {detail[:80]}")

        # Repeated error events in feed
        if any(k in event for k in ["error", "fail", "crash"]):
            add_warning("pipeline", f"Error event in feed: {event} — {detail[:80]}", "research_logs")

        # ── Silent correctness anomalies in activity feed ────────────
        # These are bugs that don't crash but produce wrong science.

        # Null→0 corrections on any property (bandgap, Tc, formation energy, lambda)
        null_to_zero = re.search(
            r"(bandgap|band.gap|bandGap|formationenergy|formation.energy|predictedtc|lambda|dos).*null\s*->\s*0\b",
            detail, re.IGNORECASE)
        if null_to_zero:
            prop = null_to_zero.group(1)
            add_issue("correctness",
                f"Data correction set {prop} null→0 — zero means metal/zero-value, null means unknown; these are not equivalent",
                f"feed: {detail[:120]}")

        # MP correction zeroing a property on a known insulator/non-metal compound
        mp_zero_correction = re.search(
            r"(bandgap|band.gap).*->\s*0.*\(mp\)", detail, re.IGNORECASE)
        if mp_zero_correction:
            add_issue("correctness",
                "MP cross-validation set bandgap to 0 — MP may have matched wrong polymorph; null should stay null",
                f"feed: {detail[:120]}")

        # A property was corrected by a suspiciously large amount (>3x or >5 eV)
        large_correction = re.search(
            r"(\w+):\s*([\-\d.]+)\s*->\s*([\-\d.]+)\s*\(mp\)", detail, re.IGNORECASE)
        if large_correction:
            prop, old_val, new_val = large_correction.groups()
            try:
                ratio = abs(float(new_val)) / max(abs(float(old_val)), 0.01)
                delta = abs(float(new_val) - float(old_val))
                if ratio > 5 and delta > 2:
                    add_warning("correctness",
                        f"Large MP correction on {prop}: {old_val}→{new_val} (ratio {ratio:.1f}x) — verify this is the right polymorph",
                        f"feed: {detail[:120]}")
            except ValueError:
                pass

        # OpenAI call in the feed that timed out or errored
        if re.search(r"openai.*timeout|openai.*error|openai.*fail|llm.*timeout|llm.*fail", detail, re.IGNORECASE):
            add_issue("openai", f"OpenAI failure recorded in activity feed: {detail[:100]}", "research_logs")

    # Pipeline perpetually stuck
    if pipeline_all_filtered_count >= 5:
        add_issue("pipeline",
            f"Pipeline filtered ALL candidates in {pipeline_all_filtered_count} of last 100 log entries — nothing passing any stage",
            "research_logs")

    # SG sweep passing zero candidates — this pattern precedes event-loop freezes.
    # The sweep runs per-candidate DB queries; with 0 passed it runs to exhaustion
    # (all 967+ candidates), blocking the event loop and preventing feed writes.
    # Threshold lowered to 100 (previously max(200, total//2)) so we catch it early.
    sg_progress_entries = [l for l in research_logs if "SG sweep progress" in (l.get("event") or "")]
    if sg_progress_entries:
        last_sg = sg_progress_entries[0]  # most recent first
        sg_detail = last_sg.get("detail") or ""
        sg_ts_raw = last_sg.get("timestamp") or ""
        m_screened = re.search(r"(\d+)/(\d+)\s+screened,\s*(\d+)\s+passed", sg_detail)
        if m_screened:
            sg_screened = int(m_screened.group(1))
            sg_total    = int(m_screened.group(2))
            sg_passed   = int(m_screened.group(3))
            # Compute how long ago the last SG progress entry was
            sg_age_min = None
            try:
                sg_ts = datetime.fromisoformat(sg_ts_raw.replace("Z", "+00:00"))
                if sg_ts.tzinfo is None:
                    sg_ts = sg_ts.replace(tzinfo=timezone.utc)
                sg_age_min = (datetime.now(timezone.utc) - sg_ts).total_seconds() / 60
            except Exception:
                pass
            if sg_screened >= 100 and sg_passed == 0:
                add_issue("pipeline",
                    f"SG sweep 0 passed after {sg_screened}/{sg_total} candidates screened "
                    f"({'stale: ' + f'{sg_age_min:.0f}min ago' if sg_age_min and sg_age_min > 15 else 'recent'}). "
                    f"A zero-pass sweep runs all {sg_total} candidates to exhaustion, saturating the event loop and "
                    f"causing feed freezes. Rejections: {sg_detail[sg_detail.find('Rejections:'):sg_detail.find('Rejections:')+120] if 'Rejections:' in sg_detail else sg_detail[:120]}. "
                    f"Fix: (1) Engine restart to interrupt the sweep. "
                    f"(2) Check server/learning/engine.ts SG sweep loop — if all candidates are being rejected by "
                    f"'stability-prefilter' or 'surrogate-reject', the candidate pool may have drifted out of distribution.",
                    "research_logs")

    # Tc anomalies
    for anomaly in tc_anomalies[:3]:
        add_warning("physics", f"Unphysical Tc at ambient pressure: {anomaly}", "research_logs")

    # No cycle completions (stuck engine)
    if recent_cycle_starts > 0 and recent_cycle_ends == 0:
        add_issue("pipeline", "Cycles are starting but none completing — engine may be stuck", "research_logs")

    # (frozen feed detection moved outside this block — see below)

# ═══════════════════════════════════════════════════════════════
# 3b. FROZEN ACTIVITY FEED DETECTION
# Runs at top level — NOT inside "if research_logs:" — so it fires even when
# the API is unreachable (HTTP blocked, server down, 500 on /api/research-logs).
# When the API returns no data, falls back to scanning the server log file for
# the last engine-activity timestamp so a 2-hour freeze isn't silently missed.
# ═══════════════════════════════════════════════════════════════
_feed_freeze_now = datetime.now(timezone.utc)

# Was Active Learning recently active? (extends grace period to 5 min)
_al_recently_active = False
for _log in research_logs[:50]:
    _ev = (_log.get("event") or "").lower()
    _det = (_log.get("detail") or "").lower()
    if "active learning" in _ev or "active learning" in _det or "al cycle" in _ev:
        try:
            _ts = datetime.fromisoformat((_log.get("timestamp") or "").replace("Z", "+00:00"))
            if (_feed_freeze_now - _ts).total_seconds() < 30 * 60:
                _al_recently_active = True
                break
        except Exception:
            pass

FEED_FREEZE_MINUTES = 12 if _al_recently_active else 12
# NOTE: AL-active was 5 min but GB retraining (2x back-to-back trains of ~2.5 min each = 5 min)
# + fast-path timeout (3 min) can produce a legitimate ~8-min gap without log entries.
# Using 12 min for both AL-active and non-AL-active cases.

# ── Feed slowing detection — gaps increasing before full freeze ──────────────
# Computes inter-entry gaps for last 8 research log entries. If recent gaps are
# significantly larger than earlier gaps, the engine is slowing down (event loop
# blocking). This catches the "13-min gap → 25-min gap" pattern before full freeze.
_gb_training_in_log = False
if log_tail:
    _gb_pat = re.compile(r'\[GradientBoost\].*(?:training|retrain|cool.down|trainEnsemble|trainVariance|Post-retrain)', re.IGNORECASE)
    _gb_training_in_log = bool(_gb_pat.search(log_tail))

_feed_slowing = False
_feed_slowing_detail = ""
if len(research_logs) >= 6:
    _gap_timestamps = []
    for _sl in research_logs[:8]:
        _sl_raw = _sl.get("timestamp") or ""
        if not _sl_raw:
            continue
        try:
            _sl_ts = datetime.fromisoformat(_sl_raw.replace("Z", "+00:00"))
            if _sl_ts.tzinfo is None:
                _sl_ts = _sl_ts.replace(tzinfo=timezone.utc)
            _gap_timestamps.append(_sl_ts)
        except Exception:
            pass
    if len(_gap_timestamps) >= 5:
        # Research logs are newest-first, reverse to get chronological order
        _gap_timestamps.sort()
        _gaps_s = [((_gap_timestamps[i+1] - _gap_timestamps[i]).total_seconds()) for i in range(len(_gap_timestamps)-1)]
        # Compare average of first half vs second half of gaps
        _mid = len(_gaps_s) // 2
        _early_avg = sum(_gaps_s[:_mid]) / max(1, _mid)
        _late_avg = sum(_gaps_s[_mid:]) / max(1, len(_gaps_s) - _mid)
        # Flag if recent gaps are 3x larger than earlier gaps AND recent avg > 5 min
        if _late_avg > _early_avg * 3.0 and _late_avg > 300:
            _feed_slowing = True
            _cause_hint = "GB training blocking event loop (local tree training not offloaded to GCP)" if _gb_training_in_log else "event loop blocked by CPU-intensive operation"
            _feed_slowing_detail = (
                f"Activity feed SLOWING: recent inter-entry gaps avg {_late_avg/60:.1f} min vs earlier {_early_avg/60:.1f} min "
                f"(3x slowdown). Cause: {_cause_hint}. "
                f"This pattern leads to a full freeze within the next 1-2 cycles. "
                f"Fix immediately — do NOT wait for full freeze."
            )

# ── Event loop block detection from network metrics ──────────────────────────
# If any endpoint took >60s in the last 2 min of network metrics, the event loop
# was completely blocked. This is the clearest signal of local GB training.
_event_loop_blocked_s = 0
_event_loop_blocked_ep = ""
for _nm in net_metrics:
    _dur = _nm.get("durationMs", 0)
    if _dur > _event_loop_blocked_s:
        _event_loop_blocked_s = _dur
        _event_loop_blocked_ep = _nm.get("endpoint", "")
_event_loop_blocked_min = _event_loop_blocked_s / 60000

# Step 1: try to get the most recent feed timestamp from the API response
_feed_most_recent_ts = None
for _log in research_logs[:5]:
    _ts_raw = _log.get("timestamp") or ""
    if _ts_raw:
        try:
            _ts = datetime.fromisoformat(_ts_raw.replace("Z", "+00:00"))
            if _ts.tzinfo is None:
                _ts = _ts.replace(tzinfo=timezone.utc)
            _feed_most_recent_ts = _ts
            break
        except Exception:
            pass

# Step 2: if the API gave us nothing (HTTP blocked, 500, etc.), fall back to
# scanning the server log for the last line that looks like engine activity.
# This is what catches a 2-hour freeze when the server is HTTP-blocked.
# Log format has two timestamp styles:
#   [express] lines: ISO timestamps embedded in JSON response bodies
#   [Engine] lines:  unix millisecond timestamps at end of line (e.g. 1775089982918)
# Only fall back when HTTP is actually blocked. A transient /api/research-logs
# timeout (>4s curl limit) on a responsive server would otherwise pick up a
# stale historical ISO from an [express] body and produce a false-positive
# FROZEN report — past fixHistory shows this happening repeatedly.
_feed_ts_source = "api"
if _feed_most_recent_ts is None and log_tail and http_blocked:
    import re as _re_ff
    _activity_line_pat = _re_ff.compile(
        r'\[Engine\]|\[express\]|Cycle #\d+|\[research-logs\]|cycleEnd|phaseUpdate|Phase \d+|'
        r'\[Active Learning\]|\[DFT\]|\[GNN\]|\[XGBoost\]|\[Synthesis\]|\[Autonomous\]',
        _re_ff.IGNORECASE
    )
    # ISO timestamp (appears in [express] JSON bodies and server-start banner)
    _iso_ts_pat = _re_ff.compile(r'"?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"?')
    # Unix ms timestamp (appears in [Engine] lines — 13-digit, year 2020-2033)
    _unix_ms_pat = _re_ff.compile(r'\b(1[5-9]\d{11}|[2-9]\d{12})\b')
    for _line in reversed(log_tail.split("\n")):
        if not _activity_line_pat.search(_line):
            continue
        # Skip client-errors responses — they contain historical UI error timestamps,
        # not engine activity timestamps, and would produce false-positive freezes.
        if "/api/client-errors" in _line:
            continue
        # [Autonomous] lines (SG sweep REJECTED/PASSED) prove the engine is alive RIGHT NOW.
        # They have no embedded timestamp, so use current time as the activity marker.
        if "[Autonomous]" in _line:
            _feed_most_recent_ts = _feed_freeze_now
            _feed_ts_source = "server-log"
            break
        # Try ISO first. For [express] lines with a response body (:: separator),
        # only accept ISO timestamps that are recent (within 2x FEED_FREEZE_MINUTES).
        # JSON response bodies often contain historical material "updatedAt"/"timestamp"
        # fields from the DB, which are stale and produce false-positive FROZEN reports.
        _m = _iso_ts_pat.search(_line)
        if _m:
            try:
                _ts_str = _m.group(1).rstrip("Z").replace(" ", "T")
                _ts = datetime.fromisoformat(_ts_str)
                if _ts.tzinfo is None:
                    _ts = _ts.replace(tzinfo=timezone.utc)
                _age_s = (_feed_freeze_now - _ts).total_seconds()
                # For [express] body lines, reject timestamps older than 2x freeze threshold
                # to filter out historical material data timestamps.
                if "[express]" in _line and "::" in _line and _age_s > FEED_FREEZE_MINUTES * 2 * 60:
                    pass  # stale body timestamp — skip, keep scanning
                else:
                    _feed_most_recent_ts = _ts
                    _feed_ts_source = "server-log"
                    break
            except Exception:
                pass
        # Try unix ms — but NOT on [express] response body lines, which contain
        # signal/material IDs that embed unix timestamps (e.g. "signal-1775070286316-xyz")
        # unrelated to actual server activity time.
        if not ("[express]" in _line and "::" in _line):
            _mu = _unix_ms_pat.search(_line)
            if _mu:
                try:
                    _ts = datetime.fromtimestamp(int(_mu.group(1)) / 1000.0, tz=timezone.utc)
                    _feed_most_recent_ts = _ts
                    _feed_ts_source = "server-log"
                    break
                except Exception:
                    pass

# Step 3: evaluate the freeze
_server_startup_age_min = float("inf")
if server_start_ts is not None:
    _server_startup_age_min = (_feed_freeze_now - server_start_ts).total_seconds() / 60
_skip_freeze = _server_startup_age_min < 5

_engine_state_ff = ""
if isinstance(engine_status, dict):
    _engine_state_ff = (engine_status.get("state") or engine_status.get("status") or "").lower()

# ── Event loop block from network metrics (most urgent) ──────────────────────
if _event_loop_blocked_min >= 5.0 and not _skip_freeze:
    _cause = "local GB training (trainGradientBoosting / trainEnsembleXGB blocking event loop)" if _gb_training_in_log else "CPU-intensive operation blocking event loop"
    add_issue("pipeline",
        f"EVENT LOOP BLOCKED {_event_loop_blocked_min:.0f} min — '{_event_loop_blocked_ep}' took "
        f"{_event_loop_blocked_min:.0f} min to respond. Cause: {_cause}. "
        f"The server is processing no requests and writing no research logs during this time. "
        f"This IS the feed freeze root cause. "
        f"FREE INVESTIGATION MODE: Read server/learning/gradient-boost.ts — verify that ALL "
        f"trainGradientBoosting / trainEnsembleXGB / trainVarianceEnsembleXGB calls check "
        f"OFFLOAD_XGB_TO_GCP env var and dispatch to GCP instead of training locally. "
        f"Also check retrainWithAccumulatedData, incorporateFailureData, and getTrainedModel. "
        f"If local training calls remain, add: "
        f"if (process.env.OFFLOAD_XGB_TO_GCP === 'true') {{ dispatch to GCP; return; }}",
        "network-metrics")
elif _event_loop_blocked_min >= 1.0 and not _skip_freeze:
    add_warning("pipeline",
        f"Event loop slow: '{_event_loop_blocked_ep}' took {_event_loop_blocked_min:.1f} min to respond. "
        f"Watch for escalation to full freeze.",
        "network-metrics")

# ── Feed slowing (pre-freeze warning) ────────────────────────────────────────
if _feed_slowing and not _skip_freeze:
    add_issue("pipeline", _feed_slowing_detail, "research_logs")

if _feed_most_recent_ts is not None and not _skip_freeze:
    _age_min = (_feed_freeze_now - _feed_most_recent_ts).total_seconds() / 60
    _src_note = f" (timestamp from {_feed_ts_source})" if _feed_ts_source != "api" else ""
    if _age_min > FEED_FREEZE_MINUTES:
        if _engine_state_ff in ("running", "active", "learning", "cycling"):
            # Engine explicitly says running — always an issue regardless of HTTP state.
            _freeze_cause = ""
            if _gb_training_in_log:
                _freeze_cause = (
                    f" GB training detected in log — likely cause: local trainGradientBoosting / "
                    f"trainEnsembleXGB blocking event loop. OFFLOAD_XGB_TO_GCP should be 'true' "
                    f"and ALL training paths must check this flag. FREE INVESTIGATION MODE: "
                    f"Read server/learning/gradient-boost.ts, find any trainGradientBoosting call "
                    f"not guarded by OFFLOAD_XGB_TO_GCP check, and add the GCP dispatch guard."
                )
            add_issue("pipeline",
                f"Activity feed FROZEN {_age_min:.0f} min — engine reports state='{_engine_state_ff}' but no new entries{_src_note}.{_freeze_cause} "
                f"Fix: (1) POST /api/engine/stop, wait 5s, POST /api/engine/start. "
                f"(2) If feed still frozen after 60s, full server restart. "
                f"(3) After restart, read the log for what was blocking and fix the root cause.",
                f"research_logs{_src_note}")
        elif http_blocked and _age_min > 20:
            # HTTP blocked so engine state is unknown (API unreachable).
            # GB ensemble training (trainEnsembleXGB + trainVarianceEnsembleXGB + 25s cool-down)
            # blocks up to 7 min for back-to-back retrains. SG sweep (8-min wall limit) +
            # post-sweep backfillGBScores + recalculatePhysics (~3 min) can span two engine cycles
            # = up to 14 min of no research-log entries during expected operation.
            # Only flag as genuine freeze if the feed has been stale >20 min.
            add_issue("pipeline",
                f"Activity feed FROZEN {_age_min:.0f} min AND server HTTP-blocked{_src_note}. "
                f"SG sweep + post-sweep backfill blocks up to 14 min — {_age_min:.0f} min is a genuine freeze. "
                f"Engine loop stuck (likely SG sweep or DB connection exhaustion). "
                f"ESCALATION: (1) Try engine restart first: POST /api/engine/stop, wait 5s, POST /api/engine/start. "
                f"Wait 60s and check /api/research-logs?limit=5 — if new entries appear, engine restart was sufficient. "
                f"(2) Only do full server restart (bash scripts/start-server-logged.sh) if engine restart does not unfreeze the feed. "
                f"(3) After fixing, grep the log for what was running at freeze time and fix the root cause in code.",
                f"server-log")
        elif _age_min > 60:
            add_issue("pipeline",
                f"Activity feed silent {_age_min:.0f} min (engine state: '{_engine_state_ff or 'unknown'}'){_src_note} — "
                f"engine may be stopped or crashed.",
                "research_logs")
        else:
            add_warning("pipeline",
                f"Activity feed quiet {_age_min:.0f} min (engine: '{_engine_state_ff or 'unknown'}'){_src_note}. "
                f"Normal if Active Learning is running or engine was stopped.",
                "research_logs")
elif _feed_most_recent_ts is None and not _skip_freeze and not http_blocked:
    # API reachable but no timestamps found at all → table empty
    add_issue("pipeline",
        "Activity feed is completely empty — engine has never run or research_logs table is not being written to.",
        "research_logs")
elif _feed_most_recent_ts is None and not _skip_freeze and http_blocked:
    # HTTP blocked AND no log timestamps found → server may be fully stuck
    add_issue("pipeline",
        f"Activity feed unavailable: server HTTP-blocked and no engine-activity timestamps found in recent log "
        f"(server up {_server_startup_age_min:.0f} min). "
        f"Fix: restart server via 'bash scripts/start-server-logged.sh'.",
        "healthcheck")

# ═══════════════════════════════════════════════════════════════
# 4. DFT QUEUE HEALTH
# ═══════════════════════════════════════════════════════════════
if dft_status and isinstance(dft_status, dict):
    queued   = dft_status.get("queued", 0) or 0
    running  = dft_status.get("running", 0) or 0
    failed   = dft_status.get("failed", 0) or 0
    if failed and int(failed) > 5:
        add_issue("dft", f"DFT queue has {failed} failed jobs", "dft-status API")
    if queued and int(queued) > 50:
        add_warning("dft", f"DFT queue backlog: {queued} jobs queued", "dft-status API")

# ── Repeated OOM crash detection ─────────────────────────────
# If the state file shows 3+ OOM crashes in fix history, the monitor must
# investigate root cause instead of just restarting.
try:
    import os as _os
    state_path = _os.path.join(_os.path.dirname(report_file), "monitor-state.json")
    with open(state_path, encoding="utf-8") as _sf:
        state_data = json.load(_sf)
    fix_history = state_data.get("fixHistory", [])
    recent_ooms = [
        f for entry in fix_history[-5:]
        for f in entry.get("fixes", [])
        if ("oom" in f.lower() or "out of memory" in f.lower() or "heap exhausted" in f.lower() or "memory leak" in f.lower() or "enomem" in f.lower())
        and "false positive" not in f.lower()
        and "root cause found" not in f.lower()
    ]
    if len(recent_ooms) >= 3:
        add_issue("infrastructure",
            f"REPEATED OOM CRASHES: {len(recent_ooms)} OOM restarts in recent fix history — "
            f"this is a memory leak, not a one-off crash. DO NOT just restart again. "
            f"Root cause: TrainingPool.featureCache in gradient-boost.ts has no size cap and grows "
            f"unboundedly across cycles. Fix: add MAX_SIZE=6000 eviction in TrainingPool.add(). "
            f"Also verify NODE_OPTIONS=--max-old-space-size=3072 is set in package.json dev script.",
            "monitor-state.json")
    elif len(recent_ooms) >= 2:
        add_warning("infrastructure",
            f"{len(recent_ooms)} OOM restarts in recent history — investigate memory growth. "
            f"Check heap= readings in server log (logged at each cycle start). "
            f"Likely culprit: TrainingPool.featureCache (gradient-boost.ts) growing without bound.",
            "monitor-state.json")
except Exception:
    pass

# ── Surrogate Tc clustering / physics-veto detection ─────────
# Check recent candidates for identical Tc values that suggest the GB model
# is returning its training mean (physics veto not firing).
try:
    cands_raw = read_file("candidates.json")
    cands = json.loads(cands_raw) if cands_raw and cands_raw.strip() not in ("null", "") else []
    if isinstance(cands, list) and len(cands) >= 5:
        recent_tcs = [c.get("predictedTc") for c in cands[:15] if c.get("predictedTc") is not None]
        if len(recent_tcs) >= 5:
            from collections import Counter
            tc_counts = Counter(recent_tcs)
            most_common_tc, most_common_n = tc_counts.most_common(1)[0]
            pct = most_common_n / len(recent_tcs)
            if pct >= 0.5 and most_common_n >= 4:
                add_issue("physics",
                    f"Surrogate Tc clustering: {most_common_n}/{len(recent_tcs)} recent candidates all show Tc={most_common_tc}K — "
                    f"GB model is returning training mean; physics veto (autoPhysExplicitlyZero) may not be firing. "
                    f"Check engine.ts finalTc block — autoPhysExplicitlyZero must come before reconciledAuto check.",
                    "candidates JSON")
except Exception:
    pass

# ── Identical band structure defaults detection ───────────────────────
# When the band surrogate returns its "zero-hidden-representation" defaults,
# multiple structurally similar compounds get identical values — which then feed
# identical physics signals into QCP/DOS/VHS detection, producing spurious identical scores.
# Signature: 3+ candidates share the same bandgap AND flatBand AND VHS values.
try:
    cands_band_raw = read_file("candidates.json")
    cands_band = json.loads(cands_band_raw) if cands_band_raw and cands_band_raw.strip() not in ("null","") else []
    if isinstance(cands_band, list) and len(cands_band) >= 3:
        band_signatures = []
        for c in cands_band[:20]:
            mf = c.get("mlFeatures") or {}
            bsig = (
                mf.get("bandGap") or mf.get("gap"),
                mf.get("flatBandScore") or mf.get("flatBand"),
                mf.get("vhsProximity") or mf.get("vhs"),
                mf.get("nestingFromBands") or mf.get("nesting"),
            )
            if all(v is not None for v in bsig):
                band_signatures.append((c.get("formula","?"), bsig))
        if len(band_signatures) >= 3:
            from collections import Counter
            sig_counts = Counter(sig for _, sig in band_signatures)
            most_common_sig, most_common_n = sig_counts.most_common(1)[0]
            if most_common_n >= 3:
                matching_formulas = [f for f, sig in band_signatures if sig == most_common_sig]
                add_issue("correctness",
                    f"Band structure surrogate returning identical defaults for {most_common_n} compounds "
                    f"({', '.join(matching_formulas[:4])}): "
                    f"gap={most_common_sig[0]}, flatBand={most_common_sig[1]}, VHS={most_common_sig[2]}, nesting={most_common_sig[3]}. "
                    f"Root cause: computePhysicsCalibration in band-structure-surrogate.ts uses only tmCount/dElectronFraction "
                    f"for scaling — compounds with same tmCount but different elements (e.g. Cu vs Ni, Pr vs Eu) "
                    f"collapse to identical outputs. Fix: add fElectronFraction/closedDShellFraction/openDShellTmFraction to calib "
                    f"and use them in the DOS/VHS/nesting/flatBand scaling.",
                    "candidates JSON")
except Exception:
    pass

# ── Internal physics contradiction detection ──────────────────────────
# Flags candidates where the physics signals are internally inconsistent:
# e.g. metallic=false + spin-fluctuation pairing, or stable ferromagnet + high QCP score.
try:
    cands_contra_raw = read_file("candidates.json")
    cands_contra = json.loads(cands_contra_raw) if cands_contra_raw and cands_contra_raw.strip() not in ("null","") else []
    if isinstance(cands_contra, list):
        for c in cands_contra[:20]:
            formula = c.get("formula") or ""
            mf = c.get("mlFeatures") or {}
            notes = (c.get("notes") or "").lower()

            is_metallic = mf.get("metallic") if "metallic" in mf else None
            pairing = (mf.get("pairingChannel") or mf.get("pairingMechanism") or "").lower()
            stoner = float(mf.get("stonerRatio") or mf.get("stonerParameter") or mf.get("stonerProduct") or 0)
            qcp_raw2 = mf.get("qcp") or mf.get("qcpScore") or 0
            qcp2 = float(qcp_raw2) if isinstance(qcp_raw2, (int, float)) else (1.0 if qcp_raw2 else 0.0)
            qcp_type = (mf.get("qcpType") or "").lower()

            # 1. metallic=false + spin-fluctuation pairing (requires itinerant electrons)
            if is_metallic is False and "spin-fluctuation" in pairing:
                key = f"metallic-sf:{formula}"
                if not any(key in str(i) for i in issues):
                    add_issue("correctness",
                        f"Internal contradiction for {formula}: metallic=false but pairing={pairing}. "
                        f"Spin-fluctuation SC requires itinerant electrons — a non-metallic compound cannot have SF pairing. "
                        f"Check whether band surrogate gap and DFT metallicity are being reconciled; "
                        f"DFT metallic=false should override surrogate pairing channel selection.",
                        f"candidates: {formula}")

            # 2. Stable ferromagnet (Stoner >= 1.0) with high QCP score and non-'ferromagnet' type
            if stoner >= 1.0 and qcp2 > 0.5 and "ferromagnet" not in qcp_type:
                key = f"fm-qcp:{formula}"
                if not any(key in str(i) for i in issues):
                    add_issue("correctness",
                        f"Stoner/QCP contradiction for {formula}: Stoner={stoner:.2f} >= 1.0 (stable ferromagnet) "
                        f"but QCP score={qcp2:.3f} with type='{qcp_type}' — a confirmed ferromagnet is NOT near a QCP. "
                        f"Fix: in computeQuantumCriticalConstraint (advanced-constraints.ts), when isStableFerromagnet=true, "
                        f"suppress QCP score by 85% and reclassify qcpType to 'ferromagnet'.",
                        f"candidates: {formula}")

            # 3. Stable ferromagnet with high Mott QCP score (non-metallic ferromagnet misclassified)
            mott_from_qc = mf.get("quantumCriticality") or {}
            mott_channel = float((mott_from_qc.get("channelScores") or {}).get("mott") or 0)
            if stoner >= 1.0 and mott_channel > 0.3:
                key = f"fm-mott:{formula}"
                if not any(key in str(i) for i in issues):
                    add_warning("correctness",
                        f"Stable ferromagnet {formula} (Stoner={stoner:.2f}) has Mott channel score={mott_channel:.2f}. "
                        f"Stable ferromagnets are NOT near Mott transitions — this is an artifact of low metallicity "
                        f"inflating the Mott score. Fix: in computeMottChannel (quantum-criticality.ts), "
                        f"cap mottScore at 0.15 when isStableFerromagnet=true.",
                        f"candidates: {formula}")
except Exception:
    pass

# ── Wrongful rejection / physics-signal mismatch detection ───────────
# Looks for candidates with strong physics signals that got a very low
# score, suggesting a spurious penalty is overriding real science.
try:
    cands_raw2 = read_file("candidates.json")
    cands2 = json.loads(cands_raw2) if cands_raw2 and cands_raw2.strip() not in ("null", "") else []
    if isinstance(cands2, list):
        for c in cands2[:20]:
            formula   = c.get("formula") or ""
            mf        = c.get("mlFeatures") or {}
            ensemble  = float(c.get("ensembleScore") or 0)
            pressure  = float(c.get("pressureGpa") or mf.get("pressureGpa") or 0)
            notes     = (c.get("notes") or "").lower()
            delib     = float(c.get("deliberationScore") or 0)

            # Parse element counts for H-rich hydride check
            el_counts = {}
            for em in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
                el, cnt = em.groups()
                el_counts[el] = int(cnt) if cnt else 1
            h_count = el_counts.get("H", 0)
            NON_H_NONMETALLOIDS = {"H","He","N","O","F","Cl","Br","I","At","Ne","Ar","Kr","Xe","Rn"}
            metal_count = sum(n for e, n in el_counts.items() if e not in NON_H_NONMETALLOIDS)
            h_ratio = h_count / max(metal_count, 1)
            is_h_rich = h_count >= 4 and metal_count > 0 and h_ratio > 3

            lifshitz = float(mf.get("lifshitz") or mf.get("lifshitzProximity") or mf.get("vanHoveProximity") or 0)
            qcp_raw  = mf.get("qcp") or mf.get("qcpScore") or 0
            qcp_val  = float(qcp_raw) if isinstance(qcp_raw, (int, float)) else (1.0 if qcp_raw else 0.0)
            stoner   = float(mf.get("stonerRatio") or mf.get("stonerParameter") or 0)

            # 1. H-rich hydride low score — d-band penalty likely applied
            if is_h_rich and (ensemble < 0.15 or delib < 0.35):
                key = f"hrich-reject:{formula}"
                if not any(key in str(i) for i in issues):
                    add_issue("correctness",
                        f"Potential wrongful rejection of H-rich hydride {formula} "
                        f"(H:metal={h_ratio:.1f}, score={ensemble:.3f}, delib={delib:.3f}): "
                        f"d-band penalty likely applied incorrectly — SC in H-rich hydrides is driven by "
                        f"H-sublattice phonons, not TM d-band filling. "
                        f"Fix: isHRichHydride check in runPhysicsMerit (deliberative-evaluator.ts).",
                        f"candidates: {formula}")

            # 2. Pressure-blind hull penalty at high pressure
            if pressure > 50 and (ensemble < 0.25 or delib < 0.35):
                if "hull" in notes or "far from hull" in notes or "synthesis challenging" in notes:
                    key = f"hull-pressure:{formula}"
                    if not any(key in str(i) for i in issues):
                        add_issue("correctness",
                            f"High-pressure candidate {formula} at {pressure:.0f} GPa penalized for ambient-pressure hull distance — "
                            f"ambient hull is physically irrelevant at >{pressure:.0f} GPa. "
                            f"Fix: runRiskAssessment in deliberative-evaluator.ts must read mlFeatures.pressureGpa "
                            f"(check it is included in the mlFeatures spread in engine.ts finalPayload).",
                            f"candidates: {formula}")

            # 3. Stoner + QCP treated as ferromagnet penalty instead of spin-fluctuation SC
            if stoner > 3.0 and qcp_val >= 0.3 and (ensemble < 0.25 or delib < 0.40):
                if "ferromagnet" in notes or "stoner" in notes:
                    key = f"stoner-qcp:{formula}"
                    if not any(key in str(i) for i in issues):
                        add_issue("correctness",
                            f"Stoner/QCP misevaluation for {formula}: Stoner={stoner:.2f} + QCP confirmed "
                            f"but treated as ferromagnet suppressing SC (score={ensemble:.3f}). "
                            f"Should be interpreted as spin-fluctuation SC near magnetic QCP — no suppression penalty. "
                            f"Fix: runPhysicsMerit Stoner block in deliberative-evaluator.ts (penalty only when QCP < 0.3).",
                            f"candidates: {formula}")

            # 4. Strong physics signals but poor overall score (generic catch-all)
            if lifshitz >= 0.8 and qcp_val >= 0.5 and ensemble < 0.20:
                key = f"strong-physics-low-score:{formula}"
                if not any(key in str(w) for w in warnings):
                    add_warning("correctness",
                        f"Strong physics signals for {formula} (lifshitz={lifshitz:.2f}, QCP={qcp_val:.2f}) "
                        f"but low ensemble score ({ensemble:.3f}) — check if a spurious penalty is overriding "
                        f"positive physics evidence (d-band, hull, Stoner, pressure viability).",
                        f"candidates: {formula}")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
# 5. ML CALIBRATION DRIFT
# ═══════════════════════════════════════════════════════════════
if ml_calibration and isinstance(ml_calibration, dict):
    mae = ml_calibration.get("mae") or ml_calibration.get("meanAbsoluteError")
    if mae and float(mae) > 30:
        add_warning("ml", f"ML model MAE is {mae} K — high prediction error, may need recalibration", "ml-calibration API")

# ── Pre-compute timing variables used by page-probe and client-error sections ──
# in_warmup and now_utc are used in sections 6 and 7; define them before section 6
# so they are always available when the page-probes loop runs.
now_utc = datetime.now(timezone.utc)
in_warmup = server_start_ts is not None and (now_utc - server_start_ts).total_seconds() < 180

# ═══════════════════════════════════════════════════════════════
# 6. PAGE ENDPOINT PROBES
# ═══════════════════════════════════════════════════════════════
page_probe_results = {}
try:
    probe_raw = read_file("page_probes.txt")
    for line in probe_raw.strip().splitlines():
        if "=" not in line:
            continue
        page_name, probe_val = line.split("=", 1)
        parts = probe_val.strip().split(":")
        if len(parts) == 2:
            code, ms = parts
            status_int = int(code)
            # Sanity-check: HTTP status codes are 1xx–5xx (100-599).
            # Values outside this range are probe parsing artifacts (e.g. date command
            # producing a non-millisecond timestamp that gets mixed into the output).
            # Treat bogus codes as "unreachable" (0) so they show as warnings, not issues.
            if not (100 <= status_int <= 599):
                status_int = 0
            page_probe_results[page_name.strip()] = {"statusCode": status_int, "durationMs": int(ms)}
except Exception:
    pass

PAGE_LABELS = {
    "dashboard":           "/api/dashboard",
    "research_pipeline":   "/api/research-logs",
    "computational_physics": "/api/pipeline-stats",
    "superconductor_lab":  "/api/superconductor-candidates",
    "materials_database":  "/api/materials",
    "novel_discovery":     "/api/novel-predictions",
    "dft_queue":           "/api/dft-status",
    "atomic_explorer":     "/api/elements",
    "pareto_frontier":     "/api/pareto-frontier",
}

for page_name, endpoint in PAGE_LABELS.items():
    probe = page_probe_results.get(page_name, {})
    code = probe.get("statusCode", 0)
    ms = probe.get("durationMs", 0)
    label = page_name.replace("_", " ").title()

    if code == 0:
        if http_blocked:
            add_warning("frontend", f"{label}: endpoint unreachable ({endpoint}) — HTTP blocked during GradientBoost training (transient)", "page-probe")
        else:
            add_warning("frontend", f"{label}: endpoint unreachable ({endpoint})", "page-probe")
    elif code >= 500:
        add_issue("frontend", f"{label}: server error {code} on {endpoint}", "page-probe")
    elif code >= 400:
        add_issue("frontend", f"{label}: client error {code} on {endpoint}", "page-probe")
    elif ms > 5000:
        if in_warmup:
            # Server just restarted — cold-cache misses make first probes slow. Not a real issue.
            add_warning("frontend", f"{label}: slow response {ms}ms on {endpoint} — within 3-min post-restart warm-up window (cold cache expected)", "page-probe")
        elif http_blocked:
            # HTTP was blocked by GradientBoost training during this probe run —
            # slow responses are expected artifacts of event-loop congestion, not real hangs.
            add_warning("frontend", f"{label}: slow response {ms}ms on {endpoint} (event-loop congestion during training — transient)", "page-probe")
        elif ms < 15000:
            # Borderline slow (5-15s): re-probe once to distinguish transient GradientBoost spikes
            # (training can start AFTER the health check, making HTTP_BLOCKED=False but probes slow).
            # If the re-probe is fast (< 3s), treat the original as a transient spike.
            import subprocess, time as _time
            _recheck_url = f"$HOST{endpoint}"
            try:
                _r = subprocess.run(
                    ["curl", "-o", "/dev/null", "-sf", "-w", "%{time_total}", "--max-time", "5", _recheck_url],
                    capture_output=True, text=True, timeout=6
                )
                _recheck_ms = int(float(_r.stdout.strip() or "99") * 1000)
            except Exception:
                _recheck_ms = 99999
            if _recheck_ms < 3000:
                add_warning("frontend", f"{label}: slow response {ms}ms on {endpoint} (re-probe: {_recheck_ms}ms — transient training spike)", "page-probe")
            else:
                add_issue("frontend", f"{label}: slow response {ms}ms on {endpoint} (>5s — re-probe also slow: {_recheck_ms}ms)", "page-probe")
        else:
            add_issue("frontend", f"{label}: slow response {ms}ms on {endpoint} (>15s — page will appear to hang)", "page-probe")
    elif ms > 3000:
        add_warning("frontend", f"{label}: response took {ms}ms on {endpoint} (approaching 5s threshold)", "page-probe")

# ═══════════════════════════════════════════════════════════════
# 7. CLIENT-SIDE ERRORS FROM BROWSER
# ═══════════════════════════════════════════════════════════════
client_errors_r = safe_json(read_file("client_errors.json"))
client_errors_list = []
if isinstance(client_errors_r, dict):
    client_errors_list = client_errors_r.get("errors", [])
elif isinstance(client_errors_r, list):
    client_errors_list = client_errors_r

# Filter client errors to only those written AFTER the server last started.
# This discards stale cold-cache / OOM-restart entries (e.g. a 169s slow-load
# from a previous session) that would otherwise keep triggering false positives
# and cause the monitor to restart the server unnecessarily.
#
# Additionally apply a 3-minute warm-up grace window after restart:
# cold-cache misses always produce slow loads on first page visit — these are
# expected and should not count as issues during the warm-up period.
from datetime import datetime, timedelta, timezone

server_start_ts = None
_ss_raw = read_file("server_start_time.txt").strip()
if _ss_raw:
    try:
        server_start_ts = datetime.fromisoformat(_ss_raw.replace("Z", "+00:00"))
        if server_start_ts.tzinfo is None:
            server_start_ts = server_start_ts.replace(tzinfo=timezone.utc)
    except Exception:
        server_start_ts = None

now_utc = datetime.now(timezone.utc)
# Warm-up: first 3 min after restart — slow loads are cold-cache artefacts
in_warmup = server_start_ts is not None and (now_utc - server_start_ts).total_seconds() < 180

if client_errors_list:
    filtered = []
    for ce in client_errors_list:
        ts_str = ce.get("timestamp") or ""
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Discard if written before server last started (pre-restart stale entry)
            if server_start_ts and ts < server_start_ts:
                continue
            # Discard if older than 15 minutes — already reported/fixed in a prior cycle.
            # Without this window, a single transient 500 (e.g. during SG sweep wind-down)
            # keeps re-appearing as an issue every cycle indefinitely.
            MAX_CLIENT_ERROR_AGE_MINUTES = 15
            if (now_utc - ts).total_seconds() > MAX_CLIENT_ERROR_AGE_MINUTES * 60:
                continue
            filtered.append(ce)
        except Exception:
            filtered.append(ce)  # include if timestamp unparseable
    client_errors_list = filtered

if client_errors_list:
    # Group by type to avoid flooding with duplicates
    seen_msgs: set = set()
    crash_count = 0
    timeout_count = 0
    slow_count = 0
    query_err_count = 0

    for ce in client_errors_list[:50]:
        ctype = (ce.get("type") or "").lower()
        msg   = ce.get("message") or ""
        page  = ce.get("page") or "?"
        ep    = ce.get("endpoint") or ""
        ms    = ce.get("durationMs") or 0
        code  = ce.get("statusCode") or 0

        if ctype == "render-crash":
            crash_count += 1
            dedup_key = f"crash:{page}:{msg[:60]}"
            if dedup_key not in seen_msgs:
                seen_msgs.add(dedup_key)
                add_issue("frontend", f"Page crash on {page}: {msg[:100]}", "client-errors API")

        elif ctype == "query-timeout":
            timeout_count += 1
            dedup_key = f"timeout:{ep}"
            if dedup_key not in seen_msgs:
                seen_msgs.add(dedup_key)
                if in_warmup:
                    add_warning("openai" if "openai" in ep.lower() else "frontend",
                        f"Query timeout on {ep} ({ms}ms) — during post-restart warm-up, may be cold-cache (page: {page})",
                        "client-errors API")
                else:
                    add_issue("openai" if "openai" in ep.lower() else "frontend",
                        f"Query timeout on {ep} ({ms}ms) — seen from page {page}",
                        "client-errors API")

        elif ctype == "slow-load":
            slow_count += 1
            dedup_key = f"slow:{ep}"
            if dedup_key not in seen_msgs:
                seen_msgs.add(dedup_key)
                if in_warmup:
                    # Cold-cache misses always hit first-load hard — not a real issue
                    add_warning("frontend",
                        f"Slow load: {ep} took {ms}ms — within 3-min post-restart warm-up window, cold-cache expected",
                        "client-errors API")
                else:
                    add_warning("frontend",
                        f"Slow load: {ep} took {ms}ms on page {page}",
                        "client-errors API")

        elif ctype == "query-error":
            query_err_count += 1
            dedup_key = f"qerr:{ep}:{code}"
            if dedup_key not in seen_msgs:
                seen_msgs.add(dedup_key)
                severity_fn = add_issue if code >= 500 else add_warning
                severity_fn("frontend",
                    f"Query error HTTP {code} on {ep} (page: {page})",
                    "client-errors API")

    # Self-referential error reporter: /api/client-errors itself appears in errors
    if any(("/api/client-errors" in (ce.get("endpoint") or "")) for ce in client_errors_list[:50]):
        if not any("error-reporter feedback loop" in i["message"] for i in issues):
            add_issue("frontend",
                "Error reporter feedback loop: /api/client-errors endpoint is appearing in query-error reports. "
                "reportClientError() must never report its own failures (creates infinite loop). "
                "Fix: add 'if (payload.endpoint === \"/api/client-errors\") return;' as the first line of "
                "reportClientError() in client/src/lib/queryClient.ts.",
                "client-errors API")

    # Error reporting flood: many dashboard query-errors clustered in time from cycleEnd keys
    # Pattern: cycleEnd invalidates 24 CYCLE_END_KEYS simultaneously → each failure calls reportClientError
    # → rapid-fire POSTs hit the writeLimiter (30/min) → 429 storm
    CYCLE_END_MARKERS = [
        "band-structure", "gnn/", "cross-engine", "synthesis-discovery",
        "synthesis-planner", "heuristic-synthesis", "ml-synthesis",
        "retrosynthesis", "synthesis-gate", "reaction-network",
        "theory-report", "tsc-candidates", "surrogate-fitness",
        "heterostructure", "disorder", "interface-relaxation",
        "energy-landscape", "distortion"
    ]
    dashboard_errs = [
        ce for ce in client_errors_list
        if (ce.get("page") in ("/", "") or "dashboard" in (ce.get("page") or ""))
        and (ce.get("type") or "") == "query-error"
    ]
    if query_err_count > 12 and len(dashboard_errs) > 8:
        ce_timestamps = []
        for ce in dashboard_errs:
            try:
                _t = datetime.fromisoformat((ce.get("timestamp") or "").replace("Z", "+00:00"))
                if _t.tzinfo is None:
                    _t = _t.replace(tzinfo=timezone.utc)
                ce_timestamps.append(_t)
            except Exception:
                pass
        if len(ce_timestamps) >= 5:
            _span = (max(ce_timestamps) - min(ce_timestamps)).total_seconds()
            if _span < 90:
                _ce_endpoints = set(ce.get("endpoint", "") for ce in dashboard_errs)
                _cycle_end_matches = sum(
                    1 for ep in _ce_endpoints
                    if any(m in ep for m in CYCLE_END_MARKERS)
                )
                if _cycle_end_matches >= 4:
                    add_issue("frontend",
                        f"Error reporting flood from dashboard: {len(dashboard_errs)} query-errors within {_span:.0f}s "
                        f"({_cycle_end_matches} distinct CYCLE_END_KEYS endpoints failing). "
                        f"cycleEnd invalidates 24 keys at once; each 500 calls reportClientError, exhausting "
                        f"writeLimiter (30/min) → 429 storm + console noise. "
                        f"Fix: (1) client/src/lib/queryClient.ts — add per-endpoint 60s throttle in reportClientError "
                        f"and skip reporting for /api/client-errors itself. "
                        f"(2) server/routes.ts — replace writeLimiter on POST /api/client-errors with a dedicated "
                        f"errorReportLimiter (max: 300/min) so it is not shared with other write routes.",
                        "client-errors API")

    if crash_count > 3:
        add_issue("frontend", f"High crash rate: {crash_count} render crashes in last 50 client events", "client-errors API")
    if timeout_count > 5:
        add_issue("frontend", f"{timeout_count} query timeouts reported by browser — consider batching or spacing requests", "client-errors API")

# ═══════════════════════════════════════════════════════════════
# 7b. NETWORK METRICS TABLE + ISSUE DETECTION
# ═══════════════════════════════════════════════════════════════
# net_metrics is the last 2 minutes of frontend fetch timings.
# Table format mirrors Chrome DevTools Network tab so issues are instantly readable.
# Detection: slow endpoints, flooding (same endpoint repeated rapidly), 204s taking >5s.
if net_metrics:
    # Print table header
    print("\n[network-table] Last 2 min of browser fetch timings:")
    print(f"  {'endpoint':<35} {'status':>6}  {'dur':>8}  {'size':>8}")
    print(f"  {'-'*35} {'-'*6}  {'-'*8}  {'-'*8}")
    for _nm in net_metrics[-50:]:
        _ep   = str(_nm.get("endpoint",""))[-35:]
        _st   = _nm.get("status", 0)
        _dur  = _nm.get("durationMs", 0)
        _sz   = _nm.get("sizeBytes")
        _sz_s = f"{_sz/1024:.1f} kB" if _sz else "—"
        _dur_s = f"{_dur/1000:.2f} s" if _dur >= 1000 else f"{_dur} ms"
        print(f"  {_ep:<35} {_st:>6}  {_dur_s:>8}  {_sz_s:>8}")

    # ── Detection: slow /api/client-errors (the "16s 204" pattern) ──
    _ce_timings = [e["durationMs"] for e in net_metrics if "/client-errors" in e.get("endpoint","")]
    if _ce_timings:
        _ce_slow = [t for t in _ce_timings if t > 5000]
        if len(_ce_slow) >= 3:
            _avg_ms = int(sum(_ce_slow) / len(_ce_slow))
            add_issue("frontend",
                f"/api/client-errors is responding slowly: {len(_ce_slow)} requests averaging {_avg_ms}ms "
                f"(expected <200ms). This means the DB write for client_errors is backing up — likely "
                f"caused by pool exhaustion during the same period. The slow endpoint causes its own "
                f"backpressure loop. Root cause: fix the underlying DB pool exhaustion first (see DB section). "
                f"If pool is healthy, check storage.insertClientError() for missing index or long-running transaction.",
                "network-metrics")

    # ── Detection: flooding (same endpoint >10 times in 2 min window) ──
    from collections import Counter
    _ep_counts = Counter(e.get("endpoint","") for e in net_metrics)
    for _ep, _count in _ep_counts.most_common(5):
        if _count > 30 and "/api/network-metrics" not in _ep:
            _ep_timings = [e["durationMs"] for e in net_metrics if e.get("endpoint") == _ep]
            _ep_avg = int(sum(_ep_timings) / len(_ep_timings)) if _ep_timings else 0
            if _count > 60:
                add_issue("frontend",
                    f"Request flood: '{_ep}' called {_count} times in 2 min (avg {_ep_avg}ms). "
                    f"This is >10x the expected rate. Check if a cycleEnd WebSocket event is triggering "
                    f"redundant invalidations, or if a polling interval is misconfigured.",
                    "network-metrics")
            else:
                add_warning("frontend",
                    f"High request rate: '{_ep}' called {_count} times in 2 min (avg {_ep_avg}ms).",
                    "network-metrics")

    # ── Detection: any endpoint consistently slow (avg >8s over 3+ calls) ──
    _ep_timings_map: dict = {}
    for _e in net_metrics:
        _ep2 = _e.get("endpoint","")
        if _ep2 not in _ep_timings_map:
            _ep_timings_map[_ep2] = []
        _ep_timings_map[_ep2].append(_e.get("durationMs", 0))
    for _ep2, _times in _ep_timings_map.items():
        if len(_times) >= 3 and "/api/network-metrics" not in _ep2:
            _avg2 = sum(_times) / len(_times)
            if _avg2 > 8000:
                add_issue("frontend",
                    f"Slow endpoint: '{_ep2}' averaged {int(_avg2)}ms over {len(_times)} calls in the last 2 min. "
                    f"Investigate the route handler — check for missing DB index, N+1 queries, or missing cache. "
                    f"If this is a background compute endpoint, add caching with stale-while-revalidate.",
                    "network-metrics")
            elif _avg2 > 4000:
                add_warning("frontend",
                    f"Slow endpoint: '{_ep2}' averaged {int(_avg2)}ms over {len(_times)} calls.",
                    "network-metrics")

# ── DB connection storm detection (log-based) ─────────────────
# Pattern: many "Connection terminated" lines in recent log = event-loop block
# during XGBoost ensemble training. The pg pool terminates idle connections when
# the CPU-bound training blocks the event loop between yields (every 10 trees).
# Fixes already applied:
#   - failure-retrain reduced to 45 trees (cycle 203)
#   - trainGradientBoosting yields every 10 trees with setTimeout(r,0) (cycle 203)
#   - trainEnsembleXGB/trainVarianceEnsembleXGB yield between each model (cycle 203)
#   - idleTimeoutMillis reduced to 10s (cycle 203)
# Threshold: 20+ entries = crash-level storm (cycle 291 had 20+ and caused a crash).
#            5-19 entries = transient (server self-recovers, all errors are caught).
if log_tail:
    # Exclude lines that match db_caught_patterns — those are already caught gracefully
    # and inflate the count with harmless noise (e.g. DFT Queue refill errors).
    _conn_term_lines = []
    for _line in log_tail.split("\n"):
        if not ("connection terminated" in _line.lower() or "connection timeout" in _line.lower()):
            continue
        _is_caught = any(re.search(p, _line) for p in db_caught_patterns)
        if not _is_caught:
            _conn_term_lines.append(_line)
    _n_conn_term = len(_conn_term_lines)
    if _n_conn_term >= 20:
        add_issue("database",
            f"DB connection storm: {_n_conn_term} 'Connection terminated/timeout' entries in recent log. "
            f"Server may have crashed. Check if engine is running and restart if needed. "
            f"Root cause: XGBoost ensemble training (5 models × 150-200 trees each) blocking event loop "
            f"long enough for pg pool to terminate idle connections.",
            "server-log")
    elif _n_conn_term >= 2:
        add_warning("database",
            f"DB connection pressure: {_n_conn_term} connection-terminated entries in recent log — "
            f"event loop may have been blocked briefly by XGBoost training.",
            "server-log")

# ═══════════════════════════════════════════════════════════════
# 8. BUILD REPORT
# ═══════════════════════════════════════════════════════════════
is_clean = len(issues) == 0

report = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "serverRunning": server_running,
    "logFile": "$LOG_FILE",
    "isClean": is_clean,
    "issueCount": len(issues),
    "warningCount": len(warnings),
    "issues": issues,
    "warnings": warnings,
    "pipelineHealth": {
        "allFilteredCount": pipeline_all_filtered_count,
        "recentCycleStarts": recent_cycle_starts,
        "recentCycleEnds": recent_cycle_ends,
    },
    "pageProbes": page_probe_results,
    "inWarmup": in_warmup,
    "serverStartTime": _ss_raw or None,
    "recentClientErrors": client_errors_list[:10],
    "networkMetricsSample": net_metrics[-20:] if net_metrics else [],
    "engineStatus": engine_status,
    "dftStatus": dft_status,
    "activityFeedSample": research_logs[:20] if research_logs else [],
    "mlCalibration": ml_calibration,
}

with open(report_file, "w") as f:
    json.dump(report, f, indent=2, default=str)

# Print summary to stdout
print(f"[monitor-check] {'CLEAN' if is_clean else 'ISSUES FOUND'}")
print(f"[monitor-check] Issues: {len(issues)}  Warnings: {len(warnings)}")
if issues:
    for issue in issues:
        print(f"  [!] [{issue['category'].upper()}] {issue['message']}")
if warnings:
    for w in warnings:
        print(f"  [~] [{w['category'].upper()}] {w['message']}")
print(f"[monitor-check] Report written to {report_file}")
PYEOF

# Clean up temp files
rm -rf "$TMPDIR_QAE" 2>/dev/null || true
