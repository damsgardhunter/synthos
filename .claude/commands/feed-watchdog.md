# QAE Feed Watchdog — One Iteration

You are the **Feed Watchdog** for the Quantum Alchemy Engine. Your single job is: **keep the activity feed running 24/7**. Nothing else matters in this role.

The activity feed = new research log entries appearing at `GET /api/research-logs`. If entries are not appearing at a normal rate, the engine is not doing science. Fix it immediately.

---

## STEP 1 — Run the watchdog check

```bash
export LOG_FILE=logs/server-latest.log && bash scripts/feed-watchdog-check.sh
```

Then read `logs/feed-watchdog-report.json`.

---

## STEP 2 — Act on the report

Read the `status` and `action` fields. Follow the relevant section below.

---

### status: `healthy`

Feed is running normally. Output:
```
✓ Feed healthy — [feedAgeSeconds]s since last entry ("[lastLogEvent]")
```
Done. No further action needed.

---

### status: `feed_slow` (5-10 min gap, warning only)

Feed is slowing but not frozen yet. This is a pre-freeze warning.

1. Check what's in the log right now:
```bash
tail -50 logs/server-latest.log | grep -E "\[GradientBoost\]|\[Engine\]|\[XGBoost\]|trainEnsemble|training|cool-down"
```
2. If GB training lines appear → go to **FREE INVESTIGATION: GB Training** section below
3. If nothing suspicious → log the warning and continue watching

---

### status: `event_loop_blocked` — action: `free_investigation_gb_training`

**CRITICAL. The event loop is completely blocked.** The server cannot process any requests.

Go to **FREE INVESTIGATION: GB Training** below immediately.

---

### status: `event_loop_blocked` — action: `free_investigation_slow_endpoint`

**CRITICAL. An endpoint took 2+ minutes to respond.** The event loop is severely degraded.

Go to **FREE INVESTIGATION: Slow Endpoint (>2 min response)** below immediately.

---

### status: `event_loop_blocked` — action: `free_investigation_event_loop`

**CRITICAL. Event loop blocked but cause is unknown.**

1. Read the server log tail:
```bash
tail -100 logs/server-latest.log
```
2. Find the last line before the block (timestamp where log goes silent)
3. Identify the function running at that point
4. Read the relevant source file and find the CPU-intensive operation
5. Fix it (add yields, offload to worker, or dispatch to GCP)
6. Restart server after fix

---

### status: `feed_frozen` — action: `restart_engine_then_investigate`

Feed has been silent ≥10 min.

1. Try engine restart first:
```bash
curl -s -X POST http://localhost:4000/api/engine/stop
sleep 5
curl -s -X POST http://localhost:4000/api/engine/start
```
2. Wait 90 seconds, then re-run `bash scripts/feed-watchdog-check.sh`
3. If still frozen → full server restart: `bash scripts/start-server-logged.sh`
4. After restart → go to **FREE INVESTIGATION** to find root cause in code
5. **Do NOT mark this as fixed without a code change.** A restart without fixing the cause = same freeze in 30-60 min.

---

### status: `feed_frozen` — action: `free_investigation_gb_training`

Go to **FREE INVESTIGATION: GB Training** below.

---

### status: `server_down` — action: `restart_server`

```bash
bash scripts/start-server-logged.sh
```
Wait 90 seconds. Re-run check. If still down, read the log for the crash reason.

---

### status: `feed_empty` — action: `start_engine`

```bash
curl -s -X POST http://localhost:4000/api/engine/start
```

---

## FREE INVESTIGATION: GB Training

**This is the most common cause of feed freezes.** Local GB training (trainGradientBoosting, trainEnsembleXGB, trainVarianceEnsembleXGB) blocks the Node.js event loop for 2-30+ minutes. With `OFFLOAD_XGB_TO_GCP=true`, ALL training must dispatch to GCP — no local tree building.

### Step 1 — Find every unguarded local training call

```bash
grep -n "trainGradientBoosting\|trainEnsembleXGB\|trainVarianceEnsemble" server/learning/gradient-boost.ts | grep -v "async function\|export function\|//"
```

For each line found, check if it's inside an `if (process.env.OFFLOAD_XGB_TO_GCP !== "true")` block. If not — that's the bug.

### Step 2 — Read the function containing each unguarded call

Use the Read tool to read the surrounding 30 lines of each hit. Understand:
- Which exported function contains it
- What data it has (`X`, `y`, `formulas`)
- What the function returns

### Step 3 — Add the GCP dispatch guard

For each unguarded call, add:
```typescript
if (process.env.OFFLOAD_XGB_TO_GCP === "true") {
  if (X.length >= 30) {
    dispatchXGBJobToGCP(X, y, formulas).catch(err =>
      console.warn("[GB] GCP dispatch failed:", err.message)
    );
  }
  // Return appropriate value for this function (null, 0, FALLBACK_MODEL, etc.)
  return;
}
// existing local training code below...
```

### Step 4 — Also check these specific functions (most likely culprits)

- `retrainWithAccumulatedData` / `_retrainWithAccumulatedDataInner`
- `incorporateFailureData`
- `getTrainedModel`
- `backfillPool`
- Any `setTimeout(async () => { ... trainEnsembleXGB ... })` deferred blocks

### Step 5 — Restart server after fix

The tsx watch server hot-reloads TS changes automatically. But if the server is currently frozen (event loop blocked), it cannot hot-reload — force-kill and restart:
```bash
bash scripts/start-server-logged.sh
```

### Step 6 — Verify fix

Wait 3 minutes, re-run `bash scripts/feed-watchdog-check.sh`. Feed should be healthy.

---

### status: `feed_errors` — act on `action` field

The feed is flowing but errors appear in the last 50 entries. Read the `action` field from the report:

- `free_investigation_null_guard` → go to **FREE INVESTIGATION: Errors in Feed → null_guard**
- `free_investigation_code_bug` → go to **FREE INVESTIGATION: Errors in Feed → code_bug**
- `free_investigation_timeout` → go to **FREE INVESTIGATION: Errors in Feed → timeout**
- `free_investigation_db` → go to **FREE INVESTIGATION: Errors in Feed → db**
- `free_investigation_openai_circuit` → go to **FREE INVESTIGATION: Errors in Feed → openai_circuit**

---

## FREE INVESTIGATION: Slow Endpoint (>2 min response)

When `netMaxMs >= 120000` (2+ minutes) — even if `gbInLog=false`:

1. Identify the endpoint from `netMaxEndpoint` in the report.
2. Check the log for what was running when the stall occurred:
```bash
tail -100 logs/server-latest.log
```
3. Find the route handler in `server/routes.ts`.
4. Determine what makes it slow and apply the matching fix:
   - **Sync CPU loop in route handler**: Add `await new Promise(r => setTimeout(r, 0))` yields every 10-50 iterations, or offload to GCP/worker thread
   - **DB query hanging**: Check pool exhaustion — reduce concurrent phases via semaphore in `server/learning/engine.ts`; or add query timeout in `server/db.ts`
   - **External API hung**: Add timeout wrapper: `await Promise.race([apiCall(), new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 8000))])`
   - **Large in-memory computation on every request**: Cache the result, recompute only when data changes
5. Restart server after fix. Re-run check — `netMaxMs` should drop below 30s.

---

## FREE INVESTIGATION: Errors in Feed

### action: `free_investigation_null_guard`

**A null/undefined reference error is crashing something.**

1. Get the exact error from `detectedErrors[0]` in the report. Note the `event`, `detail`, and `phase`.
2. Find where the error originates:
```bash
grep -n "Cannot read properties\|is not a function\|featureMask\|TypeError" logs/server-latest.log | tail -20
```
3. The detail should contain the property name, e.g. `(reading 'featureMask')`. Search for that property in the relevant phase's source files:
```bash
grep -rn "\.featureMask\|\.propertyName" server/learning/ server/physics/
```
4. Find the function that accesses the null property. Read 20 lines around it.
5. The fix is always a null guard. Add it at the earliest possible point:
   ```typescript
   // Before: crashes when model is null
   const result = model.predict(x);
   
   // After: guard against null
   if (!model) return fallbackValue;
   const result = model.predict(x);
   ```
6. If the caller should never pass null (but does), also fix the caller.

---

### action: `free_investigation_code_bug`

**A reference error or other code bug is in the feed.**

1. Get the exact error from `detectedErrors[0]`. Note the `event` and `detail`.
2. Search the log for the full error context:
```bash
grep -B2 -A5 "error" logs/server-latest.log | grep -A5 "your specific error phrase"
```
3. Identify the file and function. Read the relevant section.
4. Fix the bug. Common patterns:
   - `ReferenceError: X is not defined` → missing import or typo in variable name
   - `TypeError: X is not a function` → method doesn't exist on the object (check type definition)
   - Anything with array/object access → add `?` optional chaining or guard
5. Restart and verify.

---

### action: `free_investigation_timeout`

**A timeout is repeatedly hitting the feed.**

1. Check which phase has the timeout from `detectedErrors[0].phase`.
2. Read the last 100 log lines to see what was running before the timeout:
```bash
tail -100 logs/server-latest.log
```
3. Common timeout causes and fixes:
   - **OpenAI timeout**: Check if `openaiCircuitOpen()` is called before the API call in the relevant file. If not, add the guard. If the circuit is already supposed to be open, check `server/learning/openai-circuit.ts`.
   - **DB query timeout**: Increase `statement_timeout` in `server/db.ts`, or identify which query is slow (check `logs/health-report.json` for slow query info)
   - **External HTTP timeout**: Add `--max-time` to curl calls, or add `signal: AbortSignal.timeout(8000)` to fetch calls
   - **xTB computation timeout**: Check `server/learning/active-learning.ts` for the xTB timeout constant — increase if legitimate compute is getting cut off

---

### action: `free_investigation_db`

**DB connection errors are appearing in the feed.**

1. Check what the exact error is from `detectedErrors[0].detail`.
2. Read `server/db.ts` to understand the pool config.
3. Common DB errors and fixes:
   - **connection terminated / ETIMEDOUT**: Pool connections are dying from inactivity. Add `keepAlive: true` and `keepAliveInitialDelayMillis: 10000` to the pool config.
   - **pool exhausted / too many clients**: Concurrent phases are each holding connections. Verify the 2-permit semaphore exists in `server/learning/engine.ts`. If it's missing, add it.
   - **query timeout**: Raise `statement_timeout` in `server/db.ts` — default 15s is too low for complex queries during high-traffic periods. Use 30s.

---

### action: `free_investigation_openai_circuit`

**OpenAI requests are timing out repeatedly.**

1. Check `server/learning/openai-circuit.ts` to see the current circuit state (fail count, backoff until).
2. Identify which modules are calling OpenAI without the shared circuit guard:
```bash
grep -rn "openai.chat.completions\|openai\.embeddings" server/learning/ | grep -v "openaiCircuitOpen"
```
3. Any call not preceded by `if (openaiCircuitOpen()) return ...` needs the guard added.
4. Check the `FAIL_THRESHOLD` and `BACKOFF_MS` in `openai-circuit.ts`. If timeouts still happen after backoff resets, increase `BACKOFF_MS` to 30 minutes.

---

## FREE INVESTIGATION: General Event Loop Block

When `gbInLog=false` but event loop is blocked:

1. Read the last 200 lines of the log:
```bash
tail -200 logs/server-latest.log
```

2. Find the timestamp where log output stopped. What was the last function logged?

3. Common non-GB causes:
   - **SG sweep with 0 passing** → read `server/learning/engine.ts` SG loop, add yield every candidate
   - **Pareto O(n²) sweep** → check `server/inverse/pareto-optimizer.ts`, yield every 25 iterations, cap at 200 candidates
   - **Active Learning** → normal if <20 min, flag if >30 min with no completion log
   - **backfillGBScores** → reads all candidates, logs only at start/end; add progress emit every 10 batches
   - **Phase 11 structure prediction** → can take 5-10 min; check for missing yields in batch loop

4. Read the relevant file, find the blocking loop, add `await new Promise(r => setTimeout(r, 0))` yields every 10-50 iterations.

---

## STEP 3 — Report

After every run, output exactly this:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feed Watchdog — [timestamp]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:   [status]
Feed age: [feedAgeSeconds]s
Last entry: "[lastLogEvent]"
Event loop: [netMaxMs]ms worst response ([netMaxEndpoint])
GB training in log: [gbInLog]

Action taken: [what you did, or "none — feed healthy"]
Code fix applied: [yes/no — describe if yes]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## RULES

- **Never mark a freeze as fixed by restart alone.** A restart without a code fix = same freeze in 30-60 min. The fix must change code.
- **Never add `await new Promise(r => setTimeout(r, 100))` delays** as the "fix" — this just slows the loop without fixing blocking. Use real yields inside tight loops or offload to GCP/worker.
- **The feed must produce new entries at least every 5 minutes** during normal engine operation. If it can't, find out why and fix it.
- **Do NOT run the full qae-monitor check** — that's a separate agent. This watchdog only cares about the feed.
