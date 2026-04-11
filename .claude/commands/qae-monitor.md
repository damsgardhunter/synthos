# QAE Autonomous Monitor — One Iteration

You are running as the autonomous monitor for the **Quantum Alchemy Engine (QAE)**. This command runs every 5 minutes. Your job each cycle is:

1. Ensure the server is running
2. Run the health check
3. Analyze findings
4. Fix real issues in the code
5. Update state and report

Work through each step below in order. Be surgical — only fix what's actually broken.

---

## STEP 1 — Ensure Server Is Running

Run:
```bash
bash scripts/start-server-logged.sh
```

If it exits with an error (server died and won't restart), add an issue to the report about it but continue with the rest of the analysis using whatever data is available from previous logs.

---

## STEP 2 — Run Health Check

Run:
```bash
export LOG_FILE=logs/server-latest.log && bash scripts/qae-monitor-check.sh
```

This writes `logs/health-report.json`. Read that file, then read `logs/monitor-state.json`.

---

## STEP 3 — Read Current State

Read both files:
- `logs/health-report.json` — what was found this cycle
- `logs/monitor-state.json` — running clean cycle count and fix history

---

## STEP 4 — Check Stopping Condition

If `monitor-state.json` shows `cleanCycles >= 50`, output this exact message and stop:

```
╔══════════════════════════════════════════════════════════╗
║  MONITORING COMPLETE                                     ║
║  50 consecutive clean cycles with zero issues detected.  ║
║  All systems healthy. You can stop the loop now (/stop). ║
╚══════════════════════════════════════════════════════════╝
```

Do not make any further changes. Update `lastCheck` timestamp in state and exit.

---

## STEP 5 — Analyze Issues and Apply Fixes

For each issue in `health-report.json`, decide on the **best fix** and apply it. Guidelines below.

### ⚠️ PERSISTENT ISSUE RULE — Free Investigation Mode

**Before looking at any fix table**: scan `fixHistory` in `monitor-state.json` for the current issue pattern.

If the **same issue** (same category + same root-cause keywords) appears **2 or more times** in `fixHistory` and the fixes listed were only server/engine restarts (no code file edits), then:

1. **Do NOT apply the table fix again.** Restarting a server is not a code fix. The issue will recur.
2. **Investigate the actual root cause yourself** using the full tool suite:
   - Read the relevant source files (`Read`, `Grep`, `Glob`)
   - Trace the call chain from the error message to the code that produces it
   - Find the structural cause (not the symptom) and fix it
3. **Only restart AFTER a code fix is applied.** A restart without a code change guarantees the same issue in another 30–60 minutes.

**How to identify "same issue":** look for matching keywords in fixHistory entries — e.g. "pool exhaustion", "updatePhaseStatus failed", "research log circuit breaker", "feed FROZEN" all pointing to the same DB storm pattern.

**Example of what NOT to do:** if fixHistory shows 2 entries both saying "Force-killed PID X → PID Y, feed confirmed active" for the same "DB pool exhaustion" root cause — do not kill the server again. That is the 3rd restart, not a fix.

**When to use Free Investigation Mode:**
- "DB pool exhaustion" recurring every 40–65 min → investigate which phases run concurrently and cap their DB access with a semaphore or sequential execution
- "Feed frozen" recurring with no code change → read engine.ts and find what's blocking the event loop
- "Research log circuit breaker opened" recurring → find what's generating the burst of DB writes and rate-limit it

### ⛔ HOLDING LIMIT — Never Get Stuck in Observe-Only Mode

Free Investigation Mode is **not** an excuse to write "Holding. Peak unchanged." for 40 cycles. If you have looked at an issue and decided not to act, you must commit to a decision **within 3 monitor cycles**. Indefinite observation = a stuck monitor.

**Banned phrases in `openIssues`:** "Holding", "Peak unchanged", "Nth cycle of same issue", "monitoring", "watching", "stable so far". If you catch yourself about to write any of these, you have already failed the holding limit and must take one of the actions below **this cycle**.

**On the 3rd consecutive cycle that the same issue is in `openIssues` with no fix applied, you must do EXACTLY ONE of these — not "wait and see":**

1. **Investigate and fix the actual root cause.** Read the source files involved, trace the call chain, find the bug, edit the code. If the prior fix was wrong (the symptom got worse or didn't move), revert it as part of the same edit. The fix history is allowed to say "reverted X — wrong hypothesis."

2. **Accept the issue as a known warning** by moving it from `openIssues` into a new top-level `acceptedWarnings` array in `monitor-state.json`:
   ```json
   "acceptedWarnings": [
     { "pattern": "<short stable identifier>",
       "since": "<ISO timestamp>",
       "reason": "<why this is fine to ignore + what would change to re-open it>" }
   ]
   ```
   An accepted warning **does not block clean cycles**. It is still surfaced in the cycle report, but the cycle counts as clean if no real issues are present. Use this when you have determined the warning is either a false positive, a measurement artifact, or a known cost that does not affect the science.

3. **Tighten the warning detector in `scripts/qae-monitor-check.sh`** so the warning only fires on the genuinely-broken case. E.g. if `extractFeatures TOTAL >10s` is mostly measuring semaphore queue wait, raise the threshold to >60s, or split it into a wait-time vs cpu-time metric. Whatever you change must be a one-line edit in the check script with a comment citing the cycle number and reason.

4. **Escalate to the user** by writing a single concise issue in `openIssues` titled `NEEDS USER DECISION: <one-line summary>` with: (a) what you investigated, (b) why none of options 1–3 are appropriate, (c) the specific question you need answered. After this, you may keep the issue across cycles **without it counting as a holding violation**, but you must not add any new analysis until the user responds — any new "still observing" cycles will reset the violation counter.

**How to count holding cycles:** scan the last 3 entries of `fixHistory` plus the current `openIssues`. If the same issue (same stable keywords) has appeared in 3 cycles in a row without an entry in `fixHistory` for it, you are at the limit. Picking option 1, 2, or 3 above counts as a fix and should be added to `fixHistory` like any other fix.

**Anti-cheat:** an entry in `fixHistory` that says "decided to keep monitoring" or "no action this cycle" does NOT reset the holding counter. Only options 1–4 above reset it.

**Why this rule exists:** the monitor previously sat for 41 cycles writing "Peak unchanged. Holding." against an `extractFeatures TOTAL >10s` warning where the prior fix (cycle 1367 semaphore) had partially worked but the warning's measurement was conflating queue wait with CPU time. Forty-one observe-only cycles is forty-one missed opportunities to either fix the real issue, fix the measurement, or accept it. None of those happened because no rule forced a decision.

---

### Repeated OOM Crashes (CRITICAL — fix code, do NOT just restart again)

When `monitor-state.json` shows **3 or more OOM restarts** in recent `fixHistory`, this is a confirmed memory leak. Restarting the server is a temporary band-aid — the crash will recur in another 15–35 minutes. **You must investigate and fix the root cause before moving on.**

**Step 1 — Confirm memory growth pattern.** Check the log for `heap=` readings logged at each cycle start:
```bash
grep "Cycle #.*heap=" logs/server-latest.log | tail -20
```
If heap is growing each cycle (e.g. 400MB → 600MB → 900MB), you have a leak.

**Step 2 — Check `TrainingPool` size cap.** The most common cause is `TrainingPool.featureCache` in `server/learning/gradient-boost.ts` growing without bound:
```bash
grep -n "MAX_SIZE\|private static.*MAX\|featureCache.size" server/learning/gradient-boost.ts
```
The class must have `private static readonly MAX_SIZE = 6000` and the `add()` method must evict the oldest entry when at capacity. If the cap is missing, add it:
```typescript
// Inside TrainingPool.add(), before storing the new entry:
if (this.featureCache.size >= TrainingPool.MAX_SIZE) {
  const firstKey = this.featureCache.keys().next().value;
  if (firstKey !== undefined) this.featureCache.delete(firstKey);
  this.dirty = true;
}
```

**Step 3 — Check `--max-old-space-size` flag.** Node.js defaults to a 1.5 GB heap limit, which can be exhausted quickly. The `package.json` dev script must include this flag:
```bash
grep "max-old-space\|NODE_OPTIONS" package.json
```
If absent, add `NODE_OPTIONS=--max-old-space-size=3072` to the dev script:
```json
"dev": "cross-env NODE_ENV=development NODE_OPTIONS=--max-old-space-size=3072 tsx --env-file=.env server/index.ts"
```

**Step 4 — Check other module-level unbounded Maps.** These in `server/learning/engine.ts` should all have size caps — verify they do:
- `dftEnrichmentTracker` — should be pruned when `size > 5000` (line ~8485)
- `stagnationReanalyzedIds` — pruned when entries are old (line ~2371)
- `alreadyScreenedFormulas` — capped at `MAX_SCREENED_CACHE_SIZE = 50000`
- `rejectedFormulas` — capped at `MAX_REJECTED_CACHE_SIZE = 10000`

If any are missing their pruning, add FIFO eviction (delete the first key in iteration order).

**Step 5 — Only restart the server AFTER applying fixes.** A restart without a code fix guarantees another OOM within 20–30 minutes.

---

### EVENT LOOP BLOCKED (most critical — fix before anything else)

When the network-metrics table shows any endpoint taking **>5 minutes**, the Node.js event loop is completely blocked. No requests are processed, no research logs are written, no DB keepalives fire — this is the direct cause of feed freezes AND DB connection storms.

**The most common cause is local GB training.** Even with `OFFLOAD_XGB_TO_GCP=true`, some training paths may still run locally.

**Step 1 — Identify whether GB training is the cause:**
```bash
grep -E "trainGradientBoosting|trainEnsembleXGB|trainVarianceEnsemble|Post-retrain cool-down|GradientBoost.*training" logs/server-latest.log | tail -10
```
If you see these lines around the time the block started → GB training is running locally.

**Step 2 — Find and fix every local training call. FREE INVESTIGATION MODE:**
```bash
grep -n "trainGradientBoosting\|trainEnsembleXGB\|trainVarianceEnsemble" server/learning/gradient-boost.ts | grep -v "async function\|export\|//"
```
For every call found, check if it's guarded by `OFFLOAD_XGB_TO_GCP`. If not, add the guard:
```typescript
if (process.env.OFFLOAD_XGB_TO_GCP === "true") {
  dispatchXGBJobToGCP(X, y, formulas).catch(err => console.warn("[GB] GCP dispatch failed:", err.message));
  return; // or return appropriate value
}
```
Key functions to check: `retrainWithAccumulatedData`, `incorporateFailureData`, `getTrainedModel`, `backfillPool`, and any `setTimeout` deferred training.

**Step 3 — Restart server AFTER applying the code fix:**
```bash
bash scripts/start-server-logged.sh
```
A restart without fixing the code just delays the next block by the same training interval.

---

### Frozen Activity Feed

This is a high-priority issue. When the feed freezes, the engine is silently stalled — no science is being done.

**ESCALATION ORDER — follow these steps in sequence, do not skip to server restart:**

**Step 1 — Read the log to diagnose root cause BEFORE touching anything.**
```bash
tail -200 logs/server-latest.log
```
Look for what was happening at/after the last feed timestamp:
- **SG sweep at 0 passed / stalled**: last feed entry is `SG sweep progress: N/M screened, 0 passed` and nothing after → SG sweep is stuck. Engine restart is sufficient.
- **`TypeError` / `ReferenceError`** after the last cycle → engine loop crashed. Engine restart is sufficient.
- **`[Engine]` lines stopping mid-cycle** (e.g. `Cycle #N START` with no `END`) → engine stalled. Engine restart is sufficient.
- **DB write errors** (`connection terminated`, `ETIMEDOUT`, pool exhausted, many `[research-logs] ERROR:` lines) → DB storm. Engine restart first; if errors persist, restart server.
- **Pareto recompute blocking** (`[Pareto] Scheduled recompute` with no completion, or `/api/pareto-frontier` taking >5s repeatedly) → the O(n²) domination sweep is blocking the event loop. Root cause: `scheduleParetoRecompute` was loading 500 candidates; `nonDominatedSort` does O(n²)=125k comparisons and yields only every 100 iterations = 10-120s blocking between yields. Fix in `server/inverse/pareto-optimizer.ts`: (1) `MAX_PARETO_CANDIDATES = 200` cap in `scheduleParetoRecompute` before calling `recomputeParetoFrontier`. (2) Yield every 25 outer iterations in the O(n²) sweep instead of 100. Verify these are present — if not, apply them. Engine restart clears current blockage.
- **No log activity at all after freeze** → server process may be fully hung. Server restart needed.

**Step 2 — Try engine stop/start first (less disruptive, preserves DB state):**
```bash
curl -s -X POST http://localhost:4000/api/engine/stop
sleep 5
curl -s -X POST http://localhost:4000/api/engine/start
```
Wait 60 seconds, then check whether the feed has new entries:
```bash
curl -s "http://localhost:4000/api/research-logs?limit=3"
```
If new entries appear → **engine restart was sufficient. Stop here.** Log what caused the freeze.

**Step 3 — Only if engine restart did NOT restore the feed: full server restart.**
```bash
bash scripts/start-server-logged.sh
```
Wait 90 seconds then re-check the feed.

**Step 4 — Fix the root cause in code. Do not just restart and move on.**

| Root cause from log | Code fix |
|---------------------|----------|
| **GB training blocking event loop** (`[GradientBoost] Post-retrain cool-down` or `trainEnsembleXGB` in log around freeze time) | **FREE INVESTIGATION MODE.** Read `server/learning/gradient-boost.ts`. Find every call to `trainGradientBoosting`, `trainEnsembleXGB`, `trainVarianceEnsembleXGB` that is NOT guarded by `if (process.env.OFFLOAD_XGB_TO_GCP === "true")`. Add the guard + `dispatchXGBJobToGCP(X, y, formulas)` dispatch to each. Key functions: `retrainWithAccumulatedData`, `incorporateFailureData`, `getTrainedModel`, deferred `setTimeout` training blocks. After fixing, restart server. |
| **Feed slowing** (gaps increasing from 2min → 13min → 25min) | Same cause as above — event loop is being consumed by something CPU-intensive. Check logs for `[GradientBoost]`, `[XGBoost]`, `backfillGBScores`, `SG sweep` around the slow periods. Identify which one is running and apply the appropriate fix. |
| SG sweep stuck 0/N with no progress for >15 min | Check `server/learning/engine.ts` SG sweep loop for missing yield or deadlock with `isPhysicsAnalysisActive`. The sweep must yield every candidate and respect the pause flags. |
| `TypeError` / null crash in engine loop | Find the stack trace, add null guard at the throw point in `server/learning/engine.ts` |
| DB connection storm (>20 `connection terminated`) | See Database / Connection Issues section — check pool size and retry logic in `server/db.ts` |
| Pareto recompute blocking event loop | Route must slice results before `res.json()`: `data.results.sort((a,b)=>a.rank-b.rank).slice(0,200)`. Already fixed — confirm fix is present in `server/routes.ts`. |
| No identifiable cause in logs | Add more `[Engine]` logging around the suspect phase so next freeze is diagnosable |

### Database / Connection Issues

| Issue | Fix Location | What to Do |
|-------|-------------|------------|
| ETIMEDOUT / connection terminated | `server/db.ts` | Increase `connectionTimeoutMillis`, add/extend exponential backoff in retry logic |
| Pool exhausted / max clients — **first occurrence** | `server/db.ts` | Increase `connectionTimeoutMillis` (current: 8s), raise `max` if below 20 |
| Pool exhausted / max clients — **recurring (2+ times)** | `server/learning/engine.ts` | **Do not touch `db.ts` again.** Root cause is too many engine phases running concurrently. The batch1 `Promise.allSettled([runPhase4, runPhase8, runPhase9])` fires 3 phases simultaneously, each making multiple DB queries. Fix: verify `phaseSemaphore` (Semaphore with 2 permits) is present near the top of `engine.ts` and that `batch1` and `batch2` use `withPhaseSemaphore(runPhaseN)` instead of calling phases directly. If missing, add a `class Semaphore` with `acquire()/release()`, create `const phaseSemaphore = new Semaphore(2)`, wrap each phase in `batch1`/`batch2` with `withPhaseSemaphore(runPhaseN)`. This limits concurrent DB-using phases to 2 at a time. |
| Query timeout | `server/db.ts` | If query timeout is < 15s, raise it; also look for N+1 query patterns in `server/routes.ts` or `server/storage.ts` |
| Feature extraction too slow | `server/learning/ml-predictor.ts`, `gnn/server.py` | Add timeout guard, cache repeated feature calls, defer non-critical features |

### Frontend / HTTP Issues

| Issue | Fix Location | What to Do |
|-------|-------------|------------|
| HTTP 500 on route | `server/routes.ts` | Find the route handler, add try/catch with proper error response, check for missing null guards |
| HTTP 400 on route | `server/routes.ts` | Check schema validation, ensure Zod schema matches what frontend sends |
| Route > 5s response | `server/routes.ts`, `server/cache.ts` | Add caching for the slow route, break up expensive synchronous work, use `stale-while-revalidate` pattern |
| Unhandled promise rejection | Grep for the stack trace in `logs/server-latest.log`, then fix the async function that threw |
| TypeError / null access | Find the stack trace, add null guard at the point of failure |
| **Error reporting flood / 429 storm on `/api/client-errors`** | `client/src/lib/queryClient.ts` + `server/routes.ts` | cycleEnd invalidates 24 `CYCLE_END_KEYS` at once; each failing query calls `reportClientError`; 24+ POSTs/cycle exhaust the shared `writeLimiter` (30/min) → 429 console spam. **Two fixes required**: (1) In `reportClientError()` in `queryClient.ts`, add a per-endpoint `Map<string, number>` throttle — skip reporting the same endpoint+statusCode within 60s, AND add `if (payload.endpoint === '/api/client-errors') return;` as the first line to prevent self-referential loops. (2) In `routes.ts`, create a dedicated `errorReportLimiter` with `max: 300` (not shared with `writeLimiter`) and use it on `app.post("/api/client-errors", errorReportLimiter, ...)`. |
| **Error reporter feedback loop** | `client/src/lib/queryClient.ts` | `reportClientError` is itself triggering further error reports (i.e. the `/api/client-errors` endpoint is in the error list). Add `if (payload.endpoint === '/api/client-errors') return;` as the very first guard in `reportClientError()`. |
| **`/api/client-errors` slow (>5s, 204 responses)** — detected via network-metrics table | Root cause is DB pool exhaustion, not the route itself. Fix the pool issue first (see DB section). The route already responds 204 immediately and writes async — if it's still slow, the async write is blocking on a full pool. Check `server/routes.ts` to confirm the `res.sendStatus(204)` fires before `storage.insertClientError()`. |
| **Request flood (same endpoint >20 calls in 2 min)** — detected via network-metrics table | Read `client/src/App.tsx` and `client/src/pages/dashboard.tsx` to find what triggers the repeated calls. Common causes: (1) `cycleEnd` WebSocket event invalidating too many keys — reduce `CYCLE_END_KEYS`; (2) `refetchInterval` set too low on a query — remove it or raise to ≥60s; (3) component re-mounting on each navigation — add `staleTime: Infinity`. |
| **Slow endpoint avg >8s (3+ calls)** — detected via network-metrics table | **This is a free-investigation trigger.** Read the route handler in `server/routes.ts`, trace what DB queries it runs, check for N+1 patterns or missing cache. Apply caching via `cache.getOrSet(key, ttlMs, fetchFn)` or add a DB index for the query's WHERE clause. |

### Physics / Pipeline Issues

| Issue | Fix Location | What to Do |
|-------|-------------|------------|
| Imaginary phonon frequencies | `server/learning/multi-fidelity-pipeline.ts` Stage 2 | Tighten the `dynamicallyUnstable` check to reject earlier, or flag as warning instead of letting it crash |
| Phonon calculation failure | `server/learning/physics-engine.ts` or `server/dft/qe-worker.ts` | Add error boundary / fallback return, log the failure with the formula for debugging |
| Band structure failure | `server/dft/band-structure-calculator.ts` | Add try/catch, return null and let pipeline handle gracefully |
| Pipeline filtering ALL candidates (0 passed all) repeatedly | `server/learning/multi-fidelity-pipeline.ts` Stage 0 | Check if ML score threshold is misconfigured, inspect whether the dataset has drift, log warning. Do NOT lower threshold blindly — check if the filter itself has a bug (e.g., always returns 0 due to undefined field) |
| NaN Tc prediction | `server/learning/multi-fidelity-pipeline.ts` Stage 3 | Add NaN guard before Tc is stored — `if (!isFinite(tc)) skip or flag` |
| Unphysical Tc at ambient | `server/learning/multi-fidelity-pipeline.ts` Stage 3 | Add sanity clamp: ambient Tc > 400K should require pressure > 50 GPa evidence |
| Kinetic lifetime zero / unstable | `server/physics/kinetic-stability.ts` | Check if computation is returning 0 due to division error, add guard |
| Eliashberg solver failure | `server/learning/physics-engine.ts` | Add try/catch, fall back to Allen-Dynes formula, log which compound failed |
| Formation energy anomaly | `server/learning/multi-fidelity-pipeline.ts` Stage 4 | Already has -5.0 guard — check if it's being evaluated correctly |
| **Empty phonon frequencies from QE** (`phonon.frequencies: []` on completed job) | `server/dft/qe-worker.ts` `parsePhononOutput()` and `server/dft/dfpt-parser.ts` | QE 7.x writes degenerate modes as `omega(1-3)` (range notation). The regex must include `(?:\s*-\s*\d+)?` inside the parentheses group so both `omega(4)` and `omega(1-3)` are matched. Check both the primary pattern and the single-unit fallback. |
| **Surrogate Tc identical for many compounds** (all showing same value like 25 K regardless of λ) | `server/learning/engine.ts` — `autoPhysExplicitlyZero` block | The GB model returns its training-mean (~25 K) for out-of-distribution compounds. When physics computes Tc≈0 K (Allen-Dynes with small λ), `physicsTc=0` is excluded from `reconcileTc()` leaving only GB's estimate. The `autoPhysExplicitlyZero` guard must be checked BEFORE `reconciledAuto.reconciledTc > 0`, not after, so a physics veto is not overridden by the ML mean. |
| **Phonon-unstable candidates queued for DFT** (`phonon_stable=false` in mlFeatures but still in DFT queue) | `server/dft/dft-job-queue.ts` `preFiltered` array filter | Add a check: `if (mlFeatures.phononStable === false \|\| mlFeatures.qePhononStable === false) return false;` inside the `preFiltered` `.filter()` callback (after the atom count gate). |
| **H-rich hydride wrongfully rejected** (Physics Merit=0.00 on LaH7/LaH10-family compounds despite lifshitz≥0.8 or QCP=true) | `server/learning/deliberative-evaluator.ts` — `runPhysicsMerit()` d-band block | La (and similar transition metals) has `D_ELECTRONS=0` in the d-band table, triggering a −0.20 d-band penalty that wipes out positive SC signals. For H-rich hydrides (H:metal > 3), SC is driven by H-sublattice phonons — d-band filling is not the relevant mechanism. The `isHRichHydride` guard must wrap the TM d-band penalty block: `if (tmCount > 0 && !isHRichHydride) { ...d-band check... }` |
| **Hull distance penalizes high-pressure candidates** (score drops from ~0.21 to ~0.06 at 170 GPa due to ambient hull penalty) | `server/learning/deliberative-evaluator.ts` — `runRiskAssessment()` hull block AND `server/learning/engine.ts` finalPayload | The ambient convex hull distance is physically irrelevant for compounds at >50 GPa. Two fixes required: (1) In `engine.ts`, the `mlFeatures` spread in `finalPayload` must include `pressureGpa: enforcedPressure`. (2) In `runRiskAssessment`, read `riskPressureGpa = mlFeatures?.pressureGpa ?? 0` and skip the hull penalty (`score -= 0.1`) when `riskPressureGpa > 50`, logging a neutral note instead. Same logic applies to the H-count penalty (H≥6): skip the −0.1 when pressureGpa > 20 since the pressure requirement is already accounted for. |
| **Stoner parameter misinterpreted as ferromagnet suppressor** (Stoner=3.56 + QCP=true → ferromagnetic SC-suppression penalty applied) | `server/learning/deliberative-evaluator.ts` — `runPhysicsMerit()` Stoner block | Stoner > 1.0 WITH a confirmed QCP means spin-fluctuation SC near a magnetic quantum critical point — this is a positive indicator, not a suppressor. Only apply the ferromagnetic penalty when `stonerRatio > 3.0 AND qcpScore < 0.3`. Pattern: `if (stonerRatio > 3.0 && qcpScore < 0.3) { penalty } else if (stonerRatio > 1.0 && qcpScore >= 0.3) { reasoning note only }` |

### ML Issues

| Issue | Fix Location | What to Do |
|-------|-------------|------------|
| ML model MAE > 30K | `server/learning/ml-predictor.ts` | Log a warning to research_logs, do not auto-retrain (training is expensive) |
| XGBoost / GNN timeout | `server/learning/ml-predictor.ts`, `gcp-worker/*.ts` | Increase timeout constant, add circuit breaker to skip ML and use physics-only fallback |

### OpenAI API Issues

OpenAI timeouts and rate limits are **errors**, not warnings. When spotted, apply one of these fixes depending on the call site. Read the file before editing.

**Strategy 1 — Space out calls with a delay between them**
Use this when multiple calls fire in rapid succession from the same loop (e.g., `data-fetcher.ts` calling OpenAI for every material in a batch):
```typescript
// Before: calls fire immediately back-to-back
for (const item of batch) {
  await openai.chat.completions.create({ ... });
}

// After: 1.5s gap between calls
for (const item of batch) {
  await openai.chat.completions.create({ ... });
  await new Promise(r => setTimeout(r, 1500));
}
```

**Strategy 2 — Combine multiple small queries into one batched prompt**
Use this when the same file makes 2–4 separate calls that ask about similar things (e.g., asking for material properties one-by-one). Merge them into a single prompt asking for all at once:
```typescript
// Before: 3 separate calls for 3 materials
for (const formula of [f1, f2, f3]) {
  const r = await openai.chat.completions.create({ messages: [...single material prompt...] });
}

// After: one call, ask for all 3 in one prompt
const r = await openai.chat.completions.create({
  messages: [{ role: "user", content: `Provide data for these 3 materials: ${[f1,f2,f3].join(", ")}...` }]
});
// parse the JSON array response
```

**Strategy 3 — Add p-retry with exponential backoff** (already in package.json)
Use this when timeouts are intermittent (not systematic overload). Wrap the call:
```typescript
import pRetry from "p-retry";

const response = await pRetry(
  () => openai.chat.completions.create({ ... }),
  {
    retries: 3,
    minTimeout: 2000,
    factor: 2,
    onFailedAttempt: (err) => console.warn(`[OpenAI] Attempt ${err.attemptNumber} failed: ${err.message}`)
  }
);
```

**Strategy 4 — Reduce token budget on non-critical calls**
If `max_completion_tokens` is not set or is very high (>2000) on a call that only needs a short structured JSON response, reduce it. This speeds up responses and reduces timeout risk:
```typescript
// Non-critical material property lookup — cap at 800 tokens
max_completion_tokens: 800,
```

**Which strategy for which file:**
| File | Common Issue | Best Strategy |
|------|-------------|--------------|
| `server/learning/data-fetcher.ts` | Batch loops calling OpenAI per-material | Strategy 2 (batch) + Strategy 1 (spacing) |
| `server/learning/nlp-engine.ts` | Multiple sequential NLP calls | Strategy 1 (spacing) + Strategy 3 (retry) |
| `server/learning/insight-detector.ts` | Embedding calls in rapid succession | Strategy 1 (1s spacing) |
| `server/learning/structure-predictor.ts` | Long structure prediction prompts | Strategy 3 (retry) + Strategy 4 (token cap) |
| `server/learning/synthesis-tracker.ts` | Two calls close together | Strategy 2 (combine) |
| `server/learning/superconductor-research.ts` | Large prompts timing out | Strategy 3 (retry) + Strategy 4 (token cap) |
| `server/learning/model-llm-controller.ts` | LLM controller calling twice | Strategy 2 (combine if same context) |

### Silent Correctness Issues

These are bugs that don't crash but produce wrong science. The monitor now detects them in the activity feed. When found:

| Pattern | Fix Location | What to Do |
|---------|-------------|------------|
| `null → 0` on bandgap/lambda/Tc | Source of the `?? 0` — grep for the property name | Change `?? 0` to `?? null` for external data fields; only use `?? 0` for computed/derived values where 0 is a valid default |
| MP correction sets property to 0 | `server/learning/materials-project-client.ts` | Change `entry.<field> ?? 0` to `?? null` for that field |
| Large MP correction (>5x ratio) | `server/learning/data-fetcher.ts crossValidateWithMP` | Add a ratio guard: skip the correction if `Math.abs(mpVal / existingVal) > 5` — the MP match is probably the wrong polymorph |
| Known-safe material (H2O, NaCl, Fe, Al) getting Tc > 0 | `server/learning/multi-fidelity-pipeline.ts` Stage 0 | Check why it passed the ML filter; add formula-based early reject for common non-superconductors if needed |
| Many candidates with identical Tc (e.g. all 25 K) | `server/learning/engine.ts` `autoPhysExplicitlyZero` logic | Signal that GB model is returning training-mean for out-of-distribution inputs and physics veto isn't firing. Verify `autoPhysExplicitlyZero` check comes before `reconciledAuto.reconciledTc > 0` in the finalTc assignment block. |
| DFT queue shows `phonon_stable=false` candidates | `server/dft/dft-job-queue.ts` `preFiltered` filter | Verify the phonon stability gate is present: `if (mlFeatures.phononStable === false \|\| mlFeatures.qePhononStable === false) return false;` |
| QE phonon jobs complete but log shows `frequencies: []` | `server/dft/qe-worker.ts` `parsePhononOutput()` | Check whether QE is using the range notation `omega(N-M)` — the regex inside `parsePhononOutput` must contain `(?:\s*-\s*\d+)?` after the first `\d+` in the parentheses group. Same fix applies to `server/dft/dfpt-parser.ts`. |
| H-rich hydride (H:metal > 3) with score < 0.15 despite strong lifshitz/QCP signals | `server/learning/deliberative-evaluator.ts` `runPhysicsMerit()` | `isHRichHydride` guard must wrap the d-band penalty block — La, Y, and similar metals have D_ELECTRONS=0 which triggers a −0.20 penalty that incorrectly zeros Physics Merit for the entire compound. Check that `if (tmCount > 0 && !isHRichHydride)` is the condition, not just `if (tmCount > 0)`. |
| High-pressure candidate (>50 GPa) with ambient hull penalty in notes / low score | `server/learning/deliberative-evaluator.ts` `runRiskAssessment()` AND `server/learning/engine.ts` finalPayload | `mlFeatures.pressureGpa` must be populated (`pressureGpa: enforcedPressure` in the engine.ts finalPayload spread). Then in `runRiskAssessment`, `riskPressureGpa = mlFeatures?.pressureGpa ?? 0` must skip the `score -= 0.1` hull penalty when `riskPressureGpa > 50`. |
| Stoner > 3.0 + QCP confirmed but candidate rejected as "ferromagnetic" | `server/learning/deliberative-evaluator.ts` `runPhysicsMerit()` Stoner block | The Stoner penalty must only fire when `stonerRatio > 3.0 AND qcpScore < 0.3`. If QCP is confirmed (≥ 0.3), Stoner > 1 means spin-fluctuation SC near a magnetic QCP — no penalty, just a reasoning note. |
| Candidate with lifshitz ≥ 0.8 and QCP=true has ensemble score < 0.20 | Any penalty stage in `deliberative-evaluator.ts` | Strong physics signals should not result in a sub-0.20 score. Inspect the deliberation reasoning in the candidate's notes to identify which stage is applying a large penalty (d-band, hull, Stoner, or pressure viability) and verify that penalty is physically justified for this compound's actual conditions. |
| 3+ candidates share identical band structure values (gap, flatBand, VHS, nesting all the same) | `server/physics/band-structure-surrogate.ts` `computePhysicsCalibration()` | The band surrogate scales all outputs using only `tmCount`, `dElectronFraction`, and `metallicityHint`. Compounds with the same `tmCount` but different element types (e.g. Cu d10 vs Ni d8, Pr 4f3 vs Eu 4f7) collapse to identical values. Fix: add `fElectronFraction` (rare earth fraction), `closedDShellFraction` (Cu/Zn/Ag d10 fraction), and `openDShellTmFraction` (Ni/Fe/Co open-d fraction) to `calib`, then use them in the DOS, VHS, nesting, flatBand, and multiBand scaling. |
| `metallic=false` + `pairing=spin-fluctuation` in the same candidate | `server/learning/physics-engine.ts` — pairing mechanism selection | Spin-fluctuation SC requires itinerant conduction electrons. A non-metallic compound (DFT `metallic=false`) cannot have SF pairing. The physics engine should respect the DFT metallicity over the band surrogate when selecting the pairing channel. |
| Stable ferromagnet (Stoner I·N(Ef) ≥ 1.0) with QCP score > 0.5 and qcpType ≠ "ferromagnet" | `server/physics/advanced-constraints.ts` `computeQuantumCriticalConstraint()` | `isNearQCP = stonerProduct > 0.7 || stonerEnhancement > 10` fires for both near-QCP systems AND confirmed ferromagnets (stonerProduct ≥ 1.0). A confirmed ferromagnet has PASSED the magnetic QCP — it is in the ordered state, not at the phase boundary. Fix: when `isStableFerromagnet=true` AND NOT a heavy-fermion compound, multiply `score *= 0.15` and set `qcpType = "ferromagnet"`. |
| Stable ferromagnet with Mott channel score > 0.3 | `server/physics/quantum-criticality.ts` `detectQuantumCriticality()` | The Mott score is driven by `(1 - metallicity)`, which is high for non-metallic compounds. Non-metallic ferromagnets are exchange/band insulators — not Mott-Hubbard insulators. Fix: after computing `spinForGating = computeDynamicSpinSusceptibility()`, cap `mottScore = Math.min(mottScore, 0.15)` when `spinForGating.isStableFerromagnet === true`. |

---

## STEP 6 — Update State File

After all fixes (or if no fixes needed), update `logs/monitor-state.json`:

```json
{
  "cleanCycles": <increment if isClean, else reset to 0>,
  "totalCycles": <increment by 1>,
  "lastCheck": "<ISO timestamp>",
  "lastStatus": "<"clean" | "fixed" | "issues_remain">",
  "openIssues": [<list of issue messages that were NOT fixed this cycle>],
  "acceptedWarnings": [
    { "pattern": "<short stable identifier>",
      "since": "<ISO timestamp>",
      "reason": "<why ignored + what would re-open it>" }
  ],
  "fixHistory": [
    {
      "cycle": <totalCycles>,
      "timestamp": "<ISO>",
      "fixes": ["description of fix 1", "description of fix 2"]
    },
    ... (keep last 20 entries only)
  ],
  "notes": "Reset this file to restart the clean-cycle counter"
}
```

Rules:
- `cleanCycles` resets to 0 any time there are unfixed issues in `openIssues`. **Accepted warnings (in `acceptedWarnings`) do NOT block clean cycles.**
- `cleanCycles` increments when `isClean === true` AND `openIssues` is empty AND no new fixes were applied this cycle. Accepted warnings may still be present — that is fine.
- If you made fixes, set `lastStatus: "fixed"` and reset `cleanCycles` to 0 (fixes might have introduced new issues that need a clean run to confirm).
- Preserve `acceptedWarnings` across cycles. Only remove an entry if its `re-open` condition triggers (e.g. a peak threshold is exceeded), in which case move it back into `openIssues` and start a fresh investigation.
- When a warning fires this cycle that matches a `pattern` already in `acceptedWarnings`, **do not** add it to `openIssues` and **do not** count it against the clean streak. Just note in the cycle report that N accepted warnings are still firing.

---

## STEP 7 — Report to User

Output a concise cycle summary in this format:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QAE Monitor — Cycle #<N>  [<timestamp>]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:   CLEAN ✓ / FIXED / ISSUES REMAIN
Clean streak:  <cleanCycles> / 50

Issues found:  <count>
Fixes applied: <count>

<bullet list of fixes or "No issues — system healthy">

Warnings (no fix needed):
<bullet list or "none">
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Frontend / Page Visibility Issues

The monitor now has two real frontend signals: **page probes** (synthetic curl of each page's key endpoint) and **client error reports** (actual errors from the browser via `/api/client-errors`).

#### Page Probe Issues

| Issue | Fix Location | What to Do |
|-------|-------------|------------|
| `HTTP 500` on page probe | `server/routes.ts` — find the route handler | Add try/catch, return structured error, check for null/undefined in data fetch or transform |
| `HTTP 404` on page probe | `server/routes.ts` | Route may be missing or path has changed — add the route |
| `Slow response >5s` on page probe | The route handler + `server/cache.ts` | Add caching via `cache.getOrSet(key, ttl, fetchFn)` or reduce the query scope (add LIMIT) |
| `Endpoint unreachable` (code 000) | Server may be down | Run `bash scripts/start-server-logged.sh` |
| **`/api/pareto-frontier` slow (>5s repeatedly)** | `server/inverse/pareto-optimizer.ts` | Root cause: `scheduleParetoRecompute` loads 500 candidates; `nonDominatedSort` O(n²) sweep = 125k comparisons, yields only every 100 iters = 10-120s event loop blocks. Fix: (1) Add `const MAX_PARETO_CANDIDATES = 200` and `const candidates = all.slice(0, MAX_PARETO_CANDIDATES)` in `scheduleParetoRecompute`. (2) Change yield interval from `i % 100 === 0` to `i % 25 === 0` in the sweep loop. Both fixes already applied — if slow again, check they are still present and that loader is not bypassing the cap. |

#### Client Error Reports (from browser)

These are real errors reported by actual page navigation. They come with page path, endpoint, status code, and timing.

| Error Type | Meaning | Fix |
|------------|---------|-----|
| `render-crash` | React component threw during render — page showed error boundary | Find the component named in the stack trace, add null guard or fix the broken prop |
| `query-timeout` | `fetch()` took >5s and network error occurred | Find the endpoint in `server/routes.ts`, add caching or reduce query cost |
| `query-error` + HTTP 5xx | Server error hit by real browser request | Fix the route handler that threw 500 |
| `query-error` + HTTP 4xx | Bad request from frontend | Check the request shape the frontend is sending vs. what the route expects |
| `slow-load` | Response completed but took >5s | Add caching for the endpoint; check for N+1 DB queries in the handler |
| **Error reporting flood** (monitor reports "Error reporting flood from dashboard") | `client/src/lib/queryClient.ts` + `server/routes.ts` | See "Error reporting flood" row in the Frontend / HTTP Issues table above. The health check script detects this automatically when >12 query-errors arrive from dashboard within 90s across 4+ CYCLE_END_KEYS endpoints. |
| **Error reporter feedback loop** (monitor reports "error-reporter feedback loop") | `client/src/lib/queryClient.ts` | Add `if (payload.endpoint === '/api/client-errors') return;` as the first line of `reportClientError()`. |

#### Pattern: Too Many Concurrent Requests on One Page

The Dashboard page fires ~30 API calls simultaneously on mount. If the monitor sees many `slow-load` or `query-timeout` events all referencing `"page": "/"` (Dashboard):

1. Read `client/src/pages/dashboard.tsx` to see the `useQuery` calls
2. Identify groups that could share a single aggregated endpoint
3. Check if `server/routes.ts` has a `/api/dashboard` endpoint — if it already aggregates, check why it's slow (add timing logs or caching)
4. If individual sub-endpoints are slow, add per-route caching with a 30s TTL

#### Running the DB Migration for client_errors Table

The `client_errors` table must exist in the database. If `/api/client-errors` returns 500, the table may not be created yet. Run:
```bash
npm run db:push
```
This applies the schema to the database without data loss.

## IMPORTANT RULES

- **THE ACTIVITY FEED MUST RUN 24/7.** This is the primary objective. A frozen or slowing feed means zero science is being done. Every other metric is secondary to this. If the feed is not producing new entries at a normal rate, treat it as a CRITICAL issue and do not mark the cycle clean until it is restored AND the root cause is fixed in code so it cannot recur.
- **Do NOT restart the server** after making code changes during a loop — the dev server (tsx watch) hot-reloads automatically.
- **Do NOT change physics constants** (Tc thresholds, phonon cutoffs) without strong evidence from multiple repeated failures. Log a warning instead.
- **Do NOT lower ML thresholds** to make more candidates pass. If the pipeline is filtering everything, investigate why — it usually means a data or code bug, not that the thresholds are wrong.
- **Do NOT commit changes** — leave that for the user to review.
- **Focus on the most impactful fix first** — a DB connection issue affects everything; a single slow route affects only that page.
- **If unsure about a fix**, add a `// TODO(monitor): <description>` comment in the code and log it as a warning rather than making a potentially wrong change.
- **Read the relevant source file before editing it** — never edit blind.
