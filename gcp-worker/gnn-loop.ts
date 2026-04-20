/**
 * GCP GNN Training Loop
 * Polls gnn_training_jobs for queued jobs, trains the full GNN ensemble,
 * writes weights back, then fetches the next batch of MP data so each
 * subsequent training cycle has a progressively richer dataset.
 *
 * On the dedicated GNN GCP instance all 5 ensemble members train in parallel
 * via worker_threads (trainEnsembleParallel), cutting wall time by ~5×.
 * Falls back to sequential trainEnsemble if any worker thread fails.
 */
import { db, isConnectionError } from "../server/db";
import { storage } from "../server/storage";
import { acquireGNNTrainingSlot, releaseGNNTrainingSlot, waitForXGBIdle, isGNNTrainingSlotFree } from "./training-priority";
import { fileURLToPath } from "url";
import path from "path";
import { spawn, execSync, ChildProcess } from "child_process";
import * as http from "node:http";
import {
  ENSEMBLE_SIZE,
  ENSEMBLE_SEEDS,
  splitTrainValidation,
} from "../server/learning/graph-neural-net";
import type { TrainingSample } from "../server/learning/graph-neural-net";
import { fetchMPBatchFromAPI } from "../server/learning/materials-project-client";

// ── Python GNN service (FastAPI on localhost:8765) ────────────────────────────
const GNN_SERVICE_URL  = process.env.GNN_SERVICE_URL ?? "http://127.0.0.1:8765";
const GNN_SERVICE_PORT = process.env.GNN_SERVICE_PORT ?? "8765";
const GNN_PY_SCRIPT    = path.join(path.dirname(fileURLToPath(import.meta.url)), "../gnn/server.py");

let _pythonProcess: ChildProcess | null = null;
// True while a /train request is in flight — prevents spawnPythonGNNService
// from killing a busy-but-healthy Python process that's mid-training.
let _pythonTrainingInProgress = false;

/** Spawn the FastAPI GNN service and wait until it responds to /health. */
async function spawnPythonGNNService(): Promise<void> {
  // If we think Python is alive, verify it actually responds to /health.
  // The process can be alive but unresponsive (GIL stuck, uvicorn wedged,
  // OOM-thrashing). The old check `_pythonProcess && !_pythonProcess.killed`
  // only tests the ChildProcess reference — .killed is false even when the
  // process crashed on its own (it's only true after WE send a signal).
  if (_pythonProcess && !_pythonProcess.killed) {
    // If a /train request is in flight, the single-worker uvicorn blocks on
    // GPU training and CANNOT respond to /health. This is expected — don't
    // kill it. Just return and let callPythonTrain's long-running request
    // finish naturally.
    if (_pythonTrainingInProgress) return;

    try {
      const ac = new AbortController();
      const t = setTimeout(() => ac.abort(), 10_000);
      const res = await fetch(`${GNN_SERVICE_URL}/health`, { signal: ac.signal });
      clearTimeout(t);
      if (res.ok) return; // genuinely healthy
    } catch { /* health check failed — fall through to kill+respawn */ }
    console.warn("[GNN-GCP] Python process alive but /health unresponsive — killing and respawning");
    try { _pythonProcess.kill("SIGKILL"); } catch { /* ignore */ }
    _pythonProcess = null;
    // Give the OS a moment to release port 8765
    await new Promise(r => setTimeout(r, 3000));
  }

  // Also kill any orphan Python process bound to our port from a prior session
  // (e.g. Node restarted but old Python survived). Without this, the new Python
  // can't bind and the health check polls the stale orphan indefinitely.
  try {
    const res = await fetch(`${GNN_SERVICE_URL}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      console.warn("[GNN-GCP] Orphan Python process detected on port — killing via /shutdown or fuser");
      // Best-effort: ask it to shut down, then force-kill the port holder
      try {
        await fetch(`${GNN_SERVICE_URL}/shutdown`, { method: "POST", signal: AbortSignal.timeout(3000) });
      } catch { /* no /shutdown endpoint — use fuser */ }
      await new Promise(r => setTimeout(r, 2000));
      try {
        execSync(`fuser -k ${GNN_SERVICE_PORT}/tcp 2>/dev/null || true`);
      } catch { /* fuser not available or no process found */ }
      await new Promise(r => setTimeout(r, 3000));
    }
  } catch { /* nothing on the port — good */ }

  const python = process.env.PYTHON_BIN ?? "python3";
  console.log(`[GNN-GCP] Spawning Python GNN service: ${python} ${GNN_PY_SCRIPT}`);

  _pythonProcess = spawn(python, [GNN_PY_SCRIPT], {
    env: {
      ...process.env,
      GNN_SERVICE_PORT,
      GNN_WEIGHTS_DIR: process.env.GNN_WEIGHTS_DIR ?? "/opt/qae/gnn_weights",
      GNN_LOG_LEVEL: "INFO",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  _pythonProcess.stdout?.on("data", (d: Buffer) => process.stdout.write(`[py-gnn] ${d}`));
  _pythonProcess.stderr?.on("data", (d: Buffer) => process.stderr.write(`[py-gnn] ${d}`));
  _pythonProcess.on("exit", (code) => {
    console.warn(`[GNN-GCP] Python GNN service exited with code ${code}`);
    _pythonProcess = null;
  });

  // Wait up to 90 s for the service to start accepting requests
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2000));
    try {
      const res = await fetch(`${GNN_SERVICE_URL}/health`);
      if (res.ok) {
        const body = await res.json() as any;
        console.log(`[GNN-GCP] Python GNN service ready — device=${body.device} n_models=${body.n_models}`);
        return;
      }
    } catch { /* not ready yet */ }
  }
  throw new Error("Python GNN service did not become ready within 90s");
}

/**
 * POST a JSON body and read the JSON response, with NO 5-minute headers/body
 * timeouts. This intentionally does NOT use the global `fetch` because Node's
 * built-in fetch wraps undici, and undici has hard-coded defaults of
 * headersTimeout=300_000 and bodyTimeout=300_000. Python's /train endpoint
 * does ALL its work (graph build → ensemble training, 30-40 min for 62k
 * samples) BEFORE sending response headers, so undici's headersTimeout always
 * fires at exactly 5min 1s with `TypeError: fetch failed (cause:
 * HeadersTimeoutError)`. node:http has no equivalent client-side
 * headersTimeout — its default `timeout` is 0 (disabled). The only timeout
 * is the explicit abortMs ceiling we install ourselves.
 *
 * Cycle 1379 fix. The earlier attempt used `setGlobalDispatcher` from undici,
 * but undici is not a runtime dependency in this project (only undici-types
 * is in package.json), so the import threw MODULE_NOT_FOUND on boot and the
 * worker crashed in a tight restart loop with status=1.
 */
function postJsonLongRunning(
  url:     string,
  body:    unknown,
  abortMs: number,
): Promise<{ status: number; data: any; rawText: string }> {
  return new Promise((resolve, reject) => {
    let parsedUrl: URL;
    try {
      parsedUrl = new URL(url);
    } catch (err) {
      return reject(new Error(`Invalid URL: ${url}`));
    }
    const bodyStr = typeof body === "string" ? body : JSON.stringify(body);
    const opts: http.RequestOptions = {
      hostname: parsedUrl.hostname,
      port:     parsedUrl.port || 80,
      path:     parsedUrl.pathname + parsedUrl.search,
      method:   "POST",
      headers:  {
        "Content-Type":   "application/json",
        "Content-Length": Buffer.byteLength(bodyStr),
      },
      // Timeout disabled (0): Python's single-worker uvicorn blocks during GPU
      // training and can't accept TCP connections or send response headers until
      // the full training run finishes (30-90 min). Any finite socket timeout
      // causes ETIMEDOUT for EVERY training job. The abortMs ceiling (4 hours)
      // is the ONLY upper bound — it fires via the abortTimer below.
      timeout: 0,
    };

    const req = http.request(opts, (res) => {
      const chunks: Buffer[] = [];
      res.on("data",  (chunk: Buffer) => chunks.push(chunk));
      res.on("error", reject);
      res.on("end",   () => {
        const rawText = Buffer.concat(chunks).toString("utf-8");
        let data: any = null;
        try { data = JSON.parse(rawText); } catch { /* leave null; caller checks */ }
        resolve({ status: res.statusCode ?? 0, data, rawText });
      });
    });

    // timeout is 0 (disabled) — no timeout handler needed.
    // The only upper bound is the abortTimer below.
    req.on("error", reject);

    // Hard ceiling: kill the request after abortMs (default 60 min). Without
    // this we'd have no upper bound at all if Python truly hangs forever.
    const abortTimer = setTimeout(() => {
      req.destroy(new Error(`POST ${url} exceeded abort ceiling of ${abortMs}ms`));
    }, abortMs);
    req.on("close", () => clearTimeout(abortTimer));

    req.write(bodyStr);
    req.end();
  });
}

/** POST training data to the Python service, returns parsed response. */
async function callPythonTrain(
  jobId:             number,
  trainingData:      TrainingSample[],
  startupValR2?:     number,
  maxPretrainEpochs: number = 15,
): Promise<{
  status: "done" | "discarded" | "failed";
  reason?: string;
  r2: number; mae: number; rmse: number;
  trainR2: number; trainMae: number;
  valN: number;
  ci95Coverage: number; ci95Width: number;
  wallSeconds: number;
  modelPath?: string;
  nSamples: number;
  xgbR2: number | null; xgbMae: number | null;
  xgbRmse: number | null; xgbNTrain: number | null;
}> {
  // Cycle 1377 fix: ensure the Python child is alive before each /train call.
  // The exit handler in spawnPythonGNNService sets _pythonProcess = null when
  // Python dies (OOM, crash, manual kill). spawnPythonGNNService is idempotent
  // — its early-return check at the top makes this a no-op when Python is
  // healthy, and a fresh spawn when it's not. Without this, Python dying
  // mid-session would leave Node polling forever with every /train call
  // hitting `fetch failed` (connection refused) and the worker stuck idle.
  await spawnPythonGNNService();

  const body = {
    job_id:              jobId,
    training_data:       trainingData.map(s => ({
      ...s,
      lambda: (s as any).lambda,   // preserve field name for Python
    })),
    max_pretrain_epochs: maxPretrainEpochs,
    startup_val_r2:      startupValR2 ?? null,
  };

  // The ceiling here is the ONLY upper bound on training wall-time.
  // 4 hours leaves ~3x headroom for dataset growth and slow runs.
  const trainAbortMs = parseInt(process.env.GNN_TRAIN_TIMEOUT_MS ?? "", 10) || (4 * 60 * 60_000);
  _pythonTrainingInProgress = true;
  let res: Awaited<ReturnType<typeof postJsonLongRunning>>;
  try {
    res = await postJsonLongRunning(`${GNN_SERVICE_URL}/train`, body, trainAbortMs);
  } finally {
    _pythonTrainingInProgress = false;
  }

  if (res.status < 200 || res.status >= 300) {
    throw new Error(`Python /train failed ${res.status}: ${res.rawText.slice(0, 200)}`);
  }
  if (!res.data) {
    throw new Error(`Python /train returned non-JSON body: ${res.rawText.slice(0, 200)}`);
  }
  const data = res.data;
  const m    = data.metrics ?? {};
  return {
    status:       data.status,
    reason:       data.reason,
    r2:           m.r2        ?? 0,
    mae:          m.mae       ?? 0,
    rmse:         m.rmse      ?? 0,
    trainR2:      m.train_r2  ?? 0,
    trainMae:     m.train_mae ?? 0,
    valN:         m.val_n     ?? 0,
    ci95Coverage: m.ci95_coverage ?? 0,
    ci95Width:    m.ci95_width    ?? 0,
    wallSeconds:  m.wall_seconds  ?? 0,
    modelPath:    m.model_path,
    nSamples:     m.n_samples ?? trainingData.length,
    // Cycle 1381: XGBoost metrics from _train_xgboost on the full corpus
    xgbR2:        m.xgb_r2     ?? null as number | null,
    xgbMae:       m.xgb_mae    ?? null as number | null,
    xgbRmse:      m.xgb_rmse   ?? null as number | null,
    xgbNTrain:    m.xgb_n_train ?? null as number | null,
  };
}

/** Store the PyTorch model path in system_state for observability. */
async function recordPytorchModelPath(jobId: number, modelPath: string): Promise<void> {
  try {
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('gnn_pytorch_model_path', '{"jobId":${jobId},"path":${JSON.stringify(modelPath)}}', NOW())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`
    );
  } catch { /* non-fatal */ }
}

// trainEnsembleParallel and spawnModelWorker removed —
// training is now delegated to the Python FastAPI service (gnn/server.py)
// via callPythonTrain() defined above.

const POLL_INTERVAL_MS = 10_000;
const MP_BATCH_SIZE = 1_000;        // records per progressive fetch (2x for faster ramp-up)
const MP_MAX_CACHE = 1_000_000;     // effectively unlimited — let the GNN train on all available MP data
const MP_SKIP_STATE_KEY = "mp_batch_skip";
let running = true;

// In-memory skip counter — primary source of truth. DB write is best-effort.
let _mpSkip = 0;
let _mpSkipLoaded = false;
// Mutex: prevents concurrent GNN job completions from each firing a duplicate MP fetch.
let _mpFetchInProgress = false;

// ── DB retry helper ──────────────────────────────────────────────────────────
// Wraps a DB query with exponential-backoff retry for transient connection errors
// (Neon "Exceeded concurrency limit", "connection terminated", ETIMEDOUT, etc.).
// Without this, each loader's silent catch returns empty data when Neon throttles,
// which causes training to proceed with near-empty datasets and immediately fail.
async function withDBRetry<T>(
  label: string,
  fn: () => Promise<T>,
  maxAttempts = 4,
): Promise<T> {
  let lastErr: any;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err: any) {
      lastErr = err;
      const msg = String(err?.message ?? err ?? "").toLowerCase();
      const isRetryable =
        isConnectionError(err) ||
        msg.includes("concurrency limit") ||
        msg.includes("too many clients") ||
        msg.includes("rate limit") ||
        msg.includes("server_login_retry");
      if (!isRetryable || attempt === maxAttempts) throw err;
      const delayMs = Math.min(15_000, 1500 * 2 ** (attempt - 1));
      console.warn(
        `[GNN-GCP] ${label} attempt ${attempt}/${maxAttempts} failed: ${msg.slice(0, 100)} — retrying in ${delayMs}ms`
      );
      await new Promise(r => setTimeout(r, delayMs));
    }
  }
  throw lastErr;
}

// ── MP progressive fetch ─────────────────────────────────────────────────────

async function getMPSkip(): Promise<number> {
  if (_mpSkipLoaded) return _mpSkip;
  try {
    const rows = await db.execute(
      `SELECT value FROM system_state WHERE key = '${MP_SKIP_STATE_KEY}'`
    );
    const row = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
    _mpSkip = Number((row?.value as any)?.skip ?? 0);
  } catch {
    // DB not yet available — start from 0
  }
  _mpSkipLoaded = true;
  return _mpSkip;
}

async function setMPSkip(skip: number): Promise<void> {
  _mpSkip = skip; // always update in-memory immediately
  try {
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('${MP_SKIP_STATE_KEY}', '{"skip":${skip}}', NOW())
       ON CONFLICT (key) DO UPDATE SET value = '{"skip":${skip}}', updated_at = NOW()`
    );
  } catch (err: any) {
    console.warn(`[GNN-GCP] setMPSkip DB write failed (using in-memory ${skip}): ${err.message?.slice(0,80)}`);
  }
}

async function getCachedMPCount(): Promise<number> {
  try {
    const rows = await db.execute(
      `SELECT COUNT(*)::int AS count FROM mp_material_cache WHERE data_type = 'summary'`
    );
    const row = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
    return Number(row?.count ?? 0);
  } catch {
    return 0;
  }
}

async function fetchNextMPBatch(): Promise<void> {
  // Mutex: if a fetch is already in progress (e.g. from a concurrent GNN job completion), skip.
  if (_mpFetchInProgress) {
    console.log(`[GNN-GCP] MP fetch already in progress — skipping duplicate`);
    return;
  }
  _mpFetchInProgress = true;
  try {
    const cached = await getCachedMPCount();
    if (cached >= MP_MAX_CACHE) {
      console.log(`[GNN-GCP] MP cache full (${cached} records) — skipping fetch`);
      return;
    }

    const skip = await getMPSkip();
    console.log(`[GNN-GCP] Fetching MP batch: skip=${skip}, limit=${MP_BATCH_SIZE} (cache has ${cached} records)`);

    const records = await fetchMPBatchFromAPI(MP_BATCH_SIZE, skip);

    if (records.length === 0) {
      // Likely reached the end of MP results — reset to start cycling through again
      console.log(`[GNN-GCP] MP API returned 0 records at skip=${skip} — resetting to 0`);
      await setMPSkip(0);
      return;
    }

    await setMPSkip(skip + records.length);
    console.log(`[GNN-GCP] MP batch done: +${records.length} records cached (total ~${cached + records.length}, next skip=${skip + records.length})`);
  } catch (err: any) {
    console.warn(`[GNN-GCP] MP batch fetch failed: ${err.message}`);
  } finally {
    _mpFetchInProgress = false;
  }
}

// computeMetrics and computeCalibration removed —
// metrics are now computed inside the Python service (gnn/server.py)
// and returned directly in the callPythonTrain() response.

// Kept only for the CI95 calibration metric storage helper below.
// ── CI95 stub (values come from Python service) ───────────────────────────────
// (calibration values come from Python service response)

async function storeCalibrationMetric(
  jobId: number,
  coverage: number,
  meanWidth: number,
  n: number,
): Promise<void> {
  try {
    await db.execute(
      `INSERT INTO system_metrics (metric_name, metric_value, metadata, recorded_at)
       VALUES (
         'gnn_ci95_coverage',
         ${coverage.toFixed(4)},
         '{"jobId":${jobId},"meanWidthK":${meanWidth.toFixed(2)},"n":${n}}',
         NOW()
       )` as any
    );
  } catch (err: any) {
    // system_metrics may not exist yet — not fatal
    console.warn(`[GNN-GCP] Could not store calibration metric: ${err.message?.slice(0, 80)}`);
  }
}

// ── Comprehensive metrics writer (system_state for frontend) ─────────────────
// Writes all GNN + XGB metrics to a single system_state key so the frontend
// can read them from the DB without polling individual training job rows.
// This is the authoritative source for dashboard cards (GNN R², Active Learning
// dataset size, XGB R², etc.).

async function writeGnnMetricsToSystemState(
  jobId: number,
  metrics: {
    r2: number; mae: number; rmse: number;
    trainR2: number; trainMae: number; valN: number;
    datasetSize: number;
    xgbR2: number | null; xgbMae: number | null;
    wallSeconds: number;
    source: 'startup' | 'dispatched';
    ci95Coverage?: number; ci95Width?: number;
  },
): Promise<void> {
  try {
    const payload = JSON.stringify({
      jobId,
      ...metrics,
      updatedAt: new Date().toISOString(),
    });
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('gnn_latest_metrics', '${payload}'::jsonb, NOW())
       ON CONFLICT (key) DO UPDATE SET value = '${payload}'::jsonb, updated_at = NOW()`
    );
    console.log(`[GNN-GCP] Wrote gnn_latest_metrics to system_state — R²=${metrics.r2.toFixed(4)} dataset=${metrics.datasetSize} xgbR²=${metrics.xgbR2?.toFixed(4) ?? 'N/A'}`);
  } catch (err: any) {
    console.warn(`[GNN-GCP] Failed to write gnn_latest_metrics: ${err.message?.slice(0, 80)}`);
  }
}

// ── QE dataset augmentation (DFT-verified SC samples) ────��───────────────────

async function loadQEDatasetSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    const rows = await withDBRetry("loadQEDatasetSamples", () => db.execute(
      `SELECT material, tc, formation_energy, band_gap, lambda, omega_log, mu_star
       FROM quantum_engine_dataset
       WHERE tc >= 1.0 AND lambda IS NOT NULL AND lambda > 0
       ORDER BY tc DESC
       LIMIT 5000`
    ));
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      if (!row.material || existingFormulas.has(row.material)) continue;
      // omega_log in quantum_engine_dataset is stored in cm⁻¹; convert to Kelvin (×1.4388)
      // so TrainingSample.omegaLog is always in Kelvin (matching GNN internal units).
      const omegaLogCm1 = row.omega_log != null ? Number(row.omega_log) : null;
      const omegaLogK = omegaLogCm1 != null && omegaLogCm1 > 0 ? omegaLogCm1 * 1.4388 : undefined;
      const muStar = row.mu_star != null ? Number(row.mu_star) : undefined;
      const bgRaw = row.band_gap != null ? Number(row.band_gap) : null;
      samples.push({
        formula: row.material,
        tc: Number(row.tc) || 0,
        formationEnergy: row.formation_energy != null ? Number(row.formation_energy) : undefined,
        bandgap: bgRaw != null && Number.isFinite(bgRaw) && bgRaw >= 0 ? bgRaw : undefined,
        lambda: row.lambda != null ? Number(row.lambda) : undefined,
        omegaLog: omegaLogK != null && Number.isFinite(omegaLogK) ? omegaLogK : undefined,
        muStar: muStar != null && Number.isFinite(muStar) && muStar > 0 ? muStar : undefined,
        dataConfidence: "physics-enriched", // QE entries with lambda/omega_log/mu_star
        structure: undefined,
        prototype: undefined,
        sourceTag: 'qe-dft',
      });
    }
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadQEDatasetSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── SuperCon external entries (NIMS + JARVIS SC samples) ─────────────────────
// Loads superconductors from supercon_external_entries, which contains both the
// NIMS SuperCon database (~21k entries) and JARVIS SC datasets (SuperCon-Chem,
// SuperCon-2D, etc.). The local server dispatch payload often contains only the
// core SUPERCON_TRAINING_DATA (~274 samples); loading from the DB here ensures
// every GNN training job uses the full 16k+ SC corpus.

async function loadSuperconExternalSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    // LEFT JOIN with jarvis-dft3d by shared external_id (JVASP-XXXXX) to pull
    // formation_energy_per_atom, bandgap_ev, and other DFT properties that are
    // absent from jarvis-supercon_3d entries. ~907/1058 SC3D entries match.
    // LIMIT 50000 caps the result set so a single query can never blow past Neon's
    // statement timeout / concurrency window — there are ~16k SC entries today, so
    // the cap is purely defensive against future growth swelling the JOIN.
    const rows = await withDBRetry("loadSuperconExternalSamples", () => db.execute(
      `SELECT sc.formula, sc.tc, sc.lambda, sc.space_group, sc.crystal_system,
              sc.source,
              (sc.raw_data->>'wlog_K')::real                              AS omega_log_k,
              (sc.raw_data->>'mu_star')::real                             AS raw_mu_star,
              sc.raw_data->>'data_confidence'                             AS data_confidence,
              (dft.raw_data->>'formation_energy_per_atom')::real          AS fe_per_atom,
              (dft.raw_data->>'bandgap_ev')::real                        AS bandgap_ev
       FROM supercon_external_entries sc
       LEFT JOIN supercon_external_entries dft
         ON dft.external_id = sc.external_id AND dft.source = 'jarvis-dft3d'
       WHERE sc.is_superconductor = true AND sc.tc > 0
       ORDER BY sc.source, sc.tc DESC
       LIMIT 50000`
    ));
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    const srcCounts: Record<string, number> = {};
    for (const row of items) {
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const tc = Number(row.tc);
      if (!Number.isFinite(tc) || tc <= 0) continue;
      const lambda = row.lambda != null ? Number(row.lambda) : undefined;
      // wlog_K from JARVIS-SuperCon-3D is already in Kelvin
      const omegaLogRaw = row.omega_log_k != null ? Number(row.omega_log_k) : null;
      const omegaLog = omegaLogRaw != null && Number.isFinite(omegaLogRaw) && omegaLogRaw > 0
        ? omegaLogRaw : undefined;
      const muStarRaw = row.raw_mu_star != null ? Number(row.raw_mu_star) : null;
      const muStar = muStarRaw != null && Number.isFinite(muStarRaw) && muStarRaw > 0
        ? muStarRaw : undefined;
      const spaceGroup = row.space_group as string | null;
      const crystalSystem = row.crystal_system as string | null;
      // Formation energy from DFT3D JOIN (per_atom), with sanity bounds
      const feRaw = row.fe_per_atom != null ? Number(row.fe_per_atom) : null;
      const formationEnergy = feRaw != null && Number.isFinite(feRaw) && feRaw > -20 && feRaw < 5
        ? feRaw : undefined;
      // Bandgap from DFT3D JOIN
      const bgRaw = row.bandgap_ev != null ? Number(row.bandgap_ev) : null;
      const bandgap = bgRaw != null && Number.isFinite(bgRaw) && bgRaw >= 0 ? bgRaw : undefined;
      // data_confidence from JARVIS SC3D raw_data — "dft-verified" unlocks 5× loss weight
      const dataConfidence = row.data_confidence === "dft-verified" ? "dft-verified" as const : undefined;
      const src = (row.source as string) ?? 'unknown';
      srcCounts[src] = (srcCounts[src] ?? 0) + 1;
      // Normalize source tag to one of the GNN's known one-hot buckets
      const sourceTag = src === 'hamidieh' ? 'hamidieh'
        : src.startsWith('jarvis-supercon') ? 'jarvis-sc'
        : src === '3dsc-mp' ? '3dsc-mp'
        : src === 'jarvis-dft3d' ? 'contrast-jarvis'
        : src;
      samples.push({
        formula,
        tc,
        formationEnergy,
        bandgap,
        dataConfidence,
        lambda: lambda != null && Number.isFinite(lambda) ? lambda : undefined,
        omegaLog,
        muStar,
        structure: spaceGroup || crystalSystem
          ? { spaceGroup: spaceGroup ?? undefined, crystalSystem: crystalSystem ?? undefined, dimensionality: undefined }
          : undefined,
        prototype: undefined,
        sourceTag,
      });
    }
    const srcSummary = Object.entries(srcCounts).map(([k, v]) => `${k}=${v}`).join(', ');
    console.log(`[GNN-GCP] loadSuperconExternalSamples: ${samples.length} total (${srcSummary})`);
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadSuperconExternalSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── JARVIS DFT3D metallic negatives (Tc=0 contrast from JARVIS) ──────────────
// Loads metallic (non-SC) materials from the JARVIS DFT3D dataset stored in
// supercon_external_entries. These Tc=0 contrast examples teach the GNN that
// most metals don't superconduct.

async function loadJarvisDFT3DContrast(existingFormulas: Set<string>, scCount: number): Promise<TrainingSample[]> {
  try {
    const rows = await withDBRetry("loadJarvisDFT3DContrast", () => db.execute(
      `SELECT formula, space_group, crystal_system,
              (raw_data->>'formation_energy_per_atom')::real  AS fe_per_atom,
              (raw_data->>'bandgap_ev')::real                 AS bandgap_ev
       FROM supercon_external_entries
       WHERE source IN ('jarvis-dft3d', 'JARVIS-DFT3D-Metallic') AND (tc IS NULL OR tc = 0)
       ORDER BY RANDOM()
       LIMIT ${Math.min(scCount, 20000)}`
    ));
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const spaceGroup = row.space_group as string | null;
      const crystalSystem = row.crystal_system as string | null;
      const feRaw = row.fe_per_atom != null ? Number(row.fe_per_atom) : null;
      const formationEnergy = feRaw != null && Number.isFinite(feRaw) && feRaw > -20 && feRaw < 5
        ? feRaw : undefined;
      const bgRaw = row.bandgap_ev != null ? Number(row.bandgap_ev) : null;
      const bandgap = bgRaw != null && Number.isFinite(bgRaw) && bgRaw >= 0 ? bgRaw : undefined;
      samples.push({
        formula,
        tc: 0,
        formationEnergy,
        bandgap,
        structure: spaceGroup || crystalSystem
          ? { spaceGroup: spaceGroup ?? undefined, crystalSystem: crystalSystem ?? undefined, dimensionality: undefined }
          : undefined,
        prototype: undefined,
        sourceTag: 'contrast-jarvis',
      });
    }
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadJarvisDFT3DContrast failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── MP augmentation (Tc=0 contrast examples) ─────────────────────────────────

async function loadMPContrastSamples(existingFormulas: Set<string>, scCount: number): Promise<TrainingSample[]> {
  try {
    const rows = await withDBRetry("loadMPContrastSamples", () => db.execute(
      `SELECT formula, data FROM mp_material_cache WHERE data_type = 'summary' LIMIT 10000`
    ));
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const maxContrast = scCount;  // 1:1 cap — avoids Tc=0 majority diluting SC signal
    const samples: TrainingSample[] = [];
    for (const row of items) {
      if (samples.length >= maxContrast) break;
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const d = row.data as any;
      if (!d) continue;
      // Default to 0 (metallic) when field is absent — we only seed metallic formulas,
      // so missing bandGap means the API didn't return it, not that it's an insulator.
      if (d.bandGap != null && d.bandGap > 0.1) continue;
      samples.push({ formula, tc: 0, formationEnergy: d.formationEnergyPerAtom ?? undefined, structure: undefined, prototype: undefined, sourceTag: 'contrast-mp' });
    }
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadMPContrastSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── MP Cuprate + Hydride fetch (matches Colab Cell 6) ─────────────────────────
// Fetches cuprate (Cu+O) and hydride (H) metallic materials from the MP cache.
// These get KNOWN_TC labels applied in server.py for Tc supervision, or serve
// as unlabeled structural training examples.

async function loadMPCuprateHydrideSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    const rows = await withDBRetry("loadMPCuprateHydrideSamples", () => db.execute(
      `SELECT formula, data FROM mp_material_cache
       WHERE data_type = 'summary'
         AND (formula LIKE '%Cu%' OR formula LIKE '%H%')
       LIMIT 5000`
    ));
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const d = row.data as any;
      if (!d) continue;
      // Skip insulators — only metallic cuprates/hydrides are SC candidates
      if (d.bandGap != null && d.bandGap > 0.1) continue;
      // Skip high energy-above-hull (unstable)
      if (d.energyAboveHull != null && d.energyAboveHull > 0.15) continue;
      const isCuprate = /Cu/.test(formula) && /O/.test(formula);
      const isHydride = /H/.test(formula);
      if (!isCuprate && !isHydride) continue;
      const spaceGroup = d.symmetry?.symbol ?? undefined;
      const crystalSystem = d.symmetry?.crystal_system?.toLowerCase() ?? undefined;
      samples.push({
        formula,
        tc: 0, // unlabeled — KNOWN_TC in server.py will override if matched
        formationEnergy: d.formationEnergyPerAtom ?? undefined,
        bandgap: d.bandGap != null && d.bandGap >= 0 ? d.bandGap : undefined,
        structure: spaceGroup || crystalSystem
          ? { spaceGroup, crystalSystem, dimensionality: undefined }
          : undefined,
        prototype: undefined,
        sourceTag: isCuprate ? 'mp-cuprate' : 'mp-hydride',
      });
    }
    const cuprates = samples.filter(s => s.sourceTag === 'mp-cuprate').length;
    const hydrides = samples.filter(s => s.sourceTag === 'mp-hydride').length;
    console.log(`[GNN-GCP] loadMPCuprateHydrideSamples: ${samples.length} total (${cuprates} cuprates, ${hydrides} hydrides)`);
    return samples;
  } catch (err: any) {
    console.warn(`[GNN-GCP] loadMPCuprateHydrideSamples failed: ${err.message?.slice(0, 120)}`);
    return [];
  }
}

// ── Job processing ────────────────────────────────────────────────────────────

async function processNextGnnJob(): Promise<boolean> {
  // Cycle 1375 fix: defer to startup corpus training. While startup is in its
  // outer retry loop and hasn't successfully run yet, the dispatched-job poll
  // loop must NOT claim the GNN slot — otherwise it locks startup out for the
  // duration of a multi-hour dispatched job. Without this guard, the poll loop
  // (which starts at T+125s) races startup (which begins data-loading at T+120s
  // and reaches its slot acquisition only after ~30-90s of DB queries) and wins
  // the race, leaving startup permanently deferred.
  if (_startupCorpusActive && !_startupCorpusRan) {
    return false;
  }

  // Silent slot check — avoids the claim→augment→requeue loop without spamming
  // acquire/release log messages every poll cycle when the slot is free but idle.
  if (!isGNNTrainingSlotFree()) {
    return false; // startup training holds the slot; try again next poll cycle
  }

  // If a /train request is already in flight, Python is busy — don't claim
  // another job. The DB-based check here replaces the old health-check approach
  // which was unreliable (Python can't respond to /health while training).
  if (_pythonTrainingInProgress) {
    return false;
  }

  const rows = await db.execute(
    `UPDATE gnn_training_jobs
     SET status = 'running', started_at = NOW()
     WHERE id = (
       SELECT id FROM gnn_training_jobs
       WHERE status = 'queued'
       ORDER BY created_at ASC
       LIMIT 1
       FOR UPDATE SKIP LOCKED
     )
     RETURNING id, training_data, dataset_size, dft_samples`
  );

  const job = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
  if (!job) return false;

  const jobId: number = job.id;
  let trainingData: TrainingSample[] = job.training_data as TrainingSample[];
  const datasetSize: number = job.dataset_size ?? trainingData.length;

  const existingFormulas = new Set(trainingData.map(s => s.formula));

  // Pre-warm the Neon connection before augmentation queries. Neon may cold-start
  // after inactivity; a failed warm-up here causes silent empty results below.
  // withDBRetry handles "Exceeded concurrency limit" + transient connection errors
  // with exponential backoff, so a momentary cap hit doesn't cascade into empty
  // augmentation results.
  try {
    await withDBRetry("pre-warm", () => db.execute("SELECT 1"), 3);
  } catch (warmErr: any) {
    console.warn(`[GNN-GCP] DB pre-warm failed for job #${jobId} after retries: ${warmErr.message?.slice(0, 80)} — augmentation queries may return empty`);
    // Larger delay to let the pool recover before continuing
    await new Promise(r => setTimeout(r, 5000));
  }

  // Augment with NIMS + JARVIS superconductors from supercon_external_entries.
  // The local server dispatch payload often contains only the core SUPERCON seed
  // (~274 samples); the full 16k+ SC corpus lives in the shared Neon DB.
  const externalSCSamples = await loadSuperconExternalSamples(existingFormulas);
  if (externalSCSamples.length > 0) {
    trainingData = [...trainingData, ...externalSCSamples];
    externalSCSamples.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${externalSCSamples.length} NIMS+JARVIS SC entries — total ${trainingData.length}`);
  }

  // Augment with DFT-verified SC entries from quantum_engine_dataset
  const qeSamples = await loadQEDatasetSamples(existingFormulas);
  if (qeSamples.length > 0) {
    trainingData = [...trainingData, ...qeSamples];
    qeSamples.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${qeSamples.length} QE dataset SC entries — total ${trainingData.length}`);
  }

  // ── Fix #6: Cross-reference BCS physics from quantum_engine_dataset ────
  // supercon_external_entries has 0 omega_log/mu_star, but quantum_engine_dataset
  // has 2000 entries with all BCS parameters. Enrich by formula match.
  const physicsMap = new Map<string, { lambda?: number; omegaLog?: number; muStar?: number }>();
  for (const s of qeSamples) {
    if (s.lambda || s.omegaLog || s.muStar) {
      physicsMap.set(s.formula, { lambda: s.lambda, omegaLog: s.omegaLog, muStar: s.muStar });
    }
  }
  let enrichedCount = 0;
  for (const s of trainingData) {
    if (!s.lambda && !s.omegaLog && !s.muStar) {
      const physics = physicsMap.get(s.formula);
      if (physics) {
        if (physics.lambda && !s.lambda) s.lambda = physics.lambda;
        if (physics.omegaLog && !s.omegaLog) s.omegaLog = physics.omegaLog;
        if (physics.muStar && !s.muStar) s.muStar = physics.muStar;
        enrichedCount++;
      }
    }
  }
  if (enrichedCount > 0) {
    console.log(`[GNN-GCP] BCS physics enrichment: ${enrichedCount} samples gained lambda/omegaLog/muStar from QE dataset`);
  }

  // Augment with Tc=0 contrast: JARVIS DFT3D metallic negatives (structure-enriched)
  // plus MP metallic cache. 1:1 cap vs SC count to avoid label imbalance.
  const scCount = trainingData.filter(s => s.tc > 0).length;
  const jarvisContrast = await loadJarvisDFT3DContrast(existingFormulas, scCount);
  if (jarvisContrast.length > 0) {
    trainingData = [...trainingData, ...jarvisContrast];
    jarvisContrast.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${jarvisContrast.length} JARVIS DFT3D contrast (Tc=0) — total ${trainingData.length}`);
  }

  const remainingContrastSlots = Math.max(0, scCount - jarvisContrast.length);
  const mpContrast = await loadMPContrastSamples(existingFormulas, remainingContrastSlots);
  if (mpContrast.length > 0) {
    trainingData = [...trainingData, ...mpContrast];
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${mpContrast.length} MP contrast samples (Tc=0) — total ${trainingData.length}`);
  }

  // Augment with MP cuprates + hydrides (matches Colab Cell 6).
  // These get KNOWN_TC labels applied in server.py; unlabeled ones serve
  // as structural training examples.
  const mpCuprateHydride = await loadMPCuprateHydrideSamples(existingFormulas);
  if (mpCuprateHydride.length > 0) {
    trainingData = [...trainingData, ...mpCuprateHydride];
    mpCuprateHydride.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${mpCuprateHydride.length} MP cuprate/hydride samples — total ${trainingData.length}`);
  }

  // ── Augmentation sanity gate ──────────────────────────────────────────
  // If the local server's dispatch payload was tiny (~274 SC seed samples)
  // AND every augmentation loader exhausted its retries (e.g. Neon DB cap
  // hit during the augmentation burst), the resulting trainingData is
  // useless to train on. Requeue the job rather than marking it failed —
  // failed jobs cause the local server to dispatch a fresh one immediately,
  // which hits the same DB cap, and the loop repeats. Requeuing gives the
  // pool time to recover before the next attempt.
  const scInTrainingData = trainingData.filter(s => s.tc > 0).length;
  const MIN_JOB_SC_SAMPLES = 200;
  if (scInTrainingData < MIN_JOB_SC_SAMPLES && externalSCSamples.length === 0) {
    console.warn(
      `[GNN-GCP] Job #${jobId}: augmentation loaders all returned empty (only ${scInTrainingData} SC samples in dispatched payload). ` +
      `Likely Neon DB cap hit during data fetching. Requeuing job and waiting 60s before next poll.`
    );
    await db.execute(
      `UPDATE gnn_training_jobs SET status = 'queued', started_at = NULL WHERE id = ${jobId}`
    ).catch(() => {});
    await new Promise(r => setTimeout(r, 60_000));
    return false;
  }

  // Split BEFORE training so validation data is never seen by the models.
  // SC samples (Tc>0) are stratified: 80% train / 20% val.
  // Tc=0 contrast goes entirely into training — we validate on SC targets only.
  const scSamples  = trainingData.filter(s => s.tc > 0);
  const nonScSamples = trainingData.filter(s => s.tc === 0);
  const { train: scTrain, validation: scVal } = splitTrainValidation(scSamples, 0.20, 42);
  // Cap Tc=0 contrast samples to at most 1× SC training count to prevent
  // the model from collapsing to predict 0 (contrast imbalance was 2.4:1).
  const cappedNonSc = nonScSamples.slice(0, scTrain.length);
  const trainSet = [...scTrain, ...cappedNonSc];
  const valSet   = scVal; // held-out SC samples only — these are the discovery targets

  console.log(
    `[GNN-GCP] Starting training job #${jobId} — ${datasetSize} seed + ${externalSCSamples.length} NIMS+JARVIS + ${qeSamples.length} QE + ${jarvisContrast.length + mpContrast.length} contrast (capped ${cappedNonSc.length})` +
    ` | train=${trainSet.length} val=${valSet.length} (SC holdout)`
  );
  const startMs = Date.now();

  // Load startup R² for quality gate comparison
  let startupValR2 = -Infinity;
  try {
    const stateRow = await db.execute(
      `SELECT value FROM system_state WHERE key = '${STARTUP_CORPUS_STATE_KEY}'`
    );
    const stateVal = (stateRow as any).rows?.[0]?.value ?? (Array.isArray(stateRow) ? stateRow[0]?.value : undefined);
    if (stateVal) {
      const parsed = typeof stateVal === 'string' ? JSON.parse(stateVal) : stateVal;
      startupValR2 = parsed.valR2 ?? -Infinity;
    }
  } catch { /* non-fatal */ }

  try {
    await waitForXGBIdle();
    if (!acquireGNNTrainingSlot(`Job#${jobId}`)) {
      await db.execute(`UPDATE gnn_training_jobs SET status = 'queued', started_at = NULL WHERE id = ${jobId}`);
      return false;
    }
    let pyResult: Awaited<ReturnType<typeof callPythonTrain>>;
    try {
      console.log(`[GNN-GCP] Delegating job #${jobId} to Python GNN service…`);
      pyResult = await callPythonTrain(
        jobId, trainingData, Number.isFinite(startupValR2) ? startupValR2 : undefined,
      );
    } finally {
      releaseGNNTrainingSlot(`Job#${jobId}`);
    }

    const { r2, mae, rmse, trainR2, trainMae, valN, ci95Coverage, ci95Width, wallSeconds } = pyResult;
    const wallSec = wallSeconds.toFixed(1);

    // Yield + warm DB pool before writing (Python service may have held the connection idle)
    await new Promise(r => setTimeout(r, 50));
    try { await db.execute("SELECT 1"); } catch { /* pool will reconnect */ }

    if (pyResult.status === "discarded") {
      console.warn(`[GNN-GCP] Job #${jobId} discarded by Python service: ${pyResult.reason}`);
      await db.execute(`UPDATE gnn_training_jobs SET status = 'discarded' WHERE id = ${jobId}`);
    } else if (pyResult.status === "failed") {
      throw new Error(pyResult.reason ?? "Python service returned status=failed");
    } else {
      // PyTorch weights live in a .pt file on disk; TS DB stores metadata only.
      if (pyResult.modelPath) await recordPytorchModelPath(jobId, pyResult.modelPath);
      await storage.updateGnnTrainingJob(jobId, {
        status:      "done",
        weights:     [] as any,  // PyTorch weights stored in .pt file, not DB JSON
        r2,
        mae,
        rmse,
        trainR2,
        trainMae,
        valN,
        datasetSize: trainSet.length,
        completedAt: new Date(),
      } as any);

      // Cycle 1381: write XGB full-corpus metrics to DB (same as startup path)
      if (pyResult.xgbR2 != null) {
        try {
          const xgbJob = await storage.insertXgbTrainingJob({
            status: "queued" as any,
            featuresX: [] as any,
            labelsY: [] as any,
            datasetSize: pyResult.xgbNTrain ?? 0,
          });
          await storage.updateXgbTrainingJob(xgbJob.id, {
            status: "done",
            r2: pyResult.xgbR2,
            mae: pyResult.xgbMae ?? 0,

            completedAt: new Date(),
          } as any);
          console.log(`[GNN-GCP] XGB full-corpus metrics stored as job #${xgbJob.id} — R²=${pyResult.xgbR2.toFixed(4)}`);
        } catch (xgbErr: any) {
          console.warn(`[GNN-GCP] Failed to store XGB metrics: ${xgbErr.message?.slice(0, 80)}`);
        }
      }
    }

    // Write comprehensive metrics to system_state for frontend consumption
    await writeGnnMetricsToSystemState(jobId, {
      r2, mae, rmse, trainR2, trainMae, valN,
      datasetSize: trainSet.length,
      xgbR2: pyResult.xgbR2 ?? null,
      xgbMae: pyResult.xgbMae ?? null,
      wallSeconds: pyResult.wallSeconds,
      source: 'dispatched',
      ci95Coverage: ci95Coverage > 0 ? ci95Coverage : undefined,
      ci95Width: ci95Width > 0 ? ci95Width : undefined,
    });

    const coverageStr = ci95Coverage > 0
      ? `CI95-cov=${(ci95Coverage * 100).toFixed(1)}% width=${ci95Width.toFixed(1)}K`
      : "CI95-cov=N/A";

    console.log(
      `[GNN-GCP] Job #${jobId} complete in ${wallSec}s` +
      ` | VAL(n=${valN}): R²=${r2.toFixed(3)} MAE=${mae.toFixed(1)}K RMSE=${rmse.toFixed(1)}K` +
      ` | TRAIN(sample): R²=${trainR2.toFixed(3)} MAE=${trainMae.toFixed(1)}K` +
      ` | overfit-gap=${(trainR2 - r2).toFixed(3)}` +
      ` | ${coverageStr}`
    );

    if (ci95Coverage > 0) {
      if (ci95Coverage < 0.80) {
        console.warn(`[GNN-GCP] ⚠ CI95 under-coverage (${(ci95Coverage * 100).toFixed(1)}%) — ensemble uncertainty is underestimated`);
      } else if (ci95Coverage > 0.99) {
        console.warn(`[GNN-GCP] ⚠ CI95 over-coverage (${(ci95Coverage * 100).toFixed(1)}%) — ensemble intervals are too wide`);
      }
      await storeCalibrationMetric(jobId, ci95Coverage, ci95Width, valN);
    }

    // Fetch next MP batch after every successful training job so the next
    // cycle's training payload is richer. Cached in mp_material_cache — the
    // local server reads from there when building the next training payload.
    fetchNextMPBatch().catch(() => {});

  } catch (err: any) {
    const jobErrMsg = err instanceof Error ? (err.message || err.constructor?.name || "(empty Error)") : String(err ?? "unknown");
    console.error(`[GNN-GCP] Job #${jobId} failed: ${jobErrMsg}`);
    await storage.updateGnnTrainingJob(jobId, {
      status: "failed",
      errorMessage: err.message?.slice(0, 1000) ?? "unknown error",
      completedAt: new Date(),
    }).catch(() => {});
  }

  return true;
}

// ── Startup full corpus training ──────────────────────────────────────────────
// Runs once per GCP worker lifetime (guarded by system_state). Trains the GNN
// ensemble on the full NIMS + JARVIS SC corpus (~15k entries) plus QE/MP contrast.
// Subsequent dispatched retrains use only 5k samples so they finish in minutes.

const STARTUP_CORPUS_STATE_KEY = 'gnn_startup_corpus_training';
// Per-model checkpoint keys written to system_state as each worker completes.
// These let a restarted process skip models that already finished.
const STARTUP_CKPT_KEY_PREFIX = 'gnn_startup_ckpt_';
const STARTUP_CKPT_TTL_HOURS = 20; // checkpoints older than this are stale

// Increment this whenever the GNN forward pass formula changes in a way that
// makes old weights incompatible with the new code (e.g. hard-cap → sigmoid).
// When the stored version doesn't match, startup retrains from scratch.
const GNN_MODEL_VERSION = 14; // v14: hard BCE labels + v15 dedicated cls head (migrateWeights handles new fields — no retrain needed)

// In-process guard: run corpus training exactly once per GCP worker process lifetime.
let _startupCorpusRan = false;
// True from worker boot until startup corpus training either succeeds or exhausts
// its outer retry loop. While this is true AND _startupCorpusRan is false, the
// dispatched-job poll loop must NOT claim the GNN training slot — otherwise it
// races startup and locks startup out for the duration of the dispatched job
// (potentially hours). See cycle 1375 fix and runStartupFullCorpusTraining below.
let _startupCorpusActive = true;

// ── Startup checkpoint helpers ────────────────────────────────────────────────

/** Returns loaded ensemble weights if startup training completed within TTL, else null. */
async function checkStartupCompletedRecently(): Promise<GNNWeights[] | null> {
  try {
    const stateRows = await db.execute(
      `SELECT value FROM system_state WHERE key = '${STARTUP_CORPUS_STATE_KEY}'`
    );
    const stateRow = (stateRows as any).rows?.[0] ?? (Array.isArray(stateRows) ? stateRows[0] : undefined);
    if (!stateRow?.value) return null;
    const state = typeof stateRow.value === 'string' ? JSON.parse(stateRow.value) : stateRow.value;
    if (!state?.doneAt) return null;
    // Version check: if stored weights were trained with an older forward pass
    // formula, they're incompatible with the current code and must be retrained.
    if ((state.modelVersion ?? 1) !== GNN_MODEL_VERSION) {
      console.log(`[GNN-GCP] Startup weights are model version ${state.modelVersion ?? 1} but code requires v${GNN_MODEL_VERSION} — will retrain from scratch`);
      return null;
    }
    const ageHours = (Date.now() - new Date(state.doneAt).getTime()) / 3_600_000;
    if (ageHours > 8) {
      console.log(`[GNN-GCP] Startup training record is ${ageHours.toFixed(1)}h old — will retrain`);
      return null;
    }
    // Load weights from the most recent completed job
    const jobRows = await db.execute(
      `SELECT weights FROM gnn_training_jobs WHERE status = 'done' AND weights IS NOT NULL ORDER BY completed_at DESC LIMIT 1`
    );
    const jobRow = (jobRows as any).rows?.[0] ?? (Array.isArray(jobRows) ? jobRows[0] : undefined);
    if (!jobRow?.weights) return null;
    const weights = typeof jobRow.weights === 'string' ? JSON.parse(jobRow.weights) : jobRow.weights;
    if (!Array.isArray(weights) || weights.length === 0) return null;
    console.log(`[GNN-GCP] Startup training completed ${ageHours.toFixed(1)}h ago (v${GNN_MODEL_VERSION}) — restoring ${weights.length} models from DB`);
    return weights as GNNWeights[];
  } catch {
    return null;
  }
}

/**
 * Loads any per-model checkpoints saved from a previous interrupted run.
 * Returns a map of modelIndex → GNNWeights for models that already finished.
 */
async function loadStartupModelCheckpoints(): Promise<Map<number, GNNWeights>> {
  const result = new Map<number, GNNWeights>();
  const n = Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length);
  try {
    const keys = Array.from({ length: n }, (_, i) => `'${STARTUP_CKPT_KEY_PREFIX}${i}'`).join(', ');
    const rows = await db.execute(
      `SELECT key, value, updated_at FROM system_state WHERE key IN (${keys})`
    );
    const rowArr: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const cutoff = Date.now() - STARTUP_CKPT_TTL_HOURS * 3_600_000;
    for (const row of rowArr) {
      const updatedAt = new Date(row.updated_at ?? 0).getTime();
      if (updatedAt < cutoff) continue; // stale checkpoint
      const idx = parseInt(String(row.key).replace(STARTUP_CKPT_KEY_PREFIX, ''), 10);
      if (!Number.isFinite(idx)) continue;
      const model = typeof row.value === 'string' ? JSON.parse(row.value) : row.value;
      if (model && typeof model === 'object') {
        result.set(idx, model as GNNWeights);
        console.log(`[GNN-GCP] Restored checkpoint for model ${idx}`);
      }
    }
  } catch (err: any) {
    console.warn(`[GNN-GCP] Checkpoint load failed (non-fatal): ${err.message?.slice(0, 80)}`);
  }
  return result;
}

/** Saves a single trained model to system_state as a checkpoint. */
async function saveStartupModelCheckpoint(model: GNNWeights, modelIndex: number): Promise<void> {
  try {
    const key = `${STARTUP_CKPT_KEY_PREFIX}${modelIndex}`;
    // GNN weights are pure JSON numbers — no single-quote injection risk.
    const value = JSON.stringify(model);
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('${key}', '${value}'::jsonb, NOW())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`
    );
    console.log(`[GNN-GCP] Checkpointed model ${modelIndex} (${(value.length / 1024).toFixed(0)} KB)`);
  } catch (err: any) {
    console.warn(`[GNN-GCP] Checkpoint save failed for model ${modelIndex}: ${err.message?.slice(0, 80)}`);
  }
}

/** Clears per-model checkpoints after a full successful startup training. */
async function clearStartupModelCheckpoints(): Promise<void> {
  const n = Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length);
  try {
    const keys = Array.from({ length: n }, (_, i) => `'${STARTUP_CKPT_KEY_PREFIX}${i}'`).join(', ');
    await db.execute(`DELETE FROM system_state WHERE key IN (${keys})`);
  } catch { /* non-fatal */ }
}

async function runStartupFullCorpusTraining(): Promise<void> {
  // Skip if already ran SUCCESSFULLY in this process (set at the bottom).
  // The flag is intentionally NOT set up-front anymore — a transient failure
  // (e.g. Neon "Exceeded concurrency limit" during the first 30s rush) used to
  // permanently disable startup training for the worker's lifetime, leaving the
  // poll loop running but never actually training. Now the flag is only set on
  // a successful (or "discarded by quality gate") completion, so callers (the
  // outer retry loop in startGNNLoop) can re-enter on failure.
  if (_startupCorpusRan) {
    console.log('[GNN-GCP] Startup full corpus training already ran this session — skipping');
    return;
  }

  // ── Acquire the GNN training slot FIRST (cycle 1375 fix) ─────────────────
  // Before doing any DB loading. Otherwise the ~30-90s of corpus loading below
  // gives the poll loop (which starts at T+125s) a window to claim the slot
  // for a dispatched job, locking startup out for hours. The slot must be held
  // through the entire data load + python /train and released in the finally
  // block at the end of this function.
  await waitForXGBIdle();
  if (!acquireGNNTrainingSlot('STARTUP')) {
    // Slot is busy — only possible if a dispatched-job handler somehow grabbed
    // it despite _startupCorpusActive being true (shouldn't happen with the
    // poll-loop guard, but throw so the outer retry loop in startGNNLoop waits
    // and retries cleanly).
    throw new Error('GNN training slot busy at startup acquisition — outer retry will wait 5 min');
  }

  // Declared at function scope (cycle 1376 fix) so the finally block can DELETE
  // the placeholder by ID instead of doing a blanket `DELETE WHERE status IN
  // ('queued','running')` that would also nuke unrelated jobs from
  // active-learning.ts / gradient-boost.ts co-dispatches.
  let startupJobId: number | undefined = undefined;

  try {
  // ── Load partial checkpoints from a previous interrupted run ─────────────
  // Always run startup training on every GCP restart — the full corpus
  // training is the most reliable baseline and must not be skipped.
  // Checkpoints handle interrupted training; the "skip if recent" guard was
  // removed because it caused startup to skip when the GCP worker was
  // restarted after a code deployment.
  const checkpointedModels = await loadStartupModelCheckpoints();
  const allIndices = Array.from({ length: Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length) }, (_, i) => i);
  const indicesToTrain = allIndices.filter(i => !checkpointedModels.has(i));

  if (checkpointedModels.size > 0) {
    console.log(`[GNN-GCP] Found ${checkpointedModels.size} checkpointed model(s) — only training ${indicesToTrain.length} remaining`);
  }

  console.log('[GNN-GCP] ════════════════════════════════════════════════════════');
  console.log('[GNN-GCP]  STARTUP PHASE 3 — Full Corpus Training');
  console.log('[GNN-GCP]  Loading full SC corpus (NIMS + JARVIS + 3DSC — no limit)');
  console.log('[GNN-GCP] ════════════════════════════════════════════════════════');

  // Pre-warm DB connection before heavy queries
  try { await db.execute("SELECT 1"); } catch { await new Promise(r => setTimeout(r, 3000)); }

  const existingFormulas = new Set<string>();

  // Load full SC corpus from supercon_external_entries
  const externalSC = await loadSuperconExternalSamples(existingFormulas);
  externalSC.forEach(s => existingFormulas.add(s.formula));

  // Augment with DFT-verified QE entries
  const qeSamples = await loadQEDatasetSamples(existingFormulas);
  qeSamples.forEach(s => existingFormulas.add(s.formula));
  if (qeSamples.length > 0) console.log(`[GNN-GCP]  +${qeSamples.length} QE DFT-verified entries`);

  let trainingData: TrainingSample[] = [...externalSC, ...qeSamples];

  // Contrast examples: JARVIS metallic negatives + MP
  const scCount = trainingData.filter(s => s.tc > 0).length;
  const jarvisContrast = await loadJarvisDFT3DContrast(existingFormulas, scCount);
  jarvisContrast.forEach(s => existingFormulas.add(s.formula));
  const remainingContrast = Math.max(0, scCount - jarvisContrast.length);
  const mpContrast = await loadMPContrastSamples(existingFormulas, remainingContrast);
  trainingData = [...trainingData, ...jarvisContrast, ...mpContrast];
  if (jarvisContrast.length + mpContrast.length > 0) {
    console.log(`[GNN-GCP]  +${jarvisContrast.length} JARVIS metallic + ${mpContrast.length} MP contrast (Tc=0)`);
  }

  // Train/val split
  const scSamples = trainingData.filter(s => s.tc > 0);
  const nonScSamples = trainingData.filter(s => s.tc === 0);
  const { train: scTrain, validation: scVal } = splitTrainValidation(scSamples, 0.20, 42);
  const cappedNonSc = nonScSamples.slice(0, scTrain.length);
  const trainSet = [...scTrain, ...cappedNonSc];
  const valSet = scVal;


  console.log(`[GNN-GCP]  Dataset: ${scSamples.length} SC + ${cappedNonSc.length} contrast | train=${trainSet.length} val=${valSet.length}`);

  // ── Dataset sanity gate ────────────────────────────────────────────────
  // If every loader returned [] (e.g. Neon "Exceeded concurrency limit" during
  // the startup rush even after withDBRetry exhausted its attempts), we get a
  // tiny/empty dataset. Calling Python /train with this would either fail
  // outright or produce a worthless model that overrides the cached ensemble.
  // Throw so the outer retry loop in startGNNLoop can wait and try again.
  const MIN_SC_SAMPLES = 100;
  if (scSamples.length < MIN_SC_SAMPLES) {
    throw new Error(
      `Startup dataset too small: ${scSamples.length} SC samples (need ≥${MIN_SC_SAMPLES}). ` +
      `Likely cause: Neon DB connection cap hit during data fetching. ` +
      `Will retry after backoff.`
    );
  }

  console.log(`[GNN-GCP]  Delegating startup training to Python GNN service…`);

  let r2 = 0, mae = 0, rmse = 0, trainR2 = 0, trainMae = 0, valN = 0;
  let startupModelPath: string | undefined;
  let pyResult: Awaited<ReturnType<typeof callPythonTrain>> | undefined;

  // Slot was already acquired at the top of this function — just call Python /train
  // directly. Errors propagate to the outer retry loop in startGNNLoop. Cleanup
  // (delete placeholder + release slot) is in the top-level finally below.
  // Insert a placeholder job so the Python service result has a job ID to reference
  const placeholderJob = await storage.insertGnnTrainingJob({
    status: "running" as any,
    trainingData: [] as any,
    datasetSize: trainSet.length,
    dftSamples: qeSamples.length,
  });
  startupJobId = placeholderJob.id;

  pyResult = await callPythonTrain(startupJobId, trainingData, undefined, 15);

  // Cycle 1376 fix: throw on Python-reported failure so the outer retry loop
  // catches it and retries with backoff. Previously this path silently
  // destructured zero metrics and set _startupCorpusRan = true, marking
  // startup as "complete" even though training failed — outer retries never
  // fired and the worker fell through to idle polling forever. The companion
  // path in processGnnJob (line ~750) handles 'failed' the same way.
  if (pyResult?.status === "failed") {
    throw new Error(`Python /train returned status=failed: ${pyResult.reason ?? "(no reason)"}`);
  }

  ({ r2, mae, rmse, trainR2, trainMae, valN, startupModelPath } = {
    ...pyResult, startupModelPath: pyResult.modelPath,
  });

  const wallSec = pyResult ? pyResult.wallSeconds.toFixed(1) : "?";
  const trainMaeV = trainMae;

  console.log('[GNN-GCP] ────────────────────────────────────────────────────────');
  console.log(`[GNN-GCP]  Startup training complete in ${wallSec}s`);
  console.log(`[GNN-GCP]  VAL  (n=${valN}): R²=${r2.toFixed(3)}  MAE=${mae.toFixed(1)}K  RMSE=${rmse.toFixed(1)}K`);
  console.log(`[GNN-GCP]  TRAIN(sample): R²=${trainR2.toFixed(3)}  MAE=${trainMaeV.toFixed(1)}K  overfit-gap=${(trainR2 - r2).toFixed(3)}`);
  if (startupModelPath) console.log(`[GNN-GCP]  PyTorch ensemble saved: ${startupModelPath}`);
  console.log('[GNN-GCP] ════════════════════════════════════════════════════════');

  if (pyResult?.status === "discarded") {
    console.warn(`[GNN-GCP] Startup weights NOT stored — ${pyResult.reason}`);
  } else if (pyResult?.status === "done") {
    // Record metrics in DB — update the placeholder job instead of creating a new one
    try {
      await db.execute("SELECT 1");
      // Update the placeholder job that was created at line 1214 (startupJobId)
      // instead of inserting a duplicate. This ensures the DB id matches what
      // the Python service logged as Job#<startupJobId>.
      const targetJobId = startupJobId ?? 0;
      if (targetJobId > 0) {
        await storage.updateGnnTrainingJob(targetJobId, {
          status: "done",
          weights: [] as any,   // PyTorch weights live in .pt file on disk
          r2, mae, rmse,
          trainR2,
          trainMae,
          valN,
          datasetSize: trainSet.length,
          completedAt: new Date(),
        } as any);
        if (startupModelPath) await recordPytorchModelPath(targetJobId, startupModelPath);
        console.log(`[GNN-GCP] Startup results written to DB as job #${targetJobId} — R²=${r2.toFixed(4)} MAE=${mae.toFixed(1)}K`);
      } else {
        // Fallback: insert new job if no placeholder exists
        const insertedJob = await storage.insertGnnTrainingJob({
          status: "queued" as any,
          trainingData: [] as any,
          datasetSize: trainSet.length,
          dftSamples: qeSamples.length,
        });
        await storage.updateGnnTrainingJob(insertedJob.id, {
          status: "done",
          weights: [] as any,
          r2, mae, rmse,
          trainR2,
          trainMae,
          valN,
          completedAt: new Date(),
        } as any);
        if (startupModelPath) await recordPytorchModelPath(insertedJob.id, startupModelPath);
        console.log(`[GNN-GCP] Startup results written to DB as new job #${insertedJob.id} — R²=${r2.toFixed(4)} MAE=${mae.toFixed(1)}K`);
      }
      // Write comprehensive metrics to system_state for frontend consumption
      await writeGnnMetricsToSystemState(targetJobId || 0, {
        r2, mae, rmse, trainR2, trainMae, valN,
        datasetSize: trainSet.length,
        xgbR2: pyResult?.xgbR2 ?? null,
        xgbMae: pyResult?.xgbMae ?? null,
        wallSeconds: pyResult?.wallSeconds ?? 0,
        source: 'startup',
      });

      // Cycle 1381: write XGB metrics to xgb_training_jobs so the local
      // server's startGCPXGBPoller picks up the full-corpus model (R²≈0.91).
      // Without this, only the small-pool dispatched XGB jobs (R²≈0.43) were
      // visible to the local server — the full-corpus XGB from _train_xgboost
      // was saved to GCP disk only and never flowed back.
      if (pyResult?.xgbR2 != null) {
        try {
          const xgbJob = await storage.insertXgbTrainingJob({
            status: "queued" as any,
            featuresX: [] as any,
            labelsY: [] as any,
            datasetSize: pyResult.xgbNTrain ?? 0,
          });
          await storage.updateXgbTrainingJob(xgbJob.id, {
            status: "done",
            r2: pyResult.xgbR2,
            mae: pyResult.xgbMae ?? 0,

            completedAt: new Date(),
          } as any);
          console.log(`[GNN-GCP] XGB full-corpus metrics stored as job #${xgbJob.id} — R²=${pyResult.xgbR2.toFixed(4)}`);
        } catch (xgbErr: any) {
          console.warn(`[GNN-GCP] Failed to store XGB metrics: ${xgbErr.message?.slice(0, 120)}`);
        }
      }
    } catch (err: any) {
      console.warn(`[GNN-GCP] Failed to store startup metrics in DB: ${err.message?.slice(0, 120)}`);
    }
  }

  // Mark startup complete in system_state and clear per-model checkpoints
  try {
    const nowIso = new Date().toISOString();
    await db.execute(
      `INSERT INTO system_state (key, value, updated_at)
       VALUES ('${STARTUP_CORPUS_STATE_KEY}', '{"doneAt":"${nowIso}","trainN":${trainSet.length},"valR2":${r2.toFixed(4)},"modelVersion":${GNN_MODEL_VERSION}}', NOW())
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`
    );
  } catch { /* non-fatal */ }
  // Per-model checkpoints are superseded by the full job stored above.
  await clearStartupModelCheckpoints();

  // Only set the in-process "ran" flag after a successful (or quality-discarded)
  // training run. This must be the last line of the success path so transient
  // failures earlier in the function are eligible for retry by the outer loop.
  _startupCorpusRan = true;
  } finally {
    // Cleanup: only delete the placeholder if it was NOT updated to 'done'.
    // If the success path above updated it to 'done', keep it — it has the metrics.
    // Only delete if training failed/was discarded and the job is still 'running'.
    if (startupJobId !== undefined) {
      try {
        await db.execute(
          `DELETE FROM gnn_training_jobs WHERE id = ${startupJobId} AND status = 'running'`
        );
      } catch { /* non-fatal */ }
    }
    releaseGNNTrainingSlot('STARTUP');
  }
}

// ── Main loop ─────────────────────────────────────────────────────────────────

export async function startGNNLoop(): Promise<void> {
  console.log(`[GNN-GCP] GNN training worker started — poll every ${POLL_INTERVAL_MS / 1000}s`);
  console.log(`[GNN-GCP] Progressive MP fetch: ${MP_BATCH_SIZE} records/cycle, max cache ${MP_MAX_CACHE}`);

  // ── Start Python GNN service ──────────────────────────────────────────────
  console.log("[GNN-GCP] Starting Python GNN service (FastAPI)…");
  try {
    await spawnPythonGNNService();
  } catch (err: any) {
    console.error(`[GNN-GCP] FATAL: Could not start Python GNN service: ${err.message}`);
    console.error("[GNN-GCP] Ensure gnn/server.py dependencies are installed:");
    console.error("  pip install fastapi uvicorn torch torch-geometric");
    throw err;
  }
  console.log("[GNN-GCP] Python GNN service ready on " + GNN_SERVICE_URL);

  // Reset any jobs left in 'running' state from a previous crashed/restarted process.
  // Without this, the local server sees an active job and won't dispatch new ones.
  try {
    const reset = await db.execute(
      `UPDATE gnn_training_jobs SET status = 'queued', started_at = NULL
       WHERE status = 'running'
         AND (started_at IS NULL OR started_at < NOW() - INTERVAL '5 minutes')
       RETURNING id`
    );
    const resetIds: number[] = ((reset as any).rows ?? (Array.isArray(reset) ? reset : [])).map((r: any) => r.id);
    if (resetIds.length > 0) {
      console.log(`[GNN-GCP] Reset ${resetIds.length} stale running job(s) back to queued: #${resetIds.join(', #')}`);
    }
  } catch (err: any) {
    console.warn(`[GNN-GCP] Stale job reset failed (non-fatal): ${err.message?.slice(0, 80)}`);
  }

  // Phase 3 startup: train on full NIMS+JARVIS corpus as a background task.
  // Delayed 120s (was 30s) so SuperCon/COD/JARVIS/3DSC ingestion + DFT/ML/XGB
  // poll loops can claim their initial DB connections and settle. Hitting Neon's
  // pgBouncer with a 5-loader cascade at the same instant as ingestion's batch
  // inserts is what triggered the "Exceeded concurrency limit" cap and made
  // every loader silently return [] → empty training data → tiny model.
  //
  // Wrapped in a retry loop so a transient DB cap during the first attempt
  // doesn't permanently disable startup training for the worker's lifetime.
  // Each failed attempt waits 5 minutes before retrying so the connection
  // pressure has time to subside.
  void (async () => {
    try {
      await new Promise(r => setTimeout(r, 120_000));
      const STARTUP_MAX_ATTEMPTS = 4;
      for (let attempt = 1; attempt <= STARTUP_MAX_ATTEMPTS; attempt++) {
        try {
          await runStartupFullCorpusTraining();
          if (_startupCorpusRan) return; // success — flag is set at end of function
          console.warn(
            `[GNN-GCP] Startup attempt ${attempt}/${STARTUP_MAX_ATTEMPTS} returned without setting ran flag — likely deferred. Retrying.`
          );
        } catch (err: any) {
          console.error(
            `[GNN-GCP] Startup attempt ${attempt}/${STARTUP_MAX_ATTEMPTS} failed: ${err?.message ?? String(err)}`
          );
        }
        if (attempt < STARTUP_MAX_ATTEMPTS) {
          const waitMs = 5 * 60_000;
          console.log(`[GNN-GCP] Waiting ${waitMs / 1000}s before retrying startup corpus training…`);
          await new Promise(r => setTimeout(r, waitMs));
        }
      }
      console.error(
        `[GNN-GCP] Startup corpus training gave up after ${STARTUP_MAX_ATTEMPTS} attempts. ` +
        `Worker will continue polling for dispatched jobs but will not have a fresh full-corpus baseline.`
      );
    } finally {
      // Cycle 1375 fix: clear the active flag whether startup succeeded, failed,
      // or exhausted retries. Once cleared, the dispatched-job poll loop is free
      // to claim the GNN slot. Without this finally, a successful startup never
      // clears the flag and dispatched jobs would be locked out forever after
      // startup completes — _startupCorpusActive would stay true.
      _startupCorpusActive = false;
      console.log('[GNN-GCP] Startup corpus phase complete — dispatched-job poll loop unblocked');
    }
  })();

  // Delay job polling by 125s so the startup corpus training acquires the slot first.
  // Was 35s — increased to match the new 120s startup delay so the poll loop
  // doesn't claim jobs before startup training has had a chance to run.
  await new Promise(r => setTimeout(r, 125_000));
  console.log("[GNN-GCP] Job polling starting — startup training should hold the slot");

  while (running) {
    try {
      const processed = await processNextGnnJob();
      await new Promise(r => setTimeout(r, processed ? 1000 : POLL_INTERVAL_MS));
    } catch (err: any) {
      const msg = err instanceof Error
        ? (err.stack || err.message || err.constructor?.name || "(empty Error)")
        : (err != null ? String(err) : "unknown");
      console.error(`[GNN-GCP] Loop error (${err?.constructor?.name ?? typeof err}): ${msg || "(no message)"}`);
      await new Promise(r => setTimeout(r, isConnectionError(err) ? 5000 : POLL_INTERVAL_MS));
    }
  }
}

export function stopGNNLoop() {
  running = false;
  if (_pythonProcess && !_pythonProcess.killed) {
    console.log("[GNN-GCP] Stopping Python GNN service…");
    _pythonProcess.kill("SIGTERM");
  }
}
