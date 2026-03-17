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
import { Worker } from "worker_threads";
import { fileURLToPath } from "url";
import path from "path";
import {
  trainEnsemble, GNNPredict, buildCrystalGraph,
  ENSEMBLE_SIZE, ENSEMBLE_SEEDS, BOOTSTRAP_RATIOS,
  splitTrainValidation,
} from "../server/learning/graph-neural-net";
import type { TrainingSample } from "../server/learning/graph-neural-net";
import type { GNNWeights } from "../server/learning/graph-neural-net";
import { fetchMPBatchFromAPI } from "../server/learning/materials-project-client";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WORKER_SCRIPT = path.join(__dirname, "gnn-worker-thread.ts");

// ── Parallel ensemble training via worker threads ────────────────────────────
// Each of the 5 ensemble models trains in its own worker thread so the full
// ensemble is trained in the time it takes to train one model sequentially.
//
// TypeScript loading in workers: tsx may start the main process via NODE_OPTIONS,
// internal patching, or other means that do NOT add flags to process.execArgv.
// Passing execArgv flags is therefore unreliable. Instead we use eval:true with
// a tiny CJS bootstrap string that calls require('tsx/cjs') then loads the worker
// script. Plain .js eval content has no extension problem; tsx/cjs then handles
// the subsequent require() of the .ts file.

function spawnModelWorker(
  trainingData: TrainingSample[],
  modelIndex: number,
  maxPretrainEpochs: number,
): Promise<GNNWeights> {
  // Inline CJS bootstrap: enable tsx TypeScript loading, then run the worker.
  // Uses eval:true so Node sees a plain JS string — no ".ts extension" rejection.
  const bootstrapCode = `
require('tsx/cjs');
require(${JSON.stringify(WORKER_SCRIPT)});
`;
  return new Promise((resolve, reject) => {
    const worker = new Worker(bootstrapCode, {
      eval: true,
      workerData: {
        trainingData,
        seed: ENSEMBLE_SEEDS[modelIndex],
        bootstrapRatio: BOOTSTRAP_RATIOS[modelIndex],
        maxPretrainEpochs,
        modelIndex,
      },
    });
    worker.once("message", (msg: { ok: boolean; model?: GNNWeights; error?: string }) => {
      if (msg.ok && msg.model) {
        resolve(msg.model);
      } else {
        reject(new Error(msg.error ?? `Worker ${modelIndex} returned no model`));
      }
    });
    worker.once("error", reject);
    worker.once("exit", (code) => {
      if (code !== 0) reject(new Error(`Worker ${modelIndex} exited with code ${code}`));
    });
  });
}

async function trainEnsembleParallel(
  trainingData: TrainingSample[],
  maxPretrainEpochs = 15,
): Promise<GNNWeights[]> {
  const n = Math.min(ENSEMBLE_SIZE, ENSEMBLE_SEEDS.length);
  console.log(`[GNN-GCP] Launching ${n} worker threads for parallel ensemble training`);

  // Fan out: all n models start simultaneously
  const promises = Array.from({ length: n }, (_, i) =>
    spawnModelWorker(trainingData, i, maxPretrainEpochs).catch((err: Error) => {
      console.warn(`[GNN-GCP] Worker ${i} failed (${err.message}) — will use sequential fallback for this model`);
      return null;
    }),
  );

  const results = await Promise.all(promises);
  const models = results.filter((m): m is GNNWeights => m !== null);

  if (models.length === 0) {
    console.warn("[GNN-GCP] All worker threads failed — falling back to sequential trainEnsemble");
    return trainEnsemble(trainingData);
  }

  if (models.length < n) {
    console.warn(`[GNN-GCP] Only ${models.length}/${n} workers succeeded — ensemble will be smaller`);
  }

  return models;
}

const POLL_INTERVAL_MS = 10_000;
const MP_BATCH_SIZE = 500;          // records per progressive fetch
const MP_MAX_CACHE = 10_000;        // stop fetching after this many cached records
const MP_SKIP_STATE_KEY = "mp_batch_skip";
let running = true;

// In-memory skip counter — primary source of truth. DB write is best-effort.
let _mpSkip = 0;
let _mpSkipLoaded = false;
// Mutex: prevents concurrent GNN job completions from each firing a duplicate MP fetch.
let _mpFetchInProgress = false;

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

// ── Metrics ──────────────────────────────────────────────────────────────────
// IMPORTANT: always pass a HELD-OUT split — never the training set.
// Evaluating on training data gives inflated R² that is meaningless for
// assessing generalisation quality.

function computeMetrics(
  models: GNNWeights[],
  evalData: TrainingSample[],
): { r2: number; mae: number; rmse: number; n: number } {
  if (models.length === 0 || evalData.length === 0) return { r2: 0, mae: 0, rmse: 0, n: 0 };

  const actuals = evalData.map(d => d.tc);
  const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length;

  let ssTot = 0, ssRes = 0, sumAbs = 0, sumSq = 0, counted = 0;
  for (const sample of evalData) {
    try {
      const graph = buildCrystalGraph(sample.formula, sample.structure);
      const preds = models.map(m => GNNPredict(graph, m).predictedTc);
      const pred = preds.reduce((a, b) => a + b, 0) / preds.length;
      if (!Number.isFinite(pred)) continue;
      const diff = sample.tc - pred;
      ssTot += (sample.tc - meanActual) ** 2;
      ssRes += diff ** 2;
      sumAbs += Math.abs(diff);
      sumSq += diff ** 2;
      counted++;
    } catch { /* skip malformed samples */ }
  }

  if (counted === 0) return { r2: 0, mae: 0, rmse: 0, n: 0 };
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  const mae = sumAbs / counted;
  const rmse = Math.sqrt(sumSq / counted);
  return { r2, mae, rmse, n: counted };
}

// ── CI95 calibration check ────────────────────────────────────────────────────
// A well-calibrated model should have ~95% of true Tc values fall inside its
// predicted 95% confidence interval. We check this empirically on the held-out
// SC validation set and log it after every training job.

function computeCalibration(
  models: GNNWeights[],
  valData: TrainingSample[],
): { coverage: number; meanWidth: number; n: number } {
  if (models.length === 0 || valData.length === 0) return { coverage: 0, meanWidth: 0, n: 0 };

  let inside = 0, totalWidth = 0, counted = 0;
  for (const sample of valData) {
    try {
      const graph = buildCrystalGraph(sample.formula, sample.structure);
      const preds = models.map(m => GNNPredict(graph, m).predictedTc);
      const meanTc = preds.reduce((a, b) => a + b, 0) / preds.length;

      // Ensemble variance (epistemic uncertainty)
      const variance = preds.reduce((s, p) => s + (p - meanTc) ** 2, 0) / preds.length;
      const ensembleStd = Math.sqrt(variance);

      // CI95: mean ± 1.96σ  (Gaussian approximation)
      const lo = meanTc - 1.96 * ensembleStd;
      const hi = meanTc + 1.96 * ensembleStd;

      if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
      if (sample.tc >= lo && sample.tc <= hi) inside++;
      totalWidth += hi - lo;
      counted++;
    } catch { /* skip malformed samples */ }
  }

  if (counted === 0) return { coverage: 0, meanWidth: 0, n: 0 };
  return {
    coverage: inside / counted,   // empirical fraction — should be ~0.95
    meanWidth: totalWidth / counted,
    n: counted,
  };
}

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

// ── QE dataset augmentation (DFT-verified SC samples) ────────────────────────

async function loadQEDatasetSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    const rows = await db.execute(
      `SELECT material, tc, formation_energy, band_gap, lambda
       FROM quantum_engine_dataset
       WHERE scf_converged = true AND tc > 0 AND tier IN ('full-dft', 'xtb')
       ORDER BY tc DESC
       LIMIT 5000`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      if (!row.material || existingFormulas.has(row.material)) continue;
      samples.push({
        formula: row.material,
        tc: Number(row.tc) || 0,
        formationEnergy: row.formation_energy != null ? Number(row.formation_energy) : undefined,
        structure: undefined,
        prototype: undefined,
      });
    }
    return samples;
  } catch {
    return [];
  }
}

// ── MP augmentation (Tc=0 contrast examples) ─────────────────────────────────

async function loadMPContrastSamples(existingFormulas: Set<string>, scCount: number): Promise<TrainingSample[]> {
  try {
    const rows = await db.execute(
      `SELECT formula, data FROM mp_material_cache WHERE data_type = 'summary' LIMIT 2000`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const maxContrast = scCount;  // 1:1 cap — avoids 60/40 Tc=0 majority diluting SC signal
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
      samples.push({ formula, tc: 0, formationEnergy: d.formationEnergyPerAtom ?? undefined, structure: undefined, prototype: undefined });
    }
    return samples;
  } catch {
    return [];
  }
}

// ── Job processing ────────────────────────────────────────────────────────────

async function processNextGnnJob(): Promise<boolean> {
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

  // Augment with DFT-verified SC entries from quantum_engine_dataset
  const qeSamples = await loadQEDatasetSamples(existingFormulas);
  if (qeSamples.length > 0) {
    trainingData = [...trainingData, ...qeSamples];
    qeSamples.forEach(s => existingFormulas.add(s.formula));
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${qeSamples.length} QE dataset SC entries — total ${trainingData.length}`);
  }

  // Augment with MP metallic materials (Tc=0) so GNN learns contrast with SCs
  const scCount = trainingData.filter(s => s.tc > 0).length;
  const mpContrast = await loadMPContrastSamples(existingFormulas, scCount);
  if (mpContrast.length > 0) {
    trainingData = [...trainingData, ...mpContrast];
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${mpContrast.length} MP contrast samples (Tc=0) — total ${trainingData.length}`);
  }

  // Split BEFORE training so validation data is never seen by the models.
  // SC samples (Tc>0) are stratified: 80% train / 20% val.
  // MP contrast (Tc=0) goes entirely into training — we validate on SC targets only.
  const scSamples  = trainingData.filter(s => s.tc > 0);
  const nonScSamples = trainingData.filter(s => s.tc === 0);
  const { train: scTrain, validation: scVal } = splitTrainValidation(scSamples, 0.20, 42);
  const trainSet = [...scTrain, ...nonScSamples];
  const valSet   = scVal; // held-out SC samples only — these are the discovery targets

  console.log(
    `[GNN-GCP] Starting training job #${jobId} — ${datasetSize} seed + ${qeSamples.length} QE + ${mpContrast.length} MP contrast` +
    ` | train=${trainSet.length} val=${valSet.length} (SC holdout)`
  );
  const startMs = Date.now();

  try {
    const models = await trainEnsembleParallel(trainSet);

    // Evaluate on HELD-OUT validation set — these R²/MAE/RMSE are honest.
    const valMetrics = valSet.length >= 5
      ? computeMetrics(models, valSet)
      : { r2: 0, mae: 0, rmse: 0, n: 0 };
    // Also compute training-set metrics separately so overfitting is visible.
    const trainMetrics = computeMetrics(models, scTrain.slice(0, 200)); // sample to avoid slow eval
    // CI95 calibration: empirical fraction of true Tc values inside the predicted interval.
    // A well-calibrated ensemble should yield ~0.95; values << 0.95 mean intervals are too narrow.
    const calibration = valSet.length >= 5
      ? computeCalibration(models, valSet)
      : { coverage: 0, meanWidth: 0, n: 0 };
    const { r2, mae, rmse, n: valN } = valMetrics;
    const wallSec = ((Date.now() - startMs) / 1000).toFixed(1);

    // trainEnsembleParallel() uses worker threads and awaits them, so the event
    // loop stays responsive throughout. Still yield briefly and warm the DB pool
    // before writing since worker threads may have let idle connections go stale.
    await new Promise(r => setTimeout(r, 50));
    try { await db.execute("SELECT 1"); } catch { /* ignore — pool will reconnect */ }

    await storage.updateGnnTrainingJob(jobId, {
      status: "done",
      weights: models as any,
      r2,                           // validation R²  (honest — 20% held-out SC)
      mae,                          // validation MAE (K)
      rmse,                         // validation RMSE (K)
      trainR2: trainMetrics.r2,     // training R² — compare to val R² for overfitting
      trainMae: trainMetrics.mae,
      valN,                         // held-out sample count
      completedAt: new Date(),
    } as any);

    const coverageStr = calibration.n > 0
      ? `CI95-cov=${(calibration.coverage * 100).toFixed(1)}% width=${calibration.meanWidth.toFixed(1)}K`
      : "CI95-cov=N/A";

    console.log(
      `[GNN-GCP] Job #${jobId} complete in ${wallSec}s` +
      ` | VAL(n=${valN}): R²=${r2.toFixed(3)} MAE=${mae.toFixed(1)}K RMSE=${rmse.toFixed(1)}K` +
      ` | TRAIN(sample): R²=${trainMetrics.r2.toFixed(3)} MAE=${trainMetrics.mae.toFixed(1)}K` +
      ` | overfit-gap=${(trainMetrics.r2 - r2).toFixed(3)}` +
      ` | ${coverageStr}`
    );

    if (calibration.n > 0) {
      if (calibration.coverage < 0.80) {
        console.warn(`[GNN-GCP] ⚠ CI95 under-coverage (${(calibration.coverage * 100).toFixed(1)}%) — ensemble uncertainty is underestimated`);
      } else if (calibration.coverage > 0.99) {
        console.warn(`[GNN-GCP] ⚠ CI95 over-coverage (${(calibration.coverage * 100).toFixed(1)}%) — ensemble intervals are too wide`);
      }
      await storeCalibrationMetric(jobId, calibration.coverage, calibration.meanWidth, calibration.n);
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

// ── Main loop ─────────────────────────────────────────────────────────────────

export async function startGNNLoop(): Promise<void> {
  console.log(`[GNN-GCP] GNN training worker started — poll every ${POLL_INTERVAL_MS / 1000}s`);
  console.log(`[GNN-GCP] Progressive MP fetch: ${MP_BATCH_SIZE} records/cycle, max cache ${MP_MAX_CACHE}`);

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
}
