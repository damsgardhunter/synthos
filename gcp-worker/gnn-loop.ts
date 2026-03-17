/**
 * GCP GNN Training Loop
 * Polls gnn_training_jobs for queued jobs, trains the full GNN ensemble,
 * writes weights back, then fetches the next batch of MP data so each
 * subsequent training cycle has a progressively richer dataset.
 */
import { db } from "../server/db";
import { storage } from "../server/storage";
import { trainEnsemble, GNNPredict, buildCrystalGraph } from "../server/learning/graph-neural-net";
import type { TrainingSample } from "../server/learning/graph-neural-net";
import { fetchMPBatchFromAPI } from "../server/learning/materials-project-client";

const POLL_INTERVAL_MS = 10_000;
const MP_BATCH_SIZE = 500;          // records per progressive fetch
const MP_MAX_CACHE = 10_000;        // stop fetching after this many cached records
const MP_SKIP_STATE_KEY = "mp_batch_skip";
let running = true;

// In-memory skip counter — primary source of truth. DB write is best-effort.
let _mpSkip = 0;
let _mpSkipLoaded = false;

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
  }
}

// ── Metrics ──────────────────────────────────────────────────────────────────

function computeMetrics(
  models: any[],
  trainingData: TrainingSample[],
): { r2: number; mae: number; rmse: number } {
  if (models.length === 0 || trainingData.length === 0) return { r2: 0, mae: 0, rmse: 0 };

  const actuals = trainingData.map(d => d.tc);
  const meanActual = actuals.reduce((a, b) => a + b, 0) / actuals.length;

  let ssTot = 0, ssRes = 0, sumAbs = 0, sumSq = 0;
  for (const sample of trainingData) {
    try {
      const graph = buildCrystalGraph(sample.formula, sample.structure);
      const preds = models.map(m => GNNPredict(graph, m).predictedTc);
      const pred = preds.reduce((a, b) => a + b, 0) / preds.length;
      const diff = sample.tc - pred;
      ssTot += (sample.tc - meanActual) ** 2;
      ssRes += diff ** 2;
      sumAbs += Math.abs(diff);
      sumSq += diff ** 2;
    } catch { /* skip malformed samples */ }
  }

  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  const mae = sumAbs / trainingData.length;
  const rmse = Math.sqrt(sumSq / trainingData.length);
  return { r2, mae, rmse };
}

// ── MP augmentation (Tc=0 contrast examples) ─────────────────────────────────

async function loadMPContrastSamples(existingFormulas: Set<string>): Promise<TrainingSample[]> {
  try {
    const rows = await db.execute(
      `SELECT formula, data FROM mp_material_cache WHERE data_type = 'summary' LIMIT 2000`
    );
    const items: any[] = (rows as any).rows ?? (Array.isArray(rows) ? rows : []);
    const samples: TrainingSample[] = [];
    for (const row of items) {
      const formula = row.formula as string;
      if (!formula || existingFormulas.has(formula)) continue;
      const d = row.data as any;
      if (!d) continue;
      // Only use metallic materials with near-zero band gap as Tc=0 contrast examples
      if ((d.bandGap ?? 1) > 0.1) continue;
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

  // Augment with MP metallic materials (Tc=0) so GNN learns contrast with SCs
  const existingFormulas = new Set(trainingData.map(s => s.formula));
  const mpContrast = await loadMPContrastSamples(existingFormulas);
  if (mpContrast.length > 0) {
    trainingData = [...trainingData, ...mpContrast];
    console.log(`[GNN-GCP] Augmented job #${jobId} with ${mpContrast.length} MP contrast samples (Tc=0) — total ${trainingData.length}`);
  }

  console.log(`[GNN-GCP] Starting training job #${jobId} — ${datasetSize} SC samples + ${mpContrast.length} MP contrast`);
  const startMs = Date.now();

  try {
    const models = trainEnsemble(trainingData);
    const { r2, mae, rmse } = computeMetrics(models, trainingData);
    const wallSec = ((Date.now() - startMs) / 1000).toFixed(1);

    await storage.updateGnnTrainingJob(jobId, {
      status: "done",
      weights: models as any,
      r2,
      mae,
      rmse,
      completedAt: new Date(),
    });

    console.log(
      `[GNN-GCP] Job #${jobId} complete in ${wallSec}s — R²=${r2.toFixed(3)}, MAE=${mae.toFixed(2)}, RMSE=${rmse.toFixed(2)}, N=${datasetSize}`
    );

    // Fetch next MP batch after every successful training job so the next
    // cycle's training payload is richer. Cached in mp_material_cache — the
    // local server reads from there when building the next training payload.
    fetchNextMPBatch().catch(() => {});

  } catch (err: any) {
    console.error(`[GNN-GCP] Job #${jobId} failed: ${err.message}`);
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
      console.error(`[GNN-GCP] Loop error: ${err.message}`);
      await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
    }
  }
}

export function stopGNNLoop() {
  running = false;
}
