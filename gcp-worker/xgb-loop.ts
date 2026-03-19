/**
 * GCP XGBoost Training Loop
 * Polls xgb_training_jobs for queued jobs, trains the full XGBoost ensemble
 * (base model + mean ensemble + variance ensemble), and writes the serialized
 * models back to the DB. The local server polls for completed jobs and applies
 * the weights without any local training overhead.
 */
import { db, isConnectionError } from "../server/db";
import { storage } from "../server/storage";
import { trainGradientBoosting } from "../server/learning/gradient-boost";
import { isGNNMajorTrainingActive, setXGBTrainingActive } from "./training-priority";

const POLL_INTERVAL_MS = 15_000;
let running = true;

// XGB hyperparameters — match what the local server uses
const N_TREES_BASE = 300;
const LR_BASE = 0.05;
const DEPTH_BASE = 6;
const ENSEMBLE_SIZE = 5;
const BOOTSTRAP_RATIO = 0.8;

function bootstrapSample(X: number[][], y: number[]): { bX: number[][]; by: number[] } {
  const n = Math.floor(X.length * BOOTSTRAP_RATIO);
  const bX: number[][] = [];
  const by: number[] = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(Math.random() * X.length);
    bX.push(X[idx]);
    by.push(y[idx]);
  }
  return { bX, by };
}

async function trainEnsemble(X: number[][], y: number[]) {
  const models = [];
  for (let i = 0; i < ENSEMBLE_SIZE; i++) {
    const { bX, by } = bootstrapSample(X, y);
    const m = await trainGradientBoosting(bX, by, N_TREES_BASE, LR_BASE, DEPTH_BASE);
    models.push(m);
    console.log(`[XGB-GCP] Mean ensemble model ${i + 1}/${ENSEMBLE_SIZE} done`);
  }
  return { models, trainedAt: Date.now(), isLogVariance: false };
}

// Mirror of predictFlat + predictWithModel from gradient-boost.ts.
// FlatTree format: { nodes: {featureIndex, threshold, leftChild, rightChild}[], leafValues: number[] }
// Leaf nodes use negative indices: leafValues[-(idx+1)]
function rawTreePredict(m: any, row: number[]): number {
  const px: number[] = m.featureMask ? m.featureMask.map((fi: number) => row[fi] ?? 0) : row;
  let pred = m.basePrediction;
  if (m.flatTrees && m.flatTrees.length > 0) {
    for (const flat of m.flatTrees) {
      if (!flat.nodes || flat.nodes.length === 0) continue;
      let idx = 0;
      while (idx >= 0) {
        const node = flat.nodes[idx];
        if (!node) break;
        idx = (px[node.featureIndex] ?? 0) <= node.threshold ? node.leftChild : node.rightChild;
      }
      const treeVal = flat.leafValues[-(idx + 1)];
      if (!Number.isFinite(treeVal)) continue;
      pred += m.learningRate * treeVal;
    }
  }
  return m.logTransformed ? Math.max(0, Math.expm1(pred)) : Math.max(0, pred);
}

async function trainVarianceEnsemble(X: number[][], y: number[], meanEnsemble: any) {
  // Predict mean for each sample, compute residuals, fit variance ensemble on log(residuals²)
  const meanPreds = X.map(row => {
    const preds = meanEnsemble.models.map((m: any) => rawTreePredict(m, row));
    return preds.reduce((a: number, b: number) => a + b, 0) / preds.length;
  });

  const logResiduals = y.map((actual, i) => Math.log(Math.max((actual - meanPreds[i]) ** 2, 1e-6)));
  const models = [];
  for (let i = 0; i < ENSEMBLE_SIZE; i++) {
    const { bX, by } = bootstrapSample(X, logResiduals);
    const m = await trainGradientBoosting(bX, by, Math.floor(N_TREES_BASE * 0.7), LR_BASE, DEPTH_BASE);
    models.push(m);
    console.log(`[XGB-GCP] Variance ensemble model ${i + 1}/${ENSEMBLE_SIZE} done`);
  }
  return { models, trainedAt: Date.now(), isLogVariance: true };
}

function computeMetrics(ensemble: any, X: number[][], y: number[]): { r2: number; mae: number } {
  if (!ensemble || X.length === 0) return { r2: 0, mae: 0 };
  const meanActual = y.reduce((a, b) => a + b, 0) / y.length;
  let ssTot = 0, ssRes = 0, sumAbs = 0;
  for (let i = 0; i < X.length; i++) {
    const preds = ensemble.models.map((m: any) => rawTreePredict(m, X[i]));
    const pred = preds.reduce((a: number, b: number) => a + b, 0) / preds.length;
    ssTot += (y[i] - meanActual) ** 2;
    ssRes += (y[i] - pred) ** 2;
    sumAbs += Math.abs(y[i] - pred);
  }
  return {
    r2: ssTot > 0 ? 1 - ssRes / ssTot : 0,
    mae: sumAbs / y.length,
  };
}

async function processNextXgbJob(): Promise<boolean> {
  if (isGNNMajorTrainingActive()) return false; // yield CPU to GNN training

  const rows = await db.execute(
    `UPDATE xgb_training_jobs
     SET status = 'running', started_at = NOW()
     WHERE id = (
       SELECT id FROM xgb_training_jobs
       WHERE status = 'queued'
       ORDER BY created_at ASC
       LIMIT 1
       FOR UPDATE SKIP LOCKED
     )
     RETURNING id, features_x, labels_y, dataset_size`
  );

  const job = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
  if (!job) return false;

  const jobId: number = job.id;
  const X: number[][] = job.features_x as number[][];
  const y: number[] = job.labels_y as number[];
  const datasetSize = job.dataset_size ?? X.length;

  console.log(`[XGB-GCP] Starting XGBoost training job #${jobId} — ${datasetSize} samples`);
  const startMs = Date.now();
  setXGBTrainingActive(true);

  try {
    console.log(`[XGB-GCP] Training base model (${N_TREES_BASE} trees)...`);
    const model = await trainGradientBoosting(X, y, N_TREES_BASE, LR_BASE, DEPTH_BASE);

    console.log(`[XGB-GCP] Training mean ensemble (${ENSEMBLE_SIZE} models × ${N_TREES_BASE} trees)...`);
    const ensembleXGB = await trainEnsemble(X, y);

    console.log(`[XGB-GCP] Training variance ensemble...`);
    const varianceEnsembleXGB = await trainVarianceEnsemble(X, y, ensembleXGB);

    const { r2, mae } = computeMetrics(ensembleXGB, X, y);
    const wallSec = ((Date.now() - startMs) / 1000).toFixed(1);

    await storage.updateXgbTrainingJob(jobId, {
      status: "done",
      model: model as any,
      ensembleXGB: ensembleXGB as any,
      varianceEnsembleXGB: varianceEnsembleXGB as any,
      r2,
      mae,
      completedAt: new Date(),
    });

    console.log(
      `[XGB-GCP] Job #${jobId} complete in ${wallSec}s — R²=${r2.toFixed(3)}, MAE=${mae.toFixed(2)}, N=${datasetSize}`
    );
  } catch (err: any) {
    console.error(`[XGB-GCP] Job #${jobId} failed: ${err.message}`);
    await storage.updateXgbTrainingJob(jobId, {
      status: "failed",
      errorMessage: err.message?.slice(0, 1000) ?? "unknown error",
      completedAt: new Date(),
    }).catch(() => {});
  } finally {
    setXGBTrainingActive(false);
  }

  return true;
}

export async function startXGBLoop(): Promise<void> {
  console.log(`[XGB-GCP] XGBoost training worker started — poll every ${POLL_INTERVAL_MS / 1000}s`);

  while (running) {
    try {
      const processed = await processNextXgbJob();
      await new Promise(r => setTimeout(r, processed ? 1000 : POLL_INTERVAL_MS));
    } catch (err: any) {
      const msg = err instanceof Error
        ? (err.stack || err.message || err.constructor?.name || "(empty Error)")
        : (err != null ? String(err) : "unknown");
      console.error(`[XGB-GCP] Loop error (${err?.constructor?.name ?? typeof err}): ${msg || "(no message)"}`);
      await new Promise(r => setTimeout(r, isConnectionError(err) ? 5000 : POLL_INTERVAL_MS));
    }
  }
}

export function stopXGBLoop() {
  running = false;
}
