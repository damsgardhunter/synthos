/**
 * GNN Worker Thread — trains a single ensemble member.
 *
 * Spawned by gnn-loop.ts via worker_threads so all 5 ensemble models train
 * in parallel on the dedicated GNN GCP instance instead of sequentially.
 * Each worker receives pre-loaded training data via workerData, trains one
 * bootstrapped model with trainSingleEnsembleModel(), and posts the weights
 * back to the parent thread.
 *
 * The DB pool (imported transitively through graph-neural-net → storage → db)
 * is created but never used here — training is pure in-memory computation.
 */
import { isMainThread, workerData, parentPort } from "worker_threads";

if (!isMainThread) {
  const { trainingData, seed, bootstrapRatio, maxPretrainEpochs, modelIndex } =
    workerData as {
      trainingData: import("../server/learning/graph-neural-net").TrainingSample[];
      seed: number;
      bootstrapRatio: number;
      maxPretrainEpochs: number;
      modelIndex: number;
    };

  const startMs = Date.now();

  // require() instead of import() — the worker is loaded via tsx/cjs which
  // hooks require() but does NOT handle ESM dynamic import(). Using import()
  // bypasses the tsx hook and fails with "Cannot find module" for .ts paths.
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { trainSingleEnsembleModel } = require("../server/learning/graph-neural-net");
    try {
      const model = trainSingleEnsembleModel(
        trainingData,
        seed,
        bootstrapRatio,
        maxPretrainEpochs,
      );
      const wallSec = ((Date.now() - startMs) / 1000).toFixed(1);
      console.log(
        `[GNN-Worker-${modelIndex}] done in ${wallSec}s — ${trainingData.length} samples, seed=${seed}`,
      );
      parentPort!.postMessage({ ok: true, model, modelIndex });
    } catch (err: any) {
      const msg = err?.message ?? String(err ?? "unknown");
      console.error(`[GNN-Worker-${modelIndex}] training failed: ${msg}`);
      parentPort!.postMessage({ ok: false, error: msg, modelIndex });
    }
  } catch (err: any) {
    const msg = err?.message ?? String(err ?? "require failed");
    console.error(`[GNN-Worker-${modelIndex}] import failed: ${msg}`);
    parentPort!.postMessage({ ok: false, error: `Import error: ${msg}`, modelIndex });
  }
}
