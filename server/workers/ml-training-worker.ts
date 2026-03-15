/**
 * ML Training Worker Thread
 *
 * Runs CPU-intensive model training in a separate thread so the main
 * Node.js event loop is never blocked. Receives a task via workerData,
 * runs the training, then posts the serialized weights back so the main
 * thread can apply them without re-blocking.
 */
import { parentPort, workerData } from "worker_threads";

interface WorkerTask {
  task: "train-materials-vae" | "init-crystal-vae" | "init-diffusion-model";
  data?: { formulas?: string[]; epochs?: number };
}

async function main() {
  const { task, data = {} }: WorkerTask = workerData;

  try {
    if (task === "train-materials-vae") {
      const { trainVAE, exportMaterialsVAEWeights } = await import("../physics/materials-vae");
      await trainVAE(data.formulas ?? [], data.epochs ?? 10);
      parentPort?.postMessage({ task, success: true, weights: exportMaterialsVAEWeights() });

    } else if (task === "init-crystal-vae") {
      const { initCrystalVAE, exportCrystalVAEState } = await import("../crystal/crystal-vae");
      await initCrystalVAE();
      parentPort?.postMessage({ task, success: true, state: exportCrystalVAEState() });

    } else if (task === "init-diffusion-model") {
      const { initDiffusionModel, exportDiffusionState } = await import("../crystal/crystal-diffusion-model");
      await initDiffusionModel();
      parentPort?.postMessage({ task, success: true, state: exportDiffusionState() });

    } else {
      parentPort?.postMessage({ task, success: false, error: `Unknown task: ${task}` });
    }
  } catch (err: any) {
    parentPort?.postMessage({ task, success: false, error: err?.message ?? String(err) });
  }
}

main();
