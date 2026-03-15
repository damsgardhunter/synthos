/**
 * ML Training Bridge
 *
 * When OFFLOAD_ML_TO_GCP=true (set in .env):
 *   - Queues training jobs in the ml_training_jobs DB table
 *   - GCP worker picks them up, trains on its 32-vCPU machine, writes weights back
 *   - A background poller (startMLResultPoller) applies completed weights every 60s
 *   - Zero CPU cost on the local machine
 *
 * When OFFLOAD_ML_TO_GCP is not set:
 *   - Falls back to worker_threads so training is off the main event loop
 *   - Still no main-thread blocking, just uses local CPU
 */
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";
import { storage } from "../storage";
import { getTrainingData } from "../crystal/crystal-structure-dataset";
import { importMaterialsVAEWeights } from "../physics/materials-vae";
import { importCrystalVAEState } from "../crystal/crystal-vae";
import { importDiffusionState } from "../crystal/crystal-diffusion-model";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export type MLTask = "train-materials-vae" | "init-crystal-vae" | "init-diffusion-model";

const USE_GCP = process.env.OFFLOAD_ML_TO_GCP === "true";

// ─── GCP path ────────────────────────────────────────────────────────────────

async function queueGCPJob(task: MLTask, data?: { formulas?: string[]; epochs?: number }): Promise<void> {
  try {
    // Cancel any stuck previous jobs for this task so GCP doesn't pile up
    await storage.cancelStaleMlJobs(task);

    let inputData: any = data ?? {};

    // Crystal VAE and diffusion model need the local crystal dataset shipped with the job
    if (task === "init-crystal-vae" || task === "init-diffusion-model") {
      const dataset = getTrainingData();
      if (dataset.length < 10) {
        console.log(`[MLBridge] Skipping '${task}' — crystal dataset too small (${dataset.length} entries)`);
        return;
      }
      inputData = { ...inputData, dataset };
    }

    if (task === "train-materials-vae" && (data?.formulas?.length ?? 0) < 3) {
      console.log(`[MLBridge] Skipping '${task}' — not enough formulas`);
      return;
    }

    await storage.insertMlTrainingJob({ taskType: task, status: "queued", inputData });
    console.log(`[MLBridge] Queued GCP job for '${task}'`);
  } catch (err: any) {
    console.error(`[MLBridge] Failed to queue GCP job for '${task}':`, err.message);
  }
}

// ─── Worker-thread fallback path ──────────────────────────────────────────────

let activeWorkers = 0;

async function spawnLocalWorker(task: MLTask, data?: { formulas?: string[]; epochs?: number }): Promise<void> {
  if (activeWorkers >= 1) {
    console.log(`[MLBridge] Skipping '${task}' — local worker already active`);
    return;
  }

  const { Worker } = await import("worker_threads");
  activeWorkers++;
  console.log(`[MLBridge] Spawning local worker thread for '${task}'`);

  return new Promise((done) => {
    const worker = new Worker(resolve(__dirname, "ml-training-worker.ts"), {
      workerData: { task, data: data ?? {} },
      execArgv: process.execArgv,
    });

    worker.on("message", (msg: any) => {
      if (msg.success) {
        try {
          if (task === "train-materials-vae" && msg.weights) importMaterialsVAEWeights(msg.weights);
          else if (task === "init-crystal-vae" && msg.state)  importCrystalVAEState(msg.state);
          else if (task === "init-diffusion-model" && msg.state) importDiffusionState(msg.state);
          console.log(`[MLBridge] Applied weights for '${task}' from local worker`);
        } catch (e) {
          console.error(`[MLBridge] Weight apply failed for '${task}':`, e);
        }
      } else {
        console.error(`[MLBridge] Local worker failed for '${task}':`, msg.error);
      }
      done();
    });

    worker.on("error", (err) => { console.error(`[MLBridge] Worker error:`, err); done(); });
    worker.on("exit", () => { activeWorkers--; });
  });
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function spawnMLTraining(task: MLTask, data?: { formulas?: string[]; epochs?: number }): Promise<void> {
  if (USE_GCP) return queueGCPJob(task, data);
  return spawnLocalWorker(task, data);
}

// ─── Result poller (GCP path only) ───────────────────────────────────────────

const _lastApplied: Record<string, number> = {};

async function pollMLResults(): Promise<void> {
  const tasks: MLTask[] = ["train-materials-vae", "init-crystal-vae", "init-diffusion-model"];

  for (const task of tasks) {
    try {
      const job = await storage.getLatestCompletedMlJob(task);
      if (!job || job.id <= (_lastApplied[task] ?? 0)) continue;

      _lastApplied[task] = job.id;
      const weights = job.outputWeights as any;
      if (!weights) continue;

      if (task === "train-materials-vae") {
        importMaterialsVAEWeights(weights);
        console.log(`[GCP-ML-Poller] Applied materials-vae weights from job #${job.id}`);
      } else if (task === "init-crystal-vae") {
        importCrystalVAEState(weights);
        console.log(`[GCP-ML-Poller] Applied crystal-vae state from job #${job.id}`);
      } else if (task === "init-diffusion-model") {
        importDiffusionState(weights);
        console.log(`[GCP-ML-Poller] Applied diffusion-model state from job #${job.id}`);
      }
    } catch { /* silent — non-critical */ }
  }
}

export function startMLResultPoller(): void {
  if (!USE_GCP) return;

  async function loop() {
    await pollMLResults();
    setTimeout(loop, 60_000);
  }
  setTimeout(loop, 60_000); // first poll after 60s
  console.log("[GCP-ML-Poller] ML result poller started (60s interval)");
}
