/**
 * GCP ML Model Training Loop
 *
 * Polls ml_training_jobs for queued jobs and runs:
 *   - train-materials-vae   : trains the materials VAE on top Tc formula embeddings
 *   - init-crystal-vae      : trains the Crystal-VAE on the crystal structure dataset
 *   - init-diffusion-model  : trains the DDPM diffusion model on crystal structures
 *
 * Trained weights are serialized and written back to output_weights so the
 * local server can poll, deserialize, and apply them without any local CPU cost.
 */
import { db, isConnectionError } from "../server/db";
import { isGNNMajorTrainingActive } from "./training-priority";
import { mlTrainingJobs } from "@shared/schema";
import { eq, and, or, sql } from "drizzle-orm";
import { seedDatasetFromArray } from "../server/crystal/crystal-structure-dataset";
import { trainVAE as trainMaterialsVAE, exportMaterialsVAEWeights } from "../server/physics/materials-vae";
import { initCrystalVAE, exportCrystalVAEState } from "../server/crystal/crystal-vae";
import { initDiffusionModel, exportDiffusionState } from "../server/crystal/crystal-diffusion-model";

const POLL_INTERVAL_MS = 20_000;
let running = true;

const TASK_TYPES = ["train-materials-vae", "init-crystal-vae", "init-diffusion-model"] as const;
type TaskType = typeof TASK_TYPES[number];

async function claimNextJob(): Promise<{ id: number; taskType: TaskType; inputData: any } | null> {
  for (const taskType of TASK_TYPES) {
    const rows = await db.execute(
      `UPDATE ml_training_jobs
       SET status = 'running', started_at = NOW()
       WHERE id = (
         SELECT id FROM ml_training_jobs
         WHERE status = 'queued' AND task_type = '${taskType}'
         ORDER BY created_at ASC
         LIMIT 1
         FOR UPDATE SKIP LOCKED
       )
       RETURNING id, task_type, input_data`
    );
    const job = (rows as any).rows?.[0];
    if (job) return { id: job.id, taskType: job.task_type, inputData: job.input_data };
  }
  return null;
}

async function processJob(id: number, taskType: TaskType, inputData: any): Promise<void> {
  console.log(`[ML-GCP] Starting '${taskType}' job #${id}`);
  const startMs = Date.now();

  try {
    let outputWeights: any = null;

    if (taskType === "train-materials-vae") {
      const formulas: string[] = inputData?.formulas ?? [];
      const epochs: number = inputData?.epochs ?? 10;
      if (formulas.length < 3) throw new Error(`Not enough formulas (${formulas.length})`);
      await trainMaterialsVAE(formulas, epochs);
      outputWeights = exportMaterialsVAEWeights();

    } else if (taskType === "init-crystal-vae") {
      const dataset = inputData?.dataset ?? [];
      if (dataset.length > 0) seedDatasetFromArray(dataset);
      await initCrystalVAE();
      outputWeights = exportCrystalVAEState();

    } else if (taskType === "init-diffusion-model") {
      const dataset = inputData?.dataset ?? [];
      if (dataset.length > 0) seedDatasetFromArray(dataset);
      await initDiffusionModel();
      outputWeights = exportDiffusionState();
    }

    const weightsJson = JSON.stringify(outputWeights);
    await db.execute(
      sql`UPDATE ml_training_jobs
          SET status = 'done', output_weights = ${weightsJson}::jsonb, completed_at = NOW()
          WHERE id = ${id}`
    );

    const wallSec = ((Date.now() - startMs) / 1000).toFixed(1);
    console.log(`[ML-GCP] Job #${id} '${taskType}' done in ${wallSec}s`);

  } catch (err: any) {
    console.error(`[ML-GCP] Job #${id} '${taskType}' failed: ${err.message}`);
    const errMsg = err.message?.slice(0, 1000) ?? "unknown error";
    await db.execute(
      sql`UPDATE ml_training_jobs
          SET status = 'failed', error_message = ${errMsg}, completed_at = NOW()
          WHERE id = ${id}`
    );
  }
}

export async function startMLLoop(): Promise<void> {
  console.log(`[ML-GCP] ML training worker started — polling every ${POLL_INTERVAL_MS / 1000}s`);

  while (running) {
    try {
      if (isGNNMajorTrainingActive()) {
        await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
        continue;
      }
      const job = await claimNextJob();
      if (job) {
        await processJob(job.id, job.taskType, job.inputData);
        await new Promise(r => setTimeout(r, 1000));
      } else {
        await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
      }
    } catch (err: any) {
      const msg = err instanceof Error
        ? (err.stack || err.message || err.constructor?.name || "(empty Error)")
        : (err != null ? String(err) : "unknown");
      console.error(`[ML-GCP] Loop error (${err?.constructor?.name ?? typeof err}): ${msg || "(no message)"}`);
      // Connection errors resolve once pg-pool opens a fresh connection — retry quickly.
      await new Promise(r => setTimeout(r, isConnectionError(err) ? 5000 : POLL_INTERVAL_MS));
    }
  }
}

export function stopMLLoop(): void {
  running = false;
}
