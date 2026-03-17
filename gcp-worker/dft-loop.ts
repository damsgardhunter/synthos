/**
 * GCP DFT Worker Loop
 * Polls dft_jobs for queued jobs, runs Quantum ESPRESSO, writes results back.
 * Mirrors the logic in server/dft/dft-job-queue.ts processNextJob().
 */
import { db, isConnectionError } from "../server/db";
import { storage } from "../server/storage";
import { runFullDFT } from "../server/dft/qe-worker";
import * as fs from "fs";
import * as path from "path";

const POLL_INTERVAL_MS = 15_000;
const MAX_CONCURRENT = 4; // 4 × 8 threads = 32 vCPUs
const SCF_RETRY_START_ATTEMPT = 3;
const WORKER_NODE = "gcp";

let activeWorkers = 0;
let running = true;

async function claimAndRunJob(): Promise<boolean> {
  // Atomically claim the oldest queued job for this worker
  const rows = await db.execute(
    `UPDATE dft_jobs
     SET status = 'running', worker_node = '${WORKER_NODE}', started_at = NOW()
     WHERE id = (
       SELECT id FROM dft_jobs
       WHERE status = 'queued'
       ORDER BY priority DESC, created_at ASC
       LIMIT 1
       FOR UPDATE SKIP LOCKED
     )
     RETURNING id, formula, job_type, input_data`
  );

  const job = (rows as any).rows?.[0] ?? (Array.isArray(rows) ? rows[0] : undefined);
  if (!job) return false;

  const jobId: number = job.id;
  const formula: string = job.formula;
  const jobType: string = job.job_type ?? "scf";
  const inputData: any = job.input_data ?? {};

  console.log(`[DFT-GCP] Starting job #${jobId} — ${formula} (${jobType})`);
  activeWorkers++;
  const jobStartTime = Date.now();

  try {
    const isRetry = jobType === "scf_retry";
    const startAttempt = isRetry ? (inputData?.startAttempt ?? SCF_RETRY_START_ATTEMPT) : 0;

    const dftResult = await runFullDFT(formula, { startAttempt });

    const scfSuccess = dftResult.scf?.converged || false;
    const status = scfSuccess ? "completed" : "failed";
    const wallMs = Date.now() - jobStartTime;

    await storage.updateDftJob(jobId, {
      status,
      outputData: dftResult as any,
      completedAt: new Date(),
      errorMessage: dftResult.error || dftResult.scf?.error || null,
    } as any);

    if (scfSuccess && dftResult.scf) {
      try {
        const scalarUpdates: any = { dataConfidence: "dft-verified" };
        if (dftResult.scf.bandGap !== null) scalarUpdates.bandGap = dftResult.scf.bandGap;
        if (dftResult.scf.totalEnergyPerAtom) scalarUpdates.formationEnergy = dftResult.scf.totalEnergyPerAtom;

        const bandData = dftResult.bandStructure;
        const mlFeaturePatch: Record<string, any> = {
          qeDFT: true,
          qeConverged: true,
          qeTotalEnergy: dftResult.scf.totalEnergy,
          qeFermiEnergy: dftResult.scf.fermiEnergy,
          qePressure: dftResult.scf.pressure,
          qePhononStable: dftResult.phonon ? dftResult.phonon.lowestFrequency > -10.0 : null,
          qePhononLowestFreq: dftResult.phonon?.lowestFrequency ?? null,
          qePhononImaginaryCount: dftResult.phonon?.imaginaryCount ?? 0,
          qeMagnetization: dftResult.scf.magnetization ?? null,
          dftConfidence: 1.0,
          qeBands: bandData?.converged || false,
          qeBandGapPath: bandData?.bandGapAlongPath ?? null,
          qeFlatBandScore: bandData?.flatBandScore ?? 0,
        };

        const updated = await storage.updateCandidateByFormulaDft(formula, scalarUpdates, mlFeaturePatch);
        console.log(`[DFT-GCP] Job #${jobId} complete — ${formula}, SCF converged, updated=${updated} (${(wallMs / 1000).toFixed(1)}s)`);
      } catch (err: any) {
        console.warn(`[DFT-GCP] Job #${jobId} result write failed: ${err.message}`);
      }
    } else {
      const reason = dftResult.error || dftResult.scf?.error || "SCF did not converge";
      console.log(`[DFT-GCP] Job #${jobId} failed — ${formula}: ${reason}`);
    }
  } catch (err: any) {
    console.error(`[DFT-GCP] Job #${jobId} error — ${formula}: ${err.message}`);
    await storage.updateDftJob(jobId, {
      status: "failed",
      errorMessage: err.message?.slice(0, 500) ?? "unknown error",
      completedAt: new Date(),
    } as any).catch(() => {});
  } finally {
    activeWorkers--;
  }

  return true;
}

export async function startDFTLoop(): Promise<void> {
  console.log(`[DFT-GCP] Worker started — max ${MAX_CONCURRENT} concurrent jobs, poll every ${POLL_INTERVAL_MS / 1000}s`);
  console.log(`[DFT-GCP] QE_BIN_DIR=${process.env.QE_BIN_DIR ?? "(auto)"}`);

  // Probe for pw.x before starting the job loop. Failing early with a clear
  // message is better than silently failing 1000+ jobs with ENOENT.
  const searchDirs = [
    process.env.QE_BIN_DIR,
    "/nix/store/4rd771qjyb5mls5dkcs614clwdxsagql-quantum-espresso-7.2/bin",
    "/usr/bin",
    "/usr/local/bin",
  ].filter(Boolean) as string[];
  const pwxDir = searchDirs.find(d => fs.existsSync(path.join(d, "pw.x")));
  if (!pwxDir) {
    console.error(`[DFT-GCP] FATAL: pw.x not found (searched: ${searchDirs.map(d => d + "/pw.x").join(", ")})`);
    console.error(`[DFT-GCP] Install Quantum ESPRESSO: sudo apt-get install -y quantum-espresso`);
    console.error(`[DFT-GCP] Or set QE_BIN_DIR=/path/to/bin in /etc/quantum-alchemy.env`);
    console.error(`[DFT-GCP] DFT worker disabled — GNN/XGB loops will continue`);
    return;
  }
  if (!process.env.QE_BIN_DIR) process.env.QE_BIN_DIR = pwxDir;
  console.log(`[DFT-GCP] pw.x found at ${pwxDir} — ready to process DFT jobs`);

  while (running) {
    try {
      const slots = MAX_CONCURRENT - activeWorkers;
      const launches: Promise<boolean>[] = [];
      for (let i = 0; i < slots; i++) {
        launches.push(claimAndRunJob());
      }
      const results = await Promise.all(launches);
      const anyStarted = results.some(Boolean);
      await new Promise(r => setTimeout(r, anyStarted ? 2000 : POLL_INTERVAL_MS));
    } catch (err: any) {
      const msg = err instanceof Error
        ? (err.stack || err.message || err.constructor?.name || "(empty Error)")
        : (err != null ? String(err) : "unknown");
      console.error(`[DFT-GCP] Loop error (${err?.constructor?.name ?? typeof err}): ${msg || "(no message)"}`);
      await new Promise(r => setTimeout(r, isConnectionError(err) ? 5000 : POLL_INTERVAL_MS));
    }
  }
}

export function stopDFTLoop() {
  running = false;
}
