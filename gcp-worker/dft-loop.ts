/**
 * GCP DFT Worker Loop
 * Polls dft_jobs for queued jobs, runs Quantum ESPRESSO, writes results back.
 * Mirrors the logic in server/dft/dft-job-queue.ts processNextJob().
 */
import { db, isConnectionError } from "../server/db";
import { storage } from "../server/storage";
import { runFullDFT } from "../server/dft/qe-worker";
import { isGNNMajorTrainingActive, DFT_GNN_CONCURRENT } from "./training-priority";
import * as fs from "fs";
import * as path from "path";

const POLL_INTERVAL_MS = 15_000;
// 7 concurrent jobs × 4 OMP threads = 28 vCPUs for DFT, leaving 4 for GNN/XGB/OS.
// OMP_NUM_THREADS must be set to 4 in /etc/quantum-alchemy.env on the main instance.
// On a dedicated DFT-only instance (no GNN/XGB), raise this to 5 and set OMP_NUM_THREADS=3.
const MAX_CONCURRENT = parseInt(process.env.DFT_MAX_CONCURRENT ?? "7", 10);
const SCF_RETRY_START_ATTEMPT = 3;
const WORKER_NODE = "gcp";

let activeWorkers = 0;
let running = true;

async function claimAndRunJob(): Promise<boolean> {
  // Atomically claim the oldest queued job for this worker.
  // Retry up to 3× with backoff — GNN training can block the event loop for
  // 60-90s, leaving all pool connections stale. The first attempt after
  // unblocking may time out while pg-pool is reconnecting.
  let rows: any;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      rows = await db.execute(
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
      break; // success
    } catch (err: any) {
      if (!isConnectionError(err) || attempt === 3) throw err;
      console.warn(`[DFT-GCP] DB claim attempt ${attempt}/3 failed (${err.message?.slice(0, 60)}) — retrying in ${attempt * 5}s`);
      await new Promise(r => setTimeout(r, attempt * 5000));
    }
  }

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
    const isTSCJob = jobType === "scf_tsc";
    const startAttempt = isRetry ? (inputData?.startAttempt ?? SCF_RETRY_START_ATTEMPT) : 0;
    const jobPressureGpa: number | undefined =
      typeof inputData?.pressureGpa === "number" && inputData.pressureGpa > 20
        ? inputData.pressureGpa
        : undefined;

    // Look up ensembleScore and Stoner flag so the worker can decide whether to run DFPT EPC.
    let ensembleScore: number | undefined;
    let skipEph = false;
    try {
      const candidates = await storage.getSuperconductorsByFormula(formula);
      const best = candidates.reduce((b, c) => Math.max(b, c.ensembleScore ?? 0), 0);
      if (best > 0) ensembleScore = best;
      // Stoner ferromagnet flag stored in mlFeatures by pre-screening
      if (candidates.some(c => (c.mlFeatures as any)?.skipEph === true)) {
        skipEph = true;
        console.log(`[DFT-GCP] ${formula} flagged as Stoner ferromagnet — skipping DFPT e-ph`);
      }
    } catch { /* non-fatal */ }

    // Explicit phonon jobs must run DFPT regardless of ensembleScore.
    // The DFPT gate (opts.ensembleScore > 0.7) would otherwise skip e-ph
    // if the DB score hasn't been updated yet.
    const forcePhonon = jobType === "phonon";
    if (forcePhonon) {
      console.log(`[DFT-GCP] Job #${jobId} is phonon type — forcing DFPT EPC (ensembleScore override: 1.0)`);
    }

    const dftResult = await runFullDFT(formula, {
      startAttempt,
      pressureGpa: jobPressureGpa,
      ensembleScore: forcePhonon ? 1.0 : ensembleScore,
      forceSpin: isTSCJob,
      skipEph: forcePhonon ? false : skipEph,
    });

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
          qeVcRelaxed: dftResult.vcRelaxed ?? false,
          qeRelaxedLatticeA: dftResult.relaxedLatticeA ?? null,
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
          qeDFTPlusU: dftResult.qeDFTPlusU ?? false,
          qeDFTPlusUTcModifier: dftResult.dftPlusUTcModifier ?? null,
          qeDFPTLambda: dftResult.dfpt?.lambda ?? null,
          qeDFPTOmegaLog: dftResult.dfpt?.omegaLog ?? null,
          qeDFPTTc: dftResult.dfpt?.tcBest ?? null,
          qeDFPTSource: dftResult.dfpt?.source ?? null,
          qeDFPTPhConverged: dftResult.dfpt?.phConverged ?? null,
        };

        // DFPT Tc overrides ML estimate for top candidates
        if (dftResult.dfpt?.tcBest && dftResult.dfpt.tcBest > 0) {
          scalarUpdates.predictedTc = dftResult.dfpt.tcBest;
          console.log(`[DFT-GCP] ${formula} DFPT Tc ${dftResult.dfpt.tcBest.toFixed(1)} K written as predictedTc (λ=${dftResult.dfpt.lambda.toFixed(3)})`);
        }

        if (dftResult.vcRelaxed && dftResult.relaxedLatticeA) {
          scalarUpdates.latticeParams = { a: dftResult.relaxedLatticeA };
        }

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

async function resetZombieJobs(): Promise<void> {
  try {
    const result = await db.execute(
      `UPDATE dft_jobs
       SET status = 'queued',
           error_message = 'Auto-reset: zombie job from previous worker session',
           started_at = NULL,
           worker_node = NULL
       WHERE status = 'running'
       AND started_at < NOW() - INTERVAL '2 hours'`
    );
    const count = (result as any).rowCount ?? (result as any).rows?.length ?? 0;
    if (count > 0) {
      console.log(`[DFT-GCP] Zombie cleanup: reset ${count} stale running job(s) to queued`);
    }
  } catch (err: any) {
    console.warn(`[DFT-GCP] Zombie cleanup failed (non-fatal): ${err.message}`);
  }
}

export async function startDFTLoop(): Promise<void> {
  console.log(`[DFT-GCP] Worker started — max ${MAX_CONCURRENT} concurrent jobs, poll every ${POLL_INTERVAL_MS / 1000}s`);
  console.log(`[DFT-GCP] QE_BIN_DIR=${process.env.QE_BIN_DIR ?? "(auto)"}`);

  await resetZombieJobs();

  // Probe for pw.x before starting the job loop. Failing early with a clear
  // message is better than silently failing 1000+ jobs with ENOENT.
  // Search /nix/store dynamically — the hash segment changes per QE version/rebuild.
  const nixQEBins: string[] = (() => {
    try {
      if (!fs.existsSync("/nix/store")) return [];
      return fs.readdirSync("/nix/store")
        .filter(e => e.includes("quantum-espresso"))
        .map(e => `/nix/store/${e}/bin`)
        .filter(d => { try { return fs.existsSync(path.join(d, "pw.x")); } catch { return false; } });
    } catch { return []; }
  })();

  const searchDirs = [
    process.env.QE_BIN_DIR,
    ...nixQEBins,
    "/usr/bin",
    "/usr/local/bin",
    "/opt/conda/bin",
    "/opt/miniconda3/bin",
    "/opt/miniforge3/bin",
    "/root/miniforge3/bin",
    "/root/miniconda3/bin",
    "/opt/quantum-espresso/bin",
    "/opt/qe/bin",
    "/opt/espresso/bin",
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
      const effectiveMax = isGNNMajorTrainingActive() ? Math.min(MAX_CONCURRENT, DFT_GNN_CONCURRENT) : MAX_CONCURRENT;
      const slots = effectiveMax - activeWorkers;
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
