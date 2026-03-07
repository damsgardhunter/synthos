import { storage } from "../storage";
import { runFullDFT, isQEAvailable, isFormulaBlocked, getStageFailureCounts } from "./qe-worker";
import type { QEFullResult } from "./qe-worker";
import type { DftJob } from "@shared/schema";

const POLL_INTERVAL_MS = 30_000;
const MAX_CONCURRENT = 1;

let isProcessing = false;
let workerLoopTimer: ReturnType<typeof setTimeout> | null = null;
let totalProcessed = 0;
let totalSucceeded = 0;
let totalFailed = 0;
let currentJob: DftJob | null = null;
let broadcastFn: ((event: string, data: any) => void) | null = null;

export function setDFTBroadcast(fn: (event: string, data: any) => void) {
  broadcastFn = fn;
}

export async function submitDFTJob(
  formula: string,
  candidateId: number | null,
  priority: number = 50,
  jobType: string = "scf",
): Promise<DftJob | null> {
  if (isFormulaBlocked(formula)) {
    console.log(`[DFT-Queue] Formula ${formula} blocked due to repeated failures, skipping`);
    return null;
  }

  const existing = await storage.getDftJobsByFormula(formula);
  const activeJob = existing.find(j => j.status === "queued" || j.status === "running");
  if (activeJob) {
    console.log(`[DFT-Queue] Job already queued/running for ${formula}, skipping`);
    return activeJob;
  }

  const oneDayAgo = new Date(Date.now() - 24 * 3600_000);
  const recentFailed = existing.filter(j => {
    if (j.status !== "failed") return false;
    if (!j.completedAt || new Date(j.completedAt) < oneDayAgo) return false;
    const out = j.outputData as any;
    if (out?.ppValidated === false || out?.ppValidated === null || out?.ppValidated === undefined) return false;
    return true;
  });
  if (recentFailed.length >= 3) {
    console.log(`[DFT-Queue] Formula ${formula} has ${recentFailed.length} validated recent failures, skipping`);
    return null;
  }

  const job = await storage.insertDftJob({
    formula,
    candidateId,
    status: "queued",
    jobType,
    priority,
    inputData: { formula, candidateId, requestedAt: new Date().toISOString() },
  });

  console.log(`[DFT-Queue] Queued job #${job.id} for ${formula} (priority=${priority}, type=${jobType})`);

  if (broadcastFn) {
    broadcastFn("dftJobQueued", { jobId: job.id, formula, priority });
  }

  return job;
}

async function processNextJob(): Promise<boolean> {
  if (isProcessing) return false;

  const queued = await storage.getQueuedDftJobs(1);
  if (queued.length === 0) return false;

  const job = queued[0];
  isProcessing = true;
  currentJob = job;

  try {
    await storage.updateDftJob(job.id, {
      status: "running",
      startedAt: new Date(),
    } as any);

    console.log(`[DFT-Queue] Processing job #${job.id}: ${job.formula} (type=${job.jobType})`);

    if (broadcastFn) {
      broadcastFn("dftJobStarted", { jobId: job.id, formula: job.formula });
    }

    let dftResult: QEFullResult;

    if (isQEAvailable()) {
      dftResult = await runFullDFT(job.formula);
    } else {
      dftResult = {
        formula: job.formula,
        method: "QE-PW-PBE",
        scf: null,
        phonon: null,
        wallTimeTotal: 0,
        error: "Quantum ESPRESSO binaries not available",
      };
    }

    const success = dftResult.scf?.converged || false;
    const status = success ? "completed" : "failed";

    await storage.updateDftJob(job.id, {
      status,
      outputData: dftResult as any,
      completedAt: new Date(),
      errorMessage: dftResult.error || dftResult.scf?.error || null,
    } as any);

    totalProcessed++;
    if (success) {
      totalSucceeded++;

      try {
        const candidates = await storage.getSuperconductorsByFormula(job.formula);
        if (candidates.length > 0) {
          const candidate = candidates[0];
          const updates: any = {
            dataConfidence: "high",
          };

          if (dftResult.scf) {
            if (dftResult.scf.bandGap !== null) {
              updates.bandGap = dftResult.scf.bandGap;
            }
            if (dftResult.scf.totalEnergyPerAtom) {
              updates.formationEnergy = dftResult.scf.totalEnergyPerAtom;
            }
          }

          const mlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};
          updates.mlFeatures = {
            ...mlFeatures,
            qeDFT: true,
            qeConverged: dftResult.scf?.converged || false,
            qeTotalEnergy: dftResult.scf?.totalEnergy,
            qeFermiEnergy: dftResult.scf?.fermiEnergy,
            qePressure: dftResult.scf?.pressure,
            qePhononStable: dftResult.phonon ? !dftResult.phonon.hasImaginary : null,
            qePhononFreqs: dftResult.phonon?.frequencies?.length || 0,
            dftConfidence: 1.0,
          };

          await storage.updateSuperconductorCandidate(candidate.id, updates);
          console.log(`[DFT-Queue] Updated candidate ${job.formula} with QE DFT results`);
        }
      } catch (err: any) {
        console.log(`[DFT-Queue] Failed to update candidate: ${err.message}`);
      }

      console.log(`[DFT-Queue] Job #${job.id} completed: ${job.formula} (${dftResult.wallTimeTotal.toFixed(1)}s)`);
    } else {
      totalFailed++;
      console.log(`[DFT-Queue] Job #${job.id} failed: ${job.formula} — ${dftResult.error || "SCF did not converge"}`);
    }

    if (broadcastFn) {
      broadcastFn("dftJobCompleted", {
        jobId: job.id,
        formula: job.formula,
        status,
        wallTime: dftResult.wallTimeTotal,
        converged: success,
      });
    }

    return true;
  } catch (err: any) {
    await storage.updateDftJob(job.id, {
      status: "failed",
      completedAt: new Date(),
      errorMessage: err.message,
    } as any);
    totalProcessed++;
    totalFailed++;
    console.log(`[DFT-Queue] Job #${job.id} error: ${err.message}`);
    return false;
  } finally {
    isProcessing = false;
    currentJob = null;
  }
}

async function cleanupStaleJobs() {
  try {
    const staleRunning = await storage.getDftJobsByStatus("running");
    for (const job of staleRunning) {
      await storage.updateDftJob(job.id, {
        status: "failed",
        completedAt: new Date(),
        errorMessage: "Stale job from previous server session",
      } as any);
      console.log(`[DFT-Queue] Cleaned up stale running job #${job.id} (${job.formula})`);
    }
    if (staleRunning.length > 0) {
      console.log(`[DFT-Queue] Cleaned ${staleRunning.length} stale running job(s)`);
    }
  } catch (err: any) {
    console.log(`[DFT-Queue] Stale job cleanup error: ${err.message}`);
  }
}

export function startDFTWorkerLoop() {
  if (workerLoopTimer) return;

  if (!isQEAvailable()) {
    console.log("[DFT-Queue] WARNING: Quantum ESPRESSO pw.x not found, DFT queue will run but calculations may fail");
  } else {
    console.log("[DFT-Queue] Quantum ESPRESSO detected at pw.x, full DFT worker ready");
  }

  cleanupStaleJobs().catch(() => {});

  console.log("[DFT-Queue] Starting DFT worker loop (poll every 30s)");

  async function loop() {
    try {
      const processed = await processNextJob();
      if (processed) {
        const moreQueued = await storage.getQueuedDftJobs(1);
        if (moreQueued.length > 0) {
          workerLoopTimer = setTimeout(loop, 2000);
          return;
        }
      }
    } catch (err: any) {
      console.log(`[DFT-Queue] Worker loop error: ${err.message}`);
    }
    workerLoopTimer = setTimeout(loop, POLL_INTERVAL_MS);
  }

  workerLoopTimer = setTimeout(loop, 5000);
}

export function stopDFTWorkerLoop() {
  if (workerLoopTimer) {
    clearTimeout(workerLoopTimer);
    workerLoopTimer = null;
    console.log("[DFT-Queue] DFT worker loop stopped");
  }
}

export async function getDFTQueueStats() {
  const dbStats = await storage.getDftJobStats();
  const recentJobs = await storage.getRecentDftJobs(10);

  return {
    ...dbStats,
    totalProcessed,
    totalSucceeded,
    totalFailed,
    isProcessing,
    currentFormula: currentJob?.formula || null,
    qeAvailable: isQEAvailable(),
    stageFailures: getStageFailureCounts(),
    recentJobs: recentJobs.map(j => {
      const out = j.outputData as any;
      return {
        id: j.id,
        formula: j.formula,
        status: j.status,
        jobType: j.jobType,
        priority: j.priority,
        createdAt: j.createdAt,
        startedAt: j.startedAt,
        completedAt: j.completedAt,
        wallTime: out?.wallTimeTotal || null,
        converged: out?.scf?.converged || false,
        totalEnergy: out?.scf?.totalEnergy || null,
        retryCount: out?.retryCount ?? null,
        xtbPreRelaxed: out?.xtbPreRelaxed ?? null,
        ppValidated: out?.ppValidated ?? null,
        rejectionReason: out?.rejectionReason ?? null,
        failureStage: out?.failureStage ?? null,
        prototypeUsed: out?.prototypeUsed ?? null,
        kPoints: out?.kPoints ?? null,
        error: j.errorMessage,
      };
    }),
  };
}
