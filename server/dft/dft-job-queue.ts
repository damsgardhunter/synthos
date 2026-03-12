import { storage } from "../storage";
import { runFullDFT, isQEAvailable, isFormulaBlocked, getStageFailureCounts } from "./qe-worker";
import type { QEFullResult } from "./qe-worker";
import type { DftJob } from "@shared/schema";
import { recordStructureFailure } from "../crystal/structure-failure-db";
import { isValidFormula } from "../learning/utils";

const POLL_INTERVAL_MS = 30_000;
const MAX_CONCURRENT = 3;
const MIN_QUEUE_SIZE = 50;
const REFILL_BATCH_SIZE = 100;

type PipelineStage = "candidate_queue" | "dft_workers" | "phonon_workers" | "epc_workers";

interface StageMetrics {
  currentDepth: number;
  totalProcessed: number;
  totalSucceeded: number;
  totalFailed: number;
  avgProcessingTimeMs: number;
  lastProcessedAt: number | null;
}

const stageMetrics: Record<PipelineStage, StageMetrics> = {
  candidate_queue: { currentDepth: 0, totalProcessed: 0, totalSucceeded: 0, totalFailed: 0, avgProcessingTimeMs: 0, lastProcessedAt: null },
  dft_workers: { currentDepth: 0, totalProcessed: 0, totalSucceeded: 0, totalFailed: 0, avgProcessingTimeMs: 0, lastProcessedAt: null },
  phonon_workers: { currentDepth: 0, totalProcessed: 0, totalSucceeded: 0, totalFailed: 0, avgProcessingTimeMs: 0, lastProcessedAt: null },
  epc_workers: { currentDepth: 0, totalProcessed: 0, totalSucceeded: 0, totalFailed: 0, avgProcessingTimeMs: 0, lastProcessedAt: null },
};

function updateStageMetrics(stage: PipelineStage, success: boolean, processingTimeMs: number) {
  const m = stageMetrics[stage];
  m.totalProcessed++;
  if (success) m.totalSucceeded++;
  else m.totalFailed++;
  m.lastProcessedAt = Date.now();
  const prev = m.avgProcessingTimeMs;
  m.avgProcessingTimeMs = prev === 0 ? processingTimeMs : prev * 0.8 + processingTimeMs * 0.2;
}

let activeWorkers = 0;
let workerLoopTimer: ReturnType<typeof setTimeout> | null = null;
let totalProcessed = 0;
let totalSucceeded = 0;
let totalFailed = 0;
const activeJobs = new Map<number, DftJob>();
let staleJobsCleanedCount = 0;
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
  if (!isValidFormula(formula)) {
    console.log(`[DFT-Queue] Formula ${formula} rejected: invalid or contains noble gas`);
    return null;
  }
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

  stageMetrics.candidate_queue.currentDepth++;

  console.log(`[DFT-Queue] Queued job #${job.id} for ${formula} (priority=${priority}, type=${jobType})`);

  if (broadcastFn) {
    broadcastFn("dftJobQueued", { jobId: job.id, formula, priority });
  }

  return job;
}

export async function promoteDFTJob(formula: string, newPriority: number): Promise<boolean> {
  try {
    const jobs = await storage.getDftJobsByFormula(formula);
    const queued = jobs.filter(j => j.status === "queued");
    if (queued.length === 0) return false;
    for (const job of queued) {
      if ((job.priority ?? 0) < newPriority) {
        await storage.updateDftJobIfStatus(job.id, "queued", { priority: newPriority } as any);
        console.log(`[DFT-Queue] Promoted job #${job.id} for ${formula}: priority ${job.priority} -> ${newPriority}`);
        if (broadcastFn) {
          broadcastFn("dftJobPromoted", { jobId: job.id, formula, oldPriority: job.priority, newPriority });
        }
      }
    }
    return true;
  } catch (err: any) {
    console.log(`[DFT-Queue] Failed to promote ${formula}: ${err.message?.slice(0, 80)}`);
    return false;
  }
}

async function processNextJob(): Promise<boolean> {
  if (activeWorkers >= MAX_CONCURRENT) return false;

  const queued = await storage.getQueuedDftJobs(1);
  if (queued.length === 0) return false;

  const job = queued[0];
  activeWorkers++;
  activeJobs.set(job.id, job);

  if (stageMetrics.candidate_queue.currentDepth > 0) {
    stageMetrics.candidate_queue.currentDepth--;
  }
  stageMetrics.candidate_queue.totalProcessed++;
  stageMetrics.candidate_queue.totalSucceeded++;
  stageMetrics.candidate_queue.lastProcessedAt = Date.now();

  stageMetrics.dft_workers.currentDepth++;

  const jobStartTime = Date.now();

  try {
    const claimed = await storage.updateDftJobIfStatus(job.id, "queued", {
      status: "running",
      startedAt: new Date(),
    } as any);

    if (!claimed) {
      console.log(`[DFT-Queue] Job #${job.id} already claimed by another worker, skipping`);
      activeWorkers--;
      activeJobs.delete(job.id);
      return false;
    }

    console.log(`[DFT-Queue] Processing job #${job.id}: ${job.formula} (type=${job.jobType}, workers=${activeWorkers}/${MAX_CONCURRENT})`);

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
        bandStructure: null,
        wallTimeTotal: 0,
        error: "Quantum ESPRESSO binaries not available",
      };
    }

    const scfSuccess = dftResult.scf?.converged || false;
    const status = scfSuccess ? "completed" : "failed";
    const dftElapsed = Date.now() - jobStartTime;

    stageMetrics.dft_workers.currentDepth--;
    updateStageMetrics("dft_workers", scfSuccess, dftElapsed);

    if (scfSuccess && dftResult.phonon) {
      const phononSuccess = dftResult.phonon.converged;
      stageMetrics.phonon_workers.currentDepth++;
      updateStageMetrics("phonon_workers", phononSuccess, dftResult.phonon.wallTimeSeconds * 1000);
      stageMetrics.phonon_workers.currentDepth--;

      stageMetrics.epc_workers.currentDepth++;
      updateStageMetrics("epc_workers", phononSuccess && !dftResult.phonon.hasImaginary, 0);
      stageMetrics.epc_workers.currentDepth--;
    }

    await storage.updateDftJob(job.id, {
      status,
      outputData: dftResult as any,
      completedAt: new Date(),
      errorMessage: dftResult.error || dftResult.scf?.error || null,
    } as any);

    totalProcessed++;
    if (scfSuccess) {
      totalSucceeded++;

      try {
        const candidates = await storage.getSuperconductorsByFormula(job.formula);
        if (candidates.length > 0) {
          const candidate = candidates[0];
          const updates: any = {
            dataConfidence: "dft-verified",
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
          const bandData = dftResult.bandStructure;
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
            qeBands: bandData?.converged || false,
            qeBandCrossings: bandData?.bandCrossings?.length || 0,
            qeBandInversions: bandData?.bandInversions?.length || 0,
            qeBandGapPath: bandData?.bandGapAlongPath ?? null,
            qeMetallicPath: bandData?.isMetallicAlongPath ?? null,
            qeFlatBandScore: bandData?.flatBandScore ?? 0,
            qeDiracScore: bandData?.diracCrossingScore ?? 0,
            qeVHSCount: bandData?.vanHoveSingularities?.length || 0,
            qeBandWidth: bandData?.bandWidth ?? 0,
            qeBandInversionCount: bandData?.topologicalIndicators?.bandInversionCount || 0,
            qeNodalLineIndicator: bandData?.topologicalIndicators?.nodalLineIndicator || 0,
            qeParityChanges: bandData?.topologicalIndicators?.parityChanges || 0,
            qeDiracPointCount: bandData?.topologicalIndicators?.diracPointCount || 0,
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
      try {
        const hasPhonon = dftResult.phonon && dftResult.phonon.hasImaginary;
        let failureReason: "unstable_phonons" | "structure_collapse" | "high_formation_energy" | "non_metallic" | "scf_divergence" | "geometry_rejected" = "scf_divergence";
        if (hasPhonon) failureReason = "unstable_phonons";
        else if (dftResult.error?.includes("geometry")) failureReason = "geometry_rejected";
        recordStructureFailure({
          formula: job.formula,
          failureReason,
          failedAt: Date.now(),
          source: "dft",
          details: dftResult.error || dftResult.scf?.error || "SCF did not converge",
          lowestPhononFreq: dftResult.phonon?.lowestFrequency,
          imaginaryModeCount: dftResult.phonon?.imaginaryCount,
        });
      } catch {}
    }

    if (broadcastFn) {
      broadcastFn("dftJobCompleted", {
        jobId: job.id,
        formula: job.formula,
        status,
        wallTime: dftResult.wallTimeTotal,
        converged: scfSuccess,
      });
    }

    return true;
  } catch (err: any) {
    stageMetrics.dft_workers.currentDepth = Math.max(0, stageMetrics.dft_workers.currentDepth - 1);
    updateStageMetrics("dft_workers", false, Date.now() - jobStartTime);

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
    activeWorkers--;
    activeJobs.delete(job.id);
  }
}

async function cleanupStaleJobs() {
  try {
    const staleRunning = await storage.getDftJobsByStatus("running");
    let requeued = 0;
    let failed = 0;
    for (const job of staleRunning) {
      const attempts = (job as any).retryCount ?? 0;
      if (attempts < 2) {
        await storage.updateDftJob(job.id, {
          status: "queued",
          startedAt: null,
          completedAt: null,
          errorMessage: null,
        } as any);
        requeued++;
        console.log(`[DFT-Queue] Re-queued stale job #${job.id} (${job.formula}) for retry`);
      } else {
        await storage.updateDftJob(job.id, {
          status: "failed",
          completedAt: new Date(),
          errorMessage: "Stale job from previous server session (max retries reached)",
        } as any);
        staleJobsCleanedCount++;
        failed++;
      }
    }
    if (staleRunning.length > 0) {
      console.log(`[DFT-Queue] Stale job cleanup: ${requeued} re-queued, ${failed} failed (${staleRunning.length} total stale)`);
    }
  } catch (err: any) {
    console.log(`[DFT-Queue] Stale job cleanup error: ${err.message}`);
  }
}

async function refillQueueIfLow(): Promise<number> {
  try {
    const queued = await storage.getQueuedDftJobs(MIN_QUEUE_SIZE + 1);
    const queueSize = queued.length;
    stageMetrics.candidate_queue.currentDepth = queueSize;

    if (queueSize >= MIN_QUEUE_SIZE) return 0;

    const candidates = await storage.getSuperconductorCandidates(REFILL_BATCH_SIZE * 3);
    const eligible = candidates
      .filter(c => {
        const mlFeatures = (c.mlFeatures as Record<string, any>) ?? {};
        if (mlFeatures.qeDFT) return false;
        if ((c.ensembleScore ?? 0) <= 0) return false;
        return true;
      })
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));

    const queuedFormulas = new Set(queued.map(j => j.formula));
    let submitted = 0;
    for (const candidate of eligible) {
      if (submitted >= REFILL_BATCH_SIZE) break;
      if (queuedFormulas.has(candidate.formula)) continue;
      if (isFormulaBlocked(candidate.formula)) continue;

      try {
        const existing = await storage.getDftJobsByFormula(candidate.formula);
        const hasActive = existing.some(j => j.status === "queued" || j.status === "running");
        if (hasActive) continue;

        const oneDayAgo = new Date(Date.now() - 24 * 3600_000);
        const recentFailed = existing.filter(j => {
          if (j.status !== "failed") return false;
          if (!j.completedAt || new Date(j.completedAt) < oneDayAgo) return false;
          const out = j.outputData as any;
          if (out?.ppValidated === false || out?.ppValidated === null || out?.ppValidated === undefined) return false;
          return true;
        });
        if (recentFailed.length >= 3) continue;

        const priority = Math.round((candidate.ensembleScore ?? 0) * 100);
        await storage.insertDftJob({
          formula: candidate.formula,
          candidateId: candidate.id,
          status: "queued",
          jobType: "scf",
          priority,
          inputData: { formula: candidate.formula, candidateId: candidate.id, requestedAt: new Date().toISOString(), source: "queue_refill" },
        });
        queuedFormulas.add(candidate.formula);
        submitted++;
        stageMetrics.candidate_queue.currentDepth++;
      } catch {}
    }

    if (submitted > 0) {
      console.log(`[DFT-Queue] Refilled queue with ${submitted} candidates (queue was ${queueSize}/${MIN_QUEUE_SIZE})`);
    }
    return submitted;
  } catch (err: any) {
    console.log(`[DFT-Queue] Queue refill error: ${err.message}`);
    return 0;
  }
}

let watchdogTimer: ReturnType<typeof setInterval> | null = null;
let lastLoopRun = 0;
let loopRunning = false;

export function startDFTWorkerLoop() {
  if (workerLoopTimer) return;

  if (!isQEAvailable()) {
    console.log("[DFT-Queue] WARNING: Quantum ESPRESSO pw.x not found, DFT queue will run but calculations may fail");
  } else {
    console.log("[DFT-Queue] Quantum ESPRESSO detected at pw.x, full DFT worker ready");
  }

  cleanupStaleJobs().catch(() => {});

  console.log(`[DFT-Queue] Starting DFT worker loop (poll every 30s, max ${MAX_CONCURRENT} concurrent)`);

  async function loop() {
    loopRunning = true;
    lastLoopRun = Date.now();
    try {
      await refillQueueIfLow();

      const launched: Promise<boolean>[] = [];
      const slotsAvailable = MAX_CONCURRENT - activeWorkers;
      for (let i = 0; i < slotsAvailable; i++) {
        launched.push(processNextJob());
      }
      const results = await Promise.all(launched);
      const anyProcessed = results.some(r => r);
      if (anyProcessed) {
        const moreQueued = await storage.getQueuedDftJobs(1);
        if (moreQueued.length > 0) {
          workerLoopTimer = setTimeout(loop, 2000);
          loopRunning = false;
          return;
        }
      }
    } catch (err: any) {
      console.log(`[DFT-Queue] Worker loop error: ${err.message}`);
    }
    loopRunning = false;
    workerLoopTimer = setTimeout(loop, POLL_INTERVAL_MS);
  }

  workerLoopTimer = setTimeout(loop, 5000);

  if (!watchdogTimer) {
    watchdogTimer = setInterval(() => {
      const elapsed = Date.now() - lastLoopRun;
      if (!loopRunning && elapsed > POLL_INTERVAL_MS * 3 && lastLoopRun > 0) {
        console.log(`[DFT-Queue] WATCHDOG: Worker loop appears dead (last run ${Math.round(elapsed/1000)}s ago), restarting`);
        if (workerLoopTimer) {
          clearTimeout(workerLoopTimer);
          workerLoopTimer = null;
        }
        workerLoopTimer = setTimeout(loop, 1000);
      }
    }, POLL_INTERVAL_MS * 2);
  }
}

export function stopDFTWorkerLoop() {
  if (workerLoopTimer) {
    clearTimeout(workerLoopTimer);
    workerLoopTimer = null;
    console.log("[DFT-Queue] DFT worker loop stopped");
  }
  if (watchdogTimer) {
    clearInterval(watchdogTimer);
    watchdogTimer = null;
  }
}

export function getStageMetrics(): Record<PipelineStage, StageMetrics> {
  return JSON.parse(JSON.stringify(stageMetrics));
}

export async function getDFTQueueStats() {
  const dbStats = await storage.getDftJobStats();
  const recentJobs = await storage.getRecentDftJobs(10);

  const dbSucceeded = dbStats.completed || 0;
  const dbFailed = dbStats.failed || 0;
  const adjustedFailed = Math.max(0, dbFailed - staleJobsCleanedCount);
  const dbCompleted = dbSucceeded + adjustedFailed;

  const currentActiveFormulas = Array.from(activeJobs.values()).map(j => j.formula);

  return {
    ...dbStats,
    totalProcessed: dbCompleted,
    totalSucceeded: dbSucceeded,
    totalFailed: adjustedFailed,
    staleJobsCleaned: staleJobsCleanedCount,
    isProcessing: activeWorkers > 0,
    activeWorkers,
    maxConcurrent: MAX_CONCURRENT,
    currentFormula: currentActiveFormulas[0] || null,
    activeFormulas: currentActiveFormulas,
    qeAvailable: isQEAvailable(),
    stageFailures: getStageFailureCounts(),
    pipelineStages: stageMetrics,
    recentJobs: recentJobs.map(j => {
      const out = j.outputData as any;
      return {
        id: j.id,
        formula: j.formula,
        status: j.status,
        jobType: j.jobType,
        priority: j.priority,
        createdAt: j.createdAt instanceof Date ? j.createdAt.toISOString() : (j.createdAt || null),
        startedAt: j.startedAt instanceof Date ? j.startedAt.toISOString() : (j.startedAt || null),
        completedAt: j.completedAt instanceof Date ? j.completedAt.toISOString() : (j.completedAt || null),
        wallTime: out?.wallTimeTotal || null,
        converged: out?.scf?.converged || false,
        totalEnergy: out?.scf?.totalEnergy || null,
        retryCount: out?.retryCount || null,
        xtbPreRelaxed: out?.xtbPreRelaxed || null,
        vcRelaxed: out?.vcRelaxed || null,
        ppValidated: out?.ppValidated || null,
        rejectionReason: out?.rejectionReason || null,
        failureStage: out?.failureStage || null,
        prototypeUsed: out?.prototypeUsed || null,
        kPoints: out?.kPoints || null,
        error: j.errorMessage || null,
      };
    }),
  };
}
