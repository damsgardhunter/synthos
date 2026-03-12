import { storage } from "../storage";
import { runFullDFT, isQEAvailable, isFormulaBlocked, getStageFailureCounts } from "./qe-worker";
import type { QEFullResult } from "./qe-worker";
const SCF_RETRY_START_ATTEMPT = 3;
import type { DftJob } from "@shared/schema";
import { recordStructureFailure } from "../crystal/structure-failure-db";
import { isValidFormula, parseFormulaCounts } from "../learning/utils";

const POLL_INTERVAL_MS = 30_000;
const MAX_CONCURRENT = 3;
const MIN_QUEUE_SIZE = 50;
const REFILL_BATCH_SIZE = 100;
const IMAGINARY_PHONON_THRESHOLD_CM1 = -10.0;

type PipelineStage = "candidate_queue" | "dft_workers" | "phonon_workers" | "epc_workers";

interface StageMetrics {
  currentDepth: number;
  totalProcessed: number;
  totalSucceeded: number;
  totalFailed: number;
  avgProcessingTimeMs: number;
  avgProcessingTimeMsPerAtom: number;
  lastProcessedAt: number | null;
}

function initStageMetrics(): StageMetrics {
  return { currentDepth: 0, totalProcessed: 0, totalSucceeded: 0, totalFailed: 0, avgProcessingTimeMs: 0, avgProcessingTimeMsPerAtom: 0, lastProcessedAt: null };
}

const stageMetrics: Record<PipelineStage, StageMetrics> = {
  candidate_queue: initStageMetrics(),
  dft_workers: initStageMetrics(),
  phonon_workers: initStageMetrics(),
  epc_workers: initStageMetrics(),
};

function updateStageMetrics(stage: PipelineStage, success: boolean, processingTimeMs: number, atomCount: number = 1) {
  const m = stageMetrics[stage];
  m.totalProcessed++;
  if (success) m.totalSucceeded++;
  else m.totalFailed++;
  m.lastProcessedAt = Date.now();
  const prev = m.avgProcessingTimeMs;
  m.avgProcessingTimeMs = prev === 0 ? processingTimeMs : prev * 0.8 + processingTimeMs * 0.2;
  const perAtom = atomCount > 0 ? processingTimeMs / atomCount : processingTimeMs;
  const prevPerAtom = m.avgProcessingTimeMsPerAtom;
  m.avgProcessingTimeMsPerAtom = prevPerAtom === 0 ? perAtom : prevPerAtom * 0.8 + perAtom * 0.2;
}

let activeWorkers = 0;
let workerLoopTimer: ReturnType<typeof setTimeout> | null = null;
let totalProcessed = 0;
let totalSucceeded = 0;
let totalFailed = 0;
const activeJobs = new Map<number, DftJob>();
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

  const { activeJob, recentValidatedFailures } = await storage.hasActiveOrRecentFailedDftJobs(formula);
  if (activeJob) {
    console.log(`[DFT-Queue] Job already queued/running for ${formula}, skipping`);
    return activeJob;
  }
  if (recentValidatedFailures >= 3) {
    console.log(`[DFT-Queue] Formula ${formula} has ${recentValidatedFailures} validated recent failures, skipping`);
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

  const formulaCounts = parseFormulaCounts(job.formula);
  const atomCount = Object.values(formulaCounts).reduce((s, v) => s + v, 0) || 1;
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

    const inputData = job.inputData as any;
    const isRetry = job.jobType === "scf_retry";
    const startAttempt = isRetry ? (inputData?.startAttempt ?? SCF_RETRY_START_ATTEMPT) : 0;

    if (isQEAvailable()) {
      dftResult = await runFullDFT(job.formula, { startAttempt });
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
    updateStageMetrics("dft_workers", scfSuccess, dftElapsed, atomCount);

    if (scfSuccess && dftResult.phonon) {
      const phononSuccess = dftResult.phonon.converged;
      stageMetrics.phonon_workers.currentDepth++;
      updateStageMetrics("phonon_workers", phononSuccess, dftResult.phonon.wallTimeSeconds * 1000, atomCount);
      stageMetrics.phonon_workers.currentDepth--;

      stageMetrics.epc_workers.currentDepth++;
      const phononPhysicallyStable = dftResult.phonon.lowestFrequency > IMAGINARY_PHONON_THRESHOLD_CM1;
      updateStageMetrics("epc_workers", phononSuccess && phononPhysicallyStable, 0, atomCount);
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
        const scalarUpdates: any = {
          dataConfidence: "dft-verified",
        };

        if (dftResult.scf) {
          if (dftResult.scf.bandGap !== null) {
            scalarUpdates.bandGap = dftResult.scf.bandGap;
          }
          if (dftResult.scf.totalEnergyPerAtom) {
            scalarUpdates.formationEnergy = dftResult.scf.totalEnergyPerAtom;
          }
        }

        const bandData = dftResult.bandStructure;
        const mlFeaturePatch: Record<string, any> = {
          qeDFT: true,
          qeConverged: dftResult.scf?.converged || false,
          qeTotalEnergy: dftResult.scf?.totalEnergy,
          qeFermiEnergy: dftResult.scf?.fermiEnergy,
          qePressure: dftResult.scf?.pressure,
          qePhononStable: dftResult.phonon
            ? dftResult.phonon.lowestFrequency > IMAGINARY_PHONON_THRESHOLD_CM1
            : null,
          qePhononLowestFreq: dftResult.phonon?.lowestFrequency ?? null,
          qePhononImaginaryCount: dftResult.phonon?.imaginaryCount ?? 0,
          qePhononFreqs: dftResult.phonon?.frequencies?.length || 0,
          qeMagnetization: dftResult.scf?.magnetization ?? null,
          qeIsMagnetic: dftResult.scf?.magnetization != null
            ? Math.abs(dftResult.scf.magnetization) > 0.5
            : null,
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

        const updated = await storage.updateCandidateByFormulaDft(job.formula, scalarUpdates, mlFeaturePatch);
        if (updated) {
          console.log(`[DFT-Queue] Updated candidate ${job.formula} with QE DFT results (single-query upsert)`);
        }
      } catch (err: any) {
        console.log(`[DFT-Queue] Failed to update candidate: ${err.message}`);
      }

      console.log(`[DFT-Queue] Job #${job.id} completed: ${job.formula} (${dftResult.wallTimeTotal.toFixed(1)}s)`);
    } else {
      totalFailed++;
      const errorDetail = dftResult.error || dftResult.scf?.error || "SCF did not converge";
      console.log(`[DFT-Queue] Job #${job.id} failed: ${job.formula} — ${errorDetail}`);
      try {
        const hasPhysicalInstability = dftResult.phonon
          && dftResult.phonon.hasImaginary
          && dftResult.phonon.lowestFrequency < IMAGINARY_PHONON_THRESHOLD_CM1;
        let failureReason: "unstable_phonons" | "structure_collapse" | "high_formation_energy" | "non_metallic" | "scf_divergence" | "geometry_rejected" = "scf_divergence";
        if (hasPhysicalInstability) failureReason = "unstable_phonons";
        else if (dftResult.error?.includes("geometry")) failureReason = "geometry_rejected";
        recordStructureFailure({
          formula: job.formula,
          failureReason,
          failedAt: Date.now(),
          source: "dft",
          details: errorDetail,
          lowestPhononFreq: dftResult.phonon?.lowestFrequency,
          imaginaryModeCount: dftResult.phonon?.imaginaryCount,
        });

        if (failureReason === "scf_divergence" && !isRetry) {
          const retryJob = await storage.insertDftJob({
            formula: job.formula,
            candidateId: job.candidateId,
            status: "queued",
            jobType: "scf_retry",
            priority: Math.max((job.priority ?? 50) - 10, 1),
            inputData: {
              formula: job.formula,
              candidateId: job.candidateId,
              requestedAt: new Date().toISOString(),
              parentJobId: job.id,
              startAttempt: SCF_RETRY_START_ATTEMPT,
            },
          });
          console.log(`[DFT-Queue] Auto-requeued ${job.formula} as scf_retry job #${retryJob.id} (startAttempt=${SCF_RETRY_START_ATTEMPT})`);
          if (broadcastFn) {
            broadcastFn("dftJobQueued", { jobId: retryJob.id, formula: job.formula, priority: retryJob.priority, retry: true });
          }
        }
      } catch {}

      if (broadcastFn) {
        broadcastFn("dftJobFailed", {
          jobId: job.id,
          formula: job.formula,
          error: errorDetail,
          wallTime: dftResult.wallTimeTotal,
          willRetry: !isRetry && !(dftResult.phonon?.lowestFrequency !== undefined && dftResult.phonon.lowestFrequency < IMAGINARY_PHONON_THRESHOLD_CM1),
        });
      }
    }

    if (scfSuccess && broadcastFn) {
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

    if (broadcastFn) {
      broadcastFn("dftJobFailed", {
        jobId: job.id,
        formula: job.formula,
        error: err.message,
        wallTime: (Date.now() - jobStartTime) / 1000,
        willRetry: false,
      });
    }

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
          errorMessage: "stale_cleanup: Stale job from previous server session (max retries reached)",
        } as any);
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
    let refillErrors = 0;
    for (const candidate of eligible) {
      if (submitted >= REFILL_BATCH_SIZE) break;
      if (queuedFormulas.has(candidate.formula)) continue;
      if (isFormulaBlocked(candidate.formula)) continue;

      try {
        const { activeJob, recentValidatedFailures } = await storage.hasActiveOrRecentFailedDftJobs(candidate.formula);
        if (activeJob) continue;
        if (recentValidatedFailures >= 3) continue;

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
      } catch (err: any) {
        refillErrors++;
        if (refillErrors <= 3) {
          console.log(`[DFT-Queue] Refill insert error for ${candidate.formula}: ${err.message}`);
        }
      }
    }

    if (submitted > 0 || refillErrors > 0) {
      console.log(`[DFT-Queue] Refilled queue with ${submitted} candidates (queue was ${queueSize}/${MIN_QUEUE_SIZE})${refillErrors > 0 ? `, ${refillErrors} insert errors` : ""}`);
    }
    return submitted;
  } catch (err: any) {
    console.log(`[DFT-Queue] Queue refill error: ${err.message}`);
    return 0;
  }
}

let watchdogTimer: ReturnType<typeof setInterval> | null = null;
let lastLoopHeartbeat = 0;
let loopActive = false;
let watchdogRestarting = false;

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
    if (loopActive) {
      return;
    }
    loopActive = true;
    lastLoopHeartbeat = Date.now();
    try {
      await refillQueueIfLow();

      const launched: Promise<boolean>[] = [];
      const slotsAvailable = MAX_CONCURRENT - activeWorkers;
      for (let i = 0; i < slotsAvailable; i++) {
        launched.push(processNextJob());
      }
      const results = await Promise.all(launched);
      lastLoopHeartbeat = Date.now();
      const anyProcessed = results.some(r => r);
      if (anyProcessed) {
        const moreQueued = await storage.getQueuedDftJobs(1);
        if (moreQueued.length > 0) {
          loopActive = false;
          workerLoopTimer = setTimeout(loop, 2000);
          return;
        }
      }
    } catch (err: any) {
      console.log(`[DFT-Queue] Worker loop error: ${err.message}`);
    }
    loopActive = false;
    workerLoopTimer = setTimeout(loop, POLL_INTERVAL_MS);
  }

  workerLoopTimer = setTimeout(loop, 5000);

  if (!watchdogTimer) {
    watchdogTimer = setInterval(() => {
      const elapsed = Date.now() - lastLoopHeartbeat;
      if (elapsed > POLL_INTERVAL_MS * 3 && lastLoopHeartbeat > 0 && !watchdogRestarting) {
        watchdogRestarting = true;
        console.log(`[DFT-Queue] WATCHDOG: Worker loop appears dead (last heartbeat ${Math.round(elapsed/1000)}s ago), restarting`);
        if (workerLoopTimer) {
          clearTimeout(workerLoopTimer);
          workerLoopTimer = null;
        }
        loopActive = false;
        workerLoopTimer = setTimeout(() => {
          watchdogRestarting = false;
          loop();
        }, 1000);
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
  const staleCount = await storage.getDftStaleCleanupCount();
  const recentJobs = await storage.getRecentDftJobs(10);

  const dbSucceeded = dbStats.completed || 0;
  const dbFailed = dbStats.failed || 0;
  const adjustedFailed = Math.max(0, dbFailed - staleCount);
  const dbCompleted = dbSucceeded + adjustedFailed;

  const currentActiveFormulas = Array.from(activeJobs.values()).map(j => j.formula);

  return {
    ...dbStats,
    totalProcessed: dbCompleted,
    totalSucceeded: dbSucceeded,
    totalFailed: adjustedFailed,
    staleJobsCleaned: staleCount,
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
