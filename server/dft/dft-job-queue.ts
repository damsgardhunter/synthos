import { storage } from "../storage";
import { db } from "../db";
import { runFullDFT, isQEAvailable, isFormulaBlocked, getStageFailureCounts, scheduleQEAvailabilityProbe } from "./qe-worker";
export { scheduleQEAvailabilityProbe };
import { scheduleParetoRecompute } from "../inverse/pareto-optimizer";
import type { QEFullResult, QEDFPTResult } from "./qe-worker";
const SCF_RETRY_START_ATTEMPT = 3;
import type { DftJob } from "@shared/schema";
import { recordStructureFailure } from "../crystal/structure-failure-db";
import { isValidFormula, parseFormulaCounts, classifyFamily } from "../learning/utils";
import { recordSynthesisResult } from "../synthesis/synthesis-learning-db";

const POLL_INTERVAL_MS = 30_000;
const MAX_CONCURRENT = 3;
const MIN_QUEUE_SIZE = 50;

// Set by engine.ts when the SG prototype sweep starts/ends. While true, the GCP
// refill loop skips refillQueueIfLow() to avoid competing with 20 concurrent
// SG sweep DB queries and exhausting the Neon pool.
let _sgSweepActive = false;
export function setSGSweepActive(active: boolean): void {
  _sgSweepActive = active;
}
const REFILL_BATCH_SIZE = 100;
const IMAGINARY_PHONON_THRESHOLD_CM1 = -10.0;
const EXPLORATION_FRACTION = 0.3;
const JOB_STAGGER_MS = 2000;
const STALE_CLEANUP_INTERVAL_MS = 5 * 60_000;

// Stability gate for the EXPLOIT pool only (eV/atom above convex hull):
//   0.5 eV/atom — strict; Miedema ±0.2 eV/atom error means up to 0.5 eV is
//   plausibly stable. Only high-confidence exploit candidates are gated here.
//
// The EXPLORE pool has NO hull-distance gate:
//   Exploration deliberately targets uncertain / metastable regions. Miedema
//   estimates systematically over-penalise novel phases (A15, high-pressure
//   hydrides, Laves) by 0.3–0.8 eV/atom. Applying any hard gate to explore
//   causes complete queue starvation. DFT itself is the stability filter.
//
//   Null hull distance (unknown) is always admitted in both pools.
const STABILITY_GATE_EV_ATOM = 0.5; // exploit pool only; explore pool is ungated

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

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
  skipStabilityGate: boolean = false,
  pressureGpa?: number,
): Promise<DftJob | null> {
  if (!isValidFormula(formula)) {
    console.log(`[DFT-Queue] Formula ${formula} rejected: invalid or contains noble gas`);
    return null;
  }

  // Atom count pre-filter — mirrors qe-worker limit of 16 atoms
  const formulaAtomCounts = parseFormulaCounts(formula);
  const formulaTotalAtoms = Object.values(formulaAtomCounts).reduce((s, v) => s + v, 0);
  if (formulaTotalAtoms > 16) {
    console.log(`[DFT-Queue] Formula ${formula} rejected: ${formulaTotalAtoms} atoms > 16 atom limit`);
    return null;
  }

  if (isFormulaBlocked(formula)) {
    console.log(`[DFT-Queue] Formula ${formula} blocked due to repeated failures, skipping`);
    return null;
  }

  // Stability gate: skip if best convex hull distance exceeds threshold
  if (!skipStabilityGate) {
    const structureMap = await storage.getCrystalStructuresByFormulas([formula]);
    const structures = structureMap.get(formula) ?? [];
    if (structures.length > 0) {
      // Null hull distance means unknown — allow through (same policy as bulk refill loop).
      const bestHullDist = Math.min(...structures.map(s => s.convexHullDistance ?? 0));
      if (bestHullDist > STABILITY_GATE_EV_ATOM) {
        console.log(`[DFT-Queue] Formula ${formula} rejected by stability gate: hull dist ${bestHullDist.toFixed(3)} eV/atom > ${STABILITY_GATE_EV_ATOM}`);
        return null;
      }
    }
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
    inputData: { formula, candidateId, requestedAt: new Date().toISOString(), ...(pressureGpa != null ? { pressureGpa } : {}) },
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

/**
 * Re-prioritize all queued DFT jobs based on the candidate's current acquisition
 * score (predictedTc × uncertainty). This ensures the GCP worker always picks
 * up the highest-value candidates first, even if the queue was filled before
 * the models were retrained.
 *
 * Also promotes the top acquisition-score candidates into the queue if they
 * aren't already there.
 */
export async function syncQueueWithAcquisitionRanking(): Promise<{ reprioritized: number; promoted: number }> {
  let reprioritized = 0;
  let promoted = 0;

  try {
    // 1. Re-prioritize existing queued jobs based on their candidate's current scores
    const queuedJobs = await storage.getQueuedDftJobs(200);
    const queuedFormulas = new Set(queuedJobs.map(j => j.formula));

    // Look up candidate scores for queued formulas
    const candidates = await storage.getSuperconductorCandidatesByFormulas(
      queuedJobs.map(j => j.formula)
    );
    const candidateMap = new Map<string, typeof candidates[0]>();
    for (const c of candidates) {
      const existing = candidateMap.get(c.formula);
      if (!existing || (c.ensembleScore ?? 0) > (existing.ensembleScore ?? 0)) {
        candidateMap.set(c.formula, c);
      }
    }

    for (const job of queuedJobs) {
      // Don't touch manually promoted (priority >= 91) or critical (9999) jobs
      if ((job.priority ?? 0) >= 91) continue;

      const c = candidateMap.get(job.formula);
      if (!c) continue;

      // Compute acquisition priority: 50% Tc (norm 300K) + 50% uncertainty
      const tcNorm = Math.min(1.0, (c.predictedTc ?? 0) / 300);
      const unc = c.uncertaintyEstimate ?? 0;
      const acquisitionScore = 0.5 * tcNorm + 0.5 * unc;
      const newPriority = Math.min(90, Math.round(acquisitionScore * 100));

      if (newPriority !== (job.priority ?? 0)) {
        await storage.updateDftJobIfStatus(job.id, "queued", { priority: newPriority } as any);
        reprioritized++;
      }
    }

    // 2. Promote top acquisition candidates that aren't in the queue yet
    const topCandidates = await storage.getCandidatesForDFTRefill(50);
    const sorted = topCandidates
      .filter(c => {
        const mlFeatures = (c.mlFeatures as Record<string, any>) ?? {};
        if (mlFeatures.qeDFT) return false;
        const counts = parseFormulaCounts(c.formula);
        const totalAtoms = Object.values(counts).reduce((s, v) => s + v, 0);
        return totalAtoms <= 16;
      })
      .map(c => {
        const tcNorm = Math.min(1.0, (c.predictedTc ?? 0) / 300);
        const unc = c.uncertaintyEstimate ?? 0;
        return { ...c, acquisitionScore: 0.5 * tcNorm + 0.5 * unc };
      })
      .sort((a, b) => b.acquisitionScore - a.acquisitionScore)
      .slice(0, 10);

    for (const c of sorted) {
      if (queuedFormulas.has(c.formula)) continue;
      if (isFormulaBlocked(c.formula)) continue;
      if (hasLmaxxIncompatibleElement(c.formula)) continue;

      try {
        const { activeJob } = await storage.hasActiveOrRecentFailedDftJobs(c.formula);
        if (activeJob) continue;

        const priority = Math.min(90, Math.round(c.acquisitionScore * 100));
        const job = await submitDFTJob(c.formula, null, priority, "scf", true);
        if (job) {
          promoted++;
          queuedFormulas.add(c.formula);
        }
      } catch {}
    }

    if (reprioritized > 0 || promoted > 0) {
      console.log(`[DFT-Queue] Sync with acquisition ranking: ${reprioritized} reprioritized, ${promoted} new promoted`);
    }
  } catch (err: any) {
    console.error(`[DFT-Queue] syncQueueWithAcquisitionRanking error: ${err.message?.slice(0, 100)}`);
  }

  return { reprioritized, promoted };
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
    // scf_tsc: high-priority TSC candidate; forces nspin=2 for spin-orbit gap physics
    const isTSCJob = job.jobType === "scf_tsc";
    const startAttempt = isRetry ? (inputData?.startAttempt ?? SCF_RETRY_START_ATTEMPT) : 0;
    const jobPressureGpa: number | undefined =
      typeof inputData?.pressureGpa === "number" && inputData.pressureGpa > 20
        ? inputData.pressureGpa
        : undefined;

    // Look up ensembleScore so the worker can decide whether to run full DFPT EPC.
    let ensembleScore: number | undefined;
    try {
      const candidates = await storage.getSuperconductorsByFormula(job.formula);
      const bestScore = candidates.reduce((best, c) => Math.max(best, c.ensembleScore ?? 0), 0);
      if (bestScore > 0) ensembleScore = bestScore;
    } catch { /* non-fatal — worker will skip DFPT if undefined */ }

    if (isQEAvailable()) {
      dftResult = await runFullDFT(job.formula, { startAttempt, pressureGpa: jobPressureGpa, ensembleScore, forceSpin: isTSCJob });
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

        if (dftResult.vcRelaxed && dftResult.relaxedLatticeA) {
          scalarUpdates.latticeParams = { a: dftResult.relaxedLatticeA };
        }

        const bandData = dftResult.bandStructure;
        const mlFeaturePatch: Record<string, any> = {
          qeDFT: true,
          qeConverged: dftResult.scf?.converged || false,
          qeVcRelaxed: dftResult.vcRelaxed ?? false,
          qeRelaxedLatticeA: dftResult.relaxedLatticeA ?? null,
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
          qeDFTPlusU: dftResult.qeDFTPlusU ?? false,
          qeDFTPlusUTcModifier: dftResult.dftPlusUTcModifier ?? null,
          // DFPT electron-phonon coupling — present only for ensembleScore > 0.7 candidates.
          // These are DFT-quality λ/ω_log values; qeDFPTTc is the publishable-grade Tc.
          // The GNN uses qeDFPTTc as target label instead of ML-estimated predictedTc.
          qeDFPTLambda: dftResult.dfpt?.lambda ?? null,
          qeDFPTOmegaLog: dftResult.dfpt?.omegaLog ?? null,
          qeDFPTTc: dftResult.dfpt?.tcBest ?? null,
          qeDFPTSource: dftResult.dfpt?.source ?? null,
          qeDFPTPhConverged: dftResult.dfpt?.phConverged ?? null,
        };

        // For top candidates where DFPT completed successfully, write the DFPT Tc
        // back as the authoritative predictedTc — it replaces the ML estimate.
        if (dftResult.dfpt?.tcBest && dftResult.dfpt.tcBest > 0) {
          scalarUpdates.predictedTc = dftResult.dfpt.tcBest;
          console.log(`[DFT-Queue] ${job.formula} DFPT Tc ${dftResult.dfpt.tcBest.toFixed(1)} K written as predictedTc (λ=${dftResult.dfpt.lambda.toFixed(3)}, source=${dftResult.dfpt.source})`);
        }

        const updated = await storage.updateCandidateByFormulaDft(job.formula, scalarUpdates, mlFeaturePatch);
        if (updated) {
          console.log(`[DFT-Queue] Updated candidate ${job.formula} with QE DFT results (single-query upsert)`);
        }

        // Feed the completed DFT run into the synthesis learning DB so
        // getSynthesisPatterns() can identify which pressure/temperature/
        // quenching combinations produce phonon-stable structures.
        try {
          const phononStable = dftResult.phonon
            ? dftResult.phonon.lowestFrequency > IMAGINARY_PHONON_THRESHOLD_CM1
            : true; // assume stable when phonon step was skipped (SCF-only run)
          const pressureUsed = jobPressureGpa ?? 0;
          const isHighPressure = pressureUsed >= 20;

          const sv = {
            temperature: isHighPressure ? 300 : 1200,   // DAC runs near RT; solid-state at ~1200 K
            pressure: pressureUsed,
            coolingRate: isHighPressure ? 0 : 5,        // HP: quench; solid-state: 5 K/min
            annealTime: isHighPressure ? 0 : 24,        // HP: no anneal; solid-state: 24 h
            currentDensity: 0,
            magneticField: 0,
            thermalCycles: isHighPressure ? 1 : 3,
            strain: dftResult.vcRelaxed ? 0.05 : 0,
            oxygenPressure: 0,
            synthesisMethod: (isHighPressure ? "high-pressure" : "solid-state") as
              "high-pressure" | "solid-state",
          };

          const resultTc = dftResult.dfpt?.tcBest
            ?? (scalarUpdates.predictedTc as number | undefined)
            ?? 0;
          // stability: 1.0 = phonon-stable SCF success; 0.4 = imaginary modes present
          const stability = phononStable ? 1.0 : 0.4;
          const materialClass = classifyFamily(job.formula);

          recordSynthesisResult(job.formula, materialClass, sv, resultTc, stability, 1.0);
        } catch { /* non-fatal */ }

        // Recompute Pareto ranks for top-500 candidates after each DFT cycle.
        // Debounced so rapid successive completions collapse to one recompute.
        // The onComplete callback writes paretoRank/paretoFront back to each
        // candidate's mlFeatures so the UI can display the Pareto frontier.
        scheduleParetoRecompute(
          async () => {
            const top = await storage.getTopCandidatesMerged(490, 10);
            return top.map(c => ({
              formula: c.formula,
              predictedTc: c.predictedTc,
              decompositionEnergy: c.decompositionEnergy,
              mlFeatures: c.mlFeatures as Record<string, any> | null,
              ensembleScore: c.ensembleScore,
            }));
          },
          8_000,
          async (results) => {
            for (const r of results) {
              try {
                await storage.updateCandidateByFormulaDft(
                  r.formula,
                  {},
                  { paretoRank: r.rank, paretoFront: r.isFront },
                );
              } catch { /* non-fatal — best-effort write-back */ }
            }
          },
        );
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
      } catch (err: any) {
        console.debug(`[DFT-Queue] Auto-requeue failed for ${job.formula}: ${err?.message ?? err}`);
      }

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

async function cleanupOverstoichiometricJobs() {
  try {
    const queued = await storage.getDftJobsByStatus("queued");
    let cancelled = 0;
    for (const job of queued) {
      const counts = parseFormulaCounts(job.formula);
      const total = Object.values(counts).reduce((s, v) => s + v, 0);
      if (total > 16) {
        await storage.updateDftJob(job.id, {
          status: "failed",
          completedAt: new Date(),
          errorMessage: `Pre-filter rejected: Too many atoms (${total}), max 16`,
        } as any);
        cancelled++;
      }
    }
    if (cancelled > 0) {
      console.log(`[DFT-Queue] Cancelled ${cancelled} queued overstoichiometric jobs (>16 atoms)`);
    }
  } catch (err: any) {
    console.log(`[DFT-Queue] Overstoichiometric cleanup error: ${err.message}`);
  }
}

// Elements whose PPs require lmaxx>3, which our QE build doesn't support.
// Reject at queue level to avoid wasting a GCP worker slot on immediate failure.
const LMAXX_INCOMPATIBLE_QUEUE = new Set([
  // Lanthanides with 4f projectors
  "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
  // Actinides with 5f projectors
  "Th", "Pa", "U", "Np", "Pu", "Am",
]);

async function cleanupFBlockElementJobs() {
  try {
    const queued = await storage.getDftJobsByStatus("queued");
    let cancelled = 0;
    for (const job of queued) {
      const counts = parseFormulaCounts(job.formula);
      const elements = Object.keys(counts);
      const badEl = elements.find(el => LMAXX_INCOMPATIBLE_QUEUE.has(el));
      if (badEl) {
        await storage.updateDftJob(job.id, {
          status: "failed",
          completedAt: new Date(),
          errorMessage: `Pre-filter rejected: unsupported element ${badEl} (PPs require lmaxx>3; QE rebuild or scalar-relativistic PP needed)`,
        } as any);
        cancelled++;
      }
    }
    if (cancelled > 0) {
      console.log(`[DFT-Queue] Cancelled ${cancelled} queued jobs with f-block elements (lmaxx incompatible)`);
    }
  } catch (err: any) {
    console.log(`[DFT-Queue] F-block cleanup error: ${err.message}`);
  }
}

/** Check if a formula contains any element that will fail the PP pre-filter. */
function hasLmaxxIncompatibleElement(formula: string): boolean {
  const counts = parseFormulaCounts(formula);
  return Object.keys(counts).some(el => LMAXX_INCOMPATIBLE_QUEUE.has(el));
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

    // getCandidatesForDFTRefill filters ensemble_score>0 and qeDFT absent in SQL —
    // avoids the 500-row full scan + in-memory filter that was exhausting the DB pool.
    const candidates = await storage.getCandidatesForDFTRefill(REFILL_BATCH_SIZE * 5);
    const preFiltered = candidates.filter(c => {
      const mlFeatures = (c.mlFeatures as Record<string, any>) ?? {};
      if (mlFeatures.qeDFT) return false;
      if ((c.ensembleScore ?? 0) <= 0) return false;
      // Atom count gate — don't queue overstoichiometric formulas that QE will reject
      const counts = parseFormulaCounts(c.formula);
      const totalAtoms = Object.values(counts).reduce((s, v) => s + v, 0);
      if (totalAtoms > 16) return false;
      // f-block element gate — PPs need lmaxx>3 which our QE build doesn't support
      if (Object.keys(counts).some(el => LMAXX_INCOMPATIBLE_QUEUE.has(el))) return false;
      // Phonon stability gate — skip candidates already flagged as dynamically unstable
      // by the surrogate phonon model or prior QE phonon run. Queuing them wastes DFT
      // slots and produces empty/imaginary frequency results.
      if (mlFeatures.phononStable === false || mlFeatures.qePhononStable === false) return false;
      return true;
    });

    const formulasToCheck = preFiltered.map(c => c.formula);
    const structureMap = await storage.getCrystalStructuresByFormulas(formulasToCheck);

    // Build two pools with separate stability policies.
    // Exploit pool: strict gate (STABILITY_GATE_EV_ATOM = 0.5 eV/atom).
    // Explore pool: NO gate — DFT is the stability filter; Miedema overestimates
    //               metastable phase instability by 0.3–0.8 eV/atom.
    let skippedUnstable = 0;

    const exploitPool: typeof preFiltered = [];
    const explorePool: typeof preFiltered = [];

    for (const c of preFiltered) {
      const structures = structureMap.get(c.formula);
      if (!structures || structures.length === 0) continue; // no structure: skip both pools
      // Null hull distance = unknown → treat as 0 (allow through)
      const bestHullDist = Math.min(...structures.map(s => s.convexHullDistance ?? 0));
      if (bestHullDist <= STABILITY_GATE_EV_ATOM) {
        exploitPool.push(c); // passes strict gate → eligible for both pools
        explorePool.push(c);
      } else {
        // Above exploit gate: only explore (no upper hull-distance limit for explore)
        explorePool.push(c);
        // Count as "unstable for exploit" for diagnostic logging
        skippedUnstable++;
      }
    }

    // Exploit pool: sort by Tc-weighted score (70% predictedTc, 30% ensembleScore).
    // predictedTc is the direct signal from GNN/XGB/physics; ensembleScore can be
    // inflated for structurally exotic compounds that score well on novelty.
    // Cap normalisation at 300 K so hydride Tc values don't dominate unfairly.
    const TC_NORM_SCALE = 300;
    exploitPool.sort((a, b) => {
      const aTc = Math.min(1.0, (a.predictedTc ?? 0) / TC_NORM_SCALE);
      const bTc = Math.min(1.0, (b.predictedTc ?? 0) / TC_NORM_SCALE);
      const aScore = 0.7 * aTc + 0.3 * (a.ensembleScore ?? 0);
      const bScore = 0.7 * bTc + 0.3 * (b.ensembleScore ?? 0);
      return bScore - aScore;
    });
    explorePool.sort((a, b) => {
      const aAcq = (a.uncertaintyEstimate ?? 0) * 0.6 + (a.discoveryScore ?? 0) * 0.4;
      const bAcq = (b.uncertaintyEstimate ?? 0) * 0.6 + (b.discoveryScore ?? 0) * 0.4;
      return bAcq - aAcq;
    });

    const exploitCount = Math.ceil(REFILL_BATCH_SIZE * (1 - EXPLORATION_FRACTION));
    const exploreCount = REFILL_BATCH_SIZE - exploitCount;

    const queuedFormulas = new Set(queued.map(j => j.formula));
    const selectedFormulas = new Set<string>();
    let submitted = 0;
    let exploitSubmitted = 0;
    let exploreSubmitted = 0;
    let refillErrors = 0;
    let skippedNoStructure = 0;

    async function trySubmit(candidate: typeof exploitPool[0], source: "exploit" | "explore"): Promise<boolean> {
      if (queuedFormulas.has(candidate.formula)) return false;
      if (selectedFormulas.has(candidate.formula)) return false;
      if (isFormulaBlocked(candidate.formula)) return false;

      try {
        const { activeJob, recentValidatedFailures } = await storage.hasActiveOrRecentFailedDftJobs(candidate.formula);
        if (activeJob) return false;
        if (recentValidatedFailures >= 3) return false;

        const acquisitionPriority = source === "explore"
          ? Math.round(((candidate.uncertaintyEstimate ?? 0) * 0.6 + (candidate.discoveryScore ?? 0) * 0.4) * 100)
          // Exploit: 70% predictedTc (normalised to 300 K) + 30% ensembleScore.
          // Cap at 90 so manually promoted candidates (91–99) always win.
          : Math.min(90, Math.round(
              (0.7 * Math.min(1.0, (candidate.predictedTc ?? 0) / TC_NORM_SCALE) +
               0.3 * (candidate.ensembleScore ?? 0)) * 100
            ));

        const numericCandidateId = typeof candidate.id === "string" && /^\d+$/.test(candidate.id)
          ? parseInt(candidate.id, 10)
          : typeof candidate.id === "number" ? candidate.id : null;

        await storage.insertDftJob({
          formula: candidate.formula,
          candidateId: numericCandidateId,
          status: "queued",
          jobType: "scf",
          priority: acquisitionPriority,
          inputData: {
            formula: candidate.formula,
            candidateId: candidate.id,
            requestedAt: new Date().toISOString(),
            source: `queue_refill_${source}`,
          },
        });
        queuedFormulas.add(candidate.formula);
        selectedFormulas.add(candidate.formula);
        stageMetrics.candidate_queue.currentDepth++;
        return true;
      } catch (err: any) {
        refillErrors++;
        if (refillErrors <= 3) {
          console.log(`[DFT-Queue] Refill insert error for ${candidate.formula}: ${err.message}`);
        }
        return false;
      }
    }

    for (const candidate of exploitPool) {
      if (exploitSubmitted >= exploitCount) break;
      if (await trySubmit(candidate, "exploit")) {
        exploitSubmitted++;
        submitted++;
      }
    }

    for (const candidate of explorePool) {
      if (exploreSubmitted >= exploreCount) break;
      if (await trySubmit(candidate, "explore")) {
        exploreSubmitted++;
        submitted++;
      }
    }

    // skippedNoStructure = candidates that had no crystal structure at all
    // (can't appear in either pool since the loop `continue`s on missing structures)
    const withStructure = preFiltered.filter(c => {
      const structures = structureMap.get(c.formula);
      return structures && structures.length > 0;
    }).length;
    skippedNoStructure = preFiltered.length - withStructure;

    if (submitted > 0 || refillErrors > 0 || skippedNoStructure > 0 || skippedUnstable > 0) {
      console.log(
        `[DFT-Queue] Refilled queue with ${submitted} candidates ` +
        `(${exploitSubmitted} exploit + ${exploreSubmitted} explore, queue was ${queueSize}/${MIN_QUEUE_SIZE})` +
        (skippedNoStructure > 0 ? `, ${skippedNoStructure} skipped (no structure)` : "") +
        (skippedUnstable > 0
          ? `, ${skippedUnstable} above exploit gate (>${STABILITY_GATE_EV_ATOM} eV/atom hull, explore-eligible)`
          : "") +
        (refillErrors > 0 ? `, ${refillErrors} insert errors` : ""),
      );
    }
    // After refill, re-prioritize the entire queue so top acquisition candidates
    // are always served first to the GCP worker.
    if (submitted > 0) {
      syncQueueWithAcquisitionRanking().catch(() => {});
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
let lastStaleCleanup = 0;

// High-priority compounds that must always be at the front of the queue.
// These are submitted at priority 9999 on startup if not already running/completed.
const CRITICAL_PRIORITY_FORMULAS: string[] = [
  // Ternary high-P hydrides (flagship discoveries)
  "Li2LaH12", "LaH11Li2", "LaH12", "YH9Na2",
  // Binary high-P hydrides (benchmark superconductors)
  "LaH10", "CaH6", "YH6", "H3S", "YH9", "ThH10", "CeH9",
  // Conventional superconductors (validation targets)
  "MgB2", "Nb3Sn", "Nb3Ge", "V3Si",
  // Iron-based (pnictide/chalcogenide families)
  "BaFe2As2", "LaFeAsO", "FeSe", "LiFeAs",
  // Cuprates
  "YBa2Cu3O7", "La2CuO4", "Bi2Sr2CaCu2O8",
  // Heavy-fermion / unconventional
  "CeCoIn5", "Sr2RuO4",
];

async function bootstrapCriticalCandidates(): Promise<void> {
  for (const formula of CRITICAL_PRIORITY_FORMULAS) {
    try {
      const { activeJob } = await storage.hasActiveOrRecentFailedDftJobs(formula);
      if (activeJob) {
        // Already in queue — bump priority if it's lower
        if ((activeJob.status === "queued") && (activeJob.priority ?? 0) < 9999) {
          await storage.updateDftJobIfStatus(activeJob.id, "queued", { priority: 9999 } as any);
          console.log(`[DFT-Queue] Critical bootstrap: promoted existing job #${activeJob.id} for ${formula} to priority 9999`);
        } else {
          console.log(`[DFT-Queue] Critical bootstrap: ${formula} already ${activeJob.status}, skipping`);
        }
        continue;
      }
      const job = await submitDFTJob(formula, null, 9999, "scf", true);
      if (job) {
        console.log(`[DFT-Queue] Critical bootstrap: queued ${formula} at priority 9999 (job #${job.id})`);
      }
    } catch (err: any) {
      console.warn(`[DFT-Queue] Critical bootstrap failed for ${formula}: ${err.message?.slice(0, 80)}`);
    }
  }
}

export function startDFTWorkerLoop() {
  if (workerLoopTimer) return;

  // When DFT is offloaded to GCP, the local server only submits jobs; the GCP
  // worker picks them up via the shared Neon DB queue. Skip local processing
  // but still run the queue refill loop so GCP always has work to pick up.
  if (process.env.OFFLOAD_DFT_TO_GCP === "true") {
    console.log("[DFT-Queue] OFFLOAD_DFT_TO_GCP=true — running queue-refill only (GCP worker handles calculations)");
    // One-time cleanup of invalid jobs already in the queue
    cleanupOverstoichiometricJobs().catch(() => {});
    cleanupFBlockElementJobs().catch(() => {});

    // Ensure critical candidates are at the front before the regular refill runs
    setTimeout(() => bootstrapCriticalCandidates().catch(() => {}), 3_000);

    let _gcpRefillFailures = 0;
    async function gcpRefillLoop() {
      try {
        if (_sgSweepActive) {
          // SG sweep is running — skip refill to avoid competing with its 20 concurrent
          // DB queries and exhausting the Neon pool. Refill will run next cycle (60s).
          console.log("[DFT-Queue] GCP refill skipped — SG sweep active");
        } else {
          await refillQueueIfLow();
        }
        _gcpRefillFailures = 0; // reset on success
      } catch (err: any) {
        _gcpRefillFailures++;
        console.warn("[DFT-Queue] GCP refill error:", err.message);
      }
      // Exponential backoff on consecutive failures so a saturated DB pool doesn't
      // get hammered by repeated refill retries (max 10 min back-off).
      const delay = _gcpRefillFailures > 0
        ? Math.min(60_000 * Math.pow(2, _gcpRefillFailures - 1), 600_000)
        : 60_000;
      setTimeout(gcpRefillLoop, delay);
    }
    setTimeout(gcpRefillLoop, 5_000); // first refill after 5s
    return;
  }

  if (!isQEAvailable()) {
    console.log("[DFT-Queue] WARNING: Quantum ESPRESSO pw.x not found, DFT queue will run but calculations may fail");
  } else {
    console.log("[DFT-Queue] Quantum ESPRESSO detected at pw.x, full DFT worker ready");
  }

  cleanupStaleJobs().then(() => { lastStaleCleanup = Date.now(); }).catch(() => {});
  cleanupOverstoichiometricJobs().catch(() => {});
  setTimeout(() => bootstrapCriticalCandidates().catch(() => {}), 3_000);

  console.log(`[DFT-Queue] Starting DFT worker loop (poll every 30s, max ${MAX_CONCURRENT} concurrent)`);

  async function loop() {
    if (loopActive) {
      return;
    }
    loopActive = true;
    lastLoopHeartbeat = Date.now();
    try {
      if (Date.now() - lastStaleCleanup > STALE_CLEANUP_INTERVAL_MS) {
        await cleanupStaleJobs();
        lastStaleCleanup = Date.now();
      }

      await refillQueueIfLow();

      const slotsAvailable = MAX_CONCURRENT - activeWorkers;
      const launched: Promise<boolean>[] = [];
      for (let i = 0; i < slotsAvailable; i++) {
        if (i > 0) await delay(JOB_STAGGER_MS);
        launched.push(processNextJob());
        lastLoopHeartbeat = Date.now();
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
      loopActive = false;
      // Back off longer on connection errors — pool may be saturated
      const isConn = /connection|timeout|ECONNRESET|terminated/i.test(err.message ?? "");
      workerLoopTimer = setTimeout(loop, isConn ? POLL_INTERVAL_MS * 3 : POLL_INTERVAL_MS);
      return;
    }
    loopActive = false;
    workerLoopTimer = setTimeout(loop, POLL_INTERVAL_MS);
  }

  workerLoopTimer = setTimeout(loop, 5000);

  if (!watchdogTimer) {
    watchdogTimer = setInterval(() => {
      const elapsed = Date.now() - lastLoopHeartbeat;
      if (elapsed > 10 * 60 * 1000 && lastLoopHeartbeat > 0 && !watchdogRestarting) {
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
    }, 5 * 60 * 1000);
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
  const result = {} as Record<PipelineStage, StageMetrics>;
  for (const key of Object.keys(stageMetrics) as PipelineStage[]) {
    result[key] = { ...stageMetrics[key] };
  }
  return result;
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

  // Also query DB for running jobs (GCP worker sets status='running' in DB,
  // but the local server's activeWorkers counter stays 0 since DFT is offloaded).
  let runningFormulasFromDB: string[] = [];
  try {
    const runningRows = await db.execute(
      `SELECT formula FROM dft_jobs WHERE status = 'running' ORDER BY priority DESC LIMIT 20`
    );
    runningFormulasFromDB = ((runningRows as any).rows || runningRows || []).map((r: any) => r.formula);
  } catch {}

  // Merge local + GCP-reported running jobs (deduplicate)
  const _seenFormulas = new Set<string>();
  const allActiveFormulas: string[] = [];
  for (const f of currentActiveFormulas.concat(runningFormulasFromDB)) {
    if (!_seenFormulas.has(f)) { _seenFormulas.add(f); allActiveFormulas.push(f); }
  }
  const effectiveActiveWorkers = Math.max(activeWorkers, dbStats.running || 0);

  return {
    ...dbStats,
    totalProcessed: dbCompleted,
    totalSucceeded: dbSucceeded,
    totalFailed: adjustedFailed,
    staleJobsCleaned: staleCount,
    isProcessing: effectiveActiveWorkers > 0,
    activeWorkers: effectiveActiveWorkers,
    maxConcurrent: MAX_CONCURRENT,
    currentFormula: allActiveFormulas[0] || null,
    activeFormulas: allActiveFormulas,
    qeAvailable: isQEAvailable(),
    stabilityGateEvAtom: STABILITY_GATE_EV_ATOM,
    stageFailures: getStageFailureCounts(),
    pipelineStages: stageMetrics,
    recentJobs: recentJobs.map(j => {
      const out = j.outputData as any;
      const ppValid = out?.ppValidated ?? null;
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
        magnetization: out?.scf?.magnetization ?? null,
        retryCount: out?.retryCount || null,
        xtbPreRelaxed: out?.xtbPreRelaxed || null,
        vcRelaxed: out?.vcRelaxed || null,
        ppValidated: ppValid,
        ppMissingElements: ppValid === false ? (out?.ppMissingElements ?? out?.missingPP ?? null) : null,
        rejectionReason: out?.rejectionReason || null,
        failureStage: out?.failureStage || null,
        prototypeUsed: out?.prototypeUsed || null,
        kPoints: out?.kPoints || null,
        phononStable: out?.phonon
          ? out.phonon.lowestFrequency > IMAGINARY_PHONON_THRESHOLD_CM1
          : null,
        lowestPhononFreq: out?.phonon?.lowestFrequency ?? null,
        error: j.errorMessage || null,
      };
    }),
  };
}
