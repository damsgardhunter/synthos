import { generateHybridCandidates, getHybridGeneratorStats, type CrystalStructure } from "./hybrid-structure-generator";
import { getFailureDBStats, shouldAvoidStructure, recordStructureFailure, type FailureReason } from "./structure-failure-db";
import { getDatasetStats, getTrainingData } from "./crystal-structure-dataset";
import { trainStructurePredictor, getStructurePredictorStats } from "./structure-predictor-ml";
import { spawnMLTraining } from "../workers/ml-training-bridge";
import { runQuantumEnginePipeline } from "../dft/quantum-engine-pipeline";
import { predictStabilityScreen } from "./stability-predictor";

interface LearningCycleResult {
  cycleId: number;
  timestamp: number;
  candidatesGenerated: number;
  candidatesScreened: number;
  candidatesPassed: number;
  candidatesFailed: number;
  datasetSizeBefore: number;
  datasetSizeAfter: number;
  modelsRetrained: boolean;
  retrainedModels: string[];
  durationMs: number;
  bestCandidate: { formula: string; predictedTc: number } | null;
}

let totalCycles = 0;
let totalCandidatesGenerated = 0;
let totalCandidatesScreened = 0;
let totalCandidatesPassed = 0;
let totalCandidatesFailed = 0;
let totalRetrains = 0;
let lastRetrainDatasetSize = 0;
let lastCycleResult: LearningCycleResult | null = null;
const cycleHistory: LearningCycleResult[] = [];
const MAX_HISTORY = 50;
const RETRAIN_THRESHOLD = 50;

export async function runStructureLearningCycle(
  batchSize: number = 10,
  targetPressure?: number,
  targetSystem?: string,
): Promise<LearningCycleResult> {
  const startTime = Date.now();
  totalCycles++;
  const cycleId = totalCycles;

  const datasetStatsBefore = getDatasetStats();
  const datasetSizeBefore = datasetStatsBefore.totalCount;

  const candidates = await generateHybridCandidates(batchSize, {
    mutationRate: 0.7,
    mlWeight: 0.6,
    targetPressure,
    targetSystem,
  });

  totalCandidatesGenerated += candidates.length;

  const validCandidates: CrystalStructure[] = [];
  for (const cand of candidates) {
    if (!cand.valid) continue;
    const avoid = shouldAvoidStructure(cand.formula, cand.lattice, cand.crystalSystem);
    if (avoid) continue;
    const stability = await predictStabilityScreen(cand.formula);
    if (!stability.isLikelyStable) {
      recordStructureFailure({
        formula: cand.formula,
        failureReason: stability.phononStabilityProb < 0.3 ? "unstable_phonons" : "high_formation_energy",
        failedAt: Date.now(),
        source: "learning_loop",
        stage: 0,
        details: `Stability pre-screen rejected: FE=${stability.predictedFormationEnergy}, phonon=${stability.phononStabilityProb}`,
        formationEnergy: stability.predictedFormationEnergy,
      });
      continue;
    }
    validCandidates.push(cand);
  }

  let passed = 0;
  let failed = 0;
  let bestCandidate: { formula: string; predictedTc: number } | null = null;

  for (const cand of validCandidates.slice(0, 5)) {
    try {
      // skipXTB=true: inline xTB takes 30-90s per formula; DFT queue handles it asynchronously.
      const result = await runQuantumEnginePipeline(cand.formula, targetPressure ?? 0, true);
      const entry = result.entry;

      if (entry.scfConverged && entry.tc > 0 && entry.isMetallic) {
        passed++;
        totalCandidatesPassed++;

        if (!bestCandidate || entry.tc > bestCandidate.predictedTc) {
          bestCandidate = { formula: cand.formula, predictedTc: entry.tc };
        }
      } else {
        failed++;
        totalCandidatesFailed++;

        let failureReason: FailureReason = "geometry_rejected";
        if (!entry.scfConverged) {
          failureReason = "scf_divergence";
        } else if (!entry.isMetallic) {
          failureReason = "non_metallic";
        } else if (!entry.isPhononStable) {
          failureReason = "unstable_phonons";
        } else if (entry.formationEnergy !== null && entry.formationEnergy > 0.5) {
          failureReason = "high_formation_energy";
        }

        recordStructureFailure({
          formula: cand.formula,
          failureReason,
          failedAt: Date.now(),
          source: "learning_loop",
          stage: 0,
          details: `Learning loop cycle ${cycleId}: DFT pipeline ${entry.scfConverged ? "converged but Tc<=0 or non-metallic" : "did not converge"}`,
          bandGap: entry.bandGap ?? undefined,
          formationEnergy: entry.formationEnergy ?? undefined,
        });
      }
    } catch (err) {
      failed++;
      totalCandidatesFailed++;
      recordStructureFailure({
        formula: cand.formula,
        failureReason: "scf_divergence",
        failedAt: Date.now(),
        source: "learning_loop",
        stage: 0,
        details: `Learning loop cycle ${cycleId}: ${err instanceof Error ? err.message.slice(0, 150) : "unknown error"}`,
      });
    }
  }

  totalCandidatesScreened += validCandidates.slice(0, 5).length;

  const datasetStatsAfter = getDatasetStats();
  const datasetSizeAfter = datasetStatsAfter.totalCount;

  let modelsRetrained = false;
  const retrainedModels: string[] = [];

  const growth = datasetSizeAfter - lastRetrainDatasetSize;
  if (growth >= RETRAIN_THRESHOLD) {
    try {
      trainStructurePredictor();
      retrainedModels.push("structure-predictor-ml");
    } catch (err: any) {
      console.debug(`[structure-loop] trainStructurePredictor failed: ${err?.message ?? err}`);
    }

    // Stagger VAE and diffusion model training in worker threads so they never block the event loop
    try {
      setTimeout(() => { spawnMLTraining("init-crystal-vae").catch(() => {}); }, 5000);
      retrainedModels.push("crystal-vae");
    } catch (err: any) {
      console.debug(`[structure-loop] spawnMLTraining crystal-vae setup failed: ${err?.message ?? err}`);
    }

    try {
      setTimeout(() => { spawnMLTraining("init-diffusion-model").catch(() => {}); }, 15000);
      retrainedModels.push("crystal-diffusion-model");
    } catch (err: any) {
      console.debug(`[structure-loop] spawnMLTraining diffusion-model setup failed: ${err?.message ?? err}`);
    }

    if (retrainedModels.length > 0) {
      modelsRetrained = true;
      totalRetrains++;
      lastRetrainDatasetSize = datasetSizeAfter;
    }
  }

  const result: LearningCycleResult = {
    cycleId,
    timestamp: Date.now(),
    candidatesGenerated: candidates.length,
    candidatesScreened: validCandidates.slice(0, 5).length,
    candidatesPassed: passed,
    candidatesFailed: failed,
    datasetSizeBefore,
    datasetSizeAfter,
    modelsRetrained,
    retrainedModels,
    durationMs: Date.now() - startTime,
    bestCandidate,
  };

  lastCycleResult = result;
  cycleHistory.push(result);
  if (cycleHistory.length > MAX_HISTORY) {
    cycleHistory.shift();
  }

  return result;
}

export function getStructureLearningStats(): {
  totalCycles: number;
  totalCandidatesGenerated: number;
  totalCandidatesScreened: number;
  totalCandidatesPassed: number;
  totalCandidatesFailed: number;
  passRate: number;
  totalRetrains: number;
  lastRetrainDatasetSize: number;
  retrainThreshold: number;
  lastCycleResult: LearningCycleResult | null;
  recentCycles: LearningCycleResult[];
  failureDBStats: ReturnType<typeof getFailureDBStats>;
  datasetStats: ReturnType<typeof getDatasetStats>;
  hybridGeneratorStats: ReturnType<typeof getHybridGeneratorStats>;
  structurePredictorStats: ReturnType<typeof getStructurePredictorStats>;
} {
  const screened = totalCandidatesScreened || 1;
  return {
    totalCycles,
    totalCandidatesGenerated,
    totalCandidatesScreened,
    totalCandidatesPassed,
    totalCandidatesFailed,
    passRate: totalCandidatesPassed / screened,
    totalRetrains,
    lastRetrainDatasetSize,
    retrainThreshold: RETRAIN_THRESHOLD,
    lastCycleResult,
    recentCycles: cycleHistory.slice(-10),
    failureDBStats: getFailureDBStats(),
    datasetStats: getDatasetStats(),
    hybridGeneratorStats: getHybridGeneratorStats(),
    structurePredictorStats: getStructurePredictorStats(),
  };
}
