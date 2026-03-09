import { storage } from "../storage";
import type { SuperconductorCandidate } from "@shared/schema";
import type { EventEmitter } from "./engine";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { invalidateGNNModel, trainGNNSurrogate, trainEnsembleAsync, setCachedEnsemble, ENSEMBLE_SIZE, addDFTTrainingResult, getDFTTrainingDataset, logGNNVersion, getGNNModelVersion } from "./graph-neural-net";
import { resolveDFTFeatures, describeDFTSources } from "./dft-feature-resolver";
import { extractFeatures } from "./ml-predictor";
import { gbPredict, gbPredictWithUncertainty, incorporateFailureData, incorporateDFTResult, retrainXGBoostFromEvaluated, validateModel, getEvaluatedDatasetStats } from "./gradient-boost";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { computeDiscoveryScore } from "./family-filters";
import { recordEvaluationResult } from "./surrogate-fitness";

export interface ActiveLearningConvergence {
  totalDFTRuns: number;
  avgUncertaintyBefore: number;
  avgUncertaintyAfter: number;
  modelRetrains: number;
  bestTcFromLoop: number;
}

export interface ActiveLearningCycleRecord {
  cycle: number;
  timestamp: number;
  candidatesSelected: number;
  dftSuccesses: number;
  dftFailures: number;
  avgGnnUncertainty: number;
  avgXgbUncertainty: number;
  avgCombinedUncertainty: number;
  uncertaintyAfter: number;
  uncertaintyReductionPct: number;
  gnnRetrained: boolean;
  gnnVersion: number | null;
  tierBreakdown: { bestTc: number; highUncertainty: number; randomExploration: number };
  topFormula: string;
  topAcquisitionScore: number;
  bestTcThisCycle: number;
}

const cycleHistory: ActiveLearningCycleRecord[] = [];
const MAX_CYCLE_HISTORY = 100;

export function getActiveLearningCycleHistory(): ActiveLearningCycleRecord[] {
  return [...cycleHistory];
}

let convergenceStats: ActiveLearningConvergence = {
  totalDFTRuns: 0,
  avgUncertaintyBefore: 1.0,
  avgUncertaintyAfter: 1.0,
  modelRetrains: 0,
  bestTcFromLoop: 0,
};

let totalEnrichedSinceLastRetrain = 0;
let lastRetrainCycle = 0;
const RETRAIN_CYCLE_INTERVAL = 20;
const RETRAIN_DFT_THRESHOLD = 50;
const recentUncertaintyDrops: number[] = [];

export function getActiveLearningStats(): ActiveLearningConvergence {
  return { ...convergenceStats };
}

interface RankedCandidate {
  candidate: SuperconductorCandidate;
  acquisitionScore: number;
  normalizedTc: number;
  uncertainty: number;
  xgbUncertainty: number;
  selectionTier: "best-tc" | "high-uncertainty" | "random-exploration";
}

function computeAdaptiveAlpha(): number {
  const baseAlpha = 2.0;
  const decayRate = 0.3;
  const minAlpha = 0.5;
  return Math.max(minAlpha, baseAlpha - decayRate * convergenceStats.modelRetrains);
}

export function selectForDFT(
  candidates: SuperconductorCandidate[],
  budget: number = 20
): RankedCandidate[] {
  const alpha = computeAdaptiveAlpha();

  const bestTcSlots = Math.min(10, Math.ceil(budget * 0.5));
  const highUncertaintySlots = Math.min(5, Math.ceil(budget * 0.25));
  const randomSlots = Math.max(1, budget - bestTcSlots - highUncertaintySlots);

  const scored: {
    candidate: SuperconductorCandidate;
    normalizedTc: number;
    gnnUncertainty: number;
    xgbUncertainty: number;
    combinedUncertainty: number;
    acquisitionScore: number;
  }[] = [];

  for (const candidate of candidates) {
    const tc = candidate.predictedTc ?? 0;
    const normalizedTc = Math.min(1.0, Math.max(0, tc / 300));

    let gnnUncertainty = candidate.uncertaintyEstimate ?? 0.5;
    try {
      const gnnResult = gnnPredictWithUncertainty(candidate.formula);
      gnnUncertainty = Math.max(gnnUncertainty, gnnResult.uncertainty);
    } catch (e: any) { console.error("[ActiveLearning] GNN predict error:", e?.message?.slice(0, 200)); }

    let xgbUncertainty = 0.5;
    try {
      const features = extractFeatures(candidate.formula);
      const xgbResult = gbPredictWithUncertainty(features, candidate.formula);
      xgbUncertainty = xgbResult.normalizedUncertainty;
    } catch (e: any) { console.error("[ActiveLearning] XGB uncertainty error:", e?.message?.slice(0, 200)); }

    const combinedUncertainty = 0.5 * gnnUncertainty + 0.5 * xgbUncertainty;
    const acquisitionScore = normalizedTc + alpha * combinedUncertainty;

    scored.push({ candidate, normalizedTc, gnnUncertainty, xgbUncertainty, combinedUncertainty, acquisitionScore });
  }

  const selected: RankedCandidate[] = [];
  const seenFormulas = new Set<string>();

  const byTc = [...scored].sort((a, b) => b.normalizedTc - a.normalizedTc);
  for (const s of byTc) {
    if (selected.length >= bestTcSlots) break;
    if (seenFormulas.has(s.candidate.formula)) continue;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "best-tc",
    });
  }

  const byUncertainty = [...scored].sort((a, b) => b.combinedUncertainty - a.combinedUncertainty);
  for (const s of byUncertainty) {
    if (selected.length >= bestTcSlots + highUncertaintySlots) break;
    if (seenFormulas.has(s.candidate.formula)) continue;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "high-uncertainty",
    });
  }

  const remaining = scored.filter(s => !seenFormulas.has(s.candidate.formula));
  for (let i = remaining.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
  }
  for (const s of remaining) {
    if (selected.length >= budget) break;
    seenFormulas.add(s.candidate.formula);
    selected.push({
      candidate: s.candidate,
      acquisitionScore: s.acquisitionScore,
      normalizedTc: s.normalizedTc,
      uncertainty: s.combinedUncertainty,
      xgbUncertainty: s.xgbUncertainty,
      selectionTier: "random-exploration",
    });
  }

  return selected;
}

async function runDFTEnrichmentForCandidate(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<boolean> {
  try {
    const dftData = await resolveDFTFeatures(candidate.formula);

    const desc = describeDFTSources(dftData);
    const hasExternalData = dftData.sources.mp || dftData.sources.aflow;
    const sourceType = hasExternalData ? "external+analytical" : "analytical";

    emit("log", {
      phase: "active-learning",
      event: "DFT enrichment",
      detail: `${candidate.formula} -- DFT data (${sourceType}, coverage=${dftData.dftCoverage.toFixed(2)}): ${desc}`,
      dataSource: "Active Learning",
    });

    const features = extractFeatures(candidate.formula, undefined, undefined, undefined, dftData);
    const gb = gbPredict(features);
    const nnScore = candidate.neuralNetScore ?? candidate.quantumCoherence ?? 0.3;
    const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

    const hasExternalDFT = dftData.sources.mp || dftData.sources.aflow;
    let confidence: string;
    if (dftData.dftCoverage > 0.6 && hasExternalDFT) confidence = "high";
    else if (dftData.dftCoverage > 0.2 && hasExternalDFT) confidence = "medium";
    else confidence = "low";

    const updates: any = {
      xgboostScore: gb.score,
      ensembleScore: ensemble,
      dataConfidence: confidence,
    };

    if (dftData.formationEnergy.source !== "analytical") {
      updates.formationEnergy = dftData.formationEnergy.value;
    }
    if (dftData.bandGap.source !== "analytical") {
      updates.bandGap = dftData.bandGap.value;
    }

    await storage.updateSuperconductorCandidate(candidate.id, updates);

    const formEnergy = dftData.formationEnergy?.value ?? null;
    const isStable = formEnergy !== null ? formEnergy < 0.5 : true;
    const dftSource = hasExternalDFT ? "external" as const : "active-learning" as const;
    incorporateDFTResult(
      candidate.formula,
      gb.tcPredicted,
      formEnergy,
      isStable,
      dftSource
    );

    addDFTTrainingResult({
      formula: candidate.formula,
      tc: gb.tcPredicted,
      formationEnergy: formEnergy,
      bandGap: dftData.bandGap?.value ?? null,
      structure: undefined,
      prototype: undefined,
      source: hasExternalDFT ? "external" : "active-learning",
    });

    let gnnTcPredicted = 0;
    let gnnStablePredicted = true;
    let gnnFePredicted = 0;
    try {
      const gnnPred = gnnPredictWithUncertainty(candidate.formula);
      gnnTcPredicted = gnnPred.tc;
      gnnStablePredicted = gnnPred.phononStability;
      gnnFePredicted = gnnPred.formationEnergy;
    } catch {}

    const predictedTc = gnnTcPredicted > 0 ? gnnTcPredicted * 0.6 + gb.tcPredicted * 0.4 : gb.tcPredicted;
    recordEvaluationResult(
      candidate.formula,
      { tc: predictedTc, stable: gnnStablePredicted, formationEnergy: gnnFePredicted },
      { tc: gb.tcPredicted, stable: isStable, formationEnergy: formEnergy },
      hasExternalDFT ? "dft" : "xtb"
    );

    return true;
  } catch (err) {
    console.log(`[Active Learning] DFT enrichment failed for ${candidate.formula}: ${err instanceof Error ? err.message : String(err)}`);

    incorporateDFTResult(
      candidate.formula,
      0,
      null,
      false,
      "active-learning"
    );

    let failGnnTc = 0;
    try {
      const gnnPred = gnnPredictWithUncertainty(candidate.formula);
      failGnnTc = gnnPred.tc;
    } catch {}
    const failPredTc = failGnnTc > 0 ? failGnnTc : (candidate.predictedTc ?? 0);
    recordEvaluationResult(
      candidate.formula,
      { tc: failPredTc, stable: true, formationEnergy: 0 },
      { tc: 0, stable: false, formationEnergy: null },
      "xtb"
    );

    return false;
  }
}

async function retrainGNNWithEnrichedData(
  emit: EventEmitter
): Promise<{ r2Before: number; maeBefore: number; r2After: number; maeAfter: number }> {
  const validationBefore = validateModel();
  const r2Before = validationBefore.r2;
  const maeBefore = Math.sqrt(validationBefore.mse);

  const trainingData = SUPERCON_TRAINING_DATA
    .filter(e => e.isSuperconductor)
    .map(e => ({
      formula: e.formula,
      tc: e.tc,
      formationEnergy: undefined as number | undefined,
      structure: undefined,
      prototype: undefined as string | undefined,
    }));

  const seenFormulas = new Set(trainingData.map(t => t.formula));

  const dftDataset = getDFTTrainingDataset();
  let dftMergeCount = 0;
  for (const dftRecord of dftDataset) {
    if (seenFormulas.has(dftRecord.formula)) continue;
    if (dftRecord.tc <= 0) continue;
    seenFormulas.add(dftRecord.formula);
    trainingData.push({
      formula: dftRecord.formula,
      tc: dftRecord.tc,
      formationEnergy: dftRecord.formationEnergy ?? undefined,
      structure: dftRecord.structure,
      prototype: dftRecord.prototype,
    });
    dftMergeCount++;
  }

  try {
    const enrichedCandidates = await storage.getSuperconductorCandidates(100);
    for (const c of enrichedCandidates) {
      if (c.dataConfidence === "high" || c.dataConfidence === "medium") {
        if (seenFormulas.has(c.formula)) continue;

        const mlf = c.mlFeatures as Record<string, any> | null;
        const hasDFTBandGap = mlf?.bandGap != null && mlf.bandGap >= 0;
        const hasDFTFormationEnergy = c.decompositionEnergy != null;
        const hasDFTValidation = hasDFTBandGap || hasDFTFormationEnergy;
        if (!hasDFTValidation) continue;

        const dftFeatures = extractFeatures(c.formula, undefined, undefined, undefined, undefined);
        const gb = gbPredict(dftFeatures);
        const dftCorrectedTc = gb.tcPredicted;

        if (dftCorrectedTc > 0) {
          seenFormulas.add(c.formula);
          trainingData.push({
            formula: c.formula,
            tc: dftCorrectedTc,
            formationEnergy: c.decompositionEnergy ?? undefined,
            structure: undefined,
            prototype: undefined,
          });
        }
      }
    }
  } catch (e: any) { console.error("[ActiveLearning] enrichment error:", e?.message?.slice(0, 200)); }

  const superconCount = SUPERCON_TRAINING_DATA.filter(e => e.isSuperconductor).length;
  const enrichedCount = trainingData.length - superconCount;
  const dftDatasetForVersion = getDFTTrainingDataset();
  const dftCount = dftDatasetForVersion.length;

  const ensembleModels = await trainEnsembleAsync(trainingData);
  invalidateGNNModel();
  setCachedEnsemble(ensembleModels, trainingData);

  logGNNVersion("active-learning-retrain", trainingData.length, dftCount, enrichedCount);

  await incorporateFailureData();

  const xgbResult = await retrainXGBoostFromEvaluated();

  const validationAfter = validateModel();
  const r2After = validationAfter.r2;
  const maeAfter = Math.sqrt(validationAfter.mse);

  const uncertaintyDrop = convergenceStats.avgUncertaintyBefore - convergenceStats.avgUncertaintyAfter;
  recentUncertaintyDrops.push(uncertaintyDrop);
  if (recentUncertaintyDrops.length > 3) recentUncertaintyDrops.shift();

  const avgRecentDrop = recentUncertaintyDrops.length >= 3
    ? recentUncertaintyDrops.reduce((s, v) => s + v, 0) / recentUncertaintyDrops.length
    : 1.0;
  const converged = avgRecentDrop < 0.1 && recentUncertaintyDrops.length >= 3;

  const evalStats = getEvaluatedDatasetStats();

  emit("log", {
    phase: "active-learning",
    event: "GNN + XGBoost retrained",
    detail: `R² ${r2Before.toFixed(4)} → ${r2After.toFixed(4)}, MAE ${maeBefore.toFixed(2)} → ${maeAfter.toFixed(2)}, GNN samples: ${trainingData.length} (${dftMergeCount} from DFT dataset, total DFT pool: ${dftDataset.length}), XGBoost: ${xgbResult.datasetSize} samples (${xgbResult.newEntries} from eval dataset), evaluated pool: ${evalStats.totalEvaluated}${converged ? ' [CONVERGED]' : ''}`,
    dataSource: "Active Learning",
  });

  return { r2Before, maeBefore, r2After, maeAfter };
}

export async function runActiveLearningCycle(
  emit: EventEmitter,
  memory: { cycleCount: number }
): Promise<ActiveLearningConvergence> {
  emit("log", {
    phase: "active-learning",
    event: "Active learning cycle started",
    detail: `Cycle ${memory.cycleCount}: selecting uncertain candidates for DFT enrichment`,
    dataSource: "Active Learning",
  });

  const allCandidates = await storage.getSuperconductorCandidates(200);

  const eligibleCandidates = allCandidates.filter(c =>
    c.dataConfidence !== "high" &&
    (c.predictedTc ?? 0) > 5
  );

  if (eligibleCandidates.length === 0) {
    emit("log", {
      phase: "active-learning",
      event: "Active learning skipped",
      detail: "No eligible candidates for DFT enrichment",
      dataSource: "Active Learning",
    });
    return convergenceStats;
  }

  const selected = selectForDFT(eligibleCandidates, 20);

  const avgUncertaintyBefore = selected.length > 0
    ? selected.reduce((sum, r) => sum + r.uncertainty, 0) / selected.length
    : 0;

  const tierCounts = {
    bestTc: selected.filter(s => s.selectionTier === "best-tc").length,
    highUncertainty: selected.filter(s => s.selectionTier === "high-uncertainty").length,
    randomExploration: selected.filter(s => s.selectionTier === "random-exploration").length,
  };
  const avgXgbUnc = selected.length > 0
    ? selected.reduce((s, r) => s + r.xgbUncertainty, 0) / selected.length
    : 0;

  emit("log", {
    phase: "active-learning",
    event: "DFT candidates selected",
    detail: `Selected ${selected.length} candidates [${tierCounts.bestTc} best-tc, ${tierCounts.highUncertainty} high-uncertainty, ${tierCounts.randomExploration} random] (avg combined unc: ${avgUncertaintyBefore.toFixed(3)}, avg XGB unc: ${avgXgbUnc.toFixed(3)}, top: ${selected[0]?.candidate.formula ?? 'none'} acq=${selected[0]?.acquisitionScore.toFixed(3) ?? 0})`,
    dataSource: "Active Learning",
  });

  let dftSuccessCount = 0;
  let bestTcThisLoop = 0;

  for (const { candidate } of selected) {
    const enriched = await runDFTEnrichmentForCandidate(emit, candidate);
    if (enriched) {
      dftSuccessCount++;
      convergenceStats.totalDFTRuns++;
    }
    if ((candidate.predictedTc ?? 0) > bestTcThisLoop) {
      bestTcThisLoop = candidate.predictedTc ?? 0;
    }
  }

  totalEnrichedSinceLastRetrain += dftSuccessCount;

  let retrainResult = { r2Before: 0, maeBefore: 0, r2After: 0, maeAfter: 0 };
  const hasNewData = totalEnrichedSinceLastRetrain > 0;
  const isConverged = recentUncertaintyDrops.length >= 3 &&
    (recentUncertaintyDrops.reduce((s, v) => s + v, 0) / recentUncertaintyDrops.length) < 0.1;
  const retrainThreshold = isConverged ? RETRAIN_DFT_THRESHOLD : 25;
  const cyclesSinceLastRetrain = memory.cycleCount - lastRetrainCycle;
  const cycleTrigger = cyclesSinceLastRetrain >= RETRAIN_CYCLE_INTERVAL;
  const dataTrigger = hasNewData && totalEnrichedSinceLastRetrain >= retrainThreshold;
  const uncertaintyTrigger = hasNewData && !isConverged && avgUncertaintyBefore > 0.5;
  const coldStartTrigger = convergenceStats.modelRetrains === 0 && hasNewData;

  const shouldRetrain = dataTrigger || cycleTrigger || uncertaintyTrigger || coldStartTrigger;
  const retrainReason = coldStartTrigger ? "cold-start"
    : dataTrigger ? `data-volume (${totalEnrichedSinceLastRetrain}>=${retrainThreshold})`
    : cycleTrigger ? `cycle-interval (${cyclesSinceLastRetrain}>=${RETRAIN_CYCLE_INTERVAL})`
    : uncertaintyTrigger ? `high-uncertainty (${avgUncertaintyBefore.toFixed(3)}>0.5)`
    : "unknown";

  if (shouldRetrain) {
    emit("log", {
      phase: "active-learning",
      event: "GNN retrain triggered",
      detail: `Trigger: ${retrainReason}, enriched=${totalEnrichedSinceLastRetrain}, cycles since retrain=${cyclesSinceLastRetrain}${isConverged ? ' [model converged]' : ''}`,
      dataSource: "Active Learning",
    });
    retrainResult = await retrainGNNWithEnrichedData(emit);
    convergenceStats.modelRetrains++;
    totalEnrichedSinceLastRetrain = 0;
    lastRetrainCycle = memory.cycleCount;
  } else if (dftSuccessCount > 0) {
    emit("log", {
      phase: "active-learning",
      event: "GNN retrain deferred",
      detail: `${totalEnrichedSinceLastRetrain} enriched since last retrain (threshold: ${retrainThreshold}). Uncertainty: ${avgUncertaintyBefore.toFixed(3)}. Will retrain after more data.`,
      dataSource: "Active Learning",
    });
  } else {
    emit("log", {
      phase: "active-learning",
      event: "DFT enrichment warning",
      detail: `All ${selected.length} candidates failed DFT enrichment. Skipping retrain — no new data to incorporate.`,
      dataSource: "Active Learning",
    });
  }

  let avgUncertaintyAfter = avgUncertaintyBefore;
  if (selected.length > 0) {
    let totalUncertaintyAfter = 0;
    for (const { candidate } of selected) {
      try {
        const gnnResult = gnnPredictWithUncertainty(candidate.formula);
        totalUncertaintyAfter += gnnResult.uncertainty;
      } catch {
        totalUncertaintyAfter += 0.5;
      }
    }
    avgUncertaintyAfter = totalUncertaintyAfter / selected.length;
  }

  convergenceStats.avgUncertaintyBefore = avgUncertaintyBefore;
  convergenceStats.avgUncertaintyAfter = avgUncertaintyAfter;
  if (bestTcThisLoop > convergenceStats.bestTcFromLoop) {
    convergenceStats.bestTcFromLoop = bestTcThisLoop;
  }

  const avgGnnUnc = selected.length > 0
    ? selected.reduce((s, r) => s + (r.uncertainty - r.xgbUncertainty * 0.5) * 2, 0) / selected.length
    : 0;
  const uncReductionPct = avgUncertaintyBefore > 0
    ? (avgUncertaintyBefore - avgUncertaintyAfter) / avgUncertaintyBefore * 100
    : 0;

  const cycleRecord: ActiveLearningCycleRecord = {
    cycle: memory.cycleCount,
    timestamp: Date.now(),
    candidatesSelected: selected.length,
    dftSuccesses: dftSuccessCount,
    dftFailures: selected.length - dftSuccessCount,
    avgGnnUncertainty: Math.round(avgGnnUnc * 1000) / 1000,
    avgXgbUncertainty: Math.round(avgXgbUnc * 1000) / 1000,
    avgCombinedUncertainty: Math.round(avgUncertaintyBefore * 1000) / 1000,
    uncertaintyAfter: Math.round(avgUncertaintyAfter * 1000) / 1000,
    uncertaintyReductionPct: Math.round(uncReductionPct * 10) / 10,
    gnnRetrained: shouldRetrain,
    gnnVersion: shouldRetrain ? getGNNModelVersion() : null,
    tierBreakdown: tierCounts,
    topFormula: selected[0]?.candidate.formula ?? "",
    topAcquisitionScore: Math.round((selected[0]?.acquisitionScore ?? 0) * 1000) / 1000,
    bestTcThisCycle: Math.round(bestTcThisLoop * 10) / 10,
  };
  cycleHistory.push(cycleRecord);
  if (cycleHistory.length > MAX_CYCLE_HISTORY) cycleHistory.shift();

  for (const { candidate } of selected) {
    try {
      const existingCandidate = await storage.getSuperconductorByFormula(candidate.formula);
      if (existingCandidate) {
        const hullDist = (existingCandidate.mlFeatures as any)?.stabilityGate?.hullDistance ?? 0.05;
        const discoveryResult = computeDiscoveryScore({
          predictedTc: existingCandidate.predictedTc ?? 0,
          formula: existingCandidate.formula,
          hullDistance: hullDist,
          synthesisScore: existingCandidate.stabilityScore ?? 0.5,
          uncertaintyEstimate: (existingCandidate.mlFeatures as any)?.uncertaintyEstimate ?? 0.5,
        });
        await storage.updateSuperconductorCandidate(existingCandidate.id, {
          discoveryScore: discoveryResult.discoveryScore,
        });
      }
    } catch (e: any) { console.error("[ActiveLearning] discovery score error:", e?.message?.slice(0, 200)); }
  }

  const uncertaintyReduction = avgUncertaintyBefore > 0
    ? ((avgUncertaintyBefore - avgUncertaintyAfter) / avgUncertaintyBefore * 100).toFixed(1)
    : "0";

  emit("log", {
    phase: "active-learning",
    event: "Active learning cycle complete",
    detail: `DFT enriched: ${dftSuccessCount}/${selected.length}, uncertainty reduction: ${uncertaintyReduction}%, model retrains: ${convergenceStats.modelRetrains}, best Tc: ${convergenceStats.bestTcFromLoop.toFixed(1)}K`,
    dataSource: "Active Learning",
  });

  return convergenceStats;
}
