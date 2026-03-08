import { storage } from "../storage";
import type { SuperconductorCandidate } from "@shared/schema";
import type { EventEmitter } from "./engine";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { invalidateGNNModel, trainGNNSurrogate } from "./graph-neural-net";
import { resolveDFTFeatures, describeDFTSources } from "./dft-feature-resolver";
import { extractFeatures } from "./ml-predictor";
import { gbPredict, incorporateFailureData, validateModel } from "./gradient-boost";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { computeDiscoveryScore } from "./family-filters";

export interface ActiveLearningConvergence {
  totalDFTRuns: number;
  avgUncertaintyBefore: number;
  avgUncertaintyAfter: number;
  modelRetrains: number;
  bestTcFromLoop: number;
}

let convergenceStats: ActiveLearningConvergence = {
  totalDFTRuns: 0,
  avgUncertaintyBefore: 1.0,
  avgUncertaintyAfter: 1.0,
  modelRetrains: 0,
  bestTcFromLoop: 0,
};

let totalEnrichedSinceLastRetrain = 0;

export function getActiveLearningStats(): ActiveLearningConvergence {
  return { ...convergenceStats };
}

interface RankedCandidate {
  candidate: SuperconductorCandidate;
  acquisitionScore: number;
  normalizedTc: number;
  uncertainty: number;
}

export function selectForDFT(
  candidates: SuperconductorCandidate[],
  budget: number = 20
): RankedCandidate[] {
  const ranked: RankedCandidate[] = [];
  const highUncertainty: RankedCandidate[] = [];

  for (const candidate of candidates) {
    const tc = candidate.predictedTc ?? 0;
    const normalizedTc = Math.min(1.0, Math.max(0, tc / 300));

    let uncertainty = candidate.uncertaintyEstimate ?? 0.5;

    try {
      const gnnResult = gnnPredictWithUncertainty(candidate.formula);
      uncertainty = Math.max(uncertainty, gnnResult.uncertainty);
    } catch (e: any) { console.error("[ActiveLearning] GNN predict error:", e?.message?.slice(0, 200)); }

    const acquisitionScore = 0.5 * normalizedTc + 0.5 * uncertainty;

    const entry: RankedCandidate = {
      candidate,
      acquisitionScore,
      normalizedTc,
      uncertainty,
    };

    if (uncertainty > 0.3) {
      highUncertainty.push(entry);
    }
    ranked.push(entry);
  }

  ranked.sort((a, b) => b.acquisitionScore - a.acquisitionScore);
  highUncertainty.sort((a, b) => b.uncertainty - a.uncertainty);

  const selected: RankedCandidate[] = [];
  const seenFormulas = new Set<string>();

  for (const r of highUncertainty) {
    if (selected.length >= budget) break;
    if (seenFormulas.has(r.candidate.formula)) continue;
    seenFormulas.add(r.candidate.formula);
    selected.push(r);
  }

  for (const r of ranked) {
    if (selected.length >= budget) break;
    if (seenFormulas.has(r.candidate.formula)) continue;
    seenFormulas.add(r.candidate.formula);
    selected.push(r);
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
    return true;
  } catch (err) {
    console.log(`[Active Learning] DFT enrichment failed for ${candidate.formula}: ${err instanceof Error ? err.message : String(err)}`);
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

  try {
    const enrichedCandidates = await storage.getSuperconductorCandidates(100);
    for (const c of enrichedCandidates) {
      if (c.dataConfidence === "high" || c.dataConfidence === "medium") {
        const mlf = c.mlFeatures as Record<string, any> | null;
        const hasDFTBandGap = mlf?.bandGap != null && mlf.bandGap >= 0;
        const hasDFTFormationEnergy = c.decompositionEnergy != null;
        const hasDFTValidation = hasDFTBandGap || hasDFTFormationEnergy;
        if (!hasDFTValidation) continue;

        const existing = trainingData.find(t => t.formula === c.formula);
        if (existing) continue;

        const dftFeatures = extractFeatures(c.formula, undefined, undefined, undefined, undefined);
        const gb = gbPredict(dftFeatures);
        const dftCorrectedTc = gb.tcPredicted;

        if (dftCorrectedTc > 0) {
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

  invalidateGNNModel();
  trainGNNSurrogate(trainingData);

  await incorporateFailureData();

  const validationAfter = validateModel();
  const r2After = validationAfter.r2;
  const maeAfter = Math.sqrt(validationAfter.mse);

  emit("log", {
    phase: "active-learning",
    event: "GNN model retrained",
    detail: `R² ${r2Before.toFixed(4)} → ${r2After.toFixed(4)}, MAE ${maeBefore.toFixed(2)} → ${maeAfter.toFixed(2)}, training samples: ${trainingData.length}`,
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

  emit("log", {
    phase: "active-learning",
    event: "DFT candidates selected",
    detail: `Selected ${selected.length} candidates (avg uncertainty: ${avgUncertaintyBefore.toFixed(3)}, top: ${selected[0]?.candidate.formula ?? 'none'} acq=${selected[0]?.acquisitionScore.toFixed(3) ?? 0})`,
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
  const shouldRetrain = hasNewData && (
    totalEnrichedSinceLastRetrain >= 15 ||
    avgUncertaintyBefore > 0.3 ||
    convergenceStats.modelRetrains === 0
  );
  if (shouldRetrain) {
    emit("log", {
      phase: "active-learning",
      event: "GNN retrain triggered",
      detail: `Trigger: enriched=${totalEnrichedSinceLastRetrain}>=15 OR uncertainty=${avgUncertaintyBefore.toFixed(3)}>0.3 OR firstRetrain=${convergenceStats.modelRetrains === 0}`,
      dataSource: "Active Learning",
    });
    retrainResult = await retrainGNNWithEnrichedData(emit);
    convergenceStats.modelRetrains++;
    totalEnrichedSinceLastRetrain = 0;
  } else if (dftSuccessCount > 0) {
    emit("log", {
      phase: "active-learning",
      event: "GNN retrain deferred",
      detail: `${totalEnrichedSinceLastRetrain} enriched since last retrain (threshold: 15). Uncertainty: ${avgUncertaintyBefore.toFixed(3)}. Will retrain after more data.`,
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
