import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import type { SuperconductorCandidate } from "@shared/schema";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  predictTcEliashberg,
  evaluateCompetingPhases,
  computeCriticalFields,
  assessCorrelationStrength,
} from "./physics-engine";
import {
  predictCrystalStructure,
  evaluateConvexHullStability,
} from "./structure-predictor";
import { classifyFamily } from "./utils";
import { predictSynthesisFeasibility } from "../synthesis/ml-synthesis-predictor";
import { generateRetrosynthesisRoutes } from "../synthesis/retrosynthesis-engine";
import { computeReactionFeasibility } from "../synthesis/thermodynamic-feasibility";
import { computeEnthalpyStability } from "./enthalpy-stability";
import { findBestPrecursors, computePrecursorAvailabilityScore } from "../synthesis/precursor-database";
import { predictBandStructure } from "../physics/band-structure-surrogate";
import { runEliashbergPipeline } from "../physics/eliashberg-pipeline";
import { predictLambda } from "./lambda-regressor";
import { predictPhononProperties } from "../physics/phonon-surrogate";
import { predictTBProperties } from "../physics/tb-ml-surrogate";
import { recordStructureFailure } from "../crystal/structure-failure-db";

const FAMILY_AVG_FORMATION_ENERGY: Record<string, number> = {
  Cuprates: -1.2,
  "Heavy Fermions": -0.6,
  Pnictides: -0.8,
  Chalcogenides: -0.6,
  Sulfides: -0.5,
  Intermetallics: -0.5,
  Borides: -0.7,
  Carbides: -0.4,
  Nitrides: -0.9,
  Silicides: -0.5,
  Phosphides: -0.4,
  Hydrides: 0.1,
  Oxides: -1.0,
  Alloys: -0.3,
  Other: -0.3,
};

function getFamilyAvgFormationEnergy(family: string): number {
  return FAMILY_AVG_FORMATION_ENERGY[family] ?? FAMILY_AVG_FORMATION_ENERGY.Other;
}

export interface PipelineResult {
  candidateId: string;
  formula: string;
  finalStage: number;
  passed: boolean;
  failureReason: string | null;
  physicsData: Record<string, any>;
}

async function logComputationalResult(
  candidateId: string,
  formula: string,
  stage: number,
  computationType: string,
  results: any,
  passed: boolean,
  failureReason: string | null,
  computeTimeMs: number,
  confidence: number
) {
  const id = `cr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  try {
    await storage.insertComputationalResult({
      id,
      candidateId,
      formula,
      computationType,
      pipelineStage: stage,
      inputParams: { stage, candidateId },
      results,
      confidence,
      computeTimeMs,
      passed,
      failureReason,
    });
  } catch (logErr) {
    console.log(`[Pipeline] logComputationalResult failed for ${formula}: ${logErr instanceof Error ? logErr.message.slice(0, 80) : "unknown"}`);
  }
}

async function stage0_MLFilter(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null }> {
  const start = Date.now();

  const score = candidate.ensembleScore ?? 0;
  const xgb = candidate.xgboostScore ?? 0;

  const passed = score > 0.55 && xgb > 0.3;
  const reason = passed ? null : `Ensemble score ${score.toFixed(2)} or XGBoost ${xgb.toFixed(2)} below thresholds (0.55/0.3)`;

  await logComputationalResult(
    candidate.id, candidate.formula, 0, "ML_filter",
    { ensembleScore: score, xgboostScore: xgb, ensembleThreshold: 0.55, xgbThreshold: 0.3 },
    passed, reason, Date.now() - start, passed ? 0.7 : 0.9
  );

  return { passed, reason };
}

export function computeSynthesisScore(formula: string): {
  score: number;
  thermodynamicFeasibility: number;
  precursorAvailability: number;
  structuralSimilarity: number;
  reactionComplexity: number;
} {
  const thermoResult = computeReactionFeasibility(formula, null, []);
  const thermodynamicFeasibility = Math.max(0, Math.min(1, thermoResult.overallFeasibility));

  const formulaElements = Object.keys((() => {
    const c: Record<string, number> = {};
    const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
    let m;
    while ((m = regex.exec(formula)) !== null) {
      c[m[1]] = (c[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
    }
    return c;
  })());
  const precursorSelections = findBestPrecursors(formulaElements, "solid-state");
  const precursorResult = computePrecursorAvailabilityScore(precursorSelections);
  const precursorAvailability = Math.max(0, Math.min(1, precursorResult.overallScore));

  const mlResult = predictSynthesisFeasibility(formula);
  const structuralSimilarity = Math.max(0, Math.min(1, mlResult.feasibility));

  const countMap: Record<string, number> = {};
  const cRegex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let cm;
  while ((cm = cRegex.exec(formula)) !== null) {
    countMap[cm[1]] = (countMap[cm[1]] || 0) + (cm[2] ? parseFloat(cm[2]) : 1);
  }
  const nElements = Object.keys(countMap).length;
  const totalAtoms = Object.values(countMap).reduce((a, b) => a + b, 0);
  const complexityRaw = 1 - Math.min(1, (nElements - 1) * 0.15 + (totalAtoms - 2) * 0.03);
  const reactionComplexity = Math.max(0, Math.min(1, complexityRaw));

  const score =
    0.4 * thermodynamicFeasibility +
    0.3 * precursorAvailability +
    0.2 * structuralSimilarity +
    0.1 * reactionComplexity;

  return {
    score: Math.max(0, Math.min(1, score)),
    thermodynamicFeasibility,
    precursorAvailability,
    structuralSimilarity,
    reactionComplexity,
  };
}

async function stage0b_SynthesisPrescreen(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const synthScore = computeSynthesisScore(candidate.formula);
  const mlPrediction = predictSynthesisFeasibility(candidate.formula);
  const retroRoutes = generateRetrosynthesisRoutes(candidate.formula);

  const qualityRoutes = retroRoutes.routes.filter(
    (r: any) => r.overallScore >= 0.4 && r.type !== "direct-elemental"
  );
  const hasQualityRoutes = qualityRoutes.length > 0;
  const scoreAboveThreshold = synthScore.score > 0.3;
  const mlAboveMinimum = mlPrediction.feasibility > 0.2;

  const passed = (scoreAboveThreshold && mlAboveMinimum) || hasQualityRoutes;
  const reason = passed ? null :
    `No plausible synthesis: score=${synthScore.score.toFixed(2)} (threshold 0.3), ML feasibility=${mlPrediction.feasibility.toFixed(2)} (threshold 0.2), quality retro routes=${qualityRoutes.length}`;

  await logComputationalResult(
    candidate.id, candidate.formula, 0, "synthesis_prescreen",
    {
      synthesisScore: synthScore.score,
      thermodynamicFeasibility: synthScore.thermodynamicFeasibility,
      precursorAvailability: synthScore.precursorAvailability,
      structuralSimilarity: synthScore.structuralSimilarity,
      reactionComplexity: synthScore.reactionComplexity,
      mlFeasibility: mlPrediction.feasibility,
      retroRoutes: retroRoutes.totalRoutes,
      qualityRoutes: qualityRoutes.length,
      bestRoute: retroRoutes.bestRoute?.equation ?? null,
    },
    passed, reason, Date.now() - start, passed ? 0.6 : 0.85
  );

  return {
    passed,
    reason,
    data: { synthScore, mlPrediction, retroRoutes },
  };
}

async function stage0c_FastBandScreen(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const bandPred = predictBandStructure(candidate.formula, undefined);

  const isInsulating = bandPred.bandGap > 1.0;
  const hasMinimalDOS = bandPred.dosPredicted < 0.3;
  const hasSomePromise = bandPred.flatBandScore > 0.2
    || bandPred.nestingFromBands > 0.3
    || bandPred.vhsProximity > 0.3
    || bandPred.multiBandScore > 0.3
    || bandPred.bandTopologyClass !== "trivial";

  const passed = !isInsulating && !hasMinimalDOS;
  let reason: string | null = null;
  if (isInsulating) {
    reason = `Band surrogate predicts insulator (gap=${bandPred.bandGap.toFixed(2)}eV) — unlikely superconductor`;
  } else if (hasMinimalDOS) {
    reason = `Band surrogate predicts very low DOS at Fermi (${bandPred.dosPredicted.toFixed(2)}) — weak pairing potential`;
  }

  await logComputationalResult(
    candidate.id, candidate.formula, 0, "fast_band_screen",
    {
      bandGap: bandPred.bandGap,
      flatBandScore: bandPred.flatBandScore,
      nestingFromBands: bandPred.nestingFromBands,
      vhsProximity: bandPred.vhsProximity,
      dosPredicted: bandPred.dosPredicted,
      multiBandScore: bandPred.multiBandScore,
      bandTopologyClass: bandPred.bandTopologyClass,
      fsDimensionality: bandPred.fsDimensionality,
      confidence: bandPred.confidence,
      hasPromise: hasSomePromise,
    },
    passed, reason, Date.now() - start, bandPred.confidence
  );

  return { passed, reason, data: { bandPrediction: bandPred, hasSomePromise } };
}

async function stage1_ElectronicStructure(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const electronic = computeElectronicStructure(candidate.formula, candidate.crystalStructure);
  const correlation = assessCorrelationStrength(candidate.formula);

  const isMetallic = electronic.metallicity > 0.45;
  const hasDOS = electronic.densityOfStatesAtFermi > 0.8;

  const passed = isMetallic && hasDOS;
  const reason = passed ? null :
    !isMetallic ? `Non-metallic (metallicity=${electronic.metallicity.toFixed(2)}) - cannot superconduct` :
    `Insufficient DOS at Fermi level (${electronic.densityOfStatesAtFermi.toFixed(2)})`;

  await logComputationalResult(
    candidate.id, candidate.formula, 1, "electronic_structure",
    { ...electronic, correlation },
    passed, reason, Date.now() - start, passed ? 0.65 : 0.85
  );

  return { passed, reason, data: { electronic, correlation } };
}

async function stage2_PhononCoupling(
  emit: EventEmitter,
  candidate: SuperconductorCandidate,
  electronicData: any
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const phononSurrogate = predictPhononProperties(candidate.formula, candidate.pressureGpa ?? 0);
  if (phononSurrogate.confidence > 0.4 && !phononSurrogate.phononStability && phononSurrogate.stabilityProbability < 0.3) {
    await logComputationalResult(
      candidate.id, candidate.formula, 2, "phonon_coupling",
      { phononSurrogatePrescreen: { ...phononSurrogate } },
      false, `Phonon surrogate pre-filter: unstable (stability=${phononSurrogate.stabilityProbability.toFixed(3)})`, Date.now() - start, 0.65
    );
    const phonon = computePhononSpectrum(candidate.formula, electronicData.electronic);
    const coupling = computeElectronPhononCoupling(electronicData.electronic, phonon, candidate.formula, candidate.pressureGpa ?? 0);
    return { passed: false, reason: `Phonon surrogate pre-filter: dynamically unstable (stability=${phononSurrogate.stabilityProbability.toFixed(3)})`, data: { phonon, coupling, eliashbergPipeline: null, phononSurrogate } };
  }

  const mlLambda = predictLambda(candidate.formula, candidate.pressureGpa ?? 0);
  if (mlLambda.tier === "ml-regression" && mlLambda.confidence > 0.5 && mlLambda.lambda < 0.25) {
    await logComputationalResult(
      candidate.id, candidate.formula, 2, "phonon_coupling",
      { mlLambdaPrescreen: { lambda: mlLambda.lambda, confidence: mlLambda.confidence, tier: mlLambda.tier }, phononSurrogate },
      false, `ML lambda pre-filter: lambda=${mlLambda.lambda.toFixed(3)} too low (threshold 0.25)`, Date.now() - start, 0.7
    );
    const phonon = computePhononSpectrum(candidate.formula, electronicData.electronic);
    const coupling = computeElectronPhononCoupling(electronicData.electronic, phonon, candidate.formula, candidate.pressureGpa ?? 0);
    return { passed: false, reason: `ML lambda pre-filter: lambda=${mlLambda.lambda.toFixed(3)} too low`, data: { phonon, coupling, eliashbergPipeline: null, mlLambda, phononSurrogate } };
  }

  const phonon = computePhononSpectrum(candidate.formula, electronicData.electronic);
  const coupling = computeElectronPhononCoupling(electronicData.electronic, phonon, candidate.formula, candidate.pressureGpa ?? 0);

  let eliashbergData: any = null;
  try {
    eliashbergData = runEliashbergPipeline(
      candidate.formula,
      candidate.pressureGpa ?? 0,
      electronicData.electronic,
      phonon,
      coupling
    );
  } catch {}

  const effectiveLambda = eliashbergData?.lambda ?? coupling.lambda;

  const hasStablePhonons = !phonon.hasImaginaryModes;
  const hasCoupling = effectiveLambda > 0.5;

  const passed = hasCoupling && hasStablePhonons;
  let reason = null;

  if (!hasStablePhonons) {
    reason = `Dynamically unstable: imaginary phonon modes detected — structure may not be physically realizable`;
  } else if (!hasCoupling) {
    reason = `Weak e-ph coupling (lambda=${effectiveLambda.toFixed(3)}) - insufficient for significant Tc`;
  }

  await logComputationalResult(
    candidate.id, candidate.formula, 2, "phonon_coupling",
    { phonon, coupling, stablePhonons: hasStablePhonons, eliashbergPipeline: eliashbergData ? { lambda: eliashbergData.lambda, omegaLog: eliashbergData.omegaLog, tier: eliashbergData.tier } : null },
    passed, reason, Date.now() - start, passed ? 0.6 : 0.8
  );

  return { passed, reason, data: { phonon, coupling, eliashbergPipeline: eliashbergData } };
}

async function stage3_TcPrediction(
  emit: EventEmitter,
  candidate: SuperconductorCandidate,
  couplingData: any,
  electronicData: any
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const eliashbergPipeline = couplingData.eliashbergPipeline;
  const eliashberg = predictTcEliashberg(couplingData.coupling);

  if (eliashbergPipeline && eliashbergPipeline.tcBest > 0) {
    const pipelineTc = eliashbergPipeline.tcBest;
    const blendedTc = 0.6 * pipelineTc + 0.4 * eliashberg.predictedTc;
    eliashberg.predictedTc = Number(blendedTc.toFixed(1));
    eliashberg.gapRatio = eliashbergPipeline.gapRatio;
    eliashberg.strongCouplingCorrection = eliashbergPipeline.isStrongCoupling ? 1.5 : 1.0;
    eliashberg.confidenceBand = eliashbergPipeline.confidenceBand;
  }

  const competingPhases = evaluateCompetingPhases(candidate.formula, electronicData.electronic);

  const metalScore = electronicData.electronic?.metallicity ?? 0.5;
  if (metalScore < 0.4) {
    eliashberg.predictedTc = eliashberg.predictedTc * Math.max(0.02, metalScore);
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
  }

  const suppressingPhases = competingPhases.filter(p => p.suppressesSC);
  const hasSevereCompetition = suppressingPhases.length > 1 ||
    suppressingPhases.some(p => p.strength > 0.7);

  const tcAboveThreshold = eliashberg.predictedTc > 10;

  let dimensionality = candidate.dimensionality || "3D";
  const criticalFields = computeCriticalFields(
    eliashberg.predictedTc, couplingData.coupling, dimensionality, candidate.formula
  );

  const hc2Suspicious = eliashberg.predictedTc > 50 && criticalFields.upperCriticalField < 1;
  const passed = tcAboveThreshold && !hasSevereCompetition && !hc2Suspicious;

  let reason = null;
  if (!tcAboveThreshold) {
    reason = `Predicted Tc=${eliashberg.predictedTc}K too low for practical superconductivity`;
  } else if (hasSevereCompetition) {
    reason = `Suppressed by competing phases: ${suppressingPhases.map(p => p.phaseName).join(", ")}`;
  } else if (hc2Suspicious) {
    reason = `Hc2=${criticalFields.upperCriticalField.toFixed(1)}T inconsistent with Tc=${eliashberg.predictedTc.toFixed(0)}K — likely non-superconducting`;
  }

  await logComputationalResult(
    candidate.id, candidate.formula, 3, "tc_prediction",
    { eliashberg, competingPhases, criticalFields, dimensionality },
    passed, reason, Date.now() - start, passed ? 0.5 : 0.75
  );

  return {
    passed,
    reason,
    data: { eliashberg, competingPhases, criticalFields, dimensionality, eliashbergPipeline },
  };
}

async function stage4_SynthesisFeasibility(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const structure = await predictCrystalStructure(emit, candidate.formula);

  if (!structure) {
    await logComputationalResult(
      candidate.id, candidate.formula, 4, "synthesis_feasibility",
      { error: "Structure prediction failed" },
      false, "Could not predict crystal structure", Date.now() - start, 0.3
    );
    return { passed: false, reason: "Could not predict crystal structure", data: {} };
  }

  const stability = await evaluateConvexHullStability(structure.decompositionEnergy, candidate.formula);
  const isSynthesizable = structure.synthesizability > 0.55;
  const isStableOrMetastable = stability.isStable || stability.isMetastable;
  const optP = candidate.optimalPressureGpa ?? candidate.pressureGpa ?? 0;
  const ambientPressureStable = optP <= 1 && optP < 50 && (stability.isStable || stability.isMetastable);

  const formationEnergy = stability.formationEnergy ?? 0;
  const formationEnergyTooHigh = formationEnergy > 2.0;
  const formationEnergyDataError = formationEnergy < -5.0;

  const family = classifyFamily(candidate.formula);
  const familyAvgFormationEnergy = getFamilyAvgFormationEnergy(family);
  const hullDistance = Math.max(0, formationEnergy - familyAvgFormationEnergy);
  const isMetastableByHull = hullDistance > 0.5;

  let enthalpyResult: { isStable: boolean; isMetastable: boolean; enthalpyDifference: number; enthalpy: number } | null = null;
  const candidatePressure = candidate.pressureGpa ?? 0;
  if (candidatePressure > 5) {
    try {
      const hResult = computeEnthalpyStability(candidate.formula, candidatePressure);
      enthalpyResult = {
        isStable: hResult.isStable,
        isMetastable: hResult.isMetastable,
        enthalpyDifference: hResult.enthalpyDifference,
        enthalpy: hResult.enthalpy,
      };
    } catch {}
  }

  const enthalpyUnstable = enthalpyResult !== null && !enthalpyResult.isStable && !enthalpyResult.isMetastable;

  const candidateHc2 = candidate.upperCriticalField ?? 0;
  const isRoomTempCandidate = (candidate.predictedTc ?? 0) >= 200;
  const hc2TooLow = isRoomTempCandidate && candidateHc2 < 5;

  let passed = isSynthesizable && isStableOrMetastable && !hc2TooLow && !formationEnergyTooHigh && !formationEnergyDataError && !enthalpyUnstable;
  let reason = null;

  if (formationEnergyDataError) {
    reason = `Formation energy ${formationEnergy.toFixed(2)} eV/atom below -5.0 - likely data error`;
  } else if (formationEnergyTooHigh) {
    reason = `Formation energy ${formationEnergy.toFixed(2)} eV/atom exceeds 2.0 eV/atom - too unstable`;
  } else if (!isSynthesizable) {
    reason = `Low synthesizability (${structure.synthesizability.toFixed(2)}) - ${structure.synthesisNotes}`;
  } else if (!isStableOrMetastable) {
    reason = `Thermodynamically unstable: ${stability.verdict}`;
  } else if (hc2TooLow) {
    reason = `Insufficient magnetic robustness: Hc2=${candidateHc2.toFixed(1)}T too low for room-temperature candidate (Tc=${candidate.predictedTc}K)`;
  } else if (enthalpyUnstable && enthalpyResult) {
    reason = `Enthalpy unstable at ${candidatePressure} GPa: H=${enthalpyResult.enthalpy.toFixed(3)} eV, dH=${enthalpyResult.enthalpyDifference.toFixed(3)} eV above decomposition`;
  }

  const metastableScorePenalty = (passed && isMetastableByHull) ? 0.80 : 1.0;

  await logComputationalResult(
    candidate.id, candidate.formula, 4, "synthesis_feasibility",
    { structure, stability, ambientPressureStable, formationEnergy, hullDistance, isMetastableByHull, metastableScorePenalty, enthalpyResult },
    passed, reason, Date.now() - start, (passed ? 0.45 : 0.7) * metastableScorePenalty
  );

  return {
    passed,
    reason,
    data: { structure, stability, ambientPressureStable, formationEnergy, hullDistance, isMetastableByHull, metastableScorePenalty, enthalpyResult },
  };
}

export async function runMultiFidelityPipeline(
  emit: EventEmitter,
  candidates: SuperconductorCandidate[]
): Promise<PipelineResult[]> {
  const results: PipelineResult[] = [];

  emit("log", {
    phase: "phase-12",
    event: "Multi-fidelity pipeline started",
    detail: `Screening ${candidates.length} candidates through 7-stage pipeline: ML -> Synthesis Prescreen -> Fast Band Screen -> Electronic Structure -> Phonon/E-Ph -> Tc Prediction -> Synthesis`,
    dataSource: "Pipeline",
  });

  if (candidates.length > 8) {
    console.log(`[Pipeline] Truncating ${candidates.length} candidates to 8 for pipeline screening`);
  }
  for (const candidate of candidates.slice(0, 8)) {
    const physicsData: Record<string, any> = { _currentPredictedTc: candidate.predictedTc ?? 0 };

    const s0 = await stage0_MLFilter(emit, candidate);
    if (!s0.passed) {
      results.push({
        candidateId: candidate.id,
        formula: candidate.formula,
        finalStage: 0,
        passed: false,
        failureReason: s0.reason,
        physicsData: {},
      });
      continue;
    }

    const s0b = await stage0b_SynthesisPrescreen(emit, candidate);
    physicsData.synthesisPrescreen = s0b.data;
    if (!s0b.passed) {
      results.push({
        candidateId: candidate.id,
        formula: candidate.formula,
        finalStage: 0,
        passed: false,
        failureReason: s0b.reason,
        physicsData,
      });
      continue;
    }

    const s0c = await stage0c_FastBandScreen(emit, candidate);
    physicsData.fastBandScreen = s0c.data;
    if (!s0c.passed) {
      results.push({
        candidateId: candidate.id,
        formula: candidate.formula,
        finalStage: 0,
        passed: false,
        failureReason: s0c.reason,
        physicsData,
      });
      continue;
    }

    try {
      const tbSurr = predictTBProperties(candidate.formula);
      physicsData.tbSurrogate = tbSurr;
      if (tbSurr.lambdaProxy < 0.1 && tbSurr.dosAtEF < 0.01 && tbSurr.confidence > 0.4) {
        results.push({
          candidateId: candidate.id,
          formula: candidate.formula,
          finalStage: 0,
          passed: false,
          failureReason: `TB surrogate filter: lambdaProxy=${tbSurr.lambdaProxy.toFixed(3)} (<0.1), dosAtEF=${tbSurr.dosAtEF.toFixed(3)} (<0.01)`,
          physicsData,
        });
        continue;
      }
    } catch {}

    const s1 = await stage1_ElectronicStructure(emit, candidate);
    physicsData.electronic = s1.data.electronic;
    physicsData.correlation = s1.data.correlation;
    if (!s1.passed) {
      await updateCandidatePhysics(candidate.id, physicsData, 1, s1.data);
      results.push({
        candidateId: candidate.id,
        formula: candidate.formula,
        finalStage: 1,
        passed: false,
        failureReason: s1.reason,
        physicsData,
      });
      continue;
    }

    const s2 = await stage2_PhononCoupling(emit, candidate, s1.data);
    physicsData.phonon = s2.data.phonon;
    physicsData.coupling = s2.data.coupling;
    if (!s2.passed) {
      await updateCandidatePhysics(candidate.id, physicsData, 2, { ...s1.data, ...s2.data });
      results.push({
        candidateId: candidate.id,
        formula: candidate.formula,
        finalStage: 2,
        passed: false,
        failureReason: s2.reason,
        physicsData,
      });
      continue;
    }

    const s3 = await stage3_TcPrediction(emit, candidate, s2.data, s1.data);
    physicsData.eliashberg = s3.data.eliashberg;
    physicsData.competingPhases = s3.data.competingPhases;
    physicsData.criticalFields = s3.data.criticalFields;
    if (!s3.passed) {
      await updateCandidatePhysics(candidate.id, physicsData, 3, { ...s1.data, ...s2.data, ...s3.data });
      results.push({
        candidateId: candidate.id,
        formula: candidate.formula,
        finalStage: 3,
        passed: false,
        failureReason: s3.reason,
        physicsData,
      });
      continue;
    }

    const s4 = await stage4_SynthesisFeasibility(emit, candidate);
    physicsData.structure = s4.data.structure;
    physicsData.stability = s4.data.stability;

    const allPassed = s4.passed;
    const allData = { ...s1.data, ...s2.data, ...s3.data, ...s4.data };
    await updateCandidatePhysics(candidate.id, physicsData, 4, allData);

    results.push({
      candidateId: candidate.id,
      formula: candidate.formula,
      finalStage: 4,
      passed: allPassed,
      failureReason: s4.reason,
      physicsData,
    });
  }

  for (const result of results) {
    if (!result.passed && result.failureReason) {
      try {
        let failureReason: "unstable_phonons" | "structure_collapse" | "high_formation_energy" | "non_metallic" | "scf_divergence" | "geometry_rejected" = "geometry_rejected";
        if (result.failureReason.includes("imaginary phonon") || result.failureReason.includes("Dynamically unstable") || result.failureReason.includes("phonon")) {
          failureReason = "unstable_phonons";
        } else if (result.failureReason.includes("Non-metallic") || result.failureReason.includes("insulator") || result.failureReason.includes("non-metallic")) {
          failureReason = "non_metallic";
        } else if (result.failureReason.includes("Formation energy") || result.failureReason.includes("formation energy") || result.failureReason.includes("unstable")) {
          failureReason = "high_formation_energy";
        } else if (result.failureReason.includes("Hc2") || result.failureReason.includes("structure")) {
          failureReason = "structure_collapse";
        }
        recordStructureFailure({
          formula: result.formula,
          failureReason,
          failedAt: Date.now(),
          source: "pipeline",
          stage: result.finalStage,
          details: result.failureReason,
          bandGap: result.physicsData?.electronic?.bandGap,
          formationEnergy: result.physicsData?.stability?.formationEnergy,
          lowestPhononFreq: result.physicsData?.phonon?.lowestFrequency,
          imaginaryModeCount: result.physicsData?.phonon?.imaginaryModeCount,
        });
      } catch {}
    }
  }

  const passedCount = results.filter(r => r.passed).length;
  const stageCounts = [0, 1, 2, 3, 4].map(s => results.filter(r => r.finalStage === s).length);

  emit("log", {
    phase: "phase-12",
    event: "Pipeline screening complete",
    detail: `${results.length} screened: Stage0=${stageCounts[0]} filtered, Stage1=${stageCounts[1]}, Stage2=${stageCounts[2]}, Stage3=${stageCounts[3]}, Stage4=${stageCounts[4]} (${passedCount} passed all)`,
    dataSource: "Pipeline",
  });

  return results;
}

async function updateCandidatePhysics(
  candidateId: string,
  physicsData: Record<string, any>,
  stage: number,
  allData: any
) {
  try {
    const updates: any = {
      verificationStage: stage,
      dataConfidence: stage >= 3 ? "high" : stage >= 1 ? "medium" : "low",
    };

    if (allData.correlation) {
      updates.correlationStrength = allData.correlation.ratio;
    }
    if (allData.electronic) {
      updates.fermiSurfaceTopology = allData.electronic.fermiSurfaceTopology;
    }
    if (allData.coupling) {
      updates.electronPhononCoupling = allData.coupling.lambda;
      updates.logPhononFrequency = allData.coupling.omegaLog;
      updates.coulombPseudopotential = allData.coupling.muStar;
    }
    if (allData.eliashberg) {
      const tcRange = allData.eliashberg.confidenceBand;
      updates.uncertaintyEstimate = tcRange
        ? (tcRange[1] - tcRange[0]) / (allData.eliashberg.predictedTc || 1)
        : 0.5;

      let eliashbergTc = allData.eliashberg.predictedTc;
      if (Number.isFinite(eliashbergTc) && eliashbergTc > 0) {
        const currentTc = physicsData._currentPredictedTc ?? 0;
        const lambda = allData.coupling?.lambda ?? 0;

        const corrRatio = allData.correlation?.ratio ?? 0;
        const metalScore = allData.electronic?.metallicity ?? 0.5;
        const hasMott = allData.competingPhases?.some((p: any) => p.type === "Mott") ?? false;
        const isMottInsulator = (hasMott && corrRatio > 0.7) || corrRatio > 0.85;
        const isStronglyCorrelated = corrRatio > 0.7;
        const isNonMetallic = metalScore < 0.4;

        if (isNonMetallic) {
          eliashbergTc = eliashbergTc * Math.max(0.02, metalScore);
        } else if (isMottInsulator) {
          eliashbergTc = eliashbergTc * 0.05;
        } else if (isStronglyCorrelated) {
          eliashbergTc = eliashbergTc * 0.3;
        }

        if (currentTc > 0) {
          if (eliashbergTc > currentTc && !isMottInsulator && !isStronglyCorrelated && !isNonMetallic) {
            const tcCap = lambda > 2.5 ? 30 : lambda > 2.0 ? 25 : lambda > 1.5 ? 20 : lambda > 1.0 ? 15 : 10;
            updates.predictedTc = Math.round(Math.min(eliashbergTc, currentTc + tcCap));
          } else {
            const downBlend = eliashbergTc < currentTc * 0.3 ? 0.8 : eliashbergTc < currentTc * 0.5 ? 0.7 : (lambda > 1.5 ? 0.6 : lambda > 1.0 ? 0.5 : 0.4);
            updates.predictedTc = Math.round(currentTc * (1 - downBlend) + eliashbergTc * downBlend);
          }
        } else {
          updates.predictedTc = Math.round(eliashbergTc);
        }
      }
    }
    if (allData.competingPhases) {
      updates.competingPhases = allData.competingPhases;
    }
    if (allData.criticalFields) {
      updates.upperCriticalField = allData.criticalFields.upperCriticalField;
      updates.coherenceLength = allData.criticalFields.coherenceLength;
      updates.londonPenetrationDepth = allData.criticalFields.londonPenetrationDepth;
      updates.anisotropyRatio = allData.criticalFields.anisotropyRatio;
      updates.criticalCurrentDensity = allData.criticalFields.criticalCurrentDensity;
    }
    if (allData.dimensionality) {
      updates.dimensionality = allData.dimensionality;
    }
    if (allData.structure) {
      updates.decompositionEnergy = allData.structure.decompositionEnergy;
      const effP = (updates as any).optimalPressureGpa ?? (updates as any).pressureGpa ?? 0;
      updates.ambientPressureStable = effP < 50 && (allData.stability?.isStable ?? false);
    }
    if (allData.stability) {
      if (allData.stability.synthesizability != null) {
        (updates as any).synthesizability = allData.stability.synthesizability;
      }
      if (allData.stability.synthesisNotes) {
        (updates as any).synthesisNotes = allData.stability.synthesisNotes;
      }
    }
    if (allData.coupling) {
      const mechanism = allData.correlation?.ratio > 0.6
        ? "spin-fluctuation"
        : allData.coupling.lambda > 1.5
          ? "strong-coupling phonon-mediated"
          : "phonon-mediated BCS";
      updates.pairingMechanism = mechanism;
      if (!(updates as any).cooperPairMechanism) {
        (updates as any).cooperPairMechanism = mechanism === "spin-fluctuation"
          ? `Unconventional pairing via spin-fluctuation exchange (U/W=${(allData.correlation?.ratio ?? 0).toFixed(2)})`
          : `${mechanism} with lambda=${allData.coupling.lambda.toFixed(2)}`;
      }
    }

    await storage.updateSuperconductorCandidate(candidateId, updates);
  } catch (pipeErr) {
    console.log(`[Pipeline] updateCandidatePhysics failed for candidate ${candidateId}: ${pipeErr instanceof Error ? pipeErr.message.slice(0, 100) : "unknown"}`);
  }
}
