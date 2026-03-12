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
import { classifyFamily, parseFormulaCounts } from "./utils";
import { ELEMENTAL_DATA } from "./elemental-data";
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

const FAMILY_AVG_FORMATION_ENERGY_DEFAULTS: Record<string, number> = {
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

const familyFormationEnergyCache: Record<string, number> = { ...FAMILY_AVG_FORMATION_ENERGY_DEFAULTS };
let lastFormationEnergyUpdate = 0;
const FORMATION_ENERGY_UPDATE_INTERVAL_MS = 10 * 60 * 1000;

function updateFormationEnergyAverages(): void {
  try {
    const { getGroundTruthDataset } = require("./ground-truth-store");
    const datapoints = getGroundTruthDataset();
    if (!datapoints || datapoints.length < 10) return;

    const familySums: Record<string, { sum: number; count: number }> = {};
    for (const dp of datapoints) {
      if (dp.formation_energy == null || !Number.isFinite(dp.formation_energy)) continue;
      const family = classifyFamily(dp.formula);
      if (!familySums[family]) familySums[family] = { sum: 0, count: 0 };
      familySums[family].sum += dp.formation_energy;
      familySums[family].count++;
    }

    let updated = 0;
    for (const [family, { sum, count }] of Object.entries(familySums)) {
      if (count >= 3) {
        const gtAvg = sum / count;
        const defaultVal = FAMILY_AVG_FORMATION_ENERGY_DEFAULTS[family] ?? FAMILY_AVG_FORMATION_ENERGY_DEFAULTS.Other;
        familyFormationEnergyCache[family] = 0.6 * gtAvg + 0.4 * defaultVal;
        updated++;
      }
    }

    if (updated > 0) {
      console.log(`[Pipeline] Updated formation energy averages from ground truth: ${updated} families`);
    }
  } catch {}
}

function getFamilyAvgFormationEnergy(family: string): number {
  if (Date.now() - lastFormationEnergyUpdate > FORMATION_ENERGY_UPDATE_INTERVAL_MS) {
    lastFormationEnergyUpdate = Date.now();
    updateFormationEnergyAverages();
  }
  return familyFormationEnergyCache[family] ?? familyFormationEnergyCache.Other ?? -0.3;
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
  const payload = {
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
  };
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      await storage.insertComputationalResult(payload);
      return;
    } catch (logErr) {
      const msg = logErr instanceof Error ? logErr.message.slice(0, 100) : "unknown";
      if (attempt === 0 && /connection|timeout|ECONNRESET/i.test(msg)) {
        await new Promise(r => setTimeout(r, 200));
        continue;
      }
      console.log(`[Pipeline] logComputationalResult failed for ${formula} stage=${stage}: ${msg}`);
      if (!passed) {
        console.log(`[Pipeline] WARNING: Lost negative data for ${formula} (${computationType}) — active learning may miss this sample`);
      }
    }
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
  const reason = passed ? null : `ML filter: ensemble=${score} xgb=${xgb} below 0.55/0.3`;

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
  const countMap = parseFormulaCounts(formula);
  const formulaElements = Object.keys(countMap);

  const thermoResult = computeReactionFeasibility(formula, null, []);
  const thermodynamicFeasibility = Math.max(0, Math.min(1, thermoResult.overallFeasibility));

  const precursorSelections = findBestPrecursors(formulaElements, "solid-state");
  const precursorResult = computePrecursorAvailabilityScore(precursorSelections);
  const precursorAvailability = Math.max(0, Math.min(1, precursorResult.overallScore));

  const mlResult = predictSynthesisFeasibility(formula);
  const structuralSimilarity = Math.max(0, Math.min(1, mlResult.feasibility));

  const nElements = formulaElements.length;
  const totalAtoms = Object.values(countMap).reduce((a, b) => a + b, 0);

  let volatilityPenalty = 0;
  for (const el of formulaElements) {
    const elData = ELEMENTAL_DATA[el];
    if (elData?.meltingPoint != null) {
      if (elData.meltingPoint < 100) volatilityPenalty += 0.12;
      else if (elData.meltingPoint < 400) volatilityPenalty += 0.06;
    }
  }

  const complexityRaw = 1 - Math.min(1, (nElements - 1) * 0.15 + (totalAtoms - 2) * 0.03 + volatilityPenalty);
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

  const candidateFamily = classifyFamily(candidate.formula);
  const directElementalBlockedFamilies = new Set(["Oxides", "Hydrides", "Cuprates"]);
  const isHighComplexity = synthScore.reactionComplexity < 0.5;
  const blockDirectElemental = directElementalBlockedFamilies.has(candidateFamily) || isHighComplexity;

  const qualityRoutes = retroRoutes.routes.filter(
    (r: any) => r.overallScore >= 0.4 && (!blockDirectElemental || r.type !== "direct-elemental")
  );
  const hasQualityRoutes = qualityRoutes.length > 0;
  const scoreAboveThreshold = synthScore.score > 0.3;
  const mlAboveMinimum = mlPrediction.feasibility > 0.2;

  const passed = (scoreAboveThreshold && mlAboveMinimum) || hasQualityRoutes;
  const reason = passed ? null :
    `No plausible synthesis: score=${synthScore.score} (threshold 0.3), ML feasibility=${mlPrediction.feasibility} (threshold 0.2), quality retro routes=${qualityRoutes.length}`;

  const bestRouteScore = retroRoutes.bestRoute?.overallScore ?? 0;
  const mlConfidence = mlPrediction.confidence ?? 0.5;
  const derivedConfidence = passed
    ? 0.3 * mlConfidence + 0.3 * synthScore.score + 0.2 * bestRouteScore + 0.2 * Math.min(1, qualityRoutes.length / 3)
    : 0.5 * mlConfidence + 0.3 * (1 - synthScore.score) + 0.2 * (1 - bestRouteScore);
  const clampedConfidence = Math.max(0.1, Math.min(0.95, derivedConfidence));

  await logComputationalResult(
    candidate.id, candidate.formula, 0, "synthesis_prescreen",
    {
      synthesisScore: synthScore.score,
      thermodynamicFeasibility: synthScore.thermodynamicFeasibility,
      precursorAvailability: synthScore.precursorAvailability,
      structuralSimilarity: synthScore.structuralSimilarity,
      reactionComplexity: synthScore.reactionComplexity,
      mlFeasibility: mlPrediction.feasibility,
      mlConfidence,
      retroRoutes: retroRoutes.totalRoutes,
      qualityRoutes: qualityRoutes.length,
      bestRoute: retroRoutes.bestRoute?.equation ?? null,
      bestRouteScore,
    },
    passed, reason, Date.now() - start, clampedConfidence
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

  const family = classifyFamily(candidate.formula);
  const dopableFamilies = new Set(["Cuprates", "Iron-Based", "Nickelates", "Heavy-Fermion"]);
  const hasDopingPotential = dopableFamilies.has(family)
    || (candidate.mlFeatures as any)?.dopingIntent === true;
  const insulatorThreshold = hasDopingPotential ? 1.0 : 0.1;

  const isInsulating = bandPred.bandGap > insulatorThreshold;
  const hasMinimalDOS = bandPred.dosPredicted < 0.3;
  const hasSomePromise = bandPred.flatBandScore > 0.2
    || bandPred.nestingFromBands > 0.3
    || bandPred.vhsProximity > 0.3
    || bandPred.multiBandScore > 0.3
    || bandPred.bandTopologyClass !== "trivial";

  const passed = !isInsulating && !hasMinimalDOS;
  let reason: string | null = null;
  if (isInsulating) {
    reason = `Band surrogate predicts insulator (gap=${bandPred.bandGap.toFixed(2)}eV, threshold=${insulatorThreshold}eV for ${family}) — unlikely superconductor`;
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

  const HARD_METALLICITY_THRESHOLD = 0.3;
  const SOFT_METALLICITY_THRESHOLD = 0.45;
  const isClearlyMetallic = electronic.metallicity > SOFT_METALLICITY_THRESHOLD;
  const isMarginallyMetallic = electronic.metallicity > HARD_METALLICITY_THRESHOLD && electronic.metallicity <= SOFT_METALLICITY_THRESHOLD;
  const isNonMetallic = electronic.metallicity <= HARD_METALLICITY_THRESHOLD;
  const hasDOS = electronic.densityOfStatesAtFermi > 0.8;

  const metallicityUncertaintyMultiplier = isMarginallyMetallic
    ? 1.0 + (SOFT_METALLICITY_THRESHOLD - electronic.metallicity) / (SOFT_METALLICITY_THRESHOLD - HARD_METALLICITY_THRESHOLD)
    : 1.0;

  const passed = (isClearlyMetallic || isMarginallyMetallic) && hasDOS;
  let reason: string | null = null;
  if (isNonMetallic) {
    reason = `Non-metallic (metallicity=${electronic.metallicity}) — below hard threshold ${HARD_METALLICITY_THRESHOLD}`;
  } else if (!hasDOS) {
    reason = `Insufficient DOS at Fermi level (${electronic.densityOfStatesAtFermi})`;
  }

  await logComputationalResult(
    candidate.id, candidate.formula, 1, "electronic_structure",
    { ...electronic, correlation, marginalMetallicity: isMarginallyMetallic, metallicityUncertaintyMultiplier },
    passed, reason, Date.now() - start, passed ? (isMarginallyMetallic ? 0.50 : 0.65) : 0.85
  );

  return { passed, reason, data: { electronic, correlation, marginalMetallicity: isMarginallyMetallic, metallicityUncertaintyMultiplier } };
}

async function stage2_PhononCoupling(
  emit: EventEmitter,
  candidate: SuperconductorCandidate,
  electronicData: any
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const phononSurrogate = predictPhononProperties(candidate.formula, candidate.pressureGpa ?? 0);
  if (phononSurrogate.confidence > 0.4 && !phononSurrogate.phononStability && phononSurrogate.stabilityProbability < 0.3) {
    const reason = `Phonon surrogate pre-filter: dynamically unstable (stability=${phononSurrogate.stabilityProbability})`;
    await logComputationalResult(
      candidate.id, candidate.formula, 2, "phonon_coupling",
      { phononSurrogatePrescreen: { ...phononSurrogate }, earlyExit: true },
      false, reason, Date.now() - start, 0.65
    );
    return { passed: false, reason, data: { phonon: null, coupling: null, eliashbergPipeline: null, phononSurrogate } };
  }

  const mlLambda = predictLambda(candidate.formula, candidate.pressureGpa ?? 0);
  if (mlLambda.tier === "ml-regression" && mlLambda.confidence > 0.5 && mlLambda.lambda < 0.25) {
    const reason = `ML lambda pre-filter: lambda=${mlLambda.lambda} too low (threshold 0.25)`;
    await logComputationalResult(
      candidate.id, candidate.formula, 2, "phonon_coupling",
      { mlLambdaPrescreen: { lambda: mlLambda.lambda, confidence: mlLambda.confidence, tier: mlLambda.tier }, phononSurrogate, earlyExit: true },
      false, reason, Date.now() - start, 0.7
    );
    return { passed: false, reason, data: { phonon: null, coupling: null, eliashbergPipeline: null, mlLambda, phononSurrogate } };
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
    const effectiveLambda = eliashbergPipeline.lambda ?? couplingData.coupling?.lambda ?? 0;
    const isStrongCoupling = eliashbergPipeline.isStrongCoupling || effectiveLambda > 1.5;
    const pipelineWeight = isStrongCoupling ? 0.9 : 0.6;
    const mcMillanWeight = 1.0 - pipelineWeight;
    const blendedTc = pipelineWeight * pipelineTc + mcMillanWeight * eliashberg.predictedTc;
    eliashberg.predictedTc = Number(blendedTc.toFixed(1));
    eliashberg.gapRatio = eliashbergPipeline.gapRatio;
    eliashberg.strongCouplingCorrection = isStrongCoupling ? 1.5 : 1.0;
    eliashberg.confidenceBand = eliashbergPipeline.confidenceBand;
    eliashberg.tcBlendWeights = { pipeline: pipelineWeight, mcMillan: mcMillanWeight, isStrongCoupling, effectiveLambda };
  }

  const competingPhases = evaluateCompetingPhases(candidate.formula, electronicData.electronic);

  const metalScore = electronicData.electronic?.metallicity ?? 0.5;
  if (metalScore < 0.2) {
    eliashberg.predictedTc = 0;
    eliashberg.confidenceBand = [0, 0];
    eliashberg.metallicityClamped = true;
  } else if (metalScore < 0.4) {
    const scaleFactor = (metalScore - 0.2) / 0.2;
    eliashberg.predictedTc = eliashberg.predictedTc * scaleFactor;
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
    eliashberg.metallicityClamped = false;
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
  candidate: SuperconductorCandidate,
  cachedStructure?: any
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const structure = cachedStructure ?? await predictCrystalStructure(emit, candidate.formula);

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
    reason = `Formation energy ${formationEnergy} eV/atom below -5.0 — likely numerical garbage or data error`;
  } else if (formationEnergyTooHigh) {
    reason = `Formation energy ${formationEnergy} eV/atom exceeds +2.0 eV/atom — too unstable to synthesize`;
  } else if (!isSynthesizable) {
    reason = `Low synthesizability (${structure.synthesizability.toFixed(2)}) - ${structure.synthesisNotes}`;
  } else if (!isStableOrMetastable) {
    reason = `Thermodynamically unstable: ${stability.verdict}`;
  } else if (hc2TooLow) {
    reason = `Insufficient magnetic robustness: Hc2=${candidateHc2.toFixed(1)}T too low for room-temperature candidate (Tc=${candidate.predictedTc}K)`;
  } else if (enthalpyUnstable && enthalpyResult) {
    reason = `Enthalpy unstable at ${candidatePressure} GPa: H=${enthalpyResult.enthalpy.toFixed(3)} eV, dH=${enthalpyResult.enthalpyDifference.toFixed(3)} eV above decomposition`;
  }

  const isHighPressureMaterial = optP > 10;
  const metastableScorePenalty = (passed && isMetastableByHull && !isHighPressureMaterial) ? 0.80 : 1.0;

  await logComputationalResult(
    candidate.id, candidate.formula, 4, "synthesis_feasibility",
    { structure, stability, ambientPressureStable, formationEnergy, hullDistance, isMetastableByHull, isHighPressureMaterial, metastableScorePenalty, enthalpyResult },
    passed, reason, Date.now() - start, (passed ? 0.45 : 0.7) * metastableScorePenalty
  );

  return {
    passed,
    reason,
    data: { structure, stability, ambientPressureStable, formationEnergy, hullDistance, isMetastableByHull, isHighPressureMaterial, metastableScorePenalty, enthalpyResult },
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
  const batch = candidates.slice(0, 8);

  interface FastScreenResult {
    candidate: SuperconductorCandidate;
    physicsData: Record<string, any>;
    s1Data: any;
    earlyResult: PipelineResult | null;
  }

  const fastScreenPromises = batch.map(async (candidate): Promise<FastScreenResult> => {
    const physicsData: Record<string, any> = { _currentPredictedTc: candidate.predictedTc ?? 0 };

    const s0 = await stage0_MLFilter(emit, candidate);
    if (!s0.passed) {
      return { candidate, physicsData: {}, s1Data: null, earlyResult: {
        candidateId: candidate.id, formula: candidate.formula, finalStage: 0,
        passed: false, failureReason: s0.reason, physicsData: {},
      }};
    }

    const s0b = await stage0b_SynthesisPrescreen(emit, candidate);
    physicsData.synthesisPrescreen = s0b.data;
    if (!s0b.passed) {
      return { candidate, physicsData, s1Data: null, earlyResult: {
        candidateId: candidate.id, formula: candidate.formula, finalStage: 0,
        passed: false, failureReason: s0b.reason, physicsData,
      }};
    }

    const s0c = await stage0c_FastBandScreen(emit, candidate);
    physicsData.fastBandScreen = s0c.data;
    if (!s0c.passed) {
      return { candidate, physicsData, s1Data: null, earlyResult: {
        candidateId: candidate.id, formula: candidate.formula, finalStage: 0,
        passed: false, failureReason: s0c.reason, physicsData,
      }};
    }

    try {
      const tbSurr = predictTBProperties(candidate.formula);
      physicsData.tbSurrogate = tbSurr;
      if (tbSurr.lambdaProxy < 0.1 && tbSurr.dosAtEF < 0.01 && tbSurr.confidence > 0.4) {
        return { candidate, physicsData, s1Data: null, earlyResult: {
          candidateId: candidate.id, formula: candidate.formula, finalStage: 0,
          passed: false, failureReason: `TB surrogate filter: lambdaProxy=${tbSurr.lambdaProxy} (<0.1), dosAtEF=${tbSurr.dosAtEF} (<0.01)`,
          physicsData,
        }};
      }
    } catch (tbErr) {
      const tbMsg = tbErr instanceof Error ? tbErr.message.slice(0, 80) : "unknown";
      console.log(`[Pipeline] TB surrogate bypassed for ${candidate.formula}: ${tbMsg}`);
      physicsData.tbSurrogate = { bypassed: true, error: tbMsg };
    }

    const s1 = await stage1_ElectronicStructure(emit, candidate);
    physicsData.electronic = s1.data.electronic;
    physicsData.correlation = s1.data.correlation;
    if (!s1.passed) {
      await updateCandidatePhysics(candidate.id, physicsData, 1, s1.data, candidate);
      return { candidate, physicsData, s1Data: null, earlyResult: {
        candidateId: candidate.id, formula: candidate.formula, finalStage: 1,
        passed: false, failureReason: s1.reason, physicsData,
      }};
    }

    return { candidate, physicsData, s1Data: s1.data, earlyResult: null };
  });

  const fastScreenResults = await Promise.all(fastScreenPromises);

  const survivorsForHeavyPhysics: FastScreenResult[] = [];
  for (const fsr of fastScreenResults) {
    if (fsr.earlyResult) {
      results.push(fsr.earlyResult);
    } else {
      survivorsForHeavyPhysics.push(fsr);
    }
  }

  const HEAVY_CONCURRENCY = 3;
  for (let i = 0; i < survivorsForHeavyPhysics.length; i += HEAVY_CONCURRENCY) {
    const chunk = survivorsForHeavyPhysics.slice(i, i + HEAVY_CONCURRENCY);
    const heavyPromises = chunk.map(async ({ candidate, physicsData, s1Data }) => {
      const s2 = await stage2_PhononCoupling(emit, candidate, s1Data);
      physicsData.phonon = s2.data.phonon;
      physicsData.coupling = s2.data.coupling;
      if (!s2.passed) {
        await updateCandidatePhysics(candidate.id, physicsData, 2, { ...s1Data, ...s2.data }, candidate);
        return {
          candidateId: candidate.id, formula: candidate.formula, finalStage: 2,
          passed: false, failureReason: s2.reason, physicsData,
        } as PipelineResult;
      }

      const s3 = await stage3_TcPrediction(emit, candidate, s2.data, s1Data);
      physicsData.eliashberg = s3.data.eliashberg;
      physicsData.competingPhases = s3.data.competingPhases;
      physicsData.criticalFields = s3.data.criticalFields;
      if (!s3.passed) {
        await updateCandidatePhysics(candidate.id, physicsData, 3, { ...s1Data, ...s2.data, ...s3.data }, candidate);
        return {
          candidateId: candidate.id, formula: candidate.formula, finalStage: 3,
          passed: false, failureReason: s3.reason, physicsData,
        } as PipelineResult;
      }

      const cachedStructure = physicsData.synthesisPrescreen?.synthScore?.structure ?? null;
      const s4 = await stage4_SynthesisFeasibility(emit, candidate, cachedStructure || undefined);
      physicsData.structure = s4.data.structure;
      physicsData.stability = s4.data.stability;

      const allData = { ...s1Data, ...s2.data, ...s3.data, ...s4.data };
      await updateCandidatePhysics(candidate.id, physicsData, 4, allData, candidate);

      return {
        candidateId: candidate.id, formula: candidate.formula, finalStage: 4,
        passed: s4.passed, failureReason: s4.reason, physicsData,
      } as PipelineResult;
    });

    const chunkResults = await Promise.all(heavyPromises);
    results.push(...chunkResults);
  }

  for (const result of results) {
    if (!result.passed && result.failureReason) {
      try {
        let failureReason: "unstable_phonons" | "structure_collapse" | "high_formation_energy" | "non_metallic" | "scf_divergence" | "geometry_rejected" = "geometry_rejected";
        const fr = result.failureReason;
        if (/imaginary phonon|Dynamically unstable|phonon surrogate|phonon.*unstab/i.test(fr)) {
          failureReason = "unstable_phonons";
        } else if (/Non-metallic|insulator|metallicity.*below/i.test(fr)) {
          failureReason = "non_metallic";
        } else if (/Formation energy|thermodynamically unstable|too unstable to synthesize/i.test(fr)) {
          failureReason = "high_formation_energy";
        } else if (/Hc2.*inconsistent|structure.*collapse|Could not predict crystal/i.test(fr)) {
          failureReason = "structure_collapse";
        } else if (/SCF|convergence|diverge/i.test(fr)) {
          failureReason = "scf_divergence";
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
  allData: any,
  candidate?: SuperconductorCandidate
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
        const isNonMetallic = metalScore < 0.2;
        const isSemiMetallic = metalScore >= 0.2 && metalScore < 0.4;

        if (isNonMetallic) {
          eliashbergTc = 0;
        } else if (isSemiMetallic) {
          eliashbergTc = eliashbergTc * ((metalScore - 0.2) / 0.2);
        } else if (isMottInsulator) {
          eliashbergTc = eliashbergTc * 0.05;
        } else if (isStronglyCorrelated) {
          eliashbergTc = eliashbergTc * 0.3;
        }

        if (stage >= 3) {
          updates.predictedTc = Math.round(eliashbergTc);
        } else if (currentTc > 0) {
          if (eliashbergTc > currentTc && !isMottInsulator && !isStronglyCorrelated && !isNonMetallic) {
            const tcCap = lambda > 2.5 ? 30 : lambda > 2.0 ? 25 : lambda > 1.5 ? 20 : lambda > 1.0 ? 15 : 10;
            updates.predictedTc = Math.round(Math.min(eliashbergTc, currentTc + tcCap));
          } else {
            updates.predictedTc = Math.round(eliashbergTc);
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
      const effP = candidate?.optimalPressureGpa ?? candidate?.pressureGpa ?? 0;
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
