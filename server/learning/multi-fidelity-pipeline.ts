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

const FAMILY_AVG_FORMATION_ENERGY: Record<string, number> = {
  Cuprates: -1.2,
  Pnictides: -0.8,
  Chalcogenides: -0.6,
  Intermetallics: -0.5,
  Borides: -0.7,
  Carbides: -0.4,
  Nitrides: -0.9,
  Hydrides: 0.1,
  Oxides: -1.0,
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
  } catch {}
}

async function stage0_MLFilter(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null }> {
  const start = Date.now();

  const score = candidate.ensembleScore ?? 0;
  const xgb = candidate.xgboostScore ?? 0;

  const passed = score > 0.4 || xgb > 0.35;
  const reason = passed ? null : `Ensemble score ${score.toFixed(2)} below threshold 0.4`;

  await logComputationalResult(
    candidate.id, candidate.formula, 0, "ML_filter",
    { ensembleScore: score, xgboostScore: xgb, threshold: 0.4 },
    passed, reason, Date.now() - start, passed ? 0.7 : 0.9
  );

  return { passed, reason };
}

async function stage1_ElectronicStructure(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const electronic = computeElectronicStructure(candidate.formula, candidate.crystalStructure);
  const correlation = assessCorrelationStrength(candidate.formula);

  const isMetallic = electronic.metallicity > 0.35;
  const hasDOS = electronic.densityOfStatesAtFermi > 0.5;

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

  const phonon = computePhononSpectrum(candidate.formula, electronicData.electronic);
  const coupling = computeElectronPhononCoupling(electronicData.electronic, phonon, candidate.formula, candidate.pressureGpa ?? 0);

  const hasStablePhonons = !phonon.hasImaginaryModes;
  const hasCoupling = coupling.lambda > 0.35;

  const passed = hasCoupling && hasStablePhonons;
  let reason = null;

  if (!hasStablePhonons) {
    reason = `Dynamically unstable: imaginary phonon modes detected — structure may not be physically realizable`;
  } else if (!hasCoupling) {
    reason = `Weak e-ph coupling (lambda=${coupling.lambda.toFixed(3)}) - insufficient for significant Tc`;
  }

  await logComputationalResult(
    candidate.id, candidate.formula, 2, "phonon_coupling",
    { phonon, coupling, stablePhonons: hasStablePhonons },
    passed, reason, Date.now() - start, passed ? 0.6 : 0.8
  );

  return { passed, reason, data: { phonon, coupling } };
}

async function stage3_TcPrediction(
  emit: EventEmitter,
  candidate: SuperconductorCandidate,
  couplingData: any,
  electronicData: any
): Promise<{ passed: boolean; reason: string | null; data: any }> {
  const start = Date.now();

  const eliashberg = predictTcEliashberg(couplingData.coupling);
  const competingPhases = evaluateCompetingPhases(candidate.formula, electronicData.electronic);

  const metalScore = electronicData.electronic?.metallicity ?? 0.5;
  if (metalScore < 0.4) {
    eliashberg.predictedTc = eliashberg.predictedTc * Math.max(0.02, metalScore);
    eliashberg.confidenceBand = [0, Math.round(eliashberg.predictedTc * 2)];
  }

  const suppressingPhases = competingPhases.filter(p => p.suppressesSC);
  const hasSevereCompetition = suppressingPhases.length > 1 ||
    suppressingPhases.some(p => p.strength > 0.7);

  const tcAboveThreshold = eliashberg.predictedTc > 5;

  let dimensionality = candidate.dimensionality || "3D";
  const criticalFields = computeCriticalFields(
    eliashberg.predictedTc, couplingData.coupling, dimensionality
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
    data: { eliashberg, competingPhases, criticalFields, dimensionality },
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
  const isSynthesizable = structure.synthesizability > 0.4;
  const isStableOrMetastable = stability.isStable || stability.isMetastable;
  const ambientPressureStable = (candidate.pressureGpa ?? 0) <= 1 && (stability.isStable || stability.isMetastable);

  const formationEnergy = stability.formationEnergy ?? 0;
  const formationEnergyTooHigh = formationEnergy > 2.0;
  const formationEnergyDataError = formationEnergy < -5.0;

  const family = classifyFamily(candidate.formula);
  const familyAvgFormationEnergy = getFamilyAvgFormationEnergy(family);
  const hullDistance = Math.max(0, formationEnergy - familyAvgFormationEnergy);
  const isMetastableByHull = hullDistance > 0.5;

  const candidateHc2 = candidate.upperCriticalField ?? 0;
  const isRoomTempCandidate = (candidate.predictedTc ?? 0) >= 200;
  const hc2TooLow = isRoomTempCandidate && candidateHc2 < 5;

  let passed = isSynthesizable && isStableOrMetastable && !hc2TooLow && !formationEnergyTooHigh && !formationEnergyDataError;
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
  }

  const metastableScorePenalty = (passed && isMetastableByHull) ? 0.80 : 1.0;

  await logComputationalResult(
    candidate.id, candidate.formula, 4, "synthesis_feasibility",
    { structure, stability, ambientPressureStable, formationEnergy, hullDistance, isMetastableByHull, metastableScorePenalty },
    passed, reason, Date.now() - start, (passed ? 0.45 : 0.7) * metastableScorePenalty
  );

  return {
    passed,
    reason,
    data: { structure, stability, ambientPressureStable, formationEnergy, hullDistance, isMetastableByHull, metastableScorePenalty },
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
    detail: `Screening ${candidates.length} candidates through 5-stage pipeline: ML -> Electronic Structure -> Phonon/E-Ph -> Tc Prediction -> Synthesis`,
    dataSource: "Pipeline",
  });

  for (const candidate of candidates.slice(0, 4)) {
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
    await updateCandidatePhysics(candidate.id, physicsData, allPassed ? 4 : 3, allData);

    results.push({
      candidateId: candidate.id,
      formula: candidate.formula,
      finalStage: allPassed ? 4 : 3,
      passed: allPassed,
      failureReason: s4.reason,
      physicsData,
    });
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
      updates.ambientPressureStable = allData.stability?.isStable ?? false;
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
  } catch {}
}
