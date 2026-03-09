import OpenAI from "openai";
import type { Material, SuperconductorCandidate } from "@shared/schema";
import type { EventEmitter } from "./engine";
import { storage } from "../storage";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  assessCorrelationStrength,
  computeDimensionalityScore,
  detectStructuralMotifs,
  simulatePressureEffects,
} from "./physics-engine";
import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import {
  ELEMENTAL_DATA,
  getElementData,
  getCompositionWeightedProperty,
  getAverageMass,
  getLightestMass,
  isTransitionMetal,
  isRareEarth,
  isActinide,
  hasDOrFElectrons,
  getStonerParameter,
} from "./elemental-data";
import { gbPredict, getConfidenceBand } from "./gradient-boost";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";
import { getGNNPrediction, type GNNPrediction } from "./graph-neural-net";
export { getConfidenceBand };

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface MLFeatureVector {
  avgElectronegativity: number;
  maxAtomicMass: number;
  numElements: number;
  hasTransitionMetal: boolean;
  hasRareEarth: boolean;
  hasHydrogen: boolean;
  hasChalcogen: boolean;
  hasPnictogen: boolean;
  bandGap: number | null;
  formationEnergy: number | null;
  stability: number | null;
  crystalSymmetry: string | null;
  electronDensityEstimate: number;
  phononCouplingEstimate: number;
  dWaveSymmetry: boolean;
  layeredStructure: boolean;
  cooperPairStrength: number;
  meissnerPotential: number;
  correlationStrength: number;
  fermiSurfaceType: string;
  dimensionalityScore: number;
  anharmonicityFlag: boolean;
  electronPhononLambda: number;
  logPhononFreq: number;
  upperCriticalField: number | null;
  metallicity: number;
  avgAtomicRadius: number;
  pettiforNumber: number;
  valenceElectronConcentration: number;
  enSpread: number;
  hydrogenRatio: number;
  debyeTemperature: number;
  avgSommerfeldGamma: number;
  avgBulkModulus: number;
  dftConfidence: number;
  orbitalCharacterCode: number;
  phononSpectralCentroid: number;
  phononSpectralWidth: number;
  bondStiffnessVariance: number;
  chargeTransferMagnitude: number;
  connectivityIndex: number;
  nestingScore: number;
  vanHoveProximity: number;
  bandFlatness: number;
  softModeScore: number;
  motifScore: number;
  orbitalDFraction: number;
  mottProximityScore: number;
  topologicalBandScore: number;
  dimensionalityScoreV2: number;
  phononSofteningIndex: number;
  spinFluctuationStrength: number;
  fermiSurfaceNestingScore: number;
  dosAtEF: number;
  muStarEstimate: number;
  pressureGpa: number;
  optimalPressureGpa: number;
}

const CHALCOGENS = ["O","S","Se","Te"];
const PNICTOGENS = ["N","P","As","Sb","Bi"];

function parseFormula(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, (c) => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

export function extractFeatures(formula: string, mat?: Partial<Material>, physics?: PhysicsContext, crystal?: CrystalContext, dftData?: DFTResolvedFeatures): MLFeatureVector {
  const elements = parseFormula(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
  const massValues = elements.map(e => getElementData(e)?.atomicMass ?? 50);

  const hasTransitionMetal = elements.some(e => isTransitionMetal(e));
  const hasRareEarth = elements.some(e => isRareEarth(e));
  const hasHydrogen = elements.includes("H");
  const hasChalcogen = elements.some(e => CHALCOGENS.includes(e));
  const hasPnictogen = elements.some(e => PNICTOGENS.includes(e));

  const avgEN = getCompositionWeightedProperty(counts, "paulingElectronegativity") ?? 1.5;
  const enSpread = enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const dWaveSymmetry = hasCu && hasO && elements.length >= 3;
  const sg = mat?.spacegroup ?? "";
  const layeredStructure = sg.includes("P4") || sg.includes("Pmmm") || sg.includes("I4") || sg.includes("R-3m") || sg.includes("P63/mmc") || sg.includes("P6/mmm") || sg.includes("C2/m") || sg.includes("Cmcm");

  let cooperPairStrength = (hasTransitionMetal ? 0.3 : 0) + (hasHydrogen ? 0.25 : 0) +
    (dWaveSymmetry ? 0.2 : 0) + (layeredStructure ? 0.15 : 0) + (enSpread > 1.5 ? 0.1 : 0);

  const corrForCooper = physics?.correlationStrength ?? 0;
  if (corrForCooper >= 0.5 && corrForCooper <= 0.8) {
    if (dWaveSymmetry) cooperPairStrength += 0.15;
    else if (layeredStructure) cooperPairStrength += 0.1;
  }
  cooperPairStrength = Math.min(1, Math.max(0, cooperPairStrength));

  const electronic = computeElectronicStructure(formula, mat?.spacegroup);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula);
  const correlation = assessCorrelationStrength(formula);

  const useLambda = physics?.verifiedLambda ?? coupling.lambda;
  const useCorrelation = physics?.correlationStrength ?? correlation.ratio;

  const phononCouplingEstimate = Math.min(1.0, useLambda / 3.0);

  let electronDensityEstimate: number;
  if (electronic.metallicity > 0.5) {
    electronDensityEstimate = Math.min(1.0, 0.5 + electronic.densityOfStatesAtFermi * 0.05);
  } else if (mat?.bandGap != null && mat.bandGap > 0) {
    electronDensityEstimate = mat.bandGap > 3.0 ? 0.05 : Math.max(0, 0.3 - mat.bandGap * 0.08);
  } else {
    electronDensityEstimate = electronic.metallicity * 0.8;
  }

  const meissnerPotential = cooperPairStrength * 0.3 + phononCouplingEstimate * 0.35 +
    electronDensityEstimate * 0.2 + (layeredStructure ? 0.1 : 0) + (electronic.metallicity > 0.7 ? 0.05 : 0);

  const crystalDim = crystal?.dimensionality;
  const dimensionalityScore = crystalDim === "2D" ? 0.9 :
    crystalDim === "quasi-2D" ? 0.8 :
    layeredStructure ? 0.75 :
    electronic.fermiSurfaceTopology.includes("2D") ? 0.7 :
    electronic.fermiSurfaceTopology.includes("multi") ? 0.5 : 0.3;

  const useSpacegroup = crystal?.spaceGroup ?? mat?.spacegroup ?? null;
  const useHc2 = physics?.upperCriticalField ?? null;

  const avgAtomicRadius = getCompositionWeightedProperty(counts, "atomicRadius") ?? 130;
  const pettiforNumber = getCompositionWeightedProperty(counts, "pettiforScale") ?? 50;

  let totalVE = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
  }
  const valenceElectronConcentration = totalAtoms > 0 ? totalVE / totalAtoms : 0;

  const hCount = counts["H"] || 0;
  const metalAtomCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hydrogenRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  let debyeTemp = phonon.debyeTemperature;
  const avgGamma = getCompositionWeightedProperty(counts, "sommerfeldGamma") ?? 0;
  let avgBulk = getCompositionWeightedProperty(counts, "bulkModulus") ?? 0;

  let useMetallicity = electronic.metallicity;
  let dftConfidence = 0;

  if (dftData) {
    if (dftData.debyeTemp.source !== "analytical" && dftData.debyeTemp.value > 0) {
      debyeTemp = dftData.debyeTemp.value;
    }
    if (dftData.bulkModulus.source !== "analytical" && dftData.bulkModulus.value > 0) {
      avgBulk = dftData.bulkModulus.value;
    }
    if (dftData.isMetallic.source !== "analytical") {
      useMetallicity = dftData.isMetallic.value ? Math.max(useMetallicity, 0.85) : Math.min(useMetallicity, 0.15);
    }
    if (dftData.dosAtFermi.value != null && dftData.dosAtFermi.source !== "analytical") {
      const dftDos = dftData.dosAtFermi.value;
      if (dftDos > 0) {
        const analyticalEd = electronDensityEstimate;
        electronDensityEstimate = Math.min(1.0, 0.3 * analyticalEd + 0.7 * Math.min(1.0, dftDos / 10.0));
      }
    }
    dftConfidence = dftData.dftCoverage;
  }

  let orbitalCharacterCode = 0;
  const orbChar = electronic.orbitalCharacter.toLowerCase();
  if (orbChar.includes("f-electron") || orbChar.includes("f-")) orbitalCharacterCode = 3;
  else if (orbChar.includes("d-") || orbChar.includes("d band")) orbitalCharacterCode = 2;
  else if (orbChar.includes("1s") || orbChar.includes("sigma")) orbitalCharacterCode = 0.5;
  else if (orbChar.includes("p-")) orbitalCharacterCode = 1;

  const phononMax = phonon.maxPhononFrequency || 500;
  const phononLog = phonon.logAverageFrequency || 200;
  const phononSpectralCentroid = (phononMax + phononLog) / 2;
  const rawSpectralWidth = Math.abs(phononMax - phononLog) / Math.max(phononMax, 1);
  const phononSpectralWidth = Number.isFinite(rawSpectralWidth) ? rawSpectralWidth : 0;

  let bondStiffnessVariance = 0;
  const bulkValues: number[] = [];
  for (const el of elements) {
    const d = getElementData(el);
    if (d && d.bulkModulus && d.bulkModulus > 0) {
      bulkValues.push(d.bulkModulus);
    }
  }
  if (bulkValues.length > 1) {
    const mean = bulkValues.reduce((s, v) => s + v, 0) / bulkValues.length;
    bondStiffnessVariance = Math.sqrt(bulkValues.reduce((s, v) => s + (v - mean) ** 2, 0) / bulkValues.length) / Math.max(mean, 1);
  }

  let chargeTransferMagnitude = 0;
  if (elements.length > 1) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const en_i = getElementData(elements[i])?.paulingElectronegativity ?? 1.5;
        const en_j = getElementData(elements[j])?.paulingElectronegativity ?? 1.5;
        const frac_i = totalAtoms > 0 ? (counts[elements[i]] || 1) / totalAtoms : 0;
        const frac_j = totalAtoms > 0 ? (counts[elements[j]] || 1) / totalAtoms : 0;
        chargeTransferMagnitude += Math.abs(en_i - en_j) * Math.sqrt(frac_i * frac_j);
      }
    }
  }

  let connectivityIndex = 0.5;
  if (layeredStructure) connectivityIndex = 0.3;
  else if (dimensionalityScore > 0.7) connectivityIndex = 0.25;
  else if (elements.length === 1) connectivityIndex = 0.8;
  else if (elements.length >= 4) connectivityIndex = 0.6;
  if (hasHydrogen && hydrogenRatio >= 6) connectivityIndex = Math.max(connectivityIndex, 0.7);

  const phononSofteningIndex = phonon.softModePresent
    ? Math.min(1.0, (phonon.softModeScore ?? 0.5) + phonon.anharmonicityIndex * 0.3)
    : phonon.anharmonicityIndex * 0.5;

  let spinFluctuationStrength = 0;
  for (const el of elements) {
    const stonerI = getStonerParameter(el);
    if (stonerI !== null && stonerI > 0) {
      const frac = totalAtoms > 0 ? (counts[el] || 1) / totalAtoms : 0;
      const stonerProduct = stonerI * electronic.densityOfStatesAtFermi;
      spinFluctuationStrength = Math.max(spinFluctuationStrength, stonerProduct * frac);
    }
  }
  spinFluctuationStrength = Math.min(1.0, spinFluctuationStrength);

  const fermiSurfaceNestingScore = electronic.nestingScore ?? 0;

  const dosAtEF = electronic.densityOfStatesAtFermi;

  const muStarEstimate = coupling.muStar;

  let candidatePressure = (mat as any)?.pressureGpa ?? 0;

  if (candidatePressure === 0 && hasHydrogen && hydrogenRatio > 0.3) {
    if (hydrogenRatio >= 8) candidatePressure = 200;
    else if (hydrogenRatio >= 6) candidatePressure = 150;
    else if (hydrogenRatio >= 4) candidatePressure = 100;
  }

  let optimalPressureGpa = 0;
  try {
    const pressureResult = simulatePressureEffects(formula, electronic, phonon, coupling);
    optimalPressureGpa = pressureResult.optimalPressure;
  } catch {}

  return {
    avgElectronegativity: avgEN,
    maxAtomicMass: massValues.length > 0 ? Math.max(...massValues) : 0,
    numElements: elements.length,
    hasTransitionMetal,
    hasRareEarth,
    hasHydrogen,
    hasChalcogen,
    hasPnictogen,
    bandGap: mat?.bandGap ?? null,
    formationEnergy: mat?.formationEnergy ?? (() => { try { return computeMiedemaFormationEnergy(formula); } catch { return null; } })(),
    stability: mat?.stability ?? null,
    crystalSymmetry: useSpacegroup,
    electronDensityEstimate,
    phononCouplingEstimate,
    dWaveSymmetry,
    layeredStructure,
    cooperPairStrength,
    meissnerPotential,
    correlationStrength: useCorrelation,
    fermiSurfaceType: electronic.fermiSurfaceTopology,
    dimensionalityScore,
    anharmonicityFlag: phonon.anharmonicityIndex > 0.4,
    electronPhononLambda: useLambda,
    logPhononFreq: coupling.omegaLog,
    upperCriticalField: useHc2,
    metallicity: useMetallicity,
    avgAtomicRadius,
    pettiforNumber,
    valenceElectronConcentration,
    enSpread,
    hydrogenRatio,
    debyeTemperature: debyeTemp,
    avgSommerfeldGamma: avgGamma,
    avgBulkModulus: avgBulk,
    dftConfidence,
    orbitalCharacterCode,
    phononSpectralCentroid,
    phononSpectralWidth,
    bondStiffnessVariance,
    chargeTransferMagnitude,
    connectivityIndex,
    nestingScore: electronic.nestingScore ?? 0,
    vanHoveProximity: electronic.vanHoveProximity ?? 0,
    bandFlatness: electronic.bandFlatness ?? 0,
    softModeScore: phonon.softModeScore ?? (phonon.softModePresent ? 0.6 : 0.2),
    motifScore: detectStructuralMotifs(formula).motifScore,
    orbitalDFraction: electronic.orbitalFractions?.d ?? 0,
    mottProximityScore: electronic.mottProximityScore ?? 0,
    topologicalBandScore: electronic.topologicalBandScore ?? 0,
    dimensionalityScoreV2: computeDimensionalityScore(formula),
    phononSofteningIndex,
    spinFluctuationStrength,
    fermiSurfaceNestingScore,
    dosAtEF,
    muStarEstimate,
    pressureGpa: candidatePressure,
    optimalPressureGpa,
  };
}

function xgboostPredict(features: MLFeatureVector): { score: number; tcEstimate: number; reasoning: string[] } {
  const gb = gbPredict(features);

  const safeHc2 = features.upperCriticalField != null && Number.isFinite(features.upperCriticalField) ? features.upperCriticalField : null;
  if (safeHc2 != null) {
    if (safeHc2 > 50) {
      gb.score = Math.min(0.95, gb.score + 0.05);
      gb.reasoning.push(`High Hc2 (${safeHc2.toFixed(1)}T): robust Type-II superconductor`);
    } else if (safeHc2 === 0) {
      gb.score = Math.max(0.01, gb.score - 0.10);
      gb.reasoning.push("Hc2=0T: no upper critical field detected");
    }
  }

  return { score: gb.score, tcEstimate: Math.round(gb.tcPredicted), reasoning: gb.reasoning };
}

interface PhysicsContext {
  verifiedLambda: number | null;
  verifiedTc: number | null;
  competingPhases: any[];
  upperCriticalField: number | null;
  correlationStrength: number | null;
  verificationStage: number;
}

interface CrystalContext {
  spaceGroup: string;
  crystalSystem: string;
  dimensionality: string;
  isStable: boolean;
  convexHullDistance: number | null;
  synthesizability: number | null;
}

interface ResearchContext {
  synthesisCount: number;
  reactionCount: number;
  hasSynthesisKnowledge: boolean;
  hasReactionKnowledge: boolean;
  physicsData?: Map<string, PhysicsContext>;
  crystalData?: Map<string, CrystalContext>;
  strategyFocusAreas?: { area: string; priority: number }[];
  familyCounts?: Record<string, number>;
}

export async function runMLPrediction(
  emit: EventEmitter,
  materials: Material[],
  context?: ResearchContext
): Promise<{ candidates: Partial<SuperconductorCandidate>[]; insights: string[] }> {
  const candidates: Partial<SuperconductorCandidate>[] = [];
  const insights: string[] = [];

  const contextDetail = context
    ? ` (informed by ${context.synthesisCount} synthesis processes and ${context.reactionCount} chemical reactions)`
    : "";

  emit("log", {
    phase: "phase-7",
    event: "XGBoost+NN ensemble started",
    detail: `Feature extraction and gradient boosting on ${materials.length} materials${contextDetail}`,
    dataSource: "ML Engine",
  });

  let physicsData = context?.physicsData;
  let crystalData = context?.crystalData;
  if (!physicsData || !crystalData) {
    physicsData = physicsData ?? new Map();
    crystalData = crystalData ?? new Map();
    try {
      const existingSC = await storage.getSuperconductorCandidates(50);
      for (const sc of existingSC) {
        if (sc.electronPhononCoupling != null || sc.verificationStage != null && sc.verificationStage > 0) {
          physicsData.set(sc.formula, {
            verifiedLambda: sc.electronPhononCoupling,
            verifiedTc: sc.predictedTc,
            competingPhases: (sc.competingPhases as any[]) ?? [],
            upperCriticalField: sc.upperCriticalField,
            correlationStrength: sc.correlationStrength,
            verificationStage: sc.verificationStage ?? 0,
          });
        }
      }
      const structures = await storage.getCrystalStructures(100);
      for (const cs of structures) {
        crystalData.set(cs.formula, {
          spaceGroup: cs.spaceGroup,
          crystalSystem: cs.crystalSystem,
          dimensionality: cs.dimensionality,
          isStable: cs.isStable ?? false,
          convexHullDistance: cs.convexHullDistance,
          synthesizability: cs.synthesizability,
        });
      }
    } catch {}
  }

  const scored: { mat: Material; features: MLFeatureVector; xgb: ReturnType<typeof xgboostPredict>; hasPhysics: boolean; hasCrystal: boolean; gnn: GNNPrediction | null }[] = [];

  let preFilterSkipped = 0;
  let loopIdx = 0;
  for (const mat of materials.slice(0, 100)) {
    if (loopIdx > 0 && loopIdx % 10 === 0) {
      await new Promise<void>(resolve => setImmediate(resolve));
    }
    loopIdx++;

    const physics = physicsData.get(mat.formula);
    const crystal = crystalData.get(mat.formula);
    const features = extractFeatures(mat.formula, mat, physics, crystal);

    if ((features.bandGap ?? 0) > 0.5 || (features.metallicity ?? 0.5) < 0.2) {
      preFilterSkipped++;
      continue;
    }

    const xgb = xgboostPredict(features);

    if (physics && physics.verificationStage > 0) {
      xgb.score = Math.min(1, xgb.score + 0.05);
      xgb.reasoning.push("Physics-verified: computational analysis confirms candidate viability");
    }
    if (crystal?.isStable) {
      xgb.score = Math.min(1, xgb.score + 0.04);
      xgb.reasoning.push(`Crystal structure verified stable (${crystal.spaceGroup}, hull dist: ${crystal.convexHullDistance?.toFixed(3) ?? '?'})`);
    }
    if (physics?.competingPhases?.some((p: any) => p.suppressesSC)) {
      xgb.score = Math.max(0, xgb.score - 0.08);
      xgb.reasoning.push("WARNING: Competing phase identified that may suppress superconductivity");
    }
    if (crystal?.synthesizability != null && crystal.synthesizability < 0.3) {
      xgb.score = Math.max(0, xgb.score - 0.05);
      xgb.reasoning.push(`Low synthesizability (${(crystal.synthesizability * 100).toFixed(0)}%): practical challenges expected`);
    }

    xgb.score = Math.min(1, xgb.score);

    scored.push({ mat, features, xgb, hasPhysics: !!physics, hasCrystal: !!crystal, gnn: null as GNNPrediction | null });
  }

  scored.sort((a, b) => b.xgb.score - a.xgb.score);

  const GNN_INFERENCE_LIMIT = 15;
  for (let i = 0; i < Math.min(GNN_INFERENCE_LIMIT, scored.length); i++) {
    const entry = scored[i];
    try {
      const crystal = crystalData.get(entry.mat.formula);
      const crystalStructure = crystal ? {
        latticeParams: undefined,
        atomicPositions: undefined,
      } : undefined;
      entry.gnn = getGNNPrediction(entry.mat.formula, crystalStructure);
    } catch {}
  }

  scored.sort((a, b) => {
    const aGnnScore = a.gnn ? (Math.min(1, a.gnn.predictedTc > 100 ? 0.8 : a.gnn.predictedTc > 20 ? 0.5 : 0.2) * a.gnn.confidence) : 0;
    const bGnnScore = b.gnn ? (Math.min(1, b.gnn.predictedTc > 100 ? 0.8 : b.gnn.predictedTc > 20 ? 0.5 : 0.2) * b.gnn.confidence) : 0;
    const aEnsemble = aGnnScore * 0.6 + a.xgb.score * 0.3 + (a.hasCrystal ? 0.1 : 0) + (a.hasPhysics ? 0.05 : 0);
    const bEnsemble = bGnnScore * 0.6 + b.xgb.score * 0.3 + (b.hasCrystal ? 0.1 : 0) + (b.hasPhysics ? 0.05 : 0);
    return bEnsemble - aEnsemble;
  });
  const topCandidates = scored.slice(0, 5);

  if (topCandidates.length === 0) return { candidates, insights };

  const physicsEnriched = scored.filter(s => s.hasPhysics).length;
  const crystalEnriched = scored.filter(s => s.hasCrystal).length;
  const enrichmentDetail = (physicsEnriched > 0 || crystalEnriched > 0)
    ? ` [enriched: ${physicsEnriched} with physics, ${crystalEnriched} with crystal data]`
    : "";

  emit("log", {
    phase: "phase-7",
    event: "XGBoost screening complete",
    detail: `Top candidate: ${topCandidates[0].mat.formula} (score: ${(topCandidates[0].xgb.score*100).toFixed(0)}%, Tc raw: ${topCandidates[0].xgb.tcEstimate}K)${preFilterSkipped > 0 ? `, ${preFilterSkipped} non-metallic filtered` : ""}${enrichmentDetail}`,
    dataSource: "ML Engine",
  });

  const gnnPredicted = scored.filter(s => s.gnn !== null);
  if (gnnPredicted.length > 0) {
    const bestGnn = gnnPredicted.sort((a, b) => (b.gnn?.predictedTc ?? 0) - (a.gnn?.predictedTc ?? 0))[0];
    emit("log", {
      phase: "phase-7",
      event: "GNN prediction",
      detail: `${gnnPredicted.length} candidates evaluated, top: ${bestGnn.mat.formula} (formationEnergy: ${bestGnn.gnn?.formationEnergy}, Tc: ${bestGnn.gnn?.predictedTc}K, confidence: ${((bestGnn.gnn?.confidence ?? 0) * 100).toFixed(0)}%, phononStable: ${bestGnn.gnn?.phononStability})`,
      dataSource: "GNN Surrogate",
    });
  }

  try {
    const candidateSummaries = topCandidates.map(c => {
      const physics = physicsData!.get(c.mat.formula);
      const crystal = crystalData!.get(c.mat.formula);
      return {
        formula: c.mat.formula,
        name: c.mat.name,
        xgboostScore: c.xgb.score,
        tcEstimate: c.xgb.tcEstimate,
        features: {
          cooperPairStrength: c.features.cooperPairStrength,
          phononCoupling: c.features.phononCouplingEstimate,
          meissnerPotential: c.features.meissnerPotential,
          dWaveSymmetry: c.features.dWaveSymmetry,
          layeredStructure: c.features.layeredStructure,
        },
        physicsComputed: {
          electronPhononLambda: c.features.electronPhononLambda,
          logPhononFrequency: c.features.logPhononFreq,
          correlationStrength: c.features.correlationStrength,
          fermiSurfaceTopology: c.features.fermiSurfaceType,
          dimensionalityScore: c.features.dimensionalityScore,
          anharmonic: c.features.anharmonicityFlag,
          upperCriticalField_T: c.features.upperCriticalField,
        },
        ...(physics ? {
          verifiedPhysics: {
            verifiedLambda: physics.verifiedLambda,
            verifiedTc: physics.verifiedTc,
            upperCriticalField: physics.upperCriticalField,
            competingPhases: physics.competingPhases.length,
            hasSuppressionRisk: physics.competingPhases.some((p: any) => p.suppressesSC),
            verificationStage: physics.verificationStage,
          }
        } : {}),
        ...(crystal ? {
          crystalStructure: {
            spaceGroup: crystal.spaceGroup,
            crystalSystem: crystal.crystalSystem,
            dimensionality: crystal.dimensionality,
            isStable: crystal.isStable,
            convexHullDistance: crystal.convexHullDistance,
            synthesizability: crystal.synthesizability,
          }
        } : {}),
        xgboostReasoning: c.xgb.reasoning.slice(0, 2),
      };
    });

    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), 25000);

    let response;
    try {
      response = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: `You are an NN layer in an XGBoost+NN superconductor ensemble. Evaluate candidates using BCS/Eliashberg theory, electron-phonon coupling (lambda), competing phases, pairing symmetry, and dimensionality effects. Weight verifiedPhysics and crystalStructure data heavily. Penalize Hc2=0 severely; high Hc2 (>50T) strongly supports SC. Room-temp viable = Tc>=293K AND ambient pressure AND zero resistance. Return JSON: {'candidates': [{formula, neuralNetScore (0-1), refinedTc (K), pressureGpa, meissnerEffect (bool), zeroResistance (bool), cooperPairMechanism, pairingSymmetry, pairingMechanism, dimensionality, quantumCoherence (0-1), roomTempViable (bool), crystalStructure, uncertaintyEstimate (0-1), reasoning (<150 chars)}], 'insights': [3-5 strings <120 chars]}`,
          },
          {
            role: "user",
            content: `XGBoost-ranked candidates:\n${JSON.stringify(candidateSummaries)}`,
          },
        ],
        response_format: { type: "json_object" },
        max_completion_tokens: 900,
      }, { signal: abortController.signal });
    } finally {
      clearTimeout(timeoutId);
    }

    const content = response.choices[0]?.message?.content;
    if (!content) return { candidates, insights };

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      emit("log", { phase: "phase-7", event: "NN parse error", detail: content.slice(0, 200), dataSource: "ML Engine" });
      return { candidates, insights };
    }

    const nnCandidates = parsed.candidates ?? [];
    const nnInsights = parsed.insights ?? [];
    insights.push(...nnInsights);

    for (let i = 0; i < nnCandidates.length; i++) {
      const nn = nnCandidates[i];
      const xgb = topCandidates[i];
      if (!nn || !xgb) continue;

      const gnnPred = xgb.gnn;
      const hasStructureData = xgb.hasCrystal;
      const structuralNoveltyBonus = hasStructureData ? 0.1 : 0;
      let ensembleScore: number;
      if (gnnPred) {
        const gnnScore = Math.min(1, gnnPred.predictedTc > 100 ? 0.8 : gnnPred.predictedTc > 20 ? 0.5 : 0.2) * gnnPred.confidence;
        ensembleScore = Math.min(0.95, gnnScore * 0.6 + xgb.xgb.score * 0.3 + structuralNoveltyBonus);
      } else {
        ensembleScore = Math.min(0.95, xgb.xgb.score * 0.4 + (nn.neuralNetScore ?? 0.5) * 0.6);
      }

      let rawTc = nn.refinedTc ?? xgb.xgb.tcEstimate;
      const featureLambda = xgb.features.electronPhononLambda ?? 0;
      const featureMetal = xgb.features.metallicity ?? 0.5;
      const featureCorr = xgb.features.correlationStrength ?? 0;
      const omegaLogK = (xgb.features.logPhononFreq ?? 300) * 1.44;
      const muStar = 0.12;
      let mcMillanMax = 0;
      const denomMcM = featureLambda - muStar * (1 + 0.62 * featureLambda);
      if (featureLambda > 0.2 && Math.abs(denomMcM) > 1e-6) {
        const exponent = -1.04 * (1 + featureLambda) / denomMcM;
        mcMillanMax = (omegaLogK / 1.2) * Math.exp(exponent);
        if (!Number.isFinite(mcMillanMax) || mcMillanMax < 0) mcMillanMax = 0;
      }

      let tcCap: number;
      if (featureMetal < 0.3) {
        tcCap = Math.min(20, mcMillanMax * 0.1 || 10);
      } else if (featureMetal < 0.5) {
        tcCap = Math.min(80, mcMillanMax * 0.3 || 40);
      } else if (featureCorr > 0.85) {
        tcCap = Math.min(80, mcMillanMax * 0.3 || 30);
      } else if (featureCorr > 0.7) {
        tcCap = Math.min(200, mcMillanMax * 0.5 || 80);
      } else if (featureLambda < 0.3) {
        tcCap = Math.min(150, mcMillanMax > 0 ? mcMillanMax * 3.0 : 150);
      } else if (featureLambda < 0.5) {
        tcCap = Math.min(200, mcMillanMax > 0 ? mcMillanMax * 2.5 : 200);
      } else if (featureLambda < 1.0) {
        tcCap = Math.min(300, mcMillanMax > 0 ? mcMillanMax * 2.0 : 300);
      } else if (featureLambda < 1.5) {
        tcCap = mcMillanMax > 0 ? Math.min(400, mcMillanMax * 1.8) : 350;
      } else if (featureLambda < 2.5) {
        tcCap = mcMillanMax > 0 ? Math.min(450, mcMillanMax * 1.5) : 400;
      } else {
        tcCap = mcMillanMax > 0 ? Math.min(500, mcMillanMax * 1.3) : 450;
      }
      tcCap = Math.round(tcCap);
      let finalTc: number;
      if (mcMillanMax > 0 && rawTc > tcCap) {
        const physicsTc = mcMillanMax;
        finalTc = Math.round(0.6 * physicsTc + 0.3 * rawTc + 0.1 * tcCap);
        finalTc = Math.min(finalTc, tcCap);
      } else {
        finalTc = Math.min(rawTc, tcCap);
      }

      candidates.push({
        name: xgb.mat.name,
        formula: xgb.mat.formula,
        predictedTc: finalTc,
        pressureGpa: nn.pressureGpa ?? null,
        meissnerEffect: nn.meissnerEffect ?? false,
        zeroResistance: nn.zeroResistance ?? false,
        cooperPairMechanism: nn.cooperPairMechanism ?? "unknown",
        crystalStructure: nn.crystalStructure ?? xgb.mat.spacegroup,
        quantumCoherence: nn.quantumCoherence ?? 0,
        stabilityScore: xgb.features.cooperPairStrength,
        mlFeatures: xgb.features as any,
        xgboostScore: xgb.xgb.score,
        neuralNetScore: nn.neuralNetScore ?? 0.5,
        ensembleScore,
        roomTempViable: nn.roomTempViable ?? false,
        status: ensembleScore > 0.7 ? "promising" : "theoretical",
        notes: (finalTc < rawTc ? `[LLM proposed Tc=${rawTc}K, physics-weighted to ${finalTc}K (Allen-Dynes=${mcMillanMax}K)] ` : '') + (nn.reasoning ?? xgb.xgb.reasoning[0]),
        electronPhononCoupling: xgb.features.electronPhononLambda,
        logPhononFrequency: xgb.features.logPhononFreq,
        coulombPseudopotential: 0.12,
        pairingSymmetry: nn.pairingSymmetry ?? (xgb.features.dWaveSymmetry ? "d-wave" : "s-wave"),
        pairingMechanism: nn.pairingMechanism ?? (xgb.features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated"),
        correlationStrength: xgb.features.correlationStrength,
        dimensionality: nn.dimensionality ?? (xgb.features.layeredStructure ? "quasi-2D" : "3D"),
        fermiSurfaceTopology: xgb.features.fermiSurfaceType,
        uncertaintyEstimate: nn.uncertaintyEstimate ?? 0.5,
        verificationStage: 0,
        dataConfidence: xgb.hasPhysics ? "high" : (xgb.hasCrystal ? "medium" : "low"),
      });
    }

    emit("log", {
      phase: "phase-7",
      event: "Ensemble prediction complete",
      detail: `${candidates.length} candidates scored, ${candidates.filter(c => c.roomTempViable).length} room-temp viable${candidates.length > 0 ? `, top: ${candidates[0].formula} Tc=${candidates[0].predictedTc}K (capped)` : ''}`,
      dataSource: "ML Engine",
    });
  } catch (err: any) {
    const isTimeout = err.name === "AbortError" || err.message?.includes("aborted");
    emit("log", {
      phase: "phase-7",
      event: isTimeout ? "Neural network timeout" : "Neural network error",
      detail: isTimeout
        ? "Neural network timeout -- XGBoost-only results used. OpenAI did not respond within 25 seconds."
        : (err.message?.slice(0, 200) || "Unknown"),
      dataSource: "ML Engine",
    });

    if (candidates.length === 0 && topCandidates.length > 0) {
      for (const c of topCandidates) {
        const gnnPred = c.gnn;
        let ensembleScore: number;
        if (gnnPred) {
          const gnnScore = Math.min(1, gnnPred.predictedTc > 100 ? 0.8 : gnnPred.predictedTc > 20 ? 0.5 : 0.2) * gnnPred.confidence;
          ensembleScore = Math.min(0.95, gnnScore * 0.6 + c.xgb.score * 0.3 + (c.hasCrystal ? 0.1 : 0));
        } else {
          ensembleScore = Math.min(0.95, c.xgb.score * 0.7 + (c.hasCrystal ? 0.1 : 0));
        }

        const rawTc = c.xgb.tcEstimate;
        const featureLambda = c.features.electronPhononLambda ?? 0;
        const omegaLogK = (c.features.logPhononFreq ?? 300) * 1.44;
        const muStar = 0.12;
        let mcMillanMax = 0;
        const denomMcM = featureLambda - muStar * (1 + 0.62 * featureLambda);
        if (featureLambda > 0.2 && Math.abs(denomMcM) > 1e-6) {
          mcMillanMax = (omegaLogK / 1.2) * Math.exp(-1.04 * (1 + featureLambda) / denomMcM);
          if (mcMillanMax < 0 || !isFinite(mcMillanMax)) mcMillanMax = 0;
        }
        const finalTc = Math.max(0, Math.min(rawTc, mcMillanMax > 0 ? Math.max(rawTc * 0.7, mcMillanMax * 1.5) : rawTc));

        candidates.push({
          formula: c.mat.formula,
          name: c.mat.name,
          predictedTc: Math.round(finalTc * 10) / 10,
          pressureGpa: 0,
          meissnerEffect: ensembleScore > 0.5,
          zeroResistance: ensembleScore > 0.5,
          cooperPairMechanism: c.features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated",
          crystalStructure: c.features.layeredStructure ? "layered" : "3D",
          quantumCoherence: Math.min(1, ensembleScore * 0.8),
          stabilityScore: c.features.stabilityScore ?? 0.5,
          xgboostScore: c.xgb.score,
          neuralNetScore: 0,
          ensembleScore,
          roomTempViable: false,
          status: "theoretical",
          notes: `[XGBoost-only fallback: NN unavailable] ${c.xgb.reasoning[0] || ""}`,
          electronPhononCoupling: c.features.electronPhononLambda,
          logPhononFrequency: c.features.logPhononFreq,
          coulombPseudopotential: 0.12,
          pairingSymmetry: c.features.dWaveSymmetry ? "d-wave" : "s-wave",
          pairingMechanism: c.features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated",
          correlationStrength: c.features.correlationStrength,
          dimensionality: c.features.layeredStructure ? "quasi-2D" : "3D",
          fermiSurfaceTopology: c.features.fermiSurfaceType,
          uncertaintyEstimate: 0.7,
          verificationStage: 0,
          dataConfidence: c.hasPhysics ? "medium" : "low",
        });
      }

      emit("log", {
        phase: "phase-7",
        event: "XGBoost-only fallback applied",
        detail: `${candidates.length} candidates scored using XGBoost+GNN only (no NN refinement)`,
        dataSource: "ML Engine",
      });
    }
  }

  return { candidates, insights };
}

export interface PhysicsPrediction {
  lambda: number;
  dosAtEF: number;
  omegaLog: number;
  hullDistance: number;
  lambdaUncertainty: number;
  dosUncertainty: number;
  omegaUncertainty: number;
  hullUncertainty: number;
}

interface PhysicsTrainingSample {
  features: number[];
  lambda: number;
  dosAtEF: number;
  omegaLog: number;
  hullDistance: number;
}

interface SimpleTree {
  featureIndex: number;
  threshold: number;
  left: SimpleTree | number;
  right: SimpleTree | number;
}

interface PhysicsGBModel {
  trees: SimpleTree[];
  basePrediction: number;
  learningRate: number;
  trainedAt: number;
}

function buildSimpleTree(
  X: number[][], residuals: number[], indices: number[],
  depth: number, maxDepth: number, minSamples: number
): SimpleTree | number {
  if (depth >= maxDepth || indices.length < minSamples) {
    const sum = indices.reduce((s, i) => s + residuals[i], 0);
    return sum / indices.length;
  }
  const nFeatures = X[0].length;
  let bestFeature = -1, bestImprovement = -Infinity, bestThreshold = 0;
  let bestLeft: number[] = [], bestRight: number[] = [];
  for (let fi = 0; fi < nFeatures; fi++) {
    const pairs = indices.map(i => ({ idx: i, val: X[i][fi], res: residuals[i] })).sort((a, b) => a.val - b.val);
    const totalSum = pairs.reduce((s, p) => s + p.res, 0);
    let leftSum = 0;
    for (let i = 0; i < pairs.length - 1; i++) {
      leftSum += pairs[i].res;
      if (pairs[i].val === pairs[i + 1].val) continue;
      const lc = i + 1, rc = pairs.length - lc;
      const rightSum = totalSum - leftSum;
      const imp = (leftSum * leftSum) / lc + (rightSum * rightSum) / rc;
      if (imp > bestImprovement && lc >= 2 && rc >= 2) {
        bestImprovement = imp;
        bestFeature = fi;
        bestThreshold = (pairs[i].val + pairs[i + 1].val) / 2;
        bestLeft = pairs.slice(0, i + 1).map(p => p.idx);
        bestRight = pairs.slice(i + 1).map(p => p.idx);
      }
    }
  }
  if (bestFeature === -1) {
    return indices.reduce((s, i) => s + residuals[i], 0) / indices.length;
  }
  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildSimpleTree(X, residuals, bestLeft, depth + 1, maxDepth, minSamples),
    right: buildSimpleTree(X, residuals, bestRight, depth + 1, maxDepth, minSamples),
  };
}

function predictSimpleTree(tree: SimpleTree | number, x: number[]): number {
  if (typeof tree === "number") return tree;
  return x[tree.featureIndex] <= tree.threshold
    ? predictSimpleTree(tree.left, x)
    : predictSimpleTree(tree.right, x);
}

function trainPhysicsGB(X: number[][], y: number[], nTrees = 100, lr = 0.1, maxDepth = 3): PhysicsGBModel {
  const n = X.length;
  const allIdx = Array.from({ length: n }, (_, i) => i);
  const basePred = y.reduce((s, v) => s + v, 0) / n;
  const preds = new Array(n).fill(basePred);
  const trees: SimpleTree[] = [];
  for (let iter = 0; iter < nTrees; iter++) {
    const res = y.map((yi, i) => yi - preds[i]);
    const tree = buildSimpleTree(X, res, allIdx, 0, maxDepth, 3);
    if (typeof tree === "number") break;
    trees.push(tree);
    for (let i = 0; i < n; i++) preds[i] += lr * predictSimpleTree(tree, X[i]);
    const mse = y.reduce((s, yi, i) => s + (yi - preds[i]) ** 2, 0) / n;
    if (mse < 0.01) break;
  }
  return { trees, basePrediction: basePred, learningRate: lr, trainedAt: Date.now() };
}

function predictPhysicsGB(model: PhysicsGBModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const tree of model.trees) pred += model.learningRate * predictSimpleTree(tree, x);
  return pred;
}

function computeTreeEnsembleUncertainty(model: PhysicsGBModel, x: number[]): number {
  if (model.trees.length < 4) return 1.0;
  const blockSize = Math.max(1, Math.floor(model.trees.length / 4));
  const blockPreds: number[] = [];
  for (let b = 0; b < 4; b++) {
    let pred = model.basePrediction;
    const start = b * blockSize;
    const end = Math.min(start + blockSize, model.trees.length);
    for (let i = start; i < end; i++) pred += model.learningRate * predictSimpleTree(model.trees[i], x);
    blockPreds.push(pred);
  }
  const mean = blockPreds.reduce((s, v) => s + v, 0) / blockPreds.length;
  const variance = blockPreds.reduce((s, v) => s + (v - mean) ** 2, 0) / blockPreds.length;
  return Math.sqrt(variance);
}

function physicsFeatureVector(f: MLFeatureVector): number[] {
  return [
    f.electronPhononLambda,
    f.metallicity,
    f.logPhononFreq,
    f.debyeTemperature,
    f.correlationStrength,
    f.valenceElectronConcentration,
    f.avgElectronegativity,
    f.enSpread,
    f.hydrogenRatio,
    f.avgAtomicRadius,
    f.avgBulkModulus,
    f.maxAtomicMass,
    f.numElements,
    f.hasTransitionMetal ? 1 : 0,
    f.hasRareEarth ? 1 : 0,
    f.hasHydrogen ? 1 : 0,
    f.cooperPairStrength,
    f.dimensionalityScore,
    f.electronDensityEstimate,
    f.phononCouplingEstimate,
    f.nestingScore,
    f.dosAtEF,
    f.avgSommerfeldGamma,
    f.orbitalDFraction,
  ];
}

export class PhysicsPredictor {
  private lambdaModel: PhysicsGBModel | null = null;
  private dosModel: PhysicsGBModel | null = null;
  private omegaModel: PhysicsGBModel | null = null;
  private hullModel: PhysicsGBModel | null = null;
  private trainingSamples: PhysicsTrainingSample[] = [];
  private lastTrainedCycle = 0;
  private sampleCount = 0;

  addTrainingSample(features: MLFeatureVector, lambda: number, dosAtEF: number, omegaLog: number, hullDistance: number): void {
    const fv = physicsFeatureVector(features);
    if (fv.some(v => !Number.isFinite(v))) return;
    this.trainingSamples.push({ features: fv, lambda, dosAtEF, omegaLog, hullDistance });
    this.sampleCount++;
  }

  retrain(currentCycle: number): void {
    if (this.trainingSamples.length < 5) return;
    const X = this.trainingSamples.map(s => s.features);
    const yLambda = this.trainingSamples.map(s => s.lambda);
    const yDos = this.trainingSamples.map(s => s.dosAtEF);
    const yOmega = this.trainingSamples.map(s => s.omegaLog);
    const yHull = this.trainingSamples.map(s => s.hullDistance);
    const nTrees = this.trainingSamples.length < 100 ? 50 : 100;
    this.lambdaModel = trainPhysicsGB(X, yLambda, nTrees, 0.1, 3);
    this.dosModel = trainPhysicsGB(X, yDos, nTrees, 0.1, 3);
    this.omegaModel = trainPhysicsGB(X, yOmega, nTrees, 0.1, 3);
    this.hullModel = trainPhysicsGB(X, yHull, nTrees, 0.1, 3);
    this.lastTrainedCycle = currentCycle;
  }

  shouldRetrain(currentCycle: number): boolean {
    return (currentCycle - this.lastTrainedCycle) >= 100 || (this.lambdaModel === null && this.trainingSamples.length >= 5);
  }

  private transferPriorLambda(features: MLFeatureVector): { value: number; uncertainty: number } {
    const lambda = features.electronPhononLambda;
    const eta = features.avgSommerfeldGamma > 0
      ? 0.3 + features.avgSommerfeldGamma * 0.05
      : features.hasTransitionMetal ? 0.7 : 0.4;
    const massWeight = features.maxAtomicMass > 0 ? Math.sqrt(50 / features.maxAtomicMass) : 1.0;
    const priorLambda = Math.max(0.05, Math.min(3.5, lambda > 0.1 ? lambda : eta * massWeight));
    return { value: priorLambda, uncertainty: 0.5 };
  }

  private transferPriorDOS(features: MLFeatureVector): { value: number; uncertainty: number } {
    const dos = features.dosAtEF;
    if (dos > 0.1) return { value: dos, uncertainty: 0.4 };
    const gamma = features.avgSommerfeldGamma;
    if (gamma > 0) {
      const nEf = gamma / 2.359;
      return { value: Math.max(0.1, nEf), uncertainty: 0.35 };
    }
    const vec = features.valenceElectronConcentration;
    const estimatedDos = vec > 0 ? vec / 8.0 : 0.5;
    return { value: Math.max(0.1, estimatedDos), uncertainty: 0.6 };
  }

  private transferPriorOmega(features: MLFeatureVector): { value: number; uncertainty: number } {
    const debye = features.debyeTemperature;
    if (debye > 0) {
      const omegaLog = debye * 0.695 * 0.65;
      return { value: Math.max(50, omegaLog), uncertainty: 0.35 };
    }
    const bulkMod = features.avgBulkModulus;
    const mass = features.maxAtomicMass;
    if (bulkMod > 0 && mass > 0) {
      const est = Math.sqrt(bulkMod / mass) * 50;
      return { value: Math.max(50, est), uncertainty: 0.45 };
    }
    return { value: 200, uncertainty: 0.6 };
  }

  private transferPriorHull(features: MLFeatureVector): { value: number; uncertainty: number } {
    const formEnergy = features.formationEnergy;
    if (formEnergy !== null && formEnergy !== undefined) {
      const hull = formEnergy > 0 ? formEnergy * 0.5 : Math.abs(formEnergy) * 0.05;
      return { value: Math.max(0, Math.min(1.0, hull)), uncertainty: 0.3 };
    }
    const enSpread = features.enSpread;
    const nEl = features.numElements;
    let priorHull = 0.1 + nEl * 0.02;
    if (enSpread > 2.0) priorHull += 0.05;
    if (features.hasTransitionMetal && features.numElements <= 3) priorHull -= 0.03;
    return { value: Math.max(0, Math.min(1.0, priorHull)), uncertainty: 0.5 };
  }

  predict(features: MLFeatureVector): PhysicsPrediction {
    const fv = physicsFeatureVector(features);
    const useModel = this.lambdaModel && this.dosModel && this.omegaModel && this.hullModel
      && !fv.some(v => !Number.isFinite(v));

    if (useModel) {
      const lambda = Math.max(0, predictPhysicsGB(this.lambdaModel!, fv));
      const dos = Math.max(0, predictPhysicsGB(this.dosModel!, fv));
      const omega = Math.max(0, predictPhysicsGB(this.omegaModel!, fv));
      const hull = Math.max(0, predictPhysicsGB(this.hullModel!, fv));

      const lambdaU = computeTreeEnsembleUncertainty(this.lambdaModel!, fv);
      const dosU = computeTreeEnsembleUncertainty(this.dosModel!, fv);
      const omegaU = computeTreeEnsembleUncertainty(this.omegaModel!, fv);
      const hullU = computeTreeEnsembleUncertainty(this.hullModel!, fv);

      if (this.trainingSamples.length < 100) {
        const priorL = this.transferPriorLambda(features);
        const priorD = this.transferPriorDOS(features);
        const priorO = this.transferPriorOmega(features);
        const priorH = this.transferPriorHull(features);
        const w = Math.min(1.0, this.trainingSamples.length / 100);
        return {
          lambda: lambda * w + priorL.value * (1 - w),
          dosAtEF: dos * w + priorD.value * (1 - w),
          omegaLog: omega * w + priorO.value * (1 - w),
          hullDistance: hull * w + priorH.value * (1 - w),
          lambdaUncertainty: lambdaU * w + priorL.uncertainty * (1 - w),
          dosUncertainty: dosU * w + priorD.uncertainty * (1 - w),
          omegaUncertainty: omegaU * w + priorO.uncertainty * (1 - w),
          hullUncertainty: hullU * w + priorH.uncertainty * (1 - w),
        };
      }

      return {
        lambda, dosAtEF: dos, omegaLog: omega, hullDistance: hull,
        lambdaUncertainty: lambdaU, dosUncertainty: dosU,
        omegaUncertainty: omegaU, hullUncertainty: hullU,
      };
    }

    const priorL = this.transferPriorLambda(features);
    const priorD = this.transferPriorDOS(features);
    const priorO = this.transferPriorOmega(features);
    const priorH = this.transferPriorHull(features);
    return {
      lambda: priorL.value, dosAtEF: priorD.value,
      omegaLog: priorO.value, hullDistance: priorH.value,
      lambdaUncertainty: priorL.uncertainty, dosUncertainty: priorD.uncertainty,
      omegaUncertainty: priorO.uncertainty, hullUncertainty: priorH.uncertainty,
    };
  }

  preFilter(prediction: PhysicsPrediction): { pass: boolean; reason: string } {
    if (prediction.lambda < 0.15) return { pass: false, reason: `lambda=${prediction.lambda.toFixed(2)} < 0.15` };
    if (prediction.hullDistance > 0.3) return { pass: false, reason: `hull_dist=${prediction.hullDistance.toFixed(3)} > 0.3 eV/atom` };
    if (prediction.dosAtEF < 0.2) return { pass: false, reason: `DOS(EF)=${prediction.dosAtEF.toFixed(2)} < 0.2 states/eV` };
    return { pass: true, reason: "passed" };
  }

  getTrainingSize(): number { return this.trainingSamples.length; }
  getLastTrainedCycle(): number { return this.lastTrainedCycle; }
}

export const physicsPredictor = new PhysicsPredictor();
