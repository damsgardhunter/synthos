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
} from "./physics-engine";
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
}

const CHALCOGENS = ["O","S","Se","Te"];
const PNICTOGENS = ["N","P","As","Sb","Bi"];

function parseFormula(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, (c) => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? [...new Set(matches)] : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
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
  const layeredStructure = (mat?.spacegroup?.includes("P4") || mat?.spacegroup?.includes("Pmmm") || mat?.spacegroup?.includes("I4")) ?? false;

  let cooperPairStrength = (hasTransitionMetal ? 0.3 : 0) + (hasHydrogen ? 0.25 : 0) +
    (dWaveSymmetry ? 0.2 : 0) + (layeredStructure ? 0.15 : 0) + (enSpread > 1.5 ? 0.1 : 0);

  const corrForCooper = physics?.correlationStrength ?? 0;
  if (corrForCooper >= 0.5 && corrForCooper <= 0.8) {
    if (dWaveSymmetry) cooperPairStrength += 0.15;
    else if (layeredStructure) cooperPairStrength += 0.1;
  }

  const electronic = computeElectronicStructure(formula, mat?.spacegroup);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula);
  const correlation = assessCorrelationStrength(formula);

  const useLambda = physics?.verifiedLambda ?? coupling.lambda;
  const useCorrelation = physics?.correlationStrength ?? correlation.ratio;

  const phononCouplingEstimate = Math.min(1.0, useLambda / 3.0);

  let electronDensityEstimate = electronic.metallicity > 0.5 ? 0.5 + electronic.densityOfStatesAtFermi * 0.05 :
    (mat?.bandGap === 0 || mat?.bandGap === null) ? 0.6 : 0.3;

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
  const valenceElectronConcentration = totalVE / totalAtoms;

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
        electronDensityEstimate = 0.3 * analyticalEd + 0.7 * Math.min(1.0, dftDos / 10.0);
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
  const phononSpectralWidth = Math.abs(phononMax - phononLog) / Math.max(phononMax, 1);

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
        const frac_i = (counts[elements[i]] || 1) / totalAtoms;
        const frac_j = (counts[elements[j]] || 1) / totalAtoms;
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
      const frac = (counts[el] || 1) / totalAtoms;
      const stonerProduct = stonerI * electronic.densityOfStatesAtFermi;
      spinFluctuationStrength = Math.max(spinFluctuationStrength, stonerProduct * frac);
    }
  }
  spinFluctuationStrength = Math.min(1.0, spinFluctuationStrength);

  const fermiSurfaceNestingScore = electronic.nestingScore ?? 0;

  const dosAtEF = electronic.densityOfStatesAtFermi;

  const muStarEstimate = coupling.muStar;

  return {
    avgElectronegativity: avgEN,
    maxAtomicMass: Math.max(...massValues, 0),
    numElements: elements.length,
    hasTransitionMetal,
    hasRareEarth,
    hasHydrogen,
    hasChalcogen,
    hasPnictogen,
    bandGap: mat?.bandGap ?? null,
    formationEnergy: mat?.formationEnergy ?? null,
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
  for (const mat of materials.slice(0, 100)) {
    const preFeatures = extractFeatures(mat.formula);
    if ((preFeatures.bandGap ?? 0) > 0.5 || (preFeatures.metallicity ?? 0.5) < 0.2) {
      preFilterSkipped++;
      continue;
    }

    const physics = physicsData.get(mat.formula);
    const crystal = crystalData.get(mat.formula);
    const features = extractFeatures(mat.formula, mat, physics, crystal);
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

    let gnnResult: GNNPrediction | null = null;
    try {
      const crystalStructure = crystal ? {
        latticeParams: undefined,
        atomicPositions: undefined,
      } : undefined;
      gnnResult = getGNNPrediction(mat.formula, crystalStructure);
    } catch {}

    scored.push({ mat, features, xgb, hasPhysics: !!physics, hasCrystal: !!crystal, gnn: gnnResult });
  }

  scored.sort((a, b) => b.xgb.score - a.xgb.score);
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
        xgboostReasoning: c.xgb.reasoning,
      };
    });

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an advanced neural network layer in a hybrid XGBoost+NN ensemble for superconductor prediction. The XGBoost layer has performed feature-based gradient boosting and ranked candidates. Physics computations (electronic structure, phonon spectrum, electron-phonon coupling lambda, Fermi surface topology, correlation strength U/W) are included. Some candidates may also have VERIFIED physics data from computational analysis (verifiedPhysics) and/or predicted crystal structures (crystalStructure) - weight these heavily as they represent higher-fidelity data.

Your role:
1. Evaluate using: electron-phonon coupling (lambda), BCS/Eliashberg theory, Cooper pair formation, Meissner effect, competing phases (magnetism, CDW, Mott)
2. For strongly correlated materials (U/W > 0.6), consider unconventional pairing (spin-fluctuation, d-wave, p-wave)
3. Assess room-temperature viability (Tc >= 293K AND ambient pressure AND zero resistance)
4. Identify pairing symmetry (s-wave, d-wave, p-wave, s+/-)
5. Estimate dimensionality effects on Tc
6. For each candidate, assign uncertainty and identify what physics is missing
7. When verifiedPhysics is present, use the verified lambda and Tc as ground truth over estimates
8. When crystalStructure is present, factor in thermodynamic stability (convexHullDistance < 0.05 = good), dimensionality effects on Tc, and synthesizability
9. If competing phases suppress SC, lower the score and flag the risk
10. CRITICAL: upperCriticalField_T (Hc2 in Tesla) is a key indicator. Real superconductors have Hc2 > 0. YBCO ~100-200T, MgB2 ~40T, hydrides ~100T+. If Hc2 is 0 or very low, the material likely does NOT superconduct. Penalize candidates with Hc2=0 severely. High Hc2 (>50T) strongly supports SC viability.

Return JSON with:
- 'candidates': array with 'formula', 'neuralNetScore' (0-1), 'refinedTc' (Kelvin), 'pressureGpa', 'meissnerEffect' (boolean), 'zeroResistance' (boolean), 'cooperPairMechanism' (string), 'pairingSymmetry' (s-wave/d-wave/p-wave/s+-), 'pairingMechanism' (phonon-mediated/spin-fluctuation/charge-fluctuation/unconventional), 'dimensionality' (3D/quasi-2D/2D/1D), 'quantumCoherence' (0-1), 'roomTempViable' (boolean), 'crystalStructure', 'uncertaintyEstimate' (0-1), 'reasoning' (under 150 chars)
- 'insights': array of 3-5 physics insights (each under 120 chars)`,
        },
        {
          role: "user",
          content: `XGBoost-ranked superconductor candidates for neural network refinement:\n${JSON.stringify(candidateSummaries, null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1200,
    });

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
      let ensembleScore: number;
      if (gnnPred && hasStructureData) {
        const gnnScore = Math.min(1, gnnPred.predictedTc > 100 ? 0.8 : gnnPred.predictedTc > 20 ? 0.5 : 0.2) * gnnPred.confidence;
        ensembleScore = Math.min(0.95, xgb.xgb.score * 0.25 + (nn.neuralNetScore ?? 0.5) * 0.35 + gnnScore * 0.40);
      } else if (gnnPred) {
        const gnnScore = Math.min(1, gnnPred.predictedTc > 100 ? 0.8 : gnnPred.predictedTc > 20 ? 0.5 : 0.2) * gnnPred.confidence;
        ensembleScore = Math.min(0.95, xgb.xgb.score * 0.30 + (nn.neuralNetScore ?? 0.5) * 0.50 + gnnScore * 0.20);
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
      const cappedTc = Math.min(rawTc, tcCap);

      candidates.push({
        name: xgb.mat.name,
        formula: xgb.mat.formula,
        predictedTc: cappedTc,
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
        notes: (cappedTc < rawTc ? `[LLM proposed Tc=${rawTc}K, capped to ${cappedTc}K by McMillan physics] ` : '') + (nn.reasoning ?? xgb.xgb.reasoning[0]),
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
    emit("log", {
      phase: "phase-7",
      event: "Neural network error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "ML Engine",
    });
  }

  return { candidates, insights };
}
