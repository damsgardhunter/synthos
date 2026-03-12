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
  computePhysicsTcUQ,
  allenDynesTcRaw,
  classifyHydrogenBonding,
  type TcWithUncertainty,
} from "./physics-engine";
import { estimateFamilyPressure } from "./candidate-generator";
import { normalizeFormula, parseFormulaCounts as parseFormulaCountsCanonical } from "./utils";
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
import { gbPredict, getConfidenceBand, gbPredictWithUncertainty, type XGBUncertaintyResult } from "./gradient-boost";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";
import { getGNNPrediction, gnnPredictWithUncertainty, type GNNPrediction, type GNNPredictionWithUncertainty } from "./graph-neural-net";
import { getPhysicsFeatures, getDerivedFeatures } from "./physics-results-store";
import { predictLambda, recordLambdaValidation } from "./lambda-regressor";
import { predictTBProperties } from "../physics/tb-ml-surrogate";
import { getConformalInterval, getCalibrationState, getECE, type ConformalInterval } from "./conformal-calibrator";
import { computeOODScore } from "./ood-detector";
export { getConfidenceBand };

function yieldToEventLoop(): Promise<void> {
  return new Promise<void>(resolve => setImmediate(resolve));
}

export interface UnifiedCIResult {
  formula: string;
  tcMean: number;
  tcCI95: [number, number];
  tcTotalStd: number;
  tcEpistemicStd: number;
  tcAleatoricStd: number;
  lambdaMean: number;
  lambdaCI95: [number, number];
  gnn: {
    tcMean: number;
    tcCI95: [number, number];
    totalStd: number;
    epistemicUncertainty: number;
    aleatoricUncertainty: number;
    lambda: number;
    lambdaCI95: [number, number];
    confidence: number;
    weight: number;
  };
  xgb: {
    tcMean: number;
    tcCI95: [number, number];
    totalStd: number;
    epistemicStd: number;
    aleatoricStd: number;
    perModelPredictions: number[];
    weight: number;
  };
  modelCount: number;
  calibrationNote: string;
  calibratedCI95: [number, number];
  conformalQuantile: number;
  temperatureScale: number;
  ece: number;
  calibrationDatasetSize: number;
  conformalMethod: "conformal" | "fallback";
  oodScore: number;
  oodSigmaPenalty: number;
  isOOD: boolean;
  oodCategory: string;
  mahalanobisDistance: number;
  physicsUQ?: {
    tcMean: number;
    tcStd: number;
    tcCI95: [number, number];
    dominantUncertaintySource: string;
    lambdaContribution: number;
    omegaLogContribution: number;
    muStarContribution: number;
    mcSamples: number;
  };
}

export async function computeUnifiedCI(formula: string): Promise<UnifiedCIResult> {
  const [gnnResult, { features, xgbResult }] = await Promise.all([
    Promise.resolve(gnnPredictWithUncertainty(formula)),
    (async () => {
      const features = await extractFeatures(formula);
      const xgbResult = await gbPredictWithUncertainty(features, formula);
      return { features, xgbResult };
    })(),
  ]);

  const safe = (v: number, fallback = 0) => Number.isFinite(v) ? v : fallback;

  const gnnTcValid = Number.isFinite(gnnResult.tc) && Number.isFinite(gnnResult.totalStd) && gnnResult.totalStd > 0;
  const xgbTcValid = Number.isFinite(xgbResult.tcMean) && Number.isFinite(xgbResult.totalStd) && xgbResult.totalStd > 0;

  const gnnVar = gnnTcValid ? gnnResult.totalStd ** 2 : 1e6;
  const xgbVar = xgbTcValid ? xgbResult.totalStd ** 2 : 1e6;

  const wGnn = 1 / gnnVar;
  const wXgb = 1 / xgbVar;
  const wTotal = wGnn + wXgb;

  let tcCombined: number;
  if (gnnTcValid && xgbTcValid) {
    tcCombined = (wGnn * gnnResult.tc + wXgb * xgbResult.tcMean) / wTotal;
  } else if (xgbTcValid) {
    tcCombined = xgbResult.tcMean;
  } else if (gnnTcValid) {
    tcCombined = gnnResult.tc;
  } else {
    tcCombined = 0;
  }

  const varCombinedRaw = 1 / wTotal;
  const stdCombinedRaw = Math.sqrt(varCombinedRaw);

  const epistemicGnn = gnnTcValid ? safe(gnnResult.epistemicUncertainty) * safe(gnnResult.totalStd) : 0;
  const epistemicXgb = safe(xgbResult.epistemicStd);
  const epistemicCombined = Math.sqrt(
    ((wGnn * epistemicGnn) ** 2 + (wXgb * epistemicXgb) ** 2) / (wTotal ** 2)
  );

  const aleatoricGnn = gnnTcValid ? safe(gnnResult.aleatoricUncertainty) * safe(gnnResult.totalStd) : 0;
  const aleatoricXgb = safe(xgbResult.aleatoricStd);
  const aleatoricCombined = Math.sqrt(
    ((wGnn * aleatoricGnn) ** 2 + (wXgb * aleatoricXgb) ** 2) / (wTotal ** 2)
  );

  const gnnLatentDist = gnnTcValid ? safe(gnnResult.latentDistance) : undefined;
  const ood = computeOODScore(formula, gnnLatentDist);
  const sigmaOOD = ood.oodSigmaPenalty * stdCombinedRaw;
  const stdCombined = Math.sqrt(stdCombinedRaw ** 2 + sigmaOOD ** 2);

  const tcCI95Lower = Math.max(0, tcCombined - 1.96 * stdCombined);
  const tcCI95Upper = tcCombined + 1.96 * stdCombined;

  const gnnLambdaValid = Number.isFinite(gnnResult.lambda) && Number.isFinite(gnnResult.lambdaCI95?.[0]) && Number.isFinite(gnnResult.lambdaCI95?.[1]);
  const lambdaGnn = gnnLambdaValid ? gnnResult.lambda : 0;
  const lambdaXgb = features.electronPhononLambda ?? 0.5;
  const lambdaGnnVar = gnnLambdaValid ? (gnnResult.lambdaCI95[1] - gnnResult.lambdaCI95[0]) / (2 * 1.96) : 10;
  const lambdaGnnStd = Math.max(lambdaGnnVar, 0.01);
  const enSpreadVal = features.enSpread ?? 0;
  const lambdaXgbStd = enSpreadVal > 2.0
    ? 0.15 + (enSpreadVal - 2.0) * 0.08
    : enSpreadVal > 1.0
      ? 0.15 + (enSpreadVal - 1.0) * 0.03
      : 0.15;

  const wLGnn = 1 / (lambdaGnnStd ** 2);
  const wLXgb = 1 / (lambdaXgbStd ** 2);
  const wLTotal = wLGnn + wLXgb;
  const lambdaCombined = (wLGnn * lambdaGnn + wLXgb * lambdaXgb) / wLTotal;
  const lambdaStdCombined = Math.sqrt(1 / wLTotal);

  const conformal = getConformalInterval(tcCombined, stdCombined, 0.95);
  const calState = getCalibrationState();
  const eceMetrics = getECE();

  let calibrationNote = "Predictions are model estimates with quantified uncertainty";
  if (ood.isOOD) {
    calibrationNote = `OOD detected (${ood.oodCategory}, score=${ood.oodScore}): uncertainty inflated by OOD penalty`;
  } else if (conformal.method === "conformal") {
    calibrationNote = `Conformal calibration active (T=${calState.temperatureScale}, Q95=${conformal.quantile}, ECE=${eceMetrics.after.toFixed(4)}, n=${calState.calibrationDatasetSize})`;
  } else if (stdCombined > 50) {
    calibrationNote = "High uncertainty: prediction interval is wide, treat as exploratory";
  } else if (stdCombined > 20) {
    calibrationNote = "Moderate uncertainty: additional validation recommended";
  } else if (stdCombined < 5) {
    calibrationNote = "Low uncertainty: models show strong agreement on this composition";
  }

  const r = (v: number) => Number.isFinite(v) ? Math.round(v * 1000) / 1000 : 0;
  const rTc = (v: number) => Number.isFinite(v) ? Math.round(v * 10) / 10 : 0;

  return {
    formula,
    tcMean: rTc(tcCombined),
    tcCI95: [rTc(tcCI95Lower), rTc(tcCI95Upper)],
    tcTotalStd: rTc(stdCombined),
    tcEpistemicStd: rTc(epistemicCombined),
    tcAleatoricStd: rTc(aleatoricCombined),
    lambdaMean: r(lambdaCombined),
    lambdaCI95: [r(Math.max(0, lambdaCombined - 1.96 * lambdaStdCombined)), r(lambdaCombined + 1.96 * lambdaStdCombined)],
    gnn: {
      tcMean: gnnResult.tc,
      tcCI95: gnnResult.tcCI95,
      totalStd: gnnResult.totalStd,
      epistemicUncertainty: gnnResult.epistemicUncertainty,
      aleatoricUncertainty: gnnResult.aleatoricUncertainty,
      lambda: gnnResult.lambda,
      lambdaCI95: gnnResult.lambdaCI95,
      confidence: gnnResult.confidence,
      weight: r(wGnn / wTotal),
    },
    xgb: {
      tcMean: xgbResult.tcMean,
      tcCI95: xgbResult.tcCI95,
      totalStd: xgbResult.totalStd,
      epistemicStd: xgbResult.epistemicStd,
      aleatoricStd: xgbResult.aleatoricStd,
      perModelPredictions: xgbResult.perModelPredictions,
      weight: r(wXgb / wTotal),
    },
    modelCount: 10,
    calibrationNote,
    calibratedCI95: [conformal.lower, conformal.upper],
    conformalQuantile: conformal.quantile,
    temperatureScale: calState.temperatureScale,
    ece: eceMetrics.after,
    calibrationDatasetSize: calState.calibrationDatasetSize,
    conformalMethod: conformal.method,
    oodScore: ood.oodScore,
    oodSigmaPenalty: ood.oodSigmaPenalty,
    isOOD: ood.isOOD,
    oodCategory: ood.oodCategory,
    mahalanobisDistance: ood.mahalanobisDistance,
    physicsUQ: (() => {
      try {
        const puq = computePhysicsTcUQ(formula);
        return {
          tcMean: puq.mean,
          tcStd: puq.std,
          tcCI95: puq.ci95,
          dominantUncertaintySource: puq.dominant_uncertainty_source,
          lambdaContribution: puq.errorPropagation.lambdaContribution,
          omegaLogContribution: puq.errorPropagation.omegaLogContribution,
          muStarContribution: puq.errorPropagation.muStarContribution,
          mcSamples: puq.mcSamples,
        };
      } catch {
        return undefined;
      }
    })(),
  };
}

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
  anharmonicityScore: number;
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
  lambdaProxy: number;
  alphaCouplingStrength: number;
  phononHardness: number;
  massEnhancement: number;
  couplingAsymmetry: number;
  dosAtEF_tb: number;
  bandFlatness_tb: number;
  lambdaProxy_tb: number;
  derivedBandwidth: number;
  derivedVanHoveDistance: number;
  derivedBondStiffness: number;
  derivedEPCDensity: number;
  derivedSpectralWeight: number;
  derivedAnharmonicRatio: number;
  derivedCouplingEfficiency: number;
  disorderVacancyFraction: number;
  disorderBondVariance: number;
  disorderLatticeStrain: number;
  disorderSiteMixingEntropy: number;
  disorderConfigEntropy: number;
  disorderDosSignal: number;
  dopingCarrierDensity: number;
  dopingLatticeStrain: number;
  dopingBondVariance: number;
  dopantAtomicNumber: number;
  dopantFraction: number;
  dopantValenceDiff: number;
  dynamicLatticeScore: number;
  softPhononCount: number;
  phononVarianceScore: number;
  instabilityFlag: number;
  lightElementFraction: number;
  cageLatticeFlag: number;
  looselyBondedFraction: number;
  combinedElectronPhononScore: number;
  anharmonicityScore: number;
  maxAtomicNumber: number;
  feasibilityScore: number;
  _sourceFormula?: string;
}

const CHALCOGENS = ["O","S","Se","Te"];
const PNICTOGENS = ["N","P","As","Sb","Bi"];

function parseFormula(formula: string): string[] {
  return Object.keys(parseFormulaCountsCanonical(formula));
}

function parseFormulaCounts(formula: string): Record<string, number> {
  return parseFormulaCountsCanonical(formula);
}

export interface DisorderContext {
  vacancyFraction?: number;
  bondVariance?: number;
  latticeStrain?: number;
  siteMixingEntropy?: number;
  configurationalEntropy?: number;
  dosDisorderSignal?: number;
}

export async function extractFeatures(formula: string, mat?: Partial<Material>, physics?: PhysicsContext, crystal?: CrystalContext, dftData?: DFTResolvedFeatures, disorderCtx?: DisorderContext): Promise<MLFeatureVector> {
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

  const LIGHT_ELEMENTS = ["H","He","Li","Be","B","C","N","O","F"];
  const lightElementFraction = elements.filter(e => LIGHT_ELEMENTS.includes(e))
    .reduce((s, e) => s + (counts[e] ?? 0), 0) / totalAtoms;
  const hFraction = (counts["H"] ?? 0) / totalAtoms;

  const avgEN = getCompositionWeightedProperty(counts, "paulingElectronegativity") ?? 1.5;
  const enSpread = enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const dWaveSymmetry = hasCu && hasO && elements.length >= 3;
  const sg = mat?.spacegroup ?? "";
  const layeredStructure = sg.includes("P4") || sg.includes("Pmmm") || sg.includes("I4") || sg.includes("R-3m") || sg.includes("P63/mmc") || sg.includes("P6/mmm") || sg.includes("C2/m") || sg.includes("Cmcm");

  let cooperPairStrengthBase = (hasTransitionMetal ? 0.15 : 0) + (hasHydrogen ? 0.1 : 0) +
    (dWaveSymmetry ? 0.1 : 0) + (layeredStructure ? 0.05 : 0) + (enSpread > 1.5 ? 0.05 : 0);

  const corrForCooper = physics?.correlationStrength ?? 0;
  if (corrForCooper >= 0.5 && corrForCooper <= 0.8) {
    if (dWaveSymmetry) cooperPairStrengthBase += 0.1;
    else if (layeredStructure) cooperPairStrengthBase += 0.05;
  }
  cooperPairStrengthBase = Math.min(0.5, Math.max(0, cooperPairStrengthBase));

  const candidatePressureForLambda = (mat as any)?.pressureGpa ?? 0;

  const hCount = counts["H"] || 0;
  const metalAtomCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hydrogenRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;

  const [
    { electronic, phonon, coupling },
    correlation,
    lambdaPrediction,
    _miedemaFE,
    _tbSafe,
    _motifs,
    _dimScoreV2,
  ] = await Promise.all([
    (async () => {
      await yieldToEventLoop();
      const electronic = computeElectronicStructure(formula, mat?.spacegroup);
      await yieldToEventLoop();
      const phonon = computePhononSpectrum(formula, electronic);
      await yieldToEventLoop();
      const coupling = computeElectronPhononCoupling(electronic, phonon, formula);
      return { electronic, phonon, coupling };
    })(),
    yieldToEventLoop().then(() => assessCorrelationStrength(formula)),
    yieldToEventLoop().then(() => predictLambda(formula, candidatePressureForLambda)),
    yieldToEventLoop().then(() => { try { return computeMiedemaFormationEnergy(formula); } catch { return null as number | null; } }),
    yieldToEventLoop().then(() => predictTBProperties(formula)),
    yieldToEventLoop().then(() => detectStructuralMotifs(formula)),
    yieldToEventLoop().then(() => computeDimensionalityScore(formula)),
  ]);
  const useLambda = physics?.verifiedLambda ?? lambdaPrediction.lambda;
  const lambdaTier = physics?.verifiedLambda ? "verified" : lambdaPrediction.tier;
  const useCorrelation = physics?.correlationStrength ?? correlation.ratio;

  if (lambdaPrediction.tier === "ml-regression" && physics?.verifiedLambda) {
    recordLambdaValidation(formula, lambdaPrediction.lambda, physics.verifiedLambda);
  }

  const phononCouplingEstimate = Math.min(1.0, useLambda / 3.0);

  const lambdaContribution = useLambda / (1 + useLambda);
  let cooperPairStrength = Math.min(1, cooperPairStrengthBase * 0.3 + lambdaContribution * 0.7);

  const effPressure = (mat as any)?.pressureGpa ?? candidatePressureForLambda;
  if (hasHydrogen && hydrogenRatio >= 4 && effPressure > 50) {
    const pressureBoost = Math.min(0.15, (effPressure - 50) / 1000);
    cooperPairStrength = Math.min(1, cooperPairStrength + pressureBoost);
  }

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
    if (dimensionalityScore > 0.7) {
      chargeTransferMagnitude *= 1.0 + (dimensionalityScore - 0.7) * 1.5;
    }
  }

  let connectivityIndex = 0.5;
  if (layeredStructure) connectivityIndex = 0.3;
  else if (dimensionalityScore > 0.7) connectivityIndex = 0.25;
  else if (elements.length === 1) connectivityIndex = 0.8;
  else if (elements.length >= 4) connectivityIndex = 0.6;
  if (hasHydrogen && hydrogenRatio >= 6) connectivityIndex = Math.max(connectivityIndex, 0.7);

  const rawSoftModeSignal = phonon.softModeScore ?? (phonon.softModePresent ? 0.5 : 0);
  const phononSofteningIndex = Math.min(1.0,
    phonon.anharmonicityIndex * (1 - rawSoftModeSignal * 0.5) +
    (phonon.softModePresent ? 0.1 : 0)
  );

  let spinFluctuationStrength = 0;
  for (const el of elements) {
    const stonerI = getStonerParameter(el);
    if (stonerI !== null && stonerI > 0) {
      const frac = totalAtoms > 0 ? (counts[el] || 1) / totalAtoms : 0;
      const stonerProduct = stonerI * electronic.densityOfStatesAtFermi;
      spinFluctuationStrength += stonerProduct * frac;
    }
  }
  spinFluctuationStrength = Math.min(1.0, spinFluctuationStrength);

  const fermiSurfaceNestingScore = electronic.nestingScore ?? 0;

  let dosAtEF = electronic.densityOfStatesAtFermi;

  let muStarEstimate = coupling.muStar;
  if (muStarEstimate >= 0.10 && muStarEstimate <= 0.13) {
    if (hasHydrogen && hydrogenRatio >= 4) {
      muStarEstimate = 0.10;
    } else if (hasCu && hasO && elements.length >= 3) {
      muStarEstimate = 0.13;
    } else if (elements.some(e => {
      const z = getElementData(e)?.atomicNumber ?? 0;
      return z >= 72 && z <= 80;
    }) && hasO) {
      muStarEstimate = 0.15;
    } else if (hasRareEarth) {
      muStarEstimate = 0.12;
    } else if (elements.length >= 4 && enSpread > 1.5) {
      muStarEstimate = 0.14;
    }
  }

  let candidatePressure = (mat as any)?.pressureGpa ?? 0;

  if (candidatePressure === 0 && hasHydrogen && hydrogenRatio > 0.3) {
    const hBondType = classifyHydrogenBonding(formula, 0);
    const isClathrate = hBondType === "cage-clathrate";
    if (isClathrate) {
      candidatePressure = hydrogenRatio >= 9 ? 250 : 200;
    } else if (hydrogenRatio >= 8) candidatePressure = 200;
    else if (hydrogenRatio >= 6) candidatePressure = 150;
    else if (hydrogenRatio >= 4) candidatePressure = 100;
  }

  let optimalPressureGpa = 0;
  try {
    await yieldToEventLoop();
    const pressureResult = simulatePressureEffects(formula, electronic, phonon, coupling);
    optimalPressureGpa = pressureResult.optimalPressure;
    if (optimalPressureGpa > candidatePressure) {
      candidatePressure = optimalPressureGpa;
    }
  } catch {}

  let finalLambda = Math.max(0.001, useLambda);
  let finalLogPhononFreq = coupling.omegaLog;
  let hasStablePhonons = !phonon.hasImaginaryModes;
  let lambdaProxy = 0;
  let alphaCouplingStrength = 0;
  let phononHardnessVal = 0;
  let massEnhancementVal = 1 + useLambda;
  let couplingAsymmetryVal = 1.0;

  const physicsStore = getPhysicsFeatures(formula);
  if (physicsStore) {
    finalLambda = physicsStore.verifiedLambda;
    finalLogPhononFreq = physicsStore.verifiedOmegaLog;
    dosAtEF = 0.3 * dosAtEF + 0.7 * physicsStore.verifiedDosEF;
    hasStablePhonons = physicsStore.verifiedPhononStable;
    muStarEstimate = physicsStore.verifiedMuStar;

    lambdaProxy = physicsStore.lambdaProxy;
    alphaCouplingStrength = physicsStore.alphaCouplingStrength;
    phononHardnessVal = physicsStore.phononHardness;
    massEnhancementVal = physicsStore.massEnhancement;
    couplingAsymmetryVal = physicsStore.couplingAsymmetry;
  } else {
    const avgMass = getAverageMass(counts);
    const avgEta = getCompositionWeightedProperty(counts, "mcMillanHopfieldEta") ?? 0;
    const omega2 = coupling.omega2Avg > 0 ? coupling.omega2Avg : (phonon.maxPhononFrequency * 0.7);
    if (dosAtEF > 0 && avgMass > 0 && omega2 > 0) {
      const omega2_meV2 = omega2 * 0.124 * 0.124;
      const mass_eV = avgMass * 931.494e6;
      lambdaProxy = (dosAtEF * avgEta) / (mass_eV * omega2_meV2 * 1e-6);
      lambdaProxy = Math.min(3.0, lambdaProxy);
    }
    phononHardnessVal = coupling.omega2Avg > 0 ? coupling.omegaLog / coupling.omega2Avg : 0;
  }

  const miedemaFE = _miedemaFE;

  const tbSafe = _tbSafe;

  const normalizedFormula = normalizeFormula(formula);

  const anharmonicityScore = Math.min(1.0, phonon.anharmonicityIndex * 2.5);

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
    formationEnergy: mat?.formationEnergy ?? miedemaFE,
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
    electronPhononLambda: finalLambda,
    logPhononFreq: finalLogPhononFreq,
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
    motifScore: _motifs.motifScore,
    orbitalDFraction: electronic.orbitalFractions?.d ?? 0,
    mottProximityScore: electronic.mottProximityScore ?? 0,
    topologicalBandScore: electronic.topologicalBandScore ?? 0,
    dimensionalityScoreV2: _dimScoreV2,
    phononSofteningIndex,
    spinFluctuationStrength,
    fermiSurfaceNestingScore,
    dosAtEF,
    muStarEstimate,
    pressureGpa: candidatePressure,
    optimalPressureGpa,
    lambdaProxy,
    alphaCouplingStrength,
    phononHardness: phononHardnessVal,
    massEnhancement: massEnhancementVal,
    couplingAsymmetry: couplingAsymmetryVal,
    dosAtEF_tb: tbSafe.dosAtEF,
    bandFlatness_tb: tbSafe.bandFlatness,
    lambdaProxy_tb: tbSafe.lambdaProxy,
    ...(() => {
      const derived = getDerivedFeatures(formula);
      if (derived) {
        return {
          derivedBandwidth: derived.bandwidth,
          derivedVanHoveDistance: derived.vanHoveDistance,
          derivedBondStiffness: derived.bondStiffness,
          derivedEPCDensity: derived.electronPhononCouplingDensity,
          derivedSpectralWeight: derived.spectralWeight,
          derivedAnharmonicRatio: derived.anharmonicRatio,
          derivedCouplingEfficiency: derived.couplingEfficiency,
        };
      }
      return {
        derivedBandwidth: 0,
        derivedVanHoveDistance: 0,
        derivedBondStiffness: bondStiffnessVariance,
        derivedEPCDensity: 0,
        derivedSpectralWeight: 0,
        derivedAnharmonicRatio: 0,
        derivedCouplingEfficiency: 0,
      };
    })(),
    disorderVacancyFraction: disorderCtx?.vacancyFraction ?? 0,
    disorderBondVariance: disorderCtx?.bondVariance ?? 0,
    disorderLatticeStrain: disorderCtx?.latticeStrain ?? 0,
    disorderSiteMixingEntropy: disorderCtx?.siteMixingEntropy ?? 0,
    disorderConfigEntropy: disorderCtx?.configurationalEntropy ?? 0,
    disorderDosSignal: disorderCtx?.dosDisorderSignal ?? 0,
    dopingCarrierDensity: (mat as any)?.dopingCarrierDensity ?? 0,
    dopingLatticeStrain: (mat as any)?.dopingLatticeStrain ?? 0,
    dopingBondVariance: (mat as any)?.dopingBondVariance ?? 0,
    dopantAtomicNumber: (mat as any)?.dopantAtomicNumber ?? 0,
    dopantFraction: (mat as any)?.dopantFraction ?? 0,
    dopantValenceDiff: (mat as any)?.dopantValenceDiff ?? 0,
    dynamicLatticeScore: (() => {
      const smf = phonon.softModeScore;
      const imf = phonon.hasImaginaryModes ? 1.0 : 0.0;
      const pv = phonon.maxPhononFrequency > 0 ? Math.min(1.0, ((phonon.maxPhononFrequency - phonon.logAverageFrequency) / phonon.maxPhononFrequency) * 1.5) : 0;
      const leb = hFraction > 0.3
        ? Math.min(1.5, Math.min(1.0, 0.4 + hFraction * 0.8) + (lightElementFraction - hFraction) * 1.5)
        : Math.min(1.0, lightElementFraction * 2.0);
      const raw = smf * 1.5 + imf * 0.8 + pv * 1.2 + anharmonicityScore * 1.0 + leb * 0.7;
      return 1.0 / (1.0 + Math.exp(-1.2 * (raw - 3.0)));
    })(),
    softPhononCount: Math.round(phonon.softModeScore * totalAtoms * 3),
    phononVarianceScore: phonon.maxPhononFrequency > 0 ? Math.min(1.0, ((phonon.maxPhononFrequency - phonon.logAverageFrequency) / phonon.maxPhononFrequency) * 1.5) : 0,
    instabilityFlag: phonon.hasImaginaryModes ? 1 : 0,
    lightElementFraction,
    cageLatticeFlag: (() => {
      const CLATHRATE_SG = ["Pm-3n", "Fd-3m", "Im-3m", "P4_132", "Pm3n", "Fd3m", "Im3m"];
      if (useSpacegroup && CLATHRATE_SG.some(sg => useSpacegroup.includes(sg))) return 1;
      const hasFramework = elements.some((e: string) => ["B","C","Si","Ge","Al","Ga","Sn"].includes(e));
      const hasGuest = elements.some((e: string) => ["La","Ce","Ba","Sr","Ca","Y","K","Na","Rb","Cs"].includes(e));
      return (hasFramework && hasGuest) ? 1 : 0;
    })(),
    maxAtomicNumber: Math.max(...elements.map(e => getElementData(e)?.atomicNumber ?? 0)),
    feasibilityScore: (() => {
      const TOXIC_ELEMENTS: Record<string, number> = { Be: 0.3, Tl: 0.25, Cd: 0.3, Hg: 0.35, Pb: 0.2, As: 0.25, Os: 0.15, Cr: 0.1 };
      const SCARCE_ELEMENTS: Record<string, number> = { Re: 0.25, Ir: 0.3, Os: 0.3, Rh: 0.2, Ru: 0.15, Pd: 0.15, Pt: 0.2, Au: 0.1, Te: 0.15, In: 0.1 };
      let penalty = 0;
      for (const el of elements) {
        const frac = totalAtoms > 0 ? (counts[el] || 1) / totalAtoms : 0;
        if (TOXIC_ELEMENTS[el]) penalty += TOXIC_ELEMENTS[el] * frac;
        if (SCARCE_ELEMENTS[el]) penalty += SCARCE_ELEMENTS[el] * frac;
      }
      return Math.max(0, 1 - penalty);
    })(),
    looselyBondedFraction: elements.filter((e: string) => {
      const d = getElementData(e);
      return d && ((d.paulingElectronegativity ?? 2.0) < 1.2 && (d.atomicMass ?? 40) > 30);
    }).reduce((s: number, e: string) => s + (counts[e] ?? 0), 0) / totalAtoms,
    combinedElectronPhononScore: (() => {
      const dosNorm = electronic.densityOfStatesAtFermi > 0 ? Math.min(1, electronic.densityOfStatesAtFermi / 5.0) : 0;
      const lam = Math.min(1, useLambda / 2.5);
      const corr = useCorrelation;
      const corrBonus = (corr > 0.3 && corr < 0.8) ? 0.15 : corr * 0.05;
      return dosNorm * 0.20 + electronic.metallicity * 0.15 + lam * 0.25 + electronic.nestingScore * 0.15 + corrBonus + electronic.vanHoveProximity * 0.10;
    })(),
    anharmonicityScore,
    _sourceFormula: normalizedFormula,
  };
}

async function xgboostPredict(features: MLFeatureVector): Promise<{ score: number; tcEstimate: number; tcStd: number; tcCI95: [number, number]; reasoning: string[] }> {
  const [gb, unc] = await Promise.all([
    await gbPredict(features),
    await gbPredictWithUncertainty(features),
  ]);

  const safeHc2 = features.upperCriticalField != null && Number.isFinite(features.upperCriticalField) ? features.upperCriticalField : null;
  if (safeHc2 != null) {
    if (safeHc2 > 50) {
      gb.score = Math.min(0.95, gb.score + 0.05);
      gb.reasoning.push(`High Hc2 (${safeHc2.toFixed(1)}T): robust Type-II superconductor`);
    } else if (safeHc2 > 0 && safeHc2 <= 0.5) {
      gb.reasoning.push(`Low Hc2 (${safeHc2.toFixed(2)}T): possible Type-I superconductor — Hc2 not penalized`);
    } else if (safeHc2 === 0) {
      const isLikelyTypeI = features.numElements <= 2 && !features.hasRareEarth && !features.layeredStructure;
      if (isLikelyTypeI) {
        gb.score = Math.max(0.01, gb.score - 0.03);
        gb.reasoning.push("Hc2=0T: likely Type-I (elemental/binary, low-field) — mild penalty");
      } else {
        gb.score = Math.max(0.01, gb.score - 0.06);
        gb.reasoning.push("Hc2=0T: upper critical field absent or computation did not converge");
      }
    }
  }

  return {
    score: gb.score,
    tcEstimate: Math.round(gb.tcPredicted),
    tcStd: unc.totalStd,
    tcCI95: unc.tcCI95,
    reasoning: gb.reasoning,
  };
}

interface PhysicsContext {
  verifiedLambda: number | null;
  verifiedTc: number | null;
  tcSource: "predicted" | "physics_verified" | "dft_verified";
  competingPhases: any[];
  upperCriticalField: number | null;
  correlationStrength: number | null;
  verificationStage: number;
}

const featureCache = new Map<string, { features: MLFeatureVector; timestamp: number }>();
const FEATURE_CACHE_TTL_MS = 5 * 60 * 1000;

export function getCachedFeatures(normKey: string): MLFeatureVector | null {
  const entry = featureCache.get(normKey);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > FEATURE_CACHE_TTL_MS) {
    featureCache.delete(normKey);
    return null;
  }
  return entry.features;
}

function setCachedFeatures(normKey: string, features: MLFeatureVector): void {
  if (featureCache.size > 500) {
    const cutoff = Date.now() - FEATURE_CACHE_TTL_MS;
    for (const [k, v] of featureCache) {
      if (v.timestamp < cutoff) featureCache.delete(k);
    }
    if (featureCache.size > 500) {
      const oldest = [...featureCache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp);
      for (let i = 0; i < oldest.length - 400; i++) featureCache.delete(oldest[i][0]);
    }
  }
  featureCache.set(normKey, { features, timestamp: Date.now() });
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

  let physicsData = context?.physicsData ?? new Map<string, PhysicsContext>();
  let crystalData = context?.crystalData ?? new Map<string, CrystalContext>();
  const batchFormulas = materials.slice(0, 100).map(m => m.formula);
  const batchNormalized = batchFormulas.map(f => normalizeFormula(f));
  const uniqueFormulas = [...new Set([...batchFormulas, ...batchNormalized])];
  try {
    if (!context?.physicsData) {
      const existingSC = await storage.getSuperconductorCandidatesByFormulas(uniqueFormulas);
      for (const sc of existingSC) {
        if (sc.electronPhononCoupling != null || (sc.verificationStage != null && sc.verificationStage > 0)) {
          const stage = sc.verificationStage ?? 0;
          const tcSource: PhysicsContext["tcSource"] = stage >= 3 ? "dft_verified" : stage >= 1 ? "physics_verified" : "predicted";
          physicsData.set(normalizeFormula(sc.formula), {
            verifiedLambda: sc.electronPhononCoupling,
            verifiedTc: sc.predictedTc,
            tcSource,
            competingPhases: (sc.competingPhases as any[]) ?? [],
            upperCriticalField: sc.upperCriticalField,
            correlationStrength: sc.correlationStrength,
            verificationStage: stage,
          });
        }
      }
    }
    if (!context?.crystalData) {
      const structMap = await storage.getCrystalStructuresByFormulas(uniqueFormulas);
      for (const [formula, structs] of structMap) {
        if (structs.length > 0) {
          const cs = structs[0];
          crystalData.set(normalizeFormula(formula), {
            spaceGroup: cs.spaceGroup,
            crystalSystem: cs.crystalSystem,
            dimensionality: cs.dimensionality,
            isStable: cs.isStable ?? false,
            convexHullDistance: cs.convexHullDistance,
            synthesizability: cs.synthesizability,
          });
        }
      }
    }
  } catch (storageErr) {
    console.error("[ML] Failed to fetch physics/crystal context from storage:", storageErr instanceof Error ? storageErr.message : storageErr);
    emit("log", {
      phase: "phase-7",
      event: "Context fetch error",
      detail: `Storage query failed: ${storageErr instanceof Error ? storageErr.message : "unknown error"} — proceeding with ${physicsData.size} physics, ${crystalData.size} crystal entries`,
      dataSource: "ML Engine",
    });
  }

  const scored: { mat: Material; features: MLFeatureVector; xgb: Awaited<ReturnType<typeof xgboostPredict>>; hasPhysics: boolean; hasCrystal: boolean; gnn: GNNPrediction | null }[] = [];

  let preFilterSkipped = 0;
  const matSlice = materials.slice(0, 100);
  const BATCH_CONCURRENCY = 5;
  for (let batchStart = 0; batchStart < matSlice.length; batchStart += BATCH_CONCURRENCY) {
    const batchEnd = Math.min(batchStart + BATCH_CONCURRENCY, matSlice.length);
    const batchItems = matSlice.slice(batchStart, batchEnd);

    const batchResults = await Promise.all(batchItems.map(async (mat) => {
      const normKey = normalizeFormula(mat.formula);
      const physics = physicsData.get(normKey);
      const crystal = crystalData.get(normKey);
      let features = await getCachedFeatures(normKey);
      if (!features) {
        features = await extractFeatures(mat.formula, mat, physics, crystal);
        setCachedFeatures(normKey, features);
      }

      if ((features.bandGap ?? 0) > 0.5 || (features.metallicity ?? 0.5) < 0.2) {
        return null;
      }

      const xgb = await xgboostPredict(features);

      if (physics && physics.verificationStage > 0) {
        const verifyBonus = physics.tcSource === "dft_verified" ? 0.08 : physics.tcSource === "physics_verified" ? 0.05 : 0.02;
        xgb.score = Math.min(1, xgb.score + verifyBonus);
        xgb.reasoning.push(`${physics.tcSource === "dft_verified" ? "DFT" : "Physics"}-verified (stage ${physics.verificationStage}): Tc=${physics.verifiedTc?.toFixed(0) ?? '?'}K`);
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

      return { mat, features, xgb, hasPhysics: !!physics, hasCrystal: !!crystal, gnn: null as GNNPrediction | null };
    }));

    for (const result of batchResults) {
      if (result === null) {
        preFilterSkipped++;
      } else {
        scored.push(result);
      }
    }

    await new Promise<void>(resolve => setImmediate(resolve));
  }

  scored.sort((a, b) => b.xgb.score - a.xgb.score);

  const GNN_INFERENCE_LIMIT = 15;
  for (let i = 0; i < Math.min(GNN_INFERENCE_LIMIT, scored.length); i++) {
    if (i > 0 && i % 5 === 0) {
      await new Promise<void>(resolve => setImmediate(resolve));
    }
    const entry = scored[i];
    try {
      const crystal = crystalData.get(normalizeFormula(entry.mat.formula));
      const crystalStructure = crystal ? { spaceGroup: crystal.spaceGroup, latticeParams: [], atomicPositions: [] } : undefined;
      entry.gnn = getGNNPrediction(entry.mat.formula, crystalStructure);
    } catch {}
  }

  function computeEnsembleScore(entry: typeof scored[0], nnScore?: number): number {
    const gnnScore = entry.gnn
      ? (Math.min(1, entry.gnn.predictedTc > 100 ? 0.8 : entry.gnn.predictedTc > 20 ? 0.5 : 0.2) * entry.gnn.confidence)
      : 0;
    const structBonus = entry.hasCrystal ? 0.1 : 0;
    const physicsBonus = entry.hasPhysics ? 0.05 : 0;
    if (entry.gnn) {
      return Math.min(0.95, gnnScore * 0.6 + entry.xgb.score * 0.3 + structBonus + physicsBonus);
    }
    if (nnScore != null) {
      return Math.min(0.95, entry.xgb.score * 0.4 + nnScore * 0.6);
    }
    return Math.min(0.95, entry.xgb.score * 0.5 + structBonus + physicsBonus);
  }

  scored.sort((a, b) => computeEnsembleScore(b) - computeEnsembleScore(a));
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
    detail: `Top candidate: ${topCandidates[0].mat.formula} (score: ${(topCandidates[0].xgb.score*100).toFixed(0)}%, Tc: ${topCandidates[0].xgb.tcEstimate}K +/-${topCandidates[0].xgb.tcStd.toFixed(1)}K)${preFilterSkipped > 0 ? `, ${preFilterSkipped} non-metallic filtered` : ""}${enrichmentDetail}`,
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

  await new Promise<void>(resolve => setImmediate(resolve));

  try {
    const candidateSummaries = topCandidates.map(c => {
      const physics = physicsData!.get(normalizeFormula(c.mat.formula));
      const crystal = crystalData!.get(normalizeFormula(c.mat.formula));
      return {
        formula: c.mat.formula,
        name: c.mat.name,
        xgboostScore: c.xgb.score,
        tcEstimate: c.xgb.tcEstimate,
        tcStd: Math.round(c.xgb.tcStd * 10) / 10,
        tcCI95: [Math.round(c.xgb.tcCI95[0] * 10) / 10, Math.round(c.xgb.tcCI95[1] * 10) / 10],
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
          anharmonic: c.features.anharmonicityScore > 0.4,
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
            content: `You are an NN layer in an XGBoost+NN superconductor ensemble. Evaluate candidates using BCS/Eliashberg theory, electron-phonon coupling (lambda), competing phases, pairing symmetry, and dimensionality effects. Weight verifiedPhysics and crystalStructure data heavily. Penalize Hc2=0 severely; high Hc2 (>50T) strongly supports SC. Room-temp viable = Tc>=293K AND ambient pressure AND zero resistance. CRITICAL: Return ONLY valid JSON, no extra text. Keep each reasoning under 80 chars. Return: {"candidates": [{"formula": str, "neuralNetScore": 0-1, "refinedTc": K, "pressureGpa": number, "meissnerEffect": bool, "zeroResistance": bool, "cooperPairMechanism": str, "pairingSymmetry": str, "pairingMechanism": str, "dimensionality": str, "quantumCoherence": 0-1, "roomTempViable": bool, "crystalStructure": str, "uncertaintyEstimate": 0-1, "reasoning": "<80 chars"}], "insights": ["<100 chars", ...3-5 items]}`,
          },
          {
            role: "user",
            content: `XGBoost-ranked candidates:\n${JSON.stringify(candidateSummaries)}`,
          },
        ],
        response_format: { type: "json_object" },
        max_completion_tokens: 1500,
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
      let repaired = content.trim();
      if (!repaired.endsWith("}")) {
        const lastBrace = repaired.lastIndexOf("}");
        if (lastBrace > 0) {
          repaired = repaired.slice(0, lastBrace + 1);
          const openBraces = (repaired.match(/\{/g) || []).length;
          const closeBraces = (repaired.match(/\}/g) || []).length;
          for (let b = 0; b < openBraces - closeBraces; b++) repaired += "}";
          const openBrackets = (repaired.match(/\[/g) || []).length;
          const closeBrackets = (repaired.match(/\]/g) || []).length;
          for (let b = 0; b < openBrackets - closeBrackets; b++) repaired += "]";
          if (!repaired.endsWith("}")) repaired += "}";
        }
      }
      try {
        parsed = JSON.parse(repaired);
        emit("log", { phase: "phase-7", event: "NN JSON repaired", detail: `Truncated response recovered (${content.length} chars)`, dataSource: "ML Engine" });
      } catch {
        emit("log", { phase: "phase-7", event: "NN parse error", detail: `Unrecoverable JSON (${content.length} chars): ${content.slice(0, 150)}`, dataSource: "ML Engine" });
        return { candidates, insights };
      }
    }

    const nnCandidates = parsed.candidates ?? [];
    const nnInsights = parsed.insights ?? [];
    insights.push(...nnInsights);

    const nnByFormula = new Map<string, any>();
    for (const nn of nnCandidates) {
      if (nn?.formula) nnByFormula.set(normalizeFormula(nn.formula), nn);
    }

    for (const xgb of topCandidates) {
      const nn = nnByFormula.get(normalizeFormula(xgb.mat.formula));
      if (!nn) continue;

      const ensembleScore = computeEnsembleScore(xgb, nn.neuralNetScore);

      const featureLambda = xgb.features.electronPhononLambda ?? 0;
      const llmRefinedTc = nn.refinedTc ?? xgb.xgb.tcEstimate;
      const corrStrength = xgb.features.correlationStrength;
      const isStronglyCorrelated = corrStrength > 0.6;

      const effectiveMuStar = xgb.features.muStarEstimate;

      const isHydrideML = xgb.features.hasHydrogen && xgb.features.hydrogenRatio >= 0.5;
      let finalTc: number;
      let tcMethod: string;
      if (isStronglyCorrelated) {
        const corrSuppression = Math.max(0.1, 1.0 - (corrStrength - 0.6) * 1.5);
        const rawAD = featureLambda > 0
          ? allenDynesTcRaw(featureLambda, xgb.features.logPhononFreq ?? 300, effectiveMuStar, undefined, isHydrideML)
          : 0;
        finalTc = Math.round(Math.max(0, rawAD * corrSuppression));
        tcMethod = `Allen-Dynes*corr_supp(${corrSuppression.toFixed(2)})`;
      } else {
        const allenDynesTc = featureLambda > 0
          ? allenDynesTcRaw(featureLambda, xgb.features.logPhononFreq ?? 300, effectiveMuStar, undefined, isHydrideML)
          : 0;
        finalTc = Math.round(allenDynesTc > 0 ? allenDynesTc : 0);
        tcMethod = "Allen-Dynes";
      }

      const heuristicPressure = estimateFamilyPressure(xgb.mat.formula);
      const featurePressure = xgb.features.pressureGpa;
      const llmPressure = nn.pressureGpa ?? null;
      let resolvedPressure: number;
      let pressureSource: string;
      if (llmPressure != null && llmPressure === 0 && heuristicPressure > 50) {
        resolvedPressure = featurePressure > 0 ? featurePressure : Math.round(heuristicPressure * 0.3);
        pressureSource = `LLM=ambient, heuristic=${heuristicPressure}GPa, resolved=${resolvedPressure}GPa (LLM-favored)`;
      } else if (llmPressure != null && heuristicPressure > 0) {
        resolvedPressure = Math.round(llmPressure * 0.6 + heuristicPressure * 0.4);
        pressureSource = `blend: LLM=${llmPressure}GPa(60%) + heuristic=${heuristicPressure}GPa(40%)`;
      } else if (llmPressure != null) {
        resolvedPressure = llmPressure;
        pressureSource = `LLM=${llmPressure}GPa`;
      } else if (featurePressure > 0) {
        resolvedPressure = featurePressure;
        pressureSource = `features=${featurePressure}GPa`;
      } else {
        resolvedPressure = heuristicPressure;
        pressureSource = `heuristic=${heuristicPressure}GPa`;
      }

      const pairingMech = nn.pairingMechanism ?? (isStronglyCorrelated ? "spin-fluctuation" : "phonon-mediated");
      const uncertaintyBase = nn.uncertaintyEstimate ?? 0.5;
      const adjustedUncertainty = isStronglyCorrelated
        ? Math.min(1.0, uncertaintyBase + (corrStrength - 0.6) * 0.5)
        : uncertaintyBase;

      candidates.push({
        name: xgb.mat.name,
        formula: xgb.mat.formula,
        predictedTc: finalTc,
        pressureGpa: resolvedPressure,
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
        notes: (llmRefinedTc !== finalTc ? `[LLM Tc=${llmRefinedTc}K, ${tcMethod} Tc=${finalTc}K (lambda=${featureLambda.toFixed(2)}, mu*=${effectiveMuStar.toFixed(3)})] ` : '') + (isStronglyCorrelated ? `[correlated: Allen-Dynes unreliable] ` : '') + `[P: ${pressureSource}] ` + (nn.reasoning ?? xgb.xgb.reasoning[0]),
        electronPhononCoupling: xgb.features.electronPhononLambda,
        logPhononFrequency: xgb.features.logPhononFreq,
        coulombPseudopotential: effectiveMuStar,
        pairingMechanism: pairingMech,
        pairingSymmetry: nn.pairingSymmetry ?? ((() => { if (pairingMech.includes("spin")) return "d-wave"; if (pairingMech.includes("topolog")) return "p-wave"; return xgb.features.dWaveSymmetry ? "d-wave" : "s-wave"; })()),
        correlationStrength: corrStrength,
        dimensionality: nn.dimensionality ?? (xgb.features.layeredStructure ? "quasi-2D" : "3D"),
        fermiSurfaceTopology: xgb.features.fermiSurfaceType,
        uncertaintyEstimate: adjustedUncertainty,
        verificationStage: 0,
        dataConfidence: xgb.hasPhysics ? "high" : (xgb.hasCrystal ? "medium" : "low"),
      });
    }

    emit("log", {
      phase: "phase-7",
      event: "Ensemble prediction complete",
      detail: `${candidates.length} candidates scored, ${candidates.filter(c => c.roomTempViable).length} room-temp viable${candidates.length > 0 ? `, top: ${candidates[0].formula} Tc=${candidates[0].predictedTc}K` : ''}`,
      dataSource: "ML Engine",
    });

    if (nnInsights.length > 0) {
      emit("log", {
        phase: "phase-7",
        event: "NN ensemble insights",
        detail: nnInsights.slice(0, 5).join(" | "),
        dataSource: "ML Engine",
      });
    }
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
        const ensembleScore = computeEnsembleScore(c);

        const featureLambda = c.features.electronPhononLambda ?? 0;
        const corrStrengthFB = c.features.correlationStrength;
        const isCorrelatedFB = corrStrengthFB > 0.6;
        const effectiveMuStarFB = c.features.muStarEstimate;
        const isHydrideFB = c.features.hasHydrogen && c.features.hydrogenRatio >= 0.5;

        let finalTc: number;
        if (isCorrelatedFB) {
          const suppression = Math.max(0.1, 1.0 - (corrStrengthFB - 0.6) * 1.5);
          const rawAD = featureLambda > 0
            ? allenDynesTcRaw(featureLambda, c.features.logPhononFreq ?? 300, effectiveMuStarFB, undefined, isHydrideFB)
            : 0;
          finalTc = Math.round(Math.max(0, rawAD * suppression));
        } else {
          const physOnlyTc = featureLambda > 0
            ? allenDynesTcRaw(featureLambda, c.features.logPhononFreq ?? 300, effectiveMuStarFB, undefined, isHydrideFB)
            : 0;
          finalTc = Math.round(Math.max(0, physOnlyTc));
        }

        const heuristicPressureFB = estimateFamilyPressure(c.mat.formula);
        const fbPressure = c.features.pressureGpa > 0 ? c.features.pressureGpa : heuristicPressureFB;
        const pairingMechFB = isCorrelatedFB ? "spin-fluctuation" : "phonon-mediated";

        candidates.push({
          formula: c.mat.formula,
          name: c.mat.name,
          predictedTc: finalTc,
          pressureGpa: fbPressure,
          meissnerEffect: ensembleScore > 0.5,
          zeroResistance: ensembleScore > 0.5,
          cooperPairMechanism: pairingMechFB,
          crystalStructure: c.features.layeredStructure ? "layered" : "3D",
          quantumCoherence: Math.min(1, ensembleScore * 0.8),
          stabilityScore: c.features.stabilityScore ?? 0.5,
          xgboostScore: c.xgb.score,
          neuralNetScore: null,
          ensembleScore,
          roomTempViable: finalTc >= 293 && fbPressure < 50 && !isCorrelatedFB, // 293K = ROOM_TEMP_K
          status: "theoretical",
          notes: `[XGBoost-only fallback: NN skipped]${isCorrelatedFB ? ' [correlated: Allen-Dynes unreliable]' : ''} ${c.xgb.reasoning[0] || ""}`,
          electronPhononCoupling: c.features.electronPhononLambda,
          logPhononFrequency: c.features.logPhononFreq,
          coulombPseudopotential: effectiveMuStarFB,
          pairingMechanism: pairingMechFB,
          pairingSymmetry: pairingMechFB.includes("spin") ? "d-wave" : (c.features.dWaveSymmetry ? "d-wave" : "s-wave"),
          correlationStrength: corrStrengthFB,
          dimensionality: c.features.layeredStructure ? "quasi-2D" : "3D",
          fermiSurfaceTopology: c.features.fermiSurfaceType,
          uncertaintyEstimate: isCorrelatedFB ? Math.min(1.0, 0.7 + (corrStrengthFB - 0.6) * 0.5) : 0.7,
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

const SIMPLE_TREE_MAX_DEPTH = 10;

function simpleLeafValue(residuals: number[], indices: number[]): number {
  if (indices.length === 0) return 0;
  return indices.reduce((s, i) => s + residuals[i], 0) / indices.length;
}

function buildSimpleTree(
  X: number[][], residuals: number[], indices: number[],
  depth: number, maxDepth: number, minSamples: number
): SimpleTree | number {
  const clampedMax = Math.min(maxDepth, SIMPLE_TREE_MAX_DEPTH);
  if (depth >= clampedMax || indices.length < minSamples) {
    return simpleLeafValue(residuals, indices);
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
    return simpleLeafValue(residuals, indices);
  }
  return {
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: buildSimpleTree(X, residuals, bestLeft, depth + 1, clampedMax, minSamples),
    right: buildSimpleTree(X, residuals, bestRight, depth + 1, clampedMax, minSamples),
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
  const yVariance = y.reduce((s, v) => s + (v - basePred) ** 2, 0) / n;
  const mseThreshold = Math.max(1e-8, yVariance * 0.005);
  const preds = new Array(n).fill(basePred);
  const trees: SimpleTree[] = [];
  for (let iter = 0; iter < nTrees; iter++) {
    const res = y.map((yi, i) => yi - preds[i]);
    const tree = buildSimpleTree(X, res, allIdx, 0, maxDepth, 3);
    if (typeof tree === "number") break;
    trees.push(tree);
    for (let i = 0; i < n; i++) preds[i] += lr * predictSimpleTree(tree, X[i]);
    const mse = y.reduce((s, yi, i) => s + (yi - preds[i]) ** 2, 0) / n;
    if (mse < mseThreshold) break;
  }
  return { trees, basePrediction: basePred, learningRate: lr, trainedAt: Date.now() };
}

function predictPhysicsGB(model: PhysicsGBModel, x: number[]): number {
  let pred = model.basePrediction;
  for (const tree of model.trees) pred += model.learningRate * predictSimpleTree(tree, x);
  return pred;
}

function computeTreeEnsembleUncertainty(model: PhysicsGBModel, x: number[]): number {
  const nTrees = model.trees.length;
  if (nTrees < 4) return 1.0;
  const nBlocks = Math.min(6, nTrees);
  const blockPreds: number[] = [];
  for (let b = 0; b < nBlocks; b++) {
    let pred = model.basePrediction;
    for (let i = 0; i < nTrees; i++) {
      if (i % nBlocks === b) continue;
      pred += model.learningRate * predictSimpleTree(model.trees[i], x);
    }
    blockPreds.push(pred);
  }
  const mean = blockPreds.reduce((s, v) => s + v, 0) / blockPreds.length;
  const variance = blockPreds.reduce((s, v) => s + (v - mean) ** 2, 0) / blockPreds.length;
  const jackknifeFactor = (nBlocks - 1) / nBlocks;
  return Math.sqrt(variance * jackknifeFactor);
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
    Math.log(f.numElements + 1),
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
    let eta: number;
    let uncertainty = 0.5;
    const gamma = features.avgSommerfeldGamma;
    const dos = features.dosAtEF;
    if (gamma > 0 && dos > 0.1) {
      const gamma0 = dos * 2.359;
      const lambdaFromGamma = gamma0 > 0 ? (gamma / gamma0) - 1 : 0;
      eta = Math.max(0.05, lambdaFromGamma);
      uncertainty = 0.35;
    } else if (dos > 0.1 && gamma > 0) {
      eta = 0.3 + gamma * 0.05;
      uncertainty = 0.55;
    } else {
      eta = features.hasTransitionMetal ? 0.7 : 0.4;
      uncertainty = 0.6;
    }
    const massWeight = features.maxAtomicMass > 0 ? Math.sqrt(50 / features.maxAtomicMass) : 1.0;
    const priorLambda = Math.max(0.05, Math.min(3.5, lambda > 0.1 ? lambda : eta * massWeight));
    return { value: priorLambda, uncertainty };
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
    const nEl = features.numElements;
    if (formEnergy !== null && formEnergy !== undefined) {
      const hull = formEnergy > 0 ? formEnergy * 0.5 : Math.abs(formEnergy) * 0.05;
      const uncertainty = nEl > 3 ? 0.7 : 0.3;
      return { value: Math.max(0, Math.min(1.0, hull)), uncertainty };
    }
    const enSpread = features.enSpread;
    let priorHull = 0.1 + nEl * 0.02;
    if (enSpread > 2.0) priorHull += 0.05;
    if (features.hasTransitionMetal && nEl <= 3) priorHull -= 0.03;
    const uncertainty = nEl > 3 ? 0.7 : 0.5;
    return { value: Math.max(0, Math.min(1.0, priorHull)), uncertainty };
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

  preFilter(prediction: PhysicsPrediction): { pass: boolean; marginalPass: boolean; reason: string } {
    const lambdaThresh = 0.08;
    const hullThresh = 0.3;
    const dosThresh = 0.2;
    const marginFrac = 0.05;

    if (prediction.lambda < lambdaThresh * (1 - marginFrac)) {
      return { pass: false, marginalPass: false, reason: `lambda=${prediction.lambda.toFixed(2)} < ${(lambdaThresh * (1 - marginFrac)).toFixed(3)}` };
    }
    if (prediction.hullDistance > hullThresh * (1 + marginFrac)) {
      return { pass: false, marginalPass: false, reason: `hull_dist=${prediction.hullDistance.toFixed(3)} > ${(hullThresh * (1 + marginFrac)).toFixed(3)} eV/atom` };
    }
    if (prediction.dosAtEF < dosThresh * (1 - marginFrac)) {
      return { pass: false, marginalPass: false, reason: `DOS(EF)=${prediction.dosAtEF.toFixed(2)} < ${(dosThresh * (1 - marginFrac)).toFixed(3)} states/eV` };
    }

    const marginalLambda = prediction.lambda >= lambdaThresh * (1 - marginFrac) && prediction.lambda < lambdaThresh;
    const marginalHull = prediction.hullDistance <= hullThresh * (1 + marginFrac) && prediction.hullDistance > hullThresh;
    const marginalDos = prediction.dosAtEF >= dosThresh * (1 - marginFrac) && prediction.dosAtEF < dosThresh;
    const isMarginal = marginalLambda || marginalHull || marginalDos;

    if (isMarginal) {
      const reasons: string[] = [];
      if (marginalLambda) reasons.push(`lambda=${prediction.lambda.toFixed(3)} marginal`);
      if (marginalHull) reasons.push(`hull=${prediction.hullDistance.toFixed(3)} marginal`);
      if (marginalDos) reasons.push(`DOS=${prediction.dosAtEF.toFixed(3)} marginal`);
      return { pass: true, marginalPass: true, reason: `marginal: ${reasons.join(", ")}` };
    }

    return { pass: true, marginalPass: false, reason: "passed" };
  }

  getTrainingSize(): number { return this.trainingSamples.length; }
  getLastTrainedCycle(): number { return this.lastTrainedCycle; }
}

export const physicsPredictor = new PhysicsPredictor();
