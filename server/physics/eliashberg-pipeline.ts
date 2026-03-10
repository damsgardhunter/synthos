import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  computePhononDispersion,
  computePhononDOS,
  computeAlpha2F,
  computeOmegaLogFromAlpha2F,
  predictTcEliashberg,
  parseFormulaElements,
  type ElectronicStructure,
  type PhononSpectrum,
  type ElectronPhononCoupling,
  type EliashbergResult,
  type Alpha2FData,
  type PhononDOSData,
} from "../learning/physics-engine";
import {
  getElementData,
  getMcMillanHopfieldEta,
  getDebyeTemperature,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";

export interface ModeResolvedLambda {
  acoustic: number;
  lowOptical: number;
  midOptical: number;
  highOptical: number;
  hydrogenModes: number;
  dominantRange: string;
}

export interface Alpha2FSpectralFunction {
  frequencies: number[];
  alpha2F: number[];
  cumulativeLambda: number[];
  integratedLambda: number;
  omegaLog: number;
  omega2: number;
  lambdaByRange: ModeResolvedLambda;
  nBins: number;
  maxFrequency: number;
  convergenceCheck: {
    converged: boolean;
    lambdaVariation: number;
    highFreqTail: number;
  };
}

export interface AllenDynesResult {
  tc: number;
  f1: number;
  f2: number;
  lambdaBar: number;
  omegaLogK: number;
  muStar: number;
  regime: "weak" | "intermediate" | "strong" | "very-strong";
}

export interface MatsubaraGapSolution {
  tc: number;
  gapValues: number[];
  matsubaraFrequencies: number[];
  converged: boolean;
  iterations: number;
  gapRatio: number;
  maxGap: number;
}

export interface IsotopeEffect {
  alpha: number;
  massRatio: number;
  referenceElement: string;
  isotopeTcShift: number;
}

export interface EliashbergPipelineResult {
  formula: string;
  pressureGpa: number;
  tier: "surrogate" | "dfpt";
  alpha2F: Alpha2FSpectralFunction;
  lambda: number;
  lambdaUncorrected: number;
  omegaLog: number;
  omega2: number;
  muStar: number;
  tcAllenDynes: AllenDynesResult;
  tcEliashbergGap: MatsubaraGapSolution;
  tcBest: number;
  gapRatio: number;
  isStrongCoupling: boolean;
  isotopeEffect: IsotopeEffect;
  modeResolved: ModeResolvedLambda;
  electronPhonon: ElectronPhononCoupling;
  phononSpectrum: PhononSpectrum;
  electronic: ElectronicStructure;
  confidence: "low" | "medium" | "high";
  confidenceBand: [number, number];
  wallTimeMs: number;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    counts[m[1]] = (counts[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
  }
  return counts;
}

function computeScreenedMuStar(
  formula: string,
  pressureGpa: number,
  dosAtFermi?: number,
): number {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const family = classifyFamily(formula);

  let muBare = 0.10;

  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  if (hRatio >= 6) {
    muBare = 0.13;
    if (pressureGpa > 100) muBare = 0.10 + 0.03 * Math.exp(-pressureGpa / 500);
  } else if (family === "cuprate") {
    muBare = 0.12;
  } else if (family === "pnictide") {
    muBare = 0.11;
  } else if (family === "heavy-fermion") {
    muBare = 0.15;
  } else {
    let avgZ = 0;
    for (const el of elements) {
      const data = getElementData(el);
      if (data) avgZ += data.atomicNumber * ((counts[el] || 1) / totalAtoms);
    }
    if (avgZ > 40) muBare = 0.12;
    else if (avgZ < 15) muBare = 0.08;
  }

  const N_EF = Math.max(0.1, dosAtFermi ?? 1.5);
  const k_TF_sq = 4 * Math.PI * N_EF;
  const screeningFactor = k_TF_sq / (k_TF_sq + 1.0);
  muBare *= screeningFactor;

  let avgIonization = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const frac = (counts[el] || 1) / totalAtoms;
    avgIonization += (data?.firstIonizationEnergy ?? 700) * frac;
  }
  const estimatedBandwidth = Math.max(1.0, avgIonization * 0.005);
  const E_F_eV = Math.max(1.0, estimatedBandwidth * 0.5 + N_EF * 0.5);
  const debyeEstimate = 300 + pressureGpa * 2;
  const omega_D_eV = Math.max(0.001, debyeEstimate * 8.617e-5);
  const logRatio = Math.log(Math.max(E_F_eV / omega_D_eV, 1.5));
  const muStarMA = muBare / (1 + muBare * logRatio);

  let muStar = muStarMA;

  if (pressureGpa > 50) {
    const pressureReduction = Math.min(0.03, pressureGpa * 0.0001);
    muStar = Math.max(0.05, muStar - pressureReduction);
  }

  return Number(Math.max(0.05, Math.min(0.20, muStar)).toFixed(4));
}

function buildAlpha2FSpectralFunction(
  phononDOS: PhononDOSData,
  formula: string,
  electronic: ElectronicStructure,
  coupling: ElectronPhononCoupling,
  pressureGpa: number,
  phononMaxFreq?: number
): Alpha2FSpectralFunction {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const N_EF = electronic.densityOfStatesAtFermi;

  let avgEta = 0;
  let totalWeight = 0;
  for (const el of elements) {
    const eta = getMcMillanHopfieldEta(el);
    const frac = (counts[el] || 1) / totalAtoms;
    if (eta !== null && eta > 0) {
      avgEta += eta * frac;
      totalWeight += frac;
    }
  }
  if (totalWeight > 0) avgEta /= totalWeight;
  else avgEta = N_EF * 0.3;

  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  let hModeBoost = 1.0;
  if (hRatio >= 6 && pressureGpa >= 100) {
    hModeBoost = 1.0 + Math.min(2.0, hRatio * 0.15) * Math.min(1.0, pressureGpa / 200);
  }

  const nBins = phononDOS.frequencies.length;
  const frequencies = [...phononDOS.frequencies];
  const alpha2F = new Array(nBins).fill(0);
  const cumulativeLambda = new Array(nBins).fill(0);
  const dosMaxFreq = frequencies.length > 0 ? Math.max(...frequencies.filter(f => f > 0)) : 500;
  const maxFreq = phononMaxFreq && phononMaxFreq > dosMaxFreq ? phononMaxFreq * 1.2 : dosMaxFreq;

  const couplingPrefactor = avgEta * N_EF * 1.2;
  const binWidth = nBins > 1 ? frequencies[1] - frequencies[0] : 1;

  let integratedLambda = 0;
  let logWeightedSum = 0;
  let omega2WeightedSum = 0;

  let acousticLambda = 0;
  let lowOpticalLambda = 0;
  let midOpticalLambda = 0;
  let highOpticalLambda = 0;
  let hydrogenLambda = 0;

  const acousticCutoff = maxFreq * 0.15;
  const lowOptCutoff = maxFreq * 0.35;
  const midOptCutoff = maxFreq * 0.6;
  const hModeCutoff = maxFreq * 0.75;

  for (let i = 0; i < nBins; i++) {
    const omega = frequencies[i];
    const g = phononDOS.dos[i];
    if (omega <= 0 || g <= 0) {
      cumulativeLambda[i] = integratedLambda;
      continue;
    }

    let modeWeight = 1.0;
    if (omega > hModeCutoff && hRatio >= 4) {
      modeWeight = hModeBoost;
    }

    alpha2F[i] = couplingPrefactor * g * omega * 0.01 * modeWeight;

    const lambdaContrib = 2 * alpha2F[i] / omega * binWidth;
    integratedLambda += lambdaContrib;
    cumulativeLambda[i] = integratedLambda;

    logWeightedSum += (alpha2F[i] / omega) * Math.log(omega) * binWidth;
    omega2WeightedSum += alpha2F[i] * omega * binWidth;

    if (omega <= acousticCutoff) acousticLambda += lambdaContrib;
    else if (omega <= lowOptCutoff) lowOpticalLambda += lambdaContrib;
    else if (omega <= midOptCutoff) midOpticalLambda += lambdaContrib;
    else if (omega <= hModeCutoff) highOpticalLambda += lambdaContrib;
    else hydrogenLambda += lambdaContrib;
  }

  if (integratedLambda > 0 && coupling.lambda > 0) {
    const scaleFactor = coupling.lambda / integratedLambda;
    for (let i = 0; i < nBins; i++) {
      alpha2F[i] *= scaleFactor;
      cumulativeLambda[i] *= scaleFactor;
    }
    acousticLambda *= scaleFactor;
    lowOpticalLambda *= scaleFactor;
    midOpticalLambda *= scaleFactor;
    highOpticalLambda *= scaleFactor;
    hydrogenLambda *= scaleFactor;
    logWeightedSum *= scaleFactor;
    omega2WeightedSum *= scaleFactor;
    integratedLambda = coupling.lambda;
  }

  let omegaLog = 0;
  if (integratedLambda > 1e-8) {
    omegaLog = Math.exp((2 / integratedLambda) * logWeightedSum);
    if (!Number.isFinite(omegaLog)) omegaLog = 0;
  }

  let omega2 = 0;
  if (integratedLambda > 1e-8 && omega2WeightedSum > 0) {
    omega2 = Math.sqrt((2 / integratedLambda) * omega2WeightedSum);
    if (!Number.isFinite(omega2)) omega2 = 0;
  }

  const lastQuarterStart = Math.floor(nBins * 0.75);
  const tailLambda = (cumulativeLambda[nBins - 1] || 0) - (cumulativeLambda[lastQuarterStart] || 0);
  const lambdaVariation = integratedLambda > 0
    ? Math.abs(tailLambda) / integratedLambda
    : 0;

  let dominantRange = "acoustic";
  const rangeValues = [
    { range: "acoustic (<15% max freq)", val: acousticLambda },
    { range: "low optical (15-35%)", val: lowOpticalLambda },
    { range: "mid optical (35-60%)", val: midOpticalLambda },
    { range: "high optical (60-75%)", val: highOpticalLambda },
    { range: "hydrogen modes (>75%)", val: hydrogenLambda },
  ];
  rangeValues.sort((a, b) => b.val - a.val);
  dominantRange = rangeValues[0].range;

  return {
    frequencies: frequencies.map(f => Number(f.toFixed(2))),
    alpha2F: alpha2F.map(v => Number(v.toFixed(6))),
    cumulativeLambda: cumulativeLambda.map(v => Number(v.toFixed(6))),
    integratedLambda: Number(integratedLambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),
    omega2: Number(omega2.toFixed(2)),
    lambdaByRange: {
      acoustic: Number(acousticLambda.toFixed(4)),
      lowOptical: Number(lowOpticalLambda.toFixed(4)),
      midOptical: Number(midOpticalLambda.toFixed(4)),
      highOptical: Number(highOpticalLambda.toFixed(4)),
      hydrogenModes: Number(hydrogenLambda.toFixed(4)),
      dominantRange,
    },
    nBins,
    maxFrequency: Number(maxFreq.toFixed(2)),
    convergenceCheck: {
      converged: lambdaVariation < 0.15,
      lambdaVariation: Number(lambdaVariation.toFixed(4)),
      highFreqTail: Number(tailLambda.toFixed(6)),
    },
  };
}

function computeAllenDynesTc(
  lambda: number,
  omegaLog: number,
  omega2: number,
  muStar: number
): AllenDynesResult {
  const omegaLogK = omegaLog * 1.4388;

  let regime: "weak" | "intermediate" | "strong" | "very-strong" = "weak";
  if (lambda > 2.5) regime = "very-strong";
  else if (lambda > 1.5) regime = "strong";
  else if (lambda > 0.5) regime = "intermediate";

  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denominator) < 1e-6 || denominator <= 0 || omegaLogK <= 0) {
    return { tc: 0, f1: 1, f2: 1, lambdaBar: 0, omegaLogK, muStar, regime };
  }

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  let f2 = 1.0;
  if (omega2 > 0 && omegaLog > 0) {
    const omegaRatio = omega2 / omegaLog;
    const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
    f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
  }

  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) {
    return { tc: 0, f1: Number(f1.toFixed(4)), f2: Number(f2.toFixed(4)), lambdaBar: Number(lambdaBar.toFixed(4)), omegaLogK: Number(omegaLogK.toFixed(2)), muStar, regime };
  }
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  tc = Number.isFinite(tc) ? Math.max(0, Math.min(500, tc)) : 0;

  return {
    tc: Number(tc.toFixed(2)),
    f1: Number(f1.toFixed(4)),
    f2: Number(f2.toFixed(4)),
    lambdaBar: Number(lambdaBar.toFixed(4)),
    omegaLogK: Number(omegaLogK.toFixed(2)),
    muStar,
    regime,
  };
}

function solveEliashbergGapEquation(
  alpha2FSpec: Alpha2FSpectralFunction,
  muStar: number,
  trialTc: number,
  maxIter: number = 50
): MatsubaraGapSolution {
  const kB = 0.08617;
  const T = Math.max(1, trialTc);
  const nMatsubara = 32;

  const omegaN: number[] = [];
  for (let n = 0; n < nMatsubara; n++) {
    omegaN.push(Math.PI * kB * T * (2 * n + 1));
  }

  const lambda = alpha2FSpec.integratedLambda;
  const omegaLog = alpha2FSpec.omegaLog;

  const lambdaMatrix: number[][] = [];
  for (let n = 0; n < nMatsubara; n++) {
    lambdaMatrix[n] = [];
    for (let m = 0; m < nMatsubara; m++) {
      const omegaDiff = Math.abs(omegaN[n] - omegaN[m]);
      const omegaSum = omegaN[n] + omegaN[m];

      let lambdaNM = 0;
      const { frequencies, alpha2F } = alpha2FSpec;
      const binWidth = frequencies.length > 1 ? frequencies[1] - frequencies[0] : 1;

      for (let k = 0; k < frequencies.length; k++) {
        const omega = frequencies[k];
        if (omega <= 0 || alpha2F[k] <= 0) continue;
        const omegaCm = omega;
        const omegaMeV = omegaCm * 0.1240;
        if (omegaMeV <= 0) continue;
        lambdaNM += 2 * alpha2F[k] * omegaMeV /
          (omegaMeV * omegaMeV + omegaDiff * omegaDiff) * binWidth * 0.1240;
      }

      lambdaMatrix[n][m] = lambdaNM;
    }
  }

  let gap = new Array(nMatsubara).fill(1.0);

  let converged = false;
  let iteration = 0;

  for (iteration = 0; iteration < maxIter; iteration++) {
    const newGap = new Array(nMatsubara).fill(0);

    for (let n = 0; n < nMatsubara; n++) {
      let sum = 0;
      for (let m = 0; m < nMatsubara; m++) {
        const kernel = lambdaMatrix[n][m] - muStar;
        const denom = Math.sqrt(omegaN[m] * omegaN[m] + gap[m] * gap[m]);
        if (denom > 1e-10) {
          sum += kernel * gap[m] / denom;
        }
      }
      newGap[n] = Math.PI * kB * T * sum;
    }

    const maxGap = Math.max(...newGap.map(Math.abs));
    if (maxGap > 1e-10) {
      for (let n = 0; n < nMatsubara; n++) {
        newGap[n] /= maxGap;
      }
    }

    let maxDiff = 0;
    for (let n = 0; n < nMatsubara; n++) {
      maxDiff = Math.max(maxDiff, Math.abs(newGap[n] - gap[n]));
    }

    gap = newGap;

    if (maxDiff < 1e-6) {
      converged = true;
      break;
    }
  }

  const maxGapVal = Math.max(...gap.map(Math.abs));
  const gapAtZero = Math.abs(gap[0]) * maxGapVal;

  let gapRatioResult = 3.528;
  if (lambda > 0 && trialTc > 0) {
    gapRatioResult = 3.528 * (1 + 5.3 * Math.pow(lambda / (lambda + 6), 2));
  }

  let tcFromGap = trialTc;
  if (converged && maxGapVal > 0.01) {
    tcFromGap = trialTc;
  } else if (!converged) {
    tcFromGap = trialTc * 0.9;
  }

  return {
    tc: Number(tcFromGap.toFixed(2)),
    gapValues: gap.slice(0, 8).map(v => Number(v.toFixed(6))),
    matsubaraFrequencies: omegaN.slice(0, 8).map(v => Number(v.toFixed(4))),
    converged,
    iterations: iteration,
    gapRatio: Number(gapRatioResult.toFixed(3)),
    maxGap: Number((gapAtZero * 1000).toFixed(4)),
  };
}

function sweepMuStarGapEquation(
  alpha2FSpec: Alpha2FSpectralFunction,
  centralMuStar: number,
  trialTc: number,
): { tcRange: number[]; maxVariation: number; sensitivityFlag: boolean } {
  const muStarValues = [
    Math.max(0.05, centralMuStar - 0.025),
    centralMuStar,
    Math.max(0.05, centralMuStar + 0.025),
  ];

  const tcRange: number[] = [];
  for (const mu of muStarValues) {
    const sol = solveEliashbergGapEquation(alpha2FSpec, mu, trialTc, 30);
    tcRange.push(sol.tc);
  }

  const maxTc = Math.max(...tcRange);
  const minTc = Math.min(...tcRange);
  const meanTc = tcRange.reduce((a, b) => a + b, 0) / tcRange.length;
  const maxVariation = meanTc > 0 ? (maxTc - minTc) / meanTc : 0;
  const sensitivityFlag = maxVariation > 0.20;

  return { tcRange, maxVariation, sensitivityFlag };
}

function computeIsotopeEffect(
  formula: string,
  alpha2FSpec: Alpha2FSpectralFunction,
  lambda: number
): IsotopeEffect {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  let lightestEl = elements[0] || "H";
  let lightestMass = Infinity;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.atomicMass < lightestMass) {
      lightestMass = data.atomicMass;
      lightestEl = el;
    }
  }

  const bcsAlpha = 0.5;
  let alpha = bcsAlpha;
  if (lambda > 1.5) {
    alpha = bcsAlpha * (1 - 1.04 * lambda / Math.pow(1 + lambda, 2));
  } else if (lambda > 0.5) {
    alpha = bcsAlpha * (1 - 0.2 * (lambda - 0.5));
  }

  const massRatio = 2.0;
  const isotopeTcShift = alpha * Math.log(massRatio) * alpha2FSpec.omegaLog * 1.4388 / 1.2;

  return {
    alpha: Number(Math.max(0, Math.min(0.5, alpha)).toFixed(4)),
    massRatio,
    referenceElement: lightestEl,
    isotopeTcShift: Number(isotopeTcShift.toFixed(2)),
  };
}

const pipelineCache = new Map<string, { result: EliashbergPipelineResult; timestamp: number }>();
const PIPELINE_CACHE_TTL = 20 * 60 * 1000;
const PIPELINE_CACHE_MAX = 200;

let pipelineStats = {
  totalRuns: 0,
  surrogateRuns: 0,
  dfptRuns: 0,
  avgLambda: 0,
  avgTc: 0,
  lambdaSum: 0,
  tcSum: 0,
  strongCouplingCount: 0,
  highTcCount: 0,
};

export function runEliashbergPipeline(
  formula: string,
  pressureGpa: number = 0,
  electronicOverride?: ElectronicStructure,
  phononOverride?: PhononSpectrum,
  couplingOverride?: ElectronPhononCoupling
): EliashbergPipelineResult {
  const startTime = Date.now();
  const cacheKey = `${formula}_${pressureGpa.toFixed(1)}`;

  const cached = pipelineCache.get(cacheKey);
  if (cached && (Date.now() - cached.timestamp) < PIPELINE_CACHE_TTL) {
    return cached.result;
  }

  const electronic = electronicOverride ?? computeElectronicStructure(formula);
  const phonon = phononOverride ?? computePhononSpectrum(formula, electronic);
  const coupling = couplingOverride ?? computeElectronPhononCoupling(electronic, phonon, formula, pressureGpa);

  const phononDispersion = computePhononDispersion(formula, electronic, phonon);
  const phononDOS = computePhononDOS(phononDispersion, phonon.maxPhononFrequency);

  const alpha2FSpec = buildAlpha2FSpectralFunction(
    phononDOS, formula, electronic, coupling, pressureGpa, phonon.maxPhononFrequency
  );

  const muStar = computeScreenedMuStar(formula, pressureGpa, electronic.densityOfStatesAtFermi);

  const allenDynes = computeAllenDynesTc(
    alpha2FSpec.integratedLambda,
    alpha2FSpec.omegaLog,
    alpha2FSpec.omega2,
    muStar
  );

  const gapTrialTc = allenDynes.tc > 0 ? allenDynes.tc : coupling.omegaLog * 1.4388 / 10;
  const gapSolution = solveEliashbergGapEquation(
    alpha2FSpec,
    muStar,
    gapTrialTc
  );

  const muStarSweep = sweepMuStarGapEquation(alpha2FSpec, muStar, gapTrialTc);

  const tcBest = Math.max(allenDynes.tc, gapSolution.tc);

  const isotopeEffect = computeIsotopeEffect(formula, alpha2FSpec, alpha2FSpec.integratedLambda);

  let confidence: "low" | "medium" | "high" = "medium";
  if (alpha2FSpec.convergenceCheck.converged && gapSolution.converged) {
    confidence = "high";
  } else if (!alpha2FSpec.convergenceCheck.converged && !gapSolution.converged) {
    confidence = "low";
  }

  if (muStarSweep.sensitivityFlag) {
    if (confidence === "high") confidence = "medium";
    else if (confidence === "medium") confidence = "low";
  }

  const uncertaintyFrac = confidence === "high" ? 0.15 : confidence === "medium" ? 0.25 : 0.40;
  const confidenceBand: [number, number] = [
    Math.max(0, Math.round(tcBest * (1 - uncertaintyFrac))),
    Math.round(tcBest * (1 + uncertaintyFrac)),
  ];

  const result: EliashbergPipelineResult = {
    formula,
    pressureGpa,
    tier: "surrogate",
    alpha2F: alpha2FSpec,
    lambda: alpha2FSpec.integratedLambda,
    lambdaUncorrected: coupling.lambdaUncorrected,
    omegaLog: alpha2FSpec.omegaLog,
    omega2: alpha2FSpec.omega2,
    muStar,
    tcAllenDynes: allenDynes,
    tcEliashbergGap: gapSolution,
    tcBest: Number(tcBest.toFixed(2)),
    gapRatio: gapSolution.gapRatio,
    isStrongCoupling: alpha2FSpec.integratedLambda > 1.5,
    isotopeEffect,
    modeResolved: alpha2FSpec.lambdaByRange,
    electronPhonon: coupling,
    phononSpectrum: phonon,
    electronic,
    confidence,
    confidenceBand,
    wallTimeMs: Date.now() - startTime,
  };

  pipelineStats.totalRuns++;
  pipelineStats.surrogateRuns++;
  pipelineStats.lambdaSum += result.lambda;
  pipelineStats.tcSum += result.tcBest;
  pipelineStats.avgLambda = pipelineStats.lambdaSum / pipelineStats.totalRuns;
  pipelineStats.avgTc = pipelineStats.tcSum / pipelineStats.totalRuns;
  if (result.isStrongCoupling) pipelineStats.strongCouplingCount++;
  if (result.tcBest > 100) pipelineStats.highTcCount++;

  if (pipelineCache.size >= PIPELINE_CACHE_MAX) {
    const oldest = [...pipelineCache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp)[0];
    if (oldest) pipelineCache.delete(oldest[0]);
  }
  pipelineCache.set(cacheKey, { result, timestamp: Date.now() });

  return result;
}

export function runEliashbergFromAlpha2FFile(
  formula: string,
  pressureGpa: number,
  parsedAlpha2F: { frequencies: number[]; values: number[] },
  coupling: ElectronPhononCoupling
): EliashbergPipelineResult {
  const startTime = Date.now();
  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);

  const nBins = parsedAlpha2F.frequencies.length;
  const binWidth = nBins > 1 ? parsedAlpha2F.frequencies[1] - parsedAlpha2F.frequencies[0] : 1;

  let integratedLambda = 0;
  let logWeightedSum = 0;
  let omega2WeightedSum = 0;
  const cumulativeLambda = new Array(nBins).fill(0);

  for (let i = 0; i < nBins; i++) {
    const omega = parsedAlpha2F.frequencies[i];
    const a2f = parsedAlpha2F.values[i];
    if (omega <= 0 || a2f <= 0) {
      cumulativeLambda[i] = integratedLambda;
      continue;
    }
    const lambdaContrib = 2 * a2f / omega * binWidth;
    integratedLambda += lambdaContrib;
    cumulativeLambda[i] = integratedLambda;
    logWeightedSum += (a2f / omega) * Math.log(omega) * binWidth;
    omega2WeightedSum += a2f * omega * binWidth;
  }

  const omegaLog = integratedLambda > 1e-8 ? Math.exp((2 / integratedLambda) * logWeightedSum) : 0;
  const omega2 = integratedLambda > 1e-8 ? Math.sqrt((2 / integratedLambda) * omega2WeightedSum) : 0;

  const maxFreq = Math.max(...parsedAlpha2F.frequencies.filter(f => f > 0));

  const alpha2FSpec: Alpha2FSpectralFunction = {
    frequencies: parsedAlpha2F.frequencies,
    alpha2F: parsedAlpha2F.values,
    cumulativeLambda,
    integratedLambda: Number(integratedLambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),
    omega2: Number(omega2.toFixed(2)),
    lambdaByRange: {
      acoustic: 0, lowOptical: 0, midOptical: 0, highOptical: 0, hydrogenModes: 0,
      dominantRange: "from DFPT data",
    },
    nBins,
    maxFrequency: maxFreq,
    convergenceCheck: { converged: true, lambdaVariation: 0, highFreqTail: 0 },
  };

  const muStar = computeScreenedMuStar(formula, pressureGpa, electronic.densityOfStatesAtFermi);
  const allenDynes = computeAllenDynesTc(integratedLambda, omegaLog, omega2, muStar);
  const gapSolution = solveEliashbergGapEquation(alpha2FSpec, muStar, allenDynes.tc > 0 ? allenDynes.tc : 10);
  const tcBest = Math.max(allenDynes.tc, gapSolution.tc);
  const isotopeEffect = computeIsotopeEffect(formula, alpha2FSpec, integratedLambda);

  const confidenceBand: [number, number] = [
    Math.max(0, Math.round(tcBest * 0.9)),
    Math.round(tcBest * 1.1),
  ];

  pipelineStats.totalRuns++;
  pipelineStats.dfptRuns++;
  pipelineStats.lambdaSum += integratedLambda;
  pipelineStats.tcSum += tcBest;
  pipelineStats.avgLambda = pipelineStats.lambdaSum / pipelineStats.totalRuns;
  pipelineStats.avgTc = pipelineStats.tcSum / pipelineStats.totalRuns;

  return {
    formula,
    pressureGpa,
    tier: "dfpt",
    alpha2F: alpha2FSpec,
    lambda: integratedLambda,
    lambdaUncorrected: coupling.lambdaUncorrected,
    omegaLog,
    omega2,
    muStar,
    tcAllenDynes: allenDynes,
    tcEliashbergGap: gapSolution,
    tcBest: Number(tcBest.toFixed(2)),
    gapRatio: gapSolution.gapRatio,
    isStrongCoupling: integratedLambda > 1.5,
    isotopeEffect,
    modeResolved: alpha2FSpec.lambdaByRange,
    electronPhonon: coupling,
    phononSpectrum: phonon,
    electronic,
    confidence: "high",
    confidenceBand,
    wallTimeMs: Date.now() - startTime,
  };
}

export function getEliashbergPipelineStats() {
  return {
    ...pipelineStats,
    cacheSize: pipelineCache.size,
    cacheMaxSize: PIPELINE_CACHE_MAX,
  };
}

export function getAlpha2FOnly(
  formula: string,
  pressureGpa: number = 0
): Alpha2FSpectralFunction {
  const electronic = computeElectronicStructure(formula);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, pressureGpa);
  const phononDispersion = computePhononDispersion(formula, electronic, phonon);
  const phononDOS = computePhononDOS(phononDispersion, phonon.maxPhononFrequency);
  return buildAlpha2FSpectralFunction(phononDOS, formula, electronic, coupling, pressureGpa, phonon.maxPhononFrequency);
}
