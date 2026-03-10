import { parseFormulaElements, computeElectronicStructure, computePhononSpectrum } from "./physics-engine";
import { normalizeFormula, isValidFormula } from "./utils";
import { getElementData, getStonerParameter } from "./elemental-data";
import { runXTBOptimization, runXTBPhononCheck, runXTBAnharmonicProbe, runXTBMDSampling, type PhononStability, type AnharmonicProbeResult, type MDSamplingRawResult } from "../dft/qe-dft-engine";
import { extractFeatures } from "./ml-predictor";
import { gbPredictWithUncertainty } from "./gradient-boost";

export type DopingCharacter = "electron" | "hole" | "isovalent" | "vacancy-hole" | "interstitial-electron";

export interface DopingSpec {
  type: "substitutional" | "vacancy" | "interstitial";
  base: string;
  dopant?: string;
  site?: string;
  fraction: number;
  resultFormula: string;
  supercellSize: number;
  rationale: string;
  dopingCharacter: DopingCharacter;
  valenceChange: number;
  carrierDensity: number;
  relaxation?: RelaxationMetrics;
}

export interface RelaxationMetrics {
  converged: boolean;
  latticeStrain: number;
  bondVariance: number;
  meanDisplacement: number;
  maxDisplacement: number;
  volumeChange: number;
  energyPerAtom: number;
  wallTimeMs: number;
  phononAnalysis?: HessianPhononAnalysis | null;
}

export interface HessianPhononAnalysis {
  frequencies_cm1: number[];
  frequencies_THz: number[];
  softModeCount: number;
  softModeFrequencies_THz: number[];
  imaginaryModeCount: number;
  imaginaryFrequencies_THz: number[];
  hasImaginaryModes: boolean;
  hasLatticeInstability: boolean;
  avgFrequency_THz: number;
  avgFrequency_cm1: number;
  maxFrequency_THz: number;
  lowestFrequency_THz: number;
  latticeClassification: "stiff" | "moderate" | "soft";
  zeroPointEnergy: number | null;
  scRelevance: {
    softPhononScore: number;
    instabilityScore: number;
    couplingScore: number;
    overallPhononSCScore: number;
    interpretation: string;
  };
}

export interface AnharmonicAnalysis {
  displacements: number[];
  energies: number[];
  forces: number[];
  quadraticFitCoeffs: [number, number, number];
  residualRMS: number;
  anharmonicityScore: number;
  isAnharmonic: boolean;
  cubicContribution: number;
  quarticContribution: number;
  energyCurveType: "harmonic" | "weakly-anharmonic" | "strongly-anharmonic";
  scRelevance: string;
  source: "xtb-displacement" | "physics-engine-estimate";
}

export interface MDSamplingResult {
  temperature: number;
  totalSteps: number;
  timeStepFs: number;
  totalTimePs: number;
  meanSquareDisplacement: number;
  msdPerElement: Record<string, number>;
  velocityAutocorrelation: number[];
  vacDecayTime: number;
  rmsFluctuation: number;
  maxDisplacement: number;
  fluctuationClassification: "rigid" | "moderate" | "large" | "extreme";
  phononDensityProxy: number;
  scRelevance: string;
  source: "xtb-md" | "physics-engine-estimate";
}

export interface DebyeTemperatureResult {
  debyeTemperature: number;
  fromPhononAvg: boolean;
  avgFrequency_THz: number;
  avgFrequency_cm1: number;
  classification: "very-low" | "low" | "moderate" | "high" | "very-high";
  electronPhononCouplingHint: string;
  scRelevance: number;
}

const CM1_TO_THZ = 0.02998;
const THZ_TO_KELVIN = 47.9924;
const SOFT_PHONON_THRESHOLD_THZ = 1.0;
const MODERATE_PHONON_THRESHOLD_THZ = 5.0;

export async function analyzeHessianPhonons(formula: string): Promise<HessianPhononAnalysis | null> {
  let stability: PhononStability | null = null;
  try {
    stability = await runXTBPhononCheck(formula);
  } catch (e) {
    console.log(`[Doping] xTB Hessian failed for ${formula}, using physics-engine fallback`);
  }

  if (stability && stability.frequencies.length > 0) {
    return processPhononStability(stability);
  }

  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    if (!phonon) return null;

    const maxFreq = phonon.maxPhononFrequency;
    const logAvg = phonon.logAverageFrequency;
    const nModes = Math.max(6, Math.min(20, parseFormulaElements(formula).length * 3));
    const syntheticFreqs: number[] = [];
    for (let i = 0; i < nModes; i++) {
      const frac = (i + 1) / nModes;
      syntheticFreqs.push(maxFreq * frac * (0.7 + 0.6 * ((i * 7 + 3) % 11) / 11));
    }
    if (phonon.hasImaginaryModes) {
      syntheticFreqs[0] = -Math.abs(logAvg * 0.1);
    }
    if (phonon.softModePresent) {
      const softIdx = phonon.hasImaginaryModes ? 1 : 0;
      syntheticFreqs[softIdx] = Math.min(syntheticFreqs[softIdx], logAvg * 0.05);
    }
    syntheticFreqs.sort((a, b) => a - b);

    const syntheticStability: PhononStability = {
      hasImaginaryModes: phonon.hasImaginaryModes,
      imaginaryModeCount: phonon.hasImaginaryModes ? 1 : 0,
      lowestFrequency: syntheticFreqs[0],
      frequencies: syntheticFreqs,
      zeroPointEnergy: null,
    };
    const result = processPhononStability(syntheticStability);
    (result as any).source = "physics-engine-estimate";
    return result;
  } catch {
    return null;
  }
}

export function processPhononStability(stability: PhononStability): HessianPhononAnalysis {
  const freqs_cm1 = stability.frequencies;
  const freqs_THz = freqs_cm1.map(f => f * CM1_TO_THZ);

  const positiveFreqs_THz = freqs_THz.filter(f => f > 0);
  const softModes = positiveFreqs_THz.filter(f => f < SOFT_PHONON_THRESHOLD_THZ);
  const imaginaryModes = freqs_THz.filter(f => f < 0);

  const avgFreq_THz = positiveFreqs_THz.length > 0
    ? positiveFreqs_THz.reduce((s, f) => s + f, 0) / positiveFreqs_THz.length
    : 0;

  const maxFreq_THz = positiveFreqs_THz.length > 0 ? Math.max(...positiveFreqs_THz) : 0;
  const lowestFreq_THz = freqs_THz.length > 0 ? Math.min(...freqs_THz) : 0;

  let latticeClassification: "stiff" | "moderate" | "soft";
  if (avgFreq_THz > MODERATE_PHONON_THRESHOLD_THZ) {
    latticeClassification = "stiff";
  } else if (avgFreq_THz >= SOFT_PHONON_THRESHOLD_THZ) {
    latticeClassification = "moderate";
  } else {
    latticeClassification = "soft";
  }

  const softFraction = freqs_THz.length > 0 ? softModes.length / freqs_THz.length : 0;
  const softPhononScore = Math.min(1.0, softFraction * 3.0);

  const imagFraction = freqs_THz.length > 0 ? imaginaryModes.length / freqs_THz.length : 0;
  const instabilityScore = imaginaryModes.length > 0
    ? Math.min(1.0, 0.5 + imagFraction * 2.0)
    : 0;

  let couplingScore = 0;
  if (latticeClassification === "moderate") couplingScore = 0.7;
  else if (latticeClassification === "soft") couplingScore = 1.0;
  else couplingScore = 0.2;
  if (softModes.length > 0) couplingScore = Math.min(1.0, couplingScore + 0.2);

  const overallPhononSCScore = 0.35 * softPhononScore + 0.30 * instabilityScore + 0.35 * couplingScore;

  let interpretation: string;
  if (imaginaryModes.length > 0 && softModes.length > 2) {
    interpretation = `Lattice shows ${imaginaryModes.length} imaginary mode(s) and ${softModes.length} soft mode(s) — strong dynamic instability suggesting structural transition near SC`;
  } else if (imaginaryModes.length > 0) {
    interpretation = `${imaginaryModes.length} imaginary mode(s) detected — lattice wants to distort, possible SC-relevant instability`;
  } else if (softModes.length > 2) {
    interpretation = `${softModes.length} soft phonon mode(s) below 1 THz — lattice near structural transition, favorable for electron-phonon coupling`;
  } else if (latticeClassification === "moderate") {
    interpretation = "Moderate phonon spectrum — good electron-phonon coupling range for conventional SC";
  } else if (latticeClassification === "soft") {
    interpretation = "Soft lattice dynamics — strong phonon-mediated coupling potential";
  } else {
    interpretation = "Stiff lattice — weaker conventional phonon-mediated SC coupling expected";
  }

  return {
    frequencies_cm1: freqs_cm1,
    frequencies_THz: freqs_THz.map(f => Math.round(f * 1000) / 1000),
    softModeCount: softModes.length,
    softModeFrequencies_THz: softModes.map(f => Math.round(f * 1000) / 1000),
    imaginaryModeCount: imaginaryModes.length,
    imaginaryFrequencies_THz: imaginaryModes.map(f => Math.round(f * 1000) / 1000),
    hasImaginaryModes: imaginaryModes.length > 0,
    hasLatticeInstability: imaginaryModes.length > 0 || softModes.length > positiveFreqs_THz.length * 0.4,
    avgFrequency_THz: Math.round(avgFreq_THz * 1000) / 1000,
    avgFrequency_cm1: Math.round(avgFreq_THz / CM1_TO_THZ * 100) / 100,
    maxFrequency_THz: Math.round(maxFreq_THz * 1000) / 1000,
    lowestFrequency_THz: Math.round(lowestFreq_THz * 1000) / 1000,
    latticeClassification,
    zeroPointEnergy: stability.zeroPointEnergy,
    scRelevance: {
      softPhononScore: Math.round(softPhononScore * 1000) / 1000,
      instabilityScore: Math.round(instabilityScore * 1000) / 1000,
      couplingScore: Math.round(couplingScore * 1000) / 1000,
      overallPhononSCScore: Math.round(overallPhononSCScore * 1000) / 1000,
      interpretation,
    },
  };
}

export async function detectAnharmonicVibrations(formula: string): Promise<AnharmonicAnalysis | null> {
  let xtbResult: AnharmonicProbeResult | null = null;
  try {
    xtbResult = await runXTBAnharmonicProbe(formula);
  } catch (e) {
    console.log(`[Doping] xTB anharmonic probe failed for ${formula}, using physics-engine fallback`);
  }

  if (xtbResult && xtbResult.displacements.length >= 3) {
    return processAnharmonicResult(xtbResult);
  }

  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    if (!phonon) return null;

    const elements = parseFormulaElements(formula);
    const counts = parseFormulaCounts(formula);
    const totalAtoms = getTotalAtoms(counts);
    const avgMass = elements.reduce((s, el) => {
      const d = getElementData(el);
      return s + (d?.atomicMass ?? 40) * (counts[el] ?? 1);
    }, 0) / Math.max(1, totalAtoms);

    const displacements = [-0.10, -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.08, 0.10];
    const k_eff = phonon.maxPhononFrequency * 0.01;
    const anharmonicIndex = phonon.anharmonicityIndex;

    const energies = displacements.map(d => {
      const harmonic = 0.5 * k_eff * d * d;
      const cubic = anharmonicIndex * 0.3 * k_eff * d * d * d;
      const quartic = anharmonicIndex * 0.15 * k_eff * d * d * d * d;
      return harmonic + cubic + quartic;
    });

    const forces = displacements.map(d => {
      const linear = -k_eff * d;
      const cubicF = -3 * anharmonicIndex * 0.3 * k_eff * d * d;
      const quarticF = -4 * anharmonicIndex * 0.15 * k_eff * d * d * d;
      return linear + cubicF + quarticF;
    });

    const syntheticResult: AnharmonicProbeResult = {
      displacements,
      energies,
      forces,
      source: "physics-engine-estimate",
    };
    return processAnharmonicResult(syntheticResult);
  } catch {
    return null;
  }
}

function processAnharmonicResult(raw: AnharmonicProbeResult): AnharmonicAnalysis {
  const { displacements, energies, forces } = raw;
  const n = displacements.length;

  let sumX2 = 0, sumX4 = 0, sumX2Y = 0, sumY = 0;
  for (let i = 0; i < n; i++) {
    const x2 = displacements[i] * displacements[i];
    sumX2 += x2;
    sumX4 += x2 * x2;
    sumY += energies[i];
    sumX2Y += x2 * energies[i];
  }
  const a2 = n > 0 ? (n * sumX2Y - sumX2 * sumY) / Math.max(1e-15, n * sumX4 - sumX2 * sumX2) : 0;
  const a0 = (sumY - a2 * sumX2) / Math.max(1, n);
  const quadraticFitCoeffs: [number, number, number] = [a0, 0, a2];

  let residualSumSq = 0;
  for (let i = 0; i < n; i++) {
    const predicted = a0 + a2 * displacements[i] * displacements[i];
    const diff = energies[i] - predicted;
    residualSumSq += diff * diff;
  }
  const residualRMS = Math.sqrt(residualSumSq / Math.max(1, n));

  const maxE = Math.max(...energies.map(Math.abs));
  const anharmonicityScore = maxE > 0 ? Math.min(1.0, residualRMS / (maxE * 0.1 + 1e-10)) : 0;

  let cubicContribution = 0;
  let quarticContribution = 0;
  if (n >= 5) {
    for (let i = 0; i < n; i++) {
      const x = displacements[i];
      const eActual = energies[i];
      const eQuad = a0 + a2 * x * x;
      const residual = eActual - eQuad;
      cubicContribution += Math.abs(residual * x) / (Math.abs(x * x * x) + 1e-10);
      quarticContribution += Math.abs(residual) / (x * x * x * x + 1e-10);
    }
    cubicContribution = Math.min(1.0, cubicContribution / n * 0.5);
    quarticContribution = Math.min(1.0, quarticContribution / n * 0.01);
  }

  const isAnharmonic = anharmonicityScore > 0.15;
  let energyCurveType: "harmonic" | "weakly-anharmonic" | "strongly-anharmonic";
  if (anharmonicityScore < 0.1) energyCurveType = "harmonic";
  else if (anharmonicityScore < 0.4) energyCurveType = "weakly-anharmonic";
  else energyCurveType = "strongly-anharmonic";

  let scRelevance: string;
  if (energyCurveType === "strongly-anharmonic") {
    scRelevance = "Strong anharmonicity detected -- potential for unconventional phonon-mediated SC or lattice instability-driven pairing";
  } else if (energyCurveType === "weakly-anharmonic") {
    scRelevance = "Weak anharmonicity -- modest deviation from harmonic behavior, possible phonon renormalization effects";
  } else {
    scRelevance = "Harmonic vibrations -- conventional phonon spectrum expected, standard BCS coupling";
  }

  return {
    displacements,
    energies: energies.map(e => Math.round(e * 1e8) / 1e8),
    forces: forces.map(f => Math.round(f * 1e8) / 1e8),
    quadraticFitCoeffs: quadraticFitCoeffs.map(c => Math.round(c * 1e8) / 1e8) as [number, number, number],
    residualRMS: Math.round(residualRMS * 1e8) / 1e8,
    anharmonicityScore: Math.round(anharmonicityScore * 1000) / 1000,
    isAnharmonic,
    cubicContribution: Math.round(cubicContribution * 1000) / 1000,
    quarticContribution: Math.round(quarticContribution * 1000) / 1000,
    energyCurveType,
    scRelevance,
    source: (raw as any).source ?? "xtb-displacement",
  };
}

export async function runMDSampling(formula: string, temperatureK: number = 300): Promise<MDSamplingResult | null> {
  let xtbResult: MDSamplingRawResult | null = null;
  try {
    xtbResult = await runXTBMDSampling(formula, temperatureK);
  } catch (e) {
    console.log(`[Doping] xTB MD sampling failed for ${formula}, using physics-engine fallback`);
  }

  if (xtbResult && xtbResult.positions.length > 0) {
    return processMDResult(xtbResult, formula);
  }

  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    if (!phonon) return null;

    const elements = parseFormulaElements(formula);
    const counts = parseFormulaCounts(formula);
    const totalAtoms = getTotalAtoms(counts);

    const kB = 8.617e-5;
    const thermalEnergy = kB * temperatureK;
    const avgFreq_THz = phonon.logAverageFrequency * CM1_TO_THZ;
    const debyeT = phonon.debyeTemperature;
    const thetaRatio = debyeT > 0 ? temperatureK / debyeT : 1.0;

    const baseMSD = thermalEnergy / (Math.max(0.01, avgFreq_THz) * 0.5) * 0.01;
    const msd = baseMSD * (1 + thetaRatio * 0.5);

    const msdPerElement: Record<string, number> = {};
    for (const el of elements) {
      const data = getElementData(el);
      const mass = data?.atomicMass ?? 40;
      msdPerElement[el] = msd * (40 / mass);
    }

    const totalSteps = 500;
    const timeStepFs = 1.0;
    const vacLength = 50;
    const vac: number[] = [];
    const vacDecayRate = avgFreq_THz * 0.2;
    for (let i = 0; i < vacLength; i++) {
      vac.push(Math.exp(-i * vacDecayRate * 0.02) * Math.cos(2 * Math.PI * avgFreq_THz * 0.001 * i));
    }

    const vacDecayTime = vacDecayRate > 0 ? 1.0 / (vacDecayRate * 0.02) : 50;
    const rmsFluctuation = Math.sqrt(msd);
    const maxDisplacement = rmsFluctuation * 2.5;

    let fluctuationClassification: "rigid" | "moderate" | "large" | "extreme";
    if (rmsFluctuation < 0.05) fluctuationClassification = "rigid";
    else if (rmsFluctuation < 0.15) fluctuationClassification = "moderate";
    else if (rmsFluctuation < 0.35) fluctuationClassification = "large";
    else fluctuationClassification = "extreme";

    const phononDensityProxy = totalAtoms * avgFreq_THz * (phonon.softModeScore + 0.1);

    let scRelevance: string;
    if (fluctuationClassification === "large" || fluctuationClassification === "extreme") {
      scRelevance = "Large thermal fluctuations -- strong lattice dynamics may enhance or suppress SC depending on pairing mechanism";
    } else if (fluctuationClassification === "moderate") {
      scRelevance = "Moderate fluctuations -- typical for materials with conventional phonon-mediated SC";
    } else {
      scRelevance = "Rigid lattice -- weak atomic motion suggests low electron-phonon coupling";
    }

    return {
      temperature: temperatureK,
      totalSteps,
      timeStepFs,
      totalTimePs: totalSteps * timeStepFs * 0.001,
      meanSquareDisplacement: Math.round(msd * 1e6) / 1e6,
      msdPerElement,
      velocityAutocorrelation: vac.map(v => Math.round(v * 1000) / 1000),
      vacDecayTime: Math.round(vacDecayTime * 100) / 100,
      rmsFluctuation: Math.round(rmsFluctuation * 1e4) / 1e4,
      maxDisplacement: Math.round(maxDisplacement * 1e4) / 1e4,
      fluctuationClassification,
      phononDensityProxy: Math.round(phononDensityProxy * 100) / 100,
      scRelevance,
      source: "physics-engine-estimate",
    };
  } catch {
    return null;
  }
}

function processMDResult(raw: MDSamplingRawResult, formula: string): MDSamplingResult {
  const { positions, velocities, temperature, totalSteps, timeStepFs } = raw;
  const nFrames = positions.length;
  const nAtoms = nFrames > 0 ? positions[0].length : 0;

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const elementList: string[] = [];
  for (const el of elements) {
    for (let i = 0; i < (counts[el] ?? 1); i++) elementList.push(el);
  }

  const refPos = positions[0] ?? [];
  let totalMSD = 0;
  const msdByElement: Record<string, { sum: number; count: number }> = {};

  for (let f = 1; f < nFrames; f++) {
    for (let a = 0; a < nAtoms; a++) {
      const dx = positions[f][a][0] - refPos[a][0];
      const dy = positions[f][a][1] - refPos[a][1];
      const dz = positions[f][a][2] - refPos[a][2];
      const d2 = dx * dx + dy * dy + dz * dz;
      totalMSD += d2;
      const elName = elementList[a % elementList.length] ?? "X";
      if (!msdByElement[elName]) msdByElement[elName] = { sum: 0, count: 0 };
      msdByElement[elName].sum += d2;
      msdByElement[elName].count += 1;
    }
  }
  const meanMSD = nAtoms > 0 && nFrames > 1 ? totalMSD / (nAtoms * (nFrames - 1)) : 0;
  const msdPerElement: Record<string, number> = {};
  for (const [el, data] of Object.entries(msdByElement)) {
    msdPerElement[el] = data.count > 0 ? data.sum / data.count : 0;
  }

  const vacLength = Math.min(50, nFrames - 1);
  const vac: number[] = [];
  if (velocities && velocities.length > 0 && nAtoms > 0) {
    for (let tau = 0; tau < vacLength; tau++) {
      let corr = 0;
      let norm = 0;
      for (let f = 0; f < nFrames - tau; f++) {
        for (let a = 0; a < nAtoms; a++) {
          const v0 = velocities[f]?.[a] ?? [0, 0, 0];
          const vt = velocities[f + tau]?.[a] ?? [0, 0, 0];
          corr += v0[0] * vt[0] + v0[1] * vt[1] + v0[2] * vt[2];
          if (tau === 0) norm += v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
        }
      }
      if (tau === 0 && norm > 0) {
        vac.push(1.0);
      } else {
        vac.push(norm > 0 ? corr / (norm * (nFrames - tau)) * (nFrames) : 0);
      }
    }
  }

  let vacDecayTime = vacLength * timeStepFs * 0.001;
  for (let i = 0; i < vac.length; i++) {
    if (vac[i] < 0.37) {
      vacDecayTime = i * timeStepFs * 0.001;
      break;
    }
  }

  const rmsFluctuation = Math.sqrt(meanMSD);
  let maxDisp = 0;
  for (let f = 1; f < nFrames; f++) {
    for (let a = 0; a < nAtoms; a++) {
      const dx = positions[f][a][0] - refPos[a][0];
      const dy = positions[f][a][1] - refPos[a][1];
      const dz = positions[f][a][2] - refPos[a][2];
      maxDisp = Math.max(maxDisp, Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
  }

  let fluctuationClassification: "rigid" | "moderate" | "large" | "extreme";
  if (rmsFluctuation < 0.05) fluctuationClassification = "rigid";
  else if (rmsFluctuation < 0.15) fluctuationClassification = "moderate";
  else if (rmsFluctuation < 0.35) fluctuationClassification = "large";
  else fluctuationClassification = "extreme";

  const phononDensityProxy = nAtoms * rmsFluctuation * 10;

  let scRelevance: string;
  if (fluctuationClassification === "large" || fluctuationClassification === "extreme") {
    scRelevance = "Large thermal fluctuations -- strong lattice dynamics may enhance or suppress SC depending on pairing mechanism";
  } else if (fluctuationClassification === "moderate") {
    scRelevance = "Moderate fluctuations -- typical for materials with conventional phonon-mediated SC";
  } else {
    scRelevance = "Rigid lattice -- weak atomic motion suggests low electron-phonon coupling";
  }

  return {
    temperature,
    totalSteps,
    timeStepFs,
    totalTimePs: Math.round(totalSteps * timeStepFs * 0.001 * 1000) / 1000,
    meanSquareDisplacement: Math.round(meanMSD * 1e6) / 1e6,
    msdPerElement,
    velocityAutocorrelation: vac.map(v => Math.round(v * 1000) / 1000),
    vacDecayTime: Math.round(vacDecayTime * 100) / 100,
    rmsFluctuation: Math.round(rmsFluctuation * 1e4) / 1e4,
    maxDisplacement: Math.round(maxDisp * 1e4) / 1e4,
    fluctuationClassification,
    phononDensityProxy: Math.round(phononDensityProxy * 100) / 100,
    scRelevance,
    source: "xtb-md",
  };
}

export function computeDebyeTemp(formula: string, phononAnalysis?: HessianPhononAnalysis | null): DebyeTemperatureResult | null {
  let avgFreq_THz: number;
  let avgFreq_cm1: number;
  let fromPhonon = false;

  if (phononAnalysis && phononAnalysis.avgFrequency_THz > 0) {
    avgFreq_THz = phononAnalysis.avgFrequency_THz;
    avgFreq_cm1 = phononAnalysis.avgFrequency_cm1;
    fromPhonon = true;
  } else {
    try {
      const electronic = computeElectronicStructure(formula);
      const phonon = computePhononSpectrum(formula, electronic);
      if (!phonon) return null;
      avgFreq_cm1 = phonon.logAverageFrequency;
      avgFreq_THz = avgFreq_cm1 * CM1_TO_THZ;
    } catch {
      return null;
    }
  }

  const debyeTemperature = Math.round(avgFreq_THz * THZ_TO_KELVIN);

  let classification: "very-low" | "low" | "moderate" | "high" | "very-high";
  if (debyeTemperature < 100) classification = "very-low";
  else if (debyeTemperature < 300) classification = "low";
  else if (debyeTemperature < 600) classification = "moderate";
  else if (debyeTemperature < 1200) classification = "high";
  else classification = "very-high";

  let electronPhononCouplingHint: string;
  if (classification === "very-low" || classification === "low") {
    electronPhononCouplingHint = "Low Debye temperature implies soft lattice -- stronger electron-phonon coupling expected (higher lambda)";
  } else if (classification === "moderate") {
    electronPhononCouplingHint = "Moderate Debye temperature -- balanced phonon spectrum for conventional BCS pairing";
  } else {
    electronPhononCouplingHint = "High Debye temperature implies stiff lattice -- weaker conventional coupling but higher phonon frequencies";
  }

  let scRelevance: number;
  if (debyeTemperature < 200) scRelevance = 0.8;
  else if (debyeTemperature < 400) scRelevance = 0.6;
  else if (debyeTemperature < 800) scRelevance = 0.4;
  else scRelevance = 0.2;

  return {
    debyeTemperature,
    fromPhononAvg: fromPhonon,
    avgFrequency_THz: Math.round(avgFreq_THz * 1000) / 1000,
    avgFrequency_cm1: Math.round(avgFreq_cm1 * 100) / 100,
    classification,
    electronPhononCouplingHint,
    scRelevance: Math.round(scRelevance * 1000) / 1000,
  };
}

export interface DopingResult {
  baseFormula: string;
  variants: DopingSpec[];
  totalGenerated: number;
  validGenerated: number;
  wallTimeMs: number;
}

export interface DopingEngineStats {
  totalBaseMaterials: number;
  totalVariantsGenerated: number;
  substitutionalCount: number;
  vacancyCount: number;
  interstitialCount: number;
  validVariants: number;
  electronDopedCount: number;
  holeDopedCount: number;
  relaxationsCompleted: number;
  avgLatticeStrain: number;
  recentResults: Array<{ base: string; variants: number; timestamp: number }>;
}

const stats: DopingEngineStats = {
  totalBaseMaterials: 0,
  totalVariantsGenerated: 0,
  substitutionalCount: 0,
  vacancyCount: 0,
  interstitialCount: 0,
  validVariants: 0,
  electronDopedCount: 0,
  holeDopedCount: 0,
  relaxationsCompleted: 0,
  avgLatticeStrain: 0,
  recentResults: [],
};

let totalStrainSum = 0;

const MAX_RECENT = 100;

const COMMON_OXIDATION_STATES: Record<string, number> = {
  H: 1, Li: 1, Na: 1, K: 1, Rb: 1, Cs: 1,
  Be: 2, Mg: 2, Ca: 2, Sr: 2, Ba: 2,
  Sc: 3, Y: 3, La: 3, Ce: 3, Pr: 3, Nd: 3, Gd: 3,
  Ti: 4, Zr: 4, Hf: 4,
  V: 5, Nb: 5, Ta: 5,
  Cr: 3, Mo: 6, W: 6,
  Mn: 2, Fe: 3, Co: 3, Ni: 2, Cu: 2, Zn: 2,
  Ru: 4, Rh: 3, Pd: 2, Ir: 4, Pt: 4,
  Al: 3, Ga: 3, In: 3, Tl: 1,
  B: 3, C: 4, Si: 4, Ge: 4, Sn: 4, Pb: 2,
  N: -3, P: -3, As: -3, Sb: -3, Bi: 3,
  O: -2, S: -2, Se: -2, Te: -2,
  F: -1, Cl: -1, Br: -1, I: -1,
  Re: 7, Os: 4,
};

const ELECTRON_DOPING_PAIRS: Array<{ from: string; to: string }> = [
  { from: "O", to: "F" },
  { from: "Fe", to: "Co" },
  { from: "Ti", to: "Nb" },
  { from: "Ti", to: "V" },
  { from: "Cu", to: "Zn" },
  { from: "Ni", to: "Cu" },
  { from: "Mn", to: "Fe" },
  { from: "Cr", to: "Mn" },
  { from: "N", to: "O" },
  { from: "S", to: "Cl" },
  { from: "Se", to: "Br" },
  { from: "Zr", to: "Nb" },
  { from: "Hf", to: "Ta" },
  { from: "Mo", to: "W" },
  { from: "Al", to: "Si" },
  { from: "Ga", to: "Ge" },
  { from: "Sn", to: "Sb" },
  { from: "In", to: "Sn" },
  { from: "Ca", to: "Sc" },
  { from: "Sr", to: "Y" },
];

const HOLE_DOPING_PAIRS: Array<{ from: string; to: string }> = [
  { from: "La", to: "Sr" },
  { from: "Ba", to: "K" },
  { from: "Y", to: "Ca" },
  { from: "La", to: "Ba" },
  { from: "Ce", to: "La" },
  { from: "Sr", to: "K" },
  { from: "Ca", to: "Na" },
  { from: "Fe", to: "Mn" },
  { from: "Nb", to: "Ti" },
  { from: "Co", to: "Fe" },
  { from: "Cu", to: "Ni" },
  { from: "Bi", to: "Pb" },
  { from: "Pb", to: "Tl" },
  { from: "Sn", to: "In" },
  { from: "Ga", to: "Zn" },
  { from: "Al", to: "Mg" },
  { from: "Nd", to: "Sr" },
  { from: "Gd", to: "Ca" },
  { from: "Ta", to: "Zr" },
  { from: "V", to: "Ti" },
];

function getOxidationState(el: string): number {
  return COMMON_OXIDATION_STATES[el] ?? 0;
}

function classifyDopingCharacter(site: string, dopant: string, type: "substitutional" | "vacancy" | "interstitial"): { character: DopingCharacter; valenceChange: number } {
  if (type === "vacancy") {
    const siteOx = getOxidationState(site);
    return { character: "vacancy-hole", valenceChange: -siteOx };
  }
  if (type === "interstitial") {
    const dopantOx = getOxidationState(dopant);
    return { character: "interstitial-electron", valenceChange: dopantOx };
  }

  const siteOx = getOxidationState(site);
  const dopantOx = getOxidationState(dopant);
  const delta = dopantOx - siteOx;

  if (delta > 0) return { character: "electron", valenceChange: delta };
  if (delta < 0) return { character: "hole", valenceChange: delta };
  return { character: "isovalent", valenceChange: 0 };
}

function estimateUnitCellVolume(counts: Record<string, number>): number {
  const totalAtoms = getTotalAtoms(counts);
  let avgRadius = 0;
  let totalWeight = 0;
  for (const [el, n] of Object.entries(counts)) {
    const data = getElementData(el);
    const r = data?.atomicRadius ?? 130;
    avgRadius += r * n;
    totalWeight += n;
  }
  avgRadius = totalWeight > 0 ? avgRadius / totalWeight : 130;
  const latticeParam = avgRadius * 2.8 / 100;
  const volumePerAtom = latticeParam ** 3;
  return volumePerAtom * totalAtoms;
}

function computeCarrierDensity(valenceChange: number, nDopedAtoms: number, cellVolumeNm3: number): number {
  if (cellVolumeNm3 <= 0) return 0;
  const totalChargeChange = Math.abs(valenceChange) * nDopedAtoms;
  const volumeCm3 = cellVolumeNm3 * 1e-21;
  return totalChargeChange / volumeCm3;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (num > 0) counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function countsToFormula(counts: Record<string, number>): string {
  const sorted = Object.entries(counts)
    .filter(([, n]) => n > 0.001)
    .sort(([a], [b]) => a.localeCompare(b));
  return sorted.map(([el, n]) => {
    const rounded = Math.round(n * 100) / 100;
    if (Math.abs(rounded - 1) < 0.01) return el;
    if (Number.isInteger(rounded)) return `${el}${rounded}`;
    return `${el}${rounded}`;
  }).join("");
}

function getTotalAtoms(counts: Record<string, number>): number {
  return Object.values(counts).reduce((s, n) => s + n, 0);
}

const SC_DOPANT_MAP: Record<string, string[]> = {
  La: ["Sr", "Ba", "Ca", "Ce", "Y"],
  Sr: ["La", "Ba", "Ca", "K"],
  Ba: ["La", "Sr", "K", "Ca"],
  Y: ["La", "Ce", "Ca", "Ba"],
  Ca: ["Sr", "La", "Ba", "Na"],
  Ti: ["Nb", "V", "Zr", "Hf"],
  Fe: ["Co", "Ni", "Mn", "Cu"],
  Co: ["Fe", "Ni", "Mn"],
  Ni: ["Cu", "Co", "Fe", "Pd"],
  Cu: ["Ni", "Zn", "Co"],
  Nb: ["Ti", "Ta", "V", "Mo"],
  Zr: ["Ti", "Hf", "Nb"],
  Mn: ["Fe", "Co", "Cr"],
  Bi: ["Sb", "Pb", "Tl"],
  Pb: ["Bi", "Sn", "Tl"],
  Sn: ["In", "Pb", "Ge"],
  In: ["Sn", "Ga", "Tl"],
  Ga: ["In", "Al"],
  Al: ["Ga", "In", "B"],
  B: ["C", "N", "Al"],
  Se: ["Te", "S"],
  Te: ["Se", "S"],
  As: ["P", "Sb"],
  P: ["As", "N"],
  Hf: ["Zr", "Ti"],
  Ta: ["Nb", "V"],
  Mo: ["W", "Nb"],
  W: ["Mo", "Ta"],
  Cr: ["V", "Mn"],
  V: ["Nb", "Ti", "Cr"],
  Ru: ["Os", "Ir"],
  Pd: ["Pt", "Ni"],
  Pt: ["Pd", "Ir"],
  Re: ["Tc", "Mo"],
  Ir: ["Rh", "Pt"],
  Rh: ["Ir", "Co"],
  Ge: ["Si", "Sn"],
  Si: ["Ge", "C"],
  N: ["C", "B"],
  C: ["N", "B"],
  O: ["F", "N"],
  F: ["O", "Cl"],
};

const INTERSTITIAL_DOPANTS: Record<string, string[]> = {
  layered: ["Li", "Na", "K", "Ca"],
  cage: ["H", "Li", "Na"],
  chalcogenide: ["Li", "Na", "K", "Cu"],
  pnictide: ["Li", "Na", "H"],
  oxide: ["H", "Li", "F"],
  general: ["Li", "H", "Na", "F"],
};

const VACANCY_TARGETS = ["O", "F", "S", "Se", "Te", "N", "Cl"];

const DOPING_FRACTIONS = [0.02, 0.05, 0.10, 0.15, 0.20];

const SEARCH_LIMITS = {
  maxDopantsPerMaterial: 2,
  maxDopingFraction: 0.20,
  maxSupercellAtoms: 27,
};

function classifyLayeredOrCage(formula: string): string {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
  const hasPnictogen = elements.some(e => ["As", "P", "Sb", "Bi"].includes(e));
  const hasOxygen = elements.includes("O");
  const hFrac = (counts["H"] || 0) / totalAtoms;

  if (hFrac > 0.3) return "cage";
  if (hasChalcogen) return "chalcogenide";
  if (hasPnictogen) return "pnictide";
  if (hasOxygen) return "oxide";

  const layeredElements = ["Bi", "Sb", "Se", "Te", "S", "As", "P"];
  if (elements.some(e => layeredElements.includes(e))) return "layered";

  return "general";
}

function getSupercellMultiplier(totalAtoms: number): number {
  if (totalAtoms <= 4) return 8;
  if (totalAtoms <= 8) return 4;
  if (totalAtoms <= 12) return 2;
  return 1;
}

function getDopantPriority(site: string, dopant: string, elements: string[]): number {
  if (elements.includes(dopant)) return -1;

  for (const pair of ELECTRON_DOPING_PAIRS) {
    if (pair.from === site && pair.to === dopant) return 10;
  }
  for (const pair of HOLE_DOPING_PAIRS) {
    if (pair.from === site && pair.to === dopant) return 10;
  }

  const siteOx = getOxidationState(site);
  const dopantOx = getOxidationState(dopant);
  const delta = Math.abs(dopantOx - siteOx);
  if (delta === 1) return 8;
  if (delta === 0) return 5;
  if (delta === 2) return 6;
  return 3;
}

function generateSubstitutionalVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 8
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const cellVolume = estimateUnitCellVolume(counts);

  for (const site of elements) {
    const siteCount = counts[site];
    if (siteCount < 0.5) continue;

    const dopants = SC_DOPANT_MAP[site];
    if (!dopants) continue;

    const siteData = getElementData(site);
    if (!siteData) continue;

    const sortedDopants = [...dopants]
      .map(d => ({ dopant: d, priority: getDopantPriority(site, d, elements) }))
      .filter(d => d.priority >= 0)
      .sort((a, b) => b.priority - a.priority)
      .map(d => d.dopant);

    for (const dopant of sortedDopants) {
      if (variants.length >= maxVariants) break;

      const dopantData = getElementData(dopant);
      if (!dopantData) continue;

      const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
        ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
        : 0.5;
      if (radiusDiff > 0.3) continue;

      const { character, valenceChange } = classifyDopingCharacter(site, dopant, "substitutional");

      const fractions = radiusDiff < 0.15
        ? DOPING_FRACTIONS
        : DOPING_FRACTIONS.filter(f => f <= 0.10);

      for (const fraction of fractions) {
        if (variants.length >= maxVariants) break;

        const supercellCounts: Record<string, number> = {};
        for (const [el, n] of Object.entries(counts)) {
          supercellCounts[el] = n * supercellMult;
        }

        const sitesInSupercell = supercellCounts[site];
        const nReplace = Math.max(1, Math.round(sitesInSupercell * fraction));
        if (nReplace >= sitesInSupercell) continue;

        supercellCounts[site] = sitesInSupercell - nReplace;
        supercellCounts[dopant] = (supercellCounts[dopant] || 0) + nReplace;

        const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
        const reduced: Record<string, number> = {};
        for (const [el, n] of Object.entries(supercellCounts)) {
          if (n > 0) reduced[el] = n / gcd;
        }

        const resultFormula = countsToFormula(reduced);
        if (!isValidFormula(resultFormula)) continue;

        const supercellVolume = cellVolume * supercellMult;
        const carrierDensity = computeCarrierDensity(valenceChange, nReplace, supercellVolume);

        const dopingLabel = character === "electron" ? "electron-doping" :
          character === "hole" ? "hole-doping" : "isovalent";
        const chargeInfo = valenceChange !== 0
          ? ` [${dopingLabel}: delta_q=${valenceChange > 0 ? "+" : ""}${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3]`
          : " [isovalent substitution]";

        const rationale = `${dopant} substitution at ${site} site (${(fraction * 100).toFixed(0)}%): `
          + `radius match ${(1 - radiusDiff).toFixed(2)}, `
          + `replaces ${nReplace}/${sitesInSupercell} ${site} atoms in ${supercellMult > 1 ? supercellMult + "x supercell" : "unit cell"}`
          + chargeInfo;

        variants.push({
          type: "substitutional",
          base: formula,
          dopant,
          site,
          fraction,
          resultFormula: normalizeFormula(resultFormula),
          supercellSize: supercellMult,
          rationale,
          dopingCharacter: character,
          valenceChange,
          carrierDensity,
        });
      }
    }
    if (variants.length >= maxVariants) break;
  }

  return variants;
}

function generateVacancyVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 4
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const cellVolume = estimateUnitCellVolume(counts);

  const vacancySites = elements.filter(e => VACANCY_TARGETS.includes(e));
  if (vacancySites.length === 0) return [];

  for (const site of vacancySites) {
    const siteCount = counts[site];
    if (siteCount < 1) continue;

    const { character, valenceChange } = classifyDopingCharacter(site, "", "vacancy");

    const fracs = [0.05, 0.10, 0.15];
    for (const fraction of fracs) {
      if (variants.length >= maxVariants) break;

      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      const sitesInSupercell = supercellCounts[site];
      const nRemove = Math.max(1, Math.round(sitesInSupercell * fraction));
      if (nRemove >= sitesInSupercell) continue;

      supercellCounts[site] = sitesInSupercell - nRemove;

      const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
      const reduced: Record<string, number> = {};
      for (const [el, n] of Object.entries(supercellCounts)) {
        if (n > 0) reduced[el] = n / gcd;
      }

      const resultFormula = countsToFormula(reduced);
      if (!isValidFormula(resultFormula)) continue;

      const supercellVolume = cellVolume * supercellMult;
      const carrierDensity = computeCarrierDensity(valenceChange, nRemove, supercellVolume);

      const carrierType = valenceChange < 0 ? "hole" : "electron";

      variants.push({
        type: "vacancy",
        base: formula,
        site,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        rationale: `${site} vacancy doping (${(fraction * 100).toFixed(0)}%): removed ${nRemove}/${sitesInSupercell} ${site} atoms — creates ${carrierType} carriers (delta_q=${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3)`,
        dopingCharacter: character,
        valenceChange,
        carrierDensity,
      });
    }
  }

  return variants;
}

function generateInterstitialVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 4
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const structureType = classifyLayeredOrCage(formula);
  const cellVolume = estimateUnitCellVolume(counts);

  const dopantPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const availableDopants = dopantPool.filter(d => !elements.includes(d));

  for (const dopant of availableDopants) {
    if (variants.length >= maxVariants) break;

    const { character, valenceChange } = classifyDopingCharacter("", dopant, "interstitial");

    const fracs = [0.05, 0.10];
    for (const fraction of fracs) {
      if (variants.length >= maxVariants) break;

      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      const totalInSupercell = getTotalAtoms(supercellCounts);
      const nInsert = Math.max(1, Math.round(totalInSupercell * fraction));

      supercellCounts[dopant] = (supercellCounts[dopant] || 0) + nInsert;

      const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
      const reduced: Record<string, number> = {};
      for (const [el, n] of Object.entries(supercellCounts)) {
        if (n > 0) reduced[el] = n / gcd;
      }

      const resultFormula = countsToFormula(reduced);
      if (!isValidFormula(resultFormula)) continue;

      const totalNew = getTotalAtoms(reduced);
      if (totalNew > SEARCH_LIMITS.maxSupercellAtoms) continue;

      const supercellVolume = cellVolume * supercellMult;
      const carrierDensity = computeCarrierDensity(valenceChange, nInsert, supercellVolume);

      variants.push({
        type: "interstitial",
        base: formula,
        dopant,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        rationale: `${dopant} interstitial insertion (${(fraction * 100).toFixed(0)}%): ${nInsert} atoms into ${structureType} structure — electron-doping (delta_q=+${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3)`,
        dopingCharacter: character,
        valenceChange,
        carrierDensity,
      });
    }
  }

  return variants;
}

function findGCD(nums: number[]): number {
  if (nums.length === 0) return 1;
  const gcd2 = (a: number, b: number): number => {
    a = Math.abs(Math.round(a));
    b = Math.abs(Math.round(b));
    while (b > 0) {
      [a, b] = [b, a % b];
    }
    return a || 1;
  };
  return nums.reduce((acc, n) => gcd2(acc, n), nums[0]);
}

export async function relaxDopedStructure(formula: string): Promise<RelaxationMetrics | null> {
  try {
    const optResult = await runXTBOptimization(formula, 0);
    if (!optResult || !optResult.converged) return null;

    const atoms = optResult.optimizedAtoms;
    if (atoms.length < 2) return null;

    const dist = optResult.distortion;

    const latticeStrain = dist?.latticeDistortion?.strainMagnitude ?? 0;
    const volumeChange = dist?.latticeDistortion?.volumeChangePct ?? 0;
    const meanDisplacement = dist?.atomicDistortion?.meanDisplacement ?? 0;
    const maxDisplacement = dist?.atomicDistortion?.maxDisplacement ?? 0;

    let bondVariance = 0;
    if (atoms.length >= 2) {
      const distances: number[] = [];
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const dx = atoms[i].x - atoms[j].x;
          const dy = atoms[i].y - atoms[j].y;
          const dz = atoms[i].z - atoms[j].z;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (d < 3.5) {
            distances.push(d);
          }
        }
      }
      if (distances.length > 1) {
        const mean = distances.reduce((s, d) => s + d, 0) / distances.length;
        bondVariance = Math.sqrt(
          distances.reduce((s, d) => s + (d - mean) ** 2, 0) / distances.length
        );
      }
    }

    const energyPerAtom = atoms.length > 0 ? (optResult.optimizedEnergy * 27.2114) / atoms.length : 0;

    let phononAnalysis: HessianPhononAnalysis | null = null;
    try {
      phononAnalysis = await analyzeHessianPhonons(formula);
    } catch (phErr) {
      console.log(`[Doping] Hessian phonon analysis failed for ${formula}: ${phErr instanceof Error ? phErr.message.slice(0, 80) : String(phErr).slice(0, 80)}`);
    }

    return {
      converged: optResult.converged,
      latticeStrain,
      bondVariance,
      meanDisplacement,
      maxDisplacement,
      volumeChange,
      energyPerAtom,
      wallTimeMs: optResult.wallTimeSeconds * 1000,
      phononAnalysis,
    };
  } catch {
    return null;
  }
}

export function generateDopedVariants(
  formula: string,
  maxTotal: number = 12
): DopingResult {
  const start = Date.now();
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);

  if (elements.length === 0 || totalAtoms < 2 || elements.length > 5) {
    return { baseFormula: formula, variants: [], totalGenerated: 0, validGenerated: 0, wallTimeMs: Date.now() - start };
  }

  const subMax = Math.ceil(maxTotal * 0.5);
  const vacMax = Math.ceil(maxTotal * 0.25);
  const intMax = maxTotal - subMax - vacMax;

  const substitutional = generateSubstitutionalVariants(formula, counts, subMax);
  const vacancy = generateVacancyVariants(formula, counts, vacMax);
  const interstitial = generateInterstitialVariants(formula, counts, intMax);

  const allVariants = [...substitutional, ...vacancy, ...interstitial];

  const seen = new Set<string>();
  seen.add(normalizeFormula(formula));
  const unique = allVariants.filter(v => {
    if (seen.has(v.resultFormula)) return false;
    seen.add(v.resultFormula);
    return true;
  });

  stats.totalBaseMaterials++;
  stats.totalVariantsGenerated += unique.length;
  stats.substitutionalCount += substitutional.filter(v => unique.includes(v)).length;
  stats.vacancyCount += vacancy.filter(v => unique.includes(v)).length;
  stats.interstitialCount += interstitial.filter(v => unique.includes(v)).length;
  stats.validVariants += unique.length;
  stats.electronDopedCount += unique.filter(v =>
    v.dopingCharacter === "electron" || v.dopingCharacter === "interstitial-electron"
  ).length;
  stats.holeDopedCount += unique.filter(v =>
    v.dopingCharacter === "hole" || v.dopingCharacter === "vacancy-hole"
  ).length;

  stats.recentResults.push({ base: formula, variants: unique.length, timestamp: Date.now() });
  if (stats.recentResults.length > MAX_RECENT) {
    stats.recentResults = stats.recentResults.slice(-MAX_RECENT);
  }

  return {
    baseFormula: formula,
    variants: unique,
    totalGenerated: allVariants.length,
    validGenerated: unique.length,
    wallTimeMs: Date.now() - start,
  };
}

export async function generateDopedVariantsWithRelaxation(
  formula: string,
  maxTotal: number = 12,
  maxRelaxations: number = 4
): Promise<DopingResult> {
  const result = generateDopedVariants(formula, maxTotal);

  const toRelax = result.variants
    .filter(v => v.dopingCharacter !== "isovalent")
    .sort((a, b) => Math.abs(b.carrierDensity) - Math.abs(a.carrierDensity))
    .slice(0, maxRelaxations);

  for (const variant of toRelax) {
    const metrics = await relaxDopedStructure(variant.resultFormula);
    if (metrics) {
      variant.relaxation = metrics;
      stats.relaxationsCompleted++;
      totalStrainSum += metrics.latticeStrain;
      stats.avgLatticeStrain = stats.relaxationsCompleted > 0
        ? totalStrainSum / stats.relaxationsCompleted
        : 0;
    }
  }

  return result;
}

export function runDopingBatch(
  formulas: string[],
  maxVariantsPerBase: number = 8,
  maxTotalDoped: number = 50,
  excludeSet?: Set<string>
): { dopedFormulas: string[]; specs: DopingSpec[]; stats: { basesProcessed: number; totalVariants: number; substitutional: number; vacancy: number; interstitial: number; electronDoped: number; holeDoped: number } } {
  const dopedFormulas: string[] = [];
  const specs: DopingSpec[] = [];
  let subCount = 0, vacCount = 0, intCount = 0;
  let eDoped = 0, hDoped = 0;

  for (const base of formulas) {
    if (dopedFormulas.length >= maxTotalDoped) break;

    const result = generateDopedVariants(base, maxVariantsPerBase);
    for (const v of result.variants) {
      if (dopedFormulas.length >= maxTotalDoped) break;
      if (excludeSet && excludeSet.has(v.resultFormula)) continue;

      dopedFormulas.push(v.resultFormula);
      specs.push(v);
      if (v.type === "substitutional") subCount++;
      else if (v.type === "vacancy") vacCount++;
      else intCount++;
      if (v.dopingCharacter === "electron" || v.dopingCharacter === "interstitial-electron") eDoped++;
      if (v.dopingCharacter === "hole" || v.dopingCharacter === "vacancy-hole") hDoped++;
    }
  }

  return {
    dopedFormulas,
    specs,
    stats: {
      basesProcessed: formulas.length,
      totalVariants: dopedFormulas.length,
      substitutional: subCount,
      vacancy: vacCount,
      interstitial: intCount,
      electronDoped: eDoped,
      holeDoped: hDoped,
    },
  };
}

export function getDopingEngineStats(): DopingEngineStats {
  return { ...stats };
}

export function getDopingRecommendations(formula: string): {
  substitutional: Array<{ dopant: string; site: string; rationale: string; dopingType: string; valenceChange: number }>;
  vacancy: Array<{ site: string; rationale: string; dopingType: string; valenceChange: number }>;
  interstitial: Array<{ dopant: string; rationale: string; dopingType: string; valenceChange: number }>;
} {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const structureType = classifyLayeredOrCage(formula);

  const sub: Array<{ dopant: string; site: string; rationale: string; dopingType: string; valenceChange: number }> = [];

  for (const site of elements) {
    for (const pair of ELECTRON_DOPING_PAIRS) {
      if (pair.from === site && !elements.includes(pair.to)) {
        const { valenceChange } = classifyDopingCharacter(site, pair.to, "substitutional");
        const siteData = getElementData(site);
        const dopantData = getElementData(pair.to);
        if (siteData && dopantData) {
          const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
            ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
            : 0.5;
          sub.push({
            dopant: pair.to,
            site,
            rationale: `${pair.to} replaces ${site}: electron-doping (${site}${getOxidationState(site) > 0 ? "+" + getOxidationState(site) : getOxidationState(site)} -> ${pair.to}${getOxidationState(pair.to) > 0 ? "+" + getOxidationState(pair.to) : getOxidationState(pair.to)}), radius match ${((1 - radiusDiff) * 100).toFixed(0)}%`,
            dopingType: "electron",
            valenceChange,
          });
        }
      }
    }

    for (const pair of HOLE_DOPING_PAIRS) {
      if (pair.from === site && !elements.includes(pair.to)) {
        const { valenceChange } = classifyDopingCharacter(site, pair.to, "substitutional");
        const siteData = getElementData(site);
        const dopantData = getElementData(pair.to);
        if (siteData && dopantData) {
          const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
            ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
            : 0.5;
          sub.push({
            dopant: pair.to,
            site,
            rationale: `${pair.to} replaces ${site}: hole-doping (${site}${getOxidationState(site) > 0 ? "+" + getOxidationState(site) : getOxidationState(site)} -> ${pair.to}${getOxidationState(pair.to) > 0 ? "+" + getOxidationState(pair.to) : getOxidationState(pair.to)}), radius match ${((1 - radiusDiff) * 100).toFixed(0)}%`,
            dopingType: "hole",
            valenceChange,
          });
        }
      }
    }

    if (sub.filter(s => s.site === site).length === 0) {
      const dopants = SC_DOPANT_MAP[site];
      if (!dopants) continue;
      const siteData = getElementData(site);
      if (!siteData) continue;
      for (const dopant of dopants.slice(0, 2)) {
        if (elements.includes(dopant)) continue;
        const dopantData = getElementData(dopant);
        if (!dopantData) continue;
        const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
          ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
          : 0.5;
        if (radiusDiff <= 0.3) {
          const { character, valenceChange } = classifyDopingCharacter(site, dopant, "substitutional");
          sub.push({
            dopant,
            site,
            rationale: `${dopant} replaces ${site}: ${character}-doping (delta_q=${valenceChange}), radius match ${((1 - radiusDiff) * 100).toFixed(0)}%`,
            dopingType: character,
            valenceChange,
          });
        }
      }
    }
  }

  const vac: Array<{ site: string; rationale: string; dopingType: string; valenceChange: number }> = [];
  for (const site of elements) {
    if (VACANCY_TARGETS.includes(site) && counts[site] >= 1) {
      const { valenceChange } = classifyDopingCharacter(site, "", "vacancy");
      const carrierType = valenceChange < 0 ? "hole" : "electron";
      vac.push({
        site,
        rationale: `${site} vacancy: creates ${carrierType} carriers (removes ${site}${getOxidationState(site) > 0 ? "+" : ""}${getOxidationState(site)} charge)`,
        dopingType: "vacancy-" + carrierType,
        valenceChange,
      });
    }
  }

  const intPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const int: Array<{ dopant: string; rationale: string; dopingType: string; valenceChange: number }> = [];
  for (const dopant of intPool) {
    if (!elements.includes(dopant)) {
      const { valenceChange } = classifyDopingCharacter("", dopant, "interstitial");
      int.push({
        dopant,
        rationale: `${dopant} intercalation into ${structureType} lattice — donates ${valenceChange} electron(s), enhances electron-phonon coupling`,
        dopingType: "interstitial-electron",
        valenceChange,
      });
    }
  }

  return { substitutional: sub, vacancy: vac, interstitial: int.slice(0, 3) };
}

export interface SCSignal {
  dosAtEF: number;
  dosIncrease: number;
  magnetismSuppressed: boolean;
  stonerCriterion: number;
  structuralTransition: string | null;
  phononSoftening: number;
  overallSCIndicator: number;
  hessianPhonon: {
    softModeCount: number;
    imaginaryModeCount: number;
    avgFrequency_THz: number;
    latticeClassification: "stiff" | "moderate" | "soft";
    hasLatticeInstability: boolean;
    phononSCScore: number;
    interpretation: string;
  } | null;
}

export function detectSCSignals(baseFormula: string, dopedFormula: string, dopedHessian?: PhononStability | null): SCSignal {
  const baseElectronic = computeElectronicStructure(baseFormula);
  const dopedElectronic = computeElectronicStructure(dopedFormula);

  const baseDOS = baseElectronic?.densityOfStatesAtFermi ?? 0;
  const dopedDOS = dopedElectronic?.densityOfStatesAtFermi ?? 0;
  const dosIncrease = baseDOS > 0 ? (dopedDOS - baseDOS) / baseDOS : 0;

  const baseElements = parseFormulaElements(baseFormula);
  const dopedElements = parseFormulaElements(dopedFormula);

  let baseStonerMax = 0;
  for (const el of baseElements) {
    const I = getStonerParameter(el) ?? 0;
    baseStonerMax = Math.max(baseStonerMax, I * baseDOS);
  }
  let dopedStonerMax = 0;
  for (const el of dopedElements) {
    const I = getStonerParameter(el) ?? 0;
    dopedStonerMax = Math.max(dopedStonerMax, I * dopedDOS);
  }

  const magnetismSuppressed = baseStonerMax > 0.8 && dopedStonerMax < 0.9;

  const basePhonon = computePhononSpectrum(baseFormula, baseElectronic);
  const dopedPhonon = computePhononSpectrum(dopedFormula, dopedElectronic);
  const baseLogFreq = basePhonon.logAverageFrequency;
  const dopedLogFreq = dopedPhonon.logAverageFrequency;
  const phononSoftening = baseLogFreq > 0 ? (baseLogFreq - dopedLogFreq) / baseLogFreq : 0;

  let structuralTransition: string | null = null;
  const baseCrystal = guessCrystalSystem(baseFormula);
  const dopedCrystal = guessCrystalSystem(dopedFormula);
  if (baseCrystal !== dopedCrystal && baseCrystal && dopedCrystal) {
    structuralTransition = `${baseCrystal} -> ${dopedCrystal}`;
  }

  let hessianPhonon: SCSignal["hessianPhonon"] = null;
  if (dopedHessian && dopedHessian.frequencies.length > 0) {
    const analysis = processPhononStability(dopedHessian);
    hessianPhonon = {
      softModeCount: analysis.softModeCount,
      imaginaryModeCount: analysis.imaginaryModeCount,
      avgFrequency_THz: analysis.avgFrequency_THz,
      latticeClassification: analysis.latticeClassification,
      hasLatticeInstability: analysis.hasLatticeInstability,
      phononSCScore: analysis.scRelevance.overallPhononSCScore,
      interpretation: analysis.scRelevance.interpretation,
    };
  }

  let indicator = 0;
  if (dosIncrease > 0.05) indicator += 0.20 * Math.min(dosIncrease, 1.0);
  if (magnetismSuppressed) indicator += 0.20;
  if (phononSoftening > 0.05) indicator += 0.15 * Math.min(phononSoftening, 1.0);
  if (structuralTransition) indicator += 0.10;
  if (dopedDOS > 2.0) indicator += 0.10;
  if (hessianPhonon) {
    indicator += 0.25 * hessianPhonon.phononSCScore;
  } else {
    if (phononSoftening > 0.05) indicator += 0.05 * Math.min(phononSoftening, 1.0);
    if (dopedDOS > 2.0) indicator += 0.05;
  }

  return {
    dosAtEF: dopedDOS,
    dosIncrease,
    magnetismSuppressed,
    stonerCriterion: dopedStonerMax,
    structuralTransition,
    phononSoftening,
    overallSCIndicator: Math.min(1, indicator),
    hessianPhonon,
  };
}

function guessCrystalSystem(formula: string): string {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  if (elements.includes("Cu") && elements.includes("O") && (elements.includes("La") || elements.includes("Y") || elements.includes("Ba"))) {
    return totalAtoms > 10 ? "orthorhombic" : "tetragonal";
  }
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"))) {
    return "tetragonal";
  }
  if (elements.length === 2) return "cubic";
  if (elements.length === 3) return "tetragonal";
  return "orthorhombic";
}

export interface DopingSearchResult {
  baseFormula: string;
  dopant: string;
  site: string;
  type: "substitutional" | "vacancy" | "interstitial";
  levels: Array<{
    fraction: number;
    resultFormula: string;
    predictedTc: number;
    tcUncertainty: number;
    carrierDensity: number;
    dopingCharacter: DopingCharacter;
    scSignals: SCSignal;
  }>;
  bestFraction: number;
  bestTc: number;
  tcTrend: "increasing" | "decreasing" | "peaked" | "flat";
}

export function runDopingSearchLoop(
  formula: string,
  fractions: number[] = [0.02, 0.05, 0.10, 0.15, 0.20],
  maxDopants: number = SEARCH_LIMITS.maxDopantsPerMaterial
): { results: DopingSearchResult[]; bestOverall: { formula: string; tc: number; fraction: number; dopant: string } | null; wallTimeMs: number } {
  const start = Date.now();
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const results: DopingSearchResult[] = [];

  const clampedFracs = fractions.filter(f => f > 0 && f <= SEARCH_LIMITS.maxDopingFraction);

  let globalBestTc = 0;
  let globalBest: { formula: string; tc: number; fraction: number; dopant: string } | null = null;

  const dopantSitePairs: Array<{ dopant: string; site: string; type: "substitutional" | "vacancy" | "interstitial" }> = [];

  for (const site of elements) {
    const dopants = SC_DOPANT_MAP[site];
    if (!dopants) continue;
    const siteData = getElementData(site);
    if (!siteData) continue;

    for (const dopant of dopants) {
      if (elements.includes(dopant)) continue;
      const dopantData = getElementData(dopant);
      if (!dopantData) continue;
      const radiusDiff = siteData.atomicRadius > 0 && dopantData.atomicRadius > 0
        ? Math.abs(siteData.atomicRadius - dopantData.atomicRadius) / siteData.atomicRadius
        : 0.5;
      if (radiusDiff <= 0.3) {
        dopantSitePairs.push({ dopant, site, type: "substitutional" });
      }
    }
  }

  for (const site of elements) {
    if (VACANCY_TARGETS.includes(site) && counts[site] >= 1) {
      dopantSitePairs.push({ dopant: "", site, type: "vacancy" });
    }
  }

  const sortedPairs = dopantSitePairs
    .map(p => ({ ...p, priority: p.type === "vacancy" ? 5 : getDopantPriority(p.site, p.dopant, elements) }))
    .filter(p => p.priority >= 0)
    .sort((a, b) => b.priority - a.priority)
    .slice(0, maxDopants * 3);

  for (const pair of sortedPairs) {
    if (results.length >= maxDopants * 2) break;

    const levels: DopingSearchResult["levels"] = [];
    const supercellMult = getSupercellMultiplier(getTotalAtoms(counts));
    const cellVolume = estimateUnitCellVolume(counts);

    for (const fraction of clampedFracs) {
      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      let nChanged = 0;
      if (pair.type === "substitutional") {
        const sitesInSupercell = supercellCounts[pair.site];
        nChanged = Math.max(1, Math.round(sitesInSupercell * fraction));
        if (nChanged >= sitesInSupercell) continue;
        supercellCounts[pair.site] -= nChanged;
        supercellCounts[pair.dopant] = (supercellCounts[pair.dopant] || 0) + nChanged;
      } else {
        const sitesInSupercell = supercellCounts[pair.site];
        nChanged = Math.max(1, Math.round(sitesInSupercell * fraction));
        if (nChanged >= sitesInSupercell) continue;
        supercellCounts[pair.site] -= nChanged;
      }

      const gcd = findGCD(Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v)));
      const reduced: Record<string, number> = {};
      for (const [el, n] of Object.entries(supercellCounts)) {
        if (n > 0) reduced[el] = n / gcd;
      }

      const resultFormula = normalizeFormula(countsToFormula(reduced));
      if (!isValidFormula(resultFormula)) continue;
      if (getTotalAtoms(reduced) > SEARCH_LIMITS.maxSupercellAtoms) continue;

      const { character, valenceChange } = pair.type === "vacancy"
        ? classifyDopingCharacter(pair.site, "", "vacancy")
        : classifyDopingCharacter(pair.site, pair.dopant, "substitutional");

      const supercellVolume = cellVolume * supercellMult;
      const carrierDensity = computeCarrierDensity(valenceChange, nChanged, supercellVolume);

      try {
        const features = extractFeatures(resultFormula);
        const prediction = gbPredictWithUncertainty(features, resultFormula);
        const tc = prediction.tcMean ?? 0;
        const unc = prediction.totalStd ?? 0;

        const scSignals = detectSCSignals(formula, resultFormula);

        levels.push({
          fraction,
          resultFormula,
          predictedTc: tc,
          tcUncertainty: unc,
          carrierDensity,
          dopingCharacter: character,
          scSignals,
        });

        if (tc > globalBestTc) {
          globalBestTc = tc;
          globalBest = { formula: resultFormula, tc, fraction, dopant: pair.dopant || `${pair.site}-vacancy` };
        }
      } catch { /* skip on prediction failure */ }
    }

    if (levels.length > 0) {
      const tcs = levels.map(l => l.predictedTc);
      const bestIdx = tcs.indexOf(Math.max(...tcs));
      let trend: DopingSearchResult["tcTrend"] = "flat";
      if (tcs.length >= 3) {
        const first = tcs[0];
        const last = tcs[tcs.length - 1];
        const peak = Math.max(...tcs);
        const peakIdx = tcs.indexOf(peak);
        if (peakIdx > 0 && peakIdx < tcs.length - 1 && peak > first * 1.05 && peak > last * 1.05) {
          trend = "peaked";
        } else if (last > first * 1.1) {
          trend = "increasing";
        } else if (last < first * 0.9) {
          trend = "decreasing";
        }
      }

      results.push({
        baseFormula: formula,
        dopant: pair.dopant || `${pair.site}-vacancy`,
        site: pair.site,
        type: pair.type,
        levels,
        bestFraction: levels[bestIdx].fraction,
        bestTc: levels[bestIdx].predictedTc,
        tcTrend: trend,
      });
    }
  }

  results.sort((a, b) => b.bestTc - a.bestTc);

  stats.totalBaseMaterials++;

  return {
    results,
    bestOverall: globalBest,
    wallTimeMs: Date.now() - start,
  };
}
