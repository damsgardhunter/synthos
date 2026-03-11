import { parseFormulaElements, computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling } from "./physics-engine";
import { normalizeFormula, isValidFormula } from "./utils";
import { getElementData, getStonerParameter } from "./elemental-data";
import { runXTBOptimization, runXTBPhononCheck, runXTBAnharmonicProbe, runXTBMDSampling, type PhononStability, type AnharmonicProbeResult, type MDSamplingRawResult } from "../dft/qe-dft-engine";
import { extractFeatures } from "./ml-predictor";
import { gbPredictWithUncertainty } from "./gradient-boost";
import { checkValenceSumRule } from "../physics/advanced-constraints";
import { passesElementCountCap } from "./candidate-generator";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { getEntryByFormula } from "../crystal/crystal-structure-dataset";

export type DopingCharacter = "electron" | "hole" | "isovalent" | "vacancy-hole" | "interstitial-electron";

export interface DopingSpec {
  type: "substitutional" | "vacancy" | "interstitial";
  base: string;
  dopant?: string;
  site?: string;
  fraction: number;
  resultFormula: string;
  supercellSize: number;
  minSupercellSize?: number;
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
  latticeCollapse: boolean;
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
    return processPhononStability(stability, formula);
  }

  try {
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    if (!phonon) return null;

    const maxFreq = phonon.maxPhononFrequency;
    const logAvg = phonon.logAverageFrequency;
    const nModes = Math.max(6, Math.min(20, parseFormulaElements(formula).length * 3));
    const nAcoustic = 3;
    const nOptical = nModes - nAcoustic;
    const syntheticFreqs: number[] = [];
    for (let i = 0; i < nAcoustic; i++) {
      const frac = (i + 1) / nAcoustic;
      syntheticFreqs.push(maxFreq * 0.4 * Math.sqrt(frac) * (0.8 + 0.4 * ((i * 7 + 3) % 5) / 5));
    }
    const opticalFloor = maxFreq * 0.4;
    const opticalRange = maxFreq - opticalFloor;
    for (let i = 0; i < nOptical; i++) {
      const frac = (i + 0.5) / nOptical;
      const clustering = 0.5 + 0.5 * Math.sin(frac * Math.PI);
      syntheticFreqs.push(opticalFloor + opticalRange * frac * clustering * (0.85 + 0.3 * ((i * 11 + 7) % 9) / 9));
    }
    if (phonon.hasImaginaryModes) {
      const instabilitySeverity = 0.1 + 0.4 * phonon.anharmonicityIndex + 0.3 * phonon.softModeScore;
      syntheticFreqs[0] = -Math.abs(logAvg * instabilitySeverity);
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
    const result = processPhononStability(syntheticStability, formula);
    (result as any).source = "physics-engine-estimate";
    return result;
  } catch {
    return null;
  }
}

export function processPhononStability(stability: PhononStability, formula?: string): HessianPhononAnalysis {
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

  let isHydride = false;
  if (formula) {
    const els = parseFormulaElements(formula);
    if (els.includes("H")) {
      const counts: Record<string, number> = {};
      const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
      let m;
      const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
      while ((m = regex.exec(cleaned)) !== null) {
        const el = m[1];
        const n = m[2] ? parseFloat(m[2]) : 1;
        counts[el] = (counts[el] || 0) + n;
      }
      const total = Object.values(counts).reduce((s, n) => s + n, 0);
      isHydride = (counts["H"] || 0) / total > 0.4;
    }
  }

  let couplingScore = 0;
  if (latticeClassification === "moderate") couplingScore = 0.7;
  else if (latticeClassification === "soft") couplingScore = 0.85;
  else couplingScore = 0.2;
  if (softModes.length > 0) couplingScore = Math.min(1.0, couplingScore + 0.15);

  if (isHydride && maxFreq_THz > 30) {
    const highFreqBonus = Math.min(0.3, (maxFreq_THz - 30) / 100);
    couplingScore = Math.min(1.0, couplingScore + highFreqBonus);
  }

  const overallPhononSCScore = 0.35 * softPhononScore + 0.30 * instabilityScore + 0.35 * couplingScore;

  let interpretation: string;
  if (imaginaryModes.length > 0 && softModes.length > 2) {
    interpretation = `Lattice shows ${imaginaryModes.length} imaginary mode(s) and ${softModes.length} soft mode(s) — strong dynamic instability suggesting structural transition near SC`;
  } else if (imaginaryModes.length > 0) {
    interpretation = `${imaginaryModes.length} imaginary mode(s) detected — lattice wants to distort, possible SC-relevant instability`;
  } else if (softModes.length > 2) {
    interpretation = `${softModes.length} soft phonon mode(s) below 1 THz — lattice near structural transition, favorable for electron-phonon coupling`;
  } else if (isHydride && maxFreq_THz > 30) {
    interpretation = `High-frequency H vibrations (${maxFreq_THz.toFixed(1)} THz) — strong phonon pre-factor for Allen-Dynes Tc`;
  } else if (latticeClassification === "moderate") {
    interpretation = "Moderate phonon spectrum — good electron-phonon coupling range for conventional SC";
  } else if (latticeClassification === "soft") {
    interpretation = "Soft lattice dynamics — phonon-mediated coupling potential, but reduced phonon pre-factor";
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
    avgFrequency_THz: avgFreq_THz,
    avgFrequency_cm1: avgFreq_THz / CM1_TO_THZ,
    maxFrequency_THz: maxFreq_THz,
    lowestFrequency_THz: lowestFreq_THz,
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
    const FALLBACK_MASSES: Record<string, number> = {
      H: 1.008, He: 4.003, Li: 6.941, Be: 9.012, B: 10.81, C: 12.01, N: 14.01, O: 16.00,
      F: 19.00, Na: 22.99, Mg: 24.31, Al: 26.98, Si: 28.09, P: 30.97, S: 32.07, Cl: 35.45,
      K: 39.10, Ca: 40.08, Ti: 47.87, V: 50.94, Cr: 52.00, Mn: 54.94, Fe: 55.85, Co: 58.93,
      Ni: 58.69, Cu: 63.55, Zn: 65.38, Ga: 69.72, Ge: 72.63, As: 74.92, Se: 78.97, Br: 79.90,
      Rb: 85.47, Sr: 87.62, Y: 88.91, Zr: 91.22, Nb: 92.91, Mo: 95.95, Ru: 101.1, Rh: 102.9,
      Pd: 106.4, Ag: 107.9, In: 114.8, Sn: 118.7, Sb: 121.8, Te: 127.6, I: 126.9, Cs: 132.9,
      Ba: 137.3, La: 138.9, Ce: 140.1, Pr: 140.9, Nd: 144.2, Hf: 178.5, Ta: 180.9, W: 183.8,
      Re: 186.2, Os: 190.2, Ir: 192.2, Pt: 195.1, Au: 197.0, Tl: 204.4, Pb: 207.2, Bi: 209.0,
      Sc: 44.96, Gd: 157.3, Sm: 150.4, Dy: 162.5, Er: 167.3, Yb: 173.0, Lu: 175.0,
    };
    const avgMass = elements.reduce((s, el) => {
      const d = getElementData(el);
      const mass = d?.atomicMass ?? FALLBACK_MASSES[el] ?? 50;
      return s + mass * (counts[el] ?? 1);
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
      const cubicF = -0.9 * k_eff * anharmonicIndex * d * Math.abs(d);
      const quarticF = -0.6 * k_eff * anharmonicIndex * d * d * d;
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

  const dMax = Math.max(...displacements.map(Math.abs));
  const scale = dMax > 1e-12 ? dMax : 1.0;
  const normD = displacements.map(d => d / scale);

  let sumU2 = 0, sumU4 = 0, sumU2Y = 0, sumY = 0;
  for (let i = 0; i < n; i++) {
    const u2 = normD[i] * normD[i];
    sumU2 += u2;
    sumU4 += u2 * u2;
    sumU2Y += u2 * energies[i];
    sumY += energies[i];
  }
  const denom = n * sumU4 - sumU2 * sumU2;
  const a2_norm = (n > 0 && Math.abs(denom) > 1e-30)
    ? (n * sumU2Y - sumU2 * sumY) / denom
    : 0;
  const a0_norm = (sumY - a2_norm * sumU2) / Math.max(1, n);

  const a2 = a2_norm / (scale * scale);
  const a0 = a0_norm;
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
      const u = normD[i];
      const eActual = energies[i];
      const eQuad = a0_norm + a2_norm * u * u;
      const residual = eActual - eQuad;
      cubicContribution += Math.abs(residual * u) / (Math.abs(u * u * u) + 1e-10);
      quarticContribution += Math.abs(residual) / (u * u * u * u + 1e-10);
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

    const LIGHT_ELEMENTS = new Set(["H", "He", "Li", "Be", "B", "C", "N", "O", "F"]);
    const lightFraction = elements.reduce((s, el) => s + (LIGHT_ELEMENTS.has(el) ? (counts[el] ?? 1) : 0), 0) / Math.max(1, totalAtoms);
    const hasLightModes = lightFraction > 0.15;

    if (hasLightModes) {
      const maxFreq_THz = phonon.maxPhononFrequency * CM1_TO_THZ;
      const opticalFreq = Math.max(avgFreq_THz * 2.5, maxFreq_THz * 0.7);
      const opticalDecayRate = vacDecayRate * 0.6;
      const acousticWeight = 1.0 - lightFraction * 0.6;
      const opticalWeight = lightFraction * 0.6;
      for (let i = 0; i < vacLength; i++) {
        const acoustic = Math.exp(-i * vacDecayRate * 0.02) * Math.cos(2 * Math.PI * avgFreq_THz * 0.001 * i);
        const optical = Math.exp(-i * opticalDecayRate * 0.02) * Math.cos(2 * Math.PI * opticalFreq * 0.001 * i);
        vac.push(acousticWeight * acoustic + opticalWeight * optical);
      }
    } else {
      for (let i = 0; i < vacLength; i++) {
        vac.push(Math.exp(-i * vacDecayRate * 0.02) * Math.cos(2 * Math.PI * avgFreq_THz * 0.001 * i));
      }
    }

    const vacDecayTime = vacDecayRate > 0 ? 1.0 / (vacDecayRate * 0.02) : 50;
    const rmsFluctuation = Math.sqrt(msd);
    const maxDisplacement = rmsFluctuation * 2.5;

    let avgBondLength = 0;
    let bondCount = 0;
    for (let i = 0; i < elements.length; i++) {
      for (let j = i; j < elements.length; j++) {
        const r1 = getElementData(elements[i])?.atomicRadius ?? 100;
        const r2 = getElementData(elements[j])?.atomicRadius ?? 100;
        avgBondLength += (r1 + r2) / 100;
        bondCount++;
      }
    }
    avgBondLength = bondCount > 0 ? avgBondLength / bondCount : 2.0;
    const normalizedRMS = avgBondLength > 0.1 ? rmsFluctuation / avgBondLength : rmsFluctuation;

    let fluctuationClassification: "rigid" | "moderate" | "large" | "extreme";
    if (normalizedRMS < 0.025) fluctuationClassification = "rigid";
    else if (normalizedRMS < 0.075) fluctuationClassification = "moderate";
    else if (normalizedRMS < 0.175) fluctuationClassification = "large";
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
    let zeroLagNorm = 0;
    for (let f = 0; f < nFrames; f++) {
      for (let a = 0; a < nAtoms; a++) {
        const v = velocities[f]?.[a] ?? [0, 0, 0];
        zeroLagNorm += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
      }
    }
    const avgNorm = nFrames > 0 ? zeroLagNorm / nFrames : 1;

    for (let tau = 0; tau < vacLength; tau++) {
      const nPairs = nFrames - tau;
      let corr = 0;
      for (let f = 0; f < nPairs; f++) {
        for (let a = 0; a < nAtoms; a++) {
          const v0 = velocities[f]?.[a] ?? [0, 0, 0];
          const vt = velocities[f + tau]?.[a] ?? [0, 0, 0];
          corr += v0[0] * vt[0] + v0[1] * vt[1] + v0[2] * vt[2];
        }
      }
      const avgCorr = nPairs > 0 ? corr / nPairs : 0;
      const normalized = avgNorm > 1e-30 ? Math.max(-1, Math.min(1, avgCorr / avgNorm)) : 0;
      vac.push(normalized);
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

  let avgBondLength = 0;
  let bondCount = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i; j < elements.length; j++) {
      const r1 = getElementData(elements[i])?.atomicRadius ?? 100;
      const r2 = getElementData(elements[j])?.atomicRadius ?? 100;
      avgBondLength += (r1 + r2) / 100;
      bondCount++;
    }
  }
  avgBondLength = bondCount > 0 ? avgBondLength / bondCount : 2.0;
  const normalizedRMS = avgBondLength > 0.1 ? rmsFluctuation / avgBondLength : rmsFluctuation;

  let fluctuationClassification: "rigid" | "moderate" | "large" | "extreme";
  if (normalizedRMS < 0.025) fluctuationClassification = "rigid";
  else if (normalizedRMS < 0.075) fluctuationClassification = "moderate";
  else if (normalizedRMS < 0.175) fluctuationClassification = "large";
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

  const scRelevance = 0.2 + 0.6 / (1 + Math.exp(0.008 * (debyeTemperature - 400)));

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

export interface DynamicLatticeScore {
  formula: string;
  overallScore: number;
  components: {
    softModeFraction: number;
    imaginaryModeFlag: number;
    phononVariance: number;
    anharmonicityContribution: number;
    lightElementBonus: number;
    layeredStructureBonus: number;
    cageLatticeBonus: number;
    looselyBondedBonus: number;
  };
  electronPhononFeatures: {
    dosAtFermi: number;
    metallicity: number;
    lambda: number;
    omegaLog: number;
    nestingScore: number;
    correlationStrength: number;
    vanHoveProximity: number;
    combinedElectronPhononScore: number;
  };
  dynamicEffectsProfile: {
    hasLightElements: boolean;
    lightElements: string[];
    isLayered: boolean;
    isCageLike: boolean;
    hasLooselyBonded: boolean;
    looselyBondedAtoms: string[];
    dynamicEffectStrength: "weak" | "moderate" | "strong" | "very-strong";
  };
  mlFeatures: {
    softPhononCount: number;
    avgPhononFrequency: number;
    phononVariance: number;
    instabilityFlag: boolean;
    debyeTemperature: number;
    anharmonicityIndex: number;
    dynamicLatticeScore: number;
  };
  tcRelevance: string;
  source: string;
}

const LIGHT_ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F"];
const CAGE_FORMERS = ["B", "C", "Si", "Ge", "Al", "Ga", "Sn"];
const CAGE_GUESTS = ["La", "Ce", "Ba", "Sr", "Ca", "Y", "K", "Na", "Rb", "Cs"];
const LOOSELY_BONDED_INDICATORS = ["K", "Rb", "Cs", "Ba", "Sr", "Ca", "Na", "Tl", "Pb", "Bi", "In"];

export function computeDynamicLatticeScore(formula: string): DynamicLatticeScore | null {
  try {
    const elements = parseFormulaElements(formula);
    const counts = parseFormulaCounts(formula);
    const totalAtoms = getTotalAtoms(counts);

    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    if (!phonon) return null;
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula);

    const softModeFraction = phonon.softModeScore;
    const imaginaryModeFlag = phonon.hasImaginaryModes ? 1.0 : 0.0;

    const maxFreq = phonon.maxPhononFrequency;
    const logFreq = phonon.logAverageFrequency;
    const freqRange = maxFreq - logFreq;
    const phononVariance = maxFreq > 0 ? Math.min(1.0, (freqRange / maxFreq) * 1.5) : 0;

    const anharmonicityContribution = Math.min(1.0, phonon.anharmonicityIndex * 2.0);

    const lightEls = elements.filter(e => LIGHT_ELEMENTS.includes(e));
    const lightFraction = lightEls.reduce((s, e) => s + (counts[e] ?? 0), 0) / totalAtoms;
    const hFraction = (counts["H"] ?? 0) / totalAtoms;
    let lightElementBonus: number;
    if (hFraction > 0.3) {
      const hBonus = Math.min(1.0, 0.4 + hFraction * 0.8);
      const otherLightFrac = lightFraction - hFraction;
      lightElementBonus = Math.min(1.5, hBonus + otherLightFrac * 1.5);
    } else {
      lightElementBonus = Math.min(1.0, lightFraction * 2.0);
    }

    const layeredIndicators = ["Cu", "Fe", "Ni", "Co", "Mn"].filter(e => elements.includes(e));
    const hasOxygen = elements.includes("O");
    const hasChalcogen = elements.some(e => ["S", "Se", "Te"].includes(e));
    const hasPnictogen = elements.some(e => ["As", "P", "Sb"].includes(e));

    const fermiTopology = electronic.fermiSurfaceTopology || "";
    const topologyIs2D = fermiTopology.includes("2D") || fermiTopology.includes("cylindrical") || fermiTopology.includes("nesting");
    const chemistryIsLayered = (layeredIndicators.length > 0 && (hasOxygen || hasChalcogen || hasPnictogen))
      || elements.length >= 3 && (hasChalcogen || (hasOxygen && elements.length >= 4));
    const isLayered = topologyIs2D
      ? chemistryIsLayered
      : chemistryIsLayered && layeredIndicators.length >= 2;
    let layeredStructureBonus = 0;
    if (isLayered && topologyIs2D) layeredStructureBonus = 0.6;
    else if (isLayered) layeredStructureBonus = 0.3;

    const cageFormersPresent = elements.filter(e => CAGE_FORMERS.includes(e));
    const cageGuestsPresent = elements.filter(e => CAGE_GUESTS.includes(e));
    const isCageLike = cageFormersPresent.length > 0 && cageGuestsPresent.length > 0
      && elements.length >= 2;
    const cageLatticeBonus = isCageLike ? 0.5 : 0;

    const looselyBonded = elements.filter(e => {
      const data = getElementData(e);
      if (!data) return false;
      const en = data.paulingElectronegativity ?? 2.0;
      const mass = data.atomicMass ?? 40;
      return (en < 1.2 && mass > 30) || LOOSELY_BONDED_INDICATORS.includes(e);
    });
    const looselyBondedFraction = looselyBonded.reduce((s, e) => s + (counts[e] ?? 0), 0) / totalAtoms;
    let looselyBondedBonus = Math.min(0.8, looselyBondedFraction * 1.5);

    let cageRattlerSynergy = 0;
    if (isCageLike && looselyBonded.length > 0) {
      cageRattlerSynergy = Math.min(0.5, looselyBondedFraction * 2.0);
    }

    const rawScore =
      softModeFraction * 1.5 +
      imaginaryModeFlag * 0.8 +
      phononVariance * 1.2 +
      anharmonicityContribution * 1.0 +
      lightElementBonus * 0.7 +
      layeredStructureBonus * 0.6 +
      cageLatticeBonus * 0.5 +
      looselyBondedBonus * 0.4 +
      cageRattlerSynergy * 0.8;

    const overallScore = 1.0 / (1.0 + Math.exp(-1.2 * (rawScore - 3.0)));

    const dosAtFermi = electronic.densityOfStatesAtFermi;
    const metallicity = electronic.metallicity;
    const lambda = coupling.lambda;
    const omegaLog = coupling.omegaLog;
    const nestingScore = electronic.nestingScore;
    const correlationStrength = electronic.correlationStrength;
    const vanHoveProximity = electronic.vanHoveProximity;

    const isMottInsulator = correlationStrength > 0.8 && metallicity < 0.3;
    let correlationComponent: number;
    if (isMottInsulator) {
      correlationComponent = -0.05;
    } else if (correlationStrength > 0.3 && correlationStrength < 0.8) {
      correlationComponent = 0.15;
    } else {
      correlationComponent = correlationStrength * 0.05;
    }

    const combinedElectronPhononScore =
      (dosAtFermi > 0 ? Math.min(1, dosAtFermi / 5.0) : 0) * 0.20 +
      metallicity * 0.15 +
      Math.min(1, lambda / 2.5) * 0.25 +
      nestingScore * 0.15 +
      correlationComponent +
      vanHoveProximity * 0.10;

    const softPhononCount = Math.round(softModeFraction * totalAtoms * 3);
    const avgPhononFrequency = logFreq * CM1_TO_THZ;
    const debyeT = phonon.debyeTemperature;

    let dynamicEffectStrength: "weak" | "moderate" | "strong" | "very-strong";
    if (overallScore < 0.25) dynamicEffectStrength = "weak";
    else if (overallScore < 0.50) dynamicEffectStrength = "moderate";
    else if (overallScore < 0.75) dynamicEffectStrength = "strong";
    else dynamicEffectStrength = "very-strong";

    let tcRelevance: string;
    if (isMottInsulator) {
      tcRelevance = "High correlation with low metallicity -- likely Mott insulator, phonon-mediated SC suppressed";
    } else if (overallScore >= 0.75) {
      tcRelevance = "Very strong dynamic lattice effects -- high phonon coupling potential, candidate warrants detailed Eliashberg analysis";
    } else if (overallScore >= 0.50) {
      tcRelevance = "Strong dynamic effects detected -- enhanced electron-phonon coupling likely, favorable for conventional SC";
    } else if (overallScore >= 0.25) {
      tcRelevance = "Moderate dynamic effects -- some phonon softening present, standard BCS regime expected";
    } else {
      tcRelevance = "Weak dynamic effects -- stiff lattice with limited phonon-mediated coupling potential";
    }

    return {
      formula,
      overallScore: Math.round(overallScore * 1000) / 1000,
      components: {
        softModeFraction: Math.round(softModeFraction * 1000) / 1000,
        imaginaryModeFlag,
        phononVariance: Math.round(phononVariance * 1000) / 1000,
        anharmonicityContribution: Math.round(anharmonicityContribution * 1000) / 1000,
        lightElementBonus: Math.round(lightElementBonus * 1000) / 1000,
        layeredStructureBonus: Math.round(layeredStructureBonus * 1000) / 1000,
        cageLatticeBonus: Math.round(cageLatticeBonus * 1000) / 1000,
        looselyBondedBonus: Math.round(looselyBondedBonus * 1000) / 1000,
      },
      electronPhononFeatures: {
        dosAtFermi: Math.round(dosAtFermi * 1000) / 1000,
        metallicity: Math.round(metallicity * 1000) / 1000,
        lambda: Math.round(lambda * 1000) / 1000,
        omegaLog: Math.round(omegaLog * 100) / 100,
        nestingScore: Math.round(nestingScore * 1000) / 1000,
        correlationStrength: Math.round(correlationStrength * 1000) / 1000,
        vanHoveProximity: Math.round(vanHoveProximity * 1000) / 1000,
        combinedElectronPhononScore: Math.round(combinedElectronPhononScore * 1000) / 1000,
      },
      dynamicEffectsProfile: {
        hasLightElements: lightEls.length > 0,
        lightElements: lightEls,
        isLayered,
        isCageLike,
        hasLooselyBonded: looselyBonded.length > 0,
        looselyBondedAtoms: looselyBonded,
        dynamicEffectStrength,
      },
      mlFeatures: {
        softPhononCount,
        avgPhononFrequency: Math.round(avgPhononFrequency * 1000) / 1000,
        phononVariance: Math.round(phononVariance * 1000) / 1000,
        instabilityFlag: phonon.hasImaginaryModes,
        debyeTemperature: debyeT,
        anharmonicityIndex: Math.round(phonon.anharmonicityIndex * 1000) / 1000,
        dynamicLatticeScore: Math.round(overallScore * 1000) / 1000,
      },
      tcRelevance,
      source: "physics-engine-composite",
    };
  } catch (e: any) {
    console.error(`[DynamicLatticeScore] Error for ${formula}:`, e.message);
    return null;
  }
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

const MULTI_OXIDATION_STATES: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  Sc: [3], Y: [3], La: [3], Ce: [3, 4], Pr: [3, 4], Nd: [3], Gd: [3],
  Ti: [3, 4], Zr: [4], Hf: [4],
  V: [3, 4, 5], Nb: [3, 5], Ta: [5],
  Cr: [2, 3, 6], Mo: [4, 6], W: [4, 6],
  Mn: [2, 3, 4, 7], Fe: [2, 3], Co: [2, 3], Ni: [2, 3], Cu: [1, 2, 3], Zn: [2],
  Ru: [3, 4], Rh: [3], Pd: [2, 4], Ir: [3, 4], Pt: [2, 4],
  Al: [3], Ga: [3], In: [3], Tl: [1, 3],
  B: [3], C: [4, -4], Si: [4], Ge: [2, 4], Sn: [2, 4], Pb: [2, 4],
  N: [-3], P: [-3, 5], As: [-3, 3, 5], Sb: [-3, 3, 5], Bi: [3, 5],
  O: [-2], S: [-2, 4, 6], Se: [-2, 4, 6], Te: [-2, 4, 6],
  F: [-1], Cl: [-1, 1, 5, 7], Br: [-1, 1, 5], I: [-1, 1, 5, 7],
  Re: [4, 7], Os: [4, 8], Hg: [1, 2], Ag: [1], Au: [1, 3],
};

const COMMON_OXIDATION_STATES: Record<string, number> = {};
for (const [el, states] of Object.entries(MULTI_OXIDATION_STATES)) {
  COMMON_OXIDATION_STATES[el] = states[0];
}

const MAGNETIC_IMPURITY_ELEMENTS = new Set(["Fe", "Co", "Ni", "Mn", "Cr"]);
const UNCONVENTIONAL_SC_HOSTS = new Set(["Cu", "Fe", "As", "La", "Y", "Ba", "Sr", "Bi", "Tl", "Hg", "Nd", "Gd", "Ce"]);

interface DopingPair { from: string; to: string; magneticImpurity?: boolean }

const ELECTRON_DOPING_PAIRS: DopingPair[] = [
  { from: "O", to: "F" },
  { from: "Fe", to: "Co", magneticImpurity: true },
  { from: "Ti", to: "Nb" },
  { from: "Ti", to: "V" },
  { from: "Cu", to: "Zn" },
  { from: "Ni", to: "Cu" },
  { from: "Mn", to: "Fe", magneticImpurity: true },
  { from: "Cr", to: "Mn", magneticImpurity: true },
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

const HOLE_DOPING_PAIRS: DopingPair[] = [
  { from: "La", to: "Sr" },
  { from: "Ba", to: "K" },
  { from: "Y", to: "Ca" },
  { from: "La", to: "Ba" },
  { from: "Ce", to: "La" },
  { from: "Sr", to: "K" },
  { from: "Ca", to: "Na" },
  { from: "Fe", to: "Mn", magneticImpurity: true },
  { from: "Nb", to: "Ti" },
  { from: "Co", to: "Fe", magneticImpurity: true },
  { from: "Cu", to: "Ni", magneticImpurity: true },
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

function hasMagneticImpurityPenalty(site: string, dopant: string, hostElements: string[]): boolean {
  if (!MAGNETIC_IMPURITY_ELEMENTS.has(dopant)) return false;
  const hostHasUnconventional = hostElements.some(e => UNCONVENTIONAL_SC_HOSTS.has(e));
  if (!hostHasUnconventional) return false;
  for (const pair of [...ELECTRON_DOPING_PAIRS, ...HOLE_DOPING_PAIRS]) {
    if (pair.from === site && pair.to === dopant && pair.magneticImpurity) return true;
  }
  return MAGNETIC_IMPURITY_ELEMENTS.has(dopant) && hostHasUnconventional;
}

function getOxidationState(el: string): number {
  return COMMON_OXIDATION_STATES[el] ?? 0;
}

function getNearestOxidationDelta(siteEl: string, dopantEl: string): number {
  const siteStates = MULTI_OXIDATION_STATES[siteEl] ?? [getOxidationState(siteEl)];
  const dopantStates = MULTI_OXIDATION_STATES[dopantEl] ?? [getOxidationState(dopantEl)];

  let minAbsDelta = Infinity;
  let bestDelta = 0;
  for (const sOx of siteStates) {
    for (const dOx of dopantStates) {
      const d = dOx - sOx;
      if (Math.abs(d) < minAbsDelta) {
        minAbsDelta = Math.abs(d);
        bestDelta = d;
      }
    }
  }
  return bestDelta;
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

  const delta = getNearestOxidationDelta(site, dopant);

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

const MAX_CARRIER_DENSITY = 5e22;

function computeCarrierDensity(valenceChange: number, nDopedAtoms: number, cellVolumeNm3: number): number {
  if (cellVolumeNm3 <= 0.001) return 0;
  const totalChargeChange = Math.abs(valenceChange) * nDopedAtoms;
  const volumeCm3 = cellVolumeNm3 * 1e-21;
  const raw = totalChargeChange / volumeCm3;
  return Math.min(raw, MAX_CARRIER_DENSITY);
}

function estimatePackingFractionLocal(counts: Record<string, number>): number {
  let totalAtomVol = 0;
  let totalRadius = 0;
  let totalCount = 0;
  for (const [el, n] of Object.entries(counts)) {
    const data = getElementData(el);
    if (!data) continue;
    const r = data.atomicRadius || 150;
    totalAtomVol += (4 / 3) * Math.PI * Math.pow(r, 3) * n;
    totalRadius += r * n;
    totalCount += n;
  }
  if (totalCount === 0) return 0.5;
  const avgR = totalRadius / totalCount;
  const volPerAtom = (2 * avgR) ** 3;
  const cellVol = totalCount * volPerAtom;
  if (cellVol <= 0) return 0.5;
  return Math.max(0.1, Math.min(0.85, totalAtomVol / cellVol));
}

function hasHighVHSProximity(formula: string): boolean {
  try {
    const electronic = computeElectronicStructure(formula);
    return (electronic?.vanHoveProximity ?? 0) > 0.6;
  } catch {
    return false;
  }
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

const ANION_VACANCY_TARGETS = ["O", "F", "S", "Se", "Te", "N", "Cl"];
const CATION_VACANCY_TARGETS = ["Cu", "La", "Sr", "Ba", "Y", "Ca", "Bi", "Tl", "Hg", "Nd", "Gd", "Ce", "Fe", "Ti", "Nb", "V", "Mo", "W"];
const VACANCY_TARGETS = [...ANION_VACANCY_TARGETS, ...CATION_VACANCY_TARGETS];

const DOPING_FRACTIONS = [0.02, 0.05, 0.10, 0.15, 0.20];

const SEARCH_LIMITS = {
  maxDopantsPerMaterial: 2,
  maxDopingFraction: 0.20,
  maxSupercellAtoms: 128,
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

function getSupercellMultiplier(totalAtoms: number, targetFraction?: number): number {
  let mult: number;
  if (totalAtoms <= 4) mult = 8;
  else if (totalAtoms <= 8) mult = 4;
  else if (totalAtoms <= 12) mult = 2;
  else mult = 1;

  if (targetFraction && targetFraction > 0) {
    const minSitesNeeded = Math.ceil(1 / targetFraction);
    while (totalAtoms * mult < minSitesNeeded && mult < 16) {
      mult *= 2;
    }
  }

  if (totalAtoms * mult > 128) {
    mult = Math.max(1, Math.floor(128 / totalAtoms));
  }

  return mult;
}

function getDopantPriority(site: string, dopant: string, elements: string[]): number {
  if (elements.includes(dopant)) return -1;

  let basePriority = 3;
  for (const pair of ELECTRON_DOPING_PAIRS) {
    if (pair.from === site && pair.to === dopant) { basePriority = 10; break; }
  }
  if (basePriority < 10) {
    for (const pair of HOLE_DOPING_PAIRS) {
      if (pair.from === site && pair.to === dopant) { basePriority = 10; break; }
    }
  }

  if (basePriority < 10) {
    const siteOx = getOxidationState(site);
    const dopantOx = getOxidationState(dopant);
    const delta = Math.abs(dopantOx - siteOx);
    if (delta === 1) basePriority = 8;
    else if (delta === 0) basePriority = 5;
    else if (delta === 2) basePriority = 6;
  }

  if (hasMagneticImpurityPenalty(site, dopant, elements)) {
    basePriority = Math.max(1, basePriority - 4);
  }

  return basePriority;
}

const VHS_FINE_FRACTIONS = [0.01, 0.03];

function generateSubstitutionalVariants(
  formula: string,
  counts: Record<string, number>,
  maxVariants: number = 8,
  highVHS: boolean = false
): DopingSpec[] {
  const variants: DopingSpec[] = [];
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);
  const supercellMult = getSupercellMultiplier(totalAtoms);
  const cellVolume = estimateUnitCellVolume(counts);

  const baseFractions = highVHS
    ? [...VHS_FINE_FRACTIONS, ...DOPING_FRACTIONS]
    : DOPING_FRACTIONS;

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
      if (radiusDiff > 0.30) continue;

      const isHighStrain = radiusDiff > 0.15;
      const { character, valenceChange } = classifyDopingCharacter(site, dopant, "substitutional");

      let fractions: number[];
      if (isHighStrain) {
        fractions = baseFractions.filter(f => f <= 0.05);
      } else if (radiusDiff < 0.10) {
        fractions = baseFractions;
      } else {
        fractions = baseFractions.filter(f => f <= 0.10);
      }

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

        const reduced = reduceToFormulaUnit(supercellCounts, supercellMult);

        const resultFormula = countsToFormula(reduced);
        if (!isValidFormula(resultFormula)) continue;

        const supercellVolume = cellVolume * supercellMult;
        const carrierDensity = computeCarrierDensity(valenceChange, nReplace, supercellVolume);

        const dopingLabel = character === "electron" ? "electron-doping" :
          character === "hole" ? "hole-doping" : "isovalent";
        const chargeInfo = valenceChange !== 0
          ? ` [${dopingLabel}: delta_q=${valenceChange > 0 ? "+" : ""}${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3]`
          : " [isovalent substitution]";

        const strainNote = isHighStrain ? " [HIGH STRAIN -- reduced stability expected]" : "";
        const rationale = `${dopant} substitution at ${site} site (${(fraction * 100).toFixed(0)}%): `
          + `radius match ${((1 - radiusDiff) * 100).toFixed(0)}%, `
          + `replaces ${nReplace}/${sitesInSupercell} ${site} atoms in ${supercellMult > 1 ? supercellMult + "x supercell" : "unit cell"}`
          + chargeInfo + strainNote;

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

    const isCationVacancy = CATION_VACANCY_TARGETS.includes(site);
    const { character, valenceChange } = classifyDopingCharacter(site, "", "vacancy");

    const fracs = isCationVacancy ? [0.02, 0.05, 0.10] : [0.05, 0.10, 0.15];
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

      const reduced = reduceToFormulaUnit(supercellCounts, supercellMult);

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
        rationale: `${site} ${isCationVacancy ? "cation" : "anion"} vacancy (${(fraction * 100).toFixed(0)}%): removed ${nRemove}/${sitesInSupercell} ${site} atoms — creates ${carrierType} carriers (delta_q=${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3)`,
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

  const packingFraction = estimatePackingFractionLocal(counts);
  const isHighDensity = packingFraction > 0.68;

  if (isHighDensity && structureType === "general") {
    return variants;
  }

  const dopantPool = INTERSTITIAL_DOPANTS[structureType] || INTERSTITIAL_DOPANTS.general;
  const availableDopants = dopantPool.filter(d => !elements.includes(d));

  for (const dopant of availableDopants) {
    if (variants.length >= maxVariants) break;

    const { character, valenceChange } = classifyDopingCharacter("", dopant, "interstitial");

    const voidFraction = Math.max(0, 1 - packingFraction);

    let fracs: number[];
    if (packingFraction > 0.68) {
      fracs = [0.02];
    } else if (packingFraction > 0.60) {
      fracs = [0.03, 0.05];
    } else {
      fracs = [0.05, 0.10];
    }
    for (const fraction of fracs) {
      if (variants.length >= maxVariants) break;

      if (fraction > voidFraction * 0.5) continue;

      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      const totalInSupercell = getTotalAtoms(supercellCounts);
      const maxInsertByVolume = Math.max(1, Math.floor(totalInSupercell * voidFraction * 0.3));
      const nInsert = Math.min(
        Math.max(1, Math.round(totalInSupercell * fraction)),
        maxInsertByVolume
      );

      supercellCounts[dopant] = (supercellCounts[dopant] || 0) + nInsert;

      const reduced = reduceToFormulaUnit(supercellCounts, supercellMult);

      const resultFormula = countsToFormula(reduced);
      if (!isValidFormula(resultFormula)) continue;

      const totalNew = getTotalAtoms(reduced);
      if (totalNew > SEARCH_LIMITS.maxSupercellAtoms) continue;

      const supercellVolume = cellVolume * supercellMult;
      const carrierDensity = computeCarrierDensity(valenceChange, nInsert, supercellVolume);

      const needsSupercell = supercellMult > 1 && nInsert < supercellMult;

      variants.push({
        type: "interstitial",
        base: formula,
        dopant,
        fraction,
        resultFormula: normalizeFormula(resultFormula),
        supercellSize: supercellMult,
        minSupercellSize: needsSupercell ? supercellMult : 1,
        rationale: `${dopant} interstitial insertion (${(fraction * 100).toFixed(0)}%): ${nInsert} atoms into ${structureType} structure (APF=${packingFraction.toFixed(2)}, void=${(voidFraction * 100).toFixed(0)}%) — electron-doping (delta_q=+${valenceChange}, n=${carrierDensity.toExponential(1)} cm^-3)${needsSupercell ? ` [requires ${supercellMult}x supercell]` : ""}`,
        dopingCharacter: character,
        valenceChange,
        carrierDensity,
      });
    }
  }

  return variants;
}

function reduceToFormulaUnit(supercellCounts: Record<string, number>, supercellMult: number): Record<string, number> {
  const intCounts = Object.values(supercellCounts).filter(v => v > 0).map(v => Math.round(v));
  const gcd = findGCD(intCounts);

  const safeGcd = supercellMult > 1
    ? findGCD([gcd, supercellMult])
    : gcd;

  if (safeGcd > 1) {
    const reduced: Record<string, number> = {};
    for (const [el, n] of Object.entries(supercellCounts)) {
      if (n > 0) reduced[el] = n / safeGcd;
    }
    const total = Object.values(reduced).reduce((s, n) => s + n, 0);
    if (total <= 128) return reduced;
  }

  if (supercellMult > 1) {
    const perUnit: Record<string, number> = {};
    for (const [el, n] of Object.entries(supercellCounts)) {
      if (n > 0) {
        const v = n / supercellMult;
        perUnit[el] = Math.round(v * 1000) / 1000;
      }
    }
    const total = Object.values(perUnit).reduce((s, n) => s + n, 0);
    if (total <= 128) return perUnit;
  }

  const reduced: Record<string, number> = {};
  for (const [el, n] of Object.entries(supercellCounts)) {
    if (n > 0) reduced[el] = n / Math.max(1, safeGcd);
  }
  return reduced;
}

function findGCD(nums: number[]): number {
  if (nums.length === 0) return 1;
  for (const n of nums) {
    if (n <= 0) return 1;
    if (Math.abs(n - Math.round(n)) > 0.01) return 1;
  }
  const gcd2 = (a: number, b: number): number => {
    a = Math.abs(Math.round(a));
    b = Math.abs(Math.round(b));
    if (a === 0) return b || 1;
    if (b === 0) return a;
    while (b > 0) {
      [a, b] = [b, a % b];
    }
    return a || 1;
  };
  return nums.reduce((acc, n) => gcd2(Math.round(acc), Math.round(n)), Math.round(nums[0]));
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

          const ri = (getElementData(atoms[i].element)?.atomicRadius ?? 130) * 0.77 / 100;
          const rj = (getElementData(atoms[j].element)?.atomicRadius ?? 130) * 0.77 / 100;
          const bondCutoff = (ri + rj) * 1.3;

          if (d < bondCutoff && d > 0.3) {
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

    const latticeCollapse = maxDisplacement > 1.5 || Math.abs(volumeChange) > 30;
    if (latticeCollapse) {
      console.log(`[Doping] Lattice collapse detected for ${formula}: maxDisp=${maxDisplacement.toFixed(3)} Å, volChange=${volumeChange.toFixed(1)}%`);
    }

    let phononAnalysis: HessianPhononAnalysis | null = null;
    if (!latticeCollapse) {
      try {
        phononAnalysis = await analyzeHessianPhonons(formula);
      } catch (phErr) {
        console.log(`[Doping] Hessian phonon analysis failed for ${formula}: ${phErr instanceof Error ? phErr.message.slice(0, 80) : String(phErr).slice(0, 80)}`);
      }
    }

    return {
      converged: optResult.converged && !latticeCollapse,
      latticeStrain,
      bondVariance,
      meanDisplacement,
      maxDisplacement,
      volumeChange,
      energyPerAtom,
      wallTimeMs: optResult.wallTimeSeconds * 1000,
      latticeCollapse,
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

  const highVHS = hasHighVHSProximity(formula);

  const subMax = Math.ceil(maxTotal * 0.5);
  const vacMax = Math.ceil(maxTotal * 0.25);
  const intMax = maxTotal - subMax - vacMax;

  const substitutional = generateSubstitutionalVariants(formula, counts, subMax, highVHS);
  const vacancy = generateVacancyVariants(formula, counts, vacMax);
  const interstitial = generateInterstitialVariants(formula, counts, intMax);

  const allVariants = [...substitutional, ...vacancy, ...interstitial];

  const seen = new Set<string>();
  seen.add(normalizeFormula(formula));
  let valenceRejected = 0;
  let capRejected = 0;
  const valenceRejectedByFamily: Record<string, number> = {};
  const unique = allVariants.filter(v => {
    if (seen.has(v.resultFormula)) return false;
    seen.add(v.resultFormula);
    if (!passesElementCountCap(v.resultFormula)) {
      capRejected++;
      return false;
    }
    const valenceCheck = checkValenceSumRule(v.resultFormula);
    if (!valenceCheck.pass) {
      valenceRejected++;
      const family = classifyLayeredOrCage(v.resultFormula);
      valenceRejectedByFamily[family] = (valenceRejectedByFamily[family] || 0) + 1;
      return false;
    }
    return true;
  });
  if (valenceRejected > 0) {
    const familyBreakdown = Object.entries(valenceRejectedByFamily)
      .map(([fam, count]) => `${fam}:${count}`)
      .join(", ");
    console.log(`[Doping] Valence-sum gate rejected ${valenceRejected}/${allVariants.length} variants from ${formula} [${familyBreakdown}]`);
  }

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

  const perBase: DopingSpec[][] = [];
  for (const base of formulas) {
    const result = generateDopedVariants(base, maxVariantsPerBase);
    const filtered = result.variants.filter(v =>
      !(excludeSet && excludeSet.has(v.resultFormula))
    );
    if (filtered.length > 0) perBase.push(filtered);
  }

  const seen = new Set<string>();
  let round = 0;
  let anyLeft = true;
  while (dopedFormulas.length < maxTotalDoped && anyLeft) {
    anyLeft = false;
    for (const variants of perBase) {
      if (dopedFormulas.length >= maxTotalDoped) break;
      if (round >= variants.length) continue;
      anyLeft = true;
      const v = variants[round];
      if (seen.has(v.resultFormula)) continue;
      seen.add(v.resultFormula);

      dopedFormulas.push(v.resultFormula);
      specs.push(v);
      if (v.type === "substitutional") subCount++;
      else if (v.type === "vacancy") vacCount++;
      else intCount++;
      if (v.dopingCharacter === "electron" || v.dopingCharacter === "interstitial-electron") eDoped++;
      if (v.dopingCharacter === "hole" || v.dopingCharacter === "vacancy-hole") hDoped++;
    }
    round++;
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

  const baseDOS = Math.max(0, baseElectronic?.densityOfStatesAtFermi ?? 0);
  const dopedDOS = Math.max(0, dopedElectronic?.densityOfStatesAtFermi ?? 0);
  const dosIncrease = (dopedDOS - baseDOS) / (baseDOS + 0.1);

  const baseElements = parseFormulaElements(baseFormula);
  const dopedElements = parseFormulaElements(dopedFormula);

  const baseCounts = parseFormulaCounts(baseFormula);
  const baseTotalAtoms = getTotalAtoms(baseCounts);
  const dopedCounts = parseFormulaCounts(dopedFormula);
  const dopedTotalAtoms = getTotalAtoms(dopedCounts);

  let baseStonerMax = 0;
  for (const el of baseElements) {
    const I = getStonerParameter(el) ?? 0;
    if (I <= 0) continue;
    const elFrac = (baseCounts[el] || 0) / baseTotalAtoms;
    const pdos = baseDOS * elFrac;
    baseStonerMax = Math.max(baseStonerMax, I * pdos);
  }
  let dopedStonerMax = 0;
  for (const el of dopedElements) {
    const I = getStonerParameter(el) ?? 0;
    if (I <= 0) continue;
    const elFrac = (dopedCounts[el] || 0) / dopedTotalAtoms;
    const pdos = dopedDOS * elFrac;
    dopedStonerMax = Math.max(dopedStonerMax, I * pdos);
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
    const analysis = processPhononStability(dopedHessian, dopedFormula);
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
  const normalizedFormula = normalizeFormula(formula);
  const superconMatch = SUPERCON_TRAINING_DATA.find(e => normalizeFormula(e.formula) === normalizedFormula);
  if (superconMatch?.crystalSystem) return superconMatch.crystalSystem;

  const crystalEntry = getEntryByFormula(formula) || getEntryByFormula(normalizedFormula);
  if (crystalEntry?.crystalSystem && crystalEntry.crystalSystem !== "unknown") return crystalEntry.crystalSystem;

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const ratios = Object.values(counts).map(n => n / totalAtoms);
  const allEqual = ratios.every(r => Math.abs(r - ratios[0]) < 0.05);

  if (elements.length === 1) return "cubic";

  if (elements.length === 2) {
    if (allEqual) return "cubic";
    const ratio = Math.max(...Object.values(counts)) / Math.min(...Object.values(counts));
    if (ratio <= 1.5) return "cubic";
    if (ratio <= 3) return "tetragonal";
    return "hexagonal";
  }

  const hasOxygen = elements.includes("O");
  const oxygenFrac = hasOxygen ? (counts["O"] || 0) / totalAtoms : 0;

  if (hasOxygen && oxygenFrac > 0.5) {
    if (elements.length >= 4) return "orthorhombic";
    return "tetragonal";
  }

  if (elements.length === 3) {
    if (totalAtoms <= 5) return "tetragonal";
    return "orthorhombic";
  }

  if (elements.length >= 5) return "monoclinic";
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
    const cellVolume = estimateUnitCellVolume(counts);
    const totalAtoms = getTotalAtoms(counts);

    for (const fraction of clampedFracs) {
      const siteCount = counts[pair.site] || 0;
      const supercellMult = getSupercellMultiplier(totalAtoms, siteCount > 0 ? fraction / (siteCount / totalAtoms) : fraction);

      const supercellCounts: Record<string, number> = {};
      for (const [el, n] of Object.entries(counts)) {
        supercellCounts[el] = n * supercellMult;
      }

      let nChanged = 0;
      if (pair.type === "substitutional") {
        const sitesInSupercell = supercellCounts[pair.site];
        nChanged = Math.round(sitesInSupercell * fraction);
        if (nChanged < 1) continue;
        if (nChanged >= sitesInSupercell) continue;
        const actualFraction = nChanged / sitesInSupercell;
        if (Math.abs(actualFraction - fraction) / fraction > 0.5) continue;
        supercellCounts[pair.site] -= nChanged;
        supercellCounts[pair.dopant] = (supercellCounts[pair.dopant] || 0) + nChanged;
      } else {
        const sitesInSupercell = supercellCounts[pair.site];
        nChanged = Math.round(sitesInSupercell * fraction);
        if (nChanged < 1) continue;
        if (nChanged >= sitesInSupercell) continue;
        const actualFraction = nChanged / sitesInSupercell;
        if (Math.abs(actualFraction - fraction) / fraction > 0.5) continue;
        supercellCounts[pair.site] -= nChanged;
      }

      const reduced = reduceToFormulaUnit(supercellCounts, supercellMult);

      const resultFormula = normalizeFormula(countsToFormula(reduced));
      if (!isValidFormula(resultFormula)) continue;
      if (getTotalAtoms(reduced) > SEARCH_LIMITS.maxSupercellAtoms) continue;

      const { character, valenceChange } = pair.type === "vacancy"
        ? classifyDopingCharacter(pair.site, "", "vacancy")
        : classifyDopingCharacter(pair.site, pair.dopant, "substitutional");

      const reducedTotalAtoms = getTotalAtoms(reduced);
      const carrierDensityPerFU = reducedTotalAtoms > 0
        ? Math.abs(valenceChange) * (nChanged / supercellMult) / (cellVolume * 1e-21)
        : computeCarrierDensity(valenceChange, nChanged, cellVolume * supercellMult);

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
          carrierDensity: carrierDensityPerFU,
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
      let bestIdx = tcs.indexOf(Math.max(...tcs));
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

      if (trend === "peaked") {
        const peakFraction = levels[bestIdx].fraction;
        const zoomStep = 0.02;
        const zoomFracs: number[] = [];
        for (let d = -3; d <= 3; d++) {
          if (d === 0) continue;
          const zf = peakFraction + d * zoomStep;
          if (zf > 0 && zf <= SEARCH_LIMITS.maxDopingFraction) {
            const alreadyExists = levels.some(l => Math.abs(l.fraction - zf) < 0.005);
            if (!alreadyExists) zoomFracs.push(zf);
          }
        }

        for (const fraction of zoomFracs) {
          const siteCount = counts[pair.site] || 0;
          const supercellMult = getSupercellMultiplier(totalAtoms, siteCount > 0 ? fraction / (siteCount / totalAtoms) : fraction);
          const supercellCounts: Record<string, number> = {};
          for (const [el, n] of Object.entries(counts)) {
            supercellCounts[el] = n * supercellMult;
          }

          let nChanged = 0;
          if (pair.type === "substitutional") {
            const sitesInSupercell = supercellCounts[pair.site];
            nChanged = Math.round(sitesInSupercell * fraction);
            if (nChanged < 1 || nChanged >= sitesInSupercell) continue;
            const actualFraction = nChanged / sitesInSupercell;
            if (Math.abs(actualFraction - fraction) / fraction > 0.5) continue;
            supercellCounts[pair.site] -= nChanged;
            supercellCounts[pair.dopant] = (supercellCounts[pair.dopant] || 0) + nChanged;
          } else {
            const sitesInSupercell = supercellCounts[pair.site];
            nChanged = Math.round(sitesInSupercell * fraction);
            if (nChanged < 1 || nChanged >= sitesInSupercell) continue;
            const actualFraction = nChanged / sitesInSupercell;
            if (Math.abs(actualFraction - fraction) / fraction > 0.5) continue;
            supercellCounts[pair.site] -= nChanged;
          }

          const reduced = reduceToFormulaUnit(supercellCounts, supercellMult);
          const resultFormula = normalizeFormula(countsToFormula(reduced));
          if (!isValidFormula(resultFormula)) continue;
          if (getTotalAtoms(reduced) > SEARCH_LIMITS.maxSupercellAtoms) continue;

          const { character, valenceChange } = pair.type === "vacancy"
            ? classifyDopingCharacter(pair.site, "", "vacancy")
            : classifyDopingCharacter(pair.site, pair.dopant, "substitutional");

          const reducedTotalAtoms = getTotalAtoms(reduced);
          const carrierDensityPerFU = reducedTotalAtoms > 0
            ? Math.abs(valenceChange) * (nChanged / supercellMult) / (cellVolume * 1e-21)
            : computeCarrierDensity(valenceChange, nChanged, cellVolume * supercellMult);

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
              carrierDensity: carrierDensityPerFU,
              dopingCharacter: character,
              scSignals,
            });

            if (tc > globalBestTc) {
              globalBestTc = tc;
              globalBest = { formula: resultFormula, tc, fraction, dopant: pair.dopant || `${pair.site}-vacancy` };
            }
          } catch { /* skip */ }
        }

        const seenFormulas = new Set<string>();
        const dedupedLevels = levels.filter(l => {
          if (seenFormulas.has(l.resultFormula)) return false;
          seenFormulas.add(l.resultFormula);
          return true;
        });
        levels.length = 0;
        levels.push(...dedupedLevels);
        levels.sort((a, b) => a.fraction - b.fraction);

        const updatedTcs = levels.map(l => l.predictedTc);
        bestIdx = updatedTcs.indexOf(Math.max(...updatedTcs));

        if (updatedTcs.length >= 3) {
          const zFirst = updatedTcs[0];
          const zLast = updatedTcs[updatedTcs.length - 1];
          const zPeak = Math.max(...updatedTcs);
          const zPeakIdx = updatedTcs.indexOf(zPeak);
          if (zPeakIdx > 0 && zPeakIdx < updatedTcs.length - 1 && zPeak > zFirst * 1.05 && zPeak > zLast * 1.05) {
            trend = "peaked";
          } else if (zLast > zFirst * 1.1) {
            trend = "increasing";
          } else if (zLast < zFirst * 0.9) {
            trend = "decreasing";
          } else {
            trend = "flat";
          }
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
