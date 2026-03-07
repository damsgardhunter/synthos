import { fetchSummary, fetchElasticity, fetchElectronicStructure, fetchThermo, fetchPhonon, isApiAvailable } from "./materials-project-client";
import type { MPSummaryData, MPElasticityData, MPElectronicStructureData, MPThermoData, MPPhononData } from "./materials-project-client";
import { fetchAflowData, fetchAflowDFTData } from "./aflow-client";
import type { AflowDFTData, AflowEntry } from "./aflow-client";
import {
  getElementData,
  getCompositionWeightedProperty,
  getDebyeTemperature,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "./elemental-data";
import { runXTBEnrichment, isDFTAvailable } from "../dft/qe-dft-engine";
import type { XTBEnrichedFeatures, PhononStability } from "../dft/qe-dft-engine";
import type { FiniteDisplacementPhononResult } from "../dft/phonon-calculator";
import { estimatePhononStability } from "./crystal-prototypes";

export type DFTSource = "dft-mp" | "dft-aflow" | "dft-xtb" | "analytical";

export interface DFTResolvedFeature<T> {
  value: T;
  source: DFTSource;
}

export interface DFTResolvedFeatures {
  bandGap: DFTResolvedFeature<number>;
  isMetallic: DFTResolvedFeature<boolean>;
  dosAtFermi: DFTResolvedFeature<number | null>;
  debyeTemp: DFTResolvedFeature<number>;
  bulkModulus: DFTResolvedFeature<number>;
  formationEnergy: DFTResolvedFeature<number>;
  energyAboveHull: DFTResolvedFeature<number | null>;
  phononFreqMax: DFTResolvedFeature<number | null>;
  thermalConductivity: DFTResolvedFeature<number | null>;
  phononStability: PhononStability | null;
  finiteDisplacementPhonons: FiniteDisplacementPhononResult | null;
  dftCoverage: number;
  sources: { mp: boolean; aflow: boolean };
}

interface RawDFTSources {
  mpSummary: MPSummaryData | null;
  mpElasticity: MPElasticityData | null;
  mpElectronic: MPElectronicStructureData | null;
  mpThermo: MPThermoData | null;
  mpPhonon: MPPhononData | null;
  aflowEntry: AflowEntry | null;
  aflowDFT: AflowDFTData | null;
}

function normalizeFormula(formula: string): string {
  if (typeof formula !== "string") formula = String(formula ?? "");
  let normalized = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  const parts: { el: string; count: number }[] = [];
  let match;
  while ((match = regex.exec(normalized)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    const existing = parts.find(p => p.el === el);
    if (existing) {
      existing.count += num;
    } else {
      parts.push({ el, count: num });
    }
  }
  
  const ELECTRONEG: Record<string, number> = {
    H: 2.20, Li: 0.98, Be: 1.57, B: 2.04, C: 2.55, N: 3.04, O: 3.44, F: 3.98,
    Na: 0.93, Mg: 1.31, Al: 1.61, Si: 1.90, P: 2.19, S: 2.58, Cl: 3.16,
    K: 0.82, Ca: 1.00, Sc: 1.36, Ti: 1.54, V: 1.63, Cr: 1.66, Mn: 1.55,
    Fe: 1.83, Co: 1.88, Ni: 1.91, Cu: 1.90, Zn: 1.65, Ga: 1.81, Ge: 2.01,
    As: 2.18, Se: 2.55, Br: 2.96, Rb: 0.82, Sr: 0.95, Y: 1.22, Zr: 1.33,
    Nb: 1.60, Mo: 1.80, Ru: 2.20, Rh: 2.28, Pd: 2.20, Ag: 1.93, Cd: 1.69,
    In: 1.78, Sn: 1.96, Sb: 2.05, Te: 2.10, I: 2.66, Cs: 0.79, Ba: 0.89,
    La: 1.10, Ce: 1.12, Hf: 1.30, Ta: 1.50, W: 1.70, Re: 1.90, Os: 2.20,
    Ir: 2.20, Pt: 2.28, Au: 2.54, Tl: 1.62, Pb: 2.33, Bi: 2.02, Th: 1.30, U: 1.38,
  };
  parts.sort((a, b) => (ELECTRONEG[a.el] ?? 2.0) - (ELECTRONEG[b.el] ?? 2.0));
  
  return parts.map(p => p.count === 1 ? p.el : `${p.el}${p.count}`).join("");
}

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
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

function computeAnalyticalFallbacks(formula: string): {
  bandGap: number;
  isMetallic: boolean;
  debyeTemp: number;
  bulkModulus: number;
  formationEnergy: number;
  dosAtFermi: number | null;
  omegaLog: number;
  lambda: number | null;
  muStar: number;
  thermalConductivity: number | null;
  phononFreqMax: number;
  density: number;
  atomicPackingFraction: number;
  averagePhononFreq: number;
  estimatorCoverage: number;
} {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalFrac = metalElements.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const isMetallic = metalFrac > 0.3 || metalElements.some(e => isTransitionMetal(e));

  let bandGap = 0;
  if (!isMetallic) {
    const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
    const enSpread = enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;
    bandGap = Math.max(0, enSpread * 0.8);
  }

  const bulkWeighted = getCompositionWeightedProperty(counts, "bulkModulus");
  const bulkModulus = bulkWeighted && bulkWeighted > 0 ? bulkWeighted : estimateBulkModulus(elements, counts);

  const avgMass = getCompositionWeightedProperty(counts, "atomicMass") || 50;
  const density = estimateDensity(elements, counts, totalAtoms);

  const debyeWeighted = getCompositionWeightedProperty(counts, "debyeTemperature");
  let debyeTemp = debyeWeighted && debyeWeighted > 0 ? debyeWeighted : 300;
  if ((!debyeWeighted || debyeWeighted <= 0) && bulkModulus > 0 && density > 0) {
    debyeTemp = 41.6 * Math.sqrt(bulkModulus / density);
    debyeTemp = Math.max(50, Math.min(2500, debyeTemp));
  }

  let formationEnergy = 0;
  if (elements.length >= 2) {
    const enValues = elements.map(e => getElementData(e)?.paulingElectronegativity ?? 1.5);
    const enSpread = Math.max(...enValues) - Math.min(...enValues);
    formationEnergy = -0.5 * enSpread;
  }

  let dosAtFermi: number | null = null;
  if (isMetallic) {
    const gammaAvg = getCompositionWeightedProperty(counts, "sommerfeldGamma");
    if (gammaAvg && gammaAvg > 0) {
      dosAtFermi = gammaAvg / 2.359;
    } else {
      dosAtFermi = estimateDOSAtFermi(elements, counts, totalAtoms, isMetallic);
    }
  }

  const omegaLog = 0.5 * debyeTemp;

  const averagePhononFreq = estimateAveragePhononFreq(debyeTemp, avgMass);

  let lambda: number | null = null;
  if (isMetallic && dosAtFermi != null && dosAtFermi > 0) {
    const hopfieldEta = getCompositionWeightedProperty(counts, "mcMillanHopfieldEta");
    if (hopfieldEta && hopfieldEta > 0) {
      const omega2 = (debyeTemp * 0.695) ** 2;
      lambda = (dosAtFermi * hopfieldEta) / (avgMass * omega2) * 1e4;
      lambda = Math.min(Math.max(lambda, 0.1), 3.0);
    }
  }

  const muStar = 0.13 - 0.01 * metalFrac;

  let thermalConductivity: number | null = null;
  if (isMetallic) {
    const lorenzNumber = 2.44e-8;
    const T = 300;
    const avgConductivity = elements.reduce((sum, el) => {
      const data = getElementData(el);
      if (data && data.sommerfeldGamma != null && data.sommerfeldGamma > 0) {
        return sum + (counts[el] || 0);
      }
      return sum;
    }, 0);
    if (avgConductivity > 0) {
      const bulkEst = bulkModulus > 0 ? bulkModulus : 100;
      thermalConductivity = lorenzNumber * bulkEst * 1e9 * T / 1e6;
      thermalConductivity = Math.min(thermalConductivity, 500);
    }
  }

  const phononFreqMax = 1.2 * debyeTemp;

  const atomicPackingFraction = estimateAtomicPacking(elements, counts);

  let coveredProperties = 0;
  const totalProperties = 11;
  if (bandGap >= 0) coveredProperties++;
  coveredProperties++;
  if (debyeTemp > 0) coveredProperties++;
  if (bulkModulus > 0) coveredProperties++;
  if (formationEnergy !== 0) coveredProperties++;
  if (dosAtFermi != null) coveredProperties++;
  if (thermalConductivity != null) coveredProperties++;
  if (phononFreqMax > 0) coveredProperties++;
  if (density > 0) coveredProperties++;
  if (atomicPackingFraction > 0) coveredProperties++;
  if (averagePhononFreq > 0) coveredProperties++;
  const estimatorCoverage = coveredProperties / totalProperties;

  return { bandGap, isMetallic, debyeTemp, bulkModulus, formationEnergy, dosAtFermi, omegaLog, lambda, muStar, thermalConductivity, phononFreqMax, density, atomicPackingFraction, averagePhononFreq, estimatorCoverage };
}

function estimateBulkModulus(elements: string[], counts: Record<string, number>): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  let totalB = 0;
  let weight = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data?.bulkModulus && data.bulkModulus > 0) {
      totalB += data.bulkModulus * (counts[el] || 1);
      weight += counts[el] || 1;
    } else if (data?.meltingPoint && data.meltingPoint > 0) {
      const estB = data.meltingPoint * 0.07;
      totalB += estB * (counts[el] || 1);
      weight += counts[el] || 1;
    }
  }
  if (weight > 0) return totalB / weight;
  return 100;
}

function estimateDensity(elements: string[], counts: Record<string, number>, totalAtoms: number): number {
  let totalMass = 0;
  let totalVolume = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const n = counts[el] || 1;
    totalMass += data.atomicMass * n;
    const r_cm = (data.atomicRadius || 150) * 1e-10;
    totalVolume += (4 / 3) * Math.PI * Math.pow(r_cm, 3) * n;
  }
  if (totalVolume <= 0) return 5.0;
  const packingFraction = 0.68;
  const cellVolume = totalVolume / packingFraction;
  const amu_to_g = 1.66054e-24;
  const density = (totalMass * amu_to_g) / cellVolume;
  return Math.max(1.0, Math.min(25.0, density));
}

function estimateDOSAtFermi(elements: string[], counts: Record<string, number>, totalAtoms: number, isMetallic: boolean): number | null {
  if (!isMetallic) return null;
  let totalValence = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) {
      totalValence += data.valenceElectrons * (counts[el] || 1);
    }
  }
  const avgValence = totalValence / totalAtoms;
  const baseDOS = 0.3 + avgValence * 0.15;
  const tmCount = elements.filter(e => isTransitionMetal(e)).reduce((s, e) => s + (counts[e] || 0), 0);
  const tmFrac = tmCount / totalAtoms;
  const tmBoost = tmFrac > 0.3 ? 1.5 : 1.0;
  return Math.min(5.0, baseDOS * tmBoost);
}

function estimateAveragePhononFreq(debyeTemp: number, avgMass: number): number {
  const kB = 8.617e-5;
  const omegaD_meV = kB * debyeTemp * 1000;
  const avgFreq = omegaD_meV * 0.7 / Math.sqrt(avgMass / 50);
  return Math.max(1, Math.min(200, avgFreq));
}

function estimateAtomicPacking(elements: string[], counts: Record<string, number>): number {
  let totalAtomVol = 0;
  let totalRadius = 0;
  let totalCount = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const n = counts[el] || 1;
    const r = data.atomicRadius || 150;
    totalAtomVol += (4 / 3) * Math.PI * Math.pow(r, 3) * n;
    totalRadius += r * n;
    totalCount += n;
  }
  if (totalCount === 0) return 0.5;
  const avgR = totalRadius / totalCount;
  const cellVol = totalCount * Math.pow(2.2 * avgR, 3);
  if (cellVol <= 0) return 0.5;
  const apf = totalAtomVol / cellVol;
  return Math.max(0.1, Math.min(0.85, apf));
}

async function fetchAllDFTSources(formula: string): Promise<RawDFTSources> {
  const mpAvailable = isApiAvailable();
  const normalizedFormula = normalizeFormula(formula);

  const [mpSummary, mpElasticity, mpElectronic, mpThermo, mpPhonon, aflowResult, aflowDFT] = await Promise.all([
    mpAvailable ? fetchSummary(normalizedFormula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchElasticity(normalizedFormula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchElectronicStructure(normalizedFormula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchThermo(normalizedFormula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchPhonon(normalizedFormula).catch(() => null) : Promise.resolve(null),
    fetchAflowData(normalizedFormula).catch(() => ({ entries: [], queryFormula: normalizedFormula, source: "AFLOW" as const })),
    fetchAflowDFTData(normalizedFormula).catch(() => null),
  ]);

  return {
    mpSummary,
    mpElasticity,
    mpElectronic,
    mpThermo,
    mpPhonon,
    aflowEntry: aflowResult.entries[0] ?? null,
    aflowDFT,
  };
}

function resolve<T>(
  mpValue: T | null | undefined,
  aflowValue: T | null | undefined,
  fallback: T,
  fallbackSource: DFTSource = "analytical",
): DFTResolvedFeature<T> {
  if (mpValue != null && mpValue !== undefined) return { value: mpValue, source: "dft-mp" };
  if (aflowValue != null && aflowValue !== undefined) return { value: aflowValue, source: "dft-aflow" };
  return { value: fallback, source: fallbackSource };
}

export async function resolveDFTFeatures(formula: string): Promise<DFTResolvedFeatures> {
  const raw = await fetchAllDFTSources(formula);

  const analytical = computeAnalyticalFallbacks(formula);

  const hasExternalData = raw.mpSummary != null || raw.mpElectronic != null || raw.mpThermo != null || raw.mpElasticity != null || raw.aflowEntry != null || raw.aflowDFT != null;
  const hasPartialExternal = hasExternalData && !(raw.mpElectronic != null && raw.mpThermo != null);

  let xtbData: XTBEnrichedFeatures | null = null;
  if ((!hasExternalData || hasPartialExternal) && isDFTAvailable()) {
    const preFilter = estimatePhononStability(formula);
    if (!preFilter.stable) {
      console.log(`[DFT] ${formula}: Pre-filter rejected (score=${preFilter.score.toFixed(2)}): ${preFilter.reasons.join("; ")}`);
    } else {
      try {
        xtbData = await runXTBEnrichment(formula);
        if (xtbData) {
          console.log(`[DFT] ${formula}: xTB DFT computed (${xtbData.prototype}): gap=${xtbData.bandGap.toFixed(3)}eV, metallic=${xtbData.isMetallic}, E/atom=${xtbData.totalEnergyPerAtom.toFixed(4)}Ha${xtbData.formationEnergyPerAtom != null ? `, Ef=${xtbData.formationEnergyPerAtom.toFixed(3)}eV/atom` : ""}`);
        }
      } catch (err) {
        xtbData = null;
      }
    }
  }

  const bandGap = resolve(
    raw.mpSummary?.bandGap ?? raw.mpElectronic?.bandGap,
    raw.aflowEntry?.bandgap,
    xtbData ? xtbData.bandGap : analytical.bandGap,
    xtbData ? "dft-xtb" : "analytical",
  );

  const isMetallic = resolve(
    raw.mpSummary?.isMetallic ?? raw.mpElectronic?.isMetal,
    raw.aflowEntry?.Egap_type === "metal" ? true : undefined,
    xtbData ? xtbData.isMetallic : analytical.isMetallic,
    xtbData ? "dft-xtb" : "analytical",
  );

  const dosAtFermi = resolve<number | null>(
    raw.mpElectronic?.dosAtFermi,
    null,
    analytical.dosAtFermi,
  );

  const mpDebye = raw.mpThermo?.debyeTemperature ?? null;
  const aflowDebye = raw.aflowDFT?.ael_debye ?? null;
  const debyeTemp = resolve(mpDebye, aflowDebye, analytical.debyeTemp);

  const mpBulk = raw.mpElasticity?.bulkModulus ?? null;
  const aflowBulk = raw.aflowEntry?.Bvoigt ?? null;
  const bulkModulus = resolve(mpBulk, aflowBulk, analytical.bulkModulus);

  const mpFormE = raw.mpSummary?.formationEnergyPerAtom ?? raw.mpThermo?.formationEnergyPerAtom ?? null;
  const aflowFormE = raw.aflowEntry?.enthalpy_formation_atom ?? null;
  const xtbFormE = xtbData?.formationEnergyPerAtom ?? null;
  const formationEnergy = resolve(
    mpFormE,
    aflowFormE,
    xtbFormE != null ? xtbFormE : analytical.formationEnergy,
    xtbFormE != null ? "dft-xtb" : "analytical",
  );

  const mpEhull = raw.mpSummary?.energyAboveHull ?? raw.mpThermo?.energyAboveHull ?? null;
  const energyAboveHull = resolve<number | null>(mpEhull, null, null);

  const fdPhonons = xtbData?.finiteDisplacementPhonons ?? null;
  const rawFdPhononMax = fdPhonons ? fdPhonons.highestFrequency : null;
  const fdPhononMax = rawFdPhononMax != null ? (rawFdPhononMax > 5000 ? rawFdPhononMax / 20 : rawFdPhononMax) : null;

  const mpPhononMax = raw.mpPhonon?.hasPhononData && raw.mpPhonon.lastPhononFreq
    ? raw.mpPhonon.lastPhononFreq : null;
  const phononFreqMax = resolve<number | null>(
    mpPhononMax,
    null,
    fdPhononMax != null ? fdPhononMax : analytical.phononFreqMax,
    fdPhononMax != null ? "dft-xtb" : "analytical",
  );

  const thermalConductivity = resolve<number | null>(
    null,
    raw.aflowDFT?.agl_thermal_conductivity_300K ?? null,
    analytical.thermalConductivity,
  );

  const coreFeatures: DFTResolvedFeature<any>[] = [
    bandGap, isMetallic, debyeTemp, bulkModulus, formationEnergy,
  ];
  const optionalFeatures: DFTResolvedFeature<any>[] = [
    dosAtFermi, energyAboveHull, phononFreqMax, thermalConductivity,
  ].filter(f => f.value != null);
  const allResolved = [...coreFeatures, ...optionalFeatures];
  const dftCount = allResolved.filter(f => f.source !== "analytical").length;
  const externalCoverage = allResolved.length > 0 ? dftCount / allResolved.length : 0;

  const hasXTB = xtbData != null;
  const dftCoverage = externalCoverage > 0 ? externalCoverage : (hasXTB ? 0.75 : Math.max(0.5, analytical.estimatorCoverage));

  if (!hasExternalData && !hasXTB) {
    console.log(`[DFT] ${formula}: No external API or xTB data. Using analytical fallbacks (coverage=${dftCoverage.toFixed(2)}).`);
  } else if (!hasExternalData && hasXTB) {
  } else if (hasExternalData && externalCoverage < 0.5) {
    console.log(`[DFT] ${formula}: Partial external data (coverage=${externalCoverage.toFixed(2)}). Effective coverage=${dftCoverage.toFixed(2)}.`);
  }

  const xtbPhononStability = xtbData?.phononStability ?? null;
  if (xtbPhononStability) {
    if (xtbPhononStability.hasImaginaryModes) {
      console.log(`[DFT] ${formula}: xTB Hessian found ${xtbPhononStability.imaginaryModeCount} imaginary mode(s), lowest freq=${xtbPhononStability.lowestFrequency.toFixed(1)} cm-1 (severe: freq < -2000)`);
    } else {
      const mildCount = xtbPhononStability.frequencies?.filter(f => f < -50 && f >= -2000).length ?? 0;
      if (mildCount > 0) {
        console.log(`[DFT] ${formula}: xTB Hessian: ${mildCount} mild imaginary mode(s) (lowest freq=${xtbPhononStability.lowestFrequency.toFixed(1)} cm-1) — within xTB tolerance, accepted`);
      } else {
        console.log(`[DFT] ${formula}: xTB Hessian confirms phonon stability, lowest freq=${xtbPhononStability.lowestFrequency.toFixed(1)} cm-1`);
      }
    }
  }

  if (fdPhonons) {
    console.log(`[DFT] ${formula}: Finite displacement phonons: stable=${fdPhonons.dynamicallyStable}, freq=[${fdPhonons.lowestFrequency.toFixed(1)}, ${fdPhonons.highestFrequency.toFixed(1)}] cm⁻¹${fdPhonons.omegaLog ? `, ω_log=${fdPhonons.omegaLog.toFixed(1)} cm⁻¹` : ""}`);
  }

  return {
    bandGap,
    isMetallic,
    dosAtFermi,
    debyeTemp,
    bulkModulus,
    formationEnergy,
    energyAboveHull,
    phononFreqMax,
    thermalConductivity,
    phononStability: xtbPhononStability,
    finiteDisplacementPhonons: fdPhonons,
    dftCoverage,
    sources: {
      mp: raw.mpSummary != null || raw.mpElectronic != null || raw.mpThermo != null || raw.mpElasticity != null,
      aflow: raw.aflowEntry != null || raw.aflowDFT != null,
    },
  };
}

export function describeDFTSources(features: DFTResolvedFeatures): string {
  const parts: string[] = [];
  if (features.bandGap.source !== "analytical") parts.push(`bandGap=${features.bandGap.value.toFixed(2)}eV(${features.bandGap.source})`);
  if (features.debyeTemp.source !== "analytical") parts.push(`debye=${features.debyeTemp.value}K(${features.debyeTemp.source})`);
  if (features.bulkModulus.source !== "analytical" && features.bulkModulus.value > 0) parts.push(`B=${features.bulkModulus.value.toFixed(0)}GPa(${features.bulkModulus.source})`);
  if (features.dosAtFermi.value != null && features.dosAtFermi.source !== "analytical") parts.push(`DOS=${features.dosAtFermi.value.toFixed(2)}(${features.dosAtFermi.source})`);
  if (features.phononFreqMax.value != null && features.phononFreqMax.source !== "analytical") parts.push(`phMax=${features.phononFreqMax.value.toFixed(0)}cm-1(${features.phononFreqMax.source})`);
  if (features.formationEnergy.source === "dft-xtb") parts.push(`Ef=${features.formationEnergy.value.toFixed(3)}eV(dft-xtb)`);
  if (features.isMetallic.source === "dft-xtb") parts.push(`metallic=${features.isMetallic.value}(dft-xtb)`);
  if (parts.length === 0) {
    const analyticalParts: string[] = [];
    analyticalParts.push(`bandGap=${features.bandGap.value.toFixed(2)}eV`);
    analyticalParts.push(`debye=${features.debyeTemp.value.toFixed(0)}K`);
    analyticalParts.push(`metallic=${features.isMetallic.value}`);
    if (features.bulkModulus.value > 0) analyticalParts.push(`B=${features.bulkModulus.value.toFixed(0)}GPa`);
    if (features.formationEnergy.value !== 0) analyticalParts.push(`Ef=${features.formationEnergy.value.toFixed(2)}eV`);
    return `analytical: ${analyticalParts.join(", ")}`;
  }
  return parts.join(", ");
}
