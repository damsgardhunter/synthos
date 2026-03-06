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

export type DFTSource = "dft-mp" | "dft-aflow" | "analytical";

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
  
  parts.sort((a, b) => a.el.localeCompare(b.el));
  
  return parts.map(p => p.count === 1 ? p.el : `${p.el}${p.count}`).join("");
}

function parseFormulaElements(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
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

function computeAnalyticalFallbacks(formula: string): {
  bandGap: number;
  isMetallic: boolean;
  debyeTemp: number;
  bulkModulus: number;
  formationEnergy: number;
  dosAtFermi: number | null;
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

  const debyeWeighted = getCompositionWeightedProperty(counts, "debyeTemperature");
  const debyeTemp = debyeWeighted && debyeWeighted > 0 ? debyeWeighted : 300;

  const bulkWeighted = getCompositionWeightedProperty(counts, "bulkModulus");
  const bulkModulus = bulkWeighted && bulkWeighted > 0 ? bulkWeighted : 0;

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
    }
  }

  return { bandGap, isMetallic, debyeTemp, bulkModulus, formationEnergy, dosAtFermi };
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
): DFTResolvedFeature<T> {
  if (mpValue != null && mpValue !== undefined) return { value: mpValue, source: "dft-mp" };
  if (aflowValue != null && aflowValue !== undefined) return { value: aflowValue, source: "dft-aflow" };
  return { value: fallback, source: "analytical" };
}

export async function resolveDFTFeatures(formula: string): Promise<DFTResolvedFeatures> {
  const raw = await fetchAllDFTSources(formula);

  const analytical = computeAnalyticalFallbacks(formula);

  const bandGap = resolve(
    raw.mpSummary?.bandGap ?? raw.mpElectronic?.bandGap,
    raw.aflowEntry?.bandgap,
    analytical.bandGap,
  );

  const isMetallic = resolve(
    raw.mpSummary?.isMetallic ?? raw.mpElectronic?.isMetal,
    raw.aflowEntry?.Egap_type === "metal" ? true : undefined,
    analytical.isMetallic,
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
  const formationEnergy = resolve(mpFormE, aflowFormE, analytical.formationEnergy);

  const mpEhull = raw.mpSummary?.energyAboveHull ?? raw.mpThermo?.energyAboveHull ?? null;
  const energyAboveHull = resolve<number | null>(mpEhull, null, null);

  const mpPhononMax = raw.mpPhonon?.hasPhononData && raw.mpPhonon.lastPhononFreq
    ? raw.mpPhonon.lastPhononFreq : null;
  const phononFreqMax = resolve<number | null>(mpPhononMax, null, null);

  const thermalConductivity = resolve<number | null>(
    null,
    raw.aflowDFT?.agl_thermal_conductivity_300K ?? null,
    null,
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

  const hasExternalData = raw.mpSummary != null || raw.mpElectronic != null || raw.mpThermo != null || raw.mpElasticity != null || raw.aflowEntry != null || raw.aflowDFT != null;
  const dftCoverage = externalCoverage > 0 ? externalCoverage : 0.3;

  if (!hasExternalData) {
    console.log(`[DFT] ${formula}: No external API data found (MP/AFLOW). Using analytical fallbacks (coverage=${dftCoverage.toFixed(2)}).`);
  } else if (externalCoverage < 0.5) {
    console.log(`[DFT] ${formula}: Partial external data (coverage=${externalCoverage.toFixed(2)}), supplemented with analytical. Effective coverage=${dftCoverage.toFixed(2)}.`);
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
