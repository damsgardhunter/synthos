import { fetchSummary, fetchElasticity, fetchElectronicStructure, fetchThermo, fetchPhonon, isApiAvailable } from "./materials-project-client";
import type { MPSummaryData, MPElasticityData, MPElectronicStructureData, MPThermoData, MPPhononData } from "./materials-project-client";
import { fetchAflowData, fetchAflowDFTData } from "./aflow-client";
import type { AflowDFTData, AflowEntry } from "./aflow-client";

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

async function fetchAllDFTSources(formula: string): Promise<RawDFTSources> {
  const mpAvailable = isApiAvailable();

  const [mpSummary, mpElasticity, mpElectronic, mpThermo, mpPhonon, aflowResult, aflowDFT] = await Promise.all([
    mpAvailable ? fetchSummary(formula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchElasticity(formula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchElectronicStructure(formula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchThermo(formula).catch(() => null) : Promise.resolve(null),
    mpAvailable ? fetchPhonon(formula).catch(() => null) : Promise.resolve(null),
    fetchAflowData(formula).catch(() => ({ entries: [], queryFormula: formula, source: "AFLOW" as const })),
    fetchAflowDFTData(formula).catch(() => null),
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

  const bandGap = resolve(
    raw.mpSummary?.bandGap ?? raw.mpElectronic?.bandGap,
    raw.aflowEntry?.bandgap,
    0,
  );

  const isMetallic = resolve(
    raw.mpSummary?.isMetallic ?? raw.mpElectronic?.isMetal,
    raw.aflowEntry?.Egap_type === "metal" ? true : undefined,
    false,
  );

  const dosAtFermi = resolve<number | null>(
    raw.mpElectronic?.dosAtFermi,
    null,
    null,
  );

  const mpDebye = raw.mpThermo?.debyeTemperature ?? null;
  const aflowDebye = raw.aflowDFT?.ael_debye ?? null;
  const debyeTemp = resolve(mpDebye, aflowDebye, 300);

  const mpBulk = raw.mpElasticity?.bulkModulus ?? null;
  const aflowBulk = raw.aflowEntry?.Bvoigt ?? null;
  const bulkModulus = resolve(mpBulk, aflowBulk, 0);

  const mpFormE = raw.mpSummary?.formationEnergyPerAtom ?? raw.mpThermo?.formationEnergyPerAtom ?? null;
  const aflowFormE = raw.aflowEntry?.enthalpy_formation_atom ?? null;
  const formationEnergy = resolve(mpFormE, aflowFormE, 0);

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
  const dftCoverage = allResolved.length > 0 ? dftCount / allResolved.length : 0;

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
  return parts.length > 0 ? parts.join(", ") : "analytical only";
}
