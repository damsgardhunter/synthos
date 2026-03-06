import { db } from "../db";
import { mpMaterialCache } from "@shared/schema";
import { eq, and } from "drizzle-orm";

const MP_API_BASE = "https://api.materialsproject.org";

function getApiKey(): string | null {
  return process.env.MATERIALS_PROJECT_API_KEY || null;
}

export interface MPSummaryData {
  mpId: string;
  formula: string;
  bandGap: number;
  formationEnergyPerAtom: number;
  energyAboveHull: number;
  isStable: boolean;
  spaceGroup: string;
  crystalSystem: string;
  volume: number;
  density: number;
  nsites: number;
  magneticOrdering: string;
  totalMagnetization: number;
  isMetallic: boolean;
  efermi: number | null;
}

export interface MPElasticityData {
  bulkModulus: number;
  shearModulus: number;
  youngsModulus: number;
  poissonRatio: number;
  bulkModulusVoigt: number;
  bulkModulusReuss: number;
  shearModulusVoigt: number;
  shearModulusReuss: number;
}

export interface MPPhononData {
  hasPhononData: boolean;
  lastPhononFreq: number | null;
  phononBandStructure: any | null;
  phononDos: any | null;
}

export interface MPElectronicStructureData {
  dosAtFermi: number | null;
  bandStructureType: string | null;
  fermiEnergy: number | null;
  bandGap: number | null;
  isMetal: boolean;
}

export interface MPThermoData {
  energyAboveHull: number;
  debyeTemperature: number | null;
  formationEnergyPerAtom: number;
  isStable: boolean;
  uncorrectedEnergy: number | null;
}

export interface MPMagnetismData {
  ordering: string;
  totalMagnetization: number;
  totalMagnetizationNormalized: number;
  types: string[];
}

async function getCachedData(formula: string, dataType: string): Promise<any | null> {
  try {
    const normalizedFormula = normalizeFormula(formula);
    const results = await db
      .select()
      .from(mpMaterialCache)
      .where(
        and(
          eq(mpMaterialCache.formula, normalizedFormula),
          eq(mpMaterialCache.dataType, dataType)
        )
      )
      .limit(1);

    if (results.length > 0) {
      const cached = results[0];
      const age = Date.now() - new Date(cached.fetchedAt!).getTime();
      const maxAge = 7 * 24 * 60 * 60 * 1000;
      if (age < maxAge) {
        return cached.data;
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function setCachedData(formula: string, dataType: string, data: any, mpId?: string): Promise<void> {
  try {
    const normalizedFormula = normalizeFormula(formula);
    await db
      .insert(mpMaterialCache)
      .values({
        formula: normalizedFormula,
        mpId: mpId || null,
        dataType,
        data,
      })
      .onConflictDoUpdate({
        target: [mpMaterialCache.formula, mpMaterialCache.dataType],
        set: {
          data,
          mpId: mpId || null,
          fetchedAt: new Date(),
        },
      });
  } catch {
  }
}

function normalizeFormula(formula: string): string {
  if (typeof formula !== "string") formula = String(formula ?? "");
  return formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
}

async function mpFetch(endpoint: string, params: Record<string, string> = {}): Promise<any | null> {
  const apiKey = getApiKey();
  if (!apiKey) return null;

  const url = new URL(`${MP_API_BASE}${endpoint}`);
  for (const [k, v] of Object.entries(params)) {
    url.searchParams.set(k, v);
  }

  try {
    const response = await fetch(url.toString(), {
      headers: {
        "X-API-KEY": apiKey,
        "Accept": "application/json",
      },
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      if (response.status === 429) {
        console.log("[MP API] Rate limited, will use fallback");
      } else if (response.status === 403) {
        console.log("[MP API] Invalid API key");
      }
      return null;
    }

    return await response.json();
  } catch (err: any) {
    if (err?.name !== "AbortError") {
      console.log(`[MP API] Request failed: ${err?.message || "unknown"}`);
    }
    return null;
  }
}

export async function fetchSummary(formula: string): Promise<MPSummaryData | null> {
  const cached = await getCachedData(formula, "summary");
  if (cached) return cached as MPSummaryData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/summary/", {
    formula: normalizedFormula,
    fields: "material_id,formula_pretty,band_gap,formation_energy_per_atom,energy_above_hull,is_stable,symmetry,volume,density,nsites,ordering,total_magnetization,is_metal,efermi",
    _limit: "1",
    _sort_fields: "energy_above_hull",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const result: MPSummaryData = {
    mpId: entry.material_id,
    formula: entry.formula_pretty || normalizedFormula,
    bandGap: entry.band_gap ?? 0,
    formationEnergyPerAtom: entry.formation_energy_per_atom ?? 0,
    energyAboveHull: entry.energy_above_hull ?? 0,
    isStable: entry.is_stable ?? false,
    spaceGroup: entry.symmetry?.symbol ?? "",
    crystalSystem: entry.symmetry?.crystal_system ?? "",
    volume: entry.volume ?? 0,
    density: entry.density ?? 0,
    nsites: entry.nsites ?? 1,
    magneticOrdering: entry.ordering ?? "NM",
    totalMagnetization: entry.total_magnetization ?? 0,
    isMetallic: entry.is_metal ?? false,
    efermi: entry.efermi ?? null,
  };

  await setCachedData(formula, "summary", result, result.mpId);
  return result;
}

export async function fetchElasticity(formula: string): Promise<MPElasticityData | null> {
  const cached = await getCachedData(formula, "elasticity");
  if (cached) return cached as MPElasticityData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/elasticity/", {
    formula: normalizedFormula,
    fields: "material_id,bulk_modulus,shear_modulus,young_modulus,poisson_ratio",
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const bm = entry.bulk_modulus || {};
  const sm = entry.shear_modulus || {};

  const result: MPElasticityData = {
    bulkModulus: bm.vrh ?? 0,
    shearModulus: sm.vrh ?? 0,
    youngsModulus: entry.young_modulus ?? 0,
    poissonRatio: entry.poisson_ratio ?? 0.3,
    bulkModulusVoigt: bm.voigt ?? bm.vrh ?? 0,
    bulkModulusReuss: bm.reuss ?? bm.vrh ?? 0,
    shearModulusVoigt: sm.voigt ?? sm.vrh ?? 0,
    shearModulusReuss: sm.reuss ?? sm.vrh ?? 0,
  };

  await setCachedData(formula, "elasticity", result, entry.material_id);
  return result;
}

export async function fetchMagnetism(formula: string): Promise<MPMagnetismData | null> {
  const cached = await getCachedData(formula, "magnetism");
  if (cached) return cached as MPMagnetismData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/magnetism/", {
    formula: normalizedFormula,
    fields: "material_id,ordering,total_magnetization,total_magnetization_normalized_formula_units,types",
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const result: MPMagnetismData = {
    ordering: entry.ordering ?? "NM",
    totalMagnetization: entry.total_magnetization ?? 0,
    totalMagnetizationNormalized: entry.total_magnetization_normalized_formula_units ?? 0,
    types: entry.types ?? [],
  };

  await setCachedData(formula, "magnetism", result, entry.material_id);
  return result;
}

export async function fetchPhonon(formula: string): Promise<MPPhononData | null> {
  const cached = await getCachedData(formula, "phonon");
  if (cached) return cached as MPPhononData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/phonon/", {
    formula: normalizedFormula,
    fields: "material_id,last_phonon_freq,ph_bs,ph_dos",
    _limit: "1",
  });

  if (!data?.data?.length) {
    const noData: MPPhononData = { hasPhononData: false, lastPhononFreq: null, phononBandStructure: null, phononDos: null };
    await setCachedData(formula, "phonon", noData);
    return noData;
  }

  const entry = data.data[0];
  const result: MPPhononData = {
    hasPhononData: true,
    lastPhononFreq: entry.last_phonon_freq ?? null,
    phononBandStructure: entry.ph_bs ? { available: true } : null,
    phononDos: entry.ph_dos ? { available: true } : null,
  };

  await setCachedData(formula, "phonon", result, entry.material_id);
  return result;
}

export async function fetchElectronicStructure(formula: string): Promise<MPElectronicStructureData | null> {
  const cached = await getCachedData(formula, "electronic_structure");
  if (cached) return cached as MPElectronicStructureData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/electronic_structure/", {
    formula: normalizedFormula,
    fields: "material_id,dos,band_gap,efermi,is_metal,bandstructure",
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];

  let dosAtFermi: number | null = null;
  if (entry.dos && entry.dos.total && entry.efermi != null) {
    try {
      const totalDos = entry.dos.total;
      if (totalDos.densities && totalDos.energies) {
        const energies = totalDos.energies;
        const densities = totalDos.densities;
        let closestIdx = 0;
        let closestDist = Math.abs(energies[0] - entry.efermi);
        for (let i = 1; i < energies.length; i++) {
          const dist = Math.abs(energies[i] - entry.efermi);
          if (dist < closestDist) {
            closestDist = dist;
            closestIdx = i;
          }
        }
        dosAtFermi = densities[closestIdx] ?? null;
      }
    } catch {
      dosAtFermi = null;
    }
  }

  let bandStructureType: string | null = null;
  if (entry.bandstructure) {
    bandStructureType = entry.band_gap > 0 ? "indirect" : "metallic";
    if (entry.bandstructure.is_direct) {
      bandStructureType = "direct";
    }
  } else if (entry.band_gap != null) {
    bandStructureType = entry.band_gap > 0 ? (entry.is_metal ? "metallic" : "indirect") : "metallic";
  }

  const result: MPElectronicStructureData = {
    dosAtFermi,
    bandStructureType,
    fermiEnergy: entry.efermi ?? null,
    bandGap: entry.band_gap ?? null,
    isMetal: entry.is_metal ?? false,
  };

  await setCachedData(formula, "electronic_structure", result, entry.material_id);
  return result;
}

export async function fetchThermo(formula: string): Promise<MPThermoData | null> {
  const cached = await getCachedData(formula, "thermo");
  if (cached) return cached as MPThermoData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/thermo/", {
    formula: normalizedFormula,
    fields: "material_id,energy_above_hull,debye_temperature,formation_energy_per_atom,is_stable,uncorrected_energy_per_atom",
    _limit: "1",
    _sort_fields: "energy_above_hull",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];
  const result: MPThermoData = {
    energyAboveHull: entry.energy_above_hull ?? 0,
    debyeTemperature: entry.debye_temperature ?? null,
    formationEnergyPerAtom: entry.formation_energy_per_atom ?? 0,
    isStable: entry.is_stable ?? false,
    uncorrectedEnergy: entry.uncorrected_energy_per_atom ?? null,
  };

  await setCachedData(formula, "thermo", result, entry.material_id);
  return result;
}

export async function fetchThermodynamicStability(formula: string): Promise<{ energyAboveHull: number; isStable: boolean; decompositionProducts: string[] } | null> {
  const summary = await fetchSummary(formula);
  if (!summary) return null;
  return {
    energyAboveHull: summary.energyAboveHull,
    isStable: summary.isStable,
    decompositionProducts: [],
  };
}

export async function fetchAllData(formula: string): Promise<{
  summary: MPSummaryData | null;
  elasticity: MPElasticityData | null;
  magnetism: MPMagnetismData | null;
  phonon: MPPhononData | null;
  electronicStructure: MPElectronicStructureData | null;
  thermo: MPThermoData | null;
}> {
  const [summary, elasticity, magnetism, phonon, electronicStructure, thermo] = await Promise.all([
    fetchSummary(formula),
    fetchElasticity(formula),
    fetchMagnetism(formula),
    fetchPhonon(formula),
    fetchElectronicStructure(formula),
    fetchThermo(formula),
  ]);

  return { summary, elasticity, magnetism, phonon, electronicStructure, thermo };
}

export function isApiAvailable(): boolean {
  return !!getApiKey();
}
