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

/** Returns true if the formula string is a chemical-system notation (e.g. "Fe-C") rather than a proper compound formula. MP API cannot use these as formula filters. */
function isChemSys(formula: string): boolean {
  return /^[A-Z][a-z]?(-[A-Z][a-z]?)+$/.test(formula);
}

async function mpFetch(endpoint: string, params: Record<string, string> = {}, attempt = 1): Promise<any | null> {
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
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      let body = "";
      try { body = await response.text(); } catch {}
      if (response.status === 429) {
        console.warn(`[MP API] Rate limited (429): ${body.slice(0, 300)}`);
      } else if (response.status === 403) {
        console.warn(`[MP API] Forbidden (403) — check API key: ${body.slice(0, 300)}`);
      } else {
        console.warn(`[MP API] HTTP ${response.status} for ${url.pathname}?${url.searchParams}: ${body.slice(0, 500)}`);
      }
      return null;
    }

    return await response.json();
  } catch (err: any) {
    if (err?.name === "AbortError" || err?.name === "TimeoutError") {
      console.warn(`[MP API] Timeout (30s) for ${endpoint} (attempt ${attempt})`);
    } else {
      const cause = err?.cause?.message || err?.cause?.code || "";
      console.warn(`[MP API] Request failed for ${endpoint} (attempt ${attempt}): ${err?.message || "unknown"}${cause ? ` (cause: ${cause})` : ""}`);
    }
    if (attempt < 3) {
      await new Promise(r => setTimeout(r, attempt * 3000));
      return mpFetch(endpoint, params, attempt + 1);
    }
    return null;
  }
}

export async function fetchSummary(formula: string): Promise<MPSummaryData | null> {
  const cached = await getCachedData(formula, "summary");
  if (cached) return cached as MPSummaryData;

  const normalizedFormula = normalizeFormula(formula);
  if (isChemSys(normalizedFormula)) return null; // chem-sys notation not valid as formula filter

  const data = await mpFetch("/materials/summary/", {
    formula: normalizedFormula,
    _limit: "1",
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

  // Elasticity endpoint rejects both formula and fields; look up material_id first
  const summary = await fetchSummary(formula);
  if (!summary?.mpId) return null;

  const data = await mpFetch("/materials/elasticity/", {
    material_ids: summary.mpId,
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

  await setCachedData(formula, "elasticity", result, summary.mpId);
  return result;
}

export async function fetchMagnetism(formula: string): Promise<MPMagnetismData | null> {
  const cached = await getCachedData(formula, "magnetism");
  if (cached) return cached as MPMagnetismData;

  // Magnetism endpoint rejects formula; look up material_id first
  const summary = await fetchSummary(formula);
  if (!summary?.mpId) return null;

  const data = await mpFetch("/materials/magnetism/", {
    material_ids: summary.mpId,
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

  await setCachedData(formula, "magnetism", result, summary.mpId);
  return result;
}

export async function fetchPhonon(formula: string): Promise<MPPhononData | null> {
  const cached = await getCachedData(formula, "phonon");
  if (cached) return cached as MPPhononData;

  const normalizedFormula = normalizeFormula(formula);

  const data = await mpFetch("/materials/phonon/", {
    formula: normalizedFormula,
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
    _limit: "1",
  });

  if (!data?.data?.length) return null;

  const entry = data.data[0];

  const bandStructureType: string | null = entry.band_gap != null
    ? (entry.band_gap > 0 ? (entry.is_metal ? "metallic" : "indirect") : "metallic")
    : null;

  const result: MPElectronicStructureData = {
    dosAtFermi: null,
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
    _limit: "1",
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
  // Sequential fetches to avoid hammering the MP API with 6 simultaneous requests
  const summary = await fetchSummary(formula);
  const elasticity = await fetchElasticity(formula);
  const magnetism = await fetchMagnetism(formula);
  const phonon = await fetchPhonon(formula);
  const electronicStructure = await fetchElectronicStructure(formula);
  const thermo = await fetchThermo(formula);

  return { summary, elasticity, magnetism, phonon, electronicStructure, thermo };
}

export function isApiAvailable(): boolean {
  return !!getApiKey();
}

export async function fetchCachedFormationEnergies(formulas: string[]): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  for (const formula of formulas) {
    const cached = await getCachedData(formula, "summary");
    if (cached && (cached as MPSummaryData).formationEnergyPerAtom != null) {
      result.set(formula, (cached as MPSummaryData).formationEnergyPerAtom);
    }
  }
  return result;
}

/**
 * Fetches a paginated batch of metallic materials directly from the MP API.
 * Used for progressive GNN augmentation: call with skip=0 for batch 1, skip=500 for batch 2, etc.
 * Caches each result in the DB so future startups don't re-fetch.
 */
export async function fetchMPBatchFromAPI(limit: number, skip: number): Promise<MPGNNSeedRecord[]> {
  const results: MPGNNSeedRecord[] = [];
  if (!getApiKey()) {
    console.log(`[MP-Batch] No API key available, skipping batch fetch (skip=${skip})`);
    return results;
  }
  try {
    const data = await mpFetch("/materials/summary/", {
      is_metal: "true",
      _limit: String(limit),
      _skip: String(skip),
    });
    if (!data) {
      console.warn(`[MP-Batch] mpFetch returned null for skip=${skip} — API key invalid, network error, or rate limit`);
      return results;
    }
    if (!data.data) {
      console.warn(`[MP-Batch] Unexpected response shape (no .data field): ${JSON.stringify(data).slice(0, 200)}`);
      return results;
    }
    for (const item of data.data) {
      const formula = normalizeFormula(item.formula_pretty ?? "");
      if (!formula) continue;
      const rec: MPGNNSeedRecord = {
        formula,
        bandGap: item.band_gap ?? null,
        formationEnergy: item.formation_energy_per_atom ?? null,
        spaceGroup: item.symmetry?.symbol ?? null,
        crystalSystem: item.symmetry?.crystal_system ?? null,
        isMetallic: item.is_metal ?? true,
      };
      results.push(rec);
      const cacheEntry: MPSummaryData = {
        mpId: item.material_id ?? "",
        formula,
        bandGap: rec.bandGap ?? 0,
        formationEnergyPerAtom: rec.formationEnergy ?? 0,
        energyAboveHull: item.energy_above_hull ?? 0,
        isStable: (item.energy_above_hull ?? 999) < 0.05,
        spaceGroup: rec.spaceGroup ?? "",
        crystalSystem: rec.crystalSystem ?? "",
        volume: 0,
        density: 0,
        nsites: item.nsites ?? 1,
        magneticOrdering: "",
        totalMagnetization: 0,
        isMetallic: rec.isMetallic,
        efermi: null,
      };
      setCachedData(formula, "summary", cacheEntry, item.material_id).catch(() => {});
    }
    console.log(`[MP-Batch] Fetched ${results.length} records from API (skip=${skip}, limit=${limit})`);
  } catch (err: any) {
    console.warn(`[MP-Batch] API fetch failed (skip=${skip}): ${err?.message?.slice(0, 100)}`);
  }
  return results;
}

export interface MPGNNSeedRecord {
  formula: string;
  bandGap: number | null;
  formationEnergy: number | null;
  spaceGroup: string | null;
  crystalSystem: string | null;
  isMetallic: boolean;
}

/**
 * Loads MP summary data for GNN pre-seeding at startup.
 * Step 1: reads all cached "summary" rows from the local DB (instant, no API call).
 * Step 2: if the API key is set and the cache has fewer than 400 entries, bulk-fetches
 *         up to 500 metallic materials from the MP API and stores them in the cache.
 * Returns de-duplicated records suitable for passing to addDFTTrainingResult.
 */
export async function fetchGNNSeedData(): Promise<MPGNNSeedRecord[]> {
  const seen = new Map<string, MPGNNSeedRecord>();

  // --- Step 1: read all cached summary rows from DB ---
  try {
    const rows = await db
      .select()
      .from(mpMaterialCache)
      .where(eq(mpMaterialCache.dataType, "summary"))
      .limit(2000);

    for (const row of rows) {
      const d = row.data as MPSummaryData;
      if (!d || typeof d !== "object") continue;
      seen.set(row.formula, {
        formula: row.formula,
        bandGap: typeof d.bandGap === "number" ? d.bandGap : null,
        formationEnergy: typeof d.formationEnergyPerAtom === "number" ? d.formationEnergyPerAtom : null,
        spaceGroup: d.spaceGroup ?? null,
        crystalSystem: d.crystalSystem ?? null,
        isMetallic: d.isMetallic ?? false,
      });
    }
    console.log(`[MP-GNNSeed] Loaded ${seen.size} cached MP summaries from DB`);
  } catch (err: any) {
    console.warn(`[MP-GNNSeed] DB cache load failed: ${err?.message?.slice(0, 100)}`);
  }

  // --- Step 2: bulk-fetch from MP API if we need more ---
  if (seen.size < 400 && getApiKey()) {
    try {
      const data = await mpFetch("/materials/summary/", {
        is_metal: "true",
        _limit: "500",
      });

      if (data?.data) {
        let fetched = 0;
        for (const item of data.data) {
          const formula = normalizeFormula(item.formula_pretty ?? "");
          if (!formula || seen.has(formula)) continue;

          const rec: MPGNNSeedRecord = {
            formula,
            bandGap: item.band_gap ?? null,
            formationEnergy: item.formation_energy_per_atom ?? null,
            spaceGroup: item.symmetry?.symbol ?? null,
            crystalSystem: item.symmetry?.crystal_system ?? null,
            isMetallic: item.is_metal ?? true,
          };
          seen.set(formula, rec);

          // Cache the result so future startups use it from DB
          const cacheEntry: MPSummaryData = {
            mpId: item.material_id ?? "",
            formula,
            bandGap: rec.bandGap ?? 0,
            formationEnergyPerAtom: rec.formationEnergy ?? 0,
            energyAboveHull: item.energy_above_hull ?? 0,
            isStable: (item.energy_above_hull ?? 999) < 0.05,
            spaceGroup: rec.spaceGroup ?? "",
            crystalSystem: rec.crystalSystem ?? "",
            volume: 0,
            density: 0,
            nsites: item.nsites ?? 1,
            magneticOrdering: "",
            totalMagnetization: 0,
            isMetallic: rec.isMetallic,
            efermi: null,
          };
          setCachedData(formula, "summary", cacheEntry, item.material_id).catch(() => {});
          fetched++;
        }
        console.log(`[MP-GNNSeed] Fetched ${fetched} new MP materials from API (total: ${seen.size})`);
      }
    } catch (err: any) {
      console.warn(`[MP-GNNSeed] API bulk-fetch failed: ${err?.message?.slice(0, 100)}`);
    }
  }

  return Array.from(seen.values());
}
