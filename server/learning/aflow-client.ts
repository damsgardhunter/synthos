import { db } from "../db";
import { mpMaterialCache } from "@shared/schema";
import { eq, and } from "drizzle-orm";

const AFLOW_API_BASE = "http://aflowlib.duke.edu/API/aflux";

export interface AflowDFTData {
  formula: string;
  ael_debye: number | null;
  agl_thermal_conductivity_300K: number | null;
  energy_atom: number | null;
  enthalpy_formation_cell: number | null;
  source: "AFLOW";
}

export interface AflowEntry {
  auid: string;
  compound: string;
  spaceGroupNumber: number;
  spaceGroupSymbol: string;
  latticeSystemRelax: string;
  Bvoigt: number | null;       // GPa — bulk modulus (Voigt average)
  Gvoigt: number | null;       // GPa — shear modulus (Voigt average)
  ael_poisson_ratio: number | null;
  enthalpy_formation_atom: number | null;  // eV/atom
  bandgap: number | null;      // eV
  spinPolarization: number | null;
  Egap_type: string | null;
  volumeAtom: number | null;   // Å³/atom
  densityAtom: number | null;  // g/cm³
}

export interface AflowResult {
  entries: AflowEntry[];
  queryFormula: string;
  source: "AFLOW";
}

function normalizeFormulaForAflow(formula: string): string {
  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "");

  const elementPattern = /([A-Z][a-z]?)(\d*)/g;
  const elements: { el: string; count: string }[] = [];
  let match: RegExpExecArray | null;
  while ((match = elementPattern.exec(cleaned)) !== null) {
    if (match[1]) {
      elements.push({ el: match[1], count: match[2] || "" });
    }
  }
  if (elements.length === 0) return cleaned;

  elements.sort((a, b) => a.el.localeCompare(b.el));
  return elements.map(e => `${e.el}${e.count}`).join("");
}

let aflowConsecutiveFailures = 0;
let aflowBackoffUntil = 0;
let aflowInFlight = 0;

function aflowRecordFailure(reason: string): void {
  if (aflowInFlight <= 1) {
    if (aflowConsecutiveFailures === 0) {
      console.log(`[AFLOW API] ${reason}, backing off`);
    }
    aflowConsecutiveFailures++;
    // Gentler backoff: 10s for first failure (was 30s), scaling up to 5 min max.
    // Single timeouts shouldn't block Vegard lookups for the next 30s worth
    // of materials — different element pairs may respond faster.
    aflowBackoffUntil = Date.now() + Math.min(300000, 10000 * aflowConsecutiveFailures);
  }
}

async function aflowFetch(query: string): Promise<any[] | null> {
  if (Date.now() < aflowBackoffUntil) {
    return null;
  }

  aflowInFlight++;
  try {
    const url = `${AFLOW_API_BASE}/?${query},format(json)`;
    // 45s timeout: AFLOW can be slow for large binary endpoint queries (139+
    // entries for Bi-Ge). The prior 20s was borderline and triggered the
    // backoff circuit breaker, blocking all subsequent Vegard lookups.
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(45000),
    });

    if (!response.ok) {
      if (response.status === 429) {
        aflowRecordFailure("Rate limited");
      }
      return null;
    }

    const text = await response.text();
    if (!text || !text.trim().startsWith("[")) {
      aflowRecordFailure("Non-JSON response from server");
      return null;
    }

    let data: any;
    try {
      data = JSON.parse(text);
    } catch (parseErr: any) {
      aflowRecordFailure(`Malformed JSON body: ${parseErr?.message?.slice(0, 60) ?? "unknown"}`);
      return null;
    }

    aflowConsecutiveFailures = 0;
    if (Array.isArray(data)) return data;
    return null;
  } catch (err: any) {
    if (err?.name !== "AbortError") {
      aflowRecordFailure(`Request failed: ${err?.message || "unknown"}`);
    }
    return null;
  } finally {
    aflowInFlight--;
  }
}

export async function fetchAflowData(formula: string): Promise<AflowResult> {
  const normalized = normalizeFormulaForAflow(formula);

  const query = `compound(${normalized}),paging(1),$auid,$compound,$spacegroup_relax,$sg2,$lattice_system_relax,$ael_bulk_modulus_voigt,$ael_shear_modulus_voigt,$ael_poisson_ratio,$enthalpy_formation_atom,$Egap,$spin_polarization,$Egap_type,$volume_atom,$density`;

  const entries: AflowEntry[] = [];

  const rawEntries = await aflowFetch(query);
  if (rawEntries && rawEntries.length > 0) {
    for (const entry of rawEntries.slice(0, 5)) {
      const bVoigt = entry.ael_bulk_modulus_voigt ?? entry.Bvoigt ?? null;
      const gVoigt = entry.ael_shear_modulus_voigt ?? entry.Gvoigt ?? null;
      const sgNumber = entry.spacegroup_relax ?? entry.sg ?? null;
      entries.push({
        auid: entry.auid ?? "",
        compound: entry.compound ?? normalized,
        spaceGroupNumber: sgNumber ? Number(sgNumber) : 0,
        spaceGroupSymbol: entry.sg2 ?? "",
        latticeSystemRelax: entry.lattice_system_relax ?? "",
        Bvoigt: bVoigt != null ? Number(bVoigt) : null,
        Gvoigt: gVoigt != null ? Number(gVoigt) : null,
        ael_poisson_ratio: entry.ael_poisson_ratio != null ? Number(entry.ael_poisson_ratio) : null,
        enthalpy_formation_atom: entry.enthalpy_formation_atom != null ? Number(entry.enthalpy_formation_atom) : null,
        bandgap: entry.Egap != null ? Number(entry.Egap) : null,
        spinPolarization: entry.spin_polarization != null ? Number(entry.spin_polarization) : null,
        Egap_type: entry.Egap_type ?? null,
        volumeAtom: entry.volume_atom != null ? Number(entry.volume_atom) : null,
        densityAtom: entry.density != null ? Number(entry.density) : null,
      });
    }
  }

  return { entries, queryFormula: normalized, source: "AFLOW" };
}

async function getAflowCachedData(formula: string, dataType: string): Promise<any | null> {
  try {
    const normalizedFormula = normalizeFormulaForAflow(formula);
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

async function setAflowCachedData(formula: string, dataType: string, data: any): Promise<void> {
  try {
    const normalizedFormula = normalizeFormulaForAflow(formula);
    await db
      .insert(mpMaterialCache)
      .values({
        formula: normalizedFormula,
        mpId: null,
        dataType,
        data,
      })
      .onConflictDoUpdate({
        target: [mpMaterialCache.formula, mpMaterialCache.dataType],
        set: {
          data,
          fetchedAt: new Date(),
        },
      });
  } catch {
  }
}

export async function fetchAflowDFTData(formula: string): Promise<AflowDFTData | null> {
  const cached = await getAflowCachedData(formula, "aflow_dft");
  if (cached) return cached as AflowDFTData;

  const normalized = normalizeFormulaForAflow(formula);

  const query = `compound(${normalized}),paging(1),$auid,$compound,$ael_debye,$agl_thermal_conductivity_300K,$energy_atom,$enthalpy_formation_cell`;

  const rawEntries = await aflowFetch(query);
  if (!rawEntries || rawEntries.length === 0) return null;

  const entry = rawEntries[0];

  const hasAnyData =
    entry.ael_debye != null ||
    entry.agl_thermal_conductivity_300K != null ||
    entry.energy_atom != null ||
    entry.enthalpy_formation_cell != null;

  if (!hasAnyData) return null;

  const result: AflowDFTData = {
    formula: normalized,
    ael_debye: entry.ael_debye != null ? Number(entry.ael_debye) : null,
    agl_thermal_conductivity_300K: entry.agl_thermal_conductivity_300K != null ? Number(entry.agl_thermal_conductivity_300K) : null,
    energy_atom: entry.energy_atom != null ? Number(entry.energy_atom) : null,
    enthalpy_formation_cell: entry.enthalpy_formation_cell != null ? Number(entry.enthalpy_formation_cell) : null,
    source: "AFLOW",
  };

  await setAflowCachedData(formula, "aflow_dft", result);
  console.log(`[AFLOW DFT] Cached DFT data for ${normalized}: debye=${result.ael_debye}, thermal_cond=${result.agl_thermal_conductivity_300K}, energy_atom=${result.energy_atom}`);

  return result;
}

export interface CrossValidationResult {
  source: "Materials Project" | "AFLOW";
  property: string;
  predictedValue: number | null;
  externalValue: number;
  deviationPercent: number | null;
  agreement: "match" | "minor-discrepancy" | "major-discrepancy" | "no-comparison";
  unit: string;
}

export interface MPSummary {
  formationEnergyPerAtom: number | null;
  energyAboveHull: number | null;
  bandGap: number | null;
  isMetallic: boolean;
  density: number | null;
}

export interface MPElasticity {
  bulkModulus: number | null;
  shearModulus: number | null;
  poissonRatio: number | null;
}

export function crossValidateWithMP(
  candidate: { predictedTc?: number | null; stabilityScore?: number | null; electronPhononCoupling?: number | null; mlFeatures?: any },
  mpSummary: MPSummary | null,
  mpElasticity: MPElasticity | null,
): CrossValidationResult[] {
  const results: CrossValidationResult[] = [];
  if (!mpSummary && !mpElasticity) return results;

  if (mpSummary) {
    if (mpSummary.energyAboveHull != null) {
      const stability = candidate.stabilityScore ?? null;
      let deviation: number | null = null;
      let agreement: CrossValidationResult["agreement"] = "no-comparison";
      if (stability != null) {
        const mappedExternal = mpSummary.energyAboveHull <= 0 ? 1.0
          : mpSummary.energyAboveHull >= 0.2 ? 0
          : Math.max(0, 1 - mpSummary.energyAboveHull * 5);
        const denominator = Math.max(Math.abs(mappedExternal), Math.abs(stability), 0.001);
        deviation = Math.abs(stability - mappedExternal) / denominator * 100;
        agreement = deviation > 30 ? "major-discrepancy" : deviation > 10 ? "minor-discrepancy" : "match";
      }
      results.push({
        source: "Materials Project",
        property: "Thermodynamic Stability (E above hull)",
        predictedValue: stability,
        externalValue: mpSummary.energyAboveHull,
        deviationPercent: deviation,
        agreement,
        unit: "eV/atom",
      });
    }

    if (mpSummary.formationEnergyPerAtom != null) {
      results.push({
        source: "Materials Project",
        property: "Formation Energy",
        predictedValue: null,
        externalValue: mpSummary.formationEnergyPerAtom,
        deviationPercent: null,
        agreement: "no-comparison",
        unit: "eV/atom",
      });
    }

    if (mpSummary.bandGap != null) {
      const mlFeats = candidate.mlFeatures as Record<string, any> | undefined;
      const predictedBg = mlFeats?.bandgap ?? mlFeats?.bandGap ?? null;
      const candidateExpectsMetal = predictedBg != null ? predictedBg < 0.1
        : (candidate.predictedTc ?? 0) > 0;

      let bgDeviation: number | null = null;
      let bgAgreement: CrossValidationResult["agreement"];

      if (predictedBg != null) {
        const denom = Math.max(Math.abs(mpSummary.bandGap), Math.abs(predictedBg), 0.01);
        bgDeviation = Math.abs(predictedBg - mpSummary.bandGap) / denom * 100;
        bgAgreement = bgDeviation > 50 ? "major-discrepancy"
          : bgDeviation > 20 ? "minor-discrepancy"
          : "match";
      } else if (candidateExpectsMetal) {
        bgAgreement = mpSummary.bandGap < 0.1 ? "match"
          : mpSummary.bandGap > 0.5 ? "major-discrepancy"
          : "minor-discrepancy";
      } else {
        bgAgreement = "no-comparison";
      }

      results.push({
        source: "Materials Project",
        property: "Band Gap",
        predictedValue: predictedBg,
        externalValue: mpSummary.bandGap,
        deviationPercent: bgDeviation,
        agreement: bgAgreement,
        unit: "eV",
      });
    }
  }

  if (mpElasticity) {
    if (mpElasticity.bulkModulus != null) {
      results.push({
        source: "Materials Project",
        property: "Bulk Modulus",
        predictedValue: null,
        externalValue: mpElasticity.bulkModulus,
        deviationPercent: null,
        agreement: "no-comparison",
        unit: "GPa",
      });
    }
    if (mpElasticity.shearModulus != null) {
      results.push({
        source: "Materials Project",
        property: "Shear Modulus",
        predictedValue: null,
        externalValue: mpElasticity.shearModulus,
        deviationPercent: null,
        agreement: "no-comparison",
        unit: "GPa",
      });
    }
  }

  return results;
}

export function crossValidateWithAflow(
  candidate: { predictedTc?: number | null; stabilityScore?: number | null; mlFeatures?: any },
  aflowData: AflowResult,
): CrossValidationResult[] {
  const results: CrossValidationResult[] = [];
  if (aflowData.entries.length === 0) return results;

  const entry = aflowData.entries[0];

  if (entry.enthalpy_formation_atom != null) {
    const stability = candidate.stabilityScore ?? null;
    let deviation: number | null = null;
    let agreement: CrossValidationResult["agreement"] = "no-comparison";
    if (stability != null) {
      const hForm = entry.enthalpy_formation_atom;
      const mappedExternal = hForm <= -2.0 ? 1.0
        : hForm >= 0.5 ? 0
        : Math.max(0, Math.min(1, (0.5 - hForm) / 2.5));
      const denominator = Math.max(Math.abs(mappedExternal), Math.abs(stability), 0.001);
      deviation = Math.abs(stability - mappedExternal) / denominator * 100;
      agreement = deviation > 30 ? "major-discrepancy" : deviation > 10 ? "minor-discrepancy" : "match";
    }
    results.push({
      source: "AFLOW",
      property: "Formation Enthalpy (stability proxy)",
      predictedValue: stability,
      externalValue: entry.enthalpy_formation_atom,
      deviationPercent: deviation,
      agreement,
      unit: "eV/atom",
    });
  }

  if (entry.bandgap != null) {
    const mlFeats = candidate.mlFeatures as Record<string, any> | undefined;
    const predictedBandgap = mlFeats?.bandgap ?? mlFeats?.bandGap ?? null;
    const candidatePredictsMetal = predictedBandgap != null ? predictedBandgap < 0.1
      : (candidate.predictedTc ?? 0) > 0;

    let bandgapDeviation: number | null = null;
    let bandgapAgreement: CrossValidationResult["agreement"];

    if (predictedBandgap != null) {
      const denominator = Math.max(Math.abs(entry.bandgap), Math.abs(predictedBandgap), 0.01);
      bandgapDeviation = Math.abs(predictedBandgap - entry.bandgap) / denominator * 100;
      bandgapAgreement = bandgapDeviation > 50 ? "major-discrepancy"
        : bandgapDeviation > 20 ? "minor-discrepancy"
        : "match";
    } else if (candidatePredictsMetal) {
      bandgapAgreement = entry.bandgap < 0.1 ? "match"
        : entry.bandgap > 0.5 ? "major-discrepancy"
        : "minor-discrepancy";
    } else {
      bandgapAgreement = "no-comparison";
    }

    results.push({
      source: "AFLOW",
      property: "Band Gap",
      predictedValue: predictedBandgap,
      externalValue: entry.bandgap,
      deviationPercent: bandgapDeviation,
      agreement: bandgapAgreement,
      unit: "eV",
    });
  }

  if (entry.Bvoigt != null) {
    results.push({
      source: "AFLOW",
      property: "Bulk Modulus (Voigt)",
      predictedValue: null,
      externalValue: entry.Bvoigt,
      deviationPercent: null,
      agreement: "no-comparison",
      unit: "GPa",
    });
  }

  if (entry.Gvoigt != null) {
    results.push({
      source: "AFLOW",
      property: "Shear Modulus (Voigt)",
      predictedValue: null,
      externalValue: entry.Gvoigt,
      deviationPercent: null,
      agreement: "no-comparison",
      unit: "GPa",
    });
  }

  return results;
}

// ---------------------------------------------------------------------------
// Binary/ternary endpoint lookup for Vegard's law lattice estimation
// ---------------------------------------------------------------------------

export interface AflowStructureEndpoint {
  compound: string;
  elements: string[];
  volumeAtom: number;        // Å³/atom
  latticeSystem: string;     // "cubic" | "hexagonal" | "tetragonal" | ...
  spaceGroupNumber: number;
  spaceGroupSymbol: string;
  enthalpyFormationAtom: number | null;  // eV/atom
  bandgap: number | null;    // eV — 0 means metallic
  source: "AFLOW";
  /** DFT-relaxed lattice parameters [a, b, c, alpha, beta, gamma] from AFLOW geometry field. */
  geometry: [number, number, number, number, number, number] | null;
  /** DFT-relaxed fractional atomic positions from AFLOW. */
  positions: Array<{ element: string; x: number; y: number; z: number }> | null;
}

/**
 * Direct AFLOW fetch for Vegard endpoint lookups.
 * Bypasses the shared aflowFetch() circuit breaker so that engine-cycle
 * AFLOW failures don't block the critical-path Vegard structure lookups.
 */
/**
 * Parse AFLOW AFLUX response body into an array of entry objects.
 *
 * AFLOW responses come in several formats:
 * 1. Array: [{...}, {...}]  (rare, older API)
 * 2. Paginated object: {"1 of 139": {...}, "2 of 139": {...}}
 * 3. Concatenated paginated objects: {"1 of 139": {...},...}{"11 of 139": {...},...}
 *    (when paging returns multiple pages)
 */
function parseAflowResponse(text: string): any[] {
  const trimmed = text.trim();
  if (trimmed.length < 3) return [];

  // Try parsing as a single JSON value first
  try {
    const data = JSON.parse(trimmed);
    if (Array.isArray(data)) return data;
    if (typeof data === "object" && data !== null) {
      return extractPaginatedEntries(data);
    }
    return [];
  } catch {
    // JSON.parse failed — likely concatenated objects: {...}{...}
  }

  // Split concatenated top-level JSON objects by finding balanced braces
  const entries: any[] = [];
  let i = 0;
  while (i < trimmed.length) {
    if (trimmed[i] !== "{") { i++; continue; }
    let depth = 0;
    let endIdx = -1;
    for (let j = i; j < trimmed.length; j++) {
      if (trimmed[j] === "{") depth++;
      else if (trimmed[j] === "}") {
        depth--;
        if (depth === 0) { endIdx = j; break; }
      }
    }
    if (endIdx < 0) break;
    try {
      const obj = JSON.parse(trimmed.slice(i, endIdx + 1));
      if (typeof obj === "object" && obj !== null) {
        entries.push(...extractPaginatedEntries(obj));
      }
    } catch {
      // Skip malformed chunk
    }
    i = endIdx + 1;
  }
  return entries;
}

function extractPaginatedEntries(obj: Record<string, any>): any[] {
  const entries: any[] = [];
  for (const key of Object.keys(obj)) {
    const val = obj[key];
    if (typeof val === "object" && val !== null && !Array.isArray(val)) {
      entries.push(val);
    }
  }
  return entries;
}

async function vegardAflowFetch(query: string): Promise<any[] | null> {
  try {
    const url = `${AFLOW_API_BASE}/?${query},format(json)`;
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(45000),
    });
    if (!response.ok) {
      console.log(`[AFLOW-Vegard] HTTP ${response.status} for: ${query.slice(0, 60)}`);
      return null;
    }
    const text = await response.text();
    const entries = parseAflowResponse(text);
    if (entries.length === 0) {
      // Log a snippet so we can diagnose unexpected formats
      console.log(`[AFLOW-Vegard] 0 entries parsed from ${text.length} bytes: ${text.slice(0, 150)}`);
      return null;
    }
    return entries;
  } catch (err: any) {
    console.log(`[AFLOW-Vegard] Fetch failed: ${err?.message?.slice(0, 80) ?? "unknown"}`);
    return null;
  }
}

/**
 * Fetch all AFLOW entries containing exactly the given elements (binary pair).
 * Uses the AFLUX `species()` + `nspecies()` filters so we get every known
 * stoichiometry for that element pair (e.g., BiGe, Bi2Ge, BiGe2, Bi3Ge5, ...).
 *
 * Uses its own fetch function (vegardAflowFetch) independent of the shared
 * AFLOW circuit breaker, so engine-cycle AFLOW failures don't block Vegard.
 *
 * Results are cached per element pair with 7-day TTL via the existing
 * mpMaterialCache table (dataType = "vegard_endpoint").
 */
export async function fetchAflowByElements(
  el1: string,
  el2: string,
): Promise<AflowStructureEndpoint[]> {
  // Canonical key: alphabetical order
  const sorted = [el1, el2].sort();
  const cacheKey = `${sorted[0]}-${sorted[1]}`;

  // Check cache first
  const cached = await getAflowCachedData(cacheKey, "vegard_endpoint");
  if (cached && Array.isArray(cached) && cached.length > 0) return cached as AflowStructureEndpoint[];

  // AFLUX syntax: species(Bi,Ge) with comma separation, paging(page,perPage),
  // field names WITHOUT $ prefix ($ causes them to be silently ignored).
  // Include geometry + positions_fractional + composition + species_pp for
  // DFT-relaxed atomic positions — these feed VCA position interpolation.
  const query = `species(${sorted[0]},${sorted[1]}),nspecies(2),paging(1,10),volume_atom,lattice_system_relax,enthalpy_formation_atom,Egap,geometry,positions_fractional,composition,species_pp`;

  console.log(`[AFLOW-Vegard] Fetching binary endpoints for ${cacheKey}...`);
  const rawEntries = await vegardAflowFetch(query);
  if (!rawEntries || rawEntries.length === 0) {
    console.log(`[AFLOW-Vegard] No entries for ${cacheKey}`);
    return [];
  }
  console.log(`[AFLOW-Vegard] Got ${rawEntries.length} raw entries for ${cacheKey}`);

  const endpoints: AflowStructureEndpoint[] = [];
  let skippedNoVol = 0;
  for (const entry of rawEntries) {
    // AFLOW may return volume_atom under different keys depending on the catalog
    const volAtom = entry.volume_atom ?? entry.volume_cell ?? entry.volume ?? null;
    const volNum = volAtom != null ? Number(volAtom) : 0;
    // Accept entries even without volume — the compound/spacegroup data is still
    // valuable for prototype matching. Use 0 as placeholder.
    if (!entry.compound && !entry.auid) continue; // Skip truly empty entries

    // Parse DFT-relaxed atomic positions from AFLOW's positions_fractional + species_pp + composition
    let aflowPositions: Array<{ element: string; x: number; y: number; z: number }> | null = null;
    if (entry.positions_fractional && Array.isArray(entry.positions_fractional) &&
        entry.species_pp && Array.isArray(entry.species_pp) &&
        entry.composition && Array.isArray(entry.composition)) {
      aflowPositions = [];
      const species = (entry.species_pp as string[]).map((s: string) => s.replace(/_.*/, "")); // Nb_sv -> Nb
      const comp = entry.composition as number[];
      let atomIdx = 0;
      for (let s = 0; s < species.length && s < comp.length; s++) {
        for (let i = 0; i < comp[s]; i++) {
          if (atomIdx < entry.positions_fractional.length) {
            const pos = entry.positions_fractional[atomIdx] as number[];
            if (pos && pos.length >= 3) {
              aflowPositions.push({ element: species[s], x: pos[0], y: pos[1], z: pos[2] });
            }
          }
          atomIdx++;
        }
      }
      if (aflowPositions.length === 0) aflowPositions = null;
    }

    // Parse geometry [a, b, c, alpha, beta, gamma]
    const geom = entry.geometry && Array.isArray(entry.geometry) && entry.geometry.length >= 6
      ? entry.geometry as [number, number, number, number, number, number]
      : null;

    endpoints.push({
      compound: entry.compound ?? entry.Compound ?? `${sorted[0]}${sorted[1]}`,
      elements: sorted,
      volumeAtom: volNum > 0 ? volNum : 15,
      latticeSystem: (entry.lattice_system_relax ?? entry.lattice ?? "").toLowerCase(),
      spaceGroupNumber: entry.spacegroup_relax ?? entry.sg ?? 0,
      spaceGroupSymbol: entry.sg2 ?? entry.spacegroup ?? "",
      enthalpyFormationAtom: entry.enthalpy_formation_atom != null ? Number(entry.enthalpy_formation_atom) : null,
      bandgap: entry.Egap != null ? Number(entry.Egap) : null,
      source: "AFLOW",
      geometry: geom,
      positions: aflowPositions,
    });
  }

  await setAflowCachedData(cacheKey, "vegard_endpoint", endpoints);
  if (endpoints.length > 0) {
    console.log(`[AFLOW] Cached ${endpoints.length} binary endpoints for ${cacheKey} (vol range: ${endpoints.map(e => e.volumeAtom.toFixed(1)).join(", ")} Å³/atom)`);
  }

  return endpoints;
}

/**
 * Fetch AFLOW endpoints for a ternary element combination.
 * Same approach as binary but with nspecies(3).
 */
export async function fetchAflowByTernaryElements(
  el1: string,
  el2: string,
  el3: string,
): Promise<AflowStructureEndpoint[]> {
  const sorted = [el1, el2, el3].sort();
  const cacheKey = `${sorted[0]}-${sorted[1]}-${sorted[2]}`;

  const cached = await getAflowCachedData(cacheKey, "vegard_endpoint");
  if (cached && Array.isArray(cached) && cached.length > 0) return cached as AflowStructureEndpoint[];

  // AFLUX: comma-separated species, paging(page,perPage), field names without $
  const query = `species(${sorted.join(",")}),nspecies(3),paging(1,5),volume_atom,lattice_system_relax,enthalpy_formation_atom,Egap`;

  // Ternary queries are fire-and-forget enrichment (non-critical) — they often
  // 504 from AFLOW because ternary searches are expensive server-side.
  // Suppress verbose logging to avoid log spam.
  let rawEntries: any[] | null;
  try {
    const url = `${AFLOW_API_BASE}/?${query},format(json)`;
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(20000), // Shorter timeout for non-critical ternary
    });
    if (!response.ok) return []; // Silently skip 504s
    const text = await response.text();
    rawEntries = parseAflowResponse(text);
  } catch {
    return []; // Silently skip timeouts
  }
  if (!rawEntries || rawEntries.length === 0) return [];
  console.log(`[AFLOW-Vegard] Got ${rawEntries.length} ternary entries for ${cacheKey}`);

  const endpoints: AflowStructureEndpoint[] = [];
  for (const entry of rawEntries) {
    const volAtom = entry.volume_atom != null ? Number(entry.volume_atom) : null;
    if (volAtom == null || volAtom <= 0) continue;

    endpoints.push({
      compound: entry.compound ?? sorted.join(""),
      elements: sorted,
      volumeAtom: volAtom,
      latticeSystem: (entry.lattice_system_relax ?? "").toLowerCase(),
      spaceGroupNumber: entry.spacegroup_relax != null ? Number(entry.spacegroup_relax) : 0,
      spaceGroupSymbol: entry.sg2 ?? "",
      enthalpyFormationAtom: entry.enthalpy_formation_atom != null ? Number(entry.enthalpy_formation_atom) : null,
      bandgap: entry.Egap != null ? Number(entry.Egap) : null,
      source: "AFLOW",
      geometry: null,
      positions: null,
    });
  }

  await setAflowCachedData(cacheKey, "vegard_endpoint", endpoints);
  if (endpoints.length > 0) {
    console.log(`[AFLOW] Cached ${endpoints.length} ternary endpoints for ${cacheKey}`);
  }

  return endpoints;
}
