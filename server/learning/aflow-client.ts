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
    aflowBackoffUntil = Date.now() + Math.min(300000, 30000 * aflowConsecutiveFailures);
  }
}

async function aflowFetch(query: string): Promise<any[] | null> {
  if (Date.now() < aflowBackoffUntil) {
    return null;
  }

  aflowInFlight++;
  try {
    const url = `${AFLOW_API_BASE}/?${query},format(json)`;
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(20000),
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
