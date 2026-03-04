const AFLOW_API_BASE = "http://aflowlib.duke.edu/API/aflux";

export interface AflowEntry {
  auid: string;
  compound: string;
  spaceGroupNumber: number;
  spaceGroupSymbol: string;
  latticeSystemRelax: string;
  Bvoigt: number | null;
  Gvoigt: number | null;
  ael_poisson_ratio: number | null;
  enthalpy_formation_atom: number | null;
  bandgap: number | null;
  spinPolarization: number | null;
  Egap_type: string | null;
  volumeAtom: number | null;
  densityAtom: number | null;
}

export interface AflowResult {
  entries: AflowEntry[];
  queryFormula: string;
  source: "AFLOW";
}

function normalizeFormulaForAflow(formula: string): string {
  return formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "");
}

async function aflowFetch(query: string): Promise<any[] | null> {
  try {
    const url = `${AFLOW_API_BASE}/?${query},format(json)`;
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
      signal: AbortSignal.timeout(20000),
    });

    if (!response.ok) {
      if (response.status === 429) {
        console.log("[AFLOW API] Rate limited");
      }
      return null;
    }

    const data = await response.json();
    if (Array.isArray(data)) return data;
    return null;
  } catch (err: any) {
    if (err?.name !== "AbortError") {
      console.log(`[AFLOW API] Request failed: ${err?.message || "unknown"}`);
    }
    return null;
  }
}

export async function fetchAflowData(formula: string): Promise<AflowResult> {
  const normalized = normalizeFormulaForAflow(formula);

  const query = `compound(${normalized}),paging(1),$auid,$compound,$sg,$sg2,$lattice_system_relax,$Bvoigt,$Gvoigt,$ael_poisson_ratio,$enthalpy_formation_atom,$Egap,$spin_polarization,$Egap_type,$volume_atom,$density`;

  const entries: AflowEntry[] = [];

  const rawEntries = await aflowFetch(query);
  if (rawEntries && rawEntries.length > 0) {
    for (const entry of rawEntries.slice(0, 5)) {
      entries.push({
        auid: entry.auid ?? "",
        compound: entry.compound ?? normalized,
        spaceGroupNumber: entry.sg ? Number(entry.sg) : 0,
        spaceGroupSymbol: entry.sg2 ?? "",
        latticeSystemRelax: entry.lattice_system_relax ?? "",
        Bvoigt: entry.Bvoigt != null ? Number(entry.Bvoigt) : null,
        Gvoigt: entry.Gvoigt != null ? Number(entry.Gvoigt) : null,
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

export interface CrossValidationResult {
  source: "Materials Project" | "AFLOW";
  property: string;
  predictedValue: number | null;
  externalValue: number;
  deviationPercent: number | null;
  agreement: "match" | "minor-discrepancy" | "major-discrepancy" | "no-comparison";
  unit: string;
}

export function crossValidateWithMP(
  candidate: { predictedTc?: number | null; stabilityScore?: number | null; electronPhononCoupling?: number | null },
  mpSummary: { formationEnergyPerAtom: number; energyAboveHull: number; bandGap: number; isMetallic: boolean; density: number } | null,
  mpElasticity: { bulkModulus: number; shearModulus: number; poissonRatio: number } | null,
): CrossValidationResult[] {
  const results: CrossValidationResult[] = [];
  if (!mpSummary && !mpElasticity) return results;

  if (mpSummary) {
    if (mpSummary.energyAboveHull != null) {
      const stability = candidate.stabilityScore ?? null;
      let deviation: number | null = null;
      let agreement: CrossValidationResult["agreement"] = "no-comparison";
      if (stability != null) {
        const mappedExternal = mpSummary.energyAboveHull <= 0 ? 1.0 : Math.max(0, 1 - mpSummary.energyAboveHull * 5);
        deviation = stability > 0 ? Math.abs(stability - mappedExternal) / stability * 100 : null;
        agreement = deviation != null ? (deviation > 30 ? "major-discrepancy" : deviation > 10 ? "minor-discrepancy" : "match") : "no-comparison";
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
      results.push({
        source: "Materials Project",
        property: "Band Gap",
        predictedValue: null,
        externalValue: mpSummary.bandGap,
        deviationPercent: null,
        agreement: mpSummary.isMetallic ? "match" : (mpSummary.bandGap > 0.5 ? "major-discrepancy" : "match"),
        unit: "eV",
      });
    }
  }

  if (mpElasticity) {
    if (mpElasticity.bulkModulus > 0) {
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
  }

  return results;
}

export function crossValidateWithAflow(
  candidate: { predictedTc?: number | null; stabilityScore?: number | null },
  aflowData: AflowResult,
): CrossValidationResult[] {
  const results: CrossValidationResult[] = [];
  if (aflowData.entries.length === 0) return results;

  const entry = aflowData.entries[0];

  if (entry.enthalpy_formation_atom != null) {
    const stability = candidate.stabilityScore ?? null;
    let deviation: number | null = null;
    let agreement: CrossValidationResult["agreement"] = "no-comparison";
    if (stability != null && stability > 0) {
      const mappedExternal = entry.enthalpy_formation_atom < 0 ? Math.min(1, Math.abs(entry.enthalpy_formation_atom) / 2) : 0;
      deviation = Math.abs(stability - mappedExternal) / stability * 100;
      agreement = deviation > 30 ? "major-discrepancy" : deviation > 10 ? "minor-discrepancy" : "match";
    }
    results.push({
      source: "AFLOW",
      property: "Formation Enthalpy",
      predictedValue: stability,
      externalValue: entry.enthalpy_formation_atom,
      deviationPercent: deviation,
      agreement,
      unit: "eV/atom",
    });
  }

  if (entry.bandgap != null) {
    results.push({
      source: "AFLOW",
      property: "Band Gap",
      predictedValue: null,
      externalValue: entry.bandgap,
      deviationPercent: null,
      agreement: entry.bandgap < 0.1 ? "match" : (entry.bandgap > 0.5 ? "major-discrepancy" : "minor-discrepancy"),
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
