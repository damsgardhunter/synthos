/**
 * TSC Generator Bias
 *
 * Tracks which element families produce topological superconductor (TSC)
 * candidates at runtime and exports weight maps that steer the candidate
 * generators toward proximity-effect materials, heavy-fermion families,
 * and other TSC-relevant chemistries.
 *
 * Design:
 *   - In-memory only (no DB round-trip — this is a hot-path steering signal).
 *   - Bootstrap priors encode known TSC families from the literature.
 *   - Runtime records decay exponentially so recent hits dominate.
 */

interface TSCFamilyRecord {
  formula: string;
  topologicalClass: string;
  elements: string[];
  tscScore: number;
  recordedAt: number;
}

const tscFamilyHistory: TSCFamilyRecord[] = [];
const MAX_HISTORY = 300;
// Weight of a runtime hit halves every 6 hours.
const DECAY_HALF_LIFE_MS = 6 * 60 * 60 * 1000;

// ── Static priors ─────────────────────────────────────────────────────────────
// Known TSC-relevant element families:
//   • Proximity-effect heterostructures: Nb-InAs, Al-InAs nanowires, Pb on TI
//   • Fe-based TSC candidates: FeSe, FeTe (proximity to TI phase)
//   • Heavy-fermion + d-wave: CeCoIn5, UPt3, PrOs4Sb12
//   • Strong SOC metals essential for Majorana gap opening
//   • Topological insulator host elements: Bi2Se3, Bi2Te3
//   • Kagome/Weyl metals where normal SC + topology coexist

const TSC_ELEMENT_PRIORS: Record<string, number> = {
  // Proximity-effect s-wave SC + TI surface states
  Nb: 1.4, In: 1.3, As: 1.2, Sb: 1.3, Bi: 1.4,
  // Fe-based TSC candidates
  Fe: 1.3, Se: 1.4, Te: 1.3,
  // Heavy-fermion + unconventional pairing
  Ce: 1.3, U: 1.2, Pr: 1.2, Sm: 1.1,
  // Strong SOC elements — essential for Majorana topological gap
  Ir: 1.4, Pt: 1.3, Pb: 1.3, Tl: 1.2, Hg: 1.1,
  // Weyl/topological semimetal hosts
  V: 1.2, Sn: 1.2, K: 1.1, Cs: 1.1,
  // Common in TI proximity structures
  Al: 1.1, Mn: 1.1,
};

// Prototypes where TSC signatures frequently appear
const TSC_PROTOTYPE_PRIORS: Record<string, number> = {
  Layered: 1.5,     // FeSe-type, proximity-effect heterostructures
  Heusler: 1.4,     // Half-Heusler half-metals
  ThCr2Si2: 1.3,    // 122-pnictide parent structure
  Kagome: 1.4,      // AV3Sb5-type
  NaCl: 1.2,        // IV-VI narrow-gap semiconductors (Pb1-xSnxTe)
  Pyrochlore: 1.2,  // Topologically non-trivial oxides
};

// ── Runtime recording ─────────────────────────────────────────────────────────

export function recordTSCFamily(
  formula: string,
  topologicalClass: string,
  tscScore: number,
): void {
  const elements = formula.match(/[A-Z][a-z]?/g) ?? [];
  tscFamilyHistory.push({ formula, topologicalClass, elements, tscScore, recordedAt: Date.now() });
  if (tscFamilyHistory.length > MAX_HISTORY) {
    tscFamilyHistory.shift();
  }
}

// ── Bias exports ──────────────────────────────────────────────────────────────

/**
 * Returns element probability weights that boost elements belonging to
 * families with known TSC indicators.  Values > 1.0 increase selection
 * probability in the composition generators.
 */
export function getTSCElementBias(): Map<string, number> {
  const weights = new Map<string, number>();

  // Seed with static priors.
  for (const [el, w] of Object.entries(TSC_ELEMENT_PRIORS)) {
    weights.set(el, w);
  }

  const now = Date.now();
  for (const record of tscFamilyHistory) {
    const age = now - record.recordedAt;
    const decayFactor = Math.pow(0.5, age / DECAY_HALF_LIFE_MS);
    // Class bonus: Majorana > chiral > Z2 > generic
    const classBonus = record.topologicalClass.includes("Majorana") ? 1.6
      : record.topologicalClass.includes("chiral") ? 1.3
      : record.topologicalClass.includes("Weyl") ? 1.2
      : 1.0;
    const boost = record.tscScore * decayFactor * classBonus * 0.8;
    for (const el of record.elements) {
      weights.set(el, (weights.get(el) ?? 1.0) + boost);
    }
  }

  return weights;
}

/**
 * Returns prototype weights for TSC-favourable crystal structures.
 */
export function getTSCPrototypeBias(): Map<string, number> {
  return new Map(Object.entries(TSC_PROTOTYPE_PRIORS));
}

/**
 * Returns preferred pressure range for TSC generation.
 * Many proximity-effect TSCs are ambient; heavy-fermion and some Weyl-SC
 * hybrids stabilise at intermediate pressure (10–50 GPa).
 */
export function getTSCPressureBias(): { min: number; max: number; preferred: number } {
  if (tscFamilyHistory.length < 3) {
    // Default: favour ambient-to-moderate pressure
    return { min: 0, max: 50, preferred: 10 };
  }

  const recent = tscFamilyHistory.slice(-20);
  const hasMajorana = recent.some(r => r.topologicalClass.includes("Majorana"));
  const hasChiral = recent.some(r => r.topologicalClass.includes("chiral"));

  if (hasMajorana) return { min: 0, max: 30, preferred: 5 };
  if (hasChiral) return { min: 10, max: 60, preferred: 25 };
  return { min: 0, max: 50, preferred: 10 };
}

// ── Dashboard / stats exports ─────────────────────────────────────────────────

export interface TSCCandidateSummary {
  formula: string;
  topologicalClass: string;
  tscScore: number;
  recordedAt: number;
}

/** Top N TSC candidates recorded in the current session (highest score first). */
export function getTopTSCCandidates(limit = 20): TSCCandidateSummary[] {
  return [...tscFamilyHistory]
    .sort((a, b) => b.tscScore - a.tscScore)
    .slice(0, limit)
    .map(({ formula, topologicalClass, tscScore, recordedAt }) => ({
      formula, topologicalClass, tscScore, recordedAt,
    }));
}

export function getTSCFamilyStats() {
  const elementCounts = new Map<string, number>();
  const classCounts = new Map<string, number>();

  for (const r of tscFamilyHistory) {
    for (const el of r.elements) {
      elementCounts.set(el, (elementCounts.get(el) ?? 0) + 1);
    }
    classCounts.set(r.topologicalClass, (classCounts.get(r.topologicalClass) ?? 0) + 1);
  }

  return {
    totalRecorded: tscFamilyHistory.length,
    topElements: [...elementCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([element, count]) => ({ element, count })),
    topClasses: [...classCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([tscClass, count]) => ({ tscClass, count })),
  };
}
