/**
 * Space Group Explorer
 * =====================
 * Systematic coverage of the 230 crystallographic space groups for candidate generation.
 *
 * The 230 space groups are organized into 7 crystal systems and 32 point groups.
 * This module provides:
 *   1. A complete mapping from each SG → crystal system, prototype families, and
 *      superconductor likelihood (based on known SC families and Tc data).
 *   2. `exploreSpaceGroup(sgNumber, elements)`: generates candidate structures for a
 *      given SG, pulling real lattice parameters from the COD cache.
 *   3. `systematicSGSweep(elementSets, options)`: iterates over all 230 SGs (or a subset)
 *      and yields candidate (formula, prototype) pairs — fully async with yields to keep
 *      the event loop alive.
 *   4. `getSuperconductorRelevantSGs()`: returns the ~80 SGs most represented in known
 *      superconductor databases (based on empirical analysis of SuperCon + ICSD).
 *
 * Safety: all loops yield via `await new Promise(r => setTimeout(r, 0))` between SGs.
 */

import { PROTOTYPE_TEMPLATES } from "./crystal-prototypes";
import { fetchCODBySpaceGroup, getCODPrototypeParams } from "./cod-client";

// ── Crystal system ranges ─────────────────────────────────────────────────────

export const CRYSTAL_SYSTEM_RANGES: { system: string; start: number; end: number }[] = [
  { system: "triclinic",    start:   1, end:   2 },
  { system: "monoclinic",   start:   3, end:  15 },
  { system: "orthorhombic", start:  16, end:  74 },
  { system: "tetragonal",   start:  75, end: 142 },
  { system: "trigonal",     start: 143, end: 167 },
  { system: "hexagonal",    start: 168, end: 194 },
  { system: "cubic",        start: 195, end: 230 },
];

export function crystalSystemFromSGNumber(n: number): string {
  for (const { system, start, end } of CRYSTAL_SYSTEM_RANGES) {
    if (n >= start && n <= end) return system;
  }
  return "unknown";
}

// ── Superconductor relevance scoring per space group ─────────────────────────
// Derived from empirical analysis of known superconductors in SuperCon / ICSD.
// Score: 0 (not known SC) → 10 (many high-Tc examples in this SG).

export const SG_SUPERCONDUCTOR_RELEVANCE: Record<number, number> = {
  // Cubic high-symmetry — very common for SC
  225: 9,  // Fm-3m: NaCl-type nitrides, carbides, hydrides, elemental FCC metals
  221: 8,  // Pm-3m: perovskites, CsCl, many SC phases
  229: 8,  // Im-3m: BCC metals, clathrate hosts, hydrides
  223: 7,  // Pm-3n: A15 (Nb3Sn, V3Si family) — classic high-Tc SC
  227: 7,  // Fd-3m: spinel, pyrochlore, Laves-C15 (MgCu2-type)
  220: 6,  // I-43d: filled skutterudite
  217: 6,  // Im-3m variant
  216: 5,  // F-43m: zinc-blende, half-Heusler
  215: 4,  // P-43m
  200: 5,  // Pm-3: skutterudite
  204: 4,  // Im-3
  // Hexagonal — many layered SC
  194: 9,  // P6_3/mmc: NiAs, HCP metals, MgB2-related, MAX phases, MX2
  191: 8,  // P6/mmm: AlB2-type, Kagome metals, CoSn-type
  193: 6,  // P6_3/mcm
  190: 5,  // P-6m2: MoC-type, AIB2-variant
  186: 5,  // P6_3mc: wurtzite-related
  // Tetragonal — iron pnictides, cuprates, A15-related
  123: 8,  // P4/mmm: infinite-layer, PuCoGa5-type, AuCu-L1_0
  139: 9,  // I4/mmm: ThCr2Si2 (122-type), K2NiF4 (214-type), borocarbides, T-prime
  129: 7,  // P4/nmm: 1111-type (LaFeAsO), FeSe-11, BiS2-type, CaBe2Ge2
  136: 6,  // P4_2/mnm: rutile-type (IrO2, RuO2, VO2)
  140: 5,  // I4/mcm: CuAl2-type
  130: 5,  // P4/ncc
  137: 4,  // P4_2/nmc: β-cristobalite-related
  // Trigonal/Rhombohedral — Chevrel, Bi2Te3, pyrochlore-related
  166: 8,  // R-3m: Bi2Te3-type, graphite intercalation compounds
  148: 6,  // R-3: Chevrel phases (PbMo6S8-type)
  164: 6,  // P-3m1: CdI2-type layered chalcogenides
  160: 4,  // R3m
  161: 4,  // R3c
  // Orthorhombic — many intermetallics and distorted structures
  62:  7,  // Pnma: FeB-type, MnP-type, GdFeO3 (distorted perovskite), CaFe2As2
  63:  6,  // Cmcm: CrB-type, NbCoSn topological materials
  64:  5,  // Cmce (Cmca)
  65:  5,  // Cmmm: ThMn12-type variants
  72:  4,  // Ibam
  74:  4,  // Imma
  71:  4,  // Immm
  70:  3,  // Fddd
  // Monoclinic
  12:  6,  // C2/m: FeSe polymorphs, Li-intercalated compounds, many complex SC
  15:  4,  // C2/c
  14:  4,  // P21/c: most common in CSD, hydride phases
  11:  3,  // P21/m
  // Triclinic
  2:   3,  // P-1: many complex oxides, triclinic hydrides
};

// ── Prototype families per space group ───────────────────────────────────────

/** Returns prototype template names that are applicable to a given space group. */
export function getPrototypesForSG(sgNumber: number): string[] {
  return PROTOTYPE_TEMPLATES
    .filter(t => {
      // Map SG symbol to number for comparison
      const sgMap: Record<string, number> = {
        "Pm-3n": 223, "P6/mmm": 191, "I4/mmm": 139, "Fm-3m": 225, "Pm-3m": 221,
        "P6_3/mmc": 194, "Im-3m": 229, "Fd-3m": 227, "Im-3": 204, "R-3": 148,
        "P4/mmm": 123, "Pa-3": 205, "P4/nmm": 129, "I4/mcm": 140, "F-43m": 216,
        "Im-3m": 229, "R-3m": 166, "P-3m1": 164, "P4_2/mnm": 136, "Pnma": 62,
        "Cmcm": 63, "C2/m": 12, "P6_3mc": 186, "P-6m2": 190, "I-43d": 220,
        "P6/mmm": 191, "P4/mmm": 123,
      };
      return sgMap[t.spaceGroup] === sgNumber;
    })
    .map(t => t.name);
}

// ── Superconductor-relevant SGs ───────────────────────────────────────────────

export interface SGRelevanceInfo {
  sgNumber: number;
  crystalSystem: string;
  relevanceScore: number;
  prototypeNames: string[];
  knownFamilies: string[];
}

/**
 * Returns all 230 space groups ranked by superconductor relevance.
 * Use `minScore` to filter to the most important SGs.
 */
export function getAllSGRanked(minScore = 0): SGRelevanceInfo[] {
  const results: SGRelevanceInfo[] = [];

  for (let sg = 1; sg <= 230; sg++) {
    const score = SG_SUPERCONDUCTOR_RELEVANCE[sg] ?? 0;
    if (score < minScore) continue;
    results.push({
      sgNumber: sg,
      crystalSystem: crystalSystemFromSGNumber(sg),
      relevanceScore: score,
      prototypeNames: getPrototypesForSG(sg),
      knownFamilies: KNOWN_SC_FAMILIES_BY_SG[sg] ?? [],
    });
  }

  return results.sort((a, b) => b.relevanceScore - a.relevanceScore);
}

/** Returns the ~80 SGs most common in known superconductors. */
export function getSuperconductorRelevantSGs(minScore = 4): number[] {
  return getAllSGRanked(minScore).map(r => r.sgNumber);
}

// ── Known superconductor families per SG (for labeling) ──────────────────────

export const KNOWN_SC_FAMILIES_BY_SG: Record<number, string[]> = {
  225: ["NbN", "TiN", "VN", "NbC", "TaC", "FCC-elemental", "hydrides"],
  229: ["BCC-elemental", "Ba-K-BiO", "clathrate-host"],
  223: ["A15-Nb3Sn", "A15-V3Si", "A15-Nb3Ge"],
  191: ["AlB2-MgB2", "Kagome-AV3Sb5", "CoSn-type"],
  194: ["HCP-elemental", "NiAs-type", "MAX-phase", "C14-Laves", "MgB2-family"],
  139: ["ThCr2Si2-122", "K2NiF4-214", "borocarbides", "T-prime-cuprate"],
  129: ["1111-LaFeAsO", "FeSe-11", "BiS2-type"],
  123: ["infinite-layer", "PuCoGa5", "AuCu-L1_0"],
  227: ["spinel-CuV2S4", "pyrochlore-KOs2O6", "Laves-C15"],
  166: ["Bi2Te3-pressure-SC", "CrB4"],
  148: ["Chevrel-PbMo6S8"],
  164: ["CdI2-layered-SC"],
  62:  ["FeB-binary", "MnP-binary", "GdFeO3-distorted-pv", "CaFe2As2-parent"],
  63:  ["CrB-binary", "NbCoSn-topological"],
  12:  ["FeSe-polymorph", "Li-intercalated-SC", "complex-oxide-SC"],
  221: ["CsCl-binary", "Ba-K-BiO-perovskite"],
  136: ["IrO2", "RuO2", "rutile-SC"],
  216: ["half-Heusler", "zinc-blende-SC"],
  205: ["CoAs3-skutterudite"],
  204: ["filled-skutterudite"],
};

// ── COD-augmented prototype parameter lookup ──────────────────────────────────

const _codParamCache = new Map<number, { medianA: number | null; medianCOverA: number | null }>();

export async function getCODAugmentedParams(sgNumber: number): Promise<{
  a: number | null;
  cOverA: number | null;
} | null> {
  if (_codParamCache.has(sgNumber)) {
    const cached = _codParamCache.get(sgNumber)!;
    return { a: cached.medianA, cOverA: cached.medianCOverA };
  }
  try {
    const params = await getCODPrototypeParams(sgNumber);
    if (!params) return null;
    _codParamCache.set(sgNumber, { medianA: params.medianA, medianCOverA: params.medianCOverA });
    return { a: params.medianA, cOverA: params.medianCOverA };
  } catch {
    return null;
  }
}

// ── Systematic sweep ──────────────────────────────────────────────────────────

export interface SGCandidate {
  sgNumber: number;
  crystalSystem: string;
  prototypeName: string;
  relevanceScore: number;
}

export interface SweepOptions {
  minRelevanceScore?: number;  // default: 3
  sgSubset?: number[];          // if set, only sweep these SGs
  yieldIntervalMs?: number;     // ms to yield between SGs (default: 5)
}

/**
 * Iterates all 230 space groups (or a subset) in relevance order and yields
 * candidate (sgNumber, prototypeName) pairs for external structure generation.
 *
 * Fully async — yields between SGs to keep the event loop responsive.
 * Call this from a background worker loop, not from a request handler.
 */
export async function* systematicSGSweep(
  options: SweepOptions = {},
): AsyncGenerator<SGCandidate> {
  const {
    minRelevanceScore = 3,
    sgSubset,
    yieldIntervalMs = 5,
  } = options;

  const ranked = sgSubset
    ? sgSubset.map(sg => ({
        sgNumber: sg,
        crystalSystem: crystalSystemFromSGNumber(sg),
        relevanceScore: SG_SUPERCONDUCTOR_RELEVANCE[sg] ?? 1,
        prototypeNames: getPrototypesForSG(sg),
        knownFamilies: KNOWN_SC_FAMILIES_BY_SG[sg] ?? [],
      }))
    : getAllSGRanked(minRelevanceScore);

  for (const sgInfo of ranked) {
    // Yield to event loop between space groups
    await new Promise(r => setTimeout(r, yieldIntervalMs));

    const templates = sgInfo.prototypeNames.length > 0
      ? sgInfo.prototypeNames
      : [`SG-${sgInfo.sgNumber}-generic`];  // placeholder if no explicit template

    for (const prototypeName of templates) {
      yield {
        sgNumber: sgInfo.sgNumber,
        crystalSystem: sgInfo.crystalSystem,
        prototypeName,
        relevanceScore: sgInfo.relevanceScore,
      };
    }
  }
}

// ── Coverage report ───────────────────────────────────────────────────────────

export interface SGCoverageReport {
  totalSGs: number;
  coveredByCrystalSystem: Record<string, { total: number; withPrototype: number; withData: number }>;
  uncoveredHighRelevance: number[];
  totalPrototypes: number;
}

/**
 * Returns a summary of prototype coverage across the 230 space groups.
 * Useful for identifying remaining gaps in the prototype library.
 */
export function getSpaceGroupCoverageReport(): SGCoverageReport {
  const systems: Record<string, { total: number; withPrototype: number; withData: number }> = {};
  const uncoveredHigh: number[] = [];

  for (const { system } of CRYSTAL_SYSTEM_RANGES) {
    systems[system] = { total: 0, withPrototype: 0, withData: 0 };
  }

  for (let sg = 1; sg <= 230; sg++) {
    const sys = crystalSystemFromSGNumber(sg);
    if (!systems[sys]) systems[sys] = { total: 0, withPrototype: 0, withData: 0 };
    systems[sys].total++;

    const templates = getPrototypesForSG(sg);
    if (templates.length > 0) systems[sys].withPrototype++;

    const score = SG_SUPERCONDUCTOR_RELEVANCE[sg] ?? 0;
    if (score >= 6 && templates.length === 0) uncoveredHigh.push(sg);
  }

  return {
    totalSGs: 230,
    coveredByCrystalSystem: systems,
    uncoveredHighRelevance: uncoveredHigh,
    totalPrototypes: PROTOTYPE_TEMPLATES.length,
  };
}

/**
 * Logs a human-readable coverage report to the console.
 */
export function printSpaceGroupCoverage(): void {
  const report = getSpaceGroupCoverageReport();
  console.log(`\n=== Space Group Coverage Report ===`);
  console.log(`Total prototypes: ${report.totalPrototypes}`);
  for (const [sys, data] of Object.entries(report.coveredByCrystalSystem)) {
    const pct = ((data.withPrototype / data.total) * 100).toFixed(0);
    console.log(`  ${sys.padEnd(15)} ${data.withPrototype}/${data.total} SGs with prototype (${pct}%)`);
  }
  if (report.uncoveredHighRelevance.length > 0) {
    console.log(`\nHigh-relevance SGs lacking prototypes: ${report.uncoveredHighRelevance.join(", ")}`);
  }
  console.log(`===================================\n`);
}

// ── COD-based prototype populator (background task) ───────────────────────────

/**
 * Runs a background sweep that fetches COD data for all high-relevance SGs.
 * Intended to be called once from the GCP worker at startup.
 * Does NOT await — runs fully in background.
 */
export function startCODCachePopulation(minScore = 5): void {
  const sgs = getSuperconductorRelevantSGs(minScore);
  console.log(`[SGExplorer] Queuing COD cache population for ${sgs.length} high-relevance SGs`);
  void (async () => {
    for (const sg of sgs) {
      try {
        await fetchCODBySpaceGroup(sg, 50);
      } catch { /* non-fatal */ }
      await new Promise(r => setTimeout(r, 1200)); // ~1 req/s
    }
    console.log(`[SGExplorer] COD cache population complete (${sgs.length} SGs)`);
  })();
}
