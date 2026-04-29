/**
 * Candidate Metadata & Hydride Chemistry Analysis
 *
 * Every CSP candidate carries structured metadata for:
 * 1. Generation provenance (source, confidence, attempts)
 * 2. Chemistry sanity checks (H-H distances, cage detection, network type)
 * 3. Screening results (MLIP energy, DFT energy, enthalpy, rank)
 *
 * The closed-loop learning system uses this metadata to learn which
 * generators/mutations produce the best structures for each chemistry.
 */

import type { CSPCandidate } from "./csp-types";
import { COVALENT_RADII } from "./csp-types";

// ---------------------------------------------------------------------------
// Generation metadata
// ---------------------------------------------------------------------------

export interface GenerationMetadata {
  source: string;             // "airss" | "pyxtal" | "vegard" | "vca" | "mutant-*"
  sourceConfidence: number;   // 0-1, engine's self-assessed quality
  templateDistance: number | null;  // distance from nearest template (if template-derived)
  volumePriorError: number | null;  // |V_actual - V_estimated| / V_estimated
  minsepViolations: number;   // count of atom pairs closer than MINSEP
  spaceGroup: string;         // detected or assigned SG
  zFormulaUnits: number;      // formula unit multiplicity
  pressureGPa: number;
  generationAttempts: number;  // how many tries before this one succeeded
  relaxationStatus: "not_relaxed" | "xtb_relaxed" | "qe_relax" | "qe_vc_relax";
  mutationParent: number | null;  // parent seed if this is a mutant
  mutationType: string | null;
}

// ---------------------------------------------------------------------------
// Screening results
// ---------------------------------------------------------------------------

export interface ScreeningResults {
  mlipEnergyEvAtom: number | null;   // from xTB or MLIP pre-screen
  mlipMaxForce: number | null;       // max force from pre-screen
  dftEnergyEvAtom: number | null;    // from QE SCF
  enthalpyEvAtom: number | null;     // H = E + PV
  dftMaxForce: number | null;        // residual force after relax
  duplicateClusterId: string | null; // fingerprint cluster assignment
  rank: number | null;               // position in final ranking
}

// ---------------------------------------------------------------------------
// Hydride chemistry analysis
// ---------------------------------------------------------------------------

export type HydrogenNetworkType =
  | "isolated-H"        // H atoms far from each other (>2.5 Å)
  | "H2-molecular"      // H-H pairs at ~0.7-0.9 Å (molecular hydrogen)
  | "extended-chain"     // H atoms form 1D chains
  | "extended-sheet"     // H atoms form 2D layers
  | "clathrate-cage"     // H atoms form 3D cage around metal
  | "sodalite-cage"      // H in sodalite-like cage (Im-3m)
  | "interstitial"       // H in interstitial sites of metal lattice
  | "layered-hydride"    // alternating metal and H layers
  | "unknown";

export interface HydrideAnalysis {
  hCount: number;
  metalCount: number;
  hMetalRatio: number;

  /** H-H nearest-neighbor distances (sorted, in Angstrom). */
  hhDistances: number[];
  /** Average H-H nearest-neighbor distance. */
  hhMeanDist: number;
  /** Minimum H-H distance. */
  hhMinDist: number;

  /** M-H coordination number (avg number of H within 2.5 Å of each metal). */
  mhCoordinationNumber: number;
  /** Volume per H atom (Angstrom^3). */
  volumePerH: number;

  /** Detected hydrogen network type. */
  networkType: HydrogenNetworkType;
  /** Whether this looks like a metallic hydride (high coordination, short H-H). */
  isMetallicHydride: boolean;
  /** Whether this has a cage-like H network (interesting for superconductivity). */
  hasCageStructure: boolean;

  /** Metal sublattice type (FCC, BCC, HCP, or unknown). */
  metalSublattice: string;
}

/**
 * Analyze hydrogen chemistry of a candidate structure.
 */
export function analyzeHydrideChemistry(candidate: CSPCandidate): HydrideAnalysis | null {
  const positions = candidate.positions;
  const a = candidate.latticeA;
  const cOverA = candidate.cOverA ?? 1.0;

  const hAtoms = positions.filter(p => p.element === "H");
  const metalAtoms = positions.filter(p => p.element !== "H");

  if (hAtoms.length === 0) return null;

  const volume = candidate.cellVolume ?? (a * a * a * cOverA);
  const volumePerH = volume / hAtoms.length;

  // Compute H-H nearest-neighbor distances
  const hhDistances: number[] = [];
  for (let i = 0; i < hAtoms.length; i++) {
    let minDist = Infinity;
    for (let j = 0; j < hAtoms.length; j++) {
      if (i === j) continue;
      let dx = hAtoms[i].x - hAtoms[j].x;
      let dy = hAtoms[i].y - hAtoms[j].y;
      let dz = hAtoms[i].z - hAtoms[j].z;
      dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
      const dist = Math.sqrt((dx * a) ** 2 + (dy * a) ** 2 + (dz * a * cOverA) ** 2);
      if (dist < minDist) minDist = dist;
    }
    if (minDist < Infinity) hhDistances.push(minDist);
  }
  hhDistances.sort((a, b) => a - b);

  const hhMeanDist = hhDistances.length > 0
    ? hhDistances.reduce((s, d) => s + d, 0) / hhDistances.length
    : 0;
  const hhMinDist = hhDistances.length > 0 ? hhDistances[0] : 0;

  // Compute M-H coordination numbers
  let totalMHCoord = 0;
  const mhCutoff = 2.5; // Angstrom
  for (const metal of metalAtoms) {
    let coord = 0;
    for (const h of hAtoms) {
      let dx = metal.x - h.x, dy = metal.y - h.y, dz = metal.z - h.z;
      dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
      const dist = Math.sqrt((dx * a) ** 2 + (dy * a) ** 2 + (dz * a * cOverA) ** 2);
      if (dist < mhCutoff) coord++;
    }
    totalMHCoord += coord;
  }
  const mhCoord = metalAtoms.length > 0 ? totalMHCoord / metalAtoms.length : 0;

  // Classify hydrogen network type
  const networkType = classifyHydrogenNetwork(
    hAtoms.length, metalAtoms.length, hhMeanDist, hhMinDist,
    mhCoord, volumePerH, candidate.spaceGroup ?? "",
  );

  const isMetallicHydride = mhCoord >= 4 && hhMeanDist < 2.0;
  const hasCageStructure = networkType === "clathrate-cage" || networkType === "sodalite-cage";

  // Detect metal sublattice type
  const metalSublattice = detectMetalSublattice(metalAtoms, a, cOverA);

  return {
    hCount: hAtoms.length,
    metalCount: metalAtoms.length,
    hMetalRatio: metalAtoms.length > 0 ? hAtoms.length / metalAtoms.length : Infinity,
    hhDistances,
    hhMeanDist,
    hhMinDist,
    mhCoordinationNumber: mhCoord,
    volumePerH,
    networkType,
    isMetallicHydride,
    hasCageStructure,
    metalSublattice,
  };
}

function classifyHydrogenNetwork(
  hCount: number, metalCount: number,
  hhMeanDist: number, hhMinDist: number,
  mhCoord: number, volumePerH: number,
  spaceGroup: string,
): HydrogenNetworkType {
  const hRatio = metalCount > 0 ? hCount / metalCount : Infinity;

  // H2-like molecular units
  if (hhMinDist > 0 && hhMinDist < 0.95) return "H2-molecular";

  // Isolated H (far apart)
  if (hhMeanDist > 2.5 || hCount <= 1) return "isolated-H";

  // Cage structures (high H ratio + moderate H-H distances)
  if (hRatio >= 6 && hhMeanDist < 2.0 && mhCoord >= 8) {
    if (spaceGroup.includes("Im-3m") || spaceGroup.includes("229")) return "sodalite-cage";
    return "clathrate-cage";
  }

  if (hRatio >= 4 && hhMeanDist < 2.0) return "clathrate-cage";

  // Interstitial H in metal lattice
  if (hRatio <= 1 && mhCoord >= 4) return "interstitial";

  // Extended H network
  if (hRatio >= 2 && hhMeanDist < 1.8) {
    if (volumePerH < 5) return "extended-sheet";
    return "extended-chain";
  }

  // Layered hydride
  if (hRatio >= 2 && mhCoord >= 2 && mhCoord <= 6) return "layered-hydride";

  return "unknown";
}

function detectMetalSublattice(
  metalAtoms: Array<{ element: string; x: number; y: number; z: number }>,
  a: number,
  cOverA: number,
): string {
  if (metalAtoms.length === 0) return "none";
  if (metalAtoms.length === 1) return "single-site";

  // Check if metals are on FCC, BCC, or HCP sites
  const hasFCC = metalAtoms.some(m =>
    Math.abs(m.x - 0.5) < 0.05 && Math.abs(m.y - 0.5) < 0.05 && Math.abs(m.z) < 0.05
  );
  const hasBCC = metalAtoms.some(m =>
    Math.abs(m.x - 0.5) < 0.05 && Math.abs(m.y - 0.5) < 0.05 && Math.abs(m.z - 0.5) < 0.05
  );
  const hasHCP = Math.abs(cOverA - 1.633) < 0.3;

  if (hasBCC) return "BCC";
  if (hasFCC) return "FCC";
  if (hasHCP) return "HCP";
  return "unknown";
}

/**
 * Attach metadata to a candidate and log generation stats.
 */
export function attachMetadata(
  candidate: CSPCandidate,
  meta: Partial<GenerationMetadata>,
): void {
  // Store in the candidate's relaxationHistory for now
  // (CSPCandidate doesn't have a metadata field — this piggybacks on existing fields)
  const metaStr = [
    `src=${meta.source ?? candidate.sourceEngine}`,
    `conf=${(meta.sourceConfidence ?? candidate.confidence ?? 0).toFixed(2)}`,
    `sg=${meta.spaceGroup ?? candidate.spaceGroup ?? "?"}`,
    `z=${meta.zFormulaUnits ?? "?"}`,
    `P=${meta.pressureGPa ?? candidate.pressureGPa ?? 0}`,
    meta.minsepViolations ? `minsep_violations=${meta.minsepViolations}` : "",
    meta.mutationType ? `mutation=${meta.mutationType}` : "",
  ].filter(Boolean).join(", ");

  if (!candidate.relaxationHistory) candidate.relaxationHistory = [];
  candidate.relaxationHistory.push(`meta: ${metaStr}`);
}

/**
 * Log a batch of candidates with their generation stats.
 */
export function logCandidateStats(candidates: CSPCandidate[], formula: string): void {
  const bySource: Record<string, number> = {};
  const bySG: Record<string, number> = {};
  let totalAtoms = 0;
  let withH = 0;

  for (const c of candidates) {
    const src = c.prototype || c.sourceEngine || "unknown";
    bySource[src] = (bySource[src] || 0) + 1;
    if (c.spaceGroup) bySG[c.spaceGroup] = (bySG[c.spaceGroup] || 0) + 1;
    totalAtoms += c.positions.length;
    if (c.positions.some(p => p.element === "H")) withH++;
  }

  const avgAtoms = candidates.length > 0 ? (totalAtoms / candidates.length).toFixed(1) : "0";
  const sourceSummary = Object.entries(bySource)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");

  console.log(`[CSP-Stats] ${formula}: ${candidates.length} candidates, avg ${avgAtoms} atoms, ${withH} with H | sources: ${sourceSummary}`);

  // Log hydride analysis for H-containing candidates
  if (withH > 0) {
    const hCandidates = candidates.filter(c => c.positions.some(p => p.element === "H"));
    const analyses = hCandidates.map(c => analyzeHydrideChemistry(c)).filter(Boolean) as HydrideAnalysis[];
    if (analyses.length > 0) {
      const networkTypes: Record<string, number> = {};
      let cageCount = 0;
      for (const a of analyses) {
        networkTypes[a.networkType] = (networkTypes[a.networkType] || 0) + 1;
        if (a.hasCageStructure) cageCount++;
      }
      const networkSummary = Object.entries(networkTypes)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `${k}=${v}`)
        .join(", ");
      console.log(`[CSP-Hydride] ${formula}: ${analyses.length} hydride candidates, ${cageCount} with cage | networks: ${networkSummary}`);
    }
  }
}
