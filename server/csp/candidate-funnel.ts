/**
 * Candidate Funnel: F0 → F8
 *
 * Strict multi-stage filter that reduces thousands of raw candidates
 * down to a few DFT-worthy structures. Each stage has hard rejections
 * and soft penalties that feed into the admission score.
 *
 * F0: Basic parse + normalization
 * F1: Geometry hard filter (minsep, volume, cell shape)
 * F2: General chemistry sanity (coordination, isolation, density)
 * F3: Hydride-specific sanity (H network, cage detection, H2 penalty)
 * F4: Raw duplicate removal (cheap fingerprint)
 * F5: Fast scoring + confidence update
 * F6: (Optional) MLIP pre-relaxation placeholder
 * F7: Relaxed duplicate removal + clustering
 * F8: DFT admission selection
 *
 * Target keep rates (preview mode):
 *   Raw: ~85 → F1: ~65 → F4: ~40 → F7: ~15 clusters → F8: 3-5 DFT
 */

import type { CSPCandidate, ScreeningTierConfig } from "./csp-types";
import { COVALENT_RADII, getPairMinsep, SCREENING_TIERS } from "./csp-types";
import { analyzeHydrideChemistry, type HydrideAnalysis } from "./candidate-metadata";
import { deduplicateCandidates, clusterCandidates, type StructureCluster } from "./dedup-cluster";
import { selectForDFT, type DFTSelectionResult } from "./dft-admission";
import { recordVolumeOutcome, recordGeneratorOutcome, logLearningState } from "./adaptive-learning";

// ---------------------------------------------------------------------------
// Funnel stats (for logging + learning)
// ---------------------------------------------------------------------------

export interface FunnelStats {
  f0_input: number;
  f0_rejected: number;
  f1_rejected: number;
  f2_rejected: number;
  f3_penalized: number;
  f4_duplicates: number;
  f5_scored: number;
  f7_clusters: number;
  f8_selected: number;
  rejectionReasons: Record<string, number>;
}

interface ScoredCandidate {
  candidate: CSPCandidate;
  geometryScore: number;      // F1
  chemistryScore: number;     // F2
  hydrideScore: number;       // F3
  preScore: number;           // F5
  hydrideAnalysis: HydrideAnalysis | null;
}

// ---------------------------------------------------------------------------
// Source confidence priors
// ---------------------------------------------------------------------------

const SOURCE_BASE_SCORE: Record<string, number> = {
  "prototype": 0.80,
  "TemplateVCA": 0.65,
  "VCA-interpolated": 0.55,
  "AIRSS-buildcell": 0.60,
  "PyXtal-random": 0.65,
  "mutant-lattice-strain": 0.70,
  "mutant-volume-compress": 0.65,
  "mutant-volume-expand": 0.65,
  "mutant-atomic-displacement": 0.70,
  "mutant-symmetry-break": 0.60,
  "mutant-hydrogen-shuffle": 0.75,
  "mutant-wyckoff-perturbation": 0.70,
  "volume-sum": 0.25,
  "literature": 0.95,
  "MP-direct": 0.90,
};

function getSourceScore(prototype: string): number {
  if (SOURCE_BASE_SCORE[prototype]) return SOURCE_BASE_SCORE[prototype];
  if (prototype.startsWith("mutant-")) return 0.65;
  if (prototype.startsWith("TemplateVCA")) return 0.65;
  return 0.40;
}

// ---------------------------------------------------------------------------
// F0: Basic parse + normalization
// ---------------------------------------------------------------------------

function f0_parse(candidates: CSPCandidate[], stats: FunnelStats): CSPCandidate[] {
  const passed: CSPCandidate[] = [];
  for (const c of candidates) {
    stats.f0_input++;

    // Hard rejections
    if (!c.positions || c.positions.length === 0) {
      stats.f0_rejected++;
      stats.rejectionReasons["f0_no_positions"] = (stats.rejectionReasons["f0_no_positions"] ?? 0) + 1;
      continue;
    }
    if (!c.latticeA || c.latticeA <= 0) {
      stats.f0_rejected++;
      stats.rejectionReasons["f0_bad_lattice"] = (stats.rejectionReasons["f0_bad_lattice"] ?? 0) + 1;
      continue;
    }
    if (c.positions.some(p => isNaN(p.x) || isNaN(p.y) || isNaN(p.z))) {
      stats.f0_rejected++;
      stats.rejectionReasons["f0_nan_coords"] = (stats.rejectionReasons["f0_nan_coords"] ?? 0) + 1;
      continue;
    }

    // Normalize: wrap fractional coords to [0, 1)
    for (const pos of c.positions) {
      pos.x = ((pos.x % 1) + 1) % 1;
      pos.y = ((pos.y % 1) + 1) % 1;
      pos.z = ((pos.z % 1) + 1) % 1;
    }

    // Sort species consistently
    c.positions.sort((a, b) => a.element.localeCompare(b.element));

    // Calculate volume if not set
    if (!c.cellVolume) {
      const cOverA = c.cOverA ?? 1.0;
      c.cellVolume = c.latticeA * c.latticeA * c.latticeA * cOverA;
    }

    passed.push(c);
  }
  return passed;
}

// ---------------------------------------------------------------------------
// F1: Geometry hard filter
// ---------------------------------------------------------------------------

function f1_geometry(
  candidates: CSPCandidate[],
  pressureGPa: number,
  stats: FunnelStats,
): ScoredCandidate[] {
  const passed: ScoredCandidate[] = [];

  for (const c of candidates) {
    const a = c.latticeA;
    const cOverA = c.cOverA ?? 1.0;
    const n = c.positions.length;
    const vol = c.cellVolume ?? (a * a * a * cOverA);
    const volPerAtom = vol / Math.max(1, n);

    let rejected = false;
    let geometryScore = 1.0;
    const elements = [...new Set(c.positions.map(p => p.element))];

    // Check minimum pair distances (pressure-aware)
    for (let i = 0; i < n && !rejected; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = c.positions[i].x - c.positions[j].x;
        let dy = c.positions[i].y - c.positions[j].y;
        let dz = c.positions[i].z - c.positions[j].z;
        dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
        const dist = Math.sqrt((dx * a) ** 2 + (dy * a) ** 2 + (dz * a * cOverA) ** 2);
        const minsep = getPairMinsep(c.positions[i].element, c.positions[j].element, pressureGPa);

        if (dist < minsep * 0.5) {
          // Hard reject: atoms WAY too close (< half minsep)
          rejected = true;
          stats.rejectionReasons["f1_atom_overlap"] = (stats.rejectionReasons["f1_atom_overlap"] ?? 0) + 1;
          break;
        }
        if (dist < minsep) {
          // Soft penalty: close but not overlapping
          geometryScore -= 0.05;
        }
      }
    }
    if (rejected) { stats.f1_rejected++; continue; }

    // Volume per atom range (pressure-dependent)
    const minVPA = pressureGPa > 100 ? 2.0 : 4.0;
    const maxVPA = 80;
    if (volPerAtom < minVPA || volPerAtom > maxVPA) {
      stats.f1_rejected++;
      stats.rejectionReasons["f1_bad_volume"] = (stats.rejectionReasons["f1_bad_volume"] ?? 0) + 1;
      continue;
    }

    // Cell aspect ratio
    const bOverA = (c.latticeB ?? a) / a;
    const cOverAVal = (c.latticeC ?? a * cOverA) / a;
    if (bOverA > 8 || cOverAVal > 8 || bOverA < 0.125 || cOverAVal < 0.125) {
      stats.f1_rejected++;
      stats.rejectionReasons["f1_bad_aspect"] = (stats.rejectionReasons["f1_bad_aspect"] ?? 0) + 1;
      continue;
    }

    geometryScore = Math.max(0, Math.min(1, geometryScore));
    passed.push({ candidate: c, geometryScore, chemistryScore: 0, hydrideScore: 0.5, preScore: 0, hydrideAnalysis: null });
  }

  return passed;
}

// ---------------------------------------------------------------------------
// F2: General chemistry sanity
// ---------------------------------------------------------------------------

function f2_chemistry(scored: ScoredCandidate[], stats: FunnelStats): ScoredCandidate[] {
  const passed: ScoredCandidate[] = [];

  for (const s of scored) {
    const c = s.candidate;
    const a = c.latticeA;
    const cOverA = c.cOverA ?? 1.0;
    const n = c.positions.length;

    let chemScore = 1.0;
    let rejected = false;

    // Check for isolated atoms (no neighbors within cutoff)
    const ISOLATION_CUTOFF = a * 0.4;
    for (let i = 0; i < n; i++) {
      let hasNeighbor = false;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        let dx = c.positions[i].x - c.positions[j].x;
        let dy = c.positions[i].y - c.positions[j].y;
        let dz = c.positions[i].z - c.positions[j].z;
        dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
        const dist = Math.sqrt((dx * a) ** 2 + (dy * a) ** 2 + (dz * a * cOverA) ** 2);
        if (dist < ISOLATION_CUTOFF) { hasNeighbor = true; break; }
      }
      if (!hasNeighbor) {
        chemScore -= 0.15; // Penalty for isolated atoms
      }
    }

    // Unphysical density check
    const vol = c.cellVolume ?? (a * a * a * cOverA);
    const density = n / vol; // atoms per Å³
    if (density > 0.2) {
      chemScore -= 0.3; // Very dense — suspicious
    }

    if (chemScore < 0.1) {
      stats.f2_rejected++;
      stats.rejectionReasons["f2_chemistry_fail"] = (stats.rejectionReasons["f2_chemistry_fail"] ?? 0) + 1;
      continue;
    }

    s.chemistryScore = Math.max(0, Math.min(1, chemScore));
    passed.push(s);
  }

  return passed;
}

// ---------------------------------------------------------------------------
// F3: Hydride-specific sanity
// ---------------------------------------------------------------------------

function f3_hydride(scored: ScoredCandidate[], stats: FunnelStats): ScoredCandidate[] {
  for (const s of scored) {
    const analysis = analyzeHydrideChemistry(s.candidate);
    s.hydrideAnalysis = analysis;

    if (!analysis) {
      s.hydrideScore = 0.5; // non-hydride — neutral
      continue;
    }

    let score = 0.5;

    // Reward extended H networks and cages
    switch (analysis.networkType) {
      case "clathrate-cage": score = 0.95; break;
      case "sodalite-cage": score = 0.90; break;
      case "extended-sheet": score = 0.80; break;
      case "extended-chain": score = 0.75; break;
      case "layered-hydride": score = 0.70; break;
      case "interstitial": score = 0.60; break;
      case "isolated-H": score = 0.35; break;
      case "H2-molecular": score = 0.20; break;
      default: score = 0.50;
    }

    // M-H coordination bonus
    if (analysis.mhCoordinationNumber >= 8) score = Math.min(1, score + 0.10);
    if (analysis.mhCoordinationNumber >= 12) score = Math.min(1, score + 0.05);

    // H2-like penalty (but don't hard reject)
    if (analysis.hhMinDist > 0 && analysis.hhMinDist < 0.9) {
      score -= 0.15;
      stats.f3_penalized++;
    }

    s.hydrideScore = Math.max(0, Math.min(1, score));
  }

  return scored;
}

// ---------------------------------------------------------------------------
// F5: Fast scoring + confidence update
// ---------------------------------------------------------------------------

function f5_score(scored: ScoredCandidate[]): ScoredCandidate[] {
  for (const s of scored) {
    const sourceScore = getSourceScore(s.candidate.prototype ?? "");

    // Volume prior score: how close is the volume to typical values?
    const vol = s.candidate.cellVolume ?? 100;
    const volPerAtom = vol / Math.max(1, s.candidate.positions.length);
    const volScore = volPerAtom >= 5 && volPerAtom <= 30 ? 1.0 :
                     volPerAtom >= 3 && volPerAtom <= 50 ? 0.7 : 0.3;

    // Symmetry reasonableness (having a space group is good)
    const symmScore = s.candidate.spaceGroup ? 0.8 : 0.4;

    // Diversity / novelty
    const proto = s.candidate.prototype ?? "";
    const diversityScore = proto.startsWith("mutant-") ? 0.7 :
                          proto === "AIRSS-buildcell" ? 0.6 :
                          proto === "PyXtal-random" ? 0.65 : 0.5;

    // Prototype confidence
    const protoScore = proto === "literature" ? 1.0 :
                       proto.startsWith("TemplateVCA") ? 0.7 :
                       proto === "VCA-interpolated" ? 0.6 : 0.4;

    s.preScore =
      0.25 * s.geometryScore +
      0.20 * s.hydrideScore +
      0.15 * sourceScore +
      0.15 * volScore +
      0.10 * symmScore +
      0.10 * diversityScore +
      0.05 * protoScore;

    // Update candidate confidence with the pre-score
    s.candidate.confidence = Math.max(s.candidate.confidence ?? 0, s.preScore);
  }

  stats_f5_count = scored.length;
  return scored;
}
let stats_f5_count = 0;

// ---------------------------------------------------------------------------
// Main funnel
// ---------------------------------------------------------------------------

export interface FunnelResult {
  /** Candidates selected for DFT. */
  selected: CSPCandidate[];
  /** All clusters found. */
  clusters: StructureCluster[];
  /** DFT selection with scores. */
  dftSelection: DFTSelectionResult | null;
  /** Funnel statistics. */
  stats: FunnelStats;
}

/**
 * Run the full F0-F8 candidate funnel.
 *
 * @param rawCandidates - All generated candidates (AIRSS + PyXtal + VCA + mutations)
 * @param formula - Target composition
 * @param elements - Element list
 * @param pressureGPa - Target pressure
 * @param nDFT - How many candidates to send to DFT (default 3)
 */
export function runCandidateFunnel(
  rawCandidates: CSPCandidate[],
  formula: string,
  elements: string[],
  pressureGPa: number,
  nDFT: number = 3,
): FunnelResult {
  const stats: FunnelStats = {
    f0_input: 0, f0_rejected: 0,
    f1_rejected: 0, f2_rejected: 0,
    f3_penalized: 0, f4_duplicates: 0,
    f5_scored: 0, f7_clusters: 0, f8_selected: 0,
    rejectionReasons: {},
  };

  // F0: Parse + normalize
  const f0 = f0_parse(rawCandidates, stats);

  // F1: Geometry hard filter
  const f1 = f1_geometry(f0, pressureGPa, stats);

  // F2: Chemistry sanity
  const f2 = f2_chemistry(f1, stats);

  // F3: Hydride sanity (scoring, not rejection)
  const f3 = f3_hydride(f2, stats);

  // F4: Raw duplicate removal
  const f4candidates = f3.map(s => s.candidate);
  const { unique: f4, duplicatesRemoved } = deduplicateCandidates(f4candidates);
  stats.f4_duplicates = duplicatesRemoved;

  // Map back to scored candidates
  const f4set = new Set(f4);
  const f4scored = f3.filter(s => f4set.has(s.candidate));

  // F5: Fast scoring
  const f5 = f5_score(f4scored);
  stats.f5_scored = f5.length;

  // F7: Clustering (skip F6/MLIP for now — placeholder)
  const clusters = clusterCandidates(
    f5.map(s => s.candidate),
    formula,
    pressureGPa,
  );
  stats.f7_clusters = clusters.length;

  // F8: DFT admission
  let dftSelection: DFTSelectionResult | null = null;
  let selected: CSPCandidate[] = [];

  if (clusters.length > 0) {
    dftSelection = selectForDFT(clusters, nDFT, f5.map(s => s.candidate));
    selected = dftSelection.selected;
    stats.f8_selected = selected.length;
  }

  // Record outcomes for adaptive learning
  for (const s of f3) {
    const survived = f4set.has(s.candidate);
    const proto = s.candidate.prototype ?? "unknown";
    recordGeneratorOutcome(elements, pressureGPa, proto, survived);

    // Try to extract volume multiplier from source string
    const volMatch = s.candidate.source?.match(/vol=([0-9.]+)/);
    if (volMatch) {
      recordVolumeOutcome(elements, pressureGPa, parseFloat(volMatch[1]), survived);
    }
  }

  // Log funnel summary
  console.log(`[CSP-Funnel] ${formula}: ${stats.f0_input} raw → ${f0.length} parsed → ${f1.length} geom → ${f2.length} chem → ${f4.length} deduped → ${clusters.length} clusters → ${selected.length} DFT`);
  if (Object.keys(stats.rejectionReasons).length > 0) {
    const reasons = Object.entries(stats.rejectionReasons)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    console.log(`[CSP-Funnel] Rejections: ${reasons}`);
  }

  // Log learning state
  logLearningState(elements, pressureGPa);

  return { selected, clusters, dftSelection, stats };
}
