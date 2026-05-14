/**
 * Two-Layer Dedup & Clustering System
 *
 * Layer 1: Cheap fingerprint filter — O(N) per candidate.
 *   Bins by composition + volume + sorted pair-distance histogram.
 *   Catches exact duplicates and near-duplicates fast.
 *
 * Layer 2: Structure matcher — O(N²) on surviving candidates only.
 *   Symmetry-aware clustering using cosine distance on extended
 *   fingerprint vectors. Groups candidates into structural families.
 *
 * This is MANDATORY for the scaled-up pipeline: 50+ AIRSS + 20+ PyXtal
 * per material generates many near-identical structures, especially at
 * high symmetry and repeated Z/volume settings.
 */

import type { CSPCandidate } from "./csp-types";
import { COVALENT_RADII } from "./csp-types";

// ---------------------------------------------------------------------------
// Cluster record
// ---------------------------------------------------------------------------

export interface StructureCluster {
  clusterId: string;
  /** The lowest-energy or highest-confidence member. */
  representative: CSPCandidate;
  /** All members in this cluster. */
  members: CSPCandidate[];
  size: number;
  /** Which engines contributed to this cluster. */
  sources: string[];
  /** Z values seen in this cluster. */
  zValuesSeen: number[];
  /** Volume points seen. */
  volumePointsSeen: number[];
  bestConfidence: number;
  bestPreRelaxScore: number | null;
}

// ---------------------------------------------------------------------------
// Layer 1: Cheap fingerprint
// ---------------------------------------------------------------------------

interface CheapFingerprint {
  /** Sorted element string: "H,H,H,La" */
  compositionKey: string;
  /** Volume per atom binned to 0.5 Å³ */
  volumeBin: number;
  /** Number of atoms */
  nAtoms: number;
  /** Sorted pair-distance histogram (binned to 0.1 Å, first 20 bins). */
  pairDistHist: number[];
  /** H-H distance histogram (binned to 0.1 Å, first 10 bins) for hydrides. */
  hhDistHist: number[];
  /** Coordination fingerprint: sorted coordination numbers. */
  coordFingerprint: number[];
}

function computeCheapFingerprint(c: CSPCandidate): CheapFingerprint {
  const pos = c.positions;
  const a = c.latticeA;
  const cOverA = c.cOverA ?? 1.0;
  const n = pos.length;

  // Composition key
  const compositionKey = pos.map(p => p.element).sort().join(",");

  // Volume per atom (binned to 0.5 Å³)
  const vol = c.cellVolume ?? (a * a * a * cOverA);
  const volumeBin = Math.round(vol / n / 0.5) * 0.5;

  // Pair-distance histogram (all pairs, binned to 0.1 Å, up to 2.0 Å)
  const HIST_BINS = 20;
  const BIN_SIZE = 0.1;
  const pairDistHist = new Array(HIST_BINS).fill(0);
  const hhDistHist = new Array(10).fill(0);
  const coordCounts = new Array(n).fill(0);
  const COORD_CUTOFF = a * 0.35;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dx = pos[i].x - pos[j].x;
      let dy = pos[i].y - pos[j].y;
      let dz = pos[i].z - pos[j].z;
      dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
      const dist = Math.sqrt((dx * a) ** 2 + (dy * a) ** 2 + (dz * a * cOverA) ** 2);

      const bin = Math.min(HIST_BINS - 1, Math.floor(dist / BIN_SIZE));
      if (bin >= 0) pairDistHist[bin]++;

      if (pos[i].element === "H" && pos[j].element === "H") {
        const hhBin = Math.min(9, Math.floor(dist / BIN_SIZE));
        if (hhBin >= 0) hhDistHist[hhBin]++;
      }

      if (dist < COORD_CUTOFF) {
        coordCounts[i]++;
        coordCounts[j]++;
      }
    }
  }

  // Normalize histograms
  const pairTotal = pairDistHist.reduce((s, v) => s + v, 0) || 1;
  const pairNorm = pairDistHist.map(v => v / pairTotal);
  const hhTotal = hhDistHist.reduce((s, v) => s + v, 0) || 1;
  const hhNorm = hhDistHist.map(v => v / hhTotal);

  const coordFingerprint = [...coordCounts].sort((a, b) => a - b);

  return {
    compositionKey,
    volumeBin,
    nAtoms: n,
    pairDistHist: pairNorm,
    hhDistHist: hhNorm,
    coordFingerprint,
  };
}

/**
 * Fast check: are two fingerprints "close enough" to be duplicates?
 */
function areFingerprintsSimilar(a: CheapFingerprint, b: CheapFingerprint, threshold: number = 0.08): boolean {
  // Must have same composition and atom count
  if (a.compositionKey !== b.compositionKey) return false;
  if (a.nAtoms !== b.nAtoms) return false;

  // Volume must be within 0.5 Å³/atom — TIGHTER volume check so structures
  // at different volume points from the ensemble are NOT considered duplicates.
  // This preserves volume diversity from the ensemble (0.70×, 0.85×, 1.00×, etc.)
  if (Math.abs(a.volumeBin - b.volumeBin) > 0.5) return false;

  // Pair-distance histogram cosine distance (loosened from 0.05 to 0.08)
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.pairDistHist.length; i++) {
    dot += a.pairDistHist[i] * b.pairDistHist[i];
    na += a.pairDistHist[i] ** 2;
    nb += b.pairDistHist[i] ** 2;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  const cosDist = denom > 0 ? 1 - dot / denom : 1;

  return cosDist < threshold;
}

// ---------------------------------------------------------------------------
// Layer 1: Dedup filter
// ---------------------------------------------------------------------------

/**
 * Remove duplicate candidates using cheap fingerprints.
 * Keeps the highest-confidence member of each duplicate set.
 * O(N²) in worst case but fast because fingerprint comparison is cheap.
 */
export function deduplicateCandidates(
  candidates: CSPCandidate[],
  threshold: number = 0.05,
): { unique: CSPCandidate[]; duplicatesRemoved: number } {
  if (candidates.length === 0) return { unique: [], duplicatesRemoved: 0 };

  const fingerprints = candidates.map(c => computeCheapFingerprint(c));
  const kept: boolean[] = new Array(candidates.length).fill(true);
  let duplicatesRemoved = 0;

  for (let i = 0; i < candidates.length; i++) {
    if (!kept[i]) continue;
    for (let j = i + 1; j < candidates.length; j++) {
      if (!kept[j]) continue;
      if (areFingerprintsSimilar(fingerprints[i], fingerprints[j], threshold)) {
        // Keep the one with higher confidence (or lower enthalpy if available)
        const ci = candidates[i];
        const cj = candidates[j];
        const scoreI = ci.enthalpyPerAtom ?? -(ci.confidence ?? 0);
        const scoreJ = cj.enthalpyPerAtom ?? -(cj.confidence ?? 0);
        if (scoreJ < scoreI) {
          kept[i] = false;
          duplicatesRemoved++;
          break; // i is removed, no need to check further
        } else {
          kept[j] = false;
          duplicatesRemoved++;
        }
      }
    }
  }

  const unique = candidates.filter((_, i) => kept[i]);
  return { unique, duplicatesRemoved };
}

// ---------------------------------------------------------------------------
// Layer 2: Symmetry-aware clustering
// ---------------------------------------------------------------------------

/**
 * Extended fingerprint vector for clustering.
 *
 * Physics-aware features:
 *  - PER-ELEMENT-PAIR sorted nearest neighbors. Mixing Cu-O, Cu-Cu, O-O into
 *    one sorted list (old behavior) made all cuprate-class structures look
 *    identical because the dominant distance scale is similar. Per-pair
 *    histograms preserve coordination chemistry (Cu-O octahedral vs square
 *    pyramidal vs planar look different here).
 *  - CELL SHAPE features (a, b/a, c/a) preserved in the vector. Previously
 *    the fingerprint was unit-normalized which threw away absolute scale and
 *    cell aspect — two structures with the same shape but vastly different
 *    volumes (or layered vs cubic) came out identical.
 *  - b/a factor applied to the y-axis (was missing — same bug F1 had).
 *
 * NOT normalized to a unit vector. cosineDistance still works because the
 * vectors are anchored on dimensionless ratios + per-pair distance bins.
 *
 * For the May-2026 Bi2Sr2Ca2Cu3O10 run, 29 distinct CHGNet-relaxed candidates
 * collapsed into 1 cluster because the old fingerprint couldn't distinguish
 * cuprate stackings — the per-pair version separates them.
 */
function extendedFingerprint(c: CSPCandidate): number[] {
  const pos = c.positions;
  const a = c.latticeA;
  const cOverA = c.cOverA ?? 1.0;
  const bOverA = c.latticeB && c.latticeA ? c.latticeB / c.latticeA : 1.0;
  const n = pos.length;

  // ---------- Per-element-pair sorted distance histograms ----------
  // Group pair distances by (sorted) element pair so Cu-O vs O-O vs Cu-Cu
  // are kept in separate slots. Take the 6 shortest distances per pair type —
  // that's enough to capture the coordination environment without bloating
  // the vector for high-symmetry crystals.
  const pairBuckets = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dx = pos[i].x - pos[j].x;
      let dy = pos[i].y - pos[j].y;
      let dz = pos[i].z - pos[j].z;
      dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
      // Full lattice scaling: cubic-only (`dy*a`) underestimated distances on
      // non-cubic cells and let distinct candidates look similar.
      const dist = Math.sqrt((dx * a) ** 2 + (dy * a * bOverA) ** 2 + (dz * a * cOverA) ** 2);
      const [elA, elB] = [pos[i].element, pos[j].element].sort();
      const key = `${elA}-${elB}`;
      const bucket = pairBuckets.get(key);
      if (bucket) bucket.push(dist);
      else pairBuckets.set(key, [dist]);
    }
  }

  // Sort pair-type keys alphabetically so two candidates with the same
  // composition produce features in the same slot order.
  const sortedPairKeys = Array.from(pairBuckets.keys()).sort();
  const PER_PAIR_FEATURES = 6;
  const pairFeatures: number[] = [];
  for (const key of sortedPairKeys) {
    const bucket = pairBuckets.get(key)!;
    bucket.sort((x, y) => x - y);
    for (let i = 0; i < PER_PAIR_FEATURES; i++) {
      pairFeatures.push(bucket[i] ?? 0);
    }
  }

  // ---------- Cell shape + volume features ----------
  // Volume per atom (Å³/atom) and the two aspect ratios. These preserve the
  // scale information that unit-normalization used to destroy. c/a > ~5 is
  // a strong signal for layered structures; b/a ≠ 1 means orthorhombic, etc.
  const volPerAtom = (a * a * a * cOverA * bOverA) / Math.max(1, n);
  const shapeFeatures: number[] = [
    a,
    bOverA,
    cOverA,
    volPerAtom,
  ];

  // ---------- Concatenate, do NOT normalize ----------
  // Absolute distances + ratios stay meaningful for cosineDistance because the
  // ratios are bounded ~[0.5, 10] and the absolute distances are O(1-15 Å).
  // No normalization step — that's what was collapsing cuprates.
  return [...shapeFeatures, ...pairFeatures];
}

function cosineDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) return 1;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] ** 2;
    nb += b[i] ** 2;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? 1 - dot / denom : 1;
}

/**
 * Feature-scaled Euclidean distance for clustering. Each feature dimension is
 * divided by its expected magnitude so cell-shape and pair-distance components
 * contribute on equal footing. Used INSTEAD of cosineDistance for clustering
 * because cosine is scale-invariant (treats proportional vectors as identical),
 * which made cuprates of different c/a ratios cluster together — Bi2Sr2Ca2Cu3O10
 * collapsed 29 distinct candidates into 1 cluster in the May-2026 run.
 *
 * The first 4 features are shape (a, b/a, c/a, V/atom); the remainder are
 * per-element-pair sorted distance bins. Scale factors picked so a 5% relative
 * change in any feature contributes ~0.05 to the distance, putting the natural
 * clusterThreshold near 0.1–0.2.
 */
function featureEuclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  // [0]: lattice a (Å, typical scale 5)
  // [1]: b/a (~1, ratio)
  // [2]: c/a (~1–15, ratio)
  // [3]: V/atom (Å³/atom, typical scale 20)
  // [4+]: per-pair distances (Å, typical scale 3)
  const SCALES = [5.0, 1.0, 1.0, 20.0, /* rest default 3.0 */];
  const DEFAULT_DIST_SCALE = 3.0;
  let sumSq = 0;
  for (let i = 0; i < a.length; i++) {
    const scale = i < SCALES.length ? SCALES[i] : DEFAULT_DIST_SCALE;
    const d = (a[i] - b[i]) / scale;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / a.length); // RMS so distance is independent of vector length
}

/**
 * Cluster candidates into structural families using agglomerative
 * clustering on extended fingerprint vectors.
 */
export function clusterCandidates(
  candidates: CSPCandidate[],
  formula: string,
  pressureGPa: number,
  clusterThreshold: number = 0.15,
): StructureCluster[] {
  if (candidates.length === 0) return [];

  // Compute extended fingerprints (per-pair distances + cell shape).
  const fps = candidates.map(c => extendedFingerprint(c));

  // Simple single-linkage agglomerative clustering using feature-scaled
  // Euclidean distance. Threshold default 0.15 corresponds roughly to
  // 15% per-feature deviation on average — tight enough to separate
  // cuprate stackings (which differ by ~10% c/a + a few % pair distances)
  // while still merging genuine duplicates that share lattice+positions.
  const clusterIds = new Array(candidates.length).fill(-1);
  let nextClusterId = 0;

  for (let i = 0; i < candidates.length; i++) {
    if (clusterIds[i] >= 0) continue;
    clusterIds[i] = nextClusterId;

    // Find all candidates similar to i. Skip mixed-length fingerprints
    // (different sets of element pairs → different composition somehow) —
    // those are never the same cluster.
    for (let j = i + 1; j < candidates.length; j++) {
      if (clusterIds[j] >= 0) continue;
      if (fps[i].length !== fps[j].length) continue;
      if (featureEuclideanDistance(fps[i], fps[j]) < clusterThreshold) {
        clusterIds[j] = nextClusterId;
      }
    }
    nextClusterId++;
  }

  // Build cluster records
  const clusterMap = new Map<number, CSPCandidate[]>();
  for (let i = 0; i < candidates.length; i++) {
    const id = clusterIds[i];
    if (!clusterMap.has(id)) clusterMap.set(id, []);
    clusterMap.get(id)!.push(candidates[i]);
  }

  const clusters: StructureCluster[] = [];
  for (const [id, members] of clusterMap.entries()) {
    // Sort by enthalpy (if available) then confidence
    members.sort((a, b) => {
      if (a.enthalpyPerAtom != null && b.enthalpyPerAtom != null) {
        return a.enthalpyPerAtom - b.enthalpyPerAtom;
      }
      return (b.confidence ?? 0) - (a.confidence ?? 0);
    });

    const representative = members[0];
    const sources = [...new Set(members.map(m => m.prototype || m.sourceEngine || "unknown"))];

    // Extract Z values from source strings
    const zValues: number[] = [];
    for (const m of members) {
      const zMatch = m.source?.match(/Z=(\d+)/);
      if (zMatch) {
        const z = parseInt(zMatch[1]);
        if (!zValues.includes(z)) zValues.push(z);
      }
    }

    // Extract volume points from cell volumes
    const volPoints = [...new Set(members
      .filter(m => m.cellVolume && m.positions.length > 0)
      .map(m => Math.round((m.cellVolume! / m.positions.length) * 10) / 10)
    )].sort();

    clusters.push({
      clusterId: `${formula}_${pressureGPa}GPa_cluster_${String(id).padStart(3, "0")}`,
      representative,
      members,
      size: members.length,
      sources,
      zValuesSeen: zValues,
      volumePointsSeen: volPoints,
      bestConfidence: Math.max(...members.map(m => m.confidence ?? 0)),
      bestPreRelaxScore: members[0].enthalpyPerAtom ?? null,
    });
  }

  // Sort clusters by size (largest first — most confident basins)
  clusters.sort((a, b) => b.size - a.size);

  // Log summary
  console.log(`[CSP-Cluster] ${formula}: ${candidates.length} candidates → ${clusters.length} clusters (largest: ${clusters[0]?.size ?? 0}, sources: ${clusters.slice(0, 3).map(c => `[${c.sources.join("+")}]×${c.size}`).join(", ")})`);

  // Tag candidates with cluster IDs
  for (const cluster of clusters) {
    for (const member of cluster.members) {
      member.motifId = cluster.clusterId;
    }
  }

  return clusters;
}

// ---------------------------------------------------------------------------
// Full dedup + cluster pipeline
// ---------------------------------------------------------------------------

/**
 * Run the complete dedup + clustering pipeline on a batch of candidates.
 * Returns deduplicated cluster representatives ready for DFT admission scoring.
 */
export function dedupAndCluster(
  candidates: CSPCandidate[],
  formula: string,
  pressureGPa: number,
): {
  clusters: StructureCluster[];
  unique: CSPCandidate[];
  stats: { total: number; afterDedup: number; clusterCount: number };
} {
  const total = candidates.length;

  // Layer 1: cheap dedup
  const { unique, duplicatesRemoved } = deduplicateCandidates(candidates);
  console.log(`[CSP-Dedup] ${formula}: ${total} → ${unique.length} after fingerprint dedup (${duplicatesRemoved} removed)`);

  // Layer 2: cluster
  const clusters = clusterCandidates(unique, formula, pressureGPa);

  return {
    clusters,
    unique,
    stats: { total, afterDedup: unique.length, clusterCount: clusters.length },
  };
}
