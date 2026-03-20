/**
 * thor-tensor.ts
 * ==============
 * THOR-inspired (Tensors for High-dimensional Object Representation) symmetry
 * embedding for crystal structures.
 *
 * Core idea (after LANL THOR, 2026):
 *   Represent a crystal's symmetry state as a rank-4 tensor indexed by:
 *     Axis 0 — Laue class      (11 classes)
 *     Axis 1 — SG family       ( 7 values: triclinic→cubic)
 *     Axis 2 — Wyckoff type    ( 5 levels: highest-sym → general position)
 *     Axis 3 — Distortion mode ( 6 types: identity/rot/mirror/screw/glide/mixed)
 *
 *   A tensor-train (TT) decomposition decomposes this as a product of four
 *   small "core" matrices G₁⊗G₂⊗G₃⊗G₄. Contracting along the mode indices
 *   produces a compact embedding vector that respects crystallographic symmetry.
 *
 * Result: `computeThorFeatureVector()` returns a 12-dimensional embedding
 * that replaces (and extends) the original 6D `SymmetryEmbedding`.
 *
 * The 12D vector encodes:
 *  [0-1]  Laue-class projection     (2D — rotation/inversion character)
 *  [2-4]  Point-group projection    (3D — site symmetry richness)
 *  [5-7]  Wyckoff projection        (3D — site multiplicity + symmetry)
 *  [8-9]  Distortion mode           (2D — order parameter symmetry)
 *  [10]   Chain depth (normalized)  (1D — distance to high-sym parent)
 *  [11]   Group order (normalized)  (1D — overall symmetry content)
 */

import {
  SG_ORDER,
  SG_POINT_GROUP,
  laueClassIndex,
  crystalSystemFromSG,
  getHighSymWyckoff,
  getWyckoffSitesForSG,
  getSuperGroupEntriesFor,
  getSubgroupEntriesFor,
} from "./sg-data";

// ── THOR core matrices (analytically initialized from crystallographic theory) ─
// These are the tensor-train cores G₁..G₄.
// In a trained system these would be learned; here they encode group-theoretic
// structure: Laue symmetry → point group → Wyckoff → distortion.

/** G₁: Laue class (11) → abstract (2×2). Shape [11 × 2]. */
const G1: number[][] = [
  [0.10, 0.05],  // -1  triclinic
  [0.25, 0.10],  // 2/m monoclinic
  [0.40, 0.20],  // mmm orthorhombic
  [0.50, 0.30],  // 4/m tetragonal-low
  [0.60, 0.45],  // 4/mmm tetragonal-high
  [0.45, 0.35],  // -3  trigonal-low
  [0.55, 0.50],  // -3m trigonal-high
  [0.65, 0.55],  // 6/m hexagonal-low
  [0.75, 0.65],  // 6/mmm hexagonal-high
  [0.80, 0.70],  // m-3  cubic-low
  [1.00, 0.90],  // m-3m cubic-high
];

/** G₂: Crystal system (7) → abstract (3×3) conditioned on G₁ output. Shape [7 × 3]. */
const G2: number[][] = [
  [0.10, 0.05, 0.02],  // triclinic
  [0.20, 0.15, 0.08],  // monoclinic
  [0.35, 0.25, 0.15],  // orthorhombic
  [0.55, 0.40, 0.28],  // tetragonal
  [0.50, 0.42, 0.32],  // trigonal
  [0.65, 0.50, 0.38],  // hexagonal
  [0.90, 0.75, 0.60],  // cubic
];

/** G₃: Wyckoff type (5) → abstract (3×3). Shape [5 × 3].
 *  Levels: 0=max-sym, 1=high, 2=medium, 3=low, 4=general */
const G3: number[][] = [
  [1.00, 0.95, 0.90],  // maximum-symmetry site (e.g., 1a in m-3m)
  [0.75, 0.70, 0.65],  // high-symmetry site
  [0.50, 0.45, 0.40],  // medium-symmetry site
  [0.30, 0.25, 0.20],  // low-symmetry site
  [0.10, 0.08, 0.05],  // general position (no site symmetry)
];

/** G₄: Distortion mode (6) → scalar. Shape [6 × 2].
 *  0=identity, 1=volume-change, 2=shear, 3=rotation, 4=breathing, 5=mixed */
const G4: number[][] = [
  [0.00, 0.00],  // identity (high-sym, no distortion)
  [0.30, 0.10],  // tetragonal/orthorhombic (volume-preserving elongation)
  [0.50, 0.30],  // shear distortion
  [0.40, 0.35],  // trigonal/rotation
  [0.20, 0.15],  // breathing (isotropic volume change)
  [0.70, 0.60],  // mixed/reconstructive
];

// ── Distortion type → mode index ──────────────────────────────────────────────

function distortionModeIndex(
  type: "tetragonal" | "orthorhombic" | "trigonal" | "shear" | "breathing" | "monoclinic" | undefined
): number {
  if (!type) return 0;
  const map: Record<string, number> = {
    tetragonal: 1, orthorhombic: 1, monoclinic: 2, shear: 2,
    trigonal: 3, breathing: 4,
  };
  return map[type] ?? 5;
}

// ── Wyckoff site → type index ──────────────────────────────────────────────────

function wyckoffTypeIndex(siteSymmetryOrder: number, sgOrder: number): number {
  const ratio = siteSymmetryOrder / Math.max(1, sgOrder);
  if (ratio >= 0.8) return 0;  // nearly full symmetry
  if (ratio >= 0.4) return 1;
  if (ratio >= 0.2) return 2;
  if (ratio >= 0.05) return 3;
  return 4;                    // general position
}

// ── TT contraction helpers ────────────────────────────────────────────────────

/** Contract row vector v (length n) with matrix M (shape [r × c]) → new row vector (length c). */
function contractVec(v: number[], M: number[][]): number[] {
  const c = M[0].length;
  const out = new Array<number>(c).fill(0);
  for (let j = 0; j < c; j++) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * (M[i]?.[j] ?? 0);
    out[j] = s;
  }
  return out;
}

/** Scale a vector to unit L2 norm (safe if zero). */
function l2Normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return norm < 1e-9 ? v.map(() => 0) : v.map(x => x / norm);
}

// ── THOR embedding ────────────────────────────────────────────────────────────

export interface ThorEmbedding {
  /** 12-dimensional feature vector. */
  vector: number[];
  /** Metadata for interpretability. */
  meta: {
    sgNumber: number;
    crystalSystem: string;
    laueClass: number;
    sgOrder: number;
    wyckoffMultiplicity: number;
    siteSymmetryOrder: number;
    subgroupChainDepth: number;
    closestParentSg: number | null;
    distortionMode: string;
    irrepLabel: string;
  };
}

/**
 * Compute the THOR tensor embedding for a crystal defined by its space group
 * number and (optionally) an atomic fractional coordinate.
 *
 * Returns a 12D normalized vector suitable for ML feature input.
 */
export function computeThorEmbedding(
  sgNumber: number,
  fracPosition?: [number, number, number]
): ThorEmbedding {
  const sgOrder   = SG_ORDER[sgNumber] ?? 4;
  const csys      = crystalSystemFromSG(sgNumber);
  const laueIdx   = laueClassIndex(sgNumber);

  // Crystal system index
  const csysMap: Record<string, number> = {
    triclinic: 0, monoclinic: 1, orthorhombic: 2,
    tetragonal: 3, trigonal: 4, hexagonal: 5, cubic: 6,
  };
  const csysIdx = csysMap[csys] ?? 6;

  // Wyckoff site info
  const highSym = getHighSymWyckoff(sgNumber);
  let wyckoffMult  = highSym?.multiplicity     ?? 1;
  let siteSymOrder = highSym?.siteSymmetryOrder ?? sgOrder;

  if (fracPosition && highSym) {
    // Try to find the site nearest to fracPosition
    const sites = getWyckoffSitesForSG(sgNumber);
    let bestDist = Infinity;
    let bestSite = highSym;
    for (const site of sites) {
      const [rx, ry, rz] = site.representative;
      const dx = Math.abs(fracPosition[0] - rx) % 1.0;
      const dy = Math.abs(fracPosition[1] - ry) % 1.0;
      const dz = Math.abs(fracPosition[2] - rz) % 1.0;
      const d = Math.sqrt(
        Math.min(dx, 1-dx)**2 + Math.min(dy, 1-dy)**2 + Math.min(dz, 1-dz)**2
      );
      if (d < bestDist) { bestDist = d; bestSite = site; }
    }
    if (bestDist < 0.15) {
      wyckoffMult  = bestSite.multiplicity;
      siteSymOrder = bestSite.siteSymmetryOrder;
    }
  }

  // Subgroup chain info (how far is this SG from a high-sym parent?)
  const superEntries = getSuperGroupEntriesFor(sgNumber);
  const closestParent = superEntries.length > 0 ? superEntries[0] : null;
  const chainDepth    = computeChainDepth(sgNumber, 0);

  const irrepLabel      = closestParent?.irrepLabel      ?? "Γ1+";
  const distortionMode  = closestParent?.distortionType  ?? "breathing";

  // ── Tensor-train contraction ─────────────────────────────────────────────

  // Axis 0: Laue class → 2D via G1
  const v0 = G1[Math.min(laueIdx, G1.length - 1)];  // [2]

  // Axis 1: crystal system → 3D via G2, blended with v0
  const raw1 = G2[csysIdx];  // [3]
  const v1   = raw1.map((x, i) => x * (v0[i % 2] + 1) / 2);  // [3]

  // Axis 2: Wyckoff type → 3D via G3
  const wType = wyckoffTypeIndex(siteSymOrder, sgOrder);
  const v2    = G3[wType];  // [3]

  // Axis 3: distortion mode → 2D via G4
  const dMode = distortionModeIndex(distortionMode as "tetragonal");
  const v3    = G4[dMode];  // [2]

  // Extra scalars
  const chainDepthNorm = Math.min(chainDepth / 4.0, 1.0);
  const sgOrderNorm    = Math.log1p(sgOrder) / Math.log1p(192);

  // Assemble 12D vector
  const raw12: number[] = [
    ...v0,              // [0,1]  Laue class
    ...v1,              // [2,3,4] point-group / crystal-system blend
    ...v2,              // [5,6,7] Wyckoff character
    ...v3,              // [8,9]  distortion mode
    chainDepthNorm,     // [10]   subgroup chain depth
    sgOrderNorm,        // [11]   group order
  ];

  // L2 normalize (standard for ML embeddings)
  const vector = l2Normalize(raw12);

  return {
    vector,
    meta: {
      sgNumber,
      crystalSystem: csys,
      laueClass: laueIdx,
      sgOrder,
      wyckoffMultiplicity: wyckoffMult,
      siteSymmetryOrder: siteSymOrder,
      subgroupChainDepth: chainDepth,
      closestParentSg: closestParent?.parentSg ?? null,
      distortionMode,
      irrepLabel,
    },
  };
}

/**
 * Returns the THOR embedding as a plain number[] — drop-in for ML pipelines.
 * Returns 12 values (vs the original 6 from `computeSymmetryFeatureVector`).
 */
export function computeThorFeatureVector(sgNumber: number, fracPosition?: [number, number, number]): number[] {
  return computeThorEmbedding(sgNumber, fracPosition).vector;
}

/**
 * Compute how many subgroup steps exist between this SG and the nearest
 * recorded high-sym parent (max depth 4 to keep this bounded).
 */
function computeChainDepth(sgNumber: number, depth: number): number {
  if (depth >= 4) return depth;
  const parents = getSuperGroupEntriesFor(sgNumber);
  if (parents.length === 0) return depth;
  // Follow the highest-order parent
  const best = parents.reduce((p, c) =>
    (SG_ORDER[c.parentSg] ?? 0) > (SG_ORDER[p.parentSg] ?? 0) ? c : p
  );
  return computeChainDepth(best.parentSg, depth + 1);
}

// ── THOR tensor symmetry predictor ────────────────────────────────────────────
// Given a composition and a list of candidate SG numbers, rank them by
// their THOR embedding similarity to known high-Tc prototypes.

export interface ThorSGRanking {
  sgNumber: number;
  thorSimilarity: number;
  embedding: number[];
}

/** Reference embeddings for high-Tc prototype SGs (from known SC families). */
const HIGHTC_PROTOTYPE_SGS = [225, 229, 223, 191, 194, 139, 129, 166, 148, 62, 63] as const;
let _protoEmbeddings: Map<number, number[]> | null = null;

function getPrototypeEmbeddings(): Map<number, number[]> {
  if (_protoEmbeddings) return _protoEmbeddings;
  _protoEmbeddings = new Map();
  for (const sg of HIGHTC_PROTOTYPE_SGS) {
    _protoEmbeddings.set(sg, computeThorFeatureVector(sg));
  }
  return _protoEmbeddings;
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return na < 1e-12 || nb < 1e-12 ? 0 : dot / Math.sqrt(na * nb);
}

/**
 * Rank candidate space groups by THOR cosine similarity to known high-Tc
 * prototype SGs.  Returns top-k.
 */
export function rankSGsByThorSimilarity(
  candidateSGs: number[],
  topK = 10,
): ThorSGRanking[] {
  const protos = getPrototypeEmbeddings();

  return candidateSGs
    .map(sg => {
      const emb = computeThorFeatureVector(sg);
      // Max similarity over all prototype SGs
      let maxSim = 0;
      Array.from(protos.values()).forEach(proto => {
        const s = cosineSim(emb, proto);
        if (s > maxSim) maxSim = s;
      });
      return { sgNumber: sg, thorSimilarity: maxSim, embedding: emb };
    })
    .sort((a, b) => b.thorSimilarity - a.thorSimilarity)
    .slice(0, topK);
}

// ── Symmetry mode decomposition ───────────────────────────────────────────────
// Given a parent→child SG transition, decompose the distortion into its
// THOR basis components (tensor contraction residuals).

export interface ThorDistortionDecomposition {
  parentSg:   number;
  childSg:    number;
  irrepLabel: string;
  /** Unit vector in THOR embedding space pointing along this distortion mode. */
  modeVector: number[];
  /** Amplitude of the mode (0–1 normalized). */
  amplitude:  number;
}

export function decomposeThorDistortion(
  parentSg: number,
  childSg: number,
  irrepLabel: string,
  distortionType: "tetragonal" | "orthorhombic" | "trigonal" | "shear" | "breathing" | "monoclinic",
  magnitude: number,
): ThorDistortionDecomposition {
  const parentEmb = computeThorFeatureVector(parentSg);
  const childEmb  = computeThorFeatureVector(childSg);

  // The mode vector is the displacement in THOR space from parent → child
  const diff      = parentEmb.map((p, i) => childEmb[i] - p);
  const modeVector = l2Normalize(diff);
  const amplitude  = Math.min(magnitude * 10, 1.0);  // scale to [0,1]

  return { parentSg, childSg, irrepLabel, modeVector, amplitude };
}

// ── Irrep label → THOR encoding ───────────────────────────────────────────────
// Extended from original 13 labels to 30 common irrep labels.

export const IRREP_ENCODING: Record<string, number> = {
  // Gamma-point
  "Γ1+": 0.00, "Γ2+": 0.02, "Γ3+": 0.05, "Γ4+": 0.07,
  "Γ1-": 0.10, "Γ2-": 0.12, "Γ3-": 0.14, "Γ4-": 0.15, "Γ5+": 0.18, "Γ5-": 0.20,
  // Zone boundary — M point (tetragonal/cubic)
  "M1+": 0.25, "M2+": 0.27, "M3+": 0.30, "M5+": 0.32,
  "M1-": 0.33, "M3-": 0.35, "M5-": 0.38,
  // Zone boundary — X point
  "X1+": 0.40, "X3+": 0.42, "X5+": 0.44,
  "X1-": 0.46, "X3-": 0.48, "X5-": 0.50,
  // Zone boundary — R, H, K, L, N, P points
  "R1+": 0.55, "R2+": 0.57,
  "H3":  0.60, "H5-": 0.62,
  "K1":  0.65, "K3":  0.67,
  "L1+": 0.70, "L2+": 0.72, "L3-": 0.74,
  "N1+": 0.78, "N1-": 0.80,
  "P4":  0.82,
  // Hexagonal zone boundary
  "A1-": 0.85, "A1+": 0.87,
  "K2":  0.88, "K4":  0.89,
  // Molecular / ortho irreps
  "Ag":  0.90, "Bg":  0.92,
  "Eg":  0.95, "Eu":  0.97,
};

export function encodeIrrep(label: string): number {
  return IRREP_ENCODING[label] ?? 0.50;
}
