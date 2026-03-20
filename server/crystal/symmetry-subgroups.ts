/**
 * symmetry-subgroups.ts
 * =====================
 * Crystal symmetry embedding and broken-symmetry variant generation.
 *
 * Refactored to use:
 *  - sg-data.ts    — complete 230 SG database (order, Wyckoff, subgroup hierarchy)
 *  - thor-tensor.ts — THOR tensor embedding (12D, replaces hand-coded 6D)
 *
 * All original public API is preserved for backward compatibility with:
 *  - server/ai/crystal-generator.ts
 *  - server/learning/graph-neural-net.ts
 */

import {
  SG_ORDER,
  SG_HM_SYMBOL,
  EXTENDED_WYCKOFF_DB,
  EXTENDED_SUBGROUP_HIERARCHY,
  getHighSymWyckoff,
  getWyckoffSitesForSG,
  getSubgroupEntriesFor,
  getSuperGroupEntriesFor,
  crystalSystemFromSG,
  type WyckoffEntry,
  type SubgroupEntry,
} from "./sg-data";

import {
  computeThorEmbedding,
  computeThorFeatureVector,
  encodeIrrep,
  IRREP_ENCODING,
  decomposeThorDistortion,
  type ThorEmbedding,
} from "./thor-tensor";

// ── Re-export THOR utilities for downstream use ───────────────────────────────
export { computeThorFeatureVector, decomposeThorDistortion, IRREP_ENCODING };
export type { ThorEmbedding };

// ── Legacy type aliases (keep shapes identical for consumers) ─────────────────

export interface DistortionVector {
  axis: "x" | "y" | "z" | "xy" | "xz" | "yz" | "xyz";
  magnitude: number;
  type: "tetragonal" | "orthorhombic" | "monoclinic" | "trigonal" | "shear" | "breathing";
}

export interface SubgroupRelation {
  parent: string;
  child: string;
  parentNumber: number;
  childNumber: number;
  index: number;
  distortionVectors: DistortionVector[];
  transitionType: "displacive" | "order-disorder" | "reconstructive";
  irrepLabel: string;
}

export interface WyckoffSiteInfo {
  spaceGroup: string;
  letter: string;
  multiplicity: number;
  siteSymmetryOrder: number;
  siteSymmetryLabel: string;
  positions: [number, number, number][];
}

/** Original 6D embedding (kept for graph-neural-net.ts). */
export interface SymmetryEmbedding {
  wyckoffMultiplicity: number;
  siteSymmetryOrder: number;
  subgroupIndex: number;
  irrepEncoding: number;
  distortionMagnitude: number;
  parentGroupOrder: number;
}

export interface BrokenSymmetryVariant {
  parentSpaceGroup: string;
  childSpaceGroup: string;
  latticeDistortion: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  atomicDisplacements: { dx: number; dy: number; dz: number }[];
  distortionAmplitude: number;
  relation: SubgroupRelation;
}

// ── Build SubgroupRelation from SubgroupEntry ─────────────────────────────────

function entryToRelation(e: SubgroupEntry): SubgroupRelation {
  const parentSym = SG_HM_SYMBOL[e.parentSg] ?? `SG-${e.parentSg}`;
  const childSym  = SG_HM_SYMBOL[e.childSg]  ?? `SG-${e.childSg}`;

  const dv: DistortionVector = {
    axis: e.distortionAxis,
    magnitude: e.distortionMagnitude,
    type: e.distortionType === "monoclinic" ? "shear" : e.distortionType,
  };

  // Add secondary distortion vector for orthorhombic/mixed cases
  const dvs: DistortionVector[] = [dv];
  if (e.distortionType === "orthorhombic" && e.distortionAxis === "x") {
    dvs.push({ axis: "y", magnitude: e.distortionMagnitude * 0.7, type: "orthorhombic" });
  }
  if (e.distortionType === "shear" || e.distortionType === "monoclinic") {
    dvs.push({ axis: "z", magnitude: e.distortionMagnitude * 0.5, type: "tetragonal" });
  }

  return {
    parent: parentSym,
    child: childSym,
    parentNumber: e.parentSg,
    childNumber: e.childSg,
    index: e.index,
    distortionVectors: dvs,
    transitionType: e.transitionType,
    irrepLabel: e.irrepLabel,
  };
}

// ── Convert WyckoffEntry to legacy WyckoffSiteInfo ────────────────────────────

function entryToSiteInfo(w: WyckoffEntry): WyckoffSiteInfo {
  return {
    spaceGroup: w.sgSymbol,
    letter: w.letter,
    multiplicity: w.multiplicity,
    siteSymmetryOrder: w.siteSymmetryOrder,
    siteSymmetryLabel: w.siteSymmetryLabel,
    positions: [w.representative],
  };
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Returns all maximal subgroup relations for the given space group name or number.
 */
export function getSubgroups(spaceGroupNameOrNumber: string | number): SubgroupRelation[] {
  const sgNum = resolveSGNumber(spaceGroupNameOrNumber);
  if (sgNum === null) return [];
  return getSubgroupEntriesFor(sgNum).map(entryToRelation);
}

/**
 * Returns all supergroup relations for the given space group.
 */
export function getSupergroups(spaceGroupNameOrNumber: string | number): SubgroupRelation[] {
  const sgNum = resolveSGNumber(spaceGroupNameOrNumber);
  if (sgNum === null) return [];
  return getSuperGroupEntriesFor(sgNum).map(entryToRelation);
}

/**
 * Returns the shortest subgroup chain from `from` to `to` (by DFS over the
 * extended hierarchy).
 */
export function getSubgroupChain(from: string | number, to: string | number): SubgroupRelation[] {
  const fromNum = resolveSGNumber(from);
  const toNum   = resolveSGNumber(to);
  if (fromNum === null || toNum === null) return [];

  const chain: SubgroupRelation[] = [];
  const visited = new Set<number>();

  function dfs(current: number): boolean {
    if (current === toNum) return true;
    if (visited.has(current)) return false;
    visited.add(current);
    for (const e of getSubgroupEntriesFor(current)) {
      if (dfs(e.childSg)) {
        chain.unshift(entryToRelation(e));
        return true;
      }
    }
    return false;
  }

  dfs(fromNum);
  return chain;
}

/**
 * Returns Wyckoff sites for the given space group (name or number).
 * Falls back to a synthetic entry when the SG is not in the database.
 */
export function getWyckoffSites(spaceGroupNameOrNumber: string | number): WyckoffSiteInfo[] {
  const sgNum = resolveSGNumber(spaceGroupNameOrNumber);
  if (sgNum === null) return [];
  const entries = getWyckoffSitesForSG(sgNum);
  if (entries.length > 0) return entries.map(entryToSiteInfo);

  // Synthetic fallback: derive from SG order
  const sgOrder  = SG_ORDER[sgNum] ?? 4;
  const sgSymbol = SG_HM_SYMBOL[sgNum] ?? `SG-${sgNum}`;
  return [{
    spaceGroup: sgSymbol,
    letter: "a",
    multiplicity: Math.max(1, Math.round(sgOrder / 8)),
    siteSymmetryOrder: sgOrder,
    siteSymmetryLabel: crystalSystemFromSG(sgNum),
    positions: [[0, 0, 0]],
  }];
}

/**
 * Returns the Wyckoff site nearest to `fracPosition` for the given SG.
 */
export function getWyckoffForPosition(
  spaceGroupNameOrNumber: string | number,
  fracPosition: [number, number, number],
  tolerance = 0.1
): WyckoffSiteInfo | null {
  const sgNum = resolveSGNumber(spaceGroupNameOrNumber);
  if (sgNum === null) return null;

  for (const entry of getWyckoffSitesForSG(sgNum)) {
    const [rx, ry, rz] = entry.representative;
    const dx = Math.abs(fracPosition[0] - rx) % 1.0;
    const dy = Math.abs(fracPosition[1] - ry) % 1.0;
    const dz = Math.abs(fracPosition[2] - rz) % 1.0;
    const d = Math.sqrt(
      Math.min(dx, 1-dx)**2 + Math.min(dy, 1-dy)**2 + Math.min(dz, 1-dz)**2
    );
    if (d < tolerance) return entryToSiteInfo(entry);
  }
  return null;
}

/**
 * Compute the legacy 6-dimensional SymmetryEmbedding for a space group.
 * Internally backed by the full 230 SG database (no more 8-SG blind spot).
 */
export function computeSymmetryEmbedding(
  spaceGroupNameOrNumber: string | number,
  fracPosition?: [number, number, number]
): SymmetryEmbedding {
  const sgNum    = resolveSGNumber(spaceGroupNameOrNumber) ?? 1;
  const sgOrder  = SG_ORDER[sgNum] ?? 4;

  // Wyckoff info
  const highSym  = getHighSymWyckoff(sgNum);
  let wyckoffMult  = highSym?.multiplicity     ?? 1;
  let siteSymOrder = highSym?.siteSymmetryOrder ?? sgOrder;

  if (fracPosition) {
    const match = getWyckoffForPosition(sgNum, fracPosition);
    if (match) {
      wyckoffMult  = match.multiplicity;
      siteSymOrder = match.siteSymmetryOrder;
    }
  }

  // Subgroup info
  const superEntries = getSuperGroupEntriesFor(sgNum);
  let subgroupIndex       = 1;
  let irrepEncoding       = 0;
  let distortionMagnitude = 0;

  if (superEntries.length > 0) {
    const rel = superEntries[0];
    subgroupIndex       = rel.index;
    irrepEncoding       = encodeIrrep(rel.irrepLabel);
    distortionMagnitude = rel.distortionMagnitude;
  }

  return {
    wyckoffMultiplicity: wyckoffMult,
    siteSymmetryOrder: siteSymOrder,
    subgroupIndex,
    irrepEncoding,
    distortionMagnitude,
    parentGroupOrder: sgOrder,
  };
}

/**
 * Returns the original 6D feature vector — unchanged signature for
 * graph-neural-net.ts compatibility.
 *
 * Normalization constants use the full database range now:
 *  - wyckoffMultiplicity / 192  (max general position in Fm-3m)
 *  - siteSymmetryOrder  / 48    (max Oh site symmetry)
 *  - subgroupIndex      / 12    (extended range)
 *  - irrepEncoding              (already in [0,1])
 *  - distortionMagnitude/ 0.12  (max in database)
 *  - parentGroupOrder   / 192
 */
export function computeSymmetryFeatureVector(embedding: SymmetryEmbedding): number[] {
  return [
    Math.min(embedding.wyckoffMultiplicity / 192.0, 1.0),
    Math.min(embedding.siteSymmetryOrder   / 48.0,  1.0),
    Math.min(embedding.subgroupIndex       / 12.0,  1.0),
    embedding.irrepEncoding,
    Math.min(embedding.distortionMagnitude / 0.12,  1.0),
    Math.min(embedding.parentGroupOrder    / 192.0, 1.0),
  ];
}

// ── Broken-symmetry variant generation ───────────────────────────────────────

export function generateBrokenSymmetryVariants(
  spaceGroupNameOrNumber: string | number,
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  atomCount: number,
  amplitudeScale = 1.0
): BrokenSymmetryVariant[] {
  const sgNum     = resolveSGNumber(spaceGroupNameOrNumber) ?? 1;
  const subgroups = getSubgroups(sgNum);
  const variants: BrokenSymmetryVariant[] = [];

  for (const rel of subgroups) {
    const newLattice = { ...lattice };
    const displacements: { dx: number; dy: number; dz: number }[] = [];

    for (const dv of rel.distortionVectors) {
      const amp = dv.magnitude * amplitudeScale;
      switch (dv.type) {
        case "tetragonal":
          newLattice.c *= (1 + amp);
          newLattice.a *= (1 - amp * 0.3);
          newLattice.b *= (1 - amp * 0.3);
          break;
        case "orthorhombic":
          if (dv.axis === "x") newLattice.a *= (1 + amp);
          if (dv.axis === "y") newLattice.b *= (1 + amp * 0.7);
          if (dv.axis === "z") newLattice.c *= (1 - amp * 0.5);
          break;
        case "trigonal":
          newLattice.alpha = 90 - amp * 180 / Math.PI * 2;
          newLattice.beta  = 90 - amp * 180 / Math.PI * 2;
          newLattice.gamma = 90 - amp * 180 / Math.PI * 2;
          break;
        case "shear":
          if (dv.axis.includes("x") && dv.axis.includes("y")) newLattice.gamma = 90 + amp * 180 / Math.PI;
          if (dv.axis.includes("y") && dv.axis.includes("z")) newLattice.alpha = 90 + amp * 180 / Math.PI * 0.5;
          break;
        case "breathing":
          newLattice.a *= (1 + amp * 0.5);
          newLattice.b *= (1 + amp * 0.5);
          newLattice.c *= (1 + amp * 0.5);
          break;
        case "monoclinic":
          newLattice.beta = 90 + amp * 180 / Math.PI;
          break;
      }
    }

    for (let i = 0; i < atomCount; i++) {
      let dx = 0, dy = 0, dz = 0;
      for (const dv of rel.distortionVectors) {
        const amp   = dv.magnitude * amplitudeScale;
        const phase = (i / Math.max(1, atomCount - 1)) * Math.PI;
        if (dv.axis.includes("x")) dx += amp * Math.sin(phase) * 0.1;
        if (dv.axis.includes("y")) dy += amp * Math.cos(phase) * 0.1;
        if (dv.axis.includes("z")) dz += amp * Math.sin(phase + Math.PI / 4) * 0.1;
      }
      displacements.push({ dx, dy, dz });
    }

    const totalDistortion = rel.distortionVectors.reduce((s, d) => s + d.magnitude, 0) * amplitudeScale;
    variants.push({
      parentSpaceGroup: rel.parent,
      childSpaceGroup:  rel.child,
      latticeDistortion: newLattice,
      atomicDisplacements: displacements,
      distortionAmplitude: totalDistortion,
      relation: rel,
    });
  }

  return variants;
}

export function applyBrokenSymmetry(
  atoms: { symbol: string; x: number; y: number; z: number }[],
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  variant: BrokenSymmetryVariant
): {
  atoms: { symbol: string; x: number; y: number; z: number }[];
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
} {
  const newAtoms = atoms.map((atom, i) => {
    const disp = variant.atomicDisplacements[i] ?? { dx: 0, dy: 0, dz: 0 };
    return {
      symbol: atom.symbol,
      x: atom.x + disp.dx * variant.latticeDistortion.a,
      y: atom.y + disp.dy * variant.latticeDistortion.b,
      z: atom.z + disp.dz * variant.latticeDistortion.c,
    };
  });
  return { atoms: newAtoms, lattice: { ...variant.latticeDistortion } };
}

export function getSymmetrySubgroupStats(): {
  totalRelations: number;
  spaceGroupsCovered: number;
  wyckoffSitesCovered: number;
  maxChainDepth: number;
  avgSubgroupIndex: number;
} {
  const parents  = Array.from(new Set(EXTENDED_SUBGROUP_HIERARCHY.map(e => e.parentSg)));
  const children = Array.from(new Set(EXTENDED_SUBGROUP_HIERARCHY.map(e => e.childSg)));
  const allSGs   = Array.from(new Set([...parents, ...children]));
  const avgIndex = EXTENDED_SUBGROUP_HIERARCHY.reduce((s, e) => s + e.index, 0) /
                   Math.max(1, EXTENDED_SUBGROUP_HIERARCHY.length);

  // Find max chain depth via BFS (bounded at 6 to stay cheap)
  let maxDepth = 0;
  for (const sg of parents) {
    const d = depthBFS(sg, 0, new Set<number>());
    if (d > maxDepth) maxDepth = d;
  }

  return {
    totalRelations:     EXTENDED_SUBGROUP_HIERARCHY.length,
    spaceGroupsCovered: allSGs.length,
    wyckoffSitesCovered: EXTENDED_WYCKOFF_DB.length,
    maxChainDepth:      maxDepth,
    avgSubgroupIndex:   Math.round(avgIndex * 100) / 100,
  };
}

function depthBFS(sg: number, depth: number, visited: Set<number>): number {
  if (depth >= 6 || visited.has(sg)) return depth;
  visited.add(sg);
  const children = getSubgroupEntriesFor(sg);
  if (children.length === 0) return depth;
  return Math.max(...children.map(c => depthBFS(c.childSg, depth + 1, new Set(visited))));
}

// ── SG name / number resolver ─────────────────────────────────────────────────
// Accepts: number, "Fm-3m", "SG-225", "225"

const _symbolToNumber = new Map<string, number>();
for (const [n, sym] of Object.entries(SG_HM_SYMBOL)) {
  _symbolToNumber.set(sym.toLowerCase(), Number(n));
  _symbolToNumber.set(`sg-${n}`, Number(n));
  _symbolToNumber.set(String(n), Number(n));
}

export function resolveSGNumber(input: string | number): number | null {
  if (typeof input === "number") return input >= 1 && input <= 230 ? input : null;
  const key = String(input).trim().toLowerCase();
  return _symbolToNumber.get(key) ?? null;
}

export function resolveSGSymbol(sgNumber: number): string {
  return SG_HM_SYMBOL[sgNumber] ?? `SG-${sgNumber}`;
}

// ── Convenience: THOR embedding directly from SG name/number ─────────────────

export function computeThorEmbeddingForSG(
  spaceGroupNameOrNumber: string | number,
  fracPosition?: [number, number, number]
): ThorEmbedding {
  const sgNum = resolveSGNumber(spaceGroupNameOrNumber) ?? 1;
  return computeThorEmbedding(sgNum, fracPosition);
}
