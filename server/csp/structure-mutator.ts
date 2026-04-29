/**
 * Structure Mutation Layer
 *
 * Takes candidate structures and produces variants through physically
 * motivated perturbations. Each mutation explores a different basin
 * near the parent structure — especially useful for VCA-derived candidates
 * that may be close to the ground state but trapped in a high-symmetry
 * or wrong local minimum.
 *
 * Mutations:
 * - Lattice strain (isotropic + anisotropic)
 * - Volume compression/expansion
 * - Random atomic displacement (global + per-species)
 * - Symmetry breaking (P1 reduction + perturbation)
 * - Hydrogen shuffle (reposition H within cage)
 * - Hydrogen vacancy/addition (variable composition)
 * - Wyckoff perturbation (displace from high-symmetry positions)
 * - Supercell expansion (2×1×1, 1×2×1, etc.)
 */

import type { CSPCandidate } from "./csp-types";
import { cellVolumeFromVectors } from "./csp-types";
import { latticeParamsToVectors } from "./poscar-io";

// ---------------------------------------------------------------------------
// RNG helpers
// ---------------------------------------------------------------------------

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng: () => number): number {
  const u1 = rng() || 1e-10;
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Mutation types
// ---------------------------------------------------------------------------

export type MutationType =
  | "lattice-strain"
  | "volume-compress"
  | "volume-expand"
  | "atomic-displacement"
  | "symmetry-break"
  | "hydrogen-shuffle"
  | "hydrogen-vacancy"
  | "hydrogen-addition"
  | "wyckoff-perturbation"
  | "supercell-2x";

export interface MutationResult {
  candidate: CSPCandidate;
  mutationType: MutationType;
  parentSeed: number;
  description: string;
}

// ---------------------------------------------------------------------------
// Individual mutations
// ---------------------------------------------------------------------------

function cloneCandidate(c: CSPCandidate, mutationType: MutationType, seed: number): CSPCandidate {
  return {
    ...c,
    positions: c.positions.map(p => ({ ...p })),
    cellVectors: c.cellVectors ? c.cellVectors.map(v => [...v] as [number, number, number]) : undefined,
    seed,
    parentSeeds: [...(c.parentSeeds ?? []), c.seed],
    source: `${mutationType} from ${c.source?.slice(0, 30) ?? "unknown"}`,
    prototype: `mutant-${mutationType}`,
    relaxationLevel: "raw",
    relaxationHistory: [...(c.relaxationHistory ?? []), `mutant-${mutationType}`],
  };
}

function wrapFrac(v: number): number {
  return ((v % 1) + 1) % 1;
}

/** Isotropic or anisotropic lattice strain (±2-8%) */
function mutateLatticeStrain(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "lattice-strain", seed);

  // Random strain tensor: isotropic with small anisotropic perturbation
  const isoStrain = 1.0 + (rng() - 0.5) * 0.10; // ±5%
  const anisoX = 1.0 + (rng() - 0.5) * 0.04;     // ±2%
  const anisoY = 1.0 + (rng() - 0.5) * 0.04;
  const anisoZ = 1.0 + (rng() - 0.5) * 0.04;

  mut.latticeA = parent.latticeA * isoStrain * anisoX;
  if (mut.latticeB) mut.latticeB = mut.latticeB * isoStrain * anisoY;
  if (mut.latticeC) mut.latticeC = (mut.latticeC ?? mut.latticeA) * isoStrain * anisoZ;
  mut.cOverA = (mut.latticeC ?? mut.latticeA) / mut.latticeA;

  return {
    candidate: mut,
    mutationType: "lattice-strain",
    parentSeed: parent.seed,
    description: `strain iso=${(isoStrain - 1) * 100 | 0}% aniso=[${((anisoX - 1) * 100) | 0},${((anisoY - 1) * 100) | 0},${((anisoZ - 1) * 100) | 0}]%`,
  };
}

/** Volume compression */
function mutateVolumeCompress(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const factor = 0.90 + rng() * 0.08; // 90-98% of original volume → compress lattice
  const linearFactor = Math.pow(factor, 1 / 3);
  const mut = cloneCandidate(parent, "volume-compress", seed);
  mut.latticeA *= linearFactor;
  if (mut.latticeB) mut.latticeB *= linearFactor;
  if (mut.latticeC) mut.latticeC *= linearFactor;
  return {
    candidate: mut,
    mutationType: "volume-compress",
    parentSeed: parent.seed,
    description: `${((1 - factor) * 100).toFixed(1)}% volume compression`,
  };
}

/** Volume expansion */
function mutateVolumeExpand(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const factor = 1.02 + rng() * 0.08; // 102-110% of original volume
  const linearFactor = Math.pow(factor, 1 / 3);
  const mut = cloneCandidate(parent, "volume-expand", seed);
  mut.latticeA *= linearFactor;
  if (mut.latticeB) mut.latticeB *= linearFactor;
  if (mut.latticeC) mut.latticeC *= linearFactor;
  return {
    candidate: mut,
    mutationType: "volume-expand",
    parentSeed: parent.seed,
    description: `${((factor - 1) * 100).toFixed(1)}% volume expansion`,
  };
}

/** Random atomic displacement (all atoms or per-species) */
function mutateAtomicDisplacement(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "atomic-displacement", seed);
  const sigma = 0.02 + rng() * 0.04; // 2-6% of lattice in fractional coords

  let maxDisp = 0;
  for (const pos of mut.positions) {
    const dx = gaussianRandom(rng) * sigma;
    const dy = gaussianRandom(rng) * sigma;
    const dz = gaussianRandom(rng) * sigma;
    pos.x = wrapFrac(pos.x + dx);
    pos.y = wrapFrac(pos.y + dy);
    pos.z = wrapFrac(pos.z + dz);
    maxDisp = Math.max(maxDisp, Math.sqrt(dx * dx + dy * dy + dz * dz));
  }

  return {
    candidate: mut,
    mutationType: "atomic-displacement",
    parentSeed: parent.seed,
    description: `sigma=${sigma.toFixed(3)} frac, maxDisp=${maxDisp.toFixed(3)}`,
  };
}

/** Symmetry breaking: reduce to P1 + random perturbation */
function mutateSymmetryBreak(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "symmetry-break", seed);

  // Reduce to P1
  mut.spaceGroup = "P1";
  mut.crystalSystem = "triclinic";

  // Small lattice angle perturbation (break orthorhombic/cubic symmetry)
  if (mut.latticeParams) {
    mut.latticeParams.alpha += (rng() - 0.5) * 3; // ±1.5°
    mut.latticeParams.beta += (rng() - 0.5) * 3;
    mut.latticeParams.gamma += (rng() - 0.5) * 3;
  }

  // Small position perturbation
  const sigma = 0.01 + rng() * 0.02;
  for (const pos of mut.positions) {
    pos.x = wrapFrac(pos.x + gaussianRandom(rng) * sigma);
    pos.y = wrapFrac(pos.y + gaussianRandom(rng) * sigma);
    pos.z = wrapFrac(pos.z + gaussianRandom(rng) * sigma);
  }

  return {
    candidate: mut,
    mutationType: "symmetry-break",
    parentSeed: parent.seed,
    description: `P1 reduction + sigma=${sigma.toFixed(3)} perturbation`,
  };
}

/** Hydrogen shuffle: reposition H atoms within the cage */
function mutateHydrogenShuffle(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "hydrogen-shuffle", seed);

  const hIndices = mut.positions
    .map((p, i) => p.element === "H" ? i : -1)
    .filter(i => i >= 0);

  if (hIndices.length < 2) {
    // No H to shuffle — do regular displacement instead
    return mutateAtomicDisplacement(parent, seed);
  }

  // Larger displacement for H than for metals
  const hSigma = 0.05 + rng() * 0.10; // 5-15% fractional
  for (const idx of hIndices) {
    mut.positions[idx].x = wrapFrac(mut.positions[idx].x + gaussianRandom(rng) * hSigma);
    mut.positions[idx].y = wrapFrac(mut.positions[idx].y + gaussianRandom(rng) * hSigma);
    mut.positions[idx].z = wrapFrac(mut.positions[idx].z + gaussianRandom(rng) * hSigma);
  }

  return {
    candidate: mut,
    mutationType: "hydrogen-shuffle",
    parentSeed: parent.seed,
    description: `shuffled ${hIndices.length} H atoms, sigma=${hSigma.toFixed(3)}`,
  };
}

/** Hydrogen vacancy: remove one H atom */
function mutateHydrogenVacancy(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "hydrogen-vacancy", seed);

  const hIndices = mut.positions
    .map((p, i) => p.element === "H" ? i : -1)
    .filter(i => i >= 0);

  if (hIndices.length < 2) {
    return mutateAtomicDisplacement(parent, seed);
  }

  // Remove a random H
  const removeIdx = hIndices[Math.floor(rng() * hIndices.length)];
  mut.positions.splice(removeIdx, 1);

  return {
    candidate: mut,
    mutationType: "hydrogen-vacancy",
    parentSeed: parent.seed,
    description: `removed H at index ${removeIdx}, ${mut.positions.length} atoms remaining`,
  };
}

/** Hydrogen addition: add one H at a cage interstitial */
function mutateHydrogenAddition(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "hydrogen-addition", seed);

  // Place H at a random position that's not too close to existing atoms
  let placed = false;
  for (let attempt = 0; attempt < 50; attempt++) {
    const x = rng(), y = rng(), z = rng();
    let tooClose = false;
    for (const pos of mut.positions) {
      let dx = x - pos.x, dy = y - pos.y, dz = z - pos.z;
      dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) * mut.latticeA;
      if (dist < 0.8) { tooClose = true; break; }
    }
    if (!tooClose) {
      mut.positions.push({ element: "H", x, y, z });
      placed = true;
      break;
    }
  }

  if (!placed) return mutateAtomicDisplacement(parent, seed);

  return {
    candidate: mut,
    mutationType: "hydrogen-addition",
    parentSeed: parent.seed,
    description: `added H, ${mut.positions.length} atoms total`,
  };
}

/** Wyckoff perturbation: displace atoms slightly from high-symmetry positions */
function mutateWyckoffPerturbation(parent: CSPCandidate, seed: number): MutationResult {
  const rng = seededRandom(seed);
  const mut = cloneCandidate(parent, "wyckoff-perturbation", seed);

  // Only perturb a fraction of atoms (30-70%)
  const perturbFraction = 0.3 + rng() * 0.4;
  const sigma = 0.005 + rng() * 0.015; // Small: 0.5-2% of lattice

  let perturbed = 0;
  for (const pos of mut.positions) {
    if (rng() < perturbFraction) {
      pos.x = wrapFrac(pos.x + gaussianRandom(rng) * sigma);
      pos.y = wrapFrac(pos.y + gaussianRandom(rng) * sigma);
      pos.z = wrapFrac(pos.z + gaussianRandom(rng) * sigma);
      perturbed++;
    }
  }

  return {
    candidate: mut,
    mutationType: "wyckoff-perturbation",
    parentSeed: parent.seed,
    description: `perturbed ${perturbed}/${mut.positions.length} atoms, sigma=${sigma.toFixed(4)}`,
  };
}

// ---------------------------------------------------------------------------
// Main mutation function
// ---------------------------------------------------------------------------

/**
 * Generate mutant variants of a candidate structure.
 *
 * @param parent - The candidate to mutate
 * @param nMutants - How many mutants to generate (default 6)
 * @param seed - Base RNG seed
 * @param allowVariableComposition - If true, allow H vacancy/addition mutations
 * @returns Array of mutated candidates
 */
export function mutateCandidate(
  parent: CSPCandidate,
  nMutants: number = 6,
  seed?: number,
  allowVariableComposition: boolean = false,
): MutationResult[] {
  const baseSeed = seed ?? (parent.seed * 7 + 13);
  const hasH = parent.positions.some(p => p.element === "H");
  const hCount = parent.positions.filter(p => p.element === "H").length;

  // Build mutation menu weighted by relevance
  const mutations: Array<{ fn: (p: CSPCandidate, s: number) => MutationResult; weight: number }> = [
    { fn: mutateLatticeStrain, weight: 2.0 },
    { fn: mutateVolumeCompress, weight: 1.5 },
    { fn: mutateVolumeExpand, weight: 1.5 },
    { fn: mutateAtomicDisplacement, weight: 2.0 },
    { fn: mutateSymmetryBreak, weight: 1.0 },
    { fn: mutateWyckoffPerturbation, weight: 1.5 },
  ];

  if (hasH && hCount >= 3) {
    mutations.push({ fn: mutateHydrogenShuffle, weight: 3.0 }); // High weight for hydrides
  }
  if (allowVariableComposition && hasH && hCount >= 3) {
    mutations.push({ fn: mutateHydrogenVacancy, weight: 1.0 });
    mutations.push({ fn: mutateHydrogenAddition, weight: 1.0 });
  }

  // Weighted random selection
  const totalWeight = mutations.reduce((s, m) => s + m.weight, 0);
  const rng = seededRandom(baseSeed);
  const results: MutationResult[] = [];

  for (let i = 0; i < nMutants; i++) {
    let r = rng() * totalWeight;
    let selected = mutations[0];
    for (const m of mutations) {
      r -= m.weight;
      if (r <= 0) { selected = m; break; }
    }
    const mutSeed = baseSeed + i * 1000 + 1;
    results.push(selected.fn(parent, mutSeed));
  }

  return results;
}

/**
 * Mutate the top N candidates from a batch.
 * Produces mutants for the best candidates (by confidence or enthalpy).
 */
export function mutateTopCandidates(
  candidates: CSPCandidate[],
  topN: number = 5,
  mutantsPerCandidate: number = 6,
  allowVariableComposition: boolean = false,
): CSPCandidate[] {
  // Sort by enthalpy (if available) or confidence
  const sorted = [...candidates].sort((a, b) => {
    if (a.enthalpyPerAtom != null && b.enthalpyPerAtom != null) {
      return a.enthalpyPerAtom - b.enthalpyPerAtom;
    }
    return (b.confidence ?? 0) - (a.confidence ?? 0);
  });

  const topCandidates = sorted.slice(0, topN);
  const allMutants: CSPCandidate[] = [];

  for (const parent of topCandidates) {
    const mutations = mutateCandidate(parent, mutantsPerCandidate, undefined, allowVariableComposition);
    for (const m of mutations) {
      allMutants.push(m.candidate);
    }
  }

  console.log(`[Mutator] Generated ${allMutants.length} mutants from top ${topCandidates.length} candidates (${allMutants.map(m => m.prototype.replace("mutant-", "")).join(", ")})`);
  return allMutants;
}
