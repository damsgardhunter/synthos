/**
 * Vegard's Law Lattice Estimation & Multi-Candidate Structure Generation
 *
 * For a compound like BiGeSb, this module:
 * 1. Enumerates all binary/ternary subsystems (Bi-Ge, Bi-Sb, Ge-Sb)
 * 2. Fetches known structures from AFLOW + Materials Project
 * 3. Applies Vegard's law interpolation for lattice parameter estimation
 * 4. Generates ranked structure candidates for the staged relaxation pipeline
 */

import {
  fetchAflowByElements,
  fetchAflowByTernaryElements,
  type AflowStructureEndpoint,
} from "../learning/aflow-client";
import {
  fetchMPStructureData,
  fetchSummary,
  type MPStructureData,
  type MPSummaryData,
} from "../learning/materials-project-client";
import { getElementData } from "../learning/elemental-data";
import { predictWithLocalXGB, loadLatestXGBWeights } from "./ml-weight-loader";
import {
  selectPrototype,
  estimateLatticeConstant as protoEstimateLattice,
  type PrototypeTemplate,
} from "../learning/crystal-prototypes";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BinaryEndpoint {
  compound: string;
  elements: string[];
  volumePerAtom: number;     // A^3/atom
  latticeA: number | null;   // A (from MP if available)
  latticeB: number | null;
  latticeC: number | null;
  latticeSystem: string;
  spaceGroup: string;
  isMetallic: boolean | null;
  source: "AFLOW" | "MP";
  /** Atomic positions from MP (fractional coordinates). Null for AFLOW-only entries. */
  positions: Array<{ element: string; x: number; y: number; z: number }> | null;
}

export interface VegardEstimate {
  latticeA: number;
  volumePerAtom: number;
  confidence: number;         // 0-1, based on how many endpoints found
  endpointsUsed: string[];    // formulas of compounds used
  method: "vegard" | "vegard-volume" | "volume-sum-fallback";
  isMetallic: boolean | null; // consensus from endpoints
  /** Raw binary endpoints with positions (for VCA position interpolation). */
  binaryEndpoints?: BinaryEndpoint[];
}

export interface StructureCandidate {
  latticeA: number;
  latticeB?: number;
  latticeC?: number;
  cOverA?: number;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  prototype: string;
  crystalSystem: string;
  spaceGroup: string;
  source: string;
  confidence: number;         // 0-1
  isMetallic: boolean | null;
}

// ---------------------------------------------------------------------------
// Combinatorics helpers
// ---------------------------------------------------------------------------

function combinations<T>(arr: T[], k: number): T[][] {
  if (k === 0) return [[]];
  if (k > arr.length) return [];
  const result: T[][] = [];
  for (let i = 0; i <= arr.length - k; i++) {
    const rest = combinations(arr.slice(i + 1), k - 1);
    for (const combo of rest) {
      result.push([arr[i], ...combo]);
    }
  }
  return result;
}

/** Common stoichiometries for a binary pair AB. */
function binaryStoichiometries(el1: string, el2: string): string[] {
  const sorted = [el1, el2].sort();
  const [a, b] = sorted;
  return [
    `${a}${b}`,          // AB
    `${a}2${b}`,         // A2B
    `${a}${b}2`,         // AB2
    `${a}3${b}`,         // A3B
    `${a}${b}3`,         // AB3
    `${a}3${b}2`,        // A3B2
    `${a}2${b}3`,        // A2B3
    `${a}2${b}5`,        // A2B5
    `${a}5${b}3`,        // A5B3
  ];
}

// ---------------------------------------------------------------------------
// Binary/ternary endpoint fetching
// ---------------------------------------------------------------------------

/**
 * Fetch known structures for a binary element pair from AFLOW + MP.
 * All fetches are parallel; results are merged and deduplicated.
 */
async function fetchBinaryEndpoints(el1: string, el2: string): Promise<BinaryEndpoint[]> {
  const endpoints: BinaryEndpoint[] = [];

  // Parallel: AFLOW species query + MP per-stoichiometry queries
  const stoichs = binaryStoichiometries(el1, el2);
  const mpQueries = stoichs.slice(0, 5).map(f => fetchMPStructureData(f).catch(() => null));

  const [aflowResults, ...mpResults] = await Promise.allSettled([
    fetchAflowByElements(el1, el2),
    ...mpQueries,
  ]);

  // Process AFLOW results
  if (aflowResults.status === "fulfilled" && aflowResults.value.length > 0) {
    for (const entry of aflowResults.value) {
      endpoints.push({
        compound: entry.compound,
        elements: entry.elements,
        volumePerAtom: entry.volumeAtom,
        latticeA: null,
        latticeB: null,
        latticeC: null,
        latticeSystem: entry.latticeSystem,
        spaceGroup: entry.spaceGroupSymbol,
        isMetallic: entry.bandgap != null ? entry.bandgap < 0.01 : null,
        source: "AFLOW",
        positions: entry.positions ?? null,  // DFT-relaxed positions from AFLOW
      });
    }
  }

  // Process MP results — these include actual atomic positions
  for (let i = 0; i < mpResults.length; i++) {
    const r = mpResults[i];
    if (r.status !== "fulfilled" || !r.value) continue;
    const mp = r.value as MPStructureData;
    const formula = stoichs[i];

    const vol = mp.latticeParams.a * mp.latticeParams.b * mp.latticeParams.c;
    const nAtoms = mp.atomicPositions.length || 1;

    endpoints.push({
      compound: formula,
      elements: [el1, el2].sort(),
      volumePerAtom: vol / nAtoms,
      latticeA: mp.latticeParams.a,
      latticeB: mp.latticeParams.b,
      latticeC: mp.latticeParams.c,
      latticeSystem: guessCrystalSystem(mp.latticeParams),
      spaceGroup: mp.spaceGroup ?? "",
      isMetallic: null,
      source: "MP",
      positions: mp.atomicPositions,  // Real Wyckoff positions from DFT
    });
  }

  return endpoints;
}

function guessCrystalSystem(params: { a: number; b: number; c: number }): string {
  const { a, b, c } = params;
  const tol = 0.05; // 5% tolerance
  if (Math.abs(a - b) / a < tol && Math.abs(b - c) / b < tol) return "cubic";
  if (Math.abs(a - b) / a < tol && Math.abs(c - a) / a > tol) return "tetragonal";
  return "orthorhombic";
}

// ---------------------------------------------------------------------------
// Vegard's law interpolation
// ---------------------------------------------------------------------------

/**
 * Compute Vegard's law volume estimate for an N-ary compound from binary endpoints.
 *
 * For compound A_x B_y C_z with mole fractions f_A, f_B, f_C:
 *   V_per_atom = sum over all pairs (i,j): w_ij * V_ij
 *   where w_ij = 2 * f_i * f_j (normalized so sum = 1)
 *
 * This is the generalized Vegard's law for multi-component solid solutions.
 */
export async function vegardEstimate(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number = 0,
): Promise<VegardEstimate> {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = (counts[el] || 1) / totalAtoms;
  }

  // Fetch binary subsystem endpoints in parallel. Binary data is the
  // critical input for Vegard interpolation — ternary queries almost always
  // 504 from AFLOW and are not needed (binary pairwise volumes are sufficient).
  // Ternary data is fetched separately as fire-and-forget enrichment.
  const binaryPairs = combinations(elements, 2);

  let allEndpoints: BinaryEndpoint[] = [];
  try {
    // Binary lookups only — cached ones return in <100ms from Neon DB,
    // fresh ones take 2-5s from AFLOW. No ternary 504s to block.
    const binaryResults = await Promise.allSettled(
      binaryPairs.map(([a, b]) => fetchBinaryEndpoints(a, b))
    );
    for (const r of binaryResults) {
      if (r.status === "fulfilled") allEndpoints.push(...r.value);
    }
  } catch {
    // Network failure — proceed with fallback
  }

  // Fire-and-forget: ternary enrichment (don't block Vegard on these)
  if (elements.length >= 3) {
    const ternaryTriples = combinations(elements, 3);
    for (const [a, b, c] of ternaryTriples) {
      fetchAflowByTernaryElements(a, b, c).catch(() => {}); // populate cache for future use
    }
  }

  const endpointsUsed = allEndpoints.map(e => e.compound);
  const metallicVotes = allEndpoints.filter(e => e.isMetallic != null);
  const isMetallic = metallicVotes.length > 0
    ? metallicVotes.filter(e => e.isMetallic).length > metallicVotes.length / 2
    : null;

  // If we have enough endpoints, do Vegard interpolation
  if (allEndpoints.length >= 2) {
    // Group endpoints by element pair, take average volume per pair
    const pairVolumes: Map<string, number[]> = new Map();
    for (const ep of allEndpoints) {
      const key = ep.elements.sort().join("-");
      if (!pairVolumes.has(key)) pairVolumes.set(key, []);
      pairVolumes.get(key)!.push(ep.volumePerAtom);
    }

    // Weighted average: for each pair (i,j), weight = 2*f_i*f_j
    let vegardVol = 0;
    let totalWeight = 0;
    let pairsWithData = 0;

    for (const [pair, volumes] of Array.from(pairVolumes.entries())) {
      const [elA, elB] = pair.split("-");
      const fA = fractions[elA] ?? 0;
      const fB = fractions[elB] ?? 0;
      if (fA === 0 || fB === 0) continue; // Element not in our compound

      const avgVol = volumes.reduce((s, v) => s + v, 0) / volumes.length;
      const weight = 2 * fA * fB;
      vegardVol += weight * avgVol;
      totalWeight += weight;
      pairsWithData++;
    }

    if (totalWeight > 0 && pairsWithData >= 1) {
      const volPerAtom = vegardVol / totalWeight;

      // Apply pressure correction (Birch-Murnaghan)
      let correctedVol = volPerAtom;
      if (pressureGPa > 0) {
        const B0 = estimateBulkModulusFromElements(elements);
        const B0p = 4.0;
        const eta = 1 + B0p * (pressureGPa / B0);
        const volRatio = eta > 0 ? Math.pow(eta, -1 / B0p) : 0.5;
        correctedVol = volPerAtom * Math.max(0.5, Math.min(1.0, volRatio));
      }

      const cellVolume = correctedVol * totalAtoms;
      const latticeA = Math.cbrt(cellVolume);

      // Confidence: higher with more binary pairs covered and more endpoints
      const totalPairs = binaryPairs.length;
      const pairCoverage = pairsWithData / Math.max(totalPairs, 1);
      const endpointBonus = Math.min(allEndpoints.length / 10, 0.3);
      const confidence = Math.min(0.4 + pairCoverage * 0.4 + endpointBonus, 0.95);

      return {
        latticeA: Math.max(latticeA, 3.0),
        volumePerAtom: correctedVol,
        confidence,
        endpointsUsed,
        method: "vegard-volume",
        isMetallic,
        binaryEndpoints: allEndpoints,
      };
    }
  }

  // Fallback: use elemental lattice constants with linear Vegard
  if (allEndpoints.length >= 1) {
    const avgVol = allEndpoints.reduce((s, e) => s + e.volumePerAtom, 0) / allEndpoints.length;
    const cellVolume = avgVol * totalAtoms;
    const latticeA = Math.cbrt(cellVolume);

    return {
      latticeA: Math.max(latticeA, 3.0),
      volumePerAtom: avgVol,
      confidence: 0.3,
      endpointsUsed,
      method: "vegard",
      isMetallic,
      binaryEndpoints: allEndpoints,
    };
  }

  // Pure fallback: elemental volumes (same as existing estimateLatticeConstant)
  let totalVol = 0;
  for (const el of elements) {
    const n = counts[el] || 1;
    const elData = getElementData(el);
    const atomicVol = elData?.atomicVolume ?? ((elData?.atomicRadius ?? 130) / 100) ** 3 * (4 / 3) * Math.PI;
    totalVol += n * atomicVol;
  }
  const packingFactor = elements.some(e => {
    const d = getElementData(e);
    return d && d.valenceElectrons >= 3 && d.valenceElectrons <= 12;
  }) ? 0.68 : 0.60;
  const cellVolume = totalVol / packingFactor;
  const latticeA = Math.cbrt(cellVolume);

  return {
    latticeA: Math.max(latticeA, 3.0),
    volumePerAtom: cellVolume / totalAtoms,
    confidence: 0.15,
    endpointsUsed: [],
    method: "volume-sum-fallback",
    isMetallic: null,
  };
}

function estimateBulkModulusFromElements(elements: string[]): number {
  let sumB = 0;
  let count = 0;
  for (const el of elements) {
    const d = getElementData(el);
    if (d?.bulkModulus) {
      sumB += d.bulkModulus;
      count++;
    }
  }
  return count > 0 ? sumB / count : 100; // Default 100 GPa
}

// ---------------------------------------------------------------------------
// Virtual Crystal Approximation (VCA) for atomic positions
// ---------------------------------------------------------------------------

/**
 * Interpolate atomic positions for an N-ary compound from known binary structures.
 *
 * Analogous to Vegard's law for lattice constants: if we know the positions of
 * atoms in BiGe (from MP) and BiSb (from MP), we can construct plausible
 * positions for BiGeSb by:
 * 1. Finding the binary with the most similar stoichiometry to the target
 * 2. Scaling positions to match the target atom count
 * 3. Substituting missing elements into appropriate sites
 *
 * Falls back to null if no binary positions are available.
 */
function interpolatePositionsFromEndpoints(
  elements: string[],
  counts: Record<string, number>,
  allEndpoints: BinaryEndpoint[],
): Array<{ element: string; x: number; y: number; z: number }> | null {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  // Find endpoints with real atomic positions (from MP)
  const withPositions = allEndpoints.filter(e => e.positions && e.positions.length > 0);
  if (withPositions.length === 0) return null;

  // Score each endpoint by how useful it is for the target compound:
  // - Prefer endpoints whose elements overlap with the target
  // - Prefer endpoints with similar atom counts
  // - Prefer endpoints from MP (have positions) over AFLOW
  const scored = withPositions.map(ep => {
    const overlap = ep.elements.filter(e => elements.includes(e)).length;
    const atomRatio = ep.positions!.length / totalAtoms;
    const sizeScore = 1 - Math.abs(1 - atomRatio);  // Closer to 1:1 is better
    return { ep, score: overlap * 2 + sizeScore };
  });
  scored.sort((a, b) => b.score - a.score);

  const best = scored[0];
  if (!best || !best.ep.positions) return null;

  const templatePositions = best.ep.positions;
  const templateElements = new Set(templatePositions.map(p => p.element));

  // Build target positions by adapting the template with chemical-aware
  // site assignment. Large electropositives (Ca, Ba, Sr, K, etc.) should
  // map to sites that held large atoms in the template — NOT to framework
  // sites that held small metallic/covalent atoms. This prevents placing
  // Ca on a Bi site (ionic vs metallic bonding mismatch).
  const result: Array<{ element: string; x: number; y: number; z: number }> = [];

  // Chemical character classification
  const LARGE_ELECTROPOSITIVE = new Set([
    "K", "Rb", "Cs", "Ca", "Sr", "Ba", "Na", "Li",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Y", "Sc",
  ]);
  const ANION_LIKE = new Set(["O", "F", "S", "Se", "Te", "N", "P", "As", "Cl", "Br", "I"]);

  function getChemicalClass(el: string): "electropositive" | "anion" | "framework" {
    if (LARGE_ELECTROPOSITIVE.has(el)) return "electropositive";
    if (ANION_LIKE.has(el)) return "anion";
    return "framework"; // metals, semimetals, etc.
  }

  // Estimate "site size" from the template: sites that held large atoms
  // should receive large target atoms
  function getAtomicSize(el: string): number {
    const d = getElementData(el);
    return d?.atomicRadius ?? 130;
  }

  // Map template elements to target elements respecting chemical character
  const templateEls = Array.from(templateElements);
  const targetEls = [...elements];
  const mapping: Record<string, string> = {};

  // Direct matches first
  for (const tel of templateEls) {
    if (targetEls.includes(tel)) {
      mapping[tel] = tel;
    }
  }

  // Map remaining by chemical similarity: match electropositives to
  // electropositives, anions to anions, framework to framework.
  // Within each class, match by atomic radius similarity.
  const unmappedTemplate = templateEls.filter(e => !mapping[e]);
  const unmappedTarget = targetEls.filter(e => !Object.values(mapping).includes(e));

  // Sort both by chemical class priority then by size
  const classOrder = { "electropositive": 0, "anion": 1, "framework": 2 };

  for (const tel of unmappedTemplate) {
    const telClass = getChemicalClass(tel);
    const telSize = getAtomicSize(tel);

    // Find best matching unmapped target element:
    // 1. Same chemical class preferred
    // 2. Within class, closest atomic radius
    let bestTarget: string | null = null;
    let bestScore = -Infinity;

    const availableTargets = unmappedTarget.filter(e => !Object.values(mapping).includes(e));
    for (const te of availableTargets) {
      const teClass = getChemicalClass(te);
      const teSize = getAtomicSize(te);

      let score = 0;
      if (teClass === telClass) score += 100;  // Same chemical class: strong preference
      score -= Math.abs(telSize - teSize) * 0.5; // Penalize size mismatch

      if (score > bestScore) {
        bestScore = score;
        bestTarget = te;
      }
    }

    if (bestTarget) {
      mapping[tel] = bestTarget;
    }
  }

  // Build positions using the mapping
  const elementCounts: Record<string, number> = {};
  for (const pos of templatePositions) {
    const targetEl = mapping[pos.element];
    if (!targetEl) continue;
    const needed = counts[targetEl] || 0;
    const placed = elementCounts[targetEl] || 0;
    if (placed >= needed) continue;

    result.push({
      element: targetEl,
      x: pos.x,
      y: pos.y,
      z: pos.z,
    });
    elementCounts[targetEl] = placed + 1;
  }

  // Fill any remaining target atoms by duplicating with offsets
  for (const el of elements) {
    const needed = Math.round(counts[el] || 0);
    const placed = elementCounts[el] || 0;
    const deficit = needed - placed;
    if (deficit <= 0) continue;

    // Find existing positions of this element (or similar) to perturb
    const existing = result.filter(p => p.element === el);
    for (let i = 0; i < deficit; i++) {
      if (existing.length > 0) {
        const ref = existing[i % existing.length];
        result.push({
          element: el,
          x: (ref.x + 0.2 + i * 0.15) % 1.0,
          y: (ref.y + 0.3 + i * 0.1) % 1.0,
          z: (ref.z + 0.1 + i * 0.2) % 1.0,
        });
      } else {
        // No reference — place on a grid offset
        const n = Math.ceil(Math.cbrt(totalAtoms));
        const idx = result.length;
        result.push({
          element: el,
          x: ((idx % n) + 0.5) / n,
          y: ((Math.floor(idx / n) % n) + 0.5) / n,
          z: ((Math.floor(idx / (n * n)) % n) + 0.5) / n,
        });
      }
    }
    elementCounts[el] = needed;
  }

  return result.length > 0 ? result : null;
}

// ---------------------------------------------------------------------------
// Multi-candidate structure generation
// ---------------------------------------------------------------------------

/**
 * Generate ranked structure candidates for a given formula.
 * Returns up to `maxCandidates` candidates sorted by confidence.
 *
 * Sources (highest to lowest confidence):
 * 1. Materials Project direct structure (exact formula match)
 * 2. AFLOW + Vegard-interpolated lattice with prototype positions
 * 3. Existing prototype templates with Vegard-adjusted lattice
 * 4. Volume-sum fallback with prototype positions
 */
export async function generateStructureCandidates(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number = 0,
  maxCandidates: number = 10,
): Promise<StructureCandidate[]> {
  const candidates: StructureCandidate[] = [];
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  // Parallel: fetch MP structure + Vegard estimate + MP summary
  const [mpStructResult, vegardResult, mpSummaryResult] = await Promise.allSettled([
    fetchMPStructureData(formula).catch(() => null),
    vegardEstimate(elements, counts, pressureGPa),
    fetchSummary(formula).catch(() => null),
  ]);

  const mpStruct = mpStructResult.status === "fulfilled" ? mpStructResult.value as MPStructureData | null : null;
  const vegard = vegardResult.status === "fulfilled" ? vegardResult.value : null;
  const mpSummary = mpSummaryResult.status === "fulfilled" ? mpSummaryResult.value as MPSummaryData | null : null;

  const isMetallic = mpSummary?.isMetallic ?? vegard?.isMetallic ?? null;

  // --- Source 1: MP direct structure (highest confidence) ---
  if (mpStruct && mpStruct.atomicPositions.length > 0) {
    const a = mpStruct.latticeParams.a;
    const b = mpStruct.latticeParams.b;
    const c = mpStruct.latticeParams.c;
    candidates.push({
      latticeA: a,
      latticeB: b,
      latticeC: c,
      cOverA: a > 0 ? c / a : 1.0,
      positions: mpStruct.atomicPositions,
      prototype: "MP-direct",
      crystalSystem: guessCrystalSystem(mpStruct.latticeParams),
      spaceGroup: mpStruct.spaceGroup ?? "",
      source: `Materials Project (${formula})`,
      confidence: 0.90,
      isMetallic,
    });
  }

  // --- Source 2: Vegard estimate + prototype positions ---
  if (vegard && vegard.confidence > 0.2) {
    // Try to match a prototype template for the positions
    const proto = selectPrototype(formula);
    if (proto) {
      // Use prototype positions with Vegard-adjusted lattice
      const protoLattice = protoEstimateLattice(elements, counts, proto.template);
      const vegardA = vegard.latticeA;

      // Blend: 60% Vegard, 40% prototype estimate (if both available)
      const blendedA = protoLattice ? vegardA * 0.6 + protoLattice.a * 0.4 : vegardA;
      const cOverA = protoLattice?.c && protoLattice.a ? protoLattice.c / protoLattice.a : 1.0;

      const positions = buildPositionsFromPrototype(proto, elements, counts);
      if (positions.length > 0) {
        candidates.push({
          latticeA: blendedA,
          cOverA,
          positions,
          prototype: `Vegard+${proto.template.name ?? "prototype"}`,
          crystalSystem: proto.template.latticeType ?? "cubic",
          spaceGroup: proto.template.spaceGroup ?? "",
          source: `Vegard (${vegard.endpointsUsed.length} endpoints, conf=${vegard.confidence.toFixed(2)})`,
          confidence: vegard.confidence * 0.85,
          isMetallic: vegard.isMetallic,
        });
      }
    }

    // VCA position interpolation: use binary endpoint positions to build
    // physically-motivated atomic positions (analogous to Vegard for lattice).
    const vcaPositions = vegard.binaryEndpoints
      ? interpolatePositionsFromEndpoints(elements, counts, vegard.binaryEndpoints)
      : null;

    if (vcaPositions && vcaPositions.length > 0) {
      candidates.push({
        latticeA: vegard.latticeA,
        positions: vcaPositions,
        prototype: "VCA-interpolated",
        crystalSystem: "cubic",
        spaceGroup: "",
        source: `VCA positions from ${vegard.binaryEndpoints?.filter(e => e.positions).length ?? 0} binary structures`,
        confidence: vegard.confidence * 0.70,
        isMetallic: vegard.isMetallic,
      });
    }

    // Fallback: simple grid positions with Vegard lattice (low quality)
    const simpleCubicPositions = generateSimplePositions(elements, counts, totalAtoms);
    if (simpleCubicPositions.length > 0) {
      candidates.push({
        latticeA: vegard.latticeA,
        positions: simpleCubicPositions,
        prototype: "Vegard-simple",
        crystalSystem: "cubic",
        spaceGroup: "",
        source: `Vegard volume interpolation (${vegard.endpointsUsed.length} endpoints)`,
        confidence: vegard.confidence * 0.6,
        isMetallic: vegard.isMetallic,
      });
    }
  }

  // --- Source 3: Prototype templates with volume-sum lattice ---
  const proto = selectPrototype(formula);
  if (proto) {
    const protoLattice = protoEstimateLattice(elements, counts, proto.template);
    const positions = buildPositionsFromPrototype(proto, elements, counts);
    if (positions.length > 0 && protoLattice) {
      candidates.push({
        latticeA: protoLattice.a,
        cOverA: protoLattice.c / protoLattice.a,
        positions,
        prototype: proto.template.name ?? "prototype",
        crystalSystem: proto.template.latticeType ?? "cubic",
        spaceGroup: proto.template.spaceGroup ?? "",
        source: `Crystal prototype (${proto.template.name})`,
        confidence: 0.35,
        isMetallic,
      });
    }
  }

  // --- Source 4: Volume-sum fallback (lowest confidence) ---
  {
    const fallbackPositions = generateSimplePositions(elements, counts, totalAtoms);
    // Use elemental data for rough lattice estimate
    let totalVol = 0;
    for (const el of elements) {
      const n = counts[el] || 1;
      const d = getElementData(el);
      const r = (d?.atomicRadius ?? 130) / 100; // pm -> A
      totalVol += n * (4 / 3) * Math.PI * r * r * r;
    }
    const a = Math.max(Math.cbrt(totalVol / 0.68), 3.0);

    if (fallbackPositions.length > 0) {
      candidates.push({
        latticeA: a,
        positions: fallbackPositions,
        prototype: "volume-sum",
        crystalSystem: "cubic",
        spaceGroup: "",
        source: "Volume-sum fallback (elemental radii)",
        confidence: 0.15,
        isMetallic,
      });
    }
  }

  // --- Source 5: Related compound variants from AFLOW/MP ---
  // For BiGeSb: pull known Bi-Ge, Bi-Sb, Ge-Sb structures and create
  // variants by substituting the third element into known binary structures.
  try {
    const relatedCandidates = await generateRelatedCompoundVariants(
      formula, elements, counts, vegard, isMetallic
    );
    for (const rc of relatedCandidates) {
      candidates.push(rc);
    }
  } catch (varErr: any) {
    // Non-critical — don't block on this
  }

  // --- Source 6: Local ML structure prediction (if weights available) ---
  // Uses DB-stored model weights for lattice/stability prediction without
  // requiring the GCP service. Added as a candidate if confidence is reasonable.
  try {
    const mlCandidate = await generateMLStructureCandidate(
      formula, elements, counts, totalAtoms, isMetallic
    );
    if (mlCandidate) candidates.push(mlCandidate);
  } catch {
    // Non-critical
  }

  // Deduplicate: remove candidates with very similar lattice constants
  const deduped = deduplicateCandidates(candidates);

  // Sort by confidence descending, return top N
  deduped.sort((a, b) => b.confidence - a.confidence);
  const result = deduped.slice(0, maxCandidates);

  console.log(`[Vegard] ${formula}: generated ${result.length} structure candidates (from ${candidates.length} raw, ${deduped.length} deduped)`);
  for (const c of result) {
    console.log(`[Vegard]   ${c.source}: a=${c.latticeA.toFixed(3)} A, ${c.positions.length} atoms, conf=${c.confidence.toFixed(2)}, ${c.crystalSystem}`);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Helper: build positions from prototype template
// ---------------------------------------------------------------------------

function buildPositionsFromPrototype(
  proto: { template: PrototypeTemplate; siteMap: Record<string, string> },
  elements: string[],
  counts: Record<string, number>,
): Array<{ element: string; x: number; y: number; z: number }> {
  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

  if (!proto.template.sites || !Array.isArray(proto.template.sites)) {
    return [];
  }

  for (const site of proto.template.sites) {
    const siteLabel = site.label;
    const mappedElement = proto.siteMap[siteLabel];
    if (!mappedElement || !elements.includes(mappedElement)) continue;

    positions.push({
      element: mappedElement,
      x: site.x,
      y: site.y,
      z: site.z,
    });
  }

  return positions;
}

// ---------------------------------------------------------------------------
// Helper: generate simple cubic positions for N atoms
// ---------------------------------------------------------------------------

function generateSimplePositions(
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number,
): Array<{ element: string; x: number; y: number; z: number }> {
  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

  // Distribute atoms on a simple grid in fractional coordinates
  const n = Math.ceil(Math.cbrt(totalAtoms));
  let idx = 0;

  for (const el of elements) {
    const count = Math.round(counts[el] || 1);
    for (let i = 0; i < count; i++) {
      const ix = idx % n;
      const iy = Math.floor(idx / n) % n;
      const iz = Math.floor(idx / (n * n)) % n;
      positions.push({
        element: el,
        x: (ix + 0.1) / n,
        y: (iy + 0.1) / n,
        z: (iz + 0.1) / n,
      });
      idx++;
    }
  }

  return positions;
}

// ---------------------------------------------------------------------------
// Helper: deduplicate candidates with similar lattice constants
// ---------------------------------------------------------------------------

function deduplicateCandidates(candidates: StructureCandidate[]): StructureCandidate[] {
  const result: StructureCandidate[] = [];
  for (const c of candidates) {
    const isDupe = result.some(existing =>
      Math.abs(existing.latticeA - c.latticeA) < 0.1 &&
      existing.prototype === c.prototype
    );
    if (!isDupe) result.push(c);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Source 5: Related compound variant generator
// ---------------------------------------------------------------------------

/**
 * For a compound like BiGeSb, pull known binary structures (BiGe, BiSb, GeSb)
 * from AFLOW/MP and create variant candidates by:
 * 1. Taking a known binary structure (e.g., BiGe with a=5.3 A, hexagonal)
 * 2. Substituting the missing element into available sites
 * 3. Adjusting lattice via Vegard interpolation
 *
 * This generates physically-motivated starting structures grounded in
 * experimentally known crystal chemistry.
 */
async function generateRelatedCompoundVariants(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  vegard: VegardEstimate | null,
  isMetallic: boolean | null,
): Promise<StructureCandidate[]> {
  const candidates: StructureCandidate[] = [];
  if (elements.length < 2) return candidates;

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const binaryPairs = combinations(elements, 2);

  // For each binary pair, fetch MP structure and create ternary+ variants
  const mpFetches = binaryPairs.slice(0, 4).map(async ([el1, el2]) => {
    // Try multiple stoichiometries dynamically
    const stoichs = generateDynamicStoichiometries(el1, el2);
    for (const stoich of stoichs.slice(0, 3)) {
      try {
        const mpStruct = await fetchMPStructureData(stoich);
        if (!mpStruct || mpStruct.atomicPositions.length === 0) continue;

        // Found a known binary — create a variant by substituting the
        // remaining elements into the structure
        const missingEls = elements.filter(e => e !== el1 && e !== el2);
        if (missingEls.length === 0) continue;

        // Strategy: replace some atoms of the majority element with the missing element(s)
        const variantPositions = createSubstitutionVariant(
          mpStruct.atomicPositions, el1, el2, missingEls, counts
        );

        if (variantPositions.length > 0) {
          // Adjust lattice: scale by Vegard or volume ratio
          const origVol = mpStruct.latticeParams.a * mpStruct.latticeParams.b * mpStruct.latticeParams.c;
          const origAtomsCount = mpStruct.atomicPositions.length;
          const volRatio = totalAtoms / origAtomsCount;
          const scaledA = mpStruct.latticeParams.a * Math.cbrt(volRatio);

          // Blend with Vegard if available
          const latticeA = vegard && vegard.confidence > 0.3
            ? scaledA * 0.5 + vegard.latticeA * 0.5
            : scaledA;

          candidates.push({
            latticeA,
            latticeB: mpStruct.latticeParams.b * Math.cbrt(volRatio),
            latticeC: mpStruct.latticeParams.c * Math.cbrt(volRatio),
            cOverA: mpStruct.latticeParams.a > 0 ? mpStruct.latticeParams.c / mpStruct.latticeParams.a : 1.0,
            positions: variantPositions,
            prototype: `variant-${stoich}`,
            crystalSystem: guessCrystalSystem(mpStruct.latticeParams),
            spaceGroup: mpStruct.spaceGroup ?? "",
            source: `Related compound variant (${stoich} → ${formula})`,
            confidence: 0.45,
            isMetallic,
          });

          // One good variant per pair is enough
          break;
        }
      } catch {
        continue;
      }
    }
  });

  await Promise.allSettled(mpFetches);
  return candidates;
}

/**
 * Generate dynamic stoichiometries for an element pair based on common
 * binary compound patterns. Goes beyond hardcoded AB, A2B, etc. by also
 * checking less common but physically important ratios.
 */
function generateDynamicStoichiometries(el1: string, el2: string): string[] {
  const sorted = [el1, el2].sort();
  const [a, b] = sorted;

  // Common binary stoichiometries ordered by frequency in ICSD/MP
  return [
    `${a}${b}`,          // 1:1
    `${a}2${b}`,         // 2:1
    `${a}${b}2`,         // 1:2
    `${a}3${b}`,         // 3:1
    `${a}${b}3`,         // 1:3
    `${a}3${b}2`,        // 3:2
    `${a}2${b}3`,        // 2:3
    `${a}5${b}3`,        // 5:3 (e.g., Mn5Si3, Nb5Ge3)
    `${a}2${b}5`,        // 2:5 (e.g., V2O5, Nb2O5)
    `${a}4${b}3`,        // 4:3
    `${a}3${b}5`,        // 3:5
    `${a}${b}4`,         // 1:4
    `${a}4${b}`,         // 4:1
    `${a}5${b}`,         // 5:1 (Laves-related)
    `${a}${b}5`,         // 1:5
  ];
}

/**
 * Create a substitution variant: take a known binary structure and replace
 * some atoms with the missing element(s) to approximate the target composition.
 */
function createSubstitutionVariant(
  origPositions: Array<{ element: string; x: number; y: number; z: number }>,
  el1: string,
  el2: string,
  missingEls: string[],
  targetCounts: Record<string, number>,
): Array<{ element: string; x: number; y: number; z: number }> {
  if (origPositions.length === 0 || missingEls.length === 0) return [];

  const totalTarget = Object.values(targetCounts).reduce((s, n) => s + n, 0);
  const positions = origPositions.map(p => ({ ...p }));

  // Scale positions array to roughly match target atom count
  // If the binary has 2 atoms and we need 6, duplicate the unit cell
  while (positions.length < totalTarget && positions.length < 20) {
    const origLen = origPositions.length;
    const scale = Math.ceil(positions.length / origLen);
    for (const p of origPositions) {
      if (positions.length >= totalTarget) break;
      // Offset duplicated atoms slightly to avoid exact overlap
      positions.push({
        element: p.element,
        x: (p.x + 0.5 / (scale + 1)) % 1.0,
        y: (p.y + 0.3 / (scale + 1)) % 1.0,
        z: (p.z + 0.7 / (scale + 1)) % 1.0,
      });
    }
  }

  // Now substitute: replace atoms to match target composition
  // Sort by the element we need the least of first
  const sortedEls = [...Object.keys(targetCounts)].sort(
    (a, b) => (targetCounts[a] || 0) - (targetCounts[b] || 0)
  );

  const assigned = new Map<string, number>();
  for (const el of sortedEls) assigned.set(el, 0);

  // Assign positions to elements based on target counts
  const result: Array<{ element: string; x: number; y: number; z: number }> = [];
  let posIdx = 0;
  for (const el of sortedEls) {
    const need = Math.round(targetCounts[el] || 0);
    for (let i = 0; i < need && posIdx < positions.length; i++) {
      result.push({
        element: el,
        x: positions[posIdx].x,
        y: positions[posIdx].y,
        z: positions[posIdx].z,
      });
      posIdx++;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Source 6: Local ML structure prediction from DB-stored weights
// ---------------------------------------------------------------------------

/**
 * Use locally-available ML model (XGBoost cache or GNN weights stored in DB)
 * to predict stability and lattice properties for the target formula.
 *
 * The model weights are stored in the Neon DB (gnn_model_checkpoints table)
 * and loaded on the QE worker at startup. This avoids needing the GCP service
 * for structure prediction — the worker runs inference locally.
 *
 * Returns a structure candidate if the ML model predicts the material is
 * stable and provides lattice estimates.
 */
async function generateMLStructureCandidate(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number,
  isMetallic: boolean | null,
): Promise<StructureCandidate | null> {
  // First try: local XGB prediction using DB-stored weights
  // This runs the XGBoost ensemble entirely in TypeScript — no GCP needed.
  try {
    const mlPred = await predictWithLocalXGB(formula, 0);
    if (mlPred && mlPred.tc > 0 && mlPred.stability > 0.3) {
      const latticeA = estimateFromML(elements, counts, mlPred.tc);
      const positions = generateSimplePositions(elements, counts, totalAtoms);
      if (positions.length > 0 && latticeA > 2.5) {
        return {
          latticeA,
          positions,
          prototype: "ML-XGB-local",
          crystalSystem: "cubic",
          spaceGroup: "",
          source: `Local XGB (Tc=${mlPred.tc.toFixed(1)}K, CI95=[${mlPred.tcCI95[0].toFixed(1)},${mlPred.tcCI95[1].toFixed(1)}], stability=${mlPred.stability.toFixed(2)}, ${mlPred.nModels} models)`,
          confidence: 0.30 * mlPred.stability * mlPred.confidence,
          isMetallic: isMetallic ?? (mlPred.tc > 0),
        };
      }
    }
  } catch {
    // Local XGB not available
  }

  // Fallback: try disk-cached XGBoost predictions (from GCP service)
  try {
    const xgbCache = loadXGBCacheEntry(formula);
    if (xgbCache && xgbCache.tc > 0 && xgbCache.stability > 0.3) {
      const latticeA = xgbCache.latticeA ?? estimateFromML(elements, counts, xgbCache.tc);
      const positions = generateSimplePositions(elements, counts, totalAtoms);
      if (positions.length > 0 && latticeA > 2.5) {
        return {
          latticeA,
          positions,
          prototype: "ML-XGB-cached",
          crystalSystem: "cubic",
          spaceGroup: "",
          source: `Cached XGB (Tc=${xgbCache.tc.toFixed(1)}K, stability=${xgbCache.stability.toFixed(2)})`,
          confidence: 0.25 * xgbCache.stability,
          isMetallic: isMetallic ?? (xgbCache.tc > 0),
        };
      }
    }
  } catch {
    // Disk cache not available
  }

  return null;
}

interface XGBCacheEntry {
  tc: number;
  stability: number;
  latticeA?: number;
}

/**
 * Load a cached XGBoost prediction from the local disk cache.
 * The cache is populated by the GCP service and persisted as colab-xgb-cache.json.
 */
function loadXGBCacheEntry(formula: string): XGBCacheEntry | null {
  try {
    const fs = require("fs");
    const cachePath = require("path").resolve("colab-xgb-cache.json");
    if (!fs.existsSync(cachePath)) return null;
    const cache = JSON.parse(fs.readFileSync(cachePath, "utf-8"));
    const entry = cache[formula];
    if (!entry) return null;
    return {
      tc: entry.tc ?? entry.tcMean ?? 0,
      stability: entry.stability ?? (entry.tc > 0 ? 0.5 : 0.1),
      latticeA: entry.latticeA,
    };
  } catch {
    return null;
  }
}

/**
 * Rough lattice estimate from ML-predicted Tc.
 * Higher Tc correlates with stiffer lattice → slightly smaller cell.
 * This is a very rough heuristic, used only as a last-resort estimate.
 */
function estimateFromML(
  elements: string[],
  counts: Record<string, number>,
  predictedTc: number,
): number {
  // Base: sum of covalent radii
  let totalVol = 0;
  for (const el of elements) {
    const n = counts[el] || 1;
    const d = getElementData(el);
    const r = (d?.atomicRadius ?? 130) / 100;
    totalVol += n * (4 / 3) * Math.PI * r * r * r;
  }
  // Higher Tc → stiffer bonds → slightly smaller packing
  // Empirical: pack factor increases ~0.02 per 50K above room temp
  const tcBoost = Math.min(predictedTc / 2500, 0.08);
  const packingFactor = 0.68 + tcBoost;
  return Math.max(Math.cbrt(totalVol / packingFactor), 3.0);
}
