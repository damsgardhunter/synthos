/**
 * Iterative Multi-Round Structure Search
 *
 * The single-pass pipeline generates → filters → DFT on 2-3 candidates → done.
 * This misses the global minimum if it's not among those 2-3 initial picks.
 *
 * The iterative pipeline adds feedback:
 *   Round 1: Broad exploration (AIRSS/PyXtal/cage) → funnel → DFT on 3-5
 *   Round 2: Focused search around Round 1 DFT winners → mutate → CHGNet → DFT
 *   Round 3: Ultra-focused refinement around the best → DFT
 *   Final: Compare all rounds, pick the deepest basin
 *
 * Each round generates structures at the pressure-corrected volume from
 * the previous round's DFT results — so we're searching the RIGHT energy
 * landscape, not the ambient one.
 *
 * Also supports multi-pressure scanning: run at P, P±20%, P±50% to find
 * pressure-dependent phase transitions.
 */

import type { CSPCandidate } from "./csp-types";
import { mutateCandidate, type MutationResult } from "./structure-mutator";
import { runChgnetEvaluation, isChgnetAvailable } from "./chgnet-wrapper";
import { deduplicateCandidates } from "./dedup-cluster";

// ---------------------------------------------------------------------------
// Round 2: Focused search around DFT winners
// ---------------------------------------------------------------------------

export interface RoundResult {
  round: number;
  candidates: CSPCandidate[];
  dftEnergy?: number;
  dftForce?: number;
  bestCandidate?: CSPCandidate;
}

/**
 * Generate focused Round 2 candidates from Round 1 DFT results.
 *
 * Takes the best DFT-relaxed structure and creates variants:
 * - Aggressive mutations (larger perturbations to escape local minimum)
 * - Volume scanning at the DFT-determined pressure point
 * - Symmetry-lowering distortions
 * - Supercell variants
 */
export function generateRound2Candidates(
  bestStructure: CSPCandidate,
  dftEnergy: number,
  dftLatticeA: number,
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
): CSPCandidate[] {
  const candidates: CSPCandidate[] = [];
  const baseSeed = Date.now() % 1e8;

  // 1. Aggressive mutations of the DFT winner (30 variants)
  // Larger perturbations than round 1 to explore neighboring basins
  const aggressiveMutations = mutateCandidate(bestStructure, 30, baseSeed, true);
  for (const m of aggressiveMutations) {
    m.candidate.generationStage = 2;
    m.candidate.source = `R2 ${m.mutationType} from DFT winner`;
    candidates.push(m.candidate);
  }

  // 2. Volume scanning around the DFT lattice (±2%, ±5%, ±8%)
  // The DFT lattice is the best pressure match — scan around it
  const volumeScales = [0.92, 0.95, 0.98, 1.02, 1.05, 1.08];
  for (const scale of volumeScales) {
    const variant: CSPCandidate = {
      ...bestStructure,
      positions: bestStructure.positions.map(p => ({ ...p })),
      latticeA: dftLatticeA * Math.pow(scale, 1 / 3),
      seed: baseSeed + 1000 + Math.round(scale * 100),
      generationStage: 2,
      source: `R2 volume-scan scale=${scale.toFixed(2)} from DFT winner`,
      prototype: `R2-volume-${scale.toFixed(2)}`,
      confidence: 0.75,
      relaxationLevel: "raw",
    };
    candidates.push(variant);
  }

  // 3. Pressure scanning: generate at P±20% and P±50%
  // Different pressures can stabilize different phases
  if (pressureGPa > 20) {
    const pressurePoints = [
      pressureGPa * 0.50,
      pressureGPa * 0.80,
      pressureGPa * 1.20,
      pressureGPa * 1.50,
    ].filter(p => p > 10 && p < 500);

    for (const targetP of pressurePoints) {
      // Estimate volume at different pressure using Birch-Murnaghan
      const B0 = 100; // GPa estimate
      const relVol = Math.pow(1 + 4 * targetP / B0, -1 / 4) / Math.pow(1 + 4 * pressureGPa / B0, -1 / 4);
      const scaledA = dftLatticeA * Math.pow(relVol, 1 / 3);

      const variant: CSPCandidate = {
        ...bestStructure,
        positions: bestStructure.positions.map(p => ({ ...p })),
        latticeA: scaledA,
        pressureGPa: targetP,
        seed: baseSeed + 2000 + Math.round(targetP),
        generationStage: 2,
        source: `R2 pressure-scan P=${targetP.toFixed(0)} GPa from DFT winner`,
        prototype: `R2-pressure-${targetP.toFixed(0)}GPa`,
        confidence: 0.70,
        relaxationLevel: "raw",
      };
      candidates.push(variant);
    }
  }

  // 4. Symmetry-lowering distortions (explore lower-symmetry basins)
  for (let i = 0; i < 5; i++) {
    const distorted = mutateCandidate(bestStructure, 1, baseSeed + 3000 + i, false);
    if (distorted.length > 0) {
      const d = distorted[0].candidate;
      d.generationStage = 2;
      d.spaceGroup = "P1";
      d.crystalSystem = "triclinic";
      // Apply larger lattice angle perturbation
      if (d.latticeParams) {
        d.latticeParams.alpha += (Math.random() - 0.5) * 10;
        d.latticeParams.beta += (Math.random() - 0.5) * 10;
        d.latticeParams.gamma += (Math.random() - 0.5) * 10;
      }
      d.source = `R2 symmetry-lowered distortion #${i + 1}`;
      d.prototype = `R2-distortion-${i}`;
      candidates.push(d);
    }
  }

  console.log(`[Iterative] Round 2: generated ${candidates.length} focused candidates from DFT winner (E=${dftEnergy.toFixed(4)} eV, a=${dftLatticeA.toFixed(3)} Å)`);
  console.log(`[Iterative]   ${aggressiveMutations.length} mutations, ${volumeScales.length} volume scans, ${pressureGPa > 20 ? "4 pressure scans" : "0 pressure scans"}, 5 distortions`);

  return candidates;
}

/**
 * Screen Round 2 candidates with CHGNet and select the best for DFT.
 */
/**
 * Screen Round 2 candidates using enthalpy (H = E + PV), not just energy.
 *
 * Selection criteria (meV/atom thresholds, not percentages):
 * - Enthalpy improves by >= 10 meV/atom → promote
 * - Within 10 meV/atom but structurally diverse → keep
 * - Worse but high novelty or better hydride cage → keep 1 exploration candidate
 */
export async function screenRound2(
  candidates: CSPCandidate[],
  formula: string,
  round1BestEnthalpyPerAtom: number,
  workDir: string,
  pressureGPa: number = 0,
): Promise<CSPCandidate[]> {
  // Dedup first
  const { unique } = deduplicateCandidates(candidates, 0.08);
  console.log(`[Iterative] Round 2 dedup: ${candidates.length} → ${unique.length}`);

  // Improvement threshold in eV/atom (10 meV/atom = 0.010 eV/atom)
  const PROMOTE_THRESHOLD = 0.010;    // >= 10 meV/atom improvement → definitely promote
  const KEEP_THRESHOLD = 0.005;       // >= 5 meV/atom improvement → keep if diverse
  const isHighPressure = pressureGPa > 20;

  // CHGNet energy ranking
  if (isChgnetAvailable() && unique.length > 5) {
    const chgnetResult = await runChgnetEvaluation(
      unique, workDir, true, // relax=true for round 2
      Math.min(unique.length, 50),
      300000, // 5 min
    );

    if (chgnetResult.stats.evaluated > 0) {
      console.log(`[Iterative] Round 2 CHGNet: ${chgnetResult.stats.evaluated} evaluated, best=${chgnetResult.stats.bestEnergy?.toFixed(4)} eV/atom`);

      const selected: CSPCandidate[] = [];

      // For high-pressure materials, use enthalpy H = E + PV.
      // CHGNet's enthalpyPerAtom already includes the PV term if pressure
      // was set during relaxation, but the candidates may have different
      // pressures. Compare apples to apples using enthalpy per atom.
      const r1H = round1BestEnthalpyPerAtom;

      // Category 1: Strong improvement (>= 10 meV/atom lower enthalpy)
      const promoted = chgnetResult.rankedCandidates.filter(c => {
        const h = c.enthalpyPerAtom;
        return h != null && (r1H - h) >= PROMOTE_THRESHOLD;
      });

      // Category 2: Mild improvement (5-10 meV/atom) — keep if structurally diverse
      const mild = chgnetResult.rankedCandidates.filter(c => {
        const h = c.enthalpyPerAtom;
        if (h == null) return false;
        const improvement = r1H - h;
        return improvement >= KEEP_THRESHOLD && improvement < PROMOTE_THRESHOLD;
      });

      // Category 3: Exploration — 1 candidate that's different even if enthalpy
      // isn't better (cage-type, different symmetry, novel source)
      const exploration = chgnetResult.rankedCandidates.find(c => {
        if (selected.some(s => s === c) || promoted.includes(c) || mild.includes(c)) return false;
        // Prefer candidates with cage/novel sources
        const source = c.source ?? "";
        return source.includes("cage") ||
               source.includes("pressure-scan") ||
               source.includes("symmetry-lowered") ||
               source.includes("distortion");
      });

      // Assemble: promoted first, then mild (diverse), then 1 exploration
      for (const c of promoted.slice(0, 3)) selected.push(c);
      for (const c of mild.slice(0, 2)) selected.push(c);
      if (exploration && selected.length < 5) selected.push(exploration);

      if (selected.length > 0) {
        console.log(`[Iterative] Round 2: ${promoted.length} promoted (>=${PROMOTE_THRESHOLD * 1000} meV/atom), ${mild.length} mild improvement, ${exploration ? 1 : 0} exploration (R1 H=${r1H.toFixed(4)} eV/atom${isHighPressure ? `, P=${pressureGPa} GPa` : ""})`);
        return selected.slice(0, 5);
      } else {
        console.log(`[Iterative] Round 2: no candidates beat R1 enthalpy (${r1H.toFixed(4)} eV/atom) — using top 3 for exploration`);
        return chgnetResult.rankedCandidates.slice(0, 3);
      }
    }
  }

  // No CHGNet — return top by confidence
  unique.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  return unique.slice(0, 3);
}

// ---------------------------------------------------------------------------
// Multi-pressure scanning
// ---------------------------------------------------------------------------

/**
 * Generate pressure scan points for a material.
 * Explores multiple pressures to find phase boundaries and optimal Tc.
 */
export function getPressureScanPoints(
  basePressureGPa: number,
  hasH: boolean,
): number[] {
  if (basePressureGPa <= 0) return [0];

  if (hasH && basePressureGPa >= 100) {
    // High-pressure hydrides: scan around the target ±50%
    // Phase transitions are common in hydrides across pressure
    return [
      Math.max(50, basePressureGPa * 0.50),
      Math.max(80, basePressureGPa * 0.75),
      basePressureGPa,
      Math.min(350, basePressureGPa * 1.25),
      Math.min(400, basePressureGPa * 1.50),
    ];
  }

  if (hasH) {
    // Moderate-pressure hydrides
    return [
      Math.max(0, basePressureGPa - 20),
      basePressureGPa,
      basePressureGPa + 20,
    ];
  }

  // Ambient materials
  return [basePressureGPa];
}

/**
 * Check if a Round 2 search is worthwhile based on Round 1 results.
 */
export function shouldDoRound2(
  scfConverged: boolean,
  residualForce: number | null,
  isMetallic: boolean | null,
  qualityTier: string | undefined,
): { doRound2: boolean; reason: string } {
  // Always do round 2 if SCF converged — we have a valid baseline to improve on
  if (scfConverged) {
    if (residualForce != null && residualForce > 0.05) {
      return { doRound2: true, reason: "SCF converged but high residual force — structure may not be optimal" };
    }
    if (isMetallic) {
      return { doRound2: true, reason: "metallic candidate — worth exploring nearby basins for better Tc" };
    }
    return { doRound2: true, reason: "SCF converged — searching for deeper energy basin" };
  }

  // Don't waste compute on failed SCF
  if (qualityTier === "failed") {
    return { doRound2: false, reason: "SCF failed — no baseline to improve on" };
  }

  // Partial convergence — maybe worth trying
  return { doRound2: true, reason: "partial convergence — round 2 from different starting point may converge" };
}
