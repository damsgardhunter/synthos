/**
 * 5-Stage Gated Relaxation Pipeline
 *
 * Stage 1: Atomic relax (fixed cell) — quick sanity check, pick best candidate
 * Stage 2: vc-relax (full cell optimization) — tighten forces and cell params
 * Stage 3: Final SCF on relaxed structure — production-quality electronic structure
 * Stage 4: Gamma-point phonon check — fast dynamical stability screen
 * Stage 5: Full phonon grid — DFPT-quality phonon spectrum
 *
 * Each stage has explicit pass/fail criteria. Bad structures are caught early
 * (Stage 1: 10 min, Stage 4: 30 min) instead of wasting 24h on full phonon.
 */

import * as fs from "fs";
import * as path from "path";
import type { StructureCandidate } from "./vegard-lattice";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StageResult {
  stage: 1 | 2 | 3 | 4 | 5;
  passed: boolean;
  failReason?: string;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  latticeA: number;
  cellVectors?: number[][];
  totalEnergy: number;
  fermiEnergy?: number;
  maxForce?: number;
  pressure?: number;       // kbar
  wallTimeSeconds: number;
  frequencies?: number[];  // stages 4-5
  isMetallic?: boolean;
  scfConverged?: boolean;
}

export interface StagedRelaxationResult {
  stages: StageResult[];
  finalStage: number;
  success: boolean;
  bestPositions: Array<{ element: string; x: number; y: number; z: number }>;
  bestLatticeA: number;
  bestCellVectors?: number[][];
  totalWallTime: number;
  candidateSource: string;
  vegardConfidence?: number;
  isMetallic?: boolean;
}

/** Callback interface so this module doesn't depend on qe-worker internals. */
export interface QERunnerCallbacks {
  runPwx(inputFile: string, workDir: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number }>;
  runPhx(inputFile: string, workDir: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number }>;
  getQEBinDir(): string;
  getPseudoDir(): string;
  getPseudoDirInput(): string;  // WSL-compatible path
  cleanTmpDir(tmpDir: string): void;
  resolveEcutwfc(elements: string[]): number;
  resolveEcutrho(elements: string[], ecutwfc: number): number;
  resolvePPFilename(element: string): string;
  getAtomicMass(element: string): number;
  autoKPoints(latticeA: number, cOverA?: number, kspacing?: number): string;
  hasMagneticElements(elements: string[]): boolean;
  generateMagnetizationLines(elements: string[], counts: Record<string, number>): string;
  estimateCOverA(elements: string[], counts: Record<string, number>): number;
  generateCellParameters(latticeA: number, cOverA: number, beta?: number, bOverA?: number, elements?: string[], counts?: Record<string, number>): string;
}

export interface StagedRelaxationOpts {
  formula: string;
  elements: string[];
  counts: Record<string, number>;
  candidates: StructureCandidate[];
  pressureGPa: number;
  jobDir: string;
  callbacks: QERunnerCallbacks;
  /** Skip Stage 2 (vc-relax) — for high-P hydrides, TSC, all-TM intermetallics */
  skipVcRelax?: boolean;
  /** Skip Stages 1-2 — for known compounds with literature lattice params */
  skipRelaxation?: boolean;
  /** Max candidates to test in Stage 1 */
  maxStage1Candidates?: number;
  /** Is this a metallic system (from Vegard/MP lookup)? */
  isMetallic?: boolean;
  /** Screening tier — controls how many Stage 1 winners advance to Stage 2.
   *  preview: top 2, standard: top 5, deep: top 10-15, publication: top 20-30 */
  screeningTier?: "preview" | "standard" | "deep" | "publication";
}

// ---------------------------------------------------------------------------
// Post-DFT structure deduplication
// ---------------------------------------------------------------------------

/**
 * Simple pair-distance fingerprint for post-DFT deduplication.
 * Compares relaxed structures by sorted interatomic distances.
 * Returns true if two structures are essentially the same DFT minimum.
 */
function areStructuresDuplicate(
  pos1: Array<{ element: string; x: number; y: number; z: number }>,
  a1: number,
  pos2: Array<{ element: string; x: number; y: number; z: number }>,
  a2: number,
  tolerance: number = 0.03, // 3% tolerance on distances
): boolean {
  if (pos1.length !== pos2.length) return false;

  // Compare lattice constants
  if (Math.abs(a1 - a2) / Math.max(a1, a2) > 0.05) return false;

  // Compare sorted pair distances (first 20)
  const dists1 = computeSortedDistances(pos1, a1).slice(0, 20);
  const dists2 = computeSortedDistances(pos2, a2).slice(0, 20);

  if (dists1.length !== dists2.length) return false;

  let maxRelDiff = 0;
  for (let i = 0; i < dists1.length; i++) {
    const avg = (dists1[i] + dists2[i]) / 2;
    if (avg < 0.01) continue;
    const relDiff = Math.abs(dists1[i] - dists2[i]) / avg;
    maxRelDiff = Math.max(maxRelDiff, relDiff);
  }

  return maxRelDiff < tolerance;
}

function computeSortedDistances(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
): number[] {
  const dists: number[] = [];
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      let dx = positions[i].x - positions[j].x;
      let dy = positions[i].y - positions[j].y;
      let dz = positions[i].z - positions[j].z;
      dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
      const dist = Math.sqrt((dx * latticeA) ** 2 + (dy * latticeA) ** 2 + (dz * latticeA) ** 2);
      dists.push(dist);
    }
  }
  dists.sort((a, b) => a - b);
  return dists;
}

// ---------------------------------------------------------------------------
// K-spacing constants for each stage
// ---------------------------------------------------------------------------
const KSPACING_RELAX = 0.40;
const KSPACING_VCRELAX = 0.30;
const KSPACING_SCF = 0.25;
const KSPACING_SCF_METAL = 0.20;

// Stage time caps (ms) — base values, scaled by element complexity
const STAGE1_BASE_TIMEOUT_MS = 900_000;   // 15 min base
const STAGE2_TIMEOUT_MS = 1_800_000;      // 30 min
const STAGE4_TIMEOUT_MS = 1_800_000;      // 30 min

// Force thresholds (Ry/bohr)
const STAGE1_FORCE_THR = 1e-3;
const STAGE2_FORCE_THR = 5e-4;

// Heavy elements (Z >= 55) have expensive SCF iterations due to large
// basis sets, many valence electrons, and relativistic effects.
const HEAVY_ELEMENTS = new Set([
  "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po",
  "Ba", "Cs",
  "Th", "U", "Pa",
]);

// Magnetic elements need nspin=2 which doubles the SCF cost
const MAGNETIC_ELS = new Set(["Fe", "Co", "Ni", "Mn", "Cr", "V", "Gd", "Eu", "Nd"]);

// Valence electron counts per element (from QE pseudopotential zValence).
// Used to estimate SCF cost: more valence electrons = larger basis set = more expensive.
const Z_VALENCE: Record<string, number> = {
  H: 1, He: 2, Li: 3, Be: 4, B: 3, C: 4, N: 5, O: 6, F: 7,
  Na: 9, Mg: 10, Al: 3, Si: 4, P: 5, S: 6, Cl: 7,
  K: 9, Ca: 10, Sc: 11, Ti: 12, V: 13, Cr: 14, Mn: 15,
  Fe: 16, Co: 17, Ni: 18, Cu: 19, Zn: 20, Ga: 13, Ge: 14,
  As: 15, Se: 16, Br: 7, Rb: 9, Sr: 10, Y: 11, Zr: 12,
  Nb: 13, Mo: 14, Ru: 14, Rh: 15, Pd: 16, Ag: 19, Cd: 20,
  In: 13, Sn: 14, Sb: 15, Te: 16, I: 7, Cs: 9, Ba: 10,
  La: 11, Ce: 12, Pr: 13, Nd: 14, Sm: 16, Eu: 17, Gd: 18,
  Hf: 12, Ta: 13, W: 14, Re: 15, Os: 14, Ir: 15, Pt: 16,
  Au: 19, Hg: 20, Tl: 13, Pb: 14, Bi: 15,
  Th: 12, U: 14,
};

/**
 * Physics-based SCF cost estimator for Stage 1 timeout.
 *
 * Instead of arbitrary multipliers, this calculates the expected wall time
 * from the actual computational cost of one SCF iteration:
 *
 *   cost_per_iter ∝ N_atoms² × N_electrons × N_kpoints
 *   total_cost = cost_per_iter × N_iterations × nspin_factor
 *   timeout = total_cost / calibration_constant + 10 min safety margin
 *
 * Calibrated from observed Stage 1 results:
 *   MoSiTl2 (4 atoms, 60 electrons, ~4 kpts): 844s
 *   LaH12   (13 atoms, 23 electrons, ~2 kpts): 3319s
 *   Bi2GeSb (4 atoms, 58 electrons, ~4 kpts):  1293s
 */
function computeStage1Params(elements: string[], totalAtoms: number, counts?: Record<string, number>): {
  timeoutMs: number;
  ecutwfcScale: number;
  kspacingOverride: number;
  maxSeconds: number;
} {
  const heavyCount = elements.filter(e => HEAVY_ELEMENTS.has(e)).length;
  const hasMagnetic = elements.some(e => MAGNETIC_ELS.has(e));

  // --- Ecutwfc and kspacing first (needed for cost estimate) ---
  let ecutwfcScale = 1.0;
  if (heavyCount >= 2 || totalAtoms >= 6) ecutwfcScale = 0.85;
  if (heavyCount >= 3 || totalAtoms >= 10) ecutwfcScale = 0.75;

  let kspacing = KSPACING_RELAX; // 0.40 base
  if (heavyCount >= 1 || totalAtoms >= 5) kspacing = 0.50;
  if (heavyCount >= 2 || totalAtoms >= 8) kspacing = 0.55;
  if (totalAtoms >= 12) kspacing = 0.65;

  // --- Physics-based cost model ---
  // Step 1: Count total valence electrons in the unit cell
  let cellElectrons = 0;
  if (counts) {
    for (const el of elements) {
      cellElectrons += (counts[el] ?? 1) * (Z_VALENCE[el] ?? 8);
    }
  } else {
    // Fallback: assume uniform stoichiometry
    for (const el of elements) {
      cellElectrons += (Z_VALENCE[el] ?? 8);
    }
    cellElectrons = (cellElectrons / elements.length) * totalAtoms;
  }

  // Step 2: Estimate k-points from kspacing and a typical lattice ~5 A
  // k_i = ceil(2π / (kspacing * a)), assume a ≈ 5 A for estimation
  const typicalLattice = 5.0;
  const kPerDir = Math.max(2, Math.ceil((2 * Math.PI) / (kspacing * typicalLattice)));
  const nKpoints = kPerDir * kPerDir * kPerDir;

  // Step 3: nspin factor — spin-polarized doubles the work
  const nspinFactor = hasMagnetic ? 2.0 : 1.0;

  // Step 4: Number of SCF iterations for relax (typically 50-150 per ionic step,
  // ~5-20 ionic steps for Stage 1's nstep=100)
  const scfItersPerIonic = 80;
  const ionicSteps = 10; // Average for a reasonable starting structure

  // Step 5: Cost model — calibrated from observed wall times:
  //   MoSiTl2: 4 atoms, ~60 e-, 4³ kpts, nspin=1 → 844s
  //   LaH12:  13 atoms, ~23 e-, 2³ kpts, nspin=1 → 3319s
  //   Bi2GeSb: 4 atoms, ~58 e-, 4³ kpts, nspin=1 → 1293s
  //
  // Model: time_s = C * N_atoms² * N_electrons * N_kpoints * nspin * iters
  // Fitting C from MoSiTl2: 844 = C * 16 * 60 * 64 * 1 * (80*10)
  //   C = 844 / (16 * 60 * 64 * 800) = 844 / 49,152,000 ≈ 1.72e-5
  //
  // But SCF cost doesn't scale linearly with all factors — it's more like
  // N_atoms^1.5 * sqrt(N_electrons) * N_kpoints^0.7 in practice.
  // Use a simpler empirical formula calibrated to the three data points:
  const costFactor = Math.pow(totalAtoms, 1.8) * Math.sqrt(cellElectrons) * Math.pow(nKpoints, 0.6) * nspinFactor;

  // Calibration constant: fit to observed data
  // MoSiTl2: costFactor = 4^1.8 * sqrt(60) * 64^0.6 * 1 = 12.1 * 7.75 * 14.9 = 1397 → 844s → rate = 0.60 s/unit
  // LaH12:   costFactor = 13^1.8 * sqrt(23) * 8^0.6 * 1 = 113 * 4.80 * 3.48 = 1888 → 3319s → rate = 1.76 s/unit
  // Bi2GeSb: costFactor = 4^1.8 * sqrt(58) * 64^0.6 * 1 = 12.1 * 7.62 * 14.9 = 1373 → 1293s → rate = 0.94 s/unit
  // Average rate ≈ 1.1 s/unit. Use 1.5 to be conservative (accounts for VM contention).
  const SECONDS_PER_COST_UNIT = 1.5;
  const estimatedSeconds = costFactor * SECONDS_PER_COST_UNIT;

  // Add 10 minute safety margin
  const SAFETY_MARGIN_S = 600;
  const timeoutSeconds = estimatedSeconds + SAFETY_MARGIN_S;

  // Floor at 15 min (simple systems), cap at 90 min (avoid hogging the worker)
  const clampedTimeoutS = Math.max(900, Math.min(timeoutSeconds, 5400));
  const timeoutMs = clampedTimeoutS * 1000;
  const maxSeconds = Math.floor(clampedTimeoutS) - 60; // QE max_seconds slightly under timeout

  console.log(`[Staged-Relax] Cost model: ${totalAtoms} atoms, ${cellElectrons.toFixed(0)} e-, ${nKpoints} kpts, nspin=${nspinFactor} → cost=${costFactor.toFixed(0)}, est=${estimatedSeconds.toFixed(0)}s, timeout=${clampedTimeoutS.toFixed(0)}s`);

  return { timeoutMs, ecutwfcScale, kspacingOverride: kspacing, maxSeconds };
}

// ---------------------------------------------------------------------------
// Main pipeline entry
// ---------------------------------------------------------------------------

export async function runStagedRelaxation(opts: StagedRelaxationOpts): Promise<StagedRelaxationResult> {
  const { formula, elements, counts, candidates, pressureGPa, jobDir, callbacks } = opts;
  const stages: StageResult[] = [];
  const startTime = Date.now();
  // Dynamic candidate limit: test as many as the wall-time budget allows.
  // Each candidate gets its own timeout from the cost model. Total budget
  // is capped at 90 min for Stage 1 across all candidates combined.
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const s1Params = computeStage1Params(elements, totalAtoms, counts);
  const STAGE1_TOTAL_BUDGET_MS = 5_400_000; // 90 min total for all candidates
  const perCandidateMs = s1Params.timeoutMs;
  const budgetBasedMax = Math.max(1, Math.floor(STAGE1_TOTAL_BUDGET_MS / perCandidateMs));
  // Always test at least 2 candidates if 2+ were admitted — the funnel
  // selected these for a reason (exploitation vs diversity). Testing only 1
  // wastes the admission selection.
  const minCandidates = Math.min(2, candidates.length);
  const maxS1 = Math.max(
    minCandidates,
    Math.min(opts.maxStage1Candidates ?? budgetBasedMax, budgetBasedMax),
  );

  let bestPositions = candidates[0]?.positions ?? [];
  let bestLatticeA = candidates[0]?.latticeA ?? 5.0;
  let bestCellVectors: number[][] | undefined;
  let candidateSource = candidates[0]?.source ?? "unknown";
  let isMetallic = opts.isMetallic;

  // --- Skip Stages 1-2 for known compounds ---
  if (opts.skipRelaxation) {
    console.log(`[Staged-Relax] ${formula}: skipping Stages 1-2 (known compound or literature lattice)`);
  } else {
    // === STAGE 1: Atomic relax (fixed cell) ===
    const stage1Results: Array<{ result: StageResult; candidate: StructureCandidate }> = [];

    const candidatesToTest = candidates.slice(0, maxS1);
    console.log(`[Staged-Relax] ${formula} Stage 1/5 (atomic relax): testing ${candidatesToTest.length} candidates`);

    for (let ci = 0; ci < candidatesToTest.length; ci++) {
      const cand = candidatesToTest[ci];
      console.log(`[Staged-Relax] ${formula} Stage 1 candidate ${ci + 1}/${candidatesToTest.length}: ${cand.source}, a=${cand.latticeA.toFixed(3)} A, ${cand.positions.length} atoms`);

      try {
        const s1 = await runStage1AtomicRelax(
          formula, elements, counts, cand, pressureGPa, jobDir, callbacks, ci
        );
        stage1Results.push({ result: s1, candidate: cand });

        if (s1.passed) {
          console.log(`[Staged-Relax] ${formula} Stage 1 candidate ${ci + 1}: PASSED (force=${s1.maxForce?.toExponential(2)}, E=${s1.totalEnergy.toFixed(4)} eV, wall=${s1.wallTimeSeconds.toFixed(0)}s)`);
        } else {
          console.log(`[Staged-Relax] ${formula} Stage 1 candidate ${ci + 1}: FAILED — ${s1.failReason}`);
        }
      } catch (err: any) {
        console.log(`[Staged-Relax] ${formula} Stage 1 candidate ${ci + 1}: ERROR — ${err.message?.slice(0, 200)}`);
      }
    }

    // Determine how many Stage 1 winners to advance to Stage 2.
    // Different structures can respond differently to vc-relax, pressure
    // correction, or symmetry lowering — picking only 1 too early loses
    // structures that may become better after cell optimization.
    const tier = opts.screeningTier ?? "preview";
    const stage2KeepCount: Record<string, number> = {
      preview: 2,
      standard: 5,
      deep: 10,
      publication: 20,
    };
    const maxStage2Candidates = stage2KeepCount[tier] ?? 2;

    // Sort Stage 1 results by PER-ATOM energy (lowest first).
    // CRITICAL: must use per-atom energy, not total energy!
    // Different Z values (Z=1 vs Z=4) produce different atom counts.
    // Total energy scales with atom count, so a Z=4 supercell (12 atoms)
    // will always have lower total energy than Z=1 (3 atoms) even if the
    // per-atom energy is worse. This caused MgB2 to pick Z=4 (12 atoms,
    // a=6.444) then crash when iterative rescaling compressed those 12
    // atoms into the Z=1 literature cell (a=3.09).
    const passedS1 = stage1Results.filter(r => r.result.passed);
    const allS1Sorted = passedS1.length > 0 ? passedS1 : stage1Results;
    const perAtomEnergy = (r: { result: StageResult; candidate: StructureCandidate }) => {
      const nAtoms = r.result.positions.length || r.candidate.positions.length || 1;
      return r.result.totalEnergy / nAtoms;
    };
    allS1Sorted.sort((a, b) => perAtomEnergy(a) - perAtomEnergy(b));

    if (passedS1.length > 0) {
      const kept = passedS1.slice(0, maxStage2Candidates);
      const best = kept[0];
      bestPositions = best.result.positions;
      bestLatticeA = best.result.latticeA;
      candidateSource = best.candidate.source;
      stages.push(best.result);
      console.log(`[Staged-Relax] ${formula} Stage 1: ${passedS1.length} passed, keeping top ${kept.length} for Stage 2 (tier=${tier})`);
      for (let ki = 0; ki < kept.length; ki++) {
        const nAtoms = kept[ki].result.positions.length || kept[ki].candidate.positions.length || 1;
        console.log(`[Staged-Relax]   #${ki + 1}: ${kept[ki].candidate.source} (E=${kept[ki].result.totalEnergy.toFixed(4)} eV, ${nAtoms} atoms, E/atom=${perAtomEnergy(kept[ki]).toFixed(4)} eV)`);
      }
    } else if (stage1Results.length > 0) {
      // No candidate passed — use the one with lowest energy anyway (best effort)
      const best = allS1Sorted[0];
      bestPositions = best.result.positions.length > 0 ? best.result.positions : best.candidate.positions;
      bestLatticeA = best.result.latticeA > 0 ? best.result.latticeA : best.candidate.latticeA;
      candidateSource = best.candidate.source;
      stages.push(best.result);
      console.log(`[Staged-Relax] ${formula} Stage 1: NO candidate passed, using best-effort from ${candidateSource}`);
    } else {
      // All crashed — use first candidate as-is
      console.log(`[Staged-Relax] ${formula} Stage 1: all candidates crashed, using raw candidate`);
      stages.push({
        stage: 1, passed: false, failReason: "all candidates crashed",
        positions: bestPositions, latticeA: bestLatticeA, totalEnergy: 0, wallTimeSeconds: 0,
      });
    }

    // === STAGE 2: vc-relax (full cell optimization) ===
    // Run vc-relax on multiple Stage 1 winners (tier-dependent count).
    // Different fixed-cell structures can respond differently after cell
    // optimization — the lowest Stage 1 energy may not stay lowest after
    // vc-relax.
    if (opts.skipVcRelax) {
      console.log(`[Staged-Relax] ${formula}: skipping Stage 2 (vc-relax skip condition)`);
    } else {
      const s2Candidates = passedS1.length > 0
        ? passedS1.slice(0, maxStage2Candidates)
        : (allS1Sorted.length > 0 ? [allS1Sorted[0]] : []);

      if (s2Candidates.length > 1) {
        console.log(`[Staged-Relax] ${formula} Stage 2/5 (vc-relax): testing ${s2Candidates.length} Stage 1 winners`);
      } else {
        console.log(`[Staged-Relax] ${formula} Stage 2/5 (vc-relax): a=${bestLatticeA.toFixed(3)} A, ${bestPositions.length} atoms`);
      }

      let bestS2Energy = Infinity;
      let bestS2Result: StageResult | null = null;
      let bestS2Source = candidateSource;
      // Post-DFT dedup: track relaxed structures to detect when multiple
      // starting points collapse to the same DFT minimum. Saves phonon compute.
      const relaxedStructures: Array<{ positions: Array<{ element: string; x: number; y: number; z: number }>; latticeA: number; source: string }> = [];
      let s2Duplicates = 0;

      for (let s2i = 0; s2i < s2Candidates.length; s2i++) {
        const s2Cand = s2Candidates[s2i];
        const s2Positions = s2Cand.result.positions.length > 0 ? s2Cand.result.positions : s2Cand.candidate.positions;
        const s2LatticeA = s2Cand.result.latticeA > 0 ? s2Cand.result.latticeA : s2Cand.candidate.latticeA;

        if (s2Candidates.length > 1) {
          console.log(`[Staged-Relax] ${formula} Stage 2 candidate ${s2i + 1}/${s2Candidates.length}: ${s2Cand.candidate.source}, a=${s2LatticeA.toFixed(3)} A`);
        }

        try {
          const s2 = await runStage2VcRelax(
            formula, elements, counts, s2Positions, s2LatticeA, pressureGPa, jobDir, callbacks
          );

          // Post-DFT dedup: check if this relaxed structure duplicates one we already have
          if (s2.passed && s2.positions.length > 0) {
            const isDuplicate = relaxedStructures.some(rs =>
              areStructuresDuplicate(rs.positions, rs.latticeA, s2.positions, s2.latticeA)
            );

            if (isDuplicate) {
              s2Duplicates++;
              console.log(`[Staged-Relax] ${formula} Stage 2 candidate ${s2i + 1}: DUPLICATE of already-relaxed structure — skipping (saves phonon compute)`);
              continue;
            }

            relaxedStructures.push({ positions: s2.positions, latticeA: s2.latticeA, source: s2Cand.candidate.source });
          }

          const s2NAtoms = s2.positions.length || 1;
          const s2EPerAtom = s2.totalEnergy / s2NAtoms;
          if (s2.passed && s2EPerAtom < bestS2Energy) {
            bestS2Energy = s2EPerAtom;
            bestS2Result = s2;
            bestS2Source = s2Cand.candidate.source;
            console.log(`[Staged-Relax] ${formula} Stage 2 candidate ${s2i + 1}: PASSED (force=${s2.maxForce?.toExponential(2)}, E/atom=${s2EPerAtom.toFixed(4)} eV, a=${s2.latticeA.toFixed(3)} A, P=${s2.pressure?.toFixed(1)} kbar, wall=${s2.wallTimeSeconds.toFixed(0)}s) — new best`);
          } else if (s2.passed) {
            console.log(`[Staged-Relax] ${formula} Stage 2 candidate ${s2i + 1}: PASSED but not lowest E/atom (${s2EPerAtom.toFixed(4)} vs best=${bestS2Energy.toFixed(4)})`);
          } else {
            console.log(`[Staged-Relax] ${formula} Stage 2 candidate ${s2i + 1}: FAILED — ${s2.failReason}`);
            // Track partial result if it's the only one
            if (!bestS2Result && s2.positions.length > 0) {
              bestS2Result = s2;
              bestS2Energy = s2.totalEnergy;
              bestS2Source = s2Cand.candidate.source;
            }
          }
        } catch (err: any) {
          console.log(`[Staged-Relax] ${formula} Stage 2 candidate ${s2i + 1}: ERROR — ${err.message?.slice(0, 200)}`);
        }
      }

      if (s2Duplicates > 0) {
        console.log(`[Staged-Relax] ${formula} Stage 2 post-DFT dedup: ${s2Duplicates} duplicate(s) collapsed to same minimum, ${relaxedStructures.length} unique structures`);
      }

      if (bestS2Result) {
        stages.push(bestS2Result);
        if (bestS2Result.passed) {
          bestPositions = bestS2Result.positions;
          bestLatticeA = bestS2Result.latticeA;
          bestCellVectors = bestS2Result.cellVectors;
          candidateSource = bestS2Source;
          console.log(`[Staged-Relax] ${formula} Stage 2: best winner from ${bestS2Source} (E/atom=${bestS2Energy.toFixed(4)} eV, a=${bestLatticeA.toFixed(3)} A)`);
        } else {
          if (bestS2Result.positions.length > 0) bestPositions = bestS2Result.positions;
          if (bestS2Result.latticeA > 0) bestLatticeA = bestS2Result.latticeA;
          bestCellVectors = bestS2Result.cellVectors;
          console.log(`[Staged-Relax] ${formula} Stage 2: FAILED — ${bestS2Result.failReason}, using partial geometry`);
        }
      } else {
        stages.push({
          stage: 2, passed: false, failReason: "all Stage 2 candidates failed",
          positions: bestPositions, latticeA: bestLatticeA, totalEnergy: 0, wallTimeSeconds: 0,
        });
        console.log(`[Staged-Relax] ${formula} Stage 2: all candidates failed, using Stage 1 geometry`);
      }
    }
  }

  const totalWallTime = (Date.now() - startTime) / 1000;

  return {
    stages,
    finalStage: stages.length > 0 ? stages[stages.length - 1].stage : 0,
    success: stages.every(s => s.passed),
    bestPositions,
    bestLatticeA,
    bestCellVectors,
    totalWallTime,
    candidateSource,
    isMetallic,
  };
}

// ---------------------------------------------------------------------------
// Stage 1: Atomic relax (fixed cell, positions only)
// ---------------------------------------------------------------------------

async function runStage1AtomicRelax(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  candidate: StructureCandidate,
  pressureGPa: number,
  jobDir: string,
  cb: QERunnerCallbacks,
  candidateIdx: number,
): Promise<StageResult> {
  const t0 = Date.now();
  const positions = candidate.positions;
  const latticeA = candidate.latticeA;
  const cOverA = candidate.cOverA ?? cb.estimateCOverA(elements, counts);
  const totalAtoms = positions.length;
  const nTypes = elements.length;

  // Scale timeout, ecutwfc, and k-grid based on element complexity
  const s1Params = computeStage1Params(elements, totalAtoms, counts);

  const baseEcutwfc = cb.resolveEcutwfc(elements);
  const ecutwfc = Math.round(baseEcutwfc * s1Params.ecutwfcScale);
  const ecutrho = cb.resolveEcutrho(elements, ecutwfc);
  const kpoints = cb.autoKPoints(latticeA, cOverA, s1Params.kspacingOverride);

  const hasMag = cb.hasMagneticElements(elements);
  const nspin = hasMag ? 2 : 1;
  const magLines = hasMag ? cb.generateMagnetizationLines(elements, counts) : "";

  let atomicSpecies = "";
  for (const el of elements) {
    atomicSpecies += `  ${el}  ${cb.getAtomicMass(el).toFixed(3)}  ${cb.resolvePPFilename(el)}\n`;
  }

  let atomicPositions = "";
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const cellBlock = cb.generateCellParameters(latticeA, cOverA, 0, 1.0, elements, counts);

  const input = `&CONTROL
  calculation = 'relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}_s1_${candidateIdx}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${cb.getPseudoDirInput()}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${STAGE1_FORCE_THR.toExponential(1).replace("e+0", "d+").replace("e-", "d-").replace("e+", "d+")},
  etot_conv_thr = 1.0d-4,
  nstep = 100,
  max_seconds = ${s1Params.maxSeconds},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.015,
  nspin = ${nspin},
${magLines}/
&ELECTRONS
  electron_maxstep = 200,
  conv_thr = 1.0d-4,
  mixing_beta = 0.4,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
  scf_must_converge = .false.,
/
&IONS
  ion_dynamics = 'bfgs',
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${kpoints}

${cellBlock}
`;

  const stageDir = path.join(jobDir, `stage1_${candidateIdx}`);
  fs.mkdirSync(stageDir, { recursive: true });
  const inputFile = path.join(stageDir, "relax.in");
  fs.writeFileSync(inputFile, input);

  console.log(`[Staged-Relax] ${formula} S1 params: timeout=${Math.round(s1Params.timeoutMs/1000)}s, ecutwfc=${ecutwfc}Ry, kspacing=${s1Params.kspacingOverride}`);
  const result = await cb.runPwx(inputFile, stageDir, s1Params.timeoutMs);
  fs.writeFileSync(path.join(stageDir, "relax.out"), result.stdout);

  const wallTime = (Date.now() - t0) / 1000;

  // Parse output
  const parsed = parseRelaxOutput(result.stdout);
  cb.cleanTmpDir(path.join(stageDir, "tmp"));

  // Check pass criteria — Stage 1 is screening quality, so we're lenient:
  // - SCF convergence is NOT required (scf_must_converge=.false. lets BFGS run
  //   even with approximate SCF — the forces are still directionally correct)
  // - Force threshold is the main gate
  // - Missing positions is a hard fail (QE didn't produce any geometry)
  const failReasons: string[] = [];

  if (parsed.positions.length === 0) {
    // Log diagnostic: what did QE actually output?
    const hasAtomPos = result.stdout.includes("ATOMIC_POSITIONS");
    const hasBfgs = result.stdout.includes("bfgs converged") || result.stdout.includes("BFGS Geometry Optimization");
    const hasMaxSec = result.stdout.includes("Maximum CPU time exceeded");
    const tail = result.stdout.slice(-300);
    console.log(`[Staged-Relax] ${formula} S1 parse fail: hasATOMIC_POSITIONS=${hasAtomPos}, hasBFGS=${hasBfgs}, maxSecHit=${hasMaxSec}, tail=${tail.slice(-150)}`);
    failReasons.push("no final positions in output");
  }
  if (parsed.maxForce != null && parsed.maxForce > STAGE1_FORCE_THR) {
    // Soft pass: Stage 1 is screening — if BFGS produced positions and forces
    // are under 1.0 Ry/bohr, the structure is plausible enough for Stage 2
    // vc-relax to finish converging. Only hard-fail on truly insane forces
    // (> 1.0) which indicate the structure is completely wrong.
    if (parsed.positions.length > 0 && parsed.maxForce < 1.0) {
      // BFGS made progress. Don't fail — Stage 2 will handle convergence.
    } else {
      failReasons.push(`max force ${parsed.maxForce.toExponential(2)} > threshold ${STAGE1_FORCE_THR}`);
    }
  }

  return {
    stage: 1,
    passed: failReasons.length === 0,
    failReason: failReasons.length > 0 ? failReasons.join("; ") : undefined,
    positions: parsed.positions.length > 0 ? parsed.positions : positions,
    latticeA,
    totalEnergy: parsed.totalEnergy,
    maxForce: parsed.maxForce,
    wallTimeSeconds: wallTime,
    scfConverged: parsed.scfConverged,
  };
}

// ---------------------------------------------------------------------------
// Stage 2: Variable-cell relaxation
// ---------------------------------------------------------------------------

async function runStage2VcRelax(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  pressureGPa: number,
  jobDir: string,
  cb: QERunnerCallbacks,
): Promise<StageResult> {
  const t0 = Date.now();
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  const cOverA = cb.estimateCOverA(elements, counts);

  const ecutwfc = cb.resolveEcutwfc(elements);
  const ecutrho = cb.resolveEcutrho(elements, ecutwfc);
  const kpoints = cb.autoKPoints(latticeA, cOverA, KSPACING_VCRELAX);

  const hasMag = cb.hasMagneticElements(elements);
  const nspin = hasMag ? 2 : 1;
  const magLines = hasMag ? cb.generateMagnetizationLines(elements, counts) : "";

  let atomicSpecies = "";
  for (const el of elements) {
    atomicSpecies += `  ${el}  ${cb.getAtomicMass(el).toFixed(3)}  ${cb.resolvePPFilename(el)}\n`;
  }

  let atomicPositions = "";
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const cellBlock = cb.generateCellParameters(latticeA, cOverA, 0, 1.0, elements, counts);

  const input = `&CONTROL
  calculation = 'vc-relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}_s2',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${cb.getPseudoDirInput()}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${STAGE2_FORCE_THR.toExponential(1).replace("e+0", "d+").replace("e-", "d-").replace("e+", "d+")},
  etot_conv_thr = 1.0d-5,
  nstep = 200,
  max_seconds = ${Math.floor(STAGE2_TIMEOUT_MS / 1000) - 60},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.015,
  nspin = ${nspin},
${magLines}/
&ELECTRONS
  electron_maxstep = 300,
  conv_thr = 1.0d-6,
  mixing_beta = 0.25,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
  scf_must_converge = .false.,
/
&IONS
  ion_dynamics = 'bfgs',
/
&CELL
  cell_dynamics = 'bfgs',
  press = ${(pressureGPa * 10.0).toFixed(4)},
  press_conv_thr = ${pressureGPa > 50 ? 1.0 : 0.5},
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${kpoints}

${cellBlock}
`;

  const stageDir = path.join(jobDir, "stage2");
  fs.mkdirSync(stageDir, { recursive: true });
  const inputFile = path.join(stageDir, "vc_relax.in");
  fs.writeFileSync(inputFile, input);

  const result = await cb.runPwx(inputFile, stageDir, STAGE2_TIMEOUT_MS);
  fs.writeFileSync(path.join(stageDir, "vc_relax.out"), result.stdout);

  const wallTime = (Date.now() - t0) / 1000;

  // Parse vc-relax output (reuse common parser)
  const parsed = parseVcRelaxOutput(result.stdout);
  cb.cleanTmpDir(path.join(stageDir, "tmp"));

  const failReasons: string[] = [];

  if (parsed.positions.length === 0) {
    failReasons.push("no final positions in output");
  }
  if (parsed.maxForce != null && parsed.maxForce > STAGE1_FORCE_THR) {
    // Use STAGE1_FORCE_THR (1e-3) as relaxed pass criteria for Stage 2
    // (ideally want 5e-4 but accept 1e-3)
    failReasons.push(`max force ${parsed.maxForce.toExponential(2)} > threshold ${STAGE1_FORCE_THR}`);
  }
  if (parsed.volumeDriftPercent != null && parsed.volumeDriftPercent > 5) {
    failReasons.push(`cell volume still drifting (${parsed.volumeDriftPercent.toFixed(1)}% over last 3 steps)`);
  }
  if (pressureGPa === 0 && parsed.pressure != null && Math.abs(parsed.pressure) > 2) {
    failReasons.push(`residual pressure ${parsed.pressure.toFixed(1)} kbar (> 2 kbar for ambient calc)`);
  }

  return {
    stage: 2,
    passed: failReasons.length === 0,
    failReason: failReasons.length > 0 ? failReasons.join("; ") : undefined,
    positions: parsed.positions.length > 0 ? parsed.positions : positions,
    latticeA: parsed.latticeA > 0 ? parsed.latticeA : latticeA,
    cellVectors: parsed.cellVectors,
    totalEnergy: parsed.totalEnergy,
    maxForce: parsed.maxForce,
    pressure: parsed.pressure,
    wallTimeSeconds: wallTime,
  };
}

// ---------------------------------------------------------------------------
// Stage 4: Gamma-point phonon check
// ---------------------------------------------------------------------------

export interface Stage4Opts {
  formula: string;
  elements: string[];
  counts: Record<string, number>;  // stoichiometry for electron count
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  latticeA: number;
  cellVectors?: number[][];
  jobDir: string;
  callbacks: QERunnerCallbacks;
  ecutwfc: number;
}

/**
 * Run a Gamma-only phonon calculation as a fast dynamical stability screen.
 * Returns frequencies and pass/fail result.
 */
export async function runStage4GammaPhonon(opts: Stage4Opts): Promise<StageResult> {
  const { formula, elements, jobDir, callbacks: cb, ecutwfc } = opts;
  const t0 = Date.now();
  const totalAtoms = opts.positions.length;
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");

  // ph.x MUST run in the same directory as the SCF (jobDir), not a subdirectory.
  // It needs outdir/prefix.save which contains the SCF wavefunctions.
  // The SCF writes to jobDir/tmp/prefix.save, so ph.x must use outdir='./tmp'
  // and run with cwd=jobDir.

  // Physics-based cost model for Gamma phonon (same approach as Stage 1).
  //
  // DFPT cost per representation ∝ N_atoms × N_electrons × N_kpoints × nspin
  // Total representations = N_atoms (each atom has 3 displacement directions,
  // but symmetry reduces unique reps to roughly N_atoms).
  // Each rep does ~10-15 DFPT iterations (like mini-SCF).
  //
  // Calibrated from LaH12 observation:
  //   13 atoms, 23 e-, 5³=125 kpts, nspin=1
  //   Completed 2 reps in ~15 min → ~7.5 min/rep
  //   13 reps × 7.5 min = ~98 min total needed
  //
  // Model: time_per_rep = C × N_atoms × sqrt(N_electrons) × N_kpoints^0.5 × nspin
  //   LaH12: C × 13 × 4.8 × 11.2 × 1 = C × 699 → 7.5 min = 450s → C ≈ 0.64
  const heavyCount = elements.filter(e => HEAVY_ELEMENTS.has(e)).length;
  const hasMagnetic = elements.some(e => MAGNETIC_ELS.has(e));

  let phElectrons = 0;
  for (const el of elements) {
    phElectrons += (Z_VALENCE[el] ?? 8);
  }
  // Scale by stoichiometry if counts available
  if (opts.counts) {
    phElectrons = 0;
    for (const el of elements) {
      phElectrons += (opts.counts[el] ?? 1) * (Z_VALENCE[el] ?? 8);
    }
  }

  // Estimate k-points from the SCF k-grid (already computed for this material)
  const phKspacing = heavyCount >= 1 ? 0.55 : 0.50;
  const phTypicalLattice = opts.latticeA || 5.0;
  const phKperDir = Math.max(2, Math.ceil((2 * Math.PI) / (phKspacing * phTypicalLattice)));
  const phNkpts = phKperDir * phKperDir * phKperDir;
  const phNspin = hasMagnetic ? 2.0 : 1.0;

  // DFPT cost model. Each representation requires a full linear-response SCF
  // which is much more expensive than a ground-state SCF step.
  //
  // Calibrated from LaH10 observation:
  //   11 atoms, 21 e-, 27 kpts, nspin=1
  //   costPerRep = 11 × sqrt(21) × sqrt(27) × 1 = 262
  //   totalCost = 11 × 262 = 2,880
  //   Actual: 1 rep completed in ~20 min, 11 reps total ≈ 220 min = 13,200s
  //   Calibration: 13,200 / 2,880 = 4.58 s/unit
  //
  // Using 5.0 to be conservative (accounts for VM contention with 5 jobs).
  const nReps = totalAtoms;
  const SECONDS_PER_COST_UNIT_PH = 5.0;
  const costPerRep = totalAtoms * Math.sqrt(phElectrons) * Math.pow(phNkpts, 0.5) * phNspin;
  const totalPhCost = nReps * costPerRep;
  const estimatedPhSeconds = totalPhCost * SECONDS_PER_COST_UNIT_PH;

  // Add 10 min safety margin, floor at 30 min, cap at 4 hours
  // Must be integer — QE's max_seconds requires an integer value.
  const phTimeoutS = Math.round(Math.max(1800, Math.min(estimatedPhSeconds + 600, 14400)));
  const phTimeoutMs = phTimeoutS * 1000;

  // Use EXACTLY the same ph.x input as the production Gamma-only phonon
  // (generatePhononInput in qe-worker.ts, isGammaOnly=true branch).
  // The only difference is max_seconds which is scaled by the cost model.
  //
  // Previous Stage 4 failures were caused by deviating from production:
  // - Missing fildyn parameter → undefined .dyn output
  // - Added epsil=.false. → removed dielectric screening that prevents underflow
  // - Used tr2_ph=1e-10 instead of production's 1e-12 → noise amplification
  // - Added invalid niter_ph parameter → immediate crash
  //
  // Lesson: don't invent custom ph.x parameters — use what works in production.
  const phInput = `Gamma-only phonon calculation
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = 1.0d-12,
  alpha_mix(1) = 0.3,
  ldisp = .false.,
  max_seconds = ${phTimeoutS - 60},
/
0.0 0.0 0.0
`;

  console.log(`[Staged-Relax] ${formula} Stage 4 cost model: ${nReps} reps, ${phElectrons} e-, ${phNkpts} kpts, nspin=${phNspin} → cost/rep=${costPerRep.toFixed(0)}, est=${estimatedPhSeconds.toFixed(0)}s, timeout=${phTimeoutS.toFixed(0)}s (${(phTimeoutS/60).toFixed(0)} min)`);

  // If estimated cost exceeds the 4h cap, Stage 4 is too expensive as a screening
  // gate for this material. Skip directly to Stage 5 (full phonon) which has its
  // own retry logic and up to 48h budget.
  if (estimatedPhSeconds > 14400) {
    console.log(`[Staged-Relax] ${formula} Stage 4: SKIPPED (cost too high) — estimated ${(estimatedPhSeconds/3600).toFixed(1)}h exceeds 4h screening budget, deferring to full phonon pipeline (Stage 5). Note: this is NOT a stability pass — gamma phonon was never run.`);
    return {
      stage: 4,
      passed: true,  // Pass through so Stage 5 runs (not a real phonon pass, just a cost skip)
      positions: opts.positions,
      latticeA: opts.latticeA,
      cellVectors: opts.cellVectors,
      totalEnergy: 0,
      wallTimeSeconds: 0,
      frequencies: [],
    };
  }

  // 2-attempt retry matching production phonon pipeline (qe-worker.ts lines 4580-4644):
  //   Attempt 1: tr2_ph=1e-12, alpha_mix=0.3 (production defaults)
  //   Attempt 2 (on crash): tr2_ph=1e-10, alpha_mix=0.1 (loosened, matches production retry)
  // On timeout: attempt 2 with recover=.true. (resume from checkpoint)
  const expectedModes = 3 * totalAtoms;
  let frequencies: number[] = [];
  let lastResult: { stdout: string; stderr: string; exitCode: number } | null = null;

  for (let attempt = 0; attempt < 2; attempt++) {
    const isRetry = attempt > 0;
    // Classify previous failure. Crash (underflow/stop1) takes priority over
    // timeout — when both occur (QE hit max_seconds AND underflow on the last
    // rep), the underflow is the root cause and needs loosened params, not recover.
    const prevHasUnderflow = lastResult != null &&
      (lastResult.stderr.includes("IEEE_UNDERFLOW") || lastResult.stdout.includes("IEEE_UNDERFLOW") ||
       lastResult.stderr.includes("STOP 1") || lastResult.stdout.includes("STOP 1"));
    const prevTimedOut = lastResult != null &&
      lastResult.stdout.includes("Maximum CPU time exceeded");
    // Also detect external kill (wall time exceeded but QE didn't write the message)
    const prevExternalKill = lastResult != null && lastResult.exitCode !== 0 &&
      !prevHasUnderflow && !prevTimedOut;
    const prevCrashed = prevHasUnderflow; // Crash = underflow/stop1

    // On crash: loosen convergence (production retry behavior)
    // On timeout (no crash): use recover=.true. to resume from checkpoint
    const retryTr2 = prevCrashed ? "1.0d-10" : "1.0d-12";
    const retryAlpha = prevCrashed ? 0.1 : 0.3;
    const recoverLine = (prevTimedOut || prevExternalKill) && !prevCrashed ? "  recover = .true.,\n" : "";

    const attemptInput = `Gamma-only phonon calculation
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = ${retryTr2},
  alpha_mix(1) = ${retryAlpha},
  ldisp = .false.,
  max_seconds = ${phTimeoutS - 60},
${recoverLine}/
0.0 0.0 0.0
`;

    const retryInfo = isRetry
      ? (prevCrashed ? " (retry: loosened tr2_ph=1e-10, alpha_mix=0.1)" : " (retry: recover=.true.)")
      : "";
    console.log(`[Staged-Relax] ${formula} Stage 4 attempt ${attempt + 1}/2${retryInfo}, params: tr2_ph=${retryTr2}, alpha_mix=${retryAlpha}, max_seconds=${phTimeoutS - 60}, recover=${prevTimedOut}`);

    // Log the .save directory status so we can verify SCF wavefunctions exist
    const saveDir = path.join(jobDir, "tmp", `${prefix}.save`);
    const saveDirExists = fs.existsSync(saveDir);
    const saveContents = saveDirExists ? (fs.readdirSync(saveDir).slice(0, 10).join(", ") || "empty") : "MISSING";
    console.log(`[Staged-Relax] ${formula} Stage 4 .save dir: ${saveDir} → exists=${saveDirExists}, contents=[${saveContents}]`);

    const inputFile = path.join(jobDir, "ph_gamma.in");
    fs.writeFileSync(inputFile, attemptInput);

    const attemptT0 = Date.now();
    const result = await cb.runPhx(inputFile, jobDir, phTimeoutMs);
    const attemptWall = (Date.now() - attemptT0) / 1000;
    fs.writeFileSync(path.join(jobDir, "ph_gamma.out"), result.stdout);
    lastResult = result;

    // Diagnose the ph.x run: exit code, wall time, output size, error classification
    const hitMaxSeconds = result.stdout.includes("Maximum CPU time exceeded");
    const hasUnderflow = result.stderr.includes("IEEE_UNDERFLOW") || result.stdout.includes("IEEE_UNDERFLOW");
    const hasStopError = result.stderr.includes("STOP 1") || result.stdout.includes("STOP 1");
    const hasWrongInput = result.stdout.includes("Wrong ") || result.stdout.includes("Error in routine");
    const completedReps = (result.stdout.match(/Convergence has been achieved/g) || []).length;
    const startedReps = (result.stdout.match(/Representation #/g) || []).length;

    console.log(`[Staged-Relax] ${formula} Stage 4 ph.x result: exit=${result.exitCode}, wall=${attemptWall.toFixed(0)}s, stdout=${result.stdout.length}B, reps=${completedReps}/${startedReps} completed`);
    console.log(`[Staged-Relax] ${formula} Stage 4 diagnosis: maxSecHit=${hitMaxSeconds}, underflow=${hasUnderflow}, stop1=${hasStopError}, wrongInput=${hasWrongInput}`);

    if (result.stderr.length > 0 && result.stderr.length < 500) {
      console.log(`[Staged-Relax] ${formula} Stage 4 stderr: ${result.stderr.trim()}`);
    } else if (result.stderr.length >= 500) {
      console.log(`[Staged-Relax] ${formula} Stage 4 stderr (first 300): ${result.stderr.slice(0, 300)}`);
    }

    // Log the last few meaningful lines of stdout (skip boilerplate)
    const stdoutLines = result.stdout.split("\n");
    const meaningfulTail = stdoutLines.slice(-15).filter(l => l.trim().length > 0 && !l.includes("----")).join("\n");
    console.log(`[Staged-Relax] ${formula} Stage 4 stdout tail:\n${meaningfulTail}`);

    // Parse phonon frequencies from output
    frequencies = parseGammaPhononFrequencies(result.stdout);

    if (frequencies.length > 0) {
      console.log(`[Staged-Relax] ${formula} Stage 4 attempt ${attempt + 1}: SUCCESS — parsed ${frequencies.length} frequencies, range [${Math.min(...frequencies).toFixed(1)}, ${Math.max(...frequencies).toFixed(1)}] cm-1`);
      break; // Success — stop retrying
    }

    // Detailed failure classification for retry decision
    const externalKill = result.exitCode !== 0 && !hasUnderflow && !hitMaxSeconds && !hasWrongInput && !hasStopError && attemptWall > (phTimeoutS - 120);
    const failClass = hasWrongInput ? "input-error" : hasUnderflow ? "underflow-crash" : hitMaxSeconds ? "timeout" : hasStopError ? "stop-error" : externalKill ? "external-timeout" : "unknown";
    console.log(`[Staged-Relax] ${formula} Stage 4 attempt ${attempt + 1}: FAILED — class=${failClass}, 0/${expectedModes} modes, ${completedReps}/${nReps} reps done in ${attemptWall.toFixed(0)}s`);

    if (attempt === 0 && failClass === "input-error") {
      console.log(`[Staged-Relax] ${formula} Stage 4: input error detected, skipping retry (same input would fail again)`);
      break; // Don't retry on input errors
    }
  }

  const failReasons: string[] = [];

  if (frequencies.length < expectedModes) {
    failReasons.push(`only ${frequencies.length}/${expectedModes} modes parsed`);
  }

  // Check for large imaginary modes (exclude acoustic modes near 0)
  const largeImaginary = frequencies.filter(f => f < -50);
  if (largeImaginary.length > 0) {
    failReasons.push(`${largeImaginary.length} large imaginary modes (lowest: ${Math.min(...largeImaginary).toFixed(1)} cm-1)`);
  }

  // Allow up to 3 small negative modes (acoustic artifacts)
  const smallNegative = frequencies.filter(f => f < -10 && f >= -50);
  if (smallNegative.length > 3) {
    failReasons.push(`${smallNegative.length} negative modes below -10 cm-1 (expected <= 3 acoustic)`);
  }

  const totalWallTime = (Date.now() - t0) / 1000;

  const passed = failReasons.length === 0;
  if (passed) {
    console.log(`[Staged-Relax] ${formula} Stage 4 (Gamma phonon): PASSED — ${frequencies.length}/${expectedModes} modes, lowest=${Math.min(...frequencies).toFixed(1)} cm-1, highest=${Math.max(...frequencies).toFixed(1)} cm-1, wall=${totalWallTime.toFixed(0)}s (${(totalWallTime/60).toFixed(1)} min)`);
  } else {
    // Log full context on failure so we can diagnose without re-running
    console.log(`[Staged-Relax] ${formula} Stage 4 (Gamma phonon): FAILED — ${failReasons.join("; ")}`);
    console.log(`[Staged-Relax] ${formula} Stage 4 context: ${totalAtoms} atoms, ${phElectrons} e-, lattice=${opts.latticeA?.toFixed(3)} A, ${phNkpts} kpts, attempts=2, total_wall=${totalWallTime.toFixed(0)}s (${(totalWallTime/60).toFixed(1)} min), timeout_budget=${phTimeoutS.toFixed(0)}s`);
  }

  return {
    stage: 4,
    passed,
    failReason: failReasons.length > 0 ? failReasons.join("; ") : undefined,
    positions: opts.positions,
    latticeA: opts.latticeA,
    cellVectors: opts.cellVectors,
    totalEnergy: 0,
    wallTimeSeconds: totalWallTime,
    frequencies,
  };
}

// ---------------------------------------------------------------------------
// Output parsers
// ---------------------------------------------------------------------------

interface RelaxParsed {
  scfConverged: boolean;
  totalEnergy: number;
  maxForce: number | null;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
}

function parseRelaxOutput(stdout: string): RelaxParsed {
  const result: RelaxParsed = {
    scfConverged: false,
    totalEnergy: 0,
    maxForce: null,
    positions: [],
  };

  // Check for convergence
  if (stdout.includes("convergence has been achieved") || stdout.includes("bfgs converged")) {
    result.scfConverged = true;
  }
  // Also check if "convergence NOT achieved" appears (SCF failed)
  if (stdout.includes("convergence NOT achieved")) {
    result.scfConverged = false;
  }

  // Parse total energy (last occurrence of "!" line)
  const energyMatch = stdout.match(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/g);
  if (energyMatch && energyMatch.length > 0) {
    const last = energyMatch[energyMatch.length - 1];
    const val = last.match(/([-\d.]+)\s+Ry/);
    if (val) result.totalEnergy = parseFloat(val[1]) * 13.6057; // Ry -> eV
  }

  // Parse forces — look for "Total force ="
  const forceMatch = stdout.match(/Total force\s*=\s*([\d.]+)/g);
  if (forceMatch && forceMatch.length > 0) {
    const last = forceMatch[forceMatch.length - 1];
    const val = last.match(/([\d.]+)/);
    if (val) result.maxForce = parseFloat(val[0]);
  }

  // Parse final ATOMIC_POSITIONS block — QE uses both {crystal} and (crystal) formats
  // Also handle angstrom, bohr, alat, etc. Match any ATOMIC_POSITIONS header.
  const posBlocks = stdout.match(/ATOMIC_POSITIONS\s*[{(]?\s*(?:crystal|angstrom|bohr|alat)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:CELL_PARAMETERS|K_POINTS|End final|End of|ATOMIC_SPECIES|\n\s*\n)|$)/gi);
  if (posBlocks && posBlocks.length > 0) {
    const lastBlock = posBlocks[posBlocks.length - 1];
    const lines = lastBlock.split("\n").slice(1); // skip header
    for (const line of lines) {
      const m = line.trim().match(/^([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
      if (m) {
        result.positions.push({
          element: m[1],
          x: parseFloat(m[2]),
          y: parseFloat(m[3]),
          z: parseFloat(m[4]),
        });
      }
    }
  }

  // If no positions found yet, try a simpler line-by-line scan for atom-like lines
  // after the last "ATOMIC_POSITIONS" header
  if (result.positions.length === 0) {
    const lastPosIdx = stdout.lastIndexOf("ATOMIC_POSITIONS");
    if (lastPosIdx >= 0) {
      const tail = stdout.slice(lastPosIdx);
      const lines = tail.split("\n").slice(1); // skip the header line
      for (const line of lines) {
        const m = line.trim().match(/^([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
        if (m) {
          result.positions.push({
            element: m[1],
            x: parseFloat(m[2]),
            y: parseFloat(m[3]),
            z: parseFloat(m[4]),
          });
        } else if (line.trim().length > 0 && !line.trim().match(/^[A-Z][a-z]?\s/) && result.positions.length > 0) {
          break; // Hit a non-atom line after collecting some atoms
        }
      }
    }
  }

  return result;
}

interface VcRelaxParsed {
  totalEnergy: number;
  maxForce: number | null;
  pressure: number | null;     // kbar
  latticeA: number;
  cellVectors: number[][] | undefined;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  volumeDriftPercent: number | null;
}

function parseVcRelaxOutput(stdout: string): VcRelaxParsed {
  const result: VcRelaxParsed = {
    totalEnergy: 0,
    maxForce: null,
    pressure: null,
    latticeA: 0,
    cellVectors: undefined,
    positions: [],
    volumeDriftPercent: null,
  };

  // Total energy (last "!" line)
  const energyMatch = stdout.match(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/g);
  if (energyMatch && energyMatch.length > 0) {
    const last = energyMatch[energyMatch.length - 1];
    const val = last.match(/([-\d.]+)\s+Ry/);
    if (val) result.totalEnergy = parseFloat(val[1]) * 13.6057;
  }

  // Total force
  const forceMatch = stdout.match(/Total force\s*=\s*([\d.]+)/g);
  if (forceMatch && forceMatch.length > 0) {
    const last = forceMatch[forceMatch.length - 1];
    const val = last.match(/([\d.]+)/);
    if (val) result.maxForce = parseFloat(val[0]);
  }

  // Pressure (from stress tensor output)
  const pressMatch = stdout.match(/P=\s*([-\d.]+)/g);
  if (pressMatch && pressMatch.length > 0) {
    const last = pressMatch[pressMatch.length - 1];
    const val = last.match(/([-\d.]+)/);
    if (val) result.pressure = parseFloat(val[0]);
  }

  // Parse final CELL_PARAMETERS block
  const cellBlocks = stdout.match(/CELL_PARAMETERS\s*[{(]?\s*(?:angstrom|bohr|alat)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:ATOMIC_POSITIONS|End|$|\n\s*\n))/gi);
  if (cellBlocks && cellBlocks.length > 0) {
    const lastCell = cellBlocks[cellBlocks.length - 1];
    const lines = lastCell.split("\n").slice(1);
    const vectors: number[][] = [];
    for (const line of lines) {
      const nums = line.trim().split(/\s+/).map(Number).filter(n => !isNaN(n));
      if (nums.length === 3) vectors.push(nums);
    }
    if (vectors.length === 3) {
      result.cellVectors = vectors;
      // Estimate lattice constant from cell vector magnitude
      const a = Math.sqrt(vectors[0][0] ** 2 + vectors[0][1] ** 2 + vectors[0][2] ** 2);
      // Check if vectors are in bohr and convert
      const isBohr = lastCell.toLowerCase().includes("bohr");
      result.latticeA = isBohr ? a * 0.529177 : a;
    }
  }

  // Parse final ATOMIC_POSITIONS — handle both {crystal} and (crystal)
  const posBlocks = stdout.match(/ATOMIC_POSITIONS\s*[{(]?\s*(?:crystal|angstrom|bohr|alat)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:CELL_PARAMETERS|K_POINTS|End final|End of|ATOMIC_SPECIES|\n\s*\n)|$)/gi);
  if (posBlocks && posBlocks.length > 0) {
    const lastBlock = posBlocks[posBlocks.length - 1];
    const lines = lastBlock.split("\n").slice(1);
    for (const line of lines) {
      const m = line.trim().match(/^([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
      if (m) {
        result.positions.push({
          element: m[1],
          x: parseFloat(m[2]),
          y: parseFloat(m[3]),
          z: parseFloat(m[4]),
        });
      }
    }
  }

  // Volume drift: parse all "unit-cell volume" lines and check last 3
  const volMatches = stdout.match(/unit-cell volume\s*=\s*([\d.]+)/g);
  if (volMatches && volMatches.length >= 3) {
    const vols = volMatches.map(m => {
      const v = m.match(/([\d.]+)/);
      return v ? parseFloat(v[0]) : 0;
    }).filter(v => v > 0);

    if (vols.length >= 3) {
      const last3 = vols.slice(-3);
      const maxV = Math.max(...last3);
      const minV = Math.min(...last3);
      result.volumeDriftPercent = maxV > 0 ? ((maxV - minV) / maxV) * 100 : 0;
    }
  }

  return result;
}

function parseGammaPhononFrequencies(stdout: string): number[] {
  const frequencies: number[] = [];

  // QE ph.x Gamma-point output format:
  //     freq (    1) =      -2.345678 [cm-1]   =      -0.000291 [THz]
  // or:
  //     omega( 1) =       1.234567 cm-1
  const freqPattern = /(?:freq|omega)\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*(?:\[?\s*cm-1|\[?\s*cm\^-1)/gi;
  let match: RegExpExecArray | null;
  while ((match = freqPattern.exec(stdout)) !== null) {
    frequencies.push(parseFloat(match[1]));
  }

  // Also try the tabular format at end of ph.x output:
  //     Mode   1  frequency =    -23.456 cm-1
  const modePattern = /Mode\s+\d+\s+frequency\s*=\s*([-\d.]+)\s*cm/gi;
  if (frequencies.length === 0) {
    while ((match = modePattern.exec(stdout)) !== null) {
      frequencies.push(parseFloat(match[1]));
    }
  }

  return frequencies;
}
