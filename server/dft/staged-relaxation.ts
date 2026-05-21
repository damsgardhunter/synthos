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
import { lookupKnownStructure } from "../learning/known-structures";

// CODATA 2018 Rydberg → eV. Several inline `13.6057` rounded constants were
// scattered across this file; consolidated to a single precise constant
// matching qe-worker.ts / acbn0-pipeline.ts / sscha-worker.py.
const RY_TO_EV = 13.605693122994;

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

/** A single Stage 2 winner with its geometry and metadata. */
export interface Stage2Winner {
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  latticeA: number;
  cellVectors?: number[][];
  totalEnergy: number;
  energyPerAtom: number;
  maxForce?: number;
  pressure?: number;
  source: string;
  passed: boolean;
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
  /** All unique Stage 2 winners, ranked by per-atom energy. The caller can
   *  use multi-objective scoring to pick beyond just the lowest energy. */
  stage2Winners?: Stage2Winner[];
}

/** Callback interface so this module doesn't depend on qe-worker internals. */
export interface QERunnerCallbacks {
  runPwx(inputFile: string, workDir: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number }>;
  runPhx(inputFile: string, workDir: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number }>;
  /** Run any QE binary (dynmat.x, q2r.x, etc.) by name */
  runQEBinary?(binaryName: string, inputFile: string, workDir: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number }>;
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
  generateCellParameters(latticeA: number, cOverA: number, ibrav?: number, bOverA?: number, elements?: string[], counts?: Record<string, number>, alpha?: number, beta?: number, gamma?: number): string;
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
  // Actinides — full series, matched to qe-worker.ts:HEAVY_ELEMENTS.
  // Without these, Np/Pu/Am/Cm-containing Stage 1 calcs got the
  // light-element cost model → wrong timeout estimate → ran out of time.
  "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
]);

// Magnetic elements need nspin=2 which doubles the SCF cost
// Elements that benefit from a magnetic-search Stage 1 budget. Cu/Ag/Rh
// included for cuprate-class compounds (Cu²⁺ AFM in HgBa2CuO4 etc.) — they
// trigger the magnetic ground-state search downstream, but the cost model
// previously didn't see them as magnetic and used the default 90-min cap,
// timing out before atomic relax converged.
const MAGNETIC_ELS = new Set(["Fe", "Co", "Ni", "Mn", "Cr", "V", "Gd", "Eu", "Nd", "Cu", "Ag", "Rh", "Ru", "Ir"]);

// Valence electron counts per element (from QE pseudopotential zValence).
// Used to estimate SCF cost: more valence electrons = larger basis set = more expensive.
// Kept in sync with ELEMENT_DATA in qe-worker.ts (which is the canonical
// reference, verified against server/dft/pseudo/*.UPF z_valence attributes).
// Discrepancies here cause timeout cost-model errors, not SCF correctness bugs.
const Z_VALENCE: Record<string, number> = {
  H: 1, He: 2, Li: 3, Be: 4, B: 3, C: 4, N: 5, O: 6, F: 7,
  Na: 9, Mg: 10, Al: 3, Si: 4, P: 5, S: 6, Cl: 7,
  K: 9, Ca: 10, Sc: 11, Ti: 12, V: 13, Cr: 14, Mn: 15,
  Fe: 16, Co: 17, Ni: 18, Cu: 19, Zn: 20, Ga: 13, Ge: 4,
  As: 5, Se: 6, Br: 7, Rb: 9, Sr: 10, Y: 11, Zr: 12,
  Nb: 13, Mo: 14, Tc: 15, Ru: 16, Rh: 17, Pd: 18, Ag: 19, Cd: 12,
  In: 13, Sn: 4, Sb: 5, Te: 6, I: 7, Cs: 9, Ba: 10,
  La: 11, Ce: 11, Pr: 13, Nd: 14, Pm: 15, Sm: 16, Eu: 17, Gd: 18,
  Tb: 19, Dy: 20, Ho: 21, Er: 22, Tm: 23, Yb: 24, Lu: 25,
  Hf: 12, Ta: 13, W: 14, Re: 15, Os: 16, Ir: 15, Pt: 16,
  Au: 19, Hg: 20, Tl: 13, Pb: 4, Bi: 5,
  // Actinides — added Ac, Np, Pu, Am, Cm. Without these the cost model
  // saw actinide compounds as low-electron (fell through to a default),
  // underestimated SCF cost, and timed out before convergence on Np/Pu/Am/Cm.
  Ac: 11, Th: 12, Pa: 13, U: 14, Np: 15, Pu: 16, Am: 17, Cm: 18,
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
function computeStage1Params(elements: string[], totalAtoms: number, baseEcutwfc: number, counts?: Record<string, number>): {
  timeoutMs: number;
  ecutwfcScale: number;
  kspacingOverride: number;
  maxSeconds: number;
  isCuprate: boolean;
} {
  const heavyCount = elements.filter(e => HEAVY_ELEMENTS.has(e)).length;
  const hasMagnetic = elements.some(e => MAGNETIC_ELS.has(e));
  // Open-d 3d transition metals — Fe/Mn/Cr/Co/Ni/V — have intrinsically slow
  // magnetic SCF (the spin density takes many iterations to settle, and they
  // need high ecutwfc, ~90 Ry, which makes each iteration expensive). FeSe's
  // Stage 1 finished ZERO ionic steps in the 6000 s magnetic floor.
  const HARD_3D_MAGNETS = new Set(["V", "Cr", "Mn", "Fe", "Co", "Ni"]);
  const hasHard3dMagnet = elements.some(e => HARD_3D_MAGNETS.has(e));

  // Open-d-shell TMs (3d V→Cu and 4d Ru/Rh/Pd) have valence d-electrons whose
  // SCF forces require ecutwfc ≥ 60–80 Ry to converge. The default heavy-atom
  // ecutwfc reduction (0.75 for heavyCount≥3 or totalAtoms≥10) was cutting
  // effective cutoffs to 37–45 Ry — wrong physics for these systems.
  //   - HgBa2CuO4 (Cu): Stage 1 ecutwfc=37 Ry → force=0.125 (50× over publication)
  //   - V3Si (V):       Stage 1 ecutwfc=45 Ry → noisy Stage 1 force gradients
  //   - BaFe2As2 (Fe), LiFeAs (Fe): same risk
  // Don't reduce ecutwfc for ANY open-d TM. Cuprate-class gets a further bump
  // (1.10) + tighter k-grid because Cu d⁹ + Cu-O Fermi surface near van Hove
  // need both extra cutoff headroom AND denser k-sampling.
  const TM_NEEDING_FULL_CUTOFF = new Set([
    // 3d open-shell
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    // 4d open-shell (partial)
    "Ru", "Rh", "Pd",
  ]);
  const hasOpenDTM = elements.some(e => TM_NEEDING_FULL_CUTOFF.has(e));
  const cuprateAnions = (counts?.["Cu"] ?? 0) >= 1 && (counts?.["O"] ?? 0) >= 2;
  const cuprateCations = elements.some(e => ["Ba", "Sr", "La", "Y", "Nd", "Pr", "Sm", "Ca"].includes(e));
  const isCuprate = cuprateAnions && cuprateCations;

  // --- Ecutwfc and kspacing first (needed for cost estimate) ---
  let ecutwfcScale = 1.0;
  if (heavyCount >= 2 || totalAtoms >= 6) ecutwfcScale = 0.85;
  if (heavyCount >= 3 || totalAtoms >= 10) ecutwfcScale = 0.75;
  // Open-d TMs: never go below 1.0 (full cutoff). Cuprates: bump to 1.10.
  if (hasOpenDTM && !isCuprate) ecutwfcScale = Math.max(ecutwfcScale, 1.0);
  if (isCuprate) ecutwfcScale = 1.10;

  let kspacing = KSPACING_RELAX; // 0.40 base
  if (heavyCount >= 1 || totalAtoms >= 5) kspacing = 0.50;
  if (heavyCount >= 2 || totalAtoms >= 8) kspacing = 0.55;
  if (totalAtoms >= 12) kspacing = 0.65;
  if (isCuprate) kspacing = Math.min(kspacing, 0.45); // tighter for Cu Fermi surface

  if (isCuprate) {
    console.log(`[Staged-Relax] Cuprate detected (Cu + ${counts?.O ?? "?"}O + heavy cation): bumping ecutwfcScale=1.10, kspacing=${kspacing}`);
  } else if (hasOpenDTM) {
    const tmElement = elements.find(e => TM_NEEDING_FULL_CUTOFF.has(e));
    console.log(`[Staged-Relax] Open-d TM detected (${tmElement}): holding ecutwfcScale at 1.0 (d-electron forces need full cutoff)`);
  }

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

  // Step 3: nspin factor — spin-polarized doubles the work. Cuprates are the
  // exception: Stage 1 relaxes them at nspin=1 (geometry is insensitive to the
  // AFM order) and adds only one nspin=2 AFM SCF, so the effective cost is the
  // nspin=1 relax plus ~20% for that single magnetic SCF.
  const nspinFactor = isCuprate ? 1.2 : hasMagnetic ? 2.0 : 1.0;

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
  //
  // ecutMultiplier: the plane-wave count (and FFT grid) scales ~ecutwfc^1.5,
  // so a high-cutoff system costs proportionally more PER SCF iteration. The
  // base model ignored this — FeSe runs at 90 Ry (Fe needs it) but was
  // estimated as if it were a ~50 Ry system, a ~2x underestimate. One-sided
  // (>= 1.0): typical 45-70 Ry systems are unchanged; only high-cutoff 3d-TM
  // / hydride systems are scaled up. effEcut = baseEcutwfc * ecutwfcScale is
  // the cutoff Stage 1 will actually use.
  const effEcut = baseEcutwfc * ecutwfcScale;
  const ecutMultiplier = Math.max(1.0, Math.pow(effEcut / 70, 1.4));
  const costFactor = Math.pow(totalAtoms, 1.8) * Math.sqrt(cellElectrons) * Math.pow(nKpoints, 0.6) * nspinFactor * ecutMultiplier;

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

  // Floor at 15 min (simple systems).
  // Cap depends on system complexity:
  // - Simple non-magnetic ambient: 90 min (5400s)
  // Base timeouts calibrated for 7-atom cells; scale by (nAtoms/7)^1.2 for larger.
  // - High-P hydrides: 150 min base (tricky H convergence)
  // - Magnetic (nspin=2): 150 min base — spin-polarized SCF converges slowly
  // - Magnetic + heavy elements: 180 min base
  const hasH = elements.includes("H");
  const isHighPressureHydride = hasH && totalAtoms >= 7;
  const stageAtomScale = totalAtoms > 7 ? Math.pow(totalAtoms / 7, 1.2) : 1.0;
  // Heavy-electron-rich systems (cuprates HgBa2CuO4, perovskites with TM+heavy,
  // bismuth oxides) need more iterations per ionic step due to large basis +
  // partially-filled d shells. ≥70 valence electrons OR ≥2 heavy elements puts
  // the system into this class. HgBa2CuO4 (8 atoms, 83 e-, Hg+Ba heavy, Cu d⁹)
  // hit the default 90-min cap before atomic relax converged in the May-2026 run.
  const isHeavyElectronRich = cellElectrons >= 70 || heavyCount >= 2;
  const rawMaxTimeoutS = Math.round((hasMagnetic
    ? ((heavyCount >= 1 || hasHard3dMagnet) ? 10800 : 9000)
    : isHighPressureHydride ? 9000
    : isHeavyElectronRich ? 9000
    : 5400) * stageAtomScale);
  // Hard ceiling: Stage 1 is screening — it shouldn't burn more than 4 hours
  // on a single candidate. TlBa2Ca2Cu3O9 (34-atom magnetic cuprate) was
  // getting 71958s = 20h via stageAtomScale=6.65. If a structure genuinely
  // needs more, that's a sign it should be downgraded to a cheaper tier or
  // dropped entirely, not blocked for a day.
  // 6h ceiling: a Z=3 clathrate-hydride cell (~36 atoms) the atom cap now
  // admits costs ~5h by the model — a 4h ceiling would guarantee it times
  // out. Larger cells than that should be downgraded to a cheaper tier or
  // dropped, not blocked for a day.
  // Cuprate exception: HgBa2Ca2Cu3O8 (Hg-1223, 32 atoms) was estimated at
  // 128263s = 36 h and clipped to the 6 h cap, so Stage 1 was *guaranteed*
  // to timeout before convergence. Multi-CuO2-layer cuprates (Hg-1212,
  // Hg-1223, Tl-1223, Bi-2212/2223) — heavy-electron-rich AND >= 24 atoms —
  // get a 16 h ceiling. The cell-count gate ensures the small Hg-1201
  // (8 atoms) keeps its 6 h cap; only the genuinely expensive multi-layer
  // cuprates get the bigger budget.
  const STAGE1_HARD_CEILING_S = (isHeavyElectronRich && totalAtoms >= 24) ? 57600 : 21600;
  const maxTimeoutS = Math.min(rawMaxTimeoutS, STAGE1_HARD_CEILING_S);
  // FLOOR by system type — the cost model underestimates for magnetic systems
  // because spin-polarized SCF on Fe/Mn/Cr is intrinsically much harder than
  // the atom-count-based model predicts. Open-d 3d magnets (Fe/Mn/Cr/Co/Ni/V)
  // are the worst: FeSe finished ZERO ionic steps in the old 6000 s floor —
  // one ionic step alone is several thousand seconds. They get a 3 h floor so
  // the relax clears at least one ionic step and produces a geometry. (QE
  // stops at forc_conv_thr if it converges sooner, so the longer floor costs
  // nothing for systems that are actually fast.)
  const minTimeoutS = (hasMagnetic && hasHard3dMagnet) ? 10800 // 3 h for open-d 3d magnets
    : hasMagnetic ? 6000                       // 100 min minimum for ANY magnetic system
    : isHighPressureHydride ? 6000             // 100 min minimum for high-P hydrides
    : 900;                                     // 15 min minimum default
  const clampedTimeoutS = Math.max(minTimeoutS, Math.min(timeoutSeconds, maxTimeoutS));
  const timeoutMs = clampedTimeoutS * 1000;
  const maxSeconds = Math.floor(clampedTimeoutS) - 60; // QE max_seconds slightly under timeout

  console.log(`[Staged-Relax] Cost model: ${totalAtoms} atoms, ${cellElectrons.toFixed(0)} e-, ${nKpoints} kpts, nspin=${nspinFactor} → cost=${costFactor.toFixed(0)}, est=${estimatedSeconds.toFixed(0)}s, timeout=${clampedTimeoutS.toFixed(0)}s${hasMagnetic ? " (magnetic)" : ""}`);

  return { timeoutMs, ecutwfcScale, kspacingOverride: kspacing, maxSeconds, isCuprate };
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
  const s1Params = computeStage1Params(elements, totalAtoms, callbacks.resolveEcutwfc(elements), counts);
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
  const allS2Winners: Stage2Winner[] = [];

  // --- Skip Stages 1-2 for known compounds ---
  if (opts.skipRelaxation) {
    console.log(`[Staged-Relax] ${formula}: skipping Stages 1-2 (known compound or literature lattice)`);
  } else {
    // === STAGE 1: Atomic relax (fixed cell) ===
    const stage1Results: Array<{ result: StageResult; candidate: StructureCandidate }> = [];

    // Atom-count cap: reject supercells that are too large for Stage 1.
    // A Z=4 supercell (60 atoms) gives the same physics as Z=1 (15 atoms)
    // at 4× the cost. Li2LaH12 Z=4 estimated 12+ hours for Stage 1.
    const tier = opts.screeningTier ?? "preview";
    const maxAtomsForS1: Record<string, number> = {
      preview: 30,
      standard: 40,
      deep: 50,
      publication: 60,
    };
    const tierCap = maxAtomsForS1[tier] ?? 30;
    // Also cap relative to the formula unit. A Z=4 cell screens the same
    // chemistry as Z=1 at 4x the cost, so cap the number of formula units.
    // BUT clathrate hydrides — the materials this pipeline targets — very
    // often have a Z>=2 ground-state cell, and a candidate skipped here
    // never reaches Stage 2 either (Stage 2 only refines Stage-1 survivors).
    // So for hydride compositions at the deep/publication tiers, allow up to
    // 3 formula units (a Z=3 clathrate cell, ~36 atoms, fits the 6h Stage-1
    // ceiling); non-hydrides and lighter tiers keep the 2-f.u. cap. Z=4
    // (~48 atoms, ~9h) is still excluded — too expensive, and Z=4 cells are
    // usually reducible. Floor at 12 atoms so tiny formulas keep headroom.
    const hFrac = candidates.length > 0
      ? candidates[0].positions.filter(p => p.element === "H").length /
        Math.max(1, candidates[0].positions.length)
      : 0;
    const isHydrideClathrate = hFrac >= 0.5;
    const fuMultiplier = (isHydrideClathrate && (tier === "deep" || tier === "publication")) ? 3 : 2;
    const atomCap = Math.min(tierCap, Math.max(12, totalAtoms * fuMultiplier));

    // First pass: take the top maxS1 candidates that fit under the atom cap.
    // If an oversized candidate is skipped, backfill from the remaining pool
    // so we still test the same number of candidates.
    const candidatesToTest: StructureCandidate[] = [];
    let skipped = 0;
    for (const c of candidates) {
      if (candidatesToTest.length >= maxS1) break;
      if (c.positions.length > atomCap) {
        skipped++;
        console.log(`[Staged-Relax] ${formula} skipping candidate: ${c.positions.length} atoms > ${atomCap} cap (${c.source}) — backfilling from next`);
        continue;
      }
      candidatesToTest.push(c);
    }
    // If all candidates were filtered, use the smallest one anyway
    if (candidatesToTest.length === 0 && candidates.length > 0) {
      const smallest = [...candidates].sort((a, b) => a.positions.length - b.positions.length)[0];
      candidatesToTest.push(smallest);
      console.log(`[Staged-Relax] ${formula} all candidates exceeded atom cap — using smallest (${smallest.positions.length} atoms, ${smallest.source})`);
    }
    if (skipped > 0) {
      console.log(`[Staged-Relax] ${formula} atom cap: skipped ${skipped} oversized, backfilled to ${candidatesToTest.length} candidates`);
    }
    console.log(`[Staged-Relax] ${formula} Stage 1/5 (atomic relax): testing ${candidatesToTest.length} candidates (atom cap=${atomCap})`);

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
    // tier already declared above for atom cap
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
    // Rank by ENTHALPY per atom (H = E + P×V), not pure total energy. At high
    // pressure, an uncompressed candidate looks artificially attractive on E
    // alone (no compression penalty) but has disastrous PV — picking it forces
    // vc-relax to compress the cell 30-40% in one go, which causes refinement
    // to overshoot and degrade forces (LaH10 May-2026: Stage 1 picked a=5.57 Å
    // PyXtal SG=2 over Vegard's pre-compressed 3.94 Å, and refinement made
    // force WORSE by 12× while only partly recovering pressure).
    //
    // Conversion: 1 GPa·Å³ = 6.241e-3 eV. So PV/atom in eV = P[GPa] × V/atom[Å³] × 0.006241.
    // At ambient (P=0) this reduces to pure-E ranking. At 170 GPa with V=15 Å³/atom,
    // PV ≈ 16 eV/atom — completely dominates differences in E.
    const GPA_ANGSTROM3_TO_EV = 6.241e-3;
    // Triclinic/monoclinic volume factor: V = a·b·c·√(1 - cos²α - cos²β -
    // cos²γ + 2cosα·cosβ·cosγ). For α=β=γ=90° this reduces to 1 (orthogonal),
    // so it is safe to apply unconditionally. Without it, monoclinic candidates
    // (VO2 β=122.6°, ZrO2, HfO2, LaPO4, CePO4) get a·b·c volumes inflated by
    // up to ~20%, which inflates the PV enthalpy penalty and unfairly demotes
    // them in the Stage 1 ranking — the same orthogonal-cell assumption that
    // Stage 1 input generation (cellAlpha/cellBeta/cellGamma below) avoids.
    const enthalpyKS = lookupKnownStructure(formula);
    const volAlpha = (enthalpyKS?.alpha ?? 90) * Math.PI / 180;
    const volBeta = (enthalpyKS?.beta ?? 90) * Math.PI / 180;
    const volGamma = (enthalpyKS?.gamma ?? 90) * Math.PI / 180;
    const cosA = Math.cos(volAlpha), cosB = Math.cos(volBeta), cosG = Math.cos(volGamma);
    const cellAngleFactor = Math.sqrt(Math.max(0,
      1 - cosA * cosA - cosB * cosB - cosG * cosG + 2 * cosA * cosB * cosG));
    const perAtomEnthalpy = (r: { result: StageResult; candidate: StructureCandidate }) => {
      const nAtoms = r.result.positions.length || r.candidate.positions.length || 1;
      const E = r.result.totalEnergy; // eV total
      // Compute cell volume from lattice parameters (StructureCandidate has
      // latticeA, optional latticeB/latticeC, cOverA — no explicit volume field).
      // Stage 1 doesn't re-compute volume because calculation='relax' keeps
      // cell fixed.
      const a = r.candidate.latticeA;
      const bOverA = r.candidate.latticeB ? r.candidate.latticeB / a : 1.0;
      const cOverA = r.candidate.cOverA ?? (r.candidate.latticeC ? r.candidate.latticeC / a : 1.0);
      const cellVolAng3 = a * a * a * bOverA * cOverA * cellAngleFactor;
      const volPerAtom = cellVolAng3 / Math.max(1, nAtoms);
      const PVperAtom = pressureGPa * volPerAtom * GPA_ANGSTROM3_TO_EV;
      return E / nAtoms + PVperAtom;
    };
    allS1Sorted.sort((a, b) => perAtomEnthalpy(a) - perAtomEnthalpy(b));

    if (passedS1.length > 0) {
      const kept = passedS1.slice(0, maxStage2Candidates);
      const best = kept[0];
      bestPositions = best.result.positions;
      bestLatticeA = best.result.latticeA;
      candidateSource = best.candidate.source;
      stages.push(best.result);
      console.log(`[Staged-Relax] ${formula} Stage 1: ${passedS1.length} passed, keeping top ${kept.length} for Stage 2 (tier=${tier}, ranked by ${pressureGPa > 0 ? "enthalpy H=E+PV at " + pressureGPa + " GPa" : "energy"})`);
      for (let ki = 0; ki < kept.length; ki++) {
        const nAtoms = kept[ki].result.positions.length || kept[ki].candidate.positions.length || 1;
        const Hperatom = perAtomEnthalpy(kept[ki]);
        const Eperatom = kept[ki].result.totalEnergy / nAtoms;
        const PVperatom = Hperatom - Eperatom;
        const pvSuffix = pressureGPa > 0 ? `, PV/atom=${PVperatom.toFixed(3)} eV, H/atom=${Hperatom.toFixed(4)} eV` : "";
        console.log(`[Staged-Relax]   #${ki + 1}: ${kept[ki].candidate.source} (E=${kept[ki].result.totalEnergy.toFixed(4)} eV, ${nAtoms} atoms, E/atom=${Eperatom.toFixed(4)} eV${pvSuffix})`);
      }
    } else if (stage1Results.length > 0) {
      // No candidate passed — use the one with lowest energy anyway (best effort).
      // GUARD: if the best "best-effort" candidate has catastrophic forces
      // (>10 Ry/bohr), the structure is physically broken — atoms essentially
      // overlapping, wrong topology, or stage-1 SCF gave garbage forces. Passing
      // this to vc-relax + magnetic search is hours of wasted compute (YBa2Cu3O7
      // May-2026: force=415 Ry/bohr accepted as best-effort, then mag search
      // burned 1h/trial on broken geometry). Mark Stage 1 as fully failed in
      // that case so the downstream pipeline skips this material.
      const best = allS1Sorted[0];
      const bestForce = best.result.maxForce ?? Infinity;
      const CATASTROPHIC_FORCE_RYBOHR = 10.0;
      if (bestForce > CATASTROPHIC_FORCE_RYBOHR) {
        console.log(`[Staged-Relax] ${formula} Stage 1: best candidate has catastrophic force ${bestForce.toExponential(2)} Ry/bohr > ${CATASTROPHIC_FORCE_RYBOHR} — refusing best-effort fallback; structure is physically broken`);
        stages.push({
          stage: 1, passed: false,
          failReason: `all candidates failed; best had F=${bestForce.toExponential(2)} Ry/bohr (catastrophic)`,
          positions: bestPositions, latticeA: bestLatticeA, totalEnergy: 0, wallTimeSeconds: 0,
          maxForce: bestForce,
        });
        // Return Stage 1 result as failed without overwriting bestPositions/bestLatticeA
        // so the caller still has the original candidate's geometry as a fallback
        // signal, but Stage 1 is flagged as failed.
      } else {
        bestPositions = best.result.positions.length > 0 ? best.result.positions : best.candidate.positions;
        bestLatticeA = best.result.latticeA > 0 ? best.result.latticeA : best.candidate.latticeA;
        candidateSource = best.candidate.source;
        stages.push(best.result);
        console.log(`[Staged-Relax] ${formula} Stage 1: NO candidate passed, using best-effort from ${candidateSource} (F=${bestForce < 100 ? bestForce.toExponential(2) : "N/A"} Ry/bohr)`);
      }
    } else {
      // All crashed — use first candidate as-is
      console.log(`[Staged-Relax] ${formula} Stage 1: all candidates crashed, using raw candidate`);
      stages.push({
        stage: 1, passed: false, failReason: "all candidates crashed",
        positions: bestPositions, latticeA: bestLatticeA, totalEnergy: 0, wallTimeSeconds: 0,
      });
    }

    // Stage 2 (BFGS vc-relax) REMOVED — BFGS with cell changes causes force
    // regression (e.g. H3S force 0.0015 → 0.127). The unified vc-relax in
    // qe-worker.ts does proper cell optimization with damped dynamics + tight
    // SCF + refinement loop. Stage 1 picks the best candidate, unified vc-relax
    // finds the correct cell.
    console.log(`[Staged-Relax] ${formula}: skipping Stage 2 — unified vc-relax handles cell optimization (Stage 1 force=${(passedS1[0]?.result.maxForce ?? 999).toExponential(2)})`);
  }

  const totalWallTime = (Date.now() - startTime) / 1000;

  // Sort Stage 2 winners by per-atom energy for the caller
  const sortedWinners = allS2Winners.length > 0
    ? allS2Winners.sort((a, b) => a.energyPerAtom - b.energyPerAtom)
    : undefined;

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
    stage2Winners: sortedWinners,
  };
}

// ---------------------------------------------------------------------------
// Stage 1: Atomic relax (fixed cell, positions only)
// ---------------------------------------------------------------------------

/**
 * Verify a QE input file was written intact before invoking pw.x.
 *
 * A truncated or partially-written input makes QE abort instantly with
 * "could not find namelist &control" — observed as a burst of sub-second
 * Stage 1 candidate failures (write race / interrupted write on a busy or
 * restarting worker). The generated `expected` string always contains the
 * &CONTROL namelist, so any on-disk mismatch is an I/O problem: rewrite
 * once, and throw with diagnostics if it still cannot be persisted.
 */
function verifyQEInputWritten(file: string, expected: string, formula: string, label: string): void {
  for (let attempt = 0; attempt < 2; attempt++) {
    let onDisk = "";
    try { onDisk = fs.readFileSync(file, "utf-8"); } catch {}
    if (onDisk.length === expected.length && /&control/i.test(onDisk)) return;
    console.log(`[Staged-Relax] ${formula} ${label}: input file mismatch (on-disk ${onDisk.length}B vs ${expected.length}B, hasControl=${/&control/i.test(onDisk)}) — rewriting (attempt ${attempt + 1})`);
    try { fs.writeFileSync(file, expected); } catch (wErr: any) {
      console.log(`[Staged-Relax] ${formula} ${label}: rewrite failed — ${wErr?.message?.slice(0, 120)}`);
    }
  }
  let finalCheck = "";
  try { finalCheck = fs.readFileSync(file, "utf-8"); } catch {}
  if (!/&control/i.test(finalCheck)) {
    throw new Error(`${label} input for ${formula} could not be written intact (${finalCheck.length}B on disk, no &control namelist) — disk/IO failure`);
  }
}

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
  const cOverA = candidate.cOverA
    ?? (candidate.latticeC != null ? candidate.latticeC / latticeA : cb.estimateCOverA(elements, counts));
  // Preserve orthorhombic b/a from the candidate (MP, POSCAR, AIRSS sources
  // already supply distinct b). Without this Stage 1 forces b=a, which biases
  // BFGS into a tetragonal basin and discards structural information the
  // upstream pipeline paid for.
  const bOverA = candidate.latticeB != null && candidate.latticeB > 0
    ? candidate.latticeB / latticeA
    : 1.0;
  // Monoclinic angle from literature when available — without this the cell
  // is generated with β=90° even for known monoclinic compounds (VO2, ZrO2,
  // HfO2, LaPO4, CePO4) and the BFGS relax cannot recover the true symmetry
  // since ibrav=0 + initially orthogonal vectors keeps β locked at 90°.
  const ks = lookupKnownStructure(formula);
  const cellAlpha = ks?.alpha ?? 90;
  const cellBeta = ks?.beta ?? 90;
  const cellGamma = ks?.gamma ?? 90;
  const totalAtoms = positions.length;
  const nTypes = elements.length;

  // Scale timeout, ecutwfc, and k-grid based on element complexity
  const baseEcutwfc = cb.resolveEcutwfc(elements);
  const s1Params = computeStage1Params(elements, totalAtoms, baseEcutwfc, counts);

  const ecutwfc = Math.round(baseEcutwfc * s1Params.ecutwfcScale);
  const ecutrho = cb.resolveEcutrho(elements, ecutwfc);
  const kpoints = cb.autoKPoints(latticeA, cOverA, s1Params.kspacingOverride);

  const hasMag = cb.hasMagneticElements(elements);
  const isCuprate = s1Params.isCuprate;
  // Cuprate geometry/magnetism decoupling: the CuO2-plane geometry is
  // insensitive to the AFM spin order (magnetostriction shifts Cu-O bonds
  // < 0.02 Å), so the relax runs at nspin=1 — much cheaper and far better
  // converging — and the magnetic energy comes from one nspin=2 AFM SCF on
  // the relaxed cell (below). Stage 2 vc-relax still does the fully magnetic
  // relaxation. Non-cuprate magnets (Fe-pnictides) keep the coupled nspin=2
  // relax: their orthorhombic distortion tracks the stripe-AFM order.
  const relaxMagnetic = hasMag && !isCuprate;
  const nspin = relaxMagnetic ? 2 : 1;
  const magLines = relaxMagnetic ? cb.generateMagnetizationLines(elements, counts) : "";
  // Cuprates split the Stage 1 budget: ~70% for the nspin=1 relax, the rest
  // for the nspin=2 AFM SCF afterwards. Non-cuprates use the full budget for
  // their single relax run.
  const relaxTimeoutMs = isCuprate ? Math.round(s1Params.timeoutMs * 0.7) : s1Params.timeoutMs;
  const relaxMaxSeconds = isCuprate
    ? Math.max(300, Math.round(relaxTimeoutMs / 1000) - 60)
    : s1Params.maxSeconds;

  let atomicSpecies = "";
  for (const el of elements) {
    atomicSpecies += `  ${el}  ${cb.getAtomicMass(el).toFixed(3)}  ${cb.resolvePPFilename(el)}\n`;
  }

  let atomicPositions = "";
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const cellBlock = cb.generateCellParameters(latticeA, cOverA, 0, bOverA, elements, counts, cellAlpha, cellBeta, cellGamma);

  // Stage 1 SCF physics — tightened for magnetic systems. The previous params
  // (conv_thr=1e-4, scf_must_converge=.false., mixing_beta=0.4, degauss=0.015)
  // were calibrated for non-magnetic screening but broke magnetic systems:
  // - BaFe2As2 cand 2: force=0.84 Ry/bohr after 3.4h (FM/AFM SCF oscillation)
  // - TlBa2CaCu2O7 cand 1: force=0.39 Ry/bohr at 3.4h
  // For nspin=2 the FM-AFM energy gap is ~few meV/atom; a 1e-4 Ry SCF tolerance
  // (1.4 meV/atom) cannot resolve the magnetic minimum. scf_must_converge=.false.
  // let BFGS use forces from a still-oscillating SCF — the "directionally
  // correct" comment was wrong for magnetic systems. Now we tighten everything
  // when hasMag and let SCF converge before each ionic step.
  // Keyed on relaxMagnetic, not hasMag: a cuprate relax is nspin=1 and gets
  // the cheaper screening-quality SCF settings (the tight magnetic settings
  // are reserved for the nspin=2 AFM SCF that follows the relax).
  const scfConvThr = relaxMagnetic ? "1.0d-7" : "1.0d-4";        // tighter for magnetic
  // Ionic (BFGS) energy-convergence threshold. 1e-4 Ry ≈ 1.4 meV/atom — too
  // coarse to resolve a magnetic ground state, where FM/AFM configurations
  // differ by only a few meV/atom. Tighten to 1e-5 Ry for magnetic relaxations.
  const etotConvThr = relaxMagnetic ? "1.0d-5" : "1.0d-4";
  const mixingBeta = relaxMagnetic ? 0.2 : 0.4;                  // gentler nspin=2 mixing
  const degauss = relaxMagnetic ? 0.005 : 0.015;                 // narrower smearing for metallic magnets
  const electronMaxstep = relaxMagnetic ? 300 : 200;             // give magnetic SCF more room
  const scfMustConvergeLine = relaxMagnetic ? "" : "  scf_must_converge = .false.,\n"; // require convergence for magnetic
  // Davidson subspace dimension. The QE default (2) is too tight for the dense,
  // tightly-spaced band manifold of cuprates with heavy semicore states
  // (HgBa2CaCu2O6 May 20 aborted both Stage 1 candidates at SCF iteration 1
  // with "too many bands are not converged" — c_bands failure inside the
  // david diagonalizer). Bumping to 4 gives Davidson enough subspace to
  // resolve the lowest eigenvalues at the cost of a bit more memory.
  const cuprateDavidNdimLine = isCuprate ? "  diago_david_ndim = 4,\n" : "";

  const input = `&CONTROL
  calculation = 'relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}_s1_${candidateIdx}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${cb.getPseudoDirInput()}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${STAGE1_FORCE_THR.toExponential(1).replace(/e([+-])/, "d$1")},
  etot_conv_thr = ${etotConvThr},
  nstep = 100,
  max_seconds = ${relaxMaxSeconds},
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
  degauss = ${degauss},
  nspin = ${nspin},
${magLines}/
&ELECTRONS
  electron_maxstep = ${electronMaxstep},
  conv_thr = ${scfConvThr},
  mixing_beta = ${mixingBeta},
  mixing_mode = 'local-TF',
  diagonalization = 'david',
${cuprateDavidNdimLine}${scfMustConvergeLine}/
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

  // --- Pre-relax SCF feasibility probe ---
  // A physically broken CSP candidate (overlapping atoms, unphysical
  // geometry) makes QE abort — but only AFTER the full multi-hour Stage 1
  // relax has burned its entire wall-time budget producing no geometry
  // (logged downstream as "catastrophic force Infinity — physically
  // broken"). A short, coarse nspin=1 SCF probe catches a genuinely broken
  // structure in minutes: if QE aborts with an error, skip the candidate
  // instead of wasting the full relax budget. A merely-slow (not broken)
  // candidate produces an energy or cleanly hits the probe's max_seconds —
  // neither is flagged, so expensive-but-valid structures still proceed.
  const probeMaxSec = Math.min(900, Math.max(180, Math.round(s1Params.maxSeconds * 0.12)));
  const probeInput = `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${prefix}_s1probe_${candidateIdx}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${cb.getPseudoDirInput()}',
  tprnfor = .true.,
  max_seconds = ${probeMaxSec},
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
  degauss = 0.02,
  nspin = 1,
/
&ELECTRONS
  electron_maxstep = 60,
  conv_thr = 1.0d-3,
  mixing_beta = 0.3,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
${cuprateDavidNdimLine}  scf_must_converge = .false.,
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${kpoints}

${cellBlock}
`;
  const probeFile = path.join(stageDir, "scf_probe.in");
  fs.writeFileSync(probeFile, probeInput);
  verifyQEInputWritten(probeFile, probeInput, formula, "Stage 1 SCF probe");
  const probeResult = await cb.runPwx(probeFile, stageDir, (probeMaxSec + 120) * 1000);
  fs.writeFileSync(path.join(stageDir, "scf_probe.out"), probeResult.stdout);
  cb.cleanTmpDir(path.join(stageDir, "tmp"));
  // "could not find namelist &control" is an INPUT-DELIVERY failure (QE
  // never received its input), not a verdict on the geometry — the input
  // file itself was verified intact above. Never reject a candidate for it:
  // let the full relax proceed (and surface the infrastructure problem).
  const probeInputNotDelivered = /could not find namelist/i.test(probeResult.stdout);
  if (probeInputNotDelivered) {
    console.log(`[Staged-Relax] ${formula} Stage 1 candidate ${candidateIdx + 1}: SCF probe INCONCLUSIVE — QE reported "could not find namelist &control" (input not delivered, not a geometry fault); proceeding to the full relax.`);
  }
  // A QE error abort (non-zero exit + an "Error in routine" block) means the
  // geometry is unphysical for DFT. A clean max_seconds stop exits 0, so a
  // slow-but-valid structure is NOT flagged and proceeds to the full relax.
  if (probeResult.exitCode !== 0 && /Error in routine/.test(probeResult.stdout) && !probeInputNotDelivered) {
    const probeTail = probeResult.stdout.slice(-220).replace(/\s+/g, " ").trim();
    console.log(`[Staged-Relax] ${formula} Stage 1 candidate ${candidateIdx + 1}: SCF feasibility probe FAILED — QE aborted on this geometry (${probeMaxSec}s probe), skipping the full relax. Tail: ${probeTail}`);
    return {
      stage: 1,
      passed: false,
      failReason: "SCF feasibility probe failed — QE aborted on an unphysical geometry",
      positions,
      latticeA,
      totalEnergy: 0,
      maxForce: undefined,
      wallTimeSeconds: (Date.now() - t0) / 1000,
      scfConverged: false,
    };
  }

  const inputFile = path.join(stageDir, "relax.in");
  fs.writeFileSync(inputFile, input);
  verifyQEInputWritten(inputFile, input, formula, "Stage 1 relax");

  console.log(`[Staged-Relax] ${formula} S1 params: timeout=${Math.round(relaxTimeoutMs/1000)}s, ecutwfc=${ecutwfc}Ry, kspacing=${s1Params.kspacingOverride}, lattice=${latticeA.toFixed(3)} Å, P=${pressureGPa} GPa${isCuprate ? `, nspin=1 relax + AFM SCF` : ""}`);
  const result = await cb.runPwx(inputFile, stageDir, relaxTimeoutMs);
  fs.writeFileSync(path.join(stageDir, "relax.out"), result.stdout);

  // Parse output
  const parsed = parseRelaxOutput(result.stdout);
  cb.cleanTmpDir(path.join(stageDir, "tmp"));

  // --- Cuprate magnetic energy: one nspin=2 AFM SCF on the relaxed cell ---
  // The relax above ran nspin=1 (geometry only). For a meaningful enthalpy
  // ranking the cuprate energy must include the antiferromagnetic Cu order, so
  // do one SCF here. QE's starting_magnetization is per-SPECIES, so true AFM
  // requires splitting Cu into two species (Cu1/Cu2, same pseudo) with
  // opposite seed moments; Cu atoms are alternated between them so neither
  // sublattice is empty. QE then relaxes the spins to the broken-symmetry
  // ground state. If the SCF fails, the nspin=1 relax energy is kept.
  let stage1Energy = parsed.totalEnergy;
  if (isCuprate && hasMag && parsed.positions.length > 0) {
    try {
      const MAG_EL = "Cu";
      const CU_SEED = 0.5; // Cu(2+) d9, S=1/2 — moderate AFM seed; QE relaxes it
      // Collinear AFM needs >= 2 Cu atoms so each spin sublattice is non-empty.
      // A single-Cu primitive cell (Hg-1201 HgBa2CuO4 at Z=1, La-214 at Z=1,
      // etc.) cannot host intra-cell AFM order — splitting Cu -> Cu1/Cu2 there
      // leaves Cu2 with zero atoms, and QE rejects that as a malformed input
      // (a species declared in ATOMIC_SPECIES with no atoms). Below 2 Cu we
      // fall back to a single spin-polarised Cu species, which still captures
      // the on-site Cu moment energy for the enthalpy ranking.
      const cuCount = parsed.positions.filter(p => p.element === MAG_EL).length;
      const splitAFM = cuCount >= 2;
      const speciesList: string[] = [];
      for (const el of elements) {
        if (el === MAG_EL) {
          if (splitAFM) speciesList.push("Cu1", "Cu2");
          else speciesList.push("Cu");
        } else {
          speciesList.push(el);
        }
      }
      let afmSpecies = "";
      for (const sp of speciesList) {
        const baseEl = (sp === "Cu1" || sp === "Cu2") ? MAG_EL : sp;
        afmSpecies += `  ${sp}  ${cb.getAtomicMass(baseEl).toFixed(3)}  ${cb.resolvePPFilename(baseEl)}\n`;
      }
      let cuCounter = 0;
      let afmPositions = "";
      for (const pos of parsed.positions) {
        const label = pos.element === MAG_EL
          ? (splitAFM ? (cuCounter++ % 2 === 0 ? "Cu1" : "Cu2") : "Cu")
          : pos.element;
        afmPositions += `  ${label}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
      }
      let afmMag: string;
      if (splitAFM) {
        const iCu1 = speciesList.indexOf("Cu1") + 1;
        const iCu2 = speciesList.indexOf("Cu2") + 1;
        afmMag =
          `  starting_magnetization(${iCu1}) = ${CU_SEED.toFixed(1)},\n` +
          `  starting_magnetization(${iCu2}) = ${(-CU_SEED).toFixed(1)},\n`;
      } else {
        const iCu = speciesList.indexOf("Cu") + 1;
        afmMag = `  starting_magnetization(${iCu}) = ${CU_SEED.toFixed(1)},\n`;
      }
      const afmTimeoutMs = Math.round(s1Params.timeoutMs * 0.35);
      const afmMaxSeconds = Math.max(300, Math.round(afmTimeoutMs / 1000) - 60);
      const afmInput = `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${prefix}_s1_${candidateIdx}_afm',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${cb.getPseudoDirInput()}',
  tprnfor = .true.,
  max_seconds = ${afmMaxSeconds},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${speciesList.length},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.005,
  nspin = 2,
${afmMag}/
&ELECTRONS
  electron_maxstep = 300,
  conv_thr = 1.0d-7,
  mixing_beta = 0.2,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
  diago_david_ndim = 4,
/
ATOMIC_SPECIES
${afmSpecies}
ATOMIC_POSITIONS {crystal}
${afmPositions}
K_POINTS {automatic}
${kpoints}

${cellBlock}
`;
      const afmFile = path.join(stageDir, "scf_afm.in");
      fs.writeFileSync(afmFile, afmInput);
      verifyQEInputWritten(afmFile, afmInput, formula, "Stage 1 AFM SCF");
      console.log(`[Staged-Relax] ${formula} S1 cuprate magnetic SCF: ${splitAFM ? `Cu split into Cu1/Cu2 AFM (${cuCount} Cu atoms, seed=±${CU_SEED})` : `single-Cu cell — one spin-polarised Cu species (seed=+${CU_SEED}), intra-cell AFM not possible`}, nspin=2`);
      const afmResult = await cb.runPwx(afmFile, stageDir, afmTimeoutMs);
      fs.writeFileSync(path.join(stageDir, "scf_afm.out"), afmResult.stdout);
      const afmParsed = parseRelaxOutput(afmResult.stdout);
      cb.cleanTmpDir(path.join(stageDir, "tmp"));
      if (afmParsed.totalEnergy !== 0 && Number.isFinite(afmParsed.totalEnergy)) {
        console.log(`[Staged-Relax] ${formula} S1 cuprate AFM energy=${afmParsed.totalEnergy.toFixed(4)} eV (nspin=1 relax energy was ${parsed.totalEnergy.toFixed(4)} eV)`);
        stage1Energy = afmParsed.totalEnergy;
      } else {
        console.log(`[Staged-Relax] ${formula} S1 cuprate AFM SCF produced no energy — keeping nspin=1 relax energy`);
      }
    } catch (afmErr: any) {
      console.log(`[Staged-Relax] ${formula} S1 cuprate AFM SCF failed: ${afmErr?.message?.slice(0, 120)} — keeping nspin=1 relax energy`);
    }
  }

  const wallTime = (Date.now() - t0) / 1000;

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
    totalEnergy: stage1Energy,
    maxForce: parsed.maxForce ?? undefined,
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

  // Stage 2 SCF physics — adopt the same magnetic-aware tuning that
  // Stage 1 uses (see comment at runStage1AtomicRelax). The previous
  // hardcoded params (degauss=0.015, mixing_beta=0.25, conv_thr=1e-6,
  // scf_must_converge=.false.) were calibrated for non-magnetic vc-relax;
  // for magnetic systems they let BFGS use forces from a still-oscillating
  // SCF, producing wrong relaxed geometries (e.g., BaFe2As2 cell volume
  // drifting because the AFM-FM mixing hadn't converged).
  const s2ConvThr = hasMag ? "1.0d-7" : "1.0d-6";
  const s2MixingBeta = hasMag ? 0.20 : 0.25;
  const s2Degauss = hasMag ? 0.005 : 0.015;
  const s2ElectronMaxStep = hasMag ? 400 : 300;
  const s2ScfMustConvergeLine = hasMag ? "" : "  scf_must_converge = .false.,\n";

  const input = `&CONTROL
  calculation = 'vc-relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}_s2',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${cb.getPseudoDirInput()}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${STAGE2_FORCE_THR.toExponential(1).replace(/e([+-])/, "d$1")},
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
  degauss = ${s2Degauss},
  nspin = ${nspin},
${magLines}/
&ELECTRONS
  electron_maxstep = ${s2ElectronMaxStep},
  conv_thr = ${s2ConvThr},
  mixing_beta = ${s2MixingBeta},
  mixing_mode = 'local-TF',
  diagonalization = 'david',
${s2ScfMustConvergeLine}/
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
  verifyQEInputWritten(inputFile, input, formula, "Stage 2 vc-relax");

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
  // Residual pressure check. For ambient (P=0), absolute |P| must be small.
  // For finite-pressure calcs, the residual must be close to the TARGET
  // (QE's press_conv_thr handles this internally but we double-check the
  // parsed value). The tolerance is ABSOLUTE (10 kbar high-P, 2 kbar ambient),
  // NOT a fraction of the target: a 2%-of-target tolerance reached 60 kbar
  // (6 GPa) at 300 GPa, which shifts a hydride volume by ~2% and materially
  // changes λ/Tc — far looser than QE's own press_conv_thr (~0.5-1 kbar).
  if (parsed.pressure != null) {
    const targetKbar = pressureGPa * 10.0;
    const residualKbar = Math.abs(parsed.pressure - targetKbar);
    const tolKbar = pressureGPa === 0 ? 2.0 : 10.0;
    if (residualKbar > tolKbar) {
      failReasons.push(
        `residual pressure ${parsed.pressure.toFixed(1)} kbar deviates from target ` +
        `${targetKbar.toFixed(1)} kbar by ${residualKbar.toFixed(1)} kbar (tol ${tolKbar.toFixed(1)})`,
      );
    }
  }

  return {
    stage: 2,
    passed: failReasons.length === 0,
    failReason: failReasons.length > 0 ? failReasons.join("; ") : undefined,
    positions: parsed.positions.length > 0 ? parsed.positions : positions,
    latticeA: parsed.latticeA > 0 ? parsed.latticeA : latticeA,
    cellVectors: parsed.cellVectors,
    totalEnergy: parsed.totalEnergy,
    maxForce: parsed.maxForce ?? undefined,
    pressure: parsed.pressure ?? undefined,
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
  /** SCF metallicity (informational — from parseSCFOutput's gap detection). */
  isMetallic?: boolean;
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

  // Gamma phonon always runs — it's the cheapest stability check available.
  // Even at 6-8 hours, it's far cheaper than a 48h full grid that may reveal
  // the structure is garbage. The old 4h cap caused CaBeH8 to skip gamma
  // entirely, go straight to a 48h full grid, and waste compute on a structure
  // with 68 imaginary modes that gamma would have caught in 3h.
  //
  // Cap at 8 hours for gamma-only (vs 48h for full grid).
  const phTimeoutS = Math.round(Math.max(1800, Math.min(estimatedPhSeconds + 600, 28800)));
  const phTimeoutMs = phTimeoutS * 1000;

  console.log(`[Staged-Relax] ${formula} Stage 4 cost model: ${nReps} reps, ${phElectrons} e-, ${phNkpts} kpts, nspin=${phNspin} → cost/rep=${costPerRep.toFixed(0)}, est=${estimatedPhSeconds.toFixed(0)}s, timeout=${phTimeoutS.toFixed(0)}s (${(phTimeoutS/60).toFixed(0)} min)`);

  // No `epsil` — only `trans=.true.` (the phonon response itself). ph.x's
  // phq_readin ABORTS with "no elec. field with metals" whenever epsil/zeu/zue
  // is requested AND the SCF used smearing occupations (lgauss=.true.). Every
  // QAE SCF runs `occupations='smearing'`, so lgauss is always true and epsil
  // can never be used. The old `isMetallic===false` gate keyed on gap
  // detection, unrelated to ph.x's occupation-based check, so epsil leaked
  // through and crashed metallic runs. Born charges / LO-TO splitting only
  // matter for polar insulators, which are not superconductors.
  const epsilFlags = "  trans = .true.,\n";

  // 2-attempt retry matching production phonon pipeline (qe-worker.ts lines 4580-4644):
  //   Attempt 1: tr2_ph=1e-12, alpha_mix=0.5 (production defaults)
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
    // On timeout (no crash): use recover=.true. to resume from checkpoint.
    // Attempt 1 uses tr2_ph=1e-12 (matches publication-grade tier in
    // generatePhononInput); attempt 2 on crash drops to 1e-10 with α=0.1.
    const retryTr2 = prevCrashed ? "1.0d-10" : "1.0d-12";
    const retryAlpha = prevCrashed ? 0.1 : 0.5;
    const recoverLine = (prevTimedOut || prevExternalKill) && !prevCrashed ? "  recover = .true.,\n" : "";

    const attemptInput = `Gamma-only phonon calculation
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = ${retryTr2},
  alpha_mix(1) = ${retryAlpha},
  reduce_io = .true.,
  ldisp = .false.,
  max_seconds = ${phTimeoutS - 60},
${epsilFlags}${recoverLine}/
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

    // If ph.x completed (exit=0) but no frequencies parsed, run dynmat.x
    // to diagonalize the dynamical matrix and extract frequencies.
    // QE 7.x gamma-only writes the raw matrix to .dyn; dynmat.x post-processes
    // it to get eigenvalues (frequencies).
    if (frequencies.length === 0 && result.exitCode === 0) {
      try {
        // ph.x writes either <prefix>.dyn (ldisp=.false. single q-point) or
        // <prefix>.dyn1 (newer QE versions append the q-point index even when
        // there's only one). Probe both.
        const dynCandidates = [
          path.join(opts.jobDir, `${prefix}.dyn`),
          path.join(opts.jobDir, `${prefix}.dyn1`),
        ];
        const dynFile = dynCandidates.find(p => fs.existsSync(p));
        const dynBasename = dynFile ? path.basename(dynFile) : null;
        if (dynFile && dynBasename) {
          console.log(`[Staged-Relax] ${formula} Stage 4: ph.x wrote ${dynBasename} but no freqs in stdout — running dynmat.x`);

          // ASR ladder: 'crystal' is the general formulation that works on any
          // Bravais lattice; 'simple' is the original 1970s code path and only
          // converges on high-symmetry cubic crystals (SrCaH12 with 14 atoms /
          // 42 reps fails). 'no' is the last-resort: skip ASR entirely so the
          // diagonalization still happens — the 3 acoustic modes won't be
          // pinned to ω=0, but that doesn't change the pass/fail screen.
          const asrLadder: Array<"crystal" | "simple" | "no"> = ["crystal", "simple", "no"];
          let dynmatSucceeded = false;
          let lastExit = -1;
          let lastStderr = "";

          if (!opts.callbacks.runQEBinary) {
            console.log(`[Staged-Relax] ${formula} Stage 4: dynmat.x not available (runQEBinary callback missing)`);
          } else for (const asr of asrLadder) {
            const dynmatInput = `&INPUT\n  fildyn = '${dynBasename}',\n  asr = '${asr}'\n/\n`;
            const dynmatFile = path.join(opts.jobDir, "dynmat_gamma.in");
            fs.writeFileSync(dynmatFile, dynmatInput);

            const dynmatResult = await opts.callbacks.runQEBinary("dynmat.x", dynmatFile, opts.jobDir, 60000);
            fs.writeFileSync(path.join(opts.jobDir, `dynmat_gamma_asr_${asr}.out`), dynmatResult.stdout);
            lastExit = dynmatResult.exitCode;
            lastStderr = dynmatResult.stderr;

            if (dynmatResult.exitCode !== 0) {
              console.log(`[Staged-Relax] ${formula} Stage 4: dynmat.x asr='${asr}' exit=${dynmatResult.exitCode}: ${dynmatResult.stderr.slice(-200)}`);
              continue;
            }

            // Parse frequencies from dynmat.x output
            // Format: "# mode   [cm-1]   [THz]  IR\n   1   123.45   3.678   0.123"
            const dynmatFreqs = parseGammaPhononFrequencies(dynmatResult.stdout);
            if (dynmatFreqs.length > 0) {
              frequencies = dynmatFreqs;
              console.log(`[Staged-Relax] ${formula} Stage 4: dynmat.x asr='${asr}' extracted ${dynmatFreqs.length} frequencies from ${dynBasename}`);
              dynmatSucceeded = true;
              break;
            }

            // Try tabular format: "   1   123.45   3.678   0.123"
            let inTable = false;
            for (const line of dynmatResult.stdout.split("\n")) {
              if (line.match(/#\s*mode\s+\[cm-1\]/i)) { inTable = true; continue; }
              if (inTable) {
                const parts = line.trim().split(/\s+/);
                if (parts.length >= 2 && !isNaN(parseFloat(parts[1]))) {
                  frequencies.push(parseFloat(parts[1]));
                } else if (parts.length < 2 || line.trim() === "") {
                  break;
                }
              }
            }
            if (frequencies.length > 0) {
              console.log(`[Staged-Relax] ${formula} Stage 4: dynmat.x asr='${asr}' tabular parse got ${frequencies.length} frequencies`);
              dynmatSucceeded = true;
              break;
            }

            console.log(`[Staged-Relax] ${formula} Stage 4: dynmat.x asr='${asr}' ran but produced no parseable frequencies. Output tail: ${dynmatResult.stdout.slice(-300)}`);
          }

          if (!dynmatSucceeded) {
            console.log(`[Staged-Relax] ${formula} Stage 4: all dynmat.x ASR variants failed (last exit=${lastExit}, stderr=${lastStderr.slice(-150)})`);
          }
        }
      } catch (dynErr: any) {
        console.log(`[Staged-Relax] ${formula} Stage 4: dynmat.x failed: ${dynErr.message?.slice(0, 100)}`);
      }
    }

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

/**
 * Extract the LAST ionic-step "max per-atom |F|" from QE pw.x output.
 *
 * QE prints (after each ionic step):
 *
 *     Forces acting on atoms (cartesian axes, Ry/au):
 *
 *         atom    1 type  1   force =     0.00012345    0.00023456   -0.00001234
 *         atom    2 type  2   force =    -0.00005678    0.00009876    0.00004321
 *         ...
 *         Total force =     0.00045678     Total SCF correction =     ...
 *
 * Returns max_i sqrt(Fx² + Fy² + Fz²) for the last block; null if absent.
 * Units: Ry/Bohr.
 */
function parseMaxPerAtomForce(stdout: string): number | null {
  const idx = stdout.lastIndexOf("Forces acting on atoms");
  if (idx < 0) return null;
  // Scan forward until "Total force" (end of the block)
  const end = stdout.indexOf("Total force", idx);
  const block = end > idx ? stdout.slice(idx, end) : stdout.slice(idx);

  const atomLineRe = /atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)/gi;
  let m: RegExpExecArray | null;
  let maxMag = 0;
  while ((m = atomLineRe.exec(block)) !== null) {
    const fx = parseFloat(m[1]);
    const fy = parseFloat(m[2]);
    const fz = parseFloat(m[3]);
    if (!Number.isFinite(fx) || !Number.isFinite(fy) || !Number.isFinite(fz)) continue;
    const mag = Math.sqrt(fx * fx + fy * fy + fz * fz);
    if (mag > maxMag) maxMag = mag;
  }
  return maxMag > 0 ? maxMag : null;
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
    if (val) result.totalEnergy = parseFloat(val[1]) * RY_TO_EV; // Ry -> eV
  }

  // Parse forces. QE prints two related but DIFFERENT quantities each ionic
  // step:
  //   (a) per-atom force vectors in "Forces acting on atoms (cartesian axes)"
  //       — each "atom N type T   force = Fx Fy Fz" line in Ry/Bohr units
  //   (b) "Total force = X" — L2 norm of the full 3N-dimensional force vector
  //
  // The BFGS forc_conv_thr in QE compares against the LARGEST per-atom
  // |F| (i.e., max_i sqrt(Fx²+Fy²+Fz²)). For an N-atom cell with similar
  // forces on every atom, the L2 norm is ≈ √N · max_per_atom — so reading
  // the "Total force" and comparing against forc_conv_thr fails the
  // Stage 1 check even when QE actually converged.
  //
  // Parse the per-atom block first; fall back to "Total force" if absent
  // (some QE versions omit the per-atom block when only printing summary).
  const maxPerAtom = parseMaxPerAtomForce(stdout);
  if (maxPerAtom != null) {
    result.maxForce = maxPerAtom;
  } else {
    const forceMatch = stdout.match(/Total force\s*=\s*([\d.]+)/g);
    if (forceMatch && forceMatch.length > 0) {
      const last = forceMatch[forceMatch.length - 1];
      const val = last.match(/([\d.]+)/);
      if (val) result.maxForce = parseFloat(val[0]);
    }
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
    if (val) result.totalEnergy = parseFloat(val[1]) * RY_TO_EV; // Ry -> eV
  }

  // Forces: prefer max per-atom |F| from the "Forces acting on atoms" block;
  // fall back to L2-norm "Total force" only if the per-atom block is absent.
  // QE's BFGS forc_conv_thr is per-atom max, so comparing against per-atom is
  // the physically correct test (see parseMaxPerAtomForce docstring).
  const maxPerAtomVcrelax = parseMaxPerAtomForce(stdout);
  if (maxPerAtomVcrelax != null) {
    result.maxForce = maxPerAtomVcrelax;
  } else {
    const forceMatch = stdout.match(/Total force\s*=\s*([\d.]+)/g);
    if (forceMatch && forceMatch.length > 0) {
      const last = forceMatch[forceMatch.length - 1];
      const val = last.match(/([\d.]+)/);
      if (val) result.maxForce = parseFloat(val[0]);
    }
  }

  // Pressure (from stress tensor output)
  const pressMatch = stdout.match(/P=\s*([-\d.]+)/g);
  if (pressMatch && pressMatch.length > 0) {
    const last = pressMatch[pressMatch.length - 1];
    const val = last.match(/([-\d.]+)/);
    if (val) result.pressure = parseFloat(val[0]);
  }

  // Parse final CELL_PARAMETERS block. QE writes one of three forms:
  //   CELL_PARAMETERS (alat=  7.50000000)  ← vectors are multiples of celldm(1) (in Bohr)
  //   CELL_PARAMETERS (bohr)
  //   CELL_PARAMETERS (angstrom)
  // The previous parser only checked for "bohr" and treated everything else
  // as angstrom — for the common alat case (default for vc-relax), it would
  // return latticeA = 1.0 Å for a cubic cell (vectors are unit-fractions
  // of celldm(1)) instead of the actual celldm(1)·BOHR_TO_ANG Å.
  const cellBlocks = stdout.match(/CELL_PARAMETERS\s*[{(]?\s*(?:angstrom|bohr|alat\s*=?\s*[-\d.]*)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:ATOMIC_POSITIONS|End|$|\n\s*\n))/gi);
  if (cellBlocks && cellBlocks.length > 0) {
    const lastCell = cellBlocks[cellBlocks.length - 1];
    const header = lastCell.split("\n")[0] ?? "";
    const lines = lastCell.split("\n").slice(1);
    const vectors: number[][] = [];
    for (const line of lines) {
      const nums = line.trim().split(/\s+/).map(Number).filter(n => !isNaN(n));
      if (nums.length === 3) vectors.push(nums);
    }
    if (vectors.length === 3) {
      // Detect unit from the header
      const headerLc = header.toLowerCase();
      const isBohr = headerLc.includes("bohr");
      // "alat= X.XX" — capture the celldm(1) value (in Bohr).
      const alatMatch = header.match(/alat\s*=?\s*([\d.]+)/i);
      const isAlat = alatMatch != null || (headerLc.includes("alat") && !headerLc.includes("angstrom"));
      const BOHR_TO_ANG = 0.529177210903;  // CODATA 2018, matches qe-worker.ts / phonon-calculator.ts / acbn0-pipeline.ts

      let cellVectorsAng: number[][];
      if (isAlat && alatMatch) {
        const alatBohr = parseFloat(alatMatch[1]);
        const alatAng = alatBohr * BOHR_TO_ANG;
        cellVectorsAng = vectors.map(v => v.map(x => x * alatAng));
      } else if (isBohr) {
        cellVectorsAng = vectors.map(v => v.map(x => x * BOHR_TO_ANG));
      } else {
        // angstrom (explicit or default for newer QE)
        cellVectorsAng = vectors;
      }
      result.cellVectors = cellVectorsAng;
      const aVec = cellVectorsAng[0];
      result.latticeA = Math.sqrt(aVec[0] ** 2 + aVec[1] ** 2 + aVec[2] ** 2);
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

  // QE ph.x Gamma-point dual-unit form (standard):
  //     freq (    1) =     -2.345678 [THz] =    -78.123456 [cm-1]
  //     omega( 1) =      -2.345678 [THz] =    -78.123456 [cm-1]
  //     omega(1-3) = ...    (QE 7.x degenerate-mode range form)
  // The previous regex `freq(N) = ([-\d.]+) ... cm-1` captured the FIRST
  // number on the line, which is the THz value (off by 33.36× from cm-1),
  // and then required "cm-1" immediately after — so for this format the
  // regex failed to match at all and frequencies came back empty.
  // QE 7.x emits degenerate modes as a RANGE — `freq(1-3) = ...` is one
  // line for 3 modes. Capture the mode-index bounds and push the frequency
  // (hi-lo+1) times so the parsed count equals the true 3N mode count.
  // Otherwise the Stage 4 `frequencies.length < expectedModes` gate fails
  // dynamically-stable high-symmetry structures whose modes are degenerate.
  // Single-mode lines `freq(1)` have no `hi` group and push once — QE emits
  // the range form OR the per-mode form, never both, so no double-count.
  const dualUnitPattern = /(?:freq|omega)\s*\(\s*(\d+)(?:\s*-\s*(\d+))?\s*\)\s*=\s*[-\d.eE+]+\s*\[THz\]\s*=\s*([-\d.eE+]+)\s*\[?\s*cm\^?-?1\]?/gi;
  let match: RegExpExecArray | null;
  while ((match = dualUnitPattern.exec(stdout)) !== null) {
    const v = parseFloat(match[3]);
    if (!Number.isFinite(v)) continue;
    const lo = parseInt(match[1], 10);
    const hi = match[2] ? parseInt(match[2], 10) : lo;
    const count = hi >= lo ? Math.min(hi - lo + 1, 64) : 1;
    for (let k = 0; k < count; k++) frequencies.push(v);
  }

  // Fallback: simple "freq( N) = X cm-1" or "omega( N) = X cm-1" (no THz).
  // Same degenerate-range expansion as above.
  if (frequencies.length === 0) {
    const simplePattern = /(?:freq|omega)\s*\(\s*(\d+)(?:\s*-\s*(\d+))?\s*\)\s*=\s*([-\d.eE+]+)\s*\[?\s*cm\^?-?1\]?/gi;
    while ((match = simplePattern.exec(stdout)) !== null) {
      const v = parseFloat(match[3]);
      if (!Number.isFinite(v)) continue;
      const lo = parseInt(match[1], 10);
      const hi = match[2] ? parseInt(match[2], 10) : lo;
      const count = hi >= lo ? Math.min(hi - lo + 1, 64) : 1;
      for (let k = 0; k < count; k++) frequencies.push(v);
    }
  }

  // Final fallback — tabular format some QE versions emit:
  //     Mode   1  frequency =    -23.456 cm-1
  if (frequencies.length === 0) {
    const modePattern = /Mode\s+\d+\s+frequency\s*=\s*([-\d.eE+]+)\s*cm/gi;
    while ((match = modePattern.exec(stdout)) !== null) {
      const v = parseFloat(match[1]);
      if (Number.isFinite(v)) frequencies.push(v);
    }
  }

  return frequencies;
}
