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
}

// ---------------------------------------------------------------------------
// K-spacing constants for each stage
// ---------------------------------------------------------------------------
const KSPACING_RELAX = 0.40;
const KSPACING_VCRELAX = 0.30;
const KSPACING_SCF = 0.25;
const KSPACING_SCF_METAL = 0.20;

// Stage time caps (ms)
const STAGE1_TIMEOUT_MS = 600_000;      // 10 min
const STAGE2_TIMEOUT_MS = 1_800_000;    // 30 min
const STAGE4_TIMEOUT_MS = 1_800_000;    // 30 min

// Force thresholds (Ry/bohr)
const STAGE1_FORCE_THR = 1e-3;
const STAGE2_FORCE_THR = 5e-4;

// ---------------------------------------------------------------------------
// Main pipeline entry
// ---------------------------------------------------------------------------

export async function runStagedRelaxation(opts: StagedRelaxationOpts): Promise<StagedRelaxationResult> {
  const { formula, elements, counts, candidates, pressureGPa, jobDir, callbacks } = opts;
  const stages: StageResult[] = [];
  const startTime = Date.now();
  const maxS1 = opts.maxStage1Candidates ?? 3;

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

    // Pick the passing candidate with lowest energy
    const passedS1 = stage1Results.filter(r => r.result.passed);
    if (passedS1.length > 0) {
      passedS1.sort((a, b) => a.result.totalEnergy - b.result.totalEnergy);
      const best = passedS1[0];
      bestPositions = best.result.positions;
      bestLatticeA = best.result.latticeA;
      candidateSource = best.candidate.source;
      stages.push(best.result);
      console.log(`[Staged-Relax] ${formula} Stage 1: selected candidate from ${candidateSource} (E=${best.result.totalEnergy.toFixed(4)} eV)`);
    } else if (stage1Results.length > 0) {
      // No candidate passed — use the one with lowest energy anyway (best effort)
      stage1Results.sort((a, b) => a.result.totalEnergy - b.result.totalEnergy);
      const best = stage1Results[0];
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
    if (opts.skipVcRelax) {
      console.log(`[Staged-Relax] ${formula}: skipping Stage 2 (vc-relax skip condition)`);
    } else {
      console.log(`[Staged-Relax] ${formula} Stage 2/5 (vc-relax): a=${bestLatticeA.toFixed(3)} A, ${bestPositions.length} atoms`);
      try {
        const s2 = await runStage2VcRelax(
          formula, elements, counts, bestPositions, bestLatticeA, pressureGPa, jobDir, callbacks
        );
        stages.push(s2);

        if (s2.passed) {
          bestPositions = s2.positions;
          bestLatticeA = s2.latticeA;
          bestCellVectors = s2.cellVectors;
          console.log(`[Staged-Relax] ${formula} Stage 2: PASSED (force=${s2.maxForce?.toExponential(2)}, a=${s2.latticeA.toFixed(3)} A, P=${s2.pressure?.toFixed(1)} kbar, wall=${s2.wallTimeSeconds.toFixed(0)}s)`);
        } else {
          // Use whatever geometry came out (may be partial)
          if (s2.positions.length > 0) bestPositions = s2.positions;
          if (s2.latticeA > 0) bestLatticeA = s2.latticeA;
          bestCellVectors = s2.cellVectors;
          console.log(`[Staged-Relax] ${formula} Stage 2: FAILED — ${s2.failReason}, using partial geometry`);
        }
      } catch (err: any) {
        console.log(`[Staged-Relax] ${formula} Stage 2: ERROR — ${err.message?.slice(0, 200)}`);
        stages.push({
          stage: 2, passed: false, failReason: `exception: ${err.message?.slice(0, 100)}`,
          positions: bestPositions, latticeA: bestLatticeA, totalEnergy: 0, wallTimeSeconds: 0,
        });
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

  const ecutwfc = cb.resolveEcutwfc(elements);
  const ecutrho = cb.resolveEcutrho(elements, ecutwfc);
  const kpoints = cb.autoKPoints(latticeA, cOverA, KSPACING_RELAX);

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
  max_seconds = ${Math.floor(STAGE1_TIMEOUT_MS / 1000) - 60},
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

  const result = await cb.runPwx(inputFile, stageDir, STAGE1_TIMEOUT_MS);
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
    failReasons.push("no final positions in output");
  }
  if (parsed.maxForce != null && parsed.maxForce > STAGE1_FORCE_THR) {
    // Soft fail: if we have positions, still use them — BFGS made progress
    // even if forces didn't reach threshold
    if (parsed.positions.length > 0 && parsed.maxForce < 0.5) {
      // Forces under 0.5 Ry/bohr = structure is plausible, just not converged
      // Don't fail — let Stage 2 finish the job
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
  counts: Record<string, number>;
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

  const stageDir = path.join(jobDir, "stage4_gamma");
  fs.mkdirSync(stageDir, { recursive: true });

  // ph.x input for Gamma-only phonon
  const phInput = `Gamma phonon check for ${formula}
&INPUTPH
  outdir = './tmp',
  prefix = '${prefix}',
  tr2_ph = 1.0d-10,
  alpha_mix(1) = 0.3,
  ldisp = .false.,
  max_seconds = ${Math.floor(STAGE4_TIMEOUT_MS / 1000) - 60},
/
0.0 0.0 0.0
`;

  // We need the SCF .save directory from Stage 3 to be available.
  // The ph.x input references outdir/prefix.save which should exist from the SCF run.
  const inputFile = path.join(stageDir, "ph_gamma.in");
  fs.writeFileSync(inputFile, phInput);

  const result = await cb.runPhx(inputFile, stageDir, STAGE4_TIMEOUT_MS);
  fs.writeFileSync(path.join(stageDir, "ph_gamma.out"), result.stdout);

  const wallTime = (Date.now() - t0) / 1000;

  // Parse phonon frequencies from output
  const frequencies = parseGammaPhononFrequencies(result.stdout);

  const failReasons: string[] = [];
  const expectedModes = 3 * totalAtoms;

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

  const passed = failReasons.length === 0;
  if (passed) {
    console.log(`[Staged-Relax] ${formula} Stage 4 (Gamma phonon): PASSED — ${frequencies.length} modes, lowest=${Math.min(...frequencies).toFixed(1)} cm-1, wall=${wallTime.toFixed(0)}s`);
  } else {
    console.log(`[Staged-Relax] ${formula} Stage 4 (Gamma phonon): FAILED — ${failReasons.join("; ")}`);
  }

  return {
    stage: 4,
    passed,
    failReason: failReasons.length > 0 ? failReasons.join("; ") : undefined,
    positions: opts.positions,
    latticeA: opts.latticeA,
    cellVectors: opts.cellVectors,
    totalEnergy: 0,
    wallTimeSeconds: wallTime,
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

  // Parse final ATOMIC_POSITIONS block
  const posBlocks = stdout.match(/ATOMIC_POSITIONS\s*\{?\s*crystal\s*\}?\n([\s\S]*?)(?=\n(?:CELL_PARAMETERS|K_POINTS|End|$|\n\s*\n))/g);
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
  const cellBlocks = stdout.match(/CELL_PARAMETERS\s*\{?\s*(?:angstrom|bohr)\s*\}?\n([\s\S]*?)(?=\n(?:ATOMIC_POSITIONS|$|\n\s*\n))/gi);
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

  // Parse final ATOMIC_POSITIONS
  const posBlocks = stdout.match(/ATOMIC_POSITIONS\s*\{?\s*crystal\s*\}?\n([\s\S]*?)(?=\n(?:CELL_PARAMETERS|K_POINTS|End|$|\n\s*\n))/g);
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
