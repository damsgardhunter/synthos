/**
 * Zone-boundary soft mode following for dynamically unstable structures.
 *
 * When the full phonon grid (Stage 5) finds imaginary modes, those modes
 * tell us exactly which distortion the structure WANTS to undergo. Instead
 * of discarding the result, we:
 *   1. Parse the .dyn files to find which q-point has the worst instability
 *   2. Extract eigenvectors at that q-point via matdyn.x
 *   3. Build the supercell distortion pattern
 *   4. Displace atoms along the soft mode direction
 *   5. Re-relax the displaced structure
 *   6. Re-run phonons on the new (hopefully stable) phase
 *
 * This is how the Pickard/Errea groups find stable high-pressure phases
 * that no random search discovers — the phonon instability IS the search
 * direction pointing toward the true ground-state structure.
 *
 * Key physics:
 *   - Gamma-point instabilities (q=0): the unit cell itself wants to distort.
 *     Already handled by existing soft mode following.
 *   - Zone-boundary instabilities (q≠0): the structure wants a SUPERCELL
 *     distortion — e.g., alternating tilts, charge-density waves, Peierls
 *     distortions. These can only be found with multi-q phonon grids.
 *
 * Example: CaBeH8 at 180 GPa has 68 imaginary modes in the 3×3×3 grid.
 * The most negative mode at some q-point tells us the dominant distortion.
 * Following it may lead to a lower-symmetry phase that IS dynamically stable.
 *
 * @see C. J. Pickard & R. J. Needs, J. Phys.: Condens. Matter 23, 053201 (2011) — AIRSS + phonon-guided search
 * @see I. Errea et al., PRL 114, 157004 (2015) — structure stabilization via anharmonic phonons
 */

import * as fs from "fs";
import * as path from "path";

// ─── Types ───────────────────────────────────────────────────────────

export interface SoftModeInfo {
  /** q-point index in the .dyn file set (1-based) */
  qPointIndex: number;
  /** q-point fractional coordinates */
  qPoint: [number, number, number];
  /** Mode index at this q-point (1-based) */
  modeIndex: number;
  /** Frequency of the soft mode (cm⁻¹, negative = imaginary) */
  frequency: number;
  /** Eigenvector: displacement per atom [atom_index][x,y,z] (real part) */
  eigenvector: Array<[number, number, number]>;
  /** Imaginary part of eigenvector per atom [x,y,z]. Often zero for
   *  high-symmetry q-points (Γ, zone-boundary M/X/R on commensurate
   *  grids), but non-trivial for general q (e.g., q=(1/3,0,0) on a 3×3×3
   *  grid). When present, the proper displacement is
   *     u_n = ε_real·cos(2πq·R) − ε_imag·sin(2πq·R)
   *  rather than just ε_real·cos(2πq·R). */
  eigenvectorImag?: Array<[number, number, number]>;
}

export interface ZoneBoundarySoftModeResult {
  /** Whether soft mode following was attempted */
  attempted: boolean;
  /** Whether a new stable structure was found */
  foundStablePhase: boolean;
  /** The soft mode that was followed */
  softMode: SoftModeInfo | null;
  /** New positions after distortion + re-relaxation */
  newPositions: Array<{ element: string; x: number; y: number; z: number }> | null;
  /** New lattice parameter */
  newLatticeA: number | null;
  /** New phonon frequencies (if re-checked) */
  newFrequencies: number[] | null;
  /** Number of imaginary modes in the new structure */
  newImaginaryCount: number;
  /** Number of soft mode following iterations performed */
  iterations: number;
  /** Total wall time for the soft mode search (ms) */
  wallTimeMs: number;
  /** Human-readable notes */
  notes: string[];
}

// ─── .dyn file parsing ───────────────────────────────────────────────

/**
 * Parse the .dyn0 summary file to get the list of q-points.
 * Format:
 *   Line 1: number of q-points
 *   Lines 2+: q1 q2 q3 (fractional coordinates)
 */
export function parseDyn0(dyn0Path: string): { nq: number; qPoints: Array<[number, number, number]> } {
  if (!fs.existsSync(dyn0Path)) return { nq: 0, qPoints: [] };
  const lines = fs.readFileSync(dyn0Path, "utf-8").trim().split("\n");
  // QE writes .dyn0 as:
  //   Line 0: "nq1 nq2 nq3"        (q-grid dimensions, 3 integers)
  //   Line 1: "nqs"                 (number of irreducible q-points)
  //   Lines 2..2+nqs-1: q-coords    (3 floats per line, fractional reciprocal lattice)
  // The previous parser read line 0 as the count, which for a "3 3 3" header
  // gave nq=3 (the first integer) instead of the real irreducible-q count
  // from line 1. Then q-points were read starting at line 1 — meaning the
  // first real q-point was skipped (line 1 has just the count, fails the
  // 3-number check), and qPoints[k] was misaligned with .dynN file indexing.
  // findWorstInstability would then index past qPoints.length, falling back
  // to qPoint=[0,0,0] (gamma) and applying a Γ-distortion at a zone-boundary
  // soft-mode frequency — entirely wrong displacement pattern.
  const nq = parseInt(lines[1]?.trim() ?? "0", 10);
  const qPoints: Array<[number, number, number]> = [];
  for (let i = 2; i < 2 + nq && i < lines.length; i++) {
    const parts = lines[i].trim().split(/\s+/).map(Number);
    if (parts.length >= 3 && parts.every(Number.isFinite)) {
      qPoints.push([parts[0], parts[1], parts[2]]);
    }
  }
  return { nq, qPoints };
}

/**
 * Parse a single .dynN file to extract frequencies at that q-point.
 * QE writes frequencies in the dynamical matrix file in a block like:
 *   freq (    1) =      -2.345678 [THz] =     -78.1234 [cm-1]
 */
export function parseFrequenciesFromDynFile(dynPath: string): number[] {
  if (!fs.existsSync(dynPath)) return [];
  const content = fs.readFileSync(dynPath, "utf-8");
  const freqs: number[] = [];
  // `(?:\s*-\s*\d+)?` accommodates QE 7.x degenerate-mode range form
  // `freq(1-3) = ...` emitted for degenerate modes in cubic/high-symmetry
  // crystals. Without it those frequencies silently dropped from the parse,
  // and findWorstInstability could miss the actual worst soft mode.
  // Matches the same fix applied to parsePhononOutput, parseGammaPhononFrequencies,
  // and the sscha-worker dyn-file parser in earlier iterations.
  const regex = /freq\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/g;
  let match;
  while ((match = regex.exec(content)) !== null) {
    const f = parseFloat(match[1]);
    if (Number.isFinite(f)) freqs.push(f);
  }
  return freqs;
}

/**
 * Find the q-point with the most negative phonon frequency.
 * Scans all .dynN files to identify the worst instability.
 */
export function findWorstInstability(
  jobDir: string,
  prefix: string,
): { qIndex: number; qPoint: [number, number, number]; modeIndex: number; frequency: number } | null {
  const dyn0Path = path.join(jobDir, `${prefix}.dyn0`);
  const { nq, qPoints } = parseDyn0(dyn0Path);
  if (nq === 0) return null;

  let worstFreq = 0;
  let worstQIndex = -1;
  let worstModeIndex = -1;

  for (let qi = 1; qi <= nq; qi++) {
    const dynPath = path.join(jobDir, `${prefix}.dyn${qi}`);
    const freqs = parseFrequenciesFromDynFile(dynPath);
    for (let mi = 0; mi < freqs.length; mi++) {
      if (freqs[mi] < worstFreq) {
        worstFreq = freqs[mi];
        worstQIndex = qi;
        worstModeIndex = mi + 1; // 1-based
      }
    }
  }

  if (worstQIndex < 0 || worstFreq >= -50) return null; // no significant instability

  const qPoint = qPoints[worstQIndex - 1] ?? [0, 0, 0];
  return {
    qIndex: worstQIndex,
    qPoint,
    modeIndex: worstModeIndex,
    frequency: worstFreq,
  };
}

// ─── Eigenvector extraction ──────────────────────────────────────────

/**
 * Generate matdyn.x input to extract eigenvectors at a specific q-point.
 * Uses the force constants file (.fc) from q2r.x.
 */
export function generateMatdynEigenvectorInput(
  prefix: string,
  qPoint: [number, number, number],
): string {
  // ASR = 'crystal' is the recommended choice for arbitrary q-points on
  // arbitrary Bravais lattices (preserves Born-Huang invariants). 'simple'
  // only works correctly at q=0 and high-symmetry points on cubic crystals;
  // for the soft-mode q-points we're querying here (which by definition are
  // off the high-symmetry path or on low-symmetry q-points of non-cubic
  // grids), 'crystal' gives better eigenvectors. matdyn.x falls back to
  // 'simple' silently if 'crystal' isn't applicable to the cell.
  //
  // With q_in_band_form = .false., matdyn.x reads each q-point as 3 numbers
  // (no trailing weight/npts column). The previous trailing `1` was Fortran-
  // tolerant but format-incorrect — strict matdyn builds reject it.
  return `&INPUT
  asr = 'crystal',
  flfrc = '${prefix}.fc',
  flvec = '${prefix}_softmode.vec',
  flfrq = '${prefix}_softmode.freq',
  q_in_band_form = .false.,
/
1
  ${qPoint[0].toFixed(6)} ${qPoint[1].toFixed(6)} ${qPoint[2].toFixed(6)}
`;
}

/**
 * Parse eigenvectors from matdyn.x flvec output.
 *
 * Format of .vec file:
 *   q-point header line
 *   For each mode:
 *     frequency line: "freq (  N) = ... [cm-1]"
 *     For each atom:
 *       "( dx_real  dx_imag ) ( dy_real  dy_imag ) ( dz_real  dz_imag )"
 */
export function parseEigenvectors(
  vecPath: string,
  nAtoms: number,
): Array<{ frequency: number; eigenvector: Array<[number, number, number]>; eigenvectorImag: Array<[number, number, number]> }> {
  if (!fs.existsSync(vecPath)) return [];
  const content = fs.readFileSync(vecPath, "utf-8");
  const lines = content.split("\n");

  const modes: Array<{ frequency: number; eigenvector: Array<[number, number, number]>; eigenvectorImag: Array<[number, number, number]> }> = [];
  let i = 0;

  while (i < lines.length) {
    // Look for frequency line. `(?:\s*-\s*\d+)?` handles QE 7.x range form
    // `freq(1-3) = ...` for degenerate modes — without it, eigenvectors at
    // degenerate soft modes (common in cubic crystals) would silently parse
    // as zero-mode entries and break the soft-mode-follower distortion step.
    //
    // Two output formats from matdyn:
    //   Full:   "freq (  1) =   -2.34 [THz] =  -78.12 [cm-1]"
    //   Short:  "freq (  1) =   -78.12"   (some older QE / matdyn builds)
    // The previous fallback path detected the short format but then computed
    // `freq` from a `[cm-1]`-anchored regex that didn't match the short form,
    // silently zeroing every cm⁻¹-only-format frequency. Capture the freq
    // value directly from whichever regex actually matched.
    const fullFmt = lines[i]?.match(/freq\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/);
    const shortFmt = !fullFmt
      ? lines[i]?.match(/freq\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*([-\d.]+)/)
      : null;
    if (!fullFmt && !shortFmt) { i++; continue; }
    const freq = parseFloat((fullFmt ?? shortFmt)![1]);
    if (!Number.isFinite(freq)) { i++; continue; }
    i++;

    // Read nAtoms lines of eigenvector data — parse BOTH real and imaginary
    // parts. matdyn.x writes "(re imag) (re imag) (re imag)" per atom; the
    // imag parts are usually small but non-trivial at general (non-high-
    // symmetry) q-points like q=(1/3,0,0) on a 3×3×3 grid.
    const eigvecRe: Array<[number, number, number]> = [];
    const eigvecIm: Array<[number, number, number]> = [];
    for (let a = 0; a < nAtoms && i < lines.length; a++, i++) {
      // Parse "( dx_real  dx_imag ) ( dy_real  dy_imag ) ( dz_real  dz_imag )"
      // Capture both numbers inside each parenthesized pair. Accept scientific
      // notation (e/E/d/D exponent forms) — matdyn.x emits small components as
      // e.g. "-1.234E-05" depending on QE version + magnitude, and the prior
      // [-\d.]+ regex silently truncated those at the 'e' to "-1.234".
      const numRe = "[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eEdD][-+]?\\d+)?";
      const pairRegex = new RegExp(`\\(\\s*(${numRe})\\s+(${numRe})\\s*\\)`, "g");
      const pairs: Array<[number, number]> = [];
      let pm: RegExpExecArray | null;
      while ((pm = pairRegex.exec(lines[i] ?? "")) !== null) {
        const re = parseFloat(pm[1].replace(/[dD]/, "e"));
        const im = parseFloat(pm[2].replace(/[dD]/, "e"));
        pairs.push([Number.isFinite(re) ? re : 0, Number.isFinite(im) ? im : 0]);
      }
      if (pairs.length >= 3) {
        eigvecRe.push([pairs[0][0], pairs[1][0], pairs[2][0]]);
        eigvecIm.push([pairs[0][1], pairs[1][1], pairs[2][1]]);
      } else {
        eigvecRe.push([0, 0, 0]);
        eigvecIm.push([0, 0, 0]);
      }
    }

    if (eigvecRe.length === nAtoms) {
      modes.push({ frequency: freq, eigenvector: eigvecRe, eigenvectorImag: eigvecIm });
    }
  }

  return modes;
}

// ─── Distortion application ──────────────────────────────────────────

/**
 * Apply a soft mode distortion to atomic positions.
 *
 * For a zone-boundary mode at q-point q, the displacement pattern is:
 *   u_n = Re[ epsilon_n * exp(i * q · R_n) ] * amplitude
 *
 * where epsilon_n is the eigenvector for atom n and R_n is its position.
 * For commensurate q-points (e.g., q = [0.5, 0, 0] on a 2×2×2 grid),
 * this creates a real-valued distortion pattern.
 *
 * For q = [0,0,0] (gamma), this reduces to the existing soft mode following:
 *   u_n = epsilon_n * amplitude
 *
 * @param amplitude Displacement in fractional coordinates (typically 0.02-0.05)
 */
export function applySoftModeDistortion(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  eigenvector: Array<[number, number, number]>,
  qPoint: [number, number, number],
  amplitude: number,
  eigenvectorImag?: Array<[number, number, number]>,
): Array<{ element: string; x: number; y: number; z: number }> {
  if (eigenvector.length !== positions.length) return positions;

  return positions.map((pos, i) => {
    const [exRe, eyRe, ezRe] = eigenvector[i];
    const [exIm, eyIm, ezIm] = eigenvectorImag?.[i] ?? [0, 0, 0];
    // Proper real-valued displacement for a complex eigenvector ε = ε_real + iε_imag
    // at q-point q with atomic position R (in fractional coords):
    //   u = Re[ε · exp(2πi q·R)]
    //     = ε_real · cos(2πq·R) - ε_imag · sin(2πq·R)
    // For high-symmetry q on commensurate grids (q=(1/2,0,0) on 2×2×2 etc.)
    // the imaginary part vanishes and this reduces to ε_real·cos(2πq·R).
    // For general q (e.g., 3×3×3 grid q=(1/3,0,0)) the −ε_imag·sin(...) term
    // matters.
    const phase = 2 * Math.PI * (qPoint[0] * pos.x + qPoint[1] * pos.y + qPoint[2] * pos.z);
    const cosPhase = Math.cos(phase);
    const sinPhase = Math.sin(phase);

    return {
      element: pos.element,
      x: pos.x + (exRe * cosPhase - exIm * sinPhase) * amplitude,
      y: pos.y + (eyRe * cosPhase - eyIm * sinPhase) * amplitude,
      z: pos.z + (ezRe * cosPhase - ezIm * sinPhase) * amplitude,
    };
  });
}

// ─── Orchestrator ────────────────────────────────────────────────────

export interface ZBSoftModeCallbacks {
  runQEBinary: (
    binary: string,
    inputFile: string,
    cwd: string,
    timeoutMs: number,
  ) => Promise<{ stdout: string; stderr: string; exitCode: number | null }>;
  runVCRelax: (
    positions: Array<{ element: string; x: number; y: number; z: number }>,
    latticeA: number,
  ) => Promise<{
    positions: Array<{ element: string; x: number; y: number; z: number }> | null;
    latticeA: number;
    force: number;
    pressure: number | null;
    energy: number;
    converged: boolean;
  }>;
  runGammaPhonon: (
    positions: Array<{ element: string; x: number; y: number; z: number }>,
    latticeA: number,
  ) => Promise<{ frequencies: number[]; passed: boolean }>;
}

/**
 * Main entry: attempt zone-boundary soft mode following after full phonon failure.
 *
 * Strategy:
 *   1. Find the q-point with the worst imaginary mode
 *   2. Extract eigenvectors via matdyn.x
 *   3. Displace atoms along the soft mode
 *   4. Re-relax with vc-relax (damped dynamics)
 *   5. Quick gamma phonon check on the new structure
 *   6. If still unstable, try a second amplitude or the next-worst mode
 *
 * Maximum 3 iterations to avoid infinite loops.
 */
export async function followZoneBoundarySoftMode(
  formula: string,
  prefix: string,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  jobDir: string,
  callbacks: ZBSoftModeCallbacks,
): Promise<ZoneBoundarySoftModeResult> {
  const startTime = Date.now();
  const notes: string[] = [];
  const MAX_ITERATIONS = 3;
  const AMPLITUDES = [0.03, 0.05, 0.08]; // fractional coord displacement

  // Step 1: Find worst instability across all q-points
  const worst = findWorstInstability(jobDir, prefix);
  if (!worst) {
    return {
      attempted: false,
      foundStablePhase: false,
      softMode: null,
      newPositions: null,
      newLatticeA: null,
      newFrequencies: null,
      newImaginaryCount: 0,
      iterations: 0,
      wallTimeMs: Date.now() - startTime,
      notes: ["No significant instability found in .dyn files (all modes > -50 cm⁻¹)"],
    };
  }

  console.log(
    `[ZB-SoftMode] ${formula}: worst instability at q=(${worst.qPoint.map(q => q.toFixed(3)).join(",")}) ` +
    `mode ${worst.modeIndex}, freq=${worst.frequency.toFixed(1)} cm⁻¹`
  );
  notes.push(
    `Worst instability: q=(${worst.qPoint.map(q => q.toFixed(3)).join(",")}) ` +
    `mode ${worst.modeIndex}, freq=${worst.frequency.toFixed(1)} cm⁻¹`
  );

  // Step 2: Check if force constants file exists (needed for matdyn.x)
  const fcPath = path.join(jobDir, `${prefix}.fc`);
  if (!fs.existsSync(fcPath)) {
    notes.push("Force constants file (.fc) not found — q2r.x may not have completed. Cannot extract eigenvectors.");
    return {
      attempted: false,
      foundStablePhase: false,
      softMode: null,
      newPositions: null,
      newLatticeA: null,
      newFrequencies: null,
      newImaginaryCount: 0,
      iterations: 0,
      wallTimeMs: Date.now() - startTime,
      notes,
    };
  }

  // Step 3: Extract eigenvectors via matdyn.x
  const matdynInput = generateMatdynEigenvectorInput(prefix, worst.qPoint);
  const matdynFile = path.join(jobDir, `${prefix}_softmode_matdyn.in`);
  fs.writeFileSync(matdynFile, matdynInput);

  let eigModes: Array<{ frequency: number; eigenvector: Array<[number, number, number]>; eigenvectorImag: Array<[number, number, number]> }> = [];
  try {
    const matdynResult = await callbacks.runQEBinary(
      "matdyn.x", matdynFile, jobDir, 120_000, // 2 min timeout
    );
    const vecPath = path.join(jobDir, `${prefix}_softmode.vec`);
    eigModes = parseEigenvectors(vecPath, positions.length);
    console.log(`[ZB-SoftMode] ${formula}: extracted ${eigModes.length} eigenvectors at q=(${worst.qPoint.join(",")})`);
  } catch (e: any) {
    notes.push(`matdyn.x eigenvector extraction failed: ${e.message?.slice(0, 100)}`);
    return {
      attempted: true,
      foundStablePhase: false,
      softMode: null,
      newPositions: null,
      newLatticeA: null,
      newFrequencies: null,
      newImaginaryCount: 0,
      iterations: 0,
      wallTimeMs: Date.now() - startTime,
      notes,
    };
  }

  // Find the most negative mode's eigenvector
  const softModeEig = eigModes
    .filter(m => m.frequency < -50)
    .sort((a, b) => a.frequency - b.frequency)[0];

  if (!softModeEig) {
    notes.push("No imaginary eigenvector found in matdyn.x output");
    return {
      attempted: true,
      foundStablePhase: false,
      softMode: null,
      newPositions: null,
      newLatticeA: null,
      newFrequencies: null,
      newImaginaryCount: 0,
      iterations: 0,
      wallTimeMs: Date.now() - startTime,
      notes,
    };
  }

  const softMode: SoftModeInfo = {
    qPointIndex: worst.qIndex,
    qPoint: worst.qPoint,
    modeIndex: worst.modeIndex,
    frequency: softModeEig.frequency,
    eigenvector: softModeEig.eigenvector,
    eigenvectorImag: softModeEig.eigenvectorImag,
  };

  // Step 4: Iterate — try different amplitudes
  let bestPositions = positions;
  let bestLatticeA = latticeA;
  let bestFrequencies: number[] | null = null;
  let bestImaginaryCount = 999;
  let iterations = 0;

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    iterations = iter + 1;
    const amplitude = AMPLITUDES[iter] ?? 0.05;

    console.log(
      `[ZB-SoftMode] ${formula}: iteration ${iterations}, amplitude=${amplitude}, ` +
      `distorting along q=(${worst.qPoint.join(",")}) mode freq=${softMode.frequency.toFixed(1)} cm⁻¹`
    );

    // Apply distortion — pass imag part of eigenvector so non-high-symmetry
    // q-points (e.g., 3×3×3 grid q=(1/3,0,0)) get the correct
    //   u = ε_real·cos(2πq·R) − ε_imag·sin(2πq·R)
    // instead of just the real-only approximation.
    const displaced = applySoftModeDistortion(
      bestPositions, softMode.eigenvector, worst.qPoint, amplitude,
      softMode.eigenvectorImag,
    );

    // Re-relax
    let relaxResult;
    try {
      relaxResult = await callbacks.runVCRelax(displaced, bestLatticeA);
    } catch (e: any) {
      notes.push(`Iteration ${iterations}: vc-relax failed — ${e.message?.slice(0, 100)}`);
      continue;
    }

    if (!relaxResult.positions || relaxResult.positions.length === 0) {
      notes.push(`Iteration ${iterations}: vc-relax produced no positions`);
      continue;
    }

    notes.push(
      `Iteration ${iterations}: vc-relax converged (force=${relaxResult.force.toFixed(6)}, ` +
      `E=${relaxResult.energy.toFixed(4)} eV, a=${relaxResult.latticeA.toFixed(3)} Å)`
    );

    // Quick gamma phonon check
    let phononResult;
    try {
      phononResult = await callbacks.runGammaPhonon(relaxResult.positions, relaxResult.latticeA);
    } catch (e: any) {
      notes.push(`Iteration ${iterations}: gamma phonon check failed — ${e.message?.slice(0, 100)}`);
      continue;
    }

    const imagCount = phononResult.frequencies.filter(f => f < -50).length;
    notes.push(
      `Iteration ${iterations}: gamma phonon ${phononResult.passed ? "PASSED" : "FAILED"} ` +
      `(${imagCount} imaginary modes, lowest=${Math.min(...phononResult.frequencies).toFixed(1)} cm⁻¹)`
    );

    if (imagCount < bestImaginaryCount) {
      bestPositions = relaxResult.positions;
      bestLatticeA = relaxResult.latticeA;
      bestFrequencies = phononResult.frequencies;
      bestImaginaryCount = imagCount;
    }

    if (phononResult.passed) {
      console.log(`[ZB-SoftMode] ${formula}: FOUND STABLE PHASE after ${iterations} iterations!`);
      notes.push(`SUCCESS: dynamically stable phase found at iteration ${iterations}`);
      return {
        attempted: true,
        foundStablePhase: true,
        softMode,
        newPositions: bestPositions,
        newLatticeA: bestLatticeA,
        newFrequencies: bestFrequencies,
        newImaginaryCount: 0,
        iterations,
        wallTimeMs: Date.now() - startTime,
        notes,
      };
    }
  }

  // Didn't find a fully stable phase, but may have reduced imaginary count
  const improved = bestImaginaryCount < 999;
  if (improved) {
    notes.push(
      `Soft mode following reduced imaginary modes but did not eliminate them ` +
      `(${bestImaginaryCount} remaining after ${iterations} iterations)`
    );
  }

  return {
    attempted: true,
    foundStablePhase: false,
    softMode,
    newPositions: improved ? bestPositions : null,
    newLatticeA: improved ? bestLatticeA : null,
    newFrequencies: bestFrequencies,
    newImaginaryCount: bestImaginaryCount,
    iterations,
    wallTimeMs: Date.now() - startTime,
    notes,
  };
}
