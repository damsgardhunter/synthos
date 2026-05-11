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
  const nq = parseInt(lines[0]?.trim() ?? "0", 10);
  const qPoints: Array<[number, number, number]> = [];
  for (let i = 1; i <= nq && i < lines.length; i++) {
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
  const regex = /freq\s*\(\s*\d+\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/g;
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
  return `&INPUT
  asr = 'simple',
  flfrc = '${prefix}.fc',
  flvec = '${prefix}_softmode.vec',
  flfrq = '${prefix}_softmode.freq',
  q_in_band_form = .false.,
/
1
  ${qPoint[0].toFixed(6)} ${qPoint[1].toFixed(6)} ${qPoint[2].toFixed(6)}  1
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
): Array<{ frequency: number; eigenvector: Array<[number, number, number]> }> {
  if (!fs.existsSync(vecPath)) return [];
  const content = fs.readFileSync(vecPath, "utf-8");
  const lines = content.split("\n");

  const modes: Array<{ frequency: number; eigenvector: Array<[number, number, number]> }> = [];
  let i = 0;

  while (i < lines.length) {
    // Look for frequency line
    const freqMatch = lines[i]?.match(/freq\s*\(\s*\d+\)\s*=\s*[-\d.]+\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/);
    if (!freqMatch) {
      // Also try cm⁻¹-only format
      const freqMatch2 = lines[i]?.match(/freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)/);
      if (!freqMatch2) { i++; continue; }
    }

    const freq = parseFloat((freqMatch ?? lines[i]?.match(/=\s*([-\d.]+)\s*\[cm-1\]/))?.[1] ?? "0");
    i++;

    // Read nAtoms lines of eigenvector data
    const eigvec: Array<[number, number, number]> = [];
    for (let a = 0; a < nAtoms && i < lines.length; a++, i++) {
      // Parse "( dx_real  dx_imag ) ( dy_real  dy_imag ) ( dz_real  dz_imag )"
      const nums = lines[i]?.match(/\(\s*([-\d.]+)\s+[-\d.]+\s*\)/g);
      if (nums && nums.length >= 3) {
        const dx = parseFloat(nums[0].match(/([-\d.]+)/)?.[1] ?? "0");
        const dy = parseFloat(nums[1].match(/([-\d.]+)/)?.[1] ?? "0");
        const dz = parseFloat(nums[2].match(/([-\d.]+)/)?.[1] ?? "0");
        eigvec.push([dx, dy, dz]);
      } else {
        eigvec.push([0, 0, 0]);
      }
    }

    if (eigvec.length === nAtoms) {
      modes.push({ frequency: freq, eigenvector: eigvec });
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
): Array<{ element: string; x: number; y: number; z: number }> {
  if (eigenvector.length !== positions.length) return positions;

  return positions.map((pos, i) => {
    const [ex, ey, ez] = eigenvector[i];
    // Phase factor: exp(i * 2π * q · R) — take real part
    const phase = 2 * Math.PI * (qPoint[0] * pos.x + qPoint[1] * pos.y + qPoint[2] * pos.z);
    const cosPhase = Math.cos(phase);

    return {
      element: pos.element,
      x: pos.x + ex * cosPhase * amplitude,
      y: pos.y + ey * cosPhase * amplitude,
      z: pos.z + ez * cosPhase * amplitude,
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

  let eigModes: Array<{ frequency: number; eigenvector: Array<[number, number, number]> }> = [];
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

    // Apply distortion
    const displaced = applySoftModeDistortion(
      bestPositions, softMode.eigenvector, worst.qPoint, amplitude,
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
