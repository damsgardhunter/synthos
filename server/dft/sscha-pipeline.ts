/**
 * SSCHA Anharmonic Phonon Pipeline — Node.js Orchestrator
 *
 * Stochastic Self-Consistent Harmonic Approximation (SSCHA) for computing
 * anharmonic phonon corrections to superconducting Tc predictions.
 *
 * Only triggers for publication-ready hydride candidates:
 *   - Residual force < 0.001 Ry/Bohr
 *   - Hydrogen-rich composition
 *   - Pressure >= 20 GPa
 *
 * Calls sscha-worker.py via child_process.execFile, parses JSON output.
 */

import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { execFile } from "child_process";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SSCHAResult {
  omegaLogAnharmonic: number;   // meV — anharmonic corrected omega_log
  lambdaAnharmonic: number | null; // anharmonic lambda (null if not computed)
  freeEnergy: number;           // Ry — SSCHA free energy
  tcCorrected: number | null;   // K — Tc with anharmonic correction (if computable)
  converged: boolean;
  iterations: number;
  nConfigs: number;
  temperatureK: number;
  totalForceCalcs: number;
  nImaginaryModes?: number;
  harmonicImaginaryCount?: number;
  anharmonicFreqsCm?: number[];
  /**
   * Method tier for SSCHA results:
   *   "sscha_sc"       — full self-consistent SSCHA minimization (best)
   *   "sscha_one_shot"  — single stochastic update without self-consistency (numpy fallback)
   *   "sscha_error"     — computation failed, partial results only
   */
  method: "sscha_sc" | "sscha_one_shot" | "sscha_error";
  warnings: string[];
  elapsedSeconds?: number;
}

export interface SSCHAGateCheck {
  eligible: boolean;
  reason?: string;
}

export interface SSCHAPipelineOptions {
  temperature?: number;          // K, default 300
  supercell?: [number, number, number]; // default [2,2,2]
  nConfigs?: number;             // default 100
  maxIterations?: number;        // default 20
  ecutwfc: number;               // Ry
  ecutrho?: number;              // Ry (default: 8 * ecutwfc)
  kgrid?: [number, number, number]; // default [4,4,4]
  pseudoDir: string;
  prefix: string;
  mpiNp?: number;                // MPI ranks, 0 = serial
  mpiLauncher?: string;          // default "mpirun"
  /** Harmonic electron-phonon coupling λ from EPW. Used as a fallback for
   *  the anharmonic-Tc estimate inside the worker. Without it, Tc uses
   *  λ=1.0 as a placeholder — which under-predicts Tc by 2-3× for strong-
   *  coupling hydrides. Pass the EPW λ here whenever it's available. */
  lambdaHarmonic?: number;
}

export interface SSCHAPipelineCallbacks {
  /** Run a QE binary; used only for logging context here, SSCHA-worker runs its own pw.x */
  runQEBinary?: (
    binary: string,
    inputFile: string,
    cwd: string,
    timeoutMs: number,
  ) => Promise<{ stdout: string; stderr: string; exitCode: number | null }>;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const __sschaDirname = typeof import.meta?.url === "string" ? path.dirname(fileURLToPath(import.meta.url)) : process.cwd();
const SSCHA_WORKER_SCRIPT = path.resolve(__sschaDirname, "sscha-worker.py");
const SSCHA_TOTAL_TIMEOUT_MS = 24 * 60 * 60 * 1000; // 24 hours
const PYTHON_BIN = process.env.SSCHA_PYTHON || "python3";

// Gate thresholds for SSCHA eligibility
const MAX_RESIDUAL_FORCE = 0.001;  // Ry/Bohr

// ---------------------------------------------------------------------------
// Gate check — is this candidate worth an SSCHA run?
// ---------------------------------------------------------------------------

/**
 * Determine whether SSCHA anharmonic corrections are warranted.
 *
 * Old gate: H-fraction >= 25% AND pressure >= 20 GPa.
 * Problem: filters out real anharmonic non-hydride candidates — sulfur
 * and selenium under pressure, lithium under pressure, borides, and any
 * system with a soft TA branch all have non-trivial anharmonic Tc corrections.
 *
 * New gate: trigger on ANY system where anharmonicity indicators are present:
 *   1. Force convergence (always required — unconverged structures can't run SSCHA)
 *   2. Phonon-based trigger (at least one must be true):
 *      a. Lowest acoustic phonon frequency below threshold (soft TA branch)
 *      b. Large Grüneisen parameter (strong volume-dependent phonon shifts)
 *      c. High hydrogen content under pressure (legacy hydride trigger)
 *      d. Anharmonicity index from physics-engine above threshold
 *      e. Light elements (Li, Be, B) under significant pressure
 *
 * @see I. Errea et al., PRL 114, 157004 (2015) — SSCHA for non-hydride (H3S was thought to be hydride, but sulfur anharmonicity matters too)
 * @see M. Borinaga et al., JPCM 28, 494001 (2016) — importance of anharmonicity beyond hydrides
 */
export function checkSSCHAEligibility(opts: {
  maxForce: number;
  elements: string[];
  counts: Record<string, number>;
  pressureGPa: number;
  /** Lowest acoustic phonon frequency (cm^-1), if available from gamma phonon */
  lowestAcousticFreq?: number;
  /** Grüneisen parameter (average), if available from physics-engine */
  gruneisenParam?: number;
  /** Anharmonicity index from PhononSpectrum (0-1) */
  anharmonicityIndex?: number;
  /** Soft mode score from PhononSpectrum (0-1) */
  softModeScore?: number;
}): SSCHAGateCheck {
  const { maxForce, elements, counts, pressureGPa } = opts;

  // Gate 1: Force convergence — always required
  if (maxForce > MAX_RESIDUAL_FORCE) {
    return {
      eligible: false,
      reason: `Residual force ${maxForce.toFixed(5)} > ${MAX_RESIDUAL_FORCE} Ry/Bohr — not converged enough for SSCHA`,
    };
  }

  // Gate 2: At least one anharmonicity indicator must trigger

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hCount = counts["H"] ?? 0;
  const hFraction = totalAtoms > 0 ? hCount / totalAtoms : 0;
  const reasons: string[] = [];

  // 2a. Soft acoustic branch: lowest acoustic mode below 50 cm^-1
  // indicates a system close to a structural instability
  if (opts.lowestAcousticFreq != null && opts.lowestAcousticFreq < 50 && opts.lowestAcousticFreq > 0) {
    reasons.push(`soft acoustic mode (${opts.lowestAcousticFreq.toFixed(1)} cm^-1 < 50)`);
  }

  // 2b. Large Grüneisen parameter: γ > 2.0 means phonon frequencies are
  // strongly volume-dependent → anharmonic potential is important
  if (opts.gruneisenParam != null && opts.gruneisenParam > 2.0) {
    reasons.push(`large Grüneisen parameter (γ=${opts.gruneisenParam.toFixed(2)} > 2.0)`);
  }

  // 2c. High anharmonicity index from physics-engine
  if (opts.anharmonicityIndex != null && opts.anharmonicityIndex > 0.35) {
    reasons.push(`high anharmonicity index (${opts.anharmonicityIndex.toFixed(2)} > 0.35)`);
  }

  // 2d. Soft mode score suggesting incipient instability
  if (opts.softModeScore != null && opts.softModeScore > 0.6) {
    reasons.push(`soft mode score (${opts.softModeScore.toFixed(2)} > 0.6)`);
  }

  // 2e. Hydrogen-rich under pressure (legacy hydride trigger, still valid)
  if (hFraction >= 0.25 && pressureGPa >= 20) {
    reasons.push(`H-rich under pressure (H=${(hFraction * 100).toFixed(0)}%, P=${pressureGPa} GPa)`);
  }

  // 2f. Light elements under high pressure (Li, Be, B compress strongly)
  const lightElements = ["Li", "Be", "B"];
  const hasLight = elements.some(el => lightElements.includes(el));
  if (hasLight && pressureGPa >= 30) {
    reasons.push(`light element (${elements.filter(el => lightElements.includes(el)).join(",")}) under ${pressureGPa} GPa`);
  }

  // 2g. Chalcogens under pressure (S, Se, Te — known anharmonic under compression)
  const chalcogens = ["S", "Se", "Te"];
  const hasChalcogen = elements.some(el => chalcogens.includes(el));
  if (hasChalcogen && pressureGPa >= 50) {
    reasons.push(`chalcogen (${elements.filter(el => chalcogens.includes(el)).join(",")}) under ${pressureGPa} GPa`);
  }

  if (reasons.length === 0) {
    return {
      eligible: false,
      reason: "No anharmonicity indicators triggered (no soft modes, low Grüneisen, no H under pressure, no light/chalcogen elements under compression)",
    };
  }

  return { eligible: true, reason: reasons.join("; ") };
}

// ---------------------------------------------------------------------------
// SSCHA worker runner
// ---------------------------------------------------------------------------

function runSSCHAWorker(args: string[], timeoutMs: number): Promise<{ stdout: string; stderr: string; exitCode: number | null }> {
  return new Promise((resolve) => {
    const startTime = Date.now();
    let stdout = "";
    let stderr = "";
    let resolved = false;

    const proc = execFile(
      PYTHON_BIN,
      [SSCHA_WORKER_SCRIPT, ...args],
      {
        maxBuffer: 100 * 1024 * 1024, // 100 MB
        timeout: timeoutMs,
      },
      (error, stdoutData, stderrData) => {
        if (resolved) return;
        resolved = true;
        stdout = stdoutData ?? "";
        stderr = stderrData ?? "";
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        if (error) {
          console.error(`[SSCHA] Python worker error after ${elapsed}s: ${error.message}`);
          resolve({
            stdout,
            stderr,
            exitCode: (error as any).code ?? -1,
          });
        } else {
          console.log(`[SSCHA] Python worker completed in ${elapsed}s`);
          resolve({ stdout, stderr, exitCode: 0 });
        }
      },
    );

    // Safety timeout (should not be needed since execFile has timeout, but belt-and-suspenders)
    const safetyTimer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        console.error(`[SSCHA] Safety timeout reached (${timeoutMs}ms)`);
        try { proc.kill("SIGKILL"); } catch {}
        resolve({ stdout, stderr: stderr + "\nSafety timeout reached", exitCode: -9 });
      }
    }, timeoutMs + 30_000);

    // Clean up timer if process finishes normally
    proc.on("exit", () => clearTimeout(safetyTimer));
  });
}

// ---------------------------------------------------------------------------
// Parse SSCHA JSON output
// ---------------------------------------------------------------------------

function parseSSCHAOutput(stdout: string): Partial<SSCHAResult> & { success?: boolean; error?: string } {
  // Find the JSON block in stdout (worker prints JSON at the end)
  // There may be stray log lines before the JSON
  const jsonStart = stdout.indexOf("{");
  const jsonEnd = stdout.lastIndexOf("}");

  if (jsonStart === -1 || jsonEnd === -1 || jsonEnd <= jsonStart) {
    return { success: false, error: "No JSON found in SSCHA worker output" };
  }

  try {
    const jsonStr = stdout.substring(jsonStart, jsonEnd + 1);
    const data = JSON.parse(jsonStr);
    return data;
  } catch (e: any) {
    return { success: false, error: `JSON parse error: ${e.message}` };
  }
}

// ---------------------------------------------------------------------------
// Allen-Dynes Tc from anharmonic omega_log
// ---------------------------------------------------------------------------

function allenDynesTc(lambda: number, omegaLogMeV: number, muStar: number): number {
  if (lambda < 0.01 || omegaLogMeV < 0.01) return 0;
  const omegaLogK = omegaLogMeV / 0.08617; // meV to K
  const exponent = -1.04 * (1 + lambda) / (lambda - muStar * (1 + 0.62 * lambda));
  if (exponent > 0 || exponent < -100) return 0;
  return (omegaLogK / 1.2) * Math.exp(exponent);
}

// ---------------------------------------------------------------------------
// Main pipeline orchestrator
// ---------------------------------------------------------------------------

export async function runSSCHAPipeline(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  jobDir: string,
  pressure: number,
  harmonicLambda: number | null,
  harmonicOmegaLog: number | null,
  options: SSCHAPipelineOptions,
  callbacks?: SSCHAPipelineCallbacks,
): Promise<SSCHAResult | null> {
  const warnings: string[] = [];
  const prefix = options.prefix || formula.replace(/[^a-zA-Z0-9]/g, "");

  console.log(`[SSCHA] Starting anharmonic phonon pipeline for ${formula}`);
  console.log(`[SSCHA]   P=${pressure} GPa, T=${options.temperature ?? 300} K`);
  console.log(`[SSCHA]   Harmonic: lambda=${harmonicLambda?.toFixed(3) ?? "N/A"}, omega_log=${harmonicOmegaLog?.toFixed(1) ?? "N/A"} meV`);

  // Verify worker script exists
  if (!fs.existsSync(SSCHA_WORKER_SCRIPT)) {
    console.error(`[SSCHA] Worker script not found: ${SSCHA_WORKER_SCRIPT}`);
    warnings.push("SSCHA worker script missing");
    return makeErrorResult(warnings, options);
  }

  // Verify .dyn files exist
  const dynPrefix = path.posix.join(jobDir, `${prefix}.dyn`);
  const dynFile1 = `${dynPrefix}1`;
  // Also check without posix (Windows compat)
  const dynPrefixWin = path.join(jobDir, `${prefix}.dyn`);
  const dynFile1Win = `${dynPrefixWin}1`;

  const dynExists = fs.existsSync(dynFile1) || fs.existsSync(dynFile1Win);
  if (!dynExists) {
    // Try matdyn prefix
    const altPrefix = path.join(jobDir, "matdyn");
    if (fs.existsSync(`${altPrefix}1`)) {
      console.log(`[SSCHA] Using matdyn prefix for dynamical matrices`);
    } else {
      console.error(`[SSCHA] No .dyn files found at ${dynPrefix}* or matdyn*`);
      warnings.push("No dynamical matrix files found — DFPT phonon must run first");
      return makeErrorResult(warnings, options);
    }
  }

  // Count nqirr from available .dyn files
  let nqirr = 0;
  const usedDynPrefix = fs.existsSync(dynFile1) || fs.existsSync(dynFile1Win)
    ? (fs.existsSync(dynFile1) ? dynPrefix : dynPrefixWin)
    : path.join(jobDir, "matdyn");

  for (let iq = 1; iq <= 100; iq++) {
    if (fs.existsSync(`${usedDynPrefix}${iq}`)) {
      nqirr = iq;
    } else {
      break;
    }
  }

  if (nqirr === 0) {
    console.error(`[SSCHA] No .dyn files could be counted`);
    warnings.push("Could not count dynamical matrix files");
    return makeErrorResult(warnings, options);
  }
  console.log(`[SSCHA] Found ${nqirr} irreducible q-points`);

  // Build SSCHA output directory
  const sschaOutDir = path.join(jobDir, "sscha_work");
  try {
    fs.mkdirSync(sschaOutDir, { recursive: true });
  } catch (e: any) {
    console.error(`[SSCHA] Failed to create work directory: ${e.message}`);
    warnings.push("Failed to create SSCHA work directory");
    return makeErrorResult(warnings, options);
  }

  // Build command line arguments for the Python worker
  const temperature = options.temperature ?? 300;
  const supercell = options.supercell ?? [2, 2, 2];
  const nConfigs = options.nConfigs ?? 100;
  const maxIterations = options.maxIterations ?? 20;
  const ecutrho = options.ecutrho ?? options.ecutwfc * 8;
  const kgrid = options.kgrid ?? [4, 4, 4];
  const mpiNp = options.mpiNp ?? parseInt(process.env.QE_MPI_RANKS ?? "0", 10);
  const mpiLauncher = options.mpiLauncher ?? process.env.QE_MPI_WRAPPER ?? "mpirun";

  const workerArgs = [
    "--dyn-prefix", usedDynPrefix,
    "--nqirr", String(nqirr),
    "--temperature", String(temperature),
    "--supercell", String(supercell[0]), String(supercell[1]), String(supercell[2]),
    "--nconfigs", String(nConfigs),
    "--prefix", prefix,
    "--pseudo-dir", options.pseudoDir,
    "--ecutwfc", String(options.ecutwfc),
    "--ecutrho", String(ecutrho),
    "--kgrid", String(kgrid[0]), String(kgrid[1]), String(kgrid[2]),
    "--outdir", sschaOutDir,
    "--max-iterations", String(maxIterations),
  ];

  if (mpiNp >= 2) {
    workerArgs.push("--mpi-np", String(mpiNp));
    workerArgs.push("--mpi-launcher", mpiLauncher);
  }

  // Pass harmonic λ if known — lets the worker compute a meaningful
  // anharmonic Tc estimate instead of falling back to its λ=1.0 placeholder
  // (which under-predicts hydride Tc by 2-3×). Caller may supply via the
  // positional `harmonicLambda` arg (typical EPW workflow) OR via
  // `options.lambdaHarmonic` (manual override). Positional wins.
  const lambdaForTc =
    typeof harmonicLambda === "number" && harmonicLambda > 0
      ? harmonicLambda
      : (typeof options.lambdaHarmonic === "number" && options.lambdaHarmonic > 0
        ? options.lambdaHarmonic
        : null);
  if (lambdaForTc !== null) {
    workerArgs.push("--lambda-harmonic", String(lambdaForTc));
  }

  // Determine pw.x path
  const pwBinary = process.env.QE_PW_BINARY || "/usr/local/bin/pw.x";
  workerArgs.push("--pw-binary", pwBinary);

  console.log(`[SSCHA] Launching Python worker: ${PYTHON_BIN} sscha-worker.py`);
  console.log(`[SSCHA]   supercell=${supercell.join("x")}, nconfigs=${nConfigs}, maxiter=${maxIterations}`);
  console.log(`[SSCHA]   MPI: ${mpiNp >= 2 ? `${mpiLauncher} -np ${mpiNp}` : "serial"}`);

  const startTime = Date.now();

  // Run the Python worker
  const workerResult = await runSSCHAWorker(workerArgs, SSCHA_TOTAL_TIMEOUT_MS);

  const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(1);

  // Log stderr (worker logs)
  if (workerResult.stderr) {
    const stderrLines = workerResult.stderr.split("\n").filter((l: string) => l.trim());
    for (const line of stderrLines.slice(-20)) {
      console.log(`[SSCHA] ${line}`);
    }
  }

  if (workerResult.exitCode !== 0) {
    console.error(`[SSCHA] Worker exited with code ${workerResult.exitCode} after ${elapsedSec}s`);
    warnings.push(`SSCHA worker failed (exit ${workerResult.exitCode})`);

    // Try to parse partial output
    if (workerResult.stdout) {
      const partial = parseSSCHAOutput(workerResult.stdout);
      if (partial.error) {
        warnings.push(partial.error);
      }
    }

    return makeErrorResult(warnings, options);
  }

  // Parse JSON output from worker
  const parsed = parseSSCHAOutput(workerResult.stdout);

  if (!parsed.success) {
    console.error(`[SSCHA] Worker reported failure: ${parsed.error ?? "unknown"}`);
    warnings.push(parsed.error ?? "SSCHA worker reported failure");
    return makeErrorResult(warnings, options);
  }

  // Compute corrected Tc if we have both anharmonic omega_log and harmonic lambda
  let tcCorrected: number | null = null;
  const omegaLogAnh = parsed.omegaLogAnharmonic ?? 0;

  if (omegaLogAnh > 0 && harmonicLambda && harmonicLambda > 0) {
    // Use harmonic lambda with anharmonic omega_log — first-order correction
    // The real correction would require re-running EPW with anharmonic phonons
    const lambdaEff = parsed.lambdaAnharmonic ?? harmonicLambda;
    tcCorrected = allenDynesTc(lambdaEff, omegaLogAnh, 0.10);
    console.log(`[SSCHA] Tc(anharmonic AD): ${tcCorrected.toFixed(1)} K (lambda=${lambdaEff.toFixed(3)}, omega_log=${omegaLogAnh.toFixed(1)} meV)`);

    if (harmonicOmegaLog && harmonicOmegaLog > 0) {
      const tcHarmonic = allenDynesTc(harmonicLambda, harmonicOmegaLog, 0.10);
      const ratio = tcCorrected / Math.max(tcHarmonic, 0.01);
      console.log(`[SSCHA] Tc ratio (anh/harm): ${ratio.toFixed(3)} — harmonic Tc=${tcHarmonic.toFixed(1)} K`);
      if (ratio > 1.5 || ratio < 0.5) {
        warnings.push(`Large anharmonic correction: Tc ratio=${ratio.toFixed(2)}`);
      }
    }
  }

  const result: SSCHAResult = {
    omegaLogAnharmonic: omegaLogAnh,
    lambdaAnharmonic: parsed.lambdaAnharmonic ?? null,
    freeEnergy: parsed.freeEnergy ?? 0,
    tcCorrected,
    converged: parsed.converged ?? false,
    iterations: parsed.iterations ?? 0,
    nConfigs: parsed.nConfigs ?? nConfigs,
    temperatureK: parsed.temperatureK ?? temperature,
    totalForceCalcs: parsed.totalForceCalcs ?? 0,
    nImaginaryModes: parsed.nImaginaryModes,
    harmonicImaginaryCount: parsed.harmonicImaginaryCount,
    anharmonicFreqsCm: parsed.anharmonicFreqsCm,
    method: ((parsed.method as string) === "SSCHA-full" || (parsed.method as string) === "sscha_sc") ? "sscha_sc" as const
           : ((parsed.method as string) === "SSCHA-fallback" || (parsed.method as string) === "sscha_one_shot") ? "sscha_one_shot" as const
           : "sscha_error" as const,
    warnings,
    elapsedSeconds: parsed.elapsedSeconds,
  };

  console.log(
    `[SSCHA] ${formula}: converged=${result.converged}, ` +
    `omega_log(anh)=${result.omegaLogAnharmonic.toFixed(1)} meV, ` +
    `Tc(corrected)=${tcCorrected?.toFixed(1) ?? "N/A"} K, ` +
    `${result.totalForceCalcs} force calcs, ` +
    `method=${result.method}`,
  );

  return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeErrorResult(warnings: string[], options: SSCHAPipelineOptions): SSCHAResult {
  return {
    omegaLogAnharmonic: 0,
    lambdaAnharmonic: null,
    freeEnergy: 0,
    tcCorrected: null,
    converged: false,
    iterations: 0,
    nConfigs: 0,
    temperatureK: options.temperature ?? 300,
    totalForceCalcs: 0,
    method: "sscha_error",
    warnings,
  };
}
