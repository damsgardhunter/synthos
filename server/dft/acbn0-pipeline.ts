/**
 * ACBN0 First-Principles μ* Pipeline
 *
 * Computes the Coulomb pseudopotential μ* from first principles using
 * QE's hp.x (Hubbard parameter calculation via DFPT linear response).
 *
 * Workflow:
 *   1. SCF with DFT+U (initial U=0 or small)
 *   2. hp.x — compute Hubbard U from linear response
 *   3. Parse hp.x output for screened Coulomb parameters
 *   4. Compute μ* via Morel-Anderson retardation:
 *      μ* = μ / (1 + μ * ln(E_F / ω_D))
 *      where μ = N(E_F) * V_c (bare Coulomb interaction at Fermi level)
 *   5. Optionally iterate for self-consistent U
 *
 * Returns ACBN0Result with first-principles μ*, Hubbard U per element,
 * and screening parameters.
 */

import * as fs from "fs";
import * as path from "path";
import { getElementData } from "../learning/elemental-data";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ACBN0Result {
  muStar: number;                // Coulomb pseudopotential μ* (dimensionless, typically 0.10-0.16)
  muBare: number;                // Bare Coulomb μ before retardation
  screeningLength: number;       // Thomas-Fermi screening length (Bohr)
  hubbardU: Record<string, number>; // Hubbard U per element (eV)
  dielectricConstant?: number;   // Estimated static dielectric constant
  nEF: number;                   // N(E_F) density of states at Fermi level (states/eV/cell)
  fermiEnergy: number;           // E_F in eV
  debyeFrequency: number;       // ω_D in meV (used for Morel-Anderson)
  converged: boolean;
  selfConsistentIterations: number;
  method: string;                // "ACBN0-hp.x" | "ACBN0-SCF-only" | "ACBN0-error"
  warnings: string[];
  elapsedSeconds?: number;
}

export interface ACBN0PipelineOptions {
  ecutwfc: number;               // Ry
  ecutrho?: number;              // Ry (default: 8 * ecutwfc)
  kgrid?: [number, number, number]; // k-point grid for SCF (default auto)
  qgrid?: [number, number, number]; // q-point grid for hp.x (default: same as k)
  pseudoDir: string;
  prefix: string;
  initialU?: Record<string, number>; // Initial Hubbard U values (eV); default all 0
  maxSCFIterations?: number;     // Self-consistent U iterations (default: 3)
  uConvergenceThreshold?: number; // eV, stop when ΔU < this (default: 0.1)
  debyeFrequencyMeV?: number;   // Override Debye frequency (meV)
  cellParameters?: string;       // Formatted CELL_PARAMETERS body
}

export interface ACBN0PipelineCallbacks {
  runQEBinary: (
    binary: string,
    inputFile: string,
    cwd: string,
    timeoutMs: number,
  ) => Promise<{ stdout: string; stderr: string; exitCode: number | null }>;
  getPseudoDirInput: () => string;
  resolvePPFilename: (el: string) => string;
  /** Directory containing the QE binaries (pw.x, hp.x). Must match the
   *  project-wide QE_BIN_DIR / getQEBinDir convention. Falls back to
   *  /usr/local/bin for backwards compatibility, but apt-installed QE on
   *  Ubuntu lives at /usr/bin — leaving this unset will silently fail to
   *  find QE on most non-Docker installations. */
  getQEBinDir?: () => string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ACBN0_TOTAL_TIMEOUT_MS = 2 * 60 * 60 * 1000; // 2 hours
const SCF_TIMEOUT_MS = 30 * 60 * 1000;              // 30 min per SCF
const HP_TIMEOUT_MS = 60 * 60 * 1000;               // 1 hour for hp.x
const RY_TO_EV = 13.605693122994;
const BOHR_TO_ANG = 0.529177210903;
const ANG_TO_BOHR = 1 / BOHR_TO_ANG;  // 1.8897259886... — for celldm(1) input
const HARTREE_TO_EV = 27.211386245988;

// Elements that can have meaningful Hubbard U corrections (d/f electrons)
const HUBBARD_ELEMENTS = new Set([
  // 3d
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  // 4d
  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  // 5d
  "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  // 4f
  "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
  // 5f
  "Th", "Pa", "U", "Np", "Pu",
  // p-block with correlated states
  "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi",
]);

// Default Hubbard manifold per element type
const HUBBARD_MANIFOLD: Record<string, string> = {
  // 3d
  Sc: "3d", Ti: "3d", V: "3d", Cr: "3d", Mn: "3d",
  Fe: "3d", Co: "3d", Ni: "3d", Cu: "3d", Zn: "3d",
  // 4d
  Y: "4d", Zr: "4d", Nb: "4d", Mo: "4d", Tc: "4d",
  Ru: "4d", Rh: "4d", Pd: "4d", Ag: "4d", Cd: "4d",
  // 5d
  La: "5d", Hf: "5d", Ta: "5d", W: "5d", Re: "5d",
  Os: "5d", Ir: "5d", Pt: "5d", Au: "5d", Hg: "5d",
  // Lanthanides — full series. Previously only Ce/Pr/Nd/Eu/Gd were listed;
  // Pm/Sm/Tb/Dy/Ho/Er/Tm/Yb/Lu were in HUBBARD_ELEMENTS but missing here,
  // so the manifold-lookup fallback "3d" silently put Hubbard U on the
  // (empty/core) 3d shell instead of the correlated 4f manifold. For
  // Sm/Tb/Dy/Ho-based correlated compounds this meant ACBN0 was effectively
  // computing U on a shell QE's pseudopotential may not even have as a
  // projector → either no-op correction or hp.x error.
  Ce: "4f", Pr: "4f", Nd: "4f", Pm: "4f", Sm: "4f", Eu: "4f", Gd: "4f",
  Tb: "4f", Dy: "4f", Ho: "4f", Er: "4f", Tm: "4f", Yb: "4f", Lu: "4f",
  // Actinides — same issue. Pa/Np/Pu were in HUBBARD_ELEMENTS but mapped
  // to "3d" fallback. Correlated 5f manifold is the right target.
  Th: "5f", Pa: "5f", U: "5f", Np: "5f", Pu: "5f",
  // p-block with semicore d states. The HUBBARD_ELEMENTS set includes these
  // for cases where +U on the d shell fixes d-p hybridization in narrow-gap
  // semiconductors (e.g., GaAs needs +U on Ga 3d to lower the d band below
  // E_F — Janotti & Van de Walle PRB 75, 121201 (2007)). Without these
  // mapped, ACBN0 would target whichever default the fallback gave (3d for
  // all, which is wrong for In/Sn 4d-semicore and Tl/Pb/Bi 5d-semicore).
  Ga: "3d", Ge: "3d",  // 3d¹⁰ semicore
  In: "4d", Sn: "4d",  // 4d¹⁰ semicore
  Tl: "5d", Pb: "5d", Bi: "5d",  // 5d¹⁰ semicore
};

// ---------------------------------------------------------------------------
// Input generators
// ---------------------------------------------------------------------------

function generateDFTplusUInput(opts: {
  prefix: string;
  pseudoDir: string;
  ecutwfc: number;
  ecutrho: number;
  latticeA: number;
  cellParameters?: string;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  elements: string[];
  ppFilenames: Record<string, string>;
  kGrid: [number, number, number];
  hubbardU: Record<string, number>;
}): string {
  const {
    prefix, pseudoDir, ecutwfc, ecutrho, latticeA,
    cellParameters, positions, elements, ppFilenames,
    kGrid, hubbardU,
  } = opts;

  const ntyp = elements.length;
  const nat = positions.length;

  // Use proper atomic masses (was hardcoded 1.0 for all species). hp.x and
  // SCF don't use the mass directly, but inconsistency with the upstream
  // SCF's atomic-species mass values means QE's .save consistency check
  // can flag the ACBN0 SCF as incompatible — and any post-processing tool
  // (matdyn, ph.x retry) reading this input would see m=1.0 and recompute
  // phonons with wrong masses.
  const speciesBlock = elements
    .map(el => {
      const data = getElementData(el);
      const mass = (data as any)?.atomicMass ?? (data as any)?.mass ?? 1.0;
      return `  ${el}  ${mass.toFixed(4)}  ${ppFilenames[el]}`;
    })
    .join("\n");

  const posBlock = positions
    .map(p => `  ${p.element}  ${p.x.toFixed(10)}  ${p.y.toFixed(10)}  ${p.z.toFixed(10)}`)
    .join("\n");

  // Detect whether cellParameters already includes the QE header. Callers
  // now pass the full block from generateCellParameters() (header
  // "CELL_PARAMETERS {angstrom}" + 3 rows). Double-wrapping with the legacy
  // "CELL_PARAMETERS {alat}\n${...}" prefix made pw.x see two CELL_PARAMETERS
  // lines back-to-back and reject the input.
  const cellBlock = cellParameters
    ? (cellParameters.trim().startsWith("CELL_PARAMETERS")
        ? `${cellParameters}\n`
        : `CELL_PARAMETERS {alat}\n${cellParameters}\n`)
    : "";

  // Build HUBBARD card (QE >= 7.1 new-style)
  const hubbardLines: string[] = [];
  const hasHubbardElements = elements.some(el => HUBBARD_ELEMENTS.has(el));

  if (hasHubbardElements) {
    for (const el of elements) {
      if (HUBBARD_ELEMENTS.has(el)) {
        const u = hubbardU[el] ?? 0.0;
        const manifold = HUBBARD_MANIFOLD[el] ?? "3d";
        hubbardLines.push(`  U ${el}-${manifold}  ${u.toFixed(4)}`);
      }
    }
  }

  const hubbardCard = hubbardLines.length > 0
    ? `HUBBARD {ortho-atomic}\n${hubbardLines.join("\n")}\n`
    : "";

  // For QE 7.1+, the HUBBARD card (emitted below) is the canonical way to
  // enable DFT+U; the legacy `lda_plus_u = .true.` in &SYSTEM is DEPRECATED
  // and rejected by recent QE builds. Emitting both produced "lda_plus_u is
  // no longer allowed" errors. Drop the &SYSTEM flag; HUBBARD card alone
  // is sufficient (and matches the convention used in hubbard-workflow.ts
  // and the main qe-worker generators).
  const hubbardSystemFlags = "";

  return `&CONTROL
  calculation = 'scf',
  prefix = '${prefix}_acbn0',
  outdir = './tmp',
  pseudo_dir = '${pseudoDir}',
  verbosity = 'high',
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 0,
  celldm(1) = ${(latticeA * ANG_TO_BOHR).toFixed(6)},
  nat = ${nat},
  ntyp = ${ntyp},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = 'smearing',
  smearing = 'cold',
  degauss = 0.02,
${hubbardSystemFlags}/
&ELECTRONS
  conv_thr = 1.0d-10,
  mixing_beta = 0.7,
/
ATOMIC_SPECIES
${speciesBlock}
ATOMIC_POSITIONS {crystal}
${posBlock}
${cellBlock}${hubbardCard}K_POINTS {automatic}
  ${kGrid[0]} ${kGrid[1]} ${kGrid[2]}  0 0 0
`;
}

function generateHPInput(opts: {
  prefix: string;
  qGrid: [number, number, number];
  elements: string[];
  nAtoms: number;
}): string {
  const { prefix, qGrid, elements, nAtoms } = opts;

  // Determine which atom indices have Hubbard correction
  // hp.x needs perturb_only_atom for efficiency (perturb only Hubbard sites)
  // For simplicity, perturb all Hubbard atoms

  return `&INPUTHP
  prefix = '${prefix}_acbn0',
  outdir = './tmp',
  nq1 = ${qGrid[0]},
  nq2 = ${qGrid[1]},
  nq3 = ${qGrid[2]},
  conv_thr_chi = 1.0d-5,
  iverbosity = 2,
/
`;
}

// ---------------------------------------------------------------------------
// Output parsers
// ---------------------------------------------------------------------------

interface SCFDOSInfo {
  fermiEnergy: number;    // eV
  nEF: number;            // states/eV/cell at E_F
  totalEnergy: number;    // Ry
}

function parseSCFOutput(stdout: string): SCFDOSInfo | null {
  let fermiEnergy = NaN;
  let nEF = NaN;
  let totalEnergy = NaN;

  // Fermi energy
  // "the Fermi energy is    12.3456 ev"
  const fermiMatch = stdout.match(/the Fermi energy is\s+([-\d.]+)\s*ev/i);
  if (fermiMatch) {
    fermiEnergy = parseFloat(fermiMatch[1]);
  }

  // Total energy
  // "!    total energy              =    -123.456789 Ry"
  const energyMatch = stdout.match(/!\s*total energy\s*=\s*([-\d.]+)\s*Ry/);
  if (energyMatch) {
    totalEnergy = parseFloat(energyMatch[1]);
  }

  // N(E_F) — density of states at Fermi level
  // QE with verbosity='high' prints:
  // "     the spin up/dn Fermi energy   ..."
  // or from DOS output:
  // "     DOS =   2.345 states/spin/eV/cell"
  const dosMatch = stdout.match(/DOS\s*=\s*([\d.]+)\s*states/i);
  if (dosMatch) {
    nEF = parseFloat(dosMatch[1]);
  }

  // Alternative: parse from the partial DOS section
  // "     N(E_F) =   4.567 states/eV"
  if (isNaN(nEF)) {
    const nefMatch = stdout.match(/N\(E_F\)\s*=\s*([\d.]+)\s*states/i);
    if (nefMatch) {
      nEF = parseFloat(nefMatch[1]);
    }
  }

  // If N(E_F) could not be parsed, leave it as NaN → 0 below. Do NOT
  // fabricate it from electron count / assumed bandwidth (the old
  // `nEl/20` estimate had no relation to the real DOS and silently fed a
  // bogus number into μ*); a 0 routes computeMuStar to its honest
  // default-fallback instead.

  if (isNaN(fermiEnergy)) {
    return null;
  }

  return {
    fermiEnergy,
    nEF: isNaN(nEF) ? 0 : nEF,
    totalEnergy: isNaN(totalEnergy) ? 0 : totalEnergy,
  };
}

interface HPResult {
  hubbardU: Record<string, number>;       // eV per element
  chi0: Record<string, number>;           // bare susceptibility per site
  chi: Record<string, number>;            // screened susceptibility per site
  screeningLength?: number;               // Bohr
  converged: boolean;
}

function parseHPOutput(stdout: string): HPResult {
  const hubbardU: Record<string, number> = {};
  const chi0: Record<string, number> = {};
  const chi: Record<string, number> = {};
  let converged = false;
  let screeningLength: number | undefined;

  // Parse Hubbard U values
  // hp.x output format varies by version. Common patterns:
  //
  // "site n.  1  atom  Fe  U =  4.5678 eV"
  // or table format:
  // "   1     Fe       3d       4.5678"
  const uPattern1 = /site\s+n\.\s*\d+\s+atom\s+(\w+)\s+.*?U\s*=\s*([\d.]+)\s*eV/gi;
  let um: RegExpExecArray | null;
  while ((um = uPattern1.exec(stdout)) !== null) {
    const el = um[1];
    const u = parseFloat(um[2]);
    if (!isNaN(u)) {
      hubbardU[el] = u;
    }
  }

  // Alternative table format
  // "      atom    manifold      U(eV)
  //        Fe        3d         4.5678"
  if (Object.keys(hubbardU).length === 0) {
    const tableSection = stdout.match(
      /atom\s+manifold\s+U\s*\(eV\)\s*\n((?:\s*\w+\s+\w+\s+[\d.]+\s*\n)+)/i,
    );
    if (tableSection) {
      const lines = tableSection[1].trim().split("\n");
      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 3) {
          const el = parts[0];
          const u = parseFloat(parts[2]);
          if (!isNaN(u) && el.match(/^[A-Z][a-z]?$/)) {
            hubbardU[el] = u;
          }
        }
      }
    }
  }

  // Another common format from hp.x:
  // "Hubbard U for atom Fe: 4.5678 eV"
  if (Object.keys(hubbardU).length === 0) {
    const uPattern3 = /Hubbard\s+U\s+(?:for\s+)?(?:atom\s+)?(\w+)\s*[=:]\s*([\d.]+)\s*eV/gi;
    while ((um = uPattern3.exec(stdout)) !== null) {
      hubbardU[um[1]] = parseFloat(um[2]);
    }
  }

  // Parse chi0 and chi (bare and screened response)
  // "chi_0 =  -0.1234   chi =  -0.0567"
  const chiPattern = /(?:site|atom)\s+(?:n\.\s*)?\d+\s+(?:atom\s+)?(\w+).*?chi_?0?\s*=\s*([-\d.Ee+]+).*?chi\s*=\s*([-\d.Ee+]+)/gi;
  let cm: RegExpExecArray | null;
  while ((cm = chiPattern.exec(stdout)) !== null) {
    const el = cm[1];
    chi0[el] = parseFloat(cm[2]);
    chi[el] = parseFloat(cm[3]);
  }

  // Screening length from dielectric response
  // "Thomas-Fermi screening length =  1.2345 bohr"
  const screenMatch = stdout.match(
    /(?:Thomas[- ]Fermi|screening)\s+(?:screening\s+)?length\s*=\s*([\d.]+)\s*(?:bohr|a\.u\.)/i,
  );
  if (screenMatch) {
    screeningLength = parseFloat(screenMatch[1]);
  }

  // Convergence check. "JOB DONE" alone is too lenient — hp.x prints it on
  // clean exit even when reaching max iterations without converging. Only the
  // explicit hp.x success markers should set converged. "JOB DONE" is now
  // accepted ONLY as a weak fallback paired with at least one Hubbard U
  // value successfully extracted from the output (Object.keys(hubbardU) > 0),
  // which indicates the calculation completed enough to produce results.
  if (stdout.includes("Convergence achieved") ||
      stdout.includes("HP run completed")) {
    converged = true;
  } else if (stdout.includes("JOB DONE") && Object.keys(hubbardU).length > 0) {
    converged = true;
  }

  return { hubbardU, chi0, chi, screeningLength, converged };
}

// ---------------------------------------------------------------------------
// μ* computation
// ---------------------------------------------------------------------------

/**
 * Compute the Coulomb pseudopotential μ* from the density of states.
 *
 * μ* enters Tc exponentially, so it must be physically right. We use the
 * Thomas-Fermi screened-Coulomb model in the free-electron limit, which is
 * fully determined by the DOS at the Fermi level — no empirical fits:
 *
 *   bare Coulomb parameter   μ = ⟨⟨N(E_F)·V_screened⟩⟩_FS = ln(1+y)/y
 *   with                     y = (2k_F/q_TF)² = π³·g(E_F)
 *
 * Derivation: for the Thomas-Fermi screened interaction V(q)=4πe²/(q²+q_TF²),
 * the Fermi-surface average (momentum transfer q∈[0,2k_F], weight q dq) of
 * N(E_F)·V is q_TF²·ln(1+4k_F²/q_TF²)/(4k_F²) = ln(1+y)/y. In the free-electron
 * gas g(E_F)=k_F/π² and q_TF²=4π·g(E_F), so y=4k_F²/q_TF²=π³·g(E_F): μ is fixed
 * by g(E_F) alone. This gives μ≈0.4–0.5 for typical metals (the known range)
 * and correctly makes high-DOS metals screen harder → lower μ.
 *
 * The previous code used μ=1/ε (a screening *ratio*, no N(E_F) dependence at
 * all) and μ=0.4·√N(E_F) (a √ law with no basis — μ=N(E_F)·V_c is linear in
 * N(E_F) at fixed V_c). Both are dropped.
 *
 * Retardation (Morel-Anderson): μ* = μ / (1 + μ·ln(E_el/ω_ph)), where the
 * electronic cutoff E_el is the free-electron occupied bandwidth
 * E_F^band = k_F²/2 (measured from the band bottom — NOT the absolute QE
 * Fermi level, which is referenced to an arbitrary pseudopotential zero).
 *
 * @see P. Morel & P. W. Anderson, Phys. Rev. 125, 1263 (1962)
 * @see G. Grimvall, "The Electron-Phonon Interaction in Metals" (1981)
 */
function computeMuStar(opts: {
  nEF: number;                   // states/eV/cell
  fermiEnergy: number;           // eV
  debyeFrequency: number;        // meV
  chi0?: Record<string, number>; // (unused — see derivation above)
  chi?: Record<string, number>;  // (unused)
  screeningLength?: number;      // (unused)
  cellVolume?: number;           // Angstrom^3
}): { muStar: number; muBare: number; method: string } {
  const { nEF, debyeFrequency, cellVolume } = opts;

  if (nEF <= 0 || debyeFrequency <= 0 || !cellVolume || cellVolume <= 0) {
    return { muStar: 0.10, muBare: 0.0, method: "default-fallback" };
  }

  // g(E_F): DOS at the Fermi level in atomic units — states/Hartree/Bohr³.
  const cellVolBohr3 = cellVolume / (BOHR_TO_ANG ** 3);
  const gEF = (nEF * HARTREE_TO_EV) / cellVolBohr3;

  // Bare Coulomb parameter, Thomas-Fermi free-electron model: μ = ln(1+y)/y.
  const y = Math.PI ** 3 * gEF;
  const muBare = y > 1e-6 ? Math.log(1 + y) / y : 1.0;

  // Free-electron occupied bandwidth as the retardation cutoff:
  // k_F = π²·g(E_F)  [Bohr⁻¹],  E_F^band = k_F²/2  [Hartree].
  const kF = Math.PI * Math.PI * gEF;
  const eElEV = Math.max(1.0, 0.5 * kF * kF * HARTREE_TO_EV);
  const omegaPhEV = debyeFrequency / 1000.0;

  // Morel-Anderson retardation.
  const logRatio = Math.log(Math.max(eElEV / omegaPhEV, 1.01));
  let muStar = muBare / (1 + muBare * logRatio);

  // μ* is confined to ~0.08–0.20 for conventional superconductors. A value
  // outside that band signals bad inputs (DOS extracted in a gap, wrong
  // cell volume, …) rather than exotic physics — warn and clamp so one bad
  // input cannot blow up the exponential Tc dependence.
  let method = "thomas-fermi-DOS";
  if (muStar < 0.08 || muStar > 0.20) {
    console.warn(`[ACBN0] μ* = ${muStar.toFixed(3)} outside the physical ` +
      `0.08–0.20 band (μ_bare=${muBare.toFixed(3)}, g(E_F)=${gEF.toExponential(2)} ` +
      `st/Ha/Bohr³, E_el=${eElEV.toFixed(1)} eV) — clamping; check N(E_F)/cellVolume.`);
    muStar = Math.min(0.20, Math.max(0.08, muStar));
    method = "thomas-fermi-DOS-clamped";
  }

  return { muStar, muBare, method };
}

// ---------------------------------------------------------------------------
// Debye frequency estimator
// ---------------------------------------------------------------------------

/**
 * Estimate Debye frequency from phonon data or atomic masses.
 * If omegaLog from DFPT is available, use that instead (it's better).
 */
function estimateDebyeFrequency(
  elements: string[],
  counts: Record<string, number>,
  pressureGPa: number,
  latticeA: number,
): number {
  // Rough Debye model: ω_D ~ v_s * (6π^2 n / V)^{1/3}
  // For hydrogen-rich systems at high pressure, ω_D can be 50-200 meV
  // For transition metals, ω_D ~ 20-50 meV

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hFraction = (counts["H"] ?? 0) / totalAtoms;

  // Base estimate from typical values
  let omegaD = 30; // meV, default for transition metals

  if (hFraction > 0.5) {
    // Hydrogen-rich: higher Debye frequency
    omegaD = 80 + 2 * pressureGPa; // increases with pressure
  } else if (hFraction > 0.2) {
    omegaD = 40 + pressureGPa;
  } else {
    omegaD = 20 + 0.5 * pressureGPa;
  }

  // Floor at 10 meV to keep the Morel-Anderson log finite. NO upper clamp:
  // hydrides under pressure (LaH10 @ 170 GPa: ω_D ≈ 350–420 meV; H3S @ 200 GPa:
  // ≈ 250–300 meV) have legitimately high Debye temperatures that drive their
  // high Tc, and clamping at 300 meV silently suppressed exactly that signal.
  omegaD = Math.max(10, omegaD);

  return omegaD;
}

// ---------------------------------------------------------------------------
// Cell volume estimator
// ---------------------------------------------------------------------------

function estimateCellVolume(
  latticeA: number,
  cellParameters?: string,
): number {
  if (!cellParameters) {
    return latticeA ** 3; // cubic
  }

  // The supplied block may include the QE header line as line 0
  // (e.g. "CELL_PARAMETERS {angstrom}") followed by three vector rows,
  // OR it may already be the bare vector rows. Strip the header if present
  // and remember the declared unit so the volume scaling is correct.
  const rawLines = cellParameters.trim().split("\n").filter(l => l.trim());
  let unit: "alat" | "angstrom" | "bohr" = "alat";
  let vectorLines = rawLines;
  if (rawLines[0] && /^\s*CELL_PARAMETERS/i.test(rawLines[0])) {
    const unitMatch = rawLines[0].match(/\{?\s*(angstrom|alat|bohr)\s*\}?/i);
    if (unitMatch) {
      const u = unitMatch[1].toLowerCase();
      if (u === "angstrom" || u === "alat" || u === "bohr") unit = u as typeof unit;
    }
    vectorLines = rawLines.slice(1);
  }

  if (vectorLines.length >= 3) {
    try {
      const v1 = vectorLines[0].trim().split(/\s+/).map(Number);
      const v2 = vectorLines[1].trim().split(/\s+/).map(Number);
      const v3 = vectorLines[2].trim().split(/\s+/).map(Number);

      if (v1.length >= 3 && v2.length >= 3 && v3.length >= 3
          && v1.every(Number.isFinite) && v2.every(Number.isFinite) && v3.every(Number.isFinite)) {
        const cross = [
          v1[1] * v2[2] - v1[2] * v2[1],
          v1[2] * v2[0] - v1[0] * v2[2],
          v1[0] * v2[1] - v1[1] * v2[0],
        ];
        const volRaw = Math.abs(cross[0] * v3[0] + cross[1] * v3[1] + cross[2] * v3[2]);

        // Convert to Å³ depending on the unit the vectors were expressed in.
        // Without this, an {angstrom} block (where rows are already in Å) was
        // being multiplied by latticeA³, inflating the cell volume by ~latticeA³
        // and corrupting downstream μ* / Debye estimates that consume the volume.
        if (unit === "angstrom") return volRaw;
        if (unit === "bohr") {
          const BOHR_TO_ANG = 0.529177210903;
          return volRaw * BOHR_TO_ANG ** 3;
        }
        // alat: vectors are in units of celldm(1) (Bohr). Scaling factor
        // here is latticeA (Å), so volRaw * latticeA³ gives Å³.
        return volRaw * (latticeA ** 3);
      }
    } catch {}
  }

  return latticeA ** 3;
}

// ---------------------------------------------------------------------------
// Main pipeline orchestrator
// ---------------------------------------------------------------------------

export async function runACBN0Pipeline(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  jobDir: string,
  pressureGPa: number,
  options: ACBN0PipelineOptions,
  callbacks: ACBN0PipelineCallbacks,
): Promise<ACBN0Result | null> {
  const warnings: string[] = [];
  const prefix = options.prefix || formula.replace(/[^a-zA-Z0-9]/g, "");
  const startTime = Date.now();

  console.log(`[ACBN0] Starting first-principles μ* calculation for ${formula}`);
  console.log(`[ACBN0]   P=${pressureGPa} GPa, elements=[${elements.join(",")}]`);

  // Resolve QE binary directory. Was hardcoded to /usr/local/bin for both
  // pw.x and hp.x — broke ACBN0 on apt-installed QE (lives at /usr/bin) and
  // on any system using QE_BIN_DIR to point at a custom build path.
  const qeBinDir = callbacks.getQEBinDir?.() ?? "/usr/local/bin";

  const ecutrho = options.ecutrho ?? options.ecutwfc * 8;
  const nAtoms = positions.length;

  // Auto k-grid: denser for small cells
  const autoK = nAtoms <= 4 ? 8 : nAtoms <= 8 ? 6 : 4;
  const kGrid = options.kgrid ?? [autoK, autoK, autoK] as [number, number, number];
  // q-grid for hp.x: typically same as or coarser than k-grid
  const qGrid = options.qgrid ?? [
    Math.max(2, Math.floor(kGrid[0] / 2)),
    Math.max(2, Math.floor(kGrid[1] / 2)),
    Math.max(2, Math.floor(kGrid[2] / 2)),
  ] as [number, number, number];

  const pseudoDir = callbacks.getPseudoDirInput();
  const ppFilenames: Record<string, string> = {};
  for (const el of elements) {
    ppFilenames[el] = callbacks.resolvePPFilename(el);
  }

  // Check if any elements need Hubbard U
  const hubbardElements = elements.filter(el => HUBBARD_ELEMENTS.has(el));
  const hasHubbard = hubbardElements.length > 0;

  if (!hasHubbard) {
    console.log(`[ACBN0] No d/f-electron elements — hp.x Hubbard calculation not applicable`);
    console.log(`[ACBN0] Will compute μ* from SCF-only N(E_F) via Thomas-Fermi model`);
  }

  // Hubbard U iteration loop
  let currentU: Record<string, number> = {};
  for (const el of elements) {
    currentU[el] = options.initialU?.[el] ?? 0.0;
  }

  const maxSCFIter = hasHubbard ? (options.maxSCFIterations ?? 3) : 1;
  const uThreshold = options.uConvergenceThreshold ?? 0.1; // eV

  let lastSCFInfo: SCFDOSInfo | null = null;
  let lastHPResult: HPResult | null = null;
  let selfConsistentIter = 0;
  let uConverged = false;

  for (let scfIter = 1; scfIter <= maxSCFIter; scfIter++) {
    const elapsed = Date.now() - startTime;
    if (elapsed > ACBN0_TOTAL_TIMEOUT_MS) {
      console.error(`[ACBN0] Total timeout reached at iteration ${scfIter}`);
      warnings.push("Total timeout reached during self-consistent U loop");
      break;
    }

    selfConsistentIter = scfIter;
    console.log(`[ACBN0] SCF iteration ${scfIter}/${maxSCFIter} — U values: ${JSON.stringify(currentU)}`);

    // -----------------------------------------------------------------------
    // Step 1: SCF with DFT+U
    // -----------------------------------------------------------------------
    console.log(`[ACBN0] Step 1: SCF with DFT+U (k-grid ${kGrid.join("x")})`);
    const scfInput = generateDFTplusUInput({
      prefix, pseudoDir, ecutwfc: options.ecutwfc, ecutrho,
      latticeA, cellParameters: options.cellParameters,
      positions, elements, ppFilenames,
      kGrid, hubbardU: currentU,
    });

    const scfFile = path.join(jobDir, `${prefix}_acbn0_scf.in`);
    try {
      fs.writeFileSync(scfFile, scfInput);
    } catch (e: any) {
      console.error(`[ACBN0] Failed to write SCF input: ${e.message}`);
      warnings.push("Failed to write SCF input");
      return makeErrorResult(warnings, currentU);
    }

    const scfResult = await safeRun(
      callbacks, path.posix.join(qeBinDir, "pw.x"),
      scfFile, jobDir, SCF_TIMEOUT_MS,
    );
    if (!scfResult || scfResult.exitCode !== 0) {
      console.error(`[ACBN0] SCF failed (exit ${scfResult?.exitCode}): ${(scfResult?.stderr ?? "").slice(0, 500)}`);
      warnings.push(`SCF step ${scfIter} failed`);

      // If first iteration fails, cannot proceed
      if (scfIter === 1) {
        return makeErrorResult(warnings, currentU);
      }
      break;
    }

    lastSCFInfo = parseSCFOutput(scfResult.stdout);
    if (!lastSCFInfo) {
      console.error(`[ACBN0] Failed to parse SCF output (no Fermi energy found)`);
      warnings.push("Could not parse Fermi energy from SCF output");
      if (scfIter === 1) {
        return makeErrorResult(warnings, currentU);
      }
      break;
    }

    console.log(`[ACBN0] SCF: E_F=${lastSCFInfo.fermiEnergy.toFixed(4)} eV, N(E_F)=${lastSCFInfo.nEF.toFixed(3)} states/eV/cell`);

    if (!hasHubbard) {
      // No Hubbard elements, skip hp.x
      break;
    }

    // -----------------------------------------------------------------------
    // Step 2: hp.x — Hubbard parameter from linear response
    // -----------------------------------------------------------------------
    console.log(`[ACBN0] Step 2: hp.x (q-grid ${qGrid.join("x")})`);
    const hpInput = generateHPInput({
      prefix,
      qGrid,
      elements,
      nAtoms,
    });

    const hpFile = path.join(jobDir, `${prefix}_hp.in`);
    try {
      fs.writeFileSync(hpFile, hpInput);
    } catch (e: any) {
      console.error(`[ACBN0] Failed to write hp.x input: ${e.message}`);
      warnings.push("Failed to write hp.x input");
      break;
    }

    const hpResult = await safeRun(
      callbacks, path.posix.join(qeBinDir, "hp.x"),
      hpFile, jobDir, HP_TIMEOUT_MS,
    );
    if (!hpResult || hpResult.exitCode !== 0) {
      console.error(`[ACBN0] hp.x failed (exit ${hpResult?.exitCode}): ${(hpResult?.stderr ?? "").slice(0, 500)}`);
      warnings.push(`hp.x failed at iteration ${scfIter}`);

      // Try to parse partial output
      if (hpResult?.stdout) {
        lastHPResult = parseHPOutput(hpResult.stdout);
      }
      break;
    }

    lastHPResult = parseHPOutput(hpResult.stdout);
    console.log(`[ACBN0] hp.x: U=${JSON.stringify(lastHPResult.hubbardU)}, converged=${lastHPResult.converged}`);

    if (lastHPResult.screeningLength) {
      console.log(`[ACBN0] Screening length: ${lastHPResult.screeningLength.toFixed(3)} Bohr`);
    }

    // Check U convergence
    let maxDeltaU = 0;
    const newU: Record<string, number> = { ...currentU };
    for (const el of Object.keys(lastHPResult.hubbardU)) {
      const delta = Math.abs(lastHPResult.hubbardU[el] - (currentU[el] ?? 0));
      maxDeltaU = Math.max(maxDeltaU, delta);
      newU[el] = lastHPResult.hubbardU[el];
    }

    console.log(`[ACBN0] Max ΔU = ${maxDeltaU.toFixed(4)} eV (threshold: ${uThreshold})`);

    // Convergence requires actually receiving U values from hp.x. If parsing
    // failed (empty hubbardU map) maxDeltaU stays at its initial 0 — the
    // previous check would then declare "converged" on iteration 2 even
    // though hp.x produced nothing usable. Demand at least one U value.
    const hpProducedUValues = Object.keys(lastHPResult.hubbardU).length > 0;
    if (maxDeltaU < uThreshold && scfIter > 1 && hpProducedUValues) {
      uConverged = true;
      console.log(`[ACBN0] Hubbard U converged at iteration ${scfIter}`);
      break;
    }
    if (!hpProducedUValues) {
      console.log(`[ACBN0] hp.x produced no Hubbard U values at iteration ${scfIter} — not treating as converged`);
    }

    currentU = newU;
  }

  // -------------------------------------------------------------------------
  // Step 3: Compute μ* from the collected data
  // -------------------------------------------------------------------------
  if (!lastSCFInfo) {
    warnings.push("No SCF data available for μ* computation");
    return makeErrorResult(warnings, currentU);
  }

  // Debye frequency
  const debyeFreq = options.debyeFrequencyMeV ?? estimateDebyeFrequency(
    elements, counts, pressureGPa, latticeA,
  );

  const cellVolume = estimateCellVolume(latticeA, options.cellParameters);

  const muResult = computeMuStar({
    nEF: lastSCFInfo.nEF,
    fermiEnergy: lastSCFInfo.fermiEnergy,
    debyeFrequency: debyeFreq,
    chi0: lastHPResult?.chi0,
    chi: lastHPResult?.chi,
    screeningLength: lastHPResult?.screeningLength,
    cellVolume,
  });

  console.log(
    `[ACBN0] μ* = ${muResult.muStar.toFixed(4)} (μ_bare = ${muResult.muBare.toFixed(4)}, ` +
    `method = ${muResult.method})`,
  );
  console.log(
    `[ACBN0] N(E_F) = ${lastSCFInfo.nEF.toFixed(3)} st/eV/cell, ` +
    `E_F = ${lastSCFInfo.fermiEnergy.toFixed(2)} eV, ` +
    `ω_D = ${debyeFreq.toFixed(1)} meV`,
  );

  // Sanity check μ*
  if (muResult.muStar < 0.05 || muResult.muStar > 0.25) {
    warnings.push(
      `μ* = ${muResult.muStar.toFixed(3)} is outside typical range [0.05, 0.25] — ` +
      `check N(E_F) and screening parameters`,
    );
  }

  const elapsedSeconds = (Date.now() - startTime) / 1000;

  // Estimate dielectric constant from chi0/chi if available
  let dielectricConstant: number | undefined;
  if (lastHPResult?.chi0 && lastHPResult?.chi) {
    const epsilons: number[] = [];
    for (const el of Object.keys(lastHPResult.chi0)) {
      const c0 = lastHPResult.chi0[el];
      const c = lastHPResult.chi[el];
      if (c !== 0 && Math.abs(c) > 1e-12) {
        epsilons.push(Math.abs(c0 / c));
      }
    }
    if (epsilons.length > 0) {
      dielectricConstant = epsilons.reduce((a, b) => a + b, 0) / epsilons.length;
    }
  }

  const result: ACBN0Result = {
    muStar: muResult.muStar,
    muBare: muResult.muBare,
    screeningLength: lastHPResult?.screeningLength ?? 0,
    hubbardU: { ...currentU, ...(lastHPResult?.hubbardU ?? {}) },
    dielectricConstant,
    nEF: lastSCFInfo.nEF,
    fermiEnergy: lastSCFInfo.fermiEnergy,
    debyeFrequency: debyeFreq,
    converged: hasHubbard ? (uConverged || lastHPResult?.converged === true) : true,
    selfConsistentIterations: selfConsistentIter,
    method: muResult.method,
    warnings,
    elapsedSeconds: Math.round(elapsedSeconds),
  };

  console.log(
    `[ACBN0] ${formula}: μ*=${result.muStar.toFixed(4)}, ` +
    `U={${Object.entries(result.hubbardU).map(([k, v]) => `${k}:${v.toFixed(2)}`).join(", ")}}, ` +
    `converged=${result.converged}, ${elapsedSeconds.toFixed(0)}s`,
  );

  return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function safeRun(
  callbacks: ACBN0PipelineCallbacks,
  binary: string,
  inputFile: string,
  cwd: string,
  timeoutMs: number,
): Promise<{ stdout: string; stderr: string; exitCode: number | null } | null> {
  try {
    return await callbacks.runQEBinary(binary, inputFile, cwd, timeoutMs);
  } catch (e: any) {
    console.error(`[ACBN0] Exception running ${path.basename(binary)}: ${e.message}`);
    return null;
  }
}

function makeErrorResult(warnings: string[], hubbardU: Record<string, number>): ACBN0Result {
  return {
    muStar: 0,
    muBare: 0,
    screeningLength: 0,
    hubbardU,
    nEF: 0,
    fermiEnergy: 0,
    debyeFrequency: 0,
    converged: false,
    selfConsistentIterations: 0,
    method: "ACBN0-error",
    warnings,
  };
}
