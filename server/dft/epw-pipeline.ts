/**
 * Wannier90 + EPW Electron-Phonon Coupling Pipeline
 *
 * Full workflow for computing Eliashberg spectral function and Tc via EPW:
 *   1. NSCF on uniform k-grid (pw.x)
 *   2. Wannier90 preprocessing (wannier90.x -pp)
 *   3. pw2wannier90.x — project Bloch states onto Wannier functions
 *   4. Wannier90 full minimization (wannier90.x)
 *   5. EPW interpolation and Eliashberg solver (epw.x)
 *
 * Relies on prior SCF + phonon (DFPT) runs having produced the
 * coarse-grid dynamical matrices (.dyn files) and .dvscf potentials.
 */

import * as fs from "fs";
import * as path from "path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EPWResult {
  lambda: number;
  lambdaMax: number;
  omegaLog: number;           // meV
  tcAllenDynes: number;       // K
  tcMigdalEliashberg: number; // K
  gapZero: number;            // meV — superconducting gap at lowest T
  alpha2F: { omega: number[]; a2f: number[] };
  converged: boolean;
  nkFine: number;
  nqFine: number;
  method: string;             // "EPW-ME" or "EPW-AD"
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Wannier projection table — element → orbital string for .win file
// ---------------------------------------------------------------------------

const WANNIER_PROJECTIONS: Record<string, string> = {
  // s-block light
  H: "s",
  // s+p alkali / alkaline-earth
  Li: "s;p", Na: "s;p", K: "s;p",
  Be: "s;p", Mg: "s;p", Ca: "s;p", Sr: "s;p", Ba: "s;p",
  // p-block main group (rows 2-3)
  B: "s;p", C: "s;p", N: "s;p", O: "s;p", F: "s;p",
  Al: "s;p", Si: "s;p", P: "s;p", S: "s;p", Cl: "s;p",
  // 3d transition metals
  Sc: "s;p;d", Ti: "s;p;d", V: "s;p;d", Cr: "s;p;d", Mn: "s;p;d",
  Fe: "s;p;d", Co: "s;p;d", Ni: "s;p;d", Cu: "s;p;d", Zn: "s;p;d",
  // 4d transition metals
  Y: "s;p;d", Zr: "s;p;d", Nb: "s;p;d", Mo: "s;p;d", Tc: "s;p;d",
  Ru: "s;p;d", Rh: "s;p;d", Pd: "s;p;d", Ag: "s;p;d", Cd: "s;p;d",
  // 5d transition metals
  La: "s;p;d", Hf: "s;p;d", Ta: "s;p;d", W: "s;p;d", Re: "s;p;d",
  Os: "s;p;d", Ir: "s;p;d", Pt: "s;p;d", Au: "s;p;d", Hg: "s;p;d",
  // f-block (actinide / lanthanide with f)
  Ce: "s;p;d;f", Th: "s;p;d;f",
  // p-block heavy (rows 4-6)
  Ga: "s;p", Ge: "s;p", As: "s;p", Se: "s;p", Br: "s;p",
  In: "s;p", Sn: "s;p", Sb: "s;p", Te: "s;p", I: "s;p",
  Tl: "s;p", Pb: "s;p", Bi: "s;p",
};

/** Number of Wannier orbitals per projection keyword */
const ORBITALS_PER_PROJ: Record<string, number> = {
  s: 1, p: 3, d: 5, f: 7,
};

function countWannierOrbitals(projStr: string): number {
  return projStr.split(";").reduce((sum, p) => sum + (ORBITALS_PER_PROJ[p] ?? 0), 0);
}

// ---------------------------------------------------------------------------
// Auto grid selection
// ---------------------------------------------------------------------------

export function autoEPWGrids(
  nAtoms: number,
  phononQGrid: [number, number, number],
): { nscfK: [number, number, number]; nkFine: [number, number, number]; nqFine: [number, number, number] } {
  const q = phononQGrid[0]; // assume cubic for grid selection

  let nscf: number;
  let fine: number;
  if (q >= 6)      { nscf = 12; fine = 40; }
  else if (q >= 4) { nscf = 12; fine = 36; }
  else if (q >= 3) { nscf = 9;  fine = 30; }
  else             { nscf = 8;  fine = 24; }

  // Progressive grid reduction for large unit cells
  if (nAtoms > 20) {
    fine = Math.max(12, Math.floor(fine / 3));   // ultra-large: 1/3 grid
    nscf = Math.max(6, Math.floor(nscf * 0.6));
  } else if (nAtoms > 16) {
    fine = Math.max(14, Math.floor(fine / 2));    // very large: 1/2 grid
    nscf = Math.max(6, Math.floor(nscf * 0.75));
  } else if (nAtoms > 12) {
    fine = Math.max(16, Math.floor(fine * 0.65)); // large: 2/3 grid
  }
  // Enforce minimum
  fine = Math.max(fine, 12);

  return {
    nscfK: [nscf, nscf, nscf],
    nkFine: [fine, fine, fine],
    nqFine: [fine, fine, fine],
  };
}

// ---------------------------------------------------------------------------
// Input generators
// ---------------------------------------------------------------------------

export function generateNSCFInput(opts: {
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
  nbnd: number;
  occupations?: string;
  smearing?: string;
  degauss?: number;
}): string {
  const {
    prefix, pseudoDir, ecutwfc, ecutrho, latticeA,
    cellParameters, positions, elements, ppFilenames,
    kGrid, nbnd,
    occupations = "smearing",
    smearing = "cold",
    degauss = 0.02,
  } = opts;

  const ntyp = elements.length;
  const nat = positions.length;

  const speciesBlock = elements
    .map(el => `  ${el}  1.0  ${ppFilenames[el]}`)
    .join("\n");

  const posBlock = positions
    .map(p => `  ${p.element}  ${p.x.toFixed(10)}  ${p.y.toFixed(10)}  ${p.z.toFixed(10)}`)
    .join("\n");

  const cellBlock = cellParameters
    ? `CELL_PARAMETERS {alat}\n${cellParameters}\n`
    : "";

  return `&CONTROL
  calculation = 'nscf',
  prefix = '${prefix}',
  outdir = './tmp',
  pseudo_dir = '${pseudoDir}',
  verbosity = 'high',
/
&SYSTEM
  ibrav = 0,
  celldm(1) = ${(latticeA / 0.529177).toFixed(6)},
  nat = ${nat},
  ntyp = ${ntyp},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = '${occupations}',
  smearing = '${smearing}',
  degauss = ${degauss},
  nbnd = ${nbnd},
  nosym = .true.,
  noinv = .true.,
/
&ELECTRONS
  conv_thr = 1.0d-10,
  diago_full_acc = .true.,
/
ATOMIC_SPECIES
${speciesBlock}
ATOMIC_POSITIONS {crystal}
${posBlock}
${cellBlock}K_POINTS {automatic}
  ${kGrid[0]} ${kGrid[1]} ${kGrid[2]}  0 0 0
`;
}

export function generateWannier90Win(opts: {
  prefix: string;
  elements: string[];
  counts: Record<string, number>;
  numBands: number;
  kGrid: [number, number, number];
  fermiEnergy: number;
  latticeVectors: number[][];   // 3×3 in Angstrom
  positions: Array<{ element: string; x: number; y: number; z: number }>;
}): string {
  const {
    prefix, elements, counts, numBands, kGrid,
    fermiEnergy, latticeVectors, positions,
  } = opts;

  // Count total Wannier functions
  let numWann = 0;
  for (const el of elements) {
    const proj = WANNIER_PROJECTIONS[el] ?? "s;p";
    numWann += countWannierOrbitals(proj) * (counts[el] ?? 1);
  }

  // Build projections block
  const projLines: string[] = [];
  for (const el of elements) {
    const proj = WANNIER_PROJECTIONS[el] ?? "s;p";
    projLines.push(`${el}: ${proj.replace(/;/g, ";")}`);
  }

  // Disentanglement windows — generous window around Fermi level
  const disWinMin = fermiEnergy - 15.0;
  const disWinMax = fermiEnergy + 20.0;
  const disFrozMin = fermiEnergy - 5.0;
  const disFrozMax = fermiEnergy + 2.0;

  // Generate k-point list on the nscf mesh (Gamma-centered, no shift)
  const kPoints: string[] = [];
  const nk1 = kGrid[0], nk2 = kGrid[1], nk3 = kGrid[2];
  for (let i = 0; i < nk1; i++) {
    for (let j = 0; j < nk2; j++) {
      for (let k = 0; k < nk3; k++) {
        kPoints.push(`  ${(i / nk1).toFixed(8)}  ${(j / nk2).toFixed(8)}  ${(k / nk3).toFixed(8)}`);
      }
    }
  }

  // Lattice vectors block (in Angstrom)
  const unitCellCart = latticeVectors
    .map(v => `  ${v[0].toFixed(10)}  ${v[1].toFixed(10)}  ${v[2].toFixed(10)}`)
    .join("\n");

  // Atomic positions in fractional
  const atomsFrac = positions
    .map(p => `${p.element}  ${p.x.toFixed(10)}  ${p.y.toFixed(10)}  ${p.z.toFixed(10)}`)
    .join("\n");

  return `num_wann = ${numWann}
num_bands = ${numBands}

! Disentanglement
dis_win_min  = ${disWinMin.toFixed(4)}
dis_win_max  = ${disWinMax.toFixed(4)}
dis_froz_min = ${disFrozMin.toFixed(4)}
dis_froz_max = ${disFrozMax.toFixed(4)}
dis_num_iter = 200
dis_mix_ratio = 0.5

! Wannierization
num_iter = 200
num_print_cycles = 20

! Write AMN and MMN (needed for pw2wannier90)
write_hr = .true.

! K-mesh
mp_grid = ${nk1} ${nk2} ${nk3}

begin projections
${projLines.join("\n")}
end projections

begin unit_cell_cart
Ang
${unitCellCart}
end unit_cell_cart

begin atoms_frac
${atomsFrac}
end atoms_frac

begin kpoints
${kPoints.join("\n")}
end kpoints
`;
}

export function generatePW2Wannier90Input(prefix: string): string {
  return `&INPUTPP
  outdir = './tmp',
  prefix = '${prefix}',
  seedname = '${prefix}',
  spin_component = 'none',
  write_mmn = .true.,
  write_amn = .true.,
  write_unk = .false.,
/
`;
}

export function generateEPWInput(opts: {
  prefix: string;
  coarseKGrid: [number, number, number];
  coarseQGrid: [number, number, number];
  fineKGrid: [number, number, number];
  fineQGrid: [number, number, number];
  numBands: number;
  numWann: number;
  elements: string[];
  counts: Record<string, number>;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  ppFilenames: Record<string, string>;
  pseudoDir: string;
  ecutwfc: number;
  ecutrho: number;
  latticeA: number;
  cellParameters?: string;
  fsthick?: number;   // eV, Fermi surface thickness
  degaussw?: number;  // eV
  isHydride?: boolean;
}): string {
  const {
    prefix, coarseKGrid, coarseQGrid, fineKGrid, fineQGrid,
    numBands, numWann, elements, counts, positions,
    ppFilenames, pseudoDir, ecutwfc, ecutrho, latticeA,
    cellParameters,
    degaussw = 0.025,
    isHydride = false,
  } = opts;

  const fsthick = opts.fsthick ?? (isHydride ? 1.0 : 0.4);
  const ntyp = elements.length;
  const nat = positions.length;

  const speciesBlock = elements
    .map(el => `  ${el}  1.0  ${ppFilenames[el]}`)
    .join("\n");

  const posBlock = positions
    .map(p => `  ${p.element}  ${p.x.toFixed(10)}  ${p.y.toFixed(10)}  ${p.z.toFixed(10)}`)
    .join("\n");

  const cellBlock = cellParameters
    ? `CELL_PARAMETERS {alat}\n${cellParameters}\n`
    : "";

  // Temperature sweep: 5K to 300K in 5K steps (60 temperatures)
  const nstemp = 60;
  const temps_min = 5.0;
  const temps_max = 300.0;

  return `--
&inputepw
  prefix = '${prefix}',
  outdir = './tmp',
  dvscf_dir = './tmp',
  amass(1) = 1.0,

  ! System
  ibrav = 0,
  celldm(1) = ${(latticeA / 0.529177).toFixed(6)},
  nat = ${nat},
  ntyp = ${ntyp},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},

  ! EPW control
  epbwrite = .true.,
  epbread = .false.,
  epwwrite = .true.,
  epwread = .false.,
  ephwrite = .false.,

  ! Wannier
  wannierize = .true.,
  nbndsub = ${numWann},
  nbndskip = 0,
  num_iter = 300,
  dis_win_min = -100.0,
  dis_win_max = 100.0,
  proj(1) = 'random',

  ! Interpolation grids (coarse)
  nkf1 = ${fineKGrid[0]}, nkf2 = ${fineKGrid[1]}, nkf3 = ${fineKGrid[2]},
  nqf1 = ${fineQGrid[0]}, nqf2 = ${fineQGrid[1]}, nqf3 = ${fineQGrid[2]},
  nk1 = ${coarseKGrid[0]}, nk2 = ${coarseKGrid[1]}, nk3 = ${coarseKGrid[2]},
  nq1 = ${coarseQGrid[0]}, nq2 = ${coarseQGrid[1]}, nq3 = ${coarseQGrid[2]},

  ! Electron-phonon
  elph = .true.,
  fsthick = ${fsthick},
  degaussw = ${degaussw},

  ! Eliashberg solver
  eliashberg = .true.,
  laniso = .true.,
  limag = .true.,
  lpade = .true.,
  nstemp = ${nstemp},
  temps = ${temps_min} ${temps_max},

  ! Misc
  ncarrier = 0,
  mp_mesh_k = .true.,
  filkf = ' ',
  filqf = ' ',
/
ATOMIC_SPECIES
${speciesBlock}
ATOMIC_POSITIONS {crystal}
${posBlock}
${cellBlock}K_POINTS {automatic}
  ${coarseKGrid[0]} ${coarseKGrid[1]} ${coarseKGrid[2]}  0 0 0
`;
}

// ---------------------------------------------------------------------------
// EPW output parser
// ---------------------------------------------------------------------------

export function parseEPWOutput(stdout: string): Partial<EPWResult> {
  const warnings: string[] = [];
  let lambda = NaN;
  let lambdaMax = NaN;
  let omegaLog = NaN;
  let tcAD = NaN;
  let tcME = NaN;
  let gapZero = NaN;
  let converged = false;

  // --- lambda (isotropic) ---
  // EPW prints: "  lambda   =   1.2345"
  const lambdaMatch = stdout.match(/lambda\s*(?:\(\s*\d+\s*\))?\s*=\s*([\d.]+)/i);
  if (lambdaMatch) {
    lambda = parseFloat(lambdaMatch[1]);
  }

  // Collect all lambda values to find max
  const allLambdas: number[] = [];
  const lambdaRegex = /lambda\s*(?:\(\s*\d+\s*\))?\s*=\s*([\d.]+)/gi;
  let lm: RegExpExecArray | null;
  while ((lm = lambdaRegex.exec(stdout)) !== null) {
    const v = parseFloat(lm[1]);
    if (!isNaN(v)) allLambdas.push(v);
  }
  lambdaMax = allLambdas.length > 0 ? Math.max(...allLambdas) : lambda;

  // --- omega_log ---
  // "  omega_log =   123.456 (meV)"   or   "  omega_log =   123.456 meV"
  const omegaLogMatch = stdout.match(/omega_log\s*=\s*([\d.]+)\s*(?:\(?\s*meV)?/i);
  if (omegaLogMatch) {
    omegaLog = parseFloat(omegaLogMatch[1]);
  }

  // --- Allen-Dynes Tc ---
  // "  Tc (Allen-Dynes) =    12.34 K"
  const tcADMatch = stdout.match(/Tc\s*\(?\s*Allen[\s-]*Dynes\s*\)?\s*=\s*([\d.]+)\s*K/i);
  if (tcADMatch) {
    tcAD = parseFloat(tcADMatch[1]);
  }

  // --- Migdal-Eliashberg Tc ---
  // EPW reports the gap closing temperature from the isotropic/anisotropic ME solver.
  // Several possible formats:
  //   "  Tc_ME =   15.67 K"
  //   "  Tc (Migdal-Eliashberg) =   15.67 K"
  //   "  Superconducting transition temperature Tc =   15.67 K"
  const tcMEPatterns = [
    /Tc_ME\s*=\s*([\d.]+)\s*K/i,
    /Tc\s*\(?\s*Migdal[\s-]*Eliashberg\s*\)?\s*=\s*([\d.]+)\s*K/i,
    /[Ss]uperconducting\s+transition\s+temperature\s+Tc\s*=\s*([\d.]+)\s*K/i,
  ];
  for (const pat of tcMEPatterns) {
    const m = stdout.match(pat);
    if (m) { tcME = parseFloat(m[1]); break; }
  }

  // --- Gap at lowest temperature ---
  // "  Gap(1) =   1.234 meV" or "  Delta(1) =   1.234 meV"
  const gapPatterns = [
    /[Gg]ap\s*\(\s*1\s*\)\s*=\s*([\d.]+)\s*meV/,
    /[Dd]elta\s*\(\s*1\s*\)\s*=\s*([\d.]+)\s*meV/,
    /[Gg]ap_0\s*=\s*([\d.]+)\s*meV/,
  ];
  for (const pat of gapPatterns) {
    const m = stdout.match(pat);
    if (m) { gapZero = parseFloat(m[1]); break; }
  }

  // --- alpha2F data ---
  const a2fOmega: number[] = [];
  const a2fVals: number[] = [];
  // EPW prints a2F in tabular format:  "   freq   a2F   a2F_cum  ..."
  // or in a file. Try to parse inline data.
  const a2fRegex = /^\s*([\d.]+)\s+([\d.Ee+-]+)\s+/gm;
  // Look for the alpha2F section
  const a2fSection = stdout.match(/alpha2F.*?\n((?:\s*[\d.]+\s+[\d.Ee+-]+.*\n)+)/i);
  if (a2fSection) {
    let am: RegExpExecArray | null;
    const a2fLineRe = /^\s*([\d.]+)\s+([\d.Ee+-]+)/gm;
    const sectionText = a2fSection[1];
    while ((am = a2fLineRe.exec(sectionText)) !== null) {
      a2fOmega.push(parseFloat(am[1]));
      a2fVals.push(parseFloat(am[2]));
    }
  }

  // --- Convergence check ---
  if (stdout.includes("Eliashberg equations are converged") ||
      stdout.includes("Convergence was reached") ||
      (!isNaN(tcME) && tcME > 0)) {
    converged = true;
  }

  // --- Warnings ---
  if (stdout.includes("WARNING")) {
    const warnLines = stdout.split("\n").filter(l => l.includes("WARNING"));
    for (const w of warnLines.slice(0, 10)) {
      warnings.push(w.trim());
    }
  }
  if (isNaN(lambda)) warnings.push("Failed to parse lambda from EPW output");
  if (isNaN(omegaLog)) warnings.push("Failed to parse omega_log from EPW output");

  return {
    lambda,
    lambdaMax,
    omegaLog,
    tcAllenDynes: tcAD,
    tcMigdalEliashberg: tcME,
    gapZero,
    alpha2F: { omega: a2fOmega, a2f: a2fVals },
    converged,
    warnings,
    method: converged && !isNaN(tcME) ? "EPW-ME" : "EPW-AD",
  };
}

// ---------------------------------------------------------------------------
// Main pipeline orchestrator
// ---------------------------------------------------------------------------

export interface EPWPipelineOptions {
  fermiEnergy: number;
  ecutwfc: number;
  ecutrho?: number;
  phononQGrid: [number, number, number];
  cOverA?: number;
  nbnd?: number;
  cellParameters?: string;        // already-formatted CELL_PARAMETERS body
  latticeVectors?: number[][];    // 3×3 in Angstrom for .win file
  isHydride?: boolean;
}

export interface EPWPipelineCallbacks {
  runQEBinary: (
    binary: string,
    inputFile: string,
    cwd: string,
    timeoutMs: number,
  ) => Promise<{ stdout: string; stderr: string; exitCode: number | null }>;
  getPseudoDirInput: () => string;
  resolvePPFilename: (el: string) => string;
}

export async function runEPWPipeline(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  jobDir: string,
  pressure: number,
  options: EPWPipelineOptions,
  callbacks: EPWPipelineCallbacks,
): Promise<EPWResult | null> {
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const nAtoms = positions.length;
  const pseudoDir = callbacks.getPseudoDirInput();
  const warnings: string[] = [];

  const ppFilenames: Record<string, string> = {};
  for (const el of elements) {
    ppFilenames[el] = callbacks.resolvePPFilename(el);
  }

  // Auto-select grids
  const grids = autoEPWGrids(nAtoms, options.phononQGrid);
  console.log(`[EPW] ${formula}: grids — NSCF ${grids.nscfK.join("x")}, fine k ${grids.nkFine.join("x")}, fine q ${grids.nqFine.join("x")}`);

  // Count Wannier functions for nbnd estimate
  let numWann = 0;
  for (const el of elements) {
    const proj = WANNIER_PROJECTIONS[el] ?? "s;p";
    numWann += countWannierOrbitals(proj) * (counts[el] ?? 1);
  }
  // nbnd should be at least 20% more than numWann for disentanglement headroom
  const nbnd = options.nbnd ?? Math.max(numWann + 4, Math.ceil(numWann * 1.2));

  const ecutrho = options.ecutrho ?? options.ecutwfc * 8;

  // Default lattice vectors: cubic if not provided
  const latticeVectors = options.latticeVectors ?? [
    [latticeA, 0, 0],
    [0, latticeA, 0],
    [0, 0, latticeA],
  ];

  // -----------------------------------------------------------------------
  // Step 1: NSCF calculation
  // -----------------------------------------------------------------------
  console.log(`[EPW] Step 1/5: NSCF on ${grids.nscfK.join("x")} k-grid`);
  const nscfInput = generateNSCFInput({
    prefix, pseudoDir, ecutwfc: options.ecutwfc, ecutrho,
    latticeA, cellParameters: options.cellParameters,
    positions, elements, ppFilenames,
    kGrid: grids.nscfK, nbnd,
  });

  const nscfFile = path.posix.join(jobDir, `${prefix}_nscf.in`);
  try {
    fs.writeFileSync(nscfFile, nscfInput);
  } catch (e: any) {
    console.error(`[EPW] Failed to write NSCF input: ${e.message}`);
    return null;
  }

  const nscfResult = await safeRun(
    callbacks, path.posix.join("/usr/local/bin", "pw.x"),
    nscfFile, jobDir, 3_600_000, // 1h timeout
  );
  if (!nscfResult || nscfResult.exitCode !== 0) {
    console.error(`[EPW] NSCF failed (exit ${nscfResult?.exitCode}): ${(nscfResult?.stderr ?? "").slice(0, 500)}`);
    warnings.push("NSCF step failed");
    return makePartialResult(warnings, grids);
  }
  console.log(`[EPW] NSCF completed successfully`);

  // -----------------------------------------------------------------------
  // Step 2: Wannier90 preprocessing (-pp)
  // -----------------------------------------------------------------------
  console.log(`[EPW] Step 2/5: wannier90.x -pp (generate .nnkp)`);
  const winContent = generateWannier90Win({
    prefix, elements, counts, numBands: nbnd,
    kGrid: grids.nscfK, fermiEnergy: options.fermiEnergy,
    latticeVectors, positions,
  });

  const winFile = path.posix.join(jobDir, `${prefix}.win`);
  try {
    fs.writeFileSync(winFile, winContent);
  } catch (e: any) {
    console.error(`[EPW] Failed to write .win file: ${e.message}`);
    warnings.push("Failed to write Wannier90 .win file");
    return makePartialResult(warnings, grids);
  }

  // wannier90.x -pp takes the seedname (prefix), not an input file
  // We need to run: wannier90.x -pp <prefix>   from within jobDir
  const w90ppResult = await safeRun(
    callbacks, path.posix.join("/usr/local/bin", "wannier90.x"),
    `-pp ${prefix}`, jobDir, 600_000, // 10 min
  );
  if (!w90ppResult || w90ppResult.exitCode !== 0) {
    console.error(`[EPW] wannier90 -pp failed (exit ${w90ppResult?.exitCode}): ${(w90ppResult?.stderr ?? "").slice(0, 500)}`);
    warnings.push("Wannier90 preprocessing failed");
    return makePartialResult(warnings, grids);
  }
  console.log(`[EPW] wannier90 -pp completed, .nnkp generated`);

  // -----------------------------------------------------------------------
  // Step 3: pw2wannier90.x
  // -----------------------------------------------------------------------
  console.log(`[EPW] Step 3/5: pw2wannier90.x (Bloch → Wannier overlap matrices)`);
  const pw2wInput = generatePW2Wannier90Input(prefix);
  const pw2wFile = path.posix.join(jobDir, `${prefix}_pw2wan.in`);
  try {
    fs.writeFileSync(pw2wFile, pw2wInput);
  } catch (e: any) {
    console.error(`[EPW] Failed to write pw2wannier90 input: ${e.message}`);
    warnings.push("Failed to write pw2wannier90 input");
    return makePartialResult(warnings, grids);
  }

  const pw2wResult = await safeRun(
    callbacks, path.posix.join("/usr/local/bin", "pw2wannier90.x"),
    pw2wFile, jobDir, 1_800_000, // 30 min
  );
  if (!pw2wResult || pw2wResult.exitCode !== 0) {
    console.error(`[EPW] pw2wannier90 failed (exit ${pw2wResult?.exitCode}): ${(pw2wResult?.stderr ?? "").slice(0, 500)}`);
    warnings.push("pw2wannier90 step failed");
    return makePartialResult(warnings, grids);
  }
  console.log(`[EPW] pw2wannier90 completed (.amn, .mmn written)`);

  // -----------------------------------------------------------------------
  // Step 4: Wannier90 full minimization
  // -----------------------------------------------------------------------
  console.log(`[EPW] Step 4/5: wannier90.x (full Wannier minimization)`);
  const w90Result = await safeRun(
    callbacks, path.posix.join("/usr/local/bin", "wannier90.x"),
    prefix, jobDir, 1_800_000, // 30 min
  );
  if (!w90Result || w90Result.exitCode !== 0) {
    console.error(`[EPW] wannier90 full failed (exit ${w90Result?.exitCode}): ${(w90Result?.stderr ?? "").slice(0, 500)}`);
    warnings.push("Wannier90 minimization failed");
    return makePartialResult(warnings, grids);
  }
  // Check for convergence in Wannier90 output
  if (w90Result.stdout.includes("Spread")) {
    console.log(`[EPW] wannier90 completed (.chk written)`);
  } else {
    warnings.push("Wannier90 may not have converged — no spread data found");
  }

  // -----------------------------------------------------------------------
  // Step 5: EPW electron-phonon interpolation + Eliashberg
  // -----------------------------------------------------------------------
  console.log(`[EPW] Step 5/5: epw.x (interpolation + Eliashberg solver)`);
  const epwInput = generateEPWInput({
    prefix,
    coarseKGrid: grids.nscfK,
    coarseQGrid: options.phononQGrid,
    fineKGrid: grids.nkFine,
    fineQGrid: grids.nqFine,
    numBands: nbnd,
    numWann,
    elements, counts, positions,
    ppFilenames, pseudoDir,
    ecutwfc: options.ecutwfc, ecutrho,
    latticeA,
    cellParameters: options.cellParameters,
    isHydride: options.isHydride,
  });

  const epwFile = path.posix.join(jobDir, `${prefix}_epw.in`);
  try {
    fs.writeFileSync(epwFile, epwInput);
  } catch (e: any) {
    console.error(`[EPW] Failed to write EPW input: ${e.message}`);
    warnings.push("Failed to write EPW input");
    return makePartialResult(warnings, grids);
  }

  const epwResult = await safeRun(
    callbacks, path.posix.join("/usr/local/bin", "epw.x"),
    epwFile, jobDir, 14_400_000, // 4h timeout for EPW
  );
  if (!epwResult || epwResult.exitCode !== 0) {
    console.error(`[EPW] epw.x failed (exit ${epwResult?.exitCode}): ${(epwResult?.stderr ?? "").slice(0, 500)}`);
    warnings.push("EPW calculation failed");
    // Try to parse partial output anyway
    if (epwResult?.stdout) {
      const partial = parseEPWOutput(epwResult.stdout);
      return {
        lambda: partial.lambda ?? NaN,
        lambdaMax: partial.lambdaMax ?? NaN,
        omegaLog: partial.omegaLog ?? NaN,
        tcAllenDynes: partial.tcAllenDynes ?? NaN,
        tcMigdalEliashberg: partial.tcMigdalEliashberg ?? NaN,
        gapZero: partial.gapZero ?? NaN,
        alpha2F: partial.alpha2F ?? { omega: [], a2f: [] },
        converged: false,
        nkFine: grids.nkFine[0],
        nqFine: grids.nqFine[0],
        method: "EPW-AD",
        warnings: [...(partial.warnings ?? []), "EPW exited with non-zero status"],
      };
    }
    return makePartialResult(warnings, grids);
  }

  console.log(`[EPW] epw.x completed — parsing results`);
  const parsed = parseEPWOutput(epwResult.stdout);

  const result: EPWResult = {
    lambda: parsed.lambda ?? NaN,
    lambdaMax: parsed.lambdaMax ?? NaN,
    omegaLog: parsed.omegaLog ?? NaN,
    tcAllenDynes: parsed.tcAllenDynes ?? NaN,
    tcMigdalEliashberg: parsed.tcMigdalEliashberg ?? NaN,
    gapZero: parsed.gapZero ?? NaN,
    alpha2F: parsed.alpha2F ?? { omega: [], a2f: [] },
    converged: parsed.converged ?? false,
    nkFine: grids.nkFine[0],
    nqFine: grids.nqFine[0],
    method: parsed.method ?? "EPW-AD",
    warnings: [...warnings, ...(parsed.warnings ?? [])],
  };

  console.log(
    `[EPW] ${formula}: lambda=${result.lambda.toFixed(3)}, ` +
    `omega_log=${result.omegaLog.toFixed(1)} meV, ` +
    `Tc(AD)=${result.tcAllenDynes.toFixed(1)} K, ` +
    `Tc(ME)=${result.tcMigdalEliashberg.toFixed(1)} K, ` +
    `gap=${result.gapZero.toFixed(2)} meV, ` +
    `converged=${result.converged}`,
  );

  return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function safeRun(
  callbacks: EPWPipelineCallbacks,
  binary: string,
  inputFile: string,
  cwd: string,
  timeoutMs: number,
): Promise<{ stdout: string; stderr: string; exitCode: number | null } | null> {
  try {
    return await callbacks.runQEBinary(binary, inputFile, cwd, timeoutMs);
  } catch (e: any) {
    console.error(`[EPW] Exception running ${path.posix.basename(binary)}: ${e.message}`);
    return null;
  }
}

function makePartialResult(
  warnings: string[],
  grids: ReturnType<typeof autoEPWGrids>,
): EPWResult {
  return {
    lambda: NaN,
    lambdaMax: NaN,
    omegaLog: NaN,
    tcAllenDynes: NaN,
    tcMigdalEliashberg: NaN,
    gapZero: NaN,
    alpha2F: { omega: [], a2f: [] },
    converged: false,
    nkFine: grids.nkFine[0],
    nqFine: grids.nqFine[0],
    method: "EPW-AD",
    warnings,
  };
}
