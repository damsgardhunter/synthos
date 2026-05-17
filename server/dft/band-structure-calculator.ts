import * as fs from "fs";
import * as path from "path";
import { IS_WINDOWS, killProcessGracefully, spawnQE } from "./platform-utils";
import { lookupKnownStructure } from "../learning/known-structures";

const QE_BIN_DIR = process.env.QE_BIN_DIR ?? (IS_WINDOWS ? "/usr/bin" : "/nix/store/4rd771qjyb5mls5dkcs614clwdxsagql-quantum-espresso-7.2/bin");
// Bands timeout scales with system size. Small cells (3-4 atoms) need
// ~30 min, but 14-15 atom hydrides with nspin=2 need ~2.5h (90s/kpt ×
// 106 kpts). Li2LaH12 timed out at kpt 37/106 after 60 min; LaH11Li2
// at kpt 44/106. Use 3h as the base — env-overridable via BANDS_TIMEOUT_MS.
const BANDS_TIMEOUT_MS = parseInt(process.env.BANDS_TIMEOUT_MS ?? "", 10) || 10_800_000;

export interface KPointOnPath {
  label: string;
  coords: [number, number, number];
}

export interface OrbitalWeight {
  s: number;
  p: number;
  d: number;
  f: number;
}

export interface BandEigenvalue {
  kIndex: number;
  kCoords: [number, number, number];
  kLabel: string;
  kDistance: number;
  energies: number[];
  weights?: OrbitalWeight[];
}

export interface BandCrossing {
  bandIndex: number;
  kFraction: number;
  energy: number;
  slope: number;
}

export interface BandInversion {
  kLabel: string;
  kIndex: number;
  bandPair: [number, number];
  energyGap: number;
  orbitalSwap: boolean;
  lowerOrbital?: OrbitalWeight;
  upperOrbital?: OrbitalWeight;
  inversionType?: "s-p" | "p-d" | "d-f" | "p-s" | "d-p" | "f-d" | "s-s" | "p-p" | "d-d" | "f-f" | "s-d" | "d-s" | "s-f" | "f-s" | "p-f" | "f-p" | "unknown";
}

export interface VanHoveSingularity {
  bandIndex: number;
  kIndex: number;
  energy: number;
  type: "saddle" | "minimum" | "maximum";
  dosContribution: number;
  pathLimited: boolean;
}

export interface EffectiveMass {
  bandIndex: number;
  kLabel: string;
  direction: string;
  mass: number;
  massComponents?: [number, number, number];
}

export interface DFTBandStructureResult {
  formula: string;
  kPath: string;
  nBands: number;
  nKPoints: number;
  fermiEnergy: number;
  eigenvalues: BandEigenvalue[];
  bandCrossings: BandCrossing[];
  bandInversions: BandInversion[];
  vanHoveSingularities: VanHoveSingularity[];
  effectiveMasses: EffectiveMass[];
  bandWidth: number;
  bandGapAlongPath: number;
  isMetallicAlongPath: boolean;
  flatBandScore: number;
  diracCrossingScore: number;
  topologicalIndicators: {
    bandInversionCount: number;
    parityChanges: number;
    diracPointCount: number;
    nodalLineIndicator: number;
  };
  wallTimeSeconds: number;
  converged: boolean;
  error: string | null;
}

interface HighSymmetryPath {
  labels: string[];
  coords: [number, number, number][];
  nPointsBetween: number;
  breaks?: number[];
}

const CRYSTAL_SYSTEM_PATHS: Record<string, HighSymmetryPath> = {
  cubic_sc: {
    labels: ["G", "X", "M", "G", "R", "X", "M", "R"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0],
      [0.5, 0.5, 0], [0.5, 0.5, 0.5],
    ],
    nPointsBetween: 15,
  },
  cubic_fcc: {
    labels: ["G", "X", "W", "K", "G", "L", "U", "W"],
    coords: [
      [0, 0, 0], [0.5, 0.5, 0], [0.5, 0.75, 0.25],
      [0.375, 0.75, 0.375], [0, 0, 0], [0.5, 0.5, 0.5],
      [0.625, 0.25, 0.625], [0.5, 0.25, 0.75],
    ],
    nPointsBetween: 15,
  },
  cubic_bcc: {
    labels: ["G", "H", "N", "G", "P", "H", "P", "N"],
    coords: [
      [0, 0, 0], [0.5, -0.5, 0.5], [0, 0, 0.5],
      [0, 0, 0], [0.25, 0.25, 0.25], [0.5, -0.5, 0.5],
      [0.25, 0.25, 0.25], [0, 0, 0.5],
    ],
    nPointsBetween: 15,
  },
  hexagonal: {
    labels: ["G", "M", "K", "G", "A", "L", "H", "A"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0],
      [0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.5],
      [1 / 3, 1 / 3, 0.5], [0, 0, 0.5],
    ],
    nPointsBetween: 15,
  },
  tetragonal: {
    labels: ["G", "X", "M", "G", "Z", "R", "A", "Z"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.5],
      [0.5, 0.5, 0.5], [0, 0, 0.5],
    ],
    nPointsBetween: 15,
  },
  orthorhombic: {
    labels: ["G", "X", "S", "Y", "G", "Z", "U", "R"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0.5, 0], [0, 0, 0], [0, 0, 0.5],
      [0.5, 0, 0.5], [0.5, 0.5, 0.5],
    ],
    nPointsBetween: 12,
  },
  monoclinic: {
    labels: ["G", "B", "D", "G", "Z", "C", "E"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5],
      [0.5, 0.5, 0.5],
    ],
    nPointsBetween: 12,
  },
  triclinic: {
    labels: ["G", "X", "Y", "Z", "R", "S", "T", "G"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0],
      [0, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5],
      [0.5, 0, 0.5], [0, 0, 0],
    ],
    nPointsBetween: 10,
  },
};

function guessCrystalSystem(
  elements: string[],
  counts: Record<string, number>,
  cOverA: number,
  formula?: string,
): string {
  // Tier 0: known-structure database has the authoritative crystal system
  // when the formula is in our literature DB. Without this lookup the
  // heuristic below can only return cubic / hexagonal / tetragonal — it
  // never returns orthorhombic, monoclinic, triclinic, or rhombohedral
  // even though CRYSTAL_SYSTEM_PATHS has the proper k-paths for all of
  // them. Result: every monoclinic / triclinic / orthorhombic literature
  // candidate (VO2, ZrO2, HfO2, MnWO4, FeWO4, LaPO4, CePO4, …) silently
  // traversed a cubic high-symmetry k-path with wrong corner labels,
  // producing band plots that didn't correspond to the actual Brillouin
  // zone.
  //
  // known-structures.ts stores `latticeType` as plain "cubic" rather than
  // distinguishing simple-cubic / fcc / bcc — those Bravais variants need
  // different k-paths (cubic_sc Γ-X-M-Γ-R-X-M-R vs cubic_fcc
  // Γ-X-W-K-Γ-L-U-W vs cubic_bcc Γ-H-N-Γ-P-H-P-N). Disambiguate via the
  // first letter of the space-group symbol: P = primitive (sc), I = body-
  // centered (bcc), F = face-centered (fcc). Without this, every cubic
  // literature entry would map back through `CRYSTAL_SYSTEM_PATHS["cubic"]`
  // = undefined and fall into the atom-count heuristic — LaH10 (Fm-3m,
  // 11 atoms) would get cubic_sc instead of the correct cubic_fcc path.
  if (formula) {
    try {
      const ks = lookupKnownStructure(formula);
      if (ks) {
        if (ks.latticeType === "cubic") {
          const sg = ks.spaceGroup ?? "";
          const centering = sg.charAt(0).toUpperCase();
          if (centering === "F") return "cubic_fcc";
          if (centering === "I") return "cubic_bcc";
          if (centering === "P") return "cubic_sc";
          // Unknown centering: fall through to heuristic
        } else if (CRYSTAL_SYSTEM_PATHS[ks.latticeType]) {
          return ks.latticeType;
        }
      }
    } catch { /* fall through to heuristic */ }
  }

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasFe = elements.includes("Fe");
  const hasAs = elements.includes("As") || elements.includes("P") || elements.includes("Se");
  const hasB = elements.includes("B");

  if (hasCu && hasO && cOverA > 2.5) return "tetragonal";
  if (hasFe && hasAs) return "tetragonal";
  if (hasB && elements.length === 2) return "hexagonal";

  if (cOverA > 1.8) return "hexagonal";
  if (cOverA > 1.2) return "tetragonal";

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  if (totalAtoms <= 2) return "cubic_bcc";
  if (totalAtoms <= 4) return "cubic_fcc";
  return "cubic_sc";
}

function detectPathBreaks(path: HighSymmetryPath): number[] {
  const breaks: number[] = [];
  const BREAK_THRESHOLD = 0.15;

  for (let i = 0; i < path.coords.length - 1; i++) {
    const [x1, y1, z1] = path.coords[i];
    const [x2, y2, z2] = path.coords[i + 1];

    let minDist = Infinity;
    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        for (let oz = -1; oz <= 1; oz++) {
          const dx = (x2 + ox) - x1;
          const dy = (y2 + oy) - y1;
          const dz = (z2 + oz) - z1;
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (d < minDist) minDist = d;
        }
      }
    }

    if (minDist > BREAK_THRESHOLD && i > 0) {
      breaks.push(i);
    }
  }

  return breaks;
}

function getKPath(crystalSystem: string): HighSymmetryPath {
  const path = CRYSTAL_SYSTEM_PATHS[crystalSystem] || CRYSTAL_SYSTEM_PATHS["cubic_sc"];
  if (!path.breaks) {
    path.breaks = detectPathBreaks(path);
  }
  return path;
}

const IBRAV_MAP: Record<string, number> = {
  "cubic_sc": 1,
  "cubic_fcc": 2,
  "cubic_bcc": 3,
  "cubic": 1,
  "hexagonal": 4,
  "tetragonal": 6,
  "tetragonal_bct": 7,
  "orthorhombic": 8,
  "rhombohedral": 5,
  "trigonal": 5,
  "monoclinic": 12,
  "triclinic": 14,
};

function crystalSystemToIbrav(system: string): number {
  const ibrav = IBRAV_MAP[system];
  if (ibrav !== undefined) return ibrav;
  console.warn(`[band-structure-calculator] Unknown crystal system "${system}" — defaulting to ibrav=1 (simple cubic). k-path may not match Brillouin zone.`);
  return 1;
}

function generateBandsInput(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  kPath: HighSymmetryPath,
  ecutwfc: number,
  nspin: number,
  crystalSystem: string = "cubic_sc",
  cOverA: number = 1.0,
  latticeB: number = latticeA,
  ecutrhoOverride?: number,
  pseudoDir?: string,
): string {
  const ELEMENT_DATA: Record<string, number> = {
    H: 1.008, He: 4.003, Li: 6.941, Be: 9.012, B: 10.811, C: 12.011,
    N: 14.007, O: 15.999, F: 18.998, Na: 22.990, Mg: 24.305, Al: 26.982,
    Si: 28.086, P: 30.974, S: 32.065, Cl: 35.453, K: 39.098, Ca: 40.078,
    Sc: 44.956, Ti: 47.867, V: 50.942, Cr: 51.996, Mn: 54.938, Fe: 55.845,
    Co: 58.933, Ni: 58.693, Cu: 63.546, Zn: 65.380, Ga: 69.723, Ge: 72.640,
    As: 74.922, Se: 78.960, Rb: 85.468, Sr: 87.620, Y: 88.906, Zr: 91.224,
    Nb: 92.906, Mo: 95.960, Ru: 101.07, Rh: 102.91, Pd: 106.42, Ag: 107.87,
    Cd: 112.41, In: 114.82, Sn: 118.71, Sb: 121.76, Te: 127.60, I: 126.90,
    Cs: 132.91, Ba: 137.33, La: 138.91, Ce: 140.12, Hf: 178.49, Ta: 180.95,
    W: 183.84, Re: 186.21, Os: 190.23, Ir: 192.22, Pt: 195.08, Au: 196.97,
    Hg: 200.59, Tl: 204.38, Pb: 207.2, Bi: 208.98, Br: 79.904, Tc: 98.0,
    Pr: 140.91, Nd: 144.24, Pm: 145.0, Sm: 150.36, Eu: 151.96, Gd: 157.25, Tb: 158.93,
    Dy: 162.50, Ho: 164.93, Er: 167.26, Tm: 168.93, Yb: 173.04, Lu: 174.97,
    // Actinides — Pm and Np/Pu/Am/Cm/Bk/Cf were missing → band structure
    // generation threw "Unknown element" for any actinide compound, blocking
    // band plots for heavy-fermion SC candidates (UPt3, PuCoGa5, etc.).
    Ac: 227.03, Th: 232.04, Pa: 231.04, U: 238.03, Np: 237.05, Pu: 244.0,
    Am: 243.0, Cm: 247.0, Bk: 247.0, Cf: 251.0,
  };

  // Pseudo z_valence for each element — kept in sync with qe-worker.ts
  // ELEMENT_DATA (verified against server/dft/pseudo/*.UPF). Used to size
  // nbnd correctly: heavy elements with semicore (Cu=19, Au=19, Hg=20)
  // need many more bands than a generic 4/atom heuristic.
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
    // Actinides — Ac, Np, Pu, Am, Cm previously missing. Without these the
    // band-structure nbnd computation fell through to ?? 8 (line 319),
    // computing far too few conduction bands for the band path. For Pu
    // (real z_val=16): 8 valence electrons assumed → nbnd undercounted by
    // ~50%, causing the band plot to truncate states near E_F.
    Ac: 11, Th: 12, Pa: 13, U: 14, Np: 15, Pu: 16, Am: 17, Cm: 18,
  };

  const totalAtoms = positions.length;
  const nTypes = elements.length;

  // Count total valence electrons in the cell using actual per-position
  // elements (handles supercells correctly). nbnd must cover at least
  // ceil(nelec/2) occupied bands + a buffer for the conduction window we
  // need along the k-path (10–30 extra is standard for band plots).
  let cellElectrons = 0;
  for (const pos of positions) {
    cellElectrons += Z_VALENCE[pos.element] ?? 8;
  }
  const occupiedBands = Math.ceil(cellElectrons / 2);
  const conductionBuffer = Math.max(15, Math.ceil(occupiedBands * 0.20));
  const nbndComputed = (nspin === 2 ? 2 : 1) * (occupiedBands + conductionBuffer);
  const nbndFinal = Math.max(nbndComputed, 20);
  // Default 4x (PAW/NC). Caller should pass ecutrhoOverride computed via ecutrhoMultiplier()
  // to use 8x for USPP. The old hardcoded 8x caused FFT OOM for PAW hydrides (LaH10 etc).
  const ecutrho = ecutrhoOverride ?? ecutwfc * 4;
  const cleanPrefix = formula.replace(/[^a-zA-Z0-9]/g, "");

  let atomicSpecies = "";
  for (const el of elements) {
    const mass = ELEMENT_DATA[el];
    if (mass === undefined) {
      throw new Error(`Unknown element "${el}" — no atomic mass data available. Cannot generate valid QE input.`);
    }
    atomicSpecies += `  ${el}  ${mass.toFixed(3)}  ${el}.UPF\n`;
  }

  let atomicPositions = "";
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  // Use ibrav=0 with explicit CELL_PARAMETERS so the bands input cell matches
  // whatever cell the upstream SCF used (which goes via ibrav=0 + CELL_PARAMETERS
  // and may be monoclinic). The previous ibrav-based approach forced a cubic
  // cell (since guessCrystalSystem only returns cubic/tetragonal/hexagonal) and
  // hardcoded β=100° for monoclinic — both of which mismatch the SCF .save
  // cell. QE then either crashes on cell-inconsistency or silently uses the
  // .save cell, but the input's k-path coordinates (in {crystal_b} units) are
  // fractional reciprocal lattice — interpreting them in the wrong cell gave
  // wrong band-path traversal for non-cubic crystals.
  const ibrav = 0;
  // NO celldm(1)/A: with ibrav=0 the cell is fully defined by CELL_PARAMETERS
  // {angstrom} below. Specifying celldm(1) as well makes QE abort with
  // "lattice parameter specified twice" (the cell scale is given twice —
  // absolute Å in CELL_PARAMETERS AND via celldm). K_POINTS {crystal_b} uses
  // fractional reciprocal-lattice coordinates, which QE derives from
  // CELL_PARAMETERS — it does not need an explicit length scale (only the
  // {tpiba_b} unit would, and we don't use it).
  // Build CELL_PARAMETERS using cOverA / latticeB (passed by caller from
  // estimateCOverA/BOverA). For monoclinic candidates this still doesn't
  // capture β (would need a separate lookup), but at least gets the
  // a-b-c proportions right — orthorhombic shape matches what the SCF
  // does for orthorhombic and is a better starting approximation for
  // monoclinic than forced-cubic ibrav=1.
  const cellB = latticeB > 0 ? latticeB : latticeA;
  const cellC = cOverA > 0 ? latticeA * cOverA : latticeA;
  const cellParamsBlock = `CELL_PARAMETERS {angstrom}
  ${latticeA.toFixed(8)}  0.000000000  0.000000000
  0.000000000  ${cellB.toFixed(8)}  0.000000000
  0.000000000  0.000000000  ${cellC.toFixed(8)}`;
  void crystalSystem;  // cell shape now comes from CELL_PARAMETERS — no longer routed via ibrav

  // pseudo_dir: was hardcoded to "/tmp/qe_pseudo" which only worked on
  // the specific Linux setup where pseudos happened to land at /tmp.
  // On Windows the pseudos live under %TEMP% (translated to /mnt/c/.../
  // qe_pseudo via WSL), and on Linux they may live under $TMPDIR/qe_pseudo
  // or any other location set by getTempSubdir(). Accept the path from the
  // caller (which has access to QE_PSEUDO_DIR_INPUT) so this input file
  // matches the actual pseudo location on every platform.
  const effPseudoDir = pseudoDir ?? "/tmp/qe_pseudo";
  return `&CONTROL
  calculation = 'bands',
  restart_mode = 'from_scratch',
  prefix = '${cleanPrefix}',
  outdir = './tmp',
  pseudo_dir = '${effPseudoDir}',
  verbosity = 'high',
/
&SYSTEM
  ibrav = ${ibrav},
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.02,
  nspin = ${nspin},
  nbnd = ${nbndFinal},
/
&ELECTRONS
  electron_maxstep = 100,
  conv_thr = 1.0d-6,
  mixing_beta = 0.3,
  diagonalization = 'david',
/
ATOMIC_SPECIES
${atomicSpecies}
${cellParamsBlock}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {crystal_b}
${kPath.labels.length}
${kPath.coords.map((c, i) => `  ${c[0].toFixed(8)}  ${c[1].toFixed(8)}  ${c[2].toFixed(8)}  ${kPath.nPointsBetween}  ! ${kPath.labels[i]}`).join("\n")}
`;
}

function generateBandsPostInput(formula: string): string {
  const cleanPrefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  return `&BANDS
  prefix = '${cleanPrefix}',
  outdir = './tmp',
  filband = 'bands.dat',
  lsym = .true.,
/
`;
}

function generateProjwfcInput(formula: string): string {
  const cleanPrefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  return `&PROJWFC
  prefix = '${cleanPrefix}',
  outdir = './tmp',
  filpdos = 'pdos',
  lsym = .true.,
  lwrite_overlaps = .false.,
/
`;
}

function parseProjwfcOutput(
  jobDir: string,
  nKPoints: number,
  nBands: number,
): OrbitalWeight[][] | null {
  // IMPORTANT: this parser previously read `filpdos`-style files (energy-grid
  // resolved partial DOS, ~1000 rows per file) and incorrectly indexed them
  // as `filproj`-style (k-point × band tuples). The resulting "weights" were
  // energy-bin DOS values mis-mapped to (kpt, band), producing garbage
  // orbital character for every band. Downstream consumers (extractFermiPockets
  // orbital character, pairing-channel classification) silently used this
  // garbage instead of the stoichiometric fallback.
  //
  // To get band-resolved projections we'd need to:
  //   (a) Add `filproj = 'proj'` to the projwfc.x input (creates proj.projwfc_up
  //       in the QE-specific band-by-band format), and
  //   (b) Rewrite this function to parse that format.
  //
  // Until that refactor lands, return null so the stoichiometric fallback in
  // extractFermiPockets (which now correctly separates f from d for heavy-
  // fermion / lanthanide systems) provides the orbital character — better
  // honest stoichiometric estimates than mis-indexed DOS values pretending
  // to be band-resolved.
  void jobDir;
  void nKPoints;
  void nBands;
  return null;
}

function mergeOrbitalWeights(eigenvalues: BandEigenvalue[], orbWeights: OrbitalWeight[][] | null): void {
  if (!orbWeights) return;
  for (let ki = 0; ki < eigenvalues.length && ki < orbWeights.length; ki++) {
    eigenvalues[ki].weights = orbWeights[ki];
  }
}

function runQEBands(binary: string, inputFile: string, workDir: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve) => {
    const proc = spawnQE(binary, { cwd: workDir, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    const inputStream = fs.createReadStream(inputFile);
    inputStream.pipe(proc.stdin);

    proc.stdout.on("data", (data: Buffer) => { stdout += data.toString(); });
    proc.stderr.on("data", (data: Buffer) => { stderr += data.toString(); });

    const timeout = setTimeout(() => {
      killProcessGracefully(proc);
      resolve({ stdout, stderr: stderr + "\nTIMEOUT: Band structure calculation exceeded time limit", exitCode: -1 });
    }, BANDS_TIMEOUT_MS);

    proc.on("close", (code: number | null) => {
      clearTimeout(timeout);
      resolve({ stdout, stderr, exitCode: code ?? -1 });
    });

    proc.on("error", (err: Error) => {
      clearTimeout(timeout);
      resolve({ stdout, stderr: err.message, exitCode: -1 });
    });
  });
}

function parseBandsOutput(
  pwOutput: string,
  bandsDataPath: string,
  kPath: HighSymmetryPath,
  fermiEnergy: number,
): { eigenvalues: BandEigenvalue[]; nBands: number; nKPoints: number; converged: boolean } {
  const eigenvalues: BandEigenvalue[] = [];
  let nBands = 0;
  let nKPoints = 0;
  let converged = false;

  const convMatch = pwOutput.match(/convergence has been achieved|End of band structure calculation/);
  if (convMatch) converged = true;

  const nbndMatch = pwOutput.match(/number of Kohn-Sham states\s*=\s*(\d+)/);
  if (nbndMatch) nBands = parseInt(nbndMatch[1]);

  if (fs.existsSync(bandsDataPath)) {
    try {
      const bandsContent = fs.readFileSync(bandsDataPath, "utf-8");
      const result = parseBandsDatFile(bandsContent, kPath, fermiEnergy);
      return { eigenvalues: result.eigenvalues, nBands: result.nBands, nKPoints: result.nKPoints, converged };
    } catch (err) {
      console.log(`[BandCalc] Failed to parse bands.dat: ${err}`);
    }
  }

  const kPointBlocks = pwOutput.split(/k\s*=/).slice(1);
  let kIdx = 0;

  for (const block of kPointBlocks) {
    const coordMatch = block.match(/^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
    if (!coordMatch) continue;

    const kCoords: [number, number, number] = [
      parseFloat(coordMatch[1]),
      parseFloat(coordMatch[2]),
      parseFloat(coordMatch[3]),
    ];

    const energyLines = block.match(/[-\d.]+\s+[-\d.]+/g) || [];
    const energies: number[] = [];
    const numberPattern = /[-]?\d+\.\d+/g;
    const allNumbers = block.match(numberPattern);
    if (allNumbers && allNumbers.length > 3) {
      for (let i = 3; i < allNumbers.length && energies.length < (nBands || 100); i++) {
        const e = parseFloat(allNumbers[i]);
        if (Math.abs(e) < 100) {
          energies.push(e);
        }
      }
    }

    if (energies.length > 0) {
      const segIdx = Math.floor(kIdx / (kPath.nPointsBetween || 15));
      const label = segIdx < kPath.labels.length ? kPath.labels[segIdx] : "";

      let kDist = 0;
      if (eigenvalues.length > 0) {
        const prev = eigenvalues[eigenvalues.length - 1];
        const dk = Math.sqrt(
          (kCoords[0] - prev.kCoords[0]) ** 2 +
          (kCoords[1] - prev.kCoords[1]) ** 2 +
          (kCoords[2] - prev.kCoords[2]) ** 2,
        );
        kDist = prev.kDistance + (dk < 0.3 ? dk : 0.05);
      }

      eigenvalues.push({
        kIndex: kIdx,
        kCoords,
        kLabel: label,
        kDistance: kDist,
        energies: energies.map(e => e - fermiEnergy),
      });
      nKPoints++;
    }
    kIdx++;
  }

  if (nBands === 0 && eigenvalues.length > 0) {
    nBands = eigenvalues[0].energies.length;
  }

  return { eigenvalues, nBands, nKPoints, converged };
}

function parseBandsDatFile(
  content: string,
  kPath: HighSymmetryPath,
  fermiEnergy: number,
): { eigenvalues: BandEigenvalue[]; nBands: number; nKPoints: number } {
  const lines = content.trim().split("\n");
  const eigenvalues: BandEigenvalue[] = [];

  const headerMatch = lines[0]?.match(/nbnd=\s*(\d+),\s*nks=\s*(\d+)/);
  const nBands = headerMatch ? parseInt(headerMatch[1]) : 0;
  const nKPoints = headerMatch ? parseInt(headerMatch[2]) : 0;

  if (nBands === 0 || nKPoints === 0) {
    return { eigenvalues, nBands, nKPoints };
  }

  let lineIdx = 1;
  let cumDistance = 0;
  let prevCoords: [number, number, number] | null = null;

  for (let ki = 0; ki < nKPoints && lineIdx < lines.length; ki++) {
    const kLine = lines[lineIdx]?.trim();
    lineIdx++;

    const kMatch = kLine?.match(/([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
    if (!kMatch) continue;

    const kCoords: [number, number, number] = [
      parseFloat(kMatch[1]),
      parseFloat(kMatch[2]),
      parseFloat(kMatch[3]),
    ];

    if (prevCoords) {
      const dk = Math.sqrt(
        (kCoords[0] - prevCoords[0]) ** 2 +
        (kCoords[1] - prevCoords[1]) ** 2 +
        (kCoords[2] - prevCoords[2]) ** 2,
      );
      const PATH_BREAK_THRESHOLD = 0.3;
      if (dk < PATH_BREAK_THRESHOLD) {
        cumDistance += dk;
      } else {
        cumDistance += 0.05;
      }
    }
    prevCoords = kCoords;

    const energies: number[] = [];
    const bandsPerLine = 10;
    const nEnergyLines = Math.ceil(nBands / bandsPerLine);

    for (let el = 0; el < nEnergyLines && lineIdx < lines.length; el++) {
      const eLine = lines[lineIdx]?.trim();
      lineIdx++;
      if (!eLine) continue;
      const nums = eLine.split(/\s+/).map(Number).filter(n => !isNaN(n));
      energies.push(...nums);
    }

    let kLabel = "";
    const segLength = nKPoints / Math.max(1, kPath.labels.length - 1);
    const segIdx = Math.round(ki / segLength);
    if (ki % Math.round(segLength) < 1 && segIdx < kPath.labels.length) {
      kLabel = kPath.labels[segIdx];
    }
    if (ki === 0) kLabel = kPath.labels[0];
    if (ki === nKPoints - 1) kLabel = kPath.labels[kPath.labels.length - 1];

    eigenvalues.push({
      kIndex: ki,
      kCoords,
      kLabel,
      kDistance: cumDistance,
      energies: energies.slice(0, nBands).map(e => e - fermiEnergy),
    });
  }

  return { eigenvalues, nBands, nKPoints };
}

function isTRIMPoint(kCoords: [number, number, number]): boolean {
  const tol = 0.01;
  return kCoords.every(c => {
    const wrapped = ((c % 1) + 1) % 1;
    return wrapped < tol || wrapped > 1 - tol || Math.abs(wrapped - 0.5) < tol;
  });
}

export function isPathBreak(eigenvalues: BandEigenvalue[], ki: number): boolean {
  if (ki < 1 || ki >= eigenvalues.length) return false;
  const prev = eigenvalues[ki - 1];
  const curr = eigenvalues[ki];
  const dk = Math.sqrt(
    (curr.kCoords[0] - prev.kCoords[0]) ** 2 +
    (curr.kCoords[1] - prev.kCoords[1]) ** 2 +
    (curr.kCoords[2] - prev.kCoords[2]) ** 2,
  );
  return dk > 0.25;
}

type OrbitalLabel = "s" | "p" | "d" | "f";

function dominantOrbital(w: OrbitalWeight): OrbitalLabel {
  const entries: [OrbitalLabel, number][] = [["s", w.s], ["p", w.p], ["d", w.d], ["f", w.f]];
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0];
}

function dominantOrbitalPair(w: OrbitalWeight): [OrbitalLabel, OrbitalLabel | null] {
  const entries: [OrbitalLabel, number][] = [["s", w.s], ["p", w.p], ["d", w.d], ["f", w.f]];
  entries.sort((a, b) => b[1] - a[1]);
  const top = entries[0];
  const second = entries[1];
  if (second[1] > 0.01 && top[1] - second[1] < 0.05) {
    return [top[0], second[0]];
  }
  return [top[0], null];
}

const VALID_INVERSION_TYPES = new Set<string>([
  "s-p", "p-d", "d-f", "p-s", "d-p", "f-d",
  "s-s", "p-p", "d-d", "f-f",
  "s-d", "d-s", "s-f", "f-s", "p-f", "f-p",
]);

function classifyInversionType(lower: OrbitalLabel, upper: OrbitalLabel): BandInversion["inversionType"] {
  const key = `${lower}-${upper}`;
  if (VALID_INVERSION_TYPES.has(key)) return key as BandInversion["inversionType"];
  return "unknown";
}

function estimateAnisotropicMass(
  eigenvalues: BandEigenvalue[],
  bandIndex: number,
  kIndex: number,
  dkSq: number,
  massConv: number,
): [number, number, number] {
  const ki = kIndex;
  if (ki < 1 || ki >= eigenvalues.length - 1) return [1.0, 1.0, 1.0];

  const kPrev = eigenvalues[ki - 1].kCoords;
  const kCurr = eigenvalues[ki].kCoords;
  const kNext = eigenvalues[ki + 1].kCoords;

  const ePrev = eigenvalues[ki - 1].energies[bandIndex];
  const eCurr = eigenvalues[ki].energies[bandIndex];
  const eNext = eigenvalues[ki + 1].energies[bandIndex];

  if (ePrev === undefined || eCurr === undefined || eNext === undefined) return [1.0, 1.0, 1.0];

  const d2E = ePrev + eNext - 2 * eCurr;
  if (Math.abs(d2E) < 1e-6) return [1.0, 1.0, 1.0];

  // m*/m_e = (ℏ²/m_e) / (d²E/dk²). `massConv` carries ℏ²/m_e together with
  // the tpiba→Å⁻¹ unit conversion (see computeMassConv in analyzeBands).
  const pathMass = massConv / (d2E / dkSq);

  const dkVec = [
    kNext[0] - kPrev[0],
    kNext[1] - kPrev[1],
    kNext[2] - kPrev[2],
  ];
  const dkMag = Math.sqrt(dkVec[0] ** 2 + dkVec[1] ** 2 + dkVec[2] ** 2) || 1e-6;

  const dirFrac = [
    Math.abs(dkVec[0] / dkMag),
    Math.abs(dkVec[1] / dkMag),
    Math.abs(dkVec[2] / dkMag),
  ];

  // Effective mass m*/m_e from band curvature. The previous code clamped
  // |m*| to [0.01, 50] m_e, but real materials span a wider range:
  //   - Dirac/Weyl bands (graphene, ZrTe5): |m*| → 0
  //   - Heavy fermion / flat-band (kagome, twisted bilayer): |m*| > 100 m_e
  // The clamp silently flattened both regimes — exactly the physics signal
  // we want to capture for superconductor screening (low m* → high v_F,
  // narrow bands → flat-band SC). Loosen the bounds substantially; only
  // guard against truly pathological numerical outputs (Inf, |m*| > 10⁴).
  const sanitizeMass = (v: number) => {
    if (!Number.isFinite(v)) return NaN;
    const abs = Math.abs(v);
    if (abs > 10000) return Math.sign(v) * 10000;  // numerical sanity only
    if (abs < 1e-6) return Math.sign(v) * 1e-6;     // avoid divide-by-zero downstream
    return v;
  };
  const clampedPath = sanitizeMass(pathMass);

  return [
    dirFrac[0] > 0.3 ? clampedPath : NaN,
    dirFrac[1] > 0.3 ? clampedPath : NaN,
    dirFrac[2] > 0.3 ? clampedPath : NaN,
  ];
}

function analyzeBands(
  eigenvalues: BandEigenvalue[],
  nBands: number,
  fermiEnergy: number,
  latticeA: number,
): {
  bandCrossings: BandCrossing[];
  bandInversions: BandInversion[];
  vanHoveSingularities: VanHoveSingularity[];
  effectiveMasses: EffectiveMass[];
  bandWidth: number;
  bandGapAlongPath: number;
  isMetallicAlongPath: boolean;
  flatBandScore: number;
  diracCrossingScore: number;
  topologicalIndicators: {
    bandInversionCount: number;
    parityChanges: number;
    diracPointCount: number;
    nodalLineIndicator: number;
  };
} {
  const bandCrossings: BandCrossing[] = [];
  const bandInversions: BandInversion[] = [];
  const vanHoveSingularities: VanHoveSingularity[] = [];
  const effectiveMasses: EffectiveMass[] = [];

  // Effective-mass unit conversion. QE bands.x writes k-points in tpiba
  // units (2π/alat, alat = celldm(1) = the a-axis). The curvature 1/(d²E/dk²)
  // is therefore in tpiba²/eV, NOT m*/m_e. The physical effective mass is
  //   m*/m_e = (ℏ²/m_e) / (d²E/dk²[eV·Å²])
  // and  d²E/dk²[eV·Å²] = (d2E/dkSq)[eV/tpiba²] / (2π/a[Å])²,
  // so   m*/m_e = (ℏ²/m_e)·(2π/a)² / (d2E/dkSq).
  // ℏ²/m_e = 7.61996 eV·Å² (= 2 × the ℏ²/2m_e = 3.80998 eV·Å² kinetic
  // constant). Without this factor effective masses came out ~10-20× too
  // small AND lattice-dependent, so they were neither m*/m_e nor comparable
  // across materials. Falls back to 1 (old behavior) if latticeA is invalid.
  const HBAR_SQ_OVER_ME_EV_ANG2 = 7.61996;
  const massConv = latticeA > 0
    ? HBAR_SQ_OVER_ME_EV_ANG2 * (2 * Math.PI / latticeA) ** 2
    : 1;

  if (eigenvalues.length < 2 || nBands === 0) {
    return {
      bandCrossings, bandInversions, vanHoveSingularities, effectiveMasses,
      bandWidth: 0, bandGapAlongPath: 0, isMetallicAlongPath: false,
      flatBandScore: 0, diracCrossingScore: 0,
      topologicalIndicators: { bandInversionCount: 0, parityChanges: 0, diracPointCount: 0, nodalLineIndicator: 0 },
    };
  }

  // Eigenvalues are ALREADY Fermi-shifted in parseBandsOutput (line 556)
  // and parseBandsDatFile (line 644). Do NOT subtract fermiEnergy again —
  // double subtraction pushes all bands negative so no Fermi crossings are
  // detected, producing the contradictory "metallic=false, gap=0.000".
  // CaPbY, Bi2CuSe3 both hit this bug.

  let globalMin = Infinity;
  let globalMax = -Infinity;

  for (const kpt of eigenvalues) {
    for (const e of kpt.energies) {
      if (e < globalMin) globalMin = e;
      if (e > globalMax) globalMax = e;
    }
  }
  const bandWidth = globalMax - globalMin;

  for (let b = 0; b < nBands; b++) {
    for (let ki = 0; ki < eigenvalues.length - 1; ki++) {
      if (isPathBreak(eigenvalues, ki + 1)) continue;
      const e1 = eigenvalues[ki].energies[b];
      const e2 = eigenvalues[ki + 1].energies[b];
      if (e1 === undefined || e2 === undefined) continue;

      if ((e1 <= 0 && e2 >= 0) || (e1 >= 0 && e2 <= 0)) {
        // Normalize kFraction to [0, 1] using (length - 1) so the endpoint
        // reaches exactly 1.0. The previous `ki / length` only reached
        // (n-1)/n ≈ 0.983 for the last k-point — when downstream code
        // round-trips via `Math.round(kFrac * (nK - 1))` (see
        // dft-band-analysis.ts:113), that off-by-1/n shift caused the last
        // k-point's crossings to map back to kIdx = n-2 instead of n-1,
        // looking at the wrong eigenvalues for local-curvature analysis.
        const fraction = eigenvalues.length > 1 ? ki / (eigenvalues.length - 1) : 0;
        const slope = (e2 - e1) / (eigenvalues[ki + 1].kDistance - eigenvalues[ki].kDistance || 1);
        bandCrossings.push({
          bandIndex: b,
          kFraction: fraction,
          energy: (e1 + e2) / 2,
          slope,
        });
      }
    }
  }

  // A band that never changes sign across E_F can still be partially
  // occupied — metallic — if it grazes within the electronic smearing
  // window. The SCF uses Marzari-Vanderbilt smearing (degauss = 0.02 Ry);
  // a state within ~0.1 eV of E_F carries fractional occupation. The pure
  // sign-change test above misses such grazing bands (e.g. a band whose
  // maximum sits 0.05 eV below E_F), wrongly classifying the material
  // insulating. Detect near-Fermi bands in addition to true crossings.
  const FERMI_WINDOW_EV = 0.10;
  let hasNearFermiBand = false;
  for (const kpt of eigenvalues) {
    for (const e of kpt.energies) {
      if (e !== undefined && Math.abs(e) < FERMI_WINDOW_EV) { hasNearFermiBand = true; break; }
    }
    if (hasNearFermiBand) break;
  }
  const isMetallicAlongPath = bandCrossings.length > 0 || hasNearFermiBand;

  let bandGapAlongPath = 0;
  if (!isMetallicAlongPath) {
    // Fundamental (indirect) gap = min_k(CBM) - max_k(VBM), with the
    // conduction-band minimum and valence-band maximum taken INDEPENDENTLY
    // over all sampled k-points. The previous code computed min_k(CBM_k -
    // VBM_k) — the minimum DIRECT gap — which overestimates the gap for any
    // indirect-gap material (min(a-b) >= min(a) - max(b)). That pushed
    // small-indirect-gap semiconductors into the "insulating" bucket in
    // dft-band-analysis.ts, hiding pressure-induced-metal SC candidates.
    // Energies are already E - E_F, so occupied = E < 0, unoccupied = E >= 0.
    let globalVBM = -Infinity;  // highest occupied state over all path k-points
    let globalCBM = Infinity;   // lowest unoccupied state over all path k-points
    for (const kpt of eigenvalues) {
      for (const e of kpt.energies) {
        if (e === undefined) continue;
        if (e < 0) { if (e > globalVBM) globalVBM = e; }
        else { if (e < globalCBM) globalCBM = e; }
      }
    }
    if (globalVBM > -Infinity && globalCBM < Infinity) {
      bandGapAlongPath = Math.max(0, globalCBM - globalVBM);
    }
  }

  for (let b = 0; b < nBands - 1; b++) {
    for (let ki = 0; ki < eigenvalues.length; ki++) {
      const eLower = eigenvalues[ki].energies[b];
      const eUpper = eigenvalues[ki].energies[b + 1];
      if (eLower === undefined || eUpper === undefined) continue;

      if (ki > 0 && ki < eigenvalues.length - 1 && !isPathBreak(eigenvalues, ki) && !isPathBreak(eigenvalues, ki + 1)) {
        const eLowerPrev = eigenvalues[ki - 1].energies[b];
        const eUpperPrev = eigenvalues[ki - 1].energies[b + 1];
        const eLowerNext = eigenvalues[ki + 1].energies[b];
        const eUpperNext = eigenvalues[ki + 1].energies[b + 1];

        if (eLowerPrev !== undefined && eUpperPrev !== undefined &&
            eLowerNext !== undefined && eUpperNext !== undefined) {
          const gapHere = eUpper - eLower;
          const gapPrev = eUpperPrev - eLowerPrev;
          const gapNext = eUpperNext - eLowerNext;

          const kpt = eigenvalues[ki];
          const lowerW = kpt.weights?.[b];
          const upperW = kpt.weights?.[b + 1];

          let orbSwap = gapHere < 0;
          let invType: BandInversion["inversionType"] = "unknown";

          if (lowerW && upperW) {
            const [lowerDom, lowerSecond] = dominantOrbitalPair(lowerW);
            const [upperDom, upperSecond] = dominantOrbitalPair(upperW);
            const prevKpt = eigenvalues[ki - 1];
            const prevLowerW = prevKpt.weights?.[b];
            const prevUpperW = prevKpt.weights?.[b + 1];

            if (prevLowerW && prevUpperW) {
              const [prevLowerDom] = dominantOrbitalPair(prevLowerW);
              const [prevUpperDom] = dominantOrbitalPair(prevUpperW);
              if (prevLowerDom !== lowerDom || prevUpperDom !== upperDom) {
                orbSwap = true;
              }
            }

            invType = classifyInversionType(lowerDom, upperDom);
            if (invType === "unknown" && lowerSecond) {
              const alt = classifyInversionType(lowerSecond, upperDom);
              if (alt !== "unknown") invType = alt;
            }
            if (invType === "unknown" && upperSecond) {
              const alt = classifyInversionType(lowerDom, upperSecond);
              if (alt !== "unknown") invType = alt;
            }
          }

          const isNarrowGap = gapPrev > 0 && gapHere < 0.5 && gapNext > 0;
          if (orbSwap || (isNarrowGap && gapHere < 0.05)) {
            bandInversions.push({
              kLabel: kpt.kLabel || `k${ki}`,
              kIndex: ki,
              bandPair: [b, b + 1],
              energyGap: gapHere,
              orbitalSwap: orbSwap,
              lowerOrbital: lowerW,
              upperOrbital: upperW,
              inversionType: invType,
            });
          }
        }
      }
    }
  }

  for (let b = 0; b < nBands; b++) {
    for (let ki = 1; ki < eigenvalues.length - 1; ki++) {
      if (isPathBreak(eigenvalues, ki) || isPathBreak(eigenvalues, ki + 1)) continue;
      const ePrev = eigenvalues[ki - 1].energies[b];
      const eCurr = eigenvalues[ki].energies[b];
      const eNext = eigenvalues[ki + 1].energies[b];
      if (ePrev === undefined || eCurr === undefined || eNext === undefined) continue;

      const d2E = ePrev + eNext - 2 * eCurr;
      const dkSq = ((eigenvalues[ki + 1].kDistance - eigenvalues[ki - 1].kDistance) / 2) ** 2 || 0.01;

      // Van Hove singularity = band extremum (group velocity dE/dk = 0),
      // detected as a sign change of the discrete slope across this k-point.
      // The previous test |d2E| < 0.005 keyed on near-ZERO curvature — that
      // identifies locally LINEAR stretches, the OPPOSITE of an extremum
      // (a real band min/max has LARGE |d2E|). It mislabeled generic
      // near-linear points as vHS and missed every actual band edge.
      const vhsSlopeLeft = eCurr - ePrev;
      const vhsSlopeRight = eNext - eCurr;
      if (vhsSlopeLeft * vhsSlopeRight < 0 && Math.abs(eCurr) < 2.0) {
        // Inside a strict slope sign change: eCurr below both neighbors → min.
        const type: "saddle" | "minimum" | "maximum" = eCurr < ePrev ? "minimum" : "maximum";
        // 1D DOS pile-up at a band edge grows as the extremum flattens, so a
        // smaller |d2E| (flatter edge) → larger dosContribution.
        const dosContrib = 1.0 / (Math.abs(d2E) + 0.001);
        vanHoveSingularities.push({
          bandIndex: b,
          kIndex: ki,
          energy: eCurr,
          type,
          dosContribution: Math.min(dosContrib, 100),
          pathLimited: true,
        });
      }

      if (Math.abs(eCurr) < 1.0 && Math.abs(d2E) > 0.001) {
        // m*/m_e = (ℏ²/m_e)·(2π/a)² / (d²E/dk²) — see massConv above.
        const mEff = massConv / (d2E / dkSq);
        const absM = Math.abs(mEff);
        if (Number.isFinite(mEff) && absM > 1e-6 && absM < 10000) {
          const massComps = estimateAnisotropicMass(eigenvalues, b, ki, dkSq, massConv);
          effectiveMasses.push({
            bandIndex: b,
            kLabel: eigenvalues[ki].kLabel || `k${ki}`,
            direction: "path",
            mass: mEff,
            massComponents: massComps,
          });
        }
      }
    }
  }

  let flatBandScore = 0;
  for (let b = 0; b < nBands; b++) {
    let segStart = 0;
    for (let ki = 0; ki <= eigenvalues.length; ki++) {
      if (ki === eigenvalues.length || isPathBreak(eigenvalues, ki)) {
        const segEnergies: number[] = [];
        for (let si = segStart; si < ki; si++) {
          const e = eigenvalues[si].energies[b];
          if (e !== undefined) segEnergies.push(e);
        }
        if (segEnergies.length >= 3) {
          const bMin = Math.min(...segEnergies);
          const bMax = Math.max(...segEnergies);
          const bRange = bMax - bMin;
          if (bRange < 0.1 && Math.abs((bMin + bMax) / 2) < 2.0) {
            flatBandScore = Math.max(flatBandScore, 1.0 - bRange / 0.1);
          }
        }
        segStart = ki;
      }
    }
  }

  let diracCrossingScore = 0;
  for (const crossing of bandCrossings) {
    if (Math.abs(crossing.energy) < 0.1 && Math.abs(crossing.slope) > 1.0) {
      diracCrossingScore = Math.max(diracCrossingScore, Math.min(1.0, Math.abs(crossing.slope) / 10.0));
    }
  }

  let parityChanges = 0;
  const trimIndices: number[] = [];
  for (let ki = 0; ki < eigenvalues.length; ki++) {
    const kpt = eigenvalues[ki];
    if (isTRIMPoint(kpt.kCoords)) {
      trimIndices.push(ki);
    }
  }
  for (let b = 0; b < nBands; b++) {
    for (let ti = 1; ti < trimIndices.length; ti++) {
      const kiPrev = trimIndices[ti - 1];
      const kiCurr = trimIndices[ti];
      const ePrev = eigenvalues[kiPrev].energies[b];
      const eCurr = eigenvalues[kiCurr].energies[b];
      if (ePrev !== undefined && eCurr !== undefined) {
        if (Math.sign(ePrev) !== Math.sign(eCurr) && Math.abs(ePrev) > 0.01 && Math.abs(eCurr) > 0.01) {
          parityChanges++;
        }
      }
    }
  }

  const diracPointCount = bandCrossings.filter(c => Math.abs(c.energy) < 0.05 && Math.abs(c.slope) > 2.0).length;

  let nodalLineIndicator = 0;
  if (bandCrossings.length > 3) {
    const crossingEnergies = bandCrossings.map(c => c.energy);
    const crossRange = Math.max(...crossingEnergies) - Math.min(...crossingEnergies);
    if (crossRange < 0.1) {
      nodalLineIndicator = Math.min(1.0, bandCrossings.length / 10);
    }
  }

  return {
    bandCrossings: bandCrossings.slice(0, 50),
    bandInversions: bandInversions.slice(0, 20),
    vanHoveSingularities: vanHoveSingularities.slice(0, 30),
    effectiveMasses: effectiveMasses.slice(0, 20),
    bandWidth,
    bandGapAlongPath,
    isMetallicAlongPath,
    flatBandScore,
    diracCrossingScore,
    topologicalIndicators: {
      bandInversionCount: bandInversions.length,
      parityChanges: Math.min(parityChanges, 100),
      diracPointCount,
      nodalLineIndicator,
    },
  };
}

export async function computeDFTBandStructure(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  fermiEnergy: number,
  jobDir: string,
  cOverA: number = 1.0,
  ecutwfc: number = 45,
  nspin: number = 1,
  latticeB: number = latticeA,
  ecutrho?: number,
  pseudoDir?: string,
): Promise<DFTBandStructureResult> {
  const startTime = Date.now();
  // Pass `formula` so guessCrystalSystem can consult lookupKnownStructure
  // and return monoclinic/triclinic/orthorhombic/rhombohedral for literature
  // compounds — without it, the heuristic only returns cubic/hex/tetragonal
  // and the k-path silently misrepresents the true Brillouin zone for any
  // non-cubic literature candidate.
  const crystalSystem = guessCrystalSystem(elements, counts, cOverA, formula);
  const kPath = getKPath(crystalSystem);
  const pathString = kPath.labels.join(" -> ");

  console.log(`[BandCalc] Starting band structure for ${formula} (${crystalSystem}, path: ${pathString})`);

  const result: DFTBandStructureResult = {
    formula,
    kPath: pathString,
    nBands: 0,
    nKPoints: 0,
    fermiEnergy,
    eigenvalues: [],
    bandCrossings: [],
    bandInversions: [],
    vanHoveSingularities: [],
    effectiveMasses: [],
    bandWidth: 0,
    bandGapAlongPath: 0,
    isMetallicAlongPath: false,
    flatBandScore: 0,
    diracCrossingScore: 0,
    topologicalIndicators: {
      bandInversionCount: 0,
      parityChanges: 0,
      diracPointCount: 0,
      nodalLineIndicator: 0,
    },
    wallTimeSeconds: 0,
    converged: false,
    error: null,
  };

  try {
    // Guard: bands calc is non-self-consistent — it reads the SCF charge
    // density + wavefunctions from ${outdir}/${prefix}.save/. If the SCF
    // save directory is missing (tmp cleaned, wrong prefix, SCF never
    // finished its write), pw.x aborts immediately with MPI_ABORT=1 and
    // no useful stderr. Check upfront so we log the real cause.
    const cleanPrefix = formula.replace(/[^a-zA-Z0-9]/g, "");
    const saveDir = path.join(jobDir, "tmp", `${cleanPrefix}.save`);
    const chargeDensity = path.join(saveDir, "charge-density.dat");
    const chargeDensityHdf5 = path.join(saveDir, "charge-density.hdf5");
    if (!fs.existsSync(saveDir) || (!fs.existsSync(chargeDensity) && !fs.existsSync(chargeDensityHdf5))) {
      result.error = `SCF save directory missing or incomplete (looked for ${saveDir}). Bands calculation cannot proceed without SCF wavefunctions.`;
      console.log(`[BandCalc] Skipping bands for ${formula}: ${result.error}`);
      result.wallTimeSeconds = (Date.now() - startTime) / 1000;
      return result;
    }

    const bandsInput = generateBandsInput(formula, elements, counts, latticeA, positions, kPath, ecutwfc, nspin, crystalSystem, cOverA, latticeB, ecutrho, pseudoDir);
    const bandsInputFile = path.join(jobDir, "bands.in");
    fs.writeFileSync(bandsInputFile, bandsInput);

    console.log(`[BandCalc] Running pw.x bands for ${formula}`);
    const pwResult = await runQEBands(
      path.posix.join(QE_BIN_DIR, "pw.x"),
      bandsInputFile,
      jobDir,
    );

    fs.writeFileSync(path.join(jobDir, "bands_pw.out"), pwResult.stdout);

    if (pwResult.exitCode !== 0 && !pwResult.stdout.includes("End of band structure calculation")) {
      // Exit 139 = SIGSEGV, almost always OOM on heavy-element PAW systems
      // (W: addusdens alone is 415 MB). Retry with MPI ranks=1 to give
      // the process all available memory. One-shot retry, not a full ladder.
      if (pwResult.exitCode === 139 && process.env.QE_MPI_RANKS && parseInt(process.env.QE_MPI_RANKS) > 1) {
        console.log(`[BandCalc] pw.x bands OOM (exit=139) for ${formula} at ${process.env.QE_MPI_RANKS} ranks — retrying with 1 rank for more memory per process`);
        const savedRanks = process.env.QE_MPI_RANKS;
        process.env.QE_MPI_RANKS = "1";
        try {
          const retryResult = await runQEBands(
            path.posix.join(QE_BIN_DIR, "pw.x"),
            bandsInputFile,
            jobDir,
          );
          fs.writeFileSync(path.join(jobDir, "bands_pw_retry.out"), retryResult.stdout);
          if (retryResult.exitCode === 0 || retryResult.stdout.includes("End of band structure calculation")) {
            // Retry succeeded — continue with the bands post-processing
            console.log(`[BandCalc] pw.x bands OOM retry succeeded for ${formula} (1 rank)`);
            Object.assign(pwResult, retryResult);
          }
        } catch (retryErr: any) {
          console.log(`[BandCalc] pw.x bands OOM retry also failed for ${formula}: ${(retryErr.message || "").slice(0, 100)}`);
        } finally {
          process.env.QE_MPI_RANKS = savedRanks;
        }
      }
      // Still failed after retry (or no retry attempted)?
      if (pwResult.exitCode !== 0 && !pwResult.stdout.includes("End of band structure calculation")) {
        const stdoutTail = pwResult.stdout.slice(-1000);
        const iosysMatch = pwResult.stdout.match(/Error in routine[^\n]*\n\s*([^\n]{0,200})/);
        const detail = iosysMatch ? iosysMatch[1].trim() : stdoutTail.slice(-300);
        result.error = `pw.x bands exited with code ${pwResult.exitCode}: ${detail}`;
        console.log(`[BandCalc] pw.x bands failed for ${formula}: ${result.error.slice(0, 250)}`);
        result.wallTimeSeconds = (Date.now() - startTime) / 1000;
        return result;
      }
    }

    const bandsPostInput = generateBandsPostInput(formula);
    const bandsPostFile = path.join(jobDir, "bands_post.in");
    fs.writeFileSync(bandsPostFile, bandsPostInput);

    console.log(`[BandCalc] Running bands.x post-processing for ${formula}`);
    const bandsXResult = await runQEBands(
      path.posix.join(QE_BIN_DIR, "bands.x"),
      bandsPostFile,
      jobDir,
    );

    if (bandsXResult.exitCode !== 0) {
      console.log(`[BandCalc] bands.x warning for ${formula}: exit code ${bandsXResult.exitCode} (continuing with pw.x output)`);
    }

    // For nspin=2, QE's bands.x writes "bands.dat.spinup" and
    // "bands.dat.spindown" instead of "bands.dat". Try the spin-resolved
    // files first; if absent fall back to the non-spin-polarized name.
    const bandsDatCandidates = [
      path.join(jobDir, "bands.dat"),
      path.join(jobDir, "bands.dat.spinup"),
    ];
    const bandsDatPath = bandsDatCandidates.find(p => fs.existsSync(p))
      ?? bandsDatCandidates[0];
    const parsed = parseBandsOutput(pwResult.stdout, bandsDatPath, kPath, fermiEnergy);

    // Merge the spin-down channel. Without it, a half-metal (gap in one
    // spin channel, metallic in the other) is misclassified from the
    // spin-up channel alone. Both channels share the same k-path and band
    // count, so concatenating their energies per k-point lets analyzeBands
    // see the union of states — correct for gap, metallicity and crossings.
    const bandsDatSpinDownPath = path.join(jobDir, "bands.dat.spindown");
    if (fs.existsSync(bandsDatSpinDownPath) && !bandsDatPath.endsWith("bands.dat")
        && parsed.eigenvalues.length > 0) {
      try {
        const parsedDown = parseBandsOutput(pwResult.stdout, bandsDatSpinDownPath, kPath, fermiEnergy);
        if (parsedDown.eigenvalues.length === parsed.eigenvalues.length) {
          for (let ki = 0; ki < parsed.eigenvalues.length; ki++) {
            parsed.eigenvalues[ki].energies = [
              ...parsed.eigenvalues[ki].energies,
              ...parsedDown.eigenvalues[ki].energies,
            ];
          }
          parsed.nBands += parsedDown.nBands;
          console.log(`[BandCalc] nspin=2 — merged spin-down channel ` +
            `(+${parsedDown.nBands} bands); gap/metallicity now reflect both spins`);
        } else {
          console.log(`[BandCalc] nspin=2 — spin-down k-point count mismatch ` +
            `(${parsedDown.eigenvalues.length} vs ${parsed.eigenvalues.length}); using spin-up only`);
        }
      } catch (e: any) {
        console.log(`[BandCalc] nspin=2 — failed to parse spin-down channel: ${e?.message ?? e}`);
      }
    }

    result.eigenvalues = parsed.eigenvalues;
    result.nBands = parsed.nBands;
    result.nKPoints = parsed.nKPoints;
    result.converged = parsed.converged;

    // Skip projwfc.x: the existing parser reads filpdos files (energy-grid
    // DOS) but indexes them as filproj (k-point × band) — see comment in
    // parseProjwfcOutput. Running projwfc.x just wastes 30-60s per band-
    // structure calc since the parser returns null. The downstream
    // extractFermiPockets uses its stoichiometric orbital-character fallback
    // (which now correctly separates f from d for heavy-fermion systems).
    // Re-enable when parseProjwfcOutput is rewritten to parse filproj output
    // (also requires adding `filproj = 'proj'` to generateProjwfcInput).

    if (parsed.eigenvalues.length > 0) {
      const analysis = analyzeBands(parsed.eigenvalues, parsed.nBands, fermiEnergy, latticeA);
      result.bandCrossings = analysis.bandCrossings;
      result.bandInversions = analysis.bandInversions;
      result.vanHoveSingularities = analysis.vanHoveSingularities;
      result.effectiveMasses = analysis.effectiveMasses;
      result.bandWidth = analysis.bandWidth;
      result.bandGapAlongPath = analysis.bandGapAlongPath;
      result.isMetallicAlongPath = analysis.isMetallicAlongPath;
      result.flatBandScore = analysis.flatBandScore;
      result.diracCrossingScore = analysis.diracCrossingScore;
      result.topologicalIndicators = analysis.topologicalIndicators;

      console.log(`[BandCalc] ${formula}: ${parsed.nBands} bands, ${parsed.nKPoints} k-points, ` +
        `${result.bandCrossings.length} Fermi crossings, ${result.bandInversions.length} inversions, ` +
        `gap=${result.bandGapAlongPath.toFixed(3)} eV, metallic=${result.isMetallicAlongPath}, ` +
        `flat=${result.flatBandScore.toFixed(3)}, dirac=${result.diracCrossingScore.toFixed(3)}`);
    } else {
      console.log(`[BandCalc] ${formula}: no eigenvalues parsed from output`);
    }
  } catch (err: any) {
    result.error = `Band structure calculation error: ${err.message}`;
    console.log(`[BandCalc] Error for ${formula}: ${err.message}`);
  }

  result.wallTimeSeconds = (Date.now() - startTime) / 1000;
  return result;
}

let totalBandCalcs = 0;
let totalBandSuccess = 0;
let totalBandFailed = 0;
let avgBandTime = 0;

export function recordBandCalcOutcome(success: boolean, wallTime: number): void {
  totalBandCalcs++;
  if (success) totalBandSuccess++;
  else totalBandFailed++;
  avgBandTime = (avgBandTime * (totalBandCalcs - 1) + wallTime) / totalBandCalcs;
}

export function getBandCalcStats(): {
  totalCalcs: number;
  succeeded: number;
  failed: number;
  avgWallTimeSeconds: number;
} {
  return {
    totalCalcs: totalBandCalcs,
    succeeded: totalBandSuccess,
    failed: totalBandFailed,
    avgWallTimeSeconds: Math.round(avgBandTime * 10) / 10,
  };
}
