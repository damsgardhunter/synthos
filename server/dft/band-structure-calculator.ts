import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

const QE_BIN_DIR = "/nix/store/4rd771qjyb5mls5dkcs614clwdxsagql-quantum-espresso-7.2/bin";
const BANDS_TIMEOUT_MS = 300_000;

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
  inversionType?: "s-p" | "p-d" | "d-f" | "p-s" | "d-p" | "f-d" | "unknown";
}

export interface VanHoveSingularity {
  bandIndex: number;
  kIndex: number;
  energy: number;
  type: "saddle" | "minimum" | "maximum";
  dosContribution: number;
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
};

function guessCrystalSystem(
  elements: string[],
  counts: Record<string, number>,
  cOverA: number,
): string {
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

function crystalSystemToIbrav(system: string): number {
  if (system === "cubic_sc") return 1;
  if (system === "cubic_fcc") return 2;
  if (system === "cubic_bcc") return 3;
  if (system === "hexagonal") return 4;
  if (system === "tetragonal") return 6;
  if (system === "orthorhombic") return 8;
  if (system === "monoclinic") return 12;
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
    Pr: 140.91, Nd: 144.24, Sm: 150.36, Eu: 151.96, Gd: 157.25, Tb: 158.93,
    Dy: 162.50, Ho: 164.93, Er: 167.26, Tm: 168.93, Yb: 173.04, Lu: 174.97,
    Th: 232.04, U: 238.03, Pa: 231.04,
  };

  const totalAtoms = positions.length;
  const nTypes = elements.length;
  const ecutrho = ecutwfc * 8;
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

  const ibrav = crystalSystemToIbrav(crystalSystem);
  const celldm1 = (latticeA * 1.8897259886).toFixed(6);
  let celldmLines = `  celldm(1) = ${celldm1},\n`;
  if ((ibrav === 4 || ibrav === 6) && cOverA > 0 && Math.abs(cOverA - 1.0) > 0.01) {
    celldmLines += `  celldm(3) = ${cOverA.toFixed(6)},\n`;
  }
  if (ibrav === 8 || ibrav === 12) {
    const bOverA = latticeB / latticeA;
    celldmLines += `  celldm(2) = ${bOverA.toFixed(6)},\n`;
    celldmLines += `  celldm(3) = ${cOverA.toFixed(6)},\n`;
    if (ibrav === 12) {
      celldmLines += `  celldm(4) = ${Math.cos(100 * Math.PI / 180).toFixed(6)},\n`;
    }
  }

  return `&CONTROL
  calculation = 'bands',
  restart_mode = 'from_scratch',
  prefix = '${cleanPrefix}',
  outdir = './tmp',
  pseudo_dir = '/tmp/qe_pseudo',
  verbosity = 'high',
/
&SYSTEM
  ibrav = ${ibrav},
${celldmLines}  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.02,
  nspin = ${nspin},
  nbnd = ${Math.max(totalAtoms * 4, 20)},
/
&ELECTRONS
  electron_maxstep = 100,
  conv_thr = 1.0d-6,
  mixing_beta = 0.3,
  diagonalization = 'david',
/
ATOMIC_SPECIES
${atomicSpecies}
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
  const weights: OrbitalWeight[][] = [];

  for (let ki = 0; ki < nKPoints; ki++) {
    const bandWeights: OrbitalWeight[] = [];
    for (let b = 0; b < nBands; b++) {
      bandWeights.push({ s: 0, p: 0, d: 0, f: 0 });
    }
    weights.push(bandWeights);
  }

  const pdosFiles = fs.readdirSync(jobDir).filter(f => f.startsWith("pdos") && f.includes("atm"));
  if (pdosFiles.length === 0) return null;

  for (const pdosFile of pdosFiles) {
    const lMatch = pdosFile.match(/wfc#\d+\((\w+)\)/i) || pdosFile.match(/\((\w+)\)/);
    let orbType: "s" | "p" | "d" | "f" = "s";
    if (lMatch) {
      const label = lMatch[1].toLowerCase();
      if (label === "s" || label.startsWith("s")) orbType = "s";
      else if (label === "p" || label.startsWith("p")) orbType = "p";
      else if (label === "d" || label.startsWith("d")) orbType = "d";
      else if (label === "f" || label.startsWith("f")) orbType = "f";
    }

    try {
      const content = fs.readFileSync(path.join(jobDir, pdosFile), "utf-8");
      const lines = content.trim().split("\n").filter(l => !l.startsWith("#") && l.trim());

      let kIdx = 0;
      for (const line of lines) {
        const nums = line.trim().split(/\s+/).map(Number).filter(n => !isNaN(n));
        if (nums.length < 2) continue;

        const kPt = Math.floor(kIdx / nBands);
        const bandIdx = kIdx % nBands;

        if (kPt < nKPoints && bandIdx < nBands) {
          const projVal = nums[1] ?? 0;
          weights[kPt][bandIdx][orbType] += projVal;
        }
        kIdx++;
      }
    } catch {}
  }

  let hasData = false;
  for (const kw of weights) {
    for (const bw of kw) {
      const total = bw.s + bw.p + bw.d + bw.f;
      if (total > 0.01) {
        hasData = true;
        bw.s /= total;
        bw.p /= total;
        bw.d /= total;
        bw.f /= total;
      }
    }
  }

  return hasData ? weights : null;
}

function mergeOrbitalWeights(eigenvalues: BandEigenvalue[], orbWeights: OrbitalWeight[][] | null): void {
  if (!orbWeights) return;
  for (let ki = 0; ki < eigenvalues.length && ki < orbWeights.length; ki++) {
    eigenvalues[ki].weights = orbWeights[ki];
  }
}

function runQEBands(binary: string, inputFile: string, workDir: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve) => {
    const proc = spawn(binary, { cwd: workDir, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    const inputStream = fs.createReadStream(inputFile);
    inputStream.pipe(proc.stdin);

    proc.stdout.on("data", (data: Buffer) => { stdout += data.toString(); });
    proc.stderr.on("data", (data: Buffer) => { stderr += data.toString(); });

    const timeout = setTimeout(() => {
      proc.kill("SIGKILL");
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

function dominantOrbital(w: OrbitalWeight): "s" | "p" | "d" | "f" {
  const entries: ["s" | "p" | "d" | "f", number][] = [["s", w.s], ["p", w.p], ["d", w.d], ["f", w.f]];
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0];
}

function classifyInversionType(lower: "s" | "p" | "d" | "f", upper: "s" | "p" | "d" | "f"): BandInversion["inversionType"] {
  const key = `${lower}-${upper}`;
  const valid: BandInversion["inversionType"][] = ["s-p", "p-d", "d-f", "p-s", "d-p", "f-d"];
  return (valid.includes(key as any) ? key : "unknown") as BandInversion["inversionType"];
}

function estimateAnisotropicMass(
  eigenvalues: BandEigenvalue[],
  bandIndex: number,
  kIndex: number,
  dkSq: number,
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

  const pathMass = 1.0 / (d2E / dkSq);

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

  const mx = dirFrac[0] > 0.3 ? pathMass : pathMass * (1.0 + 0.5 * (1.0 - dirFrac[0]));
  const my = dirFrac[1] > 0.3 ? pathMass : pathMass * (1.0 + 0.5 * (1.0 - dirFrac[1]));
  const mz = dirFrac[2] > 0.3 ? pathMass : pathMass * (1.0 + 0.5 * (1.0 - dirFrac[2]));

  const clamp = (v: number) => Math.sign(v) * Math.max(0.01, Math.min(50, Math.abs(v)));
  return [clamp(mx), clamp(my), clamp(mz)];
}

function analyzeBands(
  eigenvalues: BandEigenvalue[],
  nBands: number,
  fermiEnergy: number,
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

  if (eigenvalues.length < 2 || nBands === 0) {
    return {
      bandCrossings, bandInversions, vanHoveSingularities, effectiveMasses,
      bandWidth: 0, bandGapAlongPath: 0, isMetallicAlongPath: false,
      flatBandScore: 0, diracCrossingScore: 0,
      topologicalIndicators: { bandInversionCount: 0, parityChanges: 0, diracPointCount: 0, nodalLineIndicator: 0 },
    };
  }

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
        const fraction = ki / eigenvalues.length;
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

  const isMetallicAlongPath = bandCrossings.length > 0;

  let bandGapAlongPath = Infinity;
  if (!isMetallicAlongPath) {
    for (const kpt of eigenvalues) {
      const below = kpt.energies.filter(e => e < 0);
      const above = kpt.energies.filter(e => e >= 0);
      if (below.length > 0 && above.length > 0) {
        const vbm = Math.max(...below);
        const cbm = Math.min(...above);
        const gap = cbm - vbm;
        if (gap < bandGapAlongPath) bandGapAlongPath = gap;
      }
    }
    if (bandGapAlongPath === Infinity) bandGapAlongPath = 0;
  } else {
    bandGapAlongPath = 0;
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

          if (gapPrev > 0 && gapHere < 0.05 && gapNext > 0) {
            const kpt = eigenvalues[ki];
            const lowerW = kpt.weights?.[b];
            const upperW = kpt.weights?.[b + 1];

            let orbSwap = gapHere < 0;
            let invType: BandInversion["inversionType"] = "unknown";

            if (lowerW && upperW) {
              const lowerDom = dominantOrbital(lowerW);
              const upperDom = dominantOrbital(upperW);
              const prevKpt = eigenvalues[ki - 1];
              const prevLowerW = prevKpt.weights?.[b];
              const prevUpperW = prevKpt.weights?.[b + 1];

              if (prevLowerW && prevUpperW) {
                const prevLowerDom = dominantOrbital(prevLowerW);
                const prevUpperDom = dominantOrbital(prevUpperW);
                if (prevLowerDom !== lowerDom || prevUpperDom !== upperDom) {
                  orbSwap = true;
                }
              }

              invType = classifyInversionType(lowerDom, upperDom);
            }

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

      if (Math.abs(d2E) < 0.005) {
        let type: "saddle" | "minimum" | "maximum" = "saddle";
        if (d2E > 0.001) type = "minimum";
        else if (d2E < -0.001) type = "maximum";

        const dosContrib = 1.0 / (Math.abs(d2E) + 0.001);

        if (Math.abs(eCurr) < 2.0) {
          vanHoveSingularities.push({
            bandIndex: b,
            kIndex: ki,
            energy: eCurr,
            type,
            dosContribution: Math.min(dosContrib, 100),
          });
        }
      }

      if (Math.abs(eCurr) < 1.0 && Math.abs(d2E) > 0.001) {
        const mEff = 1.0 / (d2E / dkSq);
        if (Math.abs(mEff) < 50 && Math.abs(mEff) > 0.01) {
          const massComps = estimateAnisotropicMass(eigenvalues, b, ki, dkSq);
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
  for (let b = 0; b < nBands; b++) {
    for (let ki = 1; ki < eigenvalues.length; ki++) {
      if (isPathBreak(eigenvalues, ki)) continue;
      const ePrev = eigenvalues[ki - 1].energies[b];
      const eCurr = eigenvalues[ki].energies[b];
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
): Promise<DFTBandStructureResult> {
  const startTime = Date.now();
  const crystalSystem = guessCrystalSystem(elements, counts, cOverA);
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
    const bandsInput = generateBandsInput(formula, elements, counts, latticeA, positions, kPath, ecutwfc, nspin, crystalSystem, cOverA, latticeB);
    const bandsInputFile = path.join(jobDir, "bands.in");
    fs.writeFileSync(bandsInputFile, bandsInput);

    console.log(`[BandCalc] Running pw.x bands for ${formula}`);
    const pwResult = await runQEBands(
      path.join(QE_BIN_DIR, "pw.x"),
      bandsInputFile,
      jobDir,
    );

    fs.writeFileSync(path.join(jobDir, "bands_pw.out"), pwResult.stdout);

    if (pwResult.exitCode !== 0 && !pwResult.stdout.includes("End of band structure calculation")) {
      result.error = `pw.x bands exited with code ${pwResult.exitCode}: ${pwResult.stderr.slice(0, 300)}`;
      console.log(`[BandCalc] pw.x bands failed for ${formula}: ${result.error.slice(0, 200)}`);
      result.wallTimeSeconds = (Date.now() - startTime) / 1000;
      return result;
    }

    const bandsPostInput = generateBandsPostInput(formula);
    const bandsPostFile = path.join(jobDir, "bands_post.in");
    fs.writeFileSync(bandsPostFile, bandsPostInput);

    console.log(`[BandCalc] Running bands.x post-processing for ${formula}`);
    const bandsXResult = await runQEBands(
      path.join(QE_BIN_DIR, "bands.x"),
      bandsPostFile,
      jobDir,
    );

    if (bandsXResult.exitCode !== 0) {
      console.log(`[BandCalc] bands.x warning for ${formula}: exit code ${bandsXResult.exitCode} (continuing with pw.x output)`);
    }

    const bandsDatPath = path.join(jobDir, "bands.dat");
    const parsed = parseBandsOutput(pwResult.stdout, bandsDatPath, kPath, fermiEnergy);

    result.eigenvalues = parsed.eigenvalues;
    result.nBands = parsed.nBands;
    result.nKPoints = parsed.nKPoints;
    result.converged = parsed.converged;

    if (parsed.eigenvalues.length > 0 && parsed.nBands > 0) {
      try {
        const projwfcBin = path.join(QE_BIN_DIR, "projwfc.x");
        if (fs.existsSync(projwfcBin)) {
          const projInput = generateProjwfcInput(formula);
          const projInputFile = path.join(jobDir, "projwfc.in");
          fs.writeFileSync(projInputFile, projInput);

          console.log(`[BandCalc] Running projwfc.x for orbital weights on ${formula}`);
          const projResult = await runQEBands(projwfcBin, projInputFile, jobDir);

          if (projResult.exitCode === 0) {
            const orbWeights = parseProjwfcOutput(jobDir, parsed.nKPoints, parsed.nBands);
            if (orbWeights) {
              mergeOrbitalWeights(parsed.eigenvalues, orbWeights);
              const orbCount = parsed.eigenvalues.filter(kp => kp.weights && kp.weights.some(w => w.s + w.p + w.d + w.f > 0.01)).length;
              console.log(`[BandCalc] Orbital weights merged for ${formula}: ${orbCount}/${parsed.nKPoints} k-points with data`);
            } else {
              console.log(`[BandCalc] projwfc.x completed but no orbital weights parsed for ${formula}`);
            }
          } else {
            console.log(`[BandCalc] projwfc.x failed for ${formula} (non-critical, continuing without orbital weights)`);
          }
        }
      } catch (projErr: any) {
        console.log(`[BandCalc] projwfc.x error for ${formula}: ${projErr.message} (continuing without orbital weights)`);
      }
    }

    if (parsed.eigenvalues.length > 0) {
      const analysis = analyzeBands(parsed.eigenvalues, parsed.nBands, fermiEnergy);
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
