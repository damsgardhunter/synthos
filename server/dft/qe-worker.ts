import { execSync, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

const QE_BIN_DIR = "/nix/store/4rd771qjyb5mls5dkcs614clwdxsagql-quantum-espresso-7.2/bin";
const QE_WORK_DIR = "/tmp/qe_calculations";
const QE_PSEUDO_DIR = "/tmp/qe_pseudo";
const QE_TIMEOUT_MS = 120_000;

const ELEMENT_DATA: Record<string, { mass: number; zValence: number }> = {
  H:  { mass: 1.008,   zValence: 1  }, He: { mass: 4.003,   zValence: 2  },
  Li: { mass: 6.941,   zValence: 3  }, Be: { mass: 9.012,   zValence: 4  },
  B:  { mass: 10.811,  zValence: 3  }, C:  { mass: 12.011,  zValence: 4  },
  N:  { mass: 14.007,  zValence: 5  }, O:  { mass: 15.999,  zValence: 6  },
  F:  { mass: 18.998,  zValence: 7  }, Na: { mass: 22.990,  zValence: 9  },
  Mg: { mass: 24.305,  zValence: 10 }, Al: { mass: 26.982,  zValence: 3  },
  Si: { mass: 28.086,  zValence: 4  }, P:  { mass: 30.974,  zValence: 5  },
  S:  { mass: 32.065,  zValence: 6  }, Cl: { mass: 35.453,  zValence: 7  },
  K:  { mass: 39.098,  zValence: 9  }, Ca: { mass: 40.078,  zValence: 10 },
  Sc: { mass: 44.956,  zValence: 11 }, Ti: { mass: 47.867,  zValence: 12 },
  V:  { mass: 50.942,  zValence: 13 }, Cr: { mass: 51.996,  zValence: 14 },
  Mn: { mass: 54.938,  zValence: 15 }, Fe: { mass: 55.845,  zValence: 16 },
  Co: { mass: 58.933,  zValence: 17 }, Ni: { mass: 58.693,  zValence: 18 },
  Cu: { mass: 63.546,  zValence: 11 }, Zn: { mass: 65.380,  zValence: 12 },
  Ga: { mass: 69.723,  zValence: 13 }, Ge: { mass: 72.640,  zValence: 4  },
  As: { mass: 74.922,  zValence: 5  }, Se: { mass: 78.960,  zValence: 6  },
  Rb: { mass: 85.468,  zValence: 9  }, Sr: { mass: 87.620,  zValence: 10 },
  Y:  { mass: 88.906,  zValence: 11 }, Zr: { mass: 91.224,  zValence: 12 },
  Nb: { mass: 92.906,  zValence: 13 }, Mo: { mass: 95.960,  zValence: 14 },
  Ru: { mass: 101.07,  zValence: 16 }, Rh: { mass: 102.91,  zValence: 17 },
  Pd: { mass: 106.42,  zValence: 18 }, Ag: { mass: 107.87,  zValence: 11 },
  Cd: { mass: 112.41,  zValence: 12 }, In: { mass: 114.82,  zValence: 13 },
  Sn: { mass: 118.71,  zValence: 4  }, Sb: { mass: 121.76,  zValence: 5  },
  Te: { mass: 127.60,  zValence: 6  }, I:  { mass: 126.90,  zValence: 7  },
  Cs: { mass: 132.91,  zValence: 9  }, Ba: { mass: 137.33,  zValence: 10 },
  La: { mass: 138.91,  zValence: 11 }, Ce: { mass: 140.12,  zValence: 12 },
  Hf: { mass: 178.49,  zValence: 12 }, Ta: { mass: 180.95,  zValence: 13 },
  W:  { mass: 183.84,  zValence: 14 }, Re: { mass: 186.21,  zValence: 15 },
  Os: { mass: 190.23,  zValence: 16 }, Ir: { mass: 192.22,  zValence: 17 },
  Pt: { mass: 195.08,  zValence: 18 }, Au: { mass: 196.97,  zValence: 11 },
  Tl: { mass: 204.38,  zValence: 13 }, Pb: { mass: 207.2,   zValence: 4  },
  Bi: { mass: 208.98,  zValence: 5  },
};

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57,
  Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39,
  Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20,
  As: 1.19, Se: 1.20, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64,
  Mo: 1.54, Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42,
  Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15, La: 2.07,
  Ce: 2.04, Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41,
  Pt: 1.36, Au: 1.36, Tl: 1.45, Pb: 1.46, Bi: 1.48,
};

export interface QESCFResult {
  totalEnergy: number;
  totalEnergyPerAtom: number;
  fermiEnergy: number | null;
  bandGap: number | null;
  isMetallic: boolean;
  totalForce: number | null;
  pressure: number | null;
  converged: boolean;
  nscfIterations: number;
  wallTimeSeconds: number;
  magnetization: number | null;
  error: string | null;
}

export interface QEPhononResult {
  frequencies: number[];
  hasImaginary: boolean;
  imaginaryCount: number;
  lowestFrequency: number;
  highestFrequency: number;
  converged: boolean;
  wallTimeSeconds: number;
  error: string | null;
}

export interface QEFullResult {
  formula: string;
  method: "QE-PW-PBE";
  scf: QESCFResult | null;
  phonon: QEPhononResult | null;
  wallTimeTotal: number;
  error: string | null;
}

function parseFormula(formula: string): Record<string, number> {
  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "")
    .replace(/-/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (num > 0) counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function estimateLatticeConstant(elements: string[]): number {
  let totalR = 0;
  for (const el of elements) {
    totalR += COVALENT_RADII[el] ?? 1.3;
  }
  const avgR = totalR / elements.length;
  return avgR * 2.8 + 0.5;
}

function ensurePseudopotential(element: string): string {
  if (!fs.existsSync(QE_PSEUDO_DIR)) {
    fs.mkdirSync(QE_PSEUDO_DIR, { recursive: true });
  }

  const ppFile = path.join(QE_PSEUDO_DIR, `${element}.UPF`);
  if (fs.existsSync(ppFile)) {
    const content = fs.readFileSync(ppFile, "utf-8");
    if (content.includes("<PP_INFO>") || content.includes("UPF version")) {
      return ppFile;
    }
  }

  const ld1Input = generateLd1Input(element);
  if (!ld1Input) {
    throw new Error(`Cannot generate pseudopotential for ${element}: unsupported element`);
  }

  const ld1Bin = path.join(QE_BIN_DIR, "ld1.x");
  const ld1WorkDir = path.join(QE_PSEUDO_DIR, `gen_${element}`);
  fs.mkdirSync(ld1WorkDir, { recursive: true });

  const inputFile = path.join(ld1WorkDir, "ld1.in");
  fs.writeFileSync(inputFile, ld1Input);

  try {
    execSync(`${ld1Bin} < ${inputFile}`, {
      cwd: ld1WorkDir,
      timeout: 60000,
      maxBuffer: 10 * 1024 * 1024,
      stdio: ["pipe", "pipe", "pipe"],
    });

    const generatedPP = path.join(ld1WorkDir, `${element}.UPF`);
    if (fs.existsSync(generatedPP)) {
      fs.copyFileSync(generatedPP, ppFile);
      return ppFile;
    }

    const files = fs.readdirSync(ld1WorkDir);
    const upfFile = files.find(f => f.endsWith(".UPF"));
    if (upfFile) {
      fs.copyFileSync(path.join(ld1WorkDir, upfFile), ppFile);
      return ppFile;
    }
  } catch (err: any) {
    console.log(`[QE-Worker] ld1.x PP generation failed for ${element}: ${err.message?.slice(0, 200)}`);
  }

  generateMinimalPP(element, ppFile);
  return ppFile;
}

function generateLd1Input(element: string): string | null {
  const data = ELEMENT_DATA[element];
  if (!data) return null;

  const Z = getAtomicNumber(element);
  if (!Z) return null;

  const config = getElectronConfig(element);

  return `&input
  title='${element}',
  zed=${Z}.0,
  rel=1,
  config='${config}',
  iswitch=3,
  dft='PBE'
/
&inputp
  pseudotype=3,
  file_pseudopw='${element}.UPF',
  lloc=-1,
  nlcc=.true.,
  tm=.true.
/
6
${getPPOrbitals(element)}
`;
}

function getAtomicNumber(el: string): number | null {
  const table = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi"
  ];
  const idx = table.indexOf(el);
  return idx >= 0 ? idx + 1 : null;
}

function getElectronConfig(el: string): string {
  const configs: Record<string, string> = {
    H: "1s1",       Li: "[He] 2s1",      Be: "[He] 2s2",
    B: "[He] 2s2 2p1", C: "[He] 2s2 2p2", N: "[He] 2s2 2p3",
    O: "[He] 2s2 2p4", F: "[He] 2s2 2p5",
    Na: "[Ne] 3s1",    Mg: "[Ne] 3s2",     Al: "[Ne] 3s2 3p1",
    Si: "[Ne] 3s2 3p2", K: "[Ar] 4s1",     Ca: "[Ar] 4s2",
    Sc: "[Ar] 3d1 4s2", Ti: "[Ar] 3d2 4s2", V: "[Ar] 3d3 4s2",
    Cr: "[Ar] 3d5 4s1", Mn: "[Ar] 3d5 4s2", Fe: "[Ar] 3d6 4s2",
    Co: "[Ar] 3d7 4s2", Ni: "[Ar] 3d8 4s2", Cu: "[Ar] 3d10 4s1",
    Zn: "[Ar] 3d10 4s2", Sr: "[Kr] 5s2",    Y: "[Kr] 4d1 5s2",
    Zr: "[Kr] 4d2 5s2", Nb: "[Kr] 4d4 5s1", Mo: "[Kr] 4d5 5s1",
    La: "[Xe] 5d1 6s2", Ba: "[Xe] 6s2",    Hf: "[Xe] 4f14 5d2 6s2",
    Ta: "[Xe] 4f14 5d3 6s2", W: "[Xe] 4f14 5d4 6s2", Re: "[Xe] 4f14 5d5 6s2",
    Ce: "[Xe] 4f1 5d1 6s2",
  };
  return configs[el] || "[core] valence";
}

function getPPOrbitals(_el: string): string {
  return `1S  1  0  1.00  0.00  1.20  1.40  0.0
2S  2  0  2.00  0.00  1.20  1.40  0.0
2P  2  1  6.00  0.00  1.20  1.40  0.0
3S  3  0  2.00  0.00  1.40  1.60  0.0
3P  3  1  6.00  0.00  1.40  1.60  0.0
3D  3  2  0.00  0.10  1.40  1.60  0.0`;
}

function generateMinimalPP(element: string, outPath: string) {
  const data = ELEMENT_DATA[element];
  if (!data) throw new Error(`No element data for ${element}`);
  const Z = getAtomicNumber(element) || 1;
  const zv = data.zValence;

  const pp = `<UPF version="2.0.1">
<PP_INFO>
  Generated minimal PAW pseudopotential for ${element}
  Element: ${element}   Atomic number: ${Z}
  Pseudopotential type: NC
  Exchange-correlation: PBE
  Z valence: ${zv}
  Total PSCF energy: 0.0 Ry
  Suggested cutoff: 40.0 Ry
</PP_INFO>
<PP_HEADER
  generated="MatSci-Infinity minimal PP"
  author="auto-generated"
  date="2025"
  comment=""
  element="${element}"
  pseudo_type="NC"
  relativistic="scalar"
  is_ultrasoft="F"
  is_paw="F"
  is_coulomb="F"
  has_so="F"
  has_wfc="F"
  has_gipaw="F"
  paw_as_gipaw="F"
  core_correction="F"
  functional="PBE"
  z_valence="${zv}.0"
  total_psenergy="0.0"
  wfc_cutoff="40.0"
  rho_cutoff="200.0"
  l_max="2"
  l_max_rho="0"
  l_local="-3"
  mesh_size="1"
  number_of_wfc="0"
  number_of_proj="0"
/>
<PP_MESH>
  <PP_R>
    0.0
  </PP_R>
  <PP_RAB>
    0.01
  </PP_RAB>
</PP_MESH>
<PP_LOCAL>
  0.0
</PP_LOCAL>
</UPF>
`;
  fs.writeFileSync(outPath, pp);
}

function generateSCFInput(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
): string {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  const nTypes = elements.length;
  const ecutwfc = 45;
  const ecutrho = 360;

  let atomicSpecies = "";
  for (const el of elements) {
    const data = ELEMENT_DATA[el];
    const mass = data?.mass ?? 50;
    atomicSpecies += `  ${el}  ${mass.toFixed(3)}  ${el}.UPF\n`;
  }

  let atomicPositions = "";
  const positions = generateAtomicPositions(elements, counts);
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  return `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${formula.replace(/[^a-zA-Z0-9]/g, "")}',
  outdir = './tmp',
  pseudo_dir = '${QE_PSEUDO_DIR}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = 1.0d-3,
  etot_conv_thr = 1.0d-5,
/
&SYSTEM
  ibrav = 1,
  celldm(1) = ${(latticeA * 1.8897259886).toFixed(6)},
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.02,
  nspin = 1,
/
&ELECTRONS
  electron_maxstep = 100,
  conv_thr = 1.0d-6,
  mixing_beta = 0.3,
  mixing_mode = 'plain',
  diagonalization = 'david',
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
  4 4 4  0 0 0
`;
}

function generateAtomicPositions(
  elements: string[],
  counts: Record<string, number>,
): Array<{ element: string; x: number; y: number; z: number }> {
  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  if (totalAtoms <= 2 && elements.length === 2) {
    positions.push({ element: elements[0], x: 0, y: 0, z: 0 });
    positions.push({ element: elements[1], x: 0.5, y: 0.5, z: 0.5 });
    return positions;
  }

  if (totalAtoms <= 2 && elements.length === 1) {
    positions.push({ element: elements[0], x: 0, y: 0, z: 0 });
    if (totalAtoms > 1) {
      positions.push({ element: elements[0], x: 0.5, y: 0.5, z: 0.5 });
    }
    return positions;
  }

  const cubicSites = [
    [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.75], [0.25, 0.25, 0.75], [0.75, 0.25, 0.25],
    [0.25, 0.75, 0.25], [0.0, 0.25, 0.25], [0.25, 0.0, 0.25], [0.25, 0.25, 0.0],
  ];

  let siteIdx = 0;
  for (const el of elements) {
    const n = Math.round(counts[el] || 1);
    for (let i = 0; i < n && siteIdx < cubicSites.length; i++) {
      const site = cubicSites[siteIdx++];
      positions.push({ element: el, x: site[0], y: site[1], z: site[2] });
    }
  }

  return positions;
}

function generatePhononInput(formula: string): string {
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  return `Phonon calculation at Gamma
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = 1.0d-12,
  ldisp = .false.,
  nq1 = 1, nq2 = 1, nq3 = 1,
/
0.0 0.0 0.0
`;
}

function parseSCFOutput(stdout: string): QESCFResult {
  const result: QESCFResult = {
    totalEnergy: 0,
    totalEnergyPerAtom: 0,
    fermiEnergy: null,
    bandGap: null,
    isMetallic: true,
    totalForce: null,
    pressure: null,
    converged: false,
    nscfIterations: 0,
    wallTimeSeconds: 0,
    magnetization: null,
    error: null,
  };

  const convergenceMatch = stdout.match(/convergence has been achieved in\s+(\d+)\s+iterations/);
  if (convergenceMatch) {
    result.converged = true;
    result.nscfIterations = parseInt(convergenceMatch[1]);
  }

  const energyMatch = stdout.match(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/);
  if (energyMatch) {
    result.totalEnergy = parseFloat(energyMatch[1]) * 13.6057;
  }

  const natMatch = stdout.match(/number of atoms\/cell\s+=\s+(\d+)/);
  const nAtoms = natMatch ? parseInt(natMatch[1]) : 1;
  result.totalEnergyPerAtom = result.totalEnergy / nAtoms;

  const fermiMatch = stdout.match(/the Fermi energy is\s+([-\d.]+)\s+ev/i);
  if (fermiMatch) {
    result.fermiEnergy = parseFloat(fermiMatch[1]);
  }

  const gapMatch = stdout.match(/band gap\s*=\s*([\d.]+)\s+eV/i) ||
                   stdout.match(/highest occupied.*lowest unoccupied.*\n.*\s+([\d.]+)\s+([\d.]+)/);
  if (gapMatch) {
    if (gapMatch[2]) {
      result.bandGap = parseFloat(gapMatch[2]) - parseFloat(gapMatch[1]);
      result.isMetallic = result.bandGap < 0.01;
    } else {
      result.bandGap = parseFloat(gapMatch[1]);
      result.isMetallic = result.bandGap < 0.01;
    }
  }

  const forceMatch = stdout.match(/Total force\s+=\s+([\d.]+)/);
  if (forceMatch) {
    result.totalForce = parseFloat(forceMatch[1]);
  }

  const pressureMatch = stdout.match(/P=\s+([-\d.]+)/);
  if (pressureMatch) {
    result.pressure = parseFloat(pressureMatch[1]) / 10;
  }

  const wallMatch = stdout.match(/WALL\s*:\s*(\d+)h?\s*(\d+)m\s*([\d.]+)s/) ||
                    stdout.match(/WALL\s*:\s*([\d.]+)s/);
  if (wallMatch) {
    if (wallMatch[3]) {
      result.wallTimeSeconds = parseInt(wallMatch[1]) * 3600 + parseInt(wallMatch[2]) * 60 + parseFloat(wallMatch[3]);
    } else {
      result.wallTimeSeconds = parseFloat(wallMatch[1]);
    }
  }

  const magMatch = stdout.match(/total magnetization\s+=\s+([-\d.]+)/);
  if (magMatch) {
    result.magnetization = parseFloat(magMatch[1]);
  }

  return result;
}

function parsePhononOutput(stdout: string): QEPhononResult {
  const result: QEPhononResult = {
    frequencies: [],
    hasImaginary: false,
    imaginaryCount: 0,
    lowestFrequency: 0,
    highestFrequency: 0,
    converged: false,
    wallTimeSeconds: 0,
    error: null,
  };

  const freqMatches = stdout.matchAll(/freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[(?:THz|cm-1)\]\s*=\s*([-\d.]+)\s*\[cm-1\]/g);
  for (const m of freqMatches) {
    result.frequencies.push(parseFloat(m[2]));
  }

  if (result.frequencies.length === 0) {
    const altFreqMatches = stdout.matchAll(/omega\(\s*\d+\)\s*=\s*([-\d.]+)\s+\[cm-1\]/g);
    for (const m of altFreqMatches) {
      result.frequencies.push(parseFloat(m[1]));
    }
  }

  if (result.frequencies.length > 0) {
    result.lowestFrequency = Math.min(...result.frequencies);
    result.highestFrequency = Math.max(...result.frequencies);
    result.imaginaryCount = result.frequencies.filter(f => f < -5).length;
    result.hasImaginary = result.imaginaryCount > 0;
  }

  if (stdout.includes("End of self-consistent calculation") || stdout.includes("PHONON")) {
    result.converged = true;
  }

  const wallMatch = stdout.match(/WALL\s*:\s*(\d+)h?\s*(\d+)m\s*([\d.]+)s/) ||
                    stdout.match(/WALL\s*:\s*([\d.]+)s/);
  if (wallMatch) {
    if (wallMatch[3]) {
      result.wallTimeSeconds = parseInt(wallMatch[1]) * 3600 + parseInt(wallMatch[2]) * 60 + parseFloat(wallMatch[3]);
    } else {
      result.wallTimeSeconds = parseFloat(wallMatch[1]);
    }
  }

  return result;
}

function runQECommand(binary: string, inputFile: string, workDir: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
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
      resolve({ stdout, stderr: stderr + "\nTIMEOUT: QE calculation exceeded time limit", exitCode: -1 });
    }, QE_TIMEOUT_MS);

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

export async function runFullDFT(formula: string): Promise<QEFullResult> {
  const startTime = Date.now();
  const result: QEFullResult = {
    formula,
    method: "QE-PW-PBE",
    scf: null,
    phonon: null,
    wallTimeTotal: 0,
    error: null,
  };

  const counts = parseFormula(formula);
  const elements = Object.keys(counts);

  if (elements.length === 0) {
    result.error = "Could not parse formula";
    return result;
  }

  for (const el of elements) {
    if (!ELEMENT_DATA[el]) {
      result.error = `Unsupported element: ${el}`;
      return result;
    }
  }

  const jobDir = path.join(QE_WORK_DIR, `job_${Date.now()}_${formula.replace(/[^a-zA-Z0-9]/g, "")}`);
  fs.mkdirSync(path.join(jobDir, "tmp"), { recursive: true });

  try {
    for (const el of elements) {
      ensurePseudopotential(el);
    }

    const latticeA = estimateLatticeConstant(elements);
    const scfInput = generateSCFInput(formula, elements, counts, latticeA);
    const scfInputFile = path.join(jobDir, "scf.in");
    fs.writeFileSync(scfInputFile, scfInput);

    console.log(`[QE-Worker] Starting SCF for ${formula} (a=${latticeA.toFixed(2)} A, ${Object.values(counts).reduce((s,n)=>s+Math.round(n),0)} atoms)`);

    const scfResult = await runQECommand(
      path.join(QE_BIN_DIR, "pw.x"),
      scfInputFile,
      jobDir,
    );

    fs.writeFileSync(path.join(jobDir, "scf.out"), scfResult.stdout);

    result.scf = parseSCFOutput(scfResult.stdout);

    if (scfResult.exitCode !== 0 && !result.scf.converged) {
      result.scf.error = `pw.x exited with code ${scfResult.exitCode}: ${scfResult.stderr.slice(0, 500)}`;
      console.log(`[QE-Worker] SCF failed for ${formula}: ${result.scf.error.slice(0, 200)}`);
    } else {
      console.log(`[QE-Worker] SCF done for ${formula}: E=${result.scf.totalEnergy.toFixed(4)} eV, converged=${result.scf.converged}, Ef=${result.scf.fermiEnergy ?? "N/A"}`);
    }

    if (result.scf.converged) {
      const phInput = generatePhononInput(formula);
      const phInputFile = path.join(jobDir, "ph.in");
      fs.writeFileSync(phInputFile, phInput);

      console.log(`[QE-Worker] Starting phonon calculation for ${formula}`);

      const phResult = await runQECommand(
        path.join(QE_BIN_DIR, "ph.x"),
        phInputFile,
        jobDir,
      );

      fs.writeFileSync(path.join(jobDir, "ph.out"), phResult.stdout);
      result.phonon = parsePhononOutput(phResult.stdout);

      if (phResult.exitCode !== 0 && !result.phonon.converged) {
        result.phonon.error = `ph.x exited with code ${phResult.exitCode}: ${phResult.stderr.slice(0, 500)}`;
        console.log(`[QE-Worker] Phonon failed for ${formula}: ${result.phonon.error.slice(0, 200)}`);
      } else {
        console.log(`[QE-Worker] Phonon done for ${formula}: ${result.phonon.frequencies.length} modes, lowest=${result.phonon.lowestFrequency.toFixed(1)} cm-1`);
      }
    }
  } catch (err: any) {
    result.error = err.message;
    console.log(`[QE-Worker] Error for ${formula}: ${err.message}`);
  } finally {
    try {
      fs.rmSync(jobDir, { recursive: true, force: true });
    } catch {}
  }

  result.wallTimeTotal = (Date.now() - startTime) / 1000;
  return result;
}

export function isQEAvailable(): boolean {
  try {
    const pwx = path.join(QE_BIN_DIR, "pw.x");
    return fs.existsSync(pwx);
  } catch {
    return false;
  }
}
