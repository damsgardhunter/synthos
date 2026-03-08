import { execSync, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { selectPrototype } from "../learning/crystal-prototypes";
import { isTransitionMetal, isRareEarth } from "../learning/elemental-data";

const QE_BIN_DIR = "/nix/store/4rd771qjyb5mls5dkcs614clwdxsagql-quantum-espresso-7.2/bin";
const QE_WORK_DIR = "/tmp/qe_calculations";
const QE_PSEUDO_DIR = "/tmp/qe_pseudo";
const QE_TIMEOUT_MS = 300_000;

const PROJECT_ROOT = path.resolve(process.cwd());
const PP_SOURCE_DIR = path.join(PROJECT_ROOT, "server/dft/pseudo");
const XTB_BIN = path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb");
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");

const failedFormulaTracker = new Map<string, { count: number; lastAttempt: number }>();
const MAX_FORMULA_FAILURES = 3;
const FAILURE_COOLDOWN_MS = 3600_000;

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
  Hg: { mass: 200.59,  zValence: 12 },
  Tl: { mass: 204.38,  zValence: 13 }, Pb: { mass: 207.2,   zValence: 4  },
  Bi: { mass: 208.98,  zValence: 5  },
  Br: { mass: 79.904,  zValence: 7  },
  Tc: { mass: 98.0,    zValence: 7  },
  Pr: { mass: 140.91,  zValence: 13 },
  Nd: { mass: 144.24,  zValence: 14 },
  Sm: { mass: 150.36,  zValence: 16 },
  Eu: { mass: 151.96,  zValence: 17 },
  Gd: { mass: 157.25,  zValence: 18 },
  Tb: { mass: 158.93,  zValence: 19 },
  Dy: { mass: 162.50,  zValence: 20 },
  Ho: { mass: 164.93,  zValence: 21 },
  Er: { mass: 167.26,  zValence: 22 },
  Tm: { mass: 168.93,  zValence: 23 },
  Yb: { mass: 173.04,  zValence: 24 },
  Lu: { mass: 174.97,  zValence: 25 },
  Th: { mass: 232.04,  zValence: 12 },
  U:  { mass: 238.03,  zValence: 14 },
  Pa: { mass: 231.04,  zValence: 13 },
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
  retryCount?: number;
  xtbPreRelaxed?: boolean;
  ppValidated?: boolean;
  rejectionReason?: string;
  failureStage?: string;
  prototypeUsed?: string;
  kPoints?: string;
  highPressure?: boolean;
  estimatedPressureGPa?: number;
}

const structureHashCache = new Set<string>();

const stageFailureCounts: Record<string, number> = {
  formula_filter: 0,
  pp_validation: 0,
  geometry: 0,
  duplicate: 0,
  xtb_prefilter: 0,
  scf: 0,
  phonon: 0,
};

export function getStageFailureCounts(): Record<string, number> {
  return { ...stageFailureCounts };
}

function computeStructureHash(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
): string {
  const sorted = [...positions]
    .sort((a, b) => a.element.localeCompare(b.element) || a.x - b.x || a.y - b.y || a.z - b.z)
    .map(p => `${p.element}:${p.x.toFixed(4)},${p.y.toFixed(4)},${p.z.toFixed(4)}`)
    .join("|");
  const key = `${latticeA.toFixed(3)}|${sorted}`;
  return crypto.createHash("md5").update(key).digest("hex");
}

function runXTBStabilityCheck(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  workDir: string,
): { stable: boolean; ePerAtom: number } | null {
  try {
    let xyz = `${positions.length}\nstability check\n`;
    for (const p of positions) {
      xyz += `${p.element}  ${(p.x * latticeA).toFixed(6)}  ${(p.y * latticeA).toFixed(6)}  ${(p.z * latticeA).toFixed(6)}\n`;
    }
    const xyzFile = path.join(workDir, "stability.xyz");
    fs.writeFileSync(xyzFile, xyz);

    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "1G",
    };

    const out = execSync(`${XTB_BIN} ${xyzFile} --gfn 2 --sp 2>&1 || true`, {
      cwd: workDir,
      timeout: 15000,
      maxBuffer: 5 * 1024 * 1024,
      env,
    }).toString();

    const eMatch = out.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
    if (!eMatch) return null;
    const totalEHa = parseFloat(eMatch[1]);
    const ePerAtomHa = totalEHa / positions.length;
    const ePerAtomEv = ePerAtomHa * 27.2114;
    const isStable = ePerAtomEv < -1.0;
    return { stable: isStable, ePerAtom: ePerAtomEv };
  } catch {
    return null;
  }
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

const ATOMIC_VOLUMES: Record<string, number> = {
  H: 5, He: 6, Li: 13, Be: 8, B: 7, C: 12, N: 13, O: 14, F: 15, Ne: 7,
  Na: 24, Mg: 14, Al: 17, Si: 12, P: 17, S: 16, Cl: 22, Ar: 24,
  K: 46, Ca: 26, Sc: 25, Ti: 18, V: 14, Cr: 12, Mn: 12,
  Fe: 12, Co: 11, Ni: 11, Cu: 12, Zn: 14, Ga: 12, Ge: 14,
  As: 13, Se: 17, Br: 24, Kr: 28, Rb: 56, Sr: 34, Y: 33, Zr: 23,
  Nb: 18, Mo: 16, Tc: 14, Ru: 13, Rh: 14, Pd: 15, Ag: 17, Cd: 22,
  In: 26, Sn: 27, Sb: 29, Te: 34, I: 26, Xe: 36,
  Cs: 71, Ba: 39, La: 37, Ce: 35, Pr: 35, Nd: 34,
  Pm: 33, Sm: 33, Eu: 36, Gd: 33, Tb: 32, Dy: 32,
  Ho: 32, Er: 31, Tm: 31, Yb: 35, Lu: 30,
  Hf: 22, Ta: 18, W: 16, Re: 15, Os: 14, Ir: 14, Pt: 15, Au: 17,
  Hg: 23, Tl: 29, Pb: 30, Bi: 35, Po: 34,
  Th: 32, Pa: 25, U: 21, Np: 20, Pu: 20,
};

function estimateLatticeConstant(elements: string[], counts?: Record<string, number>): number {
  let totalVolume = 0;
  if (counts) {
    for (const el of Object.keys(counts)) {
      const n = Math.round(counts[el] || 1);
      const vol = ATOMIC_VOLUMES[el] ?? 15;
      totalVolume += n * vol;
    }
  } else {
    for (const el of elements) {
      const vol = ATOMIC_VOLUMES[el] ?? 15;
      totalVolume += vol;
    }
  }
  const packingFactor = 0.65;
  const cellVolume = totalVolume / packingFactor;
  const a = Math.cbrt(cellVolume);
  return Math.max(a, 3.5);
}

function validatePseudopotential(filePath: string): boolean {
  try {
    if (!fs.existsSync(filePath)) return false;
    const stats = fs.statSync(filePath);
    if (stats.size < 10000) return false;
    const head = Buffer.alloc(4096);
    const fd = fs.openSync(filePath, "r");
    fs.readSync(fd, head, 0, 4096, 0);
    fs.closeSync(fd);
    const headStr = head.toString("utf-8");
    if (headStr.includes("<!DOCTYPE") || headStr.includes("<html")) return false;
    if (!headStr.includes("<UPF") && !headStr.includes("<PP_HEADER")) return false;
    const tail = Buffer.alloc(256);
    const fd2 = fs.openSync(filePath, "r");
    const readPos = Math.max(0, stats.size - 256);
    fs.readSync(fd2, tail, 0, 256, readPos);
    fs.closeSync(fd2);
    if (!tail.toString("utf-8").includes("</UPF>")) return false;
    return true;
  } catch {
    return false;
  }
}

function cleanupPseudoDir(): void {
  if (!fs.existsSync(QE_PSEUDO_DIR)) return;
  const entries = fs.readdirSync(QE_PSEUDO_DIR);
  for (const entry of entries) {
    const fullPath = path.join(QE_PSEUDO_DIR, entry);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      try { fs.rmSync(fullPath, { recursive: true, force: true }); } catch {}
      console.log(`[QE-Worker] Removed stale directory: ${entry}`);
      continue;
    }
    if (entry.endsWith(".UPF") && !validatePseudopotential(fullPath)) {
      try { fs.unlinkSync(fullPath); } catch {}
      console.log(`[QE-Worker] Removed invalid PP from cache: ${entry} (${stat.size} bytes)`);
    }
  }
}

cleanupPseudoDir();

function ensurePseudopotential(element: string): string {
  if (!fs.existsSync(QE_PSEUDO_DIR)) {
    fs.mkdirSync(QE_PSEUDO_DIR, { recursive: true });
  }

  const ppFile = path.join(QE_PSEUDO_DIR, `${element}.UPF`);
  if (fs.existsSync(ppFile) && validatePseudopotential(ppFile)) {
    return ppFile;
  }

  if (fs.existsSync(ppFile)) {
    console.log(`[QE-Worker] Invalid PP for ${element} (${fs.statSync(ppFile).size} bytes), removing`);
    try { fs.unlinkSync(ppFile); } catch {}
  }

  const sourceFile = path.join(PP_SOURCE_DIR, `${element}.UPF`);
  if (fs.existsSync(sourceFile) && validatePseudopotential(sourceFile)) {
    fs.copyFileSync(sourceFile, ppFile);
    console.log(`[QE-Worker] Copied valid PP for ${element} from repo (${fs.statSync(ppFile).size} bytes)`);
    return ppFile;
  }

  throw new Error(`No valid pseudopotential for ${element} — need a verified UPF file in ${PP_SOURCE_DIR}/${element}.UPF`);
}

function autoKPoints(latticeA: number): string {
  const k = Math.max(1, Math.ceil(30 / latticeA));
  return `  ${k} ${k} ${k}  0 0 0`;
}

function generateSCFInput(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
): string {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  const nTypes = elements.length;
  const ELEMENT_CUTOFFS: Record<string, number> = {
    O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
  };
  const ecutwfc = elements.reduce((max, el) => Math.max(max, ELEMENT_CUTOFFS[el] ?? 45), 45);
  const ecutrho = ecutwfc * 8;

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
${autoKPoints(latticeA)}
`;
}

function generateAtomicPositions(
  elements: string[],
  counts: Record<string, number>,
  formula?: string,
): Array<{ element: string; x: number; y: number; z: number }> {
  if (formula) {
    try {
      const proto = selectPrototype(formula);
      if (proto) {
        const { template, siteMap } = proto;
        const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
        for (const site of template.sites) {
          const element = siteMap[site.label];
          if (element) {
            positions.push({ element, x: site.x, y: site.y, z: site.z });
          }
        }
        if (positions.length > 0) {
          console.log(`[QE-Worker] Using ${template.name} prototype for ${formula} (${positions.length} atoms)`);
          return positions;
        }
      }
    } catch {}
  }

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

  const hCount = Math.round(counts["H"] || 0);
  const metalElements = elements.filter(e => e !== "H");
  const metalCount = metalElements.reduce((s, e) => s + Math.round(counts[e] || 0), 0);

  if (hCount > 0 && metalCount > 0 && hCount / metalCount >= 4) {
    const hPerMetal = Math.round(hCount / metalCount);
    const cagePositions = generateHydrideCagePositions(metalElements, counts, hPerMetal, totalAtoms);
    if (cagePositions.length === totalAtoms && cagePositions.length <= 16) {
      console.log(`[QE-Worker] Using hydride cage motif for ${formula} (H/metal=${hPerMetal}, ${cagePositions.length} atoms)`);
      return cagePositions;
    }
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

function generateHydrideCagePositions(
  metalElements: string[],
  counts: Record<string, number>,
  hPerMetal: number,
  totalAtoms: number,
): Array<{ element: string; x: number; y: number; z: number }> {
  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

  const octahedralH = [
    { x: 0.5, y: 0.0, z: 0.0 }, { x: 0.0, y: 0.5, z: 0.0 }, { x: 0.0, y: 0.0, z: 0.5 },
    { x: 0.5, y: 0.5, z: 0.5 }, { x: 0.0, y: 0.5, z: 0.5 }, { x: 0.5, y: 0.0, z: 0.5 },
  ];

  const cubeVertexH = [
    { x: 0.25, y: 0.25, z: 0.25 }, { x: 0.75, y: 0.25, z: 0.25 },
    { x: 0.25, y: 0.75, z: 0.25 }, { x: 0.25, y: 0.25, z: 0.75 },
    { x: 0.75, y: 0.75, z: 0.25 }, { x: 0.75, y: 0.25, z: 0.75 },
    { x: 0.25, y: 0.75, z: 0.75 }, { x: 0.75, y: 0.75, z: 0.75 },
  ];

  const clathrateH10 = [
    { x: 0.25, y: 0.25, z: 0.0 }, { x: 0.75, y: 0.25, z: 0.0 },
    { x: 0.25, y: 0.75, z: 0.0 }, { x: 0.75, y: 0.75, z: 0.0 },
    { x: 0.0, y: 0.25, z: 0.25 }, { x: 0.0, y: 0.75, z: 0.25 },
    { x: 0.25, y: 0.0, z: 0.25 }, { x: 0.75, y: 0.0, z: 0.25 },
    { x: 0.5, y: 0.25, z: 0.5 }, { x: 0.5, y: 0.75, z: 0.5 },
  ];

  const metalCount = metalElements.reduce((s, e) => s + Math.round(counts[e] || 0), 0);
  if (metalCount === 1) {
    const metal = metalElements[0];
    positions.push({ element: metal, x: 0.0, y: 0.0, z: 0.0 });

    let hSites: Array<{ x: number; y: number; z: number }>;
    if (hPerMetal <= 6) {
      hSites = octahedralH.slice(0, hPerMetal);
    } else if (hPerMetal <= 8) {
      hSites = cubeVertexH.slice(0, hPerMetal);
    } else if (hPerMetal === 9) {
      hSites = [...cubeVertexH, { x: 0.5, y: 0.5, z: 0.0 }];
    } else if (hPerMetal <= 10) {
      hSites = clathrateH10.slice(0, hPerMetal);
    } else {
      const extraH = [
        { x: 0.0, y: 0.5, z: 0.75 },
        { x: 0.5, y: 0.0, z: 0.75 },
      ];
      hSites = [...clathrateH10, ...extraH.slice(0, Math.min(hPerMetal - 10, 2))];
    }
    for (const h of hSites) {
      positions.push({ element: "H", ...h });
    }
  } else {
    let placed = 0;
    const metalSites = [
      { x: 0.0, y: 0.0, z: 0.0 },
      { x: 0.5, y: 0.5, z: 0.5 },
    ];
    for (const metal of metalElements) {
      const n = Math.round(counts[metal] || 1);
      for (let i = 0; i < n && placed < metalSites.length; i++) {
        positions.push({ element: metal, ...metalSites[placed++] });
      }
    }
    const hTotal = Math.round(counts["H"] || 0);
    const allH = [...octahedralH, ...cubeVertexH];
    for (let i = 0; i < hTotal && i < allH.length; i++) {
      positions.push({ element: "H", ...allH[i] });
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

function validateGeometry(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
): { valid: boolean; reason: string } {
  if (positions.length === 0) return { valid: false, reason: "No atomic positions" };
  if (positions.length > 16) return { valid: false, reason: `Too many atoms (${positions.length}), max 16 for available resources` };

  if (latticeA < 3) return { valid: false, reason: `Lattice constant too small: ${latticeA.toFixed(2)} A (min 3)` };
  if (latticeA > 40) return { valid: false, reason: `Lattice constant too large: ${latticeA.toFixed(2)} A (max 40)` };

  const totalAtoms = positions.length;
  const volumeAng3 = latticeA ** 3;
  const volumePerAtom = volumeAng3 / totalAtoms;
  const hasHydrogen = positions.some(p => p.element === "H");
  const minVolPerAtom = hasHydrogen ? 5.0 : 10.0;
  if (volumePerAtom < minVolPerAtom) return { valid: false, reason: `Volume per atom too small: ${volumePerAtom.toFixed(1)} A^3 (min ${minVolPerAtom})` };

  const COVALENT_R: Record<string, number> = {
    H: 0.31, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57,
    Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02,
    K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39,
    Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20,
    Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, La: 2.07, Ce: 2.04, Sr: 1.95, Ba: 2.15,
    As: 1.19, Se: 1.20, Br: 1.20, Kr: 1.16, Rb: 2.20,
    Tc: 1.47, Ta: 1.70, W: 1.62, Te: 1.38, Sn: 1.39,
    Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, I: 1.39,
    In: 1.42, Sb: 1.39, Cs: 2.44, Hf: 1.75, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36,
    Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Th: 2.06, U: 1.96, Pa: 2.00,
  };
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      let fdx = positions[i].x - positions[j].x;
      let fdy = positions[i].y - positions[j].y;
      let fdz = positions[i].z - positions[j].z;
      fdx -= Math.round(fdx);
      fdy -= Math.round(fdy);
      fdz -= Math.round(fdz);
      const dx = fdx * latticeA;
      const dy = fdy * latticeA;
      const dz = fdz * latticeA;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const r1 = COVALENT_R[positions[i].element] ?? 1.4;
      const r2 = COVALENT_R[positions[j].element] ?? 1.4;
      const isHPair = positions[i].element === "H" || positions[j].element === "H";
      const factor = isHPair ? 0.70 : 0.80;
      const minDist = Math.max((r1 + r2) * factor, isHPair ? 0.6 : 0.9);
      if (dist < minDist) {
        return { valid: false, reason: `Atoms ${positions[i].element}(${i}) and ${positions[j].element}(${j}) too close: ${dist.toFixed(2)} A (min ${minDist.toFixed(2)})` };
      }
    }
  }
  return { valid: true, reason: "OK" };
}

function validateFormulaForDFT(formula: string, counts: Record<string, number>): { valid: boolean; reason: string; highPressure?: boolean; estimatedPressureGPa?: number } {
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  if (elements.length > 5) {
    return { valid: false, reason: `Too many distinct elements (${elements.length}), max 5 for simple cubic DFT` };
  }
  if (totalAtoms > 16) {
    return { valid: false, reason: `Too many atoms (${totalAtoms}), max 16 for available resources` };
  }

  const ALKALINE_EARTH_SYMBOLS = new Set(["Ca", "Sr", "Ba", "Mg"]);
  const KNOWN_SUPERHYDRIDES = new Set(["LaH10", "YH6", "YH9", "CeH9", "CeH10", "ThH10", "CaH6", "ScH9", "BaH12", "LaBeH8"]);
  const hCount = counts["H"] || 0;
  if (hCount > 0 && totalAtoms > 2) {
    const hRatio = hCount / totalAtoms;
    const hasHydrideMetal = elements.some(el => el !== "H" && (
      isTransitionMetal(el) || isRareEarth(el) || ALKALINE_EARTH_SYMBOLS.has(el)
    ));

    if (hRatio > 0.95 && !hasHydrideMetal) {
      return { valid: false, reason: `Hydrogen ratio ${(hRatio * 100).toFixed(0)}% with no metal host — likely unphysical` };
    }

    const nonHAtoms = totalAtoms - hCount;
    const hPerMetal = nonHAtoms > 0 ? hCount / nonHAtoms : hCount;
    const isKnownSuperhydride = KNOWN_SUPERHYDRIDES.has(formula.replace(/\s+/g, ""));

    if (hCount > 0 && nonHAtoms > 0 && hPerMetal < 0.5 && hasHydrideMetal) {
      return { valid: false, reason: `Metal-rich hydride (H/metal=${hPerMetal.toFixed(2)}) — unphysical stoichiometry, hydrides should have H/metal >= 0.5` };
    }

    if (!isKnownSuperhydride && hPerMetal > 6) {
      return { valid: true, reason: `High H/metal ratio ${hPerMetal.toFixed(1)} — tagged as high-pressure candidate (>100 GPa required)`, highPressure: true, estimatedPressureGPa: Math.min(300, 50 + hPerMetal * 15) };
    }

    if (hRatio > 0.75 && !isKnownSuperhydride) {
      return { valid: true, reason: `Hydrogen fraction ${(hRatio * 100).toFixed(0)}% — tagged as high-pressure superhydride candidate`, highPressure: true, estimatedPressureGPa: Math.min(300, 100 + hRatio * 100) };
    }
  }

  for (const el of elements) {
    if (!ELEMENT_DATA[el]) {
      return { valid: false, reason: `Unsupported element: ${el}` };
    }
  }

  return { valid: true, reason: "OK" };
}

function tryXTBPreRelaxation(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  workDir: string,
): Array<{ element: string; x: number; y: number; z: number }> | null {
  try {
    if (!fs.existsSync(XTB_BIN)) return null;

    const xyzPath = path.join(workDir, "pre_relax.xyz");
    const nAtoms = positions.length;
    let xyzContent = `${nAtoms}\npre-relaxation\n`;
    for (const pos of positions) {
      xyzContent += `${pos.element}  ${(pos.x * latticeA).toFixed(6)}  ${(pos.y * latticeA).toFixed(6)}  ${(pos.z * latticeA).toFixed(6)}\n`;
    }
    fs.writeFileSync(xyzPath, xyzContent);

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    execSync(
      `${XTB_BIN} pre_relax.xyz --gfn 2 --opt crude 2>&1`,
      { cwd: workDir, timeout: 30000, env, maxBuffer: 5 * 1024 * 1024 }
    );

    const optPath = path.join(workDir, "xtbopt.xyz");
    if (!fs.existsSync(optPath)) return null;

    const optContent = fs.readFileSync(optPath, "utf-8");
    const lines = optContent.trim().split("\n");
    if (lines.length < 3) return null;

    const relaxed: Array<{ element: string; x: number; y: number; z: number }> = [];
    for (let i = 2; i < lines.length; i++) {
      const parts = lines[i].trim().split(/\s+/);
      if (parts.length >= 4) {
        relaxed.push({
          element: parts[0],
          x: parseFloat(parts[1]) / latticeA,
          y: parseFloat(parts[2]) / latticeA,
          z: parseFloat(parts[3]) / latticeA,
        });
      }
    }

    if (relaxed.length === positions.length) {
      console.log(`[QE-Worker] xTB pre-relaxation succeeded for ${positions.length} atoms`);
      return relaxed;
    }
    return null;
  } catch (err: any) {
    console.log(`[QE-Worker] xTB pre-relaxation failed: ${err.message?.slice(0, 100)}`);
    return null;
  }
}

export function isFormulaBlocked(formula: string): boolean {
  const tracker = failedFormulaTracker.get(formula);
  if (!tracker) return false;
  if (Date.now() - tracker.lastAttempt > FAILURE_COOLDOWN_MS) {
    failedFormulaTracker.delete(formula);
    return false;
  }
  return tracker.count >= MAX_FORMULA_FAILURES;
}

function recordFormulaFailure(formula: string) {
  const tracker = failedFormulaTracker.get(formula) || { count: 0, lastAttempt: 0 };
  tracker.count++;
  tracker.lastAttempt = Date.now();
  failedFormulaTracker.set(formula, tracker);
}

function generateSCFInputWithParams(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  params: { mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number },
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  const ELEMENT_CUTOFFS2: Record<string, number> = {
    O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
  };
  const baseEcutwfc = elements.reduce((max, el) => Math.max(max, ELEMENT_CUTOFFS2[el] ?? 45), 45);
  const ecutwfc = baseEcutwfc + (params.ecutwfcBoost ?? 0);
  const ecutrho = ecutwfc * 8;
  const smearing = params.smearing || "mv";
  const degauss = params.degauss || 0.02;

  let atomicSpecies = "";
  for (const el of elements) {
    const data = ELEMENT_DATA[el];
    const mass = data?.mass ?? 50;
    atomicSpecies += `  ${el}  ${mass.toFixed(3)}  ${el}.UPF\n`;
  }

  let atomicPositions = "";
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
  smearing = '${smearing}',
  degauss = ${degauss},
  nspin = 1,
/
&ELECTRONS
  electron_maxstep = ${params.maxSteps},
  conv_thr = 1.0d-6,
  mixing_beta = ${params.mixingBeta},
  mixing_mode = 'plain',
  diagonalization = '${params.diag}',
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${autoKPoints(latticeA)}
`;
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
    retryCount: 0,
    xtbPreRelaxed: false,
    ppValidated: false,
  };

  if (isFormulaBlocked(formula)) {
    result.error = `Formula ${formula} blocked after ${MAX_FORMULA_FAILURES} consecutive failures (cooldown ${FAILURE_COOLDOWN_MS / 60000} min)`;
    return result;
  }

  const counts = parseFormula(formula);
  const elements = Object.keys(counts);

  if (elements.length === 0) {
    result.error = "Could not parse formula";
    return result;
  }

  const formulaCheck = validateFormulaForDFT(formula, counts);
  if (!formulaCheck.valid) {
    result.error = `Pre-filter rejected: ${formulaCheck.reason}`;
    result.rejectionReason = formulaCheck.reason;
    result.failureStage = "formula_filter";
    stageFailureCounts.formula_filter++;
    console.log(`[QE-Worker] ${formula} rejected: ${formulaCheck.reason}`);
    return result;
  }

  if (formulaCheck.highPressure) {
    result.highPressure = true;
    result.estimatedPressureGPa = formulaCheck.estimatedPressureGPa;
    console.log(`[QE-Worker] ${formula} tagged as high-pressure candidate (~${formulaCheck.estimatedPressureGPa} GPa)`);
  }

  const jobDir = path.join(QE_WORK_DIR, `job_${Date.now()}_${formula.replace(/[^a-zA-Z0-9]/g, "")}`);
  fs.mkdirSync(path.join(jobDir, "tmp"), { recursive: true });

  try {
    for (const el of elements) {
      try {
        ensurePseudopotential(el);
      } catch (ppErr: any) {
        result.error = ppErr.message;
        result.ppValidated = false;
        result.rejectionReason = `PP: ${ppErr.message}`;
        result.failureStage = "pp_validation";
        stageFailureCounts.pp_validation++;
        console.log(`[QE-Worker] PP validation failed for ${formula}: ${ppErr.message}`);
        recordFormulaFailure(formula);
        return result;
      }
    }

    const latticeA = estimateLatticeConstant(elements, counts);
    let positions = generateAtomicPositions(elements, counts, formula);

    try {
      const proto = selectPrototype(formula);
      if (proto) result.prototypeUsed = proto.template.name;
    } catch {}

    const geomCheck = validateGeometry(positions, latticeA);
    if (!geomCheck.valid) {
      result.error = `Geometry rejected: ${geomCheck.reason}`;
      result.failureStage = "geometry";
      stageFailureCounts.geometry++;
      console.log(`[QE-Worker] ${formula} geometry invalid: ${geomCheck.reason}`);
      recordFormulaFailure(formula);
      return result;
    }

    const hash = computeStructureHash(positions, latticeA);
    if (structureHashCache.has(hash)) {
      result.error = `Duplicate structure (hash=${hash.slice(0, 8)})`;
      result.failureStage = "duplicate";
      result.rejectionReason = "Duplicate structure already computed";
      stageFailureCounts.duplicate++;
      console.log(`[QE-Worker] ${formula} skipped: duplicate structure`);
      return result;
    }
    structureHashCache.add(hash);

    result.ppValidated = true;

    const stabilityCheck = runXTBStabilityCheck(positions, latticeA, jobDir);
    if (stabilityCheck && !stabilityCheck.stable) {
      result.error = `xTB stability pre-filter: E/atom = ${stabilityCheck.ePerAtom.toFixed(3)} eV (must be < -1.0 eV/atom)`;
      result.failureStage = "xtb_prefilter";
      result.rejectionReason = `Unstable: E/atom=${stabilityCheck.ePerAtom.toFixed(3)} eV`;
      stageFailureCounts.xtb_prefilter++;
      console.log(`[QE-Worker] ${formula} unstable by xTB pre-filter: E/atom=${stabilityCheck.ePerAtom.toFixed(3)} eV (threshold: < -1.0 eV/atom)`);
      return result;
    }

    const relaxed = tryXTBPreRelaxation(positions, latticeA, jobDir);
    result.xtbPreRelaxed = !!relaxed;
    if (relaxed) {
      positions = relaxed;
      console.log(`[QE-Worker] Using xTB pre-relaxed geometry for ${formula}`);
    }

    result.kPoints = autoKPoints(latticeA).trim();

    const retryConfigs: Array<{ mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number }> = [
      { mixingBeta: 0.3, maxSteps: 100, diag: "david" },
      { mixingBeta: 0.2, maxSteps: 200, diag: "david", ecutwfcBoost: 10 },
      { mixingBeta: 0.1, maxSteps: 300, diag: "cg", ecutwfcBoost: 20 },
      { mixingBeta: 0.1, maxSteps: 300, diag: "cg", smearing: "gaussian", degauss: 0.02, ecutwfcBoost: 20 },
    ];

    let scfConverged = false;
    let retryCount = 0;

    for (let attempt = 0; attempt < retryConfigs.length && !scfConverged; attempt++) {
      const params = retryConfigs[attempt];
      const scfInput = generateSCFInputWithParams(formula, elements, counts, latticeA, positions, params);
      const scfInputFile = path.join(jobDir, `scf_attempt${attempt}.in`);
      fs.writeFileSync(scfInputFile, scfInput);

      if (attempt > 0) {
        try {
          fs.rmSync(path.join(jobDir, "tmp"), { recursive: true, force: true });
          fs.mkdirSync(path.join(jobDir, "tmp"), { recursive: true });
        } catch {}
      }

      const smearInfo = params.smearing ? `, smearing=${params.smearing}` : "";
      console.log(`[QE-Worker] SCF attempt ${attempt + 1}/${retryConfigs.length} for ${formula} (a=${latticeA.toFixed(2)} A, beta=${params.mixingBeta}, diag=${params.diag}, maxstep=${params.maxSteps}${smearInfo})`);

      const scfResult = await runQECommand(
        path.join(QE_BIN_DIR, "pw.x"),
        scfInputFile,
        jobDir,
      );

      fs.writeFileSync(path.join(jobDir, `scf_attempt${attempt}.out`), scfResult.stdout);
      result.scf = parseSCFOutput(scfResult.stdout);

      if (scfResult.exitCode !== 0 && !result.scf.converged) {
        const isPPError = scfResult.stderr.includes("read_upf") || scfResult.stderr.includes("readpp") || scfResult.stderr.includes("EOF marker");
        if (isPPError) {
          result.scf.error = `Pseudopotential read failure: ${scfResult.stderr.slice(0, 300)}`;
          console.log(`[QE-Worker] PP error for ${formula}, no retry will help — skipping`);
          recordFormulaFailure(formula);
          break;
        }
        result.scf.error = `pw.x exited with code ${scfResult.exitCode}: ${scfResult.stderr.slice(0, 500)}`;
        console.log(`[QE-Worker] SCF attempt ${attempt + 1} failed for ${formula}: ${result.scf.error.slice(0, 200)}`);
        retryCount = attempt + 1;
      } else if (result.scf.converged) {
        scfConverged = true;
        retryCount = attempt;
        console.log(`[QE-Worker] SCF converged for ${formula} on attempt ${attempt + 1}: E=${result.scf.totalEnergy.toFixed(4)} eV, Ef=${result.scf.fermiEnergy ?? "N/A"}`);
      } else {
        retryCount = attempt + 1;
        console.log(`[QE-Worker] SCF attempt ${attempt + 1} did not converge for ${formula}`);
      }
    }

    result.retryCount = retryCount;

    if (!scfConverged) {
      result.failureStage = "scf";
      stageFailureCounts.scf++;
      recordFormulaFailure(formula);
    }

    if (result.scf?.converged) {
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
        result.failureStage = "phonon";
        stageFailureCounts.phonon++;
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
