import { execSync, execFile } from "child_process";
import { promisify } from "util";
const execFileAsync = promisify(execFile);
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { IS_WINDOWS, binaryPath, getTempSubdir, toWslPath, killProcessGracefully, spawnQE } from "./platform-utils";
import { selectPrototype } from "../learning/crystal-prototypes";
import { matchPrototype } from "../learning/structure-predictor";
import { isTransitionMetal, isRareEarth, ELEMENTAL_DATA, getElementData } from "../learning/elemental-data";
import { generatePrototypeFreeStructure } from "../crystal/lattice-generator";
import { getAllDistributions, getElementSitePreference, type CrystalSystemDistribution } from "../ai/crystal-distribution-db";
import { computeDFTBandStructure, recordBandCalcOutcome, type DFTBandStructureResult } from "./band-structure-calculator";

// Resolve the QE binary directory lazily (on first DFT call).
// Running execSync WSL probes at module-load time would block the Node.js event loop
// for up to 10s during server startup — so we defer to first use instead.
// On Windows (WSL2): probe for conda-forge QE 7.x install first (~/miniforge3/bin),
// then fall back to apt install (/usr/bin).
// On Linux/production: use Nix store path or QE_BIN_DIR env var.
function resolveQEBinDir(): string {
  if (process.env.QE_BIN_DIR) return process.env.QE_BIN_DIR;
  if (!IS_WINDOWS) {
    // Try Nix store first, then fall back to system paths (apt/conda installs on GCP/Linux)
    const candidates = [
      "/nix/store/4rd771qjyb5mls5dkcs614clwdxsagql-quantum-espresso-7.2/bin",
      "/usr/bin",
      "/usr/local/bin",
    ];
    for (const dir of candidates) {
      if (fs.existsSync(path.join(dir, "pw.x"))) return dir;
    }
    return candidates[0]; // default — will fail clearly with ENOENT if none found
  }
  try {
    const home = execSync('wsl.exe -d Ubuntu -- bash -c "echo $HOME"',
      { encoding: "utf8", timeout: 5000 }).trim().replace(/\r/g, "");
    const condaDir = `${home}/miniforge3/bin`;
    const found = execSync(`wsl.exe -d Ubuntu -- bash -c "test -f '${condaDir}/pw.x' && echo yes || echo no"`,
      { encoding: "utf8", timeout: 5000 }).trim().replace(/\r/g, "");
    if (found === "yes") return condaDir;
  } catch { /* fall through */ }
  return "/usr/bin"; // apt quantum-espresso package fallback
}
let _qeBinDir: string | null = null;
// Lazy getter — first call runs the WSL probe (execSync), subsequent calls return cached value.
// This defers the 5-10s startup block to when DFT is first actually requested.
function getQEBinDir(): string {
  if (_qeBinDir !== null) return _qeBinDir;
  _qeBinDir = resolveQEBinDir();
  return _qeBinDir;
}
const QE_WORK_DIR = getTempSubdir("qe_calculations");
const QE_PSEUDO_DIR = getTempSubdir("qe_pseudo");
// When QE runs inside WSL on Windows, paths in the input file must use the /mnt/... form
const QE_PSEUDO_DIR_INPUT = IS_WINDOWS ? toWslPath(QE_PSEUDO_DIR) : QE_PSEUDO_DIR;
const QE_TIMEOUT_MS = 600_000;

const PROJECT_ROOT = path.resolve(process.cwd());
const PP_SOURCE_DIR = path.join(PROJECT_ROOT, "server/dft/pseudo");
const XTB_BIN = binaryPath(path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb"));
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");

function fracDistAngstrom(
  fdx: number, fdy: number, fdz: number,
  latticeA: number, cOverA: number = 1.0, bOverA: number = 1.0,
  gammaRad: number = Math.PI / 2,
): number {
  const a = latticeA;
  const b = latticeA * bOverA;
  const c = latticeA * cOverA;
  const cosG = Math.cos(gammaRad);
  const dx = fdx * a + fdy * b * cosG;
  const dy = fdy * b * Math.sin(gammaRad);
  const dz = fdz * c;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function estimateBulkModulus(elements: string[]): number {
  let totalB = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.bulkModulus != null && data.bulkModulus > 0) {
      totalB += data.bulkModulus;
      count++;
    }
  }
  return count > 0 ? totalB / count : 100;
}

function computePressureScale(pressureGpa: number, elements?: string[]): number {
  if (pressureGpa <= 0) return 1.0;
  const B0 = elements ? estimateBulkModulus(elements) : 100;
  const B0p = 4.0;
  const inner = 1 + B0p * (pressureGpa / B0);
  const eta = inner > 0 ? Math.pow(inner, -1 / B0p) : 0.5;
  return Math.max(0.8, Math.min(1.0, Math.pow(eta, 1 / 3)));
}

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
  convergenceQuality: "strict" | "loose" | "none";
  lastScfAccuracyRy: number | null;
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
  bandStructure: DFTBandStructureResult | null;
  wallTimeTotal: number;
  error: string | null;
  retryCount?: number;
  xtbPreRelaxed?: boolean;
  vcRelaxed?: boolean;
  relaxedLatticeA?: number;
  initialLatticeA?: number;
  initialPositions?: Array<{ element: string; x: number; y: number; z: number }>;
  ppValidated?: boolean;
  rejectionReason?: string;
  failureStage?: string;
  prototypeUsed?: string;
  kPoints?: string;
  highPressure?: boolean;
  estimatedPressureGPa?: number;
}

const HASH_CACHE_MAX = 2000;
const HASH_CACHE_TTL_MS = 30 * 60 * 1000;
const structureHashMap = new Map<string, number>();

function isStructureDuplicate(hash: string, formula: string): boolean {
  const key = `${formula}::${hash}`;
  const now = Date.now();
  if (structureHashMap.has(key)) {
    structureHashMap.delete(key);
    structureHashMap.set(key, now);
    return true;
  }
  if (structureHashMap.size >= HASH_CACHE_MAX) {
    const cutoff = now - HASH_CACHE_TTL_MS;
    let purged = false;
    for (const [k, ts] of structureHashMap) {
      if (ts < cutoff) {
        structureHashMap.delete(k);
        purged = true;
      }
    }
    if (!purged || structureHashMap.size >= HASH_CACHE_MAX) {
      const iter = structureHashMap.keys();
      const evictCount = Math.max(1, Math.floor(HASH_CACHE_MAX * 0.1));
      for (let i = 0; i < evictCount; i++) {
        const oldest = iter.next();
        if (oldest.done) break;
        structureHashMap.delete(oldest.value);
      }
    }
  }
  structureHashMap.set(key, now);
  return false;
}

const stageFailureCounts: Record<string, number> = {
  formula_filter: 0,
  pp_validation: 0,
  geometry: 0,
  duplicate: 0,
  xtb_prefilter: 0,
  scf: 0,
  bands: 0,
  phonon: 0,
};

export function getStageFailureCounts(): Record<string, number> {
  return { ...stageFailureCounts };
}

function getAtomicNumber(el: string): number {
  const data = getElementData(el);
  return data ? data.atomicNumber : 0;
}

const RY_TO_EV = 13.605693122994;
const BOHR_TO_ANG = 0.529177210903;

function resolvePPFilename(element: string): string {
  const ppDir = QE_PSEUDO_DIR;
  const simpleName = `${element}.UPF`;
  if (fs.existsSync(path.join(ppDir, simpleName))) return simpleName;
  try {
    const entries = fs.readdirSync(ppDir);
    const match = entries.find(f => f.startsWith(element + ".") && (f.endsWith(".UPF") || f.endsWith(".upf")));
    if (match) return match;
  } catch {}
  return simpleName;
}

function detectPPType(element: string): "paw" | "uspp" | "nc" {
  const ppPath = path.join(QE_PSEUDO_DIR, resolvePPFilename(element));
  try {
    const head = fs.readFileSync(ppPath, "utf-8").slice(0, 2000);
    if (head.includes('is_paw="true"') || head.includes("is_paw='.true.'") || head.includes("pseudo_type=\"PAW\"")) return "paw";
    if (head.includes('is_ultrasoft="true"') || head.includes("is_ultrasoft='.true.'") || head.includes("pseudo_type=\"US\"") || head.includes("Ultrasoft")) return "uspp";
    if (head.includes("pseudo_type=\"NC\"") || head.includes("Norm-Conserving") || head.includes("norm-conserving")) return "nc";
    if (head.includes('is_ultrasoft="false"') && head.includes('is_paw="false"')) return "nc";
  } catch {}
  return "paw";
}

function ecutrhoMultiplier(elements: string[]): number {
  let allNC = true;
  for (const el of elements) {
    const ppType = detectPPType(el);
    if (ppType !== "nc") { allNC = false; break; }
  }
  return allNC ? 4 : 8;
}

function getAtomicMass(el: string): number {
  const local = ELEMENT_DATA[el];
  if (local) return local.mass;
  const central = getElementData(el);
  return central ? central.atomicMass : 50;
}

function getZValence(el: string): number {
  const local = ELEMENT_DATA[el];
  if (local) return local.zValence;
  const central = getElementData(el);
  return central ? central.valenceElectrons : 4;
}

function computeStructureFingerprint(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  cOverA: number = 1.0,
  gammaRad: number = Math.PI / 2,
): string {
  const n = positions.length;
  if (n === 0) return "empty";

  const composition = [...positions]
    .map(p => p.element)
    .sort()
    .join(",");

  const labeledDistances: string[] = [];
  const coordCounts: number[] = new Array(n).fill(0);
  const coordCutoff = latticeA * 0.4;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let fdx = positions[i].x - positions[j].x;
      let fdy = positions[i].y - positions[j].y;
      let fdz = positions[i].z - positions[j].z;
      fdx -= Math.round(fdx);
      fdy -= Math.round(fdy);
      fdz -= Math.round(fdz);
      const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA, cOverA, 1.0, gammaRad);

      const [elA, elB] = [positions[i].element, positions[j].element].sort();
      labeledDistances.push(`${elA}-${elB}:${(Math.round(dist * 20) / 20).toFixed(2)}`);

      if (dist < coordCutoff) {
        coordCounts[i]++;
        coordCounts[j]++;
      }
    }
  }

  labeledDistances.sort();
  const coordProfile = coordCounts.sort((a, b) => a - b).join(",");

  const fingerprintStr = [
    `comp=${composition}`,
    `lat=${(Math.round(latticeA * 20) / 20).toFixed(2)}`,
    `n=${n}`,
    `dists=${labeledDistances.join(";")}`,
    `coord=${coordProfile}`,
  ].join("|");

  return crypto.createHash("md5").update(fingerprintStr).digest("hex");
}

function approximateEigenvalues(matrix: Float64Array[], n: number): number[] {
  const eigenvalues: number[] = [];
  const a = matrix.map(row => new Float64Array(row));
  const maxIter = Math.max(n * 10, 50);

  for (let iter = 0; iter < maxIter; iter++) {
    let maxVal = 0;
    let p = 0, q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(a[i][j]) > maxVal) {
          maxVal = Math.abs(a[i][j]);
          p = i;
          q = j;
        }
      }
    }
    if (maxVal < 1e-10) break;

    const theta = 0.5 * Math.atan2(2 * a[p][q], a[p][p] - a[q][q]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    for (let i = 0; i < n; i++) {
      if (i === p || i === q) continue;
      const aip = a[i][p];
      const aiq = a[i][q];
      a[i][p] = a[p][i] = c * aip + s * aiq;
      a[i][q] = a[q][i] = -s * aip + c * aiq;
    }
    const app = a[p][p];
    const aqq = a[q][q];
    const apq = a[p][q];
    a[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
  }

  for (let i = 0; i < n; i++) {
    eigenvalues.push(a[i][i]);
  }
  return eigenvalues;
}

function approximateHermitianEigenvalues(
  realPart: Float64Array[], imagPart: Float64Array[], n: number,
): number[] {
  const hr = realPart.map(row => new Float64Array(row));
  const hi = imagPart.map(row => new Float64Array(row));
  const maxIter = Math.max(n * 15, 100);

  for (let iter = 0; iter < maxIter; iter++) {
    let maxOff = 0;
    let p = 0, q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const mag = Math.sqrt(hr[i][j] * hr[i][j] + hi[i][j] * hi[i][j]);
        if (mag > maxOff) {
          maxOff = mag;
          p = i;
          q = j;
        }
      }
    }
    if (maxOff < 1e-10) break;

    const mag_pq = Math.sqrt(hr[p][q] * hr[p][q] + hi[p][q] * hi[p][q]);
    if (mag_pq < 1e-14) continue;
    const phaseR = hr[p][q] / mag_pq;
    const phaseI = -hi[p][q] / mag_pq;

    for (let i = 0; i < n; i++) {
      const origR = hr[i][q];
      const origI = hi[i][q];
      hr[i][q] = origR * phaseR - origI * phaseI;
      hi[i][q] = origR * phaseI + origI * phaseR;
      hr[q][i] = hr[i][q];
      hi[q][i] = -hi[i][q];
    }
    for (let j = 0; j < n; j++) {
      const origR = hr[q][j];
      const origI = hi[q][j];
      hr[q][j] = origR * phaseR + origI * phaseI;
      hi[q][j] = -origR * phaseI + origI * phaseR;
      hr[j][q] = hr[q][j];
      hi[j][q] = -hi[q][j];
    }

    const diff = hr[p][p] - hr[q][q];
    const theta = 0.5 * Math.atan2(2 * hr[p][q], diff);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    for (let i = 0; i < n; i++) {
      if (i === p || i === q) continue;
      const ripR = hr[i][p]; const riqR = hr[i][q];
      const ripI = hi[i][p]; const riqI = hi[i][q];
      hr[i][p] = c * ripR + s * riqR;
      hi[i][p] = c * ripI + s * riqI;
      hr[i][q] = -s * ripR + c * riqR;
      hi[i][q] = -s * ripI + c * riqI;
      hr[p][i] = hr[i][p]; hi[p][i] = -hi[i][p];
      hr[q][i] = hr[i][q]; hi[q][i] = -hi[i][q];
    }
    const app = hr[p][p]; const aqq = hr[q][q]; const apq = hr[p][q];
    hr[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq;
    hr[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq;
    hr[p][q] = hr[q][p] = 0;
    hi[p][q] = hi[q][p] = 0;
  }

  const eigenvalues: number[] = [];
  for (let i = 0; i < n; i++) {
    eigenvalues.push(hr[i][i]);
  }
  return eigenvalues;
}

// GFN2-xTB single-point energies for isolated neutral atoms (in Hartree).
// Source: xTB reference calculations with --gfn 2 --sp.
// These are the values that make formation-like energies well-behaved.
const GFN2_ATOMIC_REF: Record<string, number> = {
  H: -0.393_749, He: -1.718_344,
  Li: -0.188_155, Be: -0.966_432, B: -2.459_449, C: -3.741_225,
  N: -5.764_075, O: -4.768_053, F: -5.834_508, Ne: -6.820_688,
  Na: -0.261_836, Mg: -0.888_816, Al: -1.783_405, Si: -3.083_040,
  P: -4.529_353, S: -3.867_572, Cl: -4.469_341, Ar: -6.186_688,
  K: -0.219_538, Ca: -0.754_494, Sc: -2.614_937, Ti: -4.196_648,
  V: -6.097_289, Cr: -5.697_460, Mn: -7.620_613, Fe: -9.153_591,
  Co: -8.898_965, Ni: -8.666_012, Cu: -4.588_018, Zn: -1.755_420,
  Ga: -2.128_279, Ge: -3.518_424, As: -5.038_513, Se: -4.476_157,
  Br: -5.117_131, Kr: -6.989_271,
  Rb: -0.204_093, Sr: -0.697_613, Y: -2.459_204, Zr: -4.120_024,
  Nb: -5.842_013, Mo: -5.462_007, Tc: -7.244_500, Ru: -8.832_344,
  Rh: -8.505_524, Pd: -7.969_177, Ag: -4.023_944, Cd: -1.564_340,
  In: -1.818_785, Sn: -3.162_282, Sb: -4.619_468, Te: -4.125_285,
  I: -4.742_513, Xe: -6.571_015,
  Cs: -0.190_612, Ba: -0.665_960, La: -2.275_637,
  Ce: -2.338_793, Pr: -2.393_862, Nd: -2.464_419, Pm: -2.549_010,
  Sm: -2.623_527, Eu: -2.700_039, Gd: -2.789_577, Tb: -2.869_080,
  Dy: -2.944_525, Ho: -3.019_945, Er: -3.097_490, Tm: -3.178_100,
  Yb: -3.247_670, Lu: -3.341_840,
  Hf: -5.015_060, Ta: -6.892_620, W: -6.650_160, Re: -8.563_000,
  Os: -10.332_060, Ir: -9.975_440, Pt: -9.274_560, Au: -4.852_580,
  Hg: -2.002_540, Tl: -1.903_490, Pb: -3.322_700, Bi: -4.738_550,
};

function runXTBStabilityCheck(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  workDir: string,
  pressureGpa: number = 0,
): { stable: boolean; ePerAtom: number; basis: string } | null {
  try {
    const posElements = Array.from(new Set(positions.map(p => p.element)));
    const scale = pressureGpa > 0 ? computePressureScale(pressureGpa, posElements) : 1.0;
    let xyz = `${positions.length}\nstability check${pressureGpa > 0 ? ` @ ${pressureGpa} GPa` : ""}\n`;
    for (const p of positions) {
      xyz += `${p.element}  ${(p.x * latticeA * scale).toFixed(6)}  ${(p.y * latticeA * scale).toFixed(6)}  ${(p.z * latticeA * scale).toFixed(6)}\n`;
    }
    const xyzFile = path.join(workDir, "stability.xyz");
    fs.writeFileSync(xyzFile, xyz);

    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
      OMP_STACKSIZE: "1G",
    };

    const out = execSync(`${XTB_BIN} ${xyzFile} --gfn 2 --sp 2>&1 || true`, {
      cwd: workDir,
      timeout: 15000,
      maxBuffer: 20 * 1024 * 1024,
      env,
    }).toString();

    const HA_TO_EV = 27.211386;
    const eMatch = out.match(/TOTAL ENERGY\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)\s+Eh/);
    if (!eMatch) return null;
    const raw = eMatch[1].replace(/[dD]/, "e");
    const totalEHa = parseFloat(raw);
    if (!isFinite(totalEHa)) return null;
    const ePerAtomHa = totalEHa / positions.length;
    const ePerAtomEv = ePerAtomHa * HA_TO_EV;

    const elCounts: Record<string, number> = {};
    for (const p of positions) {
      elCounts[p.element] = (elCounts[p.element] || 0) + 1;
    }
    const elements = Object.keys(elCounts);
    const totalAtoms = positions.length;

    let refEPerAtom = 0;
    let hasRef = true;
    for (const el of elements) {
      const atomicE = GFN2_ATOMIC_REF[el];
      if (atomicE === undefined) { hasRef = false; break; }
      refEPerAtom += atomicE * (elCounts[el] / totalAtoms);
    }

    let isStable: boolean;
    let basis: string;
    if (hasRef) {
      const formationLike = (ePerAtomHa - refEPerAtom) * HA_TO_EV;
      isStable = formationLike < 2.0;
      basis = `relative (Ef-like=${formationLike.toFixed(3)} eV/atom)`;
    } else {
      isStable = ePerAtomEv < -1.0;
      basis = `absolute (E/atom=${ePerAtomEv.toFixed(3)} eV)`;
    }

    return { stable: isStable, ePerAtom: ePerAtomEv, basis };
  } catch {
    return null;
  }
}

const VALID_ELEMENTS = new Set([
  "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am",
]);

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
    if (!VALID_ELEMENTS.has(el)) continue;
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (num > 0) counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

const ATOMIC_VOLUMES: Record<string, number> = {
  H: 5, He: 6, Li: 20, Be: 8, B: 8, C: 9, N: 10, O: 12, F: 11, Ne: 13,
  Na: 24, Mg: 14, Al: 17, Si: 20, P: 17, S: 16, Cl: 22, Ar: 24,
  K: 46, Ca: 26, Sc: 25, Ti: 16, V: 14, Cr: 12, Mn: 12,
  Fe: 11, Co: 11, Ni: 11, Cu: 12, Zn: 15, Ga: 20, Ge: 23,
  As: 21, Se: 17, Br: 24, Kr: 27, Rb: 56, Sr: 34, Y: 25, Zr: 23,
  Nb: 18, Mo: 16, Tc: 14, Ru: 14, Rh: 14, Pd: 15, Ag: 17, Cd: 22,
  In: 26, Sn: 27, Sb: 30, Te: 34, I: 26, Xe: 36,
  Cs: 71, Ba: 39, La: 37, Ce: 35, Pr: 35, Nd: 34,
  Pm: 33, Sm: 33, Eu: 36, Gd: 33, Tb: 32, Dy: 32,
  Ho: 32, Er: 31, Tm: 31, Yb: 35, Lu: 30,
  Hf: 22, Ta: 18, W: 16, Re: 15, Os: 14, Ir: 14, Pt: 15, Au: 17,
  Hg: 23, Tl: 29, Pb: 30, Bi: 35, Po: 34,
  Th: 33, Pa: 25, U: 21, Np: 20, Pu: 20,
};

function getAtomicVolume(el: string): number {
  const local = ATOMIC_VOLUMES[el];
  if (local != null) return local;
  const central = getElementData(el);
  if (central && central.atomicRadius > 0) {
    const rAng = central.atomicRadius / 100;
    return (4 / 3) * Math.PI * rAng * rAng * rAng;
  }
  return 15;
}

function estimateLatticeConstant(elements: string[], counts?: Record<string, number>, pressureGPa: number = 0): number {
  let totalVolume = 0;
  let totalAtoms = 0;
  const effectiveCounts: Record<string, number> = {};

  if (counts) {
    for (const el of Object.keys(counts)) {
      const n = Math.round(counts[el] || 1);
      effectiveCounts[el] = n;
      totalVolume += n * getAtomicVolume(el);
      totalAtoms += n;
    }
  } else {
    for (const el of elements) {
      effectiveCounts[el] = (effectiveCounts[el] || 0) + 1;
      totalVolume += getAtomicVolume(el);
      totalAtoms++;
    }
  }

  const hCount = effectiveCounts["H"] || 0;
  const hFraction = totalAtoms > 0 ? hCount / totalAtoms : 0;
  const metalCount = totalAtoms - hCount;
  const hMetalRatio = metalCount > 0 ? hCount / metalCount : 0;
  const hasMetals = elements.some(e => e !== "H" && (
    isTransitionMetal(e) || isRareEarth(e) || ["Ca", "Sr", "Ba", "Mg", "Na", "K", "Al"].includes(e)
  ));

  let cellVolume: number;

  if (hFraction > 0.5 && hasMetals && metalCount > 0) {
    const volPerAtom = 25 + 2.5 * hMetalRatio;
    cellVolume = totalAtoms * volPerAtom;
  } else {
    let packingFactor: number;
    if (hasMetals && totalAtoms <= 4) {
      packingFactor = 0.74;
    } else if (hasMetals) {
      packingFactor = 0.68;
    } else {
      packingFactor = 0.60;
    }
    cellVolume = totalVolume / packingFactor;
  }

  if (pressureGPa > 0) {
    const B0 = estimateBulkModulus(elements);
    const B0p = 4.0;
    const eta = 1 + B0p * (pressureGPa / B0);
    const volRatio = eta > 0 ? Math.pow(eta, -1 / B0p) : 0.5;
    cellVolume = cellVolume * Math.max(0.5, Math.min(1.0, volRatio));
  }

  const a = Math.cbrt(cellVolume);
  const perturbation = 0.97 + Math.random() * 0.06;
  return Math.max(a * perturbation, 3.0);
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

const SEMICORE_REQUIRED: Set<string> = new Set([
  "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
  "Ba", "Sr", "Ca", "K", "Na",
  "Th", "U", "Pu",
]);

function validateSemicorePP(element: string, ppPath: string): boolean {
  if (!SEMICORE_REQUIRED.has(element)) return true;
  try {
    const fd = fs.openSync(ppPath, "r");
    const buf = Buffer.alloc(4096);
    const bytesRead = fs.readSync(fd, buf, 0, 4096, 0);
    fs.closeSync(fd);
    const header = buf.toString("utf-8", 0, bytesRead).toLowerCase();

    const hasSemicoreTag = header.includes("spn") || header.includes("spfn") || header.includes("spdn") || header.includes("spnl");

    if (header.includes("rrkjus") && !hasSemicoreTag) {
      console.log(`[QE-Worker] WARNING: PP for ${element} appears to lack semicore states (rrkjus without sp*). Re-downloading semicore version.`);
      return false;
    }

    const zValMatch = header.match(/z_valence\s*=\s*"?\s*([\d.]+(?:e[+-]?\d+)?)/);
    if (zValMatch) {
      const ppZVal = parseFloat(zValMatch[1]);
      const expectedZ = getZValence(element);
      if (ppZVal > 0 && expectedZ > 0 && ppZVal < expectedZ * 0.6) {
        console.log(`[QE-Worker] WARNING: PP for ${element} has z_valence=${ppZVal} but expected ~${expectedZ} (semicore likely missing). Re-downloading.`);
        return false;
      }
    }

    const nwfcMatch = header.match(/number_of_wfc\s*=\s*"?\s*(\d+)/);
    if (nwfcMatch) {
      const nwfc = parseInt(nwfcMatch[1]);
      if (nwfc < 3 && SEMICORE_REQUIRED.has(element)) {
        console.log(`[QE-Worker] WARNING: PP for ${element} has only ${nwfc} wavefunctions (semicore elements typically need ≥3). Re-downloading.`);
        return false;
      }
    }

    return true;
  } catch (err) {
    console.log(`[QE-Worker] WARNING: Failed to validate semicore PP for ${element}: ${err instanceof Error ? err.message : "unknown error"}`);
    return false;
  }
}

function cleanQETmpDir(tmpDir: string): void {
  if (!fs.existsSync(tmpDir)) return;
  try {
    const entries = fs.readdirSync(tmpDir);
    for (const entry of entries) {
      const fullPath = path.join(tmpDir, entry);
      try {
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory() && (entry.endsWith(".save") || entry.endsWith(".save_tmp"))) {
          fs.rmSync(fullPath, { recursive: true, force: true });
        } else if (entry.endsWith(".xml") || entry.endsWith(".wfc") || entry.endsWith(".mix") || entry.endsWith(".restart_xml")) {
          fs.unlinkSync(fullPath);
        }
      } catch {}
    }
  } catch {}
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

const PP_DOWNLOAD_URLS: Record<string, string> = {
  H: "https://pseudopotentials.quantum-espresso.org/upf_files/H.pbe-kjpaw_psl.1.0.0.UPF",
  Li: "https://pseudopotentials.quantum-espresso.org/upf_files/Li.pbe-s-kjpaw_psl.1.0.0.UPF",
  Be: "https://pseudopotentials.quantum-espresso.org/upf_files/Be.pbe-n-kjpaw_psl.1.0.0.UPF",
  B: "https://pseudopotentials.quantum-espresso.org/upf_files/B.pbe-n-kjpaw_psl.1.0.0.UPF",
  C: "https://pseudopotentials.quantum-espresso.org/upf_files/C.pbe-n-kjpaw_psl.1.0.0.UPF",
  N: "https://pseudopotentials.quantum-espresso.org/upf_files/N.pbe-n-kjpaw_psl.1.0.0.UPF",
  O: "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF",
  F: "https://pseudopotentials.quantum-espresso.org/upf_files/F.pbe-n-kjpaw_psl.1.0.0.UPF",
  Na: "https://pseudopotentials.quantum-espresso.org/upf_files/Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Mg: "https://pseudopotentials.quantum-espresso.org/upf_files/Mg.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Al: "https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-kjpaw_psl.1.0.0.UPF",
  Si: "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-kjpaw_psl.1.0.0.UPF",
  P: "https://pseudopotentials.quantum-espresso.org/upf_files/P.pbe-n-kjpaw_psl.1.0.0.UPF",
  S: "https://pseudopotentials.quantum-espresso.org/upf_files/S.pbe-n-kjpaw_psl.1.0.0.UPF",
  K: "https://pseudopotentials.quantum-espresso.org/upf_files/K.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ca: "https://pseudopotentials.quantum-espresso.org/upf_files/Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Sc: "https://pseudopotentials.quantum-espresso.org/upf_files/Sc.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ti: "https://pseudopotentials.quantum-espresso.org/upf_files/Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
  V: "https://pseudopotentials.quantum-espresso.org/upf_files/V.pbe-spnl-kjpaw_psl.1.0.0.UPF",
  Cr: "https://pseudopotentials.quantum-espresso.org/upf_files/Cr.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Mn: "https://pseudopotentials.quantum-espresso.org/upf_files/Mn.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Fe: "https://pseudopotentials.quantum-espresso.org/upf_files/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Co: "https://pseudopotentials.quantum-espresso.org/upf_files/Co.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ni: "https://pseudopotentials.quantum-espresso.org/upf_files/Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Cu: "https://pseudopotentials.quantum-espresso.org/upf_files/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Zn: "https://pseudopotentials.quantum-espresso.org/upf_files/Zn.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Ga: "https://pseudopotentials.quantum-espresso.org/upf_files/Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Ge: "https://pseudopotentials.quantum-espresso.org/upf_files/Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
  As: "https://pseudopotentials.quantum-espresso.org/upf_files/As.pbe-n-kjpaw_psl.1.0.0.UPF",
  Se: "https://pseudopotentials.quantum-espresso.org/upf_files/Se.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Sr: "https://pseudopotentials.quantum-espresso.org/upf_files/Sr.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Y: "https://pseudopotentials.quantum-espresso.org/upf_files/Y.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Zr: "https://pseudopotentials.quantum-espresso.org/upf_files/Zr.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Nb: "https://pseudopotentials.quantum-espresso.org/upf_files/Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Mo: "https://pseudopotentials.quantum-espresso.org/upf_files/Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Tc: "https://pseudopotentials.quantum-espresso.org/upf_files/Tc.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ru: "https://pseudopotentials.quantum-espresso.org/upf_files/Ru.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Rh: "https://pseudopotentials.quantum-espresso.org/upf_files/Rh.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Pd: "https://pseudopotentials.quantum-espresso.org/upf_files/Pd.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ag: "https://pseudopotentials.quantum-espresso.org/upf_files/Ag.pbe-nd-kjpaw_psl.1.0.0.UPF",
  Pt: "https://pseudopotentials.quantum-espresso.org/upf_files/Pt.pbe-spdn-kjpaw_psl.1.0.0.UPF",
  Au: "https://pseudopotentials.quantum-espresso.org/upf_files/Au.pbe-nd-kjpaw_psl.1.0.0.UPF",
  La: "https://pseudopotentials.quantum-espresso.org/upf_files/La.pbe-spfn-kjpaw_psl.1.0.0.UPF",
  Ce: "https://pseudopotentials.quantum-espresso.org/upf_files/Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF",
  Ba: "https://pseudopotentials.quantum-espresso.org/upf_files/Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Hf: "https://pseudopotentials.quantum-espresso.org/upf_files/Hf.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ta: "https://pseudopotentials.quantum-espresso.org/upf_files/Ta.pbe-spn-kjpaw_psl.1.0.0.UPF",
  W: "https://pseudopotentials.quantum-espresso.org/upf_files/W.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Th: "https://pseudopotentials.quantum-espresso.org/upf_files/Th.pbe-spfn-kjpaw_psl.1.0.0.UPF",
  Pb: "https://pseudopotentials.quantum-espresso.org/upf_files/Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Bi: "https://pseudopotentials.quantum-espresso.org/upf_files/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Sn: "https://pseudopotentials.quantum-espresso.org/upf_files/Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Te: "https://pseudopotentials.quantum-espresso.org/upf_files/Te.pbe-dn-kjpaw_psl.1.0.0.UPF",
  // Halogens
  Cl: "https://pseudopotentials.quantum-espresso.org/upf_files/Cl.pbe-n-kjpaw_psl.1.0.0.UPF",
  Br: "https://pseudopotentials.quantum-espresso.org/upf_files/Br.pbe-dn-kjpaw_psl.1.0.0.UPF",
  I:  "https://pseudopotentials.quantum-espresso.org/upf_files/I.pbe-n-kjpaw_psl.0.2.UPF",
  // Lanthanides (4f-electron elements — frozen-f-core PAW)
  Pr: "https://pseudopotentials.quantum-espresso.org/upf_files/Pr.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Nd: "https://pseudopotentials.quantum-espresso.org/upf_files/Nd.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Sm: "https://pseudopotentials.quantum-espresso.org/upf_files/Sm.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Eu: "https://pseudopotentials.quantum-espresso.org/upf_files/Eu.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Gd: "https://pseudopotentials.quantum-espresso.org/upf_files/Gd.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Tb: "https://pseudopotentials.quantum-espresso.org/upf_files/Tb.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Dy: "https://pseudopotentials.quantum-espresso.org/upf_files/Dy.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Ho: "https://pseudopotentials.quantum-espresso.org/upf_files/Ho.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Er: "https://pseudopotentials.quantum-espresso.org/upf_files/Er.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Tm: "https://pseudopotentials.quantum-espresso.org/upf_files/Tm.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Yb: "https://pseudopotentials.quantum-espresso.org/upf_files/Yb.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Lu: "https://pseudopotentials.quantum-espresso.org/upf_files/Lu.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  // Actinides
  U:  "https://pseudopotentials.quantum-espresso.org/upf_files/U.pbe-spfn-kjpaw_psl.1.0.0.UPF",
  // Alkali/alkaline-earth metals
  Rb: "https://pseudopotentials.quantum-espresso.org/upf_files/Rb.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Cs: "https://pseudopotentials.quantum-espresso.org/upf_files/Cs.pbe-spn-kjpaw_psl.1.0.0.UPF",
  // Remaining transition metals
  Re: "https://pseudopotentials.quantum-espresso.org/upf_files/Re.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Os: "https://pseudopotentials.quantum-espresso.org/upf_files/Os.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Ir: "https://pseudopotentials.quantum-espresso.org/upf_files/Ir.pbe-spn-kjpaw_psl.1.0.0.UPF",
  // Other
  In: "https://pseudopotentials.quantum-espresso.org/upf_files/In.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Cd: "https://pseudopotentials.quantum-espresso.org/upf_files/Cd.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Tl: "https://pseudopotentials.quantum-espresso.org/upf_files/Tl.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Hg: "https://pseudopotentials.quantum-espresso.org/upf_files/Hg.pbe-dn-kjpaw_psl.1.0.0.UPF",
};

const PP_FALLBACK_URLS: Record<string, string> = {
  H:  "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/H.pbe-kjpaw_psl.1.0.0.UPF",
  C:  "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/C.pbe-n-kjpaw_psl.1.0.0.UPF",
  N:  "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/N.pbe-n-kjpaw_psl.1.0.0.UPF",
  O:  "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/O.pbe-n-kjpaw_psl.1.0.0.UPF",
  Fe: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Cu: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
  Al: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Al.pbe-n-kjpaw_psl.1.0.0.UPF",
  Si: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Si.pbe-n-kjpaw_psl.1.0.0.UPF",
  La: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/La.pbe-spfn-kjpaw_psl.1.0.0.UPF",
  Ti: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Nb: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",
  Pb: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Pb.pbe-dn-kjpaw_psl.1.0.0.UPF",
  // Lanthanide GitHub fallbacks
  I:  "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/I.pbe-n-kjpaw_psl.0.2.UPF",
  Pr: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Pr.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Nd: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Nd.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Sm: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Sm.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Eu: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Eu.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Gd: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Gd.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Tb: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Tb.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Dy: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Dy.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Ho: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Ho.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Er: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Er.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Tm: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Tm.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Yb: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Yb.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  Lu: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Lu.pbe-spdfn-kjpaw_psl.1.0.0.UPF",
  U:  "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/U.pbe-spfn-kjpaw_psl.1.0.0.UPF",
  Ce: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF",
  Th: "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS/Th.pbe-spfn-kjpaw_psl.1.0.0.UPF",
};

function downloadPPToTemp(url: string, tmpFile: string): boolean {
  try {
    execSync(`curl -sL --max-time 25 -o "${tmpFile}" "${url}"`, { timeout: 30000 });
    return fs.existsSync(tmpFile) && fs.statSync(tmpFile).size > 10000;
  } catch {
    try { fs.unlinkSync(tmpFile); } catch {}
    return false;
  }
}

function ensurePseudopotential(element: string): string {
  if (!fs.existsSync(QE_PSEUDO_DIR)) {
    fs.mkdirSync(QE_PSEUDO_DIR, { recursive: true });
  }

  const ppFile = path.join(QE_PSEUDO_DIR, `${element}.UPF`);
  const lockFile = ppFile + ".lock";
  const tmpFile = ppFile + `.tmp.${process.pid}`;

  if (fs.existsSync(ppFile) && validatePseudopotential(ppFile) && validateSemicorePP(element, ppFile)) {
    return ppFile;
  }

  let lockFd: number | null = null;
  try {
    lockFd = fs.openSync(lockFile, "wx");
  } catch {
    for (let wait = 0; wait < 15; wait++) {
      execSync(IS_WINDOWS ? "timeout /t 1 /nobreak >nul 2>&1" : "sleep 1", { timeout: 5000 });
      if (fs.existsSync(ppFile) && validatePseudopotential(ppFile) && validateSemicorePP(element, ppFile)) {
        return ppFile;
      }
      if (!fs.existsSync(lockFile)) break;
    }
    try {
      const lockStat = fs.statSync(lockFile);
      if (Date.now() - lockStat.mtimeMs > 60000) {
        try { fs.unlinkSync(lockFile); } catch {}
      }
    } catch {}
    try {
      lockFd = fs.openSync(lockFile, "wx");
    } catch {
      if (fs.existsSync(ppFile) && validatePseudopotential(ppFile) && validateSemicorePP(element, ppFile)) return ppFile;
      throw new Error(`No valid pseudopotential for ${element} — another worker holds the lock and PP is not yet ready`);
    }
  }

  try {
    if (fs.existsSync(ppFile)) {
      const reason = !validatePseudopotential(ppFile) ? "invalid" : "missing semicore states";
      console.log(`[QE-Worker] PP for ${element} rejected (${reason}, ${fs.statSync(ppFile).size} bytes), removing`);
      try { fs.unlinkSync(ppFile); } catch {}
    }

    // Check project pseudo dir
    const sourceFile = path.join(PP_SOURCE_DIR, `${element}.UPF`);
    if (fs.existsSync(sourceFile) && validatePseudopotential(sourceFile) && validateSemicorePP(element, sourceFile)) {
      fs.copyFileSync(sourceFile, tmpFile);
      fs.renameSync(tmpFile, ppFile);
      console.log(`[QE-Worker] Copied valid PP for ${element} from repo (${fs.statSync(ppFile).size} bytes)`);
      return ppFile;
    }

    // On Windows: also check WSL system pseudo dirs (apt-installed QE pseudopotentials)
    if (IS_WINDOWS) {
      const wslPseudoDirs = ["/usr/share/espresso/pseudo", "/usr/share/espresso/sssp/efficiency", "/usr/share/espresso/sssp"];
      for (const wslDir of wslPseudoDirs) {
        try {
          const listing = execSync(`wsl.exe -d Ubuntu -- bash -c "ls '${wslDir}' 2>/dev/null"`, { encoding: "utf8", timeout: 5000 })
            .split("\n").map(f => f.trim().replace(/\r/g, "")).filter(Boolean);
          const match = listing.find(f => f.startsWith(element + ".") && (f.endsWith(".UPF") || f.endsWith(".upf")));
          if (match) {
            execSync(`wsl.exe -d Ubuntu -- bash -c "cp '${wslDir}/${match}' '${toWslPath(tmpFile)}'"`, { timeout: 10000 });
            if (fs.existsSync(tmpFile) && validatePseudopotential(tmpFile) && validateSemicorePP(element, tmpFile)) {
              fs.renameSync(tmpFile, ppFile);
              console.log(`[QE-Worker] Copied valid PP for ${element} from WSL ${wslDir} (${fs.statSync(ppFile).size} bytes)`);
              return ppFile;
            }
            try { fs.unlinkSync(tmpFile); } catch {}
          }
        } catch {}
      }
    }

    const urls = [PP_DOWNLOAD_URLS[element], PP_FALLBACK_URLS[element]].filter(Boolean);
    for (const url of urls) {
      try {
        const mirror = url!.includes("github") ? "pslibrary" : "QE";
        console.log(`[QE-Worker] Downloading PP for ${element} from ${mirror}...`);
        if (!downloadPPToTemp(url!, tmpFile)) {
          console.log(`[QE-Worker] Download from ${mirror} failed for ${element}`);
          continue;
        }
        if (!validatePseudopotential(tmpFile)) {
          try { fs.unlinkSync(tmpFile); } catch {}
          console.log(`[QE-Worker] Downloaded PP for ${element} from ${mirror} failed validation`);
          continue;
        }
        if (!validateSemicorePP(element, tmpFile)) {
          console.log(`[QE-Worker] Downloaded PP for ${element} from ${mirror} lacks semicore states, removing`);
          try { fs.unlinkSync(tmpFile); } catch {}
          continue;
        }
        fs.renameSync(tmpFile, ppFile);
        console.log(`[QE-Worker] Downloaded valid PP for ${element} from ${mirror} (${fs.statSync(ppFile).size} bytes)`);
        return ppFile;
      } catch (dlErr: any) {
        console.log(`[QE-Worker] PP download failed for ${element}: ${dlErr.message?.slice(0, 100)}`);
        try { fs.unlinkSync(tmpFile); } catch {}
      }
    }

    throw new Error(`No valid pseudopotential for ${element} — need a verified UPF file in ${PP_SOURCE_DIR}/${element}.UPF`);
  } finally {
    if (lockFd !== null) {
      try { fs.closeSync(lockFd); } catch {}
      try { fs.unlinkSync(lockFile); } catch {}
    }
    try { fs.unlinkSync(tmpFile); } catch {}
  }
}

function estimateCOverA(elements: string[], counts: Record<string, number>): number {
  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasFe = elements.includes("Fe");
  const hasAs = elements.includes("As");
  const hasP = elements.includes("P");
  const hasSe = elements.includes("Se");
  const hasTe = elements.includes("Te");
  const hasS = elements.includes("S");
  const hasBi = elements.includes("Bi");

  if (hasCu && hasO) {
    const oCount = counts["O"] || 0;
    const cuCount = counts["Cu"] || 0;
    if (oCount >= 2 && cuCount >= 1) return 3.0;
  }

  if (hasFe && (hasAs || hasP || hasSe || hasTe || hasS)) {
    return 2.5;
  }

  if (hasBi && hasS) return 2.2;
  if (hasBi && hasSe) return 2.3;

  const layeredElements = ["Bi", "Sb", "Te", "Se", "S"];
  const layeredCount = elements.filter(el => layeredElements.includes(el)).length;
  if (layeredCount >= 2) return 2.2;

  return 1.0;
}

function estimateBOverA(elements: string[], counts: Record<string, number>): number {
  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasBa = elements.includes("Ba");
  const hasY = elements.includes("Y");

  if (hasCu && hasO && hasBa && hasY) return 1.01;

  if (hasCu && hasO) {
    const oCount = counts["O"] || 0;
    const cuCount = counts["Cu"] || 0;
    if (oCount >= 2 && cuCount >= 1) return 1.02;
  }

  const hasFe = elements.includes("Fe");
  const hasAs = elements.includes("As");
  const hasP = elements.includes("P");
  const hasSe = elements.includes("Se");
  if (hasFe && (hasAs || hasP || hasSe)) return 1.0;

  return 1.0;
}

function autoKPoints(latticeA: number, cOverA?: number, minK: number = 12, dimensionality?: string): string {
  const densityFactor = 80;
  const isLayered = dimensionality === "quasi-2D" || dimensionality === "2D";
  const layeredBoost = isLayered ? 1.5 : 1.0;
  const effCOverA = cOverA ?? 1.0;
  const ka = Math.max(minK, Math.ceil(densityFactor / latticeA));
  const kb = ka;
  const baseKc = Math.ceil(densityFactor / (latticeA * effCOverA));
  const kc = Math.max(minK, isLayered ? Math.ceil(baseKc * layeredBoost) : baseKc);
  return `  ${ka} ${kb} ${kc}  0 0 0`;
}

const MAGNETIC_ELEMENTS: Record<string, number> = {
  Fe: 2.0, Co: 1.5, Ni: 0.8, Mn: 3.0, Cr: 1.5,
  V: 0.5, Gd: 7.0, Eu: 7.0, Nd: 3.0, Sm: 1.0,
};

function isAFMCandidate(elements: string[], counts: Record<string, number>): boolean {
  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasFe = elements.includes("Fe");
  const hasAs = elements.includes("As");
  const hasP = elements.includes("P");
  const hasSe = elements.includes("Se");
  const hasTe = elements.includes("Te");

  if (hasCu && hasO) {
    const oCount = counts["O"] || 0;
    const cuCount = counts["Cu"] || 0;
    if (oCount >= 2 && cuCount >= 1) return true;
  }

  if (hasFe && (hasAs || hasP || hasSe || hasTe)) return true;

  if (elements.includes("Mn") && hasO) return true;
  if (elements.includes("Cr") && hasO) return true;

  return false;
}

function determineAFMPattern(elements: string[], counts: Record<string, number>): "checkerboard" | "layered" | "alternating" {
  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasFe = elements.includes("Fe");
  const hasAs = elements.includes("As");
  const hasP = elements.includes("P");

  if (hasCu && hasO) return "layered";
  if (hasFe && (hasAs || hasP)) return "checkerboard";
  if (elements.includes("Mn") && hasO) return "checkerboard";
  return "alternating";
}

function generateMagnetizationLines(
  elements: string[],
  counts: Record<string, number>,
  useAFM: boolean,
): string {
  let lines = "";
  const magneticIndices: number[] = [];

  for (let idx = 0; idx < elements.length; idx++) {
    const el = elements[idx];
    if (el in MAGNETIC_ELEMENTS) {
      magneticIndices.push(idx);
    }
  }

  if (magneticIndices.length === 0) return "";

  const afmPattern = useAFM ? determineAFMPattern(elements, counts) : null;

  for (let idx = 0; idx < elements.length; idx++) {
    const el = elements[idx];
    let mag = MAGNETIC_ELEMENTS[el] ?? 0.0;

    if (useAFM && mag !== 0) {
      const magSubIndex = magneticIndices.indexOf(idx);
      if (magSubIndex >= 0) {
        if (afmPattern === "checkerboard") {
          mag = magSubIndex % 2 === 1 ? -mag : mag;
        } else if (afmPattern === "layered") {
          const nMag = magneticIndices.length;
          const halfPoint = Math.ceil(nMag / 2);
          mag = magSubIndex >= halfPoint ? -mag : mag;
        } else {
          mag = magSubIndex % 2 === 1 ? -mag : mag;
        }
      }
    }

    lines += `  starting_magnetization(${idx + 1}) = ${mag.toFixed(1)},\n`;
  }
  return lines;
}

function determineCrystalSystem(elements: string[], counts: Record<string, number>): { ibrav: number; cOverA: number } {
  const cOverA = estimateCOverA(elements, counts);

  if (Math.abs(cOverA - 1.0) < 0.05) {
    return { ibrav: 1, cOverA: 1.0 };
  }

  const hasB = elements.includes("B");
  const hasMg = elements.includes("Mg");
  const hasAl = elements.includes("Al");
  const hasTi = elements.includes("Ti");
  const hasZr = elements.includes("Zr");
  const hasHf = elements.includes("Hf");

  const hexagonalIndicators = (hasB && (hasMg || hasAl)) ||
    ((hasTi || hasZr || hasHf) && elements.length === 2 && cOverA > 1.4 && cOverA < 1.8);

  if (hexagonalIndicators) {
    return { ibrav: 4, cOverA };
  }

  return { ibrav: 0, cOverA };
}

function generateCellParameters(latticeA: number, cOverA: number, ibrav: number, bOverA: number = 1.0, elements?: string[], counts?: Record<string, number>): string {
  const a = latticeA;
  const b = latticeA * bOverA;
  const c = latticeA * cOverA;
  const isHexagonal = ibrav === 4 || ((elements && counts)
    ? determineCrystalSystem(elements, counts).ibrav === 4
    : (cOverA > 1.4 && cOverA < 1.8 && Math.abs(bOverA - 1.0) < 0.05));
  if (isHexagonal) {
    return `CELL_PARAMETERS {angstrom}
  ${a.toFixed(8)}  0.000000000  0.000000000
  ${(-a / 2).toFixed(8)}  ${(a * Math.sqrt(3) / 2).toFixed(8)}  0.000000000
  0.000000000  0.000000000  ${c.toFixed(8)}`;
  }
  return `CELL_PARAMETERS {angstrom}
  ${a.toFixed(8)}  0.000000000  0.000000000
  0.000000000  ${b.toFixed(8)}  0.000000000
  0.000000000  0.000000000  ${c.toFixed(8)}`;
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
    H: 100, O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
    Li: 60, Be: 60, B: 55, C: 60, Na: 60, Mg: 55, Al: 50, Si: 50,
  };
  const hasHydrogen = elements.includes("H");
  const baseEcutwfc = elements.reduce((max, el) => Math.max(max, ELEMENT_CUTOFFS[el] ?? 45), hasHydrogen ? 80 : 45);
  const ecutwfc = Math.max(baseEcutwfc, hasHydrogen ? 100 : 60);
  const ecutrho = ecutwfc * ecutrhoMultiplier(elements);

  const hasMagnetic = elements.some(el => el in MAGNETIC_ELEMENTS);
  const nspin = hasMagnetic ? 2 : 1;
  const useAFM = hasMagnetic && isAFMCandidate(elements, counts);

  let startingMagLines = "";
  if (hasMagnetic) {
    startingMagLines = generateMagnetizationLines(elements, counts, useAFM);
  }

  let atomicSpecies = "";
  for (const el of elements) {
    const mass = getAtomicMass(el);
    if (!mass) {
      throw new Error(`Unknown element "${el}" — no atomic mass data available. Cannot generate valid QE input.`);
    }
    atomicSpecies += `  ${el}  ${mass.toFixed(3)}  ${resolvePPFilename(el)}\n`;
  }

  let atomicPositions = "";
  const positions = generateAtomicPositions(elements, counts);
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  const { cOverA } = determineCrystalSystem(elements, counts);
  const bOverA = estimateBOverA(elements, counts);
  const cellBlock = `\n${generateCellParameters(latticeA, cOverA, 0, bOverA, elements, counts)}`;

  return `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${formula.replace(/[^a-zA-Z0-9]/g, "")}',
  outdir = './tmp',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = 1.0d-2,
  etot_conv_thr = 1.0d-4,
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.005,
  nspin = ${nspin},
${startingMagLines}/
&ELECTRONS
  electron_maxstep = 200,
  conv_thr = 1.0d-10,
  mixing_beta = 0.3,
  mixing_mode = 'plain',
  diagonalization = 'david',
  scf_must_converge = .false.,
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${autoKPoints(latticeA, cOverA, bOverA)}
${cellBlock}
`;
}

function perturbCoord(v: number, sigma: number = 0.005): number {
  const u1 = Math.random() || 1e-10;
  const u2 = Math.random();
  const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sigma;
  let result = v + noise;
  result = result - Math.floor(result);
  return result;
}

function perturbPositions(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  sigma: number = 0.005,
  latticeA?: number,
): Array<{ element: string; x: number; y: number; z: number }> {
  if (!latticeA || latticeA <= 0) {
    return positions.map(p => ({
      element: p.element,
      x: perturbCoord(p.x, sigma),
      y: perturbCoord(p.y, sigma),
      z: perturbCoord(p.z, sigma),
    }));
  }

  const result = positions.map(p => ({ ...p }));
  for (let i = 0; i < result.length; i++) {
    for (let attempt = 0; attempt < 10; attempt++) {
      const px = perturbCoord(result[i].x, sigma);
      const py = perturbCoord(result[i].y, sigma);
      const pz = perturbCoord(result[i].z, sigma);

      let valid = true;
      for (let j = 0; j < result.length; j++) {
        if (j === i) continue;
        const ox = j < i ? result[j].x : positions[j].x;
        const oy = j < i ? result[j].y : positions[j].y;
        const oz = j < i ? result[j].z : positions[j].z;
        let fdx = px - ox; fdx -= Math.round(fdx);
        let fdy = py - oy; fdy -= Math.round(fdy);
        let fdz = pz - oz; fdz -= Math.round(fdz);
        const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA);
        const dMin = minPairDistance(result[i].element, result[j < i ? j : j].element);
        if (dist < dMin) { valid = false; break; }
      }
      if (valid) {
        result[i].x = px;
        result[i].y = py;
        result[i].z = pz;
        break;
      }
      if (attempt === 9) {
        // keep original position unperturbed
      }
    }
  }
  return result;
}

function generateAtomicPositions(
  elements: string[],
  counts: Record<string, number>,
  formula?: string,
  latticeA?: number,
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
          return perturbPositions(positions, 0.005, latticeA);
        }
      }
    } catch {}

    try {
      const lfStruct = generatePrototypeFreeStructure(formula);
      if (lfStruct && lfStruct.atoms.length > 0) {
        const positions = lfStruct.atoms.map(a => ({
          element: a.element,
          x: a.fx,
          y: a.fy,
          z: a.fz,
        }));
        console.log(`[QE-Worker] Using lattice-free generation for ${formula} (${positions.length} atoms, ${lfStruct.bravaisType})`);
        return positions;
      }
    } catch {}
  }

  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  if (totalAtoms <= 2 && elements.length === 2) {
    positions.push({ element: elements[0], x: 0, y: 0, z: 0 });
    positions.push({ element: elements[1], x: 0.5, y: 0.5, z: 0.5 });
    return perturbPositions(positions, 0.005, latticeA);
  }

  if (totalAtoms <= 2 && elements.length === 1) {
    positions.push({ element: elements[0], x: 0, y: 0, z: 0 });
    if (totalAtoms > 1) {
      positions.push({ element: elements[0], x: 0.5, y: 0.5, z: 0.5 });
    }
    return perturbPositions(positions, 0.005, latticeA);
  }

  const hCount = Math.round(counts["H"] || 0);
  const metalElements = elements.filter(e => e !== "H");
  const metalCount = metalElements.reduce((s, e) => s + Math.round(counts[e] || 0), 0);

  if (hCount > 0 && metalCount > 0 && hCount / metalCount >= 4) {
    const hPerMetal = Math.round(hCount / metalCount);
    const effectiveLatticeA = latticeA ?? estimateLatticeConstant(elements, counts);
    const cagePositions = generateHydrideCagePositions(metalElements, counts, hPerMetal, totalAtoms, effectiveLatticeA);
    if (cagePositions.length === totalAtoms && cagePositions.length <= 16) {
      if (latticeA && latticeA > 0) {
        const distValid = validatePositionDistances(cagePositions, latticeA);
        if (distValid) {
          console.log(`[QE-Worker] Using hydride cage motif for ${formula} (H/metal=${hPerMetal}, ${cagePositions.length} atoms, dist-checked)`);
          return perturbPositions(cagePositions, 0.005, latticeA);
        } else {
          console.log(`[QE-Worker] Hydride cage for ${formula} failed distance check, trying lattice-free fallback`);
        }
      } else {
        console.log(`[QE-Worker] Using hydride cage motif for ${formula} (H/metal=${hPerMetal}, ${cagePositions.length} atoms)`);
        return perturbPositions(cagePositions, 0.005);
      }
    }
  }

  const BCC_ELEMENTS = new Set(["Fe", "Cr", "V", "Nb", "Mo", "W", "Ta", "Na", "K", "Rb", "Cs", "Ba", "Li"]);
  const HCP_ELEMENTS = new Set(["Ti", "Zr", "Hf", "Mg", "Be", "Sc", "Y", "Co", "Zn", "Cd", "Re", "Os", "Ru"]);

  const fccSites = [
    [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.75], [0.25, 0.25, 0.75], [0.75, 0.25, 0.25],
    [0.25, 0.75, 0.25], [0.0, 0.25, 0.25], [0.25, 0.0, 0.25], [0.25, 0.25, 0.0],
  ];
  const bccSites = [
    [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
    [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
    [0.25, 0.25, 0.0], [0.0, 0.25, 0.25], [0.25, 0.0, 0.25],
    [0.75, 0.75, 0.5], [0.75, 0.5, 0.75], [0.5, 0.75, 0.75], [0.75, 0.25, 0.5],
    [0.25, 0.75, 0.5], [0.5, 0.25, 0.75], [0.5, 0.75, 0.25], [0.75, 0.5, 0.25],
  ];
  const hcpSites = [
    [0.0, 0.0, 0.0], [0.333, 0.333, 0.0], [0.667, 0.667, 0.0],
    [0.0, 0.0, 0.5], [0.333, 0.333, 0.5], [0.667, 0.667, 0.5],
    [0.167, 0.167, 0.25], [0.5, 0.5, 0.25], [0.833, 0.833, 0.25],
    [0.167, 0.167, 0.75], [0.5, 0.5, 0.75], [0.833, 0.833, 0.75],
    [0.25, 0.0, 0.125], [0.0, 0.25, 0.375], [0.75, 0.5, 0.125], [0.5, 0.75, 0.375],
  ];

  const bccCount = elements.reduce((s, e) => s + (BCC_ELEMENTS.has(e) ? Math.round(counts[e] || 1) : 0), 0);
  const hcpCount = elements.reduce((s, e) => s + (HCP_ELEMENTS.has(e) ? Math.round(counts[e] || 1) : 0), 0);
  let cubicSites: number[][];
  if (bccCount > hcpCount && bccCount > 0) {
    cubicSites = bccSites;
  } else if (hcpCount > bccCount && hcpCount > 0) {
    cubicSites = hcpSites;
  } else {
    cubicSites = fccSites;
  }

  let siteIdx = 0;
  const pendingAtoms: Array<{ element: string; remaining: number }> = [];
  for (const el of elements) {
    const n = Math.round(counts[el] || 1);
    let placed = 0;
    for (let i = 0; i < n && siteIdx < cubicSites.length; i++) {
      const site = cubicSites[siteIdx++];
      positions.push({ element: el, x: site[0], y: site[1], z: site[2] });
      placed++;
    }
    if (placed < n) {
      pendingAtoms.push({ element: el, remaining: n - placed });
    }
  }

  if (pendingAtoms.length > 0) {
    for (const { element, remaining } of pendingAtoms) {
      for (let i = 0; i < remaining; i++) {
        let placed = false;
        for (let attempt = 0; attempt < 200; attempt++) {
          const x = Math.round(Math.random() * 20) / 20;
          const y = Math.round(Math.random() * 20) / 20;
          const z = Math.round(Math.random() * 20) / 20;
          const candidate = { element, x, y, z };
          if (latticeA && latticeA > 0) {
            let valid = true;
            for (const p of positions) {
              let fdx = x - p.x; fdx -= Math.round(fdx);
              let fdy = y - p.y; fdy -= Math.round(fdy);
              let fdz = z - p.z; fdz -= Math.round(fdz);
              const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA);
              const dMin = 0.75 * ((COVALENT_RADIUS[element] ?? 1.4) + (COVALENT_RADIUS[p.element] ?? 1.4));
              if (dist < dMin) { valid = false; break; }
            }
            if (!valid) continue;
          }
          positions.push(candidate);
          placed = true;
          break;
        }
        if (!placed) {
          let relaxedPlaced = false;
          for (let relaxAttempt = 0; relaxAttempt < 100; relaxAttempt++) {
            const x = Math.random();
            const y = Math.random();
            const z = Math.random();
            let valid = true;
            if (latticeA && latticeA > 0) {
              for (const p of positions) {
                let fdx = x - p.x; fdx -= Math.round(fdx);
                let fdy = y - p.y; fdy -= Math.round(fdy);
                let fdz = z - p.z; fdz -= Math.round(fdz);
                const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA);
                const dMin = 0.5 * ((COVALENT_RADIUS[element] ?? 1.4) + (COVALENT_RADIUS[p.element] ?? 1.4));
                if (dist < dMin) { valid = false; break; }
              }
            }
            if (valid) {
              positions.push({ element, x, y, z });
              relaxedPlaced = true;
              console.log(`[QE-Worker] Atom ${element} placed with relaxed distance check (0.5×r_cov) after 200 strict attempts failed`);
              break;
            }
          }
          if (!relaxedPlaced) {
            throw new Error(`Cannot place ${element} — cell too small for ${totalAtoms} atoms at latticeA=${latticeA?.toFixed(2) ?? "?"}Å`);
          }
        }
      }
    }
    console.log(`[QE-Worker] Placed ${positions.length} atoms total (${positions.length - cubicSites.length} at generated positions, dist-enforced)`);
  }

  return perturbPositions(positions, 0.005, latticeA);
}

function validatePositionDistances(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
): boolean {
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      let fdx = positions[i].x - positions[j].x; fdx -= Math.round(fdx);
      let fdy = positions[i].y - positions[j].y; fdy -= Math.round(fdy);
      let fdz = positions[i].z - positions[j].z; fdz -= Math.round(fdz);
      const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA);
      const dMin = 0.75 * ((COVALENT_RADIUS[positions[i].element] ?? 1.4) + (COVALENT_RADIUS[positions[j].element] ?? 1.4));
      if (dist < dMin) return false;
    }
  }
  return true;
}

function repairStructureGeometry(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  pressureGPa: number = 0,
): { positions: Array<{ element: string; x: number; y: number; z: number }>; latticeA: number; repaired: boolean } {
  const MAX_REPAIR_ROUNDS = 8;
  const MAX_RESCALES = 3;
  let rescaleCount = 0;
  let repaired = false;
  const repairEls = Array.from(new Set(positions.map(p => p.element)));
  const pressureScale = pressureGPa > 50 ? computePressureScale(pressureGPa, repairEls) : 1.0;
  let curPositions = positions.map(p => ({ ...p }));
  let curLattice = latticeA;

  for (let round = 0; round < MAX_REPAIR_ROUNDS; round++) {
    let violations = 0;
    let worstRatio = 1.0;

    for (let i = 0; i < curPositions.length; i++) {
      for (let j = i + 1; j < curPositions.length; j++) {
        let fdx = curPositions[i].x - curPositions[j].x;
        let fdy = curPositions[i].y - curPositions[j].y;
        let fdz = curPositions[i].z - curPositions[j].z;
        fdx -= Math.round(fdx);
        fdy -= Math.round(fdy);
        fdz -= Math.round(fdz);
        const dist = fracDistAngstrom(fdx, fdy, fdz, curLattice);
        const dMin = 0.7 * ((COVALENT_RADIUS[curPositions[i].element] ?? 1.4) + (COVALENT_RADIUS[curPositions[j].element] ?? 1.4)) * pressureScale;

        if (dist < dMin && dist > 0.01) {
          violations++;
          const ratio = dist / dMin;
          if (ratio < worstRatio) worstRatio = ratio;

          const pushFactor = (dMin / dist - 1.0) * 0.6;
          const pushX = fdx * pushFactor;
          const pushY = fdy * pushFactor;
          const pushZ = fdz * pushFactor;

          curPositions[i].x += pushX * 0.5;
          curPositions[i].y += pushY * 0.5;
          curPositions[i].z += pushZ * 0.5;
          curPositions[j].x -= pushX * 0.5;
          curPositions[j].y -= pushY * 0.5;
          curPositions[j].z -= pushZ * 0.5;

          curPositions[i].x -= Math.floor(curPositions[i].x);
          curPositions[i].y -= Math.floor(curPositions[i].y);
          curPositions[i].z -= Math.floor(curPositions[i].z);
          curPositions[j].x -= Math.floor(curPositions[j].x);
          curPositions[j].y -= Math.floor(curPositions[j].y);
          curPositions[j].z -= Math.floor(curPositions[j].z);
        }
      }
    }

    if (violations === 0) {
      if (round > 0) repaired = true;
      break;
    }

    if (worstRatio < 0.7 && rescaleCount < MAX_RESCALES) {
      const scaleFactor = 1.0 + (1.0 - worstRatio) * 0.3;
      curLattice *= scaleFactor;
      rescaleCount++;
      repaired = true;
    }

    if (round > 0) repaired = true;
  }

  return { positions: curPositions, latticeA: curLattice, repaired };
}

function inferCrystalSystem(elements: string[], counts: Record<string, number>): string {
  const { ibrav } = determineCrystalSystem(elements, counts);
  if (ibrav === 1) return "cubic";
  if (ibrav === 4) return "hexagonal";
  const cOverA = estimateCOverA(elements, counts);
  if (Math.abs(cOverA - 1.0) < 0.15) return "cubic";
  if (cOverA > 1.4 && cOverA < 1.8) return "hexagonal";
  const bOverA = estimateBOverA(elements, counts);
  if (Math.abs(bOverA - 1.0) < 0.05) return "tetragonal";
  return "orthorhombic";
}

function snapToWyckoffSites(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  elements?: string[],
  counts?: Record<string, number>,
): { positions: Array<{ element: string; x: number; y: number; z: number }>; snapped: boolean; snapCount: number } {
  const allDists = getAllDistributions();
  const system = (elements && counts) ? inferCrystalSystem(elements, counts) : "cubic";
  const dist = allDists.find(d => d.system === system) || allDists[0];
  const wyckoffSites = dist.commonWyckoff;

  if (wyckoffSites.length === 0) {
    return { positions, snapped: false, snapCount: 0 };
  }

  const SNAP_THRESHOLD = 0.12;
  const snappedPositions = positions.map(p => ({ ...p }));
  let snapCount = 0;

  for (let i = 0; i < snappedPositions.length; i++) {
    const pos = snappedPositions[i];
    const origX = pos.x, origY = pos.y, origZ = pos.z;
    const pref = getElementSitePreference(pos.element);
    let bestDist = Infinity;
    let bestSite: { typicalX: number; typicalY: number; typicalZ: number } | null = null;

    for (const site of wyckoffSites) {
      if (pref && !pref.preferredWyckoff.includes(site.letter)) continue;

      for (let ox = -1; ox <= 1; ox++) {
        for (let oy = -1; oy <= 1; oy++) {
          for (let oz = -1; oz <= 1; oz++) {
            const sx = site.typicalX + ox;
            const sy = site.typicalY + oy;
            const sz = site.typicalZ + oz;
            const dx = pos.x - sx;
            const dy = pos.y - sy;
            const dz = pos.z - sz;
            const dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 < bestDist) {
              bestDist = dist2;
              bestSite = { typicalX: site.typicalX, typicalY: site.typicalY, typicalZ: site.typicalZ };
            }
          }
        }
      }
    }

    if (!bestSite) {
      for (const site of wyckoffSites) {
        for (let ox = -1; ox <= 1; ox++) {
          for (let oy = -1; oy <= 1; oy++) {
            for (let oz = -1; oz <= 1; oz++) {
              const sx = site.typicalX + ox;
              const sy = site.typicalY + oy;
              const sz = site.typicalZ + oz;
              const dx = pos.x - sx;
              const dy = pos.y - sy;
              const dz = pos.z - sz;
              const dist2 = dx * dx + dy * dy + dz * dz;
              if (dist2 < bestDist) {
                bestDist = dist2;
                bestSite = { typicalX: site.typicalX, typicalY: site.typicalY, typicalZ: site.typicalZ };
              }
            }
          }
        }
      }
    }

    const snapDist = Math.sqrt(bestDist);
    if (bestSite && snapDist < SNAP_THRESHOLD && snapDist > 0.001) {
      const blend = 0.7;
      pos.x = pos.x * (1 - blend) + bestSite.typicalX * blend;
      pos.y = pos.y * (1 - blend) + bestSite.typicalY * blend;
      pos.z = pos.z * (1 - blend) + bestSite.typicalZ * blend;

      pos.x -= Math.floor(pos.x);
      pos.y -= Math.floor(pos.y);
      pos.z -= Math.floor(pos.z);

      let collision = false;
      for (let j = 0; j < snappedPositions.length; j++) {
        if (j === i) continue;
        let fdx = pos.x - snappedPositions[j].x; fdx -= Math.round(fdx);
        let fdy = pos.y - snappedPositions[j].y; fdy -= Math.round(fdy);
        let fdz = pos.z - snappedPositions[j].z; fdz -= Math.round(fdz);
        const pairDist = fracDistAngstrom(fdx, fdy, fdz, latticeA);
        const dMin = minPairDistance(pos.element, snappedPositions[j].element);
        if (pairDist < dMin) { collision = true; break; }
      }

      if (collision) {
        pos.x = origX;
        pos.y = origY;
        pos.z = origZ;
      } else {
        snapCount++;
      }
    }
  }

  return { positions: snappedPositions, snapped: snapCount > 0, snapCount };
}

function hasHighSymmetry(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
): boolean {
  if (positions.length <= 2) return true;

  let highSymCount = 0;
  const highSymCoords = [0, 0.25, 0.5, 0.75, 1/3, 2/3];
  for (const pos of positions) {
    const isHighSym = [pos.x, pos.y, pos.z].every(c => {
      return highSymCoords.some(h => Math.abs(c - h) < 0.05 || Math.abs(c - (1 - h)) < 0.05);
    });
    if (isHighSym) highSymCount++;
  }
  return highSymCount / positions.length >= 0.3;
}

function generateHydrideCagePositions(
  metalElements: string[],
  counts: Record<string, number>,
  hPerMetal: number,
  totalAtoms: number,
  latticeA: number = 5.0,
): Array<{ element: string; x: number; y: number; z: number }> {
  const positions: Array<{ element: string; x: number; y: number; z: number }> = [];

  const TARGET_MH_DIST = 1.9;
  const cageRadius = TARGET_MH_DIST / latticeA;

  function octahedralSites(r: number): Array<{ x: number; y: number; z: number }> {
    return [
      { x: r, y: 0, z: 0 }, { x: 0, y: r, z: 0 }, { x: 0, y: 0, z: r },
      { x: 1 - r, y: 0, z: 0 }, { x: 0, y: 1 - r, z: 0 }, { x: 0, y: 0, z: 1 - r },
    ];
  }

  function cubeVertexSites(r: number): Array<{ x: number; y: number; z: number }> {
    const d = r / Math.sqrt(3);
    return [
      { x: d, y: d, z: d }, { x: 1 - d, y: d, z: d },
      { x: d, y: 1 - d, z: d }, { x: d, y: d, z: 1 - d },
      { x: 1 - d, y: 1 - d, z: d }, { x: 1 - d, y: d, z: 1 - d },
      { x: d, y: 1 - d, z: 1 - d }, { x: 1 - d, y: 1 - d, z: 1 - d },
    ];
  }

  function clathrateSites(r: number): Array<{ x: number; y: number; z: number }> {
    const d = r / Math.sqrt(2);
    return [
      { x: d, y: d, z: 0 }, { x: 1 - d, y: d, z: 0 },
      { x: d, y: 1 - d, z: 0 }, { x: 1 - d, y: 1 - d, z: 0 },
      { x: 0, y: d, z: d }, { x: 0, y: 1 - d, z: d },
      { x: d, y: 0, z: d }, { x: 1 - d, y: 0, z: d },
      { x: 0.5, y: d, z: 0.5 }, { x: 0.5, y: 1 - d, z: 0.5 },
    ];
  }

  function wrapFrac(v: number): number { return v - Math.floor(v); }

  const metalCount = metalElements.reduce((s, e) => s + Math.round(counts[e] || 0), 0);

  function selectHSites(r: number, n: number): Array<{ x: number; y: number; z: number }> {
    if (n <= 6) return octahedralSites(r).slice(0, n);
    if (n <= 8) return cubeVertexSites(r).slice(0, n);
    if (n === 9) return [...cubeVertexSites(r), { x: r, y: r, z: 0 }];
    if (n <= 10) return clathrateSites(r).slice(0, n);
    const extraR = r * 1.1;
    return [...clathrateSites(r), { x: 0, y: extraR, z: 1 - extraR }, { x: extraR, y: 0, z: 1 - extraR }].slice(0, n);
  }

  function placeHAroundCenter(cx: number, cy: number, cz: number, nH: number): void {
    const sites = selectHSites(cageRadius, nH);
    for (const s of sites) {
      const hx = wrapFrac(cx + s.x);
      const hy = wrapFrac(cy + s.y);
      const hz = wrapFrac(cz + s.z);
      let tooClose = false;
      for (const p of positions) {
        let fdx = hx - p.x; fdx -= Math.round(fdx);
        let fdy = hy - p.y; fdy -= Math.round(fdy);
        let fdz = hz - p.z; fdz -= Math.round(fdz);
        const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA);
        const dMin = p.element === "H" ? 1.0 : 0.75 * ((COVALENT_RADIUS["H"] ?? 0.31) + (COVALENT_RADIUS[p.element] ?? 1.4));
        if (dist < dMin) { tooClose = true; break; }
      }
      if (!tooClose) {
        positions.push({ element: "H", x: hx, y: hy, z: hz });
      }
    }
  }

  if (metalCount === 1) {
    const metal = metalElements[0];
    positions.push({ element: metal, x: 0.0, y: 0.0, z: 0.0 });
    placeHAroundCenter(0.0, 0.0, 0.0, hPerMetal);
  } else {
    const metalSites = [
      { x: 0.0, y: 0.0, z: 0.0 },
      { x: 0.5, y: 0.5, z: 0.5 },
      { x: 0.5, y: 0.0, z: 0.0 },
      { x: 0.0, y: 0.5, z: 0.0 },
    ];
    let placed = 0;
    for (const metal of metalElements) {
      const n = Math.round(counts[metal] || 1);
      for (let i = 0; i < n && placed < metalSites.length; i++) {
        positions.push({ element: metal, ...metalSites[placed++] });
      }
    }
    const hTotal = Math.round(counts["H"] || 0);
    const hPerCenter = Math.ceil(hTotal / Math.min(placed, metalSites.length));
    for (let m = 0; m < placed; m++) {
      const center = metalSites[m];
      placeHAroundCenter(center.x, center.y, center.z, hPerCenter);
    }
  }

  return positions;
}

function autoPhononQGrid(elements: string[]): [number, number, number] {
  const hasHydrogen = elements.includes("H");
  if (hasHydrogen) return [6, 6, 6];
  return [4, 4, 4];
}

function generatePhononInput(formula: string, elements: string[] = []): string {
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const [nq1, nq2, nq3] = autoPhononQGrid(elements);
  return `Phonon dispersions on ${nq1}x${nq2}x${nq3} grid
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = 1.0d-14,
  ldisp = .true.,
  nq1 = ${nq1}, nq2 = ${nq2}, nq3 = ${nq3},
/
`;
}

function parseSCFOutput(stdout: string, degaussRy: number = 0.005): QESCFResult {
  const result: QESCFResult = {
    totalEnergy: 0,
    totalEnergyPerAtom: 0,
    fermiEnergy: null,
    bandGap: null,
    isMetallic: true,
    totalForce: null,
    pressure: null,
    converged: false,
    convergenceQuality: "none",
    lastScfAccuracyRy: null,
    nscfIterations: 0,
    wallTimeSeconds: 0,
    magnetization: null,
    error: null,
  };

  const convergenceMatch = stdout.match(/convergence has been achieved in\s+(\d+)\s+iterations/);
  if (convergenceMatch) {
    result.converged = true;
    result.convergenceQuality = "strict";
    result.nscfIterations = parseInt(convergenceMatch[1]);
  }

  const iterMatches = Array.from(stdout.matchAll(/estimated scf accuracy\s+<\s+([\d.Ee+-]+)\s+Ry/g));
  if (iterMatches.length > 0) {
    const lastAccuracy = parseFloat(iterMatches[iterMatches.length - 1][1]);
    result.lastScfAccuracyRy = lastAccuracy;
    result.nscfIterations = Math.max(result.nscfIterations, iterMatches.length);
    if (!result.converged && lastAccuracy < 1.0e-5) {
      result.converged = true;
      result.convergenceQuality = "loose";
    }
  }

  const energyMatch = stdout.match(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/);
  if (energyMatch) {
    result.totalEnergy = parseFloat(energyMatch[1]) * RY_TO_EV;
  }
  if (!energyMatch) {
    const energyLines = [...stdout.matchAll(/total energy\s+=\s+([-\d.]+)\s+Ry/g)];
    if (energyLines.length > 0) {
      result.totalEnergy = parseFloat(energyLines[energyLines.length - 1][1]) * RY_TO_EV;
    }
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
    } else {
      result.bandGap = parseFloat(gapMatch[1]);
    }
    const degaussEv = degaussRy * RY_TO_EV;
    const metallicThreshold = Math.max(0.01, degaussEv * 1.5);
    result.isMetallic = result.bandGap < metallicThreshold;
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
    result.imaginaryCount = result.frequencies.filter(f => f < -20).length;
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

const COVALENT_RADIUS: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Ne: 0.58,
  Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39,
  Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20, Kr: 1.16,
  Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Tc: 1.47, Ru: 1.46, Rh: 1.42, Pd: 1.39,
  Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Xe: 1.40,
  Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01, Sm: 1.98, Eu: 1.98, Gd: 1.96,
  Tb: 1.94, Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.90, Yb: 1.87, Lu: 1.87,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36,
  Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Th: 2.06, Pa: 2.00, U: 1.96,
};

function minPairDistance(elA: string, elB: string): number {
  const rA = COVALENT_RADIUS[elA] ?? 1.4;
  const rB = COVALENT_RADIUS[elB] ?? 1.4;
  return 0.7 * (rA + rB);
}

function softValidateGeometry(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  _isRelaxed: boolean,
  pressureGPa: number = 0,
  cOverA: number = 1.0,
  gammaRad: number = Math.PI / 2,
): { valid: boolean; reason: string; warnings: string[] } {
  const warnings: string[] = [];
  if (positions.length === 0) return { valid: false, reason: "No atomic positions", warnings };
  if (positions.length > 16) return { valid: false, reason: `Too many atoms (${positions.length}), max 16 for available resources`, warnings };

  const isHighPressure = pressureGPa > 50;
  const minLattice = isHighPressure ? 2.0 : 2.5;
  if (latticeA < minLattice) return { valid: false, reason: `Lattice constant too small: ${latticeA.toFixed(2)} A (min ${minLattice})`, warnings };
  if (latticeA > 50) return { valid: false, reason: `Lattice constant too large: ${latticeA.toFixed(2)} A (max 50)`, warnings };

  const totalAtoms = positions.length;
  const sinG = Math.sin(gammaRad);
  const volumeAng3 = latticeA * latticeA * latticeA * cOverA * sinG;
  const volumePerAtom = volumeAng3 / totalAtoms;
  const hasHydrogen = positions.some(p => p.element === "H");
  const minVolPerAtom = isHighPressure ? (hasHydrogen ? 1.5 : 3.0) : (hasHydrogen ? 2.5 : 5.0);
  if (volumePerAtom < minVolPerAtom) {
    return { valid: false, reason: `Volume per atom too small: ${volumePerAtom.toFixed(1)} A^3 (min ${minVolPerAtom}${isHighPressure ? " @HP" : ""})`, warnings };
  }

  const posEls = Array.from(new Set(positions.map(p => p.element)));
  const pressureDistScale = isHighPressure ? computePressureScale(pressureGPa, posEls) : 1.0;

  let closestDist = Infinity;
  let closestPair = "";
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      let fdx = positions[i].x - positions[j].x;
      let fdy = positions[i].y - positions[j].y;
      let fdz = positions[i].z - positions[j].z;
      fdx -= Math.round(fdx);
      fdy -= Math.round(fdy);
      fdz -= Math.round(fdz);
      const dist = fracDistAngstrom(fdx, fdy, fdz, latticeA, cOverA, 1.0, gammaRad);
      const dMin = minPairDistance(positions[i].element, positions[j].element) * pressureDistScale;
      if (dist < dMin) {
        return {
          valid: false,
          reason: `Atoms ${positions[i].element}(${i}) and ${positions[j].element}(${j}) too close: ${dist.toFixed(3)} A < d_min ${dMin.toFixed(3)} A${isHighPressure ? ` (pressure-scaled @${pressureGPa}GPa)` : ` [0.7*(${COVALENT_RADIUS[positions[i].element] ?? 1.4}+${COVALENT_RADIUS[positions[j].element] ?? 1.4})]`}`,
          warnings,
        };
      }
      if (dist < closestDist) {
        closestDist = dist;
        closestPair = `${positions[i].element}-${positions[j].element}`;
      }
    }
  }

  const closestDMin = closestPair ? minPairDistance(closestPair.split("-")[0], closestPair.split("-")[1]) * pressureDistScale : 1.0;
  if (closestDist < closestDMin * 1.1) {
    warnings.push(`Tight packing: ${closestPair} at ${closestDist.toFixed(2)} A (d_min=${closestDMin.toFixed(2)}${isHighPressure ? " @HP" : ""}) — vc-relax will handle`);
  }

  return { valid: true, reason: "OK", warnings };
}

function validateGeometry(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
): { valid: boolean; reason: string } {
  const soft = softValidateGeometry(positions, latticeA, false);
  return { valid: soft.valid, reason: soft.reason };
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
  const KNOWN_SUPERHYDRIDE_PRESSURES: Record<string, number> = {
    LaH10: 150, LaH9: 120, LaH11: 200,
    YH6: 160, YH9: 200,
    CeH9: 150, CeH10: 170,
    ThH10: 175, CaH6: 200,
    ScH9: 130, BaH12: 150, LaBeH8: 100,
  };
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
    const cleanFormula = formula.replace(/\s+/g, "");
    const knownPressure = KNOWN_SUPERHYDRIDE_PRESSURES[cleanFormula];

    if (hCount > 0 && nonHAtoms > 0 && hPerMetal < 0.5 && hasHydrideMetal) {
      return { valid: false, reason: `Metal-rich hydride (H/metal=${hPerMetal.toFixed(2)}) — unphysical stoichiometry, hydrides should have H/metal >= 0.5` };
    }

    if (knownPressure) {
      return { valid: true, reason: `Known superhydride ${cleanFormula} — requires ~${knownPressure} GPa`, highPressure: true, estimatedPressureGPa: knownPressure };
    }

    if (hPerMetal > 6) {
      return { valid: true, reason: `High H/metal ratio ${hPerMetal.toFixed(1)} — tagged as high-pressure candidate (>100 GPa required)`, highPressure: true, estimatedPressureGPa: Math.min(300, 50 + hPerMetal * 15) };
    }

    if (hRatio > 0.75) {
      return { valid: true, reason: `Hydrogen fraction ${(hRatio * 100).toFixed(0)}% — tagged as high-pressure superhydride candidate`, highPressure: true, estimatedPressureGPa: Math.min(300, 100 + hRatio * 100) };
    }
  }

  for (const el of elements) {
    if (!ELEMENT_DATA[el] && !getElementData(el)) {
      return { valid: false, reason: `Unsupported element: ${el}` };
    }
  }

  return { valid: true, reason: "OK" };
}

function tryXTBPreRelaxation(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  workDir: string,
  pressureGpa: number = 0,
): Array<{ element: string; x: number; y: number; z: number }> | null {
  try {
    if (!fs.existsSync(XTB_BIN)) return null;

    const preRelaxElements = Array.from(new Set(positions.map(p => p.element)));
    const scale = pressureGpa > 0 ? computePressureScale(pressureGpa, preRelaxElements) : 1.0;
    const effectiveLattice = latticeA * scale;
    const xyzPath = path.join(workDir, "pre_relax.xyz");
    const nAtoms = positions.length;
    let xyzContent = `${nAtoms}\npre-relaxation${pressureGpa > 0 ? ` @ ${pressureGpa} GPa` : ""}\n`;
    for (const pos of positions) {
      const cx = pos.x * effectiveLattice;
      const cy = pos.y * effectiveLattice;
      const cz = pos.z * effectiveLattice;
      xyzContent += `${pos.element}  ${cx.toFixed(6)}  ${cy.toFixed(6)}  ${cz.toFixed(6)}\n`;
    }
    fs.writeFileSync(xyzPath, xyzContent);

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
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
        let fx = parseFloat(parts[1]) / effectiveLattice;
        let fy = parseFloat(parts[2]) / effectiveLattice;
        let fz = parseFloat(parts[3]) / effectiveLattice;
        fx = fx - Math.floor(fx);
        fy = fy - Math.floor(fy);
        fz = fz - Math.floor(fz);
        relaxed.push({ element: parts[0], x: fx, y: fy, z: fz });
      }
    }

    if (relaxed.length !== positions.length) {
      console.log(`[QE-Worker] xTB pre-relaxation atom count mismatch: got ${relaxed.length}, expected ${positions.length}`);
      return null;
    }

    const MAX_FRAC_DISPLACEMENT = 0.35;
    let maxDisp = 0;
    let maxDispAtom = -1;
    for (let i = 0; i < relaxed.length; i++) {
      let dx = relaxed[i].x - positions[i].x;
      let dy = relaxed[i].y - positions[i].y;
      let dz = relaxed[i].z - positions[i].z;
      dx -= Math.round(dx);
      dy -= Math.round(dy);
      dz -= Math.round(dz);
      const disp = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (disp > maxDisp) {
        maxDisp = disp;
        maxDispAtom = i;
      }
    }
    if (maxDisp > MAX_FRAC_DISPLACEMENT) {
      console.log(`[QE-Worker] xTB pre-relax rejected: atom ${maxDispAtom} (${relaxed[maxDispAtom]?.element}) displaced ${maxDisp.toFixed(3)} frac units (max ${MAX_FRAC_DISPLACEMENT}) — molecular optimizer likely collapsed structure`);
      return null;
    }

    let minDist = Infinity;
    let minPair = "";
    for (let i = 0; i < relaxed.length; i++) {
      for (let j = i + 1; j < relaxed.length; j++) {
        let fdx = relaxed[i].x - relaxed[j].x;
        let fdy = relaxed[i].y - relaxed[j].y;
        let fdz = relaxed[i].z - relaxed[j].z;
        fdx -= Math.round(fdx);
        fdy -= Math.round(fdy);
        fdz -= Math.round(fdz);
        const dist = Math.sqrt((fdx * effectiveLattice) ** 2 + (fdy * effectiveLattice) ** 2 + (fdz * effectiveLattice) ** 2);
        if (dist < minDist) {
          minDist = dist;
          minPair = `${relaxed[i].element}-${relaxed[j].element}`;
        }
      }
    }
    const ABS_MIN_DIST = 0.5;
    if (minDist < ABS_MIN_DIST) {
      console.log(`[QE-Worker] xTB pre-relax rejected: ${minPair} distance ${minDist.toFixed(3)} A < ${ABS_MIN_DIST} A absolute minimum — molecular optimizer collapsed atoms`);
      return null;
    }

    console.log(`[QE-Worker] xTB pre-relaxation succeeded for ${nAtoms} atoms (scale=${scale.toFixed(3)}, effLattice=${effectiveLattice.toFixed(3)} A, maxDisp=${maxDisp.toFixed(3)} frac, minDist=${minDist.toFixed(3)} A [${minPair}])`);
    return relaxed;
  } catch (err: any) {
    console.log(`[QE-Worker] xTB pre-relaxation failed for ${positions.length} atoms: ${err.message?.slice(0, 150)}`);
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
  params: { mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number; convThr?: string; forcConvThr?: string; etotConvThr?: string },
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  const ELEMENT_CUTOFFS2: Record<string, number> = {
    H: 100, O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
    Li: 60, Be: 60, B: 55, C: 60, Na: 60, Mg: 55, Al: 50, Si: 50,
  };
  const hasHydrogen2 = elements.includes("H");
  const rawEcutwfc = elements.reduce((max, el) => Math.max(max, ELEMENT_CUTOFFS2[el] ?? 45), hasHydrogen2 ? 80 : 45);
  const baseEcutwfc = Math.max(rawEcutwfc, hasHydrogen2 ? 100 : 60);
  const ecutwfc = baseEcutwfc + (params.ecutwfcBoost ?? 0);
  const ecutrho = ecutwfc * ecutrhoMultiplier(elements);
  const smearing = params.smearing || "mv";
  const degauss = params.degauss || 0.005;
  const convThr = params.convThr ?? "1.0d-4";
  const forcConvThr = params.forcConvThr ?? "1.0d-2";
  const etotConvThr = params.etotConvThr ?? "1.0d-4";

  let atomicSpecies = "";
  for (const el of elements) {
    const mass = getAtomicMass(el);
    if (!mass) {
      throw new Error(`Unknown element "${el}" — no atomic mass data available. Cannot generate valid QE input.`);
    }
    atomicSpecies += `  ${el}  ${mass.toFixed(3)}  ${resolvePPFilename(el)}\n`;
  }

  let atomicPositions = "";
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  const cOverA2 = estimateCOverA(elements, counts);
  const bOverA2 = estimateBOverA(elements, counts);
  const cellBlock2 = `\n${generateCellParameters(latticeA, cOverA2, 0, bOverA2, elements, counts)}`;

  return `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${formula.replace(/[^a-zA-Z0-9]/g, "")}',
  outdir = './tmp',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${forcConvThr},
  etot_conv_thr = ${etotConvThr},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = 'smearing',
  smearing = '${smearing}',
  degauss = ${degauss},
  nspin = ${elements.some(el => el in MAGNETIC_ELEMENTS) ? 2 : 1},
${elements.some(el => el in MAGNETIC_ELEMENTS) ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts)) : ''}/
&ELECTRONS
  electron_maxstep = ${params.maxSteps},
  conv_thr = ${convThr},
  mixing_beta = ${params.mixingBeta},
  mixing_mode = 'plain',
  diagonalization = '${params.diag}',
  scf_must_converge = .false.,
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${autoKPoints(latticeA, cOverA2, bOverA2)}
${cellBlock2}
`;
}

function generateVCRelaxInput(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  pressureGPa: number = 0,
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  const ELEMENT_CUTOFFS_VCR: Record<string, number> = {
    H: 100, O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
    Li: 60, Be: 60, B: 55, C: 60, Na: 60, Mg: 55, Al: 50, Si: 50,
  };
  const hasHydrogen = elements.includes("H");
  const rawEcutwfc = elements.reduce((max, el) => Math.max(max, ELEMENT_CUTOFFS_VCR[el] ?? 45), hasHydrogen ? 80 : 45);
  const ecutwfc = Math.max(rawEcutwfc, hasHydrogen ? 100 : 60);
  const ecutrho = ecutwfc * ecutrhoMultiplier(elements);

  const hasMagnetic = elements.some(el => el in MAGNETIC_ELEMENTS);
  const nspin = hasMagnetic ? 2 : 1;
  let magLines = "";
  if (hasMagnetic) {
    magLines = generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts));
  }

  let atomicSpecies = "";
  for (const el of elements) {
    const mass = getAtomicMass(el);
    if (!mass) {
      throw new Error(`Unknown element "${el}" — no atomic mass data available. Cannot generate valid QE input.`);
    }
    atomicSpecies += `  ${el}  ${mass.toFixed(3)}  ${resolvePPFilename(el)}\n`;
  }

  let atomicPositions = "";
  for (const pos of positions) {
    atomicPositions += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
  }

  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const cOverA = estimateCOverA(elements, counts);
  const bOverAVcr = estimateBOverA(elements, counts);
  const cellBlock = `\n${generateCellParameters(latticeA, cOverA, 0, bOverAVcr, elements, counts)}`;
  const hasMagneticEl = elements.some(el => el in MAGNETIC_ELEMENTS);
  const vcRelaxDegauss = hasMagneticEl ? 0.02 : 0.015;

  return `&CONTROL
  calculation = 'vc-relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}',
  outdir = './tmp',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = 1.0d-3,
  etot_conv_thr = 1.0d-5,
  nstep = 100,
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  occupations = 'smearing',
  smearing = 'mv',
  degauss = ${vcRelaxDegauss},
  nspin = ${nspin},
${magLines}/
&ELECTRONS
  electron_maxstep = 200,
  conv_thr = 1.0d-8,
  mixing_beta = 0.3,
  mixing_mode = 'plain',
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
${autoKPoints(latticeA, cOverA, bOverAVcr)}
${cellBlock}
`;
}

interface VCRelaxResult {
  converged: boolean;
  finalPositions: Array<{ element: string; x: number; y: number; z: number }> | null;
  finalLatticeBohr: number | null;
  finalLatticeAng: number | null;
  finalCellVectors: number[][] | null;
  totalEnergy: number;
  wallTimeSeconds: number;
  error: string | null;
}

function parseVCRelaxOutput(stdout: string): VCRelaxResult {
  const result: VCRelaxResult = {
    converged: false,
    finalPositions: null,
    finalLatticeBohr: null,
    finalLatticeAng: null,
    finalCellVectors: null,
    totalEnergy: 0,
    wallTimeSeconds: 0,
    error: null,
  };

  if (stdout.includes("bfgs converged")) {
    result.converged = true;
  }
  if (stdout.includes("Final enthalpy")) {
    result.converged = true;
  }

  const energyMatch = stdout.match(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/);
  if (energyMatch) {
    result.totalEnergy = parseFloat(energyMatch[1]) * RY_TO_EV;
  }

  const wallMatch = stdout.match(/WALL\s*:\s*(\d+)m?\s*([\d.]+)s/i) || stdout.match(/PWSCF\s*:\s*(?:(\d+)h)?(\d+)m\s*([\d.]+)s/i);
  if (wallMatch) {
    if (wallMatch[3]) {
      result.wallTimeSeconds = (parseInt(wallMatch[1] || "0") * 3600) + (parseInt(wallMatch[2]) * 60) + parseFloat(wallMatch[3]);
    } else {
      result.wallTimeSeconds = (parseInt(wallMatch[1] || "0") * 60) + parseFloat(wallMatch[2]);
    }
  }

  const cellLines = stdout.match(/CELL_PARAMETERS\s*\(([^)]*)\)\s*\n([\s\S]*?)(?=\n\s*\n|\nATOMIC)/);
  if (cellLines) {
    const unit = cellLines[1].toLowerCase();
    const vectors: number[][] = [];
    const lines = cellLines[2].trim().split("\n");
    for (const line of lines) {
      const parts = line.trim().split(/\s+/).map(Number);
      if (parts.length >= 3 && parts.every(v => !isNaN(v))) {
        vectors.push(parts.slice(0, 3));
      }
    }
    if (vectors.length === 3) {
      result.finalCellVectors = vectors;
      const a1 = Math.sqrt(vectors[0][0]**2 + vectors[0][1]**2 + vectors[0][2]**2);
      if (unit.includes("bohr")) {
        result.finalLatticeBohr = a1;
        result.finalLatticeAng = a1 * BOHR_TO_ANG;
      } else if (unit.includes("angstrom")) {
        result.finalLatticeAng = a1;
        result.finalLatticeBohr = a1 / BOHR_TO_ANG;
      } else {
        const celldmMatch = stdout.match(/celldm\(1\)\s*=\s*([\d.]+)/);
        if (celldmMatch) {
          const celldm1 = parseFloat(celldmMatch[1]);
          result.finalLatticeBohr = a1 * celldm1;
          result.finalLatticeAng = result.finalLatticeBohr * BOHR_TO_ANG;
        } else {
          result.finalLatticeBohr = a1;
          result.finalLatticeAng = a1 * BOHR_TO_ANG;
        }
      }
    }
  }

  let cellVectorsAng: number[][] | null = null;
  if (result.finalCellVectors && cellLines) {
    const cellUnit = cellLines[1].toLowerCase();
    if (cellUnit.includes("angstrom")) {
      cellVectorsAng = result.finalCellVectors;
    } else if (cellUnit.includes("bohr")) {
      cellVectorsAng = result.finalCellVectors.map(row => row.map(v => v * BOHR_TO_ANG));
    } else {
      const celldmMatch = stdout.match(/celldm\(1\)\s*=\s*([\d.]+)/);
      const alatAng = celldmMatch ? parseFloat(celldmMatch[1]) * BOHR_TO_ANG : (result.finalLatticeAng ?? 1.0);
      cellVectorsAng = result.finalCellVectors.map(row => row.map(v => v * alatAng));
    }
  }

  function invertCell3x3(v: number[][]): number[][] | null {
    const det = v[0][0]*(v[1][1]*v[2][2]-v[1][2]*v[2][1])
              - v[0][1]*(v[1][0]*v[2][2]-v[1][2]*v[2][0])
              + v[0][2]*(v[1][0]*v[2][1]-v[1][1]*v[2][0]);
    if (Math.abs(det) < 1e-10) return null;
    return [
      [(v[1][1]*v[2][2]-v[1][2]*v[2][1])/det, (v[0][2]*v[2][1]-v[0][1]*v[2][2])/det, (v[0][1]*v[1][2]-v[0][2]*v[1][1])/det],
      [(v[1][2]*v[2][0]-v[1][0]*v[2][2])/det, (v[0][0]*v[2][2]-v[0][2]*v[2][0])/det, (v[0][2]*v[1][0]-v[0][0]*v[1][2])/det],
      [(v[1][0]*v[2][1]-v[1][1]*v[2][0])/det, (v[0][1]*v[2][0]-v[0][0]*v[2][1])/det, (v[0][0]*v[1][1]-v[0][1]*v[1][0])/det],
    ];
  }

  const posBlocks = [...stdout.matchAll(/ATOMIC_POSITIONS\s*\{?\s*(\w+)\s*\}?\s*\n([\s\S]*?)(?=\n\s*\n|\nEnd|\nCELL|\n\s*Writing)/g)];
  if (posBlocks.length > 0) {
    const lastBlock = posBlocks[posBlocks.length - 1];
    const coordType = lastBlock[1].toLowerCase();
    const lines = lastBlock[2].trim().split("\n").filter((l: string) => l.trim().length > 0 && !l.trim().startsWith("!") && !l.trim().startsWith("#"));
    const positions: Array<{ element: string; x: number; y: number; z: number }> = [];
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 4) {
        const el = parts[0];
        let x = parseFloat(parts[1]);
        let y = parseFloat(parts[2]);
        let z = parseFloat(parts[3]);
        if (!isNaN(x) && !isNaN(y) && !isNaN(z) && el.match(/^[A-Z][a-z]?$/)) {
          if (coordType !== "crystal") {
            let posAng = [x, y, z];
            if (coordType === "bohr") {
              posAng = [x * BOHR_TO_ANG, y * BOHR_TO_ANG, z * BOHR_TO_ANG];
            } else if (coordType === "alat") {
              const celldmMatch2 = stdout.match(/celldm\(1\)\s*=\s*([\d.]+)/);
              const alatAng = celldmMatch2 ? parseFloat(celldmMatch2[1]) * BOHR_TO_ANG : (result.finalLatticeAng ?? 1.0);
              posAng = [x * alatAng, y * alatAng, z * alatAng];
            }
            const inv = cellVectorsAng ? invertCell3x3(cellVectorsAng) : null;
            if (inv) {
              x = inv[0][0]*posAng[0] + inv[0][1]*posAng[1] + inv[0][2]*posAng[2];
              y = inv[1][0]*posAng[0] + inv[1][1]*posAng[1] + inv[1][2]*posAng[2];
              z = inv[2][0]*posAng[0] + inv[2][1]*posAng[1] + inv[2][2]*posAng[2];
            } else {
              const lat = result.finalLatticeAng ?? 1.0;
              x = posAng[0] / lat;
              y = posAng[1] / lat;
              z = posAng[2] / lat;
            }
          }
          positions.push({ element: el, x, y, z });
        }
      }
    }
    if (positions.length > 0) {
      result.finalPositions = positions;
    }
  }

  return result;
}

function runQECommand(binary: string, inputFile: string, workDir: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve) => {
    // On Windows, wsl.exe does not reliably forward Node.js piped stdin to the inner
    // process. Use bash -c file-redirection instead so I/O stays within WSL.
    const wslInputFile = IS_WINDOWS ? toWslPath(inputFile) : undefined;
    const proc = spawnQE(binary, { cwd: workDir, stdio: ["pipe", "pipe", "pipe"], wslInputFile });
    let stdout = "";
    let stderr = "";
    let resolved = false;
    let stdoutEnded = false;
    let stderrEnded = false;
    let exitCode: number | null = null;

    function tryResolve() {
      if (resolved) return;
      if (stdoutEnded && stderrEnded && exitCode !== null) {
        resolved = true;
        clearTimeout(timeout);
        if (killTimeout) clearTimeout(killTimeout);
        resolve({ stdout, stderr, exitCode: exitCode ?? -1 });
      }
    }

    let inputStream: ReturnType<typeof fs.createReadStream> | null = null;
    if (!IS_WINDOWS) {
      inputStream = fs.createReadStream(inputFile);
      inputStream.on("error", (err: Error) => {
        stderr += `\nInput file error: ${err.message}`;
        try { proc.stdin?.end(); } catch {}
      });
      inputStream.pipe(proc.stdin!).on("error", () => {});
    }

    proc.stdout.on("data", (data: Buffer) => { stdout += data.toString(); });
    proc.stderr.on("data", (data: Buffer) => { stderr += data.toString(); });
    proc.stdout.on("end", () => { stdoutEnded = true; tryResolve(); });
    proc.stderr.on("end", () => { stderrEnded = true; tryResolve(); });

    let killTimeout: ReturnType<typeof setTimeout> | null = null;
    const timeout = setTimeout(() => {
      if (resolved) return;
      killProcessGracefully(proc);
      killTimeout = setTimeout(() => {
        if (resolved) return;
        resolved = true;
        try { proc.kill(); } catch {}
        try { inputStream?.destroy(); } catch {}
        resolve({ stdout, stderr: stderr + "\nTIMEOUT: QE calculation exceeded time limit", exitCode: -1 });
      }, 2000);
    }, QE_TIMEOUT_MS);

    proc.on("close", (code: number | null) => {
      exitCode = code ?? -1;
      tryResolve();
    });

    proc.on("error", (err: Error) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      if (killTimeout) clearTimeout(killTimeout);
      try { inputStream?.destroy(); } catch {}
      resolve({ stdout, stderr: stderr + `\n${err.message}`, exitCode: -1 });
    });
  });
}

export async function runFullDFT(formula: string, opts?: { startAttempt?: number }): Promise<QEFullResult> {
  const startTime = Date.now();
  const result: QEFullResult = {
    formula,
    method: "QE-PW-PBE",
    scf: null,
    phonon: null,
    bandStructure: null,
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

    result.ppValidated = true;
    const workerPressure = result.estimatedPressureGPa ?? 0;
    let latticeA = estimateLatticeConstant(elements, counts, workerPressure);
    result.initialLatticeA = latticeA;

    if (workerPressure > 0) {
      console.log(`[QE-Worker] Lattice for ${formula} Murnaghan-compressed for ${workerPressure} GPa: ${latticeA.toFixed(3)} A (B0=${estimateBulkModulus(elements).toFixed(0)} GPa)`);
    }

    let positions = generateAtomicPositions(elements, counts, formula, latticeA);
    result.initialPositions = positions.map(p => ({ ...p }));

    let protoDimensionality: string | undefined;
    try {
      const proto = selectPrototype(formula);
      if (proto) {
        result.prototypeUsed = proto.template.name;
        console.log(`[QE-Worker] Prototype matched for ${formula}: ${proto.template.name}`);
      }
    } catch (protoErr: any) {
      console.log(`[QE-Worker] Prototype selection failed for ${formula}: ${protoErr.message?.slice(0, 150) ?? "unknown error"} — using generated positions`);
    }
    try {
      const protoMatch = matchPrototype(formula);
      if (protoMatch?.dimensionality) {
        protoDimensionality = protoMatch.dimensionality;
      }
    } catch {}

    const initRepair = repairStructureGeometry(positions, latticeA, workerPressure);
    if (initRepair.repaired) {
      positions = initRepair.positions;
      latticeA = initRepair.latticeA;
      console.log(`[QE-Worker] Repaired initial geometry for ${formula} (lattice=${latticeA.toFixed(3)} A)`);
    }

    if (positions.length >= 2) {
      const snapResult = snapToWyckoffSites(positions, latticeA, elements, counts);
      if (snapResult.snapped) {
        positions = snapResult.positions;
        console.log(`[QE-Worker] Wyckoff-snapped ${snapResult.snapCount}/${positions.length} atoms for ${formula} (collision-checked, no post-snap repair needed)`);
      }
    }

    const fpCoverA = estimateCOverA(elements, counts);
    const { ibrav: fpIbrav } = determineCrystalSystem(elements, counts);
    const fpGamma = fpIbrav === 4 ? (2 * Math.PI / 3) : (Math.PI / 2);

    const relaxed = tryXTBPreRelaxation(positions, latticeA, jobDir, workerPressure);
    result.xtbPreRelaxed = !!relaxed;
    if (relaxed) {
      positions = relaxed;
      console.log(`[QE-Worker] Using xTB pre-relaxed geometry for ${formula}`);

      const postXtbGeom = softValidateGeometry(positions, latticeA, true, workerPressure, fpCoverA, fpGamma);
      if (!postXtbGeom.valid) {
        const xtbRepair = repairStructureGeometry(positions, latticeA, workerPressure);
        if (xtbRepair.repaired) {
          positions = xtbRepair.positions;
          latticeA = xtbRepair.latticeA;
          console.log(`[QE-Worker] Repaired post-xTB geometry for ${formula} (lattice=${latticeA.toFixed(3)} A)`);
          const recheck = softValidateGeometry(positions, latticeA, true, workerPressure, fpCoverA, fpGamma);
          if (!recheck.valid) {
            result.error = `Geometry rejected (post-xTB repair failed): ${recheck.reason}`;
            result.failureStage = "geometry";
            stageFailureCounts.geometry++;
            console.log(`[QE-Worker] ${formula} geometry still invalid after repair: ${recheck.reason}`);
            return result;
          }
        } else {
          result.error = `Geometry rejected (post-xTB): ${postXtbGeom.reason}`;
          result.failureStage = "geometry";
          stageFailureCounts.geometry++;
          console.log(`[QE-Worker] ${formula} geometry invalid after xTB relaxation: ${postXtbGeom.reason}`);
          return result;
        }
      }
      if (postXtbGeom.valid && postXtbGeom.warnings.length > 0) {
        console.log(`[QE-Worker] ${formula} geometry warnings (proceeding): ${postXtbGeom.warnings.join("; ")}`);
      }
    } else {
      console.log(`[QE-Worker] ${formula} xTB pre-relax failed — proceeding with raw positions to vc-relax`);
    }

    const stabilityCheck = runXTBStabilityCheck(positions, latticeA, jobDir, workerPressure);
    if (stabilityCheck && !stabilityCheck.stable) {
      result.error = `xTB stability pre-filter: ${stabilityCheck.basis}`;
      result.failureStage = "xtb_prefilter";
      result.rejectionReason = `Unstable: ${stabilityCheck.basis}`;
      stageFailureCounts.xtb_prefilter++;
      console.log(`[QE-Worker] ${formula} unstable by xTB pre-filter: ${stabilityCheck.basis}${workerPressure > 0 ? ` @ ${workerPressure} GPa` : ""}`);
      return result;
    }

    const postRelaxFingerprint = computeStructureFingerprint(positions, latticeA, fpCoverA, fpGamma);
    if (isStructureDuplicate(postRelaxFingerprint, formula)) {
      result.error = `Duplicate structure after relaxation (fingerprint=${postRelaxFingerprint.slice(0, 8)})`;
      result.failureStage = "duplicate";
      result.rejectionReason = "Duplicate structure fingerprint (post-relaxation)";
      stageFailureCounts.duplicate++;
      console.log(`[QE-Worker] ${formula} skipped: duplicate structure post-xTB (fingerprint=${postRelaxFingerprint.slice(0, 12)})`);
      return result;
    }

    const cOverA = estimateCOverA(elements, counts);
    const bOverAFull = estimateBOverA(elements, counts);
    result.kPoints = autoKPoints(latticeA, cOverA, 12, protoDimensionality).trim();

    result.vcRelaxed = false;
    const preVcLatticeA = latticeA;
    try {
      console.log(`[QE-Worker] Starting vc-relax for ${formula} (lattice=${latticeA.toFixed(2)} A, ${positions.length} atoms${workerPressure > 0 ? `, P=${workerPressure} GPa (${(workerPressure * 10).toFixed(0)} kbar)` : ""})`);
      const vcRelaxInput = generateVCRelaxInput(formula, elements, counts, latticeA, positions, workerPressure);
      const vcRelaxFile = path.join(jobDir, "vc_relax.in");
      fs.writeFileSync(vcRelaxFile, vcRelaxInput);

      const vcResult = await runQECommand(
        path.posix.join(getQEBinDir(), "pw.x"),
        vcRelaxFile,
        jobDir,
      );

      fs.writeFileSync(path.join(jobDir, "vc_relax.out"), vcResult.stdout);
      const vcParsed = parseVCRelaxOutput(vcResult.stdout);

      if (vcParsed.finalPositions && vcParsed.finalPositions.length > 0) {
        positions = vcParsed.finalPositions;
        result.vcRelaxed = true;
        if (vcParsed.finalLatticeAng && vcParsed.finalLatticeAng > 0.5) {
          latticeA = vcParsed.finalLatticeAng;
          result.relaxedLatticeA = latticeA;
          console.log(`[QE-Worker] vc-relax updated lattice for ${formula}: ${latticeA.toFixed(3)} A`);
        }
        console.log(`[QE-Worker] vc-relax ${vcParsed.converged ? "converged" : "partial"} for ${formula}: ${positions.length} atoms, E=${vcParsed.totalEnergy.toFixed(4)} eV, wall=${vcParsed.wallTimeSeconds.toFixed(0)}s`);
      } else {
        console.log(`[QE-Worker] vc-relax produced no usable positions for ${formula} (exit=${vcResult.exitCode}), proceeding with original geometry`);
      }

      cleanQETmpDir(path.join(jobDir, "tmp"));
    } catch (vcErr: any) {
      console.log(`[QE-Worker] vc-relax failed for ${formula}: ${(vcErr.message || "").slice(-200)}, proceeding with original geometry`);
      cleanQETmpDir(path.join(jobDir, "tmp"));
    }

    result.kPoints = autoKPoints(latticeA, cOverA, bOverAFull).trim();
    if (Math.abs(latticeA - preVcLatticeA) > 0.01) {
      console.log(`[QE-Worker] K-points recomputed for ${formula} after vc-relax lattice change (${preVcLatticeA.toFixed(3)} -> ${latticeA.toFixed(3)} A): ${result.kPoints}`);
    }

    const retryConfigs: Array<{ mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number; convThr?: string; forcConvThr?: string; etotConvThr?: string }> = [
      { mixingBeta: 0.3, maxSteps: 300, diag: "david", convThr: "1.0d-10", forcConvThr: "1.0d-3", etotConvThr: "1.0d-6" },
      { mixingBeta: 0.2, maxSteps: 400, diag: "david", ecutwfcBoost: 10, convThr: "1.0d-10", forcConvThr: "1.0d-3", etotConvThr: "1.0d-6" },
      { mixingBeta: 0.15, maxSteps: 500, diag: "cg", ecutwfcBoost: 15, convThr: "1.0d-8", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5" },
      { mixingBeta: 0.1, maxSteps: 500, diag: "cg", smearing: "mp", degauss: 0.003, ecutwfcBoost: 20, convThr: "1.0d-6", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4" },
      { mixingBeta: 0.07, maxSteps: 800, diag: "cg", smearing: "mp", degauss: 0.001, ecutwfcBoost: 25, convThr: "1.0d-5", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4" },
    ];

    const firstAttempt = opts?.startAttempt ?? 0;
    let scfConverged = false;
    let retryCount = 0;

    for (let attempt = firstAttempt; attempt < retryConfigs.length && !scfConverged; attempt++) {
      const params = retryConfigs[attempt];
      const scfInput = generateSCFInputWithParams(formula, elements, counts, latticeA, positions, params);
      const scfInputFile = path.join(jobDir, `scf_attempt${attempt}.in`);
      fs.writeFileSync(scfInputFile, scfInput);

      if (attempt > 0) {
        cleanQETmpDir(path.join(jobDir, "tmp"));
      }

      const smearInfo = params.smearing ? `, smearing=${params.smearing}` : "";
      const convInfo = params.convThr ? `, conv_thr=${params.convThr}` : "";
      console.log(`[QE-Worker] SCF attempt ${attempt + 1}/${retryConfigs.length} for ${formula} (a=${latticeA.toFixed(2)} A, beta=${params.mixingBeta}, diag=${params.diag}, maxstep=${params.maxSteps}${convInfo}${smearInfo})`);

      const scfResult = await runQECommand(
        path.posix.join(getQEBinDir(), "pw.x"),
        scfInputFile,
        jobDir,
      );

      fs.writeFileSync(path.join(jobDir, `scf_attempt${attempt}.out`), scfResult.stdout);
      const usedDegauss = params.degauss || 0.005;
      result.scf = parseSCFOutput(scfResult.stdout, usedDegauss);

      if (scfResult.exitCode !== 0 && !result.scf.converged) {
        const isPPError = scfResult.stderr.includes("read_upf") || scfResult.stderr.includes("readpp") || scfResult.stderr.includes("EOF marker");
        if (isPPError) {
          result.scf.error = `Pseudopotential read failure: ${scfResult.stderr.slice(-300)}`;
          console.log(`[QE-Worker] PP error for ${formula}, no retry will help — skipping`);
          recordFormulaFailure(formula);
          break;
        }
        result.scf.error = `pw.x exited with code ${scfResult.exitCode}: ${scfResult.stderr.slice(-500)}`;
        console.log(`[QE-Worker] SCF attempt ${attempt + 1} failed for ${formula}: ${result.scf.error.slice(-200)}`);
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

    const scfUsable = scfConverged ||
      (result.scf && result.scf.totalEnergy !== 0 && result.scf.fermiEnergy !== null &&
       result.scf.lastScfAccuracyRy !== null && result.scf.lastScfAccuracyRy < 1.0e-6);

    if (!scfConverged && !scfUsable) {
      result.failureStage = "scf";
      stageFailureCounts.scf++;
      recordFormulaFailure(formula);
      if (result.scf && result.scf.totalEnergy !== 0 && result.scf.lastScfAccuracyRy !== null) {
        console.log(`[QE-Worker] SCF not converged for ${formula}: accuracy=${result.scf.lastScfAccuracyRy.toExponential(2)} Ry (threshold 1e-6 Ry for phonon safety) — discarding non-physical energy`);
      }
    } else if (!scfConverged && scfUsable) {
      console.log(`[QE-Worker] SCF near-converged for ${formula} (accuracy=${result.scf!.lastScfAccuracyRy?.toExponential(2)} Ry < 1e-6 Ry, quality=${result.scf!.convergenceQuality}): E=${result.scf!.totalEnergy.toFixed(4)} eV, Ef=${result.scf!.fermiEnergy} — proceeding with caution`);
    }

    if (scfUsable && result.scf?.fermiEnergy !== null) {
      try {
        const cOverAVal = estimateCOverA(elements, counts);
        const ELEMENT_CUTOFFS_BANDS: Record<string, number> = {
          H: 100, O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
          Li: 60, Be: 60, B: 55, C: 60, Na: 60, Mg: 55, Al: 50, Si: 50,
        };
        const hasHydrogenBands = elements.includes("H");
        const rawEcutwfcBands = elements.reduce((max, el) => Math.max(max, ELEMENT_CUTOFFS_BANDS[el] ?? 45), hasHydrogenBands ? 80 : 45);
        const baseEcutwfcBands = Math.max(rawEcutwfcBands, hasHydrogenBands ? 100 : 60);
        const nspinBands = elements.some(el => el in MAGNETIC_ELEMENTS) ? 2 : 1;

        const bOverAVal = estimateBOverA(elements, counts);
        const latticeBVal = latticeA * bOverAVal;

        const bandResult = await computeDFTBandStructure(
          formula,
          elements,
          counts,
          latticeA,
          positions,
          result.scf.fermiEnergy,
          jobDir,
          cOverAVal,
          baseEcutwfcBands,
          nspinBands,
          latticeBVal,
        );

        result.bandStructure = bandResult;
        recordBandCalcOutcome(bandResult.converged, bandResult.wallTimeSeconds);

        if (!bandResult.converged && bandResult.error) {
          stageFailureCounts.bands++;
          console.log(`[QE-Worker] Band structure failed for ${formula}: ${bandResult.error.slice(-200)} — cleaning tmp before phonon`);
          cleanQETmpDir(path.join(jobDir, "tmp"));
        } else {
          console.log(`[QE-Worker] Band structure done for ${formula}: ${bandResult.nBands} bands, ${bandResult.bandCrossings.length} crossings, flat=${bandResult.flatBandScore.toFixed(3)}`);
        }
      } catch (bandErr: any) {
        console.log(`[QE-Worker] Band structure error for ${formula}: ${bandErr.message?.slice(-200) ?? bandErr} — cleaning tmp before phonon`);
        stageFailureCounts.bands++;
        cleanQETmpDir(path.join(jobDir, "tmp"));
      }
    }

    if (scfUsable) {
      const phInput = generatePhononInput(formula, elements);
      const phInputFile = path.join(jobDir, "ph.in");
      fs.writeFileSync(phInputFile, phInput);

      console.log(`[QE-Worker] Starting phonon calculation for ${formula}`);

      const phResult = await runQECommand(
        path.posix.join(getQEBinDir(), "ph.x"),
        phInputFile,
        jobDir,
      );

      fs.writeFileSync(path.join(jobDir, "ph.out"), phResult.stdout);
      result.phonon = parsePhononOutput(phResult.stdout);

      if (phResult.exitCode !== 0 && !result.phonon.converged) {
        result.phonon.error = `ph.x exited with code ${phResult.exitCode}: ${phResult.stderr.slice(-500)}`;
        result.failureStage = "phonon";
        stageFailureCounts.phonon++;
        console.log(`[QE-Worker] Phonon failed for ${formula}: ${result.phonon.error.slice(-200)}`);
      } else {
        console.log(`[QE-Worker] Phonon done for ${formula}: ${result.phonon.frequencies.length} modes, lowest=${result.phonon.lowestFrequency.toFixed(1)} cm-1`);
      }
    }
  } catch (err: any) {
    result.error = err.message;
    console.log(`[QE-Worker] Error for ${formula}: ${err.message}`);
  } finally {
    for (let cleanAttempt = 0; cleanAttempt < 3; cleanAttempt++) {
      try {
        fs.rmSync(jobDir, { recursive: true, force: true });
        break;
      } catch (rmErr: any) {
        if (cleanAttempt < 2 && rmErr.code === "EBUSY") {
          await new Promise(r => setTimeout(r, 500 * (cleanAttempt + 1)));
        }
      }
    }
  }

  result.wallTimeTotal = (Date.now() - startTime) / 1000;
  return result;
}

// Cached QE availability — probed in background to avoid blocking event loop.
// Default false; updated by scheduleQEAvailabilityProbe() called from startEngine().
let _qeAvailable = false;

export function isQEAvailable(): boolean {
  return _qeAvailable; // always fast — never runs execSync
}

// Call once from startEngine() with a safe delay (e.g. 90s) to probe after startup settles.
// The execSync WSL probes (3 candidates × 5s timeout) run inside the timer callback —
// they block the event loop once but only after the critical startup window has passed.
export function scheduleQEAvailabilityProbe(delayMs = 90_000): void {
  // Use async exec to avoid blocking the event loop during the WSL probe.
  setTimeout(async () => {
    try {
      if (IS_WINDOWS) {
        const candidates = [getQEBinDir(), "/usr/bin", "/usr/local/bin"].filter((v, i, a) => Boolean(v) && a.indexOf(v) === i);
        for (const dir of candidates) {
          try {
            const { stdout } = await execFileAsync("wsl.exe", ["-d", "Ubuntu", "--", "bash", "-c", `test -f '${dir}/pw.x' && echo yes || echo no`], { timeout: 5000 });
            if (stdout.trim().replace(/\r/g, "") === "yes") { _qeAvailable = true; return; }
          } catch { /* not found in this dir */ }
        }
        _qeAvailable = false;
      } else {
        const pwx = path.join(getQEBinDir(), "pw.x");
        _qeAvailable = fs.existsSync(pwx);
      }
    } catch { _qeAvailable = false; }
  }, delayMs);
}
