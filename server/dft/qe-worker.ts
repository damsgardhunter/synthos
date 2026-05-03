import { execSync, execFile } from "child_process";
import { promisify } from "util";
const execFileAsync = promisify(execFile);
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { IS_WINDOWS, binaryPath, getTempSubdir, toWslPath, killProcessGracefully, spawnQE } from "./platform-utils";
import { selectPrototype } from "../learning/crystal-prototypes";
import { matchPrototype } from "../learning/structure-predictor";
import { isTransitionMetal, isRareEarth, ELEMENTAL_DATA, getElementData, getHubbardU } from "../learning/elemental-data";
import { estimateCorrelationEffects } from "../physics/correlation-engine";
import { generatePrototypeFreeStructure } from "../crystal/lattice-generator";
import { getAllDistributions, getElementSitePreference, type CrystalSystemDistribution } from "../ai/crystal-distribution-db";
import { computeDFTBandStructure, recordBandCalcOutcome, type DFTBandStructureResult } from "./band-structure-calculator";
import {
  generatePhononGridInput,
  generateQ2RInput,
  generateMatdynDOSInput,
  parseLambdaOutput,
  tryLoadDFPTResults,
} from "./dfpt-parser";
import { runEliashbergFromAlpha2FFile } from "../physics/eliashberg-pipeline";
import {
  computeElectronicStructure,
  computeElectronPhononCoupling,
  computePhononSpectrum,
  type ElectronPhononCoupling,
} from "../learning/physics-engine";
import { generateStructureCandidates, vegardEstimate, type StructureCandidate, type VegardEstimate } from "./vegard-lattice";
import { lookupKnownStructure, getKnownStructureFormulas } from "../learning/known-structures";
import { airssEngine } from "../csp/airss-wrapper";
import { pyxtalEngine } from "../csp/pyxtal-wrapper";
import { mutateTopCandidates } from "../csp/structure-mutator";
import { generateCageSeededCandidates } from "../csp/cage-seeder";
import { logCandidateStats } from "../csp/candidate-metadata";
import { runCandidateFunnel } from "../csp/candidate-funnel";
import { assignTier, logTierDecision } from "../csp/tier-assignment";
import type { CSPCandidate } from "../csp/csp-types";
import { generateRound2Candidates, screenRound2, shouldDoRound2 } from "../csp/iterative-search";
import { recordGeneratorOutcome, recordVolumeOutcome, getSignalWeight, classifyFamily, classifyPressureBin } from "../csp/adaptive-learning";
import {
  runStagedRelaxation,
  runStage4GammaPhonon,
  type QERunnerCallbacks,
  type StagedRelaxationResult,
  type StageResult,
} from "./staged-relaxation";

// Resolve the QE binary directory lazily (on first DFT call).
// Running execSync WSL probes at module-load time would block the Node.js event loop
// for up to 10s during server startup — so we defer to first use instead.
// On Windows (WSL2): probe for conda-forge QE 7.x install first (~/miniforge3/bin),
// then fall back to apt install (/usr/bin).
// On Linux/production: use Nix store path or QE_BIN_DIR env var.
// Search /nix/store for any quantum-espresso installation (hash changes per version/rebuild).
function findNixQEBins(): string[] {
  try {
    if (!fs.existsSync("/nix/store")) return [];
    return fs.readdirSync("/nix/store")
      .filter(e => e.includes("quantum-espresso"))
      .map(e => `/nix/store/${e}/bin`)
      .filter(d => {
        try { return fs.existsSync(path.join(d, "pw.x")); } catch { return false; }
      });
  } catch { return []; }
}

function resolveQEBinDir(): string {
  if (process.env.QE_BIN_DIR) return process.env.QE_BIN_DIR;
  if (!IS_WINDOWS) {
    // Try Nix store (glob-based — hash changes per QE version/rebuild), then apt/conda/custom installs.
    const candidates = [
      ...findNixQEBins(),
      // System package managers (apt, yum, dnf)
      "/usr/bin",
      "/usr/local/bin",
      // Conda/mamba installs (root or user)
      "/opt/conda/bin",
      "/opt/miniconda3/bin",
      "/opt/miniforge3/bin",
      "/root/miniforge3/bin",
      "/root/miniconda3/bin",
      // Common manual install prefixes on GCP/HPC
      "/opt/quantum-espresso/bin",
      "/opt/qe/bin",
      "/opt/espresso/bin",
    ];
    for (const dir of candidates) {
      if (dir && fs.existsSync(path.join(dir, "pw.x"))) return dir;
    }
    // Return a meaningful fallback that will fail with ENOENT (not a misleading path)
    return "/usr/bin";
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
// Complex hydrides (LaH10, CeH10, ReRuH6) routinely need 60-90 min on GCP.
// Default 90 min; override via QE_TIMEOUT_MS env var (e.g. 7200000 for 2 h).
const QE_TIMEOUT_MS = parseInt(process.env.QE_TIMEOUT_MS ?? "5400000", 10);
// QE graceful-stop margin: QE writes output and exits cleanly 120s before Node kills it.
// The 60s margin was too tight — QE sometimes needs extra time to flush large outputs.
const QE_MAX_SECONDS = Math.floor(QE_TIMEOUT_MS / 1000) - 120;

const PROJECT_ROOT = path.resolve(process.cwd());
const PP_SOURCE_DIR = path.join(PROJECT_ROOT, "server/dft/pseudo");
// XTB_BIN: set XTB_BIN=/usr/bin/xtb on GCP (xtb-dist/ is gitignored and not deployed there)
const XTB_BIN = binaryPath(process.env.XTB_BIN ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb"));
const XTB_HOME = process.env.XTBHOME ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = process.env.XTBPATH ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");

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
  convergenceQuality: "strict" | "loose" | "partial-walltime" | "none";
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

export interface QEDFPTResult {
  lambda: number;
  omegaLog: number;        // cm⁻¹ (log-average phonon frequency)
  tcAllenDynes: number;    // K via Allen-Dynes
  tcEliashberg: number;    // K via full Eliashberg gap equation
  tcBest: number;          // K — best estimate (max of Allen-Dynes and Eliashberg)
  nqGrid: [number, number, number];
  phConverged: boolean;
  q2rDone: boolean;
  matdynDone: boolean;
  wallTimeSeconds: number;
  source: "ph.x-stdout" | "a2F-file" | "none";
  warnings: string[];
  /** Method provenance for electron-phonon coupling — only "dfpt_eph" is physics-grade. */
  alpha2FMethod?: "dfpt_eph" | "surrogate_eph" | "heuristic_eph" | "unavailable";
  /** Method provenance for lambda — only "dfpt_integrated_alpha2F" is physics-grade. */
  lambdaMethod?: "dfpt_integrated_alpha2F" | "surrogate_alpha2F" | "estimated_from_dos_phonons";
}

export interface QEFullResult {
  formula: string;
  method: "QE-PW-PBE";
  scf: QESCFResult | null;
  phonon: QEPhononResult | null;
  bandStructure: DFTBandStructureResult | null;
  dfpt?: QEDFPTResult;
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
  qeDFTPlusU?: boolean;
  dftPlusUTcModifier?: number;
  // Staged relaxation pipeline results
  vegardEstimate?: { latticeA: number; confidence: number; endpointsUsed: string[]; method: string };
  structureCandidatesEvaluated?: number;
  stagedRelaxation?: {
    stages: Array<{
      stage: number;
      passed: boolean;
      failReason?: string;
      totalEnergy: number;
      maxForce?: number;
      wallTimeSeconds: number;
    }>;
    finalStage: number;
    success: boolean;
    totalWallTime: number;
  };
  gammaPhononPassed?: boolean;
  /** Quality tier — determines what downstream analysis is trustworthy. */
  qualityTier?: "failed" | "partial_screening" | "screening_converged" | "relaxed" | "final_converged" | "publication_ready";
  /** CSP provenance: which generator, Z, volume, cluster, DFT rank. */
  provenance?: {
    generator: string;
    zValue?: number;
    volumeMultiplier?: number;
    clusterId?: string;
    funnelTier: string;
    dft0Rank?: number;
    selectionCategory?: string;
  };
  /** Whether the DFT quality gate passed before Tc estimation. */
  qualityGatePassed?: boolean;
  qualityGateReasons?: string[];
  /** Convex hull stability assessment. */
  hullStability?: {
    hullDistanceMeVAtom: number;
    label: "on_hull" | "near_hull" | "metastable" | "highly_metastable" | "unknown_hull";
    decompositionProducts?: string[];
    computedFromDFT: boolean;
  };
  /** Uncertainty and confidence for all final results. */
  uncertainty?: {
    tcConfidence: "high" | "medium" | "low" | "surrogate";
    tcUncertaintyReason: string;
    lambdaConfidence: "high" | "medium" | "low" | "surrogate";
    phononConfidence: "high" | "medium" | "low" | "none";
    structureConfidence: "high" | "medium" | "low";
    ephMethod: "dfpt" | "surrogate" | "none";
    phononMethod: "dfpt_full" | "dfpt_gamma" | "finite_displacement" | "surrogate" | "none";
  };
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
  // USPP requires 8x cutoff due to augmentation charges.
  // PAW and NC only need 4x — using 8x for PAW causes FFT grid memory overflow
  // for large hydrides like LaH10 (ecutrho = 100 * 8 = 800 Ry → OOM → code 6 crash).
  for (const el of elements) {
    const ppType = detectPPType(el);
    if (ppType === "uspp") return 8;
  }
  return 4;
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
): { stable: boolean; ePerAtom: number; basis: string; formationEnergyEv?: number; confidencePenalty: number } | null {
  // xTB is parameterised for ambient conditions — compressed interatomic distances
  // at high pressure fall outside its valid range and produce spuriously large
  // repulsive energies (~20-30 eV/atom) that would incorrectly reject valid
  // high-pressure superconductor candidates. Skip the filter above 50 GPa.
  if (pressureGpa > 50) return null;
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
    let formationEnergyEv: number | undefined;
    let confidencePenalty = 0;
    if (hasRef) {
      const formationLike = (ePerAtomHa - refEPerAtom) * HA_TO_EV;
      formationEnergyEv = formationLike;
      // Real formation-like energies live within ±10 eV/atom. Values outside
      // that window mean xTB didn't produce a trustworthy number (SCF non-
      // convergence on heavy-element/high-pressure cells, bad atomic ref, etc).
      // Treat as "no pre-filter opinion" so QE gets to decide.
      if (!isFinite(formationLike) || Math.abs(formationLike) > 10.0) {
        return null;
      }
      // Soft penalty instead of hard rejection:
      // - < 1.0 eV/atom: stable, no penalty
      // - 1.0–2.0 eV/atom: mildly unstable, small confidence penalty
      // - 2.0–2.5 eV/atom: unstable, larger penalty but still proceeds
      // - > 2.5 eV/atom: very unstable, marked but still not hard-rejected
      //   (hard reject only if ALSO bad geometry AND no prototype support)
      if (formationLike < 1.0) {
        isStable = true;
        confidencePenalty = 0;
      } else if (formationLike < 2.0) {
        isStable = true;
        confidencePenalty = 0.15;
      } else if (formationLike < 2.5) {
        isStable = true;
        confidencePenalty = 0.30;
      } else {
        // > 2.5 eV/atom — very unstable by xTB, but xTB is unreliable for
        // high-P hydrides and unusual compositions. Mark as unstable with
        // heavy penalty, but let high-confidence candidates through.
        isStable = false;
        confidencePenalty = 0.50;
      }
      basis = `relative (Ef-like=${formationLike.toFixed(3)} eV/atom, penalty=${confidencePenalty.toFixed(2)})`;
    } else {
      isStable = ePerAtomEv < -1.0;
      confidencePenalty = isStable ? 0 : 0.30;
      basis = `absolute (E/atom=${ePerAtomEv.toFixed(3)} eV)`;
    }

    return { stable: isStable, ePerAtom: ePerAtomEv, basis, formationEnergyEv, confidencePenalty };
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
    // UPF v1 files end with </PP_RHOATOM>; UPF v2 files end with </UPF>
    const tailStr = tail.toString("utf-8");
    if (!tailStr.includes("</UPF>") && !tailStr.includes("</PP_RHOATOM>")) return false;
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

// Elements whose best-available PPs include f-orbital projectors that exceed
// QE's compile-time lmaxx=3 limit. Production DB shows 75 jobs failing with
// "momentum in pseudopotentials (lmaxx) = 3" on these. Reject upfront until
// either (a) QE is rebuilt with a higher lmaxx, or (b) scalar-relativistic
// PPs with lmax<=2 are installed for each.
const LMAXX_INCOMPATIBLE: Set<string> = new Set([
  // Lanthanides with 4f projectors
  "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
  // Actinides with 5f projectors
  "Th", "Pa", "U", "Np", "Pu", "Am",
]);

// Elements whose PP fetch has failed in this process lifetime. Populated by
// ensurePseudopotential() catches; queried by the pre-flight element gate so
// subsequent candidates containing the same element fail in ms instead of
// retrying the (failing) download path every time.
const PP_FAILED_ELEMENTS = new Map<string, number>(); // element -> timestamp ms
const PP_FAILURE_COOLDOWN_MS = 60 * 60 * 1000; // 1 hour

function markElementPPFailed(element: string): void {
  PP_FAILED_ELEMENTS.set(element, Date.now());
}

function getUnsupportedElement(elements: string[]): string | null {
  for (const el of elements) {
    if (LMAXX_INCOMPATIBLE.has(el)) return el;
    const failedAt = PP_FAILED_ELEMENTS.get(el);
    if (failedAt !== undefined && Date.now() - failedAt < PP_FAILURE_COOLDOWN_MS) {
      return el;
    }
  }
  return null;
}

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
      // Threshold: 0.5 to allow GBRV-style PPs where outer d+s electrons only
    // (e.g. Ir GBRV has 9 vs pslibrary 17 — 9/17=0.53 passes, 5/17=0.29 fails)
    if (ppZVal > 0 && expectedZ > 0 && ppZVal < expectedZ * 0.5) {
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
        } else if (
          entry.endsWith(".xml") ||
          entry.endsWith(".restart_xml") ||
          /\.(wfc|mix)\d*(_new)?$/.test(entry) // .wfc, .wfc1, .wfc2, .mix, .mix1, .mix1_new, etc.
        ) {
          fs.unlinkSync(fullPath);
        }
      } catch {}
    }
  } catch {}
}

// Remove jobDirs left behind by previous crashed server runs. Called once at startup.
function cleanStaleQEJobDirs(): void {
  if (!fs.existsSync(QE_WORK_DIR)) return;
  const staleAgeMs = 2 * 60 * 60 * 1000; // 2 hours
  const now = Date.now();
  try {
    const entries = fs.readdirSync(QE_WORK_DIR);
    for (const entry of entries) {
      if (!entry.startsWith("job_")) continue;
      const fullPath = path.join(QE_WORK_DIR, entry);
      try {
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory() && now - stat.mtimeMs > staleAgeMs) {
          fs.rmSync(fullPath, { recursive: true, force: true });
          console.log(`[QE-Worker] Cleaned stale job dir: ${entry}`);
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
cleanStaleQEJobDirs();

// GitHub pslibrary is the primary source — the QE website is often unreliable/down.
// QE website kept as fallback only.
// GBRV (Garrity-Bennett-Rabe-Vanderbilt) ultrasoft PPs from Rutgers as tertiary source.
const GH_BASE = "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS";
const QE_BASE = "https://pseudopotentials.quantum-espresso.org/upf_files";
const GBRV_BASE = "https://www.physics.rutgers.edu/gbrv/pbe";

const PP_DOWNLOAD_URLS: Record<string, string> = {
  H:  `${GH_BASE}/H.pbe-kjpaw_psl.1.0.0.UPF`,
  Li: `${GH_BASE}/Li.pbe-s-kjpaw_psl.1.0.0.UPF`,
  Be: `${GH_BASE}/Be.pbe-n-kjpaw_psl.1.0.0.UPF`,
  B:  `${GH_BASE}/B.pbe-n-kjpaw_psl.1.0.0.UPF`,
  C:  `${GH_BASE}/C.pbe-n-kjpaw_psl.1.0.0.UPF`,
  N:  `${GH_BASE}/N.pbe-n-kjpaw_psl.1.0.0.UPF`,
  O:  `${GH_BASE}/O.pbe-n-kjpaw_psl.1.0.0.UPF`,
  F:  `${GH_BASE}/F.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Na: `${GH_BASE}/Na.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Mg: `${GH_BASE}/Mg.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Al: `${GH_BASE}/Al.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Si: `${GH_BASE}/Si.pbe-n-kjpaw_psl.1.0.0.UPF`,
  P:  `${GH_BASE}/P.pbe-n-kjpaw_psl.1.0.0.UPF`,
  S:  `${GH_BASE}/S.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Cl: `${GH_BASE}/Cl.pbe-n-kjpaw_psl.1.0.0.UPF`,
  K:  `${GH_BASE}/K.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ca: `${GH_BASE}/Ca.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Sc: `${GH_BASE}/Sc.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ti: `${GH_BASE}/Ti.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  V:  `${GH_BASE}/V.pbe-spnl-kjpaw_psl.1.0.0.UPF`,
  Cr: `${GH_BASE}/Cr.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Mn: `${GH_BASE}/Mn.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Fe: `${GH_BASE}/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Co: `${GH_BASE}/Co.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ni: `${GH_BASE}/Ni.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Cu: `${GH_BASE}/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Zn: `${GH_BASE}/Zn.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Ga: `${GH_BASE}/Ga.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Ge: `${GH_BASE}/Ge.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  As: `${GH_BASE}/As.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Se: `${GH_BASE}/Se.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Br: `${GH_BASE}/Br.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Rb: `${GH_BASE}/Rb.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Sr: `${GH_BASE}/Sr.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Y:  `${GH_BASE}/Y.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Zr: `${GH_BASE}/Zr.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Nb: `${GH_BASE}/Nb.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Mo: `${GH_BASE}/Mo.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Tc: `${GH_BASE}/Tc.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ru: `${GH_BASE}/Ru.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Rh: `${GH_BASE}/Rh.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Pd: `${GH_BASE}/Pd.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ag: `${GH_BASE}/Ag.pbe-nd-kjpaw_psl.1.0.0.UPF`,
  Cd: `${GH_BASE}/Cd.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  In: `${GH_BASE}/In.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Sn: `${GH_BASE}/Sn.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Sb: `${GH_BASE}/Sb.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Te: `${GH_BASE}/Te.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  I:  `${GH_BASE}/I.pbe-n-kjpaw_psl.0.2.UPF`,
  Cs: `${GH_BASE}/Cs.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ba: `${GH_BASE}/Ba.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  La: `${GH_BASE}/La.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
  Ce: `${GH_BASE}/Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Pr: `${GH_BASE}/Pr.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Nd: `${GH_BASE}/Nd.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Sm: `${GH_BASE}/Sm.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Eu: `${GH_BASE}/Eu.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Gd: `${GH_BASE}/Gd.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Tb: `${GH_BASE}/Tb.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Dy: `${GH_BASE}/Dy.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Ho: `${GH_BASE}/Ho.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Er: `${GH_BASE}/Er.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Tm: `${GH_BASE}/Tm.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Yb: `${GH_BASE}/Yb.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Lu: `${GH_BASE}/Lu.pbe-spdfn-kjpaw_psl.1.0.0.UPF`,
  Hf: `${GH_BASE}/Hf.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ta: `${GH_BASE}/Ta.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  W:  `${GH_BASE}/W.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Re: `${GH_BASE}/Re.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Os: `${GH_BASE}/Os.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ir: `${GH_BASE}/Ir.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Pt: `${GH_BASE}/Pt.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Au: `${GH_BASE}/Au.pbe-nd-kjpaw_psl.1.0.0.UPF`,
  Hg: `${GH_BASE}/Hg.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Tl: `${GH_BASE}/Tl.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Pb: `${GH_BASE}/Pb.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Bi: `${GH_BASE}/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Th: `${GH_BASE}/Th.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
  U:  `${GH_BASE}/U.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
};

// GBRV Vanderbilt ultrasoft PPs — tertiary source for elements where QE website fails.
// Filenames use lowercase symbols. These are USPP (pseudo_type="US"), z_valence
// for 5d metals may be smaller (outer d+s only) — threshold lowered to 0.5 above.
const PP_GBRV_URLS: Record<string, string> = {
  Ag: `${GBRV_BASE}/ag_pbe_v1.4.uspp.F.UPF`,
  Cd: `${GBRV_BASE}/cd_pbe_v1.uspp.F.UPF`,
  Te: `${GBRV_BASE}/te_pbe_v1.uspp.F.UPF`,
  Rh: `${GBRV_BASE}/rh_pbe_v1.4.uspp.F.UPF`,
  Pd: `${GBRV_BASE}/pd_pbe_v1.4.uspp.F.UPF`,
  Ir: `${GBRV_BASE}/ir_pbe_v1.2.uspp.F.UPF`,
  Pt: `${GBRV_BASE}/pt_pbe_v1.4.uspp.F.UPF`,
  Au: `${GBRV_BASE}/au_pbe_v1.uspp.F.UPF`,
  Mg: `${GBRV_BASE}/mg_pbe_v1.4.uspp.F.UPF`,
  Hg: `${GBRV_BASE}/hg_pbe_v1.uspp.F.UPF`,
  Co: `${GBRV_BASE}/co_pbe_v1.2.uspp.F.UPF`,
  Zn: `${GBRV_BASE}/zn_pbe_v1.uspp.F.UPF`,
  Tc: `${GBRV_BASE}/tc_pbe_v1.uspp.F.UPF`,
  // Lanthanides
  Yb: `${GBRV_BASE}/yb_pbe_v1.uspp.F.UPF`,
  Pr: `${GBRV_BASE}/pr_pbe_v1.uspp.F.UPF`,
  Nd: `${GBRV_BASE}/nd_pbe_v1.uspp.F.UPF`,
  Sm: `${GBRV_BASE}/sm_pbe_v1.uspp.F.UPF`,
  Eu: `${GBRV_BASE}/eu_pbe_v1.uspp.F.UPF`,
  Gd: `${GBRV_BASE}/gd_pbe_v1.uspp.F.UPF`,
  Tb: `${GBRV_BASE}/tb_pbe_v1.uspp.F.UPF`,
  Dy: `${GBRV_BASE}/dy_pbe_v1.uspp.F.UPF`,
  Ho: `${GBRV_BASE}/ho_pbe_v1.uspp.F.UPF`,
  Er: `${GBRV_BASE}/er_pbe_v1.uspp.F.UPF`,
  Tm: `${GBRV_BASE}/tm_pbe_v1.uspp.F.UPF`,
  Lu: `${GBRV_BASE}/lu_pbe_v1.uspp.F.UPF`,
};

// QE website as fallback (often unreliable)
const PP_FALLBACK_URLS: Record<string, string> = {
  H:  `${QE_BASE}/H.pbe-kjpaw_psl.1.0.0.UPF`,
  Li: `${QE_BASE}/Li.pbe-s-kjpaw_psl.1.0.0.UPF`,
  Be: `${QE_BASE}/Be.pbe-n-kjpaw_psl.1.0.0.UPF`,
  B:  `${QE_BASE}/B.pbe-n-kjpaw_psl.1.0.0.UPF`,
  C:  `${QE_BASE}/C.pbe-n-kjpaw_psl.1.0.0.UPF`,
  N:  `${QE_BASE}/N.pbe-n-kjpaw_psl.1.0.0.UPF`,
  O:  `${QE_BASE}/O.pbe-n-kjpaw_psl.1.0.0.UPF`,
  F:  `${QE_BASE}/F.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Na: `${QE_BASE}/Na.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Mg: `${QE_BASE}/Mg.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Al: `${QE_BASE}/Al.pbe-n-kjpaw_psl.1.0.0.UPF`,
  Si: `${QE_BASE}/Si.pbe-n-kjpaw_psl.1.0.0.UPF`,
  P:  `${QE_BASE}/P.pbe-n-kjpaw_psl.1.0.0.UPF`,
  S:  `${QE_BASE}/S.pbe-n-kjpaw_psl.1.0.0.UPF`,
  K:  `${QE_BASE}/K.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ca: `${QE_BASE}/Ca.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Ti: `${QE_BASE}/Ti.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Fe: `${QE_BASE}/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  Cu: `${QE_BASE}/Cu.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Nb: `${QE_BASE}/Nb.pbe-spn-kjpaw_psl.1.0.0.UPF`,
  La: `${QE_BASE}/La.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
  Ce: `${QE_BASE}/Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Pb: `${QE_BASE}/Pb.pbe-dn-kjpaw_psl.1.0.0.UPF`,
  Pt: `${QE_BASE}/Pt.pbe-spdn-kjpaw_psl.1.0.0.UPF`,
  Au: `${QE_BASE}/Au.pbe-nd-kjpaw_psl.1.0.0.UPF`,
  Th: `${QE_BASE}/Th.pbe-spfn-kjpaw_psl.1.0.0.UPF`,
};

async function downloadPPToTemp(url: string, tmpFile: string): Promise<boolean> {
  // Try Node.js fetch first — works on GCP where curl may be blocked or misconfigured
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(30000) });
    if (res.ok) {
      const buf = Buffer.from(await res.arrayBuffer());
      fs.writeFileSync(tmpFile, buf);
      if (fs.existsSync(tmpFile) && fs.statSync(tmpFile).size > 10000) return true;
    }
  } catch {}
  // Fallback to curl
  try {
    execSync(`curl -sL --max-time 25 -o "${tmpFile}" "${url}"`, { timeout: 30000 });
    if (fs.existsSync(tmpFile) && fs.statSync(tmpFile).size > 10000) return true;
  } catch {}
  try { fs.unlinkSync(tmpFile); } catch {}
  return false;
}

async function ensurePseudopotential(element: string): Promise<string> {
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

    // On Linux/GCP: check PSEUDO_DIR env var and system-installed PP directories
    // (e.g. `sudo dpkg -i quantum-espresso-data-sssp_*.deb` installs to /usr/share/espresso/pseudo/)
    if (!IS_WINDOWS) {
      const linuxSearchDirs: string[] = [];
      if (process.env.PSEUDO_DIR) linuxSearchDirs.push(process.env.PSEUDO_DIR);
      linuxSearchDirs.push(
        "/usr/share/espresso/pseudo",
        "/usr/share/quantum-espresso/pseudo",
        "/usr/local/share/espresso/pseudo",
        "/opt/quantum-espresso/pseudo",
      );
      for (const dir of linuxSearchDirs) {
        try {
          if (!fs.existsSync(dir)) continue;
          const entries = fs.readdirSync(dir);
          // Flexible match: element name prefix + dot OR underscore + any suffix + .UPF/.upf
          // SSSP package uses underscores: Ir_pbe_v1.2.uspp.F.UPF, Te_pbe_v1.uspp.F.UPF
          // Prefer PBE over LDA (avoid pz), prefer non-relativistic for simpler SCF
          const candidates = entries.filter(f =>
            (f.startsWith(element + ".") || f.startsWith(element + "_")) &&
            (f.endsWith(".UPF") || f.endsWith(".upf")) &&
            !f.includes("pz")   // skip LDA (Perdew-Zunger) PPs
          );
          // Sort: prefer non-relativistic PBE, then relativistic PBE
          const match = candidates.sort((a, b) => {
            const aRel = a.includes("rel") ? 1 : 0;
            const bRel = b.includes("rel") ? 1 : 0;
            return aRel - bRel;
          })[0];
          if (!match) continue;
          const sysFile = path.join(dir, match);
          fs.copyFileSync(sysFile, tmpFile);
          // System-installed SSSP PPs are community-validated — only check UPF format,
          // not semicore heuristics (relativistic PPs may lack that tag but are still valid).
          if (validatePseudopotential(tmpFile)) {
            fs.renameSync(tmpFile, ppFile);
            console.log(`[QE-Worker] Copied valid PP for ${element} from ${dir}/${match} (${fs.statSync(ppFile).size} bytes)`);
            return ppFile;
          }
          try { fs.unlinkSync(tmpFile); } catch {}
        } catch {}
      }

      // Last-resort Linux fallback: extract SSSP .deb from repo root without sudo.
      // `dpkg -x <deb> <dir>` extracts to a local directory — no privileges needed.
      try {
        const debFile = path.join(PROJECT_ROOT, "quantum-espresso-data-sssp_1.3.0-2_all.deb");
        if (fs.existsSync(debFile)) {
          const extractDir = path.join(PP_SOURCE_DIR, ".sssp-extract");
          if (!fs.existsSync(path.join(extractDir, "usr"))) {
            console.log(`[QE-Worker] Extracting SSSP .deb to ${extractDir} (no sudo needed)...`);
            fs.mkdirSync(extractDir, { recursive: true });
            execSync(`dpkg -x "${debFile}" "${extractDir}"`, { timeout: 30000, stdio: "pipe" });
          }
          // SSSP installs to usr/share/espresso/pseudo/ inside the extract dir
          const ssspDir = path.join(extractDir, "usr/share/espresso/pseudo");
          if (fs.existsSync(ssspDir)) {
            const entries = fs.readdirSync(ssspDir);
            const candidates = entries.filter(f =>
              (f.startsWith(element + ".") || f.startsWith(element + "_")) &&
              (f.endsWith(".UPF") || f.endsWith(".upf")) &&
              !f.includes("pz")
            );
            const match = candidates.sort((a, b) => (a.includes("rel") ? 1 : 0) - (b.includes("rel") ? 1 : 0))[0];
            if (match) {
              fs.copyFileSync(path.join(ssspDir, match), tmpFile);
              if (validatePseudopotential(tmpFile)) {
                fs.renameSync(tmpFile, ppFile);
                console.log(`[QE-Worker] Copied PP for ${element} from extracted SSSP .deb (${match})`);
                return ppFile;
              }
              try { fs.unlinkSync(tmpFile); } catch {}
            }
          }
        }
      } catch (debErr: any) {
        console.warn(`[QE-Worker] SSSP .deb extract failed for ${element}: ${debErr?.message?.slice(0, 120)}`);
      }
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

    const urls = [PP_DOWNLOAD_URLS[element], PP_FALLBACK_URLS[element], PP_GBRV_URLS[element]].filter(Boolean);
    for (const url of urls) {
      try {
        const mirror = url!.includes("github") ? "pslibrary" : url!.includes("rutgers") ? "GBRV" : "QE";
        console.log(`[QE-Worker] Downloading PP for ${element} from ${mirror}...`);
        if (!await downloadPPToTemp(url!, tmpFile)) {
          console.log(`[QE-Worker] Download from ${mirror} failed for ${element} (${url})`);
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

    throw new Error(
      `No valid pseudopotential for ${element} — all download sources failed. ` +
      `Fix: sudo dpkg -i quantum-espresso-data-sssp_*.deb  (installs to /usr/share/espresso/pseudo/), ` +
      `or place a verified UPF file at ${PP_SOURCE_DIR}/${element}.UPF`
    );
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

// Default kspacing (Å⁻¹) — matches aiida-quantumespresso "fast" protocol.
// Lowering to 0.10 gives publication quality; raising to 0.30 gives
// Screening kspacing. Prior 0.157 Å⁻¹ was aiida's "balanced" protocol
// (publication-adjacent quality) — way too dense for a screening pipeline
// that tests hundreds of candidates. At a=5 Å + nspin=2, 0.157 generates
// ~729 k-points (1458 spin channels) per SCF iteration vs. 0.25 → ~125
// k-points (250 spin channels) — a 5–6× per-iteration speedup.
// aiida "fast" = 0.50 Å⁻¹, aiida "balanced" = 0.15. We pick 0.25 as
// the sweet spot: good enough for band structure + Tc screening (~0.1 eV
// accuracy), fast enough to converge within wall-time on 3 MPI ranks.
// Override via env QE_KSPACING for whole-pipeline tuning.
const DEFAULT_KSPACING = (() => {
  const env = parseFloat(process.env.QE_KSPACING ?? "");
  return Number.isFinite(env) && env > 0.02 && env < 0.5 ? env : 0.25;
})();

function autoKPoints(
  latticeA: number,
  cOverA?: number,
  minK: number = 4,
  dimensionality?: string,
  kspacing: number = DEFAULT_KSPACING,
  adaptiveOpts?: { stage?: "relax" | "vc-relax" | "scf" | "phonon"; isMetallic?: boolean; totalAtoms?: number },
): string {
  // Density-based k-point grid: n_i = ceil(2π / (kspacing * a_i)).
  // kspacing=0.157 Å⁻¹ ≈ densityFactor=40 (legacy); aiida's "fast" protocol
  // uses 0.15 (screening), "moderate" 0.125, "precise" 0.10.
  // For a 3.5Å cell this gives k≈12 per direction (4096 k-pts); precise
  // protocol ~19 (6859 k-pts) — 2-8× slower for ~0.05 eV energy improvement.
  //
  // Adaptive k-grid: stage-dependent spacing + metallicity boost.
  // Relax stages use coarser grids (forces converge with fewer k-points);
  // SCF/phonon use finer grids for energy/DOS accuracy.
  let effectiveKspacing = kspacing;
  if (adaptiveOpts?.stage) {
    switch (adaptiveOpts.stage) {
      case "relax":    effectiveKspacing = 0.40; break;
      case "vc-relax": effectiveKspacing = 0.30; break;
      case "scf":      effectiveKspacing = adaptiveOpts.isMetallic ? 0.20 : 0.25; break;
      case "phonon":   effectiveKspacing = 0.25; break;
    }
  }
  // Metallicity boost: metals need denser k-grids for Fermi surface resolution
  if (adaptiveOpts?.isMetallic && !adaptiveOpts?.stage) {
    effectiveKspacing *= 0.77;  // ~1.3x denser
  }
  // Large cells (>8 atoms) already sample well; slightly coarsen to save compute
  if (adaptiveOpts?.totalAtoms && adaptiveOpts.totalAtoms > 8) {
    effectiveKspacing *= 1.15;
  }

  const densityFactor = (2 * Math.PI) / effectiveKspacing;
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

// Full d-block. Used to detect systems where moments may form even when
// no element from MAGNETIC_ELEMENTS is present (e.g. Pd-H, Ta-N, Ru-Mn).
const TRANSITION_METALS = new Set([
  "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
  "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
]);

// Light electronegative elements that commonly induce TM moments via charge
// transfer / d-band reshaping (hydrides, nitrides, oxides, fluorides).
const MOMENT_INDUCERS = new Set(["H","N","O","F"]);

// Default seed for TM atoms that aren't in MAGNETIC_ELEMENTS but may carry an
// induced moment. Small enough to not bias chemistry, large enough to break
// spin symmetry so QE can find the polarized solution.
const INDUCED_TM_SEED = 0.5;

// Per-species wavefunction cutoff (Ry). Values are screening-quality,
// aligned with SSSP efficiency v1.3 recommendations and empirically
// validated by production runs. Consolidated from 4 duplicated tables
// (SCF / SCF-with-params / VC-relax / bands paths) so changes propagate
// consistently and the canonical source is obvious.
// Callers: use computeEcutwfc(elements, extraBoost) — it applies the
// hydrogen-presence floor and caller-specified boost.
const SPECIES_ECUTWFC: Record<string, number> = {
  H: 100, O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
  Li: 60, Be: 60, B: 55, C: 60, Na: 60, Mg: 55, Al: 50, Si: 50,
  La: 55, Ce: 55, Pr: 55, Nd: 55, Sm: 55, Eu: 55, Gd: 55, Tb: 55,
  Dy: 55, Ho: 55, Er: 55, Tm: 55, Yb: 55, Lu: 55, Sc: 50, Y: 50,
  Th: 55,
};

// Returns the recommended ecutwfc (Ry) for a composition. Applies a floor
// of 80 Ry when hydrogen is present (the raw SSSP cutoff of 60 Ry
// under-converges small-volume hydrides), plus any caller-requested boost.
function computeEcutwfc(elements: string[], extraBoost: number = 0, hydrogenFloor: number = 80, nonHFloor: number = 45): number {
  const hasH = elements.includes("H");
  const raw = elements.reduce((max, el) => Math.max(max, SPECIES_ECUTWFC[el] ?? 45), hasH ? hydrogenFloor : nonHFloor);
  return raw + extraBoost;
}

// Explicit nbnd reduces iteration cost for heavy-5d systems where the
// QE default (≈ nelec/2 + 20% buffer) drifts upward with total electron
// count and wastes effort on high-lying unoccupied bands we never use.
// Formula mirrors aiida-quantumespresso's PwBaseWorkChain default:
// nbnd = ceil(nelec/2) + max(4, ceil(nelec * 0.10))
// Doubled when nspin=2 because each band is per-spin in QE.
/**
 * Compute nbnd for QE. Uses actual atom count from positions (not formula
 * counts) to handle supercells correctly. Known structures like Nb3Sn in
 * Pm-3n have Z=2 (8 atoms), so the formula gives 53 electrons but the
 * cell has 106 electrons → nbnd must match the cell, not the formula.
 */
function computeNbnd(elements: string[], counts: Record<string, number>, nspin: number = 1, actualPositions?: Array<{ element: string }>): number {
  let nelec = 0;
  if (actualPositions && actualPositions.length > 0) {
    // Use actual positions in the cell (handles supercells correctly)
    for (const pos of actualPositions) {
      nelec += getZValence(pos.element);
    }
  } else {
    // Fallback to formula counts
    for (const el of elements) {
      const n = Math.round(counts[el] ?? 0);
      nelec += n * getZValence(el);
    }
  }
  const nbndSpin1 = Math.ceil(nelec / 2) + Math.max(4, Math.ceil(nelec * 0.10));
  return nspin === 2 ? nbndSpin1 * 2 : nbndSpin1;
}

// Tiered max_seconds. Systems with heavy elements (Z ≥ 55) have many
// valence electrons and large basis sets — each SCF iteration and phonon
// perturbation is 3-10× more expensive than light-element systems.
// The prior set (Hf-Au only) missed lanthanides (La-Lu), 6p metals
// (Bi, Pb, Tl), and alkaline earths (Ba, Cs) which are equally expensive.
// Apr-18: Bi2La2Y, Fe3LaSe4, BaBiLaTe3 all got the flat 88-min budget
// and wall-timed on every attempt.
const HEAVY_ELEMENTS = new Set([
  // Lanthanides (Z=57-71)
  "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
  // 5d transition metals (Z=72-80)
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  // 6p metals (Z=81-86)
  "Tl", "Pb", "Bi", "Po",
  // Actinides commonly used
  "Th", "U",
  // Heavy alkaline/alkaline-earth (large core, expensive PPs)
  "Cs", "Ba",
]);

function computeMaxSeconds(elements: string[], pressureGpa: number = 0): number {
  const base = QE_MAX_SECONDS;
  const heavyCount = elements.filter(el => HEAVY_ELEMENTS.has(el)).length;
  const hasH = elements.includes("H");
  // 2+ heavy elements OR high-P hydride: 3× budget.
  if (heavyCount >= 2 || (pressureGpa >= 100 && hasH)) {
    return Math.floor(base * 3);
  }
  // Single heavy element or H-content only: 1.5× budget.
  if (heavyCount >= 1 || hasH) {
    return Math.floor(base * 1.5);
  }
  return base;
}

// Returns true if the system can plausibly carry a net spin moment and should
// therefore be run with nspin=2. Catches cases the narrow MAGNETIC_ELEMENTS
// list misses: 4d/5d TMs (Ru, Pd, Ta, W…), TM-hydrides, TM-nitrides, and
// multi-TM systems where d-band frustration drives moment formation.
function mayHaveMagneticMoment(elements: string[]): boolean {
  if (elements.some(el => el in MAGNETIC_ELEMENTS)) return true;
  const tmCount = elements.filter(el => TRANSITION_METALS.has(el)).length;
  if (tmCount === 0) return false;
  if (tmCount >= 2) return true;
  if (elements.some(el => MOMENT_INDUCERS.has(el))) return true;
  return false;
}

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
  seedAllTMs: boolean = false,
): string {
  let lines = "";
  const magneticIndices: number[] = [];

  // Atoms that get a non-zero starting moment. With seedAllTMs we include any
  // d-block element so QE can reach polarized solutions for Pd-H, Ta-N, etc.
  for (let idx = 0; idx < elements.length; idx++) {
    const el = elements[idx];
    if (el in MAGNETIC_ELEMENTS) {
      magneticIndices.push(idx);
    } else if (seedAllTMs && TRANSITION_METALS.has(el)) {
      magneticIndices.push(idx);
    }
  }

  if (magneticIndices.length === 0) {
    // Safety net: callers only invoke this when nspin=2 is going to be emitted
    // (broadMagnetic=true or DFT+U nspin2 path). If nspin=2 appears without
    // any starting_magnetization line, QE aborts with "some starting_magnetization
    // MUST be set" — 75 production jobs failed on this. Emit a tiny seed for
    // species 1 so QE parses the block; 0.1 is small enough not to bias a
    // genuinely non-magnetic solution.
    return `  starting_magnetization(1) = 0.1,\n`;
  }

  const afmPattern = useAFM ? determineAFMPattern(elements, counts) : null;

  for (let idx = 0; idx < elements.length; idx++) {
    const el = elements[idx];
    let mag = MAGNETIC_ELEMENTS[el];
    if (mag === undefined) {
      if (seedAllTMs && TRANSITION_METALS.has(el)) {
        mag = INDUCED_TM_SEED;
      } else {
        mag = 0.0;
      }
    }

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

function generateCellParameters(
  latticeA: number, cOverA: number, ibrav: number,
  bOverA: number = 1.0, elements?: string[], counts?: Record<string, number>,
  alpha: number = 90, beta: number = 90, gamma: number = 90,
): string {
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

  // Monoclinic: beta ≠ 90° — v3 tilted in xz-plane
  if (Math.abs(beta - 90) > 1 && Math.abs(alpha - 90) < 1 && Math.abs(gamma - 90) < 1) {
    const betaR = beta * Math.PI / 180;
    const v3x = c * Math.cos(betaR);
    const v3z = c * Math.sin(betaR);
    return `CELL_PARAMETERS {angstrom}
  ${a.toFixed(8)}  0.000000000  0.000000000
  0.000000000  ${b.toFixed(8)}  0.000000000
  ${v3x.toFixed(8)}  0.000000000  ${v3z.toFixed(8)}`;
  }

  // Triclinic: general angles — full 3×3 matrix
  if (Math.abs(alpha - 90) > 1 || Math.abs(gamma - 90) > 1) {
    const alphaR = alpha * Math.PI / 180;
    const betaR = beta * Math.PI / 180;
    const gammaR = gamma * Math.PI / 180;
    const cosA = Math.cos(alphaR), cosB = Math.cos(betaR), cosG = Math.cos(gammaR);
    const sinG = Math.sin(gammaR);
    const cx = c * cosB;
    const cy = sinG > 1e-10 ? c * (cosA - cosB * cosG) / sinG : 0;
    const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
    return `CELL_PARAMETERS {angstrom}
  ${a.toFixed(8)}  0.000000000  0.000000000
  ${(b * cosG).toFixed(8)}  ${(b * sinG).toFixed(8)}  0.000000000
  ${cx.toFixed(8)}  ${cy.toFixed(8)}  ${cz.toFixed(8)}`;
  }

  // Orthorhombic/tetragonal/cubic: all angles 90°
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
  const hasHydrogen = elements.includes("H");
  // Uses module-level SPECIES_ECUTWFC table. Higher floor (100/60) here than
  // other call-sites because this is the default SCF path with no retry
  // boost — conservative to land convergence on attempt 1.
  const baseEcutwfc = computeEcutwfc(elements, 0, 80, 45);
  const ecutwfc = Math.max(baseEcutwfc, hasHydrogen ? 100 : 60);
  const ecutrho = ecutwfc * ecutrhoMultiplier(elements);

  const hasMagnetic = elements.some(el => el in MAGNETIC_ELEMENTS);
  // Broaden: 4d/5d TMs and TM-hydrides/nitrides also need nspin=2 to converge.
  const broadMagnetic = mayHaveMagneticMoment(elements);
  const nspin = broadMagnetic ? 2 : 1;
  const useAFM = hasMagnetic && isAFMCandidate(elements, counts);

  let startingMagLines = "";
  if (broadMagnetic) {
    startingMagLines = generateMagnetizationLines(elements, counts, useAFM, !hasMagnetic);
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

  const knownS = lookupKnownStructure(formula);
  const { cOverA } = determineCrystalSystem(elements, counts);
  const bOverA = knownS?.latticeB ? knownS.latticeB / knownS.latticeA : estimateBOverA(elements, counts);
  const effCOverA = knownS?.latticeC ? knownS.latticeC / knownS.latticeA : cOverA;
  const cellBlock = `\n${generateCellParameters(latticeA, effCOverA, 0, bOverA, elements, counts, knownS?.alpha ?? 90, knownS?.beta ?? 90, knownS?.gamma ?? 90)}`;

  return `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${formula.replace(/[^a-zA-Z0-9]/g, "")}',
  outdir = './tmp',
  disk_io = 'medium',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = 1.0d-2,
  etot_conv_thr = 1.0d-4,
  max_seconds = ${QE_MAX_SECONDS},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  input_dft = 'PBE',
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
${autoKPoints(latticeA, cOverA, bOverA, undefined, DEFAULT_KSPACING, { stage: "scf", totalAtoms: positions.length })}
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
  // Tier 0: Known structures database — exact literature Wyckoff positions.
  // These produce stable phonon spectra for verified compounds (LaH10, CaH6, etc.)
  if (formula) {
    try {
      const known = lookupKnownStructure(formula);
      if (known) {
        console.log(`[QE-Worker] Using known structure for ${formula} (${known.atoms.length} atoms, ${known.spaceGroup}, a=${known.latticeA} Å)`);
        return perturbPositions(known.atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z })), 0.002, latticeA ?? known.latticeA);
      }
    } catch {}
  }

  // Tier 1: Prototype matching
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

function autoPhononQGrid(elements: string[], totalAtoms?: number): [number, number, number] {
  // Screening-quality q-grid. Cost scales as n_atoms × n_q × 3*n_atoms
  // perturbations. Each perturbation is a mini-SCF, and heavy elements
  // make each one 3-10× more expensive than light elements.
  //
  // Scale the grid with atom count AND element weight:
  //   ≤ 4 atoms, all light:  2×2×2 (8 q-points, manageable)
  //   5-8 atoms, all light:  2×2×2 (still OK)
  //   5-8 atoms, any heavy:  1×1×1 (Gamma-only — Bi2CuSe3 7 atoms 2×2×2
  //     ran ~16h before wall-time kill; C3SnW4 8 atoms similar)
  //   9+ atoms:              1×1×1 (always Gamma-only)
  //
  // Gamma-only is enough for screening: detects imaginary modes at the
  // zone center, which catches the most common instabilities.
  const nAtoms = totalAtoms ?? 6;
  if (nAtoms >= 9) return [1, 1, 1];
  if (nAtoms >= 5 && elements.some(el => HEAVY_ELEMENTS.has(el))) return [1, 1, 1];
  return [2, 2, 2];
}

function generatePhononInput(formula: string, elements: string[] = [], totalAtoms: number = 6, opts?: { maxSeconds?: number; recover?: boolean; tr2Ph?: string; alphaMix?: number }): string {
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const [nq1, nq2, nq3] = autoPhononQGrid(elements, totalAtoms);
  // tr2_ph=1.0d-10: screening threshold — saves 30-50% iterations vs 1e-12
  // without changing screening-level Tc estimates. 1e-12 is only needed for
  // publication-quality phonon DOS.
  //
  // alpha_mix(1)=0.5: faster DFPT convergence than the conservative 0.3.
  // Safe for most systems; if it oscillates, the retry uses 0.1.
  //
  // reduce_io=.true.: reduces disk I/O during phonon, saves 10-20% on
  // IO-bound systems. Does not affect results.
  const recoverLine = opts?.recover ? `  recover = .true.,\n` : "";
  const maxSecLine = opts?.maxSeconds ? `  max_seconds = ${opts.maxSeconds},\n` : "";
  const tr2Ph = opts?.tr2Ph ?? "1.0d-10";
  const alphaMix = opts?.alphaMix ?? 0.5;
  //
  // Gamma-only (1×1×1): use ldisp=.false. so ph.x prints omega(N) = X [THz]
  // = Y [cm-1] directly to stdout, which parsePhononOutput already handles.
  // ldisp=.true. with 1×1×1 writes the dynamical matrix to .dyn files in a
  // binary/numerical format that dynmat.x must post-process, and the various
  // parsers fail to match dynmat.x's tabular output → "0 modes" bug.
  const isGammaOnly = nq1 === 1 && nq2 === 1 && nq3 === 1;
  if (isGammaOnly) {
    // QE ph.x with ldisp=.false. REQUIRES an explicit q-point card after the
    // namelist. Without it, some QE versions read garbage or crash. Specify
    // Gamma (0 0 0) explicitly.
    return `Gamma-only phonon calculation
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = ${tr2Ph},
  alpha_mix(1) = ${alphaMix},
  reduce_io = .true.,
  ldisp = .false.,
${recoverLine}${maxSecLine}/
0.0 0.0 0.0
`;
  }
  return `Phonon dispersions on ${nq1}x${nq2}x${nq3} grid
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  tr2_ph = ${tr2Ph},
  alpha_mix(1) = ${alphaMix},
  reduce_io = .true.,
  ldisp = .true.,
  nq1 = ${nq1}, nq2 = ${nq2}, nq3 = ${nq3},
${recoverLine}${maxSecLine}/
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

  // Primary format (QE 6+, ldisp=.true.):  "omega( N) = X.xxx [THz] =   Y.yyy [cm-1]"
  // Also covers gamma-only:                 "freq ( N) = X.xxx [THz] =   Y.yyy [cm-1]"
  // QE 7.x degenerate range format:         "omega(N-M) = X.xxx [THz] =   Y.yyy [cm-1]"
  // The range form (omega(1-3)) is used when modes are degenerate; the regex must
  // allow an optional "-N" suffix inside the parentheses to match both forms.
  const twoUnitMatches = stdout.matchAll(/(?:freq|omega)\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*([-\d.]+)\s+\[THz\]\s*=\s*([-\d.]+)\s+\[cm-1\]/g);
  for (const m of twoUnitMatches) {
    result.frequencies.push(parseFloat(m[2])); // cm-1 value from group 2
  }

  if (result.frequencies.length === 0) {
    // Fallback: some older QE or post-processing outputs only the cm-1 value
    // "freq ( N) = Y.yyy [cm-1]"  or  "omega( N) = Y.yyy [cm-1]"  or range "omega(N-M) = ..."
    const singleUnitMatches = stdout.matchAll(/(?:freq|omega)\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*([-\d.]+)\s+\[cm-1\]/g);
    for (const m of singleUnitMatches) {
      result.frequencies.push(parseFloat(m[1]));
    }
  }

  if (result.frequencies.length === 0) {
    // Last-resort: matdyn.x / older ph.x format "     N    freq =   Y.yyy [THz] =  Z.zzz [cm-1]"
    const matdynMatches = stdout.matchAll(/\d+\s+freq\s*=\s*([-\d.]+)\s+\[THz\]\s*=\s*([-\d.]+)\s+\[cm-1\]/g);
    for (const m of matdynMatches) {
      result.frequencies.push(parseFloat(m[2]));
    }
  }

  if (result.frequencies.length === 0) {
    // dynmat.x tabular format (QE 6.x/7.x):
    //   # mode   [cm-1]   [THz]  IR
    //      1      123.45   3.678    0.123
    //      2       45.67   1.234    0.456
    let inTabular = false;
    for (const line of stdout.split("\n")) {
      if (line.match(/#\s*mode\s+\[cm-1\]/i)) {
        inTabular = true;
        continue;
      }
      if (inTabular) {
        const cols = line.trim().split(/\s+/);
        if (cols.length >= 2 && /^\d+$/.test(cols[0])) {
          const val = parseFloat(cols[1]);
          if (Number.isFinite(val)) result.frequencies.push(val);
        } else if (line.trim() === "" || line.startsWith("*")) {
          inTabular = false;
        }
      }
    }
  }

  if (result.frequencies.length > 0) {
    result.lowestFrequency = Math.min(...result.frequencies);
    result.highestFrequency = Math.max(...result.frequencies);
    result.imaginaryCount = result.frequencies.filter(f => f < -20).length;
    result.hasImaginary = result.imaginaryCount > 0;
  }

  // ph.x convergence markers — "End of self-consistent calculation" belongs to pw.x
  result.converged = (
    stdout.includes("Phonon calculation on a mesh") ||
    stdout.includes("Writing dynmat at Gamma") ||
    stdout.includes("PHONON       :") ||
    (result.frequencies.length > 0 && !stdout.includes("ERROR") && !stdout.includes("stopping"))
  );

  // QE wall time format: "PHONON       :   2m39.47s CPU   2m52.16s WALL"
  // Pattern: optional "Xh" then "Xm" then "Y.Ys" then "WALL" (WALL is a suffix, not prefix)
  const wallMatch =
    stdout.match(/(\d+)h\s*(\d+)m\s*([\d.]+)s\s+WALL/) ||
    stdout.match(/(\d+)m\s*([\d.]+)s\s+WALL/) ||
    stdout.match(/([\d.]+)s\s+WALL/);
  if (wallMatch) {
    if (wallMatch.length >= 4 && wallMatch[3]) {
      // Xh Xm Ys form
      result.wallTimeSeconds = parseInt(wallMatch[1]) * 3600 + parseInt(wallMatch[2]) * 60 + parseFloat(wallMatch[3]);
    } else if (wallMatch.length >= 3 && wallMatch[2]) {
      // Xm Ys form
      result.wallTimeSeconds = parseInt(wallMatch[1]) * 60 + parseFloat(wallMatch[2]);
    } else {
      // Ys form
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
      if (dist < dMin - 1e-3) {
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
  // Auto-populate from known-structures.ts — any compound with pressureGPa > 0
  // in the database gets its known pressure. No manual sync needed.
  const KNOWN_SUPERHYDRIDE_PRESSURES: Record<string, number> = {};
  for (const ksFormula of getKnownStructureFormulas()) {
    const ks = lookupKnownStructure(ksFormula);
    if (ks && ks.pressureGPa > 0) {
      KNOWN_SUPERHYDRIDE_PRESSURES[ksFormula] = ks.pressureGPa;
    }
  }
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

    if (hPerMetal >= 6) {
      return { valid: true, reason: `High H/metal ratio ${hPerMetal.toFixed(1)} — tagged as high-pressure candidate (>100 GPa required)`, highPressure: true, estimatedPressureGPa: Math.min(300, 50 + hPerMetal * 15) };
    }

    // H/metal ratio 3-6: moderate-pressure hydride (H3S-class)
    // These aren't as extreme as LaH10 but still require high pressure
    if (hPerMetal >= 3) {
      return { valid: true, reason: `Moderate H/metal ratio ${hPerMetal.toFixed(1)} — tagged as high-pressure candidate`, highPressure: true, estimatedPressureGPa: Math.min(250, 100 + hPerMetal * 20) };
    }

    if (hRatio >= 0.75) {
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

function parseXTBOptXyz(
  optPath: string,
  effectiveLattice: number,
  nExpected: number,
): Array<{ element: string; x: number; y: number; z: number }> | null {
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
  return relaxed.length === nExpected ? relaxed : null;
}

function validateXTBRelaxation(
  relaxed: Array<{ element: string; x: number; y: number; z: number }>,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  effectiveLattice: number,
  maxFracDisp: number = 0.35,
  absMinDist: number = 0.5,
): { valid: boolean; maxDisp: number; maxDispAtom: number; minDist: number; minPair: string } {
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
    if (disp > maxDisp) { maxDisp = disp; maxDispAtom = i; }
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
      if (dist < minDist) { minDist = dist; minPair = `${relaxed[i].element}-${relaxed[j].element}`; }
    }
  }

  return {
    valid: maxDisp <= maxFracDisp && minDist >= absMinDist,
    maxDisp, maxDispAtom, minDist, minPair,
  };
}

function tryXTBPreRelaxation(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  workDir: string,
  pressureGpa: number = 0,
): Array<{ element: string; x: number; y: number; z: number }> | null {
  try {
    if (!fs.existsSync(XTB_BIN)) {
      console.warn(`[QE-Worker] xTB binary not found at ${XTB_BIN} — skipping pre-relax. Set XTB_BIN env var or install xtb. Run: which xtb && xtb --version`);
      return null;
    }

    const preRelaxElements = Array.from(new Set(positions.map(p => p.element)));
    const isHighPressureHydride = pressureGpa >= 50 && preRelaxElements.includes("H");
    const scale = pressureGpa > 0 ? computePressureScale(pressureGpa, preRelaxElements) : 1.0;
    const effectiveLattice = latticeA * scale;
    const nAtoms = positions.length;

    const xyzPath = path.join(workDir, "pre_relax.xyz");
    let xyzContent = `${nAtoms}\npre-relaxation${pressureGpa > 0 ? ` @ ${pressureGpa} GPa` : ""}\n`;
    for (const pos of positions) {
      xyzContent += `${pos.element}  ${(pos.x * effectiveLattice).toFixed(6)}  ${(pos.y * effectiveLattice).toFixed(6)}  ${(pos.z * effectiveLattice).toFixed(6)}\n`;
    }
    fs.writeFileSync(xyzPath, xyzContent);

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
      OMP_STACKSIZE: "512M",
    };

    const optPath = path.join(workDir, "xtbopt.xyz");
    const MAX_FRAC_DISPLACEMENT = 0.35;
    const ABS_MIN_DIST = 0.5;
    let relaxed: Array<{ element: string; x: number; y: number; z: number }> | null = null;

    // ---- Strategy 1: Constrained GFN-FF (pressure-aware) ----
    // For high-P hydrides, xTB molecular mode collapses H cages because it
    // doesn't know about pressure. Use a reference-structure constraint with
    // a spring constant that keeps atoms near their Wyckoff positions while
    // allowing local relaxation. The spring constant scales with pressure —
    // higher pressure = stiffer constraint (cage is more rigid).
    if (isHighPressureHydride) {
      // Spring constant in Hartree/Bohr^2: higher pressure → stiffer springs
      // At 200 GPa, k≈0.10; at 50 GPa, k≈0.03. This prevents cage collapse
      // while allowing ~0.05-0.10 frac displacement for position refinement.
      const kForce = Math.min(0.15, 0.02 + pressureGpa * 0.0004);
      const xcontrolPath = path.join(workDir, "xcontrol_pressure.in");
      fs.writeFileSync(xcontrolPath,
        `$constrain\n  force constant=${kForce.toFixed(4)}\n  reference=pre_relax.xyz\n  atoms: 1-${nAtoms}\n$end\n`
      );
      console.log(`[QE-Worker] xTB high-P hydride mode: constrained GFN-FF (k=${kForce.toFixed(4)}, P=${pressureGpa} GPa, a_eff=${effectiveLattice.toFixed(3)} A)`);

      try {
        // Clean stale output from prior runs
        for (const f of ["xtbopt.xyz", "gfnff_topo"]) {
          const fp = path.join(workDir, f);
          if (fs.existsSync(fp)) fs.unlinkSync(fp);
        }
        execSync(
          `${XTB_BIN} pre_relax.xyz --gfnff --opt crude --input xcontrol_pressure.in 2>&1`,
          { cwd: workDir, timeout: 30000, env, maxBuffer: 5 * 1024 * 1024 }
        );
        relaxed = parseXTBOptXyz(optPath, effectiveLattice, nAtoms);
      } catch { /* fall through */ }

      // If constrained GFN-FF didn't produce valid output, try GFN2 constrained
      if (!relaxed) {
        try {
          for (const f of ["xtbopt.xyz"]) {
            const fp = path.join(workDir, f);
            if (fs.existsSync(fp)) fs.unlinkSync(fp);
          }
          execSync(
            `${XTB_BIN} pre_relax.xyz --gfn 2 --opt crude --input xcontrol_pressure.in 2>&1`,
            { cwd: workDir, timeout: 30000, env, maxBuffer: 5 * 1024 * 1024 }
          );
          relaxed = parseXTBOptXyz(optPath, effectiveLattice, nAtoms);
        } catch { /* fall through */ }
      }
    }

    // ---- Strategy 2: Unconstrained molecular GFN-FF / GFN2 ----
    // For ambient-pressure or non-hydride materials, use the original
    // molecular optimization (no constraints needed — structure is stable at 0 GPa).
    if (!relaxed) {
      let gfnffOk = false;
      try {
        for (const f of ["xtbopt.xyz", "gfnff_topo"]) {
          const fp = path.join(workDir, f);
          if (fs.existsSync(fp)) fs.unlinkSync(fp);
        }
        execSync(
          `${XTB_BIN} pre_relax.xyz --gfnff --opt crude 2>&1`,
          { cwd: workDir, timeout: 20000, env, maxBuffer: 5 * 1024 * 1024 }
        );
        gfnffOk = fs.existsSync(optPath);
      } catch { /* fall through to GFN2 */ }

      if (!gfnffOk) {
        try {
          const fp = path.join(workDir, "xtbopt.xyz");
          if (fs.existsSync(fp)) fs.unlinkSync(fp);
          execSync(
            `${XTB_BIN} pre_relax.xyz --gfn 2 --opt crude 2>&1`,
            { cwd: workDir, timeout: 30000, env, maxBuffer: 5 * 1024 * 1024 }
          );
        } catch { /* fall through */ }
      }

      relaxed = parseXTBOptXyz(optPath, effectiveLattice, nAtoms);
    }

    if (!relaxed) {
      console.log(`[QE-Worker] xTB pre-relaxation atom count mismatch or no output for ${nAtoms} atoms`);
      return null;
    }

    // ---- Validate result ----
    const v = validateXTBRelaxation(relaxed, positions, effectiveLattice, MAX_FRAC_DISPLACEMENT, ABS_MIN_DIST);
    if (v.maxDisp > MAX_FRAC_DISPLACEMENT) {
      console.log(`[QE-Worker] xTB pre-relax rejected: atom ${v.maxDispAtom} (${relaxed[v.maxDispAtom]?.element}) displaced ${v.maxDisp.toFixed(3)} frac units (max ${MAX_FRAC_DISPLACEMENT}) — molecular optimizer likely collapsed structure`);
      return null;
    }
    if (v.minDist < ABS_MIN_DIST) {
      console.log(`[QE-Worker] xTB pre-relax rejected: ${v.minPair} distance ${v.minDist.toFixed(3)} A < ${ABS_MIN_DIST} A absolute minimum — molecular optimizer collapsed atoms`);
      return null;
    }

    const mode = isHighPressureHydride ? "constrained" : "unconstrained";
    console.log(`[QE-Worker] xTB pre-relaxation succeeded for ${nAtoms} atoms (${mode}, scale=${scale.toFixed(3)}, effLattice=${effectiveLattice.toFixed(3)} A, maxDisp=${v.maxDisp.toFixed(3)} frac, minDist=${v.minDist.toFixed(3)} A [${v.minPair}])`);
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
  params: { mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number; convThr?: string; forcConvThr?: string; etotConvThr?: string; dftPlusULines?: string; dftPlusUNspin2?: boolean; mixingMode?: string; mixingNdim?: number; startingwfc?: string; startingpot?: string; diagoThrInit?: string; restartFromScratch?: boolean; maxSecondsOverride?: number },
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  // Uses module-level SPECIES_ECUTWFC table. Screening minimum: 45 Ry non-H,
  // 80 Ry H (PAW needs less than NCPP; saves ~30% vs 60/100). ecutwfcBoost
  // comes from the retry ladder to escalate cutoff on non-convergence.
  const baseEcutwfc = computeEcutwfc(elements, 0, 80, 45);
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

  // Use known-structure lattice parameters when available (monoclinic/triclinic need angles)
  const knownStruct = lookupKnownStructure(formula);
  const cOverA2 = knownStruct?.latticeC ? knownStruct.latticeC / knownStruct.latticeA : estimateCOverA(elements, counts);
  const bOverA2 = knownStruct?.latticeB ? knownStruct.latticeB / knownStruct.latticeA : estimateBOverA(elements, counts);
  const cellAlpha = knownStruct?.alpha ?? 90;
  const cellBeta = knownStruct?.beta ?? 90;
  const cellGamma = knownStruct?.gamma ?? 90;
  const cellBlock2 = `\n${generateCellParameters(latticeA, cOverA2, 0, bOverA2, elements, counts, cellAlpha, cellBeta, cellGamma)}`;

  // DFT+U overrides normal nspin/magnetization block when activated.
  // Broaden the detector beyond MAGNETIC_ELEMENTS so 4d/5d TMs and TM-hydrides
  // (Pd-H, Ta-N, Ru-Mn, …) also get nspin=2 + seeded moments — without this
  // they show non-zero magnetization in the output but never converge SCF.
  const hasMagEl = elements.some(el => el in MAGNETIC_ELEMENTS);
  const broadMagnetic = mayHaveMagneticMoment(elements);
  const useNspin2 = (params.dftPlusUNspin2 ?? false) || broadMagnetic;
  // When DFT+U nspin2 is set, starting_magnetization is already embedded in dftPlusULines
  const magBlock = (params.dftPlusUNspin2 ?? false)
    ? ""
    : (broadMagnetic ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !hasMagEl) : "");
  const hubbardBlock = params.dftPlusULines ?? "";

  const nspinOut = useNspin2 ? 2 : 1;
  const nbnd = computeNbnd(elements, counts, nspinOut, positions);
  // restart_mode='restart' on retry attempts 2+ preserves the partial SCF
  // charge density from the previous wall-time-killed attempt instead of
  // throwing it away — aiida's standard move for ElectronicMaxStep /
  // MaxSeconds restarts. Saves 30-60% wall time on the second attempt
  // for slow-converging heavy-TM intermetallics.
  const restartMode = params.restartFromScratch === false ? "restart" : "from_scratch";
  return `&CONTROL
  calculation = 'scf',
  restart_mode = '${restartMode}',
  prefix = '${formula.replace(/[^a-zA-Z0-9]/g, "")}',
  outdir = './tmp',
  disk_io = 'medium',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${forcConvThr},
  etot_conv_thr = ${etotConvThr},
  max_seconds = ${params.maxSecondsOverride ?? QE_MAX_SECONDS},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  nbnd = ${nbnd},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = '${smearing}',
  degauss = ${degauss},
  nspin = ${nspinOut},
${magBlock}${hubbardBlock}/
&ELECTRONS
  electron_maxstep = ${params.maxSteps},
  conv_thr = ${convThr},
  mixing_beta = ${params.mixingBeta},
  mixing_mode = '${params.mixingMode ?? "plain"}',
  mixing_ndim = ${params.mixingNdim ?? 8},
  diagonalization = '${params.diag}',
  diago_thr_init = ${params.diagoThrInit ?? "1.0d-4"},
${params.startingwfc ? `  startingwfc = '${params.startingwfc}',\n` : ""}${params.startingpot ? `  startingpot = '${params.startingpot}',\n` : ""}  scf_must_converge = .false.,
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${autoKPoints(latticeA, cOverA2, bOverA2, undefined, DEFAULT_KSPACING, { stage: "scf", totalAtoms: positions.length })}
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
  opts?: { phase?: "damped" | "bfgs" },
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  // Uses module-level SPECIES_ECUTWFC table. vc-relax path historically
  // used a trimmed table (no lanthanides) but the full table is safe here —
  // unknown elements fall through to the 45 Ry default.
  const hasHydrogen = elements.includes("H");
  const baseEcutwfc = computeEcutwfc(elements, 0, 80, 45);
  const ecutwfc = Math.max(baseEcutwfc, hasHydrogen ? 100 : 60);
  const ecutrho = ecutwfc * ecutrhoMultiplier(elements);

  const hasMagnetic = elements.some(el => el in MAGNETIC_ELEMENTS);
  const broadMagnetic = mayHaveMagneticMoment(elements);
  const nspin = broadMagnetic ? 2 : 1;
  let magLines = "";
  if (broadMagnetic) {
    magLines = generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !hasMagnetic);
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

  // vc-relax wall-time cap — scaled by system complexity.
  const hasHVcr = elements.includes("H");
  const hasMagVcr = elements.some(el => el in MAGNETIC_ELEMENTS);
  const isHighPHydride = hasHVcr && pressureGPa >= 50 && totalAtoms >= 7;
  const vcRelaxMaxSeconds = isHighPHydride ? 5400    // 90 min for high-P hydrides
    : hasMagVcr ? 3600                                // 60 min for magnetic systems
    : Math.max(600, Math.min(QE_MAX_SECONDS, 1800));  // 30 min default
  const VC_RELAX_MAX_SECONDS = vcRelaxMaxSeconds;
  // Tuned vc-relax parameters for different system types
  const vcMixingMode = isHighPHydride ? "local-TF" : hasMagVcr ? "local-TF" : "plain";
  const vcMixingBeta = isHighPHydride ? 0.2 : hasMagVcr ? 0.2 : 0.3;
  const vcConvThr = isHighPHydride ? "1.0d-5" : "1.0d-4";

  // 2-PHASE vc-relax strategy:
  // Phase 1 (damped dynamics): robust basin-finding, never diverges.
  //   Atoms follow forces with friction — always makes progress downhill.
  //   200 steps, loose force threshold. Finds the right basin.
  // Phase 2 (BFGS): fast convergence from the damped result.
  //   Starts from Phase 1 geometry which is already near a minimum.
  //   250 steps, tight force threshold. Tightens to publication quality.
  //
  // If phase='damped' (Phase 1): use damp/damp-w with loose thresholds
  // If phase='bfgs' (Phase 2): use bfgs/bfgs with tight thresholds
  const phase = opts?.phase ?? "damped"; // Default to Phase 1 for all materials
  const isDamped = phase === "damped";

  const vcNstep = isDamped ? 200 : 250;
  const vcForcConvThr = isDamped ? "5.0d-3" : "1.0d-3";
  const ionDynamics = isDamped ? "damp" : "bfgs";
  const cellDynamics = isDamped ? "damp-w" : "bfgs";
  const vcDiskIo = isDamped ? "low" : "medium"; // Phase 2 saves .save for phonon

  return `&CONTROL
  calculation = 'vc-relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}',
  outdir = './tmp',
  disk_io = '${vcDiskIo}',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${vcForcConvThr},
  etot_conv_thr = 1.0d-4,
  nstep = ${vcNstep},
  max_seconds = ${VC_RELAX_MAX_SECONDS},
/
&SYSTEM
  ibrav = 0,
  nat = ${totalAtoms},
  ntyp = ${nTypes},
  ecutwfc = ${ecutwfc},
  ecutrho = ${ecutrho},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = ${vcRelaxDegauss},
  nspin = ${nspin},
${magLines}/
&ELECTRONS
  electron_maxstep = 300,
  conv_thr = ${vcConvThr},
  mixing_beta = ${vcMixingBeta},
  mixing_mode = '${vcMixingMode}',
  diagonalization = 'david',
  scf_must_converge = .false.,
/
&IONS
  ion_dynamics = '${ionDynamics}',
/
&CELL
  cell_dynamics = '${cellDynamics}',
  press = ${(pressureGPa * 10.0).toFixed(4)},
  press_conv_thr = ${pressureGPa > 50 ? 1.0 : 0.5},
/
ATOMIC_SPECIES
${atomicSpecies}
ATOMIC_POSITIONS {crystal}
${atomicPositions}
K_POINTS {automatic}
${autoKPoints(latticeA, cOverA, bOverAVcr, undefined, DEFAULT_KSPACING, { stage: "vc-relax", totalAtoms })}
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
  // Damped dynamics doesn't print "bfgs converged" — check if it reached
  // low forces instead (the damped run just stops at nstep/max_seconds)
  if (stdout.includes("cell_dynamics = 'damp-w'") || stdout.includes("ion_dynamics  = 'damp'") || stdout.includes("ion_dynamics = 'damp'")) {
    // For damped dynamics, "convergence" means it completed without crashing
    if (stdout.includes("JOB DONE") && !stdout.includes("Error in routine")) {
      result.converged = true; // damped ran to completion = usable geometry
    }
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

  // Match the LAST CELL_PARAMETERS block (damped dynamics outputs many)
  const cellMatches = [...stdout.matchAll(/CELL_PARAMETERS\s*\(([^)]*)\)\s*\n([\s\S]*?)(?=\n\s*\n|\nATOMIC|\nEnd|\n\s*Writing|\n\s*PWSCF|$)/g)];
  const cellLines = cellMatches.length > 0 ? cellMatches[cellMatches.length - 1] : null;
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

  // Match ATOMIC_POSITIONS blocks — terminator set expanded so we still
  // capture the last geometry when vc-relax crashes or damped dynamics
  // completes without the usual trailing blank line / "End final coordinates"
  // marker. Also handles damped dynamics output where positions are followed
  // by "Writing config" or "total cpu time" lines.
  const posBlocks = [...stdout.matchAll(/ATOMIC_POSITIONS\s*\{?\s*(\w+)\s*\}?\s*\n([\s\S]*?)(?=\n\s*\n|\nEnd|\nCELL_PARAMETERS|\n\s*Writing|\n\s*PWSCF\b|\n\s*init_run\b|\n\s*electrons\b|\n\s*BFGS\b|\n\s*JOB DONE|\n\s*%%%%%%%%%%|\n\s*Error in routine|\n\s*total cpu time|\n\s*General routines|\n\s*Parallel routines|\n\s*number of|\n\s*convergence has|$)/g)];
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

function runQECommand(binary: string, inputFile: string, workDir: string, timeoutMsOverride?: number): Promise<{ stdout: string; stderr: string; exitCode: number }> {
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

    proc.stdout!.on("data", (data: Buffer) => { stdout += data.toString(); });
    proc.stderr!.on("data", (data: Buffer) => { stderr += data.toString(); });
    proc.stdout!.on("end", () => { stdoutEnded = true; tryResolve(); });
    proc.stderr!.on("end", () => { stderrEnded = true; tryResolve(); });

    let killTimeout: ReturnType<typeof setTimeout> | null = null;
    // +60s grace so QE's own max_seconds (set to wall-budget - 120s) fires
    // first and produces a clean JOB DONE. Only if QE doesn't honor it do
    // we force-kill here.
    const killMs = (timeoutMsOverride ?? QE_TIMEOUT_MS) + 60_000;
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
    }, killMs);

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

// ---------------------------------------------------------------------------
// DFPT electron-phonon coupling pipeline for top-scoring candidates.
// Runs ph.x with electron_phonon='interpolated' on a coarse 2×2×2 q-grid,
// then q2r.x → matdyn.x (DOS) → Eliashberg Tc estimate.
// Called only when opts.ensembleScore > 0.7 inside runFullDFT.
// ---------------------------------------------------------------------------
async function runDFPTEPC(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  jobDir: string,
  pressureGpa: number,
): Promise<QEDFPTResult> {
  const t0 = Date.now();
  const warnings: string[] = [];
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  // Spec: 2×2×2 coarse q-grid for speed.
  const nqGrid: [number, number, number] = [2, 2, 2];

  // --- ph.x with electron_phonon = 'interpolated' ---
  const phInput = generatePhononGridInput(prefix, nqGrid[0], nqGrid[1], nqGrid[2]);
  const phInputFile = path.join(jobDir, "dfpt_ph.in");
  fs.writeFileSync(phInputFile, phInput);

  // DFPT ph.x on 2×2×2 is much heavier than screening phonon: each of 8
  // q-points × 3N perturbations is a mini-SCF. Budget 8 hours (heavy systems
  // need it), capped below the main phonon 24h cap since DFPT is secondary.
  const DFPT_PH_TIMEOUT_MS = 8 * 3600 * 1000;
  console.log(`[QE-Worker] DFPT EPC: running ph.x for ${formula} (${nqGrid.join("×")} q-grid, P=${pressureGpa} GPa, timeout=8h)`);
  const phResult = await runQECommand(
    path.posix.join(getQEBinDir(), "ph.x"),
    phInputFile,
    jobDir,
    DFPT_PH_TIMEOUT_MS,
  );
  const phOut = path.join(jobDir, "dfpt_ph.out");
  fs.writeFileSync(phOut, phResult!.stdout);

  const phParsed = parseLambdaOutput(phResult!.stdout);
  const phConverged = phResult!.exitCode === 0 && phParsed.lambda > 0;

  if (!phConverged && phParsed.lambda === 0) {
    warnings.push(`ph.x exited ${phResult!.exitCode}; no lambda parsed from stdout`);
  }
  console.log(`[QE-Worker] DFPT ph.x for ${formula}: exit=${phResult!.exitCode}, λ=${phParsed.lambda.toFixed(3)}, ω_log=${phParsed.omegaLog.toFixed(0)} K`);

  // --- q2r.x: build interatomic force constants ---
  let q2rDone = false;
  try {
    const q2rInput = generateQ2RInput(prefix, nqGrid[0], nqGrid[1], nqGrid[2]);
    const q2rFile = path.join(jobDir, "dfpt_q2r.in");
    fs.writeFileSync(q2rFile, q2rInput);
    const POST_PROCESS_TIMEOUT_MS = 5 * 60 * 1000; // 5 min — these are fast tools
    const q2rResult = await runQECommand(
      path.posix.join(getQEBinDir(), "q2r.x"),
      q2rFile,
      jobDir,
      POST_PROCESS_TIMEOUT_MS,
    );
    q2rDone = q2rResult.exitCode === 0;
    if (!q2rDone) warnings.push(`q2r.x exited ${q2rResult.exitCode}`);
    else console.log(`[QE-Worker] DFPT q2r.x done for ${formula}`);
  } catch (err: any) {
    warnings.push(`q2r.x failed: ${(err.message ?? "").slice(0, 100)}`);
  }

  // --- matdyn.x: phonon DOS on fine grid ---
  let matdynDone = false;
  if (q2rDone) {
    try {
      const POST_PROCESS_TIMEOUT_MS = 5 * 60 * 1000;
      const matdynInput = generateMatdynDOSInput(prefix, 20, 20, 20);
      const matdynFile = path.join(jobDir, "dfpt_matdyn.in");
      fs.writeFileSync(matdynFile, matdynInput);
      const matdynResult = await runQECommand(
        path.posix.join(getQEBinDir(), "matdyn.x"),
        matdynFile,
        jobDir,
        POST_PROCESS_TIMEOUT_MS,
      );
      matdynDone = matdynResult.exitCode === 0;
      if (!matdynDone) warnings.push(`matdyn.x exited ${matdynResult.exitCode}`);
      else console.log(`[QE-Worker] DFPT matdyn.x done for ${formula}`);
    } catch (err: any) {
      warnings.push(`matdyn.x failed: ${(err.message ?? "").slice(0, 100)}`);
    }
  }

  // --- Parse a2F and run Eliashberg ---
  const dfptFiles = await tryLoadDFPTResults(jobDir, prefix);
  let tcAllenDynes = 0;
  let tcEliashberg = 0;
  let lambda = phParsed.lambda;
  let omegaLog = phParsed.omegaLog;
  let source: QEDFPTResult["source"] = lambda > 0 ? "ph.x-stdout" : "none";

  if (dfptFiles.alpha2F && dfptFiles.alpha2F.frequencies.length > 0 && !dfptFiles.alpha2F.unstableStructure) {
    source = "a2F-file";
    lambda = dfptFiles.alpha2F.lambda > 0 ? dfptFiles.alpha2F.lambda : lambda;
    omegaLog = dfptFiles.alpha2F.omegaLog > 0 ? dfptFiles.alpha2F.omegaLog : omegaLog;

    // Build a minimal ElectronPhononCoupling from the DFPT-derived values.
    // The surrogate fields (bandwidth, omega2Avg, etc.) are irrelevant here because
    // runEliashbergFromAlpha2FFile only uses lambda and lambdaUncorrected for
    // anharmonic corrections — the spectral Tc comes from the a2F data directly.
    const electronic = computeElectronicStructure(formula);
    const phonon = computePhononSpectrum(formula, electronic);
    const surrogateCoupling = computeElectronPhononCoupling(electronic, phonon, formula, pressureGpa);
    const dfptCoupling: ElectronPhononCoupling = {
      ...surrogateCoupling,
      lambda,
      lambdaUncorrected: lambda,
      omegaLog,
    };

    try {
      const eliashbergResult = runEliashbergFromAlpha2FFile(
        formula,
        pressureGpa,
        { frequencies: dfptFiles.alpha2F.frequencies, values: dfptFiles.alpha2F.alpha2F },
        dfptCoupling,
      );
      tcAllenDynes = eliashbergResult.tcAllenDynes.tc;
      tcEliashberg = eliashbergResult.tcEliashbergGap.tc;
    } catch (elErr: any) {
      warnings.push(`Eliashberg solver failed: ${(elErr.message ?? "").slice(0, 100)}`);
    }
  }

  // Fallback: Allen-Dynes directly from ph.x stdout lambda/omegaLog.
  // omegaLog is already in Kelvin (QE reports "Logarithmic average frequency" in K).
  if (tcAllenDynes === 0 && lambda > 0 && omegaLog > 0) {
    const muStar = 0.10;
    const denom = lambda - muStar * (1 + 0.62 * lambda);
    if (denom > 0) {
      const exp = -1.04 * (1 + lambda) / denom;
      if (exp >= -50) {
        const lambdaBar = 2.46 * (1 + 3.8 * muStar);
        const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 1.5), 1 / 3);
        tcAllenDynes = Number(Math.max(0, Math.min(500, (omegaLog / 1.2) * f1 * Math.exp(exp))).toFixed(2));
      }
    }
    source = "ph.x-stdout";
  }

  const tcBest = Math.max(tcAllenDynes, tcEliashberg);
  console.log(`[QE-Worker] DFPT EPC result for ${formula}: λ=${lambda.toFixed(3)}, ω_log=${omegaLog.toFixed(0)} K, Tc=${tcBest.toFixed(1)} K (source=${source})`);

  // Determine method provenance — only actual DFPT e-ph is physics-grade
  const alpha2FMethod: "dfpt_eph" | "surrogate_eph" | "heuristic_eph" | "unavailable" =
    source === "a2F-file" ? "dfpt_eph" :
    source === "ph.x-stdout" ? "dfpt_eph" : "unavailable";
  const lambdaMethod: "dfpt_integrated_alpha2F" | "surrogate_alpha2F" | "estimated_from_dos_phonons" =
    source === "a2F-file" ? "dfpt_integrated_alpha2F" :
    source === "ph.x-stdout" ? "dfpt_integrated_alpha2F" : "estimated_from_dos_phonons";

  console.log(`[QE-Worker] DFPT method labels for ${formula}: alpha2F=${alpha2FMethod}, lambda=${lambdaMethod}`);

  return {
    lambda: Number(lambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),
    tcAllenDynes: Number(tcAllenDynes.toFixed(2)),
    tcEliashberg: Number(tcEliashberg.toFixed(2)),
    tcBest: Number(tcBest.toFixed(2)),
    nqGrid,
    phConverged,
    q2rDone,
    matdynDone,
    wallTimeSeconds: (Date.now() - t0) / 1000,
    source,
    warnings,
    alpha2FMethod,
    lambdaMethod,
  };
}

/** Build callbacks for staged-relaxation.ts from qe-worker internal functions. */
function buildQERunnerCallbacks(): QERunnerCallbacks {
  return {
    runPwx: (inputFile, workDir, timeoutMs) =>
      runQECommand(path.posix.join(getQEBinDir(), "pw.x"), inputFile, workDir, timeoutMs),
    runPhx: (inputFile, workDir, timeoutMs) =>
      runQECommand(path.posix.join(getQEBinDir(), "ph.x"), inputFile, workDir, timeoutMs),
    getQEBinDir,
    getPseudoDir: () => QE_PSEUDO_DIR,
    getPseudoDirInput: () => QE_PSEUDO_DIR_INPUT,
    cleanTmpDir: cleanQETmpDir,
    resolveEcutwfc: (elements) => {
      const hasH = elements.includes("H");
      const base = computeEcutwfc(elements, 0, 80, 45);
      return Math.max(base, hasH ? 100 : 60);
    },
    resolveEcutrho: (elements, ecutwfc) => ecutwfc * ecutrhoMultiplier(elements),
    resolvePPFilename,
    getAtomicMass,
    autoKPoints: (latticeA, cOverA?, kspacing?) =>
      autoKPoints(latticeA, cOverA, 4, undefined, kspacing ?? DEFAULT_KSPACING),
    hasMagneticElements: (elements) => mayHaveMagneticMoment(elements),
    generateMagnetizationLines: (elements, counts) =>
      generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !elements.some(el => el in MAGNETIC_ELEMENTS)),
    estimateCOverA,
    generateCellParameters,
  };
}

export async function runFullDFT(formula: string, opts?: { startAttempt?: number; pressureGpa?: number; ensembleScore?: number; forceSpin?: boolean; skipEph?: boolean }): Promise<QEFullResult> {
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

  // Pre-flight: reject candidates containing elements we can't compute.
  // Catches two cases: (a) f-block elements whose PPs exceed lmaxx=3, and
  // (b) elements whose PP fetch failed earlier this process lifetime
  // (cooldown-cached). Both previously wasted minutes-to-hours per job.
  const unsupported = getUnsupportedElement(elements);
  if (unsupported) {
    const reason = LMAXX_INCOMPATIBLE.has(unsupported)
      ? `unsupported element ${unsupported} (PPs require lmaxx>3; QE rebuild or scalar-relativistic PP needed)`
      : `unsupported element ${unsupported} (PP fetch failed recently)`;
    result.error = `Pre-filter rejected: ${reason}`;
    result.rejectionReason = reason;
    result.failureStage = "unsupported_element";
    stageFailureCounts.unsupported_element = (stageFailureCounts.unsupported_element ?? 0) + 1;
    console.log(`[QE-Worker] ${formula} rejected: ${reason}`);
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

  // If the job record carries an explicit pressure (e.g. from the candidate's known synthesis
  // pressure), apply it — taking the higher of the two estimates so we never under-compress.
  if (opts?.pressureGpa && opts.pressureGpa > (result.estimatedPressureGPa ?? 0)) {
    result.highPressure = true;
    result.estimatedPressureGPa = opts.pressureGpa;
    console.log(`[QE-Worker] ${formula} pressure overridden by job record: ${opts.pressureGpa} GPa`);
  }

  const jobDir = path.join(QE_WORK_DIR, `job_${Date.now()}_${formula.replace(/[^a-zA-Z0-9]/g, "")}`);
  fs.mkdirSync(path.join(jobDir, "tmp"), { recursive: true });

  try {
    for (const el of elements) {
      try {
        await ensurePseudopotential(el);
      } catch (ppErr: any) {
        markElementPPFailed(el);
        result.error = ppErr.message;
        result.ppValidated = false;
        result.rejectionReason = `PP: ${ppErr.message}`;
        result.failureStage = "pp_validation";
        stageFailureCounts.pp_validation++;
        console.log(`[QE-Worker] PP validation failed for ${formula}: ${ppErr.message} — ${el} blacklisted for ${PP_FAILURE_COOLDOWN_MS / 60000} min`);
        recordFormulaFailure(formula);
        return result;
      }
    }

    result.ppValidated = true;
    const workerPressure = result.estimatedPressureGPa ?? 0;

    // --- Vegard's law enhanced lattice estimation ---
    // Try Vegard interpolation from AFLOW/MP binary endpoints for a better
    // starting lattice. Falls back to the existing volume-sum estimate if
    // insufficient endpoint data is available (< 30s total with caching).
    let vegardResult: VegardEstimate | null = null;
    let structureCandidates: StructureCandidate[] = [];
    try {
      const vegardT0 = Date.now();
      [vegardResult, structureCandidates] = await Promise.all([
        vegardEstimate(elements, counts, workerPressure).catch(() => null),
        generateStructureCandidates(formula, elements, counts, workerPressure, 5).catch(() => []),
      ]);
      const vegardMs = Date.now() - vegardT0;
      if (vegardResult && vegardResult.confidence > 0.2) {
        console.log(`[QE-Worker] Vegard estimate for ${formula}: a=${vegardResult.latticeA.toFixed(3)} A (conf=${vegardResult.confidence.toFixed(2)}, method=${vegardResult.method}, ${vegardResult.endpointsUsed.length} endpoints, ${vegardMs}ms)`);
      }
      if (structureCandidates.length > 0) {
        console.log(`[QE-Worker] ${structureCandidates.length} structure candidates for ${formula}`);
      }
      result.vegardEstimate = vegardResult ? {
        latticeA: vegardResult.latticeA,
        confidence: vegardResult.confidence,
        endpointsUsed: vegardResult.endpointsUsed,
        method: vegardResult.method,
      } : undefined;
      result.structureCandidatesEvaluated = structureCandidates.length;
    } catch (vegardErr: any) {
      console.log(`[QE-Worker] Vegard/candidate generation failed for ${formula}: ${vegardErr.message?.slice(0, 150)}, using volume-sum fallback`);
    }

    // --- Tier assignment ---
    // Determine how many CSP candidates to generate based on the material's
    // characteristics, priority, and prior knowledge.
    const tierDecision = assignTier(
      formula, elements, workerPressure,
      opts?.ensembleScore ? Math.round(opts.ensembleScore * 100) : 50,
      false, // hasCompletedDFT — would need DB lookup
      opts?.forceSpin ? "scf_tsc" : "scf",
    );
    logTierDecision(formula, tierDecision);

    // --- AIRSS structure generation ---
    if (airssEngine.isAvailable()) {
      try {
        const airssDir = path.join(jobDir, "airss");
        const airssCandidates = await airssEngine.generateStructures(elements, counts, {
          binaryPath: "",
          workDir: airssDir,
          timeoutMs: tierDecision.timeoutMs,
          maxStructures: tierDecision.airssBudget,
          pressureGPa: workerPressure,
          baseSeed: Date.now() % 1e8,
        });
        if (airssCandidates.length > 0) {
          // Convert CSPCandidate positions to StructureCandidate format
          for (const ac of airssCandidates) {
            structureCandidates.push({
              latticeA: ac.latticeA,
              latticeB: ac.latticeB,
              latticeC: ac.latticeC,
              cOverA: ac.cOverA,
              positions: ac.positions,
              prototype: "AIRSS-buildcell",
              crystalSystem: ac.crystalSystem,
              spaceGroup: ac.spaceGroup,
              source: ac.source,
              confidence: 0.40,
              isMetallic: null,
            });
          }
          console.log(`[QE-Worker] AIRSS generated ${airssCandidates.length} candidates for ${formula} (total now: ${structureCandidates.length})`);
        }
      } catch (airssErr: any) {
        console.log(`[QE-Worker] AIRSS failed for ${formula}: ${airssErr.message?.slice(0, 100)} — continuing without`);
      }
    }

    // --- PyXtal structure generation ---
    if (pyxtalEngine.isAvailable()) {
      try {
        const pyxtalDir = path.join(jobDir, "pyxtal");
        const pyxtalCandidates = await pyxtalEngine.generateStructures(elements, counts, {
          binaryPath: "",
          workDir: pyxtalDir,
          timeoutMs: tierDecision.timeoutMs,
          maxStructures: tierDecision.pyxtalBudget,
          pressureGPa: workerPressure,
          baseSeed: (Date.now() + 7777) % 1e8,
        });
        if (pyxtalCandidates.length > 0) {
          for (const pc of pyxtalCandidates) {
            structureCandidates.push({
              latticeA: pc.latticeA,
              latticeB: pc.latticeB,
              latticeC: pc.latticeC,
              cOverA: pc.cOverA,
              positions: pc.positions,
              prototype: "PyXtal-random",
              crystalSystem: pc.crystalSystem,
              spaceGroup: pc.spaceGroup,
              source: pc.source,
              confidence: 0.35,
              isMetallic: null,
            });
          }
          console.log(`[QE-Worker] PyXtal generated ${pyxtalCandidates.length} candidates for ${formula} (total now: ${structureCandidates.length})`);
        }
      } catch (pyxtalErr: any) {
        console.log(`[QE-Worker] PyXtal failed for ${formula}: ${pyxtalErr.message?.slice(0, 100)} — continuing without`);
      }
    }

    // --- Cage-aware seeding ---
    // For hydrides, generate candidates from known cage templates using
    // parent seeding (modify existing cage structures) and Wyckoff-aware
    // generation (place H on specific cage-forming orbits like 32f, 12d, 6h).
    if (elements.includes("H")) {
      try {
        const cageCandidates = generateCageSeededCandidates(elements, counts, workerPressure);
        for (const cc of cageCandidates) {
          structureCandidates.push({
            latticeA: cc.latticeA,
            latticeB: cc.latticeB,
            latticeC: cc.latticeC,
            cOverA: cc.cOverA,
            positions: cc.positions,
            prototype: cc.prototype ?? "cage-seed",
            crystalSystem: cc.crystalSystem,
            spaceGroup: cc.spaceGroup,
            source: cc.source,
            confidence: cc.confidence ?? 0.80,
            isMetallic: null,
          });
        }
        if (cageCandidates.length > 0) {
          console.log(`[QE-Worker] Cage seeder: ${cageCandidates.length} cage-aware candidates (total now: ${structureCandidates.length})`);
        }
      } catch (cageErr: any) {
        console.log(`[QE-Worker] Cage seeder failed: ${cageErr.message?.slice(0, 80)}`);
      }
    }

    // --- Structure mutation layer ---
    // Take the best candidates and generate variants through physically
    // motivated perturbations (lattice strain, H shuffle, symmetry break, etc.)
    // This helps VCA-derived structures escape local minima.
    if (structureCandidates.length >= 2) {
      try {
        const hasH = elements.includes("H");
        // Convert StructureCandidates to CSPCandidate-like for the mutator
        const cspLike = structureCandidates.slice(0, 3).map((c, i) => ({
          ...c,
          sourceEngine: "vegard" as const,
          generationStage: 1 as const,
          seed: Date.now() + i,
          pressureGPa: workerPressure,
          relaxationLevel: "raw" as const,
          fingerprint: undefined,
        }));
        const mutants = mutateTopCandidates(cspLike as any, 3, 4, hasH);
        for (const m of mutants) {
          structureCandidates.push({
            latticeA: m.latticeA,
            latticeB: m.latticeB,
            latticeC: m.latticeC,
            cOverA: m.cOverA,
            positions: m.positions,
            prototype: m.prototype ?? "mutant",
            crystalSystem: m.crystalSystem,
            spaceGroup: m.spaceGroup,
            source: m.source,
            confidence: Math.max(0.15, (m.confidence ?? 0.3) * 0.8),
            isMetallic: null,
          });
        }
        if (mutants.length > 0) {
          console.log(`[QE-Worker] Mutation layer: ${mutants.length} variants from top 3 candidates (total now: ${structureCandidates.length})`);
        }
      } catch (mutErr: any) {
        console.log(`[QE-Worker] Mutation failed for ${formula}: ${mutErr.message?.slice(0, 80)} — continuing`);
      }
    }

    // --- Candidate stats logging ---
    try {
      logCandidateStats(structureCandidates as any, formula);
    } catch {}

    // --- Candidate Funnel (F0 → F8) ---
    // Full multi-stage filter: parse → geometry → chemistry → hydride →
    // dedup → score → cluster → DFT admission. Reduces ~85 raw candidates
    // to 3-5 high-quality DFT-worthy structures with exploitation/exploration balance.
    if (structureCandidates.length > 5) {
      try {
        const funnelResult = await runCandidateFunnel(
          structureCandidates as any,
          formula,
          elements,
          workerPressure,
          3, // nDFT fallback (overridden by tier-based budget inside funnel)
          tierDecision.tier,
        );

        if (funnelResult.selected.length > 0) {
          structureCandidates = funnelResult.selected.map(c => ({
            latticeA: c.latticeA,
            latticeB: c.latticeB,
            latticeC: c.latticeC,
            cOverA: c.cOverA,
            positions: c.positions,
            prototype: c.prototype ?? "funnel-selected",
            crystalSystem: c.crystalSystem,
            spaceGroup: c.spaceGroup,
            source: c.source,
            confidence: c.confidence ?? 0.5,
            isMetallic: null,
          }));
        }
      } catch (funnelErr: any) {
        console.log(`[QE-Worker] Funnel failed: ${funnelErr.message?.slice(0, 100)} — using all candidates`);
      }
    }

    // Use Vegard lattice if confident, otherwise fall back to volume-sum
    let latticeA: number;
    if (vegardResult && vegardResult.confidence > 0.3) {
      latticeA = vegardResult.latticeA;
      console.log(`[QE-Worker] Using Vegard lattice for ${formula}: ${latticeA.toFixed(3)} A (conf=${vegardResult.confidence.toFixed(2)})`);
    } else {
      latticeA = estimateLatticeConstant(elements, counts, workerPressure);
      console.log(`[QE-Worker] Using volume-sum lattice for ${formula}: ${latticeA.toFixed(3)} A (Vegard conf=${vegardResult?.confidence?.toFixed(2) ?? "N/A"})`);
    }
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

    // --- Pre-xTB quality gate ---
    // Known structures have DFT-quality Wyckoff positions from literature.
    // xTB can only degrade them (especially at high pressure where the
    // molecular optimizer collapses cage structures). Validate distances
    // at the DFT lattice constant and skip xTB if the geometry is sound.
    let skipXtb = false;
    const knownStruct = formula ? lookupKnownStructure(formula) : null;
    if (knownStruct && positions.length === knownStruct.atoms.length) {
      // Check distances at the ACTUAL DFT lattice (not the compressed xTB lattice)
      const dftLattice = knownStruct.latticeA;
      let minDist = Infinity;
      let minPair = "";
      for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          let dx = positions[i].x - positions[j].x;
          let dy = positions[i].y - positions[j].y;
          let dz = positions[i].z - positions[j].z;
          dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
          const dist = Math.sqrt((dx * dftLattice) ** 2 + (dy * dftLattice) ** 2 + (dz * dftLattice) ** 2);
          if (dist < minDist) { minDist = dist; minPair = `${positions[i].element}-${positions[j].element}`; }
        }
      }
      // At high pressure, real H-H distances can be 0.8-1.2 Å. Use a
      // pressure-scaled threshold: 0.5 Å at 0 GPa, 0.3 Å at 200 GPa.
      const distThreshold = workerPressure > 0
        ? Math.max(0.3, 0.5 - workerPressure * 0.001)
        : 0.5;
      if (minDist >= distThreshold) {
        skipXtb = true;
        result.xtbPreRelaxed = true;
        console.log(`[QE-Worker] Known-structure positions validated for ${formula}: minDist=${minDist.toFixed(3)} Å [${minPair}] at DFT lattice ${dftLattice.toFixed(2)} Å — skipping xTB (literature Wyckoff positions are DFT-quality)`);
      } else {
        console.log(`[QE-Worker] Known-structure distance check: ${minPair}=${minDist.toFixed(3)} Å < ${distThreshold.toFixed(2)} Å at DFT lattice — proceeding to xTB`);
      }
    }

    let relaxed: Array<{ element: string; x: number; y: number; z: number }> | null = null;
    if (!skipXtb) {
      relaxed = tryXTBPreRelaxation(positions, latticeA, jobDir, workerPressure);
    }
    result.xtbPreRelaxed = result.xtbPreRelaxed || !!relaxed;
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

    // xTB stability pre-filter: soft penalty, NOT hard rejection.
    // xTB is unreliable for high-pressure hydrides and unusual compositions —
    // it should not be allowed to kill a DFT-worthy candidate. Instead, apply
    // a confidence penalty that lowers priority in Stage 1 candidate selection.
    // Hard reject ONLY if: formation energy > 2.5 eV/atom AND bad geometry AND
    // no prototype support (cage-seeded, known-structure, high-confidence CSP).
    if (!skipXtb) {
      const stabilityCheck = runXTBStabilityCheck(positions, latticeA, jobDir, workerPressure);
      if (stabilityCheck) {
        if (!stabilityCheck.stable) {
          // Check for hard-reject conditions: ALL THREE must be true
          const hasPrototypeSupport = structureCandidates.some(c =>
            c.prototype === "literature" ||
            c.prototype.startsWith("TemplateVCA") ||
            c.prototype === "known-structure" ||
            c.source?.includes("cage") ||
            (c.confidence ?? 0) >= 0.75
          );
          const hasH = elements.includes("H");
          const isHighPressure = workerPressure > 20;
          const formE = stabilityCheck.formationEnergyEv ?? 0;
          const isSeverelyUnstable = formE > 2.5;

          if (isSeverelyUnstable && !hasPrototypeSupport && !hasH && !isHighPressure) {
            // Hard reject: very unstable by xTB, no prototype support, not a hydride, not high-P
            result.error = `xTB stability pre-filter: ${stabilityCheck.basis}`;
            result.failureStage = "xtb_prefilter";
            result.rejectionReason = `Unstable: ${stabilityCheck.basis}`;
            stageFailureCounts.xtb_prefilter++;
            console.log(`[QE-Worker] ${formula} REJECTED by xTB pre-filter: ${stabilityCheck.basis} (no prototype support, not hydride, not high-P)`);
            return result;
          }

          // Soft penalty: lower confidence on all candidates, but proceed to DFT
          const penalty = stabilityCheck.confidencePenalty;
          for (const sc of structureCandidates) {
            sc.confidence = Math.max(0.05, (sc.confidence ?? 0.5) - penalty);
          }
          console.log(`[QE-Worker] ${formula} xTB stability WARNING (proceeding): ${stabilityCheck.basis}${workerPressure > 0 ? ` @ ${workerPressure} GPa` : ""} — confidence penalty=${penalty.toFixed(2)}, hasPrototype=${hasPrototypeSupport}, isHydride=${hasH}`);
        } else {
          console.log(`[QE-Worker] ${formula} xTB stability OK: ${stabilityCheck.basis}`);
        }
      }
    } else {
      console.log(`[QE-Worker] ${formula} skipping xTB stability check (known-structure validated)`);
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
    result.kPoints = autoKPoints(latticeA, cOverA, 12, protoDimensionality, DEFAULT_KSPACING, { stage: "scf", isMetallic: vegardResult?.isMetallic ?? undefined, totalAtoms: positions.length }).trim();

    // --- Stage 1: Atomic relax (fixed cell, positions only) ---
    // Quick 10-min sanity check: optimizes atomic positions with the cell fixed.
    // Catches badly-placed atoms early before wasting 30 min on vc-relax.
    const normFormula = formula.replace(/\s+/g, "");
    // Auto-populate VERIFIED_LATTICE_A from the known-structures database
    // so any compound in known-structures.ts automatically gets its
    // literature lattice used (no manual sync needed).
    const VERIFIED_LATTICE_A: Record<string, number> = {};
    for (const ksFormula of getKnownStructureFormulas()) {
      const ks = lookupKnownStructure(ksFormula);
      if (ks) VERIFIED_LATTICE_A[ksFormula] = ks.latticeA;
    }
    const isKnownCompound = !!VERIFIED_LATTICE_A[normFormula];

    // For known compounds with no Vegard candidates, create one using
    // the literature lattice + generateAtomicPositions
    if (isKnownCompound && structureCandidates.length === 0) {
      const litA = VERIFIED_LATTICE_A[normFormula];
      const litPositions = generateAtomicPositions(elements, counts, formula, litA);
      structureCandidates.push({
        latticeA: litA,
        positions: litPositions,
        prototype: "literature",
        crystalSystem: "cubic",
        spaceGroup: "",
        source: `Literature lattice (a=${litA} A)`,
        confidence: 0.95,
        isMetallic: null,
      });
    }

    if (structureCandidates.length > 0) {
      // Run staged relaxation (Stage 1 + Stage 2) on structure candidates
      const qeCallbacks = buildQERunnerCallbacks();
      // Don't skip vc-relax for known compounds — literature lattice may not
      // match these pseudopotentials exactly. LaH10 had 17 imaginary phonon modes
      // because the cell was wrong (P=-7.7 kbar). Let DFT find the right cell.
      // Only skip for TSC (nspin=2 oscillation) and all-TM intermetallics (BFGS crash).
      const skipVcRelax = (
        !!opts?.forceSpin ||
        (elements.length >= 3 && elements.every(el => TRANSITION_METALS.has(el)))
      );

      try {
        const stagedResult = await runStagedRelaxation({
          formula,
          elements,
          counts,
          // Replace crude grid/fallback positions with proper
          // generateAtomicPositions() output — uses prototypes, Wyckoff sites,
          // cage placement for hydrides. Keep the Vegard lattice constant.
          // PRESERVE positions from high-quality sources that already have
          // physics-grounded atomic coordinates, BUT validate them first —
          // VCA interpolation can produce overlapping atoms (e.g. YH9Na2
          // Template VCA from C4Sc6Zr2 had atoms #4 and #7 overlap).
          candidates: structureCandidates.map(c => {
            const isQualitySource = c.positions.length > 0 && (
              c.prototype === "MP-direct" ||
              c.prototype.startsWith("TemplateVCA-") ||
              c.prototype === "VCA-interpolated" ||
              c.prototype === "AIRSS-buildcell" ||
              c.prototype === "PyXtal-random"
            );
            // Validate VCA positions: check no atoms overlap at the candidate lattice
            let vcaValid = isQualitySource;
            if (isQualitySource && c.prototype !== "MP-direct") {
              const cLat = c.latticeA || 5.0;
              for (let i = 0; i < c.positions.length && vcaValid; i++) {
                for (let j = i + 1; j < c.positions.length; j++) {
                  let dx = c.positions[i].x - c.positions[j].x;
                  let dy = c.positions[i].y - c.positions[j].y;
                  let dz = c.positions[i].z - c.positions[j].z;
                  dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
                  const dist = Math.sqrt((dx * cLat) ** 2 + (dy * cLat) ** 2 + (dz * cLat) ** 2);
                  if (dist < 0.3) {
                    console.log(`[QE-Worker] VCA positions rejected for ${formula} (${c.prototype}): atoms ${i} and ${j} overlap at ${dist.toFixed(3)} Å — falling back to generateAtomicPositions`);
                    vcaValid = false;
                    break;
                  }
                }
              }
            }
            return {
              ...c,
              positions: vcaValid
                ? c.positions
                : generateAtomicPositions(elements, counts, formula, c.latticeA),
            };
          }),
          pressureGPa: workerPressure,
          jobDir,
          callbacks: qeCallbacks,
          skipVcRelax,  // Only TSC and all-TM skip vc-relax. Known compounds run vc-relax from literature lattice.
          skipRelaxation: false,
          // Test ALL admitted candidates from F8, not just top 3.
          // The funnel already selected the best — don't throw any away before DFT.
          maxStage1Candidates: structureCandidates.length,
          isMetallic: vegardResult?.isMetallic ?? undefined,
          screeningTier: tierDecision.tier as "preview" | "standard" | "deep" | "publication",
        });

        result.stagedRelaxation = {
          stages: stagedResult.stages.map(s => ({
            stage: s.stage,
            passed: s.passed,
            failReason: s.failReason,
            totalEnergy: s.totalEnergy,
            maxForce: s.maxForce,
            wallTimeSeconds: s.wallTimeSeconds,
          })),
          finalStage: stagedResult.finalStage,
          success: stagedResult.success,
          totalWallTime: stagedResult.totalWallTime,
        };

        // Use staged results if any stage passed
        if (stagedResult.bestPositions.length > 0 && stagedResult.stages.some(s => s.passed)) {
          positions = stagedResult.bestPositions;
          latticeA = stagedResult.bestLatticeA;
          console.log(`[QE-Worker] Staged relaxation improved geometry for ${formula}: a=${latticeA.toFixed(3)} A, source=${stagedResult.candidateSource}`);
        } else {
          console.log(`[QE-Worker] Staged relaxation did not improve geometry for ${formula}, using pre-staged positions`);
        }
      } catch (stagedErr: any) {
        console.log(`[QE-Worker] Staged relaxation failed for ${formula}: ${stagedErr.message?.slice(0, 200)}, continuing with existing flow`);
      }
    }

    result.vcRelaxed = false;
    let preVcLatticeA = latticeA;

    // Use literature lattice as STARTING POINT for vc-relax, not as final.
    // The literature value gives a good initial guess, but DFT with these
    // specific pseudopotentials may have a slightly different equilibrium.
    // LaH10 had P=-7.7 kbar and 17 imaginary modes when vc-relax was skipped.
    if (VERIFIED_LATTICE_A[normFormula]) {
      const litA = VERIFIED_LATTICE_A[normFormula];
      console.log(`[QE-Worker] Using literature lattice a=${litA} Å as starting point for ${formula} (vc-relax will refine)`);
      latticeA = litA;
      // Do NOT set result.vcRelaxed = true — let vc-relax run
    }

    // Log vc-relax intent for all materials
    if (!result.vcRelaxed && workerPressure >= 100 && elements.includes("H") && elements.length >= 2) {
      console.log(`[QE-Worker] Running vc-relax for ${formula} — high-P hydride (P=${workerPressure} GPa), vc-relax needed to find correct cell${isKnownCompound ? " (starting from literature lattice)" : ""}`);
    }

    // Skip vc-relax for TSC candidates. With forceSpin=true we emit
    // nspin=2, and vc-relax on heavy 5d/5p elements (Bi, Sb, Te, W, Pb,
    // Hg, Ta, Au, Pt) with two spin channels oscillates between spin
    // configurations and dies on wall time (CuSbTe3 in the Apr-15 p2 run:
    // 1h28m, exit=2, no geometry). The TSC analysis uses the prototype
    // geometry anyway — vc-relax buys nothing.
    if (!result.vcRelaxed && opts?.forceSpin) {
      console.log(`[QE-Worker] Skipping vc-relax for ${formula} — TSC candidate (forceSpin=true); nspin=2 vc-relax on heavy elements oscillates indefinitely, proceeding with xTB geometry (saves ~90 min)`);
      result.vcRelaxed = true;
    }

    // Skip vc-relax for all-TM intermetallics (3+ elements, every element
    // a transition metal). Apr-16 run: Mo2Nb3Ti hit wall-time at 1h28m
    // with no geometry. Multi-TM cells have dense d-band cross-talk during
    // cell optimisation and BFGS destabilises long before convergence.
    // The prototype or xTB-relaxed lattice is a fine starting point for
    // SCF — the TM-TM bond lengths don't shift much even in the fully
    // relaxed solution.
    if (!result.vcRelaxed && elements.length >= 3 && elements.every(el => TRANSITION_METALS.has(el))) {
      console.log(`[QE-Worker] Skipping vc-relax for ${formula} — all-TM intermetallic (${elements.length} TMs: ${elements.join(",")}); BFGS destabilises, proceeding with xTB geometry (saves ~90 min)`);
      result.vcRelaxed = true;
    }

    // --- Z-mismatch guard ---
    // If Stage 1 selected a supercell candidate (e.g. Z=4, 12 atoms) but the
    // literature lattice is for the primitive cell (e.g. Z=1, 3 atoms), we must
    // regenerate positions for the correct atom count. Otherwise iterative
    // rescaling will compress 12 atoms into a cell meant for 3 → QE "too few bands".
    const expectedAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
    if (positions.length !== expectedAtoms && isKnownCompound) {
      const Z = Math.round(positions.length / expectedAtoms);
      console.log(`[QE-Worker] Z-mismatch for ${formula}: Stage 1 winner has ${positions.length} atoms (Z=${Z}) but formula expects ${expectedAtoms} — regenerating primitive cell positions`);
      // Look up the known structure positions for the primitive cell
      const ksLookup = lookupKnownStructure(normFormula);
      if (ksLookup && ksLookup.atoms.length === expectedAtoms) {
        positions = ksLookup.atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z }));
        console.log(`[QE-Worker] Using known-structure positions for ${formula} (${expectedAtoms} atoms, ${ksLookup.spaceGroup})`);
      } else {
        positions = generateAtomicPositions(elements, counts, formula, latticeA);
        console.log(`[QE-Worker] Regenerated positions for ${formula} (${positions.length} atoms) at a=${latticeA.toFixed(3)} Å`);
      }
      // CRITICAL: update preVcLatticeA to the target lattice. The regenerated
      // positions are already correct for `latticeA` (the literature value), so
      // iterative rescaling should see 0% shift and skip. Without this, the
      // rescaling loops from the old supercell lattice (e.g. 8.744 Å) to the
      // target (5.290 Å), running pointless steps at wrong lattice parameters
      // where the new positions can't converge ("no positions extracted").
      preVcLatticeA = latticeA;
      console.log(`[QE-Worker] Z-mismatch: reset preVcLatticeA to ${latticeA.toFixed(3)} Å (positions already at target lattice, no rescaling needed)`);
    }

    // --- Lattice-mismatch guard for known compounds ---
    // If the Stage 1 winner's lattice is significantly different from the
    // literature lattice, the Stage 1 positions (optimized at a different
    // lattice) are wrong for the target cell. Use the known-structure
    // positions directly — they're already correct for the literature lattice.
    // LaH11Li2 hit force=1.56 Ry/bohr because Stage 1 positions (a=4.747)
    // were used at a=5.100 — a 7.4% mismatch that iterative rescaling
    // couldn't fix in one step.
    if (isKnownCompound && result.vcRelaxed) {
      const stageLatticeShift = Math.abs(latticeA - preVcLatticeA) / Math.max(1e-6, preVcLatticeA);
      if (stageLatticeShift > 0.05) {
        const ksLookup = lookupKnownStructure(normFormula);
        if (ksLookup) {
          console.log(`[QE-Worker] Lattice-mismatch guard for ${formula}: Stage 1 positions at a=${preVcLatticeA.toFixed(3)} Å but literature is a=${latticeA.toFixed(3)} Å (${(stageLatticeShift * 100).toFixed(1)}% shift) — using known-structure positions + quick relax`);
          positions = ksLookup.atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z }));
          preVcLatticeA = latticeA;

          // Quick fixed-cell relax at the target lattice to bring forces down.
          // Ideal Wyckoff positions are crystallographically correct but haven't
          // been DFT-relaxed with these pseudopotentials. Without this step,
          // SCF runs with high residual forces (0.3+ Ry/bohr) and phonon crashes.
          try {
            const guardPrefix = formula.replace(/[^a-zA-Z0-9]/g, "") + "_guardrelax";
            const guardHasH = elements.includes("H");
            const guardEcutwfc = Math.max(computeEcutwfc(elements, 0, 80, 45), guardHasH ? 80 : 50);
            const guardEcutrho = guardEcutwfc * ecutrhoMultiplier(elements);
            const guardCOverA = estimateCOverA(elements, counts);
            const guardBOverA = estimateBOverA(elements, counts);
            const guardHasMag = mayHaveMagneticMoment(elements);
            const guardNspin = guardHasMag ? 2 : 1;
            const guardMagLines = guardHasMag ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !elements.some(el => el in MAGNETIC_ELEMENTS)) : "";
            let guardSpecies = "";
            for (const el of elements) {
              guardSpecies += `  ${el}  ${getAtomicMass(el).toFixed(3)}  ${resolvePPFilename(el)}\n`;
            }
            let guardPos = "";
            for (const p of positions) {
              guardPos += `  ${p.element}  ${p.x.toFixed(6)}  ${p.y.toFixed(6)}  ${p.z.toFixed(6)}\n`;
            }
            const guardKpts = autoKPoints(latticeA, guardCOverA, guardBOverA, undefined, 0.50, { stage: "relax", totalAtoms: positions.length }).trim();
            const guardCell = generateCellParameters(latticeA, guardCOverA, 0, guardBOverA, elements, counts);
            const guardInput = `&CONTROL
  calculation = 'relax',
  restart_mode = 'from_scratch',
  prefix = '${guardPrefix}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  forc_conv_thr = 1.0d-3,
  nstep = 100,
  max_seconds = 1200,
/
&SYSTEM
  ibrav = 0,
  nat = ${positions.length},
  ntyp = ${elements.length},
  ecutwfc = ${guardEcutwfc},
  ecutrho = ${guardEcutrho},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.015,
  nspin = ${guardNspin},
${guardMagLines}/
&ELECTRONS
  electron_maxstep = 200,
  conv_thr = 1.0d-6,
  mixing_beta = 0.3,
  mixing_mode = 'local-TF',
/
&IONS
  ion_dynamics = 'bfgs',
/
ATOMIC_SPECIES
${guardSpecies}
ATOMIC_POSITIONS {crystal}
${guardPos}
K_POINTS {automatic}
${guardKpts}

${guardCell}
`;
            const guardFile = path.join(jobDir, "guard_relax.in");
            fs.writeFileSync(guardFile, guardInput);
            console.log(`[QE-Worker] Running quick relax after lattice-mismatch guard for ${formula} (a=${latticeA.toFixed(3)} Å, ${positions.length} atoms, 20 min max)`);
            const guardResult = await runQECommand(
              path.posix.join(getQEBinDir(), "pw.x"), guardFile, jobDir, 1260000, // 21 min
            );
            fs.writeFileSync(path.join(jobDir, "guard_relax.out"), guardResult.stdout);

            // Parse relaxed positions
            const guardPosBlocks = guardResult.stdout.match(/ATOMIC_POSITIONS\s*[{(]?\s*(?:crystal|angstrom|bohr|alat)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:CELL_PARAMETERS|K_POINTS|End final|End of|ATOMIC_SPECIES|\n\s*\n)|$)/gi);
            if (guardPosBlocks && guardPosBlocks.length > 0) {
              const lastBlock = guardPosBlocks[guardPosBlocks.length - 1];
              const lines = lastBlock.split("\n").filter(l => l.trim().length > 0 && !l.match(/ATOMIC_POSITIONS/i));
              const relaxedPositions: typeof positions = [];
              for (const line of lines) {
                const parts = line.trim().split(/\s+/);
                if (parts.length >= 4) {
                  relaxedPositions.push({ element: parts[0], x: parseFloat(parts[1]), y: parseFloat(parts[2]), z: parseFloat(parts[3]) });
                }
              }
              if (relaxedPositions.length === positions.length) {
                positions = relaxedPositions;
                const forceMatch = guardResult.stdout.match(/Total force\s*=\s*([\d.]+)/g);
                const lastForce = forceMatch ? forceMatch[forceMatch.length - 1].match(/([\d.]+)$/)?.[1] : null;
                console.log(`[QE-Worker] Guard relax done for ${formula}: force=${lastForce ?? "N/A"} Ry/bohr (${relaxedPositions.length} atoms relaxed)`);
              }
            }
            // Clean guard relax tmp
            const guardSaveDir = path.join(jobDir, "tmp", `${guardPrefix}.save`);
            try { if (fs.existsSync(guardSaveDir)) fs.rmSync(guardSaveDir, { recursive: true, force: true }); } catch {}
          } catch (guardErr: any) {
            console.log(`[QE-Worker] Guard relax failed for ${formula}: ${guardErr.message?.slice(0, 100)} — proceeding with ideal Wyckoff positions`);
          }
        }
      }
    }

    // When vc-relax is skipped and the lattice was rescaled significantly
    // (e.g., from Vegard estimate to literature value), atomic positions are
    // wrong for the new cell. Run a quick fixed-cell relax to bring forces
    // down before SCF + phonon. Without this, ph.x crashes with
    // IEEE_UNDERFLOW_FLAG because DFPT needs forces < 0.05 Ry/bohr.
    const latticeShiftPct = Math.abs(latticeA - preVcLatticeA) / Math.max(1e-6, preVcLatticeA);
    if (result.vcRelaxed && latticeShiftPct > 0.03 && positions.length > 0) {
      // --- Iterative lattice rescaling ---
      // Instead of jumping from 3.557 → 5.1 Å in one step (43% — too much
      // for positions to follow), do intermediate steps of max ~10% each,
      // relaxing positions at each step. This lets the structure adjust
      // gradually to the target lattice.
      const startA = preVcLatticeA;
      const targetA = latticeA;
      // Smaller steps for larger shifts — atoms need gentler transitions
      const MAX_STEP_PCT = latticeShiftPct > 0.40 ? 0.06 :  // 40-50%: 6% per step (~7-8 steps)
                           latticeShiftPct > 0.20 ? 0.08 :  // 20-40%: 8% per step (~3-5 steps)
                           0.10;                            // <20%: 10% per step (1-2 steps)

      // Calculate intermediate lattice constants
      const nSteps = Math.max(1, Math.ceil(latticeShiftPct / MAX_STEP_PCT));
      const stepLattices: number[] = [];
      for (let s = 1; s <= nSteps; s++) {
        const frac = s / nSteps;
        stepLattices.push(startA + (targetA - startA) * frac);
      }

      console.log(`[QE-Worker] Iterative rescaling for ${formula}: ${(latticeShiftPct * 100).toFixed(1)}% in ${nSteps} steps (${startA.toFixed(3)} → ${stepLattices.map(a => a.toFixed(3)).join(" → ")} Å)`);

      // Common QE parameters for all steps
      const relaxPrefix = formula.replace(/[^a-zA-Z0-9]/g, "") + "_fixcell";
      const hasHRelax = elements.includes("H");
      const ecutwfcRelax = Math.max(computeEcutwfc(elements, 0, 80, 45), hasHRelax ? 80 : 50);
      const ecutrhoRelax = ecutwfcRelax * ecutrhoMultiplier(elements);
      const cOverARelax = estimateCOverA(elements, counts);
      const bOverARelax = estimateBOverA(elements, counts);
      const hasMagRelax = mayHaveMagneticMoment(elements);
      const nspinRelax = hasMagRelax ? 2 : 1;
      const magLinesRelax = hasMagRelax ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !elements.some(el => el in MAGNETIC_ELEMENTS)) : "";
      let atomicSpeciesRelax = "";
      for (const el of elements) {
        atomicSpeciesRelax += `  ${el}  ${getAtomicMass(el).toFixed(3)}  ${resolvePPFilename(el)}\n`;
      }

      // Run each step
      for (let step = 0; step < stepLattices.length; step++) {
        const stepA = stepLattices[step];
        const stepPct = Math.abs(stepA - (step === 0 ? startA : stepLattices[step - 1])) / (step === 0 ? startA : stepLattices[step - 1]);
        // More time for larger total shifts and the final step
        const isLastStep = step === stepLattices.length - 1;
        const isLargeShift = latticeShiftPct > 0.40;
        const stepMaxSec = isLastStep ? (isLargeShift ? 2400 : 1800) :  // last: 40 min (large) / 30 min
                           isLargeShift ? 1200 : 900;                   // intermediate: 20 min (large) / 15 min
        const stepNstep = isLastStep ? (isLargeShift ? 250 : 150) :     // last: 250 (large) / 150 steps
                          isLargeShift ? 120 : 80;                      // intermediate: 120 (large) / 80 steps

        let atomicPosRelax = "";
        for (const pos of positions) {
          atomicPosRelax += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
        }
        const kptsRelax = autoKPoints(stepA, cOverARelax, bOverARelax, undefined, 0.6, { stage: "relax", totalAtoms: positions.length }).trim();
        const cellBlockRelax = generateCellParameters(stepA, cOverARelax, 0, bOverARelax, elements, counts);

        const relaxInput = `&CONTROL
  calculation = 'relax',
  restart_mode = 'from_scratch',
  prefix = '${relaxPrefix}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = 1.0d-3,
  etot_conv_thr = 1.0d-4,
  nstep = ${stepNstep},
  max_seconds = ${stepMaxSec},
/
&SYSTEM
  ibrav = 0,
  nat = ${positions.length},
  ntyp = ${elements.length},
  ecutwfc = ${ecutwfcRelax},
  ecutrho = ${ecutrhoRelax},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.015,
  nspin = ${nspinRelax},
${magLinesRelax}/
&ELECTRONS
  electron_maxstep = 200,
  conv_thr = 1.0d-4,
  mixing_beta = 0.4,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
  scf_must_converge = .false.,
/
&IONS
  ion_dynamics = 'bfgs',
/
ATOMIC_SPECIES
${atomicSpeciesRelax}
ATOMIC_POSITIONS {crystal}
${atomicPosRelax}
K_POINTS {automatic}
${kptsRelax}

${cellBlockRelax}
`;
        try {
          const relaxFile = path.join(jobDir, `fixcell_relax_step${step}.in`);
          fs.writeFileSync(relaxFile, relaxInput);
          const relaxResult = await runQECommand(
            path.posix.join(getQEBinDir(), "pw.x"),
            relaxFile, jobDir, (stepMaxSec + 60) * 1000,
          );
          fs.writeFileSync(path.join(jobDir, `fixcell_relax_step${step}.out`), relaxResult.stdout);

          // Parse relaxed positions
          const posBlocks = relaxResult.stdout.match(/ATOMIC_POSITIONS\s*[{(]?\s*(?:crystal|angstrom|bohr|alat)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:CELL_PARAMETERS|K_POINTS|End final|End of|ATOMIC_SPECIES|\n\s*\n)|$)/gi);
          if (posBlocks && posBlocks.length > 0) {
            const lastBlock = posBlocks[posBlocks.length - 1];
            const parsedPositions: Array<{ element: string; x: number; y: number; z: number }> = [];
            for (const line of lastBlock.split("\n").slice(1)) {
              const m = line.trim().match(/^([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
              if (m) parsedPositions.push({ element: m[1], x: parseFloat(m[2]), y: parseFloat(m[3]), z: parseFloat(m[4]) });
            }
            if (parsedPositions.length === positions.length) {
              positions = parsedPositions;
              const forceMatch = relaxResult.stdout.match(/Total force\s*=\s*([\d.]+)/g);
              const lastForce = forceMatch ? forceMatch[forceMatch.length - 1].match(/([\d.]+)/) : null;
              const resForce = lastForce ? parseFloat(lastForce[0]) : null;
              console.log(`[QE-Worker] Iterative relax step ${step + 1}/${nSteps} for ${formula}: a=${stepA.toFixed(3)} Å, force=${resForce?.toFixed(4) ?? "?"} Ry/bohr`);
            } else {
              console.log(`[QE-Worker] Iterative relax step ${step + 1}/${nSteps}: position count mismatch — keeping previous positions`);
            }
          } else {
            console.log(`[QE-Worker] Iterative relax step ${step + 1}/${nSteps}: no positions extracted — keeping previous`);
          }
          cleanQETmpDir(path.join(jobDir, "tmp"));
        } catch (stepErr: any) {
          console.log(`[QE-Worker] Iterative relax step ${step + 1}/${nSteps} failed: ${(stepErr.message || "").slice(0, 100)} — continuing with current positions`);
          cleanQETmpDir(path.join(jobDir, "tmp"));
        }
      }
    }

    // --- Mini-EOS pressure correction ---
    // When vc-relax is skipped for high-P hydrides, the lattice is from
    // literature or Vegard — not optimized at the target pressure. Run a
    // mini equation-of-state: 5 volume points, short static SCF at each,
    // pick the one with stress closest to the target pressure.
    if (result.vcRelaxed && workerPressure > 0 && positions.length > 0 && positions.length <= 16) {
      try {
        const eosScales = [0.96, 0.98, 1.00, 1.02, 1.04];
        const eosPrefix = formula.replace(/[^a-zA-Z0-9]/g, "") + "_eos";
        const hasHEos = elements.includes("H");
        const ecutwfcEos = Math.max(computeEcutwfc(elements, 0, 80, 45), hasHEos ? 80 : 50);
        const ecutrhoEos = ecutwfcEos * ecutrhoMultiplier(elements);
        const cOverAEos = estimateCOverA(elements, counts);
        const bOverAEos = estimateBOverA(elements, counts);
        const hasMagEos = mayHaveMagneticMoment(elements);
        const nspinEos = hasMagEos ? 2 : 1;
        const magLinesEos = hasMagEos ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !elements.some(el => el in MAGNETIC_ELEMENTS)) : "";

        let atomicSpeciesEos = "";
        for (const el of elements) {
          atomicSpeciesEos += `  ${el}  ${getAtomicMass(el).toFixed(3)}  ${resolvePPFilename(el)}\n`;
        }
        let atomicPosEos = "";
        for (const pos of positions) {
          atomicPosEos += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
        }

        const targetKbar = workerPressure * 10;
        let bestScale = 1.0;
        let bestPressDiff = Infinity;

        console.log(`[QE-Worker] Mini-EOS for ${formula}: testing ${eosScales.length} volume points around a=${latticeA.toFixed(3)} Å (target P=${workerPressure} GPa)`);

        for (const scale of eosScales) {
          const eosA = latticeA * Math.pow(scale, 1 / 3);
          const cellBlockEos = generateCellParameters(eosA, cOverAEos, 0, bOverAEos, elements, counts);
          const kptsEos = autoKPoints(eosA, cOverAEos, bOverAEos, undefined, 0.6, { stage: "relax", totalAtoms: positions.length }).trim();
          const eosInput = `&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '${eosPrefix}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  max_seconds = 300,
/
&SYSTEM
  ibrav = 0,
  nat = ${positions.length},
  ntyp = ${elements.length},
  ecutwfc = ${ecutwfcEos},
  ecutrho = ${ecutrhoEos},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.015,
  nspin = ${nspinEos},
${magLinesEos}/
&ELECTRONS
  electron_maxstep = 100,
  conv_thr = 1.0d-4,
  mixing_beta = 0.4,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
  scf_must_converge = .false.,
/
ATOMIC_SPECIES
${atomicSpeciesEos}
ATOMIC_POSITIONS {crystal}
${atomicPosEos}
K_POINTS {automatic}
${kptsEos}

${cellBlockEos}
`;
          const eosFile = path.join(jobDir, `eos_${scale.toFixed(2)}.in`);
          fs.writeFileSync(eosFile, eosInput);
          try {
            const eosResult = await runQECommand(
              path.posix.join(getQEBinDir(), "pw.x"), eosFile, jobDir, 360000,
            );
            // Parse pressure from output
            const pressMatch = eosResult.stdout.match(/total\s+stress.*\n.*P=\s*([-\d.]+)/);
            if (pressMatch) {
              const pKbar = parseFloat(pressMatch[1]);
              const diff = Math.abs(pKbar - targetKbar);
              if (diff < bestPressDiff) {
                bestPressDiff = diff;
                bestScale = scale;
              }
              console.log(`[QE-Worker] Mini-EOS scale=${scale.toFixed(2)}: a=${eosA.toFixed(3)} Å, P=${(pKbar / 10).toFixed(1)} GPa (target ${workerPressure}), diff=${(diff / 10).toFixed(1)} GPa`);
            }
          } catch {}
          cleanQETmpDir(path.join(jobDir, "tmp"));
          try { fs.unlinkSync(eosFile); } catch {}
        }

        if (bestScale !== 1.0) {
          const oldA = latticeA;
          latticeA = latticeA * Math.pow(bestScale, 1 / 3);
          result.relaxedLatticeA = latticeA;
          console.log(`[QE-Worker] Mini-EOS selected scale=${bestScale.toFixed(2)}: a=${oldA.toFixed(3)} → ${latticeA.toFixed(3)} Å (best pressure match, diff=${(bestPressDiff / 10).toFixed(1)} GPa)`);
        } else {
          console.log(`[QE-Worker] Mini-EOS: a=${latticeA.toFixed(3)} Å unchanged (scale=1.00 was best)`);
        }
      } catch (eosErr: any) {
        console.log(`[QE-Worker] Mini-EOS failed for ${formula}: ${eosErr.message?.slice(0, 100)} — continuing with current lattice`);
      }
    }

    try {
      if (result.vcRelaxed) {
        // Already set from verified-compound shortcut, skip the actual run
      } else {
      // --- 2-PHASE vc-relax: Damped Dynamics (Phase 1) → BFGS (Phase 2) ---
      // Phase 1: Damped dynamics finds the basin (robust, never diverges)
      // Phase 2: BFGS tightens from the damped result (fast convergence)
      const hasHVcRelax = elements.includes("H");
      const isHighPHVcRelax = hasHVcRelax && workerPressure >= 50 && positions.length >= 7;
      const hasMagVcRelax = elements.some(el => el in MAGNETIC_ELEMENTS);
      const vcRelaxMaxSec = isHighPHVcRelax ? 5400 : hasMagVcRelax ? 3600 : 1800;
      const vcRelaxKillMs = vcRelaxMaxSec * 1000 + 60_000;

      console.log(`[QE-Worker] 2-phase vc-relax for ${formula} (lattice=${latticeA.toFixed(2)} A, ${positions.length} atoms${workerPressure > 0 ? `, P=${workerPressure} GPa` : ""}, timeout=${vcRelaxMaxSec}s)`);

      // === PHASE 1: Damped dynamics — find the basin ===
      console.log(`[QE-Worker] Phase 1 (damped dynamics): finding basin for ${formula}`);
      const phase1Input = generateVCRelaxInput(formula, elements, counts, latticeA, positions, workerPressure, { phase: "damped" });
      const phase1File = path.join(jobDir, "vc_relax_phase1.in");
      fs.writeFileSync(phase1File, phase1Input);

      const phase1Result = await runQECommand(
        path.posix.join(getQEBinDir(), "pw.x"), phase1File, jobDir, vcRelaxKillMs,
      );
      fs.writeFileSync(path.join(jobDir, "vc_relax_phase1.out"), phase1Result.stdout);
      const phase1Parsed = parseVCRelaxOutput(phase1Result.stdout);

      // --- Phase 1 diagnostic logging ---
      let phase1Positions = positions;
      let phase1LatticeA = latticeA;
      if (phase1Parsed.finalPositions && phase1Parsed.finalPositions.length > 0) {
        phase1Positions = phase1Parsed.finalPositions;
        phase1LatticeA = phase1Parsed.finalLatticeAng ?? latticeA;
        // Compute max force from phase 1 output
        const p1ForceMatch = phase1Result.stdout.match(/Total force\s*=\s*([\d.]+)/g);
        const p1LastForce = p1ForceMatch ? p1ForceMatch[p1ForceMatch.length - 1].match(/([\d.]+)$/)?.[1] : null;
        const p1PressMatch = phase1Result.stdout.match(/P=\s*([-\d.]+)/g);
        const p1LastPress = p1PressMatch ? p1PressMatch[p1PressMatch.length - 1].match(/([-\d.]+)$/)?.[1] : null;
        console.log(`[QE-Worker] Phase 1 DONE for ${formula}: a=${phase1LatticeA.toFixed(3)} Å, E=${phase1Parsed.totalEnergy.toFixed(4)} eV, force=${p1LastForce ?? "N/A"} Ry/bohr, P=${p1LastPress ?? "N/A"} kbar, wall=${phase1Parsed.wallTimeSeconds.toFixed(0)}s`);
        // Log first few positions for diagnosis
        for (let pi = 0; pi < Math.min(4, phase1Positions.length); pi++) {
          const p = phase1Positions[pi];
          console.log(`[QE-Worker]   Phase 1 atom[${pi}] ${p.element.padEnd(2)} (${p.x.toFixed(5)}, ${p.y.toFixed(5)}, ${p.z.toFixed(5)})`);
        }
        if (phase1Positions.length > 4) console.log(`[QE-Worker]   ... and ${phase1Positions.length - 4} more`);
      } else {
        console.log(`[QE-Worker] Phase 1 produced no positions for ${formula} (exit=${phase1Result.exitCode}) — Phase 2 will use original geometry`);
        // Diagnostic: check if output has ATOMIC_POSITIONS but parser missed them
        const hasAP = phase1Result.stdout.includes("ATOMIC_POSITIONS");
        const hasCP = phase1Result.stdout.includes("CELL_PARAMETERS");
        console.log(`[QE-Worker] Phase 1 parse diagnostic: hasATOMIC_POSITIONS=${hasAP}, hasCELL_PARAMETERS=${hasCP}, stdout_len=${phase1Result.stdout.length}`);
        if (phase1Result.exitCode !== 0 || !hasAP) {
          console.log(`[QE-Worker] Phase 1 stdout tail: ${phase1Result.stdout.slice(-300)}`);
        }
      }

      cleanQETmpDir(path.join(jobDir, "tmp"));

      // === PHASE 2: BFGS from damped result — tighten convergence ===
      console.log(`[QE-Worker] Phase 2 (BFGS): tightening from Phase 1 result for ${formula} (a=${phase1LatticeA.toFixed(3)} Å)`);
      const phase2Input = generateVCRelaxInput(formula, elements, counts, phase1LatticeA, phase1Positions, workerPressure, { phase: "bfgs" });
      const phase2File = path.join(jobDir, "vc_relax_phase2.in");
      fs.writeFileSync(phase2File, phase2Input);

      const phase2Result = await runQECommand(
        path.posix.join(getQEBinDir(), "pw.x"), phase2File, jobDir, vcRelaxKillMs,
      );
      fs.writeFileSync(path.join(jobDir, "vc_relax_phase2.out"), phase2Result.stdout);
      const vcParsed = parseVCRelaxOutput(phase2Result.stdout);

      // --- Phase 2 diagnostic logging ---
      if (vcParsed.finalPositions && vcParsed.finalPositions.length > 0) {
        const p2ForceMatch = phase2Result.stdout.match(/Total force\s*=\s*([\d.]+)/g);
        const p2LastForce = p2ForceMatch ? p2ForceMatch[p2ForceMatch.length - 1].match(/([\d.]+)$/)?.[1] : null;
        const p2PressMatch = phase2Result.stdout.match(/P=\s*([-\d.]+)/g);
        const p2LastPress = p2PressMatch ? p2PressMatch[p2PressMatch.length - 1].match(/([-\d.]+)$/)?.[1] : null;
        console.log(`[QE-Worker] Phase 2 DONE for ${formula}: a=${(vcParsed.finalLatticeAng ?? phase1LatticeA).toFixed(3)} Å, E=${vcParsed.totalEnergy.toFixed(4)} eV, force=${p2LastForce ?? "N/A"} Ry/bohr, P=${p2LastPress ?? "N/A"} kbar, wall=${vcParsed.wallTimeSeconds.toFixed(0)}s, converged=${vcParsed.converged}`);
        for (let pi = 0; pi < Math.min(4, vcParsed.finalPositions.length); pi++) {
          const p = vcParsed.finalPositions[pi];
          console.log(`[QE-Worker]   Phase 2 atom[${pi}] ${p.element.padEnd(2)} (${p.x.toFixed(5)}, ${p.y.toFixed(5)}, ${p.z.toFixed(5)})`);
        }
        if (vcParsed.finalPositions.length > 4) console.log(`[QE-Worker]   ... and ${vcParsed.finalPositions.length - 4} more`);
      }

      // Use Phase 2 result if available, otherwise Phase 1
      if (phase2Result.exitCode !== 0 && !vcParsed.finalPositions) {
        console.log(`[QE-Worker] Phase 2 exit=${phase2Result.exitCode} for ${formula}: ${phase2Result.stdout.slice(-200)}`);
      }
      if (vcParsed.finalPositions && vcParsed.finalPositions.length > 0) {
        positions = vcParsed.finalPositions;
        result.vcRelaxed = true;
        if (vcParsed.finalLatticeAng && vcParsed.finalLatticeAng > 0.5) {
          latticeA = vcParsed.finalLatticeAng;
          result.relaxedLatticeA = latticeA;
        }
        console.log(`[QE-Worker] vc-relax 2-phase ${vcParsed.converged ? "CONVERGED" : "partial"} for ${formula}: a=${latticeA.toFixed(3)} A, ${positions.length} atoms, E=${vcParsed.totalEnergy.toFixed(4)} eV`);
      } else if (phase1Parsed.finalPositions && phase1Parsed.finalPositions.length > 0) {
        // Phase 2 failed but Phase 1 produced something — use Phase 1
        positions = phase1Positions;
        latticeA = phase1LatticeA;
        result.vcRelaxed = true;
        result.relaxedLatticeA = latticeA;
        console.log(`[QE-Worker] Phase 2 failed, using Phase 1 geometry for ${formula}: a=${latticeA.toFixed(3)} A, E=${phase1Parsed.totalEnergy.toFixed(4)} eV`);
      } else {
        console.log(`[QE-Worker] vc-relax 2-phase produced no usable positions for ${formula} (Phase 1 exit=${phase1Result.exitCode}, Phase 2 exit=${phase2Result.exitCode}), proceeding with original geometry`);
        const vcDiagTail = phase2Result.stdout.slice(-1000);
        console.log(`[QE-Worker] vc-relax stdout tail for ${formula}:\n${vcDiagTail}`);

        // --- Multi-start retry for novel high-P hydrides ---
        // If vc-relax failed, try again from a different starting structure.
        // Different starting points can land in different energy basins.
        // Use the original funnel-selected structureCandidates (already in scope).
        const isNovelHighPH = elements.includes("H") && workerPressure >= 50 && !isKnownCompound;
        if (isNovelHighPH && structureCandidates.length > 1) {
          // Find a cage-seeded or parent-seeded candidate that's different from what we tried
          const retryCand = structureCandidates.find(c =>
            c.source !== result.stagedRelaxation?.stages[0]?.failReason &&
            (c.prototype.startsWith("cage-wyckoff-") || c.prototype.startsWith("parent-seed-") || c.prototype.startsWith("TemplateVCA"))
          ) ?? structureCandidates.find(c => c.latticeA !== latticeA);

          if (retryCand) {
            console.log(`[QE-Worker] vc-relax retry for ${formula}: trying ${retryCand.prototype} (a=${retryCand.latticeA.toFixed(3)} A, ${retryCand.positions.length} atoms)`);
            cleanQETmpDir(path.join(jobDir, "tmp"));
            try {
              const retryInput = generateVCRelaxInput(formula, elements, counts, retryCand.latticeA, retryCand.positions, workerPressure);
              const retryFile = path.join(jobDir, "vc_relax_retry.in");
              fs.writeFileSync(retryFile, retryInput);
              const retryResult = await runQECommand(
                path.posix.join(getQEBinDir(), "pw.x"), retryFile, jobDir,
                vcRelaxMaxSec * 1000 + 60_000,
              );
              fs.writeFileSync(path.join(jobDir, "vc_relax_retry.out"), retryResult.stdout);
              const retryParsed = parseVCRelaxOutput(retryResult.stdout);
              if (retryParsed.finalPositions && retryParsed.finalPositions.length > 0) {
                positions = retryParsed.finalPositions;
                result.vcRelaxed = true;
                if (retryParsed.finalLatticeAng && retryParsed.finalLatticeAng > 0.5) {
                  latticeA = retryParsed.finalLatticeAng;
                  result.relaxedLatticeA = latticeA;
                }
                console.log(`[QE-Worker] vc-relax RETRY succeeded for ${formula}: a=${latticeA.toFixed(3)} A, E=${retryParsed.totalEnergy.toFixed(4)} eV from ${retryCand.prototype}`);
              } else {
                console.log(`[QE-Worker] vc-relax RETRY also failed for ${formula} — using original geometry`);
              }
            } catch (retryErr: any) {
              console.log(`[QE-Worker] vc-relax RETRY error for ${formula}: ${retryErr.message?.slice(0, 100)}`);
            }
          }
        }
      }

      cleanQETmpDir(path.join(jobDir, "tmp"));
      } // close: if (result.vcRelaxed) {} else { ... vc-relax run ... }
    } catch (vcErr: any) {
      console.log(`[QE-Worker] vc-relax failed for ${formula}: ${(vcErr.message || "").slice(-200)}, proceeding with original geometry`);
      cleanQETmpDir(path.join(jobDir, "tmp"));
    }

    result.kPoints = autoKPoints(latticeA, cOverA, bOverAFull, undefined, DEFAULT_KSPACING, { stage: "scf", isMetallic: vegardResult?.isMetallic ?? undefined, totalAtoms: positions.length }).trim();
    if (Math.abs(latticeA - preVcLatticeA) > 0.01) {
      console.log(`[QE-Worker] K-points recomputed for ${formula} after vc-relax lattice change (${preVcLatticeA.toFixed(3)} -> ${latticeA.toFixed(3)} A): ${result.kPoints}`);
    }

    // --- DFT+U detection for strongly-correlated materials ---
    let dftPlusULines = "";
    let dftPlusUNspin2 = false;
    try {
      const corrEffects = await estimateCorrelationEffects(formula, {});
      const regime = corrEffects.regime.regime;
      if (regime === "strongly-correlated" || regime === "Mott-proximate") {
        const hubbardParts: string[] = ["  lda_plus_u = .true.,\n", "  lda_plus_u_kind = 0,\n"];
        for (let i = 0; i < elements.length; i++) {
          const u = getHubbardU(elements[i]);
          if (u != null && u > 0) {
            hubbardParts.push(`  Hubbard_U(${i + 1}) = ${u.toFixed(1)},\n`);
          }
        }
        dftPlusULines = hubbardParts.join("");
        const isMagCorrMat = corrEffects.materialPatterns.some(p =>
          p.includes("cuprate") || p.includes("Fe-pnictide"));
        if (isMagCorrMat) {
          dftPlusUNspin2 = true;
          for (let i = 0; i < elements.length; i++) {
            const u = getHubbardU(elements[i]);
            if (u != null && u > 0) {
              dftPlusULines += `  starting_magnetization(${i + 1}) = 0.5,\n`;
            }
          }
        }
        result.qeDFTPlusU = true;
        result.dftPlusUTcModifier = corrEffects.tcModifier;
        console.log(`[QE-Worker] DFT+U enabled for ${formula}: regime=${regime}, patterns=${corrEffects.materialPatterns.join(",")}, tcModifier=${corrEffects.tcModifier.toFixed(3)}`);
      }
    } catch (corrErr: any) {
      console.log(`[QE-Worker] Correlation detection skipped for ${formula}: ${(corrErr.message || "").slice(0, 100)}`);
    }

    // TSC jobs (submitted with jobType="scf_tsc" or opts.forceSpin) require
    // nspin=2 so that QE captures the spin-split bands relevant for Majorana
    // gap physics.  We reuse dftPlusUNspin2 which already gates the nspin=2
    // path in generateSCFInputWithParams (line ~2599).
    if (opts?.forceSpin && !dftPlusUNspin2) {
      dftPlusUNspin2 = true;
      // Must seed starting_magnetization for every species — QE aborts with
      // "some starting_magnetization MUST be set" when nspin=2 is requested
      // without any seed, and this path bypasses the magBlock generator in
      // generateSCFInputWithParams (it's gated off when dftPlusUNspin2=true).
      // Values follow aiida-qe defaults: larger seed on magnetic elements,
      // small symmetry-breaking seed elsewhere.
      for (let i = 0; i < elements.length; i++) {
        const el = elements[i];
        const seed = el in MAGNETIC_ELEMENTS
          ? 0.4
          : (TRANSITION_METALS.has(el) ? 0.2 : 0.1);
        dftPlusULines += `  starting_magnetization(${i + 1}) = ${seed.toFixed(1)},\n`;
      }
      console.log(`[QE-Worker] nspin=2 forced for TSC candidate ${formula} (spin-orbit gap physics) + per-species magnetization seeds`);
    }
    // ----------------------------------------------------------

    // Pre-classify system complexity so we start with appropriate SCF parameters
    // rather than wasting attempt 1 on settings that are known to fail.
    //
    // Root causes for attempt-1 failure on halogens / quaternary systems:
    //   1. Fluorine and other halogens create large electronegativity gradients →
    //      strong charge transfer → plain Pulay sloshing at beta=0.3.
    //   2. Quaternary systems (4+ elements) have heterogeneous charge regions that
    //      need local-TF (Thomas-Fermi) preconditioning to suppress oscillations.
    //      Plain Pulay only helps after several iterations; for complex systems it
    //      often never stabilises within 300 steps.
    //
    // Fix: detect these cases upfront and use local-TF + reduced beta from attempt 1.
    const hasHalogen = elements.some(el => ["F", "Cl", "Br", "I"].includes(el));
    const hasFluorine = elements.includes("F");
    const isQuaternaryPlus = elements.length >= 4;
    const isPentanaryPlus = elements.length >= 5;
    // Alkali + H mixtures (Li, Na, K, Rb, Cs with H) have huge electronegativity
    // spreads (Li 0.98, H 2.20, heavy metal ~1.1) plus dramatic orbital extent
    // mismatch between H 1s and alkali ns. At high pressure, charge transfer
    // becomes severe and plain Pulay mixing oscillates indefinitely.
    // Examples that fail: Li2LaH12, YH9Na2, LaH11Li2, CCaH4Li3, NaAsH4.
    const hasAlkali = elements.some(el => ["Li", "Na", "K", "Rb", "Cs"].includes(el));
    const isAlkaliHydride = hasAlkali && elements.includes("H");
    // High-pressure ternary+ hydrides (heavy metal + H + 3rd element) also need
    // local-TF — the charge sloshes between the H sublattice and the metal d/f bands.
    const hCount = (counts["H"] || 0);
    const totalAtomCount = positions.length;
    const isHighPHydride = hCount > 0 && (hCount / totalAtomCount) >= 0.5 && workerPressure > 50;

    // All-heavy quaternary+ systems (BaBiLaTe3 class) have pathological charge
    // sloshing — mixed-valence states (Bi³⁺/Bi⁵⁺) + heterogeneous ionic/covalent
    // layers create multiple local minima that plain or local-TF mixing can't resolve.
    // accuracy=2.49 Ry after 5 attempts on BaBiLaTe3 (Apr-18 run).
    const allHeavy = elements.length >= 3 && elements.every(el => HEAVY_ELEMENTS.has(el));
    const isExtremeSystem = allHeavy && isQuaternaryPlus;

    // "Very complex": F + quaternary, 5+ elements, DFT+U magnetic, alkali-hydride,
    // high-pressure (>50 GPa) hydride with ≥3 elements, or all-heavy quaternary+.
    const isVeryComplexSystem = (hasFluorine && isQuaternaryPlus) || isPentanaryPlus
      || dftPlusUNspin2 || isAlkaliHydride
      || (isHighPHydride && elements.length >= 3)
      || isExtremeSystem;
    const isComplexSystem = hasHalogen || isQuaternaryPlus || isHighPHydride || allHeavy;

    if (isExtremeSystem) {
      console.log(`[QE-Worker] ${formula}: extreme system [all-heavy quaternary+: ${elements.join(",")}] — using ultra-gentle SCF schedule (beta=0.07, atomic+random, degauss=0.03)`);
    } else if (isVeryComplexSystem) {
      const reasons = [
        hasFluorine && isQuaternaryPlus && "F+quaternary",
        isPentanaryPlus && `${elements.length} elements`,
        dftPlusUNspin2 && "DFT+U nspin2",
        isAlkaliHydride && "alkali-hydride (Li/Na/K + H)",
        isHighPHydride && elements.length >= 3 && `high-P ternary+ hydride (P=${workerPressure} GPa)`,
      ].filter(Boolean).join(", ");
      console.log(`[QE-Worker] ${formula}: very-complex system [${reasons}] — using hardened SCF schedule (local-TF from attempt 1)`);
    } else if (isComplexSystem) {
      console.log(`[QE-Worker] ${formula}: complex system (${elements.length} el, halogen=${hasHalogen}, highP-H=${isHighPHydride}) — using local-TF SCF schedule`);
    }

    type RetryConfig = { mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number; convThr?: string; forcConvThr?: string; etotConvThr?: string; mixingMode?: string; mixingNdim?: number; startingwfc?: string; startingpot?: string; diagoThrInit?: string; restartFromScratch?: boolean };

    // Extreme systems (all-heavy quaternary+: BaBiLaTe3 class).
    // Start with very gentle mixing (beta=0.07), wide smearing (degauss=0.03),
    // and startingwfc='atomic+random' to break mixed-valence symmetry traps.
    // BaBiLaTe3 scored accuracy=2.49 Ry (literal, not 2.49e-X) after 5 attempts
    // on the normal "very complex" ladder — it never even began to converge.
    const retryConfigsExtreme: RetryConfig[] = [
      { mixingBeta: 0.07, maxSteps: 500, diag: "david", degauss: 0.03,  convThr: "1.0d-6", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 16, startingwfc: "atomic+random" },
      { mixingBeta: 0.05, maxSteps: 600, diag: "david", degauss: 0.03,  convThr: "1.0d-5", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 20, ecutwfcBoost: 10 },
      { mixingBeta: 0.03, maxSteps: 800, diag: "cg",    degauss: 0.02,  convThr: "1.0d-5", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 24, ecutwfcBoost: 15, startingwfc: "random" },
      { mixingBeta: 0.02, maxSteps: 800, diag: "cg",    degauss: 0.01,  convThr: "1.0d-4", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 24, ecutwfcBoost: 20, smearing: "mp" },
      { mixingBeta: 0.01, maxSteps: 1000, diag: "cg",   degauss: 0.005, convThr: "1.0d-4", forcConvThr: "1.0d-2", etotConvThr: "1.0d-3", mixingMode: "local-TF", mixingNdim: 24, ecutwfcBoost: 25, smearing: "mp" },
    ];

    // Very complex systems (F + quaternary, 5+ elements, or DFT+U magnetic):
    // Start immediately with local-TF at beta=0.15. Each attempt also widens
    // degauss to help the Fermi surface converge before the charge density does.
    const retryConfigsVeryComplex: RetryConfig[] = [
      { mixingBeta: 0.15, maxSteps: 400, diag: "david", degauss: 0.02,  convThr: "1.0d-7", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 12 },
      { mixingBeta: 0.10, maxSteps: 500, diag: "cg",    degauss: 0.02,  convThr: "1.0d-7", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 16, ecutwfcBoost: 10 },
      { mixingBeta: 0.07, maxSteps: 600, diag: "cg",    degauss: 0.01,  convThr: "1.0d-8", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 16, ecutwfcBoost: 15 },
      { mixingBeta: 0.05, maxSteps: 600, diag: "cg",    degauss: 0.003, smearing: "mp",     convThr: "1.0d-6", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 20, ecutwfcBoost: 20 },
      { mixingBeta: 0.03, maxSteps: 800, diag: "cg",    degauss: 0.005, smearing: "mp",     convThr: "1.0d-5", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 24, ecutwfcBoost: 25 },
    ];

    // Complex systems (any halogen OR 4+ elements, but not "very complex"):
    // Start with local-TF at beta=0.2 to suppress sloshing from attempt 1;
    // conv_thr relaxed to 1d-7 (vs 1d-8) to give 10× more convergence budget
    // without sacrificing screening accuracy.
    const retryConfigsComplex: RetryConfig[] = [
      { mixingBeta: 0.20, maxSteps: 400, diag: "david", degauss: 0.02, convThr: "1.0d-7", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 12 },
      { mixingBeta: 0.15, maxSteps: 500, diag: "cg",    degauss: 0.02, convThr: "1.0d-7", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 16, ecutwfcBoost: 10 },
      { mixingBeta: 0.10, maxSteps: 500, diag: "cg",    degauss: 0.01, convThr: "1.0d-8", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 16, ecutwfcBoost: 15 },
      { mixingBeta: 0.07, maxSteps: 600, diag: "cg",    degauss: 0.003, smearing: "mp",    convThr: "1.0d-6", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 20, ecutwfcBoost: 20 },
      { mixingBeta: 0.05, maxSteps: 800, diag: "cg",    degauss: 0.005, smearing: "mp",    convThr: "1.0d-5", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 20, ecutwfcBoost: 25 },
    ];

    // Simple systems (binary/ternary, no halogens): original plain-Pulay schedule.
    // These converge well with beta=0.3; switching to local-TF adds overhead for no gain.
    const retryConfigsSimple: RetryConfig[] = [
      { mixingBeta: 0.3,  maxSteps: 300, diag: "david", degauss: 0.02, convThr: "1.0d-7", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5" },
      { mixingBeta: 0.2,  maxSteps: 400, diag: "david", degauss: 0.02, convThr: "1.0d-8", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", ecutwfcBoost: 10 },
      { mixingBeta: 0.15, maxSteps: 500, diag: "cg",    degauss: 0.01, convThr: "1.0d-8", forcConvThr: "1.0d-3", etotConvThr: "1.0d-5", mixingMode: "local-TF", mixingNdim: 16, ecutwfcBoost: 15 },
      { mixingBeta: 0.1,  maxSteps: 500, diag: "cg",    degauss: 0.003, smearing: "mp",   convThr: "1.0d-6", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 16, ecutwfcBoost: 20 },
      { mixingBeta: 0.05, maxSteps: 800, diag: "cg",    degauss: 0.005, smearing: "mp",   convThr: "1.0d-5", forcConvThr: "1.0d-2", etotConvThr: "1.0d-4", mixingMode: "local-TF", mixingNdim: 20, ecutwfcBoost: 25 },
    ];

    const retryConfigs = isExtremeSystem ? retryConfigsExtreme
      : isVeryComplexSystem ? retryConfigsVeryComplex
      : isComplexSystem ? retryConfigsComplex
      : retryConfigsSimple;

    const firstAttempt = opts?.startAttempt ?? 0;
    let scfConverged = false;
    let retryCount = 0;

    // Handler-driven retry override: mutated by the classifier at the end of
    // each failed attempt, applied on top of the static retryConfigs[] entry
    // for the NEXT attempt. Mirrors aiida-quantumespresso's PwBaseWorkChain
    // handlers (`workflows/pw/base.py::handle_*`) where the ladder progresses
    // based on what specifically went wrong, not a fixed sequence.
    let handlerOverride: Partial<RetryConfig> = {};

    // Tiered wall-time budget. Heavy-5d intermetallics (W/Re/Os/Ir/Pt) and
    // high-P hydrides need longer than the flat 88-min budget — empirically
    // they hit it every single attempt on worker2 (Apr 16 run: N4W3,
    // Re2Sn2W3, LaH12 all wall-timed ≥ 5 attempts each).
    const effectiveMaxSeconds = computeMaxSeconds(elements, workerPressure);
    const effectiveKillTimeoutMs = effectiveMaxSeconds * 1000 + 120_000;
    if (effectiveMaxSeconds !== QE_MAX_SECONDS) {
      console.log(`[QE-Worker] ${formula}: tier-adjusted max_seconds = ${effectiveMaxSeconds}s (${(effectiveMaxSeconds / 60).toFixed(0)} min) — heavy-TM or high-P hydride class`);
    }

    for (let attempt = firstAttempt; attempt < retryConfigs.length && !scfConverged; attempt++) {
      const params: RetryConfig = { ...retryConfigs[attempt], ...handlerOverride };
      // Recovery strategy on retry: disk_io='medium' writes charge-density
      // and wavefunctions, but wall-time-killed runs may not flush .wfc files.
      // restart_mode='restart' needs .wfc → davcio crash. Instead, use
      // from_scratch + startingpot='file' which reads the charge density but
      // generates fresh wavefunctions — recovers ~50-70% of SCF progress
      // without needing the wfc files that wall-time kills may not write.
      // Charge contamination control:
      //   Attempts 1-3: try to recover from previous charge density (fast)
      //   Attempt 4+: force clean restart — bad charge can make things worse
      //   After CHARGE_WRONG or wild Fermi artifact: always clean
      const forceClean = attempt >= 3 + firstAttempt || !!(handlerOverride as any)?._forceClean;
      const canRecover = attempt > firstAttempt && !(params as any).startingwfc && !forceClean;
      const recoveryParams = canRecover ? { startingpot: "file" as string } : {};
      if (forceClean && attempt > firstAttempt) {
        console.log(`[QE-Worker] SCF attempt ${attempt + 1}: forced clean restart (charge contamination control)`);
      }
      const scfInput = generateSCFInputWithParams(formula, elements, counts, latticeA, positions, {
        ...params,
        ...recoveryParams,
        restartFromScratch: true,
        maxSecondsOverride: effectiveMaxSeconds - 120,
        dftPlusULines: dftPlusULines || undefined,
        dftPlusUNspin2: dftPlusUNspin2 || undefined,
      });
      const scfInputFile = path.join(jobDir, `scf_attempt${attempt}.in`);
      fs.writeFileSync(scfInputFile, scfInput);

      // Clean tmp/ when not recovering or when forcing clean restart
      if (attempt > 0 && (!canRecover || forceClean)) {
        cleanQETmpDir(path.join(jobDir, "tmp"));
      }

      const smearInfo = params.smearing ? `, smearing=${params.smearing}` : "";
      const convInfo = params.convThr ? `, conv_thr=${params.convThr}` : "";
      const mixInfo = params.mixingMode ? `, mixing=${params.mixingMode}` : "";
      console.log(`[QE-Worker] SCF attempt ${attempt + 1}/${retryConfigs.length} for ${formula} (a=${latticeA.toFixed(2)} A, beta=${params.mixingBeta}, diag=${params.diag}, maxstep=${params.maxSteps}${convInfo}${smearInfo}${mixInfo})`);

      const scfResult = await runQECommand(
        path.posix.join(getQEBinDir(), "pw.x"),
        scfInputFile,
        jobDir,
        effectiveKillTimeoutMs,
      );

      fs.writeFileSync(path.join(jobDir, `scf_attempt${attempt}.out`), scfResult.stdout);
      const usedDegauss = params.degauss || 0.005;
      result.scf = parseSCFOutput(scfResult.stdout, usedDegauss);

      if (scfResult.exitCode !== 0 && !result.scf.converged) {
        // QE writes the actual error to stdout; stderr only has MPI_ABORT boilerplate.
        // Two different views of the output:
        //   - stdoutTail (600 chars): for the human-readable error summary.
        //   - combined (full stdout + stderr): for classifier matching. The
        //     diagnostic strings ("convergence NOT achieved", etc.) are
        //     written BEFORE the ~1.5 KB timing footer, so a last-600-char
        //     tail misses them on long runs (LaH12-class: 1h28m, huge
        //     footer). Scan the whole output for the classifier keywords.
        const stdoutTail = scfResult.stdout.slice(-600);
        const combined = scfResult.stderr + "\n" + scfResult.stdout;

        // XC functional conflicts (igcx/igcc) arise when a PP encodes a different
        // functional than input_dft — treating as a PP error skips wasteful retries.
        //
        // IMPORTANT: do NOT match bare "pseudopotential". QE prints a harmless
        // startup banner "momentum in pseudopotentials (lmaxx) = 3" on every
        // run — matching that substring previously misclassified >75 jobs
        // whose real error was elsewhere (most often missing starting_magnetization).
        // Match only on specific fatal PP strings.
        const isPPError = combined.includes("from read_upf") || combined.includes("from readpp") ||
          combined.includes("read_ps ") || combined.includes("Error reading pseudo") ||
          combined.includes("EOF marker") ||
          combined.includes("conflicting values for igcx") || combined.includes("conflicting values for igcc") ||
          combined.includes("set_dft_from_name") ||
          // lmaxx overflow only when paired with iosys/init_us error, not the
          // standalone diagnostic banner.
          (combined.includes("lmaxx") && (combined.includes("init_us_1") || combined.includes("too small")));
        if (isPPError) {
          result.scf.error = `Pseudopotential read failure: ${stdoutTail.slice(-300)}`;
          console.log(`[QE-Worker] PP error for ${formula}, no retry will help — skipping`);
          recordFormulaFailure(formula);
          break;
        }
        // Input-parse (iosys) errors abort instantly — retries won't help since
        // the input template is the problem. Label distinctly so ops can tell
        // config bugs from SCF non-convergence. Common: missing/ill-formed
        // starting_magnetization, nbnd too small, bad CELL_PARAMETERS.
        const iosysMatch = combined.match(/Error in routine\s+iosys[^\n]*\n\s*([^\n]{0,200})/);
        if (iosysMatch) {
          const detail = iosysMatch[1].trim();
          result.scf.error = `Input error (iosys): ${detail}`;
          console.log(`[QE-Worker] iosys input error for ${formula} — no retry will help: ${detail}`);
          recordFormulaFailure(formula);
          break;
        }
        // Geometry failures won't improve with SCF parameter tweaks — skip all retries.
        // NB: do NOT match "overlap" alone — it false-matches QE's normal
        // "Davidson diagonalization with overlap" banner and mislabels every
        // non-convergence as a geometry failure. Match only on specific
        // geometry-error phrases.
        const isGeomError = combined.includes("atom too close") || combined.includes("negative Jacobian") ||
          combined.includes("atoms are too close") || combined.includes("overlapping atoms") ||
          combined.includes("Wrong atomic coordinates") || combined.includes("too many atoms in the unit cell");
        if (isGeomError) {
          result.scf.error = `Geometry failure (atoms too close or bad cell): ${stdoutTail.slice(-200)}`;
          console.log(`[QE-Worker] Geometry error for ${formula}, no retry will help — ${stdoutTail.slice(-120)}`);
          recordFormulaFailure(formula);
          break;
        }
        // Capture the real error: prefer stdout tail over MPI_ABORT boilerplate in stderr.
        const errSummary = stdoutTail || scfResult.stderr.slice(-300);
        // Classify exit=-1 cases (generic process death) so ops can tell
        // timeout from OOM from segfault without spelunking raw stderr.
        // Also classify exit=2 (QE's generic "check stdout" code) so the
        // log line is actionable instead of "exited with code 2: 23 calls)"
        // which leaks the timing-block tail with no diagnostic value.
        // Error strings sourced from aiida-quantumespresso parse_raw/pw.py.
        let classifier = "";
        if (scfResult.exitCode === -1) {
          if (combined.includes("TIMEOUT")) classifier = " [TIMEOUT]";
          else if (combined.includes("Killed") || combined.includes("SIGKILL") || combined.includes("out of memory") || combined.includes("Cannot allocate")) classifier = " [OOM/KILLED]";
          else if (combined.includes("SIGSEGV") || combined.includes("Segmentation fault")) classifier = " [SEGFAULT]";
          else if (combined.includes("ENOENT") || combined.includes("command not found") || combined.includes("No such file")) classifier = " [BINARY_MISSING]";
          else classifier = " [PROCESS_DIED]";
        } else if (scfResult.exitCode === 2) {
          // WALL_TIME_EXHAUSTED must be checked FIRST. QE exits cleanly
          // (JOB DONE, exit=2) when max_seconds is hit, and the stdout
          // still contains normal per-iteration diagnostics like "too many
          // bands are not converged" from earlier iterations. Matching
          // those first misclassifies as DIAG_NOT_CONVERGED and triggers
          // a handler that makes things WORSE (david→cg is slower per
          // iteration). Detect wall-time by checking if the parsed SCF
          // wall time is within 15% of QE_MAX_SECONDS, or QE's explicit
          // "Maximum CPU time exceeded" banner.
          const scfWall = result.scf?.wallTimeSeconds ?? 0;
          const isWallTimeKill = combined.includes("Maximum CPU time exceeded") ||
            combined.includes("max_seconds") ||
            (scfWall > 0 && scfWall >= QE_MAX_SECONDS * 0.85);
          if (isWallTimeKill) classifier = " [WALL_TIME_EXHAUSTED]";
          else if (combined.includes("convergence NOT achieved")) classifier = " [SCF_NOT_CONVERGED]";
          else if (combined.includes("charge is wrong")) classifier = " [CHARGE_WRONG]";
          else if (combined.includes("S matrix not positive definite")) classifier = " [S_NOT_POSITIVE]";
          else if (combined.includes("too many bands are not converged")) classifier = " [DIAG_NOT_CONVERGED]";
          else if (combined.includes("eigenvalues not converged")) classifier = " [DIAG_NOT_CONVERGED]";
          else if (combined.includes("wrong number of electrons")) classifier = " [WRONG_NELEC]";
          else if (combined.includes("dE0s is positive")) classifier = " [BFGS_UPHILL]";
          else if (combined.includes("smearing is needed")) classifier = " [SMEARING_NEEDED]";
        }
        result.scf.error = `pw.x exited with code ${scfResult.exitCode}${classifier}: ${errSummary}`;
        console.log(`[QE-Worker] SCF attempt ${attempt + 1} failed for ${formula}${classifier}: ${errSummary.slice(-200)}`);

        // Handler-driven override for the NEXT attempt. Each branch targets
        // the specific failure mode — mirrors aiida's PwBaseWorkChain
        // handle_electronic_convergence_not_reached / handle_diagonalization_errors
        // / handle_unconverged_cholesky etc. Overrides accumulate; the static
        // retryConfigs[attempt+1] provides the base, we patch on top.
        const nextOverride: Partial<RetryConfig> = { ...handlerOverride };
        if (classifier === " [WALL_TIME_EXHAUSTED]") {
          // Wall-time kill. The system is just big/stiff and needs more
          // iterations than fit in QE_MAX_SECONDS. Switching to cg/ppcg
          // (the old DIAG handler) would be COUNTERPRODUCTIVE because
          // those are 2-3× slower per iteration than david.
          //
          // Strategy: KEEP david (fastest iterations), loosen conv_thr so
          // we converge within the time budget, keep mixing tight. If
          // we've already wall-time-killed twice, accept partial results.
          nextOverride.diag = "david";
          delete nextOverride.diagoThrInit;
          const currentConvThr = params.convThr ?? "1.0d-7";
          const looseConvThr = currentConvThr === "1.0d-8" ? "1.0d-6"
            : currentConvThr === "1.0d-7" ? "1.0d-5"
            : "1.0d-4";
          nextOverride.convThr = looseConvThr;
          // If this is the 2nd+ wall-time kill, accept partial results if
          // accuracy is close enough for screening. No point burning 5×90 min.
          const accuracy = result.scf?.lastScfAccuracyRy;
          if (attempt >= 1 && accuracy !== null && accuracy !== undefined && accuracy < 1.0e-4) {
            console.log(`[QE-Worker] Wall-time exhausted ${attempt + 1}× for ${formula}, but last accuracy ${accuracy.toExponential(1)} Ry is usable for screening — accepting partial SCF`);
            result.scf!.converged = false;
            result.scf!.convergenceQuality = "partial-walltime";
            scfConverged = false;
            // Break the retry loop — we'll use partial results below via scfUsable
            handlerOverride = nextOverride;
            retryCount = attempt + 1;
            break;
          }
        } else if (classifier === " [SCF_NOT_CONVERGED]") {
          // Halve mixing_beta (floor 0.03), bump maxSteps, force local-TF if still plain.
          nextOverride.mixingBeta = Math.max(0.03, (params.mixingBeta ?? 0.3) * 0.5);
          nextOverride.maxSteps = Math.max(params.maxSteps, (params.maxSteps ?? 300) + 200);
          if ((params.mixingMode ?? "plain") === "plain") nextOverride.mixingMode = "local-TF";
          nextOverride.mixingNdim = Math.max(params.mixingNdim ?? 8, 16);
        } else if (classifier === " [DIAG_NOT_CONVERGED]" || classifier === " [S_NOT_POSITIVE]") {
          // Escalate diagonalization: david → cg → ppcg. Raise diago_thr_init
          // so the initial diagonalization doesn't over-tighten before SCF.
          const currentDiag = params.diag ?? "david";
          nextOverride.diag = currentDiag === "david" ? "cg" : "ppcg";
          nextOverride.diagoThrInit = "1.0d-4";
        } else if (classifier === " [CHARGE_WRONG]") {
          // Restart wavefunctions/potential from atomic superposition.
          // Force clean on next attempt — bad charge density is poison.
          nextOverride.startingwfc = "random";
          nextOverride.startingpot = "atomic";
          nextOverride.mixingBeta = Math.max(0.05, (params.mixingBeta ?? 0.3) * 0.5);
          (nextOverride as any)._forceClean = true;
        } else if (classifier === " [SMEARING_NEEDED]") {
          // Metal mis-detected as insulator: widen smearing + force MV.
          nextOverride.smearing = "mv";
          nextOverride.degauss = Math.max(params.degauss ?? 0.005, 0.03);
        }
        if (Object.keys(nextOverride).length > Object.keys(handlerOverride).length) {
          const changed = Object.entries(nextOverride)
            .filter(([k, v]) => (handlerOverride as any)[k] !== v)
            .map(([k, v]) => `${k}=${v}`)
            .join(", ");
          console.log(`[QE-Worker] Handler override for ${formula} attempt ${attempt + 2}: ${changed}`);
        }
        handlerOverride = nextOverride;
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

    // Partial-convergence acceptance for screening. aiida uses ~1e-4 Ry as
    // the threshold below which electronic structure is "good enough" for
    // screening-level Tc estimates. Prior threshold of 1e-6 was too strict
    // and caused wall-time-exhausted runs to be discarded after 5×90 min
    // of compute. 1e-4 Ry ≈ 1.4 meV/atom — perfectly acceptable for
    // ranking candidates; final Tc is re-evaluated at publication quality.
    // Two acceptance paths:
    // 1. Normal: energy + Fermi level + accuracy < 1e-4 (full electronic structure)
    // 2. Partial-walltime: energy + accuracy < 1e-4 but Fermi level may be missing
    //    (QE only prints Ef after full convergence; wall-time-killed runs don't reach it).
    //    Still usable for screening — the total energy and band structure are valid,
    //    and Ef can be estimated from the last-iteration DOS if needed.
    const isPartialWalltime = result.scf?.convergenceQuality === "partial-walltime";
    const scfUsable = scfConverged ||
      (result.scf && result.scf.totalEnergy !== 0 &&
       (result.scf.fermiEnergy !== null || isPartialWalltime) &&
       result.scf.lastScfAccuracyRy !== null && result.scf.lastScfAccuracyRy < 1.0e-4);

    // --- Stage 3 post-SCF diagnostics ---
    // After SCF converges, check residual forces and stress from tprnfor/tstress
    // output to assess relaxation quality. Warnings only — don't gate on these.
    if (scfUsable && result.scf) {
      const scfForce = result.scf.totalForce;
      const scfPressure = result.scf.pressure;
      if (scfForce != null && scfForce > 0.05) {
        console.log(`[QE-Worker] Stage 3 diagnostic: ${formula} residual force ${scfForce.toFixed(4)} Ry/bohr > 0.05 — structure may benefit from better relaxation`);
      }
      if (scfPressure != null && Math.abs(scfPressure) > 5) {
        console.log(`[QE-Worker] Stage 3 diagnostic: ${formula} residual pressure ${scfPressure.toFixed(1)} kbar > 5 — cell may not be fully relaxed`);
      }
      if (result.scf.isMetallic === false && result.scf.bandGap != null && result.scf.bandGap > 0.5) {
        const metalExpected = vegardResult?.isMetallic;
        if (metalExpected === true) {
          console.log(`[QE-Worker] Stage 3 diagnostic: ${formula} bandgap=${result.scf.bandGap.toFixed(3)} eV but Vegard/MP data suggests metallic — smearing or structure issue?`);
        }
      }
      if (result.scf.fermiEnergy != null) {
        // Sanity check: Fermi energy should be within a reasonable range
        if (Math.abs(result.scf.fermiEnergy) > 50) {
          console.log(`[QE-Worker] Stage 3 diagnostic: ${formula} Ef=${result.scf.fermiEnergy.toFixed(2)} eV — unusually large, check for smearing artifacts`);
        }
      }
    }

    // Assign quality tier based on convergence state
    if (scfConverged && result.vcRelaxed) {
      result.qualityTier = "relaxed";
    } else if (scfConverged) {
      result.qualityTier = "screening_converged";
    } else if (scfUsable) {
      result.qualityTier = "partial_screening";
    } else {
      result.qualityTier = "failed";
    }
    // Upgraded later: final_converged after phonon, publication_ready after e-ph

    // --- Stage 5.5: Convex hull stability assessment ---
    // After SCF converges, estimate how far this structure is from the
    // convex hull of competing phases. This answers "is this formula
    // stable against decomposition at this pressure?"
    if (scfConverged && result.scf && result.scf.totalEnergy !== 0) {
      try {
        const ePerAtom = result.scf.totalEnergy / positions.length;
        // Use Miedema model for a quick decomposition enthalpy estimate.
        // Full DFT convex hull requires computing all competing phases
        // (future work), but Miedema gives a rough formation energy.
        const { computeMiedemaFormationEnergy } = await import("../learning/phase-diagram-engine");
        const miedemaH = computeMiedemaFormationEnergy(formula);
        if (miedemaH != null && isFinite(miedemaH)) {
          // Miedema gives formation enthalpy relative to ELEMENTS (not the hull).
          // Negative = exothermic vs elements (likely stable).
          // Positive = endothermic vs elements (likely unstable).
          //
          // This is NOT a true hull distance (which requires all competing phases).
          // It's a stability indicator: strongly negative means the compound wants
          // to form, strongly positive means it wants to decompose.
          //
          // For hydrides, Miedema hits the -8 eV/atom floor (model not calibrated
          // for metal-H bonding at extreme pressure). Treat these as "unknown_hull"
          // rather than reporting bogus numbers.
          const miedemaReliable = Math.abs(miedemaH) < 5.0; // Miedema floor/ceiling = unreliable
          const formationMeV = miedemaH * 1000;

          if (miedemaReliable) {
            // Use formation energy as a proxy for stability tendency:
            // Very negative (<-500 meV) = strongly wants to form → likely on/near hull
            // Mildly negative (-500 to 0) = moderately stable
            // Positive (0 to +200) = mildly unstable → near_hull or metastable
            // Very positive (>+200) = highly unstable
            const label: "on_hull" | "near_hull" | "metastable" | "highly_metastable" | "unknown_hull" =
              formationMeV < -200 ? "on_hull" :
              formationMeV < 0 ? "near_hull" :
              formationMeV < 100 ? "metastable" : "highly_metastable";

            result.hullStability = {
              hullDistanceMeVAtom: Math.round(Math.max(0, formationMeV) * 10) / 10, // only positive values are "above hull"
              label,
              computedFromDFT: false,
            };
            console.log(`[QE-Worker] Hull stability for ${formula}: ${label} (Miedema ΔHf=${miedemaH.toFixed(3)} eV/atom = ${formationMeV.toFixed(0)} meV/atom, E_DFT/atom=${ePerAtom.toFixed(4)} eV)`);
          } else {
            // Miedema hit floor/ceiling — unreliable for this composition (e.g. hydrides)
            result.hullStability = { hullDistanceMeVAtom: 0, label: "unknown_hull", computedFromDFT: false };
            console.log(`[QE-Worker] Hull stability for ${formula}: unknown_hull (Miedema=${miedemaH.toFixed(2)} eV/atom — hit model limits, unreliable for this composition)`);
          }
        } else {
          result.hullStability = { hullDistanceMeVAtom: 0, label: "unknown_hull", computedFromDFT: false };
          console.log(`[QE-Worker] Hull stability for ${formula}: unknown (Miedema not available for this composition)`);
        }
      } catch (hullErr: any) {
        result.hullStability = { hullDistanceMeVAtom: 0, label: "unknown_hull", computedFromDFT: false };
        console.log(`[QE-Worker] Hull stability computation failed for ${formula}: ${hullErr.message?.slice(0, 80)}`);
      }
    }

    // --- Quality-weighted adaptive learning signal ---
    // Record DFT convergence as a stronger signal than funnel survival.
    // The funnel already recorded funnel_survival (weight 0.1) for all
    // candidates. Now record dft0_converged (0.5) or dft1_low_enthalpy (1.0)
    // for the winning prototype, giving the learning system a much stronger
    // signal about which generators and volumes actually produce DFT-viable structures.
    if (scfConverged || scfUsable) {
      const winnerProto = result.prototypeUsed ?? "unknown";
      const dftWeight = scfConverged
        ? getSignalWeight("dft0_converged")  // 0.5
        : getSignalWeight("dft_selected");    // 0.3
      recordGeneratorOutcome(elements, workerPressure, winnerProto, true, dftWeight);
    }

    if (!scfConverged && !scfUsable) {
      result.failureStage = "scf";
      stageFailureCounts.scf++;
      recordFormulaFailure(formula);
      if (result.scf && result.scf.totalEnergy !== 0 && result.scf.lastScfAccuracyRy !== null) {
        console.log(`[QE-Worker] SCF not converged for ${formula}: accuracy=${result.scf.lastScfAccuracyRy.toExponential(2)} Ry (threshold 1e-4 Ry for screening) — discarding non-physical energy`);
      }
    } else if (!scfConverged && scfUsable) {
      console.log(`[QE-Worker] SCF near-converged for ${formula} (accuracy=${result.scf!.lastScfAccuracyRy?.toExponential(2)} Ry < 1e-4 Ry, quality=${result.scf!.convergenceQuality}): E=${result.scf!.totalEnergy.toFixed(4)} eV, Ef=${result.scf!.fermiEnergy} — proceeding with caution`);
    }

    // ── Round 2: Iterative search around DFT winner ──────────────────────────
    // After SCF converges, generate focused variants around the DFT-optimized
    // structure and check if any have lower energy. This catches nearby basins
    // that the initial broad search missed.
    if (scfConverged && result.scf && !isPartialWalltime && tierDecision.tier !== "preview") {
      const r2Decision = shouldDoRound2(
        scfConverged,
        result.scf.totalForce ?? null,
        result.scf.isMetallic ?? null,
        result.qualityTier,
      );

      if (r2Decision.doRound2) {
        console.log(`[QE-Worker] Round 2 search for ${formula}: ${r2Decision.reason}`);
        try {
          // Build a CSPCandidate from the DFT-optimized structure
          const dftWinner: CSPCandidate = {
            latticeA,
            positions: positions.map(p => ({ ...p })),
            prototype: result.prototypeUsed ?? "dft-optimized",
            crystalSystem: "unknown",
            spaceGroup: "",
            source: `DFT-optimized (E=${result.scf.totalEnergy.toFixed(2)} eV)`,
            confidence: 0.95,
            isMetallic: result.scf.isMetallic ?? null,
            sourceEngine: "known-structure",
            generationStage: 1,
            seed: Date.now() % 1e8,
            pressureGPa: workerPressure,
            relaxationLevel: "relax-qe",
            enthalpyPerAtom: result.scf.totalEnergy / positions.length,
          };

          const r2Candidates = generateRound2Candidates(
            dftWinner, result.scf.totalEnergy, latticeA,
            elements, counts, workerPressure,
          );

          if (r2Candidates.length > 0) {
            const r2WorkDir = path.join(jobDir, "round2_chgnet");
            const r2Selected = await screenRound2(
              r2Candidates, formula,
              result.scf.totalEnergy / positions.length,
              r2WorkDir,
              workerPressure,
            );

            if (r2Selected.length > 0) {
              // Run Stage 1 atomic relax on the best Round 2 candidate
              const r2Best = r2Selected[0];
              console.log(`[QE-Worker] Round 2 best candidate: ${r2Best.source} (CHGNet E=${r2Best.enthalpyPerAtom?.toFixed(4) ?? "?"} eV/atom)`);

              // Quick fixed-cell relax of the Round 2 candidate
              try {
                const r2Prefix = formula.replace(/[^a-zA-Z0-9]/g, "") + "_r2";
                const r2EcutwfcRelax = Math.max(computeEcutwfc(elements, 0, 80, 45), elements.includes("H") ? 80 : 50);
                const r2EcutrhoRelax = r2EcutwfcRelax * ecutrhoMultiplier(elements);
                const r2COverA = estimateCOverA(elements, counts);
                const r2BOverA = estimateBOverA(elements, counts);
                const r2Kpts = autoKPoints(r2Best.latticeA, r2COverA, r2BOverA, undefined, 0.5, { stage: "scf", totalAtoms: positions.length }).trim();
                const r2Nspin = mayHaveMagneticMoment(elements) ? 2 : 1;
                const r2MagLines = r2Nspin === 2 ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !elements.some(el => el in MAGNETIC_ELEMENTS)) : "";

                let r2Species = "";
                for (const el of elements) {
                  r2Species += `  ${el}  ${getAtomicMass(el).toFixed(3)}  ${resolvePPFilename(el)}\n`;
                }
                let r2Pos = "";
                for (const pos of r2Best.positions) {
                  r2Pos += `  ${pos.element}  ${pos.x.toFixed(6)}  ${pos.y.toFixed(6)}  ${pos.z.toFixed(6)}\n`;
                }
                const r2Cell = generateCellParameters(r2Best.latticeA, r2COverA, 0, r2BOverA, elements, counts);

                const r2Input = `&CONTROL
  calculation = 'relax',
  restart_mode = 'from_scratch',
  prefix = '${r2Prefix}',
  outdir = './tmp',
  disk_io = 'low',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = 1.0d-3,
  etot_conv_thr = 1.0d-4,
  nstep = 100,
  max_seconds = 1800,
/
&SYSTEM
  ibrav = 0,
  nat = ${r2Best.positions.length},
  ntyp = ${elements.length},
  ecutwfc = ${r2EcutwfcRelax},
  ecutrho = ${r2EcutrhoRelax},
  input_dft = 'PBE',
  occupations = 'smearing',
  smearing = 'mv',
  degauss = 0.015,
  nspin = ${r2Nspin},
${r2MagLines}/
&ELECTRONS
  electron_maxstep = 200,
  conv_thr = 1.0d-6,
  mixing_beta = 0.3,
  mixing_mode = 'local-TF',
  diagonalization = 'david',
/
&IONS
  ion_dynamics = 'bfgs',
/
ATOMIC_SPECIES
${r2Species}
ATOMIC_POSITIONS {crystal}
${r2Pos}
K_POINTS {automatic}
${r2Kpts}

${r2Cell}
`;
                const r2File = path.join(jobDir, "round2_relax.in");
                fs.writeFileSync(r2File, r2Input);
                const r2Result = await runQECommand(
                  path.posix.join(getQEBinDir(), "pw.x"), r2File, jobDir, 1860000,
                );
                fs.writeFileSync(path.join(jobDir, "round2_relax.out"), r2Result.stdout);

                // Parse Round 2 energy
                const r2EnergyMatch = r2Result.stdout.match(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/g);
                if (r2EnergyMatch) {
                  const r2LastE = r2EnergyMatch[r2EnergyMatch.length - 1].match(/([-\d.]+)\s+Ry/);
                  if (r2LastE) {
                    const r2Energy = parseFloat(r2LastE[1]) * 13.6057; // Ry → eV
                    console.log(`[QE-Worker] Round 2 DFT result: E=${r2Energy.toFixed(4)} eV (Round 1: ${result.scf!.totalEnergy.toFixed(4)} eV, diff=${(r2Energy - result.scf!.totalEnergy).toFixed(4)} eV)`);

                    if (r2Energy < result.scf!.totalEnergy - 0.01) {
                      // Round 2 found a deeper basin!
                      console.log(`[QE-Worker] Round 2 found lower energy for ${formula}! Using Round 2 structure (${(result.scf!.totalEnergy - r2Energy).toFixed(4)} eV improvement)`);

                      // Parse Round 2 positions
                      const r2PosBlocks = r2Result.stdout.match(/ATOMIC_POSITIONS\s*[{(]?\s*(?:crystal|angstrom|bohr|alat)?\s*[})]?\s*\n([\s\S]*?)(?=\n\s*(?:CELL_PARAMETERS|K_POINTS|End final|End of|ATOMIC_SPECIES|\n\s*\n)|$)/gi);
                      if (r2PosBlocks && r2PosBlocks.length > 0) {
                        const r2LastBlock = r2PosBlocks[r2PosBlocks.length - 1];
                        const r2ParsedPos: Array<{ element: string; x: number; y: number; z: number }> = [];
                        for (const line of r2LastBlock.split("\n").slice(1)) {
                          const m = line.trim().match(/^([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
                          if (m) r2ParsedPos.push({ element: m[1], x: parseFloat(m[2]), y: parseFloat(m[3]), z: parseFloat(m[4]) });
                        }
                        if (r2ParsedPos.length === positions.length) {
                          positions = r2ParsedPos;
                          latticeA = r2Best.latticeA;
                          // Re-run SCF with Round 2 positions for clean phonon input
                          console.log(`[QE-Worker] Re-running SCF with Round 2 structure for ${formula}`);
                        }
                      }
                    } else {
                      console.log(`[QE-Worker] Round 2 did not improve energy for ${formula} — keeping Round 1 structure`);
                    }
                  }
                }
                // Clean only Round 2 .save — do NOT clean Round 1 .save
                // (phonon and bands need it). cleanQETmpDir deletes ALL .save dirs.
                const r2SaveDir = path.join(jobDir, "tmp", `${r2Prefix}.save`);
                try { if (fs.existsSync(r2SaveDir)) fs.rmSync(r2SaveDir, { recursive: true, force: true }); } catch {}
                console.log(`[QE-Worker] Round 2 cleanup: removed ${r2Prefix}.save, preserved Round 1 .save for phonon/bands`);
              } catch (r2DftErr: any) {
                console.log(`[QE-Worker] Round 2 DFT failed: ${r2DftErr.message?.slice(0, 100)} — keeping Round 1`);
              }
            }
          }
        } catch (r2Err: any) {
          console.log(`[QE-Worker] Round 2 search failed: ${r2Err.message?.slice(0, 100)} — keeping Round 1`);
        }
      }
    }

    // ── Phonon BEFORE bands (cycle 1374 fix for "0 modes" bug) ───────────────
    // Order matters: bands and ph.x both target ./tmp/${prefix}.save/, and bands
    // OVERWRITES the SCF wavefunctions there with bands-path wavefunctions.
    // ph.x then sees the wrong k-point grid and exits in <1s with no frequencies
    // — exactly the LaH10 "21000s SCF, 1s phonon, 0 modes" symptom. Running
    // phonon first preserves the SCF .save/ contents for ph.x; bands afterwards
    // is free to clobber them since nothing reads .save/ after that point.
    // The earlier "Do NOT clean ./tmp" comment in the bands block was a partial
    // diagnosis — not cleaning the dir wasn't enough because pw.x rewrites the
    // wfc files inside it.
    // Skip phonon for partial-walltime SCFs: wall-time-killed runs may not
    // flush wavefunctions (.wfc files) to disk, so ph.x would fail to read
    // them and either crash or produce 0 modes.
    // --- Stage 4: Gamma-point phonon stability check ---
    // Fast dynamical stability screen (5-30 min) before committing to the
    // full phonon grid (2-24h). Catches structures with large imaginary modes
    // early — e.g., BiGeSb had 4/9 imaginary modes that would have been caught
    // here in ~10 min instead of wasting 14.5h on the full pipeline.
    let gammaPhononPassed = true;
    if (scfUsable && !isPartialWalltime && result.scf) {
      // --- Pre-phonon diagnostics ---
      // Log the structure going into phonon so we can diagnose crashes.
      // ph.x reads from .save/ but these are the positions that generated it.
      const scfForce = result.scf.totalForce ?? null;
      const scfPressure = result.scf.pressure ?? null;
      const scfAccuracy = result.scf.lastScfAccuracyRy ?? null;
      console.log(`[QE-Worker] Pre-phonon state for ${formula}:`);
      console.log(`[QE-Worker]   SCF: converged=${scfConverged}, accuracy=${scfAccuracy?.toExponential(2) ?? "N/A"} Ry, E=${result.scf.totalEnergy.toFixed(4)} eV, Ef=${result.scf.fermiEnergy ?? "N/A"}`);
      console.log(`[QE-Worker]   Structure: ${positions.length} atoms, a=${latticeA.toFixed(3)} Å, force=${scfForce?.toFixed(4) ?? "N/A"} Ry/bohr, pressure=${scfPressure?.toFixed(1) ?? "N/A"} kbar`);
      // Log first few atomic positions for diagnosis
      const posToLog = Math.min(positions.length, 8);
      for (let pi = 0; pi < posToLog; pi++) {
        const p = positions[pi];
        console.log(`[QE-Worker]   atom[${pi}] ${p.element.padEnd(2)} (${p.x.toFixed(5)}, ${p.y.toFixed(5)}, ${p.z.toFixed(5)})`);
      }
      if (positions.length > posToLog) {
        console.log(`[QE-Worker]   ... and ${positions.length - posToLog} more atoms`);
      }

      // --- Pre-phonon force gate ---
      // DFPT requires well-relaxed structures. If residual forces are too high,
      // ph.x will crash with IEEE_UNDERFLOW_FLAG. Skip gamma phonon and go
      // straight to full phonon which has its own force tolerance.
      const FORCE_GATE_THRESHOLD = 0.10; // Ry/bohr — above this, DFPT is likely to crash
      if (scfForce != null && scfForce > FORCE_GATE_THRESHOLD) {
        console.log(`[QE-Worker] Pre-phonon force gate: ${formula} residual force ${scfForce.toFixed(4)} Ry/bohr > ${FORCE_GATE_THRESHOLD} — skipping Stage 4 gamma phonon (DFPT crash likely)`);
        console.log(`[QE-Worker] Gamma phonon SKIPPED for ${formula} due to high forces — proceeding to full phonon (it may also fail, but won't waste Stage 4 budget)`);
        result.gammaPhononPassed = true; // Let full phonon try anyway
        // Don't set gammaPhononPassed = false — we want to attempt full phonon
      } else {

      try {
        const qeCallbacks = buildQERunnerCallbacks();
        const hasH = elements.includes("H");
        const ecutwfc = Math.max(computeEcutwfc(elements, 0, 80, 45), hasH ? 100 : 60);
        const gammaResult = await runStage4GammaPhonon({
          formula,
          elements,
          counts,
          positions,
          latticeA,
          cellVectors: undefined,
          jobDir,
          callbacks: qeCallbacks,
          ecutwfc,
        });

        if (!gammaResult.passed) {
          gammaPhononPassed = false;
          result.gammaPhononPassed = false;
          console.log(`[QE-Worker] Gamma phonon check FAILED for ${formula}: ${gammaResult.failReason} — skipping full phonon grid`);
          // Store gamma phonon results as a diagnostic
          if (gammaResult.frequencies && gammaResult.frequencies.length > 0) {
            result.phonon = {
              frequencies: gammaResult.frequencies,
              hasImaginary: gammaResult.frequencies.some(f => f < -10),
              imaginaryCount: gammaResult.frequencies.filter(f => f < -10).length,
              lowestFrequency: Math.min(...gammaResult.frequencies),
              highestFrequency: Math.max(...gammaResult.frequencies),
              converged: false,
              wallTimeSeconds: gammaResult.wallTimeSeconds,
              error: "Gamma phonon check failed: " + (gammaResult.failReason ?? "unknown"),
            };
          }
        } else {
          result.gammaPhononPassed = true;
          const wasSkipped = gammaResult.frequencies?.length === 0 && gammaResult.wallTimeSeconds === 0;
          if (wasSkipped) {
            console.log(`[QE-Worker] Gamma phonon check SKIPPED for ${formula} (cost too high) — proceeding to full phonon grid without gamma screening`);
          } else {
            console.log(`[QE-Worker] Gamma phonon check PASSED for ${formula} (${gammaResult.frequencies?.length ?? 0} modes, lowest=${gammaResult.frequencies && gammaResult.frequencies.length > 0 ? Math.min(...gammaResult.frequencies).toFixed(1) : "N/A"} cm⁻¹) — proceeding to full phonon grid`);
          }
        }
      } catch (gammaErr: any) {
        console.log(`[QE-Worker] Gamma phonon check error for ${formula}: ${gammaErr.message?.slice(0, 150)} — proceeding to full phonon anyway`);
        // Don't block on Gamma check errors — fall through to full phonon
      }
      } // close force-gate else
    }

    if (scfUsable && !isPartialWalltime && gammaPhononPassed) {
      // Log pre-full-phonon state for diagnosis
      console.log(`[QE-Worker] Pre-full-phonon for ${formula}: ${positions.length} atoms, a=${latticeA.toFixed(3)} Å, force=${result.scf?.totalForce?.toFixed(4) ?? "N/A"} Ry/bohr, pressure=${result.scf?.pressure?.toFixed(1) ?? "N/A"} kbar, metallic=${result.scf?.isMetallic ?? "unknown"}`);

      // Phonon budget: scale with atom count × electron weight. Heavy hydrides
      // (K2LaH8, LaH12) have 33+ perturbations × expensive mini-SCFs. The old
      // 4× SCF-time multiplier gave ~9h for K2LaH8 which was borderline —
      // ph.x was killed 1 perturbation short → 0 modes.
      //
      // New approach: atom-count-aware budget, capped at 24 hours.
      // ph.x perturbation cost scales as ~N_atoms² × N_electrons, so heavier
      // systems need disproportionately more time.
      const nAtoms = positions.length;
      const heavyCount = elements.filter(el => HEAVY_ELEMENTS.has(el)).length;
      // Base: 6× SCF time. Scale up for heavy/large systems.
      let phMultiplier = 6;
      if (nAtoms >= 8) phMultiplier = 8;
      if (nAtoms >= 12) phMultiplier = 10;
      if (heavyCount >= 2) phMultiplier = Math.max(phMultiplier, 10);
      const MAX_PHONON_TIMEOUT_MS = 48 * 3600 * 1000; // 48 hours absolute cap
      const phKillTimeoutMs = Math.min(effectiveKillTimeoutMs * phMultiplier, MAX_PHONON_TIMEOUT_MS);
      // max_seconds for ph.x: 120s before the kill timeout so QE can checkpoint
      // cleanly and write recover data for the next attempt.
      const phMaxSeconds = Math.floor(phKillTimeoutMs / 1000) - 120;

      // First attempt: no recover (fresh start).
      // If ph.x times out, retry once with recover=.true. to resume from checkpoint.
      // If ph.x crashes (non-timeout exit=1), retry with looser convergence and
      // gentler mixing — catches perturbation-SCF divergence and some wfc-read issues.
      let phResult: { stdout: string; stderr: string; exitCode: number } | null = null;
      let phTimedOut = false;
      let prevCrashed = false; // true if previous attempt was a non-timeout crash
      // Initialize phonon result so downstream code always has a non-null object.
      result.phonon = parsePhononOutput("");

      for (let phAttempt = 0; phAttempt < 2; phAttempt++) {
        const isRetry = phAttempt > 0;
        // On crash-retry: use looser convergence + gentler mixing instead of
        // recover=.true. (checkpoint data from a crash is likely garbage).
        const phInput = generatePhononInput(formula, elements, positions.length, {
          maxSeconds: phMaxSeconds,
          recover: isRetry && !prevCrashed,
          ...(prevCrashed ? { tr2Ph: "1.0d-10", alphaMix: 0.1 } : {}),
        });
        const phInputFile = path.join(jobDir, "ph.in");
        fs.writeFileSync(phInputFile, phInput);

        const phHours = (phKillTimeoutMs / 3600_000).toFixed(1);
        const retryInfo = prevCrashed ? ", tr2_ph=1e-10, alpha_mix=0.1" : (isRetry ? ", recover=.true." : "");
        console.log(`[QE-Worker] Starting phonon calculation for ${formula} (attempt ${phAttempt + 1}/2, timeout=${phHours}h, max_seconds=${phMaxSeconds}${retryInfo})`);

        phResult = await runQECommand(
          path.posix.join(getQEBinDir(), "ph.x"),
          phInputFile,
          jobDir,
          phKillTimeoutMs,
        );

        // Check if ph.x timed out: exit -1 with TIMEOUT marker, or QE's own
        // "Maximum CPU time exceeded" / "JOB DONE" from max_seconds.
        const combined = phResult!.stdout + phResult!.stderr;
        phTimedOut = phResult!.exitCode === -1 ||
          combined.includes("Maximum CPU time exceeded") ||
          combined.includes("max_seconds");

        // Parse what we got
        result.phonon = parsePhononOutput(phResult!.stdout);

        if (result.phonon.frequencies.length > 0) {
          // Got frequencies — success, no need to retry
          break;
        }
        if (!phTimedOut) {
          // ph.x finished (didn't timeout) but produced no frequencies.
          // Log stderr so we can actually diagnose the crash.
          const phStderrTail = phResult!.stderr.slice(-600);
          const phStdoutTail = phResult!.stdout.slice(-400);
          console.log(`[QE-Worker] ph.x exited (code=${phResult!.exitCode}) with 0 frequencies for ${formula} — not a timeout`);
          if (phStderrTail) console.log(`[QE-Worker] ph.x stderr for ${formula}: ${phStderrTail}`);
          if (phStdoutTail) console.log(`[QE-Worker] ph.x stdout tail for ${formula}: ${phStdoutTail}`);
          // Save stderr for post-mortem before retrying
          try { fs.writeFileSync(path.join(jobDir, "ph.err"), phResult!.stderr); } catch {}
          if (isRetry) {
            // Already retried with looser params — give up
            console.log(`[QE-Worker] ph.x crashed on retry for ${formula} — giving up on phonon`);
            break;
          }
          // First attempt crashed: retry with looser convergence + gentler mixing.
          // Crashes from missing wfc files, memory errors, or perturbation-SCF
          // divergence can sometimes be recovered with different parameters.
          prevCrashed = true;
          console.log(`[QE-Worker] Retrying ph.x for ${formula} with tr2_ph=1e-10, alpha_mix=0.1`);
          continue;
        }
        if (isRetry) {
          console.log(`[QE-Worker] ph.x timed out on retry for ${formula} — giving up on phonon`);
        } else {
          console.log(`[QE-Worker] ph.x timed out for ${formula} (attempt 1/2, exit=${phResult!.exitCode}) — retrying with recover=.true. to resume from checkpoint`);
        }
      }

      fs.writeFileSync(path.join(jobDir, "ph.out"), phResult!.stdout);
      if (phResult!.stderr) {
        try { fs.writeFileSync(path.join(jobDir, "ph.err"), phResult!.stderr); } catch {}
      }
      // result.phonon was already parsed inside the retry loop above.

      // Fallback: try extracting frequencies from .dyn files / dynmat.x.
      // Skip if ph.x was killed (exit -1 / timeout) — .dyn files are likely
      // incomplete and dynmat.x will just waste time on garbage data.
      const dynFilesExist = !phTimedOut && fs.readdirSync(jobDir).some(f => /\.dyn\d*$/.test(f));
      const fallbackNeeded = result.phonon.frequencies.length === 0 && !phTimedOut &&
        (phResult!.exitCode === 0 || result.phonon.converged || dynFilesExist);
      if (fallbackNeeded) {
        const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");

        // For Gamma-only phonon (1×1×1), q2r.x can't Fourier-transform
        // from a single q-point and always exits with code 2. Parse the
        // frequencies directly from the .dyn1 file instead — it contains
        // the dynamical matrix eigenvalues in cm⁻¹ after "freq (" or
        // "omega(" lines.
        const [pnq1, pnq2, pnq3] = autoPhononQGrid(elements, positions.length);
        if (pnq1 === 1 && pnq2 === 1 && pnq3 === 1) {
          // ldisp=.false. writes {prefix}.dyn; ldisp=.true. writes {prefix}.dyn1.
          // Check both so this fallback works regardless of which mode was used.
          const dyn1Path = path.join(jobDir, `${prefix}.dyn1`);
          const dynPath = path.join(jobDir, `${prefix}.dyn`);
          const dynFilePath = fs.existsSync(dyn1Path) ? dyn1Path : (fs.existsSync(dynPath) ? dynPath : null);
          if (dynFilePath) {
            const dynFileName = path.basename(dynFilePath);
            // Step 1: Try parsing freq lines directly from .dyn/.dyn1
            try {
              const dynContent = fs.readFileSync(dynFilePath, "utf8");
              const freqValues: number[] = [];
              for (const line of dynContent.split("\n")) {
                const cm1Match = line.match(/\[\s*cm-1\s*\]\s*$/i) ? line.match(/([-\d.]+)\s*\[\s*cm-1\s*\]/i) : null;
                const freqMatch = line.match(/freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/i);
                const omegaMatch = line.match(/omega\s*\(\s*\d+\)\s*=\s*([-\d.]+)/);
                const val = freqMatch ? parseFloat(freqMatch[2]) : (cm1Match ? parseFloat(cm1Match[1]) : (omegaMatch ? parseFloat(omegaMatch[1]) : NaN));
                if (Number.isFinite(val) && val !== 0) freqValues.push(val);
              }
              if (freqValues.length > 0) {
                result.phonon.frequencies = freqValues;
                result.phonon.lowestFrequency = Math.min(...freqValues);
                result.phonon.highestFrequency = Math.max(...freqValues);
                result.phonon.imaginaryCount = freqValues.filter(f => f < -20).length;
                result.phonon.hasImaginary = result.phonon.imaginaryCount > 0;
                result.phonon.converged = true;
                console.log(`[QE-Worker] Parsed ${freqValues.length} Gamma-only phonon modes from ${dynFileName} for ${formula} (lowest=${result.phonon.lowestFrequency.toFixed(1)} cm⁻¹)`);
              }
            } catch (dynErr: any) {
              console.log(`[QE-Worker] Failed to parse ${dynFileName} for ${formula}: ${(dynErr.message || "").slice(0, 100)}`);
            }

            // Step 2: If .dyn file didn't have freq lines, run dynmat.x to extract
            // frequencies from the raw dynamical matrix. dynmat.x diagonalizes the
            // matrix and prints frequencies.
            if (result.phonon.frequencies.length === 0) {
              try {
                const dynmatInput = `&INPUT\n  fildyn = '${path.basename(dynFilePath)}',\n  asr = 'simple',\n/\n`;
                const dynmatInputFile = path.join(jobDir, "dynmat.in");
                fs.writeFileSync(dynmatInputFile, dynmatInput);
                console.log(`[QE-Worker] Running dynmat.x for Gamma-only phonon of ${formula}`);
                const dynmatResult = await runQECommand(
                  path.posix.join(getQEBinDir(), "dynmat.x"),
                  dynmatInputFile, jobDir,
                  5 * 60 * 1000, // 5 min — dynmat.x is a fast diagonalization
                );
                fs.writeFileSync(path.join(jobDir, "dynmat.out"), dynmatResult.stdout);

                if (dynmatResult.exitCode === 0) {
                  const dynmatPhonon = parsePhononOutput(dynmatResult.stdout);
                  if (dynmatPhonon.frequencies.length > 0) {
                    result.phonon = dynmatPhonon;
                    result.phonon.converged = true;
                    console.log(`[QE-Worker] dynmat.x extracted ${dynmatPhonon.frequencies.length} modes for ${formula} (lowest=${dynmatPhonon.lowestFrequency.toFixed(1)} cm⁻¹)`);
                  } else {
                    // dynmat.x outputs in multiple formats depending on QE version:
                    //   "mode   N     freq(cm**-1) = X.XXXX"
                    //   tabular: "# mode   [cm-1]   [THz]  IR\n   1   123.45   3.678   0.12"
                    const freqValues: number[] = [];
                    let inTabular = false;
                    for (const line of dynmatResult.stdout.split("\n")) {
                      // Format 1: "mode   N     freq(cm**-1) = X.XXXX"
                      const modeMatch = line.match(/mode\s+\d+\s+freq\s*\(\s*cm\*?\*?-1\s*\)\s*=\s*([-\d.]+)/i);
                      if (modeMatch) {
                        const val = parseFloat(modeMatch[1]);
                        if (Number.isFinite(val)) freqValues.push(val);
                        continue;
                      }
                      // Format 2: tabular — header "# mode   [cm-1]" followed by
                      // "   N      XXX.XX    Y.YYYY    Z.ZZZZ" lines
                      if (line.match(/#\s*mode\s+\[cm-1\]/i)) {
                        inTabular = true;
                        continue;
                      }
                      if (inTabular) {
                        const cols = line.trim().split(/\s+/);
                        // Expect: modeNumber  freq_cm1  freq_THz  [IR_activity ...]
                        if (cols.length >= 2 && /^\d+$/.test(cols[0])) {
                          const val = parseFloat(cols[1]);
                          if (Number.isFinite(val)) freqValues.push(val);
                        } else if (line.trim() === "" || line.startsWith("*")) {
                          inTabular = false;
                        }
                      }
                    }
                    if (freqValues.length > 0) {
                      result.phonon.frequencies = freqValues;
                      result.phonon.lowestFrequency = Math.min(...freqValues);
                      result.phonon.highestFrequency = Math.max(...freqValues);
                      result.phonon.imaginaryCount = freqValues.filter(f => f < -20).length;
                      result.phonon.hasImaginary = result.phonon.imaginaryCount > 0;
                      result.phonon.converged = true;
                      console.log(`[QE-Worker] dynmat.x mode-parse extracted ${freqValues.length} frequencies for ${formula}`);
                    } else {
                      console.log(`[QE-Worker] dynmat.x produced no parseable frequencies for ${formula}`);
                    }
                  }
                } else {
                  console.log(`[QE-Worker] dynmat.x failed for ${formula}: exit ${dynmatResult.exitCode}`);
                }
              } catch (dynmatErr: any) {
                console.log(`[QE-Worker] dynmat.x error for ${formula}: ${(dynmatErr.message || "").slice(0, 120)}`);
              }
            }
          }

          // Also try parsing ph.x stdout directly — with ldisp=.false.,
          // ph.x prints frequencies to stdout in omega(N) format
          if (result.phonon.frequencies.length === 0) {
            const stdoutPhonon = parsePhononOutput(phResult!.stdout);
            if (stdoutPhonon.frequencies.length > 0) {
              result.phonon = stdoutPhonon;
              result.phonon.converged = true;
              console.log(`[QE-Worker] Found ${stdoutPhonon.frequencies.length} modes in ph.x stdout for Gamma-only ${formula}`);
            }
          }
          // Skip q2r.x+matdyn.x — not applicable for Gamma-only
        } else {
        // Guard: q2r.x needs the .dyn0 summary + every .dynN file listed
        // inside it (one per irreducible q-point). If ph.x was killed
        // mid-run (exit=-1 case), partial dyn sets will make q2r.x fail
        // with exit=2 and no useful diagnostic. Validate the set first.
        const dyn0Path = path.join(jobDir, `${prefix}.dyn0`);
        let dynSetComplete = false;
        let dynReason = "no dyn0 file";
        if (fs.existsSync(dyn0Path)) {
          try {
            const dyn0Content = fs.readFileSync(dyn0Path, "utf8");
            const qCountMatch = dyn0Content.trim().split("\n")[0]?.trim().split(/\s+/);
            const nQ = qCountMatch ? parseInt(qCountMatch[0]) || parseInt(qCountMatch[2] || "0") : 0;
            if (nQ > 0) {
              let missing = 0;
              for (let iq = 1; iq <= nQ; iq++) {
                if (!fs.existsSync(path.join(jobDir, `${prefix}.dyn${iq}`))) missing++;
              }
              if (missing === 0) dynSetComplete = true;
              else dynReason = `${missing}/${nQ} dyn files missing`;
            } else {
              dynReason = "could not parse q-point count from dyn0";
            }
          } catch (e: any) {
            dynReason = `dyn0 read error: ${(e.message || "").slice(0, 80)}`;
          }
        }
        if (!dynSetComplete) {
          console.log(`[QE-Worker] Skipping q2r.x fallback for ${formula} — incomplete dyn set (${dynReason}); ph.x was likely killed before producing all q-points`);
        } else {
        console.log(`[QE-Worker] Running q2r.x+matdyn.x fallback for ${formula} (exit=${phResult!.exitCode}, conv=${result.phonon.converged}, dyn=${dynFilesExist})`);
        try {
          // q2r.x: convert dynamical matrices to real-space force constants
          const q2rInput = `&INPUT\n  fildyn = '${prefix}.dyn',\n  zasr = 'simple',\n  flfrc = '${prefix}.fc'\n/\n`;
          const q2rInputFile = path.join(jobDir, "q2r.in");
          fs.writeFileSync(q2rInputFile, q2rInput);
          const q2rResult = await runQECommand(
            path.posix.join(getQEBinDir(), "q2r.x"),
            q2rInputFile, jobDir,
            5 * 60 * 1000, // 5 min cap
          );
          fs.writeFileSync(path.join(jobDir, "q2r.out"), q2rResult.stdout);
          console.log(`[QE-Worker] q2r.x exit=${q2rResult.exitCode} for ${formula}`);

          if (q2rResult.exitCode === 0) {
            // matdyn.x: compute phonon DOS and frequencies on a fine grid
            const matdynInput = `&INPUT\n  asr = 'simple',\n  flfrc = '${prefix}.fc',\n  flfrq = '${prefix}.freq',\n  dos = .true.,\n  fldos = '${prefix}.phdos',\n  nk1 = 10, nk2 = 10, nk3 = 10,\n/\n`;
            const matdynInputFile = path.join(jobDir, "matdyn.in");
            fs.writeFileSync(matdynInputFile, matdynInput);
            const matdynResult = await runQECommand(
              path.posix.join(getQEBinDir(), "matdyn.x"),
              matdynInputFile, jobDir,
              5 * 60 * 1000, // 5 min cap
            );
            fs.writeFileSync(path.join(jobDir, "matdyn.out"), matdynResult.stdout);

            // Parse frequencies from matdyn.x output
            const matdynPhonon = parsePhononOutput(matdynResult.stdout);
            if (matdynPhonon.frequencies.length > 0) {
              result.phonon = matdynPhonon;
              console.log(`[QE-Worker] matdyn.x extracted ${matdynPhonon.frequencies.length} modes for ${formula}`);
            } else {
              // Try parsing the .freq file directly
              try {
                const freqFile = path.join(jobDir, `${prefix}.freq`);
                if (fs.existsSync(freqFile)) {
                  const freqContent = fs.readFileSync(freqFile, "utf8");
                  const freqValues: number[] = [];
                  for (const line of freqContent.split("\n")) {
                    const nums = line.trim().split(/\s+/).map(Number).filter(n => Number.isFinite(n) && n !== 0);
                    freqValues.push(...nums);
                  }
                  if (freqValues.length > 0) {
                    result.phonon.frequencies = freqValues;
                    result.phonon.lowestFrequency = Math.min(...freqValues);
                    result.phonon.highestFrequency = Math.max(...freqValues);
                    result.phonon.imaginaryCount = freqValues.filter(f => f < -20).length;
                    result.phonon.hasImaginary = result.phonon.imaginaryCount > 0;
                    result.phonon.converged = true;
                    console.log(`[QE-Worker] Parsed ${freqValues.length} frequencies from ${prefix}.freq file`);
                  }
                }
              } catch { /* freq file parse failed */ }
            }
          } else {
            console.log(`[QE-Worker] q2r.x failed for ${formula}: exit ${q2rResult.exitCode}`);
          }
        } catch (postErr: any) {
          console.log(`[QE-Worker] Phonon post-processing failed for ${formula}: ${postErr.message?.slice(0, 150)}`);
        }
        } // close else (dyn set complete branch)
        } // close else (non-Gamma q-grid → q2r.x path)
      }

      if (phResult!.exitCode !== 0 && !result.phonon.converged) {
        result.phonon.error = `ph.x exited with code ${phResult!.exitCode}: ${phResult!.stderr.slice(-500)}`;
        result.failureStage = "phonon";
        stageFailureCounts.phonon++;
        console.log(`[QE-Worker] Phonon failed for ${formula}: ${result.phonon.error.slice(-200)}`);
      } else if (result.phonon.frequencies.length === 0) {
        // ph.x may have exited 0 (via max_seconds clean exit) but produced
        // no frequencies. This is still a failure — mark it explicitly so
        // downstream consumers (DFPT, Tc estimation) don't run on empty data.
        const reason = phTimedOut ? "timeout (all perturbations not completed)" : "no frequencies parsed from output";
        result.phonon.error = `Phonon produced 0 modes: ${reason}`;
        result.failureStage = "phonon";
        stageFailureCounts.phonon++;
        console.log(`[QE-Worker] Phonon failed for ${formula}: 0 modes — ${reason} (exit=${phResult!.exitCode})`);
      } else {
        console.log(`[QE-Worker] Phonon done for ${formula}: ${result.phonon.frequencies.length} modes, lowest=${result.phonon.lowestFrequency.toFixed(1)} cm-1`);

        // --- Stage 5 convergence diagnostic ---
        // Compare full-grid phonon with Gamma check (Stage 4) to detect
        // q-grid convergence issues. If full grid finds significantly more
        // negative modes than Gamma, the structure may need a denser q-grid.
        if (result.gammaPhononPassed && result.phonon.hasImaginary) {
          console.log(`[QE-Worker] Stage 5 warning: ${formula} Gamma check passed but full grid found ${result.phonon.imaginaryCount} imaginary modes (lowest=${result.phonon.lowestFrequency.toFixed(1)} cm-1) — q-grid convergence issue or zone-boundary instability`);
        }
        // Log full-grid vs Gamma frequency comparison for monitoring
        if (result.phonon.frequencies.length > 0) {
          const fullLowest = result.phonon.lowestFrequency;
          const fullHighest = result.phonon.highestFrequency;
          console.log(`[QE-Worker] Stage 5 spectrum: ${formula} ${result.phonon.frequencies.length} modes, range [${fullLowest.toFixed(1)}, ${fullHighest.toFixed(1)}] cm-1, ${result.phonon.imaginaryCount} imaginary`);
        }
      }
    }

    // ── Bands AFTER phonon (cycle 1374 fix) ──────────────────────────────────
    // Safe to run last because nothing downstream reads ./tmp/${prefix}.save/
    // after this point. DFPT EPC below uses runDFPTEPC which writes its own
    // ph.in and reads only the lambda/omega from its own stdout — it does not
    // depend on .save/ wavefunctions surviving the bands step.
    if (scfUsable && !isPartialWalltime && result.scf?.fermiEnergy !== null) {
      try {
        const cOverAVal = estimateCOverA(elements, counts);
        // Uses module-level SPECIES_ECUTWFC. Must match the SCF-path cutoff
        // so the bands calculation reads the same .save/ wavefunctions.
        const hasHydrogenBands = elements.includes("H");
        const rawEcutwfcBands = computeEcutwfc(elements, 0, 80, 45);
        const baseEcutwfcBands = Math.max(rawEcutwfcBands, hasHydrogenBands ? 100 : 60);
        // Use the same broad detector as the SCF — bands must run with the
        // same nspin or it will fail to read the SCF .save/ wavefunctions.
        const nspinBands = mayHaveMagneticMoment(elements) ? 2 : 1;

        const bOverAVal = estimateBOverA(elements, counts);
        const latticeBVal = latticeA * bOverAVal;

        // Isolate band structure workspace: copy .save/ so bands.x doesn't
        // clobber the phonon wavefunctions. This prevents the "0 modes" bug
        // where band-path wavefunctions overwrite SCF wavefunctions.
        const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
        const saveSrc = path.join(jobDir, "tmp", `${prefix}.save`);
        const bandsDir = path.join(jobDir, "bands_workspace");
        const bandsSaveDst = path.join(bandsDir, "tmp", `${prefix}.save`);
        try {
          if (fs.existsSync(saveSrc)) {
            fs.mkdirSync(path.join(bandsDir, "tmp"), { recursive: true });
            fs.cpSync(saveSrc, bandsSaveDst, { recursive: true });
            console.log(`[QE-Worker] Copied .save/ to isolated bands workspace for ${formula}`);
          }
        } catch (copyErr: any) {
          console.log(`[QE-Worker] Failed to copy .save/ for bands: ${copyErr.message?.slice(0, 80)} — using shared workspace`);
        }
        const bandJobDir = fs.existsSync(bandsSaveDst) ? bandsDir : jobDir;

        // Use the same PAW/USPP ecutrho multiplier as the SCF to avoid FFT grid OOM.
        const ecutrhoForBands = baseEcutwfcBands * ecutrhoMultiplier(elements);
        const bandResult = await computeDFTBandStructure(
          formula,
          elements,
          counts,
          latticeA,
          positions,
          result.scf!.fermiEnergy,
          bandJobDir,
          cOverAVal,
          baseEcutwfcBands,
          nspinBands,
          latticeBVal,
          ecutrhoForBands,
        );

        result.bandStructure = bandResult;
        recordBandCalcOutcome(bandResult.converged, bandResult.wallTimeSeconds);

        if (!bandResult.converged && bandResult.error) {
          stageFailureCounts.bands++;
          console.log(`[QE-Worker] Band structure failed for ${formula}: ${bandResult.error.slice(-200)}`);
        } else {
          console.log(`[QE-Worker] Band structure done for ${formula}: ${bandResult.nBands} bands, ${bandResult.bandCrossings.length} crossings, flat=${bandResult.flatBandScore.toFixed(3)}`);
        }
      } catch (bandErr: any) {
        stageFailureCounts.bands++;
        console.log(`[QE-Worker] Band structure error for ${formula}: ${bandErr.message?.slice(-200) ?? bandErr}`);
      }
    }

    // --- DFT Quality Gate before Tc estimation ---
    // Prevents unstable or poorly relaxed structures from getting
    // impressive-looking Tc numbers. Force thresholds are tiered:
    //   screening: force < 0.10 (rough structure, surrogate Tc only)
    //   DFPT e-ph: force < 0.03 (physics-grade e-ph requires equilibrium)
    //   publication: force < 0.01 (fully relaxed)
    const phononHasResults = result.phonon != null && result.phonon.frequencies.length > 0;
    const phononPhysicallyStable = phononHasResults
      ? result.phonon!.lowestFrequency > -10.0
      : false;
    const residualForce = result.scf?.totalForce ?? null;
    const scfForceOkScreening = residualForce != null ? residualForce < 0.10 : true;
    const scfForceOkDFPT = residualForce != null ? residualForce < 0.03 : true;
    const scfForceOkPublication = residualForce != null ? residualForce < 0.01 : true;
    const scfPressureOk = result.scf?.pressure != null ? Math.abs(result.scf.pressure) < 50 : true;
    const isMetallicForTc = result.scf?.isMetallic ?? false;

    // Basic quality gate (allows surrogate Tc)
    const qualityGatePass = scfConverged && scfForceOkScreening && scfPressureOk && isMetallicForTc && phononPhysicallyStable;
    // DFPT gate (allows physics-grade e-ph — tighter force)
    const dfptGatePass = qualityGatePass && scfForceOkDFPT;
    const qualityGateReason: string[] = [];
    if (!scfConverged) qualityGateReason.push("SCF not converged");
    if (!scfForceOkScreening) qualityGateReason.push(`force ${residualForce?.toFixed(3)} > 0.10 Ry/bohr (screening threshold)`);
    else if (!scfForceOkDFPT) qualityGateReason.push(`force ${residualForce?.toFixed(3)} > 0.03 Ry/bohr (DFPT threshold — surrogate Tc only)`);
    if (!scfPressureOk) qualityGateReason.push(`pressure ${result.scf?.pressure?.toFixed(1)} kbar > ±50`);
    if (!isMetallicForTc) qualityGateReason.push("not metallic");
    if (!phononPhysicallyStable) qualityGateReason.push(phononHasResults ? "imaginary phonon modes" : "no phonon data");

    if (dfptGatePass) {
      console.log(`[QE-Worker] Quality gate PASSED (DFPT-ready) for ${formula}: force=${residualForce?.toFixed(4) ?? "N/A"} < 0.03, pressure=${result.scf?.pressure?.toFixed(1) ?? "N/A"} kbar, phonon stable (${result.phonon!.frequencies.length} modes)`);
    } else if (qualityGatePass) {
      console.log(`[QE-Worker] Quality gate PASSED (screening only) for ${formula}: force=${residualForce?.toFixed(4) ?? "N/A"} (> 0.03, below DFPT threshold) — surrogate Tc allowed, DFPT skipped`);
    } else {
      console.log(`[QE-Worker] Quality gate FAILED for ${formula}: ${qualityGateReason.join(", ")} — Tc will be labeled as surrogate`);
    }

    // Quality-weighted learning: phonon stability is a strong signal (weight 3.0)
    if (phononHasResults && phononPhysicallyStable) {
      const winnerProto = result.prototypeUsed ?? "unknown";
      recordGeneratorOutcome(elements, workerPressure, winnerProto, true, getSignalWeight("phonon_stable"));
    }

    // DFPT electron-phonon coupling — requires DFPT gate (force < 0.03) + high ensemble score.
    // If force is between 0.03-0.10, screening passes but DFPT is skipped (force too high
    // for reliable e-ph matrix elements).
    if (scfUsable && dfptGatePass && (opts?.ensembleScore ?? 0) > 0.7 && !opts?.skipEph) {
      console.log(`[QE-Worker] ${formula} qualifies for DFPT EPC (ensembleScore=${opts!.ensembleScore!.toFixed(3)}, phononModes=${result.phonon!.frequencies.length}, lowestFreq=${result.phonon!.lowestFrequency.toFixed(1)} cm⁻¹)`);
      try {
        result.dfpt = await runDFPTEPC(formula, elements, counts, jobDir, workerPressure);

        // Quality-weighted learning: DFPT e-ph success is the strongest signal (weight 4.0)
        if (result.dfpt && (result.dfpt as any).lambda > 0) {
          const winnerProto = result.prototypeUsed ?? "unknown";
          recordGeneratorOutcome(elements, workerPressure, winnerProto, true, getSignalWeight("dfpt_good_lambda"));
        }
      } catch (dfptErr: any) {
        console.log(`[QE-Worker] DFPT EPC failed for ${formula}: ${(dfptErr.message ?? "").slice(-200)}`);
      }
    } else if (opts?.skipEph) {
      console.log(`[QE-Worker] ${formula} DFPT EPC skipped — Stoner ferromagnet flag set`);
    } else if (!phononHasResults && (opts?.ensembleScore ?? 0) > 0.7) {
      console.log(`[QE-Worker] ${formula} DFPT EPC skipped — phonon produced 0 modes (timeout/crash), no data to build on`);
    }

    // --- Populate quality gate and uncertainty fields ---
    result.qualityGatePassed = qualityGatePass;
    result.qualityGateReasons = qualityGateReason;

    // Determine uncertainty/confidence for each result dimension
    const hasDFPT = result.dfpt != null && (result.dfpt as any).lambda > 0;
    const hasFullPhonon = phononHasResults && result.phonon!.frequencies.length >= 10;
    const hasGammaOnly = phononHasResults && result.phonon!.frequencies.length < 10 && result.phonon!.frequencies.length > 0;

    const tcConfidence: "high" | "medium" | "low" | "surrogate" =
      hasDFPT && dfptGatePass ? "high" :
      hasFullPhonon && qualityGatePass ? "medium" :
      scfConverged && isMetallicForTc ? "low" : "surrogate";

    const lambdaConfidence: "high" | "medium" | "low" | "surrogate" =
      hasDFPT ? "high" :
      hasFullPhonon ? "medium" : "surrogate";

    const phononConfidence: "high" | "medium" | "low" | "none" =
      hasFullPhonon && phononPhysicallyStable ? "high" :
      hasGammaOnly ? "medium" :
      phononHasResults ? "low" : "none";

    const structureConfidence: "high" | "medium" | "low" =
      result.vcRelaxed && scfForceOkDFPT && scfPressureOk ? "high" :
      scfConverged ? "medium" : "low";

    const ephMethod: "dfpt" | "surrogate" | "none" =
      hasDFPT ? "dfpt" : "surrogate";

    const phononMethod: "dfpt_full" | "dfpt_gamma" | "finite_displacement" | "surrogate" | "none" =
      hasFullPhonon ? "dfpt_full" :
      hasGammaOnly ? "dfpt_gamma" : "none";

    const tcReasons: string[] = [];
    if (hasDFPT) tcReasons.push("DFPT e-ph coupling");
    else tcReasons.push("surrogate lambda");
    if (hasFullPhonon) tcReasons.push("full DFPT phonons");
    else if (hasGammaOnly) tcReasons.push("gamma-only phonons");
    else tcReasons.push("no phonon data");
    if (scfConverged) tcReasons.push("SCF converged");
    else tcReasons.push("SCF partial/failed");
    if (!scfForceOkScreening) tcReasons.push(`high force (${result.scf?.totalForce?.toFixed(3)} > 0.10)`);
    if (!scfPressureOk) tcReasons.push(`high pressure (${result.scf?.pressure?.toFixed(1)} kbar)`);

    result.uncertainty = {
      tcConfidence,
      tcUncertaintyReason: tcReasons.join(" + "),
      lambdaConfidence,
      phononConfidence,
      structureConfidence,
      ephMethod,
      phononMethod,
    };

    console.log(`[QE-Worker] ${formula} uncertainty: Tc=${tcConfidence}, lambda=${lambdaConfidence}, phonon=${phononConfidence}, structure=${structureConfidence}, eph=${ephMethod}, phonon_method=${phononMethod}`);
    console.log(`[QE-Worker] ${formula} Tc uncertainty reason: ${result.uncertainty.tcUncertaintyReason}`);

    // --- Reproducibility bundle for screening_converged+ ---
    // Note: at this point qualityTier is at most "relaxed" — final_converged/publication_ready
    // are assigned later based on phonon/DFPT. Save bundle for any non-failed tier.
    const bundleTier = result.qualityTier ?? "failed";
    if (bundleTier !== "failed") {
      try {
        const bundleDir = path.join(jobDir, "reproducibility_bundle");
        fs.mkdirSync(bundleDir, { recursive: true });

        fs.writeFileSync(path.join(bundleDir, "quality_report.json"), JSON.stringify({
          formula, qualityTier: bundleTier,
          qualityGatePassed: result.qualityGatePassed,
          qualityGateReasons: result.qualityGateReasons,
          uncertainty: result.uncertainty,
          hullStability: result.hullStability,
          scfConverged: result.scf?.converged,
          scfAccuracy: result.scf?.lastScfAccuracyRy,
          residualForce: result.scf?.totalForce,
          residualPressure: result.scf?.pressure,
          isMetallic: result.scf?.isMetallic,
          bandGap: result.scf?.bandGap,
          phononModes: result.phonon?.frequencies?.length ?? 0,
          phononStable: result.phonon ? result.phonon.lowestFrequency > -10 : null,
          hasDFPT: result.dfpt != null,
          dfptMethodLabels: result.dfpt ? { alpha2F: (result.dfpt as any).alpha2FMethod, lambda: (result.dfpt as any).lambdaMethod } : null,
          wallTimeTotal: (Date.now() - startTime) / 1000,
          timestamp: new Date().toISOString(),
        }, null, 2));

        fs.writeFileSync(path.join(bundleDir, "candidate_provenance.json"), JSON.stringify({
          formula, prototypeUsed: result.prototypeUsed, provenance: result.provenance,
          vegardEstimate: result.vegardEstimate,
          structureCandidatesEvaluated: result.structureCandidatesEvaluated,
          stagedRelaxation: result.stagedRelaxation,
          latticeA, pressureGPa: workerPressure, elements, counts, nAtoms: positions.length,
        }, null, 2));

        // Final structure as POSCAR
        if (positions.length > 0) {
          let poscar = `${formula} (${bundleTier})\n1.0\n`;
          const cOA = estimateCOverA(elements, counts);
          poscar += `  ${latticeA.toFixed(6)}  0.000000  0.000000\n`;
          poscar += `  0.000000  ${latticeA.toFixed(6)}  0.000000\n`;
          poscar += `  0.000000  0.000000  ${(latticeA * cOA).toFixed(6)}\n`;
          const elOrder = [...new Set(positions.map(p => p.element))].sort();
          poscar += elOrder.join(" ") + "\n";
          poscar += elOrder.map(el => positions.filter(p => p.element === el).length).join(" ") + "\n";
          poscar += "Direct\n";
          for (const el of elOrder) {
            for (const p of positions.filter(pp => pp.element === el)) {
              poscar += `  ${p.x.toFixed(8)}  ${p.y.toFixed(8)}  ${p.z.toFixed(8)}\n`;
            }
          }
          fs.writeFileSync(path.join(bundleDir, "final_structure.poscar"), poscar);
        }
        if (result.scf) fs.writeFileSync(path.join(bundleDir, "scf_summary.json"), JSON.stringify({ totalEnergy: result.scf.totalEnergy, fermiEnergy: result.scf.fermiEnergy, totalForce: result.scf.totalForce, pressure: result.scf.pressure, converged: result.scf.converged, isMetallic: result.scf.isMetallic, bandGap: result.scf.bandGap }, null, 2));
        if (result.dfpt) fs.writeFileSync(path.join(bundleDir, "dfpt_results.json"), JSON.stringify(result.dfpt, null, 2));
        if (result.phonon) fs.writeFileSync(path.join(bundleDir, "phonon_summary.json"), JSON.stringify({ frequencies: result.phonon.frequencies, lowestFrequency: result.phonon.lowestFrequency, highestFrequency: result.phonon.highestFrequency, imaginaryCount: result.phonon.imaginaryCount, hasImaginary: result.phonon.hasImaginary, converged: result.phonon.converged }, null, 2));

        console.log(`[QE-Worker] Reproducibility bundle saved for ${formula} (tier=${bundleTier}) at ${bundleDir}`);
      } catch (bundleErr: any) {
        console.log(`[QE-Worker] Failed to save reproducibility bundle for ${formula}: ${bundleErr.message?.slice(0, 100)}`);
      }
    }

    // --- Method-based quality tier caps ---
    // xTB/surrogate phonons and e-ph cannot reach high quality tiers.
    // Only full DFPT can be publication_ready.
    const currentTier = result.qualityTier ?? "failed";
    const tierRank: Record<string, number> = {
      failed: 0, partial_screening: 1, screening_converged: 2,
      relaxed: 3, final_converged: 4, publication_ready: 5,
    };
    // Max tier based on phonon method
    const phononMethodMaxTier: Record<string, string> = {
      dfpt_full: "publication_ready",
      dfpt_gamma: "final_converged",
      finite_displacement: "screening_converged",
      surrogate: "screening_converged",
      none: "relaxed",
    };
    // Max tier based on e-ph method
    const ephMethodMaxTier: Record<string, string> = {
      dfpt: "publication_ready",
      surrogate: "screening_converged",
      none: "relaxed",
    };
    const phMaxTier = phononMethodMaxTier[phononMethod] ?? "relaxed";
    const ephMaxTier = ephMethodMaxTier[ephMethod] ?? "relaxed";
    const effectiveMaxTier = (tierRank[phMaxTier] ?? 0) < (tierRank[ephMaxTier] ?? 0) ? phMaxTier : ephMaxTier;

    if ((tierRank[currentTier] ?? 0) > (tierRank[effectiveMaxTier] ?? 0)) {
      console.log(`[QE-Worker] Tier cap: ${formula} downgraded from ${currentTier} to ${effectiveMaxTier} (phonon_method=${phononMethod}, eph_method=${ephMethod})`);
      result.qualityTier = effectiveMaxTier as typeof result.qualityTier;
    }

  } catch (err: any) {
    result.error = err.message;
    console.log(`[QE-Worker] Error for ${formula}: ${err.message}`);
  } finally {
    // Tiered cleanup — keep logs/structures for debugging, delete heavy scratch
    try {
      // Always delete heavy QE scratch (wavefunctions, charge density)
      const tmpDir = path.join(jobDir, "tmp");
      if (fs.existsSync(tmpDir)) {
        fs.rmSync(tmpDir, { recursive: true, force: true });
      }
      const bandsTmp = path.join(jobDir, "bands_workspace");
      if (fs.existsSync(bandsTmp)) {
        fs.rmSync(bandsTmp, { recursive: true, force: true });
      }

      // For failed/rejected: keep error logs for 7 days, delete rest
      // For promising: keep QE inputs/outputs + structures
      // Delete everything except .in/.out/.json files and reproducibility bundle
      if (fs.existsSync(jobDir)) {
        const files = fs.readdirSync(jobDir);
        for (const f of files) {
          if (f === "reproducibility_bundle") continue; // Keep bundle
          const fp = path.join(jobDir, f);
          const stat = fs.statSync(fp);
          if (stat.isDirectory()) {
            try { fs.rmSync(fp, { recursive: true, force: true }); } catch {}
          }
          else if (!f.endsWith(".in") && !f.endsWith(".out") && !f.endsWith(".json")) {
            try { fs.unlinkSync(fp); } catch {}
          }
        }
      }
    } catch (cleanErr: any) {
      // Don't block on cleanup failure
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
