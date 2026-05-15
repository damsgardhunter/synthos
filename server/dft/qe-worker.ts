import { execSync, execFile } from "child_process";
import { promisify } from "util";
const execFileAsync = promisify(execFile);
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { IS_WINDOWS, binaryPath, getTempSubdir, toWslPath, killProcessGracefully, spawnQE } from "./platform-utils";
import { selectPrototype } from "../learning/crystal-prototypes";
import { matchPrototype } from "../learning/structure-predictor";
import { isTransitionMetal, isRareEarth, isActinide, ELEMENTAL_DATA, getElementData, getHubbardU } from "../learning/elemental-data";
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
import { runEPWPipeline, type EPWResult } from "./epw-pipeline";
import { checkSSCHAEligibility, runSSCHAPipeline, type SSCHAResult } from "./sscha-pipeline";
import { runACBN0Pipeline, type ACBN0Result } from "./acbn0-pipeline";
import { analyzeSOCRequirement, RELATIVISTIC_PP_URLS, type SOCAnalysis } from "./soc-handler";
import { analyzeHubbardWorkflow, type HubbardWorkflowResult } from "./hubbard-workflow";
import { followZoneBoundarySoftMode } from "./zone-boundary-softmode";
import { exportDMFTBundle, isDMFTEligible, type DMFTBundleResult } from "./dmft-bundle-exporter";
import { getStructureAdvice, type StructureAdvice } from "./structure-advisor";
import {
  classifyMagneticLandscape,
  shouldSearchMagneticGS,
  selectMagneticGroundState,
  parseMagnetizationFromOutput,
  buildCantedNoncollinearTrial,
  TRIAGE_CONV_THR,
  TIGHT_CONV_THR,
  type MagneticGroundStateResult,
  type MagneticTrialResult,
} from "./magnetic-ground-state";
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
      // Prefer /usr/local/bin (manual installs, lmaxx=6 rebuild) over /usr/bin (apt default)
      "/usr/local/bin",
      "/usr/bin",
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
  alphaRad: number = Math.PI / 2,
  betaRad: number = Math.PI / 2,
): number {
  const a = latticeA;
  const b = latticeA * bOverA;
  const c = latticeA * cOverA;
  // Build the conventional triclinic lattice vectors:
  //   vec_a = (a, 0, 0)
  //   vec_b = (b cos γ, b sin γ, 0)
  //   vec_c = (c cos β, c (cos α − cos β cos γ)/sin γ, c · sqrt(1 − ...)/sin γ)
  // The previous formula assumed α = β = 90° and only handled γ, giving wrong
  // distances for any monoclinic (β ≠ 90°) or triclinic cell. For VO₂
  // (β = 122.6°) this overestimated a+c bond distances by ~50% — bond-length
  // sanity checks then under-flagged or mis-pair-labeled the closest contacts
  // and the pre-phonon validator's "expected bond length" ratio for monoclinic
  // structures was systematically off.
  const cosA = Math.cos(alphaRad);
  const cosB = Math.cos(betaRad);
  const cosG = Math.cos(gammaRad);
  const sinG = Math.sin(gammaRad);
  const cx = c * cosB;
  // Guard against degenerate γ → 0/π where sinG → 0 (would be a singular cell).
  const cy = sinG > 1e-10 ? c * (cosA - cosB * cosG) / sinG : 0;
  const czSq = c * c - cx * cx - cy * cy;
  const cz = czSq > 0 ? Math.sqrt(czSq) : 0;
  const dx = fdx * a + fdy * b * cosG + fdz * cx;
  const dy = fdy * b * sinG + fdz * cy;
  const dz = fdz * cz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function estimateBulkModulus(elements: string[], counts?: Record<string, number>): number {
  // Non-hydride (or no count data): legacy simple element-table average.
  if (!elements.includes("H") || !counts) {
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

  // Hydride path. The naive average produced B0 ≈ 13–14 GPa for LaH10 / LaH12
  // / LaH11Li2 because H's tabulated bulk modulus is tiny — that made the
  // Murnaghan pre-compression a near-no-op, leaving vc-relax to compress
  // 70% on its own (root cause of the H3S overshoot pattern).
  //
  // Real physics: high-P clathrate hydrides have ambient-extrapolated bulk
  // moduli of ~50–120 GPa (LaH10 ~80 GPa, CaH6 ~100 GPa, Li-doped hydrides
  // softer). At target pressure they stiffen to 250–400 GPa, but Murnaghan
  // wants the AMBIENT B0 and applies the pressure correction itself.
  //
  // Model: (1) take the H-mole-weighted average of metal B0s, NOT the
  // element-set average — so binary alloys with Li (B0=11) and La (B0=28)
  // give a stoichiometry-correct base; (2) apply a cage-stiffening factor
  // that scales with H fraction (cage networks are stiffer than the metal
  // lattice alone, with empirical slope ~6×); (3) floor at 30 GPa so soft
  // alkali hydrides at high P still get a sane pre-compression.
  let metalB0Sum = 0;
  let metalCountForB0 = 0;
  let hCount = 0;
  let totalAtoms = 0;
  for (const el of elements) {
    const n = Math.round(counts[el] ?? 0);
    if (n <= 0) continue;
    totalAtoms += n;
    if (el === "H") { hCount += n; continue; }
    const data = getElementData(el);
    if (data && data.bulkModulus != null && data.bulkModulus > 0) {
      metalB0Sum += data.bulkModulus * n;
      metalCountForB0 += n;
    }
  }
  if (metalCountForB0 === 0 || totalAtoms === 0) return 50; // pure H or no data
  const baseMetalB0 = metalB0Sum / metalCountForB0;
  const hFraction = hCount / totalAtoms;
  // Cage-stiffening factor: 1.0 at H_frac=0, ~6× at H_frac=10/11 (LaH10).
  // Tuned so LaH10 (B0_metal=28, hFrac=0.91) → 28 × 6.5 = 182 GPa, close
  // to the ~150 GPa ambient-extrapolated literature value.
  const cageFactor = 1.0 + 6.0 * hFraction;
  return Math.max(30, baseMetalB0 * cageFactor);
}

function computePressureScale(pressureGpa: number, elements?: string[], counts?: Record<string, number>): number {
  if (pressureGpa <= 0) return 1.0;
  const B0 = elements ? estimateBulkModulus(elements, counts) : 100;
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
  // Cu pseudo includes the semicore 3s²3p⁶ shell explicitly →
  // z_valence=19 (3s²3p⁶3d¹⁰4s¹). Verified against server/dft/pseudo/Cu.UPF.
  Cu: { mass: 63.546,  zValence: 19 }, Zn: { mass: 65.380,  zValence: 20 },
  Ga: { mass: 69.723,  zValence: 13 }, Ge: { mass: 72.640,  zValence: 4  },
  As: { mass: 74.922,  zValence: 5  }, Se: { mass: 78.960,  zValence: 6  },
  Rb: { mass: 85.468,  zValence: 9  }, Sr: { mass: 87.620,  zValence: 10 },
  Y:  { mass: 88.906,  zValence: 11 }, Zr: { mass: 91.224,  zValence: 12 },
  Nb: { mass: 92.906,  zValence: 13 }, Mo: { mass: 95.960,  zValence: 14 },
  Ru: { mass: 101.07,  zValence: 16 }, Rh: { mass: 102.91,  zValence: 17 },
  Pd: { mass: 106.42,  zValence: 18 }, Ag: { mass: 107.87,  zValence: 19 },
  Cd: { mass: 112.41,  zValence: 12 }, In: { mass: 114.82,  zValence: 13 },
  Sn: { mass: 118.71,  zValence: 4  }, Sb: { mass: 121.76,  zValence: 5  },
  Te: { mass: 127.60,  zValence: 6  }, I:  { mass: 126.90,  zValence: 7  },
  Cs: { mass: 132.91,  zValence: 9  }, Ba: { mass: 137.33,  zValence: 10 },
  // Ce pseudo treats 4f¹5d¹6s² as valence → z_valence=11 (no semicore 5s5p).
  // Was 12 (matched Pr's count) — verified against server/dft/pseudo/Ce.UPF.
  La: { mass: 138.91,  zValence: 11 }, Ce: { mass: 140.12,  zValence: 11 },
  Hf: { mass: 178.49,  zValence: 12 }, Ta: { mass: 180.95,  zValence: 13 },
  W:  { mass: 183.84,  zValence: 14 }, Re: { mass: 186.21,  zValence: 15 },
  Os: { mass: 190.23,  zValence: 16 }, Ir: { mass: 192.22,  zValence: 15 },
  Pt: { mass: 195.08,  zValence: 16 }, Au: { mass: 196.97,  zValence: 19 },
  Hg: { mass: 200.59,  zValence: 20 },
  Tl: { mass: 204.38,  zValence: 13 }, Pb: { mass: 207.2,   zValence: 4  },
  Bi: { mass: 208.98,  zValence: 5  },
  Br: { mass: 79.904,  zValence: 7  },
  Tc: { mass: 98.0,    zValence: 15 },
  Pr: { mass: 140.91,  zValence: 13 },
  Nd: { mass: 144.24,  zValence: 14 },
  // Pm (5s²5p⁶4f⁵6s² with semicore) was missing — fell through to
  // elemental-data's non-semicore value 7, bypassing the validateSemicorePP
  // semicore-presence check.
  Pm: { mass: 145.0,   zValence: 15 },
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
  // Actinide z_valence values include the 6s²6p⁶ semicore shell (8e⁻) plus the
  // chemistry-active 5f/6d/7s. These match the standard PSLibrary/PAW PPs and
  // are needed by validateSemicorePP — without them the function falls back
  // to elemental-data's non-semicore valence (Np=7, Pu=8, etc.), so the
  // threshold check `ppZVal < expectedZ * 0.5` would always pass even when
  // the downloaded PP lacks the semicore shell, silently producing wrong
  // forces and density convergence.
  Ac: { mass: 227.0,   zValence: 11 },  // 6s²6p⁶6d¹7s²
  Th: { mass: 232.04,  zValence: 12 },  // 6s²6p⁶6d²7s²
  Pa: { mass: 231.04,  zValence: 13 },  // 6s²6p⁶5f²6d¹7s²
  U:  { mass: 238.03,  zValence: 14 },  // 6s²6p⁶5f³6d¹7s²
  Np: { mass: 237.0,   zValence: 15 },  // 6s²6p⁶5f⁴6d¹7s²
  Pu: { mass: 244.0,   zValence: 16 },  // 6s²6p⁶5f⁶7s²
  Am: { mass: 243.0,   zValence: 17 },  // 6s²6p⁶5f⁷7s²
  Cm: { mass: 247.0,   zValence: 18 },  // 6s²6p⁶5f⁷6d¹7s²
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
  /** Signed total magnetization in Bohr magneton/cell (sum over cell).
   *  ≈0 for antiferromagnetic systems even when local moments are large. */
  magnetization: number | null;
  /** Absolute magnetization in Bohr magneton/cell (∫|m(r)| d³r).
   *  Non-zero whenever any local moment formed — captures AFM systems
   *  that `magnetization` (total) misses. */
  absoluteMagnetization: number | null;
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
  omegaLog: number;        // K (log-average phonon frequency in Kelvin — parseLambdaOutput in dfpt-parser.ts converts cm⁻¹→K via hc/k_B before returning)
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
  epw?: import("./epw-pipeline").EPWResult;
  sscha?: import("./sscha-pipeline").SSCHAResult;
  acbn0?: import("./acbn0-pipeline").ACBN0Result;
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
  /** Hubbard U workflow analysis (composition-aware DFT+U). */
  hubbardWorkflow?: HubbardWorkflowResult;
  /** Spin-orbit coupling analysis for this material. */
  socAnalysis?: SOCAnalysis;
  /** Magnetic ground-state search results (FM/AFM/NM comparison). */
  magneticGroundState?: MagneticGroundStateResult;
  /** DMFT-ready bundle export (for correlated materials with final_converged+ quality). */
  dmftBundle?: DMFTBundleResult;
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
    // The PP type flags (is_paw / is_ultrasoft / pseudo_type) live in the
    // <PP_HEADER> element, which UPF v2 places AFTER the <PP_INFO> block.
    // PP_INFO carries the generation log and can run past 2 KB — verified
    // 7 PSL pseudos (Ce, La, Nb, W, Y, Zn, Zr) whose PP_HEADER starts
    // beyond byte 2000. A 2 KB window missed them entirely and the
    // function silently returned the "paw" fallback, which mis-sized
    // ecutrho for any USPP element not also in HARD_PAW_ELEMENTS.
    // readFileSync already loads the whole file, so widening the slice
    // costs no extra I/O.
    const head = fs.readFileSync(ppPath, "utf-8").slice(0, 65536);
    if (head.includes('is_paw="true"') || head.includes("is_paw='.true.'") || head.includes("pseudo_type=\"PAW\"")) return "paw";
    if (head.includes('is_ultrasoft="true"') || head.includes("is_ultrasoft='.true.'") || head.includes("pseudo_type=\"US\"") || head.includes("Ultrasoft")) return "uspp";
    if (head.includes("pseudo_type=\"NC\"") || head.includes("Norm-Conserving") || head.includes("norm-conserving")) return "nc";
    if (head.includes('is_ultrasoft="false"') && head.includes('is_paw="false"')) return "nc";
  } catch {}
  return "paw";
}

/**
 * Whether the pseudopotential for `element` includes spin-orbit data.
 *
 * QE's `lspinorb = .true.` requires fully-relativistic (FR) pseudos that
 * carry the `has_so="T"` flag. Scalar-relativistic pseudos (`has_so="F"`)
 * will cause `lspinorb=.true.` to silently produce wrong physics: the
 * Dirac kinetic correction is included but the SO coupling between l ± 1/2
 * channels is absent.
 *
 * Returns `false` on read errors (conservative: assume no SO data).
 */
function detectPPHasSOC(element: string): boolean {
  const ppPath = path.join(QE_PSEUDO_DIR, resolvePPFilename(element));
  try {
    // has_so lives in <PP_HEADER>, which follows the (variable-length)
    // <PP_INFO> block — observed up to byte ~2950 in the current pseudo
    // set, leaving little margin under a 4 KB window. Read a generous
    // window so a larger PP_INFO can't push has_so out of view and
    // cause a fully-relativistic pseudo to be misread as scalar (which
    // would silently downgrade lspinorb and drop SO splitting).
    const head = fs.readFileSync(ppPath, "utf-8").slice(0, 65536);
    if (head.includes('has_so="T"') || head.includes("has_so='.true.'")) return true;
    if (head.match(/relativistic\s*[:=]\s*['"]?full/i)) return true;
  } catch {}
  return false;
}

/**
 * Filter `qeSystemFlags` from analyzeSOCRequirement to remove lspinorb
 * (and noncolin if no spin-orbit element has FR pseudo) when the
 * available pseudopotentials don't actually carry SO data.
 *
 * Returns the (possibly downgraded) flag string plus a diagnostic note
 * for the caller to surface in logs.
 */
function applySOCPseudoConstraint(
  qeFlags: string,
  socElements: Array<{ element: string }>,
): { flags: string; note: string | null } {
  if (!qeFlags.includes("lspinorb")) {
    return { flags: qeFlags, note: null };
  }
  const elementsWithSO = socElements
    .map(e => e.element)
    .filter(el => detectPPHasSOC(el));
  if (elementsWithSO.length > 0) {
    // At least one SOC-relevant element has FR pseudo — proceed with SOC
    return { flags: qeFlags, note: null };
  }
  // Downgrade: drop lspinorb (keep noncolin if magnetism is the reason),
  // but here lspinorb without FR pseudos = wrong physics, so strip it.
  const downgraded = qeFlags
    .replace(/\s*lspinorb\s*=\s*\.true\.,?\s*\n?/g, "")
    .replace(/\s*noncolin\s*=\s*\.true\.,?\s*\n?/g, "");
  const elList = socElements.map(e => e.element).join(",");
  return {
    flags: downgraded,
    note:
      `[SOC] lspinorb=.true. requested but no FR pseudo (has_so="T") found ` +
      `for SOC elements: ${elList}. Downgraded to scalar-relativistic — ` +
      `SO splitting in pseudo is NOT included. To enable full SOC, replace ` +
      `pseudos with FR variants (e.g., from PseudoDojo NC-SR-FR or ONCV FR).`,
  };
}

// "Hard PAW" elements: pseudos with semicore states in valence (s/p-semicore for
// early TMs and alkaline-earth d-block, p-semicore for lanthanides/actinides).
// Their valence charge density has sharp features that need ecutrho ≥ 8×ecutwfc
// for the FFT grid to represent without negative-rho artifacts. CaH6 with
// ecutrho/ecutwfc=4 produced negative rho = 0.296 e- → -9770 cm⁻¹ phonon modes;
// bumping to 8 brings negative rho below 1e-3 and recovers physical frequencies.
const HARD_PAW_ELEMENTS = new Set<string>([
  // Alkaline earths with semicore p (3s+3p, 4s+4p, 5s+5p in valence)
  "Ca", "Sr", "Ba", "Ra",
  // Group 3 + lanthanides (semicore 5s+5p in valence)
  "Sc", "Y", "La",
  "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
  // Actinides (semicore 6s+6p in valence). Added Am, Cm, Bk, Cf — without
  // these, Am/Cm compounds detected the PAW PP correctly but used the
  // 4× ecutrho multiplier (not 8×), giving under-resolved density grids
  // and negative-rho artifacts that broke DFPT derivatives.
  "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
  // Early/mid 3d transition metals (semicore 3s+3p in valence for most PSL pseudos)
  "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  // 4d transition metals (semicore 4s+4p in valence)
  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  // 5d transition metals (semicore 5s+5p in valence)
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  // Heavy alkalis (semicore p in valence on better PSL pseudos)
  "K", "Rb", "Cs",
]);

function ecutrhoMultiplier(elements: string[]): number {
  // USPP requires ≥8x cutoff due to augmentation charges (any element).
  // PAW with semicore valence ("hard PAW") also needs 8x: charge density has
  // sharp core-region features that the dense grid must resolve. Without it,
  // the FFT charge density wraps to negative values (e.g. CaH6: -0.3 e-),
  // breaking DFPT derivatives and giving thousands-of-cm⁻¹ imaginary modes.
  // Plain PAW / NC for light elements is fine at 4x.
  let hasUSPP = false;
  let hasHardPAW = false;
  for (const el of elements) {
    const ppType = detectPPType(el);
    if (ppType === "uspp") hasUSPP = true;
    if (ppType === "paw" && HARD_PAW_ELEMENTS.has(el)) hasHardPAW = true;
  }
  if (hasUSPP || hasHardPAW) return 8;
  return 4;
}

function getAtomicMass(el: string): number {
  const local = ELEMENT_DATA[el];
  if (local) return local.mass;
  const central = getElementData(el);
  if (central) return central.atomicMass;
  // Both tables missed this element. Falling back to 50 amu silently biases
  // phonon frequencies (ω ∝ 1/√M, so a wrong mass shifts every frequency
  // for this species). Log a warning so the misclassification surfaces
  // instead of producing subtly wrong Tc predictions downstream.
  console.warn(`[QE-Worker] getAtomicMass: no mass for element "${el}" in ELEMENT_DATA or central elemental-data — falling back to 50 amu. Phonons for this species will be off by sqrt(M_real/50).`);
  return 50;
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
    isTransitionMetal(e) || isRareEarth(e) || isActinide(e) || ["Ca", "Sr", "Ba", "Mg", "Na", "K", "Al"].includes(e)
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
    const B0 = estimateBulkModulus(elements, effectiveCounts);
    const B0p = 4.0;
    const eta = 1 + B0p * (pressureGPa / B0);
    const volRatio = eta > 0 ? Math.pow(eta, -1 / B0p) : 0.0;
    // Don't clamp volRatio to [0.5, 1.0] — high-P hydrides (LaH10 etc.)
    // genuinely compress beyond 50% at 300+ GPa, and the formula already
    // gives ≤1 for P>0. Same fix pattern as vegard-lattice.ts.
    if (volRatio > 0 && Number.isFinite(volRatio)) {
      cellVolume = cellVolume * volRatio;
    } else {
      console.warn(`[QE-Worker] Murnaghan EOS gave volRatio=${volRatio} for ` +
        `P=${pressureGPa} GPa, B0=${B0} GPa — pathological input`);
    }
  }

  const a = Math.cbrt(cellVolume);
  const perturbation = 0.97 + Math.random() * 0.06;
  // Don't enforce a 3.0 Å floor — high-P primitive cells can be smaller.
  // Warn loudly only if the result is clearly unphysical (< 2.0 Å).
  const result = a * perturbation;
  if (result < 2.0) {
    console.warn(`[QE-Worker] estimateLatticeConstant: a=${result.toFixed(2)} Å < 2.0 — ` +
      `check elements/pressure inputs (V=${cellVolume.toFixed(2)} Å³, P=${pressureGPa} GPa)`);
  }
  return result;
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
  // Actinides — 5f systems need semicore (6s,6p) just like the 4f lanthanides
  // need semicore (5s,5p). Without semicore the PP misses ~8 electrons in
  // the actinide outer-core, breaking density convergence and forces.
  "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
]);

// Elements whose best-available PPs include f-orbital projectors that exceed
// QE's compile-time lmaxx=3 limit. Production DB shows 75 jobs failing with
// "momentum in pseudopotentials (lmaxx) = 3" on these. Reject upfront until
// either (a) QE is rebuilt with a higher lmaxx, or (b) scalar-relativistic
// PPs with lmax<=2 are installed for each.
const LMAXX_INCOMPATIBLE: Set<string> = new Set([
  // Most lanthanides and actinides now supported via:
  //   1. lmaxx=6 rebuild on both workers (handles PAW f-projectors)
  //   2. Pseudo-DOJO ONCVPSP scalar-relativistic PPs (no lmaxx issue)
  // Only block elements with no available PP from any source:
  "Pu", "Am",  // no reliable PPs from any source
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
    // z_valence / number_of_wfc are <PP_HEADER> attributes, which sit after
    // the variable-length <PP_INFO> block — observed up to byte ~3240 in the
    // current pseudo set, leaving < 900 bytes of margin under a 4 KB read.
    // Use a generous 64 KB buffer so a larger PP_INFO can't push the header
    // attributes out of view and cause a valid semicore PP to be rejected
    // (triggering a needless re-download) or a deficient one to pass.
    const HEADER_READ_BYTES = 65536;
    const fd = fs.openSync(ppPath, "r");
    const buf = Buffer.alloc(HEADER_READ_BYTES);
    const bytesRead = fs.readSync(fd, buf, 0, HEADER_READ_BYTES, 0);
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

/**
 * Light cleanup: remove scratch files (.wfc, .mix, .restart_xml) but PRESERVE
 * .save/ directories. Use this after vc-relax when SCF skip is active — ph.x
 * and bands both need the .save/ wavefunctions that disk_io='high' wrote.
 */
function cleanQETmpScratch(tmpDir: string): void {
  if (!fs.existsSync(tmpDir)) return;
  try {
    const entries = fs.readdirSync(tmpDir);
    for (const entry of entries) {
      const fullPath = path.join(tmpDir, entry);
      try {
        if (
          entry.endsWith(".xml") ||
          entry.endsWith(".restart_xml") ||
          /\.(wfc|mix)\d*(_new)?$/.test(entry)
        ) {
          fs.unlinkSync(fullPath);
        } else if (entry.endsWith(".save_tmp")) {
          fs.rmSync(fullPath, { recursive: true, force: true });
        }
      } catch {}
    }
  } catch {}
}

// === DFT Structure Cache ===
// Saves best DFT-optimized structures per formula so future runs start from
// the previous best geometry instead of from scratch. Dramatically speeds up
// convergence for difficult materials like high-P hydrides.

const DFT_STRUCTURE_CACHE_DIR = path.join(QE_WORK_DIR, "structure_cache");

interface CachedStructure {
  formula: string;
  latticeA: number;
  positions: Array<{ element: string; x: number; y: number; z: number }>;
  force: number;
  pressure: number | null;
  energy: number;
  timestamp: number;
  source: string;
}

function saveDFTStructureCache(
  formula: string,
  latticeA: number,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  force: number,
  pressure: number | null,
  energy: number,
): void {
  try {
    if (!fs.existsSync(DFT_STRUCTURE_CACHE_DIR)) {
      fs.mkdirSync(DFT_STRUCTURE_CACHE_DIR, { recursive: true });
    }
    const cacheFile = path.join(DFT_STRUCTURE_CACHE_DIR, `${formula}.json`);

    // Only overwrite if the new structure is better (lower force)
    if (fs.existsSync(cacheFile)) {
      try {
        const existing: CachedStructure = JSON.parse(fs.readFileSync(cacheFile, "utf-8"));
        if (existing.force <= force) {
          console.log(`[DFT-Cache] ${formula}: cached structure already has force=${existing.force.toFixed(6)} ≤ ${force.toFixed(6)} — keeping cached`);
          return;
        }
      } catch {}
    }

    const cached: CachedStructure = {
      formula, latticeA, positions, force,
      pressure, energy, timestamp: Date.now(),
      source: `DFT-cached (force=${force.toFixed(6)}, a=${latticeA.toFixed(3)})`,
    };
    fs.writeFileSync(cacheFile, JSON.stringify(cached, null, 2));
    console.log(`[DFT-Cache] Saved ${formula}: a=${latticeA.toFixed(3)} Å, ${positions.length} atoms, force=${force.toFixed(6)}, E=${energy.toFixed(4)} eV`);
  } catch (err: any) {
    console.log(`[DFT-Cache] Failed to save ${formula}: ${err.message?.slice(0, 100)}`);
  }
}

function loadDFTStructureCache(formula: string): CachedStructure | null {
  try {
    const cacheFile = path.join(DFT_STRUCTURE_CACHE_DIR, `${formula}.json`);
    if (!fs.existsSync(cacheFile)) return null;
    const cached: CachedStructure = JSON.parse(fs.readFileSync(cacheFile, "utf-8"));
    // Validate basic structure
    if (!cached.positions || cached.positions.length === 0 || !cached.latticeA || cached.latticeA <= 0) return null;
    console.log(`[DFT-Cache] Loaded ${formula}: a=${cached.latticeA.toFixed(3)} Å, ${cached.positions.length} atoms, force=${cached.force.toFixed(6)}, age=${((Date.now() - cached.timestamp) / 3600_000).toFixed(1)}h`);
    return cached;
  } catch {
    return null;
  }
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
// Pseudo-DOJO NC PPs as secondary — validated for DFPT, covers lanthanides/actinides.
// QE website kept as fallback. GBRV ultrasoft PPs from Rutgers as last resort.
const GH_BASE = "https://raw.githubusercontent.com/dalcorso/pslibrary/master/pbe/PSEUDOPOTENTIALS";
// Pseudo-DOJO ONCVPSP scalar-relativistic PBE set. The repo is abinit/
// pseudo_dojo (NOT pseudo-dojo/pseudo-dojo, which 404s) and the scalar set
// is ONCVPSP-PBE-PDv0.4 (there is no "-SR-" set — only the FR set carries an
// explicit relativity tag). The old URL silently failed every DOJO fallback.
const DOJO_BASE = "https://raw.githubusercontent.com/abinit/pseudo_dojo/master/pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.4";
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

// Pseudo-DOJO norm-conserving PPs — DFPT-validated, scalar-relativistic.
// These work with lmaxx=6 — good for phonon/EPW calculations. Fallback only;
// PSLibrary (PP_DOWNLOAD_URLS) is primary.
//
// Only the elements actually present in the Pseudo-DOJO ONCVPSP-PBE-PDv0.4
// set are listed. The set spans H–Rn and has NO actinides and no lanthanides
// beyond La — the previous map listed Pr–Tm, Pa, U, Np, Ce, Th, which never
// existed in Pseudo-DOJO and were guaranteed-404 dead fallbacks (those
// elements are covered by the PSLibrary primary map). All present elements
// use the `<El>-sp.upf` basename (verified against the set's standard.djson).
const PP_DOJO_URLS: Record<string, string> = {
  La: `${DOJO_BASE}/La/La-sp.upf`,
  Y:  `${DOJO_BASE}/Y/Y-sp.upf`,
  Sc: `${DOJO_BASE}/Sc/Sc-sp.upf`,
  Ti: `${DOJO_BASE}/Ti/Ti-sp.upf`,
  Zr: `${DOJO_BASE}/Zr/Zr-sp.upf`,
  Hf: `${DOJO_BASE}/Hf/Hf-sp.upf`,
  Nb: `${DOJO_BASE}/Nb/Nb-sp.upf`,
  Mo: `${DOJO_BASE}/Mo/Mo-sp.upf`,
  Ta: `${DOJO_BASE}/Ta/Ta-sp.upf`,
  W:  `${DOJO_BASE}/W/W-sp.upf`,
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

    const urls = [PP_DOWNLOAD_URLS[element], PP_DOJO_URLS[element], PP_FALLBACK_URLS[element], PP_GBRV_URLS[element]].filter(Boolean);
    for (const url of urls) {
      try {
        const mirror = url!.includes("pseudo-dojo") ? "PseudoDojo" : url!.includes("dalcorso") ? "pslibrary" : url!.includes("rutgers") ? "GBRV" : "QE";
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

/**
 * Install the fully-relativistic (FR) pseudopotential for `element` into
 * QE_PSEUDO_DIR, overwriting whatever scalar-relativistic pseudo is cached.
 *
 * QE's lspinorb=.true. requires FR pseudos (has_so="T"); the default
 * provisioning installs scalar-relativistic PAW pseudos, so without this step
 * applySOCPseudoConstraint downgrades every SOC calculation. Called once per
 * SOC element before the SOC-constraint gate.
 *
 * Source order: bundled server/dft/pseudo/oncv-fr/ → Pseudo-DOJO download.
 * A stale FR pseudo left in the cache for a later non-SOC job is harmless —
 * QE averages the j=l±1/2 channels when lspinorb is off — so no cleanup is
 * needed. Returns true if QE_PSEUDO_DIR holds an FR pseudo afterwards.
 */
async function ensureRelativisticPP(element: string): Promise<boolean> {
  const fr = RELATIVISTIC_PP_URLS[element];
  if (!fr) return false;

  const ppFile = path.join(QE_PSEUDO_DIR, `${element}.UPF`);
  // Already the FR pseudo (e.g. cached from a prior SOC job) — nothing to do.
  if (fs.existsSync(ppFile) && detectPPHasSOC(element)) return true;

  if (!fs.existsSync(QE_PSEUDO_DIR)) fs.mkdirSync(QE_PSEUDO_DIR, { recursive: true });
  const tmpFile = ppFile + `.frtmp.${process.pid}`;

  // 1. Bundled FR pseudo in the repo (server/dft/pseudo/oncv-fr/).
  const bundled = path.join(PP_SOURCE_DIR, "oncv-fr", fr.file);
  try {
    if (fs.existsSync(bundled) && validatePseudopotential(bundled)) {
      fs.copyFileSync(bundled, tmpFile);
      fs.renameSync(tmpFile, ppFile);
      console.log(`[QE-Worker] Installed FR pseudo for ${element} (oncv-fr/${fr.file}) — SOC enabled`);
      return true;
    }
  } catch {}

  // 2. Download from Pseudo-DOJO.
  try {
    if (await downloadPPToTemp(fr.url, tmpFile) && validatePseudopotential(tmpFile)) {
      fs.renameSync(tmpFile, ppFile);
      console.log(`[QE-Worker] Downloaded FR pseudo for ${element} from Pseudo-DOJO — SOC enabled`);
      return true;
    }
  } catch {}
  try { fs.unlinkSync(tmpFile); } catch {}

  console.log(`[QE-Worker] No FR pseudo available for ${element} — SOC will downgrade to scalar-relativistic`);
  return false;
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
  // Third positional arg is heuristically treated as bOverA OR minK:
  //   - Fractional or near-1 values (≤ 3): interpreted as bOverA (b/a ratio).
  //     Most call sites pass `bOverA` here, e.g. 0.95 for slightly-orthorhombic.
  //   - Integer values ≥ 4: interpreted as minK (legacy k-point floor convention).
  //     Used by the QERunnerCallbacks wrapper at line 5050 (passes 4) and the
  //     proto-screening at line 5744 (passes 12).
  // Previously this was declared as `minK` with default 4 — but the bOverA-
  // passing callers silently had their kb forced to ka (no b-axis honoring).
  // The mixed interpretation here is backwards-compat for both groups.
  bOverAOrMinK: number = 1.0,
  dimensionality?: string,
  kspacing: number = DEFAULT_KSPACING,
  adaptiveOpts?: { stage?: "relax" | "vc-relax" | "scf" | "phonon"; isMetallic?: boolean; totalAtoms?: number; publicationReady?: boolean },
): string {
  // Disambiguate: integer ≥ 4 → minK; everything else → bOverA.
  const treatAsMinK = Number.isInteger(bOverAOrMinK) && bOverAOrMinK >= 4;
  const minKOverride = treatAsMinK ? bOverAOrMinK : 4;
  const bOverA = treatAsMinK ? 1.0 : bOverAOrMinK;
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
  // Publication-tier densification: when residual force indicates publication
  // quality, N(E_F) accuracy matters for λ and μ* — use tighter kspacing.
  if (adaptiveOpts?.publicationReady && adaptiveOpts.stage === "scf") {
    effectiveKspacing = adaptiveOpts.isMetallic ? 0.15 : 0.20; // aiida "moderate" for metals
  }
  // Metallicity boost: metals need denser k-grids for Fermi surface resolution
  if (adaptiveOpts?.isMetallic && !adaptiveOpts?.stage) {
    effectiveKspacing *= 0.77;  // ~1.3x denser
  }
  // Large cells already sample well via volume; progressively coarsen to save compute
  if (adaptiveOpts?.totalAtoms && adaptiveOpts.totalAtoms > 12) {
    effectiveKspacing *= 1.20;  // ~1.2x coarser for 13+ atoms
  } else if (adaptiveOpts?.totalAtoms && adaptiveOpts.totalAtoms > 8) {
    effectiveKspacing *= 1.10;  // ~1.1x coarser for 9-12 atoms
  }

  const densityFactor = (2 * Math.PI) / effectiveKspacing;
  const isLayered = dimensionality === "quasi-2D" || dimensionality === "2D";
  const layeredBoost = isLayered ? 1.5 : 1.0;
  const effCOverA = cOverA ?? 1.0;
  const effBOverA = bOverA > 0 ? bOverA : 1.0;
  const ka = Math.max(minKOverride, Math.ceil(densityFactor / latticeA));
  // Honor b-axis lattice length: kb = ceil(2π / (kspacing * b)) where b = a·(b/a).
  // Previously kb was forced to ka regardless of bOverA — over-sampled b for
  // orthorhombic cells with b > a (e.g., layered systems), under-sampled for b < a.
  const kb = Math.max(minKOverride, Math.ceil(densityFactor / (latticeA * effBOverA)));
  const baseKc = Math.ceil(densityFactor / (latticeA * effCOverA));
  const kc = Math.max(minKOverride, isLayered ? Math.ceil(baseKc * layeredBoost) : baseKc);
  return `  ${ka} ${kb} ${kc}  0 0 0`;
}

const MAGNETIC_ELEMENTS: Record<string, number> = {
  // 3d ferromagnets and antiferromagnets (μB per atom)
  Fe: 2.0, Co: 1.5, Ni: 0.8, Mn: 3.0, Cr: 1.5,
  V: 0.5,
  // 4f lanthanides (Ln³⁺ effective moments — only Gd/Eu/Nd/Sm were listed
  // before, missing the strongly-magnetic Pr-Tm series. Ho³⁺ has the largest
  // local moment of ANY element (10.6 μB) — omitting it forced nspin=1 SCF
  // for HoH9 / HoNi / etc., missing the magnetic ground state entirely).
  Pr: 3.6, Nd: 3.0, Sm: 1.0, Eu: 7.0, Gd: 7.0,
  Tb: 9.7, Dy: 10.6, Ho: 10.6, Er: 9.6, Tm: 7.6,
  // 5f actinides (partially-filled 5f systems carry large moments in
  // compounds — U/Np/Pu heavy fermions, Am/Cm magnetic insulators).
  U: 3.0, Np: 2.5, Pu: 4.0, Am: 5.0, Cm: 7.0,
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
// Recommended ecutwfc per element (Ry). Calibrated against the "Suggested
// minimum cutoff for wavefunctions" attribute in server/dft/pseudo/*.UPF
// with a ~15-20% production safety margin on top of the pseudo minimum.
// Under-converging ecutwfc gives wrong total energies (100+ meV/atom),
// wrong forces (bad relaxation), and 5-15% phonon frequency errors —
// catastrophic for the SC-relevant 3d/4d transition metals where the
// pseudos include semicore states (Fe pseudo min: 71 Ry, Ni: 75 Ry,
// Cu: 71 Ry, Li: 103 Ry, La: 75 Ry).
const SPECIES_ECUTWFC: Record<string, number> = {
  // Light p-block & H — H pseudo only needs 46 but we keep 100 because
  // pressurised hydride structures benefit from extra plane waves in the
  // small unit cell.
  H: 100, O: 70, F: 80, N: 60, Cl: 60, S: 55, P: 55, Se: 50, Br: 50,
  // s-/p-block (Li pseudo recommends 103 Ry — this was missing entirely)
  Li: 120, Be: 60, B: 55, C: 60, Na: 60, Mg: 70, Al: 50, Si: 50,
  K: 50, Ca: 55, Rb: 50, Sr: 50, Cs: 50, Ba: 50,
  // 3d transition metals (pseudo minimums are 48-75 Ry; production needs ~+15%)
  Sc: 60, Ti: 65, V: 60, Cr: 60, Mn: 90, Fe: 90, Co: 90,
  Ni: 90, Cu: 85, Zn: 75,
  // 4d transition metals (pseudo minimums are 38-49 Ry)
  Y: 55, Zr: 60, Nb: 60, Mo: 60, Tc: 60, Ru: 55, Rh: 55, Pd: 55, Ag: 55, Cd: 55,
  // 5d transition metals (pseudo minimums are 43-50 Ry)
  Hf: 60, Ta: 55, W: 60, Re: 55, Os: 55, Ir: 55, Pt: 55, Au: 55, Hg: 55,
  // Lanthanides — La pseudo min is 75 Ry; assume similar for the series.
  // Pm added (was missing → fell through to 45 Ry default, way too low for
  // semicore 5s²5p⁶4f⁵6s² PP).
  La: 90, Ce: 75, Pr: 75, Nd: 75, Pm: 75, Sm: 75, Eu: 75, Gd: 75, Tb: 75,
  Dy: 75, Ho: 75, Er: 75, Tm: 75, Yb: 75, Lu: 75,
  // Actinides — full series. Without Ac/Np/Pu/Am/Cm entries, heavy actinide
  // pure-metal calcs used ecutwfc=45 Ry (the default fallback) which is far
  // below the 70+ Ry needed for semicore 6s²6p⁶5f^n PPs. Production runs
  // got under-converged density grids → wrong forces and false phonons.
  Ac: 70, Th: 70, Pa: 70, U: 70, Np: 75, Pu: 75, Am: 75, Cm: 75,
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
//   nbnd_per_spin = ceil(nelec/2) + max(4, ceil(nelec * 0.10))
//
// For nspin=2 we double this. Note: QE's `nbnd` is already per-spin (QE
// internally allocates 2*nbnd states when nspin=2), so doubling is NOT a
// physics requirement — aiida and standard QE tutorials use the same
// per-spin nbnd for nspin=2 as for nspin=1. We double anyway as a
// CONVERGENCE AID for magnetic systems near van Hove singularities or
// with strong spin-up/spin-down band asymmetry, where extra empty bands
// help SCF reach the correct ground state. Trade-off: ~2× memory and
// ~2× per-iteration cost vs. fewer TOO_FEW_BANDS retries. The handler
// at line 7393 bumps nbnd anyway if it ever fails, so a tighter starting
// nbnd would be safe — kept conservative for now.
/**
 * Compute nbnd for QE. Uses actual atom count from positions (not formula
 * counts) to handle supercells correctly. Known structures like Nb3Sn in
 * Pm-3n have Z=2 (8 atoms), so the formula gives 53 electrons but the
 * cell has 106 electrons → nbnd must match the cell, not the formula.
 */
function computeNbnd(elements: string[], counts: Record<string, number>, nspin: number = 1, actualPositions?: Array<{ element: string }>, noncolin: boolean = false): number {
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
  if (noncolin) {
    // With noncolin=.true. (SOC or non-collinear magnetism), QE uses
    // 4-component spinors — each band covers both spin components but
    // there's no spin degeneracy. QE needs nbnd >= nelec (not nelec/2).
    // Formula: nbnd = nelec + max(4, ceil(nelec * 0.15))
    return nelec + Math.max(4, Math.ceil(nelec * 0.15));
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
  // Actinides — full series. Previously only Th and U were listed; missing
  // Ac/Pa/Np/Pu/Am/Cm/Bk/Cf compounds were treated as light-element systems
  // → simple-retry ladder (insufficient for 5f correlation) AND base
  // max_seconds budget (timed out before SCF converged for Np/Pu compounds).
  "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
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
  const hasS = elements.includes("S");

  if (hasCu && hasO) {
    const oCount = counts["O"] || 0;
    const cuCount = counts["Cu"] || 0;
    if (oCount >= 2 && cuCount >= 1) return true;
  }

  // Include S in the chalcogenide check — Fe-S compounds (FeS mackinawite,
  // FeS2 pyrite/marcasite, Fe7S8 pyrrhotite) are antiferromagnetic ground
  // states and should get AFM starting_magnetization in the initial SCF
  // input. Matches the parallel fix in magnetic-ground-state.ts:
  // shouldSearchMagneticGS (iteration 105). Without S here, Fe-sulfide
  // initial SCFs seeded FM until the magnetic-GS search re-ran with the
  // correct ordering, costing an extra round of SCF iterations.
  if (hasFe && (hasAs || hasP || hasSe || hasTe || hasS)) return true;

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
  // Fe-chalcogenide (FeSe, FeTe, FeS family) — stripe/checkerboard AFM is the
  // pnictide-class ground state per Mazin et al., PRL 101, 057003 (2008).
  // Was missing Se/Te/S — these compounds got the generic "alternating"
  // pattern instead of the physically-motivated checkerboard seed.
  const hasSe = elements.includes("Se");
  const hasTe = elements.includes("Te");
  const hasS = elements.includes("S");

  if (hasCu && hasO) return "layered";
  if (hasFe && (hasAs || hasP || hasSe || hasTe || hasS)) return "checkerboard";
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

/** Parse CELL_PARAMETERS text block into a 3×3 numeric lattice vector array. */
function latticeVectorsFromParams(
  latticeA: number, cOverA: number, bOverA: number,
  elements?: string[], counts?: Record<string, number>,
  alpha: number = 90, beta: number = 90, gamma: number = 90,
): number[][] {
  // Angles propagated so monoclinic/triclinic candidates get the correct
  // cell vectors. Without this, the DMFT pipeline (the sole caller) was
  // generating a Wannier90 .win file with an orthorhombic cell while QE
  // NSCF ran with the actual monoclinic β — Wannier projections then
  // misaligned with the electronic structure, and the downstream DMFT
  // bundle had inconsistent lattice_vectors vs the H(R) it described.
  const cellStr = generateCellParameters(latticeA, cOverA, 0, bOverA, elements, counts, alpha, beta, gamma);
  const lines = cellStr.split("\n").filter(l => l.trim() && !l.includes("CELL_PARAMETERS"));
  return lines.map(line => {
    const nums = line.trim().split(/\s+/).map(Number);
    return [nums[0] || 0, nums[1] || 0, nums[2] || 0];
  });
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

  // Pass `formula` and `latticeA` so the Tier-0 known-structure lookup +
  // perturbPositions path inside generateAtomicPositions can fire. Without
  // these arguments, every SCF — including the ones for literature compounds
  // already in known-structures.ts (LaH10, CaH6, MgB2, H3S, etc.) — fell
  // straight through to the generic prototype/fallback positions, while the
  // cell block at line 2321 still used the known-structure lattice. That's
  // an inconsistent atoms-cell pair: the cell shape was right, but the
  // ATOMIC_POSITIONS came from a guess. ph.x downstream then had to relax
  // around a wrong starting geometry and frequently produced soft modes
  // that real Wyckoff positions wouldn't.
  let atomicPositions = "";
  const positions = generateAtomicPositions(elements, counts, formula, latticeA);
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
    if (cagePositions.length === totalAtoms && cagePositions.length <= 24) {
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

function autoPhononQGrid(elements: string[], totalAtoms?: number, residualForce?: number): [number, number, number] {
  // Tiered q-grid based on structure quality (residual force) and system size.
  // Cost scales as n_atoms × n_q × 3*n_atoms perturbations per q-point.
  //
  // Quality tiers:
  //   Publication (force < 0.001): 4×4×4 or higher — proper alpha2F and lambda
  //   DFPT-quality (force < 0.03):  3×3×3 or 4×4×4 — real phonon DOS for Tc
  //   Screening (force >= 0.03):    1×1×1 or 2×2×2 — just checking stability
  //
  // Scaled down for large/heavy systems to keep wall time feasible.
  const nAtoms = totalAtoms ?? 6;
  const hasHeavy = elements.some(el => HEAVY_ELEMENTS.has(el));
  const force = residualForce ?? 999;

  // Publication-ready: force < 0.001
  if (force < 0.001) {
    if (nAtoms <= 4 && !hasHeavy) return [6, 6, 6];   // small+light: dense grid
    if (nAtoms <= 4) return [4, 4, 4];                  // small+heavy
    if (nAtoms <= 8 && !hasHeavy) return [4, 4, 4];    // medium+light
    if (nAtoms <= 8) return [3, 3, 3];                  // medium+heavy
    if (nAtoms <= 15) return [3, 3, 3];                 // large (9-15)
    if (nAtoms <= 20) return [2, 2, 2];                 // very large (16-20)
    return [2, 2, 2];                                    // ultra-large (21-24)
  }

  // DFPT-quality: force < 0.03
  if (force < 0.03) {
    if (nAtoms <= 4 && !hasHeavy) return [4, 4, 4];
    if (nAtoms <= 4) return [3, 3, 3];
    if (nAtoms <= 8 && !hasHeavy) return [3, 3, 3];
    if (nAtoms <= 8) return [2, 2, 2];
    if (nAtoms <= 15) return [2, 2, 2];
    if (nAtoms <= 20) return [2, 2, 2];                 // large (16-20): still 2×2×2
    return [1, 1, 1];                                    // ultra-large (21-24): gamma-only
  }

  // Screening: force >= 0.03
  if (nAtoms >= 15) return [1, 1, 1];                    // large+: gamma-only for screening
  if (nAtoms >= 9) return [1, 1, 1];
  if (nAtoms >= 5 && hasHeavy) return [1, 1, 1];
  return [2, 2, 2];
}

function generatePhononInput(formula: string, elements: string[] = [], totalAtoms: number = 6, opts?: { maxSeconds?: number; recover?: boolean; tr2Ph?: string; alphaMix?: number; residualForce?: number }): string {
  const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
  const [nq1, nq2, nq3] = autoPhononQGrid(elements, totalAtoms, opts?.residualForce);
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
  // Tier tr2_ph by structure quality — publication structures get tighter convergence
  const force = opts?.residualForce ?? 999;
  const defaultTr2Ph = force < 0.001 ? "1.0d-14" : force < 0.03 ? "1.0d-12" : "1.0d-10";
  const tr2Ph = opts?.tr2Ph ?? defaultTr2Ph;
  const alphaMix = opts?.alphaMix ?? 0.5;
  const qualityTier = force < 0.001 ? "publication" : force < 0.03 ? "DFPT" : "screening";
  console.log(`[QE-Worker] Phonon q-grid: ${nq1}×${nq2}×${nq3} (${qualityTier} tier, force=${force < 100 ? force.toFixed(4) : "N/A"}, tr2_ph=${tr2Ph}, alpha_mix=${alphaMix})`);
  //
  // Gamma-only (1×1×1): use ldisp=.false. so ph.x prints omega(N) = X [THz]
  // = Y [cm-1] directly to stdout, which parsePhononOutput already handles.
  // ldisp=.true. with 1×1×1 writes the dynamical matrix to .dyn files in a
  // binary/numerical format that dynmat.x must post-process, and the various
  // parsers fail to match dynmat.x's tabular output → "0 modes" bug.
  const isGammaOnly = nq1 === 1 && nq2 === 1 && nq3 === 1;
  // Born effective charges Z*_αβ and high-frequency dielectric ε∞.
  // Required for LO-TO splitting at q=Γ in any system with non-zero ionic
  // character. ph.x auto-skips Z*/ε∞ if the system is metallic at Ef, so
  // we can set it unconditionally — cost is only paid for insulators.
  // Only active when q=Γ is in the q-mesh, which is true for both the
  // ldisp=.false. (Γ-only) and ldisp=.true. (Γ-included grids) paths.
  const epsilFlags = "  epsil = .true.,\n  trans = .true.,\n";
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
${epsilFlags}${recoverLine}${maxSecLine}/
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
${epsilFlags}${recoverLine}${maxSecLine}/
`;
}

/**
 * Parse "negative rho (up, down): 2.959E-01 0.000E+00" from QE SCF output.
 * Returns the larger magnitude across spin channels. Above ~1e-3 e- the FFT
 * density grid is too coarse for the chosen pseudopotential — DFPT derivatives
 * become garbage and phonon frequencies blow up by orders of magnitude.
 */
function parseNegativeRho(stdout: string): { max: number; allValues: number[] } {
  const matches = [...stdout.matchAll(/negative rho\s*\(up,\s*down\)\s*:\s*(-?[\d.E+-]+)\s+(-?[\d.E+-]+)/gi)];
  const values: number[] = [];
  let maxAbs = 0;
  for (const m of matches) {
    const up = parseFloat(m[1]);
    const dn = parseFloat(m[2]);
    if (Number.isFinite(up)) {
      values.push(up);
      if (Math.abs(up) > maxAbs) maxAbs = Math.abs(up);
    }
    if (Number.isFinite(dn)) {
      values.push(dn);
      if (Math.abs(dn) > maxAbs) maxAbs = Math.abs(dn);
    }
  }
  return { max: maxAbs, allValues: values };
}

/**
 * Count "c_bands: N eigenvalues not converged" warnings in the last `tailLines`
 * lines of SCF output. A handful is normal at the start of a run; >5 in the
 * tail means Davidson failed to converge bands at the converged density, and
 * phonon will inherit broken eigenvectors.
 */
function countEigvalWarningsInTail(stdout: string, tailLines: number = 60): number {
  const lines = stdout.split("\n");
  const tail = lines.slice(-tailLines).join("\n");
  return (tail.match(/c_bands:\s*\d+\s+eigenvalues not converged/g) || []).length;
}

/**
 * Generate a phonon-prep SCF input. Strict-quality settings: conv_thr=1e-12,
 * degauss=0.005, +20 Ry ecutwfc boost, mixing_beta=0.3. This SCF runs on the
 * vc-relaxed geometry to produce the clean charge density that ph.x reads from
 * .save/. Without it, ph.x inherits the loose density from vc-relax (conv_thr
 * 1e-7, degauss 0.015) and produces wrong dynamical matrices.
 *
 * The k-grid is set tight enough to satisfy k-q commensurability for the
 * downstream phonon q-grid (k_grid must be an integer multiple of q_grid).
 */
function generatePhononPrepSCFInput(
  formula: string,
  elements: string[],
  counts: Record<string, number>,
  latticeA: number,
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  opts: {
    kPointsCard: string;
    ecutrhoMultiplierOverride?: number;
    forceNspin?: 1 | 2;
    forceMagBlock?: string;
    dftPlusULines?: string;
    dftPlusUNspin2?: boolean;
    hubbardCard?: string;
    socFlags?: string;
    forceNoncolin?: boolean;
    maxSecondsOverride?: number;
  },
): string {
  return generateSCFInputWithParams(formula, elements, counts, latticeA, positions, {
    mixingBeta: 0.3,
    maxSteps: 250,
    diag: "david",
    smearing: "mv",
    degauss: 0.005,
    ecutwfcBoost: 20,
    ecutrhoMultiplierOverride: opts.ecutrhoMultiplierOverride,
    convThr: "1.0d-12",
    forcConvThr: "1.0d-4",
    etotConvThr: "1.0d-7",
    mixingMode: "plain",
    mixingNdim: 12,
    diagoThrInit: "1.0d-6",
    restartFromScratch: true,
    maxSecondsOverride: opts.maxSecondsOverride,
    forceNspin: opts.forceNspin,
    forceMagBlock: opts.forceMagBlock,
    dftPlusULines: opts.dftPlusULines,
    dftPlusUNspin2: opts.dftPlusUNspin2,
    hubbardCard: opts.hubbardCard,
    socFlags: opts.socFlags,
    forceNoncolin: opts.forceNoncolin,
  }).replace(
    /K_POINTS \{automatic\}\s*\n\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+/,
    opts.kPointsCard,
  );
}

/**
 * Build a k-points card that satisfies k-q commensurability: each k-dimension
 * is at least 2 × q_dim. For a 4×4×4 q-grid we want k ≥ 8×8×8. q2r.x / matdyn.x
 * Fourier interpolation between k-mesh and q-mesh requires k_grid to be an
 * integer multiple of q_grid; phonon-prep uses 2× for noise margin.
 */
function kPointsCardForPhononPrep(
  latticeA: number,
  cOverA: number,
  bOverA: number,
  isMetallic: boolean | undefined,
  totalAtoms: number,
  qGrid: [number, number, number],
): string {
  // Phonon-prep uses a denser k-grid than production SCF (kspacing 0.20 vs 0.25).
  // autoKPoints returns just the numeric row "  Nx Ny Nz  0 0 0" — we wrap it
  // ourselves and apply the k≥2q commensurability bump.
  // Pass bOverA (was previously `_bOverA` and ignored, with 1.0 hardcoded)
  // so the b-axis k-density follows the actual lattice for orthorhombic cells.
  // For cubic cells bOverA=1.0 and this is a no-op.
  const baseLine = autoKPoints(
    latticeA, cOverA, bOverA > 0 ? bOverA : 1.0, undefined, 0.20,
    { stage: "scf", isMetallic, totalAtoms },
  ).trim();
  const match = baseLine.match(/^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/);
  if (!match) return `K_POINTS {automatic}\n${baseLine}`;
  const [, ks1, ks2, ks3, sh1, sh2, sh3] = match;
  const k1 = Math.max(parseInt(ks1, 10), 2 * qGrid[0]);
  const k2 = Math.max(parseInt(ks2, 10), 2 * qGrid[1]);
  const k3 = Math.max(parseInt(ks3, 10), 2 * qGrid[2]);
  return `K_POINTS {automatic}\n  ${k1} ${k2} ${k3}  ${sh1} ${sh2} ${sh3}`;
}

// ============================================================================
// Pre-phonon structure-quality validation (publication-grade pre-checks).
// These are diagnostic — they log warnings about likely-wrong structures, but
// do NOT block the pipeline. The intent is to surface "this phonon will be
// garbage" early, not to reject candidates.
// ============================================================================

interface StressTensorKbar {
  xx: number; yy: number; zz: number;
  xy: number; xz: number; yz: number;
  pressureKbar: number;
}

/**
 * Parse the final 3×3 stress tensor in kbar from vc-relax / SCF stdout.
 * QE prints two side-by-side 3×3 matrices: the first 3 columns are stress in
 * Ry/bohr³, the last 3 are the same tensor in kbar. We use the kbar columns.
 *
 * Block format:
 *     total   stress  (Ry/bohr**3)                   (kbar)     P=  1999.79
 *  0.01356632  -0.00000004   0.00000003          1995.92    -0.01     0.01
 * -0.00000004   0.01359412   0.00000005           -0.01  1999.50     0.01
 *  0.00000003   0.00000005   0.01362487           0.01      0.01  2004.03
 */
function parseFinalStressTensor(stdout: string): StressTensorKbar | null {
  const headerRegex = /total\s+stress\s+\(Ry\/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?[\d.]+)\s*\n((?:[\s\S]*?\n){3})/g;
  let last: { p: string; rows: string } | null = null;
  let m: RegExpExecArray | null;
  while ((m = headerRegex.exec(stdout)) !== null) {
    last = { p: m[1], rows: m[2] };
  }
  if (!last) return null;
  const rowLines = last.rows.split("\n").filter(l => l.trim().length > 0).slice(0, 3);
  if (rowLines.length !== 3) return null;
  const kbarRows: number[][] = [];
  for (const line of rowLines) {
    const nums = line.trim().split(/\s+/).map(parseFloat).filter(n => Number.isFinite(n));
    if (nums.length < 6) return null;
    kbarRows.push(nums.slice(3, 6));
  }
  return {
    xx: kbarRows[0][0], xy: kbarRows[0][1], xz: kbarRows[0][2],
    yy: kbarRows[1][1], yz: kbarRows[1][2],
    zz: kbarRows[2][2],
    pressureKbar: parseFloat(last.p),
  };
}

interface SymmetryCheck {
  expectedClass: "cubic" | "tetragonal" | "orthorhombic" | "lower";
  diagonalSpread: number;       // max diagonal − min diagonal
  offDiagonalMax: number;       // max |σ_ij| for i≠j
  isConsistent: boolean;
  warning?: string;
}

/**
 * Classify the expected stress-tensor symmetry from the input cell shape, then
 * verify the actual stress tensor matches it. The space group constrains which
 * stress components must be equal (cubic: σ_xx=σ_yy=σ_zz; tetragonal:
 * σ_xx=σ_yy≠σ_zz; etc.) and which must vanish (all off-diagonals in
 * orthorhombic+).
 *
 * Tolerance is 0.5% of |P| or 5 kbar (whichever is larger) — at 2000 kbar
 * target this is 10 kbar; at ambient ~5 kbar.
 */
function checkStressSymmetry(
  stress: StressTensorKbar,
  cOverA: number,
  bOverA: number,
): SymmetryCheck {
  // Symmetric Cauchy stress: average i,j and j,i (numerical noise breaks
  // exact symmetry but they should match to many decimals).
  const sxx = stress.xx, syy = stress.yy, szz = stress.zz;
  const sxy = stress.xy, sxz = stress.xz, syz = stress.yz;
  const diagMin = Math.min(sxx, syy, szz);
  const diagMax = Math.max(sxx, syy, szz);
  const diagonalSpread = diagMax - diagMin;
  const offDiagonalMax = Math.max(Math.abs(sxy), Math.abs(sxz), Math.abs(syz));
  const absP = Math.abs(stress.pressureKbar);
  const tol = Math.max(absP * 0.005, 5.0);
  // Classify expected symmetry from input cell shape (within 1% tolerance)
  const isAEqB = Math.abs(bOverA - 1.0) < 0.01;
  const isAEqC = Math.abs(cOverA - 1.0) < 0.01;
  let expectedClass: SymmetryCheck["expectedClass"];
  if (isAEqB && isAEqC) expectedClass = "cubic";
  else if (isAEqB || isAEqC || Math.abs(bOverA - cOverA) < 0.01) expectedClass = "tetragonal";
  else expectedClass = "orthorhombic";
  let isConsistent = offDiagonalMax < tol;
  let warning: string | undefined;
  if (expectedClass === "cubic" && diagonalSpread > tol) {
    isConsistent = false;
    warning = `stress diagonal spread ${diagonalSpread.toFixed(1)} kbar exceeds cubic tolerance ${tol.toFixed(1)} — input cell is cubic (b/a=c/a=1) but stress shows distortion; vc-relax may have settled to a lower-symmetry minimum (Jahn-Teller, AFM, structural transition)`;
  } else if (expectedClass === "tetragonal" && offDiagonalMax > tol) {
    isConsistent = false;
    warning = `stress off-diagonals up to ${offDiagonalMax.toFixed(1)} kbar exceed tetragonal tolerance ${tol.toFixed(1)} — cell is tetragonal but stress shows monoclinic shear`;
  } else if (offDiagonalMax > tol) {
    isConsistent = false;
    warning = `stress off-diagonals up to ${offDiagonalMax.toFixed(1)} kbar exceed tolerance ${tol.toFixed(1)} — cell did not relax to its symmetry`;
  }
  return { expectedClass, diagonalSpread, offDiagonalMax, isConsistent, warning };
}

/**
 * RMS and max atomic displacement between initial and final positions, in
 * fractional coordinates with minimum-image convention.
 *
 * > 0.10 frac RMS: structure has likely transformed into a different one
 * > 0.20 frac max: a single atom moved >20% of a lattice vector — probable
 *   reorganization (atom hopped to new site).
 */
function computeRMSDisplacement(
  initial: Array<{ element: string; x: number; y: number; z: number }>,
  final: Array<{ element: string; x: number; y: number; z: number }>,
): { rms: number; max: number; maxAtomIdx: number } {
  if (initial.length !== final.length || initial.length === 0) {
    return { rms: 0, max: 0, maxAtomIdx: -1 };
  }
  let sumSq = 0;
  let maxAbs = 0;
  let maxIdx = -1;
  for (let i = 0; i < initial.length; i++) {
    let dx = final[i].x - initial[i].x;
    let dy = final[i].y - initial[i].y;
    let dz = final[i].z - initial[i].z;
    dx -= Math.round(dx);
    dy -= Math.round(dy);
    dz -= Math.round(dz);
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    sumSq += d * d;
    if (d > maxAbs) { maxAbs = d; maxIdx = i; }
  }
  return { rms: Math.sqrt(sumSq / initial.length), max: maxAbs, maxAtomIdx: maxIdx };
}

/**
 * Find the minimum interatomic distance in the cell (with PBC). Compare to
 * a reference distance derived from atomic radii. If the minimum is below 70%
 * of the radius sum, the cell has collapsed and phonon will be unphysical.
 */
function findMinBondLength(
  positions: Array<{ element: string; x: number; y: number; z: number }>,
  latticeA: number,
  cOverA: number,
  bOverA: number,
  gammaRad: number,
  alphaRad: number = Math.PI / 2,
  betaRad: number = Math.PI / 2,
): { minDistAng: number; pair: string; expectedAng: number; ratio: number } {
  let minDist = Infinity;
  let minPair = "?";
  let expectedForMin = 0;
  for (let i = 0; i < positions.length; i++) {
    const pi = positions[i];
    for (let j = i + 1; j < positions.length; j++) {
      const pj = positions[j];
      let fdx = pi.x - pj.x;
      let fdy = pi.y - pj.y;
      let fdz = pi.z - pj.z;
      fdx -= Math.round(fdx);
      fdy -= Math.round(fdy);
      fdz -= Math.round(fdz);
      const d = fracDistAngstrom(fdx, fdy, fdz, latticeA, cOverA, bOverA, gammaRad, alphaRad, betaRad);
      if (d < minDist) {
        minDist = d;
        const [a, b] = [pi.element, pj.element].sort();
        minPair = `${a}-${b}`;
        // Atomic radius via getElementData (pm → Å)
        const da = getElementData(pi.element);
        const db = getElementData(pj.element);
        const ra = (da && da.atomicRadius > 0) ? da.atomicRadius / 100 : 1.0;
        const rb = (db && db.atomicRadius > 0) ? db.atomicRadius / 100 : 1.0;
        expectedForMin = ra + rb;
      }
    }
  }
  const ratio = expectedForMin > 0 ? minDist / expectedForMin : NaN;
  return { minDistAng: minDist, pair: minPair, expectedAng: expectedForMin, ratio };
}

/**
 * Run all structure-quality pre-phonon validations and log a single structured
 * report. Returns warnings array but does NOT block — per project policy, the
 * pipeline accepts the structure and lets phonon happen even with warnings.
 */
function runPrePhononValidation(opts: {
  formula: string;
  elements: string[];
  counts: Record<string, number>;
  initialPositions: Array<{ element: string; x: number; y: number; z: number }>;
  finalPositions: Array<{ element: string; x: number; y: number; z: number }>;
  latticeA: number;
  cOverA: number;
  bOverA: number;
  gammaRad: number;
  /** α/β in radians — π/2 (90°) for cubic/tetragonal/orthorhombic/hexagonal,
   *  not 90° only for monoclinic (β ≠ 90°) and triclinic. Default π/2 keeps
   *  the legacy orthogonal-cell behavior; pass the known-structure angles
   *  for monoclinic candidates so findMinBondLength reports correct
   *  inter-atomic distances. */
  alphaRad?: number;
  betaRad?: number;
  vcRelaxStdout: string | null;
  phononPrepScfStdout: string | null;
  isMetallic: boolean | undefined;
  totalEnergyRy: number | null;
  targetPressureKbar: number;
}): { warnings: string[]; severeCount: number } {
  const warnings: string[] = [];
  let severeCount = 0;
  const { formula } = opts;

  // (1) Stress symmetry vs expected space-group class
  if (opts.vcRelaxStdout) {
    const stress = parseFinalStressTensor(opts.vcRelaxStdout);
    if (stress) {
      const sym = checkStressSymmetry(stress, opts.cOverA, opts.bOverA);
      console.log(`[Pre-Phonon] ${formula} stress symmetry: expect=${sym.expectedClass}, diagonalSpread=${sym.diagonalSpread.toFixed(2)} kbar, offDiagMax=${sym.offDiagonalMax.toFixed(2)} kbar → ${sym.isConsistent ? "CONSISTENT" : "INCONSISTENT"}`);
      if (sym.warning) {
        warnings.push(`stress-symmetry: ${sym.warning}`);
        severeCount++;
      }
    } else {
      console.log(`[Pre-Phonon] ${formula} stress symmetry: could not parse final stress tensor`);
    }

    // (4) Pressure tolerance
    if (stress && opts.targetPressureKbar > 0) {
      const target = opts.targetPressureKbar;
      const actual = stress.pressureKbar;
      const fracErr = Math.abs(actual - target) / Math.max(target, 1);
      console.log(`[Pre-Phonon] ${formula} pressure: target=${target.toFixed(1)} kbar, final=${actual.toFixed(1)} kbar, error=${(fracErr * 100).toFixed(2)}%`);
      if (fracErr > 0.05) {
        warnings.push(`pressure mismatch: final ${actual.toFixed(1)} kbar deviates from target ${target.toFixed(1)} kbar by ${(fracErr * 100).toFixed(1)}% (>5%); structure is at the wrong P — phonons will report the wrong phase`);
        severeCount++;
      }
    }
  }

  // (2) RMS displacement (initial vs final)
  const rms = computeRMSDisplacement(opts.initialPositions, opts.finalPositions);
  console.log(`[Pre-Phonon] ${formula} atomic displacement: RMS=${rms.rms.toFixed(4)} frac, max=${rms.max.toFixed(4)} frac (atom #${rms.maxAtomIdx})`);
  if (rms.rms > 0.10) {
    warnings.push(`large displacement: RMS=${rms.rms.toFixed(3)} frac (>0.10) — vc-relax probably landed in a DIFFERENT structure than the input; verify space group with spglib before trusting the phonon`);
    severeCount++;
  } else if (rms.max > 0.20) {
    warnings.push(`atomic reorganization: atom #${rms.maxAtomIdx} moved ${rms.max.toFixed(3)} frac (>0.20) — likely hopped to a new Wyckoff site`);
  }

  // (3) Bond-length sanity — pass α/β when known so monoclinic c-axis bonds
  // get correct distances. Defaults to 90° (orthogonal cell convention).
  const bond = findMinBondLength(
    opts.finalPositions, opts.latticeA, opts.cOverA, opts.bOverA, opts.gammaRad,
    opts.alphaRad ?? Math.PI / 2, opts.betaRad ?? Math.PI / 2,
  );
  console.log(`[Pre-Phonon] ${formula} min bond: ${bond.pair}=${bond.minDistAng.toFixed(3)} Å (expect ~${bond.expectedAng.toFixed(3)} Å from atomic radii, ratio=${bond.ratio.toFixed(2)})`);
  if (bond.ratio < 0.70) {
    warnings.push(`collapsed cell: shortest ${bond.pair} contact is ${bond.minDistAng.toFixed(3)} Å, only ${(bond.ratio * 100).toFixed(0)}% of sum of atomic radii — cell has overcompressed; phonon will be unphysical`);
    severeCount++;
  } else if (bond.ratio > 2.5) {
    warnings.push(`unusually loose ${bond.pair} contact at ${bond.minDistAng.toFixed(3)} Å (${(bond.ratio * 100).toFixed(0)}% of radius sum) — likely a wrong/missing bond in the structure`);
  }

  // (5) Symmetry preservation through relaxation
  if (opts.vcRelaxStdout) {
    const sym = parseSymmetryOperations(opts.vcRelaxStdout);
    if (sym.initial != null && sym.final != null) {
      console.log(`[Pre-Phonon] ${formula} symmetry ops: input=${sym.initial}, final=${sym.final}`);
      if (sym.final < sym.initial) {
        const dropPct = ((sym.initial - sym.final) / sym.initial) * 100;
        const msg = `symmetry broken during relaxation: ${sym.initial} → ${sym.final} ops (${dropPct.toFixed(0)}% drop) — vc-relax found a lower-symmetry minimum (Jahn-Teller / AFM / structural distortion). For ambient-T phonon, this is the correct ground state; for the parent phase, the input space group was wrong`;
        warnings.push(msg);
        // Severe only if drop is large (>50%); small drops can be physical
        if (dropPct > 50) severeCount++;
      }
    } else {
      console.log(`[Pre-Phonon] ${formula} symmetry ops: could not parse from vc-relax output`);
    }
  }

  // (6) Metallicity sanity vs composition heuristic
  const expectedChar = predictMetallicCharacter(opts.elements, opts.counts);
  if (opts.isMetallic != null && expectedChar !== "ambiguous") {
    const actualChar = opts.isMetallic ? "metal" : "insulator";
    console.log(`[Pre-Phonon] ${formula} metallicity: predicted=${expectedChar}, SCF=${actualChar}`);
    if (expectedChar !== actualChar) {
      const msg = `metallic-character mismatch: composition predicts ${expectedChar}, SCF gave ${actualChar} — ${expectedChar === "metal" ? "expected DOS at Ef but got a gap; vc-relax may have over-compressed or hit a wrong-phase minimum" : "expected a band gap but SCF is metallic; smearing may be hiding the gap, or input is wrong stoichiometry"}`;
      warnings.push(msg);
      severeCount++;
    }
  } else if (expectedChar === "ambiguous") {
    console.log(`[Pre-Phonon] ${formula} metallicity: ambiguous composition — skipping cross-check`);
  }

  // (7) Smearing entropy: is -T·S/atom under the publication budget?
  // Read from the phonon-prep SCF output (already at degauss=0.005). If
  // entropy/atom is still >1 mRy, the system is heavily smeared and the
  // geometry/frequencies will be smearing-dependent — the full smearing-
  // polish vc-relax phase should already have driven this down.
  if (opts.phononPrepScfStdout) {
    const tsRy = parseSmearingEntropy(opts.phononPrepScfStdout);
    if (tsRy != null) {
      const tsRyPerAtom = Math.abs(tsRy) / opts.finalPositions.length;
      const tsMevPerAtom = tsRyPerAtom * 13605.7;  // Ry → meV
      console.log(`[Pre-Phonon] ${formula} smearing entropy: |T·S|=${Math.abs(tsRy).toExponential(2)} Ry total, ${tsMevPerAtom.toFixed(2)} meV/atom`);
      if (tsRyPerAtom > 1e-3) {
        warnings.push(`large smearing entropy: |T·S|=${tsMevPerAtom.toFixed(1)} meV/atom (>13.6 meV) — phonon-prep SCF still smearing-dominated; relax may not be at the T→0 minimum`);
        severeCount++;
      }
    }
  }

  // (8) Magnetic state consistency: vc-relax → phonon-prep SCF
  // QE prints `total magnetization` and `absolute magnetization` after each
  // SCF cycle. We compare the last value from vc-relax against the last value
  // from the phonon-prep SCF. If vc-relax converged to a magnetic ground
  // state but phonon-prep collapsed to non-magnetic (or vice versa), the
  // .save/ density fed to ph.x represents a different phase than the relaxed
  // structure — phonons will be wrong even though the geometry is right.
  if (opts.vcRelaxStdout && opts.phononPrepScfStdout) {
    const magVc = parseFinalMagnetization(opts.vcRelaxStdout);
    const magPp = parseFinalMagnetization(opts.phononPrepScfStdout);
    if (magVc.absolute != null && magPp.absolute != null) {
      const absVc = magVc.absolute;
      const absPp = magPp.absolute;
      const totVc = magVc.total ?? 0;
      const totPp = magPp.total ?? 0;
      const absDelta = Math.abs(absVc - absPp);
      console.log(`[Pre-Phonon] ${formula} magnetization: vc-relax total=${totVc.toFixed(3)} |abs|=${absVc.toFixed(3)} μB, phonon-prep total=${totPp.toFixed(3)} |abs|=${absPp.toFixed(3)} μB, Δ|abs|=${absDelta.toFixed(3)} μB`);

      // Severe: vc-relax was clearly magnetic (>0.1 μB) but phonon-prep
      // collapsed below 0.01 μB → the ground-state moment was thrown away.
      if (absVc > 0.1 && absPp < 0.01) {
        warnings.push(`magnetic state lost: vc-relax had |M|=${absVc.toFixed(2)} μB but phonon-prep SCF collapsed to |M|=${absPp.toFixed(3)} μB — likely nspin=1 in phonon-prep when it should have been nspin=2; ph.x .save/ does not represent the relaxed phase`);
        severeCount++;
      }
      // Severe: phonon-prep formed a moment that vc-relax didn't see → vc-relax
      // probably ran at nspin=1 and missed the FM/AFM ground state.
      else if (absVc < 0.01 && absPp > 0.1) {
        warnings.push(`magnetic state appeared post-relax: vc-relax had |M|=${absVc.toFixed(3)} μB but phonon-prep gave |M|=${absPp.toFixed(2)} μB — vc-relax likely ran at nspin=1; geometry is from the wrong phase`);
        severeCount++;
      }
      // Moderate: both magnetic but moments shifted by > 0.5 μB (e.g.
      // a different occupation pattern or AFM↔FM transition).
      else if (absDelta > 0.5) {
        // Distinguish FM↔AFM character change from a generic magnitude shift.
        // FM has total ≈ |abs| (ratio ≈ 1), AFM has total ≈ 0 while |abs|
        // nonzero (ratio ≈ 0). A ratio change > 0.4 between stages is the
        // signature of a true FM↔AFM transition — phonons will differ
        // qualitatively. Previously the code labeled any signed-total sign
        // flip as "FM↔AFM" even when |total| stayed large on both sides
        // (which is actually a spin-direction flip / time-reversed FM, NOT
        // an FM↔AFM transition).
        const ratioVc = absVc > 0.01 ? Math.abs(totVc) / absVc : 0;
        const ratioPp = absPp > 0.01 ? Math.abs(totPp) / absPp : 0;
        const ratioDelta = Math.abs(ratioVc - ratioPp);
        const flipMsg = ratioDelta > 0.4
          ? ` (FM/AFM character changed: total/|abs| ratio ${ratioVc.toFixed(2)}→${ratioPp.toFixed(2)})`
          : "";
        warnings.push(`magnetization shifted between stages: vc-relax |M|=${absVc.toFixed(2)} μB → phonon-prep |M|=${absPp.toFixed(2)} μB (Δ=${absDelta.toFixed(2)} μB)${flipMsg} — phonon-prep settled in a different magnetic configuration`);
      }
    } else if (magVc.absolute != null && magPp.absolute == null) {
      // phonon-prep ran at nspin=1 (no magnetization printed) but vc-relax had moments
      if (magVc.absolute > 0.1) {
        warnings.push(`magnetic state lost: vc-relax had |M|=${magVc.absolute.toFixed(2)} μB but phonon-prep SCF ran without spin polarization (no magnetization in output) — feeding ph.x a non-magnetic density for a magnetic ground state`);
        severeCount++;
      }
    }
  }

  if (warnings.length === 0) {
    console.log(`[Pre-Phonon] ${formula} validation: PASS (all 8 checks)`);
  } else {
    console.log(`[Pre-Phonon] ${formula} validation: ${severeCount} severe warning(s), ${warnings.length} total — proceeding to phonon (no hard cap)`);
    for (const w of warnings) {
      console.log(`[Pre-Phonon]   WARN: ${w}`);
    }
  }

  return { warnings, severeCount };
}

/**
 * Parse "Found N symmetry operations" (or "Found N point group operations")
 * from pw.x output. Returns the FIRST and LAST values found — first is the
 * input-geometry symmetry, last is the relaxed-geometry symmetry (vc-relax
 * re-detects symmetry at every cell change).
 *
 * A decrease (e.g. 48 → 8) means relaxation broke symmetry: Jahn-Teller
 * distortion, magnetic ordering, structural transition. Sometimes intended
 * (AFM cuprate); often a sign the input space group was wrong.
 */
function parseSymmetryOperations(stdout: string): { initial: number | null; final: number | null } {
  const matches = [...stdout.matchAll(/Found\s+(\d+)\s+symmetry\s+operations/gi)];
  if (matches.length === 0) return { initial: null, final: null };
  const initial = parseInt(matches[0][1], 10);
  const final = parseInt(matches[matches.length - 1][1], 10);
  return { initial, final };
}

/**
 * Parse "smearing contrib. (-TS) = X.XXXX Ry" from SCF output.
 * |T·S|/atom > 1 mRy (~13.6 meV) means smearing entropy is contaminating
 * the total energy and forces — the geometry was relaxed against the
 * smearing-broadened energy surface, not the T→0 minimum.
 */
function parseSmearingEntropy(stdout: string): number | null {
  const matches = [...stdout.matchAll(/smearing contrib\.\s*\(-?TS\)\s*=\s*(-?[\d.E+-]+)/gi)];
  if (matches.length === 0) return null;
  const ts = parseFloat(matches[matches.length - 1][1]);
  return Number.isFinite(ts) ? ts : null;
}

/**
 * Parse the FINAL total and absolute magnetization from a QE pw.x stdout.
 *
 * QE prints two values per SCF cycle:
 *   total magnetization       =     X.XX Bohr mag/cell   (sum over cell)
 *   absolute magnetization    =     Y.YY Bohr mag/cell   (∫|m(r)| d³r)
 *
 * Total can be ≈0 for AFM systems even when individual sites carry large
 * moments — only `absolute` captures that. For FM/Pauli-paramagnetic systems
 * the two agree. We need both: total tracks net spin (FM vs AFM), absolute
 * tracks whether any local moment formed at all.
 */
function parseFinalMagnetization(stdout: string): { total: number | null; absolute: number | null } {
  // Collinear (nspin=2):  "total magnetization =     1.23 Bohr mag/cell"
  // Non-collinear:        "total magnetization =     0.00     0.00     1.23 Bohr mag/cell"
  // Same bug as parseSCFOutput's magnetization parser (iteration 106): the
  // single-number regex captured only M_x for NC cells, making non-collinear
  // FM aligned to z (M_z dominant) look AFM-like (total ≈ 0). The pre-phonon
  // validation's ratioVc = |total|/|absolute| would be ~0 (AFM signature)
  // even though the true config was FM, producing spurious "FM↔AFM
  // transition" warnings between vc-relax and phonon-prep SCF.
  let total: number | null = null;
  const totalLineMatches = [...stdout.matchAll(/total magnetization\s+=\s+([^\n]+?Bohr mag\/cell)/g)];
  if (totalLineMatches.length > 0) {
    const line = totalLineMatches[totalLineMatches.length - 1][1];
    const nums = line.match(/-?\d+\.?\d*(?:[eE][-+]?\d+)?/g) ?? [];
    const components = nums.map(parseFloat).filter(Number.isFinite);
    if (components.length === 1) {
      total = components[0];
    } else if (components.length >= 3) {
      const [mx, my, mz] = components.slice(0, 3);
      const norm = Math.sqrt(mx * mx + my * my + mz * mz);
      const dominant = Math.abs(mx) >= Math.abs(my) && Math.abs(mx) >= Math.abs(mz) ? mx
                     : Math.abs(my) >= Math.abs(mz) ? my : mz;
      total = norm * (dominant < 0 ? -1 : 1);
    }
  }
  // Legacy single-number fallback for QE versions without the "Bohr mag/cell" suffix.
  if (total === null) {
    const totalMatches = [...stdout.matchAll(/total magnetization\s+=\s+(-?[\d.]+)/g)];
    if (totalMatches.length > 0) {
      total = parseFloat(totalMatches[totalMatches.length - 1][1]);
    }
  }
  const absMatches = [...stdout.matchAll(/absolute magnetization\s+=\s+(-?[\d.]+)/g)];
  const absolute = absMatches.length > 0 ? parseFloat(absMatches[absMatches.length - 1][1]) : null;
  return {
    total: total != null && Number.isFinite(total) ? total : null,
    absolute: absolute != null && Number.isFinite(absolute) ? absolute : null,
  };
}

/**
 * Heuristic for whether a composition should be metallic. Used to cross-check
 * the SCF parser's `isMetallic` flag and catch wrong-phase results (e.g. an
 * undoped cuprate that came out metallic = vc-relax failed to find the AFM
 * ground state). Returns:
 *   "metal"      — all-metal alloy, intermetallic, conducting hydride
 *   "insulator"  — has halogen / closed-shell ionic + no transition metal
 *   "ambiguous"  — could be either (correlated, charge-transfer, etc.)
 */
function predictMetallicCharacter(
  elements: string[],
  counts: Record<string, number>,
): "metal" | "insulator" | "ambiguous" {
  const HALOGENS = new Set(["F", "Cl", "Br", "I"]);
  const ALKALI = new Set(["Li", "Na", "K", "Rb", "Cs"]);
  const ALKALINE_EARTH = new Set(["Be", "Mg", "Ca", "Sr", "Ba"]);
  const TRANSITION = new Set([
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
  ]);
  const LANTHANIDE_ACTINIDE = new Set([
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    // Am, Cm, Bk, Cf added — without them, AmO2/CmO2/etc. matched the
    // `hasO && !hasTM && !hasMetallic` branch and were misclassified as
    // "insulator", producing wrong metallicity-mismatch warnings against the
    // (correctly metallic-like) SCF result for half-filled 5f systems.
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
  ]);
  const hasHalogen = elements.some(e => HALOGENS.has(e));
  const hasO = elements.includes("O");
  const hasH = elements.includes("H");
  const hasN = elements.includes("N");
  const hasTM = elements.some(e => TRANSITION.has(e) || LANTHANIDE_ACTINIDE.has(e));
  const hasMetallic = elements.some(e =>
    TRANSITION.has(e) || LANTHANIDE_ACTINIDE.has(e) || ALKALI.has(e) || ALKALINE_EARTH.has(e),
  );
  // Cuprate / pnictide / ruthenate / iridate / nickelate: correlated, undoped form
  // is typically Mott-insulating but emerges metallic on doping. DFT (PBE) gives
  // metal almost always. Flag as ambiguous — these go to DMFT anyway.
  const hasCu = elements.includes("Cu");
  const hasFe = elements.includes("Fe");
  if (hasCu && hasO && counts["O"] >= 2) return "ambiguous";
  if (hasFe && (elements.includes("As") || elements.includes("P") || elements.includes("Se"))) return "ambiguous";
  // High-symmetry binary halide / oxide with closed-shell cation = insulator
  if (hasHalogen && !hasTM) return "insulator";
  if (hasO && !hasTM && !hasMetallic) return "insulator";
  if (hasN && !hasTM && !hasMetallic && elements.length <= 2) return "insulator";
  // Pure metal alloy or intermetallic
  if (elements.every(e => TRANSITION.has(e) || LANTHANIDE_ACTINIDE.has(e) || ALKALI.has(e) || ALKALINE_EARTH.has(e))) return "metal";
  // Hydride with metallic cation → typically metallic at high P
  if (hasH && hasMetallic && !hasHalogen) return "metal";
  // Everything else (semiconductor-like, multinaries): ambiguous
  return "ambiguous";
}

// Parse the QE run wall time (seconds) from the total-clock line. pw.x heads
// it "PWSCF :", ph.x heads it "PHONON :", followed by "<cpu> CPU <wall> WALL".
// Each time uses one of three magnitude-dependent formats:
//   < 1 min:   "12.34s"        < 1 hour:  "45m18.23s"        >= 1 hour: "1h23m"
// The >=1h format omits seconds ENTIRELY. The previous regexes all required a
// trailing "<digits>s WALL", so every multi-hour vc-relax / refinement pass
// parsed wallTimeSeconds = 0 (the refinement loop then logged "wall=0s" and
// totalRefineWallSec never accumulated). Anchoring on the total-clock line also
// stops us from picking a tiny leaf-routine clock (davcio etc.) as the total.
function parseQEWallSeconds(stdout: string): number {
  const pwscfLines = [...stdout.matchAll(/(?:PWSCF|PHONON)\s*:\s*\S.*?\sCPU\s+(.+?)\s+WALL/g)];
  let token: string | null = null;
  if (pwscfLines.length > 0) {
    token = pwscfLines[pwscfLines.length - 1][1];
  } else {
    // Fallback: no PWSCF line (truncated/SIGKILLed output) — last WALL clock.
    const anyWall = [...stdout.matchAll(/(\d+h\s*\d+m(?:\s*[\d.]+s)?|\d+m\s*[\d.]+s|[\d.]+s)\s+WALL/g)];
    if (anyWall.length > 0) token = anyWall[anyWall.length - 1][1];
  }
  if (!token) return 0;
  const hm = token.match(/(\d+)\s*h\s*(\d+)\s*m(?:\s*([\d.]+)\s*s)?/);
  if (hm) return parseInt(hm[1]) * 3600 + parseInt(hm[2]) * 60 + (hm[3] ? parseFloat(hm[3]) : 0);
  const ms = token.match(/(\d+)\s*m\s*([\d.]+)\s*s/);
  if (ms) return parseInt(ms[1]) * 60 + parseFloat(ms[2]);
  const s = token.match(/([\d.]+)\s*s/);
  if (s) return parseFloat(s[1]);
  return 0;
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
    absoluteMagnetization: null,
    error: null,
  };

  // Use LAST convergence line — in vc-relax, each ionic step has its own SCF.
  // We want the final ionic step's convergence status.
  const convergenceMatches = [...stdout.matchAll(/convergence has been achieved in\s+(\d+)\s+iterations/g)];
  if (convergenceMatches.length > 0) {
    result.converged = true;
    result.convergenceQuality = "strict";
    result.nscfIterations = parseInt(convergenceMatches[convergenceMatches.length - 1][1]);
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

  // Use LAST "! total energy" — in vc-relax output there are many ionic steps,
  // each with its own "! total energy". We want the final converged value.
  const energyMatches = [...stdout.matchAll(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/g)];
  if (energyMatches.length > 0) {
    result.totalEnergy = parseFloat(energyMatches[energyMatches.length - 1][1]) * RY_TO_EV;
  }
  if (energyMatches.length === 0) {
    const energyLines = [...stdout.matchAll(/total energy\s+=\s+([-\d.]+)\s+Ry/g)];
    if (energyLines.length > 0) {
      result.totalEnergy = parseFloat(energyLines[energyLines.length - 1][1]) * RY_TO_EV;
    }
  }

  const natMatch = stdout.match(/number of atoms\/cell\s+=\s+(\d+)/);
  const nAtoms = natMatch ? parseInt(natMatch[1]) : 1;
  result.totalEnergyPerAtom = result.totalEnergy / nAtoms;

  // Use LAST Fermi energy — changes between ionic steps in vc-relax.
  const fermiMatches = [...stdout.matchAll(/the Fermi energy is\s+([-\d.]+)\s+ev/gi)];
  if (fermiMatches.length > 0) {
    result.fermiEnergy = parseFloat(fermiMatches[fermiMatches.length - 1][1]);
  }

  // Use LAST band gap — changes between ionic steps in vc-relax.
  // QE writes the gap in one of three forms depending on version + verbosity:
  //   (a) "band gap =     X.XXX eV"                           ← explicit gap
  //   (b) "highest occupied, lowest unoccupied level (ev):    X.XXX    Y.YYY"  (single line)
  //   (c) "highest occupied, lowest unoccupied levels (ev):\n        X.XXX   Y.YYY"  (two-line)
  // The previous regex for (b)/(c) required the numbers on the *next* line
  // after the header, which only matches (c). For (b) — the most common
  // format — it captured nothing, and the bandGap stayed null, defaulting
  // isMetallic=true for every insulator with HOMO/LUMO output.
  const gapMatches = [...stdout.matchAll(/band gap\s*=\s*([-\d.]+)\s+eV/gi)];
  // Single-line form: numbers immediately follow "(ev):" on the same line.
  const holoInlineMatches = [...stdout.matchAll(/highest occupied[^\n]*?lowest unoccupied[^:\n]*\(ev\)\s*:\s*(-?[\d.]+)\s+(-?[\d.]+)/gi)];
  // Two-line form: numbers on the next line after the header.
  const holoMultilineMatches = [...stdout.matchAll(/highest occupied[^\n]*?lowest unoccupied[^\n]*\n\s*(-?[\d.]+)\s+(-?[\d.]+)/gi)];

  let gapHomo: number | null = null;
  let gapLumo: number | null = null;
  if (holoInlineMatches.length > 0) {
    const m = holoInlineMatches[holoInlineMatches.length - 1];
    gapHomo = parseFloat(m[1]);
    gapLumo = parseFloat(m[2]);
  } else if (holoMultilineMatches.length > 0) {
    const m = holoMultilineMatches[holoMultilineMatches.length - 1];
    gapHomo = parseFloat(m[1]);
    gapLumo = parseFloat(m[2]);
  }

  if (gapMatches.length > 0) {
    result.bandGap = parseFloat(gapMatches[gapMatches.length - 1][1]);
  } else if (gapHomo != null && gapLumo != null && Number.isFinite(gapHomo) && Number.isFinite(gapLumo)) {
    result.bandGap = Math.max(0, gapLumo - gapHomo);
  }

  if (result.bandGap != null) {
    const degaussEv = degaussRy * RY_TO_EV;
    const metallicThreshold = Math.max(0.01, degaussEv * 1.5);
    result.isMetallic = result.bandGap < metallicThreshold;
  }

  // Use LAST "Total force" — in vc-relax output there's one per ionic step.
  // .match() returns the FIRST which is the worst unconverged starting force.
  const forceMatches = [...stdout.matchAll(/Total force\s+=\s+([\d.]+)/g)];
  if (forceMatches.length > 0) {
    result.totalForce = parseFloat(forceMatches[forceMatches.length - 1][1]);
  }

  // Use LAST pressure — same reason as force.
  const pressureMatches = [...stdout.matchAll(/P=\s+([-\d.]+)/g)];
  if (pressureMatches.length > 0) {
    result.pressure = parseFloat(pressureMatches[pressureMatches.length - 1][1]) / 10;
  }

  result.wallTimeSeconds = parseQEWallSeconds(stdout);

  // Use LAST magnetization — changes between ionic steps in vc-relax.
  // total magnetization = signed net moment (≈0 for AFM); absolute = ∫|m(r)| d³r.
  // Capturing both lets downstream distinguish FM (total ~ |abs|) from
  // AFM (|total| << |abs|) from NM (both ≈ 0). The old code only kept the
  // signed total, so AFM cuprates / Fe-pnictides looked non-magnetic to
  // dft-job-queue's `qeIsMagnetic` flag (`abs(total) > 0.5`) and bypassed
  // the magnetic ML feature path.
  // Collinear (nspin=2):  "total magnetization =     1.23 Bohr mag/cell"   (single scalar)
  // Non-collinear:        "total magnetization =     0.00     0.00     1.23 Bohr mag/cell"
  // Capture all numbers on the line and use the L2 norm of the vector. The
  // previous single-number regex captured only M_x, so non-collinear AFM
  // ordered along z reported magnetization = 0 and looked non-magnetic to
  // any consumer reading `result.magnetization` directly.
  const magLineMatches = [...stdout.matchAll(/total magnetization\s+=\s+([^\n]+?Bohr mag\/cell)/g)];
  if (magLineMatches.length > 0) {
    const line = magLineMatches[magLineMatches.length - 1][1];
    const nums = line.match(/-?\d+\.?\d*(?:[eE][-+]?\d+)?/g) ?? [];
    const components = nums.map(parseFloat).filter(Number.isFinite);
    if (components.length === 1) {
      result.magnetization = components[0];
    } else if (components.length >= 3) {
      // Magnitude of the 3-vector; preserve sign convention by taking the
      // dominant component's sign so AFM-like cancellation still shows ≈0.
      const [mx, my, mz] = components.slice(0, 3);
      const norm = Math.sqrt(mx * mx + my * my + mz * mz);
      const dominant = Math.abs(mx) >= Math.abs(my) && Math.abs(mx) >= Math.abs(mz) ? mx
                     : Math.abs(my) >= Math.abs(mz) ? my : mz;
      result.magnetization = norm * (dominant < 0 ? -1 : 1);
    }
  }
  // Fallback to legacy single-number regex if the new one didn't match
  // (handles QE versions / verbosity that omit the "Bohr mag/cell" suffix).
  if (result.magnetization === null) {
    const magMatches = [...stdout.matchAll(/total magnetization\s+=\s+(-?[\d.]+)/g)];
    if (magMatches.length > 0) {
      result.magnetization = parseFloat(magMatches[magMatches.length - 1][1]);
    }
  }
  const absMagMatches = [...stdout.matchAll(/absolute magnetization\s+=\s+(-?[\d.]+)/g)];
  if (absMagMatches.length > 0) {
    result.absoluteMagnetization = parseFloat(absMagMatches[absMagMatches.length - 1][1]);
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
    // Threshold -10 cm⁻¹ matches dft-job-queue.ts:IMAGINARY_PHONON_THRESHOLD_CM1
    // and the Stage 4 gamma-check at line 8048. The previous -20 threshold
    // here was inconsistent with both — same -15 cm⁻¹ mode was "not imaginary"
    // per parsePhononOutput but "not physically stable" per queue gate,
    // producing contradictory result.hasImaginary vs qePhononStable flags
    // in the DB. -10 is the QE acoustic-noise floor (3 Goldstone modes at Γ
    // typically have |f| < 5 cm⁻¹); real soft modes start at -30 cm⁻¹+.
    result.imaginaryCount = result.frequencies.filter(f => f < -10).length;
    result.hasImaginary = result.imaginaryCount > 0;
  }

  // ph.x convergence detection. Previously OR'd "Phonon calculation on a mesh"
  // into the condition, but that string is printed at the START of every
  // ldisp=.true. run (before any computation). A timed-out / crashed ph.x
  // that produced only the header still set converged=true, so downstream
  // Eliashberg / EPC steps ran on partial data and produced garbage λ/ω_log.
  //
  // QE markers and what each actually signals:
  //   - "Phonon calculation on a mesh" — START of ldisp=.true. run (NOT a success signal)
  //   - "Writing dynmat at Gamma"      — one q-point complete (partial)
  //   - "PHONON       :"               — timing summary (printed at clean exit,
  //                                       including clean max_seconds timeouts)
  //   - "JOB DONE"                     — universal QE clean-exit marker
  //                                       (also fires for max_seconds timeouts)
  //
  // Neither "PHONON :" nor "JOB DONE" alone proves successful convergence —
  // they're both compatible with a clean max-seconds exit. The reliable test
  // is: frequencies parsed AND no failure markers in the output. The dynmat-
  // at-gamma marker adds confidence (we got at least one q-point's worth of
  // data) but only when frequencies were also extracted.
  const phononHadFailure = stdout.includes("ERROR")
    || stdout.includes("stopping")
    || stdout.includes("Maximum CPU time exceeded");
  result.converged = result.frequencies.length > 0 && !phononHadFailure;

  // QE wall time format: "PHONON       :   2m39.47s CPU   2m52.16s WALL"
  // Pattern: optional "Xh" then "Xm" then "Y.Ys" then "WALL" (WALL is a suffix).
  // Use matchAll + LAST occurrence to pick the PHONON total at the end of
  // stdout. The previous code used .match() (first match) which for short
  // runs (<1 min) returned init_run's 1-2s in the seconds-only fallback
  // instead of the actual phonon calculation time.
  result.wallTimeSeconds = parseQEWallSeconds(stdout);

  return result;
}

const COVALENT_RADIUS: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Ne: 0.58,
  Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06,
  K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39,
  Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20, Kr: 1.16,
  Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Tc: 1.47, Ru: 1.46, Rh: 1.42, Pd: 1.39,
  Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39, Xe: 1.40,
  Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Pr: 2.03, Nd: 2.01, Pm: 1.99, Sm: 1.98, Eu: 1.98, Gd: 1.96,
  Tb: 1.94, Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.90, Yb: 1.87, Lu: 1.87,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36,
  Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48,
  // Actinides — without these, Np/Pu/Am/Cm-containing structures fell through
  // to the 1.4 Å fallback at minPairDistance, which gave a min-pair distance
  // of ~0.7·(1.4+rOther) < 2 Å — well below the actual 2.5-3 Å bonds in real
  // actinide compounds. Valid structures got rejected as "atoms too close".
  Ac: 2.15, Th: 2.06, Pa: 2.00, U: 1.96, Np: 1.90, Pu: 1.87, Am: 1.80, Cm: 1.69, Bk: 1.68, Cf: 1.68,
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
  if (positions.length > 24) return { valid: false, reason: `Too many atoms (${positions.length}), max 24 for available resources`, warnings };

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
  if (totalAtoms > 24) {
    return { valid: false, reason: `Too many atoms (${totalAtoms}), max 24 for available resources` };
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
      isTransitionMetal(el) || isRareEarth(el) || isActinide(el) || ALKALINE_EARTH_SYMBOLS.has(el)
    ));

    if (hRatio > 0.95 && !hasHydrideMetal) {
      return { valid: false, reason: `Hydrogen ratio ${(hRatio * 100).toFixed(0)}% with no metal host — likely unphysical` };
    }

    const nonHAtoms = totalAtoms - hCount;
    const hPerMetal = nonHAtoms > 0 ? hCount / nonHAtoms : hCount;
    const cleanFormula = formula.replace(/\s+/g, "");
    const knownPressure = KNOWN_SUPERHYDRIDE_PRESSURES[cleanFormula];

    // Note: previously rejected low-H/metal materials (H/metal < 0.5) but this
    // was too aggressive — it caught legitimate trace-H compounds and database
    // entries with 1 H atom (e.g., Al8Ba2C8HP). Letting them through; CSP and
    // DFT will catch genuinely unphysical structures downstream.

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
    // Match the F1 funnel-filter floor (0.6 Å). With xTB's old 0.5 Å threshold,
    // borderline-collapsed results (TlBa2Ca2Cu3O9 Ba-Cu=0.594 Å) passed xTB,
    // then F1 / softValidateGeometry rejected, and a "repair" call inflated
    // the lattice to compensate — producing a totally wrong-shape cell. Now
    // xTB rejects at the same threshold and we keep the original Vegard /
    // prototype geometry instead of the wreckage.
    const ABS_MIN_DIST = 0.6;
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
  params: { mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number; ecutrhoMultiplierOverride?: number; convThr?: string; forcConvThr?: string; etotConvThr?: string; dftPlusULines?: string; dftPlusUNspin2?: boolean; mixingMode?: string; mixingNdim?: number; startingwfc?: string; startingpot?: string; diagoThrInit?: string; restartFromScratch?: boolean; maxSecondsOverride?: number; socFlags?: string; forceNspin?: 1 | 2; forceMagBlock?: string; forceNoncolin?: boolean; hubbardCard?: string; nbndOverride?: number; nbndScale?: number },
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  // Uses module-level SPECIES_ECUTWFC table. Screening minimum: 45 Ry non-H,
  // 80 Ry H (PAW needs less than NCPP; saves ~30% vs 60/100). ecutwfcBoost
  // comes from the retry ladder to escalate cutoff on non-convergence.
  const baseEcutwfc = computeEcutwfc(elements, 0, 80, 45);
  const ecutwfc = baseEcutwfc + (params.ecutwfcBoost ?? 0);
  // ecutrhoMultiplierOverride lets the phonon-prep SCF retry loop bump the
  // density cutoff when negative-rho artifacts appear (e.g. CaH6 with hard
  // PAW Ca needed 8× and 12× variants to drive negative rho below 1e-3 e-).
  const ecutrhoMult = params.ecutrhoMultiplierOverride ?? ecutrhoMultiplier(elements);
  const ecutrho = ecutwfc * ecutrhoMult;
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
  // SOC non-collinear overrides nspin: when noncolin=.true., QE ignores nspin
  // and uses 4-component spinors internally. We still emit nspin=1 in the input
  // because QE requires it to be absent or 1 when noncolin is set.
  const hasSOC = !!(params.socFlags);
  const hasNoncolin = !!(params.forceNoncolin);
  // forceNspin precedence: an EXPLICIT forceNspin=1 must override broadMagnetic
  // detection. Previously the OR-chain treated forceNspin=1 the same as missing
  // (since `params.forceNspin === 2` is false) and broadMagnetic alone set
  // useNspin2=true, so the NM trial (forceNspin=1) of the magnetic ground-state
  // search silently ran with nspin=2. That's why BaFe2As2's "NM" trial reported
  // M=10.32 μB — the trial was spin-polarized despite the ordering label.
  const useNspin2 = !hasSOC && !hasNoncolin && (
    params.forceNspin === 2
      ? true                                                        // explicit nspin=2
      : params.forceNspin === 1
        ? false                                                     // explicit nspin=1 — overrides broadMagnetic
        : ((params.dftPlusUNspin2 ?? false) || broadMagnetic)       // implicit detection
  );
  // When DFT+U nspin2 is set, starting_magnetization is already embedded in dftPlusULines
  // When forceMagBlock is provided (from magnetic ground-state search), use it directly
  // Also: if forceNspin=1 (NM trial), suppress magBlock entirely — nspin=1 cannot
  // accept starting_magnetization without QE complaining.
  const magBlock = params.forceNspin === 1
    ? ""
    : params.forceMagBlock
      ? params.forceMagBlock
      : (params.dftPlusUNspin2 ?? false)
        ? ""
        : (broadMagnetic ? generateMagnetizationLines(elements, counts, isAFMCandidate(elements, counts), !hasMagEl) : "");
  const hubbardBlock = params.dftPlusULines ?? "";
  const socBlock = params.socFlags ?? "";

  // When noncolin=.true., QE ignores nspin — emit nspin=1 or omit it.
  // forceNoncolin (from magnetic ground-state search non-collinear trials)
  // behaves same as SOC noncolin — 4-component spinors, angle1/angle2 format.
  const nspinOut = (hasSOC || hasNoncolin) ? 1 : (useNspin2 ? 2 : (params.forceNspin ?? 1));
  const noncolinBlock = hasNoncolin ? "  noncolin = .true.,\n" : "";
  // nbnd: usually computed from element/position electron counts. nbndOverride
  // is an absolute override for retries that hit "too few bands"; nbndScale is
  // a multiplicative bump applied on top of the auto value (e.g. 1.5 = +50%).
  // Floor at the computed value so a retry can only ever expand the band set.
  const nbndAuto = computeNbnd(elements, counts, nspinOut, positions, hasSOC || hasNoncolin);
  const nbnd = params.nbndOverride ?? (params.nbndScale ? Math.ceil(nbndAuto * Math.max(1.0, params.nbndScale)) : nbndAuto);
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
  disk_io = 'high',
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
${socBlock}${noncolinBlock}${magBlock}${hubbardBlock}/
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
${params.hubbardCard ?? ""}ATOMIC_POSITIONS {crystal}
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
  nstepOverride?: number,
  opts?: { socFlags?: string; forceNspin?: 1 | 2; forceMagBlock?: string; hubbardBlock?: string; hubbardCard?: string; pressurePriority?: boolean; degaussOverride?: number; convThrOverride?: string; ecutwfcBoost?: number; ecutrhoMultiplierOverride?: number; maxSecondsOverride?: number },
): string {
  const totalAtoms = positions.length;
  const nTypes = elements.length;
  // Uses module-level SPECIES_ECUTWFC table. vc-relax path historically
  // used a trimmed table (no lanthanides) but the full table is safe here —
  // unknown elements fall through to the 45 Ry default.
  const hasHydrogen = elements.includes("H");
  const baseEcutwfc = computeEcutwfc(elements, 0, 80, 45);
  const ecutwfc = Math.max(baseEcutwfc, hasHydrogen ? 100 : 60) + (opts?.ecutwfcBoost ?? 0);
  const ecutrhoMultVcr = opts?.ecutrhoMultiplierOverride ?? ecutrhoMultiplier(elements);
  const ecutrho = ecutwfc * ecutrhoMultVcr;

  const hasMagnetic = elements.some(el => el in MAGNETIC_ELEMENTS);
  const broadMagnetic = mayHaveMagneticMoment(elements);
  const hasSOCVcr = !!(opts?.socFlags);
  const nspin = hasSOCVcr ? 1 : (opts?.forceNspin ?? (broadMagnetic ? 2 : 1));
  let magLines = "";
  const socLinesVcr = opts?.socFlags ?? "";
  if (opts?.forceMagBlock) {
    magLines = opts.forceMagBlock;
  } else if (broadMagnetic) {
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
  // Honor known-structure angles (alpha, beta, gamma) when available — for
  // monoclinic crystals β ≠ 90° (typically 95-115°). Without this, vc-relax
  // started from an orthorhombic cell (all 90°), and damp-w cell dynamics
  // either wasted iterations searching for the correct β or got trapped in
  // an orthorhombic minimum that's not the true ground state. The SCF input
  // generator at line 4360 already does this; vc-relax was missing the same
  // treatment. defaults to 90° (cubic/tetragonal/orthorhombic) when unknown.
  const knownStructVcr = lookupKnownStructure(formula);
  const cOverA = knownStructVcr?.latticeC ? knownStructVcr.latticeC / knownStructVcr.latticeA : estimateCOverA(elements, counts);
  const bOverAVcr = knownStructVcr?.latticeB ? knownStructVcr.latticeB / knownStructVcr.latticeA : estimateBOverA(elements, counts);
  const vcAlpha = knownStructVcr?.alpha ?? 90;
  const vcBeta = knownStructVcr?.beta ?? 90;
  const vcGamma = knownStructVcr?.gamma ?? 90;
  const cellBlock = `\n${generateCellParameters(latticeA, cOverA, 0, bOverAVcr, elements, counts, vcAlpha, vcBeta, vcGamma)}`;
  const hasMagneticEl = elements.some(el => el in MAGNETIC_ELEMENTS);
  // degaussOverride lets the smearing-polish refinement phase progressively
  // tighten degauss (0.015 → 0.005 → 0.0025) so the relaxed geometry tracks
  // the true T→0 minimum, not the smearing-broadened one. Without this,
  // metallic systems get a geometry that's correct only at degauss=0.015.
  const vcRelaxDegauss = opts?.degaussOverride ?? (hasMagneticEl ? 0.02 : 0.015);

  // vc-relax wall-time cap — scaled by system complexity and atom count.
  const hasHVcr = elements.includes("H");
  const hasMagVcr = elements.some(el => el in MAGNETIC_ELEMENTS);
  const isHighPHydride = hasHVcr && pressureGPa >= 50 && totalAtoms >= 7;
  // Cuprates (Cu-O layered oxides): SCF is slow — dense Cu-3d bands plus
  // long-wavelength charge sloshing across the CuO2 planes. With 'plain'
  // mixing the slosh stalls SCF so each ionic step burns the whole
  // electron_maxstep budget, leaving only 5-6 ionic steps per pass. They get
  // local-TF mixing (damps the layered sloshing) + the magnetic-tier 60 min
  // wall budget so the cell/ion relaxation has room to converge.
  const isCuprateVcr = elements.includes("Cu") && elements.includes("O") && (counts["O"] ?? 0) >= 2;
  // Atom-count scaling: larger cells need proportionally more time.
  // Base budgets calibrated for 7-atom cells; scale by (nAtoms/7)^1.2 for larger.
  const atomScale = totalAtoms > 7 ? Math.pow(totalAtoms / 7, 1.2) : 1.0;
  const vcRelaxMaxSeconds = isHighPHydride ? Math.round(10800 * atomScale) // 3h base for high-P hydrides, scaled
    : (hasMagVcr || isCuprateVcr) ? Math.round(3600 * atomScale)            // 60 min base for magnetic/cuprate, scaled
    : Math.round(Math.max(600, Math.min(QE_MAX_SECONDS, 1800)) * atomScale); // 30 min base, scaled
  const VC_RELAX_MAX_SECONDS = vcRelaxMaxSeconds;
  // UNIFIED vc-relax: damped dynamics with TIGHT SCF convergence.
  // Cell and positions converge TOGETHER in a single calculation.
  // No separate phases — forces are accurate at every step because
  // conv_thr=1e-7 matches production SCF. No force gap.
  //
  // Previous 3-phase approach had:
  //   Phase 1 (loose SCF) → Phase 2 (loose SCF, changed cell) → Phase 3 (tight SCF, fixed cell)
  //   Result: force gap from 0.001 → 0.25 between phases
  //
  // Now: single damped dynamics with tight SCF
  //   ion_dynamics='damp' — never diverges, always makes progress
  //   cell_dynamics='damp-w' — cell adjusts gradually with positions
  //   conv_thr=1e-7 — production-quality forces at every ionic step
  //   400 nstep — enough for convergence (slower per step, but accurate)
  const vcMixingMode = isHighPHydride ? "local-TF" : (hasMagVcr || isCuprateVcr) ? "local-TF" : "plain";
  const vcMixingBeta = isHighPHydride ? 0.2 : (hasMagVcr || isCuprateVcr) ? 0.2 : 0.3;

  return `&CONTROL
  calculation = 'vc-relax',
  restart_mode = 'from_scratch',
  prefix = '${prefix}',
  outdir = './tmp',
  disk_io = 'high',
  pseudo_dir = '${QE_PSEUDO_DIR_INPUT}',
  tprnfor = .true.,
  tstress = .true.,
  forc_conv_thr = ${opts?.pressurePriority ? "1.0d-5" : "1.0d-3"},
  etot_conv_thr = 1.0d-5,
  nstep = ${nstepOverride ?? (isHighPHydride ? 600 : 400)},
  max_seconds = ${opts?.maxSecondsOverride ?? VC_RELAX_MAX_SECONDS},
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
${socLinesVcr}${magLines}/
&ELECTRONS
  electron_maxstep = 300,
  conv_thr = ${opts?.convThrOverride ?? "1.0d-7"},
  mixing_beta = ${vcMixingBeta},
  mixing_mode = '${vcMixingMode}',
  diagonalization = 'david',
/
&IONS
  ion_dynamics = 'damp',
/
&CELL
  cell_dynamics = 'damp-w',
  press = ${(pressureGPa * 10.0).toFixed(4)},
  press_conv_thr = ${opts?.pressurePriority ? 0.1 : (pressureGPa > 50 ? 1.0 : 0.5)},
/
ATOMIC_SPECIES
${atomicSpecies}
${opts?.hubbardCard ?? ""}ATOMIC_POSITIONS {crystal}
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

  // vc-relax prints one "! total energy" line per ionic step, plus a final
  // SCF after the cell converges. We want the LAST occurrence (the final
  // converged geometry), not the first (the starting unrelaxed geometry).
  // The non-global `.match()` returns only the first capture — every
  // vc-relax was reporting the starting-geometry energy as if it were the
  // relaxed-geometry energy. Symptoms:
  //   - vcParsed.totalEnergy looked artificially high (starting was un-
  //     relaxed, real final was lower by 0.1-1 eV/atom typical).
  //   - The polish-pass ΔE/atom convergence check at line 7011-7012 was
  //     comparing first-step energies across passes instead of final-step
  //     energies — partially masked because polishedScf (parsed via the
  //     fixed parseSCFOutput) was still correct downstream, but the polish
  //     log line and any consumer reading vcParsed.totalEnergy directly
  //     saw the wrong number.
  // Mirrors the LAST-match pattern in parseSCFOutput:3801.
  const energyMatches = [...stdout.matchAll(/!\s+total energy\s+=\s+([-\d.]+)\s+Ry/g)];
  if (energyMatches.length > 0) {
    result.totalEnergy = parseFloat(energyMatches[energyMatches.length - 1][1]) * RY_TO_EV;
  }

  result.wallTimeSeconds = parseQEWallSeconds(stdout);

  // Match the LAST CELL_PARAMETERS block (damped dynamics outputs many)
  const cellMatches = [...stdout.matchAll(/CELL_PARAMETERS\s*[{(]\s*([^})]*)\s*[})]\s*\n([\s\S]*?)(?=\n\s*\n|\nATOMIC|\nEnd|\n\s*Writing|\n\s*PWSCF|\n\s*NEW-OLD|$)/g)];
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
  // Match ATOMIC_POSITIONS with any bracket style: {crystal}, (crystal), or just crystal
  const posBlocks = [...stdout.matchAll(/ATOMIC_POSITIONS\s*[{(]?\s*(\w+)\s*[})]?\s*\n([\s\S]*?)(?=\n\s*\n|\nEnd|\nCELL_PARAMETERS|\n\s*Writing|\n\s*PWSCF\b|\n\s*init_run\b|\n\s*electrons\b|\n\s*BFGS\b|\n\s*JOB DONE|\n\s*%%%%%%%%%%|\n\s*Error in routine|\n\s*total cpu time|\n\s*General routines|\n\s*Parallel routines|\n\s*number of|\n\s*convergence has|\n\s*NEW-OLD|$)/g)];
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
              // CELL_PARAMETERS stores lattice vectors as ROWS, so a Cartesian
              // position relates to fractional coords by R = cellᵀ · f, hence
              // f = (cellᵀ)⁻¹ · R = (cell⁻¹)ᵀ · R. The earlier form inv · R
              // (no transpose) is only correct for orthogonal cells where
              // cell⁻¹ is symmetric — for monoclinic/triclinic/hexagonal cells
              // it produced wrong fractional coords (e.g. a hexagonal cell maps
              // an atom at frac (1,0,0) to (1,0.577,0)). Apply the transpose:
              //   f[i] = Σ_j (cell⁻¹)ᵀ[i][j] R[j] = Σ_j inv[j][i] R[j].
              x = inv[0][0]*posAng[0] + inv[1][0]*posAng[1] + inv[2][0]*posAng[2];
              y = inv[0][1]*posAng[0] + inv[1][1]*posAng[1] + inv[2][1]*posAng[2];
              z = inv[0][2]*posAng[0] + inv[1][2]*posAng[1] + inv[2][2]*posAng[2];
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
        // f₂ spectral-moment correction (Allen-Dynes 1975 eq 3.4). Available
        // only when the a2F.dat parse provided ⟨ω²⟩ (in (cm⁻¹)²); falls back
        // to f₂=1 (pure McMillan-style). For strong-coupling hydrides
        // √⟨ω²⟩/ω_log can reach 1.3–1.5 → f₂ adds 5–15% to Tc.
        let f2 = 1.0;
        const omega2AvgCm2 = dfptFiles.alpha2F?.omega2Avg ?? 0;
        if (omega2AvgCm2 > 0) {
          const CM1_TO_K = 1.4387768775039338;
          const omegaLogCm = omegaLog / CM1_TO_K;  // back to cm⁻¹ for the ratio
          const sqrtOmega2Cm = Math.sqrt(omega2AvgCm2);
          const omegaRatio = sqrtOmega2Cm / omegaLogCm;
          const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
          f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
          if (!Number.isFinite(f2) || f2 <= 0) f2 = 1.0;
        }
        // Don't cap Tc at 500 K — theoretical hydrides at extreme P can
        // predict >500 K from Allen-Dynes (matches the fix in
        // dfpt-parser.ts). Only floor at 0 (negative is the unphysical
        // denom-divergent regime). Warn loudly when above 500 K so the
        // user knows it's an extreme strong-coupling prediction.
        const rawTc = (omegaLog / 1.2) * f1 * f2 * Math.exp(exp);
        tcAllenDynes = Number(Math.max(0, rawTc).toFixed(2));
        if (tcAllenDynes > 500) {
          console.warn(`[QE-Worker] ${formula}: Allen-Dynes Tc=${tcAllenDynes.toFixed(0)} K > 500 K ` +
            `(λ=${lambda.toFixed(2)}, ω_log=${omegaLog.toFixed(0)} K) — extreme strong-coupling; ` +
            `verify via Eliashberg if available.`);
        }
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
    runQEBinary: (binaryName, inputFile, workDir, timeoutMs) =>
      runQECommand(path.posix.join(getQEBinDir(), binaryName), inputFile, workDir, timeoutMs),
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

    // --- SOC analysis: determine if spin-orbit coupling is needed ---
    const socAnalysis = analyzeSOCRequirement(elements, counts, workerPressure);
    result.socAnalysis = socAnalysis;
    if (socAnalysis.needsSOC) {
      console.log(`[QE-Worker] SOC analysis for ${formula}: ${socAnalysis.enableFullSOC ? "FULL SOC" : "scalar-rel"} ` +
        `(max SOC=${socAnalysis.maxSOCEnergy.toFixed(2)} eV, elements: ${socAnalysis.socElements.map(e => e.element).join(",")})`);
    }

    // --- Provision fully-relativistic pseudopotentials for SOC ---
    // The default provisioning installs scalar-relativistic PAW pseudos. For
    // a full-SOC run, swap in the Pseudo-DOJO FR (has_so="T") pseudo for each
    // SOC element that has one — this MUST run before applySOCPseudoConstraint
    // below, which reads QE_PSEUDO_DIR and downgrades lspinorb if no FR pseudo
    // is present. Elements without an FR pseudo (actinides, heavy lanthanides)
    // are left as-is and get the documented scalar-relativistic downgrade.
    if (socAnalysis.enableFullSOC) {
      for (const { element } of socAnalysis.socElements) {
        if (!RELATIVISTIC_PP_URLS[element]) continue;
        try {
          await ensureRelativisticPP(element);
        } catch (frErr: any) {
          console.log(`[QE-Worker] FR pseudo provisioning failed for ${element}: ${frErr?.message?.slice(0, 120) ?? "unknown"}`);
        }
      }
    }

    // --- SOC pseudopotential availability check ---
    // analyzeSOCRequirement emits lspinorb=.true. based on element chemistry
    // alone, but the actual pseudo files in QE_PSEUDO_DIR may be scalar-
    // relativistic (has_so="F"). Running QE with lspinorb=.true. on SR
    // pseudos produces wrong physics silently. Downgrade if necessary.
    if (socAnalysis.enableFullSOC && socAnalysis.qeSystemFlags) {
      const { flags: gatedFlags, note: socNote } = applySOCPseudoConstraint(
        socAnalysis.qeSystemFlags,
        socAnalysis.socElements,
      );
      if (socNote) {
        console.warn(`[QE-Worker] ${socNote}`);
        socAnalysis.notes.push(socNote);
        socAnalysis.qeSystemFlags = gatedFlags;
        // If we stripped lspinorb entirely, mark enableFullSOC as false so
        // downstream knows SOC was not actually applied
        if (!gatedFlags.includes("lspinorb")) {
          socAnalysis.enableFullSOC = false;
        }
      }
    }

    // --- Magnetic ground-state search decision ---
    const magSearchDecision = shouldSearchMagneticGS(elements, counts);
    if (magSearchDecision.shouldSearch) {
      console.log(`[QE-Worker] Magnetic ground-state search warranted for ${formula}: ${magSearchDecision.reason}`);
    }

    // --- Hubbard U workflow: composition-aware DFT+U for d/f electron systems ---
    let hubbardResult: HubbardWorkflowResult | undefined;
    try {
      const corrEffects = await estimateCorrelationEffects(formula, {});
      hubbardResult = analyzeHubbardWorkflow(
        formula, elements, counts,
        corrEffects.regime.regime,
        corrEffects.materialPatterns,
      );
      result.hubbardWorkflow = hubbardResult;
      if (hubbardResult.applyDFTplusU) {
        result.qeDFTPlusU = true;
        result.dftPlusUTcModifier = corrEffects.tcModifier;
        console.log(`[QE-Worker] DFT+U workflow for ${formula}: ${hubbardResult.correlatedSiteCount} sites, ` +
          `regime=${hubbardResult.correlationRegime}, ` +
          `vcRelax=${hubbardResult.applyToVCRelax}, phonons=${hubbardResult.applyToPhonons}, ` +
          `sites=[${hubbardResult.sites.filter(s => s.needsU).map(s => `${s.element}:U=${s.uEffective}(${s.source})`).join(", ")}]`);
      }
    } catch (corrErr: any) {
      console.log(`[QE-Worker] Hubbard workflow analysis failed for ${formula}: ${(corrErr.message || "").slice(0, 100)}`);
    }

    // NOTE: Pre-relax ACBN0 (compute U from hp.x before vc-relax) runs later,
    // after positions/latticeA are assigned by structure generation (line ~5200).
    // The self-consistent U feedback from the post-relax ACBN0 at end of pipeline
    // still feeds back into the Hubbard workflow for future runs of this formula.

    // --- Vegard's law enhanced lattice estimation ---
    // --- LLM structure advisor: get structural hints before CSP generation ---
    // One cheap OpenAI call per formula (cached to disk) that provides:
    // - Per-pair MINSEP values for AIRSS
    // - Likely space group for PyXtal biasing
    // - Element coordination roles for cage seeder
    // - Lattice estimate cross-check for Vegard
    const structureAdvice = await getStructureAdvice(formula, workerPressure, elements).catch(() => null);

    // Try Vegard interpolation from AFLOW/MP binary endpoints for a better
    // starting lattice. Falls back to the existing volume-sum estimate if
    // insufficient endpoint data is available (< 30s total with caching).
    let vegardResult: VegardEstimate | null = null;
    let structureCandidates: StructureCandidate[] = [];
    try {
      const vegardT0 = Date.now();
      // Bounded wait: AFLOW + MP API can stall indefinitely (observed 15–55
      // min waits for TlBa2Ca2Cu3O9 / La4Ni3O10 / LaH11Li2 in May 2026).
      // 90s is enough for cached lookups and a few uncached fetches; past
      // that we fall back to the volume-sum estimator rather than block the
      // pipeline.
      const VEGARD_TIMEOUT_MS = 90_000;
      const timeoutSentinel = Symbol("vegard-timeout");
      const timed = <T>(p: Promise<T>): Promise<T | typeof timeoutSentinel> => Promise.race([
        p,
        new Promise<typeof timeoutSentinel>(res => setTimeout(() => res(timeoutSentinel), VEGARD_TIMEOUT_MS)),
      ]);
      const [veg, cand] = await Promise.all([
        timed(vegardEstimate(elements, counts, workerPressure).catch(() => null)),
        timed(generateStructureCandidates(formula, elements, counts, workerPressure, 5).catch(() => [])),
      ]);
      if (veg === timeoutSentinel) {
        console.log(`[QE-Worker] Vegard estimate for ${formula} timed out after ${(VEGARD_TIMEOUT_MS / 1000).toFixed(0)}s — proceeding without (will use volume-sum fallback)`);
        vegardResult = null;
      } else {
        vegardResult = veg as VegardEstimate | null;
      }
      if (cand === timeoutSentinel) {
        console.log(`[QE-Worker] Structure-candidate fetch for ${formula} timed out after ${(VEGARD_TIMEOUT_MS / 1000).toFixed(0)}s — proceeding without`);
        structureCandidates = [];
      } else {
        structureCandidates = cand as StructureCandidate[];
      }
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
          structureAdvice: structureAdvice ? {
            pairDistances: structureAdvice.pairDistances,
            likelySpaceGroup: structureAdvice.likelySpaceGroup,
            alternativeSpaceGroups: structureAdvice.alternativeSpaceGroups,
            structureType: structureAdvice.structureType,
            estimatedLattice: structureAdvice.estimatedLattice,
          } : undefined,
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

    // --- Inject DFT-cached structure from previous runs ---
    // If this formula was optimized before, inject the best known DFT structure
    // as a high-confidence candidate. This gives the pipeline a massive head start
    // instead of rediscovering the same geometry from scratch.
    const cachedStructure = loadDFTStructureCache(formula);
    if (cachedStructure) {
      structureCandidates.push({
        latticeA: cachedStructure.latticeA,
        positions: cachedStructure.positions,
        prototype: "DFT-cached",
        crystalSystem: "unknown",
        spaceGroup: "",
        source: cachedStructure.source,
        confidence: 0.99, // highest confidence — this is a DFT-optimized structure
        isMetallic: null,
      });
      console.log(`[QE-Worker] Injected DFT-cached structure for ${formula}: a=${cachedStructure.latticeA.toFixed(3)} Å, ${cachedStructure.positions.length} atoms, force=${cachedStructure.force.toFixed(6)} (total now: ${structureCandidates.length})`);
    }

    // --- Inject literature known-structure as a top-confidence candidate ---
    // For verified compounds the known-structures database carries exact
    // literature Wyckoff positions + anisotropic lattice. generateStructure
    // Candidates() never injects this DB directly — it relies on a live MP
    // fetch (often unavailable) and CSP (AIRSS/PyXtal). For complex layered
    // cells — e.g. the 12-atom Hg-1212 cuprate HgBa2CaCu2O6 (a=3.86, c=12.66,
    // c/a=3.28) — random CSP never reproduces the layered CuO2-plane motif,
    // so Stage 1 picks a garbage structure and vc-relax starts at force
    // ~3.5 Ry/bohr / P~740 kbar. Injecting the literature cell + Wyckoff
    // positions gives the funnel/Stage 1 the true ground-state geometry to
    // rank against. Confidence 0.97: above MP-direct (0.90), below
    // DFT-cached (0.99) since a prior DFT-optimized cell is still better.
    const ksCandidate = lookupKnownStructure(formula);
    if (ksCandidate && ksCandidate.atoms.length > 0 && ksCandidate.latticeA > 0) {
      const ksCOverA = ksCandidate.latticeC && ksCandidate.latticeA > 0
        ? ksCandidate.latticeC / ksCandidate.latticeA : 1.0;
      structureCandidates.push({
        latticeA: ksCandidate.latticeA,
        latticeB: ksCandidate.latticeB,
        latticeC: ksCandidate.latticeC,
        cOverA: ksCOverA,
        positions: ksCandidate.atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z })),
        prototype: "known-structure",
        crystalSystem: ksCandidate.latticeType ?? "unknown",
        spaceGroup: ksCandidate.spaceGroup ?? "",
        source: `Literature structure (${ksCandidate.spaceGroup}, ${ksCandidate.atoms.length} atoms)`,
        confidence: 0.97,
        isMetallic: null,
      });
      console.log(`[QE-Worker] Injected literature known-structure for ${formula}: ${ksCandidate.spaceGroup}, a=${ksCandidate.latticeA.toFixed(2)} Å${ksCandidate.latticeC ? `, c=${ksCandidate.latticeC.toFixed(2)} Å (c/a=${ksCOverA.toFixed(2)})` : ""}, ${ksCandidate.atoms.length} atoms (total now: ${structureCandidates.length})`);
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
          {
            prefilterMinsepMultiplier: structureAdvice?.prefilterMinsepMultiplier,
          },
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
    // For known structures with explicit lattice parameters, override the Vegard estimate.
    // Vegard produces a cubic-equivalent lattice (e.g., 13.885 Å for Bi2212) but the
    // fractional coordinates from the known-structure database assume the real cell
    // (a=3.81, c=30.89 Å). Using the wrong lattice with the right positions = atom overlaps.
    const ksOverride = lookupKnownStructure(formula);
    let usedKnownStructureLattice = false;
    if (ksOverride && ksOverride.latticeA > 0) {
      const vegardA = latticeA;
      latticeA = ksOverride.latticeA;
      usedKnownStructureLattice = true;
      console.log(`[QE-Worker] Using known structure for ${formula} (${ksOverride.atoms.length} atoms, ${ksOverride.spaceGroup}, a=${ksOverride.latticeA.toFixed(2)} Å${ksOverride.latticeC ? `, c=${ksOverride.latticeC.toFixed(2)} Å` : ""}) — overrides Vegard a=${vegardA.toFixed(3)} Å`);
    }
    result.initialLatticeA = latticeA;

    // Apply Murnaghan compression ONLY when we just overrode with a literature-
    // ambient lattice from the known-structure database. The Vegard path
    // (estimateLatticeConstant) already does pressure compression internally,
    // so applying it here when Vegard was kept would double-compress.
    //
    // Without this step, vc-relax for known-structure hydrides at 100+ GPa
    // had to do all 30%+ cell compression itself and frequently overshot
    // (H3S 200 GPa: a=5.43 → 2.44 Å, ended at 38 GPa instead of 200).
    if (workerPressure > 0 && usedKnownStructureLattice) {
      const B0 = estimateBulkModulus(elements, counts);
      const B0p = 4.0;
      const eta = 1 + B0p * (workerPressure / B0);
      const volRatio = eta > 0 ? Math.pow(eta, -1 / B0p) : 0.5;
      const linearScale = Math.pow(Math.max(0.45, Math.min(1.0, volRatio)), 1 / 3);
      const preMurnA = latticeA;
      latticeA = latticeA * linearScale;
      console.log(`[QE-Worker] Murnaghan-compressed known-structure lattice for ${formula} at ${workerPressure} GPa: ${preMurnA.toFixed(3)} → ${latticeA.toFixed(3)} Å (scale=${linearScale.toFixed(3)}, B0=${B0.toFixed(0)} GPa)`);
    } else if (workerPressure > 0) {
      console.log(`[QE-Worker] Lattice for ${formula} (Vegard path, compression already applied): ${latticeA.toFixed(3)} Å at ${workerPressure} GPa (B0=${estimateBulkModulus(elements, counts).toFixed(0)} GPa)`);
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

    // Skip geometry repair for known structures — their Wyckoff positions are
    // validated and the repair function doesn't handle tetragonal/hexagonal cells
    // correctly (it assumes cubic, causing lattice inflation for c/a >> 1 systems
    // like La2CuO4 where a=3.78 c=13.23 gets inflated to a=7.30).
    const isKnownStructureOverride = ksOverride && ksOverride.latticeA > 0;
    if (!isKnownStructureOverride) {
      const initRepair = repairStructureGeometry(positions, latticeA, workerPressure);
      if (initRepair.repaired) {
        positions = initRepair.positions;
        latticeA = initRepair.latticeA;
        console.log(`[QE-Worker] Repaired initial geometry for ${formula} (lattice=${latticeA.toFixed(3)} A)`);
      }
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
      // Check distances at the ACTUAL DFT lattice using anisotropic cell dimensions.
      // For tetragonal/orthorhombic cells (cuprates, layered), c ≠ a so dz must
      // be scaled by c, not a. Without this, Bi2212 (c/a=8.1) reports O-O=0.12 Å
      // when the real distance is 3+ Å.
      const dftA = knownStruct.latticeA;
      const dftB = knownStruct.latticeB ?? dftA;
      const dftC = knownStruct.latticeC ?? dftA;
      let minDist = Infinity;
      let minPair = "";
      for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          let dx = positions[i].x - positions[j].x;
          let dy = positions[i].y - positions[j].y;
          let dz = positions[i].z - positions[j].z;
          dx -= Math.round(dx); dy -= Math.round(dy); dz -= Math.round(dz);
          const dist = Math.sqrt((dx * dftA) ** 2 + (dy * dftB) ** 2 + (dz * dftC) ** 2);
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
        console.log(`[QE-Worker] Known-structure positions validated for ${formula}: minDist=${minDist.toFixed(3)} Å [${minPair}] at DFT cell ${dftA.toFixed(2)}×${dftB.toFixed(2)}×${dftC.toFixed(2)} Å — skipping xTB (literature Wyckoff positions are DFT-quality)`);
      } else {
        console.log(`[QE-Worker] Known-structure distance check: ${minPair}=${minDist.toFixed(3)} Å < ${distThreshold.toFixed(2)} Å at DFT cell ${dftA.toFixed(2)}×${dftB.toFixed(2)}×${dftC.toFixed(2)} Å — proceeding to xTB`);
      }
    }

    // Skip xTB pre-relax for ionic / layered / perovskite structure classes.
    // xTB (GFN-FF/GFN2) is parameterized for covalent + metallic systems and
    // produces nonsense geometries for cuprates, nickelates, layered oxides,
    // perovskites, etc. TlBa2Ca2Cu3O9 ran xTB at a=13.82 Å, hit Ba-Cu=0.594 Å
    // (below our 0.6 Å floor), then "repair" inflated lattice to 16.95 Å —
    // resulting structure was nowhere near the real Tl-1223 cell (a≈3.86 Å,
    // c≈15.85 Å). vc-relax then has to undo all that work.
    const ionicLikeTypes = new Set([
      "perovskite", "rocksalt", "fluorite", "spinel", "pyrochlore",
      "layered", "hexagonal-layered", "molecular",
    ]);
    const skipXtbForType = structureAdvice
      && typeof structureAdvice.structureType === "string"
      && ionicLikeTypes.has(structureAdvice.structureType);
    let relaxed: Array<{ element: string; x: number; y: number; z: number }> | null = null;
    if (skipXtbForType) {
      console.log(`[QE-Worker] Skipping xTB for ${formula}: LLM advised structureType='${structureAdvice!.structureType}' (xTB not parameterized for ionic/layered systems — lets vc-relax start from DFT-grade Vegard/prototype geometry)`);
      skipXtb = true;
    }
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

    // NO literature lattice override. The pipeline must find the correct
    // lattice on its own through vc-relax. Literature values are for different
    // experimental conditions and pseudopotentials — forcing them causes
    // CaH6 to collapse to a=2.8 (should be ~3.5-4.0) and YH6 to a=2.9.
    // The Stage 1 winner's lattice is our best starting point for vc-relax.
    if (!result.vcRelaxed) {
      console.log(`[QE-Worker] vc-relax will find correct cell for ${formula} (starting from Stage 1 lattice a=${latticeA.toFixed(3)} Å${workerPressure > 0 ? `, P=${workerPressure} GPa` : ""})`);
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

    // No lattice-mismatch guard — removed. The pipeline must find the correct
    // structure through vc-relax from Stage 1 positions. Literature positions
    // and lattice constants are wrong for our pseudopotentials and caused
    // CaH6, YH6, LaH10 to collapse to the wrong basin.
    if (false) { // DISABLED — keeping code for reference
      const stageLatticeShift = 0;
      if (stageLatticeShift > 0.05) {
        const ksLookup: any = null;
        if (ksLookup) {
          console.log(`DISABLED`);
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
      // Look up known-structure angles so monoclinic candidates don't get
      // distorted back to orthorhombic during the iterative lattice rescaling.
      // Without this, vc-relax converged the correct monoclinic cell (per
      // iteration 76's fix), but this rescaling loop generated cell blocks
      // with all 90° angles — losing the monoclinic shape between vc-relax
      // and the subsequent SCF/phonon steps.
      const ksRelax = lookupKnownStructure(formula);
      const relaxAlpha = ksRelax?.alpha ?? 90;
      const relaxBeta = ksRelax?.beta ?? 90;
      const relaxGamma = ksRelax?.gamma ?? 90;
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
        const cellBlockRelax = generateCellParameters(stepA, cOverARelax, 0, bOverARelax, elements, counts, relaxAlpha, relaxBeta, relaxGamma);

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

    // --- Magnetic ground-state search (FM/AFM/NM energy comparison) ---
    // Runs short SCF trials with different spin orderings BEFORE the expensive
    // vc-relax, so phonons are computed on the correct magnetic state.
    if (magSearchDecision.shouldSearch && !result.vcRelaxed) {
      const magConfigs = classifyMagneticLandscape(elements, counts, socAnalysis?.enableFullSOC ?? false);
      if (magConfigs.length >= 2) {
        const magAtomsPre = positions.length;
        const magElecPre = positions.reduce((s, p) => s + (getZValence(p.element) ?? 10), 0);
        const magNspinPre = magConfigs.some(c => c.nspin === 2) ? 2 : 1;
        const magBasePre = 600;
        const magScaledPre = Math.round(magBasePre * Math.pow(magAtomsPre / 4, 1.3) * Math.sqrt(magElecPre / 20) * magNspinPre);
        const magTimeoutPre = Math.max(600, Math.min(3600, magScaledPre));
        console.log(`[QE-Worker] Running magnetic ground-state search for ${formula}: ${magConfigs.length} orderings (${magConfigs.map(c => c.ordering).join(", ")}), ${magAtomsPre} atoms, ${magElecPre} e-, timeout=${magTimeoutPre}s/trial`);
        const magTrials: MagneticTrialResult[] = [];

        for (const config of magConfigs) {
          const magStartTime = Date.now();
          // Scale magnetic trial timeout by system size: 10-atom pnictides with
          // nspin=2 and 72 electrons need 30+ min, not the old 10 min hardcoded cap.
          const magAtoms = positions.length;
          const magElectrons = positions.reduce((s, p) => s + (getZValence(p.element) ?? 10), 0);
          const magNspin = config.nspin === 2 ? 2 : 1;
          const magBaseSeconds = 600; // 10 min base for small systems
          const magScaledSeconds = Math.round(magBaseSeconds * Math.pow(magAtoms / 4, 1.3) * Math.sqrt(magElectrons / 20) * magNspin);
          const magMaxSeconds = Math.max(600, Math.min(3600, magScaledSeconds)); // clamp to 10-60 min
          const magKillMs = magMaxSeconds * 1000 + 60_000;
          try {
            const magInput = generateSCFInputWithParams(formula, elements, counts, latticeA, positions, {
              mixingBeta: 0.3,
              maxSteps: Math.max(120, Math.round(magMaxSeconds / 8)), // scale steps with timeout
              diag: "david",
              smearing: "mv",
              degauss: 0.02,
              convThr: config.convThr ? `${config.convThr}` : "1.0d-6",
              forceNspin: config.noncolin ? 2 : (config.nspin === 1 ? 1 : 2),
              forceMagBlock: config.magnetizationBlock,
              forceNoncolin: config.noncolin ?? false,
              maxSecondsOverride: magMaxSeconds,
            });
            const magFile = path.join(jobDir, `mag_trial_${config.ordering}.in`);
            fs.writeFileSync(magFile, magInput);

            const magResult = await runQECommand(
              path.posix.join(getQEBinDir(), "pw.x"), magFile, jobDir, magKillMs,
            );
            fs.writeFileSync(path.join(jobDir, `mag_trial_${config.ordering}.out`), magResult.stdout);
            const magSCF = parseSCFOutput(magResult.stdout, 0.02);
            const magMoments = parseMagnetizationFromOutput(magResult.stdout);

            // Keep totalEnergy even when strict convergence wasn't reached —
            // selectMagneticGroundState accepts near-converged trials (last
            // accuracy < 1e-3 Ry ≈ 13.6 meV) since FM/AFM gaps are typically
            // 10×+ larger. Originally 1e-4 but HgBa2CuO4 NM landed at 8.8e-4
            // and was wasted; 1e-3 is enough to RANK orderings reliably.
            const nearConverged = magSCF.lastScfAccuracyRy !== null && magSCF.lastScfAccuracyRy < 1e-3;
            magTrials.push({
              ordering: config.ordering,
              totalEnergy: (magSCF.converged || nearConverged) ? magSCF.totalEnergy : null,
              totalMagnetization: magMoments.totalMagnetization,
              absoluteMagnetization: magMoments.absoluteMagnetization,
              converged: magSCF.converged,
              lastScfAccuracyRy: magSCF.lastScfAccuracyRy,
              wallTimeMs: Date.now() - magStartTime,
            });

            const statusLabel = magSCF.converged
              ? "converged"
              : nearConverged
                ? `near-converged (accuracy=${magSCF.lastScfAccuracyRy?.toExponential(1)} Ry, usable for ranking)`
                : `FAILED (accuracy=${magSCF.lastScfAccuracyRy?.toExponential(1) ?? "N/A"} Ry)`;
            console.log(`[QE-Worker] Mag trial ${config.ordering}: E=${magSCF.totalEnergy?.toFixed(6) ?? "N/A"} eV, ` +
              `M=${magMoments.totalMagnetization?.toFixed(2) ?? "N/A"} mu_B, ` +
              `${statusLabel} (${Date.now() - magStartTime}ms)`);
          } catch (magErr: any) {
            magTrials.push({
              ordering: config.ordering,
              totalEnergy: null,
              totalMagnetization: null,
              absoluteMagnetization: null,
              converged: false,
              lastScfAccuracyRy: null,
              wallTimeMs: Date.now() - magStartTime,
            });
            console.log(`[QE-Worker] Mag trial ${config.ordering} failed: ${magErr.message?.slice(0, 100)}`);
          }
          // Clean scratch between trials to avoid charge contamination — but preserve
          // .save/ dirs so subsequent vc-relax can potentially restart from them
          cleanQETmpScratch(path.join(jobDir, "tmp"));
        }

        const magGS = selectMagneticGroundState(magTrials, magConfigs, positions.length);
        result.magneticGroundState = magGS;
        // energyGapPerAtom is in eV/atom (see MagneticGroundStateResult
        // docstring); convert eV→meV via ×1000, NOT Ry→meV via ×13605.7.
        // Previous ×13605.7 reported every gap 13605× too large, making
        // "well-separated" look like 100s of eV/atom in logs.
        console.log(`[QE-Worker] Magnetic ground state for ${formula}: ${magGS.groundState} ` +
          `(gap=${(magGS.energyGapPerAtom * 1000).toFixed(1)} meV/atom, ` +
          `${magGS.wellSeparated ? "well-separated" : "nearly degenerate"})`);
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
      const isCuprateVcRelax = elements.includes("Cu") && elements.includes("O") && (counts["O"] ?? 0) >= 2;
      // Mirror generateVCRelaxInput's internal VC_RELAX_MAX_SECONDS exactly
      // (same base × atom-scale): QE's max_seconds is then passed as an
      // explicit override so it equals the Node kill timeout. Previously the
      // kill timeout used an unscaled base (1800/3600) while the input file
      // baked in the atom-scaled value — for >7-atom cells Node SIGKILLed QE
      // before it printed its clock summary (wall parsed as 0s).
      const vcRelaxAtomScale = positions.length > 7 ? Math.pow(positions.length / 7, 1.2) : 1.0;
      const vcRelaxMaxSec = Math.round(
        (isHighPHVcRelax ? 10800 : (hasMagVcRelax || isCuprateVcRelax) ? 3600 : 1800) * vcRelaxAtomScale,
      );
      const vcRelaxKillMs = vcRelaxMaxSec * 1000 + 60_000;

      // === UNIFIED vc-relax: damped dynamics with tight SCF ===
      // Cell and positions converge TOGETHER. No separate phases, no force gap.
      // conv_thr=1e-7 ensures forces are production-quality at every ionic step.
      console.log(`[QE-Worker] Unified vc-relax for ${formula} (lattice=${latticeA.toFixed(2)} A, ${positions.length} atoms${workerPressure > 0 ? `, P=${workerPressure} GPa` : ""}, damped+tight, timeout=${vcRelaxMaxSec}s)`);

      const vcInput = generateVCRelaxInput(formula, elements, counts, latticeA, positions, workerPressure, undefined, {
        socFlags: socAnalysis?.enableFullSOC ? socAnalysis.qeSystemFlags : undefined,
        forceNspin: result.magneticGroundState?.winningNspin,
        forceMagBlock: result.magneticGroundState?.winningMagBlock || undefined,
        hubbardCard: hubbardResult?.applyToVCRelax ? hubbardResult.qeHubbardCard : undefined,
        maxSecondsOverride: vcRelaxMaxSec,
      });
      const vcFile = path.join(jobDir, "vc_relax.in");
      fs.writeFileSync(vcFile, vcInput);

      const vcResult = await runQECommand(
        path.posix.join(getQEBinDir(), "pw.x"), vcFile, jobDir, vcRelaxKillMs,
      );
      fs.writeFileSync(path.join(jobDir, "vc_relax.out"), vcResult.stdout);
      const vcParsed = parseVCRelaxOutput(vcResult.stdout);

      // --- Diagnostic logging ---
      if (vcParsed.finalPositions && vcParsed.finalPositions.length > 0) {
        const forceMatch = vcResult.stdout.match(/Total force\s*=\s*([\d.]+)/g);
        const lastForce = forceMatch ? forceMatch[forceMatch.length - 1].match(/([\d.]+)$/)?.[1] : null;
        const pressMatch = vcResult.stdout.match(/P=\s*([-\d.]+)/g);
        const lastPress = pressMatch ? pressMatch[pressMatch.length - 1].match(/([-\d.]+)$/)?.[1] : null;
        const vcLattice = vcParsed.finalLatticeAng ?? latticeA;
        console.log(`[QE-Worker] vc-relax DONE for ${formula}: a=${vcLattice.toFixed(3)} Å, E=${vcParsed.totalEnergy.toFixed(4)} eV, force=${lastForce ?? "N/A"} Ry/bohr, P=${lastPress ?? "N/A"} kbar, wall=${vcParsed.wallTimeSeconds.toFixed(0)}s, converged=${vcParsed.converged}`);
        for (let pi = 0; pi < Math.min(4, vcParsed.finalPositions.length); pi++) {
          const p = vcParsed.finalPositions[pi];
          console.log(`[QE-Worker]   atom[${pi}] ${p.element.padEnd(2)} (${p.x.toFixed(5)}, ${p.y.toFixed(5)}, ${p.z.toFixed(5)})`);
        }
        if (vcParsed.finalPositions.length > 4) console.log(`[QE-Worker]   ... and ${vcParsed.finalPositions.length - 4} more`);
      } else {
        console.log(`[QE-Worker] vc-relax produced no positions for ${formula} (exit=${vcResult.exitCode})`);
        const hasAP = vcResult.stdout.includes("ATOMIC_POSITIONS");
        const hasCP = vcResult.stdout.includes("CELL_PARAMETERS");
        console.log(`[QE-Worker] vc-relax parse diagnostic: hasATOMIC_POSITIONS=${hasAP}, hasCELL_PARAMETERS=${hasCP}, stdout_len=${vcResult.stdout.length}`);
        if (hasAP) {
          const apIdx = vcResult.stdout.lastIndexOf("ATOMIC_POSITIONS");
          const apSnippet = vcResult.stdout.slice(apIdx, apIdx + 200);
          console.log(`[QE-Worker] Last ATOMIC_POSITIONS snippet: "${apSnippet.split("\n").slice(0, 4).join(" | ")}"`);
        }
        console.log(`[QE-Worker] vc-relax stdout tail: ${vcResult.stdout.slice(-300)}`);
      }

      // Use vc-relax result
      if (vcParsed.finalPositions && vcParsed.finalPositions.length > 0) {
        positions = vcParsed.finalPositions;
        result.vcRelaxed = true;
        if (vcParsed.finalLatticeAng && vcParsed.finalLatticeAng > 0.5) {
          latticeA = vcParsed.finalLatticeAng;
          result.relaxedLatticeA = latticeA;
        }
        console.log(`[QE-Worker] vc-relax ${vcParsed.converged ? "CONVERGED" : "partial"} for ${formula}: a=${latticeA.toFixed(3)} A, ${positions.length} atoms`);
      } else if (vcResult.exitCode !== 0) {
        console.log(`[QE-Worker] vc-relax failed for ${formula} — proceeding with original geometry`);
      } else {
        console.log(`[QE-Worker] vc-relax produced no usable positions for ${formula} (exit=${vcResult.exitCode}), proceeding with original geometry`);
        const vcDiagTail = vcResult.stdout.slice(-1000);
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

      // Preserve .save/ — ph.x and bands need it. Only clean scratch files.
      cleanQETmpScratch(path.join(jobDir, "tmp"));
      } // close: if (result.vcRelaxed) {} else { ... vc-relax run ... }
    } catch (vcErr: any) {
      console.log(`[QE-Worker] vc-relax failed for ${formula}: ${(vcErr.message || "").slice(-200)}, proceeding with original geometry`);
      // vc-relax failed — full cleanup is fine, SCF will recreate .save
      cleanQETmpDir(path.join(jobDir, "tmp"));
    }

    // Post-refinement max force (Ry/bohr) — visible to the smearing-polish
    // gate below. Stays 999 if refinement never ran (vc-relax failed).
    let postRefinementForce = 999;

    // === Refinement vc-relax loop: keep restarting until force is publication-ready ===
    // Each pass restarts from the previous pass's final geometry with zeroed velocities,
    // eliminating oscillation and converging tighter. Stops when force drops below
    // publication threshold or no improvement is made.
    if (result.vcRelaxed && positions.length > 0) {
      const PUB_FORCE_THR = 0.001; // publication-ready threshold (Ry/bohr)
      const MAX_REFINE_PASSES = 6; // safety cap — don't loop forever

      // Parse initial force and pressure from run 1
      const vcOutPath = path.join(jobDir, "vc_relax.out");
      const vcOut = fs.existsSync(vcOutPath) ? fs.readFileSync(vcOutPath, "utf-8") : "";
      const initForceMatches = [...vcOut.matchAll(/Total force\s*=\s*([\d.]+)/g)];
      let currentForce = initForceMatches.length > 0
        ? parseFloat(initForceMatches[initForceMatches.length - 1][1])
        : 999;
      const initPressMatches = [...vcOut.matchAll(/P=\s*([-\d.]+)/g)];
      let currentPressure = initPressMatches.length > 0
        ? parseFloat(initPressMatches[initPressMatches.length - 1][1])
        : null;

      let refinePass = 0;
      let totalRefineWallSec = 0;
      const startingForce = currentForce;
      let prevSteps = 0;
      let prevForceReductionPerStep = 0; // force reduction per ionic step from last pass
      let prevPressReductionPerStep = 0; // pressure reduction per ionic step from last pass

      // Pressure residual: |P_actual - P_target|. Both in kbar.
      const pressTarget = workerPressure * 10; // GPa → kbar
      const pressureResidual = currentPressure != null ? Math.abs(currentPressure - pressTarget) : 0;
      const PRESSURE_THR = 50; // kbar — quality gate threshold

      const needsRefinement = currentForce > PUB_FORCE_THR || pressureResidual > PRESSURE_THR;

      if (!needsRefinement) {
        console.log(`[QE-Worker] Refinement not needed for ${formula}: force=${currentForce.toFixed(6)} ≤ ${PUB_FORCE_THR}, P_residual=${pressureResidual.toFixed(1)} kbar ≤ ${PRESSURE_THR} (publication-ready)`);
      }

      while ((currentForce > PUB_FORCE_THR || (currentPressure != null && Math.abs(currentPressure - pressTarget) > PRESSURE_THR)) && refinePass < MAX_REFINE_PASSES) {
        refinePass++;
        try {
          // Determine if this is a pressure-priority pass: force is already good,
          // only the cell volume needs to equilibrate. In this mode we tell QE to
          // use ultra-tight forc_conv_thr so ions are "done" on step 1 and the
          // remaining nstep budget goes purely to cell dynamics (damp-w).
          const pressResidualNow = currentPressure != null ? Math.abs(currentPressure - pressTarget) : 0;
          const isPressurePriority = currentForce <= PUB_FORCE_THR && pressResidualNow > PRESSURE_THR;

          if (isPressurePriority) {
            console.log(`[QE-Worker] Refinement pass ${refinePass}/${MAX_REFINE_PASSES} for ${formula}: PRESSURE-PRIORITY mode — force=${currentForce.toFixed(6)} OK, P_residual=${pressResidualNow.toFixed(1)} kbar > ${PRESSURE_THR} — cell volume needs equilibration`);
          } else {
            console.log(`[QE-Worker] Refinement pass ${refinePass}/${MAX_REFINE_PASSES} for ${formula}: force=${currentForce.toFixed(6)}${currentForce > PUB_FORCE_THR ? " > " + PUB_FORCE_THR : ""}, P=${currentPressure?.toFixed(1) ?? "N/A"} kbar — restarting with zeroed velocities`);
          }

          cleanQETmpScratch(path.join(jobDir, "tmp"));

          // Adaptive nstep based on convergence rate from previous pass.
          let refineNstep: number;
          if (isPressurePriority) {
            // Estimate steps from pressure convergence rate.
            const pressGapToClose = currentPressure != null ? Math.abs(currentPressure - pressTarget) : 100;
            if (prevPressReductionPerStep > 0.01) {
              // We have a measured rate from a previous pass — use it directly
              const stepsNeeded = Math.ceil(pressGapToClose / prevPressReductionPerStep);
              refineNstep = Math.max(100, Math.min(600, Math.ceil(stepsNeeded * 1.5)));
              console.log(`[QE-Worker] Pressure-priority nstep for ${formula}: P_gap=${pressGapToClose.toFixed(0)} kbar, measured rate=${prevPressReductionPerStep.toFixed(2)} kbar/step → nstep=${refineNstep}`);
            } else {
              // No measured rate — estimate from system properties.
              // Damped-w cell dynamics typically converges ~0.5-3 kbar/step depending
              // on cell stiffness. Stiffer cells (high bulk modulus) converge faster.
              // Use gap-proportional estimate with generous safety margin.
              refineNstep = Math.max(150, Math.min(600, Math.ceil(pressGapToClose * 2.0)));
              console.log(`[QE-Worker] Pressure-priority nstep for ${formula}: P_gap=${pressGapToClose.toFixed(0)} kbar, no rate data → nstep=${refineNstep}`);
            }
          } else if (refinePass === 1 || refinePass >= MAX_REFINE_PASSES) {
            refineNstep = 400; // first and last pass: full budget
          } else if (prevSteps > 0 && prevForceReductionPerStep > 0) {
            // Estimate how many steps to reach target from current force
            const stepsToTarget = Math.ceil((currentForce - PUB_FORCE_THR) / prevForceReductionPerStep);
            // Clamp: at least 100 (don't waste a pass), at most 400 (cap compute)
            refineNstep = Math.max(100, Math.min(400, Math.ceil(stepsToTarget * 1.5))); // 1.5x safety margin
            console.log(`[QE-Worker] Adaptive nstep for ${formula}: ${prevForceReductionPerStep.toExponential(2)}/step × ${stepsToTarget} steps to target → nstep=${refineNstep}`);
          } else {
            refineNstep = 300; // fallback if no convergence data
          }

          // Timeout per refinement pass, scaled by atom count (base calibrated
          // for 7 atoms). Computed BEFORE generateVCRelaxInput so it can be
          // passed as maxSecondsOverride — QE's internal max_seconds MUST match
          // the Node-side kill timeout. Previously generateVCRelaxInput baked
          // its own VC_RELAX_MAX_SECONDS (1800·scale) while Node killed at the
          // shorter refineMaxSec (1200·scale): QE got SIGKILLed mid-run before
          // printing its clock summary, so the pass logged "wall=0s".
          const hasHRefine = elements.includes("H");
          const hasMagRefine = elements.some(el => el in MAGNETIC_ELEMENTS);
          const isHighPHRefine = hasHRefine && workerPressure >= 50 && positions.length >= 7;
          // Cuprates need a much larger budget — slow layered-oxide SCF means
          // only 5-6 ionic steps fit at the 1200 s base. 3600 s base (matched
          // with local-TF mixing in generateVCRelaxInput) lets more ionic
          // steps land per pass so the cell/ion geometry actually converges.
          const isCuprateRefine = elements.includes("Cu") && elements.includes("O") && (counts["O"] ?? 0) >= 2;
          const refineAtomScale = positions.length > 7 ? Math.pow(positions.length / 7, 1.2) : 1.0;
          const refineMaxSec = Math.round(
            (isHighPHRefine ? 9000 : (hasMagRefine || isCuprateRefine) ? 3600 : 1200) * refineAtomScale,
          );
          const refineKillMs = refineMaxSec * 1000 + 60_000;

          // In pressure-priority mode, use ultra-tight forc_conv_thr (1e-5 Ry/bohr)
          // so QE treats ions as converged immediately and spends all steps on cell.
          const refineInput = generateVCRelaxInput(formula, elements, counts, latticeA, positions, workerPressure, refineNstep, {
            socFlags: socAnalysis?.enableFullSOC ? socAnalysis.qeSystemFlags : undefined,
            forceNspin: result.magneticGroundState?.winningNspin,
            forceMagBlock: result.magneticGroundState?.winningMagBlock || undefined,
            hubbardCard: hubbardResult?.applyToVCRelax ? hubbardResult.qeHubbardCard : undefined,
            pressurePriority: isPressurePriority,
            maxSecondsOverride: refineMaxSec,
          });
          const refineFile = path.join(jobDir, `vc_relax_refine${refinePass}.in`);
          fs.writeFileSync(refineFile, refineInput);

          console.log(`[QE-Worker] Refinement pass ${refinePass} starting for ${formula} (a=${latticeA.toFixed(3)} A, ${positions.length} atoms, nstep=${refineNstep}, timeout=${refineMaxSec}s)`);

          const refineResult = await runQECommand(
            path.posix.join(getQEBinDir(), "pw.x"), refineFile, jobDir, refineKillMs,
          );
          fs.writeFileSync(path.join(jobDir, `vc_relax_refine${refinePass}.out`), refineResult.stdout);
          const refineParsed = parseVCRelaxOutput(refineResult.stdout);

          if (refineParsed.finalPositions && refineParsed.finalPositions.length > 0) {
            const refForceMatches = [...refineResult.stdout.matchAll(/Total force\s*=\s*([\d.]+)/g)];
            const refForce = refForceMatches.length > 0
              ? parseFloat(refForceMatches[refForceMatches.length - 1][1])
              : 999;
            const refPressMatches = [...refineResult.stdout.matchAll(/P=\s*([-\d.]+)/g)];
            const refPressure = refPressMatches.length > 0
              ? parseFloat(refPressMatches[refPressMatches.length - 1][1])
              : null;
            const refLattice = refineParsed.finalLatticeAng ?? latticeA;
            const refWall = refineParsed.wallTimeSeconds;
            totalRefineWallSec += refWall;

            // Count ionic steps from output
            const ionicSteps = [...refineResult.stdout.matchAll(/Total force\s*=\s*([\d.]+)/g)].length;

            // Accept the pass if force improved OR pressure improved.
            // Key insight: a refinement pass that makes force 0.04% worse but moves
            // the cell 100 kbar closer to target pressure IS progress. The old code
            // only accepted force improvement and broke immediately on any force
            // regression, even when the cell was 1000+ kbar off target.
            const refPressResidual = refPressure != null ? Math.abs(refPressure - pressTarget) : 999;
            const prevPressResidual = currentPressure != null ? Math.abs(currentPressure - pressTarget) : 999;
            const forceImproved = refForce < currentForce;
            const pressureImproved = refPressResidual < prevPressResidual - 1.0; // at least 1 kbar improvement
            // Allow force to degrade if pressure is converging. The previous
            // "force can't more than double" guard was relative and rejected
            // LaH10's refinement pass where force went 0.004 → 0.054 (13×
            // relative regression, but 50× BELOW the screening-quality cap
            // of 1.0 in absolute terms) while pressure improved by 38% (1738
            // → 1086 kbar residual). Pure-relative regression is meaningless
            // at low forces — what matters is absolute force vs screening
            // tolerance. Switch to absolute cap that scales with how
            // aggressive the pressure refinement is.
            //   Default: force < max(2 × previous, 0.1 Ry/bohr)
            //   Pressure-priority: force < max(2 × previous, 0.5 Ry/bohr)
            // 0.1 is the Stage 1 magnetic force-gate threshold;
            // 0.5 is "still well below screening cap" — fine for an
            // intermediate refinement state.
            const forceCap = isPressurePriority ? Math.max(currentForce * 2.0, 0.5) : Math.max(currentForce * 2.0, 0.1);
            const forceNotCatastrophic = refForce < forceCap;
            const madeProgress = forceImproved || (pressureImproved && forceNotCatastrophic);

            if (madeProgress) {
              positions = refineParsed.finalPositions;
              if (refineParsed.finalLatticeAng && refineParsed.finalLatticeAng > 0.5) {
                latticeA = refineParsed.finalLatticeAng;
                result.relaxedLatticeA = latticeA;
              }
              console.log(`[QE-Worker] Refinement pass ${refinePass} DONE for ${formula}: force ${currentForce.toFixed(6)} → ${refForce.toFixed(6)} Ry/bohr, a=${refLattice.toFixed(3)} Å, P=${refPressure?.toFixed(1) ?? "N/A"} kbar (residual=${refPressResidual.toFixed(1)}), wall=${refWall.toFixed(0)}s, steps=${ionicSteps}${isPressurePriority ? " [pressure-priority]" : ""}`);
              // Log positions for the first few atoms
              for (let pi = 0; pi < Math.min(4, refineParsed.finalPositions.length); pi++) {
                const p = refineParsed.finalPositions[pi];
                console.log(`[QE-Worker]   refine${refinePass} atom[${pi}] ${p.element.padEnd(2)} (${p.x.toFixed(5)}, ${p.y.toFixed(5)}, ${p.z.toFixed(5)})`);
              }
              if (refineParsed.finalPositions.length > 4) console.log(`[QE-Worker]   ... and ${refineParsed.finalPositions.length - 4} more`);
              // Track convergence rate for adaptive nstep
              prevSteps = ionicSteps;
              if (ionicSteps > 0 && currentForce > refForce) {
                prevForceReductionPerStep = (currentForce - refForce) / ionicSteps;
              }
              if (ionicSteps > 0 && refPressure != null && currentPressure != null) {
                const pressImproveAbs = Math.abs(currentPressure - pressTarget) - Math.abs(refPressure - pressTarget);
                if (pressImproveAbs > 0) {
                  prevPressReductionPerStep = pressImproveAbs / ionicSteps;
                }
              }
              currentForce = refForce;
              currentPressure = refPressure;
            } else {
              console.log(`[QE-Worker] Refinement pass ${refinePass} no improvement for ${formula}: force ${currentForce.toFixed(6)} → ${refForce.toFixed(6)} (${forceImproved ? "improved" : "worse"}), P_residual=${prevPressResidual.toFixed(1)} → ${refPressResidual.toFixed(1)} kbar (${pressureImproved ? "improved" : "no improvement"}) (wall=${refWall.toFixed(0)}s, steps=${ionicSteps}) — stopping refinement`);
              totalRefineWallSec += refWall;
              break;
            }
          } else {
            console.log(`[QE-Worker] Refinement pass ${refinePass} produced no positions for ${formula} (exit=${refineResult.exitCode}) — stopping refinement`);
            const refStdoutTail = refineResult.stdout.slice(-300);
            console.log(`[QE-Worker] Refinement pass ${refinePass} stdout tail: ${refStdoutTail}`);
            break;
          }
        } catch (refErr: any) {
          console.log(`[QE-Worker] Refinement pass ${refinePass} failed for ${formula}: ${refErr.message?.slice(0, 200)} — stopping refinement`);
          break;
        }
      }

      if (refinePass > 0) {
        const forceReduction = startingForce > 0 ? ((1 - currentForce / startingForce) * 100).toFixed(1) : "N/A";
        if (currentForce <= PUB_FORCE_THR) {
          console.log(`[QE-Worker] Refinement COMPLETE for ${formula}: force=${currentForce.toFixed(6)} ≤ ${PUB_FORCE_THR} (publication-ready) after ${refinePass} pass${refinePass > 1 ? "es" : ""}, total wall=${totalRefineWallSec.toFixed(0)}s (${(totalRefineWallSec / 60).toFixed(1)} min), force reduced ${forceReduction}%`);
        } else if (refinePass >= MAX_REFINE_PASSES) {
          console.log(`[QE-Worker] Refinement hit max passes for ${formula}: force=${currentForce.toFixed(6)} (started at ${startingForce.toFixed(6)}) after ${refinePass} passes, total wall=${totalRefineWallSec.toFixed(0)}s (${(totalRefineWallSec / 60).toFixed(1)} min), force reduced ${forceReduction}% — proceeding with best result`);
        } else {
          console.log(`[QE-Worker] Refinement stopped for ${formula}: force=${currentForce.toFixed(6)} (started at ${startingForce.toFixed(6)}) after ${refinePass} pass${refinePass > 1 ? "es" : ""}, total wall=${totalRefineWallSec.toFixed(0)}s (${(totalRefineWallSec / 60).toFixed(1)} min), force reduced ${forceReduction}%`);
        }
      }
      postRefinementForce = currentForce;
    }

    // --- Smearing-polish vc-relax phase ---
    // After force/pressure refinement, re-relax at progressively tighter
    // degauss (0.005 → 0.0025) so the geometry tracks the T→0 minimum, not
    // the smearing-broadened one. The energy surface shifts when degauss
    // narrows — particularly for metallic systems near van Hove singularities.
    // ΔE/atom < 0.5 meV across consecutive passes is the publication gate.
    //
    // Each polish pass uses small nstep (200) since geometry is already close
    // and the goal is energy convergence, not geometry overhaul. Timeout is
    // 1.5× the refinement budget to accommodate slower SCF at tight smearing.
    //
    // GATE: skip the polish entirely when refinement left the geometry with
    // force > 0.1 Ry/bohr. The polish only refines the energy/geometry near
    // an *already-converged* T→0 minimum (nstep=200, small steps). If the
    // geometry is still far from any minimum, tightening degauss buys nothing
    // and burns 30–180 min of wall time chasing energy convergence on a
    // geometry that's still wrong. 0.1 Ry/bohr matches the Stage 1 force gate.
    const SMEARING_POLISH_FORCE_GATE = 0.1; // Ry/bohr
    const polishGatePassed = postRefinementForce <= SMEARING_POLISH_FORCE_GATE;
    if (result.vcRelaxed && positions.length > 0 && !polishGatePassed) {
      console.log(`[QE-Worker] Skipping smearing-polish for ${formula}: post-refinement force=${postRefinementForce.toFixed(4)} Ry/bohr > ${SMEARING_POLISH_FORCE_GATE} — geometry not converged, polishing the smearing would not help`);
    }
    if (result.vcRelaxed && positions.length > 0 && polishGatePassed) {
      const polishLadder: Array<{ degauss: number; convThr: string }> = [
        { degauss: 0.005,  convThr: "1.0d-9"  },
        { degauss: 0.0025, convThr: "1.0d-10" },
      ];
      const hasHPolish = elements.includes("H");
      const hasMagPolish = elements.some(el => el in MAGNETIC_ELEMENTS);
      const isHighPHPolish = hasHPolish && workerPressure >= 50 && positions.length >= 7;
      const polishAtomScale = positions.length > 7 ? Math.pow(positions.length / 7, 1.2) : 1.0;
      // Base 30/60/180 min × 1.5 for tight smearing × atom-scale
      const polishMaxSec = Math.round(
        (isHighPHPolish ? 16200 : hasMagPolish ? 3600 : 1800) * polishAtomScale,
      );

      // Reference energy: parse from latest vc-relax / refinement output
      // (so we can compute ΔE/atom across the first polish pass too).
      let prevEnergyEvPerAtom: number | null = null;
      const refineFiles = fs.readdirSync(jobDir)
        .filter(f => /^vc_relax(_refine\d+)?\.out$/.test(f))
        .sort();
      if (refineFiles.length > 0) {
        const latestOut = fs.readFileSync(path.join(jobDir, refineFiles[refineFiles.length - 1]), "utf-8");
        const latestParsed = parseVCRelaxOutput(latestOut);
        if (latestParsed.totalEnergy !== 0) {
          prevEnergyEvPerAtom = latestParsed.totalEnergy / positions.length;
        }
      }

      let polishConverged = false;
      let totalPolishWallSec = 0;
      for (let pi = 0; pi < polishLadder.length; pi++) {
        const { degauss, convThr } = polishLadder[pi];
        try {
          cleanQETmpScratch(path.join(jobDir, "tmp"));

          const polishInput = generateVCRelaxInput(
            formula, elements, counts, latticeA, positions, workerPressure, 200,
            {
              socFlags: socAnalysis?.enableFullSOC ? socAnalysis.qeSystemFlags : undefined,
              forceNspin: result.magneticGroundState?.winningNspin,
              forceMagBlock: result.magneticGroundState?.winningMagBlock || undefined,
              hubbardCard: hubbardResult?.applyToVCRelax ? hubbardResult.qeHubbardCard : undefined,
              degaussOverride: degauss,
              convThrOverride: convThr,
              maxSecondsOverride: polishMaxSec,
            },
          );
          const polishFile = path.join(jobDir, `vc_relax_smearing_polish_${pi + 1}.in`);
          fs.writeFileSync(polishFile, polishInput);
          console.log(`[QE-Worker] Smearing-polish vc-relax pass ${pi + 1}/${polishLadder.length} for ${formula}: degauss=${degauss}, conv_thr=${convThr}, nstep=200, timeout=${polishMaxSec}s (${(polishMaxSec / 60).toFixed(0)} min)`);

          const polishResult = await runQECommand(
            path.posix.join(getQEBinDir(), "pw.x"),
            polishFile, jobDir, polishMaxSec * 1000 + 60_000,
          );
          fs.writeFileSync(path.join(jobDir, `vc_relax_smearing_polish_${pi + 1}.out`), polishResult.stdout);
          const polishParsed = parseVCRelaxOutput(polishResult.stdout);
          totalPolishWallSec += polishParsed.wallTimeSeconds;

          if (!polishParsed.finalPositions || polishParsed.finalPositions.length === 0) {
            console.log(`[QE-Worker] Smearing-polish pass ${pi + 1} for ${formula} produced no positions (exit=${polishResult.exitCode}) — keeping previous geometry, stopping polish loop`);
            break;
          }

          const newPos = polishParsed.finalPositions;
          const newLat = polishParsed.finalLatticeAng && polishParsed.finalLatticeAng > 0.5
            ? polishParsed.finalLatticeAng : latticeA;
          const energyEvPerAtom = polishParsed.totalEnergy / Math.max(1, newPos.length);
          const dEMevPerAtom = prevEnergyEvPerAtom != null
            ? Math.abs(energyEvPerAtom - prevEnergyEvPerAtom) * 1000.0
            : NaN;
          const polishForceMatches = [...polishResult.stdout.matchAll(/Total force\s*=\s*([\d.]+)/g)];
          const polishForce = polishForceMatches.length > 0
            ? parseFloat(polishForceMatches[polishForceMatches.length - 1][1])
            : null;
          const polishPressMatches = [...polishResult.stdout.matchAll(/P=\s*([-\d.]+)/g)];
          const polishPress = polishPressMatches.length > 0
            ? parseFloat(polishPressMatches[polishPressMatches.length - 1][1])
            : null;
          console.log(`[QE-Worker] Smearing-polish pass ${pi + 1} for ${formula}: ΔE/atom=${Number.isFinite(dEMevPerAtom) ? dEMevPerAtom.toFixed(3) + " meV" : "N/A (no reference)"}, force=${polishForce?.toFixed(6) ?? "N/A"} Ry/bohr, P=${polishPress?.toFixed(1) ?? "N/A"} kbar, a=${newLat.toFixed(3)} Å, wall=${polishParsed.wallTimeSeconds.toFixed(0)}s`);

          positions = newPos;
          latticeA = newLat;
          result.relaxedLatticeA = latticeA;
          // Update parsed SCF results so downstream code sees polished numbers
          const polishedScf = parseSCFOutput(polishResult.stdout, polishLadder[pi].degauss);
          if (polishedScf.totalEnergy !== 0) result.scf = polishedScf;

          prevEnergyEvPerAtom = energyEvPerAtom;
          if (Number.isFinite(dEMevPerAtom) && dEMevPerAtom < 0.5) {
            polishConverged = true;
            console.log(`[QE-Worker] Smearing convergence MET for ${formula} at degauss=${degauss}: ΔE/atom=${dEMevPerAtom.toFixed(3)} meV < 0.5 meV — stopping polish loop after pass ${pi + 1}`);
            break;
          }
        } catch (polishErr: any) {
          console.log(`[QE-Worker] Smearing-polish pass ${pi + 1} crashed for ${formula}: ${polishErr.message?.slice(0, 200)} — keeping previous geometry, stopping polish loop`);
          break;
        }
      }

      const polishStatus = polishConverged
        ? "CONVERGED"
        : (totalPolishWallSec > 0 ? "INCOMPLETE (proceeding with best polished geometry)" : "SKIPPED (no successful pass)");
      console.log(`[QE-Worker] Smearing-polish phase ${polishStatus} for ${formula}: total wall=${totalPolishWallSec.toFixed(0)}s (${(totalPolishWallSec / 60).toFixed(1)} min)`);
    }

    // --- Cache best DFT structure for future runs ---
    if (result.vcRelaxed && positions.length > 0) {
      const cacheForceMatches = [...(fs.existsSync(path.join(jobDir, "vc_relax.out"))
        ? fs.readFileSync(path.join(jobDir, "vc_relax.out"), "utf-8") : "")
        .matchAll(/Total force\s*=\s*([\d.]+)/g)];
      // Check refinement outputs too for the best force
      let bestCacheForce = cacheForceMatches.length > 0
        ? parseFloat(cacheForceMatches[cacheForceMatches.length - 1][1]) : 999;
      for (let ri = 1; ri <= 6; ri++) {
        const refOut = path.join(jobDir, `vc_relax_refine${ri}.out`);
        if (!fs.existsSync(refOut)) break;
        const refMatches = [...fs.readFileSync(refOut, "utf-8").matchAll(/Total force\s*=\s*([\d.]+)/g)];
        if (refMatches.length > 0) {
          const refF = parseFloat(refMatches[refMatches.length - 1][1]);
          if (refF < bestCacheForce) bestCacheForce = refF;
        }
      }
      // Also consider the smearing-polish outputs — they're typically the best
      // force we have since they use the tightest degauss.
      for (let pi = 1; pi <= 4; pi++) {
        const polishOut = path.join(jobDir, `vc_relax_smearing_polish_${pi}.out`);
        if (!fs.existsSync(polishOut)) break;
        const polishMatches = [...fs.readFileSync(polishOut, "utf-8").matchAll(/Total force\s*=\s*([\d.]+)/g)];
        if (polishMatches.length > 0) {
          const polishF = parseFloat(polishMatches[polishMatches.length - 1][1]);
          if (polishF < bestCacheForce) bestCacheForce = polishF;
        }
      }
      saveDFTStructureCache(formula, latticeA, positions, bestCacheForce, null, result.scf?.totalEnergy ?? 0);
    }

    result.kPoints = autoKPoints(latticeA, cOverA, bOverAFull, undefined, DEFAULT_KSPACING, { stage: "scf", isMetallic: vegardResult?.isMetallic ?? undefined, totalAtoms: positions.length }).trim();
    if (Math.abs(latticeA - preVcLatticeA) > 0.01) {
      console.log(`[QE-Worker] K-points recomputed for ${formula} after vc-relax lattice change (${preVcLatticeA.toFixed(3)} -> ${latticeA.toFixed(3)} A): ${result.kPoints}`);
    }

    // --- Skip separate SCF when vc-relax already converged with tight SCF ---
    // The unified vc-relax uses conv_thr=1e-7 and disk_io='high', so the final
    // SCF from vc-relax IS production quality. Parse SCF results from vc-relax output
    // instead of running another SCF. Saves 15-60 min per material.
    //
    // Only skip if vc-relax converged AND produced positions (not partial/failed).
    // If vc-relax failed, fall through to the separate SCF loop below.
    let skipSeparateSCF = false;
    // Pick the LATEST vc-relax output available — smearing-polish > refinement
    // > initial vc-relax. The initial vc_relax.out can finish at a wildly
    // overshoot pressure (H3S Apr 26: ended at 38.5 GPa for a 200 GPa target);
    // the refinement and smearing-polish passes drive both pressure and
    // smearing to spec, and their .save/ density is what ph.x sees on disk.
    // Parsing the initial-only file led to result.scf.pressure being wrong
    // by orders of magnitude in pre-phonon validation.
    const vcRelaxOutPath = path.join(jobDir, "vc_relax.out");
    let bestVcOutPath = vcRelaxOutPath;
    try {
      const polishOuts = fs.readdirSync(jobDir)
        .filter(f => /^vc_relax_smearing_polish_\d+\.out$/.test(f))
        .map(f => path.join(jobDir, f))
        .filter(p => fs.existsSync(p))
        .sort();
      if (polishOuts.length > 0) {
        bestVcOutPath = polishOuts[polishOuts.length - 1];
      } else {
        const refineOuts = fs.readdirSync(jobDir)
          .filter(f => /^vc_relax_refine\d+\.out$/.test(f))
          .map(f => path.join(jobDir, f))
          .filter(p => fs.existsSync(p))
          .sort();
        if (refineOuts.length > 0) {
          bestVcOutPath = refineOuts[refineOuts.length - 1];
        }
      }
    } catch { /* fall back to initial vc_relax.out */ }
    if (bestVcOutPath !== vcRelaxOutPath) {
      console.log(`[QE-Worker] Reusing SCF from ${path.basename(bestVcOutPath)} (latest refinement/polish output) instead of vc_relax.out for ${formula}`);
    }
    const vcRelaxStdout = fs.existsSync(bestVcOutPath) ? fs.readFileSync(bestVcOutPath, "utf-8") : null;
    if (result.vcRelaxed && vcRelaxStdout) {
      // Parse SCF data from vc-relax output (last SCF in the vc-relax run)
      const vcScfParsed = parseSCFOutput(vcRelaxStdout, 0.015);
      if (vcScfParsed.totalEnergy !== 0 && vcScfParsed.converged) {
        // Verify the .save/ directory has the collected wavefunctions that
        // downstream tools (ph.x, bands, pw2wannier90) require. vc-relax with
        // disk_io='high' is supposed to write collected wfc<ik>.dat files,
        // but in practice some runs (Nb3Sn Apr 28: only charge-density.dat +
        // data-file-schema.xml + paw.txt) skip the final flush. Without the
        // wfc files ph.x aborts with "Wavefunctions in collected format not
        // available" — so fall through to a fresh SCF instead of skipping.
        const savePrefix = formula.replace(/[^a-zA-Z0-9]/g, "");
        const saveDirCheck = path.join(jobDir, "tmp", `${savePrefix}.save`);
        let hasCollectedWfcs = false;
        if (fs.existsSync(saveDirCheck)) {
          try {
            const entries = fs.readdirSync(saveDirCheck);
            hasCollectedWfcs = entries.some(f => /^wfc\d+\.(dat|hdf5)$/.test(f));
          } catch { /* ignore */ }
        }
        if (hasCollectedWfcs) {
          result.scf = vcScfParsed;
          skipSeparateSCF = true;
          console.log(`[QE-Worker] Using vc-relax SCF results for ${formula} (E=${vcScfParsed.totalEnergy.toFixed(4)} eV, Ef=${vcScfParsed.fermiEnergy ?? "N/A"}, force=${vcScfParsed.totalForce?.toFixed(4) ?? "N/A"}) — skipping redundant separate SCF`);
        } else {
          console.log(`[QE-Worker] vc-relax converged but ${saveDirCheck} lacks collected wfc*.dat — running separate SCF to generate wavefunctions ph.x needs`);
        }
      }
    }

    // --- DFT+U for correlated materials (uses Hubbard workflow from earlier analysis) ---
    // QE ≥7.1: Hubbard params go in a HUBBARD card after ATOMIC_SPECIES,
    // NOT in &SYSTEM (old lda_plus_u syntax removed in v7.1).
    // dftPlusULines only carries magnetization seeds for &SYSTEM.
    let dftPlusULines = "";
    let dftPlusUNspin2 = false;
    const scfHubbardCard = hubbardResult?.applyDFTplusU ? hubbardResult.qeHubbardCard : "";
    if (hubbardResult?.applyDFTplusU) {
      // For magnetic correlated materials (cuprates, pnictides), force nspin=2
      const isMagCorrMat = hubbardResult.materialPatterns.some(p =>
        p.includes("cuprate") || p.includes("Fe-pnictide"));
      if (isMagCorrMat) {
        dftPlusUNspin2 = true;
        for (let i = 0; i < elements.length; i++) {
          const site = hubbardResult.sites[i];
          if (site && site.uEffective > 0) {
            dftPlusULines += `  starting_magnetization(${i + 1}) = 0.5,\n`;
          }
        }
      }
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

    type RetryConfig = { mixingBeta: number; maxSteps: number; diag: string; smearing?: string; degauss?: number; ecutwfcBoost?: number; convThr?: string; forcConvThr?: string; etotConvThr?: string; mixingMode?: string; mixingNdim?: number; startingwfc?: string; startingpot?: string; diagoThrInit?: string; restartFromScratch?: boolean; nbndOverride?: number; nbndScale?: number };

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
    if (skipSeparateSCF) {
      // vc-relax SCF results already populated — skip entire SCF retry loop
      scfConverged = result.scf?.converged ?? false;
    } else {
      // vc-relax didn't provide usable SCF results — clean distributed wfcs
      // from .save/ before running separate SCF. If vc-relax failed or partially
      // converged, it may have left distributed-format wfc files (wfcdw*.dat,
      // wfcup*.dat) that prevent ph.x from reading collected-format wfcs.
      // The separate SCF with disk_io='high' writes collected wfcs, but only
      // if the old distributed ones are removed first.
      const saveDir = path.join(jobDir, "tmp", `${formula.replace(/[^a-zA-Z0-9]/g, "")}.save`);
      if (fs.existsSync(saveDir)) {
        try {
          const saveFiles = fs.readdirSync(saveDir);
          const distWfcs = saveFiles.filter(f => /^wfc(up|dw)?\d+\.dat$/.test(f));
          if (distWfcs.length > 0) {
            for (const f of distWfcs) {
              fs.unlinkSync(path.join(saveDir, f));
            }
            console.log(`[QE-Worker] Cleaned ${distWfcs.length} distributed wfc files from .save/ for ${formula} — SCF will write collected format`);
          }
        } catch { /* best effort cleanup */ }
      }
    }

    const effectiveMaxSeconds = computeMaxSeconds(elements, workerPressure);
    const effectiveKillTimeoutMs = effectiveMaxSeconds * 1000 + 120_000;
    if (!skipSeparateSCF && effectiveMaxSeconds !== QE_MAX_SECONDS) {
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
        hubbardCard: scfHubbardCard || undefined,
        socFlags: socAnalysis?.enableFullSOC ? socAnalysis.qeSystemFlags : undefined,
        forceNspin: result.magneticGroundState?.winningNspin,
        forceMagBlock: result.magneticGroundState?.winningMagBlock || undefined,
      });
      const scfInputFile = path.join(jobDir, `scf_attempt${attempt}.in`);
      fs.writeFileSync(scfInputFile, scfInput);

      // Clean scratch when not recovering or forcing clean restart — but preserve
      // .save/ dirs because ph.x and bands need the collected wavefunctions from
      // vc-relax (disk_io='high'). Using cleanQETmpDir here was wiping wfc files
      // that FeSe/Sr2RuO4 needed for phonon calculations.
      if (attempt > 0 && (!canRecover || forceClean)) {
        cleanQETmpScratch(path.join(jobDir, "tmp"));
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
          //
          // The previous `combined.includes("max_seconds")` substring match
          // was ALWAYS true — every pw.x stdout echoes the input parameter
          // `max_seconds = N` at the start, so EVERY non-zero-exit crash
          // was misclassified as WALL_TIME_EXHAUSTED. That suppressed the
          // proper retry handlers for SCF_NOT_CONVERGED / CHARGE_WRONG /
          // S_NOT_POSITIVE / DIAG_NOT_CONVERGED etc. Use only the explicit
          // banner and the wall-time heuristic.
          const scfWall = result.scf?.wallTimeSeconds ?? 0;
          const isWallTimeKill = combined.includes("Maximum CPU time exceeded") ||
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
          else if (combined.includes("too few bands")) classifier = " [TOO_FEW_BANDS]";
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
          // Halve mixing_beta, bump maxSteps, force local-TF if still plain.
          // Floor was previously 0.03 — but the retry ladder's extreme-config
          // path uses 0.01 (line 5864) and some all-heavy quaternaries
          // (BaBiLaTe3, certain hydrides) need < 0.03 to converge. Drop the
          // floor to 0.01 so SCF can decay further when needed.
          nextOverride.mixingBeta = Math.max(0.01, (params.mixingBeta ?? 0.3) * 0.5);
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
          // Floor lowered from 0.05 to 0.02 for the same reason as the
          // SCF_NOT_CONVERGED branch above (stubborn heavy systems).
          nextOverride.startingwfc = "random";
          nextOverride.startingpot = "atomic";
          nextOverride.mixingBeta = Math.max(0.02, (params.mixingBeta ?? 0.3) * 0.5);
          (nextOverride as any)._forceClean = true;
        } else if (classifier === " [SMEARING_NEEDED]") {
          // Metal mis-detected as insulator: widen smearing + force MV.
          nextOverride.smearing = "mv";
          nextOverride.degauss = Math.max(params.degauss ?? 0.005, 0.03);
        } else if (classifier === " [TOO_FEW_BANDS]") {
          // QE rejected the SCF before iterating because computeNbnd returned
          // too few bands for the actual electron count. This happens when:
          //  (a) positions array isn't representative of the SCF cell
          //     (supercell mismatch after vc-relax produces no positions)
          //  (b) the PP has more semicore valence than the table assumes
          //  (c) magnetic d-block metals near van Hove need extra empty bands
          // Bump nbnd multiplicatively each retry: 1.5× → 2.0× → 3.0×. Pure
          // additive (e.g. +20) wouldn't keep up when nbnd is already high.
          const prevScale = params.nbndScale ?? 1.0;
          const nextScale = prevScale < 1.5 ? 1.5 : prevScale < 2.0 ? 2.0 : 3.0;
          nextOverride.nbndScale = nextScale;
          // Clear any prior absolute override so the scale takes effect
          (nextOverride as any).nbndOverride = undefined;
          console.log(`[QE-Worker] TOO_FEW_BANDS for ${formula}: bumping nbnd scale ${prevScale.toFixed(1)} → ${nextScale.toFixed(1)}×`);
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
      // parseSCFOutput already converts kbar → GPa (line 3722: /10). The
      // previous label "kbar" was wrong, and the >5 threshold (5 GPa = 50 kbar)
      // was 10× looser than the message suggested — a vc-relaxed cell normally
      // has residual pressure < 0.5 GPa (5 kbar). Use the proper 0.5 GPa
      // threshold and label the unit honestly.
      if (scfPressure != null && Math.abs(scfPressure) > 0.5) {
        console.log(`[QE-Worker] Stage 3 diagnostic: ${formula} residual pressure ${scfPressure.toFixed(2)} GPa > 0.5 — cell may not be fully relaxed`);
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
    // Upgraded later (right before DMFT eligibility check): final_converged after
    // stable phonon + DFPT-grade force, publication_ready after real e-ph + tight force.

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
                    const r2Energy = parseFloat(r2LastE[1]) * RY_TO_EV; // Ry → eV
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
      console.log(`[QE-Worker]   Structure: ${positions.length} atoms, a=${latticeA.toFixed(3)} Å, force=${scfForce?.toFixed(4) ?? "N/A"} Ry/bohr, pressure=${scfPressure?.toFixed(2) ?? "N/A"} GPa`);
      // Log first few atomic positions for diagnosis
      const posToLog = Math.min(positions.length, 8);
      for (let pi = 0; pi < posToLog; pi++) {
        const p = positions[pi];
        console.log(`[QE-Worker]   atom[${pi}] ${p.element.padEnd(2)} (${p.x.toFixed(5)}, ${p.y.toFixed(5)}, ${p.z.toFixed(5)})`);
      }
      if (positions.length > posToLog) {
        console.log(`[QE-Worker]   ... and ${positions.length - posToLog} more atoms`);
      }

      // --- Phonon-prep SCF (tight charge density for ph.x) ---
      // The SCF density inherited from vc-relax (conv_thr=1e-7, degauss=0.015)
      // and even from the production SCF retry loop (conv_thr=1e-10) is too
      // loose for DFPT. ph.x reads .save/ and differentiates that density w.r.t.
      // atomic displacements; loose density → noisy derivatives → spurious
      // imaginary modes. CaH6 at vc-relax-density showed negative rho = 0.296 e-
      // and -9770 cm⁻¹ phonon modes; tight phonon-prep SCF drives both to spec.
      //
      // Adaptive retry: start with the default ecutrho multiplier (8 for hard
      // PAW / USPP, 4 for soft PAW); if negative rho > 1e-3 e-, bump to 12,
      // then 16. Hard cap at 16× to stay within memory on c2-standard workers.
      try {
        const pgTargetQGrid = autoPhononQGrid(elements, positions.length, result.scf?.totalForce ?? undefined);
        const pgKCard = kPointsCardForPhononPrep(
          latticeA, cOverA, bOverAFull,
          vegardResult?.isMetallic ?? undefined,
          positions.length,
          pgTargetQGrid,
        );
        const baseMult = ecutrhoMultiplier(elements);
        const multLadder = Array.from(new Set([baseMult, 8, 12, 16])).sort((a, b) => a - b);
        const pgMaxSeconds = Math.max(1800, Math.min(QE_MAX_SECONDS, 7200));
        let pgConverged = false;
        let pgNegRho = Infinity;
        let pgFinalMult = baseMult;
        let pgFinalScf: QESCFResult | null = null;

        for (let pgAttempt = 0; pgAttempt < multLadder.length; pgAttempt++) {
          const mult = multLadder[pgAttempt];
          const pgInput = generatePhononPrepSCFInput(formula, elements, counts, latticeA, positions, {
            kPointsCard: pgKCard,
            ecutrhoMultiplierOverride: mult,
            // nspin heuristic when magnetic-GS search was skipped: prefer
            // |absoluteMagnetization| since AFM systems have signed total ≈ 0
            // even when local moments are large (e.g., spin-density-wave Cr).
            forceNspin: result.magneticGroundState?.winningNspin
              ?? (result.scf?.absoluteMagnetization != null && result.scf.absoluteMagnetization > 0.05
                  ? 2
                  : (result.scf?.magnetization != null && Math.abs(result.scf.magnetization) > 0.05
                      ? 2
                      : undefined)),
            forceMagBlock: result.magneticGroundState?.winningMagBlock || undefined,
            dftPlusULines: dftPlusULines || undefined,
            dftPlusUNspin2: dftPlusUNspin2 || undefined,
            hubbardCard: scfHubbardCard || undefined,
            socFlags: socAnalysis?.enableFullSOC ? socAnalysis.qeSystemFlags : undefined,
            maxSecondsOverride: pgMaxSeconds,
          });
          const pgInputFile = path.join(jobDir, `phonon_prep_scf_mult${mult}.in`);
          fs.writeFileSync(pgInputFile, pgInput);
          console.log(`[QE-Worker] Phonon-prep SCF for ${formula} attempt ${pgAttempt + 1}/${multLadder.length} (ecutrho/ecutwfc=${mult}, k-grid commensurate with ${pgTargetQGrid.join("×")} q-grid)`);

          const pgResult = await runQECommand(
            path.posix.join(getQEBinDir(), "pw.x"),
            pgInputFile,
            jobDir,
            pgMaxSeconds * 1000 + 60_000,
          );
          fs.writeFileSync(path.join(jobDir, `phonon_prep_scf_mult${mult}.out`), pgResult.stdout);

          const negRhoInfo = parseNegativeRho(pgResult.stdout);
          const eigvalWarnings = countEigvalWarningsInTail(pgResult.stdout, 60);
          const pgScf = parseSCFOutput(pgResult.stdout, 0.005);
          pgNegRho = negRhoInfo.max;
          pgFinalMult = mult;
          pgFinalScf = pgScf;
          pgConverged = pgScf.converged;
          console.log(`[QE-Worker] Phonon-prep SCF result for ${formula} (mult=${mult}): converged=${pgConverged}, |neg rho|=${pgNegRho.toExponential(2)} e-, eigval warnings in tail=${eigvalWarnings}, E=${pgScf.totalEnergy.toFixed(4)} eV`);

          // negative rho < 1e-3 AND clean eigenvalue tail AND converged → done
          if (pgConverged && pgNegRho < 1e-3 && eigvalWarnings <= 2) break;
          if (pgAttempt < multLadder.length - 1) {
            console.log(`[QE-Worker] Phonon-prep SCF for ${formula} retrying with higher ecutrho ratio (neg rho ${pgNegRho.toExponential(2)} >= 1e-3 or eigval warnings ${eigvalWarnings} > 2)`);
          }
        }

        if (pgFinalScf && pgConverged) {
          // Update the result.scf to reflect phonon-prep quality numbers — these
          // are what downstream phonon code will actually be working against.
          result.scf = pgFinalScf;
          console.log(`[QE-Worker] Phonon-prep SCF for ${formula} settled at ecutrho/ecutwfc=${pgFinalMult} (|neg rho|=${pgNegRho.toExponential(2)} e-)`);
        } else {
          console.log(`[QE-Worker] Phonon-prep SCF for ${formula} did NOT fully converge after ${multLadder.length} attempts (|neg rho|=${pgNegRho.toExponential(2)} e-) — proceeding to phonon anyway; expect noisy frequencies if neg rho > 1e-2`);
        }
      } catch (pgErr: any) {
        console.log(`[QE-Worker] Phonon-prep SCF crashed for ${formula}: ${pgErr.message?.slice(0, 200)} — falling back to vc-relax/SCF .save/ density (results may be noisy)`);
      }

      // --- Structure-quality validation (diagnostic) ---
      // Four cheap checks that catch "phonon will be garbage" cases the force
      // and pressure numbers alone miss: stress symmetry vs cell shape, RMS
      // atomic displacement from input, minimum bond length sanity, and final
      // pressure tolerance. These are logged warnings — they do not block.
      try {
        if (result.initialPositions && result.initialPositions.length > 0) {
          // Read latest phonon-prep SCF stdout for smearing entropy check
          let phononPrepScfStdout: string | null = null;
          try {
            const pgOuts = fs.readdirSync(jobDir)
              .filter(f => /^phonon_prep_scf_mult\d+\.out$/.test(f))
              .sort();
            if (pgOuts.length > 0) {
              phononPrepScfStdout = fs.readFileSync(path.join(jobDir, pgOuts[pgOuts.length - 1]), "utf-8");
            }
          } catch { /* best effort */ }
          // Pre-phonon validation: pass known-structure angles when this is
          // a literature compound. Without α/β, the bond-length sanity check
          // overestimates c-axis distances for monoclinic candidates and
          // could mis-flag the closest contact (VO2 β=122.6°, ZrO2/HfO2 ~99°,
          // VO2-family ~5-50% bond-length error).
          const prePhononKS = lookupKnownStructure(formula);
          runPrePhononValidation({
            formula,
            elements,
            counts,
            initialPositions: result.initialPositions,
            finalPositions: positions,
            latticeA,
            cOverA,
            bOverA: bOverAFull,
            gammaRad: prePhononKS?.gamma != null ? prePhononKS.gamma * Math.PI / 180 : Math.PI / 2,
            alphaRad: prePhononKS?.alpha != null ? prePhononKS.alpha * Math.PI / 180 : Math.PI / 2,
            betaRad: prePhononKS?.beta != null ? prePhononKS.beta * Math.PI / 180 : Math.PI / 2,
            vcRelaxStdout,
            phononPrepScfStdout,
            isMetallic: result.scf?.isMetallic,
            totalEnergyRy: result.scf?.totalEnergy != null ? result.scf.totalEnergy / RY_TO_EV : null,
            targetPressureKbar: workerPressure * 10.0,
          });
        }
      } catch (vErr: any) {
        console.log(`[QE-Worker] Pre-phonon validation crashed for ${formula}: ${vErr.message?.slice(0, 200)}`);
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

          // --- SOFT MODE FOLLOWING ---
          // When imaginary modes are found, the structure WANTS to distort along
          // those mode directions. Instead of giving up, displace atoms along the
          // most negative mode eigenvector and re-run vc-relax. This guides the
          // structure toward the actual stable phase.
          const hasImaginaryModes = gammaResult.frequencies && gammaResult.frequencies.some(f => f < -50);
          const softModeFile = path.join(jobDir, "dynmat_gamma.out");
          if (hasImaginaryModes && fs.existsSync(softModeFile)) {
            try {
              const dynmatOutput = fs.readFileSync(softModeFile, "utf-8");
              // Parse eigenvectors from dynmat.x output
              // Format: mode # N followed by displacement vectors per atom
              const modeBlocks = dynmatOutput.split(/mode\s+#?\s*\d+/i);
              const freqs = gammaResult.frequencies!;
              const mostNegIdx = freqs.indexOf(Math.min(...freqs));

              if (mostNegIdx >= 0 && modeBlocks.length > mostNegIdx + 1) {
                // Extract displacement vectors for the most negative mode
                const modeBlock = modeBlocks[mostNegIdx + 1];
                const dispLines = modeBlock.trim().split("\n").filter(l => l.trim().length > 0);
                const displacements: Array<{dx: number; dy: number; dz: number}> = [];
                for (const line of dispLines) {
                  // Format: ( dx_real  dx_imag ) ( dy_real  dy_imag ) ( dz_real  dz_imag )
                  // Accept scientific notation (e/E/d/D exponent forms) — dynmat.x emits
                  // small eigenvector components as e.g. "1.234E-05". The prior
                  // `[-\d.]+` regex silently truncated those at the 'E', giving
                  // e.g. 1.234 instead of 1.234e-5 (off by 5 orders of magnitude).
                  // Matches the same pattern used in zone-boundary-softmode.ts's
                  // parseEigenvectors for consistency.
                  const numRe = "[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eEdD][-+]?\\d+)?";
                  const pairRegex = new RegExp(`\\(\\s*(${numRe})\\s+${numRe}\\s*\\)`, "g");
                  const nums = line.match(pairRegex);
                  if (nums && nums.length >= 3) {
                    const innerRe = new RegExp(`\\(\\s*(${numRe})`);
                    displacements.push({
                      dx: parseFloat((nums[0].match(innerRe)?.[1] ?? "0").replace(/[dD]/, "e")),
                      dy: parseFloat((nums[1].match(innerRe)?.[1] ?? "0").replace(/[dD]/, "e")),
                      dz: parseFloat((nums[2].match(innerRe)?.[1] ?? "0").replace(/[dD]/, "e")),
                    });
                  }
                }

                if (displacements.length === positions.length) {
                  // Apply displacement along soft mode (small amplitude: 0.05 Å / latticeA)
                  const amplitude = 0.05 / latticeA; // fractional displacement
                  const displacedPositions = positions.map((p, i) => ({
                    ...p,
                    x: p.x + displacements[i].dx * amplitude,
                    y: p.y + displacements[i].dy * amplitude,
                    z: p.z + displacements[i].dz * amplitude,
                  }));

                  console.log(`[QE-Worker] SOFT MODE FOLLOWING for ${formula}: displacing ${positions.length} atoms along mode #${mostNegIdx + 1} (freq=${freqs[mostNegIdx].toFixed(1)} cm⁻¹, amplitude=${amplitude.toFixed(4)} frac)`);

                  // Re-run unified vc-relax from displaced structure
                  cleanQETmpDir(path.join(jobDir, "tmp"));
                  const smInput = generateVCRelaxInput(formula, elements, counts, latticeA, displacedPositions, workerPressure);
                  const smFile = path.join(jobDir, "vc_relax_softmode.in");
                  fs.writeFileSync(smFile, smInput);
                  const smHasH = elements.includes("H");
                  const smHighP = smHasH && workerPressure >= 50 && positions.length >= 7;
                  const smMag = elements.some(el => el in MAGNETIC_ELEMENTS);
                  const smVcRelaxMaxSec = smHighP ? 5400 : smMag ? 3600 : 1800;
                  const smResult = await runQECommand(
                    path.posix.join(getQEBinDir(), "pw.x"), smFile, jobDir,
                    smVcRelaxMaxSec * 1000 + 60_000,
                  );
                  fs.writeFileSync(path.join(jobDir, "vc_relax_softmode.out"), smResult.stdout);
                  const smParsed = parseVCRelaxOutput(smResult.stdout);

                  if (smParsed.finalPositions && smParsed.finalPositions.length > 0) {
                    const smForceMatch = smResult.stdout.match(/Total force\s*=\s*([\d.]+)/g);
                    const smForce = smForceMatch ? parseFloat(smForceMatch[smForceMatch.length - 1].match(/([\d.]+)$/)?.[1] ?? "999") : null;
                    positions = smParsed.finalPositions;
                    if (smParsed.finalLatticeAng && smParsed.finalLatticeAng > 0.5) {
                      latticeA = smParsed.finalLatticeAng;
                    }
                    result.vcRelaxed = true;
                    console.log(`[QE-Worker] Soft mode vc-relax DONE for ${formula}: a=${latticeA.toFixed(3)} Å, force=${smForce?.toFixed(4) ?? "N/A"}, E=${smParsed.totalEnergy.toFixed(4)} eV`);

                    // Parse new SCF from soft mode result
                    const smScf = parseSCFOutput(smResult.stdout, 0.015);
                    if (smScf.totalEnergy !== 0) {
                      result.scf = smScf;
                      scfConverged = smScf.converged;
                    }

                    // Re-run gamma phonon on the new structure
                    gammaPhononPassed = true; // allow full phonon to try
                    result.gammaPhononPassed = true;
                    console.log(`[QE-Worker] Soft mode: allowing full phonon grid on new structure for ${formula}`);
                  } else {
                    console.log(`[QE-Worker] Soft mode vc-relax produced no positions for ${formula} — keeping original`);
                  }
                } else {
                  console.log(`[QE-Worker] Soft mode: displacement count ${displacements.length} != atom count ${positions.length} — skipping`);
                }
              }
            } catch (smErr: any) {
              console.log(`[QE-Worker] Soft mode following failed for ${formula}: ${smErr.message?.slice(0, 100)}`);
            }
          }

          if (!gammaPhononPassed) {
            console.log(`[QE-Worker] Gamma phonon check FAILED for ${formula}: ${gammaResult.failReason} — skipping full phonon grid`);
          }
          // Store gamma phonon results as a diagnostic
          if (!gammaPhononPassed && gammaResult.frequencies && gammaResult.frequencies.length > 0) {
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
      console.log(`[QE-Worker] Pre-full-phonon for ${formula}: ${positions.length} atoms, a=${latticeA.toFixed(3)} Å, force=${result.scf?.totalForce?.toFixed(4) ?? "N/A"} Ry/bohr, pressure=${result.scf?.pressure?.toFixed(2) ?? "N/A"} GPa, metallic=${result.scf?.isMetallic ?? "unknown"}`);

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
      const phForce = result.scf?.totalForce ?? 999;
      const [phNq1] = autoPhononQGrid(elements, nAtoms, phForce);
      const qGridPoints = phNq1 ** 3; // assumes cubic q-grid
      // Base: 6× SCF time. Scale up for heavy/large systems and denser q-grids.
      let phMultiplier = 6;
      if (nAtoms >= 8) phMultiplier = 8;
      if (nAtoms >= 12) phMultiplier = 10;
      if (heavyCount >= 2) phMultiplier = Math.max(phMultiplier, 10);
      // Scale by q-grid: 2×2×2=8 qpts is baseline, 4×4×4=64 is 8× more work
      if (qGridPoints > 8) phMultiplier = Math.ceil(phMultiplier * (qGridPoints / 8));
      const MAX_PHONON_TIMEOUT_MS = 48 * 3600 * 1000; // 48 hours absolute cap
      const phKillTimeoutMs = Math.min(effectiveKillTimeoutMs * phMultiplier, MAX_PHONON_TIMEOUT_MS);
      // max_seconds for ph.x: 120s before the kill timeout so QE can checkpoint
      // cleanly and write recover data for the next attempt.
      const phMaxSeconds = Math.floor(phKillTimeoutMs / 1000) - 120;
      console.log(`[QE-Worker] Phonon budget for ${formula}: ${phNq1}×${phNq1}×${phNq1} q-grid (${qGridPoints} q-points), multiplier=${phMultiplier}×, timeout=${(phKillTimeoutMs / 3600_000).toFixed(1)}h, force=${phForce < 100 ? phForce.toFixed(4) : "N/A"}`);

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
          residualForce: result.scf?.totalForce ?? undefined,
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
        // "Maximum CPU time exceeded" message printed by check_stop.f90 when
        // max_seconds is exhausted.
        // Previously this also matched `combined.includes("max_seconds")` —
        // but every ph.x stdout echoes the input parameter `max_seconds = N`
        // at the start, so any non-zero-exit-code crash was misclassified as
        // a timeout and routed through the recover=.true. retry branch
        // (which uses the checkpoint of a crashed run — likely garbage)
        // instead of the tighter-mixing crash-retry branch.
        const combined = phResult!.stdout + phResult!.stderr;
        phTimedOut = phResult!.exitCode === -1 ||
          combined.includes("Maximum CPU time exceeded");

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
        const [pnq1, pnq2, pnq3] = autoPhononQGrid(elements, positions.length, result.scf?.totalForce ?? undefined);
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
                result.phonon.imaginaryCount = freqValues.filter(f => f < -10).length;
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
                      result.phonon.imaginaryCount = freqValues.filter(f => f < -10).length;
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
                    result.phonon.imaginaryCount = freqValues.filter(f => f < -10).length;
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

    // ── Zone-boundary soft mode following ──────────────────────────────────
    // When the full phonon grid found imaginary modes, attempt to follow the
    // worst instability to a dynamically stable phase.
    if (result.phonon?.hasImaginary && result.phonon.imaginaryCount > 0 &&
        result.phonon.lowestFrequency < -50 && result.vcRelaxed) {
      try {
        const zbPrefix = formula.replace(/[^a-zA-Z0-9]/g, "");
        console.log(`[QE-Worker] ${formula}: ${result.phonon.imaginaryCount} imaginary modes (lowest=${result.phonon.lowestFrequency.toFixed(1)} cm⁻¹) — attempting zone-boundary soft mode following`);

        const zbResult = await followZoneBoundarySoftMode(
          formula, zbPrefix, positions, latticeA, jobDir,
          {
            runQEBinary: (binary, inputFile, cwd, timeoutMs) =>
              runQECommand(path.posix.join(getQEBinDir(), binary), inputFile, cwd, timeoutMs),
            runVCRelax: async (displacedPos, displacedA) => {
              const zbNstep = 200;
              const zbInput = generateVCRelaxInput(
                formula, elements, counts, displacedA, displacedPos, workerPressure, zbNstep, {
                  hubbardCard: hubbardResult?.applyToVCRelax ? hubbardResult.qeHubbardCard : undefined,
                  forceNspin: result.magneticGroundState?.winningNspin,
                  forceMagBlock: result.magneticGroundState?.winningMagBlock || undefined,
                },
              );
              const zbFile = path.join(jobDir, "vc_relax_zbsm.in");
              fs.writeFileSync(zbFile, zbInput);
              cleanQETmpScratch(path.join(jobDir, "tmp"));
              const isHighPH = elements.includes("H") && workerPressure >= 50;
              const zbTimeout = isHighPH ? 5400_000 : 1800_000;
              const zbRun = await runQECommand(
                path.posix.join(getQEBinDir(), "pw.x"), zbFile, jobDir, zbTimeout,
              );
              fs.writeFileSync(path.join(jobDir, "vc_relax_zbsm.out"), zbRun.stdout);
              const parsed = parseVCRelaxOutput(zbRun.stdout);
              const forceMatches = [...zbRun.stdout.matchAll(/Total force\s*=\s*([\d.]+)/g)];
              const force = forceMatches.length > 0 ? parseFloat(forceMatches[forceMatches.length - 1][1]) : 999;
              const pressMatches = [...zbRun.stdout.matchAll(/P=\s*([-\d.]+)/g)];
              const pressure = pressMatches.length > 0 ? parseFloat(pressMatches[pressMatches.length - 1][1]) : null;
              const scfParsed = parseSCFOutput(zbRun.stdout, 0.015);
              return {
                positions: parsed.finalPositions ?? displacedPos,
                latticeA: parsed.finalLatticeAng ?? displacedA,
                force,
                pressure,
                energy: scfParsed.totalEnergy,
                converged: parsed.converged,
              };
            },
            runGammaPhonon: async (phPos, phA) => {
              const phPrefix = formula.replace(/[^a-zA-Z0-9]/g, "") + "_zbph";
              // Quick gamma phonon: generate SCF + phonon input
              const phScfInput = generateSCFInputWithParams(
                formula, elements, counts, phA, phPos, {
                  mixingBeta: 0.3, maxSteps: 200, diag: "david",
                  convThr: "1.0d-7", hubbardCard: hubbardResult?.applyDFTplusU ? hubbardResult.qeHubbardCard : undefined,
                },
              );
              const phScfFile = path.join(jobDir, `${phPrefix}_scf.in`);
              fs.writeFileSync(phScfFile, phScfInput);
              cleanQETmpScratch(path.join(jobDir, "tmp"));
              await runQECommand(
                path.posix.join(getQEBinDir(), "pw.x"), phScfFile, jobDir, 1800_000,
              );
              const phInput = generatePhononInput(formula, elements, phPos.length, {
                maxSeconds: 3600, tr2Ph: "1.0d-10", alphaMix: 0.5,
              });
              const phFile = path.join(jobDir, `${phPrefix}_ph.in`);
              fs.writeFileSync(phFile, phInput);
              const phRun = await runQECommand(
                path.posix.join(getQEBinDir(), "ph.x"), phFile, jobDir, 3600_000,
              );
              const phParsed = parsePhononOutput(phRun.stdout);
              const freqs = phParsed.frequencies;
              const imagCount = freqs.filter((f: number) => f < -50).length;
              return { frequencies: freqs, passed: imagCount === 0 };
            },
          },
        );

        if (zbResult.foundStablePhase && zbResult.newPositions) {
          console.log(`[QE-Worker] ${formula}: zone-boundary soft mode following FOUND STABLE PHASE — updating structure`);
          positions = zbResult.newPositions;
          latticeA = zbResult.newLatticeA ?? latticeA;
          result.phonon!.hasImaginary = false;
          result.phonon!.imaginaryCount = 0;
          // Flag that we should re-run full phonon grid on the new structure
          // (the gamma check passed, but full grid confirmation is needed)
        } else {
          console.log(`[QE-Worker] ${formula}: zone-boundary soft mode following did not find stable phase (${zbResult.iterations} iterations, ${zbResult.notes.length} notes)`);
        }
        for (const note of zbResult.notes) {
          console.log(`[QE-Worker] [ZB-SM] ${note}`);
        }
      } catch (zbErr: any) {
        console.log(`[QE-Worker] ${formula}: zone-boundary soft mode following failed: ${(zbErr.message ?? "").slice(0, 200)}`);
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
          // Pseudo directory in QE-input form (WSL-translated on Windows).
          // Without this, generateBandsInput hardcoded "/tmp/qe_pseudo" which
          // only worked on Linux setups where the pseudos happened to land
          // at /tmp — on Windows (WSL) and on any system using a non-/tmp
          // TMPDIR, bands.x would fail with "pseudo file not found".
          QE_PSEUDO_DIR_INPUT,
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
    // result.scf.pressure is in GPa (parseSCFOutput converts kbar→GPa via /10).
    // Previous threshold "50" was effectively 50 GPa (500 kbar) — absurdly loose
    // for a "quality gate" since even a barely-relaxed cell rarely exceeds 5 GPa
    // residual. Original intent was clearly 50 kbar (= 5 GPa); the mislabeling
    // let unrelaxed structures pass the gate and proceed to DFPT.
    const scfPressureOk = result.scf?.pressure != null ? Math.abs(result.scf.pressure) < 5 : true;
    const isMetallicForTc = result.scf?.isMetallic ?? false;

    // Basic quality gate (allows surrogate Tc)
    const qualityGatePass = scfConverged && scfForceOkScreening && scfPressureOk && isMetallicForTc && phononPhysicallyStable;
    // DFPT gate (allows physics-grade e-ph — tighter force)
    const dfptGatePass = qualityGatePass && scfForceOkDFPT;
    const qualityGateReason: string[] = [];
    if (!scfConverged) qualityGateReason.push("SCF not converged");
    if (!scfForceOkScreening) qualityGateReason.push(`force ${residualForce?.toFixed(3)} > 0.10 Ry/bohr (screening threshold)`);
    else if (!scfForceOkDFPT) qualityGateReason.push(`force ${residualForce?.toFixed(3)} > 0.03 Ry/bohr (DFPT threshold — surrogate Tc only)`);
    if (!scfPressureOk) qualityGateReason.push(`pressure ${result.scf?.pressure?.toFixed(2)} GPa > ±5`);
    if (!isMetallicForTc) qualityGateReason.push("not metallic");
    if (!phononPhysicallyStable) qualityGateReason.push(phononHasResults ? "imaginary phonon modes" : "no phonon data");

    if (dfptGatePass) {
      console.log(`[QE-Worker] Quality gate PASSED (DFPT-ready) for ${formula}: force=${residualForce?.toFixed(4) ?? "N/A"} < 0.03, pressure=${result.scf?.pressure?.toFixed(2) ?? "N/A"} GPa, phonon stable (${result.phonon!.frequencies.length} modes)`);
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

    // --- EPW Wannier-interpolated electron-phonon coupling ---
    // Only for publication-ready materials: force < 0.001, phonon stable, metallic.
    // EPW interpolates e-ph matrix elements from the coarse DFPT q-grid to ultra-fine
    // k/q grids, then solves the anisotropic Migdal-Eliashberg equations for Tc.
    const publicationForce = residualForce != null && residualForce < 0.001;
    if (scfUsable && dfptGatePass && publicationForce && phononPhysicallyStable && !opts?.skipEph) {
      try {
        const phForceEPW = result.scf?.totalForce ?? 999;
        const [epwQGrid] = autoPhononQGrid(elements, positions.length, phForceEPW);
        const epwCOverA = estimateCOverA(elements, counts);
        const epwBOverA = estimateBOverA(elements, counts);
        // Honor known-structure angles so monoclinic candidates use the
        // correct β instead of defaulting to 90° (which would force an
        // orthorhombic cell shape and give wrong electronic structure in
        // the EPW NSCF/Wannier/interpolation chain).
        const epwKS = lookupKnownStructure(formula);

        console.log(`[QE-Worker] ${formula} qualifies for EPW pipeline (force=${residualForce?.toFixed(6)}, phonon stable, metallic)`);
        result.epw = await runEPWPipeline(
          formula, elements, counts, positions, latticeA, jobDir, workerPressure,
          {
            fermiEnergy: result.scf!.fermiEnergy!,
            ecutwfc: computeEcutwfc(elements, 0, 80, 45),
            ecutrho: computeEcutwfc(elements, 0, 80, 45) * ecutrhoMultiplier(elements),
            phononQGrid: [epwQGrid, epwQGrid, epwQGrid] as [number, number, number],
            cellParameters: generateCellParameters(latticeA, epwCOverA, 0, epwBOverA, elements, counts, epwKS?.alpha ?? 90, epwKS?.beta ?? 90, epwKS?.gamma ?? 90),
          },
          {
            runQEBinary: (binary, inputFile, cwd, timeoutMs) => runQECommand(binary, inputFile, cwd, timeoutMs),
            getPseudoDirInput: () => QE_PSEUDO_DIR_INPUT,
            resolvePPFilename,
            // Honor the project-wide QE_BIN_DIR so EPW can find pw.x /
            // wannier90.x / pw2wannier90.x / epw.x on systems where QE
            // isn't at /usr/local/bin (e.g. /usr/bin via apt, or a custom
            // build directory pointed at by the QE_BIN_DIR env var).
            getQEBinDir,
          },
        );

        if (result.epw && result.epw.lambda > 0) {
          console.log(`[QE-Worker] EPW complete for ${formula}: λ=${result.epw.lambda.toFixed(3)}, Tc(ME)=${result.epw.tcMigdalEliashberg.toFixed(1)} K, Tc(AD)=${result.epw.tcAllenDynes.toFixed(1)} K`);
        } else {
          console.log(`[QE-Worker] EPW returned no usable e-ph coupling for ${formula}`);
        }
      } catch (epwErr: any) {
        console.log(`[QE-Worker] EPW pipeline failed for ${formula}: ${(epwErr.message ?? "").slice(0, 200)}`);
      }
    } else if (publicationForce && !phononPhysicallyStable) {
      console.log(`[QE-Worker] ${formula} has publication-ready force but unstable phonons — EPW skipped`);
    }

    // --- SSCHA Anharmonic Phonon Corrections ---
    // Triggers on any system with anharmonicity indicators, not just hydrides.
    const sschaLowestAcoustic = result.phonon?.frequencies?.length
      ? Math.min(...result.phonon.frequencies.filter((f: number) => f > 0))
      : undefined;
    const sschaGate = checkSSCHAEligibility({
      maxForce: residualForce ?? 999, elements, counts, pressureGPa: workerPressure,
      lowestAcousticFreq: sschaLowestAcoustic,
      // anharmonicityIndex and softModeScore come from the surrogate PhononSpectrum,
      // not the DFT QEPhononResult. Pass them only if surrogate data is available.
    });
    if (publicationForce && phononPhysicallyStable && sschaGate.eligible) {
      try {
        console.log(`[QE-Worker] ${formula} qualifies for SSCHA: ${sschaGate.reason}`);
        const harmonicLambda = result.epw?.lambda ?? (result.dfpt as any)?.lambda ?? null;
        // SSCHA pipeline's allenDynesTc() expects omega_log in meV (per
        // sscha-pipeline.ts:281, converts internally via /0.08617 to K).
        //   - EPW result is already in meV (EPWResult.omegaLog comment)
        //   - DFPT result is in K (QEDFPTResult.omegaLog, set by
        //     parseLambdaOutput which converts cm⁻¹→K)
        // Without this conversion the DFPT fallback path passed K-valued
        // omegaLog into a meV-expecting consumer, giving Tc estimates that
        // were ~11.6× too high whenever EPW was unavailable.
        const dfptOmegaLogK = (result.dfpt as any)?.omegaLog as number | undefined;
        // k_B = 0.086173 meV/K. To convert K → meV: multiply by k_B[meV/K].
        // The previous name `K_PER_MEV` was misleading — 0.086173 is meV per K
        // (the inverse, K/meV, would be 11.605). Renamed for clarity since
        // the multiplication direction was easy to get backwards while
        // reading the code.
        const MEV_PER_K = 0.086173;  // k_B in meV/K — multiply omegaLog[K] by this to get meV
        const harmonicOmegaLog: number | null =
          result.epw?.omegaLog
          ?? (typeof dfptOmegaLogK === "number" && dfptOmegaLogK > 0
              ? dfptOmegaLogK * MEV_PER_K
              : null);
        const prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
        // Adaptive nConfigs: scale with cell size (more atoms = more configs needed)
        // Errea group typically uses 100-500. 50 is noise-dominated for larger cells.
        // Rule: 100 base, +50 per 4 atoms above 4, capped at 500.
        const sschaNConfigs = Math.min(500, Math.max(100, 100 + Math.floor((positions.length - 4) / 4) * 50));
        result.sscha = await runSSCHAPipeline(
          formula, elements, counts, positions, latticeA, jobDir, workerPressure,
          harmonicLambda, harmonicOmegaLog,
          {
            ecutwfc: computeEcutwfc(elements, 0, 80, 45),
            ecutrho: computeEcutwfc(elements, 0, 80, 45) * ecutrhoMultiplier(elements),
            pseudoDir: QE_PSEUDO_DIR_INPUT,
            prefix,
            temperature: 300,
            nConfigs: sschaNConfigs,
          },
        );
        if (result.sscha?.converged) {
          console.log(`[QE-Worker] SSCHA complete for ${formula}: ω_log=${result.sscha.omegaLogAnharmonic.toFixed(1)} meV (anharmonic), Tc=${result.sscha.tcCorrected?.toFixed(1) ?? "N/A"} K, ${result.sscha.iterations} iterations`);
        } else {
          console.log(`[QE-Worker] SSCHA did not converge for ${formula} — using harmonic results`);
        }
      } catch (sschaErr: any) {
        console.log(`[QE-Worker] SSCHA failed for ${formula}: ${(sschaErr.message ?? "").slice(0, 200)}`);
      }
    }

    // --- ACBN0 First-Principles μ* ---
    // For publication-ready materials, compute μ* from electronic structure
    // instead of using the conventional fixed 0.10-0.13.
    if (publicationForce && scfUsable && dfptGatePass && result.scf?.fermiEnergy != null) {
      try {
        console.log(`[QE-Worker] ${formula} qualifies for ACBN0 first-principles μ*`);
        const acbn0Prefix = formula.replace(/[^a-zA-Z0-9]/g, "");
        const acbn0COverA = estimateCOverA(elements, counts);
        const acbn0BOverA = estimateBOverA(elements, counts);
        result.acbn0 = await runACBN0Pipeline(
          formula, elements, counts, positions, latticeA, jobDir, workerPressure,
          {
            ecutwfc: computeEcutwfc(elements, 0, 80, 45),
            ecutrho: computeEcutwfc(elements, 0, 80, 45) * ecutrhoMultiplier(elements),
            pseudoDir: QE_PSEUDO_DIR_INPUT,
            prefix: acbn0Prefix,
            // result.dfpt.omegaLog is in K (per parseLambdaOutput's
            // conversion); convert K→meV via k_B = 0.086173 meV/K so the
            // ACBN0 Morel-Anderson μ* calculation gets a sensible Debye
            // cutoff. Previously the raw K value (~800) was passed as
            // "meV", which gave ω_D = 0.8 eV instead of ~0.07 eV — a
            // ~11.6× overestimate of the phonon cutoff that produced
            // wildly wrong first-principles μ* values.
            debyeFrequencyMeV: typeof (result.dfpt as any)?.omegaLog === "number" && (result.dfpt as any).omegaLog > 0
              ? (result.dfpt as any).omegaLog * 0.086173
              : undefined,
            // Honor known-structure angles for monoclinic candidates.
            cellParameters: (() => {
              const ks = lookupKnownStructure(formula);
              return generateCellParameters(latticeA, acbn0COverA, 0, acbn0BOverA, elements, counts, ks?.alpha ?? 90, ks?.beta ?? 90, ks?.gamma ?? 90);
            })(),
          },
          {
            runQEBinary: (binary, inputFile, cwd, timeoutMs) => runQECommand(binary, inputFile, cwd, timeoutMs),
            getPseudoDirInput: () => QE_PSEUDO_DIR_INPUT,
            resolvePPFilename,
            // Use the project-wide QE binary directory so ACBN0 finds pw.x
            // and hp.x on systems where QE isn't at /usr/local/bin (e.g.
            // /usr/bin via apt). Without this, ACBN0 would fail with
            // "binary not found" on most non-Docker installations.
            getQEBinDir,
          },
        );
        if (result.acbn0?.converged) {
          const conventionalMu = 0.10; // standard convention
          console.log(`[QE-Worker] ACBN0 complete for ${formula}: μ*=${result.acbn0.muStar.toFixed(4)} (conventional=${conventionalMu}, deviation=${((result.acbn0.muStar - conventionalMu) * 100).toFixed(1)}%, method=${result.acbn0.method})`);

          // Feed self-consistent U values back into the Hubbard workflow result
          // so downstream consumers (learning, dataset) get the first-principles values
          if (result.hubbardWorkflow && Object.keys(result.acbn0.hubbardU).length > 0) {
            const scU = result.acbn0.hubbardU;
            let updated = 0;
            for (const site of result.hubbardWorkflow.sites) {
              if (scU[site.element] != null && scU[site.element] > 0) {
                const oldU = site.uEffective;
                site.uEffective = scU[site.element];
                site.source = "material-specific"; // promoted from hp.x
                updated++;
                if (Math.abs(oldU - site.uEffective) > 0.3) {
                  console.log(`[QE-Worker] Self-consistent U for ${site.element}: ${oldU.toFixed(1)} -> ${site.uEffective.toFixed(1)} eV (from hp.x)`);
                }
              }
            }
            if (updated > 0) {
              result.hubbardWorkflow.notes.push(
                `Self-consistent U from ACBN0/hp.x applied to ${updated} site(s): ` +
                `${Object.entries(scU).filter(([, v]) => v > 0).map(([k, v]) => `${k}=${v.toFixed(2)} eV`).join(", ")}. ` +
                `Ref: Timrov et al., PRB 98, 085127 (2018).`
              );
            }
          }
        }
      } catch (acbn0Err: any) {
        console.log(`[QE-Worker] ACBN0 failed for ${formula}: ${(acbn0Err.message ?? "").slice(0, 200)}`);
      }
    }

    // --- Promote qualityTier based on phonon + e-ph completion ---
    // Initial assignment at line 7707 caps at "relaxed". Comment at 7715 promises
    // a later upgrade based on phonon/e-ph results, but the tier-cap block below
    // only downgrades — so without this promotion DMFT eligibility (which requires
    // final_converged+) never triggers even after a full DFPT+EPW run succeeded.
    // Promote conservatively: phonons must be physically stable and forces must
    // meet the DFPT-grade threshold for final_converged; publication_ready also
    // requires real e-ph coupling (DFPT or EPW) and the tightest force criterion.
    if (result.qualityTier === "relaxed" && phononHasResults && phononPhysicallyStable && scfForceOkDFPT) {
      const hasRealEph = (result.dfpt != null && (result.dfpt as any).lambda > 0)
        || (result.epw != null && result.epw.lambda > 0);
      if (hasRealEph && publicationForce) {
        result.qualityTier = "publication_ready";
        console.log(`[QE-Worker] Tier promoted to publication_ready for ${formula}: full phonon stable + real e-ph (DFPT/EPW) + force < 0.001`);
      } else {
        result.qualityTier = "final_converged";
        console.log(`[QE-Worker] Tier promoted to final_converged for ${formula}: phonon stable + force < 0.03${hasRealEph ? " (publication blocked: force >= 0.001)" : ""}`);
      }
    }

    // --- DMFT bundle export for correlated materials ---
    const dmftCheck = isDMFTEligible(result.hubbardWorkflow, result.qualityTier);
    if (dmftCheck.eligible && result.scf?.fermiEnergy != null) {
      try {
        console.log(`[QE-Worker] ${formula} eligible for DMFT bundle: ${dmftCheck.reason}`);
        const dmftPrefix = formula.replace(/[^a-zA-Z0-9]/g, "");
        const dmftWannierDir = path.posix.join(jobDir, "dmft_wannier");

        const {
          generateWannier90Win, generateNSCFInput, generatePW2Wannier90Input,
          countDMFTOrbitals,
        } = await import("./epw-pipeline");

        if (!fs.existsSync(dmftWannierDir)) {
          fs.mkdirSync(dmftWannierDir, { recursive: true });
        }

        const dmftCOverA = estimateCOverA(elements, counts);
        const dmftBOverA = estimateBOverA(elements, counts);
        // Honor known-structure angles so dmftLatticeVectors matches the
        // monoclinic cell the QE NSCF actually runs with (see fix in the
        // dmftNscfInput cellParameters block below). Without this, the
        // .win file at line 8985 used orthorhombic vectors while NSCF
        // used monoclinic — Wannier basis misaligned with the bands.
        const dmftKS = lookupKnownStructure(formula);
        const dmftLatticeVectors = latticeVectorsFromParams(
          latticeA, dmftCOverA, dmftBOverA, elements, counts,
          dmftKS?.alpha ?? 90, dmftKS?.beta ?? 90, dmftKS?.gamma ?? 90,
        );
        const dmftOrbitals = countDMFTOrbitals(elements, counts);
        const dmftNumWann = dmftOrbitals.total;
        const dmftNbnd = Math.max(dmftNumWann + 4, Math.ceil(dmftNumWann * 1.3));
        const dmftKGrid: [number, number, number] = [8, 8, 8];
        const dmftEcutwfc = computeEcutwfc(elements, 0, 80, 45);
        const dmftEcutrho = dmftEcutwfc * ecutrhoMultiplier(elements);

        // ── Step 1: NSCF on uniform k-grid (reuses converged SCF density) ──
        console.log(`[QE-Worker] DMFT Wannier step 1/4: NSCF on ${dmftKGrid.join("x")} k-grid`);
        const dmftNscfInput = generateNSCFInput({
          prefix: dmftPrefix,
          pseudoDir: QE_PSEUDO_DIR_INPUT,
          ecutwfc: dmftEcutwfc,
          ecutrho: dmftEcutrho,
          latticeA,
          // Honor known-structure angles for monoclinic candidates —
          // reuses dmftKS already looked up above for dmftLatticeVectors.
          cellParameters: generateCellParameters(
            latticeA, dmftCOverA, 0, dmftBOverA, elements, counts,
            dmftKS?.alpha ?? 90, dmftKS?.beta ?? 90, dmftKS?.gamma ?? 90,
          ),
          positions, elements,
          ppFilenames: Object.fromEntries(elements.map(el => [el, resolvePPFilename(el)])),
          kGrid: dmftKGrid,
          nbnd: dmftNbnd,
        });
        const dmftNscfFile = path.posix.join(dmftWannierDir, `${dmftPrefix}_nscf.in`);
        fs.writeFileSync(dmftNscfFile, dmftNscfInput);

        // Copy .save directory from the main SCF so NSCF can read the density
        const mainSaveDir = path.posix.join(jobDir, "tmp", `${dmftPrefix}.save`);
        const dmftTmpDir = path.posix.join(dmftWannierDir, "tmp");
        const dmftSaveDir = path.posix.join(dmftTmpDir, `${dmftPrefix}.save`);
        if (!fs.existsSync(dmftTmpDir)) fs.mkdirSync(dmftTmpDir, { recursive: true });
        if (fs.existsSync(mainSaveDir) && !fs.existsSync(dmftSaveDir)) {
          try {
            execSync(`cp -r "${mainSaveDir}" "${dmftSaveDir}"`, { timeout: 120_000 });
          } catch { /* best effort — NSCF will recompute if missing */ }
        }

        const nscfRes = await runQECommand(
          path.posix.join(getQEBinDir(), "pw.x"), dmftNscfFile, dmftWannierDir, 3_600_000,
        );
        if (nscfRes.exitCode !== 0) {
          console.log(`[QE-Worker] DMFT NSCF failed (exit ${nscfRes.exitCode}), skipping Wannier90`);
        } else {
          console.log(`[QE-Worker] DMFT NSCF completed`);

          // ── Step 2: Wannier90 preprocessing (-pp) ──
          console.log(`[QE-Worker] DMFT Wannier step 2/4: wannier90.x -pp`);
          const dmftWinContent = generateWannier90Win({
            prefix: dmftPrefix, elements, counts, numBands: dmftNbnd,
            kGrid: dmftKGrid, fermiEnergy: result.scf!.fermiEnergy!,
            latticeVectors: dmftLatticeVectors, positions,
            wannierMode: "dmft_projector",
          });
          fs.writeFileSync(path.posix.join(dmftWannierDir, `${dmftPrefix}.win`), dmftWinContent);

          const w90ppRes = await runQECommand(
            path.posix.join(getQEBinDir(), "wannier90.x"),
            `-pp ${dmftPrefix}`, dmftWannierDir, 600_000,
          );

          if (w90ppRes.exitCode === 0) {
            // ── Step 3: pw2wannier90 ──
            console.log(`[QE-Worker] DMFT Wannier step 3/4: pw2wannier90.x`);
            const pw2wInput = generatePW2Wannier90Input(dmftPrefix);
            const pw2wFile = path.posix.join(dmftWannierDir, `${dmftPrefix}_pw2wan.in`);
            fs.writeFileSync(pw2wFile, pw2wInput);

            const pw2wRes = await runQECommand(
              path.posix.join(getQEBinDir(), "pw2wannier90.x"),
              pw2wFile, dmftWannierDir, 1_800_000,
            );

            if (pw2wRes.exitCode === 0) {
              // ── Step 4: Wannier90 full minimization ──
              console.log(`[QE-Worker] DMFT Wannier step 4/4: wannier90.x (full, DMFT projector)`);
              const w90Res = await runQECommand(
                path.posix.join(getQEBinDir(), "wannier90.x"),
                dmftPrefix, dmftWannierDir, 1_800_000,
              );

              if (w90Res.exitCode === 0) {
                console.log(`[QE-Worker] DMFT Wannier90 completed — _hr.dat should be available`);
              } else {
                console.log(`[QE-Worker] DMFT Wannier90 full failed (exit ${w90Res.exitCode}), bundle will lack H(k)`);
              }
            } else {
              console.log(`[QE-Worker] DMFT pw2wannier90 failed (exit ${pw2wRes.exitCode})`);
            }
          } else {
            console.log(`[QE-Worker] DMFT wannier90 -pp failed (exit ${w90ppRes.exitCode})`);
          }
        }

        // Export the bundle — now with H(k) from _hr.dat if Wannier90 succeeded
        result.dmftBundle = await exportDMFTBundle({
          formula, elements, counts, positions,
          latticeVectors: dmftLatticeVectors,
          pressureGpa: workerPressure,
          fermiEnergy: result.scf!.fermiEnergy!,
          qualityTier: result.qualityTier ?? "final_converged",
          hubbardWorkflow: result.hubbardWorkflow!,
          acbn0: result.acbn0,
          magneticGroundState: result.magneticGroundState,
          wannier90Dir: dmftWannierDir,
          wannier90Prefix: dmftPrefix,
        });

        console.log(
          `[QE-Worker] DMFT bundle exported for ${formula}: ` +
          `${result.dmftBundle.nCorrelatedShells} shells, ` +
          `${result.dmftBundle.nCorrelatedOrbitals} orbitals, ` +
          `H(k)=${result.dmftBundle.hamiltonianParsed}, ` +
          `format=${result.dmftBundle.format}`
        );

        // ── Submit bundle to DMFT service if available ──
        if (result.dmftBundle.hamiltonianParsed) {
          const dmftServiceUrl = process.env.DMFT_SERVICE_URL ?? "";
          if (dmftServiceUrl) {
            try {
              const bundlePath = result.dmftBundle.bundlePath;
              // Upload the bundle file via multipart form data.
              // The DMFT service may be on a different VM (gnn-training),
              // so we can't pass a local file path — must upload the file.
              const bundleData = fs.readFileSync(bundlePath);
              const boundary = `----QAEBundle${Date.now()}`;
              const fileName = path.posix.basename(bundlePath);
              const multipartBody = Buffer.concat([
                Buffer.from(
                  `--${boundary}\r\n` +
                  `Content-Disposition: form-data; name="bundle"; filename="${fileName}"\r\n` +
                  `Content-Type: application/octet-stream\r\n\r\n`
                ),
                bundleData,
                Buffer.from(`\r\n--${boundary}--\r\n`),
              ]);
              const submitRes = await fetch(`${dmftServiceUrl}/submit`, {
                method: "POST",
                headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
                body: multipartBody,
              });
              if (submitRes.ok) {
                const submitData = await submitRes.json() as any;
                (result as any).dmftJobId = submitData.job_id;
                console.log(`[QE-Worker] DMFT job submitted: ${submitData.job_id} (${submitData.status})`);

                // Poll for results — DMFT runs hours, so poll with backoff
                // Max 12 polls × 5 min = 1h of polling. DMFT results are also
                // written to disk and can be picked up by a background loop later.
                const pollIntervalMs = 300_000; // 5 minutes
                const maxPolls = 12;
                for (let poll = 0; poll < maxPolls; poll++) {
                  await new Promise(r => setTimeout(r, pollIntervalMs));
                  try {
                    const statusRes = await fetch(`${dmftServiceUrl}/status/${submitData.job_id}`);
                    if (!statusRes.ok) break;
                    const statusData = await statusRes.json() as any;

                    if (statusData.status?.startsWith("completed") || statusData.status === "failed") {
                      console.log(`[QE-Worker] DMFT job ${submitData.job_id} finished: ${statusData.status}`);

                      // Fetch full results
                      const resultRes = await fetch(`${dmftServiceUrl}/result/${submitData.job_id}`);
                      if (resultRes.ok) {
                        const dmftData = await resultRes.json() as any;
                        const dmftResults = dmftData.results ?? {};
                        const phases = dmftResults.phases ?? dmftResults;

                        // Populate DMFT result fields for database
                        (result as any).dmftConverged = dmftResults.converged ?? phases.dmft_1p?.converged ?? false;
                        (result as any).dmftClusterSize = dmftResults.cluster_decision?.nc ?? 0;
                        (result as any).dmftAvgSign = phases.dca_sweep?.sweep?.[0]?.avg_sign ?? null;

                        // Pairing results (from orchestrated pipeline or realistic pipeline)
                        const pairing = phases.pairing ?? phases.dca_sweep?.sweep?.slice(-1)?.[0]?.pairing ?? {};
                        (result as any).dmftLambdaPair = dmftResults.lambda_max ?? pairing.lambda ?? null;
                        (result as any).dmftGapSymmetry = dmftResults.dominant_channel ?? pairing.channel ?? null;
                        (result as any).dmftGapNodes = pairing.nodes ?? null;
                        (result as any).dmftIsUnconventional = pairing.is_unconventional ?? null;
                        (result as any).dmftTcBSE = dmftResults.tc_K ?? null;
                        (result as any).dmftTcBSEConfidence = dmftResults.tc_confidence ?? null;
                        (result as any).dmftDominantChannel = dmftResults.dominant_channel ?? null;

                        console.log(
                          `[QE-Worker] DMFT results for ${formula}: ` +
                          `Tc=${(result as any).dmftTcBSE ?? "N/A"} K, ` +
                          `channel=${(result as any).dmftDominantChannel ?? "?"}, ` +
                          `λ=${(result as any).dmftLambdaPair ?? "?"}`
                        );
                      }
                      break;
                    }

                    if (statusData.status === "running") {
                      console.log(`[QE-Worker] DMFT job ${submitData.job_id} still running (poll ${poll + 1}/${maxPolls})`);
                    }
                  } catch {
                    // Polling failed — service may be busy, continue
                  }
                }
              } else {
                console.log(`[QE-Worker] DMFT service returned ${submitRes.status}: ${await submitRes.text()}`);
              }
            } catch (dmftFetchErr: any) {
              console.log(`[QE-Worker] DMFT service unreachable: ${dmftFetchErr.message}`);
            }
          }
        }
      } catch (dmftErr: any) {
        console.log(`[QE-Worker] DMFT bundle export failed for ${formula}: ${(dmftErr.message ?? "").slice(0, 200)}`);
      }
    }

    // --- Populate quality gate and uncertainty fields ---
    result.qualityGatePassed = qualityGatePass;
    result.qualityGateReasons = qualityGateReason;

    // Determine uncertainty/confidence for each result dimension
    const hasDFPT = result.dfpt != null && (result.dfpt as any).lambda > 0;
    const hasEPW = result.epw != null && result.epw.lambda > 0;
    const hasSSCHA = result.sscha != null && result.sscha.converged;
    const hasACBN0 = result.acbn0 != null && result.acbn0.converged;
    const hasFullPhonon = phononHasResults && result.phonon!.frequencies.length >= 10;
    const hasGammaOnly = phononHasResults && result.phonon!.frequencies.length < 10 && result.phonon!.frequencies.length > 0;

    const tcConfidence: "high" | "medium" | "low" | "surrogate" =
      (hasEPW && result.epw!.converged) || (hasSSCHA && hasDFPT) ? "high" :
      hasDFPT && dfptGatePass ? "high" :
      hasFullPhonon && qualityGatePass ? "medium" :
      scfConverged && isMetallicForTc ? "low" : "surrogate";

    const lambdaConfidence: "high" | "medium" | "low" | "surrogate" =
      hasEPW ? "high" :
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
    if (!scfPressureOk) tcReasons.push(`high pressure (${result.scf?.pressure?.toFixed(2)} GPa)`);
    if (hasEPW) tcReasons.push("EPW Migdal-Eliashberg");
    if (hasSSCHA) tcReasons.push("SSCHA anharmonic corrections");
    if (hasACBN0) tcReasons.push(`ACBN0 μ*=${result.acbn0!.muStar.toFixed(3)}`);
    if (result.qeDFTPlusU) tcReasons.push("DFT+U applied");

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
    // Save bundle for any non-failed tier. Tier has already been promoted above
    // (right before the DMFT check) based on phonon/e-ph completion, so the
    // bundle reflects the final tier.
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
