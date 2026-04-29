/**
 * Crystal Structure Prediction (CSP) Pipeline Types
 *
 * Unified candidate representation for multi-engine structure prediction:
 * AIRSS (sensible random), CALYPSO (PSO), USPEX (evolutionary), Vegard/VCA,
 * prototype templates, and lattice-free random.
 *
 * Three-stage pipeline:
 * Stage 1: Broad exploration (heavy AIRSS + Vegard, light CALYPSO/USPEX)
 * Stage 2: Basin refinement (shift to CALYPSO/USPEX, cross-seed engines)
 * Stage 3: Final selection (10 structural families, enthalpy-ranked)
 */

import type { StructureCandidate } from "../dft/vegard-lattice";

// ---------------------------------------------------------------------------
// Engine types
// ---------------------------------------------------------------------------

export type CSPEngineName = "airss" | "calypso" | "uspex" | "vegard" | "vca" | "prototype" | "random" | "known-structure";

export type RelaxationLevel = "raw" | "xtb" | "single-point-qe" | "relax-qe" | "vc-relax-qe";

// ---------------------------------------------------------------------------
// Unified candidate
// ---------------------------------------------------------------------------

export interface CSPCandidate extends StructureCandidate {
  // --- Engine provenance ---
  sourceEngine: CSPEngineName;
  generationStage: 1 | 2 | 3;
  /** RNG seed used to generate this structure (for reproducibility). */
  seed: number;
  /** Seeds of parent structures (for cross-seeded / mutated structures). */
  parentSeeds?: number[];

  // --- Thermodynamic scoring ---
  /** Enthalpy H = E + PV (eV). More physically meaningful than energy at P>0. */
  enthalpy?: number;
  enthalpyPerAtom?: number;
  /** Target pressure for this candidate (GPa). */
  pressureGPa: number;
  /** Cell volume (Angstrom^3), needed for H = E + PV. */
  cellVolume?: number;

  // --- Full cell description ---
  /** 3x3 lattice vectors in Angstrom (row-major: vecs[0] = a-vector). */
  cellVectors?: [number, number, number][];
  /** Full lattice parameters (for non-cubic systems). */
  latticeParams?: {
    a: number; b: number; c: number;
    alpha: number; beta: number; gamma: number;
  };

  // --- Structural identity ---
  /** Numerical fingerprint for deduplication / clustering. */
  fingerprint?: number[];
  /** Motif cluster ID (assigned in Stage 2). */
  motifId?: string;
  /** Structural family ID (assigned in Stage 3). */
  familyId?: string;

  // --- Relaxation state ---
  relaxationLevel: RelaxationLevel;
  /** Max force after relaxation (Ry/bohr). Lower = better converged. */
  postRelaxForce?: number;
  /** Relaxation history: which engines processed this candidate. */
  relaxationHistory?: string[];
}

// ---------------------------------------------------------------------------
// Stage configuration
// ---------------------------------------------------------------------------

export interface CSPStageConfig {
  stage: 1 | 2 | 3;
  /** Max total structures to generate in this stage. */
  totalBudget: number;
  /** Fractional weight for each engine (must sum to ~1.0). */
  engineWeights: Record<CSPEngineName, number>;
  /** How deeply to relax candidates in this stage. */
  relaxationLevel: RelaxationLevel;
  /** Cosine distance below which structures are considered duplicates. */
  deduplicationThreshold: number;
}

/** Default stage configs implementing the 3-stage strategy. */
export const DEFAULT_STAGE_CONFIGS: Record<1 | 2 | 3, CSPStageConfig> = {
  1: {
    stage: 1,
    totalBudget: 200,
    engineWeights: {
      airss: 0.40,
      vegard: 0.10,
      vca: 0.10,
      prototype: 0.05,
      random: 0.10,
      calypso: 0.15,
      uspex: 0.10,
      "known-structure": 0.0,
    },
    relaxationLevel: "xtb",
    deduplicationThreshold: 0.05,
  },
  2: {
    stage: 2,
    totalBudget: 100,
    engineWeights: {
      calypso: 0.35,
      uspex: 0.35,
      airss: 0.15,
      vegard: 0.0,
      vca: 0.05,
      prototype: 0.0,
      random: 0.10,
      "known-structure": 0.0,
    },
    relaxationLevel: "relax-qe",
    deduplicationThreshold: 0.03,
  },
  3: {
    stage: 3,
    totalBudget: 30,
    engineWeights: {
      calypso: 0.40,
      uspex: 0.40,
      airss: 0.10,
      vegard: 0.0,
      vca: 0.0,
      prototype: 0.0,
      random: 0.10,
      "known-structure": 0.0,
    },
    relaxationLevel: "vc-relax-qe",
    deduplicationThreshold: 0.02,
  },
};

// ---------------------------------------------------------------------------
// Motif and family clustering
// ---------------------------------------------------------------------------

export interface MotifCluster {
  motifId: string;
  members: CSPCandidate[];
  /** Average fingerprint vector (centroid). */
  centroid: number[];
  avgEnthalpy: number;
  count: number;
  bestCandidate: CSPCandidate;
}

export interface StructuralFamily {
  familyId: string;
  motifs: MotifCluster[];
  /** Single best representative for DFT follow-up. */
  representative: CSPCandidate;
  enthalpyRange: [number, number];
  spaceGroups: string[];
  /** Number of independent engines that found this family. */
  engineDiversity: number;
  /** Confidence from convergence + diversity metrics. */
  confidence: number;
}

// ---------------------------------------------------------------------------
// Pipeline result
// ---------------------------------------------------------------------------

export interface CSPPipelineResult {
  formula: string;
  pressureGPa: number;
  families: StructuralFamily[];
  /** All candidates across all stages (for analysis). */
  allCandidates: CSPCandidate[];
  stageTimings: {
    stage1Ms: number;
    stage2Ms: number;
    stage3Ms: number;
  };
  /** Which engines were available and used. */
  enginesUsed: CSPEngineName[];
}

// ---------------------------------------------------------------------------
// Engine interface
// ---------------------------------------------------------------------------

export interface CSPEngineConfig {
  /** Path to the engine binary (or auto-detected). */
  binaryPath: string;
  /** Working directory for this run. */
  workDir: string;
  /** Kill the engine after this many ms. */
  timeoutMs: number;
  /** Max structures to generate. */
  maxStructures: number;
  /** Target pressure (GPa). */
  pressureGPa: number;
  /** Seed structures for cross-seeding (Stage 2+). */
  seedStructures?: CSPCandidate[];
  /** Base RNG seed for reproducibility. */
  baseSeed?: number;
}

// ---------------------------------------------------------------------------
// Screening tiers — controls atom count caps and generation budgets
// ---------------------------------------------------------------------------

export type ScreeningTier = "preview" | "standard" | "deep" | "publication";

export interface ScreeningTierConfig {
  tier: ScreeningTier;
  /** Maximum atoms in the primitive cell. */
  maxAtoms: number;
  /** Formula-unit multiplicities to explore. */
  zValues: number[];
  /** AIRSS budget per Z × volume combo. */
  airssBudgetPerCombo: number;
  /** Total AIRSS cap (across all Z × volume combos). */
  airssTotalCap: number;
  /** PyXtal budget per Z value. */
  pyxtalBudgetPerZ: number;
  /** Total PyXtal cap. */
  pyxtalTotalCap: number;
  /** Volume fractions to explore (relative to V0 estimate). */
  volumeEnsemble: number[];
  /** DFT-0 budget: floor, cap, and cluster fraction. */
  dft0Floor: number;
  dft0Cap: number;
  dft0ClusterFraction: number;
}

export const SCREENING_TIERS: Record<ScreeningTier, ScreeningTierConfig> = {
  preview: {
    tier: "preview",
    maxAtoms: 20,
    zValues: [1, 2, 4],
    airssBudgetPerCombo: 100,
    airssTotalCap: 2500,
    pyxtalBudgetPerZ: 50,
    pyxtalTotalCap: 250,
    volumeEnsemble: [0.85, 1.0, 1.15],
    dft0Floor: 3,
    dft0Cap: 8,
    dft0ClusterFraction: 0.03,
  },
  standard: {
    tier: "standard",
    maxAtoms: 40,
    zValues: [1, 2, 3, 4, 6],
    airssBudgetPerCombo: 200,
    airssTotalCap: 5000,
    pyxtalBudgetPerZ: 100,
    pyxtalTotalCap: 500,
    volumeEnsemble: [0.70, 0.85, 1.0, 1.15, 1.30],
    dft0Floor: 15,
    dft0Cap: 40,
    dft0ClusterFraction: 0.10,
  },
  deep: {
    tier: "deep",
    maxAtoms: 80,
    zValues: [1, 2, 3, 4, 6, 8],
    airssBudgetPerCombo: 300,
    airssTotalCap: 10000,
    pyxtalBudgetPerZ: 150,
    pyxtalTotalCap: 1000,
    volumeEnsemble: [0.70, 0.85, 1.0, 1.15, 1.30],
    dft0Floor: 50,
    dft0Cap: 120,
    dft0ClusterFraction: 0.25,
  },
  publication: {
    tier: "publication",
    maxAtoms: 120,
    zValues: [1, 2, 3, 4, 6, 8],
    airssBudgetPerCombo: 500,
    airssTotalCap: 10000,
    pyxtalBudgetPerZ: 200,
    pyxtalTotalCap: 1000,
    volumeEnsemble: [0.60, 0.70, 0.85, 1.0, 1.15, 1.30, 1.50],
    dft0Floor: 120,
    dft0Cap: 300,
    dft0ClusterFraction: 0.40,
  },
};

// ---------------------------------------------------------------------------
// Pressure-aware MINSEP profiles
// ---------------------------------------------------------------------------

/**
 * Get pressure-scaled minimum interatomic distance.
 * High-pressure structures (especially hydrides) have physically shorter
 * bond lengths. This prevents structure generators from rejecting valid
 * dense packings.
 */
export function pressureScaledMinsep(baseMinsep: number, pressureGPa: number): number {
  if (pressureGPa < 20) return baseMinsep;
  if (pressureGPa < 100) return baseMinsep * 0.90;
  if (pressureGPa < 200) return baseMinsep * 0.82;
  return baseMinsep * 0.75;
}

/** Element-pair covalent radii sums (Angstrom) for MINSEP calculation. */
export const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11, P: 1.07,
  S: 1.05, Cl: 1.02, K: 2.03, Ca: 1.76, Sc: 1.70, Ti: 1.60, V: 1.53,
  Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24, Cu: 1.32, Zn: 1.22,
  Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20, Rb: 2.20, Sr: 1.95,
  Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54, Ru: 1.46, Rh: 1.42, Pd: 1.39,
  Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39, Sb: 1.39, Te: 1.38, I: 1.39,
  Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04, Hf: 1.75, Ta: 1.70, W: 1.62,
  Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36, Au: 1.36, Pb: 1.46, Bi: 1.48,
  Th: 2.06,
};

/**
 * Get element-pair minimum separation for structure generation.
 * Uses covalent radii sum scaled by pair type and pressure.
 */
/** Absolute minimum distances — never go below these regardless of scaling. */
const ABSOLUTE_FLOOR: Record<string, number> = {
  "H-H": 0.45,  // H2 bond = 0.74 Å; 0.45 catches true overlap
  "M-H": 0.80,  // no metal-H bond shorter than this exists
  "M-M": 1.50,  // metallic bonds floor
};

export function getPairMinsep(el1: string, el2: string, pressureGPa: number): number {
  const r1 = COVALENT_RADII[el1] ?? 1.2;
  const r2 = COVALENT_RADII[el2] ?? 1.2;

  // Base scaling by pair type
  const isHH = el1 === "H" && el2 === "H";
  const isMH = el1 === "H" || el2 === "H";
  const baseScale = isHH ? 0.50 : (isMH ? 0.60 : 0.75);

  const baseMinsep = (r1 + r2) * baseScale;
  const scaled = pressureScaledMinsep(baseMinsep, pressureGPa);

  // Apply absolute floor — never allow below physical minimum
  const floorKey = isHH ? "H-H" : (isMH ? "M-H" : "M-M");
  return Math.max(scaled, ABSOLUTE_FLOOR[floorKey]);
}

/**
 * Pressure-conditioned volume estimates (Angstrom^3/atom).
 * Returns an array of target volumes for the volume ensemble.
 */
export function pressureVolumeEnsemble(
  volumePerAtom: number,
  pressureGPa: number,
  fractions: number[],
): number[] {
  // Apply Birch-Murnaghan-like compression to the base volume
  let compressedV0 = volumePerAtom;
  if (pressureGPa > 0) {
    const compressionFactor = Math.pow(1 + 4 * pressureGPa / 100, -1 / 4);
    compressedV0 = volumePerAtom * Math.max(0.4, compressionFactor);
  }

  // For high pressure, shift the ensemble toward compressed volumes
  // but still keep some expanded ones for exploration
  if (pressureGPa >= 100) {
    // Add extra compressed points
    const extraCompressed = [0.55, 0.65].filter(f => !fractions.includes(f));
    return [...extraCompressed, ...fractions].map(f => compressedV0 * f);
  }

  return fractions.map(f => compressedV0 * f);
}

export interface CSPEngine {
  name: CSPEngineName;
  /** Check if the binary is installed and runnable. */
  isAvailable(): boolean;
  /** Generate candidate structures. */
  generateStructures(
    elements: string[],
    counts: Record<string, number>,
    config: CSPEngineConfig,
  ): Promise<CSPCandidate[]>;
}

// ---------------------------------------------------------------------------
// Enthalpy computation
// ---------------------------------------------------------------------------

/** 1 GPa * 1 Angstrom^3 = 0.00624150913 eV */
export const GPA_A3_TO_EV = 0.00624150913;

/**
 * Compute enthalpy H = E + PV.
 * @param totalEnergyEv Total energy in eV (from QE SCF).
 * @param pressureGPa Target pressure in GPa.
 * @param cellVolumeA3 Cell volume in Angstrom^3.
 * @returns Enthalpy in eV.
 */
export function computeEnthalpy(
  totalEnergyEv: number,
  pressureGPa: number,
  cellVolumeA3: number,
): number {
  return totalEnergyEv + pressureGPa * cellVolumeA3 * GPA_A3_TO_EV;
}

/**
 * Compute cell volume from 3x3 lattice vectors (scalar triple product).
 */
export function cellVolumeFromVectors(vecs: [number, number, number][]): number {
  if (vecs.length < 3) return 0;
  const [a, b, c] = vecs;
  return Math.abs(
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0])
  );
}
