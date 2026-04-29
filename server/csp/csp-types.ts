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
