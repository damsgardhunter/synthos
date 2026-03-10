import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import { passesValenceFilter } from "../learning/candidate-generator";
import { normalizeFormula } from "../learning/utils";

interface StructuralEmbedding {
  coordination: number;
  avgBondAngle: number;
  layerSpacing: number;
  cOverA: number;
  symmetryOrder: number;
  electronCountPerSite: number;
  dimensionality: number;
  cageFraction: number;
  interstitialFraction: number;
  connectivityIndex: number;
  anisotropy: number;
  voidFraction: number;
}

interface StructuralMotif {
  name: string;
  spaceGroup: string;
  crystalSystem: string;
  embedding: StructuralEmbedding;
  siteRoles: SiteRole[];
  latticeRatios: { a: number; b: number; c: number };
  scAffinity: number;
  tcRange: [number, number];
  pairingMechanism: string;
}

interface SiteRole {
  label: string;
  multiplicity: number;
  position: [number, number, number];
  role: "framework" | "interstitial" | "plane" | "chain" | "cage" | "spacer" | "reservoir";
  preferredCategories: string[];
}

interface StructureCandidate {
  motif: string;
  formula: string;
  siteAssignment: Record<string, string>;
  structuralScore: number;
  predictedTc: number;
  lambda: number;
  embedding: StructuralEmbedding;
}

export interface StructureDiffusionResult {
  candidates: StructureCandidate[];
  motifUsed: string;
  elementsTriedPerSite: Record<string, string[]>;
  bestFormula: string;
  bestTc: number;
  totalEvaluated: number;
}

interface StructureDiffusionStats {
  totalMotifGenerated: number;
  totalCandidatesEvaluated: number;
  bestTcAchieved: number;
  bestFormula: string;
  bestMotif: string;
  motifSuccessRates: Record<string, { tried: number; passed: number; avgTc: number }>;
  recentResults: { formula: string; tc: number; motif: string }[];
  learnedWeights: Record<string, number>;
}

const ELEMENT_CATEGORIES: Record<string, string[]> = {
  highCouplingTM: ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf"],
  lightPhonon: ["H", "B", "C", "N"],
  anion: ["O", "N", "F", "S", "Se", "Te"],
  rareEarth: ["La", "Y", "Ce", "Gd", "Nd", "Sc"],
  alkalineEarth: ["Ca", "Sr", "Ba", "Mg"],
  pnictogen: ["As", "P", "Sb", "Bi"],
  chalcogen: ["S", "Se", "Te"],
  magneticTM: ["Fe", "Co", "Ni", "Cu", "Mn"],
  pBlockMetal: ["Al", "Ga", "In", "Sn", "Ge", "Pb"],
  hydrogen: ["H"],
};

const SC_MOTIF_LIBRARY: StructuralMotif[] = [
  {
    name: "CuO2-plane",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 90, layerSpacing: 6.5, cOverA: 3.2,
      symmetryOrder: 16, electronCountPerSite: 9, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.1, connectivityIndex: 0.8,
      anisotropy: 0.85, voidFraction: 0.15,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0.35], role: "reservoir", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 2, position: [0, 0, 0], role: "plane", preferredCategories: ["magneticTM"] },
      { label: "X", multiplicity: 4, position: [0.5, 0, 0], role: "plane", preferredCategories: ["anion"] },
      { label: "X2", multiplicity: 2, position: [0, 0, 0.15], role: "spacer", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.2 },
    scAffinity: 0.92,
    tcRange: [30, 160],
    pairingMechanism: "d-wave spin fluctuation",
  },
  {
    name: "FeAs-layer",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 109.5, layerSpacing: 6.0, cOverA: 3.0,
      symmetryOrder: 16, electronCountPerSite: 6, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.05, connectivityIndex: 0.75,
      anisotropy: 0.8, voidFraction: 0.12,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "reservoir", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 2, position: [0.25, 0.25, 0.25], role: "plane", preferredCategories: ["magneticTM", "highCouplingTM"] },
      { label: "X", multiplicity: 2, position: [0, 0.5, 0.15], role: "plane", preferredCategories: ["pnictogen", "chalcogen"] },
      { label: "X2", multiplicity: 2, position: [0, 0, 0.35], role: "spacer", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.0 },
    scAffinity: 0.85,
    tcRange: [20, 80],
    pairingMechanism: "s+- spin fluctuation",
  },
  {
    name: "clathrate-cage",
    spaceGroup: "Im-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 12, avgBondAngle: 60, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 1, dimensionality: 3,
      cageFraction: 0.75, interstitialFraction: 0, connectivityIndex: 0.95,
      anisotropy: 0, voidFraction: 0.3,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "cage", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "H", multiplicity: 8, position: [0.25, 0.25, 0.25], role: "framework", preferredCategories: ["hydrogen", "lightPhonon"] },
      { label: "H2", multiplicity: 6, position: [0, 0.5, 0.5], role: "framework", preferredCategories: ["hydrogen", "lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.95,
    tcRange: [100, 300],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "A15-chain",
    spaceGroup: "Pm-3n",
    crystalSystem: "cubic",
    embedding: {
      coordination: 12, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 5, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0.15, connectivityIndex: 0.9,
      anisotropy: 0.1, voidFraction: 0.1,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "framework", preferredCategories: ["pBlockMetal", "alkalineEarth"] },
      { label: "B", multiplicity: 6, position: [0.25, 0, 0.5], role: "chain", preferredCategories: ["highCouplingTM"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.88,
    tcRange: [10, 40],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "kagome-flat",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 4, avgBondAngle: 120, layerSpacing: 5.5, cOverA: 1.8,
      symmetryOrder: 24, electronCountPerSite: 4, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.2, connectivityIndex: 0.7,
      anisotropy: 0.65, voidFraction: 0.25,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "spacer", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B", multiplicity: 3, position: [0.5, 0, 0.5], role: "plane", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 2, position: [0.333, 0.667, 0], role: "framework", preferredCategories: ["pBlockMetal", "pnictogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.8 },
    scAffinity: 0.72,
    tcRange: [2, 30],
    pairingMechanism: "flat-band enhanced",
  },
  {
    name: "hexagonal-layer",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 6, avgBondAngle: 120, layerSpacing: 3.5, cOverA: 1.15,
      symmetryOrder: 24, electronCountPerSite: 3, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.1, connectivityIndex: 0.85,
      anisotropy: 0.5, voidFraction: 0.18,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "plane", preferredCategories: ["alkalineEarth", "highCouplingTM"] },
      { label: "B", multiplicity: 2, position: [0.333, 0.667, 0.5], role: "plane", preferredCategories: ["lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.15 },
    scAffinity: 0.82,
    tcRange: [20, 50],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "perovskite-3D",
    spaceGroup: "Pm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 3, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.95,
      anisotropy: 0, voidFraction: 0.05,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "reservoir", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 1, position: [0.5, 0.5, 0.5], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 3, position: [0.5, 0.5, 0], role: "framework", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.75,
    tcRange: [5, 30],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "layered-hydride",
    spaceGroup: "P6_3/mmc",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 8, avgBondAngle: 109.5, layerSpacing: 4.0, cOverA: 2.5,
      symmetryOrder: 24, electronCountPerSite: 2, dimensionality: 2,
      cageFraction: 0.3, interstitialFraction: 0.5, connectivityIndex: 0.8,
      anisotropy: 0.6, voidFraction: 0.35,
    },
    siteRoles: [
      { label: "M", multiplicity: 2, position: [0.333, 0.667, 0.25], role: "plane", preferredCategories: ["highCouplingTM", "rareEarth"] },
      { label: "H1", multiplicity: 4, position: [0.333, 0.667, 0.1], role: "interstitial", preferredCategories: ["hydrogen"] },
      { label: "H2", multiplicity: 2, position: [0, 0, 0], role: "interstitial", preferredCategories: ["hydrogen", "lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 2.5 },
    scAffinity: 0.90,
    tcRange: [50, 250],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "NaCl-rocksalt",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 1.0,
      anisotropy: 0, voidFraction: 0.08,
    },
    siteRoles: [
      { label: "M", multiplicity: 4, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM"] },
      { label: "X", multiplicity: 4, position: [0.5, 0.5, 0.5], role: "framework", preferredCategories: ["anion", "lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.70,
    tcRange: [5, 25],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "H-channel",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 8, avgBondAngle: 90, layerSpacing: 0, cOverA: 0.8,
      symmetryOrder: 16, electronCountPerSite: 1.5, dimensionality: 3,
      cageFraction: 0.4, interstitialFraction: 0.6, connectivityIndex: 0.7,
      anisotropy: 0.3, voidFraction: 0.4,
    },
    siteRoles: [
      { label: "M", multiplicity: 1, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM"] },
      { label: "H1", multiplicity: 2, position: [0.5, 0, 0], role: "chain", preferredCategories: ["hydrogen"] },
      { label: "H2", multiplicity: 2, position: [0, 0.5, 0.5], role: "chain", preferredCategories: ["hydrogen"] },
      { label: "X", multiplicity: 1, position: [0.5, 0.5, 0], role: "interstitial", preferredCategories: ["lightPhonon", "anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 0.8 },
    scAffinity: 0.88,
    tcRange: [30, 200],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "breathing-kagome",
    spaceGroup: "P-3m1",
    crystalSystem: "trigonal",
    embedding: {
      coordination: 4, avgBondAngle: 120, layerSpacing: 7.0, cOverA: 2.2,
      symmetryOrder: 12, electronCountPerSite: 5, dimensionality: 2,
      cageFraction: 0.15, interstitialFraction: 0.1, connectivityIndex: 0.6,
      anisotropy: 0.75, voidFraction: 0.3,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "spacer", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B", multiplicity: 3, position: [0.5, 0, 0.33], role: "plane", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 3, position: [0.5, 0.5, 0.17], role: "framework", preferredCategories: ["pnictogen", "chalcogen", "pBlockMetal"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 2.2 },
    scAffinity: 0.68,
    tcRange: [1, 20],
    pairingMechanism: "flat-band enhanced",
  },
  {
    name: "Laves-MgZn2",
    spaceGroup: "P6_3/mmc",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 12, avgBondAngle: 60, layerSpacing: 0, cOverA: 1.63,
      symmetryOrder: 24, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0.3, interstitialFraction: 0, connectivityIndex: 0.9,
      anisotropy: 0.2, voidFraction: 0.12,
    },
    siteRoles: [
      { label: "A", multiplicity: 4, position: [0.333, 0.667, 0.0625], role: "framework", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 2, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "B2", multiplicity: 6, position: [0.833, 0.667, 0.25], role: "framework", preferredCategories: ["highCouplingTM", "pBlockMetal"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.63 },
    scAffinity: 0.65,
    tcRange: [1, 15],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "TMD-2H",
    spaceGroup: "P6_3/mmc",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 6, avgBondAngle: 82, layerSpacing: 6.2, cOverA: 3.6,
      symmetryOrder: 24, electronCountPerSite: 4, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.75,
      anisotropy: 0.85, voidFraction: 0.2,
    },
    siteRoles: [
      { label: "M", multiplicity: 2, position: [0.333, 0.667, 0.25], role: "plane", preferredCategories: ["highCouplingTM"] },
      { label: "X1", multiplicity: 2, position: [0.333, 0.667, 0.12], role: "plane", preferredCategories: ["chalcogen"] },
      { label: "X2", multiplicity: 2, position: [0.333, 0.667, 0.38], role: "plane", preferredCategories: ["chalcogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.6 },
    scAffinity: 0.74,
    tcRange: [2, 15],
    pairingMechanism: "CDW-phonon coupled",
  },
  {
    name: "fullerene-A3C60",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 12, avgBondAngle: 60, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 3, dimensionality: 3,
      cageFraction: 0.85, interstitialFraction: 0.15, connectivityIndex: 0.65,
      anisotropy: 0, voidFraction: 0.35,
    },
    siteRoles: [
      { label: "A", multiplicity: 3, position: [0.25, 0.25, 0.25], role: "interstitial", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "C60", multiplicity: 1, position: [0, 0, 0], role: "cage", preferredCategories: ["lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.70,
    tcRange: [15, 45],
    pairingMechanism: "molecular phonon-mediated",
  },
  {
    name: "heavy-fermion-115",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 8, avgBondAngle: 90, layerSpacing: 7.5, cOverA: 1.6,
      symmetryOrder: 16, electronCountPerSite: 5, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.05, connectivityIndex: 0.7,
      anisotropy: 0.8, voidFraction: 0.1,
    },
    siteRoles: [
      { label: "RE", multiplicity: 1, position: [0, 0, 0], role: "reservoir", preferredCategories: ["rareEarth"] },
      { label: "TM", multiplicity: 1, position: [0, 0, 0.5], role: "plane", preferredCategories: ["magneticTM"] },
      { label: "X", multiplicity: 4, position: [0.5, 0, 0.31], role: "plane", preferredCategories: ["pBlockMetal", "pnictogen"] },
      { label: "X2", multiplicity: 1, position: [0.5, 0.5, 0], role: "spacer", preferredCategories: ["pBlockMetal", "pnictogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.6 },
    scAffinity: 0.62,
    tcRange: [0.5, 5],
    pairingMechanism: "magnetically mediated",
  },
  {
    name: "borocarbide-RTBC",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 5.3, cOverA: 3.5,
      symmetryOrder: 16, electronCountPerSite: 4, dimensionality: 2.5,
      cageFraction: 0, interstitialFraction: 0.1, connectivityIndex: 0.82,
      anisotropy: 0.55, voidFraction: 0.12,
    },
    siteRoles: [
      { label: "R", multiplicity: 2, position: [0, 0, 0.35], role: "reservoir", preferredCategories: ["rareEarth"] },
      { label: "T", multiplicity: 4, position: [0, 0.5, 0], role: "plane", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "B", multiplicity: 4, position: [0, 0, 0.15], role: "framework", preferredCategories: ["lightPhonon"] },
      { label: "C", multiplicity: 2, position: [0, 0, 0], role: "interstitial", preferredCategories: ["lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.5 },
    scAffinity: 0.72,
    tcRange: [5, 25],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "carbon-clathrate",
    spaceGroup: "Pm-3n",
    crystalSystem: "cubic",
    embedding: {
      coordination: 4, avgBondAngle: 109.5, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0.9, interstitialFraction: 0.1, connectivityIndex: 0.95,
      anisotropy: 0, voidFraction: 0.25,
    },
    siteRoles: [
      { label: "C1", multiplicity: 24, position: [0.18, 0.18, 0], role: "cage", preferredCategories: ["lightPhonon"] },
      { label: "C2", multiplicity: 16, position: [0.12, 0.12, 0.12], role: "cage", preferredCategories: ["lightPhonon"] },
      { label: "G", multiplicity: 6, position: [0.25, 0, 0.5], role: "interstitial", preferredCategories: ["alkalineEarth", "rareEarth"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.78,
    tcRange: [20, 100],
    pairingMechanism: "covalent phonon-mediated",
  },
  {
    name: "ThCr2Si2-pnictide",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 109.5, layerSpacing: 6.5, cOverA: 3.2,
      symmetryOrder: 16, electronCountPerSite: 6, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.05, connectivityIndex: 0.78,
      anisotropy: 0.82, voidFraction: 0.1,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "spacer", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "T", multiplicity: 4, position: [0, 0.5, 0.25], role: "plane", preferredCategories: ["magneticTM", "highCouplingTM"] },
      { label: "X", multiplicity: 4, position: [0, 0, 0.35], role: "plane", preferredCategories: ["pnictogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.2 },
    scAffinity: 0.84,
    tcRange: [15, 60],
    pairingMechanism: "s+- spin fluctuation",
  },
  {
    name: "anti-perovskite-SC",
    spaceGroup: "Pm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 3, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0.1, connectivityIndex: 0.92,
      anisotropy: 0, voidFraction: 0.08,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0.5, 0.5, 0.5], role: "framework", preferredCategories: ["lightPhonon", "anion"] },
      { label: "B", multiplicity: 1, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM"] },
      { label: "X", multiplicity: 3, position: [0.5, 0, 0], role: "framework", preferredCategories: ["highCouplingTM", "pBlockMetal"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.68,
    tcRange: [3, 20],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "Chevrel-phase",
    spaceGroup: "R-3",
    crystalSystem: "trigonal",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 6, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0.4, interstitialFraction: 0.15, connectivityIndex: 0.75,
      anisotropy: 0.15, voidFraction: 0.2,
    },
    siteRoles: [
      { label: "M", multiplicity: 1, position: [0, 0, 0], role: "interstitial", preferredCategories: ["alkalineEarth", "rareEarth", "pBlockMetal"] },
      { label: "Mo", multiplicity: 6, position: [0.2, 0.2, 0.2], role: "cage", preferredCategories: ["highCouplingTM"] },
      { label: "X", multiplicity: 8, position: [0.35, 0.35, 0.05], role: "cage", preferredCategories: ["chalcogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.73,
    tcRange: [5, 20],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "BiS2-layer",
    spaceGroup: "P4/nmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 90, layerSpacing: 6.8, cOverA: 2.8,
      symmetryOrder: 8, electronCountPerSite: 5, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.08, connectivityIndex: 0.68,
      anisotropy: 0.88, voidFraction: 0.18,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "spacer", preferredCategories: ["rareEarth"] },
      { label: "O", multiplicity: 2, position: [0, 0.5, 0.1], role: "spacer", preferredCategories: ["anion"] },
      { label: "Bi", multiplicity: 2, position: [0, 0, 0.35], role: "plane", preferredCategories: ["pBlockMetal"] },
      { label: "S", multiplicity: 4, position: [0, 0.5, 0.4], role: "plane", preferredCategories: ["chalcogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 2.8 },
    scAffinity: 0.60,
    tcRange: [2, 12],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "infinite-layer",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 90, layerSpacing: 3.4, cOverA: 0.85,
      symmetryOrder: 16, electronCountPerSite: 9, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.9,
      anisotropy: 0.7, voidFraction: 0.05,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0.5, 0.5, 0.5], role: "reservoir", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B", multiplicity: 1, position: [0, 0, 0], role: "plane", preferredCategories: ["magneticTM"] },
      { label: "O", multiplicity: 2, position: [0.5, 0, 0], role: "plane", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 0.85 },
    scAffinity: 0.80,
    tcRange: [10, 80],
    pairingMechanism: "d-wave spin fluctuation",
  },
  {
    name: "skutterudite",
    spaceGroup: "Im-3",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 24, electronCountPerSite: 5, dimensionality: 3,
      cageFraction: 0.55, interstitialFraction: 0.12, connectivityIndex: 0.82,
      anisotropy: 0, voidFraction: 0.22,
    },
    siteRoles: [
      { label: "M", multiplicity: 8, position: [0.25, 0.25, 0.25], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 24, position: [0, 0.34, 0.16], role: "framework", preferredCategories: ["pnictogen"] },
      { label: "G", multiplicity: 2, position: [0, 0, 0], role: "cage", preferredCategories: ["rareEarth", "alkalineEarth"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.58,
    tcRange: [1, 10],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "Heusler-L21",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 8, avgBondAngle: 109.5, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 5, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.88,
      anisotropy: 0, voidFraction: 0.05,
    },
    siteRoles: [
      { label: "A", multiplicity: 8, position: [0.25, 0.25, 0.25], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "B", multiplicity: 4, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM"] },
      { label: "C", multiplicity: 4, position: [0.5, 0.5, 0.5], role: "framework", preferredCategories: ["pBlockMetal", "alkalineEarth"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.55,
    tcRange: [1, 15],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "pyrochlore",
    spaceGroup: "Fd-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 109.5, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0.3, interstitialFraction: 0.08, connectivityIndex: 0.72,
      anisotropy: 0, voidFraction: 0.15,
    },
    siteRoles: [
      { label: "A", multiplicity: 16, position: [0.5, 0.5, 0.5], role: "cage", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 16, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "O1", multiplicity: 48, position: [0.33, 0.125, 0.125], role: "framework", preferredCategories: ["anion"] },
      { label: "O2", multiplicity: 8, position: [0.375, 0.375, 0.375], role: "interstitial", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.56,
    tcRange: [0.5, 8],
    pairingMechanism: "geometrically frustrated",
  },
  {
    name: "1T-prime-TMD",
    spaceGroup: "P2_1/m",
    crystalSystem: "monoclinic",
    embedding: {
      coordination: 6, avgBondAngle: 85, layerSpacing: 6.4, cOverA: 3.4,
      symmetryOrder: 4, electronCountPerSite: 4, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.72,
      anisotropy: 0.9, voidFraction: 0.18,
    },
    siteRoles: [
      { label: "M", multiplicity: 2, position: [0.25, 0.25, 0], role: "plane", preferredCategories: ["highCouplingTM"] },
      { label: "X1", multiplicity: 2, position: [0.1, 0.6, 0.1], role: "plane", preferredCategories: ["chalcogen"] },
      { label: "X2", multiplicity: 2, position: [0.4, 0.6, 0.9], role: "plane", preferredCategories: ["chalcogen"] },
    ],
    latticeRatios: { a: 1, b: 1.73, c: 3.4 },
    scAffinity: 0.70,
    tcRange: [1, 10],
    pairingMechanism: "topological-enhanced phonon",
  },
  {
    name: "Ruddlesden-Popper",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 12.5, cOverA: 6.3,
      symmetryOrder: 16, electronCountPerSite: 5, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.05, connectivityIndex: 0.75,
      anisotropy: 0.92, voidFraction: 0.1,
    },
    siteRoles: [
      { label: "A1", multiplicity: 2, position: [0, 0, 0.5], role: "spacer", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "A2", multiplicity: 2, position: [0, 0, 0.18], role: "reservoir", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B", multiplicity: 2, position: [0, 0, 0], role: "plane", preferredCategories: ["magneticTM", "highCouplingTM"] },
      { label: "O1", multiplicity: 4, position: [0.5, 0, 0], role: "plane", preferredCategories: ["anion"] },
      { label: "O2", multiplicity: 4, position: [0, 0, 0.09], role: "spacer", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 6.3 },
    scAffinity: 0.78,
    tcRange: [15, 100],
    pairingMechanism: "d-wave spin fluctuation",
  },
  {
    name: "nickelate-IL",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 90, layerSpacing: 3.3, cOverA: 0.85,
      symmetryOrder: 16, electronCountPerSite: 8, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.88,
      anisotropy: 0.72, voidFraction: 0.06,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0.5, 0.5, 0.5], role: "reservoir", preferredCategories: ["rareEarth"] },
      { label: "Ni", multiplicity: 1, position: [0, 0, 0], role: "plane", preferredCategories: ["magneticTM"] },
      { label: "O", multiplicity: 2, position: [0.5, 0, 0], role: "plane", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 0.85 },
    scAffinity: 0.82,
    tcRange: [8, 30],
    pairingMechanism: "d-wave spin fluctuation",
  },
  {
    name: "MgB2-sigma",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 6, avgBondAngle: 120, layerSpacing: 3.5, cOverA: 1.14,
      symmetryOrder: 24, electronCountPerSite: 3, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.05, connectivityIndex: 0.92,
      anisotropy: 0.55, voidFraction: 0.12,
    },
    siteRoles: [
      { label: "M", multiplicity: 1, position: [0, 0, 0], role: "spacer", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B1", multiplicity: 2, position: [0.333, 0.667, 0.5], role: "plane", preferredCategories: ["lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.14 },
    scAffinity: 0.88,
    tcRange: [20, 60],
    pairingMechanism: "sigma-band phonon-mediated",
  },
];

interface ChemicalFamily {
  name: string;
  hostElements: string[];
  allowedAnions: string[];
  allowedStoichiometries: string[];
  compatibleMotifs: string[];
  maxAtomRatio: Record<string, number>;
  electronCountRange: [number, number];
  description: string;
}

const CHEMICAL_FAMILIES: ChemicalFamily[] = [
  {
    name: "hydride",
    hostElements: ["La", "Y", "Ce", "Ca", "Sr", "Ba", "Sc", "Th", "Ac", "Nd", "Gd"],
    allowedAnions: ["H"],
    allowedStoichiometries: ["AH3", "AH4", "AH6", "AH10", "ABH4", "ABH6"],
    compatibleMotifs: ["clathrate-cage", "layered-hydride", "H-channel"],
    maxAtomRatio: { H: 6 },
    electronCountRange: [1, 6],
    description: "High-Tc hydrogen clathrate and layered hydride family",
  },
  {
    name: "intermetallic",
    hostElements: ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "Re", "Cr"],
    allowedAnions: ["Sn", "Ge", "Al", "Ga", "In", "Si"],
    allowedStoichiometries: ["A3B", "AB", "AB2", "A2B"],
    compatibleMotifs: ["A15-chain", "Laves-MgZn2", "Heusler-L21", "NaCl-rocksalt"],
    maxAtomRatio: {},
    electronCountRange: [3, 10],
    description: "Classical A15 and Laves-phase intermetallics",
  },
  {
    name: "layered-pnictide",
    hostElements: ["Ba", "Sr", "Ca", "K", "Rb", "Cs", "La", "Ce", "Nd", "Sm"],
    allowedAnions: ["As", "P", "Se", "Te", "S"],
    allowedStoichiometries: ["AB2C2", "ABC3", "A2BC", "AB2"],
    compatibleMotifs: ["FeAs-layer", "ThCr2Si2-pnictide", "BiS2-layer"],
    maxAtomRatio: {},
    electronCountRange: [4, 8],
    description: "Iron-based and pnictide layered superconductors",
  },
  {
    name: "boride",
    hostElements: ["Mg", "Ca", "Sr", "Ba", "Al", "Y", "Sc", "La", "Ti", "Zr"],
    allowedAnions: ["B", "C"],
    allowedStoichiometries: ["AB2", "AB", "A2B3", "AB3"],
    compatibleMotifs: ["hexagonal-layer", "MgB2-sigma", "borocarbide-RTBC"],
    maxAtomRatio: { B: 4, C: 2 },
    electronCountRange: [2, 6],
    description: "MgB2-type borides and borocarbides",
  },
  {
    name: "cuprate",
    hostElements: ["La", "Y", "Bi", "Tl", "Hg", "Ba", "Sr", "Ca", "Nd", "Gd"],
    allowedAnions: ["O", "F"],
    allowedStoichiometries: ["A2BO4", "ABC3O7", "A2B2C3O10"],
    compatibleMotifs: ["CuO2-plane", "infinite-layer", "Ruddlesden-Popper", "nickelate-IL"],
    maxAtomRatio: {},
    electronCountRange: [7, 11],
    description: "Cuprate and nickelate high-Tc superconductors",
  },
  {
    name: "chalcogenide",
    hostElements: ["Nb", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "V", "Fe"],
    allowedAnions: ["S", "Se", "Te"],
    allowedStoichiometries: ["AB2", "AB", "A2B3"],
    compatibleMotifs: ["TMD-2H", "1T-prime-TMD", "Chevrel-phase", "NaCl-rocksalt"],
    maxAtomRatio: {},
    electronCountRange: [3, 8],
    description: "Transition metal dichalcogenides and Chevrel phases",
  },
  {
    name: "kagome-metal",
    hostElements: ["V", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Nb"],
    allowedAnions: ["Sb", "Sn", "Ge", "As", "Bi"],
    allowedStoichiometries: ["AB3C5", "AB3C2", "AB2"],
    compatibleMotifs: ["kagome-flat", "breathing-kagome"],
    maxAtomRatio: {},
    electronCountRange: [3, 7],
    description: "Kagome lattice metals with flat bands",
  },
  {
    name: "oxide-perovskite",
    hostElements: ["Sr", "Ba", "Ca", "La", "Y", "Bi", "Pb", "K", "Na"],
    allowedAnions: ["O", "F", "N"],
    allowedStoichiometries: ["ABO3", "A2BO4", "AB2O6"],
    compatibleMotifs: ["perovskite-3D", "anti-perovskite-SC", "Ruddlesden-Popper", "pyrochlore"],
    maxAtomRatio: {},
    electronCountRange: [2, 8],
    description: "Perovskite and anti-perovskite oxide superconductors",
  },
];

const VALENCE_ELECTRONS: Record<string, number> = {
  H: 1, He: 2, Li: 1, Be: 2, B: 3, C: 4, N: 5, O: 6, F: 7,
  Na: 1, Mg: 2, Al: 3, Si: 4, P: 5, S: 6, Cl: 7,
  K: 1, Ca: 2, Sc: 3, Ti: 4, V: 5, Cr: 6, Mn: 7, Fe: 8, Co: 9, Ni: 10, Cu: 11, Zn: 12,
  Ga: 3, Ge: 4, As: 5, Se: 6, Br: 7,
  Rb: 1, Sr: 2, Y: 3, Zr: 4, Nb: 5, Mo: 6, Ru: 8, Rh: 9, Pd: 10, Ag: 11,
  In: 3, Sn: 4, Sb: 5, Te: 6,
  Cs: 1, Ba: 2, La: 3, Ce: 4, Pr: 5, Nd: 6, Sm: 8, Gd: 10,
  Hf: 4, Ta: 5, W: 6, Re: 7, Os: 8, Ir: 9, Pt: 10, Au: 11,
  Tl: 3, Pb: 4, Bi: 5,
  Ac: 3, Th: 4,
  Dy: 12, Er: 14, Yb: 16, Lu: 17,
};

const TM_ELEMENTS = new Set([
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
]);

function parseFormulaElements(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*)/g;
  let m;
  while ((m = regex.exec(formula)) !== null) {
    if (!m[1]) continue;
    const el = m[1];
    const n = m[2] ? parseInt(m[2], 10) : 1;
    counts[el] = (counts[el] || 0) + n;
  }
  return counts;
}

function validateElectronCount(formula: string, motifName?: string): { valid: boolean; reason: string; vec: number } {
  const counts = parseFormulaElements(formula);
  const elements = Object.keys(counts);
  if (elements.length === 0) return { valid: false, reason: "empty formula", vec: 0 };

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  let totalVE = 0;
  for (const [el, n] of Object.entries(counts)) {
    totalVE += (VALENCE_ELECTRONS[el] ?? 4) * n;
  }
  const avgVEC = totalVE / totalAtoms;

  const tmElements = elements.filter(e => TM_ELEMENTS.has(e));
  const tmCount = tmElements.reduce((s, e) => s + (counts[e] || 0), 0);

  if (tmCount > 0 && totalVE / tmCount > 12) {
    return { valid: false, reason: "VEC/TM ratio exceeds 12", vec: avgVEC };
  }

  if (motifName) {
    if ((motifName === "CuO2-plane" || motifName === "infinite-layer" || motifName === "nickelate-IL") && tmCount > 0) {
      for (const tm of tmElements) {
        const ve = VALENCE_ELECTRONS[tm] ?? 4;
        if (ve < 7 || ve > 11) {
          return { valid: false, reason: `${motifName} requires d7-d11 TM, ${tm} has VE=${ve}`, vec: avgVEC };
        }
      }
    }

    if ((motifName === "FeAs-layer" || motifName === "ThCr2Si2-pnictide") && tmCount > 0) {
      for (const tm of tmElements) {
        const ve = VALENCE_ELECTRONS[tm] ?? 4;
        if (ve < 4 || ve > 10) {
          return { valid: false, reason: `${motifName} requires d4-d10 TM, ${tm} has VE=${ve}`, vec: avgVEC };
        }
      }
    }

    if (motifName === "A15-chain" && tmCount > 0) {
      for (const tm of tmElements) {
        const ve = VALENCE_ELECTRONS[tm] ?? 4;
        if (ve < 4 || ve > 7) {
          return { valid: false, reason: `A15 requires d4-d7 TM, ${tm} has VE=${ve}`, vec: avgVEC };
        }
      }
    }
  }

  if (avgVEC > 14) {
    return { valid: false, reason: "average VEC too high (>14)", vec: avgVEC };
  }

  return { valid: true, reason: "passed", vec: avgVEC };
}

function selectFamilyForMotif(motifName: string): ChemicalFamily | null {
  for (const fam of CHEMICAL_FAMILIES) {
    if (fam.compatibleMotifs.includes(motifName)) return fam;
  }
  return null;
}

function selectElementsForSiteConstrained(
  site: SiteRole,
  family: ChemicalFamily | null,
): string[] {
  const candidates: string[] = [];
  for (const cat of site.preferredCategories) {
    const pool = ELEMENT_CATEGORIES[cat];
    if (pool) candidates.push(...pool);
  }

  let filtered = Array.from(new Set(candidates));

  if (family) {
    const familyAllowed = new Set([...family.hostElements, ...family.allowedAnions]);
    const familyFiltered = filtered.filter(e => familyAllowed.has(e));
    if (familyFiltered.length >= 2) {
      filtered = familyFiltered;
    }
  }

  const shuffled = filtered.sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(4, shuffled.length));
}

function checkFamilyConstraints(formula: string, family: ChemicalFamily | null): boolean {
  if (!family) return true;
  const counts = parseFormulaElements(formula);

  for (const [el, maxR] of Object.entries(family.maxAtomRatio)) {
    const elCount = counts[el] || 0;
    const nonElCount = Object.entries(counts)
      .filter(([k]) => k !== el)
      .reduce((s, [, v]) => s + v, 0);
    if (nonElCount > 0 && elCount / nonElCount > maxR) return false;
  }

  return true;
}

export { SC_MOTIF_LIBRARY, CHEMICAL_FAMILIES, VALENCE_ELECTRONS, validateElectronCount, parseFormulaElements, selectFamilyForMotif };
export type { StructuralMotif, ChemicalFamily };

const stats: StructureDiffusionStats = {
  totalMotifGenerated: 0,
  totalCandidatesEvaluated: 0,
  bestTcAchieved: 0,
  bestFormula: "",
  bestMotif: "",
  motifSuccessRates: {},
  recentResults: [],
  learnedWeights: {},
};

for (const m of SC_MOTIF_LIBRARY) {
  stats.learnedWeights[m.name] = m.scAffinity;
}

function computeStructuralScore(embedding: StructuralEmbedding, targetTc: number): number {
  let score = 0;

  if (targetTc > 100) {
    score += embedding.cageFraction * 0.3;
    score += (embedding.coordination > 8 ? 0.2 : 0);
    score += embedding.interstitialFraction * 0.25;
  }

  if (targetTc > 30 && targetTc <= 100) {
    score += (embedding.dimensionality <= 2 ? 0.3 : 0.1);
    score += (embedding.anisotropy > 0.5 ? 0.2 : 0);
    score += (embedding.connectivityIndex > 0.7 ? 0.15 : 0);
  }

  if (targetTc <= 30) {
    score += (embedding.coordination >= 6 ? 0.2 : 0);
    score += (embedding.symmetryOrder > 20 ? 0.15 : 0);
    score += (1 - embedding.voidFraction) * 0.1;
  }

  score += embedding.connectivityIndex * 0.15;
  score += Math.min(1, embedding.symmetryOrder / 48) * 0.1;

  return Math.min(1, score);
}

function selectElementsForSite(site: SiteRole, triedCombos: Set<string>): string[] {
  const candidates: string[] = [];
  for (const cat of site.preferredCategories) {
    const pool = ELEMENT_CATEGORIES[cat];
    if (pool) candidates.push(...pool);
  }

  const unique = Array.from(new Set(candidates));
  const shuffled = unique.sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(4, shuffled.length));
}

function buildFormula(assignment: Record<string, string>, motif: StructuralMotif): string {
  const counts: Record<string, number> = {};
  for (const site of motif.siteRoles) {
    const el = assignment[site.label];
    if (!el) continue;
    counts[el] = (counts[el] || 0) + site.multiplicity;
  }

  const gcd = Object.values(counts).reduce((a, b) => {
    while (b) { [a, b] = [b, a % b]; }
    return a;
  });

  const sorted = Object.entries(counts).sort((a, b) => {
    const aIsM = ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "Fe", "Co", "Ni", "Cu", "Mn", "La", "Y", "Ce", "Ca", "Sr", "Ba", "Sc"].includes(a[0]);
    const bIsM = ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "Fe", "Co", "Ni", "Cu", "Mn", "La", "Y", "Ce", "Ca", "Sr", "Ba", "Sc"].includes(b[0]);
    if (aIsM !== bIsM) return aIsM ? -1 : 1;
    return a[0].localeCompare(b[0]);
  });

  return sorted.map(([el, c]) => {
    const reduced = c / gcd;
    return reduced === 1 ? el : `${el}${reduced}`;
  }).join("");
}

function evaluateCandidate(formula: string): { tc: number; lambda: number; gbTc: number; score: number } | null {
  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    const omegaLogK = coupling.omegaLog * 1.4388;
    const denom = coupling.lambda - coupling.muStar * (1 + 0.62 * coupling.lambda);
    let tc = 0;
    if (Math.abs(denom) > 1e-6 && denom > 0 && coupling.lambda > 0.2) {
      const lambdaBar = 2.46 * (1 + 3.8 * coupling.muStar);
      const f1 = Math.pow(1 + Math.pow(coupling.lambda / lambdaBar, 3 / 2), 1 / 3);
      const exponent = -1.04 * (1 + coupling.lambda) / denom;
      tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
      if (!Number.isFinite(tc) || tc < 0) tc = 0;
    }
    if (electronic.metallicity < 0.4) {
      tc = tc * Math.max(0.02, electronic.metallicity);
    }
    tc = Math.min(400, tc);

    let gbTc = 0;
    let gbScore = 0;
    try {
      const features = extractFeatures(formula);
      if (features) {
        const gb = gbPredict(features);
        gbTc = gb.tcPredicted;
        gbScore = gb.score;
      }
    } catch {}

    return {
      tc: Math.max(tc, gbTc * 0.3),
      lambda: coupling.lambda,
      gbTc,
      score: gbScore,
    };
  } catch {
    return null;
  }
}

function perturbEmbedding(base: StructuralEmbedding, intensity: number): StructuralEmbedding {
  const perturb = (val: number, range: number) => {
    return Math.max(0, val + (Math.random() - 0.5) * range * intensity);
  };

  return {
    coordination: Math.round(perturb(base.coordination, 4)),
    avgBondAngle: perturb(base.avgBondAngle, 30),
    layerSpacing: perturb(base.layerSpacing, 3),
    cOverA: Math.max(0.5, perturb(base.cOverA, 1)),
    symmetryOrder: base.symmetryOrder,
    electronCountPerSite: perturb(base.electronCountPerSite, 3),
    dimensionality: base.dimensionality,
    cageFraction: Math.min(1, Math.max(0, perturb(base.cageFraction, 0.3))),
    interstitialFraction: Math.min(1, Math.max(0, perturb(base.interstitialFraction, 0.3))),
    connectivityIndex: Math.min(1, Math.max(0, perturb(base.connectivityIndex, 0.2))),
    anisotropy: Math.min(1, Math.max(0, perturb(base.anisotropy, 0.3))),
    voidFraction: Math.min(1, Math.max(0, perturb(base.voidFraction, 0.2))),
  };
}

function selectMotifs(targetTc: number, count: number): StructuralMotif[] {
  const EPSILON = 0.15;
  const MIN_EXPLORATION_RATE = 0.05;
  const MAX_MOTIF_FRACTION = 0.40;

  const scored = SC_MOTIF_LIBRARY.map(m => {
    const tcFit = m.tcRange[0] <= targetTc && targetTc <= m.tcRange[1] * 1.5 ? 1.0 :
      Math.max(0, 1.0 - Math.abs(targetTc - (m.tcRange[0] + m.tcRange[1]) / 2) / (m.tcRange[1] - m.tcRange[0] + 50));
    const learned = stats.learnedWeights[m.name] ?? m.scAffinity;
    return { motif: m, score: tcFit * 0.5 + learned * 0.3 + m.scAffinity * 0.2 };
  });

  const selected: StructuralMotif[] = [];
  const motifCounts: Record<string, number> = {};
  const maxPerMotif = Math.max(1, Math.ceil(count * MAX_MOTIF_FRACTION));

  for (let i = 0; i < count; i++) {
    let chosen: StructuralMotif;

    if (Math.random() < EPSILON) {
      chosen = SC_MOTIF_LIBRARY[Math.floor(Math.random() * SC_MOTIF_LIBRARY.length)];
    } else {
      const weights = scored.map(s => {
        const currentCount = motifCounts[s.motif.name] || 0;
        if (currentCount >= maxPerMotif) return 0;
        return Math.max(MIN_EXPLORATION_RATE, s.score);
      });
      const totalWeight = weights.reduce((s, w) => s + w, 0);
      if (totalWeight <= 0) {
        chosen = SC_MOTIF_LIBRARY[Math.floor(Math.random() * SC_MOTIF_LIBRARY.length)];
      } else {
        const normalized = weights.map(w => w / totalWeight);
        let r = Math.random();
        let idx = 0;
        for (let j = 0; j < normalized.length; j++) {
          r -= normalized[j];
          if (r <= 0) { idx = j; break; }
        }
        chosen = scored[idx].motif;
      }
    }

    selected.push(chosen);
    motifCounts[chosen.name] = (motifCounts[chosen.name] || 0) + 1;
  }

  return selected;
}

export function runStructureFirstDesign(
  targetTc: number,
  motifsToTry: number = 4,
  elementsPerSite: number = 3,
): StructureDiffusionResult[] {
  const motifs = selectMotifs(targetTc, motifsToTry);
  const results: StructureDiffusionResult[] = [];

  for (const motif of motifs) {
    stats.totalMotifGenerated++;
    const candidates: StructureCandidate[] = [];
    const triedCombos = new Set<string>();
    const elementsTriedPerSite: Record<string, string[]> = {};

    const family = selectFamilyForMotif(motif.name);

    const siteLists: Record<string, string[]> = {};
    for (const site of motif.siteRoles) {
      siteLists[site.label] = selectElementsForSiteConstrained(site, family);
      elementsTriedPerSite[site.label] = siteLists[site.label];
    }

    const assignments: Record<string, string>[] = [];
    generateAssignments(motif.siteRoles, siteLists, 0, {}, assignments, elementsPerSite * 8);

    for (const assignment of assignments) {
      const rawFormula = buildFormula(assignment, motif);
      if (!rawFormula || rawFormula.length < 2) continue;
      const formula = normalizeFormula(rawFormula);
      if (triedCombos.has(formula)) continue;
      if (!passesValenceFilter(formula)) continue;
      triedCombos.add(formula);

      if (!checkFamilyConstraints(formula, family)) continue;

      const vecCheck = validateElectronCount(formula, motif.name);
      if (!vecCheck.valid) continue;

      const result = evaluateCandidate(formula);
      if (!result) continue;

      stats.totalCandidatesEvaluated++;

      const embedding = perturbEmbedding(motif.embedding, 0.1);
      const structuralScore = computeStructuralScore(embedding, targetTc);

      candidates.push({
        motif: motif.name,
        formula,
        siteAssignment: assignment,
        structuralScore,
        predictedTc: Math.round(result.tc * 10) / 10,
        lambda: Math.round(result.lambda * 1000) / 1000,
        embedding,
      });
    }

    candidates.sort((a, b) => b.predictedTc - a.predictedTc);
    const best = candidates[0];

    if (!stats.motifSuccessRates[motif.name]) {
      stats.motifSuccessRates[motif.name] = { tried: 0, passed: 0, avgTc: 0 };
    }
    const mStats = stats.motifSuccessRates[motif.name];
    mStats.tried += candidates.length;
    mStats.passed += candidates.filter(c => c.predictedTc > 5).length;
    const totalTc = candidates.reduce((s, c) => s + c.predictedTc, 0);
    const rawAvgTc = candidates.length > 0 ? totalTc / candidates.length : 0;
    mStats.avgTc = Number.isFinite(rawAvgTc) ? rawAvgTc : 0;

    if (best) {
      if (best.predictedTc > stats.bestTcAchieved) {
        stats.bestTcAchieved = best.predictedTc;
        stats.bestFormula = best.formula;
        stats.bestMotif = motif.name;
      }

      stats.recentResults.push({
        formula: best.formula,
        tc: best.predictedTc,
        motif: motif.name,
      });
      if (stats.recentResults.length > 30) stats.recentResults.shift();

      const reward = best.predictedTc / Math.max(1, targetTc);
      const currentW = stats.learnedWeights[motif.name] ?? motif.scAffinity;
      let newWeight = currentW * 0.9 + reward * 0.1;

      const allWeights = Object.values(stats.learnedWeights);
      const meanWeight = allWeights.length > 0 ? allWeights.reduce((s, w) => s + w, 0) / allWeights.length : 1.0;
      if (newWeight > meanWeight * 3) {
        newWeight = newWeight * 0.85;
      }
      stats.learnedWeights[motif.name] = newWeight;
    }

    results.push({
      candidates: candidates.slice(0, 10),
      motifUsed: motif.name,
      elementsTriedPerSite,
      bestFormula: best?.formula ?? "",
      bestTc: best?.predictedTc ?? 0,
      totalEvaluated: candidates.length,
    });
  }

  return results;
}

function generateAssignments(
  sites: SiteRole[],
  elementLists: Record<string, string[]>,
  index: number,
  current: Record<string, string>,
  results: Record<string, string>[],
  maxResults: number,
): void {
  if (results.length >= maxResults) return;
  if (index >= sites.length) {
    results.push({ ...current });
    return;
  }

  const site = sites[index];
  const elements = elementLists[site.label] || [];

  for (const el of elements) {
    const alreadyUsedForDiffRole = Object.entries(current).some(([label, assignedEl]) => {
      if (assignedEl !== el) return false;
      const otherSite = sites.find(s => s.label === label);
      return otherSite && otherSite.role !== site.role;
    });
    if (alreadyUsedForDiffRole && Math.random() > 0.3) continue;

    current[site.label] = el;
    generateAssignments(sites, elementLists, index + 1, current, results, maxResults);
  }
}

export function runStructureDiffusionCycle(
  targetTc: number = 200,
  motifCount: number = 3,
  elementsPerSite: number = 3,
): { formulas: string[]; bestFormula: string; bestTc: number; motifsUsed: string[] } {
  const results = runStructureFirstDesign(targetTc, motifCount, elementsPerSite);

  const allFormulas: string[] = [];
  let bestTc = 0;
  let bestFormula = "";
  const motifsUsed: string[] = [];

  for (const r of results) {
    motifsUsed.push(r.motifUsed);
    for (const c of r.candidates) {
      allFormulas.push(c.formula);
      if (c.predictedTc > bestTc) {
        bestTc = c.predictedTc;
        bestFormula = c.formula;
      }
    }
  }

  return { formulas: allFormulas, bestFormula, bestTc, motifsUsed };
}

export function getStructureDiffusionStats(): StructureDiffusionStats {
  return { ...stats, motifSuccessRates: { ...stats.motifSuccessRates } };
}

export function getMotifLibrarySummary(): { name: string; spaceGroup: string; tcRange: [number, number]; scAffinity: number; pairingMechanism: string; learnedWeight: number }[] {
  return SC_MOTIF_LIBRARY.map(m => ({
    name: m.name,
    spaceGroup: m.spaceGroup,
    tcRange: m.tcRange,
    scAffinity: m.scAffinity,
    pairingMechanism: m.pairingMechanism,
    learnedWeight: stats.learnedWeights[m.name] ?? m.scAffinity,
  }));
}
