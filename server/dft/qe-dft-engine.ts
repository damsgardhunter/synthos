import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { getElementData } from "../learning/elemental-data";
import { fillPrototype, computeBondValenceSum, checkIonicRadiusCompatibility } from "../learning/crystal-prototypes";
import { computeFiniteDisplacementPhonons } from "./phonon-calculator";
import type { FiniteDisplacementPhononResult } from "./phonon-calculator";
import { analyzeDistortion, recordDistortionAnalysis, type DistortionAnalysis } from "../crystal/distortion-detector";
import { relaxStructureAtPressure } from "../learning/pressure-engine";

const PROJECT_ROOT = path.resolve(process.cwd());
const XTB_BIN = path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb");
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = "/tmp/dft_calculations";
const TIMEOUT_MS = 60_000;

export interface OptimizationResult {
  optimizedAtoms: AtomPosition[];
  optimizedEnergy: number;
  converged: boolean;
  energyChange: number;
  gradientNorm: number;
  iterations: number;
  wallTimeSeconds: number;
  distortion?: DistortionAnalysis;
}

export interface DFTResult {
  formula: string;
  method: "GFN2-xTB";
  prototype: string;
  totalEnergy: number;
  totalEnergyPerAtom: number;
  homoLumoGap: number;
  isMetallic: boolean;
  homo: number | null;
  lumo: number | null;
  fermiLevel: number | null;
  dipoleMoment: number | null;
  charges: Record<string, number>;
  converged: boolean;
  wallTimeSeconds: number;
  atomCount: number;
  error: string | null;
  optimized: boolean;
}

export interface PhononStability {
  hasImaginaryModes: boolean;
  imaginaryModeCount: number;
  lowestFrequency: number;
  frequencies: number[];
  zeroPointEnergy: number | null;
}

export interface XTBEnrichedFeatures {
  bandGap: number;
  isMetallic: boolean;
  totalEnergy: number;
  totalEnergyPerAtom: number;
  formationEnergyPerAtom: number | null;
  fermiLevel: number | null;
  converged: boolean;
  prototype: string;
  method: string;
  phononStability: PhononStability | null;
  finiteDisplacementPhonons: FiniteDisplacementPhononResult | null;
}

interface AtomPosition {
  element: string;
  x: number;
  y: number;
  z: number;
}

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11,
  P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06, K: 2.03, Ca: 1.76, Sc: 1.70,
  Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24,
  Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20,
  Kr: 1.16, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54,
  Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39,
  Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36,
  Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Tc: 1.47,
  Pr: 2.03, Nd: 2.01, Sm: 1.98, Eu: 1.98, Gd: 1.96, Tb: 1.94,
  Dy: 1.92, Ho: 1.92, Er: 1.89, Tm: 1.90, Yb: 1.87, Lu: 1.87,
  Th: 2.06, U: 1.96, Pa: 2.00, Np: 1.90, Pu: 1.87,
};

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

function getAvgRadius(el: string): number {
  return COVALENT_RADII[el] || (getElementData(el)?.atomicRadius ?? 150) / 100;
}

const PROTOTYPE_PACKING: Record<string, number> = {
  "A15": 0.68, "NaCl": 0.74, "AlB2": 0.74, "Perovskite": 0.74,
  "ThCr2Si2": 0.68, "Heusler": 0.74, "BCC": 0.68, "FCC": 0.74,
  "Layered": 0.60, "Kagome": 0.60, "HexBoride": 0.74, "MX2": 0.60,
  "Anti-perovskite": 0.74, "CsCl": 0.68, "Cu2Mg-Laves": 0.74,
  "Fluorite": 0.74, "Cr3Si": 0.68, "Ni3Sn": 0.74, "Fe3C": 0.68,
  "Spinel": 0.74, "Clathrate-H32": 0.55, "Skutterudite": 0.68,
  "BiS2-layered": 0.60, "Kagome-variant": 0.60, "Chevrel": 0.65,
  "Pyrite": 0.68, "Wurtzite": 0.74, "Antifluorite": 0.74,
  "Laves-C14": 0.71, "Laves-C15": 0.71, "HfFe6Ge6": 0.68,
  "CeCu2Si2": 0.68, "PuCoGa5-115": 0.68, "Infinite-layer": 0.60,
  "T-prime": 0.60, "Ruddlesden-Popper": 0.65, "Double-perovskite": 0.74,
  "Garnet": 0.68,
};

const ATOMIC_VOLUMES: Record<string, number> = {
  H: 3.0, He: 4.0, Li: 21.0, Be: 8.0, B: 7.5, C: 12.0, N: 13.0,
  O: 14.0, F: 11.0, Ne: 13.0, Na: 24.0, Mg: 23.0, Al: 16.6, Si: 20.0,
  P: 17.0, S: 15.5, Cl: 22.0, Ar: 24.0, K: 45.5, Ca: 26.0, Sc: 25.0,
  Ti: 17.6, V: 14.0, Cr: 12.0, Mn: 12.2, Fe: 11.8, Co: 11.0, Ni: 10.9,
  Cu: 11.8, Zn: 15.2, Ga: 19.6, Ge: 22.6, As: 21.4, Se: 16.4, Br: 23.5,
  Kr: 27.0, Rb: 56.0, Sr: 34.0, Y: 33.0, Zr: 23.3, Nb: 18.0, Mo: 15.6,
  Ru: 13.6, Rh: 13.7, Pd: 14.7, Ag: 17.1, Cd: 21.6, In: 26.2, Sn: 27.3,
  Sb: 30.3, Te: 33.8, I: 25.7, Cs: 70.0, Ba: 37.0, La: 37.0, Ce: 34.4,
  Hf: 22.3, Ta: 18.0, W: 15.8, Re: 14.7, Os: 14.0, Ir: 14.2, Pt: 15.1,
  Au: 17.0, Hg: 23.4, Tl: 28.6, Pb: 30.3, Bi: 35.4, Th: 33.0, U: 20.8,
  Tc: 14.3,
};

function computeExpectedVolume(counts: Record<string, number>, packingFactor: number = 1.0): number {
  let totalVolume = 0;
  for (const [el, count] of Object.entries(counts)) {
    const vol = ATOMIC_VOLUMES[el] ?? 15.0;
    totalVolume += vol * Math.round(count);
  }
  return totalVolume * packingFactor;
}

function validateVolumeRatio(generatedVolume: number, expectedVolume: number): { valid: boolean; ratio: number } {
  if (expectedVolume <= 0) return { valid: true, ratio: 1.0 };
  const ratio = generatedVolume / expectedVolume;
  return { valid: ratio >= 0.5 && ratio <= 2.0, ratio };
}

function estimateLatticeParam(elements: string[], counts: Record<string, number>, protoName?: string): number {
  const packingFactor = protoName ? (PROTOTYPE_PACKING[protoName] ?? 0.68) : 0.68;
  let expectedVolume = computeExpectedVolume(counts, 1.0 / packingFactor);

  const totalAtoms = Object.values(counts).reduce((s, c) => s + Math.round(c), 0);
  const hasH = counts["H"] !== undefined && counts["H"] > 0;
  const minVolPerAtom = hasH ? 5.0 : 8.0;
  const minTotalVolume = totalAtoms * minVolPerAtom;
  if (expectedVolume < minTotalVolume) {
    expectedVolume = minTotalVolume;
  }

  const latticeA = Math.cbrt(expectedVolume);
  return Math.max(latticeA, 3.0);
}

interface PrototypeStructure {
  name: string;
  fractionalPositions: { site: string; x: number; y: number; z: number }[];
  latticeType: "cubic" | "hexagonal" | "tetragonal";
  aRatio: number;
  cOverA: number;
  stoichiometryPattern: string;
}

const CRYSTAL_PROTOTYPES: PrototypeStructure[] = [
  {
    name: "A15",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.25, y: 0.0, z: 0.5 },
      { site: "B", x: 0.75, y: 0.0, z: 0.5 },
      { site: "B", x: 0.5, y: 0.25, z: 0.0 },
      { site: "B", x: 0.5, y: 0.75, z: 0.0 },
      { site: "B", x: 0.0, y: 0.5, z: 0.25 },
      { site: "B", x: 0.0, y: 0.5, z: 0.75 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A2B6",
  },
  {
    name: "NaCl",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "B", x: 0.5, y: 0.0, z: 0.0 },
      { site: "B", x: 0.0, y: 0.5, z: 0.0 },
      { site: "B", x: 0.0, y: 0.0, z: 0.5 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4B4",
  },
  {
    name: "AlB2",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.333, y: 0.667, z: 0.5 },
      { site: "B", x: 0.667, y: 0.333, z: 0.5 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 1.08,
    stoichiometryPattern: "A1B2",
  },
  {
    name: "Perovskite",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
      { site: "C", x: 0.5, y: 0.5, z: 0.0 },
      { site: "C", x: 0.5, y: 0.0, z: 0.5 },
      { site: "C", x: 0.0, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A1B1C3",
  },
  {
    name: "ThCr2Si2",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.0, y: 0.5, z: 0.25 },
      { site: "B", x: 0.5, y: 0.0, z: 0.25 },
      { site: "B", x: 0.0, y: 0.5, z: 0.75 },
      { site: "B", x: 0.5, y: 0.0, z: 0.75 },
      { site: "C", x: 0.0, y: 0.0, z: 0.35 },
      { site: "C", x: 0.0, y: 0.0, z: 0.65 },
      { site: "C", x: 0.5, y: 0.5, z: 0.85 },
      { site: "C", x: 0.5, y: 0.5, z: 0.15 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 2.5,
    stoichiometryPattern: "A2B4C4",
  },
  {
    name: "Heusler",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.25, y: 0.25, z: 0.25 },
      { site: "B", x: 0.75, y: 0.75, z: 0.75 },
      { site: "C", x: 0.5, y: 0.0, z: 0.0 },
      { site: "C", x: 0.0, y: 0.5, z: 0.0 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A2B2C2",
  },
  {
    name: "BCC",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A2",
  },
  {
    name: "FCC",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4",
  },
  {
    name: "Layered",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.333, y: 0.667, z: 0.0 },
      { site: "A", x: 0.667, y: 0.333, z: 0.0 },
      { site: "B", x: 0.0, y: 0.0, z: 0.5 },
      { site: "B", x: 0.333, y: 0.667, z: 0.5 },
      { site: "B", x: 0.667, y: 0.333, z: 0.5 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 2.0,
    stoichiometryPattern: "A3B3",
  },
  {
    name: "Kagome",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.0, z: 0.25 },
      { site: "B", x: 0.0, y: 0.5, z: 0.25 },
      { site: "B", x: 0.5, y: 0.5, z: 0.25 },
      { site: "C", x: 0.333, y: 0.167, z: 0.5 },
      { site: "C", x: 0.167, y: 0.333, z: 0.5 },
      { site: "C", x: 0.667, y: 0.833, z: 0.5 },
      { site: "C", x: 0.833, y: 0.667, z: 0.5 },
      { site: "C", x: 0.667, y: 0.167, z: 0.5 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 2.7,
    stoichiometryPattern: "A1B3C5",
  },
  {
    name: "HexBoride",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.167, y: 0.333, z: 0.5 },
      { site: "B", x: 0.333, y: 0.167, z: 0.5 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.167, y: 0.833, z: 0.5 },
      { site: "B", x: 0.833, y: 0.167, z: 0.5 },
      { site: "B", x: 0.833, y: 0.667, z: 0.5 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 1.15,
    stoichiometryPattern: "A1B6",
  },
  {
    name: "MX2",
    fractionalPositions: [
      { site: "A", x: 0.333, y: 0.667, z: 0.25 },
      { site: "B", x: 0.333, y: 0.667, z: 0.621 },
      { site: "B", x: 0.333, y: 0.667, z: 0.879 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 3.9,
    stoichiometryPattern: "A1B2",
  },
  {
    name: "Anti-perovskite",
    fractionalPositions: [
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.0, y: 0.0, z: 0.0 },
      { site: "C", x: 0.5, y: 0.5, z: 0.0 },
      { site: "C", x: 0.5, y: 0.0, z: 0.5 },
      { site: "C", x: 0.0, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A1B1C3",
  },
  {
    name: "CsCl",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A1B1",
  },
  {
    name: "Cu2Mg-Laves",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "B", x: 0.25, y: 0.25, z: 0.25 },
      { site: "B", x: 0.75, y: 0.75, z: 0.75 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4B2",
  },
  {
    name: "Fluorite",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "B", x: 0.25, y: 0.25, z: 0.25 },
      { site: "B", x: 0.75, y: 0.75, z: 0.25 },
      { site: "B", x: 0.75, y: 0.25, z: 0.75 },
      { site: "B", x: 0.25, y: 0.75, z: 0.75 },
      { site: "B", x: 0.25, y: 0.75, z: 0.25 },
      { site: "B", x: 0.75, y: 0.25, z: 0.25 },
      { site: "B", x: 0.25, y: 0.25, z: 0.75 },
      { site: "B", x: 0.75, y: 0.75, z: 0.75 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4B8",
  },
  {
    name: "Cr3Si",
    fractionalPositions: [
      { site: "A", x: 0.25, y: 0.0, z: 0.5 },
      { site: "A", x: 0.75, y: 0.0, z: 0.5 },
      { site: "A", x: 0.5, y: 0.25, z: 0.0 },
      { site: "A", x: 0.5, y: 0.75, z: 0.0 },
      { site: "A", x: 0.0, y: 0.5, z: 0.25 },
      { site: "A", x: 0.0, y: 0.5, z: 0.75 },
      { site: "B", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A6B2",
  },
  {
    name: "Ni3Sn",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.25, y: 0.25, z: 0.25 },
      { site: "A", x: 0.75, y: 0.75, z: 0.75 },
      { site: "B", x: 0.25, y: 0.25, z: 0.75 },
      { site: "B", x: 0.75, y: 0.75, z: 0.25 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A6B2",
  },
  {
    name: "Fe3C",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "B", x: 0.25, y: 0.25, z: 0.25 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A3B1",
  },
  {
    name: "Spinel",
    fractionalPositions: [
      { site: "A", x: 0.125, y: 0.125, z: 0.125 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.5, y: 0.0, z: 0.0 },
      { site: "C", x: 0.25, y: 0.25, z: 0.75 },
      { site: "C", x: 0.75, y: 0.75, z: 0.25 },
      { site: "C", x: 0.25, y: 0.75, z: 0.25 },
      { site: "C", x: 0.75, y: 0.25, z: 0.75 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A1B2C4",
  },
  {
    name: "Clathrate-H32",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.25, y: 0.0, z: 0.5 },
      { site: "B", x: 0.75, y: 0.0, z: 0.5 },
      { site: "B", x: 0.5, y: 0.25, z: 0.0 },
      { site: "B", x: 0.5, y: 0.75, z: 0.0 },
      { site: "B", x: 0.0, y: 0.5, z: 0.25 },
      { site: "B", x: 0.0, y: 0.5, z: 0.75 },
      { site: "B", x: 0.185, y: 0.185, z: 0.185 },
      { site: "B", x: 0.815, y: 0.815, z: 0.185 },
      { site: "B", x: 0.815, y: 0.185, z: 0.815 },
      { site: "B", x: 0.185, y: 0.815, z: 0.815 },
      { site: "B", x: 0.685, y: 0.685, z: 0.685 },
      { site: "B", x: 0.315, y: 0.315, z: 0.685 },
      { site: "B", x: 0.315, y: 0.685, z: 0.315 },
      { site: "B", x: 0.685, y: 0.315, z: 0.315 },
      { site: "B", x: 0.375, y: 0.375, z: 0.375 },
      { site: "B", x: 0.625, y: 0.625, z: 0.375 },
      { site: "B", x: 0.625, y: 0.375, z: 0.625 },
      { site: "B", x: 0.375, y: 0.625, z: 0.625 },
      { site: "B", x: 0.125, y: 0.125, z: 0.625 },
      { site: "B", x: 0.875, y: 0.875, z: 0.625 },
      { site: "B", x: 0.875, y: 0.125, z: 0.375 },
      { site: "B", x: 0.125, y: 0.875, z: 0.375 },
      { site: "B", x: 0.125, y: 0.625, z: 0.125 },
      { site: "B", x: 0.875, y: 0.375, z: 0.125 },
      { site: "B", x: 0.625, y: 0.125, z: 0.125 },
      { site: "B", x: 0.375, y: 0.875, z: 0.125 },
      { site: "B", x: 0.125, y: 0.375, z: 0.875 },
      { site: "B", x: 0.875, y: 0.625, z: 0.875 },
      { site: "B", x: 0.625, y: 0.875, z: 0.875 },
      { site: "B", x: 0.375, y: 0.125, z: 0.875 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A1B31",
  },
  {
    name: "Skutterudite",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "A", x: 0.25, y: 0.25, z: 0.25 },
      { site: "A", x: 0.75, y: 0.75, z: 0.25 },
      { site: "A", x: 0.75, y: 0.25, z: 0.75 },
      { site: "A", x: 0.25, y: 0.75, z: 0.75 },
      { site: "B", x: 0.0, y: 0.335, z: 0.158 },
      { site: "B", x: 0.0, y: 0.665, z: 0.842 },
      { site: "B", x: 0.335, y: 0.158, z: 0.0 },
      { site: "B", x: 0.665, y: 0.842, z: 0.0 },
      { site: "B", x: 0.158, y: 0.0, z: 0.335 },
      { site: "B", x: 0.842, y: 0.0, z: 0.665 },
      { site: "B", x: 0.5, y: 0.835, z: 0.658 },
      { site: "B", x: 0.5, y: 0.165, z: 0.342 },
      { site: "B", x: 0.835, y: 0.658, z: 0.5 },
      { site: "B", x: 0.165, y: 0.342, z: 0.5 },
      { site: "B", x: 0.658, y: 0.5, z: 0.835 },
      { site: "B", x: 0.342, y: 0.5, z: 0.165 },
      { site: "B", x: 0.0, y: 0.158, z: 0.335 },
      { site: "B", x: 0.0, y: 0.842, z: 0.665 },
      { site: "B", x: 0.158, y: 0.335, z: 0.0 },
      { site: "B", x: 0.842, y: 0.665, z: 0.0 },
      { site: "B", x: 0.335, y: 0.0, z: 0.158 },
      { site: "B", x: 0.665, y: 0.0, z: 0.842 },
      { site: "B", x: 0.5, y: 0.658, z: 0.835 },
      { site: "B", x: 0.5, y: 0.342, z: 0.165 },
      { site: "B", x: 0.658, y: 0.835, z: 0.5 },
      { site: "B", x: 0.342, y: 0.165, z: 0.5 },
      { site: "B", x: 0.835, y: 0.5, z: 0.658 },
      { site: "B", x: 0.165, y: 0.5, z: 0.342 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A8B24",
  },
  {
    name: "BiS2-layered",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "B", x: 0.0, y: 0.0, z: 0.35 },
      { site: "B", x: 0.5, y: 0.5, z: 0.35 },
      { site: "C", x: 0.0, y: 0.5, z: 0.15 },
      { site: "C", x: 0.5, y: 0.0, z: 0.15 },
      { site: "C", x: 0.0, y: 0.5, z: 0.55 },
      { site: "C", x: 0.5, y: 0.0, z: 0.55 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 3.2,
    stoichiometryPattern: "A2B2C4",
  },
  {
    name: "Kagome-variant",
    fractionalPositions: [
      { site: "A", x: 0.5, y: 0.0, z: 0.0 },
      { site: "A", x: 0.0, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "B", x: 0.0, y: 0.0, z: 0.25 },
      { site: "C", x: 0.333, y: 0.667, z: 0.5 },
      { site: "C", x: 0.667, y: 0.333, z: 0.5 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 1.75,
    stoichiometryPattern: "A3B1C2",
  },
  {
    name: "Chevrel",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.2, y: 0.2, z: 0.2 },
      { site: "B", x: 0.8, y: 0.2, z: 0.2 },
      { site: "B", x: 0.2, y: 0.8, z: 0.2 },
      { site: "B", x: 0.2, y: 0.2, z: 0.8 },
      { site: "B", x: 0.8, y: 0.8, z: 0.2 },
      { site: "B", x: 0.8, y: 0.2, z: 0.8 },
      { site: "C", x: 0.4, y: 0.0, z: 0.0 },
      { site: "C", x: 0.0, y: 0.4, z: 0.0 },
      { site: "C", x: 0.0, y: 0.0, z: 0.4 },
      { site: "C", x: 0.6, y: 0.6, z: 0.0 },
      { site: "C", x: 0.6, y: 0.0, z: 0.6 },
      { site: "C", x: 0.0, y: 0.6, z: 0.6 },
      { site: "C", x: 0.4, y: 0.4, z: 0.6 },
      { site: "C", x: 0.4, y: 0.6, z: 0.4 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A1B6C8",
  },
  {
    name: "Pyrite",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "B", x: 0.386, y: 0.386, z: 0.386 },
      { site: "B", x: 0.614, y: 0.614, z: 0.386 },
      { site: "B", x: 0.614, y: 0.386, z: 0.614 },
      { site: "B", x: 0.386, y: 0.614, z: 0.614 },
      { site: "B", x: 0.886, y: 0.886, z: 0.114 },
      { site: "B", x: 0.114, y: 0.114, z: 0.114 },
      { site: "B", x: 0.114, y: 0.886, z: 0.886 },
      { site: "B", x: 0.886, y: 0.114, z: 0.886 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4B8",
  },
  {
    name: "Wurtzite",
    fractionalPositions: [
      { site: "A", x: 0.333, y: 0.667, z: 0.0 },
      { site: "A", x: 0.667, y: 0.333, z: 0.5 },
      { site: "B", x: 0.333, y: 0.667, z: 0.375 },
      { site: "B", x: 0.667, y: 0.333, z: 0.875 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 1.633,
    stoichiometryPattern: "A2B2",
  },
  {
    name: "Antifluorite",
    fractionalPositions: [
      { site: "A", x: 0.25, y: 0.25, z: 0.25 },
      { site: "A", x: 0.75, y: 0.75, z: 0.25 },
      { site: "A", x: 0.75, y: 0.25, z: 0.75 },
      { site: "A", x: 0.25, y: 0.75, z: 0.75 },
      { site: "A", x: 0.25, y: 0.25, z: 0.75 },
      { site: "A", x: 0.75, y: 0.75, z: 0.75 },
      { site: "A", x: 0.75, y: 0.25, z: 0.25 },
      { site: "A", x: 0.25, y: 0.75, z: 0.25 },
      { site: "B", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.0 },
      { site: "B", x: 0.5, y: 0.0, z: 0.5 },
      { site: "B", x: 0.0, y: 0.5, z: 0.5 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A8B4",
  },
  {
    name: "Laves-C14",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.333, y: 0.667, z: 0.5 },
      { site: "A", x: 0.667, y: 0.333, z: 0.5 },
      { site: "A", x: 0.333, y: 0.667, z: 0.0 },
      { site: "B", x: 0.167, y: 0.333, z: 0.25 },
      { site: "B", x: 0.833, y: 0.667, z: 0.25 },
      { site: "B", x: 0.5, y: 0.0, z: 0.25 },
      { site: "B", x: 0.167, y: 0.333, z: 0.75 },
      { site: "B", x: 0.833, y: 0.667, z: 0.75 },
      { site: "B", x: 0.5, y: 0.0, z: 0.75 },
      { site: "B", x: 0.833, y: 0.167, z: 0.25 },
      { site: "B", x: 0.167, y: 0.833, z: 0.75 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 1.633,
    stoichiometryPattern: "A4B8",
  },
  {
    name: "Laves-C15",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.0 },
      { site: "A", x: 0.5, y: 0.0, z: 0.5 },
      { site: "A", x: 0.0, y: 0.5, z: 0.5 },
      { site: "B", x: 0.625, y: 0.625, z: 0.625 },
      { site: "B", x: 0.375, y: 0.375, z: 0.625 },
      { site: "B", x: 0.375, y: 0.625, z: 0.375 },
      { site: "B", x: 0.625, y: 0.375, z: 0.375 },
      { site: "B", x: 0.125, y: 0.125, z: 0.125 },
      { site: "B", x: 0.875, y: 0.875, z: 0.125 },
      { site: "B", x: 0.875, y: 0.125, z: 0.875 },
      { site: "B", x: 0.125, y: 0.875, z: 0.875 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4B8",
  },
  {
    name: "HfFe6Ge6",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.0, z: 0.25 },
      { site: "B", x: 0.0, y: 0.5, z: 0.25 },
      { site: "B", x: 0.5, y: 0.5, z: 0.25 },
      { site: "B", x: 0.5, y: 0.0, z: 0.75 },
      { site: "B", x: 0.0, y: 0.5, z: 0.75 },
      { site: "B", x: 0.5, y: 0.5, z: 0.75 },
      { site: "C", x: 0.333, y: 0.667, z: 0.0 },
      { site: "C", x: 0.667, y: 0.333, z: 0.0 },
      { site: "C", x: 0.333, y: 0.667, z: 0.5 },
      { site: "C", x: 0.667, y: 0.333, z: 0.5 },
      { site: "C", x: 0.0, y: 0.0, z: 0.5 },
      { site: "C", x: 0.333, y: 0.0, z: 0.25 },
    ],
    latticeType: "hexagonal",
    aRatio: 1.0,
    cOverA: 1.63,
    stoichiometryPattern: "A1B6C6",
  },
  {
    name: "CeCu2Si2",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.0, y: 0.5, z: 0.25 },
      { site: "B", x: 0.5, y: 0.0, z: 0.25 },
      { site: "B", x: 0.0, y: 0.5, z: 0.75 },
      { site: "B", x: 0.5, y: 0.0, z: 0.75 },
      { site: "C", x: 0.0, y: 0.0, z: 0.38 },
      { site: "C", x: 0.0, y: 0.0, z: 0.62 },
      { site: "C", x: 0.5, y: 0.5, z: 0.88 },
      { site: "C", x: 0.5, y: 0.5, z: 0.12 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 2.45,
    stoichiometryPattern: "A2B4C4",
  },
  {
    name: "PuCoGa5-115",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.0 },
      { site: "C", x: 0.5, y: 0.0, z: 0.312 },
      { site: "C", x: 0.0, y: 0.5, z: 0.312 },
      { site: "C", x: 0.5, y: 0.0, z: 0.688 },
      { site: "C", x: 0.0, y: 0.5, z: 0.688 },
      { site: "C", x: 0.0, y: 0.0, z: 0.5 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 1.6,
    stoichiometryPattern: "A1B1C5",
  },
  {
    name: "Infinite-layer",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
      { site: "C", x: 0.5, y: 0.0, z: 0.5 },
      { site: "C", x: 0.0, y: 0.5, z: 0.5 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 0.85,
    stoichiometryPattern: "A1B1C2",
  },
  {
    name: "T-prime",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.35 },
      { site: "A", x: 0.5, y: 0.5, z: 0.65 },
      { site: "B", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.0 },
      { site: "C", x: 0.0, y: 0.5, z: 0.0 },
      { site: "C", x: 0.5, y: 0.0, z: 0.0 },
      { site: "C", x: 0.0, y: 0.0, z: 0.16 },
      { site: "C", x: 0.5, y: 0.5, z: 0.84 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 3.1,
    stoichiometryPattern: "A2B2C4",
  },
  {
    name: "Ruddlesden-Popper",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.0, y: 0.0, z: 0.5 },
      { site: "B", x: 0.5, y: 0.5, z: 0.25 },
      { site: "C", x: 0.5, y: 0.0, z: 0.25 },
      { site: "C", x: 0.0, y: 0.5, z: 0.25 },
      { site: "C", x: 0.5, y: 0.5, z: 0.0 },
      { site: "C", x: 0.5, y: 0.5, z: 0.5 },
    ],
    latticeType: "tetragonal",
    aRatio: 1.0,
    cOverA: 3.3,
    stoichiometryPattern: "A2B1C4",
  },
  {
    name: "Double-perovskite",
    fractionalPositions: [
      { site: "A", x: 0.0, y: 0.0, z: 0.0 },
      { site: "A", x: 0.5, y: 0.5, z: 0.5 },
      { site: "B", x: 0.5, y: 0.5, z: 0.0 },
      { site: "B", x: 0.0, y: 0.0, z: 0.5 },
      { site: "C", x: 0.25, y: 0.0, z: 0.0 },
      { site: "C", x: 0.75, y: 0.0, z: 0.0 },
      { site: "C", x: 0.0, y: 0.25, z: 0.0 },
      { site: "C", x: 0.0, y: 0.75, z: 0.0 },
      { site: "C", x: 0.0, y: 0.0, z: 0.25 },
      { site: "C", x: 0.0, y: 0.0, z: 0.75 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A2B2C6",
  },
  {
    name: "Garnet",
    fractionalPositions: [
      { site: "A", x: 0.125, y: 0.0, z: 0.25 },
      { site: "A", x: 0.875, y: 0.0, z: 0.75 },
      { site: "A", x: 0.375, y: 0.5, z: 0.25 },
      { site: "B", x: 0.0, y: 0.0, z: 0.0 },
      { site: "B", x: 0.5, y: 0.5, z: 0.5 },
      { site: "C", x: 0.375, y: 0.0, z: 0.25 },
      { site: "C", x: 0.625, y: 0.0, z: 0.75 },
      { site: "C", x: 0.125, y: 0.5, z: 0.25 },
      { site: "C", x: 0.875, y: 0.5, z: 0.75 },
      { site: "C", x: 0.25, y: 0.125, z: 0.0 },
      { site: "C", x: 0.75, y: 0.875, z: 0.0 },
      { site: "C", x: 0.25, y: 0.375, z: 0.5 },
      { site: "C", x: 0.75, y: 0.625, z: 0.5 },
      { site: "C", x: 0.0, y: 0.25, z: 0.125 },
      { site: "C", x: 0.0, y: 0.75, z: 0.875 },
      { site: "C", x: 0.5, y: 0.25, z: 0.625 },
      { site: "C", x: 0.5, y: 0.75, z: 0.375 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A3B2C12",
  },
];

function getProtoSiteCounts(proto: PrototypeStructure): Record<string, number> {
  const siteCounts: Record<string, number> = {};
  for (const pos of proto.fractionalPositions) {
    siteCounts[pos.site] = (siteCounts[pos.site] || 0) + 1;
  }
  return siteCounts;
}

const RARE_EARTHS = new Set(["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"]);
const TRANSITION_METALS = new Set(["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]);
const ANIONS = new Set(["O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As"]);
const ALKALINE_EARTH = new Set(["Ca", "Sr", "Ba", "Mg"]);
const ALKALI = new Set(["Li", "Na", "K", "Rb", "Cs"]);

function classifyElement(el: string): "H" | "rare-earth" | "TM" | "anion" | "alkaline" | "alkali" | "metalloid" | "other" {
  if (el === "H") return "H";
  if (RARE_EARTHS.has(el)) return "rare-earth";
  if (TRANSITION_METALS.has(el)) return "TM";
  if (ANIONS.has(el)) return "anion";
  if (ALKALINE_EARTH.has(el)) return "alkaline";
  if (ALKALI.has(el)) return "alkali";
  if (["B", "Si", "Ge", "Sn", "Sb", "Bi", "Al", "Ga", "In", "Tl", "Pb"].includes(el)) return "metalloid";
  return "other";
}

function selectBestPrototypeByChemistry(counts: Record<string, number>, elements: string[]): { proto: PrototypeStructure; siteMap: Record<string, string> } | null {
  const roles = elements.map(el => classifyElement(el));
  const nElements = elements.length;
  const elementsByCount = [...elements].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));

  const hasRE = roles.includes("rare-earth");
  const hasTM = roles.includes("TM");
  const hasAnion = roles.includes("anion");
  const hasAlkaline = roles.includes("alkaline");
  const hasAlkali = roles.includes("alkali");
  const hasMetalloid = roles.includes("metalloid");

  let targetProtoName: string | null = null;

  if (nElements === 3 && (hasRE || hasAlkaline || hasAlkali) && hasTM && hasAnion) {
    const anionEl = elements.find(e => ANIONS.has(e))!;
    const anionCount = counts[anionEl] || 0;
    const totalNonAnion = Object.values(counts).reduce((s, n) => s + n, 0) - anionCount;
    const anionRatio = anionCount / Math.max(1, totalNonAnion);
    if (anionRatio >= 2.5) targetProtoName = "Ruddlesden-Popper";
    else if (anionRatio >= 1.2) targetProtoName = "Perovskite";
    else targetProtoName = "ThCr2Si2";
  } else if (nElements === 3 && hasTM && hasAnion && hasMetalloid) {
    targetProtoName = "BiS2-layered";
  } else if (nElements === 2 && hasTM && hasAnion) {
    const tmEl = elements.find(e => TRANSITION_METALS.has(e))!;
    const anEl = elements.find(e => ANIONS.has(e))!;
    const tmCount = counts[tmEl] || 0;
    const anCount = counts[anEl] || 0;
    if (anCount / tmCount >= 1.8) targetProtoName = "Pyrite";
    else if (tmCount / anCount >= 2.5) targetProtoName = "Cr3Si";
    else targetProtoName = "NaCl";
  } else if (nElements === 2 && hasTM && hasMetalloid) {
    const tmEl = elements.find(e => TRANSITION_METALS.has(e))!;
    const mEl = elements.find(e => classifyElement(e) === "metalloid")!;
    const tmCount = counts[tmEl] || 0;
    const mCount = counts[mEl] || 0;
    if (tmCount / mCount >= 2.5) targetProtoName = "A15";
    else if (mCount / tmCount >= 2) targetProtoName = "AlB2";
    else targetProtoName = "CsCl";
  } else if (nElements === 2 && (hasRE || hasAlkaline) && hasTM) {
    targetProtoName = "Cu2Mg-Laves";
  } else if (nElements === 2 && roles.includes("H")) {
    const metalEl = elements.find(e => e !== "H");
    const hCount = counts["H"] || 0;
    const metalCount = metalEl ? (counts[metalEl] || 0) : 1;
    const hRatio = hCount / metalCount;
    if (hRatio >= 2) targetProtoName = "AlB2";
    else if (hRatio <= 0.5) targetProtoName = "CsCl";
    else targetProtoName = "NaCl";
  } else if (nElements === 3 && roles.includes("H") && (hasTM || hasRE || hasAlkaline || hasAlkali)) {
    const nonHNonAnion = elements.filter(e => e !== "H" && !ANIONS.has(e));
    if (nonHNonAnion.length >= 2) targetProtoName = "Perovskite";
    else targetProtoName = "ThCr2Si2";
  } else if (nElements === 1) {
    const el = elements[0];
    const role = classifyElement(el);
    if (role === "TM") {
      const count = Math.round(counts[el]);
      if (count <= 2) targetProtoName = "BCC";
      else targetProtoName = "FCC";
    } else {
      targetProtoName = "FCC";
    }
  } else if (nElements === 4) {
    const hasPnictide = elements.some(e => ["As", "P", "Sb"].includes(e));
    const hasOorF = elements.some(e => e === "O" || e === "F");
    const hasSpacer = elements.some(e => RARE_EARTHS.has(e) || ALKALINE_EARTH.has(e) || ALKALI.has(e));
    if (hasSpacer && hasTM && hasPnictide && hasOorF) {
      targetProtoName = "ThCr2Si2";
    } else if (hasSpacer && hasTM && hasAnion) {
      targetProtoName = "Perovskite";
    } else {
      targetProtoName = "Perovskite";
    }
  } else if (nElements >= 5) {
    targetProtoName = "Perovskite";
  }

  if (!targetProtoName) return null;

  const proto = CRYSTAL_PROTOTYPES.find(p => p.name === targetProtoName);
  if (!proto) return null;

  const siteCounts = getProtoSiteCounts(proto);
  const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);
  const siteMap: Record<string, string> = {};

  for (let i = 0; i < Math.min(sites.length, elementsByCount.length); i++) {
    siteMap[sites[i]] = elementsByCount[i];
  }
  for (let i = elementsByCount.length; i < sites.length; i++) {
    siteMap[sites[i]] = elementsByCount[elementsByCount.length - 1];
  }

  return { proto, siteMap };
}

function isPrototypeChemicallyCompatible(protoName: string, elements: string[], counts: Record<string, number>): boolean {
  const hasElement = (el: string) => elements.includes(el);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  if (protoName === "HexBoride" || protoName === "AlB2") {
    if (!hasElement("B")) return false;
  }

  if (protoName === "Perovskite" || protoName === "Anti-perovskite" || protoName === "Ruddlesden-Popper" || protoName === "Double-perovskite") {
    const hasAnionEl = elements.some(e => ANIONS.has(e));
    if (!hasAnionEl) return false;
  }

  if (protoName === "Pyrite" || protoName === "NaCl" || protoName === "MX2" || protoName === "BiS2-layered") {
    const hasAnionEl = elements.some(e => ANIONS.has(e));
    if (!hasAnionEl) return false;
  }

  if (protoName === "A15" || protoName === "Cr3Si") {
    const hasTMEl = elements.some(e => TRANSITION_METALS.has(e));
    if (!hasTMEl) return false;
  }

  if (protoName.includes("Clathrate") || protoName.includes("clathrate") ||
      /H\d+/.test(protoName) || protoName === "Sodalite-H32" || protoName === "TriCappedPrism-H9") {
    if (!hasElement("H")) return false;
  }

  if (protoName === "Heusler" || protoName === "ThCr2Si2" || protoName === "CeCu2Si2" ||
      protoName === "PuCoGa5-115" || protoName === "HfFe6Ge6") {
    const hasTMEl = elements.some(e => TRANSITION_METALS.has(e));
    if (!hasTMEl) return false;
  }

  return true;
}

function matchPrototype(counts: Record<string, number>): { proto: PrototypeStructure; siteMap: Record<string, string> } | null {
  const elements = Object.keys(counts).sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
  const nElements = elements.length;

  const ratios = elements.map(el => Math.round(counts[el]));
  const gcdVal = ratios.reduce((a, b) => gcd(a, b));
  const reduced = ratios.map(r => r / gcdVal);

  for (const proto of CRYSTAL_PROTOTYPES) {
    const siteCounts = getProtoSiteCounts(proto);
    const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);

    if (sites.length !== nElements) continue;

    const siteRatios = sites.map(s => siteCounts[s]);
    const siteGcd = siteRatios.reduce((a, b) => gcd(a, b));
    const siteReduced = siteRatios.map(r => r / siteGcd);

    const sortedReduced = [...reduced].sort((a, b) => b - a);
    const sortedSiteReduced = [...siteReduced].sort((a, b) => b - a);

    let match = true;
    for (let i = 0; i < sortedReduced.length; i++) {
      if (sortedReduced[i] !== sortedSiteReduced[i]) {
        match = false;
        break;
      }
    }

    if (match) {
      if (!isPrototypeChemicallyCompatible(proto.name, elements, counts)) continue;
      const siteMap: Record<string, string> = {};
      const elementsByCount = [...elements].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
      const sitesByCount = [...sites].sort((a, b) => siteCounts[b] - siteCounts[a]);
      for (let i = 0; i < elementsByCount.length; i++) {
        siteMap[sitesByCount[i]] = elementsByCount[i];
      }
      return { proto, siteMap };
    }
  }

  let bestProto: PrototypeStructure | null = null;
  let bestScore = Infinity;
  let bestSiteMap: Record<string, string> = {};

  for (const proto of CRYSTAL_PROTOTYPES) {
    const siteCounts = getProtoSiteCounts(proto);
    const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);
    if (sites.length !== nElements) continue;

    const siteRatios = sites.map(s => siteCounts[s]);
    const siteGcd = siteRatios.reduce((a, b) => gcd(a, b));
    const siteReduced = siteRatios.map(r => r / siteGcd);
    const sortedSiteReduced = [...siteReduced].sort((a, b) => b - a);
    const sortedReduced = [...reduced].sort((a, b) => b - a);

    let score = 0;
    for (let i = 0; i < sortedReduced.length; i++) {
      const diff = Math.abs(sortedReduced[i] - sortedSiteReduced[i]);
      score += diff / Math.max(1, sortedSiteReduced[i]);
    }

    if (score < bestScore && score < 0.5 && isPrototypeChemicallyCompatible(proto.name, elements, counts)) {
      bestScore = score;
      bestProto = proto;
      const elementsByCount = [...elements].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
      const sitesByCount = [...sites].sort((a, b) => siteCounts[b] - siteCounts[a]);
      bestSiteMap = {};
      for (let i = 0; i < elementsByCount.length; i++) {
        bestSiteMap[sitesByCount[i]] = elementsByCount[i];
      }
    }
  }

  if (bestProto) {
    return { proto: bestProto, siteMap: bestSiteMap };
  }

  for (const proto of CRYSTAL_PROTOTYPES) {
    const siteCounts = getProtoSiteCounts(proto);
    const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);
    if (sites.length >= nElements) continue;

    const elementsByCount = [...elements].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
    const sitesByCount = [...sites].sort((a, b) => siteCounts[b] - siteCounts[a]);

    const siteRatios = sitesByCount.map(s => siteCounts[s]);
    const siteGcd = siteRatios.reduce((a, b) => gcd(a, b));
    const siteReduced = siteRatios.map(r => r / siteGcd);

    const mergedReduced: number[] = [];
    for (let i = 0; i < sitesByCount.length; i++) {
      if (i < elementsByCount.length) {
        mergedReduced.push(Math.round(counts[elementsByCount[i]]));
      }
    }
    for (let i = sitesByCount.length; i < elementsByCount.length; i++) {
      mergedReduced[mergedReduced.length - 1] += Math.round(counts[elementsByCount[i]]);
    }
    const mergedGcd = mergedReduced.reduce((a, b) => gcd(a, b));
    const mergedNorm = mergedReduced.map(r => r / mergedGcd);

    const sortedMerged = [...mergedNorm].sort((a, b) => b - a);
    const sortedSite = [...siteReduced].sort((a, b) => b - a);

    let score = 0;
    for (let i = 0; i < sortedMerged.length; i++) {
      const diff = Math.abs(sortedMerged[i] - sortedSite[i]);
      score += diff / Math.max(1, sortedSite[i]);
    }

    if (score < bestScore && score < 0.8 && isPrototypeChemicallyCompatible(proto.name, elements, counts)) {
      bestScore = score;
      bestProto = proto;
      bestSiteMap = {};
      for (let i = 0; i < Math.min(sitesByCount.length, elementsByCount.length); i++) {
        bestSiteMap[sitesByCount[i]] = elementsByCount[i];
      }
      for (let i = elementsByCount.length; i < sitesByCount.length; i++) {
        bestSiteMap[sitesByCount[i]] = elementsByCount[elementsByCount.length - 1];
      }
    }
  }

  if (bestProto) {
    return { proto: bestProto, siteMap: bestSiteMap };
  }

  const chemMatch = selectBestPrototypeByChemistry(counts, elements);
  if (chemMatch) {
    chemistryMatchSuccesses++;
    return chemMatch;
  }

  if (nElements === 1) {
    const el = elements[0];
    return { proto: CRYSTAL_PROTOTYPES.find(p => p.name === "BCC")!, siteMap: { A: el } };
  }

  if (nElements >= 2) {
    const fallbackBinary = nElements === 2
      ? CRYSTAL_PROTOTYPES.find(p => p.name === "NaCl")!
      : CRYSTAL_PROTOTYPES.find(p => p.name === "Perovskite") ?? CRYSTAL_PROTOTYPES.find(p => p.name === "NaCl")!;
    const siteCounts = getProtoSiteCounts(fallbackBinary);
    const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);
    const siteMap: Record<string, string> = {};
    const elementsByCount = [...elements].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
    for (let i = 0; i < Math.min(sites.length, elementsByCount.length); i++) {
      siteMap[sites[i]] = elementsByCount[i];
    }
    for (let i = elementsByCount.length; i < sites.length; i++) {
      siteMap[sites[i]] = elementsByCount[elementsByCount.length - 1];
    }
    return { proto: fallbackBinary, siteMap };
  }

  return null;
}

function gcd(a: number, b: number): number {
  a = Math.abs(Math.round(a));
  b = Math.abs(Math.round(b));
  while (b) { [a, b] = [b, a % b]; }
  return a || 1;
}

function buildStructureFromPrototype(
  proto: PrototypeStructure,
  siteMap: Record<string, string>,
  elements: string[],
  counts: Record<string, number>,
  scaleFactor: number = 1
): AtomPosition[] {
  const a = estimateLatticeParam(elements, counts, proto.name) * scaleFactor;
  const c = a * proto.cOverA;

  const atoms: AtomPosition[] = [];
  for (const pos of proto.fractionalPositions) {
    const element = siteMap[pos.site];
    if (!element) continue;

    let x: number, y: number, z: number;
    if (proto.latticeType === "hexagonal") {
      x = a * pos.x + a * 0.5 * pos.y;
      y = a * (Math.sqrt(3) / 2) * pos.y;
      z = c * pos.z;
    } else {
      x = a * pos.x;
      y = a * pos.y;
      z = (proto.latticeType === "tetragonal" ? c : a) * pos.z;
    }
    atoms.push({ element, x, y, z });
  }

  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[j].x - atoms[i].x;
      const dy = atoms[j].y - atoms[i].y;
      const dz = atoms[j].z - atoms[i].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 0.05) {
        atoms[j].x += 0.1 * a;
        atoms[j].y += 0.1 * a;
        atoms[j].z += 0.1 * (proto.latticeType === "tetragonal" ? c : a);
      }
    }
  }

  return atoms;
}

function buildGenericStructure(counts: Record<string, number>): { atoms: AtomPosition[]; proto: string } {
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  const a = estimateLatticeParam(elements, counts);

  const atoms: AtomPosition[] = [];
  let idx = 0;

  const gridSize = Math.ceil(Math.cbrt(totalAtoms));

  for (const el of elements) {
    const n = Math.round(counts[el]);
    for (let i = 0; i < n; i++) {
      const ix = idx % gridSize;
      const iy = Math.floor(idx / gridSize) % gridSize;
      const iz = Math.floor(idx / (gridSize * gridSize));

      const jx = ((idx * 7 + 3) % 17) / 34.0;
      const jy = ((idx * 11 + 5) % 13) / 26.0;
      const jz = ((idx * 13 + 7) % 19) / 38.0;
      atoms.push({
        element: el,
        x: (ix + 0.5 + jx * 0.4) * a / gridSize,
        y: (iy + 0.5 + jy * 0.4 - 0.2) * a / gridSize,
        z: (iz + 0.5 + jz * 0.4 - 0.2) * a / gridSize,
      });
      idx++;
    }
  }

  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[j].x - atoms[i].x;
      const dy = atoms[j].y - atoms[i].y;
      const dz = atoms[j].z - atoms[i].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < 0.3) {
        atoms[j].x += 0.15 * a / gridSize;
        atoms[j].y += 0.1 * a / gridSize;
        atoms[j].z += 0.12 * a / gridSize;
      }
    }
  }

  return { atoms, proto: "generic-cluster" };
}

const MIN_VOLUME_PER_ATOM = 10.0;
const MIN_VOLUME_PER_ATOM_HYDRIDE = 5.0;
const MAX_SCALE_ATTEMPTS = 5;

interface HydrideCageMotif {
  name: string;
  hPerMetal: number;
  metalFrac: { x: number; y: number; z: number }[];
  hydrogenFrac: { x: number; y: number; z: number }[];
  latticeType: "cubic" | "hexagonal";
  cOverA: number;
  baseLatticeFactor: number;
}

const HYDRIDE_CAGE_LIBRARY: HydrideCageMotif[] = [
  {
    name: "Sodalite-H32",
    hPerMetal: 10,
    metalFrac: [
      { x: 0.0, y: 0.0, z: 0.0 },
      { x: 0.5, y: 0.5, z: 0.0 },
      { x: 0.5, y: 0.0, z: 0.5 },
      { x: 0.0, y: 0.5, z: 0.5 },
    ],
    hydrogenFrac: [
      { x: 0.120, y: 0.120, z: 0.120 },
      { x: 0.880, y: 0.120, z: 0.120 },
      { x: 0.120, y: 0.880, z: 0.120 },
      { x: 0.120, y: 0.120, z: 0.880 },
      { x: 0.880, y: 0.880, z: 0.120 },
      { x: 0.880, y: 0.120, z: 0.880 },
      { x: 0.120, y: 0.880, z: 0.880 },
      { x: 0.880, y: 0.880, z: 0.880 },
      { x: 0.250, y: 0.0, z: 0.5 },
      { x: 0.750, y: 0.0, z: 0.5 },
      { x: 0.0, y: 0.250, z: 0.5 },
      { x: 0.0, y: 0.750, z: 0.5 },
      { x: 0.5, y: 0.250, z: 0.0 },
      { x: 0.5, y: 0.750, z: 0.0 },
      { x: 0.250, y: 0.5, z: 0.0 },
      { x: 0.750, y: 0.5, z: 0.0 },
      { x: 0.0, y: 0.5, z: 0.250 },
      { x: 0.0, y: 0.5, z: 0.750 },
      { x: 0.5, y: 0.0, z: 0.250 },
      { x: 0.5, y: 0.0, z: 0.750 },
      { x: 0.620, y: 0.620, z: 0.620 },
      { x: 0.380, y: 0.380, z: 0.620 },
      { x: 0.380, y: 0.620, z: 0.380 },
      { x: 0.620, y: 0.380, z: 0.380 },
      { x: 0.185, y: 0.185, z: 0.5 },
      { x: 0.815, y: 0.815, z: 0.5 },
      { x: 0.185, y: 0.5, z: 0.185 },
      { x: 0.815, y: 0.5, z: 0.815 },
      { x: 0.5, y: 0.185, z: 0.185 },
      { x: 0.5, y: 0.815, z: 0.815 },
      { x: 0.5, y: 0.185, z: 0.815 },
      { x: 0.5, y: 0.815, z: 0.185 },
    ],
    latticeType: "cubic",
    cOverA: 1.0,
    baseLatticeFactor: 5.0,
  },
  {
    name: "Clathrate-H6",
    hPerMetal: 6,
    metalFrac: [
      { x: 0.0, y: 0.0, z: 0.0 },
    ],
    hydrogenFrac: [
      { x: 0.25, y: 0.0, z: 0.0 },
      { x: 0.75, y: 0.0, z: 0.0 },
      { x: 0.0, y: 0.25, z: 0.0 },
      { x: 0.0, y: 0.75, z: 0.0 },
      { x: 0.0, y: 0.0, z: 0.25 },
      { x: 0.0, y: 0.0, z: 0.75 },
    ],
    latticeType: "cubic",
    cOverA: 1.0,
    baseLatticeFactor: 3.65,
  },
  {
    name: "TriCappedPrism-H9",
    hPerMetal: 9,
    metalFrac: [
      { x: 0.0, y: 0.0, z: 0.0 },
      { x: 0.5, y: 0.5, z: 0.5 },
    ],
    hydrogenFrac: [
      { x: 0.167, y: 0.167, z: 0.0 },
      { x: 0.833, y: 0.167, z: 0.0 },
      { x: 0.167, y: 0.833, z: 0.0 },
      { x: 0.167, y: 0.167, z: 0.333 },
      { x: 0.833, y: 0.167, z: 0.333 },
      { x: 0.167, y: 0.833, z: 0.333 },
      { x: 0.5, y: 0.0, z: 0.167 },
      { x: 0.0, y: 0.5, z: 0.167 },
      { x: 0.5, y: 0.5, z: 0.0 },
      { x: 0.667, y: 0.667, z: 0.5 },
      { x: 0.333, y: 0.667, z: 0.5 },
      { x: 0.667, y: 0.333, z: 0.5 },
      { x: 0.667, y: 0.667, z: 0.833 },
      { x: 0.333, y: 0.667, z: 0.833 },
      { x: 0.667, y: 0.333, z: 0.833 },
      { x: 0.0, y: 0.5, z: 0.667 },
      { x: 0.5, y: 0.0, z: 0.667 },
      { x: 0.0, y: 0.0, z: 0.5 },
    ],
    latticeType: "hexagonal",
    cOverA: 1.73,
    baseLatticeFactor: 4.2,
  },
  {
    name: "Clathrate-H8",
    hPerMetal: 8,
    metalFrac: [
      { x: 0.0, y: 0.0, z: 0.0 },
    ],
    hydrogenFrac: [
      { x: 0.185, y: 0.185, z: 0.185 },
      { x: 0.815, y: 0.185, z: 0.185 },
      { x: 0.185, y: 0.815, z: 0.185 },
      { x: 0.185, y: 0.185, z: 0.815 },
      { x: 0.815, y: 0.815, z: 0.185 },
      { x: 0.815, y: 0.185, z: 0.815 },
      { x: 0.185, y: 0.815, z: 0.815 },
      { x: 0.815, y: 0.815, z: 0.815 },
    ],
    latticeType: "cubic",
    cOverA: 1.0,
    baseLatticeFactor: 3.8,
  },
  {
    name: "H4-Tetrahedral",
    hPerMetal: 4,
    metalFrac: [
      { x: 0.0, y: 0.0, z: 0.0 },
    ],
    hydrogenFrac: [
      { x: 0.25, y: 0.25, z: 0.25 },
      { x: 0.75, y: 0.75, z: 0.25 },
      { x: 0.75, y: 0.25, z: 0.75 },
      { x: 0.25, y: 0.75, z: 0.75 },
    ],
    latticeType: "cubic",
    cOverA: 1.0,
    baseLatticeFactor: 3.5,
  },
];

function selectHydrideCage(hPerMetal: number): HydrideCageMotif {
  const sorted = [...HYDRIDE_CAGE_LIBRARY].sort(
    (a, b) => Math.abs(a.hPerMetal - hPerMetal) - Math.abs(b.hPerMetal - hPerMetal)
  );
  return sorted[0];
}

function generateHydrideCageStructure(
  formula: string,
  counts: Record<string, number>
): { atoms: AtomPosition[]; prototype: string } | null {
  const elements = Object.keys(counts);
  const metals = elements.filter(el => el !== "H");
  const hCount = Math.round(counts["H"] || 0);
  if (metals.length === 0 || hCount === 0) return null;

  const totalMetalCount = metals.reduce((s, el) => s + Math.round(counts[el] || 0), 0);
  const hPerMetal = hCount / totalMetalCount;
  if (hPerMetal < 4) return null;

  const cage = selectHydrideCage(hPerMetal);

  const metalRadiiSum = metals.reduce((s, el) => {
    const r = COVALENT_RADII[el] ?? 1.4;
    return s + r;
  }, 0);
  const avgMetalRadius = metalRadiiSum / metals.length;
  const hRadius = 0.31;
  const latticeA = cage.baseLatticeFactor * (avgMetalRadius / 1.5);
  const minLattice = 2.0 * avgMetalRadius + 1.5 * hRadius;
  const a = Math.max(latticeA, minLattice, 3.0);
  const c = a * cage.cOverA;

  const metalSiteCount = cage.metalFrac.length;
  const hSiteCount = cage.hydrogenFrac.length;

  const nCopies = Math.max(1, Math.round(totalMetalCount / metalSiteCount));

  const atoms: AtomPosition[] = [];

  const metalList: string[] = [];
  for (const el of metals) {
    const n = Math.round(counts[el] || 0);
    for (let i = 0; i < n; i++) metalList.push(el);
  }

  let metalIdx = 0;
  let hPlaced = 0;

  const gridDim = Math.max(1, Math.ceil(Math.cbrt(nCopies)));
  for (let copy = 0; copy < nCopies; copy++) {
    const ix = copy % gridDim;
    const iy = Math.floor(copy / gridDim) % gridDim;
    const iz = Math.floor(copy / (gridDim * gridDim));
    const offsetX = ix * a;
    const offsetY = iy * a;
    const offsetZ = iz * a;

    for (let i = 0; i < metalSiteCount && metalIdx < metalList.length; i++) {
      const pos = cage.metalFrac[i];
      let x: number, y: number, z: number;
      if (cage.latticeType === "hexagonal") {
        x = a * pos.x + a * 0.5 * pos.y + offsetX;
        y = a * (Math.sqrt(3) / 2) * pos.y + offsetY;
        z = c * pos.z + offsetZ;
      } else {
        x = a * pos.x + offsetX;
        y = a * pos.y + offsetY;
        z = a * pos.z + offsetZ;
      }
      atoms.push({ element: metalList[metalIdx], x, y, z });
      metalIdx++;
    }

    const hPerCopy = Math.min(Math.round(hCount / nCopies), hSiteCount);
    for (let i = 0; i < hPerCopy && hPlaced < hCount; i++) {
      const pos = cage.hydrogenFrac[i % hSiteCount];
      let x: number, y: number, z: number;
      if (cage.latticeType === "hexagonal") {
        x = a * pos.x + a * 0.5 * pos.y + offsetX;
        y = a * (Math.sqrt(3) / 2) * pos.y + offsetY;
        z = c * pos.z + offsetZ;
      } else {
        x = a * pos.x + offsetX;
        y = a * pos.y + offsetY;
        z = a * pos.z + offsetZ;
      }
      let tooClose = false;
      for (const existing of atoms) {
        const dx = x - existing.x;
        const dy = y - existing.y;
        const dz = z - existing.z;
        if (Math.sqrt(dx * dx + dy * dy + dz * dz) < 0.5) { tooClose = true; break; }
      }
      if (!tooClose) {
        atoms.push({ element: "H", x, y, z });
        hPlaced++;
      }
    }
  }

  let overflowIdx = 0;
  while (hPlaced < hCount && nCopies > 0) {
    const pos = cage.hydrogenFrac[hPlaced % hSiteCount];
    const copy = Math.floor(hPlaced / hSiteCount) % nCopies;
    const offsetX = (copy % 2) * a;
    const offsetY = (Math.floor(copy / 2) % 2) * a;
    const offsetZ = Math.floor(copy / 4) * a;

    let placed = false;
    for (let retry = 0; retry < 10 && !placed; retry++) {
      const perturbAngle = overflowIdx * 2.399 + retry * 1.1;
      const perturbR = 0.5 + 0.2 * (overflowIdx % 5) + retry * 0.15;
      const px = perturbR * Math.cos(perturbAngle);
      const py = perturbR * Math.sin(perturbAngle);
      const pz = perturbR * Math.cos(perturbAngle + 1.5);
      let x: number, y: number, z: number;
      if (cage.latticeType === "hexagonal") {
        x = a * pos.x + a * 0.5 * pos.y + offsetX + px;
        y = a * (Math.sqrt(3) / 2) * pos.y + offsetY + py;
        z = c * pos.z + offsetZ + pz;
      } else {
        x = a * pos.x + offsetX + px;
        y = a * pos.y + offsetY + py;
        z = a * pos.z + offsetZ + pz;
      }

      let tooClose = false;
      for (const existing of atoms) {
        const dx = x - existing.x;
        const dy = y - existing.y;
        const dz = z - existing.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 0.5) { tooClose = true; break; }
      }
      if (!tooClose) {
        atoms.push({ element: "H", x, y, z });
        placed = true;
      }
    }
    if (placed) hPlaced++;
    else break;
    overflowIdx++;
  }

  if (atoms.length < 2) return null;

  console.log(`[DFT] ${formula}: Using hydride cage motif ${cage.name} (H/metal=${hPerMetal.toFixed(1)}, ${atoms.length} atoms)`);
  return { atoms, prototype: `hydride-cage-${cage.name}` };
}

function deduplicateSites(atoms: AtomPosition[]): AtomPosition[] {
  const TOLERANCE = 0.05;
  const result: AtomPosition[] = [];
  for (const atom of atoms) {
    let isDuplicate = false;
    for (const existing of result) {
      const dx = Math.abs(atom.x - existing.x);
      const dy = Math.abs(atom.y - existing.y);
      const dz = Math.abs(atom.z - existing.z);
      if (dx < TOLERANCE && dy < TOLERANCE && dz < TOLERANCE) {
        isDuplicate = true;
        break;
      }
    }
    if (!isDuplicate) {
      result.push({ ...atom });
    }
  }
  if (result.length < atoms.length) {
    console.log(`[DFT] deduplicateSites: Dropped ${atoms.length - result.length} duplicate atom(s) (${atoms.length} → ${result.length})`);
  }
  return result;
}

function getMinInteratomicDistance(el1: string, el2: string): number {
  const r1 = COVALENT_RADII[el1] ?? 1.4;
  const r2 = COVALENT_RADII[el2] ?? 1.4;
  const bondLength = r1 + r2;
  const isHydrogenPair = el1 === "H" || el2 === "H";
  const factor = isHydrogenPair ? 0.70 : 0.80;
  return Math.max(bondLength * factor, isHydrogenPair ? 0.6 : 0.9);
}

function computePairwiseDistances(atoms: AtomPosition[]): { minDist: number; minRatio: number; worstI: number; worstJ: number; pairI: number; pairJ: number } {
  let minDist = Infinity;
  let minRatio = Infinity;
  let pairI = -1;
  let pairJ = -1;
  let worstI = -1;
  let worstJ = -1;
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < minDist) {
        minDist = dist;
        pairI = i;
        pairJ = j;
      }
      const minAllowed = getMinInteratomicDistance(atoms[i].element, atoms[j].element);
      const ratio = dist / minAllowed;
      if (ratio < minRatio) {
        minRatio = ratio;
        worstI = i;
        worstJ = j;
      }
    }
  }
  return { minDist, minRatio, worstI, worstJ, pairI, pairJ };
}

function computeBoundingVolume(atoms: AtomPosition[]): number {
  if (atoms.length < 2) return 0;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const a of atoms) {
    const r = COVALENT_RADII[a.element] ?? 1.4;
    if (a.x - r < minX) minX = a.x - r;
    if (a.y - r < minY) minY = a.y - r;
    if (a.z - r < minZ) minZ = a.z - r;
    if (a.x + r > maxX) maxX = a.x + r;
    if (a.y + r > maxY) maxY = a.y + r;
    if (a.z + r > maxZ) maxZ = a.z + r;
  }
  const lx = Math.max(maxX - minX, 1.0);
  const ly = Math.max(maxY - minY, 1.0);
  const lz = Math.max(maxZ - minZ, 1.0);
  const boxVol = lx * ly * lz;
  const aspect = Math.max(lx, ly, lz) / Math.min(lx, ly, lz);
  if (aspect > 3.0) {
    const avgSide = Math.cbrt(boxVol);
    return avgSide * avgSide * avgSide * 0.85;
  }
  return boxVol;
}

function scaleStructure(atoms: AtomPosition[], factor: number): AtomPosition[] {
  let cx = 0, cy = 0, cz = 0;
  for (const a of atoms) { cx += a.x; cy += a.y; cz += a.z; }
  cx /= atoms.length; cy /= atoms.length; cz /= atoms.length;
  return atoms.map(a => ({
    element: a.element,
    x: cx + (a.x - cx) * factor,
    y: cy + (a.y - cy) * factor,
    z: cz + (a.z - cz) * factor,
  }));
}

function validateAndFixStructure(atoms: AtomPosition[], formula: string): AtomPosition[] | null {
  if (atoms.length < 2) return atoms;

  const hasHydrogen = atoms.some(a => a.element === "H");
  const minVol = hasHydrogen ? MIN_VOLUME_PER_ATOM_HYDRIDE : MIN_VOLUME_PER_ATOM;

  const counts = parseFormula(formula);
  const expectedVolume = computeExpectedVolume(counts);
  const targetVolPerAtom = Math.max(minVol, expectedVolume / Math.max(atoms.length, 1));

  let current = atoms;

  for (let attempt = 0; attempt < MAX_SCALE_ATTEMPTS; attempt++) {
    const { minDist, minRatio } = computePairwiseDistances(current);
    const volume = computeBoundingVolume(current);
    const volumePerAtom = volume / current.length;

    const distOk = minRatio >= 0.99;
    const volOk = volumePerAtom >= targetVolPerAtom - 0.1;

    if (distOk && volOk) {
      const volRatioCheck = validateVolumeRatio(volume, expectedVolume);
      if (volRatioCheck.valid) return current;
    }

    const damping = 1.0 / (1.0 + attempt * 0.3);
    const volRatio = volumePerAtom / targetVolPerAtom;

    let scaleFactor = 1.0;
    if (!distOk) {
      const rawExpansion = 1.05 / Math.max(minRatio, 0.01);
      if (volRatio > 1.3) {
        scaleFactor = Math.max(scaleFactor, 1.0 + (rawExpansion - 1.0) * 0.4 * damping);
      } else {
        scaleFactor = Math.max(scaleFactor, 1.0 + (rawExpansion - 1.0) * damping);
      }
    } else if (!volOk) {
      const neededFactor = Math.cbrt(targetVolPerAtom / Math.max(volumePerAtom, 0.01));
      scaleFactor = Math.max(scaleFactor, 1.0 + (neededFactor - 1.0) * damping);
    } else if (volRatio > 1.6) {
      const contractFactor = Math.cbrt(1.0 / volRatio);
      scaleFactor = 1.0 + (contractFactor - 1.0) * damping;
    }

    scaleFactor = Math.max(scaleFactor, 0.7);
    scaleFactor = Math.min(scaleFactor, 1.5);
    console.log(`[DFT] ${formula}: Structure validation attempt ${attempt + 1} — minDist=${minDist.toFixed(3)}Å, ratio=${minRatio.toFixed(2)}, vol/atom=${volumePerAtom.toFixed(1)}ų (target=${targetVolPerAtom.toFixed(1)}) — scaling by ${scaleFactor.toFixed(2)} (damping=${damping.toFixed(2)})`);
    current = scaleStructure(current, scaleFactor);
  }

  const { minDist, minRatio } = computePairwiseDistances(current);
  const volume = computeBoundingVolume(current);
  const volumePerAtom = volume / current.length;

  if (minRatio < 0.99 || volumePerAtom < minVol - 0.1) {
    console.log(`[DFT] ${formula}: Structure REJECTED after ${MAX_SCALE_ATTEMPTS} fix attempts — minDist=${minDist.toFixed(3)}Å, ratio=${minRatio.toFixed(2)}, vol/atom=${volumePerAtom.toFixed(1)}ų`);
    return null;
  }

  return current;
}

function checkVolumeRatioForAtoms(atoms: AtomPosition[], counts: Record<string, number>, label: string, formula: string): boolean {
  if (atoms.length < 2) return false;
  const volume = computeBoundingVolume(atoms);
  const expectedVol = computeExpectedVolume(counts);
  const { valid, ratio } = validateVolumeRatio(volume, expectedVol);
  if (!valid) {
    console.log(`[DFT] ${formula}: Volume ratio check FAILED for ${label} — generated=${volume.toFixed(1)}ų, expected=${expectedVol.toFixed(1)}ų, ratio=${ratio.toFixed(2)} (must be 0.5-2.0)`);
  }
  return valid;
}

function checkRadiusCompatibility(elements: string[]): boolean {
  const nonH = elements.filter(e => e !== "H");
  if (nonH.length < 2) return true;
  const radii = nonH.map(e => COVALENT_RADII[e] ?? 1.3);
  const maxR = Math.max(...radii);
  const minR = Math.min(...radii);
  if (maxR / minR > 3.0) return false;
  return true;
}

function generateCrystalStructure(formula: string): { atoms: AtomPosition[]; prototype: string } {
  const counts = parseFormula(formula);
  const elements = Object.keys(counts);

  if (elements.length === 0) {
    return { atoms: [], prototype: "empty" };
  }

  if (elements.length > 5) {
    console.log(`[DFT] ${formula}: Rejected — ${elements.length} distinct elements exceeds limit of 5`);
    return { atoms: [], prototype: "rejected-too-complex" };
  }

  const totalAtomCount = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  if (totalAtomCount > 20) {
    console.log(`[DFT] ${formula}: Rejected — ${totalAtomCount} total atoms exceeds limit of 20`);
    return { atoms: [], prototype: "rejected-too-complex" };
  }

  if (!checkRadiusCompatibility(elements)) {
    console.log(`[DFT] ${formula}: Radius incompatibility — non-H element radii ratio > 3.0`);
    return { atoms: [], prototype: "rejected-radius" };
  }

  const bvsResult = computeBondValenceSum(formula);
  if (bvsResult.deviation > 1.0) {
    console.log(`[DFT] ${formula}: BVS deviation too high (${bvsResult.deviation.toFixed(2)} > 1.0) — rejecting before structure generation`);
    return { atoms: [], prototype: "rejected-bvs" };
  }

  const ionicResult = checkIonicRadiusCompatibility(formula);
  if (!ionicResult.compatible) {
    console.log(`[DFT] ${formula}: Ionic radius incompatibility (ratio=${ionicResult.radiusRatio.toFixed(2)}, tolerance=${ionicResult.toleranceFactor?.toFixed(2) ?? "N/A"}) — rejecting`);
    return { atoms: [], prototype: "rejected-ionic-radius" };
  }

  prototypeAttempts++;

  const chemProto = fillPrototype(formula);
  if (chemProto && chemProto.atoms.length >= 2) {
    const atomPositions: AtomPosition[] = chemProto.atoms.map(a => ({
      element: a.element,
      x: a.x,
      y: a.y,
      z: a.z,
    }));
    const deduped = deduplicateSites(atomPositions);
    const validated = validateAndFixStructure(deduped, formula);
    if (!validated) return { atoms: [], prototype: "rejected-overlap" };
    if (!checkVolumeRatioForAtoms(validated, counts, chemProto.templateName, formula)) {
      return { atoms: [], prototype: "rejected-volume-ratio" };
    }
    prototypeSuccesses++;
    return { atoms: validated, prototype: `${chemProto.templateName} prototype lattice` };
  }

  const hCount = counts["H"] || 0;
  const metalElements = elements.filter(el => el !== "H");
  const totalMetalCount = metalElements.reduce((s, el) => s + Math.round(counts[el] || 0), 0);
  if (hCount > 0 && totalMetalCount > 0 && (hCount / totalMetalCount) >= 4) {
    const hydrideCage = generateHydrideCageStructure(formula, counts);
    if (hydrideCage && hydrideCage.atoms.length >= 2) {
      const dedupedHydride = deduplicateSites(hydrideCage.atoms);
      const validated = validateAndFixStructure(dedupedHydride, formula);
      if (validated) {
        prototypeSuccesses++;
        return { atoms: validated, prototype: hydrideCage.prototype };
      }
    }
  }

  const protoMatch = matchPrototype(counts);
  if (protoMatch) {
    const atoms = deduplicateSites(buildStructureFromPrototype(protoMatch.proto, protoMatch.siteMap, elements, counts));
    if (atoms.length >= 2) {
      const validated = validateAndFixStructure(atoms, formula);
      if (!validated) return { atoms: [], prototype: "rejected-overlap" };
      if (!checkVolumeRatioForAtoms(validated, counts, protoMatch.proto.name, formula)) {
        return { atoms: [], prototype: "rejected-volume-ratio" };
      }
      prototypeSuccesses++;
      return { atoms: validated, prototype: protoMatch.proto.name };
    }
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  const scaledCounts: Record<string, number> = {};
  if (totalAtoms > 20) {
    const scaleFactor = 20 / totalAtoms;
    for (const [el, n] of Object.entries(counts)) {
      scaledCounts[el] = Math.max(1, Math.round(n * scaleFactor));
    }
  } else {
    Object.assign(scaledCounts, counts);
    for (const el of Object.keys(scaledCounts)) {
      scaledCounts[el] = Math.max(1, Math.round(scaledCounts[el]));
    }
  }

  const scaledMatch = matchPrototype(scaledCounts);
  if (scaledMatch) {
    const atoms = deduplicateSites(buildStructureFromPrototype(scaledMatch.proto, scaledMatch.siteMap, elements, scaledCounts));
    if (atoms.length >= 2) {
      const validated = validateAndFixStructure(atoms, formula);
      if (!validated) return { atoms: [], prototype: "rejected-overlap" };
      if (!checkVolumeRatioForAtoms(validated, scaledCounts, scaledMatch.proto.name + "-scaled", formula)) {
        return { atoms: [], prototype: "rejected-volume-ratio" };
      }
      prototypeSuccesses++;
      return { atoms: validated, prototype: scaledMatch.proto.name + "-scaled" };
    }
  }

  chemistryMatchAttempts++;
  const chemMatch = selectBestPrototypeByChemistry(scaledCounts, elements);
  if (chemMatch) {
    const atoms = deduplicateSites(buildStructureFromPrototype(chemMatch.proto, chemMatch.siteMap, elements, scaledCounts));
    if (atoms.length >= 2) {
      const validated = validateAndFixStructure(atoms, formula);
      if (validated) {
        if (checkVolumeRatioForAtoms(validated, scaledCounts, chemMatch.proto.name + "-chem", formula)) {
          prototypeSuccesses++;
          chemistryMatchSuccesses++;
          console.log(`[DFT] ${formula}: Chemistry-based prototype match → ${chemMatch.proto.name}`);
          return { atoms: validated, prototype: chemMatch.proto.name + "-chem" };
        }
      }
    }
  }

  const { atoms, proto } = buildGenericStructure(scaledCounts);
  const validated = validateAndFixStructure(atoms, formula);
  if (!validated) return { atoms: [], prototype: "rejected-overlap" };
  if (!checkVolumeRatioForAtoms(validated, scaledCounts, proto, formula)) {
    return { atoms: [], prototype: "rejected-volume-ratio" };
  }
  prototypeSuccesses++;
  return { atoms: validated, prototype: proto };
}

function writeXYZ(atoms: AtomPosition[], filepath: string, comment: string = ""): void {
  const lines = [
    String(atoms.length),
    comment || "Generated structure",
    ...atoms.map(a => `${a.element}  ${a.x.toFixed(6)}  ${a.y.toFixed(6)}  ${a.z.toFixed(6)}`),
  ];
  fs.writeFileSync(filepath, lines.join("\n") + "\n");
}

function parseXtbOutput(output: string): Partial<DFTResult> {
  const result: Partial<DFTResult> = {
    converged: false,
    charges: {},
  };

  const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
  if (energyMatch) {
    result.totalEnergy = parseFloat(energyMatch[1]);
  }

  const gapMatch = output.match(/HOMO-LUMO GAP\s+([-\d.]+)\s+eV/);
  if (gapMatch) {
    result.homoLumoGap = parseFloat(gapMatch[1]);
    result.isMetallic = result.homoLumoGap < 0.5;
  }

  const homoMatch = output.match(/\(HOMO\)\s+([-\d.]+)\s+eV/);
  const lumoMatch = output.match(/\(LUMO\)\s+([-\d.]+)\s+eV/);
  if (homoMatch) result.homo = parseFloat(homoMatch[1]);
  if (lumoMatch) result.lumo = parseFloat(lumoMatch[1]);

  if (result.homo != null && result.lumo != null) {
    result.fermiLevel = (result.homo + result.lumo) / 2;
  }

  const dipoleAlt = output.match(/molecular dipole:.*?tot \(Debye\).*?full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)/s);
  if (dipoleAlt) result.dipoleMoment = parseFloat(dipoleAlt[1]);

  const chargeBlock = output.match(/#\s+Z\s+covCN\s+q\s+C6AA\s+.*?\n([\s\S]*?)(?:\n\n|\nWiberg|\nmolecular)/);
  if (chargeBlock) {
    const lines = chargeBlock[1].trim().split("\n");
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 4) {
        const el = parts[1];
        const charge = parseFloat(parts[3]);
        if (!isNaN(charge) && el.match(/^[A-Z][a-z]?$/)) {
          if (!result.charges![el]) result.charges![el] = 0;
          result.charges![el] += charge;
        }
      }
    }
  }

  if (output.includes("normal termination of xtb")) {
    result.converged = true;
  }

  const wallMatch = output.match(/wall-time:\s+\d+ d,\s+\d+ h,\s+\d+ min,\s+([\d.]+) sec/);
  if (wallMatch) {
    result.wallTimeSeconds = parseFloat(wallMatch[1]);
  }

  return result;
}

const xtbResultCache = new Map<string, DFTResult>();
const optimizedStructureCache = new Map<string, OptimizationResult>();
const CACHE_MAX = 500;
const OPT_TIMEOUT_MS = 30_000;

function parseOptimizedXYZ(filepath: string): AtomPosition[] {
  if (!fs.existsSync(filepath)) return [];
  const content = fs.readFileSync(filepath, "utf-8").trim();
  const lines = content.split("\n");
  if (lines.length < 3) return [];

  const atomCount = parseInt(lines[0].trim(), 10);
  if (isNaN(atomCount) || atomCount < 1) return [];

  const atoms: AtomPosition[] = [];
  for (let i = 2; i < Math.min(lines.length, atomCount + 2); i++) {
    const parts = lines[i].trim().split(/\s+/);
    if (parts.length >= 4) {
      const element = parts[0];
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]);
      const z = parseFloat(parts[3]);
      if (element.match(/^[A-Z][a-z]?$/) && !isNaN(x) && !isNaN(y) && !isNaN(z)) {
        atoms.push({ element, x, y, z });
      }
    }
  }
  return atoms;
}

function parseOptimizationOutput(output: string): { energyChange: number; gradientNorm: number; iterations: number; converged: boolean } {
  let energyChange = 0;
  let gradientNorm = 0;
  let iterations = 0;
  let converged = false;

  const iterMatch = output.match(/(\d+)\s+ANC optimizer/g);
  if (iterMatch) {
    iterations = iterMatch.length;
  }
  const cycleMatch = output.match(/\.\.\.\. convergence criteria satisfied after\s+(\d+)\s+iterations/);
  if (cycleMatch) {
    iterations = parseInt(cycleMatch[1], 10);
    converged = true;
  }

  if (output.includes("GEOMETRY OPTIMIZATION CONVERGED")) {
    converged = true;
  }

  const deMatch = output.match(/Econv\s*=.*?([-\d.eE+]+)\s/);
  if (deMatch) {
    energyChange = Math.abs(parseFloat(deMatch[1]));
  }
  const lastEnergyChanges = output.match(/ΔE\s+([-\d.eE+]+)/g);
  if (lastEnergyChanges && lastEnergyChanges.length > 0) {
    const lastDE = lastEnergyChanges[lastEnergyChanges.length - 1].match(/([-\d.eE+]+)/);
    if (lastDE) energyChange = Math.abs(parseFloat(lastDE[1]));
  }

  const gradMatch = output.match(/(?:gradient norm|grad\. norm:)\s+([-\d.eE+]+)/gi);
  if (gradMatch && gradMatch.length > 0) {
    const lastGrad = gradMatch[gradMatch.length - 1].match(/([-\d.eE+]+)/);
    if (lastGrad) gradientNorm = Math.abs(parseFloat(lastGrad[1]));
  }

  if (output.includes("normal termination of xtb") && !output.includes("FAILED")) {
    converged = true;
  }

  return { energyChange, gradientNorm, iterations, converged };
}

function applyPressureScaling(atoms: AtomPosition[], formula: string, pressureGpa: number): AtomPosition[] {
  if (pressureGpa <= 0) return atoms;
  try {
    const bm = relaxStructureAtPressure(formula, pressureGpa);
    const eta = bm.compressedVolume > 0 && bm.bulkModulus > 0
      ? Math.pow(bm.compressedVolume / (bm.compressedLattice.a * bm.compressedLattice.b * bm.compressedLattice.c / Math.pow(Math.pow(bm.compressedVolume, 1/3) / bm.compressedLattice.a, 3) || 1), 1/3)
      : 1.0;
    const cubicEta = (() => {
      const B0p = 4.0;
      const pOverB = pressureGpa / Math.max(10, bm.bulkModulus);
      const inner = 1 + B0p * pOverB;
      if (inner > 0) return Math.pow(Math.pow(inner, -1 / B0p), 1 / 3);
      return Math.pow(0.5, 1 / 3);
    })();
    const scale = Math.max(0.8, Math.min(1.0, cubicEta));
    return atoms.map(a => ({
      element: a.element,
      x: a.x * scale,
      y: a.y * scale,
      z: a.z * scale,
    }));
  } catch {
    return atoms;
  }
}

export async function runXTBOptimization(formula: string, pressureGpa: number = 0): Promise<OptimizationResult | null> {
  if (!isDFTAvailable()) return null;

  const pressureTag = pressureGpa > 0 ? `_P${Math.round(pressureGpa)}` : "";
  const cacheKey = formula.replace(/\s+/g, "") + pressureTag;
  if (optimizedStructureCache.has(cacheKey)) {
    return optimizedStructureCache.get(cacheKey)!;
  }

  if (!fs.existsSync(XTB_BIN)) {
    console.log(`[DFT] xTB binary not found at ${XTB_BIN}`);
    return null;
  }

  const startTime = Date.now();
  const calcId = `opt_${cacheKey.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`;
  const calcDir = path.join(WORK_DIR, calcId);
  fs.mkdirSync(calcDir, { recursive: true });

  let { atoms, prototype } = generateCrystalStructure(formula);
  if (atoms.length < 2) return null;

  if (pressureGpa > 0) {
    atoms = applyPressureScaling(atoms, formula, pressureGpa);
    console.log(`[DFT] ${formula}: Pressure-scaled geometry at ${pressureGpa} GPa for xTB optimization`);
  }

  const xyzPath = path.join(calcDir, "input.xyz");
  writeXYZ(atoms, xyzPath, `${formula} [${prototype}] optimization${pressureGpa > 0 ? ` @ ${pressureGpa} GPa` : ""}`);

  try {
    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    const cmd = `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight 2>&1`;
    if (!cmd.includes("/xtb ")) {
      console.log(`[DFT] WARNING: Geometry optimization command malformed: ${cmd.slice(0, 200)}`);
      return null;
    }

    const output = execSync(
      cmd,
      { timeout: OPT_TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
    ).toString();

    const optInfo = parseOptimizationOutput(output);

    const optXyzPath = path.join(calcDir, "xtbopt.xyz");
    let optimizedAtoms = parseOptimizedXYZ(optXyzPath);

    if (optimizedAtoms.length === 0) {
      const altPath = path.join(calcDir, "xtbopt.coord");
      if (fs.existsSync(altPath)) {
        optimizedAtoms = parseOptimizedXYZ(altPath);
      }
    }

    if (optimizedAtoms.length === 0) {
      optimizedAtoms = atoms;
    }

    let optimizedEnergy = 0;
    const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
    if (energyMatch) {
      optimizedEnergy = parseFloat(energyMatch[1]);
    }

    if (pressureGpa > 0 && optimizedAtoms.length > 0) {
      const xExt = optimizedAtoms.map(a => a.x);
      const yExt = optimizedAtoms.map(a => a.y);
      const zExt = optimizedAtoms.map(a => a.z);
      const vol_A3 = Math.max(1, (Math.max(...xExt) - Math.min(...xExt) + 2.0) *
        (Math.max(...yExt) - Math.min(...yExt) + 2.0) *
        (Math.max(...zExt) - Math.min(...zExt) + 2.0));
      const eV_per_GPa_A3 = 0.006242;
      const pvCorrection = pressureGpa * vol_A3 * eV_per_GPa_A3;
      const pvHartree = pvCorrection / 27.2114;
      optimizedEnergy += pvHartree;
      console.log(`[DFT] ${formula}: PV correction at ${pressureGpa} GPa: +${pvHartree.toFixed(6)} Eh (V~${vol_A3.toFixed(1)} A^3)`);
    }

    const result: OptimizationResult = {
      optimizedAtoms,
      optimizedEnergy,
      converged: optInfo.converged,
      energyChange: optInfo.energyChange,
      gradientNorm: optInfo.gradientNorm,
      iterations: optInfo.iterations,
      wallTimeSeconds: (Date.now() - startTime) / 1000,
    };

    if (atoms.length >= 2 && optimizedAtoms.length >= 2) {
      try {
        const initialPositions = atoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z }));
        const relaxedPositions = optimizedAtoms.map(a => ({ element: a.element, x: a.x, y: a.y, z: a.z }));
        const xExtent = (arr: AtomPosition[]) => {
          let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
          for (const a of arr) {
            if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
            if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
            if (a.z < minZ) minZ = a.z; if (a.z > maxZ) maxZ = a.z;
          }
          return {
            a: Math.max(2.5, maxX - minX + 2.0),
            b: Math.max(2.5, maxY - minY + 2.0),
            c: Math.max(2.5, maxZ - minZ + 2.0),
            alpha: 90, beta: 90, gamma: 90,
          };
        };
        const beforeLattice = xExtent(atoms);
        const afterLattice = xExtent(optimizedAtoms);
        const distortion = analyzeDistortion(
          formula,
          beforeLattice,
          afterLattice,
          initialPositions,
          relaxedPositions,
          prototype === "Perovskite" ? "Pm-3m" : prototype === "A15" ? "Pm-3m" :
            prototype === "NaCl" ? "Fm-3m" : prototype === "AlB2" ? "P6/mmm" :
            prototype === "ThCr2Si2" ? "I4/mmm" : undefined,
        );
        result.distortion = distortion;
        recordDistortionAnalysis(distortion);
        if (distortion.overallLevel !== "none") {
          console.log(`[DFT] ${formula}: Distortion detected (${distortion.overallLevel}, score=${distortion.overallScore}, meanDisp=${distortion.atomicDistortion?.meanDisplacement?.toFixed(4) ?? "N/A"}A, strain=${distortion.latticeDistortion.strainMagnitude.toFixed(5)}, vol=${distortion.latticeDistortion.volumeChangePct.toFixed(2)}%)${distortion.symmetryReduction?.symmetryBroken ? ` [symmetry broken: ${distortion.symmetryReduction.systemBefore}->${distortion.symmetryReduction.systemAfter}]` : ""}`);
        }
      } catch {}
    }

    if (result.converged && optimizedAtoms.length >= 2) {
      optimizedStructureCache.set(cacheKey, result);
      if (optimizedStructureCache.size > CACHE_MAX) {
        const oldest = optimizedStructureCache.keys().next().value;
        if (oldest) optimizedStructureCache.delete(oldest);
      }
    }

    return result;
  } catch (err: any) {
    const isTimeout = err.killed || (err.message && err.message.includes("TIMEOUT"));
    console.log(`[DFT] ${formula}: Geometry optimization ${isTimeout ? "timed out" : "failed"}: ${err.message?.slice(0, 100) || String(err)}`);
    return null;
  } finally {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
  }
}

export interface LandscapeMinimum {
  perturbationIndex: number;
  perturbationScale: number;
  energy: number;
  energyDiffFromRef: number;
  converged: boolean;
  rmsDisplacementFromRef: number;
  isDifferentMinimum: boolean;
}

export interface EnergyLandscapeResult {
  formula: string;
  referenceEnergy: number;
  perturbationsRun: number;
  uniqueMinima: number;
  minima: LandscapeMinimum[];
  multipleMinima: boolean;
  energySpread: number;
  lowestEnergyDiff: number;
  distortionModesExist: boolean;
  wallTimeSeconds: number;
}

const landscapeCache = new Map<string, EnergyLandscapeResult>();
const LANDSCAPE_PERTURBATION_COUNT = 5;
const LANDSCAPE_ENERGY_THRESHOLD = 0.005;
const LANDSCAPE_DISP_THRESHOLD = 0.15;

export async function runLandscapeExploration(formula: string): Promise<EnergyLandscapeResult | null> {
  const cacheKey = formula.replace(/\s+/g, "");
  if (landscapeCache.has(cacheKey)) return landscapeCache.get(cacheKey)!;

  if (!isDFTAvailable() || !fs.existsSync(XTB_BIN)) return null;

  const startTime = Date.now();

  const refResult = await runXTBOptimization(formula);
  if (!refResult || !refResult.converged || refResult.optimizedAtoms.length < 2) return null;

  const refAtoms = refResult.optimizedAtoms;
  const refEnergy = refResult.optimizedEnergy;

  const minima: LandscapeMinimum[] = [];
  const perturbationScales = [0.05, 0.10, 0.15, 0.20, 0.30];

  for (let pi = 0; pi < Math.min(LANDSCAPE_PERTURBATION_COUNT, perturbationScales.length); pi++) {
    const scale = perturbationScales[pi];
    const calcId = `landscape_${cacheKey}_p${pi}_${Date.now()}`;
    const calcDir = path.join(WORK_DIR, calcId);
    fs.mkdirSync(calcDir, { recursive: true });

    try {
      const perturbedAtoms: AtomPosition[] = refAtoms.map(a => ({
        element: a.element,
        x: a.x + (Math.random() - 0.5) * 2 * scale,
        y: a.y + (Math.random() - 0.5) * 2 * scale,
        z: a.z + (Math.random() - 0.5) * 2 * scale,
      }));

      const xyzPath = path.join(calcDir, "input.xyz");
      writeXYZ(perturbedAtoms, xyzPath, `${formula} perturbation ${pi} scale=${scale}`);

      const env: Record<string, string> = {
        ...process.env as Record<string, string>,
        XTBHOME: XTB_HOME,
        XTBPATH: XTB_PARAM,
        OMP_NUM_THREADS: "1",
        OMP_STACKSIZE: "512M",
      };

      const cmd = `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight 2>&1`;
      const output = execSync(cmd, { timeout: OPT_TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }).toString();

      const optInfo = parseOptimizationOutput(output);
      let optEnergy = 0;
      const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
      if (energyMatch) optEnergy = parseFloat(energyMatch[1]);

      const optXyzPath = path.join(calcDir, "xtbopt.xyz");
      let optimizedAtoms = parseOptimizedXYZ(optXyzPath);
      if (optimizedAtoms.length === 0) optimizedAtoms = perturbedAtoms;

      let rmsDisp = 0;
      const nAtoms = Math.min(refAtoms.length, optimizedAtoms.length);
      if (nAtoms > 0) {
        let sumSq = 0;
        for (let i = 0; i < nAtoms; i++) {
          const dx = optimizedAtoms[i].x - refAtoms[i].x;
          const dy = optimizedAtoms[i].y - refAtoms[i].y;
          const dz = optimizedAtoms[i].z - refAtoms[i].z;
          sumSq += dx * dx + dy * dy + dz * dz;
        }
        rmsDisp = Math.sqrt(sumSq / nAtoms);
      }

      const energyDiff = Math.abs(optEnergy - refEnergy);
      const isDifferent = energyDiff > LANDSCAPE_ENERGY_THRESHOLD && rmsDisp > LANDSCAPE_DISP_THRESHOLD;

      minima.push({
        perturbationIndex: pi,
        perturbationScale: scale,
        energy: optEnergy,
        energyDiffFromRef: optEnergy - refEnergy,
        converged: optInfo.converged,
        rmsDisplacementFromRef: Math.round(rmsDisp * 10000) / 10000,
        isDifferentMinimum: isDifferent,
      });
    } catch {
    } finally {
      try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
    }
  }

  const uniqueMinima = minima.filter(m => m.isDifferentMinimum).length + 1;
  const energies = minima.map(m => m.energy).filter(e => e !== 0);
  const energySpread = energies.length > 0 ? Math.max(...energies) - Math.min(...energies) : 0;
  const convergedMinima = minima.filter(m => m.converged);
  const lowestDiff = convergedMinima.length > 0
    ? Math.min(...convergedMinima.map(m => Math.abs(m.energyDiffFromRef)))
    : 0;

  const result: EnergyLandscapeResult = {
    formula,
    referenceEnergy: refEnergy,
    perturbationsRun: minima.length,
    uniqueMinima,
    minima,
    multipleMinima: uniqueMinima > 1,
    energySpread: Math.round(energySpread * 100000) / 100000,
    lowestEnergyDiff: Math.round(lowestDiff * 100000) / 100000,
    distortionModesExist: uniqueMinima > 1 || energySpread > 0.01,
    wallTimeSeconds: (Date.now() - startTime) / 1000,
  };

  landscapeCache.set(cacheKey, result);
  if (landscapeCache.size > 500) {
    const oldest = landscapeCache.keys().next().value;
    if (oldest) landscapeCache.delete(oldest);
  }

  if (result.distortionModesExist) {
    console.log(`[DFT] ${formula}: Energy landscape exploration found ${uniqueMinima} unique minima (spread=${result.energySpread.toFixed(5)} Eh, ${minima.length} perturbations)`);
  }

  return result;
}

export function getLandscapeStats(): {
  totalExplored: number;
  multipleMinima: number;
  multiMinimaRate: number;
  avgUniqueMinima: number;
  avgEnergySpread: number;
  recent: Array<{ formula: string; uniqueMinima: number; multipleMinima: boolean; energySpread: number; distortionModes: boolean }>;
} {
  const entries = Array.from(landscapeCache.values());
  const n = entries.length;
  if (n === 0) {
    return { totalExplored: 0, multipleMinima: 0, multiMinimaRate: 0, avgUniqueMinima: 0, avgEnergySpread: 0, recent: [] };
  }
  const multiCount = entries.filter(e => e.multipleMinima).length;
  const avgMinima = entries.reduce((s, e) => s + e.uniqueMinima, 0) / n;
  const avgSpread = entries.reduce((s, e) => s + e.energySpread, 0) / n;
  const recent = entries.slice(-10).reverse().map(e => ({
    formula: e.formula,
    uniqueMinima: e.uniqueMinima,
    multipleMinima: e.multipleMinima,
    energySpread: e.energySpread,
    distortionModes: e.distortionModesExist,
  }));
  return {
    totalExplored: n,
    multipleMinima: multiCount,
    multiMinimaRate: Math.round((multiCount / n) * 1000) / 1000,
    avgUniqueMinima: Math.round(avgMinima * 100) / 100,
    avgEnergySpread: Math.round(avgSpread * 100000) / 100000,
    recent,
  };
}

export async function runDFTCalculation(formula: string, pressureGpa: number = 0): Promise<DFTResult> {
  const pressureTag = pressureGpa > 0 ? `_P${Math.round(pressureGpa)}` : "";
  const cacheKey = formula.replace(/\s+/g, "") + pressureTag;
  if (xtbResultCache.has(cacheKey)) {
    return xtbResultCache.get(cacheKey)!;
  }

  const startTime = Date.now();
  const calcId = `${cacheKey.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`;
  const calcDir = path.join(WORK_DIR, calcId);
  fs.mkdirSync(calcDir, { recursive: true });

  let atoms: AtomPosition[] = [];
  let prototype = "unknown";
  let isOptimized = false;

  const optResult = await runXTBOptimization(formula, pressureGpa);
  if (optResult && optResult.converged && optResult.optimizedAtoms.length >= 2) {
    atoms = optResult.optimizedAtoms;
    prototype = "xTB-optimized";
    isOptimized = true;
  }

  if (atoms.length < 2) {
    let generated = generateCrystalStructure(formula);
    atoms = generated.atoms;
    prototype = generated.prototype;
    if (pressureGpa > 0 && atoms.length >= 2) {
      atoms = applyPressureScaling(atoms, formula, pressureGpa);
    }
    if (atoms.length < 2) return null;
  }

  const xyzPath = path.join(calcDir, "input.xyz");

  const result: DFTResult = {
    formula,
    method: "GFN2-xTB",
    prototype,
    totalEnergy: 0,
    totalEnergyPerAtom: 0,
    homoLumoGap: 0,
    isMetallic: false,
    homo: null,
    lumo: null,
    fermiLevel: null,
    dipoleMoment: null,
    charges: {},
    converged: false,
    wallTimeSeconds: 0,
    atomCount: atoms.length,
    error: null,
    optimized: isOptimized,
  };

  if (atoms.length < 2) {
    result.error = "Too few atoms for DFT";
    return result;
  }

  writeXYZ(atoms, xyzPath, `${formula} [${prototype}]${isOptimized ? " (optimized)" : ""}`);

  try {
    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    const output = execSync(
      `cd ${calcDir} && ${XTB_BIN} ${xyzPath} --gfn 2 --sp 2>&1`,
      { timeout: TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
    ).toString();

    const parsed = parseXtbOutput(output);
    Object.assign(result, parsed);
    result.optimized = isOptimized;

    if (result.totalEnergy !== 0 && atoms.length > 0) {
      result.totalEnergyPerAtom = result.totalEnergy / atoms.length;
    }

    result.prototype = prototype;
  } catch (err: any) {
    result.error = err.message?.slice(0, 200) || "DFT calculation failed";
    if (err.stdout) {
      const stdoutStr = err.stdout.toString();
      const parsed = parseXtbOutput(stdoutStr);
      if (parsed.totalEnergy) {
        Object.assign(result, parsed);
        result.converged = false;
        result.optimized = isOptimized;
        result.error = "Partial: " + result.error?.slice(0, 100);
      }
    }
  }

  result.wallTimeSeconds = (Date.now() - startTime) / 1000;

  try {
    fs.rmSync(calcDir, { recursive: true, force: true });
  } catch {}

  if (result.converged) {
    xtbResultCache.set(cacheKey, result);
    if (xtbResultCache.size > CACHE_MAX) {
      const oldest = xtbResultCache.keys().next().value;
      if (oldest) xtbResultCache.delete(oldest);
    }
  }

  return result;
}

const elementRefEnergies = new Map<string, number | null>();

const MOLECULAR_ELEMENTS = new Set(["H", "N", "O", "F", "Cl"]);
const MOLECULAR_BOND_LENGTHS: Record<string, number> = {
  H: 0.74,
  N: 1.10,
  O: 1.21,
  F: 1.42,
  Cl: 1.99,
};

const COHESIVE_ENERGIES_EV: Record<string, number> = {
  H: 2.24, He: 0.0, Li: 1.63, Be: 3.32, B: 5.81, C: 7.37, N: 4.92, O: 2.60, F: 0.84,
  Ne: 0.02, Na: 1.11, Mg: 1.51, Al: 3.39, Si: 4.63, P: 3.43, S: 2.85, Cl: 1.40,
  Ar: 0.08, K: 0.93, Ca: 1.84, Sc: 3.90, Ti: 4.85, V: 5.31, Cr: 4.10, Mn: 2.92,
  Fe: 4.28, Co: 4.39, Ni: 4.44, Cu: 3.49, Zn: 1.35, Ga: 2.81, Ge: 3.85,
  As: 2.96, Se: 2.46, Br: 1.22, Kr: 0.12,
  Rb: 0.85, Sr: 1.72, Y: 4.37, Zr: 6.25, Nb: 7.57, Mo: 6.82, Tc: 6.85,
  Ru: 6.74, Rh: 5.75, Pd: 3.89, Ag: 2.95, Cd: 1.16, In: 2.52, Sn: 3.14, Sb: 2.75, Te: 2.02,
  I: 1.11, Xe: 0.16,
  Cs: 0.80, Ba: 1.90, La: 4.47, Ce: 4.32, Pr: 3.70, Nd: 3.40, Pm: 3.20, Sm: 2.14, Eu: 1.86,
  Gd: 4.14, Tb: 4.05, Dy: 3.04, Ho: 3.14, Er: 3.29, Tm: 2.42, Yb: 1.60, Lu: 4.43,
  Hf: 6.44, Ta: 8.10, W: 8.90, Re: 8.03, Os: 8.17, Ir: 6.94, Pt: 5.84, Au: 3.81,
  Hg: 0.67, Tl: 1.88, Pb: 2.03, Bi: 2.18, Po: 1.50, At: 1.00, Rn: 0.20,
  Fr: 0.75, Ra: 1.66, Ac: 4.25, Th: 6.20, Pa: 5.89, U: 5.55, Np: 4.73, Pu: 3.60,
  Am: 2.73, Cm: 3.99,
};

async function computeElementalEnergy(element: string): Promise<number | null> {
  if (elementRefEnergies.has(element)) {
    return elementRefEnergies.get(element) ?? null;
  }

  const calcDir = path.join(WORK_DIR, `ref_${element}_${Date.now()}`);
  fs.mkdirSync(calcDir, { recursive: true });

  const isMolecular = MOLECULAR_ELEMENTS.has(element);
  let atoms: AtomPosition[];
  let divisor: number;

  if (isMolecular) {
    const bondLength = MOLECULAR_BOND_LENGTHS[element] ?? 1.5;
    atoms = [
      { element, x: 0, y: 0, z: 0 },
      { element, x: bondLength, y: 0, z: 0 },
    ];
    divisor = 2;
  } else {
    atoms = [{ element, x: 0, y: 0, z: 0 }];
    divisor = 1;
  }

  const xyzPath = path.join(calcDir, `${element}.xyz`);
  writeXYZ(atoms, xyzPath, `${element} ${isMolecular ? "dimer" : "atom"} reference`);

  try {
    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    const UNPAIRED_ELECTRONS: Record<string, number> = {
      H: 1, He: 0, Li: 1, Be: 0, B: 1, C: 2, N: 3, O: 2, F: 1, Ne: 0,
      Na: 1, Mg: 0, Al: 1, Si: 2, P: 3, S: 2, Cl: 1, Ar: 0,
      K: 1, Ca: 0, Sc: 1, Ti: 2, V: 3, Cr: 6, Mn: 5, Fe: 4, Co: 3, Ni: 2, Cu: 1, Zn: 0,
      Ga: 1, Ge: 2, As: 3, Se: 2, Br: 1, Kr: 0,
      Rb: 1, Sr: 0, Y: 1, Zr: 2, Nb: 5, Mo: 6, Pd: 0, Sn: 2, Te: 2,
      Cs: 1, Ba: 0, La: 1, Ce: 2, W: 4, Pt: 2, Pb: 2, Bi: 3,
      Ta: 3, Hf: 2,
    };
    const uhf = isMolecular ? 0 : (UNPAIRED_ELECTRONS[element] ?? 0);
    const uhfFlag = uhf > 0 ? `--uhf ${uhf}` : "";
    const output = execSync(
      `cd ${calcDir} && ${XTB_BIN} ${xyzPath} --gfn 2 --sp ${uhfFlag} 2>&1`,
      { timeout: 30000, env, maxBuffer: 5 * 1024 * 1024 }
    ).toString();

    const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
    if (energyMatch && output.includes("normal termination")) {
      const energyPerAtom = parseFloat(energyMatch[1]) / divisor;
      console.log(`[DFT] ${element}: xTB ref energy = ${energyPerAtom.toFixed(4)} Ha/atom (${isMolecular ? "from dimer" : "isolated atom"}, no cohesive correction — consistent with xTB compound energies)`);
      elementRefEnergies.set(element, energyPerAtom);
      try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
      return energyPerAtom;
    }
  } catch {}

  try {
    fs.rmSync(calcDir, { recursive: true, force: true });
  } catch {}

  elementRefEnergies.set(element, null);
  return null;
}

export async function computeFormationEnergy(formula: string, dftResult: DFTResult): Promise<number | null> {
  if (!dftResult.converged || dftResult.totalEnergy === 0) return null;
  if (dftResult.totalEnergy > 0) {
    console.log(`[DFT] ${formula}: Positive total energy (${dftResult.totalEnergy.toFixed(4)} Ha) — xTB produced invalid result, skipping Ef`);
    return null;
  }

  const actualAtomCount = dftResult.atomCount;
  if (actualAtomCount === 0) return null;

  let compoundEnergy = dftResult.totalEnergy;
  const optCacheKey = formula.replace(/\s+/g, "");
  if (optimizedStructureCache.has(optCacheKey)) {
    const optResult = optimizedStructureCache.get(optCacheKey)!;
    if (optResult.converged && optResult.optimizedEnergy !== 0) {
      compoundEnergy = Math.min(compoundEnergy, optResult.optimizedEnergy);
    }
  }

  const counts = parseFormula(formula);
  const elements = Object.keys(counts);
  const originalTotal = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  if (originalTotal === 0 || elements.length === 0) return null;

  const scaleFactor = actualAtomCount / originalTotal;

  let elementalTotal = 0;
  for (const [el, count] of Object.entries(counts)) {
    const scaledCount = Math.max(1, Math.round(count * scaleFactor));
    const refE = await computeElementalEnergy(el);
    if (refE === null) return null;
    elementalTotal += refE * scaledCount;
  }

  const formationTotal = compoundEnergy - elementalTotal;
  const HA_TO_EV = 27.2114;
  let efPerAtom = (formationTotal / actualAtomCount) * HA_TO_EV;

  if (efPerAtom > 5.0 || efPerAtom < -10.0) {
    console.log(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom wildly out of range, likely reference energy mismatch — discarding`);
    return null;
  }

  if (efPerAtom > 1.0) {
    console.log(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom is positive (>1.0), discarding — compound less stable than elements`);
    return null;
  }

  if (efPerAtom < -5.0) {
    console.log(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom unrealistically negative (<-5.0), clamping to -5.0`);
    efPerAtom = -5.0;
  }

  return efPerAtom;
}

let totalXTBRuns = 0;
let totalXTBSuccesses = 0;
let prototypeAttempts = 0;
let prototypeSuccesses = 0;
let chemistryMatchAttempts = 0;
let chemistryMatchSuccesses = 0;

const phononCache = new Map<string, PhononStability | null>();

export async function runXTBPhononCheck(formula: string): Promise<PhononStability | null> {
  if (!isDFTAvailable()) return null;
  if (phononCache.has(formula)) return phononCache.get(formula)!;

  const calcDir = path.join(WORK_DIR, `phonon_${formula.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`);
  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: "1",
    OMP_STACKSIZE: "1G",
  };
  try {
    fs.mkdirSync(calcDir, { recursive: true });

    const { atoms: initialAtoms, prototype } = generateCrystalStructure(formula);
    if (initialAtoms.length < 2) return null;

    let atoms = initialAtoms;
    const optCacheKey = formula.replace(/\s+/g, "");
    if (optimizedStructureCache.has(optCacheKey)) {
      const optResult = optimizedStructureCache.get(optCacheKey)!;
      if (optResult.converged && optResult.optimizedAtoms.length >= 2) {
        atoms = optResult.optimizedAtoms;
      }
    } else {
      try {
        const preOptDir = path.join(calcDir, "pre_opt");
        fs.mkdirSync(preOptDir, { recursive: true });
        const preOptAtoms = validateAndFixStructure(initialAtoms, formula);
        if (preOptAtoms && preOptAtoms.length >= 2) {
          writeXYZ(preOptAtoms, path.join(preOptDir, "input.xyz"), `${formula} pre-opt`);
          const optOut = execSync(
            `${XTB_BIN} input.xyz --gfn 2 --opt crude --iterations 200 2>&1`,
            { cwd: preOptDir, timeout: TIMEOUT_MS, env, maxBuffer: 50 * 1024 * 1024 }
          ).toString();
          if (optOut.includes("converged")) {
            const optXYZ = path.join(preOptDir, "xtbopt.xyz");
            if (fs.existsSync(optXYZ)) {
              const parsed = parseOptimizedXYZ(optXYZ);
              if (parsed.length >= 2) {
                atoms = parsed;
                optimizedStructureCache.set(optCacheKey, {
                  converged: true,
                  optimizedAtoms: parsed,
                  optimizedEnergy: 0,
                  energyChange: 0,
                  gradientNorm: 0,
                  iterations: 0,
                  wallTimeSeconds: 0,
                });
              }
            }
          }
        }
      } catch {}
    }

    const prePhononCheck = validateAndFixStructure(atoms, formula);
    if (!prePhononCheck) {
      console.log(`[DFT] ${formula}: Phonon check skipped — structure has atom overlaps`);
      return null;
    }
    atoms = prePhononCheck;

    writeXYZ(atoms, path.join(calcDir, "input.xyz"), `${formula} phonon check (${prototype})`);

    const output = execSync(
      `${XTB_BIN} input.xyz --gfn 2 --hess --iterations 200 2>&1`,
      { cwd: calcDir, timeout: TIMEOUT_MS * 2, env, maxBuffer: 50 * 1024 * 1024 }
    ).toString();

    const frequencies: number[] = [];
    const freqSection = output.match(/projected vibrational frequencies[\s\S]*?(?=\n\n|\nreduced masses)/);
    if (freqSection) {
      const lines = freqSection[0].split("\n");
      for (const line of lines) {
        const nums = line.match(/-?\d+\.\d+/g);
        if (nums) {
          for (const n of nums) {
            const freq = parseFloat(n);
            if (Number.isFinite(freq) && Math.abs(freq) > 0.01) {
              frequencies.push(freq);
            }
          }
        }
      }
    }

    if (frequencies.length === 0) {
      const vibLines = output.match(/Frequency\s+.*cm/g);
      if (vibLines) {
        for (const line of vibLines) {
          const nums = line.match(/-?\d+\.\d+/g);
          if (nums) {
            for (const n of nums) {
              const freq = parseFloat(n);
              if (Number.isFinite(freq) && Math.abs(freq) > 0.01) {
                frequencies.push(freq);
              }
            }
          }
        }
      }
    }

    if (frequencies.length === 0) {
      const allFreqs = output.match(/eigval\s*:[\s\S]*?(?=\n\s*\n)/);
      if (allFreqs) {
        const nums = allFreqs[0].match(/-?\d+\.\d+/g);
        if (nums) {
          for (const n of nums) {
            const freq = parseFloat(n);
            if (Number.isFinite(freq) && Math.abs(freq) > 0.01) {
              frequencies.push(freq);
            }
          }
        }
      }
    }

    let zpe: number | null = null;
    const zpeMatch = output.match(/zero point energy\s+([-\d.]+)\s+Eh/);
    if (zpeMatch) zpe = parseFloat(zpeMatch[1]);

    const ARTIFACT_THRESHOLD = -5000;
    const PHYSICAL_IMAG_THRESHOLD = -5;
    const lowestFreq = frequencies.length > 0 ? Math.min(...frequencies) : 0;
    const physicalImagModes = frequencies.filter(f => f < PHYSICAL_IMAG_THRESHOLD && f >= ARTIFACT_THRESHOLD);
    const artifactModes = frequencies.filter(f => f < ARTIFACT_THRESHOLD);

    const result: PhononStability = {
      hasImaginaryModes: physicalImagModes.length > 0,
      imaginaryModeCount: physicalImagModes.length,
      lowestFrequency: lowestFreq,
      frequencies: frequencies.slice(0, 20),
      zeroPointEnergy: zpe,
    };

    if (physicalImagModes.length > 10) {
      (result as any).severeInstability = true;
      (result as any).instabilityReason = `${physicalImagModes.length} imaginary modes detected (max 10 allowed for xTB screening)`;
      console.log(`[DFT] ${formula}: Severe phonon instability — ${physicalImagModes.length} imaginary modes`);
    }
    if (lowestFreq < -1500 && lowestFreq >= ARTIFACT_THRESHOLD) {
      (result as any).severeInstability = true;
      (result as any).instabilityReason = `Lowest frequency ${lowestFreq.toFixed(0)} cm-1 (threshold: -1500 cm-1)`;
      console.log(`[DFT] ${formula}: Severe phonon instability — lowest freq = ${lowestFreq.toFixed(0)} cm-1`);
    }
    if (artifactModes.length > 0) {
      console.log(`[DFT] ${formula}: ${artifactModes.length} xTB numerical artifact modes (< ${ARTIFACT_THRESHOLD} cm-1) discarded`);
    }

    phononCache.set(formula, result);
    if (phononCache.size > 100) {
      const oldest = phononCache.keys().next().value;
      if (oldest) phononCache.delete(oldest);
    }

    return result;
  } catch (err) {
    console.log(`[DFT] ${formula}: xTB --hess failed, using analytical phonon fallback: ${err instanceof Error ? err.message.slice(0, 100) : String(err).slice(0, 100)}`);

    try {
      const optOutput = execSync(
        `${XTB_BIN} input.xyz --gfn 2 --opt tight --iterations 200 2>&1`,
        { cwd: calcDir, timeout: TIMEOUT_MS, env, maxBuffer: 20 * 1024 * 1024 }
      ).toString();

      const converged = optOutput.includes("GEOMETRY OPTIMIZATION CONVERGED") || optOutput.includes("normal termination");
      const counts = parseFormula(formula);
      const elements = Object.keys(counts);
      const avgMass = elements.reduce((s, el) => {
        const masses: Record<string, number> = {
          H: 1.008, He: 4.003, Li: 6.941, Be: 9.012, B: 10.81, C: 12.01, N: 14.01, O: 16.00, F: 19.00,
          Na: 22.99, Mg: 24.31, Al: 26.98, Si: 28.09, P: 30.97, S: 32.07, Cl: 35.45,
          K: 39.10, Ca: 40.08, Sc: 44.96, Ti: 47.87, V: 50.94, Cr: 52.00, Mn: 54.94,
          Fe: 55.85, Co: 58.93, Ni: 58.69, Cu: 63.55, Zn: 65.38, Ga: 69.72, Ge: 72.63,
          As: 74.92, Se: 78.97, Br: 79.90, Y: 88.91, Zr: 91.22, Nb: 92.91, Mo: 95.95,
          Pd: 106.42, Sn: 118.71, Te: 127.60, La: 138.91, Ce: 140.12, Hf: 178.49,
          Ta: 180.95, W: 183.84, Pt: 195.08, Pb: 207.2, Bi: 208.98, Ba: 137.33, Sr: 87.62,
        };
        return s + (masses[el] ?? 50) * (counts[el] ?? 1);
      }, 0) / Object.values(counts).reduce((s, n) => s + n, 0);

      const hasHydrogen = !!counts["H"];
      const hRatio = hasHydrogen ? (counts["H"] ?? 0) / Object.values(counts).reduce((s, n) => s + n, 0) : 0;
      const baseDebye = hasHydrogen
        ? (hRatio > 0.7 ? 1200 : hRatio > 0.5 ? 800 : 400)
        : (avgMass < 30 ? 500 : avgMass < 60 ? 350 : avgMass < 100 ? 250 : 180);
      const debyeFreq = baseDebye * (converged ? 1.0 : 0.8);
      const nAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
      const nModes = Math.max(1, 3 * nAtoms - 6);
      const estimatedFreqs: number[] = [];
      for (let i = 0; i < nModes; i++) {
        const fraction = (i + 1) / nModes;
        let hash = 0;
        for (let c = 0; c < formula.length; c++) hash = ((hash << 5) - hash + formula.charCodeAt(c)) | 0;
        const seed = ((hash * 2654435761 + i * 2246822507) >>> 0) / 4294967296;
        estimatedFreqs.push(debyeFreq * fraction * (0.8 + 0.4 * seed));
      }
      estimatedFreqs.sort((a, b) => a - b);

      const result: PhononStability = {
        hasImaginaryModes: !converged,
        imaginaryModeCount: converged ? 0 : 1,
        lowestFrequency: converged ? estimatedFreqs[0] : -25,
        frequencies: estimatedFreqs.slice(0, 20),
        zeroPointEnergy: null,
      };
      (result as any).analyticalEstimate = true;
      (result as any).estimationBasis = converged ? "opt-converged" : "opt-unconverged";

      console.log(`[DFT] ${formula}: Analytical phonon estimate — Debye≈${debyeFreq.toFixed(0)} cm⁻¹, ${nModes} modes, stable=${converged}`);
      phononCache.set(formula, result);
      return result;
    } catch (fallbackErr) {
      console.log(`[DFT] ${formula}: Phonon fallback also failed: ${fallbackErr instanceof Error ? fallbackErr.message.slice(0, 80) : ""}`);
      return null;
    }
  } finally {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
  }
}

export async function runFiniteDisplacementPhonons(formula: string): Promise<FiniteDisplacementPhononResult | null> {
  if (!isDFTAvailable()) return null;

  const cacheKey = formula.replace(/\s+/g, "");
  const optResult = optimizedStructureCache.get(cacheKey);
  let rawAtoms: AtomPosition[];
  if (optResult && optResult.converged && optResult.optimizedAtoms && optResult.optimizedAtoms.length >= 2) {
    rawAtoms = optResult.optimizedAtoms;
  } else {
    rawAtoms = generateCrystalStructure(formula).atoms;
  }
  if (rawAtoms.length < 2 || rawAtoms.length > 8) return null;

  const validatedAtoms = validateAndFixStructure(rawAtoms, formula);
  if (!validatedAtoms) {
    console.log(`[DFT] ${formula}: Finite displacement phonons skipped — structure has atom overlaps`);
    return null;
  }

  return computeFiniteDisplacementPhonons(formula, validatedAtoms);
}

export async function runXTBEnrichment(formula: string, pressureGpa: number = 0): Promise<XTBEnrichedFeatures | null> {
  if (!isDFTAvailable()) return null;

  totalXTBRuns++;
  const dftResult = await runDFTCalculation(formula, pressureGpa);

  if (!dftResult.converged || dftResult.totalEnergy === 0) {
    return null;
  }

  totalXTBSuccesses++;

  const formationE = await computeFormationEnergy(formula, dftResult);

  let phononResult: PhononStability | null = null;
  let fdPhononResult: FiniteDisplacementPhononResult | null = null;

  if (dftResult.atomCount <= 8) {
    fdPhononResult = await runFiniteDisplacementPhonons(formula);
    if (fdPhononResult) {
      phononResult = {
        hasImaginaryModes: fdPhononResult.hasImaginaryModes,
        imaginaryModeCount: fdPhononResult.imaginaryModeCount,
        lowestFrequency: fdPhononResult.lowestFrequency,
        frequencies: fdPhononResult.gammaFrequencies.slice(0, 20),
        zeroPointEnergy: null,
      };
    }
  }

  if (!phononResult && dftResult.atomCount <= 12) {
    phononResult = await runXTBPhononCheck(formula);
  }

  const pressureTag = pressureGpa > 0 ? `_P${Math.round(pressureGpa)}` : "";
  const enrichCacheKey = formula.replace(/\s+/g, "") + pressureTag;
  if (phononResult) {
    const optRes = optimizedStructureCache.get(enrichCacheKey);
    if (optRes?.distortion) {
      const updatedDistortion = analyzeDistortion(
        formula,
        { a: 1, b: 1, c: 1, alpha: 90, beta: 90, gamma: 90 },
        { a: 1, b: 1, c: 1, alpha: 90, beta: 90, gamma: 90 },
        undefined,
        optRes.optimizedAtoms,
        undefined,
        undefined,
        phononResult.frequencies,
        phononResult.hasImaginaryModes,
        phononResult.imaginaryModeCount,
        phononResult.lowestFrequency,
      );
      optRes.distortion.phononInstability = updatedDistortion.phononInstability;
      if (updatedDistortion.phononInstability?.hasImaginaryModes) {
        optRes.distortion.overallScore = Math.min(1.0, optRes.distortion.overallScore + 0.05);
        optRes.distortion.scRelevance += ` Phonon instability: ${phononResult.imaginaryModeCount} imaginary mode(s), lowest freq=${phononResult.lowestFrequency.toFixed(1)} cm-1 — structure wants to distort.`;
      }
    }
  }

  return {
    bandGap: dftResult.homoLumoGap,
    isMetallic: dftResult.isMetallic,
    totalEnergy: dftResult.totalEnergy,
    totalEnergyPerAtom: dftResult.totalEnergyPerAtom,
    formationEnergyPerAtom: formationE,
    fermiLevel: dftResult.fermiLevel,
    converged: true,
    prototype: dftResult.prototype,
    method: "GFN2-xTB",
    phononStability: phononResult,
    finiteDisplacementPhonons: fdPhononResult,
  };
}

export function isDFTAvailable(): boolean {
  try {
    if (xtbHealthy) return true;
    return fs.existsSync(XTB_BIN);
  } catch {
    return false;
  }
}

export function getDFTMethodInfo(): { name: string; version: string; level: string } {
  return {
    name: "GFN2-xTB",
    version: "6.7.1",
    level: "Semi-empirical DFT (Density Functional Tight Binding)",
  };
}

export function getXTBStats() {
  return {
    runs: totalXTBRuns,
    successes: totalXTBSuccesses,
    cacheSize: xtbResultCache.size,
    optimizedStructures: optimizedStructureCache.size,
    refElements: elementRefEnergies.size,
    prototypeAttempts,
    prototypeSuccesses,
    chemistryMatchAttempts,
    chemistryMatchSuccesses,
  };
}

let xtbHealthy = false;

export async function checkXTBHealth(): Promise<{ available: boolean; version: string; canOptimize: boolean; canHess: boolean; error?: string }> {
  const result = { available: false, version: "", canOptimize: false, canHess: false, error: undefined as string | undefined };

  if (!fs.existsSync(XTB_BIN)) {
    result.error = `xTB binary not found at ${XTB_BIN}`;
    console.log(`[xTB-Health] FAIL: ${result.error}`);
    return result;
  }

  try {
    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "100M",
      PATH: `${path.join(XTB_HOME, "bin")}:${process.env.PATH}`,
    };

    const versionOutput = execSync(`${XTB_BIN} --version 2>&1`, { timeout: 10000, env }).toString();
    const vMatch = versionOutput.match(/xtb version (\S+)/);
    result.version = vMatch ? vMatch[1] : "unknown";
    result.available = true;
    console.log(`[xTB-Health] Binary found: v${result.version}`);

    const testDir = path.join(WORK_DIR, `health_check_${Date.now()}`);
    fs.mkdirSync(testDir, { recursive: true });

    const h2Xyz = `2\nH2 molecule test\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74\n`;
    fs.writeFileSync(path.join(testDir, "input.xyz"), h2Xyz);

    try {
      const optOut = execSync(`cd ${testDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight 2>&1`, { timeout: 15000, env }).toString();
      if (optOut.includes("normal termination")) {
        result.canOptimize = true;
        console.log(`[xTB-Health] Geometry optimization: OK`);
      } else {
        console.log(`[xTB-Health] Geometry optimization: completed but no normal termination marker`);
        result.canOptimize = optOut.includes("TOTAL ENERGY");
      }
    } catch (e: any) {
      const msg = e.stdout?.toString() || e.message || "";
      if (msg.includes("TOTAL ENERGY")) {
        result.canOptimize = true;
        console.log(`[xTB-Health] Geometry optimization: OK (non-zero exit but energy computed)`);
      } else {
        console.log(`[xTB-Health] Geometry optimization: FAILED — ${msg.slice(0, 200)}`);
      }
    }

    try {
      const hessOut = execSync(`cd ${testDir} && ${XTB_BIN} input.xyz --gfn 2 --hess 2>&1`, { timeout: 20000, env }).toString();
      if (hessOut.includes("projected vibrational frequencies") || hessOut.includes("normal termination")) {
        result.canHess = true;
        console.log(`[xTB-Health] Hessian calculation: OK`);
      } else {
        console.log(`[xTB-Health] Hessian calculation: completed but missing expected output`);
      }
    } catch (e: any) {
      const msg = e.stdout?.toString() || e.message || "";
      if (msg.includes("vibrational frequencies")) {
        result.canHess = true;
        console.log(`[xTB-Health] Hessian calculation: OK (non-zero exit but frequencies computed)`);
      } else {
        console.log(`[xTB-Health] Hessian calculation: FAILED — ${msg.slice(0, 200)}`);
      }
    }

    try { fs.rmSync(testDir, { recursive: true, force: true }); } catch {}

  } catch (e: any) {
    result.error = e.message || "Unknown error";
    console.log(`[xTB-Health] FAIL: ${result.error}`);
  }

  xtbHealthy = result.available && result.canOptimize && result.canHess;
  if (!xtbHealthy) {
    console.warn(`[xTB-Health] WARNING: xTB is not fully functional. DFT calculations may fail or use fallbacks.`);
  }
  console.log(`[xTB-Health] Summary: available=${result.available}, opt=${result.canOptimize}, hess=${result.canHess}, healthy=${xtbHealthy}`);
  return result;
}

export function isXTBHealthy(): boolean {
  return xtbHealthy;
}

export interface AnharmonicProbeResult {
  displacements: number[];
  energies: number[];
  forces: number[];
  source: "xtb-displacement" | "physics-engine-estimate";
}

export interface MDSamplingRawResult {
  positions: number[][][];
  velocities: number[][][];
  temperature: number;
  totalSteps: number;
  timeStepFs: number;
}

export async function runXTBAnharmonicProbe(formula: string): Promise<AnharmonicProbeResult | null> {
  if (!isDFTAvailable()) return null;

  const { atoms } = generateCrystalStructure(formula);
  if (atoms.length < 2 || atoms.length > 20) return null;

  const validated = validateAndFixStructure(atoms, formula);
  if (!validated) return null;

  const calcDir = path.join(WORK_DIR, `anharm_${formula.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`);
  fs.mkdirSync(calcDir, { recursive: true });

  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: "1",
    OMP_STACKSIZE: "512M",
  };

  const displacementScales = [-0.10, -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.08, 0.10];
  const energies: number[] = [];
  const forces: number[] = [];

  try {
    const targetAtom = 0;
    const dispDir = 0;

    for (const scale of displacementScales) {
      const displaced = validated.map((a, idx) => {
        if (idx !== targetAtom) return { ...a };
        const coords = [a.x, a.y, a.z];
        coords[dispDir] += scale;
        return { element: a.element, x: coords[0], y: coords[1], z: coords[2] };
      });

      const stepDir = path.join(calcDir, `disp_${scale.toFixed(3).replace("-", "m")}`);
      fs.mkdirSync(stepDir, { recursive: true });
      writeXYZ(displaced, path.join(stepDir, "input.xyz"), `${formula} displacement=${scale}`);

      try {
        const output = execSync(
          `${XTB_BIN} input.xyz --gfn 2 --sp 2>&1`,
          { cwd: stepDir, timeout: OPT_TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
        ).toString();

        const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
        const gradMatch = output.match(/GRADIENT NORM\s+([-\d.]+)/);

        const energy = energyMatch ? parseFloat(energyMatch[1]) : 0;
        const grad = gradMatch ? parseFloat(gradMatch[1]) : 0;

        energies.push(energy);
        forces.push(-grad * (scale >= 0 ? 1 : -1));
      } catch {
        energies.push(0);
        forces.push(0);
      }
    }

    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}

    if (energies.every(e => e === 0)) return null;

    const refE = energies[5] || energies.find(e => e !== 0) || 0;
    const relativeEnergies = energies.map(e => (e - refE) * 27.211);

    return {
      displacements: displacementScales,
      energies: relativeEnergies,
      forces,
      source: "xtb-displacement",
    };
  } catch {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
    return null;
  }
}

export async function runXTBMDSampling(formula: string, temperatureK: number = 300): Promise<MDSamplingRawResult | null> {
  if (!isDFTAvailable()) return null;

  const { atoms } = generateCrystalStructure(formula);
  if (atoms.length < 2 || atoms.length > 15) return null;

  const validated = validateAndFixStructure(atoms, formula);
  if (!validated) return null;

  const calcDir = path.join(WORK_DIR, `md_${formula.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`);
  fs.mkdirSync(calcDir, { recursive: true });

  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: "1",
    OMP_STACKSIZE: "512M",
  };

  writeXYZ(validated, path.join(calcDir, "input.xyz"), `${formula} MD sampling T=${temperatureK}K`);

  const totalSteps = 200;
  const timeStepFs = 1.0;
  const dumpFreq = 10;

  const mdInput = [
    `$md`,
    `   temp=${temperatureK}`,
    `   time=${(totalSteps * timeStepFs * 0.001).toFixed(2)}`,
    `   dump=${(dumpFreq * timeStepFs * 0.001).toFixed(4)}`,
    `   step=${timeStepFs.toFixed(1)}`,
    `   hmass=1`,
    `   shake=0`,
    `   nvt=true`,
    `$end`,
  ].join("\n");

  fs.writeFileSync(path.join(calcDir, "md.inp"), mdInput);

  try {
    execSync(
      `${XTB_BIN} input.xyz --gfn 2 --md --input md.inp 2>&1`,
      { cwd: calcDir, timeout: 45000, env, maxBuffer: 20 * 1024 * 1024 }
    );

    const trajPath = path.join(calcDir, "xtb.trj");
    if (!fs.existsSync(trajPath)) {
      try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
      return null;
    }

    const trajContent = fs.readFileSync(trajPath, "utf-8");
    const frames = trajContent.split(/(?=\s*\d+\n)/g).filter(f => f.trim().length > 0);

    const positions: number[][][] = [];
    const velocities: number[][][] = [];

    for (const frame of frames) {
      const lines = frame.trim().split("\n");
      if (lines.length < 3) continue;
      const nAtoms = parseInt(lines[0].trim());
      if (isNaN(nAtoms) || nAtoms < 2) continue;

      const framePos: number[][] = [];
      for (let i = 2; i < Math.min(lines.length, 2 + nAtoms); i++) {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length >= 4) {
          framePos.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
        }
      }
      if (framePos.length === nAtoms) {
        positions.push(framePos);
      }
    }

    if (positions.length >= 2) {
      for (let f = 1; f < positions.length; f++) {
        const frameVel: number[][] = [];
        const dt = dumpFreq * timeStepFs;
        for (let a = 0; a < positions[f].length; a++) {
          frameVel.push([
            (positions[f][a][0] - positions[f - 1][a][0]) / dt,
            (positions[f][a][1] - positions[f - 1][a][1]) / dt,
            (positions[f][a][2] - positions[f - 1][a][2]) / dt,
          ]);
        }
        velocities.push(frameVel);
      }
    }

    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}

    if (positions.length < 2) return null;

    return {
      positions,
      velocities,
      temperature: temperatureK,
      totalSteps,
      timeStepFs,
    };
  } catch {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
    return null;
  }
}
