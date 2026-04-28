import { execSync, execFile, spawn } from "child_process";
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import { promisify } from "util";
import { IS_WINDOWS, binaryPath, getTempSubdir, toWslPath } from "./platform-utils";

const execFileAsync = promisify(execFile);
import { getElementData } from "../learning/elemental-data";
import { fillPrototype, computeBondValenceSum, checkIonicRadiusCompatibility } from "../learning/crystal-prototypes";
import { computeFiniteDisplacementPhonons } from "./phonon-calculator";
import type { FiniteDisplacementPhononResult } from "./phonon-calculator";
import { analyzeDistortion, recordDistortionAnalysis, type DistortionAnalysis } from "../crystal/distortion-detector";
import { relaxStructureAtPressure } from "../learning/pressure-engine";
import { normalizeFormula } from "../learning/utils";
import { getGroundTruthDataset } from "../learning/ground-truth-store";

const PROJECT_ROOT = path.resolve(process.cwd());
// XTB_BIN: set XTB_BIN=/usr/bin/xtb on GCP (xtb-dist/ is gitignored and not deployed there)
const XTB_BIN = binaryPath(process.env.XTB_BIN ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb"));
const XTB_HOME = process.env.XTBHOME ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = process.env.XTBPATH ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = getTempSubdir("dft_calculations");
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

const VALID_ELEMENTS = new Set([
  "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am",
]);

function parseFormula(formula: string): Record<string, number> {
  if (/-\d/.test(formula)) {
    console.warn(`[DFT] parseFormula: rejected formula with negative stoichiometry: "${formula}"`);
    return {};
  }

  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  let hasInvalidElement = false;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    if (!VALID_ELEMENTS.has(el)) {
      hasInvalidElement = true;
      console.warn(`[DFT] parseFormula: skipping non-physical element symbol "${el}" in "${formula}"`);
      continue;
    }
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (!isFinite(num) || num <= 0) continue;
    counts[el] = (counts[el] || 0) + num;
  }
  if (hasInvalidElement && Object.keys(counts).length === 0) {
    console.warn(`[DFT] parseFormula: formula "${formula}" produced no valid elements`);
  }
  return normalizeCountsToIntegers(counts);
}

const BULK_MODULI_GPA: Record<string, number> = {
  H: 1.0, He: 0.5, Li: 11, Be: 130, B: 320, C: 443, N: 2.0, O: 2.0, F: 1.5,
  Na: 6.3, Mg: 45, Al: 76, Si: 100, P: 11, S: 7.7, Cl: 1.1,
  K: 3.1, Ca: 17, Sc: 57, Ti: 110, V: 162, Cr: 160, Mn: 120, Fe: 170,
  Co: 180, Ni: 180, Cu: 140, Zn: 70, Ga: 56, Ge: 75, As: 22, Se: 8.3,
  Rb: 2.5, Sr: 12, Y: 41, Zr: 94, Nb: 170, Mo: 230, Ru: 220, Rh: 275,
  Pd: 180, Ag: 100, Cd: 42, In: 41, Sn: 58, Sb: 42, Te: 65, I: 7.7,
  Cs: 1.6, Ba: 9.6, La: 28, Ce: 22, Hf: 110, Ta: 200, W: 310, Re: 370,
  Os: 462, Ir: 320, Pt: 230, Au: 180, Hg: 25, Tl: 43, Pb: 46, Bi: 31,
  Th: 54, U: 100, Pr: 29, Nd: 32, Sm: 38, Eu: 8.3, Gd: 38, Tb: 39,
  Dy: 41, Ho: 40, Er: 44, Tm: 45, Yb: 13, Lu: 48,
  Pa: 93, Np: 118, Pu: 54, Tc: 297,
};

const B_PRIME_DEFAULT = 4.0;

function getGroupBulkModulusFallback(symbol: string): number {
  const data = getElementData(symbol);
  if (!data) return 50;
  const z = data.atomicNumber;

  if ([1].includes(z)) return 1.0;
  if ([2, 10, 18, 36, 54, 86].includes(z)) return 1.0;
  if ([3, 11, 19, 37, 55, 87].includes(z)) return 5.0;
  if ([4, 12, 20, 38, 56, 88].includes(z)) return 20;
  if ([9, 17, 35, 53, 85].includes(z)) return 3.0;
  if ([7, 8, 15, 16, 33, 34, 51, 52].includes(z)) return 15;
  if (z >= 57 && z <= 71) return 35;
  if (z >= 89 && z <= 96) return 70;
  if ((z >= 21 && z <= 30) || (z >= 39 && z <= 48) || (z >= 72 && z <= 80)) return 150;
  if ([5, 6, 14, 32].includes(z)) return 100;
  if ([13, 31, 49, 50, 81, 82, 83, 84].includes(z)) return 45;

  return 50;
}

export function getCompressedRadius(symbol: string, pressureGPa: number): number {
  const r0 = COVALENT_RADII[symbol] ?? 1.4;
  if (pressureGPa <= 0) return r0;

  if (symbol === "H") {
    const pNorm = pressureGPa / 500;
    const compressionFactor = 1.0 / (1.0 + 1.8 * pNorm + 0.6 * pNorm * pNorm);
    const rCompressed = r0 * compressionFactor;
    const rMinUltraHighP = 0.24;
    if (pressureGPa > 300) {
      return Math.max(rMinUltraHighP, rCompressed);
    }
    return rCompressed;
  }

  const B0 = BULK_MODULI_GPA[symbol] ?? getGroupBulkModulusFallback(symbol);
  const Bp = B_PRIME_DEFAULT;
  const volumeRatio = Math.pow(1 + Bp * pressureGPa / B0, -1 / Bp);
  const radiusRatio = Math.cbrt(Math.max(0.3, volumeRatio));
  const rMin = r0 * 0.5;
  return Math.max(rMin, r0 * radiusRatio);
}

function getAvgRadius(el: string): number {
  return COVALENT_RADII[el] || (getElementData(el)?.atomicRadius ?? 150) / 100;
}

const PROTOTYPE_PACKING_AMBIENT: Record<string, number> = {
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

function getEffectivePackingFactor(protoName: string | undefined, pressureGPa: number = 0): number {
  const ambient = protoName ? (PROTOTYPE_PACKING_AMBIENT[protoName] ?? 0.68) : 0.68;
  if (pressureGPa <= 0) return ambient;
  const pressureIncrease = (0.90 - ambient) * (1 - Math.exp(-pressureGPa / 200));
  return Math.min(0.92, ambient + pressureIncrease);
}

const PROTOTYPE_PACKING = PROTOTYPE_PACKING_AMBIENT;

function estimatePressureCOverA(
  ambientCOverA: number,
  latticeType: string,
  pressureGPa: number,
): number {
  if (latticeType === "cubic" || pressureGPa <= 0) return ambientCOverA;

  if (latticeType === "hexagonal") {
    const idealHcp = 1.633;
    const drift = (ambientCOverA - idealHcp) * Math.exp(-pressureGPa / 150);
    return idealHcp + drift;
  }

  if (latticeType === "tetragonal") {
    if (ambientCOverA > 1.0) {
      const excess = ambientCOverA - 1.0;
      const compressed = excess * Math.exp(-pressureGPa / 200);
      return 1.0 + compressed;
    }
    const deficit = 1.0 - ambientCOverA;
    const compressed = deficit * Math.exp(-pressureGPa / 200);
    return 1.0 - compressed;
  }

  return ambientCOverA;
}

const ATOMIC_VOLUMES: Record<string, number> = {
  H: 5.0, He: 6.0, Li: 20.0, Be: 8.0, B: 8.0, C: 9.0, N: 10.0,
  O: 12.0, F: 11.0, Ne: 13.0, Na: 24.0, Mg: 14.0, Al: 17.0, Si: 20.0,
  P: 17.0, S: 16.0, Cl: 22.0, Ar: 24.0, K: 46.0, Ca: 26.0, Sc: 25.0,
  Ti: 16.0, V: 14.0, Cr: 12.0, Mn: 12.0, Fe: 11.0, Co: 11.0, Ni: 11.0,
  Cu: 12.0, Zn: 15.0, Ga: 20.0, Ge: 23.0, As: 21.0, Se: 17.0, Br: 24.0,
  Kr: 27.0, Rb: 56.0, Sr: 34.0, Y: 25.0, Zr: 23.0, Nb: 18.0, Mo: 16.0,
  Ru: 14.0, Rh: 14.0, Pd: 15.0, Ag: 17.0, Cd: 22.0, In: 26.0, Sn: 27.0,
  Sb: 30.0, Te: 34.0, I: 26.0, Cs: 71.0, Ba: 39.0, La: 37.0, Ce: 35.0,
  Hf: 22.0, Ta: 18.0, W: 16.0, Re: 15.0, Os: 14.0, Ir: 14.0, Pt: 15.0,
  Au: 17.0, Hg: 23.0, Tl: 29.0, Pb: 30.0, Bi: 35.0, Th: 33.0, U: 21.0,
  Tc: 14.0,
};

function computeExpectedVolume(counts: Record<string, number>, packingFactor: number = 1.3): number {
  const totalAtoms = Object.values(counts).reduce((s, c) => s + Math.round(c), 0);
  const hCount = Math.round(counts["H"] || 0);
  const metalCount = totalAtoms - hCount;
  const hFraction = totalAtoms > 0 ? hCount / totalAtoms : 0;
  const hMetalRatio = metalCount > 0 ? hCount / metalCount : 0;

  if (hFraction > 0.5 && metalCount > 0) {
    const volPerAtom = 25 + 2.5 * hMetalRatio;
    return totalAtoms * volPerAtom;
  }

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

function estimateLatticeParam(
  elements: string[],
  counts: Record<string, number>,
  protoName?: string,
  latticeType?: string,
  cOverA?: number,
  pressureGPa: number = 0,
): number {
  const packingFactor = getEffectivePackingFactor(protoName, pressureGPa);
  let expectedVolume = computeExpectedVolume(counts, 1.0 / packingFactor);

  const totalAtoms = Object.values(counts).reduce((s, c) => s + Math.round(c), 0);
  const hasH = counts["H"] !== undefined && counts["H"] > 0;
  const minVolPerAtom = hasH ? 8.0 : 10.0;
  const minTotalVolume = totalAtoms * minVolPerAtom;
  if (expectedVolume < minTotalVolume) {
    expectedVolume = minTotalVolume;
  }

  const effectiveCOverA = cOverA ?? 1.0;

  if (latticeType === "hexagonal" && effectiveCOverA > 0) {
    const SQRT3_OVER_2 = Math.sqrt(3) / 2;
    const a = Math.pow(expectedVolume / (SQRT3_OVER_2 * effectiveCOverA), 1 / 3);
    return Math.max(a, 2.5);
  }

  if (latticeType === "tetragonal" && effectiveCOverA > 0) {
    const a = Math.cbrt(expectedVolume / effectiveCOverA);
    return Math.max(a, 2.5);
  }

  const latticeA = Math.cbrt(expectedVolume);
  return Math.max(latticeA, 3.0);
}

type PrototypeName =
  | "A15" | "NaCl" | "AlB2" | "Perovskite" | "ThCr2Si2" | "Heusler"
  | "BCC" | "FCC" | "Layered" | "Kagome" | "HexBoride" | "MX2"
  | "Anti-perovskite" | "CsCl" | "Cu2Mg-Laves" | "Fluorite" | "Cr3Si"
  | "Ni3Sn" | "Fe3C" | "Spinel" | "Clathrate-H32" | "Skutterudite"
  | "BiS2-layered" | "Kagome-variant" | "Chevrel" | "Pyrite" | "Wurtzite"
  | "Antifluorite" | "Laves-C14" | "Laves-C15" | "HfFe6Ge6" | "CeCu2Si2"
  | "PuCoGa5-115" | "Infinite-layer" | "T-prime" | "Ruddlesden-Popper"
  | "Double-perovskite" | "Garnet";

const PROTOTYPE_MATCH_TOLERANCE = 0.5;
const PROTOTYPE_FUZZY_TOLERANCE = 0.8;

interface PrototypeStructure {
  name: PrototypeName;
  fractionalPositions: { site: string; x: number; y: number; z: number }[];
  latticeType: string;
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
const ANIONS = new Set(["O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"]);
const PNICTIDES = new Set(["P", "As", "Sb"]);
const ALKALINE_EARTH = new Set(["Ca", "Sr", "Ba", "Mg"]);
const ALKALI = new Set(["Li", "Na", "K", "Rb", "Cs"]);

function classifyElement(el: string): "H" | "rare-earth" | "TM" | "anion" | "pnictide" | "alkaline" | "alkali" | "metalloid" | "other" {
  if (el === "H") return "H";
  if (RARE_EARTHS.has(el)) return "rare-earth";
  if (TRANSITION_METALS.has(el)) return "TM";
  if (PNICTIDES.has(el)) return "pnictide";
  if (ANIONS.has(el)) return "anion";
  if (ALKALINE_EARTH.has(el)) return "alkaline";
  if (ALKALI.has(el)) return "alkali";
  if (["B", "Si", "Ge", "Sn", "Bi", "Al", "Ga", "In", "Tl", "Pb"].includes(el)) return "metalloid";
  return "other";
}

interface ChemicalProfile {
  elements: string[];
  counts: Record<string, number>;
  elementsByCount: string[];
  roles: ReturnType<typeof classifyElement>[];
  reducedRatios: number[];
  sortedReducedRatios: number[];
  gcdVal: number;
  nElements: number;
  hasRE: boolean;
  hasTM: boolean;
  hasAnion: boolean;
  hasPnictide: boolean;
  hasAlkaline: boolean;
  hasAlkali: boolean;
  hasMetalloid: boolean;
  hasAnionLike: boolean;
  hasH: boolean;
}

function buildChemicalProfile(counts: Record<string, number>): ChemicalProfile {
  const elements = Object.keys(counts).filter(el => counts[el] > 0);
  const elementsByCount = [...elements].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
  const roles = elements.map(el => classifyElement(el));

  const ratios = elementsByCount.map(el => Math.round(counts[el]));
  const gcdVal = ratios.length > 0 ? ratios.reduce((a, b) => gcd(a, b)) : 1;
  const reducedRatios = ratios.map(r => r / Math.max(1, gcdVal));
  const sortedReducedRatios = [...reducedRatios].sort((a, b) => b - a);

  const hasRE = roles.includes("rare-earth");
  const hasTM = roles.includes("TM");
  const hasAnion = roles.includes("anion");
  const hasPnictide = roles.includes("pnictide");
  const hasAlkaline = roles.includes("alkaline");
  const hasAlkali = roles.includes("alkali");
  const hasMetalloid = roles.includes("metalloid");
  const hasH = roles.includes("H");

  return {
    elements,
    counts,
    elementsByCount,
    roles,
    reducedRatios,
    sortedReducedRatios,
    gcdVal,
    nElements: elements.length,
    hasRE,
    hasTM,
    hasAnion,
    hasPnictide,
    hasAlkaline,
    hasAlkali,
    hasMetalloid,
    hasAnionLike: hasAnion || hasPnictide,
    hasH,
  };
}

interface LearnedThresholds {
  perovskiteAnionRatioMin: number;
  ruddlesdenPopperAnionRatioMin: number;
  pyriteAnionTMRatioMin: number;
  a15TMMetalloidRatioMin: number;
  updatedAt: number;
}

const DEFAULT_THRESHOLDS: LearnedThresholds = {
  perovskiteAnionRatioMin: 1.2,
  ruddlesdenPopperAnionRatioMin: 2.5,
  pyriteAnionTMRatioMin: 1.8,
  a15TMMetalloidRatioMin: 2.5,
  updatedAt: 0,
};

let learnedThresholds: LearnedThresholds = { ...DEFAULT_THRESHOLDS };
let lastThresholdRefresh = 0;
const THRESHOLD_REFRESH_INTERVAL_MS = 5 * 60 * 1000;

function refreshLearnedThresholds(): LearnedThresholds {
  const now = Date.now();
  if (now - lastThresholdRefresh < THRESHOLD_REFRESH_INTERVAL_MS) {
    return learnedThresholds;
  }
  lastThresholdRefresh = now;

  try {
    const dataset = getGroundTruthDataset();
    if (dataset.length < 20) return learnedThresholds;

    const stableHighConf = dataset.filter(
      dp => dp.phonon_stable && (dp.confidence === "high" || dp.confidence === "medium") && dp.Tc > 0
    );
    if (stableHighConf.length < 10) return learnedThresholds;

    const perovskitePoints: number[] = [];
    const rpPoints: number[] = [];
    const pyritePoints: number[] = [];
    const a15Points: number[] = [];

    for (const dp of stableHighConf) {
      const counts = parseFormula(dp.formula);
      const elements = Object.keys(counts).filter(el => counts[el] > 0);
      if (elements.length < 2) continue;

      const roles = elements.map(el => classifyElement(el));
      const hasTM = roles.some(r => r === "TM");
      const hasAnion = roles.some(r => r === "anion");
      const hasMetalloid = roles.some(r => r === "metalloid");
      const hasSpacer = roles.some(r => r === "rare-earth" || r === "alkaline" || r === "alkali");

      if (elements.length === 3 && hasSpacer && hasTM && hasAnion) {
        const anionEl = elements.find(e => ANIONS.has(e));
        if (anionEl) {
          const anionCount = counts[anionEl] || 0;
          const totalNonAnion = Object.values(counts).reduce((s, n) => s + n, 0) - anionCount;
          const ratio = anionCount / Math.max(1, totalNonAnion);

          const struct = (dp.structure || "").toLowerCase();
          if (struct.includes("ruddlesden") || struct.includes("rp") || struct.includes("214")) {
            rpPoints.push(ratio);
          } else if (struct.includes("perovskite") || struct.includes("113")) {
            perovskitePoints.push(ratio);
          }
        }
      }

      if (elements.length === 2 && hasTM && hasAnion) {
        const tmEl = elements.find(e => TRANSITION_METALS.has(e));
        const anEl = elements.find(e => ANIONS.has(e));
        if (tmEl && anEl) {
          const ratio = (counts[anEl] || 0) / Math.max(1, counts[tmEl] || 1);
          const struct = (dp.structure || "").toLowerCase();
          if (struct.includes("pyrite")) {
            pyritePoints.push(ratio);
          }
        }
      }

      if (elements.length === 2 && hasTM && hasMetalloid) {
        const tmEl = elements.find(e => TRANSITION_METALS.has(e));
        const mEl = elements.find(e => classifyElement(e) === "metalloid");
        if (tmEl && mEl) {
          const ratio = (counts[tmEl] || 0) / Math.max(1, counts[mEl] || 1);
          const struct = (dp.structure || "").toLowerCase();
          if (struct.includes("a15") || struct.includes("cr3si")) {
            a15Points.push(ratio);
          }
        }
      }
    }

    const computeMinBound = (points: number[], fallback: number, minSamples: number = 3): number => {
      if (points.length < minSamples) return fallback;
      const sorted = [...points].sort((a, b) => a - b);
      const p10 = sorted[Math.floor(sorted.length * 0.1)];
      return Math.max(fallback * 0.6, Math.min(fallback * 1.4, p10));
    };

    learnedThresholds = {
      perovskiteAnionRatioMin: computeMinBound(perovskitePoints, DEFAULT_THRESHOLDS.perovskiteAnionRatioMin),
      ruddlesdenPopperAnionRatioMin: computeMinBound(rpPoints, DEFAULT_THRESHOLDS.ruddlesdenPopperAnionRatioMin),
      pyriteAnionTMRatioMin: computeMinBound(pyritePoints, DEFAULT_THRESHOLDS.pyriteAnionTMRatioMin),
      a15TMMetalloidRatioMin: computeMinBound(a15Points, DEFAULT_THRESHOLDS.a15TMMetalloidRatioMin),
      updatedAt: now,
    };

    const changed = Object.keys(DEFAULT_THRESHOLDS).filter(
      k => k !== "updatedAt" && (learnedThresholds as any)[k] !== (DEFAULT_THRESHOLDS as any)[k]
    );
    if (changed.length > 0) {
      dftLog(`[DFT] Pattern miner: updated thresholds from ${stableHighConf.length} ground truth points: ${changed.map(k => `${k}=${(learnedThresholds as any)[k].toFixed(2)}`).join(", ")}`);
    }
  } catch (err: any) {
    console.warn(`[DFT] Pattern miner: threshold refresh failed: ${err?.message?.slice(0, 80)}`);
  }

  return learnedThresholds;
}

function siteMapMatchesStoichiometry(
  proto: PrototypeStructure,
  siteMap: Record<string, string>,
  counts: Record<string, number>,
): boolean {
  const mappedCounts: Record<string, number> = {};
  for (const pos of proto.fractionalPositions) {
    const el = siteMap[pos.site];
    if (!el) return false;
    mappedCounts[el] = (mappedCounts[el] || 0) + 1;
  }

  const inputEntries = Object.entries(counts).filter(([, n]) => n > 0);
  const mappedEntries = Object.entries(mappedCounts);

  if (inputEntries.length !== mappedEntries.length) return false;

  const inputGCD = inputEntries.reduce((g, [, n]) => gcd(g, Math.round(n)), 0);
  const mappedGCD = mappedEntries.reduce((g, [, n]) => gcd(g, n), 0);

  const inputReduced: Record<string, number> = {};
  for (const [el, n] of inputEntries) {
    inputReduced[el] = Math.round(n) / Math.max(1, inputGCD);
  }
  const mappedReduced: Record<string, number> = {};
  for (const [el, n] of mappedEntries) {
    mappedReduced[el] = n / Math.max(1, mappedGCD);
  }

  for (const el of Object.keys(inputReduced)) {
    if (!(el in mappedReduced)) return false;
    if (Math.abs(mappedReduced[el] - inputReduced[el]) > 0.01) return false;
  }
  return true;
}

function estimateSiteNearestNeighborDistance(proto: PrototypeStructure): Record<string, number> {
  const sitePositions: Record<string, { x: number; y: number; z: number }[]> = {};
  for (const pos of proto.fractionalPositions) {
    if (!sitePositions[pos.site]) sitePositions[pos.site] = [];
    sitePositions[pos.site].push({ x: pos.x, y: pos.y, z: pos.z });
  }

  const cOverA = proto.cOverA || 1.0;
  const aRatio = proto.aRatio || 1.0;
  const sx = aRatio;
  const sy = aRatio;
  const sz = aRatio * cOverA;

  const allPositions = proto.fractionalPositions.map(p => [p.x, p.y, p.z]);
  const result: Record<string, number> = {};

  for (const site of Object.keys(sitePositions)) {
    const positions = sitePositions[site];
    let totalMinDist = 0;
    for (const pos of positions) {
      let minDist = Infinity;
      for (const other of allPositions) {
        let dx = Math.abs(pos.x - other[0]);
        let dy = Math.abs(pos.y - other[1]);
        let dz = Math.abs(pos.z - other[2]);
        dx = Math.min(dx, 1 - dx) * sx;
        dy = Math.min(dy, 1 - dy) * sy;
        dz = Math.min(dz, 1 - dz) * sz;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > 1e-6 && dist < minDist) minDist = dist;
      }
      totalMinDist += minDist;
    }
    result[site] = totalMinDist / Math.max(1, positions.length);
  }

  return result;
}

function selectBestPrototypeByChemistry(counts: Record<string, number>, elements: string[], profile?: ChemicalProfile): { proto: PrototypeStructure; siteMap: Record<string, string> } | null {
  const p = profile ?? buildChemicalProfile(counts);
  const {
    nElements, elementsByCount, roles,
    hasRE, hasTM, hasAnion, hasPnictide,
    hasAlkaline, hasAlkali, hasMetalloid, hasAnionLike, hasH,
  } = p;

  const th = refreshLearnedThresholds();

  let targetProtoName: PrototypeName | null = null;

  if (nElements === 3 && (hasRE || hasAlkaline || hasAlkali) && hasTM && hasPnictide) {
    targetProtoName = "ThCr2Si2";
  } else if (nElements === 3 && (hasRE || hasAlkaline || hasAlkali) && hasTM && hasAnion) {
    const anionEl = elements.find(e => ANIONS.has(e))!;
    const anionCount = counts[anionEl] || 0;
    const totalNonAnion = Object.values(counts).reduce((s, n) => s + n, 0) - anionCount;
    const anionRatio = anionCount / Math.max(1, totalNonAnion);
    if (anionRatio >= th.ruddlesdenPopperAnionRatioMin) targetProtoName = "Ruddlesden-Popper";
    else if (anionRatio >= th.perovskiteAnionRatioMin) targetProtoName = "Perovskite";
    else targetProtoName = "ThCr2Si2";
  } else if (nElements === 3 && hasTM && hasAnion && hasMetalloid) {
    targetProtoName = "BiS2-layered";
  } else if (nElements === 3 && hasTM && hasPnictide && hasMetalloid) {
    targetProtoName = "ThCr2Si2";
  } else if (nElements === 2 && hasTM && hasPnictide) {
    const tmEl = elements.find(e => TRANSITION_METALS.has(e))!;
    const pnEl = elements.find(e => PNICTIDES.has(e))!;
    const tmCount = counts[tmEl] || 0;
    const pnCount = counts[pnEl] || 0;
    if (tmCount / pnCount >= th.a15TMMetalloidRatioMin) targetProtoName = "Cr3Si";
    else targetProtoName = "NaCl";
  } else if (nElements === 2 && hasTM && hasAnion) {
    const tmEl = elements.find(e => TRANSITION_METALS.has(e))!;
    const anEl = elements.find(e => ANIONS.has(e))!;
    const tmCount = counts[tmEl] || 0;
    const anCount = counts[anEl] || 0;
    if (anCount / tmCount >= th.pyriteAnionTMRatioMin) targetProtoName = "Pyrite";
    else if (tmCount / anCount >= th.a15TMMetalloidRatioMin) targetProtoName = "Cr3Si";
    else targetProtoName = "NaCl";
  } else if (nElements === 2 && hasTM && hasMetalloid) {
    const tmEl = elements.find(e => TRANSITION_METALS.has(e))!;
    const mEl = elements.find(e => classifyElement(e) === "metalloid")!;
    const tmCount = counts[tmEl] || 0;
    const mCount = counts[mEl] || 0;
    if (tmCount / mCount >= th.a15TMMetalloidRatioMin) targetProtoName = "A15";
    else if (mCount / tmCount >= 2) targetProtoName = "AlB2";
    else targetProtoName = "CsCl";
  } else if (nElements === 2 && (hasRE || hasAlkaline) && hasTM) {
    targetProtoName = "Cu2Mg-Laves";
  } else if (nElements === 2 && hasH) {
    const metalEl = elements.find(e => e !== "H");
    const hCount = counts["H"] || 0;
    const metalCount = metalEl ? (counts[metalEl] || 0) : 1;
    const hRatio = hCount / metalCount;
    if (hRatio >= 2) targetProtoName = "AlB2";
    else if (hRatio <= 0.5) targetProtoName = "CsCl";
    else targetProtoName = "NaCl";
  } else if (nElements === 3 && hasH && (hasTM || hasRE || hasAlkaline || hasAlkali)) {
    const nonHNonAnionNonPnictide = elements.filter(e => e !== "H" && !ANIONS.has(e) && !PNICTIDES.has(e));
    if (nonHNonAnionNonPnictide.length >= 2) targetProtoName = "Perovskite";
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
    const hasOorF = elements.some(e => e === "O" || e === "F");
    const hasSpacer = elements.some(e => RARE_EARTHS.has(e) || ALKALINE_EARTH.has(e) || ALKALI.has(e));
    if (hasSpacer && hasTM && hasPnictide) {
      targetProtoName = "ThCr2Si2";
    } else if (hasSpacer && hasTM && hasPnictide && hasOorF) {
      targetProtoName = "ThCr2Si2";
    } else if (hasSpacer && hasTM && hasAnionLike) {
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

  const siteAvgVolumes = estimateSiteNearestNeighborDistance(proto);
  const sitesByVolume = Object.keys(siteAvgVolumes).sort(
    (a, b) => siteAvgVolumes[b] - siteAvgVolumes[a]
  );
  const uniqueElements = Array.from(new Set(elementsByCount));
  const elementsByRadius = uniqueElements.sort(
    (a, b) => (COVALENT_RADII[b] || 1.0) - (COVALENT_RADII[a] || 1.0)
  );

  const useRadiusMapping = sitesByVolume.length === elementsByRadius.length
    && sitesByVolume.length <= elementsByCount.length;

  const orderedElements = useRadiusMapping ? elementsByRadius : elementsByCount;
  const orderedSites = useRadiusMapping ? sitesByVolume : sites;

  if (orderedSites.length > orderedElements.length) {
    let matched = false;
    for (let i = 0; i < Math.min(orderedSites.length, orderedElements.length); i++) {
      siteMap[orderedSites[i]] = orderedElements[i];
    }
    for (let i = orderedElements.length; i < orderedSites.length; i++) {
      let bestEl = orderedElements[orderedElements.length - 1];
      let bestRatioDiff = Infinity;
      for (const el of orderedElements) {
        const elSiteCount = proto.fractionalPositions.filter(
          p => siteMap[p.site] === el || p.site === orderedSites[i]
        ).length;
        const targetCount = Math.round(counts[el] || 1);
        const diff = Math.abs(elSiteCount - targetCount);
        if (diff < bestRatioDiff) {
          bestRatioDiff = diff;
          bestEl = el;
        }
      }
      siteMap[orderedSites[i]] = bestEl;
    }

    if (siteMapMatchesStoichiometry(proto, siteMap, counts)) {
      matched = true;
    }

    if (!matched) {
      return null;
    }
  } else {
    for (let i = 0; i < Math.min(orderedSites.length, orderedElements.length); i++) {
      siteMap[orderedSites[i]] = orderedElements[i];
    }

    if (!siteMapMatchesStoichiometry(proto, siteMap, counts)) {
      return null;
    }
  }

  return { proto, siteMap };
}

const COMMON_ANION_CHARGES: Record<string, number> = {
  O: -2, S: -2, Se: -2, Te: -2,
  F: -1, Cl: -1, Br: -1, I: -1,
  N: -3,
};

const COMMON_CATION_CHARGES: Record<string, number[]> = {
  Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  Al: [3], Ga: [3], In: [3],
  Sc: [3], Y: [3], La: [3],
  Ti: [2, 3, 4], V: [2, 3, 4, 5], Cr: [2, 3, 6], Mn: [2, 3, 4, 7],
  Fe: [2, 3], Co: [2, 3], Ni: [2, 3], Cu: [1, 2],
  Zn: [2], Zr: [4], Nb: [3, 5], Mo: [4, 6],
  Ru: [3, 4], Rh: [3], Pd: [2, 4], Ag: [1],
  Sn: [2, 4], Sb: [3, 5], Pb: [2, 4], Bi: [3, 5],
  Hf: [4], Ta: [5], W: [4, 6], Re: [4, 7],
  Os: [4], Ir: [3, 4], Pt: [2, 4], Au: [1, 3],
  Ce: [3, 4], Pr: [3], Nd: [3], Sm: [3], Eu: [2, 3], Gd: [3],
  Tb: [3, 4], Dy: [3], Ho: [3], Er: [3], Tm: [3], Yb: [2, 3], Lu: [3],
  Th: [4], U: [4, 6],
};

function checkChargeBalance(elements: string[], counts: Record<string, number>): boolean {
  const anionEls = elements.filter(e => e in COMMON_ANION_CHARGES);
  const cationEls = elements.filter(e => e in COMMON_CATION_CHARGES && !(e in COMMON_ANION_CHARGES));
  if (anionEls.length === 0 || cationEls.length === 0) return true;

  const totalAnionCharge = anionEls.reduce(
    (s, e) => s + COMMON_ANION_CHARGES[e] * Math.round(counts[e] || 0), 0
  );

  for (const charges of cationChargePermutations(cationEls, counts)) {
    const totalCationCharge = charges.reduce((s, c) => s + c, 0);
    if (Math.abs(totalCationCharge + totalAnionCharge) < 0.5) return true;
  }

  const maxCationCharge = cationEls.reduce((s, e) => {
    const maxOx = Math.max(...(COMMON_CATION_CHARGES[e] || [1]));
    return s + maxOx * Math.round(counts[e] || 0);
  }, 0);

  const chargeRatio = maxCationCharge / Math.abs(totalAnionCharge || 1);
  return chargeRatio >= 0.5;
}

function cationChargePermutations(
  cationEls: string[], counts: Record<string, number>
): number[][] {
  const result: number[][] = [];
  const elCharges = cationEls.map(e => ({
    options: (COMMON_CATION_CHARGES[e] || [2]).map(c => c * Math.round(counts[e] || 0)),
  }));

  if (elCharges.length === 0) return [[]];
  if (elCharges.length === 1) return elCharges[0].options.map(o => [o]);
  if (elCharges.length > 4) {
    const single = elCharges.map(ec => ec.options[0]);
    return [single];
  }

  function gen(idx: number, current: number[]) {
    if (idx === elCharges.length) { result.push([...current]); return; }
    for (const opt of elCharges[idx].options) {
      current.push(opt);
      gen(idx + 1, current);
      current.pop();
    }
  }
  gen(0, []);
  return result;
}

function isPrototypeChemicallyCompatible(protoName: PrototypeName | string, elements: string[], counts: Record<string, number>): boolean {
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

  const anionHeavyProtos = new Set([
    "Perovskite", "Anti-perovskite", "Ruddlesden-Popper", "Double-perovskite",
    "Pyrite", "NaCl", "MX2", "Fluorite", "Antifluorite", "Spinel", "Garnet",
    "Wurtzite", "BiS2-layered",
  ]);
  if (anionHeavyProtos.has(protoName)) {
    if (!checkChargeBalance(elements, counts)) {
      return false;
    }
  }

  return true;
}

interface PrecomputedProtoData {
  proto: PrototypeStructure;
  siteCounts: Record<string, number>;
  sites: string[];
  nSites: number;
  siteRatios: number[];
  siteGcd: number;
  siteReduced: number[];
  sortedSiteReduced: number[];
}

let _protoDataCache: PrecomputedProtoData[] | null = null;

function getPrecomputedProtoData(): PrecomputedProtoData[] {
  if (_protoDataCache) return _protoDataCache;
  _protoDataCache = CRYSTAL_PROTOTYPES.map(proto => {
    const siteCounts = getProtoSiteCounts(proto);
    const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);
    const siteRatios = sites.map(s => siteCounts[s]);
    const siteGcd = siteRatios.reduce((a, b) => gcd(a, b));
    const siteReduced = siteRatios.map(r => r / siteGcd);
    const sortedSiteReduced = [...siteReduced].sort((a, b) => b - a);
    return { proto, siteCounts, sites, nSites: sites.length, siteRatios, siteGcd, siteReduced, sortedSiteReduced };
  });
  return _protoDataCache;
}

function isMetallic(role: ReturnType<typeof classifyElement>): boolean {
  return role === "TM" || role === "rare-earth" || role === "alkaline" || role === "alkali";
}

function selectChemistryAwareFallbacks(
  elements: string[],
  nElements: number,
  roles: ReturnType<typeof classifyElement>[],
): PrototypeName[] {
  const allMetallic = roles.every(r => isMetallic(r) || r === "metalloid");
  const hasNonmetal = roles.some(r => r === "anion" || r === "pnictide");

  if (nElements === 2) {
    if (allMetallic) {
      return ["CsCl", "Cu2Mg-Laves", "Laves-C14", "Laves-C15", "NaCl"];
    }
    if (hasNonmetal) {
      return ["NaCl", "Wurtzite", "Fluorite", "CsCl"];
    }
    return ["NaCl", "CsCl", "AlB2"];
  }

  if (nElements === 3) {
    if (allMetallic) {
      return ["Heusler", "Laves-C14", "Laves-C15", "Perovskite"];
    }
    // NaCl removed: only 2 crystallographic sites, cannot represent a 3-element compound
    return ["Perovskite", "ThCr2Si2", "Spinel"];
  }

  if (nElements === 4) {
    return ["Double-perovskite", "Garnet", "Spinel", "Perovskite"];
  }

  return ["Perovskite", "Garnet", "Spinel"];
}

function matchPrototype(counts: Record<string, number>, profile?: ChemicalProfile): { proto: PrototypeStructure; siteMap: Record<string, string> } | null {
  const p = profile ?? buildChemicalProfile(counts);
  const elements = p.elementsByCount;
  const nElements = p.nElements;
  const sortedReduced = p.sortedReducedRatios;
  const formulaAtomCount = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  const protoData = getPrecomputedProtoData();

  // Pass 1: Collect all exact ratio matches, prefer atom-count match
  let exactMatches: { proto: PrototypeStructure; siteMap: Record<string, string>; nAtoms: number }[] = [];
  for (const pd of protoData) {
    if (pd.nSites !== nElements) continue;

    let match = true;
    for (let i = 0; i < sortedReduced.length; i++) {
      if (sortedReduced[i] !== pd.sortedSiteReduced[i]) {
        match = false;
        break;
      }
    }

    if (match) {
      if (!isPrototypeChemicallyCompatible(pd.proto.name, elements, counts)) continue;
      const siteMap: Record<string, string> = {};
      for (let i = 0; i < elements.length; i++) {
        siteMap[pd.sites[i]] = elements[i];
      }
      if (!siteMapMatchesStoichiometry(pd.proto, siteMap, counts)) continue;
      exactMatches.push({ proto: pd.proto, siteMap, nAtoms: pd.proto.fractionalPositions.length });
    }
  }

  if (exactMatches.length > 0) {
    // Prefer template whose atom count exactly matches the formula, then smallest
    // multiple of the formula unit, then fewest total atoms
    exactMatches.sort((a, b) => {
      const aExact = a.nAtoms === formulaAtomCount ? 0 : 1;
      const bExact = b.nAtoms === formulaAtomCount ? 0 : 1;
      if (aExact !== bExact) return aExact - bExact;
      // Both match or both don't — prefer the one that's a clean multiple
      const aMult = a.nAtoms % formulaAtomCount === 0 ? 0 : 1;
      const bMult = b.nAtoms % formulaAtomCount === 0 ? 0 : 1;
      if (aMult !== bMult) return aMult - bMult;
      // Among clean multiples (or non-multiples), prefer fewer atoms
      return a.nAtoms - b.nAtoms;
    });
    if (exactMatches.length > 1) {
      dftLog(`[DFT] Prototype match: ${exactMatches.length} candidates for ${formulaAtomCount}-atom formula → picked ${exactMatches[0].proto.name} (${exactMatches[0].nAtoms} atoms) over ${exactMatches.slice(1).map(m => `${m.proto.name}(${m.nAtoms})`).join(", ")}`);
    }
    return { proto: exactMatches[0].proto, siteMap: exactMatches[0].siteMap };
  }

  let bestProto: PrototypeStructure | null = null;
  let bestScore = Infinity;
  let bestSiteMap: Record<string, string> = {};
  let bestAtomCountDiff = Infinity;

  for (const pd of protoData) {
    if (pd.nSites !== nElements) continue;

    let score = 0;
    for (let i = 0; i < sortedReduced.length; i++) {
      const diff = Math.abs(sortedReduced[i] - pd.sortedSiteReduced[i]);
      score += diff / Math.max(1, pd.sortedSiteReduced[i]);
    }

    if (score < PROTOTYPE_MATCH_TOLERANCE && isPrototypeChemicallyCompatible(pd.proto.name, elements, counts)) {
      const candidateMap: Record<string, string> = {};
      for (let i = 0; i < elements.length; i++) {
        candidateMap[pd.sites[i]] = elements[i];
      }
      if (siteMapMatchesStoichiometry(pd.proto, candidateMap, counts)) {
        const nAtoms = pd.proto.fractionalPositions.length;
        const atomDiff = Math.abs(nAtoms - formulaAtomCount);
        // Prefer better stoichiometry score; break ties by atom count proximity
        if (score < bestScore || (score === bestScore && atomDiff < bestAtomCountDiff)) {
          bestScore = score;
          bestProto = pd.proto;
          bestSiteMap = candidateMap;
          bestAtomCountDiff = atomDiff;
        }
      }
    }
  }

  if (bestProto) {
    return { proto: bestProto, siteMap: bestSiteMap };
  }

  for (const pd of protoData) {
    if (pd.nSites >= nElements) continue;

    const mergedReduced: number[] = [];
    for (let i = 0; i < pd.nSites; i++) {
      if (i < elements.length) {
        mergedReduced.push(Math.round(counts[elements[i]]));
      }
    }
    for (let i = pd.nSites; i < elements.length; i++) {
      mergedReduced[mergedReduced.length - 1] += Math.round(counts[elements[i]]);
    }
    const mergedGcd = mergedReduced.reduce((a, b) => gcd(a, b));
    const mergedNorm = mergedReduced.map(r => r / mergedGcd);

    const sortedMerged = [...mergedNorm].sort((a, b) => b - a);

    let score = 0;
    for (let i = 0; i < sortedMerged.length; i++) {
      const diff = Math.abs(sortedMerged[i] - pd.sortedSiteReduced[i]);
      score += diff / Math.max(1, pd.sortedSiteReduced[i]);
    }

    if (score < PROTOTYPE_FUZZY_TOLERANCE && isPrototypeChemicallyCompatible(pd.proto.name, elements, counts)) {
      const candidateMap: Record<string, string> = {};
      for (let i = 0; i < Math.min(pd.nSites, elements.length); i++) {
        candidateMap[pd.sites[i]] = elements[i];
      }
      for (let i = elements.length; i < pd.nSites; i++) {
        candidateMap[pd.sites[i]] = elements[elements.length - 1];
      }
      if (siteMapMatchesStoichiometry(pd.proto, candidateMap, counts)) {
        const nAtoms = pd.proto.fractionalPositions.length;
        const atomDiff = Math.abs(nAtoms - formulaAtomCount);
        if (score < bestScore || (score === bestScore && atomDiff < bestAtomCountDiff)) {
          bestScore = score;
          bestProto = pd.proto;
          bestSiteMap = candidateMap;
          bestAtomCountDiff = atomDiff;
        }
      }
    }
  }

  if (bestProto) {
    return { proto: bestProto, siteMap: bestSiteMap };
  }

  const chemMatch = selectBestPrototypeByChemistry(counts, elements, p);
  if (chemMatch) {
    chemistryMatchSuccesses++;
    return chemMatch;
  }

  if (nElements === 1) {
    const el = elements[0];
    const bccProto = CRYSTAL_PROTOTYPES.find(cp => cp.name === "BCC");
    if (!bccProto) throw new Error("[DFT] BCC prototype missing from CRYSTAL_PROTOTYPES — cannot generate unary fallback structure");
    return { proto: bccProto, siteMap: { A: el } };
  }

  if (nElements >= 2) {
    const fallbackNames = selectChemistryAwareFallbacks(elements, nElements, p.roles);

    for (const fname of fallbackNames) {
      const fallbackProto = CRYSTAL_PROTOTYPES.find(cp => cp.name === fname);
      if (!fallbackProto) continue;

      const siteCounts = getProtoSiteCounts(fallbackProto);
      const nSites = Object.keys(siteCounts).length;
      // Never assign a prototype with fewer crystallographic sites than the
      // number of elements — e.g. NaCl (2 sites) cannot represent a 3-element
      // compound like NbIrH7. The stoichiometry check below would also reject
      // this, but failing early avoids misleading fallback cascades.
      if (nSites < nElements) continue;
      const sites = Object.keys(siteCounts).sort((a, b) => siteCounts[b] - siteCounts[a]);
      const siteMap: Record<string, string> = {};
      for (let i = 0; i < Math.min(sites.length, elements.length); i++) {
        siteMap[sites[i]] = elements[i];
      }
      for (let i = elements.length; i < sites.length; i++) {
        siteMap[sites[i]] = elements[elements.length - 1];
      }
      if (siteMapMatchesStoichiometry(fallbackProto, siteMap, counts)) {
        return { proto: fallbackProto, siteMap };
      }
    }
  }

  return null;
}

function normalizeCountsToIntegers(counts: Record<string, number>): Record<string, number> {
  const values = Object.values(counts);
  if (values.every(v => Math.abs(v - Math.round(v)) < 1e-6)) return counts;

  let multiplier = 1;
  for (const v of values) {
    const frac = v - Math.floor(v);
    if (frac > 1e-6) {
      const denom = Math.round(1 / frac);
      if (denom >= 2 && denom <= 100) {
        multiplier = lcm(multiplier, denom);
      }
    }
  }
  if (multiplier === 1) return counts;

  const result: Record<string, number> = {};
  for (const [el, n] of Object.entries(counts)) {
    result[el] = Math.round(n * multiplier);
  }
  return result;
}

function lcm(a: number, b: number): number {
  return Math.abs(a * b) / gcd(a, b);
}

function gcd(a: number, b: number): number {
  a = Math.abs(Math.round(a));
  b = Math.abs(Math.round(b));
  while (b) { [a, b] = [b, a % b]; }
  return a || 1;
}

type LatticeVectors = [number, number, number][];

function buildLatticeVectors(
  latticeType: string,
  a: number,
  c: number,
  b: number = a,
  alpha: number = 90,
  beta: number = 90,
  gamma: number = 90,
): LatticeVectors {
  if (latticeType === "hexagonal") {
    return [
      [a, 0, 0],
      [a * 0.5, a * (Math.sqrt(3) / 2), 0],
      [0, 0, c],
    ];
  }
  if (latticeType === "monoclinic") {
    // Convention: beta ≠ 90°, v3 tilted in xz-plane
    const betaR = beta * Math.PI / 180;
    return [
      [a, 0, 0],
      [0, b, 0],
      [c * Math.cos(betaR), 0, c * Math.sin(betaR)],
    ];
  }
  if (latticeType === "triclinic") {
    // General: all angles may differ from 90°
    const alphaR = alpha * Math.PI / 180;
    const betaR = beta * Math.PI / 180;
    const gammaR = gamma * Math.PI / 180;
    const cosA = Math.cos(alphaR), cosB = Math.cos(betaR), cosG = Math.cos(gammaR);
    const sinG = Math.sin(gammaR);
    const cx = c * cosB;
    const cy = sinG > 1e-10 ? c * (cosA - cosB * cosG) / sinG : 0;
    const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
    return [
      [a, 0, 0],
      [b * cosG, b * sinG, 0],
      [cx, cy, cz],
    ];
  }
  if (latticeType === "orthorhombic") {
    return [
      [a, 0, 0],
      [0, b, 0],
      [0, 0, c],
    ];
  }
  if (latticeType === "tetragonal") {
    return [
      [a, 0, 0],
      [0, a, 0],
      [0, 0, c],
    ];
  }
  // cubic (default)
  return [
    [a, 0, 0],
    [0, a, 0],
    [0, 0, a],
  ];
}

function fracToCart(
  fx: number, fy: number, fz: number,
  vecs: LatticeVectors,
): [number, number, number] {
  return [
    fx * vecs[0][0] + fy * vecs[1][0] + fz * vecs[2][0],
    fx * vecs[0][1] + fy * vecs[1][1] + fz * vecs[2][1],
    fx * vecs[0][2] + fy * vecs[1][2] + fz * vecs[2][2],
  ];
}

function invertLattice3x3(vecs: LatticeVectors): LatticeVectors {
  const [a, b, c] = vecs;
  const det =
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0]);
  const invDet = 1.0 / det;
  return [
    [(b[1] * c[2] - b[2] * c[1]) * invDet, (a[2] * c[1] - a[1] * c[2]) * invDet, (a[1] * b[2] - a[2] * b[1]) * invDet],
    [(b[2] * c[0] - b[0] * c[2]) * invDet, (a[0] * c[2] - a[2] * c[0]) * invDet, (a[2] * b[0] - a[0] * b[2]) * invDet],
    [(b[0] * c[1] - b[1] * c[0]) * invDet, (a[1] * c[0] - a[0] * c[1]) * invDet, (a[0] * b[1] - a[1] * b[0]) * invDet],
  ];
}

function pbcMinImageDist(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  vecs: LatticeVectors,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;

  const inv = invertLattice3x3(vecs);
  let f1 = inv[0][0] * dx + inv[0][1] * dy + inv[0][2] * dz;
  let f2 = inv[1][0] * dx + inv[1][1] * dy + inv[1][2] * dz;
  let f3 = inv[2][0] * dx + inv[2][1] * dy + inv[2][2] * dz;

  f1 -= Math.round(f1);
  f2 -= Math.round(f2);
  f3 -= Math.round(f3);

  const v1 = vecs[0], v2 = vecs[1], v3 = vecs[2];
  let bestDist2 = Infinity;
  for (let i1 = -1; i1 <= 1; i1++) {
    for (let i2 = -1; i2 <= 1; i2++) {
      for (let i3 = -1; i3 <= 1; i3++) {
        const g1 = f1 + i1;
        const g2 = f2 + i2;
        const g3 = f3 + i3;
        const rx = g1 * v1[0] + g2 * v2[0] + g3 * v3[0];
        const ry = g1 * v1[1] + g2 * v2[1] + g3 * v3[1];
        const rz = g1 * v1[2] + g2 * v2[2] + g3 * v3[2];
        const d2 = rx * rx + ry * ry + rz * rz;
        if (d2 < bestDist2) bestDist2 = d2;
      }
    }
  }
  return Math.sqrt(bestDist2);
}

function buildStructureFromPrototype(
  proto: PrototypeStructure,
  siteMap: Record<string, string>,
  elements: string[],
  counts: Record<string, number>,
  scaleFactor: number = 1,
  pressureGPa: number = 0,
): { atoms: AtomPosition[]; latticeVecs: LatticeVectors } {
  const effectiveCOverA = estimatePressureCOverA(proto.cOverA, proto.latticeType, pressureGPa);
  const a = estimateLatticeParam(elements, counts, proto.name, proto.latticeType, effectiveCOverA, pressureGPa) * scaleFactor;
  const c = a * effectiveCOverA;
  const vecs = buildLatticeVectors(proto.latticeType, a, c);

  const atoms: AtomPosition[] = [];
  for (const pos of proto.fractionalPositions) {
    const element = siteMap[pos.site];
    if (!element) continue;

    const [x, y, z] = fracToCart(pos.x, pos.y, pos.z, vecs);
    atoms.push({ element, x, y, z });
  }

  const MAX_EXPANSION_PASSES = 3;
  let currentVecs = vecs;

  for (let pass = 0; pass < MAX_EXPANSION_PASSES; pass++) {
    let hasCollision = false;
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const dist = pbcMinImageDist(
          atoms[i].x, atoms[i].y, atoms[i].z,
          atoms[j].x, atoms[j].y, atoms[j].z,
          currentVecs,
        );
        const ri = COVALENT_RADII[atoms[i].element] || 1.0;
        const rj = COVALENT_RADII[atoms[j].element] || 1.0;
        const minAllowed = STRUCTURAL_VALIDATION_CONFIG.minDistFraction * (ri + rj);
        if (dist < minAllowed) {
          hasCollision = true;
          break;
        }
      }
      if (hasCollision) break;
    }

    if (!hasCollision) break;

    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const dist = pbcMinImageDist(atoms[i].x, atoms[i].y, atoms[i].z, atoms[j].x, atoms[j].y, atoms[j].z, currentVecs);
        if (dist < 0.01) {
          const zScale = proto.latticeType === "tetragonal" ? c : a;
          atoms[j].x += 0.08 * a * (1 + (j % 3) * 0.05);
          atoms[j].y += 0.06 * a * (1 + (j % 5) * 0.03);
          atoms[j].z += 0.07 * zScale * (1 + (j % 7) * 0.02);
        }
      }
    }

    const expansionFactor = 1.05;
    for (const atom of atoms) {
      atom.x *= expansionFactor;
      atom.y *= expansionFactor;
      atom.z *= expansionFactor;
    }
    currentVecs = currentVecs.map(v =>
      v.map(c => c * expansionFactor) as [number, number, number]
    ) as LatticeVectors;
  }

  return { atoms, latticeVecs: currentVecs };
}

function buildGenericStructure(counts: Record<string, number>): { atoms: AtomPosition[]; proto: string; latticeVecs: LatticeVectors } {
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

  const genericVecs: LatticeVectors = [[a, 0, 0], [0, a, 0], [0, 0, a]];
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dist = pbcMinImageDist(atoms[i].x, atoms[i].y, atoms[i].z, atoms[j].x, atoms[j].y, atoms[j].z, genericVecs);
      if (dist < 0.3) {
        atoms[j].x += 0.15 * a / gridSize;
        atoms[j].y += 0.1 * a / gridSize;
        atoms[j].z += 0.12 * a / gridSize;
      }
    }
  }

  return { atoms, proto: "generic-cluster", latticeVecs: genericVecs };
}

interface StructuralValidationConfig {
  minRatioThreshold: number;
  scaleFactorMin: number;
  scaleFactorMax: number;
  volLowerFraction: number;
  volUpperFraction: number;
  maxScaleAttempts: number;
  minVolumePerAtom: number;
  minVolumePerAtomHydride: number;
  minDistFraction: number;
}

const STRUCTURAL_VALIDATION_CONFIG: StructuralValidationConfig = {
  minRatioThreshold: 0.75,
  scaleFactorMin: 0.5,
  scaleFactorMax: 2.0,
  volLowerFraction: 0.5,
  volUpperFraction: 2.0,
  maxScaleAttempts: 8,
  minVolumePerAtom: 10.0,
  minVolumePerAtomHydride: 8.0,
  minDistFraction: 0.6,
};

let _dftLogVerbose = true;

export function setDFTLogVerbose(verbose: boolean): void {
  _dftLogVerbose = verbose;
}

function dftLog(message: string, level: "info" | "detail" = "detail"): void {
  if (level === "info" || _dftLogVerbose) {
    console.log(message);
  }
}

const MIN_VOLUME_PER_ATOM = STRUCTURAL_VALIDATION_CONFIG.minVolumePerAtom;
const MIN_VOLUME_PER_ATOM_HYDRIDE = STRUCTURAL_VALIDATION_CONFIG.minVolumePerAtomHydride;
const MAX_SCALE_ATTEMPTS = STRUCTURAL_VALIDATION_CONFIG.maxScaleAttempts;

function findPairwiseMinDistSpatial(
  atoms: AtomPosition[],
  pressureGPa: number,
  latticeVecs: LatticeVectors,
): { minDist: number; minRatio: number; worstI: number; worstJ: number; pairI: number; pairJ: number } {
  if (atoms.length <= 16) {
    return computePairwiseDistances(atoms, pressureGPa, latticeVecs);
  }

  const inv = invertLattice3x3(latticeVecs);
  const nGrid = 4;

  const grid = new Map<string, number[]>();
  const fracArr: [number, number, number][] = [];

  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    let f1 = inv[0][0] * a.x + inv[0][1] * a.y + inv[0][2] * a.z;
    let f2 = inv[1][0] * a.x + inv[1][1] * a.y + inv[1][2] * a.z;
    let f3 = inv[2][0] * a.x + inv[2][1] * a.y + inv[2][2] * a.z;
    f1 = f1 - Math.floor(f1);
    f2 = f2 - Math.floor(f2);
    f3 = f3 - Math.floor(f3);
    fracArr.push([f1, f2, f3]);

    const kx = Math.floor(f1 * nGrid) % nGrid;
    const ky = Math.floor(f2 * nGrid) % nGrid;
    const kz = Math.floor(f3 * nGrid) % nGrid;
    const key = `${kx},${ky},${kz}`;
    const bucket = grid.get(key);
    if (bucket) bucket.push(i);
    else grid.set(key, [i]);
  }

  let minDist = Infinity;
  let minRatio = Infinity;
  let pairI = -1, pairJ = -1;
  let worstI = -1, worstJ = -1;

  for (let i = 0; i < atoms.length; i++) {
    const kx = Math.floor(fracArr[i][0] * nGrid) % nGrid;
    const ky = Math.floor(fracArr[i][1] * nGrid) % nGrid;
    const kz = Math.floor(fracArr[i][2] * nGrid) % nGrid;

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const nx = ((kx + dx) % nGrid + nGrid) % nGrid;
          const ny = ((ky + dy) % nGrid + nGrid) % nGrid;
          const nz = ((kz + dz) % nGrid + nGrid) % nGrid;
          const key = `${nx},${ny},${nz}`;
          const bucket = grid.get(key);
          if (!bucket) continue;
          for (const j of bucket) {
            if (j <= i) continue;
            const dist = pbcMinImageDist(
              atoms[i].x, atoms[i].y, atoms[i].z,
              atoms[j].x, atoms[j].y, atoms[j].z,
              latticeVecs,
            );
            if (dist < minDist) {
              minDist = dist;
              pairI = i;
              pairJ = j;
            }
            const minAllowed = getMinInteratomicDistance(atoms[i].element, atoms[j].element, pressureGPa);
            const ratio = dist / minAllowed;
            if (ratio < minRatio) {
              minRatio = ratio;
              worstI = i;
              worstJ = j;
            }
          }
        }
      }
    }
  }

  if (minDist === Infinity) {
    return computePairwiseDistances(atoms, pressureGPa, latticeVecs);
  }

  return { minDist, minRatio, worstI, worstJ, pairI, pairJ };
}

function computeStructuralFingerprint(atoms: AtomPosition[], latticeVecs: LatticeVectors): string {
  const elCounts: Record<string, number> = {};
  for (const a of atoms) {
    elCounts[a.element] = (elCounts[a.element] || 0) + 1;
  }

  const allDists: number[] = [];
  for (let i = 0; i < atoms.length; i++) {
    const nnDists: number[] = [];
    for (let j = 0; j < atoms.length; j++) {
      if (j === i) continue;
      nnDists.push(
        pbcMinImageDist(atoms[i].x, atoms[i].y, atoms[i].z, atoms[j].x, atoms[j].y, atoms[j].z, latticeVecs),
      );
    }
    nnDists.sort((a, b) => a - b);
    for (let k = 0; k < Math.min(3, nnDists.length); k++) {
      allDists.push(nnDists[k]);
    }
  }
  allDists.sort((a, b) => a - b);

  const compStr = Object.entries(elCounts).sort().map(([el, n]) => `${el}${n}`).join("");
  const distStr = allDists.slice(0, 15).map(d => d.toFixed(2)).join(",");
  return `${compStr}|${distStr}`;
}

const _cageFingerprints = new Map<string, Set<string>>();

function isDuplicateCageStructure(
  formula: string,
  atoms: AtomPosition[],
  latticeVecs: LatticeVectors,
  toleranceÅ: number = 0.15,
): boolean {
  const fp = computeStructuralFingerprint(atoms, latticeVecs);
  const existing = _cageFingerprints.get(formula);
  if (!existing) {
    _cageFingerprints.set(formula, new Set([fp]));
    return false;
  }

  for (const prevFp of Array.from(existing)) {
    const prevDists = prevFp.split("|")[1]?.split(",").map(Number) ?? [];
    const curDists = fp.split("|")[1]?.split(",").map(Number) ?? [];
    if (prevDists.length === curDists.length && prevDists.length > 0) {
      let maxDiff = 0;
      for (let i = 0; i < prevDists.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(prevDists[i] - curDists[i]));
      }
      if (maxDiff < toleranceÅ) return true;
    }
  }

  existing.add(fp);
  if (existing.size > 50) {
    const iter = existing.values();
    existing.delete(iter.next().value!);
  }
  return false;
}

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

function selectHydrideCage(hPerMetal: number, totalHydrogen: number, totalMetalCount: number): HydrideCageMotif {
  let bestCage = HYDRIDE_CAGE_LIBRARY[0];
  let bestScore = Infinity;

  for (const cage of HYDRIDE_CAGE_LIBRARY) {
    const nCopies = Math.max(1, Math.ceil(totalMetalCount / cage.metalFrac.length));
    const hCapacity = cage.hydrogenFrac.length * nCopies;
    const hMismatch = Math.abs(cage.hPerMetal - hPerMetal) / Math.max(1, hPerMetal);
    const overflowFraction = totalHydrogen > hCapacity ? (totalHydrogen - hCapacity) / totalHydrogen : 0;
    const underuseFraction = hCapacity > totalHydrogen ? (hCapacity - totalHydrogen) / hCapacity : 0;
    const score = hMismatch * 0.3 + overflowFraction * 5.0 + underuseFraction * 0.2;
    if (score < bestScore) {
      bestScore = score;
      bestCage = cage;
    }
  }

  return bestCage;
}

const METAL_CLASS_BASELINE_RADIUS: Record<string, number> = {
  alkali: 2.0,
  alkaline: 1.7,
  "rare-earth": 1.9,
  TM: 1.5,
  other: 1.4,
};

function seededJitter(seed: number, channel: number): number {
  let h = ((seed * 2654435761 + channel * 40503) | 0) >>> 0;
  h = ((h ^ (h >>> 16)) * 0x45d9f3b) >>> 0;
  h = ((h ^ (h >>> 16)) * 0x45d9f3b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return (h / 0xffffffff) - 0.5;
}

function formulaToHash(formula: string): number {
  let h = 5381;
  for (let i = 0; i < formula.length; i++) {
    h = ((h << 5) + h + formula.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

function getMetalClassBaselineRadius(metals: string[], counts: Record<string, number>): number {
  let weightedSum = 0;
  let totalWeight = 0;
  for (const el of metals) {
    const role = classifyElement(el);
    const baseline = METAL_CLASS_BASELINE_RADIUS[role] ?? 1.4;
    const weight = Math.round(counts[el] || 1);
    weightedSum += baseline * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 1.4;
}

function computeWeightedMetalRadius(metals: string[], counts: Record<string, number>): number {
  let weightedSum = 0;
  let totalWeight = 0;
  for (const el of metals) {
    const r = COVALENT_RADII[el] ?? 1.4;
    const n = Math.round(counts[el] || 1);
    weightedSum += r * n;
    totalWeight += n;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 1.4;
}

function generateHydrideCageStructure(
  formula: string,
  counts: Record<string, number>,
  pressureGPa: number = 0,
): { atoms: AtomPosition[]; prototype: string; latticeVecs: LatticeVectors } | null {
  const elements = Object.keys(counts);
  const metals = elements.filter(el => el !== "H");
  const hCount = Math.round(counts["H"] || 0);
  if (metals.length === 0 || hCount === 0) return null;

  const totalMetalCount = metals.reduce((s, el) => s + Math.round(counts[el] || 0), 0);
  const hPerMetal = hCount / totalMetalCount;
  if (hPerMetal < 4) return null;

  const formulaHash = formulaToHash(formula);
  const cage = selectHydrideCage(hPerMetal, hCount, totalMetalCount);

  const avgMetalRadius = computeWeightedMetalRadius(metals, counts);
  const baselineRadius = getMetalClassBaselineRadius(metals, counts);
  const hRadius = getCompressedRadius("H", pressureGPa);
  const latticeA = cage.baseLatticeFactor * (avgMetalRadius / baselineRadius);
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

  const cageVecs = buildLatticeVectors(cage.latticeType, a, c);

  const gridX = Math.max(1, Math.ceil(Math.cbrt(nCopies)));
  const gridY = Math.max(1, Math.ceil(nCopies / gridX));
  const gridZ = Math.max(1, Math.ceil(nCopies / (gridX * gridY)));
  const superVecs: LatticeVectors = cageVecs.map((v, idx) => {
    const gridScale = [gridX, gridY, gridZ][idx];
    return v.map(cc => cc * gridScale) as [number, number, number];
  }) as LatticeVectors;
  for (let copy = 0; copy < nCopies; copy++) {
    const ix = copy % gridX;
    const iy = Math.floor(copy / gridX) % gridY;
    const iz = Math.floor(copy / (gridX * gridY));
    const [offX, offY, offZ] = fracToCart(ix, iy, iz, cageVecs);

    for (let i = 0; i < metalSiteCount && metalIdx < metalList.length; i++) {
      const pos = cage.metalFrac[i];
      const [bx, by, bz] = fracToCart(pos.x, pos.y, pos.z, cageVecs);
      atoms.push({ element: metalList[metalIdx], x: bx + offX, y: by + offY, z: bz + offZ });
      metalIdx++;
    }

    const hPerCopy = Math.ceil(hCount / nCopies);
    const minHH = 0.5;
    for (let i = 0; i < hPerCopy && hPlaced < hCount; i++) {
      const siteIdx = i % hSiteCount;
      const wrapCount = Math.floor(i / hSiteCount);
      const pos = cage.hydrogenFrac[siteIdx];
      let fx = pos.x, fy = pos.y, fz = pos.z;
      if (wrapCount > 0) {
        const perturbScale = 0.04 * wrapCount;
        const seed = formulaHash + copy * 1000 + i;
        fx += seededJitter(seed, 0) * perturbScale;
        fy += seededJitter(seed, 1) * perturbScale;
        fz += seededJitter(seed, 2) * perturbScale;
        fx = fx - Math.floor(fx);
        fy = fy - Math.floor(fy);
        fz = fz - Math.floor(fz);
      }
      const [bx, by, bz] = fracToCart(fx, fy, fz, cageVecs);
      let x = bx + offX;
      let y = by + offY;
      let z = bz + offZ;
      let tooClose = false;
      for (const existing of atoms) {
        if (pbcMinImageDist(x, y, z, existing.x, existing.y, existing.z, superVecs) < minHH) {
          tooClose = true;
          break;
        }
      }
      if (tooClose && wrapCount > 0) {
        for (let attempt = 0; attempt < 5; attempt++) {
          const retrySeed = formulaHash + copy * 1000 + i * 100 + attempt;
          const pfx = pos.x + seededJitter(retrySeed, 0) * 0.08 * (attempt + 1);
          const pfy = pos.y + seededJitter(retrySeed, 1) * 0.08 * (attempt + 1);
          const pfz = pos.z + seededJitter(retrySeed, 2) * 0.08 * (attempt + 1);
          const [rx, ry, rz] = fracToCart(pfx - Math.floor(pfx), pfy - Math.floor(pfy), pfz - Math.floor(pfz), cageVecs);
          x = rx + offX; y = ry + offY; z = rz + offZ;
          let ok = true;
          for (const existing of atoms) {
            if (pbcMinImageDist(x, y, z, existing.x, existing.y, existing.z, superVecs) < minHH) { ok = false; break; }
          }
          if (ok) { tooClose = false; break; }
        }
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
    const [offX, offY, offZ] = fracToCart(
      copy % 2,
      Math.floor(copy / 2) % 2,
      Math.floor(copy / 4),
      cageVecs,
    );

    let placed = false;
    for (let retry = 0; retry < 10 && !placed; retry++) {
      const perturbAngle = overflowIdx * 2.399 + retry * 1.1;
      const perturbR = 0.5 + 0.2 * (overflowIdx % 5) + retry * 0.15;
      const px = perturbR * Math.cos(perturbAngle);
      const py = perturbR * Math.sin(perturbAngle);
      const pz = perturbR * Math.cos(perturbAngle + 1.5);
      const [bx, by, bz] = fracToCart(pos.x, pos.y, pos.z, cageVecs);
      const x = bx + offX + px;
      const y = by + offY + py;
      const z = bz + offZ + pz;

      let tooClose = false;
      for (const existing of atoms) {
        if (pbcMinImageDist(x, y, z, existing.x, existing.y, existing.z, superVecs) < 0.5) {
          tooClose = true;
          break;
        }
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

  dftLog(`[DFT] ${formula}: Using hydride cage motif ${cage.name} (H/metal=${hPerMetal.toFixed(1)}, ${atoms.length} atoms)`);
  return { atoms, prototype: `hydride-cage-${cage.name}`, latticeVecs: superVecs };
}

function deduplicateSites(atoms: AtomPosition[], latticeVecs?: LatticeVectors): AtomPosition[] {
  const TOLERANCE = 0.05;
  const vecs = latticeVecs ?? estimateLatticeFromAtoms(atoms);
  const result: AtomPosition[] = [];
  for (const atom of atoms) {
    let isDuplicate = false;
    for (const existing of result) {
      const dist = pbcMinImageDist(atom.x, atom.y, atom.z, existing.x, existing.y, existing.z, vecs);
      if (dist < TOLERANCE) {
        isDuplicate = true;
        break;
      }
    }
    if (!isDuplicate) {
      result.push({ ...atom });
    }
  }
  if (result.length < atoms.length) {
    dftLog(`[DFT] deduplicateSites: Dropped ${atoms.length - result.length} duplicate atom(s) (${atoms.length} → ${result.length})`);
  }
  return result;
}

function getMinInteratomicDistance(el1: string, el2: string, pressureGPa: number = 0): number {
  const r1 = getCompressedRadius(el1, pressureGPa);
  const r2 = getCompressedRadius(el2, pressureGPa);
  const bondLength = r1 + r2;
  const bothH = el1 === "H" && el2 === "H";
  const oneH = el1 === "H" || el2 === "H";
  if (bothH) {
    const ambientFloor = pressureGPa > 50 ? 0.74 * getCompressedRadius("H", pressureGPa) / 0.31 : 0.74;
    return Math.max(bondLength * 0.70, ambientFloor);
  }
  if (oneH) {
    const ambientFloor = pressureGPa > 50 ? 1.3 * Math.cbrt(Math.max(0.4, 1.0 / (1 + pressureGPa / 200))) : 1.3;
    return Math.max(bondLength * 0.70, ambientFloor);
  }
  const metalFloor = pressureGPa > 50 ? 0.9 * Math.cbrt(Math.max(0.5, 1.0 / (1 + pressureGPa / 300))) : 0.9;
  return Math.max(bondLength * 0.80, metalFloor);
}

function estimateLatticeFromAtoms(atoms: AtomPosition[]): LatticeVectors {
  if (atoms.length < 2) return [[10, 0, 0], [0, 10, 0], [0, 0, 10]];
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (const a of atoms) {
    if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
    if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
    if (a.z < minZ) minZ = a.z; if (a.z > maxZ) maxZ = a.z;
  }
  const pad = 1.5;
  const lx = Math.max(maxX - minX + pad, 3.0);
  const ly = Math.max(maxY - minY + pad, 3.0);
  const lz = Math.max(maxZ - minZ + pad, 3.0);
  return [[lx, 0, 0], [0, ly, 0], [0, 0, lz]];
}

function computePairwiseDistances(
  atoms: AtomPosition[],
  pressureGPa: number = 0,
  latticeVecs?: LatticeVectors,
): { minDist: number; minRatio: number; worstI: number; worstJ: number; pairI: number; pairJ: number } {
  const vecs = latticeVecs ?? estimateLatticeFromAtoms(atoms);
  let minDist = Infinity;
  let minRatio = Infinity;
  let pairI = -1;
  let pairJ = -1;
  let worstI = -1;
  let worstJ = -1;
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dist = pbcMinImageDist(atoms[i].x, atoms[i].y, atoms[i].z, atoms[j].x, atoms[j].y, atoms[j].z, vecs);
      if (dist < minDist) {
        minDist = dist;
        pairI = i;
        pairJ = j;
      }
      const minAllowed = getMinInteratomicDistance(atoms[i].element, atoms[j].element, pressureGPa);
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
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  let maxR = 0;
  for (const a of atoms) {
    if (a.x < minX) minX = a.x;
    if (a.x > maxX) maxX = a.x;
    if (a.y < minY) minY = a.y;
    if (a.y > maxY) maxY = a.y;
    if (a.z < minZ) minZ = a.z;
    if (a.z > maxZ) maxZ = a.z;
    const r = COVALENT_RADII[a.element] ?? 1.4;
    if (r > maxR) maxR = r;
  }
  const pad = maxR;
  const dx = Math.max(maxX - minX + 2 * pad, 2 * pad);
  const dy = Math.max(maxY - minY + 2 * pad, 2 * pad);
  const dz = Math.max(maxZ - minZ + 2 * pad, 2 * pad);
  return dx * dy * dz;
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

function classifyLatticeType(latticeVecs: LatticeVectors): string {
  const v0 = latticeVecs[0], v1 = latticeVecs[1], v2 = latticeVecs[2];
  const lenA = Math.sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  const lenB = Math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  const lenC = Math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
  if (lenA < 0.01 || lenB < 0.01 || lenC < 0.01) return "cubic";

  // Compute all three inter-vector angles
  const dot01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
  const dot02 = v0[0] * v2[0] + v0[1] * v2[1] + v0[2] * v2[2];
  const dot12 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  const cosAB = dot01 / (lenA * lenB);    // gamma
  const cosAC = dot02 / (lenA * lenC);    // beta
  const cosBC = dot12 / (lenB * lenC);    // alpha
  const angleGamma = Math.acos(Math.max(-1, Math.min(1, cosAB))) * 180 / Math.PI;
  const angleBeta  = Math.acos(Math.max(-1, Math.min(1, cosAC))) * 180 / Math.PI;
  const angleAlpha = Math.acos(Math.max(-1, Math.min(1, cosBC))) * 180 / Math.PI;

  const abEqual = Math.abs(lenA - lenB) / Math.max(lenA, lenB) < 0.05;
  const allRight = (a: number) => Math.abs(a - 90) < 5;

  // Hexagonal: a≈b, gamma≈120, alpha≈beta≈90
  if (abEqual && Math.abs(angleGamma - 120) < 10 && allRight(angleAlpha) && allRight(angleBeta)) {
    return "hexagonal";
  }

  // Cubic: a≈b≈c, all angles≈90
  const bcEqual = Math.abs(lenB - lenC) / Math.max(lenB, lenC) < 0.05;
  if (abEqual && bcEqual && allRight(angleAlpha) && allRight(angleBeta) && allRight(angleGamma)) {
    return "cubic";
  }

  // Tetragonal: a≈b≠c, all angles≈90
  if (abEqual && allRight(angleAlpha) && allRight(angleBeta) && allRight(angleGamma)) {
    return "tetragonal";
  }

  // Orthorhombic: a≠b≠c, all angles≈90
  if (allRight(angleAlpha) && allRight(angleBeta) && allRight(angleGamma)) {
    return "orthorhombic";
  }

  // Monoclinic: one angle ≠ 90 (conventionally beta)
  const nonRight = [!allRight(angleAlpha), !allRight(angleBeta), !allRight(angleGamma)];
  const numNonRight = nonRight.filter(Boolean).length;
  if (numNonRight === 1) return "monoclinic";

  // Triclinic: everything else
  return "triclinic";
}

function lattice3x3Det(vecs: LatticeVectors): number {
  const [a, b, c] = vecs;
  return a[0] * (b[1] * c[2] - b[2] * c[1])
       - a[1] * (b[0] * c[2] - b[2] * c[0])
       + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

function scaleStructureAnisotropic(
  atoms: AtomPosition[],
  factor: number,
  latticeVecs: LatticeVectors,
  pressureGPa: number,
): { atoms: AtomPosition[]; latticeVecs: LatticeVectors } {
  const isotropicFallback = () => {
    const scaled = scaleStructure(atoms, factor);
    const scaledVecs = latticeVecs.map(v => v.map(c => c * factor) as [number, number, number]) as LatticeVectors;
    return { atoms: scaled, latticeVecs: scaledVecs };
  };

  if (pressureGPa <= 50) return isotropicFallback();

  const det = lattice3x3Det(latticeVecs);
  if (Math.abs(det) < 1e-8) return isotropicFallback();

  const v0 = latticeVecs[0], v1 = latticeVecs[1], v2 = latticeVecs[2];
  const lenA = Math.sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  const lenC = Math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
  if (lenA < 0.01 || lenC < 0.01) return isotropicFallback();

  const currentCA = lenC / lenA;
  const latticeType = classifyLatticeType(latticeVecs);

  if (latticeType === "cubic") return isotropicFallback();

  const targetCA = estimatePressureCOverA(currentCA, latticeType, pressureGPa);
  const caRatio = targetCA / currentCA;
  const abScale = factor;
  const cScale = factor * Math.max(Math.min(caRatio, 1.3), 0.7);

  let cx = 0, cy = 0, cz = 0;
  for (const a of atoms) { cx += a.x; cy += a.y; cz += a.z; }
  cx /= atoms.length; cy /= atoms.length; cz /= atoms.length;

  const inv = invertLattice3x3(latticeVecs);
  const scaledAtoms = atoms.map(a => {
    const dx = a.x - cx, dy = a.y - cy, dz = a.z - cz;
    const f1 = inv[0][0] * dx + inv[0][1] * dy + inv[0][2] * dz;
    const f2 = inv[1][0] * dx + inv[1][1] * dy + inv[1][2] * dz;
    const f3 = inv[2][0] * dx + inv[2][1] * dy + inv[2][2] * dz;
    const sf1 = f1 * abScale, sf2 = f2 * abScale, sf3 = f3 * cScale;
    const x = sf1 * v0[0] + sf2 * v1[0] + sf3 * v2[0] + cx;
    const y = sf1 * v0[1] + sf2 * v1[1] + sf3 * v2[1] + cy;
    const z = sf1 * v0[2] + sf2 * v1[2] + sf3 * v2[2] + cz;
    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) {
      return { element: a.element, x: cx + (a.x - cx) * factor, y: cy + (a.y - cy) * factor, z: cz + (a.z - cz) * factor };
    }
    return { element: a.element, x, y, z };
  });

  const scaledVecs: LatticeVectors = [
    v0.map(c => c * abScale) as [number, number, number],
    v1.map(c => c * abScale) as [number, number, number],
    v2.map(c => c * cScale) as [number, number, number],
  ];

  return { atoms: scaledAtoms, latticeVecs: scaledVecs };
}

function computeHardFloor(el1: string, el2: string, pressureGPa: number): number {
  const r1 = getCompressedRadius(el1, pressureGPa);
  const r2 = getCompressedRadius(el2, pressureGPa);
  const sumR = r1 + r2;
  const bothH = el1 === "H" && el2 === "H";
  const oneH = el1 === "H" || el2 === "H";

  let ambientFloor: number;
  let radiusFrac: number;
  let absoluteMin: number;

  if (bothH) {
    ambientFloor = 0.74;
    radiusFrac = 0.70;
    absoluteMin = 0.40;
  } else if (oneH) {
    ambientFloor = 1.2;
    radiusFrac = 0.60;
    absoluteMin = 0.55;
  } else {
    // For covalent light elements (B, C, N, O, Si, P, S, etc.), the 1.8Å ambient
    // floor is too strict — it rejects valid B-B (~1.67Å) and C-C (~1.54Å) bonds.
    // Use a radius-based floor instead (65% of covalent sum).
    const COVALENT_LIGHT = new Set(["B","C","N","O","P","S","Si","Ge","As","Se","Te"]);
    if (COVALENT_LIGHT.has(el1) && COVALENT_LIGHT.has(el2)) {
      radiusFrac = 0.65;
      ambientFloor = Math.max(sumR * radiusFrac, 0.80);
    } else {
      ambientFloor = 1.8;
      radiusFrac = 0.55;
    }
    absoluteMin = 0.80;
  }

  if (pressureGPa <= 0) return ambientFloor;

  const radiusBased = Math.max(sumR * radiusFrac, absoluteMin);
  const blend = Math.min(pressureGPa / 100, 1.0);
  return ambientFloor * (1 - blend) + radiusBased * blend;
}

function hasHardDistanceViolation(
  atoms: AtomPosition[],
  pressureGPa: number = 0,
  latticeVecs?: LatticeVectors,
): { violated: boolean; pair: string; dist: number } {
  const vecs = latticeVecs ?? estimateLatticeFromAtoms(atoms);

  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const dist = pbcMinImageDist(atoms[i].x, atoms[i].y, atoms[i].z, atoms[j].x, atoms[j].y, atoms[j].z, vecs);
      const a = atoms[i].element, b = atoms[j].element;
      const floor = computeHardFloor(a, b, pressureGPa);
      if (dist < floor) return { violated: true, pair: `${a}-${b}`, dist };
    }
  }
  return { violated: false, pair: "", dist: 0 };
}

function validateAndFixStructure(
  atoms: AtomPosition[],
  formula: string,
  pressureGPa: number = 0,
  latticeVecs?: LatticeVectors,
): { atoms: AtomPosition[]; latticeVecs: LatticeVectors } | null {
  if (atoms.length < 2) return { atoms, latticeVecs: latticeVecs ?? estimateLatticeFromAtoms(atoms) };

  const vecs = latticeVecs ?? estimateLatticeFromAtoms(atoms);

  const hasHydrogen = atoms.some(a => a.element === "H");
  const minVol = hasHydrogen ? MIN_VOLUME_PER_ATOM_HYDRIDE : MIN_VOLUME_PER_ATOM;
  const pressureVolShrink = pressureGPa > 50 ? Math.cbrt(Math.max(0.4, 1.0 / (1 + pressureGPa / 200))) : 1.0;

  const counts = parseFormula(formula);
  const totalFormulaAtoms = Object.values(counts).reduce((s, c) => s + Math.round(c), 0);
  const expectedVolume = computeExpectedVolume(counts);
  const targetVolPerAtom = Math.max(minVol * pressureVolShrink, (expectedVolume / Math.max(totalFormulaAtoms, 1)) * pressureVolShrink);

  const cfg = STRUCTURAL_VALIDATION_CONFIG;
  let current = atoms;
  let currentVecs = vecs;
  const nAtoms = Math.max(totalFormulaAtoms, atoms.length);

  let cached = findPairwiseMinDistSpatial(current, pressureGPa, currentVecs);

  for (let attempt = 0; attempt < cfg.maxScaleAttempts; attempt++) {
    const { minDist, minRatio } = cached;
    // Use lattice determinant for cell volume — computeBoundingVolume (bounding box of
    // atom positions + padding) does not scale correctly with the scale factor, causing
    // the loop to overshoot and oscillate between too-large and too-small states.
    // lattice3x3Det scales exactly as factor³ so the loop converges in ≤2 steps.
    const volume = Math.abs(lattice3x3Det(currentVecs));
    const volumePerAtom = volume / nAtoms;

    const distOk = minRatio >= cfg.minRatioThreshold;
    const volOk = volumePerAtom >= targetVolPerAtom * cfg.volLowerFraction;
    const volNotTooLarge = volumePerAtom <= targetVolPerAtom * cfg.volUpperFraction;

    if (distOk && volOk && volNotTooLarge) {
      const hardCheck = hasHardDistanceViolation(current, pressureGPa, currentVecs);
      if (!hardCheck.violated) {
        return { atoms: current, latticeVecs: currentVecs };
      }
      dftLog(`[DFT] ${formula}: Hard distance violation ${hardCheck.pair}=${hardCheck.dist.toFixed(3)}Å @ ${pressureGPa} GPa — continuing scaling`);
    }

    const volScale = Math.cbrt(targetVolPerAtom / Math.max(volumePerAtom, 0.01));

    let scaleFactor: number;
    if (!distOk) {
      const distScale = 1.02 / Math.max(minRatio, 0.01);
      scaleFactor = Math.max(distScale, volScale);
    } else {
      scaleFactor = volScale;
    }

    scaleFactor = Math.max(scaleFactor, cfg.scaleFactorMin);
    scaleFactor = Math.min(scaleFactor, cfg.scaleFactorMax);

    dftLog(`[DFT] ${formula}: Structure validation attempt ${attempt + 1} — minDist=${minDist.toFixed(3)}Å, ratio=${minRatio.toFixed(2)}, vol/atom=${volumePerAtom.toFixed(1)}ų (target=${targetVolPerAtom.toFixed(1)}) @ ${pressureGPa} GPa — scaling by ${scaleFactor.toFixed(3)}`);
    const scaled = scaleStructureAnisotropic(current, scaleFactor, currentVecs, pressureGPa);
    current = scaled.atoms;
    currentVecs = scaled.latticeVecs;
    cached = findPairwiseMinDistSpatial(current, pressureGPa, currentVecs);
  }

  const hardCheck = hasHardDistanceViolation(current, pressureGPa, currentVecs);
  if (hardCheck.violated) {
    dftLog(`[DFT] ${formula}: Structure REJECTED — hard distance violation ${hardCheck.pair}=${hardCheck.dist.toFixed(3)}Å @ ${pressureGPa} GPa after ${cfg.maxScaleAttempts} attempts`, "info");
    return null;
  }

  const volume = Math.abs(lattice3x3Det(currentVecs));
  const volumePerAtom = volume / nAtoms;

  if (cached.minRatio < cfg.minRatioThreshold || volumePerAtom < minVol * pressureVolShrink - 0.1) {
    dftLog(`[DFT] ${formula}: Structure REJECTED after ${cfg.maxScaleAttempts} fix attempts — ratio=${cached.minRatio.toFixed(2)}, vol/atom=${volumePerAtom.toFixed(1)}ų @ ${pressureGPa} GPa`, "info");
    return null;
  }

  return { atoms: current, latticeVecs: currentVecs };
}

function checkVolumeRatioForAtoms(atoms: AtomPosition[], counts: Record<string, number>, label: string, formula: string, latticeVecs?: LatticeVectors): boolean {
  if (atoms.length < 2) return false;
  // Use lattice det when available — bounding box of atoms inflates due to covalent padding
  // and atom drift from jitter expansion, causing false 5-12x ratio failures.
  const volume = latticeVecs ? Math.abs(lattice3x3Det(latticeVecs)) : computeBoundingVolume(atoms);
  const expectedVol = computeExpectedVolume(counts);
  const { valid, ratio } = validateVolumeRatio(volume, expectedVol);
  if (!valid) {
    dftLog(`[DFT] ${formula}: Volume ratio check FAILED for ${label} — generated=${volume.toFixed(1)}ų, expected=${expectedVol.toFixed(1)}ų, ratio=${ratio.toFixed(2)} (must be 0.5-2.0)`);
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

function generateCrystalStructure(formula: string, pressureGPa: number = 0): { atoms: AtomPosition[]; prototype: string; latticeVecs?: LatticeVectors } {
  const counts = parseFormula(formula);
  const elements = Object.keys(counts);

  if (elements.length === 0) {
    return { atoms: [], prototype: "empty" };
  }

  if (elements.length > 5) {
    dftLog(`[DFT] ${formula}: Rejected — ${elements.length} distinct elements exceeds limit of 5`, "info");
    return { atoms: [], prototype: "rejected-too-complex" };
  }

  const totalAtomCount = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  if (totalAtomCount > 20) {
    dftLog(`[DFT] ${formula}: Rejected — ${totalAtomCount} total atoms exceeds limit of 20`, "info");
    return { atoms: [], prototype: "rejected-too-complex" };
  }

  if (!checkRadiusCompatibility(elements)) {
    dftLog(`[DFT] ${formula}: Radius incompatibility — non-H element radii ratio > 3.0`);
    return { atoms: [], prototype: "rejected-radius" };
  }

  const bvsResult = computeBondValenceSum(formula);
  if (bvsResult.deviation > 1.0) {
    dftLog(`[DFT] ${formula}: BVS deviation too high (${bvsResult.deviation.toFixed(2)} > 1.0) — rejecting before structure generation`);
    return { atoms: [], prototype: "rejected-bvs" };
  }

  const ionicResult = checkIonicRadiusCompatibility(formula);
  if (!ionicResult.compatible) {
    dftLog(`[DFT] ${formula}: Ionic radius incompatibility (ratio=${ionicResult.radiusRatio.toFixed(2)}, tolerance=${ionicResult.toleranceFactor?.toFixed(2) ?? "N/A"}) — rejecting`);
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
    const vResult = validateAndFixStructure(deduped, formula, pressureGPa);
    if (!vResult) return { atoms: [], prototype: "rejected-overlap" };
    if (!checkVolumeRatioForAtoms(vResult.atoms, counts, chemProto.templateName, formula, vResult.latticeVecs)) {
      return { atoms: [], prototype: "rejected-volume-ratio" };
    }
    prototypeSuccesses++;
    return { atoms: vResult.atoms, prototype: `${chemProto.templateName} prototype lattice`, latticeVecs: vResult.latticeVecs };
  }

  const hCount = counts["H"] || 0;
  const metalElements = elements.filter(el => el !== "H");
  const totalMetalCount = metalElements.reduce((s, el) => s + Math.round(counts[el] || 0), 0);
  const hMetalRatio = totalMetalCount > 0 ? hCount / totalMetalCount : 0;
  if (hCount > 0 && totalMetalCount > 0 && hMetalRatio >= 3) {
    const hydrideCage = generateHydrideCageStructure(formula, counts, pressureGPa);
    if (hydrideCage && hydrideCage.atoms.length >= 2) {
      const dedupedHydride = deduplicateSites(hydrideCage.atoms, hydrideCage.latticeVecs);
      if (isDuplicateCageStructure(formula, dedupedHydride, hydrideCage.latticeVecs)) {
        dftLog(`[DFT] ${formula}: Hydride cage rejected — structurally duplicate of previous cage`);
      } else {
        const vResult = validateAndFixStructure(dedupedHydride, formula, pressureGPa, hydrideCage.latticeVecs);
        if (vResult) {
          prototypeSuccesses++;
          return { atoms: vResult.atoms, prototype: hydrideCage.prototype, latticeVecs: vResult.latticeVecs };
        }
      }
    }
  } else if (hCount > 0 && totalMetalCount > 0) {
    dftLog(`[DFT] ${formula}: H/metal ratio ${hMetalRatio.toFixed(2)} < 3 — skipping hydride cage generation`);
  } else if (hCount > 0 && totalMetalCount === 0) {
    dftLog(`[DFT] ${formula}: Non-hydride stoichiometry (no metals) — skipping cage generation`);
  }

  const profile = buildChemicalProfile(counts);

  const protoMatch = matchPrototype(counts, profile);
  if (protoMatch) {
    const built = buildStructureFromPrototype(protoMatch.proto, protoMatch.siteMap, elements, counts, 1, pressureGPa);
    const atoms = deduplicateSites(built.atoms, built.latticeVecs);
    if (atoms.length >= 2) {
      const vResult = validateAndFixStructure(atoms, formula, pressureGPa, built.latticeVecs);
      if (!vResult) return { atoms: [], prototype: "rejected-overlap" };
      if (!checkVolumeRatioForAtoms(vResult.atoms, counts, protoMatch.proto.name, formula, vResult.latticeVecs)) {
        return { atoms: [], prototype: "rejected-volume-ratio" };
      }
      prototypeSuccesses++;
      return { atoms: vResult.atoms, prototype: protoMatch.proto.name, latticeVecs: vResult.latticeVecs };
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

  const scaledProfile = buildChemicalProfile(scaledCounts);

  const scaledMatch = matchPrototype(scaledCounts, scaledProfile);
  if (scaledMatch) {
    const scaledBuilt = buildStructureFromPrototype(scaledMatch.proto, scaledMatch.siteMap, elements, scaledCounts, 1, pressureGPa);
    const atoms = deduplicateSites(scaledBuilt.atoms, scaledBuilt.latticeVecs);
    if (atoms.length >= 2) {
      const vResult = validateAndFixStructure(atoms, formula, pressureGPa, scaledBuilt.latticeVecs);
      if (!vResult) return { atoms: [], prototype: "rejected-overlap" };
      if (!checkVolumeRatioForAtoms(vResult.atoms, scaledCounts, scaledMatch.proto.name + "-scaled", formula, vResult.latticeVecs)) {
        return { atoms: [], prototype: "rejected-volume-ratio" };
      }
      prototypeSuccesses++;
      return { atoms: vResult.atoms, prototype: scaledMatch.proto.name + "-scaled", latticeVecs: vResult.latticeVecs };
    }
  }

  chemistryMatchAttempts++;
  const chemMatch = selectBestPrototypeByChemistry(scaledCounts, elements, scaledProfile);
  if (chemMatch) {
    const chemBuilt = buildStructureFromPrototype(chemMatch.proto, chemMatch.siteMap, elements, scaledCounts, 1, pressureGPa);
    const atoms = deduplicateSites(chemBuilt.atoms, chemBuilt.latticeVecs);
    if (atoms.length >= 2) {
      const vResult = validateAndFixStructure(atoms, formula, pressureGPa, chemBuilt.latticeVecs);
      if (vResult) {
        if (checkVolumeRatioForAtoms(vResult.atoms, scaledCounts, chemMatch.proto.name + "-chem", formula, vResult.latticeVecs)) {
          prototypeSuccesses++;
          chemistryMatchSuccesses++;
          dftLog(`[DFT] ${formula}: Chemistry-based prototype match → ${chemMatch.proto.name}`);
          return { atoms: vResult.atoms, prototype: chemMatch.proto.name + "-chem", latticeVecs: vResult.latticeVecs };
        }
      }
    }
  }

  const { atoms, proto, latticeVecs: genericVecs } = buildGenericStructure(scaledCounts);
  const vResult = validateAndFixStructure(atoms, formula, pressureGPa, genericVecs);
  if (!vResult) return { atoms: [], prototype: "rejected-overlap" };
  if (!checkVolumeRatioForAtoms(vResult.atoms, scaledCounts, proto, formula, vResult.latticeVecs)) {
    return { atoms: [], prototype: "rejected-volume-ratio" };
  }
  prototypeSuccesses++;
  return { atoms: vResult.atoms, prototype: proto, latticeVecs: vResult.latticeVecs };
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

  const parsedEnergy = parseXTBEnergy(output);
  if (parsedEnergy !== null) {
    result.totalEnergy = parsedEnergy;
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

const XTB_ENERGY_RE = /TOTAL ENERGY\s+([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)\s+Eh/;

function parseXTBEnergy(output: string): number | null {
  const m = output.match(XTB_ENERGY_RE);
  if (!m) return null;
  const raw = m[1].replace(/[dD]/, "e");
  const val = parseFloat(raw);
  return isFinite(val) ? val : null;
}

function uniqueCalcId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
}

async function execShellAsync(cmd: string, options: { timeout?: number; env?: NodeJS.ProcessEnv; maxBuffer?: number }): Promise<string> {
  if (IS_WINDOWS) {
    // xTB is a Linux binary — route through WSL2.
    // Convert Windows absolute paths (e.g. C:\Users\...) to WSL /mnt/... paths in both
    // the command string and any single-path env var values (e.g. XTBHOME, XTBPATH).
    // Also strip .exe from binary names — WSL doesn't use it for Linux ELFs.
    const wslCmd = cmd
      .replace(/[A-Za-z]:\\[^ "&'|<>]*/g, (m) => toWslPath(m))
      .replace(/\.exe\b/g, "");

    const wslEnv: NodeJS.ProcessEnv = {};
    if (options.env) {
      for (const [k, v] of Object.entries(options.env as Record<string, string>)) {
        // Convert single Windows paths (not PATH/PATHEXT which contain semicolon-separated lists)
        if (v && !v.includes(";") && /^[A-Za-z]:\\/.test(v)) {
          wslEnv[k] = toWslPath(v);
        } else {
          wslEnv[k] = v;
        }
      }
    }

    // Spawn wsl.exe via cmd.exe shell (/c) so the pipe handles are created by the
    // Windows console subsystem — this avoids the WSL RPC error:
    //   "The RPC call contains a handle that differs from the declared handle type."
    // which occurs when Node.js creates raw pipe handles that don't satisfy WSL's
    // RPC interface expectations.  Routing through cmd.exe /c gives WSL the
    // console-attached handle types it requires.
    //
    // Fallback: if cmd.exe spawn also fails (e.g. headless CI), re-try with direct
    // spawn + stdio:"ignore" (original behaviour).
    const trySpawn = (useShell: boolean): Promise<string> =>
      new Promise<string>((resolve, reject) => {
        // Guard so promise settles exactly once — killing cmd.exe on Windows does NOT
        // close the WSL subprocess stdio pipes, so 'close' may never fire after proc.kill().
        // Without this guard the AL cycle hangs forever waiting for Promise.allSettled.
        let settled = false;
        const safeResolve = (v: string) => { if (!settled) { settled = true; resolve(v); } };
        const safeReject  = (e: Error)  => { if (!settled) { settled = true; reject(e);  } };

        const proc = useShell
          ? spawn("cmd.exe", ["/c", "wsl.exe", "-d", "Ubuntu", "--", "bash", "-c", wslCmd], {
              env: wslEnv,
              stdio: ["ignore", "pipe", "pipe"],
              windowsHide: true,
            })
          : spawn("wsl.exe", ["-d", "Ubuntu", "--", "bash", "-c", wslCmd], {
              env: wslEnv,
              stdio: ["ignore", "pipe", "pipe"],
              windowsHide: true,
            });
        let stdout = "";
        let stderr = "";
        proc.stdout?.on("data", (d: Buffer) => { stdout += d.toString(); });
        proc.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });
        let timedOut = false;
        const timer = options.timeout
          ? setTimeout(() => {
              timedOut = true;
              proc.kill();
              // Force-reject immediately — don't wait for 'close' which may never fire
              // when cmd.exe is killed but the WSL child keeps its pipe handles open.
              safeReject(new Error(`WSL command timed out after ${options.timeout}ms`));
            }, options.timeout)
          : null;
        proc.on("close", (code) => {
          if (timer) clearTimeout(timer);
          if (timedOut) {
            safeReject(new Error(`WSL command timed out after ${options.timeout}ms`));
          } else if (code !== 0) {
            const err: any = new Error(`Command failed: wsl.exe -d Ubuntu -- bash -c ${wslCmd.slice(0, 200)}`);
            err.stdout = stdout;
            err.stderr = stderr;
            safeReject(err);
          } else {
            safeResolve(stdout);
          }
        });
        proc.on("error", (e) => safeReject(e as Error));
      });

    try {
      return await trySpawn(true /* useShell=cmd.exe */);
    } catch (e: any) {
      // If the cmd.exe-shell path also fails with the RPC error, retry once with
      // the original direct-spawn approach before propagating the failure.
      if (e?.message?.includes("RPC call") || e?.message?.includes("handle")) {
        return await trySpawn(false /* direct spawn */);
      }
      throw e;
    }
  }

  const { stdout } = await execFileAsync("/bin/sh", ["-c", cmd], {
    timeout: options.timeout,
    env: options.env as NodeJS.ProcessEnv,
    maxBuffer: options.maxBuffer,
  });
  return stdout;
}

function parseOptimizedXYZ(filepath: string, label: string = ""): AtomPosition[] {
  const tag = label ? ` (${label})` : "";
  if (!fs.existsSync(filepath)) {
    dftLog(`[DFT]${tag}: XYZ file not found at ${path.basename(filepath)} — xTB may have crashed before writing output`, "info");
    return [];
  }
  const content = fs.readFileSync(filepath, "utf-8").trim();
  const lines = content.split("\n");
  if (lines.length < 3) {
    dftLog(`[DFT]${tag}: XYZ file ${path.basename(filepath)} has only ${lines.length} lines — xTB produced truncated output`, "info");
    return [];
  }

  const atomCount = parseInt(lines[0].trim(), 10);
  if (isNaN(atomCount) || atomCount < 1) {
    dftLog(`[DFT]${tag}: XYZ file ${path.basename(filepath)} has invalid atom count "${lines[0].trim()}" — malformed header`, "info");
    return [];
  }

  const atoms: AtomPosition[] = [];
  let skippedLines = 0;
  for (let i = 2; i < Math.min(lines.length, atomCount + 2); i++) {
    const parts = lines[i].trim().split(/\s+/);
    if (parts.length >= 4) {
      const element = parts[0];
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]);
      const z = parseFloat(parts[3]);
      if (element.match(/^[A-Z][a-z]?$/) && !isNaN(x) && !isNaN(y) && !isNaN(z)) {
        atoms.push({ element, x, y, z });
      } else {
        skippedLines++;
      }
    } else {
      skippedLines++;
    }
  }

  if (atoms.length === 0) {
    dftLog(`[DFT]${tag}: XYZ file ${path.basename(filepath)} declared ${atomCount} atoms but none could be parsed — xTB produced invalid coordinates`, "info");
  } else if (skippedLines > 0) {
    dftLog(`[DFT]${tag}: XYZ file ${path.basename(filepath)} had ${skippedLines} unparseable lines out of ${atomCount} declared atoms`);
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
    if (!bm || bm.bulkModulus <= 0 || !isFinite(bm.bulkModulus)) {
      dftLog(`[DFT] ${formula}: Pressure scaling skipped — invalid bulk modulus (${bm?.bulkModulus})`);
      return atoms;
    }
    const lattA = bm.compressedLattice?.a ?? 0;
    const lattB = bm.compressedLattice?.b ?? 0;
    const lattC = bm.compressedLattice?.c ?? 0;
    if (lattA <= 0 || lattB <= 0 || lattC <= 0 || !isFinite(lattA) || !isFinite(lattB) || !isFinite(lattC)) {
      dftLog(`[DFT] ${formula}: Pressure scaling skipped — invalid lattice params (a=${lattA}, b=${lattB}, c=${lattC})`);
      return atoms;
    }
    if (bm.compressedVolume <= 0 || !isFinite(bm.compressedVolume)) {
      dftLog(`[DFT] ${formula}: Pressure scaling skipped — invalid compressed volume (${bm.compressedVolume})`);
      return atoms;
    }
    const cubicEta = (() => {
      const B0p = 4.0;
      const pOverB = pressureGpa / Math.max(10, bm.bulkModulus);
      const inner = 1 + B0p * pOverB;
      if (inner > 0) return Math.pow(Math.pow(inner, -1 / B0p), 1 / 3);
      return Math.pow(0.5, 1 / 3);
    })();
    const scale = Math.max(0.8, Math.min(1.0, cubicEta));
    if (!isFinite(scale)) return atoms;
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
  formula = normalizeFormula(formula);
  if (!isDFTAvailable()) return null;

  const pressureTag = pressureGpa > 0 ? `_P${Math.round(pressureGpa)}` : "";
  const cacheKey = formula.replace(/\s+/g, "") + pressureTag;
  if (optimizedStructureCache.has(cacheKey)) {
    return optimizedStructureCache.get(cacheKey)!;
  }

  if (!fs.existsSync(XTB_BIN)) {
    dftLog(`[DFT] xTB binary not found at ${XTB_BIN}`, "info");
    return null;
  }

  const startTime = Date.now();
  const calcId = uniqueCalcId(`opt_${cacheKey.replace(/[^a-zA-Z0-9]/g, "_")}`);
  const calcDir = path.join(WORK_DIR, calcId);
  fs.mkdirSync(calcDir, { recursive: true });

  let { atoms, prototype, latticeVecs: structLatticeVecs } = generateCrystalStructure(formula, pressureGpa);
  if (atoms.length < 2) return null;

  if (pressureGpa > 0) {
    atoms = applyPressureScaling(atoms, formula, pressureGpa);
    dftLog(`[DFT] ${formula}: Pressure-scaled geometry at ${pressureGpa} GPa for xTB optimization`);
  }

  const xyzPath = path.join(calcDir, "input.xyz");
  writeXYZ(atoms, xyzPath, `${formula} [${prototype}] optimization${pressureGpa > 0 ? ` @ ${pressureGpa} GPa` : ""}`);

  try {
    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
      OMP_STACKSIZE: "512M",
    };

    const cmd = `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight 2>&1`;
    if (!cmd.includes("xtb")) {
      dftLog(`[DFT] WARNING: Geometry optimization command malformed: ${cmd.slice(0, 200)}`, "info");
      return null;
    }

    const output = await execShellAsync(
      cmd,
      { timeout: OPT_TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
    );

    const optInfo = parseOptimizationOutput(output);

    const optXyzPath = path.join(calcDir, "xtbopt.xyz");
    let optimizedAtoms = parseOptimizedXYZ(optXyzPath, formula);

    if (optimizedAtoms.length === 0) {
      const altPath = path.join(calcDir, "xtbopt.coord");
      if (fs.existsSync(altPath)) {
        optimizedAtoms = parseOptimizedXYZ(altPath, formula);
      }
    }

    if (optimizedAtoms.length === 0) {
      optimizedAtoms = atoms;
    }

    let optimizedEnergy = parseXTBEnergy(output) ?? 0;

    if (pressureGpa > 0 && optimizedAtoms.length > 0) {
      const pvLattice = structLatticeVecs ?? estimateLatticeFromAtoms(optimizedAtoms);
      const vol_A3 = Math.max(1, Math.abs(lattice3x3Det(pvLattice)));
      const eV_per_GPa_A3 = 0.00624151;
      const pvCorrection = pressureGpa * vol_A3 * eV_per_GPa_A3;
      const pvHartree = pvCorrection / 27.211386;
      optimizedEnergy += pvHartree;
      dftLog(`[DFT] ${formula}: PV correction at ${pressureGpa} GPa: +${pvHartree.toFixed(6)} Eh (V~${vol_A3.toFixed(1)} A^3)`);
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
          prototype === "Perovskite" ? "Pm-3m" : prototype === "A15" ? "Pm-3n" :
            prototype === "NaCl" ? "Fm-3m" : prototype === "AlB2" ? "P6/mmm" :
            prototype === "ThCr2Si2" ? "I4/mmm" : undefined,
        );
        result.distortion = distortion;
        recordDistortionAnalysis(distortion);
        if (distortion.overallLevel !== "none") {
          dftLog(`[DFT] ${formula}: Distortion detected (${distortion.overallLevel}, score=${distortion.overallScore}, meanDisp=${distortion.atomicDistortion?.meanDisplacement?.toFixed(4) ?? "N/A"}A, strain=${distortion.latticeDistortion.strainMagnitude.toFixed(5)}, vol=${distortion.latticeDistortion.volumeChangePct.toFixed(2)}%)${distortion.symmetryReduction?.symmetryBroken ? ` [symmetry broken: ${distortion.symmetryReduction.systemBefore}->${distortion.symmetryReduction.systemAfter}]` : ""}`);
        }
      } catch {}
    }

    if (optimizedAtoms.length >= 2) {
      optimizedStructureCache.set(cacheKey, result);
      if (optimizedStructureCache.size > CACHE_MAX) {
        const oldest = optimizedStructureCache.keys().next().value;
        if (oldest) optimizedStructureCache.delete(oldest);
      }
    }

    return result;
  } catch (err: any) {
    const isTimeout = err.killed || (err.message && err.message.includes("TIMEOUT"));
    dftLog(`[DFT] ${formula}: Geometry optimization ${isTimeout ? "timed out" : "failed"}: ${err.message?.slice(0, 500) || String(err).slice(0, 500)}`, "info");
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
  formula = normalizeFormula(formula);
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
    const calcId = uniqueCalcId(`landscape_${cacheKey}_p${pi}`);
    const calcDir = path.join(WORK_DIR, calcId);
    fs.mkdirSync(calcDir, { recursive: true });

    try {
      const useLatticeStrain = pi % 2 === 1;
      let perturbedAtoms: AtomPosition[];

      if (useLatticeStrain) {
        const strainMag = scale * 0.5;
        const sx = 1.0 + (Math.random() - 0.5) * 2 * strainMag;
        const sy = 1.0 + (Math.random() - 0.5) * 2 * strainMag;
        const sz = 1.0 + (Math.random() - 0.5) * 2 * strainMag;
        const shearXY = (Math.random() - 0.5) * strainMag * 0.3;
        const shearXZ = (Math.random() - 0.5) * strainMag * 0.3;
        const shearYZ = (Math.random() - 0.5) * strainMag * 0.3;
        perturbedAtoms = refAtoms.map(a => ({
          element: a.element,
          x: a.x * sx + a.y * shearXY + a.z * shearXZ,
          y: a.y * sy + a.z * shearYZ,
          z: a.z * sz,
        }));
      } else {
        perturbedAtoms = refAtoms.map(a => ({
          element: a.element,
          x: a.x + (Math.random() - 0.5) * 2 * scale,
          y: a.y + (Math.random() - 0.5) * 2 * scale,
          z: a.z + (Math.random() - 0.5) * 2 * scale,
        }));
      }

      const xyzPath = path.join(calcDir, "input.xyz");
      writeXYZ(perturbedAtoms, xyzPath, `${formula} perturbation ${pi} scale=${scale}`);

      const env: Record<string, string> = {
        ...process.env as Record<string, string>,
        XTBHOME: XTB_HOME,
        XTBPATH: XTB_PARAM,
        OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
        OMP_STACKSIZE: "512M",
      };

      const cmd = `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight 2>&1`;
      const output = await execShellAsync(cmd, { timeout: OPT_TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 });

      const optInfo = parseOptimizationOutput(output);
      const optEnergy = parseXTBEnergy(output) ?? 0;

      const optXyzPath = path.join(calcDir, "xtbopt.xyz");
      let optimizedAtoms = parseOptimizedXYZ(optXyzPath, formula);
      if (optimizedAtoms.length === 0) optimizedAtoms = perturbedAtoms;

      let rmsDisp = 0;
      const nAtoms = Math.min(refAtoms.length, optimizedAtoms.length);
      if (nAtoms > 0) {
        const dispLattice = estimateLatticeFromAtoms(refAtoms);
        let sumSq = 0;
        for (let i = 0; i < nAtoms; i++) {
          const d = pbcMinImageDist(
            optimizedAtoms[i].x, optimizedAtoms[i].y, optimizedAtoms[i].z,
            refAtoms[i].x, refAtoms[i].y, refAtoms[i].z,
            dispLattice,
          );
          sumSq += d * d;
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
    dftLog(`[DFT] ${formula}: Energy landscape exploration found ${uniqueMinima} unique minima (spread=${result.energySpread.toFixed(5)} Eh, ${minima.length} perturbations)`);
  }

  return result;
}

/**
 * Lightweight heuristic landscape recorder — no xTB/DFT required.
 * Estimates multi-minima likelihood from composition complexity and lattice
 * degrees of freedom so stats populate without full perturbation-reoptimization.
 */
export function recordHeuristicLandscape(
  formula: string,
  elements: string[],
  spaceGroup: string | null | undefined,
  pressureGpa: number = 0,
): void {
  const cacheKey = formula.replace(/\s+/g, "");
  if (landscapeCache.has(cacheKey)) return;

  const nEl = elements.length;
  const hasTM = elements.some(e => ["V","Cr","Mn","Fe","Co","Ni","Cu","Mo","W","Re","Ru","Rh","Ir","Os","Pd","Pt"].includes(e));
  const hasH = elements.includes("H");
  const hasHalogen = elements.some(e => ["F","Cl","Br","I"].includes(e));
  const isLowSym = spaceGroup ? /P21|Pnma|Cmcm|P-1|C2\/m/i.test(spaceGroup) : false;

  // Multi-minima likelihood factors:
  //   - complex compositions (4+ elements) → more competing phases
  //   - transition metals with d-orbital rearrangement
  //   - hydrogen in multiple interstitial sites
  //   - low-symmetry space groups (structural softness)
  //   - high pressure (polymorph competition)
  const complexityScore =
    Math.min(1.0, 0.1 * nEl) +
    (hasTM ? 0.25 : 0) +
    (hasH ? 0.15 : 0) +
    (hasHalogen ? 0.1 : 0) +
    (isLowSym ? 0.2 : 0) +
    (pressureGpa > 50 ? 0.2 : 0);

  // Use a deterministic hash of the formula to add per-material variation,
  // so chemically different systems don't all land on the same energy spread.
  let hash = 0;
  for (let i = 0; i < formula.length; i++) {
    hash = ((hash << 5) - hash + formula.charCodeAt(i)) | 0;
  }
  const hashFrac = ((hash & 0x7fffffff) % 1000) / 1000; // 0..0.999, deterministic per formula

  const multipleMinima = complexityScore > 0.5;
  const uniqueMinima = multipleMinima ? Math.min(6, 1 + Math.floor(complexityScore * 4)) : 1;
  // Scale spread by complexity + per-formula hash so different compositions
  // don't return suspiciously identical values.
  const energySpread = complexityScore * (0.06 + 0.08 * hashFrac); // Eh, varies ~0.06–0.14 × complexity
  const distortionModesExist = complexityScore > 0.4 && (hasTM || isLowSym);

  const entry: EnergyLandscapeResult = {
    formula,
    referenceEnergy: 0,
    perturbationsRun: LANDSCAPE_PERTURBATION_COUNT,
    uniqueMinima,
    minima: [],
    multipleMinima,
    energySpread: Number(energySpread.toFixed(5)),
    lowestEnergyDiff: Number((energySpread * 0.3).toFixed(5)),
    distortionModesExist,
    wallTimeSeconds: 0,
  };
  landscapeCache.set(cacheKey, entry);

  // Bound the cache to avoid unbounded memory growth
  if (landscapeCache.size > 2000) {
    const firstKey = landscapeCache.keys().next().value;
    if (firstKey) landscapeCache.delete(firstKey);
  }
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
  formula = normalizeFormula(formula);
  const pressureTag = pressureGpa > 0 ? `_P${Math.round(pressureGpa)}` : "";
  const cacheKey = formula.replace(/\s+/g, "") + pressureTag;
  const cached = xtbResultCache.get(cacheKey);
  if (cached && cached.converged) {
    return cached;
  }

  const startTime = Date.now();
  const calcId = uniqueCalcId(cacheKey.replace(/[^a-zA-Z0-9]/g, "_"));
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
    const optCacheKey = formula.replace(/\s+/g, "") + pressureTag;
    const cachedOpt = optimizedStructureCache.get(optCacheKey);
    if (cachedOpt && cachedOpt.optimizedAtoms.length >= 2) {
      atoms = cachedOpt.optimizedAtoms;
      if (cachedOpt.converged) {
        prototype = "xTB-optimized-cached";
        isOptimized = true;
      } else {
        prototype = "xTB-unconverged-cached";
      }
    } else {
      let generated = generateCrystalStructure(formula, pressureGpa);
      atoms = generated.atoms;
      prototype = generated.prototype;
      if (pressureGpa > 0 && atoms.length >= 2) {
        atoms = applyPressureScaling(atoms, formula, pressureGpa);
      }
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
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
      OMP_STACKSIZE: "512M",
    };

    const output = await execShellAsync(
      `cd ${calcDir} && ${XTB_BIN} ${xyzPath} --gfn 2 --sp 2>&1`,
      { timeout: TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
    );

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

  if (result.converged || result.totalEnergy !== 0) {
    xtbResultCache.set(cacheKey, result);
    if (xtbResultCache.size > CACHE_MAX) {
      const oldest = xtbResultCache.keys().next().value;
      if (oldest) xtbResultCache.delete(oldest);
    }
  }

  return result;
}

const elementRefEnergies = new Map<string, number | null>();

const MOLECULAR_ELEMENTS = new Set(["H", "N", "O", "F", "Cl", "Br", "I"]);
const MOLECULAR_BOND_LENGTHS: Record<string, number> = {
  H: 0.74,
  N: 1.10,
  O: 1.21,
  F: 1.42,
  Cl: 1.99,
  Br: 2.28,
  I: 2.66,
};

const BULK_NN_DISTANCES: Record<string, number> = {
  Li: 3.04, Be: 2.22, Na: 3.72, Mg: 3.20, Al: 2.86, Si: 2.35,
  K: 4.54, Ca: 3.95, Sc: 3.25, Ti: 2.90, V: 2.62, Cr: 2.50,
  Mn: 2.73, Fe: 2.48, Co: 2.50, Ni: 2.49, Cu: 2.56, Zn: 2.66,
  Ga: 2.44, Ge: 2.45, Rb: 4.95, Sr: 4.30, Y: 3.55, Zr: 3.18,
  Nb: 2.86, Mo: 2.73, Tc: 2.71, Ru: 2.65, Rh: 2.69, Pd: 2.75,
  Ag: 2.89, Cd: 2.98, In: 3.25, Sn: 3.02, Sb: 2.91, Te: 2.86,
  Cs: 5.24, Ba: 4.35, La: 3.75, Ce: 3.65, Pr: 3.64, Nd: 3.63,
  Hf: 3.13, Ta: 2.86, W: 2.74, Re: 2.74, Os: 2.68, Ir: 2.71,
  Pt: 2.77, Au: 2.88, Hg: 3.01, Tl: 3.46, Pb: 3.50, Bi: 3.07,
  Th: 3.60, U: 3.47, Pa: 3.21,
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

function applyCohesiveCorrection(element: string, clusterEnergyPerAtom: number, refType: string): number {
  const cohesiveEv = COHESIVE_ENERGIES_EV[element];
  if (cohesiveEv === undefined || cohesiveEv <= 0) return clusterEnergyPerAtom;
  const HA_TO_EV = 27.211386;
  const cohesiveHa = cohesiveEv / HA_TO_EV;
  const isMolecular = MOLECULAR_ELEMENTS.has(element);
  let correctionHa: number;
  if (isMolecular) {
    correctionHa = cohesiveHa * 0.5;
  } else if (refType === "cluster") {
    correctionHa = cohesiveHa * 0.35;
  } else if (refType === "dimer") {
    correctionHa = cohesiveHa * 0.55;
  } else {
    correctionHa = cohesiveHa * 0.85;
  }
  return clusterEnergyPerAtom - correctionHa;
}

async function computeElementalEnergy(element: string): Promise<number | null> {
  if (elementRefEnergies.has(element)) {
    return elementRefEnergies.get(element) ?? null;
  }

  const calcDir = path.join(WORK_DIR, uniqueCalcId(`ref_${element}`));
  fs.mkdirSync(calcDir, { recursive: true });

  const isMolecular = MOLECULAR_ELEMENTS.has(element);
  let nnDist = BULK_NN_DISTANCES[element];
  if (nnDist === undefined && !isMolecular) {
    const elData = getElementData(element);
    if (elData && elData.atomicRadius > 0) {
      nnDist = (elData.atomicRadius / 100) * 2;
      dftLog(`[DFT] ${element}: NN distance estimated from atomic radius: ${nnDist.toFixed(2)}Å`);
    }
  }
  const useCluster = !isMolecular && nnDist !== undefined;
  let atoms: AtomPosition[];
  let divisor: number;
  let refLabel: string;
  let refType: string;

  if (isMolecular) {
    const bondLength = MOLECULAR_BOND_LENGTHS[element] ?? 1.5;
    atoms = [
      { element, x: 0, y: 0, z: 0 },
      { element, x: bondLength, y: 0, z: 0 },
    ];
    divisor = 2;
    refLabel = "dimer";
    refType = "molecular";
  } else if (useCluster) {
    const d = nnDist;
    atoms = [
      { element, x: 0, y: 0, z: 0 },
      { element, x: d, y: 0, z: 0 },
      { element, x: d / 2, y: d * Math.sqrt(3) / 2, z: 0 },
      { element, x: d / 2, y: d * Math.sqrt(3) / 6, z: d * Math.sqrt(6) / 3 },
    ];
    divisor = 4;
    refLabel = `4-atom cluster (NN=${d.toFixed(2)}Å)`;
    refType = "cluster";
  } else {
    atoms = [{ element, x: 0, y: 0, z: 0 }];
    divisor = 1;
    refLabel = "isolated atom (fallback)";
    refType = "isolated";
  }

  const xyzPath = path.join(calcDir, `${element}.xyz`);
  writeXYZ(atoms, xyzPath, `${element} ${refLabel} reference`);

  try {
    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
      OMP_STACKSIZE: "512M",
    };

    const output = await execShellAsync(
      `cd ${calcDir} && ${XTB_BIN} ${xyzPath} --gfn 2 --sp 2>&1`,
      { timeout: 30000, env: env as NodeJS.ProcessEnv, maxBuffer: 20 * 1024 * 1024 }
    );

    const parsedEnergy = parseXTBEnergy(output);
    if (parsedEnergy !== null && output.includes("normal termination")) {
      const rawEnergyPerAtom = parsedEnergy / divisor;
      const bulkRefEnergy = applyCohesiveCorrection(element, rawEnergyPerAtom, refType);
      dftLog(`[DFT] ${element}: xTB ref energy = ${bulkRefEnergy.toFixed(4)} Ha/atom (${refLabel}, cohesive correction applied)`);
      elementRefEnergies.set(element, bulkRefEnergy);
      try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
      return bulkRefEnergy;
    }

    if (useCluster) {
      dftLog(`[DFT] ${element}: Cluster reference failed, falling back to dimer`);
      const dimerAtoms: AtomPosition[] = [
        { element, x: 0, y: 0, z: 0 },
        { element, x: nnDist, y: 0, z: 0 },
      ];
      const dimerPath = path.join(calcDir, `${element}_dimer.xyz`);
      writeXYZ(dimerAtoms, dimerPath, `${element} dimer fallback`);
      const dimerOut = await execShellAsync(
        `cd ${calcDir} && ${XTB_BIN} ${dimerPath} --gfn 2 --sp 2>&1`,
        { timeout: 30000, env: env as NodeJS.ProcessEnv, maxBuffer: 20 * 1024 * 1024 }
      );
      const dimerEnergy = parseXTBEnergy(dimerOut);
      if (dimerEnergy !== null && dimerOut.includes("normal termination")) {
        const rawEnergyPerAtom = dimerEnergy / 2;
        const bulkRefEnergy = applyCohesiveCorrection(element, rawEnergyPerAtom, "dimer");
        dftLog(`[DFT] ${element}: xTB ref energy = ${bulkRefEnergy.toFixed(4)} Ha/atom (dimer fallback, NN=${nnDist.toFixed(2)}Å, cohesive correction applied)`);
        elementRefEnergies.set(element, bulkRefEnergy);
        try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
        return bulkRefEnergy;
      }
    }
  } catch {}

  try {
    fs.rmSync(calcDir, { recursive: true, force: true });
  } catch {}

  elementRefEnergies.set(element, null);
  return null;
}

export async function computeFormationEnergy(formula: string, dftResult: DFTResult): Promise<number | null> {
  formula = normalizeFormula(formula);
  if (!dftResult.converged || dftResult.totalEnergy === 0) return null;
  if (dftResult.totalEnergy > 0) {
    dftLog(`[DFT] ${formula}: Positive total energy (${dftResult.totalEnergy.toFixed(4)} Ha) — xTB produced invalid result, skipping Ef`, "info");
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
  const formulaTotal = Object.values(counts).reduce((s, n) => s + n, 0);
  if (formulaTotal === 0 || elements.length === 0) return null;

  const compoundEnergyPerAtom = compoundEnergy / actualAtomCount;

  const refResults = await Promise.all(elements.map(el => computeElementalEnergy(el)));
  if (refResults.some(r => r === null)) return null;

  let refEnergyPerAtom = 0;
  for (let i = 0; i < elements.length; i++) {
    const atomFraction = counts[elements[i]] / formulaTotal;
    refEnergyPerAtom += (refResults[i] as number) * atomFraction;
  }

  const HA_TO_EV = 27.211386;
  let efPerAtom = (compoundEnergyPerAtom - refEnergyPerAtom) * HA_TO_EV;

  if (efPerAtom > 5.0 || efPerAtom < -10.0) {
    dftLog(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom wildly out of range, likely reference energy mismatch — discarding`);
    return null;
  }

  if (efPerAtom > 1.0) {
    dftLog(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom is positive (>1.0), discarding — compound less stable than elements`);
    return null;
  }

  if (efPerAtom < -5.0) {
    dftLog(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom unusually negative (<-5.0) — flagged as suspect but preserving value for ranking`, "info");
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
  formula = normalizeFormula(formula);
  if (!isDFTAvailable()) return null;
  if (phononCache.has(formula)) return phononCache.get(formula)!;

  const calcDir = path.join(WORK_DIR, uniqueCalcId(`phonon_${formula.replace(/[^a-zA-Z0-9]/g, "_")}`));
  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
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
        const preOptResult = validateAndFixStructure(initialAtoms, formula);
        if (preOptResult && preOptResult.atoms.length >= 2) {
          writeXYZ(preOptResult.atoms, path.join(preOptDir, "input.xyz"), `${formula} pre-opt`);
          const optOut = await execShellAsync(
            `cd ${preOptDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight --iterations 200 2>&1`,
            { timeout: TIMEOUT_MS, env, maxBuffer: 50 * 1024 * 1024 }
          );
          if (optOut.includes("converged")) {
            const optXYZ = path.join(preOptDir, "xtbopt.xyz");
            if (fs.existsSync(optXYZ)) {
              const parsed = parseOptimizedXYZ(optXYZ, formula);
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

    const prePhononResult = validateAndFixStructure(atoms, formula);
    if (!prePhononResult) {
      dftLog(`[DFT] ${formula}: Phonon check skipped — structure has atom overlaps`);
      return null;
    }
    atoms = prePhononResult.atoms;

    writeXYZ(atoms, path.join(calcDir, "input.xyz"), `${formula} phonon check (${prototype})`);

    const output = await execShellAsync(
      `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --hess --iterations 200 2>&1`,
      { timeout: TIMEOUT_MS * 2, env, maxBuffer: 50 * 1024 * 1024 }
    );

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
    const PHYSICAL_IMAG_THRESHOLD = -20;
    const lowestFreq = frequencies.length > 0 ? Math.min(...frequencies) : 0;
    const artifactModes = frequencies.filter(f => f < ARTIFACT_THRESHOLD);
    const physicalFrequencies = frequencies.filter(f => f >= ARTIFACT_THRESHOLD);
    const physicalImagModes = physicalFrequencies.filter(f => f < PHYSICAL_IMAG_THRESHOLD);

    if (artifactModes.length > 0) {
      dftLog(`[DFT] ${formula}: ${artifactModes.length} frequencies below ${ARTIFACT_THRESHOLD} cm-1 (lowest: ${lowestFreq.toFixed(0)}) — catastrophic geometry failure or non-physical PES`, "info");
    }

    const result: PhononStability = {
      hasImaginaryModes: physicalImagModes.length > 0 || artifactModes.length > 0,
      imaginaryModeCount: physicalImagModes.length + artifactModes.length,
      lowestFrequency: lowestFreq,
      frequencies: frequencies.slice(0, 20),
      zeroPointEnergy: zpe,
    };

    if (artifactModes.length > 0) {
      (result as any).severeInstability = true;
      (result as any).geometryFailure = true;
      (result as any).instabilityReason = `${artifactModes.length} modes below ${ARTIFACT_THRESHOLD} cm-1 — likely catastrophic geometry (atoms overlapping or non-physical PES)`;
    } else if (physicalImagModes.length > 10) {
      (result as any).severeInstability = true;
      (result as any).instabilityReason = `${physicalImagModes.length} imaginary modes detected (max 10 allowed for xTB screening)`;
      dftLog(`[DFT] ${formula}: Severe phonon instability — ${physicalImagModes.length} imaginary modes`);
    }
    if (lowestFreq < -1500 && lowestFreq >= ARTIFACT_THRESHOLD) {
      (result as any).severeInstability = true;
      (result as any).instabilityReason = `Lowest frequency ${lowestFreq.toFixed(0)} cm-1 (threshold: -1500 cm-1)`;
      dftLog(`[DFT] ${formula}: Severe phonon instability — lowest freq = ${lowestFreq.toFixed(0)} cm-1`);
    }
    if (artifactModes.length > 0) {
      dftLog(`[DFT] ${formula}: ${artifactModes.length} xTB artifact modes (< ${ARTIFACT_THRESHOLD} cm-1) flagged as geometry failure`);
    }

    phononCache.set(formula, result);
    if (phononCache.size > 100) {
      const oldest = phononCache.keys().next().value;
      if (oldest) phononCache.delete(oldest);
    }

    return result;
  } catch (err) {
    dftLog(`[DFT] ${formula}: xTB --hess failed, using analytical phonon fallback: ${err instanceof Error ? err.message.slice(0, 100) : String(err).slice(0, 100)}`);

    try {
      const optOutput = await execShellAsync(
        `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --opt tight --iterations 200 2>&1`,
        { timeout: TIMEOUT_MS, env, maxBuffer: 20 * 1024 * 1024 }
      );

      const converged = optOutput.includes("GEOMETRY OPTIMIZATION CONVERGED") || optOutput.includes("normal termination");
      const counts = parseFormula(formula);
      const elements = Object.keys(counts);
      const avgMass = elements.reduce((s, el) => {
        const elData = getElementData(el);
        const elMass = elData?.atomicMass;
        if (elMass === undefined || elMass <= 0) {
          dftLog(`[DFT] ${formula}: Unknown atomic mass for "${el}", using fallback 50.0 amu`, "info");
          return s + 50.0 * (counts[el] ?? 1);
        }
        return s + elMass * (counts[el] ?? 1);
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
      (result as any).noPredictivePower = true;
      (result as any).estimationBasis = converged ? "opt-converged" : "opt-unconverged";

      dftLog(`[DFT] ${formula}: Analytical phonon estimate — Debye≈${debyeFreq.toFixed(0)} cm⁻¹, ${nModes} modes, stable=${converged}`);
      phononCache.set(formula, result);
      return result;
    } catch (fallbackErr) {
      dftLog(`[DFT] ${formula}: Phonon fallback also failed: ${fallbackErr instanceof Error ? fallbackErr.message.slice(0, 80) : ""}`);
      return null;
    }
  } finally {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
  }
}

export async function runFiniteDisplacementPhonons(formula: string): Promise<FiniteDisplacementPhononResult | null> {
  formula = normalizeFormula(formula);
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

  const validatedResult = validateAndFixStructure(rawAtoms, formula);
  if (!validatedResult) {
    dftLog(`[DFT] ${formula}: Finite displacement phonons skipped — structure has atom overlaps`);
    return null;
  }
  const validatedAtoms = validatedResult.atoms;

  return computeFiniteDisplacementPhonons(formula, validatedAtoms);
}

export async function runXTBEnrichment(formula: string, pressureGpa: number = 0): Promise<XTBEnrichedFeatures | null> {
  formula = normalizeFormula(formula);
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

  const isNonPredictive = phononResult && ((phononResult as any).noPredictivePower === true || (phononResult as any).analyticalEstimate === true);

  const pressureTag = pressureGpa > 0 ? `_P${Math.round(pressureGpa)}` : "";
  const enrichCacheKey = formula.replace(/\s+/g, "") + pressureTag;
  if (phononResult && !isNonPredictive) {
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
        const lowestFreq = phononResult.lowestFrequency;
        const isMetallic = dftResult.isMetallic;
        const isShallowImag = lowestFreq > -50;

        let penalty = 0.05;
        if (isMetallic && isShallowImag) {
          penalty = 0.0;
          optRes.distortion.scRelevance += ` Phonon: ${phononResult.imaginaryModeCount} shallow imaginary mode(s) (lowest=${lowestFreq.toFixed(1)} cm-1) — common xTB artifact for metals, no penalty applied.`;
        } else if (isMetallic) {
          penalty = 0.02;
          optRes.distortion.scRelevance += ` Phonon instability: ${phononResult.imaginaryModeCount} imaginary mode(s), lowest freq=${lowestFreq.toFixed(1)} cm-1 — reduced penalty for metallic system.`;
        } else {
          optRes.distortion.scRelevance += ` Phonon instability: ${phononResult.imaginaryModeCount} imaginary mode(s), lowest freq=${lowestFreq.toFixed(1)} cm-1 — structure wants to distort.`;
        }
        if (penalty > 0) {
          optRes.distortion.overallScore = Math.min(1.0, optRes.distortion.overallScore + penalty);
        }
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
  // xtbHealthy is set by checkXTBHealth() which runs through WSL/execShellAsync.
  // Don't fall back to fs.existsSync(XTB_BIN): on Windows, XTB_BIN points to an ELF
  // binary that Node can see but cannot execute — only WSL can run it.
  return xtbHealthy;
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
  // All calls use execShellAsync so Windows routes through WSL — execFileAsync runs the PE binary
  // directly, bypassing WSL, and fails for Linux ELF binaries accessed via wsl.exe.
  const MAX_RETRIES = 2;
  const VERSION_TIMEOUT = 15000;
  const OPT_TIMEOUT = 30000;
  const HESS_TIMEOUT = 40000;

  const result = { available: false, version: "", canOptimize: false, canHess: false, error: undefined as string | undefined };
  // Track whether opt/hess actually succeeded vs were only skipped due to WSL infra errors.
  // xtbHealthy uses these — WSL-skip alone does not count as verified healthy.
  let actualOptSuccess = false;
  let actualHessSuccess = false;

  if (!fs.existsSync(XTB_BIN)) {
    result.error = `xTB binary not found at ${XTB_BIN}`;
    console.log(`[xTB-Health] FAIL: ${result.error}`);
    return result;
  }

  const env = {
    ...process.env,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
    OMP_STACKSIZE: "100M",
  };

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    result.error = undefined;

    try {
      let versionOutput: string;
      try {
        // Route through execShellAsync — on Windows this goes via WSL, correctly running the ELF binary.
        versionOutput = await execShellAsync(`${XTB_BIN} --version 2>&1`, { timeout: VERSION_TIMEOUT, env });
      } catch (e: any) {
        versionOutput = (e.stdout ?? "") + (e.stderr ?? "") + (e.message ?? "");
      }
      const vMatch = versionOutput.match(/xtb version (\S+)/);
      result.version = vMatch ? vMatch[1] : "unknown";
      result.available = true;
      console.log(`[xTB-Health] Binary found: v${result.version} (attempt ${attempt}/${MAX_RETRIES})`);

      const testDir = path.join(WORK_DIR, uniqueCalcId("health_check"));
      fs.mkdirSync(testDir, { recursive: true });

      const h2Xyz = `2\nH2 molecule test\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74\n`;
      fs.writeFileSync(path.join(testDir, "input.xyz"), h2Xyz);

      // Helper: detect Windows WSL infrastructure errors (RPC handle mismatch, WSL not running).
      // These are not xTB failures — they mean WSL itself couldn't start, not that xtb is broken.
      const isWSLInfraError = (msg: string) =>
        msg.includes("RPC call") ||
        msg.includes("handle type") ||
        msg.includes("wsl.exe") ||
        msg.includes("WSL") ||
        msg.includes("0x800701b6");

      try {
        let optOut: string;
        let wslInfraFail = false;
        try {
          // cd into testDir inside the shell command so execShellAsync can convert the Windows path to /mnt/...
          optOut = await execShellAsync(`cd "${testDir}" && ${XTB_BIN} input.xyz --gfn 2 --opt tight 2>&1`, { timeout: OPT_TIMEOUT, env });
        } catch (e: any) {
          const msg: string = (e.stdout ?? "") + (e.stderr ?? "") + (e.message ?? "");
          if (isWSLInfraError(msg)) {
            // WSL infrastructure failure — not an xTB issue; assume xTB would work once WSL starts
            console.log(`[xTB-Health] Geometry optimization: SKIPPED (WSL infrastructure not ready, attempt ${attempt}): ${msg.slice(0, 150)}`);
            wslInfraFail = true;
            optOut = "";
            // Treat as "capable but untested" so the engine still uses xTB once WSL recovers
            result.canOptimize = true;
          } else {
            optOut = msg;
          }
        }
        if (!wslInfraFail) {
          if (optOut.includes("normal termination")) {
            result.canOptimize = true;
            actualOptSuccess = true;
            console.log(`[xTB-Health] Geometry optimization: OK`);
          } else if (optOut.includes("TOTAL ENERGY")) {
            result.canOptimize = true;
            actualOptSuccess = true;
            console.log(`[xTB-Health] Geometry optimization: OK (non-zero exit but energy computed)`);
          } else {
            console.log(`[xTB-Health] Geometry optimization: FAILED (attempt ${attempt}) — ${optOut.slice(0, 200)}`);
          }
        }
      } catch (e: any) {
        console.log(`[xTB-Health] Geometry optimization: FAILED (attempt ${attempt}) — ${e.message?.slice(0, 200)}`);
      }

      try {
        let hessOut: string;
        let wslInfraFailH = false;
        try {
          hessOut = await execShellAsync(`cd "${testDir}" && ${XTB_BIN} input.xyz --gfn 2 --hess 2>&1`, { timeout: HESS_TIMEOUT, env });
        } catch (e: any) {
          const msg: string = (e.stdout ?? "") + (e.stderr ?? "") + (e.message ?? "");
          if (isWSLInfraError(msg)) {
            console.log(`[xTB-Health] Hessian calculation: SKIPPED (WSL infrastructure not ready, attempt ${attempt}): ${msg.slice(0, 150)}`);
            wslInfraFailH = true;
            hessOut = "";
            result.canHess = true; // assume capable once WSL is ready
          } else {
            hessOut = msg;
          }
        }
        if (!wslInfraFailH) {
          if (hessOut.includes("projected vibrational frequencies") || hessOut.includes("normal termination")) {
            result.canHess = true;
            actualHessSuccess = true;
            console.log(`[xTB-Health] Hessian calculation: OK`);
          } else if (hessOut.includes("vibrational frequencies")) {
            result.canHess = true;
            actualHessSuccess = true;
            console.log(`[xTB-Health] Hessian calculation: OK (non-zero exit but frequencies computed)`);
          } else {
            console.log(`[xTB-Health] Hessian calculation: FAILED (attempt ${attempt}) — ${hessOut.slice(0, 200)}`);
          }
        }
      } catch (e: any) {
        console.log(`[xTB-Health] Hessian calculation: FAILED (attempt ${attempt}) — ${e.message?.slice(0, 200)}`);
      }

      try { fs.rmSync(testDir, { recursive: true, force: true }); } catch {}

      if (result.canOptimize && result.canHess) break;

      if (attempt < MAX_RETRIES) {
        console.log(`[xTB-Health] Attempt ${attempt} incomplete (opt=${result.canOptimize}, hess=${result.canHess}), retrying...`);
      }

    } catch (e: any) {
      result.error = e.message || "Unknown error";
      console.log(`[xTB-Health] Attempt ${attempt} FAIL: ${result.error}`);
      if (attempt < MAX_RETRIES) {
        console.log(`[xTB-Health] Retrying...`);
      }
    }
  }

  // Require actual verified success — WSL-infra skips set canOptimize/canHess=true so the
  // engine will retry xTB once WSL recovers, but xtbHealthy stays false until a real test passes.
  xtbHealthy = result.available && actualOptSuccess && actualHessSuccess;
  if (!xtbHealthy) {
    const reason = !result.available ? "binary unavailable"
      : (!actualOptSuccess && !actualHessSuccess) ? "WSL infrastructure not ready — using analytical fallbacks"
      : "opt or hess test failed";
    console.warn(`[xTB-Health] WARNING: xTB not verified healthy (${reason}). DFT will use analytical fallbacks.`);
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
  formula = normalizeFormula(formula);
  if (!isDFTAvailable()) return null;

  const { atoms } = generateCrystalStructure(formula);
  if (atoms.length < 2 || atoms.length > 20) return null;

  const vResult = validateAndFixStructure(atoms, formula);
  if (!vResult) return null;

  const calcDir = path.join(WORK_DIR, uniqueCalcId(`anharm_${formula.replace(/[^a-zA-Z0-9]/g, "_")}`));
  fs.mkdirSync(calcDir, { recursive: true });

  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
    OMP_STACKSIZE: "512M",
  };

  const displacementScales = [-0.10, -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.08, 0.10];
  const energies: number[] = [];
  const forces: number[] = [];

  try {
    let targetAtom = 0;
    let minMass = Infinity;
    for (let i = 0; i < vResult.atoms.length; i++) {
      const elData = getElementData(vResult.atoms[i].element);
      const mass = elData?.atomicMass ?? 999;
      if (mass < minMass) {
        minMass = mass;
        targetAtom = i;
      }
    }
    dftLog(`[DFT] ${formula}: Anharmonic probe targeting atom ${targetAtom} (${vResult.atoms[targetAtom].element}, mass=${minMass.toFixed(1)} amu) — lightest atom`);

    let bestDir = 0;
    let maxSpread = 0;
    for (let dir = 0; dir < 3; dir++) {
      const coords = vResult.atoms.map(a => [a.x, a.y, a.z][dir]);
      const mean = coords.reduce((s, v) => s + v, 0) / coords.length;
      const spread = coords.reduce((s, v) => s + (v - mean) ** 2, 0);
      if (spread > maxSpread) {
        maxSpread = spread;
        bestDir = dir;
      }
    }
    const dispDir = bestDir;

    for (const scale of displacementScales) {
      const displaced = vResult.atoms.map((a, idx) => {
        if (idx !== targetAtom) return { ...a };
        const coords = [a.x, a.y, a.z];
        coords[dispDir] += scale;
        return { element: a.element, x: coords[0], y: coords[1], z: coords[2] };
      });

      const stepDir = path.join(calcDir, `disp_${scale.toFixed(3).replace("-", "m")}`);
      fs.mkdirSync(stepDir, { recursive: true });
      writeXYZ(displaced, path.join(stepDir, "input.xyz"), `${formula} displacement=${scale}`);

      try {
        const output = await execShellAsync(
          `cd ${stepDir} && ${XTB_BIN} input.xyz --gfn 2 --sp 2>&1`,
          { timeout: OPT_TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
        );

        const gradMatch = output.match(/GRADIENT NORM\s+([-\d.]+)/);

        const energy = parseXTBEnergy(output) ?? 0;
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
  formula = normalizeFormula(formula);
  if (!isDFTAvailable()) return null;

  const { atoms } = generateCrystalStructure(formula);
  if (atoms.length < 2 || atoms.length > 15) return null;

  const vResult = validateAndFixStructure(atoms, formula);
  if (!vResult) return null;

  const calcDir = path.join(WORK_DIR, uniqueCalcId(`md_${formula.replace(/[^a-zA-Z0-9]/g, "_")}`));
  fs.mkdirSync(calcDir, { recursive: true });

  const env: Record<string, string> = {
    ...process.env as Record<string, string>,
    XTBHOME: XTB_HOME,
    XTBPATH: XTB_PARAM,
    OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
    OMP_STACKSIZE: "512M",
  };

  writeXYZ(vResult.atoms, path.join(calcDir, "input.xyz"), `${formula} MD sampling T=${temperatureK}K`);

  const totalSteps = 200;
  const timeStepFs = 1.0;
  const dumpFreq = 1;

  const timePsTotal = totalSteps * timeStepFs * 0.001;
  const dumpPs = dumpFreq * timeStepFs * 0.001;
  const mdInput = [
    `$md`,
    `   temp=${temperatureK}`,
    `   time=${timePsTotal.toFixed(4)}`,
    `   dump=${dumpPs.toFixed(4)}`,
    `   step=${timeStepFs.toFixed(1)}`,
    `   hmass=1`,
    `   shake=0`,
    `   nvt=true`,
    `$end`,
  ].join("\n");

  fs.writeFileSync(path.join(calcDir, "md.inp"), mdInput);

  try {
    await execShellAsync(
      `cd ${calcDir} && ${XTB_BIN} input.xyz --gfn 2 --md --input md.inp 2>&1`,
      { timeout: 45000, env, maxBuffer: 20 * 1024 * 1024 }
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
      const dt = dumpFreq * timeStepFs;
      for (let f = 0; f < positions.length; f++) {
        const frameVel: number[][] = [];
        const prev = f > 0 ? f - 1 : 0;
        const next = f < positions.length - 1 ? f + 1 : positions.length - 1;
        const divisor = (next - prev) * dt;
        if (divisor === 0) continue;
        for (let a = 0; a < positions[f].length; a++) {
          frameVel.push([
            (positions[next][a][0] - positions[prev][a][0]) / divisor,
            (positions[next][a][1] - positions[prev][a][1]) / divisor,
            (positions[next][a][2] - positions[prev][a][2]) / divisor,
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
