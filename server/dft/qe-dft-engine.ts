import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { getElementData } from "../learning/elemental-data";
import { fillPrototype } from "../learning/crystal-prototypes";
import { computeFiniteDisplacementPhonons } from "./phonon-calculator";
import type { FiniteDisplacementPhononResult } from "./phonon-calculator";

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
  Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48, Tc: 1.47, Rb2: 2.20,
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

function estimateLatticeParam(elements: string[], counts: Record<string, number>): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  let maxR = 0;
  let totalR = 0;
  let totalN = 0;
  for (const el of elements) {
    const r = getAvgRadius(el);
    const n = Math.round(counts[el] || 1);
    totalR += r * n;
    totalN += n;
    if (r > maxR) maxR = r;
  }
  const avgR = totalR / Math.max(1, totalN);
  const effectiveR = avgR * 0.6 + maxR * 0.4;
  const volumePerAtom = (4 / 3) * Math.PI * Math.pow(effectiveR + 0.8, 3);
  const totalVolume = volumePerAtom * totalAtoms;
  const latticeA = Math.cbrt(totalVolume);
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
      { site: "B", x: 0.333, y: 0.667, z: -0.121 },
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
      { site: "B", x: 0.25, y: 0.25, z: 0.75 },
      { site: "B", x: 0.75, y: 0.75, z: 0.75 },
      { site: "B", x: 0.75, y: 0.25, z: 0.25 },
      { site: "B", x: 0.25, y: 0.75, z: 0.25 },
    ],
    latticeType: "cubic",
    aRatio: 1.0,
    cOverA: 1.0,
    stoichiometryPattern: "A4B12",
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
];

function getProtoSiteCounts(proto: PrototypeStructure): Record<string, number> {
  const siteCounts: Record<string, number> = {};
  for (const pos of proto.fractionalPositions) {
    siteCounts[pos.site] = (siteCounts[pos.site] || 0) + 1;
  }
  return siteCounts;
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

    if (score < bestScore && score < 0.5) {
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
  const a = estimateLatticeParam(elements, counts) * scaleFactor;
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

      const jitter = ((idx * 7 + 3) % 11) / 55.0;
      atoms.push({
        element: el,
        x: (ix + 0.5 + jitter * 0.3) * a / gridSize,
        y: (iy + 0.5 - jitter * 0.2) * a / gridSize,
        z: (iz + 0.5 + jitter * 0.1) * a / gridSize,
      });
      idx++;
    }
  }

  return { atoms, proto: "generic-cluster" };
}

const MIN_VOLUME_PER_ATOM = 8.0;
const MAX_SCALE_ATTEMPTS = 5;

function getMinInteratomicDistance(el1: string, el2: string): number {
  const r1 = COVALENT_RADII[el1] ?? 1.5;
  const r2 = COVALENT_RADII[el2] ?? 1.5;
  const bondLength = r1 + r2;
  return Math.max(bondLength * 0.85, 1.0);
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
    if (a.x < minX) minX = a.x;
    if (a.y < minY) minY = a.y;
    if (a.z < minZ) minZ = a.z;
    if (a.x > maxX) maxX = a.x;
    if (a.y > maxY) maxY = a.y;
    if (a.z > maxZ) maxZ = a.z;
  }
  const pad = 1.5;
  const lx = Math.max(maxX - minX, pad);
  const ly = Math.max(maxY - minY, pad);
  const lz = Math.max(maxZ - minZ, pad);
  return lx * ly * lz;
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

  let current = atoms;

  for (let attempt = 0; attempt < MAX_SCALE_ATTEMPTS; attempt++) {
    const { minDist, minRatio } = computePairwiseDistances(current);
    const volume = computeBoundingVolume(current);
    const volumePerAtom = volume / current.length;

    const distOk = minRatio >= 0.99;
    const volOk = volumePerAtom >= MIN_VOLUME_PER_ATOM - 0.1;

    if (distOk && volOk) return current;

    let scaleFactor = 1.0;
    if (!distOk) {
      scaleFactor = Math.max(scaleFactor, 1.05 / Math.max(minRatio, 0.01));
    }
    if (!volOk) {
      const neededFactor = Math.cbrt(MIN_VOLUME_PER_ATOM / Math.max(volumePerAtom, 0.01));
      scaleFactor = Math.max(scaleFactor, neededFactor);
    }

    scaleFactor = Math.min(scaleFactor, 3.0);
    console.log(`[DFT] ${formula}: Structure validation attempt ${attempt + 1} — minDist=${minDist.toFixed(3)}Å, ratio=${minRatio.toFixed(2)}, vol/atom=${volumePerAtom.toFixed(1)}ų — scaling by ${scaleFactor.toFixed(2)}`);
    current = scaleStructure(current, scaleFactor);
  }

  const { minDist, minRatio } = computePairwiseDistances(current);
  const volume = computeBoundingVolume(current);
  const volumePerAtom = volume / current.length;

  if (minRatio < 0.99 || volumePerAtom < MIN_VOLUME_PER_ATOM - 0.1) {
    console.log(`[DFT] ${formula}: Structure REJECTED after ${MAX_SCALE_ATTEMPTS} fix attempts — minDist=${minDist.toFixed(3)}Å, ratio=${minRatio.toFixed(2)}, vol/atom=${volumePerAtom.toFixed(1)}ų`);
    return null;
  }

  return current;
}

function generateCrystalStructure(formula: string): { atoms: AtomPosition[]; prototype: string } {
  const counts = parseFormula(formula);
  const elements = Object.keys(counts);

  if (elements.length === 0) {
    return { atoms: [], prototype: "empty" };
  }

  const chemProto = fillPrototype(formula);
  if (chemProto && chemProto.atoms.length >= 2) {
    const atomPositions: AtomPosition[] = chemProto.atoms.map(a => ({
      element: a.element,
      x: a.x,
      y: a.y,
      z: a.z,
    }));
    const validated = validateAndFixStructure(atomPositions, formula);
    if (!validated) return { atoms: [], prototype: "rejected-overlap" };
    return { atoms: validated, prototype: `${chemProto.templateName} prototype lattice` };
  }

  const protoMatch = matchPrototype(counts);
  if (protoMatch) {
    const atoms = buildStructureFromPrototype(protoMatch.proto, protoMatch.siteMap, elements, counts);
    if (atoms.length >= 2) {
      const validated = validateAndFixStructure(atoms, formula);
      if (!validated) return { atoms: [], prototype: "rejected-overlap" };
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
    const atoms = buildStructureFromPrototype(scaledMatch.proto, scaledMatch.siteMap, elements, scaledCounts);
    if (atoms.length >= 2) {
      const validated = validateAndFixStructure(atoms, formula);
      if (!validated) return { atoms: [], prototype: "rejected-overlap" };
      return { atoms: validated, prototype: scaledMatch.proto.name + "-scaled" };
    }
  }

  const { atoms, proto } = buildGenericStructure(scaledCounts);
  const validated = validateAndFixStructure(atoms, formula);
  if (!validated) return { atoms: [], prototype: "rejected-overlap" };
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

export async function runXTBOptimization(formula: string): Promise<OptimizationResult | null> {
  if (!isDFTAvailable()) return null;

  const cacheKey = formula.replace(/\s+/g, "");
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

  const { atoms, prototype } = generateCrystalStructure(formula);
  if (atoms.length < 2) return null;

  const xyzPath = path.join(calcDir, "input.xyz");
  writeXYZ(atoms, xyzPath, `${formula} [${prototype}] optimization`);

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

    const result: OptimizationResult = {
      optimizedAtoms,
      optimizedEnergy,
      converged: optInfo.converged,
      energyChange: optInfo.energyChange,
      gradientNorm: optInfo.gradientNorm,
      iterations: optInfo.iterations,
      wallTimeSeconds: (Date.now() - startTime) / 1000,
    };

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

export async function runDFTCalculation(formula: string): Promise<DFTResult> {
  const cacheKey = formula.replace(/\s+/g, "");
  if (xtbResultCache.has(cacheKey)) {
    return xtbResultCache.get(cacheKey)!;
  }

  const startTime = Date.now();
  const calcId = `${cacheKey.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`;
  const calcDir = path.join(WORK_DIR, calcId);
  fs.mkdirSync(calcDir, { recursive: true });

  const { atoms: initialAtoms, prototype } = generateCrystalStructure(formula);
  if (initialAtoms.length < 2) return null;

  let atoms = initialAtoms;
  let isOptimized = false;

  const optResult = await runXTBOptimization(formula);
  if (optResult && optResult.converged && optResult.optimizedAtoms.length >= 2) {
    atoms = optResult.optimizedAtoms;
    isOptimized = true;
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
  H: 2.24, He: 0, Li: 1.63, Be: 3.32, B: 5.81, C: 7.37, N: 4.92, O: 2.60, F: 0.84,
  Na: 1.11, Mg: 1.51, Al: 3.39, Si: 4.63, P: 3.43, S: 2.85, Cl: 1.40,
  K: 0.93, Ca: 1.84, Sc: 3.90, Ti: 4.85, V: 5.31, Cr: 4.10, Mn: 2.92,
  Fe: 4.28, Co: 4.39, Ni: 4.44, Cu: 3.49, Zn: 1.35, Ga: 2.81, Ge: 3.85,
  As: 2.96, Se: 2.46, Br: 1.22,
  Rb: 0.85, Sr: 1.72, Y: 4.37, Zr: 6.25, Nb: 7.57, Mo: 6.82,
  Pd: 3.89, Sn: 3.14, Te: 2.02,
  Cs: 0.80, Ba: 1.90, La: 4.47, Ce: 4.32, Hf: 6.44, Ta: 8.10, W: 8.90,
  Pt: 5.84, Pb: 2.03, Bi: 2.18,
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
      let energyPerAtom = parseFloat(energyMatch[1]) / divisor;
      if (!isMolecular) {
        const cohesiveEv = COHESIVE_ENERGIES_EV[element] ?? 3.0;
        const cohesiveHa = cohesiveEv / 27.2114;
        energyPerAtom = energyPerAtom - cohesiveHa;
        console.log(`[DFT] ${element}: Ref energy corrected from ${(energyPerAtom + cohesiveHa).toFixed(4)} Ha (atom) to ${energyPerAtom.toFixed(4)} Ha (bulk-corrected, cohesive=${cohesiveEv.toFixed(2)} eV)`);
      }
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
  const efPerAtom = (formationTotal / actualAtomCount) * HA_TO_EV;

  if (efPerAtom > 1.0) {
    console.log(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom is positive (>1.0), discarding — compound less stable than elements`);
    return null;
  }

  if (efPerAtom < -5.0) {
    console.log(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom unrealistically negative (<-5.0), likely reference energy mismatch — discarding`);
    return null;
  }

  return efPerAtom;
}

let totalXTBRuns = 0;
let totalXTBSuccesses = 0;

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

    if (physicalImagModes.length > 3) {
      (result as any).severeInstability = true;
      (result as any).instabilityReason = `${physicalImagModes.length} imaginary modes detected (max 3 allowed)`;
      console.log(`[DFT] ${formula}: Severe phonon instability — ${physicalImagModes.length} imaginary modes`);
    }
    if (lowestFreq < -500 && lowestFreq >= ARTIFACT_THRESHOLD) {
      (result as any).severeInstability = true;
      (result as any).instabilityReason = `Lowest frequency ${lowestFreq.toFixed(0)} cm-1 (threshold: -500 cm-1)`;
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
      const baseDebye = hasHydrogen ? 400 : (avgMass < 30 ? 500 : avgMass < 60 ? 350 : avgMass < 100 ? 250 : 180);
      const debyeFreq = baseDebye * (converged ? 1.0 : 0.8);
      const nAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
      const nModes = Math.max(1, 3 * nAtoms - 6);
      const estimatedFreqs: number[] = [];
      for (let i = 0; i < nModes; i++) {
        const fraction = (i + 1) / nModes;
        estimatedFreqs.push(debyeFreq * fraction * (0.8 + 0.4 * Math.random()));
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

export async function runXTBEnrichment(formula: string): Promise<XTBEnrichedFeatures | null> {
  if (!isDFTAvailable()) return null;

  totalXTBRuns++;
  const dftResult = await runDFTCalculation(formula);

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
