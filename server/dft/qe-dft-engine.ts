import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { getElementData } from "../learning/elemental-data";
import { fillPrototype } from "../learning/crystal-prototypes";

const PROJECT_ROOT = path.resolve(process.cwd());
const XTB_BIN = path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb");
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = "/tmp/dft_calculations";
const TIMEOUT_MS = 60_000;

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
  let totalR = 0;
  let totalN = 0;
  for (const el of elements) {
    const n = Math.round(counts[el] || 1);
    totalR += getAvgRadius(el) * n;
    totalN += n;
  }
  const avgR = totalR / Math.max(1, totalN);
  return avgR * 2.8 + 0.5;
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
];

function matchPrototype(counts: Record<string, number>): { proto: PrototypeStructure; siteMap: Record<string, string> } | null {
  const elements = Object.keys(counts).sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
  const nElements = elements.length;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  const ratios = elements.map(el => Math.round(counts[el]));
  const gcdVal = ratios.reduce((a, b) => gcd(a, b));
  const reduced = ratios.map(r => r / gcdVal);

  for (const proto of CRYSTAL_PROTOTYPES) {
    const siteCounts: Record<string, number> = {};
    for (const pos of proto.fractionalPositions) {
      siteCounts[pos.site] = (siteCounts[pos.site] || 0) + 1;
    }
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
    return { atoms: atomPositions, prototype: `${chemProto.templateName} prototype lattice` };
  }

  const protoMatch = matchPrototype(counts);
  if (protoMatch) {
    const atoms = buildStructureFromPrototype(protoMatch.proto, protoMatch.siteMap, elements, counts);
    if (atoms.length >= 2) {
      return { atoms, prototype: protoMatch.proto.name };
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
      return { atoms, prototype: scaledMatch.proto.name + "-scaled" };
    }
  }

  const { atoms, proto } = buildGenericStructure(scaledCounts);
  return { atoms, prototype: proto };
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
const CACHE_MAX = 500;

export async function runDFTCalculation(formula: string): Promise<DFTResult> {
  const cacheKey = formula.replace(/\s+/g, "");
  if (xtbResultCache.has(cacheKey)) {
    return xtbResultCache.get(cacheKey)!;
  }

  const startTime = Date.now();
  const calcId = `${cacheKey.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`;
  const calcDir = path.join(WORK_DIR, calcId);
  fs.mkdirSync(calcDir, { recursive: true });

  const { atoms, prototype } = generateCrystalStructure(formula);
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
  };

  if (atoms.length < 2) {
    result.error = "Too few atoms for DFT";
    return result;
  }

  writeXYZ(atoms, xyzPath, `${formula} [${prototype}]`);

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

async function computeElementalEnergy(element: string): Promise<number | null> {
  if (elementRefEnergies.has(element)) {
    return elementRefEnergies.get(element) ?? null;
  }

  const calcDir = path.join(WORK_DIR, `ref_${element}_${Date.now()}`);
  fs.mkdirSync(calcDir, { recursive: true });

  const r = getAvgRadius(element);
  const atoms: AtomPosition[] = [
    { element, x: 0, y: 0, z: 0 },
    { element, x: r * 2, y: 0, z: 0 },
  ];

  const xyzPath = path.join(calcDir, `${element}.xyz`);
  writeXYZ(atoms, xyzPath, `${element} dimer reference`);

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
      { timeout: 30000, env, maxBuffer: 5 * 1024 * 1024 }
    ).toString();

    const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
    if (energyMatch && output.includes("normal termination")) {
      const energyPerAtom = parseFloat(energyMatch[1]) / 2;
      elementRefEnergies.set(element, energyPerAtom);
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

  const actualAtomCount = dftResult.atomCount;
  if (actualAtomCount === 0) return null;

  const counts = parseFormula(formula);
  const elements = Object.keys(counts);
  const originalTotal = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);
  if (originalTotal === 0 || elements.length === 0) return null;

  const scaleFactor = actualAtomCount / originalTotal;

  let elementalTotal = 0;
  for (const [el, count] of Object.entries(counts)) {
    const scaledCount = Math.max(1, Math.round(Math.round(count) * scaleFactor));
    const refE = await computeElementalEnergy(el);
    if (refE === null) return null;
    elementalTotal += refE * scaledCount;
  }

  const formationTotal = dftResult.totalEnergy - elementalTotal;
  const HA_TO_EV = 27.2114;
  const efPerAtom = (formationTotal / actualAtomCount) * HA_TO_EV;

  if (Math.abs(efPerAtom) > 15) {
    console.log(`[DFT] ${formula}: Formation energy ${efPerAtom.toFixed(3)} eV/atom exceeds sanity bounds, discarding`);
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
  try {
    fs.mkdirSync(calcDir, { recursive: true });

    const { atoms, prototype } = generateCrystalStructure(formula);
    if (atoms.length < 2) return null;

    writeXYZ(atoms, path.join(calcDir, "input.xyz"), `${formula} phonon check (${prototype})`);

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "1G",
    };

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

    const IMAG_THRESHOLD = -50;
    const imaginaryModes = frequencies.filter(f => f < IMAG_THRESHOLD);
    const lowestFreq = frequencies.length > 0 ? Math.min(...frequencies) : 0;

    const result: PhononStability = {
      hasImaginaryModes: imaginaryModes.length > 0,
      imaginaryModeCount: imaginaryModes.length,
      lowestFrequency: lowestFreq,
      frequencies: frequencies.slice(0, 20),
      zeroPointEnergy: zpe,
    };

    phononCache.set(formula, result);
    if (phononCache.size > 100) {
      const oldest = phononCache.keys().next().value;
      if (oldest) phononCache.delete(oldest);
    }

    return result;
  } catch (err) {
    console.log(`[DFT] ${formula}: Phonon check failed: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  } finally {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
  }
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
  if (dftResult.atomCount <= 12) {
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
  };
}

export function isDFTAvailable(): boolean {
  try {
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
    refElements: elementRefEnergies.size,
  };
}
