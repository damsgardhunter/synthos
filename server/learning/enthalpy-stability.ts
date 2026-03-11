import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import { relaxStructureAtPressure } from "./pressure-engine";
import { extractFeatures } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { normalizeFormula } from "./utils";
import { getElementData } from "./elemental-data";

export interface EnthalpyPoint {
  pressureGpa: number;
  totalEnergy: number;
  pv: number;
  enthalpy: number;
  volumeA3: number;
  bulkModulus: number;
  isStable: boolean;
}

export interface EnthalpyCurve {
  formula: string;
  points: EnthalpyPoint[];
  minEnthalpy: number;
  minEnthalpyPressure: number;
  stableRange: { start: number; end: number } | null;
  computedAt: number;
}

export interface EnthalpyStabilityResult {
  formula: string;
  pressureGpa: number;
  enthalpy: number;
  decompositionEnthalpy: number;
  enthalpyDifference: number;
  isStable: boolean;
  isMetastable: boolean;
  stabilityMargin: number;
}

export interface EnthalpyStats {
  totalComputed: number;
  stableCount: number;
  metastableCount: number;
  unstableCount: number;
  avgStabilityWindow: number;
  recentResults: { formula: string; stableRange: string; minH: number }[];
}

const EV_PER_GPA_A3 = 0.00624150913;
const STABILITY_THRESHOLD = 0.025;
const METASTABLE_THRESHOLD = 0.15;
const PRESSURE_STEPS = [0, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350];

const enthalpyCache = new Map<string, EnthalpyCurve>();
const MAX_CACHE = 300;

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  return Object.values(counts).reduce((s, n) => s + n, 0);
}

function estimateElementalVolume(el: string): number {
  const data = getElementData(el);
  if (!data) return 10;
  const rPm = data.atomicRadius || 150;
  const rA = rPm / 100;
  return (4 / 3) * Math.PI * Math.pow(rA, 3);
}

function murnaghanVolume(V0: number, B0: number, pressureGpa: number, Bp: number = 4.0): number {
  if (B0 <= 0 || pressureGpa <= 0) return V0;
  const ratio = 1 + (Bp * pressureGpa) / B0;
  return V0 * Math.pow(ratio, -1 / Bp);
}

function estimateDecompositionEnthalpy(formula: string, pressureGpa: number): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = getTotalAtoms(counts);

  if (elements.length < 2) return 0;

  let decompH = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    try {
      const elResult = relaxStructureAtPressure(el, pressureGpa);
      const elVolume = elResult.compressedVolume;
      const pv = pressureGpa * elVolume * EV_PER_GPA_A3;
      decompH += frac * pv;
    } catch {
      const data = getElementData(el);
      const V0 = estimateElementalVolume(el);
      const B0 = data?.bulkModulus ?? 50;
      const compressedV = murnaghanVolume(V0, B0, pressureGpa);
      decompH += frac * pressureGpa * compressedV * EV_PER_GPA_A3;
    }
  }

  return decompH;
}

export function computeEnthalpy(formula: string, pressureGpa: number): EnthalpyPoint {
  const formationE = computeMiedemaFormationEnergy(formula);

  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const energyPerAtom = formationE / Math.max(1, totalAtoms);

  let features;
  try {
    features = extractFeatures(formula, { pressureGpa } as any);
  } catch {
    features = null;
  }

  const dftFormationE = features?.formationEnergy ?? null;
  const totalEnergy = dftFormationE !== null && dftFormationE !== 0
    ? dftFormationE
    : energyPerAtom;

  const bmResult = relaxStructureAtPressure(formula, pressureGpa);
  const volumeA3 = bmResult.compressedVolume;

  const pv = pressureGpa * volumeA3 * EV_PER_GPA_A3;
  const enthalpy = totalEnergy + pv;

  const decompH = estimateDecompositionEnthalpy(formula, pressureGpa);
  const isStable = enthalpy <= decompH + STABILITY_THRESHOLD;

  return {
    pressureGpa,
    totalEnergy,
    pv: Math.round(pv * 10000) / 10000,
    enthalpy: Math.round(enthalpy * 10000) / 10000,
    volumeA3: Math.round(volumeA3 * 1000) / 1000,
    bulkModulus: bmResult.bulkModulus,
    isStable,
  };
}

export function computeEnthalpyStability(formula: string, pressureGpa: number): EnthalpyStabilityResult {
  const point = computeEnthalpy(formula, pressureGpa);

  const decompH = estimateDecompositionEnthalpy(formula, pressureGpa);
  const formationE = computeMiedemaFormationEnergy(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const elementalEnergy = formationE / Math.max(1, totalAtoms);
  const decompositionEnthalpy = elementalEnergy + decompH;

  const diff = point.enthalpy - decompositionEnthalpy;
  const isStable = diff <= STABILITY_THRESHOLD;
  const isMetastable = !isStable && diff <= METASTABLE_THRESHOLD;
  const stabilityMargin = -diff;

  return {
    formula,
    pressureGpa,
    enthalpy: point.enthalpy,
    decompositionEnthalpy: Math.round(decompositionEnthalpy * 10000) / 10000,
    enthalpyDifference: Math.round(diff * 10000) / 10000,
    isStable,
    isMetastable,
    stabilityMargin: Math.round(stabilityMargin * 10000) / 10000,
  };
}

export function computeEnthalpyPressureCurve(formula: string): EnthalpyCurve {
  const cacheKey = normalizeFormula(formula);
  const cached = enthalpyCache.get(cacheKey);
  if (cached && Date.now() - cached.computedAt < 30 * 60 * 1000) {
    return cached;
  }

  const points: EnthalpyPoint[] = [];
  for (const p of PRESSURE_STEPS) {
    try {
      points.push(computeEnthalpy(formula, p));
    } catch {
      continue;
    }
  }

  let minEnthalpy = Infinity;
  let minEnthalpyPressure = 0;
  for (const pt of points) {
    if (pt.enthalpy < minEnthalpy) {
      minEnthalpy = pt.enthalpy;
      minEnthalpyPressure = pt.pressureGpa;
    }
  }

  let stableRange: { start: number; end: number } | null = null;
  let rangeStart = -1;
  for (const pt of points) {
    if (pt.isStable && rangeStart < 0) {
      rangeStart = pt.pressureGpa;
    } else if (!pt.isStable && rangeStart >= 0) {
      stableRange = { start: rangeStart, end: points[points.indexOf(pt) - 1]?.pressureGpa ?? rangeStart };
      break;
    }
  }
  if (rangeStart >= 0 && !stableRange) {
    stableRange = { start: rangeStart, end: points[points.length - 1].pressureGpa };
  }

  const curve: EnthalpyCurve = {
    formula: cacheKey,
    points,
    minEnthalpy: minEnthalpy === Infinity ? 0 : Math.round(minEnthalpy * 10000) / 10000,
    minEnthalpyPressure,
    stableRange,
    computedAt: Date.now(),
  };

  if (enthalpyCache.size >= MAX_CACHE) {
    const firstKey = enthalpyCache.keys().next().value;
    if (firstKey !== undefined) enthalpyCache.delete(firstKey);
  }
  enthalpyCache.set(cacheKey, curve);

  return curve;
}

export function findStabilityPressureWindow(formula: string): {
  formula: string;
  hasStableWindow: boolean;
  stableRange: { start: number; end: number } | null;
  windowWidth: number;
  minEnthalpyPressure: number;
  minEnthalpy: number;
} {
  const curve = computeEnthalpyPressureCurve(formula);

  return {
    formula,
    hasStableWindow: curve.stableRange !== null,
    stableRange: curve.stableRange,
    windowWidth: curve.stableRange ? curve.stableRange.end - curve.stableRange.start : 0,
    minEnthalpyPressure: curve.minEnthalpyPressure,
    minEnthalpy: curve.minEnthalpy,
  };
}

export function computeEnthalpyAtPressure(formula: string, pressureGpa: number): number {
  const point = computeEnthalpy(formula, pressureGpa);
  return point.enthalpy;
}

export function getEnthalpyStats(): EnthalpyStats {
  const curves = Array.from(enthalpyCache.values());

  let stableCount = 0;
  let metastableCount = 0;
  let unstableCount = 0;
  let totalWindow = 0;

  for (const curve of curves) {
    if (curve.stableRange) {
      const width = curve.stableRange.end - curve.stableRange.start;
      if (width > 50) {
        stableCount++;
      } else if (width > 0) {
        metastableCount++;
      }
      totalWindow += width;
    } else {
      unstableCount++;
    }
  }

  const recentResults = curves.slice(-10).map(c => ({
    formula: c.formula,
    stableRange: c.stableRange ? `${c.stableRange.start}-${c.stableRange.end} GPa` : "none",
    minH: c.minEnthalpy,
  }));

  return {
    totalComputed: curves.length,
    stableCount,
    metastableCount,
    unstableCount,
    avgStabilityWindow: curves.length > 0 ? Math.round(totalWindow / curves.length) : 0,
    recentResults,
  };
}

export function clearEnthalpyCache(): void {
  enthalpyCache.clear();
}
