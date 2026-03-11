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
  narrowStableCount: number;
  unstableCount: number;
  avgStabilityWindow: number;
  windowDistribution: { elite: number; broad: number; moderate: number; narrow: number; none: number };
  recentResults: { formula: string; stableRange: string; minH_eVPerAtom: number; windowGPa: number }[];
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

  let features;
  try {
    features = extractFeatures(formula, { pressureGpa } as any);
  } catch {
    features = null;
  }

  const dftFormationE = features?.formationEnergy ?? null;
  const totalEnergy = dftFormationE !== null && dftFormationE !== 0
    ? dftFormationE
    : formationE;

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
  const decompositionEnthalpy = point.totalEnergy + decompH;

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

function findRefinementPressures(points: EnthalpyPoint[]): number[] {
  const refinements: number[] = [];
  const sorted = [...points].sort((a, b) => a.pressureGpa - b.pressureGpa);

  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const curr = sorted[i];
    const gap = curr.pressureGpa - prev.pressureGpa;
    if (gap < 5) continue;

    if (prev.isStable !== curr.isStable) {
      const mid = (prev.pressureGpa + curr.pressureGpa) / 2;
      refinements.push(mid);
      if (gap > 15) {
        refinements.push(prev.pressureGpa + gap * 0.25);
        refinements.push(prev.pressureGpa + gap * 0.75);
      }
    }

    if (gap > 10) {
      const dH = Math.abs(curr.enthalpy - prev.enthalpy);
      const gradient = dH / gap;
      if (gradient > 0.005) {
        refinements.push((prev.pressureGpa + curr.pressureGpa) / 2);
      }
    }
  }

  const existingPressures = points.map(pt => pt.pressureGpa);
  const accepted: number[] = [];
  for (const p of refinements) {
    if (p <= 0 || p > 350) continue;
    let tooClose = false;
    for (let i = 0; i < existingPressures.length; i++) {
      if (Math.abs(p - existingPressures[i]) < 2) { tooClose = true; break; }
    }
    if (tooClose) continue;
    for (let i = 0; i < accepted.length; i++) {
      if (Math.abs(p - accepted[i]) < 2) { tooClose = true; break; }
    }
    if (!tooClose) accepted.push(p);
  }
  return accepted;
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

  const refinePressures = findRefinementPressures(points);
  for (const p of refinePressures.slice(0, 10)) {
    try {
      points.push(computeEnthalpy(formula, p));
    } catch {
      continue;
    }
  }

  points.sort((a, b) => a.pressureGpa - b.pressureGpa);

  const secondPass = findRefinementPressures(points);
  for (const p of secondPass.slice(0, 5)) {
    try {
      points.push(computeEnthalpy(formula, p));
    } catch {
      continue;
    }
  }

  points.sort((a, b) => a.pressureGpa - b.pressureGpa);

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
  let totalComputed = 0;
  let stableCount = 0;
  let metastableCount = 0;
  let narrowStableCount = 0;
  let unstableCount = 0;
  let totalWindow = 0;
  const dist = { elite: 0, broad: 0, moderate: 0, narrow: 0, none: 0 };

  const topByWindow: { formula: string; stableRange: string; minH_eVPerAtom: number; windowGPa: number; computedAt: number }[] = [];

  enthalpyCache.forEach((curve) => {
    totalComputed++;
    if (curve.stableRange) {
      const width = curve.stableRange.end - curve.stableRange.start;
      totalWindow += width;

      if (width >= 100) {
        dist.elite++;
        stableCount++;
      } else if (width >= 50) {
        dist.broad++;
        stableCount++;
      } else if (width >= 20) {
        dist.moderate++;
        narrowStableCount++;
      } else if (width > 0) {
        dist.narrow++;
        narrowStableCount++;
      } else {
        dist.none++;
        unstableCount++;
      }

      topByWindow.push({
        formula: curve.formula,
        stableRange: `${curve.stableRange.start}-${curve.stableRange.end} GPa`,
        minH_eVPerAtom: curve.minEnthalpy,
        windowGPa: width,
        computedAt: curve.computedAt,
      });
    } else {
      dist.none++;
      unstableCount++;

      topByWindow.push({
        formula: curve.formula,
        stableRange: "none",
        minH_eVPerAtom: curve.minEnthalpy,
        windowGPa: 0,
        computedAt: curve.computedAt,
      });
    }
  });

  topByWindow.sort((a, b) => {
    if (b.windowGPa !== a.windowGPa) return b.windowGPa - a.windowGPa;
    return a.minH_eVPerAtom - b.minH_eVPerAtom;
  });

  metastableCount = narrowStableCount;

  return {
    totalComputed,
    stableCount,
    metastableCount,
    narrowStableCount,
    unstableCount,
    avgStabilityWindow: totalComputed > 0 ? Math.round(totalWindow / totalComputed) : 0,
    windowDistribution: dist,
    recentResults: topByWindow.slice(0, 10),
  };
}

const CACHE_STALE_MS = 24 * 60 * 60 * 1000;

export function clearEnthalpyCache(fullClear: boolean = false): void {
  if (fullClear) {
    enthalpyCache.clear();
    return;
  }

  const now = Date.now();
  const keysToRemove: string[] = [];
  enthalpyCache.forEach((curve, key) => {
    if (now - curve.computedAt > CACHE_STALE_MS) {
      keysToRemove.push(key);
    }
  });
  for (const key of keysToRemove) {
    enthalpyCache.delete(key);
  }
}
