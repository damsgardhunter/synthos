import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { gbPredict, gbPredictWithUncertainty } from "./gradient-boost";
import { getGNNPrediction, type GNNPrediction } from "./graph-neural-net";
import { computeEnthalpy } from "./enthalpy-stability";
import { normalizeFormula } from "./utils";

export interface PressurePoint {
  pressureGpa: number;
  tc: number;
  formationEnergy: number;
  bandgap: number;
  enthalpy: number;
  enthalpyStable: boolean;
  xgbTc: number;
  gnnTc: number;
  gnnFormationEnergy: number;
  gnnBandgap: number;
}

export interface UncertainPressureRegion {
  pressureStart: number;
  pressureEnd: number;
  avgUncertainty: number;
  maxUncertainty: number;
  peakPressure: number;
}

export interface PressureCurve {
  formula: string;
  points: PressurePoint[];
  optimalPressureGpa: number;
  maxTc: number;
  isSensitive: boolean;
  sensitivityCategory: "high" | "moderate" | "low";
  maxDTcDP: number;
  uncertainRegions: UncertainPressureRegion[];
  computedAt: number;
}

export interface PressureSensitivityResult {
  formula: string;
  avgDTcDP: number;
  maxDTcDP: number;
  maxDTcDPPressure: number;
  isSensitive: boolean;
  sensitivityCategory: "high" | "moderate" | "low";
}

export interface PressureCurveStats {
  totalCurves: number;
  avgMaxTc: number;
  avgOptimalPressure: number;
  highTcCount: number;
  sensitiveCount: number;
  pressureDistribution: { range: string; count: number }[];
}

const PRESSURE_MIN = 0;
const PRESSURE_MAX = 350;
const PRESSURE_STEP = 10;
const SENSITIVITY_THRESHOLD = 0.5;

const pressureCurveCache = new Map<string, PressureCurve>();
const MAX_CACHE_SIZE = 500;

function evictCacheIfNeeded(): void {
  if (pressureCurveCache.size >= MAX_CACHE_SIZE) {
    const firstKey = pressureCurveCache.keys().next().value;
    if (firstKey !== undefined) pressureCurveCache.delete(firstKey);
  }
}

function pressureDomeFactor(pressureGpa: number, peakPressure: number = 150): number {
  const ratio = pressureGpa / peakPressure;
  return Math.exp(-0.5 * (ratio - 1) ** 2 / 0.3);
}

function pressureBandgapFactor(pressureGpa: number): number {
  return Math.exp(-pressureGpa * 0.003);
}

const TYPICAL_VOLUME_PER_ATOM = 12;
const EV_PER_GPA_A3 = 1 / 160.2;
const PV_FACTOR = TYPICAL_VOLUME_PER_ATOM * EV_PER_GPA_A3;

interface CachedGNNResult {
  valid: boolean;
  predictedTc: number;
  formationEnergy: number;
  bandgap: number;
}

function fetchGNNOnce(formula: string): CachedGNNResult {
  try {
    const gnnResult: GNNPrediction = getGNNPrediction(formula);
    if (Number.isFinite(gnnResult.predictedTc)) {
      return { valid: true, predictedTc: gnnResult.predictedTc, formationEnergy: gnnResult.formationEnergy, bandgap: gnnResult.bandgap };
    }
  } catch {}
  return { valid: false, predictedTc: 0, formationEnergy: 0, bandgap: 0 };
}

function predictAtPressure(
  formula: string,
  pressureGpa: number,
  baseFeatures: MLFeatureVector,
  gnn: CachedGNNResult
): PressurePoint {
  const overridden: MLFeatureVector = { ...baseFeatures, pressureGpa };

  const xgbResult = gbPredict(overridden, formula);
  const xgbTc = Math.max(0, xgbResult.tcPredicted);

  let gnnTc = 0;
  let gnnFormationEnergy = 0;
  let gnnBandgap = 0;

  if (gnn.valid) {
    const domeFactor = pressureDomeFactor(pressureGpa);
    gnnTc = Math.max(0, gnn.predictedTc * domeFactor);
    gnnFormationEnergy = gnn.formationEnergy * (1 + pressureGpa * 0.003);
    gnnBandgap = Math.max(0, gnn.bandgap * pressureBandgapFactor(pressureGpa));
  }

  let ensembleTc: number;
  if (gnn.valid) {
    ensembleTc = xgbTc * 0.5 + gnnTc * 0.5;
  } else {
    ensembleTc = xgbTc;
  }

  const formationEnergy = gnn.valid
    ? gnnFormationEnergy
    : (baseFeatures.formationEnergy ?? 0) * (1 + pressureGpa * 0.003);
  const bandgap = gnn.valid
    ? gnnBandgap
    : Math.max(0, (baseFeatures.bandGap ?? 0) * pressureBandgapFactor(pressureGpa));

  let enthalpy = 0;
  let enthalpyStable = true;
  try {
    const hPoint = computeEnthalpy(formula, pressureGpa);
    enthalpy = hPoint.enthalpy;
    enthalpyStable = hPoint.isStable;
  } catch {
    enthalpy = formationEnergy + pressureGpa * PV_FACTOR;
    enthalpyStable = enthalpy < 0.1;
  }

  return {
    pressureGpa,
    tc: ensembleTc,
    formationEnergy,
    bandgap,
    enthalpy,
    enthalpyStable,
    xgbTc,
    gnnTc,
    gnnFormationEnergy,
    gnnBandgap,
  };
}

function computeUncertainRegionsFromFeatures(formula: string, baseFeatures: MLFeatureVector): UncertainPressureRegion[] {
  const uncertainties: { pressure: number; uncertainty: number }[] = [];

  for (let p = 0; p <= PRESSURE_MAX; p += 20) {
    try {
      const overridden: MLFeatureVector = { ...baseFeatures, pressureGpa: p };
      const xgbResult = gbPredictWithUncertainty(overridden, formula);
      uncertainties.push({ pressure: p, uncertainty: xgbResult.normalizedUncertainty });
    } catch {
      uncertainties.push({ pressure: p, uncertainty: 0.5 });
    }
  }

  if (uncertainties.length < 3) return [];

  const globalAvg = uncertainties.reduce((s, u) => s + u.uncertainty, 0) / uncertainties.length;
  const threshold = Math.max(globalAvg * 1.3, 0.35);

  const regions: UncertainPressureRegion[] = [];
  let regionStart = -1;
  let regionPoints: { pressure: number; uncertainty: number }[] = [];

  for (let i = 0; i < uncertainties.length; i++) {
    if (uncertainties[i].uncertainty >= threshold) {
      if (regionStart < 0) {
        regionStart = uncertainties[i].pressure;
        regionPoints = [];
      }
      regionPoints.push(uncertainties[i]);
    } else if (regionStart >= 0) {
      const maxU = regionPoints.reduce((m, p) => Math.max(m, p.uncertainty), 0);
      const peakP = regionPoints.find(p => p.uncertainty === maxU)!.pressure;
      regions.push({
        pressureStart: regionStart,
        pressureEnd: regionPoints[regionPoints.length - 1].pressure,
        avgUncertainty: regionPoints.reduce((s, p) => s + p.uncertainty, 0) / regionPoints.length,
        maxUncertainty: maxU,
        peakPressure: peakP,
      });
      regionStart = -1;
      regionPoints = [];
    }
  }

  if (regionStart >= 0 && regionPoints.length > 0) {
    const maxU = regionPoints.reduce((m, p) => Math.max(m, p.uncertainty), 0);
    const peakP = regionPoints.find(p => p.uncertainty === maxU)!.pressure;
    regions.push({
      pressureStart: regionStart,
      pressureEnd: regionPoints[regionPoints.length - 1].pressure,
      avgUncertainty: regionPoints.reduce((s, p) => s + p.uncertainty, 0) / regionPoints.length,
      maxUncertainty: maxU,
      peakPressure: peakP,
    });
  }

  return regions.sort((a, b) => b.maxUncertainty - a.maxUncertainty);
}

export function predictPressureCurve(formula: string): PressureCurve {
  const normFormula = normalizeFormula(formula);
  const cached = pressureCurveCache.get(normFormula);
  if (cached) return cached;

  const baseFeatures = extractFeatures(normFormula, { pressureGpa: 0 } as any);
  const gnn = fetchGNNOnce(normFormula);

  const coarsePoints: PressurePoint[] = [];
  for (let p = PRESSURE_MIN; p <= PRESSURE_MAX; p += PRESSURE_STEP) {
    coarsePoints.push(predictAtPressure(normFormula, p, baseFeatures, gnn));
  }

  const refinedPressures = new Set<number>();
  for (let i = 1; i < coarsePoints.length; i++) {
    const dp = coarsePoints[i].pressureGpa - coarsePoints[i - 1].pressureGpa;
    if (dp === 0) continue;
    const slope = Math.abs((coarsePoints[i].tc - coarsePoints[i - 1].tc) / dp);
    if (slope > SENSITIVITY_THRESHOLD * 0.5) {
      const pLo = coarsePoints[i - 1].pressureGpa;
      const pHi = coarsePoints[i].pressureGpa;
      const mid = (pLo + pHi) / 2;
      refinedPressures.add(Math.round(mid));
      refinedPressures.add(Math.round(pLo + (pHi - pLo) * 0.25));
      refinedPressures.add(Math.round(pLo + (pHi - pLo) * 0.75));
    }
  }

  const allPoints = [...coarsePoints];
  const existingPressures = new Set(coarsePoints.map(p => p.pressureGpa));
  for (const rp of refinedPressures) {
    if (!existingPressures.has(rp) && rp >= PRESSURE_MIN && rp <= PRESSURE_MAX) {
      allPoints.push(predictAtPressure(normFormula, rp, baseFeatures, gnn));
    }
  }
  allPoints.sort((a, b) => a.pressureGpa - b.pressureGpa);

  let maxTc = 0;
  let optimalPressureGpa = 0;
  let maxSlope = 0;
  for (const pt of allPoints) {
    if (pt.tc > maxTc) {
      maxTc = pt.tc;
      optimalPressureGpa = pt.pressureGpa;
    }
  }
  for (let i = 1; i < allPoints.length; i++) {
    const dp = allPoints[i].pressureGpa - allPoints[i - 1].pressureGpa;
    if (dp === 0) continue;
    const slope = Math.abs((allPoints[i].tc - allPoints[i - 1].tc) / dp);
    if (slope > maxSlope) maxSlope = slope;
  }

  let sensitivityCategory: "high" | "moderate" | "low" = "low";
  if (maxSlope > SENSITIVITY_THRESHOLD * 2) sensitivityCategory = "high";
  else if (maxSlope > SENSITIVITY_THRESHOLD) sensitivityCategory = "moderate";

  const uncertainRegions = computeUncertainRegionsFromFeatures(normFormula, baseFeatures);

  const curve: PressureCurve = {
    formula: normFormula,
    points: allPoints,
    optimalPressureGpa,
    maxTc,
    isSensitive: maxSlope > SENSITIVITY_THRESHOLD,
    sensitivityCategory,
    maxDTcDP: maxSlope,
    uncertainRegions,
    computedAt: Date.now(),
  };

  evictCacheIfNeeded();
  pressureCurveCache.set(normFormula, curve);

  return curve;
}

export function findOptimalPressure(formula: string): { optimalPressureGpa: number; maxTc: number; curve: PressureCurve } {
  const curve = predictPressureCurve(formula);
  return {
    optimalPressureGpa: curve.optimalPressureGpa,
    maxTc: curve.maxTc,
    curve,
  };
}

export function pressureSensitivity(formula: string): PressureSensitivityResult {
  const curve = predictPressureCurve(formula);
  const points = curve.points;

  if (points.length < 2) {
    return {
      formula: curve.formula,
      avgDTcDP: 0,
      maxDTcDP: 0,
      maxDTcDPPressure: 0,
      isSensitive: false,
      sensitivityCategory: "low",
    };
  }

  let totalSlope = 0;
  let maxSlopePressure = 0;
  let maxSlope = 0;

  for (let i = 1; i < points.length; i++) {
    const dp = points[i].pressureGpa - points[i - 1].pressureGpa;
    if (dp === 0) continue;
    const slope = Math.abs((points[i].tc - points[i - 1].tc) / dp);
    totalSlope += slope;
    if (slope > maxSlope) {
      maxSlope = slope;
      maxSlopePressure = (points[i].pressureGpa + points[i - 1].pressureGpa) / 2;
    }
  }

  return {
    formula: curve.formula,
    avgDTcDP: totalSlope / (points.length - 1),
    maxDTcDP: curve.maxDTcDP,
    maxDTcDPPressure: maxSlopePressure,
    isSensitive: curve.isSensitive,
    sensitivityCategory: curve.sensitivityCategory,
  };
}

export function getPressureCurveStats(): PressureCurveStats {
  const curves = Array.from(pressureCurveCache.values());

  if (curves.length === 0) {
    return {
      totalCurves: 0,
      avgMaxTc: 0,
      avgOptimalPressure: 0,
      highTcCount: 0,
      sensitiveCount: 0,
      pressureDistribution: [],
    };
  }

  const totalMaxTc = curves.reduce((s, c) => s + c.maxTc, 0);
  const totalOptPressure = curves.reduce((s, c) => s + c.optimalPressureGpa, 0);
  const highTcCount = curves.filter(c => c.maxTc > 77).length;

  let sensitiveCount = 0;
  for (const curve of curves) {
    if (curve.isSensitive) sensitiveCount++;
  }

  const ranges = [
    { range: "0-50 GPa", min: 0, max: 50 },
    { range: "50-100 GPa", min: 50, max: 100 },
    { range: "100-150 GPa", min: 100, max: 150 },
    { range: "150-200 GPa", min: 150, max: 200 },
    { range: "200-250 GPa", min: 200, max: 250 },
    { range: "250-350 GPa", min: 250, max: 350 },
  ];

  const pressureDistribution = ranges.map(r => ({
    range: r.range,
    count: curves.filter(c => c.optimalPressureGpa >= r.min && c.optimalPressureGpa < r.max).length,
  }));

  return {
    totalCurves: curves.length,
    avgMaxTc: totalMaxTc / curves.length,
    avgOptimalPressure: totalOptPressure / curves.length,
    highTcCount,
    sensitiveCount,
    pressureDistribution,
  };
}

export interface AdaptivePressureSample {
  formula: string;
  pressureGpa: number;
  uncertainty: number;
  reason: string;
}

export interface PressureExplorationStats {
  totalRegionsIdentified: number;
  totalAdaptiveSamples: number;
  avgRegionUncertainty: number;
  pressureFocusDistribution: { range: string; sampleCount: number }[];
  recentSamples: AdaptivePressureSample[];
}

const MAX_ADAPTIVE_HISTORY = 200;
const adaptiveSamplesRing = new Array<AdaptivePressureSample>(MAX_ADAPTIVE_HISTORY);
let ringHead = 0;
let ringCount = 0;
const pressureCoverageMap = new Map<string, Set<number>>();

function pushAdaptiveSample(sample: AdaptivePressureSample): void {
  adaptiveSamplesRing[ringHead] = sample;
  ringHead = (ringHead + 1) % MAX_ADAPTIVE_HISTORY;
  if (ringCount < MAX_ADAPTIVE_HISTORY) ringCount++;
}

function getRecentAdaptiveSamples(n: number): AdaptivePressureSample[] {
  const count = Math.min(n, ringCount);
  const result: AdaptivePressureSample[] = [];
  for (let i = 0; i < count; i++) {
    const idx = (ringHead - count + i + MAX_ADAPTIVE_HISTORY) % MAX_ADAPTIVE_HISTORY;
    result.push(adaptiveSamplesRing[idx]);
  }
  return result;
}

export function identifyUncertainPressureRegions(formula: string): UncertainPressureRegion[] {
  const curve = predictPressureCurve(formula);
  return curve.uncertainRegions;
}

export function generateAdaptivePressureSamples(formula: string, maxSamples: number = 8): AdaptivePressureSample[] {
  const regions = identifyUncertainPressureRegions(formula);
  const samples: AdaptivePressureSample[] = [];
  const existingCoverage = pressureCoverageMap.get(formula) ?? new Set<number>();

  if (regions.length === 0) {
    const defaults = [0, 50, 100, 150, 200, 250, 300, 350];
    for (const p of defaults) {
      if (samples.length >= maxSamples) break;
      if (!existingCoverage.has(p)) {
        samples.push({ formula, pressureGpa: p, uncertainty: 0.5, reason: "uniform-coverage" });
      }
    }
    return samples;
  }

  const totalUncertainty = regions.reduce((s, r) => s + r.maxUncertainty, 0);
  for (const region of regions) {
    const regionBudget = Math.max(1, Math.round((region.maxUncertainty / totalUncertainty) * maxSamples));
    const span = region.pressureEnd - region.pressureStart;
    const step = span > 0 ? Math.max(5, Math.round(span / (regionBudget + 1))) : 10;

    samples.push({
      formula,
      pressureGpa: region.peakPressure,
      uncertainty: region.maxUncertainty,
      reason: `peak-uncertainty (${region.maxUncertainty.toFixed(3)})`,
    });

    for (let offset = step; offset <= span / 2; offset += step) {
      if (samples.length >= maxSamples) break;
      const pLow = Math.max(0, region.peakPressure - offset);
      const pHigh = Math.min(PRESSURE_MAX, region.peakPressure + offset);

      if (!existingCoverage.has(pLow) && pLow !== region.peakPressure) {
        samples.push({ formula, pressureGpa: pLow, uncertainty: region.avgUncertainty, reason: `refine-low (region ${region.pressureStart}-${region.pressureEnd} GPa)` });
      }
      if (samples.length >= maxSamples) break;
      if (!existingCoverage.has(pHigh) && pHigh !== region.peakPressure) {
        samples.push({ formula, pressureGpa: pHigh, uncertainty: region.avgUncertainty, reason: `refine-high (region ${region.pressureStart}-${region.pressureEnd} GPa)` });
      }
    }
    if (samples.length >= maxSamples) break;
  }

  for (const s of samples) {
    pushAdaptiveSample(s);
    if (!pressureCoverageMap.has(formula)) pressureCoverageMap.set(formula, new Set());
    pressureCoverageMap.get(formula)!.add(s.pressureGpa);
  }

  return samples.slice(0, maxSamples);
}

export function recordPressureCoverage(formula: string, pressureGpa: number): void {
  if (!pressureCoverageMap.has(formula)) pressureCoverageMap.set(formula, new Set());
  pressureCoverageMap.get(formula)!.add(pressureGpa);
}

export function getPressureExplorationStats(): PressureExplorationStats {
  let totalRegions = 0;
  let totalUncertainty = 0;
  for (const curve of pressureCurveCache.values()) {
    totalRegions += curve.uncertainRegions.length;
    for (const r of curve.uncertainRegions) {
      totalUncertainty += r.avgUncertainty;
    }
  }

  const allSamples = getRecentAdaptiveSamples(ringCount);

  const ranges = [
    { range: "0-50 GPa", min: 0, max: 50 },
    { range: "50-100 GPa", min: 50, max: 100 },
    { range: "100-150 GPa", min: 100, max: 150 },
    { range: "150-200 GPa", min: 150, max: 200 },
    { range: "200-250 GPa", min: 200, max: 250 },
    { range: "250-350 GPa", min: 250, max: 350 },
  ];

  const distCounts = new Array(ranges.length).fill(0);
  for (const s of allSamples) {
    for (let i = 0; i < ranges.length; i++) {
      if (s.pressureGpa >= ranges[i].min && s.pressureGpa < ranges[i].max) {
        distCounts[i]++;
        break;
      }
    }
  }

  return {
    totalRegionsIdentified: totalRegions,
    totalAdaptiveSamples: ringCount,
    avgRegionUncertainty: totalRegions > 0 ? totalUncertainty / totalRegions : 0,
    pressureFocusDistribution: ranges.map((r, i) => ({
      range: r.range,
      sampleCount: distCounts[i],
    })),
    recentSamples: getRecentAdaptiveSamples(20),
  };
}

export function clearPressureCurveCache(): void {
  pressureCurveCache.clear();
}
