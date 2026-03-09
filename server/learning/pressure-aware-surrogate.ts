import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { gbPredict, gbPredictWithUncertainty } from "./gradient-boost";
import { getGNNPrediction, type GNNPrediction } from "./graph-neural-net";
import { computeEnthalpy } from "./enthalpy-stability";

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

export interface PressureCurve {
  formula: string;
  points: PressurePoint[];
  optimalPressureGpa: number;
  maxTc: number;
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

function predictAtPressure(formula: string, pressureGpa: number): PressurePoint {
  const mat = { pressureGpa } as any;
  const features = extractFeatures(formula, mat);

  const overridden: MLFeatureVector = { ...features, pressureGpa };

  const xgbResult = gbPredict(overridden, formula);
  const xgbTc = Math.max(0, xgbResult.tcPredicted);

  let gnnTc = 0;
  let gnnFormationEnergy = 0;
  let gnnBandgap = 0;

  try {
    const gnnResult: GNNPrediction = getGNNPrediction(formula);
    const pressureFactor = 1 + pressureGpa * 0.002;
    gnnTc = Math.max(0, gnnResult.predictedTc * pressureFactor);
    gnnFormationEnergy = gnnResult.formationEnergy + pressureGpa * 0.01;
    gnnBandgap = Math.max(0, gnnResult.bandgap - pressureGpa * 0.005);
  } catch {}

  const ensembleTc = xgbTc * 0.5 + gnnTc * 0.5;
  const formationEnergy = gnnFormationEnergy !== 0
    ? gnnFormationEnergy
    : (features.formationEnergy ?? 0) + pressureGpa * 0.01;
  const bandgap = gnnBandgap !== 0
    ? gnnBandgap
    : Math.max(0, (features.bandGap ?? 0) - pressureGpa * 0.005);

  let enthalpy = 0;
  let enthalpyStable = true;
  try {
    const hPoint = computeEnthalpy(formula, pressureGpa);
    enthalpy = hPoint.enthalpy;
    enthalpyStable = hPoint.isStable;
  } catch {
    enthalpy = formationEnergy + pressureGpa * 0.006;
    enthalpyStable = enthalpy < 0.5;
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

export function predictPressureCurve(formula: string): PressureCurve {
  const cached = pressureCurveCache.get(formula);
  if (cached) return cached;

  const points: PressurePoint[] = [];

  for (let p = PRESSURE_MIN; p <= PRESSURE_MAX; p += PRESSURE_STEP) {
    const point = predictAtPressure(formula, p);
    points.push(point);
  }

  let maxTc = 0;
  let optimalPressureGpa = 0;
  for (const pt of points) {
    if (pt.tc > maxTc) {
      maxTc = pt.tc;
      optimalPressureGpa = pt.pressureGpa;
    }
  }

  const curve: PressureCurve = {
    formula,
    points,
    optimalPressureGpa,
    maxTc,
    computedAt: Date.now(),
  };

  evictCacheIfNeeded();
  pressureCurveCache.set(formula, curve);

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
      formula,
      avgDTcDP: 0,
      maxDTcDP: 0,
      maxDTcDPPressure: 0,
      isSensitive: false,
      sensitivityCategory: "low",
    };
  }

  let totalSlope = 0;
  let maxSlope = 0;
  let maxSlopePressure = 0;

  for (let i = 1; i < points.length; i++) {
    const dp = points[i].pressureGpa - points[i - 1].pressureGpa;
    if (dp === 0) continue;
    const dTc = points[i].tc - points[i - 1].tc;
    const slope = Math.abs(dTc / dp);
    totalSlope += slope;
    if (slope > maxSlope) {
      maxSlope = slope;
      maxSlopePressure = (points[i].pressureGpa + points[i - 1].pressureGpa) / 2;
    }
  }

  const avgSlope = totalSlope / (points.length - 1);

  let sensitivityCategory: "high" | "moderate" | "low" = "low";
  if (maxSlope > SENSITIVITY_THRESHOLD * 2) sensitivityCategory = "high";
  else if (maxSlope > SENSITIVITY_THRESHOLD) sensitivityCategory = "moderate";

  return {
    formula,
    avgDTcDP: avgSlope,
    maxDTcDP: maxSlope,
    maxDTcDPPressure: maxSlopePressure,
    isSensitive: maxSlope > SENSITIVITY_THRESHOLD,
    sensitivityCategory,
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
    const sens = pressureSensitivity(curve.formula);
    if (sens.isSensitive) sensitiveCount++;
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

export interface UncertainPressureRegion {
  pressureStart: number;
  pressureEnd: number;
  avgUncertainty: number;
  maxUncertainty: number;
  peakPressure: number;
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

const adaptiveSamplesHistory: AdaptivePressureSample[] = [];
const MAX_ADAPTIVE_HISTORY = 200;
const pressureCoverageMap = new Map<string, Set<number>>();

export function identifyUncertainPressureRegions(formula: string): UncertainPressureRegion[] {
  const uncertainties: { pressure: number; uncertainty: number }[] = [];

  for (let p = 0; p <= PRESSURE_MAX; p += 20) {
    try {
      const mat = { pressureGpa: p } as any;
      const features = extractFeatures(formula, mat);
      const xgbResult = gbPredictWithUncertainty(features, formula);
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
    adaptiveSamplesHistory.push(s);
    if (!pressureCoverageMap.has(formula)) pressureCoverageMap.set(formula, new Set());
    pressureCoverageMap.get(formula)!.add(s.pressureGpa);
  }
  while (adaptiveSamplesHistory.length > MAX_ADAPTIVE_HISTORY) adaptiveSamplesHistory.shift();

  return samples.slice(0, maxSamples);
}

export function recordPressureCoverage(formula: string, pressureGpa: number): void {
  if (!pressureCoverageMap.has(formula)) pressureCoverageMap.set(formula, new Set());
  pressureCoverageMap.get(formula)!.add(pressureGpa);
}

export function getPressureExplorationStats(): PressureExplorationStats {
  const allRegions: UncertainPressureRegion[] = [];
  for (const formula of pressureCurveCache.keys()) {
    allRegions.push(...identifyUncertainPressureRegions(formula));
  }

  const ranges = [
    { range: "0-50 GPa", min: 0, max: 50 },
    { range: "50-100 GPa", min: 50, max: 100 },
    { range: "100-150 GPa", min: 100, max: 150 },
    { range: "150-200 GPa", min: 150, max: 200 },
    { range: "200-250 GPa", min: 200, max: 250 },
    { range: "250-350 GPa", min: 250, max: 350 },
  ];

  return {
    totalRegionsIdentified: allRegions.length,
    totalAdaptiveSamples: adaptiveSamplesHistory.length,
    avgRegionUncertainty: allRegions.length > 0 ? allRegions.reduce((s, r) => s + r.avgUncertainty, 0) / allRegions.length : 0,
    pressureFocusDistribution: ranges.map(r => ({
      range: r.range,
      sampleCount: adaptiveSamplesHistory.filter(s => s.pressureGpa >= r.min && s.pressureGpa < r.max).length,
    })),
    recentSamples: adaptiveSamplesHistory.slice(-20),
  };
}

export function clearPressureCurveCache(): void {
  pressureCurveCache.clear();
}
