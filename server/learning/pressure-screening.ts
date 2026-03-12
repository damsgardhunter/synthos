import { getElementData } from "./elemental-data";
import { extractFeatures } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { getMetalElements, getElectronegativitySpread } from "./utils";

export interface FastPressureEstimate {
  formula: string;
  pressureGpa: number;
  volumeRatio: number;
  estimatedVolume: number;
  referenceVolume: number;
  bulkModulus: number;
  estimatedTc: number;
  estimatedBandgap: number;
  estimatedStability: number;
  screeningScore: number;
  passesPrescreen: boolean;
}

export interface PressureClusterDef {
  id: string;
  label: string;
  minGpa: number;
  maxGpa: number;
  physics: string;
  expectedFamilies: string[];
  baseWeight: number;
}

export interface ClusterAnalytics {
  weight: number;
  discoveryCount: number;
  bestTcFound: number;
  totalScreened: number;
  hitRate: number;
}

export interface PressureCluster extends PressureClusterDef, ClusterAnalytics {}

export interface PressureClusterStats {
  clusters: PressureCluster[];
  totalDiscoveries: number;
  mostProductiveCluster: string;
  explorationBias: Record<string, number>;
  recentAssignments: { formula: string; cluster: string; tc: number }[];
}

const CLUSTER_DEFS: readonly PressureClusterDef[] = [
  { id: "ambient", label: "Ambient / Low Pressure", minGpa: 0, maxGpa: 10, physics: "Conventional BCS metals, cuprates, pnictides at ambient conditions", expectedFamilies: ["cuprate", "pnictide", "conventional", "heavy-fermion", "organic"], baseWeight: 1.0 },
  { id: "low-moderate", label: "Low-Moderate Pressure", minGpa: 10, maxGpa: 50, physics: "Enhanced phonon coupling, incipient structural transitions, near-ambient SC candidates", expectedFamilies: ["conventional", "pnictide", "boride", "chalcogenide"], baseWeight: 1.2 },
  { id: "structural-transition", label: "Structural Transition Zone", minGpa: 50, maxGpa: 100, physics: "Structural phase transitions, metallization of semiconductors, enhanced e-ph coupling at transitions", expectedFamilies: ["chalcogenide", "conventional", "pnictide"], baseWeight: 1.0 },
  { id: "high-pressure", label: "High Pressure", minGpa: 100, maxGpa: 150, physics: "Dense metallic phases, incipient hydride formation, strong lattice hardening", expectedFamilies: ["hydride", "conventional"], baseWeight: 0.9 },
  { id: "hydride-onset", label: "Hydride Superconductor Onset", minGpa: 150, maxGpa: 200, physics: "Hydrogen-rich clathrate/cage structures stabilize, strong phonon-mediated pairing", expectedFamilies: ["hydride"], baseWeight: 1.3 },
  { id: "hydride-peak", label: "Peak Hydride Superconductivity", minGpa: 200, maxGpa: 300, physics: "Optimal H-cage phonon frequencies, record Tc hydrides (LaH10, YH9)", expectedFamilies: ["hydride"], baseWeight: 1.1 },
  { id: "extreme", label: "Extreme Pressure", minGpa: 300, maxGpa: 350, physics: "Ultra-dense phases, metallic hydrogen regime, exotic pairing", expectedFamilies: ["hydride"], baseWeight: 0.6 },
];

const CLUSTER_BOUNDARIES = CLUSTER_DEFS.map(c => c.maxGpa);

const clusterAnalytics = new Map<string, ClusterAnalytics>();
function getAnalytics(id: string, baseWeight: number): ClusterAnalytics {
  let a = clusterAnalytics.get(id);
  if (!a) {
    a = { weight: baseWeight, discoveryCount: 0, bestTcFound: 0, totalScreened: 0, hitRate: 0 };
    clusterAnalytics.set(id, a);
  }
  return a;
}

function getMergedClusters(): PressureCluster[] {
  return CLUSTER_DEFS.map(def => ({ ...def, ...getAnalytics(def.id, def.baseWeight) }));
}

const MAX_RECENT = 100;
const recentAssignmentsBuf: { formula: string; cluster: string; tc: number }[] = new Array(MAX_RECENT);
let raBufHead = 0;
let raBufSize = 0;

function pushRecentAssignment(entry: { formula: string; cluster: string; tc: number }): void {
  recentAssignmentsBuf[raBufHead] = entry;
  raBufHead = (raBufHead + 1) % MAX_RECENT;
  if (raBufSize < MAX_RECENT) raBufSize++;
}

function getRecentAssignments(n: number): { formula: string; cluster: string; tc: number }[] {
  const count = Math.min(n, raBufSize);
  const result: { formula: string; cluster: string; tc: number }[] = [];
  let idx = (raBufHead - count + MAX_RECENT) % MAX_RECENT;
  for (let i = 0; i < count; i++) {
    result.push(recentAssignmentsBuf[idx]);
    idx = (idx + 1) % MAX_RECENT;
  }
  return result;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function estimateBulkModulusFromCounts(counts: Record<string, number>): number {
  const elements = Object.keys(counts);

  let weightedB = 0;
  let totalCount = 0;

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const n = counts[el] || 1;
    const elBulk = data.bulkModulus ?? 50;
    weightedB += elBulk * n;
    totalCount += n;
  }

  return totalCount > 0 ? Math.max(10, weightedB / totalCount) : 50;
}

function pressureCorrectedRadius(el: string, r0: number, pressureGpa: number): number {
  if (pressureGpa <= 100) return r0;
  const excessP = pressureGpa - 100;
  if (el === "H") {
    const logScale = 1.0 / (1.0 + 0.18 * Math.log(1 + excessP / 20));
    const minR = 0.25;
    return r0 * Math.max(minR, logScale);
  }
  const scale = 1.0 / (1.0 + excessP * 0.001);
  return r0 * Math.max(0.55, scale);
}

function inferLatticeType(elements: string[], counts: Record<string, number>, totalAtoms: number): "fcc" | "bcc" | "hcp" | "cage" | "other" {
  const metalEls = getMetalElements(elements);
  const metalFrac = metalEls.reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;
  const hFrac = (counts["H"] || 0) / totalAtoms;
  if (hFrac > 0.6) return "cage";
  const FCC = new Set(["Cu", "Al", "Ni", "Ag", "Au", "Pt", "Pd", "Pb", "Ca", "Sr", "Rh", "Ir"]);
  const HCP = new Set(["Ti", "Zr", "Hf", "Mg", "Zn", "Cd", "Co", "Ru", "Os", "Y", "Sc", "La"]);
  if (metalFrac >= 0.9 && metalEls.length <= 2) {
    const primary = metalEls.sort((a, b) => (counts[b] || 0) - (counts[a] || 0))[0];
    if (FCC.has(primary)) return "fcc";
    if (HCP.has(primary)) return "hcp";
    return "bcc";
  }
  if (metalFrac > 0.5) return "bcc";
  return "other";
}

function latticeVolPerAtom(r: number, latticeType: string): number {
  switch (latticeType) {
    case "fcc": {
      const a = 2 * Math.SQRT2 * r;
      return (a * a * a) / 4;
    }
    case "bcc": {
      const a = (4 * r) / Math.sqrt(3);
      return (a * a * a) / 2;
    }
    case "hcp": {
      const a = 2 * r;
      const c = a * 1.633;
      return (Math.sqrt(3) / 2) * a * a * c / 2;
    }
    case "cage":
      return Math.pow(2.8 * r, 3);
    default:
      return Math.pow(2.3 * r, 3);
  }
}

function estimateReferenceVolumeFromCounts(counts: Record<string, number>, pressureGpa: number = 0): number {
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const latticeType = inferLatticeType(elements, counts, totalAtoms);

  let V0 = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.latticeConstant) {
      V0 += Math.pow(data.latticeConstant, 3) * (counts[el] || 1);
    } else if (data) {
      const r = pressureCorrectedRadius(el, data.atomicRadius, pressureGpa);
      V0 += latticeVolPerAtom(r, latticeType) * (counts[el] || 1);
    } else {
      V0 += 30 * (counts[el] || 1);
    }
  }

  return Math.max(10, V0 / totalAtoms);
}

export function fastVolumeAtPressure(formula: string, pressureGpa: number, precomputedCounts?: Record<string, number>): {
  volumeRatio: number;
  estimatedVolume: number;
  referenceVolume: number;
  bulkModulus: number;
} {
  const counts = precomputedCounts ?? parseFormulaCounts(formula);
  const B0 = estimateBulkModulusFromCounts(counts);
  const V0 = estimateReferenceVolumeFromCounts(counts, pressureGpa);

  const Bp = 4.0;
  const ratio = Math.max(0.3, Math.pow(1 + Bp * pressureGpa / Math.max(10, B0), -1 / Bp));
  const volume = V0 * ratio;

  return {
    volumeRatio: Math.round(ratio * 10000) / 10000,
    estimatedVolume: Math.round(volume * 1000) / 1000,
    referenceVolume: Math.round(V0 * 1000) / 1000,
    bulkModulus: Math.round(B0 * 10) / 10,
  };
}

export function fastPressureScreen(
  formula: string,
  pressureGpa: number,
  tcThreshold: number = 10,
  precomputedCounts?: Record<string, number>
): FastPressureEstimate {
  const counts = precomputedCounts ?? parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const vol = fastVolumeAtPressure(formula, pressureGpa, counts);

  let estimatedTc = 0;
  try {
    const mat = { pressureGpa } as any;
    const features = extractFeatures(formula, mat);
    if (features) {
      const xgbResult = gbPredict(features, formula);
      if (xgbResult && Number.isFinite(xgbResult.tcPredicted)) {
        estimatedTc = Math.max(0, xgbResult.tcPredicted);
      }
    }
  } catch {
    estimatedTc = 0;
  }

  const hFrac = (counts["H"] || 0) / totalAtoms;
  const isHydride = hFrac > 0.5;

  const pOpt = isHydride ? 200 : 100;
  const sigma = isHydride ? 150 : 80;
  const domeFactor = Math.exp(-0.5 * Math.pow((pressureGpa - pOpt) / sigma, 2));
  const pressureBoost = 1 + domeFactor * (isHydride ? 1.5 : 0.6);
  estimatedTc *= pressureBoost;

  if (isHydride && pressureGpa >= 100) {
    estimatedTc *= 1 + (hFrac - 0.5) * domeFactor * 0.5;
  }

  const enSpread = getElectronegativitySpread(counts);
  const ionicResistance = 1 + enSpread * 0.8;
  const estimatedBandgap = Math.max(0, (1.5 * ionicResistance) - pressureGpa * 0.01 * vol.volumeRatio);

  const volumeStability = vol.volumeRatio > 0.55 ? 1.0 : vol.volumeRatio > 0.5 ? 0.5 : 0.0;
  if (volumeStability === 0) {
    estimatedTc = 0;
  }
  const pressureFeasibility = pressureGpa <= 50 ? 1.0 : pressureGpa <= 150 ? 0.7 : pressureGpa <= 300 ? 0.4 : 0.2;
  const estimatedStability = volumeStability * 0.6 + pressureFeasibility * 0.4;

  const tcNorm = Math.min(1.0, estimatedTc / 300);
  const screeningScore = 0.5 * tcNorm + 0.3 * estimatedStability + 0.2 * (1 - pressureGpa / 500);

  return {
    formula,
    pressureGpa,
    volumeRatio: vol.volumeRatio,
    estimatedVolume: vol.estimatedVolume,
    referenceVolume: vol.referenceVolume,
    bulkModulus: vol.bulkModulus,
    estimatedTc: Math.round(estimatedTc * 10) / 10,
    estimatedBandgap: Math.round(estimatedBandgap * 1000) / 1000,
    estimatedStability,
    screeningScore: Math.round(screeningScore * 10000) / 10000,
    passesPrescreen: estimatedTc >= tcThreshold && estimatedStability > 0.3 && vol.volumeRatio > 0.5,
  };
}

export function batchPressureScreen(
  formula: string,
  pressures?: number[],
  tcThreshold: number = 10
): FastPressureEstimate[] {
  const pts = pressures ?? [0, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350];
  const counts = parseFormulaCounts(formula);
  return pts.map(p => fastPressureScreen(formula, p, tcThreshold, counts));
}

export function findBestScreeningPressure(formula: string): {
  bestPressure: number;
  bestTc: number;
  bestScore: number;
  screenResults: FastPressureEstimate[];
} {
  const results = batchPressureScreen(formula);
  let bestIdx = 0;
  let bestScore = -1;
  for (let i = 0; i < results.length; i++) {
    if (results[i].screeningScore > bestScore) {
      bestScore = results[i].screeningScore;
      bestIdx = i;
    }
  }

  return {
    bestPressure: results[bestIdx].pressureGpa,
    bestTc: results[bestIdx].estimatedTc,
    bestScore: results[bestIdx].screeningScore,
    screenResults: results,
  };
}

function findClusterIndex(pressureGpa: number): number {
  let lo = 0;
  let hi = CLUSTER_BOUNDARIES.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (pressureGpa < CLUSTER_BOUNDARIES[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return Math.min(lo, CLUSTER_DEFS.length - 1);
}

export function assignPressureCluster(pressureGpa: number): PressureCluster {
  const idx = findClusterIndex(pressureGpa);
  const def = CLUSTER_DEFS[idx];
  return { ...def, ...getAnalytics(def.id, def.baseWeight) };
}

export function recordClusterDiscovery(
  formula: string,
  pressureGpa: number,
  tc: number,
  isSuccess: boolean
): void {
  const idx = findClusterIndex(pressureGpa);
  const def = CLUSTER_DEFS[idx];
  const analytics = getAnalytics(def.id, def.baseWeight);
  analytics.totalScreened++;

  if (isSuccess && tc > 0) {
    analytics.discoveryCount++;
    if (tc > analytics.bestTcFound) {
      analytics.bestTcFound = tc;
    }
  }

  analytics.hitRate = analytics.totalScreened > 0
    ? analytics.discoveryCount / analytics.totalScreened
    : 0;

  pushRecentAssignment({ formula, cluster: def.id, tc });

  rebalanceClusterWeights();
}

function rebalanceClusterWeights(): void {
  const allAnalytics = CLUSTER_DEFS.map(def => getAnalytics(def.id, def.baseWeight));
  const totalScreened = allAnalytics.reduce((s, a) => s + a.totalScreened, 0);
  if (totalScreened < 10) return;

  const totalDiscoveries = allAnalytics.reduce((s, a) => s + a.discoveryCount, 0);
  const useDiscoveryMode = totalDiscoveries >= 5;

  for (const def of CLUSTER_DEFS) {
    const analytics = getAnalytics(def.id, def.baseWeight);
    if (analytics.totalScreened < 3) continue;

    if (useDiscoveryMode) {
      const avgHitRate = allAnalytics.reduce((s, a) => s + a.hitRate, 0) / allAnalytics.length;
      const hitRateRatio = avgHitRate > 0 ? analytics.hitRate / avgHitRate : 1;
      const tcBonus = analytics.bestTcFound > 77 ? 0.2 : 0;
      const rawWeight = 0.6 + hitRateRatio * 0.5 + tcBonus;

      if (def.id === "ambient" || def.id === "low-moderate") {
        analytics.weight = Math.max(0.8, Math.min(2.0, rawWeight * 1.2));
      } else if (def.id === "extreme") {
        analytics.weight = Math.max(0.3, Math.min(1.5, rawWeight * 0.7));
      } else {
        analytics.weight = Math.max(0.4, Math.min(2.0, rawWeight));
      }
    } else {
      const failRate = 1 - analytics.hitRate;
      const explorationPenalty = failRate * Math.min(1.0, analytics.totalScreened / 20);
      const uncertaintyBoost = analytics.totalScreened < 10 ? 0.3 : 0;
      const rawWeight = def.baseWeight * (1 - explorationPenalty * 0.4) + uncertaintyBoost;

      if (def.id === "ambient" || def.id === "low-moderate") {
        analytics.weight = Math.max(0.6, Math.min(2.0, rawWeight));
      } else if (def.id === "extreme") {
        analytics.weight = Math.max(0.2, Math.min(1.5, rawWeight));
      } else {
        analytics.weight = Math.max(0.3, Math.min(2.0, rawWeight));
      }
    }
  }
}

export function getClusterExplorationBias(): Record<string, number> {
  const merged = getMergedClusters();
  const totalWeight = merged.reduce((s, c) => s + c.weight, 0);
  const bias: Record<string, number> = {};
  for (const cluster of merged) {
    bias[cluster.id] = Math.round((cluster.weight / totalWeight) * 10000) / 10000;
  }
  return bias;
}

export function samplePressureFromClusters(count: number = 10): number[] {
  const merged = getMergedClusters();
  if (merged.length === 0) {
    return Array.from({ length: count }, () => Math.round(Math.random() * 350));
  }

  const rawWeights = merged.map(c => c.weight);
  const totalWeight = rawWeights.reduce((s, w) => s + w, 0);
  if (totalWeight <= 0) {
    return Array.from({ length: count }, () => Math.round(Math.random() * 350));
  }

  const normalizedWeights = rawWeights.map(w => w / totalWeight);

  const samples: number[] = [];
  for (let i = 0; i < count; i++) {
    let r = Math.random();
    let selectedIdx = normalizedWeights.length - 1;
    for (let j = 0; j < normalizedWeights.length; j++) {
      r -= normalizedWeights[j];
      if (r <= 0) {
        selectedIdx = j;
        break;
      }
    }

    const def = CLUSTER_DEFS[selectedIdx];
    const p = def.minGpa + Math.random() * (def.maxGpa - def.minGpa);
    samples.push(Math.round(p));
  }

  return samples;
}

export function getPressureClusterStats(): PressureClusterStats {
  const merged = getMergedClusters();
  const totalDiscoveries = merged.reduce((s, c) => s + c.discoveryCount, 0);

  let mostProductive = "ambient";
  let bestRate = 0;
  for (const cluster of merged) {
    if (cluster.hitRate > bestRate && cluster.totalScreened >= 3) {
      bestRate = cluster.hitRate;
      mostProductive = cluster.id;
    }
  }

  return {
    clusters: merged.map(c => ({
      ...c,
      hitRate: Math.round(c.hitRate * 10000) / 10000,
      weight: Math.round(c.weight * 10000) / 10000,
    })),
    totalDiscoveries,
    mostProductiveCluster: mostProductive,
    explorationBias: getClusterExplorationBias(),
    recentAssignments: getRecentAssignments(20),
  };
}
