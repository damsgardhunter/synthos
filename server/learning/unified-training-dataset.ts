import { getGroundTruthDataset, getGroundTruthSummary, type GroundTruthDatapoint } from "./ground-truth-store";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { getAllPhysicsResults, getDerivedFeatures, type PhysicsResult, type DerivedFeatures } from "./physics-results-store";

export interface UnifiedTrainingRecord {
  formula: string;
  pressure: number;
  Tc: number | null;
  lambda: number | null;
  omegaLog: number | null;
  dosAtEF: number | null;
  bandGap: number | null;
  formationEnergy: number | null;
  phononStable: boolean;
  muStar: number | null;
  isStrongCoupling: boolean;
  source: string;
  tier: string;
  derived: DerivedFeatures | null;
  hasPhysicsResult: boolean;
  timestamp: number;
}

const TIER_RANK: Record<string, number> = {
  "surrogate": 1,
  "xtb": 2,
  "full-dft": 3,
};

export interface DatasetSnapshot {
  totalRecords: number;
  withPhysicsResults: number;
  withDerivedFeatures: number;
  sourceBreakdown: Record<string, number>;
  tierBreakdown: Record<string, number>;
  tcDistribution: { min: number; max: number; mean: number; median: number; p90: number };
  featureCoverage: Record<string, number>;
  snapshotTimestamp: number;
  deltaRecords: number;
}

const snapshotHistory: DatasetSnapshot[] = [];
const MAX_SNAPSHOTS = 100;

let cachedDataset: UnifiedTrainingRecord[] | null = null;
let cachedSnapshot: DatasetSnapshot | null = null;
let dataVersion = 0;
let cachedDataVersion = -1;
let cachedSnapshotVersion = -1;

const derivedCache = new Map<string, DerivedFeatures | null>();

export function invalidateUnifiedCache(): void {
  dataVersion++;
}

function hasPhysicsFields(rec: { lambda: number | null; omegaLog: number | null; dosAtEF: number | null }): boolean {
  return rec.lambda !== null || rec.omegaLog !== null || rec.dosAtEF !== null;
}

function pressureKey(pressure: number): string {
  return pressure.toFixed(2);
}

function getCachedDerived(formula: string): DerivedFeatures | null {
  if (derivedCache.has(formula)) {
    return derivedCache.get(formula)!;
  }
  const d = getDerivedFeatures(formula);
  derivedCache.set(formula, d ?? null);
  return d ?? null;
}

function rebuildIfDirty(): UnifiedTrainingRecord[] {
  if (cachedDataset !== null && cachedDataVersion === dataVersion) {
    return cachedDataset;
  }

  const records = new Map<string, UnifiedTrainingRecord>();

  for (const sd of SUPERCON_TRAINING_DATA) {
    const tc = (sd as any).tc ?? (sd as any).Tc;
    const key = `${sd.formula}|${pressureKey(0)}`;
    records.set(key, {
      formula: sd.formula,
      pressure: 0,
      Tc: typeof tc === "number" ? tc : null,
      lambda: null,
      omegaLog: null,
      dosAtEF: null,
      bandGap: null,
      formationEnergy: null,
      phononStable: true,
      muStar: null,
      isStrongCoupling: false,
      source: "supercon-seed",
      tier: "external",
      derived: null,
      hasPhysicsResult: false,
      timestamp: 0,
    });
  }

  const groundTruth = getGroundTruthDataset();
  for (const dp of groundTruth) {
    const key = `${dp.formula}|${pressureKey(dp.pressure)}`;
    records.set(key, {
      formula: dp.formula,
      pressure: dp.pressure,
      Tc: dp.Tc,
      lambda: dp.lambda,
      omegaLog: dp.omega_log,
      dosAtEF: dp.DOS_EF,
      bandGap: dp.band_gap,
      formationEnergy: dp.formation_energy,
      phononStable: dp.phonon_stable,
      muStar: dp.mu_star,
      isStrongCoupling: dp.is_strong_coupling,
      source: dp.source,
      tier: dp.source === "full-dft" ? "full-dft" : dp.source === "xtb" ? "xtb" : "surrogate",
      derived: null,
      hasPhysicsResult: hasPhysicsFields({ lambda: dp.lambda, omegaLog: dp.omega_log, dosAtEF: dp.DOS_EF }),
      timestamp: dp.timestamp,
    });
  }

  const physicsResults = getAllPhysicsResults();
  for (const pr of physicsResults) {
    const key = `${pr.formula}|${pressureKey(pr.pressure)}`;
    const existing = records.get(key);
    if (existing) {
      const existingRank = TIER_RANK[existing.tier] ?? 0;
      const incomingRank = TIER_RANK[pr.tier] ?? 0;

      existing.lambda = pr.lambda;
      existing.omegaLog = pr.omegaLog;
      existing.dosAtEF = pr.dosAtEF;
      existing.phononStable = pr.phononStable;
      existing.muStar = pr.muStar;
      existing.isStrongCoupling = pr.isStrongCoupling;
      existing.formationEnergy = pr.formationEnergy;
      existing.bandGap = pr.bandGap;
      existing.hasPhysicsResult = true;
      if (incomingRank >= existingRank) {
        existing.tier = pr.tier;
      }
      if (pr.tc > 0) existing.Tc = pr.tc;
    } else {
      records.set(key, {
        formula: pr.formula,
        pressure: pr.pressure,
        Tc: pr.tc > 0 ? pr.tc : null,
        lambda: pr.lambda,
        omegaLog: pr.omegaLog,
        dosAtEF: pr.dosAtEF,
        bandGap: pr.bandGap,
        formationEnergy: pr.formationEnergy,
        phononStable: pr.phononStable,
        muStar: pr.muStar,
        isStrongCoupling: pr.isStrongCoupling,
        source: "physics-result",
        tier: pr.tier,
        derived: null,
        hasPhysicsResult: true,
        timestamp: pr.timestamp,
      });
    }
  }

  for (const [, record] of records) {
    record.derived = getCachedDerived(record.formula);
  }

  cachedDataset = Array.from(records.values());
  cachedDataVersion = dataVersion;
  cachedSnapshot = null;
  cachedSnapshotVersion = -1;
  return cachedDataset;
}

export function buildUnifiedDataset(): UnifiedTrainingRecord[] {
  return rebuildIfDirty();
}

function computeSnapshotFromDataset(dataset: UnifiedTrainingRecord[]): DatasetSnapshot {
  if (cachedSnapshot !== null && cachedSnapshotVersion === dataVersion) {
    return cachedSnapshot;
  }

  const sourceBreakdown: Record<string, number> = {};
  const tierBreakdown: Record<string, number> = {};
  let withPhysics = 0;
  let withDerived = 0;
  const tcValues: number[] = [];
  const featureCounts: Record<string, number> = {
    lambda: 0, omegaLog: 0, dosAtEF: 0, bandGap: 0,
    formationEnergy: 0, muStar: 0,
  };

  for (const r of dataset) {
    sourceBreakdown[r.source] = (sourceBreakdown[r.source] ?? 0) + 1;
    tierBreakdown[r.tier] = (tierBreakdown[r.tier] ?? 0) + 1;
    if (r.hasPhysicsResult) withPhysics++;
    if (r.derived) withDerived++;
    if (r.Tc !== null && r.Tc > 0) tcValues.push(r.Tc);
    if (r.lambda !== null) featureCounts.lambda++;
    if (r.omegaLog !== null) featureCounts.omegaLog++;
    if (r.dosAtEF !== null) featureCounts.dosAtEF++;
    if (r.bandGap !== null) featureCounts.bandGap++;
    if (r.formationEnergy !== null) featureCounts.formationEnergy++;
    if (r.muStar !== null) featureCounts.muStar++;
  }

  tcValues.sort((a, b) => a - b);
  const n = tcValues.length || 1;
  const tcDist = {
    min: tcValues.length > 0 ? tcValues[0] : 0,
    max: tcValues.length > 0 ? tcValues[tcValues.length - 1] : 0,
    mean: tcValues.reduce((s, v) => s + v, 0) / n,
    median: tcValues.length > 0 ? tcValues[Math.floor(n / 2)] : 0,
    p90: tcValues.length > 0 ? tcValues[Math.floor(n * 0.9)] : 0,
  };

  const prevTotal = snapshotHistory.length > 0
    ? snapshotHistory[snapshotHistory.length - 1].totalRecords
    : dataset.length;

  const snapshot: DatasetSnapshot = {
    totalRecords: dataset.length,
    withPhysicsResults: withPhysics,
    withDerivedFeatures: withDerived,
    sourceBreakdown,
    tierBreakdown,
    tcDistribution: tcDist,
    featureCoverage: featureCounts,
    snapshotTimestamp: Date.now(),
    deltaRecords: dataset.length - prevTotal,
  };

  cachedSnapshot = snapshot;
  cachedSnapshotVersion = dataVersion;
  return snapshot;
}

export function getUnifiedDatasetStats(): DatasetSnapshot {
  const dataset = rebuildIfDirty();
  const snapshot = computeSnapshotFromDataset(dataset);

  snapshotHistory.push(snapshot);
  if (snapshotHistory.length > MAX_SNAPSHOTS) {
    snapshotHistory.splice(0, snapshotHistory.length - MAX_SNAPSHOTS);
  }

  return snapshot;
}

export function getSnapshotHistory(): DatasetSnapshot[] {
  return [...snapshotHistory];
}

export function getUnifiedDatasetForLLM(): string {
  const stats = getUnifiedDatasetStats();
  const lines: string[] = [
    "=== Unified Training Dataset ===",
    `Total records: ${stats.totalRecords} (delta: ${stats.deltaRecords >= 0 ? "+" : ""}${stats.deltaRecords})`,
    `With physics results: ${stats.withPhysicsResults}`,
    `With derived features: ${stats.withDerivedFeatures}`,
    `Sources: ${Object.entries(stats.sourceBreakdown).map(([k, v]) => `${k}=${v}`).join(", ")}`,
    `Tiers: ${Object.entries(stats.tierBreakdown).map(([k, v]) => `${k}=${v}`).join(", ")}`,
    `Tc: min=${stats.tcDistribution.min.toFixed(1)}K, max=${stats.tcDistribution.max.toFixed(1)}K, mean=${stats.tcDistribution.mean.toFixed(1)}K, median=${stats.tcDistribution.median.toFixed(1)}K, p90=${stats.tcDistribution.p90.toFixed(1)}K`,
    `Feature coverage: ${Object.entries(stats.featureCoverage).map(([k, v]) => `${k}=${v}/${stats.totalRecords}`).join(", ")}`,
  ];
  return lines.join("\n");
}

export function getTrainingSlice(
  filter?: { minTc?: number; maxTc?: number; source?: string; tier?: string; requirePhysics?: boolean }
): UnifiedTrainingRecord[] {
  let dataset = rebuildIfDirty();
  if (filter) {
    if (filter.minTc !== undefined) dataset = dataset.filter(r => r.Tc !== null && r.Tc >= filter.minTc!);
    if (filter.maxTc !== undefined) dataset = dataset.filter(r => r.Tc !== null && r.Tc <= filter.maxTc!);
    if (filter.source) dataset = dataset.filter(r => r.source === filter.source);
    if (filter.tier) dataset = dataset.filter(r => r.tier === filter.tier);
    if (filter.requirePhysics) dataset = dataset.filter(r => r.hasPhysicsResult);
  }
  return dataset;
}
