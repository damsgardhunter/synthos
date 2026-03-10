import { getTrainingData } from "./crystal-structure-dataset";
import { SUPERCON_TRAINING_DATA } from "../learning/supercon-dataset";

export interface MotifReward {
  prototype: string;
  crystalSystem: string;
  spacegroup: string | null;
  avgLambda: number;
  maxLambda: number;
  avgTc: number;
  maxTc: number;
  phononStabilityRate: number;
  totalEvals: number;
  successCount: number;
  rewardScore: number;
}

interface MotifRecord {
  lambdaSum: number;
  maxLambda: number;
  tcSum: number;
  maxTc: number;
  phononStableCount: number;
  totalEvals: number;
  successCount: number;
}

function motifKey(prototype?: string, system?: string, spacegroup?: string | null): string {
  const p = (prototype || "unknown").toLowerCase();
  const s = (system || "unknown").toLowerCase();
  const sg = spacegroup ? spacegroup.toLowerCase() : "*";
  return `${p}::${s}::${sg}`;
}

const motifDB = new Map<string, MotifRecord>();
let seeded = false;

function ensureRecord(key: string): MotifRecord {
  let rec = motifDB.get(key);
  if (!rec) {
    rec = { lambdaSum: 0, maxLambda: 0, tcSum: 0, maxTc: 0, phononStableCount: 0, totalEvals: 0, successCount: 0 };
    motifDB.set(key, rec);
  }
  return rec;
}

function computeRewardScore(rec: MotifRecord): number {
  const maxTcNorm = Math.min(1, rec.maxTc / 300);
  const maxLambdaNorm = Math.min(1, rec.maxLambda / 3);
  const phononRate = rec.totalEvals > 0 ? rec.phononStableCount / rec.totalEvals : 0;
  const evalBonus = Math.min(1, Math.log(rec.totalEvals + 1) / 5);

  const score = 0.35 * maxTcNorm + 0.25 * maxLambdaNorm + 0.25 * phononRate + 0.15 * evalBonus;
  return Math.max(0, Math.min(1, score));
}

function parseKeyParts(key: string): { prototype: string; crystalSystem: string; spacegroup: string | null } {
  const [prototype, crystalSystem, spacegroup] = key.split("::");
  return { prototype, crystalSystem, spacegroup: spacegroup === "*" ? null : spacegroup };
}

function seedFromDatasets(): void {
  if (seeded) return;
  seeded = true;

  const crystalData = getTrainingData();
  const superconMap = new Map<string, { tc: number; lambda: number }>();

  for (const entry of SUPERCON_TRAINING_DATA) {
    if (entry.isSuperconductor && entry.tc > 0) {
      superconMap.set(entry.formula, { tc: entry.tc, lambda: entry.lambda ?? 0 });
    }
  }

  for (const crystal of crystalData) {
    const sc = superconMap.get(crystal.formula);
    if (!sc) continue;

    const key = motifKey(crystal.prototype, crystal.crystalSystem, crystal.spacegroupSymbol);
    const rec = ensureRecord(key);
    rec.totalEvals++;
    rec.tcSum += sc.tc;
    rec.maxTc = Math.max(rec.maxTc, sc.tc);
    rec.lambdaSum += sc.lambda;
    rec.maxLambda = Math.max(rec.maxLambda, sc.lambda);
    rec.phononStableCount++;
    if (sc.tc > 5) rec.successCount++;
  }
}

export function recordStructureOutcome(
  formula: string,
  prototype: string | null | undefined,
  system: string | null | undefined,
  spacegroup: string | null | undefined,
  lambda: number,
  tc: number,
  phononStable: boolean
): void {
  seedFromDatasets();

  const key = motifKey(prototype ?? undefined, system ?? undefined, spacegroup);
  const rec = ensureRecord(key);
  rec.totalEvals++;
  rec.tcSum += tc;
  rec.maxTc = Math.max(rec.maxTc, tc);
  rec.lambdaSum += lambda;
  rec.maxLambda = Math.max(rec.maxLambda, lambda);
  if (phononStable) rec.phononStableCount++;
  if (tc > 5) rec.successCount++;
}

export function getStructureReward(
  prototype?: string,
  system?: string,
  spacegroup?: string | null
): MotifReward | null {
  seedFromDatasets();

  const key = motifKey(prototype, system, spacegroup);
  const rec = motifDB.get(key);
  if (!rec || rec.totalEvals === 0) return null;

  const parts = parseKeyParts(key);
  return {
    prototype: parts.prototype,
    crystalSystem: parts.crystalSystem,
    spacegroup: parts.spacegroup,
    avgLambda: rec.totalEvals > 0 ? rec.lambdaSum / rec.totalEvals : 0,
    maxLambda: rec.maxLambda,
    avgTc: rec.totalEvals > 0 ? rec.tcSum / rec.totalEvals : 0,
    maxTc: rec.maxTc,
    phononStabilityRate: rec.totalEvals > 0 ? rec.phononStableCount / rec.totalEvals : 0,
    totalEvals: rec.totalEvals,
    successCount: rec.successCount,
    rewardScore: computeRewardScore(rec),
  };
}

export function getBestMotifs(n: number = 10): MotifReward[] {
  seedFromDatasets();

  const results: MotifReward[] = [];
  Array.from(motifDB.entries()).forEach(([key, rec]) => {
    if (rec.totalEvals === 0) return;
    const parts = parseKeyParts(key);
    results.push({
      prototype: parts.prototype,
      crystalSystem: parts.crystalSystem,
      spacegroup: parts.spacegroup,
      avgLambda: rec.lambdaSum / rec.totalEvals,
      maxLambda: rec.maxLambda,
      avgTc: rec.tcSum / rec.totalEvals,
      maxTc: rec.maxTc,
      phononStabilityRate: rec.phononStableCount / rec.totalEvals,
      totalEvals: rec.totalEvals,
      successCount: rec.successCount,
      rewardScore: computeRewardScore(rec),
    });
  });

  results.sort((a, b) => b.rewardScore - a.rewardScore);
  return results.slice(0, n);
}

export function getMotifGenerationWeights(): Map<string, number> {
  seedFromDatasets();

  const weights = new Map<string, number>();
  let totalScore = 0;

  Array.from(motifDB.entries()).forEach(([key, rec]) => {
    if (rec.totalEvals === 0) return;
    const score = computeRewardScore(rec);
    weights.set(key, score);
    totalScore += score;
  });

  if (totalScore > 0) {
    Array.from(weights.entries()).forEach(([key, score]) => {
      weights.set(key, score / totalScore);
    });
  }

  return weights;
}

export function sampleWeightedPrototype(): { prototype: string; crystalSystem: string; spacegroup: string | null } | null {
  seedFromDatasets();

  const weights = getMotifGenerationWeights();
  if (weights.size === 0) return null;

  const entries = Array.from(weights.entries());
  const rand = Math.random();
  let cumulative = 0;
  for (const [key, weight] of entries) {
    cumulative += weight;
    if (rand <= cumulative) {
      return parseKeyParts(key);
    }
  }

  const lastKey = entries[entries.length - 1][0];
  return parseKeyParts(lastKey);
}

export function getRewardSystemStats(): {
  totalMotifs: number;
  totalEvaluations: number;
  topMotifs: MotifReward[];
  rewardDistribution: { high: number; medium: number; low: number };
} {
  seedFromDatasets();

  let totalEvaluations = 0;
  let high = 0;
  let medium = 0;
  let low = 0;

  Array.from(motifDB.values()).forEach(rec => {
    totalEvaluations += rec.totalEvals;
    const score = computeRewardScore(rec);
    if (score >= 0.5) high++;
    else if (score >= 0.2) medium++;
    else low++;
  });

  return {
    totalMotifs: motifDB.size,
    totalEvaluations,
    topMotifs: getBestMotifs(10),
    rewardDistribution: { high, medium, low },
  };
}
