import { getTrainingData, type CrystalStructureEntry } from "./crystal-structure-dataset";
import { buildGraphFromStructure, buildGraphFromFormula, getGraphFeatureVector, type StructureGraph } from "./crystal-graph-builder";
import type { CrystalGraph } from "../learning/graph-neural-net";

export interface NoveltyResult {
  noveltyScore: number;
  nearestKnown: string;
  minDistance: number;
  meanDistance: number;
  isNovel: boolean;
}

export interface RankedNoveltyCandidate {
  formula: string;
  noveltyScore: number;
  nearestKnown: string;
  minDistance: number;
  meanDistance: number;
  isNovel: boolean;
}

const NOVEL_THRESHOLD = 0.3;
const MAX_DB_SIZE = 10000;

const fingerprintDB: Map<string, number[]> = new Map();
let initialized = false;

function normalizeVector(vec: number[]): number[] {
  const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  if (mag < 1e-12) return vec.map(() => 0);
  return vec.map(v => v / mag);
}

function cosineDistance(a: number[], b: number[]): number {
  const len = Math.max(a.length, b.length);
  let dot = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < len; i++) {
    const ai = a[i] ?? 0;
    const bi = b[i] ?? 0;
    dot += ai * bi;
    magA += ai * ai;
    magB += bi * bi;
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  if (magA < 1e-12 || magB < 1e-12) return 1.0;
  const cosSim = dot / (magA * magB);
  return 1.0 - Math.max(-1, Math.min(1, cosSim));
}

export function initFingerprintDB(): void {
  if (initialized) return;
  initialized = true;
  try {
    const seedData = getTrainingData();
    let idx = 0;
    function processBatch() {
      const batchEnd = Math.min(idx + 20, seedData.length);
      for (; idx < batchEnd; idx++) {
        try {
          const graph = buildGraphFromStructure(seedData[idx]);
          const fp = normalizeVector(getGraphFeatureVector(graph));
          fingerprintDB.set(seedData[idx].formula, fp);
        } catch (err: any) {
          console.debug(`[novelty-detector] Fingerprint init failed for ${seedData[idx].formula}: ${err?.message ?? err}`);
        }
      }
      if (idx < seedData.length) {
        setImmediate(processBatch);
      } else {
        console.log(`[NoveltyDetector] Initialized fingerprint DB with ${fingerprintDB.size} entries from seed data`);
      }
    }
    processBatch();
  } catch (err) {
    console.error(`[NoveltyDetector] Failed to initialize fingerprint DB:`, err instanceof Error ? err.message : String(err));
  }
}

export function computeStructureFingerprint(formula: string): number[] {
  if (!initialized) initFingerprintDB();

  if (fingerprintDB.has(formula)) {
    return fingerprintDB.get(formula)!;
  }

  try {
    const graph = buildGraphFromFormula(formula);
    return normalizeVector(getGraphFeatureVector(graph));
  } catch {
    return new Array(35).fill(0);
  }
}

export function computeNoveltyScore(fingerprint: number[]): NoveltyResult {
  if (!initialized) initFingerprintDB();

  if (fingerprintDB.size === 0) {
    return {
      noveltyScore: 1.0,
      nearestKnown: "none",
      minDistance: 1.0,
      meanDistance: 1.0,
      isNovel: true,
    };
  }

  let minDist = Infinity;
  let nearestKnown = "unknown";
  let totalDist = 0;
  let count = 0;

  fingerprintDB.forEach((knownFp, formula) => {
    const dist = cosineDistance(fingerprint, knownFp);
    totalDist += dist;
    count++;
    if (dist < minDist) {
      minDist = dist;
      nearestKnown = formula;
    }
  });

  const meanDist = count > 0 ? totalDist / count : 1.0;
  const noveltyScore = Math.min(1.0, minDist / 0.6);

  return {
    noveltyScore,
    nearestKnown,
    minDistance: Math.round(minDist * 10000) / 10000,
    meanDistance: Math.round(meanDist * 10000) / 10000,
    isNovel: minDist > NOVEL_THRESHOLD,
  };
}

export function addKnownFingerprint(formula: string, fingerprint: number[]): void {
  if (!initialized) initFingerprintDB();

  if (fingerprintDB.size >= MAX_DB_SIZE) {
    const firstKey = fingerprintDB.keys().next().value;
    if (firstKey) fingerprintDB.delete(firstKey);
  }

  fingerprintDB.set(formula, normalizeVector(fingerprint));
}

export function getNoveltyRanking(candidates: string[]): RankedNoveltyCandidate[] {
  if (!initialized) initFingerprintDB();

  const results: RankedNoveltyCandidate[] = [];

  for (const formula of candidates) {
    const fp = computeStructureFingerprint(formula);
    const result = computeNoveltyScore(fp);
    results.push({
      formula,
      ...result,
    });
  }

  results.sort((a, b) => b.noveltyScore - a.noveltyScore);
  return results;
}

export function scoreFormulaNovelty(formula: string): NoveltyResult {
  const fp = computeStructureFingerprint(formula);
  return computeNoveltyScore(fp);
}

export function getNoveltyStats(): {
  dbSize: number;
  initialized: boolean;
  avgNoveltyRecent: number;
  novelThreshold: number;
} {
  return {
    dbSize: fingerprintDB.size,
    initialized,
    avgNoveltyRecent: 0,
    novelThreshold: NOVEL_THRESHOLD,
  };
}
