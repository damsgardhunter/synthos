import { parseFormulaElements } from "./physics-engine";
import { classifyFamily } from "./utils";
import type { PhysicsAwareRewardContext } from "./rl-agent";

export interface PatternFingerprint {
  dosLevel: number;
  flatBandScore: number;
  vhsProximity: number;
  nestingScore: number;
  couplingStrength: number;
  hydrogenDensity: number;
  dimensionality: number;
  elementClasses: string[];
  orbitalCharacter: string;
  pairingChannel: string;
  correlationStrength: number;
  metallicity: number;
  pressureGpa: number;
  family: string;
  tc: number;
}

export interface DiscoveryRecord {
  id: string;
  formula: string;
  tc: number;
  fingerprint: PatternFingerprint;
  timestamp: number;
}

export interface PatternCluster {
  centroid: number[];
  members: DiscoveryRecord[];
  avgTc: number;
  dominantFamily: string;
  dominantElements: string[];
  dominantOrbital: string;
  dominantPairing: string;
}

export interface GenerationBias {
  preferredElements: { element: string; weight: number }[];
  preferredStructures: { structure: string; weight: number }[];
  preferredStoichiometries: { pattern: string; weight: number }[];
  preferredDimensionality: number;
  preferredHydrogenDensity: number;
  preferredCouplingRange: [number, number];
}

export interface MemoryRewardBonus {
  bonus: number;
  rawBonus: number;
  nearestPattern: string;
  similarity: number;
  rawSimilarity: number;
}

const FINGERPRINT_KEYS: (keyof PatternFingerprint)[] = [
  "dosLevel", "flatBandScore", "vhsProximity", "nestingScore",
  "couplingStrength", "hydrogenDensity", "dimensionality",
  "correlationStrength", "metallicity", "pressureGpa",
];

function fingerprintToVector(fp: PatternFingerprint): number[] {
  return [
    Math.min(1, fp.dosLevel / 5),
    Math.min(1, Math.max(0, fp.flatBandScore)),
    Math.min(1, Math.max(0, fp.vhsProximity)),
    Math.min(1, Math.max(0, fp.nestingScore)),
    Math.min(1, fp.couplingStrength / 3),
    Math.min(1, fp.hydrogenDensity / 12),
    fp.dimensionality / 3,
    Math.min(1, Math.max(0, fp.correlationStrength)),
    Math.min(1, Math.max(0, fp.metallicity)),
    Math.min(1, fp.pressureGpa / 300),
  ];
}

function vectorCosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function vectorEuclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function averageVectors(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const avg = new Array(dim).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) {
      avg[i] += v[i];
    }
  }
  for (let i = 0; i < dim; i++) {
    avg[i] /= vectors.length;
  }
  return avg;
}

function countElements(records: DiscoveryRecord[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const r of records) {
    const els = parseFormulaElements(r.formula);
    for (const el of els) {
      counts.set(el, (counts.get(el) || 0) + 1);
    }
  }
  return counts;
}

function dominantValue<T>(arr: T[]): T {
  const freq = new Map<T, number>();
  for (const v of arr) {
    freq.set(v, (freq.get(v) || 0) + 1);
  }
  let best: T = arr[0];
  let bestCount = 0;
  const entries = Array.from(freq.entries());
  for (const [val, count] of entries) {
    if (count > bestCount) {
      bestCount = count;
      best = val;
    }
  }
  return best;
}

function tcWeightedDominant(items: { value: string; tc: number }[]): string {
  if (items.length === 0) return "";
  const weights = new Map<string, number>();
  for (const item of items) {
    weights.set(item.value, (weights.get(item.value) || 0) + Math.max(1, item.tc));
  }
  let best = items[0].value;
  let bestWeight = 0;
  const entries = Array.from(weights.entries());
  for (const [val, w] of entries) {
    if (w > bestWeight) {
      bestWeight = w;
      best = val;
    }
  }
  return best;
}

const ELEMENT_CLASS_MAP: Record<string, string> = {
  Li: "alkali", Na: "alkali", K: "alkali", Rb: "alkali", Cs: "alkali",
  Be: "alkaline-earth", Mg: "alkaline-earth", Ca: "alkaline-earth", Sr: "alkaline-earth", Ba: "alkaline-earth",
  Sc: "3d-transition", Ti: "3d-transition", V: "3d-transition", Cr: "3d-transition",
  Mn: "3d-transition", Fe: "3d-transition", Co: "3d-transition", Ni: "3d-transition",
  Cu: "3d-transition", Zn: "3d-transition",
  Y: "4d-transition", Zr: "4d-transition", Nb: "4d-transition", Mo: "4d-transition",
  Ru: "4d-transition", Rh: "4d-transition", Pd: "4d-transition", Ag: "4d-transition",
  Hf: "5d-transition", Ta: "5d-transition", W: "5d-transition", Re: "5d-transition",
  Os: "5d-transition", Ir: "5d-transition", Pt: "5d-transition", Au: "5d-transition",
  La: "lanthanide", Ce: "lanthanide", Pr: "lanthanide", Nd: "lanthanide",
  Sm: "lanthanide", Gd: "lanthanide", Dy: "lanthanide", Er: "lanthanide",
  Yb: "lanthanide", Lu: "lanthanide",
  Al: "p-block-metal", Ga: "p-block-metal", In: "p-block-metal", Sn: "p-block-metal",
  Tl: "p-block-metal", Pb: "p-block-metal", Bi: "p-block-metal",
  B: "metalloid", Si: "metalloid", Ge: "metalloid",
  Sb: "metalloid",
  O: "chalcogen", S: "chalcogen", Se: "chalcogen", Te: "chalcogen",
  F: "halogen", Cl: "halogen", Br: "halogen", I: "halogen",
  N: "pnictogen", P: "pnictogen", As: "pnictogen",
  H: "hydrogen", C: "carbon",
};

function getElementClasses(formula: string): string[] {
  const els = parseFormulaElements(formula);
  const classes = new Set<string>();
  for (const el of els) {
    const cls = ELEMENT_CLASS_MAP[el];
    if (cls) classes.add(cls);
  }
  return Array.from(classes);
}

const FAMILY_ORBITAL_DEFAULTS: Record<string, string> = {
  Hydride: "s/p",
  Clathrate: "s/p",
  Cuprate: "d-x2y2",
  Pnictide: "d",
  Chalcogenide: "d",
  Intermetallic: "d",
  "Heavy-fermion": "f",
  Borocarbide: "d",
  Bismuthate: "s/p",
  Chevrel: "d",
  Organic: "p",
};

export function buildFingerprint(
  formula: string,
  tc: number,
  physicsContext?: Partial<PhysicsAwareRewardContext> & {
    dosLevel?: number;
    flatBandScore?: number;
    vhsProximity?: number;
    nestingScore?: number;
    couplingStrength?: number;
    hydrogenDensity?: number;
    dimensionality?: number;
    correlationStrength?: number;
    metallicity?: number;
    pressureGpa?: number;
    orbitalCharacter?: string;
    pairingChannel?: string;
  }
): PatternFingerprint {
  const family = classifyFamily(formula);
  const elementClasses = getElementClasses(formula);

  const defaultOrbital = FAMILY_ORBITAL_DEFAULTS[family] ?? "d";

  return {
    dosLevel: physicsContext?.dosLevel ?? 0.5,
    flatBandScore: physicsContext?.flatBandScore ?? (physicsContext?.bandFlatness ?? 0),
    vhsProximity: physicsContext?.vhsProximity ?? (physicsContext?.vanHoveProximity ?? 0),
    nestingScore: physicsContext?.nestingScore ?? 0,
    couplingStrength: physicsContext?.couplingStrength ?? (physicsContext?.lambda ?? 0.5),
    hydrogenDensity: physicsContext?.hydrogenDensity ?? (physicsContext?.hydrogenRatio ?? 0),
    dimensionality: physicsContext?.dimensionality ?? 3,
    elementClasses,
    orbitalCharacter: physicsContext?.orbitalCharacter ?? defaultOrbital,
    pairingChannel: physicsContext?.pairingChannel ?? "s-wave",
    correlationStrength: physicsContext?.correlationStrength ?? 0.3,
    metallicity: physicsContext?.metallicity ?? 0.7,
    pressureGpa: physicsContext?.pressureGpa ?? 0,
    family,
    tc,
  };
}

const MAX_MEMORY_SIZE = 500;
const FAILURE_CACHE_SIZE = 50;
const MIN_TC_THRESHOLD = 20;
const CLUSTER_SIMILARITY_THRESHOLD = 0.75;

export class DiscoveryMemory {
  private records: DiscoveryRecord[] = [];
  private failureCache: DiscoveryRecord[] = [];
  private clusters: PatternCluster[] = [];
  private nextId = 1;

  recordDiscovery(
    formula: string,
    fingerprint: PatternFingerprint,
    tc: number,
  ): DiscoveryRecord | null {
    if (tc < MIN_TC_THRESHOLD) {
      this.recordFailure(formula, fingerprint, tc);
      return null;
    }

    const existing = this.records.find(r => r.formula === formula);
    if (existing) {
      if (tc > existing.tc) {
        const oldFingerprint = existing.fingerprint;
        existing.tc = tc;
        existing.fingerprint = fingerprint;
        existing.timestamp = Date.now();
        this.updateExistingRecordCluster(existing, oldFingerprint);
      }
      return existing;
    }

    const record: DiscoveryRecord = {
      id: `dm-${this.nextId++}`,
      formula,
      tc,
      fingerprint,
      timestamp: Date.now(),
    };

    this.records.push(record);

    if (this.records.length > MAX_MEMORY_SIZE) {
      this.evictWithDiversity();
    }

    this.updateClusters(record);

    return record;
  }

  private recordFailure(formula: string, fingerprint: PatternFingerprint, tc: number): void {
    const existing = this.failureCache.find(r => r.formula === formula);
    if (existing) return;

    const vec = fingerprintToVector(fingerprint);
    let isNovel = true;
    for (const f of this.failureCache) {
      const sim = vectorCosineSimilarity(vec, fingerprintToVector(f.fingerprint));
      if (sim > 0.9) {
        isNovel = false;
        break;
      }
    }
    if (!isNovel) return;

    const record: DiscoveryRecord = {
      id: `fail-${this.nextId++}`,
      formula,
      tc,
      fingerprint,
      timestamp: Date.now(),
    };
    this.failureCache.push(record);

    if (this.failureCache.length > FAILURE_CACHE_SIZE) {
      this.failureCache.shift();
    }
  }

  private evictWithDiversity(): void {
    const keepTop = Math.floor(MAX_MEMORY_SIZE * 0.8);
    const keepNovel = MAX_MEMORY_SIZE - keepTop;

    this.records.sort((a, b) => b.tc - a.tc);
    const topRecords = this.records.slice(0, keepTop);
    const candidates = this.records.slice(keepTop);

    const keptVecs = topRecords.map(r => fingerprintToVector(r.fingerprint));
    const novelScored = candidates.map(r => {
      const vec = fingerprintToVector(r.fingerprint);
      let minSim = 1;
      for (const kv of keptVecs) {
        const sim = vectorCosineSimilarity(vec, kv);
        if (sim < minSim) minSim = sim;
      }
      return { record: r, novelty: 1 - minSim };
    });

    novelScored.sort((a, b) => b.novelty - a.novelty);
    const novelRecords = novelScored.slice(0, keepNovel).map(s => s.record);

    this.records = [...topRecords, ...novelRecords];
    this.rebuildClusters();
  }

  getFailureCache(): DiscoveryRecord[] {
    return [...this.failureCache];
  }

  queryPatternSimilarity(features: PatternFingerprint, topK: number = 5): {
    record: DiscoveryRecord;
    similarity: number;
  }[] {
    if (this.records.length === 0) return [];

    const queryVec = fingerprintToVector(features);
    const scored = this.records.map(record => ({
      record,
      similarity: vectorCosineSimilarity(queryVec, fingerprintToVector(record.fingerprint)),
    }));

    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, topK);
  }

  isKnownFailure(features: PatternFingerprint, threshold: number = 0.85): boolean {
    if (this.failureCache.length === 0) return false;
    const vec = fingerprintToVector(features);
    for (const f of this.failureCache) {
      const sim = vectorCosineSimilarity(vec, fingerprintToVector(f.fingerprint));
      if (sim > threshold) return true;
    }
    return false;
  }

  getTopPatterns(n: number = 10): DiscoveryRecord[] {
    const sorted = [...this.records].sort((a, b) => b.tc - a.tc);
    return sorted.slice(0, n);
  }

  biasGenerationFromMemory(): GenerationBias | null {
    if (this.records.length < 3) return null;

    const topRecords = this.getTopPatterns(Math.min(20, this.records.length));

    const elementCounts = countElements(topRecords);
    const preferredElements: { element: string; weight: number }[] = [];
    const totalRecords = topRecords.length;
    const elEntries = Array.from(elementCounts.entries());
    for (const [el, count] of elEntries) {
      const weight = count / totalRecords;
      if (weight >= 0.15) {
        preferredElements.push({ element: el, weight });
      }
    }
    preferredElements.sort((a, b) => b.weight - a.weight);

    const familyCounts = new Map<string, { count: number; avgTc: number }>();
    for (const r of topRecords) {
      const fam = r.fingerprint.family;
      const entry = familyCounts.get(fam) || { count: 0, avgTc: 0 };
      entry.avgTc = (entry.avgTc * entry.count + r.tc) / (entry.count + 1);
      entry.count++;
      familyCounts.set(fam, entry);
    }
    const preferredStructures: { structure: string; weight: number }[] = [];
    const famEntries = Array.from(familyCounts.entries());
    for (const [fam, data] of famEntries) {
      preferredStructures.push({
        structure: fam,
        weight: (data.count / totalRecords) * (data.avgTc / Math.max(1, topRecords[0].tc)),
      });
    }
    preferredStructures.sort((a, b) => b.weight - a.weight);

    const preferredStoichiometries: { pattern: string; weight: number }[] = [];
    const hDensities = topRecords.map(r => r.fingerprint.hydrogenDensity);
    const avgHDensity = hDensities.reduce((s, v) => s + v, 0) / hDensities.length;
    if (avgHDensity > 3) {
      const hydrideRecords = topRecords.filter(r => r.fingerprint.hydrogenDensity > 3);
      const metalFreq = new Map<string, number>();
      for (const r of hydrideRecords) {
        const els = parseFormulaElements(r.formula).filter(e => e !== "H");
        for (const el of els) {
          metalFreq.set(el, (metalFreq.get(el) || 0) + 1);
        }
      }
      const topMetals = Array.from(metalFreq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([el]) => el);

      if (topMetals.length >= 1) {
        preferredStoichiometries.push({ pattern: `${topMetals[0]}H3`, weight: 0.4 });
        preferredStoichiometries.push({ pattern: `${topMetals[0]}H6`, weight: 0.3 });
        preferredStoichiometries.push({ pattern: `${topMetals[0]}H10`, weight: 0.2 });
      }
      if (topMetals.length >= 2) {
        preferredStoichiometries.push({ pattern: `${topMetals[0]}${topMetals[1]}H4`, weight: 0.3 });
        preferredStoichiometries.push({ pattern: `${topMetals[0]}${topMetals[1]}H8`, weight: 0.2 });
      }
    }

    const dims = topRecords.map(r => r.fingerprint.dimensionality);
    const avgDim = dims.reduce((s, v) => s + v, 0) / dims.length;

    const couplings = topRecords.map(r => r.fingerprint.couplingStrength);
    const minCoupling = Math.min(...couplings);
    const maxCoupling = Math.max(...couplings);

    return {
      preferredElements: preferredElements.slice(0, 10),
      preferredStructures: preferredStructures.slice(0, 5),
      preferredStoichiometries,
      preferredDimensionality: avgDim,
      preferredHydrogenDensity: avgHDensity,
      preferredCouplingRange: [
        Math.max(0, minCoupling - 0.2),
        Math.min(4, maxCoupling + 0.3),
      ],
    };
  }

  computeMemoryRewardBonus(features: PatternFingerprint): MemoryRewardBonus {
    if (this.records.length === 0) {
      return { bonus: 0.1, rawBonus: 0.1, nearestPattern: "none", similarity: 0, rawSimilarity: 0 };
    }

    const matches = this.queryPatternSimilarity(features, 3);
    if (matches.length === 0) {
      return { bonus: 0.1, rawBonus: 0.1, nearestPattern: "none", similarity: 0, rawSimilarity: 0 };
    }

    const best = matches[0];
    const sim = best.similarity;

    const REDUNDANCY_THRESHOLD = 0.85;
    const NOVELTY_THRESHOLD = 0.4;

    let bonus: number;
    if (sim > REDUNDANCY_THRESHOLD) {
      const penaltyStrength = (sim - REDUNDANCY_THRESHOLD) / (1 - REDUNDANCY_THRESHOLD);
      bonus = -0.15 * penaltyStrength;
    } else if (sim < NOVELTY_THRESHOLD) {
      const noveltyStrength = (NOVELTY_THRESHOLD - sim) / NOVELTY_THRESHOLD;
      bonus = 0.1 * noveltyStrength;
    } else {
      bonus = 0;
    }

    const tcImprovement = features.tc > best.record.tc ? (features.tc - best.record.tc) / Math.max(1, best.record.tc) : 0;
    if (tcImprovement > 0.1 && sim > NOVELTY_THRESHOLD) {
      bonus += Math.min(0.1, tcImprovement * 0.05);
    }

    if (this.isKnownFailure(features, 0.8)) {
      bonus -= 0.05;
    }

    return {
      bonus: Math.round(bonus * 1000) / 1000,
      rawBonus: bonus,
      nearestPattern: best.record.formula,
      similarity: Math.round(sim * 1000) / 1000,
      rawSimilarity: sim,
    };
  }

  getClusters(): PatternCluster[] {
    return this.clusters;
  }

  getRecordCount(): number {
    return this.records.length;
  }

  getStats(): {
    totalRecords: number;
    clusterCount: number;
    avgTc: number;
    bestTc: number;
    failureCacheSize: number;
    topFamilies: { family: string; count: number }[];
  } {
    const avgTc = this.records.length > 0
      ? this.records.reduce((s, r) => s + r.tc, 0) / this.records.length
      : 0;
    const bestTc = this.records.length > 0
      ? Math.max(...this.records.map(r => r.tc))
      : 0;

    const famCounts = new Map<string, number>();
    for (const r of this.records) {
      const fam = r.fingerprint.family;
      famCounts.set(fam, (famCounts.get(fam) || 0) + 1);
    }
    const topFamilies = Array.from(famCounts.entries())
      .map(([family, count]) => ({ family, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    return {
      totalRecords: this.records.length,
      clusterCount: this.clusters.length,
      avgTc: Math.round(avgTc * 10) / 10,
      bestTc: Math.round(bestTc * 10) / 10,
      failureCacheSize: this.failureCache.length,
      topFamilies,
    };
  }

  private updateClusters(record: DiscoveryRecord): void {
    const vec = fingerprintToVector(record.fingerprint);

    let bestCluster: PatternCluster | null = null;
    let bestSim = 0;

    for (const cluster of this.clusters) {
      const sim = vectorCosineSimilarity(vec, cluster.centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestCluster = cluster;
      }
    }

    if (bestCluster && bestSim >= CLUSTER_SIMILARITY_THRESHOLD) {
      const n = bestCluster.members.length;
      bestCluster.members.push(record);
      bestCluster.avgTc = (bestCluster.avgTc * n + record.tc) / (n + 1);
      const newN = n + 1;
      bestCluster.centroid = bestCluster.centroid.map((c, i) => (c * n + vec[i]) / newN);
      this.updateClusterMetadata(bestCluster);
    } else {
      const newCluster: PatternCluster = {
        centroid: vec,
        members: [record],
        avgTc: record.tc,
        dominantFamily: record.fingerprint.family,
        dominantElements: parseFormulaElements(record.formula),
        dominantOrbital: record.fingerprint.orbitalCharacter,
        dominantPairing: record.fingerprint.pairingChannel,
      };
      this.clusters.push(newCluster);
    }
  }

  private updateExistingRecordCluster(record: DiscoveryRecord, oldFingerprint: PatternFingerprint): void {
    const oldVec = fingerprintToVector(oldFingerprint);
    const newVec = fingerprintToVector(record.fingerprint);

    for (const cluster of this.clusters) {
      const memberIdx = cluster.members.findIndex(m => m.id === record.id);
      if (memberIdx === -1) continue;

      const n = cluster.members.length;
      cluster.avgTc = cluster.members.reduce((s, m) => s + m.tc, 0) / n;

      if (n === 1) {
        cluster.centroid = newVec;
      } else {
        cluster.centroid = cluster.centroid.map((c, i) =>
          (c * n - oldVec[i] + newVec[i]) / n
        );
      }

      const newSim = vectorCosineSimilarity(newVec, cluster.centroid);
      if (newSim < CLUSTER_SIMILARITY_THRESHOLD && n > 1) {
        cluster.members.splice(memberIdx, 1);
        cluster.avgTc = cluster.members.reduce((s, m) => s + m.tc, 0) / cluster.members.length;
        cluster.centroid = averageVectors(cluster.members.map(m => fingerprintToVector(m.fingerprint)));
        this.updateClusterMetadata(cluster);
        this.updateClusters(record);
      } else {
        this.updateClusterMetadata(cluster);
      }
      return;
    }

    this.updateClusters(record);
  }

  private updateClusterMetadata(cluster: PatternCluster): void {
    cluster.dominantFamily = tcWeightedDominant(
      cluster.members.map(m => ({ value: m.fingerprint.family, tc: m.tc }))
    );
    cluster.dominantOrbital = tcWeightedDominant(
      cluster.members.map(m => ({ value: m.fingerprint.orbitalCharacter, tc: m.tc }))
    );
    cluster.dominantPairing = tcWeightedDominant(
      cluster.members.map(m => ({ value: m.fingerprint.pairingChannel, tc: m.tc }))
    );

    const elTcWeights = new Map<string, number>();
    for (const m of cluster.members) {
      const els = parseFormulaElements(m.formula);
      for (const el of els) {
        elTcWeights.set(el, (elTcWeights.get(el) || 0) + m.tc);
      }
    }
    cluster.dominantElements = Array.from(elTcWeights.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([el]) => el);
  }

  private rebuildClusters(): void {
    this.clusters = [];
    for (const record of this.records) {
      this.updateClusters(record);
    }
  }
}

export const discoveryMemory = new DiscoveryMemory();
