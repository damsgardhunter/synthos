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
  nearestPattern: string;
  similarity: number;
}

const FINGERPRINT_KEYS: (keyof PatternFingerprint)[] = [
  "dosLevel", "flatBandScore", "vhsProximity", "nestingScore",
  "couplingStrength", "hydrogenDensity", "dimensionality",
  "correlationStrength", "metallicity", "pressureGpa",
];

function fingerprintToVector(fp: PatternFingerprint): number[] {
  return [
    fp.dosLevel,
    fp.flatBandScore,
    fp.vhsProximity,
    fp.nestingScore,
    fp.couplingStrength,
    Math.min(1, fp.hydrogenDensity / 12),
    fp.dimensionality / 3,
    fp.correlationStrength,
    fp.metallicity,
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
  B: "metalloid", Si: "metalloid", Ge: "metalloid", As: "metalloid",
  Sb: "metalloid", Te: "metalloid", Se: "metalloid",
  H: "nonmetal", C: "nonmetal", N: "nonmetal", O: "nonmetal", F: "nonmetal",
  P: "nonmetal", S: "nonmetal", Cl: "nonmetal",
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

  return {
    dosLevel: physicsContext?.dosLevel ?? 0.5,
    flatBandScore: physicsContext?.flatBandScore ?? (physicsContext?.bandFlatness ?? 0),
    vhsProximity: physicsContext?.vhsProximity ?? (physicsContext?.vanHoveProximity ?? 0),
    nestingScore: physicsContext?.nestingScore ?? 0,
    couplingStrength: physicsContext?.couplingStrength ?? (physicsContext?.lambda ?? 0.5),
    hydrogenDensity: physicsContext?.hydrogenDensity ?? (physicsContext?.hydrogenRatio ?? 0),
    dimensionality: physicsContext?.dimensionality ?? 3,
    elementClasses,
    orbitalCharacter: physicsContext?.orbitalCharacter ?? "d",
    pairingChannel: physicsContext?.pairingChannel ?? "s-wave",
    correlationStrength: physicsContext?.correlationStrength ?? 0.3,
    metallicity: physicsContext?.metallicity ?? 0.7,
    pressureGpa: physicsContext?.pressureGpa ?? 0,
    family,
  };
}

const MAX_MEMORY_SIZE = 500;
const MIN_TC_THRESHOLD = 20;
const CLUSTER_SIMILARITY_THRESHOLD = 0.75;

export class DiscoveryMemory {
  private records: DiscoveryRecord[] = [];
  private clusters: PatternCluster[] = [];
  private nextId = 1;

  recordDiscovery(
    formula: string,
    fingerprint: PatternFingerprint,
    tc: number,
  ): DiscoveryRecord | null {
    if (tc < MIN_TC_THRESHOLD) return null;

    const existing = this.records.find(r => r.formula === formula);
    if (existing) {
      if (tc > existing.tc) {
        existing.tc = tc;
        existing.fingerprint = fingerprint;
        existing.timestamp = Date.now();
        this.rebuildClusters();
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
      this.records.sort((a, b) => b.tc - a.tc);
      this.records = this.records.slice(0, MAX_MEMORY_SIZE);
    }

    this.updateClusters(record);

    return record;
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
    if (avgHDensity > 4) {
      preferredStoichiometries.push({ pattern: "AH10", weight: 0.4 });
      preferredStoichiometries.push({ pattern: "ABH6", weight: 0.3 });
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
      return { bonus: 0, nearestPattern: "none", similarity: 0 };
    }

    const matches = this.queryPatternSimilarity(features, 3);
    if (matches.length === 0) {
      return { bonus: 0, nearestPattern: "none", similarity: 0 };
    }

    const best = matches[0];
    const tcWeight = Math.min(1, best.record.tc / 200);
    const simWeight = best.similarity;

    const bonus = simWeight * tcWeight * 0.3;

    return {
      bonus: Math.round(bonus * 1000) / 1000,
      nearestPattern: best.record.formula,
      similarity: Math.round(best.similarity * 1000) / 1000,
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
      bestCluster.members.push(record);
      bestCluster.avgTc = bestCluster.members.reduce((s, m) => s + m.tc, 0) / bestCluster.members.length;
      bestCluster.centroid = averageVectors(
        bestCluster.members.map(m => fingerprintToVector(m.fingerprint))
      );
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

  private updateClusterMetadata(cluster: PatternCluster): void {
    cluster.dominantFamily = dominantValue(cluster.members.map(m => m.fingerprint.family));
    cluster.dominantOrbital = dominantValue(cluster.members.map(m => m.fingerprint.orbitalCharacter));
    cluster.dominantPairing = dominantValue(cluster.members.map(m => m.fingerprint.pairingChannel));

    const elCounts = countElements(cluster.members);
    const sortedEls = Array.from(elCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([el]) => el);
    cluster.dominantElements = sortedEls;
  }

  private rebuildClusters(): void {
    this.clusters = [];
    for (const record of this.records) {
      this.updateClusters(record);
    }
  }
}

export const discoveryMemory = new DiscoveryMemory();
