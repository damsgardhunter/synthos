import type { FermiSurfaceResult } from "./fermi-surface-engine";

export interface FSFeatureVector {
  pocketCount: number;
  electronPocketCount: number;
  holePocketCount: number;
  electronHoleBalance: number;
  cylindricalCharacter: number;
  nestingScore: number;
  fsDimensionality: number;
  sigmaBandPresence: number;
  multiBandScore: number;
}

export interface FSClusterArchetype {
  id: string;
  name: string;
  description: string;
  centroid: number[];
}

export interface FSClusterMember {
  formula: string;
  tc: number;
  featureVector: number[];
  similarity: number;
}

export interface FSCluster {
  id: string;
  name: string;
  description: string;
  centroid: number[];
  members: FSClusterMember[];
  memberCount: number;
  avgTc: number;
  bestTc: number;
  commonElements: string[];
  avgFeatureVector: number[];
}

export interface ClusterAssignment {
  formula: string;
  clusterId: string;
  clusterName: string;
  similarity: number;
  featureVector: number[];
}

export interface ClusterGuidance {
  highPotentialClusters: { clusterId: string; name: string; avgTc: number; bestTc: number; memberCount: number }[];
  underExploredClusters: { clusterId: string; name: string; memberCount: number; avgTc: number }[];
  suggestions: string[];
}

const FS_FEATURE_WEIGHTS = [
  1.0, 1.0, 1.0, 1.2, 1.3, 1.8, 1.0, 1.2, 1.8,
];

function weightedCentroid(raw: number[]): number[] {
  return raw.map((v, i) => v * FS_FEATURE_WEIGHTS[i]);
}

const ARCHETYPES: FSClusterArchetype[] = [
  {
    id: "cuprate_cylinder",
    name: "Cuprate Cylindrical",
    description: "High cylindrical character, 2D, strong nesting",
    centroid: weightedCentroid([0.3, 0.2, 0.4, 0.6, 0.85, 0.75, 0.667, 0.2, 0.5]),
  },
  {
    id: "pnictide_eh_pockets",
    name: "Pnictide Electron-Hole",
    description: "Balanced electron-hole pockets, moderate nesting",
    centroid: weightedCentroid([0.4, 0.4, 0.4, 0.9, 0.4, 0.55, 0.833, 0.3, 0.7]),
  },
  {
    id: "kagome_flat",
    name: "Kagome Flat Band",
    description: "High multiBandScore, 2D flat bands",
    centroid: weightedCentroid([0.3, 0.2, 0.4, 0.5, 0.6, 0.4, 0.667, 0.3, 0.85]),
  },
  {
    id: "hydride_multiband",
    name: "Hydride Multiband",
    description: "High pocket count, 3D, high sigma band presence",
    centroid: weightedCentroid([0.6, 0.6, 0.6, 0.8, 0.2, 0.3, 1.0, 0.85, 0.8]),
  },
  {
    id: "heavy_fermion",
    name: "Heavy Fermion",
    description: "Multiple small pockets, extremely high effective mass, f-electron hybridization",
    centroid: weightedCentroid([0.5, 0.3, 0.4, 0.7, 0.15, 0.40, 0.833, 0.10, 0.90]),
  },
  {
    id: "conventional_3d",
    name: "Conventional 3D",
    description: "3D, low nesting, few pockets",
    centroid: weightedCentroid([0.2, 0.2, 0.2, 0.7, 0.15, 0.15, 1.0, 0.4, 0.25]),
  },
];

const clusterMembers: Map<string, FSClusterMember[]> = new Map();
const novelClusterMembers: Map<string, FSClusterMember[]> = new Map();
const assignmentCache: Map<string, ClusterAssignment> = new Map();
let novelClusterCount = 0;

function fsResultToVector(fs: FermiSurfaceResult): number[] {
  const raw = [
    Math.min(fs.pocketCount / 10, 1.0),
    Math.min(fs.electronPocketCount / 5, 1.0),
    Math.min(fs.holePocketCount / 5, 1.0),
    fs.electronHoleBalance,
    fs.cylindricalCharacter,
    fs.nestingScore,
    fs.fsDimensionality / 3,
    fs.sigmaBandPresence,
    fs.multiBandScore,
  ];
  return raw.map((v, i) => v * FS_FEATURE_WEIGHTS[i]);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 1e-10 ? dot / denom : 0;
}

function avgVector(vectors: number[][]): number[] {
  if (vectors.length === 0) return new Array(9).fill(0);
  const sum = new Array(vectors[0].length).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < v.length; i++) sum[i] += v[i];
  }
  return sum.map(s => s / vectors.length);
}

function parseElements(formula: string): string[] {
  if (typeof formula !== "string") return [];
  const matches = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c))).match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function topElements(members: FSClusterMember[], n: number): string[] {
  const counts: Record<string, number> = {};
  for (const m of members) {
    for (const el of parseElements(m.formula)) {
      counts[el] = (counts[el] || 0) + 1;
    }
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([el]) => el);
}

const assignedFormulas = new Set<string>();
const novelClusterCentroids: Map<string, { centroid: number[]; count: number }> = new Map();

interface CachedClusterStats {
  avgTc: number;
  bestTc: number;
  commonElements: string[];
  avgFeatureVector: number[];
  memberCount: number;
}
const clusterStatsCache: Map<string, CachedClusterStats> = new Map();
const clusterStatsDirty: Set<string> = new Set();

function recomputeClusterStats(clusterId: string): CachedClusterStats {
  const members = clusterMembers.get(clusterId) || novelClusterMembers.get(clusterId) || [];
  const tcs = members.map(m => m.tc).filter(t => t > 0);
  const arch = ARCHETYPES.find(a => a.id === clusterId);
  const stored = novelClusterCentroids.get(clusterId);

  let avgFV: number[];
  if (members.length > 0) {
    avgFV = stored ? stored.centroid : avgVector(members.map(m => m.featureVector));
  } else {
    avgFV = arch ? arch.centroid : new Array(9).fill(0);
  }

  const stats: CachedClusterStats = {
    avgTc: tcs.length > 0 ? tcs.reduce((a, b) => a + b, 0) / tcs.length : 0,
    bestTc: tcs.length > 0 ? Math.max(...tcs) : 0,
    commonElements: topElements(members, 5),
    avgFeatureVector: avgFV,
    memberCount: members.length,
  };
  clusterStatsCache.set(clusterId, stats);
  clusterStatsDirty.delete(clusterId);
  return stats;
}

function getClusterStatsCached(clusterId: string): CachedClusterStats {
  if (clusterStatsDirty.has(clusterId) || !clusterStatsCache.has(clusterId)) {
    return recomputeClusterStats(clusterId);
  }
  return clusterStatsCache.get(clusterId)!;
}

function markClusterDirty(clusterId: string): void {
  clusterStatsDirty.add(clusterId);
}

const ARCHETYPE_TARGET_MEMBERS: Record<string, number> = {
  cuprate_cylinder: 20,
  pnictide_eh_pockets: 20,
  kagome_flat: 15,
  hydride_multiband: 20,
  heavy_fermion: 15,
  conventional_3d: 10,
};

function makeCacheKey(formula: string, pressureGpa: number): string {
  const pBin = Math.round(pressureGpa);
  return pBin > 0 ? `${formula}_${pBin}` : formula;
}

export function assignToCluster(formula: string, fsResult: FermiSurfaceResult, tc: number = 0, pressureGpa: number = 0): ClusterAssignment {
  const vec = fsResultToVector(fsResult);
  const cacheKey = makeCacheKey(formula, pressureGpa);
  const alreadyAssigned = assignedFormulas.has(cacheKey);
  let bestId = "conventional_3d";
  let bestSim = -1;

  for (const arch of ARCHETYPES) {
    const sim = cosineSimilarity(vec, arch.centroid);
    if (sim > bestSim) {
      bestSim = sim;
      bestId = arch.id;
    }
  }

  const NOVEL_THRESHOLD = 0.65;
  let assignedId = bestId;
  let assignedName = ARCHETYPES.find(a => a.id === bestId)?.name || bestId;

  if (bestSim < NOVEL_THRESHOLD) {
    let bestNovelSim = -1;
    let bestNovelId = "";
    for (const [nid, stored] of Array.from(novelClusterCentroids.entries())) {
      const sim = cosineSimilarity(vec, stored.centroid);
      if (sim > bestNovelSim) {
        bestNovelSim = sim;
        bestNovelId = nid;
      }
    }

    if (bestNovelSim > 0.85 && bestNovelId) {
      assignedId = bestNovelId;
      assignedName = `Novel Cluster ${bestNovelId.replace("novel_", "")}`;
      bestSim = bestNovelSim;
    } else {
      novelClusterCount++;
      assignedId = `novel_${novelClusterCount}`;
      assignedName = `Novel Cluster ${novelClusterCount}`;
      novelClusterMembers.set(assignedId, []);
      novelClusterCentroids.set(assignedId, { centroid: [...vec], count: 0 });
      bestSim = 1.0;
    }
  }

  const member: FSClusterMember = { formula, tc, featureVector: vec, similarity: bestSim };

  if (!alreadyAssigned) {
    assignedFormulas.add(cacheKey);
    if (assignedId.startsWith("novel_")) {
      const arr = novelClusterMembers.get(assignedId) || [];
      arr.push(member);
      novelClusterMembers.set(assignedId, arr);

      const stored = novelClusterCentroids.get(assignedId);
      if (stored) {
        const n = stored.count;
        const updated = stored.centroid.map((c, i) => (c * n + vec[i]) / (n + 1));
        novelClusterCentroids.set(assignedId, { centroid: updated, count: n + 1 });
      }
    } else {
      const arr = clusterMembers.get(assignedId) || [];
      arr.push(member);
      clusterMembers.set(assignedId, arr);
    }
    markClusterDirty(assignedId);
  }

  const assignment: ClusterAssignment = {
    formula,
    clusterId: assignedId,
    clusterName: assignedName,
    similarity: Number(bestSim.toFixed(4)),
    featureVector: vec,
  };

  assignmentCache.set(cacheKey, assignment);
  return assignment;
}

export function getCluster(clusterId: string): FSCluster | null {
  const arch = ARCHETYPES.find(a => a.id === clusterId);
  if (arch) {
    const members = clusterMembers.get(clusterId) || [];
    const stats = getClusterStatsCached(clusterId);
    return {
      id: arch.id,
      name: arch.name,
      description: arch.description,
      centroid: arch.centroid,
      members,
      memberCount: stats.memberCount,
      avgTc: stats.avgTc,
      bestTc: stats.bestTc,
      commonElements: stats.commonElements,
      avgFeatureVector: stats.avgFeatureVector,
    };
  }

  const novelMembers = novelClusterMembers.get(clusterId);
  if (novelMembers) {
    const stats = getClusterStatsCached(clusterId);
    const stored = novelClusterCentroids.get(clusterId);
    const centroid = stored ? stored.centroid : stats.avgFeatureVector;
    return {
      id: clusterId,
      name: `Novel Cluster ${clusterId.replace("novel_", "")}`,
      description: "Automatically discovered cluster not matching known archetypes",
      centroid,
      members: novelMembers,
      memberCount: stats.memberCount,
      avgTc: stats.avgTc,
      bestTc: stats.bestTc,
      commonElements: stats.commonElements,
      avgFeatureVector: stats.avgFeatureVector,
    };
  }

  return null;
}

export function getAllClusters(): FSCluster[] {
  const clusters: FSCluster[] = [];

  for (const arch of ARCHETYPES) {
    const cluster = getCluster(arch.id);
    if (cluster) clusters.push(cluster);
  }

  for (const nid of Array.from(novelClusterMembers.keys())) {
    const cluster = getCluster(nid);
    if (cluster) clusters.push(cluster);
  }

  return clusters;
}

export function getClusterAssignment(formula: string, pressureGpa: number = 0): ClusterAssignment | null {
  const key = makeCacheKey(formula, pressureGpa);
  return assignmentCache.get(key) || assignmentCache.get(formula) || null;
}

export function getClusterGuidance(): ClusterGuidance {
  const clusters = getAllClusters();
  const populated = clusters.filter(c => c.memberCount > 0);
  const totalMembers = populated.reduce((s, c) => s + c.memberCount, 0);
  const avgMembers = totalMembers > 0 ? totalMembers / clusters.length : 0;

  const highPotential = populated
    .filter(c => c.avgTc > 0)
    .sort((a, b) => b.avgTc - a.avgTc)
    .slice(0, 3)
    .map(c => ({
      clusterId: c.id,
      name: c.name,
      avgTc: Number(c.avgTc.toFixed(2)),
      bestTc: Number(c.bestTc.toFixed(2)),
      memberCount: c.memberCount,
    }));

  const underExplored = clusters
    .map(c => {
      const isArchetype = ARCHETYPES.some(a => a.id === c.id);
      const target = isArchetype
        ? (ARCHETYPE_TARGET_MEMBERS[c.id] || 15)
        : Math.max(3, avgMembers * 0.3);
      const deficit = Math.max(0, target - c.memberCount) / target;
      const priorityWeight = isArchetype ? 2.0 : 1.0;
      return { cluster: c, score: deficit * priorityWeight };
    })
    .filter(entry => entry.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map(entry => ({
      clusterId: entry.cluster.id,
      name: entry.cluster.name,
      memberCount: entry.cluster.memberCount,
      avgTc: Number(entry.cluster.avgTc.toFixed(2)),
    }));

  const suggestions: string[] = [];
  for (const ue of underExplored) {
    const arch = ARCHETYPES.find(a => a.id === ue.clusterId);
    if (arch) {
      suggestions.push(`Explore ${arch.name} topology (${arch.description}) — only ${ue.memberCount} members so far`);
    }
  }
  if (highPotential.length > 0) {
    suggestions.push(`Focus on ${highPotential[0].name} cluster which has highest avg Tc of ${highPotential[0].avgTc.toFixed(1)}K`);
  }
  if (populated.length < ARCHETYPES.length) {
    const missing = ARCHETYPES.filter(a => !populated.find(p => p.id === a.id));
    for (const m of missing.slice(0, 2)) {
      suggestions.push(`No materials yet in ${m.name} cluster — consider generating candidates with ${m.description}`);
    }
  }

  return { highPotentialClusters: highPotential, underExploredClusters: underExplored, suggestions };
}

export function getClusterStats(): { totalAssigned: number; clusterCount: number; archetypeCount: number; novelCount: number } {
  return {
    totalAssigned: assignmentCache.size,
    clusterCount: ARCHETYPES.length + novelClusterMembers.size,
    archetypeCount: ARCHETYPES.length,
    novelCount: novelClusterMembers.size,
  };
}
