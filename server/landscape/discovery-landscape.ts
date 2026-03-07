import { encodeGenome } from "../physics/materials-genome";

export interface EmbeddingPoint {
  formula: string;
  position3D: [number, number, number];
  genomeVector: number[];
  tc: number;
  lambda: number;
  mechanism: string;
  family: string;
  clusterId: string;
}

interface UMAPConfig {
  nNeighbors: number;
  minDist: number;
  nComponents: number;
  nEpochs: number;
  learningRate: number;
  spread: number;
  repulsionStrength: number;
}

const DEFAULT_UMAP_CONFIG: UMAPConfig = {
  nNeighbors: 15,
  minDist: 0.1,
  nComponents: 3,
  nEpochs: 200,
  learningRate: 1.0,
  spread: 1.0,
  repulsionStrength: 1.0,
};

const embeddingStore = new Map<string, EmbeddingPoint>();
let lastFullRecomputeSize = 0;
let lastUpdateTime = 0;
let totalUpdates = 0;

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function findKNearestNeighbors(
  vectors: number[][],
  k: number,
): { indices: number[][]; distances: number[][] } {
  const n = vectors.length;
  const kActual = Math.min(k, n - 1);
  const indices: number[][] = new Array(n);
  const distances: number[][] = new Array(n);

  for (let i = 0; i < n; i++) {
    const dists: { idx: number; dist: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      dists.push({ idx: j, dist: euclideanDistance(vectors[i], vectors[j]) });
    }
    dists.sort((a, b) => a.dist - b.dist);
    indices[i] = dists.slice(0, kActual).map(d => d.idx);
    distances[i] = dists.slice(0, kActual).map(d => d.dist);
  }

  return { indices, distances };
}

function computeSigmas(
  distances: number[][],
  targetPerplexity: number,
): number[] {
  const n = distances.length;
  const sigmas = new Array(n);
  const logTarget = Math.log(targetPerplexity);

  for (let i = 0; i < n; i++) {
    let lo = 1e-10, hi = 1000, mid = 1;
    const dists = distances[i];
    if (dists.length === 0) { sigmas[i] = 1; continue; }

    for (let iter = 0; iter < 64; iter++) {
      mid = (lo + hi) / 2;
      let sumWeights = 0;
      for (const d of dists) {
        sumWeights += Math.exp(-d * d / (2 * mid * mid));
      }
      const entropy = Math.log(Math.max(sumWeights, 1e-10));
      if (entropy > logTarget) hi = mid;
      else lo = mid;
      if (Math.abs(hi - lo) < 1e-8) break;
    }
    sigmas[i] = mid;
  }
  return sigmas;
}

function buildFuzzyGraph(
  indices: number[][],
  distances: number[][],
  sigmas: number[],
): Map<string, number> {
  const graph = new Map<string, number>();
  const n = indices.length;

  for (let i = 0; i < n; i++) {
    const rho = distances[i].length > 0 ? distances[i][0] : 0;
    for (let k = 0; k < indices[i].length; k++) {
      const j = indices[i][k];
      const d = distances[i][k];
      const w = Math.exp(-Math.max(0, d - rho) / (sigmas[i] + 1e-10));
      const key = `${i}-${j}`;
      graph.set(key, Math.max(graph.get(key) ?? 0, w));
    }
  }

  const symmetric = new Map<string, number>();
  for (const [key, w] of graph) {
    const [iStr, jStr] = key.split("-");
    const reverseKey = `${jStr}-${iStr}`;
    const wReverse = graph.get(reverseKey) ?? 0;
    const wSym = w + wReverse - w * wReverse;
    symmetric.set(key, wSym);
    symmetric.set(reverseKey, wSym);
  }

  return symmetric;
}

function spectralInitialization(
  vectors: number[][],
  nComponents: number,
): number[][] {
  const n = vectors.length;
  if (n === 0) return [];

  const dim = vectors[0].length;
  const mean = new Array(dim).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) mean[i] += v[i] / n;
  }

  const centered = vectors.map(v => v.map((x, i) => x - mean[i]));

  const projections: number[][] = [];
  for (let comp = 0; comp < nComponents; comp++) {
    let axis = new Array(dim).fill(0);
    axis[comp % dim] = 1;

    for (let iter = 0; iter < 20; iter++) {
      const newAxis = new Array(dim).fill(0);
      for (const row of centered) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += row[d] * axis[d];
        for (let d = 0; d < dim; d++) newAxis[d] += dot * row[d];
      }

      for (let prev = 0; prev < comp; prev++) {
        const prevAxis = projections[prev];
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += newAxis[d] * (prevAxis?.[d] ?? 0);
        for (let d = 0; d < dim; d++) newAxis[d] -= dot * (prevAxis?.[d] ?? 0);
      }

      let norm = 0;
      for (let d = 0; d < dim; d++) norm += newAxis[d] * newAxis[d];
      norm = Math.sqrt(norm);
      if (norm > 1e-10) {
        for (let d = 0; d < dim; d++) axis[d] = newAxis[d] / norm;
      }
    }
    projections.push(axis);
  }

  const embedding: number[][] = [];
  for (const row of centered) {
    const coords: number[] = [];
    for (let comp = 0; comp < nComponents; comp++) {
      let val = 0;
      for (let d = 0; d < dim; d++) val += row[d] * projections[comp][d];
      coords.push(val * 10);
    }
    embedding.push(coords);
  }

  return embedding;
}

function umapOptimize(
  embedding: number[][],
  graph: Map<string, number>,
  n: number,
  config: UMAPConfig,
): number[][] {
  const { nEpochs, learningRate, minDist, spread, repulsionStrength, nComponents } = config;

  const a = 1.929;
  const b = 0.7915;

  const edges: { i: number; j: number; w: number }[] = [];
  for (const [key, w] of graph) {
    const [iStr, jStr] = key.split("-");
    const i = parseInt(iStr);
    const j = parseInt(jStr);
    if (i < j && w > 0.01) {
      edges.push({ i, j, w });
    }
  }

  if (edges.length === 0) return embedding;

  const maxWeight = Math.max(...edges.map(e => e.w));
  const epochsPerEdge = edges.map(e => {
    const freq = e.w / maxWeight;
    return freq > 0 ? Math.max(1, Math.floor(nEpochs * freq)) : 0;
  });

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    const alpha = learningRate * (1.0 - epoch / nEpochs);

    for (let eIdx = 0; eIdx < edges.length; eIdx++) {
      if (epoch % Math.max(1, Math.floor(nEpochs / epochsPerEdge[eIdx])) !== 0) continue;

      const { i, j } = edges[eIdx];
      let distSq = 0;
      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i][d] - embedding[j][d];
        distSq += diff * diff;
      }
      const dist = Math.sqrt(distSq + 1e-10);

      const attractGrad = (-2.0 * a * b * Math.pow(distSq, b - 1)) / (1 + a * Math.pow(distSq, b));

      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i][d] - embedding[j][d];
        const grad = attractGrad * diff;
        embedding[i][d] -= alpha * grad;
        embedding[j][d] += alpha * grad;
      }

      const nNeg = Math.min(5, n - 1);
      for (let neg = 0; neg < nNeg; neg++) {
        let k = Math.floor(Math.random() * n);
        if (k === i) k = (k + 1) % n;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i][d] - embedding[k][d];
          negDistSq += diff * diff;
        }

        if (negDistSq > 0.01) {
          const repelGrad = (2.0 * repulsionStrength * b) / ((0.001 + negDistSq) * (1 + a * Math.pow(negDistSq, b)));
          for (let d = 0; d < nComponents; d++) {
            const diff = embedding[i][d] - embedding[k][d];
            embedding[i][d] += alpha * repelGrad * diff;
          }
        }
      }
    }
  }

  return embedding;
}

interface MaterialRecord {
  formula: string;
  tc: number;
  lambda: number;
  mechanism: string;
  family: string;
  clusterId: string;
}

const materialRecords = new Map<string, MaterialRecord>();

export function addMaterialToDataset(
  formula: string,
  tc: number,
  lambda: number = 0,
  mechanism: string = "unknown",
  family: string = "unknown",
  clusterId: string = "unassigned",
): void {
  materialRecords.set(formula, { formula, tc, lambda, mechanism, family, clusterId });
}

export function buildEmbeddingDataset(): {
  formulas: string[];
  vectors: number[][];
  records: MaterialRecord[];
} {
  const formulas: string[] = [];
  const vectors: number[][] = [];
  const records: MaterialRecord[] = [];

  for (const [formula, record] of materialRecords) {
    try {
      const genome = encodeGenome(formula);
      if (genome && genome.vector && genome.vector.length > 0) {
        formulas.push(formula);
        vectors.push(genome.vector);
        records.push(record);
      }
    } catch {}
  }

  return { formulas, vectors, records };
}

export function runFullUMAP(
  config?: Partial<UMAPConfig>,
): EmbeddingPoint[] {
  const cfg = { ...DEFAULT_UMAP_CONFIG, ...config };
  const { formulas, vectors, records } = buildEmbeddingDataset();

  if (formulas.length < 3) {
    const points: EmbeddingPoint[] = formulas.map((f, i) => ({
      formula: f,
      position3D: [i * 2, 0, 0] as [number, number, number],
      genomeVector: vectors[i],
      tc: records[i].tc,
      lambda: records[i].lambda,
      mechanism: records[i].mechanism,
      family: records[i].family,
      clusterId: records[i].clusterId,
    }));
    for (const p of points) embeddingStore.set(p.formula, p);
    lastFullRecomputeSize = formulas.length;
    lastUpdateTime = Date.now();
    totalUpdates++;
    return points;
  }

  const k = Math.min(cfg.nNeighbors, formulas.length - 1);
  const { indices, distances } = findKNearestNeighbors(vectors, k);
  const sigmas = computeSigmas(distances, Math.log2(k));
  const graph = buildFuzzyGraph(indices, distances, sigmas);

  let embedding = spectralInitialization(vectors, cfg.nComponents);

  const nEpochsScaled = Math.min(cfg.nEpochs, Math.max(50, Math.floor(200 * Math.sqrt(30 / formulas.length))));
  embedding = umapOptimize(embedding, graph, formulas.length, { ...cfg, nEpochs: nEpochsScaled });

  let minVals = [Infinity, Infinity, Infinity];
  let maxVals = [-Infinity, -Infinity, -Infinity];
  for (const coords of embedding) {
    for (let d = 0; d < 3; d++) {
      minVals[d] = Math.min(minVals[d], coords[d]);
      maxVals[d] = Math.max(maxVals[d], coords[d]);
    }
  }

  const points: EmbeddingPoint[] = [];
  for (let i = 0; i < formulas.length; i++) {
    const normalized: [number, number, number] = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const range = maxVals[d] - minVals[d];
      normalized[d] = range > 1e-10 ? (embedding[i][d] - minVals[d]) / range * 20 - 10 : 0;
      normalized[d] = Math.round(normalized[d] * 1000) / 1000;
    }

    const point: EmbeddingPoint = {
      formula: formulas[i],
      position3D: normalized,
      genomeVector: vectors[i],
      tc: records[i].tc,
      lambda: records[i].lambda,
      mechanism: records[i].mechanism,
      family: records[i].family,
      clusterId: records[i].clusterId,
    };
    points.push(point);
    embeddingStore.set(formulas[i], point);
  }

  lastFullRecomputeSize = formulas.length;
  lastUpdateTime = Date.now();
  totalUpdates++;
  return points;
}

export function incrementalUpdate(
  newFormulas: string[],
): EmbeddingPoint[] {
  const currentSize = materialRecords.size;
  const needsFullRecompute = lastFullRecomputeSize === 0 ||
    currentSize > lastFullRecomputeSize * 1.5 ||
    embeddingStore.size < 3;

  if (needsFullRecompute) {
    return runFullUMAP();
  }

  const existingPoints = Array.from(embeddingStore.values());
  const newPoints: EmbeddingPoint[] = [];

  for (const formula of newFormulas) {
    if (embeddingStore.has(formula)) continue;
    const record = materialRecords.get(formula);
    if (!record) continue;

    try {
      const genome = encodeGenome(formula);
      if (!genome || !genome.vector || genome.vector.length === 0) continue;

      const kNearest = Math.min(5, existingPoints.length);
      const dists = existingPoints.map((p, idx) => ({
        idx,
        dist: euclideanDistance(genome.vector, p.genomeVector),
      }));
      dists.sort((a, b) => a.dist - b.dist);
      const neighbors = dists.slice(0, kNearest);

      let totalWeight = 0;
      const position: [number, number, number] = [0, 0, 0];

      for (const n of neighbors) {
        const w = 1 / (n.dist + 1e-6);
        totalWeight += w;
        for (let d = 0; d < 3; d++) {
          position[d] += w * existingPoints[n.idx].position3D[d];
        }
      }
      for (let d = 0; d < 3; d++) {
        position[d] = Math.round((position[d] / totalWeight + (Math.random() - 0.5) * 0.5) * 1000) / 1000;
      }

      const point: EmbeddingPoint = {
        formula,
        position3D: position,
        genomeVector: genome.vector,
        tc: record.tc,
        lambda: record.lambda,
        mechanism: record.mechanism,
        family: record.family,
        clusterId: record.clusterId,
      };
      newPoints.push(point);
      embeddingStore.set(formula, point);
    } catch {}
  }

  lastUpdateTime = Date.now();
  totalUpdates++;
  return newPoints;
}

export function updateLandscape(
  newMaterials: { formula: string; tc: number; lambda?: number; mechanism?: string; family?: string; clusterId?: string }[] = [],
): { added: number; totalSize: number; method: string } {
  for (const m of newMaterials) {
    addMaterialToDataset(m.formula, m.tc, m.lambda ?? 0, m.mechanism ?? "unknown", m.family ?? "unknown", m.clusterId ?? "unassigned");
  }

  const unembeddedFormulas: string[] = [];
  for (const formula of materialRecords.keys()) {
    if (!embeddingStore.has(formula)) {
      unembeddedFormulas.push(formula);
    }
  }

  if (unembeddedFormulas.length === 0 && newMaterials.length === 0) {
    return { added: 0, totalSize: embeddingStore.size, method: "no-op" };
  }

  const currentSize = materialRecords.size;
  const needsFullRecompute = lastFullRecomputeSize === 0 ||
    currentSize > lastFullRecomputeSize * 1.5 ||
    embeddingStore.size < 5;

  let method: string;
  if (needsFullRecompute) {
    runFullUMAP();
    method = "full-recompute";
  } else if (unembeddedFormulas.length > 0) {
    incrementalUpdate(unembeddedFormulas);
    method = "incremental";
  } else {
    method = "no-op";
  }

  return { added: unembeddedFormulas.length, totalSize: embeddingStore.size, method };
}

export function getEmbeddingDataset(): EmbeddingPoint[] {
  return Array.from(embeddingStore.values());
}

export function getEmbeddingPoint(formula: string): EmbeddingPoint | undefined {
  return embeddingStore.get(formula);
}

export function getLandscapeStats(): {
  totalMaterials: number;
  embeddedMaterials: number;
  lastUpdateTime: number;
  totalUpdates: number;
  lastFullRecomputeSize: number;
  familyDistribution: Record<string, number>;
  tcRange: { min: number; max: number; avg: number };
  spatialExtent: { min: [number, number, number]; max: [number, number, number] };
} {
  const points = Array.from(embeddingStore.values());
  const familyDist: Record<string, number> = {};
  let tcMin = Infinity, tcMax = -Infinity, tcSum = 0;
  const spatialMin: [number, number, number] = [Infinity, Infinity, Infinity];
  const spatialMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  for (const p of points) {
    familyDist[p.family] = (familyDist[p.family] ?? 0) + 1;
    tcMin = Math.min(tcMin, p.tc);
    tcMax = Math.max(tcMax, p.tc);
    tcSum += p.tc;
    for (let d = 0; d < 3; d++) {
      spatialMin[d] = Math.min(spatialMin[d], p.position3D[d]);
      spatialMax[d] = Math.max(spatialMax[d], p.position3D[d]);
    }
  }

  return {
    totalMaterials: materialRecords.size,
    embeddedMaterials: embeddingStore.size,
    lastUpdateTime,
    totalUpdates,
    lastFullRecomputeSize,
    familyDistribution: familyDist,
    tcRange: {
      min: points.length > 0 ? tcMin : 0,
      max: points.length > 0 ? tcMax : 0,
      avg: points.length > 0 ? Math.round(tcSum / points.length * 10) / 10 : 0,
    },
    spatialExtent: {
      min: points.length > 0 ? spatialMin : [0, 0, 0],
      max: points.length > 0 ? spatialMax : [0, 0, 0],
    },
  };
}
