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
let incrementalPointsAdded = 0;

function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    const maxLen = Math.max(a.length, b.length);
    let sum = 0;
    for (let i = 0; i < maxLen; i++) {
      const d = (a[i] ?? 0) - (b[i] ?? 0);
      sum += d * d;
    }
    return Math.sqrt(sum);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
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
    const dists = distances[i];
    if (dists.length === 0) { sigmas[i] = 1; continue; }

    let lo = 1e-10;
    let hi = 1.0;

    let hiEntropy = computeSigmaEntropy(dists, hi);
    while (hiEntropy < logTarget && hi < 1e8) {
      hi *= 2;
      hiEntropy = computeSigmaEntropy(dists, hi);
    }

    if (hiEntropy < logTarget) {
      sigmas[i] = hi;
      continue;
    }

    let mid = 1;
    for (let iter = 0; iter < 64; iter++) {
      mid = (lo + hi) / 2;
      const entropy = computeSigmaEntropy(dists, mid);

      if (!isFinite(entropy) || entropy < logTarget) {
        lo = mid;
      } else {
        hi = mid;
      }
      if (Math.abs(hi - lo) < 1e-8) break;
    }
    sigmas[i] = mid;
  }
  return sigmas;
}

function computeSigmaEntropy(dists: number[], sigma: number): number {
  let sumWeights = 0;
  const sigma2 = 2 * sigma * sigma;
  for (const d of dists) {
    sumWeights += Math.exp(-d * d / sigma2);
  }
  if (sumWeights < 1e-30) return -Infinity;
  return Math.log(sumWeights);
}

function graphKey(i: number, j: number): string {
  return `${i}:${j}`;
}

function parseGraphKey(key: string): [number, number] {
  const sep = key.indexOf(":");
  return [parseInt(key.substring(0, sep)), parseInt(key.substring(sep + 1))];
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
      const key = graphKey(i, j);
      graph.set(key, Math.max(graph.get(key) ?? 0, w));
    }
  }

  const symmetric = new Map<string, number>();
  for (const [key, w] of graph) {
    const [i, j] = parseGraphKey(key);
    const reverseKey = graphKey(j, i);
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

  const maxIter = Math.max(50, Math.min(200, dim * 2));
  const convergenceTol = 1e-6;

  const projections: number[][] = [];
  for (let comp = 0; comp < nComponents; comp++) {
    let axis = new Array(dim).fill(0);
    axis[comp % dim] = 1;

    for (let iter = 0; iter < maxIter; iter++) {
      const newAxis = new Array(dim).fill(0);
      for (const row of centered) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += row[d] * axis[d];
        for (let d = 0; d < dim; d++) newAxis[d] += dot * row[d];
      }

      for (let prev = 0; prev < comp; prev++) {
        const prevAxis = projections[prev];
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += newAxis[d] * prevAxis[d];
        for (let d = 0; d < dim; d++) newAxis[d] -= dot * prevAxis[d];
      }

      let norm = 0;
      for (let d = 0; d < dim; d++) norm += newAxis[d] * newAxis[d];
      norm = Math.sqrt(norm);
      if (norm < 1e-10) break;

      for (let d = 0; d < dim; d++) newAxis[d] /= norm;

      let changeSq = 0;
      for (let d = 0; d < dim; d++) {
        const diff = newAxis[d] - axis[d];
        changeSq += diff * diff;
      }
      axis = newAxis;

      if (changeSq < convergenceTol) break;
    }
    projections.push(axis);
  }

  const TARGET_STD = 1e-4;

  const rawEmbedding: number[][] = [];
  for (const row of centered) {
    const coords: number[] = [];
    for (let comp = 0; comp < nComponents; comp++) {
      let val = 0;
      for (let d = 0; d < dim; d++) val += row[d] * projections[comp][d];
      coords.push(val);
    }
    rawEmbedding.push(coords);
  }

  for (let comp = 0; comp < nComponents; comp++) {
    let mean = 0;
    for (const coords of rawEmbedding) mean += coords[comp];
    mean /= rawEmbedding.length;

    let variance = 0;
    for (const coords of rawEmbedding) {
      const diff = coords[comp] - mean;
      variance += diff * diff;
    }
    const std = Math.sqrt(variance / rawEmbedding.length);
    const scale = std > 1e-20 ? TARGET_STD / std : TARGET_STD;

    for (const coords of rawEmbedding) {
      coords[comp] = (coords[comp] - mean) * scale;
    }
  }

  return rawEmbedding;
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
    const [i, j] = parseGraphKey(key);
    if (i < j && w > 0.01) {
      edges.push({ i, j, w });
    }
  }

  if (edges.length === 0) return embedding;

  const DIST_EPS = 1e-6;
  const GRAD_CLIP = 4.0;

  const maxWeight = Math.max(...edges.map(e => e.w));
  const edgeProbabilities = edges.map(e => e.w / maxWeight);

  let rngState = 42;
  function fastRandom(): number {
    rngState = (rngState * 1664525 + 1013904223) & 0x7fffffff;
    return rngState / 0x7fffffff;
  }

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    const alpha = learningRate * 0.5 * (1.0 + Math.cos(Math.PI * epoch / nEpochs));

    for (let eIdx = 0; eIdx < edges.length; eIdx++) {
      if (fastRandom() > edgeProbabilities[eIdx]) continue;

      const { i, j } = edges[eIdx];
      let distSq = 0;
      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i][d] - embedding[j][d];
        distSq += diff * diff;
      }
      const distSqSafe = distSq + DIST_EPS;

      const powTerm = Math.pow(distSqSafe, b);
      const attractGrad = (-2.0 * a * b * Math.pow(distSqSafe, b - 1)) / (1 + a * powTerm);
      const clippedAttract = Math.max(-GRAD_CLIP, Math.min(GRAD_CLIP, attractGrad));

      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i][d] - embedding[j][d];
        const grad = clippedAttract * diff;
        embedding[i][d] -= alpha * grad;
        embedding[j][d] += alpha * grad;
      }

      const nNeg = Math.min(5, n - 1);
      for (let neg = 0; neg < nNeg; neg++) {
        let k = Math.floor(fastRandom() * n);
        if (k === i) k = (k + 1) % n;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i][d] - embedding[k][d];
          negDistSq += diff * diff;
        }
        const negDistSqSafe = negDistSq + DIST_EPS;

        const repelGrad = (2.0 * repulsionStrength * b) / (negDistSqSafe * (1 + a * Math.pow(negDistSqSafe, b)));
        const clippedRepel = Math.min(GRAD_CLIP, repelGrad);
        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i][d] - embedding[k][d];
          embedding[i][d] += alpha * clippedRepel * diff;
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

const genomeCache = new Map<string, number[] | null>();

export function buildEmbeddingDataset(): {
  formulas: string[];
  vectors: number[][];
  records: MaterialRecord[];
} {
  const formulas: string[] = [];
  const vectors: number[][] = [];
  const records: MaterialRecord[] = [];

  const uncached: { formula: string; record: MaterialRecord }[] = [];
  for (const [formula, record] of materialRecords) {
    const cached = genomeCache.get(formula);
    if (cached !== undefined) {
      if (cached !== null) {
        formulas.push(formula);
        vectors.push(cached);
        records.push(record);
      }
      continue;
    }
    uncached.push({ formula, record });
  }

  const BATCH_SIZE = 50;
  for (let start = 0; start < uncached.length; start += BATCH_SIZE) {
    const batch = uncached.slice(start, start + BATCH_SIZE);
    for (const { formula, record } of batch) {
      try {
        const genome = encodeGenome(formula);
        if (genome && genome.vector && genome.vector.length > 0) {
          genomeCache.set(formula, genome.vector);
          formulas.push(formula);
          vectors.push(genome.vector);
          records.push(record);
        } else {
          genomeCache.set(formula, null);
        }
      } catch {
        genomeCache.set(formula, null);
      }
    }
  }

  return { formulas, vectors, records };
}

function finalizeEmbedding(
  embedding: number[][],
  formulas: string[],
  vectors: number[][],
  records: MaterialRecord[],
): EmbeddingPoint[] {
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
  incrementalPointsAdded = 0;
  lastUpdateTime = Date.now();
  totalUpdates++;
  return points;
}

function computeUMAPPipeline(
  vectors: number[][],
  formulas: string[],
  cfg: UMAPConfig,
): number[][] {
  const k = Math.min(cfg.nNeighbors, formulas.length - 1);
  const { indices, distances } = findKNearestNeighbors(vectors, k);
  const sigmas = computeSigmas(distances, Math.log2(k));
  const graph = buildFuzzyGraph(indices, distances, sigmas);

  let embedding = spectralInitialization(vectors, cfg.nComponents);

  const nEpochsScaled = Math.min(cfg.nEpochs, Math.max(50, Math.floor(200 * Math.sqrt(30 / formulas.length))));
  embedding = umapOptimize(embedding, graph, formulas.length, { ...cfg, nEpochs: nEpochsScaled });
  return embedding;
}

let pendingAsyncCompute: { promise: Promise<EmbeddingPoint[]>; configKey: string } | null = null;

function configToKey(cfg: UMAPConfig): string {
  return `${cfg.nNeighbors}:${cfg.minDist}:${cfg.nComponents}:${cfg.nEpochs}:${cfg.learningRate}:${cfg.spread}:${cfg.repulsionStrength}`;
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
    incrementalPointsAdded = 0;
    lastUpdateTime = Date.now();
    totalUpdates++;
    return points;
  }

  const embedding = computeUMAPPipeline(vectors, formulas, cfg);
  return finalizeEmbedding(embedding, formulas, vectors, records);
}

export async function runFullUMAPAsync(
  config?: Partial<UMAPConfig>,
): Promise<EmbeddingPoint[]> {
  const cfg = { ...DEFAULT_UMAP_CONFIG, ...config };
  const key = configToKey(cfg);

  if (pendingAsyncCompute && pendingAsyncCompute.configKey === key) {
    return pendingAsyncCompute.promise;
  }

  const { formulas, vectors, records } = buildEmbeddingDataset();

  if (formulas.length < 3) {
    return runFullUMAP(config);
  }

  const promise = new Promise<EmbeddingPoint[]>((resolve, reject) => {
    setImmediate(() => {
      try {
        const embedding = computeUMAPPipeline(vectors, formulas, cfg);
        const points = finalizeEmbedding(embedding, formulas, vectors, records);
        resolve(points);
      } catch (err) {
        reject(err);
      } finally {
        if (pendingAsyncCompute?.configKey === key) {
          pendingAsyncCompute = null;
        }
      }
    });
  });

  pendingAsyncCompute = { promise, configKey: key };
  return promise;
}

export function incrementalUpdate(
  newFormulas: string[],
): EmbeddingPoint[] {
  const currentSize = materialRecords.size;
  const growthRatio = lastFullRecomputeSize > 0 ? currentSize / lastFullRecomputeSize : Infinity;
  const recomputeThreshold = currentSize < 1000 ? 1.5
    : currentSize < 5000 ? 2.0
    : currentSize < 20000 ? 3.0
    : 5.0;
  const needsFullRecompute = lastFullRecomputeSize === 0 ||
    growthRatio > recomputeThreshold ||
    embeddingStore.size < 3;

  if (needsFullRecompute) {
    return runFullUMAP();
  }

  const existingPoints = Array.from(embeddingStore.values());
  const newPoints: EmbeddingPoint[] = [];

  const TRANSFORM_EPOCHS = 30;
  const TRANSFORM_LR = 1.0;
  const DIST_EPS = 1e-6;
  const UMAP_A = 1.929;
  const UMAP_B = 0.7915;

  for (const formula of newFormulas) {
    if (embeddingStore.has(formula)) continue;
    const record = materialRecords.get(formula);
    if (!record) continue;

    try {
      const genome = encodeGenome(formula);
      if (!genome || !genome.vector || genome.vector.length === 0) continue;

      const kNearest = Math.min(10, existingPoints.length);
      const dists = existingPoints.map((p, idx) => ({
        idx,
        dist: euclideanDistance(genome.vector, p.genomeVector),
      }));
      dists.sort((a, b) => a.dist - b.dist);
      const neighbors = dists.slice(0, kNearest);

      const rho = neighbors.length > 0 ? neighbors[0].dist : 0;
      let sigmaLo = 1e-10;
      let sigmaHi = Math.max(100, neighbors[neighbors.length - 1]?.dist ?? 100);
      const logTarget = Math.log2(Math.min(kNearest, 5));
      let converged = false;
      for (let iter = 0; iter < 64; iter++) {
        let sumW = 0;
        for (const n of neighbors) {
          sumW += Math.exp(-Math.max(0, n.dist - rho) / (sigmaHi + 1e-10));
        }
        if (Math.log2(Math.max(sumW, 1e-30)) >= logTarget) break;
        sigmaHi *= 2;
        if (sigmaHi > 1e8) break;
      }
      for (let iter = 0; iter < 64; iter++) {
        const sigmaMid = (sigmaLo + sigmaHi) / 2;
        let sumW = 0;
        for (const n of neighbors) {
          sumW += Math.exp(-Math.max(0, n.dist - rho) / (sigmaMid + 1e-10));
        }
        const logSumW = Math.log2(Math.max(sumW, 1e-30));
        if (Math.abs(logSumW - logTarget) < 1e-5) { converged = true; break; }
        if (logSumW > logTarget) sigmaHi = sigmaMid;
        else sigmaLo = sigmaMid;
        if (Math.abs(sigmaHi - sigmaLo) < 1e-10) { converged = true; break; }
      }
      const localSigma = (sigmaLo + sigmaHi) / 2;

      const fuzzyWeights = neighbors.map(n => ({
        idx: n.idx,
        w: Math.exp(-Math.max(0, n.dist - rho) / (localSigma + 1e-10)),
      }));

      let totalW = 0;
      const position: [number, number, number] = [0, 0, 0];
      for (const fw of fuzzyWeights) {
        totalW += fw.w;
        for (let d = 0; d < 3; d++) {
          position[d] += fw.w * existingPoints[fw.idx].position3D[d];
        }
      }
      if (totalW < 1e-10) {
        for (let d = 0; d < 3; d++) {
          position[d] = existingPoints[neighbors[0].idx].position3D[d];
        }
      } else {
        for (let d = 0; d < 3; d++) position[d] /= totalW;
      }

      for (let epoch = 0; epoch < TRANSFORM_EPOCHS; epoch++) {
        const alpha = TRANSFORM_LR * 0.5 * (1.0 + Math.cos(Math.PI * epoch / TRANSFORM_EPOCHS));

        for (const fw of fuzzyWeights) {
          if (fw.w < 0.01) continue;
          const anchor = existingPoints[fw.idx].position3D;
          let distSq = 0;
          for (let d = 0; d < 3; d++) {
            const diff = position[d] - anchor[d];
            distSq += diff * diff;
          }
          const distSqSafe = distSq + DIST_EPS;
          const grad = (-2.0 * UMAP_A * UMAP_B * Math.pow(distSqSafe, UMAP_B - 1)) /
            (1 + UMAP_A * Math.pow(distSqSafe, UMAP_B));
          const clipped = Math.max(-4.0, Math.min(4.0, grad));
          for (let d = 0; d < 3; d++) {
            position[d] -= alpha * clipped * (position[d] - anchor[d]) * fw.w;
          }
        }

        for (let neg = 0; neg < 5; neg++) {
          const randIdx = Math.floor(Math.random() * existingPoints.length);
          const repel = existingPoints[randIdx].position3D;
          let negDistSq = 0;
          for (let d = 0; d < 3; d++) {
            const diff = position[d] - repel[d];
            negDistSq += diff * diff;
          }
          const negSafe = negDistSq + DIST_EPS;
          const repelGrad = (2.0 * UMAP_B) / (negSafe * (1 + UMAP_A * Math.pow(negSafe, UMAP_B)));
          const clippedRepel = Math.min(4.0, repelGrad);
          for (let d = 0; d < 3; d++) {
            position[d] += alpha * clippedRepel * (position[d] - repel[d]) * 0.1;
          }
        }
      }

      for (let d = 0; d < 3; d++) {
        position[d] = Math.round(position[d] * 1000) / 1000;
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

  incrementalPointsAdded += newPoints.length;
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
  const growthRatio = lastFullRecomputeSize > 0 ? currentSize / lastFullRecomputeSize : Infinity;
  const recomputeThreshold = currentSize < 1000 ? 1.5
    : currentSize < 5000 ? 2.0
    : currentSize < 20000 ? 3.0
    : 5.0;
  const needsFullRecompute = lastFullRecomputeSize === 0 ||
    growthRatio > recomputeThreshold ||
    embeddingStore.size < 5;

  let method: string;
  if (needsFullRecompute) {
    if (currentSize > 200) {
      runFullUMAPAsync();
      method = "full-recompute-async";
    } else {
      runFullUMAP();
      method = "full-recompute";
    }
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
  incrementalPointsSinceRecompute: number;
  staleness: { isStale: boolean; reason: string; ageMs: number; growthRatio: number; incrementalRatio: number };
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

  const ageMs = lastUpdateTime > 0 ? Date.now() - lastUpdateTime : 0;
  const growthRatio = lastFullRecomputeSize > 0 ? materialRecords.size / lastFullRecomputeSize : Infinity;
  const incrementalRatio = lastFullRecomputeSize > 0 ? incrementalPointsAdded / lastFullRecomputeSize : 0;

  let isStale = false;
  let staleReason = "up-to-date";
  if (lastFullRecomputeSize === 0 && materialRecords.size > 0) {
    isStale = true;
    staleReason = "no-embedding-computed";
  } else if (incrementalRatio > 0.3) {
    isStale = true;
    staleReason = "incremental-drift";
  } else if (growthRatio > 2.0) {
    isStale = true;
    staleReason = "dataset-growth";
  } else if (ageMs > 30 * 60 * 1000 && materialRecords.size > embeddingStore.size * 1.1) {
    isStale = true;
    staleReason = "age-with-unembedded";
  }

  return {
    totalMaterials: materialRecords.size,
    embeddedMaterials: embeddingStore.size,
    lastUpdateTime,
    totalUpdates,
    lastFullRecomputeSize,
    incrementalPointsSinceRecompute: incrementalPointsAdded,
    staleness: {
      isStale,
      reason: staleReason,
      ageMs,
      growthRatio: Math.round(growthRatio * 100) / 100,
      incrementalRatio: Math.round(incrementalRatio * 100) / 100,
    },
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
