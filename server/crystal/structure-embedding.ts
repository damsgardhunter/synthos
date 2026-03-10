import { getTrainingData } from "./crystal-structure-dataset";
import { buildGraphFromStructure, buildGraphFromFormula, getGraphFeatureVector, type StructureGraph } from "./crystal-graph-builder";
import type { CrystalGraph } from "../learning/graph-neural-net";
import { computeCompositionFeatures } from "../learning/composition-features";

const GRAPH_DIM = 35;
const COMP_DIM = 28;
const INPUT_DIM = GRAPH_DIM + COMP_DIM;
const HIDDEN_DIM = 32;
const EMBEDDING_DIM = 16;

interface AutoencoderWeights {
  encoderW1: number[][];
  encoderB1: number[];
  encoderW2: number[][];
  encoderB2: number[];
  decoderW1: number[][];
  decoderB1: number[];
  decoderW2: number[][];
  decoderB2: number[];
}

interface ClusterResult {
  centroid: number[];
  members: string[];
  size: number;
}

interface ClusteringResult {
  clusters: ClusterResult[];
  k: number;
  inertia: number;
}

const embeddingCache = new Map<string, number[]>();
const EMBEDDING_CACHE_MAX = 2000;
const EMBEDDING_CACHE_TTL = 30 * 60 * 1000;

let autoencoderWeights: AutoencoderWeights | null = null;
let currentClustering: ClusteringResult | null = null;
let trainingEmbeddings: Map<string, number[]> = new Map();
let initialized = false;
let initDeferred = false;
let predictionCount = 0;

function deferInit(): void {
  if (initDeferred) return;
  initDeferred = true;
  setTimeout(() => {
    try { initStructureEmbedding(); } catch {}
  }, 5000);
}

function relu(x: number): number {
  return Math.max(0, x);
}

function matVecMul(W: number[][], x: number[], bias: number[]): number[] {
  const out: number[] = new Array(W.length);
  for (let i = 0; i < W.length; i++) {
    let sum = bias[i];
    const row = W[i];
    for (let j = 0; j < row.length; j++) {
      sum += row[j] * (x[j] ?? 0);
    }
    out[i] = sum;
  }
  return out;
}

function initWeights(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols));
  const W: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((Math.random() * 2 - 1) * scale);
    }
    W.push(row);
  }
  return W;
}

function initBias(size: number): number[] {
  return new Array(size).fill(0);
}

function encode(input: number[], weights: AutoencoderWeights): number[] {
  const h1 = matVecMul(weights.encoderW1, input, weights.encoderB1).map(relu);
  return matVecMul(weights.encoderW2, h1, weights.encoderB2);
}

function decode(embedding: number[], weights: AutoencoderWeights): number[] {
  const h1 = matVecMul(weights.decoderW1, embedding, weights.decoderB1).map(relu);
  return matVecMul(weights.decoderW2, h1, weights.decoderB2);
}

function extractCompositionVector(formula: string): number[] {
  const cf = computeCompositionFeatures(formula);
  return [
    cf.enMean / 4, cf.enStd / 2, cf.enRange / 4, cf.enGeomMean / 4,
    cf.radiusMean / 300, cf.radiusStd / 150, cf.radiusRange / 300,
    cf.massMean / 250, cf.massStd / 100, cf.massMax / 250,
    cf.vecMean / 8, cf.vecStd / 4,
    cf.ieMean / 25, cf.ieStd / 10,
    cf.eaMean / 4, cf.eaStd / 2,
    cf.debyeMean / 1000, cf.debyeStd / 500,
    cf.bulkModMean / 400, cf.bulkModStd / 200,
    cf.dElectronFrac, cf.fElectronFrac, cf.pElectronFrac, cf.sElectronFrac,
    cf.shannonEntropy / 2, cf.nAtoms / 20,
    cf.pettiforMean / 100, cf.pettiforRange / 100,
  ];
}

function buildInputVector(formula: string): number[] {
  let graphVec: number[];
  try {
    const graph = buildGraphFromFormula(formula);
    graphVec = getGraphFeatureVector(graph);
  } catch {
    graphVec = new Array(GRAPH_DIM).fill(0);
  }

  while (graphVec.length < GRAPH_DIM) graphVec.push(0);
  graphVec = graphVec.slice(0, GRAPH_DIM);

  let compVec = extractCompositionVector(formula);
  while (compVec.length < COMP_DIM) compVec.push(0);
  compVec = compVec.slice(0, COMP_DIM);

  return [...graphVec, ...compVec];
}

function trainAutoencoder(data: number[][], epochs: number = 50, lr: number = 0.005): AutoencoderWeights {
  const weights: AutoencoderWeights = {
    encoderW1: initWeights(HIDDEN_DIM, INPUT_DIM),
    encoderB1: initBias(HIDDEN_DIM),
    encoderW2: initWeights(EMBEDDING_DIM, HIDDEN_DIM),
    encoderB2: initBias(EMBEDDING_DIM),
    decoderW1: initWeights(HIDDEN_DIM, EMBEDDING_DIM),
    decoderB1: initBias(HIDDEN_DIM),
    decoderW2: initWeights(INPUT_DIM, HIDDEN_DIM),
    decoderB2: initBias(INPUT_DIM),
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (const input of data) {
      const h1 = matVecMul(weights.encoderW1, input, weights.encoderB1);
      const h1a = h1.map(relu);
      const embedding = matVecMul(weights.encoderW2, h1a, weights.encoderB2);

      const h2 = matVecMul(weights.decoderW1, embedding, weights.decoderB1);
      const h2a = h2.map(relu);
      const reconstruction = matVecMul(weights.decoderW2, h2a, weights.decoderB2);

      const error = reconstruction.map((r, i) => r - (input[i] ?? 0));
      const loss = error.reduce((s, e) => s + e * e, 0) / error.length;
      totalLoss += loss;

      const dRecon = error.map(e => 2 * e / error.length);

      const dH2a = new Array(HIDDEN_DIM).fill(0);
      for (let i = 0; i < INPUT_DIM; i++) {
        for (let j = 0; j < HIDDEN_DIM; j++) {
          weights.decoderW2[i][j] -= lr * dRecon[i] * h2a[j];
          dH2a[j] += weights.decoderW2[i][j] * dRecon[i];
        }
        weights.decoderB2[i] -= lr * dRecon[i];
      }

      const dH2 = dH2a.map((d, i) => h2[i] > 0 ? d : 0);

      const dEmb = new Array(EMBEDDING_DIM).fill(0);
      for (let i = 0; i < HIDDEN_DIM; i++) {
        for (let j = 0; j < EMBEDDING_DIM; j++) {
          weights.decoderW1[i][j] -= lr * dH2[i] * embedding[j];
          dEmb[j] += weights.decoderW1[i][j] * dH2[i];
        }
        weights.decoderB1[i] -= lr * dH2[i];
      }

      const dH1a = new Array(HIDDEN_DIM).fill(0);
      for (let i = 0; i < EMBEDDING_DIM; i++) {
        for (let j = 0; j < HIDDEN_DIM; j++) {
          weights.encoderW2[i][j] -= lr * dEmb[i] * h1a[j];
          dH1a[j] += weights.encoderW2[i][j] * dEmb[i];
        }
        weights.encoderB2[i] -= lr * dEmb[i];
      }

      const dH1 = dH1a.map((d, i) => h1[i] > 0 ? d : 0);

      for (let i = 0; i < HIDDEN_DIM; i++) {
        for (let j = 0; j < INPUT_DIM; j++) {
          weights.encoderW1[i][j] -= lr * dH1[i] * (input[j] ?? 0);
        }
        weights.encoderB1[i] -= lr * dH1[i];
      }
    }
  }

  return weights;
}

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.max(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function kMeans(points: number[][], k: number, maxIter: number = 50): { centroids: number[][]; assignments: number[]; inertia: number } {
  const n = points.length;
  const dim = points[0]?.length ?? EMBEDDING_DIM;

  const centroids: number[][] = [];
  const used = new Set<number>();
  for (let i = 0; i < k && i < n; i++) {
    let idx: number;
    do {
      idx = Math.floor(Math.random() * n);
    } while (used.has(idx) && used.size < n);
    used.add(idx);
    centroids.push([...points[idx]]);
  }

  let assignments = new Array(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    const newAssignments = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let bestK = 0;
      for (let c = 0; c < k; c++) {
        const dist = euclideanDistance(points[i], centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          bestK = c;
        }
      }
      newAssignments[i] = bestK;
    }

    let changed = false;
    for (let i = 0; i < n; i++) {
      if (newAssignments[i] !== assignments[i]) {
        changed = true;
        break;
      }
    }
    assignments = newAssignments;

    for (let c = 0; c < k; c++) {
      const members = points.filter((_, i) => assignments[i] === c);
      if (members.length === 0) continue;
      for (let d = 0; d < dim; d++) {
        centroids[c][d] = members.reduce((s, p) => s + (p[d] ?? 0), 0) / members.length;
      }
    }

    if (!changed) break;
  }

  let inertia = 0;
  for (let i = 0; i < n; i++) {
    const dist = euclideanDistance(points[i], centroids[assignments[i]]);
    inertia += dist * dist;
  }

  return { centroids, assignments, inertia };
}

function silhouetteScore(points: number[][], assignments: number[], k: number): number {
  const n = points.length;
  if (n <= k || k <= 1) return 0;

  let totalScore = 0;
  let validCount = 0;

  for (let i = 0; i < n; i++) {
    const ci = assignments[i];

    const clusterMembers = points.filter((_, j) => assignments[j] === ci && j !== i);
    if (clusterMembers.length === 0) continue;

    const a = clusterMembers.reduce((s, p) => s + euclideanDistance(points[i], p), 0) / clusterMembers.length;

    let minB = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci) continue;
      const otherMembers = points.filter((_, j) => assignments[j] === c);
      if (otherMembers.length === 0) continue;
      const avgDist = otherMembers.reduce((s, p) => s + euclideanDistance(points[i], p), 0) / otherMembers.length;
      minB = Math.min(minB, avgDist);
    }

    if (!Number.isFinite(minB)) continue;

    const s = (minB - a) / Math.max(a, minB);
    totalScore += s;
    validCount++;
  }

  return validCount > 0 ? totalScore / validCount : 0;
}

export function initStructureEmbedding(): void {
  if (initialized) return;
  initialized = true;

  try {
    const allSeedData = getTrainingData();
    const seedData = allSeedData.length > 80 ? allSeedData.filter((_, i) => i % Math.ceil(allSeedData.length / 80) === 0) : allSeedData;
    const trainingVectors: number[][] = [];
    const formulas: string[] = [];

    for (const entry of seedData) {
      try {
        let graphVec: number[];
        if (entry.atomicPositions && entry.atomicPositions.length > 0) {
          const graph = buildGraphFromStructure(entry);
          graphVec = getGraphFeatureVector(graph);
        } else {
          graphVec = getGraphFeatureVector(buildGraphFromFormula(entry.formula));
        }
        while (graphVec.length < GRAPH_DIM) graphVec.push(0);
        graphVec = graphVec.slice(0, GRAPH_DIM);

        let compVec = extractCompositionVector(entry.formula);
        while (compVec.length < COMP_DIM) compVec.push(0);
        compVec = compVec.slice(0, COMP_DIM);

        const inputVec = [...graphVec, ...compVec];
        trainingVectors.push(inputVec);
        formulas.push(entry.formula);
      } catch {}
    }

    if (trainingVectors.length < 5) {
      console.log(`[StructureEmbedding] Not enough training data (${trainingVectors.length}), skipping`);
      return;
    }

    autoencoderWeights = trainAutoencoder(trainingVectors, 15, 0.005);

    for (let i = 0; i < formulas.length; i++) {
      const emb = encode(trainingVectors[i], autoencoderWeights);
      trainingEmbeddings.set(formulas[i], emb);
      embeddingCache.set(formulas[i], emb);
    }

    clusterStructures();

    console.log(`[StructureEmbedding] Initialized with ${trainingEmbeddings.size} embeddings, ${currentClustering?.k ?? 0} clusters`);
  } catch (err) {
    console.error(`[StructureEmbedding] Init failed:`, err instanceof Error ? err.message : String(err));
  }
}

export function computeStructureEmbedding(formula: string): number[] {
  if (!initialized) {
    deferInit();
    return new Array(EMBEDDING_DIM).fill(0);
  }
  predictionCount++;

  const cached = embeddingCache.get(formula);
  if (cached) return cached;

  if (!autoencoderWeights) {
    return new Array(EMBEDDING_DIM).fill(0);
  }

  const input = buildInputVector(formula);
  const embedding = encode(input, autoencoderWeights);

  if (embeddingCache.size >= EMBEDDING_CACHE_MAX) {
    const firstKey = embeddingCache.keys().next().value;
    if (firstKey) embeddingCache.delete(firstKey);
  }
  embeddingCache.set(formula, embedding);

  return embedding;
}

export function clusterStructures(): ClusteringResult {
  if (!initialized) deferInit();

  const allEmbeddings: number[][] = [];
  const allFormulas: string[] = [];

  trainingEmbeddings.forEach((emb, formula) => {
    allEmbeddings.push(emb);
    allFormulas.push(formula);
  });

  embeddingCache.forEach((emb, formula) => {
    if (!trainingEmbeddings.has(formula)) {
      allEmbeddings.push(emb);
      allFormulas.push(formula);
    }
  });

  if (allEmbeddings.length < 3) {
    currentClustering = { clusters: [{ centroid: new Array(EMBEDDING_DIM).fill(0), members: allFormulas, size: allFormulas.length }], k: 1, inertia: 0 };
    return currentClustering;
  }

  let bestK = 3;
  let bestScore = -1;
  let bestResult: ReturnType<typeof kMeans> | null = null;

  const maxK = Math.min(10, Math.floor(allEmbeddings.length / 2));

  for (let k = 3; k <= maxK; k++) {
    const result = kMeans(allEmbeddings, k);
    const score = silhouetteScore(allEmbeddings, result.assignments, k);
    if (score > bestScore) {
      bestScore = score;
      bestK = k;
      bestResult = result;
    }
  }

  if (!bestResult) {
    bestResult = kMeans(allEmbeddings, 3);
  }

  const clusters: ClusterResult[] = [];
  for (let c = 0; c < bestK; c++) {
    const memberIndices = allEmbeddings.map((_, i) => i).filter(i => bestResult!.assignments[i] === c);
    clusters.push({
      centroid: bestResult.centroids[c],
      members: memberIndices.map(i => allFormulas[i]),
      size: memberIndices.length,
    });
  }

  currentClustering = { clusters, k: bestK, inertia: bestResult.inertia };
  return currentClustering;
}

export function getClusterAssignment(embedding: number[]): number {
  if (!currentClustering || currentClustering.clusters.length === 0) return 0;

  let minDist = Infinity;
  let bestCluster = 0;

  for (let c = 0; c < currentClustering.clusters.length; c++) {
    const dist = euclideanDistance(embedding, currentClustering.clusters[c].centroid);
    if (dist < minDist) {
      minDist = dist;
      bestCluster = c;
    }
  }

  return bestCluster;
}

export function computeClusterNovelty(embedding: number[]): number {
  if (!currentClustering || currentClustering.clusters.length === 0) return 1.0;

  let minDist = Infinity;
  for (const cluster of currentClustering.clusters) {
    const dist = euclideanDistance(embedding, cluster.centroid);
    minDist = Math.min(minDist, dist);
  }

  let allDists: number[] = [];
  trainingEmbeddings.forEach(emb => {
    let d = Infinity;
    for (const cluster of currentClustering!.clusters) {
      d = Math.min(d, euclideanDistance(emb, cluster.centroid));
    }
    allDists.push(d);
  });

  if (allDists.length === 0) return Math.min(1.0, minDist);

  const avgDist = allDists.reduce((s, d) => s + d, 0) / allDists.length;
  const maxDist = Math.max(...allDists, 1e-6);

  return Math.min(1.0, minDist / (maxDist + 1e-6));
}

export function estimateStructureUncertainty(embedding: number[]): number {
  if (trainingEmbeddings.size === 0) return 1.0;

  let minDist = Infinity;
  trainingEmbeddings.forEach(trainEmb => {
    const dist = euclideanDistance(embedding, trainEmb);
    minDist = Math.min(minDist, dist);
  });

  let allDists: number[] = [];
  const embArr = Array.from(trainingEmbeddings.values());
  for (let i = 0; i < embArr.length; i++) {
    let nearest = Infinity;
    for (let j = 0; j < embArr.length; j++) {
      if (i === j) continue;
      nearest = Math.min(nearest, euclideanDistance(embArr[i], embArr[j]));
    }
    if (Number.isFinite(nearest)) allDists.push(nearest);
  }

  const avgNearestDist = allDists.length > 0 ? allDists.reduce((s, d) => s + d, 0) / allDists.length : 1.0;

  return Math.min(1.0, minDist / (avgNearestDist * 3 + 1e-6));
}

export function getEmbeddingStats(): {
  totalEmbeddings: number;
  trainingSize: number;
  cacheSize: number;
  clusterCount: number;
  clusterSizes: number[];
  avgNovelty: number;
  predictionCount: number;
  initialized: boolean;
} {
  const clusterSizes = currentClustering?.clusters.map(c => c.size) ?? [];

  let avgNovelty = 0;
  if (currentClustering && trainingEmbeddings.size > 0) {
    let totalNovelty = 0;
    let count = 0;
    trainingEmbeddings.forEach(emb => {
      totalNovelty += computeClusterNovelty(emb);
      count++;
    });
    avgNovelty = count > 0 ? totalNovelty / count : 0;
  }

  return {
    totalEmbeddings: embeddingCache.size,
    trainingSize: trainingEmbeddings.size,
    cacheSize: embeddingCache.size,
    clusterCount: currentClustering?.k ?? 0,
    clusterSizes,
    avgNovelty: Math.round(avgNovelty * 10000) / 10000,
    predictionCount,
    initialized,
  };
}

export function getClusters(): ClusteringResult | null {
  return currentClustering;
}
