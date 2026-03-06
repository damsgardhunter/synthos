import { ELEMENTAL_DATA, getElementData } from "./elemental-data";
import { extractFeatures } from "./ml-predictor";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { storage } from "../storage";

export interface NodeFeature {
  element: string;
  atomicNumber: number;
  electronegativity: number;
  atomicRadius: number;
  valenceElectrons: number;
  mass: number;
  embedding: number[];
}

export interface EdgeFeature {
  source: number;
  target: number;
  distance: number;
  bondOrderEstimate: number;
  features: number[];
}

export interface CrystalGraph {
  nodes: NodeFeature[];
  edges: EdgeFeature[];
  adjacency: number[][];
  formula: string;
}

interface GNNWeights {
  W_message: number[][];
  W_update: number[][];
  W_message2: number[][];
  W_update2: number[][];
  W_message3: number[][];
  W_update3: number[][];
  W_mlp1: number[][];
  b_mlp1: number[];
  W_mlp2: number[][];
  b_mlp2: number[];
  trainedAt: number;
  nSamples: number;
}

export interface GNNPrediction {
  formationEnergy: number;
  phononStability: boolean;
  predictedTc: number;
  confidence: number;
}

const NODE_DIM = 8;
const HIDDEN_DIM = 16;
const EDGE_DIM = 4;
const OUTPUT_DIM = 4;

let cachedGNNModel: GNNWeights | null = null;
let modelTrainedAt = 0;
const MODEL_STALE_MS = 30 * 60 * 1000;

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function initMatrix(rows: number, cols: number, rng: () => number, scale = 0.1): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((rng() - 0.5) * 2 * scale);
    }
    m.push(row);
  }
  return m;
}

function initVector(size: number, val = 0): number[] {
  return new Array(size).fill(val);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < mat.length; i++) {
    let sum = 0;
    for (let j = 0; j < vec.length; j++) {
      sum += (mat[i][j] ?? 0) * (vec[j] ?? 0);
    }
    result.push(sum);
  }
  return result;
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + (b[i] ?? 0));
}

function relu(v: number[]): number[] {
  return v.map(x => Math.max(0, x));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

export function buildCrystalGraph(formula: string, structure?: any): CrystalGraph {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const nodes: NodeFeature[] = [];
  let nodeIdx = 0;

  for (const el of elements) {
    const count = Math.round(counts[el]);
    const data = getElementData(el);
    const atomicNumber = data?.atomicNumber ?? 30;
    const en = data?.paulingElectronegativity ?? 1.5;
    const radius = data?.atomicRadius ?? 130;
    const valence = data?.valenceElectrons ?? 2;
    const mass = data?.atomicMass ?? 50;

    for (let i = 0; i < Math.min(count, 12); i++) {
      const embedding = [
        atomicNumber / 100,
        en / 4.0,
        radius / 250,
        valence / 8,
        mass / 250,
        (data?.debyeTemperature ?? 300) / 2000,
        (data?.bulkModulus ?? 50) / 500,
        (data?.firstIonizationEnergy ?? 7) / 25,
      ];
      nodes.push({ element: el, atomicNumber, electronegativity: en, atomicRadius: radius, valenceElectrons: valence, mass, embedding });
      nodeIdx++;
    }
  }

  if (nodes.length === 0) {
    nodes.push({
      element: "X", atomicNumber: 1, electronegativity: 1.5,
      atomicRadius: 100, valenceElectrons: 1, mass: 10,
      embedding: [0.01, 0.375, 0.4, 0.125, 0.04, 0.15, 0.1, 0.28],
    });
  }

  const edges: EdgeFeature[] = [];
  const adjacency: number[][] = nodes.map(() => []);

  const latticeParams = structure?.latticeParams;
  const hasPositions = structure?.atomicPositions && Array.isArray(structure.atomicPositions);

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      let distance: number;
      if (hasPositions && structure.atomicPositions[i] && structure.atomicPositions[j]) {
        const pi = structure.atomicPositions[i];
        const pj = structure.atomicPositions[j];
        const a = latticeParams?.a ?? 5;
        const b = latticeParams?.b ?? 5;
        const c = latticeParams?.c ?? 5;
        const dx = (pi.x - pj.x) * a;
        const dy = (pi.y - pj.y) * b;
        const dz = (pi.z - pj.z) * c;
        distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      } else {
        const ri = nodes[i].atomicRadius / 100;
        const rj = nodes[j].atomicRadius / 100;
        distance = (ri + rj) * (0.8 + 0.4 * Math.abs(i - j) / Math.max(nodes.length, 1));
      }

      const cutoff = 6.0;
      if (distance < cutoff || nodes.length <= 8) {
        const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
        const bondOrder = enDiff > 1.5 ? 0.5 : enDiff > 0.5 ? 1.0 : 1.5;

        const edgeFeats = [
          Math.min(distance / cutoff, 1.0),
          bondOrder / 2.0,
          enDiff / 3.0,
          (nodes[i].valenceElectrons + nodes[j].valenceElectrons) / 16.0,
        ];

        edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
        edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });

        adjacency[i].push(j);
        adjacency[j].push(i);
      }
    }
  }

  if (edges.length === 0 && nodes.length > 1) {
    for (let i = 0; i < nodes.length; i++) {
      const j = (i + 1) % nodes.length;
      const edgeFeats = [0.5, 0.5, 0.3, 0.3];
      edges.push({ source: i, target: j, distance: 2.5, bondOrderEstimate: 1.0, features: edgeFeats });
      edges.push({ source: j, target: i, distance: 2.5, bondOrderEstimate: 1.0, features: edgeFeats });
      adjacency[i].push(j);
      adjacency[j].push(i);
    }
  }

  return { nodes, edges, adjacency, formula };
}

export function messagePassingLayer(
  graph: CrystalGraph,
  W_message: number[][],
  W_update: number[][]
): number[][] {
  const nNodes = graph.nodes.length;
  const embeddings = graph.nodes.map(n => [...n.embedding]);

  const padded = embeddings.map(e => {
    while (e.length < HIDDEN_DIM) e.push(0);
    return e.slice(0, HIDDEN_DIM);
  });

  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    if (neighbors.length === 0) {
      newEmbeddings.push([...padded[i]]);
      continue;
    }

    const aggMessage = initVector(HIDDEN_DIM);
    for (const j of neighbors) {
      const msg = matVecMul(W_message, padded[j]);
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += msg[k] / neighbors.length;
      }
    }

    const combined = [...padded[i], ...aggMessage].slice(0, HIDDEN_DIM);
    const updated = relu(matVecMul(W_update, combined));
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i].slice(0, NODE_DIM);
    while (graph.nodes[i].embedding.length < NODE_DIM) {
      graph.nodes[i].embedding.push(0);
    }
  }

  return newEmbeddings;
}

export function GNNPredict(graph: CrystalGraph, weights: GNNWeights): GNNPrediction {
  messagePassingLayer(graph, weights.W_message, weights.W_update);
  messagePassingLayer(graph, weights.W_message2, weights.W_update2);
  messagePassingLayer(graph, weights.W_message3, weights.W_update3);

  const nNodes = graph.nodes.length;
  const meanPool = initVector(NODE_DIM);
  const maxPool = new Array(NODE_DIM).fill(-Infinity);

  for (const node of graph.nodes) {
    for (let k = 0; k < NODE_DIM; k++) {
      meanPool[k] += (node.embedding[k] ?? 0) / nNodes;
      maxPool[k] = Math.max(maxPool[k], node.embedding[k] ?? 0);
    }
  }

  const pooled = [...meanPool, ...maxPool.map(v => v === -Infinity ? 0 : v)];

  const h1 = relu(vecAdd(matVecMul(weights.W_mlp1, pooled), weights.b_mlp1));
  const out = vecAdd(matVecMul(weights.W_mlp2, h1), weights.b_mlp2);

  const formationEnergy = out[0] ?? 0;
  const phononStabilityRaw = sigmoid(out[1] ?? 0);
  const predictedTcRaw = Math.max(0, (out[2] ?? 0) * 100);
  const confidenceRaw = sigmoid(out[3] ?? 0);

  return {
    formationEnergy: Math.round(formationEnergy * 1000) / 1000,
    phononStability: phononStabilityRaw > 0.5,
    predictedTc: Math.round(Math.max(0, predictedTcRaw) * 10) / 10,
    confidence: Math.round(Math.max(0.05, Math.min(0.95, confidenceRaw)) * 100) / 100,
  };
}

function initWeights(rng: () => number): GNNWeights {
  return {
    W_message: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.15),
    W_update: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.15),
    W_message2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.15),
    W_update2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.15),
    W_message3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.15),
    W_update3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.15),
    W_mlp1: initMatrix(HIDDEN_DIM, NODE_DIM * 2, rng, 0.1),
    b_mlp1: initVector(HIDDEN_DIM),
    W_mlp2: initMatrix(OUTPUT_DIM, HIDDEN_DIM, rng, 0.1),
    b_mlp2: initVector(OUTPUT_DIM),
    trainedAt: 0,
    nSamples: 0,
  };
}

function cloneWeights(w: GNNWeights): GNNWeights {
  return {
    W_message: w.W_message.map(r => [...r]),
    W_update: w.W_update.map(r => [...r]),
    W_message2: w.W_message2.map(r => [...r]),
    W_update2: w.W_update2.map(r => [...r]),
    W_message3: w.W_message3.map(r => [...r]),
    W_update3: w.W_update3.map(r => [...r]),
    W_mlp1: w.W_mlp1.map(r => [...r]),
    b_mlp1: [...w.b_mlp1],
    W_mlp2: w.W_mlp2.map(r => [...r]),
    b_mlp2: [...w.b_mlp2],
    trainedAt: w.trainedAt,
    nSamples: w.nSamples,
  };
}

interface TrainingSample {
  formula: string;
  tc: number;
  formationEnergy?: number;
  structure?: any;
}

export function trainGNNSurrogate(trainingData: TrainingSample[]): GNNWeights {
  const rng = seededRandom(42);
  const weights = initWeights(rng);

  if (trainingData.length < 5) {
    weights.trainedAt = Date.now();
    weights.nSamples = trainingData.length;
    return weights;
  }

  const lr = 0.001;
  const epochs = 30;
  const batchSize = Math.min(32, trainingData.length);

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (let b = 0; b < Math.min(batchSize, trainingData.length); b++) {
      const idx = Math.floor(rng() * trainingData.length);
      const sample = trainingData[idx];

      const graph = buildCrystalGraph(sample.formula, sample.structure);
      const pred = GNNPredict(graph, weights);

      const tcTarget = sample.tc / 100;
      const feTarget = sample.formationEnergy ?? 0;
      const tcError = pred.predictedTc / 100 - tcTarget;
      const feError = pred.formationEnergy - feTarget;

      const loss = tcError * tcError + 0.1 * feError * feError;
      totalLoss += loss;

      const tcGrad = 2 * tcError * lr;
      const feGrad = 2 * feError * 0.1 * lr;

      for (let i = 0; i < weights.W_mlp2.length; i++) {
        for (let j = 0; j < weights.W_mlp2[i].length; j++) {
          const grad = i === 2 ? tcGrad : (i === 0 ? feGrad : 0);
          weights.W_mlp2[i][j] -= grad * (rng() * 0.5 + 0.5);
        }
        if (i === 2) weights.b_mlp2[i] -= tcGrad * 0.5;
        if (i === 0) weights.b_mlp2[i] -= feGrad * 0.5;
      }

      for (const wMat of [weights.W_message, weights.W_update, weights.W_message2, weights.W_update2, weights.W_message3, weights.W_update3, weights.W_mlp1]) {
        for (let i = 0; i < wMat.length; i++) {
          for (let j = 0; j < wMat[i].length; j++) {
            wMat[i][j] -= tcGrad * (rng() - 0.5) * 0.01;
          }
        }
      }
    }

    if (totalLoss / batchSize < 0.01) break;
  }

  weights.trainedAt = Date.now();
  weights.nSamples = trainingData.length;
  return weights;
}

export function getGNNModel(): GNNWeights {
  const now = Date.now();
  if (cachedGNNModel && (now - modelTrainedAt) < MODEL_STALE_MS) {
    return cachedGNNModel;
  }

  const trainingData: TrainingSample[] = SUPERCON_TRAINING_DATA
    .filter(e => e.isSuperconductor)
    .map(e => ({
      formula: e.formula,
      tc: e.tc,
      formationEnergy: undefined,
      structure: undefined,
    }));

  cachedGNNModel = trainGNNSurrogate(trainingData);
  modelTrainedAt = now;
  return cachedGNNModel;
}

export function invalidateGNNModel(): void {
  cachedGNNModel = null;
  modelTrainedAt = 0;
}

export function getGNNPrediction(formula: string, structure?: any): GNNPrediction {
  const weights = getGNNModel();
  const graph = buildCrystalGraph(formula, structure);
  return GNNPredict(graph, weights);
}
