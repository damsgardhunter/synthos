import { createHash } from "crypto";
import { ELEMENTAL_DATA, getElementData } from "./elemental-data";
import { extractFeatures } from "./ml-predictor";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { storage } from "../storage";
import { computeSymmetryEmbedding, computeSymmetryFeatureVector } from "../crystal/symmetry-subgroups";
import { predictLambda } from "./lambda-regressor";

export interface NodeFeature {
  element: string;
  atomicNumber: number;
  electronegativity: number;
  atomicRadius: number;
  valenceElectrons: number;
  mass: number;
  embedding: number[];
  multiplicity: number;
}

export interface EdgeFeature {
  source: number;
  target: number;
  distance: number;
  bondOrderEstimate: number;
  features: number[];
}

export interface ThreeBodyFeature {
  center: number;
  neighbor1: number;
  neighbor2: number;
  angle: number;
  distance1: number;
  distance2: number;
}

export interface CrystalGraph {
  nodes: NodeFeature[];
  edges: EdgeFeature[];
  threeBodyFeatures: ThreeBodyFeature[];
  adjacency: number[][];
  edgeIndex: (EdgeFeature | null)[];
  formula: string;
  prototype?: string;
  pressureGpa?: number;
}

function buildEdgeIndex(nodes: NodeFeature[], edges: EdgeFeature[]): (EdgeFeature | null)[] {
  const n = nodes.length;
  const idx: (EdgeFeature | null)[] = new Array(n * n).fill(null);
  for (const edge of edges) {
    const k = edge.source * n + edge.target;
    if (!idx[k]) idx[k] = edge;
  }
  return idx;
}

function getEdgeFromIndex(index: (EdgeFeature | null)[], n: number, i: number, j: number): EdgeFeature | null {
  return index[i * n + j] ?? index[j * n + i];
}

interface GNNWeights {
  W_message: number[][];
  W_update: number[][];
  W_message2: number[][];
  W_update2: number[][];
  W_message3: number[][];
  W_update3: number[][];
  W_message4: number[][];
  W_update4: number[][];
  W_attn_query: number[][];
  W_attn_key: number[][];
  W_attn_query2: number[][];
  W_attn_key2: number[][];
  W_attn_query3: number[][];
  W_attn_key3: number[][];
  W_attn_query4: number[][];
  W_attn_key4: number[][];
  W_conv_gate: number[][];
  W_conv_value: number[][];
  b_conv_gate: number[];
  b_conv_value: number[];
  W_input_proj: number[][];
  b_input_proj: number[];
  W_3body: number[][];
  W_3body_update: number[][];
  W_attn_pool: number[][];
  residual_gates: number[];
  W_pressure: number[];
  W_mlp1: number[][];
  b_mlp1: number[];
  W_mlp2: number[][];
  b_mlp2: number[];
  W_mlp2_var: number[][];
  b_mlp2_var: number[];
  trainedAt: number;
  nSamples: number;
}

export interface GNNPrediction {
  formationEnergy: number;
  phononStability: boolean;
  predictedTc: number;
  confidence: number;
  lambda: number;
  bandgap: number;
  dosProxy: number;
  stabilityProbability: number;
  latentEmbedding: number[];
  predictedTcVar: number;
  lambdaVar: number;
  formationEnergyVar: number;
  bandgapVar: number;
}

interface GNNForwardCache {
  pooled: number[];
  z1: number[];
  h1: number[];
  outRaw: number[];
  logVarOutRaw: number[];
  nodeEmbeddings: number[][];
  nodeMultiplicities: number[];
  totalMultiplicity: number;
}

export interface UncertaintyBreakdown {
  ensemble: number;
  mcDropout: number;
  aleatoric: number;
  latentDistance: number;
  perTarget: {
    tc: number;
    formationEnergy: number;
    lambda: number;
    bandgap: number;
  };
}

export interface GNNPredictionWithUncertainty {
  tc: number;
  formationEnergy: number;
  lambda: number;
  bandgap: number;
  dosProxy: number;
  stabilityProbability: number;
  uncertainty: number;
  uncertaintyBreakdown: UncertaintyBreakdown;
  phononStability: boolean;
  confidence: number;
  latentDistance: number;
  tcCI95: [number, number];
  lambdaCI95: [number, number];
  epistemicUncertainty: number;
  aleatoricUncertainty: number;
  totalStd: number;
}

const NODE_DIM = 32;
const HIDDEN_DIM = 48;
const EDGE_DIM = 24;
const OUTPUT_DIM = 16;
const CGCNN_CONCAT_DIM = HIDDEN_DIM * 2 + EDGE_DIM;
export const ENSEMBLE_SIZE = 5;
const MC_DROPOUT_PASSES = 10;
const MC_DROPOUT_RATE = 0.1;
const N_GAUSSIAN_BASIS = 20;
const GAUSSIAN_START = 0.5;
const GAUSSIAN_END = 6.0;
const GAUSSIAN_STEP = (GAUSSIAN_END - GAUSSIAN_START) / (N_GAUSSIAN_BASIS - 1);
const GAUSSIAN_WIDTH = GAUSSIAN_STEP;

let cachedEnsembleModels: GNNWeights[] | null = null;
let modelTrainedAt = 0;
const MODEL_STALE_MS = 6 * 60 * 60 * 1000;

const LATENT_REF_MAX = 200;
let trainingLatentEmbeddings: number[][] = [];

function computeLatentDistance(embedding: number[]): number {
  if (trainingLatentEmbeddings.length === 0) return 1.0;
  const norm = Math.sqrt(embedding.reduce((s, v) => s + v * v, 0)) || 1;
  let minDist = Infinity;
  for (const ref of trainingLatentEmbeddings) {
    const refNorm = Math.sqrt(ref.reduce((s, v) => s + v * v, 0)) || 1;
    let dotProduct = 0;
    for (let i = 0; i < Math.min(embedding.length, ref.length); i++) {
      dotProduct += embedding[i] * ref[i];
    }
    const cosineSim = dotProduct / (norm * refNorm);
    const dist = 1.0 - Math.max(-1, Math.min(1, cosineSim));
    if (dist < minDist) minDist = dist;
  }
  return Math.min(1.0, minDist);
}

function updateTrainingEmbeddings(trainingData: { formula: string; tc: number }[], weights: GNNWeights): void {
  trainingLatentEmbeddings = [];
  const sampleCount = Math.min(LATENT_REF_MAX, trainingData.length);
  const step = Math.max(1, Math.floor(trainingData.length / sampleCount));
  for (let i = 0; i < trainingData.length && trainingLatentEmbeddings.length < LATENT_REF_MAX; i += step) {
    try {
      const graph = buildCrystalGraph(trainingData[i].formula);
      const pred = GNNPredict(graph, weights);
      trainingLatentEmbeddings.push(pred.latentEmbedding);
    } catch { /* skip invalid formulas */ }
  }
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
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

function initMatrix(rows: number, cols: number, rng: () => number, scale?: number): number[][] {
  const heScale = scale ?? Math.sqrt(2.0 / cols);
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((rng() - 0.5) * 2 * heScale);
    }
    m.push(row);
  }
  return m;
}

const _bufferPool: Map<number, Float64Array[]> = new Map();

function acquireBuffer(size: number): Float64Array {
  const pool = _bufferPool.get(size);
  if (pool && pool.length > 0) {
    return pool.pop()!;
  }
  return new Float64Array(size);
}

function releaseBuffer(buf: Float64Array): void {
  const size = buf.length;
  let pool = _bufferPool.get(size);
  if (!pool) {
    pool = [];
    _bufferPool.set(size, pool);
  }
  if (pool.length < 64) {
    pool.push(buf);
  }
}

function toArray(buf: Float64Array): number[] {
  return Array.from(buf);
}

function layerNorm(vec: number[], eps: number = 1e-5): number[] {
  const n = vec.length;
  if (n === 0) return vec;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += vec[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = vec[i] - mean;
    variance += d * d;
  }
  variance /= n;
  const std = Math.sqrt(variance + eps);
  const out = acquireBuffer(n);
  for (let i = 0; i < n; i++) out[i] = (vec[i] - mean) / std;
  const result = toArray(out);
  releaseBuffer(out);
  return result;
}

function initVector(size: number, val = 0): number[] {
  return new Array(size).fill(val);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
  const { flat, rows, cols } = getFlatMat(mat);
  if (cols !== vec.length) {
    throw new Error(
      `matVecMul shape mismatch: mat has ${cols} cols but vec has ${vec.length} elements`
    );
  }
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    let sum = 0;
    for (let j = 0; j < cols; j++) sum += flat[offset + j] * vec[j];
    result[i] = sum;
  }
  return result;
}

function vecAdd(a: number[], b: number[]): number[] {
  const n = a.length;
  const out = acquireBuffer(n);
  for (let i = 0; i < n; i++) out[i] = a[i] + (b[i] ?? 0);
  const result = toArray(out);
  releaseBuffer(out);
  return result;
}

function relu(v: number[]): number[] {
  const n = v.length;
  const out = acquireBuffer(n);
  for (let i = 0; i < n; i++) out[i] = v[i] > 0 ? v[i] : 0;
  const result = toArray(out);
  releaseBuffer(out);
  return result;
}

function leakyRelu(v: number[], alpha: number = 0.01): number[] {
  const n = v.length;
  const out = acquireBuffer(n);
  for (let i = 0; i < n; i++) out[i] = v[i] >= 0 ? v[i] : alpha * v[i];
  const result = toArray(out);
  releaseBuffer(out);
  return result;
}

const _flatMatCache = new WeakMap<number[][], { flat: Float32Array; rows: number; cols: number }>();

function getFlatMat(mat: number[][]): { flat: Float32Array; rows: number; cols: number } {
  let cached = _flatMatCache.get(mat);
  if (cached) return cached;
  const rows = mat.length;
  const cols = rows > 0 ? mat[0].length : 0;
  const flat = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const row = mat[i];
    const offset = i * cols;
    for (let j = 0; j < cols; j++) flat[offset + j] = row[j];
  }
  cached = { flat, rows, cols };
  _flatMatCache.set(mat, cached);
  return cached;
}

function invalidateFlatCache(mat: number[][]): void {
  _flatMatCache.delete(mat);
}

function fusedMatVecLeakyRelu(mat: number[][], vec: number[], alpha: number = 0.01): number[] {
  const { flat, rows, cols } = getFlatMat(mat);
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    let sum = 0;
    for (let j = 0; j < cols; j++) sum += flat[offset + j] * vec[j];
    result[i] = sum >= 0 ? sum : alpha * sum;
  }
  return result;
}

function fusedMatVecAddLeakyRelu(mat: number[][], vec: number[], bias: number[], alpha: number = 0.01): number[] {
  const { flat, rows, cols } = getFlatMat(mat);
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const offset = i * cols;
    let sum = bias[i] ?? 0;
    for (let j = 0; j < cols; j++) sum += flat[offset + j] * vec[j];
    result[i] = sum >= 0 ? sum : alpha * sum;
  }
  return result;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function softplus(x: number): number {
  if (x > 20) return x;
  return Math.log(1 + Math.exp(x));
}

const COSINE_CUTOFF_RADIUS = 6.0;

function cosineCutoff(distance: number): number {
  if (distance >= COSINE_CUTOFF_RADIUS) return 0;
  if (distance <= 0) return 1;
  return 0.5 * (Math.cos(Math.PI * distance / COSINE_CUTOFF_RADIUS) + 1);
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function softmax(values: number[]): number[] {
  const n = values.length;
  if (n === 0) return [];
  let maxVal = values[0];
  for (let i = 1; i < n; i++) if (values[i] > maxVal) maxVal = values[i];
  const out = acquireBuffer(n);
  let sumExps = 0;
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(Math.min(values[i] - maxVal, 20));
    sumExps += out[i];
  }
  const invSum = 1 / Math.max(sumExps, 1e-10);
  for (let i = 0; i < n; i++) out[i] *= invSum;
  const result = toArray(out);
  releaseBuffer(out);
  return result;
}

const _gaussianBuffer = new Float64Array(N_GAUSSIAN_BASIS);
const _invTwoSigmaSq = 1 / (2 * GAUSSIAN_WIDTH * GAUSSIAN_WIDTH);
const EDGE_FEAT_DIM = N_GAUSSIAN_BASIS + 4;
const _edgeFeatBuffer = new Float64Array(EDGE_FEAT_DIM);

function gaussianDistanceExpansion(distance: number): number[] {
  for (let i = 0; i < N_GAUSSIAN_BASIS; i++) {
    const diff = distance - (GAUSSIAN_START + i * GAUSSIAN_STEP);
    _gaussianBuffer[i] = Math.exp(-(diff * diff) * _invTwoSigmaSq);
  }
  return Array.from(_gaussianBuffer);
}

function buildEdgeFeatures(distance: number, bondOrder: number, enDiff: number, ionicCharacter: number, radiusSum: number): number[] {
  for (let i = 0; i < N_GAUSSIAN_BASIS; i++) {
    const diff = distance - (GAUSSIAN_START + i * GAUSSIAN_STEP);
    _edgeFeatBuffer[i] = Math.exp(-(diff * diff) * _invTwoSigmaSq);
  }
  _edgeFeatBuffer[N_GAUSSIAN_BASIS] = bondOrder / 2.0;
  _edgeFeatBuffer[N_GAUSSIAN_BASIS + 1] = enDiff / 3.0;
  _edgeFeatBuffer[N_GAUSSIAN_BASIS + 2] = ionicCharacter;
  _edgeFeatBuffer[N_GAUSSIAN_BASIS + 3] = radiusSum;
  return Array.from(_edgeFeatBuffer);
}

function buildDefaultEdgeFeatures(): number[] {
  return buildEdgeFeatures(2.5, 1.0, 0.9, 0.3, 0.5);
}

function applyDropout(vec: number[], rate: number, rng: () => number): number[] {
  if (rate <= 0) return vec;
  const n = vec.length;
  const scale = 1.0 / (1.0 - rate);
  const out = acquireBuffer(n);
  for (let i = 0; i < n; i++) out[i] = rng() < rate ? 0 : vec[i] * scale;
  const result = toArray(out);
  releaseBuffer(out);
  return result;
}

function getPeriod(atomicNumber: number): number {
  if (atomicNumber <= 2) return 1;
  if (atomicNumber <= 10) return 2;
  if (atomicNumber <= 18) return 3;
  if (atomicNumber <= 36) return 4;
  if (atomicNumber <= 54) return 5;
  if (atomicNumber <= 86) return 6;
  return 7;
}

function getGroup(atomicNumber: number): number {
  const groupMap: Record<number, number> = {
    1: 1, 2: 18, 3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,
    11: 1, 12: 2, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10,
    29: 11, 30: 12, 31: 13, 32: 14, 33: 15, 34: 16, 35: 17, 36: 18,
    37: 1, 38: 2, 39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10,
    47: 11, 48: 12, 49: 13, 50: 14, 51: 15, 52: 16, 53: 17, 54: 18,
    55: 1, 56: 2, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10,
    79: 11, 80: 12, 81: 13, 82: 14, 83: 15,
  };
  return groupMap[atomicNumber] ?? 0;
}

function getBlockEncoding(atomicNumber: number): number {
  if (atomicNumber <= 2) return 0;
  if ([3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88].includes(atomicNumber)) return 0;
  if (atomicNumber >= 57 && atomicNumber <= 71) return 0.75;
  if (atomicNumber >= 89 && atomicNumber <= 103) return 0.75;
  if ((atomicNumber >= 21 && atomicNumber <= 30) || (atomicNumber >= 39 && atomicNumber <= 48) || (atomicNumber >= 72 && atomicNumber <= 80)) return 0.5;
  if ((atomicNumber >= 5 && atomicNumber <= 10) || (atomicNumber >= 13 && atomicNumber <= 18) || (atomicNumber >= 31 && atomicNumber <= 36) || (atomicNumber >= 49 && atomicNumber <= 54) || (atomicNumber >= 81 && atomicNumber <= 86)) return 0.25;
  return 0;
}

function getSOrbitalOccupancy(atomicNumber: number): number {
  if (atomicNumber <= 2) return atomicNumber / 2;
  if (atomicNumber <= 10) return 1.0;
  if (atomicNumber <= 18) return 1.0;
  if (atomicNumber <= 36) return 1.0;
  if (atomicNumber <= 54) return 1.0;
  return 1.0;
}

function getPOrbitalOccupancy(atomicNumber: number): number {
  if (atomicNumber <= 4) return 0;
  if (atomicNumber >= 5 && atomicNumber <= 10) return Math.min((atomicNumber - 4) / 6, 1.0);
  if (atomicNumber >= 13 && atomicNumber <= 18) return Math.min((atomicNumber - 12) / 6, 1.0);
  if (atomicNumber >= 31 && atomicNumber <= 36) return Math.min((atomicNumber - 30) / 6, 1.0);
  if (atomicNumber >= 49 && atomicNumber <= 54) return Math.min((atomicNumber - 48) / 6, 1.0);
  if (atomicNumber >= 81 && atomicNumber <= 86) return Math.min((atomicNumber - 80) / 6, 1.0);
  return 0;
}

function getDOrbitalOccupancy(atomicNumber: number): number {
  if (atomicNumber >= 21 && atomicNumber <= 30) return Math.min((atomicNumber - 20) / 10, 1.0);
  if (atomicNumber >= 39 && atomicNumber <= 48) return Math.min((atomicNumber - 38) / 10, 1.0);
  if (atomicNumber >= 72 && atomicNumber <= 80) return Math.min((atomicNumber - 71) / 10, 1.0);
  if (atomicNumber >= 57 && atomicNumber <= 71) return 0.1;
  if (atomicNumber >= 89 && atomicNumber <= 103) return 0.1;
  return 0;
}

function getFOrbitalOccupancy(atomicNumber: number): number {
  if (atomicNumber >= 57 && atomicNumber <= 71) return Math.min((atomicNumber - 56) / 14, 1.0);
  if (atomicNumber >= 89 && atomicNumber <= 103) return Math.min((atomicNumber - 88) / 14, 1.0);
  return 0;
}

function computeMagneticMomentProxy(atomicNumber: number): number {
  const unpairedElectrons: Record<number, number> = {
    24: 6, 25: 5, 26: 4, 27: 3, 28: 2, 29: 1,
    42: 6, 43: 5, 44: 4, 45: 3, 46: 0, 47: 1,
    74: 4, 75: 5, 76: 4, 77: 3, 78: 2, 79: 1,
    57: 1, 58: 2, 59: 3, 60: 4, 61: 5, 62: 6, 63: 7,
    64: 7, 65: 6, 66: 5, 67: 4, 68: 3, 69: 2, 70: 1, 71: 0,
  };
  const n = unpairedElectrons[atomicNumber] ?? 0;
  return Math.min(1.0, Math.sqrt(n * (n + 2)) / 8.0);
}

function getOrbitalBlock(atomicNumber: number): number {
  if (atomicNumber <= 0) return 0;
  const sBlock = new Set([1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88]);
  const fBlock = new Set([
    57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
    89,90,91,92,93,94,95,96,97,98,99,100,101,102,103
  ]);
  if (sBlock.has(atomicNumber)) return 0.0;
  if (fBlock.has(atomicNumber)) return 1.0;
  if (atomicNumber <= 2) return 0.0;
  const dBlockRanges = [[21,30],[39,48],[72,80],[104,112]];
  for (const [lo, hi] of dBlockRanges) {
    if (atomicNumber >= lo && atomicNumber <= hi) return 0.667;
  }
  return 0.333;
}

function computeValenceShellEncoding(atomicNumber: number, valenceElectrons: number): number {
  const blockFeature = getOrbitalBlock(atomicNumber);

  let maxOccupancy: number;
  if (blockFeature < 0.1) maxOccupancy = 2;
  else if (blockFeature < 0.4) maxOccupancy = 6;
  else if (blockFeature < 0.8) maxOccupancy = 10;
  else maxOccupancy = 14;

  const shellFill = Math.min(1.0, valenceElectrons / maxOccupancy);
  return shellFill * 0.5 + blockFeature * 0.5;
}

function pressureAwareBondOrder(enDiff: number, atomicNumberI: number, atomicNumberJ: number, pressureGpa: number): number {
  let bondOrder = enDiff > 1.5 ? 0.5 : enDiff > 0.5 ? 1.0 : 1.5;

  if (pressureGpa > 100) {
    const isLightI = atomicNumberI <= 10;
    const isLightJ = atomicNumberJ <= 10;
    if (isLightI && isLightJ) {
      const pressureFactor = Math.min((pressureGpa - 100) / 200, 1.0);
      bondOrder += pressureFactor * 1.5;
    } else if (isLightI || isLightJ) {
      const pressureFactor = Math.min((pressureGpa - 100) / 300, 1.0);
      bondOrder += pressureFactor * 0.8;
    }
  }
  return bondOrder;
}

function pressureDistanceScale(pressureGpa: number): number {
  if (pressureGpa <= 0) return 1.0;
  const B0 = 150;
  const Bp = 4.0;
  const ratio = 1 + (Bp * pressureGpa) / B0;
  const volumeRatio = Math.pow(ratio, -1.0 / Bp);
  return Math.cbrt(volumeRatio);
}

function getCovalentRadius(atomicNumber: number, atomicRadius: number): number {
  const covalentRadii: Record<number, number> = {
    1: 31, 2: 28, 3: 128, 4: 96, 5: 84, 6: 76, 7: 71, 8: 66, 9: 57, 10: 58,
    11: 166, 12: 141, 13: 121, 14: 111, 15: 107, 16: 105, 17: 102, 18: 106,
    19: 203, 20: 176, 21: 170, 22: 160, 23: 153, 24: 139, 25: 150, 26: 142,
    27: 138, 28: 124, 29: 132, 30: 122, 31: 122, 32: 120, 33: 119, 34: 120,
    35: 120, 36: 116, 37: 220, 38: 195, 39: 190, 40: 175, 41: 164, 42: 154,
    43: 147, 44: 146, 45: 142, 46: 139, 47: 145, 48: 144, 49: 142, 50: 139,
    51: 139, 52: 138, 53: 139, 54: 140, 55: 244, 56: 215, 57: 207, 72: 175,
    73: 170, 74: 162, 75: 151, 76: 144, 77: 141, 78: 136, 79: 136, 80: 132,
    81: 145, 82: 146, 83: 148, 90: 206, 92: 196,
  };
  return covalentRadii[atomicNumber] ?? (atomicRadius * 0.85);
}

function getMendeleevNumber(atomicNumber: number): number {
  const mendeleevMap: Record<number, number> = {
    1: 92, 2: 98, 3: 1, 4: 67, 5: 72, 6: 77, 7: 82, 8: 87, 9: 93, 10: 99,
    11: 2, 12: 68, 13: 73, 14: 78, 15: 83, 16: 88, 17: 94, 18: 100, 19: 3,
    20: 7, 21: 11, 22: 43, 23: 44, 24: 45, 25: 46, 26: 47, 27: 48, 28: 49,
    29: 50, 30: 69, 31: 74, 32: 79, 33: 84, 34: 89, 35: 95, 36: 101, 37: 4,
    38: 8, 39: 12, 40: 51, 41: 52, 42: 53, 43: 54, 44: 55, 45: 56, 46: 57,
    47: 58, 48: 70, 49: 75, 50: 80, 51: 85, 52: 90, 53: 96, 54: 102, 55: 5,
    56: 9, 57: 13, 72: 59, 73: 60, 74: 61, 75: 62, 76: 63, 77: 64, 78: 65,
    79: 66, 80: 71, 81: 76, 82: 81, 83: 86, 90: 16, 92: 17,
  };
  return mendeleevMap[atomicNumber] ?? atomicNumber;
}

interface PrototypeCoordination {
  siteLabels: string[];
  coordinations: Record<string, { neighbors: string[]; count: number }>;
  latticeParams: { a: number; b: number; c: number };
}

const PROTOTYPE_COORDINATIONS: Record<string, PrototypeCoordination> = {
  "AlB2": {
    siteLabels: ["A", "B"],
    coordinations: {
      "A": { neighbors: ["B"], count: 12 },
      "B": { neighbors: ["B", "A"], count: 5 },
    },
    latticeParams: { a: 3.08, b: 3.08, c: 3.52 },
  },
  "Perovskite": {
    siteLabels: ["A", "B", "O"],
    coordinations: {
      "A": { neighbors: ["O"], count: 12 },
      "B": { neighbors: ["O"], count: 6 },
      "O": { neighbors: ["B", "A"], count: 4 },
    },
    latticeParams: { a: 3.90, b: 3.90, c: 3.90 },
  },
  "A15": {
    siteLabels: ["A", "B"],
    coordinations: {
      "A": { neighbors: ["A", "B"], count: 14 },
      "B": { neighbors: ["A"], count: 12 },
    },
    latticeParams: { a: 5.29, b: 5.29, c: 5.29 },
  },
  "Clathrate": {
    siteLabels: ["M", "H"],
    coordinations: {
      "M": { neighbors: ["H"], count: 24 },
      "H": { neighbors: ["H", "M"], count: 5 },
    },
    latticeParams: { a: 5.10, b: 5.10, c: 5.10 },
  },
  "ThCr2Si2": {
    siteLabels: ["A", "B", "C"],
    coordinations: {
      "A": { neighbors: ["C"], count: 8 },
      "B": { neighbors: ["C"], count: 4 },
      "C": { neighbors: ["C", "B", "A"], count: 5 },
    },
    latticeParams: { a: 3.96, b: 3.96, c: 13.02 },
  },
  "Spinel": {
    siteLabels: ["A", "B", "O"],
    coordinations: {
      "A": { neighbors: ["O"], count: 4 },
      "B": { neighbors: ["O"], count: 6 },
      "O": { neighbors: ["A", "B"], count: 4 },
    },
    latticeParams: { a: 8.08, b: 8.08, c: 8.08 },
  },
  "MAX": {
    siteLabels: ["M", "A", "X"],
    coordinations: {
      "M": { neighbors: ["X", "A", "M"], count: 9 },
      "A": { neighbors: ["M"], count: 6 },
      "X": { neighbors: ["M"], count: 6 },
    },
    latticeParams: { a: 3.06, b: 3.06, c: 13.60 },
  },
  "Layered-nitride": {
    siteLabels: ["A", "M", "N", "X"],
    coordinations: {
      "A": { neighbors: ["N"], count: 3 },
      "M": { neighbors: ["N", "X"], count: 6 },
      "N": { neighbors: ["M", "A"], count: 4 },
      "X": { neighbors: ["M"], count: 3 },
    },
    latticeParams: { a: 3.60, b: 3.60, c: 27.0 },
  },
  "Laves": {
    siteLabels: ["A", "B"],
    coordinations: {
      "A": { neighbors: ["B", "A"], count: 16 },
      "B": { neighbors: ["B", "A"], count: 12 },
    },
    latticeParams: { a: 7.39, b: 7.39, c: 7.39 },
  },
  "Heusler": {
    siteLabels: ["A", "B", "C"],
    coordinations: {
      "A": { neighbors: ["C", "B"], count: 8 },
      "B": { neighbors: ["A", "C"], count: 8 },
      "C": { neighbors: ["A", "B"], count: 8 },
    },
    latticeParams: { a: 5.65, b: 5.65, c: 5.65 },
  },
  "Rock-salt": {
    siteLabels: ["A", "B"],
    coordinations: {
      "A": { neighbors: ["B"], count: 6 },
      "B": { neighbors: ["A"], count: 6 },
    },
    latticeParams: { a: 5.64, b: 5.64, c: 5.64 },
  },
  "Fluorite": {
    siteLabels: ["A", "X"],
    coordinations: {
      "A": { neighbors: ["X"], count: 8 },
      "X": { neighbors: ["A"], count: 4 },
    },
    latticeParams: { a: 5.46, b: 5.46, c: 5.46 },
  },
};

function gcd(a: number, b: number): number {
  a = Math.round(a); b = Math.round(b);
  while (b) { const t = b; b = a % b; a = t; }
  return Math.abs(a);
}

function normalizeFormulaCounts(counts: Record<string, number>): { normalized: Record<string, number>; multiplicities: Record<string, number> } {
  const elements = Object.keys(counts);
  const rounded = elements.map(el => Math.max(1, Math.round(counts[el])));
  let g = rounded[0];
  for (let i = 1; i < rounded.length; i++) g = gcd(g, rounded[i]);
  if (g < 1) g = 1;

  const normalized: Record<string, number> = {};
  const multiplicities: Record<string, number> = {};
  const MAX_NODES_PER_ELEMENT = 8;
  for (let i = 0; i < elements.length; i++) {
    const reduced = rounded[i] / g;
    const nodeCount = Math.min(reduced, MAX_NODES_PER_ELEMENT);
    const mult = rounded[i] / (g * nodeCount);
    normalized[elements[i]] = nodeCount;
    multiplicities[elements[i]] = mult;
  }
  return { normalized, multiplicities };
}

const SITE_TYPICAL_RADII: Record<string, Record<string, number>> = {
  "Perovskite": { "A": 160, "B": 60, "O": 73 },
  "Spinel": { "A": 65, "B": 65, "O": 73 },
  "Heusler": { "A": 140, "B": 125, "C": 110 },
  "Laves": { "A": 160, "B": 130 },
  "MAX": { "M": 140, "A": 125, "X": 70 },
  "Rock-salt": { "A": 130, "B": 100 },
  "Fluorite": { "A": 110, "X": 130 },
};

const SITE_TYPICAL_COORD: Record<string, Record<string, number>> = {
  "Perovskite": { "A": 12, "B": 6, "O": 4 },
  "Spinel": { "A": 4, "B": 6, "O": 4 },
  "Heusler": { "A": 8, "B": 8, "C": 8 },
  "Laves": { "A": 16, "B": 12 },
  "MAX": { "M": 9, "A": 6, "X": 6 },
  "Rock-salt": { "A": 6, "B": 6 },
  "Fluorite": { "A": 8, "X": 4 },
};

function assignSiteLabels(formula: string, prototype: string): Record<string, string> {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];
  if (!protoInfo) return {};

  const siteLabels = protoInfo.siteLabels;
  const assignment: Record<string, string> = {};

  if (elements.length <= 1 || siteLabels.length <= 1) {
    for (let i = 0; i < elements.length; i++) {
      assignment[elements[i]] = siteLabels[Math.min(i, siteLabels.length - 1)];
    }
    return assignment;
  }

  const typicalRadii = SITE_TYPICAL_RADII[prototype];
  const typicalCoord = SITE_TYPICAL_COORD[prototype];

  if (typicalRadii && elements.length <= siteLabels.length) {
    const nEl = elements.length;
    const nSites = siteLabels.length;
    let bestAssignment: Record<string, string> = {};
    let bestScore = -Infinity;

    const permute = (remaining: string[], usedSites: Set<string>, current: Record<string, string>) => {
      if (remaining.length === 0) {
        let score = 0;
        for (const [el, site] of Object.entries(current)) {
          const data = getElementData(el);
          const elRadius = data?.atomicRadius ?? 130;
          const siteRadius = typicalRadii[site];
          if (siteRadius) {
            score -= Math.abs(elRadius - siteRadius) / 100;
          }
          const elEN = data?.paulingElectronegativity ?? 1.5;
          const siteCoord = typicalCoord?.[site];
          if (siteCoord) {
            const elValence = data?.valenceElectrons ?? 2;
            score -= Math.abs(elValence - siteCoord / 2) * 0.1;
          }
          if (protoInfo.coordinations[site]) {
            score += 0.5;
          }
        }
        if (score > bestScore) {
          bestScore = score;
          bestAssignment = { ...current };
        }
        return;
      }

      const el = remaining[0];
      const rest = remaining.slice(1);
      for (const site of siteLabels) {
        if (usedSites.has(site)) continue;
        current[el] = site;
        usedSites.add(site);
        permute(rest, usedSites, current);
        usedSites.delete(site);
        delete current[el];
      }

      if (nEl < nSites) {
        for (const site of siteLabels) {
          if (!usedSites.has(site)) continue;
          current[el] = site;
          permute(rest, usedSites, current);
          delete current[el];
        }
      }
    };

    if (nEl <= 5) {
      permute(elements, new Set(), {});
      if (Object.keys(bestAssignment).length > 0) return bestAssignment;
    }
  }

  const sorted = [...elements].sort((a, b) => {
    const dA = getElementData(a);
    const dB = getElementData(b);
    const radiusA = dA?.atomicRadius ?? 130;
    const radiusB = dB?.atomicRadius ?? 130;
    return radiusB - radiusA;
  });

  for (let i = 0; i < sorted.length && i < siteLabels.length; i++) {
    assignment[sorted[i]] = siteLabels[i];
  }
  for (let i = siteLabels.length; i < sorted.length; i++) {
    assignment[sorted[i]] = siteLabels[siteLabels.length - 1];
  }

  return assignment;
}

export function buildPrototypeGraph(formula: string, prototype: string, pressureGpa?: number): CrystalGraph {
  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts);
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];

  if (!protoInfo) {
    return buildCrystalGraph(formula, undefined, pressureGpa);
  }

  const siteAssignment = assignSiteLabels(formula, prototype);
  const { normalized, multiplicities } = normalizeFormulaCounts(rawCounts);
  const nodes: NodeFeature[] = [];

  for (const el of elements) {
    const count = normalized[el];
    const mult = multiplicities[el];
    const data = getElementData(el);
    const atomicNumber = data?.atomicNumber ?? 30;
    const en = data?.paulingElectronegativity ?? 1.5;
    const radius = data?.atomicRadius ?? 130;
    const valence = data?.valenceElectrons ?? 2;
    const mass = data?.atomicMass ?? 50;

    const protoSymFeatures = getSymmetryAwareFeatures(undefined);

    for (let i = 0; i < count; i++) {
      const baseEmbedding = buildEnhancedEmbedding(el, data, atomicNumber);
      const embedding = baseEmbedding.slice(0, NODE_DIM - protoSymFeatures.length);
      embedding.push(...protoSymFeatures);
      while (embedding.length < NODE_DIM) embedding.push(0);
      nodes.push({ element: el, atomicNumber, electronegativity: en, atomicRadius: radius, valenceElectrons: valence, mass, embedding: embedding.slice(0, NODE_DIM), multiplicity: mult });
    }
  }

  if (nodes.length === 0) {
    nodes.push({
      element: "X", atomicNumber: 1, electronegativity: 1.5,
      atomicRadius: 100, valenceElectrons: 1, mass: 10,
      embedding: initVector(NODE_DIM, 0.1),
      multiplicity: 1,
    });
  }

  const edges: EdgeFeature[] = [];
  const adjacencySets: Set<number>[] = nodes.map(() => new Set<number>());
  const adjacency: number[][] = nodes.map(() => []);
  const lp = protoInfo.latticeParams;

  let nodeOffset = 0;
  const elementRanges: Record<string, { start: number; end: number }> = {};
  for (const el of elements) {
    const count = normalized[el];
    elementRanges[el] = { start: nodeOffset, end: nodeOffset + count };
    nodeOffset += count;
  }

  for (const el of elements) {
    const site = siteAssignment[el];
    if (!site || !protoInfo.coordinations[site]) continue;

    const coord = protoInfo.coordinations[site];
    const range = elementRanges[el];

    for (let i = range.start; i < range.end; i++) {
      for (const neighborSite of coord.neighbors) {
        const neighborElements = Object.entries(siteAssignment)
          .filter(([, s]) => s === neighborSite)
          .map(([e]) => e);

        for (const nEl of neighborElements) {
          const nRange = elementRanges[nEl];
          if (!nRange) continue;

          for (let j = nRange.start; j < nRange.end; j++) {
            if (i === j) continue;
            if (adjacencySets[i].has(j)) continue;

            const ri = nodes[i].atomicRadius / 100;
            const rj = nodes[j].atomicRadius / 100;
            const pScale = pressureDistanceScale(pressureGpa ?? 0);
            const distance = (ri + rj) * 0.9 * pScale;

            const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
            const bondOrder = pressureAwareBondOrder(enDiff, nodes[i].atomicNumber, nodes[j].atomicNumber, pressureGpa ?? 0);
            const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
            const ionicCharacter = Math.min(1.0, enDiff / 2.5);

            const edgeFeats = buildEdgeFeatures(distance, bondOrder, enDiff, ionicCharacter, radiusSum);

            edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
            edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
            adjacencySets[i].add(j);
            adjacencySets[j].add(i);
            adjacency[i].push(j);
            adjacency[j].push(i);
          }
        }
      }
    }
  }

  if (edges.length === 0 && nodes.length > 1) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const ri = nodes[i].atomicRadius / 100;
        const rj = nodes[j].atomicRadius / 100;
        const distance = (ri + rj) * 1.1;
        const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
        const bondOrder = pressureAwareBondOrder(enDiff, nodes[i].atomicNumber, nodes[j].atomicNumber, pressureGpa ?? 0);
        const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
        const ionicCharacter = Math.min(1.0, enDiff / 2.5);
        const edgeFeats = buildEdgeFeatures(distance, bondOrder, enDiff, ionicCharacter, radiusSum);
        edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
        edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
        adjacency[i].push(j);
        adjacency[j].push(i);
      }
    }
  }

  const edgeIndex = buildEdgeIndex(nodes, edges);
  const threeBodyFeatures = compute3BodyFeatures({ nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, prototype });
  return { nodes, edges, threeBodyFeatures, adjacency, edgeIndex, formula, prototype, pressureGpa };
}

function computeStressDescriptor(atomicNumber: number, bulkModulus: number, mass: number): number {
  if (bulkModulus <= 0 || mass <= 0) return 0.3;
  return Math.min(1.0, Math.sqrt(bulkModulus / mass) / 10.0);
}

function computeForceDescriptor(electronegativity: number, atomicRadius: number): number {
  return Math.min(1.0, (electronegativity * 100) / Math.max(atomicRadius, 50));
}

function computeSpinOrbitCoupling(atomicNumber: number): number {
  if (atomicNumber < 10) return 0;
  if (atomicNumber < 20) return 0.02 * (atomicNumber - 10) / 10;
  if (atomicNumber < 30) return 0.05 + 0.05 * (atomicNumber - 20) / 10;
  if (atomicNumber < 40) return 0.10 + 0.10 * (atomicNumber - 30) / 10;
  if (atomicNumber < 48) return 0.20 + 0.15 * (atomicNumber - 40) / 8;
  if (atomicNumber < 57) return 0.35 + 0.05 * (atomicNumber - 48) / 9;
  if (atomicNumber < 72) return 0.40 + 0.20 * (atomicNumber - 57) / 15;
  if (atomicNumber < 80) return 0.60 + 0.20 * (atomicNumber - 72) / 8;
  return Math.min(1.0, 0.80 + 0.02 * (atomicNumber - 80));
}

function canonicalEdgeKey(a: number, b: number): number {
  return a < b ? a * 65536 + b : b * 65536 + a;
}

function compute3BodyFeatures(graph: CrystalGraph): ThreeBodyFeature[] {
  const features: ThreeBodyFeature[] = [];
  const nNodes = graph.nodes.length;
  const ei = graph.edgeIndex;

  for (let center = 0; center < nNodes; center++) {
    const neighbors = graph.adjacency[center];
    if (neighbors.length < 2) continue;

    for (let a = 0; a < neighbors.length; a++) {
      for (let b = a + 1; b < neighbors.length; b++) {
        const n1 = neighbors[a];
        const n2 = neighbors[b];
        const e1 = getEdgeFromIndex(ei, nNodes, center, n1);
        const e2 = getEdgeFromIndex(ei, nNodes, center, n2);
        const e12 = getEdgeFromIndex(ei, nNodes, n1, n2);
        const d1 = e1?.distance ?? 2.5;
        const d2 = e2?.distance ?? 2.5;
        const d12 = e12?.distance ?? Math.sqrt(d1 * d1 + d2 * d2);

        let cosAngle = (d1 * d1 + d2 * d2 - d12 * d12) / (2 * d1 * d2);
        cosAngle = Math.max(-1, Math.min(1, cosAngle));
        const angle = Math.acos(cosAngle);

        features.push({ center, neighbor1: n1, neighbor2: n2, angle, distance1: d1, distance2: d2 });
      }
    }
  }
  return features;
}

function threeBodyInteractionLayer(
  graph: CrystalGraph,
  W_3body: number[][],
  W_3body_update: number[][],
): number[][] {
  const nNodes = graph.nodes.length;
  const embeddings = graph.nodes.map(n => n.embedding);

  const threeBodyAgg: number[][] = embeddings.map(() => initVector(HIDDEN_DIM));

  const neighborCounts = new Uint16Array(nNodes);
  for (const tb of graph.threeBodyFeatures) {
    neighborCounts[tb.center]++;
  }

  for (const tb of graph.threeBodyFeatures) {
    const angleFeature = tb.angle / Math.PI;
    const distFeature = Math.min(1.0, (tb.distance1 + tb.distance2) / 12.0);
    const asymmetry = Math.abs(tb.distance1 - tb.distance2) / Math.max(tb.distance1, tb.distance2, 0.01);

    const n1Embed = embeddings[tb.neighbor1] ?? initVector(HIDDEN_DIM);
    const n2Embed = embeddings[tb.neighbor2] ?? initVector(HIDDEN_DIM);

    const asymScale = 1.0 + asymmetry * 0.3;
    const pairMsg = n1Embed.map((v, i) => (v + (n2Embed[i] ?? 0)) * 0.5 * angleFeature * asymScale);
    const transformed = matVecMul(W_3body, pairMsg);

    for (let k = 0; k < HIDDEN_DIM; k++) {
      threeBodyAgg[tb.center][k] += (transformed[k] ?? 0) * distFeature;
    }
  }

  const newEmbeddings: number[][] = [];
  for (let i = 0; i < nNodes; i++) {
    const nc = neighborCounts[i];
    if (nc > 0) {
      const normFactor = Math.sqrt(nc);
      for (let k = 0; k < HIDDEN_DIM; k++) {
        threeBodyAgg[i][k] /= normFactor;
      }
    }

    const combined = [...embeddings[i], ...threeBodyAgg[i]];
    const updated = fusedMatVecLeakyRelu(W_3body_update, combined);
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return newEmbeddings;
}

function buildEnhancedEmbedding(el: string, data: ReturnType<typeof getElementData>, atomicNumber: number): number[] {
  const en = data?.paulingElectronegativity ?? 1.5;
  const radius = data?.atomicRadius ?? 130;
  const valence = data?.valenceElectrons ?? 2;
  const mass = data?.atomicMass ?? 50;
  const covalentR = getCovalentRadius(atomicNumber, radius);
  const electronAff = data?.electronAffinity ?? 0;
  const mendeleev = getMendeleevNumber(atomicNumber);
  const sOcc = getSOrbitalOccupancy(atomicNumber);
  const pOcc = getPOrbitalOccupancy(atomicNumber);
  const dOcc = getDOrbitalOccupancy(atomicNumber);
  const fOcc = getFOrbitalOccupancy(atomicNumber);
  const bulkMod = data?.bulkModulus ?? 50;

  const period = getPeriod(atomicNumber);
  const group = getGroup(atomicNumber);
  const block = getBlockEncoding(atomicNumber);

  return [
    atomicNumber / 100,
    en / 4.0,
    radius / 250,
    valence / 8,
    mass / 250,
    (data?.debyeTemperature ?? 300) / 2000,
    bulkMod / 500,
    (data?.firstIonizationEnergy ?? 7) / 25,
    mendeleev / 103,
    Math.max(0, electronAff) / 4.0,
    covalentR / 250,
    sOcc,
    pOcc,
    dOcc,
    fOcc,
    computeStressDescriptor(atomicNumber, bulkMod, mass),
    computeForceDescriptor(en, radius),
    computeSpinOrbitCoupling(atomicNumber),
    computeMagneticMomentProxy(atomicNumber),
    computeValenceShellEncoding(atomicNumber, valence),
    period / 7.0,
    group / 18.0,
    block,
    Math.min(1.0, (data?.meltingPoint ?? 1000) / 4000),
    Math.min(1.0, (data?.density ?? 5) / 25),
    Math.min(1.0, (data?.thermalConductivity ?? 50) / 500),
    Math.min(1.0, Math.abs(data?.electronAffinity ?? 0) / 4.0),
    Math.min(1.0, (data?.atomicVolume ?? 15) / 80),
    en > 2.0 ? 1.0 : en > 1.5 ? 0.5 : 0.0,
    atomicNumber <= 20 ? 0.0 : atomicNumber <= 30 ? 0.5 : 1.0,
    Math.min(1.0, valence * en / 16.0),
    Math.min(1.0, covalentR * dOcc / 100),
  ];
}

function getSymmetryAwareFeatures(spaceGroupName?: string, fracPosition?: [number, number, number]): number[] {
  if (!spaceGroupName) return [0, 0, 0, 0, 0, 0];
  const embedding = computeSymmetryEmbedding(spaceGroupName, fracPosition);
  return computeSymmetryFeatureVector(embedding);
}

export function buildCrystalGraph(formula: string, structure?: any, pressureGpa?: number): CrystalGraph {
  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts);
  const { normalized, multiplicities } = normalizeFormulaCounts(rawCounts);

  const nodes: NodeFeature[] = [];

  for (const el of elements) {
    const count = normalized[el];
    const mult = multiplicities[el];
    const data = getElementData(el);
    const atomicNumber = data?.atomicNumber ?? 30;
    const en = data?.paulingElectronegativity ?? 1.5;
    const radius = data?.atomicRadius ?? 130;
    const valence = data?.valenceElectrons ?? 2;
    const mass = data?.atomicMass ?? 50;

    const spaceGroupName = structure?.spaceGroup ?? structure?.spacegroupSymbol;
    const symFeatures = getSymmetryAwareFeatures(spaceGroupName);

    for (let i = 0; i < count; i++) {
      const baseEmbedding = buildEnhancedEmbedding(el, data, atomicNumber);
      const embedding = baseEmbedding.slice(0, NODE_DIM - symFeatures.length);
      embedding.push(...symFeatures);
      while (embedding.length < NODE_DIM) embedding.push(0);
      nodes.push({ element: el, atomicNumber, electronegativity: en, atomicRadius: radius, valenceElectrons: valence, mass, embedding: embedding.slice(0, NODE_DIM), multiplicity: mult });
    }
  }

  if (nodes.length === 0) {
    nodes.push({
      element: "X", atomicNumber: 1, electronegativity: 1.5,
      atomicRadius: 100, valenceElectrons: 1, mass: 10,
      embedding: initVector(NODE_DIM, 0.1),
      multiplicity: 1,
    });
  }

  const edges: EdgeFeature[] = [];
  const adjacency: number[][] = nodes.map(() => []);

  const latticeParams = structure?.latticeParams;
  const hasPositions = structure?.atomicPositions && Array.isArray(structure.atomicPositions);
  const cutoff = 6.0;

  const useVoxelGrid = hasPositions && nodes.length > 32;

  if (useVoxelGrid) {
    const a = latticeParams?.a ?? 5;
    const b = latticeParams?.b ?? 5;
    const c = latticeParams?.c ?? 5;
    const voxelSize = cutoff;
    const nxBins = Math.max(1, Math.ceil(a / voxelSize));
    const nyBins = Math.max(1, Math.ceil(b / voxelSize));
    const nzBins = Math.max(1, Math.ceil(c / voxelSize));
    const voxelGrid = new Map<number, number[]>();

    for (let idx = 0; idx < nodes.length; idx++) {
      const pos = structure.atomicPositions[idx];
      if (!pos) continue;
      const vx = Math.min(nxBins - 1, Math.max(0, Math.floor(pos.x * nxBins)));
      const vy = Math.min(nyBins - 1, Math.max(0, Math.floor(pos.y * nyBins)));
      const vz = Math.min(nzBins - 1, Math.max(0, Math.floor(pos.z * nzBins)));
      const vKey = vx * nyBins * nzBins + vy * nzBins + vz;
      const bucket = voxelGrid.get(vKey);
      if (bucket) bucket.push(idx); else voxelGrid.set(vKey, [idx]);
    }

    for (let i = 0; i < nodes.length; i++) {
      const pi = structure.atomicPositions[i];
      if (!pi) continue;
      const vx = Math.min(nxBins - 1, Math.max(0, Math.floor(pi.x * nxBins)));
      const vy = Math.min(nyBins - 1, Math.max(0, Math.floor(pi.y * nyBins)));
      const vz = Math.min(nzBins - 1, Math.max(0, Math.floor(pi.z * nzBins)));

      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const nx = ((vx + dx) % nxBins + nxBins) % nxBins;
            const ny = ((vy + dy) % nyBins + nyBins) % nyBins;
            const nz = ((vz + dz) % nzBins + nzBins) % nzBins;
            const nKey = nx * nyBins * nzBins + ny * nzBins + nz;
            const bucket = voxelGrid.get(nKey);
            if (!bucket) continue;

            for (const j of bucket) {
              if (j <= i) continue;
              const pj = structure.atomicPositions[j];
              if (!pj) continue;
              const ddx = (pi.x - pj.x) * a;
              const ddy = (pi.y - pj.y) * b;
              const ddz = (pi.z - pj.z) * c;
              const distance = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
              if (distance >= cutoff) continue;

              const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
              const bondOrder = pressureAwareBondOrder(enDiff, nodes[i].atomicNumber, nodes[j].atomicNumber, pressureGpa ?? 0);
              const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
              const ionicCharacter = Math.min(1.0, enDiff / 2.5);
              const edgeFeats = buildEdgeFeatures(distance, bondOrder, enDiff, ionicCharacter, radiusSum);

              edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
              edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
              adjacency[i].push(j);
              adjacency[j].push(i);
            }
          }
        }
      }
    }
  } else {
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
          const pScale = pressureDistanceScale(pressureGpa ?? 0);
          distance = (ri + rj) * 1.1 * pScale;
        }

        if (distance < cutoff || nodes.length <= 8) {
          const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
          const bondOrder = pressureAwareBondOrder(enDiff, nodes[i].atomicNumber, nodes[j].atomicNumber, pressureGpa ?? 0);

          const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
          const ionicCharacter = Math.min(1.0, enDiff / 2.5);

          const edgeFeats = buildEdgeFeatures(distance, bondOrder, enDiff, ionicCharacter, radiusSum);

          edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
          edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });

          adjacency[i].push(j);
          adjacency[j].push(i);
        }
      }
    }
  }

  if (edges.length === 0 && nodes.length > 1) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const ri = nodes[i].atomicRadius / 100;
        const rj = nodes[j].atomicRadius / 100;
        const pScale = pressureDistanceScale(pressureGpa ?? 0);
        const distance = (ri + rj) * 1.1 * pScale;
        const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
        const bondOrder = pressureAwareBondOrder(enDiff, nodes[i].atomicNumber, nodes[j].atomicNumber, pressureGpa ?? 0);
        const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
        const ionicCharacter = Math.min(1.0, enDiff / 2.5);
        const edgeFeats = buildEdgeFeatures(distance, bondOrder, enDiff, ionicCharacter, radiusSum);
        edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
        edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
        adjacency[i].push(j);
        adjacency[j].push(i);
      }
    }
  }

  const edgeIndex = buildEdgeIndex(nodes, edges);
  const partialGraph: CrystalGraph = { nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, pressureGpa };
  partialGraph.threeBodyFeatures = compute3BodyFeatures(partialGraph);
  return partialGraph;
}

export function cgcnnConvolutionLayer(
  graph: CrystalGraph,
  W_gate: number[][],
  W_value: number[][],
  b_gate: number[],
  b_value: number[],
): number[][] {
  const nNodes = graph.nodes.length;
  const embeddings = graph.nodes.map(n => n.embedding);

  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    if (!neighbors || neighbors.length === 0) {
      newEmbeddings.push([...embeddings[i]]);
      continue;
    }

    const aggUpdate = initVector(HIDDEN_DIM);
    let totalWeight = 0;

    for (const j of neighbors) {
      const edgeFeat = getEdgeFromIndex(graph.edgeIndex, nNodes, i, j);
      const distance = edgeFeat?.distance ?? 2.5;
      const cutoffWeight = cosineCutoff(distance);
      if (cutoffWeight <= 0) continue;

      const edgeVec = edgeFeat?.features ?? initVector(EDGE_DIM);

      const concat: number[] = new Array(CGCNN_CONCAT_DIM);
      for (let k = 0; k < HIDDEN_DIM; k++) {
        concat[k] = embeddings[i][k] ?? 0;
      }
      for (let k = 0; k < HIDDEN_DIM; k++) {
        concat[HIDDEN_DIM + k] = embeddings[j][k] ?? 0;
      }
      for (let k = 0; k < EDGE_DIM; k++) {
        concat[HIDDEN_DIM * 2 + k] = edgeVec[k] ?? 0;
      }

      const gateRaw = vecAdd(matVecMul(W_gate, concat), b_gate);
      const valueRaw = vecAdd(matVecMul(W_value, concat), b_value);

      for (let k = 0; k < HIDDEN_DIM; k++) {
        const gateVal = sigmoid(gateRaw[k] ?? 0);
        const valueVal = softplus(valueRaw[k] ?? 0);
        aggUpdate[k] += gateVal * valueVal * cutoffWeight;
      }

      totalWeight += cutoffWeight;
    }

    if (totalWeight > 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] /= totalWeight;
      }
    }

    const updated: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      updated[k] = (embeddings[i][k] ?? 0) + aggUpdate[k];
    }
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return newEmbeddings;
}

export function attentionMessagePassingLayer(
  graph: CrystalGraph,
  W_message: number[][],
  W_update: number[][],
  W_query: number[][],
  W_key: number[][],
  useLeakyMsg: boolean = false,
): number[][] {
  const nNodes = graph.nodes.length;
  const embeddings = graph.nodes.map(n => n.embedding);

  const newEmbeddings: number[][] = [];
  const fusedUpdate = useLeakyMsg
    ? (mat: number[][], vec: number[]) => fusedMatVecLeakyRelu(mat, vec)
    : (mat: number[][], vec: number[]) => {
        const rows = mat.length;
        const result = new Array(rows);
        for (let i = 0; i < rows; i++) {
          const row = mat[i];
          let sum = 0;
          for (let j = 0; j < vec.length; j++) sum += row[j] * vec[j];
          result[i] = sum > 0 ? sum : 0;
        }
        return result;
      };

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    if (neighbors.length === 0) {
      newEmbeddings.push([...embeddings[i]]);
      continue;
    }

    const query = layerNorm(matVecMul(W_query, embeddings[i]));

    const attentionScores: number[] = [];
    const messages: number[][] = [];

    for (const j of neighbors) {
      const key = layerNorm(matVecMul(W_key, embeddings[j]));
      let score = dotProduct(query, key);

      const edge = getEdgeFromIndex(graph.edgeIndex, nNodes, i, j);
      if (edge) {
        const edgeFeats = edge.features;
        for (let ef = 0; ef < edgeFeats.length && ef < HIDDEN_DIM; ef++) {
          score += (edgeFeats[ef] ?? 0) * (query[ef] ?? 0) * 0.1;
        }
      }

      attentionScores.push(score);
      messages.push(matVecMul(W_message, embeddings[j]));
    }

    const attentionWeights = softmax(attentionScores);

    const aggMessage = initVector(HIDDEN_DIM);
    for (let n = 0; n < neighbors.length; n++) {
      const w = attentionWeights[n];
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += (messages[n][k] ?? 0) * w;
      }
    }

    const combined = [...embeddings[i], ...aggMessage];
    const updated = fusedUpdate(W_update, combined);
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return newEmbeddings;
}

export function messagePassingLayer(
  graph: CrystalGraph,
  W_message: number[][],
  W_update: number[][]
): number[][] {
  const nNodes = graph.nodes.length;
  const embeddings = graph.nodes.map(n => n.embedding);

  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    if (!neighbors || neighbors.length === 0) {
      newEmbeddings.push([...embeddings[i]]);
      continue;
    }

    const aggMessage = initVector(HIDDEN_DIM);
    const nCount = Math.max(1, neighbors.length);
    for (const j of neighbors) {
      const msg = matVecMul(W_message, embeddings[j]);
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += msg[k] / nCount;
      }
    }

    const combined = [...embeddings[i], ...aggMessage];
    const updated = fusedMatVecLeakyRelu(W_update, combined);
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return newEmbeddings;
}

function attentionPooling(graph: CrystalGraph, W_pool: number[][]): number[] {
  const nNodes = graph.nodes.length;
  if (nNodes === 0) return initVector(HIDDEN_DIM);

  const scores: number[] = [];
  for (const node of graph.nodes) {
    const attnVec = matVecMul(W_pool, node.embedding);
    const rawScore = attnVec.reduce((s, v) => s + v, 0);
    scores.push(rawScore + Math.log(node.multiplicity ?? 1));
  }

  const attnWeights = softmax(scores);
  const pooled = initVector(HIDDEN_DIM);
  for (let n = 0; n < nNodes; n++) {
    const e = graph.nodes[n].embedding;
    for (let k = 0; k < HIDDEN_DIM; k++) {
      pooled[k] += (e[k] ?? 0) * attnWeights[n];
    }
  }
  return pooled;
}

export function GNNPredict(graph: CrystalGraph, weights: GNNWeights, dropoutRng?: () => number): GNNPrediction {
  for (let i = 0; i < graph.nodes.length; i++) {
    const raw = graph.nodes[i].embedding;
    const input = raw.length >= NODE_DIM ? raw.slice(0, NODE_DIM) : [...raw, ...new Array(NODE_DIM - raw.length).fill(0)];
    const projected = fusedMatVecAddLeakyRelu(weights.W_input_proj, input, weights.b_input_proj);
    graph.nodes[i].embedding = projected;
  }

  const saveResidual = (nodes: CrystalGraph["nodes"]) =>
    nodes.map(n => [...n.embedding]);

  const gates = weights.residual_gates;

  const residual0 = saveResidual(graph.nodes);

  attentionMessagePassingLayer(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true);
  if (dropoutRng) {
    for (const node of graph.nodes) {
      node.embedding = applyDropout(node.embedding, MC_DROPOUT_RATE, dropoutRng);
    }
  }

  cgcnnConvolutionLayer(graph, weights.W_conv_gate, weights.W_conv_value, weights.b_conv_gate, weights.b_conv_value);
  if (dropoutRng) {
    for (const node of graph.nodes) {
      node.embedding = applyDropout(node.embedding, MC_DROPOUT_RATE, dropoutRng);
    }
  }

  if (graph.threeBodyFeatures.length > 0) {
    threeBodyInteractionLayer(graph, weights.W_3body, weights.W_3body_update);
  }

  const g0 = sigmoid(gates[0] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual0[i][k] ?? 0) * g0;
    }
  }

  const residual1 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2);
  if (dropoutRng) {
    for (const node of graph.nodes) {
      node.embedding = applyDropout(node.embedding, MC_DROPOUT_RATE, dropoutRng);
    }
  }

  const g1 = sigmoid(gates[1] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual1[i][k] ?? 0) * g1;
    }
  }

  const residual2 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3);
  if (dropoutRng) {
    for (const node of graph.nodes) {
      node.embedding = applyDropout(node.embedding, MC_DROPOUT_RATE, dropoutRng);
    }
  }

  const g2 = sigmoid(gates[2] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual2[i][k] ?? 0) * g2;
    }
  }

  const residual3 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4);
  if (dropoutRng) {
    for (const node of graph.nodes) {
      node.embedding = applyDropout(node.embedding, MC_DROPOUT_RATE, dropoutRng);
    }
  }

  const g3 = sigmoid(gates[3] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual3[i][k] ?? 0) * g3;
    }
  }

  const nNodes = graph.nodes.length;
  const meanPool = initVector(HIDDEN_DIM);
  const maxPool = new Array(HIDDEN_DIM).fill(-Infinity);

  let totalMultiplicity = 0;
  for (const node of graph.nodes) totalMultiplicity += (node.multiplicity ?? 1);

  for (const node of graph.nodes) {
    const w = (node.multiplicity ?? 1) / totalMultiplicity;
    for (let k = 0; k < HIDDEN_DIM; k++) {
      meanPool[k] += (node.embedding[k] ?? 0) * w;
      maxPool[k] = Math.max(maxPool[k], node.embedding[k] ?? 0);
    }
  }

  const attnPool = attentionPooling(graph, weights.W_attn_pool);

  const pooled = new Array(HIDDEN_DIM * 2);
  for (let k = 0; k < HIDDEN_DIM; k++) {
    pooled[k] = (meanPool[k] + attnPool[k]) * 0.5;
    pooled[HIDDEN_DIM + k] = (maxPool[k] === -Infinity ? 0 : maxPool[k]);
  }

  const pressureNorm = (graph.pressureGpa ?? 0) / 300;
  for (let k = 0; k < HIDDEN_DIM; k++) {
    pooled[k] += pressureNorm * (weights.W_pressure[k] ?? 0);
  }

  const z1 = vecAdd(matVecMul(weights.W_mlp1, pooled), weights.b_mlp1);
  const h1 = z1.map(v => v >= 0 ? v : 0.01 * v);
  if (dropoutRng) {
    const dropped = applyDropout(h1, MC_DROPOUT_RATE, dropoutRng);
    for (let i = 0; i < h1.length; i++) h1[i] = dropped[i];
  }
  const latentEmbedding = [...h1];
  const out = vecAdd(matVecMul(weights.W_mlp2, h1), weights.b_mlp2);

  const logVarOut = vecAdd(matVecMul(weights.W_mlp2_var, h1), weights.b_mlp2_var);
  const feVarNorm = softplus(logVarOut[0] ?? 0);
  const tcVarNorm = softplus(logVarOut[2] ?? 0);
  const lambdaVarNorm = softplus(logVarOut[4] ?? 0);
  const bgVarNorm = softplus(logVarOut[5] ?? 0);

  const sf = (v: number, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const formationEnergy = sf(out[0] ?? 0);
  const phononStabilityRaw = sigmoid(sf(out[1] ?? 0));
  const predictedTcRaw = Math.max(0, sf(out[2] ?? 0) * 300);
  const confidenceRaw = sigmoid(sf(out[3] ?? 0));
  const lambdaRaw = Math.max(0, sf(out[4] ?? 0));
  const bandgapRaw = sigmoid(sf(out[5] ?? 0)) * 5.0;
  const dosProxyRaw = softplus(sf(out[6] ?? 0));
  const stabilityProbRaw = sigmoid(sf(out[7] ?? 0));
  const safeLatent = latentEmbedding.map(v => Number.isFinite(v) ? v : 0);

  return {
    formationEnergy: Math.round(formationEnergy * 1000) / 1000,
    phononStability: phononStabilityRaw > 0.5,
    predictedTc: Math.round(Math.max(0, predictedTcRaw) * 10) / 10,
    confidence: Math.round(Math.max(0.05, Math.min(0.95, confidenceRaw)) * 100) / 100,
    lambda: Math.round(Math.max(0, lambdaRaw) * 1000) / 1000,
    bandgap: Math.round(bandgapRaw * 1000) / 1000,
    dosProxy: Math.round(dosProxyRaw * 1000) / 1000,
    stabilityProbability: Math.round(stabilityProbRaw * 1000) / 1000,
    latentEmbedding: safeLatent,
    predictedTcVar: Math.round(Math.max(0.01, sf(tcVarNorm * 300 * 300, 1)) * 1000) / 1000,
    lambdaVar: Math.round(Math.max(0.001, sf(lambdaVarNorm, 0.01)) * 1000) / 1000,
    formationEnergyVar: Math.round(Math.max(0.001, sf(feVarNorm, 0.01)) * 1000) / 1000,
    bandgapVar: Math.round(Math.max(0.001, sf(bgVarNorm, 0.01)) * 1000) / 1000,
  };
}

function GNNPredictForTraining(graph: CrystalGraph, weights: GNNWeights): { pred: GNNPrediction; cache: GNNForwardCache } {
  for (let i = 0; i < graph.nodes.length; i++) {
    const raw = graph.nodes[i].embedding;
    const input = raw.length >= NODE_DIM ? raw.slice(0, NODE_DIM) : [...raw, ...new Array(NODE_DIM - raw.length).fill(0)];
    const projected = fusedMatVecAddLeakyRelu(weights.W_input_proj, input, weights.b_input_proj);
    graph.nodes[i].embedding = projected;
  }

  const saveResidual = (nodes: CrystalGraph["nodes"]) =>
    nodes.map(n => [...n.embedding]);

  const gates = weights.residual_gates;

  const residual0 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true);
  cgcnnConvolutionLayer(graph, weights.W_conv_gate, weights.W_conv_value, weights.b_conv_gate, weights.b_conv_value);
  if (graph.threeBodyFeatures.length > 0) {
    threeBodyInteractionLayer(graph, weights.W_3body, weights.W_3body_update);
  }
  const g0 = sigmoid(gates[0] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual0[i][k] ?? 0) * g0;
    }
  }

  const residual1 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2);
  const g1 = sigmoid(gates[1] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual1[i][k] ?? 0) * g1;
    }
  }

  const residual2 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3);
  const g2 = sigmoid(gates[2] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual2[i][k] ?? 0) * g2;
    }
  }

  const residual3 = saveResidual(graph.nodes);
  attentionMessagePassingLayer(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4);
  const g3 = sigmoid(gates[3] ?? 0);
  for (let i = 0; i < graph.nodes.length; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual3[i][k] ?? 0) * g3;
    }
  }

  const nNodes = graph.nodes.length;
  const meanPool = initVector(HIDDEN_DIM);
  const maxPool = new Array(HIDDEN_DIM).fill(-Infinity);
  let totalMultiplicity = 0;
  for (const node of graph.nodes) totalMultiplicity += (node.multiplicity ?? 1);
  for (const node of graph.nodes) {
    const w = (node.multiplicity ?? 1) / totalMultiplicity;
    for (let k = 0; k < HIDDEN_DIM; k++) {
      meanPool[k] += (node.embedding[k] ?? 0) * w;
      maxPool[k] = Math.max(maxPool[k], node.embedding[k] ?? 0);
    }
  }
  const attnPool = attentionPooling(graph, weights.W_attn_pool);

  const pooled = new Array(HIDDEN_DIM * 2);
  for (let k = 0; k < HIDDEN_DIM; k++) {
    pooled[k] = (meanPool[k] + attnPool[k]) * 0.5;
    pooled[HIDDEN_DIM + k] = (maxPool[k] === -Infinity ? 0 : maxPool[k]);
  }
  const pressureNorm = (graph.pressureGpa ?? 0) / 300;
  for (let k = 0; k < HIDDEN_DIM; k++) {
    pooled[k] += pressureNorm * (weights.W_pressure[k] ?? 0);
  }

  const z1 = vecAdd(matVecMul(weights.W_mlp1, pooled), weights.b_mlp1);
  const h1 = z1.map(v => v >= 0 ? v : 0.01 * v);
  const latentEmbedding = [...h1];
  const out = vecAdd(matVecMul(weights.W_mlp2, h1), weights.b_mlp2);
  const logVarOut = vecAdd(matVecMul(weights.W_mlp2_var, h1), weights.b_mlp2_var);

  const feVarNorm = softplus(logVarOut[0] ?? 0);
  const tcVarNorm = softplus(logVarOut[2] ?? 0);
  const lambdaVarNorm = softplus(logVarOut[4] ?? 0);
  const bgVarNorm = softplus(logVarOut[5] ?? 0);

  const sf = (v: number, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const formationEnergy = sf(out[0] ?? 0);
  const phononStabilityRaw = sigmoid(sf(out[1] ?? 0));
  const predictedTcRaw = Math.max(0, sf(out[2] ?? 0) * 300);
  const confidenceRaw = sigmoid(sf(out[3] ?? 0));
  const lambdaRaw = Math.max(0, sf(out[4] ?? 0));
  const bandgapRaw = sigmoid(sf(out[5] ?? 0)) * 5.0;
  const dosProxyRaw = softplus(sf(out[6] ?? 0));
  const stabilityProbRaw = sigmoid(sf(out[7] ?? 0));
  const safeLatent = latentEmbedding.map(v => Number.isFinite(v) ? v : 0);

  const nodeEmbeddings = graph.nodes.map(n => [...n.embedding]);
  const nodeMultiplicities = graph.nodes.map(n => n.multiplicity ?? 1);

  const pred: GNNPrediction = {
    formationEnergy: Math.round(formationEnergy * 1000) / 1000,
    phononStability: phononStabilityRaw > 0.5,
    predictedTc: Math.round(Math.max(0, predictedTcRaw) * 10) / 10,
    confidence: Math.round(Math.max(0.05, Math.min(0.95, confidenceRaw)) * 100) / 100,
    lambda: Math.round(Math.max(0, lambdaRaw) * 1000) / 1000,
    bandgap: Math.round(bandgapRaw * 1000) / 1000,
    dosProxy: Math.round(dosProxyRaw * 1000) / 1000,
    stabilityProbability: Math.round(stabilityProbRaw * 1000) / 1000,
    latentEmbedding: safeLatent,
    predictedTcVar: Math.round(Math.max(0.01, sf(tcVarNorm * 300 * 300, 1)) * 1000) / 1000,
    lambdaVar: Math.round(Math.max(0.001, sf(lambdaVarNorm, 0.01)) * 1000) / 1000,
    formationEnergyVar: Math.round(Math.max(0.001, sf(feVarNorm, 0.01)) * 1000) / 1000,
    bandgapVar: Math.round(Math.max(0.001, sf(bgVarNorm, 0.01)) * 1000) / 1000,
  };

  const cache: GNNForwardCache = {
    pooled: [...pooled],
    z1: [...z1],
    h1: [...h1],
    outRaw: [...out],
    logVarOutRaw: [...logVarOut],
    nodeEmbeddings,
    nodeMultiplicities,
    totalMultiplicity,
  };

  return { pred, cache };
}

function initWeights(rng: () => number): GNNWeights {
  return {
    W_message: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng),
    W_update: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    W_message2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng),
    W_update2: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    W_message3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng),
    W_update3: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    W_message4: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng),
    W_update4: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    W_attn_query: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_key: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_query2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_key2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_query3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_key3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_query4: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_attn_key4: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(1.0 / HIDDEN_DIM)),
    W_conv_gate: initMatrix(HIDDEN_DIM, CGCNN_CONCAT_DIM, rng),
    W_conv_value: initMatrix(HIDDEN_DIM, CGCNN_CONCAT_DIM, rng),
    b_conv_gate: initVector(HIDDEN_DIM),
    b_conv_value: initVector(HIDDEN_DIM),
    W_input_proj: initMatrix(HIDDEN_DIM, NODE_DIM, rng, Math.sqrt(2.0 / NODE_DIM)),
    b_input_proj: initVector(HIDDEN_DIM),
    W_3body: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(2.0 / HIDDEN_DIM) * 1.5),
    W_3body_update: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    W_attn_pool: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng),
    residual_gates: [0.5, 0.5, 0.5, 0.5],
    W_pressure: Array.from({ length: HIDDEN_DIM }, () => (rng() - 0.5) * 2 * Math.sqrt(2.0 / HIDDEN_DIM)),
    W_mlp1: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    b_mlp1: initVector(HIDDEN_DIM),
    W_mlp2: initMatrix(OUTPUT_DIM, HIDDEN_DIM, rng),
    b_mlp2: initVector(OUTPUT_DIM),
    W_mlp2_var: initMatrix(OUTPUT_DIM, HIDDEN_DIM, rng, 0.05),
    b_mlp2_var: initVector(OUTPUT_DIM, -2.0),
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
    W_message4: w.W_message4.map(r => [...r]),
    W_update4: w.W_update4.map(r => [...r]),
    W_attn_query: w.W_attn_query.map(r => [...r]),
    W_attn_key: w.W_attn_key.map(r => [...r]),
    W_attn_query2: w.W_attn_query2.map(r => [...r]),
    W_attn_key2: w.W_attn_key2.map(r => [...r]),
    W_attn_query3: w.W_attn_query3.map(r => [...r]),
    W_attn_key3: w.W_attn_key3.map(r => [...r]),
    W_attn_query4: w.W_attn_query4.map(r => [...r]),
    W_attn_key4: w.W_attn_key4.map(r => [...r]),
    W_conv_gate: w.W_conv_gate.map(r => [...r]),
    W_conv_value: w.W_conv_value.map(r => [...r]),
    b_conv_gate: [...w.b_conv_gate],
    b_conv_value: [...w.b_conv_value],
    W_input_proj: w.W_input_proj.map(r => [...r]),
    b_input_proj: [...w.b_input_proj],
    residual_gates: [...w.residual_gates],
    W_3body: w.W_3body.map(r => [...r]),
    W_3body_update: w.W_3body_update.map(r => [...r]),
    W_attn_pool: w.W_attn_pool.map(r => [...r]),
    W_pressure: [...w.W_pressure],
    W_mlp1: w.W_mlp1.map(r => [...r]),
    b_mlp1: [...w.b_mlp1],
    W_mlp2: w.W_mlp2.map(r => [...r]),
    b_mlp2: [...w.b_mlp2],
    W_mlp2_var: w.W_mlp2_var.map(r => [...r]),
    b_mlp2_var: [...w.b_mlp2_var],
    trainedAt: w.trainedAt,
    nSamples: w.nSamples,
  };
}

interface TrainingSample {
  formula: string;
  tc: number;
  formationEnergy?: number;
  structure?: any;
  prototype?: string;
  pressureGpa?: number;
  lambda?: number;
}

function structureHash(structure: any): string {
  if (!structure) return '';
  const json = typeof structure === 'string' ? structure : JSON.stringify(structure);
  return createHash('md5').update(json).digest('hex').slice(0, 12);
}

function graphCacheKey(formula: string, prototype?: string, structure?: any): string {
  if (prototype) return `${formula}::p:${prototype}`;
  return `${formula}::s:${structureHash(structure)}`;
}

export function trainGNNSurrogate(trainingData: TrainingSample[], preInitWeights?: GNNWeights): GNNWeights {
  const rng = seededRandom(42);
  const weights = preInitWeights ?? initWeights(rng);

  if (trainingData.length < 5) {
    weights.trainedAt = Date.now();
    weights.nSamples = trainingData.length;
    return weights;
  }

  const lr = 0.001;
  const epochs = 8;
  const batchSize = Math.min(32, trainingData.length);

  const graphCache = new Map<string, CrystalGraph>();
  const origEmbeddings = new Map<string, number[][]>();
  const lambdaTargets = new Map<number, number>();
  for (let si = 0; si < trainingData.length; si++) {
    const sample = trainingData[si];
    const key = graphCacheKey(sample.formula, sample.prototype, sample.structure);
    if (!graphCache.has(key)) {
      const g = sample.prototype
        ? buildPrototypeGraph(sample.formula, sample.prototype, sample.pressureGpa)
        : buildCrystalGraph(sample.formula, sample.structure, sample.pressureGpa);
      graphCache.set(key, g);
      origEmbeddings.set(key, g.nodes.map(n => [...n.embedding]));
    }
    if (sample.lambda != null && sample.lambda > 0) {
      lambdaTargets.set(si, Math.min(4.0, sample.lambda));
    } else {
      try {
        const lp = predictLambda(sample.formula, sample.pressureGpa ?? 0);
        if (lp && lp.lambda > 0 && lp.confidence >= 0.5) {
          lambdaTargets.set(si, Math.min(4.0, lp.lambda));
        }
      } catch {}
    }
  }

  const indices = Array.from({ length: trainingData.length }, (_, i) => i);

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    let totalSamples = 0;

    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const numBatches = Math.ceil(trainingData.length / batchSize);

    for (let batch = 0; batch < numBatches; batch++) {
      const batchStart = batch * batchSize;
      const batchEnd = Math.min(batchStart + batchSize, trainingData.length);

      const pooledLen = HIDDEN_DIM * 2;
      const batchSize_actual = batchEnd - batchStart;
      const gradW2 = weights.W_mlp2.map(r => new Array(r.length).fill(0));
      const gradB2 = new Array(weights.b_mlp2.length).fill(0);
      const gradW2v = weights.W_mlp2_var.map(r => new Array(r.length).fill(0));
      const gradB2v = new Array(weights.b_mlp2_var.length).fill(0);
      const gradW1 = weights.W_mlp1.map(r => new Array(r.length).fill(0));
      const gradB1 = new Array(weights.b_mlp1.length).fill(0);
      const gradPressure = new Array(weights.W_pressure.length).fill(0);
      const avgDLdPooled = new Array(pooledLen).fill(0);

      for (let b = batchStart; b < batchEnd; b++) {
        const idx = indices[b];
        const sample = trainingData[idx];

        const cacheKey = graphCacheKey(sample.formula, sample.prototype, sample.structure);
        const graph = graphCache.get(cacheKey)!;
        const orig = origEmbeddings.get(cacheKey)!;
        for (let ni = 0; ni < graph.nodes.length; ni++) {
          graph.nodes[ni].embedding = [...orig[ni]];
        }
        const { pred, cache } = GNNPredictForTraining(graph, weights);

        const tcTarget = sample.tc / 300;
        const feTarget = sample.formationEnergy ?? 0;
        const tcError = pred.predictedTc / 300 - tcTarget;
        const feError = pred.formationEnergy - feTarget;

        const phononTarget = (sample.tc > 0) ? 1.0 : 0.0;
        const phononPred = pred.phononStability ? 1.0 : 0.0;
        const phononError = phononPred - phononTarget;

        const confTarget = Math.min(1.0, sample.tc > 0 ? 0.8 : 0.3);
        const confError = pred.confidence - confTarget;

        const lambdaTarget = lambdaTargets.has(idx) ? lambdaTargets.get(idx)!
          : (sample.tc > 0 ? Math.min(4.0, sample.tc / 50) : 0.1);
        const lambdaError = pred.lambda - lambdaTarget;

        const bgTarget = (sample as any).bandgap ?? (sample.tc > 0 ? 0 : 1.5);
        const bgError = pred.bandgap - bgTarget;
        const dosTarget = sample.tc > 0 ? Math.min(5.0, sample.tc / 30) : 0.3;
        const dosError = pred.dosProxy - dosTarget;
        const stabTarget = sample.tc > 0 ? 0.8 : 0.3;
        const stabError = pred.stabilityProbability - stabTarget;

        const tcVarNorm = Math.max(1e-8, softplus(cache.logVarOutRaw[2] ?? 0));
        const lambdaVarNorm = Math.max(1e-8, softplus(cache.logVarOutRaw[4] ?? 0));
        const heteroLossTc = (tcError * tcError) / tcVarNorm + Math.log(tcVarNorm);
        const heteroLossLambda = (lambdaError * lambdaError) / lambdaVarNorm + Math.log(lambdaVarNorm);

        const loss = tcError * tcError + 0.1 * feError * feError + 0.1 * heteroLossTc + 0.05 * heteroLossLambda;
        totalLoss += loss;
        totalSamples++;

        const clipGrad = (g: number) => { const v = Number.isFinite(g) ? g : 0; return Math.max(-1, Math.min(1, v)); };

        const dLdOut = new Array(OUTPUT_DIM).fill(0);
        dLdOut[0] = clipGrad(2 * feError * 0.1);
        dLdOut[1] = clipGrad(2 * phononError * 0.05);
        dLdOut[2] = clipGrad(2 * tcError + 0.1 * 2 * tcError / tcVarNorm);
        dLdOut[3] = clipGrad(2 * confError * 0.05);
        dLdOut[4] = clipGrad(2 * lambdaError * 0.1 + 0.05 * 2 * lambdaError / lambdaVarNorm);
        dLdOut[5] = clipGrad(2 * bgError * 0.05);
        dLdOut[6] = clipGrad(2 * dosError * 0.05);
        dLdOut[7] = clipGrad(2 * stabError * 0.05);

        const tcVarLogRaw = cache.logVarOutRaw[2] ?? 0;
        const lambdaVarLogRaw = cache.logVarOutRaw[4] ?? 0;
        const spGradTc = sigmoid(tcVarLogRaw);
        const spGradLambda = sigmoid(lambdaVarLogRaw);
        const dLdTcVarNorm = -(tcError * tcError) / (tcVarNorm * tcVarNorm) + 1.0 / tcVarNorm;
        const dLdLambdaVarNorm = -(lambdaError * lambdaError) / (lambdaVarNorm * lambdaVarNorm) + 1.0 / lambdaVarNorm;

        const dLdLogVarOut = new Array(OUTPUT_DIM).fill(0);
        dLdLogVarOut[2] = clipGrad(0.1 * dLdTcVarNorm * spGradTc);
        dLdLogVarOut[4] = clipGrad(0.05 * dLdLambdaVarNorm * spGradLambda);

        for (let i = 0; i < weights.W_mlp2.length; i++) {
          for (let j = 0; j < weights.W_mlp2[i].length; j++) {
            gradW2[i][j] += dLdOut[i] * cache.h1[j];
          }
          gradB2[i] += dLdOut[i];
        }

        for (let i = 0; i < weights.W_mlp2_var.length; i++) {
          if (dLdLogVarOut[i] !== 0) {
            for (let j = 0; j < weights.W_mlp2_var[i].length; j++) {
              gradW2v[i][j] += dLdLogVarOut[i] * cache.h1[j];
            }
            gradB2v[i] += dLdLogVarOut[i];
          }
        }

        const dLdH1 = new Array(HIDDEN_DIM).fill(0);
        for (let j = 0; j < HIDDEN_DIM; j++) {
          for (let i = 0; i < OUTPUT_DIM; i++) {
            dLdH1[j] += dLdOut[i] * (weights.W_mlp2[i]?.[j] ?? 0);
            dLdH1[j] += dLdLogVarOut[i] * (weights.W_mlp2_var[i]?.[j] ?? 0);
          }
        }

        const dLdZ1 = new Array(HIDDEN_DIM);
        for (let j = 0; j < HIDDEN_DIM; j++) {
          dLdZ1[j] = dLdH1[j] * (cache.z1[j] >= 0 ? 1.0 : 0.01);
        }

        for (let i = 0; i < HIDDEN_DIM; i++) {
          for (let j = 0; j < pooledLen; j++) {
            gradW1[i][j] += clipGrad(dLdZ1[i] * cache.pooled[j]);
          }
          gradB1[i] += clipGrad(dLdZ1[i]);
        }

        const dLdPooled = new Array(pooledLen).fill(0);
        for (let j = 0; j < pooledLen; j++) {
          for (let i = 0; i < HIDDEN_DIM; i++) {
            dLdPooled[j] += dLdZ1[i] * (weights.W_mlp1[i][j] ?? 0);
          }
          dLdPooled[j] = clipGrad(dLdPooled[j]);
        }
        for (let k = 0; k < pooledLen; k++) avgDLdPooled[k] += dLdPooled[k];

        for (let k = 0; k < HIDDEN_DIM; k++) {
          gradPressure[k] += clipGrad(dLdPooled[k] * ((graph.pressureGpa ?? 0) / 300));
        }
      }

      const invN = 1.0 / batchSize_actual;
      for (let i = 0; i < weights.W_mlp2.length; i++) {
        for (let j = 0; j < weights.W_mlp2[i].length; j++) {
          weights.W_mlp2[i][j] -= lr * gradW2[i][j] * invN;
        }
        weights.b_mlp2[i] -= lr * gradB2[i] * invN;
      }
      for (let i = 0; i < weights.W_mlp2_var.length; i++) {
        for (let j = 0; j < weights.W_mlp2_var[i].length; j++) {
          weights.W_mlp2_var[i][j] -= lr * gradW2v[i][j] * invN;
        }
        weights.b_mlp2_var[i] -= lr * gradB2v[i] * invN;
      }
      for (let i = 0; i < HIDDEN_DIM; i++) {
        for (let j = 0; j < pooledLen; j++) {
          weights.W_mlp1[i][j] -= lr * gradW1[i][j] * invN;
        }
        weights.b_mlp1[i] -= lr * gradB1[i] * invN;
      }
      for (let k = 0; k < HIDDEN_DIM; k++) {
        weights.W_pressure[k] -= lr * gradPressure[k] * invN;
      }

      for (let k = 0; k < pooledLen; k++) avgDLdPooled[k] *= invN;
      const pooledGradNorm = Math.sqrt(avgDLdPooled.reduce((s, v) => s + v * v, 0));
      const graphLR = lr * Math.min(1.0, pooledGradNorm) * 0.1;

      for (const wMat of [
        weights.W_message, weights.W_update,
        weights.W_message2, weights.W_update2,
        weights.W_message3, weights.W_update3,
        weights.W_message4, weights.W_update4,
        weights.W_attn_query, weights.W_attn_key,
        weights.W_attn_query2, weights.W_attn_key2,
        weights.W_attn_query3, weights.W_attn_key3,
        weights.W_attn_query4, weights.W_attn_key4,
        weights.W_conv_gate, weights.W_conv_value,
        weights.W_3body, weights.W_3body_update,
        weights.W_input_proj,
      ]) {
        const rows = wMat.length;
        const cols = wMat[0]?.length ?? 0;
        const perturbation = new Array(cols);
        for (let j = 0; j < cols; j++) perturbation[j] = rng() > 0.5 ? 1 : -1;
        for (let i = 0; i < rows; i++) {
          const directionSign = avgDLdPooled[i % pooledLen] >= 0 ? 1 : -1;
          for (let j = 0; j < cols; j++) {
            wMat[i][j] -= graphLR * directionSign * perturbation[j] * 0.01;
          }
        }
      }

      for (let g = 0; g < weights.residual_gates.length; g++) {
        const gateGrad = avgDLdPooled[g % HIDDEN_DIM] * 0.01;
        weights.residual_gates[g] -= lr * gateGrad;
        weights.residual_gates[g] = Math.max(-3, Math.min(3, weights.residual_gates[g]));
      }
    }

    if (totalSamples > 0 && totalLoss / totalSamples < 0.01) break;
  }

  const scrubMatrix = (m: number[][]) => { for (let i = 0; i < m.length; i++) for (let j = 0; j < m[i].length; j++) if (!Number.isFinite(m[i][j])) m[i][j] = 0; };
  const scrubVector = (v: number[]) => { for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) v[i] = 0; };
  for (const wMat of [
    weights.W_message, weights.W_update, weights.W_message2, weights.W_update2,
    weights.W_message3, weights.W_update3, weights.W_message4, weights.W_update4,
    weights.W_attn_query, weights.W_attn_key, weights.W_attn_query2, weights.W_attn_key2,
    weights.W_attn_query3, weights.W_attn_key3, weights.W_attn_query4, weights.W_attn_key4,
    weights.W_conv_gate, weights.W_conv_value, weights.W_input_proj, weights.W_3body, weights.W_3body_update,
    weights.W_mlp1, weights.W_mlp2, weights.W_mlp2_var, weights.W_attn_pool,
  ]) { scrubMatrix(wMat); }
  for (const bVec of [weights.b_mlp1, weights.b_mlp2, weights.b_mlp2_var, weights.b_conv_gate, weights.b_conv_value, weights.b_input_proj, weights.W_pressure, weights.residual_gates]) {
    scrubVector(bVec);
  }

  for (const wMat of [
    weights.W_message, weights.W_update, weights.W_message2, weights.W_update2,
    weights.W_message3, weights.W_update3, weights.W_message4, weights.W_update4,
    weights.W_attn_query, weights.W_attn_key, weights.W_attn_query2, weights.W_attn_key2,
    weights.W_attn_query3, weights.W_attn_key3, weights.W_attn_query4, weights.W_attn_key4,
    weights.W_conv_gate, weights.W_conv_value, weights.W_input_proj, weights.W_3body, weights.W_3body_update,
    weights.W_mlp1, weights.W_mlp2, weights.W_mlp2_var, weights.W_attn_pool,
  ]) { invalidateFlatCache(wMat); }

  weights.trainedAt = Date.now();
  weights.nSamples = trainingData.length;
  return weights;
}

function getEnsembleModels(): GNNWeights[] {
  const now = Date.now();
  if (cachedEnsembleModels && (now - modelTrainedAt) < MODEL_STALE_MS) {
    return cachedEnsembleModels;
  }

  const trainingData: TrainingSample[] = SUPERCON_TRAINING_DATA
    .filter(e => e.isSuperconductor)
    .map(e => ({
      formula: e.formula,
      tc: e.tc,
      formationEnergy: undefined,
      structure: undefined,
    }));

  cachedEnsembleModels = trainEnsemble(trainingData);
  modelTrainedAt = now;
  updateTrainingEmbeddings(trainingData, cachedEnsembleModels[0]);
  return cachedEnsembleModels;
}

let heldOutValidationSet: TrainingSample[] = [];

function splitTrainValidation(data: TrainingSample[], valFraction: number = 0.2, seed: number = 42): {
  train: TrainingSample[];
  validation: TrainingSample[];
} {
  if (data.length < 10) {
    return { train: data, validation: [] };
  }

  const rng = seededRandom(seed);
  const indices = Array.from({ length: data.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const valSize = Math.max(2, Math.floor(data.length * valFraction));
  const valIndices = new Set(indices.slice(0, valSize));

  const train: TrainingSample[] = [];
  const validation: TrainingSample[] = [];
  for (let i = 0; i < data.length; i++) {
    if (valIndices.has(i)) {
      validation.push(data[i]);
    } else {
      train.push(data[i]);
    }
  }

  return { train, validation };
}

export function getHeldOutValidationSet(): TrainingSample[] {
  return [...heldOutValidationSet];
}

const ENSEMBLE_SEEDS = [42, 7919, 104729, 15485863, 32452843];
const BOOTSTRAP_RATIOS = [0.75, 0.80, 0.85, 0.80, 0.75];

function bootstrapSample(data: TrainingSample[], ratio: number, rng: () => number): TrainingSample[] {
  const n = Math.max(1, Math.floor(data.length * ratio));
  const sampled: TrainingSample[] = [];
  for (let i = 0; i < n; i++) {
    sampled.push(data[Math.floor(rng() * data.length)]);
  }
  return sampled;
}

export function trainEnsemble(trainingData: TrainingSample[]): GNNWeights[] {
  const { train, validation } = splitTrainValidation(trainingData);
  heldOutValidationSet = validation;

  const models: GNNWeights[] = [];
  for (let i = 0; i < ENSEMBLE_SIZE; i++) {
    const rng = seededRandom(ENSEMBLE_SEEDS[i]);
    const w = initWeights(rng);
    const bootstrapRng = seededRandom(ENSEMBLE_SEEDS[i] + 31);
    const bootstrapped = bootstrapSample(train, BOOTSTRAP_RATIOS[i], bootstrapRng);
    const trained = trainGNNSurrogate(bootstrapped, w);
    models.push(trained);
  }

  if (validation.length > 0) {
    console.log(`[GNN] Train/validation split: ${train.length} train, ${validation.length} validation (${(validation.length / trainingData.length * 100).toFixed(1)}%)`);
  }

  return models;
}

export async function trainEnsembleAsync(trainingData: TrainingSample[]): Promise<GNNWeights[]> {
  const { train, validation } = splitTrainValidation(trainingData);
  heldOutValidationSet = validation;

  const models: GNNWeights[] = [];
  for (let i = 0; i < ENSEMBLE_SIZE; i++) {
    const rng = seededRandom(ENSEMBLE_SEEDS[i]);
    const w = initWeights(rng);
    const bootstrapRng = seededRandom(ENSEMBLE_SEEDS[i] + 31);
    const bootstrapped = bootstrapSample(train, BOOTSTRAP_RATIOS[i], bootstrapRng);
    const trained = trainGNNSurrogate(bootstrapped, w);
    models.push(trained);
    await new Promise(resolve => setImmediate(resolve));
  }

  if (validation.length > 0) {
    console.log(`[GNN] Train/validation split: ${train.length} train, ${validation.length} validation (${(validation.length / trainingData.length * 100).toFixed(1)}%)`);
  }

  return models;
}

export function getGNNModel(): GNNWeights {
  return getEnsembleModels()[0];
}

export function invalidateGNNModel(): void {
  cachedEnsembleModels = null;
  modelTrainedAt = 0;
  gnnPredictionCache.clear();
}

export function setCachedEnsemble(models: GNNWeights[], trainingData?: { formula: string; tc: number }[]): void {
  cachedEnsembleModels = models;
  modelTrainedAt = Date.now();
  if (trainingData && models.length > 0) {
    updateTrainingEmbeddings(trainingData, models[0]);
  }
}

export interface GNNVersionRecord {
  version: number;
  trainedAt: number;
  datasetSize: number;
  ensembleSize: number;
  r2: number;
  mae: number;
  rmse: number;
  trigger: string;
  dftSamples: number;
  enrichedSamples: number;
}

let gnnModelVersion = 0;
const gnnVersionHistory: GNNVersionRecord[] = [];
const GNN_VERSION_HISTORY_MAX = 50;

export function logGNNVersion(trigger: string, datasetSize: number, dftSamples = 0, enrichedSamples = 0): GNNVersionRecord {
  gnnModelVersion++;

  const validationSet = heldOutValidationSet.length > 0
    ? heldOutValidationSet
    : SUPERCON_TRAINING_DATA
        .filter(e => e.isSuperconductor && e.tc > 0)
        .slice(0, 50);

  let sumSquaredError = 0;
  let sumAbsError = 0;
  let sumActual = 0;
  let sumActualSq = 0;
  const n = validationSet.length;

  for (const entry of validationSet) {
    const pred = getGNNPrediction(entry.formula);
    const actual = entry.tc;
    const predicted = pred.predictedTc;
    const error = predicted - actual;
    sumSquaredError += error * error;
    sumAbsError += Math.abs(error);
    sumActual += actual;
    sumActualSq += actual * actual;
  }

  const meanActual = n > 0 ? sumActual / n : 0;
  const ssRes = sumSquaredError;
  const ssTot = n > 0 ? sumActualSq - n * meanActual * meanActual : 1;
  const r2 = ssTot > 0 ? Math.max(-1, 1 - ssRes / ssTot) : 0;
  const mae = n > 0 ? sumAbsError / n : 0;
  const rmse = n > 0 ? Math.sqrt(sumSquaredError / n) : 0;

  const record: GNNVersionRecord = {
    version: gnnModelVersion,
    trainedAt: Date.now(),
    datasetSize,
    ensembleSize: ENSEMBLE_SIZE,
    r2: Math.round(r2 * 10000) / 10000,
    mae: Math.round(mae * 100) / 100,
    rmse: Math.round(rmse * 100) / 100,
    trigger,
    dftSamples,
    enrichedSamples,
  };

  gnnVersionHistory.push(record);
  if (gnnVersionHistory.length > GNN_VERSION_HISTORY_MAX) {
    gnnVersionHistory.shift();
  }

  const valSource = heldOutValidationSet.length > 0 ? `held-out (${heldOutValidationSet.length})` : "fallback (first 50)";
  console.log(`[GNN] Version ${record.version} logged: R²=${record.r2}, MAE=${record.mae}, RMSE=${record.rmse}, trigger=${trigger}, dataset=${datasetSize}, dft=${dftSamples}, enriched=${enrichedSamples}, validation=${valSource}`);

  return record;
}

export function getGNNVersionHistory(): GNNVersionRecord[] {
  return [...gnnVersionHistory];
}

export function getGNNModelVersion(): number {
  return gnnModelVersion;
}

const GNN_PRED_CACHE_MAX = 500;
const gnnPredictionCache = new Map<string, { prediction: GNNPrediction; trainedAt: number }>();

export function getGNNPrediction(formula: string, structure?: any): GNNPrediction {
  const weights = getGNNModel();
  const currentTrainedAt = modelTrainedAt;
  const cacheKey = formula;
  const cached = gnnPredictionCache.get(cacheKey);
  if (cached && cached.trainedAt === currentTrainedAt) {
    return cached.prediction;
  }
  const graph = buildCrystalGraph(formula, structure);
  const prediction = GNNPredict(graph, weights);
  if (gnnPredictionCache.size >= GNN_PRED_CACHE_MAX) {
    const firstKey = gnnPredictionCache.keys().next().value;
    if (firstKey !== undefined) gnnPredictionCache.delete(firstKey);
  }
  gnnPredictionCache.set(cacheKey, { prediction, trainedAt: currentTrainedAt });
  return prediction;
}

function perturbWeights(w: GNNWeights, rng: () => number, scale: number): GNNWeights {
  const perturbed = cloneWeights(w);
  const perturbMatrix = (mat: number[][]) => {
    for (let i = 0; i < mat.length; i++) {
      for (let j = 0; j < mat[i].length; j++) {
        if (rng() < 0.3) {
          mat[i][j] *= (1 + (rng() - 0.5) * scale);
        }
      }
    }
  };

  perturbMatrix(perturbed.W_message);
  perturbMatrix(perturbed.W_update);
  perturbMatrix(perturbed.W_message2);
  perturbMatrix(perturbed.W_update2);
  perturbMatrix(perturbed.W_message3);
  perturbMatrix(perturbed.W_update3);
  perturbMatrix(perturbed.W_message4);
  perturbMatrix(perturbed.W_update4);
  perturbMatrix(perturbed.W_attn_query);
  perturbMatrix(perturbed.W_attn_key);
  perturbMatrix(perturbed.W_attn_query2);
  perturbMatrix(perturbed.W_attn_key2);
  perturbMatrix(perturbed.W_attn_query3);
  perturbMatrix(perturbed.W_attn_key3);
  perturbMatrix(perturbed.W_attn_query4);
  perturbMatrix(perturbed.W_attn_key4);
  perturbMatrix(perturbed.W_conv_gate);
  perturbMatrix(perturbed.W_conv_value);
  perturbMatrix(perturbed.W_input_proj);
  perturbMatrix(perturbed.W_3body);
  perturbMatrix(perturbed.W_3body_update);
  perturbMatrix(perturbed.W_attn_pool);
  for (let i = 0; i < perturbed.W_pressure.length; i++) {
    if (rng() < 0.3) {
      perturbed.W_pressure[i] *= (1 + (rng() - 0.5) * scale);
    }
  }
  for (let i = 0; i < perturbed.residual_gates.length; i++) {
    perturbed.residual_gates[i] = Math.max(0.1, Math.min(0.9,
      perturbed.residual_gates[i] + (rng() - 0.5) * scale * 0.2));
  }
  perturbMatrix(perturbed.W_mlp1);
  perturbMatrix(perturbed.W_mlp2);
  perturbMatrix(perturbed.W_mlp2_var);

  return perturbed;
}

export function gnnPredictWithUncertainty(formula: string, prototype?: string, pressureGpa?: number): GNNPredictionWithUncertainty {
  const ensembleModels = getEnsembleModels();
  const predictions: GNNPrediction[] = [];
  const perModelMeans: { tc: number; fe: number; lambda: number; bg: number }[] = [];

  for (let m = 0; m < ensembleModels.length; m++) {
    const modelWeights = ensembleModels[m];
    const modelPreds: GNNPrediction[] = [];

    for (let d = 0; d < MC_DROPOUT_PASSES; d++) {
      const dropoutRng = seededRandom(m * 1000 + d * 137 + 7);
      const graph = prototype
        ? buildPrototypeGraph(formula, prototype, pressureGpa)
        : buildCrystalGraph(formula, undefined, pressureGpa);
      const pred = GNNPredict(graph, modelWeights, dropoutRng);
      predictions.push(pred);
      modelPreds.push(pred);
    }

    perModelMeans.push({
      tc: modelPreds.reduce((s, p) => s + p.predictedTc, 0) / modelPreds.length,
      fe: modelPreds.reduce((s, p) => s + p.formationEnergy, 0) / modelPreds.length,
      lambda: modelPreds.reduce((s, p) => s + p.lambda, 0) / modelPreds.length,
      bg: modelPreds.reduce((s, p) => s + p.bandgap, 0) / modelPreds.length,
    });
  }

  const tcValues = predictions.map(p => p.predictedTc);
  const feValues = predictions.map(p => p.formationEnergy);
  const lambdaValues = predictions.map(p => p.lambda);
  const bgValues = predictions.map(p => p.bandgap);
  const dosValues = predictions.map(p => p.dosProxy);
  const stabValues = predictions.map(p => p.stabilityProbability);

  const meanTc = tcValues.reduce((s, v) => s + v, 0) / tcValues.length;
  const meanFE = feValues.reduce((s, v) => s + v, 0) / feValues.length;
  const meanLambda = lambdaValues.reduce((s, v) => s + v, 0) / lambdaValues.length;
  const meanBG = bgValues.reduce((s, v) => s + v, 0) / bgValues.length;
  const meanDOS = dosValues.reduce((s, v) => s + v, 0) / dosValues.length;
  const meanStab = stabValues.reduce((s, v) => s + v, 0) / stabValues.length;

  const tcStd = Math.sqrt(tcValues.reduce((s, v) => s + (v - meanTc) ** 2, 0) / tcValues.length);
  const feStd = Math.sqrt(feValues.reduce((s, v) => s + (v - meanFE) ** 2, 0) / feValues.length);
  const lambdaStd = Math.sqrt(lambdaValues.reduce((s, v) => s + (v - meanLambda) ** 2, 0) / lambdaValues.length);
  const bgStd = Math.sqrt(bgValues.reduce((s, v) => s + (v - meanBG) ** 2, 0) / bgValues.length);

  const normalizedTcUnc = meanTc > 0 ? tcStd / Math.max(meanTc, 1) : tcStd;
  const normalizedFeUnc = feStd;
  const normalizedLambdaUnc = meanLambda > 0 ? lambdaStd / Math.max(meanLambda, 0.1) : lambdaStd;
  const normalizedBgUnc = bgStd / Math.max(meanBG, 0.1);

  const epistemicTcVar = perModelMeans.reduce((s, m) => s + (m.tc - meanTc) ** 2, 0) / Math.max(1, perModelMeans.length);
  const epistemicLambdaVar = perModelMeans.reduce((s, m) => s + (m.lambda - meanLambda) ** 2, 0) / Math.max(1, perModelMeans.length);
  const ensembleUncertainty = Math.min(1.0, Math.sqrt(epistemicTcVar) / Math.max(meanTc, 1));

  const aleatoricTcVar = predictions.reduce((s, p) => s + p.predictedTcVar, 0) / predictions.length;
  const aleatoricLambdaVar = predictions.reduce((s, p) => s + p.lambdaVar, 0) / predictions.length;
  const aleatoricUncNorm = Math.min(1.0, Math.sqrt(aleatoricTcVar) / Math.max(meanTc, 1));

  let mcTcVar = 0;
  let mcLambdaVar = 0;
  let mcDropoutUncertainty = 0;
  for (let m = 0; m < ensembleModels.length; m++) {
    const modelPreds = predictions.slice(m * MC_DROPOUT_PASSES, (m + 1) * MC_DROPOUT_PASSES);
    const modelMeanTc = perModelMeans[m].tc;
    const modelMeanLambda = perModelMeans[m].lambda;
    const withinTcVar = modelPreds.reduce((s, p) => s + (p.predictedTc - modelMeanTc) ** 2, 0) / modelPreds.length;
    const withinLambdaVar = modelPreds.reduce((s, p) => s + (p.lambda - modelMeanLambda) ** 2, 0) / modelPreds.length;
    mcTcVar += withinTcVar;
    mcLambdaVar += withinLambdaVar;
    mcDropoutUncertainty += Math.sqrt(withinTcVar) / Math.max(modelMeanTc, 1);
  }
  mcTcVar /= ensembleModels.length;
  mcLambdaVar /= ensembleModels.length;
  mcDropoutUncertainty = Math.min(1.0, mcDropoutUncertainty / ensembleModels.length);

  const totalTcVar = epistemicTcVar + aleatoricTcVar + mcTcVar;
  const totalTcStd = Math.sqrt(totalTcVar);
  const totalLambdaVar = epistemicLambdaVar + aleatoricLambdaVar + mcLambdaVar;
  const totalLambdaStd = Math.sqrt(totalLambdaVar);

  const tcCI95Lower = Math.max(0, meanTc - 1.96 * totalTcStd);
  const tcCI95Upper = meanTc + 1.96 * totalTcStd;
  const lambdaCI95Lower = Math.max(0, meanLambda - 1.96 * totalLambdaStd);
  const lambdaCI95Upper = meanLambda + 1.96 * totalLambdaStd;

  const avgLatent = predictions.reduce((acc, p) => {
    for (let i = 0; i < p.latentEmbedding.length; i++) {
      acc[i] = (acc[i] ?? 0) + p.latentEmbedding[i] / predictions.length;
    }
    return acc;
  }, new Array(HIDDEN_DIM).fill(0));
  const latentDist = computeLatentDistance(avgLatent);

  const combinedUncertainty = Math.min(1.0,
    0.30 * normalizedTcUnc +
    0.15 * normalizedFeUnc +
    0.10 * normalizedLambdaUnc +
    0.20 * ensembleUncertainty +
    0.15 * latentDist +
    0.10 * normalizedBgUnc
  );

  const totalPredictions = predictions.length;
  const phononStabilityVotes = predictions.filter(p => p.phononStability).length;
  const phononStable = phononStabilityVotes > totalPredictions / 2;

  const avgConfidence = predictions.reduce((s, p) => s + p.confidence, 0) / totalPredictions;
  const confidenceAdjusted = avgConfidence * (1.0 - combinedUncertainty * 0.5);

  const uncertaintyBreakdown: UncertaintyBreakdown = {
    ensemble: Math.round(ensembleUncertainty * 1000) / 1000,
    mcDropout: Math.round(mcDropoutUncertainty * 1000) / 1000,
    aleatoric: Math.round(aleatoricUncNorm * 1000) / 1000,
    latentDistance: Math.round(latentDist * 1000) / 1000,
    perTarget: {
      tc: Math.round(normalizedTcUnc * 1000) / 1000,
      formationEnergy: Math.round(normalizedFeUnc * 1000) / 1000,
      lambda: Math.round(normalizedLambdaUnc * 1000) / 1000,
      bandgap: Math.round(normalizedBgUnc * 1000) / 1000,
    },
  };

  const s = (v: number, fb = 0) => Number.isFinite(v) ? v : fb;
  return {
    tc: Math.round(s(meanTc) * 10) / 10,
    formationEnergy: Math.round(s(meanFE) * 1000) / 1000,
    lambda: Math.round(s(meanLambda) * 1000) / 1000,
    bandgap: Math.round(s(meanBG) * 1000) / 1000,
    dosProxy: Math.round(s(meanDOS) * 1000) / 1000,
    stabilityProbability: Math.round(s(meanStab) * 1000) / 1000,
    uncertainty: Math.round(s(combinedUncertainty, 0.5) * 1000) / 1000,
    uncertaintyBreakdown,
    phononStability: phononStable,
    confidence: Math.round(Math.max(0.05, Math.min(0.95, s(confidenceAdjusted, 0.5))) * 100) / 100,
    latentDistance: Math.round(s(latentDist) * 1000) / 1000,
    tcCI95: [Math.round(s(tcCI95Lower) * 10) / 10, Math.round(s(tcCI95Upper) * 10) / 10],
    lambdaCI95: [Math.round(s(lambdaCI95Lower) * 1000) / 1000, Math.round(s(lambdaCI95Upper) * 1000) / 1000],
    epistemicUncertainty: Math.round(s(Math.sqrt(epistemicTcVar)) * 100) / 100,
    aleatoricUncertainty: Math.round(s(Math.sqrt(aleatoricTcVar)) * 100) / 100,
    totalStd: Math.round(s(totalTcStd) * 100) / 100,
  };
}

export function getUncertaintyDecomposition(formula: string): UncertaintyBreakdown {
  const pred = gnnPredictWithUncertainty(formula);
  return pred.uncertaintyBreakdown;
}

export function getPrototypeCoordinations(): Record<string, PrototypeCoordination> {
  return PROTOTYPE_COORDINATIONS;
}

export interface DFTTrainingRecord {
  formula: string;
  tc: number;
  formationEnergy: number | null;
  bandGap: number | null;
  structure?: any;
  prototype?: string;
  source: "dft" | "external" | "active-learning" | "supercon";
  addedAt: number;
  lambda?: number;
  omegaLog?: number;
  dosAtEF?: number;
  phononStable?: boolean;
}

interface DFTDatasetGrowthEntry {
  timestamp: number;
  size: number;
  source: string;
}

const MAX_DFT_TRAINING_DATASET = 5000;
const dftTrainingDataset: DFTTrainingRecord[] = [];
const datasetGrowthHistory: DFTDatasetGrowthEntry[] = [];

export function addDFTTrainingResult(record: {
  formula: string;
  tc: number;
  formationEnergy?: number | null;
  bandGap?: number | null;
  structure?: any;
  prototype?: string;
  source: DFTTrainingRecord["source"];
  lambda?: number;
  omegaLog?: number;
  dosAtEF?: number;
  phononStable?: boolean;
}): boolean {
  const existing = dftTrainingDataset.find(r => r.formula === record.formula);
  if (existing) {
    if (record.formationEnergy != null) existing.formationEnergy = record.formationEnergy;
    if (record.bandGap != null) existing.bandGap = record.bandGap;
    if (record.tc > 0 && existing.tc === 0) existing.tc = record.tc;
    if (record.structure) existing.structure = record.structure;
    if (record.prototype) existing.prototype = record.prototype;
    if (record.lambda != null) existing.lambda = record.lambda;
    if (record.omegaLog != null) existing.omegaLog = record.omegaLog;
    if (record.dosAtEF != null) existing.dosAtEF = record.dosAtEF;
    if (record.phononStable != null) existing.phononStable = record.phononStable;
    return false;
  }

  if (dftTrainingDataset.length >= MAX_DFT_TRAINING_DATASET) {
    return false;
  }

  dftTrainingDataset.push({
    formula: record.formula,
    tc: record.tc,
    formationEnergy: record.formationEnergy ?? null,
    bandGap: record.bandGap ?? null,
    structure: record.structure,
    prototype: record.prototype,
    source: record.source,
    addedAt: Date.now(),
    lambda: record.lambda,
    omegaLog: record.omegaLog,
    dosAtEF: record.dosAtEF,
    phononStable: record.phononStable,
  });

  datasetGrowthHistory.push({
    timestamp: Date.now(),
    size: dftTrainingDataset.length,
    source: record.source,
  });

  if (datasetGrowthHistory.length > 200) {
    datasetGrowthHistory.splice(0, datasetGrowthHistory.length - 200);
  }

  return true;
}

export function getDFTTrainingDataset(): DFTTrainingRecord[] {
  return [...dftTrainingDataset];
}

export function getDFTTrainingDatasetStats(): {
  totalSize: number;
  bySource: Record<string, number>;
  growthHistory: DFTDatasetGrowthEntry[];
  oldestEntry: number | null;
  newestEntry: number | null;
} {
  const bySource: Record<string, number> = {};
  let oldestEntry: number | null = null;
  let newestEntry: number | null = null;

  for (const record of dftTrainingDataset) {
    bySource[record.source] = (bySource[record.source] ?? 0) + 1;
    if (oldestEntry === null || record.addedAt < oldestEntry) oldestEntry = record.addedAt;
    if (newestEntry === null || record.addedAt > newestEntry) newestEntry = record.addedAt;
  }

  return {
    totalSize: dftTrainingDataset.length,
    bySource,
    growthHistory: [...datasetGrowthHistory],
    oldestEntry,
    newestEntry,
  };
}

setImmediate(() => {
  try {
    getEnsembleModels();
    console.log(`[GNN] Pre-warmed ${ENSEMBLE_SIZE}-model ensemble at startup`);
    const superconCount = SUPERCON_TRAINING_DATA.filter(e => e.isSuperconductor).length;
    logGNNVersion("startup", superconCount, 0, 0);
  } catch (e: any) {
    console.error(`[GNN] Pre-warm failed: ${e?.message?.slice(0, 200)}`);
  }
});
