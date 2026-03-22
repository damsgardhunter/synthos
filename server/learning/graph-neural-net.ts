import { createHash } from "crypto";
import { isMainThread } from "worker_threads";
import { ELEMENTAL_DATA, getElementData, isTransitionMetal } from "./elemental-data";
import { extractFeatures } from "./ml-predictor";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { storage } from "../storage";
import { computeThorFeatureVector, resolveSGNumber } from "../crystal/symmetry-subgroups";
import { predictLambda } from "./lambda-regressor";
import { allenDynesTcRaw } from "./physics-engine";
import { normalizeSpaceGroup, matchPrototype } from "./structure-predictor";
import { parseFormulaCounts as parseFormulaCountsCanonical } from "./utils";

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
  globalFeatures?: number[]; // 23-dim: 13 composition + 6 physics hints + 4 engineered features
}

/**
 * Pre-computed physics hints that can be injected into the GNN's global feature
 * vector without leakage. These are measured/estimated material properties —
 * NOT model outputs — so passing them alongside the graph is safe at train time.
 */
export interface PhysicsFeatureHints {
  /** Electron-phonon coupling constant λ (BCS: ~0.4 weak, ~2+ strong). */
  electronPhononLambda?: number | null;
  /** Formation energy in eV/atom (negative = thermodynamically stable). */
  formationEnergy?: number | null;
  /** Distance above the convex hull in eV/atom (0 = on hull, stable). */
  hullDistance?: number | null;
  /** Density of states at the Fermi level (states/eV/atom). */
  dosAtEF?: number | null;
  /** Log mean phonon frequency proxy (from Debye temperature or phonon spectrum). */
  logPhononFreq?: number | null;
  // ── Data provenance metadata (used for source one-hot and availability mask) ──
  /** Source dataset tag: 'qe-dft' | 'hamidieh' | 'jarvis-sc' | '3dsc-mp' | 'contrast-jarvis' | 'contrast-mp' */
  sourceTag?: string;
  /** True when λ comes from a real DFPT measurement (not a composition-based proxy fallback). */
  hasLambdaMeasured?: boolean;
  /** True when ω_log comes from a real DFPT measurement. */
  hasOmegaLogMeasured?: boolean;
  /** True when formation energy was measured (not the neutral-prior fallback). */
  hasFEMeasured?: boolean;
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
  /** SchNet continuous filter MLP layer 1: HIDDEN_DIM × N_GAUSSIAN_BASIS */
  W_filter1: number[][];
  b_filter1: number[];
  /** SchNet continuous filter MLP layer 2: HIDDEN_DIM × HIDDEN_DIM */
  W_filter2: number[][];
  b_filter2: number[];
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
  // ── v15: Dedicated classification head ───────────────────────────────────
  /** CLS_DIM × pooledLen — dedicated P(SC) pathway independent of Tc regression. */
  W_cls1: number[][];
  b_cls1: number[];
  /** CLS_DIM → scalar logit for P(SC). */
  W_cls2: number[];
  b_cls2: number;
  // ── GLFN-TC: Graph Learning Module weights ───────────────────────────────
  /** 119 × GRAPH_FEAT_DIM — one learnable feature vector per element (atomic number 0–118). */
  W_elem_feat: number[][];
  /** GRAPH_FEAT_DIM × GRAPH_FEAT_DIM — asymmetric bilinear compatibility matrix.
   *  adaptive_logit(i,j) = W_elem_feat[atomI] · W_graph_adapt · W_elem_feat[atomJ]
   *  Added as a scalar bias to every CGCNN gate dimension so the model learns
   *  which element pairs should pass more (or less) information. */
  W_graph_adapt: number[][];
  // ── GLFN-TC: Dense Residual scalar gate ──────────────────────────────────
  /** Scalar raw gate. sigmoid(dense_skip_gate) controls how much H0 (original
   *  post-projection embeddings) is added to H1 before layer-2 attention,
   *  creating a dense skip connection akin to GLFN-TC §2.4. */
  dense_skip_gate: number;
  trainedAt: number;
  nSamples: number;
}

export interface GNNPrediction {
  formationEnergy: number;
  phononStability: boolean;
  predictedTc: number;
  /** Characteristic phonon frequency ω_log (K) — Allen-Dynes intermediate; inspect for physical validity. */
  omegaLog: number;
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

interface AttnLayerCache {
  inputEmbs: number[][];
  preActs: number[][];
  preNormActs: number[][];  // h + activation(update) before layerNorm
  attnWts: number[][];
  neighborLists: number[][];
}

interface CGCNNLayerCache {
  inputEmbs: number[][];
  filterPreActs: number[][][];
  filterH1s: number[][][];
  filterOuts: number[][][];
  rbfs: number[][][];
  preNormActs: number[][];
  cutoffWts: number[][];
  totalWeights: number[];
  adaptiveLogits?: number[][];
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
  attnCaches: AttnLayerCache[];
  cgcnnCache?: CGCNNLayerCache;
  inputProjInputs: number[][];
  inputProjPreActs: number[][];
  maxPoolArgmax: number[];
  attnPoolWeights: number[];
  /** GLFN-TC Dense Residual: H0 embeddings saved after input projection.
   *  Used in backward to route the dense-skip gradient back to input_proj. */
  denseH0?: number[][];
  /** v15: Classification head pre- and post-activation. */
  zCls: number[];
  hCls: number[];
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
  weightProfile?: {
    mode: 'high-tc' | 'standard';
    tc: number;
    ensemble: number;
    latent: number;
    formationEnergy: number;
    lambda: number;
    bandgap: number;
  };
}

export interface GNNPredictionWithUncertainty {
  tc: number;
  /** Ensemble-mean characteristic phonon frequency ω_log (K) — Allen-Dynes intermediate. */
  omegaLog: number;
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
const N_GAUSSIAN_BASIS = 40;             // expanded from 20 — denser RBF gives finer distance discrimination
const EDGE_DIM = N_GAUSSIAN_BASIS + 4;   // derived: 40 RBF + bond order, EN diff, ionic char, radius sum
const OUTPUT_DIM = 16;
const GLOBAL_COMP_DIM = 23;
// Coulomb pseudopotential μ* — fixed at conventional BCS value.
// Typical range: 0.10 (metals), 0.12-0.13 (hydrides, higher Coulomb screening).
const FIXED_MU_STAR = 0.10;
// Physical upper bound on λ — no known conventional SC has λ > 3.8 (LaH10 hydride); cap prevents Allen-Dynes blowup.
const LAMBDA_MAX = 5.5;
// Allen-Dynes formula is unconventional above ~300 K; cap predictions at training normalization ceiling.
const TC_MAX_K = 300;
// log1p normalisation for Tc regression (direct, no sigmoid):
//   training target = log1p(Tc / 10) / TC_LOG_SCALE  ∈ [0, 1]
//   out[8] is trained to approximate this value directly (no sigmoid activation)
//   inference Tc    = 10 * expm1(max(0, out[8]) * TC_LOG_SCALE)
//
// Why no sigmoid: sigmoid(out8) → 0 as out[8] → -∞, making gradient
//   dL/d(out8) = err * sigmoid'(out8) → 0 — the vanishing gradient trap.
// Once out[8] enters negative saturation (predicting ~0 K) it cannot recover.
// Direct regression: gradient = 2 * coeff * (out8 - target), never vanishes.
const TC_LOG_SCALE = Math.log1p(TC_MAX_K / 10);  // log1p(30) ≈ 3.434
// Dedicated classification head hidden dimension — large enough to learn P(SC)
// without competing with Tc regression in the shared h1 representation.
const CLS_DIM = 24;
// Absolute ceiling on ω_log — even high-pressure hydrides (H₃S, LaH₁₀) have ω_log < 1500 K.
const OMEGA_LOG_MAX = 1500;              // 6 composition + 7 XGBoost-inspired composition + 6 physics (λ, logω, DOS, FE, isCuprate, isIronBased)
// CGCNN_CONCAT_DIM removed - replaced by SchNet filter MLP (W_filter1/W_filter2)
export const ENSEMBLE_SIZE = 5;
const MC_DROPOUT_PASSES = 10;
const MC_DROPOUT_RATE = 0.25;
const GNN_MSG_LAYERS = 2;         // active message-passing layers (2 = less overfit for small datasets)
const WEIGHT_DECAY = 1e-4;        // AdamW L2 regularization
// ── GLFN-TC inspired modules ──────────────────────────────────────────────────
// Graph Learning Module: learnable per-element feature vectors that compute an
// adaptive adjacency score, modulating the CGCNN gate to discover latent chemical
// pair compatibility beyond fixed bond-distance geometry (paper §2.2).
const GRAPH_FEAT_DIM = 12;        // dimension of per-element learnable feature vector
// Densely Connected Residual Module: a learnable skip gate passes H0 (post-input-
// projection embeddings) forward to the layer-2 attention input, so later layers
// can access the original node representations without over-smoothing (paper §2.4).
// dense_skip_gate scalar is initialised to -2 → sigmoid(-2)≈0.12 (small initial skip).
const GAUSSIAN_START = 0.5;
const GAUSSIAN_END = 6.0;
const GAUSSIAN_STEP = (GAUSSIAN_END - GAUSSIAN_START) / (N_GAUSSIAN_BASIS - 1);
const GAUSSIAN_WIDTH = GAUSSIAN_STEP;
// Log-scale Tc normalisation: log1p(Tc) / log1p(300) maps 0–300K → 0–1
// with proportional spacing. Replacing the old linear /300 encoding.

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
  return parseFormulaCountsCanonical(formula);
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

/** Backward pass through LayerNorm (no learned scale/bias).
 *  preNorm: the pre-norm input x (same as vec passed to layerNorm).
 *  dLdY:    upstream gradient ∂L/∂y where y = (x − μ) / σ.
 *  Returns: ∂L/∂x = (1/σ) * [dL/dy − mean(dL/dy) − ŷ·mean(dL/dy·ŷ)]
 */
function layerNormBackward(preNorm: number[], dLdY: number[], eps: number = 1e-5): number[] {
  const n = preNorm.length;
  if (n === 0) return [];
  let mean = 0;
  for (let i = 0; i < n; i++) mean += preNorm[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) { const d = preNorm[i] - mean; variance += d * d; }
  variance /= n;
  const std = Math.sqrt(variance + eps);
  let sumDy = 0, sumDyY = 0;
  for (let i = 0; i < n; i++) {
    const yHat = (preNorm[i] - mean) / std;
    sumDy  += dLdY[i];
    sumDyY += dLdY[i] * yHat;
  }
  const dLdX = new Array(n);
  for (let i = 0; i < n; i++) {
    const yHat = (preNorm[i] - mean) / std;
    dLdX[i] = (dLdY[i] - sumDy / n - yHat * sumDyY / n) / std;
  }
  return dLdX;
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

/** SiLU (Swish) activation - smooth non-monotonic, good for filter MLPs. */
function silu(x: number): number {
  return x * sigmoid(x);
}

/** Derivative of SiLU: sigma(x) + x*sigma(x)*(1 - sigma(x)) */
function siluGrad(x: number): number {
  const s = sigmoid(x);
  return s + x * s * (1 - s);
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
  const elements = Object.keys(counts).sort();
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

export function buildPrototypeGraph(formula: string, prototype: string, pressureGpa?: number, hints?: PhysicsFeatureHints): CrystalGraph {
  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts).sort();
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];

  if (!protoInfo) {
    return buildCrystalGraph(formula, undefined, pressureGpa, hints);
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

  // Normalize ordering for permutation invariance
  edges.sort((a, b) => a.source !== b.source ? a.source - b.source : a.target - b.target);
  for (let i = 0; i < adjacency.length; i++) adjacency[i].sort((a, b) => a - b);
  const edgeIndex = buildEdgeIndex(nodes, edges);
  const globalFeatures = computeGlobalCompositionFeatures(rawCounts, hints, pressureGpa);
  const threeBodyFeatures = compute3BodyFeatures({ nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, prototype, globalFeatures });
  return { nodes, edges, threeBodyFeatures, adjacency, edgeIndex, formula, prototype, pressureGpa, globalFeatures };
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
    // λ_proxy: peaks for half-filled d-shells (BCS SCs) — replaces thermalConductivity (redundant with Debye)
    dOcc > 0 ? dOcc * (1 - Math.abs(dOcc - 0.5)) * Math.min(1.0, (data?.debyeTemperature ?? 300) / 800) : 0,
    Math.min(1.0, Math.abs(data?.electronAffinity ?? 0) / 4.0),
    Math.min(1.0, (data?.atomicVolume ?? 15) / 80),
    en > 2.0 ? 1.0 : en > 1.5 ? 0.5 : 0.0,
    atomicNumber <= 20 ? 0.0 : atomicNumber <= 30 ? 0.5 : 1.0,
    Math.min(1.0, valence * en / 16.0),
    Math.min(1.0, covalentR * dOcc / 100),
  ];
}

function getSymmetryAwareFeatures(spaceGroupName?: string, fracPosition?: [number, number, number]): number[] {
  if (!spaceGroupName) return new Array(12).fill(0);
  const normalized = normalizeSpaceGroup(spaceGroupName);
  const sgNum = resolveSGNumber(normalized) ?? 1;
  return computeThorFeatureVector(sgNum, fracPosition);
}

// ── Global composition feature helpers ───────────────────────────────────────

function computeVEC(counts: Record<string, number>): number {
  let totalAtoms = 0, totalVE = 0;
  for (const [el, n] of Object.entries(counts)) {
    const data = getElementData(el);
    totalVE += n * (data?.valenceElectrons ?? 0);
    totalAtoms += n;
  }
  return totalAtoms > 0 ? totalVE / totalAtoms : 0;
}

function computeAtomicMismatch(counts: Record<string, number>): number {
  const els = Object.entries(counts);
  if (els.length <= 1) return 0;
  const total = els.reduce((s, [, n]) => s + n, 0);
  if (total <= 0) return 0;
  const rBar = els.reduce((s, [el, n]) => s + (n / total) * (getElementData(el)?.atomicRadius ?? 130), 0);
  if (rBar <= 0) return 0;
  const delta2 = els.reduce((s, [el, n]) => {
    const r = getElementData(el)?.atomicRadius ?? 130;
    return s + (n / total) * (1 - r / rBar) ** 2;
  }, 0);
  return Math.sqrt(Math.max(0, delta2));
}

// Computes GLOBAL_COMP_DIM (23) composition-level features injected at the MLP head.
// [0-12]  13 composition features (EN, d-occ, VEC, Debye, mismatch, nEl, stdEN, H-flag, TM frac, entropy, ENdiff, mass, radius-std)
// [13-18]  6 physics features from PhysicsFeatureHints (λ, logω, DOS, FE, isCuprate, isIronBased)
// [19-22]  4 engineered features (pressure, MgB2-type, heavy-fermion, mean TM d-count)
// Hints contain measured/estimated physics — not model outputs — so there is no training leakage.
function computeGlobalCompositionFeatures(counts: Record<string, number>, hints?: PhysicsFeatureHints, pressureGpa?: number): number[] {
  const els = Object.entries(counts);
  if (els.length === 0) return new Array(GLOBAL_COMP_DIM).fill(0);
  const total = els.reduce((s, [, n]) => s + n, 0);
  if (total <= 0) return new Array(GLOBAL_COMP_DIM).fill(0);

  // --- original 6 ---
  const enValues  = els.map(([el, n]) => ({ w: n / total, v: getElementData(el)?.paulingElectronegativity ?? 1.5 }));
  const meanEN    = enValues.reduce((s, { w, v }) => s + w * v, 0);
  const meanD     = els.reduce((s, [el, n]) => s + (n / total) * getDOrbitalOccupancy(getElementData(el)?.atomicNumber ?? 1), 0);
  const vec       = computeVEC(counts);
  const meanDebye = els.reduce((s, [el, n]) => s + (n / total) * (getElementData(el)?.debyeTemperature ?? 300), 0);
  const mismatch  = computeAtomicMismatch(counts);
  const nElements = els.length;

  // --- 7 XGBoost-inspired additions ---
  // [6] std of Pauling EN — high spread → large ionicity gap (cuprate-like or hydride-like)
  const stdEN = Math.sqrt(enValues.reduce((s, { w, v }) => s + w * (v - meanEN) ** 2, 0));

  // [7] binary: formula contains hydrogen (hydride superconductors are a separate class at high P)
  const hasHydrogen = counts["H"] ? 1.0 : 0.0;

  // [8] fraction of stoichiometric weight from transition metals (d-electron BCS coupling density)
  const tmWeight = els.reduce((s, [el, n]) => s + (isTransitionMetal(el) ? n : 0), 0);
  const transitionMetalFraction = tmWeight / total;

  // [9] configurational (mixing) entropy −Σ x_i ln(x_i), normalized by ln(N_max=6)
  // captures HEA / medium-entropy alloy superconductors
  const cfgEntropy = els.reduce((s, [, n]) => {
    const x = n / total;
    return s + (x > 0 ? -x * Math.log(x) : 0);
  }, 0) / Math.log(Math.max(nElements, 2));

  // [10] max pairwise EN difference — ionicity extremes (e.g. Na vs F = 3.16 - 0.93 = 2.23)
  const enList = els.map(([el]) => getElementData(el)?.paulingElectronegativity ?? 1.5);
  const maxENdiff = enList.length > 1
    ? Math.max(...enList) - Math.min(...enList)
    : 0;

  // [11] mean atomic mass — heavier atoms → lower Debye freq → conventional BCS weakened
  const meanMass = els.reduce((s, [el, n]) => s + (n / total) * (getElementData(el)?.atomicMass ?? 50), 0);

  // [12] std of atomic radius — additional lattice strain signal beyond scalar mismatch
  const radii  = els.map(([el, n]) => ({ w: n / total, v: getElementData(el)?.atomicRadius ?? 130 }));
  const meanR  = radii.reduce((s, { w, v }) => s + w * v, 0);
  const stdRadius = Math.sqrt(radii.reduce((s, { w, v }) => s + w * (v - meanR) ** 2, 0));

  // --- 6 physics features from hints (no leakage: these are physics estimates, not model outputs) ---

  // [13] electron-phonon coupling λ — the central BCS scalar
  // Range 0–3: weak coupling ~0.4, strong ~1.5+; normalise by 3.
  const lambda = hints?.electronPhononLambda;
  const lambdaFeat = lambda != null && lambda >= 0 ? Math.min(1.0, lambda / 3.0) : meanD * transitionMetalFraction;

  // [14] log phonon frequency proxy — phonon energy scale; high → higher Tc ceiling
  // Use provided logPhononFreq, or fall back to log(Debye) normalised to ~0–1 over 1–2000 K range.
  const logPhononFreqHint = hints?.logPhononFreq;
  const logPhononFeat = logPhononFreqHint != null
    ? Math.min(1.0, Math.max(0, logPhononFreqHint / 8.0))
    : Math.min(1.0, Math.log(Math.max(1, meanDebye)) / 7.6);

  // [15] density of states at E_F proxy — high DOS → strong coupling per BCS N(0)V
  // Unit: states/eV/atom; typical range 0–5; fallback: d-orbital filling heuristic.
  const dosHint = hints?.dosAtEF;
  const dosFeat = dosHint != null && dosHint >= 0 ? Math.min(1.0, dosHint / 5.0) : Math.min(1.0, meanD * 2.0);

  // [16] formation energy signal — more negative = more thermodynamically stable
  // Maps -3→1, -1→0.67, 0→0.33, +1→0 eV/atom. When unknown (null), use neutral prior 0.5.
  const fe = hints?.formationEnergy;
  const hd = hints?.hullDistance;
  const feFeat = fe != null
    ? Math.max(0, Math.min(1.0, (-fe + 1.0) / 4.0))
    : hd != null
      ? Math.max(0, Math.min(1.0, 1.0 - hd / 0.5))   // hull distance 0→1 (stable), 0.5+→0 (unstable)
      : 0.5;                                           // neutral prior

  // [17] cuprate class flag — Cu + O compound → d-wave unconventional SC regime
  const isCuprate = (counts["Cu"] != null && counts["O"] != null) ? 1.0 : 0.0;

  // [18] iron-based class flag — Fe + pnictogen/chalcogen → sign-changing s± pairing
  const IRON_SC_ANIONS = new Set(["As", "P", "Se", "Te", "S"]);
  const isIronBased = (counts["Fe"] != null && Object.keys(counts).some(el => IRON_SC_ANIONS.has(el))) ? 1.0 : 0.0;

  // --- 4 engineered features ---

  // [19] pressure (log scale) — critical for high-P hydride SCs (H3S@150GPa, LaH10@170GPa)
  // log(1 + P/10) / log(1 + 200) normalises to ~0-1 over 0-2000 GPa
  const pressureFeat = pressureGpa != null && pressureGpa > 0
    ? Math.min(1.0, Math.log1p(pressureGpa / 10) / Math.log1p(200))
    : 0.0;

  // [20] MgB2-type flag — Mg:B ≈ 1:2; two-band phonon SC mechanism; Tc~39K at ambient P
  const mgCount = counts["Mg"] ?? 0;
  const bCount  = counts["B"]  ?? 0;
  const isMgB2Type = (mgCount > 0 && bCount > 0 && Math.abs(bCount / mgCount - 2) < 0.6) ? 1.0 : 0.0;

  // [21] heavy-fermion / f-electron hint — Ce, U, Yb, Pr, Sm → unconventional Kondo-lattice SC
  const HEAVY_FERMION_ELS = new Set(["Ce", "U", "Yb", "Pr", "Sm"]);
  const hfWeight = els.reduce((s, [el, n]) => s + (HEAVY_FERMION_ELS.has(el) ? n / total : 0), 0);
  const heavyFermionFeat = Math.min(1.0, hfWeight * 4.0);  // boosted — even trace amounts are diagnostic

  // [22] mean d-electron count per TM atom — finer-grained than d-orbital filling average
  // More directly related to Stoner criterion and density of states at E_F.
  const tmEls = els.filter(([el]) => isTransitionMetal(el));
  const tmCount = tmEls.reduce((s, [, n]) => s + n, 0);
  const meanTMdCount = tmCount > 0
    ? tmEls.reduce((s, [el, n]) => s + (n / tmCount) * getDOrbitalOccupancy(getElementData(el)?.atomicNumber ?? 1), 0)
    : 0.0;

  return [
    meanEN / 4.0,                           // [0] mean Pauling EN — ionicity proxy
    meanD,                                  // [1] mean d-orbital filling — BCS coupling proxy
    vec / 8.0,                              // [2] valence electron concentration — strongest empirical SC predictor
    Math.min(1.0, meanDebye / 2000),        // [3] mean Debye temperature — phonon frequency proxy
    Math.min(1.0, mismatch * 5),            // [4] atomic size mismatch δ — lattice distortion (suppresses Tc)
    Math.min(1.0, nElements / 5.0),         // [5] compound complexity
    Math.min(1.0, stdEN / 2.0),             // [6] EN heterogeneity — cuprate/hydride indicator
    hasHydrogen,                            // [7] hydrogen flag — hydride SC class separator
    transitionMetalFraction,               // [8] TM fraction — d-electron BCS coupling density
    Math.min(1.0, cfgEntropy),              // [9] configurational entropy — HEA/entropy alloy signal
    Math.min(1.0, maxENdiff / 3.5),         // [10] max EN spread — ionicity extremes
    Math.min(1.0, meanMass / 200),          // [11] mean atomic mass — phonon mass scale
    Math.min(1.0, stdRadius / 80),          // [12] radius std — lattice strain heterogeneity
    lambdaFeat,                             // [13] electron-phonon λ — core BCS coupling constant
    logPhononFeat,                          // [14] log phonon frequency — phonon energy scale
    dosFeat,                                // [15] DOS at E_F — electronic density at Fermi level
    feFeat,                                 // [16] formation energy signal — thermodynamic viability
    isCuprate,                              // [17] cuprate flag — d-wave/unconventional class
    isIronBased,                            // [18] iron-based flag — s± pairing class
    pressureFeat,                           // [19] pressure (log-normalised) — high-P hydride class
    isMgB2Type,                             // [20] MgB2-type flag — two-band phonon SC
    heavyFermionFeat,                       // [21] heavy-fermion/f-electron hint — Kondo-lattice SC
    meanTMdCount,                           // [22] mean d-count per TM — Stoner criterion / DOS proxy
  ];
}

export function buildCrystalGraph(formula: string, structure?: any, pressureGpa?: number, hints?: PhysicsFeatureHints): CrystalGraph {
  const rawCounts = parseFormulaCounts(formula);
  const elements = Object.keys(rawCounts).sort();
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

  // Enrich hints with lambda regressor output if λ not already provided.
  // predictLambda uses fast gradient-boosted trees (~1ms) when an ensemble model exists;
  // falls back to analytical physics engine otherwise. Called once per graph build
  // (graphs are cached in trainEnsemble) so this does not dominate training time.
  let enrichedHints: PhysicsFeatureHints = hints ?? {};
  if (enrichedHints.electronPhononLambda == null) {
    try {
      const lp = predictLambda(formula, pressureGpa ?? 0);
      enrichedHints = {
        ...enrichedHints,
        electronPhononLambda: lp.lambda,
        // dosAtEF and logPhononFreq from the regressor's own feature vector
        dosAtEF:       enrichedHints.dosAtEF       ?? (lp.features['dosAtEF']  as number | undefined) ?? null,
        logPhononFreq: enrichedHints.logPhononFreq ?? (lp.features['debyeTemp'] != null
          ? Math.log(Math.max(1, lp.features['debyeTemp'] as number)) : null),
      };
    } catch { /* leave as null — computeGlobalCompositionFeatures will use proxy fallbacks */ }
  }

  // Normalize ordering for permutation invariance
  edges.sort((a, b) => a.source !== b.source ? a.source - b.source : a.target - b.target);
  for (let i = 0; i < adjacency.length; i++) adjacency[i].sort((a, b) => a - b);
  const edgeIndex = buildEdgeIndex(nodes, edges);
  const globalFeatures = computeGlobalCompositionFeatures(rawCounts, enrichedHints, pressureGpa);
  const partialGraph: CrystalGraph = { nodes, edges, threeBodyFeatures: [], adjacency, edgeIndex, formula, pressureGpa, globalFeatures };
  partialGraph.threeBodyFeatures = compute3BodyFeatures(partialGraph);
  return partialGraph;
}

/**
 * SchNet continuous filter convolution (inference, non-cached).
 * h_ij = W_filter2 @ silu(W_filter1 @ rbf_ij + b_filter1) + b_filter2
 * message = h_ij * h_j (element-wise);  h_i' = h_i + mean_j(message * cutoff)
 */
/**
 * SchNet continuous filter convolution (inference, non-cached).
 * h_ij = W_filter2 @ silu(W_filter1 @ rbf_ij + b_filter1) + b_filter2
 * message = h_ij * h_j (element-wise);  h_i' = h_i + mean_j(message * cutoff)
 */
export function cgcnnConvolutionLayer(
  graph: CrystalGraph,
  W_filter1: number[][],
  b_filter1: number[],
  W_filter2: number[][],
  b_filter2: number[],
  adaptiveLogits?: number[][],
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

    for (let nIdx = 0; nIdx < neighbors.length; nIdx++) {
      const j = neighbors[nIdx];
      const edgeFeat = getEdgeFromIndex(graph.edgeIndex, nNodes, i, j);
      const distance = edgeFeat?.distance ?? 2.5;
      const cw = cosineCutoff(distance);
      if (cw <= 0) continue;

      const edgeVec = edgeFeat?.features ?? initVector(EDGE_DIM);
      const rbf_ij: number[] = new Array(N_GAUSSIAN_BASIS);
      for (let k = 0; k < N_GAUSSIAN_BASIS; k++) rbf_ij[k] = edgeVec[k] ?? 0;

      const z_ij = vecAdd(matVecMul(W_filter1, rbf_ij), b_filter1);
      const h1_ij: number[] = new Array(HIDDEN_DIM);
      for (let k = 0; k < HIDDEN_DIM; k++) h1_ij[k] = silu(z_ij[k] ?? 0);
      const h_ij = vecAdd(matVecMul(W_filter2, h1_ij), b_filter2);

      const adaptLogit = adaptiveLogits?.[i]?.[nIdx] ?? 0;
      if (adaptLogit !== 0) {
        for (let k = 0; k < HIDDEN_DIM; k++) h_ij[k] += adaptLogit;
      }

      const hj = embeddings[j];
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] ?? 0) * (hj[k] ?? 0) * cw;
      }
      totalWeight += cw;
    }

    if (totalWeight > 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) aggUpdate[k] /= totalWeight;
    }

    const updated: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      updated[k] = (embeddings[i][k] ?? 0) + aggUpdate[k];
    }
    newEmbeddings.push(updated);
  }

  for (let i = 0; i < nNodes; i++) graph.nodes[i].embedding = newEmbeddings[i];
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
    const hUpd = fusedUpdate(W_update, combined);
    // Residual + LayerNorm — consistent with training path (attnMessagePassCached).
    const hRes = embeddings[i].map((v, k) => v + (hUpd[k] ?? 0));
    newEmbeddings.push(layerNorm(hRes));
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

function attnMessagePassCached(
  graph: CrystalGraph,
  W_message: number[][], W_update: number[][],
  W_query: number[][], W_key: number[][],
  useLeakyMsg: boolean = false,
): { embeddings: number[][]; cache: AttnLayerCache } {
  const nNodes = graph.nodes.length;
  const inputEmbs = graph.nodes.map(n => [...n.embedding]);
  const preActs: number[][] = [];
  const preNormActs: number[][] = [];
  const attnWts: number[][] = [];
  const neighborLists: number[][] = [];
  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    neighborLists.push([...neighbors]);
    if (neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      preActs.push([...inputEmbs[i]]);
      preNormActs.push([...inputEmbs[i]]);
      attnWts.push([]);
      continue;
    }

    const query = layerNorm(matVecMul(W_query, inputEmbs[i]));
    const attentionScores: number[] = [];
    const messages: number[][] = [];

    for (const j of neighbors) {
      const key = layerNorm(matVecMul(W_key, inputEmbs[j]));
      let score = dotProduct(query, key);
      const edge = getEdgeFromIndex(graph.edgeIndex, nNodes, i, j);
      if (edge) {
        const edgeFeats = edge.features;
        for (let ef = 0; ef < edgeFeats.length && ef < HIDDEN_DIM; ef++) {
          score += (edgeFeats[ef] ?? 0) * (query[ef] ?? 0) * 0.1;
        }
      }
      attentionScores.push(score);
      messages.push(matVecMul(W_message, inputEmbs[j]));
    }

    const aw = softmax(attentionScores);
    attnWts.push(aw);

    const aggMessage = initVector(HIDDEN_DIM);
    for (let n = 0; n < neighbors.length; n++) {
      const w = aw[n];
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += (messages[n][k] ?? 0) * w;
      }
    }

    const combined = [...inputEmbs[i], ...aggMessage];
    const pre = matVecMul(W_update, combined);
    preActs.push([...pre]);
    const hUpd = useLeakyMsg
      ? pre.map(v => v >= 0 ? v : 0.01 * v)
      : pre.map(v => v >= 0 ? v : 0);
    // Residual: h = input + activation(update)
    const hRes = inputEmbs[i].map((v, k) => v + (hUpd[k] ?? 0));
    preNormActs.push(hRes);
    // LayerNorm after residual
    newEmbeddings.push(layerNorm(hRes));
  }

  for (let i = 0; i < nNodes; i++) {
    graph.nodes[i].embedding = newEmbeddings[i];
  }

  return { embeddings: newEmbeddings, cache: { inputEmbs, preActs, preNormActs, attnWts, neighborLists } };
}

/**
 * SchNet continuous filter convolution (cached, training).
 */
/**
 * SchNet continuous filter convolution (cached, training).
 */
function cgcnnConvCached(
  graph: CrystalGraph,
  W_filter1: number[][], b_filter1: number[],
  W_filter2: number[][], b_filter2: number[],
  adaptiveLogits?: number[][],
): { embeddings: number[][]; cache: CGCNNLayerCache } {
  const nNodes = graph.nodes.length;
  const inputEmbs = graph.nodes.map(n => [...n.embedding]);
  const filterPreActs: number[][][] = [];
  const filterH1s: number[][][] = [];
  const filterOuts: number[][][] = [];
  const rbfs: number[][][] = [];
  const preNormActs: number[][] = [];
  const cutoffWts: number[][] = [];
  const totalWeights: number[] = [];
  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    const nodeFilterPreActs: number[][] = [];
    const nodeFilterH1s: number[][] = [];
    const nodeFilterOuts: number[][] = [];
    const nodeRbfs: number[][] = [];
    const nodeCutoffs: number[] = [];

    if (!neighbors || neighbors.length === 0) {
      newEmbeddings.push([...inputEmbs[i]]);
      preNormActs.push([...inputEmbs[i]]);
      filterPreActs.push(nodeFilterPreActs);
      filterH1s.push(nodeFilterH1s);
      filterOuts.push(nodeFilterOuts);
      rbfs.push(nodeRbfs);
      cutoffWts.push(nodeCutoffs);
      totalWeights.push(0);
      continue;
    }

    const aggUpdate = initVector(HIDDEN_DIM);
    let totalWeight = 0;

    for (let nIdx = 0; nIdx < neighbors.length; nIdx++) {
      const j = neighbors[nIdx];
      const edgeFeat = getEdgeFromIndex(graph.edgeIndex, nNodes, i, j);
      const distance = edgeFeat?.distance ?? 2.5;
      const cw = cosineCutoff(distance);
      nodeCutoffs.push(cw);
      if (cw <= 0) {
        nodeFilterPreActs.push([]);
        nodeFilterH1s.push([]);
        nodeFilterOuts.push([]);
        nodeRbfs.push([]);
        continue;
      }

      const edgeVec = edgeFeat?.features ?? initVector(EDGE_DIM);
      const rbf_ij: number[] = new Array(N_GAUSSIAN_BASIS);
      for (let k = 0; k < N_GAUSSIAN_BASIS; k++) rbf_ij[k] = edgeVec[k] ?? 0;

      const z_ij = vecAdd(matVecMul(W_filter1, rbf_ij), b_filter1);
      const h1_ij = z_ij.map(v => { const s = 1 / (1 + Math.exp(-v)); return v * s; });  // silu
      const h_ij = vecAdd(matVecMul(W_filter2, h1_ij), b_filter2);

      const adaptLogit = adaptiveLogits?.[i]?.[nIdx] ?? 0;
      const hj = inputEmbs[j];

      nodeFilterPreActs.push([...z_ij]);
      nodeFilterH1s.push([...h1_ij]);
      nodeFilterOuts.push([...h_ij]);
      nodeRbfs.push([...rbf_ij]);

      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggUpdate[k] += (h_ij[k] + adaptLogit) * (hj[k] ?? 0) * cw;
      }
      totalWeight += cw;
    }

    filterPreActs.push(nodeFilterPreActs);
    filterH1s.push(nodeFilterH1s);
    filterOuts.push(nodeFilterOuts);
    rbfs.push(nodeRbfs);
    cutoffWts.push(nodeCutoffs);
    totalWeights.push(totalWeight);

    if (totalWeight > 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) aggUpdate[k] /= totalWeight;
    }

    const hRes: number[] = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      hRes[k] = (inputEmbs[i][k] ?? 0) + aggUpdate[k];
    }
    preNormActs.push(hRes);
    newEmbeddings.push(hRes);
  }

  for (let i = 0; i < nNodes; i++) graph.nodes[i].embedding = newEmbeddings[i];

  return {
    embeddings: newEmbeddings,
    cache: { inputEmbs, filterPreActs, filterH1s, filterOuts, rbfs, preNormActs, cutoffWts, totalWeights, adaptiveLogits },
  };
}

function attnLayerBackward(
  cache: AttnLayerCache,
  dLdOutput: number[][],
  graph: CrystalGraph,
  W_update: number[][], W_message: number[][],
  useLeaky: boolean,
): {
  dW_update: number[][]; dW_message: number[][]; dLdInput: number[][];
} {
  const nNodes = dLdOutput.length;
  const updateCols = W_update[0]?.length ?? 0;
  const msgCols = W_message[0]?.length ?? 0;
  const dW_update = W_update.map(r => new Array(r.length).fill(0));
  const dW_message = W_message.map(r => new Array(r.length).fill(0));
  const dLdInput: number[][] = Array.from({ length: nNodes }, () => new Array(HIDDEN_DIM).fill(0));

  for (let i = 0; i < nNodes; i++) {
    const neighbors = cache.neighborLists[i];
    if (!neighbors || neighbors.length === 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdOutput[i][k];
      continue;
    }

    // Backprop through layerNorm(h_res): upstream dLdOutput → dLd(h_res)
    const dLdPreNorm = cache.preNormActs?.[i]
      ? layerNormBackward(cache.preNormActs[i], dLdOutput[i])
      : [...dLdOutput[i]];
    // Residual pass-through: h_res = input + hUpd → ∂L/∂input += dLdPreNorm
    for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdPreNorm[k];

    const pre = cache.preActs[i];
    const dPre = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      const mask = useLeaky ? (pre[k] >= 0 ? 1.0 : 0.01) : (pre[k] >= 0 ? 1.0 : 0.0);
      dPre[k] = dLdPreNorm[k] * mask;  // gradient flows through activation from dLdPreNorm
    }

    const emb_i = cache.inputEmbs[i];
    const aggMsg = initVector(HIDDEN_DIM);
    for (let n = 0; n < neighbors.length; n++) {
      const j = neighbors[n];
      const msg = matVecMul(W_message, cache.inputEmbs[j]);
      const w = cache.attnWts[i][n] ?? 0;
      for (let k = 0; k < HIDDEN_DIM; k++) aggMsg[k] += msg[k] * w;
    }
    const combined = [...emb_i, ...aggMsg];

    for (let r = 0; r < HIDDEN_DIM; r++) {
      const dP = dPre[r];
      if (Math.abs(dP) < 1e-12) continue;
      for (let c = 0; c < updateCols; c++) {
        dW_update[r][c] += dP * (combined[c] ?? 0);
      }
    }

    const dCombined = new Array(updateCols).fill(0);
    for (let c = 0; c < updateCols; c++) {
      for (let r = 0; r < HIDDEN_DIM; r++) {
        dCombined[c] += dPre[r] * (W_update[r][c] ?? 0);
      }
    }

    for (let k = 0; k < HIDDEN_DIM; k++) {
      dLdInput[i][k] += dCombined[k];
    }
    const dAggMsg = new Array(HIDDEN_DIM);
    for (let k = 0; k < HIDDEN_DIM; k++) {
      dAggMsg[k] = dCombined[HIDDEN_DIM + k] ?? 0;
    }

    for (let n = 0; n < neighbors.length; n++) {
      const j = neighbors[n];
      const w = cache.attnWts[i][n] ?? 0;
      if (Math.abs(w) < 1e-12) continue;
      const dMsg = new Array(HIDDEN_DIM);
      for (let k = 0; k < HIDDEN_DIM; k++) dMsg[k] = dAggMsg[k] * w;

      for (let r = 0; r < HIDDEN_DIM; r++) {
        if (Math.abs(dMsg[r]) < 1e-12) continue;
        for (let c = 0; c < msgCols; c++) {
          dW_message[r][c] += dMsg[r] * (cache.inputEmbs[j][c] ?? 0);
        }
      }

      for (let c = 0; c < msgCols; c++) {
        let sum = 0;
        for (let r = 0; r < HIDDEN_DIM; r++) sum += dMsg[r] * (W_message[r][c] ?? 0);
        dLdInput[j][c] += sum;
      }
    }
  }

  return { dW_update, dW_message, dLdInput };
}

function cgcnnLayerBackward(
  cache: CGCNNLayerCache,
  dLdOutput: number[][],
  graph: CrystalGraph,
  W_filter1: number[][], W_filter2: number[][],
): {
  dW_filter1: number[][]; dW_filter2: number[][];
  db_filter1: number[]; db_filter2: number[];
  dLdInput: number[][];
  dAdaptiveLogits: number[][];
} {
  const nNodes = dLdOutput.length;
  const dW_filter1 = W_filter1.map(r => new Array(r.length).fill(0));
  const dW_filter2 = W_filter2.map(r => new Array(r.length).fill(0));
  const db_filter1 = new Array(HIDDEN_DIM).fill(0);
  const db_filter2 = new Array(HIDDEN_DIM).fill(0);
  const dLdInput: number[][] = Array.from({ length: nNodes }, () => new Array(HIDDEN_DIM).fill(0));
  const dAdaptiveLogits: number[][] = graph.adjacency.map(ns => new Array(ns.length).fill(0));

  for (let i = 0; i < nNodes; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) dLdInput[i][k] += dLdOutput[i][k];

    const totalW = cache.totalWeights[i];
    if (totalW <= 0) continue;

    const neighbors = graph.adjacency[i];
    if (!neighbors || neighbors.length === 0) continue;

    for (let nIdx = 0; nIdx < neighbors.length; nIdx++) {
      const cw = cache.cutoffWts[i][nIdx] ?? 0;
      if (cw <= 0) continue;

      const z_ij = cache.filterPreActs[i][nIdx];
      const h1_ij = cache.filterH1s[i][nIdx];
      const h_ij = cache.filterOuts[i][nIdx];
      const rbf_ij = cache.rbfs[i][nIdx];
      if (!z_ij || !h1_ij || !h_ij || !rbf_ij) continue;

      const j = neighbors[nIdx];
      const hj = cache.inputEmbs[j];

      const dH_ij: number[] = new Array(HIDDEN_DIM);
      let dAdaptLogitEdge = 0;
      for (let k = 0; k < HIDDEN_DIM; k++) {
        const dMsg_k = (dLdOutput[i][k] / totalW) * cw;
        dH_ij[k] = dMsg_k * (hj[k] ?? 0);
        dLdInput[j][k] += dMsg_k * (h_ij[k] ?? 0);
        dAdaptLogitEdge += dH_ij[k];
      }
      if (dAdaptiveLogits[i]) dAdaptiveLogits[i][nIdx] = dAdaptLogitEdge;

      const dH1_ij: number[] = new Array(HIDDEN_DIM).fill(0);
      for (let r = 0; r < HIDDEN_DIM; r++) {
        db_filter2[r] += dH_ij[r];
        for (let c = 0; c < HIDDEN_DIM; c++) {
          dW_filter2[r][c] += dH_ij[r] * (h1_ij[c] ?? 0);
          dH1_ij[c] += dH_ij[r] * (W_filter2[r]?.[c] ?? 0);
        }
      }

      for (let r = 0; r < HIDDEN_DIM; r++) {
        const z = z_ij[r] ?? 0;
        const sig = 1 / (1 + Math.exp(-z));
        const dZ_r = dH1_ij[r] * sig * (1 + z * (1 - sig));  // silu gradient
        db_filter1[r] += dZ_r;
        for (let c = 0; c < N_GAUSSIAN_BASIS; c++) {
          dW_filter1[r][c] += dZ_r * (rbf_ij[c] ?? 0);
        }
      }
    }
  }

  return { dW_filter1, dW_filter2, db_filter1, db_filter2, dLdInput, dAdaptiveLogits };
}

// ── GLFN-TC: Graph Learning Module ───────────────────────────────────────────
/**
 * Compute per-edge adaptive logits for the Graph Learning Module (GLFN-TC §2.2).
 *
 * For each directed edge i→j:
 *   logit(i,j) = W_elem_feat[atomI]ᵀ · W_graph_adapt · W_elem_feat[atomJ]
 *
 * This scalar is added uniformly to all CGCNN gate dimensions for that edge,
 * effectively learning which element pairs should exchange more information.
 * Initialised near zero (small weights) so the model starts like the baseline.
 *
 * Returns a jagged array [nNodes][nNeighbors] of raw (pre-sigmoid) logits.
 */
function computeAdaptiveLogits(
  graph: CrystalGraph,
  W_elem_feat: number[][],
  W_graph_adapt: number[][]
): number[][] {
  const nNodes = graph.nodes.length;
  const result: number[][] = [];
  for (let i = 0; i < nNodes; i++) {
    const atomI = Math.min(Math.max(graph.nodes[i].atomicNumber, 0), 118);
    const ei = W_elem_feat[atomI] ?? new Array(GRAPH_FEAT_DIM).fill(0);
    const neighbors = graph.adjacency[i] ?? [];
    const nodeLogits: number[] = [];
    for (const j of neighbors) {
      const atomJ = Math.min(Math.max(graph.nodes[j].atomicNumber, 0), 118);
      const ej = W_elem_feat[atomJ] ?? new Array(GRAPH_FEAT_DIM).fill(0);
      // Asymmetric bilinear form: logit = eᵢᵀ W_adapt eⱼ = dot(ei, W_adapt @ ej)
      nodeLogits.push(dotProduct(ei, matVecMul(W_graph_adapt, ej)));
    }
    result.push(nodeLogits);
  }
  return result;
}

export function GNNPredict(graph: CrystalGraph, weights: GNNWeights, dropoutRng?: () => number): GNNPrediction {
  for (let i = 0; i < graph.nodes.length; i++) {
    const raw = graph.nodes[i].embedding;
    const input = raw.length >= NODE_DIM ? raw.slice(0, NODE_DIM) : [...raw, ...new Array(NODE_DIM - raw.length).fill(0)];
    const projected = fusedMatVecAddLeakyRelu(weights.W_input_proj, input, weights.b_input_proj);
    graph.nodes[i].embedding = projected;
  }

  // GLFN-TC Dense Residual: save H0 (post-projection) for skip to layer-2 input.
  const denseH0 = graph.nodes.map(n => [...n.embedding]);
  // GLFN-TC Graph Learning Module: pre-compute adaptive edge logits once.
  const adaptLogits = (weights.W_elem_feat && weights.W_graph_adapt)
    ? computeAdaptiveLogits(graph, weights.W_elem_feat, weights.W_graph_adapt)
    : undefined;

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

  cgcnnConvolutionLayer(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits);
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

  // GLFN-TC Dense Residual (§2.4): add a learned fraction of H0 to H1 before
  // layer-2 attention so later convolutions can access the original node features.
  // dense_skip_gate is a learnable scalar; sigmoid(-2)≈0.12 at init → small skip.
  const denseGate = sigmoid(weights.dense_skip_gate ?? -2);
  if (denseGate > 1e-4) {
    for (let i = 0; i < graph.nodes.length; i++) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        graph.nodes[i].embedding[k] += denseGate * (denseH0[i][k] ?? 0);
      }
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

  if (GNN_MSG_LAYERS >= 3) {
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
  }

  if (GNN_MSG_LAYERS >= 4) {
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

  // Concat global composition features (VEC, mean d-occ, EN, Debye, size mismatch, nEl) before MLP head.
  // These encode compound-level physics the node aggregation cannot recover on its own.
  const compFeats = graph.globalFeatures ?? new Array(GLOBAL_COMP_DIM).fill(0);
  const pooledWithComp = [...pooled, ...compFeats];

  const z1 = vecAdd(matVecMul(weights.W_mlp1, pooledWithComp), weights.b_mlp1);
  const h1 = z1.map(v => v >= 0 ? v : 0.01 * v);
  if (dropoutRng) {
    const dropped = applyDropout(h1, MC_DROPOUT_RATE, dropoutRng);
    for (let i = 0; i < h1.length; i++) h1[i] = dropped[i];
  }
  const latentEmbedding = [...h1];
  const out = vecAdd(matVecMul(weights.W_mlp2, h1), weights.b_mlp2);
  // v15: P(SC) from dedicated classification head — independent of Tc regression h1.
  const zClsInf = vecAdd(matVecMul(weights.W_cls1, pooledWithComp), weights.b_cls1);
  const hClsInf = zClsInf.map(v => v >= 0 ? v : 0.01 * v);
  const out7Cls = dotProduct(weights.W_cls2, hClsInf) + (weights.b_cls2 ?? 0);

  const logVarOut = vecAdd(matVecMul(weights.W_mlp2_var, h1), weights.b_mlp2_var);
  const feVarNorm = softplus(logVarOut[0] ?? 0);
  const tcVarNorm = softplus(logVarOut[2] ?? 0);
  const lambdaVarNorm = softplus(logVarOut[4] ?? 0);
  const bgVarNorm = softplus(logVarOut[5] ?? 0);

  const sf = (v: number, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const formationEnergy = sf(out[0] ?? 0);
  const phononStabilityRaw = sigmoid(sf(out[1] ?? 0));
  // out[2] → ω_log (K): sigmoid-based soft map into (10, OMEGA_LOG_MAX).
  const omegaLogRaw = sf(out[2] ?? 0);
  const omegaLog = 10 + (OMEGA_LOG_MAX - 10) * sigmoid(omegaLogRaw / 3);
  // out[4] → λ via sigmoid soft cap into (0, LAMBDA_MAX).
  const lambdaRaw = LAMBDA_MAX * sigmoid(sf(out[4] ?? 0));
  // v15: P(SC) from dedicated classification head; out[8] → Tc magnitude.
  // Direct regression: out[8] ≈ log1p(Tc/10)/TC_LOG_SCALE ∈ [0,1]. No sigmoid — prevents
  // vanishing gradient trap (sigmoid'→0 as out[8]→-∞ made predictions lock at 0K forever).
  const scProb = sigmoid(sf(out7Cls));
  const predictedTcRaw = 10 * Math.expm1(Math.max(0, sf(out[8] ?? 0)) * TC_LOG_SCALE);
  const confidenceRaw = sigmoid(sf(out[3] ?? 0));
  const bandgapRaw = sigmoid(sf(out[5] ?? 0)) * 5.0;
  const dosProxyRaw = softplus(sf(out[6] ?? 0));
  const stabilityProbRaw = scProb;
  const safeLatent = latentEmbedding.map(v => Number.isFinite(v) ? v : 0);

  return {
    formationEnergy: Math.round(formationEnergy * 1000) / 1000,
    phononStability: phononStabilityRaw > 0.5,
    predictedTc: Math.round(Math.max(0, Math.min(TC_MAX_K, predictedTcRaw)) * 10) / 10,
    omegaLog: Math.round(omegaLog * 10) / 10,
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
  const nNodes = graph.nodes.length;
  const inputProjInputs: number[][] = [];
  const inputProjPreActs: number[][] = [];
  for (let i = 0; i < nNodes; i++) {
    const raw = graph.nodes[i].embedding;
    const input = raw.length >= NODE_DIM ? raw.slice(0, NODE_DIM) : [...raw, ...new Array(NODE_DIM - raw.length).fill(0)];
    inputProjInputs.push([...input]);
    const pre = vecAdd(matVecMul(weights.W_input_proj, input), weights.b_input_proj);
    inputProjPreActs.push([...pre]);
    graph.nodes[i].embedding = pre.map(v => v >= 0 ? v : 0.01 * v);
  }

  const saveResidual = (nodes: CrystalGraph["nodes"]) =>
    nodes.map(n => [...n.embedding]);

  const gates = weights.residual_gates;
  const attnCaches: AttnLayerCache[] = [];

  // GLFN-TC Dense Residual: save H0 after input projection for dense skip.
  const denseH0 = graph.nodes.map(n => [...n.embedding]);
  // GLFN-TC Graph Learning Module: pre-compute adaptive edge logits once per graph.
  const adaptLogits = (weights.W_elem_feat && weights.W_graph_adapt)
    ? computeAdaptiveLogits(graph, weights.W_elem_feat, weights.W_graph_adapt)
    : undefined;

  const residual0 = saveResidual(graph.nodes);
  const { cache: ac0 } = attnMessagePassCached(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key, true);
  attnCaches.push(ac0);

  const { cache: cgcnnC } = cgcnnConvCached(graph, weights.W_filter1, weights.b_filter1, weights.W_filter2, weights.b_filter2, adaptLogits);

  if (graph.threeBodyFeatures.length > 0) {
    threeBodyInteractionLayer(graph, weights.W_3body, weights.W_3body_update);
  }
  const g0 = sigmoid(gates[0] ?? 0);
  for (let i = 0; i < nNodes; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual0[i][k] ?? 0) * g0;
    }
  }

  // GLFN-TC Dense Residual (§2.4): inject H0 into layer-2 input.
  // residual1 is saved BEFORE the skip so the gated-residual path stays clean (gate1 → H1).
  const denseGate = sigmoid(weights.dense_skip_gate ?? -2);
  if (denseGate > 1e-4) {
    for (let i = 0; i < nNodes; i++) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        graph.nodes[i].embedding[k] += denseGate * (denseH0[i][k] ?? 0);
      }
    }
  }

  const residual1 = saveResidual(graph.nodes);
  const { cache: ac1 } = attnMessagePassCached(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2);
  attnCaches.push(ac1);
  const g1 = sigmoid(gates[1] ?? 0);
  for (let i = 0; i < nNodes; i++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual1[i][k] ?? 0) * g1;
    }
  }

  if (GNN_MSG_LAYERS >= 3) {
    const residual2 = saveResidual(graph.nodes);
    const { cache: ac2 } = attnMessagePassCached(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3);
    attnCaches.push(ac2);
    const g2 = sigmoid(gates[2] ?? 0);
    for (let i = 0; i < nNodes; i++) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual2[i][k] ?? 0) * g2;
      }
    }
  }

  if (GNN_MSG_LAYERS >= 4) {
    const residual3 = saveResidual(graph.nodes);
    const { cache: ac3 } = attnMessagePassCached(graph, weights.W_message4, weights.W_update4, weights.W_attn_query4, weights.W_attn_key4);
    attnCaches.push(ac3);
    const g3 = sigmoid(gates[3] ?? 0);
    for (let i = 0; i < nNodes; i++) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        graph.nodes[i].embedding[k] = (graph.nodes[i].embedding[k] ?? 0) + (residual3[i][k] ?? 0) * g3;
      }
    }
  }

  const meanPool = initVector(HIDDEN_DIM);
  const maxPool = new Array(HIDDEN_DIM).fill(-Infinity);
  const maxPoolArgmax = new Array(HIDDEN_DIM).fill(0);
  let totalMultiplicity = 0;
  for (const node of graph.nodes) totalMultiplicity += (node.multiplicity ?? 1);
  for (let ni = 0; ni < nNodes; ni++) {
    const node = graph.nodes[ni];
    const w = (node.multiplicity ?? 1) / totalMultiplicity;
    for (let k = 0; k < HIDDEN_DIM; k++) {
      meanPool[k] += (node.embedding[k] ?? 0) * w;
      if ((node.embedding[k] ?? 0) > maxPool[k]) {
        maxPool[k] = node.embedding[k] ?? 0;
        maxPoolArgmax[k] = ni;
      }
    }
  }

  const scores: number[] = [];
  for (const node of graph.nodes) {
    const attnVec = matVecMul(weights.W_attn_pool, node.embedding);
    scores.push(attnVec.reduce((s, v) => s + v, 0) + Math.log(node.multiplicity ?? 1));
  }
  const attnPoolWeights = softmax(scores);
  const attnPool = initVector(HIDDEN_DIM);
  for (let n = 0; n < nNodes; n++) {
    for (let k = 0; k < HIDDEN_DIM; k++) {
      attnPool[k] += (graph.nodes[n].embedding[k] ?? 0) * attnPoolWeights[n];
    }
  }

  const pooled = new Array(HIDDEN_DIM * 2);
  for (let k = 0; k < HIDDEN_DIM; k++) {
    pooled[k] = (meanPool[k] + attnPool[k]) * 0.5;
    pooled[HIDDEN_DIM + k] = (maxPool[k] === -Infinity ? 0 : maxPool[k]);
  }
  const pressureNorm = (graph.pressureGpa ?? 0) / 300;
  for (let k = 0; k < HIDDEN_DIM; k++) {
    pooled[k] += pressureNorm * (weights.W_pressure[k] ?? 0);
  }

  // Concat global composition features so W_mlp1 can learn their contribution.
  // Stored in cache.pooled so the backward pass correctly computes dW_mlp1.
  const compFeats = graph.globalFeatures ?? new Array(GLOBAL_COMP_DIM).fill(0);
  const pooledWithComp = [...pooled, ...compFeats];

  const z1 = vecAdd(matVecMul(weights.W_mlp1, pooledWithComp), weights.b_mlp1);
  const h1 = z1.map(v => v >= 0 ? v : 0.01 * v);
  const latentEmbedding = [...h1];
  const out = vecAdd(matVecMul(weights.W_mlp2, h1), weights.b_mlp2);
  // v15: Dedicated classification head — P(SC) gets its own pathway.
  const zCls = vecAdd(matVecMul(weights.W_cls1, pooledWithComp), weights.b_cls1);
  const hCls = zCls.map(v => v >= 0 ? v : 0.01 * v);
  const out7Cls = dotProduct(weights.W_cls2, hCls) + (weights.b_cls2 ?? 0);
  // Override out[7] with cls head logit so backward pass reads it via cache.outRaw[7].
  const outRawWithCls = [...out];
  outRawWithCls[7] = out7Cls;

  const logVarOut = vecAdd(matVecMul(weights.W_mlp2_var, h1), weights.b_mlp2_var);

  const feVarNorm = softplus(logVarOut[0] ?? 0);
  const tcVarNorm = softplus(logVarOut[2] ?? 0);
  const lambdaVarNorm = softplus(logVarOut[4] ?? 0);
  const bgVarNorm = softplus(logVarOut[5] ?? 0);

  const sf = (v: number, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const formationEnergy = sf(out[0] ?? 0);
  const phononStabilityRaw = sigmoid(sf(out[1] ?? 0));
  // out[2] → ω_log (K): sigmoid soft map into (10, OMEGA_LOG_MAX) — gradient everywhere.
  const omegaLogRaw = sf(out[2] ?? 0);
  const omegaLog = 10 + (OMEGA_LOG_MAX - 10) * sigmoid(omegaLogRaw / 3);
  // out[4] → λ: sigmoid soft map into (0, LAMBDA_MAX) — gradient everywhere.
  const lambdaRaw = LAMBDA_MAX * sigmoid(sf(out[4] ?? 0));
  // v15: P(SC) from dedicated cls head; out[8] → Tc magnitude.
  // Direct regression (no sigmoid): out[8] ≈ log1p(Tc/10)/TC_LOG_SCALE ∈ [0,1].
  const scProb = sigmoid(sf(out7Cls));
  const predictedTcRaw = 10 * Math.expm1(Math.max(0, sf(out[8] ?? 0)) * TC_LOG_SCALE);
  const confidenceRaw = sigmoid(sf(out[3] ?? 0));
  const bandgapRaw = sigmoid(sf(out[5] ?? 0)) * 5.0;
  const dosProxyRaw = softplus(sf(out[6] ?? 0));
  const stabilityProbRaw = scProb;
  const safeLatent = latentEmbedding.map(v => Number.isFinite(v) ? v : 0);

  const nodeEmbeddings = graph.nodes.map(n => [...n.embedding]);
  const nodeMultiplicities = graph.nodes.map(n => n.multiplicity ?? 1);

  const pred: GNNPrediction = {
    formationEnergy: Math.round(formationEnergy * 1000) / 1000,
    phononStability: phononStabilityRaw > 0.5,
    predictedTc: Math.round(Math.max(0, Math.min(TC_MAX_K, predictedTcRaw)) * 10) / 10,
    omegaLog: Math.round(omegaLog * 10) / 10,
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
    pooled: [...pooledWithComp], // includes comp features so gradW1 is computed correctly
    z1: [...z1],
    h1: [...h1],
    outRaw: outRawWithCls,  // out[7] = cls head logit; others from W_mlp2
    logVarOutRaw: [...logVarOut],
    nodeEmbeddings,
    nodeMultiplicities,
    totalMultiplicity,
    attnCaches,
    cgcnnCache: cgcnnC,
    inputProjInputs,
    inputProjPreActs,
    maxPoolArgmax,
    attnPoolWeights,
    denseH0,  // GLFN-TC: saved for dense-skip gradient in backward pass
    zCls: [...zCls],
    hCls: [...hCls],
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
    W_filter1: initMatrix(HIDDEN_DIM, N_GAUSSIAN_BASIS, rng, Math.sqrt(2.0 / N_GAUSSIAN_BASIS)),
    b_filter1: initVector(HIDDEN_DIM),
    W_filter2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(2.0 / HIDDEN_DIM)),
    b_filter2: initVector(HIDDEN_DIM),
    W_input_proj: initMatrix(HIDDEN_DIM, NODE_DIM, rng, Math.sqrt(2.0 / NODE_DIM)),
    b_input_proj: initVector(HIDDEN_DIM),
    W_3body: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(2.0 / HIDDEN_DIM) * 1.5),
    W_3body_update: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng),
    W_attn_pool: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng),
    residual_gates: [0.5, 0.5, 0.5, 0.5],
    W_pressure: Array.from({ length: HIDDEN_DIM }, () => (rng() - 0.5) * 2 * Math.sqrt(2.0 / HIDDEN_DIM)),
    W_mlp1: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2 + GLOBAL_COMP_DIM, rng),
    b_mlp1: initVector(HIDDEN_DIM),
    // GLFN-TC: Graph Learning Module weights — initialised very small so
    // adaptive logits start near 0 and the model begins identical to baseline.
    W_elem_feat: initMatrix(119, GRAPH_FEAT_DIM, rng, 0.01),
    W_graph_adapt: initMatrix(GRAPH_FEAT_DIM, GRAPH_FEAT_DIM, rng, 0.01),
    // GLFN-TC: Dense Residual scalar gate — sigmoid(-2) ≈ 0.12 (tiny initial skip).
    dense_skip_gate: -2.0,
    // Scale 0.05 (was 0.01) so gradients propagate meaningfully to the message-passing layers.
    // With 0.01: dLdH1 ≈ 0.01 × grad_out = ~0.0004 — effectively zero at graph layers.
    // With 0.05: dLdH1 ≈ 0.05 × grad_out = ~0.002 — 5× better, sufficient for Adam to update.
    // v15: Dedicated classification head — He init for W_cls1, small for W_cls2.
    W_cls1: initMatrix(CLS_DIM, HIDDEN_DIM * 2 + GLOBAL_COMP_DIM, rng, Math.sqrt(2.0 / (HIDDEN_DIM * 2 + GLOBAL_COMP_DIM))),
    b_cls1: initVector(CLS_DIM),
    W_cls2: Array.from({ length: CLS_DIM }, () => (rng() - 0.5) * 0.1),
    b_cls2: 0.0,
    W_mlp2: initMatrix(OUTPUT_DIM, HIDDEN_DIM, rng, 0.05),
    b_mlp2: (() => {
      const b = initVector(OUTPUT_DIM);
      // out[2] → ω_log via sigmoid(x/3): bias -4.0 → sigmoid(-4/3)≈0.21 → ω_log≈320K (physical BCS range)
      // Without this, sigmoid(0/3)=0.5 → ω_log=755K, giving Allen-Dynes Tc>>100K before any training.
      b[2] = -4.0;
      // out[4] → λ via sigmoid(x): bias -1.5 → sigmoid(-1.5)≈0.18 → λ≈1.0 (typical phonon-mediated SC)
      // Without this, sigmoid(0)=0.5 → λ=2.75, immediately predicting Tc≈90K for all materials.
      b[4] = -1.5;
      // out[8] → log1p-normalised Tc (direct regression, no sigmoid).
      // Bias 0.38 ≈ log1p(27/10)/TC_LOG_SCALE → ~27K initial prediction,
      // near the SC training mean. Avoids starting at 0K where gradients would
      // push correctly but initial loss is maximally bad.
      b[8] = 0.38;
      return b;
    })(),
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
    W_filter1: w.W_filter1.map(r => [...r]),
    b_filter1: [...w.b_filter1],
    W_filter2: w.W_filter2.map(r => [...r]),
    b_filter2: [...w.b_filter2],
    W_input_proj: w.W_input_proj.map(r => [...r]),
    b_input_proj: [...w.b_input_proj],
    residual_gates: [...w.residual_gates],
    W_3body: w.W_3body.map(r => [...r]),
    W_3body_update: w.W_3body_update.map(r => [...r]),
    W_attn_pool: w.W_attn_pool.map(r => [...r]),
    W_pressure: [...w.W_pressure],
    W_mlp1: w.W_mlp1.map(r => [...r]),
    b_mlp1: [...w.b_mlp1],
    W_elem_feat: w.W_elem_feat.map(r => [...r]),
    W_graph_adapt: w.W_graph_adapt.map(r => [...r]),
    dense_skip_gate: w.dense_skip_gate,
    W_cls1: w.W_cls1.map(r => [...r]),
    b_cls1: [...w.b_cls1],
    W_cls2: [...w.W_cls2],
    b_cls2: w.b_cls2,
    W_mlp2: w.W_mlp2.map(r => [...r]),
    b_mlp2: [...w.b_mlp2],
    W_mlp2_var: w.W_mlp2_var.map(r => [...r]),
    b_mlp2_var: [...w.b_mlp2_var],
    trainedAt: w.trainedAt,
    nSamples: w.nSamples,
  };
}

/**
 * GLFN-TC migration: add new weight fields to models trained before the
 * Graph Learning + Dense Residual modules were added. Old weights load fine —
 * new fields start near zero so the model behaves identically to baseline on
 * the first forward pass, then learns to use the new parameters during training.
 */
function migrateWeights(w: GNNWeights, rng: () => number): GNNWeights {
  // SchNet migration: old checkpoints had W_conv_gate/W_conv_value.
  const wAny = w as any;
  if (!wAny.W_filter1 || wAny.W_filter1.length !== HIDDEN_DIM ||
      wAny.W_filter1[0]?.length !== N_GAUSSIAN_BASIS) {
    w.W_filter1 = initMatrix(HIDDEN_DIM, N_GAUSSIAN_BASIS, rng, Math.sqrt(2.0 / N_GAUSSIAN_BASIS));
    w.b_filter1 = initVector(HIDDEN_DIM);
    w.W_filter2 = initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(2.0 / HIDDEN_DIM));
    w.b_filter2 = initVector(HIDDEN_DIM);
  }
  if (!w.b_filter1 || w.b_filter1.length !== HIDDEN_DIM) w.b_filter1 = initVector(HIDDEN_DIM);
  if (!w.W_filter2 || w.W_filter2.length !== HIDDEN_DIM ||
      w.W_filter2[0]?.length !== HIDDEN_DIM) {
    w.W_filter2 = initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, Math.sqrt(2.0 / HIDDEN_DIM));
  }
  if (!w.b_filter2 || w.b_filter2.length !== HIDDEN_DIM) w.b_filter2 = initVector(HIDDEN_DIM);
  if (!w.W_elem_feat || w.W_elem_feat.length !== 119) {
    w.W_elem_feat = initMatrix(119, GRAPH_FEAT_DIM, rng, 0.01);
  }
  if (!w.W_graph_adapt || w.W_graph_adapt.length !== GRAPH_FEAT_DIM) {
    w.W_graph_adapt = initMatrix(GRAPH_FEAT_DIM, GRAPH_FEAT_DIM, rng, 0.01);
  }
  if (typeof w.dense_skip_gate !== 'number' || !Number.isFinite(w.dense_skip_gate)) {
    w.dense_skip_gate = -2.0;
  }
  // v15: Dedicated classification head — initialize if absent; fix width if GLOBAL_COMP_DIM changed.
  const expectedClsCols = HIDDEN_DIM * 2 + GLOBAL_COMP_DIM;
  if (!w.W_cls1 || w.W_cls1.length !== CLS_DIM) {
    w.W_cls1 = initMatrix(CLS_DIM, expectedClsCols, rng, Math.sqrt(2.0 / expectedClsCols));
  } else if (w.W_cls1[0]?.length < expectedClsCols) {
    const padCls = expectedClsCols - w.W_cls1[0].length;
    for (const row of w.W_cls1) {
      for (let i = 0; i < padCls; i++) row.push(0);
    }
    invalidateFlatCache(w.W_cls1);
  } else if (w.W_cls1[0]?.length > expectedClsCols) {
    for (let r = 0; r < w.W_cls1.length; r++) {
      w.W_cls1[r] = w.W_cls1[r].slice(0, expectedClsCols);
    }
    invalidateFlatCache(w.W_cls1);
  }
  if (!w.b_cls1 || w.b_cls1.length !== CLS_DIM) {
    w.b_cls1 = initVector(CLS_DIM);
  }
  if (!w.W_cls2 || w.W_cls2.length !== CLS_DIM) {
    w.W_cls2 = Array.from({ length: CLS_DIM }, () => (rng() - 0.5) * 0.1);
  }
  if (typeof w.b_cls2 !== 'number' || !Number.isFinite(w.b_cls2)) {
    w.b_cls2 = 0.0;
  }
  // If W_mlp1 was saved with a different GLOBAL_COMP_DIM, fix the column count:
  // - Too narrow (old checkpoint): zero-pad new columns so they contribute 0 until trained.
  // - Too wide (checkpoint from an experiment with larger GLOBAL_COMP_DIM): truncate extra cols.
  const expectedCols = HIDDEN_DIM * 2 + GLOBAL_COMP_DIM;
  if (w.W_mlp1 && w.W_mlp1[0]) {
    if (w.W_mlp1[0].length < expectedCols) {
      const pad = expectedCols - w.W_mlp1[0].length;
      for (const row of w.W_mlp1) {
        for (let i = 0; i < pad; i++) row.push(0);
      }
      invalidateFlatCache(w.W_mlp1);
    } else if (w.W_mlp1[0].length > expectedCols) {
      for (let r = 0; r < w.W_mlp1.length; r++) {
        w.W_mlp1[r] = w.W_mlp1[r].slice(0, expectedCols);
      }
      invalidateFlatCache(w.W_mlp1);
    }
  }
  return w;
}

export interface TrainingSample {
  formula: string;
  tc: number;
  formationEnergy?: number;
  structure?: any;
  prototype?: string;
  pressureGpa?: number;
  lambda?: number;
  /** ω_log in Kelvin from DFPT (JARVIS wlog_K or QE omega_log×1.4388). Used as direct supervision
   *  target for the GNN's out[2] channel and for material-specific Allen-Dynes Tc. */
  omegaLog?: number;
  /** Coulomb pseudopotential μ* from DFPT (QE mu_star). When present, replaces FIXED_MU_STAR
   *  in the training Allen-Dynes calculation so the model learns correct (λ, ω_log) → Tc mappings
   *  for materials with non-standard Coulomb screening (cuprates ~0.15, hydrides ~0.08). */
  muStar?: number;
  /** "dft-verified" when this Tc came from a real DFPT run; absent for ML estimates */
  dataConfidence?: string;
  /** DFPT-derived Tc (Allen-Dynes from QE λ) — overrides `tc` as primary training target when present */
  qeDFPTTc?: number;
  /** Band gap in eV from DFT/MP — used as direct supervision for out[5]; 0 for metals */
  bandgap?: number;
  /** Phonon stability from DFPT — true if all phonon modes positive; overrides tc>0 heuristic */
  phononStable?: boolean;
  /** P(SC) from XGBoost binary classifier trained on the same dataset as the GNN.
   *  Used as soft label for out[7] BCE loss so the GNN starts with an informed
   *  SC/non-SC prior rather than learning the classification from scratch. */
  xgbScProb?: number;
  /** Source dataset tag — passed into PhysicsFeatureHints so the GNN's global feature vector
   *  includes a source one-hot (features [26-31]) and data-availability mask ([23-25]).
   *  Values: 'qe-dft' | 'hamidieh' | 'jarvis-sc' | '3dsc-mp' | 'contrast-jarvis' | 'contrast-mp' */
  sourceTag?: string;
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

/** Renders a compact ASCII progress bar, e.g. `[████████░░░░░░░░░░░░] 40%`. */
function _gnnBar(epoch: number, total: number, width = 22): string {
  const filled = Math.round(width * (epoch + 1) / total);
  return '[' + '█'.repeat(filled) + '░'.repeat(width - filled) + ']';
}

export async function trainGNNSurrogate(trainingData: TrainingSample[], preInitWeights?: GNNWeights, maxPretrainEpochs = 15, label?: string): Promise<GNNWeights> {
  const rng = seededRandom(42);
  const weights = migrateWeights(preInitWeights ?? initWeights(rng), rng);

  if (trainingData.length < 5) {
    weights.trainedAt = Date.now();
    weights.nSamples = trainingData.length;
    return weights;
  }

  const LR_INIT = 0.001;
  // Scale epochs so all dataset sizes get enough gradient steps.
  // N=6500→30, N=1200→50, N=500→100, N=100→150 (cap). Direct Tc path needs more steps
  // than Allen-Dynes alone since it learns from scratch without physics priors.
  const epochs = Math.max(30, Math.min(150, Math.ceil(60000 / trainingData.length)));
  const batchSize = Math.min(64, trainingData.length);

  // Adam state for MLP head weights and graph layer weights.
  const adamBeta1 = 0.9, adamBeta2 = 0.999, adamEps = 1e-8;
  let adamStep = 0;
  const mkAdam = (rows: number, cols: number) => ({
    m: Array.from({ length: rows }, () => new Array(cols).fill(0)),
    v: Array.from({ length: rows }, () => new Array(cols).fill(0)),
  });
  const mkAdamVec = (n: number) => ({ m: new Array(n).fill(0), v: new Array(n).fill(0) });
  const adamW1  = mkAdam(HIDDEN_DIM, HIDDEN_DIM * 2 + GLOBAL_COMP_DIM);
  const adamB1  = mkAdamVec(HIDDEN_DIM);
  const adamW2  = mkAdam(OUTPUT_DIM, HIDDEN_DIM);
  const adamB2  = mkAdamVec(OUTPUT_DIM);
  const adamW2v = mkAdam(OUTPUT_DIM, HIDDEN_DIM);
  const adamB2v = mkAdamVec(OUTPUT_DIM);
  // v15: Adam state for dedicated classification head.
  const adamWcls1 = mkAdam(CLS_DIM, HIDDEN_DIM * 2 + GLOBAL_COMP_DIM);
  const adamBcls1 = mkAdamVec(CLS_DIM);
  const adamWcls2 = mkAdamVec(CLS_DIM);
  const adamBcls2 = mkAdamVec(1);
  // Adam state for graph layer weights — replaces plain SGD (lr*0.3) for better
  // convergence on the dedicated GCP GNN instance.  Each matrix entry gets its
  // own adaptive moment estimates, yielding faster convergence and less sensitivity
  // to the 3× lower graph LR relative to the MLP head.
  const mkAdamMatG = (mat: number[][]) => ({
    m: mat.map(r => new Array(r.length).fill(0)),
    v: mat.map(r => new Array(r.length).fill(0)),
  });
  const mkAdamVecG = (n: number) => ({ m: new Array(n).fill(0), v: new Array(n).fill(0) });
  const graphAdam = {
    msg:       [weights.W_message, weights.W_message2, weights.W_message3, weights.W_message4].map(mkAdamMatG),
    upd:       [weights.W_update,  weights.W_update2,  weights.W_update3,  weights.W_update4 ].map(mkAdamMatG),
    filter1:   mkAdamMatG(weights.W_filter1),
    filter2:   mkAdamMatG(weights.W_filter2),
    inputProj: mkAdamMatG(weights.W_input_proj),
    bFilter1:   mkAdamVecG(HIDDEN_DIM),
    bFilter2:   mkAdamVecG(HIDDEN_DIM),
    bInputProj: mkAdamVecG(HIDDEN_DIM),
    pressure:   mkAdamVecG(HIDDEN_DIM),
    gates:      mkAdamVecG(weights.residual_gates.length),
    // GLFN-TC: Adam state for Graph Learning Module and Dense Residual gate.
    elemFeat:       mkAdamMatG(weights.W_elem_feat),
    graphAdapt:     mkAdamMatG(weights.W_graph_adapt),
    denseSkipGate:  mkAdamVecG(1),
  };

  const graphCache = new Map<string, CrystalGraph>();
  const origEmbeddings = new Map<string, number[][]>();
  for (let si = 0; si < trainingData.length; si++) {
    const sample = trainingData[si];
    const key = graphCacheKey(sample.formula, sample.prototype, sample.structure);
    if (!graphCache.has(key)) {
      // Pass known physics from the training sample as hints so the GNN sees
      // the same physics features at train time that it will derive at inference.
      // Also pass source provenance and data-availability flags so features [23-31]
      // correctly reflect which fields are measured vs estimated for this sample.
      const sampleHints: PhysicsFeatureHints = {
        electronPhononLambda: sample.lambda ?? null,
        formationEnergy:      sample.formationEnergy ?? null,
        sourceTag:            sample.sourceTag,
        hasLambdaMeasured:    sample.lambda != null,
        hasOmegaLogMeasured:  sample.omegaLog != null,
        hasFEMeasured:        sample.formationEnergy != null,
      };
      const g = sample.prototype
        ? buildPrototypeGraph(sample.formula, sample.prototype, sample.pressureGpa, sampleHints)
        : buildCrystalGraph(sample.formula, sample.structure, sample.pressureGpa, sampleHints);
      graphCache.set(key, g);
      origEmbeddings.set(key, g.nodes.map(n => [...n.embedding]));
    }
  }

  // Pre-bucket indices by Tc for stratified sampling.
  // high-Tc (≥40K) are the discovery targets — guarantee they appear in every batch.
  const hiTcBucket  = Array.from({ length: trainingData.length }, (_, i) => i).filter(i => trainingData[i].tc >= 40);
  const loTcBucket  = Array.from({ length: trainingData.length }, (_, i) => i).filter(i => trainingData[i].tc > 0 && trainingData[i].tc < 40);
  const nonScBucket = Array.from({ length: trainingData.length }, (_, i) => i).filter(i => trainingData[i].tc === 0);

  const indices = Array.from({ length: trainingData.length }, (_, i) => i);

  // ── Formation energy pretraining ─────────────────────────────────────────────
  // Run a warm-up phase on FE-only loss before Tc training. The message-passing
  // layers learn chemical bonding geometry from the much larger pool of MP data
  // (every sample with a formation energy). This gives better node representations
  // before the harder Tc regression task begins — analogous to ImageNet pretraining.
  const feSamples = trainingData.map((s, i) => i).filter(i => trainingData[i].formationEnergy != null);
  const PRETRAIN_EPOCHS = feSamples.length >= 20 ? maxPretrainEpochs : 0;
  if (PRETRAIN_EPOCHS > 0) {
    console.log(`[GNN] FE pretraining: ${PRETRAIN_EPOCHS} epochs on ${feSamples.length} samples with formation energy`);
    for (let preEpoch = 0; preEpoch < PRETRAIN_EPOCHS; preEpoch++) {
      const preLr = LR_INIT * (0.5 + 0.5 * Math.cos(Math.PI * preEpoch / PRETRAIN_EPOCHS));
      // Shuffle FE samples
      for (let i = feSamples.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [feSamples[i], feSamples[j]] = [feSamples[j], feSamples[i]];
      }
      const prePooledLen = HIDDEN_DIM * 2 + GLOBAL_COMP_DIM;
      for (let bStart = 0; bStart < feSamples.length; bStart += batchSize) {
        const bEnd = Math.min(bStart + batchSize, feSamples.length);
        const preGradW1 = weights.W_mlp1.map(r => new Array(r.length).fill(0));
        const preGradB1 = new Array(weights.b_mlp1.length).fill(0);
        const preGradW2 = weights.W_mlp2.map(r => new Array(r.length).fill(0));
        const preGradB2 = new Array(weights.b_mlp2.length).fill(0);
        const clipG = (g: number) => { const v = Number.isFinite(g) ? g : 0; return Math.max(-1, Math.min(1, v)); };
        for (let b = bStart; b < bEnd; b++) {
          const si = feSamples[b];
          const s = trainingData[si];
          const cKey = graphCacheKey(s.formula, s.prototype, s.structure);
          const g = graphCache.get(cKey)!;
          const orig = origEmbeddings.get(cKey)!;
          for (let ni = 0; ni < g.nodes.length; ni++) g.nodes[ni].embedding = [...orig[ni]];
          const { pred: prePred, cache: preCache } = GNNPredictForTraining(g, weights);
          const feErr = prePred.formationEnergy - s.formationEnergy!;
          // Only FE loss — zero gradient on all other outputs.
          const preDLdOut = new Array(OUTPUT_DIM).fill(0);
          preDLdOut[0] = clipG(2 * feErr);  // weight 1.0 during pretraining
          for (let i = 0; i < weights.W_mlp2.length; i++) {
            for (let j = 0; j < weights.W_mlp2[i].length; j++) preGradW2[i][j] += preDLdOut[i] * preCache.h1[j];
            preGradB2[i] += preDLdOut[i];
          }
          const preDLdH1 = new Array(HIDDEN_DIM).fill(0);
          for (let j = 0; j < HIDDEN_DIM; j++)
            for (let i = 0; i < OUTPUT_DIM; i++) preDLdH1[j] += preDLdOut[i] * (weights.W_mlp2[i]?.[j] ?? 0);
          const preDLdZ1 = preDLdH1.map((v, j) => v * (preCache.z1[j] >= 0 ? 1.0 : 0.01));
          for (let i = 0; i < HIDDEN_DIM; i++) {
            for (let j = 0; j < prePooledLen; j++) preGradW1[i][j] += clipG(preDLdZ1[i] * preCache.pooled[j]);
            preGradB1[i] += clipG(preDLdZ1[i]);
          }
        }
        // SGD update for W_mlp2 output head ONLY — W_mlp1 is intentionally frozen.
        // Updating W_mlp1 here inflates h1 (to compensate for W_mlp2's tiny 0.01 init scale),
        // which makes W_mlp2[4]@h1 a large constant for all inputs after pretraining.
        // That causes lambda=LAMBDA_MAX (5.5) for every material and kills discriminative Tc prediction.
        // Freezing W_mlp1 keeps h1 at He-init scale; Tc fine-tuning then updates W_mlp1 properly.
        const batchN = bEnd - bStart;
        for (let i = 0; i < OUTPUT_DIM; i++) {
          for (let j = 0; j < HIDDEN_DIM; j++) weights.W_mlp2[i][j] -= preLr * preGradW2[i][j] / batchN;
          weights.b_mlp2[i] -= preLr * preGradB2[i] / batchN;
        }
        // Invalidate flat matrix caches after in-place weight mutations
        invalidateFlatCache(weights.W_mlp2);
      }
      // yield after each pretrain epoch so heartbeat timers can fire
      await new Promise<void>(r => setTimeout(r, 0));
    }
    console.log(`[GNN] FE pretraining complete — starting Tc fine-tuning`);
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Cosine LR decay: LR_INIT → LR_INIT/10 over the training run.
    const lr = LR_INIT * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * epoch / epochs)));
    let totalLoss = 0;
    let totalSamples = 0;
    let tcSumErr2 = 0;    // sum of (predictedTcK - actualTcK)² — SC samples only
    let tcSumActual = 0;  // sum of actualTcK — for R² SS_tot
    let tcSumActual2 = 0; // sum of actualTcK² — for R² SS_tot
    let nTcSamples = 0;   // count of SC samples seen
    // v10 diagnostics: track P(SC) convergence and out[8]-only R² to distinguish
    // the classification gate from the Tc regression head during training.
    let sumScProbSC = 0, nScProbSC = 0;      // avg P(SC) for true SC samples
    let sumScProbNonSC = 0, nScProbNonSC = 0; // avg P(SC) for non-SC samples
    let out8SumErr2 = 0, out8SumActual = 0, out8SumActual2 = 0, nOut8 = 0; // out[8]-only R²

    // Stratified shuffle: interleave high-Tc samples so every batch sees them.
    // Shuffle each bucket independently, then interleave high-Tc evenly through others.
    const shuf = (arr: number[]) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    };
    shuf(hiTcBucket); shuf(loTcBucket); shuf(nonScBucket);
    // Merge low-Tc and non-SC into one stream, then interleave high-Tc evenly
    const mainstream = [...loTcBucket, ...nonScBucket];
    shuf(mainstream);
    if (hiTcBucket.length === 0) {
      for (let i = 0; i < indices.length; i++) indices[i] = mainstream[i] ?? i;
    } else {
      // Place one high-Tc sample every `step` mainstream samples
      const step = Math.max(1, Math.floor(mainstream.length / (hiTcBucket.length + 1)));
      let hi = 0, lo = 0, out = 0;
      while (out < indices.length) {
        for (let s = 0; s < step && lo < mainstream.length && out < indices.length; s++) indices[out++] = mainstream[lo++];
        if (hi < hiTcBucket.length && out < indices.length) indices[out++] = hiTcBucket[hi++];
      }
      while (lo < mainstream.length && out < indices.length) indices[out++] = mainstream[lo++];
      while (hi < hiTcBucket.length && out < indices.length) indices[out++] = hiTcBucket[hi++];
    }

    const numBatches = Math.ceil(trainingData.length / batchSize);

    for (let batch = 0; batch < numBatches; batch++) {
      const batchStart = batch * batchSize;
      const batchEnd = Math.min(batchStart + batchSize, trainingData.length);

      const pooledLen = HIDDEN_DIM * 2 + GLOBAL_COMP_DIM; // matches pooledWithComp in cache
      const batchSize_actual = batchEnd - batchStart;
      const clipGrad = (g: number) => { const v = Number.isFinite(g) ? g : 0; return Math.max(-1, Math.min(1, v)); };

      const gradW2 = weights.W_mlp2.map(r => new Array(r.length).fill(0));
      const gradB2 = new Array(weights.b_mlp2.length).fill(0);
      const gradW2v = weights.W_mlp2_var.map(r => new Array(r.length).fill(0));
      const gradB2v = new Array(weights.b_mlp2_var.length).fill(0);
      const gradW1 = weights.W_mlp1.map(r => new Array(r.length).fill(0));
      const gradB1 = new Array(weights.b_mlp1.length).fill(0);
      const gradPressure = new Array(weights.W_pressure.length).fill(0);
      // v15: Dedicated classification head accumulators.
      const gradWcls1 = weights.W_cls1.map(r => new Array(r.length).fill(0));
      const gradBcls1 = new Array(CLS_DIM).fill(0);
      const gradWcls2 = new Array(CLS_DIM).fill(0);
      let gradBcls2 = 0;

      const zeroMat = (m: number[][]) => m.map(r => new Array(r.length).fill(0));
      const graphGrads = {
        dW_msg: [zeroMat(weights.W_message), zeroMat(weights.W_message2), zeroMat(weights.W_message3), zeroMat(weights.W_message4)],
        dW_upd: [zeroMat(weights.W_update), zeroMat(weights.W_update2), zeroMat(weights.W_update3), zeroMat(weights.W_update4)],
        dW_filter1: zeroMat(weights.W_filter1),
        dW_filter2: zeroMat(weights.W_filter2),
        db_filter1: new Array(HIDDEN_DIM).fill(0),
        db_filter2: new Array(HIDDEN_DIM).fill(0),
        dW_input_proj: zeroMat(weights.W_input_proj),
        db_input_proj: new Array(HIDDEN_DIM).fill(0),
        dGates: new Array(4).fill(0),
        // GLFN-TC gradient accumulators
        dW_elem_feat:   zeroMat(weights.W_elem_feat),
        dW_graph_adapt: zeroMat(weights.W_graph_adapt),
        dDenseSkipGate: 0,
      };

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
        const nN = graph.nodes.length;

        // Prefer DFPT-derived Tc (from real QE λ) over the ML-estimated value when available.
        const tcTarget = (sample.qeDFPTTc != null ? sample.qeDFPTTc : sample.tc) / 300;
        const hasFormationEnergy = sample.formationEnergy != null;
        const feTarget = hasFormationEnergy ? sample.formationEnergy! : 0;
        // Weighted loss: high-Tc (discovery targets) are rare so up-weight them.
        // >100K: 4× (room-T SCs), 40-100K: 3×, >0: 2× (counteract contrast imbalance), =0: 1×
        // DFT-verified labels carry 5× the base weight — they are the ground truth.
        const feError = hasFormationEnergy ? pred.formationEnergy - feTarget : 0;
        // Bandgap: real measured value when available, otherwise skip.
        const bgTarget = sample.bandgap ?? null;
        const bgError  = bgTarget != null ? pred.bandgap - bgTarget : 0;

        const feWeight   = hasFormationEnergy ? 0.1 : 0.0;
        const actualTcK  = sample.qeDFPTTc ?? sample.tc;
        const isSC       = actualTcK > 0;
        // ── Sample weight: Tc-tier × dataConfidence multiplier ───────────────
        // Tc tiers: >100K (room-T discovery targets) = 4×, 40-100K = 3×,
        //           20-40K = 2×, >0K (low-Tc SC) = 1.5×, =0K (contrast) = 1×.
        // DFT-verified measurements get an additional 5× on top — they are ground truth.
        const tcTierW = !isSC ? 1.0
          : actualTcK >= 100 ? 4.0
          : actualTcK >= 40  ? 3.0
          : actualTcK >= 20  ? 2.0
          : 1.5;
        const dftVerified = sample.dataConfidence === 'dft-verified';
        const sampleW = dftVerified ? tcTierW * 5.0 : tcTierW;
        const scW = sampleW; // alias kept for Allen-Dynes auxiliary loss below
        const directOut7 = cache.outRaw[7] ?? 0;
        const directOut8 = cache.outRaw[8] ?? 0;

        // Track Tc predictions in Kelvin for inline R²/RMSE reporting.
        const pSC      = sigmoid(directOut7);
        if (isSC) {
          const predTcK = 10 * Math.expm1(Math.max(0, directOut8) * TC_LOG_SCALE);
          tcSumErr2    += (predTcK - actualTcK) ** 2;
          tcSumActual  += actualTcK;
          tcSumActual2 += actualTcK * actualTcK;
          nTcSamples++;
          // out[8]-only R²: raw magnitude head quality independent of gate.
          out8SumErr2    += (predTcK - actualTcK) ** 2;
          out8SumActual  += actualTcK;
          out8SumActual2 += actualTcK * actualTcK;
          nOut8++;
        }
        // P(SC) convergence tracking.
        if (isSC) { sumScProbSC += pSC; nScProbSC++; }
        else       { sumScProbNonSC += pSC; nScProbNonSC++; }

        // ── Loss: BCE classification + direct Tc regression ────────────────────
        // The previous "gated" Tc loss  3*(sigOut7*tcSigOut8 - target)²  was
        // removed because it created irreconcilable gradient conflicts:
        //   • It pulled tcSigOut8 → target/sigOut7 while the direct loss pulled
        //     tcSigOut8 → target.  With sigOut7 ≈ 0.7 these equilibria differ by
        //     ~0.26 in sigmoid space, causing out[8] to oscillate and never converge.
        //   • Its gate-gradient component (gateGrad) in dLdOut[7] opposed BCE,
        //     preventing W_cls2/W_cls1 from learning. Only b_cls2 adapted, giving
        //     a constant scProb = sigmoid(b_cls2) ≈ 0.775 for every input.
        //     Result: all 5 workers collapsed to 0.775 × 57.3 K = 44.4 K constant.
        // Fix: BCE trains classification cleanly; direct MSE trains regression cleanly.
        const scTarget = isSC ? 1.0 : 0.0;
        const sigOut7  = sigmoid(directOut7);
        const bceClamp = Math.max(1e-7, Math.min(1 - 1e-7, sigOut7));
        const bceLoss  = -(scTarget * Math.log(bceClamp) + (1 - scTarget) * Math.log(1 - bceClamp));
        // BCE loss is weighted by sampleW so DFT-verified and high-Tc samples
        // push the classifier harder than low-confidence experimental entries.
        totalLoss += feWeight * feError * feError + sampleW * bceLoss;
        // Direct Tc regression (SC samples only). Base coefficient 6.0 × sampleW.
        // out[8] is trained as a real number targeting log1p(Tc/10)/TC_LOG_SCALE ∈ [0,1].
        // No sigmoid activation — gradient = 12*(out8 - target), never vanishes.
        const tcNormTarget = isSC ? Math.log1p(actualTcK / 10) / TC_LOG_SCALE : 0;
        const tcDirectErr = isSC ? (directOut8 - tcNormTarget) : 0;
        if (isSC) totalLoss += sampleW * 6.0 * tcDirectErr * tcDirectErr;
        totalSamples++;

        // ── Allen-Dynes chain-rule gradient setup ───────────────────────────────
        // Reconstruct ω_log / λ / Tc from outRaw so we can differentiate through
        // the Allen-Dynes formula without adding fields to GNNForwardCache.
        // Reconstruct ω_log and λ using the same sigmoid soft-cap formulas as the forward pass.
        // Hard min() was replaced with sigmoid to keep gradients non-zero everywhere.
        const adOmegaLogRaw  = cache.outRaw[2] ?? 0;
        const adOmegaLogNorm = sigmoid(adOmegaLogRaw / 3);
        const adOmegaLog     = 10 + (OMEGA_LOG_MAX - 10) * adOmegaLogNorm;
        const adLambdaRaw    = cache.outRaw[4] ?? 0;
        const adLambdaSig    = sigmoid(adLambdaRaw);
        const adLambda       = LAMBDA_MAX * adLambdaSig;
        // omegaLog is in Kelvin; divide by 1.4388 to match the forward-pass unit fix
        const adTc           = allenDynesTcRaw(adLambda, adOmegaLog / 1.4388, FIXED_MU_STAR);
        // Allen-Dynes auxiliary loss (0.1× weight): soft physics prior keeping λ/ω_log calibrated.
        const allenDynesError = adTc / 300 - tcTarget;
        totalLoss += 0.1 * scW * allenDynesError * allenDynesError;
        // Use material-specific μ* from DFPT (JARVIS/QE) when available; fall back to 0.10.
        // Improves gradient accuracy for cuprates (μ*~0.15) and hydrides (μ*~0.08).
        const sampleMuStar = (sample.muStar != null && Number.isFinite(sample.muStar) && sample.muStar > 0)
          ? sample.muStar : FIXED_MU_STAR;
        // McMillan denominator: D = λ(1−0.62μ*) − μ*
        const adD = adLambda * (1 - 0.62 * sampleMuStar) - sampleMuStar;
        // dTc/dω_log = Tc / ω_log  (linear dependence in Allen-Dynes)
        const dTcdOmegaLog = (adOmegaLog > 10 && adTc > 0) ? adTc / adOmegaLog : 0;
        // dTc/dλ = Tc × 1.07952 / D²  (McMillan simplified, always positive for D>0)
        const dTcdLambda = (adD > 0.05 && adTc > 0) ? adTc * 1.07952 / (adD * adD) : 0;
        // d(ω_log)/d(out[2]) = (OMEGA_LOG_MAX−10) × sigmoid'(out[2]/3) — always non-zero
        const dOmegaLogdOut2 = (OMEGA_LOG_MAX - 10) * adOmegaLogNorm * (1 - adOmegaLogNorm) / 3;
        // d(λ)/d(out[4]) = LAMBDA_MAX × sigmoid'(out[4]) — always non-zero
        const dLambdadOut4 = LAMBDA_MAX * adLambdaSig * (1 - adLambdaSig);

        // Direct ω_log supervision.
        // Priority 1: measured DFPT value from JARVIS/QE (sample.omegaLog in Kelvin) — most accurate.
        // Priority 2: invert Allen-Dynes from (Tc_target, λ_target) — derived, less reliable.
        let omegaLogGradBoost = 0;
        const dfptOmegaLog = (sample.omegaLog != null && Number.isFinite(sample.omegaLog) && sample.omegaLog > 10)
          ? sample.omegaLog : null;
        if (dfptOmegaLog !== null) {
          // Measured DFPT ω_log: real ground truth, supervise directly.
          const omegaLogRelErr = (adOmegaLog - dfptOmegaLog) / dfptOmegaLog;
          totalLoss += 0.1 * omegaLogRelErr * omegaLogRelErr;
          omegaLogGradBoost = 0.2 * omegaLogRelErr / dfptOmegaLog;
        }

        // ── Family classification ──────────────────────────────────────────────
        // Composition-only rule to route the correct physics formula per family.
        // Different SC families have different pairing mechanisms — applying a
        // formula from the wrong family corrupts gradients in the shared backbone.
        const _fc = parseFormulaCountsCanonical(sample.formula);
        const _els = new Set(Object.keys(_fc));
        const _total = Math.max(1, Object.values(_fc).reduce((a: number, b: number) => a + b, 0));
        const _hFrac = (_fc['H'] ?? 0) / _total;
        let _trainFamily = 'conventional_bcs';
        if (_hFrac > 0.35 && actualTcK > 50) {
          _trainFamily = 'hydride';
        } else if (_els.has('Cu') && _els.has('O') && _els.size >= 3) {
          _trainFamily = 'cuprate';
        } else if (_els.has('Fe') && (_els.has('As') || _els.has('Se') || _els.has('P'))) {
          _trainFamily = 'iron_based';
        } else if (_els.has('Ni') && _els.has('O') && (_els.has('La') || _els.has('Nd') || _els.has('Pr'))) {
          _trainFamily = 'nickelate';
        }

        // ── Physics-based auxiliary losses ────────────────────────────────────
        // Each family gets the physically correct formula as a training constraint,
        // or NO constraint if no reliable formula exists (better than a wrong one).
        //
        // HYDRIDES — Full Allen-Dynes with Dynes f1/f2 strong-coupling corrections
        //   Tc = (ω_log/1.2) * f1 * f2 * exp[-1.04*(1+λ)/(λ*(1-0.62μ*)-μ*)]
        //   f1 = [1 + (λ/λ_bar)^1.5]^(1/3),  λ_bar = 2.46*(1+3.8*μ*)
        //   f2 = 1 + (λ−λ_0)²/(λ³+λ_0³),    λ_0 = sqrt(1.82*1.04*μ*/(1+6.3*μ*))
        //   Source: Allen & Dynes (1975) Phys. Rev. B 12, 905
        //   Valid for all λ including strong-coupling (LaH10 λ≈3.5, H3S λ≈2.0)
        //
        // CUPRATES — Presland-Tallon (1991) hole-doping parabola
        //   Tc(p) = Tc_max * [1 − 82.6*(p − 0.16)²]
        //   p = hole doping per CuO2 plane, estimated from formal charge balance on Cu
        //   Tc_max estimated from composition (Hg > Tl > Bi/Y > La family hierarchy)
        //   Source: Presland et al. (1991) Physica C 176, 95
        //   This is NOT Allen-Dynes — cuprates are d-wave, pairing via spin fluctuations.
        //
        // IRON-BASED — No reliable analytical formula (s± pairing, nesting-dependent)
        //   Allen-Dynes is wrong here. Direct MSE on out[8] is the only reliable signal.
        //
        // CONVENTIONAL BCS — Standard McMillan/Allen-Dynes
        //   Same as the general adTc already computed above.

        let _physAdTc = adTc;                   // formula Tc for phonon families
        let _dPhysTcdLambda = dTcdLambda;
        let _dPhysTcdOmegaLog = dTcdOmegaLog;
        let _physLossW = 0.0;                   // loss weight for each family
        let _cuprateAuxLoss = 0.0;              // Presland-Tallon loss for cuprates

        const _mu = sampleMuStar;

        if (isSC && _trainFamily === 'hydride' && adLambda > 0.5 && adOmegaLog > 10) {
          // Full Allen-Dynes with strong-coupling f1/f2 (Allen & Dynes 1975).
          // This is the accepted standard for λ > 1 superconductors.
          const _D = Math.max(0.001, adLambda * (1 - 0.62 * _mu) - _mu);
          const _expFact = Math.exp(Math.max(-40, -1.04 * (1 + adLambda) / _D));
          const _tcBase = (adOmegaLog / 1.2) * _expFact;
          const _dTcBase_dLambda = _tcBase * 1.04 * (1 + 0.38 * _mu) / (_D * _D);
          const _lambdaBar = 2.46 * (1 + 3.8 * _mu);
          const _lambda0 = Math.sqrt(Math.max(0, 1.82 * 1.04 * _mu / (1 + 6.3 * _mu)));
          const _f1base = 1 + Math.pow(adLambda / _lambdaBar, 1.5);
          const _f1 = Math.pow(Math.max(1e-10, _f1base), 1 / 3);
          const _f2num = adLambda > _lambda0 ? Math.pow(adLambda - _lambda0, 2) : 0;
          const _f2den = Math.max(1e-10, Math.pow(adLambda, 3) + Math.pow(_lambda0, 3));
          const _f2 = 1 + _f2num / _f2den;
          _physAdTc = _tcBase * _f1 * _f2;
          // dTc/dλ via exact product rule through f1, f2, and exp(...)
          const _df1dLambda = _f1base > 1e-10
            ? 0.5 * Math.pow(adLambda / _lambdaBar, 0.5) / (_lambdaBar * Math.pow(_f1base, 2 / 3))
            : 0;
          let _df2dLambda = 0;
          if (adLambda > _lambda0 && _f2den > 1e-10) {
            const _dn = 2 * (adLambda - _lambda0);
            const _dd = 3 * adLambda * adLambda;
            _df2dLambda = (_dn * _f2den - _f2num * _dd) / (_f2den * _f2den);
          }
          _dPhysTcdLambda = _f1 * _f2 * _dTcBase_dLambda
            + _tcBase * (_df1dLambda * _f2 + _f1 * _df2dLambda);
          _dPhysTcdOmegaLog = adOmegaLog > 0 ? _physAdTc / adOmegaLog : 0;
          _physLossW = 0.15;  // stronger weight for hydrides — formula is accurate

        } else if (isSC && _trainFamily === 'cuprate') {
          // Presland-Tallon (1991): Tc(p) = Tc_max * [1 − 82.6*(p − 0.16)²]
          // p = hole doping per CuO2 plane, estimated from formal charge balance.
          // Charge balance: sum of cation oxidation states must balance O²⁻ × n_O.
          // Cu nominal valence in parent compound is 2+. Holes come from:
          //   1. Heterovalent substitution (e.g., Sr²⁺→La³⁺, Ca²⁺→Y³⁺)
          //   2. Excess oxygen (e.g., YBa2Cu3O6+x, La2CuO4+δ)
          // We approximate: p = (n_O * 2 - Σ non-Cu cation oxidation) / n_Cu - 2
          // where Σ non-Cu cation oxidation is estimated from standard valences.
          const _ox: Record<string, number> = {
            H: 1, Li: 1, Na: 1, K: 1, Rb: 1, Cs: 1,
            Be: 2, Mg: 2, Ca: 2, Sr: 2, Ba: 2,
            Al: 3, Ga: 3, In: 3,
            La: 3, Nd: 3, Pr: 3, Sm: 3, Eu: 3, Gd: 3, Tb: 3, Dy: 3, Ho: 3, Er: 3, Tm: 3, Yb: 3, Lu: 3,
            Y: 3, Sc: 3, Bi: 3,
            Hg: 2, Tl: 1, Pb: 2, Cd: 2, Zn: 2,
          };
          const _nCu = _fc['Cu'] ?? 0;
          const _nO  = _fc['O'] ?? 0;
          if (_nCu > 0 && _nO > 0) {
            let _cationSum = 0;
            for (const [el, cnt] of Object.entries(_fc)) {
              if (el === 'Cu' || el === 'O') continue;
              _cationSum += (_ox[el] ?? 2) * (cnt as number);
            }
            // Cu oxidation = (2*n_O - _cationSum) / n_Cu
            const _cuOx = (2 * _nO - _cationSum) / _nCu;
            const _pEst = Math.max(0, Math.min(0.35, _cuOx - 2));  // holes above Cu²⁺
            // Tc_max from family hierarchy (literature Tc_max per compound family):
            //   Hg-family (Hg-1223): ~135 K    Tl-family (Tl-2223): ~127 K
            //   Bi-2223: ~110 K                Bi-2212 / Y-123: ~93 K
            //   La-214 (LSCO): ~40 K
            let _tcMax = 93;  // default (YBCO-class)
            if (_els.has('Hg')) _tcMax = 135;
            else if (_els.has('Tl') && (_fc['Tl'] ?? 0) >= 2) _tcMax = 127;
            else if (_els.has('Bi') && (_fc['Cu'] ?? 0) >= 3) _tcMax = 110;
            else if (_els.has('La') && !_els.has('Y') && !_els.has('Bi')) _tcMax = 40;
            // Presland-Tallon formula
            const _ptTc = _tcMax * Math.max(0, 1 - 82.6 * Math.pow(_pEst - 0.16, 2));
            // Confidence: higher near optimal doping (p=0.16), lower at edges
            const _ptConf = Math.max(0, 1 - Math.abs(_pEst - 0.16) / 0.12);
            if (_pEst > 0.02 && _pEst < 0.32 && _ptConf > 0.2) {
              // Apply loss only when doping estimate is in a physically meaningful range.
              // This loss trains out[8] directly (no gradient through λ/ωlog — correct,
              // since cuprate Tc is NOT driven by phonon coupling).
              // Both sides in log-normalized space (same units as tcNormTarget after direct-regression fix)
              const _ptErr = Math.log1p(_ptTc / 10) / TC_LOG_SCALE - tcNormTarget;
              _cuprateAuxLoss = 0.1 * _ptConf * _ptErr * _ptErr;
              totalLoss += _cuprateAuxLoss;
              // Gradient of cuprate PT loss w.r.t. out[8] (sigmoid-activated Tc head)
              // d/d(out8): 2 * 0.1 * ptConf * ptErr * d(ptTc/300)/d(sigmoid(out8))
              // ptTc has no dependence on the GNN outputs — it is computed from composition.
              // So this loss does NOT propagate through out[2]/out[4], only provides
              // an additional normalizing reference for the training loss metric.
            }
          }
          // For cuprates: no Allen-Dynes gradient — pairing is non-phonon.
          _physLossW = 0.0;

        } else if (isSC && _trainFamily === 'conventional_bcs' && adLambda > 0.1 && adOmegaLog > 10) {
          // Standard Allen-Dynes (McMillan 1968 / Allen & Dynes 1975 simplified form).
          _physAdTc = adTc;
          _dPhysTcdLambda = dTcdLambda;
          _dPhysTcdOmegaLog = dTcdOmegaLog;
          _physLossW = 0.1;

        }
        // iron_based, nickelate: _physLossW = 0 — no reliable formula, data-driven only.

        const _physErr = _physLossW > 0 ? (_physAdTc / 300 - tcTarget) : 0;
        if (_physLossW > 0) totalLoss += _physLossW * scW * _physErr * _physErr;
        const _dLossDAd = _physLossW > 0 ? (_physLossW * scW * 2 * _physErr / 300) : 0;

        const dLdOut = new Array(OUTPUT_DIM).fill(0);
        dLdOut[0] = hasFormationEnergy ? clipGrad(2 * feError * 0.1) : 0;
        dLdOut[1] = 0;
        dLdOut[2] = clipGrad((_dLossDAd * _dPhysTcdOmegaLog + omegaLogGradBoost) * dOmegaLogdOut2);
        dLdOut[3] = 0;
        dLdOut[4] = clipGrad(_dLossDAd * _dPhysTcdLambda * dLambdadOut4);
        const bgSig5 = sigmoid(cache.outRaw[5] ?? 0);
        dLdOut[5] = bgTarget != null ? clipGrad(2 * bgError * 0.05 * 5.0 * bgSig5 * (1 - bgSig5)) : 0;
        dLdOut[6] = 0;
        // out[7]: standard BCE gradient
        // dL/d(out[7]) = sampleW * (σ(out7) - target): BCE gradient scaled by sample weight
        dLdOut[7] = clipGrad(sampleW * (sigOut7 - scTarget));
        // dL/d(out[8]) = sampleW * 2*6 * tcDirectErr  (no sigmoid → no vanishing gradient)
        dLdOut[8] = isSC ? clipGrad(sampleW * 12.0 * tcDirectErr) : 0;

        // Uncertainty head (W_mlp2_var) receives no gradient — NLL removed.
        // Uncertainty outputs can be calibrated post-training via isotonic regression.
        const dLdLogVarOut = new Array(OUTPUT_DIM).fill(0);

        // v15: i=7 is handled by cls head — skip in W_mlp2 gradient loops.
        for (let i = 0; i < weights.W_mlp2.length; i++) {
          if (i === 7) continue;
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

        // v15: dLdH1 excludes i=7 (cls head) so Tc regression doesn't pollute P(SC).
        const dLdH1 = new Array(HIDDEN_DIM).fill(0);
        for (let j = 0; j < HIDDEN_DIM; j++) {
          for (let i = 0; i < OUTPUT_DIM; i++) {
            if (i === 7) continue;
            dLdH1[j] += dLdOut[i] * (weights.W_mlp2[i]?.[j] ?? 0);
            dLdH1[j] += dLdLogVarOut[i] * (weights.W_mlp2_var[i]?.[j] ?? 0);
          }
        }

        // v15: Classification head backward — out[7] → W_cls2 → hCls → W_cls1 → pooledWithComp.
        const dOut7 = dLdOut[7];
        for (let j = 0; j < CLS_DIM; j++) {
          gradWcls2[j] += dOut7 * cache.hCls[j];
        }
        gradBcls2 += dOut7;
        const dLdHcls = new Array(CLS_DIM).fill(0);
        for (let j = 0; j < CLS_DIM; j++) {
          dLdHcls[j] = dOut7 * (weights.W_cls2[j] ?? 0);
        }
        const dLdZcls = new Array(CLS_DIM);
        for (let j = 0; j < CLS_DIM; j++) {
          dLdZcls[j] = dLdHcls[j] * (cache.zCls[j] >= 0 ? 1.0 : 0.01);
        }
        for (let i = 0; i < CLS_DIM; i++) {
          for (let j = 0; j < pooledLen; j++) {
            gradWcls1[i][j] += clipGrad(dLdZcls[i] * cache.pooled[j]);
          }
          gradBcls1[i] += clipGrad(dLdZcls[i]);
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
          // v15: add cls head contribution so backbone learns from P(SC) signal.
          for (let i = 0; i < CLS_DIM; i++) {
            dLdPooled[j] += dLdZcls[i] * (weights.W_cls1[i][j] ?? 0);
          }
          dLdPooled[j] = clipGrad(dLdPooled[j]);
        }

        for (let k = 0; k < HIDDEN_DIM; k++) {
          gradPressure[k] += clipGrad(dLdPooled[k] * ((graph.pressureGpa ?? 0) / 300));
        }

        const dLdNodeEmb: number[][] = Array.from({ length: nN }, () => new Array(HIDDEN_DIM).fill(0));
        for (let ni = 0; ni < nN; ni++) {
          const mult = (cache.nodeMultiplicities[ni] ?? 1) / cache.totalMultiplicity;
          for (let k = 0; k < HIDDEN_DIM; k++) {
            dLdNodeEmb[ni][k] += dLdPooled[k] * 0.5 * mult;
          }
        }
        for (let k = 0; k < HIDDEN_DIM; k++) {
          const maxIdx = cache.maxPoolArgmax[k] ?? 0;
          dLdNodeEmb[maxIdx][k] += dLdPooled[HIDDEN_DIM + k];
        }
        for (let ni = 0; ni < nN; ni++) {
          const aw = cache.attnPoolWeights[ni] ?? 0;
          for (let k = 0; k < HIDDEN_DIM; k++) {
            dLdNodeEmb[ni][k] += dLdPooled[k] * 0.5 * aw;
          }
        }

        const gateVals = weights.residual_gates.map(g => sigmoid(g));

        const allLayerWeights: { W_msg: number[][]; W_upd: number[][]; useLeaky: boolean; gateIdx: number }[] = [
          { W_msg: weights.W_message4, W_upd: weights.W_update4, useLeaky: false, gateIdx: 3 },
          { W_msg: weights.W_message3, W_upd: weights.W_update3, useLeaky: false, gateIdx: 2 },
          { W_msg: weights.W_message2, W_upd: weights.W_update2, useLeaky: false, gateIdx: 1 },
        ];
        // Only backprop through layers that were actually run in the forward pass
        const layerWeights = allLayerWeights.slice(4 - GNN_MSG_LAYERS);
        const layerCacheIndices = layerWeights.map(lw => lw.gateIdx);

        // GLFN-TC Dense Residual backward: save layer-2 attention input gradient
        // so we can route it back to H0 through the dense skip connection.
        let dLdInput_layer2: number[][] | null = null;
        let dLdCur = dLdNodeEmb;
        for (let li = 0; li < layerWeights.length; li++) {
          const lw = layerWeights[li];
          const ci = layerCacheIndices[li];
          const ac = cache.attnCaches[ci];
          const gi = lw.gateIdx;
          const gv = gateVals[gi];

          const dLdLayerIn: number[][] = Array.from({ length: nN }, () => new Array(HIDDEN_DIM).fill(0));
          for (let ni = 0; ni < nN; ni++) {
            for (let k = 0; k < HIDDEN_DIM; k++) {
              dLdLayerIn[ni][k] = dLdCur[ni][k] * gv;
            }
          }

          const { dW_update, dW_message, dLdInput } = attnLayerBackward(ac, dLdCur, graph, lw.W_upd, lw.W_msg, lw.useLeaky);

          // GLFN-TC: capture the gradient at layer-2's attention input (= H1 + denseGate*H0).
          // This is used below to back-propagate through the dense skip to H0.
          if (gi === 1) dLdInput_layer2 = dLdInput;

          const gradIdx = lw.gateIdx;
          const addMat = (dst: number[][], src: number[][]) => {
            for (let r = 0; r < dst.length; r++)
              for (let c = 0; c < dst[r].length; c++)
                dst[r][c] += clipGrad(src[r][c]);
          };
          addMat(graphGrads.dW_msg[gradIdx], dW_message);
          addMat(graphGrads.dW_upd[gradIdx], dW_update);

          for (let ni = 0; ni < nN; ni++) {
            for (let k = 0; k < HIDDEN_DIM; k++) {
              dLdLayerIn[ni][k] += clipGrad(dLdInput[ni][k]);
            }
          }

          let dEmbSum = 0;
          for (let ni = 0; ni < nN; ni++)
            for (let k = 0; k < HIDDEN_DIM; k++)
              dEmbSum += dLdCur[ni][k] * (ac.inputEmbs[ni]?.[k] ?? 0);
          graphGrads.dGates[gi] += clipGrad(dEmbSum * gv * (1 - gv));

          dLdCur = dLdLayerIn;
        }

        // GLFN-TC Dense Residual (§2.4) backward:
        //   Forward:  layer2_input = H1 + denseGate * H0
        //   ∂L/∂H0_dense  = denseGate  * dLdInput_layer2  (added to dLdPreProj below)
        //   ∂L/∂dense_raw = Σ dLdInput_layer2[i][k] * H0[i][k]  * gate*(1-gate)
        const denseGateVal = sigmoid(weights.dense_skip_gate ?? -2);
        if (dLdInput_layer2 && cache.denseH0) {
          let dDenseSkipRaw = 0;
          for (let ni = 0; ni < nN; ni++) {
            for (let k = 0; k < HIDDEN_DIM; k++) {
              dDenseSkipRaw += (dLdInput_layer2[ni]?.[k] ?? 0) * (cache.denseH0[ni]?.[k] ?? 0);
            }
          }
          graphGrads.dDenseSkipGate += clipGrad(dDenseSkipRaw * denseGateVal * (1 - denseGateVal));
        }

        const ac0 = cache.attnCaches[0];
        if (cache.cgcnnCache) {
          const { dW_filter1, dW_filter2, db_filter1, db_filter2, dLdInput: dLdCgcnnIn, dAdaptiveLogits } =
            cgcnnLayerBackward(cache.cgcnnCache, dLdCur, graph, weights.W_filter1, weights.W_filter2);
          const addMat = (dst: number[][], src: number[][]) => {
            for (let r = 0; r < dst.length; r++)
              for (let c = 0; c < dst[r].length; c++)
                dst[r][c] += clipGrad(src[r][c]);
          };
          addMat(graphGrads.dW_filter1, dW_filter1);
          addMat(graphGrads.dW_filter2, dW_filter2);
          for (let k = 0; k < HIDDEN_DIM; k++) {
            graphGrads.db_filter1[k] += clipGrad(db_filter1[k]);
            graphGrads.db_filter2[k] += clipGrad(db_filter2[k]);
          }
          dLdCur = dLdCgcnnIn;

          // GLFN-TC Graph Learning Module backward:
          //   adaptive_logit(i,j) = eᵢᵀ W_adapt eⱼ
          //   ∂L/∂W_adapt[p][q]  += dLogit * ei[p] * ej[q]
          //   ∂L/∂ei[p]          += dLogit * (W_adapt @ ej)[p]
          //   ∂L/∂ej[q]          += dLogit * (W_adaptᵀ @ ei)[q]
          if (dAdaptiveLogits) {
            for (let i = 0; i < nN; i++) {
              const atomI = Math.min(Math.max(graph.nodes[i].atomicNumber, 0), 118);
              const ei = weights.W_elem_feat[atomI] ?? new Array(GRAPH_FEAT_DIM).fill(0);
              const neighbors = graph.adjacency[i] ?? [];
              for (let nIdx = 0; nIdx < neighbors.length; nIdx++) {
                const dLogit = dAdaptiveLogits[i]?.[nIdx] ?? 0;
                if (Math.abs(dLogit) < 1e-12) continue;
                const j = neighbors[nIdx];
                const atomJ = Math.min(Math.max(graph.nodes[j].atomicNumber, 0), 118);
                const ej = weights.W_elem_feat[atomJ] ?? new Array(GRAPH_FEAT_DIM).fill(0);
                // Gradient of the bilinear form logit = eᵢᵀ W_adapt eⱼ
                for (let p = 0; p < GRAPH_FEAT_DIM; p++) {
                  for (let q = 0; q < GRAPH_FEAT_DIM; q++) {
                    graphGrads.dW_graph_adapt[p][q] += clipGrad(dLogit * (ei[p] ?? 0) * (ej[q] ?? 0));
                  }
                  // ∂L/∂ei via W_adapt @ ej
                  let wAdaptEj_p = 0;
                  for (let q = 0; q < GRAPH_FEAT_DIM; q++) wAdaptEj_p += (weights.W_graph_adapt[p]?.[q] ?? 0) * (ej[q] ?? 0);
                  graphGrads.dW_elem_feat[atomI][p] += clipGrad(dLogit * wAdaptEj_p);
                  // ∂L/∂ej via W_adaptᵀ @ ei
                  let wAdaptTEi_p = 0;
                  for (let q = 0; q < GRAPH_FEAT_DIM; q++) wAdaptTEi_p += (weights.W_graph_adapt[q]?.[p] ?? 0) * (ei[q] ?? 0);
                  graphGrads.dW_elem_feat[atomJ][p] += clipGrad(dLogit * wAdaptTEi_p);
                }
              }
            }
          }
        }

        {
          const { dW_update, dW_message, dLdInput } = attnLayerBackward(ac0, dLdCur, graph, weights.W_update, weights.W_message, true);
          const addMat = (dst: number[][], src: number[][]) => {
            for (let r = 0; r < dst.length; r++)
              for (let c = 0; c < dst[r].length; c++)
                dst[r][c] += clipGrad(src[r][c]);
          };
          addMat(graphGrads.dW_msg[0], dW_message);
          addMat(graphGrads.dW_upd[0], dW_update);

          const g0v = gateVals[0];
          const dLdPreProj: number[][] = Array.from({ length: nN }, () => new Array(HIDDEN_DIM).fill(0));
          for (let ni = 0; ni < nN; ni++) {
            for (let k = 0; k < HIDDEN_DIM; k++) {
              // Standard gradients: layer-1 attention backward + gated residual to H0.
              // GLFN-TC Dense Residual: dLdInput_layer2 (grad at layer-2 attention input)
              // also flows back to H0 through the dense skip (H0 was added to H1).
              const denseSkipContrib = dLdInput_layer2 ? denseGateVal * (dLdInput_layer2[ni]?.[k] ?? 0) : 0;
              dLdPreProj[ni][k] = clipGrad(dLdInput[ni][k]) + dLdCur[ni][k] * g0v + clipGrad(denseSkipContrib);
            }
          }

          let dEmbSum0 = 0;
          for (let ni = 0; ni < nN; ni++)
            for (let k = 0; k < HIDDEN_DIM; k++)
              dEmbSum0 += dLdCur[ni][k] * (ac0.inputEmbs[ni]?.[k] ?? 0);
          graphGrads.dGates[0] += clipGrad(dEmbSum0 * g0v * (1 - g0v));

          for (let ni = 0; ni < nN; ni++) {
            const pre = cache.inputProjPreActs[ni];
            const inp = cache.inputProjInputs[ni];
            for (let k = 0; k < HIDDEN_DIM; k++) {
              const dAct = dLdPreProj[ni][k] * (pre[k] >= 0 ? 1.0 : 0.01);
              graphGrads.db_input_proj[k] += clipGrad(dAct);
              for (let c = 0; c < NODE_DIM; c++) {
                graphGrads.dW_input_proj[k][c] += clipGrad(dAct * (inp[c] ?? 0));
              }
            }
          }
        }
      }

      const invN = 1.0 / batchSize_actual;
      adamStep++;
      const bc1 = 1 - Math.pow(adamBeta1, adamStep);
      const bc2 = 1 - Math.pow(adamBeta2, adamStep);
      const adamUpdate = (
        w: number[][], g: number[][], am: { m: number[][]; v: number[][] }, rows: number, cols: number
      ) => {
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const gi = g[i][j] * invN;
            am.m[i][j] = adamBeta1 * am.m[i][j] + (1 - adamBeta1) * gi;
            am.v[i][j] = adamBeta2 * am.v[i][j] + (1 - adamBeta2) * gi * gi;
            w[i][j] -= lr * (am.m[i][j] / bc1) / (Math.sqrt(am.v[i][j] / bc2) + adamEps);
            w[i][j] *= (1 - lr * WEIGHT_DECAY);
          }
        }
      };
      const adamUpdateVec = (
        w: number[], g: number[], av: { m: number[]; v: number[] }, n: number
      ) => {
        for (let i = 0; i < n; i++) {
          const gi = g[i] * invN;
          av.m[i] = adamBeta1 * av.m[i] + (1 - adamBeta1) * gi;
          av.v[i] = adamBeta2 * av.v[i] + (1 - adamBeta2) * gi * gi;
          w[i] -= lr * (av.m[i] / bc1) / (Math.sqrt(av.v[i] / bc2) + adamEps);
          w[i] *= (1 - lr * WEIGHT_DECAY);
        }
      };

      adamUpdate(weights.W_mlp2, gradW2, adamW2, weights.W_mlp2.length, weights.W_mlp2[0].length);
      adamUpdateVec(weights.b_mlp2, gradB2, adamB2, weights.b_mlp2.length);
      adamUpdate(weights.W_mlp2_var, gradW2v, adamW2v, weights.W_mlp2_var.length, weights.W_mlp2_var[0].length);
      adamUpdateVec(weights.b_mlp2_var, gradB2v, adamB2v, weights.b_mlp2_var.length);
      adamUpdate(weights.W_mlp1, gradW1, adamW1, HIDDEN_DIM, pooledLen);
      adamUpdateVec(weights.b_mlp1, gradB1, adamB1, HIDDEN_DIM);
      // v15: Classification head Adam updates.
      adamUpdate(weights.W_cls1, gradWcls1, adamWcls1, CLS_DIM, pooledLen);
      adamUpdateVec(weights.b_cls1, gradBcls1, adamBcls1, CLS_DIM);
      adamUpdateVec(weights.W_cls2, gradWcls2, adamWcls2, CLS_DIM);
      {
        const gb2 = gradBcls2 * invN;
        adamBcls2.m[0] = adamBeta1 * adamBcls2.m[0] + (1 - adamBeta1) * gb2;
        adamBcls2.v[0] = adamBeta2 * adamBcls2.v[0] + (1 - adamBeta2) * gb2 * gb2;
        weights.b_cls2 -= lr * (adamBcls2.m[0] / bc1) / (Math.sqrt(adamBcls2.v[0] / bc2) + adamEps);
        weights.b_cls2 *= (1 - lr * WEIGHT_DECAY);
      }
      // Graph layer AdamW — lr * 0.5 (was 0.3). Direct Tc path delivers strong gradients
      // to the graph layers so they need a higher LR to update meaningfully.
      const graphLR = lr * 0.5;
      const adamUpdateGraph = (
        w: number[][], g: number[][], am: { m: number[][]; v: number[][] }
      ) => {
        for (let i = 0; i < w.length; i++)
          for (let j = 0; j < w[i].length; j++) {
            const gi = g[i][j] * invN;
            am.m[i][j] = adamBeta1 * am.m[i][j] + (1 - adamBeta1) * gi;
            am.v[i][j] = adamBeta2 * am.v[i][j] + (1 - adamBeta2) * gi * gi;
            w[i][j] -= graphLR * (am.m[i][j] / bc1) / (Math.sqrt(am.v[i][j] / bc2) + adamEps);
            w[i][j] *= (1 - graphLR * WEIGHT_DECAY);
          }
      };
      const adamUpdateGraphVec = (
        w: number[], g: number[], av: { m: number[]; v: number[] }
      ) => {
        for (let i = 0; i < w.length; i++) {
          const gi = g[i] * invN;
          av.m[i] = adamBeta1 * av.m[i] + (1 - adamBeta1) * gi;
          av.v[i] = adamBeta2 * av.v[i] + (1 - adamBeta2) * gi * gi;
          w[i] -= graphLR * (av.m[i] / bc1) / (Math.sqrt(av.v[i] / bc2) + adamEps);
          w[i] *= (1 - graphLR * WEIGHT_DECAY);
        }
      };
      const wMsgMats = [weights.W_message, weights.W_message2, weights.W_message3, weights.W_message4];
      const wUpdMats = [weights.W_update,  weights.W_update2,  weights.W_update3,  weights.W_update4 ];
      for (let li = 0; li < GNN_MSG_LAYERS; li++) {
        adamUpdateGraph(wMsgMats[li], graphGrads.dW_msg[li], graphAdam.msg[li]);
        adamUpdateGraph(wUpdMats[li], graphGrads.dW_upd[li], graphAdam.upd[li]);
      }
      adamUpdateGraph(weights.W_filter1, graphGrads.dW_filter1, graphAdam.filter1);
      adamUpdateGraph(weights.W_filter2, graphGrads.dW_filter2, graphAdam.filter2);
      adamUpdateGraphVec(weights.b_filter1, graphGrads.db_filter1, graphAdam.bFilter1);
      adamUpdateGraphVec(weights.b_filter2, graphGrads.db_filter2, graphAdam.bFilter2);
      adamUpdateGraph(weights.W_input_proj,    graphGrads.dW_input_proj, graphAdam.inputProj);
      adamUpdateGraphVec(weights.b_input_proj, graphGrads.db_input_proj, graphAdam.bInputProj);
      adamUpdateGraphVec(weights.W_pressure,   gradPressure,             graphAdam.pressure);
      for (let g = 0; g < weights.residual_gates.length; g++) {
        const gi = graphGrads.dGates[g] * invN;
        graphAdam.gates.m[g] = adamBeta1 * graphAdam.gates.m[g] + (1 - adamBeta1) * gi;
        graphAdam.gates.v[g] = adamBeta2 * graphAdam.gates.v[g] + (1 - adamBeta2) * gi * gi;
        weights.residual_gates[g] -= graphLR * (graphAdam.gates.m[g] / bc1) / (Math.sqrt(graphAdam.gates.v[g] / bc2) + adamEps);
        weights.residual_gates[g] = Math.max(-3, Math.min(3, weights.residual_gates[g]));
      }

      // GLFN-TC: Adam updates for Graph Learning Module and Dense Residual gate.
      adamUpdateGraph(weights.W_elem_feat,   graphGrads.dW_elem_feat,   graphAdam.elemFeat);
      adamUpdateGraph(weights.W_graph_adapt, graphGrads.dW_graph_adapt, graphAdam.graphAdapt);
      {
        const dsg = graphGrads.dDenseSkipGate * invN;
        graphAdam.denseSkipGate.m[0] = adamBeta1 * graphAdam.denseSkipGate.m[0] + (1 - adamBeta1) * dsg;
        graphAdam.denseSkipGate.v[0] = adamBeta2 * graphAdam.denseSkipGate.v[0] + (1 - adamBeta2) * dsg * dsg;
        weights.dense_skip_gate -= graphLR * (graphAdam.denseSkipGate.m[0] / bc1) / (Math.sqrt(graphAdam.denseSkipGate.v[0] / bc2) + adamEps);
        // Clamp so denseGate = sigmoid(raw) stays in (0.01, 0.99) — prevents the skip
        // from either fully vanishing or drowning the gated-residual path.
        weights.dense_skip_gate = Math.max(-4.6, Math.min(4.6, weights.dense_skip_gate));
      }
    }

    // Progress bar — log every ~20% of epochs, always log first and last.
    const logInterval = Math.max(1, Math.ceil(epochs / 5));
    const isLast = epoch === epochs - 1;
    const avgLoss = totalSamples > 0 ? totalLoss / totalSamples : 0;
    if (epoch === 0 || (epoch + 1) % logInterval === 0 || isLast) {
      const bar = _gnnBar(epoch, epochs);
      const pct = Math.round(100 * (epoch + 1) / epochs);
      const pfx = label ? `[GNN|${label}]` : '[GNN]';
      let r2Str = '?.???', rmseStr = '???K';
      if (nTcSamples > 1) {
        const ssTot = tcSumActual2 - (tcSumActual * tcSumActual) / nTcSamples;
        const r2 = ssTot > 0 ? 1 - tcSumErr2 / ssTot : 0;
        const rmse = Math.sqrt(tcSumErr2 / nTcSamples);
        r2Str   = r2.toFixed(3);
        rmseStr = `${rmse.toFixed(1)}K`;
      }
      // P(SC) gate diagnostics — shows whether classification is converging.
      const avgPscSC    = nScProbSC    > 0 ? (sumScProbSC    / nScProbSC).toFixed(2)    : '?.??';
      const avgPscNonSC = nScProbNonSC > 0 ? (sumScProbNonSC / nScProbNonSC).toFixed(2) : '?.??';
      // out[8]-only R² — shows Tc regression quality independent of P(SC) gate.
      let out8R2Str = '?.???';
      if (nOut8 > 1) {
        const ssTot8 = out8SumActual2 - (out8SumActual * out8SumActual) / nOut8;
        out8R2Str = (ssTot8 > 0 ? 1 - out8SumErr2 / ssTot8 : 0).toFixed(3);
      }
      console.log(`${pfx} ${bar} ${String(pct).padStart(3)}% | Epoch ${epoch + 1}/${epochs} | loss=${avgLoss.toFixed(4)} | R²=${r2Str} RMSE=${rmseStr} | P(SC|SC)=${avgPscSC} P(SC|¬SC)=${avgPscNonSC} | reg-R²=${out8R2Str} | lr=${lr.toExponential(2)}`);
    }

    // yield after each training epoch so heartbeat timers and HTTP requests can proceed
    await new Promise<void>(r => setTimeout(r, 0));
  }

  const scrubMatrix = (m: number[][]) => { for (let i = 0; i < m.length; i++) for (let j = 0; j < m[i].length; j++) if (!Number.isFinite(m[i][j])) m[i][j] = 0; };
  const scrubVector = (v: number[]) => { for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) v[i] = 0; };
  for (const wMat of [
    weights.W_message, weights.W_update, weights.W_message2, weights.W_update2,
    weights.W_message3, weights.W_update3, weights.W_message4, weights.W_update4,
    weights.W_attn_query, weights.W_attn_key, weights.W_attn_query2, weights.W_attn_key2,
    weights.W_attn_query3, weights.W_attn_key3, weights.W_attn_query4, weights.W_attn_key4,
    weights.W_filter1, weights.W_filter2, weights.W_input_proj, weights.W_3body, weights.W_3body_update,
    weights.W_mlp1, weights.W_mlp2, weights.W_mlp2_var, weights.W_attn_pool,
    weights.W_elem_feat, weights.W_graph_adapt,
    weights.W_cls1,  // v15: cls head — must scrub or NaN in cls head poisons all predictions
  ]) { scrubMatrix(wMat); }
  for (const bVec of [weights.b_mlp1, weights.b_mlp2, weights.b_mlp2_var, weights.b_filter1, weights.b_filter2, weights.b_input_proj, weights.W_pressure, weights.residual_gates, weights.b_cls1, weights.W_cls2]) {
    scrubVector(bVec);
  }
  if (!Number.isFinite(weights.dense_skip_gate)) weights.dense_skip_gate = -2.0;
  if (!Number.isFinite(weights.b_cls2)) weights.b_cls2 = 0.0;

  for (const wMat of [
    weights.W_message, weights.W_update, weights.W_message2, weights.W_update2,
    weights.W_message3, weights.W_update3, weights.W_message4, weights.W_update4,
    weights.W_attn_query, weights.W_attn_key, weights.W_attn_query2, weights.W_attn_key2,
    weights.W_attn_query3, weights.W_attn_key3, weights.W_attn_query4, weights.W_attn_key4,
    weights.W_filter1, weights.W_filter2, weights.W_input_proj, weights.W_3body, weights.W_3body_update,
    weights.W_mlp1, weights.W_mlp2, weights.W_mlp2_var, weights.W_attn_pool,
    weights.W_elem_feat, weights.W_graph_adapt,
    weights.W_cls1,  // v15: cls head flat cache
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

  // Training is handled exclusively by the GCP pipeline (startup warmup or active-learning).
  // Never retrain synchronously here — that would block the request thread for 60–90 s.
  // Return fresh-initialized weights so prediction can proceed; quality will improve once
  // the GCP worker applies serialized weights via applySerializedWeights().
  if (!cachedEnsembleModels) {
    console.warn("[GNN] Model cache is cold — returning fresh-initialized weights. Awaiting GCP training result.");
    const freshModels: GNNWeights[] = Array.from({ length: ENSEMBLE_SIZE }, (_, i) =>
      initWeights(seededRandom(ENSEMBLE_SEEDS[i]))
    );
    cachedEnsembleModels = freshModels;
    modelTrainedAt = now;
  }
  return cachedEnsembleModels;
}

let heldOutValidationSet: TrainingSample[] = [];

export function splitTrainValidation(data: TrainingSample[], valFraction: number = 0.2, seed: number = 42): {
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

export const ENSEMBLE_SEEDS = [42, 7919, 104729, 15485863, 32452843];
export const BOOTSTRAP_RATIOS = [0.75, 0.80, 0.85, 0.80, 0.75];

function bootstrapSample(data: TrainingSample[], ratio: number, rng: () => number): TrainingSample[] {
  const n = Math.max(1, Math.floor(data.length * ratio));
  const sampled: TrainingSample[] = [];
  for (let i = 0; i < n; i++) {
    sampled.push(data[Math.floor(rng() * data.length)]);
  }
  return sampled;
}

export async function trainEnsemble(trainingData: TrainingSample[]): Promise<GNNWeights[]> {
  const { train, validation } = splitTrainValidation(trainingData);
  heldOutValidationSet = validation;

  const models: GNNWeights[] = [];
  for (let i = 0; i < ENSEMBLE_SIZE; i++) {
    const rng = seededRandom(ENSEMBLE_SEEDS[i]);
    const w = initWeights(rng);
    const bootstrapRng = seededRandom(ENSEMBLE_SEEDS[i] + 31);
    const bootstrapped = bootstrapSample(train, BOOTSTRAP_RATIOS[i], bootstrapRng);
    const trained = await trainGNNSurrogate(bootstrapped, w);
    models.push(trained);
  }

  if (validation.length > 0) {
    console.log(`[GNN] Train/validation split: ${train.length} train, ${validation.length} validation (${(validation.length / trainingData.length * 100).toFixed(1)}%)`);
  }

  return models;
}

export async function trainEnsembleAsync(
  trainingData: TrainingSample[],
  nModels = ENSEMBLE_SIZE,
  maxPretrainEpochs = 15,
): Promise<GNNWeights[]> {
  const { train, validation } = splitTrainValidation(trainingData);
  heldOutValidationSet = validation;

  const models: GNNWeights[] = [];
  const n = Math.min(nModels, ENSEMBLE_SIZE);
  for (let i = 0; i < n; i++) {
    const rng = seededRandom(ENSEMBLE_SEEDS[i]);
    const w = initWeights(rng);
    const bootstrapRng = seededRandom(ENSEMBLE_SEEDS[i] + 31);
    const bootstrapped = bootstrapSample(train, BOOTSTRAP_RATIOS[i], bootstrapRng);
    const trained = await trainGNNSurrogate(bootstrapped, w, maxPretrainEpochs);
    models.push(trained);
    await new Promise<void>(r => setTimeout(r, 0));
  }

  if (validation.length > 0) {
    console.log(`[GNN] Train/validation split: ${train.length} train, ${validation.length} validation (${(validation.length / trainingData.length * 100).toFixed(1)}%)`);
  }

  return models;
}

/**
 * Train one member of the GNN ensemble on a bootstrap sample of `trainingData`.
 * Called by gnn-worker-thread.ts so each of the 5 ensemble models can train
 * in a separate worker thread on the dedicated GNN GCP instance.
 */
export async function trainSingleEnsembleModel(
  trainingData: TrainingSample[],
  seed: number,
  bootstrapRatio: number,
  maxPretrainEpochs = 15,
  label?: string,
): Promise<GNNWeights> {
  const rng = seededRandom(seed);
  const w = initWeights(rng);
  const bootstrapRng = seededRandom(seed + 31);
  const bootstrapped = bootstrapSample(trainingData, bootstrapRatio, bootstrapRng);
  return trainGNNSurrogate(bootstrapped, w, maxPretrainEpochs, label);
}

export function getGNNModel(): GNNWeights {
  return getEnsembleModels()[0];
}

export function invalidateGNNModel(): void {
  cachedEnsembleModels = null;
  modelTrainedAt = 0;
  gnnPredictionCache.clear();
}

/**
 * Apply serialized weights received from the GCP worker back into the local ensemble.
 * `serialized` is the `weights` JSONB array stored in gnn_training_jobs.
 */
export function applySerializedWeights(
  serialized: GNNWeights[],
  trainingData?: { formula: string; tc: number }[],
): void {
  if (!serialized || serialized.length === 0) return;
  invalidateGNNModel();
  setCachedEnsemble(serialized, trainingData);
}

export function setCachedEnsemble(models: GNNWeights[], trainingData?: { formula: string; tc: number }[]): void {
  // Migrate each model's weights before caching — ensures W_mlp1 column count matches
  // the current GLOBAL_COMP_DIM even if the weights were trained with an older feature set.
  cachedEnsembleModels = models.map((m, i) => migrateWeights(m, seededRandom(ENSEMBLE_SEEDS[i] ?? 42)));
  modelTrainedAt = Date.now();
  if (trainingData && models.length > 0) {
    const allData = [...trainingData, ...dftTrainingDataset.map(r => ({ formula: r.formula, tc: r.tc }))];
    updateTrainingEmbeddings(allData, models[0]);
    latentEmbeddingDatasetSize = dftTrainingDataset.length;
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
  /** MAE computed exclusively on DFT-verified validation samples; null when none present */
  dftVerifiedMAE: number | null;
}

let gnnModelVersion = 0;
const gnnVersionHistory: GNNVersionRecord[] = [];
const GNN_VERSION_HISTORY_MAX = 50;

export function logGNNVersion(trigger: string, datasetSize: number, dftSamples = 0, enrichedSamples = 0): GNNVersionRecord {
  gnnModelVersion++;

  let validationSet: TrainingSample[];
  if (heldOutValidationSet.length > 0) {
    validationSet = heldOutValidationSet;
  } else {
    const allSC = SUPERCON_TRAINING_DATA.filter(e => e.isSuperconductor && e.tc > 0);
    const fallbackRng = seededRandom(12345);
    const shuffled = [...allSC];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(fallbackRng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    validationSet = shuffled.slice(-50).map(e => ({
      formula: e.formula,
      tc: e.tc,
    }));
  }

  let sumSquaredError = 0;
  let sumAbsError = 0;
  let sumActual = 0;
  let sumActualSq = 0;
  const n = validationSet.length;

  // Separate accumulators for DFT-verified samples only — the true accuracy signal.
  let dftSumAbsError = 0;
  let dftN = 0;

  for (const entry of validationSet) {
    const pred = getGNNPrediction(entry.formula);
    const actual = entry.tc;
    const predicted = pred.predictedTc;
    const error = predicted - actual;
    sumSquaredError += error * error;
    sumAbsError += Math.abs(error);
    sumActual += actual;
    sumActualSq += actual * actual;
    if (entry.dataConfidence === "dft-verified") {
      dftSumAbsError += Math.abs(error);
      dftN++;
    }
  }

  const meanActual = n > 0 ? sumActual / n : 0;
  const ssRes = sumSquaredError;
  const ssTot = n > 0 ? sumActualSq - n * meanActual * meanActual : 1;
  const r2 = ssTot > 0 ? Math.max(-1, 1 - ssRes / ssTot) : 0;
  const mae = n > 0 ? sumAbsError / n : 0;
  const rmse = n > 0 ? Math.sqrt(sumSquaredError / n) : 0;
  const dftVerifiedMAE = dftN > 0 ? dftSumAbsError / dftN : null;

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
    dftVerifiedMAE: dftVerifiedMAE != null ? Math.round(dftVerifiedMAE * 100) / 100 : null,
  };

  gnnVersionHistory.push(record);
  if (gnnVersionHistory.length > GNN_VERSION_HISTORY_MAX) {
    gnnVersionHistory.shift();
  }

  const valSource = heldOutValidationSet.length > 0 ? `held-out (${heldOutValidationSet.length})` : "fallback (tail-50 shuffled)";
  const dftMaeStr = record.dftVerifiedMAE != null ? `, DFT-MAE=${record.dftVerifiedMAE}K (n=${dftN})` : "";
  console.log(`[GNN] Version ${record.version} logged: R²=${record.r2}, MAE=${record.mae}, RMSE=${record.rmse}${dftMaeStr}, trigger=${trigger}, dataset=${datasetSize}, dft=${dftSamples}, enriched=${enrichedSamples}, validation=${valSource}`);

  return record;
}

export function getGNNVersionHistory(): GNNVersionRecord[] {
  return [...gnnVersionHistory];
}

export function getGNNModelVersion(): number {
  return gnnModelVersion;
}

/** Returns the most recent logged R² (0 if GNN has never completed a positive-R² training). */
export function getGNNLatestR2(): number {
  if (gnnVersionHistory.length === 0) return 0;
  return gnnVersionHistory[gnnVersionHistory.length - 1].r2;
}

const GNN_PRED_CACHE_MAX = 500;
const gnnPredictionCache = new Map<string, { prediction: GNNPrediction; trainedAt: number }>();

export function getGNNPrediction(formula: string, structure?: any, prototype?: string): GNNPrediction {
  const weights = getGNNModel();
  const currentTrainedAt = modelTrainedAt;
  const cacheKey = graphCacheKey(formula, prototype, structure);
  const cached = gnnPredictionCache.get(cacheKey);
  if (cached && cached.trainedAt === currentTrainedAt) {
    return cached.prediction;
  }
  const graph = prototype
    ? buildPrototypeGraph(formula, prototype)
    : buildCrystalGraph(formula, structure);
  const prediction = GNNPredict(graph, weights);
  if (gnnPredictionCache.size >= GNN_PRED_CACHE_MAX) {
    const firstKey = gnnPredictionCache.keys().next().value;
    if (firstKey !== undefined) gnnPredictionCache.delete(firstKey);
  }
  gnnPredictionCache.set(cacheKey, { prediction, trainedAt: currentTrainedAt });
  return prediction;
}



export function gnnPredictWithUncertainty(formula: string, prototype?: string, pressureGpa?: number): GNNPredictionWithUncertainty {
  const t0 = Date.now();
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

  const allMeans: number[] = [];
  const allLambdaMeans: number[] = [];
  for (let m = 0; m < ensembleModels.length; m++) {
    const modelPreds = predictions.slice(m * MC_DROPOUT_PASSES, (m + 1) * MC_DROPOUT_PASSES);
    for (const p of modelPreds) {
      allMeans.push(p.predictedTc);
      allLambdaMeans.push(p.lambda);
    }
  }
  const epistemicTcVar = allMeans.reduce((s, v) => s + (v - meanTc) ** 2, 0) / Math.max(1, allMeans.length);
  const epistemicLambdaVar = allLambdaMeans.reduce((s, v) => s + (v - meanLambda) ** 2, 0) / Math.max(1, allLambdaMeans.length);
  const ensembleUncertainty = Math.min(1.0, Math.sqrt(epistemicTcVar) / Math.max(meanTc, 1));

  const aleatoricTcVar = predictions.reduce((s, p) => s + p.predictedTcVar, 0) / predictions.length;
  const aleatoricLambdaVar = predictions.reduce((s, p) => s + p.lambdaVar, 0) / predictions.length;
  const aleatoricUncNorm = Math.min(1.0, Math.sqrt(aleatoricTcVar) / Math.max(meanTc, 1));

  let mcDropoutUncertainty = 0;
  for (let m = 0; m < ensembleModels.length; m++) {
    const modelPreds = predictions.slice(m * MC_DROPOUT_PASSES, (m + 1) * MC_DROPOUT_PASSES);
    const modelMeanTc = perModelMeans[m].tc;
    const withinTcVar = modelPreds.reduce((s, p) => s + (p.predictedTc - modelMeanTc) ** 2, 0) / modelPreds.length;
    mcDropoutUncertainty += Math.sqrt(withinTcVar) / Math.max(modelMeanTc, 1);
  }
  mcDropoutUncertainty = Math.min(1.0, mcDropoutUncertainty / ensembleModels.length);

  const totalTcVar = epistemicTcVar + aleatoricTcVar;
  const totalTcStd = Math.sqrt(totalTcVar);
  const totalLambdaVar = epistemicLambdaVar + aleatoricLambdaVar;
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

  const isHighTcCandidate = meanTc > 77;
  const wTc      = isHighTcCandidate ? 0.35 : 0.25;
  const wEnsemble = isHighTcCandidate ? 0.25 : 0.20;
  const wLatent  = 0.15;
  const wFE      = isHighTcCandidate ? 0.05 : 0.15;
  const wLambda  = isHighTcCandidate ? 0.15 : 0.10;
  const wBG      = isHighTcCandidate ? 0.05 : 0.15;

  const combinedUncertainty = Math.min(1.0,
    wTc * normalizedTcUnc +
    wFE * normalizedFeUnc +
    wLambda * normalizedLambdaUnc +
    wEnsemble * ensembleUncertainty +
    wLatent * latentDist +
    wBG * normalizedBgUnc
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
    weightProfile: {
      mode: isHighTcCandidate ? 'high-tc' : 'standard',
      tc: wTc,
      ensemble: wEnsemble,
      latent: wLatent,
      formationEnergy: wFE,
      lambda: wLambda,
      bandgap: wBG,
    },
  };

  const meanOmegaLog = predictions.reduce((acc, p) => acc + p.omegaLog, 0) / predictions.length;

  const s = (v: number, fb = 0) => Number.isFinite(v) ? v : fb;
  const result = {
    tc: Math.round(s(meanTc) * 10) / 10,
    omegaLog: Math.round(s(meanOmegaLog) * 10) / 10,
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
    tcCI95: [Math.round(s(tcCI95Lower) * 10) / 10, Math.round(s(tcCI95Upper) * 10) / 10] as [number, number],
    lambdaCI95: [Math.round(s(lambdaCI95Lower) * 1000) / 1000, Math.round(s(lambdaCI95Upper) * 1000) / 1000] as [number, number],
    epistemicUncertainty: Math.round(s(Math.sqrt(epistemicTcVar)) * 100) / 100,
    aleatoricUncertainty: Math.round(s(Math.sqrt(aleatoricTcVar)) * 100) / 100,
    totalStd: Math.round(s(totalTcStd) * 100) / 100,
  };
  const elapsed = Date.now() - t0;
  try { import("./model-diagnostics").then(m => m.recordInferenceTime("gnn", elapsed)).catch(() => {}); } catch {}
  return result;
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
const dftFormulaIndex = new Map<string, number>();
const datasetGrowthHistory: DFTDatasetGrowthEntry[] = [];
let latentEmbeddingDatasetSize = 0;

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
  const existingIdx = dftFormulaIndex.get(record.formula);
  if (existingIdx !== undefined) {
    const existing = dftTrainingDataset[existingIdx];
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

  const newIdx = dftTrainingDataset.length;
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
  dftFormulaIndex.set(record.formula, newIdx);

  datasetGrowthHistory.push({
    timestamp: Date.now(),
    size: dftTrainingDataset.length,
    source: record.source,
  });

  if (datasetGrowthHistory.length > 200) {
    datasetGrowthHistory.splice(0, datasetGrowthHistory.length - 200);
  }

  if (latentEmbeddingDatasetSize > 0 &&
      dftTrainingDataset.length > latentEmbeddingDatasetSize * 1.05) {
    scheduleLatentRefresh();
  }

  return true;
}

let latentRefreshPending = false;
function scheduleLatentRefresh(): void {
  if (latentRefreshPending) return;
  latentRefreshPending = true;
  setImmediate(() => {
    try {
      if (cachedEnsembleModels && cachedEnsembleModels.length > 0) {
        const allData = [
          ...SUPERCON_TRAINING_DATA.filter(e => e.isSuperconductor).map(e => ({ formula: e.formula, tc: e.tc })),
          ...dftTrainingDataset.map(r => ({ formula: r.formula, tc: r.tc })),
        ];
        updateTrainingEmbeddings(allData, cachedEnsembleModels[0]);
        latentEmbeddingDatasetSize = dftTrainingDataset.length;
      }
    } catch {}
    latentRefreshPending = false;
  });
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

// Worker threads load this module for trainSingleEnsembleModel only.
// The startup warm-up (DB queries, MP fetches, ESM imports) must not run in them.
if (isMainThread) setTimeout(async () => {
  // When GNN training is offloaded to GCP, skip local startup training entirely.
  // The GCP worker handles all ensemble training; local server just applies weights.
  if (process.env.OFFLOAD_GNN_TO_GCP === "true") {
    // Only log on the local server (not on GCP where this env var is set by index.ts)
    if (!process.env.QE_BIN_DIR) {
      console.log("[GNN] OFFLOAD_GNN_TO_GCP=true — GCP handles GNN training, skipping local startup");
    }
    return;
  }
  try {
    // Build a Tc + lambda lookup from SUPERCON for cross-referencing with MP data
    const superconTcMap = new Map<string, { tc: number; lambda?: number }>();
    for (const e of SUPERCON_TRAINING_DATA) {
      superconTcMap.set(e.formula, { tc: e.tc, lambda: (e as any).lambda ?? undefined });
    }
    const superconFormulas = new Set(SUPERCON_TRAINING_DATA.map(e => e.formula));

    // --- Seed dftTrainingDataset with all SUPERCON superconductors ---
    let superconSeeded = 0;
    for (const e of SUPERCON_TRAINING_DATA) {
      if (e.isSuperconductor && e.tc > 0) {
        const extras: Record<string, any> = {};
        if ((e as any).lambda != null) extras.lambda = (e as any).lambda;
        if (addDFTTrainingResult({ formula: e.formula, tc: e.tc, source: "supercon", ...extras })) superconSeeded++;
      }
    }

    // Helper: build TrainingSample[] from SUPERCON + cached formation energies
    const buildSuperconSamples = async (): Promise<TrainingSample[]> => {
      const { fetchCachedFormationEnergies } = await import("./materials-project-client");
      const cachedFE = await fetchCachedFormationEnergies(SUPERCON_TRAINING_DATA.map(e => e.formula));
      return SUPERCON_TRAINING_DATA.map(e => {
        const proto = matchPrototype(e.formula);
        return {
          formula: e.formula,
          tc: e.tc,
          lambda: (e as any).lambda ?? undefined,
          formationEnergy: cachedFE.get(e.formula),
          structure: proto ? {
            spaceGroup: proto.spaceGroup,
            crystalSystem: proto.crystalSystem,
            dimensionality: proto.dimensionality,
          } : undefined,
          prototype: proto?.prototype,
        };
      });
    }

    // Helper: merge new MP records into dftTrainingDataset and extend a training set
    const mergeMPRecords = (
      mpRecords: { formula: string; bandGap: number | null; formationEnergy: number | null }[],
      base: TrainingSample[],
    ): { merged: TrainingSample[]; seeded: number } => {
      let seeded = 0;
      const seen = new Set(base.map(t => t.formula));
      for (const mp of mpRecords) {
        const known = superconTcMap.get(mp.formula);
        const tc = known?.tc ?? 0;
        if (addDFTTrainingResult({
          formula: mp.formula, tc, source: "external",
          bandGap: mp.bandGap, formationEnergy: mp.formationEnergy,
          ...(known?.lambda != null ? { lambda: known.lambda } : {}),
        })) seeded++;
      }
      // Add DFT dataset entries not already in base — include tc=0 as contrast examples
      // (non-SCs and failed candidates teach the GNN that most metals don't superconduct).
      // Cap at 1.5× SC count to avoid drowning out the superconductor signal.
      const maxContrast = Math.ceil(base.length * 1.5);
      const extra: TrainingSample[] = [];
      for (const rec of dftTrainingDataset) {
        if (extra.length >= maxContrast) break;
        if (seen.has(rec.formula) || superconFormulas.has(rec.formula)) continue;
        extra.push({
          formula: rec.formula,
          tc: Math.max(0, rec.tc ?? 0),
          formationEnergy: rec.formationEnergy ?? undefined,
          bandgap: rec.bandGap ?? undefined,
          phononStable: rec.phononStable,
        });
        seen.add(rec.formula);
      }
      return { merged: [...base, ...extra], seeded };
    }

    // =====================================================================
    // Phase 1: Train on full SUPERCON dataset (with cached formation energies)
    // Train a single model with reduced FE pretraining (~8s) so the event
    // loop stays responsive. GCP will build the full 5-model ensemble.
    // =====================================================================
    const gcpMode = process.env.OFFLOAD_GNN_TO_GCP === "true";
    const superconSamples = await buildSuperconSamples();
    let currentTrainingData = superconSamples;

    // Quick startup model: 1 member, 5 FE epochs → ~8s instead of ~85s.
    // GCP will replace it with a full 5-model ensemble shortly after.
    let ensembleModels = await trainEnsembleAsync(currentTrainingData, 1, 5);
    invalidateGNNModel();
    setCachedEnsemble(ensembleModels, currentTrainingData.map(t => ({ formula: t.formula, tc: t.tc })));
    const v1 = logGNNVersion("startup-supercon", currentTrainingData.length, dftTrainingDataset.length, 0);
    console.log(`[GNN] Phase 1 (SUPERCON N=${currentTrainingData.length}): R²=${v1.r2}`);

    // If GCP offloading is active, skip phases 2-3 — GCP will train a full
    // ensemble with the same (and richer) dataset. Blocking the event loop
    // for 3+ extra minutes adds no value when GCP returns results in ~3 min.
    if (gcpMode) {
      console.log(`[GNN] GCP mode — skipping startup phases 2-3 (GCP will train full ensemble)`);
      console.log(`[GNN] Startup warm-up complete: N=${currentTrainingData.length}, SUPERCON=${superconSeeded}, dftDataset=${dftTrainingDataset.length}`);
      return;
    }

    // Yield so other event loop work (HTTP, DFT, etc.) can proceed between phases
    await new Promise(resolve => setTimeout(resolve, 300));

    // =====================================================================
    // Phase 2: Pull MP batch 1 (DB cache + up to 500 from API) if R² ≤ 0
    // =====================================================================
    if (v1.r2 <= 0) {
      let mpSeeded1 = 0;
      try {
        const { fetchGNNSeedData } = await import("./materials-project-client");
        const mpRecords1 = await fetchGNNSeedData(true); // cacheOnly — never block startup with 772-formula network fetch
        const result1 = mergeMPRecords(mpRecords1, currentTrainingData);
        currentTrainingData = result1.merged;
        mpSeeded1 = result1.seeded;
      } catch (mpErr: any) {
        console.warn(`[GNN] Phase 2 MP fetch failed: ${mpErr?.message?.slice(0, 100)}`);
      }

      await new Promise(resolve => setTimeout(resolve, 300));
      ensembleModels = await trainEnsembleAsync(currentTrainingData);
      invalidateGNNModel();
      setCachedEnsemble(ensembleModels, currentTrainingData.map(t => ({ formula: t.formula, tc: t.tc })));
      const v2 = logGNNVersion("startup-mp-batch1", currentTrainingData.length, dftTrainingDataset.length, mpSeeded1);
      console.log(`[GNN] Phase 2 (MP batch1 +${mpSeeded1}, N=${currentTrainingData.length}): R²=${v2.r2}`);

      await new Promise(resolve => setTimeout(resolve, 300));

      // ==================================================================
      // Phase 3: Fetch next MP batch (skip=1000) if R² still ≤ 0
      // Skip retraining entirely if no new samples are added — avoids
      // wasting 60-90s retraining on exactly the same dataset as Phase 2.
      // ==================================================================
      if (v2.r2 <= 0) {
        let mpSeeded2 = 0;
        const prevN = currentTrainingData.length;
        try {
          const { fetchMPBatchFromAPI } = await import("./materials-project-client");
          // Use skip=1000 to get a batch the GCP progressive fetcher hasn't cached yet
          const mpRecords2 = await fetchMPBatchFromAPI(500, 1000);
          const result2 = mergeMPRecords(mpRecords2, currentTrainingData);
          currentTrainingData = result2.merged;
          mpSeeded2 = result2.seeded;
        } catch (mpErr: any) {
          console.warn(`[GNN] Phase 3 MP fetch failed: ${mpErr?.message?.slice(0, 100)}`);
        }

        if (currentTrainingData.length > prevN) {
          await new Promise(resolve => setTimeout(resolve, 300));
          ensembleModels = await trainEnsembleAsync(currentTrainingData);
          invalidateGNNModel();
          setCachedEnsemble(ensembleModels, currentTrainingData.map(t => ({ formula: t.formula, tc: t.tc })));
          const v3 = logGNNVersion("startup-mp-batch2", currentTrainingData.length, dftTrainingDataset.length, mpSeeded1 + mpSeeded2);
          console.log(`[GNN] Phase 3 (MP batch2 +${mpSeeded2}, N=${currentTrainingData.length}): R²=${v3.r2}`);
        } else {
          console.log(`[GNN] Phase 3 skipped — no new samples added (mpSeeded=${mpSeeded2}, N=${currentTrainingData.length} unchanged)`);
        }
      }
    }

    console.log(`[GNN] Startup warm-up complete: N=${currentTrainingData.length}, SUPERCON=${superconSeeded}, dftDataset=${dftTrainingDataset.length}`);
  } catch (e: any) {
    console.error(`[GNN] Pre-warm failed: ${e?.message?.slice(0, 200)}`);
  }
}, 190000);  // T+190s — 60s after CrystalVAE at T+130s; benchmark runs at T+250s
