import { getTrainingData, type CrystalStructureEntry } from "./crystal-structure-dataset";
import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";

const CRYSTAL_SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"] as const;
const TOP_ELEMENTS = [
  "H","Li","Be","B","C","N","O","F","Na","Mg",
  "Al","Si","P","S","Cl","K","Ca","Sc","Ti","V",
  "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As",
  "Se","Sr","Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag",
] as const;

const T_STEPS = 50;
const BETA_MIN = 0.0001;
const BETA_MAX = 0.02;
const FEATURE_DIM = 53;
const TIME_EMBED_DIM = 16;
const HIDDEN_DIM = 64;
const TRAIN_EPOCHS = 100;
const LEARNING_RATE = 0.003;

interface DiffusionWeights {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
}

export interface GeneratedCrystalStructure {
  formula: string;
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  crystalSystem: string;
  compositionVector: number[];
  noveltyScore: number;
  confidence: number;
  generationMethod: string;
}

interface DiffusionModelStats {
  trained: boolean;
  trainingEpochs: number;
  trainingLoss: number[];
  datasetSize: number;
  samplesGenerated: number;
  acceptanceRate: number;
  avgNovelty: number;
  featureDim: number;
  timesteps: number;
}

let betas: number[] = [];
let alphas: number[] = [];
let alphasCumprod: number[] = [];
let sqrtAlphasCumprod: number[] = [];
let sqrtOneMinusAlphasCumprod: number[] = [];

let scoreNetwork: DiffusionWeights | null = null;
let trained = false;
let trainingLossHistory: number[] = [];
let totalSamplesGenerated = 0;
let totalAccepted = 0;
let noveltyScores: number[] = [];
let trainingFeatures: number[][] = [];

function initSchedule() {
  betas = [];
  alphas = [];
  alphasCumprod = [];
  sqrtAlphasCumprod = [];
  sqrtOneMinusAlphasCumprod = [];

  for (let t = 0; t < T_STEPS; t++) {
    const beta = BETA_MIN + (BETA_MAX - BETA_MIN) * (t / (T_STEPS - 1));
    betas.push(beta);
    alphas.push(1 - beta);
  }

  let cumprod = 1.0;
  for (let t = 0; t < T_STEPS; t++) {
    cumprod *= alphas[t];
    alphasCumprod.push(cumprod);
    sqrtAlphasCumprod.push(Math.sqrt(cumprod));
    sqrtOneMinusAlphasCumprod.push(Math.sqrt(1 - cumprod));
  }
}

function gaussRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
}

function sinusoidalTimeEmbedding(t: number): number[] {
  const emb: number[] = [];
  for (let i = 0; i < TIME_EMBED_DIM; i++) {
    const freq = 1.0 / Math.pow(10000, (2 * Math.floor(i / 2)) / TIME_EMBED_DIM);
    if (i % 2 === 0) {
      emb.push(Math.sin(t * freq));
    } else {
      emb.push(Math.cos(t * freq));
    }
  }
  return emb;
}

function crystalSystemOneHot(system: string): number[] {
  const vec = new Array(7).fill(0);
  const idx = CRYSTAL_SYSTEMS.indexOf(system as any);
  if (idx >= 0) vec[idx] = 1;
  return vec;
}

function compositionPreferenceVector(formula: string): number[] {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  const total = Object.values(counts).reduce((s, c) => s + c, 0) || 1;
  const vec: number[] = [];
  for (const el of TOP_ELEMENTS) {
    vec.push((counts[el] || 0) / total);
  }
  return vec;
}

function entryToFeatureVector(entry: CrystalStructureEntry): number[] {
  const latticeNorm = [
    entry.lattice.a / 15.0,
    entry.lattice.b / 15.0,
    entry.lattice.c / 30.0,
    entry.lattice.alpha / 180.0,
    entry.lattice.beta / 180.0,
    entry.lattice.gamma / 180.0,
  ];
  const csOneHot = crystalSystemOneHot(entry.crystalSystem);
  const compVec = compositionPreferenceVector(entry.formula);
  const features = [...latticeNorm, ...csOneHot, ...compVec];
  while (features.length < FEATURE_DIM) features.push(0);
  return features.slice(0, FEATURE_DIM);
}

function featureVectorToProperties(vec: number[]): {
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  crystalSystem: string;
  compositionVector: number[];
  formula: string;
} {
  const a = Math.max(2.0, Math.min(15.0, vec[0] * 15.0));
  const b = Math.max(2.0, Math.min(15.0, vec[1] * 15.0));
  const c = Math.max(2.0, Math.min(30.0, vec[2] * 30.0));
  const alpha = Math.max(60, Math.min(120, vec[3] * 180.0));
  const beta = Math.max(60, Math.min(120, vec[4] * 180.0));
  const gamma = Math.max(60, Math.min(120, vec[5] * 180.0));

  const csLogits = vec.slice(6, 13);
  let maxIdx = 0;
  let maxVal = -Infinity;
  for (let i = 0; i < csLogits.length; i++) {
    if (csLogits[i] > maxVal) { maxVal = csLogits[i]; maxIdx = i; }
  }
  const crystalSystem = CRYSTAL_SYSTEMS[maxIdx];

  const compVec = vec.slice(13, 53);
  const formula = decodeCompositionVector(compVec);

  return {
    lattice: {
      a: Math.round(a * 100) / 100,
      b: Math.round(b * 100) / 100,
      c: Math.round(c * 100) / 100,
      alpha: Math.round(alpha * 10) / 10,
      beta: Math.round(beta * 10) / 10,
      gamma: Math.round(gamma * 10) / 10,
    },
    crystalSystem,
    compositionVector: compVec,
    formula,
  };
}

function decodeCompositionVector(compVec: number[]): string {
  const threshold = 0.03;
  const parts: { el: string; count: number }[] = [];

  for (let i = 0; i < compVec.length && i < TOP_ELEMENTS.length; i++) {
    if (compVec[i] > threshold) {
      const count = Math.round(compVec[i] * 10);
      if (count > 0) {
        parts.push({ el: TOP_ELEMENTS[i], count });
      }
    }
  }

  if (parts.length === 0) {
    const sorted = compVec
      .map((v, i) => ({ v, i }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 2);
    for (const { v, i } of sorted) {
      if (i < TOP_ELEMENTS.length) {
        parts.push({ el: TOP_ELEMENTS[i], count: Math.max(1, Math.round(v * 10)) });
      }
    }
  }

  return parts.map(p => p.count === 1 ? p.el : `${p.el}${p.count}`).join("");
}

function initWeights(inputDim: number, hiddenDim: number, outputDim: number): DiffusionWeights {
  const scale1 = Math.sqrt(2.0 / inputDim);
  const scale2 = Math.sqrt(2.0 / hiddenDim);

  const w1: number[][] = [];
  for (let i = 0; i < hiddenDim; i++) {
    const row: number[] = [];
    for (let j = 0; j < inputDim; j++) {
      row.push(gaussRandom() * scale1);
    }
    w1.push(row);
  }
  const b1 = new Array(hiddenDim).fill(0);

  const w2: number[][] = [];
  for (let i = 0; i < outputDim; i++) {
    const row: number[] = [];
    for (let j = 0; j < hiddenDim; j++) {
      row.push(gaussRandom() * scale2);
    }
    w2.push(row);
  }
  const b2 = new Array(outputDim).fill(0);

  return { w1, b1, w2, b2 };
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function forward(weights: DiffusionWeights, input: number[]): { output: number[]; hidden: number[] } {
  const hidden: number[] = [];
  for (let i = 0; i < weights.w1.length; i++) {
    let sum = weights.b1[i];
    for (let j = 0; j < input.length; j++) {
      sum += weights.w1[i][j] * input[j];
    }
    hidden.push(relu(sum));
  }

  const output: number[] = [];
  for (let i = 0; i < weights.w2.length; i++) {
    let sum = weights.b2[i];
    for (let j = 0; j < hidden.length; j++) {
      sum += weights.w2[i][j] * hidden[j];
    }
    output.push(sum);
  }

  return { output, hidden };
}

function predictNoise(weights: DiffusionWeights, noisyFeatures: number[], t: number): number[] {
  const timeEmb = sinusoidalTimeEmbedding(t / T_STEPS);
  const input = [...noisyFeatures, ...timeEmb];
  const { output } = forward(weights, input);
  return output;
}

function addNoise(x: number[], t: number): { noisy: number[]; noise: number[] } {
  const noise: number[] = [];
  const noisy: number[] = [];
  for (let i = 0; i < x.length; i++) {
    const eps = gaussRandom();
    noise.push(eps);
    noisy.push(sqrtAlphasCumprod[t] * x[i] + sqrtOneMinusAlphasCumprod[t] * eps);
  }
  return { noisy, noise };
}

const MINI_BATCH_SIZE = 32;

function trainStep(
  weights: DiffusionWeights,
  dataPoints: number[][],
  lr: number
): number {
  let totalLoss = 0;

  const gradW1: number[][] = weights.w1.map(row => new Array(row.length).fill(0));
  const gradB1: number[] = new Array(weights.b1.length).fill(0);
  const gradW2: number[][] = weights.w2.map(row => new Array(row.length).fill(0));
  const gradB2: number[] = new Array(weights.b2.length).fill(0);

  const batchSize = Math.min(MINI_BATCH_SIZE, dataPoints.length);
  const batch: number[][] = [];
  for (let i = 0; i < batchSize; i++) {
    batch.push(dataPoints[Math.floor(Math.random() * dataPoints.length)]);
  }

  for (const x of batch) {
    const t = Math.floor(Math.random() * T_STEPS);
    const { noisy, noise } = addNoise(x, t);

    const timeEmb = sinusoidalTimeEmbedding(t / T_STEPS);
    const input = [...noisy, ...timeEmb];

    const { output: predictedNoise, hidden } = forward(weights, input);

    let sampleLoss = 0;
    const dOutput: number[] = [];
    for (let i = 0; i < predictedNoise.length; i++) {
      const diff = predictedNoise[i] - noise[i];
      sampleLoss += diff * diff;
      dOutput.push(2 * diff / predictedNoise.length);
    }
    totalLoss += sampleLoss / predictedNoise.length;

    const dHidden: number[] = new Array(hidden.length).fill(0);
    for (let i = 0; i < weights.w2.length; i++) {
      gradB2[i] += dOutput[i];
      for (let j = 0; j < hidden.length; j++) {
        gradW2[i][j] += dOutput[i] * hidden[j];
        dHidden[j] += weights.w2[i][j] * dOutput[i];
      }
    }

    for (let i = 0; i < weights.w1.length; i++) {
      const reluGrad = hidden[i] > 0 ? 1 : 0;
      const dPre = dHidden[i] * reluGrad;
      gradB1[i] += dPre;
      for (let j = 0; j < input.length; j++) {
        gradW1[i][j] += dPre * input[j];
      }
    }
  }

  const n = batchSize;
  const clipVal = 1.0;

  for (let i = 0; i < weights.w1.length; i++) {
    for (let j = 0; j < weights.w1[i].length; j++) {
      const g = Math.max(-clipVal, Math.min(clipVal, gradW1[i][j] / n));
      weights.w1[i][j] -= lr * g;
    }
    gradB1[i] = Math.max(-clipVal, Math.min(clipVal, gradB1[i] / n));
    weights.b1[i] -= lr * gradB1[i];
  }
  for (let i = 0; i < weights.w2.length; i++) {
    for (let j = 0; j < weights.w2[i].length; j++) {
      const g = Math.max(-clipVal, Math.min(clipVal, gradW2[i][j] / n));
      weights.w2[i][j] -= lr * g;
    }
    gradB2[i] = Math.max(-clipVal, Math.min(clipVal, gradB2[i] / n));
    weights.b2[i] -= lr * gradB2[i];
  }

  return totalLoss / n;
}

function sampleReverse(
  weights: DiffusionWeights,
  conditions?: { crystalSystem?: string; elements?: string[] }
): number[] {
  let x: number[] = [];
  for (let i = 0; i < FEATURE_DIM; i++) {
    x.push(gaussRandom());
  }

  for (let t = T_STEPS - 1; t >= 0; t--) {
    const predictedNoise = predictNoise(weights, x, t);

    const alpha = alphas[t];
    const alphaCumprod = alphasCumprod[t];
    const beta = betas[t];
    const sqrtAlpha = Math.sqrt(alpha);
    const sqrtOneMinusAlphaCum = sqrtOneMinusAlphasCumprod[t];

    for (let i = 0; i < x.length; i++) {
      const mean = (1.0 / sqrtAlpha) * (x[i] - (beta / sqrtOneMinusAlphaCum) * predictedNoise[i]);
      const variance = t > 0 ? beta : 0;
      x[i] = mean + Math.sqrt(variance) * gaussRandom();
    }

    if (conditions) {
      if (conditions.crystalSystem) {
        const csIdx = CRYSTAL_SYSTEMS.indexOf(conditions.crystalSystem as any);
        if (csIdx >= 0) {
          for (let i = 0; i < 7; i++) {
            x[6 + i] = i === csIdx ? 0.8 + x[6 + i] * 0.2 : x[6 + i] * 0.1;
          }
        }
      }
      if (conditions.elements && conditions.elements.length > 0) {
        for (const el of conditions.elements) {
          const elIdx = TOP_ELEMENTS.indexOf(el as any);
          if (elIdx >= 0) {
            x[13 + elIdx] = Math.max(x[13 + elIdx], 0.1 + Math.abs(gaussRandom()) * 0.05);
          }
        }
      }
    }
  }

  for (let i = 0; i < 6; i++) {
    x[i] = Math.max(0.05, Math.min(1.0, x[i]));
  }
  for (let i = 13; i < FEATURE_DIM; i++) {
    x[i] = Math.max(0, x[i]);
  }

  return x;
}

function computeNoveltyScore(vec: number[]): number {
  if (trainingFeatures.length === 0) return 1.0;

  let minDist = Infinity;
  for (const trainVec of trainingFeatures) {
    let dist = 0;
    for (let i = 0; i < vec.length && i < trainVec.length; i++) {
      const diff = vec[i] - trainVec[i];
      dist += diff * diff;
    }
    dist = Math.sqrt(dist);
    if (dist < minDist) minDist = dist;
  }

  const maxDist = Math.sqrt(FEATURE_DIM);
  return Math.min(1.0, minDist / (maxDist * 0.3));
}

function isValidGeneration(props: ReturnType<typeof featureVectorToProperties>): boolean {
  const { lattice, formula } = props;

  if (lattice.a < 2.0 || lattice.b < 2.0 || lattice.c < 2.0) return false;
  if (lattice.a > 15.0 || lattice.b > 15.0 || lattice.c > 30.0) return false;

  if (!formula || formula.length < 1) return false;

  const elemCount = (formula.match(/[A-Z]/g) || []).length;
  if (elemCount < 1 || elemCount > 6) return false;

  const vol = lattice.a * lattice.b * lattice.c;
  if (vol < 10 || vol > 5000) return false;

  return true;
}

export async function initDiffusionModel(): Promise<void> {
  console.log("[CrystalDiffusion] Initializing DDPM model...");

  initSchedule();

  const dataset = getTrainingData();
  if (dataset.length === 0) {
    console.log("[CrystalDiffusion] No training data available");
    return;
  }

  trainingFeatures = dataset.map(entry => entryToFeatureVector(entry));
  console.log(`[CrystalDiffusion] Prepared ${trainingFeatures.length} training samples, feature_dim=${FEATURE_DIM}`);

  const inputDim = FEATURE_DIM + TIME_EMBED_DIM;
  scoreNetwork = initWeights(inputDim, HIDDEN_DIM, FEATURE_DIM);

  trainingLossHistory = [];
  let lr = LEARNING_RATE;

  const yield_ = () => new Promise<void>(r => setTimeout(r, 5));
  for (let epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {
    const loss = trainStep(scoreNetwork, trainingFeatures, lr);
    trainingLossHistory.push(loss);

    if (epoch > 0 && epoch % 25 === 0) {
      lr *= 0.9;
    }

    await yield_(); // yield every epoch with real delay so the event loop isn't starved

    if (epoch % 25 === 0) {
      console.log(`[CrystalDiffusion] Epoch ${epoch}/${TRAIN_EPOCHS}, loss=${loss.toFixed(6)}, lr=${lr.toFixed(6)}`);
    }
  }

  trained = true;
  const finalLoss = trainingLossHistory[trainingLossHistory.length - 1];
  console.log(`[CrystalDiffusion] Training complete. Final loss=${finalLoss.toFixed(6)}, dataset=${dataset.length}`);
}

export function sampleStructures(
  count: number = 5,
  conditions?: { crystalSystem?: string; elements?: string[] }
): GeneratedCrystalStructure[] {
  if (!trained || !scoreNetwork) {
    console.log("[CrystalDiffusion] Model not trained, initializing...");
    initDiffusionModel();
  }
  if (!scoreNetwork) return [];

  const results: GeneratedCrystalStructure[] = [];
  let attempts = 0;
  const maxAttempts = count * 5;

  while (results.length < count && attempts < maxAttempts) {
    attempts++;

    const sampledVec = sampleReverse(scoreNetwork, conditions);
    const props = featureVectorToProperties(sampledVec);

    totalSamplesGenerated++;

    if (!isValidGeneration(props)) continue;

    const novelty = computeNoveltyScore(sampledVec);
    noveltyScores.push(novelty);
    totalAccepted++;

    results.push({
      formula: props.formula,
      lattice: props.lattice,
      crystalSystem: props.crystalSystem,
      compositionVector: props.compositionVector,
      noveltyScore: Math.round(novelty * 1000) / 1000,
      confidence: Math.round((1 - (trainingLossHistory[trainingLossHistory.length - 1] || 1)) * 100) / 100,
      generationMethod: "ddpm-diffusion",
    });
  }

  return results;
}

export function getDiffusionModelStats(): DiffusionModelStats {
  const avgNovelty = noveltyScores.length > 0
    ? noveltyScores.reduce((s, n) => s + n, 0) / noveltyScores.length
    : 0;

  return {
    trained,
    trainingEpochs: trainingLossHistory.length,
    trainingLoss: trainingLossHistory.length > 20
      ? trainingLossHistory.filter((_, i) => i % Math.ceil(trainingLossHistory.length / 20) === 0)
      : trainingLossHistory,
    datasetSize: trainingFeatures.length,
    samplesGenerated: totalSamplesGenerated,
    acceptanceRate: totalSamplesGenerated > 0 ? totalAccepted / totalSamplesGenerated : 0,
    avgNovelty,
    featureDim: FEATURE_DIM,
    timesteps: T_STEPS,
  };
}

export function runTrainedDiffusionCycle(count: number = 3): GeneratedCrystalStructure[] {
  if (!trained) return [];
  return sampleStructures(count);
}
