import { getTrainingData, type CrystalStructureEntry } from "./crystal-structure-dataset";
import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";
import { buildGraphFromStructure, getGraphFeatureVector } from "./crystal-graph-builder";

const LATENT_DIM = 32;
const HIDDEN_DIM = 64;
const NUM_EPOCHS = 300;
const LEARNING_RATE = 0.002;
const MOMENTUM = 0.9;
const FREE_BITS = 0.1;
const BETA_CYCLES = 4;
const BETA_MAX = 0.8;
const WARMUP_EPOCHS = 60;
const KL_MIN_TARGET = 2.0;
const KL_PULL_STRENGTH = 0.05;
const ENCODER_CATCHUP_INTERVAL = 25;
const ENCODER_CATCHUP_STEPS = 5;

const CRYSTAL_SYSTEMS = ["cubic", "tetragonal", "orthorhombic", "hexagonal", "rhombohedral", "monoclinic", "triclinic"] as const;
const TOP_SPACEGROUPS = [225, 229, 227, 221, 223, 191, 194, 139, 129, 123, 99, 141, 62, 47, 166, 216, 217, 230, 12, 15] as const;
const SG_SYMBOLS: Record<number, string> = {
  225: "Fm-3m", 229: "Im-3m", 227: "Fd-3m", 221: "Pm-3m", 223: "Pm-3n",
  191: "P6/mmm", 194: "P6_3/mmc", 139: "I4/mmm", 129: "P4/nmm", 123: "P4/mmm",
  99: "P4mm", 141: "I4_1/amd", 62: "Pnma", 47: "Pmmm", 166: "R-3m",
  216: "F-43m", 217: "I-43m", 230: "Ia-3d", 12: "C2/m", 15: "C2/c",
};

const TOP_ELEMENTS = [
  "H","Li","Be","B","C","N","O","F","Na","Mg",
  "Al","Si","P","S","Cl","K","Ca","Sc","Ti","V",
  "Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As",
  "Se","Br","Sr","Y","Zr","Nb","Mo","Ru","Rh","Pd",
] as const;

interface MLPWeights {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
}

interface MLPMomentum {
  dw1: number[][];
  db1: number[];
  dw2: number[][];
  db2: number[];
}

interface EncoderOutput {
  mu: number[];
  logSigma: number[];
}

interface VAEModel {
  encoder: MLPWeights;
  decoderLattice: MLPWeights;
  decoderCrystalSystem: MLPWeights;
  decoderSpacegroup: MLPWeights;
  decoderPrototype: MLPWeights;
  decoderAtomCount: MLPWeights;
  decoderComposition: MLPWeights;
  inputDim: number;
  prototypes: string[];
}

interface TrainingStats {
  epoch: number;
  totalLoss: number;
  reconLoss: number;
  klLoss: number;
  beta: number;
}

let model: VAEModel | null = null;
let trainingHistory: TrainingStats[] = [];
let totalGenerations = 0;
let noveltyScores: number[] = [];
let encodedTrainingData: { formula: string; z: number[] }[] = [];
let trained = false;

function initWeights(rows: number, cols: number, scale: number = 0.1): number[][] {
  const w: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((Math.random() * 2 - 1) * scale * Math.sqrt(2.0 / (rows + cols)));
    }
    w.push(row);
  }
  return w;
}

function initBias(size: number): number[] {
  return new Array(size).fill(0);
}

function createMLP(inputDim: number, hiddenDim: number, outputDim: number): MLPWeights {
  return {
    w1: initWeights(inputDim, hiddenDim),
    b1: initBias(hiddenDim),
    w2: initWeights(hiddenDim, outputDim),
    b2: initBias(outputDim),
  };
}

function createMomentum(mlp: MLPWeights): MLPMomentum {
  return {
    dw1: mlp.w1.map(r => r.map(() => 0)),
    db1: mlp.b1.map(() => 0),
    dw2: mlp.w2.map(r => r.map(() => 0)),
    db2: mlp.b2.map(() => 0),
  };
}

function relu(x: number): number { return Math.max(0, x); }
function reluDeriv(x: number): number { return x > 0 ? 1 : 0; }

function forwardMLP(mlp: MLPWeights, input: number[]): { hidden: number[]; output: number[] } {
  const hidden: number[] = [];
  for (let j = 0; j < mlp.b1.length; j++) {
    let s = mlp.b1[j];
    for (let i = 0; i < input.length; i++) {
      s += input[i] * mlp.w1[i][j];
    }
    hidden.push(relu(s));
  }
  const output: number[] = [];
  for (let j = 0; j < mlp.b2.length; j++) {
    let s = mlp.b2[j];
    for (let i = 0; i < hidden.length; i++) {
      s += hidden[i] * mlp.w2[i][j];
    }
    output.push(s);
  }
  return { hidden, output };
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map(e => e / sum);
}

function encodeEntry(entry: CrystalStructureEntry): number[] {
  const compFeatures = computeCompositionFeatures(entry.formula);
  const compVec = compositionFeatureVector(compFeatures);

  let graphVec: number[];
  try {
    const graph = buildGraphFromStructure(entry);
    graphVec = getGraphFeatureVector(graph);
  } catch {
    graphVec = new Array(35).fill(0);
  }

  const latticeNorm = [
    entry.lattice.a / 15.0,
    entry.lattice.b / 15.0,
    entry.lattice.c / 35.0,
    entry.lattice.alpha / 180.0,
    entry.lattice.beta / 180.0,
    entry.lattice.gamma / 180.0,
  ];

  const csOneHot = CRYSTAL_SYSTEMS.map(cs => cs === entry.crystalSystem ? 1.0 : 0.0);

  const sgIdx = TOP_SPACEGROUPS.indexOf(entry.spacegroup as any);
  const sgOneHot = TOP_SPACEGROUPS.map((_, i) => i === sgIdx ? 1.0 : 0.0);

  return [...compVec, ...graphVec, ...latticeNorm, ...csOneHot, ...sgOneHot];
}

function reparameterize(mu: number[], logSigma: number[]): number[] {
  return mu.map((m, i) => {
    const eps = gaussianRandom();
    return m + Math.exp(logSigma[i]) * eps;
  });
}

function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function klDivergence(mu: number[], logSigma: number[]): number {
  let kl = 0;
  for (let i = 0; i < mu.length; i++) {
    const sigma2 = Math.exp(2 * logSigma[i]);
    kl += sigma2 + mu[i] * mu[i] - 1 - 2 * logSigma[i];
  }
  return 0.5 * kl;
}

function klDivergencePerDim(mu: number[], logSigma: number[]): number[] {
  return mu.map((m, i) => {
    const sigma2 = Math.exp(2 * logSigma[i]);
    return 0.5 * (sigma2 + m * m - 1 - 2 * logSigma[i]);
  });
}

function klWithFreeBits(mu: number[], logSigma: number[]): number {
  const perDim = klDivergencePerDim(mu, logSigma);
  let total = 0;
  for (const kld of perDim) {
    total += Math.max(FREE_BITS, kld);
  }
  return total;
}

function mseLoss(predicted: number[], target: number[]): number {
  let sum = 0;
  const len = Math.min(predicted.length, target.length);
  for (let i = 0; i < len; i++) {
    const diff = predicted[i] - target[i];
    sum += diff * diff;
  }
  return sum / len;
}

function crossEntropyLoss(logits: number[], targetIdx: number): number {
  const probs = softmax(logits);
  const p = Math.max(1e-10, probs[targetIdx]);
  return -Math.log(p);
}

function backpropMLP(
  mlp: MLPWeights, mom: MLPMomentum,
  input: number[], hidden: number[], outputGrad: number[],
  lr: number
): number[] {
  const hiddenGrad: number[] = new Array(hidden.length).fill(0);

  for (let j = 0; j < mlp.b2.length; j++) {
    const grad = outputGrad[j];
    mom.db2[j] = MOMENTUM * mom.db2[j] + lr * grad;
    mlp.b2[j] -= mom.db2[j];
    for (let i = 0; i < hidden.length; i++) {
      const wGrad = grad * hidden[i];
      hiddenGrad[i] += grad * mlp.w2[i][j];
      mom.dw2[i][j] = MOMENTUM * mom.dw2[i][j] + lr * wGrad;
      mlp.w2[i][j] -= mom.dw2[i][j];
    }
  }

  const inputGrad: number[] = new Array(input.length).fill(0);
  for (let j = 0; j < hidden.length; j++) {
    const preActGrad = hiddenGrad[j] * reluDeriv(hidden[j]);
    mom.db1[j] = MOMENTUM * mom.db1[j] + lr * preActGrad;
    mlp.b1[j] -= mom.db1[j];
    for (let i = 0; i < input.length; i++) {
      const wGrad = preActGrad * input[i];
      inputGrad[i] += preActGrad * mlp.w1[i][j];
      mom.dw1[i][j] = MOMENTUM * mom.dw1[i][j] + lr * wGrad;
      mlp.w1[i][j] -= mom.dw1[i][j];
    }
  }

  return inputGrad;
}

async function trainVAE(dataset: CrystalStructureEntry[]): Promise<void> {
  if (dataset.length < 10) return;

  const encodedInputs: number[][] = [];
  const latticeTargets: number[][] = [];
  const csTargets: number[] = [];
  const sgTargets: number[] = [];
  const prototypeSet = new Set<string>();
  const protoTargets: number[] = [];
  const atomCountTargets: number[] = [];
  const compositionTargets: number[][] = [];

  for (const entry of dataset) {
    prototypeSet.add(entry.prototype);
  }
  const prototypes = Array.from(prototypeSet);

  for (const entry of dataset) {
    try {
      const vec = encodeEntry(entry);
      if (vec.some(v => !Number.isFinite(v))) continue;
      encodedInputs.push(vec);

      latticeTargets.push([
        entry.lattice.a / 15.0,
        entry.lattice.b / 15.0,
        entry.lattice.c / 35.0,
        entry.lattice.alpha / 180.0,
        entry.lattice.beta / 180.0,
        entry.lattice.gamma / 180.0,
      ]);

      const csIdx = CRYSTAL_SYSTEMS.indexOf(entry.crystalSystem as any);
      csTargets.push(Math.max(0, csIdx));

      const sgIdx = TOP_SPACEGROUPS.indexOf(entry.spacegroup as any);
      sgTargets.push(sgIdx >= 0 ? sgIdx : TOP_SPACEGROUPS.length - 1);

      protoTargets.push(prototypes.indexOf(entry.prototype));
      atomCountTargets.push(entry.nsites / 50.0);

      const compVec: number[] = new Array(TOP_ELEMENTS.length).fill(0);
      const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
      let match;
      while ((match = regex.exec(entry.formula)) !== null) {
        const el = match[1];
        const count = match[2] ? parseFloat(match[2]) : 1;
        const idx = TOP_ELEMENTS.indexOf(el as any);
        if (idx >= 0) compVec[idx] = count / 10.0;
      }
      compositionTargets.push(compVec);
    } catch {
      continue;
    }
  }

  if (encodedInputs.length < 10) return;

  const inputDim = encodedInputs[0].length;
  const encoderMLP = createMLP(inputDim, HIDDEN_DIM, LATENT_DIM * 2);
  const decoderLattice = createMLP(LATENT_DIM, HIDDEN_DIM, 6);
  const decoderCS = createMLP(LATENT_DIM, HIDDEN_DIM, CRYSTAL_SYSTEMS.length);
  const decoderSG = createMLP(LATENT_DIM, HIDDEN_DIM, TOP_SPACEGROUPS.length);
  const decoderProto = createMLP(LATENT_DIM, HIDDEN_DIM, prototypes.length);
  const decoderAtom = createMLP(LATENT_DIM, HIDDEN_DIM, 1);
  const decoderComp = createMLP(LATENT_DIM, HIDDEN_DIM, TOP_ELEMENTS.length);

  const encMom = createMomentum(encoderMLP);
  const decLatMom = createMomentum(decoderLattice);
  const decCSMom = createMomentum(decoderCS);
  const decSGMom = createMomentum(decoderSG);
  const decProtoMom = createMomentum(decoderProto);
  const decAtomMom = createMomentum(decoderAtom);
  const decCompMom = createMomentum(decoderComp);

  trainingHistory = [];

  const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

  function computeBeta(epoch: number): number {
    if (epoch < WARMUP_EPOCHS) {
      return BETA_MAX * 0.1 * (epoch / WARMUP_EPOCHS);
    }
    const postWarmup = epoch - WARMUP_EPOCHS;
    const cycleLen = Math.floor((NUM_EPOCHS - WARMUP_EPOCHS) / BETA_CYCLES);
    if (cycleLen <= 0) return BETA_MAX;
    const cyclePos = postWarmup % cycleLen;
    const rampFrac = cyclePos / (cycleLen * 0.5);
    return Math.min(BETA_MAX, rampFrac * BETA_MAX);
  }

  function trainOneSample(
    idx: number, beta: number, lr: number,
    updateDecoders: boolean
  ): { reconLoss: number; rawKL: number; activeDims: number } {
    const input = encodedInputs[idx];

    const encResult = forwardMLP(encoderMLP, input);
    const muLogSigma = encResult.output;
    const mu = muLogSigma.slice(0, LATENT_DIM);
    const logSigma = muLogSigma.slice(LATENT_DIM);

    const epsilon = Array.from({ length: LATENT_DIM }, () => gaussianRandom());
    const z = mu.map((m, i) => m + Math.exp(logSigma[i]) * epsilon[i]);

    const latResult = forwardMLP(decoderLattice, z);
    const csResult = forwardMLP(decoderCS, z);
    const sgResult = forwardMLP(decoderSG, z);
    const protoResult = forwardMLP(decoderProto, z);
    const atomResult = forwardMLP(decoderAtom, z);
    const compResult = forwardMLP(decoderComp, z);

    const latticeLoss = mseLoss(latResult.output, latticeTargets[idx]);
    const csLoss = crossEntropyLoss(csResult.output, csTargets[idx]);
    const sgLoss = crossEntropyLoss(sgResult.output, sgTargets[idx]);
    const protoLoss = crossEntropyLoss(protoResult.output, protoTargets[idx]);
    const atomLoss = mseLoss(atomResult.output, [atomCountTargets[idx]]);
    const compLoss = mseLoss(compResult.output, compositionTargets[idx]);

    const reconLoss = latticeLoss * 2.0 + csLoss * 0.5 + sgLoss * 0.3 + protoLoss * 0.3 + atomLoss * 0.5 + compLoss * 1.0;
    const rawKL = klDivergence(mu, logSigma);

    const zGrad = new Array(LATENT_DIM).fill(0);

    const decoderLr = updateDecoders ? lr : 0;

    const latticeGrad = latticeTargets[idx].map((t, i) => 2.0 * (latResult.output[i] - t) / 6 * 2.0);
    const latZGrad = backpropMLP(decoderLattice, decLatMom, z, latResult.hidden, latticeGrad, decoderLr);
    for (let i = 0; i < LATENT_DIM; i++) zGrad[i] += latZGrad[i];

    const csProbs = softmax(csResult.output);
    const csGrad = csProbs.map((p, i) => (i === csTargets[idx] ? p - 1 : p) * 0.5);
    const csZGrad = backpropMLP(decoderCS, decCSMom, z, csResult.hidden, csGrad, decoderLr);
    for (let i = 0; i < LATENT_DIM; i++) zGrad[i] += csZGrad[i];

    const sgProbs = softmax(sgResult.output);
    const sgGrad = sgProbs.map((p, i) => (i === sgTargets[idx] ? p - 1 : p) * 0.3);
    const sgZGrad = backpropMLP(decoderSG, decSGMom, z, sgResult.hidden, sgGrad, decoderLr);
    for (let i = 0; i < LATENT_DIM; i++) zGrad[i] += sgZGrad[i];

    const protoProbs = softmax(protoResult.output);
    const protoGrad = protoProbs.map((p, i) => (i === protoTargets[idx] ? p - 1 : p) * 0.3);
    const protoZGrad = backpropMLP(decoderProto, decProtoMom, z, protoResult.hidden, protoGrad, decoderLr);
    for (let i = 0; i < LATENT_DIM; i++) zGrad[i] += protoZGrad[i];

    const atomGrad = [2.0 * (atomResult.output[0] - atomCountTargets[idx]) * 0.5];
    const atomZGrad = backpropMLP(decoderAtom, decAtomMom, z, atomResult.hidden, atomGrad, decoderLr);
    for (let i = 0; i < LATENT_DIM; i++) zGrad[i] += atomZGrad[i];

    const compGrad = compositionTargets[idx].map((t, i) => 2.0 * (compResult.output[i] - t) / TOP_ELEMENTS.length * 1.0);
    const compZGrad = backpropMLP(decoderComp, decCompMom, z, compResult.hidden, compGrad, decoderLr);
    for (let i = 0; i < LATENT_DIM; i++) zGrad[i] += compZGrad[i];

    const klPerDim = klDivergencePerDim(mu, logSigma);
    let activeDims = 0;

    const muGrad = mu.map((m, i) => {
      const reconThruReparam = zGrad[i];
      const dimKL = klPerDim[i];
      if (dimKL >= FREE_BITS) activeDims++;
      const softGate = dimKL < FREE_BITS
        ? FREE_BITS > 0 ? 0.1 * (dimKL / FREE_BITS) : 0.1
        : 1.0;
      const klGrad = softGate * beta * m;
      return reconThruReparam + klGrad;
    });
    const logSigmaGrad = logSigma.map((ls, i) => {
      const sigma = Math.exp(ls);
      const reconThruReparam = zGrad[i] * epsilon[i] * sigma;
      const dimKL = klPerDim[i];
      const softGate = dimKL < FREE_BITS
        ? FREE_BITS > 0 ? 0.1 * (dimKL / FREE_BITS) : 0.1
        : 1.0;
      const klGrad = softGate * beta * (Math.exp(2 * ls) - 1);
      return reconThruReparam + klGrad;
    });

    if (rawKL < KL_MIN_TARGET && beta > 0) {
      for (let i = 0; i < LATENT_DIM; i++) {
        const sigma = Math.exp(logSigma[i]);
        muGrad[i] -= KL_PULL_STRENGTH * mu[i];
        logSigmaGrad[i] -= KL_PULL_STRENGTH * (1 - sigma);
      }
    }

    const encOutGrad = [...muGrad, ...logSigmaGrad];
    backpropMLP(encoderMLP, encMom, input, encResult.hidden, encOutGrad, lr);

    return { reconLoss, rawKL, activeDims };
  }

  for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    if (epoch % 20 === 0) await yield_();
    const beta = computeBeta(epoch);

    let totalLoss = 0;
    let totalRecon = 0;
    let totalKL = 0;
    let totalActiveDims = 0;
    const lr = LEARNING_RATE * Math.max(0.05, 1.0 - epoch / (NUM_EPOCHS * 1.5));

    const indices = Array.from({ length: encodedInputs.length }, (_, i) => i);
    for (let j = indices.length - 1; j > 0; j--) {
      const k = Math.floor(Math.random() * (j + 1));
      [indices[j], indices[k]] = [indices[k], indices[j]];
    }

    const isEncoderCatchup = epoch > WARMUP_EPOCHS &&
      epoch % ENCODER_CATCHUP_INTERVAL < ENCODER_CATCHUP_STEPS;

    for (const idx of indices) {
      const result = trainOneSample(idx, beta, lr, !isEncoderCatchup);
      totalRecon += result.reconLoss;
      totalKL += result.rawKL;
      totalActiveDims += result.activeDims;
      totalLoss += result.reconLoss + beta * result.rawKL;
    }

    const n = encodedInputs.length;
    if (epoch % 10 === 0 || epoch === NUM_EPOCHS - 1) {
      const avgActiveDims = Math.round(totalActiveDims / n);
      trainingHistory.push({
        epoch,
        totalLoss: totalLoss / n,
        reconLoss: totalRecon / n,
        klLoss: totalKL / n,
        beta,
      });
      if (epoch % 50 === 0) {
        console.log(`[Crystal-VAE] epoch=${epoch} beta=${beta.toFixed(3)} reconLoss=${(totalRecon/n).toFixed(4)} KL=${(totalKL/n).toFixed(4)} activeDims=${avgActiveDims}/${LATENT_DIM}${isEncoderCatchup ? " [encoder-catchup]" : ""}`);
      }
    }
  }

  model = {
    encoder: encoderMLP,
    decoderLattice,
    decoderCrystalSystem: decoderCS,
    decoderSpacegroup: decoderSG,
    decoderPrototype: decoderProto,
    decoderAtomCount: decoderAtom,
    decoderComposition: decoderComp,
    inputDim,
    prototypes,
  };

  encodedTrainingData = [];
  for (let i = 0; i < encodedInputs.length; i++) {
    const encResult = forwardMLP(model.encoder, encodedInputs[i]);
    const mu = encResult.output.slice(0, LATENT_DIM);
    encodedTrainingData.push({ formula: dataset[i].formula, z: mu });
  }

  trained = true;
  const finalStats = trainingHistory[trainingHistory.length - 1];
  console.log(`Crystal VAE trained on ${encodedInputs.length} entries, final loss: ${finalStats?.totalLoss.toFixed(4)}, KL: ${finalStats?.klLoss.toFixed(4)}, beta: ${finalStats?.beta.toFixed(3)}`);
}

function decodeLatentVector(z: number[]): {
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  crystalSystem: string;
  crystalSystemProbs: Record<string, number>;
  spacegroup: number;
  spacegroupSymbol: string;
  spacegroupProbs: Record<number, number>;
  prototype: string;
  prototypeProbs: Record<string, number>;
  atomCount: number;
  compositionVector: Record<string, number>;
  formula: string;
} {
  if (!model) throw new Error("VAE not trained");

  const latResult = forwardMLP(model.decoderLattice, z);
  const csResult = forwardMLP(model.decoderCrystalSystem, z);
  const sgResult = forwardMLP(model.decoderSpacegroup, z);
  const protoResult = forwardMLP(model.decoderPrototype, z);
  const atomResult = forwardMLP(model.decoderAtomCount, z);
  const compResult = forwardMLP(model.decoderComposition, z);

  const rawLattice = latResult.output;
  const a = Math.max(1.5, rawLattice[0] * 15.0);
  const b = Math.max(1.5, rawLattice[1] * 15.0);
  const c = Math.max(1.5, rawLattice[2] * 35.0);
  const alpha = Math.max(60, Math.min(120, rawLattice[3] * 180.0));
  const beta_val = Math.max(60, Math.min(120, rawLattice[4] * 180.0));
  const gamma = Math.max(60, Math.min(150, rawLattice[5] * 180.0));

  const csProbs = softmax(csResult.output);
  const csIdx = csProbs.indexOf(Math.max(...csProbs));
  const crystalSystem = CRYSTAL_SYSTEMS[csIdx];
  const crystalSystemProbs: Record<string, number> = {};
  CRYSTAL_SYSTEMS.forEach((cs, i) => { crystalSystemProbs[cs] = Math.round(csProbs[i] * 10000) / 10000; });

  const sgProbs = softmax(sgResult.output);
  const sgIdx = sgProbs.indexOf(Math.max(...sgProbs));
  const spacegroup = TOP_SPACEGROUPS[sgIdx];
  const spacegroupSymbol = SG_SYMBOLS[spacegroup] || `SG-${spacegroup}`;
  const spacegroupProbs: Record<number, number> = {};
  TOP_SPACEGROUPS.forEach((sg, i) => { spacegroupProbs[sg] = Math.round(sgProbs[i] * 10000) / 10000; });

  const protoProbs = softmax(protoResult.output);
  const protoIdx = protoProbs.indexOf(Math.max(...protoProbs));
  const prototype = model.prototypes[protoIdx] || "unknown";
  const prototypeProbs: Record<string, number> = {};
  model.prototypes.forEach((p, i) => { prototypeProbs[p] = Math.round(protoProbs[i] * 10000) / 10000; });

  const atomCount = Math.max(1, Math.round(atomResult.output[0] * 50.0));

  const compVec = compResult.output;
  const compositionVector: Record<string, number> = {};
  const topCompIndices = compVec
    .map((v, i) => ({ v: Math.max(0, v), i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 6);

  for (const { v, i } of topCompIndices) {
    if (v > 0.01) {
      compositionVector[TOP_ELEMENTS[i]] = Math.round(v * 10 * 100) / 100;
    }
  }

  const formulaParts: string[] = [];
  const sortedComp = Object.entries(compositionVector)
    .filter(([, v]) => v >= 0.5)
    .sort(([, a], [, b]) => b - a);

  if (sortedComp.length === 0) {
    formulaParts.push("Fe");
  } else {
    for (const [el, count] of sortedComp) {
      const rounded = Math.round(count);
      if (rounded <= 0) continue;
      formulaParts.push(rounded === 1 ? el : `${el}${rounded}`);
    }
  }

  return {
    lattice: {
      a: Math.round(a * 100) / 100,
      b: Math.round(b * 100) / 100,
      c: Math.round(c * 100) / 100,
      alpha: Math.round(alpha * 10) / 10,
      beta: Math.round(beta_val * 10) / 10,
      gamma: Math.round(gamma * 10) / 10,
    },
    crystalSystem,
    crystalSystemProbs,
    spacegroup,
    spacegroupSymbol,
    spacegroupProbs,
    prototype,
    prototypeProbs,
    atomCount,
    compositionVector,
    formula: formulaParts.join("") || "Unknown",
  };
}

function computeNoveltyScore(z: number[]): number {
  if (encodedTrainingData.length === 0) return 1.0;

  let minDist = Infinity;
  for (const td of encodedTrainingData) {
    let dist = 0;
    for (let i = 0; i < z.length; i++) {
      const diff = z[i] - td.z[i];
      dist += diff * diff;
    }
    dist = Math.sqrt(dist);
    if (dist < minDist) minDist = dist;
  }

  const avgDist = encodedTrainingData.reduce((s, td) => {
    let d = 0;
    for (let i = 0; i < z.length; i++) {
      const diff = z[i] - td.z[i];
      d += diff * diff;
    }
    return s + Math.sqrt(d);
  }, 0) / encodedTrainingData.length;

  return Math.min(1.0, minDist / Math.max(0.1, avgDist));
}

export interface GeneratedCrystal {
  formula: string;
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  crystalSystem: string;
  crystalSystemProbs: Record<string, number>;
  spacegroup: number;
  spacegroupSymbol: string;
  prototype: string;
  atomCount: number;
  compositionVector: Record<string, number>;
  noveltyScore: number;
  generationMethod: string;
  latentVector: number[];
}

export function generateNovelCrystal(targetSystem?: string): GeneratedCrystal | null {
  if (!model || !trained) return null;

  let z: number[];
  let attempts = 0;
  let bestNovelty = 0;
  let bestZ: number[] = [];

  while (attempts < 10) {
    z = Array.from({ length: LATENT_DIM }, () => gaussianRandom() * 0.8);

    if (targetSystem) {
      const csIdx = CRYSTAL_SYSTEMS.indexOf(targetSystem as any);
      if (csIdx >= 0) {
        const matchingEntries = encodedTrainingData.filter((_, i) => {
          const ds = getTrainingData();
          return i < ds.length && ds[i].crystalSystem === targetSystem;
        });
        if (matchingEntries.length > 0) {
          const ref = matchingEntries[Math.floor(Math.random() * matchingEntries.length)];
          for (let i = 0; i < LATENT_DIM; i++) {
            z[i] = ref.z[i] + gaussianRandom() * 0.5;
          }
        }
      }
    }

    const nov = computeNoveltyScore(z);
    if (nov > bestNovelty) {
      bestNovelty = nov;
      bestZ = [...z];
    }
    attempts++;
    if (nov > 0.3) break;
  }

  z = bestZ;
  const decoded = decodeLatentVector(z);
  const nov = computeNoveltyScore(z);
  noveltyScores.push(nov);
  totalGenerations++;

  return {
    formula: decoded.formula,
    lattice: decoded.lattice,
    crystalSystem: decoded.crystalSystem,
    crystalSystemProbs: decoded.crystalSystemProbs,
    spacegroup: decoded.spacegroup,
    spacegroupSymbol: decoded.spacegroupSymbol,
    prototype: decoded.prototype,
    atomCount: decoded.atomCount,
    compositionVector: decoded.compositionVector,
    noveltyScore: Math.round(nov * 10000) / 10000,
    generationMethod: "vae-sampling",
    latentVector: z.map(v => Math.round(v * 10000) / 10000),
  };
}

export function interpolateCrystals(formula1: string, formula2: string, alpha: number): GeneratedCrystal | null {
  if (!model || !trained) return null;

  const entry1 = encodedTrainingData.find(d => d.formula === formula1);
  const entry2 = encodedTrainingData.find(d => d.formula === formula2);

  if (!entry1 || !entry2) return null;

  const clampedAlpha = Math.max(0, Math.min(1, alpha));
  const z = entry1.z.map((v, i) => v * (1 - clampedAlpha) + entry2.z[i] * clampedAlpha);

  const decoded = decodeLatentVector(z);
  const nov = computeNoveltyScore(z);
  noveltyScores.push(nov);
  totalGenerations++;

  return {
    formula: decoded.formula,
    lattice: decoded.lattice,
    crystalSystem: decoded.crystalSystem,
    crystalSystemProbs: decoded.crystalSystemProbs,
    spacegroup: decoded.spacegroup,
    spacegroupSymbol: decoded.spacegroupSymbol,
    prototype: decoded.prototype,
    atomCount: decoded.atomCount,
    compositionVector: decoded.compositionVector,
    noveltyScore: Math.round(nov * 10000) / 10000,
    generationMethod: `interpolation(${formula1},${formula2},${clampedAlpha})`,
    latentVector: z.map(v => Math.round(v * 10000) / 10000),
  };
}

export function encodeFormula(formula: string): { mu: number[]; logSigma: number[]; z: number[] } | null {
  if (!model || !trained) return null;

  const entry = getTrainingData().find(e => e.formula === formula);
  if (!entry) return null;

  try {
    const input = encodeEntry(entry);
    if (input.some(v => !Number.isFinite(v))) return null;

    const encResult = forwardMLP(model.encoder, input);
    const mu = encResult.output.slice(0, LATENT_DIM);
    const logSigma = encResult.output.slice(LATENT_DIM);
    const z = reparameterize(mu, logSigma);

    return {
      mu: mu.map(v => Math.round(v * 10000) / 10000),
      logSigma: logSigma.map(v => Math.round(v * 10000) / 10000),
      z: z.map(v => Math.round(v * 10000) / 10000),
    };
  } catch {
    return null;
  }
}

export function getCrystalVAEStats() {
  const avgNovelty = noveltyScores.length > 0
    ? noveltyScores.reduce((s, v) => s + v, 0) / noveltyScores.length
    : 0;

  let activeDimsEstimate = 0;
  if (model && encodedTrainingData.length > 0) {
    const sampleSize = Math.min(20, encodedTrainingData.length);
    const muVars = new Array(LATENT_DIM).fill(0);
    const muMeans = new Array(LATENT_DIM).fill(0);
    for (let s = 0; s < sampleSize; s++) {
      const idx = Math.floor(Math.random() * encodedTrainingData.length);
      const z = encodedTrainingData[idx].z;
      for (let d = 0; d < LATENT_DIM; d++) muMeans[d] += z[d] / sampleSize;
    }
    for (let s = 0; s < sampleSize; s++) {
      const idx = Math.floor(Math.random() * encodedTrainingData.length);
      const z = encodedTrainingData[idx].z;
      for (let d = 0; d < LATENT_DIM; d++) muVars[d] += (z[d] - muMeans[d]) ** 2 / sampleSize;
    }
    activeDimsEstimate = muVars.filter(v => v > 0.01).length;
  }

  return {
    trained,
    datasetSize: encodedTrainingData.length,
    latentDim: LATENT_DIM,
    totalGenerations,
    avgNoveltyScore: Math.round(avgNovelty * 10000) / 10000,
    trainingHistory: trainingHistory.slice(-20),
    finalLoss: trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].totalLoss : null,
    finalReconLoss: trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].reconLoss : null,
    finalKLLoss: trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].klLoss : null,
    activeDims: activeDimsEstimate,
    posteriorCollapseDetected: trainingHistory.length > 0 && trainingHistory[trainingHistory.length - 1].klLoss < 0.1,
    prototypesLearned: model?.prototypes.length ?? 0,
    encodedFormulas: encodedTrainingData.map(d => d.formula),
  };
}

export async function initCrystalVAE(): Promise<void> {
  try {
    const dataset = getTrainingData();
    if (dataset.length >= 10) {
      await trainVAE(dataset);
    }
  } catch (err) {
    console.error("Failed to initialize Crystal VAE:", err);
  }
}
