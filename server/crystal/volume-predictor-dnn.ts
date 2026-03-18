import { computeCompositionFeatures, compositionFeatureVector } from "../learning/composition-features";
import { ELEMENTAL_DATA } from "../learning/elemental-data";

interface DNNLayer {
  weights: number[][];
  biases: number[];
}

interface VolumeDNN {
  layers: DNNLayer[];
  inputMean: number[];
  inputStd: number[];
  outputMean: number;
  outputStd: number;
  trainedAt: number;
  datasetSize: number;
  metrics: {
    trainMAE: number;
    trainMAPE: number;
  };
}

let model: VolumeDNN | null = null;
let trainCount = 0;

const CRYSTAL_SYSTEMS = ["cubic", "hexagonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic", "rhombohedral"];

function encodeCrystalSystem(system: string): number[] {
  const vec = new Array(CRYSTAL_SYSTEMS.length).fill(0);
  const idx = CRYSTAL_SYSTEMS.indexOf(system.toLowerCase());
  if (idx >= 0) vec[idx] = 1;
  return vec;
}

function guessCrystalSystem(formula: string): string {
  const cf = computeCompositionFeatures(formula);
  if (cf.enRange < 0.5 && cf.nAtoms <= 2) return "cubic";
  if (cf.enRange > 1.5) return "orthorhombic";
  if (cf.nAtoms > 8) return "hexagonal";
  return "cubic";
}

function buildInputVector(formula: string, pressure: number = 0): number[] {
  const cf = computeCompositionFeatures(formula);
  const compVec = compositionFeatureVector(cf);
  const crystalEnc = encodeCrystalSystem(guessCrystalSystem(formula));
  return [...compVec, pressure, ...crystalEnc];
}

function relu(x: number): number {
  return Math.max(0, x);
}

function forwardPass(model: VolumeDNN, rawInput: number[]): number {
  let x = rawInput.map((v, i) => {
    const std = model.inputStd[i];
    return std > 1e-8 ? (v - model.inputMean[i]) / std : 0;
  });

  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    const output = new Array(layer.biases.length).fill(0);
    for (let j = 0; j < layer.biases.length; j++) {
      let sum = layer.biases[j];
      for (let i = 0; i < x.length; i++) {
        sum += layer.weights[i][j] * x[i];
      }
      output[j] = l < model.layers.length - 1 ? relu(sum) : sum;
    }
    x = output;
  }

  return x[0] * model.outputStd + model.outputMean;
}

function initLayer(inputSize: number, outputSize: number, seed: number): DNNLayer {
  const scale = Math.sqrt(2.0 / inputSize);
  const weights: number[][] = [];
  let s = seed;
  function pseudoRand(): number {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return (s / 0x7fffffff - 0.5) * 2;
  }

  for (let i = 0; i < inputSize; i++) {
    const row: number[] = [];
    for (let j = 0; j < outputSize; j++) {
      row.push(pseudoRand() * scale);
    }
    weights.push(row);
  }
  const biases = new Array(outputSize).fill(0);
  return { weights, biases };
}

interface TrainingPair {
  input: number[];
  target: number;
}

function buildTrainingData(): TrainingPair[] {
  const pairs: TrainingPair[] = [];

  const knownVolumes: Record<string, number> = {
    "MgB2": 28.98,
    "NbN": 23.94,
    "Nb3Sn": 170.89,
    "Nb3Ge": 170.5,
    "NbTi": 35.0,
    "YBa2Cu3O7": 173.5,
    "LaH10": 65.0,
    "YH6": 58.5,
    "CaH6": 52.0,
    "H3S": 42.0,
    "FeSe": 54.3,
    "LiFeAs": 94.2,
    "BaFe2As2": 204.0,
    "MgCNi3": 53.7,
    "La2CuO4": 188.0,
    "Bi2Sr2CaCu2O8": 530.0,
    "Al": 66.4,
    "Nb": 35.8,
    "Pb": 121.3,
    "Sn": 108.2,
    "V": 27.6,
    "Ta": 35.9,
    "Mo": 31.2,
    "W": 31.7,
    "Ti": 35.3,
    "Zr": 46.5,
    "Hf": 44.6,
    "Sc": 25.0,
    "Cu": 47.2,
    "Ag": 68.2,
    "Au": 67.9,
    "Fe": 23.6,
    "Co": 22.1,
    "Ni": 43.8,
    "Pd": 58.9,
    "Pt": 60.4,
    "Ir": 56.8,
    "Rh": 55.5,
    "Os": 27.8,
    "Ru": 27.1,
    "Re": 29.5,
    "Cr": 24.0,
    "Mn": 66.8,
    "Zn": 30.4,
    "Ga": 76.2,
    "In": 52.5,
    "Tl": 57.1,
    "Ge": 45.3,
    "Si": 40.0,
    "Li": 43.2,
    "Na": 79.0,
    "K": 151.5,
    "Ca": 174.6,
    "Sr": 224.6,
    "Ba": 63.4,
    "Mg": 46.5,
    "Be": 16.2,
    "LaFeAsO": 132.6,
  };

  for (const el of Object.keys(ELEMENTAL_DATA)) {
    const data = ELEMENTAL_DATA[el];
    if (!data || !data.latticeConstant) continue;
    const lc = data.latticeConstant;
    const volume = lc * lc * lc;
    if (volume > 0 && volume < 2000) {
      const input = buildInputVector(el, 0);
      if (input.every(v => Number.isFinite(v))) {
        const existing = knownVolumes[el];
        pairs.push({ input, target: existing ?? volume });
      }
    }
  }

  for (const [formula, vol] of Object.entries(knownVolumes)) {
    if (Object.keys(ELEMENTAL_DATA).includes(formula)) continue;
    try {
      const input = buildInputVector(formula, 0);
      if (input.every(v => Number.isFinite(v))) {
        pairs.push({ input, target: vol });
      }
    } catch {
      continue;
    }
  }

  return pairs;
}

function trainDNN(data: TrainingPair[], epochs: number = 200, lr: number = 0.001): VolumeDNN {
  const inputDim = data[0].input.length;

  const inputMean = new Array(inputDim).fill(0);
  const inputStd = new Array(inputDim).fill(0);
  for (const d of data) {
    for (let i = 0; i < inputDim; i++) inputMean[i] += d.input[i];
  }
  for (let i = 0; i < inputDim; i++) inputMean[i] /= data.length;
  for (const d of data) {
    for (let i = 0; i < inputDim; i++) inputStd[i] += (d.input[i] - inputMean[i]) ** 2;
  }
  for (let i = 0; i < inputDim; i++) inputStd[i] = Math.sqrt(inputStd[i] / data.length);

  const targets = data.map(d => d.target);
  const outputMean = targets.reduce((s, v) => s + v, 0) / targets.length;
  const outputStd = Math.sqrt(targets.reduce((s, v) => s + (v - outputMean) ** 2, 0) / targets.length) || 1;

  const normalizedTargets = targets.map(t => (t - outputMean) / outputStd);

  const normalizedInputs = data.map(d =>
    d.input.map((v, i) => (inputStd[i] > 1e-8 ? (v - inputMean[i]) / inputStd[i] : 0))
  );

  let layer1 = initLayer(inputDim, 64, 42);
  let layer2 = initLayer(64, 32, 137);
  let layer3 = initLayer(32, 1, 256);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const currentLr = lr * (1 - epoch / epochs * 0.5);

    for (let di = 0; di < data.length; di++) {
      const x0 = normalizedInputs[di];
      const target = normalizedTargets[di];

      const z1 = new Array(64).fill(0);
      const a1 = new Array(64).fill(0);
      for (let j = 0; j < 64; j++) {
        let sum = layer1.biases[j];
        for (let i = 0; i < inputDim; i++) sum += layer1.weights[i][j] * x0[i];
        z1[j] = sum;
        a1[j] = relu(sum);
      }

      const z2 = new Array(32).fill(0);
      const a2 = new Array(32).fill(0);
      for (let j = 0; j < 32; j++) {
        let sum = layer2.biases[j];
        for (let i = 0; i < 64; i++) sum += layer2.weights[i][j] * a1[i];
        z2[j] = sum;
        a2[j] = relu(sum);
      }

      let output = layer3.biases[0];
      for (let i = 0; i < 32; i++) output += layer3.weights[i][0] * a2[i];

      const dOutput = 2 * (output - target);

      const dA2 = new Array(32).fill(0);
      for (let i = 0; i < 32; i++) {
        dA2[i] = layer3.weights[i][0] * dOutput;
      }
      for (let i = 0; i < 32; i++) {
        layer3.weights[i][0] -= currentLr * a2[i] * dOutput;
      }
      layer3.biases[0] -= currentLr * dOutput;

      const dZ2 = dA2.map((v, i) => z2[i] > 0 ? v : 0);
      const dA1 = new Array(64).fill(0);
      for (let i = 0; i < 64; i++) {
        for (let j = 0; j < 32; j++) {
          dA1[i] += layer2.weights[i][j] * dZ2[j];
        }
      }
      for (let i = 0; i < 64; i++) {
        for (let j = 0; j < 32; j++) {
          layer2.weights[i][j] -= currentLr * a1[i] * dZ2[j];
        }
      }
      for (let j = 0; j < 32; j++) {
        layer2.biases[j] -= currentLr * dZ2[j];
      }

      const dZ1 = dA1.map((v, i) => z1[i] > 0 ? v : 0);
      for (let i = 0; i < inputDim; i++) {
        for (let j = 0; j < 64; j++) {
          layer1.weights[i][j] -= currentLr * x0[i] * dZ1[j];
        }
      }
      for (let j = 0; j < 64; j++) {
        layer1.biases[j] -= currentLr * dZ1[j];
      }
    }
  }

  const dnn: VolumeDNN = {
    layers: [layer1, layer2, layer3],
    inputMean,
    inputStd,
    outputMean,
    outputStd,
    trainedAt: Date.now(),
    datasetSize: data.length,
    metrics: { trainMAE: 0, trainMAPE: 0 },
  };

  let totalAE = 0;
  let totalAPE = 0;
  for (const d of data) {
    const pred = forwardPass(dnn, d.input);
    totalAE += Math.abs(pred - d.target);
    totalAPE += Math.abs(pred - d.target) / Math.max(d.target, 1);
  }
  dnn.metrics.trainMAE = Math.round((totalAE / data.length) * 100) / 100;
  dnn.metrics.trainMAPE = Math.round((totalAPE / data.length) * 10000) / 100;

  return dnn;
}

export function initVolumeDNN(): void {
  try {
    const data = buildTrainingData();
    if (data.length >= 10) {
      model = trainDNN(data, 300, 0.0005);
      trainCount++;
    }
  } catch (err: any) {
    console.debug(`[volume-dnn] DNN training failed: ${err?.message ?? err}`);
  }
}

export function isVolumeDNNTrained(): boolean {
  return model !== null;
}

export function predictVolume(formula: string, pressure: number = 0): {
  volume: number;
  source: "dnn" | "heuristic";
  confidence: number;
} {
  if (model) {
    try {
      const input = buildInputVector(formula, pressure);
      if (input.every(v => Number.isFinite(v))) {
        const volume = forwardPass(model, input);
        if (Number.isFinite(volume) && volume > 0 && volume < 5000) {
          const clampedVol = Math.max(5, Math.min(3000, volume));
          return {
            volume: Math.round(clampedVol * 100) / 100,
            source: "dnn",
            confidence: Math.min(0.95, 0.5 + model.datasetSize / 200),
          };
        }
      }
    } catch (err: any) {
      console.debug(`[volume-dnn] Forward pass failed for ${formula}: ${err?.message ?? err}`);
    }
  }

  return heuristicVolume(formula, pressure);
}

function heuristicVolume(formula: string, pressure: number): {
  volume: number;
  source: "heuristic";
  confidence: number;
} {
  const cf = computeCompositionFeatures(formula);
  const avgLC = cf.latticeConstMean > 0 ? cf.latticeConstMean : 4.0;
  let volume = avgLC * avgLC * avgLC;

  if (pressure > 0) {
    const B0 = cf.bulkModMean > 0 ? cf.bulkModMean : 50;
    const eta = Math.pow(1 + 4.0 * pressure / Math.max(B0, 1), -1 / 4);
    volume *= Math.max(0.5, eta);
  }

  return {
    volume: Math.round(Math.max(5, volume) * 100) / 100,
    source: "heuristic",
    confidence: 0.2,
  };
}

export function predictEquilibriumLatticeConstant(formula: string, pressure: number = 0): number {
  const result = predictVolume(formula, pressure);
  return Math.round(Math.pow(result.volume, 1 / 3) * 1000) / 1000;
}

export function getVolumeDNNStats(): {
  trained: boolean;
  trainCount: number;
  datasetSize: number;
  trainedAt: number;
  metrics: { trainMAE: number; trainMAPE: number };
  inputDim: number;
  architecture: string;
} {
  return {
    trained: model !== null,
    trainCount,
    datasetSize: model?.datasetSize ?? 0,
    trainedAt: model?.trainedAt ?? 0,
    metrics: model?.metrics ?? { trainMAE: 0, trainMAPE: 0 },
    inputDim: model?.layers[0]?.weights.length ?? 0,
    architecture: "90→64→32→1 MLP (ReLU)",
  };
}
