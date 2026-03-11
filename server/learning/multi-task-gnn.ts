import { getElementData, isTransitionMetal } from "./elemental-data";
import { buildCrystalGraph, GNNPredict, getGNNModel, type CrystalGraph } from "./graph-neural-net";
import { computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling } from "./physics-engine";
import { extractFeatures } from "./ml-predictor";

export interface MultiTaskPrediction {
  formationEnergy: number;
  phononStability: boolean;
  predictedTc: number;
  lambda: number;
  confidence: number;
  bandGap: number;
  dosAtFermi: number;
  bulkModulus: number;
  shearModulus: number;
  debyeTemperature: number;
  magneticMoment: number;
  omegaLog: number;
  muStar: number;
  nestingStrength: number;
  correlationStrength: number;
  dimensionality: number;
  metallicity: number;
  topologicalIndex: number;
  propertyVector: number[];
}

interface MultiTaskWeights {
  W_band: number[][];
  b_band: number[];
  W_dos: number[][];
  b_dos: number[];
  W_elastic: number[][];
  b_elastic: number[];
  W_magnetic: number[][];
  b_magnetic: number[];
  W_phonon_detail: number[][];
  b_phonon_detail: number[];
  W_topology: number[][];
  b_topology: number[];
}

const HIDDEN = 28;
const HEADS = 6;

function initMatrix(rows: number, cols: number, scale?: number): number[][] {
  const heScale = scale ?? Math.sqrt(2.0 / cols);
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push((Math.random() - 0.5) * 2 * heScale);
    }
    m.push(row);
  }
  return m;
}

function initVector(dim: number): number[] {
  return new Array(dim).fill(0);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
  return mat.map(row => {
    let s = 0;
    for (let i = 0; i < Math.min(row.length, vec.length); i++) {
      s += row[i] * (vec[i] ?? 0);
    }
    return s;
  });
}

function relu(vec: number[]): number[] {
  return vec.map(v => Math.max(0, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + (b[i] ?? 0));
}

let multiTaskWeights: MultiTaskWeights | null = null;

function getMultiTaskWeights(): MultiTaskWeights {
  if (multiTaskWeights) return multiTaskWeights;

  multiTaskWeights = {
    W_band: initMatrix(HEADS, HIDDEN),
    b_band: initVector(HEADS),
    W_dos: initMatrix(HEADS, HIDDEN),
    b_dos: initVector(HEADS),
    W_elastic: initMatrix(HEADS, HIDDEN),
    b_elastic: initVector(HEADS),
    W_magnetic: initMatrix(HEADS, HIDDEN),
    b_magnetic: initVector(HEADS),
    W_phonon_detail: initMatrix(HEADS, HIDDEN),
    b_phonon_detail: initVector(HEADS),
    W_topology: initMatrix(HEADS, HIDDEN),
    b_topology: initVector(HEADS),
  };

  return multiTaskWeights;
}

function extractGraphPooling(graph: CrystalGraph): number[] {
  const nNodes = graph.nodes.length;
  const meanPool = new Array(20).fill(0);
  const maxPool = new Array(20).fill(-Infinity);

  for (const node of graph.nodes) {
    for (let k = 0; k < 20; k++) {
      meanPool[k] += (node.embedding[k] ?? 0) / nNodes;
      maxPool[k] = Math.max(maxPool[k], node.embedding[k] ?? 0);
    }
  }

  const pooled = [...meanPool, ...maxPool.map(v => v === -Infinity ? 0 : v)];
  while (pooled.length < HIDDEN) pooled.push(0);
  return pooled.slice(0, HIDDEN);
}

export function multiTaskPredict(formula: string): MultiTaskPrediction {
  const gnnWeights = getGNNModel();
  const graph = buildCrystalGraph(formula);
  const basePred = GNNPredict(graph, gnnWeights);

  const pooled = extractGraphPooling(graph);
  const mtWeights = getMultiTaskWeights();

  const bandOut = vecAdd(matVecMul(mtWeights.W_band, pooled), mtWeights.b_band);
  const dosOut = vecAdd(matVecMul(mtWeights.W_dos, pooled), mtWeights.b_dos);
  const elasticOut = vecAdd(matVecMul(mtWeights.W_elastic, pooled), mtWeights.b_elastic);
  const magneticOut = vecAdd(matVecMul(mtWeights.W_magnetic, pooled), mtWeights.b_magnetic);
  const phononOut = vecAdd(matVecMul(mtWeights.W_phonon_detail, pooled), mtWeights.b_phonon_detail);
  const topoOut = vecAdd(matVecMul(mtWeights.W_topology, pooled), mtWeights.b_topology);

  let electronic: any = null;
  let phonon: any = null;
  let coupling: any = null;
  let features: any = null;

  try {
    electronic = computeElectronicStructure(formula, null);
    phonon = computePhononSpectrum(formula, electronic);
    coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);
    features = extractFeatures(formula);
  } catch {}

  const bandGap = electronic
    ? electronic.bandGap ?? Math.max(0, sigmoid(bandOut[0]) * 5)
    : Math.max(0, sigmoid(bandOut[0]) * 5);

  const dosAtFermi = electronic
    ? electronic.densityOfStatesAtFermi
    : Math.max(0, (bandOut[1] ?? 0) * 10 + 3);

  const metallicity = electronic
    ? electronic.metallicity
    : sigmoid(bandOut[2] ?? 0);

  const avgBulk = computeAvgBulkModulus(formula);
  const bulkModulus = avgBulk > 0
    ? avgBulk * (1 + (elasticOut[0] ?? 0) * 0.1)
    : Math.max(10, Math.abs(elasticOut[0] ?? 50) * 50 + 50);

  const shearModulus = bulkModulus * (0.35 + sigmoid(elasticOut[1] ?? 0) * 0.3);

  const debyeTemp = phonon
    ? phonon.debyeTemperature
    : Math.max(100, Math.abs(phononOut[0] ?? 300) * 200 + 300);

  const magneticMoment = computeMagneticMoment(formula, magneticOut);

  const omegaLog = coupling
    ? coupling.omegaLog
    : Math.max(50, Math.abs(phononOut[1] ?? 200) * 150 + 200);

  const muStar = coupling
    ? coupling.muStar
    : Math.max(0.08, Math.min(0.2, sigmoid(phononOut[2] ?? 0) * 0.15 + 0.08));

  const nestingStrength = features
    ? features.fermiSurfaceNesting ?? sigmoid(topoOut[0] ?? 0) * 0.5
    : sigmoid(topoOut[0] ?? 0) * 0.5;

  const correlationStrength = features
    ? features.correlationStrength ?? sigmoid(topoOut[1] ?? 0) * 0.5
    : sigmoid(topoOut[1] ?? 0) * 0.5;

  const dimensionality = features
    ? features.dimensionality ?? 3
    : 2 + sigmoid(topoOut[2] ?? 0);

  const topologicalIndex = sigmoid(topoOut[3] ?? 0) * 0.3;

  const propertyVector = [
    basePred.formationEnergy,
    basePred.predictedTc / 100,
    basePred.lambda,
    bandGap / 5,
    dosAtFermi / 20,
    metallicity,
    bulkModulus / 500,
    shearModulus / 200,
    debyeTemp / 2000,
    magneticMoment / 5,
    omegaLog / 500,
    muStar,
    nestingStrength,
    correlationStrength,
    dimensionality / 3,
    topologicalIndex,
  ];

  return {
    formationEnergy: basePred.formationEnergy,
    phononStability: basePred.phononStability,
    predictedTc: basePred.predictedTc,
    lambda: basePred.lambda,
    confidence: basePred.confidence,
    bandGap: Math.round(bandGap * 100) / 100,
    dosAtFermi: Math.round(dosAtFermi * 100) / 100,
    bulkModulus: Math.round(bulkModulus * 10) / 10,
    shearModulus: Math.round(shearModulus * 10) / 10,
    debyeTemperature: Math.round(debyeTemp),
    magneticMoment: Math.round(magneticMoment * 100) / 100,
    omegaLog: Math.round(omegaLog * 10) / 10,
    muStar: Math.round(muStar * 1000) / 1000,
    nestingStrength: Math.round(nestingStrength * 1000) / 1000,
    correlationStrength: Math.round(correlationStrength * 1000) / 1000,
    dimensionality: Math.round(dimensionality * 10) / 10,
    metallicity: Math.round(metallicity * 1000) / 1000,
    topologicalIndex: Math.round(topologicalIndex * 1000) / 1000,
    propertyVector,
  };
}

function computeAvgBulkModulus(formula: string): number {
  const regex = /([A-Z][a-z]?)(\d*)/g;
  let match;
  let totalBulk = 0;
  let totalCount = 0;

  while ((match = regex.exec(formula)) !== null) {
    if (!match[1]) continue;
    const el = match[1];
    const count = match[2] ? parseInt(match[2]) : 1;
    const data = getElementData(el);
    if (data?.bulkModulus) {
      totalBulk += data.bulkModulus * count;
      totalCount += count;
    }
  }

  return totalCount > 0 ? totalBulk / totalCount : 0;
}

function computeMagneticMoment(formula: string, magneticOut: number[]): number {
  const regex = /([A-Z][a-z]?)(\d*)/g;
  let match;
  let hasMagnetic = false;

  while ((match = regex.exec(formula)) !== null) {
    if (!match[1]) continue;
    if (["Fe", "Co", "Ni", "Mn", "Cr", "Gd", "Nd", "Ce"].includes(match[1])) {
      hasMagnetic = true;
    }
  }

  if (hasMagnetic) {
    return Math.max(0, Math.abs(magneticOut[0] ?? 0) * 3 + 1.5);
  }
  return Math.max(0, sigmoid(magneticOut[0] ?? 0) * 0.5);
}

export function computePropertyGradient(
  formula: string,
  targetProperty: keyof MultiTaskPrediction,
  delta: number = 0.01
): Record<string, number> {
  const basePred = multiTaskPredict(formula);
  const baseValue = Number(basePred[targetProperty]) || 0;
  const gradients: Record<string, number> = {};

  const regex = /([A-Z][a-z]?)(\d*)/g;
  let match;
  const elements: string[] = [];

  while ((match = regex.exec(formula)) !== null) {
    if (match[1] && !elements.includes(match[1])) elements.push(match[1]);
  }

  for (const el of elements) {
    const perturbedFormula = perturbElement(formula, el, 1);
    if (perturbedFormula === formula) continue;
    try {
      const perturbedPred = multiTaskPredict(perturbedFormula);
      const perturbedValue = Number(perturbedPred[targetProperty]) || 0;
      gradients[el] = perturbedValue - baseValue;
    } catch {
      gradients[el] = 0;
    }
  }

  return gradients;
}

function perturbElement(formula: string, element: string, delta: number): string {
  const regex = new RegExp(`(${element})(\\d*)`, "g");
  return formula.replace(regex, (_, el, num) => {
    const count = num ? parseInt(num) : 1;
    const newCount = Math.max(1, count + delta);
    return newCount === 1 ? el : `${el}${newCount}`;
  });
}

const multiTaskStats = {
  totalPredictions: 0,
  avgPredictionTime: 0,
  propertyCorrelations: {} as Record<string, number>,
};

const correlationAccum = {
  n: 0,
  sumTc: 0, sumTc2: 0,
  sumLambda: 0, sumLambda2: 0, sumTcLambda: 0,
  sumDos: 0, sumDos2: 0, sumTcDos: 0,
  sumMetal: 0, sumMetal2: 0, sumTcMetal: 0,
};

function pearson(n: number, sumX: number, sumX2: number, sumY: number, sumY2: number, sumXY: number): number {
  if (n < 5) return 0;
  const denom = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  if (denom < 1e-12) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

export function trackMultiTaskPrediction(pred: MultiTaskPrediction): void {
  multiTaskStats.totalPredictions++;

  if (pred.predictedTc > 0) {
    const tc = pred.predictedTc;
    const lambda = Number.isFinite(pred.lambda) ? pred.lambda : 0;
    const dos = Number.isFinite(pred.dosAtFermi) ? pred.dosAtFermi : 0;
    const metal = Number.isFinite(pred.metallicity) ? pred.metallicity : 0;

    correlationAccum.n++;
    correlationAccum.sumTc += tc;
    correlationAccum.sumTc2 += tc * tc;
    correlationAccum.sumLambda += lambda;
    correlationAccum.sumLambda2 += lambda * lambda;
    correlationAccum.sumTcLambda += tc * lambda;
    correlationAccum.sumDos += dos;
    correlationAccum.sumDos2 += dos * dos;
    correlationAccum.sumTcDos += tc * dos;
    correlationAccum.sumMetal += metal;
    correlationAccum.sumMetal2 += metal * metal;
    correlationAccum.sumTcMetal += tc * metal;

    const n = correlationAccum.n;
    multiTaskStats.propertyCorrelations["tc_lambda"] = Math.round(
      pearson(n, correlationAccum.sumTc, correlationAccum.sumTc2, correlationAccum.sumLambda, correlationAccum.sumLambda2, correlationAccum.sumTcLambda) * 1000
    ) / 1000;
    multiTaskStats.propertyCorrelations["tc_dos"] = Math.round(
      pearson(n, correlationAccum.sumTc, correlationAccum.sumTc2, correlationAccum.sumDos, correlationAccum.sumDos2, correlationAccum.sumTcDos) * 1000
    ) / 1000;
    multiTaskStats.propertyCorrelations["tc_metallicity"] = Math.round(
      pearson(n, correlationAccum.sumTc, correlationAccum.sumTc2, correlationAccum.sumMetal, correlationAccum.sumMetal2, correlationAccum.sumTcMetal) * 1000
    ) / 1000;
  }
}

export function getMultiTaskStats() {
  return { ...multiTaskStats };
}