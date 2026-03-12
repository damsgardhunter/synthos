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

const propertyNormStats = {
  tc: { mean: 50, std: 60 },
  bandGap: { mean: 1.5, std: 1.5 },
  dosAtFermi: { mean: 5, std: 5 },
  bulkModulus: { mean: 150, std: 100 },
  shearModulus: { mean: 60, std: 40 },
  debyeTemp: { mean: 350, std: 200 },
  magneticMoment: { mean: 1.0, std: 1.5 },
  omegaLog: { mean: 200, std: 150 },
  n: 0,
};

function updatePropertyNormStats(pred: {
  predictedTc: number; bandGap: number; dosAtFermi: number;
  bulkModulus: number; shearModulus: number; debyeTemp: number;
  magneticMoment: number; omegaLog: number;
}): void {
  const alpha = propertyNormStats.n < 50 ? 0.1 : 0.02;
  propertyNormStats.n++;

  for (const [key, val] of [
    ["tc", pred.predictedTc], ["bandGap", pred.bandGap], ["dosAtFermi", pred.dosAtFermi],
    ["bulkModulus", pred.bulkModulus], ["shearModulus", pred.shearModulus],
    ["debyeTemp", pred.debyeTemp], ["magneticMoment", pred.magneticMoment],
    ["omegaLog", pred.omegaLog],
  ] as [keyof typeof propertyNormStats, number][]) {
    const stat = propertyNormStats[key] as { mean: number; std: number };
    if (!stat || typeof val !== "number" || !Number.isFinite(val)) continue;
    const diff = val - stat.mean;
    stat.mean += alpha * diff;
    stat.std = Math.max(1e-6, Math.sqrt((1 - alpha) * stat.std * stat.std + alpha * diff * diff));
  }
}

function zNorm(value: number, key: keyof typeof propertyNormStats): number {
  const stat = propertyNormStats[key] as { mean: number; std: number };
  if (!stat || stat.std < 1e-6) return 0;
  return Math.max(-3, Math.min(3, (value - stat.mean) / stat.std));
}

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

function initVector(dim: number, positiveBias?: number): number[] {
  return new Array(dim).fill(positiveBias ?? 0);
}

const _mtBufferPool: Map<number, Float64Array[]> = new Map();

function mtAcquire(size: number): Float64Array {
  const pool = _mtBufferPool.get(size);
  if (pool && pool.length > 0) return pool.pop()!;
  return new Float64Array(size);
}

function mtRelease(buf: Float64Array): void {
  const size = buf.length;
  let pool = _mtBufferPool.get(size);
  if (!pool) { pool = []; _mtBufferPool.set(size, pool); }
  if (pool.length < 32) pool.push(buf);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
  const rows = mat.length;
  const out = mtAcquire(rows);
  for (let i = 0; i < rows; i++) {
    const row = mat[i];
    if (row.length !== vec.length) {
      mtRelease(out);
      throw new Error(
        `matVecMul shape mismatch: row ${i} has ${row.length} cols but vec has ${vec.length} elements`
      );
    }
    let s = 0;
    for (let j = 0; j < row.length; j++) {
      s += row[j] * vec[j];
    }
    out[i] = s;
  }
  const result = Array.from(out);
  mtRelease(out);
  return result;
}

function relu(vec: number[]): number[] {
  const n = vec.length;
  const out = mtAcquire(n);
  for (let i = 0; i < n; i++) out[i] = vec[i] > 0 ? vec[i] : 0;
  const result = Array.from(out);
  mtRelease(out);
  return result;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function vecAdd(a: number[], b: number[]): number[] {
  const n = a.length;
  const out = mtAcquire(n);
  for (let i = 0; i < n; i++) out[i] = a[i] + (b[i] ?? 0);
  const result = Array.from(out);
  mtRelease(out);
  return result;
}

let multiTaskWeights: MultiTaskWeights | null = null;

function getMultiTaskWeights(): MultiTaskWeights {
  if (multiTaskWeights) return multiTaskWeights;

  multiTaskWeights = {
    W_band: initMatrix(HEADS, HIDDEN),
    b_band: initVector(HEADS, 0.1),
    W_dos: initMatrix(HEADS, HIDDEN),
    b_dos: initVector(HEADS, 0.1),
    W_elastic: initMatrix(HEADS, HIDDEN),
    b_elastic: initVector(HEADS, 0.2),
    W_magnetic: initMatrix(HEADS, HIDDEN),
    b_magnetic: initVector(HEADS),
    W_phonon_detail: initMatrix(HEADS, HIDDEN),
    b_phonon_detail: initVector(HEADS, 0.15),
    W_topology: initMatrix(HEADS, HIDDEN),
    b_topology: initVector(HEADS),
  };

  return multiTaskWeights;
}

function extractGraphPooling(graph: CrystalGraph): number[] {
  const poolDim = Math.floor(HIDDEN / 2);
  const nNodes = graph.nodes.length;
  const meanPool = new Array(poolDim).fill(0);
  const maxPool = new Array(poolDim).fill(-Infinity);

  for (const node of graph.nodes) {
    for (let k = 0; k < poolDim; k++) {
      meanPool[k] += (node.embedding[k] ?? 0) / nNodes;
      maxPool[k] = Math.max(maxPool[k], node.embedding[k] ?? 0);
    }
  }

  const pooled = [...meanPool, ...maxPool.map(v => v === -Infinity ? 0 : v)];
  while (pooled.length < HIDDEN) pooled.push(0);
  return pooled.slice(0, HIDDEN);
}

interface MultiTaskOptions {
  skipPhysics?: boolean;
  highPriority?: boolean;
}

function multiTaskCore(formula: string) {
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

  return { basePred, bandOut, dosOut, elasticOut, magneticOut, phononOut, topoOut };
}

export function multiTaskPredict(formula: string, opts?: MultiTaskOptions): MultiTaskPrediction {
  const { basePred, bandOut, dosOut, elasticOut, magneticOut, phononOut, topoOut } = multiTaskCore(formula);

  let electronic: any = null;
  let phonon: any = null;
  let coupling: any = null;
  let features: any = null;

  const needsPhysics = !opts?.skipPhysics &&
    (opts?.highPriority || basePred.confidence < 0.3);

  if (needsPhysics) {
    try {
      electronic = computeElectronicStructure(formula, null);
      phonon = computePhononSpectrum(formula, electronic);
      coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);
      features = extractFeatures(formula);
    } catch {}
  } else {
    try { features = extractFeatures(formula); } catch {}
  }

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

  const magneticMoment = computeMagneticMoment(formula, magneticOut, correlationStrength);

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

  let dimensionality: number;
  if (features && features.dimensionality != null) {
    dimensionality = features.dimensionality;
  } else {
    const dimLogits = [
      topoOut[2] ?? -1.0,
      (topoOut[2] ?? 0) * 0.5 + (topoOut[3] ?? 0) * 0.3,
      (topoOut[2] ?? 0) * -0.3 + 1.0,
    ];
    const maxLogit = Math.max(...dimLogits);
    const exps = dimLogits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExps);
    dimensionality = 1 * probs[0] + 2 * probs[1] + 3 * probs[2];
  }

  const topologicalIndex = sigmoid(topoOut[3] ?? 0) * 0.3;

  updatePropertyNormStats({
    predictedTc: basePred.predictedTc, bandGap, dosAtFermi,
    bulkModulus, shearModulus, debyeTemp, magneticMoment, omegaLog,
  });

  const propertyVector = [
    basePred.formationEnergy,
    zNorm(basePred.predictedTc, "tc"),
    basePred.lambda,
    zNorm(bandGap, "bandGap"),
    zNorm(dosAtFermi, "dosAtFermi"),
    metallicity,
    zNorm(bulkModulus, "bulkModulus"),
    zNorm(shearModulus, "shearModulus"),
    zNorm(debyeTemp, "debyeTemp"),
    zNorm(magneticMoment, "magneticMoment"),
    zNorm(omegaLog, "omegaLog"),
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

function computeMagneticMoment(formula: string, magneticOut: number[], corrStrength?: number): number {
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
    let baseMoment = Math.max(0, Math.abs(magneticOut[0] ?? 0) * 3 + 1.5);
    if (corrStrength != null && corrStrength < 0.3) {
      const delocalPenalty = corrStrength / 0.3;
      baseMoment *= delocalPenalty;
    }
    return baseMoment;
  }
  return Math.max(0, sigmoid(magneticOut[0] ?? 0) * 0.5);
}

export function computePropertyGradient(
  formula: string,
  targetProperty: keyof MultiTaskPrediction,
  delta: number = 0.01
): Record<string, number> {
  const basePred = multiTaskPredict(formula, { skipPhysics: true });
  const baseValue = Number(basePred[targetProperty]) || 0;
  const gradients: Record<string, number> = {};

  const regex = /([A-Z][a-z]?)(\d*)/g;
  let match;
  const elements: string[] = [];

  while ((match = regex.exec(formula)) !== null) {
    if (match[1] && !elements.includes(match[1])) elements.push(match[1]);
  }

  const perturbedEntries: { el: string; formula: string }[] = [];
  for (const el of elements) {
    const pf = perturbElement(formula, el, 1);
    if (pf !== formula) perturbedEntries.push({ el, formula: pf });
  }

  const gnnWeights = getGNNModel();
  const mtWeights = getMultiTaskWeights();

  const graphs = perturbedEntries.map(e => {
    try { return buildCrystalGraph(e.formula); }
    catch { return null; }
  });

  for (let i = 0; i < perturbedEntries.length; i++) {
    const g = graphs[i];
    if (!g) { gradients[perturbedEntries[i].el] = 0; continue; }
    try {
      const pred = GNNPredict(g, gnnWeights);
      const perturbedValue = Number(pred[targetProperty as keyof typeof pred]) || 0;
      gradients[perturbedEntries[i].el] = perturbedValue - baseValue;
    } catch {
      gradients[perturbedEntries[i].el] = 0;
    }
  }

  return gradients;
}

function perturbElement(formula: string, element: string, delta: number): string {
  const regex = new RegExp(`(${element})(?![a-z])(\\d*)`, "g");
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

const CORR_WINDOW = 500;
const correlationWindow: { tc: number; lambda: number; dos: number; metal: number }[] = [];

function pearsonFromWindow(pairs: { x: number; y: number }[]): number {
  const n = pairs.length;
  if (n < 5) return 0;
  let sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0, sumXY = 0;
  for (const { x, y } of pairs) {
    sumX += x; sumY += y;
    sumX2 += x * x; sumY2 += y * y;
    sumXY += x * y;
  }
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

    correlationWindow.push({ tc, lambda, dos, metal });
    if (correlationWindow.length > CORR_WINDOW) {
      correlationWindow.shift();
    }

    const tcLambdaPairs = correlationWindow.map(w => ({ x: w.tc, y: w.lambda }));
    const tcDosPairs = correlationWindow.map(w => ({ x: w.tc, y: w.dos }));
    const tcMetalPairs = correlationWindow.map(w => ({ x: w.tc, y: w.metal }));

    multiTaskStats.propertyCorrelations["tc_lambda"] = Math.round(pearsonFromWindow(tcLambdaPairs) * 1000) / 1000;
    multiTaskStats.propertyCorrelations["tc_dos"] = Math.round(pearsonFromWindow(tcDosPairs) * 1000) / 1000;
    multiTaskStats.propertyCorrelations["tc_metallicity"] = Math.round(pearsonFromWindow(tcMetalPairs) * 1000) / 1000;
  }
}

export function getMultiTaskStats() {
  return { ...multiTaskStats };
}