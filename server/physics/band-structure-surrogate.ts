import {
  buildCrystalGraph,
  buildPrototypeGraph,
  type CrystalGraph,
  type NodeFeature,
  type EdgeFeature,
  type ThreeBodyFeature,
} from "../learning/graph-neural-net";
import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";

export interface BandSurrogatePrediction {
  formula: string;
  bandGap: number;
  bandGapType: "direct" | "indirect";
  flatBandScore: number;
  vhsProximity: number;
  nestingFromBands: number;
  dosPredicted: number;
  fsDimensionality: number;
  multiBandScore: number;
  bandwidthMin: number;
  bandTopologyClass: string;
  highSymmetryEnergies: Record<string, number[]>;
  confidence: number;
}

const NODE_DIM = 20;
const HIDDEN_DIM = 28;
const EDGE_DIM = 7;

interface BandSurrogateWeights {
  W_msg1: number[][];
  W_upd1: number[][];
  W_attn_q1: number[][];
  W_attn_k1: number[][];
  W_msg2: number[][];
  W_upd2: number[][];
  W_attn_q2: number[][];
  W_attn_k2: number[][];
  W_msg3: number[][];
  W_upd3: number[][];
  W_attn_q3: number[][];
  W_attn_k3: number[][];
  W_3body: number[][];
  W_3body_upd: number[][];
  W_readout1: number[][];
  b_readout1: number[];
  W_readout2: number[][];
  b_readout2: number[];
  W_head_bandGap: number[][];
  b_head_bandGap: number[];
  W_head_flatBand: number[][];
  b_head_flatBand: number[];
  W_head_vhs: number[][];
  b_head_vhs: number[];
  W_head_nesting: number[][];
  b_head_nesting: number[];
  W_head_dos: number[][];
  b_head_dos: number[];
  W_head_fsDim: number[][];
  b_head_fsDim: number[];
  W_head_multiBand: number[][];
  b_head_multiBand: number[];
  W_head_bwMin: number[][];
  b_head_bwMin: number[];
  W_head_topoClass: number[][];
  b_head_topoClass: number[];
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

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return sum;
}

function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  const maxVal = Math.max(...values);
  const exps = values.map(v => Math.exp(Math.min(v - maxVal, 20)));
  const sumExps = exps.reduce((s, e) => s + e, 0);
  return exps.map(e => e / Math.max(sumExps, 1e-10));
}

let cachedWeights: BandSurrogateWeights | null = null;

function getWeights(): BandSurrogateWeights {
  if (cachedWeights) return cachedWeights;

  const rng = seededRandom(314159);
  const headDim = 4;

  cachedWeights = {
    W_msg1: initMatrix(HIDDEN_DIM, NODE_DIM + EDGE_DIM, rng, 0.08),
    W_upd1: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng, 0.08),
    W_attn_q1: initMatrix(headDim, HIDDEN_DIM, rng, 0.1),
    W_attn_k1: initMatrix(headDim, HIDDEN_DIM, rng, 0.1),
    W_msg2: initMatrix(HIDDEN_DIM, HIDDEN_DIM + EDGE_DIM, rng, 0.08),
    W_upd2: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng, 0.08),
    W_attn_q2: initMatrix(headDim, HIDDEN_DIM, rng, 0.1),
    W_attn_k2: initMatrix(headDim, HIDDEN_DIM, rng, 0.1),
    W_msg3: initMatrix(HIDDEN_DIM, HIDDEN_DIM + EDGE_DIM, rng, 0.08),
    W_upd3: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng, 0.08),
    W_attn_q3: initMatrix(headDim, HIDDEN_DIM, rng, 0.1),
    W_attn_k3: initMatrix(headDim, HIDDEN_DIM, rng, 0.1),
    W_3body: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.06),
    W_3body_upd: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.06),
    W_readout1: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    b_readout1: initVector(HIDDEN_DIM),
    W_readout2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    b_readout2: initVector(HIDDEN_DIM),
    W_head_bandGap: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_bandGap: [0.5],
    W_head_flatBand: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_flatBand: [0.0],
    W_head_vhs: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_vhs: [0.0],
    W_head_nesting: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_nesting: [0.0],
    W_head_dos: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_dos: [1.0],
    W_head_fsDim: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_fsDim: [3.0],
    W_head_multiBand: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_multiBand: [0.0],
    W_head_bwMin: initMatrix(1, HIDDEN_DIM, rng, 0.15),
    b_head_bwMin: [1.0],
    W_head_topoClass: initMatrix(4, HIDDEN_DIM, rng, 0.15),
    b_head_topoClass: [0.0, 0.0, 0.0, 0.0],
  };

  return cachedWeights;
}

function attentionMessagePass(
  graph: CrystalGraph,
  nodeEmbeddings: number[][],
  W_msg: number[][],
  W_upd: number[][],
  W_attn_q: number[][],
  W_attn_k: number[][],
): number[][] {
  const n = graph.nodes.length;
  const newEmbeddings: number[][] = [];

  for (let i = 0; i < n; i++) {
    const neighbors = graph.adjacency[i] ?? [];
    if (neighbors.length === 0) {
      newEmbeddings.push([...nodeEmbeddings[i]]);
      continue;
    }

    const qi = matVecMul(W_attn_q, nodeEmbeddings[i]);
    const attnScores: number[] = [];
    const messages: number[][] = [];

    for (const j of neighbors) {
      const kj = matVecMul(W_attn_k, nodeEmbeddings[j]);
      const score = dotProduct(qi, kj) / Math.sqrt(qi.length);
      attnScores.push(score);

      const edgeIdx = graph.edges.findIndex(e => e.source === i && e.target === j);
      const edgeFeats = edgeIdx >= 0 ? graph.edges[edgeIdx].features : initVector(EDGE_DIM, 0.5);
      const msgInput = [...nodeEmbeddings[j].slice(0, W_msg[0].length - EDGE_DIM), ...edgeFeats];
      while (msgInput.length < W_msg[0].length) msgInput.push(0);
      const msg = relu(matVecMul(W_msg, msgInput.slice(0, W_msg[0].length)));
      messages.push(msg);
    }

    const attnWeights = softmax(attnScores);
    const aggMsg = initVector(HIDDEN_DIM);
    for (let k = 0; k < messages.length; k++) {
      for (let d = 0; d < HIDDEN_DIM; d++) {
        aggMsg[d] += attnWeights[k] * (messages[k][d] ?? 0);
      }
    }

    const combined = [...nodeEmbeddings[i], ...aggMsg];
    while (combined.length < W_upd[0].length) combined.push(0);
    const updated = relu(matVecMul(W_upd, combined.slice(0, W_upd[0].length)));
    newEmbeddings.push(updated);
  }

  return newEmbeddings;
}

function threeBodyLayer(
  graph: CrystalGraph,
  nodeEmbeddings: number[][],
  W_3body: number[][],
  W_3body_upd: number[][],
): number[][] {
  const n = graph.nodes.length;
  const threeBodyAgg: number[][] = nodeEmbeddings.map(() => initVector(HIDDEN_DIM));

  for (const tb of graph.threeBodyFeatures) {
    const angleFeature = tb.angle / Math.PI;
    const distFeature = Math.min(1.0, (tb.distance1 + tb.distance2) / 12.0);
    const asymmetry = Math.abs(tb.distance1 - tb.distance2) / Math.max(tb.distance1, tb.distance2, 0.01);

    const n1 = nodeEmbeddings[tb.neighbor1] ?? initVector(HIDDEN_DIM);
    const n2 = nodeEmbeddings[tb.neighbor2] ?? initVector(HIDDEN_DIM);

    const pairMsg = n1.map((v, i) => (v + (n2[i] ?? 0)) * 0.5 * angleFeature);
    const transformed = matVecMul(W_3body, pairMsg);

    for (let k = 0; k < HIDDEN_DIM; k++) {
      threeBodyAgg[tb.center][k] += (transformed[k] ?? 0) * (1.0 - asymmetry * 0.5) * distFeature;
    }
  }

  const newEmbeddings: number[][] = [];
  for (let i = 0; i < n; i++) {
    const count = graph.threeBodyFeatures.filter(tb => tb.center === i).length;
    if (count > 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        threeBodyAgg[i][k] /= Math.max(count, 1);
      }
    }
    const combined = nodeEmbeddings[i].map((v, k) => v + (threeBodyAgg[i][k] ?? 0));
    const updated = relu(matVecMul(W_3body_upd, combined));
    newEmbeddings.push(updated);
  }

  return newEmbeddings;
}

function globalReadout(embeddings: number[][]): number[] {
  if (embeddings.length === 0) return initVector(HIDDEN_DIM);

  const meanPool = initVector(HIDDEN_DIM);
  const maxPool = embeddings[0].map(v => v);

  for (const emb of embeddings) {
    for (let d = 0; d < HIDDEN_DIM; d++) {
      meanPool[d] += (emb[d] ?? 0) / embeddings.length;
      if ((emb[d] ?? 0) > maxPool[d]) maxPool[d] = emb[d] ?? 0;
    }
  }

  return meanPool.map((v, i) => (v + maxPool[i]) * 0.5);
}

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
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

function computePhysicsCalibration(formula: string): {
  metallicityHint: number;
  dElectronFraction: number;
  lightElementFraction: number;
  avgEN: number;
  enSpread: number;
  avgMass: number;
  avgDebye: number;
  hFraction: number;
  tmCount: number;
} {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  let dElectrons = 0;
  let lightAtoms = 0;
  let totalEN = 0;
  let enValues: number[] = [];
  let totalMass = 0;
  let totalDebye = 0;
  let debyeCount = 0;
  let tmCount = 0;

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const en = data.paulingElectronegativity ?? 1.5;
    totalEN += en * frac;
    enValues.push(en);
    totalMass += data.atomicMass * frac;
    if (data.debyeTemperature) {
      totalDebye += data.debyeTemperature * frac;
      debyeCount++;
    }
    if (isTransitionMetal(el) || isRareEarth(el) || isActinide(el)) {
      dElectrons += frac;
      tmCount++;
    }
    if (data.atomicMass < 15) lightAtoms += frac;
  }

  const enSpread = enValues.length >= 2
    ? Math.max(...enValues) - Math.min(...enValues)
    : 0;

  const nonmetals = ["H", "He", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalFrac = elements.filter(e => !nonmetals.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;

  const hFrac = (counts["H"] || 0) / totalAtoms;

  let metallicityHint = metalFrac;

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3
    && elements.some(e => ["Y", "La", "Ba", "Sr", "Ca", "Bi", "Tl", "Hg", "Nd", "Sm", "Gd"].includes(e));
  const isIronPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  const isBoride = elements.includes("B") && tmCount > 0;
  const isSuperhydride = hFrac > 0.5 && metalFrac > 0.05;

  if (isCuprate) metallicityHint = 0.75;
  else if (isIronPnictide) metallicityHint = 0.8;
  else if (isBoride) metallicityHint = 0.85;
  else if (isSuperhydride) metallicityHint = 0.9;
  else if (tmCount >= 1 && metalFrac > 0.3) metallicityHint = Math.max(metallicityHint, 0.7);
  else if (enSpread > 2.5) metallicityHint *= 0.5;
  if (elements.includes("O") && metalFrac < 0.3 && !isCuprate) metallicityHint *= 0.3;

  return {
    metallicityHint: Math.min(1, metallicityHint),
    dElectronFraction: dElectrons,
    lightElementFraction: lightAtoms,
    avgEN: totalEN,
    enSpread,
    avgMass: totalMass,
    avgDebye: debyeCount > 0 ? totalDebye : 300,
    hFraction: hFrac,
    tmCount,
  };
}

const HIGH_SYMMETRY_POINTS = ["Gamma", "X", "M", "R", "Z", "A"];

function estimateHighSymmetryEnergies(
  graphReadout: number[],
  calib: ReturnType<typeof computePhysicsCalibration>,
): Record<string, number[]> {
  const result: Record<string, number[]> = {};
  const rng = seededRandom(42);

  const baseEnergy = -calib.avgEN * 1.5;
  const bandwidth = Math.max(0.5, 8 - calib.dElectronFraction * 4);

  for (const point of HIGH_SYMMETRY_POINTS) {
    const nBands = Math.min(6, Math.max(2, Math.round(calib.tmCount * 2 + 2)));
    const energies: number[] = [];
    for (let b = 0; b < nBands; b++) {
      const bandCenter = baseEnergy + b * bandwidth / nBands;
      const kDispersion = (rng() - 0.5) * bandwidth * 0.3;
      const readoutInfluence = graphReadout[b % HIDDEN_DIM] * 0.5;
      energies.push(Number((bandCenter + kDispersion + readoutInfluence).toFixed(4)));
    }
    result[point] = energies.sort((a, b) => a - b);
  }

  return result;
}

const TOPO_CLASSES = ["trivial", "topological-insulator", "dirac-semimetal", "weyl-semimetal"];

function classifyTopology(logits: number[]): string {
  let maxIdx = 0;
  let maxVal = logits[0] ?? 0;
  for (let i = 1; i < logits.length && i < TOPO_CLASSES.length; i++) {
    if ((logits[i] ?? 0) > maxVal) {
      maxVal = logits[i] ?? 0;
      maxIdx = i;
    }
  }
  return TOPO_CLASSES[maxIdx];
}

function computeConfidence(graph: CrystalGraph, calib: ReturnType<typeof computePhysicsCalibration>): number {
  let conf = 0.5;

  if (graph.nodes.length >= 2 && graph.edges.length >= 2) conf += 0.1;
  if (graph.threeBodyFeatures.length > 0) conf += 0.05;
  if (calib.tmCount > 0) conf += 0.1;
  if (calib.metallicityHint > 0.3) conf += 0.1;
  if (graph.nodes.length <= 20) conf += 0.05;
  if (calib.enSpread < 2.0) conf += 0.05;

  return Math.min(0.95, Math.max(0.1, conf));
}

export function predictBandStructure(formula: string, prototype?: string): BandSurrogatePrediction {
  const graph = prototype
    ? buildPrototypeGraph(formula, prototype)
    : buildCrystalGraph(formula);

  const weights = getWeights();
  const calib = computePhysicsCalibration(formula);

  let embeddings: number[][] = graph.nodes.map(n => {
    const emb = [...n.embedding];
    while (emb.length < HIDDEN_DIM) emb.push(0);
    return emb.slice(0, HIDDEN_DIM);
  });

  embeddings = attentionMessagePass(
    graph, embeddings,
    weights.W_msg1, weights.W_upd1,
    weights.W_attn_q1, weights.W_attn_k1,
  );

  embeddings = attentionMessagePass(
    graph, embeddings,
    weights.W_msg2, weights.W_upd2,
    weights.W_attn_q2, weights.W_attn_k2,
  );

  embeddings = threeBodyLayer(graph, embeddings, weights.W_3body, weights.W_3body_upd);

  embeddings = attentionMessagePass(
    graph, embeddings,
    weights.W_msg3, weights.W_upd3,
    weights.W_attn_q3, weights.W_attn_k3,
  );

  const graphVec = globalReadout(embeddings);

  const hidden1 = relu(vecAdd(matVecMul(weights.W_readout1, graphVec), weights.b_readout1));
  const hidden2 = relu(vecAdd(matVecMul(weights.W_readout2, hidden1), weights.b_readout2));

  const rawBandGap = vecAdd(matVecMul(weights.W_head_bandGap, hidden2), weights.b_head_bandGap)[0];
  const rawFlatBand = vecAdd(matVecMul(weights.W_head_flatBand, hidden2), weights.b_head_flatBand)[0];
  const rawVhs = vecAdd(matVecMul(weights.W_head_vhs, hidden2), weights.b_head_vhs)[0];
  const rawNesting = vecAdd(matVecMul(weights.W_head_nesting, hidden2), weights.b_head_nesting)[0];
  const rawDos = vecAdd(matVecMul(weights.W_head_dos, hidden2), weights.b_head_dos)[0];
  const rawFsDim = vecAdd(matVecMul(weights.W_head_fsDim, hidden2), weights.b_head_fsDim)[0];
  const rawMultiBand = vecAdd(matVecMul(weights.W_head_multiBand, hidden2), weights.b_head_multiBand)[0];
  const rawBwMin = vecAdd(matVecMul(weights.W_head_bwMin, hidden2), weights.b_head_bwMin)[0];
  const topoLogits = vecAdd(matVecMul(weights.W_head_topoClass, hidden2), weights.b_head_topoClass);

  let bandGap = Math.max(0, rawBandGap);
  const dosProxy = Math.max(0, rawDos);
  const highDOS = dosProxy > 2.0;

  if (calib.metallicityHint > 0.7) {
    if (highDOS) {
      bandGap *= 0.05;
    } else {
      const correlationPenalty = Math.max(0.05, 1.0 - calib.metallicityHint);
      bandGap *= correlationPenalty;
    }
  } else if (calib.metallicityHint > 0.4) {
    bandGap *= 0.2 + (1.0 - calib.metallicityHint) * 0.3;
  }
  bandGap = Number(Math.min(10, bandGap).toFixed(4));

  const isMetal = calib.metallicityHint > 0.5 || bandGap < 0.1;

  let flatBandScore = sigmoid(rawFlatBand);
  if (calib.dElectronFraction > 0.3) flatBandScore = Math.min(1, flatBandScore * 1.4);
  if (calib.dElectronFraction > 0.5) flatBandScore = Math.min(1, flatBandScore + 0.15);
  if (calib.lightElementFraction > 0.3 && calib.tmCount === 0) flatBandScore *= 0.7;
  flatBandScore = Number(Math.min(1, flatBandScore).toFixed(4));

  let vhsProximity = sigmoid(rawVhs);
  if (isMetal) {
    vhsProximity = Math.min(1, vhsProximity + calib.dElectronFraction * 0.25);
    if (calib.tmCount >= 2) vhsProximity = Math.min(1, vhsProximity * 1.3);
  } else {
    vhsProximity *= 0.2;
  }
  vhsProximity = Number(vhsProximity.toFixed(4));

  let nestingFromBands = sigmoid(rawNesting);
  if (isMetal) {
    if (calib.tmCount >= 2) nestingFromBands = Math.min(1, nestingFromBands + 0.15);
    if (calib.dElectronFraction > 0.3) nestingFromBands = Math.min(1, nestingFromBands * 1.2);
  } else {
    nestingFromBands *= 0.15;
  }
  nestingFromBands = Number(nestingFromBands.toFixed(4));

  let dosPredicted = Math.max(0.01, Math.abs(rawDos));
  if (isMetal) {
    const baseDOS = 0.5 + calib.dElectronFraction * 3.0 + calib.tmCount * 0.3;
    dosPredicted = Math.max(baseDOS, dosPredicted);
    if (calib.hFraction > 0.3) dosPredicted *= 1.3;
  } else {
    dosPredicted *= 0.05;
  }
  dosPredicted = Number(Math.min(15, dosPredicted).toFixed(4));

  let fsDimensionality: number;
  if (!isMetal) {
    fsDimensionality = 0;
  } else if (calib.dElectronFraction > 0.4) {
    fsDimensionality = 2 + sigmoid(rawFsDim - 3) * 1;
  } else if (calib.tmCount >= 1) {
    fsDimensionality = 2.5 + sigmoid(rawFsDim - 3) * 0.5;
  } else {
    fsDimensionality = 3;
  }
  fsDimensionality = Number(Math.min(3, fsDimensionality).toFixed(2));

  let multiBandScore = sigmoid(rawMultiBand);
  if (isMetal) {
    if (calib.tmCount >= 2) multiBandScore = Math.min(1, multiBandScore + 0.25);
    else if (calib.tmCount >= 1) multiBandScore = Math.min(1, multiBandScore + 0.1);
    if (calib.dElectronFraction > 0.3) multiBandScore = Math.min(1, multiBandScore * 1.2);
  } else {
    multiBandScore *= 0.15;
  }
  multiBandScore = Number(multiBandScore.toFixed(4));

  let bandwidthMin = Math.max(0.001, Math.abs(rawBwMin));
  if (flatBandScore > 0.7) bandwidthMin = Math.min(bandwidthMin, 0.05);
  if (isMetal && flatBandScore < 0.4) bandwidthMin = Math.max(bandwidthMin, 0.3);
  bandwidthMin = Number(Math.min(10, bandwidthMin).toFixed(4));

  const bandTopologyClass = classifyTopology(topoLogits);
  const bandGapType: "direct" | "indirect" = calib.enSpread > 1.0 ? "indirect" : "direct";
  const highSymmetryEnergies = estimateHighSymmetryEnergies(hidden2, calib);
  const confidence = computeConfidence(graph, calib);

  return {
    formula,
    bandGap,
    bandGapType,
    flatBandScore,
    vhsProximity,
    nestingFromBands,
    dosPredicted,
    fsDimensionality,
    multiBandScore,
    bandwidthMin,
    bandTopologyClass,
    highSymmetryEnergies,
    confidence,
  };
}

export function getBandSurrogateMLFeatures(prediction: BandSurrogatePrediction): Record<string, number> {
  return {
    bandGap: prediction.bandGap,
    flatBandScore: prediction.flatBandScore,
    vhsProximity: prediction.vhsProximity,
    nestingFromBands: prediction.nestingFromBands,
    dosPredicted: prediction.dosPredicted,
    fsDimensionality: prediction.fsDimensionality,
    multiBandScore: prediction.multiBandScore,
    bandwidthMin: prediction.bandwidthMin,
  };
}
