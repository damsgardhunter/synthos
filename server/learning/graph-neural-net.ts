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
  formula: string;
  prototype?: string;
}

interface GNNWeights {
  W_message: number[][];
  W_update: number[][];
  W_message2: number[][];
  W_update2: number[][];
  W_message3: number[][];
  W_update3: number[][];
  W_attn_query: number[][];
  W_attn_key: number[][];
  W_attn_query2: number[][];
  W_attn_key2: number[][];
  W_attn_query3: number[][];
  W_attn_key3: number[][];
  W_3body: number[][];
  W_3body_update: number[][];
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
  lambda: number;
}

export interface GNNPredictionWithUncertainty {
  tc: number;
  formationEnergy: number;
  lambda: number;
  uncertainty: number;
  phononStability: boolean;
  confidence: number;
}

const NODE_DIM = 20;
const HIDDEN_DIM = 28;
const EDGE_DIM = 7;
const OUTPUT_DIM = 5;

let cachedGNNModel: GNNWeights | null = null;
let modelTrainedAt = 0;
const MODEL_STALE_MS = 30 * 60 * 1000;

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

function computeValenceShellEncoding(atomicNumber: number, valenceElectrons: number): number {
  let period = 1;
  if (atomicNumber > 2) period = 2;
  if (atomicNumber > 10) period = 3;
  if (atomicNumber > 18) period = 4;
  if (atomicNumber > 36) period = 5;
  if (atomicNumber > 54) period = 6;
  if (atomicNumber > 86) period = 7;
  const shellFill = Math.min(1.0, valenceElectrons / (2 * period * period));
  return shellFill;
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
};

function assignSiteLabels(formula: string, prototype: string): Record<string, string> {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];
  if (!protoInfo) return {};

  const siteLabels = protoInfo.siteLabels;
  const assignment: Record<string, string> = {};

  const sorted = [...elements].sort((a, b) => {
    const dA = getElementData(a);
    const dB = getElementData(b);
    return (dA?.paulingElectronegativity ?? 1.5) - (dB?.paulingElectronegativity ?? 1.5);
  });

  for (let i = 0; i < sorted.length && i < siteLabels.length; i++) {
    assignment[sorted[i]] = siteLabels[i];
  }
  for (let i = siteLabels.length; i < sorted.length; i++) {
    assignment[sorted[i]] = siteLabels[siteLabels.length - 1];
  }

  return assignment;
}

export function buildPrototypeGraph(formula: string, prototype: string): CrystalGraph {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const protoInfo = PROTOTYPE_COORDINATIONS[prototype];

  if (!protoInfo) {
    return buildCrystalGraph(formula);
  }

  const siteAssignment = assignSiteLabels(formula, prototype);
  const nodes: NodeFeature[] = [];

  for (const el of elements) {
    const count = Math.round(counts[el]);
    const data = getElementData(el);
    const atomicNumber = data?.atomicNumber ?? 30;
    const en = data?.paulingElectronegativity ?? 1.5;
    const radius = data?.atomicRadius ?? 130;
    const valence = data?.valenceElectrons ?? 2;
    const mass = data?.atomicMass ?? 50;

    for (let i = 0; i < Math.min(count, 12); i++) {
      const embedding = buildEnhancedEmbedding(el, data, atomicNumber);
      nodes.push({ element: el, atomicNumber, electronegativity: en, atomicRadius: radius, valenceElectrons: valence, mass, embedding });
    }
  }

  if (nodes.length === 0) {
    nodes.push({
      element: "X", atomicNumber: 1, electronegativity: 1.5,
      atomicRadius: 100, valenceElectrons: 1, mass: 10,
      embedding: initVector(NODE_DIM, 0.1),
    });
  }

  const edges: EdgeFeature[] = [];
  const adjacency: number[][] = nodes.map(() => []);
  const lp = protoInfo.latticeParams;

  let nodeOffset = 0;
  const elementRanges: Record<string, { start: number; end: number }> = {};
  for (const el of elements) {
    const count = Math.min(Math.round(counts[el]), 12);
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
            if (adjacency[i].includes(j)) continue;

            const ri = nodes[i].atomicRadius / 100;
            const rj = nodes[j].atomicRadius / 100;
            const distance = (ri + rj) * 0.9;

            const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
            const bondOrder = enDiff > 1.5 ? 0.5 : enDiff > 0.5 ? 1.0 : 1.5;
            const massRatio = Math.min(nodes[i].mass, nodes[j].mass) / Math.max(nodes[i].mass, nodes[j].mass, 1);
            const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
            const ionicCharacter = Math.min(1.0, enDiff / 2.5);

            const edgeFeats = [
              Math.min(distance / 6.0, 1.0),
              bondOrder / 2.0,
              enDiff / 3.0,
              (nodes[i].valenceElectrons + nodes[j].valenceElectrons) / 16.0,
              massRatio,
              radiusSum,
              ionicCharacter,
            ];

            edges.push({ source: i, target: j, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
            edges.push({ source: j, target: i, distance, bondOrderEstimate: bondOrder, features: edgeFeats });
            adjacency[i].push(j);
            adjacency[j].push(i);
          }
        }
      }
    }
  }

  if (edges.length === 0 && nodes.length > 1) {
    for (let i = 0; i < nodes.length; i++) {
      const j = (i + 1) % nodes.length;
      const edgeFeats = [0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.3];
      edges.push({ source: i, target: j, distance: 2.5, bondOrderEstimate: 1.0, features: edgeFeats });
      edges.push({ source: j, target: i, distance: 2.5, bondOrderEstimate: 1.0, features: edgeFeats });
      adjacency[i].push(j);
      adjacency[j].push(i);
    }
  }

  const threeBodyFeatures = compute3BodyFeatures({ nodes, edges, threeBodyFeatures: [], adjacency, formula, prototype });
  return { nodes, edges, threeBodyFeatures, adjacency, formula, prototype };
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

function compute3BodyFeatures(graph: CrystalGraph): ThreeBodyFeature[] {
  const features: ThreeBodyFeature[] = [];
  const edgeMap = new Map<string, number>();

  for (const edge of graph.edges) {
    edgeMap.set(`${edge.source}-${edge.target}`, edge.distance);
  }

  for (let center = 0; center < graph.nodes.length; center++) {
    const neighbors = graph.adjacency[center];
    if (neighbors.length < 2) continue;

    for (let a = 0; a < neighbors.length; a++) {
      for (let b = a + 1; b < neighbors.length; b++) {
        const n1 = neighbors[a];
        const n2 = neighbors[b];
        const d1 = edgeMap.get(`${center}-${n1}`) ?? edgeMap.get(`${n1}-${center}`) ?? 2.5;
        const d2 = edgeMap.get(`${center}-${n2}`) ?? edgeMap.get(`${n2}-${center}`) ?? 2.5;
        const d12 = edgeMap.get(`${n1}-${n2}`) ?? edgeMap.get(`${n2}-${n1}`) ?? Math.sqrt(d1 * d1 + d2 * d2);

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
  const embeddings = graph.nodes.map(n => {
    const e = [...n.embedding];
    while (e.length < HIDDEN_DIM) e.push(0);
    return e.slice(0, HIDDEN_DIM);
  });

  const threeBodyAgg: number[][] = embeddings.map(() => initVector(HIDDEN_DIM));

  for (const tb of graph.threeBodyFeatures) {
    const angleFeature = tb.angle / Math.PI;
    const distFeature = Math.min(1.0, (tb.distance1 + tb.distance2) / 12.0);
    const asymmetry = Math.abs(tb.distance1 - tb.distance2) / Math.max(tb.distance1, tb.distance2, 0.01);

    const n1Embed = embeddings[tb.neighbor1] ?? initVector(HIDDEN_DIM);
    const n2Embed = embeddings[tb.neighbor2] ?? initVector(HIDDEN_DIM);

    const pairMsg = n1Embed.map((v, i) => (v + (n2Embed[i] ?? 0)) * 0.5 * angleFeature);
    const transformed = matVecMul(W_3body, pairMsg);

    for (let k = 0; k < HIDDEN_DIM; k++) {
      threeBodyAgg[tb.center][k] += (transformed[k] ?? 0) * (1.0 - asymmetry * 0.5) * distFeature;
    }
  }

  const newEmbeddings: number[][] = [];
  for (let i = 0; i < nNodes; i++) {
    const neighborCount = graph.threeBodyFeatures.filter(tb => tb.center === i).length;
    if (neighborCount > 0) {
      for (let k = 0; k < HIDDEN_DIM; k++) {
        threeBodyAgg[i][k] /= Math.max(neighborCount, 1);
      }
    }

    const combined = embeddings[i].map((v, k) => v + (threeBodyAgg[i][k] ?? 0));
    const updated = relu(matVecMul(W_3body_update, combined));
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
  ];
}

export function buildCrystalGraph(formula: string, structure?: any): CrystalGraph {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const nodes: NodeFeature[] = [];

  for (const el of elements) {
    const count = Math.round(counts[el]);
    const data = getElementData(el);
    const atomicNumber = data?.atomicNumber ?? 30;
    const en = data?.paulingElectronegativity ?? 1.5;
    const radius = data?.atomicRadius ?? 130;
    const valence = data?.valenceElectrons ?? 2;
    const mass = data?.atomicMass ?? 50;

    for (let i = 0; i < Math.min(count, 12); i++) {
      const embedding = buildEnhancedEmbedding(el, data, atomicNumber);
      nodes.push({ element: el, atomicNumber, electronegativity: en, atomicRadius: radius, valenceElectrons: valence, mass, embedding });
    }
  }

  if (nodes.length === 0) {
    nodes.push({
      element: "X", atomicNumber: 1, electronegativity: 1.5,
      atomicRadius: 100, valenceElectrons: 1, mass: 10,
      embedding: initVector(NODE_DIM, 0.1),
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

        const massRatio = Math.min(nodes[i].mass, nodes[j].mass) / Math.max(nodes[i].mass, nodes[j].mass, 1);
        const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;
        const ionicCharacter = Math.min(1.0, enDiff / 2.5);

        const edgeFeats = [
          Math.min(distance / cutoff, 1.0),
          bondOrder / 2.0,
          enDiff / 3.0,
          (nodes[i].valenceElectrons + nodes[j].valenceElectrons) / 16.0,
          massRatio,
          radiusSum,
          ionicCharacter,
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
      const edgeFeats = [0.5, 0.5, 0.3, 0.3, 0.5, 0.5, 0.3];
      edges.push({ source: i, target: j, distance: 2.5, bondOrderEstimate: 1.0, features: edgeFeats });
      edges.push({ source: j, target: i, distance: 2.5, bondOrderEstimate: 1.0, features: edgeFeats });
      adjacency[i].push(j);
      adjacency[j].push(i);
    }
  }

  const partialGraph: CrystalGraph = { nodes, edges, threeBodyFeatures: [], adjacency, formula };
  partialGraph.threeBodyFeatures = compute3BodyFeatures(partialGraph);
  return partialGraph;
}

export function attentionMessagePassingLayer(
  graph: CrystalGraph,
  W_message: number[][],
  W_update: number[][],
  W_query: number[][],
  W_key: number[][],
): number[][] {
  const nNodes = graph.nodes.length;
  const embeddings = graph.nodes.map(n => [...n.embedding]);

  const padded = embeddings.map(e => {
    while (e.length < HIDDEN_DIM) e.push(0);
    return e.slice(0, HIDDEN_DIM);
  });

  const newEmbeddings: number[][] = [];
  const scaleFactor = Math.sqrt(HIDDEN_DIM);

  for (let i = 0; i < nNodes; i++) {
    const neighbors = graph.adjacency[i];
    if (neighbors.length === 0) {
      newEmbeddings.push([...padded[i]]);
      continue;
    }

    const query = matVecMul(W_query, padded[i]);

    const attentionScores: number[] = [];
    const messages: number[][] = [];

    for (const j of neighbors) {
      const key = matVecMul(W_key, padded[j]);
      const score = dotProduct(query, key) / scaleFactor;
      attentionScores.push(score);
      messages.push(matVecMul(W_message, padded[j]));
    }

    const attentionWeights = softmax(attentionScores);

    const aggMessage = initVector(HIDDEN_DIM);
    for (let n = 0; n < neighbors.length; n++) {
      const w = attentionWeights[n];
      for (let k = 0; k < HIDDEN_DIM; k++) {
        aggMessage[k] += (messages[n][k] ?? 0) * w;
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
  attentionMessagePassingLayer(graph, weights.W_message, weights.W_update, weights.W_attn_query, weights.W_attn_key);
  if (graph.threeBodyFeatures.length > 0) {
    threeBodyInteractionLayer(graph, weights.W_3body, weights.W_3body_update);
  }
  attentionMessagePassingLayer(graph, weights.W_message2, weights.W_update2, weights.W_attn_query2, weights.W_attn_key2);
  attentionMessagePassingLayer(graph, weights.W_message3, weights.W_update3, weights.W_attn_query3, weights.W_attn_key3);

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
  const lambdaRaw = Math.max(0, out[4] ?? 0);

  return {
    formationEnergy: Math.round(formationEnergy * 1000) / 1000,
    phononStability: phononStabilityRaw > 0.5,
    predictedTc: Math.round(Math.max(0, predictedTcRaw) * 10) / 10,
    confidence: Math.round(Math.max(0.05, Math.min(0.95, confidenceRaw)) * 100) / 100,
    lambda: Math.round(Math.max(0, lambdaRaw) * 1000) / 1000,
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
    W_attn_query: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_attn_key: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_attn_query2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_attn_key2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_attn_query3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_attn_key3: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_3body: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
    W_3body_update: initMatrix(HIDDEN_DIM, HIDDEN_DIM, rng, 0.1),
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
    W_attn_query: w.W_attn_query.map(r => [...r]),
    W_attn_key: w.W_attn_key.map(r => [...r]),
    W_attn_query2: w.W_attn_query2.map(r => [...r]),
    W_attn_key2: w.W_attn_key2.map(r => [...r]),
    W_attn_query3: w.W_attn_query3.map(r => [...r]),
    W_attn_key3: w.W_attn_key3.map(r => [...r]),
    W_3body: w.W_3body.map(r => [...r]),
    W_3body_update: w.W_3body_update.map(r => [...r]),
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
  prototype?: string;
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

      const graph = sample.prototype
        ? buildPrototypeGraph(sample.formula, sample.prototype)
        : buildCrystalGraph(sample.formula, sample.structure);
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

      for (const wMat of [
        weights.W_message, weights.W_update,
        weights.W_message2, weights.W_update2,
        weights.W_message3, weights.W_update3,
        weights.W_attn_query, weights.W_attn_key,
        weights.W_attn_query2, weights.W_attn_key2,
        weights.W_attn_query3, weights.W_attn_key3,
        weights.W_3body, weights.W_3body_update,
        weights.W_mlp1,
      ]) {
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
  perturbMatrix(perturbed.W_attn_query);
  perturbMatrix(perturbed.W_attn_key);
  perturbMatrix(perturbed.W_attn_query2);
  perturbMatrix(perturbed.W_attn_key2);
  perturbMatrix(perturbed.W_attn_query3);
  perturbMatrix(perturbed.W_attn_key3);
  perturbMatrix(perturbed.W_3body);
  perturbMatrix(perturbed.W_3body_update);
  perturbMatrix(perturbed.W_mlp1);
  perturbMatrix(perturbed.W_mlp2);

  return perturbed;
}

export function gnnPredictWithUncertainty(formula: string, prototype?: string): GNNPredictionWithUncertainty {
  const weights = getGNNModel();
  const ensembleSize = 3;
  const predictions: GNNPrediction[] = [];

  const baseGraph = prototype
    ? buildPrototypeGraph(formula, prototype)
    : buildCrystalGraph(formula);
  const basePred = GNNPredict(baseGraph, weights);
  predictions.push(basePred);

  for (let run = 1; run < ensembleSize; run++) {
    const rng = seededRandom(42 + run * 137);
    const perturbedWeights = perturbWeights(weights, rng, 0.15);
    const graph = prototype
      ? buildPrototypeGraph(formula, prototype)
      : buildCrystalGraph(formula);
    const pred = GNNPredict(graph, perturbedWeights);
    predictions.push(pred);
  }

  const tcValues = predictions.map(p => p.predictedTc);
  const feValues = predictions.map(p => p.formationEnergy);
  const lambdaValues = predictions.map(p => p.lambda);

  const meanTc = tcValues.reduce((s, v) => s + v, 0) / tcValues.length;
  const meanFE = feValues.reduce((s, v) => s + v, 0) / feValues.length;
  const meanLambda = lambdaValues.reduce((s, v) => s + v, 0) / lambdaValues.length;

  const tcVariance = tcValues.reduce((s, v) => s + (v - meanTc) ** 2, 0) / tcValues.length;
  const feVariance = feValues.reduce((s, v) => s + (v - meanFE) ** 2, 0) / feValues.length;
  const lambdaVariance = lambdaValues.reduce((s, v) => s + (v - meanLambda) ** 2, 0) / lambdaValues.length;

  const tcStd = Math.sqrt(tcVariance);
  const feStd = Math.sqrt(feVariance);
  const lambdaStd = Math.sqrt(lambdaVariance);

  const normalizedTcUncertainty = meanTc > 0 ? tcStd / Math.max(meanTc, 1) : tcStd;
  const normalizedFeUncertainty = feStd;
  const normalizedLambdaUncertainty = meanLambda > 0 ? lambdaStd / Math.max(meanLambda, 0.1) : lambdaStd;

  const combinedUncertainty = Math.min(1.0,
    0.5 * normalizedTcUncertainty +
    0.3 * normalizedFeUncertainty +
    0.2 * normalizedLambdaUncertainty
  );

  const phononStabilityVotes = predictions.filter(p => p.phononStability).length;
  const phononStable = phononStabilityVotes > ensembleSize / 2;

  const avgConfidence = predictions.reduce((s, p) => s + p.confidence, 0) / predictions.length;

  return {
    tc: Math.round(meanTc * 10) / 10,
    formationEnergy: Math.round(meanFE * 1000) / 1000,
    lambda: Math.round(meanLambda * 1000) / 1000,
    uncertainty: Math.round(combinedUncertainty * 1000) / 1000,
    phononStability: phononStable,
    confidence: Math.round(avgConfidence * 100) / 100,
  };
}

export function getPrototypeCoordinations(): Record<string, PrototypeCoordination> {
  return PROTOTYPE_COORDINATIONS;
}
