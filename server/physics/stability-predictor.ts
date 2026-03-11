import { ELEMENTAL_DATA, getElementData } from "../learning/elemental-data";
import { computeMiedemaFormationEnergy } from "../learning/phase-diagram-engine";
import { classifyFamily } from "../learning/utils";

export interface StabilityPrediction {
  formula: string;
  synthesizabilityScore: number;
  predictedFormationEnergy: number;
  stabilityClass: "stable" | "metastable" | "unstable";
  decompositionRisk: number;
  confidence: number;
  details: {
    electronegativitySpread: number;
    toleranceFactor: number | null;
    pettiforDistance: number;
    miedemaEnergy: number;
    miedemaRaw: number;
    hydrideCorrection: number;
    elementCompatibility: number;
    prototypeMatch: number;
    valenceMismatch: number;
    sizeRatioScore: number;
    volumePerAtom: number;
    packingFraction: number;
    volumeGhostFlag: boolean;
  };
}

interface CompositionNode {
  element: string;
  fraction: number;
  atomicNumber: number;
  electronegativity: number;
  atomicRadius: number;
  valenceElectrons: number;
  mass: number;
  pettiforScale: number;
  meltingPoint: number;
  bulkModulus: number;
  embedding: number[];
}

interface CompositionEdge {
  source: number;
  target: number;
  features: number[];
}

interface CompositionGraph {
  nodes: CompositionNode[];
  edges: CompositionEdge[];
  formula: string;
}

interface StabilityWeights {
  W_msg1: number[][];
  W_upd1: number[][];
  W_msg2: number[][];
  W_upd2: number[][];
  W_msg3: number[][];
  W_upd3: number[][];
  W_out1: number[][];
  b_out1: number[];
  W_out2: number[][];
  b_out2: number[];
}

const NODE_DIM = 14;
const HIDDEN_DIM = 20;
const EDGE_DIM = 8;

function expandParentheses(formula: string): string {
  let result = formula.replace(/\[/g, "(").replace(/\]/g, ")");
  const parenRegex = /\(([^()]+)\)(\d*\.?\d*)/;
  let iterations = 0;
  while (result.includes("(") && iterations < 20) {
    const prev = result;
    result = result.replace(parenRegex, (_, group: string, mult: string) => {
      const m = mult ? parseFloat(mult) : 1;
      if (isNaN(m) || m <= 0) return group;
      if (m === 1) return group;
      return group.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_x: string, el: string, num: string) => {
        const n = num ? parseFloat(num) : 1;
        const newN = (isNaN(n) || n <= 0 ? 1 : n) * m;
        return newN === 1 ? el : `${el}${newN}`;
      });
    });
    if (result === prev) break;
    iterations++;
  }
  return result.replace(/[()]/g, "");
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  let cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  cleaned = expandParentheses(cleaned);
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + (isNaN(num) || num <= 0 ? 1 : num);
  }
  return counts;
}

function computeVolumePerAtom(formula: string): { volumePerAtom: number; packingFraction: number } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  let totalVolume = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const radiusPm = data?.atomicRadius ?? 130;
    const radiusA = radiusPm / 100;
    const atomVolA3 = (4 / 3) * Math.PI * Math.pow(radiusA, 3);
    totalVolume += atomVolA3 * (counts[el] || 1);
  }

  const volumePerAtom = totalVolume / totalAtoms;

  const idealPackingFCC = 0.74;
  const packingFraction = Math.min(1.0, idealPackingFCC * (1 + 0.05 * (elements.length - 1)));

  return { volumePerAtom, packingFraction };
}

function applyHydrideCorrection(formula: string, rawMiedema: number): { corrected: number; correction: number } {
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const hFraction = (counts["H"] || 0) / totalAtoms;

  if (hFraction <= 0.5) {
    return { corrected: rawMiedema, correction: 0 };
  }

  const elements = Object.keys(counts).filter(e => e !== "H");
  let avgHydrideAffinity = 0;
  let totalFrac = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const en = data?.paulingElectronegativity ?? 1.5;
    const frac = (counts[el] || 1) / totalAtoms;
    const affinity = en < 1.5 ? -0.3 : en < 2.0 ? -0.1 : 0.2;
    avgHydrideAffinity += affinity * frac;
    totalFrac += frac;
  }
  avgHydrideAffinity /= Math.max(totalFrac, 0.01);

  const destabilizationPenalty = Math.pow(hFraction - 0.5, 1.5) * 2.0;

  const pvPenalty = hFraction > 0.7 ? (hFraction - 0.7) * 1.5 : 0;

  const correction = destabilizationPenalty + pvPenalty - avgHydrideAffinity * 0.5;

  const corrected = rawMiedema + correction;

  return { corrected, correction };
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

let cachedWeights: StabilityWeights | null = null;

function getWeights(): StabilityWeights {
  if (cachedWeights) return cachedWeights;
  const rng = seededRandom(98765);
  cachedWeights = {
    W_msg1: initMatrix(HIDDEN_DIM, NODE_DIM + EDGE_DIM, rng, 0.15),
    W_upd1: initMatrix(HIDDEN_DIM, HIDDEN_DIM + NODE_DIM, rng, 0.12),
    W_msg2: initMatrix(HIDDEN_DIM, HIDDEN_DIM + EDGE_DIM, rng, 0.12),
    W_upd2: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng, 0.10),
    W_msg3: initMatrix(HIDDEN_DIM, HIDDEN_DIM + EDGE_DIM, rng, 0.10),
    W_upd3: initMatrix(HIDDEN_DIM, HIDDEN_DIM * 2, rng, 0.08),
    W_out1: initMatrix(12, HIDDEN_DIM, rng, 0.15),
    b_out1: new Array(12).fill(0),
    W_out2: initMatrix(4, 12, rng, 0.12),
    b_out2: new Array(4).fill(0),
  };
  return cachedWeights;
}

function buildCompositionGraph(formula: string): CompositionGraph {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const nodes: CompositionNode[] = [];
  for (const el of elements) {
    const data = getElementData(el);
    const fraction = (counts[el] || 1) / totalAtoms;
    const atomicNumber = data?.atomicNumber ?? 30;
    const en = data?.paulingElectronegativity ?? 1.5;
    const radius = data?.atomicRadius ?? 130;
    const valence = data?.valenceElectrons ?? 2;
    const mass = data?.atomicMass ?? 50;
    const pettifor = data?.pettiforScale ?? 50;
    const mp = data?.meltingPoint ?? 1000;
    const bulk = data?.bulkModulus ?? 50;

    const embedding = [
      atomicNumber / 100,
      en / 4.0,
      radius / 250,
      valence / 8,
      mass / 250,
      fraction,
      pettifor / 103,
      mp / 4000,
      bulk / 500,
      (data?.debyeTemperature ?? 300) / 2000,
      (data?.firstIonizationEnergy ?? 7) / 25,
      Math.max(0, data?.electronAffinity ?? 0) / 4.0,
      (data?.miedemaPhiStar ?? 4.0) / 8.0,
      (data?.miedemaNws13 ?? 1.3) / 3.0,
    ];

    nodes.push({
      element: el,
      fraction,
      atomicNumber,
      electronegativity: en,
      atomicRadius: radius,
      valenceElectrons: valence,
      mass,
      pettiforScale: pettifor,
      meltingPoint: mp,
      bulkModulus: bulk,
      embedding,
    });
  }

  if (nodes.length === 0) {
    nodes.push({
      element: "X", fraction: 1.0, atomicNumber: 1, electronegativity: 1.5,
      atomicRadius: 100, valenceElectrons: 1, mass: 10, pettiforScale: 50,
      meltingPoint: 1000, bulkModulus: 50,
      embedding: new Array(NODE_DIM).fill(0.1),
    });
  }

  const edges: CompositionEdge[] = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const ni = nodes[i];
      const nj = nodes[j];
      const enDiff = Math.abs(ni.electronegativity - nj.electronegativity);
      const radiusRatio = Math.min(ni.atomicRadius, nj.atomicRadius) / Math.max(ni.atomicRadius, nj.atomicRadius, 1);
      const massRatio = Math.min(ni.mass, nj.mass) / Math.max(ni.mass, nj.mass, 1);
      const valenceDiff = Math.abs(ni.valenceElectrons - nj.valenceElectrons) / 8;
      const pettiforDist = Math.abs(ni.pettiforScale - nj.pettiforScale) / 103;
      const fractionProduct = ni.fraction * nj.fraction;
      const meltingPointRatio = Math.min(ni.meltingPoint, nj.meltingPoint) / Math.max(ni.meltingPoint, nj.meltingPoint, 1);
      const ionicChar = Math.min(1.0, enDiff / 2.5);

      const feats = [
        enDiff / 3.0,
        radiusRatio,
        massRatio,
        valenceDiff,
        pettiforDist,
        fractionProduct * 4,
        meltingPointRatio,
        ionicChar,
      ];

      edges.push({ source: i, target: j, features: feats });
      edges.push({ source: j, target: i, features: feats });
    }
  }

  return { nodes, edges, formula };
}

function messagePassingLayer(
  nodeEmbeddings: number[][],
  edges: CompositionEdge[],
  W_msg: number[][],
  W_upd: number[][],
): number[][] {
  const nNodes = nodeEmbeddings.length;
  const newEmbeddings: number[][] = [];

  for (let i = 0; i < nNodes; i++) {
    const incomingEdges = edges.filter(e => e.target === i);
    let aggregated = new Array(W_msg.length).fill(0);

    for (const edge of incomingEdges) {
      const sourceEmbed = nodeEmbeddings[edge.source];
      const msgInput = [...sourceEmbed, ...edge.features];
      while (msgInput.length < W_msg[0].length) msgInput.push(0);
      const msg = relu(matVecMul(W_msg, msgInput.slice(0, W_msg[0].length)));
      aggregated = vecAdd(aggregated, msg);
    }

    if (incomingEdges.length > 0) {
      aggregated = aggregated.map(v => v / incomingEdges.length);
    }

    const updateInput = [...aggregated, ...nodeEmbeddings[i]];
    while (updateInput.length < W_upd[0].length) updateInput.push(0);
    const updated = relu(matVecMul(W_upd, updateInput.slice(0, W_upd[0].length)));
    newEmbeddings.push(updated);
  }

  return newEmbeddings;
}

function globalPooling(nodeEmbeddings: number[][], fractions: number[]): number[] {
  if (nodeEmbeddings.length === 0) return new Array(HIDDEN_DIM).fill(0);
  const dim = nodeEmbeddings[0].length;
  const pooled = new Array(dim).fill(0);
  for (let i = 0; i < nodeEmbeddings.length; i++) {
    const w = fractions[i] ?? (1 / nodeEmbeddings.length);
    for (let j = 0; j < dim; j++) {
      pooled[j] += (nodeEmbeddings[i][j] ?? 0) * w;
    }
  }
  return pooled;
}

const KNOWN_STABLE_PROTOTYPES: Record<string, string[]> = {
  "Perovskite": ["ABO3"],
  "AlB2": ["AB2"],
  "A15": ["A3B"],
  "ThCr2Si2": ["ABC2"],
  "Laves": ["AB2"],
  "Heusler": ["A2BC"],
  "Spinel": ["AB2O4"],
  "NaCl": ["AB"],
  "CsCl": ["AB"],
  "Fluorite": ["AO2"],
};

function computeGoldschmidtTolerance(formula: string): number | null {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  const hasO = elements.includes("O");
  if (!hasO || elements.length < 3) return null;

  const nonOElements = elements.filter(e => e !== "O");
  if (nonOElements.length < 2) return null;

  const sorted = nonOElements.sort((a, b) => {
    const dA = getElementData(a);
    const dB = getElementData(b);
    return (dB?.atomicRadius ?? 0) - (dA?.atomicRadius ?? 0);
  });

  const rA = (getElementData(sorted[0])?.atomicRadius ?? 130) / 100;
  const rB = (getElementData(sorted[1])?.atomicRadius ?? 130) / 100;
  const rO = 1.40;

  const t = (rA + rO) / (Math.SQRT2 * (rB + rO));
  return t;
}

function computePettiforProximity(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 1.0;

  let maxDist = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const pA = getElementData(elements[i])?.pettiforScale ?? 50;
      const pB = getElementData(elements[j])?.pettiforScale ?? 50;
      maxDist = Math.max(maxDist, Math.abs(pA - pB));
    }
  }
  return Math.min(1.0, maxDist / 100);
}

function computeElementCompatibility(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0.8;

  let compatibility = 1.0;

  const enValues = elements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.5);
  const enSpread = Math.max(...enValues) - Math.min(...enValues);
  if (enSpread > 2.5) compatibility *= 0.6;
  else if (enSpread > 1.8) compatibility *= 0.8;
  else if (enSpread < 0.3 && elements.length > 2) compatibility *= 0.9;

  const radii = elements.map(el => getElementData(el)?.atomicRadius ?? 130);
  const radiusRatio = Math.min(...radii) / Math.max(...radii);
  if (radiusRatio < 0.4) compatibility *= 0.7;
  else if (radiusRatio < 0.6) compatibility *= 0.85;

  const halogens = ["F", "Cl", "Br", "I"];
  const alkali = ["Li", "Na", "K", "Rb", "Cs"];
  const hasHalogen = elements.some(e => halogens.includes(e));
  const hasAlkali = elements.some(e => alkali.includes(e));
  if (hasHalogen && !hasAlkali && !elements.includes("O")) compatibility *= 0.7;

  const nobleGases = ["He", "Ne", "Ar", "Kr", "Xe"];
  if (elements.some(e => nobleGases.includes(e))) compatibility *= 0.1;

  return Math.max(0, Math.min(1, compatibility));
}

function computePrototypeMatch(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const sorted = elements.sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
  const ratios = sorted.map(el => (counts[el] || 1) / totalAtoms);

  if (elements.length === 2) {
    const r = (counts[sorted[0]] || 1) / (counts[sorted[1]] || 1);
    if (Math.abs(r - 1) < 0.1) return 0.9;
    if (Math.abs(r - 2) < 0.2) return 0.85;
    if (Math.abs(r - 3) < 0.3) return 0.8;
  }

  if (elements.length === 3) {
    const hasO = elements.includes("O");
    if (hasO) {
      const oFrac = (counts["O"] || 0) / totalAtoms;
      if (Math.abs(oFrac - 0.6) < 0.1) return 0.85;
      if (Math.abs(oFrac - 0.5) < 0.1) return 0.8;
    }
    if (ratios[0] < 0.5 && ratios[1] > 0.15) return 0.75;
  }

  if (elements.length >= 5) return 0.5;
  if (elements.length === 4) return 0.65;

  return 0.7;
}

function computeValenceMismatch(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  let totalValence = 0;
  for (const el of elements) {
    const data = getElementData(el);
    totalValence += (data?.valenceElectrons ?? 2) * (counts[el] || 1);
  }
  const avgValence = totalValence / totalAtoms;

  const hasO = elements.includes("O");
  const hasN = elements.includes("N");
  const hasH = elements.includes("H");

  if (hasO) {
    const oCount = counts["O"] || 0;
    const nonOAtoms = totalAtoms - oCount;
    const nonOValence = totalValence - oCount * 6;
    const expectedOxygen = nonOValence / 2;
    const ratio = oCount / Math.max(expectedOxygen, 0.5);
    return Math.min(1.0, Math.abs(ratio - 1.0));
  }

  if (avgValence > 6 || avgValence < 1) return 0.5;
  return Math.min(1.0, Math.abs(avgValence - 4) / 4);
}

function computeSizeRatioScore(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0.9;

  const radii = elements.map(el => getElementData(el)?.atomicRadius ?? 130);
  const minR = Math.min(...radii);
  const maxR = Math.max(...radii);
  const ratio = minR / maxR;

  if (ratio > 0.85) return 0.95;
  if (ratio > 0.7) return 0.9;
  if (ratio > 0.55) return 0.8;
  if (ratio > 0.4) return 0.65;
  if (ratio > 0.25) return 0.5;
  return 0.3;
}

const FAMILY_STABILITY_BIAS: Record<string, { synthBias: number; decompBias: number; confBias: number }> = {
  "Cuprates":             { synthBias: 0.10, decompBias: -0.08, confBias: 0.10 },
  "Pnictides":            { synthBias: 0.08, decompBias: -0.06, confBias: 0.08 },
  "Intermetallics":       { synthBias: 0.12, decompBias: -0.10, confBias: 0.12 },
  "Chalcogenides":        { synthBias: 0.06, decompBias: -0.05, confBias: 0.06 },
  "Layered-chalcogenide": { synthBias: 0.07, decompBias: -0.05, confBias: 0.07 },
  "Layered-pnictide":     { synthBias: 0.08, decompBias: -0.06, confBias: 0.08 },
  "Nickelates":           { synthBias: 0.06, decompBias: -0.04, confBias: 0.06 },
  "Borocarbides":         { synthBias: 0.08, decompBias: -0.06, confBias: 0.08 },
  "Heavy Fermions":       { synthBias: 0.04, decompBias: -0.02, confBias: 0.04 },
  "Borides":              { synthBias: 0.09, decompBias: -0.07, confBias: 0.09 },
  "Carbides":             { synthBias: 0.07, decompBias: -0.05, confBias: 0.07 },
  "Nitrides":             { synthBias: 0.06, decompBias: -0.04, confBias: 0.06 },
  "Oxides":               { synthBias: 0.05, decompBias: -0.03, confBias: 0.05 },
  "Alloys":               { synthBias: 0.10, decompBias: -0.08, confBias: 0.10 },
  "Clathrates":           { synthBias: 0.05, decompBias: -0.03, confBias: 0.05 },
  "Kagome":               { synthBias: 0.04, decompBias: -0.02, confBias: 0.04 },
  "Silicides":            { synthBias: 0.06, decompBias: -0.04, confBias: 0.06 },
  "Hydrides":             { synthBias: -0.05, decompBias: 0.10, confBias: -0.05 },
  "Sulfides":             { synthBias: 0.05, decompBias: -0.03, confBias: 0.05 },
  "Other":                { synthBias: 0.00, decompBias: 0.00, confBias: 0.00 },
  "Mixed-mechanism":      { synthBias: 0.03, decompBias: -0.02, confBias: 0.03 },
  "Intercalated-layered": { synthBias: 0.04, decompBias: -0.02, confBias: 0.04 },
  "Phosphides":           { synthBias: 0.05, decompBias: -0.03, confBias: 0.05 },
};

function computeFamilyContext(formula: string): { family: string; synthBias: number; decompBias: number; confBias: number } {
  const family = classifyFamily(formula);
  const bias = FAMILY_STABILITY_BIAS[family] ?? { synthBias: 0, decompBias: 0, confBias: 0 };
  return { family, ...bias };
}

export function predictStability(formula: string): StabilityPrediction {
  const graph = buildCompositionGraph(formula);
  const weights = getWeights();

  let embeddings = graph.nodes.map(n => {
    const e = [...n.embedding];
    while (e.length < NODE_DIM) e.push(0);
    return e.slice(0, NODE_DIM);
  });

  embeddings = messagePassingLayer(embeddings, graph.edges, weights.W_msg1, weights.W_upd1);
  embeddings = messagePassingLayer(embeddings, graph.edges, weights.W_msg2, weights.W_upd2);
  embeddings = messagePassingLayer(embeddings, graph.edges, weights.W_msg3, weights.W_upd3);

  const fractions = graph.nodes.map(n => n.fraction);
  const pooled = globalPooling(embeddings, fractions);
  while (pooled.length < HIDDEN_DIM) pooled.push(0);

  const h1 = relu(vecAdd(matVecMul(weights.W_out1, pooled.slice(0, HIDDEN_DIM)), weights.b_out1));
  const output = vecAdd(matVecMul(weights.W_out2, h1), weights.b_out2);

  const rawSynthesizability = sigmoid(output[0] ?? 0);
  const rawFormationEnergy = (output[1] ?? 0) * 2.0;
  const rawDecompRisk = sigmoid(output[2] ?? 0);
  const rawConfidence = sigmoid(output[3] ?? 0.5);

  const miedemaRaw = computeMiedemaFormationEnergy(formula);
  const { corrected: miedemaEnergy, correction: hydrideCorrection } = applyHydrideCorrection(formula, miedemaRaw);
  const toleranceFactor = computeGoldschmidtTolerance(formula);
  const pettiforDist = computePettiforProximity(formula);
  const elementCompat = computeElementCompatibility(formula);
  const protoMatch = computePrototypeMatch(formula);
  const valenceMismatch = computeValenceMismatch(formula);
  const sizeRatio = computeSizeRatioScore(formula);
  const { volumePerAtom, packingFraction } = computeVolumePerAtom(formula);

  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const enValues = elements.map(el => getElementData(el)?.paulingElectronegativity ?? 1.5);
  const enSpread = elements.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;

  const hFraction = (counts["H"] || 0) / totalAtoms;
  const isHydride = elements.includes("H") && hFraction > 0.3;
  const MIN_VOL_HYDRIDE = 5.0;
  const MIN_VOL_OTHER = 8.0;
  const minVol = isHydride ? MIN_VOL_HYDRIDE : MIN_VOL_OTHER;
  const volumeGhostFlag = volumePerAtom < minVol;

  const familyCtx = computeFamilyContext(formula);

  let synthesizability = rawSynthesizability * 0.25
    + elementCompat * 0.20
    + protoMatch * 0.15
    + sizeRatio * 0.15
    + (1 - valenceMismatch) * 0.10
    + (miedemaEnergy < 0 ? 0.8 : miedemaEnergy < 0.1 ? 0.5 : 0.2) * 0.10
    + (1 - rawDecompRisk) * 0.05;

  synthesizability += familyCtx.synthBias;

  if (toleranceFactor !== null) {
    const tDeviation = Math.abs(toleranceFactor - 1.0);
    if (tDeviation < 0.05) synthesizability += 0.05;
    else if (tDeviation > 0.2) synthesizability -= 0.05;
  }

  if (pettiforDist > 0.8) synthesizability *= 0.85;

  if (volumeGhostFlag) {
    const volumeRatio = volumePerAtom / minVol;
    synthesizability *= Math.max(0.1, volumeRatio);
  }

  synthesizability = Math.max(0, Math.min(1, synthesizability));

  let formationEnergy = miedemaEnergy * 0.7 + rawFormationEnergy * 0.3;
  if (!Number.isFinite(formationEnergy)) formationEnergy = miedemaEnergy;
  formationEnergy = Math.round(formationEnergy * 10000) / 10000;

  let decompositionRisk = rawDecompRisk * 0.3
    + (formationEnergy > 0 ? 0.6 : formationEnergy > -0.1 ? 0.4 : 0.2) * 0.3
    + valenceMismatch * 0.2
    + (1 - elementCompat) * 0.2;

  decompositionRisk += familyCtx.decompBias;

  const LOW_MP_THRESHOLD = 600;
  const VERY_LOW_MP_THRESHOLD = 350;
  let weightedMinMp = 0;
  let minConstituentMp = Infinity;
  let lowMpFraction = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const mp = data?.meltingPoint ?? 1000;
    const frac = (counts[el] || 1) / totalAtoms;
    weightedMinMp += mp * frac;
    if (mp < minConstituentMp) minConstituentMp = mp;
    if (mp < LOW_MP_THRESHOLD) lowMpFraction += frac;
  }

  if (minConstituentMp < VERY_LOW_MP_THRESHOLD) {
    const severity = (VERY_LOW_MP_THRESHOLD - minConstituentMp) / VERY_LOW_MP_THRESHOLD;
    decompositionRisk += severity * 0.25 * lowMpFraction;
  } else if (minConstituentMp < LOW_MP_THRESHOLD) {
    const severity = (LOW_MP_THRESHOLD - minConstituentMp) / LOW_MP_THRESHOLD;
    decompositionRisk += severity * 0.12 * lowMpFraction;
  }

  if (volumeGhostFlag) {
    decompositionRisk = Math.min(1.0, decompositionRisk + 0.2);
  }

  decompositionRisk = Math.max(0, Math.min(1, decompositionRisk));

  let stabilityClass: "stable" | "metastable" | "unstable";
  if (volumeGhostFlag) {
    stabilityClass = "unstable";
  } else if (synthesizability > 0.6 && decompositionRisk < 0.4 && formationEnergy < 0.1) {
    if (minConstituentMp < VERY_LOW_MP_THRESHOLD && lowMpFraction > 0.15) {
      stabilityClass = "metastable";
    } else {
      stabilityClass = "stable";
    }
  } else if (synthesizability > 0.35 && decompositionRisk < 0.65) {
    if (minConstituentMp < VERY_LOW_MP_THRESHOLD && lowMpFraction > 0.3) {
      stabilityClass = "unstable";
    } else {
      stabilityClass = "metastable";
    }
  } else {
    stabilityClass = "unstable";
  }

  let confidence = Math.min(0.95, Math.max(0.3,
    rawConfidence * 0.3 + elementCompat * 0.3 + protoMatch * 0.2 + (elements.length <= 4 ? 0.2 : 0.1)
  ));

  confidence += familyCtx.confBias;

  if (volumeGhostFlag) {
    const volumeRatio = volumePerAtom / minVol;
    confidence *= Math.max(0.2, volumeRatio);
  }

  if (packingFraction > protoMatch) {
    const overpackPenalty = (packingFraction - protoMatch) * 0.5;
    confidence *= Math.max(0.3, 1 - overpackPenalty);
  }

  confidence = Math.max(0.05, Math.min(0.95, confidence));

  return {
    formula,
    synthesizabilityScore: Math.round(synthesizability * 1000) / 1000,
    predictedFormationEnergy: formationEnergy,
    stabilityClass,
    decompositionRisk: Math.round(decompositionRisk * 1000) / 1000,
    confidence: Math.round(confidence * 1000) / 1000,
    details: {
      electronegativitySpread: Math.round(enSpread * 1000) / 1000,
      toleranceFactor: toleranceFactor !== null ? Math.round(toleranceFactor * 1000) / 1000 : null,
      pettiforDistance: Math.round(pettiforDist * 1000) / 1000,
      miedemaEnergy: Math.round(miedemaEnergy * 10000) / 10000,
      miedemaRaw: Math.round(miedemaRaw * 10000) / 10000,
      hydrideCorrection: Math.round(hydrideCorrection * 10000) / 10000,
      elementCompatibility: Math.round(elementCompat * 1000) / 1000,
      prototypeMatch: Math.round(protoMatch * 1000) / 1000,
      valenceMismatch: Math.round(valenceMismatch * 1000) / 1000,
      sizeRatioScore: Math.round(sizeRatio * 1000) / 1000,
      volumePerAtom: Math.round(volumePerAtom * 100) / 100,
      packingFraction: Math.round(packingFraction * 1000) / 1000,
      volumeGhostFlag,
    },
  };
}

export function passesStabilityPreFilter(formula: string): { pass: boolean; reason: string; prediction: StabilityPrediction } {
  const prediction = predictStability(formula);

  if (prediction.stabilityClass === "unstable" && prediction.synthesizabilityScore < 0.25) {
    return {
      pass: false,
      reason: `unstable (synth=${prediction.synthesizabilityScore.toFixed(3)}, decompRisk=${prediction.decompositionRisk.toFixed(3)}, formE=${prediction.predictedFormationEnergy.toFixed(4)} eV/atom)`,
      prediction,
    };
  }

  if (prediction.decompositionRisk > 0.85) {
    return {
      pass: false,
      reason: `high decomposition risk (${prediction.decompositionRisk.toFixed(3)})`,
      prediction,
    };
  }

  if (prediction.details.elementCompatibility < 0.15) {
    return {
      pass: false,
      reason: `incompatible elements (compat=${prediction.details.elementCompatibility.toFixed(3)})`,
      prediction,
    };
  }

  return {
    pass: true,
    reason: `${prediction.stabilityClass} (synth=${prediction.synthesizabilityScore.toFixed(3)})`,
    prediction,
  };
}
