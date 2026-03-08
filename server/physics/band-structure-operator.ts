import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";
import {
  predictBandStructure,
  type BandSurrogatePrediction,
} from "./band-structure-surrogate";
import {
  buildCrystalGraph,
  buildPrototypeGraph,
} from "../learning/graph-neural-net";

export interface KPoint {
  label: string;
  coordinates: [number, number, number];
  index: number;
}

export interface BandDispersionPoint {
  kIndex: number;
  kLabel: string;
  kFraction: number;
  energies: number[];
}

export interface BandDispersion {
  path: string;
  kPoints: KPoint[];
  bands: BandDispersionPoint[];
  nBands: number;
  nKPoints: number;
  fermiEnergy: number;
}

export interface DerivedQuantities {
  effectiveMasses: { bandIndex: number; value: number; direction: string }[];
  fermiVelocities: { bandIndex: number; velocity: number; kLabel: string }[];
  bandCurvatures: { bandIndex: number; curvature: number; kLabel: string }[];
  vhsPositions: { bandIndex: number; kFraction: number; energy: number; type: string }[];
  nestingVectors: { q: [number, number, number]; strength: number; bandPair: [number, number] }[];
  topologicalInvariants: {
    berryPhaseProxy: number;
    bandInversionCount: number;
    topologicalClass: string;
    z2Index: number;
  };
}

export interface PhysicsCalibration {
  referenceMaterial: string | null;
  calibrationConfidence: number;
  knownFeatures: string[];
}

export interface BandOperatorResult {
  formula: string;
  dispersion: BandDispersion;
  derivedQuantities: DerivedQuantities;
  calibration: PhysicsCalibration;
  surrogateInput: BandSurrogatePrediction;
  confidence: number;
}

const HIGH_SYMMETRY_PATHS: Record<string, { labels: string[]; coords: [number, number, number][] }> = {
  cubic: {
    labels: ["G", "X", "M", "G", "R", "X"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0],
    ],
  },
  hexagonal: {
    labels: ["G", "M", "K", "G", "A", "L"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0],
      [0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.5],
    ],
  },
  tetragonal: {
    labels: ["G", "X", "M", "G", "Z", "R"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.5],
    ],
  },
};

const KNOWN_BAND_STRUCTURES: Record<string, {
  sigmaCharacter: number;
  flatBandEnergy: number | null;
  fermiVelocity: number;
  nCrossingBands: number;
  vhsOffset: number;
  topologicalClass: string;
}> = {
  MgB2: {
    sigmaCharacter: 0.85,
    flatBandEnergy: null,
    fermiVelocity: 4.8,
    nCrossingBands: 4,
    vhsOffset: 0.3,
    topologicalClass: "trivial",
  },
  YBa2Cu3O7: {
    sigmaCharacter: 0.2,
    flatBandEnergy: -0.05,
    fermiVelocity: 2.5,
    nCrossingBands: 2,
    vhsOffset: 0.02,
    topologicalClass: "trivial",
  },
  LaH10: {
    sigmaCharacter: 0.65,
    flatBandEnergy: null,
    fermiVelocity: 6.2,
    nCrossingBands: 8,
    vhsOffset: 0.15,
    topologicalClass: "trivial",
  },
  FeSe: {
    sigmaCharacter: 0.3,
    flatBandEnergy: -0.02,
    fermiVelocity: 1.8,
    nCrossingBands: 5,
    vhsOffset: 0.05,
    topologicalClass: "trivial",
  },
  Nb3Sn: {
    sigmaCharacter: 0.5,
    flatBandEnergy: null,
    fermiVelocity: 3.2,
    nCrossingBands: 6,
    vhsOffset: 0.08,
    topologicalClass: "trivial",
  },
};

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
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

function guessLatticeSystem(elements: string[]): string {
  if (elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e))) return "hexagonal";
  if (elements.includes("Cu") && elements.includes("O") && elements.length >= 3) return "tetragonal";
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"))) return "tetragonal";
  return "cubic";
}

function interpolateKPath(
  startCoord: [number, number, number],
  endCoord: [number, number, number],
  nPoints: number,
): [number, number, number][] {
  const points: [number, number, number][] = [];
  for (let i = 0; i < nPoints; i++) {
    const t = i / Math.max(1, nPoints - 1);
    points.push([
      startCoord[0] + t * (endCoord[0] - startCoord[0]),
      startCoord[1] + t * (endCoord[1] - startCoord[1]),
      startCoord[2] + t * (endCoord[2] - startCoord[2]),
    ]);
  }
  return points;
}

interface OperatorWeights {
  W_kEmbed: number[][];
  W_graphEmbed: number[][];
  W_hidden1: number[][];
  b_hidden1: number[];
  W_hidden2: number[][];
  b_hidden2: number[];
  W_output: number[][];
  b_output: number[];
}

const OPERATOR_HIDDEN = 32;
const K_EMBED_DIM = 8;
const GRAPH_EMBED_DIM = 16;
const MAX_BANDS = 12;

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
    const row = mat[i];
    for (let j = 0; j < row.length && j < vec.length; j++) {
      sum += (row[j] ?? 0) * (vec[j] ?? 0);
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

let cachedOperatorWeights: OperatorWeights | null = null;

function getOperatorWeights(): OperatorWeights {
  if (cachedOperatorWeights) return cachedOperatorWeights;
  const rng = seededRandom(271828);
  cachedOperatorWeights = {
    W_kEmbed: initMatrix(K_EMBED_DIM, 3, rng, 0.15),
    W_graphEmbed: initMatrix(GRAPH_EMBED_DIM, 28, rng, 0.08),
    W_hidden1: initMatrix(OPERATOR_HIDDEN, K_EMBED_DIM + GRAPH_EMBED_DIM, rng, 0.1),
    b_hidden1: initVector(OPERATOR_HIDDEN),
    W_hidden2: initMatrix(OPERATOR_HIDDEN, OPERATOR_HIDDEN, rng, 0.1),
    b_hidden2: initVector(OPERATOR_HIDDEN),
    W_output: initMatrix(MAX_BANDS, OPERATOR_HIDDEN, rng, 0.12),
    b_output: initVector(MAX_BANDS),
  };
  return cachedOperatorWeights;
}

function computeGraphFeatureVector(formula: string, prototype?: string): number[] {
  const graph = prototype ? buildPrototypeGraph(formula, prototype) : buildCrystalGraph(formula);
  const vec: number[] = [];
  const n = graph.nodes.length;
  if (n === 0) return initVector(28);

  const meanEmb = initVector(20);
  for (const node of graph.nodes) {
    for (let i = 0; i < 20 && i < node.embedding.length; i++) {
      meanEmb[i] += node.embedding[i] / n;
    }
  }
  vec.push(...meanEmb);

  const edgeCount = graph.edges.length;
  vec.push(edgeCount / Math.max(1, n * (n - 1) / 2));
  let avgEdgeDist = 0;
  for (const edge of graph.edges) {
    avgEdgeDist += (edge.features[0] ?? 0) / Math.max(1, edgeCount);
  }
  vec.push(avgEdgeDist);

  const tbCount = graph.threeBodyFeatures.length;
  vec.push(tbCount / Math.max(1, n));
  let avgAngle = 0;
  for (const tb of graph.threeBodyFeatures) {
    avgAngle += tb.angle / Math.max(1, tbCount);
  }
  vec.push(avgAngle / Math.PI);

  vec.push(n / 20);
  vec.push(Math.min(1, edgeCount / 50));
  vec.push(Math.min(1, tbCount / 100));
  vec.push(0);

  while (vec.length < 28) vec.push(0);
  return vec.slice(0, 28);
}

function predictEnergyAtK(
  kCoord: [number, number, number],
  graphFeatures: number[],
  weights: OperatorWeights,
  nBands: number,
  calibrationShifts: number[],
): number[] {
  const kVec = [kCoord[0], kCoord[1], kCoord[2]];
  const kEmbed = relu(matVecMul(weights.W_kEmbed, kVec));
  const gEmbed = relu(matVecMul(weights.W_graphEmbed, graphFeatures));
  const combined = [...kEmbed, ...gEmbed];
  while (combined.length < K_EMBED_DIM + GRAPH_EMBED_DIM) combined.push(0);
  const h1 = relu(vecAdd(matVecMul(weights.W_hidden1, combined.slice(0, K_EMBED_DIM + GRAPH_EMBED_DIM)), weights.b_hidden1));
  const h2 = relu(vecAdd(matVecMul(weights.W_hidden2, h1), weights.b_hidden2));
  const raw = vecAdd(matVecMul(weights.W_output, h2), weights.b_output);
  const energies = raw.slice(0, nBands).map((e, i) => {
    const shift = calibrationShifts[i] ?? 0;
    return Number((e + shift).toFixed(4));
  });
  return energies.sort((a, b) => a - b);
}

function findClosestReference(formula: string): { ref: string; similarity: number } | null {
  const elements = parseFormulaElements(formula);
  let bestRef: string | null = null;
  let bestSim = 0;

  for (const [refFormula, _] of Object.entries(KNOWN_BAND_STRUCTURES)) {
    const refEls = parseFormulaElements(refFormula);
    const common = elements.filter(e => refEls.includes(e)).length;
    const sim = common / Math.max(elements.length, refEls.length);
    if (sim > bestSim) {
      bestSim = sim;
      bestRef = refFormula;
    }
  }

  return bestRef && bestSim > 0.2 ? { ref: bestRef, similarity: bestSim } : null;
}

function computeCalibrationShifts(
  formula: string,
  surrogateData: BandSurrogatePrediction,
  nBands: number,
): number[] {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const shifts: number[] = [];

  let avgEN = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.paulingElectronegativity) {
      avgEN += data.paulingElectronegativity * ((counts[el] || 1) / totalAtoms);
    }
  }

  const baseEnergy = -avgEN * 1.2;
  const bandwidth = Math.max(0.5, 8 - surrogateData.flatBandScore * 4);

  for (let b = 0; b < nBands; b++) {
    const bandCenter = baseEnergy + b * bandwidth / nBands;
    let shift = bandCenter;
    if (b < nBands / 2) {
      shift -= surrogateData.bandGap * 0.3;
    } else {
      shift += surrogateData.bandGap * 0.3;
    }
    if (surrogateData.flatBandScore > 0.5 && b === Math.floor(nBands / 2)) {
      shift *= 0.7;
    }
    shifts.push(shift);
  }

  return shifts;
}

function computeEffectiveMasses(dispersion: BandDispersion): DerivedQuantities["effectiveMasses"] {
  const masses: DerivedQuantities["effectiveMasses"] = [];
  const hbar2Over2m = 7.62;

  for (let b = 0; b < dispersion.nBands; b++) {
    const bandEnergies = dispersion.bands.map(pt => pt.energies[b] ?? 0);
    const fermiCrossings: number[] = [];
    for (let i = 1; i < bandEnergies.length; i++) {
      if ((bandEnergies[i] - dispersion.fermiEnergy) * (bandEnergies[i - 1] - dispersion.fermiEnergy) < 0) {
        fermiCrossings.push(i);
      }
    }
    if (fermiCrossings.length === 0) continue;

    for (const crossIdx of fermiCrossings) {
      const iMinus = Math.max(0, crossIdx - 1);
      const iPlus = Math.min(bandEnergies.length - 1, crossIdx + 1);
      const dE = bandEnergies[iPlus] - 2 * bandEnergies[crossIdx] + bandEnergies[iMinus];
      const dk = 1.0 / dispersion.nKPoints;
      const curvature = dE / (dk * dk);
      const mStar = curvature !== 0 ? hbar2Over2m / Math.abs(curvature) : 1.0;
      masses.push({
        bandIndex: b,
        value: Number(Math.min(100, Math.max(0.01, mStar)).toFixed(4)),
        direction: dispersion.bands[crossIdx]?.kLabel ?? "unknown",
      });
    }
  }

  return masses;
}

function computeFermiVelocities(dispersion: BandDispersion): DerivedQuantities["fermiVelocities"] {
  const velocities: DerivedQuantities["fermiVelocities"] = [];
  const eVtoMs = 1.0e5;

  for (let b = 0; b < dispersion.nBands; b++) {
    const bandEnergies = dispersion.bands.map(pt => pt.energies[b] ?? 0);
    for (let i = 1; i < bandEnergies.length; i++) {
      const ePrev = bandEnergies[i - 1];
      const eCurr = bandEnergies[i];
      if ((ePrev - dispersion.fermiEnergy) * (eCurr - dispersion.fermiEnergy) < 0) {
        const dk = 1.0 / dispersion.nKPoints;
        const dEdK = Math.abs(eCurr - ePrev) / dk;
        const vF = dEdK * eVtoMs;
        velocities.push({
          bandIndex: b,
          velocity: Number(vF.toFixed(2)),
          kLabel: dispersion.bands[i]?.kLabel ?? "unknown",
        });
      }
    }
  }

  return velocities;
}

function computeBandCurvatures(dispersion: BandDispersion): DerivedQuantities["bandCurvatures"] {
  const curvatures: DerivedQuantities["bandCurvatures"] = [];

  for (let b = 0; b < dispersion.nBands; b++) {
    const bandEnergies = dispersion.bands.map(pt => pt.energies[b] ?? 0);
    for (let i = 1; i < bandEnergies.length - 1; i++) {
      const d2E = bandEnergies[i + 1] - 2 * bandEnergies[i] + bandEnergies[i - 1];
      const dk = 1.0 / dispersion.nKPoints;
      const curv = d2E / (dk * dk);
      if (Math.abs(curv) > 0.01) {
        curvatures.push({
          bandIndex: b,
          curvature: Number(curv.toFixed(4)),
          kLabel: dispersion.bands[i]?.kLabel ?? "unknown",
        });
      }
    }
  }

  return curvatures;
}

function detectVHSPositions(dispersion: BandDispersion): DerivedQuantities["vhsPositions"] {
  const vhsPositions: DerivedQuantities["vhsPositions"] = [];

  for (let b = 0; b < dispersion.nBands; b++) {
    const bandEnergies = dispersion.bands.map(pt => pt.energies[b] ?? 0);
    const bandMin = Math.min(...bandEnergies);
    const bandMax = Math.max(...bandEnergies);
    const bandWidth = Math.max(0.01, bandMax - bandMin);
    const vhsThreshold = 0.1 * bandWidth;
    for (let i = 1; i < bandEnergies.length - 1; i++) {
      const d2E = bandEnergies[i + 1] - 2 * bandEnergies[i] + bandEnergies[i - 1];
      const isExtremum = (bandEnergies[i] >= bandEnergies[i - 1] && bandEnergies[i] >= bandEnergies[i + 1]) ||
                          (bandEnergies[i] <= bandEnergies[i - 1] && bandEnergies[i] <= bandEnergies[i + 1]);
      const nearFermi = Math.abs(bandEnergies[i] - dispersion.fermiEnergy) < 0.2;

      if (isExtremum && nearFermi && Math.abs(d2E) < vhsThreshold) {
        vhsPositions.push({
          bandIndex: b,
          kFraction: dispersion.bands[i].kFraction,
          energy: Number(bandEnergies[i].toFixed(4)),
          type: d2E > 0 ? "saddle-point-min" : d2E < 0 ? "saddle-point-max" : "flat",
        });
      }
    }
  }

  return vhsPositions;
}

function computeNestingVectors(dispersion: BandDispersion): DerivedQuantities["nestingVectors"] {
  const nesting: DerivedQuantities["nestingVectors"] = [];
  const fermiTol = 0.3;
  const fermiCrossings: { bandIndex: number; kIndex: number; kCoord: [number, number, number] }[] = [];

  for (let b = 0; b < dispersion.nBands; b++) {
    for (let i = 0; i < dispersion.bands.length; i++) {
      const e = dispersion.bands[i].energies[b] ?? 0;
      if (Math.abs(e - dispersion.fermiEnergy) < fermiTol) {
        const kp = dispersion.kPoints.find(k => k.index === i);
        if (kp) {
          fermiCrossings.push({ bandIndex: b, kIndex: i, kCoord: kp.coordinates });
        }
      }
    }
  }

  const qMap = new Map<string, { q: [number, number, number]; count: number; bands: [number, number] }>();

  for (let i = 0; i < fermiCrossings.length; i++) {
    for (let j = i + 1; j < fermiCrossings.length; j++) {
      if (fermiCrossings[i].bandIndex === fermiCrossings[j].bandIndex) continue;
      const q: [number, number, number] = [
        Number((fermiCrossings[j].kCoord[0] - fermiCrossings[i].kCoord[0]).toFixed(2)),
        Number((fermiCrossings[j].kCoord[1] - fermiCrossings[i].kCoord[1]).toFixed(2)),
        Number((fermiCrossings[j].kCoord[2] - fermiCrossings[i].kCoord[2]).toFixed(2)),
      ];
      const key = q.join(",");
      if (!qMap.has(key)) {
        qMap.set(key, { q, count: 0, bands: [fermiCrossings[i].bandIndex, fermiCrossings[j].bandIndex] });
      }
      qMap.get(key)!.count++;
    }
  }

  const sorted = Array.from(qMap.values()).sort((a, b) => b.count - a.count).slice(0, 5);
  const maxCount = sorted[0]?.count ?? 1;

  for (const entry of sorted) {
    nesting.push({
      q: entry.q,
      strength: Number((entry.count / maxCount).toFixed(4)),
      bandPair: entry.bands,
    });
  }

  return nesting;
}

function computeTopologicalInvariants(
  dispersion: BandDispersion,
  surrogateData: BandSurrogatePrediction,
): DerivedQuantities["topologicalInvariants"] {
  let bandInversionCount = 0;
  for (let b = 0; b < dispersion.nBands - 1; b++) {
    for (let i = 1; i < dispersion.bands.length; i++) {
      const e1 = dispersion.bands[i].energies[b] ?? 0;
      const e2 = dispersion.bands[i].energies[b + 1] ?? 0;
      const e1Prev = dispersion.bands[i - 1].energies[b] ?? 0;
      const e2Prev = dispersion.bands[i - 1].energies[b + 1] ?? 0;
      if ((e1 - e2) * (e1Prev - e2Prev) < 0) {
        bandInversionCount++;
      }
    }
  }

  let berryPhaseProxy = 0;
  if (bandInversionCount > 0) {
    berryPhaseProxy = Math.min(1.0, bandInversionCount * 0.15);
  }

  let topologicalClass = surrogateData.bandTopologyClass;
  if (bandInversionCount >= 3 && topologicalClass === "trivial") {
    topologicalClass = "topological-insulator";
  }

  const z2Index = bandInversionCount % 2;

  return {
    berryPhaseProxy: Number(berryPhaseProxy.toFixed(4)),
    bandInversionCount,
    topologicalClass,
    z2Index,
  };
}

function computeOverallConfidence(
  surrogateData: BandSurrogatePrediction,
  refMatch: { ref: string; similarity: number } | null,
  nBands: number,
): number {
  let conf = surrogateData.confidence * 0.6;
  if (refMatch) {
    conf += refMatch.similarity * 0.25;
  }
  if (nBands >= 4) conf += 0.05;
  if (nBands >= 8) conf += 0.05;
  return Number(Math.min(0.95, Math.max(0.1, conf)).toFixed(3));
}

export function predictBandDispersion(formula: string, prototype?: string): BandOperatorResult {
  const surrogateData = predictBandStructure(formula, prototype);
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const latticeSystem = guessLatticeSystem(elements);
  const pathDef = HIGH_SYMMETRY_PATHS[latticeSystem] ?? HIGH_SYMMETRY_PATHS.cubic;

  const nSegments = pathDef.labels.length - 1;
  const kPointsPerSegment = 10;
  const totalKPoints = nSegments * kPointsPerSegment;

  const tmCount = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e)).length;
  const nBands = Math.min(MAX_BANDS, Math.max(3, tmCount * 2 + 2));

  const kPoints: KPoint[] = [];
  const allKCoords: { coord: [number, number, number]; label: string; fraction: number }[] = [];
  let globalIdx = 0;

  for (let seg = 0; seg < nSegments; seg++) {
    const startCoord = pathDef.coords[seg] as [number, number, number];
    const endCoord = pathDef.coords[seg + 1] as [number, number, number];
    const segPoints = interpolateKPath(startCoord, endCoord, kPointsPerSegment);

    for (let i = 0; i < segPoints.length; i++) {
      const isEndpoint = i === 0;
      const label = isEndpoint ? pathDef.labels[seg] : "";
      if (isEndpoint) {
        kPoints.push({
          label: pathDef.labels[seg],
          coordinates: segPoints[i],
          index: globalIdx,
        });
      }
      allKCoords.push({
        coord: segPoints[i],
        label: label || `k${globalIdx}`,
        fraction: globalIdx / Math.max(1, totalKPoints - 1),
      });
      globalIdx++;
    }
  }
  kPoints.push({
    label: pathDef.labels[pathDef.labels.length - 1],
    coordinates: pathDef.coords[pathDef.coords.length - 1] as [number, number, number],
    index: globalIdx - 1,
  });

  const graphFeatures = computeGraphFeatureVector(formula, prototype);
  const weights = getOperatorWeights();
  const calibrationShifts = computeCalibrationShifts(formula, surrogateData, nBands);

  const bands: BandDispersionPoint[] = [];
  for (let i = 0; i < allKCoords.length; i++) {
    const kc = allKCoords[i];
    const energies = predictEnergyAtK(kc.coord, graphFeatures, weights, nBands, calibrationShifts);
    bands.push({
      kIndex: i,
      kLabel: kc.label,
      kFraction: Number(kc.fraction.toFixed(4)),
      energies,
    });
  }

  const allEnergies = bands.flatMap(b => b.energies).sort((a, b) => a - b);
  let totalElectrons = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.atomicNumber) {
      totalElectrons += data.atomicNumber * (counts[el] || 1);
    }
  }
  const fermiIndex = Math.min(
    Math.max(0, Math.floor(totalElectrons / 2) - 1),
    allEnergies.length - 1,
  );
  const fermiEnergy = allEnergies.length > 0
    ? Number(allEnergies[fermiIndex].toFixed(4))
    : 0;

  const dispersion: BandDispersion = {
    path: pathDef.labels.join("-"),
    kPoints,
    bands,
    nBands,
    nKPoints: allKCoords.length,
    fermiEnergy,
  };

  const effectiveMasses = computeEffectiveMasses(dispersion);
  const fermiVelocities = computeFermiVelocities(dispersion);
  const bandCurvatures = computeBandCurvatures(dispersion);
  const vhsPositions = detectVHSPositions(dispersion);
  const nestingVectors = computeNestingVectors(dispersion);
  const topologicalInvariants = computeTopologicalInvariants(dispersion, surrogateData);

  const derivedQuantities: DerivedQuantities = {
    effectiveMasses,
    fermiVelocities,
    bandCurvatures,
    vhsPositions,
    nestingVectors,
    topologicalInvariants,
  };

  const refMatch = findClosestReference(formula);
  const calibration: PhysicsCalibration = {
    referenceMaterial: refMatch?.ref ?? null,
    calibrationConfidence: refMatch?.similarity ?? 0,
    knownFeatures: refMatch && KNOWN_BAND_STRUCTURES[refMatch.ref]
      ? Object.entries(KNOWN_BAND_STRUCTURES[refMatch.ref])
          .filter(([_, v]) => v !== null)
          .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(2) : v}`)
      : [],
  };

  const confidence = computeOverallConfidence(surrogateData, refMatch, nBands);

  return {
    formula,
    dispersion,
    derivedQuantities,
    calibration,
    surrogateInput: surrogateData,
    confidence,
  };
}

export function getBandOperatorMLFeatures(result: BandOperatorResult): Record<string, number> {
  const avgMass = result.derivedQuantities.effectiveMasses.length > 0
    ? result.derivedQuantities.effectiveMasses.reduce((s, m) => s + m.value, 0) / result.derivedQuantities.effectiveMasses.length
    : 1.0;
  const avgVF = result.derivedQuantities.fermiVelocities.length > 0
    ? result.derivedQuantities.fermiVelocities.reduce((s, v) => s + v.velocity, 0) / result.derivedQuantities.fermiVelocities.length
    : 0;
  const vhsCount = result.derivedQuantities.vhsPositions.length;
  const nestingStrengthMax = result.derivedQuantities.nestingVectors.length > 0
    ? Math.max(...result.derivedQuantities.nestingVectors.map(n => n.strength))
    : 0;

  return {
    nBands: result.dispersion.nBands,
    fermiEnergy: result.dispersion.fermiEnergy,
    avgEffectiveMass: Number(avgMass.toFixed(4)),
    avgFermiVelocity: Number(avgVF.toFixed(2)),
    vhsCount,
    nestingStrengthMax: Number(nestingStrengthMax.toFixed(4)),
    berryPhaseProxy: result.derivedQuantities.topologicalInvariants.berryPhaseProxy,
    bandInversionCount: result.derivedQuantities.topologicalInvariants.bandInversionCount,
    confidence: result.confidence,
  };
}

export function getBandOperatorStats(): {
  referenceCount: number;
  referenceFormulas: string[];
  supportedLattices: string[];
  maxBands: number;
  kPointsPerSegment: number;
} {
  return {
    referenceCount: Object.keys(KNOWN_BAND_STRUCTURES).length,
    referenceFormulas: Object.keys(KNOWN_BAND_STRUCTURES),
    supportedLattices: Object.keys(HIGH_SYMMETRY_PATHS),
    maxBands: MAX_BANDS,
    kPointsPerSegment: 10,
  };
}
