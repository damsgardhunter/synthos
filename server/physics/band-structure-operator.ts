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

export type MassTensor3x3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

export interface EffectiveMassEntry {
  bandIndex: number;
  value: number;
  direction: string;
  massComponents?: [number, number, number];
  massTensor?: MassTensor3x3;
  harmonicMeanMass?: number;
  anisotropyRatio?: number;
  isFlatBand?: boolean;
}

export interface VHSPositionEntry {
  bandIndex: number;
  kFraction: number;
  energy: number;
  type: string;
  confirmed?: boolean;
  refinedEnergy?: number;
  refinedKFraction?: number;
  vhsClass?: "M0" | "M1" | "M2" | "flat";
  localCurvatures?: [number, number];
  vhs3DConfidence?: number;
}

export interface DerivedQuantities {
  effectiveMasses: EffectiveMassEntry[];
  fermiVelocities: { bandIndex: number; velocity: number; kLabel: string }[];
  bandCurvatures: { bandIndex: number; curvature: number; kLabel: string }[];
  vhsPositions: VHSPositionEntry[];
  nestingVectors: { q: [number, number, number]; strength: number; bandPair: [number, number]; parallelFraction: number }[];
  topologicalInvariants: {
    berryPhaseProxy: number;
    bandInversionCount: number;
    topologicalClass: string;
    z2Index: number;
    proxyConfidence: number;
    proxySource: "eigenvalue-inversion" | "wilson-loop" | "symmetry-indicator";
    wilsonLoopAvailable: boolean;
  };
}

export interface PhysicsCalibration {
  referenceMaterial: string | null;
  calibrationConfidence: number;
  knownFeatures: string[];
}

export interface OrbitalDOS {
  s: number;
  p: number;
  d: number;
  f: number;
  fermiWeighted?: {
    s: number;
    p: number;
    d: number;
    f: number;
    dominantOrbital: string;
    dominantFraction: number;
    hydrogenSCharacter: number;
  };
}

export interface BandOperatorResult {
  formula: string;
  dispersion: BandDispersion;
  derivedQuantities: DerivedQuantities;
  calibration: PhysicsCalibration;
  surrogateInput: BandSurrogatePrediction;
  orbitalDOS: OrbitalDOS;
  confidence: number;
}

const HIGH_SYMMETRY_PATHS: Record<string, { labels: string[]; coords: [number, number, number][] }> = {
  cubic_sc: {
    labels: ["G", "X", "M", "G", "R", "X"],
    coords: [
      [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
      [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0],
    ],
  },
  cubic_fcc: {
    labels: ["G", "X", "W", "K", "G", "L"],
    coords: [
      [0, 0, 0], [0.5, 0.5, 0], [0.5, 0.75, 0.25],
      [0.375, 0.75, 0.375], [0, 0, 0], [0.5, 0.5, 0.5],
    ],
  },
  cubic_bcc: {
    labels: ["G", "H", "N", "G", "P", "H"],
    coords: [
      [0, 0, 0], [0.5, -0.5, 0.5], [0, 0, 0.5],
      [0, 0, 0], [0.25, 0.25, 0.25], [0.5, -0.5, 0.5],
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

function guessLatticeSystem(elements: string[], counts: Record<string, number>): string {
  if (elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e))) return "hexagonal";
  if (elements.includes("Cu") && elements.includes("O") && elements.length >= 3) return "tetragonal";
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"))) return "tetragonal";

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const hCount = counts["H"] || 0;
  const hFrac = hCount / totalAtoms;

  if (hFrac > 0.6) return "cubic_fcc";

  const hasBCC = elements.some(e =>
    ["Nb", "V", "Ta", "Cr", "Mo", "W", "Fe", "Na", "K", "Rb", "Cs", "Ba"].includes(e)
  );
  if (hasBCC && elements.length <= 2 && !elements.includes("O")) return "cubic_bcc";

  const hasFCC = elements.some(e =>
    ["Al", "Cu", "Ag", "Au", "Pt", "Pd", "Ni", "Pb", "Ca", "Sr", "La", "Y", "Ce"].includes(e)
  );
  if (hasFCC && elements.length <= 3) return "cubic_fcc";

  if (totalAtoms <= 2) return "cubic_bcc";
  if (totalAtoms <= 4) return "cubic_fcc";
  return "cubic_sc";
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
    const row = mat[i];
    if (row.length !== vec.length) {
      throw new Error(
        `matVecMul dimension mismatch: row ${i} has ${row.length} cols but vec has ${vec.length} elements`
      );
    }
    let sum = 0;
    for (let j = 0; j < row.length; j++) {
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

function computeLocalBandwidth(
  bandEnergies: number[],
  centerIdx: number,
  halfWindow: number,
): number {
  const iLow = Math.max(0, centerIdx - halfWindow);
  const iHigh = Math.min(bandEnergies.length - 1, centerIdx + halfWindow);
  let eMin = Infinity;
  let eMax = -Infinity;
  for (let i = iLow; i <= iHigh; i++) {
    if (bandEnergies[i] < eMin) eMin = bandEnergies[i];
    if (bandEnergies[i] > eMax) eMax = bandEnergies[i];
  }
  return eMax - eMin;
}

function computeEffectiveMasses(
  dispersion: BandDispersion,
  latticeSystem: string,
): DerivedQuantities["effectiveMasses"] {
  const masses: DerivedQuantities["effectiveMasses"] = [];
  const hbar2Over2m = 7.62;
  const isCubic = latticeSystem.startsWith("cubic");

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

      const CURVATURE_FLAT_THRESHOLD = 1e-4;
      const absCurv = Math.abs(curvature);
      let mPath: number;
      let isFlatBand = false;

      if (absCurv < CURVATURE_FLAT_THRESHOLD) {
        isFlatBand = true;
        const localBW = computeLocalBandwidth(bandEnergies, crossIdx, 5);
        if (localBW < 0.005) {
          mPath = 500;
        } else if (localBW < 0.02) {
          mPath = 200;
        } else if (localBW < 0.05) {
          mPath = 100;
        } else {
          mPath = 50;
        }
      } else {
        mPath = hbar2Over2m / absCurv;
      }

      const kCoord = dispersion.bands[crossIdx]?.kIndex !== undefined
        ? dispersion.kPoints.find(kp => kp.index === crossIdx)?.coordinates
        : undefined;

      let massComponents: [number, number, number];
      let massTensor: MassTensor3x3;
      let harmonicMeanMass: number;
      let anisotropyRatio: number;

      const MASS_MIN = 0.01;
      const MASS_MAX = 500;

      if (isCubic) {
        const clampedM = Math.min(MASS_MAX, Math.max(MASS_MIN, mPath));
        massComponents = [clampedM, clampedM, clampedM];
        massTensor = [
          [clampedM, 0, 0],
          [0, clampedM, 0],
          [0, 0, clampedM],
        ];
        harmonicMeanMass = clampedM;
        anisotropyRatio = 1.0;
      } else {
        const dirWeight = estimateDirectionalWeights(kCoord, latticeSystem);
        const mx = Math.min(MASS_MAX, Math.max(MASS_MIN, mPath * dirWeight[0]));
        const my = Math.min(MASS_MAX, Math.max(MASS_MIN, mPath * dirWeight[1]));
        const mz = Math.min(MASS_MAX, Math.max(MASS_MIN, mPath * dirWeight[2]));

        massComponents = [
          Number(mx.toFixed(4)),
          Number(my.toFixed(4)),
          Number(mz.toFixed(4)),
        ];

        massTensor = [
          [mx, 0, 0],
          [0, my, 0],
          [0, 0, mz],
        ];

        const invSum = 1 / mx + 1 / my + 1 / mz;
        harmonicMeanMass = invSum > 0 ? 3.0 / invSum : mPath;
        const maxM = Math.max(mx, my, mz);
        const minM = Math.min(mx, my, mz);
        anisotropyRatio = minM > 0 ? maxM / minM : 1.0;
      }

      masses.push({
        bandIndex: b,
        value: Number(Math.min(MASS_MAX, Math.max(MASS_MIN, harmonicMeanMass)).toFixed(4)),
        direction: dispersion.bands[crossIdx]?.kLabel ?? "unknown",
        massComponents,
        massTensor,
        harmonicMeanMass: Number(harmonicMeanMass.toFixed(4)),
        anisotropyRatio: Number(anisotropyRatio.toFixed(4)),
        isFlatBand,
      });
    }
  }

  return masses;
}

function estimateDirectionalWeights(
  kCoord: [number, number, number] | undefined,
  latticeSystem: string,
): [number, number, number] {
  if (latticeSystem === "tetragonal") {
    return [1.0, 1.0, 3.5];
  }
  if (latticeSystem === "hexagonal") {
    return [1.0, 1.0, 5.0];
  }

  if (kCoord) {
    const [kx, ky, kz] = kCoord;
    const norm = Math.sqrt(kx * kx + ky * ky + kz * kz) || 1;
    const zFrac = Math.abs(kz) / norm;
    const xyFrac = 1 - zFrac;
    return [
      Math.max(0.5, 1 + xyFrac * 0.5),
      Math.max(0.5, 1 + xyFrac * 0.5),
      Math.max(0.5, 1 + zFrac * 2.0),
    ];
  }

  return [1.0, 1.0, 2.0];
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

function refineVHSByQuadraticInterpolation(
  bandEnergies: number[],
  candidateIdx: number,
  dispersionBands: BandDispersionPoint[],
  fermiEnergy: number,
): {
  refinedEnergy: number;
  refinedKFraction: number;
  confirmed: boolean;
  vhsClass: "M0" | "M1" | "M2" | "flat";
  localCurvatures: [number, number];
  vhs3DConfidence: number;
} {
  const N_REFINE = 4;
  const iLow = Math.max(0, candidateIdx - N_REFINE);
  const iHigh = Math.min(bandEnergies.length - 1, candidateIdx + N_REFINE);
  const localE: number[] = [];
  const localK: number[] = [];

  for (let j = iLow; j <= iHigh; j++) {
    localE.push(bandEnergies[j]);
    localK.push(dispersionBands[j]?.kFraction ?? j / bandEnergies.length);
  }

  const centerLocal = candidateIdx - iLow;
  const n = localE.length;

  let sumK = 0, sumK2 = 0, sumK3 = 0, sumK4 = 0;
  let sumE = 0, sumKE = 0, sumK2E = 0;
  for (let j = 0; j < n; j++) {
    const k = localK[j];
    const e = localE[j];
    sumK += k; sumK2 += k * k; sumK3 += k * k * k; sumK4 += k * k * k * k;
    sumE += e; sumKE += k * e; sumK2E += k * k * e;
  }

  const det = n * (sumK2 * sumK4 - sumK3 * sumK3) -
              sumK * (sumK * sumK4 - sumK3 * sumK2) +
              sumK2 * (sumK * sumK3 - sumK2 * sumK2);

  let refinedK = localK[centerLocal] ?? dispersionBands[candidateIdx]?.kFraction ?? 0;
  let refinedE = localE[centerLocal] ?? bandEnergies[candidateIdx];
  let curvA = 0;

  if (Math.abs(det) > 1e-20) {
    const a0 = (sumE * (sumK2 * sumK4 - sumK3 * sumK3) -
                sumKE * (sumK * sumK4 - sumK3 * sumK2) +
                sumK2E * (sumK * sumK3 - sumK2 * sumK2)) / det;
    const a1 = (n * (sumKE * sumK4 - sumK2E * sumK3) -
                sumK * (sumE * sumK4 - sumK2E * sumK2) +
                sumK2 * (sumE * sumK3 - sumKE * sumK2)) / det;
    const a2 = (n * (sumK2 * sumK2E - sumK3 * sumKE) -
                sumK * (sumK * sumK2E - sumK3 * sumE) +
                sumK2 * (sumK * sumKE - sumK2 * sumE)) / det;

    curvA = a2;
    if (Math.abs(a2) > 1e-12) {
      const kVertex = -a1 / (2 * a2);
      if (kVertex >= localK[0] && kVertex <= localK[n - 1]) {
        refinedK = kVertex;
        refinedE = a0 + a1 * kVertex + a2 * kVertex * kVertex;
      }
    }
  }

  const leftCurv = centerLocal >= 2
    ? localE[centerLocal - 2] - 2 * localE[centerLocal - 1] + localE[centerLocal]
    : 0;
  const rightCurv = centerLocal + 2 < n
    ? localE[centerLocal + 2] - 2 * localE[centerLocal + 1] + localE[centerLocal]
    : 0;

  let vhsClass: "M0" | "M1" | "M2" | "flat";
  const absCurv = Math.abs(curvA);
  const localCurvatures: [number, number] = [
    Number(leftCurv.toFixed(6)),
    Number(rightCurv.toFixed(6)),
  ];

  if (absCurv < 1e-6 && Math.abs(leftCurv) < 1e-4 && Math.abs(rightCurv) < 1e-4) {
    vhsClass = "flat";
  } else if (leftCurv > 0 && rightCurv > 0) {
    vhsClass = "M0";
  } else if (leftCurv < 0 && rightCurv < 0) {
    vhsClass = "M2";
  } else {
    vhsClass = "M1";
  }

  const nearFermi = Math.abs(refinedE - fermiEnergy) < 0.25;

  let vhs3DConfidence = 0;
  if (nearFermi) {
    if (vhsClass === "M1") {
      vhs3DConfidence = 0.7;
      if (leftCurv * rightCurv < 0) {
        vhs3DConfidence += 0.15;
      }
    } else if (vhsClass === "flat") {
      vhs3DConfidence = 0.5;
    } else if (vhsClass === "M0" || vhsClass === "M2") {
      vhs3DConfidence = 0.25;
    }

    const curvRatio = Math.abs(leftCurv) > 1e-8 && Math.abs(rightCurv) > 1e-8
      ? Math.min(Math.abs(leftCurv), Math.abs(rightCurv)) / Math.max(Math.abs(leftCurv), Math.abs(rightCurv))
      : 0;
    if (curvRatio > 0.3 && curvRatio < 0.95) {
      vhs3DConfidence += 0.10;
    }

    const eFermiDist = Math.abs(refinedE - fermiEnergy);
    if (eFermiDist < 0.05) vhs3DConfidence += 0.05;
  }
  vhs3DConfidence = Math.min(0.95, vhs3DConfidence);

  const confirmed = nearFermi && vhs3DConfidence >= 0.5;

  return {
    refinedEnergy: Number(refinedE.toFixed(6)),
    refinedKFraction: Number(refinedK.toFixed(6)),
    confirmed,
    vhsClass,
    localCurvatures,
    vhs3DConfidence: Number(vhs3DConfidence.toFixed(4)),
  };
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
        const rawType = d2E > 0 ? "saddle-point-min" : d2E < 0 ? "saddle-point-max" : "flat";

        const refined = refineVHSByQuadraticInterpolation(
          bandEnergies, i, dispersion.bands, dispersion.fermiEnergy
        );

        vhsPositions.push({
          bandIndex: b,
          kFraction: dispersion.bands[i].kFraction,
          energy: Number(bandEnergies[i].toFixed(4)),
          type: rawType,
          confirmed: refined.confirmed,
          refinedEnergy: refined.refinedEnergy,
          refinedKFraction: refined.refinedKFraction,
          vhsClass: refined.vhsClass,
          localCurvatures: refined.localCurvatures,
          vhs3DConfidence: refined.vhs3DConfidence,
        });
      }
    }
  }

  return vhsPositions;
}

function nestingBinKey(q: [number, number, number], binWidth: number): string {
  return q.map(v => Math.round(v / binWidth) * binWidth).join(",");
}

function nestingBinCenter(q: [number, number, number], binWidth: number): [number, number, number] {
  return q.map(v => Number((Math.round(v / binWidth) * binWidth).toFixed(4))) as [number, number, number];
}

const MAX_FERMI_CROSSINGS = 200;
const NESTING_BIN_WIDTH = 0.04;

function computeNestingVectors(dispersion: BandDispersion): DerivedQuantities["nestingVectors"] {
  const nesting: DerivedQuantities["nestingVectors"] = [];
  const fermiTol = 0.3;
  const MIN_PARALLEL_FRACTION = 0.15;
  const fermiCrossings: { bandIndex: number; kIndex: number; kCoord: [number, number, number]; velocity: number }[] = [];

  for (let b = 0; b < dispersion.nBands; b++) {
    for (let i = 0; i < dispersion.bands.length; i++) {
      const e = dispersion.bands[i].energies[b] ?? 0;
      if (Math.abs(e - dispersion.fermiEnergy) < fermiTol) {
        let vel = 0;
        if (i > 0) {
          const ePrev = dispersion.bands[i - 1]?.energies[b] ?? e;
          const dk = 1.0 / Math.max(1, dispersion.nKPoints);
          vel = Math.abs(e - ePrev) / dk;
        }
        const kIdx = i;
        const kp = dispersion.kPoints.find(k => k.index === kIdx);
        const coord: [number, number, number] = kp
          ? kp.coordinates
          : [i / Math.max(1, dispersion.nKPoints), 0, 0];
        fermiCrossings.push({ bandIndex: b, kIndex: kIdx, kCoord: coord, velocity: vel });
      }
    }
  }

  if (fermiCrossings.length > MAX_FERMI_CROSSINGS) {
    const stride = Math.ceil(fermiCrossings.length / MAX_FERMI_CROSSINGS);
    const downsampled: typeof fermiCrossings = [];
    for (let i = 0; i < fermiCrossings.length; i += stride) {
      downsampled.push(fermiCrossings[i]);
    }
    fermiCrossings.length = 0;
    fermiCrossings.push(...downsampled);
  }

  const totalFermiPoints = fermiCrossings.length;
  if (totalFermiPoints < 2) return nesting;

  const qMap = new Map<string, {
    qCenter: [number, number, number];
    count: number;
    bands: [number, number];
    parallelPairs: number;
    qSum: [number, number, number];
  }>();

  for (let i = 0; i < fermiCrossings.length; i++) {
    for (let j = i + 1; j < fermiCrossings.length; j++) {
      if (fermiCrossings[i].bandIndex === fermiCrossings[j].bandIndex) continue;
      const rawQ: [number, number, number] = [
        fermiCrossings[j].kCoord[0] - fermiCrossings[i].kCoord[0],
        fermiCrossings[j].kCoord[1] - fermiCrossings[i].kCoord[1],
        fermiCrossings[j].kCoord[2] - fermiCrossings[i].kCoord[2],
      ];
      const key = nestingBinKey(rawQ, NESTING_BIN_WIDTH);
      if (!qMap.has(key)) {
        qMap.set(key, {
          qCenter: nestingBinCenter(rawQ, NESTING_BIN_WIDTH),
          count: 0,
          bands: [fermiCrossings[i].bandIndex, fermiCrossings[j].bandIndex],
          parallelPairs: 0,
          qSum: [0, 0, 0],
        });
      }
      const entry = qMap.get(key)!;
      entry.count++;
      entry.qSum[0] += rawQ[0];
      entry.qSum[1] += rawQ[1];
      entry.qSum[2] += rawQ[2];

      const vi = fermiCrossings[i].velocity;
      const vj = fermiCrossings[j].velocity;
      if (vi > 0 && vj > 0) {
        const velRatio = Math.min(vi, vj) / Math.max(vi, vj);
        if (velRatio > 0.5) {
          entry.parallelPairs++;
        }
      }
    }
  }

  const sorted = Array.from(qMap.values()).sort((a, b) => b.count - a.count).slice(0, 10);
  const maxCount = sorted[0]?.count ?? 1;

  for (const entry of sorted) {
    const parallelFraction = totalFermiPoints > 0
      ? entry.parallelPairs / (totalFermiPoints / 2)
      : 0;

    if (parallelFraction < MIN_PARALLEL_FRACTION && entry.count < maxCount * 0.5) {
      continue;
    }

    const avgQ: [number, number, number] = entry.count > 0
      ? [
          Number((entry.qSum[0] / entry.count).toFixed(4)),
          Number((entry.qSum[1] / entry.count).toFixed(4)),
          Number((entry.qSum[2] / entry.count).toFixed(4)),
        ]
      : entry.qCenter;

    nesting.push({
      q: avgQ,
      strength: Number((entry.count / maxCount).toFixed(4)),
      bandPair: entry.bands,
      parallelFraction: Number(parallelFraction.toFixed(4)),
    });

    if (nesting.length >= 5) break;
  }

  return nesting;
}

function computeTopologicalInvariants(
  dispersion: BandDispersion,
  surrogateData: BandSurrogatePrediction,
  formula: string,
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

  let proxyConfidence = 0;
  let proxySource: "eigenvalue-inversion" | "wilson-loop" | "symmetry-indicator" = "eigenvalue-inversion";
  let wilsonLoopAvailable = false;

  try {
    const { computeTopologicalInvariants: computeFullTopo } = require("./topological-invariants");
    const { computeElectronicStructure } = require("../learning/physics-engine");
    const electronic = computeElectronicStructure(formula, null);
    const fullTopo = computeFullTopo(formula, electronic);

    if (fullTopo && fullTopo.z2Invariant) {
      wilsonLoopAvailable = fullTopo.z2Invariant.wilsonLoopWindings > 0;

      if (wilsonLoopAvailable) {
        proxySource = "wilson-loop";
        berryPhaseProxy = fullTopo.chernNumber?.berryPhase ?? berryPhaseProxy;
        proxyConfidence = Math.min(1.0,
          fullTopo.z2Invariant.confidence * 0.5 +
          fullTopo.chernNumber.confidence * 0.3 +
          (fullTopo.bandInversion.isInverted ? 0.2 : 0)
        );
        topologicalClass = fullTopo.topologicalPhase !== "trivial"
          ? fullTopo.topologicalPhase
          : topologicalClass;
      } else if (fullTopo.symmetryIndicator?.compatibilityCheckPassed) {
        proxySource = "symmetry-indicator";
        proxyConfidence = Math.min(0.6,
          fullTopo.symmetryIndicator.confidence * 0.4 +
          (fullTopo.bandInversion.isInverted ? 0.2 : 0)
        );
      } else {
        proxyConfidence = estimateEigenvalueProxyConfidence(bandInversionCount, dispersion.nBands);
      }
    } else {
      proxyConfidence = estimateEigenvalueProxyConfidence(bandInversionCount, dispersion.nBands);
    }
  } catch {
    proxyConfidence = estimateEigenvalueProxyConfidence(bandInversionCount, dispersion.nBands);
  }

  return {
    berryPhaseProxy: Number(berryPhaseProxy.toFixed(4)),
    bandInversionCount,
    topologicalClass,
    z2Index,
    proxyConfidence: Number(proxyConfidence.toFixed(4)),
    proxySource,
    wilsonLoopAvailable,
  };
}

function estimateEigenvalueProxyConfidence(inversions: number, nBands: number): number {
  if (inversions === 0) return 0.1;

  const inversionRate = inversions / Math.max(1, nBands);
  if (inversionRate > 0.5) return 0.15;
  if (inversions === 1) return 0.25;
  if (inversions === 2) return 0.30;
  return Math.min(0.35, 0.20 + inversions * 0.03);
}

function getElementOrbitalProfile(el: string): { s: number; p: number; d: number; f: number } {
  const data = getElementData(el);
  const Z = data?.atomicNumber ?? 1;
  const valence = data?.valenceElectrons ?? 1;

  if (Z === 1) return { s: 0.95, p: 0.05, d: 0, f: 0 };
  if (Z === 2) return { s: 0.90, p: 0.10, d: 0, f: 0 };

  if (Z >= 57 && Z <= 71) {
    const fOcc = Math.min(1.0, (Z - 57) / 14);
    return { s: 0.05, p: 0.05, d: 0.15 + 0.1 * (1 - fOcc), f: 0.65 + 0.1 * fOcc };
  }
  if (Z >= 89 && Z <= 103) {
    const fOcc = Math.min(1.0, (Z - 89) / 14);
    return { s: 0.05, p: 0.05, d: 0.15 + 0.1 * (1 - fOcc), f: 0.65 + 0.1 * fOcc };
  }

  if (isTransitionMetal(el)) {
    let dFill: number;
    if (Z >= 21 && Z <= 30) {
      dFill = (Z - 20) / 10;
    } else if (Z >= 39 && Z <= 48) {
      dFill = (Z - 38) / 10;
    } else if (Z >= 72 && Z <= 80) {
      dFill = (Z - 71) / 10;
    } else {
      dFill = 0.5;
    }

    const earlyTM = dFill < 0.4;
    const lateTM = dFill > 0.7;

    if (earlyTM) {
      return { s: 0.15, p: 0.10, d: 0.70, f: 0.05 };
    } else if (lateTM) {
      const cu_like = dFill > 0.9;
      return {
        s: cu_like ? 0.20 : 0.12,
        p: cu_like ? 0.08 : 0.10,
        d: cu_like ? 0.68 : 0.73,
        f: 0.05,
      };
    } else {
      return { s: 0.08, p: 0.07, d: 0.80, f: 0.05 };
    }
  }

  if (Z >= 13 && Z <= 18) {
    return { s: 0.20, p: 0.70, d: 0.10, f: 0 };
  }
  if (Z >= 31 && Z <= 36) {
    return { s: 0.15, p: 0.65, d: 0.20, f: 0 };
  }
  if (Z >= 49 && Z <= 54) {
    return { s: 0.15, p: 0.60, d: 0.25, f: 0 };
  }
  if (Z >= 81 && Z <= 86) {
    return { s: 0.15, p: 0.55, d: 0.25, f: 0.05 };
  }

  if (valence >= 3) {
    return { s: 0.25, p: 0.65, d: 0.10, f: 0 };
  }

  if (valence <= 1) {
    return { s: 0.75, p: 0.15, d: 0.10, f: 0 };
  }

  return { s: 0.45, p: 0.35, d: 0.15, f: 0.05 };
}

function computeOrbitalDOS(
  dispersion: BandDispersion,
  formula: string,
): OrbitalDOS {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  let sWeight = 0;
  let pWeight = 0;
  let dWeight = 0;
  let fWeight = 0;

  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    const profile = getElementOrbitalProfile(el);
    sWeight += profile.s * frac;
    pWeight += profile.p * frac;
    dWeight += profile.d * frac;
    fWeight += profile.f * frac;
  }

  let totalDOSAtFermi = 0;
  const fermiTol = 0.15;
  for (let b = 0; b < dispersion.nBands; b++) {
    for (const pt of dispersion.bands) {
      const e = pt.energies[b] ?? 0;
      if (Math.abs(e - dispersion.fermiEnergy) < fermiTol) {
        totalDOSAtFermi += 1.0 / (Math.abs(e - dispersion.fermiEnergy) + 0.01);
      }
    }
  }

  const norm = sWeight + pWeight + dWeight + fWeight || 1;
  const scale = totalDOSAtFermi / (dispersion.nKPoints * dispersion.nBands || 1);

  let fermiS = 0, fermiP = 0, fermiD = 0, fermiF = 0;
  let fermiTotalWeight = 0;
  let hydrogenSChar = 0;

  const fermiWindow = 0.10;
  for (let b = 0; b < dispersion.nBands; b++) {
    for (const pt of dispersion.bands) {
      const e = pt.energies[b] ?? 0;
      const dist = Math.abs(e - dispersion.fermiEnergy);
      if (dist > fermiWindow) continue;

      const fermiWeight = 1.0 / (dist + 0.005);

      let bandS = 0, bandP = 0, bandD = 0, bandF = 0;
      for (const el of elements) {
        const frac = (counts[el] || 1) / totalAtoms;
        const profile = getElementOrbitalProfile(el);
        bandS += profile.s * frac;
        bandP += profile.p * frac;
        bandD += profile.d * frac;
        bandF += profile.f * frac;
      }

      fermiS += bandS * fermiWeight;
      fermiP += bandP * fermiWeight;
      fermiD += bandD * fermiWeight;
      fermiF += bandF * fermiWeight;
      fermiTotalWeight += fermiWeight;

      const hFrac = (counts["H"] || 0) / totalAtoms;
      hydrogenSChar += hFrac * 0.95 * fermiWeight;
    }
  }

  const fwNorm = fermiTotalWeight || 1;
  const fwS = fermiS / fwNorm;
  const fwP = fermiP / fwNorm;
  const fwD = fermiD / fwNorm;
  const fwF = fermiF / fwNorm;
  const hSChar = hydrogenSChar / fwNorm;

  const orbFracs = { s: fwS, p: fwP, d: fwD, f: fwF };
  const dominantOrbital = (Object.entries(orbFracs) as [string, number][])
    .sort((a, b) => b[1] - a[1])[0];

  return {
    s: Number((sWeight / norm * scale).toFixed(6)),
    p: Number((pWeight / norm * scale).toFixed(6)),
    d: Number((dWeight / norm * scale).toFixed(6)),
    f: Number((fWeight / norm * scale).toFixed(6)),
    fermiWeighted: {
      s: Number(fwS.toFixed(6)),
      p: Number(fwP.toFixed(6)),
      d: Number(fwD.toFixed(6)),
      f: Number(fwF.toFixed(6)),
      dominantOrbital: dominantOrbital[0],
      dominantFraction: Number(dominantOrbital[1].toFixed(6)),
      hydrogenSCharacter: Number(hSChar.toFixed(6)),
    },
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

function computeFermiEnergyByDOS(
  bands: BandDispersionPoint[],
  targetOccupied: number,
  nBands: number,
): number {
  if (bands.length === 0) return 0;

  const allEnergies = bands.flatMap(b => b.energies);
  const eMin = Math.min(...allEnergies);
  const eMax = Math.max(...allEnergies);
  if (eMin === eMax) return eMin;

  const BROADENING = 0.05;
  const nKPoints = bands.length;

  function integratedDOS(eFermi: number): number {
    let occupied = 0;
    for (let b = 0; b < nBands; b++) {
      for (let k = 0; k < nKPoints; k++) {
        const e = bands[k].energies[b] ?? 0;
        const x = (eFermi - e) / BROADENING;
        const fermiOcc = x > 20 ? 1.0 : x < -20 ? 0.0 : 1.0 / (1.0 + Math.exp(-x));
        occupied += fermiOcc / nKPoints;
      }
    }
    return occupied;
  }

  let eLo = eMin - 1.0;
  let eHi = eMax + 1.0;
  for (let iter = 0; iter < 60; iter++) {
    const eMid = (eLo + eHi) / 2;
    const n = integratedDOS(eMid);
    if (n < targetOccupied) {
      eLo = eMid;
    } else {
      eHi = eMid;
    }
    if (eHi - eLo < 1e-6) break;
  }

  return Number(((eLo + eHi) / 2).toFixed(4));
}

export function predictBandDispersion(formula: string, prototype?: string): BandOperatorResult {
  const surrogateData = predictBandStructure(formula, prototype);
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const latticeSystem = guessLatticeSystem(elements, counts);
  const pathDef = HIGH_SYMMETRY_PATHS[latticeSystem] ?? HIGH_SYMMETRY_PATHS.cubic_sc;

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
      if (seg > 0 && i === 0) {
        continue;
      }

      const isSegStart = i === 0;
      const isSegEnd = i === segPoints.length - 1;
      const label = isSegStart
        ? pathDef.labels[seg]
        : isSegEnd
          ? pathDef.labels[seg + 1]
          : "";

      if (isSegStart || isSegEnd) {
        const hsLabel = isSegStart ? pathDef.labels[seg] : pathDef.labels[seg + 1];
        const alreadyAdded = kPoints.some(kp => kp.label === hsLabel);
        if (!alreadyAdded) {
          kPoints.push({
            label: hsLabel,
            coordinates: segPoints[i],
            index: globalIdx,
          });
        }
      }

      allKCoords.push({
        coord: segPoints[i],
        label: label || `k${globalIdx}`,
        fraction: globalIdx / Math.max(1, totalKPoints - 1),
      });
      globalIdx++;
    }
  }

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

  let totalElectrons = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.atomicNumber) {
      totalElectrons += data.atomicNumber * (counts[el] || 1);
    }
  }
  const targetOccupied = totalElectrons / 2;

  const fermiEnergy = computeFermiEnergyByDOS(bands, targetOccupied, nBands);

  const dispersion: BandDispersion = {
    path: pathDef.labels.join("-"),
    kPoints,
    bands,
    nBands,
    nKPoints: allKCoords.length,
    fermiEnergy,
  };

  const effectiveMasses = computeEffectiveMasses(dispersion, latticeSystem);
  const fermiVelocities = computeFermiVelocities(dispersion);
  const bandCurvatures = computeBandCurvatures(dispersion);
  const vhsPositions = detectVHSPositions(dispersion);
  const nestingVectors = computeNestingVectors(dispersion);
  const topologicalInvariants = computeTopologicalInvariants(dispersion, surrogateData, formula);

  const derivedQuantities: DerivedQuantities = {
    effectiveMasses,
    fermiVelocities,
    bandCurvatures,
    vhsPositions,
    nestingVectors,
    topologicalInvariants,
  };

  const orbitalDOS = computeOrbitalDOS(dispersion, formula);

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
    orbitalDOS,
    confidence,
  };
}

export function getBandOperatorMLFeatures(result: BandOperatorResult): Record<string, number> {
  const masses = result.derivedQuantities.effectiveMasses;
  let avgMass = 1.0;
  let avgAnisotropy = 1.0;
  let flatBandCount = 0;
  if (masses.length > 0) {
    const invSum = masses.reduce((s, m) => s + 1 / Math.max(0.01, m.value), 0);
    avgMass = invSum > 0 ? masses.length / invSum : 1.0;
    avgAnisotropy = masses.reduce((s, m) => s + (m.anisotropyRatio ?? 1.0), 0) / masses.length;
    flatBandCount = masses.filter(m => m.isFlatBand).length;
  }

  const avgVF = result.derivedQuantities.fermiVelocities.length > 0
    ? result.derivedQuantities.fermiVelocities.reduce((s, v) => s + v.velocity, 0) / result.derivedQuantities.fermiVelocities.length
    : 0;
  const vhsAll = result.derivedQuantities.vhsPositions;
  const vhsCount = vhsAll.length;
  const vhsConfirmedCount = vhsAll.filter(v => v.confirmed).length;
  const nestingStrengthMax = result.derivedQuantities.nestingVectors.length > 0
    ? Math.max(...result.derivedQuantities.nestingVectors.map(n => n.strength))
    : 0;

  return {
    nBands: result.dispersion.nBands,
    fermiEnergy: result.dispersion.fermiEnergy,
    avgEffectiveMass: Number(avgMass.toFixed(4)),
    avgMassAnisotropy: Number(avgAnisotropy.toFixed(4)),
    flatBandCrossingCount: flatBandCount,
    avgFermiVelocity: Number(avgVF.toFixed(2)),
    vhsCount,
    vhsConfirmedCount,
    vhsAvg3DConfidence: vhsAll.length > 0
      ? Number((vhsAll.reduce((s, v) => s + (v.vhs3DConfidence ?? 0), 0) / vhsAll.length).toFixed(4))
      : 0,
    nestingStrengthMax: Number(nestingStrengthMax.toFixed(4)),
    berryPhaseProxy: result.derivedQuantities.topologicalInvariants.berryPhaseProxy,
    bandInversionCount: result.derivedQuantities.topologicalInvariants.bandInversionCount,
    orbitalDOS_s: result.orbitalDOS.s,
    orbitalDOS_p: result.orbitalDOS.p,
    orbitalDOS_d: result.orbitalDOS.d,
    orbitalDOS_f: result.orbitalDOS.f,
    fermiOrbital_s: result.orbitalDOS.fermiWeighted?.s ?? result.orbitalDOS.s,
    fermiOrbital_p: result.orbitalDOS.fermiWeighted?.p ?? result.orbitalDOS.p,
    fermiOrbital_d: result.orbitalDOS.fermiWeighted?.d ?? result.orbitalDOS.d,
    fermiOrbital_f: result.orbitalDOS.fermiWeighted?.f ?? result.orbitalDOS.f,
    hydrogenSCharacterAtFermi: result.orbitalDOS.fermiWeighted?.hydrogenSCharacter ?? 0,
    topoProxyConfidence: result.derivedQuantities.topologicalInvariants.proxyConfidence,
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
