import { TargetProperties, InverseCandidate, computeTargetDistance, computeReward, CompositionBias } from "./target-schema";
import { generateInverseCandidates, refineCandidate, createInitialBias } from "./inverse-generator";
import { checkPhysicsConstraints, createConstraintRegistry, ConstraintRegistry, type ConstraintResult } from "./physics-constraint-engine";
import { evaluatePillars, type PillarEvaluation, type SCPillarTargets } from "./sc-pillars-optimizer";
import { runDifferentiableOptimization } from "./differentiable-optimizer";
import { solveConstraints, type ConstraintSolution } from "./constraint-solver";
import { getConstraintGraphGuidance } from "./constraint-graph-solver";
import { updateLearningState, deriveCompositionBias, createInitialLearningState, type InverseLearningState } from "./inverse-learning";
import { gbPredict } from "../learning/gradient-boost";
import { extractFeatures } from "../learning/ml-predictor";
import { gnnPredictWithUncertainty } from "../learning/graph-neural-net";
import {
  type DesignProgram, type DesignGraph, type GraphAnalysis,
  generateDesignProgram, executeDesignProgram, mutateDesignProgram, crossoverPrograms,
  mutateDesignGraph, analyzeGraph,
  programToGraph, graphToProgram,
  registerProgram, registerGraph, recordConversion, linkProgramToGraph,
  getDesignRepresentationStats,
} from "./design-representations";

export type StrategyType =
  | "hydride-cage-optimizer"
  | "layered-intercalation"
  | "high-entropy-alloy"
  | "light-element-phonon"
  | "topological-edge"
  | "pressure-stabilized"
  | "electron-phonon-resonance"
  | "charge-transfer-layer"
  | "custom";

export interface DesignStrategy {
  id: string;
  type: StrategyType;
  name: string;
  description: string;
  parameters: StrategyParameters;
  fitness: number;
  uses: number;
  successes: number;
  avgTc: number;
  bestTc: number;
  createdAt: number;
  parentId: string | null;
  generation: number;
}

export interface StrategyParameters {
  elementPool: string[];
  stoichiometryTemplate: string;
  prototypePreference: string[];
  hydrogenDensity: "none" | "low" | "medium" | "high" | "ultra";
  targetLambdaRange: [number, number];
  pressureRegime: "ambient" | "moderate" | "high";
  branchingFactor: number;
  mutationRate: number;
  explorationWeight: number;
}

export interface FailureRecord {
  formula: string;
  strategyId: string;
  iteration: number;
  failureType: FailureType;
  failureReason: string;
  metrics: { tc: number; lambda: number; pressure: number; hullDistance: number; constraintPenalty: number };
  suggestion: string;
  timestamp: number;
}

export type FailureType =
  | "low-tc"
  | "constraint-violation"
  | "high-pressure"
  | "thermodynamic-instability"
  | "poor-electron-phonon"
  | "synthesis-infeasible"
  | "phonon-instability"
  | "structural-instability"
  | "insufficient-dos";

export interface KnowledgeEntry {
  strategyType: StrategyType;
  pattern: string;
  failureCount: number;
  successCount: number;
  suggestion: string;
  confidence: number;
  lastUpdated: number;
}

export interface ImplicitNeuralField {
  weights: number[][];
  biases: number[];
  layerSizes: number[];
  activations: string[];
}

export interface INRPrediction {
  density: number;
  gradient: [number, number, number];
  curvature: number;
}

export type CandidateDerivation = "generated" | "refinement" | "differentiable-opt";

export interface DesignCandidate {
  formula: string;
  strategyId: string;
  iteration: number;
  derivation: CandidateDerivation;
  constraintResult: ConstraintResult | null;
  pillarEval: PillarEvaluation | null;
  surrogateScores: {
    gbTc: number;
    gnnTc: number;
    ensembleTc: number;
    gnnLambda: number;
    confidence: number;
  } | null;
  inrField: ImplicitNeuralField | null;
  inrDensity: number;
  designProgram: DesignProgram | null;
  designGraph: DesignGraph | null;
  targetDistance: number;
  reward: number;
  physicsValidated: boolean;
  failureAnalysis: FailureRecord | null;
}

export interface LabIterationResult {
  iteration: number;
  activeStrategy: string;
  candidatesGenerated: number;
  constraintsPassed: number;
  surrogateEvaluated: number;
  bestTc: number;
  bestFormula: string;
  avgTargetDistance: number;
  failuresAnalyzed: number;
  knowledgeEntriesUpdated: number;
  strategiesEvolved: number;
  convergenceDelta: number;
  topCandidates: { formula: string; tc: number; distance: number; strategy: string }[];
  programsGenerated: number;
  graphsGenerated: number;
  crossRepresentationConversions: number;
  wallTimeMs: number;
}

export interface LabState {
  id: string;
  status: "idle" | "running" | "converged" | "completed" | "paused";
  iteration: number;
  targetTc: number;
  maxPressure: number;
  strategies: DesignStrategy[];
  activeStrategyId: string;
  knowledgeBase: KnowledgeEntry[];
  failureHistory: FailureRecord[];
  learningState: InverseLearningState;
  bias: CompositionBias;
  bestTcOverall: number;
  bestFormulaOverall: string;
  bestDistance: number;
  convergenceHistory: number[];
  iterationHistory: LabIterationResult[];
  totalGenerated: number;
  totalPassed: number;
  totalFailuresAnalyzed: number;
  totalStrategiesEvolved: number;
  maxIterations: number;
  startedAt: number;
  lastIterationAt: number;
  pipelineTiming: {
    surrogateMs: number;
    constraintMs: number;
    pillarMs: number;
    inrTrainingMs: number;
    designRepMs: number;
    iterationCount: number;
  };
}

const labRegistries = new Map<LabState, ConstraintRegistry>();

function getLabRegistry(state: LabState): ConstraintRegistry {
  let reg = labRegistries.get(state);
  if (!reg) {
    reg = createConstraintRegistry();
    labRegistries.set(state, reg);
  }
  return reg;
}

export interface BottleneckAnalysis {
  surrogateMs: number;
  constraintMs: number;
  pillarMs: number;
  inrTrainingMs: number;
  designRepMs: number;
  totalPipelineMs: number;
  bottleneck: string;
  percentages: Record<string, number>;
}

export interface LabStats {
  id: string;
  status: string;
  iteration: number;
  bestTc: number;
  bestFormula: string;
  bestDistance: number;
  totalGenerated: number;
  totalPassed: number;
  activeStrategy: { id: string; name: string; type: string; fitness: number; bestTc: number } | null;
  strategies: { id: string; name: string; type: string; fitness: number; uses: number; bestTc: number; generation: number }[];
  knowledgeBaseSize: number;
  failureBreakdown: Record<string, number>;
  topKnowledge: { pattern: string; suggestion: string; confidence: number }[];
  convergenceHistory: number[];
  topCandidates: { formula: string; tc: number; distance: number; strategy: string }[];
  strategiesEvolved: number;
  iterationsPerMinute: number;
  bottleneckAnalysis: BottleneckAnalysis | null;
}

const labs = new Map<string, LabState>();
let totalLabRuns = 0;

const ELEMENT_POOLS: Record<string, string[]> = {
  "hydride-cage-optimizer": ["La", "Y", "Sc", "Ca", "Sr", "Ba", "Th", "Ce", "Pr", "Nd", "Eu", "H"],
  "layered-intercalation": ["Cu", "Fe", "Ni", "Co", "Sr", "Ba", "La", "Y", "O", "As", "Se", "S"],
  "high-entropy-alloy": ["Nb", "Ti", "Zr", "Hf", "V", "Ta", "Mo", "W", "Re", "B", "C", "N"],
  "light-element-phonon": ["B", "C", "N", "Li", "Be", "Mg", "Al", "Si", "H"],
  "topological-edge": ["Bi", "Sb", "Te", "Se", "Sn", "Pb", "In", "Tl", "S"],
  "pressure-stabilized": ["La", "Y", "Ca", "H", "S", "Se", "P", "Cl", "Br"],
  "electron-phonon-resonance": ["Nb", "V", "Mo", "Sn", "Ge", "Si", "B", "N", "H"],
  "charge-transfer-layer": ["Cu", "Bi", "Sr", "Ca", "La", "Ba", "O", "Tl", "Hg", "Y"],
};

const STOICH_TEMPLATES: Record<string, string[]> = {
  "hydride-cage-optimizer": ["XH6", "XH8", "XH10", "X2H6", "XYH6", "XYH8", "XYH10"],
  "layered-intercalation": ["X2YZ2", "XYO3", "X2Y2Z", "XY2Z2"],
  "high-entropy-alloy": ["X3Y2Z", "XYZW", "X2YZW", "X3YZ"],
  "light-element-phonon": ["XB2", "XB6", "XC3", "XN2", "X2B4C"],
  "topological-edge": ["X2Y3", "XY", "X3Y2", "XYZ"],
  "pressure-stabilized": ["XH3", "XH4", "XH6", "X2H6", "XSH3"],
  "electron-phonon-resonance": ["X3Y", "XY2", "X2YZ", "XY3Z"],
  "charge-transfer-layer": ["XY2Z3O7", "X2Y2Z3O10", "XYO2", "X2YZ2O6"],
};

const PROTOTYPE_MAP: Record<string, string[]> = {
  "hydride-cage-optimizer": ["clathrate", "sodalite", "Im-3m"],
  "layered-intercalation": ["ThCr2Si2", "Perovskite", "Layered"],
  "high-entropy-alloy": ["A15", "BCC", "FCC"],
  "light-element-phonon": ["AlB2", "MgB2", "hexagonal"],
  "topological-edge": ["Bi2Se3", "NaCl", "tetradymite"],
  "pressure-stabilized": ["clathrate", "Im-3m", "Fm-3m"],
  "electron-phonon-resonance": ["A15", "Nb3Sn", "cubic"],
  "charge-transfer-layer": ["Perovskite", "YBCO", "Layered"],
};

const STRATEGY_COMPATIBLE_ELEMENTS: Record<string, string[]> = {
  "hydride-cage-optimizer": ["La", "Y", "Sc", "Ca", "Sr", "Ba", "Th", "Ce", "Pr", "Nd", "Eu", "H", "Zr", "Hf"],
  "layered-intercalation": ["Cu", "Fe", "Ni", "Co", "Sr", "Ba", "La", "Y", "O", "As", "Se", "S", "Ca", "Nd"],
  "high-entropy-alloy": ["Nb", "Ti", "Zr", "Hf", "V", "Ta", "Mo", "W", "Re", "B", "C", "N", "Cr"],
  "light-element-phonon": ["B", "C", "N", "Li", "Be", "Mg", "Al", "Si", "H", "Na", "Ca"],
  "topological-edge": ["Bi", "Sb", "Te", "Se", "Sn", "Pb", "In", "Tl", "S", "Cu", "Ag"],
  "pressure-stabilized": ["La", "Y", "Ca", "H", "S", "Se", "P", "Cl", "Br", "Sr", "Ba", "Ce"],
  "electron-phonon-resonance": ["Nb", "V", "Mo", "Sn", "Ge", "Si", "B", "N", "H", "Ti", "Zr"],
  "charge-transfer-layer": ["Cu", "Bi", "Sr", "Ca", "La", "Ba", "O", "Tl", "Hg", "Y", "Nd", "Pb"],
};

const LAMBDA_CEILING: Record<string, number> = {
  "hydride-cage-optimizer": 6.0,
  "layered-intercalation": 2.5,
  "high-entropy-alloy": 1.8,
  "light-element-phonon": 3.0,
  "topological-edge": 1.5,
  "pressure-stabilized": 5.0,
  "electron-phonon-resonance": 3.5,
  "charge-transfer-layer": 2.5,
};

function createDefaultStrategy(type: StrategyType, generation: number = 0, parentId: string | null = null): DesignStrategy {
  const pool = ELEMENT_POOLS[type] ?? ELEMENT_POOLS["hydride-cage-optimizer"];
  const templates = STOICH_TEMPLATES[type] ?? STOICH_TEMPLATES["hydride-cage-optimizer"];
  const prototypes = PROTOTYPE_MAP[type] ?? PROTOTYPE_MAP["hydride-cage-optimizer"];

  return {
    id: `strat-${type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    type,
    name: type.split("-").map(w => w[0].toUpperCase() + w.slice(1)).join(" "),
    description: `Strategy: ${type} (gen ${generation})`,
    parameters: {
      elementPool: pool,
      stoichiometryTemplate: templates[Math.floor(Math.random() * templates.length)],
      prototypePreference: prototypes,
      hydrogenDensity: type.includes("hydride") || type.includes("pressure") ? "high" : "medium",
      targetLambdaRange: [1.0, 4.0],
      pressureRegime: (type === "pressure-stabilized" || type === "hydride-cage-optimizer") ? "high" : "ambient",
      branchingFactor: 3,
      mutationRate: 0.15,
      explorationWeight: Math.max(0.1, 0.5 - generation * 0.05),
    },
    fitness: 0,
    uses: 0,
    successes: 0,
    avgTc: 0,
    bestTc: 0,
    createdAt: Date.now(),
    parentId,
    generation,
  };
}

function createImplicitNeuralField(): ImplicitNeuralField {
  const layerSizes = [3, 32, 32, 1];
  const weights: number[][] = [];
  const biases: number[] = [];

  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fanIn = layerSizes[l];
    const fanOut = layerSizes[l + 1];
    const scale = Math.sqrt(2.0 / fanIn);
    const layerWeights: number[] = [];
    for (let i = 0; i < fanIn * fanOut; i++) {
      layerWeights.push((Math.random() * 2 - 1) * scale);
    }
    weights.push(layerWeights);
    for (let i = 0; i < fanOut; i++) {
      biases.push(0);
    }
  }

  return { weights, biases, layerSizes, activations: ["relu", "relu", "sigmoid"] };
}

function estimateHullDistance(formula: string, pillarEval: PillarEvaluation | null, constraintPenalty: number = 0): number {
  const numbers = formula.match(/\d+/g)?.map(Number) || [];
  const elements = formula.match(/[A-Z][a-z]?/g) || [];
  const nElements = new Set(elements).size;

  let hullEstimate = 0.02;

  if (nElements >= 5) {
    hullEstimate += 0.03 * (nElements - 4);
  }

  const maxStoich = Math.max(...numbers, 1);
  if (maxStoich >= 10) {
    hullEstimate += 0.02 * Math.log2(maxStoich / 8);
  }

  if (pillarEval) {
    const structInstab = pillarEval.instability?.structuralInstability ?? 0;
    hullEstimate += structInstab * 0.08;

    const compositeFit = pillarEval.compositeFitness ?? 0.5;
    hullEstimate += (1 - compositeFit) * 0.04;
  }

  if (constraintPenalty > 0) {
    hullEstimate += constraintPenalty * 0.05;
  }

  return Math.min(0.5, Math.max(0.005, hullEstimate));
}

function evaluateINR(field: ImplicitNeuralField, x: number, y: number, z: number): INRPrediction {
  let input = [x, y, z];
  let biasOffset = 0;

  for (let l = 0; l < field.layerSizes.length - 1; l++) {
    const fanIn = field.layerSizes[l];
    const fanOut = field.layerSizes[l + 1];
    const output: number[] = new Array(fanOut).fill(0);

    for (let j = 0; j < fanOut; j++) {
      let sum = field.biases[biasOffset + j];
      for (let i = 0; i < fanIn; i++) {
        sum += input[i] * field.weights[l][i * fanOut + j];
      }
      if (field.activations[l] === "relu") {
        output[j] = Math.max(0, sum);
      } else if (field.activations[l] === "sigmoid") {
        output[j] = 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, sum))));
      } else {
        output[j] = sum;
      }
    }

    input = output;
    biasOffset += fanOut;
  }

  const density = input[0];

  const eps = 0.01;
  const dxPlus = evaluateINRRaw(field, x + eps, y, z);
  const dxMinus = evaluateINRRaw(field, x - eps, y, z);
  const dyPlus = evaluateINRRaw(field, x, y + eps, z);
  const dyMinus = evaluateINRRaw(field, x, y - eps, z);
  const dzPlus = evaluateINRRaw(field, x, y, z + eps);
  const dzMinus = evaluateINRRaw(field, x, y, z - eps);

  const gradient: [number, number, number] = [
    (dxPlus - dxMinus) / (2 * eps),
    (dyPlus - dyMinus) / (2 * eps),
    (dzPlus - dzMinus) / (2 * eps),
  ];

  const curvature = (dxPlus + dxMinus + dyPlus + dyMinus + dzPlus + dzMinus - 6 * density) / (eps * eps);

  return { density, gradient, curvature };
}

function evaluateINRRaw(field: ImplicitNeuralField, x: number, y: number, z: number): number {
  let input = [x, y, z];
  let biasOffset = 0;

  for (let l = 0; l < field.layerSizes.length - 1; l++) {
    const fanIn = field.layerSizes[l];
    const fanOut = field.layerSizes[l + 1];
    const output: number[] = new Array(fanOut).fill(0);

    for (let j = 0; j < fanOut; j++) {
      let sum = field.biases[biasOffset + j];
      for (let i = 0; i < fanIn; i++) {
        sum += input[i] * field.weights[l][i * fanOut + j];
      }
      if (field.activations[l] === "relu") {
        output[j] = Math.max(0, sum);
      } else if (field.activations[l] === "sigmoid") {
        output[j] = 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, sum))));
      } else {
        output[j] = sum;
      }
    }

    input = output;
    biasOffset += fanOut;
  }

  return input[0];
}

function forwardPass(field: ImplicitNeuralField, x: number, y: number, z: number): { activations: number[][]; preActivations: number[][] } {
  const activations: number[][] = [[x, y, z]];
  const preActivations: number[][] = [];
  let input = [x, y, z];
  let biasOffset = 0;

  for (let l = 0; l < field.layerSizes.length - 1; l++) {
    const fanIn = field.layerSizes[l];
    const fanOut = field.layerSizes[l + 1];
    const pre: number[] = new Array(fanOut).fill(0);
    const output: number[] = new Array(fanOut).fill(0);

    for (let j = 0; j < fanOut; j++) {
      let sum = field.biases[biasOffset + j];
      for (let i = 0; i < fanIn; i++) {
        sum += input[i] * field.weights[l][i * fanOut + j];
      }
      pre[j] = sum;
      if (field.activations[l] === "relu") {
        output[j] = Math.max(0, sum);
      } else if (field.activations[l] === "sigmoid") {
        const clamped = Math.max(-10, Math.min(10, sum));
        output[j] = 1 / (1 + Math.exp(-clamped));
      } else {
        output[j] = sum;
      }
    }

    preActivations.push(pre);
    activations.push(output);
    input = output;
    biasOffset += fanOut;
  }

  return { activations, preActivations };
}

function backpropagateINR(
  field: ImplicitNeuralField,
  activations: number[][],
  preActivations: number[][],
  outputError: number,
  lr: number,
): void {
  const numLayers = field.layerSizes.length - 1;
  let delta: number[] = [outputError];

  for (let l = numLayers - 1; l >= 0; l--) {
    const fanIn = field.layerSizes[l];
    const fanOut = field.layerSizes[l + 1];
    const act = field.activations[l];

    const localGrad: number[] = new Array(fanOut);
    for (let j = 0; j < fanOut; j++) {
      if (act === "relu") {
        localGrad[j] = preActivations[l][j] > 0 ? delta[j] : 0;
      } else if (act === "sigmoid") {
        const s = activations[l + 1][j];
        localGrad[j] = delta[j] * s * (1 - s);
      } else {
        localGrad[j] = delta[j];
      }
    }

    let biasOffset = 0;
    for (let ll = 0; ll < l; ll++) biasOffset += field.layerSizes[ll + 1];

    for (let j = 0; j < fanOut; j++) {
      field.biases[biasOffset + j] -= lr * localGrad[j];
    }
    for (let i = 0; i < fanIn; i++) {
      for (let j = 0; j < fanOut; j++) {
        field.weights[l][i * fanOut + j] -= lr * localGrad[j] * activations[l][i];
      }
    }

    if (l > 0) {
      const prevDelta: number[] = new Array(fanIn).fill(0);
      for (let i = 0; i < fanIn; i++) {
        for (let j = 0; j < fanOut; j++) {
          prevDelta[i] += field.weights[l][i * fanOut + j] * localGrad[j];
        }
      }
      delta = prevDelta;
    }
  }
}

function generateLatticeSites(nElements: number, totalAtoms: number): number[][] {
  const sites: number[][] = [];
  const nSites = Math.min(totalAtoms, 8);
  const spacing = 2.0 / Math.cbrt(Math.max(1, nSites));

  let seedVal = nElements * 7 + totalAtoms * 13;
  function seededRand(): number {
    seedVal = (seedVal * 1103515245 + 12345) & 0x7fffffff;
    return (seedVal / 0x7fffffff) * 2 - 1;
  }

  for (let i = 0; i < nSites; i++) {
    const sx = seededRand() * (1 - spacing * 0.3);
    const sy = seededRand() * (1 - spacing * 0.3);
    const sz = seededRand() * (1 - spacing * 0.3);
    sites.push([sx, sy, sz]);
  }
  return sites;
}

function computeTargetDensity(
  x: number, y: number, z: number,
  sites: number[][],
  sigma: number,
): number {
  let density = 0;
  for (const [sx, sy, sz] of sites) {
    const dx = x - sx, dy = y - sy, dz = z - sz;
    const r2 = dx * dx + dy * dy + dz * dz;
    density += Math.exp(-r2 / (2 * sigma * sigma));
  }
  return Math.min(1, density / Math.max(1, sites.length) * 2);
}

function trainINRFromComposition(formula: string): ImplicitNeuralField {
  const field = createImplicitNeuralField();

  const elements = formula.match(/[A-Z][a-z]?/g) || [];
  const numbers = formula.match(/\d+/g) || [];
  const totalAtoms = numbers.reduce((s, n) => s + parseInt(n), 0) || elements.length;

  const sites = generateLatticeSites(elements.length, totalAtoms);
  const sigma = 0.4 / Math.cbrt(Math.max(1, totalAtoms / 4));

  const lr = 0.005;
  const iterations = 50;

  for (let iter = 0; iter < iterations; iter++) {
    for (let sample = 0; sample < 8; sample++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      const z = Math.random() * 2 - 1;

      const targetDensity = computeTargetDensity(x, y, z, sites, sigma);

      const { activations, preActivations } = forwardPass(field, x, y, z);
      const predicted = activations[activations.length - 1][0];
      const error = predicted - targetDensity;

      backpropagateINR(field, activations, preActivations, error, lr);
    }
  }

  return field;
}

function analyzeFailure(
  formula: string,
  strategyId: string,
  iteration: number,
  tc: number,
  lambda: number,
  pressure: number,
  hullDistance: number,
  constraintPenalty: number,
  targetTc: number,
  maxPressure: number,
  structuralInstability: number,
): FailureRecord | null {
  let failureType: FailureType | null = null;
  let failureReason = "";
  let suggestion = "";

  if (structuralInstability > 0.6) {
    failureType = "structural-instability";
    failureReason = `Structural instability score ${structuralInstability.toFixed(2)} indicates imaginary phonon modes or lattice collapse`;
    suggestion = "Lattice is dynamically unstable; try substituting elements with different atomic radii or reducing stoichiometry of light atoms";
  } else if (tc < targetTc * 0.3) {
    failureType = "low-tc";
    failureReason = `Tc=${Math.round(tc)}K far below target ${targetTc}K (${Math.round(tc / targetTc * 100)}% of target)`;
    if (lambda < 1.0) {
      suggestion = "Increase electron-phonon coupling by targeting higher-Z elements or hydrogen-rich compositions";
    } else if (structuralInstability > 0.4) {
      suggestion = "Lambda adequate but structural instability detected; phonon softening may suppress Tc. Try pressure stabilization or stiffer lattice formers";
    } else {
      suggestion = "Lambda adequate but phonon frequencies may be too low; try lighter elements or higher-symmetry structures";
    }
  } else if (constraintPenalty > 0.5) {
    failureType = "constraint-violation";
    failureReason = `Constraint penalty ${constraintPenalty.toFixed(2)} exceeds threshold; formula may violate charge neutrality or bonding rules`;
    suggestion = "Adjust stoichiometry to satisfy charge balance; consider more stable oxidation states";
  } else if (pressure > maxPressure) {
    failureType = "high-pressure";
    failureReason = `Estimated pressure ${Math.round(pressure)} GPa exceeds limit ${maxPressure} GPa`;
    suggestion = "Try chemical pre-compression or substitute heavier analogues to reduce required pressure";
  } else if (hullDistance > 0.1) {
    failureType = "thermodynamic-instability";
    failureReason = `Hull distance ${hullDistance.toFixed(3)} eV/atom indicates thermodynamic instability`;
    suggestion = "Add stabilizing dopants or target metastable phases with kinetic barriers";
  } else if (lambda < 0.5) {
    failureType = "poor-electron-phonon";
    failureReason = `Lambda=${lambda.toFixed(2)} too low for significant superconductivity`;
    suggestion = "Target materials with van Hove singularities near Fermi level or stronger phonon modes";
  } else if (structuralInstability > 0.4) {
    failureType = "structural-instability";
    failureReason = `Moderate structural instability (${structuralInstability.toFixed(2)}) may compromise lattice dynamics; Tc=${Math.round(tc)}K prediction uncertain`;
    suggestion = "Borderline lattice stability; consider pressure stabilization or substituting elements with better size matching before optimizing DOS";
  } else if (tc < targetTc * 0.6) {
    failureType = "insufficient-dos";
    failureReason = `Tc=${Math.round(tc)}K below 60% of target; likely insufficient density of states`;
    suggestion = "Increase DOS at Fermi level through flat bands, heavy-fermion elements, or nesting";
  }

  if (!failureType) return null;

  return {
    formula,
    strategyId,
    iteration,
    failureType,
    failureReason,
    metrics: { tc, lambda, pressure, hullDistance, constraintPenalty },
    suggestion,
    timestamp: Date.now(),
  };
}

function updateKnowledgeBase(kb: KnowledgeEntry[], failure: FailureRecord, strategyType: StrategyType): void {
  const pattern = `${strategyType}:${failure.failureType}`;

  const existing = kb.find(e => e.pattern === pattern);
  if (existing) {
    existing.failureCount++;
    existing.confidence = Math.min(0.95, existing.confidence + 0.02);
    existing.lastUpdated = Date.now();
    if (failure.suggestion.length > existing.suggestion.length) {
      existing.suggestion = failure.suggestion;
    }
  } else {
    kb.push({
      strategyType,
      pattern,
      failureCount: 1,
      successCount: 0,
      suggestion: failure.suggestion,
      confidence: 0.3,
      lastUpdated: Date.now(),
    });
  }
}

function recordSuccess(kb: KnowledgeEntry[], strategyType: StrategyType): void {
  const entries = kb.filter(e => e.strategyType === strategyType);
  for (const entry of entries) {
    entry.successCount++;
    entry.confidence = Math.max(0.1, entry.confidence - 0.01);
    entry.lastUpdated = Date.now();
  }
}

function evolveStrategy(
  parent: DesignStrategy,
  kb: KnowledgeEntry[],
  allStrategies: DesignStrategy[],
): DesignStrategy {
  const child = createDefaultStrategy(parent.type, parent.generation + 1, parent.id);
  child.parameters = JSON.parse(JSON.stringify(parent.parameters));

  const relevantKnowledge = kb.filter(e => e.strategyType === parent.type && e.confidence > 0.4);

  const evolveCeiling = LAMBDA_CEILING[parent.type] ?? 4.0;

  for (const knowledge of relevantKnowledge) {
    if (knowledge.pattern.includes("low-tc") && knowledge.confidence > 0.5) {
      child.parameters.targetLambdaRange = [
        Math.min(child.parameters.targetLambdaRange[0] + 0.3, evolveCeiling),
        Math.min(child.parameters.targetLambdaRange[1] + 0.5, evolveCeiling),
      ];
      if (!child.parameters.elementPool.includes("H") && parent.type !== "topological-edge") {
        child.parameters.elementPool.push("H");
        child.parameters.hydrogenDensity = "high";
      }
    }

    if (knowledge.pattern.includes("constraint-violation") && knowledge.confidence > 0.5) {
      child.parameters.branchingFactor = Math.max(2, child.parameters.branchingFactor - 1);
      child.parameters.mutationRate = Math.max(0.05, child.parameters.mutationRate - 0.03);
    }

    if (knowledge.pattern.includes("high-pressure") && knowledge.confidence > 0.5) {
      child.parameters.pressureRegime = "ambient";
      const heavierElements = ["La", "Y", "Ba", "Sr", "Ca"];
      for (const el of heavierElements) {
        if (!child.parameters.elementPool.includes(el)) {
          child.parameters.elementPool.push(el);
          break;
        }
      }
    }

    if (knowledge.pattern.includes("structural-instability") && knowledge.confidence > 0.4) {
      child.parameters.mutationRate = Math.max(0.05, child.parameters.mutationRate - 0.05);
      child.parameters.branchingFactor = Math.max(2, child.parameters.branchingFactor - 1);
      const stabilizers = ["La", "Y", "Zr"];
      for (const el of stabilizers) {
        if (!child.parameters.elementPool.includes(el)) {
          child.parameters.elementPool.push(el);
          break;
        }
      }
    }

    if (knowledge.pattern.includes("poor-electron-phonon") && knowledge.confidence > 0.4) {
      const ephElements = ["Nb", "V", "Ti", "Mo", "B"];
      for (const el of ephElements) {
        if (!child.parameters.elementPool.includes(el)) {
          child.parameters.elementPool.push(el);
          break;
        }
      }
    }
  }

  const basePoolSize = (ELEMENT_POOLS[parent.type] ?? ELEMENT_POOLS["hydride-cage-optimizer"]).length;
  const maxPoolSize = basePoolSize + 4;

  if (Math.random() < child.parameters.mutationRate) {
    const allPools = Object.values(ELEMENT_POOLS);
    const randomPool = allPools[Math.floor(Math.random() * allPools.length)];
    const randomElement = randomPool[Math.floor(Math.random() * randomPool.length)];
    if (!child.parameters.elementPool.includes(randomElement)) {
      if (child.parameters.elementPool.length >= maxPoolSize) {
        const basePool = ELEMENT_POOLS[parent.type] ?? ELEMENT_POOLS["hydride-cage-optimizer"];
        const removable = child.parameters.elementPool.filter(el => !basePool.includes(el));
        if (removable.length > 0) {
          const removeIdx = child.parameters.elementPool.indexOf(removable[Math.floor(Math.random() * removable.length)]);
          child.parameters.elementPool.splice(removeIdx, 1);
        }
      }
      if (child.parameters.elementPool.length < maxPoolSize) {
        child.parameters.elementPool.push(randomElement);
      }
    }
  }

  if (Math.random() < child.parameters.mutationRate) {
    const templates = Object.values(STOICH_TEMPLATES).flat();
    child.parameters.stoichiometryTemplate = templates[Math.floor(Math.random() * templates.length)];
  }

  const bestStrategy = allStrategies.reduce((best, s) => s.fitness > best.fitness ? s : best, allStrategies[0]);
  if (bestStrategy && bestStrategy.id !== parent.id && Math.random() < 0.3) {
    const crossoverPool = bestStrategy.parameters.elementPool;
    const crossoverElement = crossoverPool[Math.floor(Math.random() * crossoverPool.length)];
    if (!child.parameters.elementPool.includes(crossoverElement)) {
      if (child.parameters.elementPool.length >= maxPoolSize) {
        const basePool = ELEMENT_POOLS[parent.type] ?? ELEMENT_POOLS["hydride-cage-optimizer"];
        const removable = child.parameters.elementPool.filter(el => !basePool.includes(el));
        if (removable.length > 0) {
          const removeIdx = child.parameters.elementPool.indexOf(removable[Math.floor(Math.random() * removable.length)]);
          child.parameters.elementPool.splice(removeIdx, 1);
        }
      }
      if (child.parameters.elementPool.length < maxPoolSize) {
        child.parameters.elementPool.push(crossoverElement);
      }
    }
  }

  child.parameters.explorationWeight = Math.max(0.05, parent.parameters.explorationWeight - 0.02);

  return child;
}

function selectStrategy(strategies: DesignStrategy[], iteration: number): DesignStrategy {
  if (strategies.length === 0) throw new Error("No strategies available");

  const temperature = Math.max(0.1, 1.0 - iteration * 0.005);

  const combined = strategies.map(s => {
    const exploration = s.uses === 0 ? 1.0 : Math.sqrt(Math.log(iteration + 1) / s.uses);
    return { strategy: s, exploration };
  });

  const sorted = [...combined].sort((a, b) => a.strategy.fitness - b.strategy.fitness);
  const rankMap = new Map<string, number>();
  for (let i = 0; i < sorted.length; i++) {
    rankMap.set(sorted[i].strategy.id, (i + 1) / sorted.length);
  }

  const scores = combined.map(c => {
    const rankScore = rankMap.get(c.strategy.id)!;
    return rankScore * (1 - temperature) + c.exploration * temperature;
  });

  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp((s - maxScore) / Math.max(0.1, temperature)));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const probs = expScores.map(s => s / sumExp);

  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return strategies[i];
  }

  return strategies[strategies.length - 1];
}

function generateCandidatesFromStrategy(
  strategy: DesignStrategy,
  target: TargetProperties,
  labId: string,
  iteration: number,
  bias: CompositionBias,
  count: number,
): InverseCandidate[] {
  const modifiedTarget: TargetProperties = {
    ...target,
    preferredElements: strategy.parameters.elementPool,
    preferredPrototypes: strategy.parameters.prototypePreference,
  };

  return generateInverseCandidates(modifiedTarget, labId, iteration, bias, count);
}

export function createLab(id: string, targetTc: number = 293, maxPressure: number = 50, maxIterations: number = 500): LabState {
  const target: TargetProperties = {
    targetTc,
    maxPressure,
    minLambda: 1.5,
    maxHullDistance: 0.05,
    metallicRequired: true,
    phononStable: true,
  };

  const initialStrategies: DesignStrategy[] = [
    createDefaultStrategy("hydride-cage-optimizer"),
    createDefaultStrategy("layered-intercalation"),
    createDefaultStrategy("high-entropy-alloy"),
    createDefaultStrategy("light-element-phonon"),
    createDefaultStrategy("electron-phonon-resonance"),
    createDefaultStrategy("charge-transfer-layer"),
    createDefaultStrategy("pressure-stabilized"),
    createDefaultStrategy("topological-edge"),
  ];

  const state: LabState = {
    id,
    status: "idle",
    iteration: 0,
    targetTc,
    maxPressure,
    strategies: initialStrategies,
    activeStrategyId: initialStrategies[0].id,
    knowledgeBase: [],
    failureHistory: [],
    learningState: createInitialLearningState(),
    bias: createInitialBias(target),
    bestTcOverall: 0,
    bestFormulaOverall: "",
    bestDistance: 1.0,
    convergenceHistory: [],
    iterationHistory: [],
    totalGenerated: 0,
    totalPassed: 0,
    totalFailuresAnalyzed: 0,
    totalStrategiesEvolved: 0,
    maxIterations,
    startedAt: Date.now(),
    lastIterationAt: 0,
    pipelineTiming: {
      surrogateMs: 0,
      constraintMs: 0,
      pillarMs: 0,
      inrTrainingMs: 0,
      designRepMs: 0,
      iterationCount: 0,
    },
  };

  labs.set(id, state);
  return state;
}

export function runLabIteration(id: string): LabIterationResult | null {
  const state = labs.get(id);
  if (!state || state.status === "converged" || state.status === "completed" || state.status === "paused") return null;

  const startTime = Date.now();
  state.status = "running";
  state.iteration++;
  state.lastIterationAt = Date.now();

  const strategy = selectStrategy(state.strategies, state.iteration);
  state.activeStrategyId = strategy.id;
  strategy.uses++;

  const target: TargetProperties = {
    targetTc: state.targetTc,
    maxPressure: state.maxPressure,
    minLambda: 1.5,
    maxHullDistance: 0.05,
    metallicRequired: true,
    phononStable: true,
  };

  let constraintGuidance: { suggestedElements?: string[]; lambdaRange?: [number, number] } = {};
  try {
    const guidance = getConstraintGraphGuidance(state.targetTc);
    if (guidance) {
      constraintGuidance = {
        suggestedElements: guidance.preferredElements,
      };
      if (constraintGuidance.suggestedElements) {
        const compatible = STRATEGY_COMPATIBLE_ELEMENTS[strategy.type] ?? [];
        const basePool = ELEMENT_POOLS[strategy.type] ?? ELEMENT_POOLS["hydride-cage-optimizer"];
        const maxPoolSize = basePool.length + 4;
        const filtered = constraintGuidance.suggestedElements
          .filter(el => compatible.includes(el));
        for (const el of filtered.slice(0, 2)) {
          if (!strategy.parameters.elementPool.includes(el) && strategy.parameters.elementPool.length < maxPoolSize) {
            strategy.parameters.elementPool.push(el);
          }
        }
      }
    }
  } catch {}

  const lambdaCeiling = LAMBDA_CEILING[strategy.type] ?? 4.0;
  try {
    const solution = solveConstraints(state.targetTc, 0.10);
    if (solution?.requiredLambda) {
      strategy.parameters.targetLambdaRange = [
        Math.min(lambdaCeiling, Math.max(strategy.parameters.targetLambdaRange[0], solution.requiredLambda.min)),
        Math.min(lambdaCeiling, Math.max(strategy.parameters.targetLambdaRange[1], solution.requiredLambda.optimal)),
      ];
    }
  } catch {}

  const candidateCount = 30 + Math.floor(strategy.parameters.branchingFactor * 5);
  const rawCandidates = generateCandidatesFromStrategy(
    strategy, target, state.id, state.iteration, state.bias, candidateCount,
  );

  if (state.iteration > 5 && state.learningState.bestCandidates.length > 0) {
    const topN = state.learningState.bestCandidates.slice(0, 3);
    for (const best of topN) {
      try {
        const refined = refineCandidate(best, target, state.bias);
        for (const r of refined.slice(0, 3)) {
          r.derivation = "refinement";
          rawCandidates.push(r);
        }
      } catch {}
    }
  }

  if (state.iteration > 8 && state.learningState.bestCandidates.length > 0) {
    const pillarTargets: SCPillarTargets = {
      minCoupling: strategy.parameters.targetLambdaRange[0],
      minPhonon: 0.5,
      minDos: 0.3,
      minNesting: 0.2,
      minPairingGlue: 0.3,
      minInstability: 0.10,
      minHydrogenCage: strategy.parameters.hydrogenDensity === "high" ? 0.4 : 0.0,
    };
    const topForGrad = state.learningState.bestCandidates.slice(0, 2);
    for (const best of topForGrad) {
      try {
        const preEval = evaluatePillars(best.formula, pillarTargets, { maxPressureGPa: state.maxPressure });
        if (preEval.compositeFitness < 0.7) continue;
        const diffResult = runDifferentiableOptimization(best.formula, target);
        if (diffResult?.optimizedFormula && diffResult.optimizedFormula !== best.formula) {
          rawCandidates.push({
            formula: diffResult.optimizedFormula,
            source: "inverse" as const,
            derivation: "differentiable-opt" as const,
            campaignId: state.id,
            targetDistance: 1.0,
            iteration: state.iteration,
          });
        }
      } catch {}
    }
  }

  const uniqueFormulas = new Set<string>();
  const deduplicated = rawCandidates.filter(c => {
    if (uniqueFormulas.has(c.formula)) return false;
    uniqueFormulas.add(c.formula);
    return true;
  });

  state.totalGenerated += deduplicated.length;

  const labRegistry = getLabRegistry(state);
  const candidates: DesignCandidate[] = [];
  let iterConversions = 0;
  let iterProgramsGenerated = 0;
  let iterGraphsGenerated = 0;

  let iterSurrogateMs = 0;
  let iterConstraintMs = 0;
  let iterPillarMs = 0;
  let iterInrMs = 0;
  let iterDesignMs = 0;

  for (const raw of deduplicated) {
    const candidate: DesignCandidate = {
      formula: raw.formula,
      strategyId: strategy.id,
      iteration: state.iteration,
      derivation: (raw.derivation as CandidateDerivation) ?? "generated",
      constraintResult: null,
      pillarEval: null,
      surrogateScores: null,
      inrField: null,
      inrDensity: 0,
      designProgram: null,
      designGraph: null,
      targetDistance: 1.0,
      reward: 0,
      physicsValidated: false,
      failureAnalysis: null,
    };

    try {
      const t0Constraint = Date.now();
      try {
      candidate.constraintResult = checkPhysicsConstraints(raw.formula, { maxPressureGPa: state.maxPressure, registry: labRegistry });
      } finally { iterConstraintMs += Date.now() - t0Constraint; }
      if (!candidate.constraintResult.isValid && candidate.constraintResult.totalPenalty > 0.5) {
        const failure = analyzeFailure(
          raw.formula, strategy.id, state.iteration,
          0, 0, 0, 0, candidate.constraintResult.totalPenalty,
          state.targetTc, state.maxPressure,
          0,
        );
        if (failure) {
          candidate.failureAnalysis = failure;
          state.failureHistory.push(failure);
          state.totalFailuresAnalyzed++;
          updateKnowledgeBase(state.knowledgeBase, failure, strategy.type);
        }
        continue;
      }
    } catch {}

    try {
      const t0Surrogate = Date.now();
      try {
        const features = extractFeatures(raw.formula);
        const gb = gbPredict(features);
        const gnn = gnnPredictWithUncertainty(raw.formula);
        const gnnConf = gnn.confidence ?? 0;
        const gnnWeight = gnnConf > 0.5 ? 0.65 : gnnConf > 0.3 ? 0.45 : 0.25;
        const gbWeight = 1 - gnnWeight;
        const ensembleTc = gb.tcPredicted * gbWeight + (gnn.tc ?? 0) * gnnWeight;

        candidate.surrogateScores = {
          gbTc: gb.tcPredicted,
          gnnTc: gnn.tc ?? 0,
          ensembleTc,
          gnnLambda: gnn.lambda ?? 0,
          confidence: gnn.confidence ?? 0,
        };

        const hullEst = estimateHullDistance(raw.formula, candidate.pillarEval, candidate.constraintResult?.totalPenalty ?? 0);

        candidate.targetDistance = computeTargetDistance(target, {
          tc: ensembleTc,
          lambda: gnn.lambda ?? gb.score * 2,
          hull: hullEst,
          pressure: 0,
        });
        candidate.reward = computeReward(candidate.targetDistance);
        candidate.physicsValidated = true;
      } finally { iterSurrogateMs += Date.now() - t0Surrogate; }

      try {
        const t0Pillar = Date.now();
        try {
          const pillarTargets: SCPillarTargets = {
            minLambda: strategy.parameters.targetLambdaRange[1],
            minOmegaLogK: 500,
            minDOS: 3.0,
            minNesting: 0.5,
            minFlatBand: 0.6,
            minPairingGlue: 0.5,
            minInstability: 0.3,
            minHydrogenCage: strategy.parameters.hydrogenDensity === "high" ? 0.6 : 0.3,
            preferredMotifs: ["cage", "layered", "kagome"],
          };
          candidate.pillarEval = evaluatePillars(raw.formula, pillarTargets, { maxPressureGPa: state.maxPressure });
        } finally { iterPillarMs += Date.now() - t0Pillar; }
      } catch {}
    } catch {
      continue;
    }

    if (candidate.surrogateScores) {
      const tc = candidate.surrogateScores.ensembleTc;
      const lambda = candidate.surrogateScores.gnnLambda;
      const penalty = candidate.constraintResult?.totalPenalty ?? 0;
      const phononScore = candidate.pillarEval?.pillarScores?.phonon ?? 0.5;
      const phononStable = phononScore > 0.3;
      const synthFeasibility = candidate.pillarEval?.compositeFitness ?? 0.5;

      const candidateHull = estimateHullDistance(raw.formula, candidate.pillarEval, penalty);

      if (!phononStable) {
        const failure: FailureRecord = {
          formula: raw.formula,
          strategyId: strategy.id,
          iteration: state.iteration,
          failureType: "phonon-instability",
          failureReason: `Phonon instability (score=${phononScore.toFixed(2)}) invalidates Tc prediction; McMillan/Allen-Dynes equations require a stable lattice (lambda=${lambda.toFixed(2)})`,
          metrics: { tc, lambda, pressure: 0, hullDistance: candidateHull, constraintPenalty: penalty },
          suggestion: "Lattice has imaginary phonon modes; target stiffer lattice, apply pressure stabilization, or use lighter elements for better phonon stability",
          timestamp: Date.now(),
        };
        candidate.failureAnalysis = failure;
        state.failureHistory.push(failure);
        state.totalFailuresAnalyzed++;
        updateKnowledgeBase(state.knowledgeBase, failure, strategy.type);
      } else if (synthFeasibility < 0.2 && tc < state.targetTc * 0.5) {
        const failure: FailureRecord = {
          formula: raw.formula,
          strategyId: strategy.id,
          iteration: state.iteration,
          failureType: "synthesis-infeasible",
          failureReason: `Low synthesis feasibility (${(synthFeasibility * 100).toFixed(0)}%) combined with Tc=${Math.round(tc)}K`,
          metrics: { tc, lambda, pressure: 0, hullDistance: candidateHull, constraintPenalty: penalty },
          suggestion: "Use more commonly synthesized compositions or standard structure types",
          timestamp: Date.now(),
        };
        candidate.failureAnalysis = failure;
        state.failureHistory.push(failure);
        state.totalFailuresAnalyzed++;
        updateKnowledgeBase(state.knowledgeBase, failure, strategy.type);
      } else if (tc < state.targetTc * 0.3 || lambda < 0.5 || penalty > 0.3) {
        const structInstab = candidate.pillarEval?.instability?.structuralInstability ?? 0;
        const failure = analyzeFailure(
          raw.formula, strategy.id, state.iteration,
          tc, lambda, 0, candidateHull, penalty,
          state.targetTc, state.maxPressure,
          structInstab,
        );
        if (failure) {
          candidate.failureAnalysis = failure;
          state.failureHistory.push(failure);
          state.totalFailuresAnalyzed++;
          updateKnowledgeBase(state.knowledgeBase, failure, strategy.type);
        }
      } else {
        recordSuccess(state.knowledgeBase, strategy.type);
      }
    }

    { const t0Inr = Date.now();
    try {
      candidate.inrField = trainINRFromComposition(raw.formula);
      let densitySum = 0;
      let densityMax = 0;
      const gridN = 3;
      const step = 2 / gridN;
      for (let ix = 0; ix < gridN; ix++) {
        for (let iy = 0; iy < gridN; iy++) {
          for (let iz = 0; iz < gridN; iz++) {
            const gx = -1 + step * (ix + 0.5);
            const gy = -1 + step * (iy + 0.5);
            const gz = -1 + step * (iz + 0.5);
            const pred = evaluateINR(candidate.inrField, gx, gy, gz);
            densitySum += pred.density;
            if (pred.density > densityMax) densityMax = pred.density;
          }
        }
      }
      const totalSamples = gridN * gridN * gridN;
      candidate.inrDensity = densitySum / totalSamples * 0.5 + densityMax * 0.5;
    } catch {} finally { iterInrMs += Date.now() - t0Inr; } }

    { const t0Design = Date.now();
    try {
      candidate.designProgram = generateDesignProgram(
        strategy.type, strategy.parameters.elementPool, state.iteration, null);
      const tc = candidate.surrogateScores?.ensembleTc ?? 0;
      registerProgram(candidate.designProgram, tc);
      iterProgramsGenerated++;

      const derivedGraph = programToGraph(candidate.designProgram);
      candidate.designGraph = derivedGraph;
      registerGraph(derivedGraph, tc);
      linkProgramToGraph(candidate.designProgram.id, derivedGraph.id);
      recordConversion();
      iterGraphsGenerated++;
      iterConversions++;
    } catch {} finally { iterDesignMs += Date.now() - t0Design; } }

    candidates.push(candidate);
  }

  state.totalPassed += candidates.filter(c => c.physicsValidated).length;

  state.pipelineTiming.surrogateMs += iterSurrogateMs;
  state.pipelineTiming.constraintMs += iterConstraintMs;
  state.pipelineTiming.pillarMs += iterPillarMs;
  state.pipelineTiming.inrTrainingMs += iterInrMs;
  state.pipelineTiming.designRepMs += iterDesignMs;
  state.pipelineTiming.iterationCount++;

  const scored = candidates
    .filter(c => c.surrogateScores !== null)
    .sort((a, b) => {
      const maxTc = Math.max(1, state.targetTc);
      const aNormTc = Math.min(1, a.surrogateScores!.ensembleTc / maxTc);
      const bNormTc = Math.min(1, b.surrogateScores!.ensembleTc / maxTc);
      const aStability = 1 - (a.pillarEval?.instability?.compositeInstability ?? 0.5);
      const bStability = 1 - (b.pillarEval?.instability?.compositeInstability ?? 0.5);
      const aFitness = a.pillarEval?.compositeFitness ?? 0;
      const bFitness = b.pillarEval?.compositeFitness ?? 0;
      const aScore = aNormTc * 0.5 + aStability * 0.3 + aFitness * 0.2;
      const bScore = bNormTc * 0.5 + bStability * 0.3 + bFitness * 0.2;
      return bScore - aScore;
    });

  const bestTcIter = scored.length > 0 ? Math.max(...scored.map(c => c.surrogateScores!.ensembleTc)) : 0;
  const bestFormulaIter = scored.length > 0 ? scored[0].formula : "";

  if (bestTcIter > state.bestTcOverall) {
    state.bestTcOverall = bestTcIter;
    state.bestFormulaOverall = bestFormulaIter;
  }

  if (bestTcIter > strategy.bestTc) {
    strategy.bestTc = bestTcIter;
  }
  strategy.avgTc = strategy.uses > 0
    ? (strategy.avgTc * (strategy.uses - 1) + bestTcIter) / strategy.uses
    : bestTcIter;
  strategy.successes += scored.filter(c => c.surrogateScores!.ensembleTc > state.targetTc * 0.3).length;
  strategy.fitness = strategy.bestTc / state.targetTc * 0.6 + (strategy.successes / Math.max(1, strategy.uses)) * 0.4;

  const successRate = strategy.successes / Math.max(1, strategy.uses);
  if (strategy.uses >= 3) {
    if (successRate > 0.5) {
      strategy.parameters.branchingFactor = Math.min(8, strategy.parameters.branchingFactor + 1);
    } else if (successRate < 0.1) {
      strategy.parameters.branchingFactor = Math.max(1, strategy.parameters.branchingFactor - 1);
    }
  }

  const bestDistIter = scored.length > 0 ? Math.min(...scored.map(c => c.targetDistance)) : 1.0;
  if (bestDistIter < state.bestDistance) {
    state.bestDistance = bestDistIter;
  }

  const avgDist = scored.length > 0
    ? scored.reduce((s, c) => s + c.targetDistance, 0) / scored.length
    : 1.0;

  const prevBestDist = state.convergenceHistory.length > 0
    ? state.convergenceHistory[state.convergenceHistory.length - 1]
    : 1.0;
  const convergenceDelta = prevBestDist - state.bestDistance;
  state.convergenceHistory.push(state.bestDistance);
  if (state.convergenceHistory.length > 200) {
    state.convergenceHistory = state.convergenceHistory.slice(-100);
  }

  const inverseCandidates: InverseCandidate[] = scored.map(c => ({
    formula: c.formula,
    source: "inverse" as const,
    derivation: c.derivation,
    campaignId: state.id,
    targetDistance: c.targetDistance,
    iteration: state.iteration,
    predictedTc: c.surrogateScores!.ensembleTc,
    predictedLambda: c.surrogateScores!.gnnLambda,
    predictedHull: estimateHullDistance(c.formula, c.pillarEval, c.constraintResult?.totalPenalty ?? 0),
    predictedPressure: 0,
  }));

  const results = scored.map(c => ({
    formula: c.formula,
    tc: c.surrogateScores!.ensembleTc,
    lambda: c.surrogateScores!.gnnLambda,
    hull: estimateHullDistance(c.formula, c.pillarEval, c.constraintResult?.totalPenalty ?? 0),
    pressure: 0,
    passedPipeline: c.physicsValidated,
  }));

  state.learningState = updateLearningState(state.learningState, inverseCandidates, results, target);
  const baseBias = createInitialBias(target);
  state.bias = deriveCompositionBias(state.learningState, baseBias);

  let strategiesEvolved = 0;
  if (state.iteration % 10 === 0 && state.iteration > 0) {
    const worstStrategies = [...state.strategies]
      .sort((a, b) => a.fitness - b.fitness)
      .slice(0, 2);

    const bestStrategies = [...state.strategies]
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, 2);

    const activeTypes = new Set(state.strategies.map(s => s.type));

    for (const worst of worstStrategies) {
      if (worst.uses >= 3 && worst.fitness < 0.1) {
        const parent = bestStrategies[Math.floor(Math.random() * bestStrategies.length)];

        const sameTypeCount = state.strategies.filter(s => s.type === parent.type).length;
        if (sameTypeCount >= Math.ceil(state.strategies.length / 2)) continue;

        if (activeTypes.size <= 3 && worst.type !== parent.type) continue;

        const evolved = evolveStrategy(parent, state.knowledgeBase, state.strategies);

        const parentPool = new Set(parent.parameters.elementPool);
        const evolvedPool = new Set(evolved.parameters.elementPool);
        const intersection = [...evolvedPool].filter(el => parentPool.has(el)).length;
        const union = new Set([...parentPool, ...evolvedPool]).size;
        const jaccard = union > 0 ? intersection / union : 1;
        if (jaccard > 0.85) {
          const basePool = ELEMENT_POOLS[evolved.type] ?? ELEMENT_POOLS["hydride-cage-optimizer"];
          const extraElements = basePool.filter(el => !evolvedPool.has(el));
          if (extraElements.length > 0) {
            const addCount = Math.min(2, extraElements.length);
            for (let i = 0; i < addCount; i++) {
              evolved.parameters.elementPool.push(extraElements[i]);
            }
          }
        }

        const idx = state.strategies.indexOf(worst);
        if (idx >= 0) {
          state.strategies[idx] = evolved;
          strategiesEvolved++;
          state.totalStrategiesEvolved++;
        }
      }
    }
  }

  if (state.convergenceHistory.length >= 20) {
    const last20 = state.convergenceHistory.slice(-20);
    const range = Math.max(...last20) - Math.min(...last20);
    if (range < 0.01) {
      if (state.bestDistance < 0.15) {
        state.status = "converged";
      } else {
        for (const s of state.strategies) {
          if (s.parameters.pressureRegime === "ambient") {
            s.parameters.pressureRegime = "moderate";
          } else if (s.parameters.pressureRegime === "moderate") {
            s.parameters.pressureRegime = "high";
          }

          const basePool = ELEMENT_POOLS[s.type] ?? ELEMENT_POOLS["hydride-cage-optimizer"];
          const compatible = STRATEGY_COMPATIBLE_ELEMENTS[s.type] ?? [];
          const unexplored = compatible.filter(el => !s.parameters.elementPool.includes(el));
          if (unexplored.length > 0) {
            const addCount = Math.min(3, unexplored.length);
            for (let i = 0; i < addCount; i++) {
              s.parameters.elementPool.push(unexplored[i]);
            }
          }

          s.parameters.mutationRate = Math.min(0.30, s.parameters.mutationRate + 0.05);
          s.parameters.explorationWeight = Math.min(0.8, s.parameters.explorationWeight + 0.15);
        }

        state.convergenceHistory = state.convergenceHistory.slice(-5);
      }
    }
  }

  if (state.iteration >= state.maxIterations) {
    state.status = "completed";
  }

  if (state.status !== "converged" && state.status !== "completed") {
    state.status = "running";
  }

  const topCands = scored.slice(0, 10).map(c => ({
    formula: c.formula,
    tc: c.surrogateScores!.ensembleTc,
    distance: c.targetDistance,
    strategy: strategy.name,
  }));

  const iterResult: LabIterationResult = {
    iteration: state.iteration,
    activeStrategy: strategy.name,
    candidatesGenerated: deduplicated.length,
    constraintsPassed: candidates.length,
    surrogateEvaluated: scored.length,
    bestTc: bestTcIter,
    bestFormula: bestFormulaIter,
    avgTargetDistance: avgDist,
    failuresAnalyzed: state.failureHistory.filter(f => f.iteration === state.iteration).length,
    knowledgeEntriesUpdated: state.knowledgeBase.length,
    strategiesEvolved,
    convergenceDelta,
    topCandidates: topCands,
    programsGenerated: iterProgramsGenerated,
    graphsGenerated: iterGraphsGenerated,
    crossRepresentationConversions: iterConversions,
    wallTimeMs: Date.now() - startTime,
  };

  state.iterationHistory.push(iterResult);
  if (state.iterationHistory.length > 50) {
    state.iterationHistory = state.iterationHistory.slice(-50);
  }

  if (state.failureHistory.length > 500) {
    state.failureHistory = state.failureHistory.slice(-500);
  }

  totalLabRuns++;

  return iterResult;
}

function computeBottleneckAnalysis(state: LabState): BottleneckAnalysis | null {
  const t = state.pipelineTiming;
  if (t.iterationCount === 0) return null;

  const total = t.surrogateMs + t.constraintMs + t.pillarMs + t.inrTrainingMs + t.designRepMs;
  if (total === 0) return null;

  const segments: Record<string, number> = {
    Surrogates: t.surrogateMs,
    Constraints: t.constraintMs,
    Pillars: t.pillarMs,
    "INR Training": t.inrTrainingMs,
    "Design Rep": t.designRepMs,
  };

  const percentages: Record<string, number> = {};
  let maxLabel = "Surrogates";
  let maxMs = 0;
  for (const [label, ms] of Object.entries(segments)) {
    percentages[label] = Math.round((ms / total) * 1000) / 10;
    if (ms > maxMs) {
      maxMs = ms;
      maxLabel = label;
    }
  }

  return {
    surrogateMs: Math.round(t.surrogateMs / t.iterationCount),
    constraintMs: Math.round(t.constraintMs / t.iterationCount),
    pillarMs: Math.round(t.pillarMs / t.iterationCount),
    inrTrainingMs: Math.round(t.inrTrainingMs / t.iterationCount),
    designRepMs: Math.round(t.designRepMs / t.iterationCount),
    totalPipelineMs: Math.round(total / t.iterationCount),
    bottleneck: maxLabel,
    percentages,
  };
}

export function getLabStats(id: string): LabStats | null {
  const state = labs.get(id);
  if (!state) return null;

  const elapsedMs = Date.now() - state.startedAt;
  const iterPerMin = elapsedMs > 0 ? (state.iteration / (elapsedMs / 60000)) : 0;

  const failureBreakdown: Record<string, number> = {};
  for (const f of state.failureHistory) {
    failureBreakdown[f.failureType] = (failureBreakdown[f.failureType] || 0) + 1;
  }

  const activeStrat = state.strategies.find(s => s.id === state.activeStrategyId);

  return {
    id: state.id,
    status: state.status,
    iteration: state.iteration,
    bestTc: state.bestTcOverall,
    bestFormula: state.bestFormulaOverall,
    bestDistance: state.bestDistance,
    totalGenerated: state.totalGenerated,
    totalPassed: state.totalPassed,
    activeStrategy: activeStrat ? {
      id: activeStrat.id,
      name: activeStrat.name,
      type: activeStrat.type,
      fitness: Math.round(activeStrat.fitness * 1000) / 1000,
      bestTc: Math.round(activeStrat.bestTc),
    } : null,
    strategies: state.strategies.map(s => ({
      id: s.id,
      name: s.name,
      type: s.type,
      fitness: Math.round(s.fitness * 1000) / 1000,
      uses: s.uses,
      bestTc: Math.round(s.bestTc),
      generation: s.generation,
    })),
    knowledgeBaseSize: state.knowledgeBase.length,
    failureBreakdown,
    topKnowledge: state.knowledgeBase
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8)
      .map(k => ({ pattern: k.pattern, suggestion: k.suggestion, confidence: Math.round(k.confidence * 100) / 100 })),
    convergenceHistory: state.convergenceHistory.slice(-100),
    topCandidates: state.learningState.bestCandidates.slice(0, 10).map(c => ({
      formula: c.formula,
      tc: c.predictedTc ?? 0,
      distance: c.targetDistance,
      strategy: state.strategies.find(s => s.id === state.activeStrategyId)?.name ?? "unknown",
    })),
    strategiesEvolved: state.totalStrategiesEvolved,
    iterationsPerMinute: Math.round(iterPerMin * 10) / 10,
    bottleneckAnalysis: computeBottleneckAnalysis(state),
  };
}

export function pauseLab(id: string): boolean {
  const state = labs.get(id);
  if (!state) return false;
  state.status = "paused";
  return true;
}

export function resumeLab(id: string): boolean {
  const state = labs.get(id);
  if (!state || state.status !== "paused") return false;
  state.status = "running";
  return true;
}

export function deleteLab(id: string): boolean {
  return labs.delete(id);
}

export function getAllLabStats(): {
  activeLabs: number;
  totalRuns: number;
  labs: { id: string; status: string; iteration: number; bestTc: number; bestFormula: string; strategiesEvolved: number; knowledgeEntries: number }[];
} {
  const labSummaries = Array.from(labs.entries()).map(([lid, s]) => ({
    id: lid,
    status: s.status,
    iteration: s.iteration,
    bestTc: s.bestTcOverall,
    bestFormula: s.bestFormulaOverall,
    strategiesEvolved: s.totalStrategiesEvolved,
    knowledgeEntries: s.knowledgeBase.length,
  }));

  return {
    activeLabs: labSummaries.filter(l => l.status === "running").length,
    totalRuns: totalLabRuns,
    labs: labSummaries,
  };
}

interface GlobalKnowledgeEntry extends KnowledgeEntry {
  sourceLabIds: string[];
  mergedAt: number;
  propagated?: boolean;
}

let globalKnowledgeBase: GlobalKnowledgeEntry[] = [];

export function syncGlobalKnowledgeBase(): {
  totalEntries: number;
  newEntriesMerged: number;
  labsSynced: string[];
} {
  const merged = new Map<string, GlobalKnowledgeEntry>();
  const labsSynced: string[] = [];

  for (const [labId, state] of labs.entries()) {
    labsSynced.push(labId);

    for (const entry of state.knowledgeBase) {
      if (entry.confidence < 0.3) continue;
      if ((entry as any)._fromGlobalSync) continue;

      const key = `${entry.strategyType}::${entry.pattern}`;
      const existing = merged.get(key);

      if (existing) {
        existing.failureCount += entry.failureCount;
        existing.successCount += entry.successCount;
        if (!existing.sourceLabIds.includes(labId)) {
          existing.sourceLabIds.push(labId);
        }
        if (entry.suggestion.length > existing.suggestion.length) {
          existing.suggestion = entry.suggestion;
        }
        existing.lastUpdated = Math.max(existing.lastUpdated, entry.lastUpdated);
      } else {
        merged.set(key, {
          strategyType: entry.strategyType,
          pattern: entry.pattern,
          failureCount: entry.failureCount,
          successCount: entry.successCount,
          suggestion: entry.suggestion,
          confidence: 0,
          lastUpdated: entry.lastUpdated,
          sourceLabIds: [labId],
          mergedAt: Date.now(),
        });
      }
    }
  }

  for (const entry of merged.values()) {
    const totalObs = entry.failureCount + entry.successCount;
    entry.confidence = totalObs > 0
      ? entry.successCount / totalObs * 0.6 + Math.min(1, totalObs / 20) * 0.4
      : 0;
  }

  const previousSize = globalKnowledgeBase.length;
  globalKnowledgeBase = Array.from(merged.values());

  let propagatedCount = 0;
  for (const [labId, state] of labs.entries()) {
    for (const globalEntry of globalKnowledgeBase) {
      if (globalEntry.confidence < 0.5) continue;
      if (globalEntry.sourceLabIds.length < 2) continue;
      if (globalEntry.sourceLabIds.length === 1 && globalEntry.sourceLabIds[0] === labId) continue;

      const localMatch = state.knowledgeBase.find(
        k => k.pattern === globalEntry.pattern && k.strategyType === globalEntry.strategyType
      );

      if (!localMatch) {
        const propagated: KnowledgeEntry & { _fromGlobalSync?: boolean } = {
          strategyType: globalEntry.strategyType,
          pattern: globalEntry.pattern,
          failureCount: globalEntry.failureCount,
          successCount: globalEntry.successCount,
          suggestion: globalEntry.suggestion,
          confidence: globalEntry.confidence * 0.8,
          lastUpdated: Date.now(),
          _fromGlobalSync: true,
        };
        state.knowledgeBase.push(propagated as KnowledgeEntry);
        propagatedCount++;
      }
    }
  }

  return {
    totalEntries: globalKnowledgeBase.length,
    newEntriesMerged: Math.max(0, globalKnowledgeBase.length - previousSize),
    labsSynced,
  };
}

export function getGlobalKnowledgeBase(): {
  entries: GlobalKnowledgeEntry[];
  totalEntries: number;
  topPatterns: { pattern: string; confidence: number; observations: number }[];
} {
  const sorted = [...globalKnowledgeBase].sort((a, b) => b.confidence - a.confidence);
  return {
    entries: sorted,
    totalEntries: sorted.length,
    topPatterns: sorted.slice(0, 15).map(e => ({
      pattern: e.pattern,
      confidence: Math.round(e.confidence * 100) / 100,
      observations: e.failureCount + e.successCount,
    })),
  };
}
