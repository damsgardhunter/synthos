import { TargetProperties, InverseCandidate, computeTargetDistance, computeReward, CompositionBias } from "./target-schema";
import { generateInverseCandidates, refineCandidate, createInitialBias } from "./inverse-generator";
import { checkPhysicsConstraints, type ConstraintResult } from "./physics-constraint-engine";
import { evaluatePillars, type PillarEvaluation, type SCPillarTargets } from "./sc-pillars-optimizer";
import { runDifferentiableOptimization } from "./differentiable-optimizer";
import { solveConstraints, type ConstraintSolution } from "./constraint-solver";
import { getConstraintGraphGuidance } from "./constraint-graph-solver";
import { updateLearningState, deriveCompositionBias, createInitialLearningState, type InverseLearningState } from "./inverse-learning";
import { gbPredict } from "../learning/gradient-boost";
import { extractFeatures } from "../learning/ml-predictor";
import { gnnPredictWithUncertainty } from "../learning/graph-neural-net";

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

export interface DesignCandidate {
  formula: string;
  strategyId: string;
  iteration: number;
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
      pressureRegime: type === "pressure-stabilized" ? "high" : "ambient",
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

function trainINRFromComposition(formula: string): ImplicitNeuralField {
  const field = createImplicitNeuralField();

  const elements = formula.match(/[A-Z][a-z]?/g) || [];
  const numbers = formula.match(/\d+/g) || [];
  const totalAtoms = numbers.reduce((s, n) => s + parseInt(n), 0) || elements.length;

  const lr = 0.01;
  const iterations = 50;

  for (let iter = 0; iter < iterations; iter++) {
    for (let sample = 0; sample < 8; sample++) {
      const x = Math.random() * 2 - 1;
      const y = Math.random() * 2 - 1;
      const z = Math.random() * 2 - 1;

      const r = Math.sqrt(x * x + y * y + z * z);
      const targetDensity = Math.exp(-r * totalAtoms / 5) * (0.3 + 0.7 * Math.sin(r * Math.PI * elements.length));
      const clampedTarget = Math.max(0, Math.min(1, targetDensity));

      const predicted = evaluateINRRaw(field, x, y, z);
      const error = predicted - clampedTarget;

      const lastLayerIdx = field.layerSizes.length - 2;
      const fanIn = field.layerSizes[lastLayerIdx];
      const fanOut = field.layerSizes[lastLayerIdx + 1];
      let biasOffset = 0;
      for (let l = 0; l < lastLayerIdx; l++) {
        biasOffset += field.layerSizes[l + 1];
      }

      for (let j = 0; j < fanOut; j++) {
        field.biases[biasOffset + j] -= lr * error * 0.1;
      }

      for (let i = 0; i < fanIn * fanOut; i++) {
        field.weights[lastLayerIdx][i] -= lr * error * 0.01;
      }
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
): FailureRecord | null {
  let failureType: FailureType | null = null;
  let failureReason = "";
  let suggestion = "";

  if (tc < targetTc * 0.3) {
    failureType = "low-tc";
    failureReason = `Tc=${Math.round(tc)}K far below target ${targetTc}K (${Math.round(tc / targetTc * 100)}% of target)`;
    if (lambda < 1.0) {
      suggestion = "Increase electron-phonon coupling by targeting higher-Z elements or hydrogen-rich compositions";
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

  for (const knowledge of relevantKnowledge) {
    if (knowledge.pattern.includes("low-tc") && knowledge.confidence > 0.5) {
      child.parameters.targetLambdaRange = [
        Math.min(child.parameters.targetLambdaRange[0] + 0.3, 3.0),
        Math.min(child.parameters.targetLambdaRange[1] + 0.5, 6.0),
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

  if (Math.random() < child.parameters.mutationRate) {
    const allPools = Object.values(ELEMENT_POOLS);
    const randomPool = allPools[Math.floor(Math.random() * allPools.length)];
    const randomElement = randomPool[Math.floor(Math.random() * randomPool.length)];
    if (!child.parameters.elementPool.includes(randomElement)) {
      child.parameters.elementPool.push(randomElement);
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
      child.parameters.elementPool.push(crossoverElement);
    }
  }

  child.parameters.explorationWeight = Math.max(0.05, parent.parameters.explorationWeight - 0.02);

  return child;
}

function selectStrategy(strategies: DesignStrategy[], iteration: number): DesignStrategy {
  if (strategies.length === 0) throw new Error("No strategies available");

  const temperature = Math.max(0.1, 1.0 - iteration * 0.005);

  const scores = strategies.map(s => {
    const exploitation = s.fitness;
    const exploration = s.uses === 0 ? 1.0 : Math.sqrt(Math.log(iteration + 1) / s.uses);
    return exploitation * (1 - temperature) + exploration * temperature;
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
        for (const el of constraintGuidance.suggestedElements.slice(0, 3)) {
          if (!strategy.parameters.elementPool.includes(el)) {
            strategy.parameters.elementPool.push(el);
          }
        }
      }
    }
  } catch {}

  try {
    const solution = solveConstraints(state.targetTc, 0.10);
    if (solution?.requiredLambda) {
      strategy.parameters.targetLambdaRange = [
        Math.max(strategy.parameters.targetLambdaRange[0], solution.requiredLambda.min),
        Math.max(strategy.parameters.targetLambdaRange[1], solution.requiredLambda.optimal),
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
        rawCandidates.push(...refined.slice(0, 3));
      } catch {}
    }
  }

  if (state.iteration > 8 && state.learningState.bestCandidates.length > 0) {
    const topForGrad = state.learningState.bestCandidates.slice(0, 2);
    for (const best of topForGrad) {
      try {
        const diffResult = runDifferentiableOptimization(best.formula, target);
        if (diffResult?.optimizedFormula && diffResult.optimizedFormula !== best.formula) {
          rawCandidates.push({
            formula: diffResult.optimizedFormula,
            source: "inverse" as const,
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

  const candidates: DesignCandidate[] = [];

  for (const raw of deduplicated) {
    const candidate: DesignCandidate = {
      formula: raw.formula,
      strategyId: strategy.id,
      iteration: state.iteration,
      constraintResult: null,
      pillarEval: null,
      surrogateScores: null,
      inrField: null,
      inrDensity: 0,
      targetDistance: 1.0,
      reward: 0,
      physicsValidated: false,
      failureAnalysis: null,
    };

    try {
      candidate.constraintResult = checkPhysicsConstraints(raw.formula);
      if (!candidate.constraintResult.isValid && candidate.constraintResult.totalPenalty > 0.5) {
        const failure = analyzeFailure(
          raw.formula, strategy.id, state.iteration,
          0, 0, 0, 0, candidate.constraintResult.totalPenalty,
          state.targetTc, state.maxPressure,
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
      const features = extractFeatures(raw.formula);
      const gb = gbPredict(features);
      const gnn = gnnPredictWithUncertainty(raw.formula);
      const ensembleTc = gb.tcPredicted * 0.35 + (gnn.tc ?? 0) * 0.65;

      candidate.surrogateScores = {
        gbTc: gb.tcPredicted,
        gnnTc: gnn.tc ?? 0,
        ensembleTc,
        gnnLambda: gnn.lambda ?? 0,
        confidence: gnn.confidence ?? 0,
      };

      candidate.targetDistance = computeTargetDistance(target, {
        tc: ensembleTc,
        lambda: gnn.lambda ?? gb.score * 2,
        hull: 0.02,
        pressure: 0,
      });
      candidate.reward = computeReward(candidate.targetDistance);
      candidate.physicsValidated = true;

      try {
        const pillarTargets: SCPillarTargets = {
          targetTc: state.targetTc,
          targetLambda: strategy.parameters.targetLambdaRange[1],
          targetOmegaLog: 500,
          targetDOS: 3.0,
          targetNesting: 0.5,
          targetMotifMatch: 0.6,
          targetPairingGlue: 0.5,
          targetInstability: 0.3,
          targetHydrogenCage: strategy.parameters.hydrogenDensity === "high" ? 0.6 : 0.3,
        };
        candidate.pillarEval = evaluatePillars(raw.formula, pillarTargets);
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

      if (!phononStable && lambda < 0.8) {
        const failure: FailureRecord = {
          formula: raw.formula,
          strategyId: strategy.id,
          iteration: state.iteration,
          failureType: "phonon-instability",
          failureReason: `Phonon instability detected; lambda=${lambda.toFixed(2)} insufficient to compensate`,
          metrics: { tc, lambda, pressure: 0, hullDistance: 0.02, constraintPenalty: penalty },
          suggestion: "Target stiffer lattice or lighter elements for better phonon stability",
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
          metrics: { tc, lambda, pressure: 0, hullDistance: 0.02, constraintPenalty: penalty },
          suggestion: "Use more commonly synthesized compositions or standard structure types",
          timestamp: Date.now(),
        };
        candidate.failureAnalysis = failure;
        state.failureHistory.push(failure);
        state.totalFailuresAnalyzed++;
        updateKnowledgeBase(state.knowledgeBase, failure, strategy.type);
      } else if (tc < state.targetTc * 0.3 || lambda < 0.5 || penalty > 0.3) {
        const failure = analyzeFailure(
          raw.formula, strategy.id, state.iteration,
          tc, lambda, 0, 0.02, penalty,
          state.targetTc, state.maxPressure,
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

    try {
      candidate.inrField = trainINRFromComposition(raw.formula);
      const centerPred = evaluateINR(candidate.inrField, 0, 0, 0);
      candidate.inrDensity = centerPred.density;
    } catch {}

    candidates.push(candidate);
  }

  state.totalPassed += candidates.filter(c => c.physicsValidated).length;

  const scored = candidates
    .filter(c => c.surrogateScores !== null)
    .sort((a, b) => {
      const aScore = a.surrogateScores!.ensembleTc * 0.7 + a.reward * a.surrogateScores!.ensembleTc * 0.3;
      const bScore = b.surrogateScores!.ensembleTc * 0.7 + b.reward * b.surrogateScores!.ensembleTc * 0.3;
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

  const inverseCandidates: InverseCandidate[] = scored.map(c => ({
    formula: c.formula,
    source: "inverse" as const,
    campaignId: state.id,
    targetDistance: c.targetDistance,
    iteration: state.iteration,
    predictedTc: c.surrogateScores!.ensembleTc,
    predictedLambda: c.surrogateScores!.gnnLambda,
    predictedHull: 0.02,
    predictedPressure: 0,
  }));

  const results = scored.map(c => ({
    formula: c.formula,
    tc: c.surrogateScores!.ensembleTc,
    lambda: c.surrogateScores!.gnnLambda,
    hull: 0.02,
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

    for (const worst of worstStrategies) {
      if (worst.uses >= 3 && worst.fitness < 0.1) {
        const parent = bestStrategies[Math.floor(Math.random() * bestStrategies.length)];
        const evolved = evolveStrategy(parent, state.knowledgeBase, state.strategies);
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
    if (range < 0.01 && state.bestDistance < 0.15) {
      state.status = "converged";
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
