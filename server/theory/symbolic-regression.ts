type NodeType = "op" | "var" | "const";

type OpName = "+" | "-" | "*" | "/" | "^" | "sqrt" | "exp" | "log";

interface OpNode {
  type: "op";
  op: OpName;
  children: ExprNode[];
}

interface VarNode {
  type: "var";
  name: string;
}

interface ConstNode {
  type: "const";
  value: number;
}

type ExprNode = OpNode | VarNode | ConstNode;

const BINARY_OPS: OpName[] = ["+", "-", "*", "/", "^"];
const UNARY_OPS: OpName[] = ["sqrt", "exp", "log"];
const ALL_OPS: OpName[] = [...BINARY_OPS, ...UNARY_OPS];

function isUnary(op: OpName): boolean {
  return op === "sqrt" || op === "exp" || op === "log";
}

function arity(op: OpName): number {
  return isUnary(op) ? 1 : 2;
}

function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomFloat(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function treeDepth(node: ExprNode): number {
  if (node.type !== "op") return 1;
  return 1 + Math.max(...node.children.map(treeDepth));
}

function treeSize(node: ExprNode): number {
  if (node.type !== "op") return 1;
  return 1 + node.children.reduce((s, c) => s + treeSize(c), 0);
}

function cloneTree(node: ExprNode): ExprNode {
  if (node.type === "const") return { type: "const", value: node.value };
  if (node.type === "var") return { type: "var", name: node.name };
  return { type: "op", op: node.op, children: node.children.map(cloneTree) };
}

function generateRandomTree(vars: string[], maxDepth: number, currentDepth: number = 0): ExprNode {
  if (currentDepth >= maxDepth || (currentDepth > 1 && Math.random() < 0.4)) {
    if (Math.random() < 0.6) {
      return { type: "var", name: randomChoice(vars) };
    }
    return { type: "const", value: roundConst(randomFloat(-5, 5)) };
  }

  const op = randomChoice(ALL_OPS);
  const n = arity(op);
  const children: ExprNode[] = [];
  for (let i = 0; i < n; i++) {
    children.push(generateRandomTree(vars, maxDepth, currentDepth + 1));
  }
  return { type: "op", op, children };
}

function roundConst(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function evaluateNode(node: ExprNode, row: Record<string, number>): number {
  if (node.type === "const") return node.value;
  if (node.type === "var") return row[node.name] ?? 0;

  const vals = node.children.map(c => evaluateNode(c, row));
  switch (node.op) {
    case "+": return vals[0] + vals[1];
    case "-": return vals[0] - vals[1];
    case "*": return vals[0] * vals[1];
    case "/": return Math.abs(vals[1]) < 1e-10 ? 1e6 : vals[0] / vals[1];
    case "^": {
      const base = vals[0];
      const exp = vals[1];
      if (Math.abs(exp) > 5) return 1e6;
      if (base < 0 && Math.abs(exp - Math.round(exp)) > 1e-6) return 1e6;
      const result = Math.pow(Math.abs(base) + 1e-10, exp);
      return base < 0 && Math.round(exp) % 2 !== 0 ? -result : result;
    }
    case "sqrt": return vals[0] >= 0 ? Math.sqrt(vals[0]) : 1e6;
    case "exp": return Math.abs(vals[0]) > 50 ? 1e6 : Math.exp(vals[0]);
    case "log": return vals[0] > 1e-10 ? Math.log(vals[0]) : -1e6;
  }
}

function nodeToString(node: ExprNode): string {
  if (node.type === "const") return node.value.toString();
  if (node.type === "var") return node.name;

  const childStrs = node.children.map(c => nodeToString(c));
  if (isUnary(node.op)) {
    return `${node.op}(${childStrs[0]})`;
  }
  return `(${childStrs[0]} ${node.op} ${childStrs[1]})`;
}

function collectAllNodes(node: ExprNode, path: number[] = []): { node: ExprNode; path: number[] }[] {
  const result: { node: ExprNode; path: number[] }[] = [{ node, path: [...path] }];
  if (node.type === "op") {
    for (let i = 0; i < node.children.length; i++) {
      result.push(...collectAllNodes(node.children[i], [...path, i]));
    }
  }
  return result;
}

function getNodeAtPath(root: ExprNode, path: number[]): ExprNode {
  let current = root;
  for (const idx of path) {
    if (current.type === "op") current = current.children[idx];
    else return current;
  }
  return current;
}

function setNodeAtPath(root: ExprNode, path: number[], replacement: ExprNode): ExprNode {
  if (path.length === 0) return cloneTree(replacement);
  const clone = cloneTree(root);
  let current = clone;
  for (let i = 0; i < path.length - 1; i++) {
    if (current.type === "op") current = current.children[path[i]];
    else return clone;
  }
  if (current.type === "op") {
    current.children[path[path.length - 1]] = cloneTree(replacement);
  }
  return clone;
}

function crossover(parent1: ExprNode, parent2: ExprNode, vars: string[]): ExprNode {
  const nodes1 = collectAllNodes(parent1);
  const nodes2 = collectAllNodes(parent2);
  if (nodes1.length <= 1 || nodes2.length <= 1) return cloneTree(parent1);

  const point1 = randomChoice(nodes1);
  const point2 = randomChoice(nodes2);

  let child = setNodeAtPath(parent1, point1.path, point2.node);
  if (treeDepth(child) > 8) {
    child = cloneTree(parent1);
  }
  return child;
}

function mutate(tree: ExprNode, vars: string[]): ExprNode {
  const clone = cloneTree(tree);
  const nodes = collectAllNodes(clone);
  if (nodes.length === 0) return clone;

  const target = randomChoice(nodes);
  const r = Math.random();

  if (r < 0.25) {
    const newSubtree = generateRandomTree(vars, 3);
    return setNodeAtPath(clone, target.path, newSubtree);
  } else if (r < 0.5 && target.node.type === "op") {
    const opTarget = target.node as OpNode;
    const currentArity = arity(opTarget.op);
    const candidates = ALL_OPS.filter(o => arity(o) === currentArity && o !== opTarget.op);
    if (candidates.length > 0) {
      opTarget.op = randomChoice(candidates);
    }
    return clone;
  } else if (r < 0.7 && target.node.type === "const") {
    target.node.value = roundConst(target.node.value + randomFloat(-2, 2));
    return clone;
  } else if (r < 0.85 && target.node.type === "var") {
    target.node.name = randomChoice(vars);
    return clone;
  } else {
    if (treeSize(clone) > 3) {
      const leafOrSimple = nodes.filter(n => n.path.length > 0 && (n.node.type !== "op" || treeSize(n.node) <= 3));
      if (leafOrSimple.length > 0) {
        const pruneTarget = randomChoice(leafOrSimple);
        const replacement: ExprNode = Math.random() < 0.5
          ? { type: "var", name: randomChoice(vars) }
          : { type: "const", value: roundConst(randomFloat(-3, 3)) };
        return setNodeAtPath(clone, pruneTarget.path, replacement);
      }
    }
    return clone;
  }
}

function hasNonphysicalExponents(node: ExprNode): boolean {
  if (node.type !== "op") return false;
  const opNode = node as OpNode;
  if (opNode.op === "^" && opNode.children[1].type === "const") {
    if (Math.abs(opNode.children[1].value) > 5) return true;
  }
  return opNode.children.some(hasNonphysicalExponents);
}

function producesNegativeTcScaling(tree: ExprNode, dataset: DataPoint[], target: string): boolean {
  if (dataset.length < 5) return false;
  const sorted = [...dataset].sort((a, b) => (a[target] ?? 0) - (b[target] ?? 0));
  const low = sorted.slice(0, Math.ceil(sorted.length * 0.25));
  const high = sorted.slice(Math.floor(sorted.length * 0.75));

  const avgPredLow = low.reduce((s, r) => s + evaluateNode(tree, r), 0) / low.length;
  const avgPredHigh = high.reduce((s, r) => s + evaluateNode(tree, r), 0) / high.length;

  if (!isFinite(avgPredLow) || !isFinite(avgPredHigh)) return true;
  return avgPredHigh < avgPredLow;
}

function physicsConstraintBonus(tree: ExprNode, vars: string[]): number {
  const str = nodeToString(tree);
  let bonus = 0;

  const physicsTerms = ["DOS_EF", "electron_phonon_lambda", "phonon_log_frequency", "debye_temp", "nesting_score", "pairing_strength"];
  for (const term of physicsTerms) {
    if (str.includes(term)) bonus += 0.01;
  }

  const s = treeSize(tree);
  if (s >= 3 && s <= 15) bonus += 0.02;

  return bonus;
}

type DataPoint = Record<string, number>;

interface FitnessResult {
  r2: number;
  complexity: number;
  physicsBonus: number;
  fitness: number;
  mae: number;
}

function computeFitness(tree: ExprNode, dataset: DataPoint[], target: string, vars: string[]): FitnessResult {
  if (dataset.length === 0) return { r2: 0, complexity: 0, physicsBonus: 0, fitness: -Infinity, mae: Infinity };

  const predictions: number[] = [];
  const actuals: number[] = [];
  let invalidCount = 0;

  for (const row of dataset) {
    const pred = evaluateNode(tree, row);
    const actual = row[target] ?? 0;
    if (!isFinite(pred) || Math.abs(pred) > 1e8) {
      invalidCount++;
      predictions.push(0);
    } else {
      predictions.push(pred);
    }
    actuals.push(actual);
  }

  if (invalidCount > dataset.length * 0.3) {
    return { r2: -1, complexity: treeSize(tree), physicsBonus: 0, fitness: -Infinity, mae: Infinity };
  }

  const meanActual = actuals.reduce((s, v) => s + v, 0) / actuals.length;
  let ssTot = 0;
  let ssRes = 0;
  let maeSum = 0;

  for (let i = 0; i < actuals.length; i++) {
    ssTot += (actuals[i] - meanActual) ** 2;
    ssRes += (actuals[i] - predictions[i]) ** 2;
    maeSum += Math.abs(actuals[i] - predictions[i]);
  }

  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const mae = maeSum / actuals.length;
  const complexity = treeSize(tree);
  const complexityPenalty = 0.005 * complexity;
  const pBonus = physicsConstraintBonus(tree, vars);

  const fitness = r2 - complexityPenalty + pBonus;

  return { r2, complexity, physicsBonus: pBonus, fitness, mae };
}

function computeR2AndMAE(tree: ExprNode, dataset: DataPoint[], target: string): { r2: number; mae: number } {
  if (dataset.length === 0) return { r2: 0, mae: Infinity };

  const predictions: number[] = [];
  const actuals: number[] = [];
  let invalidCount = 0;

  for (const row of dataset) {
    const pred = evaluateNode(tree, row);
    const actual = row[target] ?? 0;
    if (!isFinite(pred) || Math.abs(pred) > 1e8) {
      invalidCount++;
      predictions.push(0);
    } else {
      predictions.push(pred);
    }
    actuals.push(actual);
  }

  if (invalidCount > dataset.length * 0.3) {
    return { r2: -1, mae: Infinity };
  }

  const meanActual = actuals.reduce((s, v) => s + v, 0) / actuals.length;
  let ssTot = 0;
  let ssRes = 0;
  let maeSum = 0;

  for (let i = 0; i < actuals.length; i++) {
    ssTot += (actuals[i] - meanActual) ** 2;
    ssRes += (actuals[i] - predictions[i]) ** 2;
    maeSum += Math.abs(actuals[i] - predictions[i]);
  }

  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const mae = maeSum / actuals.length;
  return { r2, mae };
}

function kFoldCrossValidation(tree: ExprNode, dataset: DataPoint[], target: string, k: number = 5): { avgR2: number; foldR2s: number[] } {
  if (dataset.length < k) return { avgR2: 0, foldR2s: [] };

  const shuffled = [...dataset].sort(() => Math.random() - 0.5);
  const foldSize = Math.floor(shuffled.length / k);
  const foldR2s: number[] = [];

  for (let i = 0; i < k; i++) {
    const valStart = i * foldSize;
    const valEnd = i === k - 1 ? shuffled.length : valStart + foldSize;
    const valFold = shuffled.slice(valStart, valEnd);
    const trainFold = [...shuffled.slice(0, valStart), ...shuffled.slice(valEnd)];

    if (trainFold.length === 0 || valFold.length === 0) continue;

    const { r2 } = computeR2AndMAE(tree, valFold, target);
    foldR2s.push(r2);
  }

  const avgR2 = foldR2s.length > 0 ? foldR2s.reduce((s, v) => s + v, 0) / foldR2s.length : 0;
  return { avgR2, foldR2s };
}

function computeCVFitness(tree: ExprNode, dataset: DataPoint[], target: string, vars: string[]): FitnessResult & { cvScore: number } {
  const baseFitness = computeFitness(tree, dataset, target, vars);
  const { avgR2 } = kFoldCrossValidation(tree, dataset, target, 5);
  const complexity = treeSize(tree);
  const complexityPenalty = 0.005 * complexity;
  const pBonus = physicsConstraintBonus(tree, vars);
  const cvFitness = avgR2 - complexityPenalty + pBonus;

  return {
    ...baseFitness,
    fitness: cvFitness,
    cvScore: avgR2,
  };
}

export interface PlausibilityDetails {
  dimensionalConsistency: number;
  monotonicityScore: number;
  knownRelationshipBonus: number;
  physicsTermCount: number;
  complexityScore: number;
  exponentPenalty: number;
  scalingPenalty: number;
}

export interface TheoryCandidate {
  equation: string;
  tree: ExprNode;
  target: string;
  r2: number;
  mae: number;
  complexity: number;
  fitness: number;
  plausibility: number;
  discoveredAt: number;
  generation: number;
  validationR2: number;
  validationMAE: number;
  cvScore: number;
  overfitRatio: number;
  isOverfit: boolean;
  plausibilityDetails: PlausibilityDetails;
}

const theoryStore: TheoryCandidate[] = [];

export const theoryKnowledgeBase = {
  get theories(): TheoryCandidate[] {
    return [...theoryStore];
  },
  add(theory: TheoryCandidate): void {
    const isDuplicate = theoryStore.some(t =>
      t.equation === theory.equation && t.target === theory.target
    );
    if (!isDuplicate) {
      theoryStore.push(theory);
      if (theoryStore.length > 200) {
        theoryStore.sort((a, b) => b.fitness - a.fitness);
        theoryStore.length = 150;
      }
    }
  },
  clear(): void {
    theoryStore.length = 0;
  },
  getByTarget(target: string): TheoryCandidate[] {
    return theoryStore.filter(t => t.target === target).sort((a, b) => b.fitness - a.fitness);
  },
};

export function getDiscoveredTheories(): TheoryCandidate[] {
  return [...theoryStore].sort((a, b) => b.fitness - a.fitness);
}

export function getValidationStats(): {
  totalTheories: number;
  avgCvScore: number;
  avgOverfitRatio: number;
  overfitCount: number;
  avgValidationR2: number;
  avgValidationMAE: number;
  theoriesByTarget: Record<string, { count: number; avgCvScore: number; overfitCount: number }>;
} {
  const theories = theoryStore;
  if (theories.length === 0) {
    return {
      totalTheories: 0,
      avgCvScore: 0,
      avgOverfitRatio: 0,
      overfitCount: 0,
      avgValidationR2: 0,
      avgValidationMAE: 0,
      theoriesByTarget: {},
    };
  }

  const totalTheories = theories.length;
  const avgCvScore = theories.reduce((s, t) => s + t.cvScore, 0) / totalTheories;
  const avgOverfitRatio = theories.reduce((s, t) => s + t.overfitRatio, 0) / totalTheories;
  const overfitCount = theories.filter(t => t.isOverfit).length;
  const avgValidationR2 = theories.reduce((s, t) => s + t.validationR2, 0) / totalTheories;
  const validTheories = theories.filter(t => isFinite(t.validationMAE));
  const avgValidationMAE = validTheories.length > 0 ? validTheories.reduce((s, t) => s + t.validationMAE, 0) / validTheories.length : Infinity;

  const targetMap: Record<string, TheoryCandidate[]> = {};
  for (const t of theories) {
    if (!targetMap[t.target]) targetMap[t.target] = [];
    targetMap[t.target].push(t);
  }

  const theoriesByTarget: Record<string, { count: number; avgCvScore: number; overfitCount: number }> = {};
  for (const [target, tList] of Object.entries(targetMap)) {
    theoriesByTarget[target] = {
      count: tList.length,
      avgCvScore: tList.reduce((s, t) => s + t.cvScore, 0) / tList.length,
      overfitCount: tList.filter(t => t.isOverfit).length,
    };
  }

  return { totalTheories, avgCvScore, avgOverfitRatio, overfitCount, avgValidationR2, avgValidationMAE, theoriesByTarget };
}

function checkDimensionalConsistency(tree: ExprNode): number {
  const str = nodeToString(tree);

  const dimensionGroups: Record<string, string[]> = {
    energy: ["DOS_EF", "bandwidth", "debye_temp", "phonon_log_frequency"],
    coupling: ["electron_phonon_lambda", "pairing_strength", "nesting_score", "correlation_strength"],
    structural: ["atomic_mass", "density", "volume"],
  };

  const usedGroups = new Set<string>();
  for (const [group, terms] of Object.entries(dimensionGroups)) {
    for (const term of terms) {
      if (str.includes(term)) {
        usedGroups.add(group);
        break;
      }
    }
  }

  let score = 1.0;

  if (tree.type === "op" && tree.op === "+") {
    const leftVars = collectVarNames(tree.children[0]);
    const rightVars = collectVarNames(tree.children[1]);

    const leftGroups = new Set<string>();
    const rightGroups = new Set<string>();

    for (const [group, terms] of Object.entries(dimensionGroups)) {
      for (const v of leftVars) {
        if (terms.includes(v)) leftGroups.add(group);
      }
      for (const v of rightVars) {
        if (terms.includes(v)) rightGroups.add(group);
      }
    }

    if (leftGroups.size > 0 && rightGroups.size > 0) {
      const leftArr = Array.from(leftGroups);
      const rightArr = Array.from(rightGroups);
      const hasOverlap = leftArr.some(g => rightGroups.has(g));
      if (!hasOverlap && leftGroups.size > 0 && rightGroups.size > 0) {
        score -= 0.3;
      }
    }
  }

  return Math.max(0, Math.min(1, score));
}

function collectVarNames(node: ExprNode): string[] {
  if (node.type === "var") return [node.name];
  if (node.type === "const") return [];
  return node.children.flatMap(collectVarNames);
}

function checkMonotonicity(tree: ExprNode, dataset: DataPoint[], target: string): number {
  if (dataset.length < 5) return 0.5;

  const lambdaKey = ["electron_phonon_lambda", "pairing_strength"].find(k =>
    dataset.some(r => r[k] !== undefined && r[k] > 0)
  );
  if (!lambdaKey) return 0.5;

  const sorted = [...dataset]
    .filter(r => r[lambdaKey] !== undefined && r[lambdaKey] > 0)
    .sort((a, b) => (a[lambdaKey] ?? 0) - (b[lambdaKey] ?? 0));

  if (sorted.length < 5) return 0.5;

  const bucketSize = Math.max(1, Math.floor(sorted.length / 5));
  const bucketAvgs: number[] = [];
  for (let i = 0; i < 5; i++) {
    const start = i * bucketSize;
    const end = Math.min(start + bucketSize, sorted.length);
    const bucket = sorted.slice(start, end);
    if (bucket.length === 0) continue;
    const avg = bucket.reduce((s, r) => s + evaluateNode(tree, r), 0) / bucket.length;
    if (!isFinite(avg)) return 0;
    bucketAvgs.push(avg);
  }

  let increasing = 0;
  for (let i = 1; i < bucketAvgs.length; i++) {
    if (bucketAvgs[i] >= bucketAvgs[i - 1]) increasing++;
  }

  return increasing / (bucketAvgs.length - 1);
}

function checkKnownRelationshipBonus(tree: ExprNode): number {
  const str = nodeToString(tree);
  let bonus = 0;

  if (str.includes("electron_phonon_lambda") && str.includes("phonon_log_frequency")) {
    bonus += 0.1;
  }
  if (str.includes("electron_phonon_lambda") && str.includes("debye_temp")) {
    bonus += 0.08;
  }

  if (str.includes("exp") && str.includes("electron_phonon_lambda")) {
    bonus += 0.05;
  }

  if (str.includes("DOS_EF") && (str.includes("electron_phonon_lambda") || str.includes("pairing_strength"))) {
    bonus += 0.05;
  }

  return Math.min(0.25, bonus);
}

function assessPlausibility(tree: ExprNode, r2: number, dataset: DataPoint[], target: string): { score: number; details: PlausibilityDetails } {
  const dimensionalConsistency = checkDimensionalConsistency(tree);
  const monotonicityScore = checkMonotonicity(tree, dataset, target);
  const knownRelationshipBonus = checkKnownRelationshipBonus(tree);

  const str = nodeToString(tree);
  const physicsRelevant = ["DOS_EF", "electron_phonon_lambda", "phonon_log_frequency", "debye_temp", "pairing_strength", "nesting_score", "bandwidth", "correlation_strength"];
  const physicsTermCount = physicsRelevant.filter(t => str.includes(t)).length;

  const size = treeSize(tree);
  let complexityScore = 0;
  if (size >= 3 && size <= 12) complexityScore = 0.2;
  else if (size >= 2 && size <= 20) complexityScore = 0.1;

  const exponentPenalty = hasNonphysicalExponents(tree) ? 0.3 : 0;
  const scalingPenalty = producesNegativeTcScaling(tree, dataset, target) ? 0.3 : 0;

  let score = 0;
  if (r2 > 0.5) score += 0.2;
  if (r2 > 0.7) score += 0.15;
  if (r2 > 0.85) score += 0.1;

  score += complexityScore;
  score += dimensionalConsistency * 0.1;
  score += monotonicityScore * 0.15;
  score += knownRelationshipBonus;
  score += Math.min(0.1, physicsTermCount * 0.03);
  score -= exponentPenalty;
  score -= scalingPenalty;

  const details: PlausibilityDetails = {
    dimensionalConsistency,
    monotonicityScore,
    knownRelationshipBonus,
    physicsTermCount,
    complexityScore,
    exponentPenalty,
    scalingPenalty,
  };

  return { score: Math.max(0, Math.min(1, score)), details };
}

interface SymbolicRegressionConfig {
  populationSize: number;
  generations: number;
  maxTreeDepth: number;
  tournamentSize: number;
  crossoverRate: number;
  mutationRate: number;
  eliteCount: number;
}

const DEFAULT_CONFIG: SymbolicRegressionConfig = {
  populationSize: 200,
  generations: 50,
  maxTreeDepth: 6,
  tournamentSize: 5,
  crossoverRate: 0.7,
  mutationRate: 0.25,
  eliteCount: 10,
};

function tournamentSelect(
  population: ExprNode[],
  fitnesses: (FitnessResult & { cvScore: number })[],
  tournamentSize: number,
): number {
  let bestIdx = Math.floor(Math.random() * population.length);
  let bestFit = fitnesses[bestIdx].fitness;

  for (let i = 1; i < tournamentSize; i++) {
    const idx = Math.floor(Math.random() * population.length);
    if (fitnesses[idx].fitness > bestFit) {
      bestIdx = idx;
      bestFit = fitnesses[idx].fitness;
    }
  }
  return bestIdx;
}

function splitDataset(dataset: DataPoint[], holdoutFraction: number = 0.2): { train: DataPoint[]; holdout: DataPoint[] } {
  const shuffled = [...dataset].sort(() => Math.random() - 0.5);
  const holdoutSize = Math.max(1, Math.floor(shuffled.length * holdoutFraction));
  return {
    holdout: shuffled.slice(0, holdoutSize),
    train: shuffled.slice(holdoutSize),
  };
}

export function runSymbolicRegression(
  dataset: DataPoint[],
  target: string,
  config?: Partial<SymbolicRegressionConfig>,
): TheoryCandidate[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  if (dataset.length < 5) return [];

  const allKeys = new Set<string>();
  for (const row of dataset) {
    for (const key of Object.keys(row)) {
      if (key !== target) allKeys.add(key);
    }
  }
  const vars = Array.from(allKeys);
  if (vars.length === 0) return [];

  const { train: trainData, holdout: holdoutData } = splitDataset(dataset, 0.2);

  let population: ExprNode[] = [];
  for (let i = 0; i < cfg.populationSize; i++) {
    const depth = 2 + Math.floor(Math.random() * (cfg.maxTreeDepth - 1));
    population.push(generateRandomTree(vars, depth));
  }

  let bestOverall: { tree: ExprNode; fit: FitnessResult & { cvScore: number }; gen: number } | null = null;
  const discovered: TheoryCandidate[] = [];

  for (let gen = 0; gen < cfg.generations; gen++) {
    const fitnesses = population.map(tree => computeCVFitness(tree, trainData, target, vars));

    const indexed = fitnesses.map((f, i) => ({ f, i }));
    indexed.sort((a, b) => b.f.fitness - a.f.fitness);

    const best = indexed[0];
    if (!bestOverall || best.f.fitness > bestOverall.fit.fitness) {
      bestOverall = { tree: cloneTree(population[best.i]), fit: best.f, gen };
    }

    if (gen % 10 === 9 || gen === cfg.generations - 1) {
      for (let k = 0; k < Math.min(3, indexed.length); k++) {
        const idx = indexed[k].i;
        const tree = population[idx];
        const fit = fitnesses[idx];

        if (fit.r2 < 0.2) continue;
        if (hasNonphysicalExponents(tree)) continue;
        if (producesNegativeTcScaling(tree, trainData, target)) continue;

        const { score: plausibility, details: plausibilityDetails } = assessPlausibility(tree, fit.r2, trainData, target);
        if (plausibility < 0.2) continue;

        const holdoutResult = holdoutData.length >= 3 ? computeR2AndMAE(tree, holdoutData, target) : { r2: fit.cvScore, mae: fit.mae };
        const trainR2 = fit.r2;
        const overfitRatio = holdoutResult.r2 > 0.05 ? Math.min(5, trainR2 / holdoutResult.r2) : (trainR2 > 0.3 ? 3 : 1);
        const isOverfit = overfitRatio > 1.5 || (fit.cvScore < 0.3 && trainR2 > 0.5);

        const equation = nodeToString(tree);
        const candidate: TheoryCandidate = {
          equation,
          tree: cloneTree(tree),
          target,
          r2: trainR2,
          mae: fit.mae,
          complexity: fit.complexity,
          fitness: fit.fitness,
          plausibility,
          discoveredAt: Date.now(),
          generation: gen,
          validationR2: holdoutResult.r2,
          validationMAE: holdoutResult.mae,
          cvScore: fit.cvScore,
          overfitRatio,
          isOverfit,
          plausibilityDetails,
        };

        const isDup = discovered.some(d => d.equation === equation);
        if (!isDup) {
          discovered.push(candidate);
          theoryKnowledgeBase.add(candidate);
        }
      }
    }

    const nextPop: ExprNode[] = [];

    for (let k = 0; k < cfg.eliteCount && k < indexed.length; k++) {
      nextPop.push(cloneTree(population[indexed[k].i]));
    }

    while (nextPop.length < cfg.populationSize) {
      const r = Math.random();
      if (r < cfg.crossoverRate) {
        const p1 = tournamentSelect(population, fitnesses, cfg.tournamentSize);
        const p2 = tournamentSelect(population, fitnesses, cfg.tournamentSize);
        const child = crossover(population[p1], population[p2], vars);
        nextPop.push(child);
      } else if (r < cfg.crossoverRate + cfg.mutationRate) {
        const p = tournamentSelect(population, fitnesses, cfg.tournamentSize);
        nextPop.push(mutate(population[p], vars));
      } else {
        const depth = 2 + Math.floor(Math.random() * (cfg.maxTreeDepth - 1));
        nextPop.push(generateRandomTree(vars, depth));
      }
    }

    population = nextPop;
  }

  if (bestOverall && !discovered.some(d => d.equation === nodeToString(bestOverall!.tree))) {
    const { score: plausibility, details: plausibilityDetails } = assessPlausibility(bestOverall.tree, bestOverall.fit.r2, trainData, target);
    if (bestOverall.fit.r2 >= 0.1 && plausibility >= 0.1) {
      const holdoutResult = holdoutData.length >= 3 ? computeR2AndMAE(bestOverall.tree, holdoutData, target) : { r2: bestOverall.fit.cvScore, mae: bestOverall.fit.mae };
      const trainR2 = bestOverall.fit.r2;
      const overfitRatio = holdoutResult.r2 > 0.05 ? Math.min(5, trainR2 / holdoutResult.r2) : (trainR2 > 0.3 ? 3 : 1);
      const isOverfit = overfitRatio > 1.5 || (bestOverall.fit.cvScore < 0.3 && trainR2 > 0.5);

      const candidate: TheoryCandidate = {
        equation: nodeToString(bestOverall.tree),
        tree: cloneTree(bestOverall.tree),
        target,
        r2: trainR2,
        mae: bestOverall.fit.mae,
        complexity: bestOverall.fit.complexity,
        fitness: bestOverall.fit.fitness,
        plausibility,
        discoveredAt: Date.now(),
        generation: bestOverall.gen,
        validationR2: holdoutResult.r2,
        validationMAE: holdoutResult.mae,
        cvScore: bestOverall.fit.cvScore,
        overfitRatio,
        isOverfit,
        plausibilityDetails,
      };
      discovered.push(candidate);
      theoryKnowledgeBase.add(candidate);
    }
  }

  return discovered.sort((a, b) => b.fitness - a.fitness);
}
