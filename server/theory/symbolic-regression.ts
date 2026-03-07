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

function assessPlausibility(tree: ExprNode, r2: number, dataset: DataPoint[], target: string): number {
  let score = 0;

  if (r2 > 0.5) score += 0.3;
  if (r2 > 0.7) score += 0.2;
  if (r2 > 0.85) score += 0.1;

  const size = treeSize(tree);
  if (size >= 3 && size <= 12) score += 0.2;
  else if (size >= 2 && size <= 20) score += 0.1;

  if (hasNonphysicalExponents(tree)) score -= 0.3;
  if (producesNegativeTcScaling(tree, dataset, target)) score -= 0.3;

  const str = nodeToString(tree);
  const physicsRelevant = ["DOS_EF", "electron_phonon_lambda", "phonon_log_frequency", "debye_temp", "pairing_strength", "nesting_score", "bandwidth", "correlation_strength"];
  const usedPhysics = physicsRelevant.filter(t => str.includes(t));
  score += Math.min(0.2, usedPhysics.length * 0.05);

  return Math.max(0, Math.min(1, score));
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
  fitnesses: FitnessResult[],
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

  let population: ExprNode[] = [];
  for (let i = 0; i < cfg.populationSize; i++) {
    const depth = 2 + Math.floor(Math.random() * (cfg.maxTreeDepth - 1));
    population.push(generateRandomTree(vars, depth));
  }

  let bestOverall: { tree: ExprNode; fit: FitnessResult; gen: number } | null = null;
  const discovered: TheoryCandidate[] = [];

  for (let gen = 0; gen < cfg.generations; gen++) {
    const fitnesses = population.map(tree => computeFitness(tree, dataset, target, vars));

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
        if (producesNegativeTcScaling(tree, dataset, target)) continue;

        const plausibility = assessPlausibility(tree, fit.r2, dataset, target);
        if (plausibility < 0.2) continue;

        const equation = nodeToString(tree);
        const candidate: TheoryCandidate = {
          equation,
          tree: cloneTree(tree),
          target,
          r2: fit.r2,
          mae: fit.mae,
          complexity: fit.complexity,
          fitness: fit.fitness,
          plausibility,
          discoveredAt: Date.now(),
          generation: gen,
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
    const plausibility = assessPlausibility(bestOverall.tree, bestOverall.fit.r2, dataset, target);
    if (bestOverall.fit.r2 >= 0.1 && plausibility >= 0.1) {
      const candidate: TheoryCandidate = {
        equation: nodeToString(bestOverall.tree),
        tree: cloneTree(bestOverall.tree),
        target,
        r2: bestOverall.fit.r2,
        mae: bestOverall.fit.mae,
        complexity: bestOverall.fit.complexity,
        fitness: bestOverall.fit.fitness,
        plausibility,
        discoveredAt: Date.now(),
        generation: bestOverall.gen,
      };
      discovered.push(candidate);
      theoryKnowledgeBase.add(candidate);
    }
  }

  return discovered.sort((a, b) => b.fitness - a.fitness);
}
