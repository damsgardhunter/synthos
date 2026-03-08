interface GeneratorStats {
  candidatesGenerated: number;
  candidatesPassed: number;
  bestTc: number;
  avgTc: number;
  noveltyScore: number;
  currentWeight: number;
  totalTcSum: number;
}

interface GeneratorEntry {
  name: string;
  defaultWeight: number;
  stats: GeneratorStats;
}

export interface GeneratorBudget {
  allocations: Record<string, number>;
  totalSlots: number;
  weights: Record<string, number>;
}

export interface GeneratorAllocationInfo {
  generators: {
    name: string;
    weight: number;
    stats: GeneratorStats;
  }[];
  totalCandidatesGenerated: number;
  totalCandidatesPassed: number;
  overallPassRate: number;
  rebalanceCount: number;
}

const MINIMUM_WEIGHT_FLOOR = 0.02;

const DEFAULT_ALLOCATIONS: Record<string, number> = {
  structure_diffusion: 0.35,
  rl: 0.25,
  bo_exploration: 0.20,
  motif_diffusion: 0.15,
  random_exploration: 0.05,
};

const generators: Map<string, GeneratorEntry> = new Map();
let rebalanceCount = 0;
let cyclesSinceRebalance = 0;
const REBALANCE_INTERVAL = 3;

function initializeGenerators() {
  if (generators.size > 0) return;
  for (const [name, weight] of Object.entries(DEFAULT_ALLOCATIONS)) {
    generators.set(name, {
      name,
      defaultWeight: weight,
      stats: {
        candidatesGenerated: 0,
        candidatesPassed: 0,
        bestTc: 0,
        avgTc: 0,
        noveltyScore: 0,
        currentWeight: weight,
        totalTcSum: 0,
      },
    });
  }
}

export function registerGenerator(name: string, weight: number) {
  initializeGenerators();
  if (!generators.has(name)) {
    generators.set(name, {
      name,
      defaultWeight: weight,
      stats: {
        candidatesGenerated: 0,
        candidatesPassed: 0,
        bestTc: 0,
        avgTc: 0,
        noveltyScore: 0,
        currentWeight: weight,
        totalTcSum: 0,
      },
    });
  }
}

export function getGeneratorAllocations(): GeneratorAllocationInfo {
  initializeGenerators();
  let totalGen = 0;
  let totalPassed = 0;
  const genList: GeneratorAllocationInfo["generators"] = [];

  generators.forEach((entry) => {
    totalGen += entry.stats.candidatesGenerated;
    totalPassed += entry.stats.candidatesPassed;
    genList.push({
      name: entry.name,
      weight: entry.stats.currentWeight,
      stats: { ...entry.stats },
    });
  });

  return {
    generators: genList,
    totalCandidatesGenerated: totalGen,
    totalCandidatesPassed: totalPassed,
    overallPassRate: totalGen > 0 ? totalPassed / totalGen : 0,
    rebalanceCount,
  };
}

export function allocateBudget(totalSlots: number): GeneratorBudget {
  initializeGenerators();

  const weights: Record<string, number> = {};
  const allocations: Record<string, number> = {};

  generators.forEach((entry, name) => {
    weights[name] = entry.stats.currentWeight;
  });

  let allocated = 0;
  const entries = Array.from(generators.entries());

  for (let i = 0; i < entries.length; i++) {
    const [name, entry] = entries[i];
    if (i === entries.length - 1) {
      allocations[name] = Math.max(1, totalSlots - allocated);
    } else {
      const slots = Math.max(1, Math.round(totalSlots * entry.stats.currentWeight));
      allocations[name] = slots;
      allocated += slots;
    }
  }

  cyclesSinceRebalance++;

  return { allocations, totalSlots, weights };
}

export function recordGeneratorOutcome(
  name: string,
  passed: boolean,
  tc: number,
  novelty: number = 0
) {
  initializeGenerators();
  const entry = generators.get(name);
  if (!entry) return;

  entry.stats.candidatesGenerated++;
  if (passed) {
    entry.stats.candidatesPassed++;
  }
  if (tc > entry.stats.bestTc) {
    entry.stats.bestTc = tc;
  }
  entry.stats.totalTcSum += tc;
  entry.stats.avgTc =
    entry.stats.candidatesGenerated > 0
      ? entry.stats.totalTcSum / entry.stats.candidatesGenerated
      : 0;

  const alpha = 0.1;
  entry.stats.noveltyScore = entry.stats.noveltyScore * (1 - alpha) + novelty * alpha;
}

export function applyTheoryBias(boosts: Record<string, number>) {
  initializeGenerators();

  const maxBoost = Math.max(...Object.values(boosts).map(Math.abs), 0.01);

  for (const [name, boost] of Object.entries(boosts)) {
    const entry = generators.get(name);
    if (!entry) continue;

    const normalizedBoost = boost / maxBoost;
    const adjustment = normalizedBoost * 0.15;
    entry.stats.currentWeight = Math.max(
      MINIMUM_WEIGHT_FLOOR,
      entry.stats.currentWeight * (1 + adjustment)
    );
  }

  let totalWeight = 0;
  generators.forEach(entry => { totalWeight += entry.stats.currentWeight; });
  if (totalWeight > 0) {
    generators.forEach(entry => {
      entry.stats.currentWeight = Math.max(
        MINIMUM_WEIGHT_FLOOR,
        entry.stats.currentWeight / totalWeight
      );
    });
  }

  const summary = Array.from(generators.entries())
    .map(([n, e]) => `${n}=${(e.stats.currentWeight * 100).toFixed(1)}%`)
    .join(", ");
  console.log(`[Generator] Theory bias applied: ${summary}`);
}

export function rebalanceWeights() {
  initializeGenerators();
  if (cyclesSinceRebalance < REBALANCE_INTERVAL) return;

  const yieldScores: Record<string, number> = {};
  let hasActivity = false;

  generators.forEach((entry, name) => {
    const s = entry.stats;
    if (s.candidatesGenerated === 0) {
      yieldScores[name] = 0;
      return;
    }
    hasActivity = true;

    const passRate = s.candidatesPassed / s.candidatesGenerated;
    const tcNorm = Math.min(1, s.bestTc / 200);
    const noveltyNorm = Math.min(1, s.noveltyScore);

    yieldScores[name] = passRate * 0.4 + tcNorm * 0.4 + noveltyNorm * 0.2;
  });

  if (!hasActivity) return;

  const names = Array.from(generators.keys());
  const scores = names.map(n => yieldScores[n]);
  const maxScore = Math.max(...scores, 1e-8);
  const normalizedScores = scores.map(s => s / maxScore);

  const temperature = 2.0;
  const expScores = normalizedScores.map(s => Math.exp(s / temperature));
  const expSum = expScores.reduce((a, b) => a + b, 0);
  const softmaxWeights = expScores.map(e => e / expSum);

  const numGenerators = names.length;
  const totalFloor = MINIMUM_WEIGHT_FLOOR * numGenerators;
  const remainingWeight = 1.0 - totalFloor;

  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    const entry = generators.get(name)!;
    const newWeight = MINIMUM_WEIGHT_FLOOR + remainingWeight * softmaxWeights[i];
    entry.stats.currentWeight = newWeight;
  }

  rebalanceCount++;
  cyclesSinceRebalance = 0;

  const summary = names.map(n => {
    const e = generators.get(n)!;
    return `${n}=${(e.stats.currentWeight * 100).toFixed(1)}%`;
  }).join(", ");
  console.log(`[Generator] Rebalanced weights (#${rebalanceCount}): ${summary}`);
}
