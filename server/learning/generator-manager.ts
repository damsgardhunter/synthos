type EngineTempo = "excited" | "exploring" | "contemplating";
let currentTempo: EngineTempo = "exploring";

export function setEngineTempo(tempo: EngineTempo): void {
  currentTempo = tempo;
}

export function getEngineTempo(): EngineTempo {
  return currentTempo;
}

interface GeneratorStats {
  candidatesGenerated: number;
  candidatesPassed: number;
  bestTc: number;
  avgTc: number;
  noveltyScore: number;
  currentWeight: number;
  totalTcSum: number;
  dftSuccesses: number;
  dftFailures: number;
  dftBestTc: number;
  dftAvgTc: number;
  dftTotalTc: number;
  discoveryRate: number;
  verifiedPassed: number;
  verifiedTotal: number;
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

const MINIMUM_WEIGHT_FLOOR = 0.05;

const DEFAULT_ALLOCATIONS: Record<string, number> = {
  structure_diffusion: 0.25,
  rl: 0.20,
  bo_exploration: 0.15,
  motif_diffusion: 0.15,
  inverse_design: 0.10,
  massive_combinatorial: 0.10,
  random_exploration: 0.05,
};

const generators: Map<string, GeneratorEntry> = new Map();
let rebalanceCount = 0;
let cyclesSinceRebalance = 0;

function getRebalanceInterval(): number {
  switch (currentTempo) {
    case "excited": return 2;
    case "exploring": return 3;
    case "contemplating": return 5;
  }
}

function defaultStats(weight: number): GeneratorStats {
  return {
    candidatesGenerated: 0,
    candidatesPassed: 0,
    bestTc: 0,
    avgTc: 0,
    noveltyScore: 0,
    currentWeight: weight,
    totalTcSum: 0,
    dftSuccesses: 0,
    dftFailures: 0,
    dftBestTc: 0,
    dftAvgTc: 0,
    dftTotalTc: 0,
    discoveryRate: 0,
    verifiedPassed: 0,
    verifiedTotal: 0,
  };
}

function initializeGenerators() {
  if (generators.size > 0) return;
  for (const [name, weight] of Object.entries(DEFAULT_ALLOCATIONS)) {
    generators.set(name, {
      name,
      defaultWeight: weight,
      stats: defaultStats(weight),
    });
  }
}

export function registerGenerator(name: string, weight: number) {
  initializeGenerators();
  if (!generators.has(name)) {
    generators.set(name, {
      name,
      defaultWeight: weight,
      stats: defaultStats(weight),
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
  const entries = Array.from(generators.entries());

  generators.forEach((entry, name) => {
    weights[name] = entry.stats.currentWeight;
  });

  const numGenerators = entries.length;
  const guaranteed = numGenerators;
  const distributable = Math.max(0, totalSlots - guaranteed);

  const idealSlots: { name: string; ideal: number; floor: number; remainder: number }[] = [];
  for (const [name, entry] of entries) {
    const ideal = 1 + distributable * entry.stats.currentWeight;
    const floor = Math.floor(ideal);
    idealSlots.push({ name, ideal, floor, remainder: ideal - floor });
  }

  let allocated = idealSlots.reduce((s, e) => s + e.floor, 0);
  idealSlots.sort((a, b) => b.remainder - a.remainder);

  for (const entry of idealSlots) {
    if (allocated >= totalSlots) break;
    entry.floor++;
    allocated++;
  }

  for (const entry of idealSlots) {
    allocations[entry.name] = entry.floor;
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

  const batchCount = entry.stats.candidatesGenerated;
  const alpha = batchCount < 20 ? 0.2 : batchCount < 100 ? 0.05 : 0.02;
  entry.stats.noveltyScore = entry.stats.noveltyScore * (1 - alpha) + novelty * alpha;
}

export function recordVerificationOutcome(
  name: string,
  passed: boolean,
): void {
  initializeGenerators();
  const entry = generators.get(name);
  if (!entry) return;

  entry.stats.verifiedTotal++;
  if (passed) {
    entry.stats.verifiedPassed++;
  }
}

export function recordDFTOutcome(
  name: string,
  success: boolean,
  tc: number
): void {
  initializeGenerators();
  const entry = generators.get(name);
  if (!entry) return;

  if (success) {
    entry.stats.dftSuccesses++;
    if (tc > entry.stats.dftBestTc) {
      entry.stats.dftBestTc = tc;
    }
    entry.stats.dftTotalTc += tc;
  } else {
    entry.stats.dftFailures++;
  }

  const totalDft = entry.stats.dftSuccesses + entry.stats.dftFailures;
  entry.stats.dftAvgTc = totalDft > 0 ? entry.stats.dftTotalTc / totalDft : 0;
  entry.stats.discoveryRate = totalDft > 0 ? entry.stats.dftSuccesses / totalDft : 0;
}

export function getGeneratorCompetitionStats(): {
  generators: { name: string; weight: number; discoveryRate: number; dftSuccesses: number; dftFailures: number; dftBestTc: number; pipelinePassRate: number; verifiedYield: number; verifiedTotal: number }[];
  totalDFTSuccesses: number;
  totalDFTFailures: number;
  rebalanceCount: number;
} {
  initializeGenerators();
  let totalSuccess = 0;
  let totalFailure = 0;
  const genList: { name: string; weight: number; discoveryRate: number; dftSuccesses: number; dftFailures: number; dftBestTc: number; pipelinePassRate: number; verifiedYield: number; verifiedTotal: number }[] = [];

  generators.forEach((entry) => {
    totalSuccess += entry.stats.dftSuccesses;
    totalFailure += entry.stats.dftFailures;
    genList.push({
      name: entry.name,
      weight: Math.round(entry.stats.currentWeight * 1000) / 1000,
      discoveryRate: Math.round(entry.stats.discoveryRate * 1000) / 1000,
      dftSuccesses: entry.stats.dftSuccesses,
      dftFailures: entry.stats.dftFailures,
      dftBestTc: Math.round(entry.stats.dftBestTc * 10) / 10,
      pipelinePassRate: entry.stats.candidatesGenerated > 0
        ? Math.round((entry.stats.candidatesPassed / entry.stats.candidatesGenerated) * 1000) / 1000
        : 0,
      verifiedYield: entry.stats.verifiedTotal > 0
        ? Math.round((entry.stats.verifiedPassed / entry.stats.verifiedTotal) * 1000) / 1000
        : 0,
      verifiedTotal: entry.stats.verifiedTotal,
    });
  });

  genList.sort((a, b) => b.discoveryRate - a.discoveryRate);

  return {
    generators: genList,
    totalDFTSuccesses: totalSuccess,
    totalDFTFailures: totalFailure,
    rebalanceCount,
  };
}

export function applyTheoryBias(boosts: Record<string, number>) {
  initializeGenerators();

  const maxBoost = Math.max(...Object.values(boosts).map(Math.abs), 0.01);

  for (const [name, boost] of Object.entries(boosts)) {
    const entry = generators.get(name);
    if (!entry) continue;

    const normalizedBoost = boost / maxBoost;
    const adjustment = normalizedBoost * 0.15;
    entry.stats.currentWeight = entry.stats.currentWeight * (1 + adjustment);
  }

  generators.forEach(entry => {
    entry.stats.currentWeight = Math.max(MINIMUM_WEIGHT_FLOOR, entry.stats.currentWeight);
  });

  const numGenerators = generators.size;
  const totalFloor = MINIMUM_WEIGHT_FLOOR * numGenerators;
  let aboveFloorSum = 0;
  generators.forEach(entry => {
    if (entry.stats.currentWeight > MINIMUM_WEIGHT_FLOOR) {
      aboveFloorSum += entry.stats.currentWeight;
    }
  });

  const remainingWeight = 1.0 - totalFloor;
  if (aboveFloorSum > 0 && remainingWeight > 0) {
    generators.forEach(entry => {
      if (entry.stats.currentWeight > MINIMUM_WEIGHT_FLOOR) {
        entry.stats.currentWeight = MINIMUM_WEIGHT_FLOOR +
          remainingWeight * (entry.stats.currentWeight / aboveFloorSum);
      } else {
        entry.stats.currentWeight = MINIMUM_WEIGHT_FLOOR;
      }
    });
  } else {
    const equalWeight = 1.0 / numGenerators;
    generators.forEach(entry => { entry.stats.currentWeight = equalWeight; });
  }

  const summary = Array.from(generators.entries())
    .map(([n, e]) => `${n}=${(e.stats.currentWeight * 100).toFixed(1)}%`)
    .join(", ");
  console.log(`[Generator] Theory bias applied: ${summary}`);
}

export function resetToDefaultWeights(): void {
  initializeGenerators();
  generators.forEach((entry) => {
    entry.stats.currentWeight = entry.defaultWeight;
  });
  const summary = Array.from(generators.entries())
    .map(([n, e]) => `${n}=${(e.stats.currentWeight * 100).toFixed(1)}%`)
    .join(", ");
  console.log(`[Generator] Reset to default weights: ${summary}`);
}

export function rebalanceWeights() {
  initializeGenerators();
  if (cyclesSinceRebalance < getRebalanceInterval()) return;

  const GRACE_PERIOD_CYCLES = 5;
  const yieldScores: Record<string, number> = {};
  let hasActivity = false;
  const activeScores: number[] = [];

  generators.forEach((entry, name) => {
    const s = entry.stats;
    if (s.candidatesGenerated === 0) {
      yieldScores[name] = -1;
      return;
    }
    hasActivity = true;

    const passRate = s.candidatesPassed / s.candidatesGenerated;
    const verifiedYield = s.verifiedTotal > 0 ? s.verifiedPassed / s.verifiedTotal : 0;
    const hasVerifiedData = s.verifiedTotal >= 5;
    const tcNorm = Math.min(1, s.bestTc / 300);
    const noveltyNorm = Math.min(1, s.noveltyScore);
    const dftDiscovery = s.discoveryRate;
    const dftTcNorm = Math.min(1, s.dftBestTc / 300);
    const hasDFTData = (s.dftSuccesses + s.dftFailures) > 0;

    const pipelineScore = hasDFTData
      ? (hasVerifiedData
        ? verifiedYield * 0.15 + tcNorm * 0.1 + noveltyNorm * 0.05
        : passRate * 0.15 + tcNorm * 0.1 + noveltyNorm * 0.05)
      : (hasVerifiedData
        ? verifiedYield * 0.4 + tcNorm * 0.2 + noveltyNorm * 0.1
        : passRate * 0.3 + tcNorm * 0.3 + noveltyNorm * 0.1);
    const dftScore = hasDFTData ? (dftDiscovery * 0.45 + dftTcNorm * 0.25) : 0;
    const score = pipelineScore + dftScore;
    yieldScores[name] = score;
    activeScores.push(score);
  });

  if (!hasActivity) return;

  const medianScore = activeScores.length > 0
    ? activeScores.sort((a, b) => a - b)[Math.floor(activeScores.length / 2)]
    : 0;

  generators.forEach((entry, name) => {
    if (yieldScores[name] < 0) {
      if (rebalanceCount < GRACE_PERIOD_CYCLES) {
        yieldScores[name] = medianScore;
      } else {
        yieldScores[name] = medianScore * 0.5;
      }
    }
  });

  const names = Array.from(generators.keys());
  const scores = names.map(n => yieldScores[n]);
  const maxScore = Math.max(...scores, 1e-8);
  const normalizedScores = scores.map(s => s / maxScore);

  const temperature = Math.max(0.5, 2.0 - rebalanceCount * 0.1);
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
