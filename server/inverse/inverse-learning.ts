import {
  InverseCandidate,
  InverseLearningState,
  TargetProperties,
  CompositionBias,
  computeTargetDistance,
  computeReward,
} from "./target-schema";

export type { InverseLearningState } from "./target-schema";

export function createInitialLearningState(): InverseLearningState {
  return {
    elementSuccessMatrix: new Map(),
    pairSuccessMatrix: new Map(),
    prototypeSuccessMatrix: new Map(),
    stoichiometryBias: new Map(),
    bestCandidates: [],
    bestDistance: 1.0,
    iterationsRun: 0,
    convergenceHistory: [],
  };
}

function parseFormulaElements(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  return Array.from(new Set(matches || []));
}

function makePairKey(a: string, b: string): string {
  return [a, b].sort().join("-");
}

export function updateLearningState(
  state: InverseLearningState,
  candidates: InverseCandidate[],
  results: {
    formula: string;
    tc: number;
    lambda: number;
    hull: number;
    pressure: number;
    passedPipeline: boolean;
  }[],
  target: TargetProperties
): InverseLearningState {
  const resultMap = new Map(results.map(r => [r.formula, r]));

  for (const candidate of candidates) {
    const result = resultMap.get(candidate.formula);
    if (!result) continue;

    const distance = candidate.targetDistance > 0
      ? candidate.targetDistance
      : computeTargetDistance(target, {
          tc: result.tc,
          lambda: result.lambda,
          hull: result.hull,
          pressure: result.pressure,
        });

    let reward = computeReward(distance);

    if (candidate.synthesisVector) {
      const sv = candidate.synthesisVector;
      if (sv.temperature > 3000) reward *= 0.7;
      else if (sv.temperature > 2000) reward *= 0.85;
      if (sv.pressure > 300) reward *= 0.7;
      else if (sv.pressure > 150) reward *= 0.85;
    }

    candidate.targetDistance = distance;
    candidate.predictedTc = result.tc;
    candidate.predictedLambda = result.lambda;
    candidate.predictedHull = result.hull;
    candidate.predictedPressure = result.pressure;
    candidate.reward = reward;

    const elements = parseFormulaElements(candidate.formula);

    for (const el of elements) {
      const existing = state.elementSuccessMatrix.get(el) || { totalReward: 0, count: 0, avgDistance: 1.0 };
      existing.totalReward += reward;
      existing.count += 1;
      existing.avgDistance = (existing.avgDistance * (existing.count - 1) + distance) / existing.count;
      state.elementSuccessMatrix.set(el, existing);
    }

    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const key = makePairKey(elements[i], elements[j]);
        const existing = state.pairSuccessMatrix.get(key) || { totalReward: 0, count: 0, avgDistance: 1.0 };
        existing.totalReward += reward;
        existing.count += 1;
        existing.avgDistance = (existing.avgDistance * (existing.count - 1) + distance) / existing.count;
        state.pairSuccessMatrix.set(key, existing);
      }
    }

    if (candidate.prototype) {
      const existing = state.prototypeSuccessMatrix.get(candidate.prototype) || { totalReward: 0, count: 0, avgDistance: 1.0 };
      existing.totalReward += reward;
      existing.count += 1;
      existing.avgDistance = (existing.avgDistance * (existing.count - 1) + distance) / existing.count;
      state.prototypeSuccessMatrix.set(candidate.prototype, existing);
    }

    if (distance < state.bestDistance) {
      state.bestDistance = distance;
    }

    const shouldKeep = state.bestCandidates.length < 20 ||
      distance < state.bestCandidates[state.bestCandidates.length - 1].targetDistance;

    if (shouldKeep) {
      state.bestCandidates.push(candidate);
      state.bestCandidates.sort((a, b) => a.targetDistance - b.targetDistance);
      if (state.bestCandidates.length > 20) {
        state.bestCandidates = state.bestCandidates.slice(0, 20);
      }
    }
  }

  state.iterationsRun += 1;
  state.convergenceHistory.push(state.bestDistance);
  if (state.convergenceHistory.length > 200) {
    state.convergenceHistory = state.convergenceHistory.slice(-100);
  }

  return state;
}

export function deriveCompositionBias(
  state: InverseLearningState,
  baseBias: CompositionBias
): CompositionBias {
  const elementWeights = new Map(baseBias.elementWeights);
  const prototypeWeights = new Map(baseBias.prototypeWeights);
  const stoichiometryPatterns = [...baseBias.stoichiometryPatterns];

  for (const [el, stats] of state.elementSuccessMatrix) {
    if (stats.count < 2) continue;
    const avgReward = stats.totalReward / stats.count;
    const baseW = elementWeights.get(el) ?? 1.0;
    elementWeights.set(el, baseW * (0.7 + 0.6 * avgReward));
  }

  for (const [pair, stats] of state.pairSuccessMatrix) {
    if (stats.count < 3) continue;
    const avgReward = stats.totalReward / stats.count;
    if (avgReward > 0.5) {
      const [a, b] = pair.split("-");
      elementWeights.set(a, (elementWeights.get(a) ?? 1.0) * (1.0 + avgReward * 0.3));
      elementWeights.set(b, (elementWeights.get(b) ?? 1.0) * (1.0 + avgReward * 0.3));
    }
  }

  for (const [proto, stats] of state.prototypeSuccessMatrix) {
    if (stats.count < 2) continue;
    const avgReward = stats.totalReward / stats.count;
    const baseW = prototypeWeights.get(proto) ?? 1.0;
    prototypeWeights.set(proto, baseW * (0.7 + 0.6 * avgReward));
  }

  for (const [stoich, bias] of state.stoichiometryBias) {
    const existing = stoichiometryPatterns.find(p => p.pattern === stoich);
    if (existing) {
      existing.weight = Math.max(0.1, existing.weight * (0.7 + 0.6 * Math.max(0, bias)));
    }
  }

  if (state.iterationsRun > 5 && state.convergenceHistory.length >= 3) {
    const last3 = state.convergenceHistory.slice(-3);
    const isStagnant = Math.abs(last3[0] - last3[2]) < 0.01;
    if (isStagnant) {
      for (const [el, w] of elementWeights) {
        elementWeights.set(el, w * (0.8 + Math.random() * 0.4));
      }
      for (const [proto, w] of prototypeWeights) {
        prototypeWeights.set(proto, w * (0.8 + Math.random() * 0.4));
      }
      for (const p of stoichiometryPatterns) {
        p.weight = p.weight * (0.8 + Math.random() * 0.4);
      }
    }
  }

  return { elementWeights, prototypeWeights, stoichiometryPatterns };
}

export function serializeLearningState(state: InverseLearningState): any {
  return {
    elementSuccessMatrix: Object.fromEntries(state.elementSuccessMatrix),
    pairSuccessMatrix: Object.fromEntries(state.pairSuccessMatrix),
    prototypeSuccessMatrix: Object.fromEntries(state.prototypeSuccessMatrix),
    stoichiometryBias: Object.fromEntries(state.stoichiometryBias),
    bestCandidates: state.bestCandidates.slice(0, 10).map(c => ({
      formula: c.formula,
      targetDistance: c.targetDistance,
      predictedTc: c.predictedTc,
      predictedLambda: c.predictedLambda,
      prototype: c.prototype,
      reward: c.reward,
    })),
    bestDistance: state.bestDistance,
    iterationsRun: state.iterationsRun,
    convergenceHistory: state.convergenceHistory.slice(-50),
  };
}

export function deserializeLearningState(data: any): InverseLearningState {
  if (!data) return createInitialLearningState();

  return {
    elementSuccessMatrix: new Map(Object.entries(data.elementSuccessMatrix || {})),
    pairSuccessMatrix: new Map(Object.entries(data.pairSuccessMatrix || {})),
    prototypeSuccessMatrix: new Map(Object.entries(data.prototypeSuccessMatrix || {})),
    stoichiometryBias: new Map(Object.entries(data.stoichiometryBias || {})),
    bestCandidates: (data.bestCandidates || []).map((c: any) => ({
      formula: c.formula,
      source: "inverse" as const,
      campaignId: "",
      targetDistance: c.targetDistance ?? 1.0,
      iteration: 0,
      predictedTc: c.predictedTc,
      predictedLambda: c.predictedLambda,
      prototype: c.prototype,
      reward: c.reward,
    })),
    bestDistance: data.bestDistance ?? 1.0,
    iterationsRun: data.iterationsRun ?? 0,
    convergenceHistory: data.convergenceHistory ?? [],
  };
}
