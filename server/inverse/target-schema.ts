export interface TargetProperties {
  targetTc: number;
  maxPressure: number;
  minLambda: number;
  maxHullDistance: number;
  metallicRequired: boolean;
  phononStable: boolean;
  preferredPrototypes?: string[];
  preferredElements?: string[];
  excludeElements?: string[];
}

export interface InverseCandidate {
  formula: string;
  source: "inverse";
  derivation?: "generated" | "refinement" | "differentiable-opt";
  campaignId: string;
  targetDistance: number;
  iteration: number;
  predictedTc?: number;
  predictedLambda?: number;
  predictedHull?: number;
  predictedPressure?: number;
  prototype?: string;
  reward?: number;
  synthesisVector?: {
    temperature: number;
    pressure: number;
    coolingRate: number;
    annealTime: number;
    strain: number;
  };
  synthesisPath?: {
    steps: { order: number; method: string; temperature: number; pressure: number }[];
    complexity: number;
    feasibility: number;
  };
}

export interface InverseLearningState {
  elementSuccessMatrix: Map<string, { totalReward: number; count: number; avgDistance: number }>;
  pairSuccessMatrix: Map<string, { totalReward: number; count: number; avgDistance: number }>;
  prototypeSuccessMatrix: Map<string, { totalReward: number; count: number; avgDistance: number }>;
  stoichiometryBias: Map<string, number>;
  bestCandidates: InverseCandidate[];
  bestDistance: number;
  iterationsRun: number;
  convergenceHistory: number[];
}

export interface CampaignStats {
  id: string;
  target: TargetProperties;
  status: string;
  cyclesRun: number;
  bestTcAchieved: number;
  bestDistance: number;
  candidatesGenerated: number;
  candidatesPassedPipeline: number;
  topCandidates: { formula: string; tc: number; distance: number }[];
  convergenceHistory: number[];
}

export interface CompositionBias {
  elementWeights: Map<string, number>;
  prototypeWeights: Map<string, number>;
  stoichiometryPatterns: { pattern: string; weight: number }[];
}

export function computeTargetDistance(
  target: TargetProperties,
  predicted: { tc: number; lambda: number; hull: number; pressure: number }
): number {
  const tcWeight = 0.50;
  const lambdaWeight = 0.20;
  const hullWeight = 0.15;
  const pressureWeight = 0.15;

  const tcDist = predicted.tc < target.targetTc
    ? (target.targetTc - predicted.tc) / Math.max(target.targetTc, 1)
    : 0;

  const lambdaDist = predicted.lambda < target.minLambda
    ? (target.minLambda - predicted.lambda) / target.minLambda
    : 0;

  const hullDist = predicted.hull > target.maxHullDistance
    ? Math.pow((predicted.hull - target.maxHullDistance) / 0.1, 2)
    : 0;

  const pressureDist = predicted.pressure > target.maxPressure
    ? Math.pow((predicted.pressure - target.maxPressure) / 100, 2)
    : 0;

  return tcWeight * tcDist + lambdaWeight * lambdaDist + hullWeight * hullDist + pressureWeight * pressureDist;
}

export function computeReward(distance: number, scale: number = 0.3): number {
  const LINEAR_FLOOR = 0.05;
  const BLEND_THRESHOLD = 1.0;

  const expReward = Math.exp(-distance / scale);

  if (distance <= BLEND_THRESHOLD) {
    return expReward;
  }

  const linearReward = LINEAR_FLOOR + (1 - LINEAR_FLOOR) * Math.max(0, 1 - distance / 5.0);
  const blendFactor = Math.min(1, (distance - BLEND_THRESHOLD) / 1.0);
  return expReward * (1 - blendFactor) + linearReward * blendFactor;
}
