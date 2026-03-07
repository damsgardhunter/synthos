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
  campaignId: string;
  targetDistance: number;
  iteration: number;
  predictedTc?: number;
  predictedLambda?: number;
  predictedHull?: number;
  predictedPressure?: number;
  prototype?: string;
  reward?: number;
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

  const tcDist = Math.abs(target.targetTc - predicted.tc) / Math.max(target.targetTc, 1);
  const lambdaDist = predicted.lambda < target.minLambda
    ? (target.minLambda - predicted.lambda) / target.minLambda
    : 0;
  const hullDist = predicted.hull > target.maxHullDistance
    ? (predicted.hull - target.maxHullDistance) / 0.3
    : 0;
  const pressureDist = predicted.pressure > target.maxPressure
    ? (predicted.pressure - target.maxPressure) / 200
    : 0;

  return tcWeight * tcDist + lambdaWeight * lambdaDist + hullWeight * hullDist + pressureWeight * pressureDist;
}

export function computeReward(distance: number, scale: number = 0.3): number {
  return Math.exp(-distance / scale);
}
