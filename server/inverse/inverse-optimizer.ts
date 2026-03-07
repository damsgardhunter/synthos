import { TargetProperties, InverseCandidate, CampaignStats, computeTargetDistance } from "./target-schema";
import { generateInverseCandidates, refineCandidate, createInitialBias } from "./inverse-generator";
import {
  InverseLearningState,
  createInitialLearningState,
  updateLearningState,
  deriveCompositionBias,
  serializeLearningState,
  deserializeLearningState,
} from "./inverse-learning";
import type { CompositionBias } from "./target-schema";

export interface InverseCampaign {
  id: string;
  target: TargetProperties;
  status: "active" | "paused" | "converged" | "completed";
  learningState: InverseLearningState;
  bias: CompositionBias;
  cyclesRun: number;
  candidatesGenerated: number;
  candidatesPassedPipeline: number;
  bestTcAchieved: number;
  maxCycles: number;
}

const activeCampaigns = new Map<string, InverseCampaign>();

export function createCampaign(id: string, target: TargetProperties, maxCycles: number = 100): InverseCampaign {
  const bias = createInitialBias(target);
  const campaign: InverseCampaign = {
    id,
    target,
    status: "active",
    learningState: createInitialLearningState(),
    bias,
    cyclesRun: 0,
    candidatesGenerated: 0,
    candidatesPassedPipeline: 0,
    bestTcAchieved: 0,
    maxCycles,
  };
  activeCampaigns.set(id, campaign);
  return campaign;
}

export function loadCampaign(
  id: string,
  target: TargetProperties,
  learningStateData: any,
  stats: { cyclesRun: number; candidatesGenerated: number; candidatesPassedPipeline: number; bestTcAchieved: number; status: string }
): InverseCampaign {
  const learningState = deserializeLearningState(learningStateData);
  const baseBias = createInitialBias(target);
  const bias = deriveCompositionBias(learningState, baseBias);

  const campaign: InverseCampaign = {
    id,
    target,
    status: stats.status as any,
    learningState,
    bias,
    cyclesRun: stats.cyclesRun,
    candidatesGenerated: stats.candidatesGenerated,
    candidatesPassedPipeline: stats.candidatesPassedPipeline,
    bestTcAchieved: stats.bestTcAchieved,
    maxCycles: 100,
  };
  activeCampaigns.set(id, campaign);
  return campaign;
}

export function getCampaign(id: string): InverseCampaign | undefined {
  return activeCampaigns.get(id);
}

export function getAllActiveCampaigns(): InverseCampaign[] {
  return Array.from(activeCampaigns.values()).filter(c => c.status === "active");
}

export function pauseCampaign(id: string): boolean {
  const campaign = activeCampaigns.get(id);
  if (!campaign) return false;
  campaign.status = "paused";
  return true;
}

export function removeCampaign(id: string): boolean {
  return activeCampaigns.delete(id);
}

export function runInverseCycle(
  campaign: InverseCampaign
): InverseCandidate[] {
  if (campaign.status !== "active") return [];

  const freshCount = 30;
  const refinedCount = 15;
  let allCandidates: InverseCandidate[] = [];

  const freshCandidates = generateInverseCandidates(
    campaign.target,
    campaign.id,
    campaign.cyclesRun,
    campaign.bias,
    freshCount
  );
  allCandidates.push(...freshCandidates);

  if (campaign.learningState.bestCandidates.length > 0) {
    const topN = campaign.learningState.bestCandidates.slice(0, 5);
    for (const best of topN) {
      const refined = refineCandidate(best, campaign.target, campaign.bias);
      const shuffled = refined.sort(() => Math.random() - 0.5);
      allCandidates.push(...shuffled.slice(0, Math.ceil(refinedCount / topN.length)));
    }
  }

  const uniqueFormulas = new Set<string>();
  allCandidates = allCandidates.filter(c => {
    if (uniqueFormulas.has(c.formula)) return false;
    uniqueFormulas.add(c.formula);
    return true;
  });

  campaign.candidatesGenerated += allCandidates.length;
  campaign.cyclesRun += 1;

  if (campaign.cyclesRun >= campaign.maxCycles) {
    campaign.status = "completed";
  }

  return allCandidates;
}

export function processInverseResults(
  campaign: InverseCampaign,
  results: {
    formula: string;
    tc: number;
    lambda: number;
    hull: number;
    pressure: number;
    passedPipeline: boolean;
  }[]
): void {
  const passedCount = results.filter(r => r.passedPipeline).length;
  campaign.candidatesPassedPipeline += passedCount;

  const bestTcInBatch = Math.max(0, ...results.map(r => r.tc));
  if (bestTcInBatch > campaign.bestTcAchieved) {
    campaign.bestTcAchieved = bestTcInBatch;
  }

  const candidatesForLearning = results.map(r => {
    const candidate: InverseCandidate = {
      formula: r.formula,
      source: "inverse",
      campaignId: campaign.id,
      targetDistance: computeTargetDistance(campaign.target, r),
      iteration: campaign.cyclesRun,
      predictedTc: r.tc,
      predictedLambda: r.lambda,
      predictedHull: r.hull,
      predictedPressure: r.pressure,
    };
    return candidate;
  });

  campaign.learningState = updateLearningState(
    campaign.learningState,
    candidatesForLearning,
    results,
    campaign.target
  );

  const baseBias = createInitialBias(campaign.target);
  campaign.bias = deriveCompositionBias(campaign.learningState, baseBias);

  if (campaign.learningState.convergenceHistory.length >= 10) {
    const last10 = campaign.learningState.convergenceHistory.slice(-10);
    const range = Math.max(...last10) - Math.min(...last10);
    if (range < 0.005 && campaign.learningState.bestDistance < 0.1) {
      campaign.status = "converged";
    }
  }
}

export function getCampaignStats(campaign: InverseCampaign): CampaignStats {
  return {
    id: campaign.id,
    target: campaign.target,
    status: campaign.status,
    cyclesRun: campaign.cyclesRun,
    bestTcAchieved: campaign.bestTcAchieved,
    bestDistance: campaign.learningState.bestDistance,
    candidatesGenerated: campaign.candidatesGenerated,
    candidatesPassedPipeline: campaign.candidatesPassedPipeline,
    topCandidates: campaign.learningState.bestCandidates.slice(0, 10).map((c: InverseCandidate) => ({
      formula: c.formula,
      tc: c.predictedTc ?? 0,
      distance: c.targetDistance,
    })),
    convergenceHistory: campaign.learningState.convergenceHistory.slice(-50),
  };
}

export function getSerializableCampaignState(campaign: InverseCampaign) {
  return {
    learningState: serializeLearningState(campaign.learningState),
    convergenceHistory: campaign.learningState.convergenceHistory.slice(-50),
    topCandidates: campaign.learningState.bestCandidates.slice(0, 10).map((c: InverseCandidate) => ({
      formula: c.formula,
      tc: c.predictedTc ?? 0,
      distance: c.targetDistance,
      lambda: c.predictedLambda,
      prototype: c.prototype,
    })),
  };
}

export async function restoreCampaignsFromDB(
  dbCampaigns: { id: string; target: any; learningState: any; cyclesRun: number; candidatesGenerated: number; candidatesPassedPipeline: number; bestTcAchieved: number; status: string }[]
): Promise<number> {
  let restored = 0;
  for (const row of dbCampaigns) {
    if (activeCampaigns.has(row.id)) continue;
    if (row.status !== "active" && row.status !== "paused") continue;
    try {
      loadCampaign(row.id, row.target as TargetProperties, row.learningState, {
        cyclesRun: row.cyclesRun,
        candidatesGenerated: row.candidatesGenerated,
        candidatesPassedPipeline: row.candidatesPassedPipeline,
        bestTcAchieved: row.bestTcAchieved,
        status: row.status,
      });
      restored++;
    } catch {}
  }
  return restored;
}

export function getInverseDesignStats(): {
  activeCampaigns: number;
  totalCandidatesGenerated: number;
  totalPassed: number;
  bestTcAcrossAll: number;
  campaigns: { id: string; status: string; cyclesRun: number; bestTc: number; bestDistance: number }[];
} {
  let totalGen = 0;
  let totalPassed = 0;
  let bestTc = 0;
  const campaignSummaries: { id: string; status: string; cyclesRun: number; bestTc: number; bestDistance: number }[] = [];

  for (const [, c] of activeCampaigns) {
    totalGen += c.candidatesGenerated;
    totalPassed += c.candidatesPassedPipeline;
    if (c.bestTcAchieved > bestTc) bestTc = c.bestTcAchieved;
    campaignSummaries.push({
      id: c.id,
      status: c.status,
      cyclesRun: c.cyclesRun,
      bestTc: c.bestTcAchieved,
      bestDistance: c.learningState.bestDistance,
    });
  }

  return {
    activeCampaigns: getAllActiveCampaigns().length,
    totalCandidatesGenerated: totalGen,
    totalPassed,
    bestTcAcrossAll: bestTc,
    campaigns: campaignSummaries,
  };
}
