import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { classifyFamily } from "./utils";

interface MilestoneState {
  knownFamilies: Set<string>;
  bestTcEver: number;
  maxPipelineStage: number;
  maxFamilyDiversity: number;
  knowledgeThresholds: Set<number>;
  lastCascadeCycle: number;
}

const state: MilestoneState = {
  knownFamilies: new Set(),
  bestTcEver: 0,
  maxPipelineStage: 0,
  maxFamilyDiversity: 0,
  knowledgeThresholds: new Set(),
  lastCascadeCycle: -10,
};

let initialized = false;

async function initState() {
  if (initialized) return;
  try {
    const candidates = await storage.getSuperconductorCandidates(100);
    for (const c of candidates) {
      state.knownFamilies.add(classifyFamily(c.formula));
      if ((c.predictedTc ?? 0) > state.bestTcEver) state.bestTcEver = c.predictedTc ?? 0;
      if ((c.verificationStage ?? 0) > state.maxPipelineStage) state.maxPipelineStage = c.verificationStage ?? 0;
    }
    const top50 = candidates.slice(0, 50);
    state.maxFamilyDiversity = new Set(top50.map(c => classifyFamily(c.formula))).size;

    const stats = await storage.getStats();
    const totalKnowledge = stats.materialsIndexed + stats.predictionsGenerated + stats.chemicalReactions;
    for (const t of [100, 250, 500, 1000, 2500, 5000]) {
      if (totalKnowledge >= t) state.knowledgeThresholds.add(t);
    }

    const existingMilestones = await storage.getMilestones(1000);
    for (const m of existingMilestones) {
      if (m.type === "knowledge-milestone") {
        const match = m.description.match(/(\d+)/);
        if (match) state.knowledgeThresholds.add(Number(match[1]));
      }
      if (m.type === "pipeline-graduate" && m.title) {
        const thresholdMatch = m.title.match(/^(\d+)\s+candidates/);
        if (thresholdMatch) {
          const threshold = Number(thresholdMatch[1]);
          state.knowledgeThresholds.add(threshold * 1000 + 4);
        }
      }
    }
  } catch {}
  initialized = true;
}

interface DetectedMilestone {
  type: string;
  title: string;
  description: string;
  significance: number;
  relatedFormula?: string;
}

export async function checkMilestones(
  emit: EventEmitter,
  broadcast: (type: string, data: any) => void,
  cycleNumber: number,
  cycleInsightCount: number
): Promise<DetectedMilestone[]> {
  await initState();
  const detected: DetectedMilestone[] = [];

  try {
    const candidates = await storage.getSuperconductorCandidates(100);

    for (const c of candidates) {
      const family = classifyFamily(c.formula);
      if (!state.knownFamilies.has(family)) {
        state.knownFamilies.add(family);
        detected.push({
          type: "new-family",
          title: `New material family: ${family}`,
          description: `First candidate from the ${family} family discovered: ${c.formula}. Expanding chemical search space.`,
          significance: 2,
          relatedFormula: c.formula,
        });
      }
    }

    for (const c of candidates) {
      const tc = c.predictedTc ?? 0;
      if (tc > state.bestTcEver && tc > state.bestTcEver + 5) {
        const improvement = Math.round(tc - state.bestTcEver);
        detected.push({
          type: "tc-record",
          title: `Tc record: ${Math.round(tc)}K`,
          description: `${c.formula} surpasses previous best by ${improvement}K. New highest predicted Tc: ${Math.round(tc)}K.`,
          significance: tc > 350 ? 3 : 2,
          relatedFormula: c.formula,
        });
        state.bestTcEver = tc;
      }
    }

    let newGraduates = 0;
    for (const c of candidates) {
      const stage = c.verificationStage ?? 0;
      if (stage >= 4 && state.maxPipelineStage < 4) {
        detected.push({
          type: "pipeline-graduate",
          title: `Pipeline graduate: ${c.formula}`,
          description: `${c.formula} passed all multi-fidelity verification stages. Full computational validation complete with Tc=${Math.round(c.predictedTc ?? 0)}K.`,
          significance: 3,
          relatedFormula: c.formula,
        });
        state.maxPipelineStage = 4;
        newGraduates++;
      }
    }

    const stage4Count = candidates.filter(c => (c.verificationStage ?? 0) >= 4).length;
    const stage4Thresholds = [10, 25, 50, 100];
    for (const threshold of stage4Thresholds) {
      if (stage4Count >= threshold && !state.knowledgeThresholds.has(threshold * 1000 + 4)) {
        state.knowledgeThresholds.add(threshold * 1000 + 4);
        detected.push({
          type: "pipeline-graduate",
          title: `${threshold} candidates fully validated`,
          description: `${threshold} superconductor candidates have completed all pipeline stages. Deep computational validation accelerating.`,
          significance: threshold >= 50 ? 3 : 2,
        });
      }
    }

    const top50 = candidates.slice(0, 50);
    const currentDiversity = new Set(top50.map(c => classifyFamily(c.formula))).size;
    if (currentDiversity > state.maxFamilyDiversity && currentDiversity >= state.maxFamilyDiversity + 2) {
      detected.push({
        type: "diversity-threshold",
        title: `Diversity expanded to ${currentDiversity} families`,
        description: `Search space now spans ${currentDiversity} distinct material families. Broader exploration increases discovery probability.`,
        significance: currentDiversity >= 8 ? 2 : 1,
      });
      state.maxFamilyDiversity = currentDiversity;
    }

    const stats = await storage.getStats();
    const totalKnowledge = stats.materialsIndexed + stats.predictionsGenerated + stats.chemicalReactions;
    for (const threshold of [100, 250, 500, 1000, 2500, 5000]) {
      if (totalKnowledge >= threshold && !state.knowledgeThresholds.has(threshold)) {
        state.knowledgeThresholds.add(threshold);
        detected.push({
          type: "knowledge-milestone",
          title: `Knowledge base: ${threshold}+ data points`,
          description: `Total knowledge surpassed ${threshold}: ${stats.materialsIndexed} materials, ${stats.predictionsGenerated} predictions, ${stats.chemicalReactions} reactions.`,
          significance: threshold >= 1000 ? 2 : 1,
        });
      }
    }

    if (cycleInsightCount >= 3 && (cycleNumber - state.lastCascadeCycle) >= 10) {
      detected.push({
        type: "insight-cascade",
        title: `Insight cascade: ${cycleInsightCount} discoveries`,
        description: `${cycleInsightCount} novel insights discovered in a single cycle. Rapid knowledge accumulation indicates productive exploration.`,
        significance: cycleInsightCount >= 5 ? 3 : 2,
      });
      state.lastCascadeCycle = cycleNumber;
    }
  } catch {}

  for (const m of detected) {
    try {
      const milestoneId = `ms-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const saved = await storage.insertMilestone({
        id: milestoneId,
        cycle: cycleNumber,
        type: m.type,
        title: m.title,
        description: m.description,
        significance: m.significance,
        relatedFormula: m.relatedFormula ?? null,
      });

      broadcast("milestone", {
        id: milestoneId,
        cycle: cycleNumber,
        type: m.type,
        title: m.title,
        description: m.description,
        significance: m.significance,
        relatedFormula: m.relatedFormula ?? null,
      });

      emit("log", {
        phase: "engine",
        event: `Milestone: ${m.title}`,
        detail: m.description,
        dataSource: "Internal",
      });
    } catch {}
  }

  return detected;
}
