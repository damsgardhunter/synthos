import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { classifyFamily, sanitizeForbiddenWords } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

interface FocusArea {
  area: string;
  priority: number;
  reasoning: string;
}

interface StrategyResult {
  focusAreas: FocusArea[];
  summary: string;
  performanceSignals: Record<string, any>;
}

let previousFocusAreas: FocusArea[] = [];

export async function analyzeAndEvolveStrategy(
  emit: EventEmitter,
  cycleNumber: number
): Promise<StrategyResult | null> {
  try {
    const candidates = await storage.getSuperconductorCandidates(100);
    if (candidates.length < 5) return null;

    const familyStats: Record<string, {
      count: number;
      avgScore: number;
      maxTc: number;
      maxScore: number;
      pipelinePasses: number;
      totalScores: number;
    }> = {};

    for (const c of candidates) {
      const family = classifyFamily(c.formula);
      if (!familyStats[family]) {
        familyStats[family] = { count: 0, avgScore: 0, maxTc: 0, maxScore: 0, pipelinePasses: 0, totalScores: 0 };
      }
      const stats = familyStats[family];
      stats.count++;
      const score = c.ensembleScore ?? 0;
      stats.totalScores += score;
      stats.avgScore = stats.totalScores / stats.count;
      if ((c.predictedTc ?? 0) > stats.maxTc) stats.maxTc = c.predictedTc ?? 0;
      if (score > stats.maxScore) stats.maxScore = score;
      if ((c.verificationStage ?? 0) >= 2) stats.pipelinePasses++;
    }

    const failedResults = await storage.getFailedComputationalResults(50);
    const failureByFamily: Record<string, number> = {};
    for (const r of failedResults) {
      const family = classifyFamily(r.formula);
      failureByFamily[family] = (failureByFamily[family] || 0) + 1;
    }

    const recentInsights = await storage.getNovelInsightsOnly(10);
    const insightSummary = recentInsights.map(i => i.insightText).join("; ");

    const totalSamples = candidates.length;
    const priorMean = 0.5;
    const k = 10;

    const adjustedFamilyStats: Record<string, {
      count: number;
      adjustedScore: number;
      rawAvgScore: number;
      maxTc: number;
      maxScore: number;
      pipelinePasses: number;
      confidence: number;
    }> = {};

    for (const [family, stats] of Object.entries(familyStats)) {
      const adjustedScore = (stats.totalScores + priorMean * k) / (stats.count + k);
      const familyConfidence = Math.sqrt(stats.count / totalSamples);
      const weightedScore = adjustedScore * familyConfidence;

      adjustedFamilyStats[family] = {
        count: stats.count,
        adjustedScore: weightedScore,
        rawAvgScore: stats.avgScore,
        maxTc: stats.maxTc,
        maxScore: stats.maxScore,
        pipelinePasses: stats.pipelinePasses,
        confidence: familyConfidence,
      };
    }

    const allFamilies = new Set([
      ...Object.keys(familyStats),
      "Hydrides", "Cuprates", "Pnictides", "Chalcogenides", "Borides"
    ]);
    const underExplored = [...allFamilies].filter(f => (familyStats[f]?.count ?? 0) < 3);

    const signalText = Object.entries(adjustedFamilyStats)
      .sort((a, b) => b[1].adjustedScore - a[1].adjustedScore)
      .map(([f, s]) => `${f}: ${s.count} candidates, adjusted score ${s.adjustedScore.toFixed(3)} (raw avg ${s.rawAvgScore.toFixed(2)}, confidence ${s.confidence.toFixed(2)}), max Tc ${s.maxTc.toFixed(0)}K, ${s.pipelinePasses} pipeline passes, ${failureByFamily[f] || 0} failures`)
      .join("\n");

    let previousStrategyContext = "";
    if (previousFocusAreas.length > 0) {
      previousStrategyContext = `\nPrevious cycle priorities: ${previousFocusAreas.map(f => `${f.area} (${(f.priority * 100).toFixed(0)}%)`).join(", ")}`;
      previousStrategyContext += `\nIf recommending different priorities than above, explicitly explain what changed and why in your summary.`;
    }

    const prompt = `You are a materials science research strategist for an AI superconductor discovery platform.

Current candidate family performance (cycle ${cycleNumber}, scores are Bayesian-adjusted with confidence weighting — families with few samples are penalized):
${signalText}

Under-explored families: ${underExplored.join(", ") || "None"}

Recent novel insights: ${insightSummary || "None yet"}

Pipeline failure patterns: ${Object.entries(failureByFamily).map(([f, n]) => `${f}: ${n} failures`).join(", ") || "No failures yet"}
${previousStrategyContext}

Based on this data, recommend 3-5 material families to focus research on. For each, give a priority (0.0-1.0) and a brief reason.
Also write a 1-2 sentence overall strategy summary that references specific data trends.

Respond in JSON:
{
  "focusAreas": [{"area": "Hydrides", "priority": 0.9, "reasoning": "..."}],
  "summary": "..."
}`;

    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      temperature: 0.7,
      max_tokens: 500,
    });

    const content = resp.choices[0]?.message?.content;
    if (!content) return null;

    const parsed = JSON.parse(content) as { focusAreas: FocusArea[]; summary: string };

    const prevMap = new Map(previousFocusAreas.map(f => [f.area, f.priority]));
    const MOMENTUM = 0.7;

    const focusAreas = (parsed.focusAreas || [])
      .slice(0, 5)
      .map(fa => {
        const area = String(fa.area || "Unknown");
        let priority = Math.max(0, Math.min(1, Number(fa.priority) || 0.5));
        const familySampleCount = familyStats[area]?.count ?? 0;
        if (familySampleCount < 3) {
          priority = Math.min(priority, 0.5);
        }
        const prevPriority = prevMap.get(area);
        if (prevPriority !== undefined) {
          priority = MOMENTUM * prevPriority + (1 - MOMENTUM) * priority;
        }
        return {
          area,
          priority: Math.round(priority * 1000) / 1000,
          reasoning: sanitizeForbiddenWords(String(fa.reasoning || "")),
        };
      })
      .sort((a, b) => b.priority - a.priority);

    const summary = sanitizeForbiddenWords(String(parsed.summary || "Continuing broad exploration."));

    const performanceSignals = { familyStats, failureByFamily, underExplored, insightCount: recentInsights.length };

    const strategyId = `strat-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    await storage.insertResearchStrategy({
      id: strategyId,
      cycle: cycleNumber,
      focusAreas,
      summary,
      performanceSignals,
    });

    emit("log", {
      phase: "engine",
      event: "Research strategy evolved",
      detail: `Focus: ${focusAreas.map(f => `${f.area} (${(f.priority * 100).toFixed(0)}%)`).join(", ")}. ${summary}`,
      dataSource: "Strategy Analyzer",
    });

    if (previousFocusAreas.length > 0) {
      const pivots: string[] = [];
      const prevMap = new Map(previousFocusAreas.map(f => [f.area, f.priority]));
      const newMap = new Map(focusAreas.map(f => [f.area, f.priority]));

      for (const fa of focusAreas) {
        const prevPriority = prevMap.get(fa.area);
        if (prevPriority === undefined) {
          const newTop3 = focusAreas.slice(0, 3).map(f => f.area);
          if (newTop3.includes(fa.area)) {
            pivots.push(`${fa.area} is new in top priorities at ${(fa.priority * 100).toFixed(0)}%`);
          }
        } else {
          const delta = fa.priority - prevPriority;
          if (Math.abs(delta) > 0.15) {
            const direction = delta > 0 ? "promoted" : "deprioritized";
            pivots.push(`${fa.area} ${direction}: ${(prevPriority * 100).toFixed(0)}% -> ${(fa.priority * 100).toFixed(0)}%`);
          }
        }
      }

      for (const prev of previousFocusAreas) {
        if (!newMap.has(prev.area) && prev.priority > 0.5) {
          pivots.push(`${prev.area} dropped from focus (was ${(prev.priority * 100).toFixed(0)}%)`);
        }
      }

      if (pivots.length > 0) {
        emit("log", {
          phase: "engine",
          event: "Strategy pivot",
          detail: pivots.join(". ") + ".",
          dataSource: "Strategy Analyzer",
        });
      }
    }

    previousFocusAreas = [...focusAreas];

    return { focusAreas, summary, performanceSignals };
  } catch (err: any) {
    emit("log", {
      phase: "engine",
      event: "Strategy analysis error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Strategy Analyzer",
    });
    return null;
  }
}

let cumulativeDuplicatesSkipped = 0;

export function trackDuplicatesSkipped(count: number): void {
  cumulativeDuplicatesSkipped += count;
}

export async function captureConvergenceSnapshot(
  emit: EventEmitter,
  cycleNumber: number,
  strategyFocus?: string
): Promise<void> {
  try {
    const topCandidates = await storage.getSuperconductorCandidates(50);
    const topByTc = await storage.getSuperconductorCandidatesByTc(10);
    const totalCount = await storage.getSuperconductorCount();
    const insightCount = await storage.getNovelInsightCount();

    const seenFormulas = new Set<string>();
    const merged = [...topCandidates, ...topByTc];
    const uniqueCandidates = merged.filter(c => {
      if (seenFormulas.has(c.formula)) return false;
      seenFormulas.add(c.formula);
      return true;
    });

    let bestTc = 0;
    let bestPhysicsTc = 0;
    let bestScore = 0;
    let topFormula = "";
    let scoreSum = 0;
    const topByScore10 = uniqueCandidates
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0))
      .slice(0, 10);

    for (const c of uniqueCandidates) {
      const tc = c.predictedTc ?? 0;
      if (tc > bestTc) {
        bestTc = tc;
        topFormula = c.formula;
      }
      if ((c.verificationStage ?? 0) >= 1 && c.electronPhononCoupling != null) {
        if (tc > bestPhysicsTc) bestPhysicsTc = tc;
      }
    }
    for (const c of topByScore10) {
      const score = c.ensembleScore ?? 0;
      if (score > bestScore) bestScore = Math.min(score, 0.95);
      scoreSum += Math.min(score, 0.95);
    }

    const avgTopScore = topByScore10.length > 0 ? scoreSum / topByScore10.length : 0;

    const stats = await storage.getStats();
    const stage0 = stats.pipelineStages.find(p => p.stage === 0);
    const stage4 = stats.pipelineStages.find(p => p.stage === 4);
    const totalEnteredPipeline = stage0 ? stage0.count : 0;
    const totalPassedAllStages = stage4 ? stage4.passed : 0;
    const pipelinePassRate = totalEnteredPipeline > 0 ? totalPassedAllStages / totalEnteredPipeline : 0;

    const allCandidatesForDiversity = await storage.getSuperconductorCandidates(500);
    const allFamilies = new Set(allCandidatesForDiversity.map(c => classifyFamily(c.formula)));
    allFamilies.delete("Other");
    const familyDiversity = allFamilies.size + (allCandidatesForDiversity.some(c => classifyFamily(c.formula) === "Other") ? 1 : 0);

    await storage.deleteConvergenceSnapshotByCycle(cycleNumber);
    const snapshotId = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    await storage.insertConvergenceSnapshot({
      id: snapshotId,
      cycle: cycleNumber,
      bestTc: Math.round(bestTc * 100) / 100,
      bestPhysicsTc: bestPhysicsTc > 0 ? Math.round(bestPhysicsTc * 100) / 100 : null,
      bestScore: Math.round(bestScore * 1000) / 1000,
      avgTopScore: Math.round(avgTopScore * 1000) / 1000,
      candidatesTotal: totalCount,
      pipelinePassRate: Math.round(pipelinePassRate * 1000) / 1000,
      novelInsightCount: insightCount,
      topFormula,
      strategyFocus: strategyFocus || null,
      familyDiversity,
      duplicatesSkipped: cumulativeDuplicatesSkipped,
    });

    cumulativeDuplicatesSkipped = 0;
  } catch (err: any) {
    emit("log", {
      phase: "engine",
      event: "Convergence snapshot error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Strategy Analyzer",
    });
  }
}
