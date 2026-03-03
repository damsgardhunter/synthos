import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";

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

const MATERIAL_FAMILIES: Record<string, RegExp> = {
  "Hydrides": /H\d|LaH|YH|CeH|CaH|BaH|SrH|MgH|hydride/i,
  "Cuprates": /Cu.*O|Ba.*Cu|Sr.*Cu|La.*Cu|cuprate|YBCO|BSCCO/i,
  "Pnictides": /Fe.*As|Ba.*Fe|Sr.*Fe|La.*Fe.*As|pnictide/i,
  "Chalcogenides": /Se|Te|FeSe|FeTe|chalcogenide/i,
  "Borides": /B\d|MgB|boride/i,
  "Carbides": /C\d|carbide|SiC/i,
  "Nitrides": /N\d|nitride|BN|GaN|AlN/i,
  "Oxides": /O\d|oxide|perovskite|SrTiO|BaTiO/i,
  "Intermetallics": /Nb.*Sn|Nb.*Ti|V.*Si|intermetallic/i,
};

function classifyFamily(formula: string): string {
  for (const [family, pattern] of Object.entries(MATERIAL_FAMILIES)) {
    if (pattern.test(formula)) return family;
  }
  return "Other";
}

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

    const allFamilies = new Set([
      ...Object.keys(familyStats),
      "Hydrides", "Cuprates", "Pnictides", "Chalcogenides", "Borides"
    ]);
    const underExplored = [...allFamilies].filter(f => (familyStats[f]?.count ?? 0) < 3);

    const signalText = Object.entries(familyStats)
      .sort((a, b) => b[1].avgScore - a[1].avgScore)
      .map(([f, s]) => `${f}: ${s.count} candidates, avg score ${s.avgScore.toFixed(2)}, max Tc ${s.maxTc.toFixed(0)}K, ${s.pipelinePasses} pipeline passes, ${failureByFamily[f] || 0} failures`)
      .join("\n");

    const prompt = `You are a materials science research strategist for an AI superconductor discovery platform.

Current candidate family performance (cycle ${cycleNumber}):
${signalText}

Under-explored families: ${underExplored.join(", ") || "None"}

Recent novel insights: ${insightSummary || "None yet"}

Pipeline failure patterns: ${Object.entries(failureByFamily).map(([f, n]) => `${f}: ${n} failures`).join(", ") || "No failures yet"}

Based on this data, recommend 3-5 material families to focus research on. For each, give a priority (0.0-1.0) and a brief reason.
Also write a 1-2 sentence overall strategy summary.

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

    const focusAreas = (parsed.focusAreas || [])
      .slice(0, 5)
      .map(fa => ({
        area: String(fa.area || "Unknown"),
        priority: Math.max(0, Math.min(1, Number(fa.priority) || 0.5)),
        reasoning: String(fa.reasoning || ""),
      }))
      .sort((a, b) => b.priority - a.priority);

    const summary = String(parsed.summary || "Continuing broad exploration.");

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

export async function captureConvergenceSnapshot(
  emit: EventEmitter,
  cycleNumber: number,
  strategyFocus?: string
): Promise<void> {
  try {
    const topCandidates = await storage.getSuperconductorCandidates(10);
    const totalCount = await storage.getSuperconductorCount();
    const insightCount = await storage.getNovelInsightCount();

    let bestTc = 0;
    let bestScore = 0;
    let topFormula = "";
    let scoreSum = 0;

    for (const c of topCandidates) {
      const tc = c.predictedTc ?? 0;
      const score = c.ensembleScore ?? 0;
      if (tc > bestTc) bestTc = tc;
      if (score > bestScore) {
        bestScore = score;
        topFormula = c.formula;
      }
      scoreSum += score;
    }

    const avgTopScore = topCandidates.length > 0 ? scoreSum / topCandidates.length : 0;

    const stats = await storage.getStats();
    const totalPipelineResults = stats.pipelineStages.reduce((s, p) => s + p.count, 0);
    const totalPipelinePassed = stats.pipelineStages.reduce((s, p) => s + p.passed, 0);
    const pipelinePassRate = totalPipelineResults > 0 ? totalPipelinePassed / totalPipelineResults : 0;

    const snapshotId = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    await storage.insertConvergenceSnapshot({
      id: snapshotId,
      cycle: cycleNumber,
      bestTc: Math.round(bestTc * 100) / 100,
      bestScore: Math.round(bestScore * 1000) / 1000,
      avgTopScore: Math.round(avgTopScore * 1000) / 1000,
      candidatesTotal: totalCount,
      pipelinePassRate: Math.round(pipelinePassRate * 1000) / 1000,
      novelInsightCount: insightCount,
      topFormula,
      strategyFocus: strategyFocus || null,
    });
  } catch (err: any) {
    emit("log", {
      phase: "engine",
      event: "Convergence snapshot error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Strategy Analyzer",
    });
  }
}
