import OpenAI from "openai";
import crypto from "crypto";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { classifyFamily, sanitizeForbiddenWords } from "./utils";
import { getModelDiagnosticsSummaryForStrategy } from "./model-improvement-loop";
import { getPerformanceMetrics } from "../theory/model-performance-tracker";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  timeout: 60_000,
  maxRetries: 0, // Connection errors do not self-resolve; avoid 3x retry amplification
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
    // Only call the OpenAI LLM every 5 cycles — the response is rarely different
    // cycle-to-cycle and each call takes ~2s. Null return leaves currentStrategyFocusAreas unchanged.
    if (cycleNumber % 5 !== 0 && previousFocusAreas.length > 0) return null;

    const candidates = await storage.getSuperconductorCandidates(100);
    if (candidates.length < 5) return null;

    const familyStats: Record<string, {
      count: number;
      avgScore: number;
      maxTc: number;
      maxScore: number;
      pipelinePasses: number;
      totalScores: number;
      tcValues: number[];
    }> = {};

    for (const c of candidates) {
      const family = classifyFamily(c.formula);
      if (!familyStats[family]) {
        familyStats[family] = { count: 0, avgScore: 0, maxTc: 0, maxScore: 0, pipelinePasses: 0, totalScores: 0, tcValues: [] };
      }
      const stats = familyStats[family];
      stats.count++;
      const tc = c.predictedTc ?? 0;
      const score = c.ensembleScore ?? 0;
      stats.totalScores += score;
      stats.avgScore = stats.totalScores / stats.count;
      stats.tcValues.push(tc);
      if (tc > stats.maxTc) stats.maxTc = tc;
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

    const adjustedFamilyStats: Record<string, {
      count: number;
      familyScore: number;
      rawAvgScore: number;
      maxTc: number;
      maxScore: number;
      pipelinePasses: number;
      bayesianTc: number;
      successRate: number;
      explorationBonus: number;
      pipelinePassRate: number;
    }> = {};

    const allAvgTcs = Object.values(familyStats).map(s => s.maxTc).sort((a, b) => a - b);
    const globalMedianTc = allAvgTcs.length > 0
      ? (allAvgTcs.length % 2 === 0
        ? (allAvgTcs[allAvgTcs.length / 2 - 1] + allAvgTcs[allAvgTcs.length / 2]) / 2
        : allAvgTcs[Math.floor(allAvgTcs.length / 2)])
      : 30;
    const priorCount = 5;
    const priorTc = globalMedianTc;

    const rawScores: number[] = [];

    for (const [family, stats] of Object.entries(familyStats)) {
      const sorted = stats.tcValues.slice().sort((a, b) => b - a);
      const topDecileN = Math.max(1, Math.ceil(sorted.length * 0.1));
      const representativeTc = sorted.slice(0, topDecileN).reduce((s, v) => s + v, 0) / topDecileN;
      const bayesianTc = (stats.count * representativeTc + priorCount * priorTc) / (stats.count + priorCount);
      const successRate = stats.pipelinePasses / Math.max(1, stats.count);
      const explorationBonus = stats.count < 10 ? 0.1 : 0;
      const rawScore = bayesianTc * Math.log2(stats.count + 1) * (0.5 + 0.5 * successRate);
      rawScores.push(rawScore);

      adjustedFamilyStats[family] = {
        count: stats.count,
        familyScore: rawScore,
        rawAvgScore: stats.avgScore,
        maxTc: stats.maxTc,
        maxScore: stats.maxScore,
        pipelinePasses: stats.pipelinePasses,
        bayesianTc: Math.round(bayesianTc * 100) / 100,
        successRate: Math.round(successRate * 1000) / 1000,
        explorationBonus,
        pipelinePassRate: stats.count > 0 ? stats.pipelinePasses / stats.count : 0,
      };
    }

    const maxRawScore = Math.max(...rawScores, 0.001);
    for (const family of Object.keys(adjustedFamilyStats)) {
      const entry = adjustedFamilyStats[family];
      entry.familyScore = entry.familyScore / maxRawScore + entry.explorationBonus;
    }

    const allFamilies = new Set([
      ...Object.keys(familyStats),
      "Hydrides", "Cuprates", "Pnictides", "Chalcogenides", "Borides"
    ]);
    const underExplored = [...allFamilies].filter(f => (familyStats[f]?.count ?? 0) < 3);

    const signalText = Object.entries(adjustedFamilyStats)
      .sort((a, b) => b[1].familyScore - a[1].familyScore)
      .map(([f, s]) => `${f}: familyScore=${s.familyScore.toFixed(3)} (bayesianTc=${s.bayesianTc.toFixed(1)} × log2(${s.count}+1) × (0.5+0.5×successRate=${s.successRate.toFixed(2)})${s.explorationBonus > 0 ? ' +explorationBonus=0.1' : ''}), maxTc=${s.maxTc.toFixed(0)}K, ${s.count} candidates, raw avg score ${s.rawAvgScore.toFixed(2)}, ${s.pipelinePasses} pipeline passes, ${failureByFamily[f] || 0} failures`)
      .join("\n");

    let previousStrategyContext = "";
    if (previousFocusAreas.length > 0) {
      previousStrategyContext = `\nPrevious cycle priorities: ${previousFocusAreas.map(f => `${f.area} (${(f.priority * 100).toFixed(0)}%)`).join(", ")}`;
      previousStrategyContext += `\nIf recommending different priorities than above, explicitly explain what changed and why in your summary.`;
    }

    const prompt = `You are the STRATEGY LLM for an AI superconductor discovery platform.
Your SOLE responsibility is controlling EXPLORATION: which material families to investigate, pressure ranges to target, and search direction.
You do NOT control model training, hyperparameters, features, or architecture — that is handled by a separate Model LLM.

Current candidate family performance (cycle ${cycleNumber}, scored by Bayesian formula: bayesianTc × log2(count+1) × (0.5 + 0.5 × successRate), normalized to [0,1]. bayesianTc shrinks small-sample maxTc toward the global median using a prior of 5 samples. Families with < 10 candidates get a +0.1 exploration bonus):
${signalText}

Under-explored families: ${underExplored.join(", ") || "None"}

Recent novel insights: ${insightSummary || "None yet"}

Pipeline failure patterns: ${Object.entries(failureByFamily).map(([f, n]) => `${f}: ${n} failures`).join(", ") || "No failures yet"}
${await getModelDiagnosticsSummaryForStrategy()}
${previousStrategyContext}

YOUR SCOPE (exploration decisions only):
- Which material families to prioritize (Hydrides, Cuprates, Borides, Pnictides, etc.)
- Whether to explore new families or exploit known high-performers
- Pressure ranges to focus on (ambient vs high-pressure)
- Search direction: broad exploration vs focused exploitation

OUT OF SCOPE (handled by Model LLM):
- Model training parameters, hyperparameters, learning rates
- Feature engineering or model architecture
- Dataset augmentation or retraining decisions

IMPORTANT SCORING RULES:
- The familyScore uses Bayesian shrinkage: small-sample families have their Tc pulled toward the global median, preventing single outliers from dominating.
- The log2(count+1) factor means families need statistical mass (many candidates) to score high.
- The successRate factor rewards families with actual pipeline passes, not just high predicted Tc.
- Do NOT over-weight families with fewer than 5 candidates. A single high-Tc candidate is interesting but not statistically reliable — it could be a prediction artifact.
- Families with >= 10 candidates and consistently high scores deserve the highest priorities.
- Under-explored families (< 10 candidates) receive a small exploration bonus but should NOT dominate the strategy.

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

    const focusAreas = (parsed.focusAreas || [])
      .slice(0, 5)
      .map(fa => {
        const area = String(fa.area || "Unknown");
        let priority = Math.max(0, Math.min(1, Number(fa.priority) || 0.5));
        const familySampleCount = familyStats[area]?.count ?? 0;
        if (familySampleCount < 5) {
          const cap = 0.3 + 0.14 * familySampleCount;
          priority = Math.min(priority, cap);
        }
        const prevPriority = prevMap.get(area);
        if (prevPriority !== undefined) {
          const isIncrease = priority > prevPriority;
          const momentum = isIncrease ? 0.5 : 0.85;
          priority = momentum * prevPriority + (1 - momentum) * priority;
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

    const strategyId = `strat-${crypto.randomUUID()}`;
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
        const pivotDetail = pivots.join(". ") + ".";
        emit("log", {
          phase: "engine",
          event: "Strategy pivot",
          detail: pivotDetail,
          dataSource: "Strategy Analyzer",
        });
        try {
          await storage.insertResearchLog({
            phase: "strategy",
            event: "Strategy pivot",
            detail: `Cycle ${cycleNumber}: ${pivotDetail} Summary: ${summary}`,
            dataSource: "Strategy Analyzer",
          });
        } catch {}
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
    const uniqueCandidates = await storage.getTopCandidatesMerged(50, 10);
    const totalCount = await storage.getSuperconductorCount();
    const insightCount = await storage.getNovelInsightCount();

    let bestTc = 0;
    let bestPhysicsTc = 0;
    let bestScore = 0;
    let topFormula = "";
    let dftSelectedTc = 0;
    let avgTop10Tc = 0;

    // Sort all candidates by Tc descending to get top 10
    const allByTc = [...uniqueCandidates].sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));
    const top10ByTc = allByTc.slice(0, 10);

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

    // bestScore & avgTopScore: use physics engine Tc of DFT-selected candidates this cycle
    // Priority: 1) candidates selected for DFT (high uncertainty + high Tc)
    //           2) best physics engine Tc from current cycle
    //           3) zero
    const dftCandidates = uniqueCandidates
      .filter(c => c.dataConfidence === "dft-verified" || ((c.uncertaintyEstimate ?? 0) > 0.3 && (c.predictedTc ?? 0) > 20))
      .sort((a, b) => {
        const aScore = 0.5 * Math.min(1, (a.predictedTc ?? 0) / 300) + 0.5 * (a.uncertaintyEstimate ?? 0);
        const bScore = 0.5 * Math.min(1, (b.predictedTc ?? 0) / 300) + 0.5 * (b.uncertaintyEstimate ?? 0);
        return bScore - aScore;
      });

    if (dftCandidates.length > 0) {
      // Best DFT-selected candidate's physics Tc
      dftSelectedTc = dftCandidates[0].predictedTc ?? 0;
      bestScore = dftSelectedTc;
    } else if (bestPhysicsTc > 0) {
      dftSelectedTc = bestPhysicsTc;
      bestScore = bestPhysicsTc;
    }

    // Average Tc of top 10 candidates by Tc
    if (top10ByTc.length > 0) {
      const tcSum = top10ByTc.reduce((s, c) => s + (c.predictedTc ?? 0), 0);
      avgTop10Tc = tcSum / top10ByTc.length;
    }

    // avgTopScore tracks the avg ensemble score of top 10 by ensemble
    const topByScore10 = [...uniqueCandidates]
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0))
      .slice(0, 10);
    let scoreSum = 0;
    for (const c of topByScore10) {
      scoreSum += Math.min(c.ensembleScore ?? 0, 0.95);
    }
    const avgTopScore = topByScore10.length > 0 ? scoreSum / topByScore10.length : 0;

    const stats = await storage.getStats();
    // Use the highest stage with data for pass rate calculation
    // If only stage 0 exists (common), use stage 0's pass rate
    // Otherwise use the ratio of last stage passed to first stage entered
    const sortedStages = stats.pipelineStages.sort((a, b) => a.stage - b.stage);
    const firstStage = sortedStages[0];
    const lastStage = sortedStages.length > 1 ? sortedStages[sortedStages.length - 1] : firstStage;
    const totalEnteredPipeline = firstStage ? firstStage.count : 0;
    const totalPassed = lastStage ? lastStage.passed : 0;
    const pipelinePassRate = totalEnteredPipeline > 0 ? totalPassed / totalEnteredPipeline : 0;

    const allCandidatesForDiversity = await storage.getSuperconductorCandidates(500);
    const allFamilies = new Set<string>();
    const otherSpaceGroups = new Map<string, number>();
    for (const c of allCandidatesForDiversity) {
      const fam = classifyFamily(c.formula);
      if (fam === "Other") {
        const sg = (c as any).spaceGroup || "unknown";
        otherSpaceGroups.set(sg, (otherSpaceGroups.get(sg) || 0) + 1);
      } else {
        allFamilies.add(fam);
      }
    }
    let familyDiversity = allFamilies.size;
    if (otherSpaceGroups.size > 0) {
      let otherEntropy = 0;
      const otherTotal = Array.from(otherSpaceGroups.values()).reduce((s, v) => s + v, 0);
      for (const count of otherSpaceGroups.values()) {
        const p = count / otherTotal;
        if (p > 0) otherEntropy -= p * Math.log2(p);
      }
      const maxEntropy = Math.log2(Math.max(2, otherSpaceGroups.size));
      const normalizedEntropy = maxEntropy > 0 ? otherEntropy / maxEntropy : 0;
      const otherContribution = Math.max(1, Math.round(normalizedEntropy * Math.min(otherSpaceGroups.size, 5)));
      familyDiversity += otherContribution;
    }

    // Use GNN R² from GCP training (authoritative) instead of in-session performance tracker
    let r2Score: number | null = null;
    try {
      const latestGnn = await storage.getLatestCompletedGnnJob();
      if (latestGnn && latestGnn.r2 != null) {
        r2Score = Math.round(latestGnn.r2 * 10000) / 10000;
      }
    } catch (e) { /* GNN R² unavailable, fall back to null */ }

    // Fallback: if GNN R² not available, try in-session tracker (but only if positive)
    if (r2Score == null) {
      const perfMetrics = getPerformanceMetrics();
      if (perfMetrics.totalPredictions >= 10 && perfMetrics.last100.r2 > 0) {
        r2Score = Math.round(perfMetrics.last100.r2 * 1000) / 1000;
      }
    }

    // If bestTc is still 0, try a direct DB query as fallback
    if (bestTc === 0) {
      try {
        const directTop = await storage.getSuperconductorCandidates(1, 0);
        if (directTop.length > 0 && (directTop[0].predictedTc ?? 0) > 0) {
          bestTc = directTop[0].predictedTc ?? 0;
          topFormula = directTop[0].formula;
        }
      } catch (e) { /* fallback failed */ }
    }

    await storage.deleteConvergenceSnapshotByCycle(cycleNumber);
    const snapshotId = `conv-${crypto.randomUUID()}`;
    await storage.insertConvergenceSnapshot({
      id: snapshotId,
      cycle: cycleNumber,
      bestTc: Math.round(bestTc * 100) / 100,
      bestPhysicsTc: bestPhysicsTc > 0 ? Math.round(bestPhysicsTc * 100) / 100 : null,
      bestScore: Math.round(bestScore * 1000) / 1000,
      avgTopScore: Math.round(avgTopScore * 1000) / 1000,
      avgTop10Tc: Math.round(avgTop10Tc * 100) / 100,
      dftSelectedTc: dftSelectedTc > 0 ? Math.round(dftSelectedTc * 100) / 100 : null,
      candidatesTotal: totalCount,
      pipelinePassRate: Math.round(pipelinePassRate * 1000) / 1000,
      novelInsightCount: insightCount,
      topFormula,
      strategyFocus: strategyFocus || null,
      familyDiversity,
      duplicatesSkipped: cumulativeDuplicatesSkipped,
      r2Score,
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
