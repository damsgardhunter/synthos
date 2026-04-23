import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { classifyFamily, parseFormulaCounts } from "./utils";
import { checkFormulaNovelty } from "./cod-client";

interface MilestoneState {
  knownFamilies: Set<string>;
  // Tracks compositionally distinct candidates per family. A family is only
  // "confirmed" once it reaches MIN_FAMILY_CORROBORATION *distinct* hits.
  familyCandidateCounts: Map<string, { count: number; formulas: string[]; elementSets: Set<string>[]; stoichiometries: Record<string, number>[] }>;
  bestTcEver: number;
  maxPipelineStage: number;
  maxFamilyDiversity: number;
  knowledgeThresholds: Set<number>;
  lastCascadeCycle: number;
  // Cache novelty lookups to avoid repeated DB/API calls
  noveltyCache: Map<string, boolean>;
}

const MIN_FAMILY_CORROBORATION = 3;
// Jaccard threshold for element-set overlap. Two candidates with >= this overlap
// are considered near-duplicates and only one counts toward corroboration.
const MAX_CORROBORATOR_OVERLAP = 0.8;
// Cosine-similarity threshold for stoichiometric ratios. If two formulas have
// the same elements (or very similar element sets) AND their stoichiometric
// ratio vectors are >= this similar, they're treated as near-duplicates.
// This catches Co2HKLi5Sb vs Co2HKLi3Sb (same elements, similar ratios).
const MAX_STOICHIOMETRY_SIMILARITY = 0.92;

/** Compute cosine similarity between two stoichiometry vectors.
 *  Both are Records mapping element → count. Missing elements are treated as 0. */
function stoichiometrySimilarity(a: Record<string, number>, b: Record<string, number>): number {
  const allElements = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, magA = 0, magB = 0;
  for (const el of allElements) {
    const va = a[el] ?? 0;
    const vb = b[el] ?? 0;
    dot += va * vb;
    magA += va * va;
    magB += vb * vb;
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

/** Returns true if `elSet` is compositionally distinct from all existing corroborators.
 *  Two candidates are "too similar" when EITHER:
 *  (a) their element-set Jaccard similarity >= 0.8, OR
 *  (b) their elements overlap significantly AND their stoichiometric ratios are
 *      very similar (cosine >= 0.92), catching Co2HKLi5Sb vs Co2HKLi3Sb. */
function isCompositionallyDistinct(
  elSet: Set<string>,
  stoich: Record<string, number>,
  existing: Set<string>[],
  existingStoich: Record<string, number>[],
): boolean {
  for (let i = 0; i < existing.length; i++) {
    const other = existing[i];
    let intersection = 0;
    for (const el of elSet) {
      if (other.has(el)) intersection++;
    }
    const union = new Set([...elSet, ...other]).size;
    // Check 1: element-set Jaccard overlap
    if (union > 0 && intersection / union >= MAX_CORROBORATOR_OVERLAP) return false;
    // Check 2: stoichiometric similarity for overlapping element sets
    // If elements overlap >= 60% AND stoichiometry is very similar, it's a near-duplicate
    if (union > 0 && intersection / union >= 0.6 && existingStoich[i]) {
      const sim = stoichiometrySimilarity(stoich, existingStoich[i]);
      if (sim >= MAX_STOICHIOMETRY_SIMILARITY) return false;
    }
  }
  return true;
}

const state: MilestoneState = {
  knownFamilies: new Set(),
  familyCandidateCounts: new Map(),
  bestTcEver: 0,
  maxPipelineStage: 0,
  maxFamilyDiversity: 0,
  knowledgeThresholds: new Set(),
  lastCascadeCycle: -10,
  noveltyCache: new Map(),
};

let initialized = false;

/** Check if a formula is a known compound (in COD, Materials Project, etc.).
 *  Results are cached to avoid repeated DB lookups. */
async function isKnownCompound(formula: string): Promise<boolean> {
  if (state.noveltyCache.has(formula)) return state.noveltyCache.get(formula)!;
  try {
    const result = await checkFormulaNovelty(formula);
    state.noveltyCache.set(formula, result.isKnown);
    // Cap cache size
    if (state.noveltyCache.size > 500) {
      const firstKey = state.noveltyCache.keys().next().value;
      if (firstKey) state.noveltyCache.delete(firstKey);
    }
    return result.isKnown;
  } catch {
    return false; // If check fails, assume novel (don't penalize)
  }
}

async function initState() {
  if (initialized) return;
  try {
    const topByTc = await storage.getSuperconductorCandidatesByTc(1);
    if (topByTc.length > 0) {
      state.bestTcEver = topByTc[0].predictedTc ?? 0;
    }

    const candidates = await storage.getSuperconductorCandidates(100);
    for (const c of candidates) {
      const family = classifyFamily(c.formula);
      const entry = state.familyCandidateCounts.get(family) ?? { count: 0, formulas: [], elementSets: [], stoichiometries: [] };
      const counts = parseFormulaCounts(c.formula);
      const elSet = new Set(Object.keys(counts));
      if (isCompositionallyDistinct(elSet, counts, entry.elementSets, entry.stoichiometries)) {
        // Don't count known compounds toward corroboration during init
        const known = await isKnownCompound(c.formula);
        if (!known) {
          entry.count++;
          if (entry.formulas.length < 5) entry.formulas.push(c.formula);
        }
        entry.elementSets.push(elSet);
        entry.stoichiometries.push(counts);
      }
      state.familyCandidateCounts.set(family, entry);
      if (entry.count >= MIN_FAMILY_CORROBORATION) {
        state.knownFamilies.add(family);
      }
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
      if (m.type === "tc-record" && m.title) {
        const tcMatch = m.title.match(/(\d+)\s*K/);
        if (tcMatch) {
          const milestoneTC = Number(tcMatch[1]);
          if (milestoneTC > state.bestTcEver) state.bestTcEver = milestoneTC;
        }
      }
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
      // Restore previously confirmed families so we never re-fire the event.
      // This is the primary deduplication mechanism — even if in-memory state
      // is lost on restart, we rebuild from persisted milestones.
      if (m.type === "new-family" && m.title) {
        const familyMatch = m.title.match(/New material family confirmed:\s*(.+)/);
        if (familyMatch) {
          state.knownFamilies.add(familyMatch[1].trim());
        }
        // Also try matching the expansion event format
        const expandMatch = m.title.match(/(.+?)\s+family expanded/);
        if (expandMatch) {
          state.knownFamilies.add(expandMatch[1].trim());
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
      const entry = state.familyCandidateCounts.get(family) ?? { count: 0, formulas: [], elementSets: [], stoichiometries: [] };
      const counts = parseFormulaCounts(c.formula);
      const elSet = new Set(Object.keys(counts));
      if (!entry.formulas.includes(c.formula) && isCompositionallyDistinct(elSet, counts, entry.elementSets, entry.stoichiometries)) {
        // Only count NOVEL compounds as corroboration — skip known/textbook materials
        const known = await isKnownCompound(c.formula);
        if (!known) {
          entry.count++;
          if (entry.formulas.length < 5) entry.formulas.push(c.formula);
        }
        // Always track element sets/stoichiometries for deduplication, even for
        // known compounds (so future near-duplicates of known compounds are also caught)
        entry.elementSets.push(elSet);
        entry.stoichiometries.push(counts);
        state.familyCandidateCounts.set(family, entry);
      }
      // Family confirmation: fires ONCE per family, then never again.
      // After confirmation, expansion events track new additions.
      if (!state.knownFamilies.has(family) && entry.count >= MIN_FAMILY_CORROBORATION) {
        state.knownFamilies.add(family);
        detected.push({
          type: "new-family",
          title: `New material family confirmed: ${family}`,
          description: `${entry.count} novel candidates from the ${family} family corroborated (${entry.formulas.slice(0, 3).join(", ")}). Expanding chemical search space.`,
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
