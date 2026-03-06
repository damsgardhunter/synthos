import { storage } from "../storage";
import { extractFeatures, type MLFeatureVector } from "./ml-predictor";
import { classifyFamily } from "./utils";
import type { EventEmitter } from "./engine";

export interface PatternRuleCondition {
  property: string;
  operator: ">" | "<" | "between";
  threshold: number;
  upperThreshold?: number;
}

export interface PatternRule {
  conditions: PatternRuleCondition[];
  consequent: "high-tc" | "low-tc";
  confidence: number;
  f1Score: number;
  support: number;
  precision: number;
  recall: number;
  discoveredAt: string;
  weight: number;
}

const TOP_FEATURES: (keyof MLFeatureVector)[] = [
  "electronPhononLambda",
  "dosAtEF",
  "logPhononFreq",
  "correlationStrength",
  "metallicity",
  "valenceElectronConcentration",
  "dimensionalityScore",
  "nestingScore",
  "hydrogenRatio",
  "phononSofteningIndex",
];

const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  electronPhononLambda: "lambda",
  dosAtEF: "DOS_EF",
  logPhononFreq: "omegaLog",
  correlationStrength: "correlationStrength",
  metallicity: "metallicity",
  valenceElectronConcentration: "VEC",
  dimensionalityScore: "dimensionalityScore",
  nestingScore: "nestingScore",
  hydrogenRatio: "hydrogenRatio",
  phononSofteningIndex: "anharmonicity",
};

const TC_HIGH_THRESHOLD = 30;

let cachedRules: PatternRule[] = [];
let lastMineTime = 0;

interface ScoredCandidate {
  formula: string;
  predictedTc: number;
  features: MLFeatureVector;
  isHighTc: boolean;
}

function getFeatureValue(features: MLFeatureVector, property: string): number {
  return (features as any)[property] ?? 0;
}

function generateThresholds(values: number[]): number[] {
  if (values.length < 5) return [];
  const sorted = [...values].sort((a, b) => a - b);
  const percentiles = [0.2, 0.4, 0.5, 0.6, 0.8];
  const thresholds: number[] = [];
  for (const p of percentiles) {
    const idx = Math.floor(p * (sorted.length - 1));
    const val = sorted[idx];
    if (Number.isFinite(val) && !thresholds.includes(val)) {
      thresholds.push(val);
    }
  }
  return thresholds;
}

function evaluateRule(
  rule: PatternRuleCondition[],
  candidates: ScoredCandidate[],
  consequent: "high-tc" | "low-tc"
): { precision: number; recall: number; f1: number; support: number } {
  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (const c of candidates) {
    const matches = rule.every((cond) => {
      const val = getFeatureValue(c.features, cond.property);
      if (cond.operator === ">") return val > cond.threshold;
      if (cond.operator === "<") return val < cond.threshold;
      if (cond.operator === "between")
        return val > cond.threshold && val < (cond.upperThreshold ?? Infinity);
      return false;
    });

    const isTarget = consequent === "high-tc" ? c.isHighTc : !c.isHighTc;

    if (matches && isTarget) tp++;
    else if (matches && !isTarget) fp++;
    else if (!matches && isTarget) fn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const support = tp;

  return { precision, recall, f1, support };
}

export function mineQuantitativeRules(candidates: ScoredCandidate[]): PatternRule[] {
  if (candidates.length < 10) return [];

  const rules: PatternRule[] = [];
  const featureValues: Record<string, number[]> = {};

  for (const feat of TOP_FEATURES) {
    featureValues[feat] = candidates.map((c) => getFeatureValue(c.features, feat));
  }

  for (const feat of TOP_FEATURES) {
    const thresholds = generateThresholds(featureValues[feat]);

    for (const threshold of thresholds) {
      for (const operator of [">" as const, "<" as const]) {
        const condition: PatternRuleCondition = {
          property: feat,
          operator,
          threshold: Math.round(threshold * 1000) / 1000,
        };

        const result = evaluateRule([condition], candidates, "high-tc");
        if (result.f1 > 0.5 && result.support >= 3) {
          rules.push({
            conditions: [condition],
            consequent: "high-tc",
            confidence: result.precision,
            f1Score: Math.round(result.f1 * 1000) / 1000,
            support: result.support,
            precision: Math.round(result.precision * 1000) / 1000,
            recall: Math.round(result.recall * 1000) / 1000,
            discoveredAt: new Date().toISOString(),
            weight: 1.0,
          });
        }
      }
    }
  }

  rules.sort((a, b) => b.f1Score - a.f1Score);
  return rules.slice(0, 20);
}

export function mineInteractionRules(candidates: ScoredCandidate[]): PatternRule[] {
  if (candidates.length < 15) return [];

  const rules: PatternRule[] = [];
  const featureValues: Record<string, number[]> = {};

  for (const feat of TOP_FEATURES) {
    featureValues[feat] = candidates.map((c) => getFeatureValue(c.features, feat));
  }

  for (let i = 0; i < TOP_FEATURES.length; i++) {
    const feat1 = TOP_FEATURES[i];
    const thresholds1 = generateThresholds(featureValues[feat1]);

    for (let j = i + 1; j < TOP_FEATURES.length; j++) {
      const feat2 = TOP_FEATURES[j];
      const thresholds2 = generateThresholds(featureValues[feat2]);

      for (const t1 of thresholds1) {
        for (const t2 of thresholds2) {
          const conditions: PatternRuleCondition[] = [
            { property: feat1, operator: ">", threshold: Math.round(t1 * 1000) / 1000 },
            { property: feat2, operator: ">", threshold: Math.round(t2 * 1000) / 1000 },
          ];

          const result = evaluateRule(conditions, candidates, "high-tc");
          if (result.f1 > 0.5 && result.support >= 3) {
            rules.push({
              conditions,
              consequent: "high-tc",
              confidence: result.precision,
              f1Score: Math.round(result.f1 * 1000) / 1000,
              support: result.support,
              precision: Math.round(result.precision * 1000) / 1000,
              recall: Math.round(result.recall * 1000) / 1000,
              discoveredAt: new Date().toISOString(),
              weight: 1.0,
            });
          }
        }
      }
    }
  }

  rules.sort((a, b) => b.f1Score - a.f1Score);
  return rules.slice(0, 20);
}

export function validateRule(
  rule: PatternRule,
  testSet: ScoredCandidate[]
): { precision: number; recall: number; f1: number; support: number } {
  return evaluateRule(rule.conditions, testSet, rule.consequent);
}

export function applyRulesToScreen(
  candidates: ScoredCandidate[],
  rules: PatternRule[]
): { formula: string; theoryScore: number }[] {
  if (rules.length === 0) {
    return candidates.map((c) => ({ formula: c.formula, theoryScore: 0.5 }));
  }

  const totalWeight = rules.reduce((s, r) => s + r.weight, 0);
  if (totalWeight === 0) {
    return candidates.map((c) => ({ formula: c.formula, theoryScore: 0.5 }));
  }

  return candidates.map((c) => {
    let weightedSum = 0;

    for (const rule of rules) {
      const matches = rule.conditions.every((cond) => {
        const val = getFeatureValue(c.features, cond.property);
        if (cond.operator === ">") return val > cond.threshold;
        if (cond.operator === "<") return val < cond.threshold;
        if (cond.operator === "between")
          return val > cond.threshold && val < (cond.upperThreshold ?? Infinity);
        return false;
      });

      if (matches) {
        const ruleContribution = rule.consequent === "high-tc" ? rule.weight : -rule.weight;
        weightedSum += ruleContribution;
      }
    }

    const theoryScore = Math.max(0, Math.min(1, 0.5 + weightedSum / (2 * totalWeight)));
    return { formula: c.formula, theoryScore: Math.round(theoryScore * 1000) / 1000 };
  });
}

export function getMinedRules(): PatternRule[] {
  return cachedRules;
}

export async function evolveRules(emit: EventEmitter): Promise<PatternRule[]> {
  try {
    const allCandidates = await storage.getSuperconductorCandidates(500);
    if (allCandidates.length < 20) {
      return cachedRules;
    }

    const scored: ScoredCandidate[] = [];
    for (const c of allCandidates) {
      try {
        const features = extractFeatures(c.formula);
        scored.push({
          formula: c.formula,
          predictedTc: c.predictedTc ?? 0,
          features,
          isHighTc: (c.predictedTc ?? 0) >= TC_HIGH_THRESHOLD,
        });
      } catch {
        continue;
      }
    }

    if (scored.length < 15) return cachedRules;

    const shuffled = [...scored].sort(() => Math.random() - 0.5);
    const splitIdx = Math.floor(shuffled.length * 0.7);
    const trainSet = shuffled.slice(0, splitIdx);
    const testSet = shuffled.slice(splitIdx);

    const quantRules = mineQuantitativeRules(trainSet);
    const interRules = mineInteractionRules(trainSet);

    const allNewRules = [...quantRules, ...interRules];

    const validatedRules: PatternRule[] = [];
    for (const rule of allNewRules) {
      const testResult = validateRule(rule, testSet);
      if (testResult.f1 > 0.5 && testResult.support >= 2) {
        rule.f1Score = Math.round(
          ((rule.f1Score + testResult.f1) / 2) * 1000
        ) / 1000;
        rule.precision = Math.round(
          ((rule.precision + testResult.precision) / 2) * 1000
        ) / 1000;
        rule.recall = Math.round(
          ((rule.recall + testResult.recall) / 2) * 1000
        ) / 1000;
        validatedRules.push(rule);
      }
    }

    for (const existing of cachedRules) {
      existing.weight *= 0.95;
    }

    const ruleSignatures = new Set<string>();
    const mergedRules: PatternRule[] = [];

    for (const rule of validatedRules) {
      const sig = rule.conditions
        .map(
          (c) =>
            `${c.property}${c.operator}${c.threshold}${c.upperThreshold ?? ""}`
        )
        .sort()
        .join("|");

      if (!ruleSignatures.has(sig)) {
        ruleSignatures.add(sig);
        mergedRules.push(rule);
      }
    }

    for (const existing of cachedRules) {
      const sig = existing.conditions
        .map(
          (c) =>
            `${c.property}${c.operator}${c.threshold}${c.upperThreshold ?? ""}`
        )
        .sort()
        .join("|");

      if (!ruleSignatures.has(sig) && existing.weight >= 0.1) {
        ruleSignatures.add(sig);
        mergedRules.push(existing);
      }
    }

    mergedRules.sort((a, b) => b.f1Score * b.weight - a.f1Score * a.weight);
    cachedRules = mergedRules.filter((r) => r.weight >= 0.1).slice(0, 30);
    lastMineTime = Date.now();

    for (const rule of cachedRules.slice(0, 5)) {
      const condStr = rule.conditions
        .map((c) => {
          const name = FEATURE_DISPLAY_NAMES[c.property] || c.property;
          if (c.operator === "between")
            return `${c.threshold} < ${name} < ${c.upperThreshold}`;
          return `${name} ${c.operator} ${c.threshold}`;
        })
        .join(" AND ");

      emit("log", {
        phase: "engine",
        event: "Pattern rule discovered",
        detail: `High Tc when ${condStr} (F1=${rule.f1Score}, support=${rule.support}, weight=${rule.weight.toFixed(2)})`,
        dataSource: "Pattern Miner",
      });
    }

    emit("log", {
      phase: "engine",
      event: "Pattern mining complete",
      detail: `Mined ${quantRules.length} quantitative + ${interRules.length} interaction rules from ${scored.length} candidates. ${validatedRules.length} validated (F1>0.5). ${cachedRules.length} active rules total.`,
      dataSource: "Pattern Miner",
    });

    return cachedRules;
  } catch (err: any) {
    emit("log", {
      phase: "engine",
      event: "Pattern mining error",
      detail: err.message?.slice(0, 150) || "unknown",
      dataSource: "Pattern Miner",
    });
    return cachedRules;
  }
}

export function screenWithPatterns(
  formulas: string[]
): { formula: string; theoryScore: number }[] {
  if (cachedRules.length === 0) {
    return formulas.map((f) => ({ formula: f, theoryScore: 0.5 }));
  }

  const candidates: ScoredCandidate[] = [];
  for (const formula of formulas) {
    try {
      const features = extractFeatures(formula);
      candidates.push({
        formula,
        predictedTc: 0,
        features,
        isHighTc: false,
      });
    } catch {
      candidates.push({
        formula,
        predictedTc: 0,
        features: {} as MLFeatureVector,
        isHighTc: false,
      });
    }
  }

  return applyRulesToScreen(candidates, cachedRules);
}
