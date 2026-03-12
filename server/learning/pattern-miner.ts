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
  family?: string;
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
const MIN_FAMILY_SIZE = 10;
const WEIGHT_DECAY_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000;

let cachedRules: PatternRule[] = [];
let lastMineTime = 0;

interface ScoredCandidate {
  formula: string;
  predictedTc: number;
  features: MLFeatureVector;
  isHighTc: boolean;
  family: string;
}

type FeatureMatrix = Float64Array[];
interface PrecomputedData {
  matrix: FeatureMatrix;
  isHighTc: boolean[];
  featureIndices: Map<string, number>;
}

function buildFeatureMatrix(candidates: ScoredCandidate[]): PrecomputedData {
  const featureIndices = new Map<string, number>();
  TOP_FEATURES.forEach((f, i) => featureIndices.set(f as string, i));

  const n = candidates.length;
  const d = TOP_FEATURES.length;
  const matrix: FeatureMatrix = new Array(n);
  const isHighTc: boolean[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const row = new Float64Array(d);
    const feat = candidates[i].features;
    for (let j = 0; j < d; j++) {
      row[j] = (feat as any)[TOP_FEATURES[j]] ?? 0;
    }
    matrix[i] = row;
    isHighTc[i] = candidates[i].isHighTc;
  }

  return { matrix, isHighTc, featureIndices };
}

function jenksNaturalBreaks(values: number[], numBreaks: number): number[] {
  const sorted = [...values].filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  const n = sorted.length;
  if (n < numBreaks + 1) return [];

  const unique = [...new Set(sorted)];
  if (unique.length <= numBreaks) return unique.slice(0, -1);

  const k = Math.min(numBreaks, unique.length - 1);

  const mat1: number[][] = Array.from({ length: n }, () => new Array(k + 1).fill(Infinity));
  const mat2: number[][] = Array.from({ length: n }, () => new Array(k + 1).fill(0));

  for (let i = 0; i < n; i++) {
    let sumVals = 0;
    let sumSq = 0;
    for (let m = 0; m <= i; m++) {
      const val = sorted[i - m];
      sumVals += val;
      sumSq += val * val;
      const count = m + 1;
      const variance = sumSq - (sumVals * sumVals) / count;
      if (m === 0) {
        mat1[i][0] = variance;
        continue;
      }
      const baseIdx = i - m - 1;
      if (baseIdx >= 0) {
        for (let j = 1; j <= k; j++) {
          const cost = mat1[baseIdx][j - 1] + variance;
          if (cost < mat1[i][j]) {
            mat1[i][j] = cost;
            mat2[i][j] = i - m;
          }
        }
      }
    }
    mat1[i][0] = mat1[i][0] || 0;
  }

  const breaks: number[] = [];
  let idx = n - 1;
  for (let j = k; j >= 1; j--) {
    const breakIdx = mat2[idx][j];
    if (breakIdx > 0 && breakIdx < n) {
      const breakVal = (sorted[breakIdx - 1] + sorted[breakIdx]) / 2;
      breaks.unshift(breakVal);
    }
    idx = breakIdx - 1;
    if (idx < 0) break;
  }

  return breaks;
}

function generateThresholds(values: number[]): number[] {
  if (values.length < 5) return [];

  const jenks = jenksNaturalBreaks(values, 5);

  const sorted = [...values].sort((a, b) => a - b);
  const percentiles = [0.2, 0.4, 0.5, 0.6, 0.8];
  const percThresholds: number[] = [];
  for (const p of percentiles) {
    const idx = Math.floor(p * (sorted.length - 1));
    percThresholds.push(sorted[idx]);
  }

  const seen = new Set<number>();
  const combined: number[] = [];
  for (const t of [...jenks, ...percThresholds]) {
    if (Number.isFinite(t) && !seen.has(t)) {
      seen.add(t);
      combined.push(t);
    }
  }
  combined.sort((a, b) => a - b);
  return combined;
}

function computeDynamicSupport(candidateCount: number): number {
  if (candidateCount < 30) return 2;
  if (candidateCount < 100) return Math.max(2, Math.floor(candidateCount * 0.05));
  return Math.max(3, Math.floor(candidateCount * 0.1));
}

function computeMinF1(highTcRate: number): number {
  if (highTcRate > 0.85 || highTcRate < 0.15) return 0.6;
  return 0.5;
}

function evaluateRuleMatrix(
  conditionProps: string[],
  conditionOps: string[],
  conditionThresholds: number[],
  conditionUpperThresholds: (number | undefined)[],
  data: PrecomputedData,
  consequentIsHigh: boolean
): { precision: number; recall: number; f1: number; support: number } {
  const { matrix, isHighTc, featureIndices } = data;
  const n = matrix.length;
  const numConds = conditionProps.length;

  const colIndices = new Int32Array(numConds);
  for (let c = 0; c < numConds; c++) {
    colIndices[c] = featureIndices.get(conditionProps[c]) ?? -1;
  }

  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < n; i++) {
    const row = matrix[i];
    let matches = true;
    for (let c = 0; c < numConds; c++) {
      const colIdx = colIndices[c];
      if (colIdx < 0) { matches = false; break; }
      const val = row[colIdx];
      const op = conditionOps[c];
      if (op === ">") { if (!(val > conditionThresholds[c])) { matches = false; break; } }
      else if (op === "<") { if (!(val < conditionThresholds[c])) { matches = false; break; } }
      else if (op === "between") {
        if (!(val > conditionThresholds[c] && val < (conditionUpperThresholds[c] ?? Infinity))) {
          matches = false; break;
        }
      }
      else { matches = false; break; }
    }

    const isTarget = consequentIsHigh ? isHighTc[i] : !isHighTc[i];
    if (matches && isTarget) tp++;
    else if (matches && !isTarget) fp++;
    else if (!matches && isTarget) fn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1, support: tp };
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
      const val = (c.features as any)[cond.property] ?? 0;
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
  return { precision, recall, f1, support: tp };
}

export function mineQuantitativeRules(candidates: ScoredCandidate[], precomputed?: PrecomputedData): PatternRule[] {
  if (candidates.length < 10) return [];

  const data = precomputed ?? buildFeatureMatrix(candidates);
  const rules: PatternRule[] = [];
  const featureValues: Record<string, number[]> = {};

  for (let fi = 0; fi < TOP_FEATURES.length; fi++) {
    const feat = TOP_FEATURES[fi] as string;
    featureValues[feat] = [];
    for (let i = 0; i < data.matrix.length; i++) {
      featureValues[feat].push(data.matrix[i][fi]);
    }
  }

  const highTcRate = data.isHighTc.filter(h => h).length / data.isHighTc.length;
  const minF1 = computeMinF1(highTcRate);
  const minSupport = computeDynamicSupport(candidates.length);

  for (const feat of TOP_FEATURES) {
    const thresholds = generateThresholds(featureValues[feat as string]);

    for (const threshold of thresholds) {
      for (const operator of [">" as const, "<" as const]) {
        const result = evaluateRuleMatrix(
          [feat as string], [operator], [threshold], [undefined],
          data, true
        );
        if (result.f1 > minF1 && result.support >= minSupport) {
          rules.push({
            conditions: [{ property: feat as string, operator, threshold }],
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

const INTERACTION_OPERATORS: (">" | "<")[] = [">", "<"];

export function mineInteractionRules(candidates: ScoredCandidate[], precomputed?: PrecomputedData): PatternRule[] {
  if (candidates.length < 15) return [];

  const data = precomputed ?? buildFeatureMatrix(candidates);
  const rules: PatternRule[] = [];
  const featureValues: Record<string, number[]> = {};

  for (let fi = 0; fi < TOP_FEATURES.length; fi++) {
    const feat = TOP_FEATURES[fi] as string;
    featureValues[feat] = [];
    for (let i = 0; i < data.matrix.length; i++) {
      featureValues[feat].push(data.matrix[i][fi]);
    }
  }

  const highTcRate = data.isHighTc.filter(h => h).length / data.isHighTc.length;
  const minSupport = computeDynamicSupport(candidates.length);
  const minF1 = computeMinF1(highTcRate);

  for (let i = 0; i < TOP_FEATURES.length; i++) {
    const feat1 = TOP_FEATURES[i] as string;
    const thresholds1 = generateThresholds(featureValues[feat1]);

    for (let j = i + 1; j < TOP_FEATURES.length; j++) {
      const feat2 = TOP_FEATURES[j] as string;
      const thresholds2 = generateThresholds(featureValues[feat2]);

      for (const op1 of INTERACTION_OPERATORS) {
        for (const op2 of INTERACTION_OPERATORS) {
          let bestF1 = 0;
          let bestRule: PatternRule | null = null;

          for (const t1 of thresholds1) {
            for (const t2 of thresholds2) {
              const result = evaluateRuleMatrix(
                [feat1, feat2], [op1, op2], [t1, t2], [undefined, undefined],
                data, true
              );
              if (result.f1 > minF1 && result.support >= minSupport && result.f1 > bestF1) {
                bestF1 = result.f1;
                bestRule = {
                  conditions: [
                    { property: feat1, operator: op1, threshold: t1 },
                    { property: feat2, operator: op2, threshold: t2 },
                  ],
                  consequent: "high-tc",
                  confidence: result.precision,
                  f1Score: Math.round(result.f1 * 1000) / 1000,
                  support: result.support,
                  precision: Math.round(result.precision * 1000) / 1000,
                  recall: Math.round(result.recall * 1000) / 1000,
                  discoveredAt: new Date().toISOString(),
                  weight: 1.0,
                };
              }
            }
          }

          if (bestRule) rules.push(bestRule);
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
    const candidateFamily = c.family;

    for (const rule of rules) {
      if (rule.family && rule.family !== "all" && rule.family !== candidateFamily) continue;

      const matches = rule.conditions.every((cond) => {
        const val = (c.features as any)[cond.property] ?? 0;
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

function applyWeightDecay(rule: PatternRule, now: number): void {
  const discoveredMs = new Date(rule.discoveredAt).getTime();
  const age = now - discoveredMs;
  if (age > 0) {
    const decayFactor = Math.pow(0.5, age / WEIGHT_DECAY_HALF_LIFE_MS);
    rule.weight = Math.max(0.05, rule.weight * decayFactor);
  }
}

function ruleSignature(rule: PatternRule): string {
  const condSig = rule.conditions
    .map((c) => `${c.property}${c.operator}${c.threshold}${c.upperThreshold ?? ""}`)
    .sort()
    .join("|");
  return `${rule.family ?? "all"}::${condSig}`;
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
          family: classifyFamily(c.formula),
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

    const trainData = buildFeatureMatrix(trainSet);
    const quantRules = mineQuantitativeRules(trainSet, trainData);
    const interRules = mineInteractionRules(trainSet, trainData);

    const globalRules = [...quantRules, ...interRules];
    for (const r of globalRules) r.family = "all";

    const familyRules: PatternRule[] = [];
    const familyGroups = new Map<string, ScoredCandidate[]>();
    for (const c of trainSet) {
      const existing = familyGroups.get(c.family) ?? [];
      existing.push(c);
      familyGroups.set(c.family, existing);
    }

    for (const [family, members] of familyGroups) {
      if (members.length < MIN_FAMILY_SIZE) continue;
      const famData = buildFeatureMatrix(members);
      const famQuant = mineQuantitativeRules(members, famData);
      const famInter = mineInteractionRules(members, famData);
      for (const r of [...famQuant, ...famInter]) {
        r.family = family;
        familyRules.push(r);
      }
    }

    const allNewRules = [...globalRules, ...familyRules];

    const validatedRules: PatternRule[] = [];
    for (const rule of allNewRules) {
      const relevantTest = rule.family && rule.family !== "all"
        ? testSet.filter(c => c.family === rule.family)
        : testSet;

      if (relevantTest.length < 5) {
        if (rule.f1Score > 0.6 && rule.support >= 3) {
          validatedRules.push(rule);
        }
        continue;
      }

      const testResult = validateRule(rule, relevantTest);
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

    const now = Date.now();
    for (const existing of cachedRules) {
      applyWeightDecay(existing, now);
    }

    const ruleSignatures = new Set<string>();
    const mergedRules: PatternRule[] = [];

    for (const rule of validatedRules) {
      const sig = ruleSignature(rule);
      if (!ruleSignatures.has(sig)) {
        ruleSignatures.add(sig);
        mergedRules.push(rule);
      }
    }

    for (const existing of cachedRules) {
      const sig = ruleSignature(existing);
      if (!ruleSignatures.has(sig) && existing.weight >= 0.05) {
        ruleSignatures.add(sig);
        mergedRules.push(existing);
      }
    }

    mergedRules.sort((a, b) => b.f1Score * b.weight - a.f1Score * a.weight);
    cachedRules = mergedRules.filter((r) => r.weight >= 0.05).slice(0, 30);
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

      const familyTag = rule.family && rule.family !== "all" ? ` [${rule.family}]` : "";
      emit("log", {
        phase: "engine",
        event: "Pattern rule discovered",
        detail: `High Tc when ${condStr}${familyTag} (F1=${rule.f1Score}, support=${rule.support}, weight=${rule.weight.toFixed(2)})`,
        dataSource: "Pattern Miner",
      });
    }

    const familyCount = new Set(cachedRules.filter(r => r.family && r.family !== "all").map(r => r.family)).size;
    emit("log", {
      phase: "engine",
      event: "Pattern mining complete",
      detail: `Mined ${quantRules.length} quantitative + ${interRules.length} interaction rules from ${scored.length} candidates. ${familyRules.length} family-specific rules across ${familyCount} families. ${validatedRules.length} validated. ${cachedRules.length} active rules total.`,
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
        family: classifyFamily(formula),
      });
    } catch {
      candidates.push({
        formula,
        predictedTc: 0,
        features: {} as MLFeatureVector,
        isHighTc: false,
        family: "unknown",
      });
    }
  }

  return applyRulesToScreen(candidates, cachedRules);
}
