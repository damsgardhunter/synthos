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

const TC_HIGH_THRESHOLD_DEFAULT = 30;
const MIN_FAMILY_SIZE = 10;
const WEIGHT_DECAY_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000;

const FAMILY_TC_THRESHOLDS: Record<string, number> = {
  "hydride": 150,
  "cuprate": 60,
  "iron-based": 25,
  "heavy-fermion": 3,
  "nickelate": 15,
  "nitride": 12,
  "boride": 20,
  "carbide": 10,
  "chalcogenide": 8,
  "bismuthate": 20,
  "organic": 10,
  "oxide": 15,
};

function getFamilyTcThreshold(family: string): number {
  return FAMILY_TC_THRESHOLDS[family] ?? TC_HIGH_THRESHOLD_DEFAULT;
}

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

  const consequents: ("high-tc" | "low-tc")[] = ["high-tc", "low-tc"];

  for (const feat of TOP_FEATURES) {
    const thresholds = generateThresholds(featureValues[feat as string]);

    for (const threshold of thresholds) {
      for (const operator of [">" as const, "<" as const]) {
        for (const consequent of consequents) {
          const result = evaluateRuleMatrix(
            [feat as string], [operator], [threshold], [undefined],
            data, consequent === "high-tc"
          );
          if (result.f1 > minF1 && result.support >= minSupport) {
            rules.push({
              conditions: [{ property: feat as string, operator, threshold }],
              consequent,
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
  return rules.slice(0, 30);
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
          for (const consequent of ["high-tc" as const, "low-tc" as const]) {
            let bestF1 = 0;
            let bestRule: PatternRule | null = null;

            for (const t1 of thresholds1) {
              for (const t2 of thresholds2) {
                const result = evaluateRuleMatrix(
                  [feat1, feat2], [op1, op2], [t1, t2], [undefined, undefined],
                  data, consequent === "high-tc"
                );
                if (result.f1 > minF1 && result.support >= minSupport && result.f1 > bestF1) {
                  bestF1 = result.f1;
                  bestRule = {
                    conditions: [
                      { property: feat1, operator: op1, threshold: t1 },
                      { property: feat2, operator: op2, threshold: t2 },
                    ],
                    consequent,
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
  return `${rule.family ?? "all"}::${rule.consequent}::${condSig}`;
}

export async function evolveRules(emit: EventEmitter): Promise<PatternRule[]> {
  try {
    const allCandidates = await storage.getSuperconductorCandidates(500);
    if (allCandidates.length < 20) {
      return cachedRules;
    }

    const scored: ScoredCandidate[] = [];
    const uncached: { id: string; features: MLFeatureVector; existingMl: Record<string, any> }[] = [];
    for (const c of allCandidates) {
      try {
        const existingMl = (c.mlFeatures as Record<string, any>) ?? {};
        const cachedFeatures = existingMl.patternMinerFeatures as MLFeatureVector | undefined;
        const features = cachedFeatures ?? await extractFeatures(c.formula);
        if (!cachedFeatures && c.id) {
          uncached.push({ id: c.id, features, existingMl });
        }
        const family = classifyFamily(c.formula);
        const tc = c.predictedTc ?? 0;
        scored.push({
          formula: c.formula,
          predictedTc: tc,
          features,
          isHighTc: tc >= getFamilyTcThreshold(family),
          family,
        });
      } catch {
        continue;
      }
    }

    if (uncached.length > 0) {
      Promise.resolve().then(async () => {
        for (const { id, features, existingMl } of uncached.slice(0, 100)) {
          try {
            await storage.updateSuperconductorCandidate(id, {
              mlFeatures: { ...existingMl, patternMinerFeatures: features },
            });
          } catch {}
        }
      }).catch(() => {});
    }

    if (scored.length < 15) return cachedRules;

    const shuffled = [...scored].sort(() => Math.random() - 0.5);
    const useKFold = shuffled.length < 80;
    const K = useKFold ? Math.min(5, Math.max(3, Math.floor(shuffled.length / 10))) : 1;

    const folds: { train: ScoredCandidate[]; test: ScoredCandidate[] }[] = [];
    if (useKFold) {
      const foldSize = Math.floor(shuffled.length / K);
      for (let f = 0; f < K; f++) {
        const testStart = f * foldSize;
        const testEnd = f === K - 1 ? shuffled.length : testStart + foldSize;
        folds.push({
          train: [...shuffled.slice(0, testStart), ...shuffled.slice(testEnd)],
          test: shuffled.slice(testStart, testEnd),
        });
      }
    } else {
      const splitIdx = Math.floor(shuffled.length * 0.7);
      folds.push({
        train: shuffled.slice(0, splitIdx),
        test: shuffled.slice(splitIdx),
      });
    }

    const ruleScoreAccum = new Map<string, { rule: PatternRule; f1Sum: number; precSum: number; recSum: number; foldCount: number }>();

    for (const { train: trainSet, test: testSet } of folds) {
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

      const allFoldRules = [...globalRules, ...familyRules];

      for (const rule of allFoldRules) {
        const relevantTest = rule.family && rule.family !== "all"
          ? testSet.filter(c => c.family === rule.family)
          : testSet;

        if (relevantTest.length < 3) continue;

        const testResult = validateRule(rule, relevantTest);
        const sig = ruleSignature(rule);
        const existing = ruleScoreAccum.get(sig);

        if (existing) {
          existing.f1Sum += testResult.f1;
          existing.precSum += testResult.precision;
          existing.recSum += testResult.recall;
          existing.foldCount++;
        } else {
          ruleScoreAccum.set(sig, {
            rule: { ...rule },
            f1Sum: testResult.f1,
            precSum: testResult.precision,
            recSum: testResult.recall,
            foldCount: 1,
          });
        }
      }
    }

    const validatedRules: PatternRule[] = [];
    const minFoldsRequired = useKFold ? Math.max(2, Math.floor(K / 2)) : 1;

    for (const [, accum] of ruleScoreAccum) {
      if (accum.foldCount < minFoldsRequired) continue;

      const avgF1 = accum.f1Sum / accum.foldCount;
      const avgPrec = accum.precSum / accum.foldCount;
      const avgRec = accum.recSum / accum.foldCount;

      if (avgF1 > 0.5) {
        const rule = accum.rule;
        rule.f1Score = Math.round(((rule.f1Score + avgF1) / 2) * 1000) / 1000;
        rule.precision = Math.round(((rule.precision + avgPrec) / 2) * 1000) / 1000;
        rule.recall = Math.round(((rule.recall + avgRec) / 2) * 1000) / 1000;
        validatedRules.push(rule);
      }
    }

    const now = Date.now();
    const validatedSigs = new Set(validatedRules.map(r => ruleSignature(r)));

    const ruleSignaturesSet = new Set<string>();
    const mergedRules: PatternRule[] = [];

    for (const rule of validatedRules) {
      const sig = ruleSignature(rule);
      if (!ruleSignaturesSet.has(sig)) {
        ruleSignaturesSet.add(sig);
        mergedRules.push(rule);
      }
    }

    const lastTestSet = folds[folds.length - 1].test;
    for (const existing of cachedRules) {
      const sig = ruleSignature(existing);
      if (ruleSignaturesSet.has(sig)) continue;

      const relevantTest = existing.family && existing.family !== "all"
        ? lastTestSet.filter(c => c.family === existing.family)
        : lastTestSet;

      if (relevantTest.length >= 3) {
        const retestResult = validateRule(existing, relevantTest);
        if (retestResult.f1 < 0.3) {
          applyWeightDecay(existing, now);
        }
      } else {
        applyWeightDecay(existing, now);
      }

      if (existing.weight >= 0.05) {
        ruleSignaturesSet.add(sig);
        mergedRules.push(existing);
      }
    }

    const prevRuleCount = cachedRules.length;
    mergedRules.sort((a, b) => b.f1Score * b.weight - a.f1Score * a.weight);
    cachedRules = mergedRules.filter((r) => r.weight >= 0.05).slice(0, 30);
    lastMineTime = Date.now();

    const newlyDiscovered = validatedRules.filter(r => {
      const sig = ruleSignature(r);
      return !ruleSignaturesSet.has(sig) || !cachedRules.some(cr => ruleSignature(cr) === sig && cr.discoveredAt !== r.discoveredAt);
    });

    if (newlyDiscovered.length > 0) {
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
        const label = rule.consequent === "high-tc" ? "High Tc" : "Low Tc";
        emit("log", {
          phase: "engine",
          event: "Pattern rule discovered",
          detail: `${label} when ${condStr}${familyTag} (F1=${rule.f1Score}, support=${rule.support}, weight=${rule.weight.toFixed(2)})`,
          dataSource: "Pattern Miner",
        });
      }

      const familyCount = new Set(cachedRules.filter(r => r.family && r.family !== "all").map(r => r.family)).size;
      const highTcRuleCount = cachedRules.filter(r => r.consequent === "high-tc").length;
      const lowTcRuleCount = cachedRules.filter(r => r.consequent === "low-tc").length;
      const validationMethod = useKFold ? `${K}-fold CV` : "70/30 split";
      emit("log", {
        phase: "engine",
        event: "Pattern mining complete",
        detail: `${validatedRules.length} rules validated (${validationMethod}) from ${scored.length} candidates. ${highTcRuleCount} high-Tc + ${lowTcRuleCount} low-Tc rules across ${familyCount} families. ${cachedRules.length} active rules total.`,
        dataSource: "Pattern Miner",
      });
    }

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

const SCREEN_YIELD_BATCH = 50;

export async function screenWithPatterns(
  formulas: string[]
): Promise<{ formula: string; theoryScore: number }[]> {
  if (cachedRules.length === 0) {
    return formulas.map((f) => ({ formula: f, theoryScore: 0.5 }));
  }

  const candidates: ScoredCandidate[] = [];
  for (let i = 0; i < formulas.length; i++) {
    const formula = formulas[i];
    try {
      const features = await extractFeatures(formula);
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
    if ((i + 1) % SCREEN_YIELD_BATCH === 0) {
      await new Promise<void>(resolve => setImmediate(resolve));
    }
  }

  return applyRulesToScreen(candidates, cachedRules);
}
