import { getFeatureDataset, type FeatureRecord, type PhysicsFeatureVector, FEATURE_NAMES } from "./physics-feature-db";

export type HypothesisStatus = "proposed" | "testing" | "supported" | "weakened" | "refuted";

export interface Hypothesis {
  id: string;
  statement: string;
  mathematicalForm: string;
  supportingEvidence: { formula: string; score: number }[];
  confidenceScore: number;
  testCount: number;
  supportCount: number;
  refuteCount: number;
  requiredConditions: string[];
  predictedTcRange: [number, number];
  discoveredAt: number;
  status: HypothesisStatus;
  lastTestedAt: number;
}

interface CorrelationPattern {
  feature1: string;
  feature2: string;
  correlation: number;
  highTcAssociation: number;
}

interface ConditionalRule {
  conditions: { feature: string; operator: ">" | "<" | ">=" | "<="; threshold: number }[];
  outcome: string;
  confidence: number;
  sampleCount: number;
  avgTc: number;
}

interface CoOccurrencePattern {
  features: string[];
  frequency: number;
  avgTc: number;
  maxTc: number;
}

const hypothesisStore: Map<string, Hypothesis> = new Map();
let nextHypothesisId = 1;

function generateId(): string {
  return `hyp-${nextHypothesisId++}-${Date.now().toString(36)}`;
}

function computeCorrelations(records: FeatureRecord[]): CorrelationPattern[] {
  const patterns: CorrelationPattern[] = [];
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 10) return patterns;

  const tcValues = withTc.map(r => r.tc!);
  const tcMean = tcValues.reduce((s, v) => s + v, 0) / tcValues.length;
  const tcStd = Math.sqrt(tcValues.reduce((s, v) => s + (v - tcMean) ** 2, 0) / tcValues.length);
  if (tcStd < 1e-6) return patterns;

  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    for (let j = i + 1; j < FEATURE_NAMES.length; j++) {
      const f1 = FEATURE_NAMES[i];
      const f2 = FEATURE_NAMES[j];
      const vals1 = withTc.map(r => r.featureVector[f1]);
      const vals2 = withTc.map(r => r.featureVector[f2]);

      const mean1 = vals1.reduce((s, v) => s + v, 0) / vals1.length;
      const mean2 = vals2.reduce((s, v) => s + v, 0) / vals2.length;
      const std1 = Math.sqrt(vals1.reduce((s, v) => s + (v - mean1) ** 2, 0) / vals1.length);
      const std2 = Math.sqrt(vals2.reduce((s, v) => s + (v - mean2) ** 2, 0) / vals2.length);

      if (std1 < 1e-6 || std2 < 1e-6) continue;

      let cov12 = 0;
      for (let k = 0; k < vals1.length; k++) {
        cov12 += (vals1[k] - mean1) * (vals2[k] - mean2);
      }
      cov12 /= vals1.length;
      const corr = cov12 / (std1 * std2);

      let covTc1 = 0;
      let covTc2 = 0;
      for (let k = 0; k < vals1.length; k++) {
        covTc1 += (vals1[k] - mean1) * (tcValues[k] - tcMean);
        covTc2 += (vals2[k] - mean2) * (tcValues[k] - tcMean);
      }
      covTc1 /= vals1.length;
      covTc2 /= vals2.length;
      const tcCorr1 = covTc1 / (std1 * tcStd);
      const tcCorr2 = covTc2 / (std2 * tcStd);

      const highTcAssociation = (Math.abs(tcCorr1) + Math.abs(tcCorr2)) / 2;

      if (Math.abs(corr) > 0.3 && highTcAssociation > 0.15) {
        patterns.push({ feature1: f1, feature2: f2, correlation: corr, highTcAssociation });
      }
    }
  }

  patterns.sort((a, b) => b.highTcAssociation - a.highTcAssociation);
  return patterns.slice(0, 30);
}

function discoverConditionalRules(records: FeatureRecord[]): ConditionalRule[] {
  const rules: ConditionalRule[] = [];
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 10) return rules;

  const overallAvgTc = withTc.reduce((s, r) => s + r.tc!, 0) / withTc.length;

  const thresholds: Record<string, number[]> = {};
  for (const fname of FEATURE_NAMES) {
    const vals = withTc.map(r => r.featureVector[fname]).sort((a, b) => a - b);
    thresholds[fname] = [
      vals[Math.floor(vals.length * 0.25)],
      vals[Math.floor(vals.length * 0.5)],
      vals[Math.floor(vals.length * 0.75)],
    ];
  }

  const keyFeatures = ["DOS_EF", "electron_phonon_lambda", "nesting_score", "pairing_strength",
    "phonon_log_frequency", "band_flatness", "correlation_strength", "orbital_degeneracy",
    "debye_temp", "hydrogen_density", "charge_transfer", "quantum_critical_score"];

  for (const f1 of keyFeatures) {
    if (!thresholds[f1]) continue;
    for (const t1 of thresholds[f1]) {
      const matchingHigh = withTc.filter(r => r.featureVector[f1 as keyof PhysicsFeatureVector] > t1);
      if (matchingHigh.length < 3) continue;
      const avgTcHigh = matchingHigh.reduce((s, r) => s + r.tc!, 0) / matchingHigh.length;
      if (avgTcHigh <= overallAvgTc * 1.2) continue;

      for (const f2 of keyFeatures) {
        if (f2 === f1 || !thresholds[f2]) continue;
        for (const t2 of thresholds[f2]) {
          const matching = withTc.filter(r => r.featureVector[f1 as keyof PhysicsFeatureVector] > t1 && r.featureVector[f2 as keyof PhysicsFeatureVector] > t2);
          if (matching.length < 3) continue;
          const avgTc = matching.reduce((s, r) => s + r.tc!, 0) / matching.length;
          if (avgTc <= overallAvgTc * 1.3) continue;

          const confidence = Math.min(1, (avgTc / overallAvgTc - 1) * matching.length / withTc.length * 10);
          if (confidence < 0.2) continue;

          rules.push({
            conditions: [
              { feature: f1, operator: ">", threshold: Number(t1.toFixed(3)) },
              { feature: f2, operator: ">", threshold: Number(t2.toFixed(3)) },
            ],
            outcome: `Tc likely > ${Math.round(avgTc * 0.7)}K`,
            confidence: Number(confidence.toFixed(3)),
            sampleCount: matching.length,
            avgTc: Number(avgTc.toFixed(1)),
          });
        }
      }
    }
  }

  rules.sort((a, b) => b.confidence * b.avgTc - a.confidence * a.avgTc);
  return rules.slice(0, 20);
}

function findCoOccurrencePatterns(records: FeatureRecord[]): CoOccurrencePattern[] {
  const patterns: CoOccurrencePattern[] = [];
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 10) return patterns;

  const overallAvgTc = withTc.reduce((s, r) => s + r.tc!, 0) / withTc.length;

  const featureMedians: Record<string, number> = {};
  for (const fname of FEATURE_NAMES) {
    const vals = withTc.map(r => r.featureVector[fname]).sort((a, b) => a - b);
    featureMedians[fname] = vals[Math.floor(vals.length / 2)];
  }

  const importantFeatures = ["DOS_EF", "electron_phonon_lambda", "nesting_score",
    "band_flatness", "pairing_strength", "orbital_degeneracy", "charge_transfer",
    "quantum_critical_score", "hydrogen_density"];

  for (let i = 0; i < importantFeatures.length; i++) {
    for (let j = i + 1; j < importantFeatures.length; j++) {
      for (let k = j + 1; k < importantFeatures.length; k++) {
        const f1 = importantFeatures[i];
        const f2 = importantFeatures[j];
        const f3 = importantFeatures[k];

        const matching = withTc.filter(r =>
          r.featureVector[f1 as keyof PhysicsFeatureVector] > featureMedians[f1] &&
          r.featureVector[f2 as keyof PhysicsFeatureVector] > featureMedians[f2] &&
          r.featureVector[f3 as keyof PhysicsFeatureVector] > featureMedians[f3]
        );

        if (matching.length < 3) continue;

        const avgTc = matching.reduce((s, r) => s + r.tc!, 0) / matching.length;
        const maxTc = Math.max(...matching.map(r => r.tc!));

        if (avgTc > overallAvgTc * 1.3) {
          patterns.push({
            features: [f1, f2, f3],
            frequency: matching.length / withTc.length,
            avgTc: Number(avgTc.toFixed(1)),
            maxTc: Number(maxTc.toFixed(1)),
          });
        }
      }
    }
  }

  patterns.sort((a, b) => b.avgTc * b.frequency - a.avgTc * a.frequency);
  return patterns.slice(0, 15);
}

function featureToReadable(f: string): string {
  const map: Record<string, string> = {
    DOS_EF: "high density of states at Fermi level",
    electron_phonon_lambda: "strong electron-phonon coupling",
    nesting_score: "strong Fermi surface nesting",
    band_flatness: "flat bands near Fermi level",
    pairing_strength: "strong pairing interaction",
    orbital_degeneracy: "orbital degeneracy",
    charge_transfer: "significant charge transfer",
    quantum_critical_score: "quantum criticality proximity",
    hydrogen_density: "high hydrogen content",
    phonon_log_frequency: "high phonon frequency",
    debye_temp: "high Debye temperature",
    correlation_strength: "strong electronic correlations",
    van_hove_distance: "van Hove singularity proximity",
    lattice_anisotropy: "lattice anisotropy",
    mott_proximity: "Mott insulator proximity",
    spin_fluctuation: "spin fluctuations",
    cdw_proximity: "CDW instability proximity",
    bandwidth: "narrow bandwidth",
    anharmonicity: "phonon anharmonicity",
    fermi_surface_dimensionality: "quasi-2D Fermi surface",
  };
  return map[f] || f.replace(/_/g, " ");
}

function generateHypothesesFromPatterns(
  correlations: CorrelationPattern[],
  rules: ConditionalRule[],
  coOccurrences: CoOccurrencePattern[],
): Hypothesis[] {
  const hypotheses: Hypothesis[] = [];

  for (const rule of rules.slice(0, 5)) {
    const condDescriptions = rule.conditions.map(c =>
      `${featureToReadable(c.feature)} ${c.operator} ${c.threshold}`
    );
    const statement = `Materials with ${condDescriptions.join(" AND ")} show enhanced superconducting Tc`;
    const mathForm = rule.conditions.map(c => `${c.feature}${c.operator}${c.threshold}`).join(" && ");

    hypotheses.push({
      id: generateId(),
      statement,
      mathematicalForm: mathForm,
      supportingEvidence: [],
      confidenceScore: rule.confidence * 0.5,
      testCount: 0,
      supportCount: 0,
      refuteCount: 0,
      requiredConditions: rule.conditions.map(c => `${c.feature}${c.operator}${c.threshold}`),
      predictedTcRange: [Math.round(rule.avgTc * 0.5), Math.round(rule.avgTc * 1.5)],
      discoveredAt: Date.now(),
      status: "proposed",
      lastTestedAt: 0,
    });
  }

  for (const pattern of coOccurrences.slice(0, 5)) {
    const featureDescs = pattern.features.map(f => featureToReadable(f));
    const statement = `Co-occurrence of ${featureDescs.join(", ")} correlates with enhanced Tc (avg ${pattern.avgTc}K)`;
    const mathForm = pattern.features.join(" * ");

    hypotheses.push({
      id: generateId(),
      statement,
      mathematicalForm: mathForm,
      supportingEvidence: [],
      confidenceScore: Math.min(0.8, pattern.frequency * pattern.avgTc / 100),
      testCount: 0,
      supportCount: 0,
      refuteCount: 0,
      requiredConditions: pattern.features.map(f => `${f}>median`),
      predictedTcRange: [Math.round(pattern.avgTc * 0.5), Math.round(pattern.maxTc)],
      discoveredAt: Date.now(),
      status: "proposed",
      lastTestedAt: 0,
    });
  }

  for (const corr of correlations.slice(0, 3)) {
    const dir = corr.correlation > 0 ? "positively" : "negatively";
    const statement = `${featureToReadable(corr.feature1)} and ${featureToReadable(corr.feature2)} are ${dir} correlated in high-Tc materials (r=${corr.correlation.toFixed(2)})`;
    const sign = corr.correlation > 0 ? "*" : "/";
    const mathForm = `Tc ~ ${corr.feature1} ${sign} ${corr.feature2}`;

    hypotheses.push({
      id: generateId(),
      statement,
      mathematicalForm: mathForm,
      supportingEvidence: [],
      confidenceScore: Math.abs(corr.highTcAssociation) * 0.6,
      testCount: 0,
      supportCount: 0,
      refuteCount: 0,
      requiredConditions: [`${corr.feature1}>0`, `${corr.feature2}>0`],
      predictedTcRange: [10, 200],
      discoveredAt: Date.now(),
      status: "proposed",
      lastTestedAt: 0,
    });
  }

  return hypotheses;
}

function testHypothesis(hypothesis: Hypothesis, records: FeatureRecord[]): {
  supportScore: number;
  testedMaterials: { formula: string; score: number }[];
} {
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 5) return { supportScore: 0.5, testedMaterials: [] };

  const overallAvgTc = withTc.reduce((s, r) => s + r.tc!, 0) / withTc.length;
  const testedMaterials: { formula: string; score: number }[] = [];
  let supportCount = 0;
  let testCount = 0;

  const featureMedians: Record<string, number> = {};
  for (const fname of FEATURE_NAMES) {
    const vals = withTc.map(r => r.featureVector[fname]).sort((a, b) => a - b);
    featureMedians[fname] = vals[Math.floor(vals.length / 2)];
  }

  for (const record of withTc) {
    let meetsConditions = true;

    for (const cond of hypothesis.requiredConditions) {
      const matchOp = cond.match(/^(\w+)(>=|<=|>|<)(.+)$/);
      if (!matchOp) continue;

      const [, feature, operator, thresholdStr] = matchOp;
      const featureKey = feature as keyof PhysicsFeatureVector;
      if (!(featureKey in record.featureVector)) continue;

      const val = record.featureVector[featureKey];
      let threshold: number;

      if (thresholdStr === "median") {
        threshold = featureMedians[feature] ?? 0;
      } else {
        threshold = parseFloat(thresholdStr);
      }

      if (isNaN(threshold)) continue;

      switch (operator) {
        case ">": if (!(val > threshold)) meetsConditions = false; break;
        case "<": if (!(val < threshold)) meetsConditions = false; break;
        case ">=": if (!(val >= threshold)) meetsConditions = false; break;
        case "<=": if (!(val <= threshold)) meetsConditions = false; break;
      }
    }

    if (!meetsConditions) continue;

    testCount++;
    const tc = record.tc!;
    const inPredictedRange = tc >= hypothesis.predictedTcRange[0] && tc <= hypothesis.predictedTcRange[1];
    const aboveAvg = tc > overallAvgTc;
    const score = (inPredictedRange ? 0.6 : 0) + (aboveAvg ? 0.4 : 0);

    if (score > 0.5) supportCount++;

    testedMaterials.push({ formula: record.formula, score: Number(score.toFixed(3)) });
  }

  const supportScore = testCount > 0 ? supportCount / testCount : 0.5;
  return { supportScore: Number(supportScore.toFixed(3)), testedMaterials };
}

function bayesianUpdate(priorConfidence: number, supportScore: number, sampleSize: number): number {
  const weight = Math.min(1, sampleSize / 20);
  const likelihood = supportScore;
  const updatedConfidence = priorConfidence * (1 - weight) + likelihood * weight;
  return Number(Math.max(0, Math.min(1, updatedConfidence)).toFixed(4));
}

function updateHypothesisStatus(hypothesis: Hypothesis): void {
  if (hypothesis.testCount < 3) {
    hypothesis.status = hypothesis.testCount > 0 ? "testing" : "proposed";
    return;
  }

  const supportRatio = hypothesis.supportCount / Math.max(1, hypothesis.testCount);

  if (supportRatio >= 0.7 && hypothesis.confidenceScore >= 0.6) {
    hypothesis.status = "supported";
  } else if (supportRatio <= 0.3 && hypothesis.testCount >= 5) {
    hypothesis.status = "refuted";
  } else if (supportRatio < 0.5 && hypothesis.confidenceScore < 0.4) {
    hypothesis.status = "weakened";
  } else {
    hypothesis.status = "testing";
  }
}

export function runHypothesisCycle(): {
  newHypotheses: number;
  testedHypotheses: number;
  activeCount: number;
  supportedCount: number;
  refutedCount: number;
} {
  const records = getFeatureDataset();
  if (records.length < 10) {
    return { newHypotheses: 0, testedHypotheses: 0, activeCount: hypothesisStore.size, supportedCount: 0, refutedCount: 0 };
  }

  const correlations = computeCorrelations(records);
  const rules = discoverConditionalRules(records);
  const coOccurrences = findCoOccurrencePatterns(records);

  let newHypotheses = 0;

  if (hypothesisStore.size < 30) {
    const candidates = generateHypothesesFromPatterns(correlations, rules, coOccurrences);

    for (const candidate of candidates) {
      const isDuplicate = Array.from(hypothesisStore.values()).some(existing =>
        existing.mathematicalForm === candidate.mathematicalForm ||
        (existing.requiredConditions.length === candidate.requiredConditions.length &&
          existing.requiredConditions.every(c => candidate.requiredConditions.includes(c)))
      );

      if (!isDuplicate) {
        hypothesisStore.set(candidate.id, candidate);
        newHypotheses++;
      }
    }
  }

  let testedHypotheses = 0;
  const activeHypotheses = Array.from(hypothesisStore.values()).filter(h =>
    h.status !== "refuted" && h.status !== "supported"
  );

  for (const hypothesis of activeHypotheses) {
    const result = testHypothesis(hypothesis, records);

    hypothesis.testCount++;
    hypothesis.lastTestedAt = Date.now();

    const supportMaterials = result.testedMaterials.filter(m => m.score > 0.5);
    const refuteMaterials = result.testedMaterials.filter(m => m.score <= 0.5);
    hypothesis.supportCount += supportMaterials.length;
    hypothesis.refuteCount += refuteMaterials.length;

    for (const m of supportMaterials.slice(0, 5)) {
      if (!hypothesis.supportingEvidence.some(e => e.formula === m.formula)) {
        hypothesis.supportingEvidence.push(m);
      }
    }
    if (hypothesis.supportingEvidence.length > 20) {
      hypothesis.supportingEvidence.sort((a, b) => b.score - a.score);
      hypothesis.supportingEvidence = hypothesis.supportingEvidence.slice(0, 20);
    }

    hypothesis.confidenceScore = bayesianUpdate(
      hypothesis.confidenceScore,
      result.supportScore,
      result.testedMaterials.length,
    );

    updateHypothesisStatus(hypothesis);
    testedHypotheses++;
  }

  if (hypothesisStore.size > 50) {
    const sorted = Array.from(hypothesisStore.entries())
      .sort((a, b) => {
        if (a[1].status === "refuted" && b[1].status !== "refuted") return 1;
        if (a[1].status !== "refuted" && b[1].status === "refuted") return -1;
        return b[1].confidenceScore - a[1].confidenceScore;
      });
    const toRemove = sorted.slice(40);
    for (const [id] of toRemove) {
      hypothesisStore.delete(id);
    }
  }

  const all = Array.from(hypothesisStore.values());

  return {
    newHypotheses,
    testedHypotheses,
    activeCount: all.filter(h => h.status === "proposed" || h.status === "testing").length,
    supportedCount: all.filter(h => h.status === "supported").length,
    refutedCount: all.filter(h => h.status === "refuted").length,
  };
}

export function getActiveHypotheses(): Hypothesis[] {
  return Array.from(hypothesisStore.values())
    .filter(h => h.status !== "refuted")
    .sort((a, b) => b.confidenceScore - a.confidenceScore);
}

export function getAllHypotheses(): Hypothesis[] {
  return Array.from(hypothesisStore.values())
    .sort((a, b) => b.confidenceScore - a.confidenceScore);
}

export function testHypothesisById(id: string): {
  hypothesis: Hypothesis | null;
  testResult: { supportScore: number; testedMaterials: { formula: string; score: number }[] } | null;
} {
  const hypothesis = hypothesisStore.get(id);
  if (!hypothesis) return { hypothesis: null, testResult: null };

  const records = getFeatureDataset();
  const testResult = testHypothesis(hypothesis, records);

  hypothesis.testCount++;
  hypothesis.lastTestedAt = Date.now();

  const supportMaterials = testResult.testedMaterials.filter(m => m.score > 0.5);
  const refuteMaterials = testResult.testedMaterials.filter(m => m.score <= 0.5);
  hypothesis.supportCount += supportMaterials.length;
  hypothesis.refuteCount += refuteMaterials.length;

  for (const m of supportMaterials.slice(0, 5)) {
    if (!hypothesis.supportingEvidence.some(e => e.formula === m.formula)) {
      hypothesis.supportingEvidence.push(m);
    }
  }

  hypothesis.confidenceScore = bayesianUpdate(
    hypothesis.confidenceScore,
    testResult.supportScore,
    testResult.testedMaterials.length,
  );

  updateHypothesisStatus(hypothesis);

  return { hypothesis, testResult };
}

export function getTopHypothesesForGeneratorBias(): {
  preferredConditions: string[];
  featureTargets: Record<string, number>;
} {
  const supported = Array.from(hypothesisStore.values())
    .filter(h => h.status === "supported" || (h.status === "testing" && h.confidenceScore > 0.5))
    .sort((a, b) => b.confidenceScore - a.confidenceScore)
    .slice(0, 5);

  const preferredConditions: string[] = [];
  const featureTargets: Record<string, number> = {};

  for (const hyp of supported) {
    for (const cond of hyp.requiredConditions) {
      if (!preferredConditions.includes(cond)) {
        preferredConditions.push(cond);
      }
      const matchOp = cond.match(/^(\w+)(>=|<=|>|<)(.+)$/);
      if (matchOp) {
        const [, feature, , thresholdStr] = matchOp;
        const threshold = parseFloat(thresholdStr);
        if (!isNaN(threshold)) {
          featureTargets[feature] = Math.max(featureTargets[feature] ?? 0, threshold);
        }
      }
    }
  }

  return { preferredConditions, featureTargets };
}

export function getHypothesisStats(): {
  total: number;
  proposed: number;
  testing: number;
  supported: number;
  weakened: number;
  refuted: number;
  avgConfidence: number;
  topHypothesis: Hypothesis | null;
} {
  const all = Array.from(hypothesisStore.values());
  const avgConfidence = all.length > 0
    ? all.reduce((s, h) => s + h.confidenceScore, 0) / all.length
    : 0;

  const sorted = [...all].sort((a, b) => b.confidenceScore - a.confidenceScore);

  return {
    total: all.length,
    proposed: all.filter(h => h.status === "proposed").length,
    testing: all.filter(h => h.status === "testing").length,
    supported: all.filter(h => h.status === "supported").length,
    weakened: all.filter(h => h.status === "weakened").length,
    refuted: all.filter(h => h.status === "refuted").length,
    avgConfidence: Number(avgConfidence.toFixed(4)),
    topHypothesis: sorted[0] ?? null,
  };
}
