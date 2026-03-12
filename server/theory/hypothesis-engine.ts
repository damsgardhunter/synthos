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
  conservativeTc: number;
}

interface CoOccurrencePattern {
  features: string[];
  featureRangeLabels: string[];
  frequency: number;
  avgTc: number;
  maxTc: number;
}

const HYPOTHESIS_CONFIG = {
  minSampleFraction: 0.05,
  minSampleFloor: 5,
  tcLiftThresholdSingle: 1.2,
  tcLiftThresholdPair: 1.3,
  tcLiftThresholdCoOccurrence: 1.3,
  minConfidence: 0.15,
  conservativeTcPenalty: 0.5,
  maxRules: 20,
  maxCoOccurrencePatterns: 15,
  maxCorrelationPatterns: 30,
  maxCoOccurrenceFeatures: 12,
  minFeatureCorrelation: 0.05,
  aprioriSingleLiftThreshold: 1.15,
  bonferroniAlphaFamily: 0.05,
  correlationThreshold: 0.3,
  tcAssociationThreshold: 0.15,
} as const;

const hypothesisStore: Map<string, Hypothesis> = new Map();
let nextHypothesisId = 1;

function generateId(): string {
  return `hyp-${nextHypothesisId++}-${Date.now().toString(36)}`;
}

function toRanks(x: number[]): number[] {
  const n = x.length;
  const indexed = x.map((val, i) => ({ val, i }));
  indexed.sort((a, b) => a.val - b.val);
  const ranks = new Array(n);
  let pos = 0;
  while (pos < n) {
    let end = pos + 1;
    while (end < n && indexed[end].val === indexed[pos].val) end++;
    const avgRank = (pos + end - 1) / 2;
    for (let k = pos; k < end; k++) ranks[indexed[k].i] = avgRank;
    pos = end;
  }
  return ranks;
}

function pearsonR(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] - mx;
    const yi = y[i] - my;
    num += xi * yi;
    dx += xi * xi;
    dy += yi * yi;
  }
  if (dx < 1e-10 || dy < 1e-10) return 0;
  const denom = Math.sqrt(dx) * Math.sqrt(dy);
  if (denom < 1e-10) return 0;
  return Math.max(-1, Math.min(1, num / denom));
}

function robustCorrelation(x: number[], y: number[]): number {
  const p = pearsonR(x, y);
  const s = pearsonR(toRanks(x), toRanks(y));
  return Math.abs(s) > Math.abs(p) ? s : p;
}

function computeCorrelations(records: FeatureRecord[]): CorrelationPattern[] {
  const patterns: CorrelationPattern[] = [];
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 10) return patterns;

  const tcValues = withTc.map(r => r.tc!);
  const tcRanks = toRanks(tcValues);

  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    for (let j = i + 1; j < FEATURE_NAMES.length; j++) {
      const f1 = FEATURE_NAMES[i];
      const f2 = FEATURE_NAMES[j];
      const vals1 = withTc.map(r => r.featureVector[f1]);
      const vals2 = withTc.map(r => r.featureVector[f2]);

      const corr = robustCorrelation(vals1, vals2);

      const tcCorr1 = robustCorrelation(vals1, tcValues);
      const tcCorr2 = robustCorrelation(vals2, tcValues);

      const highTcAssociation = (Math.abs(tcCorr1) + Math.abs(tcCorr2)) / 2;

      if (Math.abs(corr) > HYPOTHESIS_CONFIG.correlationThreshold && highTcAssociation > HYPOTHESIS_CONFIG.tcAssociationThreshold) {
        patterns.push({ feature1: f1, feature2: f2, correlation: corr, highTcAssociation });
      }
    }
  }

  patterns.sort((a, b) => b.highTcAssociation - a.highTcAssociation);
  return patterns.slice(0, HYPOTHESIS_CONFIG.maxCorrelationPatterns);
}

function findElbowThresholds(vals: number[]): number[] {
  const sorted = [...vals].sort((a, b) => a - b);
  const n = sorted.length;
  if (n < 6) {
    return [sorted[Math.floor(n / 2)]];
  }

  const gaps: { idx: number; gap: number }[] = [];
  for (let i = 1; i < n; i++) {
    gaps.push({ idx: i, gap: sorted[i] - sorted[i - 1] });
  }
  gaps.sort((a, b) => b.gap - a.gap);

  const thresholds: number[] = [];
  const usedRegions = new Set<number>();
  for (const g of gaps.slice(0, 5)) {
    const region = Math.floor(g.idx / n * 4);
    if (usedRegions.has(region)) continue;
    usedRegions.add(region);
    const threshold = (sorted[g.idx - 1] + sorted[g.idx]) / 2;
    const tooClose = thresholds.some(t => Math.abs(t - threshold) < (sorted[n - 1] - sorted[0]) * 0.05);
    if (tooClose) continue;
    thresholds.push(threshold);
    if (thresholds.length >= 3) break;
  }

  if (thresholds.length === 0) {
    thresholds.push(sorted[Math.floor(n / 2)]);
  }

  return thresholds.sort((a, b) => a - b);
}

function lnGamma(z: number): number {
  if (z <= 0) return 0;
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179, -5.395239384953e-6];
  let x = z;
  let y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function regularizedIncompleteBeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedIncompleteBeta(b, a, 1 - x);
  }
  const lnPre = a * Math.log(x) + b * Math.log(1 - x) - Math.log(a) - lnGamma(a) - lnGamma(b) + lnGamma(a + b);
  const front = Math.exp(lnPre);
  let num = 1;
  let den = 1;
  let f = 1;
  for (let m = 1; m <= 200; m++) {
    const d2m = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    f = 1 + d2m / f; if (Math.abs(f) < 1e-30) f = 1e-30;
    num = 1 + d2m / num; if (Math.abs(num) < 1e-30) num = 1e-30;
    f = 1 / f;
    den *= f * num;
    const d2m1 = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    f = 1 + d2m1 / f; if (Math.abs(f) < 1e-30) f = 1e-30;
    num = 1 + d2m1 / num; if (Math.abs(num) < 1e-30) num = 1e-30;
    f = 1 / f;
    const delta = f * num;
    den *= delta;
    if (Math.abs(delta - 1) < 1e-10) break;
  }
  return Math.max(0, Math.min(1, front * den));
}

function studentTCdf(t: number, df: number): number {
  if (df <= 0) return 0.5;
  const x = df / (df + t * t);
  const ibeta = regularizedIncompleteBeta(df / 2, 0.5, x);
  const p = 0.5 * ibeta;
  return t > 0 ? 1 - p : p;
}

function welchOneSidedPValue(
  groupMean: number, groupStd: number, groupN: number,
  compMean: number, compStd: number, compN: number,
): number {
  if (groupN < 2 || compN < 2) return 1;
  const se = Math.sqrt((groupStd ** 2) / groupN + (compStd ** 2) / compN);
  if (se < 1e-10) return groupMean > compMean ? 0 : 1;
  const t = (groupMean - compMean) / se;
  const v1 = (groupStd ** 2) / groupN;
  const v2 = (compStd ** 2) / compN;
  const dfNum = (v1 + v2) ** 2;
  const dfDen = (v1 ** 2) / (groupN - 1) + (v2 ** 2) / (compN - 1);
  const df = dfDen > 0 ? dfNum / dfDen : 1;
  return 1 - studentTCdf(t, df);
}

function discoverConditionalRules(records: FeatureRecord[]): ConditionalRule[] {
  const rules: ConditionalRule[] = [];
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 10) return rules;

  const minSamples = Math.max(HYPOTHESIS_CONFIG.minSampleFloor, Math.ceil(withTc.length * HYPOTHESIS_CONFIG.minSampleFraction));
  const overallAvgTc = withTc.reduce((s, r) => s + r.tc!, 0) / withTc.length;
  const overallSumTc = withTc.reduce((s, r) => s + r.tc!, 0);

  const thresholds: Record<string, number[]> = {};
  for (const fname of FEATURE_NAMES) {
    const vals = withTc.map(r => r.featureVector[fname]);
    thresholds[fname] = findElbowThresholds(vals);
  }

  const keyFeatures: (keyof PhysicsFeatureVector)[] = FEATURE_NAMES.filter(f => thresholds[f] && thresholds[f].length > 0);

  let totalTests = 0;
  for (const f1 of keyFeatures) {
    for (const f2 of keyFeatures) {
      if (f2 === f1) continue;
      totalTests += thresholds[f1].length * thresholds[f2].length;
    }
  }
  const bonferroniAlpha = HYPOTHESIS_CONFIG.bonferroniAlphaFamily / Math.max(1, totalTests);

  const featureVecs = new Map<string, Float64Array>();
  for (const f of keyFeatures) {
    const arr = new Float64Array(withTc.length);
    for (let i = 0; i < withTc.length; i++) {
      arr[i] = withTc[i].featureVector[f];
    }
    featureVecs.set(f, arr);
  }

  const tcArr = new Float64Array(withTc.length);
  for (let i = 0; i < withTc.length; i++) tcArr[i] = withTc[i].tc!;

  for (const f1 of keyFeatures) {
    const vec1 = featureVecs.get(f1)!;
    for (const t1 of thresholds[f1]) {
      const mask1 = new Uint8Array(withTc.length);
      let count1 = 0;
      let sumTc1 = 0;
      for (let i = 0; i < withTc.length; i++) {
        if (vec1[i] > t1) { mask1[i] = 1; count1++; sumTc1 += tcArr[i]; }
      }
      if (count1 < minSamples) continue;
      const avgTcHigh = sumTc1 / count1;
      if (avgTcHigh <= overallAvgTc * HYPOTHESIS_CONFIG.tcLiftThresholdSingle) continue;

      for (const f2 of keyFeatures) {
        if (f2 === f1) continue;
        const vec2 = featureVecs.get(f2)!;
        for (const t2 of thresholds[f2]) {
          let count = 0;
          let sumTc = 0;
          let sumTcSq = 0;
          for (let i = 0; i < withTc.length; i++) {
            if (mask1[i] && vec2[i] > t2) {
              const tc = tcArr[i];
              count++;
              sumTc += tc;
              sumTcSq += tc * tc;
            }
          }
          if (count < minSamples) continue;
          const avgTc = sumTc / count;
          if (avgTc <= overallAvgTc * HYPOTHESIS_CONFIG.tcLiftThresholdPair) continue;

          const tcLift = avgTc / overallAvgTc - 1;
          const prevalence = count / withTc.length;
          const samplePenalty = Math.sqrt(count);
          const confidence = Math.min(1, tcLift * prevalence * samplePenalty);
          if (confidence < HYPOTHESIS_CONFIG.minConfidence) continue;

          const groupVar = count > 1 ? (sumTcSq - count * avgTc * avgTc) / (count - 1) : 0;
          const groupStd = Math.sqrt(Math.max(0, groupVar));

          const compN = withTc.length - count;
          const compSumTc = overallSumTc - sumTc;
          const compMean = compN > 0 ? compSumTc / compN : 0;
          const compSumSq = withTc.reduce((s, r, i) => {
            if (!(mask1[i] && vec2[i] > t2)) s += (r.tc! - compMean) ** 2;
            return s;
          }, 0);
          const compStd = compN > 1 ? Math.sqrt(compSumSq / (compN - 1)) : 0;

          const pValue = welchOneSidedPValue(avgTc, groupStd, count, compMean, compStd, compN);
          if (pValue > bonferroniAlpha) continue;

          const tcStd = groupStd;
          const conservativeTc = avgTc - tcStd * HYPOTHESIS_CONFIG.conservativeTcPenalty;

          rules.push({
            conditions: [
              { feature: f1, operator: ">", threshold: Number(t1.toFixed(3)) },
              { feature: f2, operator: ">", threshold: Number(t2.toFixed(3)) },
            ],
            outcome: `Tc likely > ${Math.round(Math.max(0, conservativeTc))}K (n=${count}, p=${pValue.toExponential(1)})`,
            confidence: Number(confidence.toFixed(3)),
            sampleCount: count,
            avgTc: Number(avgTc.toFixed(1)),
            conservativeTc: Number(Math.max(0, conservativeTc).toFixed(1)),
          });
        }
      }
    }
  }

  rules.sort((a, b) => b.confidence * b.avgTc - a.confidence * a.avgTc);
  return rules.slice(0, HYPOTHESIS_CONFIG.maxRules);
}

function selectTopTcCorrelatedFeatures(withTc: FeatureRecord[], maxFeatures: number): (keyof PhysicsFeatureVector)[] {
  const tcValues = withTc.map(r => r.tc!);
  const featureCorrelations: { name: keyof PhysicsFeatureVector; absCorr: number }[] = [];

  for (const fname of FEATURE_NAMES) {
    const vals = withTc.map(r => r.featureVector[fname]);
    const corr = robustCorrelation(vals, tcValues);
    featureCorrelations.push({ name: fname, absCorr: Math.abs(corr) });
  }

  featureCorrelations.sort((a, b) => b.absCorr - a.absCorr);
  return featureCorrelations
    .filter(f => f.absCorr >= HYPOTHESIS_CONFIG.minFeatureCorrelation)
    .slice(0, maxFeatures)
    .map(f => f.name);
}

interface FeatureRange {
  feature: keyof PhysicsFeatureVector;
  lo: number;
  hi: number;
  label: string;
}

function computeFeatureRanges(withTc: FeatureRecord[], feature: keyof PhysicsFeatureVector): FeatureRange[] {
  const vals = withTc.map(r => r.featureVector[feature]).sort((a, b) => a - b);
  const n = vals.length;
  if (n < 8) return [];
  const q25 = vals[Math.floor(n * 0.25)];
  const q50 = vals[Math.floor(n * 0.50)];
  const q75 = vals[Math.floor(n * 0.75)];

  return [
    { feature, lo: q50, hi: Infinity, label: "high" },
    { feature, lo: q75, hi: Infinity, label: "very high" },
    { feature, lo: q25, hi: q75, label: "moderate" },
    { feature, lo: -Infinity, hi: q25, label: "low" },
  ];
}

function findCoOccurrencePatterns(records: FeatureRecord[]): CoOccurrencePattern[] {
  const patterns: CoOccurrencePattern[] = [];
  const withTc = records.filter(r => r.tc !== null && r.tc > 0);
  if (withTc.length < 10) return patterns;

  const minSamples = Math.max(HYPOTHESIS_CONFIG.minSampleFloor, Math.ceil(withTc.length * HYPOTHESIS_CONFIG.minSampleFraction));
  const overallAvgTc = withTc.reduce((s, r) => s + r.tc!, 0) / withTc.length;

  const topFeatures = selectTopTcCorrelatedFeatures(withTc, HYPOTHESIS_CONFIG.maxCoOccurrenceFeatures);
  if (topFeatures.length < 3) return patterns;

  const featureRanges = new Map<keyof PhysicsFeatureVector, FeatureRange[]>();
  for (const f of topFeatures) {
    featureRanges.set(f, computeFeatureRanges(withTc, f));
  }

  const featureVectors = new Map<string, Uint8Array>();
  for (const f of topFeatures) {
    const ranges = featureRanges.get(f)!;
    for (const range of ranges) {
      const key = `${f}:${range.label}`;
      const mask = new Uint8Array(withTc.length);
      for (let i = 0; i < withTc.length; i++) {
        const v = withTc[i].featureVector[f];
        if (v > range.lo && (range.hi === Infinity || v <= range.hi)) mask[i] = 1;
      }
      featureVectors.set(key, mask);
    }
  }

  const singleItemsets: { key: string; feature: keyof PhysicsFeatureVector; label: string; mask: Uint8Array; count: number; avgTc: number }[] = [];

  for (const f of topFeatures) {
    const ranges = featureRanges.get(f)!;
    for (const range of ranges) {
      const key = `${f}:${range.label}`;
      const mask = featureVectors.get(key)!;
      let count = 0;
      let sumTc = 0;
      for (let i = 0; i < withTc.length; i++) {
        if (mask[i]) { count++; sumTc += withTc[i].tc!; }
      }
      if (count < minSamples) continue;
      const avgTc = sumTc / count;
      if (avgTc > overallAvgTc * HYPOTHESIS_CONFIG.aprioriSingleLiftThreshold) {
        singleItemsets.push({ key, feature: f, label: range.label, mask, count, avgTc });
      }
    }
  }

  const pairItemsets: { keys: string[]; features: (keyof PhysicsFeatureVector)[]; mask: Uint8Array; count: number; avgTc: number }[] = [];

  for (let i = 0; i < singleItemsets.length; i++) {
    for (let j = i + 1; j < singleItemsets.length; j++) {
      if (singleItemsets[i].feature === singleItemsets[j].feature) continue;
      const mask = new Uint8Array(withTc.length);
      let count = 0;
      let sumTc = 0;
      for (let k = 0; k < withTc.length; k++) {
        if (singleItemsets[i].mask[k] && singleItemsets[j].mask[k]) {
          mask[k] = 1;
          count++;
          sumTc += withTc[k].tc!;
        }
      }
      if (count < minSamples) continue;
      const avgTc = sumTc / count;
      if (avgTc > overallAvgTc * HYPOTHESIS_CONFIG.tcLiftThresholdPair) {
        pairItemsets.push({
          keys: [singleItemsets[i].key, singleItemsets[j].key],
          features: [singleItemsets[i].feature, singleItemsets[j].feature],
          mask, count, avgTc,
        });
      }
    }
  }

  for (let i = 0; i < pairItemsets.length; i++) {
    for (let j = 0; j < singleItemsets.length; j++) {
      const pair = pairItemsets[i];
      const single = singleItemsets[j];
      if (pair.features.includes(single.feature)) continue;

      let count = 0;
      let sumTc = 0;
      let maxTc = 0;
      for (let k = 0; k < withTc.length; k++) {
        if (pair.mask[k] && single.mask[k]) {
          const tc = withTc[k].tc!;
          count++;
          sumTc += tc;
          if (tc > maxTc) maxTc = tc;
        }
      }
      if (count < minSamples) continue;
      const avgTc = sumTc / count;

      if (avgTc > overallAvgTc * HYPOTHESIS_CONFIG.tcLiftThresholdCoOccurrence) {
        const allKeys = [...pair.keys, single.key];
        const featureNames = allKeys.map(k => k.split(":")[0]);
        const rangeLabels = allKeys.map(k => k.split(":")[1] || "high");
        patterns.push({
          features: featureNames,
          featureRangeLabels: rangeLabels,
          frequency: count / withTc.length,
          avgTc: Number(avgTc.toFixed(1)),
          maxTc: Number(maxTc.toFixed(1)),
        });
      }
    }
  }

  const seen = new Set<string>();
  const deduped = patterns.filter(p => {
    const key = p.features.map((f, i) => `${f}:${p.featureRangeLabels[i]}`).sort().join("+");
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });

  deduped.sort((a, b) => b.avgTc * b.frequency - a.avgTc * a.frequency);
  return deduped.slice(0, HYPOTHESIS_CONFIG.maxCoOccurrencePatterns);
}

function featureToReadable(f: string): string {
  const map: Record<string, string> = {
    DOS_EF: "high density of states at E_F (N(E_F))",
    electron_phonon_lambda: "strong electron-phonon coupling (λ)",
    nesting_score: "strong Fermi surface nesting (χ₀)",
    band_flatness: "flat bands near Fermi level (∂²ε/∂k²)",
    pairing_strength: "strong pairing interaction (V_pair)",
    orbital_degeneracy: "orbital degeneracy (N_orb)",
    charge_transfer: "significant charge transfer (Δq)",
    quantum_critical_score: "quantum criticality proximity (δ_QCP)",
    hydrogen_density: "high hydrogen content (n_H)",
    phonon_log_frequency: "high logarithmic phonon frequency (ω_log)",
    debye_temp: "high Debye temperature (Θ_D)",
    correlation_strength: "strong electronic correlations (U/t)",
    van_hove_distance: "van Hove singularity proximity (ε_vH)",
    lattice_anisotropy: "lattice anisotropy (c/a ratio)",
    mott_proximity: "Mott insulator proximity (U/W)",
    spin_fluctuation: "spin fluctuations (χ_spin)",
    cdw_proximity: "CDW instability proximity (χ_CDW)",
    bandwidth: "narrow bandwidth (W)",
    anharmonicity: "phonon anharmonicity (γ_ph)",
    fermi_surface_dimensionality: "quasi-2D Fermi surface (d_FS)",
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
      confidenceScore: rule.confidence,
      testCount: 0,
      supportCount: 0,
      refuteCount: 0,
      requiredConditions: rule.conditions.map(c => `${c.feature}${c.operator}${c.threshold}`),
      predictedTcRange: [Math.round(rule.conservativeTc * 0.7), Math.round(rule.avgTc * 1.3)],
      discoveredAt: Date.now(),
      status: "proposed",
      lastTestedAt: 0,
    });
  }

  for (const pattern of coOccurrences.slice(0, 5)) {
    const featureDescs = pattern.features.map((f, i) => {
      const rangeLabel = pattern.featureRangeLabels[i] || "high";
      return `${rangeLabel} ${featureToReadable(f)}`;
    });
    const statement = `Co-occurrence of ${featureDescs.join(", ")} correlates with enhanced Tc (avg ${pattern.avgTc}K)`;
    const mathForm = pattern.features.map((f, i) => `${f}∈${pattern.featureRangeLabels[i] || "high"}`).join(" ∧ ");

    hypotheses.push({
      id: generateId(),
      statement,
      mathematicalForm: mathForm,
      supportingEvidence: [],
      confidenceScore: Math.min(0.8, pattern.frequency * pattern.avgTc / 100),
      testCount: 0,
      supportCount: 0,
      refuteCount: 0,
      requiredConditions: pattern.features.map((f, i) => `${f}∈${pattern.featureRangeLabels[i] || "high"}`),
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
