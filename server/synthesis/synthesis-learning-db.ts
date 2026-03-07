import type { SynthesisVector } from "../physics/synthesis-simulator";

interface SynthesisRecord {
  formula: string;
  materialClass: string;
  synthesisVector: SynthesisVector;
  resultTc: number;
  stability: number;
  feasibilityScore: number;
  timestamp: number;
}

interface SynthesisPattern {
  description: string;
  confidence: number;
  sampleCount: number;
  avgTcImprovement: number;
}

interface LearningStats {
  totalRecords: number;
  uniqueFormulas: number;
  avgTc: number;
  bestTc: number;
  bestFormula: string;
  patterns: SynthesisPattern[];
  classBreakdown: Record<string, { count: number; avgTc: number; bestTc: number }>;
  parameterCorrelations: { parameter: string; correlation: number }[];
}

const records: SynthesisRecord[] = [];
const formulaMap = new Map<string, SynthesisRecord[]>();
const MAX_RECORDS = 2000;

export function recordSynthesisResult(
  formula: string,
  materialClass: string,
  synthesisVector: SynthesisVector,
  resultTc: number,
  stability: number = 0.5,
  feasibilityScore: number = 1.0
): void {
  const record: SynthesisRecord = {
    formula,
    materialClass,
    synthesisVector: { ...synthesisVector },
    resultTc,
    stability,
    feasibilityScore,
    timestamp: Date.now(),
  };

  records.push(record);
  if (records.length > MAX_RECORDS) records.shift();

  const existing = formulaMap.get(formula) || [];
  existing.push(record);
  if (existing.length > 20) existing.shift();
  formulaMap.set(formula, existing);
}

export function querySimilarSynthesis(formula: string, topK: number = 5): SynthesisRecord[] {
  const exact = formulaMap.get(formula);
  if (exact && exact.length > 0) {
    return exact.sort((a, b) => b.resultTc - a.resultTc).slice(0, topK);
  }

  const elements = extractElements(formula);
  const scored: { record: SynthesisRecord; similarity: number }[] = [];

  for (const r of records) {
    const rElements = extractElements(r.formula);
    const intersection = elements.filter(e => rElements.includes(e));
    const union = new Set([...elements, ...rElements]);
    const jaccard = intersection.length / union.size;
    if (jaccard > 0.3) {
      scored.push({ record: r, similarity: jaccard });
    }
  }

  return scored
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK)
    .map(s => s.record);
}

export function getBestSynthesisFor(formula: string): SynthesisVector | null {
  const history = formulaMap.get(formula);
  if (!history || history.length === 0) return null;
  const best = history.reduce((a, b) => a.resultTc > b.resultTc ? a : b);
  return { ...best.synthesisVector };
}

export function getSynthesisPatterns(): SynthesisPattern[] {
  if (records.length < 10) return [];
  const patterns: SynthesisPattern[] = [];

  const highP = records.filter(r => r.synthesisVector.pressure > 50);
  const lowP = records.filter(r => r.synthesisVector.pressure <= 50);
  if (highP.length >= 5 && lowP.length >= 5) {
    const avgTcHighP = highP.reduce((s, r) => s + r.resultTc, 0) / highP.length;
    const avgTcLowP = lowP.reduce((s, r) => s + r.resultTc, 0) / lowP.length;
    if (avgTcHighP > avgTcLowP * 1.2) {
      patterns.push({
        description: "High pressure (>50 GPa) correlates with higher Tc",
        confidence: Math.min(0.9, highP.length / 20),
        sampleCount: highP.length,
        avgTcImprovement: avgTcHighP - avgTcLowP,
      });
    }
  }

  const fastQuench = records.filter(r => r.synthesisVector.coolingRate > 500);
  const slowCool = records.filter(r => r.synthesisVector.coolingRate <= 500);
  if (fastQuench.length >= 5 && slowCool.length >= 5) {
    const avgFQ = fastQuench.reduce((s, r) => s + r.resultTc, 0) / fastQuench.length;
    const avgSC = slowCool.reduce((s, r) => s + r.resultTc, 0) / slowCool.length;
    if (avgFQ > avgSC * 1.1) {
      patterns.push({
        description: "Rapid quenching (>500 K/s) preserves metastable high-Tc phases",
        confidence: Math.min(0.85, fastQuench.length / 15),
        sampleCount: fastQuench.length,
        avgTcImprovement: avgFQ - avgSC,
      });
    }
  }

  const strained = records.filter(r => Math.abs(r.synthesisVector.strain) > 2);
  const unstrained = records.filter(r => Math.abs(r.synthesisVector.strain) <= 2);
  if (strained.length >= 5 && unstrained.length >= 5) {
    const avgS = strained.reduce((s, r) => s + r.resultTc, 0) / strained.length;
    const avgU = unstrained.reduce((s, r) => s + r.resultTc, 0) / unstrained.length;
    if (avgS > avgU * 1.05) {
      patterns.push({
        description: "Epitaxial strain (>2%) enhances Tc through band structure modification",
        confidence: Math.min(0.8, strained.length / 15),
        sampleCount: strained.length,
        avgTcImprovement: avgS - avgU,
      });
    }
  }

  const longAnneal = records.filter(r => r.synthesisVector.annealTime > 12);
  const shortAnneal = records.filter(r => r.synthesisVector.annealTime <= 12);
  if (longAnneal.length >= 5 && shortAnneal.length >= 5) {
    const avgL = longAnneal.reduce((s, r) => s + r.resultTc, 0) / longAnneal.length;
    const avgSh = shortAnneal.reduce((s, r) => s + r.resultTc, 0) / shortAnneal.length;
    if (avgL > avgSh * 1.05) {
      patterns.push({
        description: "Extended annealing (>12h) improves phase purity and Tc",
        confidence: Math.min(0.75, longAnneal.length / 15),
        sampleCount: longAnneal.length,
        avgTcImprovement: avgL - avgSh,
      });
    }
  }

  const classes = new Map<string, SynthesisRecord[]>();
  for (const r of records) {
    const existing = classes.get(r.materialClass) || [];
    existing.push(r);
    classes.set(r.materialClass, existing);
  }
  for (const [cls, recs] of classes) {
    if (recs.length < 5) continue;
    const highTC = recs.filter(r => r.resultTc > 30);
    if (highTC.length < 3) continue;
    const avgP = highTC.reduce((s, r) => s + r.synthesisVector.pressure, 0) / highTC.length;
    const avgT = highTC.reduce((s, r) => s + r.synthesisVector.temperature, 0) / highTC.length;
    patterns.push({
      description: `${cls}: best results at ~${Math.round(avgT)}K, ~${Math.round(avgP)} GPa`,
      confidence: Math.min(0.7, highTC.length / 10),
      sampleCount: highTC.length,
      avgTcImprovement: highTC.reduce((s, r) => s + r.resultTc, 0) / highTC.length,
    });
  }

  return patterns.sort((a, b) => b.confidence - a.confidence);
}

export function getSynthesisLearningStats(): LearningStats {
  const count = records.length;
  if (count === 0) {
    return {
      totalRecords: 0,
      uniqueFormulas: 0,
      avgTc: 0,
      bestTc: 0,
      bestFormula: "",
      patterns: [],
      classBreakdown: {},
      parameterCorrelations: [],
    };
  }

  let bestTc = 0;
  let bestFormula = "";
  let totalTc = 0;

  const classStats = new Map<string, { count: number; totalTc: number; bestTc: number }>();

  for (const r of records) {
    totalTc += r.resultTc;
    if (r.resultTc > bestTc) {
      bestTc = r.resultTc;
      bestFormula = r.formula;
    }
    const cs = classStats.get(r.materialClass) || { count: 0, totalTc: 0, bestTc: 0 };
    cs.count++;
    cs.totalTc += r.resultTc;
    if (r.resultTc > cs.bestTc) cs.bestTc = r.resultTc;
    classStats.set(r.materialClass, cs);
  }

  const classBreakdown: Record<string, { count: number; avgTc: number; bestTc: number }> = {};
  for (const [cls, cs] of classStats) {
    classBreakdown[cls] = { count: cs.count, avgTc: cs.totalTc / cs.count, bestTc: cs.bestTc };
  }

  const paramCorrelations = computeParameterCorrelations();

  return {
    totalRecords: count,
    uniqueFormulas: formulaMap.size,
    avgTc: totalTc / count,
    bestTc,
    bestFormula,
    patterns: getSynthesisPatterns(),
    classBreakdown,
    parameterCorrelations: paramCorrelations,
  };
}

function computeParameterCorrelations(): { parameter: string; correlation: number }[] {
  if (records.length < 10) return [];

  const params: { name: keyof SynthesisVector; label: string }[] = [
    { name: "temperature", label: "Temperature" },
    { name: "pressure", label: "Pressure" },
    { name: "coolingRate", label: "Cooling Rate" },
    { name: "annealTime", label: "Anneal Time" },
    { name: "strain", label: "Strain" },
    { name: "magneticField", label: "Magnetic Field" },
    { name: "thermalCycles", label: "Thermal Cycles" },
  ];

  const tcs = records.map(r => r.resultTc);
  const meanTc = tcs.reduce((s, v) => s + v, 0) / tcs.length;

  const results: { parameter: string; correlation: number }[] = [];

  for (const p of params) {
    const vals = records.map(r => r.synthesisVector[p.name]);
    const meanV = vals.reduce((s, v) => s + v, 0) / vals.length;

    let covXY = 0, varX = 0, varY = 0;
    for (let i = 0; i < records.length; i++) {
      const dx = vals[i] - meanV;
      const dy = tcs[i] - meanTc;
      covXY += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }

    const denom = Math.sqrt(varX * varY);
    const corr = denom > 1e-10 ? covXY / denom : 0;
    results.push({ parameter: p.label, correlation: Math.round(corr * 1000) / 1000 });
  }

  return results.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
}

function extractElements(formula: string): string[] {
  const els: string[] = [];
  const re = /([A-Z][a-z]?)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(formula)) !== null) {
    if (m[1] && !els.includes(m[1])) els.push(m[1]);
  }
  return els;
}
