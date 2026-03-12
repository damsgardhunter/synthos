import {
  parseFormulaElements,
  parseFormulaCounts,
  estimateBandwidthW,
} from "../learning/physics-engine";
import {
  getHubbardU,
  isTransitionMetal,
  isRareEarth,
} from "../learning/elemental-data";

export type CorrelationRegime =
  | "weakly-correlated"
  | "moderately-correlated"
  | "strongly-correlated"
  | "Mott-proximate";

export interface CorrelationRegimeResult {
  isCorrelated: boolean;
  regime: CorrelationRegime;
}

export interface PairingWeights {
  spinFluctuationWeight: number;
  orbitalFluctuationWeight: number;
  phononWeight: number;
  effectiveLambdaModifier: number;
}

export interface CorrelationAnalysis {
  regime: CorrelationRegimeResult;
  correlationScore: number;
  pairingWeights: PairingWeights;
  tcModifier: number;
  materialPatterns: string[];
}

export interface CorrelationEngineStats {
  materialsAnalyzed: number;
  regimeBreakdown: Record<CorrelationRegime, number>;
  avgCorrelationScore: number;
  totalScoreSum: number;
}

const engineStats: CorrelationEngineStats = {
  materialsAnalyzed: 0,
  regimeBreakdown: {
    "weakly-correlated": 0,
    "moderately-correlated": 0,
    "strongly-correlated": 0,
    "Mott-proximate": 0,
  },
  avgCorrelationScore: 0,
  totalScoreSum: 0,
};

const CUPRATE_ELEMENTS = ["Cu", "Ba", "La", "Y", "Sr", "Ca", "Bi", "Tl", "Hg"];
const FE_PNICTIDE_ELEMENTS = ["Fe", "As", "Se", "Te", "P"];
const HEAVY_FERMION_ELEMENTS = ["Ce", "U", "Yb", "Pr", "Sm"];

function detectMaterialClass(elements: string[], formula?: string): string[] {
  const syms = elements;
  const patterns: string[] = [];

  if (formula) {
    const counts = parseFormulaCounts(formula);
    const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    const hCount = counts["H"] || 0;
    const hFraction = hCount / totalAtoms;
    const metalAtoms = syms.filter(e => isTransitionMetal(e) || isRareEarth(e))
      .reduce((s, e) => s + (counts[e] || 0), 0);
    const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

    if (hFraction > 0.7 || hRatio >= 6) {
      patterns.push("superhydride-phonon-dominated");
    } else if (hFraction > 0.4 && hRatio >= 3) {
      patterns.push("hydride-phonon-enhanced");
    }
  }

  const hasCu = syms.includes("Cu");
  const hasO = syms.includes("O");
  const hasCuprateHost = syms.some((s) => CUPRATE_ELEMENTS.includes(s));
  if (hasCu && hasO && hasCuprateHost) {
    patterns.push("cuprate-Mott-proximity");
  }

  const hasFe = syms.includes("Fe");
  const hasPnictide = syms.some((s) => ["As", "P", "Se", "Te"].includes(s));
  if (hasFe && hasPnictide) {
    patterns.push("Fe-pnictide-spin-fluctuation");
  }

  const hasHeavyFermion = syms.some((s) => HEAVY_FERMION_ELEMENTS.includes(s));
  if (hasHeavyFermion) {
    patterns.push("heavy-fermion-Kondo-lattice");
  }

  const hasTM = syms.some((s) => isTransitionMetal(s));
  if (hasTM && !hasCu && !hasFe) {
    patterns.push("correlated-d-electron");
  }

  const hasRE = syms.some((s) => isRareEarth(s));
  if (hasRE && !hasHeavyFermion) {
    patterns.push("f-electron-system");
  }

  if (patterns.length === 0) {
    patterns.push("conventional");
  }

  return patterns;
}

export function detectCorrelatedRegime(
  UoverW: number,
  dosAtEF: number,
  flatBandScore: number,
  nestingScore: number,
): CorrelationRegimeResult {
  const score = computeCorrelationScore(UoverW, dosAtEF, flatBandScore, nestingScore);

  let regime: CorrelationRegime;
  if (UoverW > 1 && score >= 0.8) {
    regime = "Mott-proximate";
  } else if (UoverW > 1 && score >= 0.55) {
    regime = "strongly-correlated";
  } else if (score >= 0.3 || UoverW > 0.7) {
    regime = "moderately-correlated";
  } else {
    regime = "weakly-correlated";
  }

  return {
    isCorrelated: UoverW > 1 || score >= 0.3,
    regime,
  };
}

export function computeCorrelationScore(
  UoverW: number,
  dosAtEF: number,
  flatBandScore: number,
  nestingScore: number,
): number {
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const uNorm = clamp(UoverW / 2.0);
  const dosNorm = clamp(dosAtEF / 5.0);
  const fbNorm = clamp(flatBandScore);
  const nNorm = clamp(nestingScore);

  return 0.3 * uNorm + 0.3 * dosNorm + 0.2 * fbNorm + 0.2 * nNorm;
}

function normalizeWeights(
  spin: number, orbital: number, phonon: number,
): [number, number, number] {
  const total = spin + orbital + phonon;
  if (total <= 0) return [0.1, 0.1, 0.8];
  return [spin / total, orbital / total, phonon / total];
}

export function adjustPairingWeights(
  correlationScore: number,
  regime: CorrelationRegime,
  materialClass: string,
): PairingWeights {
  let spinFluctuationWeight: number;
  let orbitalFluctuationWeight: number;
  let phononWeight: number;
  let effectiveLambdaModifier: number;

  if (materialClass.includes("superhydride-phonon-dominated")) {
    spinFluctuationWeight = 0.03;
    orbitalFluctuationWeight = 0.02;
    phononWeight = 0.95;
    effectiveLambdaModifier = 1.0 + correlationScore * 0.1;

    const [ns, no, np] = normalizeWeights(spinFluctuationWeight, orbitalFluctuationWeight, phononWeight);
    return {
      spinFluctuationWeight: ns,
      orbitalFluctuationWeight: no,
      phononWeight: np,
      effectiveLambdaModifier,
    };
  }

  if (materialClass.includes("hydride-phonon-enhanced")) {
    spinFluctuationWeight = 0.08;
    orbitalFluctuationWeight = 0.07;
    phononWeight = 0.85;
    effectiveLambdaModifier = 1.0 + correlationScore * 0.15;

    const [ns, no, np] = normalizeWeights(spinFluctuationWeight, orbitalFluctuationWeight, phononWeight);
    return {
      spinFluctuationWeight: ns,
      orbitalFluctuationWeight: no,
      phononWeight: np,
      effectiveLambdaModifier,
    };
  }

  switch (regime) {
    case "Mott-proximate":
      spinFluctuationWeight = 0.6;
      orbitalFluctuationWeight = 0.25;
      phononWeight = 0.15;
      effectiveLambdaModifier = 1.8 + correlationScore * 0.5;
      break;
    case "strongly-correlated":
      spinFluctuationWeight = 0.45;
      orbitalFluctuationWeight = 0.25;
      phononWeight = 0.3;
      effectiveLambdaModifier = 1.4 + correlationScore * 0.4;
      break;
    case "moderately-correlated":
      spinFluctuationWeight = 0.3;
      orbitalFluctuationWeight = 0.2;
      phononWeight = 0.5;
      effectiveLambdaModifier = 1.1 + correlationScore * 0.3;
      break;
    case "weakly-correlated":
    default:
      spinFluctuationWeight = 0.1;
      orbitalFluctuationWeight = 0.1;
      phononWeight = 0.8;
      effectiveLambdaModifier = 1.0;
      break;
  }

  if (materialClass.includes("cuprate") || materialClass.includes("Mott")) {
    const boostFactor = 1.25;
    spinFluctuationWeight *= boostFactor;
    phononWeight *= (1.0 / boostFactor);
  }

  if (materialClass.includes("Fe-pnictide") || materialClass.includes("spin-fluctuation")) {
    const spinBoost = 1.4;
    const orbBoost = 1.2;
    spinFluctuationWeight *= spinBoost;
    orbitalFluctuationWeight *= orbBoost;
    phononWeight *= (1.0 / (spinBoost * 0.5 + orbBoost * 0.5));
  }

  if (materialClass.includes("heavy-fermion") || materialClass.includes("Kondo")) {
    orbitalFluctuationWeight *= 1.5;
    effectiveLambdaModifier *= 0.8;
  }

  const [ns, no, np] = normalizeWeights(spinFluctuationWeight, orbitalFluctuationWeight, phononWeight);

  return {
    spinFluctuationWeight: ns,
    orbitalFluctuationWeight: no,
    phononWeight: np,
    effectiveLambdaModifier,
  };
}

function estimateElementWeightedBandwidth(elements: string[], formula: string): number {
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  let wAvg = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    wAvg += estimateBandwidthW(el) * frac;
  }
  return Math.max(0.5, wAvg);
}

async function estimateBandwidthFromSurrogate(formula: string): Promise<number | null> {
  try {
    const { predictBandStructure } = await import("./band-structure-surrogate");
    const surr = predictBandStructure(formula);
    if (surr && surr.bandwidthMin > 0) {
      return surr.bandwidthMin;
    }
  } catch {}
  return null;
}

export async function estimateCorrelationEffects(
  formula: string,
  mlFeatures: {
    UoverW?: number;
    dosAtEF?: number;
    flatBandScore?: number;
    nestingScore?: number;
    effectiveMass?: number;
    pressureGpa?: number;
    bandwidthOverride?: number;
  },
): Promise<CorrelationAnalysis> {
  const elements = parseFormulaElements(formula);

  const avgU = elements.reduce((sum, el) => {
    const u = getHubbardU(el);
    return sum + (u ?? 0);
  }, 0) / Math.max(1, elements.length);

  let UoverW: number;
  if (mlFeatures.UoverW != null) {
    UoverW = mlFeatures.UoverW;
  } else {
    let bandwidth = mlFeatures.bandwidthOverride ?? null;

    if (bandwidth == null) {
      const surrogateW = await estimateBandwidthFromSurrogate(formula);
      if (surrogateW != null && surrogateW > 0.01) {
        bandwidth = surrogateW;
      }
    }

    if (bandwidth == null) {
      bandwidth = estimateElementWeightedBandwidth(elements, formula);
    }

    const pressure = mlFeatures.pressureGpa ?? 0;
    if (pressure > 50) {
      const pressureWidening = 1.0 + (pressure - 50) / 100;
      bandwidth *= pressureWidening;
    }

    UoverW = Math.min(2.0, avgU / Math.max(0.5, bandwidth));
  }

  const dosAtEF = mlFeatures.dosAtEF ?? 1.5;
  const flatBandScore = mlFeatures.flatBandScore ?? 0.3;
  const nestingScore = mlFeatures.nestingScore ?? 0.2;

  const correlationScore = computeCorrelationScore(UoverW, dosAtEF, flatBandScore, nestingScore);
  const regimeResult = detectCorrelatedRegime(UoverW, dosAtEF, flatBandScore, nestingScore);
  const materialPatterns = detectMaterialClass(elements, formula);

  const primaryClass = materialPatterns[0] || "conventional";
  const pairingWeights = adjustPairingWeights(correlationScore, regimeResult.regime, primaryClass);

  let tcModifier = pairingWeights.effectiveLambdaModifier;

  const effectiveMass = mlFeatures.effectiveMass ?? null;
  if (materialPatterns.includes("heavy-fermion-Kondo-lattice") && effectiveMass != null) {
    if (effectiveMass > 200) {
      tcModifier *= 0.3;
    } else if (effectiveMass > 100) {
      tcModifier *= 0.5;
    } else if (effectiveMass > 50) {
      tcModifier *= 0.65;
    }
  } else if (effectiveMass != null && effectiveMass > 50) {
    const massPenalty = Math.max(0.5, 1.0 - (effectiveMass - 50) / 200);
    tcModifier *= massPenalty;
  }

  if (materialPatterns.includes("cuprate-Mott-proximity")) {
    tcModifier *= 1.2;
  }
  if (materialPatterns.includes("Fe-pnictide-spin-fluctuation")) {
    tcModifier *= 1.1;
  }

  engineStats.materialsAnalyzed++;
  engineStats.regimeBreakdown[regimeResult.regime]++;
  engineStats.totalScoreSum += correlationScore;
  engineStats.avgCorrelationScore = engineStats.totalScoreSum / engineStats.materialsAnalyzed;

  return {
    regime: regimeResult,
    correlationScore,
    pairingWeights,
    tcModifier,
    materialPatterns,
  };
}

export function getCorrelationEngineStats(): CorrelationEngineStats {
  return { ...engineStats };
}

export function resetCorrelationEngineStats(): void {
  engineStats.materialsAnalyzed = 0;
  engineStats.regimeBreakdown = {
    "weakly-correlated": 0,
    "moderately-correlated": 0,
    "strongly-correlated": 0,
    "Mott-proximate": 0,
  };
  engineStats.avgCorrelationScore = 0;
  engineStats.totalScoreSum = 0;
}
