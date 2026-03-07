import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";

export interface ConstraintSolution {
  targetTc: number;
  muStar: number;
  requiredLambda: { min: number; max: number; optimal: number };
  requiredOmegaLog: { min: number; max: number; optimal: number };
  feasibilityScore: number;
  feasibilityNote: string;
  structuralTargets: StructuralTarget[];
  elementSuggestions: ElementSuggestion[];
  constraintChain: ConstraintStep[];
}

interface StructuralTarget {
  type: string;
  reason: string;
  priority: number;
  exemplars: string[];
}

interface ElementSuggestion {
  elements: string[];
  role: string;
  reason: string;
  confidence: number;
}

interface ConstraintStep {
  step: number;
  parameter: string;
  constraint: string;
  value: string;
  status: "satisfied" | "feasible" | "challenging" | "unlikely";
}

interface SweepPoint {
  lambda: number;
  omegaLogK: number;
  tc: number;
}

function mcMillanTc(lambda: number, omegaLogK: number, muStar: number): number {
  const denom = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denom) < 1e-6 || denom <= 0) return 0;
  let f1: number;
  if (lambda < 1.5) {
    f1 = Math.pow(1 + lambda / (2.46 * (1 + 3.8 * muStar)), 1 / 3);
  } else {
    f1 = Math.sqrt(1 + lambda / 2.46);
  }
  const exponent = -1.04 * (1 + lambda) / denom;
  const tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  return Number.isFinite(tc) && tc > 0 ? tc : 0;
}

function solveRequiredLambda(
  targetTc: number,
  omegaLogK: number,
  muStar: number,
): number {
  let lo = 0.1, hi = 5.0;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    const tc = mcMillanTc(mid, omegaLogK, muStar);
    if (tc < targetTc) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function solveRequiredOmegaLog(
  targetTc: number,
  lambda: number,
  muStar: number,
): number {
  let lo = 50, hi = 5000;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    const tc = mcMillanTc(lambda, mid, muStar);
    if (tc < targetTc) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

function sweepLambdaOmega(
  targetTc: number,
  muStar: number,
): SweepPoint[] {
  const solutions: SweepPoint[] = [];
  for (let lam = 0.5; lam <= 4.0; lam += 0.1) {
    const omega = solveRequiredOmegaLog(targetTc, lam, muStar);
    if (omega > 0 && omega < 5000) {
      const actualTc = mcMillanTc(lam, omega, muStar);
      if (Math.abs(actualTc - targetTc) < targetTc * 0.05) {
        solutions.push({ lambda: Math.round(lam * 100) / 100, omegaLogK: Math.round(omega), tc: Math.round(actualTc * 10) / 10 });
      }
    }
  }
  return solutions;
}

function assessFeasibility(lambda: number, omegaLogK: number): { score: number; note: string } {
  let score = 1.0;
  const notes: string[] = [];

  if (lambda > 3.5) {
    score *= 0.2;
    notes.push("lambda > 3.5 is extremely rare");
  } else if (lambda > 2.5) {
    score *= 0.5;
    notes.push("lambda > 2.5 requires strong e-ph coupling (hydride-like)");
  } else if (lambda > 1.5) {
    score *= 0.8;
    notes.push("lambda 1.5-2.5 is achievable in hydrides and some compounds");
  }

  if (omegaLogK > 2000) {
    score *= 0.3;
    notes.push("omegaLog > 2000K requires very light elements (H-dominant)");
  } else if (omegaLogK > 1200) {
    score *= 0.6;
    notes.push("omegaLog 1200-2000K achievable with H, B, C");
  } else if (omegaLogK > 600) {
    score *= 0.9;
    notes.push("omegaLog 600-1200K achievable with light-element compounds");
  }

  if (lambda > 2.0 && omegaLogK > 1500) {
    score *= 1.2;
    notes.push("high lambda + high omegaLog is the hydride sweet spot");
  }

  return { score: Math.min(1, Math.round(score * 100) / 100), note: notes.join("; ") };
}

function suggestStructures(lambda: number, omegaLogK: number): StructuralTarget[] {
  const targets: StructuralTarget[] = [];

  if (omegaLogK > 1200 && lambda > 1.5) {
    targets.push({
      type: "clathrate-cage",
      reason: "High phonon freq + strong coupling requires hydrogen cage structures",
      priority: 1,
      exemplars: ["LaH10", "YH6", "CaH6", "ThH10"],
    });
    targets.push({
      type: "sodalite-cage",
      reason: "Sodalite cages maximize H-H metallic bonding and phonon hardening",
      priority: 2,
      exemplars: ["LaH10", "CeH9", "YH9"],
    });
  }

  if (lambda > 1.0 && omegaLogK > 600 && omegaLogK < 1500) {
    targets.push({
      type: "layered-hexagonal",
      reason: "Layered boride/carbide structures achieve moderate lambda with high phonon freq",
      priority: omegaLogK > 900 ? 1 : 2,
      exemplars: ["MgB2", "NbB2", "TaB2"],
    });
  }

  if (lambda > 0.8 && omegaLogK < 800) {
    targets.push({
      type: "A15",
      reason: "A15 structures offer high DOS and moderate coupling for transition metals",
      priority: 2,
      exemplars: ["Nb3Sn", "Nb3Ge", "V3Si"],
    });
  }

  if (lambda > 1.2 && omegaLogK < 600) {
    targets.push({
      type: "cuprate-layered",
      reason: "Low phonon freq + high coupling suggests unconventional layered mechanism",
      priority: 3,
      exemplars: ["YBCO", "Bi2212", "HgBaCuO"],
    });
  }

  if (omegaLogK > 800 && lambda > 0.8) {
    targets.push({
      type: "honeycomb",
      reason: "Honeycomb lattices with light atoms combine high phonon freq with good nesting",
      priority: 3,
      exemplars: ["MgB2", "LiBC"],
    });
  }

  return targets.sort((a, b) => a.priority - b.priority);
}

function suggestElements(lambda: number, omegaLogK: number): ElementSuggestion[] {
  const suggestions: ElementSuggestion[] = [];

  if (omegaLogK > 1200) {
    suggestions.push({
      elements: ["H"],
      role: "phonon hardener",
      reason: "Hydrogen provides the highest phonon frequencies due to its low mass",
      confidence: 0.9,
    });
  }

  if (omegaLogK > 600) {
    suggestions.push({
      elements: ["B", "C"],
      role: "phonon hardener",
      reason: "Light covalent elements provide high phonon freq and strong bonding",
      confidence: 0.8,
    });
  }

  if (lambda > 1.5) {
    suggestions.push({
      elements: ["La", "Y", "Ca", "Sr", "Ba"],
      role: "electron donor / cage stabilizer",
      reason: "Alkaline earth and rare earth metals stabilize cage structures and donate electrons",
      confidence: 0.85,
    });
    suggestions.push({
      elements: ["Nb", "V", "Ta", "Mo", "W"],
      role: "strong coupling provider",
      reason: "4d/5d transition metals have high Hopfield parameters for strong e-ph coupling",
      confidence: 0.75,
    });
  }

  if (lambda > 0.8 && lambda < 1.5) {
    suggestions.push({
      elements: ["Nb", "V", "Ti", "Zr", "Hf"],
      role: "transition metal backbone",
      reason: "Moderate coupling from d-electron compounds with good metallicity",
      confidence: 0.8,
    });
  }

  if (omegaLogK < 400 && lambda > 1.2) {
    suggestions.push({
      elements: ["Cu", "Fe", "Ni"],
      role: "correlated electron host",
      reason: "Low phonon scale + high coupling may indicate unconventional pairing",
      confidence: 0.5,
    });
  }

  return suggestions.sort((a, b) => b.confidence - a.confidence);
}

export function solveConstraints(
  targetTc: number,
  muStar: number = 0.10,
  pressureGpa: number = 0,
): ConstraintSolution {
  const muStarClamped = Math.max(0.08, Math.min(0.20, muStar));

  const optimalOmegaLog = targetTc > 200 ? 1500 : targetTc > 100 ? 800 : targetTc > 50 ? 500 : 300;
  const optimalLambda = solveRequiredLambda(targetTc, optimalOmegaLog, muStarClamped);

  const sweepSolutions = sweepLambdaOmega(targetTc, muStarClamped);

  let lambdaMin = Infinity, lambdaMax = 0;
  let omegaMin = Infinity, omegaMax = 0;
  for (const sol of sweepSolutions) {
    if (sol.lambda < lambdaMin) lambdaMin = sol.lambda;
    if (sol.lambda > lambdaMax) lambdaMax = sol.lambda;
    if (sol.omegaLogK < omegaMin) omegaMin = sol.omegaLogK;
    if (sol.omegaLogK > omegaMax) omegaMax = sol.omegaLogK;
  }

  if (lambdaMin === Infinity) {
    lambdaMin = optimalLambda * 0.7;
    lambdaMax = optimalLambda * 1.5;
  }
  if (omegaMin === Infinity) {
    omegaMin = optimalOmegaLog * 0.5;
    omegaMax = optimalOmegaLog * 2.0;
  }

  const feasibility = assessFeasibility(optimalLambda, optimalOmegaLog);
  const structuralTargets = suggestStructures(optimalLambda, optimalOmegaLog);
  const elementSuggestions = suggestElements(optimalLambda, optimalOmegaLog);

  if (pressureGpa > 50) {
    feasibility.score = Math.min(feasibility.score * 1.3, 1.0);
  }

  const chain: ConstraintStep[] = [
    {
      step: 1,
      parameter: "Tc",
      constraint: `Target Tc = ${targetTc}K`,
      value: `${targetTc}K`,
      status: "satisfied",
    },
    {
      step: 2,
      parameter: "mu*",
      constraint: `mu* in [0.08, 0.20]`,
      value: muStarClamped.toFixed(3),
      status: "satisfied",
    },
    {
      step: 3,
      parameter: "lambda",
      constraint: `lambda in [${lambdaMin.toFixed(2)}, ${lambdaMax.toFixed(2)}]`,
      value: `optimal ${optimalLambda.toFixed(3)}`,
      status: optimalLambda < 3.5 ? (optimalLambda < 2.5 ? "feasible" : "challenging") : "unlikely",
    },
    {
      step: 4,
      parameter: "omegaLog",
      constraint: `omegaLog in [${Math.round(omegaMin)}, ${Math.round(omegaMax)}]K`,
      value: `optimal ${Math.round(optimalOmegaLog)}K`,
      status: optimalOmegaLog < 2000 ? (optimalOmegaLog < 1200 ? "feasible" : "challenging") : "unlikely",
    },
    {
      step: 5,
      parameter: "structure",
      constraint: structuralTargets.length > 0 ? `Prefer ${structuralTargets[0].type}` : "No strong structural preference",
      value: structuralTargets.map(s => s.type).join(", ") || "open",
      status: structuralTargets.length > 0 ? "feasible" : "challenging",
    },
    {
      step: 6,
      parameter: "elements",
      constraint: elementSuggestions.length > 0 ? `Core elements: ${elementSuggestions[0].elements.join(",")}` : "No specific element preference",
      value: elementSuggestions.flatMap(s => s.elements).slice(0, 5).join(", ") || "open",
      status: elementSuggestions.length > 0 ? "feasible" : "challenging",
    },
  ];

  return {
    targetTc,
    muStar: muStarClamped,
    requiredLambda: {
      min: Math.round(lambdaMin * 100) / 100,
      max: Math.round(lambdaMax * 100) / 100,
      optimal: Math.round(optimalLambda * 1000) / 1000,
    },
    requiredOmegaLog: {
      min: Math.round(omegaMin),
      max: Math.round(omegaMax),
      optimal: Math.round(optimalOmegaLog),
    },
    feasibilityScore: feasibility.score,
    feasibilityNote: feasibility.note,
    structuralTargets,
    elementSuggestions,
    constraintChain: chain,
  };
}

export function evaluateFormulaAgainstConstraints(
  formula: string,
  solution: ConstraintSolution,
): {
  formula: string;
  satisfiesLambda: boolean;
  satisfiesOmega: boolean;
  lambdaValue: number;
  omegaLogValue: number;
  predictedTc: number;
  gapToTarget: number;
  matchScore: number;
} {
  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);
    const omegaLogK = coupling.omegaLog * 1.44;

    const satisfiesLambda = coupling.lambda >= solution.requiredLambda.min * 0.8 &&
      coupling.lambda <= solution.requiredLambda.max * 1.2;
    const satisfiesOmega = omegaLogK >= solution.requiredOmegaLog.min * 0.8 &&
      omegaLogK <= solution.requiredOmegaLog.max * 1.2;

    const tc = mcMillanTc(coupling.lambda, omegaLogK, solution.muStar);
    const gapToTarget = Math.abs(tc - solution.targetTc);

    const lambdaDist = Math.abs(coupling.lambda - solution.requiredLambda.optimal) / Math.max(0.1, solution.requiredLambda.optimal);
    const omegaDist = Math.abs(omegaLogK - solution.requiredOmegaLog.optimal) / Math.max(1, solution.requiredOmegaLog.optimal);
    const matchScore = Math.max(0, 1 - 0.5 * lambdaDist - 0.5 * omegaDist);

    return {
      formula,
      satisfiesLambda,
      satisfiesOmega,
      lambdaValue: Math.round(coupling.lambda * 1000) / 1000,
      omegaLogValue: Math.round(omegaLogK),
      predictedTc: Math.round(tc * 10) / 10,
      gapToTarget: Math.round(gapToTarget * 10) / 10,
      matchScore: Math.round(matchScore * 1000) / 1000,
    };
  } catch {
    return {
      formula,
      satisfiesLambda: false,
      satisfiesOmega: false,
      lambdaValue: 0,
      omegaLogValue: 0,
      predictedTc: 0,
      gapToTarget: solution.targetTc,
      matchScore: 0,
    };
  }
}

const constraintCache = new Map<string, ConstraintSolution>();

export function getCachedConstraints(targetTc: number, muStar: number = 0.10): ConstraintSolution {
  const key = `${targetTc}-${muStar.toFixed(3)}`;
  if (constraintCache.has(key)) return constraintCache.get(key)!;
  const solution = solveConstraints(targetTc, muStar);
  constraintCache.set(key, solution);
  if (constraintCache.size > 50) {
    const firstKey = constraintCache.keys().next().value;
    if (firstKey) constraintCache.delete(firstKey);
  }
  return solution;
}

export function getConstraintGuidanceForGenerator(targetTc: number): {
  lambdaRange: [number, number];
  omegaLogRange: [number, number];
  preferredElements: string[];
  preferredStructures: string[];
  feasibility: number;
} {
  const solution = getCachedConstraints(targetTc);
  return {
    lambdaRange: [solution.requiredLambda.min, solution.requiredLambda.max],
    omegaLogRange: [solution.requiredOmegaLog.min, solution.requiredOmegaLog.max],
    preferredElements: solution.elementSuggestions.flatMap(s => s.elements).slice(0, 8),
    preferredStructures: solution.structuralTargets.map(s => s.type).slice(0, 4),
    feasibility: solution.feasibilityScore,
  };
}
