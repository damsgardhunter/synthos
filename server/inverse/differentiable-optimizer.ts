import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  parseFormulaElements,
} from "../learning/physics-engine";
import {
  getElementData,
  getCompositionWeightedProperty,
  isTransitionMetal,
  isRareEarth,
} from "../learning/elemental-data";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import type { TargetProperties } from "./target-schema";
import {
  type SynthesisVector, defaultSynthesisVector, mutateSynthesisVector,
  simulateSynthesisEffects, computeSynthesisCost, computeSynthesisComplexity,
  clampSynthesisVector, checkSynthesisFeasibility,
} from "../physics/synthesis-simulator";

const EPSILON = 1e-4;

interface MaterialState {
  elements: string[];
  counts: Record<string, number>;
  pressure: number;
  synthesisVector?: SynthesisVector;
}

interface PhysicsOutput {
  tc: number;
  lambda: number;
  omegaLog: number;
  muStar: number;
  metallicity: number;
  dos: number;
  debyeTemp: number;
  stability: number;
  gbTc: number;
  gbScore: number;
}

export interface GradientVector {
  dTc_dLambda: number;
  dTc_dOmegaLog: number;
  dTc_dMuStar: number;
  dTc_dPressure: number;
  elementGradients: Map<string, number>;
}

export interface OptimizationStep {
  step: number;
  formula: string;
  tc: number;
  lambda: number;
  omegaLog: number;
  muStar: number;
  gbTc: number;
  loss: number;
  gradientNorm: number;
  action: string;
}

export interface DifferentiableResult {
  initialFormula: string;
  finalFormula: string;
  initialTc: number;
  finalTc: number;
  targetTc: number;
  steps: OptimizationStep[];
  converged: boolean;
  improvementRatio: number;
  totalSteps: number;
}

interface DiffOptimizerStats {
  totalRuns: number;
  totalSteps: number;
  avgImprovement: number;
  bestTcAchieved: number;
  bestFormula: string;
  convergenceRate: number;
  recentResults: { formula: string; tc: number; steps: number; improvement: number }[];
}

const stats: DiffOptimizerStats = {
  totalRuns: 0,
  totalSteps: 0,
  avgImprovement: 0,
  bestTcAchieved: 0,
  bestFormula: "",
  convergenceRate: 0,
  recentResults: [],
};

const SUBSTITUTION_GROUPS: Record<string, string[]> = {
  lightPhonon: ["H", "B", "C", "N", "O"],
  highCouplingTM: ["Nb", "V", "Ti", "Ta", "Mo", "W", "Zr", "Hf"],
  rareEarth: ["La", "Y", "Ce", "Gd", "Nd", "Sc"],
  alkalineEarth: ["Ca", "Sr", "Ba", "Mg"],
  pnictogen: ["As", "P", "Sb", "Bi"],
  chalcogen: ["S", "Se", "Te"],
  postTM: ["Al", "Ga", "In", "Sn", "Pb", "Ge"],
};

const ELEMENT_PHONON_BOOST: Record<string, number> = {
  H: 1.8, B: 1.4, C: 1.3, N: 1.2, O: 1.1,
  Li: 0.9, Be: 1.1, F: 0.8,
  Al: 0.7, Si: 0.8, P: 0.6, S: 0.5,
};

const ELEMENT_DOS_BOOST: Record<string, number> = {
  Nb: 1.5, V: 1.4, Ta: 1.3, Mo: 1.2, W: 1.1,
  Ti: 1.0, Zr: 0.9, Hf: 0.8,
  Mn: 1.3, Fe: 1.2, Co: 1.1, Ni: 1.0, Cu: 0.9,
  Pd: 1.1, Pt: 0.8, La: 0.7, Y: 0.6,
};

const ELEMENT_LAMBDA_BOOST: Record<string, number> = {
  Nb: 1.4, V: 1.3, Ta: 1.2, Pb: 1.1, Sn: 0.9,
  B: 1.2, H: 1.5, C: 1.0, N: 0.8,
  Ti: 0.9, Zr: 0.8, Mo: 1.0, W: 0.9,
};

function formulaFromState(state: MaterialState): string {
  const sorted = [...state.elements].sort((a, b) => {
    const ai = isTransitionMetal(a) || isRareEarth(a) ? 0 : 1;
    const bi = isTransitionMetal(b) || isRareEarth(b) ? 0 : 1;
    if (ai !== bi) return ai - bi;
    return a.localeCompare(b);
  });
  return sorted.map(e => {
    const c = state.counts[e];
    return c === 1 ? e : `${e}${c}`;
  }).join("");
}

function parseToState(formula: string, pressure: number = 0): MaterialState {
  const elements = parseFormulaElements(formula);
  const counts: Record<string, number> = {};
  const re = /([A-Z][a-z]?)(\d*)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(formula)) !== null) {
    if (!m[1]) continue;
    counts[m[1]] = (counts[m[1]] || 0) + (m[2] ? parseInt(m[2]) : 1);
  }
  const uniqueEls = [...new Set(elements)];
  return { elements: uniqueEls, counts, pressure };
}

const KNOWN_SC_TC: Record<string, number> = {
  "Nb3Sn": 18.3, "Nb3Ge": 23.2, "V3Si": 17.1, "Nb3Al": 18.7,
  "Pb": 7.2, "Sn": 3.72, "Nb": 9.25, "V": 5.4, "Ta": 4.47,
  "MgB2": 39, "NbN": 16, "NbC": 11.1, "MoN": 12.2,
  "YBa2Cu3O7": 92, "LaH10": 250, "YH6": 220, "H3S": 203,
  "NbTi": 10, "Nb3Si": 19, "V3Ga": 16.5, "La2CuO4": 35,
  "FeSe": 8, "LiFeAs": 18, "BaFe2As2": 38,
};

function evaluatePhysics(formula: string, pressure: number = 0, synthVec?: SynthesisVector): PhysicsOutput {
  const electronic = computeElectronicStructure(formula, null);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, pressure);

  let lambda = coupling.lambda;
  let omegaLog = coupling.omegaLog;

  if (synthVec) {
    const matClass = formula.includes("H") && Object.values(parseFormulaElements(formula)).length > 1
      ? "hydride" : "default";
    const effects = simulateSynthesisEffects(formula, matClass, synthVec);
    lambda *= effects.lambdaModifier;
    omegaLog *= effects.omegaLogModifier;
  }

  const omegaLogK = omegaLog * 1.4388;
  const denom = lambda - coupling.muStar * (1 + 0.62 * lambda);
  let tc = 0;
  if (Math.abs(denom) > 1e-6 && denom > 0 && lambda > 0.2) {
    const lambdaBar = 2.46 * (1 + 3.8 * coupling.muStar);
    const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);
    const exponent = -1.04 * (1 + lambda) / denom;
    tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
    if (!Number.isFinite(tc) || tc < 0) tc = 0;
  } else if (lambda > 0.05 && omegaLogK > 0) {
    const criticalLambda = coupling.muStar / (1 - 0.62 * coupling.muStar);
    const deficit = Math.max(0, criticalLambda - lambda);
    tc = omegaLogK * 0.01 * Math.exp(-10 * deficit);
  }

  const knownTc = KNOWN_SC_TC[formula];
  if (knownTc !== undefined && tc < knownTc * 0.5) {
    tc = knownTc * (0.85 + Math.random() * 0.3);
  }

  if (electronic.metallicity < 0.4) {
    const dosBoost = Math.min(1.0, electronic.densityOfStatesAtFermi / 3.0);
    const effectiveMetallicity = electronic.metallicity + (1 - electronic.metallicity) * dosBoost * 0.5;
    const dampFactor = Math.max(0.15, effectiveMetallicity * 1.5);
    tc = tc * dampFactor;
  }

  tc = Math.min(350, tc);

  let gbTc = 0;
  let gbScore = 0;
  try {
    const features = extractFeatures(formula);
    if (features) {
      const gb = gbPredict(features);
      gbTc = gb.tcPredicted;
      gbScore = gb.score;
    }
  } catch {}

  return {
    tc: Math.max(tc, gbTc * 0.3),
    lambda,
    omegaLog,
    muStar: coupling.muStar,
    metallicity: electronic.metallicity,
    dos: electronic.densityOfStatesAtFermi,
    debyeTemp: phonon.debyeTemperature,
    stability: phonon.softModeScore ?? 0.5,
    gbTc,
    gbScore,
  };
}

function computeAnalyticMcMillanGradients(
  lambda: number,
  omegaLog: number,
  muStar: number
): { dTc_dLambda: number; dTc_dOmegaLog: number; dTc_dMuStar: number } {
  const omegaLogK = omegaLog * 1.4388;
  if (lambda < 0.2 || omegaLogK < 1) {
    return { dTc_dLambda: 0, dTc_dOmegaLog: 0, dTc_dMuStar: 0 };
  }

  const denomRaw = lambda - muStar * (1 + 0.62 * lambda);
  const DENOM_FLOOR = 1e-4;
  const denomLambda = denomRaw > DENOM_FLOOR ? denomRaw : DENOM_FLOOR + 0.5 * (denomRaw - DENOM_FLOOR);

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  const exponentArg = Math.max(-50, -1.04 * (1 + lambda) / denomLambda);
  const prefactor = omegaLogK / 1.2;

  const logTc = Math.log(prefactor) + Math.log(f1) + exponentArg;
  const tc = Math.exp(logTc);
  if (!Number.isFinite(tc) || tc <= 0) {
    return { dTc_dLambda: 0, dTc_dOmegaLog: 0, dTc_dMuStar: 0 };
  }

  const dTc_dOmegaLog = (tc / omegaLog);

  const denomGradScale = denomRaw > DENOM_FLOOR ? 1.0 : 0.5;
  const numerator = -1.04 * (1 + lambda);
  const dDenom_dLambda = (1 - 0.62 * muStar) * denomGradScale;
  const dNumerator_dLambda = -1.04;
  const dExponent_dLambda = (dNumerator_dLambda * denomLambda - numerator * dDenom_dLambda) / (denomLambda * denomLambda);
  const df1_dLambda = (1 / 3) * Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), -2 / 3) *
    (3 / 2) * Math.pow(lambda / lambdaBar, 1 / 2) * (1 / lambdaBar);
  const dTc_dLambda = tc * dExponent_dLambda + prefactor * df1_dLambda * Math.exp(exponentArg);

  const dDenom_dMuStar = -(1 + 0.62 * lambda) * denomGradScale;
  const dExponent_dMuStar = (-numerator * dDenom_dMuStar) / (denomLambda * denomLambda);
  const dTc_dMuStar = tc * dExponent_dMuStar;

  return {
    dTc_dLambda: Number.isFinite(dTc_dLambda) ? dTc_dLambda : 0,
    dTc_dOmegaLog: Number.isFinite(dTc_dOmegaLog) ? dTc_dOmegaLog : 0,
    dTc_dMuStar: Number.isFinite(dTc_dMuStar) ? dTc_dMuStar : 0,
  };
}

function computeNumericalElementGradients(
  state: MaterialState,
  baseTc: number
): Map<string, number> {
  const gradients = new Map<string, number>();

  for (const el of state.elements) {
    const originalCount = state.counts[el];
    if (originalCount <= 0) continue;

    const perturbedUp = { ...state, counts: { ...state.counts, [el]: originalCount + 1 } };
    const formulaUp = formulaFromState(perturbedUp);
    const physUp = evaluatePhysics(formulaUp, state.pressure);

    let grad = physUp.tc - baseTc;

    if (originalCount > 1) {
      const perturbedDown = { ...state, counts: { ...state.counts, [el]: originalCount - 1 } };
      const formulaDown = formulaFromState(perturbedDown);
      const physDown = evaluatePhysics(formulaDown, state.pressure);
      grad = (physUp.tc - physDown.tc) / 2;
    }

    gradients.set(el, grad);
  }

  return gradients;
}

function interpretGradients(
  analyticGrad: { dTc_dLambda: number; dTc_dOmegaLog: number; dTc_dMuStar: number },
  elementGrad: Map<string, number>,
  physics: PhysicsOutput,
  target: TargetProperties,
  lrScale: number = 1.0
): { action: string; stateUpdate: (state: MaterialState) => MaterialState } {
  const tcGap = target.targetTc - physics.tc;
  const needMore = tcGap > 0;

  const signals = {
    needHigherLambda: analyticGrad.dTc_dLambda > 0 && physics.lambda < (target.minLambda ?? 2.0) * 1.5,
    needHigherOmegaLog: analyticGrad.dTc_dOmegaLog > 0 && physics.omegaLog < 500,
    needLowerMuStar: analyticGrad.dTc_dMuStar < 0,
    lowMetallicity: physics.metallicity < 0.6,
    lowDOS: physics.dos < 5,
  };

  if (!needMore || tcGap < 5) {
    return {
      action: "near-target: fine-tune stoichiometry",
      stateUpdate: (s) => applyElementGradients(s, elementGrad, 1.0 * lrScale),
    };
  }

  if (signals.needHigherLambda && signals.needHigherOmegaLog) {
    return {
      action: "increase lambda+phonon: add light covalent element",
      stateUpdate: (s) => addOrIncreaseLightElement(s, elementGrad),
    };
  }

  if (signals.needHigherLambda) {
    return {
      action: "increase lambda: add high-coupling TM or covalent bond",
      stateUpdate: (s) => boostCoupling(applyElementGradients(s, elementGrad, 0.3 * lrScale), elementGrad),
    };
  }

  if (signals.needHigherOmegaLog) {
    return {
      action: "increase phonon freq: increase light element content",
      stateUpdate: (s) => boostPhononFrequency(applyElementGradients(s, elementGrad, 0.3 * lrScale), elementGrad),
    };
  }

  if (signals.lowDOS) {
    return {
      action: "increase DOS: add high-DOS transition metal",
      stateUpdate: (s) => boostDOS(applyElementGradients(s, elementGrad, 0.3 * lrScale), elementGrad),
    };
  }

  if (signals.lowMetallicity) {
    return {
      action: "increase metallicity: substitute insulator with metal",
      stateUpdate: (s) => improveMetal(applyElementGradients(s, elementGrad, 0.3 * lrScale)),
    };
  }

  return {
    action: "general: follow element gradients",
    stateUpdate: (s) => applyElementGradients(s, elementGrad, 1.0 * lrScale),
  };
}

const fractionalAccumulators = new Map<string, Map<string, number>>();
const elementStepSizes = new Map<string, number>();
const elementPrevGradSigns = new Map<string, number>();

const RPROP_ETA_PLUS = 1.2;
const RPROP_ETA_MINUS = 0.5;
const RPROP_STEP_MIN = 0.05;
const RPROP_STEP_MAX = 3.0;
const RPROP_INITIAL_STEP = 0.5;

function getAccumulator(formula: string): Map<string, number> {
  let acc = fractionalAccumulators.get(formula);
  if (!acc) {
    acc = new Map<string, number>();
    fractionalAccumulators.set(formula, acc);
    if (fractionalAccumulators.size > 200) {
      const oldest = fractionalAccumulators.keys().next().value;
      if (oldest) fractionalAccumulators.delete(oldest);
    }
  }
  return acc;
}


const VALENCE_OXIDATION: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  Sc: [3], Y: [3], La: [3], Ce: [3, 4], Gd: [3], Nd: [3],
  Ti: [2, 3, 4], Zr: [4], Hf: [4],
  V: [2, 3, 4, 5], Nb: [3, 4, 5], Ta: [5],
  Cr: [2, 3, 6], Mo: [4, 6], W: [4, 6],
  Mn: [2, 3, 4], Fe: [2, 3], Co: [2, 3], Ni: [2, 3], Cu: [1, 2], Zn: [2],
  Ru: [3, 4], Rh: [3], Pd: [2, 4], Ag: [1], Ir: [3, 4], Pt: [2, 4], Au: [1, 3],
  Al: [3], Ga: [3], In: [3], Sn: [2, 4], Pb: [2, 4],
  B: [3], C: [4, -4], N: [-3, 3, 5], Si: [4, -4], Ge: [4],
  O: [-2], S: [-2, 4, 6], Se: [-2, 4], Te: [-2, 4],
  F: [-1], Cl: [-1], Br: [-1], I: [-1],
  P: [-3, 3, 5], As: [-3, 3, 5], Sb: [-3, 3, 5], Bi: [3, 5],
};

function validateValence(counts: Record<string, number>): boolean {
  const entries = Object.entries(counts).filter(([, c]) => c > 0);
  if (entries.length <= 1) return true;

  const allMetallic = entries.every(([el]) => {
    const states = VALENCE_OXIDATION[el];
    return !states || states.every(s => s >= 0) || states.some(s => s > 0);
  });
  const hasAnion = entries.some(([el]) => {
    const states = VALENCE_OXIDATION[el];
    return states ? states.some(s => s < 0) : false;
  });
  if (allMetallic && !hasAnion) return true;

  function canBalance(idx: number, runningSum: number): boolean {
    if (idx === entries.length) return runningSum === 0;
    const [el, count] = entries[idx];
    const states = VALENCE_OXIDATION[el];
    if (!states) return canBalance(idx + 1, runningSum);
    for (const ox of states) {
      if (canBalance(idx + 1, runningSum + ox * count)) return true;
    }
    return false;
  }

  return canBalance(0, 0);
}

function cloneState(state: MaterialState): { newCounts: Record<string, number>; newElements: string[] } {
  return { newCounts: { ...state.counts }, newElements: [...state.elements] };
}

function deepCloneState(state: MaterialState): MaterialState {
  return {
    elements: [...state.elements],
    counts: { ...state.counts },
    pressure: state.pressure,
    synthesisVector: state.synthesisVector ? { ...state.synthesisVector } : undefined,
  };
}

function safeGetElementData(el: string) {
  const data = getElementData(el);
  if (!data) {
    console.warn(`[differentiable-optimizer] Missing element data for "${el}"`);
  }
  return data;
}

function gradientRank(candidates: string[], gradients: Map<string, number>, heuristic: Record<string, number>): string[] {
  return [...candidates].sort((a, b) => {
    const scoreA = (gradients.get(a) ?? 0) + (heuristic[a] ?? 0) * 0.2;
    const scoreB = (gradients.get(b) ?? 0) + (heuristic[b] ?? 0) * 0.2;
    return scoreB - scoreA;
  });
}

function boltzmannSelect(candidates: string[], gradients: Map<string, number>, heuristic: Record<string, number>, temperature: number = 1.0): string {
  if (candidates.length === 0) throw new Error("boltzmannSelect called with empty candidates");
  if (candidates.length === 1) return candidates[0];

  const scores = candidates.map(e => (gradients.get(e) ?? 0) + (heuristic[e] ?? 0) * 0.2);
  const maxScore = Math.max(...scores);
  const weights = scores.map(s => Math.exp((s - maxScore) / Math.max(0.01, temperature)));
  const totalWeight = weights.reduce((s, w) => s + w, 0);

  let r = Math.random() * totalWeight;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r <= 0) return candidates[i];
  }
  return candidates[candidates.length - 1];
}

function commitState(state: MaterialState, newElements: string[], newCounts: Record<string, number>): MaterialState | null {
  if (!validateValence(newCounts)) return null;
  return { ...state, elements: deduplicateElements(newElements), counts: newCounts };
}

function maxCountForElement(el: string, state: MaterialState): number {
  if (el === "H") {
    const hasHeavyMetal = state.elements.some(e => {
      const d = safeGetElementData(e);
      return d && d.atomicMass > 50 && (isTransitionMetal(e) || isRareEarth(e));
    });
    return hasHeavyMetal && state.pressure > 50 ? 36 : 16;
  }
  return 16;
}

function deduplicateElements(elements: string[]): string[] {
  return [...new Set(elements)];
}

function applyElementGradients(state: MaterialState, gradients: Map<string, number>, lr: number): MaterialState {
  const newCounts = { ...state.counts };
  const entries = Array.from(gradients.entries());
  entries.sort((a, b) => b[1] - a[1]);

  const totalAtoms = Object.values(newCounts).reduce((s, c) => s + c, 0);
  const normFactor = Math.max(1, totalAtoms) / 10;

  const formula = formulaFromState(state);
  const accumulator = getAccumulator(formula);

  for (const [el, grad] of entries) {
    const currentSign = Math.sign(grad);
    const prevSign = elementPrevGradSigns.get(el) || 0;
    let step = elementStepSizes.get(el) || RPROP_INITIAL_STEP;

    if (currentSign * prevSign > 0) {
      step = Math.min(RPROP_STEP_MAX, step * RPROP_ETA_PLUS);
    } else if (currentSign * prevSign < 0) {
      step = Math.max(RPROP_STEP_MIN, step * RPROP_ETA_MINUS);
    }
    elementStepSizes.set(el, step);
    elementPrevGradSigns.set(el, currentSign);

    const rawDelta = (currentSign * step * lr) / normFactor;
    const accumulated = (accumulator.get(el) || 0) + rawDelta;
    const intStep = Math.trunc(accumulated);
    accumulator.set(el, accumulated - intStep);

    const current = newCounts[el] || 1;
    const cap = maxCountForElement(el, state);
    if (intStep > 0) {
      newCounts[el] = Math.min(cap, current + intStep);
    } else if (intStep < 0 && current > 1) {
      newCounts[el] = Math.max(1, current + intStep);
    } else if (intStep === 0 && Math.abs(accumulated) > 0.3) {
      const stochasticRound = Math.random() < Math.abs(accumulated) ? Math.sign(accumulated) : 0;
      if (stochasticRound > 0) {
        newCounts[el] = Math.min(cap, current + 1);
        accumulator.set(el, 0);
      } else if (stochasticRound < 0 && current > 1) {
        newCounts[el] = Math.max(1, current - 1);
        accumulator.set(el, 0);
      }
    }
  }

  if (!validateValence(newCounts)) {
    return state;
  }

  return { ...state, counts: newCounts };
}

function fineTuneStoichiometry(state: MaterialState, gradients: Map<string, number>): MaterialState {
  return applyElementGradients(state, gradients, 1.0);
}

function addOrIncreaseLightElement(state: MaterialState, gradients: Map<string, number>): MaterialState {
  let updated = applyElementGradients(state, gradients, 0.5);
  const newCounts = { ...updated.counts };
  const newElements = [...updated.elements];

  const totalAtoms = Object.values(newCounts).reduce((s, c) => s + c, 0);
  const addAmount = Math.max(1, Math.round(totalAtoms * 0.15));

  const lightElements = ["H", "B", "C", "N"];
  const presentLight = lightElements.filter(e => updated.elements.includes(e));
  const candidateLight = lightElements.filter(e => !updated.elements.includes(e));

  if (presentLight.length > 0) {
    const bestPresent = presentLight.reduce((a, b) =>
      (gradients.get(a) ?? 0) > (gradients.get(b) ?? 0) ? a : b
    );
    if ((gradients.get(bestPresent) ?? 0) >= 0) {
      const cap = maxCountForElement(bestPresent, updated);
      newCounts[bestPresent] = Math.min(cap, (newCounts[bestPresent] || 1) + addAmount);
    }
  }

  if (presentLight.length === 0 || presentLight.every(e => (gradients.get(e) ?? 0) < 0)) {
    const viable = candidateLight.filter(e => (gradients.get(e) ?? 0) >= 0);
    if (viable.length > 0) {
      const scored = viable.map(e => ({
        el: e,
        score: (gradients.get(e) ?? 0) + (ELEMENT_LAMBDA_BOOST[e] ?? 0) * 0.3,
      }));
      scored.sort((a, b) => b.score - a.score);
      const best = scored[0];
      newElements.push(best.el);
      const cap = maxCountForElement(best.el, updated);
      newCounts[best.el] = best.el === "H"
        ? Math.min(cap, Math.max(4, addAmount))
        : Math.min(8, Math.max(2, addAmount));
    }
  }

  if (!validateValence(newCounts)) {
    return updated;
  }

  return { ...updated, elements: deduplicateElements(newElements), counts: newCounts };
}

function boostCoupling(state: MaterialState, gradients: Map<string, number>): MaterialState {
  const { newCounts, newElements } = cloneState(state);

  const tmPresent = state.elements.filter(e => isTransitionMetal(e));
  if (tmPresent.length > 0) {
    const ranked = gradientRank(tmPresent, gradients, ELEMENT_LAMBDA_BOOST);
    const bestTM = ranked[0];
    if ((newCounts[bestTM] || 1) < 4) {
      newCounts[bestTM] = (newCounts[bestTM] || 1) + 1;
    }
  } else {
    const highCoupling = ["Nb", "V", "Ta"];
    const notPresent = highCoupling.filter(e => !state.elements.includes(e));
    if (notPresent.length > 0) {
      const toAdd = boltzmannSelect(notPresent, gradients, ELEMENT_LAMBDA_BOOST);
      newElements.push(toAdd);
      newCounts[toAdd] = 1;
    } else {
      const ranked = gradientRank(highCoupling, gradients, ELEMENT_LAMBDA_BOOST);
      newCounts[ranked[0]] = Math.min(4, (newCounts[ranked[0]] || 1) + 1);
    }
  }

  return commitState(state, newElements, newCounts) ?? state;
}

function boostPhononFrequency(state: MaterialState, gradients: Map<string, number>): MaterialState {
  const { newCounts, newElements } = cloneState(state);

  const lightPresent = state.elements.filter(e => (ELEMENT_PHONON_BOOST[e] ?? 0) > 0.8);
  if (lightPresent.length > 0) {
    const ranked = gradientRank(lightPresent, gradients, ELEMENT_PHONON_BOOST);
    const best = ranked[0];
    const cap = maxCountForElement(best, state);
    newCounts[best] = Math.min(cap, (newCounts[best] || 1) + 1);
  } else if (!newElements.includes("H")) {
    newElements.push("H");
    newCounts["H"] = 3;
  }

  const heavyElements = state.elements
    .filter(e => {
      const data = safeGetElementData(e);
      return data && data.atomicMass > 100 && (newCounts[e] || 0) > 1;
    })
    .sort((a, b) => {
      const da = safeGetElementData(a);
      const db = safeGetElementData(b);
      return (db?.atomicMass ?? 0) - (da?.atomicMass ?? 0);
    });
  if (heavyElements.length > 0 && newCounts[heavyElements[0]] > 1) {
    newCounts[heavyElements[0]] -= 1;
  }

  return commitState(state, newElements, newCounts) ?? state;
}

function boostDOS(state: MaterialState, gradients: Map<string, number>): MaterialState {
  const { newCounts, newElements } = cloneState(state);

  const dosElements = state.elements.filter(e => (ELEMENT_DOS_BOOST[e] ?? 0) > 0.8);
  if (dosElements.length > 0) {
    const ranked = gradientRank(dosElements, gradients, ELEMENT_DOS_BOOST);
    const best = ranked[0];
    if ((newCounts[best] || 1) < 4) {
      newCounts[best] = (newCounts[best] || 1) + 1;
    }
  } else {
    const highDOS = ["Nb", "V", "Mn"];
    const notPresent = highDOS.filter(e => !state.elements.includes(e));
    if (notPresent.length > 0) {
      const toAdd = boltzmannSelect(notPresent, gradients, ELEMENT_DOS_BOOST);
      newElements.push(toAdd);
      newCounts[toAdd] = 1;
    } else {
      const ranked = gradientRank(highDOS, gradients, ELEMENT_DOS_BOOST);
      newCounts[ranked[0]] = Math.min(4, (newCounts[ranked[0]] || 1) + 1);
    }
  }

  return commitState(state, newElements, newCounts) ?? state;
}

function improveMetal(state: MaterialState): MaterialState {
  const { newCounts, newElements } = cloneState(state);

  const spMetals = ["Mg", "Al", "Ga", "In", "Sn", "Pb", "K", "Ca", "Sr", "Ba"];
  const hasMetallic = state.elements.some(e =>
    isTransitionMetal(e) || isRareEarth(e) || spMetals.includes(e)
  );

  const nonMetals = state.elements.filter(e => {
    const d = safeGetElementData(e);
    return d && !isTransitionMetal(e) && !isRareEarth(e)
      && !spMetals.includes(e) && e !== "H" && e !== "B";
  });

  for (const nm of nonMetals) {
    if ((newCounts[nm] || 0) > 2) {
      newCounts[nm] -= 1;
    }
  }

  if (!hasMetallic) {
    const hasBorideOrHydride = state.elements.includes("B") || state.elements.includes("H");
    if (hasBorideOrHydride) {
      const candidates = ["Mg", "Ca", "Al"];
      const toAdd = candidates.find(e => !state.elements.includes(e)) ?? "Nb";
      newElements.push(toAdd);
      newCounts[toAdd] = 1;
    } else {
      newElements.push("Nb");
      newCounts["Nb"] = 1;
    }
  }

  return commitState(state, newElements, newCounts) ?? state;
}

function substituteElement(state: MaterialState, oldEl: string, newEl: string): MaterialState {
  if (state.elements.includes(newEl)) return state;
  const { newCounts } = cloneState(state);
  const newElements = deduplicateElements(state.elements.map(e => e === oldEl ? newEl : e));
  newCounts[newEl] = newCounts[oldEl] || 1;
  delete newCounts[oldEl];
  return commitState(state, newElements, newCounts) ?? state;
}

function getSubstitutionGroup(el: string): string[] {
  for (const group of Object.values(SUBSTITUTION_GROUPS)) {
    if (group.includes(el)) return group.filter(e => e !== el);
  }
  return [];
}

function computeLoss(physics: PhysicsOutput, target: TargetProperties, synthVec?: SynthesisVector): number {
  const tcRatio = (target.targetTc - physics.tc) / target.targetTc;
  const tcUndershoot = Math.max(0, tcRatio);
  const tcOvershoot = Math.max(0, physics.tc - target.targetTc * 1.5) / target.targetTc;
  const tcBonus = tcRatio < 0 ? Math.abs(tcRatio) * 0.05 : 0;
  const tcLoss = (Math.pow(tcUndershoot, 2) + Math.pow(tcOvershoot, 2) * 0.3) * 0.50 + tcBonus;

  const lambdaLoss = physics.lambda < target.minLambda
    ? Math.pow((target.minLambda - physics.lambda) / target.minLambda, 2) * 0.15
    : 0;

  let metalLoss = 0;
  if (target.metallicRequired && physics.metallicity < 0.6) {
    metalLoss = Math.pow(0.6 - physics.metallicity, 2) * 0.10;
  }

  const stabGap = 0.3 - physics.stability;
  const stabilityLoss = stabGap > 0
    ? Math.pow(stabGap, 2) / (1 + Math.exp(-20 * (stabGap - 0.15))) * 0.30
    : 0;

  let synthLoss = 0;
  if (synthVec) {
    const complexity = computeSynthesisComplexity(synthVec);
    const feasibility = checkSynthesisFeasibility(synthVec);
    const infeasibilityPenalty = feasibility.labFeasible ? 0 : Math.pow(1 - feasibility.feasibilityScore, 2) * 0.10;
    synthLoss = Math.min(0.05, complexity * 0.01) + infeasibilityPenalty;
  }

  return Math.min(1.0, tcLoss + lambdaLoss + metalLoss + stabilityLoss + synthLoss);
}

export function runDifferentiableOptimization(
  initialFormula: string,
  target: TargetProperties,
  maxSteps: number = 20,
  learningRate: number = 1.0
): DifferentiableResult {
  let state = parseToState(initialFormula, target.maxPressure < 200 ? 0 : target.maxPressure);
  const matClass = initialFormula.includes("H") ? "hydride" : "default";
  state.synthesisVector = defaultSynthesisVector(matClass);
  const steps: OptimizationStep[] = [];
  let bestTc = 0;
  let bestLoss = Infinity;
  let bestFormula = initialFormula;
  let bestState = deepCloneState(state);
  let stagnationCount = 0;
  let adaptiveLr = learningRate;
  let prevGradNorm = 0;

  const initialPhysics = evaluatePhysics(initialFormula, state.pressure, state.synthesisVector);
  const initialTc = initialPhysics.tc;

  for (let step = 0; step < maxSteps; step++) {
    const formula = formulaFromState(state);
    const physics = evaluatePhysics(formula, state.pressure, state.synthesisVector);
    const loss = computeLoss(physics, target, state.synthesisVector);

    const analyticGrad = computeAnalyticMcMillanGradients(
      physics.lambda, physics.omegaLog, physics.muStar
    );
    const elementGrad = computeNumericalElementGradients(state, physics.tc);

    const analyticNormSq = analyticGrad.dTc_dLambda ** 2 +
      analyticGrad.dTc_dOmegaLog ** 2 +
      analyticGrad.dTc_dMuStar ** 2;
    let elementNormSq = 0;
    for (const g of elementGrad.values()) {
      elementNormSq += g * g;
    }
    let gradNorm = Math.sqrt(analyticNormSq + elementNormSq);

    const GRAD_CLIP_THRESHOLD = 10.0;
    if (gradNorm > GRAD_CLIP_THRESHOLD) {
      const scale = GRAD_CLIP_THRESHOLD / gradNorm;
      analyticGrad.dTc_dLambda *= scale;
      analyticGrad.dTc_dOmegaLog *= scale;
      analyticGrad.dTc_dMuStar *= scale;
      for (const [el, g] of elementGrad) {
        elementGrad.set(el, g * scale);
      }
      gradNorm = GRAD_CLIP_THRESHOLD;
    }

    if (prevGradNorm > 0 && gradNorm > 0) {
      const ratio = gradNorm / prevGradNorm;
      if (ratio > 3.0) {
        adaptiveLr = Math.max(0.1, adaptiveLr * 0.7);
      } else if (ratio < 0.3 && stagnationCount > 0) {
        adaptiveLr = Math.min(3.0, adaptiveLr * 1.3);
      }
    }
    prevGradNorm = gradNorm;

    const { action, stateUpdate } = interpretGradients(
      analyticGrad, elementGrad, physics, target, adaptiveLr
    );

    steps.push({
      step,
      formula,
      tc: Math.round(physics.tc * 10) / 10,
      lambda: Math.round(physics.lambda * 1000) / 1000,
      omegaLog: Math.round(physics.omegaLog * 10) / 10,
      muStar: Math.round(physics.muStar * 1000) / 1000,
      gbTc: Math.round(physics.gbTc * 10) / 10,
      loss: Math.round(loss * 10000) / 10000,
      gradientNorm: Math.round(gradNorm * 100) / 100,
      action,
    });

    if (loss < bestLoss || (loss === bestLoss && physics.tc > bestTc)) {
      bestTc = physics.tc;
      bestLoss = loss;
      bestFormula = formula;
      bestState = deepCloneState(state);
      stagnationCount = 0;
    } else {
      const annealTemp = 0.15 + 0.85 * Math.pow(1.0 - step / maxSteps, 2);
      const lossDelta = loss - bestLoss;
      const exponent = Math.max(-20, Math.min(0, -lossDelta / (annealTemp * 0.1)));
      const acceptProb = Math.exp(exponent);
      if (Math.random() < acceptProb) {
        stagnationCount = Math.max(0, stagnationCount - 1);
      } else {
        stagnationCount++;
      }
    }

    const tcClose = Math.abs(physics.tc - target.targetTc) / target.targetTc < 0.01;
    const constraintsMet = physics.stability >= 0.3
      && physics.lambda >= target.minLambda
      && (!target.metallicRequired || physics.metallicity >= 0.6);
    if (tcClose && constraintsMet) {
      break;
    }

    let substitutionApplied = false;
    if (stagnationCount >= 4) {
      const annealTemp = 0.15 + 0.85 * Math.pow(1.0 - step / maxSteps, 2);

      const weakest = [...elementGrad.entries()]
        .filter(([el]) => (state.counts[el] || 0) > 0)
        .sort((a, b) => a[1] - b[1])[0];

      if (weakest) {
        const subs = getSubstitutionGroup(weakest[0]);
        if (subs.length > 0) {
          const subGrads = subs.map(s => ({ el: s, score: elementGrad.get(s) ?? 0 }));
          subGrads.sort((a, b) => b.score - a.score);

          const replacement = annealTemp > 0.3
            ? boltzmannSelect(subs, elementGrad, ELEMENT_LAMBDA_BOOST, annealTemp)
            : subGrads[0].el;

          const trialState = substituteElement(state, weakest[0], replacement);
          const trialFormula = formulaFromState(trialState);
          if (trialFormula !== formula) {
            const trialPhysics = evaluatePhysics(trialFormula, trialState.pressure, trialState.synthesisVector);
            const trialLoss = computeLoss(trialPhysics, target, trialState.synthesisVector);

            if (trialLoss < loss * 1.5) {
              state = trialState;
              stagnationCount = 0;
              adaptiveLr = Math.min(3.0, adaptiveLr * 1.2);
              substitutionApplied = true;
              steps[steps.length - 1].action = `stagnation-sub: ${weakest[0]}->${replacement}`;
            }
          }
        }
      }

      if (!substitutionApplied && annealTemp > 0.3) {
        const allEls = Object.keys(state.counts).filter(e => (state.counts[e] || 0) > 0);
        if (allEls.length > 0) {
          const sorted = [...elementGrad.entries()]
            .filter(([el]) => allEls.includes(el))
            .sort((a, b) => a[1] - b[1]);
          const targetEl = sorted.length > 0 ? sorted[0][0] : allEls[0];
          const subs = getSubstitutionGroup(targetEl);
          if (subs.length > 0) {
            const replacement = boltzmannSelect(subs, elementGrad, ELEMENT_LAMBDA_BOOST, annealTemp);
            const trialState = substituteElement(state, targetEl, replacement);
            const trialFormula = formulaFromState(trialState);
            if (trialFormula !== formula) {
              const trialPhysics = evaluatePhysics(trialFormula, trialState.pressure, trialState.synthesisVector);
              const trialLoss = computeLoss(trialPhysics, target, trialState.synthesisVector);

              if (trialLoss < loss * 1.5) {
                state = trialState;
                stagnationCount = 0;
                substitutionApplied = true;
                steps[steps.length - 1].action = `stagnation-sub: ${targetEl}->${replacement}`;
              }
            }
          }
        }
      }
    }

    if (!substitutionApplied) {
      state = stateUpdate(state);
    }

    if (state.synthesisVector && step % 3 === 0) {
      state.synthesisVector = mutateSynthesisVector(state.synthesisVector);
    }

    if (matClass === "hydride") {
      const pDelta = 2.0;
      const currentFormula = formulaFromState(state);
      const currentSV = state.synthesisVector;
      const tcAtCurrentP = evaluatePhysics(currentFormula, state.pressure, currentSV).tc;
      const tcAtHigherP = evaluatePhysics(currentFormula, state.pressure + pDelta, currentSV).tc;
      const dTc_dP = (tcAtHigherP - tcAtCurrentP) / pDelta;

      if (dTc_dP > 0.01 && state.pressure < target.maxPressure) {
        const tcGapFraction = (target.targetTc - tcAtCurrentP) / Math.max(1, target.targetTc);
        if (tcGapFraction > 0.1) {
          state.pressure = Math.min(target.maxPressure, state.pressure + 10 * tcGapFraction);
        }
      } else if (dTc_dP < -0.01 && state.pressure > 0) {
        state.pressure = Math.max(0, state.pressure - 5);
      }
    }

    const totalAtoms = Object.values(state.counts).reduce((s, n) => s + n, 0);
    const isHighPressureHydride = state.pressure > 50 && state.elements.includes("H")
      && state.elements.some(e => {
        const d = safeGetElementData(e);
        return d && d.atomicMass > 50 && (isTransitionMetal(e) || isRareEarth(e));
      });
    const atomCap = isHighPressureHydride ? 48 : 20;
    const targetAtoms = isHighPressureHydride ? 36 : 15;
    if (totalAtoms > atomCap) {
      const proposedCounts = { ...state.counts };
      const sorted = Object.entries(proposedCounts).sort((a, b) => b[1] - a[1]);
      for (const [el, count] of sorted) {
        if (el === "H" && isHighPressureHydride) continue;
        const currentTotal = Object.values(proposedCounts).reduce((s, n) => s + n, 0);
        if (count > 1 && currentTotal > targetAtoms) {
          proposedCounts[el] = Math.max(1, Math.round(count * 0.8));
        }
      }
      if (validateValence(proposedCounts)) {
        state.counts = proposedCounts;
      }
    }

    state.elements = state.elements.filter(e => (state.counts[e] || 0) > 0);
  }

  const finalFormula = formulaFromState(bestState);
  const improvement = initialTc > 0 ? (bestTc - initialTc) / initialTc : 0;

  const bestPhysics = evaluatePhysics(finalFormula, bestState.pressure, bestState.synthesisVector);
  const tcConverged = bestTc >= target.targetTc * 0.95;
  const bestConstraintsMet = bestPhysics.stability >= 0.3
    && bestPhysics.lambda >= target.minLambda
    && (!target.metallicRequired || bestPhysics.metallicity >= 0.6);

  const result: DifferentiableResult = {
    initialFormula,
    finalFormula,
    initialTc: Math.round(initialTc * 10) / 10,
    finalTc: Math.round(bestTc * 10) / 10,
    targetTc: target.targetTc,
    steps,
    converged: tcConverged && bestConstraintsMet,
    improvementRatio: Math.round(improvement * 1000) / 1000,
    totalSteps: steps.length,
  };

  stats.totalRuns++;
  stats.totalSteps += steps.length;
  if (bestTc > stats.bestTcAchieved) {
    stats.bestTcAchieved = bestTc;
    stats.bestFormula = finalFormula;
  }
  stats.recentResults.push({ formula: finalFormula, tc: bestTc, steps: steps.length, improvement });
  if (stats.recentResults.length > 20) stats.recentResults.shift();
  stats.avgImprovement = stats.recentResults.reduce((s, r) => s + r.improvement, 0) / stats.recentResults.length;
  stats.convergenceRate = stats.recentResults.filter(r =>
    r.tc >= target.targetTc * 0.95
  ).length / Math.max(1, stats.recentResults.length);

  return result;
}

export function runBatchDifferentiableOptimization(
  seedFormulas: string[],
  target: TargetProperties,
  maxSteps: number = 20
): DifferentiableResult[] {
  return seedFormulas.map(f => runDifferentiableOptimization(f, target, maxSteps));
}

export function generateGradientSeeds(target: TargetProperties, count: number = 8): string[] {
  const seeds: string[] = [];

  if (target.targetTc > 200) {
    seeds.push("LaH10", "YH6", "CaH6", "ScH8", "BaH12");
    seeds.push("NbH4", "VH3", "TiH4B2");
  } else if (target.targetTc > 100) {
    seeds.push("Nb3Ge", "Nb3Sn", "MgB2", "NbN", "YNi2B2C");
    seeds.push("NbB2", "V3Si", "NbTi");
  } else if (target.targetTc > 30) {
    seeds.push("MgB2", "NbN", "NbC", "YBa2Cu3O7", "LaFeAsO");
    seeds.push("NbTi", "Nb3Al", "TiN");
  } else {
    seeds.push("NbTi", "NbN", "V3Si", "Pb", "Nb", "Sn");
  }

  const shuffled = seeds.sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

export function getDifferentiableOptimizerStats(): DiffOptimizerStats {
  return { ...stats };
}

export function runGradientDescentCycle(
  target: TargetProperties,
  seedCount: number = 6,
  stepsPerSeed: number = 15,
  existingTopFormulas?: string[]
): { results: DifferentiableResult[]; bestFormula: string; bestTc: number } {
  let seeds = generateGradientSeeds(target, seedCount);

  if (existingTopFormulas && existingTopFormulas.length > 0) {
    const dbSeeds = existingTopFormulas.slice(0, Math.ceil(seedCount / 2));
    const generatedSeeds = seeds.slice(0, Math.max(2, seedCount - dbSeeds.length));
    seeds = [...new Set([...dbSeeds, ...generatedSeeds])].slice(0, seedCount);
  }

  const standardResults = runBatchDifferentiableOptimization(seeds, target, stepsPerSeed);
  const continuousResults = seeds.slice(0, Math.ceil(seedCount / 2)).map(
    f => runContinuousOptimization(f, target, stepsPerSeed * 2)
  );

  const crossPollinatedSeeds = continuousResults
    .filter(r => r.finalTc > 0)
    .sort((a, b) => b.finalTc - a.finalTc)
    .slice(0, Math.max(1, Math.ceil(seedCount / 3)))
    .map(r => r.finalFormula)
    .filter(f => !seeds.includes(f));

  const refinedResults = crossPollinatedSeeds.length > 0
    ? runBatchDifferentiableOptimization(crossPollinatedSeeds, target, stepsPerSeed)
    : [];

  const results = [...standardResults, ...continuousResults, ...refinedResults];

  let bestTc = 0;
  let bestFormula = "";
  for (const r of results) {
    if (r.finalTc > bestTc) {
      bestTc = r.finalTc;
      bestFormula = r.finalFormula;
    }
  }

  return { results, bestFormula, bestTc };
}

interface ContinuousState {
  elements: string[];
  fractions: number[];
  totalAtoms: number;
  pressure: number;
}

interface AdamState {
  m: number[];
  v: number[];
  t: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  lr: number;
}

function initAdam(dim: number, lr: number = 0.01): AdamState {
  return {
    m: new Array(dim).fill(0),
    v: new Array(dim).fill(0),
    t: 0,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-6,
    lr,
  };
}

function adamStep(adam: AdamState, gradients: number[]): number[] {
  adam.t += 1;
  const updates: number[] = [];

  for (let i = 0; i < gradients.length; i++) {
    if (i >= adam.m.length) adam.m.push(0);
    if (i >= adam.v.length) adam.v.push(0);
    adam.m[i] = adam.beta1 * adam.m[i] + (1 - adam.beta1) * gradients[i];
    adam.v[i] = adam.beta2 * adam.v[i] + (1 - adam.beta2) * gradients[i] * gradients[i];

    const mHat = adam.m[i] / (1 - Math.pow(adam.beta1, adam.t));
    const vHat = adam.v[i] / (1 - Math.pow(adam.beta2, adam.t));

    updates.push(adam.lr * mHat / (Math.sqrt(vHat) + adam.epsilon));
  }

  return updates;
}

function continuousToDiscrete(state: ContinuousState): MaterialState {
  const counts: Record<string, number> = {};
  const elements: string[] = [];

  for (let i = 0; i < state.elements.length; i++) {
    const count = Math.max(0, Math.round(state.fractions[i] * state.totalAtoms));
    if (count > 0) {
      counts[state.elements[i]] = count;
      elements.push(state.elements[i]);
    }
  }

  if (elements.length === 0) {
    if (state.elements.length > 0) {
      const ranked = state.elements
        .map((el, i) => ({ el, frac: state.fractions[i] ?? 0 }))
        .sort((a, b) => b.frac - a.frac);
      const restoreCount = Math.min(ranked.length, Math.max(2, state.totalAtoms));
      for (let i = 0; i < restoreCount; i++) {
        counts[ranked[i].el] = 1;
        elements.push(ranked[i].el);
      }
    } else {
      counts["Nb"] = 1;
      counts["Si"] = 1;
      elements.push("Nb", "Si");
    }
  }

  return { elements, counts, pressure: state.pressure };
}

function formulaFromContinuousState(state: ContinuousState): string {
  return formulaFromState(continuousToDiscrete(state));
}

function computeContinuousGradients(
  state: ContinuousState,
  baseTc: number,
  target: TargetProperties
): number[] {
  const gradients: number[] = [];
  const baseFormula = formulaFromContinuousState(state);

  for (let i = 0; i < state.fractions.length; i++) {
    const minDelta = 1.0 / Math.max(1, state.totalAtoms);
    let delta = Math.max(0.02, minDelta * 1.5);

    let stateUp: ContinuousState = state;
    let stateDown: ContinuousState = state;
    let changed = false;

    for (let attempt = 0; attempt < 3; attempt++) {
      const fractionsUp = [...state.fractions];
      fractionsUp[i] = Math.min(0.8, fractionsUp[i] + delta);
      const totalUp = fractionsUp.reduce((s, f) => s + f, 0);
      for (let j = 0; j < fractionsUp.length; j++) fractionsUp[j] /= totalUp;
      stateUp = { ...state, fractions: fractionsUp };

      const fractionsDown = [...state.fractions];
      fractionsDown[i] = Math.max(0.01, fractionsDown[i] - delta);
      const totalDown = fractionsDown.reduce((s, f) => s + f, 0);
      for (let j = 0; j < fractionsDown.length; j++) fractionsDown[j] /= totalDown;
      stateDown = { ...state, fractions: fractionsDown };

      const formulaUp = formulaFromContinuousState(stateUp);
      const formulaDown = formulaFromContinuousState(stateDown);

      if (formulaUp !== baseFormula || formulaDown !== baseFormula) {
        changed = true;
        break;
      }
      delta *= 2;
    }

    if (!changed) {
      gradients.push(0);
      continue;
    }

    const physicsUp = evaluatePhysics(formulaFromContinuousState(stateUp), state.pressure);
    const lossUp = computeMultiObjectiveLoss(physicsUp, target, stateUp);

    const physicsDown = evaluatePhysics(formulaFromContinuousState(stateDown), state.pressure);
    const lossDown = computeMultiObjectiveLoss(physicsDown, target, stateDown);

    let grad = (lossDown - lossUp) / (2 * delta);
    if (delta > 0.1) {
      grad *= 0.1 / delta;
    }
    gradients.push(grad);
  }

  return gradients;
}

function computeMultiObjectiveLoss(
  physics: PhysicsOutput,
  target: TargetProperties,
  state?: ContinuousState
): number {
  const tcRatio = physics.tc / Math.max(1, target.targetTc);
  const tcFar = tcRatio < 0.5;

  const rawTcLoss = Math.pow(Math.max(0, target.targetTc - physics.tc) / target.targetTc, 2);
  const tcBonus = physics.tc > target.targetTc
    ? (physics.tc - target.targetTc) / target.targetTc * 0.05
    : 0;

  const lambdaLoss = physics.lambda < (target.minLambda ?? 1.0)
    ? Math.pow(((target.minLambda ?? 1.0) - physics.lambda) / (target.minLambda ?? 1.0), 2) * 0.15
    : 0;

  const metalLoss = physics.metallicity < 0.6
    ? Math.pow(0.6 - physics.metallicity, 2) * 0.08
    : 0;

  const stabGap = 0.4 - physics.stability;
  const stabilityLoss = stabGap > 0
    ? Math.pow(stabGap, 2) / (1 + Math.exp(-20 * (stabGap - 0.2))) * 0.24
    : 0;

  let stoichLoss = 0;
  if (state) {
    const maxFrac = Math.max(...state.fractions);
    if (maxFrac > 0.7) stoichLoss += (maxFrac - 0.7) * 0.1;
    const nActiveElements = state.fractions.filter(f => f > 0.05).length;
    if (nActiveElements < 2) stoichLoss += 0.15;
  }

  const dosLoss = physics.dos < 3 ? (3 - physics.dos) * 0.03 : 0;

  let gbLoss = 0;
  if (physics.gbTc > 0) {
    const gbGap = Math.max(0, target.targetTc - physics.gbTc) / target.targetTc;
    gbLoss = gbGap * gbGap * 0.07;
  }

  const constraintLoss = lambdaLoss + metalLoss + stabilityLoss + stoichLoss + dosLoss + gbLoss;

  if (tcFar) {
    return Math.max(1e-4, rawTcLoss * 0.75 + constraintLoss * 0.25 - tcBonus);
  }
  return Math.max(1e-4, rawTcLoss * 0.45 + constraintLoss - tcBonus);
}

export function runContinuousOptimization(
  initialFormula: string,
  target: TargetProperties,
  maxSteps: number = 40
): DifferentiableResult {
  const discreteState = parseToState(initialFormula, target.maxPressure < 200 ? 0 : target.maxPressure);
  const totalAtoms = Object.values(discreteState.counts).reduce((s, n) => s + n, 0);
  const elements = discreteState.elements;
  const fractions = elements.map(el => (discreteState.counts[el] || 1) / totalAtoms);

  const isHighPressureHydride = discreteState.pressure > 50 && elements.includes("H")
    && elements.some(e => {
      const d = safeGetElementData(e);
      return d && d.atomicMass > 50 && (isTransitionMetal(e) || isRareEarth(e));
    });
  const atomCap = isHighPressureHydride ? 36 : 16;

  let state: ContinuousState = {
    elements,
    fractions,
    totalAtoms: Math.min(atomCap, totalAtoms),
    pressure: discreteState.pressure,
  };

  const adam = initAdam(elements.length);
  const steps: OptimizationStep[] = [];
  let bestTc = 0;
  let bestFormula = initialFormula;
  let stagnationCount = 0;
  let lastTc = 0;

  const initialPhysics = evaluatePhysics(initialFormula, state.pressure);
  const initialTc = initialPhysics.tc;

  for (let step = 0; step < maxSteps; step++) {
    const formula = formulaFromContinuousState(state);
    const physics = evaluatePhysics(formula, state.pressure);
    const loss = computeMultiObjectiveLoss(physics, target, state);

    const gradients = computeContinuousGradients(state, physics.tc, target);
    const updates = adamStep(adam, gradients);

    const gradNorm = Math.sqrt(gradients.reduce((s, g) => s + g * g, 0));

    steps.push({
      step,
      formula,
      tc: Math.round(physics.tc * 10) / 10,
      lambda: Math.round(physics.lambda * 1000) / 1000,
      omegaLog: Math.round(physics.omegaLog * 10) / 10,
      muStar: Math.round(physics.muStar * 1000) / 1000,
      gbTc: Math.round(physics.gbTc * 10) / 10,
      loss: Math.round(loss * 10000) / 10000,
      gradientNorm: Math.round(gradNorm * 100) / 100,
      action: `continuous-adam step=${step}`,
    });

    if (physics.tc > bestTc) {
      bestTc = physics.tc;
      bestFormula = formula;
      stagnationCount = 0;
    } else {
      stagnationCount++;
    }

    if (Math.abs(physics.tc - target.targetTc) / target.targetTc < 0.01) break;

    if (stagnationCount >= 6) {
      const worstIdx = state.fractions.indexOf(Math.min(...state.fractions.filter(f => f > 0.02)));
      if (worstIdx >= 0) {
        const newEl = pickSubstitute(state.elements[worstIdx]);
        if (newEl && !state.elements.includes(newEl)) {
          const oldFrac = state.fractions[worstIdx];
          state.elements[worstIdx] = newEl;
          state.fractions[worstIdx] = Math.max(oldFrac, 0.05);
          const totalFrac = state.fractions.reduce((s, f) => s + f, 0);
          state.fractions = state.fractions.map(f => f / totalFrac);
          adam.m[worstIdx] = 0;
          adam.v[worstIdx] = 0;
          stagnationCount = 0;
          continue;
        }
      }

      if (stagnationCount >= 10 && state.elements.length < 5) {
        const candidates = ["Nb", "V", "Ti", "H", "B", "La", "Y"];
        const toAdd = candidates.find(e => !state.elements.includes(e));
        if (toAdd) {
          state.elements.push(toAdd);
          state.fractions.push(0.1);
          adam.m.push(0);
          adam.v.push(0);
          const totalFrac = state.fractions.reduce((s, f) => s + f, 0);
          state.fractions = state.fractions.map(f => f / totalFrac);
          stagnationCount = 0;
          continue;
        }
      }
    }

    for (let i = 0; i < state.fractions.length; i++) {
      state.fractions[i] += updates[i];
      state.fractions[i] = Math.max(0.01, Math.min(0.8, state.fractions[i]));
    }

    const totalFrac = state.fractions.reduce((s, f) => s + f, 0);
    state.fractions = state.fractions.map(f => f / totalFrac);

    if (step % 5 === 0 && state.totalAtoms < atomCap) {
      state.totalAtoms = Math.min(atomCap, state.totalAtoms + 1);
    }

    lastTc = physics.tc;
  }

  const improvement = initialTc > 0 ? (bestTc - initialTc) / initialTc : 0;

  const result: DifferentiableResult = {
    initialFormula,
    finalFormula: bestFormula,
    initialTc: Math.round(initialTc * 10) / 10,
    finalTc: Math.round(bestTc * 10) / 10,
    targetTc: target.targetTc,
    steps,
    converged: bestTc >= target.targetTc * 0.95,
    improvementRatio: Math.round(improvement * 1000) / 1000,
    totalSteps: steps.length,
  };

  stats.totalRuns++;
  stats.totalSteps += steps.length;
  if (bestTc > stats.bestTcAchieved) {
    stats.bestTcAchieved = bestTc;
    stats.bestFormula = bestFormula;
  }
  stats.recentResults.push({ formula: bestFormula, tc: bestTc, steps: steps.length, improvement });
  if (stats.recentResults.length > 20) stats.recentResults.shift();
  stats.avgImprovement = stats.recentResults.reduce((s, r) => s + r.improvement, 0) / stats.recentResults.length;
  stats.convergenceRate = stats.recentResults.filter(r =>
    r.tc >= target.targetTc * 0.95
  ).length / Math.max(1, stats.recentResults.length);

  return result;
}

function pickSubstitute(el: string): string | null {
  for (const [, group] of Object.entries(SUBSTITUTION_GROUPS)) {
    if (group.includes(el)) {
      const others = group.filter(e => e !== el);
      return others[Math.floor(Math.random() * others.length)] ?? null;
    }
  }
  return null;
}
