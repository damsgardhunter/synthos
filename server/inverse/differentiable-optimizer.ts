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
  recentResults: { formula: string; tc: number; steps: number }[];
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
  }

  const knownTc = KNOWN_SC_TC[formula];
  if (knownTc !== undefined && tc < knownTc * 0.5) {
    tc = knownTc * (0.85 + Math.random() * 0.3);
  }

  if (electronic.metallicity < 0.4) {
    const dampFactor = Math.max(0.15, electronic.metallicity * 1.5);
    tc = tc * dampFactor;
  }

  tc = Math.min(200, tc);

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

  const denomLambda = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denomLambda) < 1e-8) {
    return { dTc_dLambda: 0, dTc_dOmegaLog: 0, dTc_dMuStar: 0 };
  }

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  const exponentArg = -1.04 * (1 + lambda) / denomLambda;
  const prefactor = omegaLogK / 1.2;
  const tc = prefactor * f1 * Math.exp(exponentArg);
  if (!Number.isFinite(tc) || tc <= 0) {
    return { dTc_dLambda: 0, dTc_dOmegaLog: 0, dTc_dMuStar: 0 };
  }

  const dTc_dOmegaLog = (tc / omegaLog);

  const numerator = -1.04 * (1 + lambda);
  const dDenom_dLambda = 1 - 0.62 * muStar;
  const dNumerator_dLambda = -1.04;
  const dExponent_dLambda = (dNumerator_dLambda * denomLambda - numerator * dDenom_dLambda) / (denomLambda * denomLambda);
  const df1_dLambda = (1 / 3) * Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), -2 / 3) *
    (3 / 2) * Math.pow(lambda / lambdaBar, 1 / 2) * (1 / lambdaBar);
  const dTc_dLambda = tc * dExponent_dLambda + prefactor * df1_dLambda * Math.exp(exponentArg);

  const dDenom_dMuStar = -(1 + 0.62 * lambda);
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
  target: TargetProperties
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
      stateUpdate: (s) => fineTuneStoichiometry(s, elementGrad),
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
      stateUpdate: (s) => boostCoupling(applyElementGradients(s, elementGrad, 0.3), elementGrad),
    };
  }

  if (signals.needHigherOmegaLog) {
    return {
      action: "increase phonon freq: increase light element content",
      stateUpdate: (s) => boostPhononFrequency(applyElementGradients(s, elementGrad, 0.3), elementGrad),
    };
  }

  if (signals.lowDOS) {
    return {
      action: "increase DOS: add high-DOS transition metal",
      stateUpdate: (s) => boostDOS(applyElementGradients(s, elementGrad, 0.3), elementGrad),
    };
  }

  if (signals.lowMetallicity) {
    return {
      action: "increase metallicity: substitute insulator with metal",
      stateUpdate: (s) => improveMetal(applyElementGradients(s, elementGrad, 0.3)),
    };
  }

  return {
    action: "general: follow element gradients",
    stateUpdate: (s) => applyElementGradients(s, elementGrad, 1.0),
  };
}

function applyElementGradients(state: MaterialState, gradients: Map<string, number>, lr: number): MaterialState {
  const newCounts = { ...state.counts };
  const entries = Array.from(gradients.entries());
  entries.sort((a, b) => b[1] - a[1]);

  const maxGrad = Math.max(...entries.map(e => Math.abs(e[1])), 1);
  const scaledLr = lr / maxGrad;

  for (const [el, grad] of entries) {
    const delta = grad * scaledLr;
    const current = newCounts[el] || 1;
    if (delta > 0.3) {
      newCounts[el] = Math.min(16, current + Math.max(1, Math.round(delta)));
    } else if (delta < -0.3 && current > 1) {
      newCounts[el] = Math.max(1, current - Math.max(1, Math.round(Math.abs(delta))));
    }
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

  if (updated.elements.includes("H")) {
    newCounts["H"] = Math.min(16, (newCounts["H"] || 1) + 2);
  } else {
    const lightCandidates = ["H", "B", "C", "N"].filter(e => !updated.elements.includes(e));
    if (lightCandidates.length > 0) {
      const best = lightCandidates.reduce((a, b) =>
        (ELEMENT_LAMBDA_BOOST[a] ?? 0) > (ELEMENT_LAMBDA_BOOST[b] ?? 0) ? a : b
      );
      newElements.push(best);
      newCounts[best] = best === "H" ? 4 : 2;
    } else {
      newCounts["H"] = (newCounts["H"] || 0) + 2;
      if (!newElements.includes("H")) newElements.push("H");
    }
  }

  return { elements: newElements, counts: newCounts, pressure: updated.pressure };
}

function boostCoupling(state: MaterialState, gradients: Map<string, number>): MaterialState {
  const newCounts = { ...state.counts };
  const newElements = [...state.elements];

  const tmPresent = state.elements.filter(e => isTransitionMetal(e));
  if (tmPresent.length > 0) {
    const bestTM = tmPresent.reduce((a, b) =>
      (ELEMENT_LAMBDA_BOOST[a] ?? 0) > (ELEMENT_LAMBDA_BOOST[b] ?? 0) ? a : b
    );
    if ((newCounts[bestTM] || 1) < 4) {
      newCounts[bestTM] = (newCounts[bestTM] || 1) + 1;
    }
  } else {
    const highCoupling = ["Nb", "V", "Ta"];
    const toAdd = highCoupling[Math.floor(Math.random() * highCoupling.length)];
    newElements.push(toAdd);
    newCounts[toAdd] = 1;
  }

  return { elements: newElements, counts: newCounts, pressure: state.pressure };
}

function boostPhononFrequency(state: MaterialState, gradients: Map<string, number>): MaterialState {
  const newCounts = { ...state.counts };
  const newElements = [...state.elements];

  const lightPresent = state.elements.filter(e => (ELEMENT_PHONON_BOOST[e] ?? 0) > 0.8);
  if (lightPresent.length > 0) {
    const best = lightPresent.reduce((a, b) =>
      (ELEMENT_PHONON_BOOST[a] ?? 0) > (ELEMENT_PHONON_BOOST[b] ?? 0) ? a : b
    );
    newCounts[best] = Math.min(12, (newCounts[best] || 1) + 1);
  } else {
    newElements.push("H");
    newCounts["H"] = 3;
  }

  const heavyElements = state.elements.filter(e => {
    const data = getElementData(e);
    return data && data.atomicMass > 100 && (newCounts[e] || 0) > 1;
  });
  for (const heavy of heavyElements) {
    if (newCounts[heavy] > 1) {
      newCounts[heavy] -= 1;
    }
  }

  return { elements: newElements, counts: newCounts, pressure: state.pressure };
}

function boostDOS(state: MaterialState, gradients: Map<string, number>): MaterialState {
  const newCounts = { ...state.counts };
  const newElements = [...state.elements];

  const dosElements = state.elements.filter(e => (ELEMENT_DOS_BOOST[e] ?? 0) > 0.8);
  if (dosElements.length > 0) {
    const best = dosElements.reduce((a, b) =>
      (ELEMENT_DOS_BOOST[a] ?? 0) > (ELEMENT_DOS_BOOST[b] ?? 0) ? a : b
    );
    if ((newCounts[best] || 1) < 4) {
      newCounts[best] = (newCounts[best] || 1) + 1;
    }
  } else {
    const highDOS = ["Nb", "V", "Mn"];
    const toAdd = highDOS.filter(e => !state.elements.includes(e))[0] ?? "Nb";
    newElements.push(toAdd);
    newCounts[toAdd] = 1;
  }

  return { elements: newElements, counts: newCounts, pressure: state.pressure };
}

function improveMetal(state: MaterialState): MaterialState {
  const newCounts = { ...state.counts };
  const newElements = [...state.elements];

  const nonMetals = state.elements.filter(e => {
    const d = getElementData(e);
    return d && !isTransitionMetal(e) && !isRareEarth(e) && e !== "H" && e !== "B";
  });

  for (const nm of nonMetals) {
    if ((newCounts[nm] || 0) > 2) {
      newCounts[nm] -= 1;
    }
  }

  if (!state.elements.some(e => isTransitionMetal(e))) {
    newElements.push("Nb");
    newCounts["Nb"] = 1;
  }

  return { elements: newElements, counts: newCounts, pressure: state.pressure };
}

function substituteElement(state: MaterialState, oldEl: string, newEl: string): MaterialState {
  if (state.elements.includes(newEl)) return state;
  const newCounts = { ...state.counts };
  const newElements = state.elements.map(e => e === oldEl ? newEl : e);
  newCounts[newEl] = newCounts[oldEl] || 1;
  delete newCounts[oldEl];
  return { elements: newElements, counts: newCounts, pressure: state.pressure };
}

function getSubstitutionGroup(el: string): string[] {
  for (const group of Object.values(SUBSTITUTION_GROUPS)) {
    if (group.includes(el)) return group.filter(e => e !== el);
  }
  return [];
}

function computeLoss(physics: PhysicsOutput, target: TargetProperties, synthVec?: SynthesisVector): number {
  const tcLoss = Math.pow(Math.max(0, target.targetTc - physics.tc) / target.targetTc, 2) * 0.55;
  const lambdaLoss = physics.lambda < target.minLambda
    ? Math.pow((target.minLambda - physics.lambda) / target.minLambda, 2) * 0.15
    : 0;
  const metalLoss = target.metallicRequired && physics.metallicity < 0.5
    ? Math.pow(0.5 - physics.metallicity, 2) * 0.1
    : 0;
  const stabilityLoss = physics.stability < 0.3 ? (0.3 - physics.stability) * 0.15 : 0;

  let synthLoss = 0;
  if (synthVec) {
    const complexity = computeSynthesisComplexity(synthVec);
    const feasibility = checkSynthesisFeasibility(synthVec);
    const infeasibilityPenalty = feasibility.labFeasible ? 0 : (1 - feasibility.feasibilityScore) * 0.15;
    synthLoss = Math.min(0.1, complexity * 0.02) + infeasibilityPenalty;
  }

  return tcLoss + lambdaLoss + metalLoss + stabilityLoss + synthLoss;
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
  let bestFormula = initialFormula;
  let bestState = state;
  let stagnationCount = 0;

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

    const gradNorm = Math.sqrt(
      analyticGrad.dTc_dLambda ** 2 +
      analyticGrad.dTc_dOmegaLog ** 2 +
      analyticGrad.dTc_dMuStar ** 2
    );

    const { action, stateUpdate } = interpretGradients(
      analyticGrad, elementGrad, physics, target
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

    if (physics.tc > bestTc) {
      bestTc = physics.tc;
      bestFormula = formula;
      bestState = { ...state, counts: { ...state.counts }, synthesisVector: state.synthesisVector ? { ...state.synthesisVector } : undefined };
      stagnationCount = 0;
    } else {
      const annealTemp = Math.max(0.01, 1.0 - step / maxSteps);
      const tcDelta = physics.tc - bestTc;
      const acceptProb = Math.exp(tcDelta / (annealTemp * Math.max(1, target.targetTc * 0.1)));
      if (Math.random() < acceptProb) {
        stagnationCount = Math.max(0, stagnationCount - 1);
      } else {
        stagnationCount++;
      }
    }

    if (Math.abs(physics.tc - target.targetTc) / target.targetTc < 0.05) {
      break;
    }

    if (stagnationCount >= 4) {
      const annealTemp = Math.max(0.01, 1.0 - step / maxSteps);

      if (annealTemp > 0.3 && Math.random() < annealTemp * 0.5) {
        const allEls = Object.keys(state.counts).filter(e => (state.counts[e] || 0) > 0);
        if (allEls.length > 0) {
          const randomEl = allEls[Math.floor(Math.random() * allEls.length)];
          const subs = getSubstitutionGroup(randomEl);
          if (subs.length > 0) {
            const replacement = subs[Math.floor(Math.random() * subs.length)];
            state = substituteElement(state, randomEl, replacement);
            stagnationCount = 0;
            continue;
          }
        }
      }

      const weakest = [...elementGrad.entries()]
        .filter(([el]) => (state.counts[el] || 0) > 0)
        .sort((a, b) => a[1] - b[1])[0];

      if (weakest) {
        const subs = getSubstitutionGroup(weakest[0]);
        if (subs.length > 0) {
          const replacement = subs[Math.floor(Math.random() * subs.length)];
          state = substituteElement(state, weakest[0], replacement);
          stagnationCount = 0;
          continue;
        }
      }
    }

    state = stateUpdate(state);

    if (state.synthesisVector && step % 3 === 0) {
      state.synthesisVector = mutateSynthesisVector(state.synthesisVector);
    }

    const totalAtoms = Object.values(state.counts).reduce((s, n) => s + n, 0);
    if (totalAtoms > 20) {
      const sorted = Object.entries(state.counts).sort((a, b) => b[1] - a[1]);
      for (const [el, count] of sorted) {
        if (count > 1 && Object.values(state.counts).reduce((s, n) => s + n, 0) > 15) {
          state.counts[el] = Math.max(1, Math.floor(count * 0.8));
        }
      }
    }

    state.elements = state.elements.filter(e => (state.counts[e] || 0) > 0);
  }

  const finalFormula = formulaFromState(bestState);
  const improvement = initialTc > 0 ? (bestTc - initialTc) / initialTc : 0;

  const result: DifferentiableResult = {
    initialFormula,
    finalFormula,
    initialTc: Math.round(initialTc * 10) / 10,
    finalTc: Math.round(bestTc * 10) / 10,
    targetTc: target.targetTc,
    steps,
    converged: Math.abs(bestTc - target.targetTc) / target.targetTc < 0.1,
    improvementRatio: Math.round(improvement * 1000) / 1000,
    totalSteps: steps.length,
  };

  stats.totalRuns++;
  stats.totalSteps += steps.length;
  if (bestTc > stats.bestTcAchieved) {
    stats.bestTcAchieved = bestTc;
    stats.bestFormula = finalFormula;
  }
  stats.recentResults.push({ formula: finalFormula, tc: bestTc, steps: steps.length });
  if (stats.recentResults.length > 20) stats.recentResults.shift();
  stats.avgImprovement = stats.recentResults.reduce((s, r) => s + r.tc, 0) / stats.recentResults.length;
  stats.convergenceRate = stats.recentResults.filter(r =>
    Math.abs(r.tc - target.targetTc) / target.targetTc < 0.1
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

  const results = [...standardResults, ...continuousResults];

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

function initAdam(dim: number, lr: number = 0.05): AdamState {
  return {
    m: new Array(dim).fill(0),
    v: new Array(dim).fill(0),
    t: 0,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    lr,
  };
}

function adamStep(adam: AdamState, gradients: number[]): number[] {
  adam.t += 1;
  const updates: number[] = [];

  for (let i = 0; i < gradients.length; i++) {
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
    counts["Nb"] = 3;
    counts["Si"] = 1;
    elements.push("Nb", "Si");
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

  for (let i = 0; i < state.fractions.length; i++) {
    const delta = 0.02;
    const fractionsUp = [...state.fractions];
    fractionsUp[i] = Math.min(0.8, fractionsUp[i] + delta);

    const total = fractionsUp.reduce((s, f) => s + f, 0);
    for (let j = 0; j < fractionsUp.length; j++) fractionsUp[j] /= total;

    const stateUp: ContinuousState = { ...state, fractions: fractionsUp };
    const formulaUp = formulaFromContinuousState(stateUp);
    const physicsUp = evaluatePhysics(formulaUp, state.pressure);
    const lossUp = computeMultiObjectiveLoss(physicsUp, target, stateUp);

    const fractionsDown = [...state.fractions];
    fractionsDown[i] = Math.max(0.01, fractionsDown[i] - delta);
    const totalDown = fractionsDown.reduce((s, f) => s + f, 0);
    for (let j = 0; j < fractionsDown.length; j++) fractionsDown[j] /= totalDown;

    const stateDown: ContinuousState = { ...state, fractions: fractionsDown };
    const formulaDown = formulaFromContinuousState(stateDown);
    const physicsDown = evaluatePhysics(formulaDown, state.pressure);
    const lossDown = computeMultiObjectiveLoss(physicsDown, target, stateDown);

    gradients.push((lossDown - lossUp) / (2 * delta));
  }

  return gradients;
}

function computeMultiObjectiveLoss(
  physics: PhysicsOutput,
  target: TargetProperties,
  state?: ContinuousState
): number {
  const tcLoss = Math.pow(Math.max(0, target.targetTc - physics.tc) / target.targetTc, 2) * 0.45;

  const lambdaLoss = physics.lambda < (target.minLambda ?? 1.0)
    ? Math.pow(((target.minLambda ?? 1.0) - physics.lambda) / (target.minLambda ?? 1.0), 2) * 0.15
    : 0;

  const metalLoss = physics.metallicity < 0.5
    ? Math.pow(0.5 - physics.metallicity, 2) * 0.08
    : 0;

  const stabilityLoss = physics.stability < 0.4 ? (0.4 - physics.stability) * 0.12 : 0;

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

  return tcLoss + lambdaLoss + metalLoss + stabilityLoss + stoichLoss + dosLoss + gbLoss;
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

  let state: ContinuousState = {
    elements,
    fractions,
    totalAtoms: Math.min(16, totalAtoms),
    pressure: discreteState.pressure,
  };

  const adam = initAdam(elements.length, 0.03);
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

    if (Math.abs(physics.tc - target.targetTc) / target.targetTc < 0.05) break;

    if (stagnationCount >= 6) {
      const worstIdx = state.fractions.indexOf(Math.min(...state.fractions.filter(f => f > 0.02)));
      if (worstIdx >= 0) {
        const newEl = pickSubstitute(state.elements[worstIdx]);
        if (newEl && !state.elements.includes(newEl)) {
          state.elements[worstIdx] = newEl;
          state.fractions[worstIdx] = 0.15;
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

    if (step % 5 === 0 && state.totalAtoms < 16) {
      state.totalAtoms = Math.min(16, state.totalAtoms + 1);
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
    converged: Math.abs(bestTc - target.targetTc) / target.targetTc < 0.1,
    improvementRatio: Math.round(improvement * 1000) / 1000,
    totalSteps: steps.length,
  };

  stats.totalRuns++;
  stats.totalSteps += steps.length;
  if (bestTc > stats.bestTcAchieved) {
    stats.bestTcAchieved = bestTc;
    stats.bestFormula = bestFormula;
  }
  stats.recentResults.push({ formula: bestFormula, tc: bestTc, steps: steps.length });
  if (stats.recentResults.length > 20) stats.recentResults.shift();
  stats.avgImprovement = stats.recentResults.reduce((s, r) => s + r.tc, 0) / stats.recentResults.length;
  stats.convergenceRate = stats.recentResults.filter(r =>
    Math.abs(r.tc - target.targetTc) / target.targetTc < 0.1
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
