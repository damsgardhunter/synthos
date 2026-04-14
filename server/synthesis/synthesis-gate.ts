import { findBestPrecursors, computePrecursorAvailabilityScore, type PrecursorSelection } from "./precursor-database";
import { predictSynthesisFeasibility } from "./ml-synthesis-predictor";
import { classifyFamily } from "../learning/utils";
import { buildReactionNetwork } from "./reaction-network";

export interface ChemicalDistanceResult {
  totalDistance: number;
  stepEstimate: number;
  precursorAvailability: number;
  bottleneckElement: string | null;
  costTier: string;
  toxicityPenalty: number;
  toxicElements: string[];
  preferredMethod: string;
  isOnePot: boolean;
}

export interface PhysicsInput {
  hullDistanceEv?: number;
  requiredPressureGpa?: number;
}

export interface SynthesisGateResult {
  pass: boolean;
  compositeScore: number;
  mlFeasibility: number;
  chemicalDistance: ChemicalDistanceResult;
  graphPathCost: number | null;
  dijkstraBottleneck: string | null;
  dijkstraMethod: string | null;
  dijkstraStepCount: number | null;
  rejectionReasons: string[];
  pressureFlag: string | null;
  classification: "trivial" | "one-pot" | "multi-step" | "complex" | "impractical";
  deprioritize: boolean;
}

// Hard maximum hull distance above convex hull — ambient-pressure baseline.
// High-pressure synthesis uses tiered thresholds (see getHullDistanceMax).
// Materials Project considers <0.1 eV/atom stable, 0.1-0.2 eV/atom metastable-synthesizable;
// many known superconductors sit at 0.15-0.3 eV/atom. 0.5 eV/atom is the ambient ceiling.
export const MAX_HULL_DISTANCE_EV = 0.5;

// Pressure above which a DAC (diamond-anvil cell) synthesis path is required.
const HIGH_PRESSURE_THRESHOLD_GPA = 50;

/**
 * Returns the hull-distance ceiling (eV/atom) appropriate for a compound's
 * synthesis pressure. Ambient-pressure compounds use a tighter bound because
 * metastability must be kinetically trapped without external pressure. High-
 * pressure synthesis stabilizes phases that are "above hull" at 0 GPa — the
 * hull distance computed at the synthesis pressure is what matters, so the
 * threshold scales with the confidence we have in the ambient-pressure estimate.
 */
function getHullDistanceMax(pressureGpa: number): number {
  if (pressureGpa > 100) return 2.0;  // DAC: ambient hull is unreliable for these phases
  if (pressureGpa > 20) return 0.8;   // Moderate pressure: generous allowance
  return MAX_HULL_DISTANCE_EV;         // Ambient: 0.5 eV/atom
}

interface SynthesisGateStats {
  totalEvaluated: number;
  totalRejected: number;
  totalPassed: number;
  rejectionRate: number;
  rejectionsByReason: Record<string, number>;
  classificationCounts: Record<string, number>;
  avgCompositeScore: number;
  recentRejections: Array<{ formula: string; score: number; reasons: string[]; at: number }>;
}

const TOXIC_ELEMENTS: Record<string, number> = {
  "As": 0.35,
  "Cd": 0.30,
  "Hg": 0.40,
  "Pb": 0.20,
  "Tl": 0.35,
  "Be": 0.30,
  "Cr": 0.15,
  "Se": 0.10,
  "Te": 0.08,
  "Sb": 0.12,
  "U": 0.40,
  "Th": 0.35,
  "Pu": 0.50,
  "F": 0.15,
  "Os": 0.20,
};

const PREFERRED_METHODS_BY_FAMILY: Record<string, string> = {
  // Pluralized names as returned by classifyFamily() in utils.ts
  "Hydrides": "high-pressure",
  "Cuprates": "solid-state",
  "Pnictides": "solid-state",
  "Borides": "arc-melting",
  "Chalcogenides": "solid-state",
  "Oxides": "solid-state",
  "Nitrides": "solid-state",
  "Carbides": "arc-melting",
  "Silicides": "arc-melting",
  "Sulfides": "solid-state",
  "Phosphides": "solid-state",
  "Alloys": "arc-melting",
  "Intermetallics": "arc-melting",
  "Nickelates": "solid-state",
  "Borocarbides": "arc-melting",
  "Clathrates": "solid-state",
  "Kagome": "solid-state",
  "Layered-chalcogenide": "solid-state",
  "Layered-pnictide": "solid-state",
  "Intercalated-layered": "solid-state",
  "Heavy Fermions": "solid-state",
  "Mixed-mechanism": "solid-state",
  // Fallback — also keep legacy singular spellings for any callers that still use them
  "Hydride": "high-pressure",
  "Other": "solid-state",
};

const ONE_POT_METHODS = new Set(["solid-state", "arc-melting", "ball-milling", "sputtering"]);
const NOBLE_GASES = new Set(["He", "Ne", "Ar", "Kr", "Xe", "Rn"]);

const HARD_GATE_THRESHOLD = 0.45;
const DEPRIORITIZE_THRESHOLD = 0.60;

const stats: SynthesisGateStats = {
  totalEvaluated: 0,
  totalRejected: 0,
  totalPassed: 0,
  rejectionRate: 0,
  rejectionsByReason: {},
  classificationCounts: {},
  avgCompositeScore: 0,
  recentRejections: [],
};

let scoreSum = 0;

function expandParentheses(formula: string): string {
  let result = formula.replace(/\[/g, "(").replace(/\]/g, ")");
  const parenRegex = /\(([^()]+)\)(\d*\.?\d*)/;
  let iterations = 0;
  while (result.includes("(") && iterations < 20) {
    const prev = result;
    result = result.replace(parenRegex, (_, group: string, mult: string) => {
      const m = mult ? parseFloat(mult) : 1;
      if (isNaN(m) || m <= 0) return group;
      if (m === 1) return group;
      return group.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_m: string, el: string, num: string) => {
        const n = num ? parseFloat(num) : 1;
        const newN = (isNaN(n) || n <= 0 ? 1 : n) * m;
        return newN === 1 ? el : `${el}${newN}`;
      });
    });
    if (result === prev) break;
    iterations++;
  }
  return result.replace(/[()]/g, "");
}

function normalizeFormulaString(formula: string): string {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const subscriptMap: Record<string, string> = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
  };
  const superscriptMap: Record<string, string> = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
  };
  let cleaned = formula;
  for (const [sub, digit] of Object.entries(subscriptMap)) {
    cleaned = cleaned.split(sub).join(digit);
  }
  for (const [sup, digit] of Object.entries(superscriptMap)) {
    cleaned = cleaned.split(sup).join(digit);
  }
  cleaned = cleaned.replace(/[^\x20-\x7E]/g, "");
  cleaned = expandParentheses(cleaned.trim());
  return cleaned;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = normalizeFormulaString(formula);
  return parseNestedFormula(cleaned);
}

function parseNestedFormula(s: string): Record<string, number> {
  const counts: Record<string, number> = {};
  let i = 0;
  while (i < s.length) {
    if (s[i] === '(') {
      let depth = 1;
      let j = i + 1;
      while (j < s.length && depth > 0) {
        if (s[j] === '(') depth++;
        else if (s[j] === ')') depth--;
        j++;
      }
      const inner = parseNestedFormula(s.substring(i + 1, j - 1));
      let numStr = '';
      while (j < s.length && (s[j] >= '0' && s[j] <= '9' || s[j] === '.')) {
        numStr += s[j]; j++;
      }
      const mult = numStr ? parseFloat(numStr) : 1;
      for (const [el, cnt] of Object.entries(inner)) {
        counts[el] = (counts[el] || 0) + cnt * mult;
      }
      i = j;
    } else if (s[i] >= 'A' && s[i] <= 'Z') {
      let el = s[i]; i++;
      while (i < s.length && s[i] >= 'a' && s[i] <= 'z') { el += s[i]; i++; }
      let numStr = '';
      while (i < s.length && (s[i] >= '0' && s[i] <= '9' || s[i] === '.')) { numStr += s[i]; i++; }
      const num = numStr ? parseFloat(numStr) : 1;
      if (isNaN(num) || num <= 0) {
        counts[el] = (counts[el] || 0) + 1;
      } else {
        counts[el] = (counts[el] || 0) + num;
      }
    } else { i++; }
  }
  return counts;
}

function computeChemicalDistance(formula: string): ChemicalDistanceResult {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);

  const family = classifyFamily(formula);
  const preferredMethod = PREFERRED_METHODS_BY_FAMILY[family] || "solid-state";

  const selections = findBestPrecursors(elements, preferredMethod);
  const availResult = computePrecursorAvailabilityScore(selections);

  let toxicityPenalty = 0;
  const toxicElements: string[] = [];
  for (const el of elements) {
    if (TOXIC_ELEMENTS[el]) {
      const frac = (counts[el] || 0) / totalAtoms;
      const penalty = TOXIC_ELEMENTS[el] * (0.5 + 0.5 * frac);
      toxicityPenalty += penalty;
      toxicElements.push(el);
    }
  }
  toxicityPenalty = Math.min(1.0, toxicityPenalty);

  let stepEstimate = 1;
  if (elements.length <= 2) {
    stepEstimate = 1;
  } else if (elements.length <= 3) {
    stepEstimate = 2;
  } else if (elements.length <= 4) {
    stepEstimate = 3;
  } else {
    stepEstimate = Math.min(10, elements.length);
  }

  const hFrac = (counts["H"] || 0) / totalAtoms;
  if (hFrac > 0.5) {
    stepEstimate += 2;
  }

  const hasSafetyHazard = selections.some(s =>
    s.precursor.safetyNotes.toLowerCase().includes("extremely toxic") ||
    s.precursor.safetyNotes.toLowerCase().includes("carcinogenic") ||
    s.precursor.safetyNotes.toLowerCase().includes("radioactive")
  );
  if (hasSafetyHazard) {
    stepEstimate += 1;
  }

  if (preferredMethod === "high-pressure") {
    stepEstimate += 2;
  }

  const isOnePot = ONE_POT_METHODS.has(preferredMethod) && stepEstimate <= 2 && toxicElements.length === 0;

  const availPenalty = 1 - availResult.overallScore;
  const stepPenalty = Math.min(1.0, (stepEstimate - 1) / 9);
  const costPenalty = availResult.costEstimate === "very-high" ? 0.3
    : availResult.costEstimate === "high" ? 0.2
    : availResult.costEstimate === "medium" ? 0.1
    : 0;

  const totalDistance = Math.min(1.0,
    availPenalty * 0.35 +
    stepPenalty * 0.25 +
    toxicityPenalty * 0.25 +
    costPenalty * 0.15
  );

  return {
    totalDistance: Math.round(totalDistance * 10000) / 10000,
    stepEstimate,
    precursorAvailability: availResult.overallScore,
    bottleneckElement: availResult.bottleneckElement,
    costTier: availResult.costEstimate,
    toxicityPenalty: Math.round(toxicityPenalty * 10000) / 10000,
    toxicElements,
    preferredMethod,
    isOnePot,
  };
}

function classifySynthesisComplexity(
  compositeScore: number,
  chemDist: ChemicalDistanceResult
): "trivial" | "one-pot" | "multi-step" | "complex" | "impractical" {
  if (compositeScore >= 0.8 && chemDist.isOnePot) return "trivial";
  if (compositeScore >= 0.6 && chemDist.stepEstimate <= 2) return "one-pot";
  if (compositeScore >= 0.35 && chemDist.stepEstimate <= 5) return "multi-step";
  if (compositeScore >= HARD_GATE_THRESHOLD) return "complex";
  return "impractical";
}

function recordRejection(formula: string, score: number, reasons: string[]) {
  for (const r of reasons) {
    stats.rejectionsByReason[r] = (stats.rejectionsByReason[r] || 0) + 1;
  }
  stats.recentRejections.push({ formula, score, reasons, at: Date.now() });
  if (stats.recentRejections.length > 50) {
    stats.recentRejections = stats.recentRejections.slice(-50);
  }
}

export interface KineticInput {
  kineticScore: number;
  metastableLifetime300K: number;
  lifetimeString: string;
  stabilizationStrategies: Array<{ name: string }>;
}

export interface NoveltyInput {
  uncertaintyEstimate: number;
  gnnUncertainty?: number;
}

function computeNoveltyBonus(input?: NoveltyInput | null): number {
  if (!input) return 0;
  const unc = Math.max(input.uncertaintyEstimate, input.gnnUncertainty ?? 0);
  if (unc < 0.4) return 0;
  if (unc >= 0.85) return 0.12;
  return Math.round((unc - 0.4) * 0.2667 * 10000) / 10000;
}

export function evaluateSynthesisGate(
  formula: string,
  kineticInput?: KineticInput | null,
  noveltyInput?: NoveltyInput | null,
  physicsInput?: PhysicsInput | null,
): SynthesisGateResult {
  stats.totalEvaluated++;

  const mlResult = predictSynthesisFeasibility(formula);
  const chemDist = computeChemicalDistance(formula);

  let graphPathCost: number | null = null;
  let dijkstraBottleneck: string | null = null;
  let dijkstraMethod: string | null = null;
  let dijkstraStepCount: number | null = null;
  try {
    const network = buildReactionNetwork(formula);
    if (network.bestRoute) {
      graphPathCost = network.graphPathCost;
      dijkstraBottleneck = network.bestRoute.bottleneck;
      dijkstraMethod = network.bestRoute.method;
      dijkstraStepCount = network.bestRoute.stepCount;
    }
  } catch (_) {
  }

  const graphBonus = graphPathCost !== null
    ? Math.max(0, 0.1 * (1 - Math.min(1, graphPathCost / 2.0)))
    : 0;

  let kineticPenalty = 0;
  if (kineticInput) {
    if (kineticInput.kineticScore < 0.3) {
      kineticPenalty = (0.3 - kineticInput.kineticScore) * 0.5;
      if (kineticInput.stabilizationStrategies.length > 0) {
        kineticPenalty *= 0.6;
      }
    }
  }

  const noveltyBonus = computeNoveltyBonus(noveltyInput);

  const compositeScore = Math.round(
    Math.min(1.0, Math.max(0,
      mlResult.feasibility * 0.35 +
      (1 - chemDist.totalDistance) * 0.25 +
      chemDist.precursorAvailability * 0.15 +
      (chemDist.isOnePot ? 0.10 : 0) +
      (chemDist.toxicElements.length === 0 ? 0.05 : 0) -
      (chemDist.toxicityPenalty * 0.05) +
      graphBonus -
      kineticPenalty +
      noveltyBonus
    )) * 10000
  ) / 10000;

  const chemDistThreshold = 0.5 + noveltyBonus * 0.3;
  const stepThreshold = 6 + (noveltyBonus > 0.05 ? 1 : 0);

  const rejectionReasons: string[] = [];

  // ── Chemistry hard gates (catch obviously nonsensical formulas) ─────────────
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalFormulaAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  const nobleInFormula = elements.filter(el => NOBLE_GASES.has(el));
  if (nobleInFormula.length > 0) {
    rejectionReasons.push(`Noble gas elements not chemically bondable: ${nobleInFormula.join(", ")}`);
  }

  if (elements.length > 6) {
    rejectionReasons.push(`Too many distinct elements: ${elements.length} > 6 (synthesis impractical)`);
  }

  const maxCount = Math.max(...Object.values(counts));
  if (maxCount > 15) {
    rejectionReasons.push(`Extreme stoichiometry: single element count ${maxCount} > 15`);
  }

  if (totalFormulaAtoms > 30) {
    rejectionReasons.push(`Formula unit too large: ${totalFormulaAtoms} atoms > 30`);
  }
  // ────────────────────────────────────────────────────────────────────────────

  // ── Physics-based hard gates ────────────────────────────────────────────────
  // Hull distance cutoff: materials more than the pressure-appropriate ceiling above
  // the convex hull are too thermodynamically unstable for automated route planning.
  // The ceiling scales with synthesis pressure — high-pressure phases are computed
  // at ambient conditions so their hull distance is artificially inflated relative
  // to what they would show at the target pressure.
  //
  // EXCEPTION: values > 5 eV/atom are physically impossible at any real synthesis
  // condition and are almost certainly ambient-pressure Miedema estimates for
  // compounds (e.g. polyhydrides) that are only stable under high pressure.
  // Applying the gate to these would reject every novel high-pressure hydride
  // candidate before DFT validates them at the correct pressure.
  if (physicsInput?.hullDistanceEv !== undefined &&
      physicsInput.hullDistanceEv <= 5.0) {
    const hullMax = getHullDistanceMax(physicsInput.requiredPressureGpa ?? 0);
    if (physicsInput.hullDistanceEv > hullMax) {
      rejectionReasons.push(
        `Hull distance too high: ${physicsInput.hullDistanceEv.toFixed(3)} eV/atom > ${hullMax.toFixed(1)} eV/atom (at ${(physicsInput.requiredPressureGpa ?? 0).toFixed(0)} GPa)`,
      );
    }
  }

  // Pressure mismatch: if the physics engine requires high-pressure synthesis (DAC)
  // but the family lookup gave an ambient-pressure method, reclassify to "high-pressure"
  // rather than rejecting. Rejecting would discard every polyhydride whose family
  // wasn't matched correctly. The pressureFlag records the DAC requirement for
  // downstream reporting; the compound continues through the gate with updated scoring.
  let pressureFlag: string | null = null;
  if (physicsInput?.requiredPressureGpa !== undefined && physicsInput.requiredPressureGpa > HIGH_PRESSURE_THRESHOLD_GPA) {
    if (chemDist.preferredMethod !== "high-pressure") {
      // Silently reclassify — no rejection reason added.
      // chemDist.preferredMethod is read-only here; downstream scoring already uses
      // the pressureFlag to annotate the result and add the DAC step penalty.
      chemDist.preferredMethod = "high-pressure";
      chemDist.stepEstimate += 2;  // mirror the +2 DAC penalty from computeChemicalDistance
      chemDist.isOnePot = false;
    }
    pressureFlag = `High-Pressure Path (DAC): ${physicsInput.requiredPressureGpa.toFixed(0)} GPa required`;
  }
  // ────────────────────────────────────────────────────────────────────────────

  if (mlResult.feasibility < HARD_GATE_THRESHOLD) {
    rejectionReasons.push(`ML feasibility too low: ${mlResult.feasibility.toFixed(3)}`);
  }

  if (chemDist.totalDistance > chemDistThreshold) {
    rejectionReasons.push(`Chemical distance too high: ${chemDist.totalDistance.toFixed(3)} > ${chemDistThreshold.toFixed(2)}`);
  }

  if (chemDist.stepEstimate >= stepThreshold) {
    rejectionReasons.push(`Too many synthesis steps: ${chemDist.stepEstimate}`);
  }

  if (chemDist.toxicityPenalty > 0.5) {
    rejectionReasons.push(`Severe toxicity hazard: ${chemDist.toxicElements.join(", ")}`);
  }

  if (chemDist.precursorAvailability < 0.40) {
    rejectionReasons.push(`Precursors unavailable: avail=${chemDist.precursorAvailability.toFixed(3)}`);
  }

  if (kineticInput) {
    // Hard reject: sub-microsecond lifetime means the material decomposes faster
    // than any measurement or stabilization could act. No stabilization strategy
    // can rescue a material that lives < 1 µs at 300K.
    if (kineticInput.metastableLifetime300K < 1e-6) {
      rejectionReasons.push(`Kinetically unstable: lifetime=${kineticInput.lifetimeString} (< 1 µs) — too short-lived for any stabilization`);
    } else if (kineticInput.kineticScore < 0.15 && kineticInput.stabilizationStrategies.length === 0) {
      rejectionReasons.push(`Kinetically unstable: score=${kineticInput.kineticScore}, lifetime=${kineticInput.lifetimeString}, no stabilization routes`);
    }
  }

  const pass = compositeScore >= HARD_GATE_THRESHOLD && rejectionReasons.length === 0;
  const deprioritize = compositeScore < DEPRIORITIZE_THRESHOLD && pass;
  const classification = classifySynthesisComplexity(compositeScore, chemDist);

  scoreSum += compositeScore;
  stats.avgCompositeScore = Math.round((scoreSum / stats.totalEvaluated) * 10000) / 10000;
  stats.classificationCounts[classification] = (stats.classificationCounts[classification] || 0) + 1;

  if (pass) {
    stats.totalPassed++;
  } else {
    stats.totalRejected++;
    recordRejection(formula, compositeScore, rejectionReasons);
  }
  stats.rejectionRate = stats.totalEvaluated > 0
    ? Math.round((stats.totalRejected / stats.totalEvaluated) * 10000) / 10000
    : 0;

  return {
    pass,
    compositeScore,
    mlFeasibility: mlResult.feasibility,
    chemicalDistance: chemDist,
    graphPathCost,
    dijkstraBottleneck,
    dijkstraMethod,
    dijkstraStepCount,
    rejectionReasons,
    pressureFlag,
    classification,
    deprioritize,
  };
}

export function getSynthesisGateStats(): SynthesisGateStats {
  return { ...stats };
}

export { HARD_GATE_THRESHOLD, DEPRIORITIZE_THRESHOLD };
