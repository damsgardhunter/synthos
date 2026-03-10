import { findBestPrecursors, computePrecursorAvailabilityScore, type PrecursorSelection } from "./precursor-database";
import { predictSynthesisFeasibility } from "./ml-synthesis-predictor";
import { classifyFamily } from "../learning/utils";

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

export interface SynthesisGateResult {
  pass: boolean;
  compositeScore: number;
  mlFeasibility: number;
  chemicalDistance: ChemicalDistanceResult;
  rejectionReasons: string[];
  classification: "trivial" | "one-pot" | "multi-step" | "complex" | "impractical";
  deprioritize: boolean;
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
  "Hydride": "high-pressure",
  "Cuprate": "solid-state",
  "Pnictide": "solid-state",
  "A15": "arc-melting",
  "Boride": "arc-melting",
  "Chalcogenide": "solid-state",
  "Oxide": "solid-state",
  "Nitride": "solid-state",
  "Carbide": "arc-melting",
  "Silicide": "arc-melting",
  "Other": "solid-state",
};

const ONE_POT_METHODS = new Set(["solid-state", "arc-melting", "ball-milling", "sputtering"]);

const HARD_GATE_THRESHOLD = 0.2;
const DEPRIORITIZE_THRESHOLD = 0.35;

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

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
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

export function evaluateSynthesisGate(formula: string): SynthesisGateResult {
  stats.totalEvaluated++;

  const mlResult = predictSynthesisFeasibility(formula);
  const chemDist = computeChemicalDistance(formula);

  const compositeScore = Math.round(
    Math.min(1.0,
      mlResult.feasibility * 0.40 +
      (1 - chemDist.totalDistance) * 0.30 +
      chemDist.precursorAvailability * 0.15 +
      (chemDist.isOnePot ? 0.15 : 0) +
      (chemDist.toxicElements.length === 0 ? 0.05 : 0) -
      (chemDist.toxicityPenalty * 0.05)
    ) * 10000
  ) / 10000;

  const rejectionReasons: string[] = [];

  if (mlResult.feasibility < HARD_GATE_THRESHOLD) {
    rejectionReasons.push(`ML feasibility too low: ${mlResult.feasibility.toFixed(3)}`);
  }

  if (chemDist.totalDistance > 0.8) {
    rejectionReasons.push(`Chemical distance too high: ${chemDist.totalDistance.toFixed(3)}`);
  }

  if (chemDist.stepEstimate >= 8) {
    rejectionReasons.push(`Too many synthesis steps: ${chemDist.stepEstimate}`);
  }

  if (chemDist.toxicityPenalty > 0.5) {
    rejectionReasons.push(`Severe toxicity hazard: ${chemDist.toxicElements.join(", ")}`);
  }

  if (chemDist.precursorAvailability < 0.2) {
    rejectionReasons.push(`Precursors unavailable: avail=${chemDist.precursorAvailability.toFixed(3)}`);
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
    rejectionReasons,
    classification,
    deprioritize,
  };
}

export function getSynthesisGateStats(): SynthesisGateStats {
  return { ...stats };
}

export { HARD_GATE_THRESHOLD, DEPRIORITIZE_THRESHOLD };
