import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  parseFormulaElements,
} from "../learning/physics-engine";
import {
  ELEMENTAL_DATA,
  getElementData,
  isTransitionMetal,
  isRareEarth,
} from "../learning/elemental-data";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import { checkPhysicsConstraints } from "./physics-constraint-engine";

export interface SCPillarTargets {
  minLambda: number;
  minOmegaLogK: number;
  minDOS: number;
  minNesting: number;
  minFlatBand: number;
  preferredMotifs: ("layered" | "cage" | "kagome" | "A15" | "perovskite")[];
}

export const DEFAULT_PILLAR_TARGETS: SCPillarTargets = {
  minLambda: 1.5,
  minOmegaLogK: 700,
  minDOS: 2.0,
  minNesting: 0.5,
  minFlatBand: 0.5,
  preferredMotifs: ["cage", "layered", "kagome"],
};

export interface PillarEvaluation {
  formula: string;
  lambda: number;
  omegaLogK: number;
  dos: number;
  nestingScore: number;
  flatBandScore: number;
  motifMatch: string;
  motifScore: number;
  pillarScores: {
    coupling: number;
    phonon: number;
    dos: number;
    nesting: number;
    structure: number;
  };
  compositeFitness: number;
  satisfiedPillars: number;
  weakestPillar: string;
  tcPredicted: number;
  metallicity: number;
  muStar: number;
  physicsValid: boolean;
}

export interface PillarGuidedCandidate {
  formula: string;
  evaluation: PillarEvaluation;
  designRationale: string;
}

export interface PillarOptimizerStats {
  totalEvaluated: number;
  totalGenerated: number;
  avgCompositeFitness: number;
  bestFitness: number;
  bestFormula: string;
  pillarSatisfactionRates: Record<string, number>;
  elementAffinityScores: Record<string, number>;
  topCandidates: { formula: string; fitness: number; tc: number; pillars: number }[];
  pillarWeights: Record<string, number>;
}

const LIGHT_ATOMS = ["H", "B", "C", "N"];
const HIGH_COUPLING_TM = ["Nb", "V", "Ta", "Mo", "Ti", "Zr", "Hf", "W"];
const CAGE_FORMERS = ["La", "Y", "Ca", "Sr", "Ba", "Sc", "Ce", "Th"];
const LAYER_FORMERS = ["Cu", "Fe", "Ni", "Co", "Mn"];
const KAGOME_ELEMENTS = ["V", "Mn", "Fe", "Co", "Nb"];
const PNICTOGEN_ELEMENTS = ["As", "P", "Sb", "Bi"];
const CHALCOGEN_ELEMENTS = ["S", "Se", "Te"];

function parseCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    const el = m[1];
    const num = m[2] ? parseFloat(m[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function countsToFormula(counts: Record<string, number>): string {
  const metals: string[] = [];
  const nonmetals: string[] = [];
  for (const el of Object.keys(counts)) {
    if (counts[el] <= 0) continue;
    const ed = ELEMENTAL_DATA[el];
    const en = ed?.paulingElectronegativity ?? 2.0;
    if (en <= 2.0 || isTransitionMetal(el) || isRareEarth(el)) metals.push(el);
    else nonmetals.push(el);
  }
  metals.sort((a, b) => a.localeCompare(b));
  nonmetals.sort((a, b) => a.localeCompare(b));
  return [...metals, ...nonmetals].map(el => {
    const n = counts[el];
    return n === 1 ? el : `${el}${Math.round(n)}`;
  }).join("");
}

function detectMotif(formula: string, elements: string[], counts: Record<string, number>): { match: string; score: number } {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const hCount = counts["H"] || 0;
  const hRatio = hCount / Math.max(1, totalAtoms - hCount);

  if (hRatio >= 4) return { match: "cage-clathrate", score: 0.95 };
  if (hRatio >= 2) return { match: "layered-hydride", score: 0.85 };

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  if (hasCu && hasO && elements.length >= 3) return { match: "layered-cuprate", score: 0.90 };

  const hasFeOrCo = elements.includes("Fe") || elements.includes("Co");
  const hasPnictogen = elements.some(e => PNICTOGEN_ELEMENTS.includes(e));
  const hasChalcogen = elements.some(e => CHALCOGEN_ELEMENTS.includes(e));
  if (hasFeOrCo && (hasPnictogen || hasChalcogen)) return { match: "layered-pnictide", score: 0.85 };

  const hasKagomeEl = elements.some(e => KAGOME_ELEMENTS.includes(e));
  const hasSb = elements.includes("Sb");
  if (hasKagomeEl && hasSb && elements.length >= 3) return { match: "kagome", score: 0.80 };

  if (elements.length === 2) {
    const hasTM = elements.some(e => isTransitionMetal(e));
    const hasLightAtom = elements.some(e => LIGHT_ATOMS.includes(e));
    if (hasTM && hasLightAtom) {
      const tmEl = elements.find(e => isTransitionMetal(e))!;
      const tmCount = counts[tmEl] || 0;
      if (tmCount === 3) return { match: "A15", score: 0.88 };
    }
  }

  const hasRE = elements.some(e => isRareEarth(e));
  if (hasRE && hasO) return { match: "perovskite", score: 0.70 };

  const hasTM = elements.some(e => isTransitionMetal(e));
  const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
  if (hasTM && hasLight) return { match: "metal-light-bond", score: 0.60 };

  return { match: "generic", score: 0.30 };
}

function scorePillar(value: number, target: number, softness: number = 0.5): number {
  if (value >= target) return Math.min(1.0, 0.8 + 0.2 * (value / target));
  const ratio = value / Math.max(target, 0.001);
  return Math.pow(ratio, softness);
}

let pillarWeights = {
  coupling: 0.30,
  phonon: 0.20,
  dos: 0.20,
  nesting: 0.15,
  structure: 0.15,
};

let totalEvaluated = 0;
let totalGenerated = 0;
let fitnessSum = 0;
let bestFitness = 0;
let bestFormula = "";
let bestTc = 0;
let pillarSatisfied: Record<string, number> = {
  coupling: 0, phonon: 0, dos: 0, nesting: 0, structure: 0,
};
let elementAffinity: Record<string, { totalFitness: number; count: number }> = {};
let topCandidates: { formula: string; fitness: number; tc: number; pillars: number }[] = [];

export function evaluatePillars(
  formula: string,
  targets: SCPillarTargets = DEFAULT_PILLAR_TARGETS
): PillarEvaluation {
  totalEvaluated++;

  const elements = parseFormulaElements(formula);
  const counts = parseCounts(formula);

  const electronic = computeElectronicStructure(formula, null);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

  const lambda = coupling.lambda;
  const omegaLogK = coupling.omegaLog * 1.44;
  const dos = electronic.densityOfStatesAtFermi;
  const nestingScore = electronic.nestingScore ?? 0;
  const flatBandScore = electronic.flatBandIndicator ?? 0;
  const muStar = coupling.muStar;
  const metallicity = electronic.metallicity;

  const motif = detectMotif(formula, elements, counts);

  const motifBonus = targets.preferredMotifs.some(pm => motif.match.includes(pm)) ? 0.2 : 0;

  const pillarScores = {
    coupling: scorePillar(lambda, targets.minLambda, 0.6),
    phonon: scorePillar(omegaLogK, targets.minOmegaLogK, 0.4),
    dos: scorePillar(dos, targets.minDOS, 0.5),
    nesting: scorePillar(nestingScore, targets.minNesting, 0.3),
    structure: Math.min(1.0, motif.score + motifBonus + scorePillar(flatBandScore, targets.minFlatBand, 0.3) * 0.3),
  };

  let compositeFitness =
    pillarWeights.coupling * pillarScores.coupling +
    pillarWeights.phonon * pillarScores.phonon +
    pillarWeights.dos * pillarScores.dos +
    pillarWeights.nesting * pillarScores.nesting +
    pillarWeights.structure * pillarScores.structure;

  if (metallicity < 0.3) {
    compositeFitness *= 0.5;
  }

  const satisfiedPillars =
    (lambda >= targets.minLambda ? 1 : 0) +
    (omegaLogK >= targets.minOmegaLogK ? 1 : 0) +
    (dos >= targets.minDOS ? 1 : 0) +
    (nestingScore >= targets.minNesting ? 1 : 0) +
    (flatBandScore >= targets.minFlatBand || motif.score >= 0.7 ? 1 : 0);

  const weakestPillar = (Object.entries(pillarScores) as [string, number][])
    .reduce((a, b) => a[1] < b[1] ? a : b)[0];

  let tcPredicted = 0;
  try {
    const features = extractFeatures(formula);
    const gb = gbPredict(features);
    tcPredicted = gb.tcPredicted;
  } catch {}

  const constraint = checkPhysicsConstraints(formula);

  if (!constraint.isValid && constraint.totalPenalty > 1.0) {
    compositeFitness *= 0.5;
  }

  fitnessSum += compositeFitness;

  for (const [key, score] of Object.entries(pillarScores)) {
    if (score >= 0.7) {
      pillarSatisfied[key] = (pillarSatisfied[key] || 0) + 1;
    }
  }

  for (const el of elements) {
    if (!elementAffinity[el]) elementAffinity[el] = { totalFitness: 0, count: 0 };
    elementAffinity[el].totalFitness += compositeFitness;
    elementAffinity[el].count++;
  }

  if (compositeFitness > bestFitness) {
    bestFitness = compositeFitness;
    bestFormula = formula;
    bestTc = tcPredicted;
  }

  if (compositeFitness > 0.5) {
    topCandidates.push({ formula, fitness: compositeFitness, tc: tcPredicted, pillars: satisfiedPillars });
    topCandidates.sort((a, b) => b.fitness - a.fitness);
    if (topCandidates.length > 20) topCandidates = topCandidates.slice(0, 20);
  }

  return {
    formula,
    lambda,
    omegaLogK,
    dos,
    nestingScore,
    flatBandScore,
    motifMatch: motif.match,
    motifScore: motif.score,
    pillarScores,
    compositeFitness,
    satisfiedPillars,
    weakestPillar,
    tcPredicted,
    metallicity,
    muStar,
    physicsValid: constraint.isValid,
  };
}

interface DesignTemplate {
  name: string;
  targetMotif: string;
  baseElements: string[][];
  lightAtom: string;
  stoichiometryRange: [number, number];
  lightAtomRange: [number, number];
}

const DESIGN_TEMPLATES: DesignTemplate[] = [
  {
    name: "clathrate-hydride",
    targetMotif: "cage",
    baseElements: [
      ["La", "H"], ["Y", "H"], ["Ca", "H"], ["Sr", "H"], ["Ba", "H"],
      ["Sc", "H"], ["Ce", "H"], ["Th", "H"], ["Zr", "H"], ["Ti", "H"],
    ],
    lightAtom: "H",
    stoichiometryRange: [1, 2],
    lightAtomRange: [6, 10],
  },
  {
    name: "metal-boride",
    targetMotif: "layered",
    baseElements: [
      ["Nb", "B"], ["Mo", "B"], ["Ti", "B"], ["V", "B"], ["Zr", "B"],
      ["Ta", "B"], ["W", "B"], ["Hf", "B"], ["Mg", "B"], ["Al", "B"],
    ],
    lightAtom: "B",
    stoichiometryRange: [1, 3],
    lightAtomRange: [2, 6],
  },
  {
    name: "metal-carbide",
    targetMotif: "A15",
    baseElements: [
      ["Nb", "C"], ["Mo", "C"], ["Ti", "C"], ["V", "C"], ["Zr", "C"],
      ["Ta", "C"], ["W", "C"], ["Hf", "C"],
    ],
    lightAtom: "C",
    stoichiometryRange: [1, 3],
    lightAtomRange: [1, 4],
  },
  {
    name: "metal-nitride",
    targetMotif: "layered",
    baseElements: [
      ["Nb", "N"], ["Ti", "N"], ["V", "N"], ["Zr", "N"], ["Ta", "N"],
      ["Mo", "N"], ["Hf", "N"], ["W", "N"],
    ],
    lightAtom: "N",
    stoichiometryRange: [1, 3],
    lightAtomRange: [1, 4],
  },
  {
    name: "ternary-hydride",
    targetMotif: "cage",
    baseElements: [
      ["La", "B", "H"], ["Y", "C", "H"], ["Ca", "B", "H"], ["Sr", "N", "H"],
      ["Ba", "C", "H"], ["La", "N", "H"], ["Y", "B", "H"], ["Sc", "C", "H"],
    ],
    lightAtom: "H",
    stoichiometryRange: [1, 2],
    lightAtomRange: [4, 8],
  },
  {
    name: "high-dos-intermetallic",
    targetMotif: "A15",
    baseElements: [
      ["Nb", "Ge"], ["Nb", "Sn"], ["V", "Si"], ["V", "Ga"],
      ["Nb", "Al"], ["Mo", "Ge"], ["Ta", "Si"], ["Nb", "Ga"],
    ],
    lightAtom: "",
    stoichiometryRange: [1, 3],
    lightAtomRange: [1, 1],
  },
  {
    name: "layered-pnictide",
    targetMotif: "layered",
    baseElements: [
      ["Ba", "Fe", "As"], ["Sr", "Fe", "As"], ["La", "Fe", "P"],
      ["Ca", "Fe", "As"], ["Ba", "Co", "As"], ["Sr", "Ni", "P"],
    ],
    lightAtom: "",
    stoichiometryRange: [1, 2],
    lightAtomRange: [1, 2],
  },
  {
    name: "cuprate-layered",
    targetMotif: "layered",
    baseElements: [
      ["Y", "Ba", "Cu", "O"], ["La", "Sr", "Cu", "O"], ["Bi", "Sr", "Cu", "O"],
      ["La", "Ba", "Cu", "O"], ["Nd", "Ce", "Cu", "O"],
    ],
    lightAtom: "O",
    stoichiometryRange: [1, 3],
    lightAtomRange: [4, 7],
  },
];

function generateFromTemplate(template: DesignTemplate, count: number): string[] {
  const formulas: string[] = [];
  const seen = new Set<string>();

  for (const baseEls of template.baseElements) {
    if (formulas.length >= count) break;

    for (let metalStoich = template.stoichiometryRange[0]; metalStoich <= template.stoichiometryRange[1]; metalStoich++) {
      for (let lightCount = template.lightAtomRange[0]; lightCount <= template.lightAtomRange[1]; lightCount++) {
        if (formulas.length >= count) break;

        const counts: Record<string, number> = {};
        for (let i = 0; i < baseEls.length; i++) {
          const el = baseEls[i];
          if (el === template.lightAtom) {
            counts[el] = lightCount;
          } else if (i === 0) {
            counts[el] = metalStoich;
          } else if (LIGHT_ATOMS.includes(el) || PNICTOGEN_ELEMENTS.includes(el) || CHALCOGEN_ELEMENTS.includes(el)) {
            counts[el] = Math.max(1, Math.round(metalStoich * (1 + Math.random())));
          } else {
            counts[el] = Math.max(1, Math.round(metalStoich * (0.5 + Math.random() * 0.5)));
          }
        }

        const f = countsToFormula(counts);
        if (!seen.has(f) && f.length > 1) {
          seen.add(f);
          formulas.push(f);
        }
      }
    }
  }

  return formulas;
}

export function runPillarGuidedGeneration(
  targets: SCPillarTargets = DEFAULT_PILLAR_TARGETS,
  candidatesPerTemplate: number = 8,
): PillarGuidedCandidate[] {
  const results: PillarGuidedCandidate[] = [];
  const allFormulas: string[] = [];

  for (const template of DESIGN_TEMPLATES) {
    const formulas = generateFromTemplate(template, candidatesPerTemplate);
    allFormulas.push(...formulas);
  }

  totalGenerated += allFormulas.length;

  for (const formula of allFormulas) {
    try {
      const evaluation = evaluatePillars(formula, targets);
      if (!evaluation.physicsValid) continue;

      const strengths: string[] = [];
      if (evaluation.pillarScores.coupling >= 0.7) strengths.push(`strong e-ph coupling (lambda=${evaluation.lambda.toFixed(2)})`);
      if (evaluation.pillarScores.phonon >= 0.7) strengths.push(`high phonons (omega=${evaluation.omegaLogK.toFixed(0)}K)`);
      if (evaluation.pillarScores.dos >= 0.7) strengths.push(`high DOS (${evaluation.dos.toFixed(2)})`);
      if (evaluation.pillarScores.nesting >= 0.7) strengths.push(`good nesting (${evaluation.nestingScore.toFixed(2)})`);
      if (evaluation.pillarScores.structure >= 0.7) strengths.push(`favorable ${evaluation.motifMatch}`);

      const rationale = strengths.length > 0
        ? `${evaluation.satisfiedPillars}/5 pillars met: ${strengths.join("; ")}. Weakest: ${evaluation.weakestPillar}`
        : `Low pillar satisfaction (${evaluation.satisfiedPillars}/5). Weakest: ${evaluation.weakestPillar}`;

      results.push({
        formula,
        evaluation,
        designRationale: rationale,
      });
    } catch {}
  }

  results.sort((a, b) => b.evaluation.compositeFitness - a.evaluation.compositeFitness);
  return results;
}

let rewardBaseline = 30;
let rewardCount = 0;

export function updatePillarWeightsFromReward(tcReward: number, evaluation: PillarEvaluation): void {
  rewardCount++;
  rewardBaseline = rewardBaseline * 0.99 + tcReward * 0.01;

  const lr = 0.003;
  const tcDelta = tcReward - rewardBaseline;
  const normalizedReward = Math.min(1.0, Math.max(-1.0, tcDelta / 100));

  const pillarEntries = Object.entries(evaluation.pillarScores) as [string, number][];
  for (const [pillar, score] of pillarEntries) {
    const key = pillar as keyof typeof pillarWeights;
    if (score >= 0.7 && normalizedReward > 0.1) {
      pillarWeights[key] = Math.min(0.5, pillarWeights[key] + lr);
    } else if (score < 0.3 && normalizedReward < -0.1) {
      pillarWeights[key] = Math.max(0.05, pillarWeights[key] - lr * 0.5);
    }
  }

  const totalWeight = Object.values(pillarWeights).reduce((s, w) => s + w, 0);
  for (const key of Object.keys(pillarWeights) as (keyof typeof pillarWeights)[]) {
    pillarWeights[key] /= totalWeight;
  }
}

export function runPillarCycle(
  existingFormulas: string[],
  targetTc: number = 200,
): { formulas: string[]; evaluations: PillarEvaluation[]; bestFormula: string; bestFitness: number; bestTc: number } {
  const targets: SCPillarTargets = {
    ...DEFAULT_PILLAR_TARGETS,
    minLambda: targetTc > 150 ? 2.0 : targetTc > 50 ? 1.5 : 1.0,
    minOmegaLogK: targetTc > 150 ? 800 : targetTc > 50 ? 600 : 400,
    minDOS: targetTc > 150 ? 3.0 : 2.0,
  };

  const guided = runPillarGuidedGeneration(targets, 6);

  const reEvalExisting: PillarEvaluation[] = [];
  for (const f of existingFormulas.slice(0, 10)) {
    try {
      reEvalExisting.push(evaluatePillars(f, targets));
    } catch {}
  }

  for (const ex of reEvalExisting) {
    if (ex.compositeFitness > 0.6) {
      try {
        const mutated = mutateTowardWeakPillar(ex, targets);
        if (mutated) {
          const mutEval = evaluatePillars(mutated, targets);
          if (mutEval.compositeFitness > ex.compositeFitness) {
            guided.push({
              formula: mutated,
              evaluation: mutEval,
              designRationale: `Mutation of ${ex.formula}: improved ${ex.weakestPillar}`,
            });
          }
        }
      } catch {}
    }
  }

  guided.sort((a, b) => b.evaluation.compositeFitness - a.evaluation.compositeFitness);

  const passingFormulas = guided
    .filter(g => g.evaluation.compositeFitness > 0.4 && g.evaluation.physicsValid)
    .map(g => g.formula);

  const evals = guided.map(g => g.evaluation);

  let cycleBest = "";
  let cycleBestFitness = 0;
  let cycleBestTc = 0;
  for (const g of guided) {
    if (g.evaluation.compositeFitness > cycleBestFitness) {
      cycleBestFitness = g.evaluation.compositeFitness;
      cycleBest = g.formula;
      cycleBestTc = g.evaluation.tcPredicted;
    }
  }

  return {
    formulas: passingFormulas,
    evaluations: evals,
    bestFormula: cycleBest,
    bestFitness: cycleBestFitness,
    bestTc: cycleBestTc,
  };
}

function mutateTowardWeakPillar(
  evaluation: PillarEvaluation,
  targets: SCPillarTargets,
): string | null {
  const counts = parseCounts(evaluation.formula);
  const elements = Object.keys(counts);
  const mutated = { ...counts };

  switch (evaluation.weakestPillar) {
    case "coupling": {
      const hasLight = elements.some(e => LIGHT_ATOMS.includes(e));
      if (!hasLight) {
        const lightEl = ["H", "B", "C"][Math.floor(Math.random() * 3)];
        mutated[lightEl] = 2 + Math.floor(Math.random() * 4);
      } else {
        for (const el of elements) {
          if (LIGHT_ATOMS.includes(el)) {
            mutated[el] = Math.min(10, mutated[el] + 2);
            break;
          }
        }
      }
      break;
    }
    case "phonon": {
      if (!elements.includes("H")) {
        mutated["H"] = 4 + Math.floor(Math.random() * 4);
      } else {
        mutated["H"] = Math.min(12, (mutated["H"] || 0) + 2);
      }
      break;
    }
    case "dos": {
      const hasTM = elements.some(e => HIGH_COUPLING_TM.includes(e));
      if (!hasTM) {
        const tm = HIGH_COUPLING_TM[Math.floor(Math.random() * HIGH_COUPLING_TM.length)];
        mutated[tm] = 1 + Math.floor(Math.random() * 2);
      } else {
        for (const el of elements) {
          if (HIGH_COUPLING_TM.includes(el)) {
            mutated[el] = Math.min(4, mutated[el] + 1);
            break;
          }
        }
      }
      break;
    }
    case "nesting": {
      const hasPnictogen = elements.some(e => PNICTOGEN_ELEMENTS.includes(e));
      if (!hasPnictogen && !elements.includes("O")) {
        const pn = PNICTOGEN_ELEMENTS[Math.floor(Math.random() * PNICTOGEN_ELEMENTS.length)];
        mutated[pn] = 2;
      }
      const hasLayerFormer = elements.some(e => LAYER_FORMERS.includes(e));
      if (!hasLayerFormer) {
        const lf = LAYER_FORMERS[Math.floor(Math.random() * LAYER_FORMERS.length)];
        mutated[lf] = 2;
      }
      break;
    }
    case "structure": {
      const hasCageFormer = elements.some(e => CAGE_FORMERS.includes(e));
      if (!hasCageFormer) {
        const cf = CAGE_FORMERS[Math.floor(Math.random() * CAGE_FORMERS.length)];
        mutated[cf] = 1;
      }
      if (!elements.includes("H") && !elements.includes("B")) {
        mutated["H"] = 6;
      }
      break;
    }
  }

  const result = countsToFormula(mutated);
  if (result === evaluation.formula) return null;

  const constraint = checkPhysicsConstraints(result);
  if (!constraint.isValid && !constraint.repairedFormula) return null;

  return constraint.isValid ? result : constraint.repairedFormula;
}

export function getPillarOptimizerStats(): PillarOptimizerStats {
  const affinityScores: Record<string, number> = {};
  for (const [el, data] of Object.entries(elementAffinity)) {
    if (data.count >= 3) {
      affinityScores[el] = Math.round((data.totalFitness / data.count) * 1000) / 1000;
    }
  }

  const sorted = Object.entries(affinityScores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);
  const topAffinity: Record<string, number> = {};
  for (const [el, score] of sorted) topAffinity[el] = score;

  const satisfactionRates: Record<string, number> = {};
  for (const [key, count] of Object.entries(pillarSatisfied)) {
    satisfactionRates[key] = totalEvaluated > 0 ? Math.round((count / totalEvaluated) * 1000) / 1000 : 0;
  }

  return {
    totalEvaluated,
    totalGenerated,
    avgCompositeFitness: totalEvaluated > 0 ? Math.round((fitnessSum / totalEvaluated) * 1000) / 1000 : 0,
    bestFitness: Math.round(bestFitness * 1000) / 1000,
    bestFormula,
    pillarSatisfactionRates: satisfactionRates,
    elementAffinityScores: topAffinity,
    topCandidates: topCandidates.slice(0, 10),
    pillarWeights: { ...pillarWeights },
  };
}
