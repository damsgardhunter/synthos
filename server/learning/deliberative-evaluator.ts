import { storage } from "../storage";
import { classifyFamily, normalizeFormula, isValidFormula } from "./utils";
import { extractFeatures } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import { parseFormulaElements, computeElectronPhononCoupling, computePhononSpectrum, computeElectronicStructure } from "./physics-engine";
import { SUPERCON_TRAINING_DATA } from "./supercon-dataset";
import { predictStability } from "../physics/stability-predictor";
import { passesElementCountCap } from "./candidate-generator";

export interface DeliberationStage {
  name: string;
  score: number;
  weight: number;
  reasoning: string[];
  verdict: "strong" | "moderate" | "weak" | "reject";
}

export interface DeliberationResult {
  formula: string;
  deliberationScore: number;
  verdict: "accept" | "borderline" | "reject";
  stages: DeliberationStage[];
  reasoningChain: string[];
  selfCritiqueFlags: string[];
  comparativeRank: number | null;
  estimatedNovelty: number;
  confidenceLevel: number;
  totalTimeMs: number;
}

const KNOWN_HIGH_TC_FAMILIES: Record<string, { maxKnownTc: number; mechanism: string }> = {
  "Cuprate": { maxKnownTc: 133, mechanism: "d-wave spin fluctuation" },
  "Hydride": { maxKnownTc: 250, mechanism: "conventional phonon BCS" },
  "Pnictide": { maxKnownTc: 56, mechanism: "s+- nesting" },
  "Boride": { maxKnownTc: 39, mechanism: "sigma-bond phonon" },
  "Chalcogenide": { maxKnownTc: 15, mechanism: "nematic spin fluctuation" },
  "Bismuthate": { maxKnownTc: 30, mechanism: "charge density wave proximity" },
  "Nickelate": { maxKnownTc: 80, mechanism: "d-wave analog" },
  "Intermetallic": { maxKnownTc: 23, mechanism: "BCS phonon" },
  "HeavyFermion": { maxKnownTc: 2, mechanism: "Kondo lattice" },
  "Organic": { maxKnownTc: 14, mechanism: "charge fluctuation" },
  "Elemental": { maxKnownTc: 9, mechanism: "BCS phonon" },
};

const TC_FAMILY_CEILING_MULTIPLIER = 2.5;

const STAGE_WEIGHTS = {
  chemistry: 0.20,
  physicsMerit: 0.30,
  comparative: 0.15,
  risk: 0.20,
  selfCritique: 0.15,
};

function stageVerdict(score: number): "strong" | "moderate" | "weak" | "reject" {
  if (score >= 0.7) return "strong";
  if (score >= 0.4) return "moderate";
  if (score >= 0.2) return "weak";
  return "reject";
}

function runChemistryReasoning(formula: string): DeliberationStage {
  const reasoning: string[] = [];
  let score = 0.5;

  const elements = parseFormulaElements(formula);
  const elKeys = Array.from(elements.keys());
  const family = classifyFamily(formula);

  reasoning.push(`Family: ${family}, ${elKeys.length} elements: ${elKeys.join(", ")}`);

  const SC_ACTIVE_TM = new Set(["Nb", "V", "Ti", "Ta", "Mo", "W", "Zr", "Hf", "Fe", "Ni", "Cu", "Ru", "Co", "Mn", "Ir", "Os", "Rh", "Pd", "Pt"]);
  const LIGHT_PHONON = new Set(["H", "B", "C", "N"]);
  const CHARGE_RESERVOIR = new Set(["La", "Y", "Ca", "Sr", "Sc", "Ce", "Nd", "Ba", "K"]);
  const SC_ACTIVE_MAIN = new Set(["Pb", "Bi", "Sn", "In", "Al", "Ga", "Tl", "Mg"]);

  const hasActiveTM = elKeys.some(e => SC_ACTIVE_TM.has(e));
  const hasLightPhonon = elKeys.some(e => LIGHT_PHONON.has(e));
  const hasReservoir = elKeys.some(e => CHARGE_RESERVOIR.has(e));
  const hasActiveMain = elKeys.some(e => SC_ACTIVE_MAIN.has(e));

  if (hasActiveTM && hasLightPhonon) {
    score += 0.2;
    reasoning.push("Strong: active SC transition metal + light phonon element present");
  } else if (hasActiveTM) {
    score += 0.1;
    reasoning.push("Has SC-active transition metal");
  } else if (hasLightPhonon) {
    score += 0.05;
    reasoning.push("Has light phonon element but no active TM");
  }

  if (hasReservoir) {
    score += 0.1;
    reasoning.push("Charge reservoir element present (favorable for layered SC)");
  }

  if (hasActiveMain) {
    score += 0.05;
    reasoning.push("SC-active main group element present");
  }

  const knownFamily = KNOWN_HIGH_TC_FAMILIES[family];
  if (knownFamily) {
    const familyBonus = Math.min(0.15, knownFamily.maxKnownTc / 300 * 0.3);
    score += familyBonus;
    reasoning.push(`Known SC family (${family}): max known Tc=${knownFamily.maxKnownTc}K via ${knownFamily.mechanism}`);
  } else {
    reasoning.push(`Novel/unclassified family (${family}) — higher risk, higher potential`);
  }

  if (!passesElementCountCap(formula)) {
    return {
      name: "Chemistry Reasoning",
      score: 0,
      weight: STAGE_WEIGHTS.chemistry,
      reasoning: [...reasoning, "REJECTED: exceeds composition complexity caps"],
      verdict: "reject",
    };
  }

  const hasH = elKeys.includes("H");
  const hCount = elements.get("H") ?? 0;
  const totalAtoms = Array.from(elements.values()).reduce((s, n) => s + n, 0);

  if (hasH && hCount / totalAtoms > 0.5) {
    score += 0.05;
    reasoning.push(`Hydrogen-rich (${(hCount / totalAtoms * 100).toFixed(0)}% H) — favorable for phonon-mediated SC`);
  }

  return {
    name: "Chemistry Reasoning",
    score: Math.min(1.0, Math.max(0, score)),
    weight: STAGE_WEIGHTS.chemistry,
    reasoning,
    verdict: stageVerdict(Math.min(1.0, score)),
  };
}

function runPhysicsMerit(formula: string, predictedTc: number, mlFeatures?: any): DeliberationStage {
  const reasoning: string[] = [];
  let score = 0;

  let lambda = mlFeatures?.lambda ?? mlFeatures?.gnnLambda ?? 0;
  let omegaLog = mlFeatures?.omegaLog ?? 0;
  let dosAtFermi = mlFeatures?.dosAtFermi ?? 0;
  let bandGap = mlFeatures?.bandGap ?? 0;

  try {
    const features = extractFeatures(formula);
    if (features.electronPhononCoupling > 0) lambda = Math.max(lambda, features.electronPhononCoupling);
    dosAtFermi = Math.max(dosAtFermi, features.densityOfStatesAtFermi ?? 0);
  } catch {}

  try {
    const electronic = computeElectronicStructure(formula);
    dosAtFermi = Math.max(dosAtFermi, electronic.densityOfStatesAtFermi);
    bandGap = electronic.bandGap;
    if (electronic.metallicity > 0.7) {
      score += 0.1;
      reasoning.push(`Metallic character: ${(electronic.metallicity * 100).toFixed(0)}% — essential for SC`);
    } else if (bandGap > 0.5) {
      score -= 0.2;
      reasoning.push(`Non-metallic: bandGap=${bandGap.toFixed(2)}eV — unfavorable for SC`);
    }
  } catch {}

  if (lambda > 1.5) {
    score += 0.3;
    reasoning.push(`Very strong coupling: lambda=${lambda.toFixed(2)} (>1.5 — strong-coupling regime)`);
  } else if (lambda > 1.0) {
    score += 0.2;
    reasoning.push(`Strong coupling: lambda=${lambda.toFixed(2)} (>1.0)`);
  } else if (lambda > 0.5) {
    score += 0.1;
    reasoning.push(`Moderate coupling: lambda=${lambda.toFixed(2)} (0.5-1.0)`);
  } else if (lambda > 0) {
    reasoning.push(`Weak coupling: lambda=${lambda.toFixed(2)} (<0.5 — limited SC potential)`);
  }

  if (dosAtFermi > 5.0) {
    score += 0.15;
    reasoning.push(`High DOS at Fermi level: ${dosAtFermi.toFixed(1)} — strong N(Ef)`);
  } else if (dosAtFermi > 2.0) {
    score += 0.05;
    reasoning.push(`Moderate DOS at Fermi: ${dosAtFermi.toFixed(1)}`);
  }

  const tcNorm = Math.min(1.0, predictedTc / 300);
  score += tcNorm * 0.25;
  reasoning.push(`Predicted Tc: ${predictedTc}K (normalized: ${(tcNorm * 100).toFixed(0)}%)`);

  let gnnTc = 0;
  let gnnUncertainty = 1.0;
  try {
    const gnn = gnnPredictWithUncertainty(formula);
    gnnTc = gnn.tc;
    gnnUncertainty = gnn.uncertainty;
    if (gnnTc > 0) {
      reasoning.push(`GNN ensemble prediction: Tc=${gnnTc.toFixed(1)}K (uncertainty: ${(gnnUncertainty * 100).toFixed(0)}%)`);
    }
  } catch {}

  let gbTc = 0;
  try {
    const gb = gbPredict(extractFeatures(formula));
    gbTc = gb.tcPredicted;
    if (gbTc > 0) {
      reasoning.push(`GB model prediction: Tc=${gbTc.toFixed(1)}K`);
    }
  } catch {}

  if (gnnTc > 0 && gbTc > 0 && predictedTc > 0) {
    const modelSpread = Math.abs(gnnTc - gbTc) / Math.max(gnnTc, gbTc, 1);
    const physicsSpread = Math.abs(predictedTc - (gnnTc + gbTc) / 2) / Math.max(predictedTc, 1);
    if (modelSpread < 0.3 && physicsSpread < 0.5) {
      score += 0.1;
      reasoning.push(`Model consensus: GNN/GB/Physics Tc estimates within ${(modelSpread * 100).toFixed(0)}% — high confidence`);
    } else if (modelSpread > 0.7) {
      score -= 0.05;
      reasoning.push(`Model disagreement: GNN=${gnnTc.toFixed(0)}K vs GB=${gbTc.toFixed(0)}K (${(modelSpread * 100).toFixed(0)}% spread) — uncertain`);
    }
  }

  return {
    name: "Physics Merit",
    score: Math.min(1.0, Math.max(0, score)),
    weight: STAGE_WEIGHTS.physicsMerit,
    reasoning,
    verdict: stageVerdict(Math.min(1.0, score)),
  };
}

async function runComparativeRanking(formula: string, predictedTc: number): Promise<DeliberationStage & { rank: number | null; novelty: number }> {
  const reasoning: string[] = [];
  let score = 0.5;
  let rank: number | null = null;
  let novelty = 0.5;

  try {
    const existing = await storage.getSuperconductorCandidates(50);
    const sorted = existing
      .filter(c => (c.predictedTc ?? 0) > 0)
      .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));

    if (sorted.length > 0) {
      const betterCount = sorted.filter(c => (c.predictedTc ?? 0) > predictedTc).length;
      rank = betterCount + 1;
      const percentile = 1 - betterCount / sorted.length;
      reasoning.push(`Would rank #${rank} of ${sorted.length} candidates (${(percentile * 100).toFixed(0)}th percentile)`);

      if (percentile >= 0.9) {
        score += 0.3;
        reasoning.push("Top 10% — elite candidate");
      } else if (percentile >= 0.7) {
        score += 0.15;
        reasoning.push("Top 30% — competitive candidate");
      } else if (percentile >= 0.5) {
        score += 0.05;
        reasoning.push("Median range — not exceptional");
      } else {
        score -= 0.1;
        reasoning.push("Below median — many existing candidates outperform");
      }
    }

    const myFamily = classifyFamily(formula);
    const myElements = new Set(Array.from(parseFormulaElements(formula).keys()));
    const sameFamily = sorted.filter(c => classifyFamily(c.formula) === myFamily);

    if (sameFamily.length === 0) {
      novelty = 0.9;
      score += 0.15;
      reasoning.push(`Novel family (${myFamily}) — no existing candidates in this category`);
    } else {
      const bestInFamily = Math.max(...sameFamily.map(c => c.predictedTc ?? 0));
      if (predictedTc > bestInFamily * 1.1) {
        novelty = 0.7;
        score += 0.1;
        reasoning.push(`Exceeds best in ${myFamily} family (${bestInFamily.toFixed(0)}K) by ${((predictedTc / bestInFamily - 1) * 100).toFixed(0)}%`);
      } else {
        const elementOverlaps = sameFamily.map(c => {
          const cElements = new Set(Array.from(parseFormulaElements(c.formula).keys()));
          const intersection = new Set([...myElements].filter(e => cElements.has(e)));
          return intersection.size / Math.max(myElements.size, cElements.size);
        });
        const maxOverlap = Math.max(...elementOverlaps);
        if (maxOverlap > 0.8) {
          novelty = 0.2;
          score -= 0.1;
          reasoning.push(`High element overlap (${(maxOverlap * 100).toFixed(0)}%) with existing ${myFamily} candidates — low novelty`);
        } else {
          novelty = 0.5;
          reasoning.push(`Moderate novelty within ${myFamily} family (max overlap: ${(maxOverlap * 100).toFixed(0)}%)`);
        }
      }
    }
  } catch (e) {
    reasoning.push("Comparative ranking unavailable (database error)");
  }

  return {
    name: "Comparative Ranking",
    score: Math.min(1.0, Math.max(0, score)),
    weight: STAGE_WEIGHTS.comparative,
    reasoning,
    verdict: stageVerdict(Math.min(1.0, score)),
    rank,
    novelty,
  };
}

function runRiskAssessment(formula: string, predictedTc: number, mlFeatures?: any): DeliberationStage {
  const reasoning: string[] = [];
  let score = 0.6;

  let formationEnergy = 0;
  try {
    formationEnergy = computeMiedemaFormationEnergy(formula);
    if (formationEnergy < -0.5) {
      score += 0.15;
      reasoning.push(`Favorable formation energy: ${formationEnergy.toFixed(3)} eV/atom — thermodynamically stable`);
    } else if (formationEnergy < 0) {
      score += 0.05;
      reasoning.push(`Marginally stable: formation energy ${formationEnergy.toFixed(3)} eV/atom`);
    } else {
      score -= 0.15;
      reasoning.push(`Positive formation energy: ${formationEnergy.toFixed(3)} eV/atom — thermodynamically unfavorable`);
    }
  } catch {}

  let stabilityPred;
  try {
    stabilityPred = predictStability(formula);
    if (stabilityPred.stabilityClass === "stable") {
      score += 0.1;
      reasoning.push(`Stability prediction: stable (synthesizability: ${(stabilityPred.synthesizabilityScore * 100).toFixed(0)}%)`);
    } else if (stabilityPred.stabilityClass === "metastable") {
      reasoning.push(`Metastable: may require specific synthesis conditions (synthesizability: ${(stabilityPred.synthesizabilityScore * 100).toFixed(0)}%)`);
    } else {
      score -= 0.15;
      reasoning.push(`Predicted unstable: high decomposition risk (${(stabilityPred.decompositionRisk * 100).toFixed(0)}%)`);
    }
  } catch {}

  const hullDistance = mlFeatures?.stabilityGate?.hullDistance ?? 0;
  if (hullDistance > 0) {
    if (hullDistance < 0.02) {
      score += 0.05;
      reasoning.push(`Near convex hull: ${(hullDistance * 1000).toFixed(1)} meV/atom — competitive with stable phases`);
    } else if (hullDistance < 0.05) {
      reasoning.push(`Moderately above hull: ${(hullDistance * 1000).toFixed(1)} meV/atom`);
    } else {
      score -= 0.1;
      reasoning.push(`Far from hull: ${(hullDistance * 1000).toFixed(1)} meV/atom — synthesis challenging`);
    }
  }

  const elements = parseFormulaElements(formula);
  const hasH = elements.has("H");
  const hCount = elements.get("H") ?? 0;
  if (hasH && hCount >= 6) {
    score -= 0.1;
    reasoning.push(`High hydrogen content (H${hCount}) — likely requires extreme pressure for stabilization`);
  }

  return {
    name: "Risk Assessment",
    score: Math.min(1.0, Math.max(0, score)),
    weight: STAGE_WEIGHTS.risk,
    reasoning,
    verdict: stageVerdict(Math.min(1.0, score)),
  };
}

function runSelfCritique(formula: string, predictedTc: number, previousStages: DeliberationStage[]): DeliberationStage {
  const reasoning: string[] = [];
  const flags: string[] = [];
  let score = 0.5;

  const family = classifyFamily(formula);
  const knownFamily = KNOWN_HIGH_TC_FAMILIES[family];

  if (knownFamily && predictedTc > knownFamily.maxKnownTc * TC_FAMILY_CEILING_MULTIPLIER) {
    const ratio = predictedTc / knownFamily.maxKnownTc;
    score -= 0.2;
    flags.push(`Tc prediction (${predictedTc}K) is ${ratio.toFixed(1)}x the known max for ${family} (${knownFamily.maxKnownTc}K) — likely overestimated`);
    reasoning.push(`CAUTION: Predicted Tc exceeds family ceiling by ${ratio.toFixed(1)}x — treating with skepticism`);
  } else if (knownFamily && predictedTc > knownFamily.maxKnownTc * 1.5) {
    score -= 0.05;
    reasoning.push(`Tc exceeds known ${family} max by ${((predictedTc / knownFamily.maxKnownTc - 1) * 100).toFixed(0)}% — ambitious but not implausible`);
  } else if (knownFamily) {
    score += 0.1;
    reasoning.push(`Tc within plausible range for ${family} family`);
  }

  if (predictedTc > 293 && family !== "Hydride" && family !== "Cuprate") {
    score -= 0.15;
    flags.push(`Room-temperature claim (${predictedTc}K) in non-hydride/non-cuprate family — historically unprecedented`);
    reasoning.push("Self-critique: room-temperature prediction in non-conventional family warrants extreme caution");
  }

  const stageScores = previousStages.map(s => s.score);
  const minStage = Math.min(...stageScores);
  const maxStage = Math.max(...stageScores);

  if (maxStage - minStage > 0.6) {
    score -= 0.1;
    flags.push(`Large score disparity across evaluation stages (${minStage.toFixed(2)} to ${maxStage.toFixed(2)}) — inconsistent assessment`);
    reasoning.push(`Stage scores vary widely: min=${minStage.toFixed(2)}, max=${maxStage.toFixed(2)} — mixed signals`);
  } else {
    score += 0.05;
    reasoning.push("Consistent assessment across all stages");
  }

  const rejectCount = previousStages.filter(s => s.verdict === "reject").length;
  const weakCount = previousStages.filter(s => s.verdict === "weak").length;
  if (rejectCount > 0) {
    score -= 0.2;
    reasoning.push(`${rejectCount} stage(s) flagged as reject — fundamental concerns exist`);
  } else if (weakCount >= 2) {
    score -= 0.1;
    reasoning.push(`${weakCount} stages rated weak — multiple areas of concern`);
  } else if (weakCount === 0) {
    score += 0.15;
    reasoning.push("No weak stages — balanced candidate");
  }

  const superconMatches = SUPERCON_TRAINING_DATA.filter(d => {
    const dFamily = classifyFamily(d.formula);
    return dFamily === family;
  });

  if (superconMatches.length > 0) {
    const avgKnownTc = superconMatches.reduce((s, d) => s + d.tc, 0) / superconMatches.length;
    const maxKnownTc = Math.max(...superconMatches.map(d => d.tc));
    reasoning.push(`Reference: ${superconMatches.length} known ${family} superconductors in dataset (avg Tc=${avgKnownTc.toFixed(1)}K, max=${maxKnownTc.toFixed(0)}K)`);

    if (predictedTc > maxKnownTc * 3) {
      score -= 0.1;
      flags.push(`Prediction ${(predictedTc / maxKnownTc).toFixed(1)}x higher than best known ${family} SC — extraordinary claim`);
    }
  } else {
    reasoning.push(`No reference superconductors found for ${family} in training data — novel territory`);
  }

  for (const flag of flags) {
    reasoning.push(`FLAG: ${flag}`);
  }

  return {
    name: "Self-Critique",
    score: Math.min(1.0, Math.max(0, score)),
    weight: STAGE_WEIGHTS.selfCritique,
    reasoning,
    verdict: stageVerdict(Math.min(1.0, score)),
  };
}

export async function deliberateOnCandidate(
  formula: string,
  predictedTc: number,
  mlFeatures?: any
): Promise<DeliberationResult> {
  const startTime = Date.now();

  const chemistryStage = runChemistryReasoning(formula);
  const physicsStage = runPhysicsMerit(formula, predictedTc, mlFeatures);
  const riskStage = runRiskAssessment(formula, predictedTc, mlFeatures);

  const comparativeStage = await runComparativeRanking(formula, predictedTc);

  const priorStages = [chemistryStage, physicsStage, comparativeStage, riskStage];
  const selfCritiqueStage = runSelfCritique(formula, predictedTc, priorStages);

  const allStages: DeliberationStage[] = [
    chemistryStage,
    physicsStage,
    comparativeStage,
    riskStage,
    selfCritiqueStage,
  ];

  const weightedSum = allStages.reduce((sum, s) => sum + s.score * s.weight, 0);
  const totalWeight = allStages.reduce((sum, s) => sum + s.weight, 0);
  const deliberationScore = totalWeight > 0 ? weightedSum / totalWeight : 0;

  const rejectStages = allStages.filter(s => s.verdict === "reject").length;
  const strongStages = allStages.filter(s => s.verdict === "strong").length;

  let verdict: "accept" | "borderline" | "reject";
  if (rejectStages >= 2 || deliberationScore < 0.25) {
    verdict = "reject";
  } else if (deliberationScore >= 0.50 || strongStages >= 3) {
    verdict = "accept";
  } else {
    verdict = "borderline";
  }

  const reasoningChain: string[] = [];
  for (const stage of allStages) {
    reasoningChain.push(`[${stage.name}] Score: ${stage.score.toFixed(3)} (${stage.verdict})`);
    for (const r of stage.reasoning) {
      reasoningChain.push(`  - ${r}`);
    }
  }
  reasoningChain.push(`[Final] Deliberation score: ${deliberationScore.toFixed(3)}, verdict: ${verdict}`);

  const selfCritiqueFlags = allStages
    .flatMap(s => s.reasoning.filter(r => r.startsWith("FLAG:") || r.startsWith("CAUTION:")));

  const modelScores = allStages.map(s => s.score);
  const scoreVariance = modelScores.reduce((sum, s) => sum + Math.pow(s - deliberationScore, 2), 0) / modelScores.length;
  const confidenceLevel = Math.max(0.1, 1 - Math.sqrt(scoreVariance));

  return {
    formula,
    deliberationScore: Math.round(deliberationScore * 1000) / 1000,
    verdict,
    stages: allStages,
    reasoningChain,
    selfCritiqueFlags,
    comparativeRank: comparativeStage.rank,
    estimatedNovelty: comparativeStage.novelty,
    confidenceLevel: Math.round(confidenceLevel * 1000) / 1000,
    totalTimeMs: Date.now() - startTime,
  };
}

export function formatDeliberationSummary(result: DeliberationResult): string {
  const stagesSummary = result.stages
    .map(s => `${s.name}=${s.score.toFixed(2)}(${s.verdict})`)
    .join(", ");
  return `[Deliberation] ${result.formula}: score=${result.deliberationScore.toFixed(3)}, verdict=${result.verdict}, confidence=${result.confidenceLevel.toFixed(2)}, rank=#${result.comparativeRank ?? "?"}, novelty=${result.estimatedNovelty.toFixed(2)} | ${stagesSummary}`;
}
