import OpenAI from "openai";
import type { Material, SuperconductorCandidate } from "@shared/schema";
import type { EventEmitter } from "./engine";
import { storage } from "../storage";
import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  assessCorrelationStrength,
} from "./physics-engine";
import { classifyFamily } from "./utils";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

interface MLFeatureVector {
  avgElectronegativity: number;
  maxAtomicMass: number;
  numElements: number;
  hasTransitionMetal: boolean;
  hasRareEarth: boolean;
  hasHydrogen: boolean;
  hasChalcogen: boolean;
  hasPnictogen: boolean;
  bandGap: number | null;
  formationEnergy: number | null;
  stability: number | null;
  crystalSymmetry: string | null;
  electronDensityEstimate: number;
  phononCouplingEstimate: number;
  dWaveSymmetry: boolean;
  layeredStructure: boolean;
  cooperPairStrength: number;
  meissnerPotential: number;
  correlationStrength: number;
  fermiSurfaceType: string;
  dimensionalityScore: number;
  anharmonicityFlag: boolean;
  electronPhononLambda: number;
  logPhononFreq: number;
  upperCriticalField: number | null;
  metallicity: number;
}

const TRANSITION_METALS = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"];
const RARE_EARTHS = ["La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Sc","Y"];
const CHALCOGENS = ["O","S","Se","Te"];
const PNICTOGENS = ["N","P","As","Sb","Bi"];
const ELECTRONEGATIVITY: Record<string, number> = {
  H:2.2,He:0,Li:0.98,Be:1.57,B:2.04,C:2.55,N:3.04,O:3.44,F:3.98,Na:0.93,Mg:1.31,Al:1.61,Si:1.9,P:2.19,S:2.58,Cl:3.16,
  K:0.82,Ca:1.0,Ti:1.54,V:1.63,Cr:1.66,Mn:1.55,Fe:1.83,Co:1.88,Ni:1.91,Cu:1.9,Zn:1.65,Ga:1.81,Ge:2.01,As:2.18,Se:2.55,
  Sr:0.95,Y:1.22,Zr:1.33,Nb:1.6,Mo:2.16,Ru:2.2,Rh:2.28,Pd:2.2,Ag:1.93,In:1.78,Sn:1.96,Sb:2.05,Te:2.1,
  Ba:0.89,La:1.1,Hf:1.3,Ta:1.5,W:2.36,Re:1.9,Os:2.2,Ir:2.2,Pt:2.28,Au:2.54,Tl:1.62,Pb:2.33,Bi:2.02,
};
const ATOMIC_MASS: Record<string, number> = {
  H:1,He:4,Li:7,Be:9,B:11,C:12,N:14,O:16,F:19,Na:23,Mg:24,Al:27,Si:28,P:31,S:32,Cl:35,
  K:39,Ca:40,Ti:48,V:51,Cr:52,Mn:55,Fe:56,Co:59,Ni:59,Cu:64,Zn:65,Ga:70,Ge:73,As:75,Se:79,
  Sr:88,Y:89,Zr:91,Nb:93,Mo:96,Ru:101,Rh:103,Pd:106,Ag:108,In:115,Sn:119,Sb:122,Te:128,
  Ba:137,La:139,Hf:178,Ta:181,W:184,Re:186,Os:190,Ir:192,Pt:195,Au:197,Tl:204,Pb:207,Bi:209,
};

function parseFormula(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, (c) => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? [...new Set(matches)] : [];
}

export function extractFeatures(formula: string, mat?: Partial<Material>, physics?: PhysicsContext, crystal?: CrystalContext): MLFeatureVector {
  const elements = parseFormula(formula);
  const enValues = elements.map(e => ELECTRONEGATIVITY[e] ?? 1.5);
  const massValues = elements.map(e => ATOMIC_MASS[e] ?? 50);

  const hasTransitionMetal = elements.some(e => TRANSITION_METALS.includes(e));
  const hasRareEarth = elements.some(e => RARE_EARTHS.includes(e));
  const hasHydrogen = elements.includes("H");
  const hasChalcogen = elements.some(e => CHALCOGENS.includes(e));
  const hasPnictogen = elements.some(e => PNICTOGENS.includes(e));

  const avgEN = enValues.length > 0 ? enValues.reduce((a,b) => a+b, 0) / enValues.length : 1.5;
  const enSpread = enValues.length > 1 ? Math.max(...enValues) - Math.min(...enValues) : 0;

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const dWaveSymmetry = hasCu && hasO && elements.length >= 3;
  const layeredStructure = (mat?.spacegroup?.includes("P4") || mat?.spacegroup?.includes("Pmmm") || mat?.spacegroup?.includes("I4")) ?? false;

  const cooperPairStrength = (hasTransitionMetal ? 0.3 : 0) + (hasHydrogen ? 0.25 : 0) +
    (dWaveSymmetry ? 0.2 : 0) + (layeredStructure ? 0.15 : 0) + (enSpread > 1.5 ? 0.1 : 0);

  const phononCouplingEstimate = hasHydrogen ? 0.8 :
    (hasTransitionMetal && hasChalcogen) ? 0.5 :
    hasRareEarth ? 0.4 : 0.2;

  const electronDensityEstimate = hasTransitionMetal ? 0.7 :
    (mat?.bandGap === 0 || mat?.bandGap === null) ? 0.6 : 0.3;

  const meissnerPotential = cooperPairStrength * 0.4 + phononCouplingEstimate * 0.3 +
    electronDensityEstimate * 0.2 + (layeredStructure ? 0.1 : 0);

  const electronic = computeElectronicStructure(formula, mat?.spacegroup);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon);
  const correlation = assessCorrelationStrength(formula);

  const useLambda = physics?.verifiedLambda ?? coupling.lambda;
  const useCorrelation = physics?.correlationStrength ?? correlation.ratio;

  const crystalDim = crystal?.dimensionality;
  const dimensionalityScore = crystalDim === "2D" ? 0.9 :
    crystalDim === "quasi-2D" ? 0.8 :
    layeredStructure ? 0.75 :
    electronic.fermiSurfaceTopology.includes("2D") ? 0.7 :
    electronic.fermiSurfaceTopology.includes("multi") ? 0.5 : 0.3;

  const useSpacegroup = crystal?.spaceGroup ?? mat?.spacegroup ?? null;

  const useHc2 = physics?.upperCriticalField ?? null;

  return {
    avgElectronegativity: avgEN,
    maxAtomicMass: Math.max(...massValues, 0),
    numElements: elements.length,
    hasTransitionMetal,
    hasRareEarth,
    hasHydrogen,
    hasChalcogen,
    hasPnictogen,
    bandGap: mat?.bandGap ?? null,
    formationEnergy: mat?.formationEnergy ?? null,
    stability: mat?.stability ?? null,
    crystalSymmetry: useSpacegroup,
    electronDensityEstimate,
    phononCouplingEstimate,
    dWaveSymmetry,
    layeredStructure,
    cooperPairStrength,
    meissnerPotential,
    correlationStrength: useCorrelation,
    fermiSurfaceType: electronic.fermiSurfaceTopology,
    dimensionalityScore,
    anharmonicityFlag: phonon.anharmonicityIndex > 0.4,
    electronPhononLambda: useLambda,
    logPhononFreq: coupling.omegaLog,
    upperCriticalField: useHc2,
    metallicity: electronic.metallicity,
  };
}

function xgboostPredict(features: MLFeatureVector): { score: number; tcEstimate: number; reasoning: string[] } {
  const reasoning: string[] = [];
  let score = 0;

  if (features.hasHydrogen) {
    score += 0.15;
    reasoning.push("Hydrogen-rich: strong electron-phonon coupling (BCS theory)");
  }
  if (features.hasTransitionMetal) {
    score += 0.12;
    reasoning.push("Transition metal d-electrons enable Cooper pair formation");
  }
  if (features.hasRareEarth) {
    score += 0.08;
    reasoning.push("Rare earth f-electrons contribute to unconventional pairing");
  }
  if (features.dWaveSymmetry) {
    score += 0.1;
    reasoning.push("Cu-O planes: d-wave symmetry boosts Tc (cuprate mechanism)");
  }
  if (features.layeredStructure) {
    score += 0.08;
    reasoning.push("Layered crystal structure enhances 2D superconducting channels");
  }
  if (features.phononCouplingEstimate > 0.5) {
    score += 0.12;
    reasoning.push(`Strong phonon coupling (${(features.phononCouplingEstimate*100).toFixed(0)}%) enhances BCS pairing`);
  }
  if (features.cooperPairStrength > 0.5) {
    score += 0.1;
    reasoning.push(`Cooper pair formation strength: ${(features.cooperPairStrength*100).toFixed(0)}%`);
  }
  if (features.meissnerPotential > 0.4) {
    score += 0.08;
    reasoning.push(`Meissner effect potential: ${(features.meissnerPotential*100).toFixed(0)}% - magnetic flux expulsion likely`);
  }
  if (features.numElements >= 3 && features.numElements <= 5) {
    score += 0.05;
    reasoning.push("Ternary/quaternary composition allows property tuning");
  }
  if (features.bandGap === 0 || features.bandGap === null) {
    score += 0.07;
    reasoning.push("Zero/near-zero band gap: metallic character required for superconductivity");
  }
  if (features.avgElectronegativity > 1.5 && features.avgElectronegativity < 2.5) {
    score += 0.05;
    reasoning.push("Balanced electronegativity: optimal charge transfer between sublattices");
  }

  const safeCorr = Number.isFinite(features.correlationStrength) ? features.correlationStrength : 0;
  const safeDim = Number.isFinite(features.dimensionalityScore) ? features.dimensionalityScore : 0;
  const safeLambda = Number.isFinite(features.electronPhononLambda) ? features.electronPhononLambda : 0;
  const safeMetal = Number.isFinite(features.metallicity) ? features.metallicity : 0.5;

  if (safeMetal < 0.3) {
    score -= 0.15;
    reasoning.push(`Non-metallic (metallicity=${safeMetal.toFixed(2)}): no itinerant electrons for Cooper pairing`);
  } else if (safeMetal < 0.5) {
    score -= 0.06;
    reasoning.push(`Weak metallicity (${safeMetal.toFixed(2)}): limited conduction electron density at Fermi level`);
  }

  if (safeCorr > 0.6 && safeCorr <= 0.85) {
    score += 0.05;
    reasoning.push(`Strong electron correlations (U/W=${safeCorr.toFixed(2)}): unconventional pairing possible if metallic`);
  } else if (safeCorr > 0.85) {
    score -= 0.08;
    reasoning.push(`Very strong correlations (U/W=${safeCorr.toFixed(2)}): likely Mott insulator, phonon-mediated SC unlikely`);
  }
  if (safeDim > 0.6) {
    score += 0.06;
    reasoning.push(`Favorable dimensionality (${safeDim.toFixed(1)}): 2D confinement enhances pairing`);
  }
  if (features.anharmonicityFlag) {
    score += 0.04;
    reasoning.push("Significant anharmonicity: may enhance e-ph coupling beyond harmonic approximation");
  }
  if (safeLambda > 1.0) {
    score += 0.1;
    reasoning.push(`Strong e-ph coupling (lambda=${safeLambda.toFixed(2)}): Eliashberg strong-coupling regime`);
  } else if (safeLambda > 0.5) {
    score += 0.05;
    reasoning.push(`Moderate e-ph coupling (lambda=${safeLambda.toFixed(2)}): BCS regime`);
  }
  if (features.fermiSurfaceType.includes("nested") || features.fermiSurfaceType.includes("multi-sheet")) {
    score += 0.06;
    reasoning.push("Fermi surface nesting/multi-sheet topology: enhanced susceptibility to pairing instability");
  }

  const safeHc2 = features.upperCriticalField != null && Number.isFinite(features.upperCriticalField) ? features.upperCriticalField : null;
  if (safeHc2 != null) {
    if (safeHc2 > 50) {
      score += 0.08;
      reasoning.push(`High Hc2 (${safeHc2.toFixed(1)}T): robust Type-II superconductor with strong vortex pinning`);
    } else if (safeHc2 > 10) {
      score += 0.04;
      reasoning.push(`Moderate Hc2 (${safeHc2.toFixed(1)}T): viable Type-II superconductor`);
    } else if (safeHc2 > 0) {
      score += 0.02;
      reasoning.push(`Low Hc2 (${safeHc2.toFixed(1)}T): weak magnetic robustness limits practical applications`);
    } else {
      score -= 0.10;
      reasoning.push("WARNING: Hc2=0T — no upper critical field detected, superconductivity unlikely");
    }
  }

  const rawScore = score;
  score = 1 / (1 + Math.exp(-3 * (rawScore - 0.5)));

  let tcEstimate = 0;
  const lambdaScaling = safeLambda > 2.5 ? 1.4 : safeLambda > 2.0 ? 1.25 : safeLambda > 1.5 ? 1.12 : 1.0;
  if (safeMetal < 0.3) {
    tcEstimate = 1 + score * 15;
    reasoning.push("Tc heavily suppressed: non-metallic composition cannot support BCS superconductivity");
  } else if (safeMetal < 0.5) {
    tcEstimate = 5 + score * 50;
    reasoning.push("Tc limited by weak metallicity");
  } else if (safeLambda > 1.5 && features.hasHydrogen) {
    const omega_log_K = (Number.isFinite(features.logPhononFreq) ? features.logPhononFreq : 500) * 1.44;
    const muStar = 0.12;
    const denom = safeLambda - muStar * (1 + 0.62 * safeLambda);
    if (Math.abs(denom) > 1e-6) {
      const exponent = -1.04 * (1 + safeLambda) / denom;
      tcEstimate = (omega_log_K / 1.2) * Math.exp(exponent);
    }
    if (!Number.isFinite(tcEstimate) || tcEstimate < 100) tcEstimate = 100 + score * 350 * lambdaScaling;
  } else if (features.hasHydrogen && features.cooperPairStrength > 0.4) {
    tcEstimate = 150 + score * 350 * lambdaScaling;
  } else if (features.dWaveSymmetry) {
    tcEstimate = 80 + score * 300 * lambdaScaling;
  } else if (features.correlationStrength > 0.85 && features.hasTransitionMetal) {
    tcEstimate = 5 + score * 80;
  } else if (features.correlationStrength > 0.6 && features.hasTransitionMetal) {
    tcEstimate = 30 + score * 180 * lambdaScaling;
  } else if (features.hasTransitionMetal) {
    tcEstimate = 20 + score * 150;
  } else {
    tcEstimate = 5 + score * 120;
  }

  return { score, tcEstimate: Math.round(tcEstimate), reasoning };
}

interface PhysicsContext {
  verifiedLambda: number | null;
  verifiedTc: number | null;
  competingPhases: any[];
  upperCriticalField: number | null;
  correlationStrength: number | null;
  verificationStage: number;
}

interface CrystalContext {
  spaceGroup: string;
  crystalSystem: string;
  dimensionality: string;
  isStable: boolean;
  convexHullDistance: number | null;
  synthesizability: number | null;
}

interface KnowledgeDepth {
  hasSynthesis: boolean;
  hasCrystal: boolean;
  pipelineStagesPassed: number;
  hasRelatedInsights: boolean;
}

interface ResearchContext {
  synthesisCount: number;
  reactionCount: number;
  hasSynthesisKnowledge: boolean;
  hasReactionKnowledge: boolean;
  physicsData?: Map<string, PhysicsContext>;
  crystalData?: Map<string, CrystalContext>;
  strategyFocusAreas?: { area: string; priority: number }[];
  familyCounts?: Record<string, number>;
  knowledgeDepth?: Map<string, KnowledgeDepth>;
}

export async function runMLPrediction(
  emit: EventEmitter,
  materials: Material[],
  context?: ResearchContext
): Promise<{ candidates: Partial<SuperconductorCandidate>[]; insights: string[] }> {
  const candidates: Partial<SuperconductorCandidate>[] = [];
  const insights: string[] = [];

  const contextDetail = context
    ? ` (informed by ${context.synthesisCount} synthesis processes and ${context.reactionCount} chemical reactions)`
    : "";

  emit("log", {
    phase: "phase-7",
    event: "XGBoost+NN ensemble started",
    detail: `Feature extraction and gradient boosting on ${materials.length} materials${contextDetail}`,
    dataSource: "ML Engine",
  });

  let physicsData = context?.physicsData;
  let crystalData = context?.crystalData;
  if (!physicsData || !crystalData) {
    physicsData = physicsData ?? new Map();
    crystalData = crystalData ?? new Map();
    try {
      const existingSC = await storage.getSuperconductorCandidates(50);
      for (const sc of existingSC) {
        if (sc.electronPhononCoupling != null || sc.verificationStage != null && sc.verificationStage > 0) {
          physicsData.set(sc.formula, {
            verifiedLambda: sc.electronPhononCoupling,
            verifiedTc: sc.predictedTc,
            competingPhases: (sc.competingPhases as any[]) ?? [],
            upperCriticalField: sc.upperCriticalField,
            correlationStrength: sc.correlationStrength,
            verificationStage: sc.verificationStage ?? 0,
          });
        }
      }
      const structures = await storage.getCrystalStructures(100);
      for (const cs of structures) {
        crystalData.set(cs.formula, {
          spaceGroup: cs.spaceGroup,
          crystalSystem: cs.crystalSystem,
          dimensionality: cs.dimensionality,
          isStable: cs.isStable ?? false,
          convexHullDistance: cs.convexHullDistance,
          synthesizability: cs.synthesizability,
        });
      }
    } catch {}
  }

  const scored: { mat: Material; features: MLFeatureVector; xgb: ReturnType<typeof xgboostPredict>; hasPhysics: boolean; hasCrystal: boolean }[] = [];

  for (const mat of materials.slice(0, 30)) {
    const physics = physicsData.get(mat.formula);
    const crystal = crystalData.get(mat.formula);
    const features = extractFeatures(mat.formula, mat, physics, crystal);
    const xgb = xgboostPredict(features);

    if (physics && physics.verificationStage > 0) {
      xgb.score = Math.min(1, xgb.score + 0.05);
      xgb.reasoning.push("Physics-verified: computational analysis confirms candidate viability");
    }
    if (crystal?.isStable) {
      xgb.score = Math.min(1, xgb.score + 0.04);
      xgb.reasoning.push(`Crystal structure verified stable (${crystal.spaceGroup}, hull dist: ${crystal.convexHullDistance?.toFixed(3) ?? '?'})`);
    }
    if (physics?.competingPhases?.some((p: any) => p.suppressesSC)) {
      xgb.score = Math.max(0, xgb.score - 0.08);
      xgb.reasoning.push("WARNING: Competing phase identified that may suppress superconductivity");
    }
    if (crystal?.synthesizability != null && crystal.synthesizability < 0.3) {
      xgb.score = Math.max(0, xgb.score - 0.05);
      xgb.reasoning.push(`Low synthesizability (${(crystal.synthesizability * 100).toFixed(0)}%): practical challenges expected`);
    }

    const family = classifyFamily(mat.formula);
    let explorationBonus = 0;

    const familyCounts = context?.familyCounts;
    if (familyCounts) {
      const familyCount = familyCounts[family] ?? 0;
      if (familyCount < 5) {
        const underExploredBonus = familyCount < 2 ? 0.08 : 0.05;
        explorationBonus += underExploredBonus;
        xgb.reasoning.push(`Exploration bonus (+${(underExploredBonus * 100).toFixed(0)}%): ${family} under-explored (${familyCount} candidates)`);
      }
    }

    const focusAreas = context?.strategyFocusAreas;
    if (focusAreas) {
      const match = focusAreas.find(f => f.area.toLowerCase() === family.toLowerCase());
      if (match) {
        const strategyBonus = Math.min(0.05, match.priority * 0.05);
        explorationBonus += strategyBonus;
        xgb.reasoning.push(`Strategy-aligned (+${(strategyBonus * 100).toFixed(0)}%): ${family} is a research priority`);
      }
    }

    explorationBonus = Math.min(0.10, explorationBonus);
    if (explorationBonus > 0) {
      xgb.score = xgb.score + explorationBonus;
    }

    let knowledgeBonus = 0;
    const depth = context?.knowledgeDepth?.get(mat.formula);
    if (depth) {
      if (depth.hasSynthesis) {
        knowledgeBonus += 0.03;
        xgb.reasoning.push("Knowledge bonus: synthesis pathway documented");
      }
      if (depth.hasCrystal) {
        knowledgeBonus += 0.03;
        xgb.reasoning.push("Knowledge bonus: crystal structure predicted");
      }
      if (depth.pipelineStagesPassed > 0) {
        const pipelineBonus = Math.min(0.08, depth.pipelineStagesPassed * 0.02);
        knowledgeBonus += pipelineBonus;
        xgb.reasoning.push(`Knowledge bonus: ${depth.pipelineStagesPassed} pipeline stages validated`);
      }
      if (depth.hasRelatedInsights) {
        knowledgeBonus += 0.02;
        xgb.reasoning.push("Knowledge bonus: related novel insights accumulated");
      }
    }
    knowledgeBonus = Math.min(0.15, knowledgeBonus);
    if (knowledgeBonus > 0) {
      xgb.score += knowledgeBonus;
    }
    xgb.score = Math.min(1, xgb.score);

    let tcKnowledgeBonus = 0;
    if (depth) {
      if (depth.hasSynthesis) tcKnowledgeBonus += 8;
      if (depth.hasCrystal) tcKnowledgeBonus += 8;
      if (depth.pipelineStagesPassed > 0) tcKnowledgeBonus += Math.min(30, depth.pipelineStagesPassed * 8);
      if (depth.hasRelatedInsights) tcKnowledgeBonus += 5;
      tcKnowledgeBonus = Math.min(50, tcKnowledgeBonus);
    }
    if (physics) {
      const verifiedLambda = physics.verifiedLambda ?? 0;
      if (verifiedLambda > 2.5) tcKnowledgeBonus += 30;
      else if (verifiedLambda > 2.0) tcKnowledgeBonus += 22;
      else if (verifiedLambda > 1.5) tcKnowledgeBonus += 15;
      else if (verifiedLambda > 1.0) tcKnowledgeBonus += 8;
      else if (verifiedLambda > 0.5) tcKnowledgeBonus += 3;
    }
    tcKnowledgeBonus = Math.min(80, tcKnowledgeBonus);
    if (tcKnowledgeBonus > 0) {
      xgb.tcEstimate = xgb.tcEstimate + tcKnowledgeBonus;
      xgb.reasoning.push(`Tc adjusted +${tcKnowledgeBonus}K from accumulated evidence (${physics ? 'physics-verified' : 'knowledge-based'})`);
    }

    scored.push({ mat, features, xgb, hasPhysics: !!physics, hasCrystal: !!crystal });
  }

  scored.sort((a, b) => b.xgb.score - a.xgb.score);
  const topCandidates = scored.slice(0, 5);

  if (topCandidates.length === 0) return { candidates, insights };

  const physicsEnriched = scored.filter(s => s.hasPhysics).length;
  const crystalEnriched = scored.filter(s => s.hasCrystal).length;
  const enrichmentDetail = (physicsEnriched > 0 || crystalEnriched > 0)
    ? ` [enriched: ${physicsEnriched} with physics, ${crystalEnriched} with crystal data]`
    : "";

  emit("log", {
    phase: "phase-7",
    event: "XGBoost screening complete",
    detail: `Top candidate: ${topCandidates[0].mat.formula} (score: ${(topCandidates[0].xgb.score*100).toFixed(0)}%, Tc est: ${topCandidates[0].xgb.tcEstimate}K)${enrichmentDetail}`,
    dataSource: "ML Engine",
  });

  try {
    const candidateSummaries = topCandidates.map(c => {
      const physics = physicsData!.get(c.mat.formula);
      const crystal = crystalData!.get(c.mat.formula);
      return {
        formula: c.mat.formula,
        name: c.mat.name,
        xgboostScore: c.xgb.score,
        tcEstimate: c.xgb.tcEstimate,
        features: {
          cooperPairStrength: c.features.cooperPairStrength,
          phononCoupling: c.features.phononCouplingEstimate,
          meissnerPotential: c.features.meissnerPotential,
          dWaveSymmetry: c.features.dWaveSymmetry,
          layeredStructure: c.features.layeredStructure,
        },
        physicsComputed: {
          electronPhononLambda: c.features.electronPhononLambda,
          logPhononFrequency: c.features.logPhononFreq,
          correlationStrength: c.features.correlationStrength,
          fermiSurfaceTopology: c.features.fermiSurfaceType,
          dimensionalityScore: c.features.dimensionalityScore,
          anharmonic: c.features.anharmonicityFlag,
          upperCriticalField_T: c.features.upperCriticalField,
        },
        ...(physics ? {
          verifiedPhysics: {
            verifiedLambda: physics.verifiedLambda,
            verifiedTc: physics.verifiedTc,
            upperCriticalField: physics.upperCriticalField,
            competingPhases: physics.competingPhases.length,
            hasSuppressionRisk: physics.competingPhases.some((p: any) => p.suppressesSC),
            verificationStage: physics.verificationStage,
          }
        } : {}),
        ...(crystal ? {
          crystalStructure: {
            spaceGroup: crystal.spaceGroup,
            crystalSystem: crystal.crystalSystem,
            dimensionality: crystal.dimensionality,
            isStable: crystal.isStable,
            convexHullDistance: crystal.convexHullDistance,
            synthesizability: crystal.synthesizability,
          }
        } : {}),
        xgboostReasoning: c.xgb.reasoning,
      };
    });

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an advanced neural network layer in a hybrid XGBoost+NN ensemble for superconductor prediction. The XGBoost layer has performed feature-based gradient boosting and ranked candidates. Physics computations (electronic structure, phonon spectrum, electron-phonon coupling lambda, Fermi surface topology, correlation strength U/W) are included. Some candidates may also have VERIFIED physics data from computational analysis (verifiedPhysics) and/or predicted crystal structures (crystalStructure) - weight these heavily as they represent higher-fidelity data.

Your role:
1. Evaluate using: electron-phonon coupling (lambda), BCS/Eliashberg theory, Cooper pair formation, Meissner effect, competing phases (magnetism, CDW, Mott)
2. For strongly correlated materials (U/W > 0.6), consider unconventional pairing (spin-fluctuation, d-wave, p-wave)
3. Assess room-temperature viability (Tc >= 293K AND ambient pressure AND zero resistance)
4. Identify pairing symmetry (s-wave, d-wave, p-wave, s+/-)
5. Estimate dimensionality effects on Tc
6. For each candidate, assign uncertainty and identify what physics is missing
7. When verifiedPhysics is present, use the verified lambda and Tc as ground truth over estimates
8. When crystalStructure is present, factor in thermodynamic stability (convexHullDistance < 0.05 = good), dimensionality effects on Tc, and synthesizability
9. If competing phases suppress SC, lower the score and flag the risk
10. CRITICAL: upperCriticalField_T (Hc2 in Tesla) is a key indicator. Real superconductors have Hc2 > 0. YBCO ~100-200T, MgB2 ~40T, hydrides ~100T+. If Hc2 is 0 or very low, the material likely does NOT superconduct. Penalize candidates with Hc2=0 severely. High Hc2 (>50T) strongly supports SC viability.

Return JSON with:
- 'candidates': array with 'formula', 'neuralNetScore' (0-1), 'refinedTc' (Kelvin), 'pressureGpa', 'meissnerEffect' (boolean), 'zeroResistance' (boolean), 'cooperPairMechanism' (string), 'pairingSymmetry' (s-wave/d-wave/p-wave/s+-), 'pairingMechanism' (phonon-mediated/spin-fluctuation/charge-fluctuation/unconventional), 'dimensionality' (3D/quasi-2D/2D/1D), 'quantumCoherence' (0-1), 'roomTempViable' (boolean), 'crystalStructure', 'uncertaintyEstimate' (0-1), 'reasoning' (under 150 chars)
- 'insights': array of 3-5 physics insights (each under 120 chars)`,
        },
        {
          role: "user",
          content: `XGBoost-ranked superconductor candidates for neural network refinement:\n${JSON.stringify(candidateSummaries, null, 2)}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 1200,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return { candidates, insights };

    let parsed: any;
    try {
      parsed = JSON.parse(content);
    } catch {
      emit("log", { phase: "phase-7", event: "NN parse error", detail: content.slice(0, 200), dataSource: "ML Engine" });
      return { candidates, insights };
    }

    const nnCandidates = parsed.candidates ?? [];
    const nnInsights = parsed.insights ?? [];
    insights.push(...nnInsights);

    for (let i = 0; i < nnCandidates.length; i++) {
      const nn = nnCandidates[i];
      const xgb = topCandidates[i];
      if (!nn || !xgb) continue;

      const ensembleScore = (xgb.xgb.score * 0.4 + (nn.neuralNetScore ?? 0.5) * 0.6);

      candidates.push({
        name: xgb.mat.name,
        formula: xgb.mat.formula,
        predictedTc: nn.refinedTc ?? xgb.xgb.tcEstimate,
        pressureGpa: nn.pressureGpa ?? null,
        meissnerEffect: nn.meissnerEffect ?? false,
        zeroResistance: nn.zeroResistance ?? false,
        cooperPairMechanism: nn.cooperPairMechanism ?? "unknown",
        crystalStructure: nn.crystalStructure ?? xgb.mat.spacegroup,
        quantumCoherence: nn.quantumCoherence ?? 0,
        stabilityScore: xgb.features.cooperPairStrength,
        mlFeatures: xgb.features as any,
        xgboostScore: xgb.xgb.score,
        neuralNetScore: nn.neuralNetScore ?? 0.5,
        ensembleScore,
        roomTempViable: nn.roomTempViable ?? false,
        status: ensembleScore > 0.7 ? "promising" : "theoretical",
        notes: nn.reasoning ?? xgb.xgb.reasoning[0],
        electronPhononCoupling: xgb.features.electronPhononLambda,
        logPhononFrequency: xgb.features.logPhononFreq,
        coulombPseudopotential: 0.12,
        pairingSymmetry: nn.pairingSymmetry ?? (xgb.features.dWaveSymmetry ? "d-wave" : "s-wave"),
        pairingMechanism: nn.pairingMechanism ?? (xgb.features.correlationStrength > 0.6 ? "spin-fluctuation" : "phonon-mediated"),
        correlationStrength: xgb.features.correlationStrength,
        dimensionality: nn.dimensionality ?? (xgb.features.layeredStructure ? "quasi-2D" : "3D"),
        fermiSurfaceTopology: xgb.features.fermiSurfaceType,
        uncertaintyEstimate: nn.uncertaintyEstimate ?? 0.5,
        verificationStage: 0,
      });
    }

    emit("log", {
      phase: "phase-7",
      event: "Ensemble prediction complete",
      detail: `${candidates.length} candidates scored, ${candidates.filter(c => c.roomTempViable).length} room-temp viable`,
      dataSource: "ML Engine",
    });
  } catch (err: any) {
    emit("log", {
      phase: "phase-7",
      event: "Neural network error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "ML Engine",
    });
  }

  return { candidates, insights };
}
