import type { DiscoveredTheory, DiscoveryFeedback } from "../theory/symbolic-physics-discovery";
import { getTheoryDatabase, generateDiscoveryFeedback } from "../theory/symbolic-physics-discovery";
import type { CausalGraph, CausalEdge, CausalRule, DesignGuidance } from "../theory/causal-physics-discovery";
import { getLatestGraph, getCausalRules } from "../theory/causal-physics-discovery";

export interface TheoryGeneratorBias {
  generatorWeightBoosts: Record<string, number>;
  familyPreferences: Record<string, number>;
  elementBoosts: Record<string, number>;
  motifBiases: Record<string, number>;
  structuralGuidance: string[];
  confidence: number;
  sourceTheories: number;
  sourceCausalEdges: number;
  timestamp: number;
}

interface TheoryPerformanceRecord {
  biasAppliedAt: number;
  generatorWeightsBefore: Record<string, number>;
  generatorWeightsAfter: Record<string, number>;
  candidatesGeneratedAfter: number;
  bestTcAfter: number;
  avgTcAfter: number;
  passRateAfter: number;
}

const VARIABLE_TO_FAMILY_MAP: Record<string, string[]> = {
  hydrogen_density: ["hydride"],
  lambda: ["hydride", "intermetallic", "boride"],
  DOS_EF: ["intermetallic", "cuprate", "iron-based"],
  nesting_score: ["iron-based", "cuprate", "chalcogenide"],
  layeredness: ["cuprate", "iron-based", "chalcogenide"],
  phonon_freq: ["hydride", "boride"],
  omega_log: ["hydride", "boride"],
  mott_proximity: ["cuprate"],
  spin_fluct_strength: ["iron-based", "cuprate"],
  charge_transfer: ["cuprate", "oxide-perovskite"],
  bandwidth: ["intermetallic", "boride"],
  pressure: ["hydride"],
  dimensionality: ["cuprate", "iron-based"],
  coupling_strength: ["hydride", "intermetallic"],
  band_flatness: ["kagome-metal", "cuprate"],
};

const VARIABLE_TO_ELEMENT_MAP: Record<string, string[]> = {
  hydrogen_density: ["H", "La", "Y", "Ca", "Ce", "Th"],
  lambda: ["Nb", "V", "Ta", "La", "Y"],
  DOS_EF: ["Fe", "Co", "Ni", "Cu", "Mn"],
  nesting_score: ["Fe", "Co", "Ni", "Cu"],
  layeredness: ["Bi", "Sr", "Ca", "Cu", "La"],
  phonon_freq: ["H", "B", "C", "N"],
  omega_log: ["H", "B", "C"],
  mott_proximity: ["Cu", "Ni", "V", "Ti"],
  spin_fluct_strength: ["Fe", "Co", "Mn"],
  charge_transfer: ["Cu", "O", "Ba", "Sr", "La"],
  bandwidth: ["Nb", "V", "Ti", "Zr", "Mo"],
  pressure: ["H", "La", "Y", "Ca"],
  dimensionality: ["Bi", "Cu", "Fe", "Se"],
  coupling_strength: ["La", "Y", "H", "Nb"],
  band_flatness: ["V", "Mn", "Co", "Kagome"],
};

const VARIABLE_TO_MOTIF_MAP: Record<string, string[]> = {
  hydrogen_density: ["clathrate", "BCC-hydride", "FCC-hydride"],
  nesting_score: ["ThCr2Si2", "layered-cuprate"],
  layeredness: ["layered-cuprate", "Ruddlesden-Popper", "BiS2-layer", "TMD"],
  mott_proximity: ["layered-cuprate", "infinite-layer", "nickelate"],
  spin_fluct_strength: ["ThCr2Si2", "layered-cuprate"],
  lambda: ["A15", "clathrate", "AlB2-type"],
  bandwidth: ["A15", "Heusler"],
  band_flatness: ["kagome", "pyrochlore"],
  charge_transfer: ["perovskite", "anti-perovskite"],
  dimensionality: ["TMD", "1T-prime-TMD"],
};

const VARIABLE_TO_GENERATOR_MAP: Record<string, string[]> = {
  hydrogen_density: ["structure_diffusion", "motif_diffusion"],
  lambda: ["bo_exploration", "rl"],
  DOS_EF: ["rl", "bo_exploration"],
  nesting_score: ["rl", "structure_diffusion"],
  layeredness: ["structure_diffusion", "motif_diffusion"],
  phonon_freq: ["structure_diffusion", "bo_exploration"],
  mott_proximity: ["rl", "bo_exploration"],
  bandwidth: ["bo_exploration", "rl"],
  pressure: ["structure_diffusion", "motif_diffusion"],
  coupling_strength: ["bo_exploration", "structure_diffusion"],
  band_flatness: ["motif_diffusion", "rl"],
};

const MAX_FAMILY_BOOST = 1.5;
const MAX_ELEMENT_BOOST = 1.0;
const MAX_MOTIF_BOOST = 1.2;
const MAX_GENERATOR_BOOST = 0.8;
const MIN_EFFECTIVENESS_FLOOR = 0.1;

const VARIABLE_SYNONYMS: Record<string, string> = {
  electron_phonon_coupling: "lambda",
  epc: "lambda",
  el_ph_coupling: "lambda",
  dos_at_ef: "DOS_EF",
  dos_ef: "DOS_EF",
  density_of_states: "DOS_EF",
  fermi_nesting: "nesting_score",
  fs_nesting: "nesting_score",
  phonon_frequency: "phonon_freq",
  debye_frequency: "phonon_freq",
  omega_ln: "omega_log",
  h_density: "hydrogen_density",
  h_content: "hydrogen_density",
  hydrogen_content: "hydrogen_density",
  layer_count: "layeredness",
  mott_gap: "mott_proximity",
  spin_fluctuation: "spin_fluct_strength",
  magnetic_fluctuation: "spin_fluct_strength",
  ct_energy: "charge_transfer",
  band_width: "bandwidth",
  w_band: "bandwidth",
  flat_band: "band_flatness",
  flatness: "band_flatness",
  applied_pressure: "pressure",
  hydrostatic_pressure: "pressure",
  el_ph_lambda: "coupling_strength",
};

function normalizeVariableName(v: string): string {
  const lower = v.toLowerCase().replace(/[\s-]+/g, "_");
  return VARIABLE_SYNONYMS[lower] || v;
}

function cappedAdd(current: number, boost: number, cap: number): number {
  return Math.min(cap, current + boost);
}

const KNOWN_BIAS_VARIABLES = new Set<string>([
  ...Object.keys(VARIABLE_TO_FAMILY_MAP),
  ...Object.keys(VARIABLE_TO_ELEMENT_MAP),
  ...Object.keys(VARIABLE_TO_MOTIF_MAP),
  ...Object.keys(VARIABLE_TO_GENERATOR_MAP),
  "Tc", "mu_star", "omega_log", "phonon_softening", "anharmonicity",
  "debye_temp", "temperature", "strain", "doping", "defect_density",
  "topology_z2", "berry_phase", "atomic_mass_avg", "electronegativity_spread",
  "coordination_number", "bond_length_dist", "van_hove_distance",
  "pairing_symmetry", "orbital_fluct",
]);

export function validateBiasedVariables(variables: string[]): { valid: string[]; rejected: string[] } {
  const valid: string[] = [];
  const rejected: string[] = [];
  for (const v of variables) {
    const normalized = normalizeVariableName(v);
    if (KNOWN_BIAS_VARIABLES.has(normalized)) {
      valid.push(normalized);
    } else {
      rejected.push(v);
    }
  }
  if (rejected.length > 0) {
    console.log(`[TheoryBias] Rejected ${rejected.length} unknown biased variables: ${rejected.slice(0, 5).join(", ")}${rejected.length > 5 ? "..." : ""}`);
  }
  return { valid, rejected };
}

let currentBias: TheoryGeneratorBias | null = null;
let performanceHistory: TheoryPerformanceRecord[] = [];
let totalBiasApplications = 0;
let cumulativeTcImpact = 0;
let biasEffectivenessScores: Map<string, number> = new Map();

interface SafetyResetState {
  preBiasPassRate: number;
  preBiasBestTc: number;
  preBiasAvgTc: number;
  biasAppliedCycle: number;
  evaluationWindow: number;
  postBiasPassRates: number[];
  postBiasBestTcs: number[];
  resetTriggered: boolean;
  resetCount: number;
  discardedTheoryTimestamps: number[];
}

const safetyReset: SafetyResetState = {
  preBiasPassRate: 0,
  preBiasBestTc: 0,
  preBiasAvgTc: 0,
  biasAppliedCycle: 0,
  evaluationWindow: 3,
  postBiasPassRates: [],
  postBiasBestTcs: [],
  resetTriggered: false,
  resetCount: 0,
  discardedTheoryTimestamps: [],
};

const EFFICIENCY_DROP_THRESHOLD = 0.4;
const TC_DROP_THRESHOLD = 0.25;
const MIN_EVAL_SAMPLES = 3;

export function computeTheoryGeneratorBias(): TheoryGeneratorBias {
  const theories = getTheoryDatabase();
  const graph = getLatestGraph();
  const rules = getCausalRules();

  const generatorWeightBoosts: Record<string, number> = {};
  const familyPreferences: Record<string, number> = {};
  const elementBoosts: Record<string, number> = {};
  const motifBiases: Record<string, number> = {};
  const structuralGuidance: string[] = [];

  let confidence = 0;
  let sourceTheories = 0;
  let sourceCausalEdges = 0;

  const topTheories = theories
    .filter(t => t.theoryScore > 0.3 && t.dimensionallyValid)
    .sort((a, b) => b.theoryScore - a.theoryScore)
    .slice(0, 8);
  sourceTheories = topTheories.length;

  for (const theory of topTheories) {
    const weight = theory.theoryScore;

    for (const fi of theory.featureImportance.slice(0, 4)) {
      const varName = normalizeVariableName(fi.variable);
      const importance = fi.importance * weight;

      const families = VARIABLE_TO_FAMILY_MAP[varName];
      if (families) {
        for (const fam of families) {
          familyPreferences[fam] = cappedAdd(familyPreferences[fam] || 0, importance * 0.3, MAX_FAMILY_BOOST);
        }
      }

      const elements = VARIABLE_TO_ELEMENT_MAP[varName];
      if (elements) {
        for (const el of elements) {
          elementBoosts[el] = cappedAdd(elementBoosts[el] || 0, importance * 0.2, MAX_ELEMENT_BOOST);
        }
      }

      const motifs = VARIABLE_TO_MOTIF_MAP[varName];
      if (motifs) {
        for (const m of motifs) {
          motifBiases[m] = cappedAdd(motifBiases[m] || 0, importance * 0.25, MAX_MOTIF_BOOST);
        }
      }

      const generators = VARIABLE_TO_GENERATOR_MAP[varName];
      if (generators) {
        for (const gen of generators) {
          generatorWeightBoosts[gen] = cappedAdd(generatorWeightBoosts[gen] || 0, importance * 0.15, MAX_GENERATOR_BOOST);
        }
      }
    }

    if (theory.variables.includes("lambda") && theory.variables.includes("omega_log")) {
      structuralGuidance.push("Prioritize high-phonon-frequency materials with strong electron-phonon coupling");
    }
    if (theory.variables.includes("nesting_score") && theory.variables.includes("DOS_EF")) {
      structuralGuidance.push("Focus on materials with Fermi surface nesting and high DOS");
    }
    if (theory.variables.includes("layeredness")) {
      structuralGuidance.push("Explore layered structures with charge reservoir layers");
    }
  }

  if (graph) {
    const strongEdges = graph.edges
      .filter(e => e.strength > 0.3 && e.confidence > 0.4)
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 15);
    sourceCausalEdges = strongEdges.length;

    for (const edge of strongEdges) {
      const normalizedTarget = normalizeVariableName(edge.target);
      if (normalizedTarget === "Tc" || normalizedTarget === "lambda" || normalizedTarget === "coupling_strength") {
        const sourceVar = normalizeVariableName(edge.source);
        const boost = edge.strength * 0.4;

        const families = VARIABLE_TO_FAMILY_MAP[sourceVar];
        if (families) {
          for (const fam of families) {
            familyPreferences[fam] = cappedAdd(familyPreferences[fam] || 0, boost, MAX_FAMILY_BOOST);
          }
        }

        const elements = VARIABLE_TO_ELEMENT_MAP[sourceVar];
        if (elements) {
          for (const el of elements) {
            elementBoosts[el] = cappedAdd(elementBoosts[el] || 0, boost * 0.5, MAX_ELEMENT_BOOST);
          }
        }

        const generators = VARIABLE_TO_GENERATOR_MAP[sourceVar];
        if (generators) {
          for (const gen of generators) {
            generatorWeightBoosts[gen] = cappedAdd(generatorWeightBoosts[gen] || 0, boost * 0.3, MAX_GENERATOR_BOOST);
          }
        }
      }
    }
  }

  for (const rule of rules.filter(r => r.strength > 0.3).slice(0, 10)) {
    const normalizedConsequent = normalizeVariableName(rule.consequent);
    if (normalizedConsequent === "Tc" || normalizedConsequent === "lambda") {
      const antVar = normalizeVariableName(rule.antecedent);
      const motifs = VARIABLE_TO_MOTIF_MAP[antVar];
      if (motifs) {
        for (const m of motifs) {
          motifBiases[m] = cappedAdd(motifBiases[m] || 0, rule.strength * 0.2, MAX_MOTIF_BOOST);
        }
      }
    }
  }

  const effectivenessMultiplier = computeEffectivenessMultiplier();
  for (const key of Object.keys(generatorWeightBoosts)) {
    generatorWeightBoosts[key] *= effectivenessMultiplier;
  }
  for (const key of Object.keys(familyPreferences)) {
    familyPreferences[key] *= effectivenessMultiplier;
  }
  for (const key of Object.keys(elementBoosts)) {
    elementBoosts[key] *= effectivenessMultiplier;
  }
  for (const key of Object.keys(motifBiases)) {
    motifBiases[key] *= effectivenessMultiplier;
  }

  confidence = Math.min(1, (sourceTheories * 0.1 + sourceCausalEdges * 0.05) * effectivenessMultiplier);

  const bias: TheoryGeneratorBias = {
    generatorWeightBoosts,
    familyPreferences,
    elementBoosts,
    motifBiases,
    structuralGuidance: Array.from(new Set(structuralGuidance)),
    confidence,
    sourceTheories,
    sourceCausalEdges,
    timestamp: Date.now(),
  };

  currentBias = bias;
  return bias;
}

function computeEffectivenessMultiplier(): number {
  if (performanceHistory.length < 2) return 1.0;

  const recent = performanceHistory.slice(-5);
  let positiveCount = 0;
  for (const record of recent) {
    if (record.bestTcAfter > 0 && record.passRateAfter > 0.05) {
      positiveCount++;
    }
  }

  const ratio = positiveCount / recent.length;
  return Math.max(MIN_EFFECTIVENESS_FLOOR, 0.5 + ratio);
}

export function recordTheoryBiasOutcome(
  generatorWeightsBefore: Record<string, number>,
  generatorWeightsAfter: Record<string, number>,
  candidatesGenerated: number,
  bestTc: number,
  avgTc: number,
  passRate: number,
) {
  const record: TheoryPerformanceRecord = {
    biasAppliedAt: Date.now(),
    generatorWeightsBefore,
    generatorWeightsAfter,
    candidatesGeneratedAfter: candidatesGenerated,
    bestTcAfter: bestTc,
    avgTcAfter: avgTc,
    passRateAfter: passRate,
  };

  performanceHistory.push(record);
  if (performanceHistory.length > 50) {
    performanceHistory = performanceHistory.slice(-50);
  }

  totalBiasApplications++;

  if (performanceHistory.length >= 2) {
    const prev = performanceHistory[performanceHistory.length - 2];
    const tcDelta = bestTc - prev.bestTcAfter;
    cumulativeTcImpact += tcDelta;

    for (const [gen, afterWeight] of Object.entries(generatorWeightsAfter)) {
      const beforeWeight = generatorWeightsBefore[gen] ?? afterWeight;
      const weightDelta = afterWeight - beforeWeight;
      if (Math.abs(weightDelta) > 0.01) {
        const currentScore = biasEffectivenessScores.get(gen) ?? 0;
        const reward = tcDelta > 0 ? weightDelta * tcDelta * 0.01 : 0;
        biasEffectivenessScores.set(gen, currentScore + reward);
      }
    }
  }
}

export function recordPreBiasBaseline(passRate: number, bestTc: number, avgTc: number, cycle: number): void {
  safetyReset.preBiasPassRate = passRate;
  safetyReset.preBiasBestTc = bestTc;
  safetyReset.preBiasAvgTc = avgTc;
  safetyReset.biasAppliedCycle = cycle;
  safetyReset.postBiasPassRates = [];
  safetyReset.postBiasBestTcs = [];
  safetyReset.resetTriggered = false;
}

export function recordPostBiasPerformance(passRate: number, bestTc: number): void {
  safetyReset.postBiasPassRates.push(passRate);
  safetyReset.postBiasBestTcs.push(bestTc);
  if (safetyReset.postBiasPassRates.length > 10) {
    safetyReset.postBiasPassRates.shift();
    safetyReset.postBiasBestTcs.shift();
  }
}

export function evaluateTheoryBiasSafety(): { shouldReset: boolean; reason: string; degradation: number } {
  if (!currentBias || safetyReset.resetTriggered) {
    return { shouldReset: false, reason: "no_active_bias", degradation: 0 };
  }
  if (safetyReset.postBiasPassRates.length < MIN_EVAL_SAMPLES) {
    return { shouldReset: false, reason: "insufficient_data", degradation: 0 };
  }

  const avgPostPassRate = safetyReset.postBiasPassRates.reduce((a, b) => a + b, 0) / safetyReset.postBiasPassRates.length;
  const avgPostBestTc = safetyReset.postBiasBestTcs.reduce((a, b) => a + b, 0) / safetyReset.postBiasBestTcs.length;

  const baselinePassRate = Math.max(safetyReset.preBiasPassRate, 0.001);
  const baselineBestTc = Math.max(safetyReset.preBiasBestTc, 1);

  const passRateDrop = (baselinePassRate - avgPostPassRate) / baselinePassRate;
  const tcDrop = (baselineBestTc - avgPostBestTc) / baselineBestTc;

  const passRateDegraded = passRateDrop > EFFICIENCY_DROP_THRESHOLD;
  const tcDegraded = tcDrop > TC_DROP_THRESHOLD;
  const degradation = Math.max(passRateDrop, tcDrop);

  if (passRateDegraded || tcDegraded) {
    const reasons: string[] = [];
    if (passRateDegraded) reasons.push(`pass_rate_drop=${(passRateDrop * 100).toFixed(1)}%`);
    if (tcDegraded) reasons.push(`tc_drop=${(tcDrop * 100).toFixed(1)}%`);
    return { shouldReset: true, reason: reasons.join(", "), degradation };
  }

  return { shouldReset: false, reason: "performance_stable", degradation };
}

export function resetTheoryBias(): { resetBias: TheoryGeneratorBias | null; reason: string } {
  const discardedBias = currentBias;
  if (discardedBias) {
    safetyReset.discardedTheoryTimestamps.push(discardedBias.timestamp);
    if (safetyReset.discardedTheoryTimestamps.length > 20) {
      safetyReset.discardedTheoryTimestamps.shift();
    }
  }

  currentBias = null;
  safetyReset.resetTriggered = true;
  safetyReset.resetCount++;
  safetyReset.postBiasPassRates = [];
  safetyReset.postBiasBestTcs = [];

  biasEffectivenessScores.clear();

  console.log(`[TheoryBias] Safety reset #${safetyReset.resetCount}: discarded theory bias, reverting to baseline generator weights`);
  return { resetBias: discardedBias, reason: `safety_reset_#${safetyReset.resetCount}` };
}

export function getTheoryBiasSafetyStats(): {
  resetCount: number;
  currentlyActive: boolean;
  preBiasPassRate: number;
  postBiasAvgPassRate: number;
  preBiasBestTc: number;
  postBiasAvgBestTc: number;
  evalSamples: number;
  discardedCount: number;
} {
  const avgPost = safetyReset.postBiasPassRates.length > 0
    ? safetyReset.postBiasPassRates.reduce((a, b) => a + b, 0) / safetyReset.postBiasPassRates.length
    : 0;
  const avgPostTc = safetyReset.postBiasBestTcs.length > 0
    ? safetyReset.postBiasBestTcs.reduce((a, b) => a + b, 0) / safetyReset.postBiasBestTcs.length
    : 0;
  return {
    resetCount: safetyReset.resetCount,
    currentlyActive: currentBias !== null,
    preBiasPassRate: safetyReset.preBiasPassRate,
    postBiasAvgPassRate: avgPost,
    preBiasBestTc: safetyReset.preBiasBestTc,
    postBiasAvgBestTc: avgPostTc,
    evalSamples: safetyReset.postBiasPassRates.length,
    discardedCount: safetyReset.discardedTheoryTimestamps.length,
  };
}

export function getTheoryGeneratorBias(): TheoryGeneratorBias | null {
  return currentBias;
}

export function getRLBiasFromTheory(): {
  elementGroupBias: number[];
  chemicalFamilyBias: number[];
  hydrogenDensityBias: number[];
  structureTypeBias: number[];
} {
  const bias = currentBias;

  const ELEMENT_GROUPS = [
    "alkali", "alkaline-earth", "3d-transition", "4d-transition", "5d-transition",
    "lanthanide", "p-block-metal", "metalloid", "nonmetal",
  ];
  const FAMILY_NAMES = [
    "hydride", "intermetallic", "layered-pnictide", "boride",
    "cuprate", "chalcogenide", "kagome-metal", "oxide-perovskite",
  ];

  const elementGroupBias = new Array(ELEMENT_GROUPS.length).fill(0);
  const chemicalFamilyBias = new Array(FAMILY_NAMES.length).fill(0);
  const hydrogenDensityBias = [0, 0, 0, 0];
  const structureTypeBias = new Array(20).fill(0);

  if (!bias || bias.confidence < 0.1) {
    return { elementGroupBias, chemicalFamilyBias, hydrogenDensityBias, structureTypeBias };
  }

  const ELEMENT_TO_GROUP: Record<string, number> = {
    Li: 0, Na: 0, K: 0, Rb: 0, Cs: 0,
    Be: 1, Mg: 1, Ca: 1, Sr: 1, Ba: 1,
    Sc: 2, Ti: 2, V: 2, Cr: 2, Mn: 2, Fe: 2, Co: 2, Ni: 2, Cu: 2, Zn: 2,
    Y: 3, Zr: 3, Nb: 3, Mo: 3, Ru: 3, Rh: 3, Pd: 3, Ag: 3,
    Hf: 4, Ta: 4, W: 4, Re: 4, Os: 4, Ir: 4, Pt: 4, Au: 4,
    La: 5, Ce: 5, Pr: 5, Nd: 5, Sm: 5, Gd: 5, Dy: 5, Er: 5, Yb: 5, Lu: 5,
    Al: 6, Ga: 6, In: 6, Sn: 6, Tl: 6, Pb: 6, Bi: 6,
    B: 7, Si: 7, Ge: 7, As: 7, Sb: 7, Te: 7, Se: 7,
    H: 8, C: 8, N: 8, O: 8, F: 8, P: 8, S: 8, Cl: 8,
  };

  for (const [el, boost] of Object.entries(bias.elementBoosts)) {
    const groupIdx = ELEMENT_TO_GROUP[el];
    if (groupIdx !== undefined) {
      elementGroupBias[groupIdx] += boost;
    }
  }

  for (const [fam, boost] of Object.entries(bias.familyPreferences)) {
    const famIdx = FAMILY_NAMES.indexOf(fam);
    if (famIdx >= 0) {
      chemicalFamilyBias[famIdx] += boost;
    }
  }

  const hBoost = bias.elementBoosts["H"] ?? 0;
  const hydrideBoost = bias.familyPreferences["hydride"] ?? 0;
  if (hBoost > 0.1 || hydrideBoost > 0.2) {
    hydrogenDensityBias[2] += (hBoost + hydrideBoost) * 0.5;
    hydrogenDensityBias[3] += (hBoost + hydrideBoost) * 0.7;
  }

  const MOTIF_TO_STRUCTURE_IDX: Record<string, number[]> = {
    A15: [0], AlB2: [1], perovskite: [3], ThCr2Si2: [4],
    Heusler: [5], layered: [8], kagome: [9],
    clathrate: [7], "BCC-hydride": [6], "FCC-hydride": [7],
    "layered-cuprate": [8], TMD: [11],
  };

  for (const [motif, boost] of Object.entries(bias.motifBiases)) {
    const indices = MOTIF_TO_STRUCTURE_IDX[motif];
    if (indices) {
      for (const idx of indices) {
        if (idx < structureTypeBias.length) {
          structureTypeBias[idx] += boost;
        }
      }
    }
  }

  return { elementGroupBias, chemicalFamilyBias, hydrogenDensityBias, structureTypeBias };
}

export interface TheoryGuidedGeneratorStats {
  totalBiasApplications: number;
  currentBiasConfidence: number;
  sourceTheories: number;
  sourceCausalEdges: number;
  topFamilyPreferences: { family: string; boost: number }[];
  topElementBoosts: { element: string; boost: number }[];
  topMotifBiases: { motif: string; boost: number }[];
  topGeneratorBoosts: { generator: string; boost: number }[];
  structuralGuidance: string[];
  performanceHistorySize: number;
  cumulativeTcImpact: number;
  effectivenessMultiplier: number;
}

export function getTheoryGuidedGeneratorStats(): TheoryGuidedGeneratorStats {
  const bias = currentBias;

  const topFamilies = bias
    ? Object.entries(bias.familyPreferences)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([family, boost]) => ({ family, boost: Math.round(boost * 1000) / 1000 }))
    : [];

  const topElements = bias
    ? Object.entries(bias.elementBoosts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([element, boost]) => ({ element, boost: Math.round(boost * 1000) / 1000 }))
    : [];

  const topMotifs = bias
    ? Object.entries(bias.motifBiases)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([motif, boost]) => ({ motif, boost: Math.round(boost * 1000) / 1000 }))
    : [];

  const topGenerators = bias
    ? Object.entries(bias.generatorWeightBoosts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([generator, boost]) => ({ generator, boost: Math.round(boost * 1000) / 1000 }))
    : [];

  return {
    totalBiasApplications,
    currentBiasConfidence: bias?.confidence ?? 0,
    sourceTheories: bias?.sourceTheories ?? 0,
    sourceCausalEdges: bias?.sourceCausalEdges ?? 0,
    topFamilyPreferences: topFamilies,
    topElementBoosts: topElements,
    topMotifBiases: topMotifs,
    topGeneratorBoosts: topGenerators,
    structuralGuidance: bias?.structuralGuidance ?? [],
    performanceHistorySize: performanceHistory.length,
    cumulativeTcImpact: Math.round(cumulativeTcImpact * 100) / 100,
    effectivenessMultiplier: Math.round(computeEffectivenessMultiplier() * 1000) / 1000,
  };
}
