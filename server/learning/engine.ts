import { WebSocketServer, WebSocket } from "ws";
import type { Server } from "http";
import { storage } from "../storage";
import { fetchOQMDMaterials, fetchElementFocusedMaterials, fetchKnownMaterials, getNextOQMDOffset } from "./data-fetcher";
import { analyzeBondingPatterns, analyzePropertyPredictionPatterns } from "./nlp-engine";
import { generateNovelFormulas, setBoundaryHuntingMode, setInverseDesignMode, setChemicalSpaceExpansionMode, getGenerationModes } from "./formula-generator";
import { runSuperconductorResearch, generateInverseDesignCandidates, getInverseDesignCount } from "./superconductor-research";
import { getAllActiveCampaigns, runInverseCycle, processInverseResults, getSerializableCampaignState, getInverseDesignStats as getInverseOptimizerStats, loadCampaign, restoreCampaignsFromDB, createCampaign } from "../inverse/inverse-optimizer";
import { getNextGenPipelineStats as getNextGenPipelineStatsForEngine } from "../inverse/next-gen-pipeline";
import { getAllLabStats as getSelfImprovingLabStatsForEngine } from "../inverse/self-improving-lab";
import { runGradientDescentCycle, getDifferentiableOptimizerStats } from "../inverse/differentiable-optimizer";
import { runStructureDiffusionCycle, getStructureDiffusionStats } from "../ai/structure-diffusion";
import { constraintGuidedGenerate, checkPhysicsConstraints, updateConstraintWeightsFromReward, getConstraintEngineStats } from "../inverse/physics-constraint-engine";
import { runPillarCycle, evaluatePillars, updatePillarWeightsFromReward, getPillarOptimizerStats, incorporateDFTFeedbackIntoPillars, getPillarDFTFeedbackStats } from "../inverse/sc-pillars-optimizer";
import type { InverseCandidate } from "../inverse/target-schema";
import { discoverSynthesisProcesses, discoverChemicalReactions, getNextReactionTopic } from "./synthesis-tracker";
import { runFullPhysicsAnalysis, applyAmbientTcCap, setConstraintMode, getConstraintMode, parseFormulaElements, computeElectronicStructure, reconcileTc, FAMILY_TC_CAPS, computeCapExtensionFactor, allenDynesTcRaw } from "./physics-engine";
import type { CapExtensionEvidence } from "./physics-engine";
import { runPressureAnalysis } from "./pressure-engine";
import { runStructurePredictionBatch, runGenerativeStructureDiscovery, getStructuralVariantCount, runNovelPrototypeGeneration, getNovelPrototypeCount, runEvolutionaryStructureSearch, setMutationIntensity } from "./structure-predictor";
import { runMultiFidelityPipeline } from "./multi-fidelity-pipeline";
import { evaluateInsightNovelty, requiresQuantitativeContent } from "./insight-detector";
import { analyzeAndEvolveStrategy, captureConvergenceSnapshot, trackDuplicatesSkipped } from "./strategy-analyzer";
import { checkMilestones } from "./milestone-tracker";
import { extractFeatures, physicsPredictor } from "./ml-predictor";
import type { PhysicsPrediction } from "./ml-predictor";
import { gbPredict, incorporateFailureData, getFailureExampleCount, surrogateScreen, getSurrogateStats, incorporateSuccessData, retrainWithAccumulatedData, incorporateDFTResult, retrainXGBoostFromEvaluated, getEvaluatedDatasetStats, getModelVersionHistory } from "./gradient-boost";
import { normalizeFormula, classifyFamily, sanitizeForbiddenWords, isValidFormula } from "./utils";
import { runMassiveGeneration, passesValenceFilter, passesElementCountCap, estimateFamilyPressure, mutatePressure, generatePressureVariants, type MassiveGenerationStats } from "./candidate-generator";
import { deliberateOnCandidate, formatDeliberationSummary } from "./deliberative-evaluator";
import { scanMaterialSignals } from "./material-signal-scanner";
import { resolveDFTFeatures, describeDFTSources } from "./dft-feature-resolver";
import type { DFTResolvedFeatures } from "./dft-feature-resolver";
import { runSynthesisReasoning } from "./synthesis-reasoning";
import { runConvexHullAnalysis, passesStabilityGate, computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import type { StabilityGateResult } from "./phase-diagram-engine";
import { invalidateGNNModel, trainGNNSurrogate } from "./graph-neural-net";
import { runStructuralMutations } from "./structural-mutator";
import { runDopingBatch, getDopingEngineStats, type DopingSpec } from "./doping-engine";
import { evolveRules, screenWithPatterns, getMinedRules } from "./pattern-miner";
import { findOptimalRegion, getPhaseExplorationSeedFormulas } from "./phase-explorer";
import { runFamilyAwareGeneration } from "./family-generators";
import { bayesianOptimizer } from "./bayesian-optimizer";
import { rlAgent } from "./rl-agent";
import { applyFamilyFilter, rankCandidate, computeDiscoveryScore } from "./family-filters";
import { runPrototypeGeneration, type PrototypeCandidate } from "./prototype-generator";
import { generatePrototypeFreeStructures, getLatticeGeneratorStats, seedEvoPopulation, addToEvoPopulation, runEvolutionaryGeneration, getEvoPopulationSummary, type GeneratedStructure } from "../crystal/lattice-generator";
import { gnnPredictWithUncertainty } from "./graph-neural-net";
import { runActiveLearningCycle, getActiveLearningStats } from "./active-learning";
import { predictPressureCurve, findOptimalPressure, getPressureCurveStats } from "./pressure-aware-surrogate";
import { findStabilityPressureWindow, getEnthalpyStats } from "./enthalpy-stability";
import { buildPressureResponseProfile, getPressurePropertyMapStats } from "./pressure-property-map";
import { optimizePressureForFormula, getBayesianPressureStats, addPressureObservation } from "./bayesian-pressure-optimizer";
import { recordClusterDiscovery, getPressureClusterStats, fastPressureScreen, samplePressureFromClusters } from "./pressure-screening";
import { detectPhaseTransitions, getPhaseTransitionStats } from "./pressure-phase-detector";
import { recordEvaluationResult, getCalibrationStats } from "./surrogate-fitness";
import { getXTBStats, runXTBPhononCheck, checkXTBHealth } from "../dft/qe-dft-engine";
import { submitDFTJob, getDFTQueueStats, setDFTBroadcast } from "../dft/dft-job-queue";
import { runDiffusionGenerationCycle, getDiffusionStats } from "../ai/crystal-generator";
import { runCrystalDiffusionCycle, getCrystalDiffusionStats, runDistributionBasedDiffusion, getDistributionDiffusionStats } from "../ai/crystal-structure-diffusion";
import { multiTaskPredict, trackMultiTaskPrediction, getMultiTaskStats } from "./multi-task-gnn";
import { runLatentSpaceInverseDesign, trainVAE, getVAEStats, encodeToLatent } from "../physics/materials-vae";
import { analyzeTopology, trackTopologyResult, getTopologyStats, type TopologicalAnalysis } from "../physics/topology-engine";
import { computeTopologicalInvariants, trackInvariantResult } from "../physics/topological-invariants";
import { computePairingProfile, type PairingProfile } from "../physics/pairing-mechanisms";
import { encodeGenome, genomeDiversity, type MaterialGenome } from "../physics/materials-genome";
import { computeFermiSurface, type FermiSurfaceResult } from "../physics/fermi-surface-engine";
import { assignToCluster, getClusterGuidance } from "../physics/fermi-surface-clustering";
import { analyzeHydrogenNetwork, trackHydrogenNetworkResult, type HydrogenNetworkAnalysis } from "../physics/hydrogen-network-engine";
import { analyzeReactionNetwork } from "../physics/reaction-network-engine";
import { predictBandStructure, getBandSurrogateMLFeatures, type BandSurrogatePrediction } from "../physics/band-structure-surrogate";
import { predictBandDispersion, getBandOperatorMLFeatures, type BandOperatorResult } from "../physics/band-structure-operator";
import { buildFermiSurfaceFromDFT, extractTopologyFromDFT, enhanceElectronicStructure, recordDFTBandAnalysis, getDFTBandAnalysisStats, classifyDFTTopology, computeFermiIsosurface, computeLindhardNesting, computeBandFeatureScore, runAutomatedTopologyPipeline } from "../dft/dft-band-analysis";
import type { DFTTopologicalClassification, BandFeatureScore } from "../dft/dft-band-analysis";
import type { DFTBandStructureResult } from "../dft/band-structure-calculator";
import { passesStabilityPreFilter } from "../physics/stability-predictor";
import { predictKineticStability, formatKineticStabilityNote, type KineticStabilityResult } from "../physics/kinetic-stability";
import { detectQuantumCriticality, type QuantumCriticalAnalysis } from "../physics/quantum-criticality";
import { discoveryMemory, buildFingerprint } from "./discovery-memory";
import { getGeneratorAllocations, allocateBudget, recordGeneratorOutcome, recordDFTOutcome, getGeneratorCompetitionStats, rebalanceWeights, applyTheoryBias } from "./generator-manager";
import { computeTheoryGeneratorBias, recordTheoryBiasOutcome, getTheoryGuidedGeneratorStats, getRLBiasFromTheory, type TheoryGeneratorBias } from "./theory-guided-generator";
import { buildAndStoreFeatureRecord, getDatasetSize, getFeatureDataset } from "../theory/physics-feature-db";
import { updatePhysicsParameters } from "../theory/self-improving-physics";
import { addMaterialToDataset, updateLandscape, getLandscapeStats } from "../landscape/discovery-landscape";
import { getZoneBonus, getLandscapeRLBias } from "../landscape/landscape-guidance";
import { getZoneMap } from "../landscape/zone-detector";
import { updateZoneHistory, getIntelligenceGeneratorBias, getLandscapeIntelligenceStats } from "../landscape/landscape-intelligence";
import { getConstraintGuidanceForGenerator } from "../inverse/constraint-solver";
import { getConstraintGraphGuidance } from "../inverse/constraint-graph-solver";
import { getPathwayForCandidate, getPathwayStats } from "../inverse/pressure-pathway";
import { triggerSynthesisPathwayForCandidate } from "../synthesis/reaction-pathway";
import { optimizeSynthesisConditions, getSynthesisOptimizerStats, type MaterialContext } from "../synthesis/synthesis-condition-optimizer";
import { getParameterSpace } from "../synthesis/synthesis-variables";
import {
  simulateSynthesisEffects, getSimulatorStats, optimizeSynthesisForFixedMaterial,
  defaultSynthesisVector, type SynthesisVector,
} from "../physics/synthesis-simulator";
import { recordSynthesisResult, getSynthesisLearningStats } from "../synthesis/synthesis-learning-db";
import { generateDefectVariants, adjustElectronicStructure, getDefectEngineStats } from "../physics/defect-engine";
import { generateAllDisorderVariants, suggestDisorders } from "../crystal/disorder-generator";
import { computeConfigurationalEntropy, estimateDOSDisorderSignal } from "../crystal/disorder-metrics";
import type { DisorderContext } from "./ml-predictor";
import { estimateCorrelationEffects, getCorrelationEngineStats } from "../physics/correlation-engine";
import { simulateCrystalGrowth, getCrystalGrowthStats } from "../synthesis/crystal-growth-simulator";
import { getExperimentPlannerStats, generateExperimentPlan, type ExperimentCandidate } from "../experiment-planner";
import { recordPrediction, shouldRetrain as shouldRetrainPerf, getPerformanceMetrics, recordCandidateOutcome } from "../theory/model-performance-tracker";
import { runSymbolicRegression, theoryKnowledgeBase, getDiscoveredTheories } from "../theory/symbolic-regression";
import { runHypothesisCycle, getTopHypothesesForGeneratorBias, getHypothesisStats } from "../theory/hypothesis-engine";
import { createPipeline, runPipelineIteration, getPipelineStats, getAllPipelines, type PipelineState } from "../inverse/next-gen-pipeline";
import { createLab, runLabIteration, getLabStats, type LabState } from "../inverse/self-improving-lab";
import { generateDesignProgram, executeDesignProgram, registerProgram, getDesignRepresentationStats, type DesignProgram } from "../inverse/design-representations";
import {
  runSymbolicPhysicsDiscovery, generateSyntheticDataset, getSymbolicDiscoveryStats,
  getTheoryDatabase, generateDiscoveryFeedback,
} from "../theory/symbolic-physics-discovery";
import {
  runCausalDiscovery, generateCausalDataset, getCausalDiscoveryStats,
  getDiscoveredHypotheses as getCausalHypotheses, getCausalRules,
  getLatestGraph, type CausalRule,
} from "../theory/causal-physics-discovery";
import { crossEngineHub } from "./cross-engine-hub";
import { discoverNovelSynthesisPaths, getSynthesisDiscoveryStats, recordDFTFeedbackForGA, getGAEvolutionStats, getStructuralMotifStats, type MultiEngineInsights } from "./synthesis-discovery";
import { planAndTrack, getSynthesisPlannerStats } from "../synthesis/synthesis-planner";
import { generateHeuristicRoutes, getHeuristicGeneratorStats } from "../synthesis/heuristic-synthesis-generator";
import { recordStructureOutcome } from "../crystal/structure-reward-system";
import { ELEMENTAL_DATA } from "./elemental-data";
import { runModelImprovementCycle, getModelImprovementStats } from "./model-improvement-loop";
import { evaluateSynthesisGate, getSynthesisGateStats } from "../synthesis/synthesis-gate";
import { recordPredictionOutcome } from "./model-diagnostics";

function derivePairingSymmetry(mechanism: string | null | undefined, dWaveFlag?: boolean): string {
  const mech = (mechanism ?? "").toLowerCase();
  if (mech.includes("spin") || mech.includes("excitonic")) return "d-wave";
  if (mech.includes("topolog")) return "p-wave";
  if (mech.includes("orbital")) return "s+/-";
  if (mech.includes("phonon") || mech.includes("bcs")) return "s-wave";
  if (dWaveFlag) return "d-wave";
  return "s-wave";
}

function classifyPressureViability(optimalPressureGpa: number): { label: string; penalty: number } {
  if (optimalPressureGpa > 100) return { label: "high_pressure_only", penalty: 0.6 };
  if (optimalPressureGpa > 50) return { label: "high_pressure_only", penalty: 0.4 };
  if (optimalPressureGpa > 30) return { label: "moderate_pressure", penalty: 0.15 };
  if (optimalPressureGpa > 10) return { label: "low_pressure", penalty: 0.05 };
  return { label: "ambient_possible", penalty: 0 };
}

function enforcePhysicsPressure(formula: string, llmPressure: number | null | undefined): number {
  const physPressure = estimateFamilyPressure(formula);
  const llmP = llmPressure ?? 0;
  return Math.max(llmP, physPressure);
}

function computePhysicsOnlyTc(lambda: number, omegaLogCm1: number | null | undefined, muStar?: number): number {
  if (!lambda || lambda <= 0) return 0;
  const freq = omegaLogCm1 ?? 300;
  const mu = muStar ?? 0.12;
  return allenDynesTcRaw(lambda, freq, mu);
}

const VALID_ELEMENTS_SET = new Set([
  "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am",
]);

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = (formula ?? "").replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const result: Record<string, number> = {};
  const re = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(cleaned)) !== null) {
    const el = m[1];
    if (!VALID_ELEMENTS_SET.has(el)) continue;
    const n = m[2] ? parseFloat(m[2]) : 1;
    result[el] = (result[el] || 0) + n;
  }
  return result;
}

const COMMON_OXIDATION_STATES: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  B: [3], Al: [3], Ga: [3], In: [3], Tl: [1, 3],
  C: [-4, 4], Si: [4, -4], Ge: [4], Sn: [2, 4], Pb: [2, 4],
  N: [-3, 3, 5], P: [-3, 3, 5], As: [-3, 3, 5], Sb: [-3, 3, 5], Bi: [3, 5],
  O: [-2], S: [-2, 4, 6], Se: [-2, 4, 6], Te: [-2, 4, 6],
  F: [-1], Cl: [-1, 1, 3, 5, 7], Br: [-1, 1, 3, 5], I: [-1, 1, 3, 5, 7],
  Ti: [2, 3, 4], V: [2, 3, 4, 5], Cr: [2, 3, 6], Mn: [2, 3, 4, 7],
  Fe: [2, 3], Co: [2, 3], Ni: [2, 3], Cu: [1, 2, 3], Zn: [2],
  Zr: [4], Nb: [3, 5], Mo: [4, 6], Tc: [4, 7], Ru: [3, 4], Rh: [3], Pd: [2, 4],
  Ag: [1], Cd: [2], Hf: [4], Ta: [5], W: [4, 6], Re: [4, 7],
  Os: [4, 8], Ir: [3, 4], Pt: [2, 4], Au: [1, 3], Hg: [1, 2],
  Sc: [3], Y: [3], La: [3], Ce: [3, 4], Pr: [3], Nd: [3], Sm: [3],
  Eu: [2, 3], Gd: [3], Tb: [3], Dy: [3], Ho: [3], Er: [3], Tm: [3], Yb: [2, 3], Lu: [3],
  Ac: [3], Th: [4], U: [3, 4, 5, 6],
};

function passesChemistryFilter(formula: string): { pass: boolean; reason: string } {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return { pass: true, reason: "" };

  const totalAtoms = Object.values(counts).reduce((s, v) => s + v, 0);
  const avgValence = elements.reduce((sum, el) => {
    const data = ELEMENTAL_DATA[el];
    return sum + (data?.valenceElectrons ?? 4) * (counts[el] / totalAtoms);
  }, 0);
  if (avgValence < 1 || avgValence > 12) {
    return { pass: false, reason: `avg valence electrons/atom ${avgValence.toFixed(1)} outside [1,12]` };
  }

  const ens = elements
    .map(el => ELEMENTAL_DATA[el]?.paulingElectronegativity ?? null)
    .filter((v): v is number => v !== null);
  if (ens.length >= 2) {
    const spread = Math.max(...ens) - Math.min(...ens);
    if (spread > 3.5) {
      return { pass: false, reason: `electronegativity spread ${spread.toFixed(2)} > 3.5 (too ionic)` };
    }
  }

  const canBalance = checkOxidationBalance(counts);
  if (!canBalance) {
    return { pass: false, reason: "no valid oxidation state balance found" };
  }

  return { pass: true, reason: "" };
}

function checkOxidationBalance(counts: Record<string, number>): boolean {
  const elements = Object.keys(counts);
  if (elements.length < 2) return true;

  const statesPerElement = elements.map(el => {
    const ox = COMMON_OXIDATION_STATES[el];
    if (!ox || ox.length === 0) return [0, 2, 3, -2];
    return ox;
  });

  function tryBalance(idx: number, runningSum: number): boolean {
    if (idx === elements.length) return Math.abs(runningSum) < 0.01;
    const count = counts[elements[idx]];
    for (const ox of statesPerElement[idx]) {
      if (tryBalance(idx + 1, runningSum + ox * count)) return true;
    }
    return false;
  }

  return tryBalance(0, 0);
}

export type EventEmitter = (type: string, data: any) => void;

function shouldContinue(): boolean {
  return state === "running";
}

type EngineState = "stopped" | "running" | "paused";
type EngineTempo = "excited" | "exploring" | "contemplating";

interface EngineStatus {
  state: EngineState;
  activeTasks: string[];
  cycleCount: number;
  lastCycleAt: string | null;
  totalMaterialsFetched: number;
  totalInsightsGenerated: number;
  totalPredictionsMade: number;
  totalSynthesisDiscovered: number;
  totalReactionsLearned: number;
  totalScCandidates: number;
  totalPhysicsComputed: number;
  totalStructuresPredicted: number;
  totalPipelineScreened: number;
  totalNovelSynthesisProposed: number;
  totalInverseDesigned: number;
  totalStructuralVariants: number;
  tempo: EngineTempo;
  statusMessage: string;
}

let wss: WebSocketServer | null = null;
let state: EngineState = "stopped";
let cycleTimer: ReturnType<typeof setTimeout> | null = null;
let cycleCount = 0;
let totalMaterialsFetched = 0;
let totalInsightsGenerated = 0;
let totalPredictionsMade = 0;
let totalSynthesisDiscovered = 0;
let totalReactionsLearned = 0;
let totalScCandidates = 0;
let totalPhysicsComputed = 0;
let totalStructuresPredicted = 0;
let totalPipelineScreened = 0;
let totalNovelSynthesisProposed = 0;
let activeTasks: Set<string> = new Set();
let lastCycleAt: string | null = null;
let allInsights: string[] = [];
let isRunningCycle = false;
let phase7Offset = 0;
let currentStrategyHint: string | null = null;
let currentStrategyFocusAreas: { area: string; priority: number }[] = [];
let currentFamilyCounts: Record<string, number> = {};
let engineTempo: EngineTempo = "exploring";
let cycleIntervalMs = 15000;
let exploitCyclesRemaining = 0;
let currentExploitFamily: string | null = null;
let totalDFTEnriched = 0;
let currentStatusMessage = "Initializing research systems";

let autonomousTotalScreened = 0;
let autonomousTotalPassed = 0;
let autonomousBestTc = 0;
let autonomousStartTime = Date.now();

const pipelineStageMetrics = {
  chemistryRejects: 0,
  stabilityPrefilterRejects: 0,
  surrogateRejects: 0,
  formationEnergyRejects: 0,
  gbTcRejects: 0,
  physicsPrefilterRejects: 0,
  phononRejects: 0,
  belowTierRejects: 0,
  duplicateRejects: 0,
  featureExtractionFails: 0,
  prototypeAttempts: 0,
  prototypeSuccesses: 0,
  xtbAttempts: 0,
  xtbSuccesses: 0,
  totalPassed: 0,
};

interface LastCycleCandidate {
  formula: string;
  tc: number;
  passed: boolean;
  reason: string;
  family: string;
}
let lastCycleCandidates: LastCycleCandidate[] = [];
let lastCycleFamilyCounts: Record<string, number> = {};
let autonomousGNNRetrainCount = 0;
const alreadyScreenedFormulas = new Set<string>();
const MAX_SCREENED_CACHE_SIZE = 50000;

export function getAlreadyScreenedFormulas(): Set<string> {
  return alreadyScreenedFormulas;
}

const KNOWN_COMPOUNDS = new Set<string>([
  "MgB2", "NbTi", "Nb3Sn", "Nb3Ge", "Nb3Al", "NbN", "NbC", "V3Si", "V3Ga",
  "YBa2Cu3O7", "Bi2Sr2CaCu2O8", "Bi2Sr2Ca2Cu3O10", "Tl2Ba2CaCu2O8",
  "HgBa2CaCu2O6", "HgBa2Ca2Cu3O8", "La2CuO4", "LaFeAsO", "BaFe2As2",
  "FeSe", "LiFeAs", "NaFeAs", "SrFe2As2", "CaFe2As2",
  "LaH10", "YH6", "YH9", "CeH9", "CeH10", "ThH10", "ThH9", "PrH9",
  "CaH6", "ScH9", "LaBeH8", "BaH12",
  "H3S", "SH3", "PH3", "AsH3", "GeH4", "SiH4", "SnH4",
  "TiH2", "TiH3", "VH2", "CrH", "MnH", "FeH", "CoH", "NiH", "CuH",
  "ZrH2", "ZrH3", "NbH", "NbH2", "MoH", "PdH", "AgH",
  "HfH2", "TaH", "WH", "PtH", "AuH",
  "YH2", "YH3", "LaH2", "LaH3", "LaH4", "LaH5",
  "CeH2", "CeH3", "PrH2", "PrH3", "NdH2", "NdH3", "GdH2", "GdH3",
  "ScH2", "ScH3", "TiH", "VH", "CrH2",
  "MgH2", "CaH2", "SrH2", "BaH2", "LiH", "NaH", "KH", "RbH", "CsH",
  "AlH3", "GaH3", "BeH2",
  "TiC", "ZrC", "HfC", "VC", "NbC2", "TaC", "WC", "MoC", "Mo2C",
  "TiN", "ZrN", "HfN", "VN", "NbN2", "TaN", "WN", "MoN", "CrN",
  "TiB2", "ZrB2", "HfB2", "VB2", "NbB2", "TaB2", "MoB2", "WB2", "CrB2",
  "MgB4", "AlB2", "CaB6", "LaB6", "YB6",
  "PbTe", "SnTe", "GeTe", "Bi2Te3", "Bi2Se3", "Sb2Te3",
  "ZrTe5", "HfTe5", "WTe2", "MoTe2", "MoS2", "WS2", "NbSe2", "TaSe2",
  "Fe2O3", "TiO2", "SrTiO3", "BaTiO3", "LaAlO3", "LaNiO3",
  "PbMo6S8", "Pb", "Nb", "V", "Ta", "Hg", "Sn", "In", "Al", "Ti",
]);

const FAMILY_CAPS: Record<string, number> = {
  Hydrides: 0.15,
  Carbides: 0.12,
  Nitrides: 0.12,
  Borides: 0.12,
  Chalcogenides: 0.12,
  Oxides: 0.12,
  Sulfides: 0.10,
  Pnictides: 0.12,
  Intermetallics: 0.12,
  Kagome: 0.10,
  Alloys: 0.12,
  Cuprates: 0.12,
  Nickelates: 0.10,
  Borocarbides: 0.08,
  Clathrates: 0.08,
};
let lastActiveLearningCycle = 0;
let recentTcImproved = false;
let recentNewCandidates = 0;
let failuresSinceLastRetrain = 0;
let lastRetrainCycle = 0;
let integratedPipelineId: string | null = null;
let integratedLabId: string | null = null;
let lastTheoryDiscoveryCycle = 0;
let lastCausalDiscoveryCycle = 0;
let causalDesignGuidance: { variable: string; direction: string; causalImpactOnTc: number }[] = [];
let theoryFeedbackBias: { biasedVariables: string[]; biasedElements: string[] } = { biasedVariables: [], biasedElements: [] };
let lastTheoryBiasCycle = 0;
let latestTheoryBias: TheoryGeneratorBias | null = null;

let feedbackLoopStats = {
  defectCandidatesAdded: 0,
  defectTotalTcBoost: 0,
  correlationBoostsApplied: 0,
  correlationTotalTcBoost: 0,
  synthesisFeasibilityBonuses: 0,
  synthesisTotalFeasibilityBoost: 0,
  growthQualityBonuses: 0,
  growthTotalQualityBoost: 0,
  experimentPlansGenerated: 0,
  experimentDFTPrioritized: 0,
  pressurePathwayBoosts: 0,
  pressurePathwayBestAmbientTc: 0,
  pressurePathwayBestFormula: "",
};

interface PreviousCycleMetrics {
  bestTc: number;
  bestScore: number;
  candidateCount: number;
  familyDiversity: number;
  insightCount: number;
  topFamily: string;
  pipelinePassed: number;
  pipelineTotal: number;
}
let previousCycleMetrics: PreviousCycleMetrics | null = null;
let cycleInsightsThisCycle = 0;

type ThoughtCategory = "strategy" | "discovery" | "stagnation" | "milestone";

function broadcastThought(text: string, category: ThoughtCategory) {
  broadcast("thought", { text, category });
}

function updateTempo() {
  const prevTempo = engineTempo;
  if (recentTcImproved || recentNewCandidates >= 3) {
    engineTempo = "excited";
    cycleIntervalMs = 10000;
  } else if (cyclesSinceTcImproved > 10) {
    engineTempo = "contemplating";
    cycleIntervalMs = 22000;
  } else {
    engineTempo = "exploring";
    cycleIntervalMs = 15000;
  }
  if (prevTempo !== engineTempo) {
    broadcast("tempoChange", { tempo: engineTempo, intervalMs: cycleIntervalMs });
  }
}

function generateStatusMessage(): string {
  if (state === "stopped") return "Engine offline";
  if (state === "paused") return "Research paused";

  const topFocus = currentStrategyFocusAreas[0]?.area || "promising material";
  const tasks = Array.from(activeTasks);

  if (engineTempo === "excited") {
    return `Actively pursuing ${topFocus.toLowerCase()} candidates`;
  }
  if (engineTempo === "contemplating") {
    if (cyclesSinceTcImproved > 15) return "Deep analysis mode — reconsidering approach";
    return "Re-analyzing top candidates with stricter physics";
  }
  if (tasks.includes("SC Research")) return "Screening superconductor candidates";
  if (tasks.includes("Computational Physics")) return "Running physics verification";
  if (tasks.includes("Data Fetching")) return "Scanning scientific databases for new materials";
  return `Exploring ${topFocus.toLowerCase()} chemical space`;
}

function broadcast(type: string, data: any) {
  if (!wss) return;
  const msg = JSON.stringify({ type, data, timestamp: new Date().toISOString() });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

const recentLogCache = new Set<string>();
const RECENT_LOG_CACHE_MAX = 100;

const DEDUP_EVENT_PATTERNS = [
  "started", "discovery started", "fetch started", "import started",
  "analysis started", "Prediction patterns discovered",
  "All top candidates have crystal structures",
];

const emit: EventEmitter = (type: string, data: any) => {
  if (type === "log" && data.event && data.phase) {
    const evt = data.event as string;
    const shouldDedup = DEDUP_EVENT_PATTERNS.some(p => evt.includes(p));
    if (shouldDedup) {
      const cacheKey = `${evt}::${data.detail || ""}`;
      if (recentLogCache.has(cacheKey)) {
        return;
      }
      recentLogCache.add(cacheKey);
      if (recentLogCache.size > RECENT_LOG_CACHE_MAX) {
        const first = recentLogCache.values().next().value;
        if (first !== undefined) recentLogCache.delete(first);
      }
    }
  }

  broadcast(type, data);

  if (type === "log" && data.event && data.phase) {
    storage.insertResearchLog({
      phase: data.phase,
      event: data.event,
      detail: data.detail || null,
      dataSource: data.dataSource || null,
    }).catch(() => {});
    if (data.event === "Novel insight discovered") {
      const detail = data.detail || "";
      const insightText = detail.replace(/^\[NOVEL \d+%\]\s*/, "");
      if (requiresQuantitativeContent(insightText)) {
        cycleInsightsThisCycle++;
      }
    }
  }
};

const PHASE_BASE_TARGETS: Record<number, number> = {
  1: 47,
  2: 118,
  3: 50,
  4: 500,
  5: 50,
  6: 200,
  7: 500,
  8: 300,
  9: 300,
  10: 200,
  11: 150,
  12: 300,
};

function dynamicTarget(phaseId: number, itemsLearned: number): number {
  const base = PHASE_BASE_TARGETS[phaseId] ?? 100;
  if (itemsLearned <= base) return base;
  return Math.ceil(itemsLearned * 1.1);
}

function computeProgress(phaseId: number, itemsLearned: number): number {
  const target = dynamicTarget(phaseId, itemsLearned);
  return Math.min(99, Math.floor((itemsLearned / target) * 100));
}

async function updatePhaseStatus(phaseId: number, status: string, progress: number, itemsLearned: number, totalItems?: number) {
  try {
    const phase = await storage.getLearningPhaseById(phaseId);
    if (!phase) return;

    const newProgress = Math.min(100, progress);
    const resolvedTotal = totalItems ?? dynamicTarget(phaseId, itemsLearned);
    const updates: any = { progress: newProgress, itemsLearned, totalItems: resolvedTotal };
    if (status === "active" && phase.status !== "active") {
      updates.startedAt = new Date();
    }
    if (status === "completed") {
      updates.completedAt = new Date();
    }
    updates.status = status;

    await storage.upsertLearningPhase({
      ...phase,
      ...updates,
    });

    broadcast("phaseUpdate", { phaseId, status, progress: newProgress, itemsLearned, totalItems: resolvedTotal });
  } catch (e) {
    console.error("updatePhaseStatus failed:", e);
  }
}

async function addInsightsToPhase(phaseId: number, newInsights: string[]) {
  if (newInsights.length === 0) return;
  try {
    const phase = await storage.getLearningPhaseById(phaseId);
    if (!phase) return;
    const existing = phase.insights ?? [];
    const combined = [...existing, ...newInsights.map(s => sanitizeForbiddenWords(s))].slice(-20);
    await storage.upsertLearningPhase({
      ...phase,
      insights: combined,
    });
  } catch (e) {
    console.error("addInsightsToPhase failed:", e);
  }
}

async function runPhase3_Bonding() {
  if (!shouldContinue()) return;
  activeTasks.add("Bonding Analysis");
  broadcast("taskStart", { task: "Bonding Analysis" });
  try {
    const mats = await storage.getMaterials(200, 0);
    if (mats.length === 0) return;

    await updatePhaseStatus(3, "active", 0, 0);
    if (!shouldContinue()) return;
    const insights = await analyzeBondingPatterns(emit, mats);
    allInsights.push(...insights);
    totalInsightsGenerated += insights.length;

    await addInsightsToPhase(3, insights);
    const formulas = mats.slice(0, 5).map(m => m.formula);
    await evaluateInsightNovelty(emit, insights, 3, "Chemical Bonding", formulas);
    const phase3 = await storage.getLearningPhaseById(3);
    const totalBondingInsights = (phase3?.insights ?? []).length;
    const progress = computeProgress(3, totalBondingInsights);
    await updatePhaseStatus(3, "active", progress, totalBondingInsights);
  } finally {
    activeTasks.delete("Bonding Analysis");
    broadcast("taskEnd", { task: "Bonding Analysis" });
  }
}

async function runPhase4_Materials() {
  if (!shouldContinue()) return;
  activeTasks.add("Data Fetching");
  broadcast("taskStart", { task: "Data Fetching" });
  try {
    await updatePhaseStatus(4, "active", 0, 0);

    const [oqmdCount, elementCount, knownCount] = await Promise.all([
      fetchOQMDMaterials(emit, 10, getNextOQMDOffset()),
      fetchElementFocusedMaterials(emit),
      fetchKnownMaterials(emit),
    ]);

    const total = oqmdCount + elementCount + knownCount;
    totalMaterialsFetched += total;

    const matCount = await storage.getMaterialCount();
    const progress = computeProgress(4, matCount);
    await updatePhaseStatus(4, "active", progress, matCount);
  } finally {
    activeTasks.delete("Data Fetching");
    broadcast("taskEnd", { task: "Data Fetching" });
  }
}

async function runPhase5_Prediction() {
  if (!shouldContinue()) return;
  activeTasks.add("Property Prediction");
  broadcast("taskStart", { task: "Property Prediction" });
  try {
    const mats = await storage.getMaterials(200, 0);
    if (mats.length === 0) return;

    await updatePhaseStatus(5, "active", 0, 0);
    if (!shouldContinue()) return;
    const insights = await analyzePropertyPredictionPatterns(emit, mats);
    allInsights.push(...insights);
    totalInsightsGenerated += insights.length;

    await addInsightsToPhase(5, insights);
    const predFormulas = mats.slice(0, 5).map(m => m.formula);
    await evaluateInsightNovelty(emit, insights, 5, "Property Prediction", predFormulas);
    const phase5 = await storage.getLearningPhaseById(5);
    const totalPredInsights = (phase5?.insights ?? []).length;
    const crCount5 = await storage.getComputationalResultCount();
    const scCount5 = await storage.getSuperconductorCount();
    const predictionWork = totalPredInsights + crCount5 + scCount5;
    const progress5 = computeProgress(5, predictionWork);
    await updatePhaseStatus(5, "active", progress5, predictionWork);
  } finally {
    activeTasks.delete("Property Prediction");
    broadcast("taskEnd", { task: "Property Prediction" });
  }
}

async function runPhase6_Discovery() {
  if (!shouldContinue()) return;
  activeTasks.add("Novel Discovery");
  broadcast("taskStart", { task: "Novel Discovery" });
  try {
    const prevPredCount = (await storage.getNovelPredictions()).length;
    const prevProgress = computeProgress(6, prevPredCount + totalPredictionsMade);
    await updatePhaseStatus(6, "active", prevProgress, prevPredCount + totalPredictionsMade);
    if (!shouldContinue()) return;

    const generated = await generateNovelFormulas(emit, allInsights.slice(-10), undefined, currentStrategyHint || undefined);
    totalPredictionsMade += generated;

    const predCount = (await storage.getNovelPredictions()).length;
    const scCount = await storage.getSuperconductorCount();
    const discoveryWork = predCount + scCount;
    const progress6 = computeProgress(6, discoveryWork);
    await updatePhaseStatus(6, "active", progress6, discoveryWork);
  } finally {
    activeTasks.delete("Novel Discovery");
    broadcast("taskEnd", { task: "Novel Discovery" });
  }
}

async function runPhase7_Superconductor() {
  if (!shouldContinue()) return;
  activeTasks.add("SC Research");
  broadcast("taskStart", { task: "SC Research" });
  try {
    await updatePhaseStatus(7, "active", 0, 0);
    if (!shouldContinue()) return;

    const matTotal = await storage.getMaterialCount();
    if (matTotal > 0) {
      phase7Offset = phase7Offset % matTotal;
    }
    const result = await runSuperconductorResearch(emit, allInsights.slice(-15), phase7Offset, {
      strategyFocusAreas: currentStrategyFocusAreas.length > 0 ? currentStrategyFocusAreas : undefined,
      familyCounts: Object.keys(currentFamilyCounts).length > 0 ? currentFamilyCounts : undefined,
      stagnationInfo: lastBestTcSeen > 0 ? { cyclesSinceImproved: cyclesSinceTcImproved, currentBestTc: lastBestTcSeen } : undefined,
    });
    phase7Offset += 200;
    totalScCandidates += result.generated;
    if (result.duplicatesSkipped > 0) {
      trackDuplicatesSkipped(result.duplicatesSkipped);
    }
    allInsights.push(...result.insights);
    totalInsightsGenerated += result.insights.length;

    await addInsightsToPhase(7, result.insights);
    await evaluateInsightNovelty(emit, result.insights, 7, "Superconductor Research");

    if (cycleCount % 5 === 0 && shouldContinue()) {
      try {
        const inverseDesigned = await generateInverseDesignCandidates(emit, allInsights);
        totalScCandidates += inverseDesigned;
        if (inverseDesigned > 0) {
          recordGeneratorOutcome("inverse_design", true, 0, 0.5);
        }
      } catch (err: any) {
        emit("log", { phase: "phase-7", event: "Inverse design error", detail: err.message?.slice(0, 150), dataSource: "Inverse Design" });
      }
    }

    if (cycleCount % 3 === 0 && shouldContinue()) {
      try {
        const activeCampaigns = getAllActiveCampaigns();
        for (const campaign of activeCampaigns) {
          const inverseCandidates = runInverseCycle(campaign);
          if (inverseCandidates.length === 0) continue;

          emit("log", {
            phase: "inverse-optimizer",
            event: `Inverse design cycle ${campaign.cyclesRun}`,
            detail: `Campaign ${campaign.id}: ${inverseCandidates.length} candidates, target Tc=${campaign.target.targetTc}K`,
          });

          const inverseResults: { formula: string; tc: number; lambda: number; hull: number; pressure: number; passedPipeline: boolean }[] = [];

          for (const ic of inverseCandidates) {
            try {
              const features = extractFeatures(ic.formula);
              if (!features) continue;
              const gbResult = gbPredict(features);
              if (!gbResult || gbResult.tcPredicted < 3) continue;

              const candidateId = `inv-cand-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
              const passedStability = await insertCandidateWithStabilityCheck({
                id: candidateId,
                name: `Inverse-${ic.formula}`,
                formula: ic.formula,
                predictedTc: gbResult.tcPredicted,
                status: "theoretical",
                xgboostScore: gbResult.score,
                mlFeatures: features as any,
                notes: `Inverse design campaign ${campaign.id}, target Tc=${campaign.target.targetTc}K`,
                electronPhononCoupling: features.electronPhononLambda,
                crystalStructure: ic.prototype ?? null,
              }, "inverse_design");

              inverseResults.push({
                formula: ic.formula,
                tc: gbResult.tcPredicted,
                lambda: features.electronPhononLambda,
                hull: features.formationEnergy !== null ? Math.abs(features.formationEnergy) * 0.1 : 0.05,
                pressure: features.pressureGpa ?? 0,
                passedPipeline: passedStability,
              });

              if (passedStability) totalScCandidates++;
            } catch (invErr: any) {
              console.log(`[Engine] Inverse design candidate insert failed: ${invErr?.message?.slice(0, 100) ?? "unknown"}`);
            }
          }

          processInverseResults(campaign, inverseResults);

          try {
            const ls = campaign.learningState;
            if (ls.elementSuccessMatrix && ls.elementSuccessMatrix.size > 0) {
              for (const [el, stats] of ls.elementSuccessMatrix) {
                if (stats.count >= 2 && stats.totalReward > 0) {
                  const avgReward = stats.totalReward / stats.count;
                  const syntheticTc = avgReward * campaign.target.targetTc;
                  rlAgent.recordElementOutcome([el], syntheticTc, syntheticTc > 20);
                }
              }
            }
            for (const r of inverseResults) {
              if (r.tc > 3 && r.passedPipeline) {
                bayesianOptimizer.addObservation(r.formula, r.tc, r.lambda, r.passedPipeline ? 1 : 0);
                incorporateSuccessData(r.formula, r.tc).catch(() => {});
              }
            }
          } catch (crossPollErr) {
            console.error(`[Engine] Inverse->RL/BO cross-pollination failed:`, crossPollErr instanceof Error ? crossPollErr.message.slice(0, 100) : "unknown");
          }

          {
            const serializable = getSerializableCampaignState(campaign);
            await storage.updateInverseDesignCampaign(campaign.id, {
              cyclesRun: campaign.cyclesRun,
              bestTcAchieved: campaign.bestTcAchieved,
              bestDistance: campaign.learningState.bestDistance,
              candidatesGenerated: campaign.candidatesGenerated,
              candidatesPassedPipeline: campaign.candidatesPassedPipeline,
              status: campaign.status,
              learningState: serializable.learningState,
              convergenceHistory: serializable.convergenceHistory,
              topCandidates: serializable.topCandidates,
            });

            emit("log", {
              phase: "inverse-optimizer",
              event: `Inverse results processed`,
              detail: `Campaign ${campaign.id}: ${inverseResults.length} evaluated, ${inverseResults.filter(r => r.passedPipeline).length} passed, best distance=${campaign.learningState.bestDistance.toFixed(3)}, best Tc=${campaign.bestTcAchieved.toFixed(1)}K`,
            });
          }
        }
      } catch (err: any) {
        emit("log", { phase: "inverse-optimizer", event: "Inverse optimizer error", detail: err.message?.slice(0, 200) });
      }
    }

    if (cycleCount % 12 === 0 && shouldContinue()) {
      try {
        const topExistingForGrad = await storage.getSuperconductorCandidatesByTc(8);
        const topFormulasForGrad = topExistingForGrad.map(c => c.formula);
        const activeCampaignList = getAllActiveCampaigns();
        for (const campaign of activeCampaignList) {
          const gradResult = runGradientDescentCycle(campaign.target, 4, 12, topFormulasForGrad);
          if (gradResult.bestTc > 10) {
            for (const r of gradResult.results) {
              if (r.finalTc > 10) {
                try {
                  const features = extractFeatures(r.finalFormula);
                  if (!features) continue;
                  const gb = gbPredict(features);
                  if (gb.tcPredicted >= 10) {
                    const gdInserted = await insertCandidateWithStabilityCheck({
                      formula: normalizeFormula(r.finalFormula),
                      predictedTc: Math.round(gb.tcPredicted),
                      dataConfidence: "low",
                      ensembleScore: Math.min(0.9, gb.score),
                      verificationStage: 0,
                      notes: `[gradient-descent: ${r.totalSteps} steps, ${r.initialFormula}->${r.finalFormula}, improvement=${r.improvementRatio}]`,
                    }, "bo_exploration");
                    if (gdInserted) totalScCandidates++;
                  }
                } catch (gdErr: any) {
                  console.log(`[Engine] Gradient descent candidate insert failed: ${gdErr?.message?.slice(0, 100) ?? "unknown"}`);
                }
              }
            }
            emit("log", {
              phase: "gradient-optimizer",
              event: `Gradient descent cycle`,
              detail: `Campaign ${campaign.id}: best=${gradResult.bestFormula} Tc=${gradResult.bestTc.toFixed(1)}K from ${gradResult.results.length} seeds`,
            });
          }
        }
      } catch (err: any) {
        emit("log", { phase: "gradient-optimizer", event: "Gradient optimizer error", detail: err.message?.slice(0, 200) });
      }
    }

    if (cycleCount % 9 === 0 && shouldContinue()) {
      try {
        const topExisting = await storage.getSuperconductorCandidatesByTc(10);
        const existingFormulas = topExisting.map(c => c.formula);
        const targetTc = autonomousBestTc > 100 ? autonomousBestTc * 1.5 : 200;
        const pillarResult = runPillarCycle(existingFormulas, targetTc);
        let pillarInserted = 0;

        for (const formula of pillarResult.formulas) {
          if (!isValidFormula(formula)) continue;
          const normalized = normalizeFormula(formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          try {
            const features = extractFeatures(normalized);
            const gb = gbPredict(features);
            if (gb.tcPredicted >= 8) {
              const eval5 = pillarResult.evaluations.find(e => e.formula === formula);
              const inserted = await insertCandidateWithStabilityCheck({
                formula: normalized,
                predictedTc: Math.round(gb.tcPredicted),
                dataConfidence: "low",
                ensembleScore: Math.min(0.9, gb.score),
                verificationStage: 0,
                notes: `[5-pillar: fitness=${eval5?.compositeFitness.toFixed(2) ?? "?"}, pillars=${eval5?.satisfiedPillars ?? "?"}/5, motif=${eval5?.motifMatch ?? "?"}, weak=${eval5?.weakestPillar ?? "?"}]`,
              }, "bo_exploration");
              if (inserted) {
                totalScCandidates++;
                pillarInserted++;
                if (eval5) {
                  updatePillarWeightsFromReward(gb.tcPredicted, eval5);
                }
              }
            }
          } catch (pillarErr: any) {
            console.log(`[Engine] Pillar search failed: ${pillarErr?.message?.slice(0, 100) ?? "unknown"}`);
          }
        }

        if (pillarResult.formulas.length > 0) {
          emit("log", {
            phase: "pillar-optimizer",
            event: "5-pillar guided generation",
            detail: `Generated ${pillarResult.evaluations.length} candidates, ${pillarResult.formulas.length} passed fitness threshold, inserted ${pillarInserted}. Best: ${pillarResult.bestFormula} (fitness=${pillarResult.bestFitness.toFixed(2)}, Tc=${pillarResult.bestTc.toFixed(1)}K)`,
            dataSource: "SC Pillars Optimizer",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "pillar-optimizer", event: "5-pillar optimizer error", detail: err.message?.slice(0, 200) });
      }
    }

    const scCount = await storage.getSuperconductorCount();
    const progress7 = computeProgress(7, scCount);
    await updatePhaseStatus(7, "active", progress7, scCount);
  } finally {
    activeTasks.delete("SC Research");
    broadcast("taskEnd", { task: "SC Research" });
  }
}

async function runPhase8_Synthesis() {
  if (!shouldContinue()) return;
  activeTasks.add("Synthesis Mapping");
  broadcast("taskStart", { task: "Synthesis Mapping" });
  try {
    await updatePhaseStatus(8, "active", 0, 0);
    if (!shouldContinue()) return;

    const mats = await storage.getMaterials(15, 0);
    const discovered = await discoverSynthesisProcesses(emit, mats);
    totalSynthesisDiscovered += discovered;

    const synthCount = await storage.getSynthesisCount();
    const progress8 = computeProgress(8, synthCount);
    await updatePhaseStatus(8, "active", progress8, synthCount);
  } finally {
    activeTasks.delete("Synthesis Mapping");
    broadcast("taskEnd", { task: "Synthesis Mapping" });
  }
}

async function runPhase9_Reactions() {
  if (!shouldContinue()) return;
  activeTasks.add("Reaction Discovery");
  broadcast("taskStart", { task: "Reaction Discovery" });
  try {
    await updatePhaseStatus(9, "active", 0, 0);
    if (!shouldContinue()) return;

    const topic = getNextReactionTopic();
    const discovered = await discoverChemicalReactions(emit, topic);
    totalReactionsLearned += discovered;

    const rxnCount = await storage.getReactionCount();
    const progress9 = computeProgress(9, rxnCount);
    await updatePhaseStatus(9, "active", progress9, rxnCount);
  } finally {
    activeTasks.delete("Reaction Discovery");
    broadcast("taskEnd", { task: "Reaction Discovery" });
  }
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

async function insertCandidateWithStabilityCheck(candidateData: Parameters<typeof storage.insertSuperconductorCandidate>[0], generatorSource?: string): Promise<boolean> {
  try {
    if (!passesElementCountCap(candidateData.formula)) {
      emit("log", {
        phase: "engine",
        event: "Composition cap rejected",
        detail: `${candidateData.formula}: exceeds element/atom count caps (max 4 non-H elements, 5 total, 20 atoms, 12 per element)`,
        dataSource: "Composition Filter",
      });
      if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
      return false;
    }

    const chemFilter = passesChemistryFilter(candidateData.formula);
    if (!chemFilter.pass) {
      emit("log", {
        phase: "engine",
        event: "Chemistry filter rejected",
        detail: `${candidateData.formula}: ${chemFilter.reason}`,
        dataSource: "Chemistry Validity Filter",
      });
      if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
      return false;
    }

    const preFilter = passesStabilityPreFilter(candidateData.formula);
    if (!preFilter.pass) {
      emit("log", {
        phase: "engine",
        event: "Stability pre-filter rejected",
        detail: `Fast stability screen rejected ${candidateData.formula}: ${preFilter.reason}`,
        dataSource: "Stability Predictor (GNN)",
      });
      if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
      return false;
    }

    const stabilityResult = await passesStabilityGate(candidateData.formula);

    if (!stabilityResult.pass) {
      emit("log", {
        phase: "engine",
        event: "Stability gate rejected",
        detail: `Stability gate rejected ${candidateData.formula}: ${stabilityResult.reason}`,
        dataSource: "Stability Gate",
      });
      if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
      return false;
    }

    let kineticResult: KineticStabilityResult | null = null;
    try {
      if (stabilityResult.verdict === "metastable" || stabilityResult.hullDistance > 0.005) {
        kineticResult = predictKineticStability(candidateData.formula, stabilityResult.hullDistance);

        if (kineticResult.kineticScore < 0.15 && kineticResult.stabilizationStrategies.length === 0) {
          emit("log", {
            phase: "engine",
            event: "Kinetic stability rejected",
            detail: `${candidateData.formula}: kineticScore=${kineticResult.kineticScore} too low, lifetime=${kineticResult.lifetimeString}, no stabilization strategies`,
            dataSource: "Kinetic Stability Engine",
          });
          if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
          return false;
        }

        emit("log", {
          phase: "engine",
          event: "Kinetic stability assessed",
          detail: `${candidateData.formula}: kineticScore=${kineticResult.kineticScore}, lifetime=${kineticResult.lifetimeString}, strategies=${kineticResult.stabilizationStrategies.length}, ambientStable=${kineticResult.pressureStabilization.ambientStabilizable}`,
          dataSource: "Kinetic Stability Engine",
        });
      }
    } catch (kineticErr: any) {
      console.log(`[Engine] Kinetic stability check failed: ${kineticErr?.message?.slice(0, 80) ?? "unknown"}`);
    }

    let synthesisGateResult;
    try {
      synthesisGateResult = evaluateSynthesisGate(candidateData.formula);
      if (!synthesisGateResult.pass) {
        emit("log", {
          phase: "engine",
          event: "Synthesis gate rejected",
          detail: `${candidateData.formula}: compositeScore=${synthesisGateResult.compositeScore.toFixed(3)}, mlFeas=${synthesisGateResult.mlFeasibility.toFixed(3)}, chemDist=${synthesisGateResult.chemicalDistance.totalDistance.toFixed(3)}, class=${synthesisGateResult.classification}. ${synthesisGateResult.rejectionReasons.join("; ")}`,
          dataSource: "Synthesis Gate",
        });
        if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
        return false;
      }
      if (synthesisGateResult.deprioritize) {
        emit("log", {
          phase: "engine",
          event: "Synthesis gate deprioritized",
          detail: `${candidateData.formula}: compositeScore=${synthesisGateResult.compositeScore.toFixed(3)}, class=${synthesisGateResult.classification} — low synthesis feasibility, will be deprioritized`,
          dataSource: "Synthesis Gate",
        });
      }
    } catch (synthGateErr: any) {
      console.log(`[Engine] Synthesis gate check failed: ${synthGateErr?.message?.slice(0, 80) ?? "unknown"}`);
    }

    const existingMlFeatures = (candidateData.mlFeatures as Record<string, any>) ?? {};
    const enrichedMlFeatures = {
      ...existingMlFeatures,
      stabilityGate: {
        hullDistance: stabilityResult.hullDistance,
        formationEnergy: stabilityResult.formationEnergy,
        verdict: stabilityResult.verdict,
        kineticBarrier: stabilityResult.kineticBarrier,
      },
      ...(kineticResult ? {
        kineticStability: {
          kineticScore: kineticResult.kineticScore,
          lifetime: kineticResult.lifetimeString,
          diffusionBarrier: kineticResult.diffusionBarriers.effectiveBarrier,
          nucleationBarrier: kineticResult.nucleationBarrier.nucleationBarrier,
          gbEnergy: kineticResult.grainBoundary.averageGBEnergy,
          pressureGPa: kineticResult.pressureStabilization.minStabilizationPressure,
          ambientStabilizable: kineticResult.pressureStabilization.ambientStabilizable,
          strategies: kineticResult.stabilizationStrategies.map(s => s.strategy),
        },
      } : {}),
      ...(synthesisGateResult ? {
        synthesisGate: {
          compositeScore: synthesisGateResult.compositeScore,
          mlFeasibility: synthesisGateResult.mlFeasibility,
          chemicalDistance: synthesisGateResult.chemicalDistance.totalDistance,
          stepEstimate: synthesisGateResult.chemicalDistance.stepEstimate,
          precursorAvailability: synthesisGateResult.chemicalDistance.precursorAvailability,
          toxicElements: synthesisGateResult.chemicalDistance.toxicElements,
          classification: synthesisGateResult.classification,
          isOnePot: synthesisGateResult.chemicalDistance.isOnePot,
          deprioritized: synthesisGateResult.deprioritize,
        },
      } : {}),
    };

    let deliberationResult;
    try {
      deliberationResult = await deliberateOnCandidate(
        candidateData.formula,
        candidateData.predictedTc ?? 0,
        enrichedMlFeatures
      );

      emit("log", {
        phase: "engine",
        event: "Deliberative evaluation",
        detail: formatDeliberationSummary(deliberationResult),
        dataSource: "Deliberative Evaluator",
      });

      if (deliberationResult.verdict === "reject") {
        emit("log", {
          phase: "engine",
          event: "Deliberation rejected",
          detail: `${candidateData.formula}: verdict=reject, score=${deliberationResult.deliberationScore.toFixed(3)}. Concerns: ${deliberationResult.selfCritiqueFlags.slice(0, 2).join("; ") || "low multi-stage scores"}`,
          dataSource: "Deliberative Evaluator",
        });
        if (generatorSource) recordGeneratorOutcome(generatorSource, false, 0, 0);
        return false;
      }
    } catch (deliberationErr: any) {
      console.log(`[Engine] Deliberation failed for ${candidateData.formula}: ${deliberationErr?.message?.slice(0, 80) ?? "unknown"}`);
    }

    const existingNotes = candidateData.notes || "";
    const stabilityNote = `[Stability: ${stabilityResult.verdict}, hullDist=${stabilityResult.hullDistance.toFixed(4)} eV/atom, formE=${stabilityResult.formationEnergy.toFixed(4)} eV/atom]`;
    const kineticNote = kineticResult ? ` ${formatKineticStabilityNote(kineticResult)}` : "";
    const deliberationNote = deliberationResult
      ? ` [Deliberation: score=${deliberationResult.deliberationScore.toFixed(3)}, verdict=${deliberationResult.verdict}, confidence=${deliberationResult.confidenceLevel.toFixed(2)}, rank=#${deliberationResult.comparativeRank ?? "?"}, novelty=${deliberationResult.estimatedNovelty.toFixed(2)}]`
      : "";

    const deliberationFeatures = deliberationResult ? {
      deliberation: {
        score: deliberationResult.deliberationScore,
        verdict: deliberationResult.verdict,
        confidence: deliberationResult.confidenceLevel,
        rank: deliberationResult.comparativeRank,
        novelty: deliberationResult.estimatedNovelty,
        stageScores: Object.fromEntries(deliberationResult.stages.map(s => [s.name, { score: s.score, verdict: s.verdict }])),
        critiqueFlags: deliberationResult.selfCritiqueFlags.slice(0, 5),
        reasoning: deliberationResult.reasoningChain.slice(0, 30),
      },
    } : {};

    const enforcedPressure = enforcePhysicsPressure(candidateData.formula, candidateData.pressureGpa);
    const pressureViability = classifyPressureViability(enforcedPressure);
    const pressureNote = pressureViability.penalty > 0
      ? ` [Pressure: ${pressureViability.label}, ${enforcedPressure} GPa, viability penalty=${pressureViability.penalty.toFixed(2)}]`
      : "";

    const adjustedEnsemble = candidateData.ensembleScore != null
      ? Math.max(0, (candidateData.ensembleScore ?? 0) * (1 - pressureViability.penalty))
      : undefined;

    const isHighPressureOnly = enforcedPressure > 50;

    await storage.insertSuperconductorCandidate({
      ...candidateData,
      pressureGpa: enforcedPressure,
      ensembleScore: adjustedEnsemble,
      ambientPressureStable: isHighPressureOnly ? false : candidateData.ambientPressureStable,
      roomTempViable: isHighPressureOnly ? false : candidateData.roomTempViable,
      mlFeatures: { ...enrichedMlFeatures, ...deliberationFeatures, pressureViability: pressureViability.label } as any,
      notes: `${existingNotes} ${stabilityNote}${kineticNote}${deliberationNote}${pressureNote}`.trim(),
    });

    const candidateTc = candidateData.predictedTc ?? 0;
    if (generatorSource) {
      recordGeneratorOutcome(generatorSource, true, candidateTc, 0.5);
    }
    if (candidateTc > 0) {
      incorporateSuccessData(candidateData.formula, candidateTc).catch(() => {});
    }

    try {
      const lambda = candidateData.electronPhononCoupling ?? 0;
      const mlFeats = candidateData.mlFeatures as Record<string, any> | undefined;
      const prototype = mlFeats?.prototype ?? mlFeats?.predictedPrototype ?? null;
      const system = mlFeats?.crystalSystem ?? candidateData.crystalStructure ?? null;
      const sg = mlFeats?.spacegroupSymbol ?? null;
      recordStructureOutcome(candidateData.formula, prototype, system, sg, lambda, candidateTc, true);
    } catch {}

    return true;
  } catch (err: any) {
    console.log(`[Engine] insertCandidateWithStabilityCheck failed for ${candidateData.formula}: ${err?.message?.slice(0, 120) ?? "unknown"}`);
    return false;
  }
}

const reEvalApplied = new Map<string, { lambda: number; omegaLog: number; muStar: number; hasCrystal: boolean }>();
let cyclesSinceTcImproved = 0;
let lastBestTcSeen = 0;
let lastBestPairingSusc = 0;
let explorationModeActive = false;
let explorationModeSavedConstraints: { allowBeyondEmpirical: boolean; empiricalPenaltyStrength: number } | null = null;

function computeEliashbergTc(lambda: number, omegaLog: number, muStar: number): number {
  if (lambda < 0.05 || omegaLog <= 0) return 0;
  const omegaLogK = omegaLog * 1.4388;
  const denom = lambda - muStar * (1 + 0.62 * lambda);
  if (denom <= 1e-6) return 0;
  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);
  const exponent = -1.04 * (1 + lambda) / denom;
  if (exponent > 50) return 0;
  const tc = (omegaLogK / 1.2) * f1 * Math.exp(exponent);
  if (!Number.isFinite(tc) || tc < 0) return 0;
  return Math.round(tc);
}

function estimateRawTc(lambdaML: number, logPhononFreq: number | null | undefined, muStar?: number): number {
  const freq = logPhononFreq ?? 200;
  const mu = muStar ?? 0.1;
  const lambda = Math.max(0.001, lambdaML);

  const tc = allenDynesTcRaw(lambda, freq, mu);
  return Math.max(0, Math.round(tc));
}

async function reEvaluateTopCandidates() {
  try {
    const topByTc = await storage.getSuperconductorCandidatesByTc(50);

    const currentBestTc = Math.max(...topByTc.map(c => c.predictedTc ?? 0), 0);
    const currentBestPairing = Math.max(...topByTc.map(c => {
      const lambda = c.electronPhononCoupling ?? 0;
      const score = c.ensembleScore ?? 0;
      return lambda * 0.4 + score * 0.6;
    }), 0);

    if (currentBestTc > lastBestTcSeen + 2 || currentBestPairing > lastBestPairingSusc + 0.05) {
      cyclesSinceTcImproved = 0;
      lastBestTcSeen = Math.max(lastBestTcSeen, currentBestTc);
      lastBestPairingSusc = Math.max(lastBestPairingSusc, currentBestPairing);
    } else {
      cyclesSinceTcImproved++;
    }

    let updated = 0;
    for (const candidate of topByTc) {
      const lambda = candidate.electronPhononCoupling ?? 0;
      const omegaLog = candidate.logPhononFrequency ?? 0;
      const muStar = candidate.coulombPseudopotential ?? 0.12;

      const crystals = await storage.getCrystalStructuresByFormula(candidate.formula);
      const hasCrystal = crystals.some(c => c.synthesizability != null && c.synthesizability > 0.7);

      const prev = reEvalApplied.get(candidate.id);
      const inputsChanged = !prev ||
        (lambda > 0 && Math.abs(lambda - prev.lambda) > 0.05) ||
        (omegaLog > 0 && Math.abs(omegaLog - prev.omegaLog) > 5) ||
        (Math.abs(muStar - prev.muStar) > 0.01) ||
        (hasCrystal && !prev.hasCrystal);

      if (!inputsChanged) continue;

      reEvalApplied.set(candidate.id, { lambda, omegaLog, muStar, hasCrystal });

      let newTc = computeEliashbergTc(lambda, omegaLog, muStar);
      if (newTc <= 0) continue;

      const features = extractFeatures(candidate.formula);
      const capEvidence: CapExtensionEvidence = {
        eliashbergLambda: lambda > 0 ? lambda : undefined,
        eliashbergTc: newTc > 0 ? newTc : undefined,
        gnnEnsembleStd: (candidate.uncertaintyEstimate != null && candidate.uncertaintyEstimate > 0)
          ? candidate.uncertaintyEstimate * (candidate.predictedTc ?? 50)
          : undefined,
      };
      newTc = applyAmbientTcCap(newTc, lambda, candidate.pressureGpa ?? 0, features.metallicity ?? 0.5, candidate.formula, capEvidence);

      const reEvalFamily = classifyFamily(candidate.formula);
      if (reEvalFamily && FAMILY_TC_CAPS[reEvalFamily]) {
        const caps = FAMILY_TC_CAPS[reEvalFamily];
        const pressure = candidate.pressureGpa ?? 0;
        const pFactor = pressure > 50 ? 1.0 : pressure < 10 ? 0.0 : (pressure - 10) / 40;
        const extensionFactor = computeCapExtensionFactor(capEvidence);
        const familyCap = Math.round((caps.ambient + (caps.highPressure - caps.ambient) * pFactor) * extensionFactor);
        newTc = Math.min(newTc, familyCap);
      }

      const currentTc = candidate.predictedTc ?? 0;
      if (newTc === currentTc) continue;

      await storage.updateSuperconductorCandidate(candidate.id, { predictedTc: newTc });
      updated++;
      emit("log", {
        phase: "engine",
        event: "Tc recomputed from physics",
        detail: `${candidate.formula}: ${currentTc}K -> ${newTc}K (lambda=${lambda.toFixed(2)}, omegaLog=${omegaLog.toFixed(0)}cm-1, mu*=${muStar.toFixed(2)})`,
        dataSource: "Learning Feedback",
      });
    }

    if (updated > 0) {
      emit("log", {
        phase: "engine",
        event: "Re-evaluation complete",
        detail: `${updated}/${topByTc.length} candidates updated. Stagnation: ${cyclesSinceTcImproved} cycles. Best Tc: ${Math.round(lastBestTcSeen)}K`,
        dataSource: "Learning Feedback",
      });
    }
  } catch (err: any) {
    emit("log", { phase: "engine", event: "Re-evaluation error", detail: err.message?.slice(0, 150) ?? "unknown", dataSource: "Learning Feedback" });
  }
}

const dftEnrichmentTracker = new Map<string, number>();
const candidateGeneratorSource = new Map<string, string>();
const MAX_GENERATOR_SOURCE_ENTRIES = 2000;

async function runDFTEnrichment() {
  if (!shouldContinue()) return;
  try {
    const candidates = await storage.getSuperconductorCandidates(100);
    const sorted = candidates
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));

    const toEnrich: typeof candidates = [];
    for (const c of sorted) {
      if (toEnrich.length >= 12) break;
      if (c.dataConfidence === "high") continue;
      toEnrich.push(c);
    }

    const analyticalCandidates = candidates
      .filter(c => !c.dataConfidence || c.dataConfidence === "analytical")
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));
    for (const c of analyticalCandidates) {
      if (toEnrich.length >= 20) break;
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    const highScoreAnalytical = candidates
      .filter(c => (c.ensembleScore ?? 0) > 0.7 && (!c.dataConfidence || c.dataConfidence === "analytical"))
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));
    for (const c of highScoreAnalytical) {
      if (toEnrich.length >= 25) break;
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    const stage1Candidates = candidates
      .filter(c => (c.predictedTc ?? 0) > 40 && c.dataConfidence !== "high" && c.dataConfidence !== "medium")
      .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));
    for (const c of stage1Candidates) {
      if (toEnrich.length >= 30) break;
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    const currentCycle = cycleCount;
    const staleThreshold = 30;
    const staleMedium = candidates
      .filter(c => c.dataConfidence === "medium" && (currentCycle - (dftEnrichmentTracker.get(c.id) ?? 0)) > staleThreshold)
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0))
      .slice(0, 5);
    for (const c of staleMedium) {
      if (!toEnrich.some(e => e.id === c.id)) {
        toEnrich.push(c);
      }
    }

    if (toEnrich.length === 0) return;

    const totalCount = await storage.getSuperconductorCount();
    const highCount = candidates.filter(c => c.dataConfidence === "high").length;
    const medCount = candidates.filter(c => c.dataConfidence === "medium").length;
    const coveragePct = totalCount > 0 ? ((highCount + medCount) / totalCount * 100).toFixed(1) : "0";

    broadcastThought(
      `DFT coverage at ${coveragePct}% -- enriching next batch of ${toEnrich.length} candidates...`,
      "strategy"
    );

    let enriched = 0;
    for (const candidate of toEnrich) {
      if (!shouldContinue()) break;
      try {
        const dftData = await resolveDFTFeatures(candidate.formula);
        dftEnrichmentTracker.set(candidate.id, currentCycle);
        if (dftData.dftCoverage === 0) continue;

        const desc = describeDFTSources(dftData);
        emit("log", {
          phase: "engine",
          event: "DFT enrichment",
          detail: `${candidate.formula} -- found DFT data: ${desc}. Re-scoring...`,
          dataSource: "DFT Resolver",
        });

        const features = extractFeatures(candidate.formula, undefined, undefined, undefined, dftData);
        const gb = gbPredict(features);
        const nnScore = candidate.neuralNetScore ?? candidate.quantumCoherence ?? 0.3;
        const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

        const existingMl = (candidate.mlFeatures as Record<string, any>) ?? {};
        const updates: any = {
          xgboostScore: gb.score,
          ensembleScore: ensemble,
          dataConfidence: dftData.dftCoverage > 0.4 ? "high" : "medium",
          mlFeatures: { ...existingMl, dftConfidence: dftData.dftCoverage },
        };

        if (dftData.formationEnergy.source !== "analytical") {
          updates.formationEnergy = dftData.formationEnergy.value;
        } else if (candidate.formationEnergy == null) {
          try {
            updates.formationEnergy = computeMiedemaFormationEnergy(candidate.formula);
          } catch (e) { console.error("[Engine] Miedema formation energy failed:", e); }
        }
        if (dftData.bandGap.source !== "analytical") {
          updates.bandGap = dftData.bandGap.value;
        }

        const formEnergy = dftData.formationEnergy?.value ?? null;
        const dftStable = formEnergy !== null ? formEnergy < 0.5 : true;
        const dftSrc = (dftData.sources.mp || dftData.sources.aflow) ? "external" as const : "xtb" as const;
        incorporateDFTResult(candidate.formula, gb.tcPredicted, formEnergy, dftStable, dftSrc);

        const priorTc = candidate.predictedTc ?? 0;
        recordEvaluationResult(
          candidate.formula,
          { tc: priorTc, stable: true, formationEnergy: candidate.formationEnergy ?? 0 },
          { tc: gb.tcPredicted, stable: dftStable, formationEnergy: formEnergy },
          dftSrc === "external" ? "dft" : "xtb"
        );
        recordPredictionOutcome("xgboost", candidate.formula, priorTc, gb.tcPredicted);
        incorporateDFTFeedbackIntoPillars(candidate.formula, priorTc, gb.tcPredicted, dftStable);
        recordDFTFeedbackForGA(candidate.formula, { tc: gb.tcPredicted, stable: dftStable, formationEnergy: formEnergy }, candidate.materialClass ?? undefined);

        const sourceGen = candidateGeneratorSource.get(candidate.formula);
        if (sourceGen) {
          recordDFTOutcome(sourceGen, dftStable && gb.tcPredicted > 5, gb.tcPredicted);
        }

        if (dftData.phononStability) {
          const ps = dftData.phononStability;
          const lowestFreq = ps.lowestFrequency ?? 0;
          const imaCount = ps.imaginaryModeCount ?? 0;

          if (lowestFreq < -2000) {
            emit("log", {
              phase: "engine",
              event: "phonon artifact discarded",
              detail: `${candidate.formula}: xTB Hessian produced ${imaCount} imaginary mode(s) with lowest freq ${lowestFreq.toFixed(0)} cm-1 — values below -2000 cm-1 are xTB numerical artifacts, discarded. Phonon data unreliable for this structure.`,
              dataSource: "xTB-Hessian",
            });
            updates.dataConfidence = "low";
          } else if (ps.hasImaginaryModes && lowestFreq < -100) {
            const physicalCount = ps.frequencies?.filter((f: number) => f < -100).length ?? imaCount;
            const penalty = Math.min(0.25, physicalCount * 0.05);
            if (penalty > 0) {
              updates.ensembleScore = Math.max(0.05, (updates.ensembleScore ?? ensemble) - penalty);
            }
            if (physicalCount >= 5) {
              updates.dataConfidence = "low";
            }
            emit("log", {
              phase: "engine",
              event: "phonon instability",
              detail: `${candidate.formula}: ${physicalCount} physical imaginary mode(s) (lowest: ${lowestFreq.toFixed(0)} cm-1, threshold: -100 cm-1) — ensemble score penalized by ${penalty.toFixed(2)}`,
              dataSource: "xTB-Hessian",
            });
          } else if (imaCount > 0 && lowestFreq >= -100) {
            emit("log", {
              phase: "engine",
              event: "phonon mild instability",
              detail: `${candidate.formula}: ${imaCount} soft mode(s) (lowest: ${lowestFreq.toFixed(0)} cm-1) — within acoustic/soft-mode tolerance (-100 cm-1), no penalty`,
              dataSource: "xTB-Hessian",
            });
          }
        }

        await storage.updateSuperconductorCandidate(candidate.id, updates);
        enriched++;
        totalDFTEnriched++;
      } catch (err: any) {
        emit("log", {
          phase: "engine",
          event: "DFT enrichment error",
          detail: `${candidate.formula}: ${err.message?.slice(0, 100)}`,
          dataSource: "DFT Resolver",
        });
      }
    }

    if (enriched > 0) {
      emit("log", {
        phase: "engine",
        event: "DFT enrichment complete",
        detail: `Enriched ${enriched}/${toEnrich.length} candidates with DFT data (${totalDFTEnriched} total, coverage ~${coveragePct}%)`,
        dataSource: "DFT Resolver",
      });
    }

    try {
      const allCandidates = await storage.getSuperconductorCandidates(5000);
      const scoredCandidates = allCandidates
        .filter(c => (c.ensembleScore ?? 0) > 0)
        .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0));

      if (scoredCandidates.length >= 10) {
        const topIdx = Math.max(1, Math.floor(scoredCandidates.length * 0.001));
        const eliteCandidates = scoredCandidates.slice(0, topIdx);

        let submitted = 0;
        for (const elite of eliteCandidates) {
          const mlFeatures = (elite.mlFeatures as Record<string, any>) ?? {};
          if (mlFeatures.qeDFT) continue;

          try {
            const priority = Math.round((elite.ensembleScore ?? 0) * 100);
            await submitDFTJob(elite.formula, null, priority, "scf");
            submitted++;

            emit("log", {
              phase: "engine",
              event: "full DFT queued",
              detail: `${elite.formula} (top 0.1%, score=${(elite.ensembleScore ?? 0).toFixed(3)}, Tc=${(elite.predictedTc ?? 0).toFixed(1)}K) → Quantum ESPRESSO SCF+phonon queue`,
              dataSource: "QE-DFT Queue",
            });
          } catch (err: any) {
            emit("log", {
              phase: "engine",
              event: "full DFT queue error",
              detail: `Failed to queue ${elite.formula}: ${err.message?.slice(0, 100)}`,
              dataSource: "QE-DFT Queue",
            });
          }
        }

        if (submitted > 0) {
          broadcastThought(
            `Submitted ${submitted} top-0.1% candidates to Quantum ESPRESSO full DFT queue (${topIdx} elite out of ${scoredCandidates.length})`,
            "computation"
          );
        }
      }
    } catch (err: any) {
      emit("log", {
        phase: "engine",
        event: "full DFT queue error",
        detail: `Top 0.1% DFT submission failed: ${err.message?.slice(0, 200)}`,
        dataSource: "QE-DFT Queue",
      });
    }
  } catch (e) { console.error("[Engine] DFT enrichment outer error:", e); }
}

async function runPhase10_Physics() {
  if (!shouldContinue()) return;
  activeTasks.add("Computational Physics");
  broadcast("taskStart", { task: "Computational Physics" });
  try {
    await updatePhaseStatus(10, "active", 0, 0);
    if (!shouldContinue()) return;

    const stage0 = await storage.getSuperconductorsByStage(0);
    const toAnalyze = shuffle(stage0).slice(0, 8);

    for (const candidate of toAnalyze) {
      if (!shouldContinue()) break;
      try {
        const result = await runFullPhysicsAnalysis(emit, candidate);
        totalPhysicsComputed++;

        if (result.electronicStructure.metallicity < 0.1 && result.electronicStructure.bandGap > 0.3) {
          emit("log", {
            phase: "phase-10",
            event: "Insulator skipped",
            detail: `${candidate.formula}: bandGap=${result.electronicStructure.bandGap.toFixed(2)} eV, metallicity=${result.electronicStructure.metallicity.toFixed(2)} - suppressing SC predictions`,
            dataSource: "Physics Engine",
          });
          await storage.updateSuperconductorCandidate(candidate.id, {
            predictedTc: 0,
            notes: `${candidate.notes || ""} [Insulator: bandGap=${result.electronicStructure.bandGap.toFixed(2)}eV]`.trim(),
          });
          continue;
        }

        try {
          crossEngineHub.recordInsight("physics", candidate.formula, {
            lambda: result.coupling.lambda,
            dosAtFermi: result.electronicStructure.densityOfStatesAtFermi,
            omegaLog: result.coupling.omegaLog,
            metallicity: result.electronicStructure.metallicity,
            correlationStrength: result.correlation?.ratio ?? 0,
            instabilityFlags: result.instabilityProximity?.nearestBoundary ? [result.instabilityProximity.nearestBoundary] : [],
          });
        } catch (hubErr) { console.error(`[Engine] CrossEngineHub physics insight failed for ${candidate.formula}:`, hubErr); }

        const rawPhysicsTc = result.eliashberg.predictedTc;
        const physicsTc = (Number.isFinite(rawPhysicsTc) && rawPhysicsTc > 0 && rawPhysicsTc < 1000) ? rawPhysicsTc : 0;
        const currentTc = candidate.predictedTc ?? 0;
        const cappedPhysicsTc = physicsTc > 0
          ? applyAmbientTcCap(Math.round(physicsTc), result.coupling.lambda, candidate.pressureGpa ?? 0, result.electronicStructure.metallicity ?? 0.5, candidate.formula)
          : 0;
        const existingMlF = (candidate.mlFeatures as Record<string, any>) ?? {};
        const xTbTcFromPrior = (existingMlF.xTbTc as number) ?? undefined;
        const dftTcFromPrior = (existingMlF.dftTc as number) ?? undefined;
        const reconciled = reconcileTc({
          gbPredicted: candidate.xgboostScore != null ? currentTc : undefined,
          physicsTc: cappedPhysicsTc > 0 ? cappedPhysicsTc : undefined,
          xTbTc: xTbTcFromPrior,
          dftTc: dftTcFromPrior,
        });
        let updatedTc = reconciled.reconciledTc > 0 ? reconciled.reconciledTc : (cappedPhysicsTc > 0 ? cappedPhysicsTc : currentTc);

        const instProx = result.instabilityProximity;
        const existingNotes = candidate.notes || "";
        const instabilityNote = `[Instability: ${instProx.nearestBoundary}=${instProx.overallProximity.toFixed(2)}, QCP=${instProx.magneticQCP.toFixed(2)}, CDW=${instProx.cdwInstability.toFixed(2)}, MIT=${instProx.metalInsulatorTransition.toFixed(2)}]`;
        const pairingNote = `[Pairing: ${result.pairingAnalysis.dominant.mechanism} (Tc=${result.pairingAnalysis.dominant.tcEstimate.toFixed(0)}K, conf=${result.pairingAnalysis.dominant.confidence.toFixed(2)})]`;
        const newNotes = existingNotes.replace(/\[Instability:.*?\]/g, "").replace(/\[Pairing:.*?\]/g, "").trim();
        const updatedNotes = `${pairingNote} ${instabilityNote} ${newNotes}`.trim();

        const boundaryBoost = instProx.overallProximity > 0.5 ? instProx.overallProximity * 0.05 : 0;
        const currentEnsemble = candidate.ensembleScore ?? 0;
        const boostedEnsemble = Math.min(0.98, currentEnsemble + boundaryBoost);

        const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};
        const updatedMlFeatures = {
          ...existingMlFeatures,
          phononDispersion: {
            qPath: result.phononDispersion.qPath,
            branchCount: result.phononDispersion.branches.length,
            softModeQPoints: result.phononDispersion.softModeQPoints,
            imaginaryFrequencies: result.phononDispersion.imaginaryFrequencies,
            maxAcousticFreq: result.phononDispersion.maxAcousticFreq,
            minOpticalFreq: result.phononDispersion.minOpticalFreq,
            phononGap: result.phononDispersion.phononGap,
          },
          manyBodyCorrections: {
            quasiparticleWeight: result.manyBodyCorrections.quasiparticleWeight,
            gwDOSRenormalization: result.manyBodyCorrections.gwDOSRenormalization,
            gwBandwidthCorrection: result.manyBodyCorrections.gwBandwidthCorrection,
            vertexCorrectionLambda: result.manyBodyCorrections.vertexCorrectionLambda,
            correctedLambda: result.manyBodyCorrections.correctedLambda,
          },
          nestingFunction: {
            peakNestingQ: result.nestingFunction.peakNestingQ,
            peakNestingValue: result.nestingFunction.peakNestingValue,
            averageNesting: result.nestingFunction.averageNesting,
            nestingAnisotropy: result.nestingFunction.nestingAnisotropy,
            dominantInstability: result.nestingFunction.dominantInstability,
          },
          spinSusceptibility: {
            chiStaticPeak: result.spinSusceptibility.chiStaticPeak,
            chiDynamicPeak: result.spinSusceptibility.chiDynamicPeak,
            spinFluctuationEnergy: result.spinSusceptibility.spinFluctuationEnergy,
            correlationLength: result.spinSusceptibility.correlationLength,
            stonerEnhancement: result.spinSusceptibility.stonerEnhancement,
            isNearQCP: result.spinSusceptibility.isNearQCP,
          },
          phononDOS: {
            totalStates: result.phononDOS.totalStates,
            binCount: result.phononDOS.frequencies.length,
          },
          alpha2F: {
            integratedLambda: result.alpha2F.integratedLambda,
            binCount: result.alpha2F.frequencies.length,
          },
          advancedConstraints: {
            compositeScore: result.advancedConstraints.compositeScore,
            compositeBoost: result.advancedConstraints.compositeBoost,
            nesting: result.advancedConstraints.fermiSurfaceNesting.score,
            nestingStrength: result.advancedConstraints.fermiSurfaceNesting.nestingStrength,
            hybridization: result.advancedConstraints.orbitalHybridization.score,
            hybridizationType: result.advancedConstraints.orbitalHybridization.hybridizationType,
            lifshitz: result.advancedConstraints.lifshitzProximity.score,
            qcp: result.advancedConstraints.quantumCriticalFluctuation.score,
            qcpType: result.advancedConstraints.quantumCriticalFluctuation.qcpType,
            dimensionality: result.advancedConstraints.electronicDimensionality.score,
            dimensionClass: result.advancedConstraints.electronicDimensionality.dimensionClass,
            anisotropy: result.advancedConstraints.electronicDimensionality.anisotropy,
            softMode: result.advancedConstraints.phononSoftMode.score,
            softModeStable: result.advancedConstraints.phononSoftMode.isStable,
            chargeTransfer: result.advancedConstraints.chargeTransferEnergy.score,
            chargeTransferDelta: result.advancedConstraints.chargeTransferEnergy.delta,
            polarizability: result.advancedConstraints.latticePolarizability.score,
            dielectricConstant: result.advancedConstraints.latticePolarizability.dielectricConstant,
          },
        };

        let topoAnalysis: TopologicalAnalysis | undefined;
        try {
          const mlFeaturesForDFT = (candidate.mlFeatures as Record<string, any>) ?? {};
          const hasDFTBands = mlFeaturesForDFT.qeBands === true;
          let electronicForTopo = result.electronicStructure;
          let dftTopoClassification: DFTTopologicalClassification | undefined;

          if (hasDFTBands) {
            try {
              const dftJobs = await storage.getDftJobsByFormula(candidate.formula);
              const completedJob = dftJobs.find(j => j.status === "completed" && (j.outputData as any)?.bandStructure?.converged);
              if (completedJob) {
                const dftBandData = (completedJob.outputData as any).bandStructure as DFTBandStructureResult;
                recordDFTBandAnalysis(dftBandData);

                const dftTopo = extractTopologyFromDFT(dftBandData);
                const dftElecEnhance = enhanceElectronicStructure(dftBandData);

                const socStr = result.electronicStructure.socStrength ?? 0;
                dftTopoClassification = classifyDFTTopology(dftBandData, socStr);

                const isosurface = computeFermiIsosurface(dftBandData);

                electronicForTopo = {
                  ...result.electronicStructure,
                  bandStructureType: dftElecEnhance.bandStructureType,
                  fermiSurfaceTopology: dftElecEnhance.fermiSurfaceTopology,
                  densityOfStatesAtFermi: dftElecEnhance.densityOfStatesAtFermi,
                  metallicity: dftElecEnhance.metallicity,
                  nestingScore: dftElecEnhance.nestingScore,
                  vanHoveProximity: dftElecEnhance.vanHoveProximity,
                  bandFlatness: dftElecEnhance.bandFlatness,
                  flatBandIndicator: dftElecEnhance.flatBandIndicator,
                  tightBindingTopology: {
                    hasFlatBand: dftTopo.hasFlatBand,
                    hasVHS: dftTopo.hasVHS,
                    hasDiracCrossing: dftTopo.hasDiracCrossing,
                    hasBandInversion: dftTopo.hasBandInversion,
                    topologyScore: dftTopo.topologyScore,
                    flatBandCount: dftTopo.flatBandCount,
                    vhsCount: dftTopo.vhsCount,
                    diracCrossingCount: dftTopo.diracCrossingCount,
                    dosAtFermi: dftTopo.dosAtFermi,
                  },
                };

                (updatedMlFeatures as any).dftBandTopology = {
                  hasFlatBand: dftTopo.hasFlatBand,
                  hasVHS: dftTopo.hasVHS,
                  hasDiracCrossing: dftTopo.hasDiracCrossing,
                  hasBandInversion: dftTopo.hasBandInversion,
                  topologyScore: dftTopo.topologyScore,
                  flatBandCount: dftTopo.flatBandCount,
                  vhsCount: dftTopo.vhsCount,
                  dosAtFermi: dftTopo.dosAtFermi,
                  bandInversionCount: dftTopo.bandInversionCount,
                  nodalLineIndicator: dftTopo.nodalLineIndicator,
                  parityChanges: dftTopo.parityChanges,
                  diracPointCount: dftTopo.diracPointCount,
                  source: "DFT-bands",
                };

                (updatedMlFeatures as any).dftTopologicalClassification = {
                  topologicalClass: dftTopoClassification.topologicalClass,
                  confidence: dftTopoClassification.confidence,
                  socGapMeV: dftTopoClassification.socGapMeV,
                  diracPointCount: dftTopoClassification.diracPointCount,
                  weylPointCount: dftTopoClassification.weylPointCount,
                  nodalLineCount: dftTopoClassification.nodalLineCount,
                  z2Indicator: dftTopoClassification.z2Indicator,
                  evidence: dftTopoClassification.evidence,
                  chain: dftTopoClassification.classificationChain,
                };

                (updatedMlFeatures as any).fermiIsosurface = {
                  totalPoints: isosurface.totalPoints,
                  sheetCount: isosurface.sheetCount,
                  enclosedVolumeFraction: isosurface.enclosedVolumeFraction,
                  avgVelocity: isosurface.avgVelocity,
                  anisotropy: isosurface.anisotropy,
                };

                console.log(`[Engine] ${candidate.formula}: DFT topology classified — class=${dftTopoClassification.topologicalClass} (confidence=${dftTopoClassification.confidence.toFixed(2)}), inversions=${dftTopo.bandInversionCount}, VHS=${dftTopo.vhsCount}, Dirac=${dftTopo.diracPointCount}, Weyl=${dftTopoClassification.weylPointCount}, SOCgap=${dftTopoClassification.socGapMeV.toFixed(0)}meV, isosurface=${isosurface.sheetCount}sheets/${isosurface.totalPoints}pts`);
              }
            } catch (dftBandErr) {
              console.error(`[Engine] DFT band topology extraction failed for ${candidate.formula}:`, dftBandErr instanceof Error ? dftBandErr.message.slice(0, 100) : "unknown");
            }
          }

          topoAnalysis = analyzeTopology(
            candidate.formula,
            electronicForTopo,
            candidate.crystalStructure?.split(" ")[0],
            candidate.crystalStructure?.match(/\((\w+)\)/)?.[1],
            dftTopoClassification
          );
          trackTopologyResult(topoAnalysis);
          crossEngineHub.recordInsight("topology", candidate.formula, topoAnalysis);
          (updatedMlFeatures as any).topology = {
            topologicalScore: topoAnalysis.topologicalScore,
            z2Score: topoAnalysis.z2Score,
            chernScore: topoAnalysis.chernScore,
            mirrorSymmetryIndicator: topoAnalysis.mirrorSymmetryIndicator,
            socStrength: topoAnalysis.socStrength,
            bandInversionProbability: topoAnalysis.bandInversionProbability,
            diracNodeProbability: topoAnalysis.diracNodeProbability,
            majoranaFeasibility: topoAnalysis.majoranaFeasibility,
            topologicalClass: topoAnalysis.topologicalClass,
            indicators: topoAnalysis.indicators,
            dftBandEnhanced: hasDFTBands,
          };
          if (topoAnalysis.topologicalScore > 0.4) {
            emit("log", {
              phase: "phase-10",
              event: "Topological candidate detected",
              detail: `${candidate.formula}: class=${topoAnalysis.topologicalClass}, score=${topoAnalysis.topologicalScore}, SOC=${topoAnalysis.socStrength}, Z2=${topoAnalysis.z2Score}, Majorana=${topoAnalysis.majoranaFeasibility}${hasDFTBands ? " [DFT-enhanced]" : ""}, [${topoAnalysis.indicators.join(", ")}]`,
              dataSource: hasDFTBands ? "Topology Engine + DFT Bands" : "Topology Engine",
            });
          }

          try {
            const invariants = computeTopologicalInvariants(
              candidate.formula, electronicForTopo,
              candidate.crystalStructure?.split(" ")[0],
              candidate.crystalStructure?.match(/\((\w+)\)/)?.[1]
            );
            trackInvariantResult(candidate.formula, invariants);
            (updatedMlFeatures as any).topologicalInvariants = {
              bandInversion: {
                isInverted: invariants.bandInversion.isInverted,
                inversionType: invariants.bandInversion.inversionType,
                inversionStrength: invariants.bandInversion.inversionStrength,
                inversionGapMeV: invariants.bandInversion.inversionGapMeV,
              },
              z2: { index: invariants.z2Invariant.z2Index, isNontrivial: invariants.z2Invariant.isNontrivial },
              chern: { number: invariants.chernNumber.chernNumber, isQuantized: invariants.chernNumber.isQuantized },
              weylNodes: { count: invariants.weylNodes.nodeCount, isType2: invariants.weylNodes.isType2 },
              surfaceStates: { count: invariants.surfaceStates.surfaceStateCount, diracCones: invariants.surfaceStates.diracConeCount },
              symmetryIndicator: {
                spaceGroup: invariants.symmetryIndicator.spaceGroupName,
                indicator: invariants.symmetryIndicator.symmetryIndicator,
                topologyFromSymmetry: invariants.symmetryIndicator.topologyFromSymmetry,
                isObstructedAtomicLimit: invariants.symmetryIndicator.isObstructedAtomicLimit,
                fragileTopo: invariants.symmetryIndicator.fragileTopo,
                confidence: invariants.symmetryIndicator.confidence,
              },
              mlTopology: {
                topologyProbability: invariants.mlTopology.topologyProbability,
                tiProb: invariants.mlTopology.topologicalInsulatorProb,
                weylProb: invariants.mlTopology.weylSemimetalProb,
                diracProb: invariants.mlTopology.diracSemimetalProb,
                confidence: invariants.mlTopology.confidence,
              },
              tscScore: {
                score: invariants.tscScore.tscScore,
                isTSCCandidate: invariants.tscScore.isTSCCandidate,
                tscClass: invariants.tscScore.tscClass,
                interfacePotential: invariants.tscScore.interfacePotential,
              },
              phase: invariants.topologicalPhase,
              compositeScore: invariants.compositeTopologicalScore,
            };
            if (invariants.compositeTopologicalScore > 0.3) {
              emit("log", {
                phase: "phase-10",
                event: "Topological invariants computed",
                detail: `${candidate.formula}: phase=${invariants.topologicalPhase}, Z2=(${invariants.z2Invariant.z2Index.join(";")}), C=${invariants.chernNumber.chernNumber}, Weyl=${invariants.weylNodes.nodeCount}, surfaces=${invariants.surfaceStates.surfaceStateCount}, inversion=${invariants.bandInversion.inversionType}, SI=${invariants.symmetryIndicator.topologyFromSymmetry}, MLtopo=${(invariants.mlTopology.topologyProbability * 100).toFixed(0)}%, TSC=${invariants.tscScore.isTSCCandidate ? invariants.tscScore.tscClass : "no"}(${(invariants.tscScore.tscScore * 100).toFixed(0)}%)`,
                dataSource: "Topological Invariants Engine",
              });
            }
          } catch (invErr) {
            // invariant computation is optional
          }
        } catch (topoErr) {
          console.error(`[Engine] Topology analysis failed for ${candidate.formula}:`, topoErr instanceof Error ? topoErr.message.slice(0, 100) : "unknown");
        }

        let pairingProfile: PairingProfile | undefined;
        try {
          pairingProfile = computePairingProfile(candidate.formula);
          crossEngineHub.recordInsight("pairing", candidate.formula, pairingProfile);
          if (pairingProfile.compositePairingStrength > 0.4) {
            emit("log", {
              phase: "phase-10",
              event: "Pairing mechanism analysis",
              detail: `${candidate.formula}: dominant=${pairingProfile.dominantMechanism}, secondary=${pairingProfile.secondaryMechanism}, symmetry=${pairingProfile.pairingSymmetry}, composite=${pairingProfile.compositePairingStrength.toFixed(3)}, phonon=${pairingProfile.phonon.phononPairingStrength.toFixed(3)}, spin=${pairingProfile.spin.spinPairingStrength.toFixed(3)}, orbital=${pairingProfile.orbital.orbitalPairingStrength.toFixed(3)}`,
              dataSource: "Pairing Mechanism Simulator",
            });
          }
        } catch (pairErr) {
          console.error(`[Engine] Pairing profile failed for ${candidate.formula}:`, pairErr instanceof Error ? pairErr.message.slice(0, 100) : "unknown");
        }

        try {
          const reactionResult = analyzeReactionNetwork(candidate.formula);
          (updatedMlFeatures as any).reactionNetwork = {
            reactionStabilityScore: reactionResult.reactionStabilityScore,
            metastableLifetime: reactionResult.metastableLifetime,
            decompositionComplexity: reactionResult.decompositionComplexity,
            pathwayCount: reactionResult.reactionGraph?.edges?.length ?? 0,
          };
          if (reactionResult.reactionStabilityScore < 0.3) {
            emit("log", {
              phase: "phase-10",
              event: "Reaction stability warning",
              detail: `${candidate.formula}: stabilityScore=${reactionResult.reactionStabilityScore.toFixed(3)}, lifetime=${reactionResult.metastableLifetime}, complexity=${reactionResult.decompositionComplexity.toFixed(3)}, verdict=${reactionResult.stabilityVerdict}`,
              dataSource: "Reaction Network Engine",
            });
          }
        } catch (rxnErr) {
          console.error(`[Engine] Reaction network failed for ${candidate.formula}:`, rxnErr instanceof Error ? rxnErr.message.slice(0, 100) : "unknown");
        }

        let genomeResult: MaterialGenome | undefined;
        try {
          genomeResult = encodeGenome(candidate.formula);
          (updatedMlFeatures as any).genome = {
            family: genomeResult.metadata.family,
            dominantOrbital: genomeResult.metadata.dominantOrbital,
            genomeDim: genomeResult.vector.length,
          };
        } catch (genErr) {
          console.error(`[Engine] Genome encoding failed for ${candidate.formula}:`, genErr instanceof Error ? genErr.message.slice(0, 100) : "unknown");
        }

        let fermiSurfaceAnalysis: FermiSurfaceResult | undefined;
        try {
          const mlFeaturesFS = (candidate.mlFeatures as Record<string, any>) ?? {};
          const hasDFTBandsFS = mlFeaturesFS.qeBands === true;
          let usedDFTFermi = false;

          if (hasDFTBandsFS) {
            try {
              const dftJobsFS = await storage.getDftJobsByFormula(candidate.formula);
              const completedJobFS = dftJobsFS.find(j => j.status === "completed" && (j.outputData as any)?.bandStructure?.converged);
              if (completedJobFS) {
                const dftBandDataFS = (completedJobFS.outputData as any).bandStructure as DFTBandStructureResult;
                fermiSurfaceAnalysis = buildFermiSurfaceFromDFT(dftBandDataFS);
                usedDFTFermi = true;

                const lindhard = computeLindhardNesting(dftBandDataFS);
                if (lindhard.nestingStrength > fermiSurfaceAnalysis.nestingScore) {
                  fermiSurfaceAnalysis = {
                    ...fermiSurfaceAnalysis,
                    nestingScore: lindhard.nestingStrength,
                    nestingVectors: lindhard.qVectors.slice(0, 5).map((qv, i) => ({
                      q: qv.q,
                      strength: qv.chi / (lindhard.peakChi + 1e-6),
                      connectedPockets: [0, Math.min(1, fermiSurfaceAnalysis.pocketCount - 1)] as [number, number],
                    })),
                  };
                }

                const stabilityEst = candidate.ensembleScore ?? 0.5;
                const tcEst = candidate.predictedTc ?? 0;
                const topoEst = topoAnalysis?.topologicalScore ?? 0;
                const bfScore = computeBandFeatureScore(tcEst, stabilityEst, fermiSurfaceAnalysis.nestingScore, topoEst);

                (updatedMlFeatures as any).lindhardNesting = {
                  peakChi: lindhard.peakChi,
                  nestingStrength: lindhard.nestingStrength,
                  peakQ: lindhard.peakQ,
                  nQVectors: lindhard.qVectors.length,
                };

                (updatedMlFeatures as any).bandFeatureScore = {
                  composite: bfScore.compositeScore,
                  tcComponent: bfScore.tcComponent,
                  stabilityComponent: bfScore.stabilityComponent,
                  nestingComponent: bfScore.nestingComponent,
                  topologyComponent: bfScore.topologyComponent,
                };

                const isoFS = computeFermiIsosurface(dftBandDataFS);
                console.log(`[Engine] ${candidate.formula}: Fermi surface from DFT E(k)=EF isosurface — ${fermiSurfaceAnalysis.pocketCount} pockets (e=${fermiSurfaceAnalysis.electronPocketCount}, h=${fermiSurfaceAnalysis.holePocketCount}), nesting=${fermiSurfaceAnalysis.nestingScore.toFixed(3)} (Lindhard chi=${lindhard.peakChi.toFixed(3)}, strength=${lindhard.nestingStrength.toFixed(3)}), iso=${isoFS.totalPoints}pts/${isoFS.sheetCount}sheets, bandScore=${bfScore.compositeScore.toFixed(3)}`);
              }
            } catch (dftFsErr) {
              console.error(`[Engine] DFT Fermi surface extraction failed for ${candidate.formula}:`, dftFsErr instanceof Error ? dftFsErr.message.slice(0, 100) : "unknown");
            }
          }

          if (!fermiSurfaceAnalysis) {
            fermiSurfaceAnalysis = computeFermiSurface(candidate.formula);
          }

          crossEngineHub.recordInsight("fermi", candidate.formula, fermiSurfaceAnalysis);
          (updatedMlFeatures as any).fermiSurface = {
            fermiPocketCount: fermiSurfaceAnalysis.mlFeatures.fermiPocketCount,
            electronHoleBalance: fermiSurfaceAnalysis.mlFeatures.electronHoleBalance,
            fsDimensionality: fermiSurfaceAnalysis.mlFeatures.fsDimensionality,
            sigmaBandPresence: fermiSurfaceAnalysis.mlFeatures.sigmaBandPresence,
            multiBandScore: fermiSurfaceAnalysis.mlFeatures.multiBandScore,
            source: usedDFTFermi ? "DFT-bands" : "tight-binding",
          };
          if (fermiSurfaceAnalysis.pocketCount > 1 || fermiSurfaceAnalysis.nestingScore > 0.3) {
            emit("log", {
              phase: "phase-10",
              event: "Fermi surface reconstructed",
              detail: `${candidate.formula}: pockets=${fermiSurfaceAnalysis.pocketCount} (e=${fermiSurfaceAnalysis.electronPocketCount}, h=${fermiSurfaceAnalysis.holePocketCount}), e-h balance=${fermiSurfaceAnalysis.electronHoleBalance.toFixed(3)}, nesting=${fermiSurfaceAnalysis.nestingScore.toFixed(3)}, dim=${fermiSurfaceAnalysis.fsDimensionality}, sigma=${fermiSurfaceAnalysis.sigmaBandPresence.toFixed(3)}, multiBand=${fermiSurfaceAnalysis.multiBandScore.toFixed(3)}${usedDFTFermi ? " [DFT-derived]" : ""}`,
              dataSource: usedDFTFermi ? "DFT Band Structure -> Fermi Surface" : "Fermi Surface Engine",
            });
          }
          try {
            const clusterResult = assignToCluster(candidate.formula, fermiSurfaceAnalysis, candidate.predictedTc ?? 0);
            (updatedMlFeatures as any).fermiCluster = {
              clusterId: clusterResult.clusterId,
              clusterName: clusterResult.clusterName,
              similarity: clusterResult.similarity,
            };
          } catch (clErr) {
            console.error(`[Engine] Fermi cluster assignment failed for ${candidate.formula}:`, clErr instanceof Error ? clErr.message.slice(0, 100) : "unknown");
          }
        } catch (fsErr) {
          console.error(`[Engine] Fermi surface failed for ${candidate.formula}:`, fsErr instanceof Error ? fsErr.message.slice(0, 100) : "unknown");
        }

        let bandSurrogatePrediction: BandSurrogatePrediction | undefined;
        try {
          bandSurrogatePrediction = predictBandStructure(
            candidate.formula,
            candidate.crystalStructure?.match(/\((\w+)\)/)?.[1],
          );
          (updatedMlFeatures as any).bandSurrogate = getBandSurrogateMLFeatures(bandSurrogatePrediction);
          if (bandSurrogatePrediction.flatBandScore > 0.5 || bandSurrogatePrediction.vhsProximity > 0.4 || bandSurrogatePrediction.nestingFromBands > 0.4) {
            emit("log", {
              phase: "phase-10",
              event: "Band structure surrogate prediction",
              detail: `${candidate.formula}: gap=${bandSurrogatePrediction.bandGap}eV(${bandSurrogatePrediction.bandGapType}), flatBand=${bandSurrogatePrediction.flatBandScore.toFixed(3)}, VHS=${bandSurrogatePrediction.vhsProximity.toFixed(3)}, nesting=${bandSurrogatePrediction.nestingFromBands.toFixed(3)}, DOS(EF)=${bandSurrogatePrediction.dosPredicted.toFixed(3)}, fsDim=${bandSurrogatePrediction.fsDimensionality}, multiBand=${bandSurrogatePrediction.multiBandScore.toFixed(3)}, bwMin=${bandSurrogatePrediction.bandwidthMin.toFixed(4)}, topo=${bandSurrogatePrediction.bandTopologyClass}, conf=${bandSurrogatePrediction.confidence.toFixed(2)}`,
              dataSource: "Band Structure Surrogate",
            });
          }
        } catch (bsErr) {
          console.error(`[Engine] Band surrogate failed for ${candidate.formula}:`, bsErr instanceof Error ? bsErr.message.slice(0, 80) : "unknown");
        }

        let bandOperatorResult: BandOperatorResult | undefined;
        try {
          bandOperatorResult = predictBandDispersion(
            candidate.formula,
            candidate.crystalStructure?.match(/\((\w+)\)/)?.[1],
          );
          (updatedMlFeatures as any).bandOperator = getBandOperatorMLFeatures(bandOperatorResult);
          if (bandOperatorResult.derivedQuantities.vhsPositions.length > 0 || bandOperatorResult.derivedQuantities.topologicalInvariants.bandInversionCount > 0) {
            emit("log", {
              phase: "phase-10",
              event: "Band operator dispersion predicted",
              detail: `${candidate.formula}: path=${bandOperatorResult.dispersion.path}, nBands=${bandOperatorResult.dispersion.nBands}, VHS=${bandOperatorResult.derivedQuantities.vhsPositions.length}, inversions=${bandOperatorResult.derivedQuantities.topologicalInvariants.bandInversionCount}, topo=${bandOperatorResult.derivedQuantities.topologicalInvariants.topologicalClass}, berry=${bandOperatorResult.derivedQuantities.topologicalInvariants.berryPhaseProxy.toFixed(3)}, conf=${bandOperatorResult.confidence.toFixed(2)}`,
              dataSource: "Band Structure Operator",
            });
          }
        } catch (boErr) {
          console.error(`[Engine] Band operator failed for ${candidate.formula}:`, boErr instanceof Error ? boErr.message.slice(0, 80) : "unknown");
        }

        let qcAnalysis: QuantumCriticalAnalysis | undefined;
        try {
          qcAnalysis = detectQuantumCriticality(candidate.formula, {
            electronic: result.electronicStructure,
            coupling: result.coupling,
          });
          (updatedMlFeatures as any).quantumCriticality = {
            score: qcAnalysis.quantumCriticalScore,
            primaryQCP: qcAnalysis.qcpType,
            pairingBoost: qcAnalysis.pairingBoostFromQCP,
          };
          if (qcAnalysis.quantumCriticalScore > 0.5) {
            const qcBoost = 1 + qcAnalysis.pairingBoostFromQCP * 0.15;
            updatedTc = Math.min(400, updatedTc * qcBoost);
            emit("log", {
              phase: "phase-10",
              event: "Quantum criticality detected",
              detail: `${candidate.formula}: QCP=${qcAnalysis.qcpType}, score=${qcAnalysis.quantumCriticalScore.toFixed(3)}, dome=${qcAnalysis.dome.domeAmplitude.toFixed(2)}, boost=${qcAnalysis.pairingBoostFromQCP.toFixed(3)}, channels=[mott=${qcAnalysis.channelScores.mott.toFixed(2)},sdw=${qcAnalysis.channelScores.sdw.toFixed(2)},cdw=${qcAnalysis.channelScores.cdw.toFixed(2)},nematic=${qcAnalysis.channelScores.nematic.toFixed(2)}]`,
              dataSource: "Quantum Criticality Detector",
            });
          }
        } catch (qcErr) {
          console.error(`[Engine] Quantum criticality failed for ${candidate.formula}:`, qcErr instanceof Error ? qcErr.message.slice(0, 80) : "unknown");
        }

        await storage.updateSuperconductorCandidate(candidate.id, {
          electronPhononCoupling: result.coupling.lambda,
          logPhononFrequency: result.coupling.omegaLog,
          coulombPseudopotential: result.coupling.muStar,
          correlationStrength: result.correlation.ratio,
          fermiSurfaceTopology: result.electronicStructure.fermiSurfaceTopology,
          dimensionality: result.dimensionality,
          competingPhases: result.competingPhases as any,
          upperCriticalField: result.criticalFields.upperCriticalField,
          coherenceLength: result.criticalFields.coherenceLength,
          londonPenetrationDepth: result.criticalFields.londonPenetrationDepth,
          anisotropyRatio: result.criticalFields.anisotropyRatio,
          criticalCurrentDensity: result.criticalFields.criticalCurrentDensity,
          uncertaintyEstimate: result.uncertaintyEstimate,
          pairingMechanism: pairingProfile?.dominantMechanism ?? result.pairingAnalysis.dominant.mechanism,
          cooperPairMechanism: pairingProfile ? `${pairingProfile.dominantMechanism} (${pairingProfile.pairingSymmetry}), composite=${pairingProfile.compositePairingStrength.toFixed(3)}` : result.pairingAnalysis.dominant.description,
          predictedTc: updatedTc,
          verificationStage: 1,
          notes: updatedNotes,
          ensembleScore: boostedEnsemble,
          mlFeatures: updatedMlFeatures as any,
        });

        if (updatedTc !== currentTc) {
          emit("log", { phase: "phase-10", event: "Tc updated by physics", detail: `${candidate.formula}: ML estimate ${currentTc}K -> Eliashberg ${updatedTc}K (lambda=${result.coupling.lambda.toFixed(2)}, Hc2=${result.criticalFields.upperCriticalField}T, ${result.correlation.regime}, ${result.competingPhases.length} competing phases)`, dataSource: "Physics Engine" });
        }

        try {
          const trainFeatures = extractFeatures(candidate.formula);
          const hullDist = result.competingPhases.length > 0 ? 0.05 * result.competingPhases.length : 0.02;
          physicsPredictor.addTrainingSample(
            trainFeatures,
            result.coupling.lambda,
            result.electronicStructure.densityOfStatesAtFermi,
            result.coupling.omegaLog,
            hullDist
          );
        } catch (tsErr) {
          console.error(`[Engine] Training sample add failed for ${candidate.formula}:`, tsErr instanceof Error ? tsErr.message.slice(0, 80) : "unknown");
        }

        try {
          buildAndStoreFeatureRecord(candidate.formula, updatedTc);
          recordPrediction(candidate.formula, currentTc, updatedTc);
          updatePhysicsParameters(updatedTc, currentTc, [], candidate.formula);
        } catch (frErr) {
          console.error(`[Engine] Feature record/prediction update failed for ${candidate.formula}:`, frErr instanceof Error ? frErr.message.slice(0, 80) : "unknown");
        }

        try {
          addMaterialToDataset(candidate.formula, updatedTc, result.coupling.lambda, result.pairingAnalysis?.symmetry ?? "unknown", result.pairingAnalysis?.dominantMechanism ?? "unknown");
        } catch (dsErr) {
          console.error(`[Engine] Dataset add failed for ${candidate.formula}:`, dsErr instanceof Error ? dsErr.message.slice(0, 80) : "unknown");
        }

        if (updatedTc > 5) {
          const candFamily = classifyFamily(candidate.formula);
          try {
            const defects = generateDefectVariants(candidate.formula);
            if (defects.length > 0) {
              const bestDefect = defects.reduce((best: any, d: any) => {
                const adj = adjustElectronicStructure(
                  result.electronicStructure.densityOfStatesAtFermi ?? 1.0,
                  result.coupling.lambda ?? 0.5,
                  d.defectDensity,
                  d.type,
                  candidate.formula
                );
                return adj.tcModifier > (best?.tcMod ?? 0) ? { defect: d, tcMod: adj.tcModifier } : best;
              }, null as { defect: any; tcMod: number } | null);
              if (bestDefect && bestDefect.tcMod > 1.05) {
                feedbackLoopStats.defectCandidatesAdded++;
                feedbackLoopStats.defectTotalTcBoost += bestDefect.tcMod - 1;
                crossEngineHub.recordInsight("defect", candidate.formula, {
                  optimalDopants: [bestDefect.defect.element],
                  bestTcModifier: bestDefect.tcMod,
                  bestDefectType: bestDefect.defect.type,
                  variantCount: defects.length,
                  bestMutatedFormula: bestDefect.defect.mutatedFormula || candidate.formula,
                });
              }
            }
          } catch (e) { console.error(`[Engine] Phase-10 defect analysis failed for ${candidate.formula}:`, e); }

          try {
            const suggestions = suggestDisorders(candidate.formula);
            if (suggestions.length > 0) {
              const topSuggestions = suggestions.slice(0, 3);
              for (const spec of topSuggestions) {
                const disordered = generateAllDisorderVariants(candidate.formula, [spec.fraction]);
                const bestDisorder = disordered.reduce((best, d) =>
                  d.tcModifierEstimate > (best?.tcModifierEstimate ?? 0) ? d : best, disordered[0]);
                if (bestDisorder && bestDisorder.tcModifierEstimate > 1.05) {
                  let disorderAwareTc = updatedTc * bestDisorder.tcModifierEstimate;
                  if (bestDisorder.metrics) {
                    try {
                      const dCtx: DisorderContext = {
                        vacancyFraction: bestDisorder.metrics.vacancyFraction,
                        bondVariance: bestDisorder.metrics.bondVariance,
                        latticeStrain: bestDisorder.metrics.localStrainMean,
                        siteMixingEntropy: bestDisorder.metrics.siteMixingFraction > 0
                          ? -bestDisorder.metrics.siteMixingFraction * Math.log(bestDisorder.metrics.siteMixingFraction) : 0,
                        configurationalEntropy: bestDisorder.metrics.configurationalEntropy,
                        dosDisorderSignal: bestDisorder.metrics.dosDisorderSignal,
                      };
                      const disorderFeatures = extractFeatures(candidate.formula, undefined, undefined, undefined, undefined, dCtx);
                      const disorderPred = gbPredict(disorderFeatures);
                      disorderAwareTc = Math.max(disorderAwareTc, disorderPred.tcPredicted);
                    } catch { /* use modifier-based estimate */ }
                  }
                  crossEngineHub.recordInsight("defect", candidate.formula, {
                    disorderType: bestDisorder.disorder.type,
                    disorderElement: bestDisorder.disorder.element,
                    disorderFraction: bestDisorder.disorder.fraction,
                    tcModifier: bestDisorder.tcModifierEstimate,
                    defectCount: bestDisorder.defectCount,
                    totalAtoms: bestDisorder.totalAtoms,
                    disorderAwareTc,
                    configEntropy: bestDisorder.metrics?.configurationalEntropy ?? 0,
                    dosSignal: bestDisorder.metrics?.dosDisorderSignal ?? 0,
                  });
                }
              }
            }
          } catch (e) { console.error(`[Engine] Phase-10 disorder generation failed for ${candidate.formula}:`, e); }

          try {
            const corrEffects = estimateCorrelationEffects(candidate.formula, {
              UoverW: result.correlation?.ratio,
              dosAtEF: result.electronicStructure?.densityOfStatesAtFermi,
            });
            if (corrEffects.tcModifier > 1.05) {
              feedbackLoopStats.correlationBoostsApplied++;
              feedbackLoopStats.correlationTotalTcBoost += Math.min(corrEffects.tcModifier, 1.5) - 1;
            }
          } catch (e) { console.error(`[Engine] Phase-10 correlation analysis failed for ${candidate.formula}:`, e); }

          try {
            const sv = defaultSynthesisVector(candFamily);
            const growthResult = simulateCrystalGrowth(candidate.formula, candFamily, sv);
            if (growthResult.qualityScore >= 0.6) {
              feedbackLoopStats.growthQualityBonuses++;
              feedbackLoopStats.growthTotalQualityBoost += growthResult.qualityScore * 0.03;
            }
          } catch (e) { console.error(`[Engine] Phase-10 crystal growth failed for ${candidate.formula}:`, e); }

          try {
            const synthCtx: MaterialContext = {
              formula: candidate.formula,
              materialClass: candFamily,
              predictedTc: updatedTc,
              lambda: result.coupling.lambda ?? 0.5,
              pressure: 0,
              isHydride: candFamily.toLowerCase().includes("hydride"),
              isCuprate: candFamily.toLowerCase().includes("cuprate"),
              isLayered: false,
              meltingPointEstimate: 1500,
              stabilityClass: "metastable-accessible",
              energyAboveHull: 0.05,
            };
            const synthResult = optimizeSynthesisConditions(synthCtx);
            const sv = defaultSynthesisVector(candFamily);
            recordSynthesisResult(candidate.formula, candFamily, sv, updatedTc, 0.7);
            crossEngineHub.recordInsight("synthesis", candidate.formula, {
              feasibilityScore: synthResult.overallFeasibility,
              complexity: 0.5,
              bestMethod: candFamily.toLowerCase().includes("hydride") ? "high-pressure" : "solid-state",
              pathStepCount: 4,
            });
            if (synthResult.overallFeasibility > 0.6) {
              feedbackLoopStats.synthesisFeasibilityBonuses++;
              feedbackLoopStats.synthesisTotalFeasibilityBoost += synthResult.overallFeasibility * 0.05;
            }
          } catch (e) { console.error(`[Engine] Phase-10 synthesis analysis failed for ${candidate.formula}:`, e); }

          if (updatedTc > 15) {
            try {
              const hubInsights = crossEngineHub.getInsightsFor(candidate.formula);
              const multiInsights: MultiEngineInsights = {
                formula: candidate.formula,
                materialClass: candFamily,
                predictedTc: updatedTc,
                physics: hubInsights?.physics ? {
                  lambda: hubInsights.physics.lambda,
                  omegaLog: hubInsights.physics.omegaLog,
                  dosAtFermi: hubInsights.physics.dosAtFermi,
                  metallicity: hubInsights.physics.metallicity,
                  stabilityScore: 1 - (result.instabilityProximity?.overallProximity ?? 0.5),
                  correlationStrength: hubInsights.physics.correlationStrength,
                } : undefined,
                topology: topoAnalysis ? {
                  topologicalScore: topoAnalysis.topologicalScore,
                  z2Score: topoAnalysis.z2Score,
                  socStrength: topoAnalysis.socStrength,
                  topologicalClass: topoAnalysis.topologicalClass,
                  majoranaFeasibility: topoAnalysis.majoranaFeasibility,
                  bandInversionProbability: topoAnalysis.bandInversionProbability,
                } : undefined,
                fermi: fermiSurfaceAnalysis ? {
                  nestingScore: fermiSurfaceAnalysis.nestingScore,
                  pocketCount: fermiSurfaceAnalysis.pocketCount,
                  electronHoleBalance: fermiSurfaceAnalysis.electronHoleBalance,
                  cylindricalCharacter: fermiSurfaceAnalysis.mlFeatures.fsDimensionality > 2 ? 0.3 : 0.7,
                  multiBandScore: fermiSurfaceAnalysis.mlFeatures.multiBandScore,
                  sigmaBandPresence: fermiSurfaceAnalysis.mlFeatures.sigmaBandPresence,
                } : undefined,
                pairing: pairingProfile ? {
                  dominantMechanism: pairingProfile.dominantMechanism,
                  pairingSymmetry: pairingProfile.pairingSymmetry,
                  compositePairingStrength: pairingProfile.compositePairingStrength,
                  phononStrength: pairingProfile.phonon.phononPairingStrength,
                  spinStrength: pairingProfile.spin.spinPairingStrength,
                  orbitalStrength: pairingProfile.orbital.orbitalPairingStrength,
                } : undefined,
              };
              const synthDiscovery = discoverNovelSynthesisPaths(multiInsights, 12, 8);
              if (synthDiscovery.bestRoute) {
                crossEngineHub.recordInsight("synthesis", candidate.formula, {
                  feasibilityScore: synthDiscovery.bestRoute.feasibilityScore,
                  complexity: synthDiscovery.bestRoute.steps.length,
                  bestMethod: synthDiscovery.bestRoute.steps[0]?.method ?? "novel",
                  pathStepCount: synthDiscovery.bestRoute.steps.length,
                });
                if (synthDiscovery.bestRoute.fitnessScore > 0.5) {
                  emit("log", {
                    phase: "phase-10",
                    event: "Novel synthesis path discovered",
                    detail: `${candidate.formula}: ${synthDiscovery.bestRoute.steps.length} steps, fitness=${synthDiscovery.bestRoute.fitnessScore.toFixed(3)}, feasibility=${synthDiscovery.bestRoute.feasibilityScore.toFixed(3)}, novelty=${synthDiscovery.bestRoute.noveltyScore.toFixed(3)}, engines=[${synthDiscovery.bestRoute.engineContributions.join(",")}]`,
                    dataSource: "Synthesis Discovery",
                  });
                }
              }
            } catch (e) { console.error(`[Engine] Phase-10 novel synthesis discovery failed for ${candidate.formula}:`, e); }
          }

          if (updatedTc > 25 && cycleCount % 3 === 0) {
            try {
              const expCandidate: ExperimentCandidate = {
                formula: candidate.formula,
                predictedTc: updatedTc,
                stability: 0.7,
                synthesisFeasibility: 0.5,
                novelty: 0.6,
                uncertainty: 0.5,
                materialClass: candFamily,
                crystalStructure: "predicted",
              };
              const plan = generateExperimentPlan(expCandidate);
              feedbackLoopStats.experimentPlansGenerated++;
              if (plan.ranking.experimentScore > 0.6) {
                feedbackLoopStats.experimentDFTPrioritized++;
              }
            } catch (e) { console.error(`[Engine] Phase-10 experiment plan failed for ${candidate.formula}:`, e); }
          }
        }
      } catch (err: any) {
        emit("log", { phase: "phase-10", event: "Physics analysis error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Physics Engine" });
      }
    }

    if (shouldContinue()) {
      try {
        const allCandidates = await storage.getSuperconductorCandidates(50);
        const hydrideCandidates = allCandidates.filter(c => {
          const els = parseFormulaElements(c.formula);
          const cts: Record<string, number> = {};
          const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
          let m;
          while ((m = regex.exec(c.formula)) !== null) {
            cts[m[1]] = (cts[m[1]] || 0) + (m[2] ? parseFloat(m[2]) : 1);
          }
          const hCount = cts["H"] || 0;
          const nonmetals = ["H","He","B","C","N","O","F","Ne","Si","P","S","Cl","Ar","Ge","As","Se","Br","Kr","Te","I","Xe"];
          const metalAtoms = els.filter(e => !nonmetals.includes(e)).reduce((s, e) => s + (cts[e] || 0), 0);
          return hCount > 0 && metalAtoms > 0 && hCount / metalAtoms >= 2;
        });
        const hydrideToScan = shuffle(hydrideCandidates).slice(0, 5);
        for (const candidate of hydrideToScan) {
          if (!shouldContinue()) break;
          try {
            const pressureResult = runPressureAnalysis(emit, candidate.formula);
            const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};

            let hydrogenNetworkData: Record<string, any> = {};
            try {
              const hNetwork = analyzeHydrogenNetwork(candidate.formula);
              trackHydrogenNetworkResult(hNetwork);
              hydrogenNetworkData = {
                hydrogenNetworkDim: hNetwork.hydrogenNetworkDim,
                hydrogenCageScore: hNetwork.hydrogenCageScore,
                Hcoordination: hNetwork.Hcoordination,
                hydrogenConnectivity: hNetwork.hydrogenConnectivity,
                hydrogenPhononCouplingScore: hNetwork.hydrogenPhononCouplingScore,
                networkClass: hNetwork.networkClass,
                compositeSCScore: hNetwork.compositeSCScore,
                bondingType: hNetwork.bondingType,
              };
              if (hNetwork.compositeSCScore > 0.4) {
                emit("log", {
                  phase: "phase-10",
                  event: "Hydrogen network analysis",
                  detail: `${candidate.formula}: class=${hNetwork.networkClass}, dim=${hNetwork.hydrogenNetworkDim}, cage=${hNetwork.hydrogenCageScore.toFixed(3)}, coord=${hNetwork.Hcoordination}, phonon=${hNetwork.hydrogenPhononCouplingScore.toFixed(3)}, composite=${hNetwork.compositeSCScore.toFixed(3)}`,
                  dataSource: "Hydrogen Network Engine",
                });
              }
            } catch (e) { console.error(`[Engine] Hydrogen network analysis failed for ${candidate.formula}:`, e); }

            try {
              crossEngineHub.recordInsight("pressure", candidate.formula, {
                optimalPressure: pressureResult.optimalPressure,
                maxTc: pressureResult.maxTc,
                retentionFraction: pressureResult.maxTc > 0 ? (candidate.predictedTc ?? 0) / pressureResult.maxTc : 0,
                stabilizationStrategy: "pressure-scan",
                hydridePhaseCount: pressureResult.hydrideFormation?.stableHydrides.length ?? 0,
              });
            } catch (hubErr) { console.error(`[Engine] CrossEngineHub pressure insight failed for ${candidate.formula}:`, hubErr); }
            const updatedMlFeatures = {
              ...existingMlFeatures,
              pressureTcCurve: {
                optimalPressure: pressureResult.optimalPressure,
                maxTc: pressureResult.maxTc,
                points: pressureResult.pressureTcCurve.length,
                stablePoints: pressureResult.pressureTcCurve.filter(p => p.stable).length,
                hydridePhases: pressureResult.hydrideFormation?.stableHydrides.length ?? 0,
              },
              ...(Object.keys(hydrogenNetworkData).length > 0 ? { hydrogenNetwork: hydrogenNetworkData } : {}),
            };
            const hydrideUpdates: any = { mlFeatures: updatedMlFeatures as any };
            if (pressureResult.optimalPressure > 50) {
              hydrideUpdates.ambientPressureStable = false;
            }
            await storage.updateSuperconductorCandidate(candidate.id, hydrideUpdates);
          } catch (err: any) {
            emit("log", { phase: "phase-10", event: "Pressure scan error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Pressure Engine" });
          }
        }

        const metallicHighTc = allCandidates.filter(c => {
          const tc = c.predictedTc ?? 0;
          const met = ((c.mlFeatures as Record<string, any>)?.metallicity) ?? 0;
          const existingPressure = ((c.mlFeatures as Record<string, any>)?.pressureTcCurve?.optimalPressure) ?? null;
          return tc >= 20 && met > 0.4 && existingPressure === null && !hydrideCandidates.some(h => h.id === c.id);
        });
        const nonHydrideToScan = shuffle(metallicHighTc).slice(0, 5);
        for (const candidate of nonHydrideToScan) {
          if (!shouldContinue()) break;
          try {
            const pressureResult = runPressureAnalysis(emit, candidate.formula);
            const ambientTc = candidate.predictedTc ?? 0;
            const pressureTc = pressureResult.maxTc;
            const existingMlFeatures = (candidate.mlFeatures as Record<string, any>) ?? {};
            const updatedMlFeatures = {
              ...existingMlFeatures,
              pressureTcCurve: {
                optimalPressure: pressureResult.optimalPressure,
                maxTc: pressureTc,
                points: pressureResult.pressureTcCurve.length,
                stablePoints: pressureResult.pressureTcCurve.filter(p => p.stable).length,
                pressureBoost: pressureTc > ambientTc,
              },
            };
            const updates: any = { mlFeatures: updatedMlFeatures as any };
            if (pressureResult.optimalPressure > 50) {
              updates.ambientPressureStable = false;
            }
            if (pressureTc > ambientTc && pressureResult.optimalPressure <= 50 && pressureResult.optimalPressure > 0) {
              updates.pressureGpa = pressureResult.optimalPressure;
              updates.optimalPressureGpa = pressureResult.optimalPressure;
              emit("log", {
                phase: "phase-10",
                event: "Pressure-enhanced Tc",
                detail: `${candidate.formula}: Tc ${ambientTc}K -> ${pressureTc.toFixed(1)}K at ${pressureResult.optimalPressure.toFixed(1)} GPa (moderate pressure, non-hydride)`,
                dataSource: "Pressure Engine",
              });
            }
            await storage.updateSuperconductorCandidate(candidate.id, updates);
          } catch (err: any) {
            emit("log", { phase: "phase-10", event: "Pressure scan error", detail: `${candidate.formula}: ${err.message?.slice(0, 150)}`, dataSource: "Pressure Engine" });
          }
        }
      } catch (e) { console.error("[Engine] Pressure scan outer error:", e); }
    }

    if (cyclesSinceTcImproved > 3 && shouldContinue()) {
      const stage4 = await storage.getSuperconductorsByStage(4, 20);
      const highLambda = stage4
        .filter(c => (c.electronPhononCoupling ?? 0) > 2.0)
        .sort((a, b) => (b.predictedTc ?? 0) - (a.predictedTc ?? 0));
      const toReanalyze = shuffle(highLambda).slice(0, 2);
      for (const candidate of toReanalyze) {
        if (!shouldContinue()) break;
        try {
          const result = await runFullPhysicsAnalysis(emit, candidate);
          totalPhysicsComputed++;
          const newLambda = result.coupling.lambda ?? 0;
          const oldLambda = candidate.electronPhononCoupling ?? 0;
          if (Math.abs(newLambda - oldLambda) > 0.05) {
            const rawTc = result.eliashberg.predictedTc;
            const physicsTc = (Number.isFinite(rawTc) && rawTc > 0 && rawTc < 1000) ? rawTc : 0;
            const currentTc = candidate.predictedTc ?? 0;
            const cappedRePhys = physicsTc > 0
              ? applyAmbientTcCap(Math.round(physicsTc), newLambda, candidate.pressureGpa ?? 0, result.electronicStructure.metallicity ?? 0.5, candidate.formula)
              : 0;
            const reconciledRePhys = reconcileTc({
              gbPredicted: candidate.xgboostScore != null ? currentTc : undefined,
              physicsTc: cappedRePhys > 0 ? cappedRePhys : undefined,
            });
            let updatedTc = reconciledRePhys.reconciledTc > 0 ? reconciledRePhys.reconciledTc : (cappedRePhys > 0 ? cappedRePhys : currentTc);
            await storage.updateSuperconductorCandidate(candidate.id, {
              electronPhononCoupling: newLambda,
              logPhononFrequency: result.coupling.omegaLog,
              predictedTc: updatedTc,
            });
            if (updatedTc !== currentTc) {
              emit("log", { phase: "phase-10", event: "Re-physics corrected Tc", detail: `${candidate.formula}: ${currentTc}K -> ${updatedTc}K (lambda ${oldLambda.toFixed(2)} -> ${newLambda.toFixed(2)}, Hc2=${result.criticalFields.upperCriticalField}T, ${result.correlation.regime}, ${result.competingPhases.length} competing phases)`, dataSource: "Physics Engine" });
            }
          }
        } catch (e) { console.error("[Engine] Re-physics correction failed:", e); }
      }
    }

    const crCount = await storage.getComputationalResultCount();
    const progress10 = computeProgress(10, crCount);
    await updatePhaseStatus(10, "active", progress10, crCount);
  } finally {
    activeTasks.delete("Computational Physics");
    broadcast("taskEnd", { task: "Computational Physics" });
  }
}

async function runPhase11_StructurePrediction() {
  if (!shouldContinue()) return;
  activeTasks.add("Crystal Structure Prediction");
  broadcast("taskStart", { task: "Crystal Structure Prediction" });
  try {
    await updatePhaseStatus(11, "active", 0, 0);
    if (!shouldContinue()) return;

    const candidates = await storage.getSuperconductorCandidates(50);
    const uniqueFormulas = shuffle(candidates
      .map(c => c.formula)
      .filter((f, i, arr) => arr.indexOf(f) === i));

    const needsPrediction: string[] = [];
    for (const formula of uniqueFormulas) {
      if (needsPrediction.length >= 5) break;
      const existing = await storage.getCrystalStructuresByFormula(formula);
      if (existing.length === 0) {
        needsPrediction.push(formula);
      }
    }

    if (needsPrediction.length === 0) {
      emit("log", { phase: "phase-11", event: "All top candidates have crystal structures", detail: `Checked ${uniqueFormulas.length} unique formulas, all already predicted`, dataSource: "Structure Predictor" });
    }

    const predicted = needsPrediction.length > 0
      ? await runStructurePredictionBatch(emit, needsPrediction)
      : 0;
    totalStructuresPredicted += predicted;

    if (shouldContinue()) {
      for (const f of uniqueFormulas.slice(0, 5)) {
        if (!shouldContinue()) break;
        try {
          const hullResult = await runConvexHullAnalysis(emit, f);
          const matchingCandidates = candidates.filter(c => c.formula === f);
          for (const cand of matchingCandidates) {
            const existingNotes = cand.notes || "";
            const hullNote = `[ConvexHull: eAboveHull=${hullResult.energyAboveHull.toFixed(4)}, onHull=${hullResult.isOnHull}, decomp=${hullResult.decompositionProducts.join("+")}]`;
            if (!existingNotes.includes("[ConvexHull:")) {
              await storage.updateSuperconductorCandidate(cand.id, {
                notes: `${existingNotes} ${hullNote}`.trim(),
              });
            }
          }
        } catch (hullErr) {
          console.log(`[Engine] ConvexHull analysis failed for ${f}: ${hullErr instanceof Error ? hullErr.message.slice(0, 100) : "unknown"}`);
        }
      }
    }

    if (shouldContinue() && cycleCount % 3 === 0) {
      try {
        const topCandidates = candidates
          .filter(c => (c.ensembleScore ?? 0) > 0.3)
          .map(c => ({ formula: c.formula, predictedTc: c.predictedTc ?? 0, ensembleScore: c.ensembleScore ?? 0 }));

        if (topCandidates.length > 0) {
          const variants = await runGenerativeStructureDiscovery(emit, topCandidates);

          for (const variant of variants) {
            if (!isValidFormula(variant.formula)) continue;
            variant.formula = normalizeFormula(variant.formula);
            const existingSC = await storage.getSuperconductorByFormula(variant.formula);
            if (!existingSC) {
              const features = extractFeatures(variant.formula);
              const gbResult = gbPredict(features);
              const lambdaML = features.electronPhononLambda ?? 0;
              const metallicityML = features.metallicity ?? 0.5;
              let rawTc = estimateRawTc(lambdaML, features.logPhononFreq);
              rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, variant.formula);

              const id = `sc-struct-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
              try {
                const inserted = await insertCandidateWithStabilityCheck({
                  id,
                  name: variant.formula,
                  formula: variant.formula,
                  predictedTc: rawTc,
                  pressureGpa: estimateFamilyPressure(variant.formula),
                  meissnerEffect: false,
                  zeroResistance: false,
                  cooperPairMechanism: `Structural variant from ${variant.parentFormula} via ${variant.variationType}`,
                  crystalStructure: `${variant.spaceGroup} (${variant.crystalSystem})`,
                  quantumCoherence: variant.structuralNovelty,
                  stabilityScore: features.cooperPairStrength,
                  synthesisPath: null,
                  mlFeatures: features as any,
                  xgboostScore: gbResult.score,
                  neuralNetScore: variant.structuralNovelty,
                  ensembleScore: Math.min(0.9, (gbResult.score + variant.structuralNovelty) / 2),
                  roomTempViable: false,
                  status: "theoretical",
                  notes: `[Structural variant: ${variant.variationType}, topology=${variant.topology}, novelty=${variant.structuralNovelty.toFixed(2)}] ${variant.description}`,
                  electronPhononCoupling: features.electronPhononLambda ?? null,
                  logPhononFrequency: features.logPhononFreq ?? null,
                  coulombPseudopotential: 0.12,
                  pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
                  pairingMechanism: "phonon-mediated",
                  correlationStrength: features.correlationStrength ?? null,
                  dimensionality: variant.dimensionality,
                  fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                  uncertaintyEstimate: 0.6,
                  verificationStage: 0,
                  dataConfidence: "low",
                }, "structure_diffusion");
                if (inserted) totalScCandidates++;
              } catch (e) { console.error("[Engine] Structural variant insert failed:", e); }
            }
          }
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Generative structure error", detail: err.message?.slice(0, 150), dataSource: "Structure Generator" });
      }
    }

    if (shouldContinue() && cycleCount % 10 === 0) {
      try {
        const novelVariants = await runNovelPrototypeGeneration(emit);

        for (const variant of novelVariants) {
          if (!isValidFormula(variant.formula)) continue;
          const existingSC = await storage.getSuperconductorByFormula(variant.formula);
          if (!existingSC) {
            const features = extractFeatures(variant.formula);
            const gbResult = gbPredict(features);
            const lambdaML = features.electronPhononLambda ?? 0;
            const metallicityML = features.metallicity ?? 0.5;
            let rawTc = estimateRawTc(lambdaML, features.logPhononFreq);
            rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, variant.formula);

            const id = `sc-novel-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            try {
              const inserted = await insertCandidateWithStabilityCheck({
                id,
                name: variant.formula,
                formula: variant.formula,
                predictedTc: rawTc,
                pressureGpa: estimateFamilyPressure(variant.formula),
                meissnerEffect: false,
                zeroResistance: false,
                cooperPairMechanism: `Novel prototype: ${variant.topology}`,
                crystalStructure: `${variant.spaceGroup} (${variant.crystalSystem})`,
                quantumCoherence: variant.structuralNovelty,
                stabilityScore: features.cooperPairStrength,
                synthesisPath: null,
                mlFeatures: features as any,
                xgboostScore: gbResult.score,
                neuralNetScore: variant.structuralNovelty,
                ensembleScore: Math.min(0.9, (gbResult.score + variant.structuralNovelty) / 2),
                roomTempViable: false,
                status: "theoretical",
                notes: `[Novel prototype: ${variant.topology}, novelty=${variant.structuralNovelty.toFixed(2)}] ${variant.description}`,
                electronPhononCoupling: features.electronPhononLambda ?? null,
                logPhononFrequency: features.logPhononFreq ?? null,
                coulombPseudopotential: 0.12,
                pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
                pairingMechanism: "phonon-mediated",
                correlationStrength: features.correlationStrength ?? null,
                dimensionality: variant.dimensionality,
                fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                uncertaintyEstimate: 0.7,
                verificationStage: 0,
                dataConfidence: "low",
              }, "motif_diffusion");
              if (inserted) totalScCandidates++;
            } catch (e) { console.error("[Engine] Mutation variant insert failed:", e); }
          }
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Novel prototype generation error", detail: err.message?.slice(0, 150), dataSource: "Novel Prototype Generator" });
      }
    }

    if (shouldContinue() && cycleCount % 10 === 0) {
      try {
        const { runStructureLearningCycle } = await import("../crystal/structure-learning-loop");
        const loopResult = await runStructureLearningCycle(8);
        if (loopResult.candidatesPassed > 0 || loopResult.modelsRetrained) {
          emit("log", {
            phase: "phase-11",
            event: "Structure learning loop cycle",
            detail: `Cycle #${loopResult.cycleId}: generated=${loopResult.candidatesGenerated}, screened=${loopResult.candidatesScreened}, passed=${loopResult.candidatesPassed}, failed=${loopResult.candidatesFailed}${loopResult.modelsRetrained ? `, retrained=[${loopResult.retrainedModels.join(",")}]` : ""}${loopResult.bestCandidate ? `, best=${loopResult.bestCandidate.formula} Tc=${loopResult.bestCandidate.predictedTc.toFixed(1)}K` : ""}`,
            dataSource: "Structure Learning Loop",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Structure learning loop error", detail: err.message?.slice(0, 150), dataSource: "Structure Learning Loop" });
      }
    }

    if (shouldContinue() && cycleCount % 15 === 0) {
      try {
        const topForEvo = await storage.getSuperconductorCandidates(20);
        const evoInput = topForEvo.map((c: any) => ({
          formula: c.formula,
          predictedTc: c.predictedTc ?? 0,
          ensembleScore: c.ensembleScore ?? 0,
        }));
        const evoResults = await runEvolutionaryStructureSearch(evoInput, emit);
        let evoInserted = 0;
        for (const evoFormula of evoResults) {
          if (!isValidFormula(evoFormula)) continue;
          const existingSC = await storage.getSuperconductorByFormula(evoFormula);
          if (!existingSC) {
            const features = extractFeatures(evoFormula);
            const gbResult = gbPredict(features);
            const lambdaML = features.electronPhononLambda ?? 0;
            const metallicityML = features.metallicity ?? 0.5;
            let rawTc = estimateRawTc(lambdaML, features.logPhononFreq);
            rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, evoFormula);
            const id = `sc-evo-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            try {
              const inserted = await insertCandidateWithStabilityCheck({
                id,
                name: evoFormula,
                formula: evoFormula,
                predictedTc: rawTc,
                pressureGpa: estimateFamilyPressure(evoFormula),
                meissnerEffect: false,
                zeroResistance: false,
                cooperPairMechanism: "Evolutionary structure search",
                crystalStructure: null,
                quantumCoherence: 0.5,
                stabilityScore: features.cooperPairStrength,
                synthesisPath: null,
                mlFeatures: features as any,
                xgboostScore: gbResult.score,
                neuralNetScore: 0.5,
                ensembleScore: Math.min(0.9, (gbResult.score + 0.5) / 2),
                roomTempViable: false,
                status: "theoretical",
                notes: `[Evolutionary structure search: mutated from top candidates]`,
                electronPhononCoupling: lambdaML || null,
                logPhononFrequency: features.logPhononFreq ?? null,
                coulombPseudopotential: 0.12,
                pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
                pairingMechanism: "phonon-mediated",
                correlationStrength: features.correlationStrength ?? null,
                dimensionality: "3D",
                fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                uncertaintyEstimate: 0.6,
                verificationStage: 0,
                dataConfidence: "low",
              }, "bo_exploration");
              if (inserted) {
                totalScCandidates++;
                evoInserted++;
              }
            } catch (e) { console.error("[Engine] Evolutionary structure insert failed:", e); }
          }
        }
        if (evoInserted > 0) {
          emit("log", {
            phase: "phase-11",
            event: "Evolutionary candidates inserted",
            detail: `${evoInserted} novel structures from evolutionary search`,
            dataSource: "Structure Evolution",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Evolutionary search error", detail: err.message?.slice(0, 150), dataSource: "Structure Evolution" });
      }
    }

    if (shouldContinue() && cycleCount % 5 === 0) {
      try {
        const diffResult = runDiffusionGenerationCycle(30);
        let diffInserted = 0;
        for (const crystal of diffResult.structures) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          const features = extractFeatures(normalized);
          const gbResult = gbPredict(features);
          const lambdaML = features.electronPhononLambda ?? 0;
          const metallicityML = features.metallicity ?? 0.5;
          let rawTc = estimateRawTc(lambdaML, features.logPhononFreq);
          rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, normalized);

          const id = `sc-diff-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          try {
            const inserted = await insertCandidateWithStabilityCheck({
              id,
              name: normalized,
              formula: normalized,
              predictedTc: rawTc,
              pressureGpa: estimateFamilyPressure(normalized),
              meissnerEffect: false,
              zeroResistance: false,
              cooperPairMechanism: `Diffusion-generated ${crystal.spaceGroup} (${crystal.crystalSystem})`,
              crystalStructure: `${crystal.spaceGroup} (${crystal.crystalSystem})`,
              quantumCoherence: crystal.noveltyScore,
              stabilityScore: features.cooperPairStrength,
              synthesisPath: null,
              mlFeatures: features as any,
              xgboostScore: gbResult.score,
              neuralNetScore: crystal.noveltyScore,
              ensembleScore: Math.min(0.9, (gbResult.score + crystal.noveltyScore) / 2),
              roomTempViable: false,
              status: "theoretical",
              notes: `[Crystal diffusion: ${crystal.prototypeMatch || "novel"}, SG=${crystal.spaceGroup}, density=${crystal.densityGcm3} g/cm3, novelty=${crystal.noveltyScore.toFixed(2)}]`,
              electronPhononCoupling: features.electronPhononLambda ?? null,
              logPhononFrequency: features.logPhononFreq ?? null,
              coulombPseudopotential: 0.12,
              pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
              pairingMechanism: "phonon-mediated",
              correlationStrength: features.correlationStrength ?? null,
              dimensionality: "3D",
              fermiSurfaceTopology: features.fermiSurfaceType ?? null,
              uncertaintyEstimate: crystal.prototypeMatch ? 0.55 : 0.7,
              verificationStage: 0,
              dataConfidence: "low",
            }, "motif_diffusion");
            if (inserted) {
              totalScCandidates++;
              diffInserted++;
              bayesianOptimizer.addObservation(normalized, rawTc, lambdaML, crystal.noveltyScore);
              if (candidateGeneratorSource.size < MAX_GENERATOR_SOURCE_ENTRIES) candidateGeneratorSource.set(normalized, "motif_diffusion");
            }
          } catch (e) { console.error("[Engine] Motif diffusion candidate insert failed:", e); }
        }
        if (diffInserted > 0 || diffResult.structures.length > 0) {
          emit("log", {
            phase: "phase-11",
            event: "Crystal diffusion generation",
            detail: `Generated ${diffResult.structures.length} structures (${diffResult.stats.novel} novel), inserted ${diffInserted}. Avg novelty: ${diffResult.stats.avgNovelty}, protos: ${Object.entries(diffResult.stats.protoBreakdown).map(([k, v]) => `${k}:${v}`).join(", ")}`,
            dataSource: "Crystal Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Crystal diffusion error", detail: err.message?.slice(0, 150), dataSource: "Crystal Diffusion" });
      }
    }

    if (shouldContinue() && cycleCount % 8 === 0) {
      try {
        const targetTcForDiffusion = autonomousBestTc > 100 ? autonomousBestTc * 1.5 : 200;
        const cdvaeCrystals = runCrystalDiffusionCycle(15, targetTcForDiffusion, 25, alreadyScreenedFormulas);
        let cdvaeInserted = 0;
        for (const crystal of cdvaeCrystals) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          const features = extractFeatures(normalized);
          const gbResult = gbPredict(features);
          const cdvaeLambda = crystal.lambda > 0 ? crystal.lambda : (features.electronPhononLambda ?? 0);
          const cdvaePhysTc = cdvaeLambda > 0
            ? Math.round(computePhysicsOnlyTc(cdvaeLambda, features.logPhononFreq))
            : 0;
          const cappedTc = applyAmbientTcCap(cdvaePhysTc, cdvaeLambda, estimateFamilyPressure(normalized), features.metallicity ?? 0.5, normalized);

          const id = `sc-cdvae-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          try {
            const inserted = await insertCandidateWithStabilityCheck({
              id,
              name: normalized,
              formula: normalized,
              predictedTc: cappedTc,
              pressureGpa: estimateFamilyPressure(normalized),
              meissnerEffect: false,
              zeroResistance: false,
              cooperPairMechanism: `CDVAE diffusion: ${crystal.motif} (${crystal.spaceGroup})`,
              crystalStructure: `${crystal.spaceGroup} (${crystal.crystalSystem}) ${crystal.atoms.length} atoms`,
              quantumCoherence: crystal.noveltyScore,
              stabilityScore: crystal.stabilityScore,
              synthesisPath: null,
              mlFeatures: features as any,
              xgboostScore: gbResult.score,
              neuralNetScore: crystal.noveltyScore,
              ensembleScore: Math.min(0.9, (gbResult.score + crystal.stabilityScore) / 2),
              roomTempViable: false,
              status: "theoretical",
              notes: `[CDVAE crystal diffusion: motif=${crystal.motif}, SG=${crystal.spaceGroup}, atoms=${crystal.atoms.length}, lattice=a${crystal.lattice.a.toFixed(1)}b${crystal.lattice.b.toFixed(1)}c${crystal.lattice.c.toFixed(1)}, lambda=${crystal.lambda}, novelty=${crystal.noveltyScore}]`,
              electronPhononCoupling: crystal.lambda > 0 ? crystal.lambda : null,
              logPhononFrequency: features.logPhononFreq ?? null,
              coulombPseudopotential: 0.12,
              pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
              pairingMechanism: "phonon-mediated",
              correlationStrength: features.correlationStrength ?? null,
              dimensionality: "3D",
              fermiSurfaceTopology: features.fermiSurfaceType ?? null,
              uncertaintyEstimate: 0.6,
              verificationStage: 0,
              dataConfidence: "low",
            }, "structure_diffusion");
            if (inserted) {
              totalScCandidates++;
              cdvaeInserted++;
              bayesianOptimizer.addObservation(normalized, cappedTc, crystal.lambda, crystal.noveltyScore);
              if (candidateGeneratorSource.size < MAX_GENERATOR_SOURCE_ENTRIES) candidateGeneratorSource.set(normalized, "structure_diffusion");
              incorporateSuccessData(normalized, cappedTc);
            }
          } catch (e) { console.error("[Engine] CDVAE crystal insert failed:", e); }
        }
        if (cdvaeCrystals.length > 0) {
          const cdvaeStats = getCrystalDiffusionStats();
          emit("log", {
            phase: "phase-11",
            event: "CDVAE crystal structure diffusion",
            detail: `Generated ${cdvaeCrystals.length} full crystal structures, inserted ${cdvaeInserted}. Best: ${cdvaeStats.bestFormula} Tc=${cdvaeStats.bestTc.toFixed(1)}K (${cdvaeStats.bestMotif}). Motifs used: ${Object.keys(cdvaeStats.motifBreakdown).join(", ")}`,
            dataSource: "CDVAE Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "CDVAE diffusion error", detail: err.message?.slice(0, 150), dataSource: "CDVAE Diffusion" });
      }
    }

    if (shouldContinue() && cycleCount % 10 === 0) {
      try {
        const distTarget = autonomousBestTc > 100 ? autonomousBestTc * 1.4 : 200;
        const distCrystals = runDistributionBasedDiffusion(12, distTarget, 25, alreadyScreenedFormulas);
        let distInserted = 0;
        for (const crystal of distCrystals) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;
          const features = extractFeatures(normalized);
          const gbResult = gbPredict(features);
          const distLambda = crystal.lambda > 0 ? crystal.lambda : (features.electronPhononLambda ?? 0);
          const distPhysTc = distLambda > 0
            ? Math.round(computePhysicsOnlyTc(distLambda, features.logPhononFreq))
            : 0;
          const cappedTc = applyAmbientTcCap(distPhysTc, distLambda, estimateFamilyPressure(normalized), features.metallicity ?? 0.5, normalized);
          const id = `sc-distdiff-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          try {
            const inserted = await insertCandidateWithStabilityCheck({
              id,
              name: normalized,
              formula: normalized,
              predictedTc: cappedTc,
              pressureGpa: estimateFamilyPressure(normalized),
              meissnerEffect: false,
              zeroResistance: false,
              cooperPairMechanism: `Distribution diffusion: ${crystal.motif} (${crystal.spaceGroup})`,
              crystalStructure: `${crystal.spaceGroup} (${crystal.crystalSystem}) ${crystal.atoms.length} atoms`,
              quantumCoherence: crystal.noveltyScore,
              stabilityScore: crystal.stabilityScore,
              synthesisPath: null,
              mlFeatures: features as any,
              xgboostScore: gbResult.score,
              neuralNetScore: crystal.noveltyScore,
              ensembleScore: Math.min(0.9, (gbResult.score + crystal.stabilityScore) / 2),
              roomTempViable: false,
              status: "theoretical",
              notes: `[Distribution-based crystal diffusion: system=${crystal.crystalSystem}, SG=${crystal.spaceGroup}, atoms=${crystal.atoms.length}, lambda=${crystal.lambda}]`,
              electronPhononCoupling: crystal.lambda > 0 ? crystal.lambda : null,
              logPhononFrequency: features.logPhononFreq ?? null,
              coulombPseudopotential: 0.12,
              pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
              pairingMechanism: "phonon-mediated",
              correlationStrength: features.correlationStrength ?? null,
              dimensionality: "3D",
              fermiSurfaceTopology: features.fermiSurfaceType ?? null,
              uncertaintyEstimate: 0.6,
              verificationStage: 0,
              dataConfidence: "low",
            }, "structure_diffusion");
            if (inserted) {
              totalScCandidates++;
              distInserted++;
              bayesianOptimizer.addObservation(normalized, cappedTc, crystal.lambda, crystal.noveltyScore);
              if (candidateGeneratorSource.size < MAX_GENERATOR_SOURCE_ENTRIES) candidateGeneratorSource.set(normalized, "structure_diffusion");
              incorporateSuccessData(normalized, cappedTc);
            }
          } catch (e) { console.error("[Engine] Distribution diffusion insert failed:", e); }
        }
        if (distCrystals.length > 0) {
          const dStats = getDistributionDiffusionStats();
          emit("log", {
            phase: "phase-11",
            event: "Distribution-based crystal diffusion",
            detail: `Generated ${distCrystals.length} crystals from learned distributions, inserted ${distInserted}. DB covers ${dStats.crystallographicDB.totalSpaceGroups} space groups, ${dStats.crystallographicDB.totalSystems} systems. Best: ${dStats.bestFormula} Tc=${dStats.bestTc.toFixed(1)}K (${dStats.bestCrystalSystem})`,
            dataSource: "Distribution Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Distribution diffusion error", detail: err.message?.slice(0, 150), dataSource: "Distribution Diffusion" });
      }
    }

    if (shouldContinue() && cycleCount % 12 === 0) {
      try {
        const topFormulas: string[] = [];
        const allCandidates = await storage.getSuperconductors(10, 0);
        for (const c of allCandidates) {
          if (c.predictedTc && c.predictedTc > 30 && c.formula) {
            topFormulas.push(c.formula);
          }
        }
        if (topFormulas.length >= 3) {
          trainVAE(topFormulas, 10);
        }
        const vaeResult = runLatentSpaceInverseDesign(
          autonomousBestTc > 100 ? autonomousBestTc * 1.3 : 200,
          topFormulas.length > 0 ? topFormulas[0] : undefined,
          30,
          0.02,
          2,
        );
        let vaeInserted = 0;
        for (const formula of vaeResult.decodedFormulas) {
          if (!isValidFormula(formula)) continue;
          const normalized = normalizeFormula(formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;
          try {
            const features = extractFeatures(normalized);
            const gbResult = gbPredict(features);
            const vaeLambda = features.electronPhononLambda ?? 0;
            const vaePhysTc = vaeLambda > 0
              ? Math.round(computePhysicsOnlyTc(vaeLambda, features.logPhononFreq))
              : 0;
            const cappedTc = applyAmbientTcCap(vaePhysTc, vaeLambda, estimateFamilyPressure(normalized), features.metallicity ?? 0.5, normalized);
            const inserted = await insertCandidateWithStabilityCheck({
              formula: normalized,
              predictedTc: cappedTc,
              dataConfidence: "low",
              ensembleScore: Math.min(0.9, gbResult.score),
              verificationStage: 0,
              notes: `[VAE inverse design: target=${vaeResult.bestTc}K, steps=${vaeResult.optimizationSteps}, converged=${vaeResult.converged}]`,
              cooperPairMechanism: "VAE latent-space gradient descent",
              status: "theoretical",
            }, "inverse_design");
            if (inserted) {
              totalScCandidates++;
              vaeInserted++;
              incorporateSuccessData(normalized, cappedTc);
            }
          } catch (e) { console.error("[Engine] VAE inverse design insert failed:", e); }
        }
        const vStats = getVAEStats();
        emit("log", {
          phase: "phase-11",
          event: "VAE latent-space inverse design",
          detail: `Best: ${vaeResult.bestFormula} Tc=${vaeResult.bestTc}K, steps=${vaeResult.optimizationSteps}, converged=${vaeResult.converged}, decoded=${vaeResult.decodedFormulas.length}, inserted=${vaeInserted}. VAE convergence rate: ${(vStats.convergenceRate * 100).toFixed(0)}%`,
          dataSource: "Crystal VAE",
        });
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "VAE inverse design error", detail: err.message?.slice(0, 150), dataSource: "Crystal VAE" });
      }
    }

    if (shouldContinue() && cycleCount % 7 === 0) {
      try {
        const structResult = runStructureDiffusionCycle(200, 3, 3);
        let structInserted = 0;
        for (const formula of structResult.formulas) {
          if (!isValidFormula(formula)) continue;
          const normalized = normalizeFormula(formula);
          const existing = await storage.getSuperconductorByFormula(normalized);
          if (existing) continue;

          try {
            const features = extractFeatures(normalized);
            const gbResult = gbPredict(features);
            if (gbResult.tcPredicted >= 10) {
              const inserted = await insertCandidateWithStabilityCheck({
                formula: normalized,
                predictedTc: Math.round(gbResult.tcPredicted),
                dataConfidence: "low",
                ensembleScore: Math.min(0.9, gbResult.score),
                verificationStage: 0,
                notes: `[structure-first: motif-designed, target=200K]`,
              }, "structure_diffusion");
              if (inserted) {
                totalScCandidates++;
                structInserted++;
              }
            }
          } catch (e) { console.error("[Engine] Structure-first design insert failed:", e); }
        }
        if (structResult.formulas.length > 0) {
          emit("log", {
            phase: "phase-11",
            event: "Structure-first design",
            detail: `Generated ${structResult.formulas.length} from motifs [${structResult.motifsUsed.join(", ")}], inserted ${structInserted}, best=${structResult.bestFormula} Tc=${structResult.bestTc.toFixed(1)}K`,
            dataSource: "Structure Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "phase-11", event: "Structure-first design error", detail: err.message?.slice(0, 150), dataSource: "Structure Diffusion" });
      }
    }

    const csCount = await storage.getCrystalStructureCount();
    const progress11 = computeProgress(11, csCount);
    await updatePhaseStatus(11, "active", progress11, csCount);
  } finally {
    activeTasks.delete("Crystal Structure Prediction");
    broadcast("taskEnd", { task: "Crystal Structure Prediction" });
  }
}

async function runPhase12_MultiFidelity() {
  if (!shouldContinue()) return;
  activeTasks.add("Multi-Fidelity Screening");
  broadcast("taskStart", { task: "Multi-Fidelity Screening" });
  try {
    await updatePhaseStatus(12, "active", 0, 0);
    if (!shouldContinue()) return;

    const stage0 = await storage.getSuperconductorsByStage(0, 50);
    const stage1 = await storage.getSuperconductorsByStage(1, 50);
    const eligible = [...stage0, ...stage1].filter(c => (c.ensembleScore ?? 0) > 0.25);
    const unscreened = shuffle(eligible).slice(0, 8);

    if (unscreened.length > 0) {
      const results = await runMultiFidelityPipeline(emit, unscreened);
      totalPipelineScreened += results.length;

      const passedCount = results.filter(r => r.passed).length;
      if (passedCount > 0) {
        allInsights.push(`Multi-fidelity pipeline: ${passedCount}/${results.length} candidates passed all 5 stages`);
      }

      for (const r of results.filter(pr => pr.passed)) {
        try {
          const candidate = unscreened.find(c => c.id === r.candidateId);
          if (!candidate) continue;
          const pathwayResult = triggerSynthesisPathwayForCandidate(
            r.formula,
            candidate.predictedTc ?? 0,
            r.finalStage,
          );
          if (pathwayResult && pathwayResult.bestRoute) {
            emit("log", {
              phase: "phase-12",
              event: "Synthesis pathway computed",
              detail: `${r.formula}: best=${pathwayResult.bestRoute.routeName} (${(pathwayResult.bestRoute.feasibilityScore * 100).toFixed(1)}%), ${pathwayResult.routes.length} routes, max ${pathwayResult.bestRoute.maxTemperature} C`,
              dataSource: "Synthesis Pathway Engine",
            });
            const existingPath = (candidate.synthesisPath as any) ?? {};
            const existingRoutes = Array.isArray(existingPath?.routes) ? existingPath.routes : [];
            await storage.updateSuperconductorCandidate(candidate.id, {
              synthesisPath: {
                routes: [...existingRoutes, ...pathwayResult.routes.slice(0, 3).map(route => ({
                  ...route,
                  source: "reaction-pathway-engine",
                }))],
                lastUpdated: new Date().toISOString(),
              },
            });
          }
        } catch (e) { console.error("[Engine] Reaction pathway failed:", e); }
      }
    }

    const crCount = await storage.getComputationalResultCount();
    const progress12 = computeProgress(12, crCount);
    await updatePhaseStatus(12, "active", progress12, crCount);
  } finally {
    activeTasks.delete("Multi-Fidelity Screening");
    broadcast("taskEnd", { task: "Multi-Fidelity Screening" });
  }
}

async function runPhase13_SynthesisReasoning() {
  if (!shouldContinue()) return;
  activeTasks.add("Novel Synthesis Reasoning");
  broadcast("taskStart", { task: "Novel Synthesis Reasoning" });
  try {
    const topCandidates = await storage.getSuperconductorCandidatesByTc(20);
    const eligible = topCandidates.filter(c =>
      (c.verificationStage ?? 0) >= 2 || (c.predictedTc ?? 0) > 100
    );
    const toProcess = eligible.slice(0, 3);

    if (toProcess.length === 0) return;

    let proposed = 0;
    for (const candidate of toProcess) {
      if (!shouldContinue()) return;
      try {
        const existingPath = candidate.synthesisPath as any;
        const hasPhysicsReasoned = Array.isArray(existingPath?.routes)
          && existingPath.routes.some((r: any) => r.source === "physics-reasoned");
        if (hasPhysicsReasoned) continue;
        const hasReasoningFailed = existingPath?.reasoningFailed === true;
        if (hasReasoningFailed) continue;

        const routes = await runSynthesisReasoning(emit, candidate);
        const allNewRoutes = routes && routes.length > 0 ? [...routes] : [];

        const hasAnalogyTransfer = Array.isArray(existingPath?.routes)
          && existingPath.routes.some((r: any) => r.source === "analogy-transfer");
        if (!hasAnalogyTransfer) {
          try {
            const { proposeAnalogousRoutes } = require("../synthesis/synthesis-analogy-engine");
            const analogyResult = await proposeAnalogousRoutes(candidate.formula);
            if (analogyResult.routes.length > 0) {
              const analogyRoutes = analogyResult.routes.slice(0, 3).map((r: any) => ({
                ...r,
                source: "analogy-transfer",
              }));
              allNewRoutes.push(...analogyRoutes);
              const analogueDetail = analogyResult.analogues.slice(0, 2)
                .map((a: any) => `${a.sourceFormula}(${(a.similarity * 100).toFixed(0)}%)`)
                .join(", ");
              const rxnDetail = analogyResult.reactionsApplied.slice(0, 2)
                .map((r: any) => r.reactionType)
                .join(", ");
              emit("log", {
                phase: "phase-13",
                event: "Synthesis analogy transfer",
                detail: `${candidate.formula}: ${analogyRoutes.length} routes from analogues [${analogueDetail}]${rxnDetail ? ` + reactions [${rxnDetail}]` : ""}`,
                dataSource: "Synthesis Analogy Engine",
              });
            }
          } catch (err: any) {
            emit("log", {
              phase: "phase-13",
              event: "Analogy transfer skipped",
              detail: `${candidate.formula}: ${err.message?.slice(0, 80)}`,
              dataSource: "Synthesis Analogy Engine",
            });
          }
        }

        const hasPlannedRoutes = Array.isArray(existingPath?.routes)
          && existingPath.routes.some((r: any) => r.source === "synthesis-planner");
        if (!hasPlannedRoutes) {
          try {
            const formationEnergy = typeof candidate.formationEnergy === "number" ? candidate.formationEnergy : undefined;
            const planResult = planAndTrack(candidate.formula, { formationEnergy: formationEnergy ?? null, maxRoutes: 5 });
            const plannedRoutes = planResult.routes;
            if (plannedRoutes.length > 0) {
              const plannerRoutes = plannedRoutes.slice(0, 3).map((r: any) => ({
                routeId: r.routeId,
                routeName: r.routeName,
                method: r.method,
                feasibilityScore: r.feasibilityScore,
                precursors: r.precursors,
                steps: (r.steps ?? []).map((s: any) => `${s.reactionType}: ${s.reactants?.join("+")} at ${s.temperature ?? 0}K, ${s.pressure ?? 0} GPa`),
                stepCount: r.steps?.length ?? 0,
                maxTemperature: r.maxTemperature,
                maxPressure: r.maxPressure,
                difficulty: r.difficulty,
                estimatedYield: r.estimatedYield,
                totalDuration: r.totalDuration,
                source: "synthesis-planner",
              }));
              allNewRoutes.push(...plannerRoutes);
              const bestPlanned = plannedRoutes[0];
              emit("log", {
                phase: "phase-13",
                event: "Synthesis route planned",
                detail: `${candidate.formula}: ${plannedRoutes.length} routes via planner, best=${bestPlanned.routeName} (feasibility=${(bestPlanned.feasibilityScore * 100).toFixed(1)}%, method=${bestPlanned.method})`,
                dataSource: "Synthesis Planner",
              });
              try {
                await storage.insertSynthesisProcess({
                  id: `sp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
                  materialId: candidate.id,
                  materialName: candidate.formula,
                  formula: candidate.formula,
                  method: bestPlanned.method,
                  conditions: { temperature: bestPlanned.maxTemperature, pressure: bestPlanned.maxPressure, atmosphere: bestPlanned.steps[0]?.atmosphere ?? "argon" },
                  steps: bestPlanned.steps.map((s: any) => `Step ${s.stepNumber}: ${s.reactionType} at ${s.temperature ?? 0}K, ${s.pressure ?? 0} GPa — ${s.notes ?? ""}`),
                  precursors: bestPlanned.precursors,
                  equipment: bestPlanned.equipmentList,
                  difficulty: bestPlanned.difficulty,
                  timeEstimate: bestPlanned.totalDuration,
                  safetyNotes: Array.isArray(bestPlanned.safetyNotes) ? bestPlanned.safetyNotes.join("; ") : (bestPlanned.safetyNotes ?? ""),
                  yieldPercent: Math.round((bestPlanned.feasibilityScore ?? 0.5) * 80),
                });
              } catch (_) {}
              for (const route of plannedRoutes.slice(0, 2)) {
                try {
                  const equation = `${route.precursors.join(" + ")} → ${candidate.formula}`;
                  await storage.insertChemicalReaction({
                    id: `rxn-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
                    name: `${route.method} synthesis of ${candidate.formula}`,
                    equation,
                    reactionType: route.method,
                    reactants: route.precursors.map((p: string) => ({ formula: p, role: "precursor" })),
                    products: [{ formula: candidate.formula, role: "product" }],
                    conditions: { temperature: route.maxTemperature, pressure: route.maxPressure },
                    energetics: {},
                    mechanism: route.method,
                    relevanceToSuperconductor: Math.min(1, (candidate.predictedTc ?? 0) / 300),
                    source: "synthesis-planner",
                  });
                } catch (_) {}
              }
            }
          } catch (err: any) {
            console.error(`[Engine] Synthesis planner failed for ${candidate.formula}:`, err.message?.slice(0, 100));
          }
        }

        const hasHeuristicRoutes = Array.isArray(existingPath?.routes)
          && existingPath.routes.some((r: any) => r.source === "heuristic-generator");
        if (!hasHeuristicRoutes) {
          try {
            const heuristicRoutes = generateHeuristicRoutes(candidate.formula, true);
            if (heuristicRoutes.length > 0) {
              const hRoutes = heuristicRoutes.slice(0, 3).map((r: any) => ({
                routeName: `${r.rule}: ${r.method}`,
                method: r.method,
                rule: r.rule,
                precursors: r.precursors,
                steps: r.steps,
                equation: r.equation,
                temperature: r.temperature,
                pressure: r.pressure,
                atmosphere: r.atmosphere,
                difficulty: r.difficulty,
                confidence: r.confidence,
                notes: r.notes,
                source: "heuristic-generator",
              }));
              allNewRoutes.push(...hRoutes);
              emit("log", {
                phase: "phase-13",
                event: "Heuristic synthesis routes generated",
                detail: `${candidate.formula}: ${heuristicRoutes.length} rule-based routes [${heuristicRoutes.map(r => r.rule).join(", ")}]`,
                dataSource: "Heuristic Synthesis Generator",
              });
            }
          } catch (err: any) {
            console.error(`[Engine] Heuristic synthesis generator failed for ${candidate.formula}:`, err.message?.slice(0, 100));
          }
        }

        if (allNewRoutes.length > 0) {
          const existingRoutes = Array.isArray(existingPath?.routes) ? existingPath.routes : [];
          const taggedExisting = existingRoutes.map((r: any) => ({
            ...r,
            source: r.source || "literature-based",
          }));
          await storage.updateSuperconductorCandidate(candidate.id, {
            synthesisPath: {
              routes: [...taggedExisting, ...allNewRoutes],
              lastUpdated: new Date().toISOString(),
            },
          });
          proposed += allNewRoutes.length;
          totalNovelSynthesisProposed += allNewRoutes.length;
        }
      } catch (err: any) {
        emit("log", {
          phase: "phase-13",
          event: "Synthesis reasoning candidate error",
          detail: `${candidate.formula}: ${err.message?.slice(0, 100)}`,
          dataSource: "Synthesis Reasoning",
        });
        try {
          const failPath = candidate.synthesisPath as any;
          await storage.updateSuperconductorCandidate(candidate.id, {
            synthesisPath: { ...(failPath || {}), reasoningFailed: true },
          });
        } catch (_) {}
      }
    }

    if (proposed > 0) {
      allInsights.push(`Novel synthesis reasoning: proposed ${proposed} physics-reasoned routes for ${toProcess.length} candidates`);
    }
  } finally {
    activeTasks.delete("Novel Synthesis Reasoning");
    broadcast("taskEnd", { task: "Novel Synthesis Reasoning" });
  }
}

async function runAutonomousDiscoveryCycle(formula: string): Promise<{ passed: boolean; tc: number; reason: string; physicsPred?: PhysicsPrediction }> {
  try {
    if (typeof formula !== "string") {
      formula = String(formula ?? "");
    }
    if (!formula || !isValidFormula(formula)) {
      pipelineStageMetrics.chemistryRejects++;
      return { passed: false, tc: 0, reason: "invalid-elements" };
    }

    if (!passesValenceFilter(formula)) {
      pipelineStageMetrics.chemistryRejects++;
      return { passed: false, tc: 0, reason: "valence-filter-failed" };
    }

    const existingCandidate = await storage.getSuperconductorByFormula(formula);
    if (existingCandidate) {
      pipelineStageMetrics.duplicateRejects++;
      return { passed: false, tc: existingCandidate.predictedTc ?? 0, reason: "duplicate" };
    }

    const stabilityScreen = passesStabilityPreFilter(formula);
    if (!stabilityScreen.pass) {
      pipelineStageMetrics.stabilityPrefilterRejects++;
      emit("log", {
        phase: "autonomous-loop",
        event: "Stability pre-filter rejected",
        detail: `${formula}: ${stabilityScreen.reason}`,
        dataSource: "Stability Predictor (GNN)",
      });
      return { passed: false, tc: 0, reason: `stability-prefilter: ${stabilityScreen.reason}` };
    }

    const surrogateResult = surrogateScreen(formula, 3);
    const surrogateTc = Number.isFinite(surrogateResult.predictedTc) ? surrogateResult.predictedTc : 0;
    if (!surrogateResult.pass) {
      pipelineStageMetrics.surrogateRejects++;
      return { passed: false, tc: surrogateTc, reason: `surrogate-reject: Tc=${surrogateTc}K, ${surrogateResult.reasoning.join("; ")}` };
    }

    try {
      const miedemaEf = computeMiedemaFormationEnergy(formula);
      if (miedemaEf < -4.0 || miedemaEf > 2.0) {
        pipelineStageMetrics.formationEnergyRejects++;
        console.log(`[Autonomous] ${formula}: Formation energy ${miedemaEf.toFixed(3)} eV/atom outside physical range [-4, 2], rejecting`);
        return { passed: false, tc: 0, reason: `formation-energy-insane: ${miedemaEf.toFixed(3)} eV/atom` };
      }
    } catch (e) { console.error(`[Autonomous] Formation energy check failed for ${formula}:`, e); }

    if (KNOWN_COMPOUNDS.has(normalizeFormula(formula))) {
      pipelineStageMetrics.duplicateRejects++;
      return { passed: false, tc: 0, reason: "known-compound" };
    }

    const family = classifyFamily(formula);
    const features = extractFeatures(formula);
    if (!features) {
      pipelineStageMetrics.featureExtractionFails++;
      return { passed: false, tc: 0, reason: "feature-extraction-failed" };
    }

    const gbResult = gbPredict(features);
    if (gbResult.tcPredicted < 5) {
      pipelineStageMetrics.gbTcRejects++;
      return { passed: false, tc: gbResult.tcPredicted, reason: "low-gb-tc" };
    }

    let gnnResult: ReturnType<typeof gnnPredictWithUncertainty> | null = null;
    try {
      gnnResult = gnnPredictWithUncertainty(formula, undefined, features.pressureGpa);
    } catch (e) { console.error(`[Autonomous] GNN prediction failed for ${formula}:`, e); }

    try {
      const mtPred = multiTaskPredict(formula);
      trackMultiTaskPrediction(mtPred);
    } catch (e) { console.error(`[Autonomous] Multi-task prediction failed for ${formula}:`, e); }

    let primaryTc = gbResult.tcPredicted;
    let ensembleConfidence = 0.3;
    if (gnnResult && gnnResult.confidence > 0.3) {
      primaryTc = gnnResult.tc * 0.6 + gbResult.tcPredicted * 0.4;
      ensembleConfidence = gnnResult.confidence * 0.6 + gbResult.score * 0.4;
    }

    crossEngineHub.recordInsight("ml", formula, {
      predictedTc: primaryTc,
      uncertainty: gnnResult?.uncertainty ?? 0.5,
      featureImportance: {},
      confidence: ensembleConfidence > 0.6 ? "high" : ensembleConfidence > 0.3 ? "medium" : "low",
    });

    const physicsPred = physicsPredictor.predict(features);
    const preFilter = physicsPredictor.preFilter(physicsPred);
    if (!preFilter.pass) {
      pipelineStageMetrics.physicsPrefilterRejects++;
      return { passed: false, tc: Math.round(primaryTc), reason: `physics-prefilter: ${preFilter.reason}`, physicsPred };
    }

    const candidatePressureEst = features.pressureGpa > 0 ? features.pressureGpa : estimateFamilyPressure(formula);

    const candidate = {
      formula,
      family,
      predictedTc: Math.round(primaryTc),
      confidence: "low" as const,
      source: "autonomous-loop",
      ensembleScore: Math.min(0.95, ensembleConfidence),
      pressureGpa: candidatePressureEst,
      verificationStage: 0,
    };

    pipelineStageMetrics.prototypeAttempts++;
    const structureBatch = await runStructurePredictionBatch(emit, [candidate as any]);
    const structureResult = Array.isArray(structureBatch) ? structureBatch[0] : undefined;
    if (structureResult) {
      pipelineStageMetrics.prototypeSuccesses++;
    }

    try {
      await runConvexHullAnalysis(emit, formula);
    } catch (e) { console.error(`[Autonomous] Convex hull analysis failed for ${formula}:`, e); }

    try {
      pipelineStageMetrics.xtbAttempts++;
      const phononCheck = await runXTBPhononCheck(formula);
      if (phononCheck) {
        pipelineStageMetrics.xtbSuccesses++;
        const severeInstability = (phononCheck as any).severeInstability;
        if (severeInstability) {
          pipelineStageMetrics.phononRejects++;
          const reason = (phononCheck as any).instabilityReason || "severe phonon instability";
          console.log(`[Autonomous] ${formula}: REJECTED — ${reason}`);
          return { passed: false, tc: 0, reason: `phonon-instability: ${reason}` };
        }
      }
    } catch (e) { console.error(`[Autonomous] xTB phonon check failed for ${formula}:`, e); }

    const physicsResult = await runFullPhysicsAnalysis(emit, candidate as any);
    crossEngineHub.recordInsight("physics", formula, {
      lambda: physicsResult.coupling.lambda,
      dosAtFermi: physicsResult.electronicStructure.densityOfStatesAtFermi,
      omegaLog: physicsResult.coupling.omegaLog,
      metallicity: physicsResult.electronicStructure.metallicity,
      correlationStrength: physicsResult.correlation?.ratio ?? 0,
      instabilityFlags: physicsResult.instabilityProximity?.nearestBoundary ? [physicsResult.instabilityProximity.nearestBoundary] : [],
    });
    const rawTc = physicsResult.eliashberg.predictedTc;
    const physicsTc = (Number.isFinite(rawTc) && rawTc > 0 && rawTc < 1000) ? rawTc : 0;
    const cappedPhysicsTcAuto = physicsTc > 0
      ? applyAmbientTcCap(Math.round(physicsTc), physicsResult.coupling.lambda, features.pressureGpa ?? 0, physicsResult.electronicStructure.metallicity ?? 0.5, formula)
      : 0;
    const reconciledAuto = reconcileTc({
      gbPredicted: gbResult.tcPredicted > 0 ? Math.round(gbResult.tcPredicted) : undefined,
      physicsTc: cappedPhysicsTcAuto > 0 ? cappedPhysicsTcAuto : undefined,
    });
    let finalTc = reconciledAuto.reconciledTc > 0 ? reconciledAuto.reconciledTc : (cappedPhysicsTcAuto > 0 ? cappedPhysicsTcAuto : Math.round(gbResult.tcPredicted));

    try {
      const hullDist = physicsResult.competingPhases.length > 0 ? 0.05 * physicsResult.competingPhases.length : 0.02;
      physicsPredictor.addTrainingSample(
        features,
        physicsResult.coupling.lambda,
        physicsResult.electronicStructure.densityOfStatesAtFermi,
        physicsResult.coupling.omegaLog,
        hullDist
      );
    } catch (e) { console.error(`[Autonomous] Physics predictor training failed for ${formula}:`, e); }

    let autonomousQC: QuantumCriticalAnalysis | undefined;
    try {
      autonomousQC = detectQuantumCriticality(formula, {
        electronic: physicsResult.electronicStructure,
        coupling: physicsResult.coupling,
      });
      if (autonomousQC.quantumCriticalScore > 0.5 && autonomousQC.pairingBoostFromQCP > 0.1) {
        const qcTcBoost = 1 + autonomousQC.pairingBoostFromQCP * 0.15;
        finalTc = Math.min(400, Math.round(finalTc * qcTcBoost));
      }
    } catch (e) { console.error(`[Autonomous] Quantum criticality detection failed for ${formula}:`, e); }

    try {
      const zoneInfo = getZoneBonus(formula);
      if (zoneInfo.inZone && zoneInfo.bonus > 0.01) {
        const zoneTcBoost = 1 + zoneInfo.bonus * 0.1;
        finalTc = Math.min(400, Math.round(finalTc * zoneTcBoost));
      }
    } catch (e) { console.error(`[Autonomous] Zone bonus check failed for ${formula}:`, e); }

    const synthesizabilityScore = structureResult?.synthesizability ?? 0.5;
    const lambda = physicsResult.coupling.lambda;
    const rawHullDist = physicsResult.competingPhases.length > 0 ? 0.05 * physicsResult.competingPhases.length : 0.02;
    const stabilityHullDist = Math.min(rawHullDist, 0.20);

    let tier: 1 | 2 | 3 | 0 = 0;
    let confidenceLevel: "high" | "medium" | "low" = "low";
    let verificationStage = 0;

    if (finalTc > 70 && lambda > 1.2 && stabilityHullDist < 0.10) {
      tier = 1;
      confidenceLevel = "high";
      verificationStage = 2;
    } else if (finalTc > 25 && lambda > 0.5 && stabilityHullDist < 0.20) {
      tier = 2;
      confidenceLevel = "medium";
      verificationStage = 1;
    } else if (finalTc > 5 && lambda > 0.15) {
      tier = 3;
      confidenceLevel = "low";
      verificationStage = 0;
    }

    if (tier === 0) {
      pipelineStageMetrics.belowTierRejects++;
      return { passed: false, tc: finalTc, reason: "below-tier3-thresholds" };
    }

    if (synthesizabilityScore < 0.2 && tier > 1) {
      tier = Math.min(tier + 1, 3) as 2 | 3;
      if (tier === 3) confidenceLevel = "low";
    }

    const instProx = physicsResult.instabilityProximity;
    const pairingNote = `[Pairing: ${physicsResult.pairingAnalysis.dominant.mechanism} (Tc=${physicsResult.pairingAnalysis.dominant.tcEstimate.toFixed(0)}K)]`;
    const instNote = `[Instability: ${instProx.nearestBoundary}=${instProx.overallProximity.toFixed(2)}]`;
    const tierNote = `[Tier ${tier}]`;

    const normalizedFormula = normalizeFormula(formula);
    const candidatePayload = {
      id: `sc-auto-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name: normalizedFormula,
      formula: normalizedFormula,
      predictedTc: finalTc,
      dataConfidence: confidenceLevel,
      source: "autonomous-loop",
      family,
      electronPhononCoupling: physicsResult.coupling.lambda,
      logPhononFrequency: physicsResult.coupling.omegaLog,
      coulombPseudopotential: physicsResult.coupling.muStar,
      correlationStrength: physicsResult.correlation.ratio,
      fermiSurfaceTopology: physicsResult.electronicStructure.fermiSurfaceTopology,
      dimensionality: physicsResult.dimensionality,
      competingPhases: physicsResult.competingPhases as any,
      upperCriticalField: physicsResult.criticalFields.upperCriticalField,
      coherenceLength: physicsResult.criticalFields.coherenceLength,
      londonPenetrationDepth: physicsResult.criticalFields.londonPenetrationDepth,
      pairingMechanism: physicsResult.pairingAnalysis.dominant.mechanism,
      cooperPairMechanism: physicsResult.pairingAnalysis.dominant.description,
      ensembleScore: Math.min(0.95, (() => {
        const gnnScore = gnnResult ? (Math.min(1, gnnResult.tc > 100 ? 0.8 : gnnResult.tc > 20 ? 0.5 : 0.2) * gnnResult.confidence) : 0;
        const gbScore = gbResult.score;
        const noveltyBonus = synthesizabilityScore * 0.1;
        return gnnScore > 0 ? (gnnScore * 0.6 + gbScore * 0.3 + noveltyBonus) : (0.3 + (finalTc / 400) + (synthesizabilityScore * 0.2));
      })()),
      verificationStage,
      notes: `${pairingNote} ${instNote} ${tierNote} [autonomous-loop]${gnnResult ? ` [GNN: Tc=${gnnResult.tc}K, λ=${gnnResult.lambda}, conf=${(gnnResult.confidence * 100).toFixed(0)}%]` : ''}`,
      formationEnergy: (() => { try { return computeMiedemaFormationEnergy(normalizedFormula); } catch { return null; } })(),
      mlFeatures: {
        phononDOS: { totalStates: physicsResult.phononDOS.totalStates, binCount: physicsResult.phononDOS.frequencies.length },
        alpha2F: { integratedLambda: physicsResult.alpha2F.integratedLambda, binCount: physicsResult.alpha2F.frequencies.length },
        tier,
        dftConfidence: 0,
        ...(gnnResult ? { gnnTc: gnnResult.tc, gnnLambda: gnnResult.lambda, gnnUncertainty: gnnResult.uncertainty, gnnConfidence: gnnResult.confidence } : {}),
      } as any,
    };

    let autonomousInserted = false;
    try {
      autonomousInserted = await insertCandidateWithStabilityCheck(candidatePayload, "rl");
    } catch (insertErr: any) {
      const isDuplicate = insertErr?.message?.includes("duplicate") || insertErr?.code === "23505";
      if (isDuplicate) {
        autonomousInserted = true;
      } else {
        autonomousInserted = false;
      }
    }

    if (!autonomousInserted) {
      return { passed: false, tc: finalTc, reason: `insert-failed (λ=${lambda.toFixed(2)},tier=${tier})` };
    }

    totalScCandidates++;
    recentNewCandidates++;

    try {
      const memFingerprint = buildFingerprint(formula, finalTc, {
        lambda: physicsResult.coupling.lambda,
        metallicity: physicsResult.electronicStructure.metallicity,
        nestingScore: physicsResult.electronicStructure.nestingScore,
        vanHoveProximity: physicsResult.electronicStructure.vanHoveProximity,
        dimensionality: physicsResult.dimensionality,
        correlationStrength: physicsResult.correlation.ratio,
      });
      discoveryMemory.recordDiscovery(formula, memFingerprint, finalTc);
    } catch (e) { console.error(`[Autonomous] Discovery memory record failed for ${formula}:`, e); }

    try {
      buildAndStoreFeatureRecord(formula, finalTc);
      recordPrediction(formula, gbResult.tcPredicted, finalTc);
      recordCandidateOutcome(formula, true);
      updatePhysicsParameters(finalTc, gbResult.tcPredicted, [], formula);
    } catch (e) { console.error(`[Autonomous] Feature record/physics update failed for ${formula}:`, e); }

    try {
      const family = physicsResult.pairingAnalysis?.dominantMechanism ?? "unknown";
      addMaterialToDataset(formula, finalTc, physicsResult.coupling.lambda, physicsResult.pairingAnalysis?.symmetry ?? "unknown", family);
    } catch (e) { console.error(`[Autonomous] Landscape dataset add failed for ${formula}:`, e); }

    const candidatePressure = features.pressureGpa ?? 0;
    if (finalTc > 30 && candidatePressure > 5) {
      try {
        const pathway = getPathwayForCandidate(formula, finalTc, candidatePressure);
        if (pathway.bestAmbientTc > 20 && pathway.strategies.length > 0) {
          if (pathway.retentionPercent > 50) {
            const retentionBoost = 1 + (pathway.retentionPercent / 100) * 0.08;
            finalTc = Math.min(400, Math.round(finalTc * retentionBoost));
            feedbackLoopStats.pressurePathwayBoosts++;
          }
          if (pathway.bestAmbientTc > feedbackLoopStats.pressurePathwayBestAmbientTc) {
            feedbackLoopStats.pressurePathwayBestAmbientTc = pathway.bestAmbientTc;
            feedbackLoopStats.pressurePathwayBestFormula = pathway.bestAmbientFormula;
          }
          if (pathway.bestAmbientFormula && isValidFormula(pathway.bestAmbientFormula) && !alreadyScreenedFormulas.has(normalizeFormula(pathway.bestAmbientFormula))) {
            alreadyScreenedFormulas.add(normalizeFormula(pathway.bestAmbientFormula));
            try {
              const ppId = `sc-pp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
              await insertCandidateWithStabilityCheck({
                id: ppId,
                name: pathway.bestAmbientFormula,
                formula: pathway.bestAmbientFormula,
                predictedTc: pathway.bestAmbientTc,
                ensembleScore: Math.min(1, pathway.bestAmbientTc / 293 * 0.6),
                crystalStructure: "pressure-stabilized",
                status: "theoretical",
                notes: `Ambient variant of ${formula} (${pathway.retentionPercent}% Tc retention from ${candidatePressure}GPa). Strategy: ${pathway.strategies[0]?.type}`,
              }, "random_exploration");
            } catch (e) { console.error(`[Autonomous] Pressure pathway candidate insert failed for ${formula}:`, e); }
          }
          emit("log", {
            phase: "engine",
            event: "Pressure-to-ambient pathway",
            detail: `${formula} (Tc=${finalTc}K, P=${candidatePressure}GPa): best ambient candidate ${pathway.bestAmbientFormula} est. Tc=${pathway.bestAmbientTc}K (${pathway.retentionPercent}% retention). Strategy: ${pathway.strategies[0]?.type}. Feasibility: ${pathway.feasibility}. Variant added to pool.`,
            dataSource: "Pressure Pathway",
          });
        }
      } catch (e) { console.error(`[Autonomous] Pressure pathway analysis failed for ${formula}:`, e); }
    }

    if (finalTc > autonomousBestTc) {
      autonomousBestTc = finalTc;
    }

    return { passed: true, tc: finalTc, reason: `accepted-tier${tier}`, physicsPred };
  } catch (err: any) {
    return { passed: false, tc: 0, reason: `error: ${err.message?.slice(0, 80)}${err.stack ? ' @ ' + (err.stack.split('\n')[1] || '').trim().slice(0, 60) : ''}` };
  }
}

function generateFastPathFormulas(focusArea: string): string[] {
  const topCandidatesForGen: { formula: string; predictedTc?: number }[] = [];
  const scElements: Record<string, string[][]> = {
    Carbides: [["Nb","C"],["Ti","C"],["Mo","C"],["V","C"],["Zr","C"],["Hf","C"],["Ta","C"],["W","C"]],
    Borides: [["Nb","B"],["Ti","B"],["Zr","B"],["Mo","B"],["V","B"],["Ta","B"],["Hf","B"],["W","B"]],
    Nitrides: [["Nb","N"],["Ti","N"],["Zr","N"],["V","N"],["Mo","N"],["Ta","N"],["Hf","N"]],
    Hydrides: [["La","H"],["Y","H"],["Ca","H"],["Sr","H"],["Ba","H"],["Th","H"],["Sc","H"]],
    Intermetallics: [["Nb","Ge"],["Nb","Sn"],["V","Si"],["Nb","Al"],["Mo","Ge"],["V","Ga"]],
    Cuprates: [["Ba","Cu","O"],["La","Cu","O"],["Y","Ba","Cu","O"],["Bi","Sr","Cu","O"]],
  };
  const basePairs = scElements[focusArea] || scElements["Carbides"];
  for (const pair of basePairs) {
    topCandidatesForGen.push({ formula: pair.join(""), predictedTc: 20 });
  }
  const { formulas, stats } = runMassiveGeneration(topCandidatesForGen, focusArea);
  console.log(`Massive generation: ${stats.totalGenerated} generated, ${stats.uniqueAfterDedup} unique, ${stats.passedPreScreen} passed pre-screen`);
  return formulas;
}

async function runAutonomousFastPath() {
  if (!shouldContinue()) return;
  activeTasks.add("Autonomous Screening");
  broadcast("taskStart", { task: "Autonomous Screening" });

  try {
    const rlState = {
      bestTc: autonomousBestTc || (previousCycleMetrics?.bestTc ?? 0),
      avgRecentTc: previousCycleMetrics?.bestTc ?? 0,
      recentRewardTrend: 0,
      familyDiversity: previousCycleMetrics?.familyDiversity ?? 1,
      stagnationCycles: cyclesSinceTcImproved,
      explorationBudgetUsed: autonomousTotalScreened / Math.max(1, autonomousTotalScreened + 1000),
      elementSuccessEntropy: 0.5,
      cycleNumber: cycleCount,
    };

    const rlAction = rlAgent.selectAction(rlState);
    const rlDescription = rlAgent.getActionDescription(rlAction);

    const budget = allocateBudget(200);
    const rlSlots = budget.allocations["rl"] ?? 30;

    const rlCandidates = rlAgent.generateCandidatesFromAction(rlAction, rlSlots);

    let focusArea = currentStrategyFocusAreas[0]?.area || "Carbides";
    const EXPLORATION_FAMILIES = [
      "Pnictides", "Chalcogenides", "Cuprates", "Hydrides", "Kagome",
      "Sulfides", "Intermetallics", "Alloys", "Oxides", "Nitrides",
    ];
    const familyNameMap: Record<string, string> = {
      hydride: "Hydrides", cuprate: "Cuprates", chalcogenide: "Chalcogenides",
      "layered-pnictide": "Pnictides", intermetallic: "Intermetallics",
      "kagome-metal": "Kagome", "oxide-perovskite": "Oxides", boride: "Carbides",
    };
    const EXPLORATION_PROB = 0.15;
    if (latestTheoryBias && Object.keys(latestTheoryBias.familyPreferences).length > 0 && Math.random() < 0.5) {
      const topTheoryFamily = Object.entries(latestTheoryBias.familyPreferences)
        .sort((a, b) => b[1] - a[1])[0]?.[0];
      const mapped = topTheoryFamily ? familyNameMap[topTheoryFamily] : null;
      if (mapped) focusArea = mapped;
    } else if (Math.random() < EXPLORATION_PROB) {
      const explorationPool = EXPLORATION_FAMILIES.filter(f => f !== focusArea);
      focusArea = explorationPool[Math.floor(Math.random() * explorationPool.length)];
    }

    let constraintGuidance: ReturnType<typeof getConstraintGuidanceForGenerator> | null = null;
    let graphGuidance: ReturnType<typeof getConstraintGraphGuidance> | null = null;
    try {
      const targetTcForConstraints = Math.max(100, autonomousBestTc * 1.1, 200);
      constraintGuidance = getConstraintGuidanceForGenerator(targetTcForConstraints);
      if (cycleCount % 5 === 0) {
        graphGuidance = getConstraintGraphGuidance(targetTcForConstraints);
      }
    } catch (e) { console.error("[Engine] Constraint guidance failed:", e); }

    emit("log", {
      phase: "engine",
      event: "RL agent action",
      detail: `RL selected: ${rlDescription}. ${rlCandidates.length} RL candidates. Focus: ${focusArea}. Epsilon=${rlAgent.getStats().epsilon.toFixed(3)}, temp=${rlAgent.getStats().temperature.toFixed(3)}${constraintGuidance ? `. Constraints: lambda=[${constraintGuidance.lambdaRange[0].toFixed(2)},${constraintGuidance.lambdaRange[1].toFixed(2)}], feasibility=${constraintGuidance.feasibility.toFixed(2)}` : ""}${graphGuidance ? `. Graph: regimes=[${graphGuidance.topRegimes.join(",")}], rare=${graphGuidance.rareOpportunities.length}` : ""}`,
      dataSource: "RL Agent",
    });

    let topCandidatesForGen: { formula: string; predictedTc?: number }[] = [];
    try {
      const existingTop = await storage.getSuperconductorCandidatesByTc(20);
      topCandidatesForGen = existingTop.map(c => ({ formula: c.formula, predictedTc: c.predictedTc ?? 0 }));
    } catch (e) { console.error("[Engine] Top candidates fetch for generation failed:", e); }

    const shuffled = [...topCandidatesForGen].sort(() => Math.random() - 0.5);
    const { formulas: massiveCandidates, stats: genStats } = runMassiveGeneration(shuffled, focusArea);

    const boCandidatePool = [...new Set([...rlCandidates, ...massiveCandidates])];
    const boSuggestions = bayesianOptimizer.suggestNextCandidates(boCandidatePool, 50, "mixed");
    const boTopFormulas = boSuggestions.map(s => s.formula);

    const remainingMassive = massiveCandidates.filter(f => !boTopFormulas.includes(f));

    const pressureVariantFormulas: string[] = [];
    const candidatePressureMap = new Map<string, number>();
    for (const f of [...boTopFormulas, ...remainingMassive]) {
      const basePressure = estimateFamilyPressure(f);
      candidatePressureMap.set(f, basePressure);
      if (basePressure > 0 && Math.random() < 0.3) {
        const variants = generatePressureVariants(f, basePressure);
        for (const v of variants) {
          if (!candidatePressureMap.has(`${v.formula}@${v.pressureGpa}`)) {
            candidatePressureMap.set(`${v.formula}@${v.pressureGpa}`, v.pressureGpa);
            pressureVariantFormulas.push(v.formula);
          }
        }
      }
    }

    const candidates = [...boTopFormulas, ...remainingMassive, ...pressureVariantFormulas.slice(0, 20)];

    if (boSuggestions.length > 0) {
      const topBO = boSuggestions[0];
      emit("log", {
        phase: "engine",
        event: "Bayesian optimization ranking",
        detail: `BO ranked ${boSuggestions.length} candidates. Top: ${topBO.formula} (acq=${topBO.acquisitionValue.toFixed(2)}, mean=${topBO.predictedMean.toFixed(1)}K, std=${topBO.predictedStd.toFixed(2)}, source=${topBO.source}). GP observations: ${bayesianOptimizer.getStats().observationCount}`,
        dataSource: "Bayesian Optimizer",
      });
    }

    const constraintFiltered = constraintGuidedGenerate(candidates);
    const physicsCleanCandidates = [...constraintFiltered.valid, ...constraintFiltered.repaired];

    if (constraintFiltered.rejected.length > 0 || constraintFiltered.repaired.length > 0) {
      emit("log", {
        phase: "engine",
        event: "Physics constraint filter",
        detail: `${candidates.length} candidates → ${constraintFiltered.valid.length} valid, ${constraintFiltered.repaired.length} repaired, ${constraintFiltered.rejected.length} rejected. Violations: ${Object.entries(getConstraintEngineStats().violationCounts).map(([k, v]) => `${k}:${v}`).join(", ")}`,
        dataSource: "Physics Constraints",
      });
    }

    let inverseDesignCandidates: string[] = [];
    {
      try {
        let campaigns = getAllActiveCampaigns();
        if (campaigns.length === 0) {
          const campaign = createCampaign(`fastpath-200K-${Date.now()}`, {
            targetTc: 200,
            maxPressure: 50,
            minLambda: 0.5,
            maxHullDistance: 0.3,
            metallicRequired: true,
            phononStable: true,
            preferredElements: ["Nb", "Ti", "B", "C", "N", "La", "Y", "H"],
          }, 200);
          try {
            await storage.insertInverseDesignCampaign({
              id: campaign.id,
              targetTc: 200,
              targetPressure: 0,
              status: "active",
              cyclesRun: 0,
              bestTcAchieved: 0,
              bestDistance: 1,
              candidatesGenerated: 0,
              candidatesPassedPipeline: 0,
              learningState: {} as any,
              convergenceHistory: [],
              topCandidates: [],
            });
          } catch (e) { console.error("[Engine] Auto-create inverse campaign failed:", e); }
          campaigns = getAllActiveCampaigns();
          emit("log", {
            phase: "engine",
            event: "Auto-created inverse design campaign",
            detail: `No active campaigns found. Created fastpath campaign targeting 200K.`,
            dataSource: "Inverse Optimizer",
          });
        }

        const inverseSeen = new Set<string>();
        for (const campaign of campaigns) {
          const inverseCandidates = runInverseCycle(campaign);
          for (const ic of inverseCandidates) {
            if (ic.formula && isValidFormula(ic.formula)) {
              const norm = normalizeFormula(ic.formula);
              if (!alreadyScreenedFormulas.has(norm) && !inverseSeen.has(norm)) {
                inverseSeen.add(norm);
                alreadyScreenedFormulas.add(norm);
                inverseDesignCandidates.push(norm);
              }
            }
          }

          const gradResult = runGradientDescentCycle(campaign.target, 4, 12);
          for (const r of gradResult.results) {
            if (r.finalTc > 10 && isValidFormula(r.finalFormula)) {
              const norm = normalizeFormula(r.finalFormula);
              if (!alreadyScreenedFormulas.has(norm) && !inverseSeen.has(norm)) {
                inverseSeen.add(norm);
                alreadyScreenedFormulas.add(norm);
                inverseDesignCandidates.push(norm);
              }
            }
          }

          emit("log", {
            phase: "engine",
            event: `Inverse + gradient in fast path`,
            detail: `Campaign ${campaign.id}: ${inverseCandidates.length} inverse + ${gradResult.results.length} gradient candidates. Best gradient: ${gradResult.bestFormula} Tc=${gradResult.bestTc.toFixed(1)}K`,
            dataSource: "Inverse Optimizer",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "engine", event: "Fast-path inverse error", detail: err.message?.slice(0, 200), dataSource: "Inverse Optimizer" });
      }
    }

    let structDiffusionCandidates: string[] = [];
    {
      try {
        const structResult = runStructureDiffusionCycle(200, 3, 3);
        for (const formula of structResult.formulas) {
          if (isValidFormula(formula)) {
            const normalized = normalizeFormula(formula);
            if (!alreadyScreenedFormulas.has(normalized)) {
              alreadyScreenedFormulas.add(normalized);
              structDiffusionCandidates.push(normalized);
            }
          }
        }
        if (structResult.formulas.length > 0) {
          emit("log", {
            phase: "engine",
            event: "Structure-first design in fast path",
            detail: `Generated ${structResult.formulas.length} from motifs [${structResult.motifsUsed.join(", ")}], ${structDiffusionCandidates.length} novel. Best: ${structResult.bestFormula} Tc=${structResult.bestTc.toFixed(1)}K`,
            dataSource: "Structure Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "engine", event: "Fast-path structure diffusion error", detail: err.message?.slice(0, 150), dataSource: "Structure Diffusion" });
      }
    }

    let motifDiffusionCandidates: string[] = [];
    if (cycleCount % 2 === 0) {
      try {
        const theoryElements = (latestTheoryBias && Object.keys(latestTheoryBias.elementBoosts).length >= 3)
          ? Object.entries(latestTheoryBias.elementBoosts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([el]) => el)
          : undefined;
        const diffResult = runDiffusionGenerationCycle(15, theoryElements);
        for (const crystal of diffResult.structures) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          if (alreadyScreenedFormulas.has(normalized)) continue;
          alreadyScreenedFormulas.add(normalized);
          motifDiffusionCandidates.push(normalized);
          recordGeneratorOutcome("motif_diffusion", true, 0, crystal.noveltyScore);
        }
        if (diffResult.structures.length > 0) {
          emit("log", {
            phase: "engine",
            event: "Fast-path motif diffusion",
            detail: `Generated ${diffResult.structures.length} structures, ${motifDiffusionCandidates.length} novel. Avg novelty: ${diffResult.stats.avgNovelty}`,
            dataSource: "Motif Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "engine", event: "Fast-path motif diffusion error", detail: err.message?.slice(0, 150), dataSource: "Motif Diffusion" });
      }
    }

    let cdvaeCandidates: string[] = [];
    if (cycleCount % 4 === 0) {
      try {
        const cdvaeTargetTc = autonomousBestTc > 100 ? autonomousBestTc * 1.5 : 200;
        const cdvaeResult = runCrystalDiffusionCycle(10, cdvaeTargetTc, 20, alreadyScreenedFormulas);
        for (const crystal of cdvaeResult) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          if (alreadyScreenedFormulas.has(normalized)) continue;
          alreadyScreenedFormulas.add(normalized);
          cdvaeCandidates.push(normalized);
          recordGeneratorOutcome("structure_diffusion", true, crystal.predictedTc, crystal.noveltyScore);
        }
        if (cdvaeResult.length > 0) {
          emit("log", {
            phase: "engine",
            event: "Fast-path CDVAE crystal diffusion",
            detail: `Generated ${cdvaeResult.length} full crystal structures, ${cdvaeCandidates.length} novel. Motifs: ${Array.from(new Set(cdvaeResult.map(c => c.motif))).join(", ")}`,
            dataSource: "CDVAE Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "engine", event: "Fast-path CDVAE error", detail: err.message?.slice(0, 150), dataSource: "CDVAE Diffusion" });
      }
    }

    if (cycleCount % 6 === 0) {
      try {
        const distTarget = autonomousBestTc > 100 ? autonomousBestTc * 1.4 : 200;
        const distResult = runDistributionBasedDiffusion(8, distTarget, 20, alreadyScreenedFormulas);
        for (const crystal of distResult) {
          if (!isValidFormula(crystal.formula)) continue;
          const normalized = normalizeFormula(crystal.formula);
          if (alreadyScreenedFormulas.has(normalized)) continue;
          alreadyScreenedFormulas.add(normalized);
          cdvaeCandidates.push(normalized);
          recordGeneratorOutcome("structure_diffusion", true, crystal.predictedTc, crystal.noveltyScore);
        }
        if (distResult.length > 0) {
          emit("log", {
            phase: "engine",
            event: "Fast-path distribution diffusion",
            detail: `Generated ${distResult.length} crystals from learned distributions, ${cdvaeCandidates.length} total novel`,
            dataSource: "Distribution Diffusion",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "engine", event: "Fast-path distribution diffusion error", detail: err.message?.slice(0, 150), dataSource: "Distribution Diffusion" });
      }
    }

    if (cycleCount % 15 === 0) {
      try {
        const vaeResult = runLatentSpaceInverseDesign(
          autonomousBestTc > 100 ? autonomousBestTc * 1.3 : 200,
          cdvaeCandidates.length > 0 ? cdvaeCandidates[0] : undefined,
          20, 0.02, 2,
        );
        for (const formula of vaeResult.decodedFormulas) {
          if (!isValidFormula(formula)) continue;
          const normalized = normalizeFormula(formula);
          if (alreadyScreenedFormulas.has(normalized)) continue;
          alreadyScreenedFormulas.add(normalized);
          cdvaeCandidates.push(normalized);
        }
        if (vaeResult.decodedFormulas.length > 0) {
          emit("log", {
            phase: "engine",
            event: "Fast-path VAE inverse design",
            detail: `VAE: best=${vaeResult.bestFormula} Tc=${vaeResult.bestTc}K, steps=${vaeResult.optimizationSteps}, decoded=${vaeResult.decodedFormulas.length}, converged=${vaeResult.converged}`,
            dataSource: "Crystal VAE",
          });
        }
      } catch (err: any) {
        emit("log", { phase: "engine", event: "Fast-path VAE error", detail: err.message?.slice(0, 150), dataSource: "Crystal VAE" });
      }
    }

    let integratedCandidates: string[] = [];
    try {
      if (integratedPipelineId) {
        const pipelineResult = runPipelineIteration(integratedPipelineId);
        if (pipelineResult) {
          for (const c of (pipelineResult.topCandidates ?? [])) {
            if (c.formula && isValidFormula(c.formula)) {
              const norm = normalizeFormula(c.formula);
              if (!alreadyScreenedFormulas.has(norm)) {
                alreadyScreenedFormulas.add(norm);
                integratedCandidates.push(norm);
              }
            }
          }
          const pStats = getPipelineStats(integratedPipelineId);
          if (pipelineResult.topCandidates?.length > 0 || (pStats && pStats.iteration % 5 === 0)) {
            emit("log", { phase: "engine", event: "Integrated pipeline iteration", detail: `Pipeline ${integratedPipelineId}: ${pipelineResult.candidatesGenerated} generated, ${pipelineResult.topCandidates?.length ?? 0} top candidates, ${integratedCandidates.length} novel. Best Tc: ${pipelineResult.bestTcThisIteration?.toFixed(1) ?? "N/A"}K`, dataSource: "Integrated Subsystems" });
          }
        }
      }
    } catch (e) { console.error("[Engine] Integrated pipeline iteration failed:", e); }

    try {
      if (integratedLabId) {
        const labResult = runLabIteration(integratedLabId);
        if (labResult) {
          for (const c of (labResult.topCandidates ?? [])) {
            if (c.formula && isValidFormula(c.formula)) {
              const norm = normalizeFormula(c.formula);
              if (!alreadyScreenedFormulas.has(norm)) {
                alreadyScreenedFormulas.add(norm);
                integratedCandidates.push(norm);
              }
            }
          }
          if ((labResult.topCandidates?.length ?? 0) > 0) {
            emit("log", { phase: "engine", event: "Integrated lab iteration", detail: `Lab ${integratedLabId}: strategy=${labResult.activeStrategy}, ${labResult.candidatesGenerated} generated, ${labResult.topCandidates?.length ?? 0} top, ${integratedCandidates.length} total novel. Best: ${labResult.bestFormula} Tc=${labResult.bestTc?.toFixed(1)}K`, dataSource: "Integrated Subsystems" });
          }
        }
      }
    } catch (e) { console.error("[Engine] Integrated lab iteration failed:", e); }

    if (cycleCount % 4 === 0) {
      try {
        const strategies = ["hydride-cage-optimizer", "layered-intercalation", "high-entropy-alloy", "light-element-phonon", "topological-edge", "pressure-stabilized", "electron-phonon-resonance", "charge-transfer-layer"] as const;
        const familyToStrategy: Record<string, typeof strategies[number]> = {
          hydride: "hydride-cage-optimizer",
          intermetallic: "high-entropy-alloy",
          "layered-pnictide": "layered-intercalation",
          boride: "light-element-phonon",
          cuprate: "charge-transfer-layer",
          chalcogenide: "layered-intercalation",
          "kagome-metal": "topological-edge",
          "oxide-perovskite": "electron-phonon-resonance",
        };
        let st: typeof strategies[number];
        if (latestTheoryBias && Object.keys(latestTheoryBias.familyPreferences).length > 0 && Math.random() < 0.6) {
          const topFamily = Object.entries(latestTheoryBias.familyPreferences)
            .sort((a, b) => b[1] - a[1])[0]?.[0];
          st = (topFamily && familyToStrategy[topFamily]) ? familyToStrategy[topFamily] : strategies[cycleCount % strategies.length];
        } else {
          st = strategies[cycleCount % strategies.length];
        }
        let elemPool = ["La", "Y", "H", "Ca", "B", "Nb", "Ti"];
        if (latestTheoryBias && Object.keys(latestTheoryBias.elementBoosts).length > 0) {
          const sortedElements = Object.entries(latestTheoryBias.elementBoosts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 7)
            .map(([el]) => el);
          if (sortedElements.length >= 3) elemPool = sortedElements;
        } else if (theoryFeedbackBias.biasedElements.length > 0) {
          elemPool = theoryFeedbackBias.biasedElements.slice(0, 7);
        }
        const prog = generateDesignProgram(st, elemPool);
        const execResult = executeDesignProgram(prog);
        if (execResult.formula && isValidFormula(execResult.formula)) {
          const norm = normalizeFormula(execResult.formula);
          if (!alreadyScreenedFormulas.has(norm)) {
            alreadyScreenedFormulas.add(norm);
            integratedCandidates.push(norm);
            registerProgram(prog, execResult.predictedTc ?? 0);
          }
        }
      } catch (e) { console.error("[Engine] Design program generation failed:", e); }
    }

    if (cycleCount - lastTheoryDiscoveryCycle >= 15 && cycleCount % 15 === 0) {
      try {
        const synthDataset = generateSyntheticDataset(40);
        const theories = runSymbolicPhysicsDiscovery(synthDataset);
        lastTheoryDiscoveryCycle = cycleCount;
        if (theories.length > 0) {
          const allTheories = getTheoryDatabase();
          const feedback = generateDiscoveryFeedback(allTheories);
          theoryFeedbackBias = { biasedVariables: feedback.biasedVariables ?? [], biasedElements: feedback.biasedElements ?? [] };
          crossEngineHub.recordInsight("theory", "global", {
            discoveredEquations: theories.slice(0, 3).map((t: any) => t.equation?.slice(0, 60) ?? ""),
            biases: [...(feedback.biasedVariables ?? []), ...(feedback.biasedElements ?? [])],
            symbolicScore: theories[0]?.theoryScore ?? 0,
          });
          emit("log", { phase: "engine", event: "Theory discovery cycle", detail: `Symbolic physics: ${theories.length} theories. Top: ${theories[0]?.equation?.slice(0, 80)} (score=${theories[0]?.theoryScore?.toFixed(3)})`, dataSource: "Integrated Subsystems" });
        }
      } catch (e) { console.error("[Engine] Theory discovery cycle failed:", e); }
    }

    if (cycleCount - lastCausalDiscoveryCycle >= 20 && cycleCount % 20 === 0) {
      try {
        const causalDataset = generateCausalDataset(60);
        const causalResult = runCausalDiscovery(causalDataset);
        lastCausalDiscoveryCycle = cycleCount;
        if (causalResult.designGuidance.length > 0) {
          causalDesignGuidance = causalResult.designGuidance.map(g => ({
            variable: g.variable,
            direction: g.direction,
            causalImpactOnTc: g.causalImpactOnTc,
          }));
        }
        emit("log", { phase: "engine", event: "Causal discovery cycle", detail: `Causal graph: ${causalResult.graph.edges.length} edges, ${causalResult.hypotheses.length} hypotheses, ${causalResult.rules.length} rules. Top guidance: ${causalResult.designGuidance[0]?.variable ?? "none"} (${causalResult.designGuidance[0]?.direction ?? ""})`, dataSource: "Integrated Subsystems" });
      } catch (e) { console.error("[Engine] Causal discovery cycle failed:", e); }
    }

    if (
      (cycleCount - lastTheoryBiasCycle >= 10) &&
      (lastTheoryDiscoveryCycle > lastTheoryBiasCycle || lastCausalDiscoveryCycle > lastTheoryBiasCycle)
    ) {
      try {
        const weightsBefore: Record<string, number> = {};
        const allocInfo = getGeneratorAllocations();
        for (const g of allocInfo.generators) {
          weightsBefore[g.name] = g.weight;
        }

        const theoryBias = computeTheoryGeneratorBias();
        latestTheoryBias = theoryBias;
        lastTheoryBiasCycle = cycleCount;

        if (Object.keys(theoryBias.generatorWeightBoosts).length > 0 && theoryBias.confidence > 0.1) {
          applyTheoryBias(theoryBias.generatorWeightBoosts);

          const weightsAfter: Record<string, number> = {};
          const allocInfoAfter = getGeneratorAllocations();
          for (const g of allocInfoAfter.generators) {
            weightsAfter[g.name] = g.weight;
          }

          recordTheoryBiasOutcome(
            weightsBefore, weightsAfter,
            allocInfoAfter.totalCandidatesGenerated,
            autonomousBestTc,
            autonomousTotalScreened > 0 ? autonomousBestTc * 0.5 : 0,
            autonomousTotalScreened > 0 ? autonomousTotalPassed / autonomousTotalScreened : 0,
          );

          const topFam = Object.entries(theoryBias.familyPreferences)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([f, b]) => `${f}=${b.toFixed(2)}`)
            .join(", ");
          const topEl = Object.entries(theoryBias.elementBoosts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4)
            .map(([e, b]) => `${e}=${b.toFixed(2)}`)
            .join(", ");

          emit("log", {
            phase: "engine",
            event: "Theory-driven generator bias applied",
            detail: `Confidence=${theoryBias.confidence.toFixed(2)}, theories=${theoryBias.sourceTheories}, causal edges=${theoryBias.sourceCausalEdges}. Top families: ${topFam}. Top elements: ${topEl}. Guidance: ${theoryBias.structuralGuidance[0] ?? "none"}`,
            dataSource: "Theory-Guided Generator",
          });
        }
      } catch (e) { console.error("[Engine] Theory-driven generator bias failed:", e); }
    }

    const normalizedPhysicsClean = physicsCleanCandidates.map(f => normalizeFormula(f));
    const dedupedPhysicsClean: string[] = [];
    for (const f of normalizedPhysicsClean) {
      if (!alreadyScreenedFormulas.has(f)) {
        alreadyScreenedFormulas.add(f);
        dedupedPhysicsClean.push(f);
      }
    }
    const allEngineCandidates = [...dedupedPhysicsClean, ...inverseDesignCandidates, ...structDiffusionCandidates, ...motifDiffusionCandidates, ...cdvaeCandidates, ...integratedCandidates];

    let dopingCandidates: string[] = [];
    let dopingSpecs: DopingSpec[] = [];
    try {
      const topForDoping = allEngineCandidates
        .filter(f => isValidFormula(f))
        .slice(0, 20);

      if (topForDoping.length > 0) {
        const dopingResult = runDopingBatch(topForDoping, 6, 30, alreadyScreenedFormulas);
        dopingCandidates = dopingResult.dopedFormulas.filter(f => !alreadyScreenedFormulas.has(f));
        dopingSpecs = dopingResult.specs;
        for (const f of dopingCandidates) {
          alreadyScreenedFormulas.add(f);
        }

        if (dopingCandidates.length > 0) {
          emit("log", {
            phase: "engine",
            event: "Doping engine variants",
            detail: `${dopingResult.stats.basesProcessed} bases → ${dopingCandidates.length} doped variants (sub=${dopingResult.stats.substitutional}, vac=${dopingResult.stats.vacancy}, int=${dopingResult.stats.interstitial}, e-doped=${dopingResult.stats.electronDoped}, h-doped=${dopingResult.stats.holeDoped}). Top: ${dopingCandidates.slice(0, 3).join(", ")}`,
            dataSource: "Doping Engine",
          });
        }
      }
    } catch (err: any) {
      emit("log", { phase: "engine", event: "Doping engine error", detail: err.message?.slice(0, 200), dataSource: "Doping Engine" });
    }

    const allWithDoping = [...allEngineCandidates, ...dopingCandidates];
    const seenInBatch = new Set<string>();
    const novelCandidates = allWithDoping.filter(f => {
      if (seenInBatch.has(f)) return false;
      seenInBatch.add(f);
      return true;
    });
    const rlNoveltyRatio = rlCandidates.filter(f => !alreadyScreenedFormulas.has(normalizeFormula(f))).length / Math.max(1, rlCandidates.length);
    if (alreadyScreenedFormulas.size > MAX_SCREENED_CACHE_SIZE) {
      const toRemove = alreadyScreenedFormulas.size - MAX_SCREENED_CACHE_SIZE;
      const iter = alreadyScreenedFormulas.values();
      for (let i = 0; i < toRemove; i++) {
        alreadyScreenedFormulas.delete(iter.next().value as string);
      }
    }

    emit("log", {
      phase: "engine",
      event: `Massive generation: ${genStats.totalGenerated} generated, ${genStats.uniqueAfterDedup} unique, ${genStats.passedPreScreen} passed pre-screen, ${novelCandidates.length} novel`,
      detail: `Valence filter: ${genStats.passedValenceFilter}, compatibility filter: ${genStats.passedCompatibilityFilter}. Focus: ${focusArea}. Screened cache: ${alreadyScreenedFormulas.size}. Engines: inverse=${inverseDesignCandidates.length}, structDiffusion=${structDiffusionCandidates.length}, motifDiffusion=${motifDiffusionCandidates.length}, cdvae=${cdvaeCandidates.length}, integrated=${integratedCandidates.length}, doping=${dopingCandidates.length}. Feeding ${novelCandidates.length} novel formulas through pipeline.`,
      dataSource: "Candidate Generator",
    });

    let passed = 0;
    let bestTcThisBatch = 0;
    let bestFormulaThisBatch = "";
    const failedFormulas: { formula: string; tc: number }[] = [];
    const thisCycleCandidates: LastCycleCandidate[] = [];
    const thisCycleFamilyCounts: Record<string, number> = {};
    const formulasInFlight = new Set<string>();

    const activeRules = getMinedRules();
    let filteredCandidates = novelCandidates;
    let patternFiltered = 0;
    if (activeRules.length > 0) {
      const patternScores = screenWithPatterns(novelCandidates);
      const scored = patternScores.sort((a, b) => b.theoryScore - a.theoryScore);
      const beforeCount = novelCandidates.length;
      filteredCandidates = scored
        .filter(s => s.theoryScore >= 0.3)
        .map(s => s.formula);
      patternFiltered = beforeCount - filteredCandidates.length;
      if (filteredCandidates.length === 0) filteredCandidates = novelCandidates;
    }

    const familyQuotaCounts: Record<string, number> = {};
    const totalBatchSize = filteredCandidates.length;
    const quotaBalanced: string[] = [];
    for (const formula of filteredCandidates) {
      const fam = classifyFamily(formula);
      const cap = FAMILY_CAPS[fam];
      if (cap !== undefined) {
        const currentCount = familyQuotaCounts[fam] || 0;
        const maxAllowed = Math.max(3, Math.ceil(totalBatchSize * cap));
        if (currentCount >= maxAllowed) {
          continue;
        }
        familyQuotaCounts[fam] = currentCount + 1;
      } else {
        familyQuotaCounts[fam] = (familyQuotaCounts[fam] || 0) + 1;
      }
      quotaBalanced.push(formula);
    }
    const quotaSkipped = filteredCandidates.length - quotaBalanced.length;
    if (quotaSkipped > 0) {
      console.log(`[Autonomous] Family quota: skipped ${quotaSkipped} excess candidates (caps: ${Object.entries(familyQuotaCounts).map(([k,v]) => `${k}=${v}`).join(", ")})`);
    }
    filteredCandidates = quotaBalanced;

    let physicsPrefiltered = 0;
    let batchRewardAccum = 0;
    let batchBestTc = 0;
    let batchPassCount = 0;
    let batchNovelCount = 0;
    let batchCount = 0;
    const RL_UPDATE_INTERVAL = 25;

    for (const formula of filteredCandidates) {
      if (!shouldContinue()) break;
      const normF = normalizeFormula(formula);
      if (formulasInFlight.has(normF)) continue;
      formulasInFlight.add(normF);
      autonomousTotalScreened++;
      batchCount++;

      const result = await runAutonomousDiscoveryCycle(formula);

      const candFamily = classifyFamily(formula);
      thisCycleCandidates.push({
        formula,
        tc: result.tc,
        passed: result.passed,
        reason: result.reason,
        family: candFamily,
      });
      thisCycleFamilyCounts[candFamily] = (thisCycleFamilyCounts[candFamily] || 0) + 1;

      bayesianOptimizer.addObservation(formula, result.tc, result.physicsPred?.lambda ?? 0, result.passed ? 1 : 0);

      const isRlCandidate = rlCandidates.includes(formula);
      const isBoCandidate = boTopFormulas.includes(formula);
      let generatorName = "massive_combinatorial";
      if (isRlCandidate) {
        generatorName = "rl";
        recordGeneratorOutcome("rl", result.passed, result.tc, result.passed ? 0.6 : 0.1);
      }
      if (isBoCandidate) {
        generatorName = "bo_exploration";
        recordGeneratorOutcome("bo_exploration", result.passed, result.tc, result.passed ? 0.7 : 0.1);
      }
      if (!isRlCandidate && !isBoCandidate) {
        recordGeneratorOutcome("massive_combinatorial", result.passed, result.tc, result.passed ? 0.5 : 0.1);
      }
      if (candidateGeneratorSource.size < MAX_GENERATOR_SOURCE_ENTRIES) {
        candidateGeneratorSource.set(formula, generatorName);
      }

      try {
        const els = parseFormulaElements(formula);
        let rejectCategory: string | undefined;
        if (!result.passed && result.reason) {
          if (result.reason.startsWith("invalid-elements") || result.reason.startsWith("formation-energy")) {
            rejectCategory = "chemistry_reject";
          } else if (result.reason.startsWith("stability-prefilter") || result.reason.startsWith("surrogate-reject")) {
            rejectCategory = "stability_reject";
          } else if (result.reason.startsWith("phonon-instability")) {
            rejectCategory = "phonon_reject";
          } else if (result.reason.startsWith("physics-prefilter") || result.reason.startsWith("low-gb-tc")) {
            rejectCategory = "tc_too_low";
          } else if (result.reason.startsWith("below-tier3")) {
            rejectCategory = "tc_too_low";
          }
        }
        rlAgent.recordElementOutcome(els, result.tc, result.passed, rejectCategory);

        if (result.passed && result.tc > 10) {
          const hubIns = crossEngineHub.getInsightsFor(formula);
          let crossEngineBonus = 0;
          const topoScore = hubIns?.topology?.topologicalScore ?? 0;
          const nestScore = hubIns?.fermi?.nestingScore ?? 0;
          const pairingScore = hubIns?.pairing?.compositePairingStrength ?? 0;
          if (topoScore > 0.3) crossEngineBonus += topoScore * 0.2;
          if (nestScore > 0.3) crossEngineBonus += nestScore * 0.15;
          if (pairingScore > 0.4) crossEngineBonus += pairingScore * 0.15;

          const stabilityEst = result.physicsPred?.hullDistance != null ? (1 - result.physicsPred.hullDistance) : 0.5;
          const bfComposite = computeBandFeatureScore(result.tc, stabilityEst, nestScore, topoScore);
          if (bfComposite.compositeScore > 0.4) {
            crossEngineBonus += bfComposite.compositeScore * 0.1;
          }

          if (crossEngineBonus > 0.05) {
            const enrichedTc = Math.round(result.tc * (1 + crossEngineBonus));
            rlAgent.recordElementOutcome(els, enrichedTc, true);
          }
        }

        if (result.passed) {
          try {
            const hubIns = crossEngineHub.getInsightsFor(formula);
            const obsLambda = hubIns?.physics?.lambda ?? (result.physicsPred?.lambda ?? 0.5);
            const obsStability = 1 - (result.physicsPred?.hullDistance ?? 0.2);
            bayesianOptimizer.addObservation(formula, result.tc, obsLambda, obsStability);
            if (hubIns?.topology && hubIns.topology.topologicalScore > 0.3) {
              incorporateSuccessData(formula, result.tc, { topologicalScore: hubIns.topology.topologicalScore });
            }
          } catch (e) { console.error("[Engine] Cross-engine BO feed failed:", e); }
        }
      } catch (e) { console.error("[Engine] RL element outcome recording failed:", e); }

      if (result.tc > batchBestTc) batchBestTc = result.tc;
      if (result.passed) batchPassCount++;
      if (!alreadyScreenedFormulas.has(formula)) batchNovelCount++;
      batchRewardAccum += (result.tc > 0 ? Math.min(1, result.tc / 400) * 0.5 : 0)
        + (result.passed ? 0.3 : 0)
        + (result.tc > autonomousBestTc ? 0.5 : 0);

      if (batchCount % RL_UPDATE_INTERVAL === 0) {
        const batchSize = RL_UPDATE_INTERVAL;
        const interimReward = rlAgent.computeReward(
          batchBestTc,
          autonomousBestTc,
          batchPassCount > 0,
          batchPassCount / batchSize,
          batchNovelCount / batchSize * 0.5
        );
        rlAgent.updatePolicy(rlState, rlAction, interimReward);
        batchBestTc = 0;
        batchPassCount = 0;
        batchNovelCount = 0;
        batchRewardAccum = 0;
      }

      try {
        const constraintDetail = constraintFiltered.details.find(d => d.formula === formula);
        if (constraintDetail && constraintDetail.violations.length > 0) {
          updateConstraintWeightsFromReward(formula, result.tc, constraintDetail.violations);
        }
      } catch (e) { console.error("[Engine] Constraint weight update failed:", e); }

      try {
        if (result.tc > 0) {
          buildAndStoreFeatureRecord(formula, result.tc, null, result.passed ? 0.5 : 0.1);
        }
      } catch (e) { console.error("[Engine] Feature record build failed:", e); }

      const isPromising = result.passed || result.tc >= 5;

      if (result.passed) {
        try {
          const electronic = computeElectronicStructure(formula);
          const topoResult = analyzeTopology(formula, electronic);
          trackTopologyResult(topoResult);
          crossEngineHub.recordInsight("topology", formula, topoResult);
        } catch (e) { console.error(`[Engine] Topology analysis failed for ${formula}:`, e); }
        try {
          const fsResult = computeFermiSurface(formula);
          assignToCluster(formula, fsResult, result.tc);
          crossEngineHub.recordInsight("fermi", formula, fsResult);
        } catch (e) { console.error(`[Engine] Fermi surface cluster assignment failed for ${formula}:`, e); }
        try {
          const pairResult = computePairingProfile(formula);
          crossEngineHub.recordInsight("pairing", formula, pairResult);
        } catch (e) { console.error(`[Engine] Pairing profile failed for ${formula}:`, e); }
      }
      if (isPromising) {
        try {
          const family = classifyFamily(formula);
          const synthCtx: MaterialContext = {
            formula,
            materialClass: family,
            predictedTc: result.tc,
            lambda: result.physicsPred?.lambda ?? 0.5,
            pressure: 0,
            isHydride: family.toLowerCase().includes("hydride"),
            isCuprate: family.toLowerCase().includes("cuprate"),
            isLayered: false,
            meltingPointEstimate: 1500,
            stabilityClass: result.physicsPred?.hullDistance != null && result.physicsPred.hullDistance < 0.005 ? "thermodynamically-stable"
              : result.physicsPred?.hullDistance != null && result.physicsPred.hullDistance < 0.1 ? "metastable-accessible"
              : "metastable-difficult",
            energyAboveHull: result.physicsPred?.hullDistance ?? 0.1,
          };
          const synthResult = optimizeSynthesisConditions(synthCtx);

          const sv = defaultSynthesisVector(family);
          const effects = simulateSynthesisEffects(formula, family, sv);
          recordSynthesisResult(formula, family, sv, result.tc, 1 - (result.physicsPred?.hullDistance ?? 0.1));

          if (synthResult.overallFeasibility > 0.6) {
            const feasBonus = synthResult.overallFeasibility * 0.05;
            result.tc = Math.min(400, Math.round(result.tc * (1 + feasBonus)));
            feedbackLoopStats.synthesisFeasibilityBonuses++;
            feedbackLoopStats.synthesisTotalFeasibilityBoost += feasBonus;
          }

          if (result.tc > 30 && cycleCount % 10 === 0) {
            try {
              const synthOpt = optimizeSynthesisForFixedMaterial(
                formula, family,
                (testSv: SynthesisVector) => {
                  const eff = simulateSynthesisEffects(formula, family, testSv);
                  return result.tc * eff.effectiveTcMultiplier;
                },
                15, 8
              );
              if (synthOpt.bestTc > result.tc) {
                recordSynthesisResult(formula, family, synthOpt.bestVector, synthOpt.bestTc, 0.7);
              }
            } catch (e) { console.error(`[Engine] Synthesis optimization failed for ${formula}:`, e); }
          }

          try {
            const defects = generateDefectVariants(formula);
            if (defects.length > 0 && result.physicsPred) {
              const bestDefect = defects.reduce((best, d) => {
                const adj = adjustElectronicStructure(
                  result.physicsPred!.dosAtEF ?? 1.0,
                  result.physicsPred!.lambda ?? 0.5,
                  d.defectDensity,
                  d.type,
                  formula
                );
                return adj.tcModifier > (best?.tcMod ?? 0) ? { defect: d, tcMod: adj.tcModifier } : best;
              }, null as { defect: any; tcMod: number } | null);
              if (bestDefect && bestDefect.tcMod > 1.05) {
                const defectTc = Math.round(result.tc * bestDefect.tcMod);
                const defectFormulaRaw = bestDefect.defect.mutatedFormula || formula;
                const defectFormula = normalizeFormula(defectFormulaRaw);
                crossEngineHub.recordInsight("defect", formula, {
                  optimalDopants: [bestDefect.defect.element],
                  bestTcModifier: bestDefect.tcMod,
                  bestDefectType: bestDefect.defect.type,
                  variantCount: defects.length,
                  bestMutatedFormula: defectFormula,
                });
                if (!alreadyScreenedFormulas.has(defectFormula) && defectFormula !== formula) {
                  alreadyScreenedFormulas.add(defectFormula);
                  feedbackLoopStats.defectCandidatesAdded++;
                  feedbackLoopStats.defectTotalTcBoost += bestDefect.tcMod - 1;
                  try {
                    const defectId = `sc-def-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                    await insertCandidateWithStabilityCheck({
                      id: defectId,
                      name: defectFormula,
                      formula: defectFormula,
                      predictedTc: defectTc,
                      ensembleScore: Math.min(1, (result.tc > 0 ? defectTc / 293 : 0) * 0.7),
                      crystalStructure: "defect-engineered",
                      status: "theoretical",
                      notes: `Defect: ${bestDefect.defect.type} at ${bestDefect.defect.element}, Tc boost ${bestDefect.tcMod.toFixed(3)}x from ${formula}`,
                    }, "random_exploration");
                  } catch (e) { console.error(`[Engine] Defect candidate insert failed for ${defectFormula}:`, e); }
                }
                emit("log", { phase: "defect-engine", event: "Defect enhancement found", detail: `${formula}: ${bestDefect.defect.type} defect at ${bestDefect.defect.element} -> Tc modifier ${bestDefect.tcMod.toFixed(3)}, defect variant ${defectFormula} (est. ${defectTc}K) added to pool` });
              }
            }
          } catch (e) { console.error(`[Engine] Defect variant generation failed for ${formula}:`, e); }

          try {
            const corrEffects = estimateCorrelationEffects(formula, {
              UoverW: result.physicsPred?.UoverW,
              dosAtEF: result.physicsPred?.dosAtEF,
            });
            if (corrEffects.tcModifier > 1.05) {
              const corrBoost = Math.min(corrEffects.tcModifier, 1.5);
              const boostedTc = Math.min(400, Math.round(result.tc * corrBoost));
              if (boostedTc > result.tc) {
                feedbackLoopStats.correlationBoostsApplied++;
                feedbackLoopStats.correlationTotalTcBoost += corrBoost - 1;
                result.tc = boostedTc;
              }
              emit("log", { phase: "correlation-engine", event: "Correlation boost applied", detail: `${formula}: ${corrEffects.regime.regime} regime, Tc ${result.tc}K -> ${boostedTc}K (modifier ${corrEffects.tcModifier.toFixed(3)}), patterns: ${corrEffects.materialPatterns.join(", ")}` });
            }
          } catch (e) { console.error(`[Engine] Correlation effects estimation failed for ${formula}:`, e); }

          try {
            const growthResult = simulateCrystalGrowth(formula, family, sv);
            if (growthResult.qualityScore >= 0.6) {
              const growthBonus = growthResult.qualityScore * 0.03;
              result.tc = Math.min(400, Math.round(result.tc * (1 + growthBonus)));
              feedbackLoopStats.growthQualityBonuses++;
              feedbackLoopStats.growthTotalQualityBoost += growthBonus;
            }
            if (growthResult.qualityScore < 0.3) {
              emit("log", { phase: "crystal-growth", event: "Growth challenge identified", detail: `${formula}: quality=${growthResult.qualityScore.toFixed(2)}, grain=${growthResult.grainStructure.grainSize.toFixed(0)}nm` });
            }
          } catch (e) { console.error(`[Engine] Crystal growth simulation failed for ${formula}:`, e); }

          if (result.tc > 15) {
            try {
              const hubIns = crossEngineHub.getInsightsFor(formula);
              const multiIns: MultiEngineInsights = {
                formula, materialClass: family, predictedTc: result.tc,
                physics: hubIns?.physics ? {
                  lambda: hubIns.physics.lambda, omegaLog: hubIns.physics.omegaLog,
                  dosAtFermi: hubIns.physics.dosAtFermi, metallicity: hubIns.physics.metallicity,
                  stabilityScore: 1 - (result.physicsPred?.hullDistance ?? 0.3),
                  correlationStrength: hubIns.physics.correlationStrength,
                } : undefined,
                topology: hubIns?.topology ? {
                  topologicalScore: hubIns.topology.topologicalScore, z2Score: hubIns.topology.z2Score,
                  socStrength: hubIns.topology.socStrength, topologicalClass: hubIns.topology.topologicalClass,
                  majoranaFeasibility: hubIns.topology.majoranaFeasibility, bandInversionProbability: hubIns.topology.bandInversionProbability,
                } : undefined,
                fermi: hubIns?.fermi ? {
                  nestingScore: hubIns.fermi.nestingScore, pocketCount: hubIns.fermi.pocketCount,
                  electronHoleBalance: hubIns.fermi.electronHoleBalance,
                  cylindricalCharacter: hubIns.fermi.cylindricalCharacter,
                  multiBandScore: hubIns.fermi.multiBandScore, sigmaBandPresence: hubIns.fermi.sigmaBandPresence,
                } : undefined,
                pairing: hubIns?.pairing ? {
                  dominantMechanism: hubIns.pairing.dominantMechanism, pairingSymmetry: hubIns.pairing.pairingSymmetry,
                  compositePairingStrength: hubIns.pairing.compositePairingStrength,
                  phononStrength: hubIns.pairing.phononStrength,
                  spinStrength: hubIns.pairing.spinStrength, orbitalStrength: hubIns.pairing.orbitalStrength,
                } : undefined,
                defect: hubIns?.defect ? {
                  variants: [], bestDopant: hubIns.defect.optimalDopants[0] ?? "",
                  bestTcBoost: hubIns.defect.bestTcModifier, optimalDefectType: hubIns.defect.bestDefectType,
                } : undefined,
              };
              const synthDisc = discoverNovelSynthesisPaths(multiIns, 10, 6);
              if (synthDisc.bestRoute && synthDisc.bestRoute.fitnessScore > 0.5) {
                crossEngineHub.recordInsight("synthesis", formula, {
                  feasibilityScore: synthDisc.bestRoute.feasibilityScore,
                  complexity: synthDisc.bestRoute.steps.length,
                  bestMethod: synthDisc.bestRoute.steps[0]?.method ?? "novel",
                  pathStepCount: synthDisc.bestRoute.steps.length,
                });
                emit("log", {
                  phase: "synthesis-discovery",
                  event: "Novel synthesis path discovered",
                  detail: `${formula}: ${synthDisc.bestRoute.steps.length} steps, fitness=${synthDisc.bestRoute.fitnessScore.toFixed(3)}, novelty=${synthDisc.bestRoute.noveltyScore.toFixed(3)}, engines=[${synthDisc.bestRoute.engineContributions.join(",")}]`,
                  dataSource: "Synthesis Discovery",
                });
              }
            } catch (e) { console.error(`[Engine] Autonomous synthesis discovery failed for ${formula}:`, e); }
          }

          if (result.tc > 25 && cycleCount % 3 === 0) {
            try {
              const expCandidate: ExperimentCandidate = {
                formula,
                predictedTc: result.tc,
                stability: 1 - (result.physicsPred?.hullDistance ?? 0.1),
                synthesisFeasibility: synthResult?.overallFeasibility ?? 0.5,
                novelty: alreadyScreenedFormulas.has(formula) ? 0.3 : 0.8,
                uncertainty: result.physicsPred?.lambdaUncertainty ?? 0.5,
                materialClass: family,
                crystalStructure: "predicted",
              };
              const plan = generateExperimentPlan(expCandidate);
              feedbackLoopStats.experimentPlansGenerated++;
              if (plan.ranking.experimentScore > 0.6) {
                feedbackLoopStats.experimentDFTPrioritized++;
                const expBonus = (plan.ranking.experimentScore - 0.6) * 0.08;
                result.tc = Math.min(400, Math.round(result.tc * (1 + expBonus)));
                emit("log", { phase: "experiment-planner", event: "Experiment plan generated", detail: `${formula}: score=${plan.ranking.experimentScore.toFixed(3)}, Tc boosted by ${(expBonus * 100).toFixed(1)}%, timeline=${plan.timeline}, risk=${plan.riskAssessment}. ${plan.characterization.length} characterization methods suggested.` });
              }
            } catch (e) { console.error(`[Engine] Experiment plan generation failed for ${formula}:`, e); }
          }
        } catch (e) { console.error(`[Engine] Synthesis/feedback loop failed for ${formula}:`, e); }
      }

      if (result.passed) {
        passed++;
        autonomousTotalPassed++;
        pipelineStageMetrics.totalPassed++;
        if (result.tc > bestTcThisBatch) {
          bestTcThisBatch = result.tc;
          bestFormulaThisBatch = formula;
        }
        if (result.physicsPred) {
          const p = result.physicsPred;
          emit("log", {
            phase: "engine",
            event: "Physics ML prediction",
            detail: `Physics ML: lambda=${p.lambda.toFixed(2)}±${p.lambdaUncertainty.toFixed(2)}, DOS=${p.dosAtEF.toFixed(2)}±${p.dosUncertainty.toFixed(2)}, omega=${p.omegaLog.toFixed(0)}±${p.omegaUncertainty.toFixed(0)}, hull=${p.hullDistance.toFixed(3)}±${p.hullUncertainty.toFixed(3)} for ${formula}`,
            dataSource: "Physics ML",
          });
        }
        console.log(`[Autonomous] PASSED: ${formula} Tc=${result.tc}K reason=${result.reason}`);
      } else {
        failedFormulas.push({ formula, tc: result.tc });
        if (result.reason.startsWith("physics-prefilter")) physicsPrefiltered++;
        if (autonomousTotalScreened <= 200 || autonomousTotalScreened % 50 === 0) {
          console.log(`[Autonomous] REJECTED: ${formula} Tc=${result.tc}K reason=${result.reason}`);
        }
      }
    }

    lastCycleCandidates = thisCycleCandidates;
    lastCycleFamilyCounts = thisCycleFamilyCounts;

    if (failedFormulas.length > 0 && autonomousTotalScreened % 100 === 0) {
      try {
        const failureTrainingData = failedFormulas
          .filter(f => f.tc > 0)
          .map(f => ({ formula: f.formula, tc: f.tc, formationEnergy: undefined as number | undefined, structure: undefined as any }));
        if (failureTrainingData.length >= 5) {
          invalidateGNNModel();
          autonomousGNNRetrainCount++;
        }
      } catch (e) { console.error("[Engine] GNN model invalidation failed:", e); }
    }

    if (physicsPredictor.shouldRetrain(cycleCount)) {
      physicsPredictor.retrain(cycleCount);
      emit("log", {
        phase: "engine",
        event: "Physics ML retrained",
        detail: `PhysicsPredictor retrained on ${physicsPredictor.getTrainingSize()} samples at cycle ${cycleCount}`,
        dataSource: "Physics ML",
      });
    }

    let rlReward = rlAgent.computeReward(
      bestTcThisBatch,
      autonomousBestTc,
      passed > 0,
      passed / Math.max(1, filteredCandidates.length),
      rlNoveltyRatio * 0.5
    );

    const memStats = discoveryMemory.getStats();
    if (memStats.totalRecords > 5 && bestTcThisBatch > 20 && bestFormulaThisBatch) {
      try {
        const topFp = buildFingerprint(bestFormulaThisBatch, bestTcThisBatch, {});
        const memBonus = discoveryMemory.computeMemoryRewardBonus(topFp);
        rlReward += memBonus.bonus * 0.5;
      } catch (e) { console.error("[Engine] Memory reward bonus computation failed:", e); }
    }

    if (feedbackLoopStats.defectCandidatesAdded > 0) {
      rlReward += 0.05 * Math.min(3, feedbackLoopStats.defectCandidatesAdded);
    }
    if (feedbackLoopStats.correlationBoostsApplied > 0) {
      rlReward += 0.03 * Math.min(5, feedbackLoopStats.correlationBoostsApplied);
    }
    if (feedbackLoopStats.synthesisFeasibilityBonuses > 0) {
      rlReward += 0.02 * Math.min(5, feedbackLoopStats.synthesisFeasibilityBonuses);
    }
    if (feedbackLoopStats.growthQualityBonuses > 0) {
      rlReward += 0.02 * Math.min(5, feedbackLoopStats.growthQualityBonuses);
    }
    if (feedbackLoopStats.experimentDFTPrioritized > 0) {
      rlReward += 0.03 * Math.min(3, feedbackLoopStats.experimentDFTPrioritized);
    }
    if (feedbackLoopStats.pressurePathwayBoosts > 0) {
      rlReward += 0.04 * Math.min(3, feedbackLoopStats.pressurePathwayBoosts);
    }

    rlAgent.updatePolicy(rlState, rlAction, rlReward);

    if (cycleCount % 10 === 0 && getDatasetSize() >= 20) {
      try {
        const dataset = getFeatureDataset();
        const srData = dataset
          .filter(r => r.tc > 0 && Number.isFinite(r.tc))
          .map(r => ({ ...(r.featureVector as Record<string, number>), tc: r.tc }));
        if (srData.length >= 15) {
          const theories = runSymbolicRegression(srData, "tc", { populationSize: 100, generations: 40 });
          if (theories.length > 0) {
            emit("log", {
              phase: "engine",
              event: "Theory discovery cycle",
              detail: `Symbolic regression on ${srData.length} samples: found ${theories.length} candidate equations. Best: ${theories[0].equation} (R²=${theories[0].r2.toFixed(3)}, cvScore=${theories[0].cvScore.toFixed(3)}, complexity=${theories[0].complexity}, plausibility=${theories[0].plausibility.toFixed(2)}, overfit=${theories[0].isOverfit})`,
              dataSource: "Theory Discovery Engine",
            });
          }
        }
      } catch (e) { console.error("[Engine] Symbolic regression cycle failed:", e); }

      try {
        const hypResult = runHypothesisCycle();
        if (hypResult.newHypotheses > 0 || hypResult.testedHypotheses > 0) {
          const hypStats = getHypothesisStats();
          emit("log", {
            phase: "engine",
            event: "Hypothesis engine cycle",
            detail: `Generated ${hypResult.newHypotheses} new hypotheses, tested ${hypResult.testedHypotheses}. Active: ${hypResult.activeCount}, supported: ${hypResult.supportedCount}, refuted: ${hypResult.refutedCount}. Avg confidence: ${hypStats.avgConfidence.toFixed(3)}.${hypStats.topHypothesis ? ` Top: "${hypStats.topHypothesis.statement.slice(0, 100)}" (conf=${hypStats.topHypothesis.confidenceScore.toFixed(3)})` : ""}`,
            dataSource: "Hypothesis Engine",
          });
        }
      } catch (e) { console.error("[Engine] Hypothesis engine cycle failed:", e); }
    }

    rebalanceWeights();

    try {
      const hubPatterns = crossEngineHub.getGlobalPatterns();
      const physGuidance = crossEngineHub.getPhysicsGuidance();
      const synthGuidance = crossEngineHub.getSynthesisGuidance();
      const topoGuidance = crossEngineHub.getTopologicalGuidance();

      if (hubPatterns.length > 0) {
        const highTcConvergence = hubPatterns.find(p => p.name === "multi-engine-convergence");
        if (highTcConvergence && highTcConvergence.formulaExamples.length > 0) {
          for (const exFormula of highTcConvergence.formulaExamples.slice(0, 3)) {
            const els = parseFormulaElements(exFormula);
            rlAgent.recordElementOutcome(els, highTcConvergence.avgTcBoost, true);
          }
        }
      }

      if (topoGuidance.recommendedElements.length > 0) {
        rlAgent.recordElementOutcome(topoGuidance.recommendedElements, 30, true);
      }

      if (synthGuidance.dopingRecommended && synthGuidance.dopantSuggestions.length > 0) {
        rlAgent.recordElementOutcome(synthGuidance.dopantSuggestions.slice(0, 3), 20, true);
      }
    } catch (e) { console.error("[Engine] Cross-engine hub steering failed:", e); }

    if (cycleCount % 5 === 0) {
      try {
        const landscapeUpdate = updateLandscape([]);
        const lStats = getLandscapeStats();
        const zoneMap = getZoneMap();
        if (lStats.embeddedMaterials > 5) {
          const rlBias = getLandscapeRLBias();

          updateZoneHistory(cycleCount);
          const intelStats = getLandscapeIntelligenceStats();
          const intelBias = getIntelligenceGeneratorBias();
          const intelDetail = intelStats.highPriorityZoneCount > 0
            ? ` Intelligence: ${intelStats.highPriorityZoneCount} high-priority zones, ${intelStats.frontierRegionCount} frontier regions. Exploration ratio: ${intelBias.explorationRatio.toFixed(2)}.`
            : "";

          emit("log", {
            phase: "engine",
            event: "Discovery landscape update",
            detail: `Landscape: ${lStats.embeddedMaterials} materials, ${zoneMap.zones.length} zones (${zoneMap.topZones.length} high-priority). Coverage: ${zoneMap.coveragePercent}%. Tc: ${lStats.tcRange.min}-${lStats.tcRange.max}K. RL bias: ${Object.entries(rlBias.elementGroupWeights).filter(([,v]) => v > 0.3).map(([k,v]) => `${k}=${v.toFixed(2)}`).slice(0, 5).join(", ")}.${intelDetail}`,
            dataSource: "Discovery Landscape",
          });
        }
      } catch (e) { console.error("[Engine] Discovery landscape update failed:", e); }
    }

    const clusterGuidance = getClusterGuidance();
    if (clusterGuidance.highPotentialClusters.length > 0 || clusterGuidance.underExploredClusters.length > 0) {
      const highPotential = clusterGuidance.highPotentialClusters.map(c => `${c.clusterId}(avgTc=${(c.avgTc ?? 0).toFixed(0)}K)`).join(", ");
      const underExplored = clusterGuidance.underExploredClusters.map(c => c.clusterId).join(", ");
      emit("log", {
        phase: "engine",
        event: "Fermi surface cluster guidance",
        detail: `High-potential FS clusters: [${highPotential}]. Under-explored: [${underExplored}]. Total clustered: ${clusterGuidance.totalMaterials}. Suggestions: ${clusterGuidance.suggestions.slice(0, 2).join("; ")}`,
        dataSource: "FS Clustering",
      });
    }

    const perfMetrics = getPerformanceMetrics();
    const boStats = bayesianOptimizer.getStats();
    const rlStats = rlAgent.getStats();
    const genAllocations = getGeneratorAllocations();

    const patternDetail = patternFiltered > 0 ? ` Pattern filter removed ${patternFiltered}/${candidates.length} (${activeRules.length} rules active).` : "";
    const physicsDetail = physicsPrefiltered > 0 ? ` Physics pre-filter rejected ${physicsPrefiltered}/${filteredCandidates.length}.` : "";
    const memDetail = memStats.totalRecords > 0 ? ` Memory: ${memStats.totalRecords} patterns, ${memStats.clusterCount} clusters.` : "";
    const theoryDetail = getDatasetSize() > 0 ? ` FeatureDB: ${getDatasetSize()} records, theories: ${getDiscoveredTheories().length}.` : "";
    const perfDetail = perfMetrics.totalPredictions > 0 ? ` PerfTrack: MAE=${perfMetrics.overall.mae.toFixed(1)}K, R²=${perfMetrics.overall.r2.toFixed(3)}.` : "";
    const genDetail = ` Generators: ${genAllocations.generators.map(g => `${g.name}=${(g.weight * 100).toFixed(0)}%`).join(", ")}.`;
    const fsClusterDetail = clusterGuidance.totalMaterials > 0 ? ` FS clusters: ${clusterGuidance.totalMaterials} materials in ${clusterGuidance.highPotentialClusters.length} high-potential clusters.` : "";
    const lsStats = getLandscapeStats();
    const landscapeDetail = lsStats.embeddedMaterials > 0 ? ` Landscape: ${lsStats.embeddedMaterials} embedded, ${lsStats.tcRange.max.toFixed(0)}K max.` : "";
    emit("log", {
      phase: "engine",
      event: "Autonomous loop: " + filteredCandidates.length + " screened, " + passed + " passed" + (bestTcThisBatch > 0 ? ", best Tc = " + bestTcThisBatch + "K" : ""),
      detail: `RL+BO pipeline from ${focusArea} (${genStats.totalGenerated} massive + ${rlCandidates.length} RL). RL reward=${rlReward.toFixed(3)}, BO obs=${boStats.observationCount}. Pass rate: ${(autonomousTotalPassed / Math.max(1, autonomousTotalScreened) * 100).toFixed(1)}%. Total screened: ${autonomousTotalScreened}.${patternDetail}${physicsDetail}${memDetail}${theoryDetail}${perfDetail}${genDetail}${fsClusterDetail}${landscapeDetail}`,
      dataSource: "Autonomous Loop",
    });
  } finally {
    activeTasks.delete("Autonomous Screening");
    broadcast("taskEnd", { task: "Autonomous Screening" });
  }
}

function sanitizeStatsNumeric(obj: any): any {
  if (obj === null || obj === undefined) return 0;
  if (typeof obj === "number") return Number.isFinite(obj) ? obj : 0;
  if (typeof obj === "string") return obj;
  if (typeof obj === "boolean") return obj;
  if (Array.isArray(obj)) return obj.map(sanitizeStatsNumeric);
  if (typeof obj === "object") {
    const result: any = {};
    for (const key of Object.keys(obj)) {
      const val = obj[key];
      if (val === null || val === undefined) {
        result[key] = 0;
      } else {
        result[key] = sanitizeStatsNumeric(val);
      }
    }
    return result;
  }
  return obj;
}

export async function getAutonomousLoopStats() {
  const elapsedHours = (Date.now() - autonomousStartTime) / 3600000;
  const alStats = getActiveLearningStats();
  const xtbStats = getXTBStats();
  const allTimeStats = await storage.getStats().catch(() => null);
  const allTimePipelineTotal = allTimeStats ? allTimeStats.pipelineStages.reduce((s: number, p: any) => s + (p.count || 0), 0) : 0;
  const allTimePipelinePassed = allTimeStats ? allTimeStats.pipelineStages.reduce((s: number, p: any) => s + (p.passed || 0), 0) : 0;

  const reconciledScreened = Math.max(autonomousTotalScreened, allTimePipelineTotal);
  const reconciledPassed = Math.max(autonomousTotalPassed, allTimePipelinePassed);

  let dbBestTc = 0;
  try {
    const topCandidates = await storage.getSuperconductorCandidatesByTc(1);
    if (topCandidates.length > 0) dbBestTc = topCandidates[0].predictedTc ?? 0;
  } catch {} 
  const reconciledBestTc = Math.max(autonomousBestTc, dbBestTc);

  const rawStats = {
    totalScreened: reconciledScreened,
    totalPassed: reconciledPassed,
    passRate: reconciledScreened > 0 ? reconciledPassed / reconciledScreened : 0,
    bestTc: reconciledBestTc,
    throughputPerHour: (elapsedHours >= (5 / 60) && autonomousTotalScreened > 0) ? Math.round(autonomousTotalScreened / Math.max(elapsedHours, 0.1)) : 0,
    gnnRetrainCount: autonomousGNNRetrainCount,
    activeLearning: alStats,
    xtb: {
      method: "GFN2-xTB v6.7.1",
      level: "Semi-empirical (tight-binding DFT)",
      runs: xtbStats.runs,
      successes: xtbStats.successes,
      cacheSize: xtbStats.cacheSize,
      successRate: xtbStats.runs > 0 ? `${(xtbStats.successes / xtbStats.runs * 100).toFixed(1)}%` : "N/A",
    },
    surrogateModel: getSurrogateStats(),
    rlAgent: rlAgent.getStats(),
    bayesianOptimizer: bayesianOptimizer.getStats(),
    crystalDiffusion: getDiffusionStats(),
    crystalStructureDiffusion: getCrystalDiffusionStats(),
    distributionDiffusion: getDistributionDiffusionStats(),
    multiTaskGNN: getMultiTaskStats(),
    crystalVAE: getVAEStats(),
    topologyDetection: getTopologyStats(),
    inverseOptimizer: getInverseOptimizerStats(),
    nextGenPipeline: getNextGenPipelineStatsForEngine(),
    selfImprovingLab: getSelfImprovingLabStatsForEngine(),
    differentiableOptimizer: getDifferentiableOptimizerStats(),
    structureFirstDesign: getStructureDiffusionStats(),
    physicsConstraints: getConstraintEngineStats(),
    scPillarsOptimizer: getPillarOptimizerStats(),
    generatorAllocations: getGeneratorAllocations(),
    fermiSurfaceClusters: getClusterGuidance(),
    discoveryLandscape: getLandscapeStats(),
    landscapeIntelligence: getLandscapeIntelligenceStats(),
    pressurePathways: getPathwayStats(),
    hypothesisEngine: getHypothesisStats(),
    synthesisOptimizer: getSynthesisOptimizerStats(),
    synthesisParameterSpace: getParameterSpace(),
    synthesisSimulator: getSimulatorStats(),
    synthesisLearning: getSynthesisLearningStats(),
    defectEngine: getDefectEngineStats(),
    correlationEngine: getCorrelationEngineStats(),
    crystalGrowth: getCrystalGrowthStats(),
    experimentPlanner: getExperimentPlannerStats(),
    crossEngineHub: crossEngineHub.getStats(),
    synthesisDiscovery: getSynthesisDiscoveryStats(),
    pipelineStageMetrics: {
      ...pipelineStageMetrics,
      prototypeSuccessRate: pipelineStageMetrics.prototypeAttempts > 0
        ? Math.round(pipelineStageMetrics.prototypeSuccesses / pipelineStageMetrics.prototypeAttempts * 1000) / 1000 : 0,
      chemistryRejectRate: autonomousTotalScreened > 0
        ? Math.round((pipelineStageMetrics.chemistryRejects + pipelineStageMetrics.stabilityPrefilterRejects) / autonomousTotalScreened * 1000) / 1000 : 0,
      phononRejectRate: autonomousTotalScreened > 0
        ? Math.round(pipelineStageMetrics.phononRejects / autonomousTotalScreened * 1000) / 1000 : 0,
      xtbSuccessRate: pipelineStageMetrics.xtbAttempts > 0
        ? Math.round(pipelineStageMetrics.xtbSuccesses / pipelineStageMetrics.xtbAttempts * 1000) / 1000 : 0,
      overallPassRate: reconciledScreened > 0
        ? Math.round(pipelineStageMetrics.totalPassed / reconciledScreened * 1000) / 1000 : 0,
    },
    lastCycleCandidates,
    lastCycleFamilyCounts,
    qeDFT: await getDFTQueueStats().catch(() => null),
    integratedSubsystems: {
      pipelineId: integratedPipelineId,
      labId: integratedLabId,
      designRepresentations: getDesignRepresentationStats(),
      theoryDiscovery: getSymbolicDiscoveryStats(),
      causalDiscovery: getCausalDiscoveryStats(),
      causalDesignGuidance,
      theoryFeedbackBias,
      lastTheoryDiscoveryCycle,
      lastCausalDiscoveryCycle,
      theoryGuidedGenerator: getTheoryGuidedGeneratorStats(),
    },
    feedbackLoops: {
      defect: {
        candidatesAdded: feedbackLoopStats.defectCandidatesAdded,
        avgTcBoost: feedbackLoopStats.defectCandidatesAdded > 0 ? feedbackLoopStats.defectTotalTcBoost / feedbackLoopStats.defectCandidatesAdded : 0,
      },
      correlation: {
        boostsApplied: feedbackLoopStats.correlationBoostsApplied,
        avgTcBoost: feedbackLoopStats.correlationBoostsApplied > 0 ? feedbackLoopStats.correlationTotalTcBoost / feedbackLoopStats.correlationBoostsApplied : 0,
      },
      synthesis: {
        feasibilityBonuses: feedbackLoopStats.synthesisFeasibilityBonuses,
        avgBoost: feedbackLoopStats.synthesisFeasibilityBonuses > 0 ? feedbackLoopStats.synthesisTotalFeasibilityBoost / feedbackLoopStats.synthesisFeasibilityBonuses : 0,
      },
      crystalGrowth: {
        qualityBonuses: feedbackLoopStats.growthQualityBonuses,
        avgBoost: feedbackLoopStats.growthQualityBonuses > 0 ? feedbackLoopStats.growthTotalQualityBoost / feedbackLoopStats.growthQualityBonuses : 0,
      },
      experimentPlanner: {
        plansGenerated: feedbackLoopStats.experimentPlansGenerated,
        dftPrioritized: feedbackLoopStats.experimentDFTPrioritized,
      },
      pressurePathways: {
        boostsApplied: feedbackLoopStats.pressurePathwayBoosts,
        bestAmbientTc: feedbackLoopStats.pressurePathwayBestAmbientTc,
        bestAmbientFormula: feedbackLoopStats.pressurePathwayBestFormula,
      },
    },
  };

  return sanitizeStatsNumeric(rawStats);
}

async function runLearningCycle() {
  if (state !== "running" || isRunningCycle) return;
  isRunningCycle = true;

  cycleCount++;
  lastCycleAt = new Date().toISOString();
  cycleInsightsThisCycle = 0;
  recentNewCandidates = 0;
  recentTcImproved = false;
  recentLogCache.clear();
  broadcast("cycleStart", { cycle: cycleCount });

  let cycleStartDetail = "";
  if (previousCycleMetrics && currentStrategyFocusAreas.length > 0) {
    const topFocus = currentStrategyFocusAreas[0]?.area || "broad exploration";
    if (previousCycleMetrics.bestTc > 0) {
      const tcTrend = previousCycleMetrics.bestScore >= 0.9 ? "scores are strong" : "still building evidence";
      cycleStartDetail = `Cycle ${cycleCount}: Focusing on ${topFocus} (${tcTrend}, best Tc: ${Math.round(previousCycleMetrics.bestTc)}K). ${previousCycleMetrics.familyDiversity} families explored so far.`;

      if (cyclesSinceTcImproved > 5) {
        setBoundaryHuntingMode(true);
        setInverseDesignMode(true);
        broadcastThought(
          `No Tc improvement in ${cyclesSinceTcImproved} cycles. Current best: ${Math.round(previousCycleMetrics.bestTc)}K. Activating boundary hunting and inverse design modes to explore instability edges...`,
          "stagnation"
        );
      }

      if (cyclesSinceTcImproved >= 12 && cyclesSinceTcImproved < 16) {
        setMutationIntensity(2);
        emit("log", {
          phase: "engine",
          event: "Tc plateau level-2 diversification",
          detail: `Tc plateau: ${cyclesSinceTcImproved} cycles without improvement. Increasing mutation magnitude (level 2: wider element swaps).`,
          dataSource: "Engine",
        });
      } else if (cyclesSinceTcImproved >= 16 && cyclesSinceTcImproved < 20) {
        setMutationIntensity(3);
        exploitCyclesRemaining = 0;
        currentExploitFamily = null;
        emit("log", {
          phase: "engine",
          event: "Tc plateau level-3 diversification",
          detail: `Tc plateau: ${cyclesSinceTcImproved} cycles without improvement. Forcing strategy switch + exotic element substitutions (level 3).`,
          dataSource: "Engine",
        });
      } else if (cyclesSinceTcImproved >= 20) {
        setMutationIntensity(3);
        setChemicalSpaceExpansionMode(true);
        exploitCyclesRemaining = 0;
        currentExploitFamily = null;
        emit("log", {
          phase: "engine",
          event: "Tc plateau level-4 chemical space expansion",
          detail: `Tc plateau: ${cyclesSinceTcImproved} cycles without improvement. Activating chemical space expansion: novel elements + unexplored structure types.`,
          dataSource: "Engine",
        });
      } else if (cyclesSinceTcImproved <= 5) {
        setChemicalSpaceExpansionMode(false);
        setMutationIntensity(1);
      }

      if (cyclesSinceTcImproved >= 8 && cyclesSinceTcImproved % 8 === 0) {
        const savedMode = getConstraintMode();
        setConstraintMode({
          empiricalPenaltyStrength: 0.5,
          allowBeyondEmpirical: true,
        });
        explorationModeActive = true;
        explorationModeSavedConstraints = savedMode;
        broadcastThought(
          `Activating adaptive exploration mode: relaxing physics constraints for 1 cycle to search unexplored compositional space. Penalty strength reduced to 0.5, lambda caps raised 30%.`,
          "strategy"
        );
        emit("log", {
          phase: "phase-6",
          event: "Adaptive exploration activated",
          detail: `Stagnation at ${cyclesSinceTcImproved} cycles. Temporarily relaxing empirical penalties for broader search.`,
          dataSource: "Engine",
        });
      } else if (previousCycleMetrics.insightCount > 0) {
        broadcastThought(
          `Cycle ${cycleCount}: Last cycle produced ${cycleInsightsThisCycle || 0} insights. Focusing on ${topFocus} — ${previousCycleMetrics.familyDiversity} families in the search space, best Tc at ${Math.round(previousCycleMetrics.bestTc)}K.`,
          "strategy"
        );
      }
    } else {
      cycleStartDetail = `Cycle ${cycleCount}: Building knowledge base. Targeting ${topFocus} for superconductor discovery.`;
      broadcastThought(`Still building the knowledge foundation. Targeting ${topFocus} as the most promising direction.`, "strategy");
    }
  } else if (cycleCount <= 3) {
    cycleStartDetail = `Cycle ${cycleCount}: Initializing knowledge base. Gathering materials, synthesis paths, and reaction data before superconductor screening.`;
    broadcastThought(`Starting up. Gathering materials from OQMD, AFLOW, and literature databases before I can begin screening for superconductors.`, "discovery");
  } else {
    cycleStartDetail = `Cycle ${cycleCount}: Continuing exploration. Materials + synthesis + reactions first, then analysis, then SC research.`;
  }

  emit("log", {
    phase: "engine",
    event: `Learning cycle ${cycleCount} started`,
    detail: cycleStartDetail,
    dataSource: "Internal",
  });

  try {
    const batch1 = await Promise.allSettled([
      runPhase4_Materials(),
      runPhase8_Synthesis(),
      runPhase9_Reactions(),
    ]);
    for (const r of batch1) {
      if (r.status === "rejected") console.log(`[Engine] Batch-1 phase failed: ${r.reason?.message?.slice(0, 120) ?? "unknown"}`);
    }

    if (state !== "running") return;

    const batch2 = await Promise.allSettled([
      runPhase3_Bonding(),
      runPhase5_Prediction(),
    ]);
    for (const r of batch2) {
      if (r.status === "rejected") console.log(`[Engine] Batch-2 phase failed: ${r.reason?.message?.slice(0, 120) ?? "unknown"}`);
    }

    if (state !== "running") return;

    await runPhase6_Discovery();

    if (state !== "running") return;

    const matCount = await storage.getMaterialCount();
    const synthCount = await storage.getSynthesisCount();
    const rxnCount = await storage.getReactionCount();

    if (matCount >= 5 && synthCount >= 3 && rxnCount >= 3) {
      await runPhase7_Superconductor();
    } else {
      emit("log", {
        phase: "phase-7",
        event: "SC research deferred",
        detail: `Waiting for more knowledge: ${matCount} materials (need 5+), ${synthCount} synthesis paths (need 3+), ${rxnCount} reactions (need 3+)`,
        dataSource: "SC Research",
      });
    }

    if (state !== "running") return;

    const scCount = await storage.getSuperconductorCount();
    if (scCount >= 3) {
      await runPhase11_StructurePrediction();
      if (state !== "running") return;
      await runPhase10_Physics();

      if (state !== "running") return;

      await runPhase12_MultiFidelity();

      if (state === "running") {
        await reEvaluateTopCandidates();
      }

      if (state === "running") {
        await runDFTEnrichment();
      }

      if (state === "running" && cycleCount % 20 === 0) {
        try {
          const failedResults = await storage.getFailedComputationalResults(500);
          const newFailures = failedResults.length - failuresSinceLastRetrain;
          if (newFailures >= 10) {
            const added = await incorporateFailureData();
            if (added > 0) {
              failuresSinceLastRetrain = failedResults.length;
              lastRetrainCycle = cycleCount;
            }
          }

          const xgbResult = await retrainXGBoostFromEvaluated(cycleCount);

          if (xgbResult.retrained) {
            const evalStats = getEvaluatedDatasetStats();
            const vHistory = getModelVersionHistory();
            const vLatest = vHistory.latestMetrics;
            emit("log", {
              phase: "engine",
              event: `XGBoost active learning retrain (cycle ${cycleCount})`,
              detail: `v${vHistory.currentVersion} | Dataset: ${xgbResult.datasetSize} samples (${xgbResult.newEntries} from eval pool). ` +
                `R²=${vLatest?.r2?.toFixed(4) ?? '?'} MAE=${vLatest?.mae?.toFixed(2) ?? '?'}K RMSE=${vLatest?.rmse?.toFixed(2) ?? '?'}K ` +
                `predVar=${vLatest?.predictionVariance?.toFixed(2) ?? '?'}K. ` +
                `Evaluated: ${evalStats.totalEvaluated} (DFT:${evalStats.bySource.dft} xTB:${evalStats.bySource.xtb} ext:${evalStats.bySource.external} AL:${evalStats.bySource.activeLearning}). ` +
                `Failures: ${getFailureExampleCount()}. Retrain #${evalStats.xgboostRetrainCount}.`,
              dataSource: "ML Engine",
            });
          }
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "XGBoost retrain error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "ML Engine",
          });
        }
      }

      if (state === "running" && cycleCount % 50 === 0) {
        try {
          const surrogateRetrained = await retrainWithAccumulatedData();
          if (surrogateRetrained > 0) {
            const sStats = getSurrogateStats();
            emit("log", {
              phase: "engine",
              event: `Surrogate model retrained`,
              detail: `v${sStats.modelVersion} | ${sStats.successExamples} success + ${sStats.failureExamples} failure examples. ` +
                `R²=${sStats.latestMetrics?.r2?.toFixed(4) ?? '?'} MAE=${sStats.latestMetrics?.mae?.toFixed(2) ?? '?'}K. ` +
                `Screen stats: ${sStats.totalScreened} screened, ${sStats.totalPassed} passed (${(sStats.passRate * 100).toFixed(1)}%)`,
              dataSource: "Surrogate Model",
            });
          }
        } catch (surrErr) {
          console.error(`[Engine] Surrogate retrain failed:`, surrErr instanceof Error ? surrErr.message.slice(0, 100) : "unknown");
        }

        try {
          await evolveRules(emit);
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Pattern mining error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Pattern Miner",
          });
        }
      }

      if (state === "running") {
        await runPhase13_SynthesisReasoning();
      }

      const alStats0 = getActiveLearningStats();
      const alCooldown = cycleCount - lastActiveLearningCycle >= 5;
      const shouldRunAL = state === "running" && cycleCount >= 15 && alCooldown && (
        (cycleCount - lastActiveLearningCycle >= 15) ||
        (alStats0.totalDFTRuns === 0 && lastActiveLearningCycle === 0)
      );
      if (shouldRunAL) {
        try {
          console.log(`[Active Learning] Triggered at cycle ${cycleCount} (last AL cycle: ${lastActiveLearningCycle})`);
          const alStats = await runActiveLearningCycle(emit, { cycleCount });
          lastActiveLearningCycle = cycleCount;
          emit("log", {
            phase: "engine",
            event: "Active learning cycle complete",
            detail: `DFT runs: ${alStats.totalDFTRuns}, retrains: ${alStats.modelRetrains}, uncertainty: ${alStats.avgUncertaintyBefore.toFixed(3)} → ${alStats.avgUncertaintyAfter.toFixed(3)}, best Tc: ${alStats.bestTcFromLoop.toFixed(1)}K`,
            dataSource: "Active Learning",
          });
          autonomousGNNRetrainCount += alStats.modelRetrains > 0 ? 1 : 0;

          const calStats = getCalibrationStats();
          if (calStats.totalEvaluations > 0) {
            emit("log", {
              phase: "engine",
              event: "Feedback loop calibration",
              detail: `${calStats.totalEvaluations} evaluations, mean abs error: ${calStats.globalMeanAbsError.toFixed(1)}K, overestimate ratio: ${(calStats.globalOverestimateRatio * 100).toFixed(0)}%, families tracked: ${calStats.familyCalibrations.length}, exploration weight: ${(calStats.explorationWeight * 100).toFixed(1)}%`,
              dataSource: "Surrogate Fitness",
            });
          }

          try {
            const topCandidates = await storage.getSuperconductorCandidates(10);
            let pressureCurvesAnalyzed = 0;
            let transitionsFound = 0;
            for (const c of topCandidates.slice(0, 5)) {
              try {
                predictPressureCurve(c.formula);
                const optimal = findOptimalPressure(c.formula);
                const transitions = detectPhaseTransitions(c.formula);
                pressureCurvesAnalyzed++;
                transitionsFound += transitions.length;
                if (optimal.maxTc > 77 && optimal.optimalPressureGpa > 0) {
                  emit("log", {
                    phase: "engine",
                    event: "Pressure-Tc curve",
                    detail: `${c.formula}: peak Tc=${optimal.maxTc.toFixed(1)}K at ${optimal.optimalPressureGpa} GPa${transitions.length > 0 ? `, ${transitions.length} phase transition(s)` : ''}`,
                    dataSource: "Pressure Surrogate",
                  });
                }
              } catch { /* skip */ }
            }
            const pcStats = getPressureCurveStats();
            const ptStats = getPhaseTransitionStats();
            emit("log", {
              phase: "engine",
              event: "Pressure analysis summary",
              detail: `Curves: ${pcStats.totalCurves} cached (${pressureCurvesAnalyzed} new this cycle), transitions: ${ptStats.totalTransitionsDetected} total (${transitionsFound} this cycle), high-Tc materials: ${pcStats.highTcCount}, pressure-sensitive: ${pcStats.sensitiveCount}`,
              dataSource: "Pressure Surrogate",
            });

            let enthalpyStableCount = 0;
            for (const c of topCandidates.slice(0, 5)) {
              try {
                const window = findStabilityPressureWindow(c.formula);
                if (window && window.maxPressureGpa > window.minPressureGpa) {
                  enthalpyStableCount++;
                }
              } catch { /* skip */ }
            }
            const hStats = getEnthalpyStats();
            emit("log", {
              phase: "engine",
              event: "Enthalpy stability summary",
              detail: `Computed: ${hStats.totalComputed}, stable: ${hStats.stableCount}, metastable: ${hStats.metastableCount}, unstable: ${hStats.unstableCount}, avg stability window: ${hStats.avgStabilityWindow} GPa, stable windows found: ${enthalpyStableCount}/5 top candidates`,
              dataSource: "Enthalpy H=E+PV",
            });

            let profilesBuilt = 0;
            let bayesOptCount = 0;
            for (const c of topCandidates.slice(0, 5)) {
              try {
                const profile = buildPressureResponseProfile(c.formula);
                profilesBuilt++;

                for (const pt of profile.tcVsPressure) {
                  const stabPt = profile.stabilityVsPressure.find(s => s.pressure === pt.pressure);
                  addPressureObservation(c.formula, pt.pressure, pt.tc, stabPt?.stable ?? false, stabPt?.enthalpy ?? 0);
                }

                const bayesResult = optimizePressureForFormula(c.formula, 3, 15);
                bayesOptCount++;
                if (bayesResult.predictedTcAtOptimal > 50) {
                  emit("log", {
                    phase: "engine",
                    event: "Bayesian pressure optimization",
                    detail: `${c.formula}: optimal P=${bayesResult.optimalPressure} GPa, predicted Tc=${bayesResult.predictedTcAtOptimal}K, stable=${bayesResult.stableAtOptimal}, confidence=${bayesResult.confidence}`,
                    dataSource: "Bayesian Pressure GP",
                  });
                }
              } catch {}
            }
            for (const c of topCandidates.slice(0, 8)) {
              try {
                const cPressure = c.pressureGpa ?? 0;
                const screen = fastPressureScreen(c.formula, cPressure);
                const isHit = screen.passesPrescreen && (c.predictedTc ?? 0) > 20;
                recordClusterDiscovery(c.formula, cPressure, c.predictedTc ?? 0, isHit);
              } catch {}
            }

            const mapStats = getPressurePropertyMapStats();
            const bpStats = getBayesianPressureStats();
            const clusterStats = getPressureClusterStats();
            emit("log", {
              phase: "engine",
              event: "Pressure maps and Bayesian summary",
              detail: `Profiles: ${mapStats.totalProfiles} (${profilesBuilt} new), avg peak Tc=${mapStats.avgPeakTc}K at ${mapStats.avgPeakPressure} GPa, low-P high-Tc: ${mapStats.lowPressureHighTcCount} | Bayesian: ${bpStats.totalOptimizations} optimized, avg optimal P=${bpStats.avgOptimalPressure} GPa, low-P optimal: ${bpStats.lowPressureOptimalCount}`,
              dataSource: "Pressure Property Maps + Bayesian GP",
            });

            emit("log", {
              phase: "engine",
              event: "Pressure cluster analysis",
              detail: `Discoveries: ${clusterStats.totalDiscoveries}, most productive: ${clusterStats.mostProductiveCluster}, clusters: ${clusterStats.clusters.map(c => `${c.id}(w=${c.weight.toFixed(2)},hits=${c.discoveryCount})`).join(', ')}`,
              dataSource: "Pressure Clusters",
            });
          } catch (pressureErr: any) {
            console.error(`[Pressure Analysis] Error: ${pressureErr?.message?.slice(0, 150)}`);
          }
        } catch (err: any) {
          lastActiveLearningCycle = cycleCount;
          console.log(`[Active Learning] Error at cycle ${cycleCount}: ${err.message?.slice(0, 150)}`);
          emit("log", {
            phase: "engine",
            event: "Active learning error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Active Learning",
          });
        }
      }

      if (state === "running" && cycleCount % 10 === 0) {
        try {
          const topForMutation = await storage.getSuperconductorCandidatesByTc(10);
          if (topForMutation.length > 0) {
            const mutationInput = topForMutation.map(c => ({
              formula: c.formula,
              predictedTc: c.predictedTc,
            }));
            const mutResult = runStructuralMutations(mutationInput, emit);

            let mutInserted = 0;
            const mutantFormulas = [
              ...mutResult.distorted.filter(d => d.energyPenalty < 0.3).map(d => d.formula),
              ...mutResult.layered.map(l => l.formula),
              ...mutResult.vacancy.map(v => v.formula),
              ...mutResult.strained.filter(s => Math.abs(s.strainPercent) < 5).map(s => s.formula),
            ];
            for (const mf of mutantFormulas.slice(0, 20)) {
              if (!shouldContinue()) break;
              if (!isValidFormula(mf)) continue;
              const existing = await storage.getSuperconductorByFormula(mf);
              if (!existing) {
                const features = extractFeatures(mf);
                const gb = gbPredict(features);
                if (gb.tcPredicted >= 10) {
                  const lambdaML = features.electronPhononLambda ?? 0;
                  const metallicityML = features.metallicity ?? 0.5;
                  let rawTc = estimateRawTc(lambdaML, features.logPhononFreq);
                  rawTc = applyAmbientTcCap(rawTc, lambdaML, 0, metallicityML, mf);
                  const id = `sc-mut-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                  try {
                    const inserted = await insertCandidateWithStabilityCheck({
                      id,
                      name: mf,
                      formula: mf,
                      predictedTc: rawTc,
                      pressureGpa: estimateFamilyPressure(mf),
                      meissnerEffect: false,
                      zeroResistance: false,
                      cooperPairMechanism: "Structural mutation",
                      crystalStructure: null,
                      quantumCoherence: 0.5,
                      stabilityScore: features.cooperPairStrength,
                      synthesisPath: null,
                      mlFeatures: features as any,
                      xgboostScore: gb.score,
                      neuralNetScore: 0.5,
                      ensembleScore: Math.min(0.9, (gb.score + 0.5) / 2),
                      roomTempViable: false,
                      status: "theoretical",
                      notes: "[structural-mutation]",
                      electronPhononCoupling: lambdaML || null,
                      logPhononFrequency: features.logPhononFreq ?? null,
                      coulombPseudopotential: 0.12,
                      pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
                      pairingMechanism: "phonon-mediated",
                      correlationStrength: features.correlationStrength ?? null,
                      dimensionality: "3D",
                      fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                      uncertaintyEstimate: 0.6,
                      verificationStage: 0,
                      dataConfidence: "low",
                    }, "massive_combinatorial");
                    if (inserted) {
                      totalScCandidates++;
                      mutInserted++;
                    }
                  } catch (mutInsErr) {
                    console.error(`[Engine] Mutation candidate insert failed:`, mutInsErr instanceof Error ? mutInsErr.message.slice(0, 80) : "unknown");
                  }
                }
              }
            }
            if (mutInserted > 0) {
              emit("log", {
                phase: "engine",
                event: "Structural mutation candidates inserted",
                detail: `${mutInserted} viable mutant candidates from ${mutResult.totalGenerated} structural variants`,
                dataSource: "Structural Mutator",
              });
            }
          }
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Structural mutation error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Structural Mutator",
          });
        }
      }

      if (state === "running" && cycleCount >= 20 && cycleCount % 20 === 0) {
        try {
          const focusFamily = currentStrategyFocusAreas[0]?.area || "Carbides";
          const FAMILY_ELEMENT_SETS: Record<string, string[][]> = {
            Carbides: [["Nb","C"],["Ti","C"],["Mo","C"],["V","C"],["Nb","Ti","C"],["Nb","Mo","C"]],
            Borides: [["Nb","B"],["Ti","B"],["Zr","B"],["Mg","B"],["Nb","Ti","B"]],
            Nitrides: [["Nb","N"],["Ti","N"],["Zr","N"],["V","N"],["Nb","Ti","N"]],
            Hydrides: [["La","H"],["Y","H"],["Ca","H"],["Sr","H"],["La","Y","H"]],
            Intermetallics: [["Nb","Sn"],["V","Si"],["Nb","Ge"],["Nb","Al"]],
          };
          const elementSets = FAMILY_ELEMENT_SETS[focusFamily] || FAMILY_ELEMENT_SETS["Carbides"];
          const chosenSet = elementSets[cycleCount % elementSets.length];
          const optimalResults = findOptimalRegion(chosenSet, emit);
          const seedFormulas = optimalResults
            .filter(r => r.predictedTc > 10 && r.hullDistance < 0.3)
            .map(r => r.formula)
            .slice(0, 5);
          if (seedFormulas.length > 0) {
            for (const sf of seedFormulas) {
              if (!shouldContinue()) break;
              if (!isValidFormula(sf)) continue;
              const existing = await storage.getSuperconductorByFormula(sf);
              if (!existing) {
                const features = extractFeatures(sf);
                const gb = gbPredict(features);
                if (gb.tcPredicted >= 10) {
                  try {
                    const inserted = await insertCandidateWithStabilityCheck({
                      formula: normalizeFormula(sf),
                      predictedTc: Math.round(gb.tcPredicted),
                      dataConfidence: "low",
                      ensembleScore: Math.min(0.9, gb.score),
                      verificationStage: 0,
                      notes: `[phase-explorer: optimal from ${chosenSet.join("-")} scan, family=${classifyFamily(sf)}]`,
                    }, "massive_combinatorial");
                    if (inserted) {
                      totalScCandidates++;
                      recentNewCandidates++;
                    }
                  } catch (optInsErr) {
                    console.error(`[Engine] Optimal region insert failed:`, optInsErr instanceof Error ? optInsErr.message.slice(0, 80) : "unknown");
                  }
                }
              }
            }
          }
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Phase exploration error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Phase Explorer",
          });
        }
      }

      if (state === "running" && cycleCount >= 15 && cycleCount % 25 === 0) {
        try {
          const prototypeCandidates = runPrototypeGeneration();
          const prototypeCounts: Record<string, { generated: number; passed: number; inserted: number }> = {};
          let protoGenerated = 0;
          let protoPassedStability = 0;
          let protoInserted = 0;
          let bestDiscoveryScore = 0;

          const existingCandidates = await storage.getSuperconductorCandidates(500);
          const existingFormulas = new Set(existingCandidates.map(c => c.formula));
          const existingFormulaList = existingCandidates.map(c => c.formula);

          const deduped = prototypeCandidates.filter(pc => {
            const normalized = normalizeFormula(pc.formula);
            return !existingFormulas.has(normalized) && isValidFormula(pc.formula);
          });

          protoGenerated = deduped.length;

          const scored: {
            pc: PrototypeCandidate;
            normalized: string;
            features: ReturnType<typeof extractFeatures>;
            gbResult: ReturnType<typeof gbPredict>;
            gnnResult: ReturnType<typeof gnnPredictWithUncertainty>;
            discoveryScore: number;
            discoveryDetails: ReturnType<typeof computeDiscoveryScore>;
          }[] = [];

          for (const pc of deduped) {
            if (!shouldContinue()) break;
            const normalized = normalizeFormula(pc.formula);

            const features = extractFeatures(normalized);
            const gbResult = gbPredict(features);
            const protoIsHydride = pc.prototype?.toLowerCase().includes("hydride") || pc.prototype?.toLowerCase().includes("clathrate");
            const protoPressure = protoIsHydride ? 150 : estimateFamilyPressure(normalized);
            const gnnResult = gnnPredictWithUncertainty(normalized, pc.prototype, protoPressure);

            const familyMap: Record<string, string> = {
              "MAX-phase": "MAX-phase",
              "AlB2-type": "Boride",
              "Clathrate": "Hydride",
              "Sodalite": "Hydride",
              "Layered nitride": "Nitride",
            };
            const familyKey = familyMap[pc.prototype];
            if (familyKey) {
              const filterResult = applyFamilyFilter(normalized, familyKey, features);
              if (!filterResult.pass) continue;
            }

            let protoTopoScore = 0;
            try {
              const protoElectronic = computeElectronicStructure(normalized, null);
              const protoTopo = analyzeTopology(normalized, protoElectronic, undefined, pc.crystalSystem);
              protoTopoScore = protoTopo.topologicalScore;
              trackTopologyResult(protoTopo);
            } catch (e) { console.error(`[Engine] Prototype topology analysis failed for ${normalized}:`, e); }

            const discoveryDetails = computeDiscoveryScore({
              predictedTc: gbResult.tcPredicted,
              formula: normalized,
              hullDistance: null,
              synthesisScore: null,
              prototype: pc.prototype,
              existingFormulas: existingFormulaList.slice(0, 100),
              topologicalScore: protoTopoScore,
              uncertaintyEstimate: gnnResult?.uncertainty ?? 0.5,
            });

            if (!prototypeCounts[pc.prototype]) {
              prototypeCounts[pc.prototype] = { generated: 0, passed: 0, inserted: 0 };
            }
            prototypeCounts[pc.prototype].generated++;

            scored.push({
              pc,
              normalized,
              features,
              gbResult,
              gnnResult,
              discoveryScore: discoveryDetails.discoveryScore,
              discoveryDetails,
            });
          }

          scored.sort((a, b) => b.discoveryScore - a.discoveryScore);
          const topProto = scored.slice(0, 30);

          for (const entry of topProto) {
            if (!shouldContinue()) break;
            const { pc, normalized, features, gbResult, gnnResult, discoveryScore, discoveryDetails } = entry;

            const lambdaML = features.electronPhononLambda ?? 0;
            const metallicityML = features.metallicity ?? 0.5;
            const isHydride = pc.prototype === "Clathrate" || pc.prototype === "Sodalite";
            let predictedTc: number;
            if (gnnResult.confidence > 0.3 && gnnResult.tc > 0) {
              predictedTc = Math.round(gnnResult.tc * 0.6 + gbResult.tcPredicted * 0.4);
            } else {
              predictedTc = Math.round(gbResult.tcPredicted);
            }
            predictedTc = applyAmbientTcCap(predictedTc, lambdaML, isHydride ? 150 : 0, metallicityML, normalized);

            try {
              const id = `sc-proto-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
              const siteStr = Object.entries(pc.siteAssignment).map(([k, v]) => `${k}=${v.join(",")}`).join("; ");
              const inserted = await insertCandidateWithStabilityCheck({
                id,
                name: `${pc.prototype} ${normalized}`,
                formula: normalized,
                predictedTc,
                pressureGpa: isHydride ? 150 : estimateFamilyPressure(normalized),
                meissnerEffect: false,
                zeroResistance: false,
                cooperPairMechanism: `${pc.prototype} prototype search`,
                crystalStructure: `${pc.spaceGroup} (${pc.crystalSystem})`,
                quantumCoherence: null,
                stabilityScore: features.cooperPairStrength ?? null,
                synthesisPath: null,
                mlFeatures: {
                  ...features as any,
                  prototype: pc.prototype,
                  spaceGroup: pc.spaceGroup,
                  crystalSystem: pc.crystalSystem,
                  dimensionality: pc.dimensionality,
                  siteAssignment: pc.siteAssignment,
                  gnnUncertainty: gnnResult.uncertainty,
                  gnnLambda: gnnResult.lambda,
                  gnnTc: gnnResult.tc,
                },
                xgboostScore: gbResult.score,
                neuralNetScore: gnnResult.confidence,
                ensembleScore: Math.min(0.9, gnnResult.confidence * 0.6 + gbResult.score * 0.3 + (discoveryScore > 0.5 ? 0.1 : 0.05)),
                roomTempViable: false,
                status: "theoretical",
                notes: `[${pc.prototype} prototype: ${pc.spaceGroup}, ${pc.crystalSystem}, ${pc.dimensionality}, sites: ${siteStr}] [Discovery: ${discoveryScore.toFixed(3)}]`,
                electronPhononCoupling: gnnResult.lambda || lambdaML || null,
                logPhononFrequency: features.logPhononFreq ?? null,
                coulombPseudopotential: isHydride ? 0.10 : 0.12,
                pairingSymmetry: derivePairingSymmetry("phonon-mediated", features.dWaveSymmetry),
                pairingMechanism: "phonon-mediated",
                correlationStrength: features.correlationStrength ?? null,
                dimensionality: pc.dimensionality === "3D" ? "3D" : "2D",
                fermiSurfaceTopology: features.fermiSurfaceType ?? null,
                uncertaintyEstimate: gnnResult.uncertainty,
                verificationStage: 0,
                dataConfidence: "low",
                discoveryScore,
              }, "structure_diffusion");
              bayesianOptimizer.addObservation(normalized, predictedTc, gnnResult.lambda || lambdaML, discoveryScore);

              if (inserted) {
                prototypeCounts[pc.prototype].passed++;
                protoPassedStability++;
                prototypeCounts[pc.prototype].inserted++;
                protoInserted++;
                totalScCandidates++;
                recentNewCandidates++;
                if (discoveryScore > bestDiscoveryScore) {
                  bestDiscoveryScore = discoveryScore;
                }
              }
            } catch (e) { console.error(`[Engine] Prototype candidate insert failed:`, e); }
          }

          const protoSummary = Object.entries(prototypeCounts)
            .map(([p, c]) => `${c.generated} ${p}`)
            .join(", ");
          emit("log", {
            phase: "engine",
            event: "Prototype search complete",
            detail: `Prototype search: ${protoSummary}, ${protoPassedStability} passed stability, ${protoInserted} inserted (best discovery score: ${bestDiscoveryScore.toFixed(3)})`,
            dataSource: "Prototype Generator",
          });
        } catch (err: any) {
          emit("log", {
            phase: "engine",
            event: "Prototype search error",
            detail: err.message?.slice(0, 150) || "unknown",
            dataSource: "Prototype Generator",
          });
        }
      }

      if (state === "running" && cycleCount >= 12 && cycleCount % 25 === 12) {
        try {
          const existingCandidates = await storage.getSuperconductorCandidates(500);
          const existingFormulas = new Set(existingCandidates.map(c => c.formula));

          const scElements = ["Nb", "Ti", "V", "Zr", "Mo", "Ta", "W", "La", "Y", "Ca", "Sr", "Ba",
            "Fe", "Cu", "Ni", "Co", "Mn", "Sc", "Hf", "Re"];
          const anions = ["H", "N", "B", "C", "O", "S", "Se", "P", "As", "Te"];
          const lfFormulas: string[] = [];
          for (let fi = 0; fi < 60; fi++) {
            const m1 = scElements[Math.floor(Math.random() * scElements.length)];
            const m2 = scElements[Math.floor(Math.random() * scElements.length)];
            const an = anions[Math.floor(Math.random() * anions.length)];
            if (m1 === m2) continue;
            const n1 = Math.floor(Math.random() * 3) + 1;
            const n2 = Math.floor(Math.random() * 3) + 1;
            const n3 = Math.floor(Math.random() * 6) + 1;
            const f = `${m1}${n1 > 1 ? n1 : ""}${m2}${n2 > 1 ? n2 : ""}${an}${n3 > 1 ? n3 : ""}`;
            const nf = normalizeFormula(f);
            if (!existingFormulas.has(nf) && isValidFormula(f)) lfFormulas.push(f);
          }

          const lfStructures = generatePrototypeFreeStructures(lfFormulas, 30);
          let lfInserted = 0;
          let lfBest = 0;

          for (const struct of lfStructures) {
            try {
              const normalized = normalizeFormula(struct.formula);
              if (existingFormulas.has(normalized)) continue;
              const features = extractFeatures(struct.formula);
              if (!features) continue;
              const gb = gbPredict(features);
              if (gb.tcPredicted < 5) continue;

              const cappedTc = Math.round(gb.tcPredicted);
              const inserted = await insertCandidateWithStabilityCheck({
                formula: normalized,
                predictedTc: cappedTc,
                dataConfidence: "low",
                ensembleScore: Math.min(0.9, gb.score),
                verificationStage: 0,
                crystalStructure: struct.bravaisType,
                notes: `[lattice-free: ${struct.bravaisType}, ${struct.atoms.length} atoms, vol/atom=${struct.volumePerAtom.toFixed(1)} A^3]`,
              }, "structure_diffusion");
              if (inserted) {
                lfInserted++;
                totalScCandidates++;
                existingFormulas.add(normalized);
                if (cappedTc > lfBest) lfBest = cappedTc;
              }
            } catch (lfErr: any) {
              console.error(`[Engine] Lattice-free insert failed:`, lfErr?.message?.slice(0, 80) ?? "unknown");
            }
          }

          seedEvoPopulation(lfStructures);

          const evoOffspring = runEvolutionaryGeneration(15);
          let evoInserted = 0;
          let evoBest = 0;
          for (const evoStruct of evoOffspring) {
            try {
              const normalized = normalizeFormula(evoStruct.formula);
              if (existingFormulas.has(normalized)) continue;
              const features = extractFeatures(evoStruct.formula);
              if (!features) continue;
              const gb = gbPredict(features);
              if (gb.tcPredicted < 5) continue;

              const cappedTc = Math.round(gb.tcPredicted);
              const inserted = await insertCandidateWithStabilityCheck({
                formula: normalized,
                predictedTc: cappedTc,
                dataConfidence: "low",
                ensembleScore: Math.min(0.9, gb.score),
                verificationStage: 0,
                crystalStructure: evoStruct.bravaisType,
                notes: `[evo-${evoStruct.generationMethod}: ${evoStruct.bravaisType}, ${evoStruct.atoms.length} atoms, vol/atom=${evoStruct.volumePerAtom.toFixed(1)} A^3]`,
              }, "structure_diffusion");
              if (inserted) {
                evoInserted++;
                totalScCandidates++;
                existingFormulas.add(normalized);
                if (cappedTc > evoBest) evoBest = cappedTc;
                addToEvoPopulation(evoStruct, gb.score);
              }
            } catch (evoErr: any) {
              console.error(`[Engine] Evo insert failed:`, evoErr?.message?.slice(0, 80) ?? "unknown");
            }
          }

          const lfStats = getLatticeGeneratorStats();
          const evoSummary = getEvoPopulationSummary();
          emit("log", {
            phase: "engine",
            event: "Lattice-free generation complete",
            detail: `Random: ${lfStructures.length} generated, ${lfInserted} inserted (best ${lfBest}K). Evo: gen ${evoSummary.generation}, pop ${evoSummary.size}, ${evoOffspring.length} offspring, ${evoInserted} inserted (best ${evoBest}K). Fitness: best=${evoSummary.bestFitness.toFixed(3)}, avg=${evoSummary.avgFitness.toFixed(3)}, diversity=${evoSummary.formulaDiversity}`,
            dataSource: "Lattice Generator",
          });
        } catch (lfOuterErr: any) {
          emit("log", {
            phase: "engine",
            event: "Lattice-free generation error",
            detail: lfOuterErr?.message?.slice(0, 150) || "unknown",
            dataSource: "Lattice Generator",
          });
        }
      }

      if (state === "running" && cycleCount >= 10) {
        await runAutonomousFastPath();
      }
    } else {
      emit("log", {
        phase: "phase-10",
        event: "Physics pipeline deferred",
        detail: `Waiting for SC candidates: ${scCount} (need 3+)`,
        dataSource: "Physics Engine",
      });
    }

    if (state === "running") {
      try {
        const prevFocusAreas = currentStrategyFocusAreas.map(f => ({ ...f }));
        const strategy = await analyzeAndEvolveStrategy(emit, cycleCount);
        if (strategy) {
          const llmTopFamily = strategy.focusAreas[0]?.area || "";
          const llmTopMaxTc = (strategy.performanceSignals?.familyStats as any)?.[llmTopFamily]?.maxTc ?? 0;
          const currentMaxTc = currentExploitFamily
            ? ((strategy.performanceSignals?.familyStats as any)?.[currentExploitFamily]?.maxTc ?? 0)
            : 0;

          const exploreProbability = 0.15;
          const shouldRandomExplore = Math.random() < exploreProbability;
          const underExplored = strategy.performanceSignals?.underExplored as string[] | undefined;

          if (exploitCyclesRemaining > 0 && currentExploitFamily) {
            const tcGapJustifiesSwitch = llmTopFamily !== currentExploitFamily && llmTopMaxTc > currentMaxTc + 20;

            if (tcGapJustifiesSwitch) {
              currentExploitFamily = llmTopFamily;
              exploitCyclesRemaining = 8;
              emit("log", {
                phase: "engine",
                event: "Strategy override",
                detail: `Switching to ${llmTopFamily} (Tc ${llmTopMaxTc}K vs ${currentMaxTc}K) — large Tc gap overrides exploit window`,
                dataSource: "Strategy Analyzer",
              });
            } else if (shouldRandomExplore && underExplored && underExplored.length > 0) {
              const randomFamily = underExplored[Math.floor(Math.random() * underExplored.length)];
              const explorationAreas = [
                { area: randomFamily, priority: 0.8, reasoning: "Random exploration of under-explored family" },
                ...strategy.focusAreas.filter(f => f.area !== randomFamily).slice(0, 4),
              ];
              strategy.focusAreas = explorationAreas as any;
              emit("log", {
                phase: "engine",
                event: "Exploration probe",
                detail: `Random exploration: probing ${randomFamily} (${exploitCyclesRemaining} exploit cycles remaining for ${currentExploitFamily})`,
                dataSource: "Strategy Analyzer",
              });
            } else {
              const currentFamilyArea = strategy.focusAreas.find(f => f.area === currentExploitFamily);
              if (currentFamilyArea) {
                currentFamilyArea.priority = Math.max(currentFamilyArea.priority, 0.8);
              }
              strategy.focusAreas.sort((a, b) => {
                if (a.area === currentExploitFamily) return -1;
                if (b.area === currentExploitFamily) return 1;
                return b.priority - a.priority;
              });
            }
            exploitCyclesRemaining--;
          } else {
            currentExploitFamily = llmTopFamily;
            exploitCyclesRemaining = 8;
            emit("log", {
              phase: "engine",
              event: "Exploit window started",
              detail: `Locking onto ${llmTopFamily} for 8 cycles (Tc: ${llmTopMaxTc}K)`,
              dataSource: "Strategy Analyzer",
            });
          }

          currentStrategyHint = strategy.focusAreas
            .slice(0, 3)
            .map(f => f.area)
            .join(", ");
          currentStrategyFocusAreas = strategy.focusAreas.map(f => ({ area: f.area, priority: f.priority }));
          if (strategy.performanceSignals?.familyStats) {
            currentFamilyCounts = {};
            for (const [fam, stats] of Object.entries(strategy.performanceSignals.familyStats as Record<string, { count: number }>)) {
              currentFamilyCounts[fam] = stats.count;
            }
          }
          broadcast("strategyUpdate", {
            cycle: cycleCount,
            focusAreas: strategy.focusAreas,
            summary: strategy.summary,
          });

          for (const fa of strategy.focusAreas) {
            const prev = prevFocusAreas.find(p => p.area === fa.area);
            if (prev && Math.abs(fa.priority - prev.priority) > 0.15) {
              const dir = fa.priority > prev.priority ? "Promoting" : "Deprioritizing";
              broadcastThought(
                `${dir} ${fa.area} from ${(prev.priority * 100).toFixed(0)}% to ${(fa.priority * 100).toFixed(0)}% priority based on recent performance data.`,
                "strategy"
              );
            }
          }
          const newFamilies = strategy.focusAreas.filter(fa => !prevFocusAreas.find(p => p.area === fa.area));
          for (const nf of newFamilies) {
            broadcastThought(`New focus area: ${nf.area} added to research strategy at ${(nf.priority * 100).toFixed(0)}% priority.`, "discovery");
          }
        }
      } catch (e) { console.error("[Engine] Strategy analysis failed:", e); }

      try {
        await runModelImprovementCycle(emit, cycleCount);
      } catch (e) { console.error("[Engine] Model improvement cycle failed:", e); }

      try {
        await captureConvergenceSnapshot(emit, cycleCount, currentStrategyHint || undefined);
        const snapshots = await storage.getConvergenceSnapshots(5);
        if (snapshots.length > 0) {
          broadcast("convergenceUpdate", {
            latest: snapshots[snapshots.length - 1],
            total: snapshots.length,
          });
        }
      } catch (e) { console.error("[Engine] Convergence snapshot failed:", e); }

      try {
        await checkMilestones(emit, broadcast, cycleCount, cycleInsightsThisCycle);
      } catch (e) { console.error("[Engine] Milestone check failed:", e); }

      try {
        await scanMaterialSignals(emit, broadcast, cycleCount);
      } catch (e) { console.error("[Engine] Material signal scan failed:", e); }

      try {
        const currentCandidates = await storage.getSuperconductorCandidates(50);
        let currentBestTc = 0;
        let currentBestScore = 0;
        for (const c of currentCandidates) {
          if ((c.predictedTc ?? 0) > currentBestTc) currentBestTc = c.predictedTc ?? 0;
          if ((c.ensembleScore ?? 0) > currentBestScore) currentBestScore = c.ensembleScore ?? 0;
        }
        const { classifyFamily: classifyFam } = await import("./utils");
        const currentFamilies = new Set(currentCandidates.map(c => classifyFam(c.formula)));
        const currentDiversity = currentFamilies.size;
        const currentInsightCount = await storage.getNovelInsightCount();
        const stats = await storage.getStats();
        const pipelineTotal = stats.pipelineStages.reduce((s, p) => s + p.count, 0);
        const pipelinePassed = stats.pipelineStages.reduce((s, p) => s + p.passed, 0);

        let endSummaryParts: string[] = [];
        endSummaryParts.push(`${currentCandidates.length} total candidates`);

        if (previousCycleMetrics) {
          const tcDelta = currentBestTc - previousCycleMetrics.bestTc;
          if (tcDelta > 1) {
            endSummaryParts.push(`best Tc improved by ${Math.round(tcDelta)}K to ${Math.round(currentBestTc)}K`);
            recentTcImproved = true;
          } else if (currentBestTc > 0) {
            endSummaryParts.push(`best Tc unchanged at ${Math.round(currentBestTc)}K`);
          }

          const scoreDelta = currentBestScore - previousCycleMetrics.bestScore;
          if (scoreDelta > 0.005) {
            endSummaryParts.push(`top score rose to ${currentBestScore.toFixed(3)}`);
          }

          const diversityDelta = currentDiversity - previousCycleMetrics.familyDiversity;
          if (diversityDelta > 0) {
            endSummaryParts.push(`diversity expanded to ${currentDiversity} families (+${diversityDelta})`);
            broadcastThought(`Search space expanded to ${currentDiversity} material families. New territory to explore.`, "discovery");
          }

          recentNewCandidates = currentCandidates.length - previousCycleMetrics.candidateCount;

          if (cycleInsightsThisCycle > 0) {
            endSummaryParts.push(`${cycleInsightsThisCycle} novel insights`);
          }
        } else {
          if (currentBestTc > 0) endSummaryParts.push(`best Tc: ${Math.round(currentBestTc)}K`);
          if (cycleInsightsThisCycle > 0) endSummaryParts.push(`${cycleInsightsThisCycle} novel insights`);
        }

        emit("log", {
          phase: "engine",
          event: `Cycle ${cycleCount} complete`,
          detail: endSummaryParts.join(". ") + ".",
          dataSource: "Internal",
        });

        const narrativeParts: string[] = [`Cycle ${cycleCount}:`];
        if (recentNewCandidates > 0) narrativeParts.push(`${recentNewCandidates} new candidates discovered.`);
        if (recentTcImproved) narrativeParts.push(`Tc record improved to ${Math.round(currentBestTc)}K.`);
        else if (currentBestTc > 0) narrativeParts.push(`Best Tc holds at ${Math.round(currentBestTc)}K.`);
        if (cycleInsightsThisCycle > 0) narrativeParts.push(`${cycleInsightsThisCycle} novel insight${cycleInsightsThisCycle > 1 ? "s" : ""} discovered.`);
        const topFam = currentStrategyFocusAreas[0]?.area;
        if (topFam) narrativeParts.push(`Strategy: ${topFam} focus.`);

        storage.insertResearchLog({
          phase: "engine",
          event: "cycle-narrative",
          detail: narrativeParts.join(" "),
          dataSource: "Internal",
        }).catch(() => {});

        updateTempo();
        currentStatusMessage = generateStatusMessage();
        broadcast("statusMessage", { message: currentStatusMessage, tempo: engineTempo });

        const topFamily = currentStrategyFocusAreas[0]?.area || "";
        previousCycleMetrics = {
          bestTc: currentBestTc,
          bestScore: currentBestScore,
          candidateCount: currentCandidates.length,
          familyDiversity: currentDiversity,
          insightCount: currentInsightCount,
          topFamily,
          pipelinePassed,
          pipelineTotal,
        };
      } catch (e) { console.error("[Engine] Cycle metrics capture failed:", e); }
    }
  } catch (err: any) {
    emit("log", {
      phase: "engine",
      event: "Cycle error",
      detail: err.message?.slice(0, 200) || "Unknown",
      dataSource: "Internal",
    });
  } finally {
    if (explorationModeActive && explorationModeSavedConstraints) {
      setConstraintMode(explorationModeSavedConstraints);
      explorationModeActive = false;
      explorationModeSavedConstraints = null;
      emit("log", {
        phase: "engine",
        event: "Adaptive exploration deactivated",
        detail: "Restored normal physics constraints after exploration cycle.",
        dataSource: "Engine",
      });
    }

    isRunningCycle = false;
    broadcast("cycleEnd", { cycle: cycleCount });

    if (state === "running") {
      cycleTimer = setTimeout(runLearningCycle, cycleIntervalMs);
    }
  }
}

export function initWebSocket(server: Server) {
  wss = new WebSocketServer({ server, path: "/ws" });

  wss.on("connection", (ws) => {
    ws.send(
      JSON.stringify({
        type: "status",
        data: getStatus(),
        timestamp: new Date().toISOString(),
      })
    );
  });

  setDFTBroadcast((event: string, data: any) => {
    broadcast(event, data);
  });

  console.log("WebSocket server initialized on /ws");
}

async function backfillGBScores() {
  const yield_ = () => new Promise<void>(r => setTimeout(r, 0));
  try {
    let totalUpdated = 0;
    let totalFailed = 0;
    let batch: any[];
    do {
      batch = await storage.getUnscoredCandidates(200);
      if (batch.length === 0) break;

      for (let i = 0; i < batch.length; i++) {
        if (i % 10 === 0) await yield_();
        const c = batch[i];
        try {
          const features = extractFeatures(c.formula);
          const gb = gbPredict(features);
          const nnScore = c.quantumCoherence ?? 0.3;
          const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);
          await storage.updateSuperconductorCandidate(c.id, {
            xgboostScore: gb.score,
            neuralNetScore: nnScore,
            ensembleScore: ensemble,
          });
          totalUpdated++;
        } catch {
          try {
            await storage.updateSuperconductorCandidate(c.id, {
              xgboostScore: 0.3,
              neuralNetScore: 0.3,
              ensembleScore: 0.3,
            });
          } catch (e) { console.error("[Engine] GB score backfill update failed:", e); }
          totalFailed++;
        }
      }
    } while (batch.length === 200);

    if (totalUpdated > 0 || totalFailed > 0) {
      emit("log", {
        phase: "engine",
        event: "GB score backfill complete",
        detail: `Scored ${totalUpdated} candidates with gradient boosting model${totalFailed > 0 ? `, ${totalFailed} failed (set to default)` : ''}`,
        dataSource: "Internal",
      });
    }
  } catch (e) { console.error("[Engine] GB score backfill failed:", e); }
}

const PHYSICS_VERSION = 15;

async function recalculatePhysics() {
  const yield_ = () => new Promise<void>(r => setTimeout(r, 0));
  try {
    let totalRecalculated = 0;
    const batchSize = 200;

    while (true) {
      const needsRecalc = await storage.getCandidatesNeedingPhysicsRecalc(PHYSICS_VERSION, batchSize);
      if (needsRecalc.length === 0) break;

      for (let i = 0; i < needsRecalc.length; i++) {
        if (i % 10 === 0) await yield_();
        const c = needsRecalc[i];
        try {
          const features = extractFeatures(c.formula);
          const gb = gbPredict(features);
          const nnScore = c.neuralNetScore ?? c.quantumCoherence ?? 0.3;
          const ensemble = Math.min(0.95, gb.score * 0.4 + nnScore * 0.6);

          const featureLambda = features.electronPhononLambda ?? 0;
          const metalScore = features.metallicity ?? 0.5;
          const recalcPressure = enforcePhysicsPressure(c.formula, c.pressureGpa);

          let newTc: number | null = null;
          if (featureLambda > 0) {
            const adTc = computePhysicsOnlyTc(featureLambda, features.logPhononFreq, features.muStarEstimate);
            newTc = adTc > 0 ? Math.round(adTc) : 0;
          }
          if (newTc != null) {
            newTc = Math.round(applyAmbientTcCap(newTc, featureLambda, recalcPressure, metalScore, c.formula));
          }

          const updatedFeatures = { ...features, physicsVersion: PHYSICS_VERSION };

          const isRoomTemp = (newTc ?? 0) >= 293 &&
            c.zeroResistance === true &&
            c.meissnerEffect === true &&
            recalcPressure <= 50;

          const recalcViability = classifyPressureViability(recalcPressure);

          await storage.updateSuperconductorCandidate(c.id, {
            predictedTc: newTc,
            pressureGpa: recalcPressure,
            mlFeatures: { ...updatedFeatures, pressureViability: recalcViability.label } as any,
            xgboostScore: gb.score,
            neuralNetScore: nnScore,
            ensembleScore: Math.max(0, ensemble * (1 - recalcViability.penalty)),
            electronPhononCoupling: features.electronPhononLambda ?? null,
            roomTempViable: isRoomTemp,
            ambientPressureStable: recalcPressure <= 50 ? c.ambientPressureStable : false,
          });
          totalRecalculated++;
        } catch (e) { console.error("[Engine] Physics recalculation update failed:", e); }
      }

      if (totalRecalculated > 10000) break;
    }

    if (totalRecalculated > 0) {
      emit("log", {
        phase: "engine",
        event: "Physics recalculation complete",
        detail: `Recalculated ${totalRecalculated} candidates with corrected metallicity, lambda, and ambient-pressure Tc caps (v${PHYSICS_VERSION})`,
        dataSource: "Internal",
      });
    }
  } catch (e) { console.error("[Engine] Physics recalculation failed:", e); }
}

export async function startEngine() {
  if (state === "running") return getStatus();
  state = "running";
  broadcast("engineState", { state: "running" });

  try {
    const maxCycle = await storage.getMaxConvergenceCycle();
    if (maxCycle > cycleCount) {
      cycleCount = maxCycle;
    }
  } catch (e) { console.error("[Engine] Max convergence cycle restore failed:", e); }

  try {
    const xtbHealth = await checkXTBHealth();
    emit("log", {
      phase: "engine",
      event: "xTB health check",
      detail: `xTB ${xtbHealth.available ? "v" + xtbHealth.version : "NOT FOUND"}. Opt: ${xtbHealth.canOptimize ? "OK" : "FAIL"}. Hess: ${xtbHealth.canHess ? "OK" : "FAIL"}${xtbHealth.error ? ". Error: " + xtbHealth.error : ""}`,
      dataSource: "xTB Health Check",
    });
  } catch (e: any) {
    emit("log", { phase: "engine", event: "xTB health check failed", detail: e.message, dataSource: "xTB Health Check" });
  }

  await backfillGBScores();
  await recalculatePhysics();

  try {
    const dbCampaigns = await storage.getInverseDesignCampaigns();
    if (dbCampaigns.length > 0) {
      const restored = await restoreCampaignsFromDB(dbCampaigns as any);
      if (restored > 0) {
        emit("log", { phase: "inverse-optimizer", event: "Campaigns restored", detail: `${restored} inverse design campaigns restored from database` });
      }
    }
  } catch (e) { console.error("[Engine] Campaign restoration failed:", e); }

  try {
    const scCount = await storage.getSuperconductorCount();
    autonomousTotalScreened = Math.max(autonomousTotalScreened, scCount * 3);
    autonomousTotalPassed = Math.max(autonomousTotalPassed, scCount);
    const topByTc = await storage.getSuperconductorCandidatesByTc(1);
    if (topByTc.length > 0 && (topByTc[0].predictedTc ?? 0) > autonomousBestTc) {
      autonomousBestTc = topByTc[0].predictedTc ?? 0;
    }
    if (scCount > 0) {
      emit("log", { phase: "engine", event: "Stats restored from DB", detail: `Restored baseline: ~${autonomousTotalScreened} screened, ~${autonomousTotalPassed} passed, best Tc=${autonomousBestTc}K`, dataSource: "Internal" });
    }
  } catch (e) { console.error("[Engine] Stats restore from DB failed:", e); }

  try {
    const allCandidates = await storage.getSuperconductorCandidates(5000);
    let preSeeded = 0;
    let featureBackfilled = 0;
    for (const c of allCandidates) {
      const norm = normalizeFormula(c.formula);
      if (!alreadyScreenedFormulas.has(norm)) {
        alreadyScreenedFormulas.add(norm);
        preSeeded++;
      }
      if (featureBackfilled >= 100) continue;
      try {
        buildAndStoreFeatureRecord(c.formula, c.predictedTc ?? null, null, (c.ensembleScore ?? 0.3));
        featureBackfilled++;
      } catch (e) { console.error(`[Engine] Feature backfill failed for ${c.formula}:`, e); }
    }
    if (featureBackfilled > 0 || preSeeded > 0) {
      emit("log", { phase: "engine", event: "Feature DB backfilled", detail: `Loaded ${featureBackfilled} candidate feature records for hypothesis engine (dataset size: ${getDatasetSize()}). Pre-seeded ${preSeeded} formulas into screened cache (total: ${alreadyScreenedFormulas.size}).`, dataSource: "Internal" });
    }
  } catch (e) { console.error("[Engine] Feature DB backfill outer error:", e); }

  try {
    const topForBO = await storage.getSuperconductorCandidatesByTc(50);
    let boSeeded = 0;
    for (const c of topForBO) {
      try {
        const tc = c.predictedTc ?? 0;
        const lambda = c.electronPhononCoupling ?? 0.5;
        const score = c.ensembleScore ?? 0.3;
        bayesianOptimizer.addObservation(c.formula, tc, lambda, score);
        boSeeded++;
      } catch (e) { /* skip */ }
    }
    if (boSeeded > 0) {
      emit("log", { phase: "engine", event: "Bayesian optimizer seeded from DB", detail: `Loaded ${boSeeded} observations into BO. Stats: ${bayesianOptimizer.getStats().observationCount} obs, best Tc=${bayesianOptimizer.getStats().bestTc}K`, dataSource: "Internal" });
    }
  } catch (e) { console.error("[Engine] BO seeding from DB failed:", e); }

  try {
    const topoCandidates = await storage.getSuperconductorCandidatesByTc(100);
    let topoSeeded = 0;
    let fermiSeeded = 0;
    const seenFormulas = new Set<string>();
    for (const c of topoCandidates) {
      if (topoSeeded >= 80 && fermiSeeded >= 80) break;
      if (seenFormulas.has(c.formula)) continue;
      seenFormulas.add(c.formula);
      if (topoSeeded < 80) {
        try {
          const electronic = computeElectronicStructure(c.formula);
          const topoResult = analyzeTopology(
            c.formula,
            electronic,
            c.crystalStructure?.split(" ")[0],
            c.crystalStructure?.match(/\((\w+)\)/)?.[1]
          );
          trackTopologyResult(topoResult);
          topoSeeded++;
        } catch (e) { console.error(`[Engine] Topology seed failed for ${c.formula}:`, e); }
      }
      if (fermiSeeded < 80) {
        try {
          const fsResult = computeFermiSurface(c.formula);
          assignToCluster(c.formula, fsResult, c.predictedTc ?? 0);
          fermiSeeded++;
        } catch (e) { console.error(`[Engine] Fermi cluster seed failed for ${c.formula}:`, e); }
      }
    }
    if (topoSeeded > 0 || fermiSeeded > 0) {
      const topoStats = getTopologyStats();
      emit("log", { phase: "engine", event: "Topology & Fermi clusters seeded", detail: `Analyzed ${topoSeeded} candidates: ${topoStats.totalTopological} topological, ${topoStats.totalTSC} TSC. Fermi clusters: ${fermiSeeded} assigned.`, dataSource: "Internal" });
    }
  } catch (e) { console.error("[Engine] Topology/Fermi seed outer error:", e); }

  try {
    const activeCampaigns = getAllActiveCampaigns();
    if (activeCampaigns.length === 0) {
      const scCount = await storage.getSuperconductorCount();
      if (scCount >= 20) {
        const topCandidates = await storage.getSuperconductorCandidatesByTc(5);
        const currentBestTc = topCandidates[0]?.predictedTc ?? 100;

        const campaign200 = createCampaign(`auto-200K-${Date.now()}`, {
          targetTc: 200,
          maxPressure: 50,
          minLambda: 0.5,
          maxHullDistance: 0.3,
          metallicRequired: true,
          phononStable: true,
          preferredElements: ["Nb", "Ti", "B", "C", "N"],
        }, 200);
        await storage.insertInverseDesignCampaign({
          id: campaign200.id,
          targetTc: 200,
          targetPressure: 0,
          status: "active",
          cyclesRun: 0,
          bestTcAchieved: 0,
          bestDistance: 1,
          candidatesGenerated: 0,
          candidatesPassedPipeline: 0,
          learningState: {} as any,
          convergenceHistory: [],
          topCandidates: [],
        });

        const targetHighTc = Math.max(300, Math.round(currentBestTc * 1.5));
        const campaign300 = createCampaign(`auto-${targetHighTc}K-${Date.now()}`, {
          targetTc: targetHighTc,
          maxPressure: 50,
          minLambda: 1.0,
          maxHullDistance: 0.5,
          metallicRequired: true,
          phononStable: true,
          preferredElements: ["La", "Y", "H", "Ca", "B"],
        }, 200);
        await storage.insertInverseDesignCampaign({
          id: campaign300.id,
          targetTc: targetHighTc,
          targetPressure: 0,
          status: "active",
          cyclesRun: 0,
          bestTcAchieved: 0,
          bestDistance: 1,
          candidatesGenerated: 0,
          candidatesPassedPipeline: 0,
          learningState: {} as any,
          convergenceHistory: [],
          topCandidates: [],
        });

        emit("log", {
          phase: "inverse-optimizer",
          event: "Auto-created inverse design campaigns",
          detail: `Created 2 campaigns targeting ${200}K and ${targetHighTc}K with ${scCount} existing candidates as seed data`,
          dataSource: "Inverse Optimizer",
        });
      }
    }
  } catch (e) { console.error("[Engine] Inverse campaign auto-create failed:", e); }

  try {
    const existingPipelines = getAllPipelines();
    if (existingPipelines.length === 0) {
      const pipelineId = `integrated-pipeline-${Date.now()}`;
      createPipeline(pipelineId, {
        targetTc: 293,
        maxPressure: 50,
        minLambda: 0.8,
        metallicRequired: true,
        phononStable: true,
        preferredElements: ["La", "Y", "H", "Ca", "B", "Nb"],
      });
      integratedPipelineId = pipelineId;
      emit("log", { phase: "engine", event: "Inverse design pipeline auto-started", detail: `Created integrated pipeline ${pipelineId} targeting 293K`, dataSource: "Integrated Subsystems" });
    } else {
      integratedPipelineId = existingPipelines[0].id;
    }
  } catch (e: any) {
    emit("log", { phase: "engine", event: "Pipeline auto-start skipped", detail: e.message?.slice(0, 150), dataSource: "Integrated Subsystems" });
  }

  try {
    const labId = `integrated-lab-${Date.now()}`;
    createLab(labId, 293, 50, 1000);
    integratedLabId = labId;
    emit("log", { phase: "engine", event: "Design lab auto-started", detail: `Created integrated design lab ${labId} with 8 strategies targeting 293K`, dataSource: "Integrated Subsystems" });
  } catch (e: any) {
    emit("log", { phase: "engine", event: "Design lab auto-start skipped", detail: e.message?.slice(0, 150), dataSource: "Integrated Subsystems" });
  }

  try {
    const strategyTypes = ["hydride-cage-optimizer", "layered-intercalation", "light-element-phonon"] as const;
    let reprGenerated = 0;
    for (const st of strategyTypes) {
      try {
        const prog = generateDesignProgram(st, ["La", "Y", "H", "Ca", "B", "Nb", "Ti"]);
        const result = executeDesignProgram(prog);
        if (result.formula && isValidFormula(result.formula)) {
          registerProgram(prog, result.predictedTc ?? 0);
          alreadyScreenedFormulas.add(normalizeFormula(result.formula));
          reprGenerated++;
        }
      } catch (e) { console.error("[Engine] Design representation seed failed:", e); }
    }
    if (reprGenerated > 0) {
      emit("log", { phase: "engine", event: "Design representations seeded", detail: `Generated ${reprGenerated} initial design programs from 3 strategy types`, dataSource: "Integrated Subsystems" });
    }
  } catch (e) { console.error("[Engine] Design representation seed outer error:", e); }

  try {
    const synthDataset = generateSyntheticDataset(40);
    const theories = runSymbolicPhysicsDiscovery(synthDataset);
    lastTheoryDiscoveryCycle = cycleCount;
    if (theories.length > 0) {
      const allTheories = getTheoryDatabase();
      const feedback = generateDiscoveryFeedback(allTheories);
      theoryFeedbackBias = { biasedVariables: feedback.biasedVariables ?? [], biasedElements: feedback.biasedElements ?? [] };
      emit("log", { phase: "engine", event: "Theory discovery auto-started", detail: `Initial run: ${theories.length} theories discovered, top score=${theories[0]?.theoryScore?.toFixed(3)}`, dataSource: "Integrated Subsystems" });
    }
  } catch (e: any) {
    emit("log", { phase: "engine", event: "Theory discovery auto-start skipped", detail: e.message?.slice(0, 150), dataSource: "Integrated Subsystems" });
  }

  try {
    const causalDataset = generateCausalDataset(60);
    const causalResult = runCausalDiscovery(causalDataset);
    lastCausalDiscoveryCycle = cycleCount;
    if (causalResult.designGuidance.length > 0) {
      causalDesignGuidance = causalResult.designGuidance.map(g => ({
        variable: g.variable,
        direction: g.direction,
        causalImpactOnTc: g.causalImpactOnTc,
      }));
    }
    emit("log", { phase: "engine", event: "Causal discovery auto-started", detail: `Initial run: ${causalResult.graph.edges.length} causal edges, ${causalResult.hypotheses.length} mechanism hypotheses, ${causalResult.rules.length} causal rules, ${causalResult.designGuidance.length} design guidance vars`, dataSource: "Integrated Subsystems" });
  } catch (e: any) {
    emit("log", { phase: "engine", event: "Causal discovery auto-start skipped", detail: e.message?.slice(0, 150), dataSource: "Integrated Subsystems" });
  }

  emit("log", {
    phase: "engine",
    event: "Learning engine started",
    detail: `Balanced learning: resuming from cycle ${cycleCount}. All subsystems initialized: pipeline, design lab, representations, theory discovery, causal discovery.`,
    dataSource: "Internal",
  });

  if (!isRunningCycle) {
    setTimeout(runLearningCycle, 2000);
  }
  return getStatus();
}

export function stopEngine() {
  state = "stopped";
  if (cycleTimer) {
    clearTimeout(cycleTimer);
    cycleTimer = null;
  }
  broadcast("engineState", { state: "stopped" });

  emit("log", {
    phase: "engine",
    event: "Learning engine stopped",
    detail: `Completed ${cycleCount} cycles, ${totalScCandidates} SC candidates, ${totalSynthesisDiscovered} synthesis paths`,
    dataSource: "Internal",
  });

  return getStatus();
}

export function pauseEngine() {
  if (state !== "running") return getStatus();
  state = "paused";
  if (cycleTimer) {
    clearTimeout(cycleTimer);
    cycleTimer = null;
  }
  broadcast("engineState", { state: "paused" });
  return getStatus();
}

export function resumeEngine() {
  if (state !== "paused") return getStatus();
  state = "running";
  broadcast("engineState", { state: "running" });
  if (!isRunningCycle) {
    setTimeout(runLearningCycle, 2000);
  }
  return getStatus();
}

export function getStatus(): EngineStatus {
  return {
    state,
    activeTasks: Array.from(activeTasks),
    cycleCount,
    lastCycleAt,
    totalMaterialsFetched,
    totalInsightsGenerated,
    totalPredictionsMade,
    totalSynthesisDiscovered,
    totalReactionsLearned,
    totalScCandidates,
    totalPhysicsComputed,
    totalStructuresPredicted,
    totalPipelineScreened,
    totalNovelSynthesisProposed,
    totalInverseDesigned: getInverseDesignCount(),
    totalStructuralVariants: getStructuralVariantCount(),
    tempo: engineTempo,
    statusMessage: currentStatusMessage,
  };
}
