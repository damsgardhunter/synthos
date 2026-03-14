import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertMaterialSchema, insertResearchLogSchema, insertExperimentalValidationSchema } from "@shared/schema";
import { initWebSocket, startEngine, stopEngine, pauseEngine, resumeEngine, getStatus, getAutonomousLoopStats } from "./learning/engine";
import { getSignalDefinitions } from "./learning/material-signal-scanner";
import { enumeratePrototypesForFormula } from "./learning/crystal-prototypes";
import { isDFTAvailable, getDFTMethodInfo, getXTBStats, runLandscapeExploration, getLandscapeStats as getEnergyLandscapeStats } from "./dft/qe-dft-engine";
import { generateDopedVariants, generateDopedVariantsWithRelaxation, getDopingEngineStats, getDopingRecommendations, runDopingBatch, detectSCSignals, runDopingSearchLoop, analyzeHessianPhonons, detectAnharmonicVibrations, runMDSampling, computeDebyeTemp, computeDynamicLatticeScore } from "./learning/doping-engine";
import { getDFTQueueStats, startDFTWorkerLoop, submitDFTJob } from "./dft/dft-job-queue";
import {
  createPipeline as createNextGenPipeline,
  runPipelineIteration as runNextGenIteration,
  getPipelineStats as getNextGenPipelineDetail,
  pausePipeline as pauseNextGenPipeline,
  resumePipeline as resumeNextGenPipeline,
  getNextGenPipelineStats as getNextGenStats,
} from "./inverse/next-gen-pipeline";
import {
  createLab as createSelfImprovingLab,
  runLabIteration as runSelfImprovingIteration,
  getLabStats as getSelfImprovingLabDetail,
  pauseLab as pauseSelfImprovingLab,
  resumeLab as resumeSelfImprovingLab,
  getAllLabStats as getSelfImprovingLabOverview,
  syncGlobalKnowledgeBase,
  getGlobalKnowledgeBase,
} from "./inverse/self-improving-lab";
import {
  generateDesignProgram, executeDesignProgram, mutateDesignProgram, crossoverPrograms,
  generateDesignGraph, mutateDesignGraph, analyzeGraph,
  programToGraph, graphToProgram,
  getDesignRepresentationStats,
  type DesignProgram, type DesignGraph,
} from "./inverse/design-representations";
import { getCalibrationData, getConfidenceBand, getEvaluatedDatasetStats, gbPredictWithUncertainty, getXGBEnsembleStats, getModelVersionHistory, getFailureExampleCount, startPoolInit } from "./learning/gradient-boost";
import { gnnPredictWithUncertainty, getGNNVersionHistory, getGNNModelVersion, getDFTTrainingDatasetStats, buildCrystalGraph, getHeldOutValidationSet, getGNNPrediction } from "./learning/graph-neural-net";
import { predictPressureCurve, findOptimalPressure, pressureSensitivity, getPressureCurveStats, getPressureExplorationStats } from "./learning/pressure-aware-surrogate";
import { detectPhaseTransitions, getPhaseTransitionStats } from "./learning/pressure-phase-detector";
import { computeEnthalpyPressureCurve, findStabilityPressureWindow, getEnthalpyStats } from "./learning/enthalpy-stability";
import { buildPressureResponseProfile, interpolateAtPressure, getPressurePropertyMapStats } from "./learning/pressure-property-map";
import { optimizePressureForFormula, getBayesianPressureStats } from "./learning/bayesian-pressure-optimizer";
import { fastPressureScreen, batchPressureScreen, findBestScreeningPressure, getPressureClusterStats, recordClusterDiscovery, samplePressureFromClusters, assignPressureCluster } from "./learning/pressure-screening";
let _eliashbergModule: typeof import("./physics/eliashberg-pipeline") | null = null;
async function getEliashbergModule() {
  if (!_eliashbergModule) {
    _eliashbergModule = await import("./physics/eliashberg-pipeline");
  }
  return _eliashbergModule;
}
import { getActiveLearningStats, getActiveLearningCycleHistory, getPressureCoverageStats } from "./learning/active-learning";
import { getPhysicsStoreStats, recalculateAllDerivedFeatures, getDerivedFeatures, recordPhysicsResult } from "./learning/physics-results-store";
import { buildUnifiedDataset, getUnifiedDatasetStats, getUnifiedDatasetForLLM, getTrainingSlice, getSnapshotHistory } from "./learning/unified-training-dataset";
import { generateCycleDiagnostics, getReportHistory, getLatestReport, formatDiagnosticReportText, getDiagnosticsForLLM } from "./learning/cycle-diagnostics";
import { predictLambda, getLambdaRegressorStats, initLambdaRegressor } from "./learning/lambda-regressor";
import { predictPhononProperties, getPhononSurrogateStats, initPhononSurrogate } from "./physics/phonon-surrogate";
import { getCalibrationStats as getSurrogateFitnessStats } from "./learning/surrogate-fitness";
import { getPillarDFTFeedbackStats } from "./inverse/sc-pillars-optimizer";
import { extractFeatures, computeUnifiedCI } from "./learning/ml-predictor";
import { computeCompositionFeatures, COMPOSITION_FEATURE_NAMES } from "./learning/composition-features";
import { cache, TTL, CACHE_KEYS } from "./cache";
import rateLimit from "express-rate-limit";
import { fetchAllData as fetchMPAllData, isApiAvailable as isMPAvailable } from "./learning/materials-project-client";
import { fetchAflowData, crossValidateWithMP, crossValidateWithAflow } from "./learning/aflow-client";
import { sanitizeForbiddenWords } from "./learning/utils";
import { runDiffusionGenerationCycle, getDiffusionStats as getDiffusionGeneratorStats } from "./ai/crystal-generator";
import { analyzeTopology, getTopologyStats } from "./physics/topology-engine";
import { computeTopologicalInvariants, getInvariantStats, trackInvariantResult } from "./physics/topological-invariants";
import { computeElectronicStructure, computePhysicsTcUQ } from "./learning/physics-engine";
import {
  createCampaign, getCampaign, getAllActiveCampaigns, pauseCampaign,
  removeCampaign, getCampaignStats, getInverseDesignStats,
  loadCampaign, getSerializableCampaignState,
} from "./inverse/inverse-optimizer";
import type { TargetProperties } from "./inverse/target-schema";
import {
  runDifferentiableOptimization, runGradientDescentCycle,
  getDifferentiableOptimizerStats, generateGradientSeeds,
} from "./inverse/differentiable-optimizer";
import {
  runStructureDiffusionCycle, getStructureDiffusionStats,
  getMotifLibrarySummary, runStructureFirstDesign,
} from "./ai/structure-diffusion";
import {
  checkPhysicsConstraints, constraintGuidedGenerate,
  getConstraintEngineStats,
} from "./inverse/physics-constraint-engine";
import {
  evaluatePillars, runPillarGuidedGeneration,
  getPillarOptimizerStats, DEFAULT_PILLAR_TARGETS,
} from "./inverse/sc-pillars-optimizer";
import {
  computePairingProfile, computePairingFeatureVector,
} from "./physics/pairing-mechanisms";
import {
  encodeGenome, findSimilar, genomeDiversity,
  genomeGuidedInverseDesign, getGenomeCacheStats,
} from "./physics/materials-genome";
import {
  analyzeHydrogenNetwork, getHydrogenNetworkStats,
} from "./physics/hydrogen-network-engine";
import { analyzeReactionNetwork } from "./physics/reaction-network-engine";
import { computeFermiSurface } from "./physics/fermi-surface-engine";
import { getAllClusters, getCluster, getClusterGuidance, getClusterStats } from "./physics/fermi-surface-clustering";
import { predictBandStructure } from "./physics/band-structure-surrogate";
import { predictBandDispersion, getBandOperatorStats } from "./physics/band-structure-operator";
import { getBandCalcStats } from "./dft/band-structure-calculator";
import { getDFTBandAnalysisStats } from "./dft/dft-band-analysis";
import { predictStability } from "./physics/stability-predictor";
import { analyzeInterface, generateHeterostructureCandidates } from "./physics/interface-engine";
import {
  generateHeterostructure, generateBilayerCandidates, findBestSubstrates,
  getHeterostructureStats, getSubstrateDatabase,
} from "./crystal/heterostructure-generator";
import {
  relaxInterface, getInterfaceRelaxationStats,
  scoreInterfaceCandidatesForActiveLearning, runInterfaceDiscoveryForActiveLearning,
} from "./crystal/interface-relaxation";
import { detectQuantumCriticality } from "./physics/quantum-criticality";
import { discoveryMemory } from "./learning/discovery-memory";
import { computeFeatureVector, buildAndStoreFeatureRecord, getFeatureDataset, getDatasetSize, getFeatureRecord } from "./theory/physics-feature-db";
import { runSymbolicRegression, getDiscoveredTheories, theoryKnowledgeBase, getValidationStats } from "./theory/symbolic-regression";
import {
  runSymbolicPhysicsDiscovery, buildPhysicsDiscoveryRecord,
  generateSyntheticDataset, generateDiscoveryFeedback,
  getSymbolicDiscoveryStats, getTheoryDatabase, getFeatureLibrary,
  validatePhysicsConstraints, PHYSICS_VARIABLES,
  type PhysicsDiscoveryRecord, type SymbolicDiscoveryConfig,
} from "./theory/symbolic-physics-discovery";
import {
  runCausalDiscovery, generateCausalDataset, getCausalDiscoveryStats,
  buildCausalDataRecord, simulateIntervention, runCounterfactual,
  getOntology, getCausalVariables, getDiscoveredHypotheses,
  getCausalRules, getLatestGraph, getDesignGuidance,
} from "./theory/causal-physics-discovery";
import { computeMultiScaleFeatures, computeCrossScaleCoupling, runSensitivityAnalysis } from "./theory/multi-scale-engine";
import { getPhysicsParameters, getParameterHistory, getModelPerformance } from "./theory/self-improving-physics";
import { getPerformanceMetrics, recordPrediction } from "./theory/model-performance-tracker";
import { getGeneratorAllocations, getGeneratorCompetitionStats } from "./learning/generator-manager";
import { getEmbeddingDataset, getLandscapeStats } from "./landscape/discovery-landscape";
import { getZoneMap } from "./landscape/zone-detector";
import { getFullLandscapeGuidance } from "./landscape/landscape-guidance";
import { solveConstraints, evaluateFormulaAgainstConstraints } from "./inverse/constraint-solver";
import { searchPressurePathways, getPathwayStats, getAmbientCandidatesFromPathways } from "./inverse/pressure-pathway";
import { solveConstraintGraph, getFeasibleRegions } from "./inverse/constraint-graph-solver";
import { computeSynthesisPathway, getSynthesisPathwayStats } from "./synthesis/reaction-pathway";
import { buildReactionNetwork, getReactionNetworkStats } from "./synthesis/reaction-network";
import { getParameterSpace } from "./synthesis/synthesis-variables";
import { getSynthesisOptimizerStats } from "./synthesis/synthesis-condition-optimizer";
import {
  getSimulatorStats, simulateSynthesisEffects, defaultSynthesisVector,
  optimizeSynthesisForFixedMaterial, optimizeSynthesisPath, computeSynthesisVerdict,
} from "./physics/synthesis-simulator";
import { getSynthesisLearningStats, querySimilarSynthesis } from "./synthesis/synthesis-learning-db";
import { generateDefectVariants, adjustElectronicStructure, getDefectEngineStats } from "./physics/defect-engine";
import {
  generateDisorderedStructure, generateAllDisorderVariants, suggestDisorders,
  getDisorderGeneratorStats, getSearchLimits, type DisorderSpec, type DisorderType,
} from "./crystal/disorder-generator";
import {
  computeDisorderMetrics, getDisorderMetricsStats, extractMLFeatures,
} from "./crystal/disorder-metrics";
import { crossEngineHub } from "./learning/cross-engine-hub";
import { discoverNovelSynthesisPaths, getSynthesisDiscoveryStats, getGAEvolutionStats, getStructuralMotifStats } from "./learning/synthesis-discovery";
import { planSynthesisRoutes, getSynthesisPlannerStats } from "./synthesis/synthesis-planner";
import { generateHeuristicRoutes, getHeuristicGeneratorStats } from "./synthesis/heuristic-synthesis-generator";
import { predictSynthesisFeasibility, getSynthesisPredictorStats } from "./synthesis/ml-synthesis-predictor";
import { evaluateSynthesisGate, getSynthesisGateStats } from "./synthesis/synthesis-gate";
import { generateRetrosynthesisRoutes, getRetrosynthesisStats } from "./synthesis/retrosynthesis-engine";
import { predictDOS, dosPrefilter, getDOSSurrogateStats, detectVanHoveSingularities, physicsHeuristicDOS, type DOSSurrogateResult } from "./physics/dos-surrogate";
import { computeSynthesisScore } from "./learning/multi-fidelity-pipeline";
import { estimateCorrelationEffects, getCorrelationEngineStats } from "./physics/correlation-engine";
import {
  runQuantumEnginePipeline, getQuantumEngineStats,
  getQuantumEngineDataset, getRecentQuantumEngineResults,
} from "./dft/quantum-engine-pipeline";
import { simulateCrystalGrowth, getCrystalGrowthStats } from "./synthesis/crystal-growth-simulator";
import { generateExperimentPlan, getExperimentPlannerStats, type ExperimentCandidate } from "./experiment-planner";
import {
  analyzeFrontier, computeNoveltyScore, analyzeZoneIntelligence,
  generateExplorationStrategy, getLandscapeIntelligenceStats,
} from "./landscape/landscape-intelligence";
import {
  getActiveHypotheses, getAllHypotheses, testHypothesisById,
  getHypothesisStats, runHypothesisCycle,
} from "./theory/hypothesis-engine";
import {
  runRelaxation, computePhonons, computeEph, computeTc,
  getPhysicsApiStats,
} from "./physics/physics-api";
import {
  getDatasetStats as getCrystalDatasetStats,
  getEntryByFormula as getCrystalDatasetEntry,
  getEntriesByPrototype as getCrystalDatasetByPrototype,
  getEntriesBySystem as getCrystalDatasetBySystem,
  getTrainingData as getCrystalTrainingData,
  fetchMPStructures,
  fetchOQMDStructures,
} from "./crystal/crystal-structure-dataset";
import {
  predictStructure as mlPredictStructure,
  getStructurePredictorStats as getStructureMLStats,
  initStructurePredictorML,
} from "./crystal/structure-predictor-ml";
import {
  initCrystalVAE, getCrystalVAEStats, generateNovelCrystal,
  interpolateCrystals, encodeFormula,
} from "./crystal/crystal-vae";
import {
  initDiffusionModel,
  sampleStructures as diffusionSampleStructures,
  getDiffusionModelStats,
} from "./crystal/crystal-diffusion-model";
import {
  generateCandidates as generativeCrystalGenerate,
  getGenerativeEngineStats,
  type GenerationStrategy,
} from "./crystal/generative-crystal-engine";
import {
  generateHybridCandidates,
  getHybridGeneratorStats,
} from "./crystal/hybrid-structure-generator";
import {
  getFailureDBStats, getFailurePatterns, getFailureFeatureVector,
  shouldAvoidStructure, recordStructureFailure,
  type FailureReason, type FailureSource,
} from "./crystal/structure-failure-db";
import {
  runStructureLearningCycle,
  getStructureLearningStats,
} from "./crystal/structure-learning-loop";
import {
  predictStabilityScreen,
  getStabilityPredictorStats,
  trainStabilityPredictor,
} from "./crystal/stability-predictor";
import {
  scoreFormulaNovelty,
  getNoveltyStats,
  getNoveltyRanking,
  initFingerprintDB,
} from "./crystal/structure-novelty-detector";
import {
  getRelaxationStats, getRelaxationPatterns,
  getRelaxationEntry, predictRelaxationMagnitude,
} from "./crystal/relaxation-tracker";
import {
  predictVolume, getVolumeDNNStats, initVolumeDNN,
} from "./crystal/volume-predictor-dnn";
import {
  getDistortionStats, getDistortionForFormula,
  classifyFormulaDistortion, getClassifierStats,
} from "./crystal/distortion-detector";
import {
  computeTBProperties, computeBandStructure, computeDOS,
  computeFermiProperties, detectFlatBands, computeElectronPhononProxies,
  getTBEngineStats,
} from "./physics/tight-binding-engine";
import {
  computeStructureEmbedding, clusterStructures as clusterStructureEmbeddings,
  getEmbeddingStats, getClusters as getEmbeddingClusters,
  getClusterAssignment, computeClusterNovelty, estimateStructureUncertainty,
  initStructureEmbedding,
} from "./crystal/structure-embedding";
import {
  predictStructureAtPressure, getPressurePhaseMap,
  learnPressureTransition, getPressureStructureStats,
  initPressureStructureModel,
} from "./crystal/pressure-structure-model";
import {
  getRewardSystemStats, getBestMotifs, getStructureReward,
} from "./crystal/structure-reward-system";
import {
  predictTBProperties as predictTBSurrogate, getTBSurrogateStats, retrainTBSurrogate,
} from "./physics/tb-ml-surrogate";
import {
  getComprehensiveModelDiagnostics, getModelHealthSummary, getPerFamilyBias,
  getModelDiagnosticsForLLM, recordPredictionOutcome,
  getErrorAnalysis, getFeatureImportanceReport,
  getFailureSummary, getModelBenchmark, getFailedMaterialsForLLM, getBenchmarkForLLM,
} from "./learning/model-diagnostics";
import {
  getModelLLMStatus, proposeNewFeatures, selectArchitecture,
  enableBuiltinFeature, disableCustomFeature, runModelLLMCycle,
  getAvailableFeatureDefinitions, getCurrentArchitecture,
} from "./learning/model-llm-controller";
import {
  proposeModelExperiments, executeExperiment, getExperimentHistory,
  getActiveExperiments, getExperimentStats, getAllDataRequests,
} from "./learning/model-experiment-controller";
import {
  getModelImprovementStats, getModelImprovementTrends,
  runModelImprovementCycle,
} from "./learning/model-improvement-loop";
import {
  getUncertaintyStatus, getUncertaintyReport, getFullUncertaintyReport,
  getHighUncertaintyPredictions, getVarianceByFamily, proposeUncertaintyImprovements,
} from "./learning/uncertainty-tracker";
import {
  getCalibrationState, recalibrateFromLedger, getConformalInterval,
  getFamilyConformalQuantiles, getECE, getCalibrationSummaryForLLM,
  validateIntervalsCoverage,
} from "./learning/conformal-calibrator";
import {
  computeOODScore, getOODStats, updateOODModel,
} from "./learning/ood-detector";
import {
  evaluateRetrainNeed, getSchedulerStats, getSchedulerForLLM, recordRetrainOutcome,
} from "./learning/retrain-scheduler";
import {
  getGroundTruthDataset, getGroundTruthSummary, getGroundTruthForLLM,
  getRecentBatchCycles, getBatchCycles, getDatapointsByCycle,
  getGroundTruthDatasetSlice, getDatapointsByFormula, getDatasetForTraining,
} from "./learning/ground-truth-store";
import {
  computeMetrics as computeLedgerMetrics, computeRecentMetrics as computeRecentLedgerMetrics,
  computeMetricsByFamily as computeLedgerByFamily, getWorstPredictions, getBestPredictions,
  getOverpredictions, getUnderpredictions, getLedgerSize, getLedgerSlice,
  getRetrainTriggerState, setRetrainThreshold, setRetrainEnabled, getMetricsForLLM as getLedgerLLMReport,
  getCycleImprovementHistory, getImprovementTrend,
} from "./learning/prediction-reality-ledger";

const generalLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 600,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests, please try again later." },
});

const engineLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many engine control requests, please try again later." },
});

const writeLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many write requests, please try again later." },
});

export async function registerRoutes(httpServer: Server, app: Express): Promise<Server> {
  initWebSocket(httpServer);

  // Start the DFT worker immediately on startup — no R² gating.
  startDFTWorkerLoop();

  // Warm the DB connection immediately so the first API requests on page load
  // don't hit a Neon cold-start timeout. Fire-and-forget; don't block registration.
  storage.getLearningPhases().catch(() => {});

  app.use("/api", generalLimiter);

  app.get("/api/elements", async (_req, res) => {
    try {
      const cached = cache.get(CACHE_KEYS.ELEMENTS);
      if (cached) return res.json(cached);
      const els = await storage.getElements();
      cache.set(CACHE_KEYS.ELEMENTS, els, TTL.ELEMENTS);
      res.json(els);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch elements" });
    }
  });

  app.get("/api/elements/:id", async (req, res) => {
    try {
      const el = await storage.getElementById(Number(req.params.id));
      if (!el) return res.status(404).json({ error: "Element not found" });
      res.json(el);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch element" });
    }
  });

  app.get("/api/elements/:symbol/related-counts", async (req, res) => {
    try {
      const symbol = req.params.symbol;
      const [materialCount, candidateCount] = await Promise.all([
        storage.getMaterialCountByElement(symbol),
        storage.getCandidateCountByElement(symbol),
      ]);
      res.json({ materialCount, candidateCount });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch related counts" });
    }
  });

  app.get("/api/materials", async (req, res) => {
    try {
      const rawLimit = Number(req.query.limit);
      const rawOffset = Number(req.query.offset);
      const limit = Math.min(Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 200, 1000);
      const offset = Number.isFinite(rawOffset) && rawOffset >= 0 ? rawOffset : 0;
      const mats = await storage.getMaterials(limit, offset);
      const total = await storage.getMaterialCount();
      res.json({ materials: mats, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch materials" });
    }
  });

  app.get("/api/materials/:id", async (req, res) => {
    try {
      const mat = await storage.getMaterialById(req.params.id);
      if (!mat) return res.status(404).json({ error: "Material not found" });
      res.json(mat);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch material" });
    }
  });

  app.get("/api/learning-phases", async (_req, res) => {
    try {
      const result = await cache.getOrSet(CACHE_KEYS.LEARNING_PHASES, TTL.LEARNING_PHASES, async () => {
        const phases = await storage.getLearningPhases();
        return phases.map(p => ({
          ...p,
          insights: (p.insights ?? []).map((s: string) => sanitizeForbiddenWords(s)),
        }));
      });
      res.json(result);
    } catch (e: any) {
      console.error("[learning-phases] ERROR:", e?.message, e?.stack?.slice(0, 500));
      res.status(500).json({ error: "Failed to fetch learning phases", detail: e?.message });
    }
  });

  app.get("/api/novel-predictions", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 50, 500);
      const offset = Number(req.query.offset) || 0;
      const preds = await storage.getNovelPredictions(limit, offset);
      res.json(preds);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch predictions" });
    }
  });

  app.get("/api/research-logs", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 100, 500);
      const cacheKey = `${CACHE_KEYS.RESEARCH_LOGS}:${limit}`;
      const result = await cache.getOrSet(cacheKey, TTL.RESEARCH_LOGS, async () => {
        const logs = await storage.getResearchLogs(limit);
        return logs.map(log => ({
          ...log,
          detail: sanitizeForbiddenWords(log.detail || ""),
          event: sanitizeForbiddenWords(log.event || ""),
        }));
      });
      res.json(result);
    } catch (e: any) {
      console.error("[research-logs] ERROR:", e?.message, e?.stack?.slice(0, 500));
      res.status(500).json({ error: "Failed to fetch research logs", detail: e?.message });
    }
  });

  app.get("/api/stats", async (_req, res) => {
    try {
      const cached = cache.get(CACHE_KEYS.STATS);
      if (cached) return res.json(cached);
      const stats = await storage.getStats();
      cache.set(CACHE_KEYS.STATS, stats, TTL.STATS);
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch stats" });
    }
  });

  app.post("/api/research-logs", writeLimiter, async (req, res) => {
    try {
      const parsed = insertResearchLogSchema.safeParse(req.body);
      if (!parsed.success) return res.status(400).json({ error: parsed.error });
      const log = await storage.insertResearchLog(parsed.data);
      cache.invalidate(CACHE_KEYS.STATS);
      cache.invalidatePrefix(CACHE_KEYS.RESEARCH_LOGS);
      res.json(log);
    } catch (e) {
      res.status(500).json({ error: "Failed to insert log" });
    }
  });

  app.post("/api/engine/start", engineLimiter, async (_req, res) => {
    try {
      cache.invalidate(CACHE_KEYS.STATS);
      cache.invalidatePrefix("crystal-structures:");
      cache.invalidatePrefix("computational-results:");
      const status = await startEngine();
      res.json(status);
    } catch (e) {
      res.status(500).json({ error: "Failed to start engine" });
    }
  });

  app.post("/api/engine/stop", engineLimiter, async (_req, res) => {
    try {
      const status = stopEngine();
      res.json(status);
    } catch (e) {
      res.status(500).json({ error: "Failed to stop engine" });
    }
  });

  app.post("/api/engine/pause", engineLimiter, async (_req, res) => {
    try {
      const status = pauseEngine();
      res.json(status);
    } catch (e) {
      res.status(500).json({ error: "Failed to pause engine" });
    }
  });

  app.post("/api/engine/resume", engineLimiter, async (_req, res) => {
    try {
      const status = resumeEngine();
      res.json(status);
    } catch (e) {
      res.status(500).json({ error: "Failed to resume engine" });
    }
  });

  app.get("/api/engine/status", async (_req, res) => {
    try {
      const status = getStatus();
      res.json(status);
    } catch (e) {
      res.status(500).json({ error: "Failed to get engine status" });
    }
  });

  app.get("/api/superconductor-candidates", async (req, res) => {
    try {
      const rawLim = Number(req.query.limit);
      const limit = Math.min(Number.isFinite(rawLim) && rawLim > 0 ? rawLim : 200, 1000);
      const cacheKey = `${CACHE_KEYS.CANDIDATES}:${limit}`;
      const result = await cache.getOrSet(cacheKey, TTL.CANDIDATES, async () => {
        const [candidates, total] = await Promise.all([
          storage.getSuperconductorCandidates(limit),
          storage.getSuperconductorCount(),
        ]);
        return { candidates, total };
      });
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch superconductor candidates" });
    }
  });

  app.get("/api/dft-status", async (_req, res) => {
    try {
      const result = await cache.getOrSetStale(CACHE_KEYS.DFT_STATUS, TTL.DFT_STATUS, async () => {
        const breakdown = await storage.getConfidenceBreakdown();
        const analyticalCount = breakdown.total - breakdown.high - breakdown.medium;

        const dftAvailable = isDFTAvailable();
        const methodInfo = dftAvailable ? getDFTMethodInfo() : null;
        const xtbStats = getXTBStats();

        return {
          total: breakdown.total,
          dftEnrichedCount: breakdown.high + breakdown.medium,
          breakdown: {
            high: breakdown.high,
            medium: breakdown.medium,
            analytical: analyticalCount,
          },
          recentEnriched: breakdown.recentEnriched.map(c => ({
            formula: c.formula,
            confidence: c.dataConfidence,
            ensembleScore: c.ensembleScore,
            predictedTc: c.predictedTc,
          })),
          xtb: {
            available: dftAvailable,
            method: methodInfo?.name ?? "unavailable",
            version: methodInfo?.version ?? "N/A",
            level: methodInfo?.level ?? "N/A",
            totalRuns: xtbStats.runs,
            successfulRuns: xtbStats.successes,
            successRate: xtbStats.runs > 0 ? (xtbStats.successes / xtbStats.runs * 100).toFixed(1) + "%" : "N/A",
            cacheSize: xtbStats.cacheSize,
            refElements: xtbStats.refElements,
          },
        };
      });
      res.json(result);
    } catch (e: any) {
      console.error("[dft-status] ERROR:", e?.message, e?.stack?.slice(0, 500));
      res.status(500).json({ error: "Failed to fetch DFT status", detail: e?.message });
    }
  });

  app.get("/api/dft-queue/stats", async (_req, res) => {
    try {
      const stats = await getDFTQueueStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch DFT queue stats" });
    }
  });

  app.get("/api/dft-band-structure/stats", async (_req, res) => {
    try {
      res.json(getBandCalcStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch band structure stats" });
    }
  });

  app.get("/api/dft-band-analysis/stats", async (_req, res) => {
    try {
      res.json(getDFTBandAnalysisStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch band analysis stats" });
    }
  });

  app.get("/api/synthesis-processes", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const processes = await storage.getSynthesisProcesses(limit);
      const total = await storage.getSynthesisCount();
      res.json({ processes, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch synthesis processes" });
    }
  });

  app.get("/api/chemical-reactions", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const reactions = await storage.getChemicalReactions(limit);
      const total = await storage.getReactionCount();
      res.json({ reactions, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch chemical reactions" });
    }
  });

  app.get("/api/crystal-structures", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const structures = await storage.getCrystalStructures(limit);
      const total = await storage.getCrystalStructureCount();
      res.json({ structures, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch crystal structures" });
    }
  });

  app.get("/api/computational-results", async (req, res) => {
    try {
      const rawCrLim = Number(req.query.limit);
      const limit = Math.min(Number.isFinite(rawCrLim) && rawCrLim > 0 ? rawCrLim : 200, 1000);
      const rawStage = req.query.stage != null ? Number(req.query.stage) : undefined;
      const stage = rawStage !== undefined && Number.isFinite(rawStage) ? rawStage : undefined;
      let results;
      if (stage !== undefined) {
        results = await storage.getComputationalResultsByStage(stage);
      } else {
        results = await storage.getComputationalResults(limit);
      }
      const total = await storage.getComputationalResultCount();
      res.json({ results, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch computational results" });
    }
  });

  app.get("/api/computational-results/failed", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const results = await storage.getFailedComputationalResults(limit);
      res.json({ results });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch failed results" });
    }
  });

  app.get("/api/pipeline-stats", async (req, res) => {
    try {
      const cached = cache.get<Awaited<ReturnType<typeof storage.getStats>>>(CACHE_KEYS.STATS);
      const stats = cached || await storage.getStats();
      if (!cached) cache.set(CACHE_KEYS.STATS, stats, TTL.STATS);
      res.json({
        pipelineStages: stats.pipelineStages,
        crystalStructures: stats.crystalStructures,
        computationalResults: stats.computationalResults,
      });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch pipeline stats" });
    }
  });

  app.get("/api/candidate-profile/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const csKey = CACHE_KEYS.crystalStructuresByFormula(formula);
      const crKey = CACHE_KEYS.computationalResultsByFormula(formula);

      let crystalStructureData = cache.get(csKey);
      let computationalResultData = cache.get(crKey);

      const [candidates, crystalStructures, computationalResults, synthesisProcesses, chemicalReactions] = await Promise.all([
        storage.getSuperconductorsByFormula(formula),
        crystalStructureData ? Promise.resolve(crystalStructureData) : storage.getCrystalStructuresByFormula(formula),
        computationalResultData ? Promise.resolve(computationalResultData) : storage.getComputationalResultsByFormula(formula),
        storage.getSynthesisProcessesByFormula(formula),
        storage.getChemicalReactionsByFormula(formula),
      ]);

      if (!crystalStructureData) cache.set(csKey, crystalStructures, TTL.CRYSTAL_STRUCTURES_BY_FORMULA);
      if (!computationalResultData) cache.set(crKey, computationalResults, TTL.COMPUTATIONAL_RESULTS_BY_FORMULA);

      res.json({ formula, candidates, crystalStructures, computationalResults, synthesisProcesses, chemicalReactions });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch candidate profile" });
    }
  });

  app.get("/api/ml-calibration", async (req, res) => {
    try {
      const calibration = await getCalibrationData();
      const tc = req.query.tc ? Number(req.query.tc) : undefined;
      const confidenceBand = tc !== undefined && Number.isFinite(tc) ? await getConfidenceBand(tc) : undefined;
      res.json({ ...calibration, ...(confidenceBand ? { confidenceBand } : {}) });
    } catch (e) {
      res.status(500).json({ error: "Failed to compute ML calibration" });
    }
  });

  app.get("/api/research-strategy", async (_req, res) => {
    try {
      const result = await cache.getOrSet(CACHE_KEYS.STRATEGY_LATEST, TTL.STRATEGY, async () => {
        const strategy = await storage.getLatestStrategy();
        return strategy || null;
      });
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch research strategy" });
    }
  });

  app.get("/api/research-strategy/history", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 20, 100);
      const cacheKey = `${CACHE_KEYS.STRATEGY_HISTORY}:${limit}`;
      const result = await cache.getOrSet(cacheKey, TTL.STRATEGY, async () => {
        return storage.getStrategyHistory(limit);
      });
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch strategy history" });
    }
  });

  app.get("/api/convergence", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 200, 500);
      const snapshots = await storage.getConvergenceSnapshots(limit);
      res.json(snapshots);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch convergence data" });
    }
  });

  app.get("/api/milestones", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 20, 100);
      const cacheKey = `${CACHE_KEYS.MILESTONES}:${limit}`;
      const result = await cache.getOrSet(cacheKey, TTL.MILESTONES, async () => {
        const [ms, total] = await Promise.all([
          storage.getMilestones(limit),
          storage.getMilestoneCount(),
        ]);
        return { milestones: ms, total };
      });
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch milestones" });
    }
  });

  app.get("/api/material-signals", async (req, res) => {
    try {
      const signals = getSignalDefinitions().map(s => ({
        id: s.id,
        name: s.name,
        description: s.description,
        keywords: s.keywords,
      }));
      const milestones = await storage.getMilestones(500);
      const signalMilestones = milestones.filter(m => m.type?.startsWith("signal-"));
      res.json({ signals, discoveries: signalMilestones });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch material signals" });
    }
  });

  app.get("/api/novel-insights", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 20, 200);
      const novelOnly = req.query.novelOnly === "true";
      const cacheKey = `${CACHE_KEYS.NOVEL_INSIGHTS}:${novelOnly ? "novel" : "all"}:${limit}`;
      const result = await cache.getOrSet(cacheKey, TTL.NOVEL_INSIGHTS, async () => {
        const [insights, total] = await Promise.all([
          novelOnly ? storage.getNovelInsightsOnly(limit) : storage.getNovelInsights(limit),
          storage.getNovelInsightCount(),
        ]);
        return { insights, total };
      });
      res.json(result);
    } catch (e: any) {
      console.error("[novel-insights] ERROR:", e?.message, e?.stack?.slice(0, 500));
      res.status(500).json({ error: "Failed to fetch novel insights", detail: e?.message });
    }
  });

  // Alias used by some dashboard queries
  app.get("/api/novel-insights/recent", async (_req, res) => {
    try {
      const result = await cache.getOrSet(
        `${CACHE_KEYS.NOVEL_INSIGHTS}:recent`,
        TTL.NOVEL_INSIGHTS,
        async () => {
          const insights = await storage.getNovelInsightsOnly(20);
          return { insights, total: insights.length };
        }
      );
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch recent novel insights", detail: e?.message });
    }
  });

  app.get("/api/cross-validation/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const candidates = await storage.getSuperconductorsByFormula(formula);
      const candidate = candidates[0] ?? { predictedTc: null, stabilityScore: null, electronPhononCoupling: null };

      const mpAvailable = isMPAvailable();
      let mpData = null;
      let mpValidation: any[] = [];

      if (mpAvailable) {
        try {
          mpData = await fetchMPAllData(formula);
          mpValidation = crossValidateWithMP(candidate, mpData.summary, mpData.elasticity);
        } catch {
          console.log(`[Cross-validation] MP fetch failed for ${formula}`);
        }
      }

      let aflowData = null;
      let aflowValidation: any[] = [];
      try {
        aflowData = await fetchAflowData(formula);
        aflowValidation = crossValidateWithAflow(candidate, aflowData);
      } catch {
        console.log(`[Cross-validation] AFLOW fetch failed for ${formula}`);
      }

      const allValidations = [...mpValidation, ...aflowValidation];
      const hasDiscrepancies = allValidations.some(v => v.agreement === "major-discrepancy");

      res.json({
        formula,
        materialsProject: {
          available: mpAvailable,
          hasData: !!mpData?.summary,
          summary: mpData?.summary ?? null,
          elasticity: mpData?.elasticity ?? null,
          phonon: mpData?.phonon ?? null,
          magnetism: mpData?.magnetism ?? null,
        },
        aflow: {
          available: true,
          hasData: (aflowData?.entries?.length ?? 0) > 0,
          entries: aflowData?.entries ?? [],
        },
        crossValidation: allValidations,
        hasDiscrepancies,
      });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch cross-validation data" });
    }
  });

  app.get("/api/engine/memory", async (_req, res) => {
    try {
      console.log(`[engine/memory] START handler at ${new Date().toLocaleTimeString()}`);
      const result = await cache.getOrSetStale(CACHE_KEYS.ENGINE_MEMORY, TTL.ENGINE_MEMORY, async () => {
        console.log(`[engine/memory] cache MISS, computing...`);
        const [strategies, insights, milestones, milestoneCount, snapshots, narratives] = await Promise.all([
          storage.getStrategyHistory(10),
          storage.getNovelInsightsOnly(50),
          storage.getMilestones(20),
          storage.getMilestoneCount(),
          storage.getConvergenceSnapshots(50),
          storage.getResearchLogsByEvent("cycle-narrative", 10),
        ]);
        const latestStrategy = strategies[0] ?? null;
        const topInsights = insights.slice(0, 5);
        const latestSnapshot = snapshots[snapshots.length - 1] ?? null;

        const familyStats = latestStrategy?.performanceSignals
          ? (latestStrategy.performanceSignals as any).familyStats ?? {}
          : {};

        const focusAreas = (latestStrategy?.focusAreas as any[]) ?? [];
        const currentHypothesis = focusAreas.length > 0
          ? { family: focusAreas[0].area, priority: focusAreas[0].priority, reasoning: focusAreas[0].reasoning }
          : null;

        const abandonedStrategies: string[] = [];
        if (strategies.length >= 2) {
          const latestFamilies = new Set(((strategies[0]?.focusAreas as any[]) ?? []).map((f: any) => f.area));
          for (let i = 1; i < strategies.length; i++) {
            const oldFamilies = ((strategies[i]?.focusAreas as any[]) ?? []).map((f: any) => f.area);
            for (const fam of oldFamilies) {
              if (!latestFamilies.has(fam) && !abandonedStrategies.includes(fam)) {
                abandonedStrategies.push(fam);
              }
            }
          }
        }

        console.log(`[engine/memory] calling getAutonomousLoopStats at ${new Date().toLocaleTimeString()}`);
        const loopStats = await getAutonomousLoopStats();
        console.log(`[engine/memory] getAutonomousLoopStats done at ${new Date().toLocaleTimeString()}`);

        return {
          currentHypothesis: currentHypothesis ? {
            ...currentHypothesis,
            reasoning: sanitizeForbiddenWords(currentHypothesis.reasoning || ""),
          } : null,
          familyStats,
          topInsights: topInsights.map(i => ({
            text: sanitizeForbiddenWords(i.insightText || ""),
            noveltyScore: i.noveltyScore,
            category: i.category,
            discoveredAt: i.discoveredAt,
          })),
          abandonedStrategies,
          milestoneCount,
          recentMilestones: milestones.slice(0, 5).map(m => ({
            ...m,
            description: sanitizeForbiddenWords(m.description || ""),
            title: sanitizeForbiddenWords(m.title || ""),
          })),
          totalCycles: latestSnapshot?.cycle ?? 0,
          bestTc: latestSnapshot?.bestTc ?? 0,
          bestScore: latestSnapshot?.bestScore ?? 0,
          familyDiversity: latestSnapshot?.familyDiversity ?? 0,
          pipelinePassRate: latestSnapshot?.pipelinePassRate ?? 0,
          cycleNarratives: narratives.map(n => ({ detail: sanitizeForbiddenWords(n.detail || ""), timestamp: n.timestamp })),
          autonomousLoopStats: loopStats,
          designRepresentations: getDesignRepresentationStats(),
        };
      });
      res.json(result);
    } catch (e: any) {
      console.error("[engine-memory] ERROR:", e?.message, e?.stack?.slice(0, 500));
      res.status(500).json({ error: "Failed to fetch engine memory", detail: e?.message });
    }
  });

  app.post("/api/validations", writeLimiter, async (req, res) => {
    try {
      const parsed = insertExperimentalValidationSchema.safeParse(req.body);
      if (!parsed.success) return res.status(400).json({ error: parsed.error });
      const validation = await storage.insertValidation(parsed.data);
      if (parsed.data.result === "positive" || parsed.data.result === "confirmed") {
        try {
          const candidate = await storage.getSuperconductorByFormula(parsed.data.formula);
          if (candidate && (candidate.verificationStage ?? 0) < 5) {
            await storage.updateSuperconductorCandidate(candidate.id, { verificationStage: 5, status: "experimentally-tested" });
          }
        } catch (promoteErr) {
          console.error(`[Routes] Auto-promote failed for ${parsed.data.formula}:`, promoteErr instanceof Error ? promoteErr.message.slice(0, 100) : "unknown");
        }
      }
      res.json(validation);
    } catch (e) {
      res.status(500).json({ error: "Failed to insert validation" });
    }
  });

  app.get("/api/validations/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const validations = await storage.getValidationsByFormula(formula);
      res.json(validations);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch validations" });
    }
  });

  app.get("/api/validation-stats", async (_req, res) => {
    try {
      const stats = await storage.getValidationStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch validation stats" });
    }
  });

  app.post("/api/generate-crystal", writeLimiter, async (req, res) => {
    try {
      const { count = 30, elements, targetTc } = req.body || {};
      const safeCount = Math.max(5, Math.min(50, Number(count) || 30));
      const targetEls = Array.isArray(elements) ? elements.filter((e: any) => typeof e === "string") : undefined;
      const result = runDiffusionGenerationCycle(safeCount, targetEls);
      res.json({
        generated: result.structures.length,
        formulas: result.formulas,
        structures: result.structures.map(s => ({
          formula: s.formula,
          spaceGroup: s.spaceGroup,
          crystalSystem: s.crystalSystem,
          prototypeMatch: s.prototypeMatch,
          noveltyScore: s.noveltyScore,
          densityGcm3: s.densityGcm3,
          minBondLength: s.minBondLength,
          atomCount: s.atoms.length,
          lattice: s.lattice,
        })),
        stats: result.stats,
      });
    } catch (e) {
      res.status(500).json({ error: "Failed to generate crystal structures" });
    }
  });

  app.get("/api/diffusion-stats", async (_req, res) => {
    try {
      res.json(getDiffusionGeneratorStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch diffusion stats" });
    }
  });

  app.get("/api/topology-stats", async (_req, res) => {
    try {
      res.json(getTopologyStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch topology stats" });
    }
  });

  app.get("/api/topology/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const crystalSystem = typeof req.query.crystalSystem === "string" ? req.query.crystalSystem : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const analysis = analyzeTopology(formula, electronic, spaceGroup, crystalSystem);
      res.json(analysis);
    } catch (e) {
      res.status(500).json({ error: "Failed to analyze topology" });
    }
  });

  app.get("/api/topological-invariants/stats", async (_req, res) => {
    try {
      res.json(getInvariantStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch invariant stats" });
    }
  });

  app.get("/api/topological-invariants/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const crystalSystem = typeof req.query.crystalSystem === "string" ? req.query.crystalSystem : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const result = computeTopologicalInvariants(formula, electronic, spaceGroup, crystalSystem);
      trackInvariantResult(formula, result);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to compute topological invariants" });
    }
  });

  app.get("/api/band-inversion/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const result = computeTopologicalInvariants(formula, electronic, spaceGroup);
      trackInvariantResult(formula, result);
      res.json(result.bandInversion);
    } catch (e) {
      res.status(500).json({ error: "Failed to detect band inversions" });
    }
  });

  app.get("/api/surface-states/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const result = computeTopologicalInvariants(formula, electronic, spaceGroup);
      trackInvariantResult(formula, result);
      res.json(result.surfaceStates);
    } catch (e) {
      res.status(500).json({ error: "Failed to detect surface states" });
    }
  });

  app.get("/api/symmetry-indicators/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const crystalSystem = typeof req.query.crystalSystem === "string" ? req.query.crystalSystem : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const result = computeTopologicalInvariants(formula, electronic, spaceGroup, crystalSystem);
      trackInvariantResult(formula, result);
      res.json(result.symmetryIndicator);
    } catch (e) {
      res.status(500).json({ error: "Failed to compute symmetry indicators" });
    }
  });

  app.get("/api/ml-topology/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const result = computeTopologicalInvariants(formula, electronic, spaceGroup);
      trackInvariantResult(formula, result);
      res.json(result.mlTopology);
    } catch (e) {
      res.status(500).json({ error: "Failed to predict topology via ML" });
    }
  });

  app.get("/api/tsc-score/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const spaceGroup = typeof req.query.spaceGroup === "string" ? req.query.spaceGroup : undefined;
      const crystalSystem = typeof req.query.crystalSystem === "string" ? req.query.crystalSystem : undefined;
      const tcParam = typeof req.query.tc === "string" ? parseFloat(req.query.tc) : undefined;
      const electronic = computeElectronicStructure(formula, spaceGroup || null);
      const result = computeTopologicalInvariants(formula, electronic, spaceGroup, crystalSystem, tcParam);
      trackInvariantResult(formula, result);
      res.json(result.tscScore);
    } catch (e) {
      res.status(500).json({ error: "Failed to compute TSC score" });
    }
  });

  app.get("/api/fermi-surface/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const result = computeFermiSurface(formula);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to compute Fermi surface" });
    }
  });

  app.get("/api/hydrogen-network/:formula", async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const analysis = analyzeHydrogenNetwork(formula);
      res.json(analysis);
    } catch (e) {
      res.status(500).json({ error: "Failed to analyze hydrogen network" });
    }
  });

  app.get("/api/hydrogen-network-stats", async (_req, res) => {
    try {
      res.json(getHydrogenNetworkStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch hydrogen network stats" });
    }
  });

  app.post("/api/inverse-design/start", generalLimiter, async (req, res) => {
    try {
      const { targetTc, maxPressure, minLambda, maxHullDistance, metallicRequired, phononStable, preferredPrototypes, preferredElements, excludeElements } = req.body;
      if (!targetTc || typeof targetTc !== "number" || targetTc < 1 || targetTc > 500) {
        return res.status(400).json({ error: "targetTc must be between 1 and 500" });
      }
      const id = `inv-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const target: TargetProperties = {
        targetTc,
        maxPressure: maxPressure ?? 1,
        minLambda: minLambda ?? 1.5,
        maxHullDistance: maxHullDistance ?? 0.05,
        metallicRequired: metallicRequired ?? true,
        phononStable: phononStable ?? true,
        preferredPrototypes: preferredPrototypes ?? [],
        preferredElements: preferredElements ?? [],
        excludeElements: excludeElements ?? [],
      };
      const campaign = createCampaign(id, target);
      await storage.insertInverseDesignCampaign({
        id,
        targetTc: target.targetTc,
        maxPressure: target.maxPressure,
        minLambda: target.minLambda,
        maxHullDistance: target.maxHullDistance,
        metallicRequired: target.metallicRequired,
        phononStable: target.phononStable,
        preferredPrototypes: target.preferredPrototypes ?? [],
        preferredElements: target.preferredElements ?? [],
        excludeElements: target.excludeElements ?? [],
        status: "active",
        cyclesRun: 0,
        bestTcAchieved: 0,
        bestDistance: 1.0,
        candidatesGenerated: 0,
        candidatesPassedPipeline: 0,
      });
      res.json({ id, status: "active", target });
    } catch (e: any) {
      res.status(500).json({ error: e.message || "Failed to start campaign" });
    }
  });

  app.get("/api/inverse-design/campaigns", generalLimiter, async (_req, res) => {
    try {
      const dbCampaigns = await storage.getInverseDesignCampaigns();
      const results = dbCampaigns.map(c => {
        const live = getCampaign(c.id);
        if (live) {
          return getCampaignStats(live);
        }
        return {
          id: c.id,
          target: { targetTc: c.targetTc, maxPressure: c.maxPressure, minLambda: c.minLambda, maxHullDistance: c.maxHullDistance, metallicRequired: c.metallicRequired, phononStable: c.phononStable },
          status: c.status,
          cyclesRun: c.cyclesRun,
          bestTcAchieved: c.bestTcAchieved,
          bestDistance: c.bestDistance,
          candidatesGenerated: c.candidatesGenerated,
          candidatesPassedPipeline: c.candidatesPassedPipeline,
          topCandidates: (c.topCandidates as any) ?? [],
          convergenceHistory: (c.convergenceHistory as any) ?? [],
        };
      });
      res.json(results);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch campaigns" });
    }
  });

  app.get("/api/inverse-design/campaign/:id", generalLimiter, async (req, res) => {
    try {
      const { id } = req.params;
      const live = getCampaign(id);
      if (live) {
        return res.json(getCampaignStats(live));
      }
      const dbCampaign = await storage.getInverseDesignCampaignById(id);
      if (!dbCampaign) {
        return res.status(404).json({ error: "Campaign not found" });
      }
      res.json({
        id: dbCampaign.id,
        target: { targetTc: dbCampaign.targetTc, maxPressure: dbCampaign.maxPressure, minLambda: dbCampaign.minLambda, maxHullDistance: dbCampaign.maxHullDistance },
        status: dbCampaign.status,
        cyclesRun: dbCampaign.cyclesRun,
        bestTcAchieved: dbCampaign.bestTcAchieved,
        bestDistance: dbCampaign.bestDistance,
        candidatesGenerated: dbCampaign.candidatesGenerated,
        candidatesPassedPipeline: dbCampaign.candidatesPassedPipeline,
        topCandidates: (dbCampaign.topCandidates as any) ?? [],
        convergenceHistory: (dbCampaign.convergenceHistory as any) ?? [],
      });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch campaign" });
    }
  });

  app.delete("/api/inverse-design/campaign/:id", generalLimiter, async (req, res) => {
    try {
      const { id } = req.params;
      removeCampaign(id);
      await storage.deleteInverseDesignCampaign(id);
      res.json({ deleted: true });
    } catch (e) {
      res.status(500).json({ error: "Failed to delete campaign" });
    }
  });

  app.post("/api/inverse-design/campaign/:id/pause", generalLimiter, async (req, res) => {
    try {
      const { id } = req.params;
      pauseCampaign(id);
      await storage.updateInverseDesignCampaign(id, { status: "paused" });
      res.json({ paused: true });
    } catch (e) {
      res.status(500).json({ error: "Failed to pause campaign" });
    }
  });

  app.get("/api/inverse-design/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getInverseDesignStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to get stats" });
    }
  });

  app.post("/api/gradient-design/optimize", writeLimiter, async (req, res) => {
    try {
      const { formula, targetTc, maxPressure, minLambda, maxSteps } = req.body;
      if (!formula || !targetTc) {
        return res.status(400).json({ error: "formula and targetTc are required" });
      }
      const target: TargetProperties = {
        targetTc: Number(targetTc),
        maxPressure: Number(maxPressure ?? 50),
        minLambda: Number(minLambda ?? 1.5),
        maxHullDistance: 0.05,
        metallicRequired: true,
        phononStable: true,
      };
      const result = await runDifferentiableOptimization(formula, target, Number(maxSteps ?? 20));
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Gradient optimization failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/gradient-design/batch", writeLimiter, async (req, res) => {
    try {
      const { targetTc, maxPressure, minLambda, seedCount, stepsPerSeed } = req.body;
      if (!targetTc) {
        return res.status(400).json({ error: "targetTc is required" });
      }
      const target: TargetProperties = {
        targetTc: Number(targetTc),
        maxPressure: Number(maxPressure ?? 50),
        minLambda: Number(minLambda ?? 1.5),
        maxHullDistance: 0.05,
        metallicRequired: true,
        phononStable: true,
      };
      const cycle = await runGradientDescentCycle(target, Number(seedCount ?? 6), Number(stepsPerSeed ?? 15));
      res.json(cycle);
    } catch (e: any) {
      res.status(500).json({ error: "Batch gradient optimization failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/gradient-design/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDifferentiableOptimizerStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to get gradient design stats" });
    }
  });

  app.post("/api/structure-design/generate", writeLimiter, async (req, res) => {
    try {
      const { targetTc, motifCount, elementsPerSite } = req.body;
      const numTargetTc = Number(targetTc);
      if (numTargetTc == null || !Number.isFinite(numTargetTc) || numTargetTc <= 0) {
        return res.status(400).json({ error: "targetTc must be a positive number" });
      }
      const numMotifCount = Number(motifCount ?? 4);
      const numElementsPerSite = Number(elementsPerSite ?? 3);
      const results = await runStructureFirstDesign(
        numTargetTc,
        Number.isFinite(numMotifCount) && numMotifCount > 0 ? numMotifCount : 4,
        Number.isFinite(numElementsPerSite) && numElementsPerSite > 0 ? numElementsPerSite : 3,
      );
      res.json(results);
    } catch (e: any) {
      res.status(500).json({ error: "Structure-first design failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/structure-design/motifs", generalLimiter, (_req, res) => {
    try {
      res.json(getMotifLibrarySummary());
    } catch (e) {
      res.status(500).json({ error: "Failed to get motif library" });
    }
  });

  app.get("/api/structure-design/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getStructureDiffusionStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to get structure design stats" });
    }
  });

  app.post("/api/physics-constraints/check", writeLimiter, (req, res) => {
    try {
      const { formula, maxPressureGPa } = req.body;
      if (!formula || typeof formula !== "string") {
        return res.status(400).json({ error: "formula is required" });
      }
      const result = checkPhysicsConstraints(formula, {
        maxPressureGPa: typeof maxPressureGPa === "number" ? maxPressureGPa : undefined,
      });
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Constraint check failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/physics-constraints/batch", writeLimiter, (req, res) => {
    try {
      const { formulas, maxPressureGPa } = req.body;
      if (!Array.isArray(formulas) || formulas.length === 0) {
        return res.status(400).json({ error: "formulas array is required" });
      }
      const result = constraintGuidedGenerate(formulas.slice(0, 200), {
        maxPressureGPa: typeof maxPressureGPa === "number" ? maxPressureGPa : undefined,
      });
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Batch constraint check failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-constraints/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getConstraintEngineStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to get constraint stats" });
    }
  });

  app.post("/api/sc-pillars/evaluate", writeLimiter, async (req, res) => {
    try {
      const { formula, targets, maxPressureGPa } = req.body;
      if (!formula || typeof formula !== "string") {
        return res.status(400).json({ error: "formula is required" });
      }
      const result = await evaluatePillars(formula, targets ?? DEFAULT_PILLAR_TARGETS, {
        maxPressureGPa: typeof maxPressureGPa === "number" ? maxPressureGPa : undefined,
      });
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Pillar evaluation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/sc-pillars/generate", writeLimiter, async (req, res) => {
    try {
      const { candidatesPerTemplate, targets } = req.body;
      const result = await runPillarGuidedGeneration(
        targets ?? DEFAULT_PILLAR_TARGETS,
        Number(candidatesPerTemplate ?? 6),
      );
      res.json({
        totalGenerated: result.length,
        candidates: result.slice(0, 50).map(c => ({
          formula: c.formula,
          fitness: c.evaluation.compositeFitness,
          tc: c.evaluation.tcPredicted,
          pillars: c.evaluation.satisfiedPillars,
          lambda: c.evaluation.lambda,
          omegaLogK: c.evaluation.omegaLogK,
          dos: c.evaluation.dos,
          nesting: c.evaluation.nestingScore,
          motif: c.evaluation.motifMatch,
          pairingGlue: c.evaluation.pairingGlue.compositePairingGlue,
          pairingMechanism: c.evaluation.pairingGlue.dominantMechanism,
          instabilityScore: c.evaluation.instability.compositeInstability,
          dominantInstability: c.evaluation.instability.dominantInstability,
          hCageScore: c.evaluation.hydrogenCage.compositeHydrogenScore,
          hCageType: c.evaluation.hydrogenCage.bondingType,
          fsCylindrical: c.evaluation.fermiSurface.cylindricalScore,
          fsNesting: c.evaluation.fermiSurface.nestingStrength,
          fsVanHoveDist: c.evaluation.fermiSurface.vanHoveDistance,
          fsDominant: c.evaluation.fermiSurface.dominantGeometry,
          weakest: c.evaluation.weakestPillar,
          rationale: c.designRationale,
        })),
      });
    } catch (e: any) {
      res.status(500).json({ error: "Pillar generation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/sc-pillars/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getPillarOptimizerStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to get pillar optimizer stats" });
    }
  });

  app.get("/api/pairing/profile/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length > 80) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const profile = computePairingProfile(formula);
      res.json(profile);
    } catch (e: any) {
      res.status(500).json({ error: "Pairing profile failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pairing/features/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length > 80) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const features = computePairingFeatureVector(formula);
      res.json(features);
    } catch (e: any) {
      res.status(500).json({ error: "Pairing features failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/reaction-network/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const pressureGpa = parseFloat(req.query.pressure as string) || 0;
      const temperatureK = parseFloat(req.query.temperature as string) || 300;
      const result = analyzeReactionNetwork(formula, Math.max(0, Math.min(pressureGpa, 500)), Math.max(1, Math.min(temperatureK, 5000)));
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Reaction network analysis failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/genome/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length > 80) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const genome = encodeGenome(formula);
      res.json(genome);
    } catch (e: any) {
      res.status(500).json({ error: "Genome encoding failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/genome/similarity", generalLimiter, (req, res) => {
    try {
      const { target, candidates, topK } = req.body;
      if (!target || typeof target !== "string") {
        return res.status(400).json({ error: "Missing target formula" });
      }
      if (!candidates || !Array.isArray(candidates) || candidates.length === 0) {
        return res.status(400).json({ error: "Missing candidates array" });
      }
      const results = findSimilar(target, candidates.slice(0, 100), topK ?? 10);
      res.json({ target, results });
    } catch (e: any) {
      res.status(500).json({ error: "Similarity search failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/genome/diversity", generalLimiter, (req, res) => {
    try {
      const { formulas } = req.body;
      if (!formulas || !Array.isArray(formulas) || formulas.length < 2) {
        return res.status(400).json({ error: "Need at least 2 formulas" });
      }
      const diversity = genomeDiversity(formulas.slice(0, 50));
      res.json({ diversity, count: Math.min(formulas.length, 50) });
    } catch (e: any) {
      res.status(500).json({ error: "Diversity computation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/genome/inverse", generalLimiter, (req, res) => {
    try {
      const { target, pool, topK } = req.body;
      if (!target || typeof target !== "string") {
        return res.status(400).json({ error: "Missing target formula" });
      }
      if (!pool || !Array.isArray(pool) || pool.length === 0) {
        return res.status(400).json({ error: "Missing candidate pool" });
      }
      const results = genomeGuidedInverseDesign(target, pool.slice(0, 100), topK ?? 5);
      res.json({ target, results });
    } catch (e: any) {
      res.status(500).json({ error: "Genome inverse design failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/genome/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getGenomeCacheStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to get genome stats" });
    }
  });

  app.get("/api/band-surrogate/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length > 80 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const prediction = predictBandStructure(formula);
      res.json(prediction);
    } catch (e: any) {
      res.status(500).json({ error: "Band structure surrogate prediction failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/stability-predict/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const pressureGpa = parseFloat(req.query.pressure as string) || 0;
      const prediction = predictStability(formula, Math.max(0, Math.min(pressureGpa, 500)));
      res.json(prediction);
    } catch (e: any) {
      res.status(500).json({ error: "Stability prediction failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/interface/:layerA/:layerB", generalLimiter, (req, res) => {
    try {
      const layerA = decodeURIComponent(req.params.layerA as string);
      const layerB = decodeURIComponent(req.params.layerB as string);
      if (!layerA || !layerB || layerA.length > 100 || layerB.length > 100) {
        return res.status(400).json({ error: "Invalid layer formulas" });
      }
      const analysis = analyzeInterface(layerA, layerB);
      res.json(analysis);
    } catch (e: any) {
      res.status(500).json({ error: "Interface analysis failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/interface-candidates", generalLimiter, (_req, res) => {
    try {
      const candidates = generateHeterostructureCandidates();
      res.json({ candidates, count: candidates.length });
    } catch (e: any) {
      res.status(500).json({ error: "Heterostructure generation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/heterostructure/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getHeterostructureStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch heterostructure stats" });
    }
  });

  app.get("/api/heterostructure/substrates", generalLimiter, async (_req, res) => {
    try {
      const substrates = getSubstrateDatabase();
      res.json({ substrates, count: substrates.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch substrates" });
    }
  });

  app.get("/api/heterostructure/best-substrates/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const topN = parseInt(req.query.topN as string) || 5;
      const results = findBestSubstrates(formula, topN);
      res.json({ formula, matches: results });
    } catch (e) {
      res.status(500).json({ error: "Failed to find best substrates" });
    }
  });

  app.post("/api/heterostructure/generate", generalLimiter, async (req, res) => {
    try {
      const { film, substrate } = req.body;
      if (!film || !substrate) {
        return res.status(400).json({ error: "Both film and substrate formulas required" });
      }
      const result = generateHeterostructure(film, substrate);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to generate heterostructure" });
    }
  });

  app.post("/api/heterostructure/batch-generate", generalLimiter, async (req, res) => {
    try {
      const { films, maxPerFilm } = req.body;
      if (!films || !Array.isArray(films) || films.length === 0) {
        return res.status(400).json({ error: "Array of film formulas required" });
      }
      const results = generateBilayerCandidates(films, maxPerFilm || 5);
      res.json({ results, count: results.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to batch generate heterostructures" });
    }
  });

  app.get("/api/interface-relaxation/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getInterfaceRelaxationStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch interface relaxation stats" });
    }
  });

  app.post("/api/interface-relaxation/relax", generalLimiter, async (req, res) => {
    try {
      const { film, substrate } = req.body;
      if (!film || !substrate) {
        return res.status(400).json({ error: "Both film and substrate formulas required" });
      }
      const result = await relaxInterface(film, substrate);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Interface relaxation failed" });
    }
  });

  app.post("/api/interface-relaxation/score-candidates", generalLimiter, async (req, res) => {
    try {
      const { films, topN } = req.body;
      if (!films || !Array.isArray(films) || films.length === 0) {
        return res.status(400).json({ error: "Array of film formulas required" });
      }
      const candidates = scoreInterfaceCandidatesForActiveLearning(films, topN || 10);
      res.json({ candidates, count: candidates.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to score interface candidates" });
    }
  });

  app.post("/api/interface-relaxation/discover", generalLimiter, async (req, res) => {
    try {
      const { films, budget } = req.body;
      if (!films || !Array.isArray(films) || films.length === 0) {
        return res.status(400).json({ error: "Array of film formulas required" });
      }
      const results = await runInterfaceDiscoveryForActiveLearning(films, budget || 3);
      res.json({ results, count: results.length });
    } catch (e) {
      res.status(500).json({ error: "Interface discovery failed" });
    }
  });

  app.get("/api/quantum-criticality/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula as string);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const analysis = detectQuantumCriticality(formula);
      res.json(analysis);
    } catch (e: any) {
      res.status(500).json({ error: "Quantum criticality analysis failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/discovery-memory/patterns", generalLimiter, (_req, res) => {
    try {
      const topPatterns = discoveryMemory.getTopPatterns(20);
      const stats = discoveryMemory.getStats();
      const clusters = discoveryMemory.getClusters();
      const bias = discoveryMemory.biasGenerationFromMemory();
      res.json({ topPatterns, stats, clusters, bias });
    } catch (e: any) {
      res.status(500).json({ error: "Discovery memory query failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory/features/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula as string);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const existing = getFeatureRecord(formula);
      if (existing) return res.json(existing);
      const record = buildAndStoreFeatureRecord(formula);
      res.json(record);
    } catch (e: any) {
      res.status(500).json({ error: "Feature extraction failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory/discovered", generalLimiter, (_req, res) => {
    try {
      const theories = getDiscoveredTheories();
      const datasetSize = getDatasetSize();
      const validationStats = getValidationStats();
      res.json({ theories, datasetSize, theoryCount: theories.length, validationStats });
    } catch (e: any) {
      res.status(500).json({ error: "Theory query failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/theory/discover", generalLimiter, (_req, res) => {
    try {
      const dataset = getFeatureDataset();
      if (dataset.length < 10) {
        return res.status(400).json({ error: "Insufficient data", detail: `Need at least 10 feature records, have ${dataset.length}` });
      }
      const targets: Array<"tc" | "pairing_strength" | "lambda"> = ["tc", "pairing_strength"];
      const results: Record<string, any> = {};
      for (const target of targets) {
        const fieldMap: Record<string, string> = { tc: "tc", pairing_strength: "pairingStrength", lambda: "lambda" };
        const targetField = fieldMap[target];
        const validData = dataset.filter(r => {
          const val = target === "tc" ? r.tc : target === "pairing_strength" ? r.pairingStrength : r.featureVector.electron_phonon_lambda;
          return val !== undefined && val !== null && Number.isFinite(val) && val > 0;
        });
        if (validData.length < 10) continue;
        const dataForSR = validData.map(r => ({
          features: r.featureVector as Record<string, number>,
          target: target === "tc" ? r.tc : target === "pairing_strength" ? r.pairingStrength : r.featureVector.electron_phonon_lambda,
        }));
        const theories = runSymbolicRegression(dataForSR, target, { populationSize: 100, generations: 30 });
        results[target] = theories.slice(0, 5);
      }
      res.json({ results, datasetSize: dataset.length });
    } catch (e: any) {
      res.status(500).json({ error: "Theory discovery failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory/multi-scale/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula as string);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const multiScale = computeMultiScaleFeatures(formula);
      const crossScale = computeCrossScaleCoupling(multiScale);
      res.json({ multiScale, crossScale });
    } catch (e: any) {
      res.status(500).json({ error: "Multi-scale analysis failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory/sensitivity/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula as string);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const sensitivity = runSensitivityAnalysis(formula);
      res.json(sensitivity);
    } catch (e: any) {
      res.status(500).json({ error: "Sensitivity analysis failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory/parameters", generalLimiter, (_req, res) => {
    try {
      const params = getPhysicsParameters();
      const history = getParameterHistory();
      const performance = getModelPerformance();
      res.json({ current: params, history, performance });
    } catch (e: any) {
      res.status(500).json({ error: "Parameter query failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/generator-allocations", generalLimiter, (_req, res) => {
    try {
      const allocations = getGeneratorAllocations();
      res.json(allocations);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch generator allocations", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory/performance", generalLimiter, (_req, res) => {
    try {
      const metrics = getPerformanceMetrics();
      res.json(metrics);
    } catch (e: any) {
      res.status(500).json({ error: "Performance metrics failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/fermi-clusters", generalLimiter, (_req, res) => {
    try {
      const clusters = getAllClusters();
      const stats = getClusterStats();
      const guidance = getClusterGuidance();
      res.json({ clusters, stats, guidance });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch Fermi clusters", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/fermi-clusters/:clusterId", generalLimiter, (req, res) => {
    try {
      const clusterId = req.params.clusterId;
      const cluster = getCluster(clusterId);
      if (!cluster) return res.status(404).json({ error: "Cluster not found" });
      res.json(cluster);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch Fermi cluster", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape/embedding", generalLimiter, (_req, res) => {
    try {
      const points = getEmbeddingDataset();
      res.json({ points, count: points.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch landscape embedding", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape/zones", generalLimiter, (_req, res) => {
    try {
      const zoneMap = getZoneMap();
      res.json(zoneMap);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch discovery zones", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getLandscapeStats();
      const zoneMap = getZoneMap();
      res.json({
        ...stats,
        zoneCount: zoneMap.zones.length,
        topZoneCount: zoneMap.topZones.length,
        coveragePercent: zoneMap.coveragePercent,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch landscape stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape/guidance", generalLimiter, (_req, res) => {
    try {
      const guidance = getFullLandscapeGuidance();
      res.json(guidance);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch landscape guidance", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/constraint-solver/solve", generalLimiter, (req, res) => {
    try {
      const targetTc = Number(req.query.targetTc) || 200;
      const muStar = Number(req.query.muStar) || 0.10;
      const pressure = Number(req.query.pressure) || 0;
      const solution = solveConstraints(targetTc, muStar, pressure);
      res.json(solution);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to solve constraints", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/constraint-solver/evaluate/:formula", generalLimiter, (req, res) => {
    try {
      const formula = req.params.formula;
      const targetTc = Number(req.query.targetTc) || 200;
      const muStar = Number(req.query.muStar) || 0.10;
      const pressure = Number(req.query.pressure) || 0;
      const solution = solveConstraints(targetTc, muStar, pressure);
      const evaluation = evaluateFormulaAgainstConstraints(formula, solution);
      res.json({ ...evaluation, targetConstraints: solution });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to evaluate formula against constraints", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-pathways/search/:formula", generalLimiter, (req, res) => {
    try {
      const formula = req.params.formula;
      const sourceTc = Number(req.query.tc) || 100;
      const sourcePressure = Number(req.query.pressure) || 100;
      const pathway = searchPressurePathways(formula, sourceTc, sourcePressure);
      res.json(pathway);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to search pressure pathways", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-pathways/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getPathwayStats();
      const candidates = getAmbientCandidatesFromPathways();
      res.json({ ...stats, ambientCandidates: candidates });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch pathway stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-pathway/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getSynthesisPathwayStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch synthesis pathway stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-pathway/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = computeSynthesisPathway(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute synthesis pathway", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis/reaction-network/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getReactionNetworkStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch reaction network stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis/reaction-network/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = buildReactionNetwork(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to build reaction network", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/band-operator/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getBandOperatorStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch band operator stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/band-operator/dispersion/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = await predictBandDispersion(formula);
      res.json({
        formula: result.formula,
        dispersion: result.dispersion,
        derivedQuantities: result.derivedQuantities,
        confidence: result.confidence,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict band dispersion", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/band-operator/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = await predictBandDispersion(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict band structure", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/constraint-graph/solve", generalLimiter, (req, res) => {
    try {
      const targetTc = Number(req.query.targetTc) || 200;
      const muStar = Number(req.query.muStar) || 0.10;
      const solution = solveConstraintGraph(targetTc, muStar);
      res.json(solution);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to solve constraint graph", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/constraint-graph/feasible-regions", generalLimiter, (req, res) => {
    try {
      const targetTc = Number(req.query.targetTc) || 200;
      const muStar = Number(req.query.muStar) || 0.10;
      const result = getFeasibleRegions(targetTc, muStar);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute feasible regions", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/hypothesis/active", generalLimiter, (_req, res) => {
    try {
      const hypotheses = getActiveHypotheses();
      const stats = getHypothesisStats();
      res.json({ hypotheses, stats });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch active hypotheses", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/hypothesis/all", generalLimiter, (_req, res) => {
    try {
      const hypotheses = getAllHypotheses();
      const stats = getHypothesisStats();
      res.json({ hypotheses, stats });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch all hypotheses", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/hypothesis/test/:id", generalLimiter, (req, res) => {
    try {
      const id = req.params.id;
      const result = testHypothesisById(id);
      if (!result.hypothesis) {
        return res.status(404).json({ error: "Hypothesis not found" });
      }
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to test hypothesis", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape-intelligence/frontier", generalLimiter, (_req, res) => {
    try {
      const frontier = analyzeFrontier();
      res.json(frontier);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to analyze frontier", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape-intelligence/novelty/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const novelty = computeNoveltyScore(formula);
      res.json(novelty);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute novelty score", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape-intelligence/zones", generalLimiter, (_req, res) => {
    try {
      const zones = analyzeZoneIntelligence();
      const stats = getLandscapeIntelligenceStats();
      res.json({ zones, stats });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to analyze zone intelligence", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/landscape-intelligence/strategy", generalLimiter, (_req, res) => {
    try {
      const strategy = generateExplorationStrategy();
      res.json(strategy);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate exploration strategy", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-variables/stats", generalLimiter, (_req, res) => {
    try {
      const parameterSpace = getParameterSpace();
      const optimizerStats = getSynthesisOptimizerStats();
      res.json({ parameterSpace, optimizerStats });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch synthesis variable stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-simulator/stats", generalLimiter, (_req, res) => {
    try {
      const simulator = getSimulatorStats();
      const learning = getSynthesisLearningStats();
      res.json({ simulator, learning });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch synthesis simulator stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-verdict/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 1 || formula.length > 100 || !/^[A-Za-z0-9.]+$/.test(formula)) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const materialClass = (req.query.class as string) || "default";
      const targetTc = parseFloat(req.query.targetTc as string) || 100;
      const verdict = computeSynthesisVerdict(formula, materialClass, Math.max(1, Math.min(targetTc, 500)));
      res.json(verdict);
    } catch (e: any) {
      res.status(500).json({ error: "Synthesis verdict failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-discovery/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getSynthesisDiscoveryStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch synthesis discovery stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-discovery/ga-evolution", generalLimiter, (_req, res) => {
    try {
      res.json(getGAEvolutionStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch GA evolution stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/generator-competition/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getGeneratorCompetitionStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch generator competition stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/structural-motifs/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getStructuralMotifStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch structural motif stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-discovery/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const matClass = formula.includes("H") ? "Hydrides" : "default";

      const path = optimizeSynthesisPath(formula, matClass);
      const similar = querySimilarSynthesis(formula, 5);
      const sv = defaultSynthesisVector(matClass);
      const effects = simulateSynthesisEffects(formula, matClass, sv);

      res.json({
        formula,
        synthesisPath: path,
        defaultEffects: effects,
        similarSyntheses: similar.map(s => ({
          formula: s.formula,
          tc: s.resultTc,
          temperature: s.synthesisVector.temperature,
          pressure: s.synthesisVector.pressure,
          coolingRate: s.synthesisVector.coolingRate,
        })),
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute synthesis discovery", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-planner/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getSynthesisPlannerStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch synthesis planner stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-planner/routes/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = planSynthesisRoutes(formula, { maxRoutes: 8 });
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to plan synthesis routes", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-planner/recent", generalLimiter, async (_req, res) => {
    try {
      const processes = await storage.getSynthesisProcesses(20);
      const recent = processes
        .filter((p: any) => p.discoveredAt)
        .sort((a: any, b: any) => new Date(b.discoveredAt).getTime() - new Date(a.discoveredAt).getTime())
        .slice(0, 10);
      res.json(recent);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch recent synthesis routes", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/heuristic-synthesis/routes/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const routes = generateHeuristicRoutes(formula);
      res.json({ formula, routes, ruleCount: routes.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate heuristic routes", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/heuristic-synthesis/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getHeuristicGeneratorStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch heuristic synthesis stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/ml-synthesis/predict/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = predictSynthesisFeasibility(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict synthesis feasibility", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/ml-synthesis/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getSynthesisPredictorStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch ML synthesis stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/ml-synthesis/score/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const score = computeSynthesisScore(formula);
      res.json({ formula, ...score });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute synthesis score", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-gate/evaluate/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = evaluateSynthesisGate(formula);
      res.json({ formula, ...result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to evaluate synthesis gate", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/synthesis-gate/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getSynthesisGateStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch synthesis gate stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/xgboost/evaluated-stats", generalLimiter, (_req, res) => {
    try {
      res.json(getEvaluatedDatasetStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch evaluated dataset stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/xgboost/composition-features/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const cf = computeCompositionFeatures(formula);
      res.json({
        formula,
        featureCount: COMPOSITION_FEATURE_NAMES.length,
        featureNames: COMPOSITION_FEATURE_NAMES,
        features: cf,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute composition features", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/xgboost/uncertainty/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const features = await extractFeatures(formula);
      const result = await gbPredictWithUncertainty(features, formula);
      res.json({ formula, ...result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute XGB uncertainty", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/xgboost/ensemble-stats", generalLimiter, (_req, res) => {
    try {
      res.json(getXGBEnsembleStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch ensemble stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/xgboost/version-history", generalLimiter, (_req, res) => {
    try {
      res.json(getModelVersionHistory());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch version history", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/gnn/version-history", generalLimiter, (_req, res) => {
    try {
      const history = getGNNVersionHistory();
      const currentVersion = getGNNModelVersion();
      const latest = history.length > 0 ? history[history.length - 1] : null;
      const r2Trend = history.slice(-10).map(h => ({ version: h.version, r2: h.r2 }));
      const maeTrend = history.slice(-10).map(h => ({ version: h.version, mae: h.mae }));
      res.json({
        currentVersion,
        ensembleSize: latest?.ensembleSize ?? 4,
        latestMetrics: latest ? { r2: latest.r2, mae: latest.mae, rmse: latest.rmse, datasetSize: latest.datasetSize } : null,
        r2Trend,
        maeTrend,
        history: history.slice(-20),
        uncertaintyMethods: ["ensemble-variance", "mc-dropout", "latent-distance"],
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch GNN version history", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/gnn/active-learning-stats", generalLimiter, (_req, res) => {
    try {
      const convergence = getActiveLearningStats();
      const cycles = getActiveLearningCycleHistory();
      const dftStats = getDFTTrainingDatasetStats();
      const recentCycles = cycles.slice(-20);
      const avgUncertaintyTrend = recentCycles.map(c => ({
        cycle: c.cycle,
        before: c.avgCombinedUncertainty,
        after: c.uncertaintyAfter,
        reductionPct: c.uncertaintyReductionPct,
      }));
      res.json({
        convergence,
        totalCycles: cycles.length,
        recentCycles,
        avgUncertaintyTrend,
        dftDatasetStats: dftStats,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch active learning stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/surrogate-fitness/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getSurrogateFitnessStats();
      const pillarFeedback = getPillarDFTFeedbackStats();
      res.json({ ...stats, pillarDFTFeedback: pillarFeedback });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch surrogate fitness stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/gnn/predict/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const prediction = gnnPredictWithUncertainty(formula);
      const graph = buildCrystalGraph(formula);
      res.json({
        formula,
        prediction: {
          tc: prediction.tc,
          formationEnergy: prediction.formationEnergy,
          lambda: prediction.lambda,
          bandgap: prediction.bandgap,
          dosProxy: prediction.dosProxy,
          stabilityProbability: prediction.stabilityProbability,
          uncertainty: prediction.uncertainty,
          phononStability: prediction.phononStability,
          confidence: prediction.confidence,
          latentDistance: prediction.latentDistance,
        },
        uncertaintyBreakdown: prediction.uncertaintyBreakdown,
        graphStats: {
          nodes: graph.nodes.length,
          edges: graph.edges.length,
          threeBodyFeatures: graph.threeBodyFeatures.length,
          elements: [...new Set(graph.nodes.map(n => n.element))],
        },
        modelVersion: getGNNModelVersion(),
        ensembleSize: 4,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict with GNN", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/gnn/training-dataset-stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDFTTrainingDatasetStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch DFT training dataset stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/dos-surrogate/predict/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const useGNN = req.query.useGNN === "true";

      let latent: number[] | undefined;
      let gnnTc: number | null = null;

      if (useGNN) {
        try {
          const singlePred = getGNNPrediction(formula);
          const embLatent = singlePred?.latentEmbedding;
          const hasValidLatent = embLatent && embLatent.length > 0 && embLatent.some(v => v !== 0 && isFinite(v));
          if (hasValidLatent) latent = embLatent;
          gnnTc = singlePred?.predictedTc ?? null;
        } catch {}
      }

      const dosResult = predictDOS(formula, latent);

      const safe = (v: number) => isFinite(v) ? Math.round(v * 10000) / 10000 : 0;

      res.json({
        formula,
        orbitalDOS: {
          energyGrid: dosResult.orbitalDOS.energyGrid,
          totalDOS: dosResult.orbitalDOS.totalDOS.map(safe),
          s: dosResult.orbitalDOS.orbitalDOS.s.map(safe),
          p: dosResult.orbitalDOS.orbitalDOS.p.map(safe),
          d: dosResult.orbitalDOS.orbitalDOS.d.map(safe),
          f: dosResult.orbitalDOS.orbitalDOS.f.map(safe),
          dosAtFermi: safe(dosResult.orbitalDOS.dosAtFermi),
          orbitalDOSAtFermi: {
            s: safe(dosResult.orbitalDOS.orbitalDOSAtFermi.s),
            p: safe(dosResult.orbitalDOS.orbitalDOSAtFermi.p),
            d: safe(dosResult.orbitalDOS.orbitalDOSAtFermi.d),
            f: safe(dosResult.orbitalDOS.orbitalDOSAtFermi.f),
          },
        },
        vanHoveSingularities: dosResult.vanHoveSingularities,
        scores: {
          vhsScore: safe(dosResult.vhsScore),
          scFavorability: safe(dosResult.scFavorability),
          flatBandIndicator: safe(dosResult.flatBandIndicator),
          nestingScore: safe(dosResult.nestingScore),
          orbitalMixingAtFermi: safe(dosResult.orbitalMixingAtFermi),
        },
        isMetallic: dosResult.isMetallic,
        predictionTier: dosResult.predictionTier,
        wallTimeMs: dosResult.wallTimeMs,
        gnnTcPrediction: gnnTc,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict DOS", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/dos-surrogate/prefilter/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = dosPrefilter(formula);
      res.json({ formula, ...result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run DOS pre-filter", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/dos-surrogate/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDOSSurrogateStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get DOS surrogate stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/dos-surrogate/batch", writeLimiter, (req, res) => {
    try {
      const formulas: string[] = req.body?.formulas ?? [];
      if (!Array.isArray(formulas) || formulas.length === 0) {
        return res.status(400).json({ error: "Provide formulas array" });
      }

      const results = formulas.slice(0, 50).map(formula => {
        const dos = predictDOS(formula);
        return {
          formula,
          dosAtFermi: dos.orbitalDOS.dosAtFermi,
          vhsScore: dos.vhsScore,
          scFavorability: dos.scFavorability,
          flatBandIndicator: dos.flatBandIndicator,
          nestingScore: dos.nestingScore,
          isMetallic: dos.isMetallic,
          vhsCount: dos.vanHoveSingularities.length,
          topVHS: dos.vanHoveSingularities.slice(0, 3).map(v => ({
            energyEv: v.energyEv,
            type: v.type,
            dominantOrbital: v.dominantOrbital,
            strength: v.strength,
          })),
          wallTimeMs: dos.wallTimeMs,
        };
      });

      res.json({ results, count: results.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to batch predict DOS", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/retrosynthesis/routes/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = generateRetrosynthesisRoutes(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate retrosynthesis routes", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/retrosynthesis/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getRetrosynthesisStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch retrosynthesis stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/defect-engine/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDefectEngineStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch defect engine stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/cross-engine/stats", generalLimiter, (_req, res) => {
    try {
      res.json(crossEngineHub.getStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch cross-engine stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/cross-engine/patterns", generalLimiter, (_req, res) => {
    try {
      res.json({
        patterns: crossEngineHub.getGlobalPatterns(),
        physicsGuidance: crossEngineHub.getPhysicsGuidance(),
        synthesisGuidance: crossEngineHub.getSynthesisGuidance(),
        topologicalGuidance: crossEngineHub.getTopologicalGuidance(),
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch cross-engine patterns", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/cross-engine/insights/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const insight = crossEngineHub.getInsightsFor(formula);
      if (!insight) return res.status(404).json({ error: "No insights found for formula" });
      res.json(insight);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch cross-engine insights", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/defect-engine/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const variants = generateDefectVariants(formula);
      const withAdjustments = variants.map(v => {
        const adj = adjustElectronicStructure(1.0, 0.5, v.defectDensity, v.type);
        return { ...v, electronicAdjustment: adj };
      });
      res.json({ formula, defectVariants: withAdjustments });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate defect variants", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/disorder-generator/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDisorderGeneratorStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch disorder generator stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/disorder-generator/search-limits", generalLimiter, (_req, res) => {
    try {
      res.json(getSearchLimits());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch search limits", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/disorder-generator/generate", writeLimiter, (req, res) => {
    try {
      const { base, disorder, supercellSize } = req.body;
      if (!base || !disorder || !disorder.type || !disorder.element) {
        return res.status(400).json({ error: "Missing required fields: base, disorder.type, disorder.element" });
      }
      const validTypes: DisorderType[] = ["vacancy", "substitution", "interstitial", "site-mixing", "amorphous"];
      if (!validTypes.includes(disorder.type)) {
        return res.status(400).json({ error: `Invalid disorder type. Must be one of: ${validTypes.join(", ")}` });
      }
      const fraction = disorder.fraction ?? 0.05;
      const maxFraction = disorder.type === "amorphous" ? 1.0 : 0.30;
      if (fraction < 0.001 || fraction > maxFraction) {
        return res.status(400).json({ error: `Fraction must be between 0.001 and ${maxFraction}` });
      }
      const spec: DisorderSpec = {
        type: disorder.type,
        element: disorder.element,
        fraction,
        substituent: disorder.substituent,
      };
      const result = generateDisorderedStructure(base, spec, supercellSize);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate disordered structure", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/disorder-generator/batch", writeLimiter, (req, res) => {
    try {
      const { base, fractions } = req.body;
      if (!base) return res.status(400).json({ error: "Missing required field: base" });
      const results = generateAllDisorderVariants(base, fractions);
      res.json({
        base,
        totalVariants: results.length,
        variants: results.map(r => ({
          type: r.disorder.type,
          element: r.disorder.element,
          substituent: r.disorder.substituent,
          fraction: r.disorder.fraction,
          totalAtoms: r.totalAtoms,
          defectCount: r.defectCount,
          defectFraction: r.defectFraction,
          formationEnergy: r.formationEnergyEstimate,
          tcModifier: r.tcModifierEstimate,
          notes: r.notes,
          disorderScore: r.metrics?.disorderScore ?? null,
          disorderClass: r.metrics?.disorderClass ?? null,
          amorphousMethod: r.amorphousMethod ?? null,
        })),
        bestVariant: results.length > 0 ?
          results.reduce((best, r) => r.tcModifierEstimate > best.tcModifierEstimate ? r : best) : null,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate batch disorder variants", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/disorder-generator/suggest/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const suggestions = suggestDisorders(formula);
      res.json({ formula, suggestions });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to suggest disorders", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/disorder-metrics/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDisorderMetricsStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch disorder metrics stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/disorder-metrics/analyze", writeLimiter, (req, res) => {
    try {
      const { atoms, bondCutoffFactor } = req.body;
      if (!atoms || !Array.isArray(atoms) || atoms.length === 0) {
        return res.status(400).json({ error: "Missing required field: atoms (array of {element, x, y, z})" });
      }
      if (atoms.length > 500) {
        return res.status(400).json({ error: "Too many atoms (max 500)" });
      }
      const metrics = computeDisorderMetrics(atoms, bondCutoffFactor);
      const mlFeatures = extractMLFeatures(metrics);
      res.json({
        metrics: {
          disorderScore: metrics.disorderScore,
          disorderClass: metrics.disorderClass,
          bondMean: metrics.bondMean,
          bondVariance: metrics.bondVariance,
          bondStdDev: metrics.bondStdDev,
          bondMin: metrics.bondMin,
          bondMax: metrics.bondMax,
          totalBonds: metrics.totalBonds,
          coordinationMean: metrics.coordinationMean,
          coordinationVariance: metrics.coordinationVariance,
          coordinationMin: metrics.coordinationMin,
          coordinationMax: metrics.coordinationMax,
          idealCoordination: metrics.idealCoordination,
          coordinationDeficit: metrics.coordinationDeficit,
          localStrainMean: metrics.localStrainMean,
          localStrainMax: metrics.localStrainMax,
          defectNeighborStrain: metrics.defectNeighborStrain,
          vacancyFraction: metrics.vacancyFraction,
          substitutionFraction: metrics.substitutionFraction,
          interstitialFraction: metrics.interstitialFraction,
          siteMixingFraction: metrics.siteMixingFraction,
          amorphousFraction: metrics.amorphousFraction,
        },
        mlFeatures,
        totalAtoms: atoms.length,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to analyze disorder metrics", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/correlation-engine/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getCorrelationEngineStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch correlation engine stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/correlation-engine/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const analysis = await estimateCorrelationEffects(formula, {});
      res.json({ formula, ...analysis });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to analyze correlations", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/crystal-growth/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getCrystalGrowthStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch crystal growth stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/crystal-growth/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const matClass = formula.includes("H") ? "Hydrides" : "default";
      const result = simulateCrystalGrowth(formula, matClass, {});
      res.json({ formula, ...result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to simulate crystal growth", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/experiment-planner/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getExperimentPlannerStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch experiment planner stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/experiment-planner/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const matClass = formula.includes("H") ? "Hydrides" : formula.includes("C") ? "Carbides" : "default";
      const candidate: ExperimentCandidate = {
        formula,
        predictedTc: 50,
        stability: 0.7,
        synthesisFeasibility: 0.6,
        novelty: 0.5,
        uncertainty: 0.3,
        materialClass: matClass,
        crystalStructure: "cubic",
      };
      const plan = generateExperimentPlan(candidate);
      res.json({ formula, plan });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate experiment plan", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/next-gen-pipeline", writeLimiter, (req, res) => {
    try {
      const { id, goal } = req.body;
      if (!id) return res.status(400).json({ error: "Pipeline id required" });
      const pipeline = createNextGenPipeline(id, goal);
      res.json({ id, status: pipeline.status, iteration: pipeline.iteration });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to create pipeline", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/next-gen-pipeline/:id/iterate", writeLimiter, (req, res) => {
    try {
      const result = runNextGenIteration(req.params.id);
      if (!result) return res.status(404).json({ error: "Pipeline not found or completed" });
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run iteration", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/next-gen-pipeline/stats", (_req, res) => {
    try {
      res.json(getNextGenStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch pipeline stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/next-gen-pipeline/:id", (req, res) => {
    try {
      const stats = getNextGenPipelineDetail(req.params.id);
      if (!stats) return res.status(404).json({ error: "Pipeline not found" });
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch pipeline detail", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/next-gen-pipeline/:id/pause", writeLimiter, (req, res) => {
    try {
      const success = pauseNextGenPipeline(req.params.id);
      res.json({ success });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to pause pipeline", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/next-gen-pipeline/:id/resume", writeLimiter, (req, res) => {
    try {
      const success = resumeNextGenPipeline(req.params.id);
      res.json({ success });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to resume pipeline", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/self-improving-lab", writeLimiter, (req, res) => {
    try {
      const { id, targetTc, maxPressure, maxIterations } = req.body;
      if (!id) return res.status(400).json({ error: "Lab id required" });
      const lab = createSelfImprovingLab(id, targetTc ?? 293, maxPressure ?? 50, maxIterations ?? 500);
      res.json({ id, status: lab.status, iteration: lab.iteration });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to create lab", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/self-improving-lab/:id/iterate", writeLimiter, (req, res) => {
    try {
      const result = runSelfImprovingIteration(req.params.id);
      if (!result) return res.status(404).json({ error: "Lab not found or completed" });
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run lab iteration", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/self-improving-lab/stats", (_req, res) => {
    try {
      res.json(getSelfImprovingLabOverview());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch lab stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/self-improving-lab/sync-knowledge", writeLimiter, (_req, res) => {
    try {
      res.json(syncGlobalKnowledgeBase());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to sync knowledge base", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/self-improving-lab/global-knowledge", (_req, res) => {
    try {
      res.json(getGlobalKnowledgeBase());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch global knowledge", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/self-improving-lab/:id", (req, res) => {
    try {
      const stats = getSelfImprovingLabDetail(req.params.id);
      if (!stats) return res.status(404).json({ error: "Lab not found" });
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch lab detail", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/self-improving-lab/:id/pause", writeLimiter, (req, res) => {
    try {
      const success = pauseSelfImprovingLab(req.params.id);
      res.json({ success });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to pause lab", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/self-improving-lab/:id/resume", writeLimiter, (req, res) => {
    try {
      const success = resumeSelfImprovingLab(req.params.id);
      res.json({ success });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to resume lab", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/design-representations/stats", (_req, res) => {
    try {
      res.json(getDesignRepresentationStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get design representation stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/program/generate", writeLimiter, (req, res) => {
    try {
      const { strategyType, elementPool, generation } = req.body;
      const program = generateDesignProgram(
        strategyType || "hydride-cage-optimizer",
        elementPool || ["La", "Y", "H", "Nb"],
        generation || 0,
      );
      const execution = executeDesignProgram(program);
      res.json({ program, execution });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate program", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/program/mutate", writeLimiter, (req, res) => {
    try {
      const { program, elementPool } = req.body;
      if (!program) return res.status(400).json({ error: "program is required" });
      const mutated = mutateDesignProgram(program, elementPool || []);
      const execution = executeDesignProgram(mutated);
      res.json({ program: mutated, execution });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to mutate program", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/program/crossover", writeLimiter, (req, res) => {
    try {
      const { parent1, parent2 } = req.body;
      if (!parent1 || !parent2) return res.status(400).json({ error: "parent1 and parent2 are required" });
      const child = crossoverPrograms(parent1, parent2);
      const execution = executeDesignProgram(child);
      res.json({ program: child, execution });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to crossover programs", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/program/execute", writeLimiter, (req, res) => {
    try {
      const { program } = req.body;
      if (!program) return res.status(400).json({ error: "program is required" });
      const execution = executeDesignProgram(program);
      res.json(execution);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to execute program", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/graph/generate", writeLimiter, (req, res) => {
    try {
      const { strategyType, elementPool, generation } = req.body;
      const graph = generateDesignGraph(
        strategyType || "hydride-cage-optimizer",
        elementPool || ["La", "Y", "H", "Nb"],
        generation || 0,
      );
      const analysis = analyzeGraph(graph);
      res.json({ graph, analysis });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate graph", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/graph/mutate", writeLimiter, (req, res) => {
    try {
      const { graph, elementPool } = req.body;
      if (!graph) return res.status(400).json({ error: "graph is required" });
      const mutated = mutateDesignGraph(graph, elementPool || []);
      const analysis = analyzeGraph(mutated);
      res.json({ graph: mutated, analysis });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to mutate graph", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/graph/analyze", writeLimiter, (req, res) => {
    try {
      const { graph } = req.body;
      if (!graph) return res.status(400).json({ error: "graph is required" });
      const analysis = analyzeGraph(graph);
      res.json(analysis);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to analyze graph", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/convert/program-to-graph", writeLimiter, (req, res) => {
    try {
      const { program } = req.body;
      if (!program) return res.status(400).json({ error: "program is required" });
      const graph = programToGraph(program);
      const analysis = analyzeGraph(graph);
      res.json({ graph, analysis });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to convert program to graph", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/design-representations/convert/graph-to-program", writeLimiter, (req, res) => {
    try {
      const { graph } = req.body;
      if (!graph) return res.status(400).json({ error: "graph is required" });
      const program = graphToProgram(graph);
      const execution = executeDesignProgram(program);
      res.json({ program, execution });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to convert graph to program", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/symbolic-discovery/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getSymbolicDiscoveryStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get discovery stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/symbolic-discovery/theories", generalLimiter, (_req, res) => {
    try {
      const theories = getTheoryDatabase();
      res.json(theories.map(t => ({
        id: t.id,
        equation: t.equation,
        simplified: t.simplified,
        target: t.target,
        theoryScore: Math.round(t.theoryScore * 1000) / 1000,
        accuracy: Math.round(t.r2 * 1000) / 1000,
        mae: Math.round(t.mae * 100) / 100,
        complexity: t.complexity,
        simplicity: Math.round(t.simplicity * 1000) / 1000,
        generalization: Math.round(t.generalization * 1000) / 1000,
        physicsCompliance: Math.round(t.physicsCompliance * 1000) / 1000,
        novelty: Math.round(t.novelty * 1000) / 1000,
        dimensionallyValid: t.dimensionallyValid,
        variables: t.variables,
        applicableFamilies: t.applicableFamilies,
        crossScaleValidation: t.crossScaleValidation,
        featureImportance: t.featureImportance,
        discoveredAt: t.discoveredAt,
      })));
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get theories", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/symbolic-discovery/feature-library", generalLimiter, (_req, res) => {
    try {
      const library = getFeatureLibrary();
      res.json(library.map(t => ({
        name: t.name,
        expression: t.expression,
        variables: t.variables,
        category: t.category,
        physicsInspired: t.physicsInspired,
      })));
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get feature library", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/symbolic-discovery/run", writeLimiter, async (req, res) => {
    try {
      const config: Partial<SymbolicDiscoveryConfig> = req.body.config ?? {};
      const dataset = await generateSyntheticDataset(50);
      const discovered = runSymbolicPhysicsDiscovery(dataset, config);
      const feedback = generateDiscoveryFeedback(discovered);
      res.json({
        theoriesDiscovered: discovered.length,
        topTheories: discovered.slice(0, 10).map(t => ({
          id: t.id,
          equation: t.equation,
          simplified: t.simplified,
          theoryScore: Math.round(t.theoryScore * 1000) / 1000,
          r2: Math.round(t.r2 * 1000) / 1000,
          mae: Math.round(t.mae * 100) / 100,
          complexity: t.complexity,
          generalization: Math.round(t.generalization * 1000) / 1000,
          physicsCompliance: Math.round(t.physicsCompliance * 1000) / 1000,
          novelty: Math.round(t.novelty * 1000) / 1000,
          dimensionallyValid: t.dimensionallyValid,
          variables: t.variables,
          applicableFamilies: t.applicableFamilies,
        })),
        feedback,
        datasetSize: dataset.length,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run discovery", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-discovery-dataset", generalLimiter, async (_req, res) => {
    try {
      const dataset = await generateSyntheticDataset(50);
      res.json({
        records: dataset,
        featureCount: PHYSICS_VARIABLES.length,
        variables: PHYSICS_VARIABLES,
        recordCount: dataset.length,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate dataset", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/physics-discovery-dataset/record", writeLimiter, async (req, res) => {
    try {
      const { formula } = req.body;
      if (!formula) return res.status(400).json({ error: "formula is required" });
      const record = await buildPhysicsDiscoveryRecord(formula);
      res.json(record);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to build record", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/symbolic-discovery/validate-constraints", writeLimiter, (req, res) => {
    try {
      const { record } = req.body;
      if (!record) return res.status(400).json({ error: "record is required" });
      const checks = validatePhysicsConstraints(record);
      res.json({ checks, allSatisfied: checks.every(c => c.satisfied) });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to validate constraints", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/symbolic-discovery/feedback", generalLimiter, (_req, res) => {
    try {
      const theories = getTheoryDatabase();
      const feedback = generateDiscoveryFeedback(theories);
      res.json(feedback);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate feedback", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getCausalDiscoveryStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get causal stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/variables", generalLimiter, (_req, res) => {
    try {
      res.json(getCausalVariables());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get variables", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/ontology", generalLimiter, (_req, res) => {
    try {
      res.json(getOntology());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get ontology", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/graph", generalLimiter, (_req, res) => {
    try {
      const graph = getLatestGraph();
      res.json(graph ?? { nodes: [], edges: [], discoveredAt: 0, method: "none", datasetSize: 0 });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get graph", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/hypotheses", generalLimiter, (_req, res) => {
    try {
      res.json(getDiscoveredHypotheses());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get hypotheses", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/rules", generalLimiter, (_req, res) => {
    try {
      res.json(getCausalRules());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get rules", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/causal-discovery/guidance", generalLimiter, (_req, res) => {
    try {
      res.json(getDesignGuidance());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get design guidance", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/theory-report", generalLimiter, (_req, res) => {
    try {
      const causalStats = getCausalDiscoveryStats();
      const symbolicStats = getSymbolicDiscoveryStats();
      const graph = getLatestGraph();
      const hypotheses = getDiscoveredHypotheses();
      const rules = getCausalRules();
      const guidance = getDesignGuidance();

      const categoryDiscoveryProgress: Record<string, { variables: number; rulesDiscovered: number; coveragePercent: number }> = {};
      const ruleChains = rules.flatMap(r => r.causalChain ?? []);
      for (const [cat, count] of Object.entries(causalStats.variableCategories)) {
        const rulesInCat = ruleChains.filter(v => {
          const meta = getCausalVariables().find(cv => cv.name === v);
          return meta?.category === cat;
        }).length;
        categoryDiscoveryProgress[cat] = {
          variables: count,
          rulesDiscovered: rulesInCat,
          coveragePercent: count > 0 ? Math.round((Math.min(rulesInCat, count) / count) * 100) : 0,
        };
      }

      res.json({
        causal: {
          totalRuns: causalStats.totalRuns,
          graphSize: causalStats.latestGraphSize,
          hypothesesCount: causalStats.totalHypotheses,
          rulesCount: causalStats.totalCausalRules,
          topHypotheses: causalStats.topHypotheses.slice(0, 5),
          topEdges: causalStats.topEdges.slice(0, 5),
          variableCategories: causalStats.variableCategories,
          categoryDiscoveryProgress,
        },
        symbolic: {
          totalTheories: symbolicStats.totalTheories,
          bestScore: symbolicStats.bestTheoryScore,
          averageScore: symbolicStats.averageTheoryScore,
          familyCoverage: symbolicStats.familyCoverage,
          featureLibrarySize: symbolicStats.featureLibrarySize,
        },
        guidance: guidance.slice(0, 10),
        summary: {
          totalDiscoveryRuns: causalStats.totalRuns,
          totalTheories: symbolicStats.totalTheories,
          totalHypotheses: causalStats.totalHypotheses,
          totalRules: causalStats.totalCausalRules,
          graphNodes: graph?.nodes.length ?? 0,
          graphEdges: graph?.edges.length ?? 0,
          topRecommendation: guidance.length > 0
            ? guidance[0].recommendation
            : "Run causal discovery to generate recommendations",
        },
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate theory report", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/causal-discovery/run", writeLimiter, async (req, res) => {
    try {
      const count = req.body?.datasetSize ?? 60;
      const dataset = await generateCausalDataset(Math.min(count, 100));
      const result = runCausalDiscovery(dataset);
      res.json({
        graphNodes: result.graph.nodes.length,
        graphEdges: result.graph.edges.length,
        hypothesesDiscovered: result.hypotheses.length,
        rulesExtracted: result.rules.length,
        graph: result.graph,
        hypotheses: result.hypotheses.slice(0, 10),
        rules: result.rules.slice(0, 15),
        crossFamilyValidation: result.crossFamilyValidation,
        designGuidance: result.designGuidance,
        pressureComparison: result.pressureComparison,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Causal discovery failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/causal-discovery/intervene", writeLimiter, async (req, res) => {
    try {
      const { formula, variable, newValue } = req.body;
      if (!formula || !variable || newValue === undefined) {
        return res.status(400).json({ error: "Missing formula, variable, or newValue" });
      }
      const graph = getLatestGraph();
      if (!graph) {
        return res.status(400).json({ error: "No causal graph discovered yet. Run discovery first." });
      }
      const record = await buildCausalDataRecord(formula);
      const result = simulateIntervention(record, variable, newValue, graph);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Intervention failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/causal-discovery/counterfactual", writeLimiter, async (req, res) => {
    try {
      const { formula, variable, modificationPercent } = req.body;
      if (!formula || !variable || modificationPercent === undefined) {
        return res.status(400).json({ error: "Missing formula, variable, or modificationPercent" });
      }
      const graph = getLatestGraph();
      if (!graph) {
        return res.status(400).json({ error: "No causal graph discovered yet. Run discovery first." });
      }
      const record = await buildCausalDataRecord(formula);
      const result = runCounterfactual(record, variable, modificationPercent, graph);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Counterfactual failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/causal-discovery/dataset", writeLimiter, async (req, res) => {
    try {
      const count = req.body?.count ?? 60;
      const dataset = await generateCausalDataset(Math.min(count, 100));
      res.json({ count: dataset.length, records: dataset.slice(0, 20) });
    } catch (e: any) {
      res.status(500).json({ error: "Dataset generation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-curves/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getPressureCurveStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Pressure curve stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-curves/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const curve = await predictPressureCurve(formula);
      const optimal = await findOptimalPressure(formula);
      const sensitivity = await pressureSensitivity(formula);
      res.json({ curve, optimal, sensitivity });
    } catch (e: any) {
      res.status(500).json({ error: "Pressure curve failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/phase-transitions/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getPhaseTransitionStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Phase transition stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/phase-transitions/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const transitions = await detectPhaseTransitions(formula);
      res.json({ formula, transitions, count: transitions.length });
    } catch (e: any) {
      res.status(500).json({ error: "Phase transition detection failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-exploration/stats", generalLimiter, (_req, res) => {
    try {
      const explorationStats = getPressureExplorationStats();
      const coverageStats = getPressureCoverageStats();
      res.json({ exploration: explorationStats, coverage: coverageStats });
    } catch (e: any) {
      res.status(500).json({ error: "Pressure exploration stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/enthalpy/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getEnthalpyStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Enthalpy stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/enthalpy/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const curve = await computeEnthalpyPressureCurve(formula);
      const stabilityWindow = await findStabilityPressureWindow(formula);
      res.json({ formula, curve, stabilityWindow });
    } catch (e: any) {
      res.status(500).json({ error: "Enthalpy computation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-profiles/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getPressurePropertyMapStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Pressure profile stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-profiles/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const profile = await buildPressureResponseProfile(formula);
      res.json(profile);
    } catch (e: any) {
      res.status(500).json({ error: "Pressure profile failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-profiles/:formula/interpolate", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string ?? "0");
      if (isNaN(pressure) || pressure < 0 || pressure > 400) {
        return res.status(400).json({ error: "Invalid pressure parameter (0-400 GPa)" });
      }
      const result = await interpolateAtPressure(formula, pressure);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Interpolation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/bayesian-pressure/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getBayesianPressureStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Bayesian pressure stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/bayesian-pressure/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = await optimizePressureForFormula(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Bayesian pressure optimization failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-screening/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getPressureClusterStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Pressure cluster stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-screening/sample", generalLimiter, (req, res) => {
    try {
      const count = parseInt(req.query.count as string ?? "10", 10);
      const pressures = samplePressureFromClusters(Math.min(50, Math.max(1, count)));
      res.json({ pressures, bias: getPressureClusterStats().explorationBias });
    } catch (e: any) {
      res.status(500).json({ error: "Pressure sampling failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-screening/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = req.query.pressure ? parseFloat(req.query.pressure as string) : undefined;

      if (pressure !== undefined) {
        const result = await fastPressureScreen(formula, pressure);
        return res.json(result);
      }

      const best = await findBestScreeningPressure(formula);
      const cluster = assignPressureCluster(best.bestPressure);
      res.json({ ...best, cluster: cluster.id, clusterLabel: cluster.label });
    } catch (e: any) {
      res.status(500).json({ error: "Pressure screening failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/pressure-screening/:formula/sweep", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const results = await batchPressureScreen(formula);
      res.json({ formula, results, count: results.length });
    } catch (e: any) {
      res.status(500).json({ error: "Pressure sweep failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/eliashberg/stats", generalLimiter, async (_req, res) => {
    try {
      const mod = await getEliashbergModule();
      res.json(mod.getEliashbergPipelineStats());
    } catch (e: any) {
      res.status(500).json({ error: "Eliashberg stats failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/eliashberg/:formula/alpha2f", generalLimiter, async (req, res) => {
    try {
      const mod = await getEliashbergModule();
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const alpha2F = mod.getAlpha2FOnly(formula, pressure);
      res.json({ formula, pressureGpa: pressure, ...alpha2F });
    } catch (e: any) {
      res.status(500).json({ error: "Alpha2F computation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/eliashberg/:formula/dfpt", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const existingJobs = await storage.getDftJobsByFormula(formula);
      const existing = existingJobs.length > 0 ? existingJobs[0] : null;
      if (existing && existing.status === "completed" && existing.outputData) {
        const parsed = typeof existing.outputData === "string" ? JSON.parse(existing.outputData) : existing.outputData;
        const mod = await getEliashbergModule();
        const surrogateResult = mod.runEliashbergPipeline(formula, 0);
        res.json({
          status: "completed",
          jobId: existing.id,
          formula,
          dftResult: parsed,
          eliashbergSurrogate: surrogateResult,
          message: "DFPT result available from previous calculation",
        });
        return;
      }
      if (existing && (existing.status === "pending" || existing.status === "running" || existing.status === "queued")) {
        res.json({
          status: existing.status,
          jobId: existing.id,
          formula,
          message: `DFPT job already ${existing.status}`,
        });
        return;
      }
      const job = await submitDFTJob(formula, null, 80, "phonon");
      if (!job) {
        res.status(400).json({ error: "Could not submit DFPT job", detail: `Formula ${formula} may be blocked or invalid` });
        return;
      }
      res.json({
        status: "submitted",
        jobId: job.id,
        formula,
        message: "DFPT phonon calculation queued for high-fidelity Eliashberg analysis",
      });
    } catch (e: any) {
      res.status(500).json({ error: "DFPT trigger failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/eliashberg/:formula", generalLimiter, async (req, res) => {
    try {
      const mod = await getEliashbergModule();
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const result = mod.runEliashbergPipeline(formula, pressure);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Eliashberg pipeline failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/quantum-engine/stats", async (_req, res) => {
    try {
      const stats = getQuantumEngineStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch quantum engine stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/quantum-engine/dataset", async (_req, res) => {
    try {
      const dataset = await getQuantumEngineDataset();
      res.json({ dataset, total: dataset.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch quantum engine dataset", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/quantum-engine/recent", async (req, res) => {
    try {
      const rawLimit = Number(req.query.limit);
      const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? Math.min(rawLimit, 200) : 20;
      const results = await getRecentQuantumEngineResults(limit);
      res.json({ results, total: results.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch recent quantum engine results", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-results/stats", async (_req, res) => {
    try {
      const stats = getPhysicsStoreStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch physics results stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/quantum-engine/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const result = await runQuantumEnginePipeline(formula, pressure);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Quantum engine pipeline failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/lambda-regressor/stats", async (_req, res) => {
    try {
      const stats = getLambdaRegressorStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch lambda regressor stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/lambda-regressor/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const prediction = predictLambda(formula, pressure);
      res.json(prediction);
    } catch (e: any) {
      res.status(500).json({ error: "Lambda prediction failed", detail: e.message?.slice(0, 200) });
    }
  });

  setTimeout(() => {
    try { startPoolInit(); } catch {}
  }, 20000);

  setTimeout(async () => {
    try { await initLambdaRegressor(); } catch {}
  }, 600000);

  setTimeout(() => {
    try { initPhononSurrogate(); } catch {}
  }, 660000);

  setTimeout(() => {
    try { initStructurePredictorML(); } catch {}
  }, 720000);

  app.get("/api/phonon-surrogate/stats", async (_req, res) => {
    try {
      const stats = getPhononSurrogateStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch phonon surrogate stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/phonon-surrogate/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const prediction = predictPhononProperties(formula, pressure);
      res.json(prediction);
    } catch (e: any) {
      res.status(500).json({ error: "Phonon surrogate prediction failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-api/stats", async (_req, res) => {
    try {
      res.json(getPhysicsApiStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch physics API stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-api/relaxation/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = await runRelaxation(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Relaxation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-api/phonons/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = await computePhonons(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Phonon computation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-api/eph/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const result = await computeEph(formula, pressure);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Electron-phonon coupling computation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/physics-api/tc/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = parseFloat(req.query.pressure as string) || 0;
      const result = await computeTc(formula, pressure);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Tc computation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/crystal-dataset/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getCrystalDatasetStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch crystal dataset stats" });
    }
  });

  app.get("/api/crystal-dataset/entry/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const entry = getCrystalDatasetEntry(formula);
      if (!entry) return res.status(404).json({ error: "Entry not found" });
      res.json(entry);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch crystal dataset entry" });
    }
  });

  app.get("/api/prototype-enum/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const results = enumeratePrototypesForFormula(formula);
      res.json({ formula, prototypes: results, count: results.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to enumerate prototypes" });
    }
  });

  app.get("/api/crystal-dataset/prototype/:prototype", generalLimiter, async (req, res) => {
    try {
      const prototype = decodeURIComponent(req.params.prototype);
      const entries = getCrystalDatasetByPrototype(prototype);
      res.json({ entries, count: entries.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch entries by prototype" });
    }
  });

  app.get("/api/crystal-dataset/system/:system", generalLimiter, async (req, res) => {
    try {
      const system = decodeURIComponent(req.params.system);
      const entries = getCrystalDatasetBySystem(system);
      res.json({ entries, count: entries.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch entries by crystal system" });
    }
  });

  app.get("/api/structure-ml/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const prediction = mlPredictStructure(formula);
      res.json(prediction);
    } catch (e) {
      res.status(500).json({ error: "Failed to predict structure" });
    }
  });

  app.get("/api/structure-ml/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getStructureMLStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch structure ML stats" });
    }
  });

  app.get("/api/crystal-diffusion-model/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getDiffusionModelStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch diffusion model stats" });
    }
  });

  app.get("/api/crystal-diffusion-model/sample", generalLimiter, async (req, res) => {
    try {
      const count = Math.min(Math.max(1, Number(req.query.count) || 5), 50);
      const system = req.query.system ? String(req.query.system) : undefined;
      const elementsStr = req.query.elements ? String(req.query.elements) : undefined;
      const elements = elementsStr ? elementsStr.split(",").map(e => e.trim()).filter(Boolean) : undefined;
      const conditions = (system || elements) ? { crystalSystem: system, elements } : undefined;
      const samples = diffusionSampleStructures(count, conditions);
      res.json({ samples, count: samples.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to sample diffusion structures" });
    }
  });

  setTimeout(async () => {
    try {
      await initDiffusionModel();
    } catch (e) {
      console.error("[CrystalDiffusion] Init failed:", e);
    }
  }, 70000);  // T+70s — well before VAE at T+130s

  setTimeout(async () => {
    try {
      await initCrystalVAE();
    } catch (e) {
      console.error("[CrystalVAE] Init failed:", e);
    }
  }, 130000);  // T+130s — 60s after diffusion model

  setTimeout(async () => {
    try {
      console.log("[Benchmark] Running reference compound predictions on startup...");
      await runReferenceBenchmark();
      console.log("[Benchmark] Reference benchmark complete.");
    } catch (e: any) {
      console.error("[Benchmark] Startup benchmark failed:", e.message);
    }
  }, 250000);  // T+250s — 60s after GNN pre-warm at T+190s

  app.get("/api/crystal-vae/stats", generalLimiter, async (_req, res) => {
    try {
      res.json(getCrystalVAEStats());
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch crystal VAE stats" });
    }
  });

  app.get("/api/crystal-vae/generate", generalLimiter, async (req, res) => {
    try {
      const count = Math.min(Math.max(1, Number(req.query.count) || 1), 50);
      const targetSystem = req.query.targetSystem as string | undefined;
      const results = [];
      for (let i = 0; i < count; i++) {
        const crystal = generateNovelCrystal(targetSystem);
        if (crystal) results.push(crystal);
      }
      res.json({ generated: results, count: results.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to generate crystals" });
    }
  });

  app.get("/api/crystal-vae/interpolate", generalLimiter, async (req, res) => {
    try {
      const formula1 = req.query.formula1 as string;
      const formula2 = req.query.formula2 as string;
      const alpha = Number(req.query.alpha) || 0.5;
      if (!formula1 || !formula2) {
        return res.status(400).json({ error: "formula1 and formula2 are required" });
      }
      const result = interpolateCrystals(formula1, formula2, alpha);
      if (!result) {
        return res.status(404).json({ error: "Could not interpolate - formulas may not be in training set" });
      }
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to interpolate crystals" });
    }
  });

  app.get("/api/crystal-vae/encode/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = encodeFormula(formula);
      if (!result) {
        return res.status(404).json({ error: "Formula not found in training set or VAE not trained" });
      }
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to encode formula" });
    }
  });

  app.get("/api/generative-crystals/generate", generalLimiter, async (req, res) => {
    try {
      const count = Math.min(Math.max(1, Number(req.query.count) || 10), 100);
      const strategy = (req.query.strategy as GenerationStrategy) || "hybrid";
      const targetSystem = req.query.targetSystem ? String(req.query.targetSystem) : undefined;
      const elementsStr = req.query.elements ? String(req.query.elements) : undefined;
      const elements = elementsStr ? elementsStr.split(",").map(e => e.trim()).filter(Boolean) : undefined;
      const candidates = generativeCrystalGenerate(count, strategy, { targetSystem, elements });
      res.json({ candidates, count: candidates.length, strategy });
    } catch (e) {
      res.status(500).json({ error: "Failed to generate crystal candidates" });
    }
  });

  app.get("/api/generative-crystals/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getGenerativeEngineStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch generative engine stats" });
    }
  });

  app.get("/api/structure-failures/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getFailureDBStats();
      const patterns = getFailurePatterns();
      res.json({ ...stats, patterns });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch structure failure stats" });
    }
  });

  app.get("/api/structure-failures/check/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const system = req.query.system ? String(req.query.system) : undefined;
      const avoid = shouldAvoidStructure(formula, undefined, system);
      const featureVector = getFailureFeatureVector(formula);
      res.json({ formula, shouldAvoid: avoid, featureVector });
    } catch (e) {
      res.status(500).json({ error: "Failed to check structure" });
    }
  });

  app.post("/api/structure-failures/record", writeLimiter, async (req, res) => {
    try {
      const { formula, failureReason, source, lattice, crystalSystem, spacegroup,
              formationEnergy, imaginaryModeCount, lowestPhononFreq, bandGap,
              stage, details } = req.body;
      if (!formula || !failureReason || !source) {
        return res.status(400).json({ error: "Missing required fields: formula, failureReason, source" });
      }
      const validReasons: FailureReason[] = [
        "unstable_phonons", "structure_collapse", "high_formation_energy",
        "non_metallic", "scf_divergence", "geometry_rejected",
      ];
      const validSources: FailureSource[] = ["dft", "xtb", "pipeline", "phonon_surrogate"];
      if (!validReasons.includes(failureReason)) {
        return res.status(400).json({ error: `Invalid failureReason. Valid: ${validReasons.join(", ")}` });
      }
      if (!validSources.includes(source)) {
        return res.status(400).json({ error: `Invalid source. Valid: ${validSources.join(", ")}` });
      }
      recordStructureFailure({
        formula,
        failureReason,
        source,
        failedAt: Date.now(),
        lattice,
        crystalSystem,
        spacegroup,
        formationEnergy,
        imaginaryModeCount,
        lowestPhononFreq,
        bandGap,
        stage,
        details,
      });
      res.json({ success: true });
    } catch (e) {
      res.status(500).json({ error: "Failed to record structure failure" });
    }
  });

  app.get("/api/hybrid-generator/generate", generalLimiter, async (req, res) => {
    try {
      const count = Math.min(Math.max(1, Number(req.query.count) || 10), 100);
      const mutationRate = req.query.mutationRate ? Number(req.query.mutationRate) : undefined;
      const mlWeight = req.query.mlWeight ? Number(req.query.mlWeight) : undefined;
      const targetPressure = req.query.targetPressure ? Number(req.query.targetPressure) : undefined;
      const targetSystem = req.query.targetSystem ? String(req.query.targetSystem) : undefined;
      const candidates = await generateHybridCandidates(count, {
        mutationRate,
        mlWeight,
        targetPressure,
        targetSystem,
      });
      res.json({ candidates, count: candidates.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to generate hybrid candidates" });
    }
  });

  app.get("/api/hybrid-generator/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getHybridGeneratorStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch hybrid generator stats" });
    }
  });

  app.get("/api/structure-learning/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getStructureLearningStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch structure learning stats" });
    }
  });

  app.post("/api/structure-learning/trigger", writeLimiter, async (req, res) => {
    try {
      const batchSize = Math.min(Math.max(1, Number(req.body.batchSize) || 10), 50);
      const targetPressure = req.body.targetPressure ? Number(req.body.targetPressure) : undefined;
      const targetSystem = req.body.targetSystem ? String(req.body.targetSystem) : undefined;
      const result = await runStructureLearningCycle(batchSize, targetPressure, targetSystem);
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to trigger structure learning cycle" });
    }
  });

  setTimeout(() => initFingerprintDB(), 3000);

  // Pool init is deferred to engine start to avoid blocking server startup

  app.get("/api/structure-novelty/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getNoveltyStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch novelty stats" });
    }
  });

  app.get("/api/structure-novelty/score/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = req.params.formula;
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const result = scoreFormulaNovelty(formula);
      res.json({ formula, ...result });
    } catch (e) {
      res.status(500).json({ error: "Failed to compute novelty score" });
    }
  });

  app.get("/api/relaxation/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getRelaxationStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch relaxation stats" });
    }
  });

  app.get("/api/relaxation/patterns", generalLimiter, async (_req, res) => {
    try {
      const patterns = getRelaxationPatterns();
      res.json(patterns);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch relaxation patterns" });
    }
  });

  app.get("/api/relaxation/entry/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = req.params.formula;
      const entries = getRelaxationEntry(formula);
      res.json({ formula, entries, count: entries.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch relaxation entry" });
    }
  });

  app.get("/api/relaxation/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = req.params.formula;
      const a = Number(req.query.a) || 4.0;
      const b = Number(req.query.b) || a;
      const c = Number(req.query.c) || a;
      const alpha = Number(req.query.alpha) || 90;
      const beta = Number(req.query.beta) || 90;
      const gamma = Number(req.query.gamma) || 90;
      const prediction = predictRelaxationMagnitude(formula, { a, b, c, alpha, beta, gamma });
      res.json({ formula, ...prediction });
    } catch (e) {
      res.status(500).json({ error: "Failed to predict relaxation magnitude" });
    }
  });

  app.get("/api/distortion/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getDistortionStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch distortion stats" });
    }
  });

  app.get("/api/distortion/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = getDistortionForFormula(formula);
      if (!result) {
        return res.status(404).json({ error: "No distortion data for formula" });
      }
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch distortion data" });
    }
  });

  app.get("/api/distortion/classify/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = classifyFormulaDistortion(formula);
      if (!result) {
        return res.status(404).json({ error: "No distortion data for formula to classify" });
      }
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to classify distortion" });
    }
  });

  app.get("/api/distortion/classifier/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getClassifierStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch classifier stats" });
    }
  });

  app.get("/api/energy-landscape/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getEnergyLandscapeStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch energy landscape stats" });
    }
  });

  app.post("/api/energy-landscape/explore/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const result = await runLandscapeExploration(formula);
      if (!result) {
        return res.status(404).json({ error: "Could not run landscape exploration (DFT unavailable or optimization failed)" });
      }
      res.json(result);
    } catch (e) {
      res.status(500).json({ error: "Failed to run landscape exploration" });
    }
  });

  app.get("/api/stability-predictor/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const prediction = await predictStabilityScreen(formula);
      res.json({ formula, ...prediction });
    } catch (e) {
      res.status(500).json({ error: "Failed to predict stability" });
    }
  });

  app.get("/api/stability-predictor/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getStabilityPredictorStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch stability predictor stats" });
    }
  });

  app.get("/api/structure-rewards/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getRewardSystemStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch structure reward stats" });
    }
  });

  app.get("/api/structure-rewards/best", generalLimiter, async (req, res) => {
    try {
      const n = Math.min(Math.max(1, Number(req.query.n) || 10), 100);
      const best = getBestMotifs(n);
      res.json({ motifs: best, count: best.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch best motifs" });
    }
  });

  app.get("/api/structure-rewards/motif/:prototype", generalLimiter, async (req, res) => {
    try {
      const prototype = decodeURIComponent(req.params.prototype);
      const system = req.query.system as string | undefined;
      const spacegroup = req.query.spacegroup as string | undefined;
      const reward = getStructureReward(prototype, system, spacegroup ?? null);
      if (!reward) {
        return res.status(404).json({ error: "Motif not found" });
      }
      res.json(reward);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch motif reward" });
    }
  });

  app.get("/api/pressure-structure/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const pressure = Number(req.query.pressure) || 0;
      const prediction = predictStructureAtPressure(formula, pressure);
      res.json({ formula, pressureGPa: pressure, ...prediction });
    } catch (e) {
      res.status(500).json({ error: "Failed to predict pressure structure" });
    }
  });

  app.get("/api/pressure-structure/phase-map/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const phaseMap = getPressurePhaseMap(formula);
      res.json({ formula, phases: phaseMap, totalSteps: phaseMap.length });
    } catch (e) {
      res.status(500).json({ error: "Failed to compute pressure phase map" });
    }
  });

  app.get("/api/pressure-structure/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getPressureStructureStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch pressure structure stats" });
    }
  });

  app.get("/api/structure-embedding/embed/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const embedding = computeStructureEmbedding(formula);
      const clusterIdx = getClusterAssignment(embedding);
      const novelty = computeClusterNovelty(embedding);
      const uncertainty = estimateStructureUncertainty(embedding);
      res.json({ formula, embedding, cluster: clusterIdx, novelty, uncertainty });
    } catch (e) {
      res.status(500).json({ error: "Failed to compute structure embedding" });
    }
  });

  app.get("/api/structure-embedding/clusters", generalLimiter, async (_req, res) => {
    try {
      const clusters = getEmbeddingClusters();
      if (!clusters) {
        return res.json({ clusters: [], k: 0, inertia: 0 });
      }
      res.json(clusters);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch structure clusters" });
    }
  });

  app.get("/api/structure-embedding/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getEmbeddingStats();
      res.json(stats);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch embedding stats" });
    }
  });

  app.get("/api/tight-binding/properties/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const pressure = Number(req.query.pressure) || 0;
      const props = computeTBProperties(formula, pressure);
      res.json(props);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute TB properties", detail: e.message });
    }
  });

  app.get("/api/tight-binding/bands/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const nPoints = Math.min(Number(req.query.nPoints) || 30, 100);
      const bands = computeBandStructure(formula, nPoints);
      res.json(bands);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute band structure", detail: e.message });
    }
  });

  app.get("/api/tight-binding/dos/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const nKpoints = Math.min(Number(req.query.nKpoints) || 12, 20);
      const nBins = Math.min(Number(req.query.nBins) || 200, 500);
      const dos = computeDOS(formula, nKpoints, nBins);
      res.json(dos);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute DOS", detail: e.message });
    }
  });

  app.get("/api/tight-binding/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getTBEngineStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch TB engine stats" });
    }
  });

  app.get("/api/tb-surrogate/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length === 0) {
        return res.status(400).json({ error: "Formula is required" });
      }
      const prediction = predictTBSurrogate(formula);
      res.json(prediction);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict TB properties", detail: e.message });
    }
  });

  app.get("/api/tb-surrogate/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getTBSurrogateStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch TB surrogate stats" });
    }
  });

  app.post("/api/tb-surrogate/retrain", writeLimiter, async (_req, res) => {
    try {
      const result = retrainTBSurrogate();
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to retrain TB surrogate", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/report", generalLimiter, async (_req, res) => {
    try {
      const report = await getComprehensiveModelDiagnostics();
      res.json(report);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch model diagnostics", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/health", generalLimiter, async (_req, res) => {
    try {
      const health = await getModelHealthSummary();
      res.json(health);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch model health", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/bias", generalLimiter, async (req, res) => {
    try {
      const model = req.query.model as string | undefined;
      const bias = getPerFamilyBias(model);
      res.json(bias);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch bias analysis", detail: e.message });
    }
  });

  app.post("/api/model-experiments/propose", engineLimiter, async (_req, res) => {
    try {
      const report = await getModelDiagnosticsForLLM();
      const proposals = await proposeModelExperiments(report);
      if (proposals.length > 0) {
        const topProposal = proposals.sort((a, b) => a.priority - b.priority)[0];
        const result = await executeExperiment(topProposal);
        res.json({ proposals, executed: result });
      } else {
        res.json({ proposals: [], executed: null, message: "No experiments proposed by LLM" });
      }
    } catch (e: any) {
      res.status(500).json({ error: "Failed to propose model experiments", detail: e.message });
    }
  });

  app.get("/api/model-experiments/history", generalLimiter, async (_req, res) => {
    try {
      const history = getExperimentHistory();
      res.json(history);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch experiment history", detail: e.message });
    }
  });

  app.get("/api/model-experiments/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getExperimentStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch experiment stats", detail: e.message });
    }
  });

  app.get("/api/model-improvement/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = await getModelImprovementStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch model improvement stats", detail: e.message });
    }
  });

  app.post("/api/model-improvement/trigger", engineLimiter, async (req, res) => {
    try {
      const cycle = Number(req.body?.cycle) || 0;
      const dummyEmit: import("./learning/engine").EventEmitter = (type, data) => {
        console.log(`[Model Improvement Trigger] ${type}: ${JSON.stringify(data).slice(0, 200)}`);
      };
      const result = await runModelImprovementCycle(dummyEmit, cycle);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to trigger model improvement", detail: e.message });
    }
  });

  app.get("/api/model-improvement/trends", generalLimiter, async (_req, res) => {
    try {
      const trends = getModelImprovementTrends();
      res.json(trends);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch model improvement trends", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/feature-importance", generalLimiter, async (req, res) => {
    try {
      const topN = parseInt(req.query.topN as string) || 25;
      const features = getFeatureImportanceReport(topN);
      res.json(features);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch feature importance", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/error-analysis", generalLimiter, async (_req, res) => {
    try {
      const analysis = getErrorAnalysis();
      res.json(analysis);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch error analysis", detail: e.message });
    }
  });

  app.get("/api/model-experiments/data-requests", generalLimiter, async (_req, res) => {
    try {
      const requests = getAllDataRequests();
      res.json(requests);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch data requests", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/failed-materials", generalLimiter, async (_req, res) => {
    try {
      const summary = getFailureSummary();
      res.json(summary);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch failure summary", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/failed-materials/llm-report", generalLimiter, async (_req, res) => {
    try {
      const report = getFailedMaterialsForLLM();
      res.json({ report });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate failure report", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/benchmark", generalLimiter, async (_req, res) => {
    try {
      const benchmark = getModelBenchmark();
      res.json(benchmark);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch benchmark", detail: e.message });
    }
  });

  app.get("/api/model-diagnostics/benchmark/llm-report", generalLimiter, async (_req, res) => {
    try {
      const report = getBenchmarkForLLM();
      res.json({ report });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate benchmark report", detail: e.message });
    }
  });

  app.get("/api/model-llm/status", generalLimiter, async (_req, res) => {
    try {
      const status = getModelLLMStatus();
      res.json(status);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch Model LLM status", detail: e.message });
    }
  });

  app.get("/api/model-llm/architecture", generalLimiter, async (_req, res) => {
    try {
      const arch = getCurrentArchitecture();
      res.json(arch ?? { primaryModel: "xgboost", reasoning: "No architecture assessment yet" });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch architecture", detail: e.message });
    }
  });

  app.post("/api/model-llm/architecture/select", engineLimiter, async (_req, res) => {
    try {
      const result = await selectArchitecture();
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to select architecture", detail: e.message });
    }
  });

  app.get("/api/model-llm/features/available", generalLimiter, async (_req, res) => {
    try {
      const features = getAvailableFeatureDefinitions();
      res.json(features);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch available features", detail: e.message });
    }
  });

  app.post("/api/model-llm/features/propose", engineLimiter, async (_req, res) => {
    try {
      const proposals = await proposeNewFeatures();
      for (const p of proposals) {
        enableBuiltinFeature(p.name);
      }
      res.json({ proposals, enabledCount: proposals.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to propose features", detail: e.message });
    }
  });

  app.post("/api/model-llm/features/enable", generalLimiter, async (req, res) => {
    try {
      const { name } = req.body;
      if (!name) return res.status(400).json({ error: "Feature name required" });
      const ok = enableBuiltinFeature(name);
      res.json({ enabled: ok, feature: name });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to enable feature", detail: e.message });
    }
  });

  app.post("/api/model-llm/features/disable", generalLimiter, async (req, res) => {
    try {
      const { name } = req.body;
      if (!name) return res.status(400).json({ error: "Feature name required" });
      const ok = disableCustomFeature(name);
      res.json({ disabled: ok, feature: name });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to disable feature", detail: e.message });
    }
  });

  app.post("/api/model-llm/cycle", engineLimiter, async (req, res) => {
    try {
      const cycle = req.body.cycle ?? 0;
      const report = await runModelLLMCycle(cycle);
      res.json(report);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run Model LLM cycle", detail: e.message });
    }
  });

  app.get("/api/uncertainty/status", generalLimiter, async (_req, res) => {
    try {
      const status = getUncertaintyStatus();
      res.json(status);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch uncertainty status", detail: e.message });
    }
  });

  app.get("/api/uncertainty/report", generalLimiter, async (_req, res) => {
    try {
      const report = getUncertaintyReport();
      res.json(report);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate uncertainty report", detail: e.message });
    }
  });

  app.post("/api/uncertainty/report/full", engineLimiter, async (_req, res) => {
    try {
      const report = await getFullUncertaintyReport();
      res.json(report);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate full uncertainty report", detail: e.message });
    }
  });

  app.get("/api/uncertainty/high-variance", generalLimiter, async (req, res) => {
    try {
      const topN = Math.min(50, Math.max(1, parseInt(String(req.query.topN)) || 20));
      const predictions = getHighUncertaintyPredictions(topN);
      res.json({ predictions, count: predictions.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch high-variance predictions", detail: e.message });
    }
  });

  app.get("/api/uncertainty/by-family", generalLimiter, async (_req, res) => {
    try {
      const byFamily = getVarianceByFamily();
      res.json(byFamily);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch family variance", detail: e.message });
    }
  });

  app.get("/api/unified-ci/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 2 || formula.length > 100) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const result = await computeUnifiedCI(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute unified CI", detail: e.message });
    }
  });

  app.get("/api/physics/tc-uq/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 2 || formula.length > 100) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const pressureGpa = Number(req.query.pressure) || 0;
      const cacheKey = `tc-uq:${formula}:${pressureGpa}`;
      const cached = cache.get(cacheKey);
      if (cached) {
        res.json(cached);
        return;
      }
      const t0 = Date.now();
      const result = computePhysicsTcUQ(formula, pressureGpa);
      console.log(`[tc-uq] computePhysicsTcUQ(${formula}) took ${Date.now() - t0}ms`);
      const response = { formula, pressureGpa, ...result };
      cache.set(cacheKey, response, 60000);
      res.json(response);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute physics Tc UQ", detail: e.message });
    }
  });

  app.post("/api/uncertainty/propose", engineLimiter, async (_req, res) => {
    try {
      const proposals = await proposeUncertaintyImprovements();
      res.json({ proposals, count: proposals.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to propose uncertainty improvements", detail: e.message });
    }
  });

  app.get("/api/calibration/status", generalLimiter, async (_req, res) => {
    try {
      const state = getCalibrationState();
      res.json(state);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get calibration status", detail: e.message });
    }
  });

  app.post("/api/calibration/recalibrate", engineLimiter, async (_req, res) => {
    try {
      const state = await recalibrateFromLedger();
      res.json({ status: "recalibrated", ...state });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to recalibrate", detail: e.message });
    }
  });

  app.get("/api/calibration/conformal-interval/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 2 || formula.length > 100) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const ci = await computeUnifiedCI(formula);
      const conformalResult = {
        formula,
        rawCI95: ci.tcCI95,
        calibratedCI95: ci.calibratedCI95,
        conformalQuantile: ci.conformalQuantile,
        temperatureScale: ci.temperatureScale,
        ece: ci.ece,
        conformalMethod: ci.conformalMethod,
        calibrationDatasetSize: ci.calibrationDatasetSize,
        tcMean: ci.tcMean,
        tcTotalStd: ci.tcTotalStd,
        epistemicStd: ci.tcEpistemicStd,
        aleatoricStd: ci.tcAleatoricStd,
      };
      res.json(conformalResult);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute conformal interval", detail: e.message });
    }
  });

  app.get("/api/calibration/ece", generalLimiter, async (_req, res) => {
    try {
      const ece = getECE();
      const families = getFamilyConformalQuantiles();
      res.json({ ...ece, perFamily: families });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get ECE", detail: e.message });
    }
  });

  app.get("/api/calibration/validate-intervals", generalLimiter, async (_req, res) => {
    try {
      const result = validateIntervalsCoverage();
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to validate intervals", detail: e.message });
    }
  });

  app.get("/api/ood/score/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      if (!formula || formula.length < 2 || formula.length > 100) {
        return res.status(400).json({ error: "Invalid formula" });
      }
      const result = computeOODScore(formula);
      res.json({ formula, ...result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute OOD score", detail: e.message });
    }
  });

  app.get("/api/ood/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getOODStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get OOD stats", detail: e.message });
    }
  });

  app.post("/api/ood/update", engineLimiter, async (_req, res) => {
    try {
      await updateOODModel();
      const stats = getOODStats();
      res.json({ status: "updated", ...stats });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to update OOD model", detail: e.message });
    }
  });

  app.get("/api/uncertainty-quality/history", generalLimiter, async (_req, res) => {
    try {
      const { getCycleImprovementHistory } = await import("./learning/prediction-reality-ledger");
      const history = getCycleImprovementHistory();
      const summary = history.map(h => ({
        cycle: h.cycle,
        timestamp: h.timestamp,
        rmse: h.rmse,
        mae: h.mae,
        r2: h.r2,
        gnn_rmse: h.gnn_rmse,
        xgb_rmse: h.xgb_rmse,
        ciCoverage90: h.ciCoverage90 ?? null,
        ciCoverage95: h.ciCoverage95 ?? null,
        ciCoverage99: h.ciCoverage99 ?? null,
        count: h.count,
      }));
      const latest = summary.length > 0 ? summary[summary.length - 1] : null;
      const ciGoalMet = latest && latest.ciCoverage95 !== null ? Math.abs(latest.ciCoverage95 - 0.95) < 0.05 : null;
      res.json({
        totalCycles: summary.length,
        latestRMSE: latest?.rmse ?? null,
        latestCICoverage95: latest?.ciCoverage95 ?? null,
        ciGoalMet,
        history: summary,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get uncertainty quality history", detail: e.message });
    }
  });

  app.get("/api/retrain-scheduler/status", generalLimiter, async (_req, res) => {
    try {
      const stats = getSchedulerStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch scheduler status", detail: e.message });
    }
  });

  app.post("/api/retrain-scheduler/evaluate", engineLimiter, async (_req, res) => {
    try {
      const decision = await evaluateRetrainNeed();
      res.json(decision);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to evaluate retrain need", detail: e.message });
    }
  });

  app.get("/api/retrain-scheduler/summary", generalLimiter, async (_req, res) => {
    try {
      const summary = getSchedulerForLLM();
      res.json({ summary });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch scheduler summary", detail: e.message });
    }
  });

  app.get("/api/ground-truth/summary", generalLimiter, async (_req, res) => {
    try {
      const summary = getGroundTruthSummary();
      res.json(summary);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch ground-truth summary", detail: e.message });
    }
  });

  app.get("/api/ground-truth/dataset", generalLimiter, async (req, res) => {
    try {
      const offset = Math.max(0, parseInt(String(req.query.offset)) || 0);
      const limit = Math.min(200, Math.max(1, parseInt(String(req.query.limit)) || 50));
      const data = getGroundTruthDatasetSlice(offset, limit);
      const total = getGroundTruthDataset().length;
      res.json({ data, total, offset, limit });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch ground-truth dataset", detail: e.message });
    }
  });

  app.get("/api/ground-truth/formula/:formula", generalLimiter, async (req, res) => {
    try {
      const datapoints = getDatapointsByFormula(req.params.formula);
      res.json({ formula: req.params.formula, datapoints, count: datapoints.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch formula datapoints", detail: e.message });
    }
  });

  app.get("/api/ground-truth/cycles", generalLimiter, async (req, res) => {
    try {
      const n = Math.min(50, Math.max(1, parseInt(String(req.query.n)) || 10));
      const cycles = getRecentBatchCycles(n);
      res.json({ cycles, count: cycles.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch batch cycles", detail: e.message });
    }
  });

  app.get("/api/ground-truth/cycles/all", generalLimiter, async (_req, res) => {
    try {
      const cycles = getBatchCycles();
      res.json({ cycles, count: cycles.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch all batch cycles", detail: e.message });
    }
  });

  app.get("/api/ground-truth/cycle/:cycleNumber", generalLimiter, async (req, res) => {
    try {
      const cycleNum = parseInt(req.params.cycleNumber);
      const datapoints = getDatapointsByCycle(cycleNum);
      res.json({ cycle: cycleNum, datapoints, count: datapoints.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch cycle datapoints", detail: e.message });
    }
  });

  app.get("/api/ground-truth/training-data", generalLimiter, async (_req, res) => {
    try {
      const data = getDatasetForTraining();
      res.json({ ...data, size: data.formulas.length });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch training data", detail: e.message });
    }
  });

  app.get("/api/ground-truth/llm-report", generalLimiter, async (_req, res) => {
    try {
      const report = getGroundTruthForLLM();
      res.json({ report });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate ground-truth LLM report", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/metrics", generalLimiter, async (_req, res) => {
    try {
      const metrics = computeLedgerMetrics();
      res.json(metrics);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute ledger metrics", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/metrics/recent", generalLimiter, async (req, res) => {
    try {
      const window = parseInt(req.query.window as string) || 100;
      const metrics = computeRecentLedgerMetrics(window);
      res.json(metrics);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute recent metrics", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/metrics/by-family", generalLimiter, async (_req, res) => {
    try {
      const families = computeLedgerByFamily();
      res.json({ families });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute family metrics", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/entries", generalLimiter, async (req, res) => {
    try {
      const offset = parseInt(req.query.offset as string) || 0;
      const limit = Math.min(parseInt(req.query.limit as string) || 50, 200);
      const entries = getLedgerSlice(offset, limit);
      res.json({ total: getLedgerSize(), offset, limit, entries });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch ledger entries", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/worst", generalLimiter, async (req, res) => {
    try {
      const topN = Math.min(parseInt(req.query.n as string) || 20, 100);
      const worst = getWorstPredictions(topN);
      res.json({ count: worst.length, entries: worst });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch worst predictions", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/best", generalLimiter, async (req, res) => {
    try {
      const topN = Math.min(parseInt(req.query.n as string) || 20, 100);
      const best = getBestPredictions(topN);
      res.json({ count: best.length, entries: best });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch best predictions", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/overpredictions", generalLimiter, async (req, res) => {
    try {
      const minError = parseFloat(req.query.minError as string) || 30;
      const over = getOverpredictions(minError);
      res.json({ count: over.length, entries: over });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch overpredictions", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/underpredictions", generalLimiter, async (req, res) => {
    try {
      const minError = parseFloat(req.query.minError as string) || 30;
      const under = getUnderpredictions(minError);
      res.json({ count: under.length, entries: under });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch underpredictions", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/llm-report", generalLimiter, async (_req, res) => {
    try {
      const report = getLedgerLLMReport();
      res.json({ report });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate ledger LLM report", detail: e.message });
    }
  });

  app.get("/api/retrain-trigger/status", generalLimiter, async (_req, res) => {
    try {
      const state = getRetrainTriggerState();
      res.json(state);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch retrain trigger status", detail: e.message });
    }
  });

  app.post("/api/retrain-trigger/configure", engineLimiter, async (req, res) => {
    try {
      const { threshold, enabled } = req.body;
      if (typeof threshold === "number") setRetrainThreshold(threshold);
      if (typeof enabled === "boolean") setRetrainEnabled(enabled);
      const state = getRetrainTriggerState();
      res.json({ updated: true, state });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to configure retrain trigger", detail: e.message });
    }
  });

  app.get("/api/discovery-efficiency", generalLimiter, async (_req, res) => {
    try {
      const cycles = getActiveLearningCycleHistory();
      const efficiencyHistory = cycles.map(c => ({
        cycle: c.cycle,
        usefulDiscoveries: c.discoveryEfficiency.usefulDiscoveries,
        totalEvaluations: c.discoveryEfficiency.totalEvaluations,
        efficiencyRatio: c.discoveryEfficiency.efficiencyRatio,
        stableCount: c.discoveryEfficiency.stableCount,
        highTcCount: c.discoveryEfficiency.highTcCount,
        failureBreakdown: c.discoveryEfficiency.failureBreakdown,
      }));

      const totalUseful = efficiencyHistory.reduce((s, e) => s + e.usefulDiscoveries, 0);
      const totalEvals = efficiencyHistory.reduce((s, e) => s + e.totalEvaluations, 0);
      const totalFailures = {
        unstablePhonons: efficiencyHistory.reduce((s, e) => s + e.failureBreakdown.unstablePhonons, 0),
        highFormationEnergy: efficiencyHistory.reduce((s, e) => s + e.failureBreakdown.highFormationEnergy, 0),
        nonMetallic: efficiencyHistory.reduce((s, e) => s + e.failureBreakdown.nonMetallic, 0),
        lowTc: efficiencyHistory.reduce((s, e) => s + e.failureBreakdown.lowTc, 0),
        pipelineCrash: efficiencyHistory.reduce((s, e) => s + e.failureBreakdown.pipelineCrash, 0),
      };

      const improving = efficiencyHistory.length >= 3;
      let trend = "insufficient data";
      if (improving) {
        const recent = efficiencyHistory.slice(-3);
        const early = efficiencyHistory.slice(0, Math.min(3, efficiencyHistory.length));
        const recentAvg = recent.reduce((s, e) => s + e.efficiencyRatio, 0) / recent.length;
        const earlyAvg = early.reduce((s, e) => s + e.efficiencyRatio, 0) / early.length;
        trend = recentAvg > earlyAvg ? "improving" : recentAvg < earlyAvg ? "degrading" : "stable";
      }

      res.json({
        cumulativeEfficiency: totalEvals > 0 ? totalUseful / totalEvals : 0,
        totalUseful,
        totalEvaluations: totalEvals,
        totalFailures,
        negativeExamplesInTraining: getFailureExampleCount(),
        trend,
        perCycle: efficiencyHistory,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch discovery efficiency", detail: e.message });
    }
  });

  app.get("/api/prediction-ledger/cycle-improvement", generalLimiter, async (_req, res) => {
    try {
      const history = getCycleImprovementHistory();
      const trend = getImprovementTrend();
      res.json({ history, trend });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch cycle improvement history", detail: e.message });
    }
  });

  app.get("/api/model/validation-set", generalLimiter, async (_req, res) => {
    try {
      const valSet = getHeldOutValidationSet();
      res.json({
        size: valSet.length,
        formulas: valSet.map(v => v.formula),
        avgTc: valSet.length > 0 ? valSet.reduce((s, v) => s + v.tc, 0) / valSet.length : 0,
        tcRange: valSet.length > 0 ? {
          min: Math.min(...valSet.map(v => v.tc)),
          max: Math.max(...valSet.map(v => v.tc)),
        } : null,
      });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch validation set info", detail: e.message });
    }
  });

  const REFERENCE_COMPOUNDS = [
    {
      formula: "MgB2",
      name: "Magnesium Diboride",
      family: "Boride",
      textbook: {
        tc: 39.0,
        lambda: 0.87,
        omegaLog: 670,
        crystalSystem: "hexagonal",
        spaceGroup: "P6/mmm",
        yearDiscovered: 2001,
        pressureGpa: 0,
        pairingMechanism: "Conventional (phonon-mediated)",
        notes: "Two-gap s-wave superconductor with sigma and pi bands",
      },
    },
    {
      formula: "YBa2Cu3O7",
      name: "YBCO",
      family: "Cuprate",
      textbook: {
        tc: 92.0,
        lambda: 2.5,
        omegaLog: 350,
        crystalSystem: "orthorhombic",
        spaceGroup: "Pmmm",
        yearDiscovered: 1987,
        pressureGpa: 0,
        pairingMechanism: "Unconventional (d-wave, likely spin-fluctuation mediated)",
        notes: "High-Tc cuprate, CuO2 planes, d-wave order parameter",
      },
    },
    {
      formula: "Nb3Sn",
      name: "Niobium-Tin",
      family: "A15",
      textbook: {
        tc: 18.3,
        lambda: 1.80,
        omegaLog: 215,
        crystalSystem: "cubic",
        spaceGroup: "Pm-3n",
        yearDiscovered: 1954,
        pressureGpa: 0,
        pairingMechanism: "Conventional (phonon-mediated)",
        notes: "A15 structure, used in MRI magnets and particle accelerators",
      },
    },
  ];

  interface BenchmarkResult {
    formula: string;
    name: string;
    family: string;
    textbook: typeof REFERENCE_COMPOUNDS[0]["textbook"];
    predicted: {
      xgboostTc: number;
      xgboostScore: number;
      gnnTc: number;
      gnnUncertainty: number;
      gnnLambda: number;
      ensembleTc: number;
      reasoning: string[];
    };
    accuracy: {
      tcErrorK: number;
      tcErrorPercent: number;
      lambdaError: number | null;
      rating: string;
    };
    computedAt: number;
  }

  let benchmarkCache: { results: BenchmarkResult[]; computedAt: number } | null = null;

  async function runReferenceBenchmark(): Promise<BenchmarkResult[]> {
    const results: BenchmarkResult[] = [];

    for (const ref of REFERENCE_COMPOUNDS) {
      try {
        const features = await extractFeatures(ref.formula);
        const xgb = await gbPredictWithUncertainty(features, ref.formula);
        const gnn = gnnPredictWithUncertainty(ref.formula);

        const xgboostTc = xgb.tcMean ?? 0;
        const gnnTc = gnn.tc ?? 0;
        const ensembleTc = (xgboostTc * 0.4 + gnnTc * 0.6);

        const tcError = Math.abs(ensembleTc - ref.textbook.tc);
        const tcErrorPercent = ref.textbook.tc > 0 ? (tcError / ref.textbook.tc) * 100 : 0;

        let lambdaError: number | null = null;
        if (ref.textbook.lambda > 0 && gnn.lambda > 0) {
          lambdaError = Math.abs(gnn.lambda - ref.textbook.lambda);
        }

        let rating = "poor";
        if (tcErrorPercent < 10) rating = "excellent";
        else if (tcErrorPercent < 25) rating = "good";
        else if (tcErrorPercent < 50) rating = "fair";

        results.push({
          formula: ref.formula,
          name: ref.name,
          family: ref.family,
          textbook: ref.textbook,
          predicted: {
            xgboostTc: Math.round(xgboostTc * 10) / 10,
            xgboostScore: Math.round((xgb.score ?? 0) * 1000) / 1000,
            gnnTc: Math.round(gnnTc * 10) / 10,
            gnnUncertainty: Math.round((gnn.uncertainty ?? 0) * 10) / 10,
            gnnLambda: Math.round((gnn.lambda ?? 0) * 1000) / 1000,
            ensembleTc: Math.round(ensembleTc * 10) / 10,
            reasoning: (xgb.reasoning ?? []).slice(0, 5),
          },
          accuracy: {
            tcErrorK: Math.round(tcError * 10) / 10,
            tcErrorPercent: Math.round(tcErrorPercent * 10) / 10,
            lambdaError: lambdaError !== null ? Math.round(lambdaError * 1000) / 1000 : null,
            rating,
          },
          computedAt: Date.now(),
        });
      } catch (e: any) {
        results.push({
          formula: ref.formula,
          name: ref.name,
          family: ref.family,
          textbook: ref.textbook,
          predicted: {
            xgboostTc: 0, xgboostScore: 0,
            gnnTc: 0, gnnUncertainty: 0, gnnLambda: 0,
            ensembleTc: 0, reasoning: [`Error: ${e.message}`],
          },
          accuracy: { tcErrorK: ref.textbook.tc, tcErrorPercent: 100, lambdaError: null, rating: "error" },
          computedAt: Date.now(),
        });
      }
    }

    benchmarkCache = { results, computedAt: Date.now() };
    return results;
  }

  app.get("/api/reference-benchmark", generalLimiter, async (_req, res) => {
    try {
      if (benchmarkCache && Date.now() - benchmarkCache.computedAt < 300000) {
        res.json(benchmarkCache);
        return;
      }
      if (!benchmarkCache) {
        res.json({ results: [], computedAt: Date.now(), pending: true });
        return;
      }
      res.json(benchmarkCache);
    } catch (e: any) {
      res.status(500).json({ error: "Benchmark failed", detail: e.message });
    }
  });

  app.post("/api/reference-benchmark/refresh", engineLimiter, async (_req, res) => {
    try {
      const results = await runReferenceBenchmark();
      res.json({ results, computedAt: Date.now() });
    } catch (e: any) {
      res.status(500).json({ error: "Benchmark refresh failed", detail: e.message });
    }
  });

  app.get("/api/feature-recalculation/run", generalLimiter, async (_req, res) => {
    try {
      const count = recalculateAllDerivedFeatures();
      res.json({ recalculated: count, timestamp: Date.now() });
    } catch (e: any) {
      res.status(500).json({ error: "Feature recalculation failed", detail: e.message });
    }
  });

  app.get("/api/feature-recalculation/derived/:formula", generalLimiter, async (req, res) => {
    try {
      const derived = getDerivedFeatures(req.params.formula);
      if (!derived) {
        res.json({ formula: req.params.formula, derived: null, message: "No physics results available for this formula" });
        return;
      }
      res.json({ formula: req.params.formula, derived });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get derived features", detail: e.message });
    }
  });

  app.get("/api/training-dataset/stats", generalLimiter, async (_req, res) => {
    try {
      const stats = getUnifiedDatasetStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get training dataset stats", detail: e.message });
    }
  });

  app.get("/api/training-dataset/records", generalLimiter, async (req, res) => {
    try {
      const minTc = req.query.minTc ? parseFloat(req.query.minTc as string) : undefined;
      const maxTc = req.query.maxTc ? parseFloat(req.query.maxTc as string) : undefined;
      const source = req.query.source as string | undefined;
      const tier = req.query.tier as string | undefined;
      const requirePhysics = req.query.requirePhysics === "true";
      const limit = Math.min(parseInt(req.query.limit as string) || 100, 500);
      const offset = parseInt(req.query.offset as string) || 0;
      const records = getTrainingSlice({ minTc, maxTc, source, tier, requirePhysics });
      res.json({ total: records.length, offset, limit, records: records.slice(offset, offset + limit) });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get training records", detail: e.message });
    }
  });

  app.get("/api/training-dataset/llm-report", generalLimiter, async (_req, res) => {
    try {
      const report = getUnifiedDatasetForLLM();
      res.json({ report });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate LLM report", detail: e.message });
    }
  });

  app.get("/api/training-dataset/snapshots", generalLimiter, async (_req, res) => {
    try {
      res.json({ snapshots: getSnapshotHistory() });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get snapshots", detail: e.message });
    }
  });

  app.get("/api/cycle-diagnostics/current", generalLimiter, async (_req, res) => {
    try {
      const report = generateCycleDiagnostics();
      res.json(report);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate cycle diagnostics", detail: e.message });
    }
  });

  app.get("/api/cycle-diagnostics/history", generalLimiter, async (_req, res) => {
    try {
      res.json({ reports: getReportHistory() });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get diagnostics history", detail: e.message });
    }
  });

  app.get("/api/cycle-diagnostics/latest", generalLimiter, async (_req, res) => {
    try {
      const latest = getLatestReport();
      if (!latest) {
        const fresh = generateCycleDiagnostics();
        res.json(fresh);
        return;
      }
      res.json(latest);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get latest diagnostics", detail: e.message });
    }
  });

  app.get("/api/cycle-diagnostics/text", generalLimiter, async (_req, res) => {
    try {
      const report = generateCycleDiagnostics();
      const text = formatDiagnosticReportText(report);
      res.json({ text });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to format diagnostics text", detail: e.message });
    }
  });

  app.get("/api/cycle-diagnostics/llm-report", generalLimiter, async (_req, res) => {
    try {
      const report = getDiagnosticsForLLM();
      res.json({ report });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate LLM diagnostics", detail: e.message });
    }
  });

  try { initVolumeDNN(); } catch {}

  app.get("/api/volume/predict/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const pressure = Number(req.query.pressure) || 0;
      const result = predictVolume(formula, pressure);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to predict volume", detail: e.message });
    }
  });

  app.get("/api/volume/stats", generalLimiter, async (_req, res) => {
    try {
      res.json(getVolumeDNNStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get volume DNN stats", detail: e.message });
    }
  });

  app.get("/api/doping/recommendations/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const recommendations = getDopingRecommendations(formula);
      res.json({ formula, recommendations });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get doping recommendations", detail: e.message });
    }
  });

  app.get("/api/doping/hessian-phonons/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const analysis = await analyzeHessianPhonons(formula);
      if (!analysis) {
        res.json({ formula, analysis: null, message: "Hessian phonon calculation unavailable for this formula" });
        return;
      }
      res.json({ formula, analysis });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to analyze Hessian phonons", detail: e.message });
    }
  });

  app.get("/api/doping/anharmonic/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const analysis = await detectAnharmonicVibrations(formula);
      if (!analysis) {
        res.json({ formula, analysis: null, message: "Anharmonic analysis unavailable for this formula" });
        return;
      }
      res.json({ formula, analysis });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to detect anharmonic vibrations", detail: e.message });
    }
  });

  app.get("/api/doping/md-sampling/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const temp = Math.min(3000, Math.max(10, Number(req.query.temperature) || 300));
      const result = await runMDSampling(formula, temp);
      if (!result) {
        res.json({ formula, result: null, message: "MD sampling unavailable for this formula" });
        return;
      }
      res.json({ formula, result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run MD sampling", detail: e.message });
    }
  });

  app.get("/api/doping/dynamic-lattice-score/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = computeDynamicLatticeScore(formula);
      if (!result) {
        res.json({ formula, result: null, message: "Dynamic lattice score unavailable for this formula" });
        return;
      }
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute dynamic lattice score", detail: e.message });
    }
  });

  app.get("/api/doping/debye-temperature/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = computeDebyeTemp(formula);
      if (!result) {
        res.json({ formula, result: null, message: "Debye temperature unavailable for this formula" });
        return;
      }
      res.json({ formula, result });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to compute Debye temperature", detail: e.message });
    }
  });

  app.get("/api/doping/:formula", generalLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const maxVariants = Math.min(20, Number(req.query.max) || 12);
      const result = generateDopedVariants(formula, maxVariants);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to generate doped variants", detail: e.message });
    }
  });

  app.get("/api/doping-engine/stats", generalLimiter, async (_req, res) => {
    try {
      res.json(getDopingEngineStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to get doping engine stats", detail: e.message });
    }
  });

  app.post("/api/doping/batch", writeLimiter, async (req, res) => {
    try {
      const { formulas, maxPerBase, maxTotal } = req.body;
      if (!Array.isArray(formulas) || formulas.length === 0) {
        return res.status(400).json({ error: "formulas array required" });
      }
      const result = runDopingBatch(
        formulas.slice(0, 50),
        Math.min(12, maxPerBase || 8),
        Math.min(100, maxTotal || 50)
      );
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run doping batch", detail: e.message });
    }
  });

  app.post("/api/doping/relax/:formula", writeLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const maxRelaxations = Math.min(6, Number(req.query.maxRelax) || 4);
      const maxVariants = Math.min(20, Number(req.query.max) || 12);
      const result = await generateDopedVariantsWithRelaxation(formula, maxVariants, maxRelaxations);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to relax doped variants", detail: e.message });
    }
  });

  app.get("/api/doping/sc-signals/:base/:doped", generalLimiter, async (req, res) => {
    try {
      const base = decodeURIComponent(req.params.base);
      const doped = decodeURIComponent(req.params.doped);
      const signals = detectSCSignals(base, doped);
      res.json({ base, doped, signals });
    } catch (e: any) {
      res.status(500).json({ error: "Failed to detect SC signals", detail: e.message });
    }
  });

  app.post("/api/doping/search/:formula", writeLimiter, async (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const fractions = req.body?.fractions ?? [0.02, 0.05, 0.10, 0.15, 0.20];
      const maxDopants = Math.min(4, Number(req.body?.maxDopants) || 2);
      const result = await runDopingSearchLoop(formula, fractions, maxDopants);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to run doping search", detail: e.message });
    }
  });

  return httpServer;
}
