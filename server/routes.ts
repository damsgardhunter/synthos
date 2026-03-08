import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertMaterialSchema, insertResearchLogSchema, insertExperimentalValidationSchema } from "@shared/schema";
import { initWebSocket, startEngine, stopEngine, pauseEngine, resumeEngine, getStatus, getAutonomousLoopStats } from "./learning/engine";
import { isDFTAvailable, getDFTMethodInfo, getXTBStats } from "./dft/qe-dft-engine";
import { getDFTQueueStats, startDFTWorkerLoop } from "./dft/dft-job-queue";
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
} from "./inverse/self-improving-lab";
import {
  generateDesignProgram, executeDesignProgram, mutateDesignProgram, crossoverPrograms,
  generateDesignGraph, mutateDesignGraph, analyzeGraph,
  programToGraph, graphToProgram,
  getDesignRepresentationStats,
  type DesignProgram, type DesignGraph,
} from "./inverse/design-representations";
import { getCalibrationData, getConfidenceBand } from "./learning/gradient-boost";
import { cache, TTL, CACHE_KEYS } from "./cache";
import rateLimit from "express-rate-limit";
import { fetchAllData as fetchMPAllData, isApiAvailable as isMPAvailable } from "./learning/materials-project-client";
import { fetchAflowData, crossValidateWithMP, crossValidateWithAflow } from "./learning/aflow-client";
import { sanitizeForbiddenWords } from "./learning/utils";
import { runDiffusionGenerationCycle, getDiffusionStats as getDiffusionGeneratorStats } from "./ai/crystal-generator";
import { analyzeTopology, getTopologyStats } from "./physics/topology-engine";
import { computeElectronicStructure } from "./learning/physics-engine";
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
import { predictStability } from "./physics/stability-predictor";
import { analyzeInterface, generateHeterostructureCandidates } from "./physics/interface-engine";
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
  getCausalRules, getLatestGraph,
} from "./theory/causal-physics-discovery";
import { computeMultiScaleFeatures, computeCrossScaleCoupling, runSensitivityAnalysis } from "./theory/multi-scale-engine";
import { getPhysicsParameters, getParameterHistory, getModelPerformance } from "./theory/self-improving-physics";
import { getPerformanceMetrics, recordPrediction } from "./theory/model-performance-tracker";
import { getGeneratorAllocations } from "./learning/generator-manager";
import { getEmbeddingDataset, getLandscapeStats } from "./landscape/discovery-landscape";
import { getZoneMap } from "./landscape/zone-detector";
import { getFullLandscapeGuidance } from "./landscape/landscape-guidance";
import { solveConstraints, evaluateFormulaAgainstConstraints } from "./inverse/constraint-solver";
import { searchPressurePathways, getPathwayStats, getAmbientCandidatesFromPathways } from "./inverse/pressure-pathway";
import { solveConstraintGraph, getFeasibleRegions } from "./inverse/constraint-graph-solver";
import { computeSynthesisPathway, getSynthesisPathwayStats } from "./synthesis/reaction-pathway";
import { getParameterSpace } from "./synthesis/synthesis-variables";
import { getSynthesisOptimizerStats } from "./synthesis/synthesis-condition-optimizer";
import {
  getSimulatorStats, simulateSynthesisEffects, defaultSynthesisVector,
  optimizeSynthesisForFixedMaterial, optimizeSynthesisPath,
} from "./physics/synthesis-simulator";
import { getSynthesisLearningStats, querySimilarSynthesis } from "./synthesis/synthesis-learning-db";
import { generateDefectVariants, adjustElectronicStructure, getDefectEngineStats } from "./physics/defect-engine";
import { estimateCorrelationEffects, getCorrelationEngineStats } from "./physics/correlation-engine";
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

  startDFTWorkerLoop();

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
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const offset = Number(req.query.offset) || 0;
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
      const phases = await storage.getLearningPhases();
      const sanitized = phases.map(p => ({
        ...p,
        insights: (p.insights ?? []).map((s: string) => sanitizeForbiddenWords(s)),
      }));
      res.json(sanitized);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch learning phases" });
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
      const logs = await storage.getResearchLogs(limit);
      const sanitized = logs.map(log => ({
        ...log,
        detail: sanitizeForbiddenWords(log.detail || ""),
        event: sanitizeForbiddenWords(log.event || ""),
      }));
      res.json(sanitized);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch research logs" });
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
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const candidates = await storage.getSuperconductorCandidates(limit);
      const total = await storage.getSuperconductorCount();
      res.json({ candidates, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch superconductor candidates" });
    }
  });

  app.get("/api/dft-status", async (_req, res) => {
    try {
      const candidates = await storage.getSuperconductorCandidates(1000);
      const total = await storage.getSuperconductorCount();
      const dftEnriched = candidates.filter(c => c.dataConfidence === "high" || c.dataConfidence === "medium");
      const dftHighCount = candidates.filter(c => c.dataConfidence === "high").length;
      const dftMediumCount = candidates.filter(c => c.dataConfidence === "medium").length;
      const analyticalCount = total - dftHighCount - dftMediumCount;

      const recentEnriched = dftEnriched
        .sort((a, b) => (b.id ?? 0) - (a.id ?? 0))
        .slice(0, 10)
        .map(c => ({
          formula: c.formula,
          confidence: c.dataConfidence,
          ensembleScore: c.ensembleScore,
          predictedTc: c.predictedTc,
        }));

      const dftAvailable = isDFTAvailable();
      const methodInfo = dftAvailable ? getDFTMethodInfo() : null;
      const xtbStats = getXTBStats();

      res.json({
        total,
        dftEnrichedCount: dftEnriched.length,
        breakdown: {
          high: dftHighCount,
          medium: dftMediumCount,
          analytical: analyticalCount,
        },
        recentEnriched,
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
      });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch DFT status" });
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
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const stage = req.query.stage ? Number(req.query.stage) : undefined;
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
      const calibration = getCalibrationData();
      const tc = req.query.tc ? Number(req.query.tc) : undefined;
      const confidenceBand = tc !== undefined && Number.isFinite(tc) ? getConfidenceBand(tc) : undefined;
      res.json({ ...calibration, ...(confidenceBand ? { confidenceBand } : {}) });
    } catch (e) {
      res.status(500).json({ error: "Failed to compute ML calibration" });
    }
  });

  app.get("/api/research-strategy", async (_req, res) => {
    try {
      const strategy = await storage.getLatestStrategy();
      res.json(strategy || null);
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch research strategy" });
    }
  });

  app.get("/api/research-strategy/history", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 20, 100);
      const history = await storage.getStrategyHistory(limit);
      res.json(history);
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
      const ms = await storage.getMilestones(limit);
      const total = await storage.getMilestoneCount();
      res.json({ milestones: ms, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch milestones" });
    }
  });

  app.get("/api/novel-insights", async (req, res) => {
    try {
      const limit = Math.min(Number(req.query.limit) || 200, 1000);
      const novelOnly = req.query.novelOnly === "true";
      const insights = novelOnly
        ? await storage.getNovelInsightsOnly(limit)
        : await storage.getNovelInsights(limit);
      const total = await storage.getNovelInsightCount();
      res.json({ insights, total });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch novel insights" });
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
      const strategies = await storage.getStrategyHistory(10);
      const latestStrategy = strategies[0] ?? null;
      const insights = await storage.getNovelInsightsOnly(50);
      const topInsights = insights.slice(0, 5);
      const milestones = await storage.getMilestones(20);
      const milestoneCount = await storage.getMilestoneCount();
      const snapshots = await storage.getConvergenceSnapshots(50);
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

      const narratives = await storage.getResearchLogsByEvent("cycle-narrative", 10);

      res.json({
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
        ...(await (async () => {
          const loopStats = await getAutonomousLoopStats();
          return {
            autonomousLoopStats: loopStats,
            lastCycleCandidates: loopStats.lastCycleCandidates ?? [],
            lastCycleFamilyCounts: loopStats.lastCycleFamilyCounts ?? {},
          };
        })()),
        designRepresentations: getDesignRepresentationStats(),
      });
    } catch (e) {
      res.status(500).json({ error: "Failed to fetch engine memory" });
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
        } catch {}
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
      const result = runDifferentiableOptimization(formula, target, Number(maxSteps ?? 20));
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
      const cycle = runGradientDescentCycle(target, Number(seedCount ?? 6), Number(stepsPerSeed ?? 15));
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
      if (!targetTc) {
        return res.status(400).json({ error: "targetTc is required" });
      }
      const results = runStructureFirstDesign(
        Number(targetTc),
        Number(motifCount ?? 4),
        Number(elementsPerSite ?? 3),
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
      const { formula } = req.body;
      if (!formula || typeof formula !== "string") {
        return res.status(400).json({ error: "formula is required" });
      }
      const result = checkPhysicsConstraints(formula);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Constraint check failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/physics-constraints/batch", writeLimiter, (req, res) => {
    try {
      const { formulas } = req.body;
      if (!Array.isArray(formulas) || formulas.length === 0) {
        return res.status(400).json({ error: "formulas array is required" });
      }
      const result = constraintGuidedGenerate(formulas.slice(0, 200));
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

  app.post("/api/sc-pillars/evaluate", writeLimiter, (req, res) => {
    try {
      const { formula, targets } = req.body;
      if (!formula || typeof formula !== "string") {
        return res.status(400).json({ error: "formula is required" });
      }
      const result = evaluatePillars(formula, targets ?? DEFAULT_PILLAR_TARGETS);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Pillar evaluation failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/sc-pillars/generate", writeLimiter, (req, res) => {
    try {
      const { candidatesPerTemplate, targets } = req.body;
      const result = runPillarGuidedGeneration(
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
      const result = analyzeReactionNetwork(formula);
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
      const prediction = predictStability(formula);
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

  app.get("/api/band-operator/stats", generalLimiter, (_req, res) => {
    try {
      const stats = getBandOperatorStats();
      res.json(stats);
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch band operator stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/band-operator/dispersion/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = predictBandDispersion(formula);
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

  app.get("/api/band-operator/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const result = predictBandDispersion(formula);
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

  app.get("/api/defect-engine/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getDefectEngineStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch defect engine stats", detail: e.message?.slice(0, 200) });
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

  app.get("/api/correlation-engine/stats", generalLimiter, (_req, res) => {
    try {
      res.json(getCorrelationEngineStats());
    } catch (e: any) {
      res.status(500).json({ error: "Failed to fetch correlation engine stats", detail: e.message?.slice(0, 200) });
    }
  });

  app.get("/api/correlation-engine/:formula", generalLimiter, (req, res) => {
    try {
      const formula = decodeURIComponent(req.params.formula);
      const analysis = estimateCorrelationEffects(formula, {});
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

  app.post("/api/symbolic-discovery/run", writeLimiter, (req, res) => {
    try {
      const config: Partial<SymbolicDiscoveryConfig> = req.body.config ?? {};
      const dataset = generateSyntheticDataset(50);
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

  app.get("/api/physics-discovery-dataset", generalLimiter, (_req, res) => {
    try {
      const dataset = generateSyntheticDataset(50);
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

  app.post("/api/physics-discovery-dataset/record", writeLimiter, (req, res) => {
    try {
      const { formula } = req.body;
      if (!formula) return res.status(400).json({ error: "formula is required" });
      const record = buildPhysicsDiscoveryRecord(formula);
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

  app.post("/api/causal-discovery/run", writeLimiter, (req, res) => {
    try {
      const count = req.body?.datasetSize ?? 60;
      const dataset = generateCausalDataset(Math.min(count, 100));
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

  app.post("/api/causal-discovery/intervene", writeLimiter, (req, res) => {
    try {
      const { formula, variable, newValue } = req.body;
      if (!formula || !variable || newValue === undefined) {
        return res.status(400).json({ error: "Missing formula, variable, or newValue" });
      }
      const graph = getLatestGraph();
      if (!graph) {
        return res.status(400).json({ error: "No causal graph discovered yet. Run discovery first." });
      }
      const record = buildCausalDataRecord(formula);
      const result = simulateIntervention(record, variable, newValue, graph);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Intervention failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/causal-discovery/counterfactual", writeLimiter, (req, res) => {
    try {
      const { formula, variable, modificationPercent } = req.body;
      if (!formula || !variable || modificationPercent === undefined) {
        return res.status(400).json({ error: "Missing formula, variable, or modificationPercent" });
      }
      const graph = getLatestGraph();
      if (!graph) {
        return res.status(400).json({ error: "No causal graph discovered yet. Run discovery first." });
      }
      const record = buildCausalDataRecord(formula);
      const result = runCounterfactual(record, variable, modificationPercent, graph);
      res.json(result);
    } catch (e: any) {
      res.status(500).json({ error: "Counterfactual failed", detail: e.message?.slice(0, 200) });
    }
  });

  app.post("/api/causal-discovery/dataset", writeLimiter, (req, res) => {
    try {
      const count = req.body?.count ?? 60;
      const dataset = generateCausalDataset(Math.min(count, 100));
      res.json({ count: dataset.length, records: dataset.slice(0, 20) });
    } catch (e: any) {
      res.status(500).json({ error: "Dataset generation failed", detail: e.message?.slice(0, 200) });
    }
  });

  return httpServer;
}
