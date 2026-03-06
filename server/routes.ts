import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertMaterialSchema, insertResearchLogSchema, insertExperimentalValidationSchema } from "@shared/schema";
import { initWebSocket, startEngine, stopEngine, pauseEngine, resumeEngine, getStatus, getAutonomousLoopStats } from "./learning/engine";
import { isDFTAvailable, getDFTMethodInfo, getXTBStats } from "./dft/qe-dft-engine";
import { getCalibrationData, getConfidenceBand } from "./learning/gradient-boost";
import { cache, TTL, CACHE_KEYS } from "./cache";
import rateLimit from "express-rate-limit";
import { fetchAllData as fetchMPAllData, isApiAvailable as isMPAvailable } from "./learning/materials-project-client";
import { fetchAflowData, crossValidateWithMP, crossValidateWithAflow } from "./learning/aflow-client";
import { sanitizeForbiddenWords } from "./learning/utils";

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
        realDFT: {
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
        autonomousLoopStats: getAutonomousLoopStats(),
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

  return httpServer;
}
