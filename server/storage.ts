import { db } from "./db";
import {
  elements, materials, learningPhases, novelPredictions, researchLogs,
  synthesisProcesses, chemicalReactions, superconductorCandidates,
  crystalStructures, computationalResults, novelInsights,
  researchStrategies, convergenceSnapshots, milestones,
  experimentalValidations, inverseDesignCampaigns, dftJobs
} from "@shared/schema";
import type {
  Element, Material, LearningPhase, NovelPrediction, ResearchLog,
  InsertElement, InsertMaterial, InsertLearningPhase, InsertNovelPrediction, InsertResearchLog,
  SynthesisProcess, ChemicalReaction, SuperconductorCandidate,
  InsertSynthesisProcess, InsertChemicalReaction, InsertSuperconductorCandidate,
  CrystalStructure, ComputationalResult, NovelInsight,
  InsertCrystalStructure, InsertComputationalResult, InsertNovelInsight,
  ResearchStrategy, ConvergenceSnapshot, Milestone,
  InsertResearchStrategy, InsertConvergenceSnapshot, InsertMilestone,
  ExperimentalValidation, InsertExperimentalValidation,
  InverseDesignCampaign, InsertInverseDesignCampaign,
  DftJob, InsertDftJob
} from "@shared/schema";
import { eq, desc, asc, sql, ilike, isNull } from "drizzle-orm";


export interface IStorage {
  getElements(): Promise<Element[]>;
  getElementById(id: number): Promise<Element | undefined>;
  getMaterials(limit?: number, offset?: number): Promise<Material[]>;
  getMaterialById(id: string): Promise<Material | undefined>;
  insertMaterial(material: InsertMaterial): Promise<Material>;
  getMaterialCount(): Promise<number>;

  getLearningPhases(): Promise<LearningPhase[]>;
  getLearningPhaseById(id: number): Promise<LearningPhase | undefined>;
  upsertLearningPhase(phase: InsertLearningPhase): Promise<LearningPhase>;

  getNovelPredictions(limit?: number, offset?: number): Promise<NovelPrediction[]>;
  getNovelPredictionById(id: string): Promise<NovelPrediction | undefined>;
  insertNovelPrediction(pred: InsertNovelPrediction): Promise<NovelPrediction>;

  getResearchLogs(limit?: number): Promise<ResearchLog[]>;
  getResearchLogsByEvent(event: string, limit?: number): Promise<ResearchLog[]>;
  insertResearchLog(log: InsertResearchLog): Promise<ResearchLog>;

  getSynthesisProcesses(limit?: number): Promise<SynthesisProcess[]>;
  insertSynthesisProcess(proc: InsertSynthesisProcess): Promise<SynthesisProcess>;
  getSynthesisCount(): Promise<number>;

  getChemicalReactions(limit?: number): Promise<ChemicalReaction[]>;
  insertChemicalReaction(rxn: InsertChemicalReaction): Promise<ChemicalReaction>;
  getReactionCount(): Promise<number>;

  getSuperconductorCandidates(limit?: number): Promise<SuperconductorCandidate[]>;
  getSuperconductorCandidatesByTc(limit?: number): Promise<SuperconductorCandidate[]>;
  getUnscoredCandidates(limit?: number): Promise<SuperconductorCandidate[]>;
  getCandidatesNeedingPhysicsRecalc(physicsVersion: number, limit?: number): Promise<SuperconductorCandidate[]>;
  insertSuperconductorCandidate(sc: InsertSuperconductorCandidate): Promise<SuperconductorCandidate>;
  updateSuperconductorCandidate(id: string, updates: Partial<InsertSuperconductorCandidate>): Promise<void>;
  getSuperconductorCount(): Promise<number>;
  getSuperconductorsByStage(stage: number, limit?: number): Promise<SuperconductorCandidate[]>;

  getCrystalStructures(limit?: number): Promise<CrystalStructure[]>;
  insertCrystalStructure(cs: InsertCrystalStructure): Promise<CrystalStructure>;
  getCrystalStructureCount(): Promise<number>;
  getCrystalStructuresByFormula(formula: string): Promise<CrystalStructure[]>;

  getComputationalResults(limit?: number): Promise<ComputationalResult[]>;
  insertComputationalResult(cr: InsertComputationalResult): Promise<ComputationalResult>;
  getComputationalResultCount(): Promise<number>;
  getComputationalResultsByStage(stage: number): Promise<ComputationalResult[]>;
  getFailedComputationalResults(limit?: number): Promise<ComputationalResult[]>;

  getSuperconductorsByFormula(formula: string): Promise<SuperconductorCandidate[]>;
  getComputationalResultsByFormula(formula: string): Promise<ComputationalResult[]>;
  getSynthesisProcessesByFormula(formula: string): Promise<SynthesisProcess[]>;
  getChemicalReactionsByFormula(formula: string): Promise<ChemicalReaction[]>;

  getNovelInsights(limit?: number): Promise<NovelInsight[]>;
  getNovelInsightsOnly(limit?: number): Promise<NovelInsight[]>;
  insertNovelInsight(ni: InsertNovelInsight): Promise<NovelInsight>;
  getNovelInsightCount(): Promise<number>;

  insertResearchStrategy(strategy: InsertResearchStrategy): Promise<ResearchStrategy>;
  getLatestStrategy(): Promise<ResearchStrategy | undefined>;
  getStrategyHistory(limit?: number): Promise<ResearchStrategy[]>;

  insertConvergenceSnapshot(snapshot: InsertConvergenceSnapshot): Promise<ConvergenceSnapshot>;
  deleteConvergenceSnapshotByCycle(cycle: number): Promise<void>;
  getConvergenceSnapshots(limit?: number): Promise<ConvergenceSnapshot[]>;
  getMaxConvergenceCycle(): Promise<number>;

  insertMilestone(milestone: InsertMilestone): Promise<Milestone>;
  getMilestones(limit?: number): Promise<Milestone[]>;
  getMilestoneCount(): Promise<number>;

  getSuperconductorByFormula(formula: string): Promise<SuperconductorCandidate | undefined>;
  getNovelPredictionByFormula(formula: string): Promise<NovelPrediction | undefined>;
  updateNovelPrediction(id: string, data: Partial<InsertNovelPrediction>): Promise<void>;
  getTopPredictionFormulas(limit?: number): Promise<string[]>;

  getMaterialCountByElement(symbol: string): Promise<number>;
  getCandidateCountByElement(symbol: string): Promise<number>;

  insertValidation(validation: InsertExperimentalValidation): Promise<ExperimentalValidation>;
  getValidationsByFormula(formula: string): Promise<ExperimentalValidation[]>;
  getValidationStats(): Promise<{ total: number; byResult: Record<string, number> }>;

  getStats(): Promise<{
    elementsLearned: number;
    materialsIndexed: number;
    predictionsGenerated: number;
    overallProgress: number;
    synthesisProcesses: number;
    chemicalReactions: number;
    superconductorCandidates: number;
    crystalStructures: number;
    computationalResults: number;
    pipelineStages: { stage: number; count: number; passed: number }[];
  }>;

  insertInverseDesignCampaign(campaign: InsertInverseDesignCampaign): Promise<InverseDesignCampaign>;
  getInverseDesignCampaigns(): Promise<InverseDesignCampaign[]>;
  getInverseDesignCampaignById(id: string): Promise<InverseDesignCampaign | undefined>;
  updateInverseDesignCampaign(id: string, updates: Partial<InsertInverseDesignCampaign>): Promise<void>;
  deleteInverseDesignCampaign(id: string): Promise<void>;

  insertDftJob(job: InsertDftJob): Promise<DftJob>;
  getDftJob(id: number): Promise<DftJob | undefined>;
  getQueuedDftJobs(limit?: number): Promise<DftJob[]>;
  updateDftJob(id: number, updates: Partial<DftJob>): Promise<void>;
  getDftJobsByFormula(formula: string): Promise<DftJob[]>;
  getDftJobStats(): Promise<{ queued: number; running: number; completed: number; failed: number }>;
  getRecentDftJobs(limit?: number): Promise<DftJob[]>;
}

export class DatabaseStorage implements IStorage {
  async getElements(): Promise<Element[]> {
    return db.select().from(elements).orderBy(asc(elements.id));
  }

  async getElementById(id: number): Promise<Element | undefined> {
    const [el] = await db.select().from(elements).where(eq(elements.id, id));
    return el;
  }

  async getMaterials(limit = 50, offset = 0): Promise<Material[]> {
    return db.select().from(materials).orderBy(desc(materials.learnedAt)).limit(limit).offset(offset);
  }

  async getMaterialById(id: string): Promise<Material | undefined> {
    const [m] = await db.select().from(materials).where(eq(materials.id, id));
    return m;
  }

  async insertMaterial(material: InsertMaterial): Promise<Material> {
    const [m] = await db.insert(materials).values(material).onConflictDoNothing().returning();
    return m;
  }

  async getMaterialCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(materials);
    return Number(count);
  }

  async getLearningPhases(): Promise<LearningPhase[]> {
    return db.select().from(learningPhases).orderBy(asc(learningPhases.id));
  }

  async getLearningPhaseById(id: number): Promise<LearningPhase | undefined> {
    const [p] = await db.select().from(learningPhases).where(eq(learningPhases.id, id));
    return p;
  }

  async upsertLearningPhase(phase: InsertLearningPhase): Promise<LearningPhase> {
    const [p] = await db.insert(learningPhases).values(phase)
      .onConflictDoUpdate({ target: learningPhases.id, set: { name: phase.name, description: phase.description, status: phase.status, progress: phase.progress, itemsLearned: phase.itemsLearned, totalItems: phase.totalItems, insights: phase.insights } })
      .returning();
    return p;
  }

  async getNovelPredictions(limit = 50, offset = 0): Promise<NovelPrediction[]> {
    return db.select().from(novelPredictions).orderBy(desc(novelPredictions.predictedAt)).limit(limit).offset(offset);
  }

  async getNovelPredictionById(id: string): Promise<NovelPrediction | undefined> {
    const [p] = await db.select().from(novelPredictions).where(eq(novelPredictions.id, id));
    return p;
  }

  async insertNovelPrediction(pred: InsertNovelPrediction): Promise<NovelPrediction> {
    const [p] = await db.insert(novelPredictions).values(pred).onConflictDoNothing().returning();
    return p;
  }

  async getResearchLogs(limit = 100): Promise<ResearchLog[]> {
    return db.select().from(researchLogs).orderBy(desc(researchLogs.timestamp)).limit(limit);
  }

  async getResearchLogsByEvent(event: string, limit = 10): Promise<ResearchLog[]> {
    return db.select().from(researchLogs)
      .where(eq(researchLogs.event, event))
      .orderBy(desc(researchLogs.timestamp))
      .limit(limit);
  }

  async insertResearchLog(log: InsertResearchLog): Promise<ResearchLog> {
    const [l] = await db.insert(researchLogs).values(log).returning();
    return l;
  }

  async getSynthesisProcesses(limit = 50): Promise<SynthesisProcess[]> {
    return db.select().from(synthesisProcesses).orderBy(desc(synthesisProcesses.discoveredAt)).limit(limit);
  }

  async insertSynthesisProcess(proc: InsertSynthesisProcess): Promise<SynthesisProcess> {
    const [p] = await db.insert(synthesisProcesses).values(proc).onConflictDoNothing().returning();
    return p;
  }

  async getSynthesisCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(synthesisProcesses);
    return Number(count);
  }

  async getChemicalReactions(limit = 50): Promise<ChemicalReaction[]> {
    return db.select().from(chemicalReactions).orderBy(desc(chemicalReactions.relevanceToSuperconductor)).limit(limit);
  }

  async insertChemicalReaction(rxn: InsertChemicalReaction): Promise<ChemicalReaction> {
    const [r] = await db.insert(chemicalReactions).values(rxn).onConflictDoNothing().returning();
    return r;
  }

  async getReactionCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(chemicalReactions);
    return Number(count);
  }

  async getSuperconductorCandidates(limit = 50): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates).orderBy(desc(superconductorCandidates.predictedTc)).limit(limit);
  }

  async getSuperconductorCandidatesByTc(limit = 10): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates).orderBy(desc(superconductorCandidates.predictedTc)).limit(limit);
  }

  async getUnscoredCandidates(limit = 200): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates).where(
      sql`${superconductorCandidates.xgboostScore} IS NULL OR ${superconductorCandidates.neuralNetScore} IS NULL`
    ).limit(limit);
  }

  async getCandidatesNeedingPhysicsRecalc(physicsVersion: number, limit = 200): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates).where(
      sql`(${superconductorCandidates.mlFeatures}->>'physicsVersion')::int IS DISTINCT FROM ${physicsVersion}`
    ).limit(limit);
  }

  async insertSuperconductorCandidate(sc: InsertSuperconductorCandidate): Promise<SuperconductorCandidate> {
    const [s] = await db.insert(superconductorCandidates).values(sc)
      .onConflictDoUpdate({
        target: superconductorCandidates.formula,
        set: {
          ensembleScore: sql`GREATEST(${superconductorCandidates.ensembleScore}, EXCLUDED.ensemble_score)`,
          xgboostScore: sql`CASE WHEN EXCLUDED.ensemble_score > ${superconductorCandidates.ensembleScore} THEN EXCLUDED.xgboost_score ELSE ${superconductorCandidates.xgboostScore} END`,
          neuralNetScore: sql`CASE WHEN EXCLUDED.ensemble_score > ${superconductorCandidates.ensembleScore} THEN EXCLUDED.neural_net_score ELSE ${superconductorCandidates.neuralNetScore} END`,
          predictedTc: sql`CASE WHEN EXCLUDED.ensemble_score > ${superconductorCandidates.ensembleScore} THEN EXCLUDED.predicted_tc ELSE ${superconductorCandidates.predictedTc} END`,
          mlFeatures: sql`CASE WHEN EXCLUDED.ensemble_score > ${superconductorCandidates.ensembleScore} THEN EXCLUDED.ml_features ELSE ${superconductorCandidates.mlFeatures} END`,
          notes: sql`CASE WHEN EXCLUDED.ensemble_score > ${superconductorCandidates.ensembleScore} THEN EXCLUDED.notes ELSE ${superconductorCandidates.notes} END`,
        },
      })
      .returning();
    return s;
  }

  async updateSuperconductorCandidate(id: string, updates: Partial<InsertSuperconductorCandidate>): Promise<void> {
    await db.update(superconductorCandidates).set(updates).where(eq(superconductorCandidates.id, id));
  }

  async getSuperconductorCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(superconductorCandidates);
    return Number(count);
  }

  async getSuperconductorsByStage(stage: number, limit = 100): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates)
      .where(eq(superconductorCandidates.verificationStage, stage))
      .orderBy(desc(superconductorCandidates.ensembleScore))
      .limit(limit);
  }

  async getCrystalStructures(limit = 50): Promise<CrystalStructure[]> {
    return db.select().from(crystalStructures).orderBy(desc(crystalStructures.predictedAt)).limit(limit);
  }

  async insertCrystalStructure(cs: InsertCrystalStructure): Promise<CrystalStructure> {
    const [c] = await db.insert(crystalStructures).values(cs).onConflictDoNothing().returning();
    return c;
  }

  async getCrystalStructureCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(crystalStructures);
    return Number(count);
  }

  async getCrystalStructuresByFormula(formula: string): Promise<CrystalStructure[]> {
    return db.select().from(crystalStructures).where(eq(crystalStructures.formula, formula));
  }

  async getSuperconductorsByFormula(formula: string): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates).where(eq(superconductorCandidates.formula, formula)).orderBy(desc(superconductorCandidates.ensembleScore));
  }

  async getComputationalResultsByFormula(formula: string): Promise<ComputationalResult[]> {
    return db.select().from(computationalResults).where(eq(computationalResults.formula, formula)).orderBy(desc(computationalResults.computedAt));
  }

  async getSynthesisProcessesByFormula(formula: string): Promise<SynthesisProcess[]> {
    return db.select().from(synthesisProcesses).where(eq(synthesisProcesses.formula, formula)).orderBy(desc(synthesisProcesses.discoveredAt));
  }

  async getChemicalReactionsByFormula(formula: string): Promise<ChemicalReaction[]> {
    return db.select().from(chemicalReactions).where(ilike(chemicalReactions.equation, `%${formula}%`)).orderBy(desc(chemicalReactions.relevanceToSuperconductor));
  }

  async getComputationalResults(limit = 50): Promise<ComputationalResult[]> {
    return db.select().from(computationalResults).orderBy(desc(computationalResults.computedAt)).limit(limit);
  }

  async insertComputationalResult(cr: InsertComputationalResult): Promise<ComputationalResult> {
    const [c] = await db.insert(computationalResults).values(cr).onConflictDoNothing().returning();
    return c;
  }

  async getComputationalResultCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(computationalResults);
    return Number(count);
  }

  async getComputationalResultsByStage(stage: number): Promise<ComputationalResult[]> {
    return db.select().from(computationalResults)
      .where(eq(computationalResults.pipelineStage, stage))
      .orderBy(desc(computationalResults.computedAt));
  }

  async getFailedComputationalResults(limit = 50): Promise<ComputationalResult[]> {
    return db.select().from(computationalResults)
      .where(eq(computationalResults.passed, false))
      .orderBy(desc(computationalResults.computedAt))
      .limit(limit);
  }

  async getNovelInsights(limit = 50): Promise<NovelInsight[]> {
    return db.select().from(novelInsights).orderBy(desc(novelInsights.discoveredAt)).limit(limit);
  }

  async getNovelInsightsOnly(limit = 50): Promise<NovelInsight[]> {
    return db.select().from(novelInsights)
      .where(eq(novelInsights.isNovel, true))
      .orderBy(desc(novelInsights.discoveredAt))
      .limit(limit);
  }

  async insertNovelInsight(ni: InsertNovelInsight): Promise<NovelInsight> {
    const [c] = await db.insert(novelInsights).values(ni).onConflictDoNothing().returning();
    return c;
  }

  async getNovelInsightCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(novelInsights);
    return Number(count);
  }

  async insertResearchStrategy(strategy: InsertResearchStrategy): Promise<ResearchStrategy> {
    const [s] = await db.insert(researchStrategies).values(strategy).returning();
    return s;
  }

  async getLatestStrategy(): Promise<ResearchStrategy | undefined> {
    const [s] = await db.select().from(researchStrategies).orderBy(desc(researchStrategies.cycle)).limit(1);
    return s;
  }

  async getStrategyHistory(limit = 20): Promise<ResearchStrategy[]> {
    return db.select().from(researchStrategies).orderBy(desc(researchStrategies.cycle)).limit(limit);
  }

  async insertConvergenceSnapshot(snapshot: InsertConvergenceSnapshot): Promise<ConvergenceSnapshot> {
    const [s] = await db.insert(convergenceSnapshots).values(snapshot).returning();
    return s;
  }

  async deleteConvergenceSnapshotByCycle(cycle: number): Promise<void> {
    await db.delete(convergenceSnapshots).where(eq(convergenceSnapshots.cycle, cycle));
  }

  async getConvergenceSnapshots(limit = 50): Promise<ConvergenceSnapshot[]> {
    const rows = await db.select().from(convergenceSnapshots).orderBy(desc(convergenceSnapshots.cycle)).limit(limit);
    return rows.reverse();
  }

  async getMaxConvergenceCycle(): Promise<number> {
    const [row] = await db.select({ maxCycle: sql<number>`COALESCE(MAX(${convergenceSnapshots.cycle}), 0)` }).from(convergenceSnapshots);
    return row?.maxCycle ?? 0;
  }

  async insertMilestone(milestone: InsertMilestone): Promise<Milestone> {
    const [m] = await db.insert(milestones).values(milestone).returning();
    return m;
  }

  async getMilestones(limit = 50): Promise<Milestone[]> {
    return db.select().from(milestones).orderBy(desc(milestones.createdAt)).limit(limit);
  }

  async getMilestoneCount(): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` }).from(milestones);
    return Number(count);
  }

  async getSuperconductorByFormula(formula: string): Promise<SuperconductorCandidate | undefined> {
    const [c] = await db.select().from(superconductorCandidates)
      .where(eq(superconductorCandidates.formula, formula))
      .orderBy(desc(superconductorCandidates.ensembleScore))
      .limit(1);
    return c;
  }

  async getNovelPredictionByFormula(formula: string): Promise<NovelPrediction | undefined> {
    const [p] = await db.select().from(novelPredictions)
      .where(eq(novelPredictions.formula, formula))
      .limit(1);
    return p;
  }

  async updateNovelPrediction(id: string, data: Partial<InsertNovelPrediction>): Promise<void> {
    await db.update(novelPredictions).set(data).where(eq(novelPredictions.id, id));
  }

  async getTopPredictionFormulas(limit = 20): Promise<string[]> {
    const rows = await db.select({ formula: novelPredictions.formula })
      .from(novelPredictions)
      .groupBy(novelPredictions.formula)
      .orderBy(sql`count(*) desc`)
      .limit(limit);
    return rows.map(r => r.formula);
  }

  async getStats() {
    const countsResult = await db.execute(sql`
      SELECT
        (SELECT count(*) FROM elements) AS elements_count,
        (SELECT count(*) FROM materials) AS materials_count,
        (SELECT count(*) FROM novel_predictions) AS predictions_count,
        (SELECT count(*) FROM synthesis_processes) AS synthesis_count,
        (SELECT count(*) FROM chemical_reactions) AS reactions_count,
        (SELECT count(*) FROM superconductor_candidates) AS sc_count,
        (SELECT count(*) FROM crystal_structures) AS cs_count,
        (SELECT count(*) FROM computational_results) AS cr_count
    `);
    const rows = (countsResult as any).rows ?? countsResult;
    const c = (Array.isArray(rows) ? rows[0] : rows) ?? {};

    const phases = await db.select().from(learningPhases);
    const overallProgress = phases.length > 0
      ? phases.reduce((sum, p) => sum + p.progress, 0) / phases.length
      : 0;

    const stageRows = await db.select({
      stage: computationalResults.pipelineStage,
      count: sql<number>`count(*)`,
      passed: sql<number>`count(*) filter (where ${computationalResults.passed} = true)`,
    }).from(computationalResults).groupBy(computationalResults.pipelineStage);

    const pipelineStages = stageRows.map(r => ({
      stage: r.stage,
      count: Number(r.count),
      passed: Number(r.passed),
    })).sort((a, b) => a.stage - b.stage);

    return {
      elementsLearned: Number(c.elements_count),
      materialsIndexed: Number(c.materials_count),
      predictionsGenerated: Number(c.predictions_count),
      overallProgress,
      synthesisProcesses: Number(c.synthesis_count),
      chemicalReactions: Number(c.reactions_count),
      superconductorCandidates: Number(c.sc_count),
      crystalStructures: Number(c.cs_count),
      computationalResults: Number(c.cr_count),
      pipelineStages,
    };
  }

  async getMaterialCountByElement(symbol: string): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` })
      .from(materials)
      .where(sql`${materials.formula} ~ ('(^|[^a-z])' || ${symbol} || '([^a-z]|[0-9]|$)')`);
    return Number(count);
  }

  async getCandidateCountByElement(symbol: string): Promise<number> {
    const [{ count }] = await db.select({ count: sql<number>`count(*)` })
      .from(superconductorCandidates)
      .where(sql`${superconductorCandidates.formula} ~ ('(^|[^a-z])' || ${symbol} || '([^a-z]|[0-9]|$)')`);
    return Number(count);
  }

  async insertValidation(validation: InsertExperimentalValidation): Promise<ExperimentalValidation> {
    const [v] = await db.insert(experimentalValidations).values(validation).returning();
    return v;
  }

  async getValidationsByFormula(formula: string): Promise<ExperimentalValidation[]> {
    return db.select().from(experimentalValidations)
      .where(eq(experimentalValidations.formula, formula))
      .orderBy(desc(experimentalValidations.performedAt));
  }

  async getValidationStats(): Promise<{ total: number; byResult: Record<string, number> }> {
    const rows = await db.select({
      result: experimentalValidations.result,
      count: sql<number>`count(*)`,
    }).from(experimentalValidations).groupBy(experimentalValidations.result);
    const byResult: Record<string, number> = {};
    let total = 0;
    for (const r of rows) {
      byResult[r.result] = Number(r.count);
      total += Number(r.count);
    }
    return { total, byResult };
  }

  async insertInverseDesignCampaign(campaign: InsertInverseDesignCampaign): Promise<InverseDesignCampaign> {
    const [result] = await db.insert(inverseDesignCampaigns).values(campaign).returning();
    return result;
  }

  async getInverseDesignCampaigns(): Promise<InverseDesignCampaign[]> {
    return db.select().from(inverseDesignCampaigns).orderBy(desc(inverseDesignCampaigns.createdAt));
  }

  async getInverseDesignCampaignById(id: string): Promise<InverseDesignCampaign | undefined> {
    const [result] = await db.select().from(inverseDesignCampaigns).where(eq(inverseDesignCampaigns.id, id));
    return result;
  }

  async updateInverseDesignCampaign(id: string, updates: Partial<InsertInverseDesignCampaign>): Promise<void> {
    await db.update(inverseDesignCampaigns).set(updates).where(eq(inverseDesignCampaigns.id, id));
  }

  async deleteInverseDesignCampaign(id: string): Promise<void> {
    await db.delete(inverseDesignCampaigns).where(eq(inverseDesignCampaigns.id, id));
  }

  async deduplicateSuperconductorCandidates(): Promise<number> {
    const result = await db.execute(sql`
      DELETE FROM superconductor_candidates
      WHERE id NOT IN (
        SELECT DISTINCT ON (formula) id
        FROM superconductor_candidates
        ORDER BY formula, COALESCE(ensemble_score, 0) DESC, generated_at DESC
      )
    `);
    return Number((result as any).rowCount ?? 0);
  }

  async insertDftJob(job: InsertDftJob): Promise<DftJob> {
    const [result] = await db.insert(dftJobs).values(job).returning();
    return result;
  }

  async getDftJob(id: number): Promise<DftJob | undefined> {
    const [result] = await db.select().from(dftJobs).where(eq(dftJobs.id, id));
    return result;
  }

  async getQueuedDftJobs(limit = 10): Promise<DftJob[]> {
    return db.select().from(dftJobs)
      .where(eq(dftJobs.status, "queued"))
      .orderBy(desc(dftJobs.priority), asc(dftJobs.createdAt))
      .limit(limit);
  }

  async updateDftJob(id: number, updates: Partial<DftJob>): Promise<void> {
    await db.update(dftJobs).set(updates).where(eq(dftJobs.id, id));
  }

  async getDftJobsByFormula(formula: string): Promise<DftJob[]> {
    return db.select().from(dftJobs)
      .where(eq(dftJobs.formula, formula))
      .orderBy(desc(dftJobs.createdAt));
  }

  async getDftJobStats(): Promise<{ queued: number; running: number; completed: number; failed: number }> {
    const rows = await db.execute(sql`
      SELECT status, COUNT(*)::int as count FROM dft_jobs GROUP BY status
    `);
    const stats = { queued: 0, running: 0, completed: 0, failed: 0 };
    for (const row of (rows as any).rows || rows) {
      const s = (row as any).status as keyof typeof stats;
      if (s in stats) stats[s] = (row as any).count;
    }
    return stats;
  }

  async getRecentDftJobs(limit = 20): Promise<DftJob[]> {
    return db.select().from(dftJobs)
      .orderBy(desc(dftJobs.createdAt))
      .limit(limit);
  }
}

export const storage = new DatabaseStorage();
