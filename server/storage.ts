import { db } from "./db";
import {
  elements, materials, learningPhases, novelPredictions, researchLogs,
  synthesisProcesses, chemicalReactions, superconductorCandidates,
  crystalStructures, computationalResults, novelInsights,
  researchStrategies, convergenceSnapshots, milestones
} from "@shared/schema";
import type {
  Element, Material, LearningPhase, NovelPrediction, ResearchLog,
  InsertElement, InsertMaterial, InsertLearningPhase, InsertNovelPrediction, InsertResearchLog,
  SynthesisProcess, ChemicalReaction, SuperconductorCandidate,
  InsertSynthesisProcess, InsertChemicalReaction, InsertSuperconductorCandidate,
  CrystalStructure, ComputationalResult, NovelInsight,
  InsertCrystalStructure, InsertComputationalResult, InsertNovelInsight,
  ResearchStrategy, ConvergenceSnapshot, Milestone,
  InsertResearchStrategy, InsertConvergenceSnapshot, InsertMilestone
} from "@shared/schema";
import { eq, desc, asc, sql, ilike } from "drizzle-orm";
import { classifyFamily } from "./learning/utils";

export interface IStorage {
  getElements(): Promise<Element[]>;
  getElementById(id: number): Promise<Element | undefined>;
  insertElement(element: InsertElement): Promise<Element>;

  getMaterials(limit?: number, offset?: number): Promise<Material[]>;
  getMaterialById(id: string): Promise<Material | undefined>;
  insertMaterial(material: InsertMaterial): Promise<Material>;
  getMaterialCount(): Promise<number>;

  getLearningPhases(): Promise<LearningPhase[]>;
  getLearningPhaseById(id: number): Promise<LearningPhase | undefined>;
  upsertLearningPhase(phase: InsertLearningPhase): Promise<LearningPhase>;
  updatePhaseProgress(id: number, progress: number, itemsLearned: number): Promise<void>;

  getNovelPredictions(): Promise<NovelPrediction[]>;
  getNovelPredictionById(id: string): Promise<NovelPrediction | undefined>;
  insertNovelPrediction(pred: InsertNovelPrediction): Promise<NovelPrediction>;

  getResearchLogs(limit?: number): Promise<ResearchLog[]>;
  insertResearchLog(log: InsertResearchLog): Promise<ResearchLog>;

  getSynthesisProcesses(limit?: number): Promise<SynthesisProcess[]>;
  insertSynthesisProcess(proc: InsertSynthesisProcess): Promise<SynthesisProcess>;
  getSynthesisCount(): Promise<number>;

  getChemicalReactions(limit?: number): Promise<ChemicalReaction[]>;
  insertChemicalReaction(rxn: InsertChemicalReaction): Promise<ChemicalReaction>;
  getReactionCount(): Promise<number>;

  getSuperconductorCandidates(limit?: number): Promise<SuperconductorCandidate[]>;
  getSuperconductorCandidatesByTc(limit?: number): Promise<SuperconductorCandidate[]>;
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
  getDistinctScFamilyCount(): Promise<number>;

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
}

export class DatabaseStorage implements IStorage {
  async getElements(): Promise<Element[]> {
    return db.select().from(elements).orderBy(asc(elements.id));
  }

  async getElementById(id: number): Promise<Element | undefined> {
    const [el] = await db.select().from(elements).where(eq(elements.id, id));
    return el;
  }

  async insertElement(element: InsertElement): Promise<Element> {
    const [el] = await db.insert(elements).values(element).onConflictDoNothing().returning();
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

  async updatePhaseProgress(id: number, progress: number, itemsLearned: number): Promise<void> {
    await db.update(learningPhases).set({ progress, itemsLearned }).where(eq(learningPhases.id, id));
  }

  async getNovelPredictions(): Promise<NovelPrediction[]> {
    return db.select().from(novelPredictions).orderBy(desc(novelPredictions.predictedAt));
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
    return db.select().from(superconductorCandidates).orderBy(desc(superconductorCandidates.ensembleScore)).limit(limit);
  }

  async getSuperconductorCandidatesByTc(limit = 10): Promise<SuperconductorCandidate[]> {
    return db.select().from(superconductorCandidates).orderBy(desc(superconductorCandidates.predictedTc)).limit(limit);
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

  async getDistinctScFamilyCount(): Promise<number> {
    const rows = await db.select({ formula: superconductorCandidates.formula })
      .from(superconductorCandidates)
      .orderBy(desc(superconductorCandidates.ensembleScore))
      .limit(50);
    const families = new Set(rows.map(r => classifyFamily(r.formula)));
    return families.size;
  }

  async getStats() {
    const [elCount] = await db.select({ count: sql<number>`count(*)` }).from(elements);
    const [matCount] = await db.select({ count: sql<number>`count(*)` }).from(materials);
    const [predCount] = await db.select({ count: sql<number>`count(*)` }).from(novelPredictions);
    const [synthCount] = await db.select({ count: sql<number>`count(*)` }).from(synthesisProcesses);
    const [rxnCount] = await db.select({ count: sql<number>`count(*)` }).from(chemicalReactions);
    const [scCount] = await db.select({ count: sql<number>`count(*)` }).from(superconductorCandidates);
    const [csCount] = await db.select({ count: sql<number>`count(*)` }).from(crystalStructures);
    const [crCount] = await db.select({ count: sql<number>`count(*)` }).from(computationalResults);
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
    }));

    return {
      elementsLearned: Number(elCount.count),
      materialsIndexed: Number(matCount.count),
      predictionsGenerated: Number(predCount.count),
      overallProgress,
      synthesisProcesses: Number(synthCount.count),
      chemicalReactions: Number(rxnCount.count),
      superconductorCandidates: Number(scCount.count),
      crystalStructures: Number(csCount.count),
      computationalResults: Number(crCount.count),
      pipelineStages,
    };
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
}

export const storage = new DatabaseStorage();
