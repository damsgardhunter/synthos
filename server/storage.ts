import { db } from "./db";
import { elements, materials, learningPhases, novelPredictions, researchLogs } from "@shared/schema";
import type {
  Element, Material, LearningPhase, NovelPrediction, ResearchLog,
  InsertElement, InsertMaterial, InsertLearningPhase, InsertNovelPrediction, InsertResearchLog
} from "@shared/schema";
import { eq, desc, asc, sql } from "drizzle-orm";

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

  getStats(): Promise<{ elementsLearned: number; materialsIndexed: number; predictionsGenerated: number; overallProgress: number }>;
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

  async getStats(): Promise<{ elementsLearned: number; materialsIndexed: number; predictionsGenerated: number; overallProgress: number }> {
    const [elCount] = await db.select({ count: sql<number>`count(*)` }).from(elements);
    const [matCount] = await db.select({ count: sql<number>`count(*)` }).from(materials);
    const [predCount] = await db.select({ count: sql<number>`count(*)` }).from(novelPredictions);
    const phases = await db.select().from(learningPhases);
    const overallProgress = phases.length > 0
      ? phases.reduce((sum, p) => sum + p.progress, 0) / phases.length
      : 0;
    return {
      elementsLearned: Number(elCount.count),
      materialsIndexed: Number(matCount.count),
      predictionsGenerated: Number(predCount.count),
      overallProgress,
    };
  }
}

export const storage = new DatabaseStorage();
