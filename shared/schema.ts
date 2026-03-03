import { pgTable, text, varchar, integer, real, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const elements = pgTable("elements", {
  id: integer("id").primaryKey(),
  symbol: varchar("symbol", { length: 4 }).notNull(),
  name: text("name").notNull(),
  atomicMass: real("atomic_mass"),
  period: integer("period"),
  group: integer("group_num"),
  category: text("category"),
  electronegativity: real("electronegativity"),
  electronConfig: text("electron_config"),
  meltingPoint: real("melting_point"),
  boilingPoint: real("boiling_point"),
  density: real("density"),
  discoveredYear: integer("discovered_year"),
  description: text("description"),
});

export const materials = pgTable("materials", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  formula: text("formula").notNull(),
  spacegroup: text("spacegroup"),
  bandGap: real("band_gap"),
  formationEnergy: real("formation_energy"),
  stability: real("stability"),
  source: text("source").notNull(),
  properties: jsonb("properties"),
  learnedAt: timestamp("learned_at").defaultNow(),
});

export const learningPhases = pgTable("learning_phases", {
  id: integer("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description").notNull(),
  status: text("status").notNull().default("pending"),
  progress: real("progress").notNull().default(0),
  itemsLearned: integer("items_learned").notNull().default(0),
  totalItems: integer("total_items").notNull().default(0),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  insights: text("insights").array(),
});

export const novelPredictions = pgTable("novel_predictions", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  formula: text("formula").notNull(),
  predictedProperties: jsonb("predicted_properties").notNull(),
  confidence: real("confidence").notNull(),
  targetApplication: text("target_application").notNull(),
  status: text("status").notNull().default("predicted"),
  notes: text("notes"),
  predictedAt: timestamp("predicted_at").defaultNow(),
});

export const researchLogs = pgTable("research_logs", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  timestamp: timestamp("timestamp").defaultNow(),
  phase: text("phase").notNull(),
  event: text("event").notNull(),
  detail: text("detail"),
  dataSource: text("data_source"),
});

export const synthesisProcesses = pgTable("synthesis_processes", {
  id: varchar("id").primaryKey(),
  materialId: varchar("material_id"),
  materialName: text("material_name").notNull(),
  formula: text("formula").notNull(),
  method: text("method").notNull(),
  conditions: jsonb("conditions").notNull(),
  steps: text("steps").array().notNull(),
  precursors: text("precursors").array().notNull(),
  equipment: text("equipment").array(),
  difficulty: text("difficulty").notNull().default("moderate"),
  timeEstimate: text("time_estimate"),
  safetyNotes: text("safety_notes"),
  yieldPercent: real("yield_percent"),
  discoveredAt: timestamp("discovered_at").defaultNow(),
});

export const chemicalReactions = pgTable("chemical_reactions", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  equation: text("equation").notNull(),
  reactionType: text("reaction_type").notNull(),
  reactants: jsonb("reactants").notNull(),
  products: jsonb("products").notNull(),
  conditions: jsonb("conditions").notNull(),
  energetics: jsonb("energetics"),
  mechanism: text("mechanism"),
  relevanceToSuperconductor: real("relevance_to_superconductor").default(0),
  labProcess: text("lab_process"),
  source: text("source"),
  learnedAt: timestamp("learned_at").defaultNow(),
});

export const superconductorCandidates = pgTable("superconductor_candidates", {
  id: varchar("id").primaryKey(),
  name: text("name").notNull(),
  formula: text("formula").notNull(),
  predictedTc: real("predicted_tc"),
  pressureGpa: real("pressure_gpa"),
  meissnerEffect: boolean("meissner_effect").default(false),
  zeroResistance: boolean("zero_resistance").default(false),
  cooperPairMechanism: text("cooper_pair_mechanism"),
  crystalStructure: text("crystal_structure"),
  quantumCoherence: real("quantum_coherence"),
  stabilityScore: real("stability_score"),
  synthesisPath: jsonb("synthesis_path"),
  mlFeatures: jsonb("ml_features"),
  xgboostScore: real("xgboost_score"),
  neuralNetScore: real("neural_net_score"),
  ensembleScore: real("ensemble_score"),
  roomTempViable: boolean("room_temp_viable").default(false),
  status: text("status").notNull().default("theoretical"),
  notes: text("notes"),
  generatedAt: timestamp("generated_at").defaultNow(),
});

export const insertElementSchema = createInsertSchema(elements);
export const insertMaterialSchema = createInsertSchema(materials).omit({ learnedAt: true });
export const insertLearningPhaseSchema = createInsertSchema(learningPhases).omit({ startedAt: true, completedAt: true });
export const insertNovelPredictionSchema = createInsertSchema(novelPredictions).omit({ predictedAt: true });
export const insertResearchLogSchema = createInsertSchema(researchLogs).omit({ id: true, timestamp: true });
export const insertSynthesisProcessSchema = createInsertSchema(synthesisProcesses).omit({ discoveredAt: true });
export const insertChemicalReactionSchema = createInsertSchema(chemicalReactions).omit({ learnedAt: true });
export const insertSuperconductorCandidateSchema = createInsertSchema(superconductorCandidates).omit({ generatedAt: true });

export type Element = typeof elements.$inferSelect;
export type Material = typeof materials.$inferSelect;
export type LearningPhase = typeof learningPhases.$inferSelect;
export type NovelPrediction = typeof novelPredictions.$inferSelect;
export type ResearchLog = typeof researchLogs.$inferSelect;
export type SynthesisProcess = typeof synthesisProcesses.$inferSelect;
export type ChemicalReaction = typeof chemicalReactions.$inferSelect;
export type SuperconductorCandidate = typeof superconductorCandidates.$inferSelect;

export type InsertElement = z.infer<typeof insertElementSchema>;
export type InsertMaterial = z.infer<typeof insertMaterialSchema>;
export type InsertLearningPhase = z.infer<typeof insertLearningPhaseSchema>;
export type InsertNovelPrediction = z.infer<typeof insertNovelPredictionSchema>;
export type InsertResearchLog = z.infer<typeof insertResearchLogSchema>;
export type InsertSynthesisProcess = z.infer<typeof insertSynthesisProcessSchema>;
export type InsertChemicalReaction = z.infer<typeof insertChemicalReactionSchema>;
export type InsertSuperconductorCandidate = z.infer<typeof insertSuperconductorCandidateSchema>;

export * from "./models/chat";
