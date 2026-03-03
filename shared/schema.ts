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
  electronPhononCoupling: real("electron_phonon_coupling"),
  logPhononFrequency: real("log_phonon_frequency"),
  coulombPseudopotential: real("coulomb_pseudopotential"),
  pairingSymmetry: text("pairing_symmetry"),
  pairingMechanism: text("pairing_mechanism"),
  competingPhases: jsonb("competing_phases"),
  upperCriticalField: real("upper_critical_field"),
  coherenceLength: real("coherence_length"),
  londonPenetrationDepth: real("london_penetration_depth"),
  anisotropyRatio: real("anisotropy_ratio"),
  criticalCurrentDensity: real("critical_current_density"),
  dimensionality: text("dimensionality"),
  fermiSurfaceTopology: text("fermi_surface_topology"),
  correlationStrength: real("correlation_strength"),
  decompositionEnergy: real("decomposition_energy"),
  ambientPressureStable: boolean("ambient_pressure_stable").default(false),
  verificationStage: integer("verification_stage").default(0),
  uncertaintyEstimate: real("uncertainty_estimate"),
});

export const crystalStructures = pgTable("crystal_structures", {
  id: varchar("id").primaryKey(),
  formula: text("formula").notNull(),
  spaceGroup: text("space_group").notNull(),
  crystalSystem: text("crystal_system").notNull(),
  latticeParams: jsonb("lattice_params"),
  atomicPositions: jsonb("atomic_positions"),
  prototype: text("prototype"),
  dimensionality: text("dimensionality").notNull().default("3D"),
  isStable: boolean("is_stable").default(false),
  isMetastable: boolean("is_metastable").default(false),
  decompositionEnergy: real("decomposition_energy"),
  convexHullDistance: real("convex_hull_distance"),
  synthesizability: real("synthesizability"),
  synthesisNotes: text("synthesis_notes"),
  source: text("source"),
  predictedAt: timestamp("predicted_at").defaultNow(),
});

export const computationalResults = pgTable("computational_results", {
  id: varchar("id").primaryKey(),
  candidateId: varchar("candidate_id"),
  formula: text("formula").notNull(),
  computationType: text("computation_type").notNull(),
  pipelineStage: integer("pipeline_stage").notNull().default(0),
  inputParams: jsonb("input_params"),
  results: jsonb("results").notNull(),
  confidence: real("confidence"),
  computeTimeMs: integer("compute_time_ms"),
  passed: boolean("passed").default(false),
  failureReason: text("failure_reason"),
  computedAt: timestamp("computed_at").defaultNow(),
});

export const novelInsights = pgTable("novel_insights", {
  id: varchar("id").primaryKey(),
  phaseId: integer("phase_id").notNull(),
  phaseName: text("phase_name").notNull(),
  insightText: text("insight_text").notNull(),
  isNovel: boolean("is_novel").default(false),
  noveltyScore: real("novelty_score"),
  noveltyReason: text("novelty_reason"),
  category: text("category"),
  relatedFormulas: text("related_formulas").array(),
  discoveredAt: timestamp("discovered_at").defaultNow(),
});

export const researchStrategies = pgTable("research_strategies", {
  id: varchar("id").primaryKey(),
  cycle: integer("cycle").notNull(),
  focusAreas: jsonb("focus_areas").notNull(),
  summary: text("summary").notNull(),
  performanceSignals: jsonb("performance_signals"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const convergenceSnapshots = pgTable("convergence_snapshots", {
  id: varchar("id").primaryKey(),
  cycle: integer("cycle").notNull(),
  bestTc: real("best_tc"),
  bestScore: real("best_score"),
  avgTopScore: real("avg_top_score"),
  candidatesTotal: integer("candidates_total"),
  pipelinePassRate: real("pipeline_pass_rate"),
  novelInsightCount: integer("novel_insight_count"),
  topFormula: text("top_formula"),
  strategyFocus: text("strategy_focus"),
  familyDiversity: integer("family_diversity"),
  duplicatesSkipped: integer("duplicates_skipped"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertElementSchema = createInsertSchema(elements);
export const insertMaterialSchema = createInsertSchema(materials).omit({ learnedAt: true });
export const insertLearningPhaseSchema = createInsertSchema(learningPhases).omit({ startedAt: true, completedAt: true });
export const insertNovelPredictionSchema = createInsertSchema(novelPredictions).omit({ predictedAt: true });
export const insertResearchLogSchema = createInsertSchema(researchLogs).omit({ id: true, timestamp: true });
export const insertSynthesisProcessSchema = createInsertSchema(synthesisProcesses).omit({ discoveredAt: true });
export const insertChemicalReactionSchema = createInsertSchema(chemicalReactions).omit({ learnedAt: true });
export const insertSuperconductorCandidateSchema = createInsertSchema(superconductorCandidates).omit({ generatedAt: true });
export const insertCrystalStructureSchema = createInsertSchema(crystalStructures).omit({ predictedAt: true });
export const insertComputationalResultSchema = createInsertSchema(computationalResults).omit({ computedAt: true });
export const insertNovelInsightSchema = createInsertSchema(novelInsights).omit({ discoveredAt: true });
export const insertResearchStrategySchema = createInsertSchema(researchStrategies).omit({ createdAt: true });
export const insertConvergenceSnapshotSchema = createInsertSchema(convergenceSnapshots).omit({ createdAt: true });

export type Element = typeof elements.$inferSelect;
export type Material = typeof materials.$inferSelect;
export type LearningPhase = typeof learningPhases.$inferSelect;
export type NovelPrediction = typeof novelPredictions.$inferSelect;
export type ResearchLog = typeof researchLogs.$inferSelect;
export type SynthesisProcess = typeof synthesisProcesses.$inferSelect;
export type ChemicalReaction = typeof chemicalReactions.$inferSelect;
export type SuperconductorCandidate = typeof superconductorCandidates.$inferSelect;
export type CrystalStructure = typeof crystalStructures.$inferSelect;
export type ComputationalResult = typeof computationalResults.$inferSelect;
export type NovelInsight = typeof novelInsights.$inferSelect;
export type ResearchStrategy = typeof researchStrategies.$inferSelect;
export type ConvergenceSnapshot = typeof convergenceSnapshots.$inferSelect;

export type InsertElement = z.infer<typeof insertElementSchema>;
export type InsertMaterial = z.infer<typeof insertMaterialSchema>;
export type InsertLearningPhase = z.infer<typeof insertLearningPhaseSchema>;
export type InsertNovelPrediction = z.infer<typeof insertNovelPredictionSchema>;
export type InsertResearchLog = z.infer<typeof insertResearchLogSchema>;
export type InsertSynthesisProcess = z.infer<typeof insertSynthesisProcessSchema>;
export type InsertChemicalReaction = z.infer<typeof insertChemicalReactionSchema>;
export type InsertSuperconductorCandidate = z.infer<typeof insertSuperconductorCandidateSchema>;
export type InsertCrystalStructure = z.infer<typeof insertCrystalStructureSchema>;
export type InsertComputationalResult = z.infer<typeof insertComputationalResultSchema>;
export type InsertNovelInsight = z.infer<typeof insertNovelInsightSchema>;
export type InsertResearchStrategy = z.infer<typeof insertResearchStrategySchema>;
export type InsertConvergenceSnapshot = z.infer<typeof insertConvergenceSnapshotSchema>;

export * from "./models/chat";
