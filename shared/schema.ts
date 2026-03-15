import { pgTable, text, varchar, integer, real, boolean, timestamp, jsonb, uniqueIndex, index } from "drizzle-orm/pg-core";
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
  formula: text("formula").notNull().unique(),
  predictedProperties: jsonb("predicted_properties").notNull(),
  confidence: real("confidence").notNull(),
  targetApplication: text("target_application").notNull(),
  status: text("status").notNull().default("predicted"),
  notes: text("notes"),
  signalMetadata: jsonb("signal_metadata"),
  predictedAt: timestamp("predicted_at").defaultNow(),
});

export const researchLogs = pgTable("research_logs", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  timestamp: timestamp("timestamp").defaultNow(),
  phase: text("phase").notNull(),
  event: text("event").notNull(),
  detail: text("detail"),
  dataSource: text("data_source"),
}, (table) => [
  index("research_logs_timestamp_idx").on(table.timestamp),
  index("research_logs_event_idx").on(table.event),
]);

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
}, (table) => [
  index("synthesis_processes_formula_idx").on(table.formula),
]);

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
  formula: text("formula").notNull().unique(),
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
  discoveryScore: real("discovery_score"),
  dataConfidence: text("data_confidence").default("medium"),
}, (table) => [
  index("sc_candidates_predicted_tc_idx").on(table.predictedTc),
  index("sc_candidates_ensemble_score_idx").on(table.ensembleScore),
  index("sc_candidates_data_confidence_idx").on(table.dataConfidence),
]);

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
  isGroundTruth: boolean("is_ground_truth").default(false),
  predictedAt: timestamp("predicted_at").defaultNow(),
}, (table) => [
  index("crystal_structures_formula_idx").on(table.formula),
]);

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
}, (table) => [
  index("computational_results_formula_idx").on(table.formula),
  index("computational_results_pipeline_stage_idx").on(table.pipelineStage),
]);

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
}, (table) => [
  index("novel_insights_discovered_at_idx").on(table.discoveredAt),
  index("novel_insights_is_novel_idx").on(table.isNovel, table.discoveredAt),
]);

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
  bestPhysicsTc: real("best_physics_tc"),
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

export const milestones = pgTable("milestones", {
  id: varchar("id").primaryKey(),
  cycle: integer("cycle").notNull(),
  type: varchar("type", { length: 50 }).notNull(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  significance: integer("significance").notNull(),
  relatedFormula: text("related_formula"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const mpMaterialCache = pgTable("mp_material_cache", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  formula: text("formula").notNull(),
  mpId: text("mp_id"),
  dataType: text("data_type").notNull(),
  data: jsonb("data").notNull(),
  fetchedAt: timestamp("fetched_at").defaultNow(),
}, (table) => [
  uniqueIndex("mp_cache_formula_type_idx").on(table.formula, table.dataType),
]);

export const insertMpMaterialCacheSchema = createInsertSchema(mpMaterialCache).omit({ fetchedAt: true } as any);
export type MpMaterialCache = typeof mpMaterialCache.$inferSelect;
export type InsertMpMaterialCache = z.infer<typeof insertMpMaterialCacheSchema>;

export const dftJobs = pgTable("dft_jobs", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  formula: text("formula").notNull(),
  candidateId: integer("candidate_id"),
  status: varchar("status", { length: 20 }).notNull().default("queued"),
  jobType: varchar("job_type", { length: 20 }).notNull().default("scf"),
  priority: integer("priority").notNull().default(50),
  inputData: jsonb("input_data"),
  outputData: jsonb("output_data"),
  errorMessage: text("error_message"),
  workerNode: varchar("worker_node", { length: 20 }),  // "local" | "gcp"
  createdAt: timestamp("created_at").defaultNow(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
}, (table) => [
  index("dft_jobs_status_priority_idx").on(table.status, table.priority),
  index("dft_jobs_formula_idx").on(table.formula),
]);

export const insertDftJobSchema = createInsertSchema(dftJobs).omit({ id: true, createdAt: true, startedAt: true, completedAt: true });
export type DftJob = typeof dftJobs.$inferSelect;

// GNN training jobs dispatched to the GCP worker
export const gnnTrainingJobs = pgTable("gnn_training_jobs", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  status: varchar("status", { length: 20 }).notNull().default("queued"), // queued | running | done | failed
  trainingData: jsonb("training_data").notNull(), // TrainingSample[]
  weights: jsonb("weights"),                      // GNNWeights[] when done
  r2: real("r2"),
  mae: real("mae"),
  rmse: real("rmse"),
  datasetSize: integer("dataset_size"),
  dftSamples: integer("dft_samples"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
}, (table) => [
  index("gnn_training_jobs_status_idx").on(table.status, table.createdAt),
]);

export const insertGnnTrainingJobSchema = createInsertSchema(gnnTrainingJobs).omit({ id: true, createdAt: true, startedAt: true, completedAt: true });
export type GnnTrainingJob = typeof gnnTrainingJobs.$inferSelect;
export type InsertGnnTrainingJob = z.infer<typeof insertGnnTrainingJobSchema>;
export type InsertDftJob = z.infer<typeof insertDftJobSchema>;

export const xgbTrainingJobs = pgTable("xgb_training_jobs", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  status: varchar("status", { length: 20 }).notNull().default("queued"), // queued | running | done | failed
  featuresX: jsonb("features_x").notNull(),   // number[][] training features
  labelsY: jsonb("labels_y").notNull(),        // number[] training labels
  datasetSize: integer("dataset_size"),
  model: jsonb("model"),                        // GBModel when done
  ensembleXGB: jsonb("ensemble_xgb"),           // GBEnsemble (mean) when done
  varianceEnsembleXGB: jsonb("variance_ensemble_xgb"), // GBEnsemble (variance) when done
  r2: real("r2"),
  mae: real("mae"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
}, (table) => [
  index("xgb_training_jobs_status_idx").on(table.status, table.createdAt),
]);

export const insertXgbTrainingJobSchema = createInsertSchema(xgbTrainingJobs).omit({ id: true, createdAt: true, startedAt: true, completedAt: true });
export type XgbTrainingJob = typeof xgbTrainingJobs.$inferSelect;
export type InsertXgbTrainingJob = z.infer<typeof insertXgbTrainingJobSchema>;

// ML model training jobs — offloaded to GCP (crystal VAE, diffusion model, materials VAE)
export const mlTrainingJobs = pgTable("ml_training_jobs", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  taskType: varchar("task_type", { length: 50 }).notNull(), // 'train-materials-vae' | 'init-crystal-vae' | 'init-diffusion-model'
  status: varchar("status", { length: 20 }).notNull().default("queued"), // queued | running | done | failed
  inputData: jsonb("input_data"),       // { formulas?: string[], epochs?: number, dataset?: CrystalStructureEntry[] }
  outputWeights: jsonb("output_weights"), // serialized trained weights/state
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
}, (table) => [
  index("ml_training_jobs_status_idx").on(table.status, table.taskType, table.createdAt),
]);

export const insertMlTrainingJobSchema = createInsertSchema(mlTrainingJobs).omit({ id: true, createdAt: true, startedAt: true, completedAt: true });
export type MlTrainingJob = typeof mlTrainingJobs.$inferSelect;
export type InsertMlTrainingJob = z.infer<typeof insertMlTrainingJobSchema>;

export const experimentalValidations = pgTable("experimental_validations", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  formula: text("formula").notNull(),
  validationType: text("validation_type").notNull(),
  result: text("result").notNull(),
  measuredTc: real("measured_tc"),
  measuredPressure: real("measured_pressure"),
  notes: text("notes"),
  performedAt: timestamp("performed_at").defaultNow(),
}, (table) => [
  index("experimental_validations_formula_idx").on(table.formula),
]);

export const insertExperimentalValidationSchema = createInsertSchema(experimentalValidations).omit({ id: true, performedAt: true });
export type ExperimentalValidation = typeof experimentalValidations.$inferSelect;
export type InsertExperimentalValidation = z.infer<typeof insertExperimentalValidationSchema>;

export const insertElementSchema = createInsertSchema(elements);
export const insertMaterialSchema = createInsertSchema(materials).omit({ learnedAt: true });
export const insertLearningPhaseSchema = createInsertSchema(learningPhases).omit({ startedAt: true, completedAt: true });
export const insertNovelPredictionSchema = createInsertSchema(novelPredictions).omit({ predictedAt: true });
export const insertResearchLogSchema = createInsertSchema(researchLogs).omit({ id: true, timestamp: true });
export const inverseDesignCampaigns = pgTable("inverse_design_campaigns", {
  id: varchar("id").primaryKey(),
  targetTc: real("target_tc").notNull(),
  maxPressure: real("max_pressure").notNull().default(1),
  minLambda: real("min_lambda").notNull().default(1.5),
  maxHullDistance: real("max_hull_distance").notNull().default(0.05),
  metallicRequired: boolean("metallic_required").notNull().default(true),
  phononStable: boolean("phonon_stable").notNull().default(true),
  preferredPrototypes: text("preferred_prototypes").array(),
  preferredElements: text("preferred_elements").array(),
  excludeElements: text("exclude_elements").array(),
  status: text("status").notNull().default("active"),
  cyclesRun: integer("cycles_run").notNull().default(0),
  bestTcAchieved: real("best_tc_achieved").notNull().default(0),
  bestDistance: real("best_distance").notNull().default(1.0),
  candidatesGenerated: integer("candidates_generated").notNull().default(0),
  candidatesPassedPipeline: integer("candidates_passed_pipeline").notNull().default(0),
  learningState: jsonb("learning_state"),
  convergenceHistory: jsonb("convergence_history"),
  topCandidates: jsonb("top_candidates"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertInverseDesignCampaignSchema = createInsertSchema(inverseDesignCampaigns).omit({ createdAt: true });

export const quantumEngineDataset = pgTable("quantum_engine_dataset", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  material: text("material").notNull(),
  pressure: real("pressure").notNull().default(0),
  lambda: real("lambda").notNull().default(0),
  omegaLog: real("omega_log").notNull().default(0),
  tc: real("tc").notNull().default(0),
  dosAtEF: real("dos_at_ef").notNull().default(0),
  phononSpectrum: jsonb("phonon_spectrum").notNull().default([]),
  alpha2FSummary: jsonb("alpha2f_summary").notNull().default({}),
  formationEnergy: real("formation_energy"),
  bandGap: real("band_gap"),
  isMetallic: boolean("is_metallic").notNull().default(false),
  isPhononStable: boolean("is_phonon_stable").notNull().default(false),
  scfConverged: boolean("scf_converged").notNull().default(false),
  gapRatio: real("gap_ratio").notNull().default(3.53),
  muStar: real("mu_star").notNull().default(0.1),
  omega2: real("omega2").notNull().default(0),
  isStrongCoupling: boolean("is_strong_coupling").notNull().default(false),
  isotopeAlpha: real("isotope_alpha").notNull().default(0.5),
  tcAllenDynes: real("tc_allen_dynes").notNull().default(0),
  tcEliashberg: real("tc_eliashberg").notNull().default(0),
  confidence: text("confidence").notNull().default("low"),
  tier: text("tier").notNull().default("surrogate"),
  wallTimeMs: real("wall_time_ms").notNull().default(0),
  dosPrefilter: jsonb("dos_prefilter"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  index("qe_dataset_material_idx").on(table.material),
  index("qe_dataset_tc_idx").on(table.tc),
  index("qe_dataset_created_at_idx").on(table.createdAt),
]);

export const insertQuantumEngineDatasetSchema = createInsertSchema(quantumEngineDataset).omit({ id: true, createdAt: true });
export type QuantumEngineDatasetRow = typeof quantumEngineDataset.$inferSelect;
export type InsertQuantumEngineDatasetRow = z.infer<typeof insertQuantumEngineDatasetSchema>;

export const insertSynthesisProcessSchema = createInsertSchema(synthesisProcesses).omit({ discoveredAt: true });
export const insertChemicalReactionSchema = createInsertSchema(chemicalReactions).omit({ learnedAt: true });
export const insertSuperconductorCandidateSchema = createInsertSchema(superconductorCandidates).omit({ generatedAt: true });
export const insertCrystalStructureSchema = createInsertSchema(crystalStructures).omit({ predictedAt: true });
export const insertComputationalResultSchema = createInsertSchema(computationalResults).omit({ computedAt: true });
export const insertNovelInsightSchema = createInsertSchema(novelInsights).omit({ discoveredAt: true });
export const insertResearchStrategySchema = createInsertSchema(researchStrategies).omit({ createdAt: true });
export const insertConvergenceSnapshotSchema = createInsertSchema(convergenceSnapshots).omit({ createdAt: true });
export const insertMilestoneSchema = createInsertSchema(milestones).omit({ createdAt: true });

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
export type Milestone = typeof milestones.$inferSelect;

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
export type InsertMilestone = z.infer<typeof insertMilestoneSchema>;
export type InverseDesignCampaign = typeof inverseDesignCampaigns.$inferSelect;
export type InsertInverseDesignCampaign = z.infer<typeof insertInverseDesignCampaignSchema>;

export const systemMetrics = pgTable("system_metrics", {
  id: integer("id").primaryKey().generatedAlwaysAsIdentity(),
  metricName: text("metric_name").notNull(),
  metricValue: real("metric_value").notNull(),
  metadata: jsonb("metadata"),
  recordedAt: timestamp("recorded_at").defaultNow(),
});

export const insertSystemMetricSchema = createInsertSchema(systemMetrics).omit({ id: true, recordedAt: true });
export type SystemMetric = typeof systemMetrics.$inferSelect;
export type InsertSystemMetric = z.infer<typeof insertSystemMetricSchema>;

export const systemState = pgTable("system_state", {
  key: text("key").primaryKey(),
  value: jsonb("value").notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export type SystemState = typeof systemState.$inferSelect;

export * from "./models/chat";
