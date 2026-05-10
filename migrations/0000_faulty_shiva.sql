CREATE TABLE "chemical_reactions" (
	"id" varchar PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"equation" text NOT NULL,
	"reaction_type" text NOT NULL,
	"reactants" jsonb NOT NULL,
	"products" jsonb NOT NULL,
	"conditions" jsonb NOT NULL,
	"energetics" jsonb,
	"mechanism" text,
	"relevance_to_superconductor" real DEFAULT 0,
	"lab_process" text,
	"source" text,
	"learned_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "client_errors" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "client_errors_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"timestamp" timestamp DEFAULT now(),
	"type" text NOT NULL,
	"page" text,
	"message" text NOT NULL,
	"stack" text,
	"endpoint" text,
	"status_code" integer,
	"duration_ms" integer,
	"build_info" text
);
--> statement-breakpoint
CREATE TABLE "cod_structure_cache" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "cod_structure_cache_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"cod_id" integer NOT NULL,
	"formula" text NOT NULL,
	"space_group_number" integer NOT NULL,
	"space_group_symbol" text,
	"crystal_system" text,
	"elements" text[] NOT NULL,
	"a" real,
	"b" real,
	"c" real,
	"alpha" real,
	"beta" real,
	"gamma" real,
	"volume_per_atom" real,
	"raw_data" jsonb,
	"fetched_at" timestamp DEFAULT now(),
	CONSTRAINT "cod_structure_cache_cod_id_unique" UNIQUE("cod_id")
);
--> statement-breakpoint
CREATE TABLE "computational_results" (
	"id" varchar PRIMARY KEY NOT NULL,
	"candidate_id" varchar,
	"formula" text NOT NULL,
	"computation_type" text NOT NULL,
	"pipeline_stage" integer DEFAULT 0 NOT NULL,
	"input_params" jsonb,
	"results" jsonb NOT NULL,
	"confidence" real,
	"compute_time_ms" integer,
	"passed" boolean DEFAULT false,
	"failure_reason" text,
	"computed_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "convergence_snapshots" (
	"id" varchar PRIMARY KEY NOT NULL,
	"cycle" integer NOT NULL,
	"best_tc" real,
	"best_physics_tc" real,
	"best_score" real,
	"avg_top_score" real,
	"avg_top10_tc" real,
	"dft_selected_tc" real,
	"candidates_total" integer,
	"pipeline_pass_rate" real,
	"novel_insight_count" integer,
	"top_formula" text,
	"strategy_focus" text,
	"family_diversity" integer,
	"duplicates_skipped" integer,
	"r2_score" real,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "cross_engine_insights_log" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "cross_engine_insights_log_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"engine" text NOT NULL,
	"insight_data" jsonb NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "crystal_structures" (
	"id" varchar PRIMARY KEY NOT NULL,
	"formula" text NOT NULL,
	"space_group" text NOT NULL,
	"crystal_system" text NOT NULL,
	"lattice_params" jsonb,
	"atomic_positions" jsonb,
	"prototype" text,
	"dimensionality" text DEFAULT '3D' NOT NULL,
	"is_stable" boolean DEFAULT false,
	"is_metastable" boolean DEFAULT false,
	"decomposition_energy" real,
	"convex_hull_distance" real,
	"synthesizability" real,
	"synthesis_notes" text,
	"source" text,
	"is_ground_truth" boolean DEFAULT false,
	"predicted_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "cycle_diagnostic_reports_log" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "cycle_diagnostic_reports_log_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"cycle" integer NOT NULL,
	"report_data" jsonb NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "dft_jobs" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "dft_jobs_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"candidate_id" integer,
	"status" varchar(20) DEFAULT 'queued' NOT NULL,
	"job_type" varchar(20) DEFAULT 'scf' NOT NULL,
	"priority" integer DEFAULT 50 NOT NULL,
	"input_data" jsonb,
	"output_data" jsonb,
	"error_message" text,
	"worker_node" varchar(20),
	"created_at" timestamp DEFAULT now(),
	"started_at" timestamp,
	"completed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "elements" (
	"id" integer PRIMARY KEY NOT NULL,
	"symbol" varchar(4) NOT NULL,
	"name" text NOT NULL,
	"atomic_mass" real,
	"period" integer,
	"group_num" integer,
	"category" text,
	"electronegativity" real,
	"electron_config" text,
	"melting_point" real,
	"boiling_point" real,
	"density" real,
	"discovered_year" integer,
	"description" text
);
--> statement-breakpoint
CREATE TABLE "engine_insights_log" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "engine_insights_log_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"insight_text" text NOT NULL,
	"cycle" integer,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "experimental_validations" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "experimental_validations_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"validation_type" text NOT NULL,
	"result" text NOT NULL,
	"measured_tc" real,
	"measured_pressure" real,
	"notes" text,
	"performed_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "formula_screen_log" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "formula_screen_log_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"status" text NOT NULL,
	"reason" text,
	"tc" real,
	"lambda" real,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "gnn_training_jobs" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "gnn_training_jobs_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"status" varchar(20) DEFAULT 'queued' NOT NULL,
	"training_data" jsonb NOT NULL,
	"weights" jsonb,
	"r2" real,
	"mae" real,
	"rmse" real,
	"train_r2" real,
	"train_mae" real,
	"val_n" integer,
	"dataset_size" integer,
	"dft_samples" integer,
	"error_message" text,
	"created_at" timestamp DEFAULT now(),
	"started_at" timestamp,
	"completed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "inverse_design_campaigns" (
	"id" varchar PRIMARY KEY NOT NULL,
	"target_tc" real NOT NULL,
	"max_pressure" real DEFAULT 1 NOT NULL,
	"min_lambda" real DEFAULT 1.5 NOT NULL,
	"max_hull_distance" real DEFAULT 0.05 NOT NULL,
	"metallic_required" boolean DEFAULT true NOT NULL,
	"phonon_stable" boolean DEFAULT true NOT NULL,
	"preferred_prototypes" text[],
	"preferred_elements" text[],
	"exclude_elements" text[],
	"status" text DEFAULT 'active' NOT NULL,
	"cycles_run" integer DEFAULT 0 NOT NULL,
	"best_tc_achieved" real DEFAULT 0 NOT NULL,
	"best_distance" real DEFAULT 1 NOT NULL,
	"candidates_generated" integer DEFAULT 0 NOT NULL,
	"candidates_passed_pipeline" integer DEFAULT 0 NOT NULL,
	"learning_state" jsonb,
	"convergence_history" jsonb,
	"top_candidates" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "learning_phases" (
	"id" integer PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"description" text NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"progress" real DEFAULT 0 NOT NULL,
	"items_learned" integer DEFAULT 0 NOT NULL,
	"total_items" integer DEFAULT 0 NOT NULL,
	"started_at" timestamp,
	"completed_at" timestamp,
	"insights" text[]
);
--> statement-breakpoint
CREATE TABLE "materials" (
	"id" varchar PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"formula" text NOT NULL,
	"spacegroup" text,
	"band_gap" real,
	"formation_energy" real,
	"stability" real,
	"source" text NOT NULL,
	"properties" jsonb,
	"learned_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "milestones" (
	"id" varchar PRIMARY KEY NOT NULL,
	"cycle" integer NOT NULL,
	"type" varchar(50) NOT NULL,
	"title" text NOT NULL,
	"description" text NOT NULL,
	"significance" integer NOT NULL,
	"related_formula" text,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "ml_training_jobs" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "ml_training_jobs_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"task_type" varchar(50) NOT NULL,
	"status" varchar(20) DEFAULT 'queued' NOT NULL,
	"input_data" jsonb,
	"output_weights" jsonb,
	"error_message" text,
	"created_at" timestamp DEFAULT now(),
	"started_at" timestamp,
	"completed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "mp_material_cache" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "mp_material_cache_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"mp_id" text,
	"data_type" text NOT NULL,
	"data" jsonb NOT NULL,
	"fetched_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "novel_insights" (
	"id" varchar PRIMARY KEY NOT NULL,
	"phase_id" integer NOT NULL,
	"phase_name" text NOT NULL,
	"insight_text" text NOT NULL,
	"is_novel" boolean DEFAULT false,
	"novelty_score" real,
	"novelty_reason" text,
	"category" text,
	"related_formulas" text[],
	"discovered_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "novel_predictions" (
	"id" varchar PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"formula" text NOT NULL,
	"predicted_properties" jsonb NOT NULL,
	"confidence" real NOT NULL,
	"target_application" text NOT NULL,
	"status" text DEFAULT 'predicted' NOT NULL,
	"notes" text,
	"signal_metadata" jsonb,
	"predicted_at" timestamp DEFAULT now(),
	CONSTRAINT "novel_predictions_formula_unique" UNIQUE("formula")
);
--> statement-breakpoint
CREATE TABLE "pressure_observations_log" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "pressure_observations_log_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"pressure_gpa" real NOT NULL,
	"tc" real NOT NULL,
	"stable" boolean,
	"enthalpy" real,
	"created_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "quantum_engine_dataset" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "quantum_engine_dataset_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"material" text NOT NULL,
	"pressure" real DEFAULT 0 NOT NULL,
	"lambda" real DEFAULT 0 NOT NULL,
	"omega_log" real DEFAULT 0 NOT NULL,
	"tc" real DEFAULT 0 NOT NULL,
	"dos_at_ef" real DEFAULT 0 NOT NULL,
	"phonon_spectrum" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"alpha2f_summary" jsonb DEFAULT '{}'::jsonb NOT NULL,
	"formation_energy" real,
	"band_gap" real,
	"is_metallic" boolean DEFAULT false NOT NULL,
	"is_phonon_stable" boolean DEFAULT false NOT NULL,
	"scf_converged" boolean DEFAULT false NOT NULL,
	"gap_ratio" real DEFAULT 3.53 NOT NULL,
	"mu_star" real DEFAULT 0.1 NOT NULL,
	"omega2" real DEFAULT 0 NOT NULL,
	"is_strong_coupling" boolean DEFAULT false NOT NULL,
	"isotope_alpha" real DEFAULT 0.5 NOT NULL,
	"tc_allen_dynes" real DEFAULT 0 NOT NULL,
	"tc_eliashberg" real DEFAULT 0 NOT NULL,
	"confidence" text DEFAULT 'low' NOT NULL,
	"tier" text DEFAULT 'surrogate' NOT NULL,
	"wall_time_ms" real DEFAULT 0 NOT NULL,
	"dos_prefilter" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "research_logs" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "research_logs_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"timestamp" timestamp DEFAULT now(),
	"phase" text NOT NULL,
	"event" text NOT NULL,
	"detail" text,
	"data_source" text
);
--> statement-breakpoint
CREATE TABLE "research_strategies" (
	"id" varchar PRIMARY KEY NOT NULL,
	"cycle" integer NOT NULL,
	"focus_areas" jsonb NOT NULL,
	"summary" text NOT NULL,
	"performance_signals" jsonb,
	"created_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "supercon_external_entries" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "supercon_external_entries_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"formula" text NOT NULL,
	"tc" real,
	"is_superconductor" boolean DEFAULT true NOT NULL,
	"source" varchar(40) NOT NULL,
	"external_id" text,
	"space_group" text,
	"crystal_system" text,
	"family" text,
	"lambda" real,
	"pressure_gpa" real DEFAULT 0 NOT NULL,
	"raw_data" jsonb,
	"imported_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "superconductor_candidates" (
	"id" varchar PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"formula" text NOT NULL,
	"predicted_tc" real,
	"pressure_gpa" real,
	"pressure_assumed" boolean DEFAULT false,
	"meissner_effect" boolean DEFAULT false,
	"zero_resistance" boolean DEFAULT false,
	"cooper_pair_mechanism" text,
	"crystal_structure" text,
	"quantum_coherence" real,
	"stability_score" real,
	"synthesis_path" jsonb,
	"ml_features" jsonb,
	"xgboost_score" real,
	"neural_net_score" real,
	"ensemble_score" real,
	"room_temp_viable" boolean DEFAULT false,
	"status" text DEFAULT 'theoretical' NOT NULL,
	"notes" text,
	"generated_at" timestamp DEFAULT now(),
	"electron_phonon_coupling" real,
	"log_phonon_frequency" real,
	"coulomb_pseudopotential" real,
	"pairing_symmetry" text,
	"pairing_mechanism" text,
	"competing_phases" jsonb,
	"upper_critical_field" real,
	"coherence_length" real,
	"london_penetration_depth" real,
	"anisotropy_ratio" real,
	"critical_current_density" real,
	"dimensionality" text,
	"fermi_surface_topology" text,
	"correlation_strength" real,
	"decomposition_energy" real,
	"ambient_pressure_stable" boolean DEFAULT false,
	"verification_stage" integer DEFAULT 0,
	"uncertainty_estimate" real,
	"discovery_score" real,
	"data_confidence" text DEFAULT 'medium',
	CONSTRAINT "superconductor_candidates_formula_unique" UNIQUE("formula")
);
--> statement-breakpoint
CREATE TABLE "synthesis_processes" (
	"id" varchar PRIMARY KEY NOT NULL,
	"material_id" varchar,
	"material_name" text NOT NULL,
	"formula" text NOT NULL,
	"method" text NOT NULL,
	"conditions" jsonb NOT NULL,
	"steps" text[] NOT NULL,
	"precursors" text[] NOT NULL,
	"equipment" text[],
	"difficulty" text DEFAULT 'moderate' NOT NULL,
	"time_estimate" text,
	"safety_notes" text,
	"yield_percent" real,
	"discovered_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "system_metrics" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "system_metrics_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"metric_name" text NOT NULL,
	"metric_value" real NOT NULL,
	"metadata" jsonb,
	"recorded_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "system_state" (
	"key" text PRIMARY KEY NOT NULL,
	"value" jsonb NOT NULL,
	"updated_at" timestamp DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "xgb_training_jobs" (
	"id" integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "xgb_training_jobs_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 2147483647 START WITH 1 CACHE 1),
	"status" varchar(20) DEFAULT 'queued' NOT NULL,
	"features_x" jsonb NOT NULL,
	"labels_y" jsonb NOT NULL,
	"dataset_size" integer,
	"model" jsonb,
	"ensemble_xgb" jsonb,
	"variance_ensemble_xgb" jsonb,
	"r2" real,
	"mae" real,
	"error_message" text,
	"created_at" timestamp DEFAULT now(),
	"started_at" timestamp,
	"completed_at" timestamp
);
--> statement-breakpoint
CREATE TABLE "conversations" (
	"id" serial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
CREATE TABLE "messages" (
	"id" serial PRIMARY KEY NOT NULL,
	"conversation_id" integer NOT NULL,
	"role" text NOT NULL,
	"content" text NOT NULL,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP NOT NULL
);
--> statement-breakpoint
ALTER TABLE "messages" ADD CONSTRAINT "messages_conversation_id_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."conversations"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "client_errors_timestamp_idx" ON "client_errors" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "client_errors_type_idx" ON "client_errors" USING btree ("type");--> statement-breakpoint
CREATE INDEX "cod_sg_number_idx" ON "cod_structure_cache" USING btree ("space_group_number");--> statement-breakpoint
CREATE INDEX "cod_crystal_system_idx" ON "cod_structure_cache" USING btree ("crystal_system");--> statement-breakpoint
CREATE INDEX "computational_results_formula_idx" ON "computational_results" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "computational_results_pipeline_stage_idx" ON "computational_results" USING btree ("pipeline_stage");--> statement-breakpoint
CREATE INDEX "cei_formula_idx" ON "cross_engine_insights_log" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "cei_engine_idx" ON "cross_engine_insights_log" USING btree ("engine");--> statement-breakpoint
CREATE INDEX "cei_created_idx" ON "cross_engine_insights_log" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "crystal_structures_formula_idx" ON "crystal_structures" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "cdrl_cycle_idx" ON "cycle_diagnostic_reports_log" USING btree ("cycle");--> statement-breakpoint
CREATE INDEX "cdrl_created_idx" ON "cycle_diagnostic_reports_log" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "dft_jobs_status_priority_idx" ON "dft_jobs" USING btree ("status","priority");--> statement-breakpoint
CREATE INDEX "dft_jobs_formula_idx" ON "dft_jobs" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "eil_created_idx" ON "engine_insights_log" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "eil_cycle_idx" ON "engine_insights_log" USING btree ("cycle");--> statement-breakpoint
CREATE INDEX "experimental_validations_formula_idx" ON "experimental_validations" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "fsl_formula_idx" ON "formula_screen_log" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "fsl_status_idx" ON "formula_screen_log" USING btree ("status");--> statement-breakpoint
CREATE INDEX "gnn_training_jobs_status_idx" ON "gnn_training_jobs" USING btree ("status","created_at");--> statement-breakpoint
CREATE INDEX "ml_training_jobs_status_idx" ON "ml_training_jobs" USING btree ("status","task_type","created_at");--> statement-breakpoint
CREATE UNIQUE INDEX "mp_cache_formula_type_idx" ON "mp_material_cache" USING btree ("formula","data_type");--> statement-breakpoint
CREATE INDEX "novel_insights_discovered_at_idx" ON "novel_insights" USING btree ("discovered_at");--> statement-breakpoint
CREATE INDEX "novel_insights_is_novel_idx" ON "novel_insights" USING btree ("is_novel","discovered_at");--> statement-breakpoint
CREATE INDEX "pol_formula_idx" ON "pressure_observations_log" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "pol_tc_idx" ON "pressure_observations_log" USING btree ("tc");--> statement-breakpoint
CREATE INDEX "qe_dataset_material_idx" ON "quantum_engine_dataset" USING btree ("material");--> statement-breakpoint
CREATE INDEX "qe_dataset_tc_idx" ON "quantum_engine_dataset" USING btree ("tc");--> statement-breakpoint
CREATE INDEX "qe_dataset_created_at_idx" ON "quantum_engine_dataset" USING btree ("created_at");--> statement-breakpoint
CREATE INDEX "research_logs_timestamp_idx" ON "research_logs" USING btree ("timestamp");--> statement-breakpoint
CREATE INDEX "research_logs_event_idx" ON "research_logs" USING btree ("event");--> statement-breakpoint
CREATE UNIQUE INDEX "supercon_ext_formula_source_idx" ON "supercon_external_entries" USING btree ("formula","source","external_id");--> statement-breakpoint
CREATE INDEX "supercon_ext_tc_idx" ON "supercon_external_entries" USING btree ("tc");--> statement-breakpoint
CREATE INDEX "supercon_ext_source_idx" ON "supercon_external_entries" USING btree ("source");--> statement-breakpoint
CREATE INDEX "sc_candidates_predicted_tc_idx" ON "superconductor_candidates" USING btree ("predicted_tc");--> statement-breakpoint
CREATE INDEX "sc_candidates_ensemble_score_idx" ON "superconductor_candidates" USING btree ("ensemble_score");--> statement-breakpoint
CREATE INDEX "sc_candidates_data_confidence_idx" ON "superconductor_candidates" USING btree ("data_confidence");--> statement-breakpoint
CREATE INDEX "sc_candidates_confidence_tc_idx" ON "superconductor_candidates" USING btree ("data_confidence","predicted_tc");--> statement-breakpoint
CREATE INDEX "synthesis_processes_formula_idx" ON "synthesis_processes" USING btree ("formula");--> statement-breakpoint
CREATE INDEX "xgb_training_jobs_status_idx" ON "xgb_training_jobs" USING btree ("status","created_at");