-- Add DMFT pipeline columns to quantum_engine_dataset
-- These columns store results from the DMFT service (TRIQS/CTHYB + DCA)

ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_bundle_exported" boolean;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_hamiltonian_parsed" boolean;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_correlated_shells" integer;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_correlated_orbitals" integer;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_job_id" text;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_converged" boolean;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_lambda_pair" real;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_gap_symmetry" text;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_gap_nodes" text;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_is_unconventional" boolean;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_tc_bse" real;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_tc_bse_confidence" text;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_dominant_channel" text;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_cluster_size" integer;--> statement-breakpoint
ALTER TABLE "quantum_engine_dataset" ADD COLUMN IF NOT EXISTS "dmft_avg_sign" real;
