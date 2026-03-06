# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform dedicated to materials science research, specifically focused on accelerating the discovery of room-temperature superconductors. It integrates AI for natural language processing, novel material generation, and machine learning predictions. The platform aims to revolutionize material discovery, with significant market potential across high-tech industries, by continuously learning and expanding its knowledge base from subatomic modeling to advanced computational physics and synthesis tracking.

## User Preferences
- No emoji in UI text
- Use FileText icon (not ScrollText) throughout
- NEVER use "breakthrough" or "confirmed" status for superconductor predictions – all are theoretical until lab verification.

## System Architecture

### Stack
- **Frontend**: React, Vite, TanStack Query, wouter
- **Backend**: Express.js, TypeScript
- **Database**: PostgreSQL (via Drizzle ORM)
- **UI**: shadcn/ui, Tailwind CSS, Recharts
- **AI**: OpenAI gpt-4o-mini (via Replit AI Integrations)
- **Real-time**: WebSocket (ws)

### Core Features
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases covering subatomic to multi-fidelity screening and novel synthesis reasoning, with continuous cycling and balanced priority.
- **ML Prediction Engine**: Employs a trained gradient boosting model (depth-4 decision trees, ~106 trees, R2=1.000) combined with an OpenAI gpt-4o-mini neural network ensemble. It uses physics-informed features and strict room-temperature criteria (Tc >= 293K) for superconductor scoring.
- **SC Verification Pipeline**: A multi-step process for categorizing superconductor candidates from theoretical to `requires-verification`, preventing premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline with calibrated thresholds for metallicity, density of states, lambda, Tc, and synthesis feasibility.
- **Novel Insight Detection**: Pre-filters known knowledge, deduplicates similar insights, and uses OpenAI to evaluate remaining insights for novelty against an expanded list of well-known patterns.
- **Data Confidence Tracking**: Candidates are tagged with `dataConfidence` (high, medium, low) based on their origin (DFT, model, LLM-estimated), displayed in the UI.
- **LLM Data Validation**: Properties from LLM are cross-validated against Materials Project API, enforcing physical bounds for band gap, formation energy, lattice parameters, and density.
- **Physics as Tc Authority**: All Tc values are grounded in Eliashberg/McMillan physics. The system includes ambient-pressure Tc caps and ensures Tc only changes when physics inputs change.
- **Diversity & Deduplication**: Mechanisms are in place to ensure diversity in formula generation and prevent redundant work in SC candidate processing.
- **Evolving Research Strategy**: The system analyzes top SC candidates and uses OpenAI to generate ranked focus areas to bias the formula generator.
- **Milestone Detection System**: Detects various research milestones such as new-family discoveries, Tc records, and pipeline graduations.
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters like Hc2, coherence length, pressure, and defines strict criteria for `roomTempViable` candidates.
- **Stagnation Breaking**: The system detects research stagnation and triggers re-analysis of high-lambda candidates and adapts candidate generation strategies via LLM prompts.

### Physics Engine
- **Deterministic Calculations**: All physics calculations are fully deterministic, with no randomness.
- **Comprehensive Elemental Data**: Utilizes a database of 96 elements with tabulated properties.
- **Advanced Parameter Calculation**: Includes methods for calculating DOS at Fermi level, bandwidth, Hubbard U/W, lambda, phonon frequencies, and metallicity using various models and lookups.
- **Specialized Material Handling**: Incorporates specific physics adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductor mechanisms.
- **Soft Empirical Limits (v8)**: Empirical Tc caps converted from hard walls to soft ceilings via `softCeiling()`. Lambda soft-capped above 4.0 (max 6.0). McMillan cap 600K soft ceiling. `PhysicsConstraintMode` allows toggling between conservative/exploratory. Material-class bonuses now physics-derived from composition.
- **Multiple Pairing Mechanisms**: `runUnifiedPairingAnalysis()` evaluates BCS, spin-fluctuation, excitonic, plasmonic, and flat-band mechanisms. Dominant mechanism determines Tc.
- **Instability Proximity Scoring**: `computeInstabilityProximity()` scores materials near magnetic QCP, structural boundary, metal-insulator transition, CDW instability, and soft phonon collapse. Boundary-proximate materials get priority.
- **Inverse Design**: `generateInverseDesignCandidates()` runs every 5th cycle, designing materials optimized for pairing susceptibility (lambda, DOS, nesting) rather than just Tc.
- **Generative Crystal Structures**: `runGenerativeStructureDiscovery()` generates structural variants via chemical substitution, intercalation, and topology mapping (Kagome, pyrochlore, etc.) every 3rd cycle.
- **Boundary Hunting Mode**: Activated on stagnation (>5 cycles without Tc improvement). Formula generator targets instability edges. Combined with inverse design mode.
- **Enriched ML Features**: 34 features (was 28): orbital character, phonon spectral centroid/width, bond stiffness variance, charge transfer magnitude, connectivity index.
- **Structure Predictor**: Uses crystallographic rules, Goldschmidt tolerance factor, Vegard's law, and Tammann rule for structure and synthesis temperature prediction.
- **Novel Synthesis Reasoning (Phase 13)**: Physics-constrained synthesis path generator in `server/learning/synthesis-reasoning.ts`. Analyzes thermodynamic landscape (hull distance, formation energy, decomposition barriers) and proposes non-equilibrium processing routes for metastable candidates. Material-class-specific routes: hydrides (DAC hydrogenation), HEAs (mechanical alloying + SPS, combinatorial sputtering), cuprates (sol-gel + O2 annealing), plus rapid quench, thin film, and reactive milling routes. Routes tagged `source: "physics-reasoned"` vs `"literature-based"`. Runs after DFT enrichment, limited to 3 candidates/cycle.
- **DFT Feature Resolver**: A unified service merging Materials Project and AFLOW DFT data, providing DFT-resolved features for ML prediction and physics analysis.
- **Auto-DFT Enrichment**: An engine cycle mechanism to enrich superconductor candidates with DFT data based on score, pipeline stage, and staleness.

### Gradient Boosting Model
- Trained on 130+ known superconductors from the SuperCon database.
- Features include physical properties like lambda, metallicity, omegaLog, debyeTemp, and various compositional and structural indicators.

### Alive Engine
- **Real-time Feedback**: Broadcasts `thought` WebSocket messages at key decision points, providing a live-scrolling thought feed on the dashboard.
- **Adaptive Tempo**: Adjusts the cycle interval based on research progress (excited, exploring, contemplating states).
- **Research Memory**: Aggregates knowledge summaries including current hypotheses, family statistics, and abandoned strategies.
- **Dynamic Status Messages**: Displays contextual status messages in the sidebar based on engine tempo and activity.

### Infrastructure
- **Rate Limiting**: Implements `express-rate-limit` for API request control.
- **In-Memory Cache**: Utilizes a TTL cache for frequently accessed data like elements, stats, and computational results.
- **DB Indexes**: Optimized database queries with indexes on key columns such as formula, timestamp, predictedTc, and pipelineStage.
- **API Pagination**: Supports pagination for novel-predictions endpoints.
- **Experimental Validations**: Tracks experimental validation results for superconductor candidates.
- **ML Calibration**: Provides confidence bands and error bars for Tc predictions based on gradient-boost training residuals.
- **Cross-Validation**: Integrates AFLOW REST API and enhanced Materials Project queries for external data validation.
- **Interactive Periodic Table**: Offers an interactive UI component linking elements to materials and candidates.

## External Dependencies
- **OpenAI**: For gpt-4o-mini, used in NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.