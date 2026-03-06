# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform aimed at accelerating the discovery of room-temperature superconductors. It integrates AI for natural language processing, novel material generation, and machine learning predictions. The platform is designed to revolutionize material discovery by continuously learning and expanding its knowledge base, from subatomic modeling to advanced computational physics and synthesis tracking, with significant market potential across high-tech industries.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases, covering subatomic to multi-fidelity screening and novel synthesis reasoning, with continuous cycling and balanced priority.
- **ML Prediction Engine**: Employs a trained gradient boosting model combined with an OpenAI gpt-4o-mini neural network ensemble. It uses physics-informed features and strict room-temperature criteria (Tc >= 293K) for superconductor scoring.
- **SC Verification Pipeline**: A multi-step process categorizes superconductor candidates from theoretical to `requires-verification`, preventing premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline with calibrated thresholds for key material properties.
- **Novel Insight Detection**: Pre-filters known knowledge, deduplicates similar insights, rejects vague content, requires quantitative data, and uses OpenAI to evaluate novelty.
- **Data Confidence Tracking**: Candidates are tagged with `dataConfidence` (high, medium, low) based on their origin.
- **LLM Data Validation**: Properties from LLM are cross-validated against Materials Project API, enforcing physical bounds.
- **Physics as Tc Authority**: All Tc values are grounded in Eliashberg/McMillan physics, with ambient-pressure Tc caps and updates only when physics inputs change.
- **Diversity & Deduplication**: Mechanisms ensure diversity in formula generation and prevent redundant work.
- **Evolving Research Strategy**: Utilizes Bayesian-adjusted family scoring and an exploit-then-explore policy with dynamic strategy switching.
- **Milestone Detection System**: Detects research milestones like new family discoveries, Tc records, and pipeline graduations.
- **Tc Plateau Escalation**: Triggers progressive strategies (boundary hunting, inverse design, wider element swaps, exotic substitutions, chemical space expansion) upon stagnation.
- **Stability Filtering**: Incorporates formation energy bounds, hull distance estimation, and a stability modifier in pairing scores.
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters and defines strict criteria for `roomTempViable` candidates.
- **Stagnation Breaking**: Detects research stagnation and triggers re-analysis and adaptation of candidate generation strategies via LLM prompts.

### Physics Engine
- **Deterministic Calculations**: All physics calculations are fully deterministic.
- **Comprehensive Elemental Data**: Utilizes a database of 96 elements with tabulated properties.
- **Advanced Parameter Calculation**: Includes methods for calculating DOS at Fermi level, bandwidth, Hubbard U/W, lambda, phonon frequencies, and metallicity.
- **Specialized Material Handling**: Incorporates specific physics adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductor mechanisms.
- **Material-Class-Aware Constraints**: Implements class-specific lambda caps and hydrogen bonding classification for accurate lambda boosting.
- **Grounded Allen-Dynes Inputs**: Clamps omega_log to class-appropriate ranges and widens mu* with corrections.
- **Pairing Susceptibility Optimization**: Optimizes for pairing conditions (DOS, nesting, coupling channels) rather than just raw Tc.
- **Multiple Pairing Mechanisms**: Evaluates BCS, spin-fluctuation, excitonic, plasmonic, and flat-band mechanisms to determine dominant Tc.
- **Instability Proximity Scoring**: Prioritizes materials near quantum critical points and structural/electronic instabilities.
- **Inverse Design**: Generates materials optimized for pairing susceptibility.
- **Generative Crystal Structures**: Discovers structural variants through chemical substitution, intercalation, and topology mapping, including novel prototype generation via LLM.
- **Boundary Hunting Mode**: Targets instability edges in formula generation when stagnation occurs.
- **Enriched ML Features**: Uses approximately 48 diverse physical and structural features for ML prediction.
- **CDW/SDW Auto-Kill**: Suppresses Tc for materials with strong charge/spin density wave instabilities.
- **Strategy Momentum**: Uses EMA smoothing to prevent rapid strategy changes.
- **Isostructural Duplicate Detection**: Skips lower-score duplicates based on prototype hashing.
- **Metallicity Pre-Filter**: Rejects candidates with low metallicity or high band gaps early in the process.
- **Lambda-Based Tc Clamp**: Penalizes high Tc predictions for materials with low lambda values to prevent ML hallucination.
- **Family-Specific Tc Caps**: Applies specific Tc caps based on material family (e.g., carbides, nitrides).
- **Element-Presence Family Classifier**: Classifies material families using regex and element presence for comprehensive coverage.
- **Pre-ML Insulator Filter**: Filters out insulating candidates before ML scoring to optimize resource usage.
- **Increased ML Batch Size**: Enhanced processing capacity for materials per cycle.
- **Fractional Stoichiometry Normalization**: Ensures integer stoichiometry for all formulas.
- **Insight Quality Filter**: Rejects vague insights, requiring quantitative data or specific correlations.
- **Milestone Deduplication**: Prevents repeated milestone events on system restarts.
- **End-to-End Pipeline Pass Rate**: Implements tightened stage thresholds for a more rigorous pipeline.
- **Family Diversity Metric**: Tracks unique families across a larger set of candidates.
- **Canonical Formula Normalization**: Standardizes formula representation across all insertion points.
- **Dynamic Progress Calculation**: Adjusts phase targets dynamically based on item counts.
- **Semantic Insight Deduplication**: Uses concept fingerprinting to identify and remove paraphrased duplicate insights.
- **Structure Predictor**: Predicts structure and synthesis temperature using crystallographic rules and material science principles.
- **Novel Synthesis Reasoning**: Generates physics-constrained synthesis paths for metastable candidates, proposing material-class-specific processing routes.
- **DFT Feature Resolver**: A unified service for merging and resolving DFT data from various sources.
- **Auto-DFT Enrichment**: A mechanism to enrich superconductor candidates with DFT data based on score, pipeline stage, and staleness.

### Gradient Boosting Model
- Trained on 500+ superconductor and non-superconductor entries.
- Uses 300 trees, learning rate 0.1, max depth 4, and 34 physical properties as features.

### Alive Engine
- **Real-time Feedback**: Broadcasts `thought` WebSocket messages for a live-scrolling feed.
- **Adaptive Tempo**: Adjusts cycle intervals based on research progress (excited, exploring, contemplating states).
- **Research Memory**: Aggregates knowledge summaries, hypotheses, family statistics, and abandoned strategies.
- **Dynamic Status Messages**: Displays contextual status messages in the sidebar.

### Infrastructure
- **Rate Limiting**: Implements `express-rate-limit` for API request control.
- **In-Memory Cache**: Utilizes a TTL cache for frequently accessed data.
- **DB Indexes**: Optimized database queries with indexes on key columns.
- **API Pagination**: Supports pagination for novel-predictions endpoints.
- **Experimental Validations**: Tracks experimental validation results.
- **ML Calibration**: Provides confidence bands and error bars for Tc predictions.
- **Cross-Validation**: Integrates AFLOW REST API and enhanced Materials Project queries for external data validation.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.