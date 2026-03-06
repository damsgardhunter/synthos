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
- **Novel Insight Detection**: Pre-filters known knowledge, deduplicates similar insights (Levenshtein + Jaccard similarity), rejects meta-commentary and vague qualitative statements, requires quantitative content (formulas, metrics, numbers), and uses OpenAI to evaluate remaining insights for novelty. Max 3 novel insights per cycle with overflow queuing.
- **Data Confidence Tracking**: Candidates are tagged with `dataConfidence` (high, medium, low) based on their origin (DFT, model, LLM-estimated), displayed in the UI.
- **LLM Data Validation**: Properties from LLM are cross-validated against Materials Project API, enforcing physical bounds for band gap, formation energy, lattice parameters, and density.
- **Physics as Tc Authority**: All Tc values are grounded in Eliashberg/McMillan physics. The system includes ambient-pressure Tc caps and ensures Tc only changes when physics inputs change.
- **Diversity & Deduplication**: Mechanisms are in place to ensure diversity in formula generation and prevent redundant work in SC candidate processing.
- **Evolving Research Strategy**: Bayesian-adjusted family scoring prevents single-sample bias (k=10 prior). Exploit-then-explore policy locks onto families for 8-cycle windows with 15% random exploration probability. Strategy switching requires >20K Tc gap to override the exploit window.
- **Milestone Detection System**: Detects various research milestones such as new-family discoveries, Tc records, and pipeline graduations. Insight cascade milestones require 3+ novel insights and are capped to 1 per 10 cycles.
- **Tc Plateau Escalation**: Level-1 (5+ stagnant cycles): boundary hunting + inverse design. Level-2 (12+): wider element swaps. Level-3 (16+): exotic substitutions + forced strategy switch. Level-4 (20+): chemical space expansion with novel elements.
- **Stability Filtering**: Formation energy bounds (reject >2.0 or <-5.0 eV/atom), hull distance estimation with 20% score penalty for metastable materials, and stability modifier in pairing score.
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters like Hc2, coherence length, pressure, and defines strict criteria for `roomTempViable` candidates.
- **Stagnation Breaking**: The system detects research stagnation and triggers re-analysis of high-lambda candidates and adapts candidate generation strategies via LLM prompts.

### Physics Engine
- **Deterministic Calculations**: All physics calculations are fully deterministic, with no randomness.
- **Comprehensive Elemental Data**: Utilizes a database of 96 elements with tabulated properties.
- **Advanced Parameter Calculation**: Includes methods for calculating DOS at Fermi level, bandwidth, Hubbard U/W, lambda, phonon frequencies, and metallicity using various models and lookups.
- **Specialized Material Handling**: Incorporates specific physics adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductor mechanisms.
- **Physics v10 Material-Class-Aware Constraints**: Lambda caps are now material-class-specific: conventional metals 2.0, cuprates 1.5, iron-pnictides 1.8, superhydrides 3.5, light-element compounds 2.5. Light-element boost reduced from 2.5x to 1.2x. Hydrogen lambda contribution now pressure-gated and bonding-type-aware.
- **Hydrogen Bonding Classification**: `classifyHydrogenBonding()` distinguishes metallic-network (LaH10-type), cage-clathrate (CaH6-type), covalent-molecular (NH3/CH4-type), and interstitial (PdH-type) hydrogen. Only metallic networks at high pressure get full lambda boost.
- **Grounded Allen-Dynes Inputs**: omega_log clamped to class-appropriate ranges (conventional 50-400K, hydrides 500-1500K). mu* widened to 0.10-0.20 with corrections for electronegativity spread and d-electron character. McMillan cap reduced from 600K to 400K.
- **Pairing Susceptibility Optimization**: Engine and LLM prompts now optimize for pairing conditions (DOS, nesting, coupling channels) rather than raw Tc. Stagnation detection considers pairing susceptibility improvement alongside Tc.
- **Multiple Pairing Mechanisms**: `runUnifiedPairingAnalysis()` evaluates BCS, spin-fluctuation, excitonic, plasmonic, and flat-band mechanisms. Dominant mechanism determines Tc.
- **Instability Proximity Scoring**: `computeInstabilityProximity()` scores materials near magnetic QCP, structural boundary, metal-insulator transition, CDW instability, and soft phonon collapse. Boundary-proximate materials get priority.
- **Inverse Design**: `generateInverseDesignCandidates()` runs every 5th cycle, designing materials optimized for pairing susceptibility (lambda, DOS, nesting) rather than just Tc.
- **Generative Crystal Structures**: `runGenerativeStructureDiscovery()` generates structural variants via chemical substitution, intercalation, and topology mapping (Kagome, pyrochlore, etc.) every 3rd cycle. Novel crystal prototype generation via LLM runs every 10th cycle with design principles (flat bands, Kagome planes, breathing pyrochlore, etc.).
- **Boundary Hunting Mode**: Activated on stagnation (>5 cycles without Tc/pairing improvement). Formula generator targets instability edges. Combined with inverse design mode.
- **Enriched ML Features**: ~48 features: orbital character, phonon spectral centroid/width, bond stiffness variance, charge transfer magnitude, connectivity index, nestingScore, vanHoveProximity, bandFlatness, softModeScore, motifScore, orbitalDFraction, mottProximityScore, topologicalBandScore, dimensionalityScoreV2, phononSofteningIndex, spinFluctuationStrength, fermiSurfaceNestingScore, dosAtEF, muStarEstimate.
- **CDW/SDW Auto-Kill**: If CDW instability > 0.8 and lambda < 1.5, Tc is suppressed (near-zero penalty). Same for SDW. Prevents non-superconducting candidates from progressing.
- **Strategy Momentum**: EMA smoothing (0.7 * previous + 0.3 * new) prevents cycle-to-cycle strategy thrashing. Priorities evolve gradually.
- **Isostructural Duplicate Detection**: Prototype hashing detects stoichiometrically identical compositions (e.g., Ca8Si46H12 vs Ba8Si46H12) and skips lower-score duplicates.
- **Metallicity Pre-Filter**: Candidates with bandGap > 0.5 eV or metallicity < 0.2 are rejected before insertion into the SC candidate pool.
- **Lambda-Based Tc Clamp**: Pre-physics safeguard: if Tc > 80K but lambda < 1.5, Tc is penalized (×0.15 for lambda<0.5, ×0.25 for lambda<1.0, ×0.3 for lambda<1.5). Prevents ML hallucination of high-Tc carbides/nitrides.
- **Canonical Formula Normalization**: `normalizeFormula()` in utils.ts sorts elements by electronegativity, reduces stoichiometry by GCD, and removes trailing "1"s. Applied at all insertion points: SC research, novel generation, inverse design, structure variants, formula generator.
- **Dynamic Progress Calculation**: Phase progress uses `dynamicTarget()` — when items exceed base target, the target scales to 110% of items. Prevents 99%-forever phases with inflated counts.
- **Semantic Insight Deduplication**: Concept fingerprinting with synonym normalization catches paraphrased duplicates (e.g., "stability correlates with formation energy" ≈ "lower formation energy indicates stability"). 80% concept overlap threshold.
- **Structure Predictor**: Uses crystallographic rules, Goldschmidt tolerance factor, Vegard's law, and Tammann rule for structure and synthesis temperature prediction.
- **Novel Synthesis Reasoning (Phase 13)**: Physics-constrained synthesis path generator in `server/learning/synthesis-reasoning.ts`. Analyzes thermodynamic landscape (hull distance, formation energy, decomposition barriers) and proposes non-equilibrium processing routes for metastable candidates. Material-class-specific routes: hydrides (DAC hydrogenation), HEAs (mechanical alloying + SPS, combinatorial sputtering), cuprates (sol-gel + O2 annealing), plus rapid quench, thin film, and reactive milling routes. Routes tagged `source: "physics-reasoned"` vs `"literature-based"`. Runs after DFT enrichment, limited to 3 candidates/cycle.
- **DFT Feature Resolver**: A unified service merging Materials Project and AFLOW DFT data, providing DFT-resolved features for ML prediction and physics analysis.
- **Auto-DFT Enrichment**: An engine cycle mechanism to enrich superconductor candidates with DFT data based on score, pipeline stage, and staleness.

### Gradient Boosting Model
- Trained on 500+ entries (superconductors and non-superconductors) from the SuperCon database.
- 300 trees, learning rate 0.1, max depth 4, min samples per leaf 5.
- Features include 34 physical properties like lambda, metallicity, omegaLog, debyeTemp, and various compositional and structural indicators.

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