# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform designed to accelerate the discovery of room-temperature superconductors. It integrates AI for natural language processing, novel material generation, and machine learning predictions. The platform aims to revolutionize material discovery by continuously learning and expanding its knowledge base, from subatomic modeling to advanced computational physics and synthesis tracking, with significant market potential across high-tech industries.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases, covering subatomic to multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines a gradient boosting model with an OpenAI gpt-4o-mini neural network ensemble for superconductor scoring, applying strict room-temperature criteria (Tc >= 293K).
- **SC Verification Pipeline**: Categorizes superconductor candidates from theoretical to `requires-verification` to prevent premature claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline with calibrated thresholds for material properties.
- **Novel Insight Detection**: Filters known knowledge, deduplicates content, rejects vague insights, and uses OpenAI for novelty evaluation.
- **Data Confidence Tracking**: Candidates are tagged with `dataConfidence` (high, medium, low).
- **LLM Data Validation**: Cross-validates LLM properties against external APIs, enforcing physical bounds.
- **Physics as Tc Authority**: Grounds all Tc values in Eliashberg/McMillan physics with ambient-pressure caps.
- **Diversity & Deduplication**: Mechanisms to ensure diverse formula generation and prevent redundant work.
- **Evolving Research Strategy**: Balanced family scoring (40% normalized avg score + 40% normalized maxTc + 20% pipeline pass rate) with exploit-then-explore policy and dynamic switching. High-maxTc families are not penalized for low sample counts.
- **Milestone Detection System**: Identifies research milestones like new family discoveries and pipeline graduations.
- **Tc Plateau Escalation**: Triggers progressive strategies (boundary hunting, inverse design, chemical space expansion) upon stagnation.
- **Stability Filtering**: Incorporates formation energy bounds and a stability modifier in pairing scores.
- **Physical Plausibility Guardrails**: Enforces physical limits and criteria for `roomTempViable` candidates.
- **Stagnation Breaking**: Detects research stagnation and adapts candidate generation strategies via LLM prompts.

### Physics Engine
- **Deterministic Calculations**: All physics calculations are fully deterministic, utilizing a comprehensive elemental database of 96 elements.
- **Advanced Parameter Calculation**: Includes methods for DOS at Fermi level, bandwidth, Hubbard U/W, lambda, phonon frequencies, and metallicity.
- **Specialized Material Handling**: Incorporates specific physics adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductor mechanisms.
- **Pairing Susceptibility Optimization**: Optimizes for pairing conditions (DOS, nesting, coupling channels) rather than just raw Tc.
- **Multiple Pairing Mechanisms**: Evaluates BCS, spin-fluctuation, excitonic, plasmonic, and flat-band mechanisms.
- **Instability Proximity Scoring**: Prioritizes materials near quantum critical points and instabilities.
- **Inverse Design**: Generates materials optimized for pairing susceptibility.
- **Generative Crystal Structures**: Discovers structural variants and novel prototypes via LLM.
- **Boundary Hunting Mode**: Targets instability edges in formula generation during stagnation.
- **Enriched ML Features**: Uses approximately 48 diverse physical and structural features for ML prediction.
- **CDW/SDW Auto-Kill**: Suppresses Tc for materials with strong charge/spin density wave instabilities.
- **Metallicity Pre-Filter**: Rejects candidates with low metallicity or high band gaps early.
- **Lambda-Based Tc Clamp**: Penalizes high Tc predictions for materials with low lambda values.
- **Family-Specific Tc Caps**: Applies specific Tc caps based on material family.
- **Forbidden Word Sanitizer**: Replaces "breakthrough" and "confirmed" with "notable finding" and "verified" in all API responses and LLM outputs.
- **Milestone Deduplication**: Prevents repeated milestone events.
- **End-to-End Pipeline Pass Rate**: Implements tightened stage thresholds for a more rigorous pipeline.
- **Semantic Insight Deduplication**: Uses concept fingerprinting to identify and remove paraphrased duplicate insights.
- **Structure Predictor**: Predicts structure and synthesis temperature using crystallographic rules.
- **Novel Synthesis Reasoning**: Generates physics-constrained synthesis paths for metastable candidates.
- **DFT Feature Resolver**: Unifies and resolves DFT data from various sources.
- **Auto-DFT Enrichment**: Enriches superconductor candidates with DFT data based on score, pipeline stage, and staleness.

### Gradient Boosting Model
- Trained on 500+ superconductor and non-superconductor entries with 300 trees and 34 physical properties as features.
- **Failure Feedback Loop**: Retrains XGBoost every 50 cycles with pipeline failure examples as negative training data.

### Advanced Quantum Physics Modeling
- **Phonon Dispersion**: Computes phonon branches along high-symmetry paths, detecting soft modes.
- **Phonon DOS**: Histogram of phonon frequencies.
- **Eliashberg Spectral Function**: alpha2F(omega) computed from phonon DOS, used in Allen-Dynes Tc.
- **GW Many-Body Corrections**: Quasiparticle renormalization and corrections to DOS, bandwidth, and lambda.
- **Dynamic Spin Susceptibility**: Lindhard-function-based chi(q,omega) with Stoner enhancement and QCP proximity detection.
- **Fermi Surface Nesting**: chi_0(q) along high-symmetry vectors, identifying instabilities.

### Convex Hull Phase Diagram Engine
- Computes energy above hull, decomposition products, and hull vertices for binary/ternary systems.
- **Metastability Assessment**: Estimates kinetic barriers, lifetime, and decomposition pathways.
- **Phase Diagram**: Generates full binary/ternary phase diagrams.

### Pressure Modeling Engine
- **Birch-Murnaghan EOS**: Calculates volume compression, bulk modulus, and pressure derivative.
- **Hydride Formation**: Predicts hydrogen uptake and stable hydride stoichiometry under pressure.
- **High-Pressure Stability**: Combines volume compression, phonon stability, and enthalpy comparison.
- **Pressure-Tc Curves**: Sweeps pressure 0-300 GPa to find optimal conditions.

### Graph Neural Network Surrogate (CGCNN-style)
- **Prototype-Aware Graph Construction**: `buildPrototypeGraph(formula, prototype)` uses prototype-specific coordination environments (AlB2: B bonded to 3B+2M; Perovskite: B-site octahedral 6O; A15: chain connectivity).
- **Enhanced Node Features**: 13-dimensional embeddings: atomic number, electronegativity, atomic radius, valence electrons, mass, Debye temperature, bulk modulus, ionization energy, Mendeleev number, electron affinity, covalent radius, d-orbital occupancy, f-orbital occupancy.
- **Attention-Weighted Message Passing**: 3-layer GNN with dot-product attention coefficients + softmax normalization for expressive aggregation.
- **Multi-Target Predictions**: Formation energy, phonon stability, Tc, confidence, and electron-phonon lambda.
- **Ensemble Uncertainty**: `gnnPredictWithUncertainty(formula, prototype?)` runs 3× with dropout-style perturbation, returns mean ± std as uncertainty estimate.
- **Ensemble Integration**: GNN is integrated with XGBoost and LLM-NN for predictions.

### Massive Candidate Generator
- **Element Substitution**: Swaps elements with chemically similar alternatives using expanded atom swap maps.
- **Composition Interpolation**: Interpolates between promising compositions.
- **Doped Variants**: Adds common dopants at various concentrations.
- **Composition Sweep**: Enumerates integer stoichiometries up to a cap.
- **Valence Sanity Filter**: Rejects chemically impossible compositions using charge balance heuristics.
- **Periodic Table Element Validation**: All formulas validated against full 118-element periodic table before insertion. Rejects industrial designations (e.g., A356) and invalid element tokens.
- **Rapid GB Screen**: Pre-filters candidates using gradient boosting prediction only.

### Physics-Aware ML Predictor
- **Multi-Target Prediction**: Predicts lambda, DOS(EF), omega_log, and hull distance simultaneously.
- **Transfer Priors**: Uses elemental data as priors for limited training data.
- **Uncertainty Estimation**: Computes model variance from tree ensemble disagreement.
- **Pre-Filter**: Rejects candidates based on predicted lambda, hull distance, and DOS(EF).
- **Self-Reinforcing Loop**: Predicted physics values are fed into the Tc predictor feature vector.

### Pattern Mining / Theory Generator
- **Quantitative Rules**: Extracts decision-tree rules based on key physical features.
- **Rule Validation**: Uses cross-validation with F1 threshold.
- **Rule Aging**: Rules are aged and removed if their weight falls below a threshold.
- **Screening**: Scores candidates by weighted sum of satisfied rules.

### Structural Mutation Engine
- **Prototype Assignment**: Assigns structure prototypes from composition.
- **Distorted Lattices**: Generates tetragonal, orthorhombic, and monoclinic variants with energy penalty filtering.
- **Layered Structures**: Creates Ruddlesden-Popper series and superlattices.
- **Vacancy Structures**: Introduces ordered vacancies and anti-site defects.
- **Strain Variants**: Generates epitaxial strain variants from common substrates.

### Multi-Dimensional Phase Explorer
- **Composition Space**: Scans composition grids at coarse then fine resolution.
- **Pressure-Composition Sweep**: 2D adaptive sweep to identify optimal (composition, pressure) pairs.
- **Temperature Stability**: Estimates phonon stability, Gibbs free energy decomposition risk, and max operating temperature.
- **Adaptive Sampling**: Coarse scan to identify peaks, then refines around them.
- **Uncertainty-Aware Selection**: Scores encourage exploration of uncertain regions.

### Crystal Prototype Structure Generator
- **10 Structure Types**: AlB2, Perovskite, A15, Clathrate/Sodalite, ThCr2Si2, Spinel, MAX-phase, Layered nitride, Laves, Heusler.
- **807+ Structurally-Typed Candidates**: Each includes formula, prototype, spaceGroup, crystalSystem, dimensionality, siteAssignment.
- **Prototype-Aware Naming**: Candidates stored as "AlB2-type NbB2", "Perovskite BaTiO3", "A15-type Nb3Sn", etc.
- **Engine Integration**: Fires every 25 cycles, replaces old family-aware generation.

### Convex Hull Stability Gate
- **Hard Rejection**: All candidates must pass hull distance ≤ 0.1 eV/atom before database insertion.
- **`passesStabilityGate(formula)`**: Computes Miedema formation energy → convex hull distance → verdict (stable/near-hull/metastable/unstable).
- **Metastability Assessment**: Borderline candidates (0.05-0.1 eV/atom) require kinetic barrier > 0.5 eV.
- **Universal Enforcement**: All 10 insertion points (7 in engine.ts + 3 in superconductor-research.ts) route through the stability gate.

### Discovery Score
- **Composite Metric**: `0.4 × normalizedTc + 0.3 × noveltyScore + 0.2 × stabilityScore + 0.1 × synthesisFeasibility`.
- **Chemical Novelty**: Bonuses for multi-element combos, rare elements, unexplored prototypes, distance from known SCs.
- **Stored in DB**: `discovery_score` column on superconductor_candidates table.
- **DFT Queue Priority**: Higher-scoring candidates selected first for DFT enrichment.

### Active Learning Loop
- **Uncertainty-Driven DFT**: `selectForDFT(candidates, budget=10)` ranks by acquisition score = `0.5 × normalizedTc + 0.5 × uncertainty`.
- **Model Retraining**: After DFT enrichment, retrains GNN surrogate with expanded dataset, calls `incorporateFailureData()` for Tc=0 candidates.
- **Convergence Tracking**: Monitors totalDFTRuns, avgUncertaintyBefore/After, modelRetrains, bestTcFromLoop.
- **Engine Integration**: Fires every 30 cycles.

### Autonomous Discovery Loop
- **Massive Generation Pipeline**: Generates 500-2000 candidates per cycle.
- **Multi-Stage Filtering**: GB pre-screen → pattern mining filter → physics ML pre-filter → full pipeline.
- **Full Pipeline**: Each candidate undergoes structure prediction, convex hull, full physics, Tc cap, synthesizability check, and storage.

### NLP Engine
- **Cross-Property Correlation Insights**: Generates physics relationship insights (e.g., "High DOS(EF) correlates with elevated Tc in carbides") instead of dataset statistics.
- **Superconductor Correlation Analysis**: Computes lambda vs Tc, stability vs Tc, dimensionality vs Tc/lambda, per-family breakdowns, and pairing mechanism correlations.
- **Statistical Summary Filter**: Rejects pure statistical summaries (e.g., "X% of materials have...", "average band gap is...") from insight generation.
- Provides correlation coefficients between material properties and element frequency analysis.

### Alive Engine
- **Real-time Feedback**: Broadcasts `thought` WebSocket messages for a live feed.
- **Adaptive Tempo**: Adjusts cycle intervals based on research progress.
- **Research Memory**: Aggregates knowledge summaries, hypotheses, and family statistics.
- **Dynamic Status Messages**: Displays contextual status messages.

### Infrastructure
- **Rate Limiting**: Implements `express-rate-limit` for API request control.
- **In-Memory Cache**: Utilizes a TTL cache for frequently accessed data.
- **DB Indexes**: Optimized database queries with indexes.
- **API Pagination**: Supports pagination for novel-predictions endpoints.
- **ML Calibration**: Provides confidence bands and error bars for Tc predictions.
- **Cross-Validation**: Integrates AFLOW REST API and enhanced Materials Project queries for external data validation.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.