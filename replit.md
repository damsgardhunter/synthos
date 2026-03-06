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
- **Forbidden Word Sanitizer**: `sanitizeForbiddenWords()` in utils.ts replaces "breakthrough" and "confirmed" with "notable finding" and "verified" in all API responses and LLM outputs.
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
- **Failure Feedback Loop**: Accumulates pipeline failure examples and retrains XGBoost every 50 cycles when 20+ new failures exist. Failed candidates become negative training examples (Tc=0).

### Advanced Quantum Physics Modeling
- **Phonon Dispersion**: Computes phonon branches along high-symmetry path (Gamma-X-M-Gamma) with soft mode detection and imaginary frequency identification.
- **Phonon DOS**: Histogram of phonon frequencies from dispersion branches (100 bins). Computed via `computePhononDOS()`.
- **Eliashberg Spectral Function**: alpha2F(omega) computed from phonon DOS weighted by McMillan-Hopfield eta parameters. Integrated lambda and omega_log used in Allen-Dynes Tc. Via `computeAlpha2F()`.
- **GW Many-Body Corrections**: Quasiparticle renormalization (Z factor), DOS and bandwidth corrections, vertex corrections to lambda.
- **Dynamic Spin Susceptibility**: Lindhard-function-based chi(q,omega) with Stoner enhancement, magnetic correlation length, and QCP proximity detection.
- **Fermi Surface Nesting**: chi_0(q) along high-symmetry vectors (Gamma, X, M, R, A) identifying SDW/AFM, stripe-SDW, and CDW instabilities.

### Convex Hull Phase Diagram Engine (`phase-diagram-engine.ts`)
- **Convex Hull**: Computes energy above hull, decomposition products, and hull vertices for binary/ternary systems.
- **Competing Phases**: Queries storage for materials with same element set, builds hull, identifies decomposition pathways.
- **Metastability Assessment**: Arrhenius-based kinetic barrier estimation, lifetime prediction, decomposition pathway analysis.
- **Phase Diagram**: Full binary/ternary phase diagrams with stable/unstable phases and phase boundaries.

### Pressure Modeling Engine (`pressure-engine.ts`)
- **Birch-Murnaghan EOS**: Volume compression at pressure, bulk modulus, pressure derivative.
- **Hydride Formation**: Predicts hydrogen uptake, formation enthalpy under pressure, stable hydride stoichiometry.
- **High-Pressure Stability**: Combines volume compression, phonon stability, and enthalpy comparison with decomposition products.
- **Pressure-Tc Curves**: Sweeps pressure 0-300 GPa, computing Tc at each point to find optimal pressure.

### Graph Neural Network Surrogate (`graph-neural-net.ts`)
- **Crystal Graph Builder**: Converts formula + structure into graph with atom nodes and bond edges.
- **Message Passing**: 3-layer message-passing GNN with learned weight matrices.
- **Predictions**: Formation energy, phonon stability, Tc, and confidence.
- **Ensemble Integration**: GNN gets 40% weight when structure data available, alongside XGBoost (25%) and LLM-NN (35%).
- **Training**: Trains on known superconductor data with MSE loss, 30-minute cache TTL.

### Massive Candidate Generator (`candidate-generator.ts`)
- **Element Substitution**: Swaps each element with chemically similar alternatives using expanded 96-element atom swap maps with compatibility maps (carbide_formers, nitride_formers, boride_formers, hydride_formers).
- **Composition Interpolation**: Interpolates between two promising compositions with integer rounding (10-20 per pair).
- **Doped Variants**: Adds 1-3 dopant atoms from 14 common dopants at various concentrations (50+ per base).
- **Composition Sweep**: Enumerates integer stoichiometries up to maxAtoms with MAX_FORMULAS=2000 cap per element set.
- **Canonical Normalization**: `normalizeFormula()` + `reduceStoichiometry()` + sort by electronegativity before dedup.
- **Valence Sanity Filter**: Rejects chemically impossible compositions using charge balance heuristic and max oxidation state checks.
- **Rapid GB Screen**: Runs GB prediction only (no LLM, no physics) to filter 2000 candidates down to top 50 in milliseconds.

### Physics-Aware ML Predictor (`PhysicsPredictor` in `ml-predictor.ts`)
- **Multi-Target Prediction**: Predicts 4 key physics properties simultaneously: lambda, DOS(EF), omega_log, and hull distance.
- **Transfer Priors**: Uses elemental data (weighted DOS, Debye temps, Miedema formation energy) as priors when training data < 100 samples.
- **Uncertainty Estimation**: Computes model variance from tree ensemble disagreement. Returns `uncertainty: number` for each prediction.
- **Pre-Filter**: Rejects candidates where predicted lambda < 0.3 OR hull_distance > 0.2 eV/atom OR DOS(EF) < 0.5 states/eV.
- **Self-Reinforcing Loop**: Predicted physics values fed into Tc predictor feature vector. Retrains every 100 cycles.

### Pattern Mining / Theory Generator (`pattern-miner.ts`)
- **Quantitative Rules**: Decision-tree splitting on top 10 features (lambda, DOS_EF, omegaLog, correlationStrength, metallicity, VEC, dimensionality, nestingScore, hydrogenRatio, anharmonicity). Max 2 conditions per rule.
- **Rule Validation**: 70/30 cross-validation with F1 > 0.5 threshold. Rules have precision, recall, support counts.
- **Rule Aging**: `rule.weight *= 0.95` each re-mine cycle. Rules with weight < 0.1 removed.
- **Screening**: `applyRulesToScreen()` scores candidates by weighted sum of satisfied rules (theory score 0-1).
- **Integration**: `evolveRules()` runs every 50 cycles. `screenWithPatterns()` filters autonomous loop candidates.

### Structural Mutation Engine (`structural-mutator.ts`)
- **Prototype Assignment**: Assigns structure prototypes (rocksalt, perovskite, hexagonal, layered, bcc, fcc, spinel, fluorite, rutile, pyrite) from composition.
- **Distorted Lattices**: Tetragonal (c/a ±10-30%), orthorhombic (b/a), monoclinic tilt (beta 85-95°). Energy penalty filter > 0.5 eV/atom.
- **Layered Structures**: Ruddlesden-Popper series (n=1,2,3), spacer layers, superlattices.
- **Vacancy Structures**: 5-25% element removal, ordered vacancies, anti-site defects. Energy penalty filter > 0.5 eV.
- **Strain Variants**: Epitaxial strain from 5 substrates (SrTiO3, LaAlO3, MgO, Si, Al2O3) with lattice mismatch calculation.
- **Integration**: Runs on top 10 candidates every 10 cycles. Viable mutants screened with GB and inserted as candidates.

### Multi-Dimensional Phase Explorer (`phase-explorer.ts`)
- **Composition Space**: 2-4 element composition grid scans at coarse (10%) then fine (2%) resolution.
- **Pressure-Composition Sweep**: 2D adaptive sweep: composition x pressure 0-300 GPa. Identifies optimal (composition, pressure) pairs.
- **Temperature Stability**: Sweeps 0-1000K estimating phonon stability, Gibbs free energy decomposition risk, max operating temperature.
- **Adaptive Sampling**: Coarse scan → identify peaks → refine around peaks. Reduces compute.
- **Uncertainty-Aware Selection**: `score = predictedTc + sqrt(modelVariance) * 15`. Encourages exploration of uncertain regions.
- **Integration**: `findOptimalRegion()` runs every 20 cycles on focused element sets. Hotspots fed as seed compositions into candidate generator.

### Autonomous Discovery Loop
- **Massive Generation Pipeline**: Generates 500-2000 candidates per cycle via element substitution, interpolation, doping, and composition sweep.
- **Multi-Stage Filtering**: GB pre-screen (top 50) → pattern mining filter → physics ML pre-filter → full pipeline.
- **Full Pipeline**: Each candidate runs through GB pre-screen -> structure prediction -> convex hull -> full physics (alpha2F) -> Tc cap -> synthesizability check -> store.
- **Stats Tracking**: Total screened, passed, pass rate, throughput/hour, best Tc, GNN retrain count, physics ML training size.
- **API Endpoint**: `autonomousLoopStats` included in `/api/engine/memory` response.

### NLP Engine
- Statistical analysis dataset: 2000 materials (up from 200).
- LLM context window: 100 material summaries (up from 50).
- Enhanced statistics: band gap histograms, metal/semiconductor/insulator classification, std deviations, correlation matrices, element frequency analysis.

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