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
- **Evolving Research Strategy**: Balanced family scoring with confidence weighting, implementing an exploit-then-explore policy with dynamic switching.
- **Milestone Detection System**: Identifies research milestones like new family discoveries and pipeline graduations.
- **Tc Plateau Escalation**: Triggers progressive strategies (boundary hunting, inverse design, chemical space expansion) upon stagnation.
- **Stability Filtering**: Incorporates formation energy bounds and a stability modifier in pairing scores.
- **Physical Plausibility Guardrails**: Enforces physical limits and criteria for `roomTempViable` candidates.
- **Stagnation Breaking**: Detects research stagnation and adapts candidate generation strategies via LLM prompts.

### Physics Engine
- **Deterministic Calculations**: All physics calculations are fully deterministic, utilizing a comprehensive elemental database of 96 elements.
- **Advanced Parameter Calculation**: Includes methods for DOS at Fermi level, bandwidth, Hubbard U/W, lambda, phonon frequencies, and metallicity.
- **Specialized Material Handling**: Incorporates specific physics adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductor mechanisms.
- **Screened Coulomb Pseudopotential (mu*)**: Thomas-Fermi screening from DOS(EF), bandwidth-aware Morel-Anderson renormalization, material-class-specific defaults (hydrides ~0.10, cuprates ~0.13, conventional ~0.12, heavy fermion ~0.15). Blends 50/50 with class default. Range [0.08, 0.20].
- **Anharmonic Phonon Correction**: `lambda_corrected = lambda / (1 + anharmonic_factor)`. Hydride-specific stronger corrections: superhydrides factor 0.8× with min floor 0.25, high-p hydrides floor 0.15. Tracks both corrected and uncorrected lambda via `lambdaUncorrected` and `anharmonicCorrectionFactor`.
- **Electron Correlation Correction**: U/W-based lambda suppression per material class. Cuprates: nonlinear suppression via `1/(1 + 1.8*(U-0.6)^1.5)` (min 0.3). Heavy fermion: strongest via `1/(1 + 2.5*(U-0.3)^2)` (min 0.15). Iron pnictides: moderate. Nickelates detected as distinct class (Ni+O+RE).
- **Pairing Susceptibility Optimization**: Optimizes for pairing conditions (DOS, nesting, coupling channels) rather than just raw Tc, evaluating multiple pairing mechanisms.
- **Inverse Design (LLM)**: Generates materials optimized for pairing susceptibility via GPT-4o-mini.
- **Generative Crystal Structures**: Discovers structural variants and novel prototypes via LLM.
- **Enriched ML Features**: Uses approximately 50 diverse physical and structural features for ML prediction, including `pressureGpa` and `optimalPressureGpa`.
- **Lambda Caps (per class)**: conventional-metal 1.5, cuprate 1.2, iron-pnictide 1.5, heavy-fermion 0.8, hydride-low-p 2.0, hydride-high-p 3.0, superhydride 3.5, light-element 1.8, other 1.5. CouplingPrefactor = avgEta × N_EF × 1.2.
- **Magnetic/Stoner Tc Suppression**: Competing magnetic phases with strength > 0.3 reduce Tc by up to 70%. Stoner enhancement > 5 (stonerProduct > 0.8) suppresses phonon-mediated SC.
- **CDW/SDW Gradual Suppression**: Threshold lowered to 0.4 with gradual penalty (1 - instability × 0.6). Hydrides with lambda > 2.0 are exempt.
- **Van Hove Proximity**: Formula `(vhsRatio - 2.0) × 0.25` prevents over-triggering. Cuprate floor 0.7, kagome 0.6.
- **Phonon Frequency Diversity**: logAvgFreqRatio computed from average atomic mass (4 buckets: >100→0.25, >60→0.35, >30→0.40, else→0.50) with mass-range correction.
- **Formula Normalization**: Strips whitespace, handles fractional coefficients via integer scaling with GCD reduction. Rejects formulas with >50 atoms after scaling (returns cleaned raw).

### Topological Superconductor Detection Engine
- **Band structure topology**: Estimates Z2 invariant, Chern number indicator, and mirror symmetry indicator from SOC strength, orbital mixing, and band inversion probability.
- **SOC modeling**: Z^4 scaling with 60+ elements mapped to SOC strength values; weighted by concentration for multi-element compounds.
- **Band inversion detection**: Combines SOC, p-d orbital mixing, tight-binding band inversions, and Dirac crossings to estimate band inversion probability.
- **Material pattern matching**: 16 known topological material patterns (Bi2Se3-class TI, WTe2-class Weyl, CuxBi2Se3 TSC, Sr2RuO4-type TSC, etc.) with pattern-specific bonuses.
- **Majorana feasibility**: Estimates Majorana quasiparticle hosting potential from Z2, SOC, band inversion, and Dirac node probabilities.
- **Topological classification**: Classifies as topological-superconductor, strong-TI, Chern-insulator, topological-crystalline-insulator, weak-topological, SOC-enhanced, or trivial.
- **Score composition**: topologicalScore = 0.30 SOC + 0.25 band_inversion + 0.25 symmetry_invariants + 0.20 flat_band + pattern_bonus.
- **Discovery score integration**: Weights unified to 0.35 Tc + 0.20 stability + 0.15 novelty + 0.10 synthesis + 0.10 topology + 0.10 uncertainty_bonus.
- **API**: `GET /api/topology/:formula` (full analysis), `GET /api/topology-stats` (aggregate stats).
- Files: `server/physics/topology-engine.ts`

### Quantum Pairing Mechanism Simulator (7-Channel)
- **7 pairing channels**: phonon (mode-resolved λ_qν), spin-fluctuation (χ(q) via Hubbard U/W), orbital-fluctuation (d-orbital degeneracy, Hund's coupling), excitonic (DOS/bandGap), CDW (nestingScore + vanHoveProximity, CDW-SC competition), polaronic (strong-coupling BCS-BEC crossover, bipolaron binding), plasmon (collective electron oscillations, dimensionality-driven).
- **Mode-resolved phonon coupling**: Decomposes total λ into acoustic/optical/high-freq/soft-mode contributions per dispersion branch.
- **CDW coupling**: Models CDW-SC competition; strong CDW order suppresses SC; recognizes NbSe2/TaSe2-type materials.
- **Polaronic coupling**: BCS-BEC crossover physics for lambda > 2.0; bipolaron binding for low metallicity; recognizes bismuthate/titanate families.
- **Plasmon pairing**: Collective electron oscillations in low-carrier-density and 2D systems; relevant for SrTiO3-type materials.
- **Material-aware weighting**: 7 weights summing to 1.0, class-specific (cuprates, pnictides, hydrides, nickelates, bismuthates, CDW materials).
- **Pairing symmetry inference**: d-wave, s+/-, CDW-modulated s-wave, polaronic/BEC-like s-wave, anisotropic s-wave.
- **API**: `GET /api/pairing/profile/:formula` (full 7-channel analysis), `GET /api/pairing/features/:formula` (ML feature vector).
- Files: `server/physics/pairing-mechanisms.ts`

### Fermi Surface Reconstruction Engine
- **Full BZ grid computation**: E(k) across entire Brillouin zone (not just high-symmetry paths) using tight-binding Hamiltonians.
- **Isosurface detection**: Finds FS surfaces where E(k) = EF, classifies pockets as electron/hole type with orbital character.
- **Extracted features**: pocket_count, pocket_volume, cylindrical_character, electron_hole_balance, nesting_vectors.
- **ML features**: fermiPocketCount, electronHoleBalance, fsDimensionality, sigmaBandPresence, multiBandScore.
- **Lattice-aware sampling**: BCC, FCC, hexagonal, cubic BZ grids with proper high-symmetry awareness.
- **LRU cache**: 200-entry cache for performance.
- **API**: `GET /api/fermi-surface/:formula`
- Files: `server/physics/fermi-surface-engine.ts`

### Hydrogen Network Topology Analyzer
- **H-H distance distribution**: Metallic bond fractions, mean/min/max distances based on H:M ratio and bonding type.
- **Network dimensionality**: 0-3D classification based on bonding type and hydrogen ratio.
- **Cage topology**: Classifies as sodalite, clathrate-I/II, dodecahedron, icosahedron with cage completeness and symmetry.
- **Coordination number**: Estimated from cage topology and hydrogen ratio.
- **Hydrogen density**: Volume-based H density estimation.
- **Phonon coupling score**: Integrated with physics engine for H-specific phonon contributions.
- **Network class**: sodalite-cage, clathrate-cage, 3D-metallic-network, layered-hydride, etc.
- **ML features**: hydrogenNetworkDim, hydrogenCageScore, Hcoordination, hydrogenConnectivity, hydrogenPhononCouplingScore.
- **API**: `GET /api/hydrogen-network/:formula`, `GET /api/hydrogen-network-stats`
- Files: `server/physics/hydrogen-network-engine.ts`

### Advanced Physics Constraints (8-Channel)
- **Fermi Surface Nesting**: Evaluates FS topology, pocket count, quasi-2D character, peak nesting vs DOS ratio. Penalizes weak nesting (score < 0.2).
- **Orbital Hybridization**: d-p overlap, s-p overlap, sigma-bond contribution. Detects Cu-d/O-p, Fe-d multi-orbital, sigma-bond (MgB2-like) types. Penalizes score < 0.15.
- **Lifshitz Transition Proximity**: Band edge distance to Fermi level, DOS spike detection, pocket transition probability. Uses vanHoveProximity, flatBandIndicator.
- **Quantum Critical Fluctuation**: Stoner enhancement, spin fluctuation energy, near-QCP detection. Classifies QCP type (magnetic-HF, doping-driven, magnetic-pnictide, itinerant).
- **Electronic Dimensionality**: Band dispersion anisotropy (bandwidth_xy/bandwidth_z), classifies strongly-2D/quasi-2D/3D-isotropic. Cuprates get anisotropy >= 15, pnictides >= 8.
- **Phonon Soft Mode**: Stable soft mode scoring, enhancement factor for coupling, imaginary mode penalty. Rejects unstable materials (hasImaginaryModes).
- **Charge Transfer Energy**: Delta = |d_band_center - p_band_center|, optimal range 0-3 eV. Cuprates fixed at 1.8 eV, pnictides 2.5 eV.
- **Lattice Polarizability**: Dielectric constant estimation from ionic/anion contributions, soft mode enhancement. SrTiO3 >= 300, BaTiO3 >= 1000. Threshold epsilon > 20.
- **Composite scoring**: Weighted sum (nesting 0.15, hybridization 0.15, lifshitz 0.10, QCP 0.12, dimensionality 0.10, soft-mode 0.13, charge-transfer 0.10, polarizability 0.15).
- **Tc boost/penalty**: compositeBoost = 1.0 + (compositeScore - 0.5) * 0.3 - penalty * 0.2, clamped [0.7, 1.4]. Applied to Eliashberg Tc.
- **ML feature storage**: All 8 constraint scores + types stored in mlFeatures.advancedConstraints.
- Files: `server/physics/advanced-constraints.ts`, integrated in `server/learning/physics-engine.ts` runFullPhysicsAnalysis.

### Chemical Stability Reaction Network
- **Reaction graph engine**: Builds compound → decomposition pathway graphs with competing phases.
- **Decomposition pathways**: Binary/ternary phase decomposition with Miedema formation energies.
- **Kinetic barriers**: Arrhenius-based barrier estimation from melting points and structural complexity.
- **Metastable lifetime**: Estimated from reaction barriers (effectively infinite to seconds).
- **Decomposition mechanisms**: elemental, oxidative, dehydrogenation, hydrogen-redistribution, binary-disproportionation, multi-phase, phase-separation.
- **Features**: reactionStabilityScore, metastableLifetime, decompositionComplexity.
- **API**: `GET /api/reaction-network/:formula`
- Files: `server/physics/reaction-network-engine.ts`

### Materials Genome Representation (256-dim)
- **Latent genome vector**: 256-dimensional encoding from 8 physics-derived segments: structure (40d), orbitals (36d), phonons (32d), coupling (32d), topology (28d), dimensionality (24d), composition (40d), pairing (24d).
- **Fourier + hash encoding**: Rich feature representation using existing physics engines.
- **Genome-space similarity search**: Cosine + Euclidean distance with segment-level analysis.
- **Genome-guided inverse design**: Searches candidate pool in genome space rather than chemical space.
- **Genome diversity**: Average pairwise distance for diversity assessment.
- **Genome interpolation**: Linear interpolation between material genomes.
- **LRU cache**: 500-entry cache.
- **API**: `GET /api/genome/:formula`, `POST /api/genome/similarity`, `POST /api/genome/diversity`, `POST /api/genome/inverse`, `GET /api/genome/stats`
- Files: `server/physics/materials-genome.ts`

### Advanced Quantum Physics Modeling
- Computes phonon dispersion, phonon DOS, Eliashberg spectral function, GW many-body corrections, dynamic spin susceptibility, and Fermi surface nesting.

### Autonomous Theory Discovery Engine
- **Symbolic regression** (genetic programming) discovers mathematical relationships from simulation data.
- **Unified Physics Feature Database** (`server/theory/physics-feature-db.ts`): 20-dim feature vector (DOS_EF, VHS, band_flatness, FS_dim, phonon_log_freq, lambda, nesting, orbital_degeneracy, charge_transfer, lattice_anisotropy, mott_proximity, spin_fluctuation, cdw_proximity, quantum_critical_score, pairing_strength, H_density, correlation, bandwidth, debye_temp, anharmonicity). LRU cache 2000 entries.
- **Symbolic Regression** (`server/theory/symbolic-regression.ts`): Expression tree GP with operations (+,-,*,/,^,sqrt,exp,log). Population 200, 50 generations, tournament selection. Physics constraint filters (reject negative Tc scaling, exponents >5). Theory knowledge base stores best equations.
- **Overfitting Mitigation**: K-fold cross-validation (k=5, uses avg validation R² as fitness), holdout validation (20% reserved), physics plausibility scoring (dimensional consistency, monotonicity checks, McMillan-like term bonuses). `TheoryCandidate` includes `validationR2`, `validationMAE`, `cvScore`, `overfitRatio`, `isOverfit`, `plausibilityDetails`. Overfit flag when overfitRatio > 1.5 or cvScore < 0.3. `getValidationStats()` provides aggregate validation metrics.
- **Theory feedback**: Discovered equations (e.g., Tc ∝ DOS^1.2 * nesting^0.8) translate into design constraints for inverse design guidance.
- **APIs**: `GET /api/theory/features/:formula`, `GET /api/theory/discovered` (includes validationStats), `POST /api/theory/discover`
- **Pipeline integration**: Features recorded after every physics evaluation; symbolic regression runs every 10 autonomous cycles when ≥20 records.

### Multi-Scale Physics Modeling
- **Three formalized feature layers** (`server/theory/multi-scale-engine.ts`):
  - Atomic: mass, bond length distribution, coordination, charge transfer, radius variance, EN spread
  - Electronic: DOS_EF, bandwidth, VHS distance, orbital character (s/p/d/f), FS shape, nesting, flatness, Mott proximity
  - Mesoscopic: layeredness, lattice anisotropy, strain sensitivity, defect tolerance, interlayer coupling, dimensionality
- **Cross-Scale Coupling**: electron_phonon_mass_ratio, strain_band_shift, layer_coupling_strength, bond_stiffness_vs_phonon, orbital_phonon_coupling, charge_transfer_vs_nesting
- **Sensitivity Analysis**: Gradient-based (finite-difference perturbation) importance of each scale for Tc prediction, identifies dominant scale.
- **APIs**: `GET /api/theory/multi-scale/:formula`, `GET /api/theory/sensitivity/:formula`

### Self-Improving Physics Models
- **Parameterized physics equations** (`server/theory/self-improving-physics.ts`): mu_star (screening, logRatio, blending), phonon_scale (H-Debye, maxFreq, logAvg), anharmonic_factor (Gruneisen, H-fraction, lambda suppression), pairing_weight (correlation penalty, soft-mode threshold).
- **Bayesian parameter optimization**: GP regression on prediction error with RBF kernel, Expected Improvement acquisition. Conservative blending (30%) of suggested updates. Minimum 10 observations, 60s throttle.
- **Model Performance Tracker** (`server/theory/model-performance-tracker.ts`): MAE/RMSE/R² across rolling windows (50/100/500/all), retrain triggers (MAE increase >20% or R² < 0.3), theory discovery rate, candidate success rate, parameter drift tracking.
- **Continuous Learning Loop**: generate → simulate → extract features → update dataset → discover equations → update parameters → improve surrogates → guide next search.
- **APIs**: `GET /api/theory/parameters`, `GET /api/theory/performance`

### Multi-Material Interface Discovery Engine
- **Heterostructure superconductivity analyzer** for interface SC discovery (2D superconductors).
- **Interface analysis**: charge transfer (electronegativity/work function mismatch, doping type), interface phonon enhancement (acoustic mismatch model, soft-mode coupling), epitaxial strain (lattice mismatch, critical thickness, dome-shaped strain coupling), dimensional confinement (2D enhancement factor).
- **Heterostructure generation**: Generates A/B/A/B stacked candidates from substrate/film pools (SrTiO3, LaAlO3, MgO, TiO2 substrates; FeSe, NbSe2, WTe2, FeTe, cuprate films), ranks top 50 by interface SC score.
- **Known system calibration**: FeSe/SrTiO3 (~65K), LAO/STO (~0.3K interface SC), twisted bilayer graphene (~1.7K).
- **APIs**: `GET /api/interface/:layerA/:layerB`, `GET /api/interface-candidates`
- Files: `server/physics/interface-engine.ts`

### Generator Resource Manager
- **Centralized allocation** (`server/learning/generator-manager.ts`) for 7 candidate generators: RL (40%), inverse_design (15%), BO_exploration (15%), massive_combinatorial (10%), structure_diffusion (10%), motif_diffusion (5%), random_exploration (5%).
- **Adaptive rebalancing**: Every 5 cycles, adjusts weights via softmax on normalized yield scores (pass rate 40% + best Tc 40% + novelty 20%). Minimum 2% floor per generator.
- **Budget system**: `allocateBudget(totalSlots)` distributes candidate slots proportionally. Each generator gets at least 1 slot.
- **Stats tracking**: Per-generator candidates generated/passed, best Tc, avg Tc, novelty score (exponential moving average).
- **Pipeline integration**: Called at start of autonomous loop, parameterizes RL/massive gen counts, records outcomes after screening.
- **API**: `GET /api/generator-allocations`

### Fermi Surface Topology Clustering
- **9-dim FS feature vector** (`server/physics/fermi-surface-clustering.ts`): pocketCount, electronPocketCount, holePocketCount, electronHoleBalance, cylindricalCharacter, nestingScore, fsDimensionality, sigmaBandPresence, multiBandScore.
- **5 archetype clusters**: cuprate_cylinder (high cylindrical, 2D, strong nesting), pnictide_eh_pockets (balanced e-h, moderate nesting), kagome_flat (high multiBand, 2D), hydride_multiband (high pockets, 3D, high sigma), conventional_3d (3D, low nesting, few pockets).
- **Clustering**: Cosine similarity to archetypes (threshold 0.65). Novel clusters auto-discovered when no archetype matches.
- **Search guidance**: `getClusterGuidance()` identifies high-Tc clusters and under-explored clusters for targeted search.
- **Pipeline integration**: After FS computation in Phase 10, `assignToCluster(formula, fsResult, tc)` assigns materials. Cluster ID stored in `mlFeatures.fermiCluster`. Formula dedup prevents double-counting.
- **Startup seeding**: Top 100 candidates (by Tc) are backfilled through topology analysis and Fermi surface clustering on startup to populate stats immediately.
- **Autonomous loop**: Topology and Fermi analysis run for every passed candidate in the autonomous fast-path loop.
- **APIs**: `GET /api/fermi-clusters`, `GET /api/fermi-clusters/:clusterId`

### Materials Discovery Landscape
- **UMAP embedding engine** (`server/landscape/discovery-landscape.ts`): Pure TypeScript UMAP reducing 256D Materials Genome vectors to 3D latent space. k=15 neighbors, fuzzy simplicial set, spectral initialization, SGD optimization (200 epochs). Full recompute when dataset grows 50%+, otherwise incremental k-NN interpolation.
- **Discovery zone detection** (`server/landscape/zone-detector.ts`): 8x8x8 voxel grid over latent space. Per-voxel density, SC-probability (Tc>20K fraction), Tc gradient (steepest increase direction). Zone scoring: tcProximity*0.35 + lowDensity*0.25 + gradient*0.2 + scFraction*0.2. Top zones identified for targeted exploration.
- **Landscape guidance** (`server/landscape/landscape-guidance.ts`): Converts zone data into generator biases. RL element-group weights from high-Tc zone compositions. Inverse design seeds from zone-proximal materials. Diffusion guidance from zone structural motifs. Zone-based Tc bonus (up to ~15% of zone score) for candidates in discovery zones.
- **Pipeline integration**: Materials added to landscape in Phase 10 and autonomous loop. Landscape updated every 5 cycles. Zone bonus applied to candidate Tc scoring. Stats logged in autonomous loop summary.
- **APIs**: `GET /api/landscape/embedding` (3D points), `/api/landscape/zones` (discovery zones), `/api/landscape/stats` (statistics), `/api/landscape/guidance` (generator biases).

### Constraint-Based Physics Solver
- **Backward McMillan/Eliashberg solver** (`server/inverse/constraint-solver.ts`): Given a target Tc, binary-searches for required electron-phonon coupling (lambda) and logarithmic phonon frequency (omegaLog) ranges.
- **4 additional constraint solvers**:
  - **DOS Constraint**: Solves required DOS(Ef) from target Tc (strong SC > 4 states/eV, weak < 2), with orbital character and VHS proximity requirements.
  - **Phonon Frequency Constraint**: Determines required bond stiffness, light-element fraction, Debye temperature range, and max avg atomic mass from target omegaLog.
  - **Electron-Phonon Coupling Constraint**: Computes required Hopfield parameter, orbital overlap, bonding network type, and phonon softness from target lambda and DOS.
  - **Charge Transfer Constraint**: Identifies when unconventional charge-transfer mechanism is needed (low omegaLog + high lambda or layered high-Tc), with donor/acceptor candidates and interlayer coupling requirements.
- **Composite feasibility scoring**: Weighted average across all constraint dimensions (lambda/omega 30%, DOS 20%, phonon 20%, coupling 20%, charge transfer 10%).
- **Formula evaluation**: Evaluates any formula against all constraints, returning per-constraint satisfaction (5 boolean checks), gap analysis, Hopfield estimates, and a composite match score.
- **Generator guidance**: Provides constraint-aware guidance including DOS range, phonon-preferred elements, charge transfer requirement, and Hopfield parameter range.
- **APIs**: `GET /api/constraint-solver/solve?targetTc=200&muStar=0.10&pressure=0`, `GET /api/constraint-solver/evaluate/:formula?targetTc=200&pressure=0`

### Pressure-to-Ambient Pathway Search
- **Stabilization strategies** (`server/inverse/pressure-pathway.ts`): For high-pressure superconductors, generates isovalent substitution, chemical doping, and anion substitution pathways to stabilize at ambient pressure.
- **Tc retention estimation**: Estimates how much Tc is retained after pressure reduction, with feasibility scoring.
- **Pipeline integration**: Triggered in autonomous loop when Tc > 50K and pressure > 10 GPa. Best ambient candidates logged.
- **APIs**: `GET /api/pressure-pathways/search/:formula?tc=250&pressure=180`, `GET /api/pressure-pathways/stats`

### Physics Constraint Graph Solver
- **Coupled constraint system** (`server/inverse/constraint-graph-solver.ts`): Solves all SC physics constraints simultaneously as a graph rather than sequentially.
- **15 parameter nodes**: Tc, lambda, omegaLog, DOS, phonon_softness, hopfield_eta, nesting, charge_transfer, structure, pressure, element_mass, bond_stiffness, orbital_character, debye_temp, gap_ratio.
- **29 coupling edges**: Representing physical dependencies (McMillan, Allen-Dynes, Hopfield, etc.) between parameters.
- **Constraint propagation**: Starting from target Tc, narrows parameter ranges through the coupled graph.
- **8 feasibility regimes**: conventional-BCS, strong-coupling-metal, light-element-compound, hydride-moderate-pressure, superhydride, unconventional-layered, kagome-flat-band, topological-SC.
- **8 rare chemical space regions**: ternary hydrides, borohydrides, nickelates, kagome VHS, heavy-fermion, topological heterostructures, HEA, carbon intercalated.
- **Pipeline integration**: Graph guidance applied every 5th autonomous cycle for regime and element suggestions.
- **APIs**: `GET /api/constraint-graph/solve?targetTc=200`, `GET /api/constraint-graph/feasible-regions?targetTc=200`

### Synthesis Pathway Modeling
- **Multi-step reaction pathway simulation** (`server/synthesis/reaction-pathway.ts`): Models precursors → intermediates → target phase with thermodynamics.
- **6 synthesis methods**: solid-state reaction, arc-melting, high-pressure (DAC/laser heating), ball-milling, CVD, magnetron sputtering.
- **Per-step detail**: reactants, products, temperature, pressure, atmosphere, reaction type, duration, lab notes.
- **Thermodynamic scoring**: Gibbs free energy chain (Miedema model), kinetic barriers (Arrhenius), metastable quenching feasibility.
- **Family-aware routing**: Different synthesis methods prioritized per material family (Cuprates, Hydrides, Intermetallics, etc.).
- **Pipeline integration**: Triggered in Phase 12 (multi-fidelity) when candidates pass stage 4.
- **APIs**: `GET /api/synthesis-pathway/:formula`, `GET /api/synthesis-pathway/stats`

### Band Structure Neural Operator
- **Full E(k) dispersion predictor** (`server/physics/band-structure-operator.ts`): Predicts complete band structure along high-symmetry k-paths, not just features.
- **Output**: Energy values at ~50 k-points per band, for up to 12 bands near the Fermi level. Supports cubic, hexagonal, and tetragonal lattices.
- **Derived quantities**: effective masses, Fermi velocities, band curvatures, exact VHS positions, nesting vectors, Berry phase proxy, Z2 topological index, band inversion count.
- **Physics calibration**: Reference band structures for MgB2 (sigma bands), YBa2Cu3O7 (flat bands), LaH10 (multi-band), FeSe (hole/electron pockets), Nb3Sn (A15 narrow bands).
- **Pipeline integration**: Called after band surrogate in Phase 10, enriches Fermi surface analysis with ML features.
- **APIs**: `GET /api/band-operator/:formula`, `GET /api/band-operator/dispersion/:formula`, `GET /api/band-operator/stats`

### Autonomous Hypothesis Engine
- **Scientific theory generation & testing** (`server/theory/hypothesis-engine.ts`): AI that invents, tests, and ranks superconductivity theories automatically.
- **Pattern discovery**: Three analysis engines:
  - Feature correlation analysis (which features cluster in high-Tc materials)
  - Conditional rule discovery (IF nesting > 0.6 AND DOS > 4 THEN Tc > 50K)
  - Co-occurrence pattern mining (structural/compositional patterns correlating with SC)
- **Hypothesis structure**: ID, natural language statement, mathematical form, supporting evidence, confidence score, test/support/refute counts, required conditions, predicted Tc range, status (proposed/testing/supported/weakened/refuted).
- **Testing loop**: Evaluates materials meeting hypothesis conditions, computes support score, updates confidence via Bayesian update, marks status based on evidence threshold.
- **Generator bias**: Top hypotheses produce preferred conditions, feature targets, and family preferences to guide material generation.
- **Pipeline integration**: `runHypothesisCycle()` runs every 10 autonomous cycles. Stats included in autonomous loop stats.
- **APIs**: `GET /api/hypothesis/active`, `GET /api/hypothesis/all`, `GET /api/hypothesis/test/:id`
- Files: `server/theory/hypothesis-engine.ts`

### Discovery Landscape Intelligence
- **Frontier analysis, novelty scoring, and strategic exploration** (`server/landscape/landscape-intelligence.ts`): Enhanced landscape intelligence beyond UMAP + zones.
- **Frontier analysis**: Convex hull vertex detection, 12-direction frontier scoring with Tc gradient extrapolation, discovery corridors (steepest Tc increase directions), explored volume fraction.
- **Novelty scoring**: Per-candidate novelty = nearest neighbor distance (40%) + local density (35%) + family dissimilarity (25%). Returns `isInExploredRegion` flag.
- **Zone intelligence**: UCB acquisition function (Tc_mean + 1.5 * uncertainty), zone evolution tracking (improving/stable/declining/new), cross-zone correlation (shared elements), suggested elements/structures/stoichiometries per zone.
- **Exploration strategy**: Interpolation candidates between high-Tc materials from different families, bridge candidates connecting high-Tc clusters, budget allocation proportional to acquisition scores, natural language recommendation.
- **Pipeline integration**: `updateZoneHistory()` called every 5 cycles. Intelligence stats logged in discovery landscape update events.
- **APIs**: `GET /api/landscape-intelligence/frontier`, `GET /api/landscape-intelligence/novelty/:formula`, `GET /api/landscape-intelligence/zones`, `GET /api/landscape-intelligence/strategy`
- Files: `server/landscape/landscape-intelligence.ts`

### Quantum Criticality Detector
- **Unified quantum critical point (QCP) detector** formalizing existing spin susceptibility, CDW, SDW, Mott detection into a single QuantumCriticalScore.
- **Six QCP channels**: Mott (Hubbard U/W ratio), SDW (Stoner enhancement + nesting), CDW (nesting + DOS), nematic (orbital anisotropy), structural (soft phonon modes), orbital-selective (mixed d-orbital).
- **Dome model**: SC enhancement follows Gaussian profile around optimal QCP proximity — boosts Tc when near but not past the transition. Optimal at control parameter ~0.75, suppressed past 0.95 (ordered phase).
- **Known calibration**: Cuprates → Mott QCP (score ≥ 0.90), Fe-pnictides → SDW (0.85) + Mott, NbSe2 → CDW (0.89), nickelates → Mott (0.75) + orbital-selective.
- **Pipeline integration**: Runs in Phase 10 (physics evaluation) and autonomous loop. QC score > 0.5 with pairingBoost > 0.1 applies Tc enhancement (capped at 15% boost, max 400K).
- **API**: `GET /api/quantum-criticality/:formula`
- Files: `server/physics/quantum-criticality.ts`

### Discovery Memory System
- **Pattern-based learning system** that remembers which physics patterns produced high Tc discoveries.
- **14-dim fingerprint**: DOS level, flat band score, VHS proximity, nesting score, coupling strength, hydrogen density, dimensionality, element classes, orbital character, pairing channel, correlation strength, metallicity, pressure, family.
- **Memory operations**: `recordDiscovery()` (Tc ≥ 20K threshold), `queryPatternSimilarity()` (cosine similarity k-nearest), `getTopPatterns()`, `biasGenerationFromMemory()` (element/structure/stoichiometry preferences from top clusters).
- **RL integration**: Memory reward bonus (capped at 0.3) applied to RL agent reward when pattern matches successful discoveries. Auto-clustering at 0.75 cosine similarity threshold.
- **Capacity**: 500 records max, pruned by Tc ranking. Automatic clustering for generation bias.
- **API**: `GET /api/discovery-memory/patterns`
- Files: `server/learning/discovery-memory.ts`

### Band Structure Neural Network Surrogate
- **GNN-based band structure predictor** inspired by DeepDFT/OrbNet/ALIGNN/M3GNet approaches.
- **Input**: Crystal structure graph (20-dim node features, 7-dim edge features, 3-body interactions) from prototype or generic construction.
- **Output heads**: bandGap (direct/indirect), flatBandScore, vhsProximity, nestingFromBands, dosPredicted, fsDimensionality, multiBandScore, bandwidthMin, bandTopologyClass (trivial/TI/Dirac/Weyl), highSymmetryEnergies (Γ/X/M/R/Z/A).
- **Architecture**: 3 attention message passing layers + 3-body interaction layer → mean+max pooling → 2-layer MLP with multi-head outputs.
- **Physics calibration**: Post-GNN heuristic corrections for known material classes (cuprates enhance VHS/nesting, hydrides enhance multiBand, pnictides enhance nesting).
- **Pipeline position**: Runs after crystal structure assignment, feeds predicted electronic features into pairing/coupling models before full Eliashberg physics.
- **API**: `GET /api/band-surrogate/:formula`
- Files: `server/physics/band-structure-surrogate.ts`

### Phase Stability Prediction Network
- **GNN-based stability pre-filter** for rapid screening before expensive physics calculations.
- **Input**: Composition graph (elements as nodes with 14-dim features including Miedema φ*/nws, Pettifor scale; compositional relationship edges with 8-dim features).
- **Output**: synthesizabilityScore (0-1), predictedFormationEnergy (eV/atom), stabilityClass (stable/metastable/unstable), decompositionRisk (0-1).
- **Heuristic knowledge**: Goldschmidt tolerance factor, Pettifor map proximity, electronegativity spread, element compatibility, prototype matching, valence mismatch, size ratio scoring, Miedema formation energy.
- **Pipeline position**: Runs FIRST in `insertCandidateWithStabilityCheck` and `runAutonomousDiscoveryCycle` before feature extraction, ML screening, structure prediction, and expensive physics. Rejects unstable candidates (synth < 0.25, decomp risk > 0.85, incompatible elements < 0.15).
- **Impact**: Reduces search space by filtering thermodynamically implausible candidates before expensive calculations.
- **API**: `GET /api/stability-predict/:formula`
- Files: `server/physics/stability-predictor.ts`

### Convex Hull Phase Diagram Engine
- Computes energy above hull, decomposition products, hull vertices, and assesses metastability for binary/ternary systems.

### Pressure Modeling Engine
- Calculates volume compression, bulk modulus, pressure derivative, predicts hydrogen uptake, and determines high-pressure stability and pressure-Tc curves.

### Graph Neural Network Surrogate (CGCNN/MEGNet/M3GNet-style) — PRIMARY PREDICTOR
- **Primary ML predictor** with 0.6 weight in ensemble scoring (GB reduced to 0.3 weight, 0.1 structural novelty).
- Uses prototype-aware graph construction, enhanced 20-dimensional node features (s/p/d/f orbital occupancy, magnetic moment proxy, valence shell encoding, stress/force/SOC descriptors), 7-dimensional edge features (with bond-angle encoding), and attention-weighted message passing with 3-body interaction layers for multi-target predictions (formation energy, phonon stability, Tc, confidence, electron-phonon lambda) with uncertainty estimation.
- **Orbital-aware features**: Explicit [s,p,d,f] occupancy channels, magnetic moment proxy via Hund's rule √(n(n+2)), valence shell filling fraction, enhanced piecewise-linear SOC model.
- MEGNet-style multi-body interactions: 3-body angle features between bonded triplets with angular weighting.
- GB retained as fast pre-filter (Tc < 5K rejection gate) but no longer drives ensemble scores.

### Reinforcement Learning Chemical Space Agent
- **REINFORCE policy gradient** agent that learns which elements, stoichiometries, and structures produce better superconductors.
- **State**: best Tc, family diversity, stagnation cycles, exploration budget, element success entropy.
- **Action space**: (1) element group pair (9 groups), (2) stoichiometry template (10 templates), (3) crystal structure type (20 prototypes), (4) layering dimension (3D-isotropic/quasi-2D/quasi-1D/mixed-dim), (5) hydrogen density (none/low/medium/high H/M), (6) electron count (low/mid/high/very-high VEC targets), (7) orbital configuration (s/p/d/f-dominant, sp-hybrid, sd-hybrid).
- **Policy**: softmax over learned weights with temperature annealing (1.0 → 0.3), epsilon-greedy exploration with decay (0.15 → 0.05), stagnation-boosted exploration.
- **Physics-aware rewards**: lambda range, metallicity, nesting, VHS proximity, dimensionality, hydrogen ratio, d/sd orbital character, band flatness, phonon frequencies, correlation strength.
- **Reward**: Tc improvement × 2.0 + relative improvement × 5.0 + pipeline bonus (1.0) + stability × 0.5 + novelty × 0.3 + physics-principle reward.
- **Experience replay**: 2000-entry buffer with periodic 32-sample batch replay for stable learning.
- **Element pair tracking**: Records element-group success rates and element-pair average Tc for contextual biasing. `elementPairSpecific` Map stores per-pair learned weights.
- **Known pair priors**: 20 explicit superconducting element pair priors (La+H, Fe+As, Nb+B, Cu+O, Y+Ba, Bi+Se, Mg+B, etc.) with calibrated initial biases.
- **Pair-weighted sampling**: `getWeightedElements` and `sampleWeightedElement` preferentially sample elements with higher pair priors and historical success rates during candidate generation.
- **Pair reward feedback**: Element pair specific weights updated based on observed Tc (Tc>50 reinforced, Tc<5 with sufficient samples penalized).
- **Physics-informed priors**: 4d-transition metals, hydride-rich stoichiometries, A15/ThCr2Si2 structures initialized with positive bias.
- Files: `server/learning/rl-agent.ts`

### Bayesian Optimization for Chemical Space Search
- **Gaussian Process surrogate** with Matern 5/2 kernel over composition feature vectors (62-dim: element fractions + structural descriptors).
- **Acquisition functions**: Upper Confidence Bound (UCB, beta=2.0), Expected Improvement (EI, xi=0.01), Thompson Sampling.
- **Mixed acquisition**: 0.4×UCB + 0.3×EI + 0.3×Thompson for balanced exploration/exploitation.
- **GP inference**: Cholesky decomposition for O(n^3) exact GP, capped at 200 training points for efficiency.
- **Observation management**: 500-entry LRU, retaining 70% top-Tc observations + 50% random sampling of remainder.
- **Integration**: BO ranks combined candidate pool (RL-generated + massive-generation) by acquisition value; top-50 BO-ranked candidates screened first, followed by remaining candidates.
- Files: `server/learning/bayesian-optimizer.ts`

### Crystal Diffusion Generator
- **Diffusion-inspired denoising**: Generates novel crystal structures by iteratively refining random atomic positions using physics-based score functions (Lennard-Jones repulsion, bond-length targets, symmetry constraints, wall confinement).
- **Symmetry enforcement**: Space group constraint projection during denoising (gradual Wyckoff position pull after 30% progress), symmetry-aware denoising preserving Wyckoff positions via symmetry operation forces (after 40% progress), symmetry penalty in scoring (50% space group ops + 25% inversion + 25% mirror).
- **Wyckoff position sampling**: Seeds atoms at high-symmetry Wyckoff sites from 8 space groups (Pm-3m, Fm-3m, Im-3m, P6/mmm, P4/mmm, I4/mmm, R-3m, Pnma) covering cubic, hexagonal, tetragonal, trigonal, and orthorhombic systems.
- **Composition strategies**: 8 sampling strategies — hydride, ternary, binary TM, quaternary, exotic, cage compound, layered, borocarbide — targeting SC-relevant chemistries.
- **Prototype matcher**: Classifies generated structures against 10 known prototypes using coordination number analysis and c/a ratio matching.
- **Physics validation**: Rejects structures with min bond length < 0.5 A, density outside 0.5-25 g/cm3, or lattice params outside physical bounds.
- **Integration**: Runs every 5 engine cycles, generates 30 structures per batch, feeds through GB/GNN scoring and stability gate, results added as BO observations.
- **API**: `POST /api/generate-crystal` (count, elements, targetTc params), `GET /api/diffusion-stats`.
- Files: `server/ai/crystal-generator.ts`

### Massive Candidate Generator
- Utilizes element substitution, composition interpolation, doped variants, and composition sweeps, with valence sanity filters and periodic table element validation.
- **Fractional doping generator**: `generateFractionalDopedVariants()` creates fine-grained doped variants with fractions [0.05..0.25] across 19 SC-relevant dopants. Seeds include known SC materials (MgB2, LaH10, NbN, etc.). Generates ~150 variants per cycle.
- Employs a rapid Gradient Boosting screen to pre-filter candidates.
- **RL+BO integration**: RL agent generates 30 directed candidates per cycle; combined pool ranked by BO acquisition function before screening.

### Physics-Aware ML Predictor
- Performs multi-target prediction (lambda, DOS(EF), omega_log, hull distance) with transfer priors and uncertainty estimation.
- Incorporates a self-reinforcing loop where predicted physics values feed into the Tc predictor.

### Pattern Mining / Theory Generator
- Extracts and validates quantitative rules from physical features, using rule aging and screening for candidate scoring.

### Structural Mutation Engine
- Assigns structure prototypes, generates distorted lattices, layered structures, vacancy structures, and strain variants.

### Multi-Dimensional Phase Explorer
- Scans composition, pressure, and temperature spaces with adaptive sampling and uncertainty-aware selection to identify optimal material conditions.

### Crystal Prototype Structure Generator
- Generates 10 distinct crystal structure types (e.g., AlB2, Perovskite, A15), producing over 807 structurally-typed candidates with detailed crystallographic information.

### Convex Hull Stability Gate
- Enforces a hard rejection criteria for candidates with hull distance > 0.1 eV/atom, with additional metastability assessment for borderline cases.

### Discovery Score
- Unified composite metric: `0.35 Tc + 0.20 stability + 0.15 novelty + 0.10 synthesis + 0.10 topology + 0.10 uncertainty_bonus`.
- Uncertainty bonus encourages exploration of under-sampled chemical regions (scaled 1.2× from raw uncertainty estimate).
- Prioritizes higher-scoring candidates for DFT enrichment.

### Active Learning Loop
- Uses uncertainty-driven DFT selection with analytical DFT fallback when external data is unavailable.
- Budget: 20 DFT runs per cycle, triggered every 15 cycles.
- GNN retrain triggers: 15+ enriched candidates since last retrain, OR avg uncertainty > 0.3, OR first retrain with any DFT success.
- Retrains the GNN surrogate with expanded datasets and monitors convergence.
- **Real DFT tier**: Top 3 candidates per AL cycle are computed with GFN2-xTB (xtb v6.7.1) for real quantum-mechanical total energy, HOMO-LUMO gap, formation energy, and metallicity assessment.

### Real DFT Backend (GFN2-xTB)
- **Engine**: xtb v6.7.1 (Grimme group) at `server/dft/xtb-dist/bin/xtb`.
- **Method**: GFN2-xTB — semi-empirical density functional tight binding, providing real quantum-mechanical energies.
- **Outputs**: Total energy (Hartree), HOMO-LUMO gap (eV), formation energy (eV/atom), metallicity assessment, dipole moment, Mulliken charges.
- **Integration**: Runs on EVERY candidate in the main screening pipeline via `runXTBEnrichment()` in `dft-feature-resolver.ts`. Falls back to analytical estimates only when xTB fails or formula is unsupported.
- **DFT Source Type**: `"dft-xtb"` — tracked as a distinct source in DFTResolvedFeatures alongside `"analytical"`, `"mp"`, `"oqmd"`, `"aflow"`, `"nist"`.
- **Performance**: ~5 seconds per calculation for typical 3-20 atom structures. ~85% success rate.
- **Structure generation**: 20 crystallographic prototypes (A15, AlB2, NaCl, Perovskite, ThCr2Si2, Heusler, BCC, FCC, Layered, Kagome, HexBoride, MX2, Anti-perovskite, CsCl, Cu2Mg-Laves, Fluorite, Cr3Si, Ni3Sn, Fe3C, Spinel) with real fractional coordinates and lattice parameters. Approximate prototype matching (ratio score < 0.5) for imperfect stoichiometries. Generic cluster fallback for fully unmatched cases. Structures capped at 20 atoms with proportional scaling.
- **Geometry optimization**: Runs `xtb --gfn 2 --opt tight` before single-point calculations (30s timeout). Optimized geometry used for all subsequent energy and phonon calculations. Cached in `optimizedStructureCache`.
- **Finite-displacement phonon calculator**: For structures ≤8 atoms, displaces each atom ±0.01 Å in x/y/z, runs xTB on each displaced structure, builds force constant matrix from finite differences (6N+1 calculations). Computes dynamical matrix D(q) at arbitrary q-points along high-symmetry paths. Produces phonon dispersion, phonon DOS, ω_log, and full Brillouin zone stability assessment. Uses optimized geometry when available.
- **Phonon stability check**: For 9-12 atom structures, falls back to xTB `--hess` Hessian calculation. IMAG_THRESHOLD = -2000 cm⁻¹ (relaxed from -50 to accommodate xTB tolerance). Mild imaginary modes (-50 to -2000 cm⁻¹) accepted without penalty; severe modes (< -2000 cm⁻¹) penalized (0.08 per mode above 1, max 0.2). Results cached (100 entries).
- **Formation energy**: Computed relative to molecular/dimer reference calculations using MOLECULAR_BOND_LENGTHS (H: 0.74 Å, N: 1.10, O: 1.21 Å) for accurate reference energies. Sanity guard: |Ef| > 15 eV/atom is discarded as unphysical. Uses actual DFT atom count (not formula count) to handle scaled structures correctly.
- **Cache**: In-memory LRU caches for DFT results (200 entries) and elemental reference energies.
- **Stats API**: `getXTBStats()` exposes runs, successes, cacheSize, refElements via `/api/dft-status`.

### Inverse Design Optimizer Engine
- **5-layer architecture**: Target property interface, goal-driven candidate generator, pipeline integration, inverse learning system, closed-loop optimization.
- **Target property schema**: `TargetProperties` interface with targetTc, maxPressure, minLambda, maxHullDistance, metallicRequired, phononStable, preferred/excluded elements and prototypes.
- **Campaign system**: Multiple simultaneous inverse design campaigns with independent learning states. DB-persisted via `inverse_design_campaigns` table.
- **Constraint-driven generator**: Maps target properties to composition bias rules (high Tc → light elements + TM, high lambda → covalent bonding). 12 prototype-Tc affinity mappings (A15, AlB2, Perovskite, ThCr2Si2, Heusler, Layered, Kagome, NaCl, BCC, FCC, Clathrate, Fluorite). Element substitution and stoichiometry sweep for refinement.
- **Inverse learning**: Element success matrix, pair success matrix, prototype success matrix tracking total reward and avg distance. Reward = exp(-distance/0.3) for gradual improvement. Composition bias derived from learned weights. Stagnation detection with randomized exploration.
- **Target distance metric**: 0.50 Tc + 0.20 lambda + 0.15 hull + 0.15 pressure (weighted, normalized).
- **Closed-loop optimizer**: Generate → evaluate in existing pipeline (GB/stability gate) → compare to target → update biases → refine top performers → repeat. Convergence detection when best distance range < 0.005 over 10 cycles.
- **Pipeline integration**: Runs every 8 engine cycles. Inverse candidates fed through same stability gate and scoring as all other candidates. 30 fresh + 15 refined per cycle.
- **API**: `POST /api/inverse-design/start`, `GET /api/inverse-design/campaigns`, `GET /api/inverse-design/campaign/:id`, `DELETE /api/inverse-design/campaign/:id`, `POST /api/inverse-design/campaign/:id/pause`, `GET /api/inverse-design/stats`.
- Files: `server/inverse/target-schema.ts`, `server/inverse/inverse-generator.ts`, `server/inverse/inverse-learning.ts`, `server/inverse/inverse-optimizer.ts`

### Differentiable Materials Design (Gradient-Based Inverse Physics)
- **Analytic McMillan gradients**: Computes ∂Tc/∂λ, ∂Tc/∂ωlog, ∂Tc/∂μ* analytically from the Allen-Dynes equation. Full chain rule through prefactor and exponent terms.
- **Numerical element gradients**: Finite-difference ∂Tc/∂composition by perturbing each element count ±1 and re-evaluating the full physics pipeline (electronic structure → phonon spectrum → electron-phonon coupling → Tc).
- **Gradient-to-composition mapping**: Interprets gradient signals into discrete composition actions — add light covalent elements (boost phonon freq), add high-coupling TM (boost λ), increase DOS, improve metallicity. Substitution groups for stagnation escape (e.g., Nb↔V↔Ta, La↔Y↔Ce).
- **Loss function**: Weighted L2 loss: 0.60 Tc gap + 0.15 lambda deficit + 0.10 metallicity + 0.15 stability.
- **Iterative optimizer**: Up to 20 gradient-descent steps per seed. Stagnation detection after 4 non-improving steps triggers element substitution from chemically similar groups. Atom count capped at 20 with proportional scaling.
- **Seed generation**: Target-Tc-aware seed selection (>200K: hydride seeds like LaH10/YH6; >100K: A15 seeds like Nb3Ge; >30K: MgB2/NbN).
- **Pipeline integration**: Runs every 12 engine cycles using active campaign targets. 4 seeds × 12 steps per campaign. Results passing GB gate (Tc≥10K) inserted as SC candidates.
- **API**: `POST /api/gradient-design/optimize` (single formula), `POST /api/gradient-design/batch` (multi-seed), `GET /api/gradient-design/stats`.
- File: `server/inverse/differentiable-optimizer.ts`

### 8-Pillar SC Optimizer (Multi-Objective Superconductivity Targeting)
- **8 SC pillars**: (1) Electron-phonon coupling λ≥1.5, (2) Phonon frequencies ω_log≥700K, (3) DOS at Fermi level ≥2.0, (4) Fermi nesting ≥0.5, (5) Structural motifs (cage/layered/kagome), (6) Pairing glue strength ≥0.5, (7) Electronic instability proximity ≥0.4, (8) Hydrogen cage geometry ≥0.5 (hydrides only, 7 active pillars for non-hydrides).
- **Pairing glue score**: Composite of phonon (50%), spin-fluctuation (25%), charge-fluctuation (10%), excitonic (15%) contributions. Uses `computeDynamicSpinSusceptibility` for Stoner enhancement and QCP proximity. Cuprates/pnictides correctly identify spin-fluctuation as dominant mechanism.
- **Electronic instability proximity**: Combines van Hove proximity (20%), nesting (15%), DOS (10%), spin susceptibility (15%), Mott proximity (15%), CDW (10%), SDW (10%), structural (5%). Uses competing phase evaluation for SDW detection. NbSe2 correctly flags CDW, BaFe2As2 flags SDW.
- **Hydrogen cage geometry**: Evaluates H-network dimensionality (1D-3D), cage score (sodalite/clathrate detection), H-H bond distribution, cage symmetry, H coordination. LaH10 scores 0.98 (metallic-network sodalite), CaH6 scores 0.96. Non-hydrides get pillar weight redistributed to other 7 pillars.
- **Fermi surface geometry** (`FermiSurfaceGeometry`): Three SC-critical FS shape detectors that feed into existing pillar scores:
  - *Cylindrical (2D) FS*: `cylindricalScore` (0-1), `kzVariance`, `fsDimensionality` (2.0=quasi-2D, 3.0=3D). Detects layered structures (cuprates=0.95, pnictides=0.90, dichalcogenides=0.85). Uses `computeDimensionalityScore` + FS topology.
  - *Nested electron-hole pockets*: `nestingStrength` (0-1), `electronHolePocketOverlap` (0-1), `nestingVectorQ` (e.g., "(pi,pi)"). BaFe2As2=1.0 with Q=(pi,pi), NbSe2=0.70 with Q=(2/3pi,0). Enhances nesting pillar.
  - *Van Hove saddle points*: `vanHoveDistance` (eV from EF), `vanHoveSaddleCount`, `vanHoveNearFermi` (true if <0.05eV). Uses tight-binding VHS detection. YBa2Cu3O7=0.0015eV, H3S=0.0026eV. Boosts DOS pillar.
  - `compositeFSScore` = 0.35*cylindrical + 0.35*nesting + 0.30*vanHove. Feeds into nesting (+40% boost), DOS (+15% vH boost), structure (+10% cylindrical boost), pairingGlue (+10% composite) pillars.
- **Pillar evaluation**: `evaluatePillars()` computes all 8 pillar scores (0-1 each) + FS geometry, composite weighted fitness, satisfied count, weakest pillar identification. Non-hydrides automatically redistribute H-cage weight across other pillars.
- **13 design templates**: clathrate-hydride, metal-boride, metal-carbide, metal-nitride, ternary-hydride, high-DOS-intermetallic, layered-pnictide, cuprate-layered, sodalite-superhydride, spin-fluctuation-pnictide, mott-proximate, dichalcogenide-nested, nickelate-layered.
- **Adaptive pillar weights**: 8 weights (coupling=0.18, phonon=0.12, dos=0.12, nesting=0.10, structure=0.10, pairingGlue=0.18, instability=0.10, hydrogenCage=0.10). RL-learned via Tc delta from running baseline (not raw Tc).
- **Weakness-targeted mutation**: 8 mutation strategies including pairingGlue (add magnetic elements + pnictogens), instability (add CuO planes or TM), hydrogenCage (increase H count + add cage formers).
- **Pipeline integration**: Runs every 9 engine cycles. Adaptive targets scale with targetTc. Inserts candidates with Tc≥8K.
- **API**: `POST /api/sc-pillars/evaluate` (returns full PairingGlueBreakdown, InstabilityBreakdown, HydrogenCageMetrics, FermiSurfaceGeometry), `POST /api/sc-pillars/generate` (all scores + FS data), `GET /api/sc-pillars/stats`.
- File: `server/inverse/sc-pillars-optimizer.ts`

### Physics-Constrained Generative AI (Rule-Aware Material Creation)
- **Constraint checks**: Charge neutrality (with metallic compound bypass), atomic radius compatibility, coordination number limits, bond stability (electronegativity spread, composition dominance, atom count), electron count rules, noble gas rejection, stoichiometry excess.
- **Metallic compound awareness**: Intermetallics (Nb3Sn), hydrides (LaH10, CaH6), borides (MgB2), and metal-metalloid compounds bypass ionic charge balance since bonding is metallic/covalent.
- **Hydrogen handling**: H bonds treated specially — no radius incompatibility violations for H pairs, coordination bypass, compatible in all metallic contexts.
- **Constraint-guided generation**: `constraintGuidedGenerate()` processes raw formula arrays, splitting into valid/repaired/rejected. Invalid formulas are auto-repaired when possible (charge balance adjustment, stoichiometry reduction, noble gas removal).
- **RL integration**: `updateConstraintWeightsFromReward()` adjusts constraint weights based on Tc reward from the autonomous loop. Constraints that reject good SCs get softened; constraints that pass bad candidates get strengthened.
- **Pipeline integration**: Inserted into autonomous fast-path between massive generation and screening loop. Filters all candidates before they enter the expensive physics pipeline.
- **API**: `POST /api/physics-constraints/check` (single formula), `POST /api/physics-constraints/batch` (up to 200 formulas), `GET /api/physics-constraints/stats`.
- File: `server/inverse/physics-constraint-engine.ts`

### Structure-First Design (Structure Diffusion)
- **Motif library**: 12 SC structural motifs (CuO2-plane, FeAs-layer, clathrate-cage, A15-chain, kagome-flat, hexagonal-layer, perovskite-3D, layered-hydride, NaCl-rocksalt, H-channel, breathing-kagome, Laves-MgZn2). Each motif defines site roles, space group, Tc range, SC affinity, and pairing mechanism.
- **Structural embedding**: 8D vectors encoding layering, cage-character, H-density, correlation, flatness, bond-covalency, dimensionality, spin-orbit.
- **Design flow**: Motifs selected by SC affinity + learned weights → site roles filled with element candidates → combinatorial evaluation → GB Tc prediction → ranked by structural score.
- **Learned motif weights**: Updated by Tc reward signal (motifWeight += 0.01 * normalized_tc). Successful motifs get preferentially selected in future cycles.
- **Pipeline integration**: Runs every 7 engine cycles. Generates formulas from top motifs, evaluates via GB model, inserts candidates with Tc≥10K.
- **API**: `POST /api/structure-design/generate` (params: targetTc, motifCount, elementsPerSite), `GET /api/structure-design/motifs`, `GET /api/structure-design/stats`.
- File: `server/ai/structure-diffusion.ts`

### Chemical Synthesis Realism
- **Precursor availability scoring**: ~70-element lookup table (COMMON_ELEMENTS) mapping elements to availability scores (1.0 for Fe/Al/Si/O down to 0.2 for Os/Ir). Weighted by compositional fraction.
- **Family-specific synthesis defaults**: Calibrated per-family base scores, reaction temperatures, pressure requirements, and atmosphere complexity (e.g., hydrides: base 0.25, 150 GPa; MAX-phases: base 0.70, no pressure).
- **Reaction temperature factor**: Penalizes materials where synthesis temperature approaches or exceeds constituent melting points.
- **Phase competition penalty**: Combines hull distance and competing phase count. Materials with hull > 0.2 eV or > 5 competing phases receive significant penalties.
- **Pressure penalty**: High-pressure synthesis requirements (especially hydrides with high H/metal ratios) penalized for difficulty and cost.
- **Exported**: `computeSynthesisScore()` with `SynthesisScoreBreakdown` interface for transparency.
- Files: `server/learning/family-filters.ts`

### Analytical Physics Estimators (coverage ~1.00)
- **Debye temperature**: θD ≈ 41.6 * sqrt(B / ρ) when elemental data unavailable.
- **Bulk modulus**: Estimated from melting point (B ≈ 0.07 * T_melt) when elemental data missing.
- **Density**: Computed from atomic masses and radii with BCC packing assumption.
- **DOS at Fermi level**: Estimated from valence electron count with transition metal boost.
- **Average phonon frequency**: Derived from Debye temperature and average atomic mass.
- **Atomic packing fraction**: Geometric estimate from atomic radii.
- Coverage metric tracks how many of 11 properties have valid estimates.

### Autonomous Discovery Loop
- A massive generation pipeline generating 500-2000 candidates per cycle, undergoing multi-stage filtering (GB pre-screen, pattern mining, physics ML pre-filter) before a full pipeline processing.
- **Tiered Acceptance System**: Tier 1 (Tc>70K, λ>1.2, hull<0.10) → high confidence; Tier 2 (Tc>25K, λ>0.5, hull<0.20) → medium; Tier 3 (Tc>10K, λ>0.3) → low. Tier 2/3 bypass stability gate.
- **Exploration Probability**: 30% of cycles randomly explore an underexplored family instead of the strategy-recommended one. Pool: Pnictides, Chalcogenides, Cuprates, Hydrides, Kagome, Sulfides, Intermetallics, Alloys, Oxides, Nitrides.
- **Unconventional Seeds**: scBiasedSeeds reduced to 4 random per cycle; 8 unconventional seeds (FeSe, FeAs, BaFeAs, KVSb, LaNiO, etc.) injected each cycle.
- **Pressure Exploration**: Non-hydride metallic candidates with Tc>20K scanned at 5-50 GPa; pressure-enhanced Tc recorded with optimal pressure.
- Pass rate ~46% after tiered acceptance fix.

### Semantic Insight Deduplication
- Uses OpenAI text-embedding-3-small embeddings with cosine similarity (threshold 0.85) to reject semantically duplicate insights before LLM novelty scoring.
- Category quotas (2 per 30-min window) across 10 categories: novel-correlation, new-mechanism, cross-domain, computational-discovery, design-principle, phonon-softening, fermi-nesting, charge-transfer, structural-motif, electron-density.
- NLP prompts instruct diverse topic rotation: phonon softening, Fermi surface nesting, charge transfer layers, structural motifs, electron density redistribution, spin-orbit coupling, pressure-dependent phonon hardening.

### Kagome Metals Family
- Generates AV3X5, AV3X4, A2V3X5 stoichiometries with alkali/alkaline-earth A-sites and pnictogen/metalloid X-sites.
- Family filter checks frustrated lattice metal count, pnictogen sublattice, DOS at Fermi, van Hove singularity proximity, 2D character, and electron-phonon coupling.
- Seeds: KV3Sb5, CsV3Sb5, RbV3Sb5 and variants.

### Layered Structure Generators
- **Layered Chalcogenides**: MX2/MX3/M2X3 (NbSe2, TaS2, MoSe2 pattern) with intercalated variants (Li/Na/K/Ca/Sr/Cu into MX2 hosts).
- **Layered Pnictides**: Iron pnictide patterns — RE-TM-Pn-O (1111-type), AE-TM2-Pn2 (122-type), plus binary TM-Pn.
- **Intercalated Layered**: MX2 intercalation compounds, graphite intercalation compounds (AC6/AC8/AC12), intercalated oxide layers.

### Mixed-Mechanism Systems
- Targets Fe/Ni/Co/Cu in layered compounds for phonon + magnetic fluctuation superconductivity.
- Generates FeAs-based, FeSe-based, NiO-based (infinite-layer nickelates), CuO-based (cuprates), and combinatorial TM-chalcogenide/pnictide.
- Filter checks for magnetic TM presence, λ ≥ 0.3, magnetic fluctuation proximity, and layered character.

### Tight-Binding Electronic Structure Model
- **Slater-Koster tight-binding Hamiltonian**: Builds H(k) matrix from tabulated hopping parameters for s, p, d orbitals based on elemental properties.
- **Band structure solver**: Solves H(k) along high-symmetry k-paths (Γ-X-M-Γ for cubic, Γ-K-M-Γ for hexagonal, Γ-H-P-Γ-N for BCC).
- **Wannier projection**: Extracts orbital-resolved band character, effective hopping parameters, and Wannier spread/localization metrics.
- **Band topology detection**: Identifies flat bands (bandwidth < 0.1 eV), van Hove singularities (∂²E/∂k² → 0), Dirac crossings (linear dispersion), and topological band inversions (parity eigenvalue check).
- **TB Confidence Score**: `tbConfidence = structurePrototypeScore × elementCoverage × orbitalCompleteness`. structurePrototypeScore: 1.0 for known lattices (BCC/FCC/HEX), 0.5 for guessed. elementCoverage: fraction of elements with proper Slater-Koster params. orbitalCompleteness: reduced for actinides (0.6×), rare earths (0.75×), TM oxides (0.85×).
- **TB DOS**: Histogram-based DOS from band eigenvalues with Gaussian broadening. Blended into physics-engine DOS estimate (60% heuristic + 40% TB-derived).
- **Integration**: Called in `computeElectronicStructure()` to refine flatBandIndicator, vanHoveProximity, topologicalBandScore, and densityOfStatesAtFermi. Results cached (500 entries).
- Files: `server/learning/tight-binding.ts`, `server/learning/physics-engine.ts`

### Flat-Band Detection
- Computes DOS(EF)/avgDOS ratio as flat-band indicator.
- Ratio > 3 = strong flat-band (indicator ≥ 0.7), > 2 = moderate (≥ 0.3).
- Cuprates get 0.8 floor, Kagome 0.7, TM with high bandFlatness 0.5.
- Flat-band indicator > 0.5 boosts lambda by up to 40% (flatBoost = 1 + (fbi - 0.5) * 0.8).
- **Flat band logging**: Emits log event when flatBandIndicator > 0.3 with bandFlatness, DOS(EF), and mottProximity details.

### Bayesian Family Strategy Scoring
- Uses Bayesian shrinkage (prior_count=5) to prevent single-candidate families from dominating. Includes exploration bonus for under-explored families (<10 candidates).

### NLP Engine
- Generates cross-property correlation insights and superconductor correlation analysis, rejecting pure statistical summaries.

### Alive Engine
- Provides real-time feedback via WebSocket, adjusts tempo based on research progress, and maintains research memory with dynamic status messages.

### Infrastructure
- Includes rate limiting, an in-memory cache, optimized database indexes, API pagination, ML calibration with confidence bands, and cross-validation with external APIs.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.