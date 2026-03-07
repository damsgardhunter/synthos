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

### Quantum Pairing Mechanism Simulator
- **4 pairing channels**: phonon (mode-resolved λ_qν from dispersion branches), spin-fluctuation (χ(q) from DOS(EF)×nestingScore×correlationFactor via Hubbard U/W), orbital-fluctuation (d-orbital degeneracy, inter-orbital hopping, Hund's coupling), excitonic (DOS(EF)/bandGap proxy, electron-hole asymmetry, dielectric screening).
- **Mode-resolved phonon coupling**: Decomposes total λ into acoustic/optical/high-freq/soft-mode contributions per dispersion branch with frequency ranges.
- **Material-aware weighting**: Cuprates → 55% spin + 25% orbital; pnictides → 40% spin + 35% orbital; hydrides → 75% phonon; excitonic candidates → 40% excitonic.
- **Pairing symmetry inference**: d-wave (dx2-y2) for cuprates, s+/- for pnictides, anisotropic s-wave for soft-mode phonon, s-wave for conventional.
- **Composite pairing strength**: Weighted sum across all 4 channels, integrated into autonomous loop candidate notes.
- **Feature vector export**: 6-dimensional pairing feature vector (phononPairing, spinPairing, orbitalPairing, excitonPairing, dominantType, composite) for ML integration.
- **API**: `GET /api/pairing/profile/:formula` (full 4-channel analysis), `GET /api/pairing/features/:formula` (ML feature vector).
- Files: `server/physics/pairing-mechanisms.ts`

### Advanced Quantum Physics Modeling
- Computes phonon dispersion, phonon DOS, Eliashberg spectral function, GW many-body corrections, dynamic spin susceptibility, and Fermi surface nesting.

### Convex Hull Phase Diagram Engine
- Computes energy above hull, decomposition products, hull vertices, and assesses metastability for binary/ternary systems.

### Pressure Modeling Engine
- Calculates volume compression, bulk modulus, pressure derivative, predicts hydrogen uptake, and determines high-pressure stability and pressure-Tc curves.

### Graph Neural Network Surrogate (CGCNN/MEGNet/M3GNet-style) — PRIMARY PREDICTOR
- **Primary ML predictor** with 0.6 weight in ensemble scoring (GB reduced to 0.3 weight, 0.1 structural novelty).
- Uses prototype-aware graph construction, enhanced 16-dimensional node features (including M3GNet-inspired stress/force/SOC descriptors), 7-dimensional edge features (with bond-angle encoding), and attention-weighted message passing with 3-body interaction layers for multi-target predictions (formation energy, phonon stability, Tc, confidence, electron-phonon lambda) with uncertainty estimation.
- MEGNet-style multi-body interactions: 3-body angle features between bonded triplets with angular weighting.
- GB retained as fast pre-filter (Tc < 5K rejection gate) but no longer drives ensemble scores.

### Reinforcement Learning Chemical Space Agent
- **REINFORCE policy gradient** agent that learns which elements, stoichiometries, and structures produce better superconductors.
- **State**: best Tc, family diversity, stagnation cycles, exploration budget, element success entropy.
- **Action space**: (1) element group pair (9 groups: alkali, alkaline-earth, 3d/4d/5d TM, lanthanide, p-block metal, metalloid, nonmetal), (2) stoichiometry template (10 templates: binary-metal-rich, AB, AB3, 122, perovskite, balanced, quaternary, hydride-rich AH10, ternary-hydride ABH6, boride-carbide A2B3C), (3) crystal structure type (20 prototypes).
- **Policy**: softmax over learned weights with temperature annealing (1.0 → 0.3), epsilon-greedy exploration with decay (0.15 → 0.05), stagnation-boosted exploration.
- **Reward**: Tc improvement (normalized to 400K) × 2.0 + relative improvement × 5.0 + pipeline pass bonus (1.0) + stability × 0.5 + novelty × 0.3.
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
- **Wyckoff position sampling**: Seeds atoms at high-symmetry Wyckoff sites from 8 space groups (Pm-3m, Fm-3m, Im-3m, P6/mmm, P4/mmm, I4/mmm, R-3m, Pnma) covering cubic, hexagonal, tetragonal, trigonal, and orthorhombic systems.
- **Composition strategies**: 8 sampling strategies — hydride, ternary, binary TM, quaternary, exotic, cage compound, layered, borocarbide — targeting SC-relevant chemistries.
- **Prototype matcher**: Classifies generated structures against 10 known prototypes (Perovskite, A15, NaCl, AlB2, ThCr2Si2, Fluorite, Heusler, Layered, Kagome, Clathrate) using coordination number analysis and c/a ratio matching.
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