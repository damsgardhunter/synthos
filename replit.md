# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform designed to accelerate the discovery of room-temperature superconductors (Tc >= 293K). It integrates advanced AI for natural language processing, novel material generation, and machine learning predictions, aiming to revolutionize material discovery from subatomic modeling to synthesis tracking. The platform continuously learns and expands its knowledge base, offering significant market potential across high-tech industries.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases covering multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines gradient boosting and OpenAI gpt-4o-mini for superconductor scoring, focusing on Tc >= 293K.
- **Physics Engine**: Performs deterministic calculations using a 96-element database, including advanced parameter calculations and specialized adjustments for superhydrides, HEAs, and unconventional superconductors.
- **Topological Superconductor Detection**: Estimates Z2 invariant, Chern number, and Majorana feasibility.
- **Quantum Pairing Mechanism Simulator**: Analyzes 7 pairing channels (phonon, spin-fluctuation, orbital-fluctuation, excitonic, CDW, polaronic, plasmon).
- **Fermi Surface Reconstruction Engine**: Computes and analyzes Fermi surface features.
- **Hydrogen Network Topology Analyzer**: Analyzes H-H distances and network topology for hydrides.
- **Chemical Stability Reaction Network**: Builds compound decomposition graphs with kinetic and metastable lifetime estimation.
- **Materials Genome Representation**: 256-dimensional latent vector for similarity search and inverse design.
- **Autonomous Theory Discovery Engine**: Uses symbolic regression to uncover mathematical relationships from simulation data.
- **Multi-Scale Physics Modeling**: Analyzes atomic, electronic, and mesoscopic feature layers and their cross-scale coupling.
- **Self-Improving Physics Models**: Parameterized physics equations optimized via Bayesian methods with continuous learning.
- **Multi-Material Interface Discovery Engine**: Analyzes heterostructure superconductivity phenomena.
- **Generator Resource Manager**: Centralized allocation and adaptive rebalancing for 5 candidate generators (structure_diffusion, rl, bo_exploration, motif_diffusion, random_exploration).
- **Fermi Surface Topology Clustering**: Clusters FS feature vectors into 5 archetypes for search guidance.
- **Constraint-Based Physics Solver**: Multiple solvers, including backward McMillan/Eliashberg, for inverse design.
- **Pressure-to-Ambient Pathway Search**: Generates stabilization strategies for high-pressure superconductors.
- **Physics Constraint Graph Solver**: Solves all SC physics constraints simultaneously.
- **Synthesis Pathway Modeling & Optimizer**: Simulates multi-step reaction pathways and optimizes synthesis conditions across 6 methods.
- **Synthesis Simulator & Active Optimization**: Co-searches material composition and 9 active synthesis variables.
- **Defect Physics Engine**: Models defects, formation energy, and electronic structure adjustments.
- **Strong Correlation Physics Engine**: Detects correlated regimes and adjusts pairing weights.
- **Experimental Validation Planner**: Ranks candidates, generates synthesis instructions, and suggests characterization methods.
- **Band Structure Neural Operator**: Predicts complete band structure, effective masses, and topological indices.
- **Autonomous Hypothesis Engine**: AI for generating, testing, and ranking superconductivity theories.
- **Quantum Criticality Detector**: Unified QCP detector formalizing various detections into a `QuantumCriticalScore`.
- **Discovery Memory System**: Pattern-based learning that remembers physics patterns leading to high Tc discoveries.
- **Phase Stability Prediction Network**: GNN-based pre-filter for rapid stability screening.
- **Convex Hull Phase Diagram Engine**: Computes energy above hull and assesses metastability.
- **Pressure Modeling Engine**: Calculates volume compression, bulk modulus, and predicts high-pressure stability/pressure-Tc curves.
- **Graph Neural Network Surrogate**: Primary ML predictor for formation energy, phonon stability, Tc, confidence, and electron-phonon lambda, with uncertainty estimation.
- **Reinforcement Learning Chemical Space Agent**: REINFORCE policy gradient agent learning optimal elements, stoichiometries, and structures with physics-aware rewards.
- **Bayesian Optimization for Chemical Space Search**: Gaussian Process surrogate with mixed acquisition functions.
- **Crystal Diffusion Generator**: Generates novel crystal structures by refining random atomic positions.
- **Massive Candidate Generator**: Utilizes element substitution, composition interpolation, doped variants, and composition sweeps with a rapid Gradient Boosting screen.
- **Physics-Aware ML Predictor**: Multi-target prediction with transfer priors and uncertainty estimation.
- **Structural Mutation Engine**: Assigns prototypes, generates distorted lattices, layered structures, and strain variants.
- **Multi-Dimensional Phase Explorer**: Scans composition, pressure, and temperature spaces with adaptive sampling.
- **Crystal Prototype Structure Generator**: Generates 10 distinct crystal structure types, producing over 807 structurally-typed candidates.
- **Active Learning Loop**: Uncertainty-driven DFT selection and GNN surrogate retraining.
- **Real DFT Backend (GFN2-xTB)**: Integrates xtb for quantum-mechanical properties, geometry optimization, and finite-displacement phonon calculations with bulk-corrected formation energies.
- **Full DFT Backend (Quantum ESPRESSO 7.2)**: pw.x and ph.x calculations for top candidates via async job queue, with multi-layer reliability checks and extensive pre-filters (pseudopotential, xTB pre-relaxation, geometry, formula, structure dedup, k-points, SCF retry). Supports 29 PSlibrary/SG15 UPF pseudopotentials.
- **Async DFT Job Queue**: PostgreSQL-backed job queue with priority scheduling and status tracking.
- **Inverse Design Optimizer Engine**: 5-layer architecture for goal-driven candidate generation and closed-loop optimization.
- **Next-Generation Inverse Design Pipeline**: Unified 5-component orchestrator for design generation and learning, with convergence tracking and real-time stats.
- **Self-Improving Design Lab**: Strategy-level optimization that evolves design architectures across 8 concurrent strategy types. Includes Failure Analysis, Knowledge Base, and Strategy Evolution.
- **Design Representations System**: Dual representation for material design (Programmatic/Procedural and Graph-based) with mutation/crossover operators and bidirectional conversion.
- **Symbolic Physics Discovery Layer**: 12-component subsystem for automated physics equation discovery, using multi-objective evolutionary search, dimensional analysis, and physics constraint validation.
- **Causal Physics Discovery Layer**: 11-component subsystem for causal mechanism discovery in superconductivity, identifying directed causal graphs and enabling intervention/counterfactual simulations.
- **Crystal Distribution Database**: Learned crystallographic distributions from ~500k structures used by diffusion generators.
- **Distribution-Based Crystal Diffusion**: Generates crystal structures by sampling from learned crystallographic distributions, with adaptive system weights and Wyckoff denoising.
- **Multi-Task GNN Surrogate**: Extended GNN predicting 18 properties simultaneously, including formation energy, phonon stability, and Tc.
- **Crystal VAE Latent Space**: Variational autoencoder wrapping the 256D MaterialGenome for inverse design in a 64D latent space.
- **Theory-Guided Generator Bias**: Integrates theory/causal discoveries to bias generator weights, chemical family preferences, and structural guidance.
- **Kinetic Metastability Prediction Model**: Predicts metastable material lifetimes based on multi-factor analysis, gates candidates, and identifies stabilization strategies.
- **Differentiable Materials Design**: Computes analytical McMillan gradients and numerical element gradients for Tc.
- **8-Pillar SC Optimizer**: Targets 8 superconductivity pillars with adaptive weights and weakness-targeted mutation.
- **Physics-Constrained Generative AI**: Enforces charge neutrality, atomic radius compatibility, coordination number, and bond stability.
- **Structure-First Design (Primary Generator)**: Motif-first architecture with 29 superconducting structural motifs and 8 chemical families.
- **RL Family-First Action Selection**: RL incorporates chemicalFamily dimension, biasing element groups and structure types.
- **Chemical Synthesis Realism**: Scores precursor availability, family-specific synthesis defaults, and phase competition penalties.
- **Autonomous Discovery Loop**: Massive generation pipeline with multi-stage filtering and a tiered acceptance system, providing closed-loop feedback from all analysis subsystems.
- **Semantic Insight Deduplication**: Uses OpenAI text-embedding-3-small for semantic deduplication.
- **NLP Engine**: Generates cross-property correlation insights and superconductor correlation analysis.
- **Milestone Event Tracker**: Detects and persists research milestones to DB.

### Bug Fixes Applied (Session 2026-03-08)
- **RL Weight Clamping**: All RL policy weights clamped to >= 0 after updates; entropy regularization prevents exploration collapse (top weight > 3x mean gets 0.9 decay).
- **avgTc Null Guard**: All avgTc calculations guarded with `Number.isFinite()`; `sanitizeStatsNumeric()` recursively cleans stats before API response.
- **Valence Filter Enforcement**: `passesValenceFilter` applied in engine pipeline, crystal diffusion, and structure diffusion to reject impossible compounds (FO2, FN3O2, etc.).
- **Duplicate Detection**: Formulas normalized before dedup checks; shared screened set passed to all generators; intra-batch dedup added.
- **Throughput Metric**: Reports 0 for first 5 minutes; minimum divisor increased to 0.1 hours.
- **Surrogate Tc Cap**: Differentiable optimizer and gradient boost predictions capped at 200K (was 400K).
- **DFT Metric Clarity**: Stats keys renamed: `realDFT` → `xtb`, `fullDFTQueue` → `qeDFT` for clear xTB vs QE distinction.
- **Motif Entropy Regularization**: Epsilon-greedy (15%) + min 5% floor + max 40% cap per motif in structure/crystal diffusion.
- **Log Truncation Fix**: Live event feed and log details use `break-words line-clamp-3` instead of CSS truncate.
- **Volume Prior in Structure Generation**: Per-element atomic volume table; `computeExpectedVolume` + `validateVolumeRatio` (0.6–1.6 range); `estimateLatticeParam` uses atomic volumes instead of covalent radii cubed; `MIN_VOLUME_PER_ATOM` raised to 10.0 Å³ (non-hydride) and 5.0 Å³ (hydride).
- **Hydrogen Cage Library**: 5 cage motifs (Sodalite-H32, Clathrate-H6, TriCappedPrism-H9, Clathrate-H8, H4-Tetrahedral) for hydrides with H/metal ≥ 4. `generateHydrideCageStructure` places metal at cage center, H at cage vertices.
- **Prototype-First Generation**: Added Ruddlesden-Popper, Double-perovskite, Garnet prototypes. Chemistry-aware element classification + `selectBestPrototypeByChemistry` for fuzzy matching. `prototypeAttempts`/`prototypeSuccesses` counters tracked.

### Physics Filtering Rules
- **Phonon stability**: Hard rejection for severe instability (>3 imaginary modes or lowest freq < -500 cm⁻¹). Penalties for physical imaginary modes.
- **Formation energy**: Hard stop for Ef > 1.0 eV/atom or Ef < -5.0 eV/atom.
- **Hull distance**: Hard reject > 0.50 eV/atom; exploratory for 0.25-0.50 eV/atom (requires kinetic barrier).
- **Family quotas**: Apply caps to prevent over-representation of specific material families.
- **Known compound filter**: Rejects ~200 known compounds.
- **Interatomic distance validation**: Element-pair-aware minimum distances.
- **Hydrogen stoichiometry limits**: Tags high-pressure candidates for H/metal > 6 or H fraction > 75%.
- **Chemistry grammar validation**: Pre-filters compositions based on max elements, atoms, noble gases, metal presence, H fraction, anion/cation ratios, and charge balance.
- **Surrogate pre-filter pipeline**: Screens candidates with GB model before expensive physics evaluation, rejecting insulators, low Tc, or low GB scores.
- **Discovery score weights**: Combines Tc, stability, novelty, synthesis, topology, and uncertainty.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.