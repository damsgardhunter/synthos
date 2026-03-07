# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform accelerating the discovery of room-temperature superconductors. It integrates AI for natural language processing, novel material generation, and machine learning predictions. The platform aims to revolutionize material discovery by continuously learning and expanding its knowledge base, from subatomic modeling to advanced computational physics and synthesis tracking, with significant market potential across high-tech industries. Its core mission is to find materials with a critical temperature (Tc) >= 293K.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases covering subatomic to multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines a gradient boosting model with an OpenAI gpt-4o-mini neural network ensemble for superconductor scoring, applying strict room-temperature criteria (Tc >= 293K).
- **Physics Engine**: Deterministic calculations using a 96-element database, including advanced parameter calculation (DOS, bandwidth, Hubbard U/W, lambda, phonon frequencies, metallicity). Includes specialized adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductors.
- **Topological Superconductor Detection Engine**: Estimates Z2 invariant, Chern number, and Majorana feasibility.
- **Quantum Pairing Mechanism Simulator (7-Channel)**: Analyzes phonon, spin-fluctuation, orbital-fluctuation, excitonic, CDW, polaronic, and plasmon pairing channels.
- **Fermi Surface Reconstruction Engine**: Computes Fermi surface isosurfaces, classifies pockets, and extracts features.
- **Hydrogen Network Topology Analyzer**: Analyzes H-H distance distribution, network dimensionality, and cage topology for hydrides.
- **Chemical Stability Reaction Network**: Builds compound decomposition pathway graphs with kinetic barriers and metastable lifetime estimation.
- **Materials Genome Representation (256-dim)**: Latent genome vector encoding for similarity search and inverse design.
- **Autonomous Theory Discovery Engine**: Uses symbolic regression to discover mathematical relationships from simulation data.
- **Multi-Scale Physics Modeling**: Analyzes atomic, electronic, and mesoscopic feature layers and their cross-scale coupling.
- **Self-Improving Physics Models**: Parameterized physics equations optimized via Bayesian methods with continuous learning loops.
- **Multi-Material Interface Discovery Engine**: Analyzes heterostructure superconductivity, including charge transfer, interface phonon enhancement, epitaxial strain, and dimensional confinement for 2D superconductors.
- **Generator Resource Manager**: Centralized allocation and adaptive rebalancing for 7 candidate generators.
- **Fermi Surface Topology Clustering**: Clusters FS feature vectors into 5 archetype clusters for search guidance.
- **Materials Discovery Landscape**: UMAP embedding engine for 256D genome vectors, 3D latent space, and discovery zone detection.
- **Constraint-Based Physics Solver**: Backward McMillan/Eliashberg solver and 4 additional constraint solvers for inverse design.
- **Pressure-to-Ambient Pathway Search**: Generates stabilization strategies for high-pressure superconductors to achieve ambient pressure stability.
- **Physics Constraint Graph Solver**: Solves all SC physics constraints simultaneously as a graph.
- **Synthesis Pathway Modeling**: Simulates multi-step reaction pathways, including thermodynamics and kinetic barriers for 6 synthesis methods.
- **Synthesis Condition Optimizer**: Selects optimal synthesis conditions, estimates complexity/duration, and tracks feasibility.
- **Synthesis Simulator & Active Optimization**: Co-searches material composition and 9 active synthesis variables, simulating synthesis effects on material properties.
- **Defect Physics Engine**: Models various defects with formation energy, defect density, and electronic structure adjustment.
- **Strong Correlation Physics Engine**: Detects correlated regimes, computes correlation scores, and adjusts pairing weights.
- **Experimental Validation Planner**: Ranks candidates for lab experiments, generates synthesis instructions, and suggests characterization methods.
- **Band Structure Neural Operator**: Predicts complete band structure along high-symmetry k-paths, deriving effective masses, Fermi velocities, and topological indices.
- **Autonomous Hypothesis Engine**: AI for generating, testing, and ranking superconductivity theories.
- **Quantum Criticality Detector**: Unified QCP detector formalizing various detections into a `QuantumCriticalScore`.
- **Discovery Memory System**: Pattern-based learning system that remembers physics patterns leading to high Tc discoveries.
- **Phase Stability Prediction Network**: GNN-based pre-filter for rapid screening of material stability.
- **Convex Hull Phase Diagram Engine**: Computes energy above hull, decomposition products, and assesses metastability.
- **Pressure Modeling Engine**: Calculates volume compression, bulk modulus, and predicts high-pressure stability and pressure-Tc curves.
- **Graph Neural Network Surrogate**: Primary ML predictor for formation energy, phonon stability, Tc, confidence, and electron-phonon lambda, with uncertainty estimation.
- **Reinforcement Learning Chemical Space Agent**: REINFORCE policy gradient agent learning optimal elements, stoichiometries, and structures with physics-aware rewards.
- **Bayesian Optimization for Chemical Space Search**: Gaussian Process surrogate with mixed acquisition functions.
- **Crystal Diffusion Generator**: Generates novel crystal structures by refining random atomic positions.
- **Massive Candidate Generator**: Utilizes element substitution, composition interpolation, doped variants, and composition sweeps with a rapid Gradient Boosting screen.
- **Physics-Aware ML Predictor**: Multi-target prediction with transfer priors and uncertainty estimation.
- **Structural Mutation Engine**: Assigns structure prototypes, generates distorted lattices, layered structures, vacancy structures, and strain variants.
- **Multi-Dimensional Phase Explorer**: Scans composition, pressure, and temperature spaces with adaptive sampling.
- **Crystal Prototype Structure Generator**: Generates 10 distinct crystal structure types, producing over 807 structurally-typed candidates.
- **Active Learning Loop**: Uncertainty-driven DFT selection with analytical DFT fallback, retraining GNN surrogates.
- **Real DFT Backend (GFN2-xTB)**: Integrates xtb v6.7.1 for quantum-mechanical properties, geometry optimization, and finite-displacement phonon calculations.
- **Full DFT Backend (Quantum ESPRESSO 7.2)**: pw.x and ph.x calculations for top 0.1% candidates via async job queue.
- **Async DFT Job Queue**: PostgreSQL-backed job queue with priority scheduling, status tracking, and WebSocket broadcast.
- **Inverse Design Optimizer Engine**: 5-layer architecture for goal-driven candidate generation, inverse learning, and closed-loop optimization.
- **Next-Generation Inverse Design Pipeline**: Unified 5-component orchestrator (Goal Spec → Design Generator → Constraint Solver → Surrogate Model → Learning Loop) with convergence tracking, gradient refinement after iteration 3, and real-time stats. API at `/api/next-gen-pipeline/`. Frontend tab in Computational Physics.
- **Differentiable Materials Design**: Computes analytical McMillan gradients and numerical element gradients for Tc.
- **8-Pillar SC Optimizer**: Targets 8 superconductivity pillars with adaptive weights and weakness-targeted mutation strategies.
- **Physics-Constrained Generative AI**: Enforces charge neutrality, atomic radius compatibility, coordination number limits, and bond stability.
- **Structure-First Design**: Selects SC structural motifs, fills site roles with element candidates, and evaluates using GB Tc prediction.
- **Chemical Synthesis Realism**: Scores precursor availability, family-specific synthesis defaults, reaction temperature factors, and phase competition penalties.
- **Autonomous Discovery Loop**: Massive generation pipeline (500-2000 candidates/cycle) with multi-stage filtering and a tiered acceptance system.
- **Semantic Insight Deduplication**: Uses OpenAI text-embedding-3-small for semantic deduplication of insights.
- **NLP Engine**: Generates cross-property correlation insights and superconductor correlation analysis.
- **Milestone Event Tracker**: Detects and persists research milestones to DB, displayed on Research Pipeline page.

### Navigation Order (Sidebar)
1. Command Center (bold, text-base)
2. Research Pipeline (bold, text-[15px])
3. Computational Physics
4. Materials Database
5. Novel Discovery
6. Superconductor Lab
7. Atomic Explorer

### Key Data Paths
- `/api/engine/memory` returns `lastCycleCandidates`, `lastCycleFamilyCounts`
- `autonomousLoopStats.inverseOptimizer.bestTcAcrossAll` — inverse optimizer best Tc, displayed in Active Learning card and Knowledge Map
- `/api/milestones` — reads milestones from DB

### Physics Filtering Rules
- **Phonon artifact threshold**: Imaginary modes with freq < -2000 cm⁻¹ are xTB numerical artifacts — discarded entirely, not penalized
- **Physical instability threshold**: -100 cm⁻¹
- **Penalty**: Physical imaginary modes penalize ensemble score by 0.05 per mode (max 0.25); >= 5 modes → dataConfidence = "low"
- **Discovery score weights**: 0.55 Tc + 0.15 stability + 0.10 novelty + 0.10 synthesis + 0.05 topology + 0.05 uncertainty

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.