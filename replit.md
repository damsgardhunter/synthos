# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform dedicated to accelerating the discovery of room-temperature superconductors (Tc >= 293K). It integrates advanced AI for natural language processing, novel material generation, and machine learning predictions to revolutionize material discovery. The platform aims to continuously learn, expand its knowledge base, and offers significant market potential across high-tech industries.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases, including multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines gradient boosting and OpenAI gpt-4o-mini for superconductor scoring.
- **Physics Engine**: Performs deterministic calculations using a 96-element database, including advanced parameter calculations.
- **Topological Superconductor Detection**: Estimates Z2 invariant, Chern number, and Majorana feasibility.
- **Quantum Pairing Mechanism Simulator**: Analyzes 7 pairing channels.
- **Materials Genome Representation**: Utilizes a 256-dimensional latent vector for similarity search and inverse design.
- **Autonomous Theory Discovery Engine**: Uses symbolic regression to uncover mathematical relationships.
- **Self-Improving Physics Models**: Parameterized physics equations optimized via Bayesian methods.
- **Generator Resource Manager**: Centralized allocation and adaptive rebalancing for 5 candidate generators.
- **Synthesis Pathway Modeling & Optimizer**: Simulates multi-step reaction pathways and optimizes synthesis conditions.
- **Experimental Validation Planner**: Ranks candidates, generates synthesis instructions, and suggests characterization methods.
- **Autonomous Hypothesis Engine**: AI for generating, testing, and ranking superconductivity theories.
- **Discovery Memory System**: Pattern-based learning for high Tc discoveries.
- **Graph Neural Network Surrogate**: Primary ML predictor for formation energy, phonon stability, Tc, and electron-phonon lambda with uncertainty estimation.
- **Reinforcement Learning Chemical Space Agent**: REINFORCE policy gradient agent learning optimal elements, stoichiometries, and structures.
- **Bayesian Optimization for Chemical Space Search**: Gaussian Process surrogate with mixed acquisition functions.
- **Crystal Diffusion Generator**: Generates novel crystal structures by refining random atomic positions.
- **Active Learning Loop**: Uncertainty-driven DFT selection and GNN surrogate retraining.
- **Real DFT Backend (GFN2-xTB)**: Integrates xtb for quantum-mechanical properties, geometry optimization, and phonon calculations.
- **Full DFT Backend (Quantum ESPRESSO 7.2)**: Executes pw.x and ph.x calculations for top candidates via async job queue, with multi-layer reliability checks and pre-filters.
- **Inverse Design Optimizer Engine**: 5-layer architecture for goal-driven candidate generation and closed-loop optimization.
- **Self-Improving Design Lab**: Strategy-level optimization that evolves design architectures across 8 concurrent strategy types.
- **Symbolic Physics Discovery Layer**: 12-component subsystem for automated physics equation discovery.
- **Causal Physics Discovery Layer**: 11-component subsystem for causal mechanism discovery in superconductivity.
- **Crystal Distribution Database**: Learned crystallographic distributions from ~500k structures.
- **Multi-Task GNN Surrogate**: Extended GNN predicting 18 properties simultaneously.
- **Autonomous Discovery Loop**: Massive generation pipeline with multi-stage filtering and tiered acceptance.
- **Semantic Insight Deduplication**: Uses OpenAI text-embedding-3-small for semantic deduplication.

### Physics Filtering Rules
- **Phonon stability**: Hard rejection for severe instability.
- **Formation energy**: Hard stop for Ef > 1.0 eV/atom or Ef < -5.0 eV/atom.
- **Hull distance**: Hard reject > 0.50 eV/atom.
- **Chemistry grammar validation**: Pre-filters compositions based on various chemical constraints.
- **Surrogate pre-filter pipeline**: Screens candidates with GB model before expensive physics evaluation.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.