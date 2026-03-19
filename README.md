# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform accelerating the discovery of room-temperature superconductors (Tc >= 293K). It integrates AI for natural language processing, novel material generation, and machine learning predictions. The platform continuously learns, expands its knowledge base, and aims to revolutionize material discovery across high-tech industries.

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

### Startup Initialization Timeline
Heavy initializers are staggered to avoid event loop blocking:
- **+20s**: Pool init Phase 1 (15 entries, ~25s) — trains quick GBM model
- **+90s**: CrystalDiffusion DDPM (100 epochs, ~2s)
- **+110s**: Crystal-VAE (50+ epochs, ~2s)
- **+130s**: Benchmark reference predictions
- **+150s**: GNN 5-model ensemble warmup (~100s blocking, unavoidable)
- **+380s**: Phase 2 pool backfill (~498 entries, async with 50ms yields)
- **+400s**: Learning engine auto-start (xTB health ~60s execSync, backfill scores, recalculate physics)
- **+600s**: Lambda regressor (async with yields every 10 entries)
- **+660s**: Phonon surrogate
- **+720s**: Structure predictor ML

### Event Loop Protection
- `extractFeatures()` in ml-predictor.ts calls 3 expensive physics functions (~0.3s/call)
- `predictLambda()` returns heuristic fallback (lambda=0.5) until lambda ensemble trains at +600s
- `FEATURE_MEANS` Proxy returns static values until pool backfill computes them
- `getCrystalSymTargetEncoded()` uses `STATIC_CRYSTAL_SYM_ENCODING` without iterating training data
- `getTrainedModel()` returns `FALLBACK_MODEL` until pool trains a real model
- Phase 1 and Phase 2 yield `setTimeout(r, 50)` between each entry
- Lambda regressor `buildTrainingDataAsync()` yields every 10 entries with 20ms pauses
- First 150s: fully responsive API (<50ms). GNN blocks +150s to ~+270s (~75s). Responsive again until +380s.
- Engine start at +400s blocks ~120s (xTB health 60s + backfill/recalc).

### Core Architectural Decisions
- **Dynamic Module Loading**: `server/index.ts` initializes quickly, dynamically loading heavier modules post-startup to ensure rapid port availability.
- **AI-Driven Learning Engine**: Orchestrates 13 learning phases, including multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines gradient boosting (XGBoost) and Graph Neural Networks (GNN) with OpenAI gpt-4o-mini for superconductor scoring and feature proposals. Model architecture adapts based on dataset size.
- **Multi-Fidelity Quantum Engines**: Integrates GFN2-xTB for fast quantum-mechanical properties and Quantum ESPRESSO (QE 7.2) for high-fidelity DFT calculations (SCF, phonons, band structure, electron-phonon coupling).
- **Physics Engine**: Performs deterministic calculations, including Eliashberg theory, critical field estimations, and orbital-resolved DOS surrogates. Features uncertainty quantification and physics-guided ML feature proxies.
- **Topological Superconductor Detection**: Estimates Z2 invariant, Chern number, and Majorana feasibility using symmetry indicators, ML, and DFT band analysis.
- **Materials Generation**: Utilizes a 256-dimensional latent vector for similarity search and inverse design. Includes Crystal VAE, Diffusion Models, and Hybrid Structure Generators for novel crystal structures.
- **Reinforcement Learning Agent**: A REINFORCE policy gradient agent optimizes chemical space search, learning optimal elements, stoichiometries, and structures. Includes family-aware and structure-aware reward systems.
- **Bayesian Optimization**: Employs Gaussian Processes for chemical space search with mixed acquisition functions.
- **Synthesis Pathway Modeling**: Graph-based synthesis planner, heuristic generator, and ML predictor for multi-step reaction pathways, precursor selection, and feasibility assessment. Includes a "Synthesis-First Gate" for early rejection of impractical candidates.
- **Pressure as a First-Class Feature**: All models and calculations are pressure-aware, including pressure-dependent surrogates, phase transition detectors, and a Bayesian pressure optimizer.
- **Doping and Disorder Engines**: Generates chemically feasible doping and disorder variants, leveraging xTB relaxation and ML prediction with specialized features for lattice dynamics.
- **Heterostructure Generator**: Builds bilayer structures, predicting interface stability and enhanced superconductivity.
- **Active Learning Loop**: Drives iterative model improvement by selecting high-value candidates for DFT evaluation based on combined acquisition functions (EI, UCB, uncertainty, novelty).
- **Model Governance & Calibration**: Monitors ML model performance, detects out-of-distribution inputs, calibrates uncertainty via conformal prediction, and triggers retraining or emergency data fetching as needed.
- **LLM Meta-Learning Controller**: AI-driven system for diagnosing model performance, proposing experiments (hyperparameter tuning, dataset expansion), and executing improvements.
- **Comprehensive Data Filtering**: Implements rigorous physics-based filtering rules (phonon stability, hull distance, valence sum, element caps, min/max volume, etc.) to prune unphysical or unstable candidates early in the pipeline.

### Core Features Specifications
- **ML Governance**: Architecture selection heuristic favors XGBoost for <100 samples, 70/30 XGBoost/GNN for 200-500, GNN-primary for >1000 samples. Architecture reassessment triggers at dataset bucket thresholds [128, 256, 512, 1024, 2048, 4096].
- **DFT Backend (Quantum ESPRESSO)**: Handles high-pressure calculations with Murnaghan EOS, pressure-dependent geometry validation, and parallel pseudopotential download. Dynamic cutoffs and k-point meshing. Includes vc-relax pre-processing. UPF z_valence parser handles scientific notation (e.g., `1.300000000000e1` = 13).
- **GNN Surrogate**: 5-model deep ensemble with heteroscedastic heads, MC dropout (10 passes), and uncertainty decomposition (epistemic + aleatoric). Node embeddings are symmetry-augmented.
- **Eliashberg Electron-Phonon Pipeline**: Computes full alpha2F(omega) spectral function, lambda, omega_log, Allen-Dynes Tc, and isotope effect. Supports both surrogate and DFPT tiers. Includes atom-projected mode-resolved lambda analysis.
- **Orbital-Resolved DOS Surrogate**: Predicts 64-bin eDOS across 4 orbital channels (s, p, d, f) with Van Hove Singularity (VHS) detection.
- **Tight-Binding Hamiltonian Engine**: Graph-based tight-binding with Harrison's Slater-Koster parameters, band structure, DOS, and Fermi properties. Includes Householder tridiagonalization and eigenvector-based orbital character projection.
- **Adaptive Volume DNN**: 3-layer MLP replacing fixed radius-based volume scaling, self-trained from elemental and compound data.
- **Reaction Network Graph**: Dijkstra shortest-path algorithm for multi-step synthesis routes, integrated with thermodynamic costs and precursor availability.
- **Structural Motif Reward System**: Tracks performance of 23 structural motifs, biasing GA mutation probabilities.
- **XGBoost Heteroscedastic Ensemble Uncertainty**: 5 bootstrap-trained models with diversified hyperparameters, predicting both mean and variance.
- **Conformal Prediction Calibrator**: Calibrates raw model uncertainty, tracks Expected Calibration Error (ECE), and automatically recalibrates.
- **OOD Detector**: Combines Mahalanobis distance, GMM density estimation, and GNN latent cosine distance for out-of-distribution detection.
- **Chemistry Filters**: Enforces average valence electron count, electronegativity spread, and oxidation state balance.
- **Tc Prediction**: Physics-only Tc pipeline using Allen-Dynes equation across all generation paths, replacing LLM-computed Tc. Includes adaptive Tc caps and strong-coupling corrections.

## External Dependencies
- **OpenAI**: gpt-4o-mini for NLP, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: Primary persistent data storage.
- **OQMD API**: Live materials data fetching.
- **NIST WebBook**: Thermodynamic and spectroscopic data.
- **Materials Project**: DFT-computed band gaps and formation energies.
- **AFLOW REST API**: Crystal structure and electronic property cross-validation.
- **xtb**: Quantum-mechanical properties, geometry optimization, phonon calculations.
- **Quantum ESPRESSO**: Full DFT calculations (pw.x, ph.x, bands.x, lambda.x).