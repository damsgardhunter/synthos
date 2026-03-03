# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform designed for comprehensive materials science research. Its primary goal is to accelerate the discovery of a room-temperature superconductor by simulating a progressive learning engine. The platform integrates AI for natural language processing, novel material generation, and machine learning-driven predictions. It covers various aspects of materials science, from subatomic modeling to advanced computational physics and synthesis tracking, aiming to continuously learn and expand its knowledge base to achieve its ambitious research objective.

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
- **Command Center**: Dashboard with system statistics, learning pipeline status, engine controls, and live activity feed.
- **Atomic Explorer**: Element browser with Bohr model diagrams.
- **Materials Database**: Comprehensive indexed materials from multiple sources.
- **Novel Discovery**: AI-predicted new material candidates.
- **Superconductor Lab**: Details on superconductor candidates, including physics fields, synthesis, reactions, and ML scores.
- **Computational Physics**: Multi-fidelity screening pipeline, physics property cards, crystal structures, and negative results analysis.
- **Research Pipeline**: Tracks the full learning phase progress and activity log.
- **Learning Engine**: Orchestrates 12 distinct learning phases covering subatomic to multi-fidelity screening, with balanced priority and continuous cycling every 60 seconds.
- **ML Prediction Engine**: Hybrid XGBoost and OpenAI gpt-4o-mini neural network ensemble (40% XGBoost, 60% NN) for superconductor scoring, incorporating 18+ physics-informed features. Strict room-temperature criteria (Tc >= 293K, zero resistance, Meissner effect, ambient pressure).
- **SC Verification Pipeline**: A multi-step process for superconductor candidates: theoretical -> promising -> high-tc-candidate -> under-review -> requires-verification, emphasizing no premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline for candidate evaluation: ML filter, electronic structure, phonon/e-ph coupling, Tc prediction/competing phases, and synthesis feasibility/convex hull stability. Failures are logged with reasons.
- **Data Fetching & NLP**: Live fetching of materials data and OpenAI-powered NLP for analysis and property prediction.
- **Formula Generation**: AI-driven generation of novel chemical compositions.
- **Synthesis & Reaction Tracking**: Detailed tracking of material synthesis processes and a broad chemical reaction knowledge base.
- **Crystal Structure Prediction**: AI-powered prediction of crystal structures, including stability and synthesizability.

### Database Schema Highlights
- `elements`, `materials`, `learning_phases`, `novel_predictions`, `research_logs`.
- `synthesis_processes`: Detailed synthesis conditions.
- `chemical_reactions`: Balanced equations, thermodynamic data, SC relevance.
- `superconductor_candidates`: ML scores, physics fields, verification status.
- `crystal_structures`: Space groups, lattice parameters, stability, synthesizability.
- `computational_results`: Pipeline results, pass/fail, failure reasons.

## External Dependencies
- **OpenAI**: Used for gpt-4o-mini via Replit AI Integrations for NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: Relational database for persistent storage.
- **OQMD API**: `http://oqmd.org/oqmdapi/formationenergy` for live materials data fetching.
- **NIST WebBook**: Source for thermodynamic data and spectroscopic properties.
- **Materials Project**: Source for DFT-computed band gaps and formation energies.
- **Materials Science Knowledge Base**: OpenAI-sourced real materials data from peer-reviewed literature and reputable databases.