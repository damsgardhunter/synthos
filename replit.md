# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform dedicated to materials science research, focused on accelerating the discovery of room-temperature superconductors. It integrates AI for natural language processing, novel material generation, and machine learning predictions, covering aspects from subatomic modeling to advanced computational physics and synthesis tracking. The platform continuously learns and expands its knowledge base to achieve its research objectives, including a business vision to revolutionize material discovery and a market potential across various high-tech industries.

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
- **Learning Engine**: Orchestrates 12 distinct learning phases covering subatomic to multi-fidelity screening, with balanced priority and continuous cycling.
- **ML Prediction Engine**: Hybrid XGBoost and OpenAI gpt-4o-mini neural network ensemble for superconductor scoring, incorporating physics-informed features and strict room-temperature criteria (Tc >= 293K).
- **SC Verification Pipeline**: A multi-step process for superconductor candidates: theoretical -> promising -> high-tc-candidate -> under-review -> requires-verification, avoiding premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline for candidate evaluation including ML filtering, electronic structure, phonon/e-ph coupling, Tc prediction/competing phases, and synthesis feasibility/convex hull stability.
- **Novel Insight Detection**: Pre-filters known knowledge, deduplicates, and uses OpenAI to evaluate remaining insights for novelty.
- **Diversity & Deduplication**: Mechanisms across formula generation and SC candidate processing to ensure diversity and prevent redundant work.
- **Self-Improving SC Model**: The ML predictor loads verified physics data and crystal structures to enrich XGBoost feature extraction and train the neural network.
- **Evolving Research Strategy**: `analyzeAndEvolveStrategy()` classifies top SC candidates, and OpenAI generates ranked focus areas to bias the formula generator towards priority material families.
- **Milestone Detection System**: Detects various milestones such as new-family, tc-record, pipeline-graduate, diversity-threshold, knowledge-milestone, and insight-cascade.
- **Progressive Scoring**: XGBoost uses sigmoid scoring, knowledge depth bonuses, and Lambda-aware Tc scaling.
- **Physics as Tc Authority**: Implements strict physics-based capping and penalties for LLM-generated Tc values, especially for Mott insulators and strongly correlated materials, using frameworks like the McMillan formula and Eliashberg theory.
- **Hydrogen-Ratio-Aware Coupling**: Adjusts electron-phonon coupling and phonon spectrum calculations based on the hydrogen-to-metal ratio in hydrides.
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters like Hc2, pressure for ambient stability, and strict criteria for `roomTempViable` candidates. It also corrects unphysical values via bulk SQL operations at startup.
- **Learning Feedback Loop (Re-evaluation)**: `reEvaluateTopCandidates()` provides stage-based Tc boosts and dynamically adjusts the Tc ceiling based on accumulated evidence.
- **Stagnation Breaking**: When `cyclesSinceTcImproved` indicates stagnation, the system triggers re-analysis of high-lambda candidates and adapts candidate generation strategies via LLM prompts.
- **SC Candidate Deduplication**: Ensures unique superconductor candidates through database constraints and startup cleanup.
- **UI/UX Decisions**: Real-time updates via WebSocket, NaN safety, and a clear status taxonomy for novel predictions.

## External Dependencies
- **OpenAI**: For gpt-4o-mini via Replit AI Integrations, used in NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **Materials Science Knowledge Base**: OpenAI-sourced real materials data from peer-reviewed literature.