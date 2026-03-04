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
- **Physics as Tc Authority**: All Tc values grounded in Eliashberg/McMillan physics. LLM candidates capped at insertion by lambda-based McMillan bounds (lambda<0.3: max 50K, lambda<0.5: max 80K, lambda<1.0: max 150K, lambda<1.5: max 250K, lambda<2.5: max 350K, lambda>=2.5: max 350K). Multi-fidelity pipeline writes physics Tc back to candidates at Stage 3 with conservative upward caps (+10-30K). Phase 10 physics flowback blends downward (0.7-0.8 weight) when Eliashberg < current. Startup bulk correction uses `getSuperconductorCandidatesByTc(100)` to catch ALL high-Tc candidates regardless of ensemble score, recomputes Eliashberg Tc, and blends down with 0.8 weight. Hard cap: 350K absolute max.
- **Hydrogen-Ratio-Aware Coupling**: Adjusts electron-phonon coupling and phonon spectrum calculations based on the hydrogen-to-metal ratio in hydrides.
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters like Hc2 (max 300T), coherence length (min 1nm), pressure for ambient stability, and strict criteria for `roomTempViable` candidates (Tc>=293K AND zeroResistance AND meissnerEffect AND P<50GPa). Corrects unphysical values via bulk SQL operations at startup. Convergence snapshots corrected to match actual DB max Tc.
- **Learning Feedback Loop (Re-evaluation)**: `reEvaluateTopCandidates()` provides conservative stage-based Tc boosts (max +4K from lambda, +2K/stage, +1K crystal) with McMillan cap. Evidence-gated: requires real new evidence (stage increase, lambda change >0.15, new crystal) — ceiling-rose no longer triggers re-evaluation. Re-evaluation map persists across cycles (resets on restart).
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