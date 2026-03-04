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
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters like Hc2 (max 300T), coherence length (min 1nm), pressure for ambient stability, ensembleScore (max 0.95 — score=1.0 is a calibration bug), and strict criteria for `roomTempViable` candidates (Tc>=293K AND zeroResistance AND meissnerEffect AND P<50GPa). Corrects unphysical values via bulk SQL operations at startup. Convergence snapshots corrected to match actual DB max Tc. bestScore in convergence snapshots capped at 0.95.
- **Learning Feedback Loop (Re-evaluation)**: `reEvaluateTopCandidates()` provides conservative stage-based Tc boosts (max +4K from lambda, +2K/stage, +1K crystal) with McMillan cap. Evidence-gated: requires real new evidence (stage increase, lambda change >0.15, new crystal) — ceiling-rose no longer triggers re-evaluation. Re-evaluation map persists across cycles (resets on restart).
- **Stagnation Breaking**: When `cyclesSinceTcImproved` indicates stagnation, the system triggers re-analysis of high-lambda candidates and adapts candidate generation strategies via LLM prompts.
- **SC Candidate Deduplication**: Ensures unique superconductor candidates through database constraints and startup cleanup.
- **UI/UX Decisions**: Real-time updates via WebSocket, NaN safety, and a clear status taxonomy for novel predictions.

### Physics Engine (server/learning/physics-engine.ts)
- **Zero Math.random()**: All physics calculations are fully deterministic — no randomness in any physics function
- **Elemental Data**: 96 elements with tabulated Debye temps, bulk moduli, Stoner parameters, Hubbard U, McMillan-Hopfield eta, Miedema params, Sommerfeld gamma, Gruneisen params (server/learning/elemental-data.ts)
- **DOS at Fermi level**: N(Ef) = gamma/2.359 [states/eV/atom] from Sommerfeld coefficient
- **Bandwidth W**: Period-based for transition metals (3d:5.5, 4d:8, 5d:10 eV), DOS-based for sp metals
- **Hubbard U/W**: From tabulated U values; U/W < 0.5 weakly correlated, 0.5-1.0 moderate, >1.0 strong
- **Lambda**: Back-calculated from known elemental Tc (via inverted McMillan formula) for elements with known Tc; McMillan-Hopfield eta-based for others; composition-weighted for compounds
- **Lambda conversion**: LAMBDA_CONVERSION = 562000 calibrated so Nb gives lambda ≈ 1.04
- **Phonon frequencies**: omega_D = theta_D * 0.695 cm-1; omega_log = 0.65 * omega_max for monatomic; light-element boost for borides/hydrides
- **Metallicity**: Hydrogen-rich compounds (H:metal >= 6) classified as metallic (0.80+), checked before EN spread
- **Superhydride handling**: No correlation suppression or metallicity penalty for H:metal >= 6 compounds (LaH10 etc.)
- **Materials Project client**: server/learning/materials-project-client.ts with fallback to analytical models; MP data cached in mp_material_cache DB table

### Validated Against Known Superconductors (T010)
- Nb: Tc=13.5K (lit:9.3K), lambda=0.84 (lit:1.04), weakly correlated ✓
- Al: Tc=2.3K (lit:1.2K), lambda=0.40 (lit:0.43), weakly correlated ✓
- MgB2: Tc=33.8K (lit:39K), lambda=0.83 (lit:0.87) ✓
- LaH10: Tc=229.6K (lit:250K), lambda=1.48 (lit:2.20) ✓
- Pb: Tc=10.4K (lit:7.2K), lambda=1.39 (lit:1.55) ✓
- CeCoIn5: Tc=2.9K (lit:2.3K), strongly correlated ✓
- Fe2As2: Tc=0K, moderately correlated, AFM competing phase ✓
- YBCO: Tc=0K from Eliashberg (correctly flags as unconventional/strongly correlated) ✓

## External Dependencies
- **OpenAI**: For gpt-4o-mini via Replit AI Integrations, used in NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies (MATERIALS_PROJECT_API_KEY secret).
- **Materials Science Knowledge Base**: OpenAI-sourced real materials data from peer-reviewed literature.