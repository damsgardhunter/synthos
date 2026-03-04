# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform dedicated to materials science research, specifically aiming to accelerate the discovery of a room-temperature superconductor. It achieves this by integrating AI for natural language processing, novel material generation, and machine learning predictions across various aspects of materials science, from subatomic modeling to advanced computational physics and synthesis tracking. The platform is designed to continuously learn and expand its knowledge base to achieve its ambitious research objective.

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
- **SC Verification Pipeline**: A multi-step process for superconductor candidates: theoretical -> promising -> high-tc-candidate -> under-review -> requires-verification, emphasizing no premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline for candidate evaluation including ML filtering, electronic structure, phonon/e-ph coupling, Tc prediction/competing phases, and synthesis feasibility/convex hull stability.
- **Novel Insight Detection**: Integrated into learning phases, pre-filters known knowledge, deduplicates, and uses OpenAI to evaluate remaining insights for novelty.
- **Diversity & Deduplication**: Mechanisms across formula generation and SC candidate processing to ensure diversity and prevent redundant work, including exclusion lists in LLM prompts and exploration bonuses for under-explored material families.
- **Self-Improving SC Model**: ML predictor loads verified physics data and crystal structures from the database, enriching XGBoost feature extraction with physics context and penalties, and treating verified data as ground truth for the neural network prompt.

### Evolving Research Strategy
- After each engine cycle, `analyzeAndEvolveStrategy()` classifies top SC candidates by material family.
- OpenAI generates ranked focus areas with priorities and reasoning, influencing the formula generator to bias towards priority material families.
- The system tracks research strategy evolution, convergence metrics (e.g., best Tc, best score), and detects significant strategy pivots.

### Milestone Detection System
- Detects various types of milestones: new-family, tc-record, pipeline-graduate, diversity-threshold, knowledge-milestone, insight-cascade.
- Pipeline-graduate triggers at verification stage >= 4 (not 5), with batch milestones at 10/25/50/100 validated candidates.
- tc-record fires when Tc improves by >5K over previous best.

### Progressive Scoring
- XGBoost uses sigmoid scoring and incorporates knowledge depth bonuses for synthesis, crystal structure, pipeline stages, and insights. Final scores are clamped to [0, 1].
- Lambda-aware Tc scaling: base Tc estimates scale with electron-phonon coupling (lambda > 2.5: x1.4, > 2.0: x1.25, > 1.5: x1.12).

### Physics Tc Flowback
- Phase 10 writes Eliashberg-calculated Tc back to candidate's `predictedTc`.
- Raw physics Tc is clamped: must be >0 and <1000K, otherwise discarded.
- Tc increase cap is coupling-aware: lambda>2.5: +150K, lambda>2.0: +120K, lambda>1.5: +90K, lambda>1.0: +70K, else +50K.
- Strong coupling boost: lambda > 2.0 materials get additional +(lambda-2.0)*15K when physics Tc > 80% of ML Tc.
- Blend weight: lambda>1.5: 0.6, lambda>1.0: 0.5, else 0.4.

### Learning Feedback Loop (Re-evaluation)
- `reEvaluateTopCandidates()` runs every cycle after Phase 12.
- Checks top 50 candidates by Tc; uses `Map<string, number>` for apply count tracking (max 3 applications per key).
- Re-evaluation map clears every 8 cycles to allow repeat boosts with diminishing returns.
- Diminishing returns: each reapplication reduces boost by 50% (1 / (1 + prevCount * 0.5)).
- **Stagnation multiplier**: if Tc hasn't improved in N cycles, boosts are amplified (>20 cycles: x2.0, >10: x1.5, >5: x1.2).
- Stage-based Tc boosts (cumulative, not stage-exact): stage>=1 with coupling (+4 to +15K by lambda), stage>=2 (+4K), stage>=3 (+6K), stage>=4 (+8K, +6K crystal stability).
- Tc hard cap: 550K.
- `cyclesSinceTcImproved` and `lastBestTcSeen` tracked as module-level vars.

### Stagnation-Aware Candidate Generation
- `generateNovelSuperconductors()` receives stagnation info (cycles since improved, current best Tc).
- When stagnating (>5 cycles), the LLM prompt includes explicit context about the ceiling and instructions to generate candidates targeting higher Tc via ultra-high coupling, multi-component hydrides, and novel pairing mechanisms.

### Tc Knowledge Bonus (ML Predictor)
- Synthesis documented: +8K, Crystal structure: +8K, Pipeline stages: +8K each (max +30K), Related insights: +5K.
- Verified lambda bonus: >2.5: +30K, >2.0: +22K, >1.5: +15K, >1.0: +8K, >0.5: +3K.
- Total cap: +80K (was 45K).

### Phase Throughput
- Phase 10 (Physics): 5 candidates/cycle, queries `getSuperconductorsByStage(0)` directly.
- Phase 12 (Multi-Fidelity): 6 candidates/cycle, queries stages 0 and 1 combined.

### Convergence Snapshots
- Merges top-by-score and top-by-Tc candidates to find true bestTc across all candidates.
- Uses delete-before-insert per cycle to prevent duplicate cycle entries on engine restart.
- Score clamped to max 1.0 in snapshot computation.
- Ordered by `createdAt` (not cycle number) to handle engine restarts correctly.
- `storage.getSuperconductorCandidatesByTc(limit)` provides Tc-ordered query.
- `storage.deleteConvergenceSnapshotByCycle(cycle)` prevents duplicate cycle snapshots.

### SC Candidate Deduplication
- `formula` column has UNIQUE constraint; `insertSuperconductorCandidate` uses upsert on formula conflict (keeps highest score).
- Startup cleanup `deduplicateSuperconductorCandidates()` removes any existing duplicates.

### Database Schema Highlights
- Key tables include `elements`, `materials`, `learning_phases`, `novel_predictions`, `research_logs`, `synthesis_processes`, `chemical_reactions`, `superconductor_candidates`, `crystal_structures`, `computational_results`, `novel_insights`, `research_strategies`, `convergence_snapshots`, and `milestones`.

### UI/UX Decisions
- Real-time updates are provided via WebSocket for various components like the Command Center, Research Pipeline, and Convergence Tracker.
- NaN safety is implemented on both backend and frontend to prevent display and calculation errors.
- Status taxonomy for novel predictions includes `predicted`, `under_review`, and `literature-reported`.

## External Dependencies
- **OpenAI**: Used for gpt-4o-mini via Replit AI Integrations for NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: Relational database for persistent storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: Source for thermodynamic data and spectroscopic properties.
- **Materials Project**: Source for DFT-computed band gaps and formation energies.
- **Materials Science Knowledge Base**: OpenAI-sourced real materials data from peer-reviewed literature and reputable databases.
