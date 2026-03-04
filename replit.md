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

### Physics as Tc Authority
- **LLM Tc capping** (superconductor-research.ts): New LLM-generated candidates have their Tc capped at physics-plausible maxima using McMillan formula and lambda-based tiers (lambda<0.3: 150K, <0.5: 200K, <1.0: 300K, <1.5: 400K, else 500K). Strongly correlated materials (corr>0.7) capped at 200K; Mott insulators (corr>0.85) at 80K. Original LLM estimate stored in notes.
- **Mott insulator penalty**: Eliashberg Tc is multiplied by 0.05 for Mott insulators (correlation>0.7 + Mott phase), 0.3 for strongly correlated, 0.7 for moderately correlated. Eliashberg theory assumes itinerant electrons; Mott insulators have localized electrons so the framework doesn't apply.
- **Downward physics correction**: Phase 10 physics flowback always blends physics Tc with current Tc when physics is lower. Blend weight is 0.7 physics / 0.3 current when physics < 50% of current. Strong coupling boost disabled for Mott insulators and strongly correlated materials.
- **Multi-fidelity Tc writeback**: Pipeline Stage 3 Eliashberg Tc now updates `predictedTc` using same directional blend logic with lambda-based weights.
- **Retroactive correction**: Startup corrects ALL candidates with physics data where Eliashberg Tc < current Tc and current > 100K. Applies Mott/correlation penalty before blend.
- Raw physics Tc is clamped: must be >0 and <1000K, otherwise discarded.
- Tc increase cap is coupling-aware: Mott: +10K, strongly correlated: +30K, else lambda>2.5: +150K, lambda>2.0: +120K, lambda>1.5: +90K, lambda>1.0: +70K, else +50K.
- ML predictor: Mott insulators (corr>0.85) get score penalty (-0.08) and low Tc estimate (5+score*80). Non-metallic materials (metallicity<0.3) get -0.15 score and Tc capped at 1+score*15. Moderate correlation (0.6-0.85) gets smaller bonus and capped Tc scaling.
- **Metallicity gate**: `computeElectronicStructure` estimates metallicity from composition using stoichiometric ratio analysis. Key checks: (1) molecular detection (N-H ammine complexes, hydroxides, polyhalides), (2) light-to-metal atom ratio (high ratio = likely ionic/molecular), (3) electronegative ligand count. Non-metallic materials (metallicity<0.4) get Eliashberg Tc multiplied by metallicity factor. Applies in physics engine, engine flowback, multi-fidelity pipeline, LLM capping, and ML predictor.
- **Formula parsing**: `parseFormulaCounts()` extracts element counts from formulas for stoichiometric analysis. Pure hydrides (LaH₁₀, TaH₉) are protected from ratio penalty since H IS the metallic sublattice; only penalized when electronegative ligands (N, O, Cl) are present alongside H.

### Hydrogen-Ratio-Aware Coupling
- `computeElectronPhononCoupling` accepts optional `formula` parameter for stoichiometric analysis.
- Low-hydrogen hydrides (H:metal <= 3, e.g. ZrH₂) get 0.25x lambda; H:metal <= 5 get 0.5x; <= 7 get 0.75x.
- `computePhononSpectrum` uses hydrogen ratio: H-rich (H:metal >= 6) get full 3500-4300 cm⁻¹ phonons; low-H hydrides get 1500-2100 cm⁻¹ (ionic H⁻ vibrations, not metallic H).
- Rationale: ZrH₂ is fluorite-structure Zr⁴⁺ + 2H⁻ (ionic, not metallic H sublattice), experimentally Tc ≈ 0K. LaH₁₀/TaH₉ have H-dominated metallic cage.
- **Retroactive correction at startup** now recomputes coupling from scratch using formula-aware functions, updating both lambda and Tc. Also seeds physics data into candidates that had none (Tc > 200K), and hard-caps lambda < 0.2 candidates to Tc = max(1, 5% of stored Tc).

### Physics-as-Authority Tc Governance
- **LLM cap tightened**: New candidates capped using McMillan formula as ceiling, not floor. Lambda >= 2.5 caps at min(500, McMillan*1.3); lambda >= 1.5 at min(450, McMillan*1.5); lambda < 0.3 at min(150, McMillan*3.0).
- **Re-physics downward correction**: When re-analyzing stage-4 candidates, physics Tc < current Tc now blends downward (0.7 weight for large gaps). Previously `Math.max(currentTc, updatedTc)` prevented any reduction.
- **Multi-fidelity pipeline**: Already writes Eliashberg Tc back to `predictedTc` at Stage 3 with physics-dominant blending.
- **Convergence dual-line**: `bestPhysicsTc` tracks highest Tc among physics-validated candidates (stage >= 1, lambda != null). Displayed alongside `bestTc` on convergence chart.

### Learning Feedback Loop (Re-evaluation)
- `reEvaluateTopCandidates()` runs every cycle after Phase 12.
- Evidence-gated: fires only when stage changes, lambda delta >0.1, new crystal, or ceiling rises.
- Uses `Map<string, {stage, lambda, hasCrystal, lastCeiling}>` to track what's been applied.
- **Blocked for**: Mott-like (corr>0.85), strongly correlated (only +2K at stage 4), and low-H-ratio hydrides (H:metal <= 3, zero boost).
- Stage-based Tc boosts (cumulative): stage>=1 with coupling (+4 to +15K by lambda), stage>=2 (+4K), stage>=3 (+6K), stage>=4 (+8K, +6K crystal stability).
- **Dynamic Tc ceiling** replaces fixed 550K cap. Ceiling grows with accumulated evidence:
  - Base: 500K
  - +3K per 10 stage-4 candidates (max +50K)
  - +5K per 500 novel insights (max +40K)
  - +5K per 100 crystal structures (max +30K)
  - +5K per 500 computational results (max +30K)
  - +10-20K for top-candidate avg lambda >2.0/2.5
  - Cached for 3 cycles, recomputed via `computeDynamicTcCeiling()`
- When ceiling rises, candidates near the old ceiling get a partial boost (50% of ceiling delta).
- `cyclesSinceTcImproved` and `lastBestTcSeen` tracked as module-level vars.

### Re-Physics for Stagnation Breaking
- When `cyclesSinceTcImproved > 3`, Phase 10 re-analyzes 2 stage-4 high-lambda candidates.
- If lambda changes by >0.05, Tc is updated based on the delta: `+(newLambda-oldLambda)*20K`.
- This produces new evidence that triggers re-evaluation in the next cycle.

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
- `bestPhysicsTc` tracks highest Tc among physics-validated candidates (verificationStage >= 1 AND electronPhononCoupling IS NOT NULL).
- Convergence chart shows two Tc lines: red for overall best, green for physics-validated best.
- Uses delete-before-insert per cycle to prevent duplicate cycle entries on engine restart.
- Score clamped to max 1.0 in snapshot computation.
- Ordered by cycle number (ASC) for consistent trajectory display.
- `storage.getSuperconductorCandidatesByTc(limit)` provides Tc-ordered query.
- `storage.deleteConvergenceSnapshotByCycle(cycle)` prevents duplicate cycle snapshots.
- `storage.getMaxConvergenceCycle()` returns the highest cycle number for cumulative counting.

### Cumulative Cycle Counter
- `startEngine()` initializes `cycleCount` from `getMaxConvergenceCycle()` to resume from the last cycle.
- Prevents cycle number collisions and ensures convergence trajectory is monotonically ordered.
- ML predictor no longer caps Tc at 550K — ceiling enforcement is solely in engine.ts via `computeDynamicTcCeiling()`.

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
