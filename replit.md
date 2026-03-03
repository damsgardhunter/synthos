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

### Real-Time Updates
- WebSocket hook (`client/src/hooks/use-websocket.ts`) provides real-time data refresh on all pages.
- Each page listens for relevant WebSocket events and invalidates its TanStack Query cache keys.
- Engine state changes, phase updates, and new data broadcast to all connected clients.

### NaN Safety
- Backend: `safeNumber()`, `safeDivide()`, `safeFixed()` in `server/learning/utils.ts` guard all physics calculations.
- Frontend: `safeNum()` and `safeDisplay()` in `client/src/lib/utils.ts` prevent NaN display.
- `Number.isFinite()` guards on all `.toFixed()` calls and conditional rendering of numeric values.
- `PhysicsValue` component skips rendering if value is NaN/Infinity.
- `toNum()` helper validates OpenAI JSON responses in structure predictor.

### Core Features
- **Command Center**: Dashboard with system statistics, learning pipeline status, engine controls, and live activity feed.
- **Atomic Explorer**: Full periodic table (all 118 elements, H through Og) with Bohr model diagrams.
- **Materials Database**: Comprehensive indexed materials from multiple sources.
- **Novel Discovery**: AI-predicted new material candidates.
- **Superconductor Lab**: Details on superconductor candidates, including physics fields, synthesis, reactions, and ML scores.
- **Computational Physics**: Multi-fidelity screening pipeline, physics property cards, crystal structures, and negative results analysis.
- **Research Pipeline**: Tracks the full learning phase progress and activity log. Finite phases (1-2) show fraction progress; open-ended phases (3-12) show activity bar with accumulated count and milestone tracking.
- **Learning Engine**: Orchestrates 12 distinct learning phases covering subatomic to multi-fidelity screening, with balanced priority and continuous cycling every 15 seconds.
- **ML Prediction Engine**: Hybrid XGBoost and OpenAI gpt-4o-mini neural network ensemble (40% XGBoost, 60% NN) for superconductor scoring, incorporating 18+ physics-informed features. Strict room-temperature criteria (Tc >= 293K, zero resistance, Meissner effect, ambient pressure).
- **SC Verification Pipeline**: A multi-step process for superconductor candidates: theoretical -> promising -> high-tc-candidate -> under-review -> requires-verification, emphasizing no premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline for candidate evaluation: ML filter, electronic structure, phonon/e-ph coupling, Tc prediction/competing phases, and synthesis feasibility/convex hull stability. Failures are logged with reasons.
- **Data Fetching & NLP**: Live fetching of materials data and OpenAI-powered NLP for analysis and property prediction.
- **Formula Generation**: AI-driven generation of novel chemical compositions.
- **Synthesis & Reaction Tracking**: Detailed tracking of material synthesis processes and a broad chemical reaction knowledge base.
- **Crystal Structure Prediction**: AI-powered prediction of crystal structures, including stability and synthesizability.
- **Candidate Detail Page**: Unified profile at `/candidate/:formula` aggregating SC candidate data, ML scores, crystal structures, synthesis pathways, chemical reactions, and pipeline results. Cross-linked from SC Lab, Physics, Novel Discovery, and Materials pages via "View Full Profile" links.

### Evolving Research Strategy
- After each engine cycle, `analyzeAndEvolveStrategy()` in `server/learning/strategy-analyzer.ts` classifies top SC candidates by material family (Hydrides, Cuprates, Pnictides, Chalcogenides, Borides, Carbides, Nitrides, Oxides, Intermetallics, Other).
- OpenAI generates ranked focus areas with priorities and reasoning based on per-family performance signals.
- Strategy hint is passed to the formula generator (Phase 6) to bias LLM toward priority material families.
- API: `GET /api/research-strategy`, `GET /api/research-strategy/history?limit=20`
- UI: "Research Strategy" card on Dashboard showing ranked priority bars, AI summary, and evolution counter. Real-time updates via `strategyUpdate` WebSocket event.

### Convergence Tracking
- `captureConvergenceSnapshot()` records bestTc, bestScore, avgTopScore, candidatesTotal, pipelinePassRate, novelInsightCount, topFormula per cycle.
- API: `GET /api/convergence?limit=50`
- UI: "Convergence Tracker" on Research Pipeline page with dual-axis LineChart (Best Tc + Best Score over cycles), momentum indicator (Improving/Plateaued/Declining based on recent vs previous 3 snapshots), best candidate callout, and pipeline pass rate. Real-time updates via `convergenceUpdate` WebSocket event.

### Database Schema Highlights
- `elements`, `materials`, `learning_phases`, `novel_predictions`, `research_logs`.
- `synthesis_processes`: Detailed synthesis conditions.
- `chemical_reactions`: Balanced equations, thermodynamic data, SC relevance.
- `superconductor_candidates`: ML scores, physics fields, verification status.
- `crystal_structures`: Space groups, lattice parameters, stability, synthesizability.
- `computational_results`: Pipeline results, pass/fail, failure reasons.
- `novel_insights`: Insight novelty tracking with deterministic deduplication (content+phase hash IDs), novelty scores, categories, and OpenAI evaluation.
- `research_strategies`: Strategy evolution history with focus areas (jsonb), summary, and performance signals.
- `convergence_snapshots`: Per-cycle metrics tracking bestTc, bestScore, avgTopScore, pipelinePassRate, topFormula, familyDiversity, duplicatesSkipped.

### Novel Insight Detection
- Integrated into phases 3, 5, and 7 of the learning engine.
- Pre-filters well-known textbook knowledge (25+ patterns).
- Levenshtein similarity deduplication (>0.85 threshold).
- OpenAI evaluates remaining insights for novelty (categories: novel-correlation, new-mechanism, cross-domain, computational-discovery, design-principle, textbook, known-pattern, incremental).
- Deterministic IDs via SHA-256 hash of `{phaseId}:{insightText}` prevent duplicate storage.
- Novel insights (score >= 0.4) emitted as WebSocket events and logged.
- API: `GET /api/novel-insights?limit=50&novelOnly=true`
- UI: "Insight Novelty Tracker" section on Research Pipeline page with novelty bars, category badges, and novel/textbook visual distinction.

### Status Taxonomy
- Novel predictions use statuses: `predicted` (AI-generated), `under_review`, `literature-reported` (known from published research).
- `synthesized` status is deprecated — mapped to `literature-reported` in UI for backward compatibility.
- LaH₁₀ seed data: status `literature-reported`, Tc ~250K, citing Drozdov et al. 2019. Labeled as reference data, not a platform prediction.

### Engine Diversity & Deduplication
- **SC Formula Deduplication**: Before inserting a new SC candidate, `getSuperconductorByFormula()` checks for existing records. If duplicate with higher score, updates in place; otherwise skips. Duplicates skipped count tracked per cycle.
- **Novel Prediction Deduplication**: `getNovelPredictionByFormula()` prevents duplicate novel predictions. Higher-confidence duplicates update existing records.
- **In-memory exclusion**: `recentlyGenerated` array (last 50 formulas) in formula generator prevents re-generating known formulas within a session.
- **Exclusion lists in LLM prompts**: Both formula generator and SC novel generation include exclusion lists of top 20 existing formulas to prevent LLM from re-proposing known compositions.
- **Exploration bonus**: XGBoost scoring adds +0.05-0.08 bonus for under-explored families (<5 candidates), +up to 0.05 for strategy-aligned families, capped at +0.10 total. Uses `classifyFamily()` from `server/learning/utils.ts`.
- **Diversity-forcing novel SC generation**: Selects 1 example per distinct material family (not top-5 by score) for few-shot LLM prompt. Instructs LLM to generate from different families (pnictides, borides, kagome metals, clathrate hydrides, heavy fermion compounds).
- **Convergence diversity metrics**: `familyDiversity` (distinct families in top 50) and `duplicatesSkipped` tracked per convergence snapshot and displayed in UI.
- Phase 7 uses rotating material offset (increments by 50 each cycle, wraps around) to avoid re-scoring the same materials.
- Phase 11 checks `getCrystalStructuresByFormula()` before predicting — skips formulas that already have structures.
- Phases 10/11/12 fetch 50 candidates (not 8-10) and shuffle before slicing to ensure diverse candidate selection.
- Crystal structures in UI are deduplicated by formula, keeping the entry with lowest `convexHullDistance`.
- `classifyFamily()` shared utility in `server/learning/utils.ts` used by strategy-analyzer, ml-predictor, superconductor-research, and storage.

### Cross-Page Navigation
- `GET /api/candidate-profile/:formula` returns unified data: SC candidates, crystal structures, computational results, synthesis processes, and chemical reactions for a formula.
- Storage methods: `getSuperconductorsByFormula`, `getComputationalResultsByFormula`, `getSynthesisProcessesByFormula`, `getChemicalReactionsByFormula` (plus existing `getCrystalStructuresByFormula`).
- Route `/candidate/:formula` renders `candidate-detail.tsx` with all sections.
- "View Full Profile" links on SC Lab, Physics, Crystal Structures, Novel Discovery, and Materials Database pages.

### Hc2 (Upper Critical Field) Integration
- Physics engine computes Hc2 via BCS coherence length: `xi = hbar*vF / (pi*Delta0)` with `Delta0 = 1.764*kB*Tc*(1+lambda*0.3)`, then `Hc2 = Phi0/(2*pi*xi^2)`.
- Realistic values: YBCO ~79T, MgB2 ~10T, hydrides ~700T, weak SC ~0.6T.
- XGBoost scoring: Hc2>50T gets +0.08 bonus, Hc2=0 gets -0.10 penalty.
- NN prompt instruction #10 explains Hc2 significance.
- Multi-fidelity pipeline: stage 3 flags Hc2=0 with Tc>50K as suspicious; stage 4 flags Hc2<5T for room-temp candidates.
- Seed migration resets stale Hc2=0 values to NULL for recomputation.

### Self-Improving SC Model
- ML predictor loads verified physics data (lambda, Tc) and crystal structures for existing SC candidates from database.
- XGBoost feature extraction enriched with physics context (+0.05 for physics-verified, +0.04 for stable crystal) and penalties (-0.08 for competing phases, -0.05 for low synthesizability).
- Neural network prompt treats verifiedPhysics and crystalStructure as high-fidelity ground truth over estimates.
- Enrichment counts logged during XGBoost screening.
- `upperCriticalField` added to `MLFeatureVector` for Hc2-aware scoring.

## External Dependencies
- **OpenAI**: Used for gpt-4o-mini via Replit AI Integrations for NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: Relational database for persistent storage.
- **OQMD API**: `http://oqmd.org/oqmdapi/formationenergy` for live materials data fetching.
- **NIST WebBook**: Source for thermodynamic data and spectroscopic properties.
- **Materials Project**: Source for DFT-computed band gaps and formation energies.
- **Materials Science Knowledge Base**: OpenAI-sourced real materials data from peer-reviewed literature and reputable databases.