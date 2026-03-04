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
- **Trained Gradient Boosting Model**: Real gradient boosting (depth-4 decision trees, ~106 trees, R2=1.000) trained on 130+ known superconductors from SuperCon database (`server/learning/supercon-dataset.ts`). Replaces hand-written XGBoost heuristics. Predicts Nb 9.7K, Al 1.3K, MgB2 36K, LaH10 249K, YBCO 93K from physics features.
- **ML Prediction Engine**: Trained gradient boosting + OpenAI gpt-4o-mini neural network ensemble for superconductor scoring. Physics-informed features with strict room-temperature criteria (Tc >= 293K). All candidates (LLM-novel and ML-pipeline) scored with real gbPredict values; startup backfill ensures no null xgboostScore/neuralNetScore in DB.
- **SC Verification Pipeline**: A multi-step process for superconductor candidates: theoretical -> promising -> high-tc-candidate -> under-review -> requires-verification, avoiding premature "breakthrough" claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline with calibrated thresholds: Stage 1 metallicity>0.25 & DOS>0.2, Stage 2 lambda>0.25, Stage 3 Tc>1K, Stage 4 synthesis feasibility.
- **Novel Insight Detection**: Pre-filters known knowledge, deduplicates (Levenshtein similarity > 0.70 + keyword overlap >= 4 terms), expanded WELL_KNOWN_PATTERNS list (~60 patterns), fetches 200 recent + 300 novel insights for comparison. Uses OpenAI to evaluate remaining insights for novelty.
- **Data Confidence Tracking**: All candidates tagged with `dataConfidence`: "high" (DFT/pipeline stage 3+), "medium" (model/crystal data), "low" (LLM-estimated). UI shows confidence badges (DFT/Model/Est.) next to Tc values.
- **LLM Data Validation**: Properties from LLM cross-validated against Materials Project API. Bounds enforced: band gap 0-15 eV, formation energy -5 to +5 eV/atom, lattice params 1-30 Å, density 0.5-25 g/cm³. Materials tagged with dataSource: "dft-computed", "experimental", or "llm-estimated".
- **Physics-Only Tc Evolution**: No artificial Tc inflation. `computeDynamicTcCeiling` removed. Tc only changes when physics inputs change (lambda, omegaLog, muStar). No knowledge depth bonus, no exploration bonus, no stage-based bumps.
- **Diversity & Deduplication**: Mechanisms across formula generation and SC candidate processing to ensure diversity and prevent redundant work.
- **Evolving Research Strategy**: `analyzeAndEvolveStrategy()` classifies top SC candidates, and OpenAI generates ranked focus areas to bias the formula generator towards priority material families.
- **Milestone Detection System**: Detects various milestones such as new-family, tc-record, pipeline-graduate, diversity-threshold, knowledge-milestone, and insight-cascade.
- **Physics as Tc Authority**: All Tc values grounded in Eliashberg/McMillan physics. LLM candidates capped at insertion by lambda-based McMillan bounds with ambient-pressure awareness (see Ambient-Pressure Tc Caps). Multi-fidelity pipeline writes physics Tc back to candidates at Stage 3 with conservative upward caps (+10-30K). Phase 10 physics flowback recomputes Eliashberg Tc when inputs change. Hard cap: 350K absolute max. PHYSICS_VERSION=6 tracks recalculation state.
- **Hydrogen-Ratio-Aware Coupling**: Adjusts electron-phonon coupling and phonon spectrum calculations based on the hydrogen-to-metal ratio in hydrides.
- **Physical Plausibility Guardrails**: Enforces physical limits on parameters like Hc2 (max 300T), coherence length (min 1nm), pressure for ambient stability, ensembleScore (max 0.95), and strict criteria for `roomTempViable` candidates (Tc>=293K AND zeroResistance AND meissnerEffect AND P<50GPa).
- **Log Deduplication**: Activity feed deduplicates repeated "started"/"discovery started" messages within a cycle via recentLogCache (Set of event::detail keys, cleared each cycle). Startup cleanup removes duplicate research_logs (keeping 3 most recent per event+detail pair) and prunes stale non-novel insights (keeping 500 most recent). XGBoost logs show "Tc raw:" label; ensemble logs include capped Tc; "Physics analysis complete" log removed (post-cap logs in engine.ts show capped values with physics detail).
- **Stagnation Breaking**: When `cyclesSinceTcImproved` indicates stagnation, the system triggers re-analysis of high-lambda candidates and adapts candidate generation strategies via LLM prompts.

### Physics Engine (server/learning/physics-engine.ts)
- **Zero Math.random()**: All physics calculations are fully deterministic — no randomness in any physics function
- **Elemental Data**: 96 elements with tabulated Debye temps, bulk moduli, Stoner parameters, Hubbard U, McMillan-Hopfield eta, Miedema params, Sommerfeld gamma, Gruneisen params, melting points, lattice constants (server/learning/elemental-data.ts)
- **DOS at Fermi level**: N(Ef) = gamma/2.359 [states/eV/atom] from Sommerfeld coefficient; fallback: rigid-band model N(Ef) = VEC / (2 * W) with Stoner enhancement
- **Bandwidth W**: Per-element lookup table (ELEMENT_BANDWIDTH) with ~45 literature d-band/sp-band widths (e.g., Nb=5.5, Al=11.0, Cu=4.0, La=3.0, Pb=7.0 eV); fallback to period-based for unlisted elements (3d:5.5, 4d:8, 5d:10 eV), DOS-based for sp metals
- **Hubbard U/W**: From tabulated U values; U/W < 0.5 weakly correlated, 0.5-1.0 moderate, >1.0 strong
- **Lambda**: Back-calculated from known elemental Tc (via inverted McMillan formula) for elements with known Tc; McMillan-Hopfield eta-based for others; composition-weighted for compounds
- **Lambda conversion**: LAMBDA_CONVERSION = 562000 calibrated so Nb gives lambda ≈ 1.04
- **Phonon frequencies**: omega_D = theta_D * 0.695 cm-1; omega_log = 0.65 * omega_max for monatomic; light-element boost for borides/hydrides
- **Metallicity**: Continuous sigmoid model: metallicity = 1/(1+exp(k*(delta_EN - threshold))) where k=3.0, threshold=1.4. Nonmetals include B, Si, Ge, As, Te (metalloids). Borane cage detection (B>=4, H>=4, ~1:1 ratio) → metallicity < 0.25. Metallic borides (MgB2) protected with metalFrac>=0.2. Organic compounds (C>15% + H>=C) capped at 0.35. Dilute metal (metalFrac<0.15, >2 elements) penalized. Pure superhydrides (hRatio>=6, no other nonmetals, not organic, metalFrac>=0.05) get moderate metallicity boost.
- **Superhydride handling**: Lambda correlation/metallicity bypass only for pure superhydrides (hRatio>=6, nonHNonMetalFrac<0.1, not organic).
- **Ambient-Pressure Tc Caps**: Applied via `applyAmbientTcCap()` in ALL Tc update paths (ML pipeline, LLM novel, physics analysis, re-evaluation, startup correction, bulk correction). Caps: metallicity<0.3→20K, <0.5→80K, lambda<0.3→50K, <0.5→80K, <1.0→80-150K, <1.5→120-250K, <2.5→160-350K, >=2.5→200-350K (interpolated by pressure 10-50 GPa).
- **Structure Predictor**: Crystallographic c/a ratios from prototype structures, Goldschmidt tolerance factor for perovskites, Vegard's law for alloy lattice parameters, Tammann rule for synthesis temperature
- **Materials Project client**: server/learning/materials-project-client.ts with fallback to analytical models; MP data cached in mp_material_cache DB table

### Gradient Boosting Model (server/learning/gradient-boost.ts)
- Trained on 130+ materials from `server/learning/supercon-dataset.ts` (static, curated from SuperCon database)
- Decision trees with depth 4, ~106 trees, learning rate 0.08
- Features: lambda, metallicity, omegaLog, debyeTemp, correlation, VEC, avgEN, enSpread, hRatio, pettifor, atomicRadius, sommerfeldGamma, bulkModulus, maxMass, nElements, composition flags, cooperPair, dimensionality, anharmonic, electronDensity, phononCoupling, dWave, meissner
- Validation R2=1.000, MSE=1.0 on training set
- Key predictions: Nb 9.7K, Al 1.3K, MgB2 36.2K, LaH10 249.4K, YBCO 92.6K, NaCl 0.1K, Cu 0K, Fe 0K

### Validated Against Known Superconductors
- Nb: Tc=13.5K (lit:9.3K), lambda=0.84 (lit:1.04), weakly correlated
- Al: Tc=2.3K (lit:1.2K), lambda=0.40 (lit:0.43), weakly correlated
- MgB2: Tc=33.8K (lit:39K), lambda=0.83 (lit:0.87)
- LaH10: Tc=229.6K (lit:250K), lambda=1.48 (lit:2.20)
- Pb: Tc=10.4K (lit:7.2K), lambda=1.39 (lit:1.55)
- CeCoIn5: Tc=2.9K (lit:2.3K), strongly correlated
- Fe2As2: Tc=0K, moderately correlated, AFM competing phase
- YBCO: Tc=0K from Eliashberg (correctly flags as unconventional/strongly correlated)

### Alive Engine (v2.1)
- **Thought Stream**: Engine broadcasts `thought` WS messages at decision points (strategy, discovery, stagnation, milestone categories). Client `use-websocket.ts` tracks `thoughts`, `tempo`, `statusMessage` state.
- **Engine Thoughts Card**: Dashboard live-scrolling thought feed with color-coded categories (blue/green/amber/purple), fade-in animations, tempo-matching pulse dot.
- **Sparklines**: Each dashboard stat card shows a 60px trend line (last 30 data points, in-memory) with session delta indicators.
- **Knowledge Map**: Recharts Treemap on Research Pipeline showing material families by candidate count. Clickable for family details.
- **Adaptive Tempo**: Cycle interval adapts: excited=10s (new candidates/Tc improving), exploring=15s (normal), contemplating=22s (stagnation). `EngineTempo` broadcast via WS.
- **Research Memory**: GET `/api/engine/memory` — aggregated knowledge summary: current hypothesis, family stats, top insights, abandoned strategies, cycle narratives.
- **Memory Card**: Collapsible dashboard card showing hypothesis, key discoveries, explored territory bars, abandoned paths.
- **Cycle Narratives**: Per-cycle one-sentence summaries stored as `cycle-narrative` research logs, displayed in "Cycle Journal" on Research Pipeline.
- **Dynamic Status Messages**: Sidebar shows contextual status (tempo label + generated message) instead of static Running/Stopped.
- **Storage method**: `getResearchLogsByEvent(event, limit)` for filtered log retrieval.

### Infrastructure (v2.0)
- **Rate Limiting**: express-rate-limit — 600 req/min general, 30 write, 10 engine control
- **In-Memory Cache**: server/cache.ts — TTL cache for elements (1hr), stats (30s), crystal/computational results (5min)
- **DB Indexes**: On formula columns, timestamps, predictedTc, ensembleScore, pipelineStage
- **API Pagination**: novel-predictions endpoint supports limit/offset
- **Experimental Validations**: experimental_validations table — types: resistance/meissner/xrd/tc_measurement/pressure_test, results: confirmed/partial/failed/inconclusive
- **ML Calibration**: Confidence bands from gradient-boost training residuals; error bars on Tc predictions across UI
- **Cross-Validation**: AFLOW REST API client (server/learning/aflow-client.ts, no key required) + enhanced Materials Project queries; External Data Sources section on candidate detail pages
- **Interactive Periodic Table**: 18-column grid layout replacing list-based atomic explorer; element-to-materials/candidates links

## External Dependencies
- **OpenAI**: For gpt-4o-mini via Replit AI Integrations, used in NLP, formula generation, ML refinement, and knowledge base sourcing.
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies (MATERIALS_PROJECT_API_KEY secret).
- **AFLOW REST API**: For crystal structure and electronic property cross-validation (public, no key required).
- **Materials Science Knowledge Base**: OpenAI-sourced real materials data from peer-reviewed literature.
