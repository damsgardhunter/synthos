# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
A comprehensive materials science research platform that simulates an AI supercomputer progressively learning everything about materials science — from subatomic particles through molecular modeling to novel material discovery. Features a real learning engine with live data fetching, NLP pattern recognition, novel formula generation, XGBoost+NN hybrid ML for superconductor prediction, synthesis process tracking, chemical reaction knowledge base, and real-time WebSocket updates. Primary research goal: discovering a room-temperature superconductor.

## Architecture

### Stack
- **Frontend**: React + Vite + TanStack Query + wouter routing
- **Backend**: Express.js + TypeScript
- **Database**: PostgreSQL via Drizzle ORM
- **UI**: shadcn/ui components + Tailwind CSS + Recharts
- **AI**: OpenAI gpt-4o-mini (via Replit AI Integrations) for NLP, ML ensemble, synthesis analysis
- **Real-time**: WebSocket (ws) for live progress updates

### Key Routes
- `/` — Command Center (dashboard with 8 stat cards, learning pipeline, engine controls, live activity feed)
- `/atoms` — Atomic Explorer (element browser with Bohr model diagrams)
- `/materials` — Materials Database (browse indexed materials from 4 databases)
- `/discovery` — Novel Discovery (AI-predicted new material candidates)
- `/superconductor` — Superconductor Lab (SC candidates, synthesis processes, chemical reactions, ML scores)
- `/research` — Research Pipeline (full learning phase progress + activity log)

### API Endpoints
- `GET /api/elements` — All learned elements
- `GET /api/elements/:id` — Single element details
- `GET /api/materials?limit&offset` — Paginated materials list
- `GET /api/materials/:id` — Single material details
- `GET /api/learning-phases` — All 9 learning phases with progress
- `GET /api/novel-predictions` — AI-generated novel material predictions
- `GET /api/research-logs?limit` — Research event log
- `GET /api/stats` — Aggregate system statistics (elements, materials, predictions, SC candidates, synthesis, reactions)
- `GET /api/superconductor-candidates?limit` — ML-scored superconductor candidates
- `GET /api/synthesis-processes?limit` — Lab synthesis processes
- `GET /api/chemical-reactions?limit` — Chemical reaction knowledge base
- `POST /api/engine/start` — Start the learning engine
- `POST /api/engine/stop` — Stop the learning engine
- `POST /api/engine/pause` — Pause the learning engine
- `POST /api/engine/resume` — Resume the learning engine
- `GET /api/engine/status` — Current engine state and metrics

### WebSocket
- Path: `/ws`
- Broadcasts: engineState, phaseUpdate, progress, log, prediction, insight, taskStart, taskEnd, cycleStart, cycleEnd

### Database Tables
- `elements` — 30+ real elements with properties (atomic mass, electronegativity, electron config, etc.)
- `materials` — Materials from OQMD, AFLOW, and Materials Science Knowledge Base (grows via live fetching + known materials import)
- `learning_phases` — 9 phases: Subatomic -> Elements -> Bonding -> Materials -> Prediction -> Discovery -> SC Research -> Synthesis -> Reactions
- `novel_predictions` — AI-predicted novel materials (grows via formula generator)
- `research_logs` — Activity log of research events by data source
- `synthesis_processes` — How materials are created (method, exact temperatures, heating rates, hold times, cooling methods, precursors, equipment, safety)
- `chemical_reactions` — Lab reactions with balanced equations, thermodynamic data, mechanisms, SC relevance scores
- `superconductor_candidates` — ML-scored SC candidates with XGBoost/NN/ensemble scores, Meissner/Cooper pair/coherence data, verification status
- `conversations` — Chat conversations (AI integrations)
- `messages` — Chat messages (AI integrations)

## Learning Engine (server/learning/)
The learning engine orchestrates balanced AI research across 9 phases:

- **data-fetcher.ts** — Fetches real materials from OQMD/AFLOW APIs + imports known human-created materials (12 topic categories, beginner to master level) from OpenAI sourced from reputable journals/databases
- **nlp-engine.ts** — Uses OpenAI to analyze bonding patterns, predict properties, classify materials
- **formula-generator.ts** — Uses OpenAI to generate novel chemical compositions with predicted properties
- **ml-predictor.ts** — XGBoost feature extraction (18 physics features) + OpenAI neural network refinement for superconductor scoring. Ensemble = XGB*0.4 + NN*0.6. Now accepts ResearchContext (synthesis/reaction counts) for informed predictions.
- **superconductor-research.ts** — Strict SC research: both zero resistance AND room temperature required. Multi-step verification pipeline: theoretical -> promising -> high-tc-candidate -> under-review -> requires-verification. No premature breakthroughs.
- **synthesis-tracker.ts** — Discovers how materials are made with exact conditions (peak temperature in C, heating rate, hold time, cooling method, atmosphere, pressure). Broader reaction topics covering all materials chemistry.
- **engine.ts** — Main orchestrator: balanced phase priority, manages WebSocket broadcasts, tracks progress

### Engine Cycle (Balanced Priority)
1. **Foundation first**: Phase 4 (Materials + Known Materials Import) + Phase 8 (Synthesis) + Phase 9 (Reactions) run in parallel
2. **Analysis second**: Phase 3 (Bonding) + Phase 5 (Prediction) run next
3. **Discovery third**: Phase 6 (Novel Discovery) runs
4. **SC Research last**: Phase 7 (SC Research) only runs after minimum knowledge threshold (5+ materials, 3+ synthesis paths, 3+ reactions)
5. Cycles repeat every 60 seconds while engine is running

### ML Prediction Engine
- XGBoost layer: Pure TypeScript gradient boosting with 18 physics-informed features (electronegativity, Cooper pair strength, phonon coupling, Meissner potential, d-wave symmetry, layered structure, etc.)
- Neural network layer: OpenAI gpt-4o-mini refines XGBoost predictions with deep physics reasoning
- Ensemble: 40% XGBoost + 60% Neural Network weighted scoring
- Room-temp criteria (STRICT): Tc >= 293K AND zero electrical resistance AND Meissner effect AND ambient/low pressure (< 50 GPa) — ALL must be verified

### SC Verification Pipeline
Candidates go through multi-step verification before any claims:
- `theoretical` — Initial ML prediction, criteria not fully met
- `promising` — Some criteria met, needs more evaluation
- `high-tc-candidate` — High Tc predicted with zero resistance
- `under-review` — Room-temp + zero resistance + Meissner predicted, needs detailed review
- `requires-verification` — All criteria appear met, requires: independent synthesis confirmation, four-probe resistance measurement, SQUID magnetometry, reproducibility by 2+ independent labs
- No "breakthrough" or "confirmed" status exists — all predictions are theoretical until experimentally verified

### Known Materials Import
12 topic categories across 4 difficulty levels:
- **Beginner**: Common everyday materials, basic ceramics and oxides
- **Intermediate**: Semiconductors, batteries, structural alloys
- **Advanced**: Known superconductors, topological materials, functional ceramics
- **Master**: MXenes/MOFs/perovskites, extreme environment materials, metamaterials, high-entropy alloys

### Chemical Reaction Topics (Broad Coverage)
20 topics covering all materials chemistry: oxide formation, metal reduction, sol-gel chemistry, battery reactions, combustion synthesis, CVD, corrosion, precipitation, polymer synthesis, semiconductor doping, photocatalysis, ALD, and more (not just superconductor-focused).

### Synthesis Process Detail Level
Each synthesis process includes:
- Peak temperature (exact C), heating rate (C/min), hold time (hours at specific temp)
- Cooling method (furnace cool, quench, controlled ramp)
- Atmosphere (specific gas, flow rate), pressure (atm)
- Intermediate steps (regrinding, multiple sintering cycles)
- Precursors with purity levels, equipment list, safety notes, yield percent

### OpenAI Configuration
- Model: gpt-4o-mini for all NLP tasks
- Uses Replit AI Integrations (env vars: AI_INTEGRATIONS_OPENAI_BASE_URL, AI_INTEGRATIONS_OPENAI_API_KEY)
- Batch processing with p-limit (concurrency: 2) and p-retry (retries: 5)
- Synthesis tracker: max_completion_tokens 3500 (batch size 4 materials for detail)
- Reaction discovery: max_completion_tokens 2000 (5-7 reactions per topic)

## Data Sources
- **NIST WebBook** — Thermodynamic data, spectroscopic properties
- **Materials Project** — DFT-computed band gaps, formation energies
- **OQMD** — `http://oqmd.org/oqmdapi/formationenergy` (live fetching, no auth)
- **AFLOW** — `http://aflow.org/API/aflowlib/` (live fetching, no auth)
- **Materials Science KB** — OpenAI-sourced real materials from peer-reviewed literature (Nature, Science, NIST, CRC Handbook, Callister textbook)

## File Structure
```
client/src/
  pages/
    dashboard.tsx           — Command Center + Engine Controls + 8 stat cards
    atomic-explorer.tsx     — Element browser + Bohr diagrams
    materials-database.tsx  — Materials browser
    novel-discovery.tsx     — AI predictions
    superconductor-lab.tsx  — SC candidates, synthesis, reactions (tabbed)
    research-pipeline.tsx   — Pipeline + activity log
  components/
    app-sidebar.tsx         — Navigation sidebar with 6 pages + phase progress
    engine-controls.tsx     — Start/Stop/Pause engine + live activity feed
  hooks/
    use-websocket.ts        — WebSocket hook for real-time updates

server/
  index.ts       — Express server entry
  routes.ts      — API routes + engine control + SC/synthesis/reaction endpoints
  storage.ts     — Database abstraction (IStorage) with CRUD for all 8 tables
  db.ts          — Drizzle DB connection
  seed.ts        — Seed data (elements, phases 1-9, materials, predictions, logs)
  learning/
    engine.ts              — Main orchestrator + WebSocket server (9 phases, balanced priority)
    data-fetcher.ts        — OQMD/AFLOW live fetching + known materials import (12 categories)
    nlp-engine.ts          — OpenAI NLP analysis
    formula-generator.ts   — Novel formula generation
    ml-predictor.ts        — XGBoost + neural network ensemble for SC prediction
    superconductor-research.ts — Strict SC discovery with multi-step verification pipeline
    synthesis-tracker.ts   — Detailed synthesis process discovery (exact conditions) + broad chemical reactions
  replit_integrations/
    chat/    — OpenAI chat integration
    audio/   — Audio integration
    image/   — Image integration
    batch/   — Batch processing utilities (p-limit, p-retry)

shared/
  schema.ts        — Drizzle schema + Zod types (8 domain tables + chat tables)
  models/chat.ts   — Chat conversations/messages schema
```

## Conventions
- No emoji in UI text
- Font: Open Sans (configured in index.css)
- Use FileText icon (not ScrollText) throughout
- Server runs on port 5000
- Deployment target: autoscale
- Storage pattern: onConflictDoNothing().returning() for inserts; onConflictDoUpdate for upserts
- Cycle timing: 60 seconds between learning cycles
- Phases run with Promise.allSettled for fault-tolerant concurrency
- SC status: NEVER use "breakthrough" or "confirmed" — all are theoretical until lab verification
