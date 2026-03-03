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
- `materials` — 15+ materials from NIST, Materials Project, OQMD, AFLOW (grows via live fetching)
- `learning_phases` — 9 phases: Subatomic -> Elements -> Bonding -> Materials -> Prediction -> Discovery -> SC Research -> Synthesis -> Reactions
- `novel_predictions` — AI-predicted novel materials (grows via formula generator)
- `research_logs` — Activity log of research events by data source
- `synthesis_processes` — How materials are created (method, steps, precursors, equipment, conditions, safety)
- `chemical_reactions` — Lab reactions with equations, energetics, mechanisms, SC relevance scores
- `superconductor_candidates` — ML-scored SC candidates with XGBoost/NN/ensemble scores, Meissner/Cooper pair/coherence data
- `conversations` — Chat conversations (AI integrations)
- `messages` — Chat messages (AI integrations)

## Learning Engine (server/learning/)
The learning engine orchestrates concurrent AI research across 9 phases:

- **data-fetcher.ts** — Fetches real materials data from OQMD and AFLOW public APIs
- **nlp-engine.ts** — Uses OpenAI to analyze bonding patterns, predict properties, classify materials
- **formula-generator.ts** — Uses OpenAI to generate novel chemical compositions with predicted properties
- **ml-predictor.ts** — XGBoost feature extraction (18 physics features) + OpenAI neural network refinement for superconductor scoring. Ensemble = XGB*0.4 + NN*0.6
- **superconductor-research.ts** — Dedicated SC research: evaluates known materials, generates novel room-temp SC designs with synthesis pathways
- **synthesis-tracker.ts** — Discovers how materials are made (lab processes, precursors, equipment) and catalogs chemical reactions
- **engine.ts** — Main orchestrator: runs phases concurrently, manages WebSocket broadcasts, tracks progress

### Engine Cycle
1. Phase 4 (Materials) + Phase 3 (Bonding) + Phase 9 (Reactions) run in parallel
2. Phase 5 (Prediction) + Phase 8 (Synthesis) run next
3. Phase 6 (Discovery) + Phase 7 (SC Research) run last
4. Cycles repeat every 60 seconds while engine is running

### ML Prediction Engine
- XGBoost layer: Pure TypeScript gradient boosting with 18 physics-informed features (electronegativity, Cooper pair strength, phonon coupling, Meissner potential, d-wave symmetry, layered structure, etc.)
- Neural network layer: OpenAI gpt-4o-mini refines XGBoost predictions with deep physics reasoning
- Ensemble: 40% XGBoost + 60% Neural Network weighted scoring
- Room-temp criteria: Tc >= 293K, pressure < 50 GPa, Meissner effect, zero resistance, Cooper pair formation

### OpenAI Configuration
- Model: gpt-4o-mini for all NLP tasks
- Uses Replit AI Integrations (env vars: AI_INTEGRATIONS_OPENAI_BASE_URL, AI_INTEGRATIONS_OPENAI_API_KEY)
- Batch processing with p-limit (concurrency: 2) and p-retry (retries: 5)

## Data Sources
- **NIST WebBook** — Thermodynamic data, spectroscopic properties
- **Materials Project** — DFT-computed band gaps, formation energies
- **OQMD** — `http://oqmd.org/oqmdapi/formationenergy` (live fetching, no auth)
- **AFLOW** — `http://aflow.org/API/aflowlib/` (live fetching, no auth)

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
    engine.ts              — Main orchestrator + WebSocket server (9 phases)
    data-fetcher.ts        — OQMD/AFLOW live data fetching
    nlp-engine.ts          — OpenAI NLP analysis
    formula-generator.ts   — Novel formula generation
    ml-predictor.ts        — XGBoost + neural network ensemble for SC prediction
    superconductor-research.ts — Room-temp SC discovery + novel design generation
    synthesis-tracker.ts   — Lab synthesis process discovery + chemical reaction cataloguing
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
