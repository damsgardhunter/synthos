# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
A comprehensive materials science research platform that simulates an AI supercomputer progressively learning everything about materials science — from subatomic particles through molecular modeling to novel material discovery. Features a real learning engine with live data fetching, NLP pattern recognition, novel formula generation, and real-time WebSocket updates.

## Architecture

### Stack
- **Frontend**: React + Vite + TanStack Query + wouter routing
- **Backend**: Express.js + TypeScript
- **Database**: PostgreSQL via Drizzle ORM
- **UI**: shadcn/ui components + Tailwind CSS + Recharts
- **AI**: OpenAI (via Replit AI Integrations) for NLP analysis and formula generation
- **Real-time**: WebSocket (ws) for live progress updates

### Key Routes
- `/` — Command Center (dashboard with stats, learning pipeline, engine controls, live activity feed)
- `/atoms` — Atomic Explorer (element browser with Bohr model diagrams)
- `/materials` — Materials Database (browse indexed materials from 4 databases)
- `/discovery` — Novel Discovery (AI-predicted new material candidates)
- `/research` — Research Pipeline (full learning phase progress + activity log)

### API Endpoints
- `GET /api/elements` — All learned elements
- `GET /api/elements/:id` — Single element details
- `GET /api/materials?limit&offset` — Paginated materials list
- `GET /api/materials/:id` — Single material details
- `GET /api/learning-phases` — All 6 learning phases with progress
- `GET /api/novel-predictions` — AI-generated novel material predictions
- `GET /api/research-logs?limit` — Research event log
- `GET /api/stats` — Aggregate system statistics
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
- `learning_phases` — 6 phases: Subatomic → Elements → Bonding → Materials → Prediction → Discovery
- `novel_predictions` — AI-predicted novel materials (grows via formula generator)
- `research_logs` — Activity log of research events by data source
- `conversations` — Chat conversations (AI integrations)
- `messages` — Chat messages (AI integrations)

## Learning Engine (server/learning/)
The learning engine orchestrates concurrent AI research across all phases:

- **data-fetcher.ts** — Fetches real materials data from OQMD and AFLOW public APIs
- **nlp-engine.ts** — Uses OpenAI to analyze bonding patterns, predict properties, classify materials
- **formula-generator.ts** — Uses OpenAI to generate novel chemical compositions with predicted properties
- **engine.ts** — Main orchestrator: runs phases concurrently, manages WebSocket broadcasts, tracks progress

### Engine Cycle
1. Phase 4 (Materials) + Phase 3 (Bonding) run in parallel
2. Phase 5 (Prediction) + Phase 6 (Discovery) run after
3. Cycles repeat every 60 seconds while engine is running

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
    dashboard.tsx           — Command Center + Engine Controls
    atomic-explorer.tsx     — Element browser + Bohr diagrams
    materials-database.tsx  — Materials browser
    novel-discovery.tsx     — AI predictions
    research-pipeline.tsx   — Pipeline + activity log
  components/
    app-sidebar.tsx         — Navigation sidebar with phase progress
    engine-controls.tsx     — Start/Stop/Pause engine + live activity feed
  hooks/
    use-websocket.ts        — WebSocket hook for real-time updates

server/
  index.ts       — Express server entry
  routes.ts      — API routes + engine control endpoints
  storage.ts     — Database abstraction (IStorage)
  db.ts          — Drizzle DB connection
  seed.ts        — Seed data
  learning/
    engine.ts          — Main orchestrator + WebSocket server
    data-fetcher.ts    — OQMD/AFLOW live data fetching
    nlp-engine.ts      — OpenAI NLP analysis
    formula-generator.ts — Novel formula generation
  replit_integrations/
    chat/    — OpenAI chat integration
    audio/   — Audio integration
    image/   — Image integration
    batch/   — Batch processing utilities (p-limit, p-retry)

shared/
  schema.ts        — Drizzle schema + Zod types
  models/chat.ts   — Chat conversations/messages schema
```

## Conventions
- No emoji in UI text
- Font: Open Sans (configured in index.css)
- Use FileText icon (not ScrollText) throughout
- Server runs on port 5000
- Deployment target: autoscale
