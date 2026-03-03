# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
A comprehensive materials science research platform that simulates an AI supercomputer progressively learning everything about materials science — from subatomic particles through molecular modeling to novel material discovery.

## Architecture

### Stack
- **Frontend**: React + Vite + TanStack Query + wouter routing
- **Backend**: Express.js + TypeScript
- **Database**: PostgreSQL via Drizzle ORM
- **UI**: shadcn/ui components + Tailwind CSS + Recharts

### Key Routes
- `/` — Command Center (dashboard with stats, learning pipeline, research log)
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

### Database Tables
- `elements` — 30+ real elements with properties (atomic mass, electronegativity, electron config, etc.)
- `materials` — 15+ materials from NIST, Materials Project, OQMD, AFLOW
- `learning_phases` — 6 phases: Subatomic → Elements → Bonding → Materials → Prediction → Discovery
- `novel_predictions` — 6 AI-predicted novel materials (room-temp superconductors, ultra-hard materials, etc.)
- `research_logs` — Activity log of research events by data source

## Learning Phases
1. **Subatomic Particle Mastery** — 100% complete
2. **Periodic Table & Elemental Properties** — 100% complete (118 elements)
3. **Chemical Bonding & Molecular Structures** — 78% active
4. **Known Materials & Crystal Structures** — 42% active (from NIST/MP/OQMD/AFLOW)
5. **Property Prediction & Modeling** — 8% pending
6. **Novel Material Discovery** — 2% pending

## Data Sources
- **NIST WebBook** — Thermodynamic data, spectroscopic properties
- **Materials Project** — DFT-computed band gaps, formation energies
- **OQMD (Open Quantum Materials Database)** — Crystal structures, stability
- **AFLOW** — High-throughput alloy calculations, 3.5M+ entries

## Features
- Live learning progress visualization with radar chart
- Bohr model electron shell diagrams for any element
- Material property browser with electronic classification (metal/semiconductor/insulator)
- Novel material predictions with confidence scores and synthesis status tracking
- Area chart showing learning progress over time for all 6 phases
- Activity log with data source attribution

## File Structure
```
client/src/
  pages/
    dashboard.tsx       — Command Center
    atomic-explorer.tsx — Element browser + Bohr diagrams
    materials-database.tsx — Materials browser
    novel-discovery.tsx — AI predictions
    research-pipeline.tsx — Pipeline + activity log
  components/
    app-sidebar.tsx     — Navigation sidebar with phase progress

server/
  index.ts     — Express server entry
  routes.ts    — API routes
  storage.ts   — Database abstraction
  db.ts        — Drizzle DB connection
  seed.ts      — Seed data (elements, materials, phases, predictions, logs)

shared/
  schema.ts    — Drizzle schema + Zod types
```
