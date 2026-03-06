# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform designed to accelerate the discovery of room-temperature superconductors. It integrates AI for natural language processing, novel material generation, and machine learning predictions. The platform aims to revolutionize material discovery by continuously learning and expanding its knowledge base, from subatomic modeling to advanced computational physics and synthesis tracking, with significant market potential across high-tech industries.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases, covering subatomic to multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines a gradient boosting model with an OpenAI gpt-4o-mini neural network ensemble for superconductor scoring, applying strict room-temperature criteria (Tc >= 293K).
- **SC Verification Pipeline**: Categorizes superconductor candidates from theoretical to `requires-verification` to prevent premature claims.
- **Multi-Fidelity Screening**: A 5-stage pipeline with calibrated thresholds for material properties.
- **Novel Insight Detection**: Filters known knowledge, deduplicates content, rejects vague insights, and uses OpenAI for novelty evaluation.
- **Data Confidence Tracking**: Candidates are tagged with `dataConfidence` (high, medium, low).
- **LLM Data Validation**: Cross-validates LLM properties against external APIs, enforcing physical bounds.
- **Physics as Tc Authority**: Grounds all Tc values in Eliashberg/McMillan physics with ambient-pressure caps.
- **Diversity & Deduplication**: Mechanisms to ensure diverse formula generation and prevent redundant work.
- **Evolving Research Strategy**: Balanced family scoring with confidence weighting, implementing an exploit-then-explore policy with dynamic switching.
- **Milestone Detection System**: Identifies research milestones like new family discoveries and pipeline graduations.
- **Tc Plateau Escalation**: Triggers progressive strategies (boundary hunting, inverse design, chemical space expansion) upon stagnation.
- **Stability Filtering**: Incorporates formation energy bounds and a stability modifier in pairing scores.
- **Physical Plausibility Guardrails**: Enforces physical limits and criteria for `roomTempViable` candidates.
- **Stagnation Breaking**: Detects research stagnation and adapts candidate generation strategies via LLM prompts.

### Physics Engine
- **Deterministic Calculations**: All physics calculations are fully deterministic, utilizing a comprehensive elemental database of 96 elements.
- **Advanced Parameter Calculation**: Includes methods for DOS at Fermi level, bandwidth, Hubbard U/W, lambda, phonon frequencies, and metallicity.
- **Specialized Material Handling**: Incorporates specific physics adjustments for superhydrides, High-Entropy Alloys (HEAs), and unconventional superconductor mechanisms.
- **Pairing Susceptibility Optimization**: Optimizes for pairing conditions (DOS, nesting, coupling channels) rather than just raw Tc, evaluating multiple pairing mechanisms.
- **Inverse Design**: Generates materials optimized for pairing susceptibility.
- **Generative Crystal Structures**: Discovers structural variants and novel prototypes via LLM.
- **Enriched ML Features**: Uses approximately 50 diverse physical and structural features for ML prediction, including `pressureGpa` and `optimalPressureGpa`.

### Advanced Quantum Physics Modeling
- Computes phonon dispersion, phonon DOS, Eliashberg spectral function, GW many-body corrections, dynamic spin susceptibility, and Fermi surface nesting.

### Convex Hull Phase Diagram Engine
- Computes energy above hull, decomposition products, hull vertices, and assesses metastability for binary/ternary systems.

### Pressure Modeling Engine
- Calculates volume compression, bulk modulus, pressure derivative, predicts hydrogen uptake, and determines high-pressure stability and pressure-Tc curves.

### Graph Neural Network Surrogate (CGCNN-style)
- Uses prototype-aware graph construction, enhanced node features, and attention-weighted message passing for multi-target predictions (formation energy, phonon stability, Tc, confidence, electron-phonon lambda) with uncertainty estimation.

### Massive Candidate Generator
- Utilizes element substitution, composition interpolation, doped variants, and composition sweeps, with valence sanity filters and periodic table element validation.
- Employs a rapid Gradient Boosting screen to pre-filter candidates.

### Physics-Aware ML Predictor
- Performs multi-target prediction (lambda, DOS(EF), omega_log, hull distance) with transfer priors and uncertainty estimation.
- Incorporates a self-reinforcing loop where predicted physics values feed into the Tc predictor.

### Pattern Mining / Theory Generator
- Extracts and validates quantitative rules from physical features, using rule aging and screening for candidate scoring.

### Structural Mutation Engine
- Assigns structure prototypes, generates distorted lattices, layered structures, vacancy structures, and strain variants.

### Multi-Dimensional Phase Explorer
- Scans composition, pressure, and temperature spaces with adaptive sampling and uncertainty-aware selection to identify optimal material conditions.

### Crystal Prototype Structure Generator
- Generates 10 distinct crystal structure types (e.g., AlB2, Perovskite, A15), producing over 807 structurally-typed candidates with detailed crystallographic information.

### Convex Hull Stability Gate
- Enforces a hard rejection criteria for candidates with hull distance > 0.1 eV/atom, with additional metastability assessment for borderline cases.

### Discovery Score
- A composite metric (`0.4 × normalizedTc + 0.3 × noveltyScore + 0.2 × stabilityScore + 0.1 × synthesisFeasibility`) that includes bonuses for chemical novelty.
- Prioritizes higher-scoring candidates for DFT enrichment.

### Active Learning Loop
- Uses uncertainty-driven DFT selection with analytical DFT fallback when external data is unavailable.
- Budget: 20 DFT runs per cycle, triggered every 15 cycles.
- GNN retrain triggers: 15+ enriched candidates since last retrain, OR avg uncertainty > 0.3, OR first retrain with any DFT success.
- Retrains the GNN surrogate with expanded datasets and monitors convergence.
- **Real DFT tier**: Top 3 candidates per AL cycle are computed with GFN2-xTB (xtb v6.7.1) for real quantum-mechanical total energy, HOMO-LUMO gap, formation energy, and metallicity assessment.

### Real DFT Backend (GFN2-xTB)
- **Engine**: xtb v6.7.1 (Grimme group) at `server/dft/xtb-dist/bin/xtb`.
- **Method**: GFN2-xTB — semi-empirical density functional tight binding, providing real quantum-mechanical energies.
- **Outputs**: Total energy (Hartree), HOMO-LUMO gap (eV), formation energy (eV/atom), metallicity assessment, dipole moment, Mulliken charges.
- **Integration**: Runs on top candidates during active learning cycles; results stored in candidate DB with `dataConfidence: "high"`.
- **Performance**: ~5 seconds per calculation for typical 3-8 atom clusters.
- **Structure generation**: Automatic cluster geometry from formula using covalent radii and coordination.
- **Formation energy**: Computed relative to single-element reference calculations.
- **Cache**: In-memory LRU cache (200 entries) to avoid redundant calculations.

### Analytical Physics Estimators (coverage ~1.00)
- **Debye temperature**: θD ≈ 41.6 * sqrt(B / ρ) when elemental data unavailable.
- **Bulk modulus**: Estimated from melting point (B ≈ 0.07 * T_melt) when elemental data missing.
- **Density**: Computed from atomic masses and radii with BCC packing assumption.
- **DOS at Fermi level**: Estimated from valence electron count with transition metal boost.
- **Average phonon frequency**: Derived from Debye temperature and average atomic mass.
- **Atomic packing fraction**: Geometric estimate from atomic radii.
- Coverage metric tracks how many of 11 properties have valid estimates.

### Autonomous Discovery Loop
- A massive generation pipeline generating 500-2000 candidates per cycle, undergoing multi-stage filtering (GB pre-screen, pattern mining, physics ML pre-filter) before a full pipeline processing.
- **Tiered Acceptance System**: Tier 1 (Tc>70K, λ>1.2, hull<0.10) → high confidence; Tier 2 (Tc>25K, λ>0.5, hull<0.20) → medium; Tier 3 (Tc>10K, λ>0.3) → low. Tier 2/3 bypass stability gate.
- Pass rate ~46% after tiered acceptance fix.

### Semantic Insight Deduplication
- Uses OpenAI text-embedding-3-small embeddings with cosine similarity (threshold 0.85) to reject semantically duplicate insights before LLM novelty scoring.
- Category quotas (2 per 30-min window) across 10 categories: novel-correlation, new-mechanism, cross-domain, computational-discovery, design-principle, phonon-softening, fermi-nesting, charge-transfer, structural-motif, electron-density.
- NLP prompts instruct diverse topic rotation: phonon softening, Fermi surface nesting, charge transfer layers, structural motifs, electron density redistribution, spin-orbit coupling, pressure-dependent phonon hardening.

### Bayesian Family Strategy Scoring
- Uses Bayesian shrinkage (prior_count=5) to prevent single-candidate families from dominating. Includes exploration bonus for under-explored families (<10 candidates).

### NLP Engine
- Generates cross-property correlation insights and superconductor correlation analysis, rejecting pure statistical summaries.

### Alive Engine
- Provides real-time feedback via WebSocket, adjusts tempo based on research progress, and maintains research memory with dynamic status messages.

### Infrastructure
- Includes rate limiting, an in-memory cache, optimized database indexes, API pagination, ML calibration with confidence bands, and cross-validation with external APIs.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP, formula generation, ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.