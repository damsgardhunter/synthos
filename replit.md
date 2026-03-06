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

### Graph Neural Network Surrogate (CGCNN/MEGNet/M3GNet-style) — PRIMARY PREDICTOR
- **Primary ML predictor** with 0.6 weight in ensemble scoring (GB reduced to 0.3 weight, 0.1 structural novelty).
- Uses prototype-aware graph construction, enhanced 16-dimensional node features (including M3GNet-inspired stress/force/SOC descriptors), 7-dimensional edge features (with bond-angle encoding), and attention-weighted message passing with 3-body interaction layers for multi-target predictions (formation energy, phonon stability, Tc, confidence, electron-phonon lambda) with uncertainty estimation.
- MEGNet-style multi-body interactions: 3-body angle features between bonded triplets with angular weighting.
- GB retained as fast pre-filter (Tc < 5K rejection gate) but no longer drives ensemble scores.

### Massive Candidate Generator
- Utilizes element substitution, composition interpolation, doped variants, and composition sweeps, with valence sanity filters and periodic table element validation.
- **Fractional doping generator**: `generateFractionalDopedVariants()` creates fine-grained doped variants with fractions [0.05..0.25] across 19 SC-relevant dopants. Seeds include known SC materials (MgB2, LaH10, NbN, etc.). Generates ~150 variants per cycle.
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
- **Integration**: Runs on EVERY candidate in the main screening pipeline via `runXTBEnrichment()` in `dft-feature-resolver.ts`. Falls back to analytical estimates only when xTB fails or formula is unsupported.
- **DFT Source Type**: `"dft-xtb"` — tracked as a distinct source in DFTResolvedFeatures alongside `"analytical"`, `"mp"`, `"oqmd"`, `"aflow"`, `"nist"`.
- **Performance**: ~5 seconds per calculation for typical 3-20 atom structures. ~85% success rate.
- **Structure generation**: 20 crystallographic prototypes (A15, AlB2, NaCl, Perovskite, ThCr2Si2, Heusler, BCC, FCC, Layered, Kagome, HexBoride, MX2, Anti-perovskite, CsCl, Cu2Mg-Laves, Fluorite, Cr3Si, Ni3Sn, Fe3C, Spinel) with real fractional coordinates and lattice parameters. Approximate prototype matching (ratio score < 0.5) for imperfect stoichiometries. Generic cluster fallback for fully unmatched cases. Structures capped at 20 atoms with proportional scaling.
- **Geometry optimization**: Runs `xtb --gfn 2 --opt tight` before single-point calculations (30s timeout). Optimized geometry used for all subsequent energy and phonon calculations. Cached in `optimizedStructureCache`.
- **Finite-displacement phonon calculator**: For structures ≤8 atoms, displaces each atom ±0.01 Å in x/y/z, runs xTB on each displaced structure, builds force constant matrix from finite differences (6N+1 calculations). Computes dynamical matrix D(q) at arbitrary q-points along high-symmetry paths. Produces phonon dispersion, phonon DOS, ω_log, and full Brillouin zone stability assessment. Uses optimized geometry when available.
- **Phonon stability check**: For 9-12 atom structures, falls back to xTB `--hess` Hessian calculation. IMAG_THRESHOLD = -2000 cm⁻¹ (relaxed from -50 to accommodate xTB tolerance). Mild imaginary modes (-50 to -2000 cm⁻¹) accepted without penalty; severe modes (< -2000 cm⁻¹) penalized (0.08 per mode above 1, max 0.2). Results cached (100 entries).
- **Formation energy**: Computed relative to molecular/dimer reference calculations using MOLECULAR_BOND_LENGTHS (H: 0.74 Å, N: 1.10, O: 1.21 Å) for accurate reference energies. Sanity guard: |Ef| > 15 eV/atom is discarded as unphysical. Uses actual DFT atom count (not formula count) to handle scaled structures correctly.
- **Cache**: In-memory LRU caches for DFT results (200 entries) and elemental reference energies.
- **Stats API**: `getXTBStats()` exposes runs, successes, cacheSize, refElements via `/api/dft-status`.

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
- **Exploration Probability**: 30% of cycles randomly explore an underexplored family instead of the strategy-recommended one. Pool: Pnictides, Chalcogenides, Cuprates, Hydrides, Kagome, Sulfides, Intermetallics, Alloys, Oxides, Nitrides.
- **Unconventional Seeds**: scBiasedSeeds reduced to 4 random per cycle; 8 unconventional seeds (FeSe, FeAs, BaFeAs, KVSb, LaNiO, etc.) injected each cycle.
- **Pressure Exploration**: Non-hydride metallic candidates with Tc>20K scanned at 5-50 GPa; pressure-enhanced Tc recorded with optimal pressure.
- Pass rate ~46% after tiered acceptance fix.

### Semantic Insight Deduplication
- Uses OpenAI text-embedding-3-small embeddings with cosine similarity (threshold 0.85) to reject semantically duplicate insights before LLM novelty scoring.
- Category quotas (2 per 30-min window) across 10 categories: novel-correlation, new-mechanism, cross-domain, computational-discovery, design-principle, phonon-softening, fermi-nesting, charge-transfer, structural-motif, electron-density.
- NLP prompts instruct diverse topic rotation: phonon softening, Fermi surface nesting, charge transfer layers, structural motifs, electron density redistribution, spin-orbit coupling, pressure-dependent phonon hardening.

### Kagome Metals Family
- Generates AV3X5, AV3X4, A2V3X5 stoichiometries with alkali/alkaline-earth A-sites and pnictogen/metalloid X-sites.
- Family filter checks frustrated lattice metal count, pnictogen sublattice, DOS at Fermi, van Hove singularity proximity, 2D character, and electron-phonon coupling.
- Seeds: KV3Sb5, CsV3Sb5, RbV3Sb5 and variants.

### Layered Structure Generators
- **Layered Chalcogenides**: MX2/MX3/M2X3 (NbSe2, TaS2, MoSe2 pattern) with intercalated variants (Li/Na/K/Ca/Sr/Cu into MX2 hosts).
- **Layered Pnictides**: Iron pnictide patterns — RE-TM-Pn-O (1111-type), AE-TM2-Pn2 (122-type), plus binary TM-Pn.
- **Intercalated Layered**: MX2 intercalation compounds, graphite intercalation compounds (AC6/AC8/AC12), intercalated oxide layers.

### Mixed-Mechanism Systems
- Targets Fe/Ni/Co/Cu in layered compounds for phonon + magnetic fluctuation superconductivity.
- Generates FeAs-based, FeSe-based, NiO-based (infinite-layer nickelates), CuO-based (cuprates), and combinatorial TM-chalcogenide/pnictide.
- Filter checks for magnetic TM presence, λ ≥ 0.3, magnetic fluctuation proximity, and layered character.

### Tight-Binding Electronic Structure Model
- **Slater-Koster tight-binding Hamiltonian**: Builds H(k) matrix from tabulated hopping parameters for s, p, d orbitals based on elemental properties.
- **Band structure solver**: Solves H(k) along high-symmetry k-paths (Γ-X-M-Γ for cubic, Γ-K-M-Γ for hexagonal, Γ-H-P-Γ-N for BCC).
- **Wannier projection**: Extracts orbital-resolved band character, effective hopping parameters, and Wannier spread/localization metrics.
- **Band topology detection**: Identifies flat bands (bandwidth < 0.1 eV), van Hove singularities (∂²E/∂k² → 0), Dirac crossings (linear dispersion), and topological band inversions (parity eigenvalue check).
- **TB DOS**: Histogram-based DOS from band eigenvalues with Gaussian broadening. Blended into physics-engine DOS estimate (60% heuristic + 40% TB-derived).
- **Integration**: Called in `computeElectronicStructure()` to refine flatBandIndicator, vanHoveProximity, topologicalBandScore, and densityOfStatesAtFermi. Results cached (500 entries).
- Files: `server/learning/tight-binding.ts`, `server/learning/physics-engine.ts`

### Flat-Band Detection
- Computes DOS(EF)/avgDOS ratio as flat-band indicator.
- Ratio > 3 = strong flat-band (indicator ≥ 0.7), > 2 = moderate (≥ 0.3).
- Cuprates get 0.8 floor, Kagome 0.7, TM with high bandFlatness 0.5.
- Flat-band indicator > 0.5 boosts lambda by up to 40% (flatBoost = 1 + (fbi - 0.5) * 0.8).
- **Flat band logging**: Emits log event when flatBandIndicator > 0.3 with bandFlatness, DOS(EF), and mottProximity details.

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