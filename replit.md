# MatSci-∞ Supercomputer — Materials Science AI Platform

## Overview
MatSci-∞ is an AI-powered supercomputer platform dedicated to accelerating the discovery of room-temperature superconductors (Tc >= 293K). It integrates advanced AI for natural language processing, novel material generation, and machine learning predictions to revolutionize material discovery. The platform aims to continuously learn, expand its knowledge base, and offers significant market potential across high-tech industries.

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
- **AI-Driven Learning Engine**: Orchestrates 13 distinct learning phases, including multi-fidelity screening and novel synthesis reasoning.
- **ML Prediction Engine**: Combines gradient boosting and OpenAI gpt-4o-mini for superconductor scoring.
- **Physics Engine**: Performs deterministic calculations using a 96-element database, including advanced parameter calculations.
- **Topological Superconductor Detection**: Estimates Z2 invariant, Chern number, and Majorana feasibility.
- **Quantum Pairing Mechanism Simulator**: Analyzes 7 pairing channels.
- **Materials Genome Representation**: Utilizes a 256-dimensional latent vector for similarity search and inverse design.
- **Autonomous Theory Discovery Engine**: Uses symbolic regression to uncover mathematical relationships.
- **Self-Improving Physics Models**: Parameterized physics equations optimized via Bayesian methods.
- **Generator Resource Manager**: Centralized allocation and adaptive rebalancing for 5 candidate generators.
- **Synthesis Pathway Modeling & Optimizer**: Simulates multi-step reaction pathways and optimizes synthesis conditions.
- **Experimental Validation Planner**: Ranks candidates, generates synthesis instructions, and suggests characterization methods.
- **Autonomous Hypothesis Engine**: AI for generating, testing, and ranking superconductivity theories.
- **Discovery Memory System**: Pattern-based learning for high Tc discoveries.
- **Graph Neural Network Surrogate**: Primary ML predictor for formation energy, phonon stability, Tc, and electron-phonon lambda with uncertainty estimation.
- **Reinforcement Learning Chemical Space Agent**: REINFORCE policy gradient agent learning optimal elements, stoichiometries, and structures.
- **Bayesian Optimization for Chemical Space Search**: Gaussian Process surrogate with mixed acquisition functions.
- **Crystal Diffusion Generator**: Generates novel crystal structures by refining random atomic positions.
- **Active Learning Loop**: Uncertainty-driven DFT selection and GNN surrogate retraining.
- **Real DFT Backend (GFN2-xTB)**: Integrates xtb for quantum-mechanical properties, geometry optimization, and phonon calculations.
- **Full DFT Backend (Quantum ESPRESSO 7.2)**: Executes pw.x and ph.x calculations for top candidates via async job queue, with multi-layer reliability checks and pre-filters.
- **Inverse Design Optimizer Engine**: 5-layer architecture for goal-driven candidate generation and closed-loop optimization.
- **Self-Improving Design Lab**: Strategy-level optimization that evolves design architectures across 8 concurrent strategy types.
- **Symbolic Physics Discovery Layer**: 12-component subsystem for automated physics equation discovery.
- **Causal Physics Discovery Layer**: 11-component subsystem for causal mechanism discovery in superconductivity.
- **Crystal Distribution Database**: Learned crystallographic distributions from ~500k structures.
- **Multi-Task GNN Surrogate**: Extended GNN predicting 18 properties simultaneously.
- **Autonomous Discovery Loop**: Massive generation pipeline with multi-stage filtering and tiered acceptance.
- **Semantic Insight Deduplication**: Uses OpenAI text-embedding-3-small for semantic deduplication.

### Physics Filtering Rules
- **Phonon stability**: xTB screening uses relaxed thresholds (maxImagModes=10, lowestFreq=-1500 cm-1) to avoid premature rejection of unrelaxed structures. Extreme artifacts (< -5000 cm-1) still rejected.
- **xTB pre-optimization**: Hessian calculations always run on at least crude-optimized structures (inline --opt crude fallback if no cached optimization exists).
- **Prototype chemistry compatibility**: `isPrototypeChemicallyCompatible()` enforces HexBoride requires B, Perovskite/NaCl/Pyrite require anion elements, A15 requires transition metals, Clathrate/H-named prototypes require H, Heusler/ThCr2Si2 require TMs.
- **PBC-aware distance check**: `validateGeometry` uses minimum-image convention (fractional coordinate wrapping) for correct periodic boundary distance calculations.
- **Dynamic DFT cutoffs**: ecutwfc adapts per-element (O:70, F:80, N:60 Ry etc.) instead of fixed 45 Ry; ecutrho = 8x ecutwfc.
- **Volume scaling**: `validateAndFixStructure` uses cbrt-based scaling with bidirectional correction (scale down when volume overshoots after distance fix). Volume ratio bounds: 0.5-2.0.
- **Radius compatibility**: `checkRadiusCompatibility()` rejects non-H element pairs with covalent radius ratio > 3.0 before structure generation.
- **Hydride stoichiometry**: Metal-rich hydrides (H/metal < 1.0) rejected as unphysical in QE worker validation.
- **Noble gas rejection**: `isValidFormula` rejects compositions containing noble gases (He, Ne, Ar, Kr, Xe, Rn).
- **Formation energy**: Hard stop for Ef > 1.0 eV/atom or Ef < -5.0 eV/atom.
- **Hull distance**: Hard reject > 0.50 eV/atom.
- **Chemistry grammar validation**: Pre-filters compositions based on various chemical constraints.
- **Surrogate pre-filter pipeline**: Screens candidates with GB model before expensive physics evaluation; failed feature extraction rejects (not passes).
- **Deterministic phonon estimates**: Analytical phonon fallback uses formula-hash seeding for reproducible frequencies.
- **RL agent templates**: STOICH_TEMPLATES includes superhydride patterns (AH6/AH9/AH10/AH12, ABH8); 30+ element pair priors.
- **WebSocket reconnect**: Exponential backoff (3s→30s max), resets on successful connection.
- **Fluorite prototype**: Corrected to A4B8 (Fm-3m symmetry, 8 anion sites).
- **GB model**: Tc cap raised to 350K; 54 features including bandGap, formationEnergy, stability, crystalSymmetry; score tier for >293K.
- **QE timeout**: 300s (was 120s) for higher-cutoff calculations.
- **Hydride phonon Debye**: Scaled by H-ratio (1200 cm⁻¹ for hRatio>0.7, 800 for >0.5).
- **Hydride cage positions**: Supports up to 12 H atoms per metal (was 10).
- **Defect engine**: Boltzmann-scaled concentrations for all 4 defect types (vacancy, interstitial, antisite, dopant).
- **COHESIVE_ENERGIES_EV**: 60+ elements including all 4d/5d TMs, lanthanides, actinides.
- **ELEMENT_DATA (QE)**: 77+ elements including Br, Tc, Cd, lanthanides, Th, U, Pa.
- **RL agent**: Weight clamping floor -0.5 (allows demotion); physics bonus factor 0.08; hydride anionGroups includes nonmetals [8,9]; expanded ELEMENT_GROUPS (Tc, Cd, Hg, Eu, Tb, Ho, Tm).
- **Phonon frequency**: `logAvgFreqRatio` uses deterministic formula hash (not Math.random()).
- **Crystal prototypes**: Added 1111-Type (LaFeAsO/P4/nmm) and K2NiF4-214 (La2CuO4/I4/mmm).
- **GB crystalSymmetry**: String-to-numeric mapping (cubic=7, hexagonal=6, tetragonal=5, etc.).
- **Synthesis display**: Units corrected to K (temperature) and GPa (pressure).
- **Multi-fidelity pipeline**: catch block now logs errors with candidate ID.
- **Validation route**: Auto-promotes candidate to stage 5 on positive experimental result.
- **STATUS_COLORS**: Expanded with experimentally-tested, dft-verified, failed, rejected.
- **Engine stability**: Removed blind catch insertion in insertCandidateWithStabilityCheck; isRunningCycle guard on start/resume; Promise.allSettled results logged.
- **Physics NaN guards**: lambda denom check, isFinite guard, integratedLambda minimum 1e-8; tight-binding catch logged.
- **GB tree minSamples**: Increased from 5 to 8 for better regularization.
- **Pipeline stage fix**: Failed Stage 4 correctly reports finalStage=4 (was 3); candidate limit 4→8; synthesizability/synthesisNotes now saved.
- **QE worker**: Hg added to ELEMENT_DATA+ATOMIC_VOLUMES; COVALENT_R expanded (As,Br,Rb,Cs,Hg,Pa); hydride hPerMetal threshold 1.0→0.5.
- **Crystal prototypes**: Added YBCO-123 (Pmmm orthorhombic), FeSe-11 (P4/nmm tetragonal), and NaCl-B1 (Fm-3m rock-salt for nitrides/carbides).
- **Routes validation**: NaN-safe query param validation on limit/offset/stage/targetTc.
- **RL agent**: Upper weight clamp 5.0 added (was unbounded); tcBonus uses safeTc.
- **Allen-Dynes correction**: Strong-coupling (lambda>1.5) now uses f1*f2 factors instead of simplified McMillan.
- **mcMillanHopfieldEta**: Added values for C (6.0), N (5.0), O (3.5) in elemental-data.
- **Tight-binding**: H added to KNOWN_SK_ELEMENTS; Fermi level uses sorted eigenvalue method.
- **COVALENT_R alignment**: Worker and engine both default to 1.4 A for unknown elements; Tc, I, Kr added to worker.
- **NEIGHBOR_MAP**: Added Be, Hg, Cd, Tc, F, Cl, Br, I, Th, U to defect-engine.
- **Inverse generator**: Clathrate hydrides now generated only when maxPressure >= 50 GPa (was < 50).
- **Engine catch blocks**: 65+ previously silent catch blocks now log errors across entire engine.ts.
- **Supercon dataset**: Removed duplicate entries (ScH9, NbC, SrTiO3); 550 entries total.
- **Auto-promote catch**: Routes validation auto-promote now logs errors on failure.
- **Round 8 comprehensive fixes**:
  - ml-predictor.ts: electronDensityEstimate clamped to [0,1]; phononSpectralWidth NaN guard with isFinite fallback.
  - physics-engine.ts: Stoner enhancement denominator guard (stonerProduct<0.95, |denom|>1e-6); FAMILY_TC_CAPS raised (Nitrides 65K/110K, Borides 55K/150K, Carbides 45K/100K).
  - bayesian-optimizer.ts: noiseVariance 0.01→0.1 for better GP conditioning; Cholesky fallback 1e-6→1e-4; thompsonSample Math.log(0) guard; acquisitionEI isFinite check.
  - symbolic-regression.ts: R² clamped to [-1,1]; evaluateNode NaN propagation guard (early return 1e6).
  - supercon-dataset.ts: Added lambda and pressureGPa fields; lambda values for 15+ key superconductors (Nb=0.82, Pb=1.55, MgB2=0.87, Nb3Sn=1.80, H3S=2.19, LaH10=2.2, etc.); pressure for 8 hydrides; removed duplicate Bi2Sr2Cu1O6.
  - crystal-prototypes.ts: Skutterudite stoichiometryRatio fixed [1,3]→[4,12]; new Heusler-L21 prototype (Fm-3m, 16-atom conventional cell, 8c/4a/4b Wyckoff, A2BC).
  - inverse-generator.ts: Added isValidFormula import and filtering before returning candidates.
  - engine.ts: Phase 11 needsPrediction cap 3→5; Phase 12 unscreened cap 6→8; isValidFormula check on pressure pathway insertions.
  - qe-worker.ts: Removed duplicate Cd entry; ecutwfcBoost in retry configs (0/+10/+20/+20 Ry); generateSCFInputWithParams accepts boost.
  - next-gen-pipeline.ts: 7 empty catch blocks now log errors with module prefix.
  - active-learning.ts: 3 empty catch blocks now log errors with module prefix.
- **Round 7 comprehensive fixes**:
  - ml-predictor.ts: Guards for totalAtoms=0 division, enSpread empty array, chargeTransfer NaN, massValues spread.
  - fermi-surface-engine.ts: Guards for nestingVectors.length=0, kxy+kz variations near zero.
  - pairing-mechanisms.ts: Allen-Dynes denominator guard (>1e-6), empty catch fixed, isFinite check.
  - physics-engine.ts: ELEMENT_BANDWIDTH added for H(15), C(12), N(11), O(10), P(8.5), S(9), Se(7.5); isFinite on kappa+anisotropyRatio.
  - topology-engine.ts: ATOMIC_NUMBERS expanded to full 118 elements (was ~70).
  - rl-agent.ts: Bounds checking on all updatePolicy/replayBatch array accesses; Br, I added to nonmetal; new actinide group (Th, U, Np, Pu).
  - pressure-engine.ts: Removed unused a0est/compressionRatio; element-class bulk modulus defaults (H=1, noble=2, alkali=12, etc.) replacing flat 100.
  - qe-worker.ts: ATOMIC_VOLUMES expanded with lanthanides (Pm-Lu), Tc, noble gases, actinides (Pa, Np, Pu).
  - elemental-data.ts: mcMillanHopfieldEta added for Si(1.0), P(1.5), S(2.0), Cl(0.5), Ge(1.2), As(1.8), Se(2.5), Br(0.8), Sb(1.5), Te(2.2), I(1.0).
  - engine.ts: Phase 10 candidates 5→8, hydride scan 3→5, symbolic regression generations 25→40.

## External Dependencies
- **OpenAI**: For gpt-4o-mini (NLP,  ML refinement, knowledge base sourcing).
- **PostgreSQL**: For persistent data storage.
- **OQMD API**: For live materials data fetching.
- **NIST WebBook**: For thermodynamic and spectroscopic data.
- **Materials Project**: For DFT-computed band gaps and formation energies.
- **AFLOW REST API**: For crystal structure and electronic property cross-validation.