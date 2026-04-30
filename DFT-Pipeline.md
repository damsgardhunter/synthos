# Quantum Alchemy Engine — Full DFT Pipeline

End-to-end superconductor discovery pipeline from candidate generation through phonon calculation and Tc prediction.

---

## Stage 0: Learning Engine Picks a Formula

**File**: `server/learning/engine.ts`

The active learning loop selects which formulas to investigate — prioritizing high-entropy compositions, known superconductor families, and materials the ML model thinks are promising. It calls `runQuantumEnginePipeline()` for each formula.

---

## Stage 1: Candidate Generation (Multi-Engine Fusion)

**File**: `server/dft/qe-worker.ts`

Five generators run in parallel to create a diverse pool of crystal structure candidates:

1. **Vegard/VCA** — interpolates lattice + positions from binary endpoint structures (AFLOW/MP/known-structures)
2. **AIRSS** (`server/csp/airss-wrapper.ts`) — fully random cells via `buildcell`, pressure-aware MINSEP, Z-sweeps (Z=1,2,3,4,6,8), volume ensemble from Birch-Murnaghan
3. **PyXtal** (`server/csp/pyxtal-wrapper.ts`) — Wyckoff-aware random generation respecting space group symmetry, tiered SG sampling (40% high-sym / 30% med / 20% low / 10% P1)
4. **Cage Seeder** (`server/csp/cage-seeder.ts`) — for hydrides: places H on cage-forming Wyckoff orbits (sodalite, clathrate, hex-clathrate, bcc-hydride) from 141 tagged prototype templates
5. **Mutations** (`server/csp/structure-mutator.ts`) — 8 mutation types on top-3 candidates: lattice strain +/-10%, volume compress/expand, H shuffle, symmetry break, Wyckoff perturbation

Budget is tier-dependent (preview: ~85 candidates, deep: ~10K AIRSS + 1K PyXtal). Tier assignment is automatic based on composition: known compounds get preview, ternary high-P hydrides get deep, binary high-P get standard.

---

## Stage 2: Candidate Funnel (F0 - F8)

**File**: `server/csp/candidate-funnel.ts`

Strict multi-stage filter reducing ~85+ raw candidates to 3-5 DFT-worthy structures:

| Stage | What | Action |
|-------|------|--------|
| **F0** | Parse + normalize | Reject missing positions, bad lattice, NaN coords. Wrap fractional coords to [0,1) |
| **F1** | Geometry hard filter | Pair distance vs pressure-scaled MINSEP (hard reject at <0.5x minsep, soft penalty below). Volume/atom bounds, cell aspect ratio |
| **F2** | Chemistry sanity | Isolated atom penalty, pressure-dependent density check |
| **F3** | Hydride scoring | H-network type scoring (clathrate cage 0.95 -> H2 molecular 0.20), M-H coordination bonus |
| **F4** | Dedup | Cheap fingerprint (composition + volume + pair histogram, cosine < 0.08) |
| **F5** | Fast scoring | Weighted composite: 25% geometry + 20% hydride + 15% source confidence + 15% volume prior + 10% symmetry + 10% diversity + 5% prototype |
| **F6** | CHGNet MLIP | Full relaxation of all candidates (scaled steps: min(500, n_atoms x 10), fmax=0.02 eV/A). Ranks by relaxed energy. Updates candidate lattice with CHGNet-relaxed geometry, but preserves the raw CSP lattice (`preMLIPLatticeA/B/C`). **Drift gate**: if CHGNet volume change > 30% or lattice collapses, the raw CSP geometry is kept instead (MLIP can collapse high-P hydride cages). Tracks `mlipVolumeChangePct` and `mlipDriftRejected` per candidate |
| **F7** | Clustering | Extended fingerprint clustering (50 sorted pair distances) to group structurally similar candidates |
| **F8** | DFT admission | Score: 30% confidence + 25% CHGNet energy rank + 15% diversity + 10% hydride + 10% source diversity + 5% novelty + 5% exploration. Tier-based budget (preview 3-8, deep 50-120) |

---

## Stage 3: Structure Validation & Repair

**File**: `server/dft/qe-worker.ts`

Before DFT, each admitted candidate gets:

1. **Geometry repair** — iterative push-apart for overlapping atoms
2. **Wyckoff site snapping** — aligns atoms to nearby high-symmetry positions
3. **Known structure override** — if `lookupKnownStructure(formula)` matches, use exact literature Wyckoff positions and skip xTB. Only experimentally verified structures are in the database — predicted/speculative ternary hydrides (LaH11Li2, YH9Na2, Li2LaH12, LaH12) are NOT included so the pipeline can discover them independently

---

## Stage 4: xTB Pre-Relaxation (Optional)

**File**: `server/dft/qe-dft-engine.ts`

Semiempirical geometry optimization (~30 min). Skipped for:

- Known structures with validated Wyckoff positions
- High-pressure hydrides (>50 GPa — xTB not parameterized for extreme pressure)
- Constrained mode available for moderate-pressure hydrides (spring constant scales with P)

### xTB Stability Check (Soft Penalty)

xTB formation energy is used as a **soft confidence penalty**, not a hard rejection. xTB is unreliable for high-pressure hydrides and unusual compositions, so it should not kill DFT-worthy candidates:

- < 1.0 eV/atom above refs: stable, no penalty
- 1.0-2.0 eV/atom: mildly unstable, 15% confidence penalty
- 2.0-2.5 eV/atom: unstable, 30% confidence penalty
- > 2.5 eV/atom: very unstable, 50% confidence penalty

**Hard reject only if ALL THREE conditions are met:**
1. Formation energy > 2.5 eV/atom
2. No prototype support (no cage-seeded, known-structure, or high-confidence CSP candidates)
3. Not a hydride AND not high-pressure (> 20 GPa)

This means cage-seeded hydrides, known structures, and high-pressure materials always proceed to DFT regardless of xTB opinion.

---

## Stage 5: Staged DFT Relaxation

**File**: `server/dft/staged-relaxation.ts`

Five sequential stages with gating between each:

| Stage | Type | Time | What |
|-------|------|------|------|
| **1** | Atomic relax (fixed cell) | 10-15 min x N candidates | Tests all admitted candidates. Ranks by **per-atom energy** (not total energy — prevents Z-mismatch where a Z=4 supercell beats Z=1 on total E but is worse per atom). Keeps top K for Stage 2 (tier-dependent: preview=2, standard=5, deep=10, publication=20). Budget: floor(90min / perCandidateMs), min 2. **Z-mismatch guard**: if the winner has more atoms than the primitive cell expects, regenerates positions from known-structure or generateAtomicPositions |
| **2** | vc-relax (variable cell) | 30 min x K candidates | Full cell + position optimization on each Stage 1 winner. **Post-DFT dedup**: detects when multiple starting structures collapse to the same DFT minimum (pair-distance fingerprint, 3% tolerance) — skips duplicates to save phonon compute. Picks the lowest-energy unique result. Iterative lattice rescaling if shift >6-10% (max 6% per step). Mini-EOS pressure correction (5 volume points) |
| **3** | Final SCF | ~15 min | Production-quality electronic structure at relaxed geometry |
| **4** | Gamma-point DFPT phonon | 30 min | Fast dynamical stability screen. 2-attempt retry. Pass: <=3 small imaginary modes. **Pre-phonon diagnostics**: logs atomic positions, SCF convergence, forces, pressure before ph.x. **Force gate**: skips gamma phonon if residual force > 0.10 Ry/bohr (DFPT crash likely). If cost estimate > 4h, skips gamma and defers to full phonon (Stage 5) |
| **5** | Full DFPT phonon grid | up to 48h | Complete phonon dispersion on high-symmetry q-path |

Between stages: charge contamination control (force clean on attempt 4+), quality tier assignment (failed -> partial -> screening_converged -> relaxed -> final_converged -> publication_ready).

### Lattice Rescaling Detail

When the target lattice differs significantly from the initial guess, the pipeline uses iterative rescaling rather than a single large jump:

- **0-10% shift**: single step, 15 min relax
- **10-20% shift**: 2 steps of ~6% each
- **20-30% shift**: 3 steps of ~6% each
- **30-40% shift**: 4 steps of ~6% each
- **40-50% shift**: 5 steps of ~6% each, more time per step

Each step runs a fixed-cell relax at the intermediate lattice before proceeding to the next.

### Mini-EOS Pressure Correction

After vc-relax, the pipeline runs 5 SCF calculations at volumes around the relaxed cell (+/-2%, +/-5%) to fit a Birch-Murnaghan equation of state. This determines the true equilibrium volume at the target pressure, correcting for any pressure mismatch.

---

## Stage 6: Round 2 Iterative Search

**File**: `server/csp/iterative-search.ts`

If Stage 5 SCF converged, the DFT winner seeds a focused Round 2 search:

- **30 aggressive mutations** of the DFT winner (larger perturbations than Round 1)
- **6 volume scans** around DFT lattice (+/-2%, +/-5%, +/-8%)
- **4 pressure scans** (P +/-20%, P +/-50%) if high-pressure material
- **5 symmetry-lowering distortions** (force P1, perturb lattice angles)

Screened through CHGNet using **enthalpy** (H = E + PV), not just energy — critical for high-pressure materials. Selection uses meV/atom thresholds instead of percentages:

- **Promoted** (>= 10 meV/atom enthalpy improvement): top 3 advance
- **Mild improvement** (5-10 meV/atom): top 2 kept if structurally diverse
- **Exploration** (worse enthalpy but novel): 1 candidate kept if cage-type, pressure-scan, or symmetry-lowered source

If no candidates beat the threshold, top 3 are sent anyway for exploration.

---

## Stage 7: Band Structure

**File**: `server/dft/qe-worker.ts`

Post-relaxation electronic structure analysis:

- SCF at relaxed geometry -> high-symmetry k-path band calculation
- Extracts: DOS at Fermi level, flat band score, van Hove singularity count, band crossings
- Workspace isolation (copies `.save/` directory to avoid corrupting relaxation data)

---

## Stage 8: Phonon Calculation

**File**: `server/dft/phonon-calculator.ts`

Two paths depending on available compute:

### Path A: DFPT (from Stage 5, steps 4-5)
- QE `ph.x` on q-grid -> phonon dispersion + DOS
- Full dynamical matrix at each q-point
- Most accurate, but expensive (hours to days)

### Path B: Finite Displacement (fallback)
- Force-constant matrix via xTB Hessian
- Builds dynamical matrix from Hessian -> eigenvalues -> frequencies
- Acoustic sum rule (ASR) correction applied
- Faster but less accurate

### Output
- Frequencies at each q-point
- Phonon dispersion along high-symmetry path
- Phonon DOS
- omega_log (log-average frequency — key input for Tc)
- Stability assessment: imaginary mode count, artifact detection (modes below -2000 cm-1 flagged as xTB force-constant blow-up)

---

## Stage 9: Electron-Phonon Coupling & Eliashberg -> Tc

**File**: `server/physics/eliashberg-pipeline.ts`

Only runs if the structure is phonon-stable (no large imaginary modes):

1. **alpha2F(omega) spectrum** — combines electronic DOS, phonon spectrum, and electron-phonon matrix elements into the spectral function
2. **lambda = integral of alpha2F(omega)/omega** — dimensionless coupling strength (typical: 0.3-2.0 for superconductors)
3. **Allen-Dynes Tc**: `Tc = (omega_log / 1.2) * exp(-1.04 * (1 + lambda) / (lambda - mu_star * (1 + 0.62 * lambda)))`
4. **Eliashberg Tc**: self-consistent linearized gap equation solution (gives higher Tc than Allen-Dynes for strong coupling lambda > 1.0)
5. **Diagnostics**:
   - Gap ratio: 2*Delta(0) / kB*Tc (weak-coupling: 3.53, strong: > 4.0)
   - Isotope effect alpha = -d(ln Tc) / d(ln M)
   - Strong coupling flag (lambda > 0.7)
   - Mode-resolved lambda contributions (which phonon branches drive superconductivity)

---

## Stage 10: Results -> Database -> Next Iteration

**File**: `server/dft/quantum-engine-pipeline.ts`

Results recorded to `quantumEngineDataset` table:

- Material formula and pressure
- lambda (electron-phonon coupling)
- omega_log (log-average phonon frequency)
- Tc (best of Allen-Dynes and Eliashberg)
- DOS at Fermi level
- Phonon spectrum and alpha2F summary
- Formation energy, band gap, metallicity
- SCF convergence status
- Confidence tier (full-dft / xtb / surrogate)
- Wall time

The learning engine uses these results to update its ML models (gradient boost, lambda regressor) and select the next batch of formulas — closing the active learning loop.

---

## Adaptive Learning

**File**: `server/csp/adaptive-learning.ts`

The pipeline tracks which generators and volume multipliers produce successful candidates:

- **Per-family volume learning**: tracks which volume multipliers pass the funnel for each element combination and pressure bin. Bayesian smoothing prevents overfitting.
- **Per-generator weighting**: tracks success rates of AIRSS vs PyXtal vs VCA vs cage seeder vs mutations. Future runs bias generation budgets toward historically successful methods.
- **Cage seeder subtype tracking**: tracks individual cage geometries (sodalite, clathrate, hex_clathrate, bcc_hydride, custom_template) separately so the system learns e.g. "for rare-earth hydrides at 150-250 GPa, sodalite seeds survive best."
- **Persisted to `learning-store.json`** across restarts.

### Quality-Weighted Learning Signals

Learning updates are weighted by the quality of the signal that produced them, preventing over-learning from cheap surrogate results:

| Signal | Weight | When recorded |
|--------|--------|---------------|
| Survived F4 dedup (funnel) | 0.1 | Candidate passes dedup in funnel |
| Selected for DFT (F8) | 0.3 | Candidate admitted by DFT admission |
| DFT-0 converged | 0.5 | SCF converges on this candidate's structure |
| DFT-1 low enthalpy | 1.0 | (future: near convex hull) |
| DFT-2 near hull | 2.0 | (future: thermodynamic stability) |
| Phonon stable | 3.0 | No large imaginary modes in phonon spectrum |
| DFPT e-ph good lambda | 4.0 | Electron-phonon coupling yields finite lambda |

A funnel survival (cheap) contributes 0.1 to the generator's success tally, while a DFPT-validated phonon-stable candidate contributes 3.0 — so one strong result outweighs 30 funnel passes.

---

## Quality Tiers

Each candidate progresses through quality tiers as it passes successive stages:

```
failed -> partial_screening -> screening_converged -> relaxed -> final_converged -> publication_ready
```

Tier determines what downstream analysis is attempted (e.g., publication_ready gets full phonon grid, screening_converged only gets gamma-point phonon).

---

## Key Physics Pathways

**Path A: Full DFT** (high-confidence, known compounds)
```
Formula -> Vegard/CSP (~30 min) -> xTB pre-relax (~30 min) -> Staged relaxation (1-3h) -> DFPT phonon (4-6h) -> Eliashberg (seconds) -> Tc
```

**Path B: xTB + Surrogate** (screening, high pressure, large cells)
```
Formula -> xTB relax (~30 min) -> Surrogate electronic + phonon (~1s) -> Eliashberg -> Tc estimate (lower confidence)
```

**Path C: Pure Surrogate** (fast screening, QE unavailable)
```
Formula -> ML electronic structure -> ML phonon -> Eliashberg -> Tc screening only
```
