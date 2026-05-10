# Quantum Alchemy Engine — Full DFT Pipeline

End-to-end superconductor discovery pipeline from candidate generation through phonon calculation and Tc prediction. The system discovers crystal structures independently — no literature lattice overrides, no forced positions.

---

## Stage 0: Learning Engine Picks a Formula

**File**: `server/learning/engine.ts`

The active learning loop selects which formulas to investigate — prioritizing high-entropy compositions, known superconductor families, and materials the ML model thinks are promising. Uses a multi-objective learning score (not raw max Tc) that weights stability, phonon quality, and metallicity alongside Tc predictions.

---

## Stage 1: Candidate Generation (Multi-Engine Fusion)

**File**: `server/dft/qe-worker.ts`

Six sources run in parallel to create a diverse pool of crystal structure candidates:

1. **Vegard/VCA** — interpolates lattice + positions from binary endpoint structures (AFLOW/MP/known-structures)
2. **AIRSS** (`server/csp/airss-wrapper.ts`) — fully random cells via `buildcell`, pressure-aware MINSEP, Z-sweeps (Z=1,2,3,4,6,8), volume ensemble from Birch-Murnaghan
3. **PyXtal** (`server/csp/pyxtal-wrapper.ts`) — Wyckoff-aware random generation respecting space group symmetry, tiered SG sampling (40% high-sym / 30% med / 20% low / 10% P1)
4. **Cage Seeder** (`server/csp/cage-seeder.ts`) — for hydrides: 20-30 candidates at 3-5 pressure-compressed volumes each. Proper ternary A/M site placement: guest metals (Li, Na, K) go to interstitial "A" sites, host metals (La, Y, Ca) go to cage-center "M" sites. Uses Wyckoff orbits from 141 tagged prototype templates (sodalite, clathrate, hex-clathrate, bcc-hydride)
5. **Mutations** (`server/csp/structure-mutator.ts`) — 8 mutation types on top-3 candidates: lattice strain +/-10%, volume compress/expand, H shuffle, symmetry break, Wyckoff perturbation
6. **DFT Structure Cache** — previous best DFT-optimized structures per formula, injected at confidence=0.99. Stored in `/tmp/qe_calculations/structure_cache/{formula}.json`. Only overwrites if new structure has lower force than cached. Gives the pipeline a head start instead of rediscovering the same geometry from scratch.

Budget is tier-dependent (preview: ~85 candidates, deep: ~10K AIRSS + 1K PyXtal). Atom count cap per candidate: preview 30, standard 40, deep 50, publication 60. Oversized candidates are backfilled from the next best.

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
| **F6** | CHGNet MLIP | Full relaxation of all candidates. Drift gate: if volume change > 30% or lattice collapses, raw CSP geometry is preserved |
| **F7** | Clustering | Extended fingerprint clustering (50 sorted pair distances) |
| **F8** | DFT admission | Tier-based budget (preview 3-8, deep 50-120) |

---

## Stage 3: Structure Validation & Repair

**File**: `server/dft/qe-worker.ts`

Before DFT, each admitted candidate gets:

1. **Geometry repair** — iterative push-apart for overlapping atoms
2. **Wyckoff site snapping** — aligns atoms to nearby high-symmetry positions
3. **Z-mismatch guard** — if candidate has more atoms than the primitive cell expects, regenerates positions
4. **xTB pre-relaxation** (optional) — semiempirical optimization, skipped for high-P hydrides

No literature lattice overrides. The pipeline discovers the correct lattice independently.

---

## Stage 4: Staged DFT Relaxation

**File**: `server/dft/staged-relaxation.ts` + `server/dft/qe-worker.ts`

### Stage 1: Fixed-Cell Atomic Relax

Tests all admitted candidates with fixed cell (BFGS optimizer). Ranks by **per-atom energy**. Keeps top K for the unified vc-relax (preview=2, standard=5, deep=10).

Stage 2 (BFGS vc-relax) has been **removed** — BFGS with simultaneous cell changes causes force regression (e.g. H3S force 0.0015 → 0.127) because the Hessian becomes invalid as the cell changes shape. The unified vc-relax handles cell optimization properly.

### Unified vc-relax (damped dynamics with tight SCF)

**The core structure optimization.** Single calculation where cell and positions converge together:

- `ion_dynamics = 'damp'` — damped dynamics, never diverges
- `cell_dynamics = 'damp-w'` — Wentzcovitch cell dynamics
- `conv_thr = 1e-7` — production-quality forces at EVERY ionic step
- `disk_io = 'high'` — writes collected wavefunctions to .save/ for ph.x and bands
- `nstep = 600` for high-P hydrides, 400 default — enough steps for convergence
- No separate phases — forces during optimization ARE the true forces

This eliminates the "force gap" where old loose-SCF vc-relax reported force=0.001 but production SCF saw force=0.25. Now what you see is what you get.

Timeout: 3h (high-P hydrides), 60 min (magnetic), 30 min (default).

### Refinement vc-relax Loop

After the initial vc-relax, if residual force > 0.001 Ry/bohr (publication threshold), the pipeline loops up to **6 refinement passes**. Each pass:

1. Restarts from the previous pass's final geometry with **zeroed velocities**
2. Eliminates oscillation inherited from the bad starting structure
3. Converges tighter in fewer steps since forces are already small

Variable nstep per pass: 400, 300, 300, 150, 150, 400. The last pass gets 400 again for one final push if still not converged.

Stops early if a pass doesn't improve force. Timeout per pass: 2.5h (high-P hydrides), 40 min (magnetic), 20 min (default).

The best structure from the refinement loop is saved to the DFT structure cache for future runs.

### SCF Skip

The unified vc-relax uses `disk_io='high'` and `conv_thr=1e-7`, so its final SCF IS production quality. SCF results are parsed directly from vc-relax output — no separate SCF step needed. Falls back to separate SCF only if vc-relax didn't converge. The `parseSCFOutput` function uses `matchAll` and takes the **last** match for all fields (energy, force, pressure, Fermi energy, etc.) to correctly parse vc-relax output with multiple ionic steps.

---

## Stage 5: Gamma-Point Phonon + Soft Mode Following

### Gamma Phonon

Fast dynamical stability screen. Uses `dynmat.x` post-processing to extract frequencies from `.dyn` file when `ph.x` doesn't print them to stdout.

### Pre-Phonon Force Gate

- force < 0.10 Ry/bohr: screening gate, allows surrogate Tc
- force < 0.03 Ry/bohr: DFPT gate, allows physics-grade e-ph coupling

### Soft Mode Following

When gamma phonon finds imaginary modes (freq < -50 cm-1), the structure WANTS to distort along those directions. Instead of giving up:

1. Parse eigenvectors from `dynmat.x` output for the most negative mode
2. Displace atoms along that eigenvector (0.05 Å amplitude)
3. Re-run unified vc-relax from the displaced structure
4. If converged, allow full phonon grid on the new structure

This turns imaginary modes from failure signals into search directions — guiding the structure toward the actual stable phase.

---

## Stage 6: Full Phonon Grid (Quality-Tiered)

QE `ph.x` on q-grid → phonon dispersion + DOS. Timeout scales with q-grid density, capped at 48h.

### Tiered Q-Grid by Structure Quality

| Force | ≤4 atoms light | ≤4 heavy | 5-8 light | 5-8 heavy | 9-15 | 16+ |
|-------|---------------|----------|-----------|-----------|------|-----|
| < 0.001 (publication) | 6×6×6 | 4×4×4 | 4×4×4 | 3×3×3 | 3×3×3 | 2×2×2 |
| < 0.03 (DFPT) | 4×4×4 | 3×3×3 | 3×3×3 | 2×2×2 | 2×2×2 | 1×1×1 |
| ≥ 0.03 (screening) | 2×2×2 | 2×2×2 | 2×2×2 | 1×1×1 | 1×1×1 | 1×1×1 |

### Tiered Convergence Threshold (tr2_ph)

- Publication (force < 0.001): `tr2_ph = 1e-14`
- DFPT (force < 0.03): `tr2_ph = 1e-12`
- Screening (force ≥ 0.03): `tr2_ph = 1e-10`

### Phonon Parameters

- `alpha_mix = 0.5` (default), `0.1` on retry — controls mixing aggressiveness in Sternheimer equations
- `reduce_io = .true.` — saves 10-20% on IO-bound systems
- Retry strategy: if ph.x crashes, retry with `tr2_ph=1e-10, alpha_mix=0.1` (gentler convergence)

Output: frequencies, dispersion, phonon DOS, omega_log, stability assessment.

---

## Stage 7: Round 2 Iterative Search

If SCF converged, the DFT winner seeds 45 focused candidates (30 mutations, 6 volume scans, 4 pressure scans, 5 distortions). Screened through CHGNet using enthalpy (H=E+PV) with meV/atom thresholds.

---

## Stage 8: Band Structure

Post-relaxation electronic structure: SCF → high-symmetry k-path → band crossings, DOS at Fermi level, flat band score. Workspace isolation (copies `.save/`).

---

## DFT Quality Gate (before e-ph/Tc)

| Check | Screening threshold | DFPT threshold |
|-------|--------------------|----|
| SCF converged | true | true |
| Residual force | < 0.10 Ry/bohr | < 0.03 Ry/bohr |
| Residual pressure | < +/- 50 kbar | < +/- 50 kbar |
| Metallic | true | true |
| Phonon stable | no large imaginary modes | no large imaginary modes |

### Method-Based Quality Tier Caps

| Phonon method | Max tier |
|---------------|----------|
| DFPT full q-grid | publication_ready |
| DFPT gamma-only | final_converged |
| xTB finite displacement | screening_converged |

---

## Stage 9: Electron-Phonon Coupling & Eliashberg → Tc

Only runs if DFPT quality gate passes. Method labels on every result:

- `alpha2FMethod`: dfpt_eph / surrogate_eph / unavailable
- `lambdaMethod`: dfpt_integrated_alpha2F / surrogate_alpha2F / estimated_from_dos_phonons

Only `dfpt_eph` is physics-grade.

---

## Stage 10: Results → Database → Next Iteration

Extended dataset fields: tcConservative, tcUpperBound, tcMethod, lambdaMethod, phononMethod, tcConfidence, learningScore, qualityTier, hullLabel, residualForce.

### Multi-Objective Learning Score

```
learning_score =
  0.35 × conservative_Tc_score (confidence-weighted)
+ 0.25 × phonon_stability_score
+ 0.20 × hull_stability_score
+ 0.10 × metallicity_score
+ 0.10 × novelty_score
```

### Result Validation

`validateResultConsistency()` before DB save checks for contradictions (tier vs method caps, force thresholds, confidence consistency).

---

## Uncertainty & Confidence

Every result carries: tcConfidence (high/medium/low/surrogate), lambdaConfidence, phononConfidence, structureConfidence, ephMethod, phononMethod, tcUncertaintyReason.

---

## Reproducibility Bundles

For every non-failed candidate: quality_report.json, candidate_provenance.json, final_structure.poscar, scf_summary.json, phonon_summary.json, dfpt_results.json.

---

## Adaptive Learning

Per-family volume learning, per-generator weighting, cage seeder subtype tracking (sodalite/clathrate/hex/bcc). Quality-weighted signals: funnel survival (0.1) → DFT converged (0.5) → phonon stable (3.0) → DFPT e-ph (4.0).

---

## Infrastructure

- **Old worker**: c2-standard-8 (8 vCPUs, 32 GB), QE 7.3.1 (lmaxx=6), QE_MPI_RANKS=3
- **New worker**: c2-standard-30 (30 vCPUs, 120 GB), QE 7.3.1 (lmaxx=6), QE_MPI_RANKS=24, QE_NPOOL=6
- Both pull from shared Neon DB job queue
- 2 GCP VMs processing materials in parallel
- QE binary search prefers `/usr/local/bin` (manual lmaxx=6 rebuild) over `/usr/bin` (apt default)
- Supported f-block elements: La, Ce, Th (lmaxx=6 rebuild). Remaining lanthanides (Pr-Tm) and actinides (Pa-Am) still blocked pending PP validation.
