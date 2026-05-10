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

**Adaptive nstep** based on observed convergence rate: after each pass, the pipeline measures force reduction per ionic step and estimates how many steps the next pass needs to reach the 0.001 target (with 1.5x safety margin, clamped to 100-400). First and last passes always get 400. This replaces hand-tuned nstep schedules with a principled estimate that adapts to each material's convergence behavior.

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

| e-ph method | Max tier |
|-------------|----------|
| EPW Migdal-Eliashberg | publication_ready |
| DFPT full q-grid | publication_ready |
| DFPT gamma-only | final_converged |
| xTB finite displacement | screening_converged |

---

## Stage 9: Electron-Phonon Coupling & Eliashberg → Tc

Only runs if DFPT quality gate passes. Method labels on every result:

- `alpha2FMethod`: epw_migdal_eliashberg / dfpt_eph / surrogate_eph / unavailable
- `lambdaMethod`: epw_anisotropic / dfpt_integrated_alpha2F / surrogate_alpha2F / estimated_from_dos_phonons

`epw_migdal_eliashberg` is the highest-grade method; `dfpt_eph` is physics-grade.

---

## Stage 9a: EPW Wannier-Interpolated Electron-Phonon Coupling

**File**: `server/dft/epw-pipeline.ts`

For publication-ready materials (force < 0.001, phonon stable, metallic), the pipeline runs the full EPW workflow to compute Tc via anisotropic Migdal-Eliashberg equations on ultra-dense Brillouin zone grids.

### EPW Pipeline Steps

1. **NSCF** (pw.x `calculation='nscf'`) — dense uniform k-grid with `nosym=.true.`, `noinv=.true.` for Wannier compatibility. Grid sizes: 12×12×12 (small cells) to 4×4×4 (large cells).

2. **Wannier90 preprocessing** (wannier90.x -pp) — generates .nnkp nearest-neighbor k-point information. Element-specific orbital projections from a 50+ element lookup table (H→s, TM→s;p;d, Ce/Th→s;p;d;f).

3. **pw2wannier90.x** — projects Bloch states onto Wannier functions, produces .amn (projections), .mmn (overlaps), .eig (eigenvalues).

4. **Wannier90 full** (wannier90.x) — computes maximally-localized Wannier functions (MLWFs) via spread minimization. Produces .chk checkpoint for EPW.

5. **EPW** (epw.x) — interpolates electron-phonon matrix elements from coarse DFPT q-grid to ultra-fine k/q grids (up to 40×40×40), then solves the anisotropic Migdal-Eliashberg equations.

### EPW Parameters

- **Fine grids**: adaptive — 40×40×40 for small cells with 6×6×6 phonon, scaled down for larger systems (minimum 16×16×16)
- `fsthick = 0.4 eV` (conventional), `1.0 eV` (hydrides with wide Fermi surfaces)
- `degaussw = 0.025 eV` — delta function smearing
- `laniso = .true.` — anisotropic gap equations
- `limag = .true.`, `lpade = .true.` — imaginary axis + Padé analytic continuation
- Temperature sweep: 5–300 K in 5 K steps

### EPW Outputs

- `lambda` — total electron-phonon coupling constant (fine-grid)
- `omega_log` — logarithmic average phonon frequency (meV)
- `Tc` — anisotropic Migdal-Eliashberg critical temperature (K)
- `Delta(0)` — superconducting gap at T=0 (meV)
- `alpha2F(omega)` — Eliashberg spectral function on fine grid

### Fallback Strategy

| Step | Failure mode | Fallback |
|------|-------------|----------|
| NSCF | Doesn't converge | Skip EPW, keep DFPT result |
| Wannier90 | Spread doesn't converge | Retry with SCDM auto-projections; if still fails, skip |
| EPW | Eliashberg doesn't converge | Use Allen-Dynes on EPW lambda/omega_log |
| EPW | lambda=0 or wildly different from DFPT | Flag warning, keep DFPT value |

Timeout budget: NSCF 1h, Wannier90 30min, EPW 4h. Total ~6h maximum.

---

## Stage 9b: Nuclear Quantum Effects (NQE) Correction

**File**: `server/physics/nqe-correction.ts`

For high-hydrogen-content compounds under pressure, hydrogen behaves quantum-mechanically — its zero-point motion is comparable to its mean displacement. The harmonic DFPT approximation overestimates phonon frequencies because it ignores the anharmonic potential surface that hydrogen explores.

### SSCHA-Model Correction

Implements a Stochastic Self-Consistent Harmonic Approximation (SSCHA)-inspired correction:

1. **Zero-point displacement**: `u_zp = sqrt(hbar / (2 * M_H * omega_H))` — typically 0.08–0.12 Å for hydrides
2. **Anharmonic strength parameter**: `sigma = (u_zp / d_nn)^2 * sqrt(M_avg / M_H)` — captures how much of the interatomic potential hydrogen explores
3. **Lambda renormalization**: `lambda_NQE = lambda_harm * R(sigma, P)` where `R = 1 - alpha * sigma / (1 + beta * sigma)` with pressure-dependent coefficients
4. **Omega_log renormalization**: Milder than lambda (logarithmic average weights all modes)
5. **Stability pressure shift**: Estimates how much the stability boundary moves (typically -20 to -40 GPa)

### Calibration Benchmarks

| Material | Pressure | DFPT λ | SSCHA λ | Reduction |
|----------|----------|--------|---------|-----------|
| H3S (Im-3m) | 200 GPa | 2.19 | 1.84 | -16% |
| LaH10 (Fm-3m) | 170 GPa | 3.41 | 2.29 | -33% |
| YH6 (Im-3m) | 165 GPa | 2.56 | 2.07 | -19% |
| CaH6 (Im-3m) | 150 GPa | 2.69 | 2.25 | -16% |
| LiH6 (R-3m) | 300 GPa | 2.80 | 1.68 | -40% |

### Application Gate

NQE corrections only apply when:
- H-fraction ≥ 0.3 AND H:metal ratio ≥ 3
- Pressure ≥ 20 GPa (dense hydrogen packing)
- Anharmonic strength sigma ≥ 0.03

Ref: Errea et al., Nature 578, 66 (2020); Monacelli et al., JPCM 33, 363001 (2021).

---

## Stage 9c: Ab-Initio Coulomb Pseudopotential (μ*)

**File**: `server/physics/mu-star-ab-initio.ts`

Replaces the conventional fixed μ* = 0.10–0.13 with a first-principles computation using the RPA-enhanced Morel-Anderson formula.

### RPA Morel-Anderson Method

1. **Thomas-Fermi screening**: `k_TF = sqrt(4π * N(E_F))` from computed DOS at Fermi level
2. **Screened Coulomb matrix element**: `V_c = 4π / (k_F² + k_TF²)` via RPA dielectric function
3. **Bare Coulomb parameter**: `mu_c = N(E_F) * V_c / epsilon_RPA`
4. **Morel-Anderson retardation**: `mu* = mu_c / (1 + mu_c * ln(E_F / omega_D))`
5. **Pressure correction**: Higher pressure → wider bandwidth → larger E_F/ω_D → lower μ*
6. **Orbital character correction**: d/f character increases effective Coulomb repulsion

### Key Outputs

- `muStar`: Computed value (physical range [0.05, 0.20])
- `conventionalMuStar`: What a fixed assumption would give (for comparison)
- `tcSensitivity`: How much Tc changes per 0.01 μ* shift (typically 2–10 K for hydrides)
- `deviationFromConventional`: Shows where fixed μ* was wrong

Ref: Morel & Anderson, Phys. Rev. 125, 1263 (1962); Agapito et al., PRX 5, 011006 (2015).

---

## Stage 10: Results → Database → Next Iteration

Extended dataset fields: tcConservative, tcUpperBound, tcMethod, lambdaMethod, phononMethod, tcConfidence, learningScore, qualityTier, hullLabel, residualForce, nqeApplied, nqeMethod, lambdaNQE, lambdaReduction, nqeAnharmonicStrength, nqeStabilityShift, muStarMethod, muStarConventional, muStarDeviation, muStarTcSensitivity, epwLambda, epwTcME, epwGapZero, epwMethod.

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

Every result carries: tcConfidence (high/medium/low/surrogate), lambdaConfidence, phononConfidence, structureConfidence, ephMethod (epw_migdal_eliashberg / dfpt / surrogate / none), phononMethod, tcUncertaintyReason.

---

## Reproducibility Bundles

For every non-failed candidate: quality_report.json, candidate_provenance.json, final_structure.poscar, scf_summary.json, phonon_summary.json, dfpt_results.json, epw_results.json (when EPW runs).

---

## Adaptive Learning

Per-family volume learning, per-generator weighting, cage seeder subtype tracking (sodalite/clathrate/hex/bcc). Quality-weighted signals: funnel survival (0.1) → DFT converged (0.5) → phonon stable (3.0) → DFPT e-ph (4.0) → EPW publication (5.0).

---

## Pseudopotential Sources

Download chain (priority order):
1. **Local repo** (`server/dft/pseudo/`) — pre-cached PPs
2. **System directories** — `/usr/share/espresso/pseudo`, SSSP .deb extraction
3. **PSLibrary** (GitHub dalcorso/pslibrary) — primary remote source, PBE PAW
4. **Pseudo-DOJO** (ONCVPSP-PBE-SR) — DFPT-validated norm-conserving, scalar-relativistic. Covers lanthanides/actinides without lmaxx issues. Ideal for phonon and EPW calculations.
5. **QE website** — fallback (often unreliable)
6. **GBRV** (Rutgers) — ultrasoft PPs, tertiary source

PP validation: UPF format check (header + closing tag), semicore state verification for TM/lanthanides, ≥10KB size gate. Failed downloads cached with 1h cooldown to prevent retry loops.

---

## Infrastructure

- **Old worker**: c2-standard-8 (8 vCPUs, 32 GB), QE 7.3.1 (lmaxx=6), EPW, QE_MPI_RANKS=3
- **New worker**: c2-standard-30 (30 vCPUs, 120 GB), QE 7.3.1 (lmaxx=6), EPW, QE_MPI_RANKS=24, QE_NPOOL=6
- Both pull from shared Neon DB job queue
- 2 GCP VMs processing materials in parallel
- QE binary search prefers `/usr/local/bin` (manual lmaxx=6 rebuild) over `/usr/bin` (apt default)
- Supported elements: nearly full periodic table. La, Ce, Th, Pr-Tm, Pa, U, Np all supported via lmaxx=6 + Pseudo-DOJO PPs. Only Pu, Am blocked (no reliable PPs).

---

## Known Physics Gaps & Roadmap

### Current Limitations

**1. Anharmonic phonon corrections (Priority: CRITICAL for hydrides)**
Hydrides at high pressure are notoriously anharmonic — the SCDFT/SSCHA framework (Errea, Calandra, Mauri) routinely shows that Tc predictions from harmonic DFPT are off by 20-40% for compounds like LaH10 and H3S. The pipeline applies a semi-empirical SSCHA-model correction (mass-dependent, H-cage-aware, pressure-stiffened) calibrated to published results, but full self-consistent SSCHA would be more accurate. Full SSCHA requires 100-1000+ DFT force calculations per material — practical only for the very best candidates.

**2. Anisotropic Eliashberg solver (Priority: ADDRESSED via EPW)**
EPW now provides anisotropic Migdal-Eliashberg gap equations for publication-ready materials. Multi-band superconductors (MgB2, iron pnictides, hydrides with multiple Fermi sheets) get properly resolved gap functions.

### Surrogate Tc Integrity

Surrogate Tc predictions (XGBoost/GNN) are allowed when force < 0.10 but ≥ 0.03 Ry/bohr. These surrogate models are useful for screening but **not reliable enough for superconductor claims**. The pipeline must ensure:

- Surrogate-tier results never propagate to "best Tc" claims without DFPT validation
- Dashboard/API clearly distinguishes `surrogate_eph` from `dfpt_eph` from `epw_migdal_eliashberg` in all displays
- The `tcConfidence` field accurately reflects the method: `surrogate` for non-DFPT, `high` only for DFPT/EPW e-ph

### Implementation Roadmap

| Phase | Addition | Status |
|-------|----------|--------|
| **Phase 1** | EPW integration (Wannier90 → EPW → Migdal-Eliashberg) | **DONE** |
| **Phase 2** | Pseudo-DOJO PP integration (lanthanide/actinide coverage) | **DONE** |
| **Phase 3** | NQE correction (SSCHA-model lambda/omega_log renormalization) | **DONE** |
| **Phase 4** | Ab-initio μ* (RPA Morel-Anderson) | **DONE** |
| **Phase 5** | Full SSCHA/PIMD integration | Future (major — external code, days of compute per material) |
| **Phase 6** | Full ACBN0 μ* with Wannier functions | Future (requires EPW Wannier data) |
