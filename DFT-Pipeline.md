# Quantum Alchemy Engine — Full DFT Pipeline

End-to-end superconductor discovery pipeline from candidate generation through phonon calculation and Tc prediction. The system discovers crystal structures independently — no literature lattice overrides, no forced positions.

---

## Stage 0: Learning Engine Picks a Formula

**File**: `server/learning/engine.ts`

The active learning loop selects which formulas to investigate — prioritizing high-entropy compositions, known superconductor families, and materials the ML model thinks are promising. Uses a multi-objective learning score (not raw max Tc) that weights stability, phonon quality, and metallicity alongside Tc predictions.

---

## Stage 1: Candidate Generation (Multi-Engine Fusion)

**File**: `server/dft/qe-worker.ts`

### LLM Structure Advisor (Pre-Generation)

**File**: `server/dft/structure-advisor.ts`

Before generating candidates, one gpt-4o-mini call per formula (~$0.001, ~3s, cached to disk permanently) provides structural hints:
- Expected crystal structure type (clathrate, perovskite, layered, A15, etc.)
- Likely space group number and alternatives
- Per-pair minimum distances (H-H, M-H, M-M) — used to set AIRSS MINSEP
- Element coordination roles (cage-center, vertex, network)
- Approximate lattice parameters for cross-checking Vegard

Falls back gracefully to heuristics if OpenAI is unavailable (circuit breaker).

### Candidate Generators

Seven sources run in parallel to create a diverse pool of crystal structure candidates:

1. **Vegard/VCA** — interpolates lattice + positions from binary endpoint structures (AFLOW/MP/known-structures)
2. **AIRSS** (`server/csp/airss-wrapper.ts`) — fully random cells via `buildcell`, pressure-aware MINSEP (informed by LLM-advised pair distances when available), Z-sweeps (Z=1,2,3,4,6,8), volume ensemble from Birch-Murnaghan
3. **PyXtal** (`server/csp/pyxtal-wrapper.ts`) — Wyckoff-aware random generation respecting space group symmetry, tiered SG sampling (40% high-sym / 30% med / 20% low / 10% P1)
4. **Cage Seeder** (`server/csp/cage-seeder.ts`) — for hydrides: 20-30 candidates at 3-5 pressure-compressed volumes each. Proper ternary A/M site placement: guest metals (Li, Na, K) go to interstitial "A" sites, host metals (La, Y, Ca) go to cage-center "M" sites. Uses Wyckoff orbits from 141 tagged prototype templates (sodalite, clathrate, hex-clathrate, bcc-hydride)
5. **Mutations** (`server/csp/structure-mutator.ts`) — 8 mutation types on top-3 candidates: lattice strain +/-10%, volume compress/expand, H shuffle, symmetry break, Wyckoff perturbation
6. **DFT Structure Cache** — previous best DFT-optimized structures per formula, injected at confidence=0.99. Stored in `/tmp/qe_calculations/structure_cache/{formula}.json`. Only overwrites if new structure has lower force than cached. Gives the pipeline a head start instead of rediscovering the same geometry from scratch.

Budget is tier-dependent (preview: ~85 candidates, deep: ~10K AIRSS + 1K PyXtal). DFT atom limit: 24 atoms per formula unit. Per-candidate supercell caps: preview 30, standard 40, deep 50, publication 60. Oversized candidates are backfilled from the next best. All timeouts, phonon grids, and EPW grids scale with atom count (calibrated base × (nAtoms/7)^1.2).

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

## Stage 3a: Spin-Orbit Coupling (SOC) Analysis

**File**: `server/dft/soc-handler.ts`

For heavy-element candidates (5d metals, 6p metals, lanthanides, actinides), non-relativistic DFT gets band ordering wrong — SOC splits degenerate states, reshapes the Fermi surface, and changes N(E_F). This propagates directly into λ and μ*.

### SOC Decision Tree

1. Any 6p element (Tl, Pb, Bi) or actinide (Th, U) → **full SOC** (noncolin + lspinorb)
2. Multiple 5d elements with SOC > 0.3 eV → **full SOC**
3. Single lanthanide with SOC < 0.2 eV → **scalar-relativistic** (standard PP sufficient)
4. Only 4d/5p elements → **scalar-relativistic**

### QE Flags When Full SOC Enabled

```
noncolin = .true.
lspinorb = .true.
```

nspin is ignored (QE uses 4-component spinors). Starting magnetization uses angle1/angle2 format. Computational cost ~2.5× scalar-relativistic.

### SOC Energy Scales

| Group | Elements | SOC (eV) | Priority |
|-------|----------|----------|----------|
| 6p metals | Tl, Pb, Bi | 1.0–2.0 | Critical |
| Actinides | Th, U | 0.8–1.2 | Critical |
| 5d metals | Hf–Au | 0.3–1.0 | Recommended |
| Lanthanides | La–Lu | 0.1–0.35 | Optional |
| 4d metals | Nb–Cd | 0.05–0.15 | Optional |

Ref: Dal Corso, CMS 95, 337 (2014); MacDonald et al., J. Phys. F 10, 2005 (1980).

---

## Stage 3b: Magnetic Ground-State Search

**File**: `server/dft/magnetic-ground-state.ts`

Before phonon calculations, the pipeline determines the correct magnetic ordering by running short SCF trials with different spin configurations. Running phonons on the wrong magnetic state produces frequencies that look fine but correspond to a metastable state.

### When Search Is Triggered

| Pattern | Orderings Tested | Rationale |
|---------|-----------------|-----------|
| Fe + As/P/Se/Te | NM, FM, AFM-stripe, AFM-checkerboard | Fe-pnictide: stripe vs checkerboard competition |
| Cu + O | NM, FM, AFM-layered | Cuprate: Neel order in CuO₂ planes |
| Mn/Cr + O | NM, FM, AFM-checkerboard, AFM-alternating | Complex magnetic landscapes |
| Ni + O | NM, FM, AFM-checkerboard | Nickelate magnetic ordering |
| 2+ strong magnetic species | NM, FM, AFM-alternating, ferrimagnetic | Competing exchange interactions |
| 1 magnetic + anion mediator | NM, FM, AFM-alternating | Superexchange may favor AFM |

### Search Protocol

1. Run short SCF (80 steps, conv_thr=1e-5, 10 min cap) for each magnetic ordering
2. Compare total energies — lowest wins
3. Parse total and absolute magnetization from QE output
4. If energy gap > 1 mRy/atom (~14 meV/atom): well-separated, high confidence
5. If nearly degenerate: warn that phonons may be sensitive to ordering

### Result Propagation

The winning magnetic state's nspin and starting_magnetization block are passed to all subsequent calculations (vc-relax, SCF, phonons). This ensures phonons are computed on the true magnetic ground state.

Ref: Mazin et al., PRL 101, 057003 (2008); Johannes & Mazin, PRB 79, 220510 (2009).

---

## Stage 3c: DFT+U Hubbard Workflow for Correlated Systems

**File**: `server/dft/hubbard-workflow.ts`

Plain GGA (PBE) gives qualitatively wrong electronic structure for any material with localized d or f electrons — band gaps are underestimated, orbital ordering is wrong, and magnetic moments are too small. DFT+U adds an on-site Coulomb correction that fixes this for the correlated orbital manifold.

### Composition-Aware U Values

U values are selected using a 3-level priority hierarchy (not just element-specific):

1. **Material-specific overrides** — validated U for known compounds:
   - Fe₂O₃: Fe U=4.3 eV (Materials Project)
   - Cuprates (La/Y/Ba...CuO): Cu U=5.0 eV (Anisimov 1991)
   - Nickelates (Nd/La...NiO): Ni U=5.1 eV (Lechermann 2020)
   - Fe-pnictides: Fe U=3.0 eV (tetrahedral, lower than octahedral)
   - NiO: Ni U=6.4 eV (Dudarev 1998)
   - MnO: Mn U=3.9 eV (Cococcioni 2005)

2. **Oxidation-state-aware** — estimates oxidation from anion:TM ratio:
   - High-oxidation (anion:TM > 2): uses higher U (e.g., Fe³⁺ → 4.3 eV)
   - Low-oxidation (anion:TM ≤ 2): uses lower U (e.g., Fe²⁺ → 3.0 eV)

3. **Element default** — from ELEMENTAL_DATA table (fallback)

### Broadened Trigger Conditions

DFT+U now activates for **all** materials with significant d/f correlation, not just "strongly-correlated" / "Mott-proximate":

- Any 4f/5f element → always apply
- Any d-electron element with U ≥ 3.0 eV → apply
- Moderately-correlated regime → apply (previously skipped)
- Exception: hydrogen-dominated (H-fraction > 60%) → skip (phonon BCS dominates)

### DFT+U Applied to vc-relax (Not Just SCF)

**Critical fix**: DFT+U is now applied during structural relaxation (vc-relax), not just the final SCF. For strongly-correlated materials, the relaxed geometry depends on U — without it, the structure optimizes on the wrong potential energy surface.

Applied to vc-relax when:
- Regime is "strongly-correlated" or "Mott-proximate"
- Any site has U ≥ 3.0 eV

### QE Input Generation

```
lda_plus_u = .true.
lda_plus_u_kind = 0        (Dudarev simplified)
Hubbard_U(1) = 4.3         (composition-aware value)
```

Ref: Dudarev et al., PRB 57, 1505 (1998); Cococcioni & de Gironcoli, PRB 71, 035105 (2005); Himmetoglu et al., IJQC 114, 14 (2014).

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

After the initial vc-relax, if residual force > 0.001 Ry/bohr OR residual pressure > ±50 kbar, the pipeline loops up to **6 refinement passes**. Each pass:

1. Restarts from the previous pass's final geometry with **zeroed velocities**
2. Eliminates oscillation inherited from the bad starting structure
3. Converges tighter in fewer steps since forces are already small

**Dual convergence gate**: Both force AND pressure must be within tolerance before declaring "publication-ready." Previously, force < 0.001 alone was sufficient, which caused materials like MgH6 to skip refinement with 140 kbar residual pressure — sending wrong-volume structures to phonons.

**Pressure-priority mode**: When force is already publication-ready (< 0.001) but pressure residual exceeds 50 kbar, the refinement enters a cell-equilibration mode:
- `forc_conv_thr = 1e-5` (ultra-tight, ions freeze on step 1)
- `press_conv_thr = 0.1 kbar` (tight cell convergence)
- Remaining nstep budget goes purely to cell dynamics (damp-w)
- Progress accepted if pressure improves, even if force stays flat
- Force degradation guard: rejects pass if force exceeds 1.5× threshold

**Adaptive nstep**: based on observed convergence rate from previous pass:
- Force-priority: estimates steps from force reduction per step × remaining gap
- Pressure-priority: estimates from pressure reduction per step (tracked across passes)
- Gap-proportional fallback when no rate data available
- All timeouts scale with atom count: base × (nAtoms/7)^1.2

Stops early if neither force nor pressure improved. The best structure from the refinement loop is saved to the DFT structure cache for future runs.

### SCF Skip

The unified vc-relax uses `disk_io='high'` and `conv_thr=1e-7`, so its final SCF IS production quality. SCF results are parsed directly from vc-relax output — no separate SCF step needed. Falls back to separate SCF only if vc-relax didn't converge. The `parseSCFOutput` function uses `matchAll` and takes the **last** match for all fields (energy, force, pressure, Fermi energy, etc.) to correctly parse vc-relax output with multiple ionic steps.

---

## Stage 5: Gamma-Point Phonon + Soft Mode Following

### Gamma Phonon

Fast dynamical stability screen. Always runs regardless of cost estimate (timeout cap raised to 8h). Uses `dynmat.x` post-processing to extract frequencies from `.dyn` file when `ph.x` doesn't print them to stdout.

### Pre-Phonon Force Gate

- force < 0.10 Ry/bohr: screening gate, allows surrogate Tc
- force < 0.03 Ry/bohr: DFPT gate, allows physics-grade e-ph coupling

### Gamma Soft Mode Following

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

## Stage 6a: Zone-Boundary Soft Mode Following

**File**: `server/dft/zone-boundary-softmode.ts`

When the full phonon grid (Stage 6) finds imaginary modes at q≠0 that gamma didn't catch, those modes tell us exactly which supercell distortion the structure wants. Instead of discarding the result:

1. **Scan .dyn files** for all q-points — find the worst instability (most negative frequency)
2. **Extract eigenvectors** at that q-point via matdyn.x with `flvec` output
3. **Apply the distortion**: u_n = Re[ε_n · exp(i·q·R_n)] × amplitude. For commensurate q-points (e.g., q=[0.5,0,0]), this creates a real-valued supercell modulation pattern.
4. **Re-relax** the displaced structure with vc-relax (200 steps, damped dynamics)
5. **Quick gamma phonon check** on the new structure
6. **Iterate** up to 3 times with increasing amplitude (0.03, 0.05, 0.08 fractional coords)

This is how the Pickard/Errea groups find stable high-pressure phases that no random search discovers — the phonon instability IS the search direction pointing toward the true ground-state structure.

Ref: Pickard & Needs, JPCM 23, 053201 (2011); Errea et al., PRL 114, 157004 (2015).

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

## Stage 9c: Full SSCHA Anharmonic Phonon Corrections

**Files**: `server/dft/sscha-pipeline.ts` + `server/dft/sscha-worker.py`

For publication-ready hydrides, runs the full Stochastic Self-Consistent Harmonic Approximation to replace harmonic DFPT phonons with anharmonic ones. This is the single biggest physics correction for hydride Tc predictions (20-40% reduction in λ for materials like LaH10 and H3S).

### SSCHA Workflow

1. Load harmonic dynamical matrices from DFPT (.dyn files)
2. Generate stochastic displaced atomic configurations (50 per iteration)
3. Compute DFT forces on each configuration via pw.x SCF
4. Feed forces to SSCHA free energy minimizer
5. Self-consistently update the dynamical matrix until the free energy Hessian converges
6. Extract: anharmonic ω_log, corrected λ, free energy, Tc with anharmonic correction

### Two Execution Modes

- **Full SSCHA** (when python-sscha + cellconstructor installed): proper stochastic sampling + self-consistent minimization matching Errea/Monacelli group methodology
- **Numpy fallback**: stochastic displacement + least-squares force constant fitting. Less rigorous but still captures dominant anharmonic effects without external dependencies

### Eligibility Gate

- Residual force < 0.001 Ry/bohr (publication-ready)
- H-fraction ≥ 25% of total atoms
- Pressure ≥ 20 GPa (dense hydrogen packing where anharmonicity is significant)

### Compute Budget

- 50 displaced configurations × 3-8 iterations = 150-400 DFT force calculations
- Each force calculation: 5-30 min depending on system size
- Total: 12-200 hours per material
- Timeout: 24h cap

Ref: Errea et al., Nature 578, 66 (2020); Monacelli et al., JPCM 33, 363001 (2021).

---

## Stage 9d: ACBN0 First-Principles Coulomb Pseudopotential (μ*)

**File**: `server/dft/acbn0-pipeline.ts`

Computes μ* from first principles via QE's hp.x (Hubbard parameters from DFPT linear response) instead of using the conventional fixed μ* = 0.10-0.13.

### ACBN0 Self-Consistent Workflow

1. Run DFT+U SCF with initial Hubbard U values (or U=0)
2. Run hp.x — DFPT linear response computes screened Coulomb interaction
3. Parse hp.x output: chi0/chi susceptibilities, Thomas-Fermi screening length, Hubbard U
4. Check convergence (ΔU < 0.1 eV) — if not converged, update U and repeat from step 1
5. Compute μ* from screening data + Morel-Anderson retardation:
   - `mu_bare = N(E_F) * V_screened` from hp.x screening
   - `mu* = mu_bare / (1 + mu_bare * ln(E_F / omega_D))`

### Three μ* Computation Methods (ranked by data quality)

1. **hp.x chi0/chi screening** — direct from DFPT linear response (best)
2. **Thomas-Fermi screening length** — from hp.x k_TF output
3. **N(E_F) estimate** — pure DOS-based (fallback)

### Key Outputs

- `muStar`: First-principles value (physical range [0.05, 0.20])
- `hubbardU`: Per-element computed U values from hp.x linear response (eV)
- `screeningLength`: Thomas-Fermi screening length (Bohr)
- `converged`: Whether self-consistent U loop converged

### Self-Consistent U Feedback Loop

When ACBN0 computes U from first principles via hp.x, those values are fed back into the Hubbard workflow result:
- Each element's U in `hubbardWorkflow.sites` is updated with the hp.x value
- The source is promoted to "material-specific" (first-principles quality)
- Large shifts (>0.3 eV from initial U) are logged as warnings
- These values persist in the dataset for use in future runs of the same formula

### Compute Budget

- 2-3 iterations of DFT+U SCF + hp.x
- Total: 1-2 hours per material
- Timeout: 2h cap

Ref: Morel & Anderson, Phys. Rev. 125, 1263 (1962); Agapito et al., PRX 5, 011006 (2015); Timrov et al., PRB 98, 085127 (2018).

---

## Stage 9e: Spin-Fluctuation Pairing Channel

**File**: `server/physics/spin-fluctuation-pairing.ts`

For cuprates, iron-pnictides, nickelates, and heavy-fermion systems, superconductivity is driven by spin fluctuations rather than phonons. The Eliashberg pipeline now identifies and quantifies this channel.

### Pairing Channel Classifier

Uses existing pipeline data to determine the dominant pairing mechanism:

| Signal | Phonon-BCS | Spin-Fluctuation |
|--------|-----------|-----------------|
| Magnetic ground state | NM or FM | AFM (stripe, checkerboard, layered) |
| Fermi surface | No nesting | Strong nesting at Q≠0 |
| Correlation regime | Weakly correlated | Mott-proximate or strongly correlated |
| Band character | sp-dominated | d/f orbital, flat bands at E_F |
| Material pattern | Hydride, elemental metal | Cuprate, pnictide, heavy-fermion |

Output: `phonon-bcs` / `spin-fluctuation` / `mixed-phonon-spin` / `orbital-fluctuation`

### Lindhard + RPA Spin Susceptibility

1. **Static Lindhard χ₀(Q)**: Non-interacting susceptibility from DOS at Fermi level, enhanced by nesting score. Cheap — uses existing electronic structure data.
2. **RPA enhancement**: χ_RPA(Q) = χ₀(Q) / (1 - U·χ₀(Q)) where U is from the DFT+U Hubbard workflow. Stoner factor S = 1/(1-U·χ₀) detects proximity to magnetic instability.
3. **Characteristic frequency**: ω_sf = W/S (bandwidth / Stoner enhancement). For cuprates near optimal doping, ω_sf ≈ 40-80 meV.

### Spin-Fluctuation Tc

- **Coupling constant**: λ_sf ≈ N(E_F) · U² · χ_RPA(Q) / ω_sf
- **d-wave formula**: Monthoux-Scalapino with μ*_sf = 0 (d-wave symmetry cancels isotropic Coulomb repulsion)
- **s±-wave formula**: for pnictides, μ*_sf = 0.05 (partial cancellation from sign change between pockets)

### Combined Tc (Phonon + Spin Fluctuation)

The interaction between channels depends on pairing symmetry:
- **d-wave cuprates**: phonons are weakly pair-breaking. Tc = Tc_sf - 0.15·Tc_ph (destructive)
- **s± pnictides**: both channels constructive. Tc = sqrt(Tc_ph² + Tc_sf²)
- **Conventional BCS**: spin fluctuations negligible. Tc = Tc_ph

Ref: Berk & Schrieffer, PRL 17, 433 (1966); Moriya, Spin Fluctuations (1985); Scalapino, Rev. Mod. Phys. 84, 1383 (2012); Monthoux et al., Nature 450, 1177 (2007).

---

## Stage 9f: DMFT-Ready Bundle Export

**File**: `server/dft/dmft-bundle-exporter.ts`

For correlated materials (Mott-proximate, strongly-correlated, moderately-correlated regimes) with publication-ready or final-converged quality, the pipeline exports a self-contained DMFT bundle. This bundle packages everything needed for solid_dmft + TRIQS/CTHYB on the gnn-training VM.

### DMFT Eligibility Gate

- DFT+U applied (correlated d/f orbitals present)
- Quality tier: `final_converged` or `publication_ready`
- Correlation regime: Mott-proximate, strongly-correlated, or moderately-correlated

### Bundle Contents (HDF5)

| Group | Contents |
|-------|----------|
| `/hamiltonian/` | H(k) on uniform k-mesh (Fourier-transformed from Wannier90 `_hr.dat`), k-points, mesh dimensions |
| `/correlated_subspace/` | Correlated shell definitions (atom index, l, dim), projector matrices, corr_to_inequiv mapping |
| `/interaction/` | Per-shell U and J values (from Hubbard workflow + ACBN0 first-principles), interaction type |
| `/structure/` | Lattice vectors, fractional positions, elements, formula, pressure |
| `/electronic/` | Fermi energy, n_electrons in correlated subspace, magnetic ordering, correlation regime |
| `/metadata/` | Bundle version, creation timestamp, QE quality tier |

### Wannier90 DMFT Projector Mode

**File**: `server/dft/epw-pipeline.ts` — `wannierMode = 'dmft_projector'`

The existing Wannier90 pipeline supports two modes via the `WannierMode` type:

| Parameter | EPW Mode | DMFT Projector Mode |
|-----------|----------|-------------------|
| **Orbitals** | All (s;p;d;f) per element | Only correlated d or f |
| **Energy window** | E_F ± 15/20 eV (wide) | E_F ± 5 eV (tight) |
| **Disentanglement** | 200 iterations, mix 0.5 | Disabled (frozen = full window) |
| **Convergence** | 200 iter, default tol | 500 iter, 1e-10 tol |
| **Output** | `.chk` for EPW | `_hr.dat` H(R) for TRIQS import |
| **Projections** | `Cu: s;p;d` | `Cu: d` |

### DMFT Infrastructure (gnn-training VM)

**Files**: `dmft/Dockerfile`, `dmft/docker-compose.yml`, `dmft/dmft-service.py`, `dmft/run-dmft.py`

Docker container running TRIQS 3.3.x + solid_dmft + CTHYB on the gnn-training VM (`34.130.121.199`). HTTP service on port 8780 accepts bundle submissions and runs DMFT calculations using 20 MPI ranks.

| Component | Version | Purpose |
|-----------|---------|---------|
| TRIQS | 3.3.x | Core Green's function library |
| TRIQS/CTHYB | 3.3.x | Continuous-time hybridization expansion QMC solver |
| TRIQS/DFTTools | 3.3.x | Wannier90 converter, SumkDFT |
| TRIQS/maxent | 3.3.x | Analytic continuation (Matsubara → real frequency) |
| solid_dmft | stable | High-level DMFT driver (reads bundles, drives CTHYB) |

Setup: `sudo bash dmft/setup-dmft.sh` on the gnn-training VM.

### DMFT Solver Parameters

- **Double-counting**: cFLL (fully localized limit) for Mott-proximate/strongly-correlated; cAMF (around mean field) for moderately-correlated
- **Interaction**: Kanamori for Liechtenstein (kind=1) materials; density-density for Dudarev (kind=0)
- **Temperature**: β = 40 eV⁻¹ (~300 K) default, adjustable per job
- **QMC**: 50K warmup cycles, 5M measurement cycles, cycle length 200

Ref: Georges et al., Rev. Mod. Phys. 68, 13 (1996); Aichhorn et al., CPC 204, 200 (2016); Merkel et al., CPC 264, 107surface (2021).

---

## Stage 9g: Two-Particle Vertex Measurement (G²)

**File**: `dmft/vertex_measurement.py`

After single-particle DMFT converges, a second CTHYB run measures the local two-particle Green's function G²(iν, iν', iΩ) using the `measure_G2_iw_ph` flag. This is 10-100× more expensive than the 1P measurement because G² is a three-frequency object with O(n_orb⁴) orbital entries.

### Frequency Grid Sizing

| Parameter | Typical range | Scaling |
|-----------|--------------|---------|
| n_iw_f (fermionic) | 10-80 | Limited by memory (~n_iw_f² × n_orb⁴) |
| n_iw_b (bosonic) | 5-40 | Ω=0 most important for pairing |
| QMC cycles | 5-20× base | G² needs more statistics per bin |
| Memory | 1-32 GB | complex128 per element |

Memory formula: `(2·n_iw_f)² × (2·n_iw_b+1) × n_orb⁴ × 16 bytes`

### Outputs

- **G²_loc(iν, iν', iΩ)**: full two-particle Green's function from CTHYB
- **χ⁰_loc(iν, iΩ)**: bare bubble susceptibility: `χ⁰ = -β · G(iν) · G(iν+iΩ)` (vectorized via einsum)
- **χ_loc(iν, iν', iΩ)**: connected susceptibility: `χ = G² - β·G·G·δ(Ω)`

Budget: 4-24 additional hours per material on 20 MPI ranks.

Ref: Boehnke et al., PRB 84, 075145 (2011); Hafermann et al., EPL 85, 27007 (2009); Rohringer et al., Rev. Mod. Phys. 90, 025003 (2018).

---

## Stage 9h: Local Bethe-Salpeter Equation (BSE)

**File**: `dmft/bse_solver.py`

Extracts the local irreducible vertex Γ_loc from χ_loc and χ⁰_loc by inverting the BSE:

```
Γ_loc(iΩ) = [χ⁰_loc(iΩ)]⁻¹ - [χ_loc(iΩ)]⁻¹
```

For each bosonic frequency iΩ, this is a matrix inversion in the compound index I = (iν, a, b) where a,b are orbital indices.

### Implementation Details

- **Compound index**: I = iν × n_orb² + a × n_orb + b, giving matrix dimension N = 2·n_iw_f × n_orb²
- **SVD-stabilized inversion**: truncated SVD with configurable cutoff (default 1e-8) handles ill-conditioned matrices at high frequencies
- **Diagnostics**: condition numbers, number of truncated singular values tracked per bosonic frequency
- **Channel decomposition**:
  - Γ_charge = (Γ + Γᵀ)/2 (symmetric, density fluctuations)
  - Γ_spin = (Γ - Γᵀ)/2 (antisymmetric, magnetic fluctuations)
  - Γ_singlet = (3/2)·Γ_spin + (1/2)·Γ_charge (singlet pairing vertex)
  - Γ_triplet = -(1/2)·Γ_spin + (1/2)·Γ_charge (triplet pairing vertex)

Ref: Rohringer et al., Rev. Mod. Phys. 90, 025003 (2018) Sec. III; Galler et al., PRB 95, 115107 (2017).

---

## Stage 9i: Pairing Susceptibility & Tc from DMFT

**File**: `dmft/pairing_susceptibility.py`

With Γ_singlet from the BSE, constructs the linearized Eliashberg equation on the lattice k-mesh:

```
λ · Δ(k, iν) = -(T/N_k) Σ_{k',ν'} Γ_singlet(ν,ν') · G(k',ν') · G(-k',-ν') · Δ(k',ν')
```

The leading eigenvalue λ_pair(T) → 1 from below signals the superconducting transition. The eigenvector gives the gap function Δ(k).

### Pairing Kernel Construction

1. **Lattice G(k,iω)**: computed from H(k), Σ(iω), μ via batch matrix inversion
2. **Time-reversal**: G(-k,-iω) = G(k,iω)* for paramagnetic systems; explicit -k mapping via nearest-neighbor search on the k-mesh
3. **Sparse mode**: for systems with n_k × N_vertex > 5000, uses scipy `LinearOperator` + Lanczos instead of dense eigendecomposition

### Gap Symmetry Classification

The gap eigenvector Δ(k) is projected at the lowest Matsubara frequency and orbital-traced, then overlapped with symmetry basis functions:

| Symmetry | Basis function | Typical system |
|----------|---------------|----------------|
| s-wave | 1 | Conventional BCS |
| s±-wave | cos(kx) + cos(ky) | Fe-pnictides |
| d-x²-y²-wave | cos(kx) - cos(ky) | Cuprates |
| d-xy-wave | sin(kx)·sin(ky) | Some heavy-fermion |
| p-x-wave | sin(kx) | Sr₂RuO₄ (debated) |
| g-wave | sin(kx)·sin(ky)·(cos(kx)-cos(ky)) | Exotic |

Nodal structure classified by sign changes of Re(Δ(k)) along high-symmetry directions.

### Tc Extrapolation

From λ_pair(T) at multiple temperatures:
- **λ ≥ 1**: Tc bracketed by interpolation between data points
- **0.3 < λ < 1**: Linear extrapolation of 1/λ(T) → 1
- **λ < 0.1**: No SC instability at accessible temperatures

Ref: Maier et al., PRL 95, 237001 (2005); Scalapino, Rev. Mod. Phys. 84, 1383 (2012); Gull et al., Rev. Mod. Phys. 83, 349 (2011).

---

## Stage 9j: Cluster DMFT via Dynamical Cluster Approximation (DCA)

**Files**: `dmft/dca_solver.py`, `dmft/hubbard_benchmark.py`

Single-site DMFT treats the self-energy as k-independent: Σ(k,iω) → Σ(iω). This makes d-wave pairing invisible because d-wave requires Σ at K=(0,0) to differ from Σ at K=(π,0). Cluster DMFT fixes this by embedding a cluster of N_c sites in the lattice.

### Why DCA (not CDMFT)

| Aspect | CDMFT (real-space) | DCA (momentum-space) |
|--------|-------------------|---------------------|
| Translational symmetry | Broken | Preserved |
| Self-energy | Σ(r, r'; iω) | Σ(K, iω) |
| d-wave pairing | Artifacts from broken symmetry | Clean d-wave channel |
| Preferred for | Small-gap physics | Pairing instabilities |

DCA is the standard for cuprate d-wave studies (Maier, Jarrell, Scalapino).

### DCA Algorithm

For N_c = 4 (2×2 plaquette), the cluster momenta are:
- **K₀** = (0,0) — Γ point (nodal region)
- **K₁** = (π,0) — X point (antinodal, where d-wave gap is maximal)
- **K₂** = (0,π) — Y point (antinodal)
- **K₃** = (π,π) — M point

Self-consistency loop:
1. **Σ_c(K, iω)** → lattice G(k, iω) = [iω + μ - ε(k) - Σ_c(K(k), iω)]⁻¹
2. **Coarse-grain**: Ḡ(K, iω) = (1/N_patch) Σ_{k∈patch(K)} G(k, iω)
3. **Cluster bath**: G⁰_c = [Ḡ⁻¹ + Σ_c]⁻¹
4. **Solve cluster**: CTHYB on N_c-site problem → G_c(K, iω)
5. **Extract Σ**: Σ_c = G⁰_c⁻¹ - G_c⁻¹
6. **Mix and iterate** until ||ΔΣ|| < tolerance

### DCA Pairing Susceptibility

The particle-particle bubble in DCA:
```
χ⁰_pp(K, iν) = -(T/N_patch) Σ_{k∈patch(K)} G(k,iν) · G(-k,-iν)
```

d-wave form factor: φ_d(K) = cos(Kx) - cos(Ky). For the 2×2 cluster, φ_d vanishes at Γ and M, is -2 at X and +2 at Y — d-wave lives entirely on the antinodal patches.

### 2D Hubbard Benchmark (Calibration Gate)

**File**: `dmft/hubbard_benchmark.py`

Before running DCA on real materials, the pipeline must reproduce the canonical result:

| Parameter | Value |
|-----------|-------|
| Model | 2D square lattice Hubbard |
| U | 8t (intermediate coupling) |
| t' | -0.3t (hole-like Fermi surface) |
| Doping | 15% hole (n ≈ 0.85) |
| N_c | 4 (2×2 DCA) |
| Expected Tc | T_c/t ≈ 0.02 (~100-200K for cuprate t) |

Validation checks:
1. d-wave eigenvalue λ_d > s-wave λ_s at all temperatures
2. λ_d increases monotonically as T decreases
3. λ_d → 1 near T/t ≈ 0.02

Run: `POST /benchmark {"quick": true}` on the DMFT service, or `python3 hubbard_benchmark.py --quick`.

Ref: Maier et al., PRL 95, 237001 (2005); Hettler et al., PRB 58, R7475 (1998); Jarrell et al., PRB 64, 195130 (2001).

---

## Stage 9k: Multi-Orbital DCA Cluster

**File**: `dmft/multiorbital_dca.py`

Extends the single-band DCA (Stage 9j) to multi-orbital systems. Cost scales as ~exp(β·U·n_orb·N_c) due to the fermionic sign problem — 10-100× more expensive than single-band for 3-5 orbital models.

### Supported Models

| Model | n_orb | Basis | Use case |
|-------|-------|-------|----------|
| 3-band Emery | 3 | Cu-d_{x²-y²}, O-p_x, O-p_y | Cuprates (La₂CuO₄, YBCO) |
| 5-band d-shell | 5 | All d orbitals | Fe-pnictides, nickelates |
| Arbitrary | n | From Wannier90 downfolding | Any correlated material |

### Kanamori Interaction

Full rotationally-invariant interaction (`KanamoriInteraction` class):
- **Intra-orbital**: U · n_{a↑} n_{a↓}
- **Inter-orbital opposite spin**: U' · n_{a↑} n_{b↓} where U' = U - 2J
- **Inter-orbital same spin**: (U'-J) · n_{aσ} n_{bσ}
- **Spin-flip**: -J · c†_{a↑} c_{a↓} c†_{b↓} c_{b↑}
- **Pair-hopping**: J · c†_{a↑} c†_{a↓} c_{b↓} c_{b↑}

Falls back to density-density-only (no spin-flip / pair-hopping) when sign problem is too severe (typically n_orb ≥ 5).

### Multi-Orbital DCA Self-Consistency

Same loop as single-band but with matrix-valued quantities at every step:
- G(k,iω) is [n_orb × n_orb] matrix inversion at each (k, iω)
- Σ_c(K,iω) is [nc × n_orb × n_orb] cluster self-energy
- CTHYB cluster solver has block size N_c × n_orb per spin channel

Ref: Emery, PRL 58, 2794 (1987); Werner et al., PRL 97, 076405 (2006); Gull et al., PRB 82, 155101 (2010).

---

## Stage 9l: Charge Self-Consistent DFT+DMFT

**File**: `dmft/charge_selfconsistency.py`

In standard DFT+DMFT the DFT Hamiltonian is computed once. But DMFT changes the orbital occupations, which should feed back into the DFT charge density. The CSC loop:

```
DFT → H(k) → DMFT → ρ_DMFT(r) → DFT(ρ_new) → H'(k) → DMFT → ...
```

### Outer Loop Protocol

1. **QE SCF** with current density → updated band structure (via `qe_callback`)
2. **Wannier90 re-projection** → updated H(k) (via `wannier_callback`)
3. **DMFT** (single-site or DCA) → Σ(iω), new density matrix n_DMFT
4. **Density correction**: Δn = n_DMFT - n_DFT, written as QE occupation restart
5. **Mixing**: ρ_next = α·ρ_old + (1-α)·ρ_DMFT (default α=0.3)
6. **Convergence**: ||ρ_new - ρ_old|| < 10⁻⁴ → stop (typically 5-15 iterations)

### Density Matrix Extraction

From Matsubara Green's function with proper tail correction:
```
n_{ab} = δ_{ab}/2 + (1/β) Σ_n [G_{ab}(iω_n) - δ_{ab}/(iω_n)]
```

Ref: Savrasov et al., PRL 87, 216405 (2001); Haule et al., PRB 81, 195107 (2010); Aichhorn et al., PRB 84, 054529 (2011).

---

## Stage 9m: Realistic-Model Pairing Pipeline

**File**: `dmft/realistic_pairing.py`

Combines all DMFT stages into a single pipeline for real superconductor candidates:

1. **Phase A**: Load DMFT bundle, auto-detect material class (cuprate/pnictide/nickelate)
2. **Phase B**: (Optional) Charge self-consistency → CSC-corrected H(k)
3. **Phase C**: Multi-orbital DCA temperature sweep (5+ temperatures)
4. **Phase D**: Pairing eigenvalues at each T with orbital resolution
5. **Phase E**: Tc extrapolation from λ_pair(T) → 1, gap symmetry classification

### Material Auto-Detection

| Pattern | Class | Model | Correlated orbitals |
|---------|-------|-------|-------------------|
| Cu + O | cuprate | 3-band Emery | Cu-d only |
| Ni + O | nickelate | d-shell | Ni-d |
| Fe + As/Se | pnictide | 5-band d | All Fe-d |
| Other | generic | From Wannier | All with U > 0 |

### Invoke

```bash
# Via HTTP service:
curl -X POST -F bundle=@material.h5 \
  -F 'options={"run_realistic_pairing": true, "run_csc": true, "dca_nc": 4}' \
  http://localhost:8780/submit

# Direct:
python3 realistic_pairing.py /data/bundles/LaCuO4.h5
```

Ref: Gull et al., PRB 82, 155101 (2010); Kitatani et al., PRB 102, 220502 (2020); Kent et al., PRB 72, 060411 (2005).

---

## Stage 9n: Pipeline Orchestrator — Production Automation

**File**: `dmft/pipeline_orchestrator.py`

Wraps all DMFT stages (A-D) into an unattended production pipeline with automated decision-making, rigorous convergence detection, and sign-problem fallback.

### Automated Cluster Size Selection

Based on material symmetry, orbital count, and resource budget:

| n_orb | Sign estimate formula | N_c=4 | N_c=8 | N_c=16 |
|-------|----------------------|-------|-------|--------|
| 1 (single-band) | exp(-0.015·β·N_c) | Always | β<70 | β<35 |
| 3 (Emery) | exp(-0.015·β·3·N_c) | β<45 | Marginal | Infeasible |
| 5 (d-shell) | exp(-0.015·β·5·N_c) | β<25 | Infeasible | Infeasible |

Threshold: ⟨sign⟩ > 0.05 to accept a configuration. Memory and walltime budgets also considered.

### Convergence Gates (`ConvergenceGate`)

Every DCA iteration is validated against configurable criteria:

| Check | Threshold | Action on failure |
|-------|-----------|-------------------|
| NaN/Inf in Σ or G | Zero tolerance | Immediate fallback |
| ||ΔΣ|| convergence | 10⁻⁴ (configurable) | Continue iterating |
| Average QMC sign | < 0.05 | Trigger fallback chain |
| Average QMC sign | < 0.2 | Log warning |
| Density vs target | > 0.02 | Log warning |
| Causal self-energy | Im[Σ(ω≈0)] > 0 | Trigger fallback |
| Max iterations | 30 | Stop, mark unconverged |

### Numerical Validators

- `validate_array()` — NaN/Inf/suspiciously-large checks on any numpy array
- `validate_green_function()` — tail decay (1/iω), spectral weight positivity
- `validate_self_energy()` — causality (Im[Σ] ≤ 0 at low frequency)

Applied to every intermediate result before advancing to the next phase.

### Sign-Problem Fallback Chain (`FallbackChain`)

Ordered strategies, most accurate → cheapest:

1. **Full Kanamori** at selected N_c — includes spin-flip + pair-hopping
2. **Density-density only** at same N_c — drops off-diagonal interaction terms
3. **Reduced cluster** N_c/2 with density-density — halves cluster
4. **Single-site DMFT** — no momentum dependence, no d-wave

Each fallback is triggered by: NaN/Inf in output, ⟨sign⟩ < 0.05, or uncaught exception. The chain is logged with rationale for every transition.

### Resource Accounting (`ResourceTracker`)

- Per-phase walltime, peak memory, status tracked
- Budget enforcement: pre-flight check before each phase
- Phase budgets: DMFT 10%, vertex 30%, BSE 5%, pairing 5%, DCA 40%, CSC 10%
- Automatic temperature sweep truncation when budget runs low

### Pairing Channel Identification

Automated with confidence scoring:
- **High confidence**: dominant channel > 2× runner-up
- **Medium confidence**: dominant > 1.3× runner-up
- **Low confidence**: channels nearly degenerate
- Consistency check against material class (cuprates → d-wave expected)

### Structured Logging

Dual output:
- **Console** (INFO): concise phase-level status
- **File** (`pipeline.log`, DEBUG): full diagnostics with timestamps

### Service Integration

Default production mode:
```bash
curl -X POST -F bundle=@material.h5 http://localhost:8780/submit
# Automatically uses orchestrated mode with auto cluster selection
```

Manual override:
```json
{"mode": "orchestrated", "dca_nc": 8, "max_walltime_hours": 48, "run_csc": false}
```

### Production Integration (Gap Fixes)

The pipeline is now end-to-end connected:

| Connection | Implementation |
|-----------|---------------|
| H(k) in bundle | qe-worker runs full NSCF → wannier90 -pp → pw2wannier90 → wannier90 in DMFT projector mode before bundle export |
| QE → DMFT service | `DMFT_SERVICE_URL` env var triggers `POST /submit` with bundle path after export; job_id stored in QEFullResult |
| DMFT result polling | qe-worker polls `GET /status/{id}` every 5 min (up to 1h), fetches `GET /result/{id}` on completion, populates all 15 DMFT database fields |
| DMFT → database | 15 columns in `quantumEngineDataset` + migration `0001_add_dmft_columns.sql` (registered in Drizzle journal) |
| Sign problem detection | avg_sign extracted from CTHYB `S.average_sign` in both single-band and multi-orbital solvers; DCA loop aborts at <0.05, warns at <0.2 |
| CSC density feedback | `csc_callbacks.py` auto-creates QE SCF + Wannier90 callbacks when binaries are available; `.win` built by Python-native `_build_dmft_win_from_bundle()` (no TS dependency) |
| Environment config | `DMFT_SERVICE_URL=http://localhost:8780` set in `gcp-worker/setup.sh` env template + auto-appended by `dmft/setup-dmft.sh` |
| DCA data flow | Both `run_dca()` and `run_multiorbital_dca()` return `g0_c` in their result dicts, enabling cumulant periodization (the causal method) |
| Module integration | All 6 post-processing modules (analytic continuation, Tc correction, periodization, cluster vertex, DCA++, adaptive temperature) are imported and called from the orchestrator |
| Docker image | All 20 Python modules COPY'd into container (including `convert-bundle.py`) |

---

## Stage 9o: Analytic Continuation — Real-Axis Spectral Functions

**File**: `dmft/analytic_continuation.py`

DMFT produces Σ(iω_n) on the Matsubara axis. Experimentalists measure A(ω) on the real axis (ARPES, STM/STS, optical conductivity). The analytic continuation iω → ω+iδ is an ill-posed inverse problem — small noise in G(iω) produces large artifacts in A(ω).

### Methods (ranked by reliability)

| Method | When to use | Reliability |
|--------|-------------|-------------|
| **MaxEnt** (TRIQS/maxent) | Default — Bayesian inference with entropy prior | High for single peaks, medium for fine structure |
| **Padé approximants** | Quick check, few Matsubara points | Low — unstable for noisy QMC data |
| **Stochastic analytic continuation** | Research — multiple independent reconstructions averaged | Highest but expensive |

### Outputs

- **A(ω)**: orbital-resolved spectral function on real-frequency grid
- **Σ(ω)**: real-axis self-energy (real + imaginary parts)
- **N(E_F)**: DMFT-corrected density of states at Fermi level
- **Z**: quasiparticle weight from Re[Σ(ω)] slope at ω=0

---

## Stage 9p: DMFT-Corrected Tc for Phonon-Mediated Superconductors

**File**: `dmft/dmft_tc_correction.py`

For correlated metals where both phonons and electronic correlations matter (e.g., A15 compounds, doped SrTiO₃, nickelates near metallicity), the DMFT spectral function gives a better N(E_F) than DFT. The Allen-Dynes formula uses N(E_F) directly:

```
Tc = (ω_log / 1.2) · exp[-1.04(1+λ) / (λ - μ*(1+0.62λ))]
```

where λ ∝ N(E_F). A one-shot correction:
1. Run DMFT → get A(ω) via analytic continuation
2. Extract N_DMFT(E_F) from A(ω=0)
3. Scale λ: λ_corrected = λ_DFT × [N_DMFT(E_F) / N_DFT(E_F)]
4. Recompute Tc with corrected λ

Also computes the mass enhancement m*/m = 1/(1 - ∂Σ/∂ω|_{ω=0}) which renormalizes the electron-phonon coupling.

---

## Stage 9q: DCA Self-Energy Periodization

**File**: `dmft/dca_periodization.py`

DCA gives Σ at N_c cluster momenta K. For Fermi surface plots, ARPES comparison, and gap function visualization, we need Σ(k) on the full BZ. The DCA periodization prescription:

```
Σ_lat(k, iω) = Σ_K Σ(K, iω) · φ_K(k)
```

where φ_K(k) are interpolation basis functions localized around each cluster momentum K.

### Methods

| Method | Formula | Properties |
|--------|---------|------------|
| **Nearest-patch** | Σ(k) = Σ(K(k)) (step function) | Discontinuous but causal |
| **Cumulant periodization** | M(k) = Σ_K M(K)·exp(iK·r) | Smooth, preserves causality |
| **Self-energy periodization** | Σ(k) = Σ_K Σ(K)·exp(iK·r) | Smooth but can violate causality |

Cumulant periodization (via M = Σ/(1+Σ·G_0)) is the default — smooth and causal.

### Outputs

- **Σ(k, iω)**: full BZ self-energy on the lattice k-mesh
- **A(k, ω)**: momentum-resolved spectral function (after analytic continuation)
- **Fermi surface**: contour plot of A(k_F, ω=0) identifying arcs, pockets, nesting

---

## Stage 9r: DCA Cluster Two-Particle Vertex

**File**: `dmft/cluster_vertex.py`

The single-site vertex (Stage 9g) misses k-dependent vertex structure. The DCA cluster vertex G²(K₁,K₂; iν,iν',iΩ) captures momentum dependence directly — the pairing vertex at K=(π,0) differs from K=(0,0), which is exactly what drives d-wave pairing.

Cost: 10-100× more expensive than single-site G² because the vertex now has cluster-momentum indices. For N_c=4 with 3 orbitals: G² is a (4×3)⁴ × n_ν² × n_Ω tensor.

### When to Use

- Single-site vertex gives pairing eigenvalue but the k-dependence is approximate (projected from Γ_loc)
- Cluster vertex gives exact k-dependence within DCA resolution
- Use cluster vertex when: single-site λ_pair > 0.5 AND budget allows 10-100× cost increase

---

## Stage 9s: DCA++ GPU Solver Integration

**File**: `dmft/dcaplus_integration.py`, `dmft/Dockerfile.dcaplus`

For cuprates at low T (β > 50), the TRIQS/CTHYB sign problem kills N_c=4 calculations even with density-density interaction. DCA++ from ORNL is GPU-accelerated (CUDA) and uses the CT-AUX algorithm which has better sign properties for the Hubbard model.

### Architecture

Separate Docker image (`qae-dcaplus`) alongside the TRIQS container:
- Base: NVIDIA CUDA 12 + OpenMPI + HDF5
- DCA++ built from source (github.com/CompFUSE/DCA)
- GPU-accelerated CT-AUX solver
- Uses same bundle HDF5 format as TRIQS pipeline

### When DCA++ Is Selected

The orchestrator falls back to DCA++ when:
1. TRIQS/CTHYB ⟨sign⟩ < 0.05 after density-density fallback
2. Material is a cuprate/nickelate with β·U > 40
3. GPU is available (`nvidia-smi` succeeds)

---

## Stage 9t: Adaptive Temperature Grid

**File**: `dmft/adaptive_temperature.py`

Instead of a fixed log-spaced grid, the orchestrator now uses an adaptive strategy:

1. **Phase 1 — Coarse scan**: 3 temperatures (high, mid, low in the expected range)
2. **Phase 2 — Trend detection**: if λ(T) is increasing, estimate T where λ ≈ 0.7
3. **Phase 3 — Refinement**: add 2-3 points densely around the λ ≈ 0.5-0.9 region
4. **Phase 4 — Tc bracket**: if λ crosses 1.0, bisect to locate Tc within ±10K

This halves the compute cost compared to a 10-point fixed grid while achieving better Tc accuracy because points are concentrated where they matter.

---

## Stage 9u: Physics Validation Harness

**File**: `dmft/physics_validation.py`

Four concrete, runnable tests that must pass before trusting any real-material DMFT result. Run with `python3 physics_validation.py --quick` (~1 second without TRIQS).

### V1: 2D Hubbard d-Wave Benchmark

Tests the DCA loop + pairing extraction at U=8t, t'=-0.3t, n=0.85, N_c=4. Assertions:
- d-wave eigenvalue |lambda_d| > |lambda_s| at all temperatures
- lambda_d magnitude increases with decreasing T (20% noise tolerance)
- lambda_d sign is consistent (no flips)
- DCA pairing eigenvalues are nonzero (extraction actually works)

With the Hubbard-I fallback solver (no TRIQS), this tests the entire DCA self-consistency loop, BZ patching, coarse-graining, pairing susceptibility computation, and form-factor projection. The absolute lambda values won't match QMC, but the qualitative physics must be right.

### V2: BSE SVD Cutoff Sensitivity

Constructs a synthetic chi_loc with a known vertex (Gamma = -U), adds realistic noise, and runs BSE inversion at SVD cutoffs 1e-6 through 1e-12. Assertions:
- Gamma_singlet leading eigenvalue sign is stable across all cutoffs
- Gamma_singlet magnitude varies less than 2x across cutoffs
- No cutoff produces a sign flip (which would misidentify attractive vs repulsive channels)

### V3: Kanamori Single-Orbital Limit

Verifies the Kanamori interaction class produces correct results in limiting cases:
- n_orb=1, J=0: U_prime = 0, J_pair = 0 (pure Hubbard, no inter-orbital terms)
- n_orb=2, J=0: U' = U (Kanamori constraint with zero Hund's)
- n_orb=2, J=0.9: U' = U - 2J = 2.2 (proper Kanamori relation)
- n_orb=3: U'[a,b] = U'[b,a] (symmetry)
- Equal-U orbitals: U' = U - 2J for all off-diagonal pairs

If any test fails, there's a sign error in the operator construction that would corrupt multi-orbital DCA results.

### V4: CSC Density Matrix Tail Subtraction

Tests the `compute_density_matrix_from_gf` function against exact Fermi-Dirac values for non-interacting Green's functions:
- Half-filling (eps=0, mu=0): n = 0.5 exactly (to machine precision)
- Off half-filling: n matches Fermi-Dirac to < 0.02
- Multi-orbital: different filling per orbital, all within tolerance
- Low temperature (beta=100): tail correction works where it matters most
- Physical bounds: 0 < n < 1 for all chemical potentials
- Off-diagonal: zero for diagonal G (no spurious orbital mixing)

### Current Status

All four tests **PASS** on the local development machine (without TRIQS, using Hubbard-I fallback for V1).

---

## K-Point and Smearing Convergence

K-mesh density and smearing width directly affect N(E_F) for metals, which propagates into λ (via DOS-weighted e-ph coupling) and μ* (via Morel-Anderson). The pipeline uses stage-dependent and quality-tiered convergence parameters.

### Adaptive K-Mesh (kspacing in Å⁻¹, lower = denser)

| Stage | Screening | Publication (force < 0.001) |
|-------|-----------|---------------------------|
| Relax | 0.40 | 0.40 |
| vc-relax | 0.30 | 0.30 |
| SCF (metal) | 0.20 | **0.15** (aiida "moderate") |
| SCF (insulator) | 0.25 | **0.20** |
| Phonon | tiered by force | tiered by force |
| EPW NSCF | 8-12 per direction | 8-12 per direction |
| EPW fine | up to 40×40×40 | up to 40×40×40 |

Additional modifiers:
- **Metallicity boost**: metals get 1.3× denser grids (kspacing × 0.77)
- **Large cell coarsening**: >8 atoms get 1.15× coarser (already well-sampled by volume)
- **Layered boost**: quasi-2D materials get 1.5× denser in the stacking direction

### Smearing (degauss in Ry)

| Stage | Value | In eV | Method | Notes |
|-------|-------|-------|--------|-------|
| SCF (default) | 0.005 | 68 meV | mv | Tight — good for N(E_F) accuracy |
| vc-relax (non-mag) | 0.015 | 204 meV | mv | Wider for convergence stability |
| vc-relax (magnetic) | 0.020 | 272 meV | mv | Extra width for spin stability |
| EPW NSCF | 0.020 | 272 meV | cold | Marzari-Vanderbilt cold smearing |
| SCF retry (diverging) | 0.030 | 408 meV | mp | Emergency Methfessel-Paxton |

**Key insight**: For publication materials, the N(E_F) that matters for λ and μ* comes from three independent sources:
1. **SCF** (degauss=0.005, mv) — used by the semi-empirical physics engine
2. **EPW** — recomputes electronic structure on ultra-dense k-grids with proper BZ integration
3. **ACBN0** — runs its own DFT+U SCF with hp.x for screening parameters

The vc-relax smearing (0.015-0.02) is intentionally loose for convergence — it only affects the screening-tier N(E_F). Publication-grade analysis always uses tighter parameters from downstream stages.

---

## Stage 10: Results → Database → Next Iteration

Extended dataset fields: tcConservative, tcUpperBound, tcMethod, lambdaMethod, phononMethod, tcConfidence, learningScore, qualityTier, hullLabel, residualForce, nqeApplied, nqeMethod, lambdaNQE, lambdaReduction, nqeAnharmonicStrength, nqeStabilityShift, muStarMethod, muStarConventional, muStarDeviation, muStarTcSensitivity, epwLambda, epwTcME, epwGapZero, epwMethod, socEnabled, socMaxEnergy, socDosImpact, magneticOrdering, magneticEnergyGap, magneticMagnetization, hubbardApplied, hubbardCorrelatedSites, hubbardRegime, hubbardAppliedToVCRelax, sschaConverged, sschaOmegaLog, sschaTcCorrected, acbn0Converged, acbn0MuStar, acbn0Method, epwConverged, epwLambda, epwTcME, epwMethod, pairingChannel, pairingSymmetry, spinFluctuationLambda, spinFluctuationTc, tcCombined, dmftBundleExported, dmftCorrelatedShells, dmftCorrelatedOrbitals, dmftBundleFormat, dmftConverged, dmftVertexMeasured, dmftLambdaPair, dmftGapSymmetry, dmftGapNodes, dmftIsUnconventional, dmftTcBSE, dmftTcBSEConfidence.

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

For every non-failed candidate: quality_report.json, candidate_provenance.json, final_structure.poscar, scf_summary.json, phonon_summary.json, dfpt_results.json, epw_results.json (when EPW runs), sscha_results.json (when SSCHA runs), acbn0_results.json (when ACBN0 runs).

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

- **DFT worker** (`instance-20260502-184120`): c2-standard-30 (30 vCPUs, 120 GB), Debian 12, us-central1-c
  - QE 7.3.1 (lmaxx=6), EPW, QE_MPI_RANKS=24, QE_NPOOL=6
  - Exports DMFT bundles and uploads to gnn-training via HTTP
  - `DMFT_SERVICE_URL=http://34.130.121.199:8780`
- **GNN + DMFT worker** (`gnn-training`): g2-standard-4 (4 vCPUs, 16 GB, 1x NVIDIA L4 GPU, 100GB disk)
  - Python GNN service on port 8765 (predictions + training on L4 GPU)
  - DMFT Docker container on port 8780 (TRIQS/CTHYB, 2 MPI ranks, 10GB memory limit)
  - Startup order: DMFT container first (T+0), GNN loop (T+2s), DFT loop (T+5s)
  - DMFT and GNN training run in parallel — DMFT is CPU-only (2 cores), GNN is GPU-only
  - DMFT bundles uploaded as multipart file (not path reference — cross-VM)
  - GCP firewall rule `allow-dmft-8780` needed: `tcp:8780` from internal VMs
- Both pull from shared Neon DB job queue
- QE binary search prefers `/usr/local/bin` (manual lmaxx=6 rebuild) over `/usr/bin` (apt default)
- Supported elements: nearly full periodic table. La, Ce, Th, Pr-Tm, Pa, U, Np all supported via lmaxx=6 + Pseudo-DOJO PPs. Only Pu, Am blocked (no reliable PPs).

---

## Known Physics Gaps & Roadmap

### Current Limitations

**1. SSCHA compute cost**
Full SSCHA requires 150-400 DFT force calculations per material (12-200 hours). Only practical for the very best candidates (force < 0.001, H-rich, P ≥ 20 GPa). The semi-empirical SSCHA-model correction (Stage 9b) is applied to all hydrides as a fast approximation; full SSCHA (Stage 9c) runs only when compute budget allows.

**2. ACBN0 accuracy vs. SCDFT**
The hp.x-based ACBN0 μ* is more accurate than fixed μ*=0.10-0.13 but still relies on RPA screening. Full SCDFT (superconducting DFT) would give the most accurate μ* but requires specialized codes not yet integrated.

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
| **Phase 5** | Spin-orbit coupling (SOC analysis + noncolin/lspinorb for heavy elements) | **DONE** |
| **Phase 6** | Magnetic ground-state search (FM/AFM/NM energy comparison before phonons) | **DONE** |
| **Phase 7** | DFT+U Hubbard workflow (composition-aware U, vc-relax integration, broadened triggers) | **DONE** |
| **Phase 8** | Full SSCHA anharmonic phonons (sscha-pipeline.ts + sscha-worker.py) | **DONE** |
| **Phase 9** | ACBN0 first-principles μ* via hp.x + self-consistent U feedback to Hubbard workflow | **DONE** |
| **Phase 10** | K-mesh/smearing convergence tiering for publication | **DONE** |
| **Phase 11** | Zone-boundary soft mode following (phonon-guided structure search) | **DONE** |
| **Phase 12** | LLM structure advisor (gpt-4o-mini hints for CSP generation) | **DONE** |
| **Phase 13** | Spin-fluctuation pairing channel (Lindhard+RPA χ, I²χ, combined Tc) | **DONE** |
| **Phase 14** | Pressure-priority refinement + dual convergence gate (force + pressure) | **DONE** |
| **Phase 15** | DFT atom limit raised to 24 + atom-scaled timeouts/grids | **DONE** |
| **Phase 16** | Liechtenstein DFT+U (kind=1 with Hund's J for nickelates/ruthenates) | **DONE** |
| **Phase 17** | DMFT bundle export + Wannier90 projector mode + TRIQS Docker infrastructure | **DONE** |
| **Phase 18** | Two-particle vertex (G² measurement + local BSE + Γ_loc extraction) | **DONE** |
| **Phase 19** | Pairing susceptibility (λ_pair eigenvalue solver + gap symmetry classification) | **DONE** |
| **Phase 20** | DCA cluster DMFT (N_c=4 DCA solver, BZ patching, cluster CTHYB) | **DONE** |
| **Phase 21** | 2D Hubbard d-wave benchmark (Maier-Jarrell-Scalapino calibration gate) | **DONE** |
| **Phase 22** | Multi-orbital DCA cluster (3-band Emery, 5-band d-shell, Kanamori interaction) | **DONE** |
| **Phase 23** | Charge self-consistent DFT+DMFT (density feedback loop DFT↔DMFT) | **DONE** |
| **Phase 24** | Realistic-model pairing pipeline (CSC + multi-orbital DCA → Tc) | **DONE** |
| **Phase 25** | Pipeline orchestrator (auto cluster selection, convergence gates, fallback chains, resource accounting) | **DONE** |
| **Phase 26** | Production integration (Wannier90 execution, HTTP submit, schema, sign extraction, CSC callbacks) | **DONE** |
| **Phase 27** | Analytic continuation (MaxEnt + Padé for Σ(iω)→Σ(ω) real-axis spectral functions) | **DONE** |
| **Phase 28** | DMFT-corrected Tc (DMFT spectral function → N(E_F) → revised Allen-Dynes) | **DONE** |
| **Phase 29** | DCA self-energy periodization (Σ(K)→Σ(k) on full BZ for Fermi surface plots) | **DONE** |
| **Phase 30** | DCA cluster two-particle vertex (G² on cluster for k-dependent pairing) | **DONE** |
| **Phase 31** | DCA++ GPU solver integration (CUDA CT-AUX for low-T cuprates) | **DONE** |
| **Phase 32** | Adaptive temperature grid (coarse scan -> refine around lam~0.5-0.9 -> Tc bisection) | **DONE** |
| **Phase 33** | Physics validation harness (Hubbard benchmark, BSE sensitivity, Kanamori limits, CSC tail) | **DONE** |
| **Phase 34** | SCDFT (superconducting DFT) for beyond-RPA mu* | Future (specialized code) |
| **Phase 35** | Path-integral MD for NQE beyond SSCHA | Future (PIMD integration) |
