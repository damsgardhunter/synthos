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

Extended dataset fields: tcConservative, tcUpperBound, tcMethod, lambdaMethod, phononMethod, tcConfidence, learningScore, qualityTier, hullLabel, residualForce, nqeApplied, nqeMethod, lambdaNQE, lambdaReduction, nqeAnharmonicStrength, nqeStabilityShift, muStarMethod, muStarConventional, muStarDeviation, muStarTcSensitivity, epwLambda, epwTcME, epwGapZero, epwMethod, socEnabled, socMaxEnergy, socDosImpact, magneticOrdering, magneticEnergyGap, magneticMagnetization, hubbardApplied, hubbardCorrelatedSites, hubbardRegime, hubbardAppliedToVCRelax, sschaConverged, sschaOmegaLog, sschaTcCorrected, acbn0Converged, acbn0MuStar, acbn0Method, epwConverged, epwLambda, epwTcME, epwMethod, pairingChannel, pairingSymmetry, spinFluctuationLambda, spinFluctuationTc, tcCombined.

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

- **Old worker**: c2-standard-8 (8 vCPUs, 32 GB), QE 7.3.1 (lmaxx=6), EPW, QE_MPI_RANKS=3
- **New worker**: c2-standard-30 (30 vCPUs, 120 GB), QE 7.3.1 (lmaxx=6), EPW, QE_MPI_RANKS=24, QE_NPOOL=6
- Both pull from shared Neon DB job queue
- 2 GCP VMs processing materials in parallel
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
| **Phase 17** | SCDFT (superconducting DFT) for beyond-RPA μ* | Future (specialized code) |
| **Phase 18** | Path-integral MD for NQE beyond SSCHA | Future (PIMD integration) |
