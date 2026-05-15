import * as fs from "fs";
import * as path from "path";

export interface DFPTAlpha2FParsed {
  frequencies: number[];
  alpha2F: number[];
  lambda: number;
  omegaLog: number;
  /** Allen-Dynes spectral second moment ⟨ω²⟩ = (2/λ)·∫ω·α²F(ω) dω, in
   *  (cm⁻¹)². NOTE: this is in cm⁻¹ units even though `omegaLog` (above) is
   *  in K — the two have intentionally different conventions because the
   *  canonical Allen-Dynes f₂ formula in `tc-formulas.ts` expects
   *  `omega2Avg` in (cm⁻¹)² and computes √omega2Avg/omegaLog_cm internally.
   *  Used by the f₂ strong-coupling correction (5–15% Tc boost for hydrides
   *  with √⟨ω²⟩/ω_log > 1.2). Returns 0 when not computable. */
  omega2Avg?: number;
  nqPoints: number;
  source: "lambda.x" | "matdyn" | "reconstructed";
  unstableStructure?: boolean;
}

export interface DFPTPhononDOS {
  frequencies: number[];
  dos: number[];
  totalStates: number;
  maxFrequency: number;
  hasImaginaryModes: boolean;
}

export interface DFPTDynmatResult {
  qPoint: [number, number, number];
  frequencies: number[];
  modes: { frequency: number; irRep: string; activity: string }[];
  dielectricTensor: number[][] | null;
  bornCharges: { atom: string; tensor: number[][] }[] | null;
}

export interface DFPTPipelineFiles {
  prefix: string;
  scfOutput?: string;
  phOutput?: string;
  dynFiles?: string[];
  dosFile?: string;
  alpha2fFile?: string;
  lambdaOutput?: string;
}

export function generatePhononGridInput(
  prefix: string,
  nq1: number = 4,
  nq2: number = 4,
  nq3: number = 4
): string {
  return `Phonon dispersions on ${nq1}x${nq2}x${nq3} grid
&INPUTPH
  prefix = '${prefix}',
  outdir = './tmp',
  fildyn = '${prefix}.dyn',
  fildvscf = '${prefix}.dvscf',
  tr2_ph = 1.0d-14,
  ldisp = .true.,
  nq1 = ${nq1}, nq2 = ${nq2}, nq3 = ${nq3},
  electron_phonon = 'interpolated',
  el_ph_sigma = 0.005,
  el_ph_nsigma = 10,
/
`;
}

export function generateQ2RInput(prefix: string, nq1: number = 4, nq2: number = 4, nq3: number = 4): string {
  return `&INPUT
  fildyn = '${prefix}.dyn',
  zasr = 'crystal',
  flfrc = '${prefix}.fc',
/
`;
}

export function generateMatdynDOSInput(
  prefix: string,
  nk1: number = 20,
  nk2: number = 20,
  nk3: number = 20
): string {
  return `&INPUT
  asr = 'crystal',
  flfrc = '${prefix}.fc',
  flvec = '${prefix}.modes',
  dos = .true.,
  fldos = '${prefix}.phdos',
  nk1 = ${nk1}, nk2 = ${nk2}, nk3 = ${nk3},
  deltaE = 1.0,
/
`;
}

export function generateMatdynDispersionInput(prefix: string): string {
  return `&INPUT
  asr = 'crystal',
  flfrc = '${prefix}.fc',
  flvec = '${prefix}.modes',
  flfrq = '${prefix}.freq',
  q_in_band_form = .true.,
/
6
0.0  0.0  0.0  20
0.5  0.0  0.0  20
0.5  0.5  0.0  20
0.0  0.0  0.0  20
0.5  0.5  0.5  20
0.5  0.0  0.0  1
`;
}

export function generateLambdaInput(prefix: string, muStar: number = 0.10): string {
  return `10
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
1
${muStar}
`;
}

export function parsePhDynmatOutput(stdout: string): DFPTDynmatResult[] {
  const results: DFPTDynmatResult[] = [];
  const qBlocks = stdout.split(/Dynamical matrix at q\s*=/);

  for (let i = 1; i < qBlocks.length; i++) {
    const block = qBlocks[i];
    const qMatch = block.match(/\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)/);
    if (!qMatch) continue;

    const qPoint: [number, number, number] = [
      parseFloat(qMatch[1]),
      parseFloat(qMatch[2]),
      parseFloat(qMatch[3]),
    ];

    const freqMatches = block.matchAll(/freq\s*\(\s*\d+(?:\s*-\s*\d+)?\s*\)\s*=\s*([-\d.]+)\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/g);
    const frequencies: number[] = [];
    const modes: { frequency: number; irRep: string; activity: string }[] = [];

    for (const m of freqMatches) {
      const freqCm1 = parseFloat(m[2]);
      frequencies.push(freqCm1);
      modes.push({
        frequency: freqCm1,
        irRep: "",
        activity: freqCm1 < -50 ? "imaginary" : "real",
      });
    }

    let dielectricTensor: number[][] | null = null;
    const epsMatch = block.match(/Dielectric Tensor:(.+?)(?=Born|Dynamical|$)/s);
    if (epsMatch) {
      const rows = epsMatch[1].trim().split("\n").slice(0, 3);
      dielectricTensor = rows.map(row => {
        const vals = row.trim().split(/\s+/).map(parseFloat).filter(v => Number.isFinite(v));
        return vals.length >= 3 ? vals.slice(0, 3) : [0, 0, 0];
      }).filter(r => r.length === 3);
      if (dielectricTensor.length !== 3) dielectricTensor = null;
    }

    results.push({ qPoint, frequencies, modes, dielectricTensor, bornCharges: null });
  }

  return results;
}

export function parseMatdynDOS(dosContent: string): DFPTPhononDOS {
  const frequencies: number[] = [];
  const dos: number[] = [];
  let hasImaginary = false;

  const lines = dosContent.trim().split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#") || trimmed.length === 0) continue;
    const parts = trimmed.split(/\s+/);
    if (parts.length >= 2) {
      const freq = parseFloat(parts[0]);
      const density = parseFloat(parts[1]);
      if (Number.isFinite(freq) && Number.isFinite(density)) {
        frequencies.push(freq);
        dos.push(Math.max(0, density));
        if (freq < -20) hasImaginary = true;
      }
    }
  }

  // totalStates = ∫ DOS(ω) dω (number of phonon modes, should be ~3·N_atoms).
  // The previous `dos.reduce(s + d)` summed RAW DOS values without bin width
  // — for matdyn's default deltaE=1 cm⁻¹ this happened to be numerically
  // close to the true integral, but for deltaE=2 it was off by 2×, etc.
  // Proper trapezoidal integration on the frequency grid:
  let totalStates = 0;
  for (let i = 0; i < dos.length; i++) {
    const dw = dynamicBinWidth(frequencies, i);
    totalStates += dos[i] * dw;
  }
  const maxFreq = frequencies.length > 0 ? Math.max(...frequencies) : 0;

  return { frequencies, dos, totalStates, maxFrequency: maxFreq, hasImaginaryModes: hasImaginary };
}

function dynamicBinWidth(freqs: number[], i: number): number {
  if (freqs.length < 2) return 1;
  // Trapezoidal weights: ∫f dx ≈ Σ w_i f(x_i)
  //   endpoint w_0 = (x_1 - x_0)/2
  //   interior w_i = (x_{i+1} - x_{i-1})/2
  //   endpoint w_N = (x_N - x_{N-1})/2
  // Previously endpoints used the full neighbor gap (no /2), which double-counted
  // the endpoint contribution. Negligible for typical phonon DOS where the
  // edges have ~zero density, but matters for α²F integrals because lambda.x
  // a2F output often has nonzero α²F at the very first positive frequency bin.
  if (i === 0) return (freqs[1] - freqs[0]) / 2;
  if (i === freqs.length - 1) return (freqs[i] - freqs[i - 1]) / 2;
  return (freqs[i + 1] - freqs[i - 1]) / 2;
}

// Threshold (cm⁻¹) below which a negative phonon frequency in the α²F grid
// counts as a GENUINE dynamic instability rather than numerical noise.
// matdyn/lambda.x α²F grids routinely dip a few cm⁻¹ below zero from ASR
// residuals and Gaussian-broadening tails around the Γ acoustic modes; a
// bare `freq < 0` test flagged perfectly stable materials as unstable and
// zeroed their λ. Matches the imaginary-mode threshold in parsePhDynmatOutput.
const SIGNIFICANT_IMAG_CM1 = 50;

export function parseAlpha2FOutput(content: string): DFPTAlpha2FParsed {
  const frequencies: number[] = [];
  const alpha2F: number[] = [];
  let lambda = 0;
  let omegaLog = 0;
  let nqPoints = 0;
  let hasImaginaryModes = false;

  const lines = content.trim().split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#") || trimmed.length === 0) continue;

    const lambdaMatch = trimmed.match(/lambda\s*=\s*([\d.]+)/i);
    if (lambdaMatch) {
      lambda = parseFloat(lambdaMatch[1]);
      continue;
    }

    const omegaLogMatch = trimmed.match(/omega_log\s*=\s*([\d.]+)/i);
    if (omegaLogMatch) {
      omegaLog = parseFloat(omegaLogMatch[1]);
      continue;
    }

    const nqMatch = trimmed.match(/nq\s*=\s*(\d+)/i);
    if (nqMatch) {
      nqPoints = parseInt(nqMatch[1]);
      continue;
    }

    const parts = trimmed.split(/\s+/);
    if (parts.length >= 2) {
      const freq = parseFloat(parts[0]);
      const a2f = parseFloat(parts[1]);
      if (!Number.isFinite(freq) || !Number.isFinite(a2f)) continue;
      if (freq < 0) {
        // Negative bins are always skipped from the arrays and the
        // λ/ω_log integrals, but only a SIGNIFICANT imaginary mode marks
        // the structure dynamically unstable (→ early return, λ=0). A
        // few-cm⁻¹ negative bin is ASR/broadening noise around the Γ
        // acoustic modes, not an instability — flagging it discarded the
        // DFPT result of stable materials.
        if (freq < -SIGNIFICANT_IMAG_CM1) hasImaginaryModes = true;
        continue;
      }
      frequencies.push(freq);
      alpha2F.push(Math.max(0, a2f));
    }
  }

  if (hasImaginaryModes) {
    return {
      frequencies,
      alpha2F,
      lambda: 0,
      omegaLog: 0,
      nqPoints,
      source: frequencies.length > 0 ? "lambda.x" : "reconstructed",
      unstableStructure: true,
    };
  }

  const minLen = Math.min(frequencies.length, alpha2F.length);
  if (frequencies.length !== alpha2F.length) {
    console.warn(`[DFPT] frequencies/alpha2F length mismatch: ${frequencies.length} vs ${alpha2F.length}, truncating to ${minLen}`);
    frequencies.length = minLen;
    alpha2F.length = minLen;
  }

  const LOW_FREQ_CUTOFF = 1.0;
  const LOG_FLOOR = 1e-3;

  if (lambda === 0 && minLen > 0) {
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] < LOW_FREQ_CUTOFF || alpha2F[i] <= 0) continue;
      const dw = dynamicBinWidth(frequencies, i);
      lambda += 2 * alpha2F[i] / frequencies[i] * dw;
    }
  }

  const CM1_TO_K = 1.4387768775039338;  // hc/k_B in K·cm

  if (omegaLog === 0 && lambda > 0 && frequencies.length > 0) {
    let logSum = 0;
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] < LOW_FREQ_CUTOFF || alpha2F[i] <= 0) continue;
      const dw = dynamicBinWidth(frequencies, i);
      logSum += (alpha2F[i] / frequencies[i]) * Math.log(Math.max(frequencies[i], LOG_FLOOR)) * dw;
    }
    omegaLog = Math.exp((2 / lambda) * logSum);
    if (!Number.isFinite(omegaLog) || omegaLog < 0) omegaLog = 0;
    // Convert from cm⁻¹ (the unit of `frequencies` parsed from .a2f) to K
    // so the returned ω_log matches the project-wide convention. The
    // consumer at qe-worker:4774 overwrites `omegaLog` with this value
    // and then plugs it directly into Allen-Dynes (Tc[K] = ω_log[K]/1.2 ·…),
    // which only gives Tc in K if ω_log is in K. Previously this was in
    // cm⁻¹, so the fallback Allen-Dynes Tc was off by a factor of 1/1.4388
    // (~30% under-predict) whenever the a2f file path was used.
    if (omegaLog > 0) {
      omegaLog *= CM1_TO_K;
    }
  }

  // Compute ⟨ω²⟩ for the Allen-Dynes f₂ correction. Formula:
  //   ⟨ω²⟩ = (2/λ) · ∫ ω · α²F(ω) dω
  // Output in (cm⁻¹)² — DELIBERATELY keeps cm⁻¹ units because the canonical
  // Allen-Dynes consumers in tc-formulas.ts expect omega2Avg in (cm⁻¹)² and
  // do the K↔cm⁻¹ conversion internally. (omegaLog above is in K because
  // qe-worker.ts:4774 plugs it directly into a Tc[K] formula. The two
  // unit conventions are intentional and documented.)
  let omega2AvgCm2 = 0;
  if (lambda > 0 && frequencies.length > 0) {
    let omega2Sum = 0;
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] < LOW_FREQ_CUTOFF || alpha2F[i] <= 0) continue;
      const dw = dynamicBinWidth(frequencies, i);
      omega2Sum += alpha2F[i] * frequencies[i] * dw;
    }
    omega2AvgCm2 = (2 / lambda) * omega2Sum;
    if (!Number.isFinite(omega2AvgCm2) || omega2AvgCm2 < 0) omega2AvgCm2 = 0;
  }

  return {
    frequencies,
    alpha2F,
    lambda: Number(lambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),  // in K (project convention)
    omega2Avg: omega2AvgCm2 > 0 ? Number(omega2AvgCm2.toFixed(4)) : 0,  // in (cm⁻¹)²
    nqPoints,
    source: frequencies.length > 0 ? "lambda.x" : "reconstructed",
  };
}

export function parseLambdaOutput(stdout: string): {
  lambda: number;
  omegaLog: number;
  tc: number[];
  tcCorrected: number[];
  muStarValues: number[];
  strongCoupling: boolean;
} {
  let lambda = 0;
  let omegaLog = 0;
  const tc: number[] = [];
  const muStarValues: number[] = [];

  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (lambda === 0) {
      // QE 7.x ph.x electron_phonon='interpolated' format (with `=`):
      //   "Electron-phonon coupling constant =   1.234"
      //   "lambda =   1.234   omega_log =   800.0 K"
      // Older QE format (no `=` after the keyword):
      //   "Electron-phonon coupling constant is    1.234"
      //   "     lambda     1.234   omega_log=   800.00 K"
      // lambda.x (always uses `=`):
      //   "lambda =   1.234"
      // The two original regexes only matched the `=`-form, so on older QE
      // builds (and on the `is`-form newer prints) lambda stayed 0 and the
      // Allen-Dynes Tc fallback was silently skipped. Cover both forms.
      // First regex handles all three "Electron-phonon coupling constant"
      // header variants emitted by different QE versions:
      //   "Electron-phonon coupling constant = 1.234"     (newer QE, `=`)
      //   "Electron-phonon coupling constant is 1.234"    (older QE, `is`)
      //   "Electron-phonon coupling constant lambda is 1.234"  (some QE 6.x)
      // Second regex handles the standalone summary line in newer QE 7.x:
      //   "lambda = 1.234   omega_log = ..."     (with =)
      //   "lambda    1.234   omega_log= ..."     (older, no =)
      // `\s+` after `\blambda` is mandatory whitespace so the regex doesn't
      // match per-q-point forms like `lambda(1)=` or `lambda_av=` that share
      // the substring but have different semantics. `(?:[=:]\s*)?` then
      // optionally consumes any `=`/`:` so the same regex covers both forms.
      const lm =
        trimmed.match(/Electron-phonon coupling constant(?:\s+lambda)?\s*(?:is|=|:)?\s*([\d.]+)/i) ||
        trimmed.match(/\blambda\s+(?:[=:]\s*)?([\d.]+)/i);
      if (lm) lambda = parseFloat(lm[1]);
    }

    if (omegaLog === 0) {
      // ph.x (newer 7.x): "Logarithmic average frequency (t.s.) =   1000.00 K"
      // ph.x (older 6.x): "lambda  0.85  omega_log=  800.00 K"  (no space before =)
      // ph.x (oldest):    "lambda  0.85  omega_log   800.00 K"  (no = at all)
      // ph.x (alt name):  "omega_ln =   800.00 K"  (some QE versions)
      // lambda.x:         "omega_log (K) =   800.0"  (K in parens before =)
      // EPW 4.x:          "omega_log (meV) = 80.0"
      //
      // The previous regex required `=` between `omega_log` and the value
      // — older QE builds without the `=` (paired with the `lambda` bare-
      // whitespace summary line fixed above) silently returned omegaLog=0,
      // which combined with the lambda=0 bug from iteration 116 to make
      // the entire DFPT Allen-Dynes fallback skip. `(?:[=:]\s*)?` makes
      // the `=` optional so the regex covers all four ph.x variants.
      //
      // Unit-aware: explicitly inspect the parenthesized unit (meV / cm⁻¹
      // / THz / K) and convert. No annotation defaults to Kelvin per QE
      // default. ph.x's "Logarithmic average frequency" form is always K.
      const phMatch = trimmed.match(/Logarithmic average frequency[^=]*=\s*([\d.]+)/i);
      if (phMatch) {
        omegaLog = parseFloat(phMatch[1]);  // ph.x always Kelvin
      } else {
        const lambdaMatch = trimmed.match(/omega_l(?:og|n)\s*(?:\(([^)]*)\)\s*)?(?:[=:]\s*)?([\d.]+)/i);
        if (lambdaMatch) {
          const unitRaw = (lambdaMatch[1] ?? "").toLowerCase().trim();
          let val = parseFloat(lambdaMatch[2]);
          if (unitRaw.includes("mev")) {
            val *= 11.6045181;       // meV → K
          } else if (unitRaw.includes("cm-1") || unitRaw.includes("cm^-1") || unitRaw === "cm") {
            val *= 1.4387768775;     // cm⁻¹ → K
          } else if (unitRaw.includes("thz")) {
            val *= 47.9924;          // THz → K (hf/k_B with f in THz)
          }
          // No annotation OR (K)/(t.s.)/etc. → treat as Kelvin per QE default
          omegaLog = val;
        }
      }
    }

    // lambda.x table: "mu* =  0.10   Tc =   55.5 K" or "0.10  55.5"
    const tcm =
      trimmed.match(/mu\*?\s*=\s*([\d.]+)\s+Tc\s*=\s*([\d.]+)/i) ||
      (muStarValues.length === 0 && trimmed.match(/^(0\.\d+)\s+([\d.]+)\s*K?\s*$/));
    if (tcm) {
      muStarValues.push(parseFloat(tcm[1]));
      tc.push(parseFloat(tcm[2]));
    }
  }

  // "strongCoupling" is a user-facing classification only (λ > 1.5 = clearly
  // beyond McMillan regime). It does NOT gate whether the Allen-Dynes f1
  // correction is applied — f1 is well-defined and reduces continuously to 1
  // for small λ, so always computing it is safe and more accurate than
  // returning the McMillan result from lambda.x for marginal coupling
  // (1.0 < λ < 1.5), where f1 contributes a 5-10% Tc boost that was
  // previously being suppressed.
  const strongCoupling = lambda > 1.5;
  const tcCorrected: number[] = [];

  // omegaLog from QE is always in Kelvin — no unit conversion needed.
  if (omegaLog > 0 && lambda > 0.01) {
    for (const muStar of muStarValues) {
      // Allen-Dynes f1 prefactor with the μ*-dependent Λ₁ = 2.46·(1 + 3.8·μ*).
      // Λ₁ depends on μ* (Allen & Dynes, PRB 12, 905 (1975), Eq. 3.3) so it
      // must be computed inside the μ* loop, not once outside.
      const lambdaBar = 2.46 * (1 + 3.8 * muStar);
      const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 1.5), 1 / 3);

      const denom = lambda - muStar * (1 + 0.62 * lambda);
      if (denom <= 0) { tcCorrected.push(0); continue; }
      const exponent = -1.04 * (1 + lambda) / denom;
      if (exponent < -50) { tcCorrected.push(0); continue; }
      let tcAD = (omegaLog / 1.2) * f1 * Math.exp(exponent);
      // Don't cap Tc — theoretical hydrides at extreme pressure can predict
      // > 500 K from Allen-Dynes. A hard cap silently corrupts the ML training
      // signal for the exact materials we care about discovering. Only filter
      // negative (unphysical denominator) values.
      tcAD = Math.max(0, tcAD);
      if (tcAD > 500) {
        console.warn(`[Allen-Dynes] Tc = ${tcAD.toFixed(1)} K > 500 K predicted ` +
          `(λ=${lambda.toFixed(2)}, ω_log=${omegaLog.toFixed(0)} K, μ*=${muStar}) — ` +
          `likely a strong-coupling extreme; verify by Eliashberg if available.`);
      }
      tcCorrected.push(Number(tcAD.toFixed(2)));
    }
  }

  return {
    lambda: Number(lambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),
    tc,
    // Always return the f1-corrected Allen-Dynes Tc when available — it
    // is strictly more accurate than the McMillan Tc that lambda.x prints
    // (which is what `tc` holds). For very small λ the two are
    // numerically nearly identical.
    tcCorrected: tcCorrected.length > 0 ? tcCorrected : tc,
    muStarValues,
    strongCoupling,
  };
}

function countFormulaAtoms(formula: string): number {
  let total = 0;
  const re = /[A-Z][a-z]?(\d*)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(formula)) !== null) {
    total += m[1] ? parseInt(m[1], 10) : 1;
  }
  return Math.max(total, 1);
}

function scaleQGrid(nAtoms: number): [number, number, number] {
  if (nAtoms <= 2) return [6, 6, 6];
  if (nAtoms <= 4) return [4, 4, 4];
  if (nAtoms <= 8) return [3, 3, 3];
  return [2, 2, 2];
}

export function buildDFPTJobSpec(
  formula: string,
  prefix: string,
  nq?: [number, number, number],
  nkDos: [number, number, number] = [20, 20, 20],
  muStar: number = 0.10
): {
  phInput: string;
  q2rInput: string;
  matdynDosInput: string;
  matdynDispInput: string;
  lambdaInput: string;
  stages: string[];
  nqUsed: [number, number, number];
} {
  const effectiveNq = nq ?? scaleQGrid(countFormulaAtoms(formula));
  return {
    phInput: generatePhononGridInput(prefix, effectiveNq[0], effectiveNq[1], effectiveNq[2]),
    q2rInput: generateQ2RInput(prefix, effectiveNq[0], effectiveNq[1], effectiveNq[2]),
    matdynDosInput: generateMatdynDOSInput(prefix, nkDos[0], nkDos[1], nkDos[2]),
    matdynDispInput: generateMatdynDispersionInput(prefix),
    lambdaInput: generateLambdaInput(prefix, muStar),
    stages: ["scf", "ph.x", "q2r.x", "matdyn.x (dos)", "matdyn.x (disp)", "lambda.x"],
    nqUsed: effectiveNq,
  };
}

async function tryReadFile(filePath: string): Promise<string | null> {
  try {
    return await fs.promises.readFile(filePath, "utf-8");
  } catch {
    return null;
  }
}

export async function tryLoadDFPTResults(jobDir: string, prefix: string): Promise<{
  alpha2F: DFPTAlpha2FParsed | null;
  phononDOS: DFPTPhononDOS | null;
  dynmat: DFPTDynmatResult[] | null;
}> {
  let alpha2F: DFPTAlpha2FParsed | null = null;
  let phononDOS: DFPTPhononDOS | null = null;
  let dynmat: DFPTDynmatResult[] | null = null;

  const a2fContent = await tryReadFile(path.join(jobDir, `${prefix}.a2F.dat`));
  if (a2fContent) {
    alpha2F = parseAlpha2FOutput(a2fContent);
  }

  if (!alpha2F) {
    const altA2fPaths = [
      path.join(jobDir, "a2F.dos1"),
      path.join(jobDir, "a2F.dos"),
      path.join(jobDir, `${prefix}.a2f`),
    ];
    for (const p of altA2fPaths) {
      const content = await tryReadFile(p);
      if (content) {
        alpha2F = parseAlpha2FOutput(content);
        if (alpha2F.frequencies.length > 0) break;
        alpha2F = null;
      }
    }
  }

  const dosContent = await tryReadFile(path.join(jobDir, `${prefix}.phdos`));
  if (dosContent) {
    phononDOS = parseMatdynDOS(dosContent);
  }

  const phOutContent = await tryReadFile(path.join(jobDir, "ph.out"));
  if (phOutContent) {
    dynmat = parsePhDynmatOutput(phOutContent);
  }

  if (alpha2F && phononDOS?.hasImaginaryModes) {
    alpha2F.unstableStructure = true;
  }

  return { alpha2F, phononDOS, dynmat };
}
