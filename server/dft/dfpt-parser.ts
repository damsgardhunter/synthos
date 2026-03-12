import * as fs from "fs";
import * as path from "path";

export interface DFPTAlpha2FParsed {
  frequencies: number[];
  alpha2F: number[];
  lambda: number;
  omegaLog: number;
  nqPoints: number;
  source: "lambda.x" | "matdyn" | "reconstructed";
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

    const freqMatches = block.matchAll(/freq\s*\(\s*\d+\)\s*=\s*([-\d.]+)\s*\[THz\]\s*=\s*([-\d.]+)\s*\[cm-1\]/g);
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

  const totalStates = dos.reduce((s, d) => s + d, 0);
  const maxFreq = frequencies.length > 0 ? Math.max(...frequencies) : 0;

  return { frequencies, dos, totalStates, maxFrequency: maxFreq, hasImaginaryModes: hasImaginary };
}

function dynamicBinWidth(freqs: number[], i: number): number {
  if (freqs.length < 2) return 1;
  if (i === 0) return freqs[1] - freqs[0];
  if (i === freqs.length - 1) return freqs[i] - freqs[i - 1];
  return (freqs[i + 1] - freqs[i - 1]) / 2;
}

export function parseAlpha2FOutput(content: string): DFPTAlpha2FParsed {
  const frequencies: number[] = [];
  const alpha2F: number[] = [];
  let lambda = 0;
  let omegaLog = 0;
  let nqPoints = 0;

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
      if (Number.isFinite(freq) && Number.isFinite(a2f) && freq >= 0) {
        frequencies.push(freq);
        alpha2F.push(Math.max(0, a2f));
      }
    }
  }

  const LOW_FREQ_CUTOFF = 1.0;
  const LOG_FLOOR = 1e-3;

  if (lambda === 0 && frequencies.length > 0 && alpha2F.length > 0) {
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] < LOW_FREQ_CUTOFF || alpha2F[i] <= 0) continue;
      const dw = dynamicBinWidth(frequencies, i);
      lambda += 2 * alpha2F[i] / frequencies[i] * dw;
    }
  }

  if (omegaLog === 0 && lambda > 0 && frequencies.length > 0) {
    let logSum = 0;
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] < LOW_FREQ_CUTOFF || alpha2F[i] <= 0) continue;
      const dw = dynamicBinWidth(frequencies, i);
      logSum += (alpha2F[i] / frequencies[i]) * Math.log(Math.max(frequencies[i], LOG_FLOOR)) * dw;
    }
    omegaLog = Math.exp((2 / lambda) * logSum);
    if (!Number.isFinite(omegaLog) || omegaLog < 0) omegaLog = 0;
  }

  return {
    frequencies,
    alpha2F,
    lambda: Number(lambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),
    nqPoints,
    source: frequencies.length > 0 ? "lambda.x" : "reconstructed",
  };
}

export function parseLambdaOutput(stdout: string): {
  lambda: number;
  omegaLog: number;
  tc: number[];
  muStarValues: number[];
} {
  let lambda = 0;
  let omegaLog = 0;
  const tc: number[] = [];
  const muStarValues: number[] = [];

  const lambdaMatch = stdout.match(/lambda\s*=\s*([\d.]+)/i);
  if (lambdaMatch) lambda = parseFloat(lambdaMatch[1]);

  const omLogMatch = stdout.match(/omega_log\s*=\s*([\d.]+)\s*\(K\)/i) ||
    stdout.match(/omega_log\s*=\s*([\d.]+)/i);
  if (omLogMatch) omegaLog = parseFloat(omLogMatch[1]);

  const tcMatches = stdout.matchAll(/mu\*?\s*=\s*([\d.]+)\s+Tc\s*=\s*([\d.]+)/gi);
  for (const m of tcMatches) {
    muStarValues.push(parseFloat(m[1]));
    tc.push(parseFloat(m[2]));
  }

  return {
    lambda: Number(lambda.toFixed(4)),
    omegaLog: Number(omegaLog.toFixed(2)),
    tc,
    muStarValues,
  };
}

export function buildDFPTJobSpec(
  formula: string,
  prefix: string,
  nq: [number, number, number] = [4, 4, 4],
  nkDos: [number, number, number] = [20, 20, 20],
  muStar: number = 0.10
): {
  phInput: string;
  q2rInput: string;
  matdynDosInput: string;
  matdynDispInput: string;
  lambdaInput: string;
  stages: string[];
} {
  return {
    phInput: generatePhononGridInput(prefix, nq[0], nq[1], nq[2]),
    q2rInput: generateQ2RInput(prefix, nq[0], nq[1], nq[2]),
    matdynDosInput: generateMatdynDOSInput(prefix, nkDos[0], nkDos[1], nkDos[2]),
    matdynDispInput: generateMatdynDispersionInput(prefix),
    lambdaInput: generateLambdaInput(prefix, muStar),
    stages: ["scf", "ph.x", "q2r.x", "matdyn.x (dos)", "matdyn.x (disp)", "lambda.x"],
  };
}

export function tryLoadDFPTResults(jobDir: string, prefix: string): {
  alpha2F: DFPTAlpha2FParsed | null;
  phononDOS: DFPTPhononDOS | null;
  dynmat: DFPTDynmatResult[] | null;
} {
  let alpha2F: DFPTAlpha2FParsed | null = null;
  let phononDOS: DFPTPhononDOS | null = null;
  let dynmat: DFPTDynmatResult[] | null = null;

  try {
    const a2fFile = path.join(jobDir, `${prefix}.a2F.dat`);
    if (fs.existsSync(a2fFile)) {
      alpha2F = parseAlpha2FOutput(fs.readFileSync(a2fFile, "utf-8"));
    }
  } catch {}

  const altA2fPaths = [
    path.join(jobDir, "a2F.dos1"),
    path.join(jobDir, "a2F.dos"),
    path.join(jobDir, `${prefix}.a2f`),
  ];
  if (!alpha2F) {
    for (const p of altA2fPaths) {
      try {
        if (fs.existsSync(p)) {
          alpha2F = parseAlpha2FOutput(fs.readFileSync(p, "utf-8"));
          if (alpha2F.frequencies.length > 0) break;
          alpha2F = null;
        }
      } catch {}
    }
  }

  try {
    const dosFile = path.join(jobDir, `${prefix}.phdos`);
    if (fs.existsSync(dosFile)) {
      phononDOS = parseMatdynDOS(fs.readFileSync(dosFile, "utf-8"));
    }
  } catch {}

  try {
    const phOut = path.join(jobDir, "ph.out");
    if (fs.existsSync(phOut)) {
      dynmat = parsePhDynmatOutput(fs.readFileSync(phOut, "utf-8"));
    }
  } catch {}

  return { alpha2F, phononDOS, dynmat };
}
