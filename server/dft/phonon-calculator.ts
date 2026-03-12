import { execSync, exec } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { promisify } from "util";
import { getElementData } from "../learning/elemental-data";

const execAsync = promisify(exec);

const PROJECT_ROOT = path.resolve(process.cwd());
const XTB_BIN = path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb");
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = "/tmp/dft_calculations";
const DEFAULT_DISPLACEMENT_DELTA = 0.015;
const BOHR_TO_ANG = 0.529177210903;
const AMU_TO_ELECTRONMASS = 1822.888486209;
const HARTREE_CM1 = 219474.6313632;
const PHONON_TIMEOUT_MS = 90_000;
const MAX_PHYSICAL_FREQ_CM1 = 5000;
const MAX_FC_ENTRY = 50.0;
const FC_CLAMP_WARNING_THRESHOLD = 0.1;
const JACOBI_TOLERANCE = 1e-12;
const DEGENERATE_PAIR_TOLERANCE = 1e-3;
const MAX_PARALLEL_XTB = Math.max(2, Math.min(os.cpus().length, 6));
const XTB_SINGLE_POINT_TIMEOUT = 30_000;

interface AtomPosition {
  element: string;
  x: number;
  y: number;
  z: number;
}

export interface PhononDispersionPoint {
  qLabel: string;
  qFrac: [number, number, number];
  frequencies: number[];
}

export interface PhononDOSBin {
  frequency: number;
  density: number;
}

export interface FiniteDisplacementPhononResult {
  formula: string;
  atomCount: number;
  forceConstantMatrix: number[][];
  gammaFrequencies: number[];
  dispersion: PhononDispersionPoint[];
  dos: PhononDOSBin[];
  omegaLog: number | null;
  lambdaContribution: number | null;
  hasImaginaryModes: boolean;
  imaginaryModeCount: number;
  lowestFrequency: number;
  highestFrequency: number;
  dynamicallyStable: boolean;
  calculationCount: number;
  wallTimeSeconds: number;
  forceConstantClampedEntries: number;
}

interface PhononCacheEntry {
  result: Omit<FiniteDisplacementPhononResult, "forceConstantMatrix">;
  matrixFile: string | null;
}

const phononResultCache = new Map<string, PhononCacheEntry>();
const PHONON_CACHE_MAX = 50;
const PHONON_CACHE_DIR = path.join(WORK_DIR, "phonon_cache");

function writeMatrixToDisk(cacheKey: string, matrix: number[][]): string | null {
  try {
    fs.mkdirSync(PHONON_CACHE_DIR, { recursive: true });
    const safeKey = cacheKey.replace(/[^a-zA-Z0-9_-]/g, "_");
    const filePath = path.join(PHONON_CACHE_DIR, `${safeKey}.json`);
    fs.writeFileSync(filePath, JSON.stringify(matrix));
    return filePath;
  } catch {
    return null;
  }
}

function readMatrixFromDisk(filePath: string): number[][] | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

function getCachedResult(cacheKey: string): FiniteDisplacementPhononResult | null {
  const entry = phononResultCache.get(cacheKey);
  if (!entry) return null;
  if (entry.matrixFile) {
    const matrix = readMatrixFromDisk(entry.matrixFile);
    if (!matrix || matrix.length === 0) {
      phononResultCache.delete(cacheKey);
      return null;
    }
    return { ...entry.result, forceConstantMatrix: matrix };
  }
  return { ...entry.result, forceConstantMatrix: [] };
}

function setCachedResult(cacheKey: string, result: FiniteDisplacementPhononResult): void {
  const matrixFile = writeMatrixToDisk(cacheKey, result.forceConstantMatrix);
  if (!matrixFile && result.forceConstantMatrix.length > 0) {
    console.warn(`[Phonon] Cache: failed to write matrix to disk for ${cacheKey.slice(0, 50)}, skipping cache`);
    return;
  }
  const { forceConstantMatrix, ...lightweight } = result;
  phononResultCache.set(cacheKey, { result: lightweight, matrixFile });
  if (phononResultCache.size > PHONON_CACHE_MAX) {
    const oldest = phononResultCache.keys().next().value;
    if (oldest) {
      const oldEntry = phononResultCache.get(oldest);
      if (oldEntry?.matrixFile) {
        try { fs.unlinkSync(oldEntry.matrixFile); } catch {}
      }
      phononResultCache.delete(oldest);
    }
  }
}

function writeXYZ(atoms: AtomPosition[], filepath: string, comment: string = ""): void {
  const lines = [
    String(atoms.length),
    comment || "Generated structure",
    ...atoms.map(a => `${a.element}  ${a.x.toFixed(6)}  ${a.y.toFixed(6)}  ${a.z.toFixed(6)}`),
  ];
  fs.writeFileSync(filepath, lines.join("\n") + "\n");
}

function getAtomicMass(element: string): number {
  const data = getElementData(element);
  if (!data?.atomicMass) {
    throw new Error(`Missing atomic mass for element "${element}" — cannot compute phonon spectrum`);
  }
  return data.atomicMass;
}

function runXTBSinglePoint(atoms: AtomPosition[], parentCalcDir: string, label: string): number[] | null {
  const subDir = path.join(parentCalcDir, label);
  fs.mkdirSync(subDir, { recursive: true });
  const xyzPath = path.join(subDir, `${label}.xyz`);
  writeXYZ(atoms, xyzPath, label);

  try {
    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    const output = execSync(
      `${XTB_BIN} ${xyzPath} --gfn 2 --sp --grad 2>&1`,
      { cwd: subDir, timeout: XTB_SINGLE_POINT_TIMEOUT, env, maxBuffer: 10 * 1024 * 1024 }
    ).toString();

    if (!output.includes("normal termination")) {
      console.warn(`[Phonon] xTB abnormal termination for ${label}`);
      return null;
    }

    const gradFile = path.join(subDir, "gradient");
    if (!fs.existsSync(gradFile)) {
      console.warn(`[Phonon] Gradient file missing for ${label}`);
      return null;
    }

    const gradContent = fs.readFileSync(gradFile, "utf-8");
    const gradLines = gradContent.split("\n");

    const forces: number[] = [];
    let inGrad = false;
    let lineCount = 0;
    const N = atoms.length;
    for (const line of gradLines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("$grad")) {
        inGrad = true;
        lineCount = 0;
        continue;
      }
      if (trimmed.startsWith("$end")) break;
      if (!inGrad) continue;

      if (trimmed.match(/^[-\d.]+[ED]?[-+]?\d*\s+[-\d.]+[ED]?[-+]?\d*\s+[-\d.]+[ED]?[-+]?\d*/)) {
        lineCount++;
        if (lineCount > N) {
          const parts = trimmed.split(/\s+/);
          for (const p of parts) {
            const val = parseFloat(p.replace(/[ED]/g, 'e'));
            if (Number.isFinite(val)) forces.push(val);
          }
        }
      }
    }

    if (forces.length < atoms.length * 3) {
      const eneGradFile = path.join(subDir, "xtb.engrad");
      if (fs.existsSync(eneGradFile)) {
        const enegradContent = fs.readFileSync(eneGradFile, "utf-8");
        const enegradLines = enegradContent.split("\n");
        forces.length = 0;

        let inGradSection = false;
        let gradLineCount = 0;
        const totalGradComponents = atoms.length * 3;

        for (const line of enegradLines) {
          const trimmed = line.trim();
          if (trimmed.includes("# The current gradient")) {
            inGradSection = true;
            gradLineCount = 0;
            continue;
          }
          if (inGradSection && gradLineCount < totalGradComponents) {
            const val = parseFloat(trimmed.replace(/[ED]/g, 'e'));
            if (Number.isFinite(val)) {
              forces.push(val);
              gradLineCount++;
            }
          }
          if (gradLineCount >= totalGradComponents) break;
        }
      }
    }

    if (forces.length >= atoms.length * 3) {
      return forces.slice(0, atoms.length * 3);
    }

    console.warn(`[Phonon] Insufficient gradient components for ${label}: got ${forces.length}, expected ${atoms.length * 3}`);
    return null;
  } catch (err: unknown) {
    const errObj = err instanceof Error ? err : null;
    const killed = errObj && (err as any).killed;
    const signal = errObj && (err as any).signal;
    const msg = errObj?.message || "";
    if (killed || signal === "SIGTERM" || msg.includes("TIMEOUT") || msg.includes("timed out")) {
      console.error(`[Phonon] xTB timeout for ${label} (limit: ${XTB_SINGLE_POINT_TIMEOUT}ms, signal: ${signal || "none"})`);
    } else if (msg.includes("ENOMEM") || msg.includes("Cannot allocate")) {
      console.error(`[Phonon] xTB memory error for ${label}: ${msg}`);
    } else if (errObj) {
      console.error(`[Phonon] xTB execution error for ${label}: ${msg}`);
    } else {
      console.error(`[Phonon] xTB unknown error for ${label}`);
    }
    return null;
  } finally {
    try {
      fs.rmSync(subDir, { recursive: true, force: true });
    } catch {}
  }
}

function runXTBSinglePointAsync(atoms: AtomPosition[], parentCalcDir: string, label: string): Promise<number[] | null> {
  return new Promise((resolve) => {
    const subDir = path.join(parentCalcDir, label);
    fs.mkdirSync(subDir, { recursive: true });
    const xyzPath = path.join(subDir, `${label}.xyz`);
    writeXYZ(atoms, xyzPath, label);

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    execAsync(
      `${XTB_BIN} ${xyzPath} --gfn 2 --sp --grad 2>&1`,
      { cwd: subDir, timeout: XTB_SINGLE_POINT_TIMEOUT, env, maxBuffer: 10 * 1024 * 1024 }
    ).then(({ stdout }) => {
      try {
        if (!stdout.includes("normal termination")) {
          console.warn(`[Phonon] xTB abnormal termination for ${label}`);
          resolve(null);
          return;
        }

        const gradFile = path.join(subDir, "gradient");
        if (!fs.existsSync(gradFile)) {
          console.warn(`[Phonon] Gradient file missing for ${label}`);
          resolve(null);
          return;
        }

        const gradContent = fs.readFileSync(gradFile, "utf-8");
        const gradLines = gradContent.split("\n");
        const N = atoms.length;

        const forces: number[] = [];
        let inGrad = false;
        let lineCount = 0;
        for (const line of gradLines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("$grad")) {
            inGrad = true;
            lineCount = 0;
            continue;
          }
          if (trimmed.startsWith("$end")) break;
          if (!inGrad) continue;

          if (trimmed.match(/^[-\d.]+[ED]?[-+]?\d*\s+[-\d.]+[ED]?[-+]?\d*\s+[-\d.]+[ED]?[-+]?\d*/)) {
            lineCount++;
            if (lineCount > N) {
              const parts = trimmed.split(/\s+/);
              for (const p of parts) {
                const val = parseFloat(p.replace(/[ED]/g, 'e'));
                if (Number.isFinite(val)) forces.push(val);
              }
            }
          }
        }

        if (forces.length < atoms.length * 3) {
          const eneGradFile = path.join(subDir, "xtb.engrad");
          if (fs.existsSync(eneGradFile)) {
            const enegradContent = fs.readFileSync(eneGradFile, "utf-8");
            const enegradLines = enegradContent.split("\n");
            forces.length = 0;

            let inGradSection = false;
            let gradLineCount = 0;
            const totalGradComponents = atoms.length * 3;

            for (const line of enegradLines) {
              const trimmed = line.trim();
              if (trimmed.includes("# The current gradient")) {
                inGradSection = true;
                gradLineCount = 0;
                continue;
              }
              if (inGradSection && gradLineCount < totalGradComponents) {
                const val = parseFloat(trimmed.replace(/[ED]/g, 'e'));
                if (Number.isFinite(val)) {
                  forces.push(val);
                  gradLineCount++;
                }
              }
              if (gradLineCount >= totalGradComponents) break;
            }
          }
        }

        if (forces.length >= atoms.length * 3) {
          resolve(forces.slice(0, atoms.length * 3));
        } else {
          console.warn(`[Phonon] Insufficient gradient components for ${label}: got ${forces.length}, expected ${atoms.length * 3}`);
          resolve(null);
        }
      } finally {
        try { fs.rmSync(subDir, { recursive: true, force: true }); } catch {}
      }
    }).catch((err: unknown) => {
      const errObj = err instanceof Error ? err : null;
      const killed = errObj && (err as any).killed;
      const signal = errObj && (err as any).signal;
      const msg = errObj?.message || "";
      if (killed || signal === "SIGTERM" || msg.includes("TIMEOUT") || msg.includes("timed out")) {
        console.error(`[Phonon] xTB timeout for ${label} (limit: ${XTB_SINGLE_POINT_TIMEOUT}ms, signal: ${signal || "none"})`);
      } else if (msg.includes("ENOMEM") || msg.includes("Cannot allocate")) {
        console.error(`[Phonon] xTB memory error for ${label}: ${msg}`);
      } else if (errObj) {
        console.error(`[Phonon] xTB execution error for ${label}: ${msg}`);
      } else {
        console.error(`[Phonon] xTB unknown error for ${label}`);
      }
      try { fs.rmSync(path.join(parentCalcDir, label), { recursive: true, force: true }); } catch {}
      resolve(null);
    });
  });
}

async function buildForceConstantMatrix(
  atoms: AtomPosition[],
  calcDir: string,
  displacementDelta: number = DEFAULT_DISPLACEMENT_DELTA,
): Promise<{ matrix: number[][]; calcCount: number; clampedEntries: number } | null> {
  const N = atoms.length;
  const dim = 3 * N;
  const matrix: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  let clampedEntries = 0;

  const displacementDeltaBohr = displacementDelta / BOHR_TO_ANG;

  const refForces = runXTBSinglePoint(atoms, calcDir, "ref");
  if (!refForces) return null;

  interface DisplacementTask {
    atomIdx: number;
    dir: number;
    plusAtoms: AtomPosition[];
    minusAtoms: AtomPosition[];
  }

  const tasks: DisplacementTask[] = [];
  for (let atomIdx = 0; atomIdx < N; atomIdx++) {
    for (let dir = 0; dir < 3; dir++) {
      const plusAtoms = atoms.map(a => ({ ...a }));
      const minusAtoms = atoms.map(a => ({ ...a }));
      const dirKey = dir === 0 ? "x" : dir === 1 ? "y" : "z";
      (plusAtoms[atomIdx] as any)[dirKey] += displacementDelta;
      (minusAtoms[atomIdx] as any)[dirKey] -= displacementDelta;
      tasks.push({ atomIdx, dir, plusAtoms, minusAtoms });
    }
  }

  const tasksPerBatch = Math.max(1, Math.floor(MAX_PARALLEL_XTB / 2));
  for (let batchStart = 0; batchStart < tasks.length; batchStart += tasksPerBatch) {
    const batch = tasks.slice(batchStart, batchStart + tasksPerBatch);
    const batchPromises = batch.map(async (task) => {
      const [plusForces, minusForces] = await Promise.all([
        runXTBSinglePointAsync(task.plusAtoms, calcDir, `disp_p_${task.atomIdx}_${task.dir}`),
        runXTBSinglePointAsync(task.minusAtoms, calcDir, `disp_m_${task.atomIdx}_${task.dir}`),
      ]);
      return { ...task, plusForces, minusForces };
    });

    const results = await Promise.all(batchPromises);
    for (const result of results) {
      if (!result.plusForces || !result.minusForces) return null;
      const colIdx = result.atomIdx * 3 + result.dir;
      for (let j = 0; j < dim; j++) {
        let fc = -(result.plusForces[j] - result.minusForces[j]) / (2 * displacementDeltaBohr);
        if (Math.abs(fc) > MAX_FC_ENTRY) {
          clampedEntries++;
          fc = Math.sign(fc) * MAX_FC_ENTRY;
        }
        matrix[j][colIdx] = fc;
      }
    }
  }

  const clampFraction = clampedEntries / (dim * dim);
  if (clampedEntries > 0) {
    console.warn(`[Phonon] Force constant clamping: ${clampedEntries} entries (${(clampFraction * 100).toFixed(1)}% of matrix) clamped to ±${MAX_FC_ENTRY} — structure may have unphysically close atoms`);
  }

  const calcCount = 1 + tasks.length * 2;

  const ASR_MAX_ITER = 10;
  const ASR_CONVERGENCE_THRESHOLD = 1e-10;

  for (let iter = 0; iter < ASR_MAX_ITER; iter++) {
    for (let i = 0; i < N; i++) {
      for (let a = 0; a < 3; a++) {
        const row = i * 3 + a;
        for (let b = 0; b < 3; b++) {
          let offDiagSum = 0;
          for (let j = 0; j < N; j++) {
            if (j !== i) {
              offDiagSum += matrix[row][j * 3 + b];
            }
          }
          matrix[row][i * 3 + b] = -offDiagSum;
        }
      }
    }

    for (let i = 0; i < dim; i++) {
      for (let j = i + 1; j < dim; j++) {
        const avg = (matrix[i][j] + matrix[j][i]) / 2;
        matrix[i][j] = avg;
        matrix[j][i] = avg;
      }
    }

    let maxDeviation = 0;
    for (let i = 0; i < N; i++) {
      for (let a = 0; a < 3; a++) {
        const row = i * 3 + a;
        for (let b = 0; b < 3; b++) {
          let rowSum = 0;
          for (let j = 0; j < N; j++) {
            rowSum += matrix[row][j * 3 + b];
          }
          maxDeviation = Math.max(maxDeviation, Math.abs(rowSum));
        }
      }
    }

    if (maxDeviation < ASR_CONVERGENCE_THRESHOLD) break;
  }

  return { matrix, calcCount, clampedEntries };
}

function estimateEffectiveLattice(atoms: AtomPosition[]): [number, number, number] {
  if (atoms.length <= 1) return [1, 1, 1];
  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let zMin = Infinity, zMax = -Infinity;
  for (const a of atoms) {
    if (a.x < xMin) xMin = a.x; if (a.x > xMax) xMax = a.x;
    if (a.y < yMin) yMin = a.y; if (a.y > yMax) yMax = a.y;
    if (a.z < zMin) zMin = a.z; if (a.z > zMax) zMax = a.z;
  }
  const pad = 2.0;
  return [
    Math.max(xMax - xMin + pad, 1.0),
    Math.max(yMax - yMin + pad, 1.0),
    Math.max(zMax - zMin + pad, 1.0),
  ];
}

function buildDynamicalMatrix(
  forceConstants: number[][],
  masses: number[],
  q: [number, number, number],
  atoms: AtomPosition[],
): { realPart: number[][]; imagPart: number[][] } {
  const N = atoms.length;
  const dim = 3 * N;
  const realPart: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  const imagPart: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));

  const effLattice = estimateEffectiveLattice(atoms);

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const massFactor = 1.0 / Math.sqrt(masses[i] * AMU_TO_ELECTRONMASS * masses[j] * AMU_TO_ELECTRONMASS);

      const fracDx = (atoms[j].x - atoms[i].x) / effLattice[0];
      const fracDy = (atoms[j].y - atoms[i].y) / effLattice[1];
      const fracDz = (atoms[j].z - atoms[i].z) / effLattice[2];
      const phase = 2 * Math.PI * (q[0] * fracDx + q[1] * fracDy + q[2] * fracDz);
      const cosPhase = Math.cos(phase);
      const sinPhase = Math.sin(phase);

      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          const fc = forceConstants[i * 3 + a][j * 3 + b];
          realPart[i * 3 + a][j * 3 + b] += fc * massFactor * cosPhase;
          imagPart[i * 3 + a][j * 3 + b] += fc * massFactor * sinPhase;
        }
      }
    }
  }

  for (let i = 0; i < dim; i++) {
    imagPart[i][i] = 0;
    for (let j = i + 1; j < dim; j++) {
      const reAvg = (realPart[i][j] + realPart[j][i]) / 2;
      realPart[i][j] = reAvg;
      realPart[j][i] = reAvg;

      const imAvg = (imagPart[i][j] - imagPart[j][i]) / 2;
      imagPart[i][j] = imAvg;
      imagPart[j][i] = -imAvg;
    }
  }

  return { realPart, imagPart };
}

function eigenvaluesSymmetric(matrix: number[][]): number[] {
  const n = matrix.length;
  const A: number[][] = matrix.map(row => [...row]);
  const eigenvalues: number[] = new Array(n).fill(0);

  const maxIter = 100 * n;
  for (let iter = 0; iter < maxIter; iter++) {
    let maxOff = 0;
    let p = 0, qq = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i][j]) > maxOff) {
          maxOff = Math.abs(A[i][j]);
          p = i;
          qq = j;
        }
      }
    }

    if (maxOff < JACOBI_TOLERANCE) break;

    const theta = (A[qq][qq] - A[p][p]) / (2 * A[p][qq]);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    const App = A[p][p];
    const Aqq = A[qq][qq];
    const Apq = A[p][qq];

    A[p][p] = App - t * Apq;
    A[qq][qq] = Aqq + t * Apq;
    A[p][qq] = 0;
    A[qq][p] = 0;

    for (let i = 0; i < n; i++) {
      if (i === p || i === qq) continue;
      const Aip = A[i][p];
      const Aiq = A[i][qq];
      A[i][p] = c * Aip - s * Aiq;
      A[p][i] = A[i][p];
      A[i][qq] = s * Aip + c * Aiq;
      A[qq][i] = A[i][qq];
    }
  }

  for (let i = 0; i < n; i++) {
    eigenvalues[i] = A[i][i];
  }

  return eigenvalues.sort((a, b) => a - b);
}

function eigenvaluesToFrequencies(eigenvalues: number[]): number[] {
  const CONV = HARTREE_CM1;
  return eigenvalues.map(ev => {
    let freq: number;
    if (ev < 0) {
      freq = -Math.sqrt(Math.abs(ev)) * CONV;
    } else {
      freq = Math.sqrt(ev) * CONV;
    }
    if (Math.abs(freq) > MAX_PHYSICAL_FREQ_CM1) {
      freq = Math.sign(freq) * MAX_PHYSICAL_FREQ_CM1;
    }
    return freq;
  });
}

function getHighSymmetryPath(atomCount: number, crystalSystem: string = "cubic"): { label: string; q: [number, number, number] }[] {
  const sys = crystalSystem.toLowerCase();

  if (sys === "hexagonal" || sys === "trigonal") {
    return [
      { label: "Γ", q: [0, 0, 0] },
      { label: "M", q: [0.5, 0, 0] },
      { label: "K", q: [1/3, 1/3, 0] },
      { label: "Γ", q: [0, 0, 0] },
      { label: "A", q: [0, 0, 0.5] },
    ];
  }

  if (sys === "tetragonal") {
    return [
      { label: "Γ", q: [0, 0, 0] },
      { label: "X", q: [0.5, 0, 0] },
      { label: "M", q: [0.5, 0.5, 0] },
      { label: "Γ", q: [0, 0, 0] },
      { label: "Z", q: [0, 0, 0.5] },
    ];
  }

  if (atomCount <= 4) {
    return [
      { label: "Γ", q: [0, 0, 0] },
      { label: "X", q: [0.5, 0, 0] },
      { label: "M", q: [0.5, 0.5, 0] },
      { label: "R", q: [0.5, 0.5, 0.5] },
      { label: "Γ", q: [0, 0, 0] },
    ];
  }
  return [
    { label: "Γ", q: [0, 0, 0] },
    { label: "X", q: [0.5, 0, 0] },
    { label: "M", q: [0.5, 0.5, 0] },
    { label: "Γ", q: [0, 0, 0] },
  ];
}

function interpolateQPoints(
  path: { label: string; q: [number, number, number] }[],
  pointsPerSegment: number = 10,
): { label: string; q: [number, number, number] }[] {
  const result: { label: string; q: [number, number, number] }[] = [];

  for (let seg = 0; seg < path.length - 1; seg++) {
    const start = path[seg];
    const end = path[seg + 1];

    for (let i = 0; i <= (seg === path.length - 2 ? pointsPerSegment : pointsPerSegment - 1); i++) {
      const t = i / pointsPerSegment;
      const q: [number, number, number] = [
        start.q[0] + t * (end.q[0] - start.q[0]),
        start.q[1] + t * (end.q[1] - start.q[1]),
        start.q[2] + t * (end.q[2] - start.q[2]),
      ];
      const label = i === 0 ? start.label : (i === pointsPerSegment && seg === path.length - 2 ? end.label : "");
      result.push({ label, q });
    }
  }

  return result;
}

export function computePhononDOS(frequencies: number[]): { dos: PhononDOSBin[]; omegaLog: number | null; lambdaContribution: number | null } {
  if (frequencies.length === 0) return { dos: [], omegaLog: null, lambdaContribution: null };

  const allFreqs = frequencies.filter(f => Number.isFinite(f) && Math.abs(f) <= MAX_PHYSICAL_FREQ_CM1);
  if (allFreqs.length === 0) return { dos: [], omegaLog: null, lambdaContribution: null };
  const minFreq = Math.min(...allFreqs);
  const maxFreq = Math.max(...allFreqs);

  const nBins = 100;
  const padding = Math.max(50, (maxFreq - minFreq) * 0.1);
  const binStart = Math.min(minFreq - padding, -100);
  const binEnd = maxFreq + padding;
  const binWidth = (binEnd - binStart) / nBins;

  const dos: PhononDOSBin[] = [];
  for (let i = 0; i < nBins; i++) {
    dos.push({ frequency: binStart + (i + 0.5) * binWidth, density: 0 });
  }

  const sigma = Math.max(10, binWidth * 2);
  for (const freq of allFreqs) {
    for (let i = 0; i < nBins; i++) {
      const center = dos[i].frequency;
      const diff = center - freq;
      dos[i].density += Math.exp(-0.5 * (diff / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
    }
  }

  const totalDensity = dos.reduce((s, b) => s + b.density, 0);
  if (totalDensity > 0) {
    for (const bin of dos) {
      bin.density /= totalDensity;
    }
  }

  const positiveFreqs = allFreqs.filter(f => f > 1);
  let omegaLog: number | null = null;
  if (positiveFreqs.length > 0) {
    const logSum = positiveFreqs.reduce((s, f) => s + Math.log(f), 0);
    omegaLog = Math.exp(logSum / positiveFreqs.length);
  }

  let lambdaContribution: number | null = null;
  if (omegaLog != null && omegaLog > 0 && positiveFreqs.length > 0) {
    const omega2Avg = positiveFreqs.reduce((s, f) => s + f * f, 0) / positiveFreqs.length;
    if (omega2Avg > 0) {
      lambdaContribution = (omegaLog * omegaLog) / omega2Avg;
    }
  }

  return { dos, omegaLog, lambdaContribution };
}

export function assessDynamicStability(dispersion: PhononDispersionPoint[]): {
  stable: boolean;
  imaginaryCount: number;
  worstQPoint: string | null;
  worstFrequency: number;
  numericalArtifact: boolean;
  positiveArtifact: boolean;
  physicalImaginaryCount: number;
  softModeCount: number;
} {
  let imaginaryCount = 0;
  let physicalImaginaryCount = 0;
  let softModeCount = 0;
  let negativeArtifactCount = 0;
  let positiveArtifactCount = 0;
  let worstFreq = 0;
  let worstPhysicalFreq = 0;
  let worstQ: string | null = null;

  const ACOUSTIC_THRESHOLD = -5;
  const SOFT_MODE_THRESHOLD = -20;
  const ARTIFACT_THRESHOLD = -2000;
  const POSITIVE_ARTIFACT_THRESHOLD = MAX_PHYSICAL_FREQ_CM1;

  for (const point of dispersion) {
    for (const freq of point.frequencies) {
      if (freq >= POSITIVE_ARTIFACT_THRESHOLD) {
        positiveArtifactCount++;
      }
      if (freq < ACOUSTIC_THRESHOLD) {
        imaginaryCount++;
        if (freq < worstFreq) {
          worstFreq = freq;
          worstQ = point.qLabel || `(${point.qFrac.map(v => v.toFixed(2)).join(",")})`;
        }

        if (freq < ARTIFACT_THRESHOLD) {
          negativeArtifactCount++;
        } else if (freq < SOFT_MODE_THRESHOLD) {
          physicalImaginaryCount++;
          if (freq < worstPhysicalFreq) worstPhysicalFreq = freq;
        } else {
          softModeCount++;
        }
      }
    }
  }

  const numericalArtifact = negativeArtifactCount > 0;
  const positiveArtifact = positiveArtifactCount > 0;
  const hasPhysicalInstability = physicalImaginaryCount > 0;

  return {
    stable: !hasPhysicalInstability && !numericalArtifact && !positiveArtifact && imaginaryCount === 0,
    imaginaryCount,
    physicalImaginaryCount,
    softModeCount,
    worstQPoint: worstQ,
    worstFrequency: worstFreq,
    numericalArtifact,
    positiveArtifact,
  };
}

export async function computeFiniteDisplacementPhonons(
  formula: string,
  atoms: AtomPosition[],
  crystalSystem: string = "cubic",
  displacementDelta: number = DEFAULT_DISPLACEMENT_DELTA,
): Promise<FiniteDisplacementPhononResult | null> {
  const structFingerprint = atoms.map(a => `${a.element}:${a.x.toFixed(3)},${a.y.toFixed(3)},${a.z.toFixed(3)}`).sort().join(";");
  const cacheKey = `${formula.replace(/\s+/g, "")}|${crystalSystem}|d=${displacementDelta}|${structFingerprint.length > 200 ? structFingerprint.slice(0, 200) : structFingerprint}`;
  const cached = getCachedResult(cacheKey);
  if (cached) {
    return cached;
  }

  const startTime = Date.now();
  const safeDirKey = `${formula.replace(/\s+/g, "")}_${crystalSystem}_${Date.now()}`;
  const calcDir = path.join(WORK_DIR, `fdphonon_${safeDirKey.replace(/[^a-zA-Z0-9_-]/g, "_")}`);

  try {
    fs.mkdirSync(calcDir, { recursive: true });

    if (atoms.length < 2) return null;

    const N = atoms.length;
    const expectedCalcs = 6 * N + 1;
    console.log(`[Phonon] ${formula}: Starting finite displacement phonon calculation (${N} atoms, ${expectedCalcs} xTB calculations)`);

    const fcResult = await buildForceConstantMatrix(atoms, calcDir, displacementDelta);
    if (!fcResult) {
      console.log(`[Phonon] ${formula}: Force constant matrix construction failed`);
      return null;
    }

    const masses = atoms.map(a => getAtomicMass(a.element));

    const gammaDynMatrix = buildDynamicalMatrix(fcResult.matrix, masses, [0, 0, 0], atoms);
    const gammaEigenvalues = eigenvaluesSymmetric(gammaDynMatrix.realPart);
    const gammaFrequencies = eigenvaluesToFrequencies(gammaEigenvalues);

    const symPath = getHighSymmetryPath(N, crystalSystem);
    const qDensity = N <= 3 ? 16 : N <= 6 ? 12 : N <= 10 ? 8 : 6;
    const qPoints = interpolateQPoints(symPath, qDensity);

    const dispersion: PhononDispersionPoint[] = [];
    for (const qp of qPoints) {
      const dynMatrix = buildDynamicalMatrix(fcResult.matrix, masses, qp.q, atoms);
      const isGamma = qp.q[0] === 0 && qp.q[1] === 0 && qp.q[2] === 0;
      let eigenvalues: number[];
      if (isGamma) {
        eigenvalues = eigenvaluesSymmetric(dynMatrix.realPart);
      } else {
        const dim = dynMatrix.realPart.length;
        const blockDim = 2 * dim;
        const block: number[][] = Array.from({ length: blockDim }, () => new Array(blockDim).fill(0));
        for (let i = 0; i < dim; i++) {
          for (let j = 0; j < dim; j++) {
            const re = dynMatrix.realPart[i][j];
            const im = dynMatrix.imagPart[i][j];
            block[i][j] = re;
            block[i][j + dim] = -im;
            block[i + dim][j] = im;
            block[i + dim][j + dim] = re;
          }
        }
        const blockEigs = eigenvaluesSymmetric(block);
        const sorted = [...blockEigs].sort((a, b) => a - b);
        const physicalEigs: number[] = [];
        let degenerateWarnings = 0;
        for (let i = 0; i < sorted.length; i += 2) {
          const e1 = sorted[i];
          const e2 = i + 1 < sorted.length ? sorted[i + 1] : e1;
          const avg = (e1 + e2) / 2;
          const diff = Math.abs(e1 - e2);
          const scale = Math.max(Math.abs(avg), 1e-10);
          const relDiff = diff / scale;
          if (relDiff > DEGENERATE_PAIR_TOLERANCE) {
            degenerateWarnings++;
          }
          physicalEigs.push(avg);
        }
        if (degenerateWarnings > 0) {
          console.warn(`[Phonon] q=[${qp.q.map(v => v.toFixed(3)).join(",")}]: ${degenerateWarnings}/${physicalEigs.length} eigenvalue pairs exceed relative tolerance ${DEGENERATE_PAIR_TOLERANCE}`);
        }
        eigenvalues = physicalEigs.sort((a, b) => a - b);
      }
      const frequencies = eigenvaluesToFrequencies(eigenvalues);

      dispersion.push({
        qLabel: qp.label,
        qFrac: qp.q,
        frequencies,
      });
    }

    const allDispersionFreqs = dispersion.flatMap(d => d.frequencies);
    const { dos, omegaLog, lambdaContribution } = computePhononDOS(allDispersionFreqs);

    const stability = assessDynamicStability(dispersion);

    const lowestFreq = allDispersionFreqs.length > 0 ? Math.min(...allDispersionFreqs) : 0;
    const rawHighestFreq = allDispersionFreqs.length > 0 ? Math.max(...allDispersionFreqs) : 0;
    const highestFreq = rawHighestFreq;
    if (rawHighestFreq >= MAX_PHYSICAL_FREQ_CM1) {
      console.log(`[Phonon] ${formula}: ARTIFACT CLAMPED — highest raw frequency ${rawHighestFreq.toFixed(0)} cm⁻¹ exceeds ${MAX_PHYSICAL_FREQ_CM1} cm⁻¹ physical limit, clamped to ±${MAX_PHYSICAL_FREQ_CM1}`);
    }

    const wallTime = (Date.now() - startTime) / 1000;

    const effectiveImaginaryCount = stability.numericalArtifact
      ? stability.physicalImaginaryCount
      : stability.physicalImaginaryCount + stability.softModeCount;
    const effectiveStable = stability.numericalArtifact
      ? stability.physicalImaginaryCount === 0 && !stability.positiveArtifact
      : stability.stable;

    const dim = 3 * N;
    const clampFraction = fcResult.clampedEntries / (dim * dim);
    const fcClampedUnreliable = clampFraction > FC_CLAMP_WARNING_THRESHOLD;

    if (fcClampedUnreliable) {
      console.warn(`[Phonon] ${formula}: HIGH CLAMP RATE — ${(clampFraction * 100).toFixed(1)}% of force constant entries were clamped. Stability assessment is unreliable; structure likely has unphysically close atoms.`);
    }

    const finalStable = fcClampedUnreliable ? false : effectiveStable;

    const result: FiniteDisplacementPhononResult = {
      formula,
      atomCount: N,
      forceConstantMatrix: fcResult.matrix,
      gammaFrequencies,
      dispersion,
      dos,
      omegaLog,
      lambdaContribution,
      hasImaginaryModes: !effectiveStable,
      imaginaryModeCount: effectiveImaginaryCount,
      lowestFrequency: lowestFreq,
      highestFrequency: highestFreq,
      dynamicallyStable: finalStable,
      calculationCount: fcResult.calcCount,
      wallTimeSeconds: wallTime,
      forceConstantClampedEntries: fcResult.clampedEntries,
    };

    if (stability.numericalArtifact) {
      const negArtifactCount = stability.imaginaryCount - stability.physicalImaginaryCount - stability.softModeCount;
      console.log(`[Phonon] ${formula}: NEGATIVE ARTIFACT — ${negArtifactCount} mode(s) below -2000 cm⁻¹ are xTB numerical explosions (lowest=${lowestFreq.toFixed(0)} cm⁻¹), discarded. ${stability.physicalImaginaryCount} physical imaginary + ${stability.softModeCount} soft mode(s) remain.`);
    }
    if (stability.positiveArtifact) {
      console.log(`[Phonon] ${formula}: POSITIVE ARTIFACT — frequencies ≥${MAX_PHYSICAL_FREQ_CM1} cm⁻¹ detected (highest=${highestFreq.toFixed(0)} cm⁻¹), indicates force constant blow-up.`);
    }
    if (stability.softModeCount > 0 && !stability.numericalArtifact) {
      console.log(`[Phonon] ${formula}: ${stability.softModeCount} soft mode(s) between -5 and -20 cm⁻¹ (likely ASR residuals from xTB).`);
    }
    console.log(`[Phonon] ${formula}: Finite displacement complete in ${wallTime.toFixed(1)}s — ${fcResult.calcCount} calcs, stable=${finalStable}, physImag=${stability.physicalImaginaryCount}, softModes=${stability.softModeCount}, freq range [${lowestFreq.toFixed(1)}, ${highestFreq.toFixed(1)}] cm⁻¹${omegaLog ? `, ω_log=${omegaLog.toFixed(1)} cm⁻¹` : ""}${fcResult.clampedEntries > 0 ? `, fc_clamped=${fcResult.clampedEntries}` : ""}`);

    setCachedResult(cacheKey, result);

    return result;
  } catch (err) {
    console.log(`[Phonon] ${formula}: Finite displacement phonon calculation failed: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  } finally {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
  }
}
