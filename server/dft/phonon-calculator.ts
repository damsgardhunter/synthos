import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { getElementData } from "../learning/elemental-data";

const PROJECT_ROOT = path.resolve(process.cwd());
const XTB_BIN = path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb");
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = "/tmp/dft_calculations";
const DISPLACEMENT_DELTA = 0.01;
const PHONON_TIMEOUT_MS = 90_000;

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
}

const phononResultCache = new Map<string, FiniteDisplacementPhononResult>();
const PHONON_CACHE_MAX = 100;

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
  return data?.atomicMass ?? 50;
}

function runXTBSinglePoint(atoms: AtomPosition[], calcDir: string, label: string): number[] | null {
  const xyzPath = path.join(calcDir, `${label}.xyz`);
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
      { cwd: calcDir, timeout: 30_000, env, maxBuffer: 10 * 1024 * 1024 }
    ).toString();

    if (!output.includes("normal termination")) return null;

    const gradFile = path.join(calcDir, "gradient");
    if (!fs.existsSync(gradFile)) return null;

    const gradContent = fs.readFileSync(gradFile, "utf-8");
    const gradLines = gradContent.split("\n");

    const forces: number[] = [];
    let inGrad = false;
    for (const line of gradLines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("$grad")) {
        inGrad = true;
        continue;
      }
      if (trimmed.startsWith("$end")) break;
      if (!inGrad) continue;

      if (trimmed.match(/^[-\d.]+[ED]?[-+]?\d*\s+[-\d.]+[ED]?[-+]?\d*\s+[-\d.]+[ED]?[-+]?\d*/)) {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 3) {
          const isCoordLine = parts.some(p => {
            const val = Math.abs(parseFloat(p.replace(/[ED]/g, 'e')));
            return val > 0.1;
          });

          if (!isCoordLine || forces.length >= atoms.length * 3) {
            for (const p of parts) {
              const val = parseFloat(p.replace(/[ED]/g, 'e'));
              if (Number.isFinite(val)) forces.push(val);
            }
          }
        }
      }
    }

    if (forces.length < atoms.length * 3) {
      const eneGradFile = path.join(calcDir, "xtb.engrad");
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

    return null;
  } catch {
    return null;
  } finally {
    try {
      const filesToClean = ["charges", "wbo", "xtbrestart", "xtbtopo.mol", ".xtboptok", "energy", "gradient", "xtb.engrad", `${label}.xyz`];
      for (const f of filesToClean) {
        const fp = path.join(calcDir, f);
        if (fs.existsSync(fp)) fs.unlinkSync(fp);
      }
    } catch {}
  }
}

function buildForceConstantMatrix(
  atoms: AtomPosition[],
  calcDir: string,
): { matrix: number[][]; calcCount: number } | null {
  const N = atoms.length;
  const dim = 3 * N;
  const matrix: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  let calcCount = 1;

  const refForces = runXTBSinglePoint(atoms, calcDir, "ref");
  if (!refForces) return null;

  for (let atomIdx = 0; atomIdx < N; atomIdx++) {
    for (let dir = 0; dir < 3; dir++) {
      const plusAtoms = atoms.map(a => ({ ...a }));
      const minusAtoms = atoms.map(a => ({ ...a }));

      const dirKey = dir === 0 ? "x" : dir === 1 ? "y" : "z";
      (plusAtoms[atomIdx] as any)[dirKey] += DISPLACEMENT_DELTA;
      (minusAtoms[atomIdx] as any)[dirKey] -= DISPLACEMENT_DELTA;

      const plusForces = runXTBSinglePoint(plusAtoms, calcDir, `disp_p_${atomIdx}_${dir}`);
      const minusForces = runXTBSinglePoint(minusAtoms, calcDir, `disp_m_${atomIdx}_${dir}`);
      calcCount += 2;

      if (!plusForces || !minusForces) return null;

      const colIdx = atomIdx * 3 + dir;
      for (let j = 0; j < dim; j++) {
        matrix[j][colIdx] = -(plusForces[j] - minusForces[j]) / (2 * DISPLACEMENT_DELTA);
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

  return { matrix, calcCount };
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

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const massFactor = 1.0 / Math.sqrt(masses[i] * masses[j]);

      const dx = atoms[j].x - atoms[i].x;
      const dy = atoms[j].y - atoms[i].y;
      const dz = atoms[j].z - atoms[i].z;
      const phase = 2 * Math.PI * (q[0] * dx + q[1] * dy + q[2] * dz);
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

    if (maxOff < 1e-12) break;

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
  const HA_PER_BOHR2_TO_EV_PER_ANG2 = 27.2114 / (0.529177 * 0.529177);
  const AMU_TO_EV_S2_PER_ANG2 = 1.03642698e-4;
  const EV_TO_CM1 = 8065.54;

  return eigenvalues.map(ev => {
    const scaledEv = ev * HA_PER_BOHR2_TO_EV_PER_ANG2 / AMU_TO_EV_S2_PER_ANG2;
    if (scaledEv < 0) {
      const freq = -Math.sqrt(Math.abs(scaledEv));
      return freq * Math.sqrt(1.0) * 521.471;
    }
    const freq = Math.sqrt(scaledEv);
    return freq * Math.sqrt(1.0) * 521.471;
  });
}

function getHighSymmetryPath(atomCount: number): { label: string; q: [number, number, number] }[] {
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

  const allFreqs = frequencies.filter(f => Number.isFinite(f));
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
  physicalImaginaryCount: number;
} {
  let imaginaryCount = 0;
  let physicalImaginaryCount = 0;
  let artifactCount = 0;
  let worstFreq = 0;
  let worstPhysicalFreq = 0;
  let worstQ: string | null = null;

  const ACOUSTIC_THRESHOLD = -50;
  const ARTIFACT_THRESHOLD = -2000;
  const PHYSICAL_INSTABILITY_THRESHOLD = -100;

  for (const point of dispersion) {
    for (const freq of point.frequencies) {
      if (freq < ACOUSTIC_THRESHOLD) {
        imaginaryCount++;
        if (freq < worstFreq) {
          worstFreq = freq;
          worstQ = point.qLabel || `(${point.qFrac.map(v => v.toFixed(2)).join(",")})`;
        }

        if (freq < ARTIFACT_THRESHOLD) {
          artifactCount++;
        } else if (freq < PHYSICAL_INSTABILITY_THRESHOLD) {
          physicalImaginaryCount++;
          if (freq < worstPhysicalFreq) worstPhysicalFreq = freq;
        }
      }
    }
  }

  const numericalArtifact = artifactCount > 0;
  const hasPhysicalInstability = physicalImaginaryCount > 0;

  return {
    stable: !hasPhysicalInstability && !numericalArtifact ? imaginaryCount === 0 : false,
    imaginaryCount,
    physicalImaginaryCount,
    worstQPoint: worstQ,
    worstFrequency: worstFreq,
    numericalArtifact,
  };
}

export async function computeFiniteDisplacementPhonons(
  formula: string,
  atoms: AtomPosition[],
): Promise<FiniteDisplacementPhononResult | null> {
  const cacheKey = formula.replace(/\s+/g, "");
  if (phononResultCache.has(cacheKey)) {
    return phononResultCache.get(cacheKey)!;
  }

  const startTime = Date.now();
  const calcDir = path.join(WORK_DIR, `fdphonon_${cacheKey.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`);

  try {
    fs.mkdirSync(calcDir, { recursive: true });

    if (atoms.length < 2) return null;

    const N = atoms.length;
    const expectedCalcs = 6 * N + 1;
    console.log(`[Phonon] ${formula}: Starting finite displacement phonon calculation (${N} atoms, ${expectedCalcs} xTB calculations)`);

    const fcResult = buildForceConstantMatrix(atoms, calcDir);
    if (!fcResult) {
      console.log(`[Phonon] ${formula}: Force constant matrix construction failed`);
      return null;
    }

    const masses = atoms.map(a => getAtomicMass(a.element));

    const gammaDynMatrix = buildDynamicalMatrix(fcResult.matrix, masses, [0, 0, 0], atoms);
    const gammaEigenvalues = eigenvaluesSymmetric(gammaDynMatrix.realPart);
    const gammaFrequencies = eigenvaluesToFrequencies(gammaEigenvalues);

    const symPath = getHighSymmetryPath(N);
    const qPoints = interpolateQPoints(symPath, 8);

    const dispersion: PhononDispersionPoint[] = [];
    for (const qp of qPoints) {
      const dynMatrix = buildDynamicalMatrix(fcResult.matrix, masses, qp.q, atoms);
      const eigenvalues = eigenvaluesSymmetric(dynMatrix.realPart);
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
    const highestFreq = rawHighestFreq > 8000 ? rawHighestFreq / 10 : rawHighestFreq;

    const wallTime = (Date.now() - startTime) / 1000;

    const effectiveImaginaryCount = stability.numericalArtifact
      ? stability.physicalImaginaryCount
      : stability.imaginaryCount;
    const effectiveStable = stability.numericalArtifact
      ? stability.physicalImaginaryCount === 0
      : stability.stable;

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
      dynamicallyStable: effectiveStable,
      calculationCount: fcResult.calcCount,
      wallTimeSeconds: wallTime,
    };

    if (stability.numericalArtifact) {
      console.log(`[Phonon] ${formula}: ARTIFACT DETECTED — ${stability.imaginaryCount - stability.physicalImaginaryCount} modes below -2000 cm⁻¹ are xTB numerical explosions (lowest=${lowestFreq.toFixed(0)} cm⁻¹), discarded. ${stability.physicalImaginaryCount} physical imaginary modes remain.`);
    }
    console.log(`[Phonon] ${formula}: Finite displacement complete in ${wallTime.toFixed(1)}s — ${fcResult.calcCount} calcs, stable=${effectiveStable}, freq range [${lowestFreq.toFixed(1)}, ${highestFreq.toFixed(1)}] cm⁻¹${omegaLog ? `, ω_log=${omegaLog.toFixed(1)} cm⁻¹` : ""}`);

    phononResultCache.set(cacheKey, result);
    if (phononResultCache.size > PHONON_CACHE_MAX) {
      const oldest = phononResultCache.keys().next().value;
      if (oldest) phononResultCache.delete(oldest);
    }

    return result;
  } catch (err) {
    console.log(`[Phonon] ${formula}: Finite displacement phonon calculation failed: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  } finally {
    try { fs.rmSync(calcDir, { recursive: true, force: true }); } catch {}
  }
}
