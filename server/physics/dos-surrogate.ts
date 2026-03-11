import { getElementData, ELEMENTAL_DATA } from "../learning/elemental-data";

const DOS_BINS = 64;
const ENERGY_RANGE_EV = 10.0;
const ENERGY_MIN = -5.0;
const ENERGY_MAX = 5.0;
const BIN_WIDTH = ENERGY_RANGE_EV / DOS_BINS;

const ORBITAL_CHANNELS = ["s", "p", "d", "f"] as const;
type OrbitalChannel = typeof ORBITAL_CHANNELS[number];

export interface OrbitalDOS {
  energyGrid: number[];
  totalDOS: number[];
  orbitalDOS: Record<OrbitalChannel, number[]>;
  fermiIndex: number;
  dosAtFermi: number;
  orbitalDOSAtFermi: Record<OrbitalChannel, number>;
}

export interface VanHoveSingularity {
  energyEv: number;
  binIndex: number;
  dosValue: number;
  relativeToFermi: number;
  type: "M0-onset" | "M1-saddle" | "M2-peak" | "logarithmic";
  dominantOrbital: OrbitalChannel;
  strength: number;
}

export interface DOSSurrogateResult {
  formula: string;
  orbitalDOS: OrbitalDOS;
  vanHoveSingularities: VanHoveSingularity[];
  vhsScore: number;
  scFavorability: number;
  isMetallic: boolean;
  flatBandIndicator: number;
  nestingScore: number;
  orbitalMixingAtFermi: number;
  magneticRisk: boolean;
  stonerParameter: number;
  predictionTier: "gnn-dos-head" | "physics-heuristic";
  wallTimeMs: number;
}

export interface DOSPrefilterResult {
  pass: boolean;
  score: number;
  reason: string;
  vhsCount: number;
  dosAtFermi: number;
  flatBandIndicator: number;
  scFavorability: number;
  magneticRisk: boolean;
  stonerParameter: number;
}

interface DOSHeadWeights {
  W_dos_hidden: number[][];
  b_dos_hidden: number[];
  W_dos_orbital: Record<OrbitalChannel, number[][]>;
  b_dos_orbital: Record<OrbitalChannel, number[]>;
  trained: boolean;
  trainedAt: number;
}

let dosHeadWeights: DOSHeadWeights | null = null;

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function initDOSHeadWeights(rng: () => number): DOSHeadWeights {
  const hiddenDim = 48;
  const dosHiddenDim = 32;

  const initMatrix = (rows: number, cols: number, scale: number): number[][] => {
    const xavierScale = scale * Math.sqrt(2.0 / (rows + cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (rng() - 0.5) * 2 * xavierScale)
    );
  };

  const initVec = (len: number, val = 0): number[] => new Array(len).fill(val);

  const orbitalWeights = {} as Record<OrbitalChannel, number[][]>;
  const orbitalBias = {} as Record<OrbitalChannel, number[]>;
  for (const orb of ORBITAL_CHANNELS) {
    orbitalWeights[orb] = initMatrix(DOS_BINS, dosHiddenDim, 0.1);
    orbitalBias[orb] = initVec(DOS_BINS);
  }

  return {
    W_dos_hidden: initMatrix(dosHiddenDim, hiddenDim, 0.15),
    b_dos_hidden: initVec(dosHiddenDim),
    W_dos_orbital: orbitalWeights,
    b_dos_orbital: orbitalBias,
    trained: false,
    trainedAt: 0,
  };
}

function softplus(x: number): number {
  if (x > 20) return x;
  if (x < -20) return 0;
  return Math.log(1 + Math.exp(x));
}

function relu(x: number): number {
  return Math.max(0, x);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
  return mat.map(row => row.reduce((s, w, j) => s + w * (vec[j] ?? 0), 0));
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map((v, i) => v + (b[i] ?? 0));
}

export function getDOSHeadWeights(): DOSHeadWeights {
  if (!dosHeadWeights) {
    const rng = seededRandom(31415);
    dosHeadWeights = initDOSHeadWeights(rng);
  }
  return dosHeadWeights;
}

export function dosHeadForward(latentEmbedding: number[]): OrbitalDOS {
  const weights = getDOSHeadWeights();

  const safeLatent = latentEmbedding.map(v => isFinite(v) ? v : 0);

  const hidden = vecAdd(matVecMul(weights.W_dos_hidden, safeLatent), weights.b_dos_hidden)
    .map(v => relu(isFinite(v) ? v : 0));

  const orbitalDOS: Record<OrbitalChannel, number[]> = { s: [], p: [], d: [], f: [] };
  const totalDOS: number[] = new Array(DOS_BINS).fill(0);

  for (const orb of ORBITAL_CHANNELS) {
    const raw = vecAdd(matVecMul(weights.W_dos_orbital[orb], hidden), weights.b_dos_orbital[orb]);
    orbitalDOS[orb] = raw.map(v => {
      const sp = softplus(isFinite(v) ? v : 0);
      const safe = isFinite(sp) ? sp : 0;
      return Math.max(0, safe);
    });

    for (let i = 0; i < DOS_BINS; i++) {
      totalDOS[i] += orbitalDOS[orb][i];
    }
  }

  for (let i = 0; i < DOS_BINS; i++) {
    totalDOS[i] = Math.max(0, totalDOS[i]);
    if (!isFinite(totalDOS[i])) totalDOS[i] = 0;
  }

  const energyGrid = Array.from({ length: DOS_BINS }, (_, i) =>
    ENERGY_MIN + (i + 0.5) * BIN_WIDTH
  );

  const fermiIndex = Math.floor((0 - ENERGY_MIN) / BIN_WIDTH);
  const dosAtFermi = totalDOS[fermiIndex] ?? 0;

  const orbitalDOSAtFermi: Record<OrbitalChannel, number> = {
    s: orbitalDOS.s[fermiIndex] ?? 0,
    p: orbitalDOS.p[fermiIndex] ?? 0,
    d: orbitalDOS.d[fermiIndex] ?? 0,
    f: orbitalDOS.f[fermiIndex] ?? 0,
  };

  return { energyGrid, totalDOS, orbitalDOS, fermiIndex, dosAtFermi, orbitalDOSAtFermi };
}

function estimateOrbitalOccupancy(Z: number, valence: number): { s: number; p: number; d: number; f: number } {
  if (Z <= 2) return { s: valence, p: 0, d: 0, f: 0 };
  if (Z <= 4) return { s: valence, p: 0, d: 0, f: 0 };
  if (Z <= 10) return { s: Math.min(2, valence), p: Math.max(0, valence - 2), d: 0, f: 0 };
  if (Z <= 12) return { s: valence, p: 0, d: 0, f: 0 };
  if (Z <= 18) return { s: Math.min(2, valence), p: Math.max(0, valence - 2), d: 0, f: 0 };
  if (Z <= 20) return { s: valence, p: 0, d: 0, f: 0 };
  if (Z <= 30) {
    const dElectrons = Math.min(10, Math.max(0, valence - 2));
    return { s: Math.max(0, valence - dElectrons), p: 0, d: dElectrons, f: 0 };
  }
  if (Z <= 36) return { s: Math.min(2, valence), p: Math.max(0, valence - 2), d: 0, f: 0 };
  if (Z <= 38) return { s: valence, p: 0, d: 0, f: 0 };
  if (Z <= 48) {
    const dElectrons = Math.min(10, Math.max(0, valence - 2));
    return { s: Math.max(0, valence - dElectrons), p: 0, d: dElectrons, f: 0 };
  }
  if (Z <= 54) return { s: Math.min(2, valence), p: Math.max(0, valence - 2), d: 0, f: 0 };
  if (Z <= 56) return { s: valence, p: 0, d: 0, f: 0 };
  if (Z <= 71) {
    const fElectrons = Math.min(14, Math.max(0, valence - 3));
    const dElectrons = Math.min(1, Math.max(0, valence - fElectrons - 2));
    return { s: 2, p: 0, d: dElectrons, f: fElectrons };
  }
  if (Z <= 80) {
    const dElectrons = Math.min(10, Math.max(0, valence - 2));
    return { s: Math.max(0, valence - dElectrons), p: 0, d: dElectrons, f: 0 };
  }
  if (Z <= 86) return { s: Math.min(2, valence), p: Math.max(0, valence - 2), d: 0, f: 0 };
  if (Z <= 103) {
    const fElectrons = Math.min(14, Math.max(0, valence - 3));
    const dElectrons = Math.min(1, Math.max(0, valence - fElectrons - 2));
    return { s: 2, p: 0, d: dElectrons, f: fElectrons };
  }
  return { s: 2, p: 0, d: Math.max(0, valence - 2), f: 0 };
}

export function physicsHeuristicDOS(formula: string): OrbitalDOS {
  const elements = parseFormula(formula);
  const energyGrid = Array.from({ length: DOS_BINS }, (_, i) =>
    ENERGY_MIN + (i + 0.5) * BIN_WIDTH
  );

  const orbitalDOS: Record<OrbitalChannel, number[]> = {
    s: new Array(DOS_BINS).fill(0),
    p: new Array(DOS_BINS).fill(0),
    d: new Array(DOS_BINS).fill(0),
    f: new Array(DOS_BINS).fill(0),
  };

  for (const { element, count } of elements) {
    const elData = getElementData(element);
    if (!elData) continue;

    const weight = count;
    const Z = elData.atomicNumber;
    const eneg = elData.paulingElectronegativity ?? 2.0;
    const valence = elData.valenceElectrons;
    const orbitals = estimateOrbitalOccupancy(Z, valence);
    const dOcc = orbitals.d;
    const fOcc = orbitals.f;
    const sOcc = orbitals.s;
    const pOcc = orbitals.p;

    const centerShift = (eneg - 2.0) * 0.3;

    for (let i = 0; i < DOS_BINS; i++) {
      const E = energyGrid[i] - centerShift;

      const sContrib = weight * sOcc * 0.3 * Math.exp(-E * E / 8.0);
      orbitalDOS.s[i] += sContrib;

      const pWidth = 3.0 + pOcc * 0.5;
      const pContrib = weight * pOcc * 0.5 * Math.exp(-E * E / (2 * pWidth));
      orbitalDOS.p[i] += pContrib;

      if (dOcc > 0) {
        const dCenter = E + 0.5;
        const dWidth = 1.5 + (10 - dOcc) * 0.15;
        const dPeak = dOcc * 0.8;
        const mainPeak = Math.exp(-dCenter * dCenter / (2 * dWidth));

        let vhsBump = 0;
        if (dOcc >= 3 && dOcc <= 7) {
          const vhsCenter = -0.3 + (dOcc - 5) * 0.1;
          vhsBump = 0.4 * Math.exp(-(E - vhsCenter) * (E - vhsCenter) / 0.3);
        }

        orbitalDOS.d[i] += weight * dPeak * (mainPeak + vhsBump);
      }

      if (fOcc > 0) {
        const fCenter = E + 1.0;
        const fWidth = 0.8;
        const fPeak = fOcc * 0.6;
        orbitalDOS.f[i] += weight * fPeak * Math.exp(-fCenter * fCenter / (2 * fWidth));
      }
    }
  }

  const totalDOS = new Array(DOS_BINS).fill(0);
  for (let i = 0; i < DOS_BINS; i++) {
    totalDOS[i] = orbitalDOS.s[i] + orbitalDOS.p[i] + orbitalDOS.d[i] + orbitalDOS.f[i];
  }

  const maxDOS = Math.max(...totalDOS, 1e-6);
  for (let i = 0; i < DOS_BINS; i++) {
    totalDOS[i] /= maxDOS;
    for (const orb of ORBITAL_CHANNELS) {
      orbitalDOS[orb][i] /= maxDOS;
    }
  }

  const fermiIndex = Math.floor((0 - ENERGY_MIN) / BIN_WIDTH);
  const dosAtFermi = totalDOS[fermiIndex] ?? 0;

  return {
    energyGrid,
    totalDOS,
    orbitalDOS,
    fermiIndex,
    dosAtFermi,
    orbitalDOSAtFermi: {
      s: orbitalDOS.s[fermiIndex] ?? 0,
      p: orbitalDOS.p[fermiIndex] ?? 0,
      d: orbitalDOS.d[fermiIndex] ?? 0,
      f: orbitalDOS.f[fermiIndex] ?? 0,
    },
  };
}

export function detectVanHoveSingularities(dos: OrbitalDOS): VanHoveSingularity[] {
  const singularities: VanHoveSingularity[] = [];
  const totalDOS = dos.totalDOS;
  const nBins = totalDOS.length;

  const smoothed = totalDOS.map((v, i) => {
    if (i === 0 || i === nBins - 1) return v;
    return (totalDOS[i - 1] + 2 * v + totalDOS[i + 1]) / 4;
  });

  const dosMax = Math.max(...smoothed, 1e-6);
  const dosMedian = [...smoothed].sort((a, b) => a - b)[Math.floor(nBins / 2)];
  const threshold = Math.max(dosMedian * 1.5, dosMax * 0.3);

  const SALIENCY_WINDOW = Math.max(3, Math.floor(nBins * 0.08));
  const SALIENCY_D2_THRESHOLD = 0.12;
  const SALIENCY_PROMINENCE_THRESHOLD = 0.25;

  for (let i = 2; i < nBins - 2; i++) {
    const val = smoothed[i];
    if (val < threshold) continue;

    const isPeak = val > smoothed[i - 1] && val > smoothed[i + 1];
    const isSaddle = (smoothed[i - 1] > val && val < smoothed[i + 1]) ||
                     (smoothed[i - 1] < val && val > smoothed[i + 1] && Math.abs(smoothed[i - 1] - smoothed[i + 1]) > dosMax * 0.1);

    const d2 = smoothed[i - 1] - 2 * val + smoothed[i + 1];
    const isLogDiv = Math.abs(d2) > dosMax * 0.15 && val > dosMedian * 2;

    const d1Left = val - smoothed[i - 1];
    const d1Right = smoothed[i + 1] - val;
    const isOnset = d1Left > dosMax * 0.1 && Math.abs(d1Right) < dosMax * 0.03;

    if (!isPeak && !isSaddle && !isLogDiv && !isOnset) continue;

    let localSum = 0;
    let localCount = 0;
    const wStart = Math.max(0, i - SALIENCY_WINDOW);
    const wEnd = Math.min(nBins - 1, i + SALIENCY_WINDOW);
    for (let j = wStart; j <= wEnd; j++) {
      if (j === i) continue;
      localSum += smoothed[j];
      localCount++;
    }
    const localAvg = localCount > 0 ? localSum / localCount : val;

    const prominence = localAvg > 1e-8 ? (val - localAvg) / localAvg : 0;
    if (prominence < SALIENCY_PROMINENCE_THRESHOLD) continue;

    const d2Magnitude = Math.abs(d2);
    const d2Relative = localAvg > 1e-8 ? d2Magnitude / localAvg : 0;
    if (d2Relative < SALIENCY_D2_THRESHOLD) continue;

    if (i >= 2 && i < nBins - 2) {
      const d2Left = smoothed[i - 2] - 2 * smoothed[i - 1] + smoothed[i];
      const d2Right = smoothed[i] - 2 * smoothed[i + 1] + smoothed[i + 2];
      const curvatureConsistent = (isPeak || isLogDiv)
        ? (d2 < 0 && (d2Left <= 0 || d2Right <= 0))
        : true;
      if (!curvatureConsistent && Math.abs(d2) < dosMax * 0.25) continue;
    }

    let type: VanHoveSingularity["type"];
    if (isPeak) type = "M2-peak";
    else if (isLogDiv) type = "logarithmic";
    else if (isSaddle) type = "M1-saddle";
    else type = "M0-onset";

    let dominantOrbital: OrbitalChannel = "s";
    let maxOrbVal = 0;
    for (const orb of ORBITAL_CHANNELS) {
      const orbVal = dos.orbitalDOS[orb][i] ?? 0;
      if (orbVal > maxOrbVal) {
        maxOrbVal = orbVal;
        dominantOrbital = orb;
      }
    }

    const distToFermi = Math.abs(i - dos.fermiIndex);
    const proximityWeight = Math.exp(-distToFermi * distToFermi / (DOS_BINS * 0.1));
    const saliencyBoost = Math.min(1.0, prominence / 1.5);
    const strength = (val / dosMax) * proximityWeight * (0.5 + 0.5 * saliencyBoost);

    singularities.push({
      energyEv: dos.energyGrid[i],
      binIndex: i,
      dosValue: val,
      relativeToFermi: dos.energyGrid[i],
      type,
      dominantOrbital,
      strength,
    });
  }

  singularities.sort((a, b) => b.strength - a.strength);

  const deduped: VanHoveSingularity[] = [];
  for (const vhs of singularities) {
    const tooClose = deduped.some(d => Math.abs(d.binIndex - vhs.binIndex) < 3);
    if (!tooClose) {
      deduped.push(vhs);
    }
    if (deduped.length >= 10) break;
  }

  return deduped;
}

function computeVHSScore(singularities: VanHoveSingularity[], dos: OrbitalDOS): number {
  if (singularities.length === 0) return 0;

  let score = 0;
  const fermiWindow = 0.5;

  for (const vhs of singularities) {
    const inFermiWindow = Math.abs(vhs.relativeToFermi) < fermiWindow;
    const typeWeight =
      vhs.type === "logarithmic" ? 1.0 :
      vhs.type === "M2-peak" ? 0.9 :
      vhs.type === "M1-saddle" ? 0.7 :
      0.5;

    const orbitalWeight =
      vhs.dominantOrbital === "d" ? 1.0 :
      vhs.dominantOrbital === "f" ? 0.9 :
      vhs.dominantOrbital === "p" ? 0.6 :
      0.4;

    const contrib = vhs.strength * typeWeight * orbitalWeight * (inFermiWindow ? 2.0 : 0.5);
    score += contrib;
  }

  const dFraction = dos.orbitalDOSAtFermi.d / Math.max(dos.dosAtFermi, 1e-6);
  score *= (1 + dFraction * 0.5);

  return Math.min(1.0, score);
}

function computeFlatBandIndicator(dos: OrbitalDOS): number {
  const MIN_DOS_AT_FERMI_FOR_FLAT_BAND = 0.1;

  if (dos.dosAtFermi < MIN_DOS_AT_FERMI_FOR_FLAT_BAND) return 0;

  const fermiIdx = dos.fermiIndex;
  const windowBins = 3;
  const start = Math.max(0, fermiIdx - windowBins);
  const end = Math.min(dos.totalDOS.length - 1, fermiIdx + windowBins);

  let sumDOS = 0;
  let count = 0;
  for (let i = start; i <= end; i++) {
    sumDOS += dos.totalDOS[i];
    count++;
  }

  const avgNearFermi = count > 0 ? sumDOS / count : 0;

  if (avgNearFermi < MIN_DOS_AT_FERMI_FOR_FLAT_BAND) return 0;

  const totalAvg = dos.totalDOS.reduce((s, v) => s + v, 0) / dos.totalDOS.length;

  if (totalAvg < 1e-8) return 0;

  const ratio = avgNearFermi / totalAvg;
  if (ratio <= 1.0) return 0;

  const stdDev = Math.sqrt(
    dos.totalDOS.reduce((s, v) => s + (v - totalAvg) * (v - totalAvg), 0) / dos.totalDOS.length
  );
  const coeffOfVariation = totalAvg > 1e-8 ? stdDev / totalAvg : 0;
  const narrowPeakBonus = coeffOfVariation > 0.5 ? Math.min(0.3, (coeffOfVariation - 0.5) * 0.3) : 0;

  return Math.min(1.0, (ratio - 1) * 0.5 + narrowPeakBonus);
}

function computeNestingScore(dos: OrbitalDOS): number {
  const fermiIdx = dos.fermiIndex;
  const totalDOS = dos.totalDOS;
  const n = totalDOS.length;

  let autoCorr = 0;
  let norm = 0;
  const qRange = Math.floor(n / 4);

  for (let q = 1; q <= qRange; q++) {
    let corr = 0;
    let cnt = 0;
    for (let i = 0; i < n - q; i++) {
      corr += totalDOS[i] * totalDOS[i + q];
      cnt++;
    }
    const avgCorr = cnt > 0 ? corr / cnt : 0;
    const qWeight = Math.exp(-(q - n / 8) * (q - n / 8) / (n * 2));
    autoCorr += avgCorr * qWeight;
    norm += qWeight;
  }

  if (norm < 1e-8) return 0;
  const fermiDOS = totalDOS[fermiIdx] ?? 0;
  const avgDOS = totalDOS.reduce((s, v) => s + v, 0) / n;
  const selfNorm = avgDOS * avgDOS;

  if (selfNorm < 1e-8) return 0;

  const normalizedNesting = (autoCorr / norm) / selfNorm;
  const fermiBoost = fermiDOS > avgDOS * 1.5 ? 1.3 : 1.0;

  return Math.min(1.0, normalizedNesting * fermiBoost * 0.5);
}

function computeOrbitalMixing(dos: OrbitalDOS): number {
  const total = dos.dosAtFermi;
  if (total < 1e-8) return 0;

  const fracs: Record<OrbitalChannel, number> = {
    s: dos.orbitalDOSAtFermi.s / total,
    p: dos.orbitalDOSAtFermi.p / total,
    d: dos.orbitalDOSAtFermi.d / total,
    f: dos.orbitalDOSAtFermi.f / total,
  };

  const fractions = ORBITAL_CHANNELS.map(orb => fracs[orb]);
  const nonZero = fractions.filter(f => f > 0.05);

  if (nonZero.length <= 1) return 0;

  let entropy = 0;
  for (const f of fractions) {
    if (f > 1e-8) {
      entropy -= f * Math.log(f);
    }
  }

  const maxEntropy = Math.log(ORBITAL_CHANNELS.length);
  let mixing = entropy / maxEntropy;

  const SP_HYBRIDIZATION_THRESHOLD = 0.30;
  const SD_HYBRIDIZATION_THRESHOLD = 0.30;

  const spHybridized = fracs.s >= SP_HYBRIDIZATION_THRESHOLD && fracs.p >= SP_HYBRIDIZATION_THRESHOLD;
  const sdHybridized = fracs.s >= SD_HYBRIDIZATION_THRESHOLD && fracs.d >= SD_HYBRIDIZATION_THRESHOLD;
  const pdHybridized = fracs.p >= SP_HYBRIDIZATION_THRESHOLD && fracs.d >= SD_HYBRIDIZATION_THRESHOLD;

  if (spHybridized) {
    const overlap = Math.min(fracs.s, fracs.p);
    mixing += 0.25 * (overlap / 0.5);
  }
  if (sdHybridized) {
    const overlap = Math.min(fracs.s, fracs.d);
    mixing += 0.20 * (overlap / 0.5);
  }
  if (pdHybridized) {
    const overlap = Math.min(fracs.p, fracs.d);
    mixing += 0.15 * (overlap / 0.5);
  }

  return Math.min(1.0, mixing);
}

function computeStonerParameter(dos: OrbitalDOS): { magneticRisk: boolean; stonerParameter: number } {
  const total = dos.dosAtFermi;
  if (total < 0.05) return { magneticRisk: false, stonerParameter: 0 };

  const dFrac = total > 0 ? dos.orbitalDOSAtFermi.d / total : 0;
  const fFrac = total > 0 ? dos.orbitalDOSAtFermi.f / total : 0;

  const STONER_I_D = 0.7;
  const STONER_I_F = 0.5;

  const effectiveDOS_d = dos.orbitalDOSAtFermi.d;
  const effectiveDOS_f = dos.orbitalDOSAtFermi.f;

  const stonerProduct_d = STONER_I_D * effectiveDOS_d;
  const stonerProduct_f = STONER_I_F * effectiveDOS_f;
  const stonerParameter = Math.max(stonerProduct_d, stonerProduct_f);

  let magneticRisk = false;

  if (stonerParameter > 0.8) {
    magneticRisk = true;
  }

  if ((dFrac > 0.6 || fFrac > 0.5) && total > 0.5) {
    const fermiIdx = dos.fermiIndex;
    const nBins = dos.totalDOS.length;
    const windowStart = Math.max(0, fermiIdx - 2);
    const windowEnd = Math.min(nBins - 1, fermiIdx + 2);
    let peakSharpness = 0;
    for (let i = windowStart; i <= windowEnd; i++) {
      const dVal = dos.orbitalDOS.d[i] + dos.orbitalDOS.f[i];
      peakSharpness = Math.max(peakSharpness, dVal);
    }
    const globalAvgDF = dos.totalDOS.reduce((s, _, i) =>
      s + dos.orbitalDOS.d[i] + dos.orbitalDOS.f[i], 0) / nBins;

    if (globalAvgDF > 1e-8 && peakSharpness / globalAvgDF > 2.5) {
      magneticRisk = true;
    }
  }

  return { magneticRisk, stonerParameter: Number(stonerParameter.toFixed(4)) };
}

export function predictDOS(formula: string, latentEmbedding?: number[]): DOSSurrogateResult {
  const startTime = Date.now();

  let orbitalDOS: OrbitalDOS;
  let tier: DOSSurrogateResult["predictionTier"];

  if (latentEmbedding && latentEmbedding.length > 0) {
    orbitalDOS = dosHeadForward(latentEmbedding);
    tier = "gnn-dos-head";
  } else {
    orbitalDOS = physicsHeuristicDOS(formula);
    tier = "physics-heuristic";
  }

  const vanHoveSingularities = detectVanHoveSingularities(orbitalDOS);
  const vhsScore = computeVHSScore(vanHoveSingularities, orbitalDOS);
  const flatBandIndicator = computeFlatBandIndicator(orbitalDOS);
  const nestingScore = computeNestingScore(orbitalDOS);
  const orbitalMixingAtFermi = computeOrbitalMixing(orbitalDOS);

  const isMetallic = orbitalDOS.dosAtFermi > 0.1;

  const stoner = computeStonerParameter(orbitalDOS);

  let scFavorability = Math.min(1.0,
    0.25 * vhsScore +
    0.25 * (isMetallic ? Math.min(1.0, orbitalDOS.dosAtFermi) : 0) +
    0.18 * flatBandIndicator +
    0.12 * nestingScore +
    0.20 * orbitalMixingAtFermi
  );

  if (stoner.magneticRisk) {
    scFavorability *= 0.7;
  }

  return {
    formula,
    orbitalDOS,
    vanHoveSingularities,
    vhsScore,
    scFavorability,
    isMetallic,
    flatBandIndicator,
    nestingScore,
    orbitalMixingAtFermi,
    magneticRisk: stoner.magneticRisk,
    stonerParameter: stoner.stonerParameter,
    predictionTier: tier,
    wallTimeMs: Date.now() - startTime,
  };
}

export function dosPrefilter(formula: string, latentEmbedding?: number[]): DOSPrefilterResult {
  const result = predictDOS(formula, latentEmbedding);

  const MIN_DOS_AT_FERMI = 0.05;
  const MIN_SC_FAVORABILITY = 0.05;

  let pass = true;
  let reason = "DOS pre-filter passed";

  if (!result.isMetallic && result.orbitalDOS.dosAtFermi < MIN_DOS_AT_FERMI) {
    pass = false;
    reason = `Non-metallic: DOS(EF)=${result.orbitalDOS.dosAtFermi.toFixed(3)} < ${MIN_DOS_AT_FERMI}`;
  } else if (result.scFavorability < MIN_SC_FAVORABILITY && result.vhsScore < 0.01) {
    pass = false;
    reason = `Low SC favorability: ${result.scFavorability.toFixed(3)}, no VHS near Fermi`;
  }

  if (result.magneticRisk && pass) {
    reason += ` [MAGNETIC WARNING: Stoner parameter=${result.stonerParameter.toFixed(3)}, high d/f DOS at EF — check spin-fluctuation competition]`;
  }

  return {
    pass,
    score: result.scFavorability,
    reason,
    vhsCount: result.vanHoveSingularities.length,
    dosAtFermi: result.orbitalDOS.dosAtFermi,
    flatBandIndicator: result.flatBandIndicator,
    scFavorability: result.scFavorability,
    magneticRisk: result.magneticRisk,
    stonerParameter: result.stonerParameter,
  };
}

export function trainDOSHead(
  trainingPairs: { latent: number[]; dosTarget: number[]; orbitalTargets?: Record<OrbitalChannel, number[]> }[]
): void {
  if (trainingPairs.length < 5) return;

  const weights = getDOSHeadWeights();
  const lr = 0.001;
  const epochs = 20;
  const rng = seededRandom(42);

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (const pair of trainingPairs) {
      const hidden = vecAdd(matVecMul(weights.W_dos_hidden, pair.latent), weights.b_dos_hidden)
        .map(v => relu(v));

      for (const orb of ORBITAL_CHANNELS) {
        const pred = vecAdd(matVecMul(weights.W_dos_orbital[orb], hidden), weights.b_dos_orbital[orb])
          .map(v => softplus(v));
        const target = pair.orbitalTargets?.[orb] ?? pair.dosTarget.map(v => v / 4);

        for (let i = 0; i < DOS_BINS; i++) {
          const error = pred[i] - (target[i] ?? 0);
          totalLoss += error * error;

          const grad = 2 * error * lr;
          for (let j = 0; j < hidden.length; j++) {
            weights.W_dos_orbital[orb][i][j] -= grad * hidden[j] * (rng() * 0.5 + 0.5);
          }
          weights.b_dos_orbital[orb][i] -= grad * 0.5;
        }
      }
    }

    if (totalLoss / trainingPairs.length < 0.01) break;
  }

  weights.trained = true;
  weights.trainedAt = Date.now();
  console.log(`[DOS-Surrogate] Trained DOS head on ${trainingPairs.length} samples`);
}

export function getDOSSurrogateStats(): {
  trained: boolean;
  dosBins: number;
  energyRangeEv: [number, number];
  orbitalChannels: string[];
  weightsDimensions: { hiddenDim: number; outputBinsPerOrbital: number };
} {
  const weights = getDOSHeadWeights();
  return {
    trained: weights.trained,
    dosBins: DOS_BINS,
    energyRangeEv: [ENERGY_MIN, ENERGY_MAX],
    orbitalChannels: [...ORBITAL_CHANNELS],
    weightsDimensions: {
      hiddenDim: weights.W_dos_hidden.length,
      outputBinsPerOrbital: DOS_BINS,
    },
  };
}

function parseFormula(formula: string): { element: string; count: number }[] {
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  const elements: { element: string; count: number }[] = [];
  let match;

  while ((match = regex.exec(formula)) !== null) {
    const element = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    if (element && ELEMENTAL_DATA[element]) {
      elements.push({ element, count });
    }
  }

  return elements;
}
