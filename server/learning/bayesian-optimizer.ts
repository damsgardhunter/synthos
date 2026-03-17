function expandParentheses(formula: string): string {
  let s = formula;
  let changed = true;
  let iterations = 0;
  while (changed && iterations < 20) {
    changed = false;
    iterations++;
    s = s.replace(/\(([^()]+)\)(\d*\.?\d*)/g, (_match, inner: string, mult: string) => {
      changed = true;
      const m = mult ? parseFloat(mult) : 1;
      if (m === 1) return inner;
      return inner.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_: string, el: string, cnt: string) => {
        const n = cnt ? parseFloat(cnt) : 1;
        return `${el}${(n * m).toFixed(4).replace(/\.?0+$/, "")}`;
      });
    });
  }
  return s;
}

function parseFormula(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const cleaned = expandParentheses(
    (typeof formula === "string" ? formula : String(formula ?? ""))
      .replace(/[₀-₉]/g, (c) => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
  );
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  return counts;
}

interface Observation {
  formula: string;
  features: number[];
  tc: number;
  lambda: number;
  stability: number;
  timestamp: number;
}

interface AcquisitionResult {
  formula: string;
  acquisitionValue: number;
  predictedMean: number;
  predictedStd: number;
  source: "ucb" | "ei" | "thompson" | "mixed";
}

interface GPPrediction {
  mean: number;
  std: number;
}

const SC_ELEMENTS = [
  "H", "Li", "Be", "B", "C", "N", "O", "F",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl",
  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br",
  "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
  "In", "Sn", "Sb", "Te",
  "Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
  "Tl", "Pb", "Bi"
];

const ELEMENT_INDEX: Map<string, number> = new Map();
SC_ELEMENTS.forEach((el, i) => ELEMENT_INDEX.set(el, i));

const ELEMENT_DIM = SC_ELEMENTS.length;
const UNKNOWN_EL_DIM = 1;
const STRUCT_DIM = 5;
const FEATURE_DIM = ELEMENT_DIM + UNKNOWN_EL_DIM + STRUCT_DIM;

function compositionToFeatures(formula: string): number[] {
  const features = new Array(FEATURE_DIM).fill(0);

  const parsed = parseFormula(formula);
  let totalAtoms = 0;
  const elements: string[] = [];
  for (const [el, count] of Object.entries(parsed)) {
    totalAtoms += count;
    elements.push(el);
  }
  if (totalAtoms === 0) return features;

  let unknownFrac = 0;
  for (const [el, count] of Object.entries(parsed)) {
    const idx = ELEMENT_INDEX.get(el);
    if (idx !== undefined) {
      features[idx] = count / totalAtoms;
    } else {
      unknownFrac += count / totalAtoms;
    }
  }
  features[ELEMENT_DIM] = unknownFrac;

  const structBase = ELEMENT_DIM + UNKNOWN_EL_DIM;
  const nElements = elements.length;
  features[structBase] = nElements === 2 ? 1 : 0;
  features[structBase + 1] = nElements === 3 ? 1 : 0;
  features[structBase + 2] = nElements >= 4 ? 1 : 0;

  const hasH = parsed["H"] !== undefined;
  features[structBase + 3] = hasH ? parsed["H"]! / totalAtoms : 0;

  const TM_3D = new Set(["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]);
  const TM_4D = new Set(["Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag"]);
  const TM_5D = new Set(["Hf","Ta","W","Re","Os","Ir","Pt","Au"]);
  let tmWeight = 0;
  for (const el of elements) {
    if (TM_3D.has(el)) tmWeight += 1.0;
    else if (TM_4D.has(el)) tmWeight += 0.7;
    else if (TM_5D.has(el)) tmWeight += 0.4;
  }
  features[structBase + 4] = tmWeight / Math.max(1, nElements);

  return features;
}

function maternKernel52ARD(
  x1: number[], x2: number[],
  lengthScales: number[], signalVariance: number
): number {
  let sqDist = 0;
  for (let i = 0; i < x1.length; i++) {
    const d = (x1[i] - x2[i]) / lengthScales[Math.min(i, lengthScales.length - 1)];
    sqDist += d * d;
  }
  const r = Math.sqrt(sqDist);
  const sqrt5r = Math.sqrt(5) * r;
  return signalVariance * (1 + sqrt5r + (5 / 3) * r * r) * Math.exp(-sqrt5r);
}

const N_LS_GROUPS = 3;
const LS_GROUP_ELEMENT = 0;
const LS_GROUP_UNKNOWN = 1;
const LS_GROUP_STRUCT = 2;

function buildLengthScaleArray(groupLS: number[]): number[] {
  const ls = new Array(FEATURE_DIM);
  for (let i = 0; i < ELEMENT_DIM; i++) ls[i] = groupLS[LS_GROUP_ELEMENT];
  ls[ELEMENT_DIM] = groupLS[LS_GROUP_UNKNOWN];
  for (let i = ELEMENT_DIM + UNKNOWN_EL_DIM; i < FEATURE_DIM; i++) ls[i] = groupLS[LS_GROUP_STRUCT];
  return ls;
}

const CHOLESKY_JITTER = 1e-6;
const CHOLESKY_MAX_RETRIES = 3;

function choleskyDecompose(K: number[][]): number[][] {
  const n = K.length;
  let jitter = CHOLESKY_JITTER;

  for (let attempt = 0; attempt <= CHOLESKY_MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      for (let i = 0; i < n; i++) K[i][i] += jitter;
      jitter *= 10;
    }

    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    let failed = false;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        if (i === j) {
          const diag = K[i][i] - sum;
          if (diag <= 0) {
            failed = true;
            break;
          }
          L[i][j] = Math.sqrt(diag);
        } else {
          L[i][j] = L[j][j] > 1e-15 ? (K[i][j] - sum) / L[j][j] : 0;
        }
      }
      if (failed) break;
    }

    if (!failed) return L;
  }

  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) L[i][i] = Math.sqrt(Math.max(K[i][i], 1e-6));
  return L;
}

function choleskySolve(L: number[][], b: number[]): number[] {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
    y[i] = L[i][i] > 1e-10 ? (b[i] - sum) / L[i][i] : 0;
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j];
    x[i] = L[i][i] > 1e-10 ? (y[i] - sum) / L[i][i] : 0;
  }
  return x;
}

function choleskyForwardSolve(L: number[][], b: number[]): number[] {
  const n = L.length;
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
    y[i] = L[i][i] > 1e-10 ? (b[i] - sum) / L[i][i] : 0;
  }
  return y;
}

const INV_SQRT_2 = 1 / Math.sqrt(2);
const INV_SQRT_2PI = 0.3989422804014327;

function erf(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

function standardNormalCDF(x: number): number {
  return 0.5 * (1 + erf(x * INV_SQRT_2));
}

function standardNormalPDF(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}

function tcToLog(tc: number): number {
  return Math.log(Math.max(tc, 0) + 1);
}

function logToTc(logTc: number): number {
  return Math.max(0, Math.exp(logTc) - 1);
}

export class BayesianOptimizer {
  private observations: Observation[] = [];
  private groupLengthScales = [0.5, 1.0, 0.3];
  private lengthScales: number[] = buildLengthScaleArray([0.5, 1.0, 0.3]);
  private signalVariance = 1.0;
  private noiseVariance = 0.1;
  private bestTcObserved = 0;
  private bestLogTcObserved = 0;
  private cachedL: number[][] | null = null;
  private cachedAlpha: number[] | null = null;
  private cachedYMean: number = 0;
  private maxObservations = 500;
  private featureCache = new Map<string, number[]>();

  private getFeatures(formula: string): number[] {
    let cached = this.featureCache.get(formula);
    if (!cached) {
      cached = compositionToFeatures(formula);
      this.featureCache.set(formula, cached);
      if (this.featureCache.size > 2000) {
        const first = this.featureCache.keys().next().value;
        if (first !== undefined) this.featureCache.delete(first);
      }
    }
    return cached;
  }

  addObservation(formula: string, tc: number, lambda: number = 0, stability: number = 0): void {
    const safeTc = (tc != null && Number.isFinite(tc)) ? tc : 0;
    const safeLambda = (lambda != null && Number.isFinite(lambda)) ? lambda : 0;
    const safeStability = (stability != null && Number.isFinite(stability)) ? stability : 0;

    const features = this.getFeatures(formula);
    const allZero = features.every(f => f === 0);
    if (allZero) return;

    const existing = this.observations.find(o => o.formula === formula);
    if (existing) {
      if (safeTc > existing.tc) {
        existing.tc = safeTc;
        existing.lambda = safeLambda;
        existing.stability = safeStability;
        existing.timestamp = Date.now();
        this.cachedL = null;
        this.cachedAlpha = null;
        this.cachedGPObs = null;
      }
      return;
    }

    this.observations.push({
      formula,
      features,
      tc: safeTc,
      lambda: safeLambda,
      stability: safeStability,
      timestamp: Date.now(),
    });

    if (safeTc > this.bestTcObserved) {
      this.bestTcObserved = safeTc;
      this.bestLogTcObserved = tcToLog(safeTc);
    }

    if (this.observations.length > this.maxObservations) {
      this.observations.sort((a, b) => b.tc - a.tc);
      const keepCount = Math.floor(this.maxObservations * 0.7);
      const topSlice = this.observations.slice(0, keepCount);
      const bottomSlice = this.observations.slice(keepCount);

      const targetFromBottom = this.maxObservations - keepCount;
      const diverseKeep = this.diversityPrune(bottomSlice, targetFromBottom);
      this.observations = [...topSlice, ...diverseKeep];

      if (this.observations.length > 0) {
        this.bestTcObserved = Math.max(...this.observations.map(o => o.tc));
        this.bestLogTcObserved = tcToLog(this.bestTcObserved);
      } else {
        this.bestTcObserved = 0;
        this.bestLogTcObserved = 0;
      }
    }

    this.cachedL = null;
    this.cachedAlpha = null;
    this.cachedGPObs = null;

    if (this.observations.length > 50 && this.observations.length % 25 === 0) {
      this.adaptLengthScale();
    }
  }

  private diversityPrune(pool: Observation[], keep: number): Observation[] {
    if (pool.length <= keep) return pool;

    const n = pool.length;
    const distSq = (a: number[], b: number[]): number => {
      let s = 0;
      for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
      }
      return s;
    };

    const minDists = new Float64Array(n).fill(Infinity);
    const selected = new Uint8Array(n);
    const result: Observation[] = [];

    let firstIdx = 0;
    for (let i = 1; i < n; i++) {
      if (pool[i].tc > pool[firstIdx].tc) firstIdx = i;
    }
    selected[firstIdx] = 1;
    result.push(pool[firstIdx]);

    for (let i = 0; i < n; i++) {
      if (i !== firstIdx) {
        minDists[i] = distSq(pool[i].features, pool[firstIdx].features);
      }
    }

    while (result.length < keep) {
      let farthestIdx = -1;
      let farthestDist = -1;
      for (let i = 0; i < n; i++) {
        if (!selected[i] && minDists[i] > farthestDist) {
          farthestDist = minDists[i];
          farthestIdx = i;
        }
      }
      if (farthestIdx < 0) break;

      selected[farthestIdx] = 1;
      result.push(pool[farthestIdx]);

      for (let i = 0; i < n; i++) {
        if (!selected[i]) {
          const d = distSq(pool[i].features, pool[farthestIdx].features);
          if (d < minDists[i]) minDists[i] = d;
        }
      }
    }

    return result;
  }

  private computeLogMarginalLikelihood(obs: Observation[], groupLS: number[]): number {
    const n = obs.length;
    if (n === 0) return -Infinity;

    const ls = buildLengthScaleArray(groupLS);
    const K: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const k = maternKernel52ARD(obs[i].features, obs[j].features, ls, this.signalVariance);
        K[i][j] = k;
        K[j][i] = k;
      }
      K[i][i] += this.noiseVariance;
    }

    const L = choleskyDecompose(K);

    const yRaw = obs.map(o => tcToLog(o.tc));
    const yMean = yRaw.reduce((s, v) => s + v, 0) / n;
    const y = yRaw.map(v => v - yMean);
    const alpha = choleskySolve(L, y);

    let dataFit = 0;
    for (let i = 0; i < n; i++) dataFit += y[i] * alpha[i];

    let logDet = 0;
    for (let i = 0; i < n; i++) logDet += Math.log(Math.max(L[i][i], 1e-15));
    logDet *= 2;

    return -0.5 * dataFit - 0.5 * logDet - 0.5 * n * Math.log(2 * Math.PI);
  }

  private adaptLengthScale(): void {
    const subset = this.observations.length > 80
      ? this.observations.slice(-80)
      : this.observations;

    const clampMin = [0.15, 0.3, 0.1];
    const clampMax = [2.0, 3.0, 1.5];

    const currentLML = this.computeLogMarginalLikelihood(subset, this.groupLengthScales);

    const perturbFactors = [0.8, 1.25];
    let bestLML = currentLML;
    let bestGroup = [...this.groupLengthScales];

    for (let g = 0; g < N_LS_GROUPS; g++) {
      for (const factor of perturbFactors) {
        const candidate = [...this.groupLengthScales];
        candidate[g] = Math.max(clampMin[g], Math.min(clampMax[g], candidate[g] * factor));
        if (candidate[g] === this.groupLengthScales[g]) continue;

        const lml = this.computeLogMarginalLikelihood(subset, candidate);
        if (lml > bestLML) {
          bestLML = lml;
          bestGroup = candidate;
        }
      }
    }

    if (bestLML > currentLML) {
      this.groupLengthScales = bestGroup;
      this.lengthScales = buildLengthScaleArray(this.groupLengthScales);
    }

    this.cachedL = null;
    this.cachedAlpha = null;
    this.cachedGPObs = null;
  }

  private cachedGPObs: Observation[] | null = null;

  private buildGP(): { L: number[][]; alpha: number[]; obs: Observation[] } {
    if (this.cachedL && this.cachedAlpha && this.cachedGPObs) {
      return { L: this.cachedL, alpha: this.cachedAlpha, obs: this.cachedGPObs };
    }

    const n = this.observations.length;
    if (n === 0) {
      this.cachedL = [];
      this.cachedAlpha = [];
      this.cachedGPObs = [];
      return { L: [], alpha: [], obs: [] };
    }

    let obs: typeof this.observations;
    if (n > 200) {
      const sorted = [...this.observations].sort((a, b) => b.tc - a.tc);
      const topSlice = sorted.slice(0, 140);
      const remaining = sorted.slice(140);
      const diverseRemaining = this.diversityPrune(remaining, 60);
      obs = [...topSlice, ...diverseRemaining];
    } else {
      obs = [...this.observations];
    }

    const K: number[][] = Array.from({ length: obs.length }, () => new Array(obs.length).fill(0));
    for (let i = 0; i < obs.length; i++) {
      for (let j = i; j < obs.length; j++) {
        const k = maternKernel52ARD(obs[i].features, obs[j].features, this.lengthScales, this.signalVariance);
        K[i][j] = k;
        K[j][i] = k;
      }
      K[i][i] += this.noiseVariance;
    }

    const L = choleskyDecompose(K);
    const yRaw = obs.map(o => tcToLog(o.tc));
    const yMean = yRaw.reduce((s, v) => s + v, 0) / yRaw.length;
    const y = yRaw.map(v => v - yMean);
    const alpha = choleskySolve(L, y);

    this.cachedL = L;
    this.cachedAlpha = alpha;
    this.cachedYMean = yMean;
    this.cachedGPObs = obs;
    return { L, alpha, obs };
  }

  predict(formula: string): GPPrediction {
    const features = this.getFeatures(formula);
    return this.predictFromFeatures(features);
  }

  private predictLogSpace(features: number[]): GPPrediction {
    const { L, alpha, obs: usedObs } = this.buildGP();
    const yMean = this.cachedYMean;

    const kStar: number[] = new Array(usedObs.length);
    for (let i = 0; i < usedObs.length; i++) {
      kStar[i] = maternKernel52ARD(features, usedObs[i].features, this.lengthScales, this.signalVariance);
    }

    let mean = yMean;
    for (let i = 0; i < usedObs.length; i++) {
      mean += kStar[i] * alpha[i];
    }

    const v = choleskyForwardSolve(L, kStar);
    let variance = this.signalVariance;
    for (let i = 0; i < v.length; i++) {
      variance -= v[i] * v[i];
    }
    const MIN_VARIANCE = 1e-6;
    variance = Math.max(variance, MIN_VARIANCE);

    return { mean, std: Math.sqrt(variance) };
  }

  private predictFromFeatures(features: number[]): GPPrediction {
    const n = this.observations.length;
    if (n === 0) {
      return { mean: 0, std: Math.sqrt(this.signalVariance) };
    }

    const logPred = this.predictLogSpace(features);
    const tcMean = logToTc(logPred.mean);
    const tcStd = tcMean * (Math.exp(logPred.std) - 1);

    return { mean: tcMean, std: Math.max(tcStd, 1e-3) };
  }

  acquisitionUCB(formula: string, beta: number = 2.0): number {
    const { mean, std } = this.predict(formula);
    return mean + beta * std;
  }

  acquisitionEI(formula: string, xi: number = 0.01): number {
    if (this.observations.length === 0) return 0;
    const features = this.getFeatures(formula);
    const logPred = this.predictLogSpace(features);
    if (logPred.std < 1e-3) return 0;
    const logXi = xi > 0 ? Math.log(1 + xi) : 0;
    const improvement = logPred.mean - this.bestLogTcObserved - logXi;
    const z = improvement / logPred.std;
    const eiLog = improvement * standardNormalCDF(z) + logPred.std * standardNormalPDF(z);
    if (!Number.isFinite(eiLog) || eiLog <= 0) return 0;
    return logToTc(this.bestLogTcObserved + eiLog) - this.bestTcObserved;
  }

  thompsonSample(formula: string): number {
    if (this.observations.length === 0) return 0;
    const features = this.getFeatures(formula);
    const logPred = this.predictLogSpace(features);
    const u1 = 1 - Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const logSample = logPred.mean + logPred.std * z;
    const sample = logToTc(logSample);
    return Number.isFinite(sample) ? sample : logToTc(logPred.mean);
  }

  suggestNextCandidates(
    candidatePool: string[],
    nSuggestions: number = 10,
    method: "ucb" | "ei" | "thompson" | "mixed" = "mixed"
  ): AcquisitionResult[] {
    if (this.observations.length < 5) {
      const VEC_MAP: Record<string, number> = {
        H:1,Li:1,Be:2,B:3,C:4,N:5,O:6,F:7,Na:1,Mg:2,Al:3,Si:4,P:5,S:6,Cl:7,
        K:1,Ca:2,Sc:3,Ti:4,V:5,Cr:6,Mn:7,Fe:8,Co:9,Ni:10,Cu:11,Zn:12,
        Ga:3,Ge:4,As:5,Se:6,Br:7,Rb:1,Sr:2,Y:3,Zr:4,Nb:5,Mo:6,Ru:8,Rh:9,Pd:10,Ag:11,
        In:3,Sn:4,Sb:5,Te:6,Cs:1,Ba:2,La:3,Hf:4,Ta:5,W:6,Re:7,Os:8,Ir:9,Pt:10,Au:11,
        Tl:3,Pb:4,Bi:5,
      };
      const ranked = candidatePool.map(f => {
        const parsed = parseFormula(f);
        const entries = Object.entries(parsed);
        const totalAtoms = entries.reduce((s, [, c]) => s + c, 0);
        if (totalAtoms === 0) return { formula: f, score: -1 };
        const vec = entries.reduce((s, [el, c]) => s + (VEC_MAP[el] ?? 4) * c, 0) / totalAtoms;
        const hasTM = entries.some(([el]) =>
          ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
           "Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag",
           "Hf","Ta","W","Re","Os","Ir","Pt","Au"].includes(el));
        const features = this.getFeatures(f);
        const allZero = features.every(v => v === 0);
        if (allZero) return { formula: f, score: -1 };
        const vecPenalty = Math.abs(vec - 4.7);
        const tmBonus = hasTM ? 1.0 : 0;
        const nElBonus = entries.length >= 2 && entries.length <= 5 ? 0.5 : 0;
        return { formula: f, score: tmBonus + nElBonus - vecPenalty * 0.3 + Math.random() * 0.2 };
      });
      ranked.sort((a, b) => b.score - a.score);
      return ranked.slice(0, nSuggestions).map(r => ({
        formula: r.formula,
        acquisitionValue: Math.max(0, r.score),
        predictedMean: 0,
        predictedStd: 1,
        source: "ucb" as const,
      }));
    }

    const scored: AcquisitionResult[] = [];

    let filteredPool = candidatePool;
    try {
      const { getAlreadyScreenedFormulas } = require("./engine");
      const { normalizeFormula } = require("./utils");
      const screened = getAlreadyScreenedFormulas();
      const novel = candidatePool.filter(f => !screened.has(normalizeFormula(f)));
      if (novel.length > 0) {
        filteredPool = novel;
      }
    } catch {}

    const tcRange = Math.max(this.bestTcObserved, 1);

    for (const formula of filteredPool) {
      const { mean, std } = this.predict(formula);

      let acqValue: number;
      let source: "ucb" | "ei" | "thompson" | "mixed";

      if (method === "mixed") {
        const ucb = (mean + 2.0 * std) / tcRange;
        const ei = this.acquisitionEI(formula) / tcRange;
        const ts = this.thompsonSample(formula) / tcRange;
        acqValue = ucb * 0.4 + ei * 0.3 + ts * 0.3;
        source = "mixed";
      } else if (method === "ucb") {
        acqValue = this.acquisitionUCB(formula) / tcRange;
        source = "ucb";
      } else if (method === "ei") {
        acqValue = this.acquisitionEI(formula) / tcRange;
        source = "ei";
      } else {
        acqValue = this.thompsonSample(formula) / tcRange;
        source = "thompson";
      }

      scored.push({ formula, acquisitionValue: acqValue, predictedMean: mean, predictedStd: std, source });
    }

    scored.sort((a, b) => b.acquisitionValue - a.acquisitionValue);

    const selected: AcquisitionResult[] = [];
    const seenFormulas = new Set<string>();
    for (const s of scored) {
      if (!seenFormulas.has(s.formula)) {
        selected.push(s);
        seenFormulas.add(s.formula);
        if (selected.length >= nSuggestions) break;
      }
    }

    return selected;
  }

  // Generates novel candidate formulas by exploring the neighbourhood of the
  // best observed compounds, then scores them with the GP acquisition function.
  // This makes BO a true proposer rather than just a ranker of external pools.
  generateCandidates(nCandidates: number = 20): AcquisitionResult[] {
    if (this.observations.length < 3) return [];

    // Isoelectronic / isovalent substitution groups — swapping within a group
    // preserves the broad chemical class while exploring new compositions.
    const SUB_GROUPS: string[][] = [
      ["Li","Na","K","Rb","Cs"],
      ["Be","Mg","Ca","Sr","Ba"],
      ["Sc","Y","La"],
      ["Ti","Zr","Hf"],
      ["V","Nb","Ta"],
      ["Cr","Mo","W"],
      ["Mn","Re"],
      ["Fe","Ru","Os"],
      ["Co","Rh","Ir"],
      ["Ni","Pd","Pt"],
      ["Cu","Ag","Au"],
      ["B","Al","Ga","In","Tl"],
      ["C","Si","Ge","Sn","Pb"],
      ["N","P","As","Sb","Bi"],
      ["O","S","Se","Te"],
      ["F","Cl","Br"],
    ];

    // Elements commonly found in high-Tc environments — used for addition moves.
    const HIGH_TC_ADJACENT = [
      "H","B","C","N","O","S","Se","Te","As","P","F",
      "Cu","Fe","Nb","La","Ba","Sr","Ca","K",
    ];

    function buildFormula(counts: Record<string, number>): string {
      return Object.entries(counts)
        .filter(([, c]) => c > 0)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([el, c]) => c === 1 ? el : `${el}${c}`)
        .join("");
    }

    // Seed from the top observed formulas (by Tc).
    const topObs = [...this.observations]
      .sort((a, b) => b.tc - a.tc)
      .slice(0, Math.min(8, this.observations.length));

    const candidates = new Set<string>();

    for (const obs of topObs) {
      const counts = parseFormula(obs.formula);
      const elements = Object.keys(counts);

      // 1. Element substitution within the same chemical family
      for (const el of elements) {
        const group = SUB_GROUPS.find(g => g.includes(el));
        if (!group) continue;
        for (const sub of group) {
          if (sub === el || !SC_ELEMENTS.includes(sub)) continue;
          const nc = { ...counts, [sub]: counts[el] };
          delete nc[el];
          candidates.add(buildFormula(nc));
        }
      }

      // 2. Stoichiometry walk — nudge each element count by ±1
      for (const el of elements) {
        for (const delta of [-1, 1]) {
          const newCount = (counts[el] || 1) + delta;
          if (newCount < 1 || newCount > 8) continue;
          candidates.add(buildFormula({ ...counts, [el]: newCount }));
        }
      }

      // 3. Element addition (only for binary/ternary seeds to keep complexity down)
      if (elements.length <= 3) {
        for (const add of HIGH_TC_ADJACENT) {
          if (elements.includes(add)) continue;
          candidates.add(buildFormula({ ...counts, [add]: 1 }));
          candidates.add(buildFormula({ ...counts, [add]: 2 }));
        }
      }

      // 4. Element removal — simplify to find promising sub-compounds
      if (elements.length >= 3) {
        for (const drop of elements) {
          const nc = { ...counts };
          delete nc[drop];
          if (Object.keys(nc).length >= 2) candidates.add(buildFormula(nc));
        }
      }
    }

    // Filter to chemically valid formulas before acquisition scoring.
    const pool = [...candidates].filter(f => {
      const c = parseFormula(f);
      const els = Object.keys(c);
      const total = Object.values(c).reduce((s, n) => s + n, 0);
      return els.length >= 2 && els.length <= 5 && total <= 12 &&
        els.every(el => SC_ELEMENTS.includes(el));
    });

    if (pool.length === 0) return [];
    return this.suggestNextCandidates(pool, nCandidates, "mixed");
  }

  getStats(): {
    observationCount: number;
    bestTc: number;
    avgTc: number;
    explorationRatio: number;
  } {
    const avgTc = this.observations.length > 0
      ? this.observations.reduce((s, o) => s + o.tc, 0) / this.observations.length
      : 0;

    const recentObs = this.observations.filter(o => o.timestamp > Date.now() - 600000);
    const highUncertaintyCount = recentObs.filter(o => {
      const { std } = this.predictFromFeatures(o.features);
      return std > 0.5 * Math.sqrt(this.signalVariance);
    }).length;
    const explorationRatio = recentObs.length > 0 ? highUncertaintyCount / recentObs.length : 0.5;

    return {
      observationCount: this.observations.length,
      bestTc: this.bestTcObserved,
      avgTc: Math.round(avgTc * 10) / 10,
      explorationRatio: Math.round(explorationRatio * 1000) / 1000,
    };
  }
}

export const bayesianOptimizer = new BayesianOptimizer();
