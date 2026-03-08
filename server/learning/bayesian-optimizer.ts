function parseFormula(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
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
  source: "ucb" | "ei" | "thompson";
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

  const tmCount = elements.filter(el =>
    ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
     "Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag",
     "Hf","Ta","W","Re","Os","Ir","Pt","Au"].includes(el)
  ).length;
  features[structBase + 4] = tmCount / Math.max(1, nElements);

  return features;
}

function rbfKernel(x1: number[], x2: number[], lengthScale: number, signalVariance: number): number {
  let sqDist = 0;
  for (let i = 0; i < x1.length; i++) {
    const d = x1[i] - x2[i];
    sqDist += d * d;
  }
  return signalVariance * Math.exp(-0.5 * sqDist / (lengthScale * lengthScale));
}

function maternKernel52(x1: number[], x2: number[], lengthScale: number, signalVariance: number): number {
  let sqDist = 0;
  for (let i = 0; i < x1.length; i++) {
    const d = x1[i] - x2[i];
    sqDist += d * d;
  }
  const r = Math.sqrt(sqDist) / lengthScale;
  const sqrt5r = Math.sqrt(5) * r;
  return signalVariance * (1 + sqrt5r + (5 / 3) * r * r) * Math.exp(-sqrt5r);
}

function choleskyDecompose(K: number[][]): number[][] {
  const n = K.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      if (i === j) {
        const diag = K[i][i] - sum;
        L[i][j] = diag > 1e-10 ? Math.sqrt(diag) : 1e-4;
      } else {
        L[i][j] = L[j][j] > 1e-10 ? (K[i][j] - sum) / L[j][j] : 0;
      }
    }
  }
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

function standardNormalCDF(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327;
  const p = d * Math.exp(-x * x / 2) * t *
    (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

function standardNormalPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

export class BayesianOptimizer {
  private observations: Observation[] = [];
  private lengthScale = 0.5;
  private signalVariance = 1.0;
  private noiseVariance = 0.1;
  private bestTcObserved = 0;
  private cachedL: number[][] | null = null;
  private cachedAlpha: number[] | null = null;
  private maxObservations = 500;

  addObservation(formula: string, tc: number, lambda: number = 0, stability: number = 0): void {
    const safeTc = (tc != null && Number.isFinite(tc)) ? tc : 0;
    const safeLambda = (lambda != null && Number.isFinite(lambda)) ? lambda : 0;
    const safeStability = (stability != null && Number.isFinite(stability)) ? stability : 0;

    const features = compositionToFeatures(formula);
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
    }

    if (this.observations.length > this.maxObservations) {
      this.observations.sort((a, b) => b.tc - a.tc);
      const topHalf = this.observations.slice(0, Math.floor(this.maxObservations * 0.7));
      const bottomHalf = this.observations.slice(Math.floor(this.maxObservations * 0.7));
      const sampled = bottomHalf.filter(() => Math.random() < 0.5);
      this.observations = [...topHalf, ...sampled].slice(0, this.maxObservations);
    }

    this.cachedL = null;
    this.cachedAlpha = null;
    this.cachedGPObs = null;

    if (this.observations.length > 50 && this.observations.length % 25 === 0) {
      this.adaptLengthScale();
    }
  }

  private adaptLengthScale(): void {
    const recent = this.observations.slice(-20);
    const predictions = recent.map(o => this.predictFromFeatures(o.features));
    const avgStd = predictions.reduce((s, p) => s + p.std, 0) / predictions.length;
    const avgMean = predictions.reduce((s, p) => s + Math.abs(p.mean), 0) / predictions.length;
    const relUncertainty = avgMean > 0 ? avgStd / avgMean : avgStd;
    if (relUncertainty > 0.8) {
      this.lengthScale = Math.min(2.0, this.lengthScale * 1.15);
    } else if (relUncertainty < 0.2) {
      this.lengthScale = Math.max(0.15, this.lengthScale * 0.9);
    }
    this.cachedL = null;
    this.cachedAlpha = null;
    this.cachedGPObs = null;
  }

  private cachedGPObs: BayesianObservation[] | null = null;

  private buildGP(): { L: number[][]; alpha: number[]; obs: BayesianObservation[] } {
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
      const randomSample: typeof this.observations = [];
      for (let i = remaining.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
      }
      randomSample.push(...remaining.slice(0, 60));
      obs = [...topSlice, ...randomSample];
    } else {
      obs = [...this.observations];
    }

    const K: number[][] = Array.from({ length: obs.length }, () => new Array(obs.length).fill(0));
    for (let i = 0; i < obs.length; i++) {
      for (let j = i; j < obs.length; j++) {
        const k = maternKernel52(obs[i].features, obs[j].features, this.lengthScale, this.signalVariance);
        K[i][j] = k;
        K[j][i] = k;
      }
      K[i][i] += this.noiseVariance;
    }

    const L = choleskyDecompose(K);
    const y = obs.map(o => o.tc);
    const alpha = choleskySolve(L, y);

    this.cachedL = L;
    this.cachedAlpha = alpha;
    this.cachedGPObs = obs;
    return { L, alpha, obs };
  }

  predict(formula: string): GPPrediction {
    const features = compositionToFeatures(formula);
    return this.predictFromFeatures(features);
  }

  private predictFromFeatures(features: number[]): GPPrediction {
    const n = this.observations.length;
    if (n === 0) {
      return { mean: 0, std: Math.sqrt(this.signalVariance) };
    }

    const { L, alpha, obs: usedObs } = this.buildGP();

    const kStar: number[] = new Array(usedObs.length);
    for (let i = 0; i < usedObs.length; i++) {
      kStar[i] = maternKernel52(features, usedObs[i].features, this.lengthScale, this.signalVariance);
    }

    let mean = 0;
    for (let i = 0; i < usedObs.length; i++) {
      mean += kStar[i] * alpha[i];
    }

    const v = choleskyForwardSolve(L, kStar);
    let variance = this.signalVariance;
    for (let i = 0; i < v.length; i++) {
      variance -= v[i] * v[i];
    }
    variance = Math.max(variance, 1e-6);

    return { mean: Math.max(0, mean), std: Math.sqrt(variance) };
  }

  acquisitionUCB(formula: string, beta: number = 2.0): number {
    const { mean, std } = this.predict(formula);
    return mean + beta * std;
  }

  acquisitionEI(formula: string, xi: number = 0.01): number {
    const { mean, std } = this.predict(formula);
    if (std < 1e-8) return 0;
    const improvement = mean - this.bestTcObserved - xi;
    const z = improvement / std;
    const result = improvement * standardNormalCDF(z) + std * standardNormalPDF(z);
    return Number.isFinite(result) ? result : 0;
  }

  thompsonSample(formula: string): number {
    const { mean, std } = this.predict(formula);
    const u1 = Math.max(1e-15, Math.random());
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const sample = mean + std * z;
    return Number.isFinite(sample) ? sample : mean;
  }

  suggestNextCandidates(
    candidatePool: string[],
    nSuggestions: number = 10,
    method: "ucb" | "ei" | "thompson" | "mixed" = "mixed"
  ): AcquisitionResult[] {
    if (this.observations.length < 5) {
      return candidatePool.slice(0, nSuggestions).map(f => ({
        formula: f,
        acquisitionValue: Math.random(),
        predictedMean: 0,
        predictedStd: 1,
        source: "ucb" as const,
      }));
    }

    const scored: AcquisitionResult[] = [];

    for (const formula of candidatePool) {
      const { mean, std } = this.predict(formula);

      let acqValue: number;
      let source: "ucb" | "ei" | "thompson";

      if (method === "mixed") {
        const ucb = mean + 2.0 * std;
        const ei = this.acquisitionEI(formula);
        const ts = this.thompsonSample(formula);
        acqValue = ucb * 0.4 + ei * 0.3 + ts * 0.3;
        source = ucb >= ei && ucb >= ts ? "ucb" : ei >= ts ? "ei" : "thompson";
      } else if (method === "ucb") {
        acqValue = this.acquisitionUCB(formula);
        source = "ucb";
      } else if (method === "ei") {
        acqValue = this.acquisitionEI(formula);
        source = "ei";
      } else {
        acqValue = this.thompsonSample(formula);
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
