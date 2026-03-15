import { encodeGenome } from "./materials-genome";
import { computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling } from "../learning/physics-engine";
import { isTransitionMetal, isRareEarth } from "../learning/elemental-data";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";
import { isValidFormula } from "../learning/utils";

const GENOME_DIM = 256;
const LATENT_DIM = 64;
const HIDDEN_DIM = 128;

let vaeBeta = 1.0;

export function setVAEBeta(beta: number): void {
  vaeBeta = Math.max(0.01, Math.min(10.0, beta));
}

export function getVAEBeta(): number {
  return vaeBeta;
}

interface VAEWeights {
  W_enc1: number[][];
  b_enc1: number[];
  W_enc2: number[][];
  b_enc2: number[];
  W_mu: number[][];
  b_mu: number[];
  W_logvar: number[][];
  b_logvar: number[];
  W_dec1: number[][];
  b_dec1: number[];
  W_dec2: number[][];
  b_dec2: number[];
}

export interface VAELatentPoint {
  z: number[];
  mu: number[];
  logvar: number[];
  reconstructionLoss: number;
  klDivergence: number;
}

export interface InverseDesignResult {
  startZ: number[];
  finalZ: number[];
  optimizationSteps: number;
  decodedFormulas: string[];
  bestFormula: string;
  bestTc: number;
  bestLambda: number;
  trajectory: { step: number; loss: number; tc: number; formula: string }[];
  converged: boolean;
}

function gaussRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(Math.max(1e-10, u1))) * Math.cos(2 * Math.PI * u2);
}

function initMatrix(rows: number, cols: number, scale: number = 0.1): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(gaussRandom() * scale);
    }
    m.push(row);
  }
  return m;
}

function initVector(dim: number): number[] {
  return new Array(dim).fill(0);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
  return mat.map(row => {
    let s = 0;
    for (let i = 0; i < Math.min(row.length, vec.length); i++) {
      s += row[i] * (vec[i] ?? 0);
    }
    return s;
  });
}

function vecAdd(a: number[], b: number[]): number[] {
  const len = Math.max(a.length, b.length);
  const result: number[] = [];
  for (let i = 0; i < len; i++) {
    result.push((a[i] ?? 0) + (b[i] ?? 0));
  }
  return result;
}

function relu(vec: number[]): number[] {
  return vec.map(v => Math.max(0, v));
}

function leakyRelu(vec: number[], alpha: number = 0.01): number[] {
  return vec.map(v => v > 0 ? v : alpha * v);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function tanh(vec: number[]): number[] {
  return vec.map(v => Math.tanh(Math.max(-10, Math.min(10, v))));
}

let vaeWeights: VAEWeights | null = null;

function getVAEWeights(): VAEWeights {
  if (vaeWeights) return vaeWeights;

  vaeWeights = {
    W_enc1: initMatrix(HIDDEN_DIM, GENOME_DIM, Math.sqrt(2 / GENOME_DIM)),
    b_enc1: initVector(HIDDEN_DIM),
    W_enc2: initMatrix(HIDDEN_DIM, HIDDEN_DIM, Math.sqrt(2 / HIDDEN_DIM)),
    b_enc2: initVector(HIDDEN_DIM),
    W_mu: initMatrix(LATENT_DIM, HIDDEN_DIM, Math.sqrt(2 / HIDDEN_DIM)),
    b_mu: initVector(LATENT_DIM),
    W_logvar: initMatrix(LATENT_DIM, HIDDEN_DIM, Math.sqrt(2 / HIDDEN_DIM)),
    b_logvar: initVector(LATENT_DIM),
    W_dec1: initMatrix(HIDDEN_DIM, LATENT_DIM, Math.sqrt(2 / LATENT_DIM)),
    b_dec1: initVector(HIDDEN_DIM),
    W_dec2: initMatrix(GENOME_DIM, HIDDEN_DIM, Math.sqrt(2 / HIDDEN_DIM)),
    b_dec2: initVector(GENOME_DIM),
  };

  return vaeWeights;
}

function encode(genome: number[]): { mu: number[]; logvar: number[] } {
  const w = getVAEWeights();

  const h1 = leakyRelu(vecAdd(matVecMul(w.W_enc1, genome), w.b_enc1));
  const h2 = leakyRelu(vecAdd(matVecMul(w.W_enc2, h1), w.b_enc2));

  const mu = vecAdd(matVecMul(w.W_mu, h2), w.b_mu);
  const logvar = vecAdd(matVecMul(w.W_logvar, h2), w.b_logvar);

  const clampedLogvar = logvar.map(v => Math.max(-10, Math.min(5, v)));

  return { mu, logvar: clampedLogvar };
}

function reparameterize(mu: number[], logvar: number[]): number[] {
  return mu.map((m, i) => {
    const std = Math.exp(0.5 * (logvar[i] ?? -2));
    return m + gaussRandom() * std;
  });
}

const POSITIVE_DIMS = new Set<number>();
(function initPositiveDims() {
  for (let i = 0; i <= 7; i++) POSITIVE_DIMS.add(i);
  for (let i = 40; i <= 51; i++) POSITIVE_DIMS.add(i);
  for (let i = 168; i <= 207; i++) POSITIVE_DIMS.add(i);
})();

function adaptiveOutput(vec: number[]): number[] {
  return vec.map((v, i) => {
    const clamped = Math.max(-10, Math.min(10, v));
    if (POSITIVE_DIMS.has(i)) {
      return 1 / (1 + Math.exp(-clamped));
    }
    return Math.tanh(clamped);
  });
}

function decode(z: number[]): number[] {
  const w = getVAEWeights();

  const h1 = leakyRelu(vecAdd(matVecMul(w.W_dec1, z), w.b_dec1));
  const out = adaptiveOutput(vecAdd(matVecMul(w.W_dec2, h1), w.b_dec2));

  return out;
}

function reconstructionLoss(original: number[], reconstructed: number[]): number {
  let loss = 0;
  for (let i = 0; i < Math.min(original.length, reconstructed.length); i++) {
    const diff = (original[i] ?? 0) - (reconstructed[i] ?? 0);
    loss += diff * diff;
  }
  return loss / original.length;
}

function klDivergence(mu: number[], logvar: number[]): number {
  let kl = 0;
  for (let i = 0; i < mu.length; i++) {
    kl += -0.5 * (1 + (logvar[i] ?? 0) - (mu[i] ?? 0) ** 2 - Math.exp(logvar[i] ?? 0));
  }
  return kl / mu.length;
}

export function encodeToLatent(formula: string): VAELatentPoint {
  const genome = encodeGenome(formula);
  const { mu, logvar } = encode(genome.vector);
  const z = reparameterize(mu, logvar);

  const reconstructed = decode(z);
  const recLoss = reconstructionLoss(genome.vector, reconstructed);
  const kl = klDivergence(mu, logvar);

  return { z, mu, logvar, reconstructionLoss: recLoss, klDivergence: kl * vaeBeta };
}

export function decodeFromLatent(z: number[]): number[] {
  return decode(z);
}

const DECODE_ELEMENT_POOL = [
  "La", "Y", "Ce", "Ba", "Sr", "Ca", "Sc",
  "Nb", "V", "Ti", "Ta", "Mo", "W", "Zr", "Hf",
  "Fe", "Co", "Ni", "Cu", "Mn", "Cr",
  "H", "B", "C", "N",
  "O", "S", "Se", "Te",
  "As", "P", "Sb", "Bi",
  "Al", "Ga", "Ge", "Sn", "Si", "Pb", "In",
  "Mg",
];

function genomeVectorToFormula(genomeVec: number[]): string {
  const structureSegment = genomeVec.slice(0, 40);
  const orbitalSegment = genomeVec.slice(40, 76);
  const phononSegment = genomeVec.slice(76, 108);
  const couplingSegment = genomeVec.slice(108, 140);
  const compositionSegment = genomeVec.slice(168, 208);

  const metalFrac = Math.max(0, Math.min(1, structureSegment[2] ?? 0.5));
  const hFrac = Math.max(0, Math.min(0.7, structureSegment[3] ?? 0));
  const nElements = Math.max(2, Math.min(5, Math.round((compositionSegment[6] ?? 0.3) * 10)));
  const totalAtoms = Math.max(2, Math.min(16, Math.round((structureSegment[1] ?? 0.5) * 20)));

  const dFrac = orbitalSegment[3] ?? 0;
  const hasDElectrons = dFrac > 0.3;
  const lambdaIndicator = couplingSegment[0] ?? 0;
  const highLambda = lambdaIndicator > 0.3;

  const elementScores: { el: string; score: number }[] = [];
  for (const el of DECODE_ELEMENT_POOL) {
    let score = 0;

    if (isTransitionMetal(el)) {
      score += metalFrac * 3;
      if (hasDElectrons) score += 1.5;
      if (highLambda && ["Nb", "V", "Ta", "Mo"].includes(el)) score += 2;
    }

    if (isRareEarth(el)) {
      score += metalFrac * 2;
    }

    if (el === "H") {
      score += hFrac * 8;
      if (highLambda) score += 2;
    }

    if (["B", "C", "N"].includes(el)) {
      const phononIndicator = phononSegment[0] ?? 0;
      score += phononIndicator * 2;
      if (highLambda) score += 1;
    }

    if (["O", "S", "Se", "Te"].includes(el)) {
      const oFrac = structureSegment[4] ?? 0;
      score += oFrac * 5;
    }

    score += (Math.random() - 0.3) * 0.5;
    elementScores.push({ el, score });
  }

  elementScores.sort((a, b) => b.score - a.score);
  const chosenElements = elementScores.slice(0, nElements).map(e => e.el);

  if (!chosenElements.some(e => isTransitionMetal(e) || isRareEarth(e))) {
    chosenElements[chosenElements.length - 1] = "Nb";
  }

  const ANION_SET = new Set(["O", "S", "Se", "Te", "N", "F", "Cl", "Br"]);
  const CATION_SET_FN = (el: string) => isTransitionMetal(el) || isRareEarth(el) ||
    ["Ba", "Sr", "Ca", "Mg", "Al", "Ga", "In", "Sn", "Pb", "Ge", "Si", "Sb", "Bi"].includes(el);

  const anionCount = chosenElements.filter(e => ANION_SET.has(e)).length;
  const cationCount = chosenElements.filter(e => CATION_SET_FN(e)).length;

  if (cationCount > 0 && anionCount === 0 && !chosenElements.includes("H")) {
    const anionRanked = elementScores
      .filter(e => ANION_SET.has(e.el) && !chosenElements.includes(e.el));
    const bestAnion = anionRanked.length > 0 ? anionRanked[0].el : "O";

    let worstIdx = chosenElements.length - 1;
    let worstScore = Infinity;
    for (let i = 0; i < chosenElements.length; i++) {
      const el = chosenElements[i];
      if (!CATION_SET_FN(el) && el !== "H") {
        const sc = elementScores.find(e => e.el === el)?.score ?? 0;
        if (sc < worstScore) { worstScore = sc; worstIdx = i; }
      }
    }
    if (CATION_SET_FN(chosenElements[worstIdx])) {
      const nonCationIdx = chosenElements.findIndex(e => !CATION_SET_FN(e) && e !== "H");
      if (nonCationIdx >= 0) worstIdx = nonCationIdx;
    }
    chosenElements[worstIdx] = bestAnion;
  }

  const counts: Record<string, number> = {};
  let remaining = totalAtoms;

  for (let i = 0; i < chosenElements.length; i++) {
    const el = chosenElements[i];
    let count: number;

    if (i === chosenElements.length - 1) {
      count = Math.max(1, remaining);
    } else if (el === "H") {
      count = Math.max(2, Math.min(6, Math.round(hFrac * totalAtoms)));
    } else if (isTransitionMetal(el) || isRareEarth(el)) {
      count = Math.max(1, Math.min(3, Math.round(metalFrac * totalAtoms / Math.max(1, chosenElements.filter(e => isTransitionMetal(e) || isRareEarth(e)).length))));
    } else {
      count = Math.max(1, Math.min(4, Math.round(remaining / (chosenElements.length - i))));
    }

    count = Math.max(1, Math.min(remaining - (chosenElements.length - i - 1), count));
    counts[el] = count;
    remaining -= count;
  }

  const metalOrder = ["La", "Y", "Ce", "Ba", "Sr", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Nb", "Mo", "Zr", "Hf", "Ta", "W"];
  const sorted = Object.entries(counts).sort((a, b) => {
    const ai = metalOrder.indexOf(a[0]);
    const bi = metalOrder.indexOf(b[0]);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return a[0].localeCompare(b[0]);
  });

  return sorted.map(([el, n]) => n === 1 ? el : `${el}${n}`).join("");
}

async function evaluateFormulaTc(formula: string): Promise<{ tc: number; lambda: number; score: number }> {
  try {
    const features = await extractFeatures(formula);
    const gb = await gbPredict(features);
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    let tc = gb.tcPredicted;
    if (coupling.lambda > 0.2) {
      const omegaLogK = coupling.omegaLog * 1.4388;
      const denom = coupling.lambda - coupling.muStar * (1 + 0.62 * coupling.lambda);
      if (Math.abs(denom) > 1e-6 && denom > 0) {
        const lambdaBar = 2.46 * (1 + 3.8 * coupling.muStar);
        const f1 = Math.pow(1 + Math.pow(coupling.lambda / lambdaBar, 3 / 2), 1 / 3);
        let f2 = 1.0;
        if (coupling.omega2Avg > 0 && coupling.omegaLog > 0) {
          const omegaRatio = Math.sqrt(coupling.omega2Avg) / coupling.omegaLog;
          const Lambda2 = 1.82 * (1 + 6.3 * coupling.muStar) * omegaRatio;
          f2 = 1 + (omegaRatio - 1) * coupling.lambda * coupling.lambda / (coupling.lambda * coupling.lambda + Lambda2 * Lambda2);
          f2 = Math.max(0.8, f2);
        }
        const exponent = -1.04 * (1 + coupling.lambda) / denom;
        const mcmillanTc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);
        if (Number.isFinite(mcmillanTc) && mcmillanTc > 0) {
          tc = Math.max(tc, mcmillanTc);
        }
      }
    }

    return {
      tc: Math.min(400, Math.max(0, tc)),
      lambda: coupling.lambda,
      score: gb.score,
    };
  } catch {
    return { tc: 0, lambda: 0, score: 0 };
  }
}

export async function runLatentSpaceInverseDesign(
  targetTc: number = 293,
  startFormula?: string,
  maxSteps: number = 50,
  learningRate: number = 0.02,
  nRestarts: number = 3,
): Promise<InverseDesignResult> {
  let bestOverallFormula = "";
  let bestOverallTc = 0;
  let bestOverallLambda = 0;
  let bestOverallZ: number[] = [];
  let bestTrajectory: InverseDesignResult["trajectory"] = [];
  let bestStartZ: number[] = [];

  for (let restart = 0; restart < nRestarts; restart++) {
    let z: number[];
    const startZ: number[] = [];

    if (startFormula && restart === 0) {
      const latent = encodeToLatent(startFormula);
      z = [...latent.mu];
    } else {
      z = new Array(LATENT_DIM).fill(0).map(() => gaussRandom() * 0.5);
    }

    for (let i = 0; i < z.length; i++) startZ.push(z[i]);

    const m = new Array(LATENT_DIM).fill(0);
    const v = new Array(LATENT_DIM).fill(0);
    let t = 0;

    let bestTc = 0;
    let bestLambdaRestart = 0;
    let bestFormula = "";
    let bestZ = [...z];
    let stagnation = 0;
    const trajectory: InverseDesignResult["trajectory"] = [];

    for (let step = 0; step < maxSteps; step++) {
      const genomeVec = decode(z);
      const formula = genomeVectorToFormula(genomeVec);

      if (!formula || !isValidFormula(formula)) {
        z = z.map((v, i) => v + gaussRandom() * 0.1);
        continue;
      }

      const evaluation = await evaluateFormulaTc(formula);
      const tcGap = Math.max(0, targetTc - evaluation.tc);
      const loss = (tcGap / targetTc) ** 2;

      trajectory.push({
        step,
        loss: Math.round(loss * 10000) / 10000,
        tc: Math.round(evaluation.tc * 10) / 10,
        formula,
      });

      if (evaluation.tc > bestTc) {
        bestTc = evaluation.tc;
        bestLambdaRestart = evaluation.lambda;
        bestFormula = formula;
        bestZ = [...z];
        stagnation = 0;
      } else {
        stagnation++;
      }

      if (tcGap < targetTc * 0.05) break;

      const gradients = new Array(LATENT_DIM).fill(0);
      const delta = 0.01;

      const dims = Math.min(LATENT_DIM, 16 + Math.floor(step / 5) * 4);
      const dimIndices = Array.from({ length: LATENT_DIM }, (_, i) => i)
        .sort(() => Math.random() - 0.5)
        .slice(0, dims);

      for (const d of dimIndices) {
        const zPlus = [...z];
        zPlus[d] += delta;
        const genPlus = decode(zPlus);
        const fPlus = genomeVectorToFormula(genPlus);

        if (fPlus && isValidFormula(fPlus)) {
          const evalPlus = await evaluateFormulaTc(fPlus);
          gradients[d] = (evalPlus.tc - evaluation.tc) / delta;
        }
      }

      t++;
      const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
      const scaledLr = learningRate * (1 + step * 0.01);

      for (let d = 0; d < LATENT_DIM; d++) {
        m[d] = beta1 * m[d] + (1 - beta1) * gradients[d];
        v[d] = beta2 * v[d] + (1 - beta2) * gradients[d] * gradients[d];

        const mHat = m[d] / (1 - Math.pow(beta1, t));
        const vHat = v[d] / (1 - Math.pow(beta2, t));

        z[d] += scaledLr * mHat / (Math.sqrt(vHat) + eps);
        z[d] = Math.max(-3, Math.min(3, z[d]));
      }

      if (stagnation >= 8) {
        const jumpScale = 0.3 + Math.random() * 0.3;
        for (let d = 0; d < LATENT_DIM; d++) {
          z[d] = bestZ[d] + gaussRandom() * jumpScale;
        }
        stagnation = 0;
      }
    }

    if (bestTc > bestOverallTc) {
      bestOverallTc = bestTc;
      bestOverallLambda = bestLambdaRestart;
      bestOverallFormula = bestFormula;
      bestOverallZ = bestZ;
      bestTrajectory = trajectory;
      bestStartZ = startZ;
    }
  }

  const decodedFormulas: string[] = [];
  for (let i = 0; i < 5; i++) {
    const perturbedZ = bestOverallZ.map(v => v + gaussRandom() * 0.05);
    const genomeVec = decode(perturbedZ);
    const formula = genomeVectorToFormula(genomeVec);
    if (formula && isValidFormula(formula) && !decodedFormulas.includes(formula)) {
      decodedFormulas.push(formula);
    }
  }
  if (bestOverallFormula && !decodedFormulas.includes(bestOverallFormula)) {
    decodedFormulas.unshift(bestOverallFormula);
  }

  vaeStats.totalDesigns++;
  vaeStats.totalSteps += bestTrajectory.length;
  if (bestOverallTc > vaeStats.bestTc) {
    vaeStats.bestTc = bestOverallTc;
    vaeStats.bestFormula = bestOverallFormula;
  }
  vaeStats.recentDesigns.push({
    formula: bestOverallFormula,
    tc: bestOverallTc,
    steps: bestTrajectory.length,
    converged: bestOverallTc >= targetTc * 0.9,
  });
  if (vaeStats.recentDesigns.length > 30) vaeStats.recentDesigns.shift();

  return {
    startZ: bestStartZ,
    finalZ: bestOverallZ,
    optimizationSteps: bestTrajectory.length,
    decodedFormulas,
    bestFormula: bestOverallFormula,
    bestTc: Math.round(bestOverallTc * 10) / 10,
    bestLambda: Math.round(bestOverallLambda * 1000) / 1000,
    trajectory: bestTrajectory,
    converged: bestOverallTc >= targetTc * 0.9,
  };
}

export async function interpolateAndDecode(
  formulaA: string,
  formulaB: string,
  nPoints: number = 5
): Promise<{ alpha: number; formula: string; tc: number }[]> {
  const latentA = encodeToLatent(formulaA);
  const latentB = encodeToLatent(formulaB);
  const results: { alpha: number; formula: string; tc: number }[] = [];

  for (let i = 0; i <= nPoints; i++) {
    const alpha = i / nPoints;
    const interpolated = latentA.mu.map((a, j) => (1 - alpha) * a + alpha * (latentB.mu[j] ?? 0));
    const genomeVec = decode(interpolated);
    const formula = genomeVectorToFormula(genomeVec);

    if (formula && isValidFormula(formula)) {
      const eval_ = await evaluateFormulaTc(formula);
      results.push({
        alpha: Math.round(alpha * 100) / 100,
        formula,
        tc: Math.round(eval_.tc * 10) / 10,
      });
    }
  }

  return results;
}

export async function trainVAE(formulas: string[], epochs: number = 20): Promise<void> {
  const w = getVAEWeights();
  const lr = 0.0005;

  for (let epoch = 0; epoch < epochs; epoch++) {
    await new Promise<void>(r => setImmediate(r)); // yield each epoch so event loop breathes
    let totalLoss = 0;
    const batchSize = Math.min(16, formulas.length);

    for (let b = 0; b < batchSize; b++) {
      const idx = Math.floor(Math.random() * formulas.length);
      const formula = formulas[idx];

      try {
        const genome = encodeGenome(formula);
        const { mu, logvar } = encode(genome.vector);
        const z = reparameterize(mu, logvar);
        const reconstructed = decode(z);

        const recLoss = reconstructionLoss(genome.vector, reconstructed);
        const kl = klDivergence(mu, logvar);
        const loss = recLoss + vaeBeta * kl;
        totalLoss += loss;

        const h1Dec = leakyRelu(vecAdd(matVecMul(w.W_dec1, z), w.b_dec1));

        for (let i = 0; i < w.W_dec2.length; i++) {
          const error = (2.0 / genome.vector.length) * ((reconstructed[i] ?? 0) - (genome.vector[i] ?? 0));
          for (let j = 0; j < w.W_dec2[i].length; j++) {
            w.W_dec2[i][j] -= lr * error * (h1Dec[j] ?? 0);
          }
          w.b_dec2[i] -= lr * error;
        }

        for (let i = 0; i < w.W_mu.length; i++) {
          const muGrad = vaeBeta * (mu[i] ?? 0) / mu.length;
          for (let j = 0; j < w.W_mu[i].length; j++) {
            w.W_mu[i][j] -= lr * muGrad;
          }
          w.b_mu[i] -= lr * muGrad;
        }

      } catch {}
    }

    if (totalLoss / batchSize < 0.01) break;
  }

  vaeStats.lastTrainedAt = Date.now();
  vaeStats.trainingSamples = formulas.length;
}

const vaeStats = {
  totalDesigns: 0,
  totalSteps: 0,
  bestTc: 0,
  bestFormula: "",
  latentDim: LATENT_DIM,
  genomeDim: GENOME_DIM,
  lastTrainedAt: 0,
  trainingSamples: 0,
  recentDesigns: [] as { formula: string; tc: number; steps: number; converged: boolean }[],
  convergenceRate: 0,
};

export function getVAEStats() {
  if (vaeStats.recentDesigns.length > 0) {
    vaeStats.convergenceRate = vaeStats.recentDesigns.filter(d => d.converged).length / vaeStats.recentDesigns.length;
  }
  return { ...vaeStats, beta: vaeBeta };
}

export function exportMaterialsVAEWeights(): VAEWeights | null { return vaeWeights; }
export function importMaterialsVAEWeights(w: VAEWeights | null): void { if (w) vaeWeights = w; }