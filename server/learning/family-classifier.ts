/**
 * family-classifier.ts
 *
 * Hierarchical family classification + family-specific Tc regression for the GNN pipeline.
 *
 * Pipeline:
 *   1. GNN produces a pooled graph embedding (global material representation)
 *   2. classifyFamily()  → soft probability distribution over SC families
 *   3. familyTcPredict() → family-specific Tc formula (physics-derived for each class)
 *
 * Family taxonomy:
 *   cuprate          — Cu+O, d-wave, Presland-Tallon doping parabola
 *   hydride          — H-rich high-pressure, full Allen-Dynes (f1/f2 strong-coupling)
 *   iron_based       — Fe+pnictogen/chalcogen, s± pairing, modified Allen-Dynes
 *   conventional_bcs — Standard s-wave phonon-mediated (Nb, Pb, Hg, V3Si, ...)
 *   boride           — B-containing two-band (MgB2-type), σ-band Allen-Dynes
 *   kagome           — AV3Sb5-type, van Hove singularity enhanced
 *   heavy_fermion    — Ce/U/Yb heavy fermion, Kondo-scale empirical
 *   nickelate        — Infinite-layer NdNiO2-type, reduced cuprate analog
 *   unknown          — Fallback to raw GNN prediction
 */

import { parseFormulaCounts } from "./utils";
import { getElementData } from "./elemental-data";

// ─── Family taxonomy ──────────────────────────────────────────────────────────

export const FAMILY_LABELS = [
  "cuprate",
  "hydride",
  "iron_based",
  "conventional_bcs",
  "boride",
  "kagome",
  "heavy_fermion",
  "nickelate",
  "unknown",
] as const;

export type FamilyLabel = (typeof FAMILY_LABELS)[number];
export const N_FAMILIES = FAMILY_LABELS.length; // 9

// ─── Interfaces ───────────────────────────────────────────────────────────────

export interface FamilyClassification {
  /** Winning family label (argmax of blended probabilities). */
  label: FamilyLabel;
  /** Soft probability for each family (sums to 1). */
  probabilities: Record<FamilyLabel, number>;
  /** Max probability — measure of classification confidence. */
  confidence: number;
  /** Rule-based label independent of neural net (for debugging / training bootstrap). */
  ruleLabel: FamilyLabel;
}

export interface FamilyClassifierWeights {
  /** FC_HIDDEN × INPUT_DIM — first linear layer. */
  W_fc1: number[][];
  b_fc1: number[];
  /** N_FAMILIES × FC_HIDDEN — output layer. */
  W_fc2: number[][];
  b_fc2: number[];
  /** How strongly to weight rule-based logits vs. neural logits.
   *  Starts at 3.0 (rules dominate) and decays toward 0.5 as nSamples grows. */
  priorWeight: number;
  trainedAt: number;
  nSamples: number;
}

export interface FamilyTcParams {
  lambda: number;
  omegaLog: number;
  muStar: number;
  dosAtFermi: number;
  formula: string;
  /** Raw GNN Tc regression output — used as fallback / blend component. */
  gnnTc: number;
}

export interface FamilyTcResult {
  tc: number;
  /** Human-readable description of which formula was applied. */
  formulaUsed: string;
  /** Fraction of the result that came from the physics formula vs. GNN (0 = all GNN). */
  familyWeight: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

/** Hidden dimension of the learnable family classifier MLP. */
const FC_HIDDEN = 16;
/** Input dimension: 28-dim pooled embedding produced by extractGraphPooling() in multi-task-gnn.ts. */
const INPUT_DIM = 28;
const PRIOR_WEIGHT_INIT = 3.0;

// ─── Weight management ────────────────────────────────────────────────────────

let _classifierWeights: FamilyClassifierWeights | null = null;

function _seededRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return (s >>> 0) / 4294967296;
  };
}

function _initMatrix(rows: number, cols: number, rng: () => number): number[][] {
  const scale = Math.sqrt(2.0 / cols);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rng() - 0.5) * 2 * scale)
  );
}

export function getFamilyClassifierWeights(): FamilyClassifierWeights {
  if (_classifierWeights) return _classifierWeights;
  const rng = _seededRng(42137);
  _classifierWeights = {
    W_fc1: _initMatrix(FC_HIDDEN, INPUT_DIM, rng),
    b_fc1: new Array(FC_HIDDEN).fill(0),
    W_fc2: _initMatrix(N_FAMILIES, FC_HIDDEN, rng),
    b_fc2: new Array(N_FAMILIES).fill(0),
    priorWeight: PRIOR_WEIGHT_INIT,
    trainedAt: Date.now(),
    nSamples: 0,
  };
  return _classifierWeights;
}

export function setFamilyClassifierWeights(w: FamilyClassifierWeights): void {
  _classifierWeights = w;
}

// ─── Rule-based family prior ──────────────────────────────────────────────────

/**
 * Produces unnormalized logits for each family based purely on composition rules.
 * Used as a strong prior before the neural net accumulates training signal.
 */
function _getRuleLogits(formula: string): number[] {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const logits = new Array(N_FAMILIES).fill(0);

  const hasCu = counts["Cu"] != null;
  const hasO  = counts["O"]  != null;
  const hasH  = counts["H"]  != null;
  const hasFe = counts["Fe"] != null;
  const hasB  = counts["B"]  != null;
  const hasNi = counts["Ni"] != null;

  const PNICTOGEN    = new Set(["As", "P"]);
  const CHALCOGEN    = new Set(["Se", "Te", "S"]);
  const RARE_EARTHS  = new Set(["Ce", "Pr", "Yb", "U", "Np", "Pu", "Sm", "Tm", "Eu"]);
  const KAGOME_HOSTS = new Set(["Sb", "Sn", "Ge", "Bi"]);
  const KAGOME_METAL = new Set(["V", "Nb", "Cr", "Mn", "Fe", "Co"]);

  const hasPnictogen = elements.some(e => PNICTOGEN.has(e));
  const hasChalcogen = elements.some(e => CHALCOGEN.has(e));
  const hasRareEarth = elements.some(e => RARE_EARTHS.has(e));

  // idx 0: CUPRATE — Cu + O + charge-reservoir blocker
  const cuprateReservoir = new Set(["Y", "Ba", "La", "Bi", "Tl", "Hg", "Ca", "Sr"]);
  if (hasCu && hasO && elements.some(e => cuprateReservoir.has(e))) {
    logits[0] += 4.5;
  } else if (hasCu && hasO) {
    logits[0] += 2.0;
  }

  // idx 1: HYDRIDE — H present with substantial H/metal ratio
  if (hasH) {
    const metals = elements.filter(e => e !== "H");
    const metalCount = metals.reduce((s, e) => s + (counts[e] || 0), 0);
    const hRatio = metalCount > 0 ? (counts["H"] || 0) / metalCount : 0;
    if (hRatio >= 6)      logits[1] += 5.0;
    else if (hRatio >= 4) logits[1] += 4.0;
    else if (hRatio >= 2) logits[1] += 2.5;
    else if (hRatio >= 1) logits[1] += 1.2;
  }

  // idx 2: IRON_BASED — Fe + pnictogen or chalcogen
  if (hasFe && (hasPnictogen || hasChalcogen)) {
    logits[2] += 4.5;
  } else if (hasFe && hasO) {
    logits[2] += 1.5; // some FeSe/LaFeAsO systems include O
  }

  // idx 3: CONVENTIONAL_BCS — classic s-wave elemental/intermetallic SCs
  const conventionalSet = new Set([
    "Nb", "Pb", "Hg", "Sn", "Al", "In", "V", "Ta", "Mo",
    "Re", "Zr", "Hf", "Tc", "La", "Lu",
  ]);
  const nConv = elements.filter(e => conventionalSet.has(e)).length;
  if (!hasH && !hasCu && !(hasFe && (hasPnictogen || hasChalcogen)) && nConv >= 1) {
    logits[3] += 2.0 + nConv * 0.5;
  } else if (nConv >= 1) {
    logits[3] += 0.5;
  }

  // idx 4: BORIDE — B present at appreciable ratio
  if (hasB) {
    const bCount = counts["B"] || 0;
    const nonBMetal = elements.filter(e => e !== "B" && e !== "H")
      .reduce((s, e) => s + (counts[e] || 0), 0);
    const bRatio = nonBMetal > 0 ? bCount / nonBMetal : 0;
    if (bRatio >= 2)      logits[4] += 4.0;
    else if (bRatio >= 1) logits[4] += 2.5;
    else                  logits[4] += 0.8;
  }

  // idx 5: KAGOME — AV3Sb5, Mn3X, FeSn-type with kagome motif
  if (elements.some(e => KAGOME_METAL.has(e)) && elements.some(e => KAGOME_HOSTS.has(e))) {
    // Kagome motif: transition metal + anion in small-unit-cell compound
    const nEl = elements.length;
    if (nEl <= 4) logits[5] += 3.5;
    else          logits[5] += 1.5;
  }

  // idx 6: HEAVY_FERMION — rare earth/actinide + transition metal
  if (hasRareEarth) {
    const hasTransMetal = elements.some(e => {
      if (RARE_EARTHS.has(e)) return false;
      const d = getElementData(e);
      return d != null && d.atomicNumber >= 21 && d.atomicNumber <= 30;
    });
    logits[6] += hasTransMetal ? 4.0 : 2.0;
  }

  // idx 7: NICKELATE — Ni + O + rare-earth-like reservoir (infinite-layer / R-P)
  const nickelateReservoir = new Set(["La", "Nd", "Pr", "Sm", "Ca", "Sr", "Ba"]);
  if (hasNi && hasO && elements.some(e => nickelateReservoir.has(e))) {
    logits[7] += 4.0;
  } else if (hasNi && hasO) {
    logits[7] += 1.5;
  }

  // idx 8: UNKNOWN — always small positive to prevent zero probabilities
  logits[8] += 0.5;

  return logits;
}

/**
 * Rule-only family classification (no neural net, always deterministic).
 * Used for training label bootstrapping and debugging.
 */
export function ruleBasedFamily(formula: string): FamilyLabel {
  const logits = _getRuleLogits(formula);
  let maxIdx = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[maxIdx]) maxIdx = i;
  }
  return logits[maxIdx] >= 1.0 ? FAMILY_LABELS[maxIdx] : "unknown";
}

// ─── Neural classifier forward pass ──────────────────────────────────────────

function _matVecAdd(mat: number[][], vec: number[], bias: number[]): number[] {
  return mat.map((row, i) =>
    row.reduce((s, w, j) => s + w * (vec[j] ?? 0), 0) + (bias[i] ?? 0)
  );
}

function _softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(Math.max(-20, Math.min(20, l - max))));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map(e => e / sum);
}

// ─── Public classifier ────────────────────────────────────────────────────────

/**
 * Classify a material's superconductor family using a hybrid rule-prior + learned MLP.
 *
 * Rule logits dominate at first (priorWeight = 3.0); as the neural net trains via
 * trainFamilyClassifierStep(), priorWeight decays toward 0.5, letting the network
 * discover latent family distinctions beyond composition rules.
 *
 * @param formula  Chemical formula (e.g. "YBa2Cu3O7")
 * @param pooled   28-dim pooled GNN embedding from extractGraphPooling() in multi-task-gnn.ts
 * @param weights  Optional explicit weight object; defaults to global singleton
 */
export function classifyFamily(
  formula: string,
  pooled: number[],
  weights?: FamilyClassifierWeights
): FamilyClassification {
  const w = weights ?? getFamilyClassifierWeights();

  // Pad / trim pooled vector to INPUT_DIM
  const x = pooled.length >= INPUT_DIM
    ? pooled.slice(0, INPUT_DIM)
    : [...pooled, ...new Array(INPUT_DIM - pooled.length).fill(0)];

  // Neural forward: INPUT_DIM → FC_HIDDEN (ReLU) → N_FAMILIES
  const preH       = _matVecAdd(w.W_fc1, x, w.b_fc1);
  const h          = preH.map(v => Math.max(0, v));
  const neuralLogits = _matVecAdd(w.W_fc2, h, w.b_fc2);

  // Rule-based prior logits
  const ruleLogits = _getRuleLogits(formula);
  const ruleLabel  = (() => {
    let mi = 0;
    for (let i = 1; i < ruleLogits.length; i++) if (ruleLogits[i] > ruleLogits[mi]) mi = i;
    return ruleLogits[mi] >= 1.0 ? FAMILY_LABELS[mi] : ("unknown" as FamilyLabel);
  })();

  // Blend: neural + prior weighted logits → softmax
  const pw       = w.priorWeight ?? PRIOR_WEIGHT_INIT;
  const combined = neuralLogits.map((nl, i) => nl + pw * ruleLogits[i]);
  const probs    = _softmax(combined);

  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;

  const probMap = {} as Record<FamilyLabel, number>;
  for (let i = 0; i < N_FAMILIES; i++) {
    probMap[FAMILY_LABELS[i]] = Math.round(probs[i] * 1000) / 1000;
  }

  return {
    label:         FAMILY_LABELS[maxIdx],
    probabilities: probMap,
    confidence:    Math.round(probs[maxIdx] * 1000) / 1000,
    ruleLabel,
  };
}

// ─── Family-specific Tc physics formulas ─────────────────────────────────────

/**
 * Full Allen-Dynes formula with strong-coupling corrections (f1, f2).
 * Ref: Allen & Dynes, PRB 12, 905 (1975).
 * Valid domain: λ ∈ (0.1, 5.5), ω_log > 0.
 */
function _allenDynesFull(lambda: number, omegaLog: number, muStar: number): number {
  if (lambda <= 0.1 || omegaLog <= 0) return 0;
  const denom = lambda - muStar * (1 + 0.62 * lambda);
  if (denom <= 0.01) return 0;

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const lambda0   = Math.sqrt(1.82 * 1.04 * muStar / (1 + 6.3 * muStar));

  // Strong-coupling correction factors
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 1.5), 1 / 3);
  const f2 = lambda > lambda0
    ? 1 + Math.pow(lambda - lambda0, 2) / (Math.pow(lambda, 3) + Math.pow(lambda0, 3))
    : 1.0;

  const tc = (omegaLog / 1.2) * f1 * f2 * Math.exp(-1.04 * (1 + lambda) / denom);
  return Math.max(0, Math.min(500, tc));
}

/**
 * Estimate cuprate Tc_max and optimal hole doping p from formula composition.
 *
 * Uses Presland-Tallon universal doping parabola:
 *   Tc = Tc_max * (1 − 82.6 * (p − 0.16)²)
 * Ref: Presland et al., Physica C 176, 95 (1991).
 */
function _estimateCuprateParams(formula: string): { p: number; tcMax: number } {
  const counts = parseFormulaCounts(formula);

  // Tc_max by blocking-layer cation (experimental family records)
  const tcByBlocker: Record<string, number> = {
    Hg: 133, Tl: 128, Bi: 110, Y: 93, La: 38, Ba: 90, Ca: 80,
  };
  let tcMax = 60; // generic CuO2 default
  for (const [el, val] of Object.entries(tcByBlocker)) {
    if (counts[el] != null) { tcMax = val; break; }
  }

  // Rough CuO2 layer count adjustment (n=1: ~65%, n=2: 100%, n=3: 115% of Tc_max)
  const cuCount = counts["Cu"] || 1;
  const layerScale = cuCount === 1 ? 0.65 : cuCount === 2 ? 1.0 : cuCount >= 3 ? 1.15 : 1.0;
  tcMax = Math.min(165, Math.round(tcMax * layerScale));

  // Assume optimal doping (p = 0.16) without explicit doping information.
  // At optimal doping the parabola gives Tc = Tc_max.
  return { p: 0.16, tcMax };
}

function _cuprateTc(formula: string, gnnTc: number): number {
  const { p, tcMax } = _estimateCuprateParams(formula);
  const parabolaTc = tcMax * (1 - 82.6 * Math.pow(p - 0.16, 2));
  // 30% GNN weight captures doping hints embedded in the latent representation
  return Math.max(0, Math.round((0.70 * parabolaTc + 0.30 * gnnTc) * 10) / 10);
}

function _ironBasedTc(
  lambda: number, omegaLog: number, muStar: number, dosAtFermi: number, gnnTc: number
): number {
  // s± pairing: ~15% interband nesting enhancement to effective λ
  const lambdaEff = lambda * 1.15;
  const adTc = _allenDynesFull(lambdaEff, omegaLog, muStar);

  // Empirical DOS-weighted ceiling (max known: 56 K in SmFeAsO1-xFx)
  const empiricalScale = Math.min(1.0, dosAtFermi / 4.0) * 0.7 + 0.3;
  const empiricalTc    = 56 * empiricalScale;

  const blended = 0.5 * Math.min(adTc, 56) + 0.5 * empiricalTc;
  return Math.max(0, Math.min(56, Math.round((0.70 * blended + 0.30 * gnnTc) * 10) / 10));
}

function _borideTc(
  lambda: number, omegaLog: number, muStar: number, formula: string, gnnTc: number
): number {
  const counts    = parseFormulaCounts(formula);
  const total     = Math.max(1, Object.values(counts).reduce((s, n) => s + n, 0));
  const bFraction = (counts["B"] || 0) / total;

  // σ-band (dominant in MgB2-type) carries ~70% of total λ
  const lambdaSigma = 0.70 * lambda;
  // Boron E2g modes push ω_log higher; scale proportional to B fraction
  const omegaLogB   = omegaLog * (1 + 0.5 * bFraction);
  // Interband screening slightly reduces effective μ*
  const muStarB     = Math.max(0.06, muStar - 0.02);

  const adTc = _allenDynesFull(lambdaSigma, omegaLogB, muStarB);
  return Math.max(0, Math.min(60, Math.round((0.70 * adTc + 0.30 * gnnTc) * 10) / 10));
}

function _kagomeTc(lambda: number, omegaLog: number, muStar: number, gnnTc: number): number {
  // van Hove singularity at the M point enhances effective coupling by ~35%
  const lambdaVH = lambda * 1.35;
  const adTc     = _allenDynesFull(lambdaVH, omegaLog, muStar);
  // AV3Sb5 family Tc typically < 10 K; wider kagome class can reach ~30 K
  return Math.max(0, Math.min(30, Math.round((0.60 * adTc + 0.40 * gnnTc) * 10) / 10));
}

function _heavyFermionTc(dosAtFermi: number, lambda: number, gnnTc: number): number {
  // Kondo-mediated pairing; exact Kondo temperature requires f-electron DFT.
  // Use empirical DOS + λ scaling capped at the CeCoIn5 record (18.5 K at pressure).
  const empiricalMax = 18;
  const dosScale     = Math.min(1.0, dosAtFermi / 3.0);
  const lambdaScale  = Math.min(1.0, lambda / 0.5);
  const empiricalTc  = empiricalMax * dosScale * lambdaScale;
  return Math.max(0, Math.min(empiricalMax, Math.round((0.50 * empiricalTc + 0.50 * gnnTc) * 10) / 10));
}

function _nickelateTc(formula: string, gnnTc: number): number {
  const { p, tcMax } = _estimateCuprateParams(formula);
  // Infinite-layer nickelates have ~38% of the cuprate Tc_max for the same structure
  const nicTcMax    = tcMax * 0.38;
  const parabolaTc  = nicTcMax * (1 - 82.6 * Math.pow(p - 0.16, 2));
  return Math.max(0, Math.min(80, Math.round((0.60 * parabolaTc + 0.40 * gnnTc) * 10) / 10));
}

// ─── Public Tc dispatcher ─────────────────────────────────────────────────────

/**
 * Predict Tc using the physics formula appropriate for the classified family.
 *
 * Each family uses a different theoretical/empirical model:
 *   cuprate          → Presland-Tallon parabola   (d-wave, hole-doping)
 *   hydride          → Full Allen-Dynes (f1/f2)   (strong-coupling phonon)
 *   iron_based       → Modified Allen-Dynes + 56K cap (s± pairing)
 *   conventional_bcs → Full Allen-Dynes (f1/f2)   (s-wave phonon)
 *   boride           → Two-band σ-channel AD       (MgB2 two-gap)
 *   kagome           → AD + vHS factor 1.35        (flat-band enhancement)
 *   heavy_fermion    → Empirical DOS/λ scale       (Kondo-mediated)
 *   nickelate        → Reduced Presland-Tallon     (38% cuprate analog)
 *   unknown          → Raw GNN fallback
 */
export function familyTcPredict(family: FamilyLabel, params: FamilyTcParams): FamilyTcResult {
  const { lambda, omegaLog, muStar, dosAtFermi, formula, gnnTc } = params;

  // Clamp inputs to physically valid ranges before dispatch
  const λ    = Math.max(0, Math.min(5.5,  lambda));
  const ωlog = Math.max(10, Math.min(1500, omegaLog));
  const μ    = Math.max(0.05, Math.min(0.20, muStar));
  const dos  = Math.max(0, dosAtFermi);
  const tg   = Math.max(0, gnnTc);

  switch (family) {
    case "cuprate":
      return {
        tc: _cuprateTc(formula, tg),
        formulaUsed: "Presland-Tallon doping parabola (cuprate d-wave)",
        familyWeight: 0.70,
      };

    case "hydride": {
      const ad = _allenDynesFull(λ, ωlog, μ);
      return {
        tc: Math.max(0, Math.min(300, Math.round((0.75 * ad + 0.25 * tg) * 10) / 10)),
        formulaUsed: "Allen-Dynes full (f1/f2 strong-coupling, high-pressure hydride)",
        familyWeight: 0.75,
      };
    }

    case "iron_based":
      return {
        tc: _ironBasedTc(λ, ωlog, μ, dos, tg),
        formulaUsed: "Modified Allen-Dynes × 1.15λ + 56K empirical cap (iron-based s±)",
        familyWeight: 0.70,
      };

    case "conventional_bcs": {
      const ad = _allenDynesFull(λ, ωlog, μ);
      return {
        tc: Math.max(0, Math.min(300, Math.round((0.65 * ad + 0.35 * tg) * 10) / 10)),
        formulaUsed: "Allen-Dynes full (f1/f2, conventional BCS phonon-mediated)",
        familyWeight: 0.65,
      };
    }

    case "boride":
      return {
        tc: _borideTc(λ, ωlog, μ, formula, tg),
        formulaUsed: "Two-band σ-channel Allen-Dynes (MgB2-type boride, λσ = 0.7λ)",
        familyWeight: 0.70,
      };

    case "kagome":
      return {
        tc: _kagomeTc(λ, ωlog, μ, tg),
        formulaUsed: "Allen-Dynes + van Hove singularity factor ×1.35 (kagome flat-band)",
        familyWeight: 0.60,
      };

    case "heavy_fermion":
      return {
        tc: _heavyFermionTc(dos, λ, tg),
        formulaUsed: "Kondo-scale empirical blend (heavy fermion, cap 18 K)",
        familyWeight: 0.50,
      };

    case "nickelate":
      return {
        tc: _nickelateTc(formula, tg),
        formulaUsed: "Presland-Tallon analog at 38% cuprate scale (infinite-layer nickelate)",
        familyWeight: 0.60,
      };

    case "unknown":
    default:
      return {
        tc: Math.max(0, Math.round(tg * 10) / 10),
        formulaUsed: "GNN direct regression (no family matched)",
        familyWeight: 0.0,
      };
  }
}

// ─── Training helpers ─────────────────────────────────────────────────────────

/**
 * One stochastic gradient descent step on cross-entropy loss for the family classifier.
 *
 * Call this from the engine training loop once per training sample.
 * If no explicit family label is available, pass inferFamilyLabel(formula) as trueFamily
 * to bootstrap from composition rules until labeled data is available.
 *
 * @param formula    Chemical formula
 * @param trueFamily Ground-truth (or rule-inferred) family label
 * @param pooled     28-dim pooled GNN embedding for this sample
 * @param lr         Learning rate (default 0.001)
 * @returns          Cross-entropy loss for this step
 */
export function trainFamilyClassifierStep(
  formula: string,
  trueFamily: FamilyLabel,
  pooled: number[],
  lr: number = 0.001
): { loss: number } {
  const w = getFamilyClassifierWeights();

  const x = pooled.length >= INPUT_DIM
    ? pooled.slice(0, INPUT_DIM)
    : [...pooled, ...new Array(INPUT_DIM - pooled.length).fill(0)];

  // ── Forward pass ──────────────────────────────────────────────────────────
  const preH   = _matVecAdd(w.W_fc1, x, w.b_fc1);
  const h      = preH.map(v => Math.max(0, v));
  const logits = _matVecAdd(w.W_fc2, h, w.b_fc2);
  const probs  = _softmax(logits);

  const trueIdx = FAMILY_LABELS.indexOf(trueFamily);
  if (trueIdx < 0) return { loss: 0 };

  const loss = -Math.log(Math.max(1e-10, probs[trueIdx]));

  // ── Backward pass ─────────────────────────────────────────────────────────
  // ∂L/∂logits = probs − one_hot(trueIdx)
  const dLogits = probs.map((p, i) => p - (i === trueIdx ? 1 : 0));

  // Update W_fc2, b_fc2
  for (let i = 0; i < N_FAMILIES; i++) {
    w.b_fc2[i] -= lr * dLogits[i];
    for (let j = 0; j < FC_HIDDEN; j++) {
      w.W_fc2[i][j] -= lr * dLogits[i] * h[j];
    }
  }

  // ∂L/∂h (sum over output neurons)
  const dH = new Array(FC_HIDDEN).fill(0);
  for (let j = 0; j < FC_HIDDEN; j++) {
    for (let i = 0; i < N_FAMILIES; i++) {
      dH[j] += dLogits[i] * w.W_fc2[i][j];
    }
  }

  // ∂L/∂preH through ReLU gate
  const dPreH = dH.map((g, j) => (preH[j] > 0 ? g : 0));

  // Update W_fc1, b_fc1
  for (let i = 0; i < FC_HIDDEN; i++) {
    w.b_fc1[i] -= lr * dPreH[i];
    for (let j = 0; j < INPUT_DIM; j++) {
      w.W_fc1[i][j] -= lr * dPreH[i] * x[j];
    }
  }

  // Slowly decay prior weight as neural net accumulates training signal
  w.nSamples++;
  if (w.nSamples % 50 === 0 && w.priorWeight > 0.5) {
    w.priorWeight = Math.max(0.5, w.priorWeight * 0.97);
  }
  w.trainedAt = Date.now();

  return { loss };
}

/**
 * Infer a bootstrap training label from composition rules.
 * Use this when no explicit family label is available for a training sample.
 */
export function inferFamilyLabel(formula: string): FamilyLabel {
  return ruleBasedFamily(formula);
}

// ─── Serialization ────────────────────────────────────────────────────────────

export function serializeFamilyWeights(w: FamilyClassifierWeights): Record<string, unknown> {
  return {
    W_fc1: w.W_fc1, b_fc1: w.b_fc1,
    W_fc2: w.W_fc2, b_fc2: w.b_fc2,
    priorWeight: w.priorWeight,
    trainedAt: w.trainedAt,
    nSamples: w.nSamples,
  };
}

export function deserializeFamilyWeights(raw: Record<string, unknown>): FamilyClassifierWeights {
  const rng = _seededRng(42137);
  return {
    W_fc1:       (raw.W_fc1 as number[][] | null) ?? _initMatrix(FC_HIDDEN, INPUT_DIM, rng),
    b_fc1:       (raw.b_fc1 as number[] | null)   ?? new Array(FC_HIDDEN).fill(0),
    W_fc2:       (raw.W_fc2 as number[][] | null) ?? _initMatrix(N_FAMILIES, FC_HIDDEN, rng),
    b_fc2:       (raw.b_fc2 as number[] | null)   ?? new Array(N_FAMILIES).fill(0),
    priorWeight: typeof raw.priorWeight === "number" ? raw.priorWeight : PRIOR_WEIGHT_INIT,
    trainedAt:   typeof raw.trainedAt   === "number" ? raw.trainedAt   : Date.now(),
    nSamples:    typeof raw.nSamples    === "number" ? raw.nSamples    : 0,
  };
}
