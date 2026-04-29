/**
 * Adaptive Learning Store
 *
 * Tracks which volume multipliers and generators produce candidates that
 * survive each funnel stage. Uses this history to shift future budgets
 * toward successful regions of search space.
 *
 * Learns per material family + pressure bin (not per exact formula).
 * Families: rare_earth_hydride, alkaline_hydride, pnictide, chalcogenide, etc.
 * Pressure bins: 0-25, 25-75, 75-150, 150-250, 250+ GPa
 */

import * as fs from "fs";
import * as path from "path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MaterialFamily =
  | "rare_earth_hydride"
  | "alkaline_earth_hydride"
  | "alkali_hydride"
  | "transition_metal_hydride"
  | "binary_hydride"
  | "ternary_hydride"
  | "pnictide"
  | "chalcogenide"
  | "cuprate"
  | "intermetallic"
  | "oxide"
  | "other";

export type PressureBin = "0-25" | "25-75" | "75-150" | "150-250" | "250+";

export interface VolumeSuccessRecord {
  /** Volume multiplier → success rate (0-1). */
  multiplierSuccess: Record<string, number>;
  /** Total trials per multiplier. */
  multiplierTrials: Record<string, number>;
}

export interface GeneratorSuccessRecord {
  /** Generator name → success rate (0-1). */
  generatorSuccess: Record<string, number>;
  /** Total trials per generator. */
  generatorTrials: Record<string, number>;
}

export interface FamilyLearningRecord {
  family: MaterialFamily;
  pressureBin: PressureBin;
  volumeData: VolumeSuccessRecord;
  generatorData: GeneratorSuccessRecord;
  lastUpdated: number;
}

interface LearningStore {
  records: Record<string, FamilyLearningRecord>; // key: "family:pressureBin"
}

// ---------------------------------------------------------------------------
// Element classification
// ---------------------------------------------------------------------------

const RARE_EARTHS = new Set(["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"]);
const ALKALINE_EARTH = new Set(["Ca", "Sr", "Ba", "Mg"]);
const ALKALI = new Set(["Li", "Na", "K", "Rb", "Cs"]);
const TRANSITION_METALS = new Set(["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]);
const PNICTIDES = new Set(["P", "As", "Sb", "Bi"]);
const CHALCOGENS = new Set(["S", "Se", "Te"]);

export function classifyFamily(elements: string[]): MaterialFamily {
  const hasH = elements.includes("H");
  const hasO = elements.includes("O");
  const hasRE = elements.some(e => RARE_EARTHS.has(e));
  const hasAE = elements.some(e => ALKALINE_EARTH.has(e));
  const hasAlkali = elements.some(e => ALKALI.has(e));
  const hasTM = elements.some(e => TRANSITION_METALS.has(e));
  const hasPnictide = elements.some(e => PNICTIDES.has(e));
  const hasChalcogen = elements.some(e => CHALCOGENS.has(e));
  const hasCu = elements.includes("Cu");

  if (hasH && hasRE) return "rare_earth_hydride";
  if (hasH && hasAE) return "alkaline_earth_hydride";
  if (hasH && hasAlkali) return "alkali_hydride";
  if (hasH && hasTM) return "transition_metal_hydride";
  if (hasH && elements.length === 2) return "binary_hydride";
  if (hasH) return "ternary_hydride";
  if (hasCu && hasO) return "cuprate";
  if (hasPnictide && hasTM) return "pnictide";
  if (hasChalcogen) return "chalcogenide";
  if (hasO) return "oxide";
  if (elements.every(e => TRANSITION_METALS.has(e) || RARE_EARTHS.has(e) || ALKALINE_EARTH.has(e) || ALKALI.has(e))) return "intermetallic";
  return "other";
}

export function classifyPressureBin(pressureGPa: number): PressureBin {
  if (pressureGPa < 25) return "0-25";
  if (pressureGPa < 75) return "25-75";
  if (pressureGPa < 150) return "75-150";
  if (pressureGPa < 250) return "150-250";
  return "250+";
}

// ---------------------------------------------------------------------------
// Persistent store (file-based, JSON)
// ---------------------------------------------------------------------------

const STORE_PATH = path.join(process.cwd(), "server", "csp", "learning-store.json");

let _store: LearningStore | null = null;

function loadStore(): LearningStore {
  if (_store) return _store;
  try {
    if (fs.existsSync(STORE_PATH)) {
      const raw = fs.readFileSync(STORE_PATH, "utf-8");
      _store = JSON.parse(raw);
      return _store!;
    }
  } catch {}
  _store = { records: {} };
  return _store;
}

function saveStore(): void {
  try {
    const store = loadStore();
    fs.writeFileSync(STORE_PATH, JSON.stringify(store, null, 2));
  } catch {}
}

function getKey(family: MaterialFamily, pressureBin: PressureBin): string {
  return `${family}:${pressureBin}`;
}

function getOrCreateRecord(family: MaterialFamily, pressureBin: PressureBin): FamilyLearningRecord {
  const store = loadStore();
  const key = getKey(family, pressureBin);
  if (!store.records[key]) {
    store.records[key] = {
      family,
      pressureBin,
      volumeData: { multiplierSuccess: {}, multiplierTrials: {} },
      generatorData: { generatorSuccess: {}, generatorTrials: {} },
      lastUpdated: Date.now(),
    };
  }
  return store.records[key];
}

// ---------------------------------------------------------------------------
// Recording outcomes
// ---------------------------------------------------------------------------

/**
 * Record that a candidate at a given volume multiplier survived (or didn't)
 * through a funnel stage.
 */
export function recordVolumeOutcome(
  elements: string[],
  pressureGPa: number,
  volumeMultiplier: number,
  survived: boolean,
): void {
  const family = classifyFamily(elements);
  const pBin = classifyPressureBin(pressureGPa);
  const record = getOrCreateRecord(family, pBin);
  const key = volumeMultiplier.toFixed(2);

  record.volumeData.multiplierTrials[key] = (record.volumeData.multiplierTrials[key] ?? 0) + 1;
  if (survived) {
    record.volumeData.multiplierSuccess[key] = (record.volumeData.multiplierSuccess[key] ?? 0) + 1;
  }
  record.lastUpdated = Date.now();
  saveStore();
}

/**
 * Record that a candidate from a given generator survived (or didn't).
 */
export function recordGeneratorOutcome(
  elements: string[],
  pressureGPa: number,
  generatorName: string,
  survived: boolean,
): void {
  const family = classifyFamily(elements);
  const pBin = classifyPressureBin(pressureGPa);
  const record = getOrCreateRecord(family, pBin);

  record.generatorData.generatorTrials[generatorName] = (record.generatorData.generatorTrials[generatorName] ?? 0) + 1;
  if (survived) {
    record.generatorData.generatorSuccess[generatorName] = (record.generatorData.generatorSuccess[generatorName] ?? 0) + 1;
  }
  record.lastUpdated = Date.now();
  saveStore();
}

// ---------------------------------------------------------------------------
// Querying learned priors
// ---------------------------------------------------------------------------

/**
 * Get learned volume multiplier weights for a material family + pressure bin.
 * Returns a map of multiplier → weight (0-1), biased toward successful volumes.
 * Falls back to uniform weights if no data available.
 */
export function getLearnedVolumeWeights(
  elements: string[],
  pressureGPa: number,
  defaultMultipliers: number[],
): Record<string, number> {
  const family = classifyFamily(elements);
  const pBin = classifyPressureBin(pressureGPa);
  const store = loadStore();
  const key = getKey(family, pBin);
  const record = store.records[key];

  const weights: Record<string, number> = {};

  if (!record || Object.keys(record.volumeData.multiplierTrials).length === 0) {
    // No data — uniform weights
    for (const m of defaultMultipliers) {
      weights[m.toFixed(2)] = 1.0 / defaultMultipliers.length;
    }
    return weights;
  }

  // Compute success rates with Bayesian smoothing (prior = 0.5, pseudocount = 2)
  const PRIOR = 0.5;
  const PSEUDO = 2;
  let totalWeight = 0;

  for (const m of defaultMultipliers) {
    const mKey = m.toFixed(2);
    const trials = record.volumeData.multiplierTrials[mKey] ?? 0;
    const successes = record.volumeData.multiplierSuccess[mKey] ?? 0;
    const rate = (successes + PRIOR * PSEUDO) / (trials + PSEUDO);
    weights[mKey] = rate;
    totalWeight += rate;
  }

  // Normalize
  if (totalWeight > 0) {
    for (const k of Object.keys(weights)) {
      weights[k] /= totalWeight;
    }
  }

  return weights;
}

/**
 * Get learned generator budget weights for a material family + pressure bin.
 * Returns a map of generator → weight (0-1), biased toward successful generators.
 */
export function getLearnedGeneratorWeights(
  elements: string[],
  pressureGPa: number,
): Record<string, number> {
  const family = classifyFamily(elements);
  const pBin = classifyPressureBin(pressureGPa);
  const store = loadStore();
  const key = getKey(family, pBin);
  const record = store.records[key];

  const BASE_WEIGHTS: Record<string, number> = {
    "AIRSS-buildcell": 0.40,
    "PyXtal-random": 0.25,
    "VCA-interpolated": 0.10,
    "TemplateVCA": 0.10,
    "prototype": 0.10,
    "mutant": 0.05,
  };

  if (!record || Object.keys(record.generatorData.generatorTrials).length === 0) {
    return BASE_WEIGHTS;
  }

  // Blend base weights with learned success rates
  const PRIOR = 0.5;
  const PSEUDO = 5;
  const LEARN_RATE = 0.4; // how much to shift from base toward learned

  const learned: Record<string, number> = {};
  let totalWeight = 0;

  for (const [gen, baseW] of Object.entries(BASE_WEIGHTS)) {
    const trials = record.generatorData.generatorTrials[gen] ?? 0;
    const successes = record.generatorData.generatorSuccess[gen] ?? 0;
    const learnedRate = trials > 0 ? (successes + PRIOR * PSEUDO) / (trials + PSEUDO) : PRIOR;
    const blended = baseW * (1 - LEARN_RATE) + learnedRate * LEARN_RATE;
    learned[gen] = blended;
    totalWeight += blended;
  }

  // Normalize
  if (totalWeight > 0) {
    for (const k of Object.keys(learned)) {
      learned[k] /= totalWeight;
    }
  }

  return learned;
}

/**
 * Log the current learning state for a family.
 */
export function logLearningState(elements: string[], pressureGPa: number): void {
  const family = classifyFamily(elements);
  const pBin = classifyPressureBin(pressureGPa);
  const store = loadStore();
  const key = getKey(family, pBin);
  const record = store.records[key];

  if (!record) {
    console.log(`[CSP-Learn] ${family}/${pBin}: no learning data yet`);
    return;
  }

  const totalVTrials = Object.values(record.volumeData.multiplierTrials).reduce((s, v) => s + v, 0);
  const totalGTrials = Object.values(record.generatorData.generatorTrials).reduce((s, v) => s + v, 0);

  if (totalVTrials > 0) {
    const vSummary = Object.entries(record.volumeData.multiplierTrials)
      .map(([k, trials]) => {
        const succ = record.volumeData.multiplierSuccess[k] ?? 0;
        return `${k}=${succ}/${trials}(${(succ / trials * 100).toFixed(0)}%)`;
      }).join(", ");
    console.log(`[CSP-Learn] ${family}/${pBin} volumes: ${vSummary}`);
  }

  if (totalGTrials > 0) {
    const gSummary = Object.entries(record.generatorData.generatorTrials)
      .map(([k, trials]) => {
        const succ = record.generatorData.generatorSuccess[k] ?? 0;
        return `${k}=${succ}/${trials}(${(succ / trials * 100).toFixed(0)}%)`;
      }).join(", ");
    console.log(`[CSP-Learn] ${family}/${pBin} generators: ${gSummary}`);
  }
}
