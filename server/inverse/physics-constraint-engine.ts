import { ELEMENTAL_DATA } from "../learning/elemental-data";
import { parseFormulaElements } from "../learning/physics-engine";

const OXIDATION_STATES: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  Sc: [3], Y: [3], La: [3],
  Ti: [2, 3, 4], Zr: [4], Hf: [4],
  V: [2, 3, 4, 5], Nb: [3, 5], Ta: [5],
  Cr: [2, 3, 6], Mo: [4, 6], W: [4, 6],
  Mn: [2, 3, 4, 7], Re: [4, 7],
  Fe: [2, 3], Ru: [3, 4], Os: [4],
  Co: [2, 3], Rh: [3], Ir: [3, 4],
  Ni: [2, 3], Pd: [2, 4], Pt: [2, 4],
  Cu: [1, 2], Ag: [1], Au: [1, 3],
  Zn: [2], Cd: [2],
  B: [3], Al: [3], Ga: [3], In: [3], Tl: [1, 3],
  C: [4, -4], Si: [4, -4], Ge: [4], Sn: [2, 4], Pb: [2, 4],
  N: [-3, 3, 5], P: [-3, 3, 5], As: [-3, 3, 5], Sb: [3, 5], Bi: [3, 5],
  O: [-2], S: [-2, 4, 6], Se: [-2, 4, 6], Te: [-2, 4, 6],
  F: [-1], Cl: [-1, 1, 3, 5, 7], Br: [-1, 1, 3, 5], I: [-1, 1, 3, 5, 7],
  Ce: [3, 4], Pr: [3, 4], Nd: [3], Sm: [2, 3], Eu: [2, 3], Gd: [3],
  Tb: [3, 4], Dy: [3], Ho: [3], Er: [3], Tm: [3], Yb: [2, 3], Lu: [3],
  Th: [4], U: [3, 4, 5, 6],
};

const COORDINATION_LIMITS: Record<string, [number, number]> = {
  H: [1, 2], Li: [4, 8], Na: [4, 8], K: [6, 12], Rb: [6, 12], Cs: [8, 12],
  Be: [4, 4], Mg: [4, 6], Ca: [6, 8], Sr: [6, 12], Ba: [6, 12],
  Sc: [6, 8], Y: [6, 9], La: [8, 12],
  Ti: [4, 6], Zr: [6, 8], Hf: [6, 8],
  V: [4, 6], Nb: [4, 8], Ta: [6, 8],
  Cr: [4, 6], Mo: [4, 8], W: [4, 8],
  Mn: [4, 6], Re: [6, 8],
  Fe: [4, 6], Ru: [6, 6], Os: [6, 6],
  Co: [4, 6], Rh: [6, 6], Ir: [6, 6],
  Ni: [4, 6], Pd: [4, 6], Pt: [4, 6],
  Cu: [4, 6], Ag: [2, 6], Au: [2, 6],
  Zn: [4, 6], Cd: [4, 6],
  B: [3, 4], Al: [4, 6], Ga: [4, 6], In: [4, 6],
  C: [3, 4], Si: [4, 6], Ge: [4, 6], Sn: [4, 6], Pb: [4, 8],
  N: [2, 4], P: [4, 6], As: [4, 6], Sb: [4, 6], Bi: [4, 6],
  O: [2, 4], S: [2, 6], Se: [2, 6], Te: [2, 6],
  F: [1, 2], Cl: [1, 4], Br: [1, 4], I: [1, 4],
};

const NOBLE_GASES = new Set(["He", "Ne", "Ar", "Kr", "Xe", "Rn"]);

export interface ConstraintViolation {
  type: "charge_neutrality" | "radius_incompatibility" | "coordination_violation" |
        "noble_gas" | "electron_count" | "stoichiometry_excess" | "bond_instability";
  severity: number;
  detail: string;
  repairSuggestion?: string;
}

export interface ConstraintResult {
  formula: string;
  isValid: boolean;
  violations: ConstraintViolation[];
  totalPenalty: number;
  chargeImbalance: number;
  radiusScore: number;
  coordinationScore: number;
  bondStabilityScore: number;
  electronCountScore: number;
  repairedFormula: string | null;
}

export interface ConstraintStats {
  totalChecked: number;
  totalValid: number;
  totalRepaired: number;
  totalRejected: number;
  violationCounts: Record<string, number>;
  avgPenalty: number;
  constraintWeights: Record<string, number>;
  repairSuccessRate: number;
  topRepairPatterns: { from: string; to: string; count: number }[];
}

function parseCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") return {};
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    const el = m[1];
    const num = m[2] ? parseFloat(m[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function countsToFormula(counts: Record<string, number>): string {
  const entries = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort(([a], [b]) => {
      const ed = ELEMENTAL_DATA;
      const aEN = ed[a]?.paulingElectronegativity ?? 2.0;
      const bEN = ed[b]?.paulingElectronegativity ?? 2.0;
      return aEN - bEN;
    });
  return entries.map(([el, n]) => n === 1 ? el : `${el}${n}`).join("");
}

function getOxidationStates(el: string): number[] {
  return OXIDATION_STATES[el] || [0];
}

function getRadius(el: string): number {
  return ELEMENTAL_DATA[el]?.atomicRadius ?? 130;
}

function getEN(el: string): number {
  return ELEMENTAL_DATA[el]?.paulingElectronegativity ?? 2.0;
}

function checkChargeNeutrality(counts: Record<string, number>): {
  imbalance: number;
  bestAssignment: Record<string, number>;
  achievable: boolean;
} {
  const elements = Object.keys(counts);
  if (elements.length === 0) return { imbalance: 0, bestAssignment: {}, achievable: true };

  const metalloids = new Set(["B", "Si", "Ge", "As", "Sb", "Te"]);
  const metalOrMetalloid = (el: string): boolean => {
    if (el === "H") return true;
    if (metalloids.has(el)) return true;
    const ed = ELEMENTAL_DATA[el];
    if (!ed) return false;
    const en = ed.paulingElectronegativity ?? 2.0;
    if (en < 2.0) return true;
    const an = ed.atomicNumber;
    if ((an >= 21 && an <= 32) || (an >= 39 && an <= 52) || (an >= 72 && an <= 84)) return true;
    if ((an >= 3 && an <= 4) || (an >= 11 && an <= 13) || (an >= 19 && an <= 20) ||
        (an >= 37 && an <= 38) || (an >= 55 && an <= 57) || (an >= 87 && an <= 89)) return true;
    if (an >= 57 && an <= 71) return true;
    return false;
  };

  const isMetallic = elements.every(metalOrMetalloid);

  const hasHighENAnion = elements.some(el => {
    if (el === "H") return false;
    const en = getEN(el);
    return en >= 3.0 && !metalloids.has(el);
  });

  if (isMetallic && !hasHighENAnion) {
    const assignment: Record<string, number> = {};
    elements.forEach(el => assignment[el] = 0);
    return { imbalance: 0, bestAssignment: assignment, achievable: true };
  }

  const oxStates = elements.map(el => getOxidationStates(el));
  let bestImbalance = Infinity;
  let bestAssignment: Record<string, number> = {};

  function search(idx: number, assignment: Record<string, number>, currentSum: number) {
    if (idx === elements.length) {
      const imb = Math.abs(currentSum);
      if (imb < bestImbalance) {
        bestImbalance = imb;
        bestAssignment = { ...assignment };
      }
      return;
    }
    const el = elements[idx];
    const count = counts[el];
    for (const ox of oxStates[idx]) {
      assignment[el] = ox;
      search(idx + 1, assignment, currentSum + ox * count);
    }
  }

  if (elements.length <= 6) {
    search(0, {}, 0);
  } else {
    const anions = elements.filter(el => getEN(el) > 2.5);
    const cations = elements.filter(el => getEN(el) <= 2.5);

    let totalAnionCharge = 0;
    const assignment: Record<string, number> = {};
    for (const el of anions) {
      const states = getOxidationStates(el);
      const negState = states.find(s => s < 0) ?? states[0];
      assignment[el] = negState;
      totalAnionCharge += negState * counts[el];
    }

    let totalCationCharge = 0;
    for (const el of cations) {
      const states = getOxidationStates(el).filter(s => s > 0);
      if (states.length === 0) {
        assignment[el] = 0;
        continue;
      }
      const targetPerAtom = Math.abs(totalAnionCharge) / cations.reduce((s, c) => s + counts[c], 0);
      const best = states.reduce((a, b) => Math.abs(a - targetPerAtom) < Math.abs(b - targetPerAtom) ? a : b);
      assignment[el] = best;
      totalCationCharge += best * counts[el];
    }

    bestImbalance = Math.abs(totalAnionCharge + totalCationCharge);
    bestAssignment = assignment;
  }

  return {
    imbalance: bestImbalance,
    bestAssignment,
    achievable: bestImbalance === 0,
  };
}

function checkRadiusCompatibility(counts: Record<string, number>): {
  score: number;
  violations: string[];
} {
  const elements = Object.keys(counts);
  if (elements.length < 2) return { score: 1.0, violations: [] };

  const violations: string[] = [];
  let totalPairs = 0;
  let compatiblePairs = 0;

  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      totalPairs++;
      const elA = elements[i];
      const elB = elements[j];

      if (elA === "H" || elB === "H") {
        compatiblePairs++;
        continue;
      }

      const rA = getRadius(elA);
      const rB = getRadius(elB);
      const ratio = Math.max(rA, rB) / Math.max(Math.min(rA, rB), 1);

      if (ratio > 4.0) {
        violations.push(`${elA}/${elB} radius ratio ${ratio.toFixed(1)} exceeds 4.0`);
      } else {
        compatiblePairs++;
      }
    }
  }

  return {
    score: totalPairs > 0 ? compatiblePairs / totalPairs : 1.0,
    violations,
  };
}

function checkCoordination(counts: Record<string, number>): {
  score: number;
  violations: string[];
} {
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms === 0) return { score: 0, violations: ["empty formula"] };

  const violations: string[] = [];
  let validElements = 0;
  let total = 0;

  for (const el of elements) {
    total++;
    const limits = COORDINATION_LIMITS[el];
    if (!limits) { validElements++; continue; }

    if (el === "H") { validElements++; continue; }

    const elCount = counts[el];
    const neighborCount = totalAtoms - elCount;
    const effectiveCoord = neighborCount > 0 ? Math.min(neighborCount, limits[1] + 2) : 0;

    if (effectiveCoord >= limits[0] * 0.5) {
      validElements++;
    } else if (elements.length >= 3) {
      validElements++;
    } else {
      violations.push(`${el} has insufficient neighbors for coordination (need ${limits[0]}, available ${effectiveCoord})`);
    }
  }

  return {
    score: total > 0 ? validElements / total : 1.0,
    violations,
  };
}

function checkBondStability(counts: Record<string, number>): {
  score: number;
  violations: string[];
} {
  const elements = Object.keys(counts);
  if (elements.length < 2) return { score: 0.5, violations: [] };

  const violations: string[] = [];
  const ENs = elements.map(el => getEN(el));
  const maxEN = Math.max(...ENs);
  const minEN = Math.min(...ENs);
  const enSpread = maxEN - minEN;

  let score = 1.0;

  if (enSpread > 3.0) {
    score -= 0.3;
    violations.push(`Extreme electronegativity spread ${enSpread.toFixed(2)} > 3.0`);
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  for (const el of elements) {
    const fraction = counts[el] / totalAtoms;
    if (fraction > 0.85 && elements.length > 1) {
      score -= 0.2;
      violations.push(`${el} dominates composition at ${(fraction * 100).toFixed(0)}%`);
    }
  }

  if (totalAtoms > 30) {
    score -= 0.3;
    violations.push(`Total atom count ${totalAtoms} is unreasonably large`);
  }

  const hasAnyMetal = elements.some(el => {
    if (el === "H") return false;
    const en = getEN(el);
    return en <= 2.0;
  });
  if (!hasAnyMetal && elements.length > 1 && !elements.includes("H")) {
    score -= 0.15;
    violations.push("No metallic elements - limited metallic bonding for SC");
  }

  return { score: Math.max(0, score), violations };
}

function checkElectronCount(counts: Record<string, number>): {
  score: number;
  violations: string[];
} {
  const elements = Object.keys(counts);
  if (elements.length === 0) return { score: 0, violations: ["no elements"] };

  let totalValence = 0;
  for (const el of elements) {
    const ed = ELEMENTAL_DATA[el];
    if (ed) totalValence += ed.valenceElectrons * counts[el];
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const valencePerAtom = totalValence / Math.max(totalAtoms, 1);
  const violations: string[] = [];
  let score = 1.0;

  if (valencePerAtom < 1.0) {
    score -= 0.4;
    violations.push(`Very low valence electron density: ${valencePerAtom.toFixed(2)} e/atom`);
  }

  if (totalValence % 1 !== 0) {
    score -= 0.1;
    violations.push("Fractional total valence electron count");
  }

  const hasTransitionMetal = elements.some(el => {
    const an = ELEMENTAL_DATA[el]?.atomicNumber ?? 0;
    return (an >= 21 && an <= 30) || (an >= 39 && an <= 48) || (an >= 72 && an <= 80);
  });

  if (!hasTransitionMetal && valencePerAtom < 2.0) {
    score -= 0.2;
    violations.push("No transition metals and low valence density - poor for SC");
  }

  return { score: Math.max(0, score), violations };
}

let constraintWeights: Record<string, number> = {
  charge_neutrality: 1.0,
  radius_incompatibility: 0.6,
  coordination_violation: 0.5,
  bond_instability: 0.7,
  electron_count: 0.4,
  noble_gas: 1.0,
  stoichiometry_excess: 0.8,
};

let totalChecked = 0;
let totalValid = 0;
let totalRepaired = 0;
let totalRejected = 0;
let violationCounts: Record<string, number> = {};
let penaltySum = 0;
let repairAttempts = 0;
let repairSuccesses = 0;
let repairPatterns: Map<string, { from: string; to: string; count: number }> = new Map();
let constraintRewards: Record<string, { totalReward: number; count: number }> = {};

let inRepairCheck = false;

export function checkPhysicsConstraints(formula: string): ConstraintResult {
  totalChecked++;
  const counts = parseCounts(formula);
  const elements = Object.keys(counts);
  const violations: ConstraintViolation[] = [];

  for (const el of elements) {
    if (NOBLE_GASES.has(el)) {
      violations.push({
        type: "noble_gas",
        severity: 1.0,
        detail: `Noble gas ${el} cannot form stable compounds`,
      });
    }
    if (!ELEMENTAL_DATA[el]) {
      violations.push({
        type: "bond_instability",
        severity: 0.3,
        detail: `Unknown element ${el}`,
      });
    }
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms > 25) {
    violations.push({
      type: "stoichiometry_excess",
      severity: 0.7,
      detail: `Excessive total atoms: ${totalAtoms}`,
      repairSuggestion: "Reduce stoichiometric coefficients",
    });
  }

  for (const [el, n] of Object.entries(counts)) {
    if (n > 12) {
      violations.push({
        type: "stoichiometry_excess",
        severity: 0.6,
        detail: `${el}${n} exceeds stoichiometry limit (max 12)`,
        repairSuggestion: `Reduce ${el} count`,
      });
    }
  }

  const chargeResult = checkChargeNeutrality(counts);
  if (chargeResult.imbalance > 0) {
    violations.push({
      type: "charge_neutrality",
      severity: Math.min(1.0, chargeResult.imbalance / 6),
      detail: `Charge imbalance: ${chargeResult.imbalance} (states: ${Object.entries(chargeResult.bestAssignment).map(([e, s]) => `${e}${s > 0 ? "+" : ""}${s}`).join(", ")})`,
      repairSuggestion: chargeResult.imbalance <= 4 ? "Adjust stoichiometry for charge balance" : undefined,
    });
  }

  const radiusResult = checkRadiusCompatibility(counts);
  for (const v of radiusResult.violations) {
    violations.push({
      type: "radius_incompatibility",
      severity: 0.5,
      detail: v,
    });
  }

  const coordResult = checkCoordination(counts);
  for (const v of coordResult.violations) {
    violations.push({
      type: "coordination_violation",
      severity: 0.4,
      detail: v,
    });
  }

  const bondResult = checkBondStability(counts);
  for (const v of bondResult.violations) {
    violations.push({
      type: "bond_instability",
      severity: 0.5,
      detail: v,
    });
  }

  const electronResult = checkElectronCount(counts);
  for (const v of electronResult.violations) {
    violations.push({
      type: "electron_count",
      severity: 0.3,
      detail: v,
    });
  }

  let totalPenalty = 0;
  for (const v of violations) {
    const w = constraintWeights[v.type] ?? 0.5;
    totalPenalty += v.severity * w;
    violationCounts[v.type] = (violationCounts[v.type] || 0) + 1;
  }

  const isValid = totalPenalty < 0.5;
  if (isValid) totalValid++;
  else totalRejected++;

  penaltySum += totalPenalty;

  let repairedFormula: string | null = null;
  if (!isValid && totalPenalty < 2.0 && !inRepairCheck) {
    repairedFormula = repairFormula(formula, counts, violations, chargeResult);
    if (repairedFormula && repairedFormula !== formula) {
      totalRepaired++;
    }
  }

  return {
    formula,
    isValid,
    violations,
    totalPenalty,
    chargeImbalance: chargeResult.imbalance,
    radiusScore: radiusResult.score,
    coordinationScore: coordResult.score,
    bondStabilityScore: bondResult.score,
    electronCountScore: electronResult.score,
    repairedFormula,
  };
}

function repairFormula(
  formula: string,
  counts: Record<string, number>,
  violations: ConstraintViolation[],
  chargeResult: ReturnType<typeof checkChargeNeutrality>,
): string | null {
  repairAttempts++;
  const repaired = { ...counts };

  for (const el of Object.keys(repaired)) {
    if (NOBLE_GASES.has(el)) {
      delete repaired[el];
    }
  }

  for (const [el, n] of Object.entries(repaired)) {
    if (n > 12) {
      repaired[el] = Math.min(n, 8);
    }
  }

  if (chargeResult.imbalance > 0 && chargeResult.imbalance <= 6) {
    repairChargeBalance(repaired, chargeResult);
  }

  const totalAtoms = Object.values(repaired).reduce((s, n) => s + n, 0);
  if (totalAtoms > 25) {
    const scale = 20 / totalAtoms;
    for (const el of Object.keys(repaired)) {
      repaired[el] = Math.max(1, Math.round(repaired[el] * scale));
    }
  }

  const remaining = Object.keys(repaired).filter(el => repaired[el] > 0);
  if (remaining.length < 2) return null;

  const result = countsToFormula(repaired);
  if (result === formula) return null;

  inRepairCheck = true;
  const recheck = checkPhysicsConstraints(result);
  totalChecked--;

  if (recheck.totalPenalty < 0.5) {
    repairSuccesses++;
    const key = `${formula}->${result}`;
    const existing = repairPatterns.get(key);
    if (existing) existing.count++;
    else repairPatterns.set(key, { from: formula, to: result, count: 1 });
    inRepairCheck = false;
    return result;
  }

  const origRecheck = checkPhysicsConstraints(formula);
  totalChecked--;
  inRepairCheck = false;

  if (recheck.totalPenalty < origRecheck.totalPenalty * 0.7) {
    repairSuccesses++;
    return result;
  }

  return null;
}

function repairChargeBalance(counts: Record<string, number>, chargeResult: ReturnType<typeof checkChargeNeutrality>): void {
  const assignment = chargeResult.bestAssignment;
  const elements = Object.keys(counts);

  const anions = elements.filter(el => (assignment[el] ?? 0) < 0);
  const cations = elements.filter(el => (assignment[el] ?? 0) > 0);

  if (anions.length === 0 || cations.length === 0) return;

  let totalPos = cations.reduce((s, el) => s + (assignment[el] ?? 0) * counts[el], 0);
  let totalNeg = anions.reduce((s, el) => s + Math.abs(assignment[el] ?? 0) * counts[el], 0);

  if (totalPos === totalNeg) return;

  for (let iter = 0; iter < 5 && totalPos !== totalNeg; iter++) {
    if (totalPos > totalNeg) {
      const target = anions.sort((a, b) => counts[a] - counts[b])[0];
      if (!target) break;
      counts[target]++;
      totalNeg += Math.abs(assignment[target] ?? 1);
    } else {
      const target = cations.sort((a, b) => counts[a] - counts[b])[0];
      if (!target) break;
      counts[target]++;
      totalPos += (assignment[target] ?? 1);
    }
  }
}

export function constraintGuidedGenerate(
  rawFormulas: string[],
): { valid: string[]; repaired: string[]; rejected: string[]; details: ConstraintResult[] } {
  const valid: string[] = [];
  const repaired: string[] = [];
  const rejected: string[] = [];
  const details: ConstraintResult[] = [];

  for (const formula of rawFormulas) {
    const result = checkPhysicsConstraints(formula);
    details.push(result);

    if (result.isValid) {
      valid.push(formula);
    } else if (result.repairedFormula) {
      repaired.push(result.repairedFormula);
    } else {
      rejected.push(formula);
    }
  }

  return { valid, repaired, rejected, details };
}

export function updateConstraintWeightsFromReward(
  formula: string,
  tcReward: number,
  violations: ConstraintViolation[],
): void {
  const normalizedReward = Math.min(1.0, Math.max(-1.0, tcReward / 200));

  for (const v of violations) {
    if (!constraintRewards[v.type]) {
      constraintRewards[v.type] = { totalReward: 0, count: 0 };
    }
    constraintRewards[v.type].totalReward += normalizedReward;
    constraintRewards[v.type].count++;
  }

  const lr = 0.005;
  for (const [ctype, data] of Object.entries(constraintRewards)) {
    if (data.count < 5) continue;
    const avgReward = data.totalReward / data.count;
    if (avgReward > 0.3) {
      constraintWeights[ctype] = Math.max(0.1, constraintWeights[ctype] - lr);
    } else if (avgReward < -0.1) {
      constraintWeights[ctype] = Math.min(2.0, constraintWeights[ctype] + lr);
    }
  }
}

export function getConstraintEngineStats(): ConstraintStats {
  const topPatterns = Array.from(repairPatterns.values())
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  return {
    totalChecked,
    totalValid,
    totalRepaired,
    totalRejected,
    violationCounts: { ...violationCounts },
    avgPenalty: totalChecked > 0 ? penaltySum / totalChecked : 0,
    constraintWeights: { ...constraintWeights },
    repairSuccessRate: repairAttempts > 0 ? repairSuccesses / repairAttempts : 0,
    topRepairPatterns: topPatterns,
  };
}
