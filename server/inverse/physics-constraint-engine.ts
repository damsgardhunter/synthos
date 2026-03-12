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

function flattenParentheses(input: string): string {
  let result = input;
  let safety = 20;
  while (result.includes("(") && safety-- > 0) {
    result = result.replace(/\(([^()]+)\)(\d*\.?\d*)/g, (_, inner, mult) => {
      const m = mult ? parseFloat(mult) : 1;
      if (m === 1) return inner;
      return inner.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_: string, el: string, n: string) => {
        const count = n ? parseFloat(n) : 1;
        return `${el}${count * m}`;
      });
    });
  }
  return result;
}

function parseCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") return {};
  let cleaned = formula;
  cleaned = cleaned.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻]/g, "");
  cleaned = cleaned.replace(/\d*[+\-](?![a-z\d])/g, "");
  cleaned = cleaned.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  cleaned = cleaned.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]/g, c => String("⁰¹²³⁴⁵⁶⁷⁸⁹".indexOf(c)));
  cleaned = cleaned.replace(/[^\x20-\x7E]/g, "");
  cleaned = cleaned.replace(/,/g, "");
  cleaned = flattenParentheses(cleaned);
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

function formulaElementSort(a: string, b: string): number {
  const enA = ELEMENTAL_DATA[a]?.paulingElectronegativity ?? 2.0;
  const enB = ELEMENTAL_DATA[b]?.paulingElectronegativity ?? 2.0;
  if (enA !== enB) return enA - enB;
  return a.localeCompare(b);
}

function countsToFormula(counts: Record<string, number>): string {
  const entries = Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort(([a], [b]) => formulaElementSort(a, b));
  return entries.map(([el, n]) => {
    if (Number.isInteger(n)) return n === 1 ? el : `${el}${n}`;
    if (n < 1) return `${el}${parseFloat(n.toFixed(2))}`;
    const rounded = Math.round(n);
    if (Math.abs(n - rounded) < 0.01) return rounded === 1 ? el : `${el}${rounded}`;
    return `${el}${parseFloat(n.toFixed(1))}`;
  }).join("");
}

function getOxidationStates(el: string): number[] {
  return OXIDATION_STATES[el] || [];
}

function getRadius(el: string): number {
  return ELEMENTAL_DATA[el]?.atomicRadius ?? 130;
}

const IONIC_RADII: Record<string, Record<number, number>> = {
  H: { 1: 10, "-1": 154 }, Li: { 1: 76 }, Na: { 1: 102 }, K: { 1: 138 }, Rb: { 1: 152 }, Cs: { 1: 167 },
  Be: { 2: 45 }, Mg: { 2: 72 }, Ca: { 2: 100 }, Sr: { 2: 118 }, Ba: { 2: 135 },
  Sc: { 3: 75 }, Y: { 3: 90 }, La: { 3: 103 },
  Ti: { 2: 86, 3: 67, 4: 61 }, Zr: { 4: 72 }, Hf: { 4: 71 },
  V: { 2: 79, 3: 64, 4: 58, 5: 54 }, Nb: { 3: 72, 5: 64 }, Ta: { 5: 64 },
  Cr: { 2: 80, 3: 62, 6: 44 }, Mo: { 4: 65, 6: 59 }, W: { 4: 66, 6: 60 },
  Mn: { 2: 83, 3: 65, 4: 53, 7: 46 }, Re: { 4: 63, 7: 53 },
  Fe: { 2: 78, 3: 65 }, Ru: { 3: 68, 4: 62 }, Os: { 4: 63 },
  Co: { 2: 75, 3: 61 }, Rh: { 3: 67 }, Ir: { 3: 68, 4: 63 },
  Ni: { 2: 69, 3: 56 }, Pd: { 2: 86, 4: 62 }, Pt: { 2: 80, 4: 63 },
  Cu: { 1: 77, 2: 73 }, Ag: { 1: 115 }, Au: { 1: 137, 3: 85 },
  Zn: { 2: 74 }, Cd: { 2: 95 },
  B: { 3: 27 }, Al: { 3: 54 }, Ga: { 3: 62 }, In: { 3: 80 }, Tl: { 1: 150, 3: 89 },
  C: { 4: 16, "-4": 260 }, Si: { 4: 40, "-4": 271 }, Ge: { 4: 53 }, Sn: { 2: 93, 4: 69 }, Pb: { 2: 119, 4: 78 },
  N: { "-3": 146, 3: 16, 5: 13 }, P: { "-3": 212, 3: 44, 5: 38 }, As: { "-3": 222, 3: 58, 5: 46 }, Sb: { 3: 76, 5: 60 }, Bi: { 3: 103, 5: 76 },
  O: { "-2": 140 }, S: { "-2": 184, 4: 37, 6: 29 }, Se: { "-2": 198, 4: 50, 6: 42 }, Te: { "-2": 221, 4: 66, 6: 56 },
  F: { "-1": 133 }, Cl: { "-1": 181 }, Br: { "-1": 196 }, I: { "-1": 220 },
  Ce: { 3: 101, 4: 87 }, Pr: { 3: 99, 4: 85 }, Nd: { 3: 98 }, Sm: { 2: 122, 3: 96 },
  Eu: { 2: 117, 3: 95 }, Gd: { 3: 94 }, Tb: { 3: 92, 4: 76 }, Dy: { 3: 91 },
  Ho: { 3: 90 }, Er: { 3: 89 }, Tm: { 3: 88 }, Yb: { 2: 102, 3: 87 }, Lu: { 3: 86 },
  Th: { 4: 94 }, U: { 3: 103, 4: 89, 5: 76, 6: 73 },
};

function getIonicRadius(el: string, oxState?: number): number {
  if (oxState !== undefined) {
    const radii = IONIC_RADII[el];
    if (radii) {
      const key = String(oxState);
      if (radii[key as any] !== undefined) return radii[key as any];
      const entries = Object.entries(radii);
      if (entries.length > 0) {
        const closest = entries.reduce((best, curr) =>
          Math.abs(Number(curr[0]) - oxState) < Math.abs(Number(best[0]) - oxState) ? curr : best
        );
        return Number(closest[1]);
      }
    }
  }
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
  const nonMetals = new Set(["C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"]);
  const hasNonMetalPartner = elements.some(el => nonMetals.has(el) || metalloids.has(el));
  const metalOrMetalloid = (el: string): boolean => {
    if (el === "H") return !hasNonMetalPartner;
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

    const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
    if (totalAtoms > 0) {
      let totalVE = 0;
      for (const el of elements) {
        const ve = ELEMENTAL_DATA[el]?.valenceElectrons ?? 0;
        totalVE += ve * counts[el];
      }
      const vec = totalVE / totalAtoms;

      if (vec < 0.5 || vec > 14.0) {
        const deviation = vec < 0.5 ? 0.5 - vec : vec - 14.0;
        return {
          imbalance: Math.max(1, Math.ceil(deviation)),
          bestAssignment: assignment,
          achievable: false,
        };
      }
    }

    return { imbalance: 0, bestAssignment: assignment, achievable: true };
  }

  const oxStates = elements.map(el => getOxidationStates(el));
  let bestImbalance = Infinity;
  let bestAssignment: Record<string, number> = {};

  const totalCombinations = oxStates.reduce((prod, s) => prod * Math.max(s.length, 1), 1);

  if (totalCombinations <= 5000) {
    type DPEntry = { sum: number; assignment: Record<string, number> };
    let frontier: DPEntry[] = [{ sum: 0, assignment: {} }];

    for (let idx = 0; idx < elements.length; idx++) {
      const el = elements[idx];
      const count = counts[el];
      const states = oxStates[idx];
      if (states.length === 0) {
        for (const entry of frontier) {
          entry.assignment[el] = 0;
        }
        continue;
      }
      const next: DPEntry[] = [];
      for (const entry of frontier) {
        for (const ox of states) {
          next.push({
            sum: entry.sum + ox * count,
            assignment: { ...entry.assignment, [el]: ox },
          });
        }
      }
      const seen = new Map<number, DPEntry>();
      for (const entry of next) {
        if (!seen.has(entry.sum)) {
          seen.set(entry.sum, entry);
        }
      }
      frontier = Array.from(seen.values());
    }

    for (const entry of frontier) {
      const imb = Math.abs(entry.sum);
      if (imb < bestImbalance) {
        bestImbalance = imb;
        bestAssignment = entry.assignment;
      }
    }
  } else {
    const assignment: Record<string, number> = {};
    for (let i = 0; i < elements.length; i++) {
      const states = oxStates[i];
      if (states.length === 0) { assignment[elements[i]] = 0; continue; }
      const en = getEN(elements[i]);
      if (en > 2.5) {
        assignment[elements[i]] = states.find(s => s < 0) ?? states[0];
      } else {
        assignment[elements[i]] = states.find(s => s > 0) ?? states[0];
      }
    }

    function computeSum(a: Record<string, number>): number {
      let s = 0;
      for (const el of elements) s += (a[el] ?? 0) * counts[el];
      return s;
    }

    bestImbalance = Math.abs(computeSum(assignment));
    bestAssignment = { ...assignment };

    const SA_RESTARTS = 5;
    const ITERS_PER_RESTART = 400;
    const cooling = 0.993;

    for (let restart = 0; restart < SA_RESTARTS && bestImbalance > 0; restart++) {
      const current: Record<string, number> = {};
      for (let i = 0; i < elements.length; i++) {
        const states = oxStates[i];
        if (states.length === 0) { current[elements[i]] = 0; continue; }
        if (restart === 0) {
          current[elements[i]] = assignment[elements[i]];
        } else {
          current[elements[i]] = states[Math.floor(Math.random() * states.length)];
        }
      }
      let currentImb = Math.abs(computeSum(current));
      let temp = 1.0;

      for (let iter = 0; iter < ITERS_PER_RESTART && currentImb > 0; iter++) {
        const elIdx = Math.floor(Math.random() * elements.length);
        const el = elements[elIdx];
        const states = oxStates[elIdx];
        if (states.length <= 1) { temp *= cooling; continue; }

        const oldOx = current[el];
        const candidates = states.filter(s => s !== oldOx);
        if (candidates.length === 0) { temp *= cooling; continue; }
        const newOx = candidates[Math.floor(Math.random() * candidates.length)];

        current[el] = newOx;
        const newImb = Math.abs(computeSum(current));
        const delta = newImb - currentImb;

        if (delta <= 0 || Math.random() < Math.exp(-delta / Math.max(temp, 0.01))) {
          currentImb = newImb;
          if (currentImb < bestImbalance) {
            bestImbalance = currentImb;
            bestAssignment = { ...current };
          }
        } else {
          current[el] = oldOx;
        }
        temp *= cooling;
      }
    }
  }

  return {
    imbalance: bestImbalance,
    bestAssignment,
    achievable: bestImbalance < 0.01,
  };
}

const radiusCache = new Map<string, number>();

function getCachedRadius(el: string, oxState?: number): number {
  const key = oxState !== undefined && oxState !== 0 ? `${el}:${oxState}` : el;
  let r = radiusCache.get(key);
  if (r === undefined) {
    r = (oxState !== undefined && oxState !== 0) ? getIonicRadius(el, oxState) : getRadius(el);
    radiusCache.set(key, r);
  }
  return r;
}

const INTERSTITIAL_RATIOS: Record<string, number> = {
  octahedral: 0.414,
  tetrahedral: 0.225,
};

function checkRadiusCompatibility(counts: Record<string, number>, chargeAssignment?: Record<string, number>): {
  score: number;
  violations: string[];
} {
  const elements = Object.keys(counts);
  if (elements.length < 2) return { score: 1.0, violations: [] };

  const violations: string[] = [];
  let totalPairs = 0;
  let compatiblePairs = 0;

  const hCount = counts["H"] ?? 0;
  const hasH = hCount > 0;
  const nonHElements = elements.filter(el => el !== "H");

  if (hasH && nonHElements.length > 0) {
    let weightedRadiusSum = 0;
    let totalHostAtoms = 0;
    for (const el of nonHElements) {
      const r = getCachedRadius(el, chargeAssignment?.[el]);
      weightedRadiusSum += r * counts[el];
      totalHostAtoms += counts[el];
    }
    const avgHostRadius = totalHostAtoms > 0 ? weightedRadiusSum / totalHostAtoms : 130;

    const hRadius = getCachedRadius("H", chargeAssignment?.["H"]);

    const tetraHoleRadius = avgHostRadius * INTERSTITIAL_RATIOS.tetrahedral;
    const octaHoleRadius = avgHostRadius * INTERSTITIAL_RATIOS.octahedral;

    if (hRadius > avgHostRadius * 0.8) {
      totalPairs++;
      violations.push(`H radius (${hRadius} pm) too large relative to host lattice (avg ${avgHostRadius.toFixed(0)} pm)`);
    } else {
      totalPairs++;
      compatiblePairs++;
    }

    const maxOctaSites = totalHostAtoms;
    const maxTetraSites = totalHostAtoms * 2;
    let maxHSites: number;
    if (hRadius <= tetraHoleRadius) {
      maxHSites = maxOctaSites + maxTetraSites;
    } else if (hRadius <= octaHoleRadius) {
      maxHSites = maxOctaSites;
    } else {
      maxHSites = Math.ceil(maxOctaSites * 0.25);
    }

    if (hCount > maxHSites * 1.5) {
      violations.push(`H count (${hCount}) exceeds estimated interstitial capacity (~${maxHSites} sites)`);
    }
  }

  for (let i = 0; i < nonHElements.length; i++) {
    for (let j = i + 1; j < nonHElements.length; j++) {
      totalPairs++;
      const elA = nonHElements[i];
      const elB = nonHElements[j];

      const rA = getCachedRadius(elA, chargeAssignment?.[elA]);
      const rB = getCachedRadius(elB, chargeAssignment?.[elB]);
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

const RADIUS_RATIO_COORD: [number, number][] = [
  [0.155, 2], [0.225, 3], [0.414, 4], [0.732, 6], [1.0, 8],
];

function maxCoordByRadiusRatio(rCation: number, rAnion: number): number {
  if (rAnion <= 0) return 12;
  const ratio = rCation / rAnion;
  for (const [threshold, coord] of RADIUS_RATIO_COORD) {
    if (ratio < threshold) return coord;
  }
  return 12;
}

function weightedAnionRadius(anions: string[], counts: Record<string, number>, chargeAssignment?: Record<string, number>): number {
  let totalWeight = 0;
  let weightedSum = 0;
  for (const a of anions) {
    const w = counts[a] ?? 1;
    weightedSum += getCachedRadius(a, chargeAssignment?.[a]) * w;
    totalWeight += w;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 130;
}

function checkCoordination(counts: Record<string, number>, chargeAssignment?: Record<string, number>, pressureGPa?: number): {
  score: number;
  violations: string[];
} {
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms === 0) return { score: 0, violations: ["empty formula"] };

  const violations: string[] = [];
  let validElements = 0;
  let total = 0;

  const cations = chargeAssignment
    ? elements.filter(el => (chargeAssignment[el] ?? 0) > 0)
    : [];
  const anions = chargeAssignment
    ? elements.filter(el => (chargeAssignment[el] ?? 0) < 0)
    : [];

  const isHighPressure = (pressureGPa ?? 0) > 50;

  for (const el of elements) {
    total++;
    const limits = COORDINATION_LIMITS[el];
    if (!limits) { validElements++; continue; }

    if (el === "H") {
      if (isHighPressure) {
        const hCount = counts["H"] ?? 0;
        const nonHAtoms = totalAtoms - hCount;
        if (nonHAtoms > 0 && hCount / nonHAtoms > 12) {
          violations.push(`H/host ratio ${(hCount / nonHAtoms).toFixed(1)} exceeds high-pressure cage limit (12)`);
        } else {
          validElements++;
        }
      } else {
        validElements++;
      }
      continue;
    }

    let maxByRadius = limits[1];
    if (chargeAssignment && (chargeAssignment[el] ?? 0) > 0 && anions.length > 0) {
      const cationR = getCachedRadius(el, chargeAssignment[el]);
      const wAnionR = weightedAnionRadius(anions, counts, chargeAssignment);
      maxByRadius = Math.min(limits[1], maxCoordByRadiusRatio(cationR, wAnionR));
    }

    const coordLimit = Math.max(maxByRadius, limits[0]);

    if (coordLimit >= limits[0]) {
      validElements++;
    } else if (elements.length >= 3) {
      if (maxByRadius < limits[0]) {
        violations.push(`${el} radius ratio limits coordination to ${maxByRadius} but needs ${limits[0]}`);
      }
      validElements++;
    } else {
      violations.push(`${el} coordination limited to ${maxByRadius} by radius ratio (min required: ${limits[0]})`);
    }
  }

  return {
    score: total > 0 ? validElements / total : 1.0,
    violations,
  };
}

function checkBondStability(counts: Record<string, number>, pressureGPa?: number): {
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

  const isHighPressure = (pressureGPa ?? 0) > 50;
  const hasAnyMetal = elements.some(el => {
    if (el === "H") return false;
    const en = getEN(el);
    return en <= 2.0;
  });
  const isHighPressureHydride = isHighPressure && elements.includes("H");
  if (!hasAnyMetal && elements.length > 1 && !isHighPressureHydride) {
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

  const SC_VEC_LOW = 2.0;
  const SC_VEC_HIGH = 7.0;
  if (valencePerAtom < SC_VEC_LOW) {
    const deviation = SC_VEC_LOW - valencePerAtom;
    const penalty = Math.min(0.3, deviation * 0.15);
    score -= penalty;
    violations.push(`Low valence density ${valencePerAtom.toFixed(2)} e/atom (SC sweet spot: ${SC_VEC_LOW}-${SC_VEC_HIGH})`);
  } else if (valencePerAtom > SC_VEC_HIGH) {
    const deviation = valencePerAtom - SC_VEC_HIGH;
    const penalty = Math.min(0.2, deviation * 0.05);
    score -= penalty;
    violations.push(`High valence density ${valencePerAtom.toFixed(2)} e/atom (SC sweet spot: ${SC_VEC_LOW}-${SC_VEC_HIGH})`);
  }

  return { score: Math.max(0, score), violations };
}

let constraintWeights: Record<string, number> = {
  charge_neutrality: 1.0,
  radius_incompatibility: 0.6,
  coordination_violation: 0.5,
  bond_instability: 0.7,
  electron_count: 0.15,
  noble_gas: 1.0,
  stoichiometry_excess: 0.8,
};

const WEIGHT_FLOORS: Record<string, number> = {
  charge_neutrality: 0.8,
  noble_gas: 0.7,
  bond_instability: 0.4,
  stoichiometry_excess: 0.5,
  radius_incompatibility: 0.3,
  coordination_violation: 0.2,
  electron_count: 0.1,
};

export class ConstraintRegistry {
  totalChecked = 0;
  totalValid = 0;
  totalRepaired = 0;
  totalRejected = 0;
  violationCounts: Record<string, number> = {};
  penaltySum = 0;
  repairAttempts = 0;
  repairSuccesses = 0;
  repairPatterns: Map<string, { from: string; to: string; count: number }> = new Map();
  constraintRewards: Record<string, { totalReward: number; count: number }> = {};
  weightUpdateCount = 0;
  private _topPatternsCache: { from: string; to: string; count: number }[] | null = null;
  private _topPatternsCacheAt = 0;

  getTopPatterns(): { from: string; to: string; count: number }[] {
    if (this._topPatternsCache && this.totalChecked - this._topPatternsCacheAt < 100) {
      return this._topPatternsCache;
    }
    this._topPatternsCache = Array.from(this.repairPatterns.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
    this._topPatternsCacheAt = this.totalChecked;
    return this._topPatternsCache;
  }

  invalidateTopPatternsCache(): void {
    this._topPatternsCache = null;
  }

  reset(): void {
    this.totalChecked = 0;
    this.totalValid = 0;
    this.totalRepaired = 0;
    this.totalRejected = 0;
    this.violationCounts = {};
    this.penaltySum = 0;
    this.repairAttempts = 0;
    this.repairSuccesses = 0;
    this.repairPatterns = new Map();
    this.constraintRewards = {};
    this.weightUpdateCount = 0;
    this._topPatternsCache = null;
    this._topPatternsCacheAt = 0;
  }
}

const globalRegistry = new ConstraintRegistry();

export function createConstraintRegistry(): ConstraintRegistry {
  return new ConstraintRegistry();
}

export function checkPhysicsConstraints(formula: string, options?: { maxPressureGPa?: number; autoRepair?: boolean; _inRepairCheck?: boolean; registry?: ConstraintRegistry }): ConstraintResult {
  const registry = options?.registry ?? globalRegistry;
  registry.totalChecked++;
  const counts = parseCounts(formula);
  const elements = Object.keys(counts);
  const violations: ConstraintViolation[] = [];
  const pressureGPa = options?.maxPressureGPa ?? 0;

  const NOBLE_PRESSURE_THRESHOLDS: Record<string, number> = { Xe: 50, Kr: 50, Ar: 100 };
  for (const el of elements) {
    if (NOBLE_GASES.has(el)) {
      const threshold = NOBLE_PRESSURE_THRESHOLDS[el];
      if (threshold !== undefined && pressureGPa > threshold) {
        violations.push({
          type: "noble_gas",
          severity: 0.2,
          detail: `Noble gas ${el} — may form compounds above ${threshold} GPa (target: ${pressureGPa} GPa)`,
        });
      } else {
        violations.push({
          type: "noble_gas",
          severity: 1.0,
          detail: `Noble gas ${el} cannot form stable compounds`,
        });
      }
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

  const elLimit = pressureGPa > 50 ? 16 : 12;
  for (const [el, n] of Object.entries(counts)) {
    if (n > elLimit) {
      violations.push({
        type: "stoichiometry_excess",
        severity: 0.6,
        detail: `${el}${Math.round(n)} exceeds stoichiometry limit (max ${elLimit})`,
        repairSuggestion: `Reduce ${el} count`,
      });
    }
  }

  let chargeResult = checkChargeNeutrality(counts);
  let autoRepaired = false;
  if (chargeResult.imbalance >= 0.01 && options?.autoRepair && chargeResult.imbalance <= 6) {
    const repairedCounts = { ...counts };
    repairChargeBalance(repairedCounts, chargeResult);
    const recharged = checkChargeNeutrality(repairedCounts);
    if (recharged.imbalance < chargeResult.imbalance) {
      Object.assign(counts, repairedCounts);
      chargeResult = recharged;
      autoRepaired = true;

      const newTotal = Object.values(counts).reduce((s, n) => s + n, 0);
      if (newTotal > 25) {
        violations.push({
          type: "stoichiometry_excess",
          severity: 0.7,
          detail: `Excessive total atoms after auto-repair: ${newTotal}`,
          repairSuggestion: "Reduce stoichiometric coefficients",
        });
      }
      for (const [el, n] of Object.entries(counts)) {
        if (n > elLimit) {
          violations.push({
            type: "stoichiometry_excess",
            severity: 0.6,
            detail: `${el}${Math.round(n)} exceeds stoichiometry limit (max ${elLimit}) after auto-repair`,
            repairSuggestion: `Reduce ${el} count`,
          });
        }
      }
    }
  }
  if (chargeResult.imbalance >= 0.01) {
    violations.push({
      type: "charge_neutrality",
      severity: Math.min(1.0, chargeResult.imbalance / 6),
      detail: `Charge imbalance: ${chargeResult.imbalance.toFixed(2)} (states: ${Object.entries(chargeResult.bestAssignment).map(([e, s]) => `${e}${s > 0 ? "+" : ""}${s}`).join(", ")})`,
      repairSuggestion: chargeResult.imbalance <= 4 ? "Adjust stoichiometry for charge balance" : undefined,
    });
  }

  const radiusResult = checkRadiusCompatibility(counts, chargeResult.achievable ? chargeResult.bestAssignment : undefined);
  for (const v of radiusResult.violations) {
    violations.push({
      type: "radius_incompatibility",
      severity: 0.5,
      detail: v,
    });
  }

  const coordResult = checkCoordination(counts, chargeResult.achievable ? chargeResult.bestAssignment : undefined, pressureGPa);
  for (const v of coordResult.violations) {
    violations.push({
      type: "coordination_violation",
      severity: 0.4,
      detail: v,
    });
  }

  const bondResult = checkBondStability(counts, pressureGPa);
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
    registry.violationCounts[v.type] = (registry.violationCounts[v.type] || 0) + 1;
  }

  const isValid = totalPenalty < 0.5;
  if (isValid) registry.totalValid++;
  else registry.totalRejected++;

  registry.penaltySum += totalPenalty;

  const autoRepairedFormula = options?.autoRepair ? countsToFormula(counts) : null;
  const effectiveFormula = (autoRepairedFormula && autoRepairedFormula !== formula) ? autoRepairedFormula : formula;

  let repairedFormula: string | null = null;
  if (!isValid && totalPenalty < 3.5 && !options?._inRepairCheck) {
    repairedFormula = repairFormula(effectiveFormula, counts, violations, chargeResult, pressureGPa, registry);
    if (repairedFormula && repairedFormula !== effectiveFormula) {
      registry.totalRepaired++;
    }
  }

  if (!repairedFormula && autoRepaired && autoRepairedFormula && autoRepairedFormula !== formula) {
    repairedFormula = autoRepairedFormula;
    registry.totalRepaired++;
  }

  return {
    formula: effectiveFormula,
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
  pressureGPa?: number,
  registry: ConstraintRegistry = globalRegistry,
): string | null {
  registry.repairAttempts++;
  const repaired = { ...counts };
  const NOBLE_PRESSURE_THRESHOLDS: Record<string, number> = { Xe: 50, Kr: 50, Ar: 100 };

  let removedNobleGas = false;
  for (const el of Object.keys(repaired)) {
    if (NOBLE_GASES.has(el)) {
      const threshold = NOBLE_PRESSURE_THRESHOLDS[el];
      if (threshold !== undefined && (pressureGPa ?? 0) > threshold) continue;
      delete repaired[el];
      removedNobleGas = true;
    }
  }

  const repairLimit = (pressureGPa ?? 0) > 50 ? 16 : 12;
  for (const [el, n] of Object.entries(repaired)) {
    if (n > repairLimit) {
      repaired[el] = Math.min(n, repairLimit - 4);
    }
  }

  const effectiveChargeResult = removedNobleGas ? checkChargeNeutrality(repaired) : chargeResult;
  if (effectiveChargeResult.imbalance >= 0.01 && effectiveChargeResult.imbalance <= 6) {
    repairChargeBalance(repaired, effectiveChargeResult);
  }

  const totalAtoms = Object.values(repaired).reduce((s, n) => s + n, 0);
  if (totalAtoms > 25) {
    const targetTotal = 20;
    const scale = targetTotal / totalAtoms;
    const elKeys = Object.keys(repaired);
    const scaled = elKeys.map(el => Math.max(1, repaired[el] * scale));
    const floored = scaled.map(v => Math.max(1, Math.floor(v)));
    let deficit = targetTotal - floored.reduce((s, n) => s + n, 0);
    if (deficit > 0) {
      const remainders = scaled.map((v, i) => ({ i, r: v - floored[i] }))
        .sort((a, b) => b.r - a.r);
      for (const { i } of remainders) {
        if (deficit <= 0) break;
        floored[i]++;
        deficit--;
      }
    } else if (deficit < 0) {
      const sortedByCount = floored.map((v, i) => ({ i, v }))
        .sort((a, b) => b.v - a.v);
      for (const { i } of sortedByCount) {
        if (deficit >= 0) break;
        if (floored[i] > 1) {
          floored[i]--;
          deficit++;
        }
      }
    }
    for (let j = 0; j < elKeys.length; j++) {
      repaired[elKeys[j]] = floored[j];
    }
    const postScaleCharge = checkChargeNeutrality(repaired);
    if (postScaleCharge.imbalance >= 0.01 && postScaleCharge.imbalance <= 6) {
      repairChargeBalance(repaired, postScaleCharge);
    }
  }

  const remainingEls = Object.keys(repaired).filter(el => repaired[el] > 0);
  if (remainingEls.length < 2) return null;

  const result = countsToFormula(repaired);
  if (result === formula) return null;

  const recheck = checkPhysicsConstraints(result, { maxPressureGPa: pressureGPa, _inRepairCheck: true, registry });
  registry.totalChecked--;

  const origRecheck = recheck.totalPenalty >= 0.5
    ? checkPhysicsConstraints(formula, { maxPressureGPa: pressureGPa, _inRepairCheck: true, registry })
    : null;
  if (origRecheck) registry.totalChecked--;

  const accepted = recheck.totalPenalty < 0.5
    || (origRecheck && recheck.totalPenalty < origRecheck.totalPenalty * 0.9);

  if (!accepted) return null;

  registry.repairSuccesses++;
  const key = `${formula}->${result}`;
  const existing = registry.repairPatterns.get(key);
  if (existing) existing.count++;
  else {
    registry.repairPatterns.set(key, { from: formula, to: result, count: 1 });
    if (registry.repairPatterns.size > 5000) {
      const oldest = registry.repairPatterns.keys().next().value;
      if (oldest) registry.repairPatterns.delete(oldest);
    }
  }
  registry.invalidateTopPatternsCache();
  return result;
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

  for (let iter = 0; iter < 10; iter++) {
    const freshCharge = checkChargeNeutrality(counts);
    if (freshCharge.imbalance < 0.01) break;

    const freshAssignment = freshCharge.bestAssignment;
    const freshAnions = elements.filter(el => counts[el] > 0 && (freshAssignment[el] ?? 0) < 0);
    const freshCations = elements.filter(el => counts[el] > 0 && (freshAssignment[el] ?? 0) > 0);
    const freshPos = freshCations.reduce((s, el) => s + (freshAssignment[el] ?? 0) * counts[el], 0);
    const freshNeg = freshAnions.reduce((s, el) => s + Math.abs(freshAssignment[el] ?? 0) * counts[el], 0);
    const candidates = freshPos > freshNeg ? freshAnions : freshCations;
    if (candidates.length === 0) break;

    let bestEl: string | null = null;
    let bestPenalty = Infinity;

    for (const el of candidates) {
      const testCounts = { ...counts };
      testCounts[el]++;
      const testCharge = checkChargeNeutrality(testCounts);
      const radiusCheck = checkRadiusCompatibility(testCounts, testCharge.achievable ? testCharge.bestAssignment : undefined);
      const penalty = testCharge.imbalance + (1.0 - radiusCheck.score) * 0.5;
      if (penalty < bestPenalty) {
        bestPenalty = penalty;
        bestEl = el;
      }
    }

    if (!bestEl || bestPenalty >= freshCharge.imbalance) break;
    counts[bestEl]++;
  }
}

export function constraintGuidedGenerate(
  rawFormulas: string[],
  options?: { maxPressureGPa?: number; registry?: ConstraintRegistry },
): { valid: string[]; repaired: string[]; rejected: string[]; details: ConstraintResult[] } {
  const valid: string[] = [];
  const repaired: string[] = [];
  const rejected: string[] = [];
  const details: ConstraintResult[] = [];

  for (const formula of rawFormulas) {
    const result = checkPhysicsConstraints(formula, { maxPressureGPa: options?.maxPressureGPa, registry: options?.registry });
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
  reg?: ConstraintRegistry,
): void {
  const registry = reg ?? globalRegistry;
  const normalizedReward = Math.min(1.0, Math.max(-1.0, tcReward / 200));

  for (const v of violations) {
    if (!registry.constraintRewards[v.type]) {
      registry.constraintRewards[v.type] = { totalReward: 0, count: 0 };
    }
    registry.constraintRewards[v.type].totalReward += normalizedReward;
    registry.constraintRewards[v.type].count++;
  }

  const baseLr = 0.01;
  const lr = baseLr / (1 + 0.001 * registry.weightUpdateCount);

  let anyUpdated = false;
  for (const [ctype, data] of Object.entries(registry.constraintRewards)) {
    if (data.count < 5) continue;
    const avgReward = data.totalReward / data.count;
    const floor = WEIGHT_FLOORS[ctype] ?? 0.1;
    if (avgReward > 0.3) {
      constraintWeights[ctype] = Math.max(floor, constraintWeights[ctype] - lr);
      anyUpdated = true;
    } else if (avgReward < -0.1) {
      constraintWeights[ctype] = Math.min(2.0, constraintWeights[ctype] + lr);
      anyUpdated = true;
    }
  }
  if (anyUpdated) registry.weightUpdateCount++;
}

export function resetConstraintStats(reg?: ConstraintRegistry): void {
  (reg ?? globalRegistry).reset();
}

export function getConstraintEngineStats(reg?: ConstraintRegistry): ConstraintStats {
  const registry = reg ?? globalRegistry;

  return {
    totalChecked: registry.totalChecked,
    totalValid: registry.totalValid,
    totalRepaired: registry.totalRepaired,
    totalRejected: registry.totalRejected,
    violationCounts: { ...registry.violationCounts },
    avgPenalty: registry.totalChecked > 0 ? registry.penaltySum / registry.totalChecked : 0,
    constraintWeights: { ...constraintWeights },
    repairSuccessRate: registry.repairAttempts > 0 ? registry.repairSuccesses / registry.repairAttempts : 0,
    topRepairPatterns: registry.getTopPatterns(),
  };
}
