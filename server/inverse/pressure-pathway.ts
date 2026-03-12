import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";
import {
  getElementData,
  ELEMENTAL_DATA,
} from "../learning/elemental-data";

export interface PressurePathway {
  sourceFormula: string;
  sourceTc: number;
  sourcePressure: number;
  strategies: StabilizationStrategy[];
  bestAmbientTc: number;
  bestAmbientFormula: string;
  retentionPercent: number;
  feasibility: number;
}

interface StabilizationStrategy {
  type: "chemical-doping" | "isovalent-substitution" | "cage-filler" | "anion-substitution" | "pre-compression";
  formula: string;
  description: string;
  estimatedAmbientTc: number;
  tcRetention: number;
  stabilityGain: number;
  pressureReduction: number;
  confidence: number;
}

interface PathwayCandidate {
  formula: string;
  strategy: string;
  estimatedTc: number;
  ambientStabilityScore: number;
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function normalizeToIntegers(counts: Record<string, number>): Record<string, number> {
  const entries = Object.entries(counts).filter(([, n]) => n > 0);
  if (entries.length === 0) return counts;

  const vals = entries.map(([, n]) => n);
  const allInteger = vals.every(n => Number.isInteger(n));
  if (allInteger) return counts;

  let multiplier = 1;
  for (let m = 1; m <= 100; m++) {
    if (vals.every(n => Math.abs(n * m - Math.round(n * m)) < 0.01)) {
      multiplier = m;
      break;
    }
  }

  const result: Record<string, number> = {};
  for (const [el, n] of entries) {
    const rounded = Math.round(n * multiplier);
    if (rounded > 0) result[el] = rounded;
  }

  const resultVals = Object.values(result);
  if (resultVals.length === 0) {
    const fallback: Record<string, number> = {};
    for (const [el, n] of entries) fallback[el] = Math.max(1, Math.round(n));
    return fallback;
  }

  const gcdAll = resultVals.reduce((a, b) => gcd(a, b));
  if (gcdAll > 1) {
    for (const el of Object.keys(result)) {
      result[el] /= gcdAll;
    }
  }

  return result;
}

function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

function hillSort(a: string, b: string): number {
  const priority = (el: string): number => {
    if (el === "C") return 0;
    if (el === "H") return 1;
    return 2;
  };
  const pa = priority(a);
  const pb = priority(b);
  if (pa !== pb) return pa - pb;
  return a.localeCompare(b);
}

function buildFormula(counts: Record<string, number>): string {
  const intCounts = normalizeToIntegers(counts);
  return Object.entries(intCounts)
    .filter(([, n]) => n > 0)
    .sort(([a], [b]) => hillSort(a, b))
    .map(([el, n]) => n === 1 ? el : `${el}${n}`)
    .join("");
}

const PERIODIC_GROUPS: Record<number, string[]> = {
  1: ["Li", "Na", "K", "Rb", "Cs"],
  2: ["Be", "Mg", "Ca", "Sr", "Ba"],
  3: ["Sc", "Y", "La", "Lu"],
  4: ["Ti", "Zr", "Hf"],
  5: ["V", "Nb", "Ta"],
  6: ["Cr", "Mo", "W"],
  7: ["Mn", "Re"],
  8: ["Fe", "Ru", "Os"],
  9: ["Co", "Rh", "Ir"],
  10: ["Ni", "Pd", "Pt"],
  11: ["Cu", "Ag", "Au"],
  12: ["Zn", "Cd"],
  13: ["B", "Al", "Ga", "In", "Tl"],
  14: ["C", "Si", "Ge", "Sn", "Pb"],
  15: ["N", "P", "As", "Sb", "Bi"],
  16: ["O", "S", "Se", "Te"],
  17: ["F", "Cl", "Br", "I"],
};

const _elementToGroup = new Map<string, number>();
for (const [g, els] of Object.entries(PERIODIC_GROUPS)) {
  for (const el of els) _elementToGroup.set(el, Number(g));
}

function getChemicalPressureSubstitutes(el: string): { smaller: string[]; larger: string[] } {
  const data = getElementData(el);
  if (!data) return { smaller: [], larger: [] };
  const group = _elementToGroup.get(el);
  if (group === undefined) return { smaller: [], larger: [] };

  const groupMembers = PERIODIC_GROUPS[group] ?? [];
  const elRadius = data.atomicRadius;
  if (!elRadius) return { smaller: [], larger: [] };

  const smaller: string[] = [];
  const larger: string[] = [];
  for (const candidate of groupMembers) {
    if (candidate === el) continue;
    const cData = getElementData(candidate);
    if (!cData || !cData.atomicRadius) continue;
    if (cData.atomicRadius < elRadius) smaller.push(candidate);
    else if (cData.atomicRadius > elRadius) larger.push(candidate);
  }

  smaller.sort((a, b) => {
    const ra = getElementData(a)!.atomicRadius;
    const rb = getElementData(b)!.atomicRadius;
    return rb - ra;
  });
  larger.sort((a, b) => {
    const ra = getElementData(a)!.atomicRadius;
    const rb = getElementData(b)!.atomicRadius;
    return ra - rb;
  });

  return { smaller, larger };
}

const STABILIZING_DOPANTS: Record<string, { dopant: string; fraction: number; reason: string }[]> = {
  hydride: [
    { dopant: "B", fraction: 0.1, reason: "boron provides covalent cage reinforcement" },
    { dopant: "C", fraction: 0.05, reason: "carbon strengthens framework bonds" },
    { dopant: "N", fraction: 0.05, reason: "nitrogen adds chemical stability" },
    { dopant: "Si", fraction: 0.1, reason: "silicon stabilizes cage structures" },
    { dopant: "Al", fraction: 0.1, reason: "aluminum electron-donates and stabilizes" },
  ],
  cage: [
    { dopant: "Li", fraction: 0.1, reason: "lithium fills voids and stabilizes cage" },
    { dopant: "Be", fraction: 0.05, reason: "beryllium provides lightweight cage support" },
    { dopant: "Mg", fraction: 0.1, reason: "magnesium stabilizes through chemical bonding" },
  ],
  layered: [
    { dopant: "F", fraction: 0.1, reason: "fluorination stabilizes layered structures" },
    { dopant: "O", fraction: 0.05, reason: "oxygen insertion stabilizes intercalation" },
  ],
};

function classifyMaterial(formula: string): string[] {
  const counts = parseFormulaCounts(formula);
  const hCount = counts["H"] || 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  const types: string[] = [];

  const hFrac = totalAtoms > 0 ? hCount / totalAtoms : 0;
  if (hFrac > 0.5) types.push("hydride");
  if (hCount > 0 && hFrac > 0.3 && Object.keys(counts).length <= 3) types.push("cage");

  const layeredElements = ["Cu", "Fe", "Ni", "Bi", "Tl"];
  const hasLayered = Object.keys(counts).some(el => layeredElements.includes(el));
  if (hasLayered && Object.keys(counts).length >= 3) types.push("layered");

  if (types.length === 0) types.push("conventional");
  return types;
}

function estimateChemicalPressureEffect(
  originalFormula: string,
  originalPressure: number,
  substitutionElement: string,
  targetElement: string,
): { pressureReduction: number; tcRetention: number } {
  const origData = getElementData(targetElement);
  const subData = getElementData(substitutionElement);

  if (!origData || !subData) return { pressureReduction: 0, tcRetention: 0.5 };

  const origRadius = origData.atomicRadius;
  const subRadius = subData.atomicRadius;
  if (!origRadius || !subRadius) return { pressureReduction: 0, tcRetention: 0.5 };

  const radiusDiff = (origRadius - subRadius) / origRadius;

  const pressureEquivalent = radiusDiff * originalPressure * 0.3;
  const pressureReduction = Math.max(0, Math.min(originalPressure * 0.8, pressureEquivalent));

  const origMass = origData.atomicMass;
  const subMass = subData.atomicMass;
  if (!origMass || !subMass) return { pressureReduction: Math.round(pressureReduction * 10) / 10, tcRetention: 0.5 };

  const massDiff = Math.abs(subMass - origMass) / origMass;
  const electronDiff = Math.abs((subData.paulingElectronegativity ?? 1.5) - (origData.paulingElectronegativity ?? 1.5));
  const tcRetention = Math.max(0.1, 1 - 0.3 * massDiff - 0.2 * electronDiff - 0.1 * Math.abs(radiusDiff));

  return { pressureReduction: Math.round(pressureReduction * 10) / 10, tcRetention: Math.round(tcRetention * 100) / 100 };
}

function generateIsovalentSubstitutions(formula: string, sourcePressure: number): StabilizationStrategy[] {
  const counts = parseFormulaCounts(formula);
  const strategies: StabilizationStrategy[] = [];

  for (const [el, amount] of Object.entries(counts)) {
    if (el === "H") continue;
    const chemPressure = getChemicalPressureSubstitutes(el);
    if (chemPressure.smaller.length === 0) continue;

    for (const smaller of chemPressure.smaller) {
      const effect = estimateChemicalPressureEffect(formula, sourcePressure, smaller, el);
      if (effect.pressureReduction < 5) continue;

      const newCounts = { ...counts };
      newCounts[smaller] = (newCounts[smaller] || 0) + amount;
      delete newCounts[el];
      const newFormula = buildFormula(newCounts);

      strategies.push({
        type: "isovalent-substitution",
        formula: newFormula,
        description: `Replace ${el} with smaller ${smaller} to create chemical pre-compression`,
        estimatedAmbientTc: 0,
        tcRetention: effect.tcRetention,
        stabilityGain: 0.3,
        pressureReduction: effect.pressureReduction,
        confidence: 0.6,
      });
    }
  }

  return strategies;
}

function generateChemicalDopings(formula: string, sourcePressure: number, sourceTc: number): StabilizationStrategy[] {
  const counts = parseFormulaCounts(formula);
  const matTypes = classifyMaterial(formula);
  const strategies: StabilizationStrategy[] = [];

  const seenDopants = new Set<string>();
  const dopants: { dopant: string; fraction: number; reason: string }[] = [];
  for (const t of matTypes) {
    for (const d of STABILIZING_DOPANTS[t] ?? []) {
      if (!seenDopants.has(d.dopant)) {
        seenDopants.add(d.dopant);
        dopants.push(d);
      }
    }
  }

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  for (const { dopant, fraction, reason } of dopants) {
    if (counts[dopant] && counts[dopant] > 0.5) continue;

    const rawDopant = totalAtoms * fraction;
    let scaleFactor = 1;
    if (rawDopant < 0.5 && rawDopant > 0) {
      scaleFactor = Math.ceil(1 / rawDopant);
    }
    const dopantAmount = Math.max(1, Math.round(rawDopant * scaleFactor));

    const newCounts: Record<string, number> = {};
    for (const [el, n] of Object.entries(counts)) {
      newCounts[el] = n * scaleFactor;
    }
    newCounts[dopant] = (newCounts[dopant] || 0) + dopantAmount;
    const newFormula = buildFormula(newCounts);

    const dopData = getElementData(dopant);
    const dopMass = dopData?.atomicMass ?? 20;
    const tcPenalty = dopMass > 30 ? 0.15 : 0.05;
    const stabilityBoost = 0.2 + (dopMass < 20 ? 0.1 : 0);

    strategies.push({
      type: "chemical-doping",
      formula: newFormula,
      description: `Add ${dopant} doping (${dopantAmount} atoms): ${reason}`,
      estimatedAmbientTc: 0,
      tcRetention: Math.max(0.3, 1 - tcPenalty - fraction),
      stabilityGain: stabilityBoost,
      pressureReduction: sourcePressure * 0.15,
      confidence: 0.5,
    });
  }

  return strategies;
}

function generateAnionSubstitutions(formula: string, sourcePressure: number): StabilizationStrategy[] {
  const counts = parseFormulaCounts(formula);
  const strategies: StabilizationStrategy[] = [];
  const hCount = counts["H"] || 0;

  if (hCount < 3) return strategies;

  const replacements = [
    { anion: "B", replaceFrac: 0.1, reason: "partial H->B replacement strengthens cage bonds", stabilityBoost: 0.35 },
    { anion: "C", replaceFrac: 0.08, reason: "partial H->C replacement adds covalent stability", stabilityBoost: 0.3 },
    { anion: "N", replaceFrac: 0.06, reason: "partial H->N replacement increases bond strength", stabilityBoost: 0.25 },
  ];

  for (const { anion, replaceFrac, reason, stabilityBoost } of replacements) {
    const replaceCount = Math.max(1, Math.round(hCount * replaceFrac));
    const newCounts = { ...counts };
    newCounts["H"] = hCount - replaceCount;
    newCounts[anion] = (newCounts[anion] || 0) + replaceCount;
    if (newCounts["H"] <= 0) delete newCounts["H"];
    const newFormula = buildFormula(newCounts);

    const replaceFraction = hCount > 0 ? replaceCount / hCount : 0;
    const anionData = getElementData(anion);
    const anionMass = anionData?.atomicMass ?? 11;
    const hMass = 1.008;
    const phononPenalty = replaceFraction * Math.min(1.0, (anionMass - hMass) / (12 - hMass)) * 0.6;
    const massPenalty = replaceFraction * 0.3;

    strategies.push({
      type: "anion-substitution",
      formula: newFormula,
      description: `Replace ${replaceCount}H with ${anion}: ${reason}`,
      estimatedAmbientTc: 0,
      tcRetention: Math.max(0.1, 1 - massPenalty - phononPenalty),
      stabilityGain: stabilityBoost,
      pressureReduction: sourcePressure * stabilityBoost * 0.5,
      confidence: 0.55,
    });
  }

  return strategies;
}

function estimateAmbientTc(
  sourceTc: number,
  sourcePressure: number,
  strategy: StabilizationStrategy,
): number {
  const residualPressure = Math.max(0, sourcePressure - strategy.pressureReduction);

  let pressureLossFactor = 1.0;
  if (sourcePressure > 0) {
    const pressureFractionRemaining = residualPressure / sourcePressure;
    pressureLossFactor = 0.3 + 0.7 * pressureFractionRemaining;
  }

  const baseTcAtAmbient = sourceTc * pressureLossFactor * strategy.tcRetention;
  const stabilityBonus = strategy.stabilityGain * 0.1 * sourceTc;
  const ambientTc = Math.max(1, Math.round(baseTcAtAmbient + stabilityBonus));

  return Math.min(ambientTc, 400);
}

export function searchPressurePathways(
  formula: string,
  sourceTc: number,
  sourcePressure: number,
  maxStrategies: number = 20,
): PressurePathway {
  if (sourcePressure <= 1) {
    return {
      sourceFormula: formula,
      sourceTc,
      sourcePressure,
      strategies: [],
      bestAmbientTc: sourceTc,
      bestAmbientFormula: formula,
      retentionPercent: 100,
      feasibility: 1.0,
    };
  }

  const allStrategies: StabilizationStrategy[] = [
    ...generateIsovalentSubstitutions(formula, sourcePressure),
    ...generateChemicalDopings(formula, sourcePressure, sourceTc),
    ...generateAnionSubstitutions(formula, sourcePressure),
  ];

  for (const strategy of allStrategies) {
    strategy.estimatedAmbientTc = estimateAmbientTc(sourceTc, sourcePressure, strategy);
  }

  allStrategies.sort((a, b) => b.estimatedAmbientTc - a.estimatedAmbientTc);
  const topStrategies = allStrategies.slice(0, maxStrategies);

  for (const strategy of topStrategies) {
    try {
      const electronic = computeElectronicStructure(strategy.formula, null);
      const phonon = computePhononSpectrum(strategy.formula, electronic);
      const coupling = computeElectronPhononCoupling(electronic, phonon, strategy.formula, 0);
      const omegaLogK = coupling.omegaLog * 1.4388;
      const denom = coupling.lambda - coupling.muStar * (1 + 0.62 * coupling.lambda);

      const lambdaMuRatio = coupling.muStar > 0 ? coupling.lambda / coupling.muStar : Infinity;
      const isDefaultMuStar = coupling.muStar === 0.1 || coupling.muStar === 0.13;

      if (Math.abs(denom) > 1e-6 && denom > 0 && coupling.lambda > 0.2) {
        if (lambdaMuRatio < 1.5) {
          strategy.confidence *= 0.3;
          strategy.estimatedAmbientTc = Math.round(strategy.estimatedAmbientTc * 0.1);
        } else {
          const lambdaBar = 2.46 * (1 + 3.8 * coupling.muStar);
          const f1 = Math.pow(1 + Math.pow(coupling.lambda / lambdaBar, 3 / 2), 1 / 3);
          const physicsTc = (omegaLogK / 1.2) * f1 * Math.exp(-1.04 * (1 + coupling.lambda) / denom);

          if (Number.isFinite(physicsTc) && physicsTc > 0) {
            if (isDefaultMuStar) {
              strategy.confidence *= 0.8;
            }
            const heuristic = strategy.estimatedAmbientTc;
            const ratio = physicsTc / Math.max(1, heuristic);
            let blended: number;
            if (ratio < 0.3) {
              blended = physicsTc;
              strategy.confidence *= 0.4;
            } else if (ratio < 0.6) {
              blended = Math.round(0.3 * heuristic + 0.7 * physicsTc);
              strategy.confidence *= 0.7;
            } else {
              blended = Math.round(0.5 * heuristic + 0.5 * physicsTc);
            }
            strategy.estimatedAmbientTc = Math.min(400, Math.max(1, blended));
          }
        }
      } else if (denom <= 0 && coupling.lambda > 0.2) {
        console.log(
          `[pressure-pathway] Super-coupled regime for ${strategy.formula}: ` +
          `λ=${coupling.lambda.toFixed(3)}, μ*=${coupling.muStar.toFixed(3)}, denom=${denom.toFixed(6)} — ` +
          `possible high-interest candidate or simulation artifact`
        );
        strategy.confidence *= 0.5;
      }

      const metallicityFactor = Math.min(1, electronic.metallicity / 0.3);
      if (metallicityFactor < 1) {
        strategy.estimatedAmbientTc = Math.round(strategy.estimatedAmbientTc * metallicityFactor);
        strategy.confidence *= 0.3 + 0.7 * metallicityFactor;
      }
    } catch {}
  }

  topStrategies.sort((a, b) => b.estimatedAmbientTc - a.estimatedAmbientTc);

  const best = topStrategies[0];
  const bestAmbientTc = best?.estimatedAmbientTc ?? 0;
  const bestAmbientFormula = best?.formula ?? formula;
  const retentionPercent = sourceTc > 0 ? Math.round((bestAmbientTc / sourceTc) * 100) : 0;

  const pressureDifficulty = 1 / (1 + Math.exp((sourcePressure - 100) / 50));
  const tcFeasibility = 1 / (1 + Math.exp(-(bestAmbientTc - 30) / 15));
  const feasibility = Math.round(pressureDifficulty * tcFeasibility * 100) / 100;

  return {
    sourceFormula: formula,
    sourceTc,
    sourcePressure,
    strategies: topStrategies,
    bestAmbientTc,
    bestAmbientFormula,
    retentionPercent,
    feasibility,
  };
}

const PATHWAY_CACHE_LIMIT = 200;
const pathwayHistory: PressurePathway[] = [];
const pathwayCache = new Map<string, PressurePathway>();

export function getPathwayForCandidate(
  formula: string,
  tc: number,
  pressure: number,
): PressurePathway {
  const key = `${formula}-${tc}-${pressure}`;
  if (pathwayCache.has(key)) return pathwayCache.get(key)!;
  const pathway = searchPressurePathways(formula, tc, pressure);
  pathwayCache.set(key, pathway);
  if (pathwayHistory.length < PATHWAY_CACHE_LIMIT) pathwayHistory.push(pathway);
  if (pathwayCache.size > PATHWAY_CACHE_LIMIT) {
    const firstKey = pathwayCache.keys().next().value;
    if (firstKey) pathwayCache.delete(firstKey);
  }
  return pathway;
}

export function getPathwayStats(): {
  totalSearched: number;
  avgRetention: number;
  bestAmbientTc: number;
  bestAmbientFormula: string;
  highRetentionCount: number;
  topPathways: { source: string; ambient: string; sourceTc: number; ambientTc: number; retention: number }[];
} {
  if (pathwayHistory.length === 0) {
    return {
      totalSearched: 0,
      avgRetention: 0,
      bestAmbientTc: 0,
      bestAmbientFormula: "",
      highRetentionCount: 0,
      topPathways: [],
    };
  }

  const avgRetention = pathwayHistory.reduce((s, p) => s + p.retentionPercent, 0) / pathwayHistory.length;
  const sorted = [...pathwayHistory].sort((a, b) => b.bestAmbientTc - a.bestAmbientTc);
  const best = sorted[0];

  return {
    totalSearched: pathwayHistory.length,
    avgRetention: Math.round(avgRetention),
    bestAmbientTc: best.bestAmbientTc,
    bestAmbientFormula: best.bestAmbientFormula,
    highRetentionCount: pathwayHistory.filter(p => p.retentionPercent > 50).length,
    topPathways: sorted.slice(0, 10).map(p => ({
      source: p.sourceFormula,
      ambient: p.bestAmbientFormula,
      sourceTc: p.sourceTc,
      ambientTc: p.bestAmbientTc,
      retention: p.retentionPercent,
    })),
  };
}

export function getAmbientCandidatesFromPathways(): PathwayCandidate[] {
  const candidates: PathwayCandidate[] = [];

  for (const pathway of pathwayHistory) {
    for (const strategy of pathway.strategies) {
      if (strategy.estimatedAmbientTc > 20 && strategy.confidence > 0.3) {
        candidates.push({
          formula: strategy.formula,
          strategy: strategy.type,
          estimatedTc: strategy.estimatedAmbientTc,
          ambientStabilityScore: strategy.stabilityGain,
        });
      }
    }
  }

  candidates.sort((a, b) => b.estimatedTc - a.estimatedTc);
  return candidates.slice(0, 50);
}
