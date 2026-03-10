import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";
import {
  getElementData,
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

function buildFormula(counts: Record<string, number>): string {
  return Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([el, n]) => n === 1 ? el : `${el}${n}`)
    .join("");
}

const CHEMICAL_PRESSURE_ELEMENTS: Record<string, { smaller: string[]; larger: string[] }> = {
  La: { smaller: ["Y", "Sc", "Lu"], larger: ["Ce", "Pr", "Nd"] },
  Y: { smaller: ["Sc", "Lu"], larger: ["La", "Gd"] },
  Ca: { smaller: ["Mg", "Be"], larger: ["Sr", "Ba"] },
  Sr: { smaller: ["Ca", "Mg"], larger: ["Ba"] },
  Ba: { smaller: ["Sr", "Ca"], larger: [] },
  Nb: { smaller: ["V", "Mo"], larger: ["Ta"] },
  Ti: { smaller: ["V", "Cr"], larger: ["Zr", "Hf"] },
  Zr: { smaller: ["Ti"], larger: ["Hf"] },
  Th: { smaller: ["U", "Ce"], larger: [] },
  Ce: { smaller: ["Pr", "Nd"], larger: ["La"] },
};

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

function classifyMaterial(formula: string): "hydride" | "cage" | "layered" | "conventional" {
  const counts = parseFormulaCounts(formula);
  const hCount = counts["H"] || 0;
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);

  if (hCount / totalAtoms > 0.5) return "hydride";
  if (hCount > 0 && hCount / totalAtoms > 0.3) return "cage";

  const layeredElements = ["Cu", "Fe", "Ni", "Bi", "Tl"];
  const hasLayered = Object.keys(counts).some(el => layeredElements.includes(el));
  if (hasLayered && Object.keys(counts).length >= 3) return "layered";

  return "conventional";
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

  const origRadius = origData.atomicRadius ?? 150;
  const subRadius = subData.atomicRadius ?? 150;
  const radiusDiff = (subRadius - origRadius) / origRadius;

  const pressureEquivalent = radiusDiff * originalPressure * 0.3;
  const pressureReduction = Math.max(0, Math.min(originalPressure * 0.8, pressureEquivalent));

  const massDiff = Math.abs((subData.atomicMass ?? 100) - (origData.atomicMass ?? 100)) / (origData.atomicMass ?? 100);
  const electronDiff = Math.abs((subData.paulingElectronegativity ?? 1.5) - (origData.paulingElectronegativity ?? 1.5));
  const tcRetention = Math.max(0.1, 1 - 0.3 * massDiff - 0.2 * electronDiff - 0.1 * Math.abs(radiusDiff));

  return { pressureReduction: Math.round(pressureReduction * 10) / 10, tcRetention: Math.round(tcRetention * 100) / 100 };
}

function generateIsovalentSubstitutions(formula: string, sourcePressure: number): StabilizationStrategy[] {
  const counts = parseFormulaCounts(formula);
  const strategies: StabilizationStrategy[] = [];

  for (const [el, amount] of Object.entries(counts)) {
    if (el === "H") continue;
    const chemPressure = CHEMICAL_PRESSURE_ELEMENTS[el];
    if (!chemPressure) continue;

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
  const matType = classifyMaterial(formula);
  const strategies: StabilizationStrategy[] = [];

  const dopants = STABILIZING_DOPANTS[matType] ?? STABILIZING_DOPANTS["conventional"] ?? [];

  for (const { dopant, fraction, reason } of dopants) {
    if (counts[dopant] && counts[dopant] > 0.5) continue;

    const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
    const dopantAmount = Math.max(0.5, Math.round(totalAtoms * fraction * 10) / 10);

    const newCounts = { ...counts };
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

    const massPenalty = replaceCount / hCount;

    strategies.push({
      type: "anion-substitution",
      formula: newFormula,
      description: `Replace ${replaceCount}H with ${anion}: ${reason}`,
      estimatedAmbientTc: 0,
      tcRetention: Math.max(0.2, 1 - massPenalty * 0.8),
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

      if (Math.abs(denom) > 1e-6 && denom > 0 && coupling.lambda > 0.2) {
        const lambdaBar = 2.46 * (1 + 3.8 * coupling.muStar);
        const f1 = Math.pow(1 + Math.pow(coupling.lambda / lambdaBar, 3 / 2), 1 / 3);
        const physicsTc = (omegaLogK / 1.2) * f1 * Math.exp(-1.04 * (1 + coupling.lambda) / denom);

        if (Number.isFinite(physicsTc) && physicsTc > 0) {
          const blended = Math.round(0.6 * strategy.estimatedAmbientTc + 0.4 * physicsTc);
          strategy.estimatedAmbientTc = Math.min(400, Math.max(1, blended));
        }
      }

      if (electronic.metallicity < 0.3) {
        strategy.estimatedAmbientTc = Math.round(strategy.estimatedAmbientTc * 0.3);
        strategy.confidence *= 0.5;
      }
    } catch {}
  }

  topStrategies.sort((a, b) => b.estimatedAmbientTc - a.estimatedAmbientTc);

  const best = topStrategies[0];
  const bestAmbientTc = best?.estimatedAmbientTc ?? 0;
  const bestAmbientFormula = best?.formula ?? formula;
  const retentionPercent = sourceTc > 0 ? Math.round((bestAmbientTc / sourceTc) * 100) : 0;

  const pressureDifficulty = sourcePressure > 200 ? 0.3 : sourcePressure > 100 ? 0.5 : sourcePressure > 50 ? 0.7 : 0.9;
  const tcFeasibility = bestAmbientTc > 50 ? 0.8 : bestAmbientTc > 20 ? 0.6 : 0.3;
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
  if (pathwayHistory.length < 200) pathwayHistory.push(pathway);
  if (pathwayCache.size > 100) {
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
