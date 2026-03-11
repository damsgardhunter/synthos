import {
  computeMiedemaFormationEnergy,
  assessMetastability,
  computeConvexHull,
  type ConvexHullResult,
  type MetastabilityAssessment,
} from "../learning/phase-diagram-engine";
import { ELEMENTAL_DATA, getElementData } from "../learning/elemental-data";

function expandParentheses(formula: string): string {
  let result = formula.replace(/\[/g, "(").replace(/\]/g, ")");
  const parenRegex = /\(([^()]+)\)(\d*\.?\d*)/;
  let iterations = 0;
  while (result.includes("(") && iterations < 20) {
    const prev = result;
    result = result.replace(parenRegex, (_, group: string, mult: string) => {
      const m = mult ? parseFloat(mult) : 1;
      if (isNaN(m) || m <= 0) return group;
      if (m === 1) return group;
      return group.replace(/([A-Z][a-z]?)(\d*\.?\d*)/g, (_m: string, el: string, num: string) => {
        const n = num ? parseFloat(num) : 1;
        const newN = (isNaN(n) || n <= 0 ? 1 : n) * m;
        return newN === 1 ? el : `${el}${newN}`;
      });
    });
    if (result === prev) break;
    iterations++;
  }
  return result.replace(/[()]/g, "");
}

function normalizeFormulaString(formula: string): string {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const subscriptMap: Record<string, string> = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
  };
  const superscriptMap: Record<string, string> = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
  };
  let cleaned = formula;
  for (const [sub, digit] of Object.entries(subscriptMap)) {
    cleaned = cleaned.split(sub).join(digit);
  }
  for (const [sup, digit] of Object.entries(superscriptMap)) {
    cleaned = cleaned.split(sup).join(digit);
  }
  cleaned = cleaned.replace(/[^\x20-\x7E]/g, "");
  cleaned = expandParentheses(cleaned.trim());
  return cleaned;
}

function parseFormulaElements(formula: string): string[] {
  const cleaned = normalizeFormulaString(formula);
  const matches = cleaned.match(/[A-Z][a-z]?/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = normalizeFormulaString(formula);
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (isNaN(num) || num <= 0) {
      counts[el] = (counts[el] || 0) + 1;
    } else {
      counts[el] = (counts[el] || 0) + num;
    }
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

export interface DecompositionPathway {
  products: string[];
  reactionEnergy: number;
  barrier: number;
  probability: number;
  mechanism: string;
}

export interface ReactionNode {
  formula: string;
  formationEnergy: number;
  isStable: boolean;
}

export interface ReactionEdge {
  from: string;
  to: string[];
  energy: number;
  barrier: number;
  type: string;
}

export interface ReactionNetworkResult {
  formula: string;
  formationEnergy: number;
  energyAboveHull: number;
  hullApproximationNoise: number;
  reactionStabilityScore: number;
  metastableLifetime: string;
  metastableLifetimeLog10s: number;
  decompositionComplexity: number;
  decompositionPathways: DecompositionPathway[];
  reactionGraph: {
    nodes: ReactionNode[];
    edges: ReactionEdge[];
  };
  metastabilityAssessment: MetastabilityAssessment;
  stabilityVerdict: string;
}

function canonicalizeFormula(formula: string): string {
  const counts = parseFormulaCounts(formula);
  const sorted = Object.keys(counts).sort();
  return sorted.map(el => {
    const n = counts[el];
    return n === 1 ? el : `${el}${n}`;
  }).join("");
}

function generateBinaryFormulas(elements: string[]): string[] {
  const binaries: string[] = [];
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const a = elements[i];
      const b = elements[j];
      binaries.push(`${a}${b}`);
      binaries.push(`${a}${b}2`);
      binaries.push(`${a}2${b}`);
      binaries.push(`${a}${b}3`);
      binaries.push(`${a}3${b}`);
    }
  }
  return binaries;
}

function generateTernaryFormulas(elements: string[]): string[] {
  const ternaries: string[] = [];
  if (elements.length < 3) return ternaries;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      for (let k = j + 1; k < elements.length; k++) {
        ternaries.push(`${elements[i]}${elements[j]}${elements[k]}`);
        ternaries.push(`${elements[i]}${elements[j]}2${elements[k]}`);
        ternaries.push(`${elements[i]}${elements[j]}${elements[k]}2`);
      }
    }
  }
  return ternaries;
}

function murnaghanVolume(P: number, B0: number, Bp: number): number {
  if (B0 <= 0 || P <= 0) return 1.0;
  return Math.pow(1 + Bp * P / B0, -1 / Bp);
}

function pressureCorrectedRadius(el: string, r0: number, pressureGpa: number): number {
  if (pressureGpa <= 100) return r0;
  if (el === "H") {
    const excessP = pressureGpa - 100;
    const collapseScale = 1.0 / (1.0 + excessP * 0.003 + Math.pow(excessP / 300, 2) * 0.5);
    return r0 * Math.max(0.35, collapseScale);
  }
  const excessP = pressureGpa - 100;
  const scale = 1.0 / (1.0 + excessP * 0.001);
  return r0 * Math.max(0.55, scale);
}

function estimatePVCorrection(
  elements: string[],
  counts: Record<string, number>,
  pressureGpa: number
): number {
  if (pressureGpa <= 0) return 0;

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  let avgB0 = 0;
  let avgVolume = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const frac = (counts[el] || 1) / totalAtoms;
    const r0 = data ? data.atomicRadius : 1.5;
    const r = pressureCorrectedRadius(el, r0, pressureGpa);
    const atomicVolume = (4 / 3) * Math.PI * Math.pow(r, 3);
    avgVolume += atomicVolume * frac;
    avgB0 += (data?.bulkModulus ?? 50) * frac;
  }

  const Bp = 4.0;
  const compressionFactor = murnaghanVolume(pressureGpa, Math.max(10, avgB0), Bp);
  const effectiveVolume = avgVolume * compressionFactor;

  const pvTerm = pressureGpa * effectiveVolume * 1e-4;

  return pvTerm;
}

function estimateEntropyCorrection(
  elements: string[],
  counts: Record<string, number>,
  temperatureK: number = 300
): number {
  const nElements = elements.length;
  if (nElements <= 1) return 0;

  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  let mixingEntropy = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    if (frac > 0) {
      mixingEntropy -= frac * Math.log(frac);
    }
  }

  const kB = 8.617e-5;
  return -temperatureK * kB * mixingEntropy;
}

const BEP_ALPHA: Record<string, number> = {
  "elemental-decomposition": 0.75,
  "oxidative-decomposition": 0.65,
  "dehydrogenation": 0.80,
  "hydrogen-redistribution": 0.70,
  "binary-disproportionation": 0.60,
  "multi-phase-decomposition": 0.55,
  "phase-separation": 0.65,
};

const BEP_E0: Record<string, number> = {
  "elemental-decomposition": 0.5,
  "oxidative-decomposition": 0.3,
  "dehydrogenation": 0.6,
  "hydrogen-redistribution": 0.4,
  "binary-disproportionation": 0.35,
  "multi-phase-decomposition": 0.25,
  "phase-separation": 0.4,
};

function estimateReactionBarrier(
  deltaE: number,
  elements: string[],
  counts: Record<string, number>,
  pressureGpa: number = 0,
  mechanism: string = "phase-separation"
): number {
  const alpha = BEP_ALPHA[mechanism] ?? 0.65;
  const e0 = BEP_E0[mechanism] ?? 0.4;
  const bepBarrier = Math.max(0.02, e0 + alpha * Math.abs(deltaE));

  let avgMeltingPoint = 0;
  let count = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.meltingPoint) {
      avgMeltingPoint += data.meltingPoint * (counts[el] || 1);
      count += (counts[el] || 1);
    }
  }
  avgMeltingPoint = count > 0 ? avgMeltingPoint / count : 1000;

  const thermalFactor = 0.15 * Math.log(Math.max(avgMeltingPoint, 300) / 300);

  const structuralComplexity = Math.log(Math.max(2, elements.length)) / Math.log(6);
  const complexityBonus = structuralComplexity * 0.2;

  let pressureBarrierBoost = 0;
  if (pressureGpa > 0) {
    if (pressureGpa <= 150) {
      pressureBarrierBoost = 0.1 * Math.log(1 + pressureGpa / 50);
    } else {
      const logBase = 0.1 * Math.log(1 + 150 / 50);
      const excessP = pressureGpa - 150;
      pressureBarrierBoost = logBase + 0.002 * excessP + 3e-6 * excessP * excessP;
    }

    const hCount = counts["H"] || 0;
    const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    const hFrac = hCount / totalAtoms;
    if (hFrac > 0.5 && pressureGpa > 100) {
      const cagePressure = pressureGpa - 100;
      const cageBoost = 0.15 * Math.pow(cagePressure / 100, 1.5) * Math.min(2.0, hFrac);
      pressureBarrierBoost += Math.min(1.5, cageBoost);
    }
  }

  return Math.round((bepBarrier + thermalFactor + complexityBonus + pressureBarrierBoost) * 1000) / 1000;
}

function classifyDecompositionMechanism(
  parentElements: string[],
  productFormulas: string[]
): string {
  const productElementSets = productFormulas.map(f => parseFormulaElements(f));

  const allSingle = productFormulas.every(f => {
    const els = parseFormulaElements(f);
    return els.length === 1;
  });
  if (allSingle) return "elemental-decomposition";

  const hasOxygen = parentElements.includes("O");
  const anyProductHasO = productElementSets.some(s => s.includes("O"));
  if (hasOxygen && anyProductHasO) return "oxidative-decomposition";

  const hasH = parentElements.includes("H");
  const anyProductHasH = productElementSets.some(s => s.includes("H"));
  if (hasH && !anyProductHasH) return "dehydrogenation";
  if (hasH && anyProductHasH) return "hydrogen-redistribution";

  if (productFormulas.length === 2) return "binary-disproportionation";
  if (productFormulas.length >= 3) return "multi-phase-decomposition";

  return "phase-separation";
}

function computeDecompositionProbability(barrier: number, temperatureK: number = 300): { probability: number; lifetimeLog10s: number } {
  const kB = 8.617e-5;
  const attemptFreq = 1e13;
  const rate = attemptFreq * Math.exp(-barrier / (kB * temperatureK));
  const lifetimeS = 1 / Math.max(rate, 1e-100);
  const lifetimeLog10s = Math.log10(Math.max(lifetimeS, 1e-30));
  const oneYearS = 3.15e7;
  if (lifetimeS > 1e20 * oneYearS) return { probability: 0.0, lifetimeLog10s: Math.min(lifetimeLog10s, 40) };
  if (lifetimeS < 1) return { probability: 1.0, lifetimeLog10s };
  const prob = 1 - Math.exp(-oneYearS / lifetimeS);
  return { probability: Math.round(prob * 10000) / 10000, lifetimeLog10s: Math.round(lifetimeLog10s * 100) / 100 };
}

export function analyzeReactionNetwork(formula: string, pressureGpa: number = 0, temperatureK: number = 300): ReactionNetworkResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const formationEnergyRaw = computeMiedemaFormationEnergy(formula);

  const pvCorrection = estimatePVCorrection(elements, counts, pressureGpa);
  const entropyCorrection = estimateEntropyCorrection(elements, counts, temperatureK);
  const formationEnergy = formationEnergyRaw + pvCorrection + entropyCorrection;

  const nodes: ReactionNode[] = [];
  const edges: ReactionEdge[] = [];
  const pathways: DecompositionPathway[] = [];

  nodes.push({
    formula,
    formationEnergy,
    isStable: formationEnergy <= 0,
  });

  for (const el of elements) {
    nodes.push({
      formula: el,
      formationEnergy: 0,
      isStable: true,
    });
  }

  const competingFormulas: { formula: string; energy: number }[] = [];

  const binaries = generateBinaryFormulas(elements);
  for (const bin of binaries) {
    const binEls = parseFormulaElements(bin);
    if (binEls.every(e => elements.includes(e))) {
      const binCounts = parseFormulaCounts(bin);
      const rawEnergy = computeMiedemaFormationEnergy(bin);
      const binPV = estimatePVCorrection(binEls, binCounts, pressureGpa);
      const binEntropy = estimateEntropyCorrection(binEls, binCounts, temperatureK);
      const energy = rawEnergy + binPV + binEntropy;
      competingFormulas.push({ formula: bin, energy });
      nodes.push({
        formula: bin,
        formationEnergy: energy,
        isStable: energy < 0,
      });
    }
  }

  const ternaries = generateTernaryFormulas(elements);
  for (const tern of ternaries) {
    const ternEls = parseFormulaElements(tern);
    if (ternEls.every(e => elements.includes(e))) {
      const ternCounts = parseFormulaCounts(tern);
      const rawEnergy = computeMiedemaFormationEnergy(tern);
      const ternPV = estimatePVCorrection(ternEls, ternCounts, pressureGpa);
      const ternEntropy = estimateEntropyCorrection(ternEls, ternCounts, temperatureK);
      const energy = rawEnergy + ternPV + ternEntropy;
      competingFormulas.push({ formula: tern, energy });
      nodes.push({
        formula: tern,
        formationEnergy: energy,
        isStable: energy < 0,
      });
    }
  }

  const canonicalFormula = canonicalizeFormula(formula);
  const canonicalizedCompeting = competingFormulas.map(c => ({
    formula: canonicalizeFormula(c.formula),
    energy: c.energy,
  }));
  const hullResult = computeConvexHull(canonicalFormula, elements, canonicalizedCompeting);
  const energyAboveHull = hullResult.energyAboveHull;

  const stableCompetitors = competingFormulas
    .filter(c => c.energy < 0)
    .sort((a, b) => a.energy - b.energy);

  let globalMinLifetimeLog10s = 40;

  if (elements.length >= 2) {
    const elementalEnergy = 0 - formationEnergy;
    const mechType = "elemental-decomposition";
    const barrier = estimateReactionBarrier(elementalEnergy, elements, counts, pressureGpa, mechType);
    const { probability: prob, lifetimeLog10s } = computeDecompositionProbability(barrier);
    globalMinLifetimeLog10s = Math.min(globalMinLifetimeLog10s, lifetimeLog10s);
    pathways.push({
      products: [...elements],
      reactionEnergy: Math.round(elementalEnergy * 10000) / 10000,
      barrier,
      probability: prob,
      mechanism: mechType,
    });
    edges.push({
      from: formula,
      to: [...elements],
      energy: elementalEnergy,
      barrier,
      type: mechType,
    });
  }

  const seenProductSets = new Set<string>();
  for (const comp of stableCompetitors.slice(0, 8)) {
    const compEls = parseFormulaElements(comp.formula);
    const remainingEls = elements.filter(e => !compEls.includes(e));

    const products = [comp.formula];
    if (remainingEls.length > 0) {
      products.push(...remainingEls);
    }

    const key = [...products].sort().join("+");
    if (seenProductSets.has(key)) continue;
    seenProductSets.add(key);

    let productTotalEnergy = comp.energy;
    const reactionEnergy = productTotalEnergy - formationEnergy;
    const mechanism = classifyDecompositionMechanism(elements, products);
    const barrier = estimateReactionBarrier(Math.abs(reactionEnergy), elements, counts, pressureGpa, mechanism);
    const { probability: prob, lifetimeLog10s } = computeDecompositionProbability(barrier);
    globalMinLifetimeLog10s = Math.min(globalMinLifetimeLog10s, lifetimeLog10s);

    pathways.push({
      products,
      reactionEnergy: Math.round(reactionEnergy * 10000) / 10000,
      barrier,
      probability: prob,
      mechanism,
    });
    edges.push({
      from: formula,
      to: products,
      energy: reactionEnergy,
      barrier,
      type: mechanism,
    });
  }

  const pairLimit = Math.min(stableCompetitors.length, 5);
  for (let i = 0; i < pairLimit; i++) {
    for (let j = i + 1; j < pairLimit; j++) {
      const compA = stableCompetitors[i];
      const compB = stableCompetitors[j];
      const elsA = parseFormulaElements(compA.formula);
      const elsB = parseFormulaElements(compB.formula);
      const combined = new Set([...elsA, ...elsB]);
      if (elements.every(e => combined.has(e))) {
        const products = [compA.formula, compB.formula];
        const key = [...products].sort().join("+");
        if (seenProductSets.has(key)) continue;
        seenProductSets.add(key);

        const productEnergy = compA.energy + compB.energy;
        const reactionEnergy = productEnergy - formationEnergy;
        if (reactionEnergy < 0) {
          const mechanism = classifyDecompositionMechanism(elements, products);
          const barrier = estimateReactionBarrier(Math.abs(reactionEnergy), elements, counts, pressureGpa, mechanism);
          const { probability: prob, lifetimeLog10s } = computeDecompositionProbability(barrier);
          globalMinLifetimeLog10s = Math.min(globalMinLifetimeLog10s, lifetimeLog10s);
          pathways.push({
            products,
            reactionEnergy: Math.round(reactionEnergy * 10000) / 10000,
            barrier,
            probability: prob,
            mechanism,
          });
          edges.push({
            from: formula,
            to: products,
            energy: reactionEnergy,
            barrier,
            type: mechanism,
          });
        }
      }
    }
  }

  if (elements.length >= 4) {
    const tripletLimit = Math.min(stableCompetitors.length, 4);
    for (let i = 0; i < tripletLimit; i++) {
      for (let j = i + 1; j < tripletLimit; j++) {
        for (let k = j + 1; k < tripletLimit; k++) {
          const compA = stableCompetitors[i];
          const compB = stableCompetitors[j];
          const compC = stableCompetitors[k];
          const elsA = parseFormulaElements(compA.formula);
          const elsB = parseFormulaElements(compB.formula);
          const elsC = parseFormulaElements(compC.formula);
          const combined = new Set([...elsA, ...elsB, ...elsC]);
          if (elements.every(e => combined.has(e))) {
            const products = [compA.formula, compB.formula, compC.formula];
            const key = [...products].sort().join("+");
            if (seenProductSets.has(key)) continue;
            seenProductSets.add(key);

            const productEnergy = compA.energy + compB.energy + compC.energy;
            const reactionEnergy = productEnergy - formationEnergy;
            if (reactionEnergy < 0) {
              const mechanism = classifyDecompositionMechanism(elements, products);
              const barrier = estimateReactionBarrier(Math.abs(reactionEnergy), elements, counts, pressureGpa, mechanism);
              const { probability: prob, lifetimeLog10s } = computeDecompositionProbability(barrier);
              globalMinLifetimeLog10s = Math.min(globalMinLifetimeLog10s, lifetimeLog10s);
              pathways.push({
                products,
                reactionEnergy: Math.round(reactionEnergy * 10000) / 10000,
                barrier,
                probability: prob,
                mechanism,
              });
              edges.push({
                from: formula,
                to: products,
                energy: reactionEnergy,
                barrier,
                type: mechanism,
              });
            }
          }
        }
      }
    }
  }

  pathways.sort((a, b) => a.reactionEnergy - b.reactionEnergy);

  const metastability = assessMetastability(formula, energyAboveHull);

  const decompositionComplexity = computeDecompositionComplexity(elements, pathways, edges);

  const hullNoiseEstimate = estimateHullApproximationNoise(elements, competingFormulas.length);

  const reactionStabilityScore = computeReactionStabilityScore(
    formationEnergy,
    energyAboveHull,
    pathways,
    decompositionComplexity,
    hullNoiseEstimate
  );

  let stabilityVerdict: string;
  const hullConfidenceTag = hullNoiseEstimate > 0.03
    ? " (approx. hull, confidence reduced)"
    : "";
  if (energyAboveHull <= 0.005) {
    if (hullNoiseEstimate > 0.02) {
      stabilityVerdict = "near-hull within sampling uncertainty";
    } else {
      stabilityVerdict = "thermodynamically stable (on convex hull)";
    }
  } else if (energyAboveHull <= 0.05) {
    stabilityVerdict = "near-hull, likely synthesizable" + hullConfidenceTag;
  } else if (metastability.isMetastable) {
    stabilityVerdict = `metastable (barrier=${metastability.kineticBarrier.toFixed(3)} eV, lifetime=${metastability.estimatedLifetime})`;
  } else if (energyAboveHull <= 0.2) {
    stabilityVerdict = "marginally unstable, may be metastable under specific conditions";
  } else {
    stabilityVerdict = "thermodynamically unstable";
  }

  return {
    formula,
    formationEnergy: Math.round(formationEnergy * 10000) / 10000,
    energyAboveHull: Math.round(energyAboveHull * 10000) / 10000,
    hullApproximationNoise: Math.round(hullNoiseEstimate * 10000) / 10000,
    reactionStabilityScore,
    metastableLifetime: metastability.estimatedLifetime,
    metastableLifetimeLog10s: Math.round(globalMinLifetimeLog10s * 100) / 100,
    decompositionComplexity,
    decompositionPathways: pathways.slice(0, 10),
    reactionGraph: {
      nodes: deduplicateNodes(nodes),
      edges,
    },
    metastabilityAssessment: metastability,
    stabilityVerdict,
  };
}

function deduplicateNodes(nodes: ReactionNode[]): ReactionNode[] {
  const seen = new Set<string>();
  const result: ReactionNode[] = [];
  for (const node of nodes) {
    if (!seen.has(node.formula)) {
      seen.add(node.formula);
      result.push(node);
    }
  }
  return result;
}

function computeDecompositionComplexity(
  elements: string[],
  pathways: DecompositionPathway[],
  edges: ReactionEdge[]
): number {
  if (pathways.length === 0) return 0;

  const nPathways = pathways.length;
  const stepCount = edges.length;

  const allProducts = new Set<string>();
  for (const p of pathways) {
    for (const prod of p.products) {
      allProducts.add(prod);
    }
  }
  const distinctPhases = allProducts.size;

  const avgProducts = pathways.reduce((s, p) => s + p.products.length, 0) / nPathways;
  const mechanismTypes = new Set(pathways.map(p => p.mechanism)).size;

  const maxBarrier = Math.max(...pathways.map(p => p.barrier));
  const minBarrier = Math.min(...pathways.map(p => p.barrier));
  const barrierSpread = maxBarrier > 0 ? (maxBarrier - minBarrier) / maxBarrier : 0;

  const score = Math.min(1.0,
    (stepCount / 15) * 0.20 +
    (distinctPhases / 10) * 0.20 +
    (avgProducts / 4) * 0.15 +
    (mechanismTypes / 5) * 0.15 +
    (nPathways / 10) * 0.15 +
    barrierSpread * 0.15
  );

  return Math.round(score * 1000) / 1000;
}

function estimateHullApproximationNoise(elements: string[], competingPhaseCount: number): number {
  const nElem = elements.length;
  const stoichCombinationsInFullHull = nElem <= 2
    ? 5
    : nElem === 3
      ? 30
      : nElem === 4
        ? 150
        : Math.pow(nElem, 3) * 2;
  const sampledFraction = Math.min(1.0, competingPhaseCount / stoichCombinationsInFullHull);
  const missingPhasePenalty = (1 - sampledFraction) * 0.05 * Math.sqrt(nElem);
  const baseNoise = nElem <= 2 ? 0.005 : nElem === 3 ? 0.015 : nElem === 4 ? 0.03 : 0.05;
  return Math.round(Math.min(0.15, baseNoise + missingPhasePenalty) * 10000) / 10000;
}

function computeReactionStabilityScore(
  formationEnergy: number,
  energyAboveHull: number,
  pathways: DecompositionPathway[],
  decompositionComplexity: number,
  hullNoise: number = 0
): number {
  let score = 1.0;

  if (formationEnergy > 0) {
    score -= Math.min(0.4, formationEnergy * 2);
  } else {
    score += Math.min(0.1, Math.abs(formationEnergy) * 0.5);
  }

  score -= Math.min(0.3, energyAboveHull * 3);

  if (energyAboveHull < hullNoise * 2) {
    const uncertaintyPenalty = Math.min(0.1, hullNoise * 1.5);
    score -= uncertaintyPenalty;
  }

  if (pathways.length > 0) {
    const minBarrier = Math.min(...pathways.map(p => p.barrier));
    if (minBarrier < 0.3) {
      score -= 0.2 * (1 - minBarrier / 0.3);
    }
    const maxProb = Math.max(...pathways.map(p => p.probability));
    score -= maxProb * 0.2;
  }

  score += decompositionComplexity * 0.1;

  return Math.round(Math.max(0, Math.min(1, score)) * 1000) / 1000;
}
