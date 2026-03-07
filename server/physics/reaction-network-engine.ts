import {
  computeMiedemaFormationEnergy,
  assessMetastability,
  computeConvexHull,
  type ConvexHullResult,
  type MetastabilityAssessment,
} from "../learning/phase-diagram-engine";
import { ELEMENTAL_DATA, getElementData } from "../learning/elemental-data";

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
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
  reactionStabilityScore: number;
  metastableLifetime: string;
  decompositionComplexity: number;
  decompositionPathways: DecompositionPathway[];
  reactionGraph: {
    nodes: ReactionNode[];
    edges: ReactionEdge[];
  };
  metastabilityAssessment: MetastabilityAssessment;
  stabilityVerdict: string;
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

function estimateReactionBarrier(
  deltaE: number,
  elements: string[],
  counts: Record<string, number>
): number {
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

  const structuralComplexity = Math.log(Math.max(2, elements.length)) / Math.log(6);

  const kineticBase = Math.abs(deltaE) > 0
    ? Math.max(0.05, Math.abs(deltaE) * 10 + 0.2)
    : 1.5;

  const thermalFactor = 0.3 * Math.log(Math.max(avgMeltingPoint, 300) / 300);

  const complexityBonus = structuralComplexity * 0.4;

  return Math.round((kineticBase + thermalFactor + complexityBonus) * 1000) / 1000;
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

function computeDecompositionProbability(barrier: number, temperatureK: number = 300): number {
  const kB = 8.617e-5;
  const attemptFreq = 1e13;
  const rate = attemptFreq * Math.exp(-barrier / (kB * temperatureK));
  const lifetimeS = 1 / Math.max(rate, 1e-100);
  const oneYearS = 3.15e7;
  if (lifetimeS > 1e20 * oneYearS) return 0.0;
  if (lifetimeS < 1) return 1.0;
  const prob = 1 - Math.exp(-oneYearS / lifetimeS);
  return Math.round(prob * 10000) / 10000;
}

export function analyzeReactionNetwork(formula: string): ReactionNetworkResult {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  const formationEnergy = computeMiedemaFormationEnergy(formula);

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
      const energy = computeMiedemaFormationEnergy(bin);
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
      const energy = computeMiedemaFormationEnergy(tern);
      competingFormulas.push({ formula: tern, energy });
      nodes.push({
        formula: tern,
        formationEnergy: energy,
        isStable: energy < 0,
      });
    }
  }

  const hullResult = computeConvexHull(formula, elements, competingFormulas);
  const energyAboveHull = hullResult.energyAboveHull;

  const stableCompetitors = competingFormulas
    .filter(c => c.energy < 0)
    .sort((a, b) => a.energy - b.energy);

  if (elements.length >= 2) {
    const elementalEnergy = 0 - formationEnergy;
    const barrier = estimateReactionBarrier(elementalEnergy, elements, counts);
    const prob = computeDecompositionProbability(barrier);
    pathways.push({
      products: [...elements],
      reactionEnergy: Math.round(elementalEnergy * 10000) / 10000,
      barrier,
      probability: prob,
      mechanism: "elemental-decomposition",
    });
    edges.push({
      from: formula,
      to: [...elements],
      energy: elementalEnergy,
      barrier,
      type: "elemental-decomposition",
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
    const barrier = estimateReactionBarrier(Math.abs(reactionEnergy), elements, counts);
    const prob = computeDecompositionProbability(barrier);
    const mechanism = classifyDecompositionMechanism(elements, products);

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

  for (let i = 0; i < stableCompetitors.length && i < 5; i++) {
    for (let j = i + 1; j < stableCompetitors.length && j < 5; j++) {
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
          const barrier = estimateReactionBarrier(Math.abs(reactionEnergy), elements, counts);
          const prob = computeDecompositionProbability(barrier);
          const mechanism = classifyDecompositionMechanism(elements, products);
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

  pathways.sort((a, b) => a.reactionEnergy - b.reactionEnergy);

  const metastability = assessMetastability(formula, energyAboveHull);

  const decompositionComplexity = computeDecompositionComplexity(elements, pathways);

  const reactionStabilityScore = computeReactionStabilityScore(
    formationEnergy,
    energyAboveHull,
    pathways,
    decompositionComplexity
  );

  let stabilityVerdict: string;
  if (energyAboveHull <= 0.005) {
    stabilityVerdict = "thermodynamically stable (on convex hull)";
  } else if (energyAboveHull <= 0.05) {
    stabilityVerdict = "near-hull, likely synthesizable";
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
    reactionStabilityScore,
    metastableLifetime: metastability.estimatedLifetime,
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
  pathways: DecompositionPathway[]
): number {
  if (pathways.length === 0) return 0;

  const nElements = elements.length;
  const nPathways = pathways.length;
  const avgProducts = pathways.reduce((s, p) => s + p.products.length, 0) / nPathways;
  const mechanismTypes = new Set(pathways.map(p => p.mechanism)).size;

  const score = Math.min(1.0,
    (nElements / 6) * 0.25 +
    (nPathways / 10) * 0.25 +
    (avgProducts / 4) * 0.25 +
    (mechanismTypes / 5) * 0.25
  );

  return Math.round(score * 1000) / 1000;
}

function computeReactionStabilityScore(
  formationEnergy: number,
  energyAboveHull: number,
  pathways: DecompositionPathway[],
  decompositionComplexity: number
): number {
  let score = 1.0;

  if (formationEnergy > 0) {
    score -= Math.min(0.4, formationEnergy * 2);
  } else {
    score += Math.min(0.1, Math.abs(formationEnergy) * 0.5);
  }

  score -= Math.min(0.3, energyAboveHull * 3);

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
