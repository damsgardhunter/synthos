import { ELEMENTAL_DATA, getMeltingPoint, getElementData } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";
import { findBestPrecursors, computePrecursorAvailabilityScore } from "./precursor-database";

export interface RetrosynthesisRoute {
  type: "direct-elemental" | "binary-intermediate" | "oxide-decomposition" | "precursor-substitution";
  precursors: string[];
  product: string;
  equation: string;
  deltaE: number;
  complexity: number;
  availability: number;
  similarity: number;
  overallScore: number;
  reasoning: string[];
}

export interface RetrosynthesisResult {
  formula: string;
  family: string;
  totalRoutes: number;
  routes: RetrosynthesisRoute[];
  bestRoute: RetrosynthesisRoute | null;
  analysisNotes: string[];
}

const NONMETALS = new Set(["H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "At", "Rn", "Te", "As"]);

const COMMON_BINARIES: Record<string, { formula: string; elements: string[]; stability: number }[]> = {
  Mg: [{ formula: "MgB2", elements: ["Mg", "B"], stability: 0.85 }, { formula: "MgO", elements: ["Mg", "O"], stability: 0.95 }],
  Ba: [{ formula: "BaO", elements: ["Ba", "O"], stability: 0.9 }, { formula: "BaCO3", elements: ["Ba", "O", "C"], stability: 0.95 }],
  Y: [{ formula: "Y2O3", elements: ["Y", "O"], stability: 0.95 }],
  La: [{ formula: "La2O3", elements: ["La", "O"], stability: 0.95 }],
  Cu: [{ formula: "CuO", elements: ["Cu", "O"], stability: 0.9 }, { formula: "Cu2O", elements: ["Cu", "O"], stability: 0.85 }],
  Fe: [{ formula: "Fe2O3", elements: ["Fe", "O"], stability: 0.9 }, { formula: "FeAs", elements: ["Fe", "As"], stability: 0.75 }, { formula: "FeSe", elements: ["Fe", "Se"], stability: 0.8 }],
  Nb: [{ formula: "Nb2O5", elements: ["Nb", "O"], stability: 0.9 }, { formula: "NbN", elements: ["Nb", "N"], stability: 0.85 }, { formula: "Nb3Sn", elements: ["Nb", "Sn"], stability: 0.8 }],
  Ti: [{ formula: "TiO2", elements: ["Ti", "O"], stability: 0.95 }, { formula: "TiN", elements: ["Ti", "N"], stability: 0.85 }],
  Sr: [{ formula: "SrO", elements: ["Sr", "O"], stability: 0.9 }, { formula: "SrCO3", elements: ["Sr", "O", "C"], stability: 0.95 }],
  Ca: [{ formula: "CaO", elements: ["Ca", "O"], stability: 0.95 }, { formula: "CaH2", elements: ["Ca", "H"], stability: 0.8 }],
  Al: [{ formula: "Al2O3", elements: ["Al", "O"], stability: 0.95 }],
  Sn: [{ formula: "SnO2", elements: ["Sn", "O"], stability: 0.9 }],
  V: [{ formula: "V2O5", elements: ["V", "O"], stability: 0.9 }],
  Zr: [{ formula: "ZrO2", elements: ["Zr", "O"], stability: 0.95 }],
  B: [{ formula: "B2O3", elements: ["B", "O"], stability: 0.9 }],
  Bi: [{ formula: "Bi2O3", elements: ["Bi", "O"], stability: 0.9 }],
  Pb: [{ formula: "PbO", elements: ["Pb", "O"], stability: 0.85 }],
};

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  return parseNestedFormula(cleaned);
}

function parseNestedFormula(s: string): Record<string, number> {
  const counts: Record<string, number> = {};
  let i = 0;
  while (i < s.length) {
    if (s[i] === '(') {
      let depth = 1;
      let j = i + 1;
      while (j < s.length && depth > 0) {
        if (s[j] === '(') depth++;
        else if (s[j] === ')') depth--;
        j++;
      }
      const inner = parseNestedFormula(s.substring(i + 1, j - 1));
      let numStr = '';
      while (j < s.length && (s[j] >= '0' && s[j] <= '9' || s[j] === '.')) {
        numStr += s[j]; j++;
      }
      const mult = numStr ? parseFloat(numStr) : 1;
      for (const [el, cnt] of Object.entries(inner)) {
        counts[el] = (counts[el] || 0) + cnt * mult;
      }
      i = j;
    } else if (s[i] >= 'A' && s[i] <= 'Z') {
      let el = s[i]; i++;
      while (i < s.length && s[i] >= 'a' && s[i] <= 'z') { el += s[i]; i++; }
      let numStr = '';
      while (i < s.length && (s[i] >= '0' && s[i] <= '9' || s[i] === '.')) { numStr += s[i]; i++; }
      const num = numStr ? parseFloat(numStr) : 1;
      counts[el] = (counts[el] || 0) + num;
    } else { i++; }
  }
  return counts;
}

function parseFormulaElements(formula: string): string[] {
  return Object.keys(parseFormulaCounts(formula));
}

function computeMiedemaFromCounts(counts: Record<string, number>): number {
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

  let deltaH = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const dA = ELEMENTAL_DATA[elements[i]];
      const dB = ELEMENTAL_DATA[elements[j]];
      if (!dA || !dB) continue;

      const phiA = dA.miedemaPhiStar;
      const phiB = dB.miedemaPhiStar;
      const nwsA = dA.miedemaNws13;
      const nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvgInv = 2 / (1 / nwsA + 1 / nwsB);
      const fAB = 2 * fractions[elements[i]] * fractions[elements[j]];
      const vAvg = (vA * fractions[elements[i]] + vB * fractions[elements[j]]) / (fractions[elements[i]] + fractions[elements[j]]);
      let interfaceEnergy = (-14.1 * deltaPhi * deltaPhi + 9.4 * deltaNws * deltaNws) / nwsAvgInv;

      const aIsNonmetal = NONMETALS.has(elements[i]);
      const bIsNonmetal = NONMETALS.has(elements[j]);
      if (aIsNonmetal !== bIsNonmetal) {
        const ionicityFactor = 0.73 * deltaPhi * deltaPhi;
        interfaceEnergy -= ionicityFactor / nwsAvgInv;
      }

      deltaH += fAB * vAvg * interfaceEnergy;
    }
  }

  return deltaH / totalAtoms;
}

function computeMiedemaFormationEnergy(formula: string): number {
  return computeMiedemaFromCounts(parseFormulaCounts(formula));
}

function computeComplexityScore(nPrecursors: number, nSteps: number): number {
  const precursorPenalty = Math.min(1, nPrecursors / 6);
  const stepPenalty = Math.min(1, nSteps / 5);
  return 1.0 - (precursorPenalty * 0.6 + stepPenalty * 0.4);
}

function computeStructuralSimilarity(precursorElements: string[], targetElements: string[]): number {
  const overlap = precursorElements.filter(e => targetElements.includes(e)).length;
  const combined = [...precursorElements, ...targetElements];
  const unique: string[] = [];
  for (const e of combined) {
    if (!unique.includes(e)) unique.push(e);
  }
  const total = unique.length;
  return total > 0 ? overlap / total : 0;
}

function formatEquation(reactants: string[], products: string[]): string {
  return `${reactants.join(" + ")} → ${products.join(" + ")}`;
}

function generateDirectElementalRoute(formula: string, elements: string[], counts: Record<string, number>): RetrosynthesisRoute {
  const precursors = elements.map(el => el);
  const equationParts = elements.map(el => {
    const c = counts[el];
    return c > 1 ? `${c}${el}` : el;
  });

  const deltaE = computeMiedemaFormationEnergy(formula);
  const nSteps = elements.length <= 2 ? 1 : 2;
  const complexity = computeComplexityScore(elements.length, nSteps);

  const precursorSelections = findBestPrecursors(elements, "arc-melting");
  const availResult = computePrecursorAvailabilityScore(precursorSelections);

  const similarity = 1.0;

  const reasoning = [
    `Direct combination of ${elements.length} elements`,
    `Miedema formation energy: ${deltaE.toFixed(3)} eV/atom`,
    deltaE < 0 ? "Exothermic formation expected" : "Endothermic - may require energy input",
  ];

  const overallScore = computeRouteScore(deltaE, availResult.overallScore, similarity, complexity);

  return {
    type: "direct-elemental",
    precursors,
    product: formula,
    equation: formatEquation(equationParts, [formula]),
    deltaE: Number(deltaE.toFixed(4)),
    complexity: Number(complexity.toFixed(4)),
    availability: Number(availResult.overallScore.toFixed(4)),
    similarity: Number(similarity.toFixed(4)),
    overallScore: Number(overallScore.toFixed(4)),
    reasoning,
  };
}

function generateBinaryIntermediateRoutes(formula: string, elements: string[], counts: Record<string, number>): RetrosynthesisRoute[] {
  const routes: RetrosynthesisRoute[] = [];

  for (const el of elements) {
    const binaries = COMMON_BINARIES[el];
    if (!binaries) continue;

    for (const binary of binaries) {
      const binaryElements = binary.elements;
      const allBinaryInTarget = binaryElements.every(be => elements.includes(be));
      if (!allBinaryInTarget) continue;
      if (binaryElements.length >= elements.length) continue;

      const remainingElements = elements.filter(e => !binaryElements.includes(e));
      if (remainingElements.length === 0) continue;

      const precursors = [binary.formula, ...remainingElements];

      const deltaE = computeMiedemaFormationEnergy(formula);
      const nSteps = 2;
      const complexity = computeComplexityScore(precursors.length, nSteps);

      const allPrecursorElements = [...binaryElements, ...remainingElements];
      const precursorSelections = findBestPrecursors(allPrecursorElements, "solid-state");
      const availResult = computePrecursorAvailabilityScore(precursorSelections);
      const adjustedAvail = availResult.overallScore * binary.stability;

      const similarity = computeStructuralSimilarity(binaryElements, elements);
      const overallScore = computeRouteScore(deltaE, adjustedAvail, similarity, complexity);

      const reasoning = [
        `Via binary intermediate ${binary.formula}`,
        `Binary stability: ${(binary.stability * 100).toFixed(0)}%`,
        `Remaining elements: ${remainingElements.join(", ")}`,
        `Miedema ΔE: ${deltaE.toFixed(3)} eV/atom`,
      ];

      routes.push({
        type: "binary-intermediate",
        precursors,
        product: formula,
        equation: formatEquation(precursors, [formula]),
        deltaE: Number(deltaE.toFixed(4)),
        complexity: Number(complexity.toFixed(4)),
        availability: Number(adjustedAvail.toFixed(4)),
        similarity: Number(similarity.toFixed(4)),
        overallScore: Number(overallScore.toFixed(4)),
        reasoning,
      });
    }
  }

  return routes;
}

function generateOxideDecompositionRoutes(formula: string, elements: string[], counts: Record<string, number>): RetrosynthesisRoute[] {
  const routes: RetrosynthesisRoute[] = [];
  if (!elements.includes("O")) return routes;

  const metals = elements.filter(e => !NONMETALS.has(e));
  if (metals.length === 0) return routes;

  const CARBONATE_ELEMENTS = new Set(["Ba", "Sr", "Ca", "Mg", "Li", "Na", "K", "Rb", "Cs"]);
  const carbonateMetals = metals.filter(m => CARBONATE_ELEMENTS.has(m));

  if (carbonateMetals.length > 0) {
    const precursors: string[] = [];
    const byproducts: string[] = [];

    for (const m of carbonateMetals) {
      const isAlkali = ["Li", "Na", "K", "Rb", "Cs"].includes(m);
      precursors.push(isAlkali ? `${m}2CO3` : `${m}CO3`);
      byproducts.push("CO2");
    }

    const OXIDE_FORMULAS: Record<string, string> = {
      Cu: "CuO", Fe: "Fe2O3", Al: "Al2O3", Ti: "TiO2", Nb: "Nb2O5",
      V: "V2O5", Cr: "Cr2O3", Mn: "MnO2", Co: "Co3O4", Ni: "NiO",
      Zn: "ZnO", Sn: "SnO2", Bi: "Bi2O3", Pb: "PbO", Y: "Y2O3",
      La: "La2O3", Zr: "ZrO2", Sc: "Sc2O3", Hf: "HfO2", Ta: "Ta2O5",
      W: "WO3", Mo: "MoO3", Si: "SiO2", Ge: "GeO2", In: "In2O3",
    };
    const otherMetals = metals.filter(m => !CARBONATE_ELEMENTS.has(m));
    for (const m of otherMetals) {
      precursors.push(OXIDE_FORMULAS[m] ?? `${m}2O3`);
    }

    const nonMetalNonO = elements.filter(e => NONMETALS.has(e) && e !== "O" && e !== "C");
    for (const nm of nonMetalNonO) {
      precursors.push(nm);
    }

    const deltaE = computeMiedemaFormationEnergy(formula);
    const complexity = computeComplexityScore(precursors.length, 3);
    const precursorSelections = findBestPrecursors(elements, "solid-state");
    const availResult = computePrecursorAvailabilityScore(precursorSelections);
    const similarity = computeStructuralSimilarity(metals, elements);
    const overallScore = computeRouteScore(deltaE, availResult.overallScore, similarity, complexity);

    const uniqueByproducts: string[] = [];
    for (const bp of byproducts) {
      if (!uniqueByproducts.includes(bp)) uniqueByproducts.push(bp);
    }

    routes.push({
      type: "oxide-decomposition",
      precursors,
      product: formula,
      equation: formatEquation(precursors, [formula, ...uniqueByproducts]),
      deltaE: Number(deltaE.toFixed(4)),
      complexity: Number(complexity.toFixed(4)),
      availability: Number(availResult.overallScore.toFixed(4)),
      similarity: Number(similarity.toFixed(4)),
      overallScore: Number(overallScore.toFixed(4)),
      reasoning: [
        `Carbonate/oxide decomposition route for ${carbonateMetals.join(", ")}`,
        `Byproducts: ${uniqueByproducts.join(", ")}`,
        "Decomposition provides reactive oxide intermediates",
        `Miedema ΔE: ${deltaE.toFixed(3)} eV/atom`,
      ],
    });
  }

  const NITRATE_FORMULAS: Record<string, string> = {
    Li: "LiNO3", Na: "NaNO3", K: "KNO3", Rb: "RbNO3", Cs: "CsNO3",
    Ba: "Ba(NO3)2", Sr: "Sr(NO3)2", Ca: "Ca(NO3)2", Mg: "Mg(NO3)2",
    Cu: "Cu(NO3)2", Fe: "Fe(NO3)3", Al: "Al(NO3)3", Y: "Y(NO3)3",
    La: "La(NO3)3", Zn: "Zn(NO3)2", Ni: "Ni(NO3)2", Co: "Co(NO3)2",
    Mn: "Mn(NO3)2", Bi: "Bi(NO3)3", Pb: "Pb(NO3)2", In: "In(NO3)3",
  };
  const nitratePrecursors: string[] = [];
  for (const m of metals) {
    nitratePrecursors.push(NITRATE_FORMULAS[m] ?? `${m}(NO3)2`);
  }
  const nonMetalNonO = elements.filter(e => NONMETALS.has(e) && e !== "O");
  for (const nm of nonMetalNonO) {
    nitratePrecursors.push(nm);
  }

  if (metals.length >= 2) {
    const deltaE = computeMiedemaFormationEnergy(formula);
    const complexity = computeComplexityScore(nitratePrecursors.length, 4);
    const precursorSelections = findBestPrecursors(elements, "sol-gel");
    const availResult = computePrecursorAvailabilityScore(precursorSelections);
    const similarity = computeStructuralSimilarity(metals, elements);
    const overallScore = computeRouteScore(deltaE, availResult.overallScore, similarity, complexity);

    routes.push({
      type: "oxide-decomposition",
      precursors: nitratePrecursors,
      product: formula,
      equation: formatEquation(nitratePrecursors, [formula, "NOx", "O2"]),
      deltaE: Number(deltaE.toFixed(4)),
      complexity: Number(complexity.toFixed(4)),
      availability: Number(availResult.overallScore.toFixed(4)),
      similarity: Number(similarity.toFixed(4)),
      overallScore: Number(overallScore.toFixed(4)),
      reasoning: [
        "Nitrate decomposition / sol-gel route",
        "Solution mixing provides atomic-level homogeneity",
        `${metals.length} metal nitrates decomposed to form target oxide`,
        `Miedema ΔE: ${deltaE.toFixed(3)} eV/atom`,
      ],
    });
  }

  return routes;
}

function generatePrecursorSubstitutionRoutes(formula: string, elements: string[], counts: Record<string, number>): RetrosynthesisRoute[] {
  const routes: RetrosynthesisRoute[] = [];
  const family = classifyFamily(formula);

  const methods = ["solid-state", "arc-melting", "sol-gel", "high-pressure"];
  for (const method of methods) {
    const precursorSelections = findBestPrecursors(elements, method);
    const availResult = computePrecursorAvailabilityScore(precursorSelections);

    if (availResult.overallScore < 0.2) continue;

    const precursorFormulas = precursorSelections.map(s => s.precursor.formula);
    const deltaE = computeMiedemaFormationEnergy(formula);
    const complexity = computeComplexityScore(precursorFormulas.length, method === "sol-gel" ? 4 : 2);
    const allPrecursorEls = precursorSelections.flatMap(s => parseFormulaElements(s.precursor.formula));
    const similarity = computeStructuralSimilarity(allPrecursorEls, elements);
    const overallScore = computeRouteScore(deltaE, availResult.overallScore, similarity, complexity);

    const reasoning = [
      `Precursor-based route via ${method}`,
      `Precursors: ${precursorFormulas.join(", ")}`,
      `Availability score: ${(availResult.overallScore * 100).toFixed(0)}%`,
      availResult.bottleneckElement ? `Bottleneck element: ${availResult.bottleneckElement}` : "No supply bottleneck",
      `Cost estimate: ${availResult.costEstimate}`,
    ];

    routes.push({
      type: "precursor-substitution",
      precursors: precursorFormulas,
      product: formula,
      equation: formatEquation(precursorFormulas, [formula]),
      deltaE: Number(deltaE.toFixed(4)),
      complexity: Number(complexity.toFixed(4)),
      availability: Number(availResult.overallScore.toFixed(4)),
      similarity: Number(similarity.toFixed(4)),
      overallScore: Number(overallScore.toFixed(4)),
      reasoning,
    });
  }

  return routes;
}

function computeRouteScore(deltaE: number, availability: number, similarity: number, complexity: number): number {
  const thermoScore = deltaE < -0.5 ? 1.0 : deltaE < 0 ? 0.8 : deltaE < 0.3 ? 0.5 : deltaE < 1.0 ? 0.3 : 0.1;
  return 0.4 * thermoScore + 0.3 * availability + 0.2 * similarity + 0.1 * complexity;
}

let stats = {
  totalAnalyzed: 0,
  totalRoutes: 0,
  topMethods: {} as Record<string, number>,
};

export function generateRetrosynthesisRoutes(formula: string): RetrosynthesisResult {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const family = classifyFamily(formula);
  const analysisNotes: string[] = [];

  if (elements.length < 2) {
    return {
      formula,
      family,
      totalRoutes: 0,
      routes: [],
      bestRoute: null,
      analysisNotes: ["Single-element formula - no decomposition possible"],
    };
  }

  const allRoutes: RetrosynthesisRoute[] = [];

  try {
    const directRoute = generateDirectElementalRoute(formula, elements, counts);
    allRoutes.push(directRoute);
    analysisNotes.push(`Direct elemental route: ${elements.join(" + ")} → ${formula}`);
  } catch (_) {
    analysisNotes.push("Failed to generate direct elemental route");
  }

  try {
    const binaryRoutes = generateBinaryIntermediateRoutes(formula, elements, counts);
    allRoutes.push(...binaryRoutes);
    if (binaryRoutes.length > 0) {
      analysisNotes.push(`Found ${binaryRoutes.length} binary intermediate route(s)`);
    }
  } catch (_) {
    analysisNotes.push("Failed to generate binary intermediate routes");
  }

  try {
    const oxideRoutes = generateOxideDecompositionRoutes(formula, elements, counts);
    allRoutes.push(...oxideRoutes);
    if (oxideRoutes.length > 0) {
      analysisNotes.push(`Found ${oxideRoutes.length} oxide/carbonate decomposition route(s)`);
    }
  } catch (_) {
    analysisNotes.push("Failed to generate oxide decomposition routes");
  }

  try {
    const precursorRoutes = generatePrecursorSubstitutionRoutes(formula, elements, counts);
    allRoutes.push(...precursorRoutes);
    if (precursorRoutes.length > 0) {
      analysisNotes.push(`Found ${precursorRoutes.length} precursor substitution route(s)`);
    }
  } catch (_) {
    analysisNotes.push("Failed to generate precursor substitution routes");
  }

  allRoutes.sort((a, b) => b.overallScore - a.overallScore);

  stats.totalAnalyzed++;
  stats.totalRoutes += allRoutes.length;
  for (const route of allRoutes) {
    stats.topMethods[route.type] = (stats.topMethods[route.type] || 0) + 1;
  }

  return {
    formula,
    family,
    totalRoutes: allRoutes.length,
    routes: allRoutes,
    bestRoute: allRoutes.length > 0 ? allRoutes[0] : null,
    analysisNotes,
  };
}

export function getRetrosynthesisStats(): { totalAnalyzed: number; avgRoutesPerTarget: number; topMethods: Record<string, number> } {
  return {
    totalAnalyzed: stats.totalAnalyzed,
    avgRoutesPerTarget: stats.totalAnalyzed > 0 ? Number((stats.totalRoutes / stats.totalAnalyzed).toFixed(2)) : 0,
    topMethods: { ...stats.topMethods },
  };
}
