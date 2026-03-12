import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { ELEMENTAL_DATA, getElementData, getMeltingPoint, getLatticeConstant } from "./elemental-data";
import { fetchSummary, fetchElasticity } from "./materials-project-client";
import { computeDimensionalityScore, detectStructuralMotifs } from "./physics-engine";
import { extractFeatures } from "./ml-predictor";
import { passesValenceFilter } from "./candidate-generator";
import { getCompetingPhases, assessMetastability } from "./phase-diagram-engine";
import { predictHydrideFormation } from "./pressure-engine";
import { IONIC_RADII } from "./crystal-prototypes";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

export interface ParsedComposition {
  counts: Record<string, number>;
  elements: string[];
  fractions: Record<string, number>;
  totalAtoms: number;
}

export function parseComposition(formula: string): ParsedComposition {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = totalAtoms > 0 ? counts[el] / totalAtoms : 0;
  }
  return { counts, elements, fractions, totalAtoms };
}

export interface GoldschmidtResult {
  factor: number;
  prediction: string;
  stoichiometryValid: boolean;
}

export interface StructuralHint {
  source: string;
  hint: string;
  severity: "info" | "warning" | "critical";
}

export interface StructurePrediction {
  spaceGroup: string;
  crystalSystem: string;
  latticeParams: {
    a: number; b: number; c: number;
    alpha: number; beta: number; gamma: number;
  };
  prototype: string;
  dimensionality: string;
  isStable: boolean;
  isMetastable: boolean;
  decompositionEnergy: number;
  convexHullDistance: number;
  synthesizability: number;
  synthesisNotes: string;
  goldschmidtTolerance?: GoldschmidtResult | null;
  structuralHints?: StructuralHint[];
}

const SG_NUMBER_TO_HM: Record<string, string> = {
  "221": "Pm-3m", "225": "Fm-3m", "229": "Im-3m", "227": "Fd-3m",
  "223": "Pm-3n", "204": "Im-3", "148": "R-3", "166": "R-3m",
  "139": "I4/mmm", "129": "P4/nmm", "123": "P4/mmm", "47": "Pmmm",
  "191": "P6/mmm", "194": "P6_3/mmc", "186": "P6_3mc", "164": "P-3m1",
  "187": "P-6m2", "136": "P4_2/mnm", "140": "I4/mcm", "220": "I-43d",
  "216": "F-43m", "205": "Pa-3",
};

export function normalizeSpaceGroup(sg: string): string {
  if (!sg) return sg;
  const trimmed = sg.trim();
  if (SG_NUMBER_TO_HM[trimmed]) return SG_NUMBER_TO_HM[trimmed];
  return trimmed
    .replace(/P63\/mmc/g, "P6_3/mmc")
    .replace(/P63mc/g, "P6_3mc")
    .replace(/P42\/mnm/g, "P4_2/mnm");
}

const KNOWN_PROTOTYPES: Record<string, { spaceGroup: string; crystalSystem: string; prototype: string; dimensionality: string }> = {
  "perovskite": { spaceGroup: "Pm-3m", crystalSystem: "cubic", prototype: "CaTiO3", dimensionality: "3D" },
  "anti-perovskite": { spaceGroup: "Pm-3m", crystalSystem: "cubic", prototype: "Li3OBr", dimensionality: "3D" },
  "cuprate": { spaceGroup: "I4/mmm", crystalSystem: "tetragonal", prototype: "La2CuO4", dimensionality: "quasi-2D" },
  "YBCO": { spaceGroup: "Pmmm", crystalSystem: "orthorhombic", prototype: "YBa2Cu3O7", dimensionality: "quasi-2D" },
  "rocksalt": { spaceGroup: "Fm-3m", crystalSystem: "cubic", prototype: "NaCl", dimensionality: "3D" },
  "fluorite": { spaceGroup: "Fm-3m", crystalSystem: "cubic", prototype: "CaF2", dimensionality: "3D" },
  "spinel": { spaceGroup: "Fd-3m", crystalSystem: "cubic", prototype: "MgAl2O4", dimensionality: "3D" },
  "ThCr2Si2": { spaceGroup: "I4/mmm", crystalSystem: "tetragonal", prototype: "ThCr2Si2", dimensionality: "quasi-2D" },
  "iron-pnictide": { spaceGroup: "P4/nmm", crystalSystem: "tetragonal", prototype: "LaFeAsO", dimensionality: "quasi-2D" },
  "MgB2": { spaceGroup: "P6/mmm", crystalSystem: "hexagonal", prototype: "AlB2", dimensionality: "quasi-2D" },
  "A15": { spaceGroup: "Pm-3n", crystalSystem: "cubic", prototype: "Cr3Si", dimensionality: "3D" },
  "clathrate": { spaceGroup: "Im-3m", crystalSystem: "cubic", prototype: "CaH6", dimensionality: "3D" },
  "sodalite": { spaceGroup: "Im-3m", crystalSystem: "cubic", prototype: "LaH10", dimensionality: "3D" },
  "Laves": { spaceGroup: "Fd-3m", crystalSystem: "cubic", prototype: "MgCu2", dimensionality: "3D" },
  "diamond": { spaceGroup: "Fd-3m", crystalSystem: "cubic", prototype: "C", dimensionality: "3D" },
  "graphite": { spaceGroup: "P6_3/mmc", crystalSystem: "hexagonal", prototype: "C", dimensionality: "2D" },
  "bismuth-selenide": { spaceGroup: "R-3m", crystalSystem: "rhombohedral", prototype: "Bi2Se3", dimensionality: "2D" },
  "Heusler": { spaceGroup: "Fm-3m", crystalSystem: "cubic", prototype: "Cu2MnAl", dimensionality: "3D" },
  "half-Heusler": { spaceGroup: "F-43m", crystalSystem: "cubic", prototype: "LiAlSi", dimensionality: "3D" },
  "Skutterudite": { spaceGroup: "Im-3", crystalSystem: "cubic", prototype: "CoSb3", dimensionality: "3D" },
  "Chevrel": { spaceGroup: "R-3", crystalSystem: "rhombohedral", prototype: "PbMo6S8", dimensionality: "3D" },
  "K2NiF4": { spaceGroup: "I4/mmm", crystalSystem: "tetragonal", prototype: "K2NiF4", dimensionality: "quasi-2D" },
  "infinite-layer": { spaceGroup: "P4/mmm", crystalSystem: "tetragonal", prototype: "NdNiO2", dimensionality: "quasi-2D" },
  "Pyrochlore": { spaceGroup: "Fd-3m", crystalSystem: "cubic", prototype: "Cd2Re2O7", dimensionality: "3D" },
  "T-prime": { spaceGroup: "P4/nmm", crystalSystem: "tetragonal", prototype: "Nd2CuO4", dimensionality: "quasi-2D" },
  "1111-Type": { spaceGroup: "P4/nmm", crystalSystem: "tetragonal", prototype: "LaFeAsO", dimensionality: "quasi-2D" },
  "CaBe2Ge2": { spaceGroup: "P4/nmm", crystalSystem: "tetragonal", prototype: "CaBe2Ge2", dimensionality: "quasi-2D" },
  "BiS2-type": { spaceGroup: "P4/nmm", crystalSystem: "tetragonal", prototype: "LaOBiS2", dimensionality: "quasi-2D" },
  "FeSe-11": { spaceGroup: "P4/nmm", crystalSystem: "tetragonal", prototype: "FeSe", dimensionality: "quasi-2D" },
  "MX2": { spaceGroup: "P-3m1", crystalSystem: "hexagonal", prototype: "MoS2", dimensionality: "2D" },
  "rutile": { spaceGroup: "P4_2/mnm", crystalSystem: "tetragonal", prototype: "TiO2", dimensionality: "3D" },
  "pyrite": { spaceGroup: "Pa-3", crystalSystem: "cubic", prototype: "FeS2", dimensionality: "3D" },
  "wurtzite": { spaceGroup: "P6_3mc", crystalSystem: "hexagonal", prototype: "ZnS", dimensionality: "3D" },
  "NiAs": { spaceGroup: "P6_3/mmc", crystalSystem: "hexagonal", prototype: "NiAs", dimensionality: "3D" },
};

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

const MIEDEMA_NONMETALS = new Set(["H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "Te", "As"]);

function computeMiedemaCore(
  elements: string[],
  fractions: Record<string, number>,
): number {
  const P = 14.1;
  const Q_P = 9.4;
  const R_P = 0;

  let deltaH = 0;

  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const elA = elements[i];
      const elB = elements[j];
      const dA = ELEMENTAL_DATA[elA];
      const dB = ELEMENTAL_DATA[elB];

      if (!dA || !dB) continue;

      const phiA = dA.miedemaPhiStar;
      const phiB = dB.miedemaPhiStar;
      const nwsA = Math.max(1e-6, dA.miedemaNws13 ?? 0);
      const nwsB = Math.max(1e-6, dB.miedemaNws13 ?? 0);
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || vA == null || vB == null) continue;
      if (dA.miedemaNws13 == null || dB.miedemaNws13 == null) continue;

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvgInv = 2 / (1 / nwsA + 1 / nwsB);

      const fAB = 2 * fractions[elA] * fractions[elB];

      const isTransitionA = (dA.atomicNumber >= 21 && dA.atomicNumber <= 30) ||
                            (dA.atomicNumber >= 39 && dA.atomicNumber <= 48) ||
                            (dA.atomicNumber >= 72 && dA.atomicNumber <= 80);
      const isTransitionB = (dB.atomicNumber >= 21 && dB.atomicNumber <= 30) ||
                            (dB.atomicNumber >= 39 && dB.atomicNumber <= 48) ||
                            (dB.atomicNumber >= 72 && dB.atomicNumber <= 80);

      let Q = Q_P;
      if (isTransitionA && isTransitionB) {
        Q = Q_P;
      } else if (!isTransitionA && !isTransitionB) {
        Q = Q_P * 0.73;
      }

      const fracSum = fractions[elA] + fractions[elB];
      const vAvg = fracSum > 0
        ? (vA * fractions[elA] + vB * fractions[elB]) / fracSum
        : (vA + vB) / 2;

      let interfaceEnergy = (-P * deltaPhi * deltaPhi + Q * deltaNws * deltaNws - R_P) / nwsAvgInv;

      const aIsNonmetal = MIEDEMA_NONMETALS.has(elA);
      const bIsNonmetal = MIEDEMA_NONMETALS.has(elB);
      if (aIsNonmetal !== bIsNonmetal) {
        interfaceEnergy -= (0.73 * deltaPhi * deltaPhi) / nwsAvgInv;
      }

      const contribution = fAB * vAvg * interfaceEnergy;

      deltaH += contribution;
    }
  }

  return deltaH;
}

function computeMiedemaFormationEnergy(formula: string, precomputed?: { elements: string[]; fractions: Record<string, number> }): number {
  let elements: string[];
  let fractions: Record<string, number>;

  if (precomputed) {
    elements = precomputed.elements;
    fractions = precomputed.fractions;
  } else {
    const counts = parseFormulaCounts(formula);
    elements = Object.keys(counts);
    if (elements.length < 2) return 0;
    const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
    if (totalAtoms === 0) return 0;
    fractions = {};
    for (const el of elements) {
      fractions[el] = counts[el] / totalAtoms;
    }
  }

  if (elements.length < 2) return 0;

  const hasHydrogen = elements.includes("H");
  if (hasHydrogen) {
    const metalEls = elements.filter(el => !MIEDEMA_NONMETALS.has(el));
    if (metalEls.length > 0) {
      try {
        const hydrideResult = predictHydrideFormation(metalEls, 0);
        if (hydrideResult && hydrideResult.stableHydrides && hydrideResult.stableHydrides.length > 0) {
          const avgHf = hydrideResult.stableHydrides.reduce(
            (sum: number, h: { Hf: number }) => sum + h.Hf, 0
          ) / hydrideResult.stableHydrides.length;
          return avgHf;
        }
      } catch {}
    }
  }

  return computeMiedemaCore(elements, fractions);
}

function estimateDecompositionEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  if (totalAtoms === 0) return 0;

  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

  const compoundEnergy = computeMiedemaFormationEnergy(formula, { elements, fractions });

  if (elements.length === 2) {
    return Math.max(0, compoundEnergy);
  }

  const binaryPhases: { elA: string; elB: string; energy: number }[] = [];
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const elA = elements[i];
      const elB = elements[j];
      const binaryFractions: Record<string, number> = { [elA]: 0.5, [elB]: 0.5 };
      const binaryEnergy = computeMiedemaCore([elA, elB], binaryFractions);
      if (binaryEnergy < 0) {
        binaryPhases.push({ elA, elB, energy: binaryEnergy });
      }
    }
  }

  if (binaryPhases.length === 0) {
    return Math.max(0, compoundEnergy);
  }

  let bestDecompEnergy = 0;
  const remainingFractions = { ...fractions };

  const sorted = binaryPhases.sort((a, b) => a.energy - b.energy);
  for (const phase of sorted) {
    const availA = remainingFractions[phase.elA] ?? 0;
    const availB = remainingFractions[phase.elB] ?? 0;
    if (availA <= 0 || availB <= 0) continue;
    const weight = Math.min(availA, availB);
    bestDecompEnergy += phase.energy * weight * 2;
    remainingFractions[phase.elA] = availA - weight;
    remainingFractions[phase.elB] = availB - weight;
  }

  const decomp = compoundEnergy - bestDecompEnergy;
  return Math.max(0, decomp);
}

export interface ConvexHullStabilityResult {
  isStable: boolean;
  isMetastable: boolean;
  verdict: string;
  formationEnergy: number;
  hullDistance: number;
  source: string;
  confidence: number;
  mpNoveltySignal?: boolean;
}

export async function evaluateConvexHullStability(
  decompositionEnergy: number,
  formula?: string
): Promise<ConvexHullStabilityResult> {
  let mpNoveltySignal = false;
  if (formula) {
    try {
      const mpData = await fetchSummary(formula);
      if (mpData) {
        const eAboveHull = mpData.energyAboveHull;
        const formE = mpData.formationEnergyPerAtom;
        const isStable = eAboveHull <= 0.005;
        const isMetastable = !isStable && eAboveHull <= 0.15;

        let verdict: string;
        if (isStable) {
          verdict = "Thermodynamically stable (on convex hull) [MP DFT]";
        } else if (eAboveHull <= 0.05) {
          verdict = `Near hull (${eAboveHull.toFixed(3)} eV/atom) - likely synthesizable [MP DFT]`;
        } else if (isMetastable) {
          verdict = `Moderately above hull (${eAboveHull.toFixed(3)} eV/atom) - may be kinetically trapped [MP DFT]`;
        } else {
          verdict = `Far above hull (${eAboveHull.toFixed(3)} eV/atom) - unlikely to be synthesized [MP DFT]`;
        }

        return {
          isStable,
          isMetastable,
          verdict,
          formationEnergy: formE,
          hullDistance: eAboveHull,
          source: "Materials Project",
          confidence: 1.0,
        };
      } else {
        mpNoveltySignal = true;
        console.log(`[Stability] ${formula} not found in Materials Project — high novelty signal`);
      }
    } catch (mpErr: unknown) {
      const isNetwork = mpErr instanceof Error && (
        mpErr.message.includes("ECONNREFUSED") ||
        mpErr.message.includes("ETIMEDOUT") ||
        mpErr.message.includes("fetch failed") ||
        mpErr.message.includes("rate limit")
      );
      if (isNetwork) {
        console.warn(`[Stability] MP network error for ${formula}: ${mpErr instanceof Error ? mpErr.message.slice(0, 100) : "unknown"}`);
      } else {
        mpNoveltySignal = true;
        console.log(`[Stability] ${formula} MP lookup failed (not network) — high novelty signal: ${mpErr instanceof Error ? mpErr.message.slice(0, 100) : "unknown"}`);
      }
    }

    try {
      const hullResult = await getCompetingPhases(formula);
      const metastability = assessMetastability(formula, hullResult.energyAboveHull, hullResult.decompositionProducts);
      const miedemaFormE = computeMiedemaFormationEnergy(formula);
      const eAboveHull = hullResult.energyAboveHull;
      const isStable = hullResult.isOnHull;
      const isMetastable = metastability.isMetastable;

      let verdict: string;
      if (isStable) {
        verdict = `Thermodynamically stable (convex hull, dHf=${miedemaFormE.toFixed(3)} eV/atom)`;
      } else if (eAboveHull <= 0.05) {
        verdict = `Near hull (${eAboveHull.toFixed(3)} eV/atom, barrier=${metastability.kineticBarrier.toFixed(2)} eV) - likely synthesizable`;
      } else if (isMetastable) {
        verdict = `Metastable (${eAboveHull.toFixed(3)} eV/atom, lifetime=${metastability.estimatedLifetime}) - kinetically trapped`;
      } else {
        verdict = `Far above hull (${eAboveHull.toFixed(3)} eV/atom, decomp=${hullResult.decompositionProducts.join("+")}) - unlikely to be synthesized`;
      }

      return {
        isStable,
        isMetastable,
        verdict: mpNoveltySignal ? `${verdict} [MP-novel]` : verdict,
        formationEnergy: miedemaFormE,
        hullDistance: eAboveHull,
        source: "Convex Hull Engine",
        confidence: 0.7,
        mpNoveltySignal,
      };
    } catch {
      const miedemaFormE = computeMiedemaFormationEnergy(formula);
      const miedemaDecomp = estimateDecompositionEnergy(formula);
      const effectiveDecomp = miedemaDecomp > 0 ? miedemaDecomp : Math.max(0, decompositionEnergy);

      const isStable = effectiveDecomp <= 0.005;
      const isMetastable = !isStable && effectiveDecomp <= 0.15;

      let verdict: string;
      if (isStable) {
        verdict = `Thermodynamically stable (Miedema dHf=${miedemaFormE.toFixed(3)} eV/atom)`;
      } else if (effectiveDecomp <= 0.05) {
        verdict = `Near hull (${effectiveDecomp.toFixed(3)} eV/atom, Miedema dHf=${miedemaFormE.toFixed(3)}) - likely synthesizable`;
      } else if (isMetastable) {
        verdict = `Moderately above hull (${effectiveDecomp.toFixed(3)} eV/atom, Miedema) - may be kinetically trapped`;
      } else {
        verdict = `Far above hull (${effectiveDecomp.toFixed(3)} eV/atom, Miedema) - unlikely to be synthesized`;
      }

      return {
        isStable,
        isMetastable,
        verdict: mpNoveltySignal ? `${verdict} [MP-novel]` : verdict,
        formationEnergy: miedemaFormE,
        hullDistance: effectiveDecomp,
        source: "Miedema model",
        confidence: 0.3,
        mpNoveltySignal,
      };
    }
  }

  const isStable = decompositionEnergy <= 0.005;
  const isMetastable = !isStable && decompositionEnergy <= 0.15;

  let verdict: string;
  if (isStable) {
    verdict = "Thermodynamically stable (on convex hull)";
  } else if (decompositionEnergy <= 0.05) {
    verdict = "Near hull - likely synthesizable as metastable phase";
  } else if (isMetastable) {
    verdict = "Moderately above hull - may be kinetically trapped";
  } else {
    verdict = "Far above hull - unlikely to be synthesized";
  }

  return {
    isStable,
    isMetastable,
    verdict,
    formationEnergy: -decompositionEnergy,
    hullDistance: decompositionEnergy,
    source: "decomposition energy input",
    confidence: 0.2,
  };
}

export function matchPrototype(formula: string, comp?: ParsedComposition): typeof KNOWN_PROTOTYPES[string] | null {
  const { elements, counts } = comp || parseComposition(formula);

  const has = (el: string) => elements.includes(el);
  const hasAny = (...els: string[]) => els.some(e => has(e));

  if (has("Ni") && has("O") && elements.length === 3 &&
      hasAny("Nd", "Pr", "La") &&
      (counts["Ni"] || 0) === 1 && (counts["O"] || 0) <= 2) {
    return KNOWN_PROTOTYPES["infinite-layer"];
  }

  if (has("Cu") && has("O") && elements.length >= 4 &&
      has("Y") && has("Ba") &&
      (counts["Cu"] || 0) >= 2) {
    return KNOWN_PROTOTYPES["YBCO"];
  }

  if (has("Fe") && has("Se") && elements.length === 2) {
    return KNOWN_PROTOTYPES["FeSe-11"];
  }

  if (has("Fe") && hasAny("As", "P")) {
    if (has("O") && elements.length === 4) {
      return KNOWN_PROTOTYPES["1111-Type"];
    }
    if (elements.length === 3 && hasAny("Ba", "Sr", "K", "Ca", "Cs", "Rb")) {
      return KNOWN_PROTOTYPES["ThCr2Si2"];
    }
    return KNOWN_PROTOTYPES["iron-pnictide"];
  }

  if (has("Fe") && has("Se") && elements.length >= 3) {
    return KNOWN_PROTOTYPES["iron-pnictide"];
  }

  if (has("Cu") && has("O") && elements.length >= 3) {
    const chargeRes = ["Ba", "Sr", "La", "Y", "Ca", "Tl", "Bi", "Hg"];
    if (chargeRes.some(e => has(e))) {
      return KNOWN_PROTOTYPES["cuprate"];
    }
  }

  if ((has("Co") || has("Rh") || has("Ir")) &&
      hasAny("Sb", "As", "P") &&
      elements.length === 2) {
    return KNOWN_PROTOTYPES["Skutterudite"];
  }

  if (has("Mo") && hasAny("S", "Se", "Te") && elements.length >= 2) {
    if (elements.length === 2 && (counts["S"] || counts["Se"] || counts["Te"] || 0) === 2) {
      return KNOWN_PROTOTYPES["MX2"];
    }
    if (elements.length === 3) {
      return KNOWN_PROTOTYPES["Chevrel"];
    }
  }

  if (has("Bi") && hasAny("S", "Se") && has("O") && elements.length >= 4) {
    return KNOWN_PROTOTYPES["BiS2-type"];
  }

  if (has("H") && elements.length === 2) {
    const metal = elements.find(e => e !== "H");
    const hCount = counts["H"] || 0;
    if (metal && hCount >= 6) return KNOWN_PROTOTYPES["clathrate"];
    if (metal && hCount >= 10) return KNOWN_PROTOTYPES["sodalite"];
    return KNOWN_PROTOTYPES["clathrate"];
  }

  if (has("Mg") && has("B")) {
    return KNOWN_PROTOTYPES["MgB2"];
  }

  if (hasAny("Nb", "V", "Cr") && hasAny("Sn", "Si", "Ge", "Ga", "Al")) {
    if (elements.length === 2) {
      const tm = elements.find(e => ["Nb", "V", "Cr"].includes(e));
      if (tm && (counts[tm] || 0) === 3) return KNOWN_PROTOTYPES["A15"];
    }
  }

  if (elements.length === 3 &&
      hasAny("Mn", "Ti", "V", "Fe", "Co", "Ni") &&
      hasAny("Al", "Ga", "Sn", "Sb", "In", "Si", "Ge")) {
    const totalNonTM = elements.filter(e => !["Mn", "Ti", "V", "Fe", "Co", "Ni"].includes(e))
      .reduce((s, e) => s + (counts[e] || 0), 0);
    const totalTM = elements.filter(e => ["Mn", "Ti", "V", "Fe", "Co", "Ni"].includes(e))
      .reduce((s, e) => s + (counts[e] || 0), 0);
    if (totalTM === 2 && totalNonTM === 1) return KNOWN_PROTOTYPES["Heusler"];
    if (totalTM === 1 && totalNonTM === 2) return KNOWN_PROTOTYPES["half-Heusler"];
  }

  if (has("O") && elements.length === 3 &&
      hasAny("Cd", "Tl") && hasAny("Re", "Os", "Ir")) {
    return KNOWN_PROTOTYPES["Pyrochlore"];
  }

  if (has("F") && elements.length === 3 &&
      hasAny("K", "Rb", "Ba", "Sr")) {
    return KNOWN_PROTOTYPES["K2NiF4"];
  }

  if (has("O") && elements.length === 3) {
    const nonO = elements.filter(e => e !== "O");
    const totalMetal = nonO.reduce((s, e) => s + (counts[e] || 0), 0);
    const oCount = counts["O"] || 0;
    if (totalMetal === 2 && oCount === 3) {
      return KNOWN_PROTOTYPES["perovskite"];
    }
    if (totalMetal === 2 && oCount === 4) {
      return KNOWN_PROTOTYPES["spinel"];
    }
  }

  if (has("O") && elements.length >= 3) {
    return KNOWN_PROTOTYPES["perovskite"];
  }

  return null;
}

const PROTOTYPE_CA_RATIOS: Record<string, number> = {
  perovskite: 1.0,
  rutile: 0.64,
  wurtzite: 1.63,
  fluorite: 1.0,
  spinel: 1.0,
  rocksalt: 1.0,
  NaCl: 1.0,
  CsCl: 1.0,
  "A15": 1.0,
  MgB2: 1.14,
  cuprate: 3.2,
  YBCO: 3.0,
  "iron-pnictide": 2.2,
  "ThCr2Si2": 2.6,
  clathrate: 1.0,
  sodalite: 1.0,
  Laves: 1.0,
  diamond: 1.0,
  "bismuth-selenide": 7.0,
  Heusler: 1.0,
  Skutterudite: 1.0,
  Chevrel: 1.4,
  K2NiF4: 3.2,
  "infinite-layer": 1.7,
  Pyrochlore: 1.0,
};

function getPrototypeCARatio(prototype: string | null): number | null {
  if (!prototype) return null;
  for (const [key, ratio] of Object.entries(PROTOTYPE_CA_RATIOS)) {
    if (prototype.toLowerCase().includes(key.toLowerCase())) return ratio;
  }
  return null;
}

export function computeGoldschmidtTolerance(formula: string, comp?: ParsedComposition): { factor: number; prediction: string; stoichiometryValid: boolean } | null {
  const { counts, elements } = comp || parseComposition(formula);

  if (elements.length < 3 || !elements.includes("O")) return null;

  const oCount = counts["O"] || 0;
  const nonO = elements.filter(e => e !== "O");
  if (nonO.length < 2) return null;

  const totalNonO = nonO.reduce((s, e) => s + (counts[e] || 0), 0);
  const metalToOxygen = totalNonO > 0 ? oCount / totalNonO : 0;

  const isABO3Ratio = Math.abs(metalToOxygen - 1.5) < 0.3;

  if (!isABO3Ratio) {
    return {
      factor: 0,
      prediction: `non-perovskite stoichiometry (metal:O = ${totalNonO}:${oCount}, need ~2:3)`,
      stoichiometryValid: false,
    };
  }

  const sorted = nonO.sort((a, b) => {
    const rA = getElementData(a)?.atomicRadius ?? 0;
    const rB = getElementData(b)?.atomicRadius ?? 0;
    return rB - rA;
  });

  const aSite = sorted[0];
  const bSite = sorted[1];
  const dataA = getElementData(aSite);
  const dataB = getElementData(bSite);
  if (!dataA?.atomicRadius || !dataB?.atomicRadius) return null;
  const rA = dataA.atomicRadius / 100;
  const rB = dataB.atomicRadius / 100;
  const rO = 1.40;

  const t = (rA + rO) / (Math.sqrt(2) * (rB + rO));

  let prediction: string;
  if (t > 1.05) {
    prediction = "hexagonal polytypes likely";
  } else if (t >= 0.95) {
    prediction = "ideal cubic perovskite";
  } else if (t >= 0.85) {
    prediction = "distorted perovskite (orthorhombic/rhombohedral tilting)";
  } else if (t >= 0.75) {
    prediction = "strongly distorted perovskite, possible ilmenite";
  } else {
    prediction = "perovskite structure unlikely";
  }

  return { factor: Number(t.toFixed(4)), prediction, stoichiometryValid: true };
}

function buildStructuralHints(
  tolerance: GoldschmidtResult | null,
  crystalSystem: string,
  dimensionality: string,
  prototype: string,
  elements: string[]
): StructuralHint[] {
  const hints: StructuralHint[] = [];

  if (tolerance && tolerance.stoichiometryValid) {
    const t = tolerance.factor;
    if (t > 1.05) {
      hints.push({
        source: "goldschmidt",
        hint: `t=${t.toFixed(4)} > 1.05: favor hexagonal polytypes (e.g. 2H, 4H, 6H) over cubic perovskite during relaxation`,
        severity: "warning",
      });
    } else if (t >= 0.95) {
      hints.push({
        source: "goldschmidt",
        hint: `t=${t.toFixed(4)}: ideal cubic perovskite expected, start relaxation from Pm-3m`,
        severity: "info",
      });
    } else if (t >= 0.85) {
      hints.push({
        source: "goldschmidt",
        hint: `t=${t.toFixed(4)}: expect octahedral tilting, initialize with Pnma or R-3c distortion`,
        severity: "info",
      });
    } else if (t >= 0.75) {
      hints.push({
        source: "goldschmidt",
        hint: `t=${t.toFixed(4)}: strongly distorted, consider ilmenite (R-3) or LiNbO3-type as starting structure`,
        severity: "warning",
      });
    } else if (t > 0) {
      hints.push({
        source: "goldschmidt",
        hint: `t=${t.toFixed(4)} < 0.75: perovskite structure unlikely, explore alternative ABO3 polymorphs`,
        severity: "critical",
      });
    }
  }

  const cs = crystalSystem.toLowerCase();
  if ((cs === "hexagonal" || cs === "trigonal") && dimensionality === "quasi-2D") {
    hints.push({
      source: "dimensionality",
      hint: "quasi-2D layered hexagonal/trigonal: use anisotropic k-mesh with boosted c*-axis sampling",
      severity: "info",
    });
  }

  const hasH = elements.includes("H");
  const protoLc = prototype.toLowerCase();
  if (hasH && (protoLc.includes("hydride") || protoLc.includes("clathrate"))) {
    hints.push({
      source: "prototype",
      hint: "hydrogen clathrate/cage structure: ensure H sublattice is fully relaxed before computing Tc",
      severity: "info",
    });
  }

  if (protoLc.includes("infinite-layer")) {
    hints.push({
      source: "prototype",
      hint: "infinite-layer cuprate: apical-oxygen-free, use P4/mmm starting symmetry",
      severity: "info",
    });
  }

  return hints;
}

export function vegardLatticeParameter(formula: string, comp?: ParsedComposition, proto?: typeof KNOWN_PROTOTYPES[string] | null): number | null {
  const { counts, elements, totalAtoms } = comp || parseComposition(formula);

  if (elements.length < 2 || totalAtoms === 0) return null;

  const ANION_SET = new Set(["O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "As"]);
  const anions = elements.filter(e => ANION_SET.has(e));
  const cations = elements.filter(e => !ANION_SET.has(e));
  const isIonic = anions.length > 0 && cations.length > 0;

  if (isIonic) {
    let cationRadiusSum = 0;
    let cationCount = 0;
    let anionRadiusSum = 0;
    let anionCount = 0;

    for (const el of cations) {
      const r = IONIC_RADII[el];
      if (r === undefined) continue;
      const n = counts[el] || 0;
      cationRadiusSum += r * n;
      cationCount += n;
    }
    for (const el of anions) {
      const r = IONIC_RADII[el];
      if (r === undefined) continue;
      const n = counts[el] || 0;
      anionRadiusSum += r * n;
      anionCount += n;
    }

    if (cationCount === 0 || anionCount === 0) return null;

    const avgCation = cationRadiusSum / cationCount;
    const avgAnion = anionRadiusSum / anionCount;

    const resolvedProto = proto !== undefined ? proto : matchPrototype(formula, comp);
    const protoName = resolvedProto?.prototype?.toLowerCase() || "";

    if (protoName.includes("perovskite") || protoName.includes("anti-perovskite")) {
      return 2 * (avgCation * 0.4 + avgAnion * 0.6) * Math.sqrt(2);
    }
    if (protoName.includes("fluorite") || protoName.includes("spinel")) {
      return 4 * (avgCation + avgAnion) / Math.sqrt(3);
    }

    return 2 * (avgCation + avgAnion);
  }

  let weightedSum = 0;
  let totalWeight = 0;

  for (const el of elements) {
    let lc = getLatticeConstant(el);
    if (lc === null) {
      const data = getElementData(el);
      if (data?.atomicRadius) {
        lc = 2 * Math.SQRT2 * (data.atomicRadius / 100);
      } else {
        continue;
      }
    }
    const frac = counts[el] / totalAtoms;
    weightedSum += frac * lc;
    totalWeight += frac;
  }

  if (totalWeight < 0.5) return null;
  return weightedSum / totalWeight;
}

const PROTOTYPE_BA_RATIOS: Record<string, number> = {
  YBCO: 1.016,
  cuprate: 1.0,
  "iron-pnictide": 1.0,
  "ThCr2Si2": 1.0,
  Chevrel: 1.0,
  Skutterudite: 1.0,
  Heusler: 1.0,
  "half-Heusler": 1.0,
  perovskite: 1.0,
  "anti-perovskite": 1.0,
};

function getPrototypeBAOverA(prototype: string | null): number | null {
  if (!prototype) return null;
  for (const [key, ratio] of Object.entries(PROTOTYPE_BA_RATIOS)) {
    if (prototype.toLowerCase().includes(key.toLowerCase())) return ratio;
  }
  return null;
}

function estimateLatticeFromVolume(
  volume: number,
  nsites: number,
  crystalSystem: string,
  prototype?: string | null,
  formula?: string | null,
  comp?: ParsedComposition,
  protoMatch?: typeof KNOWN_PROTOTYPES[string] | null
): { a: number; b: number; c: number } {
  const vegardA = formula ? vegardLatticeParameter(formula, comp, protoMatch) : null;

  const protoCARatio = getPrototypeCARatio(prototype || null);
  const protoBAOverA = getPrototypeBAOverA(prototype || null);

  const Z = Math.max(nsites, 1);
  const cellVol = volume;

  if (vegardA && protoCARatio) {
    const a = vegardA;
    const c = a * protoCARatio;
    const cs = crystalSystem.toLowerCase();
    if (cs === "cubic") return { a, b: a, c: a };
    if (cs === "tetragonal") return { a, b: a, c };
    if (cs === "hexagonal" || cs === "trigonal") return { a, b: a, c };
    if (cs === "orthorhombic") {
      const ba = protoBAOverA ?? 1.0;
      if (cellVol > 0) {
        const solvedA = Math.pow(cellVol / (ba * protoCARatio), 1 / 3);
        return { a: solvedA, b: solvedA * ba, c: solvedA * protoCARatio };
      }
      return { a, b: a * ba, c };
    }
    return { a, b: a, c };
  }

  const caRatio = protoCARatio || (() => {
    switch (crystalSystem.toLowerCase()) {
      case "cubic": return 1.0;
      case "tetragonal": return 1.2;
      case "hexagonal":
      case "trigonal": return 1.63;
      case "orthorhombic": return 1.1;
      case "monoclinic": return 1.15;
      default: return 1.0;
    }
  })();

  const cs = crystalSystem.toLowerCase();
  let a: number, b: number, c: number;

  if (cs === "cubic") {
    a = Math.pow(cellVol, 1 / 3);
    b = a;
    c = a;
  } else if (cs === "tetragonal") {
    a = Math.pow(cellVol / (caRatio), 1 / 3);
    b = a;
    c = a * caRatio;
  } else if (cs === "hexagonal" || cs === "trigonal") {
    a = Math.pow(cellVol / (caRatio * Math.sqrt(3) / 2), 1 / 3);
    b = a;
    c = a * caRatio;
  } else if (cs === "orthorhombic") {
    const ba = protoBAOverA ?? 1.02;
    a = Math.pow(cellVol / (ba * caRatio), 1 / 3);
    b = a * ba;
    c = a * caRatio;
  } else if (cs === "monoclinic") {
    a = Math.pow(cellVol / (1.0 * caRatio), 1 / 3);
    b = a;
    c = a * caRatio;
  } else {
    a = Math.pow(cellVol, 1 / 3);
    b = a;
    c = a;
  }

  return { a, b, c };
}

function anglesForCrystalSystem(cs: string): { alpha: number; beta: number; gamma: number } {
  switch (cs.toLowerCase()) {
    case "cubic":
    case "tetragonal":
    case "orthorhombic":
      return { alpha: 90, beta: 90, gamma: 90 };
    case "hexagonal":
    case "trigonal":
      return { alpha: 90, beta: 90, gamma: 120 };
    case "monoclinic":
      return { alpha: 90, beta: 100, gamma: 90 };
    case "rhombohedral":
      return { alpha: 80, beta: 80, gamma: 80 };
    default:
      return { alpha: 90, beta: 90, gamma: 90 };
  }
}

function estimateSynthesizability(
  hullDistance: number,
  formationEnergy: number,
  elements: string[],
  formula?: string,
  comp?: ParsedComposition
): { score: number; notes: string; estimatedSynthesisTemp?: number } {
  let score = 1.0;
  const notes: string[] = [];

  const miedemaFormE = formula ? computeMiedemaFormationEnergy(formula) : formationEnergy;
  const absMiedema = Math.abs(miedemaFormE);

  if (miedemaFormE < -0.5) {
    score += 0.1;
    notes.push(`strongly exothermic (dHf=${miedemaFormE.toFixed(2)} eV/atom)`);
  } else if (miedemaFormE < -0.1) {
    notes.push(`mildly exothermic (dHf=${miedemaFormE.toFixed(2)} eV/atom)`);
  } else if (miedemaFormE < 0.1) {
    score -= 0.05;
    notes.push(`near-zero formation energy (dHf=${miedemaFormE.toFixed(2)} eV/atom)`);
  } else {
    score -= 0.15 - Math.min(0.15, absMiedema * 0.1);
    notes.push(`endothermic (dHf=${miedemaFormE.toFixed(2)} eV/atom)`);
  }

  if (hullDistance > 0.3) {
    score -= 0.4;
    notes.push("far above convex hull");
  } else if (hullDistance > 0.1) {
    score -= 0.2;
    notes.push("moderately above hull");
  } else if (hullDistance > 0.05) {
    score -= 0.08;
    notes.push("slightly above hull");
  }

  let maxMeltingPoint = 0;
  let minMeltingPoint = Infinity;
  let estimatedSynthesisTemp: number | undefined;

  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > 0) {
      if (mp > maxMeltingPoint) maxMeltingPoint = mp;
      if (mp < minMeltingPoint) minMeltingPoint = mp;
    }
  }
  if (minMeltingPoint === Infinity) minMeltingPoint = 0;

  if (maxMeltingPoint > 0) {
    const tammannBase = minMeltingPoint > 0 ? minMeltingPoint : maxMeltingPoint;
    estimatedSynthesisTemp = Math.round(0.5 * tammannBase);
    notes.push(`Tammann T_synth ~ ${estimatedSynthesisTemp} K (0.5 * T_melt of ${tammannBase} K)`);

    if (estimatedSynthesisTemp > 3000) {
      score -= 0.15;
      notes.push("very high synthesis temperature required");
    } else if (estimatedSynthesisTemp > 2000) {
      score -= 0.05;
    }
  }

  const hasH = elements.includes("H");
  if (hasH) {
    const resolved = comp || (formula ? parseComposition(formula) : null);
    const hCount = resolved?.counts["H"] || 0;
    const resolvedTotal = resolved?.totalAtoms || 1;
    const hFrac = hCount / resolvedTotal;

    if (hFrac > 0.6) {
      score -= 0.2;
      notes.push("superhydride requires very high pressure synthesis (>100 GPa)");
    } else if (hFrac > 0.3) {
      score -= 0.1;
      notes.push("hydride may require moderate-high pressure synthesis");
    }
  }

  const actinideHydrideTargets = new Set(["Th", "U", "Ac", "Pa"]);
  const radioactive = ["Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"];
  for (const el of elements) {
    if (radioactive.includes(el)) {
      if (hasH && actinideHydrideTargets.has(el)) {
        notes.push(`actinide ${el} (handling precautions required)`);
      } else {
        score -= 0.25;
        notes.push(`radioactive element ${el}`);
      }
      break;
    }
  }

  score = Math.max(0.05, Math.min(1.0, score));
  const noteStr = notes.length > 0 ? notes.join("; ") : "Standard synthesis conditions expected";

  return { score, notes: noteStr, estimatedSynthesisTemp };
}

export async function predictCrystalStructure(
  emit: EventEmitter,
  formula: string
): Promise<StructurePrediction | null> {
  if (typeof formula !== "string") {
    formula = (formula as any)?.formula || String(formula ?? "");
  }
  if (!formula || formula === "[object Object]") return null;

  const comp = parseComposition(formula);
  const protoMatch = matchPrototype(formula, comp);
  const elements = comp.elements;

  try {
    const mpData = await fetchSummary(formula);

    if (mpData) {
      const cs = mpData.crystalSystem || protoMatch?.crystalSystem || "cubic";
      const sg = mpData.spaceGroup || protoMatch?.spaceGroup || "P1";

      const proto = protoMatch?.prototype || `${cs} (${sg})`;
      const lattice = estimateLatticeFromVolume(mpData.volume, mpData.nsites, cs, proto, formula, comp, protoMatch);
      const angles = anglesForCrystalSystem(cs);

      const eAboveHull = mpData.energyAboveHull;
      const formE = mpData.formationEnergyPerAtom;
      const isStable = eAboveHull <= 0.005;
      const isMetastable = !isStable && eAboveHull <= 0.15;

      let dim = protoMatch?.dimensionality || "3D";
      if (!protoMatch) {
        if (cs.toLowerCase() === "hexagonal" || cs.toLowerCase() === "trigonal") {
          dim = "quasi-2D";
        }
      }

      const tolerance = computeGoldschmidtTolerance(formula, comp);
      const synth = estimateSynthesizability(eAboveHull, formE, elements, formula, comp);
      const hints = buildStructuralHints(tolerance, cs, dim, proto, elements);

      const result: StructurePrediction = {
        spaceGroup: sg,
        crystalSystem: cs,
        latticeParams: {
          a: lattice.a,
          b: lattice.b,
          c: lattice.c,
          alpha: angles.alpha,
          beta: angles.beta,
          gamma: angles.gamma,
        },
        prototype: proto,
        dimensionality: dim,
        isStable,
        isMetastable,
        decompositionEnergy: eAboveHull,
        convexHullDistance: eAboveHull,
        synthesizability: synth.score,
        synthesisNotes: synth.notes,
        goldschmidtTolerance: tolerance,
        structuralHints: hints,
      };

      const id = `cs-${Date.now()}-${formula.replace(/[^a-zA-Z0-9]/g, "").slice(0, 8)}`;
      await storage.insertCrystalStructure({
        id,
        formula,
        spaceGroup: result.spaceGroup,
        crystalSystem: result.crystalSystem,
        latticeParams: result.latticeParams,
        atomicPositions: null,
        prototype: result.prototype,
        dimensionality: result.dimensionality,
        isStable: result.isStable,
        isMetastable: result.isMetastable,
        decompositionEnergy: result.decompositionEnergy,
        convexHullDistance: result.convexHullDistance,
        synthesizability: result.synthesizability,
        synthesisNotes: result.synthesisNotes,
        source: "Materials Project + Structure Predictor",
        isGroundTruth: true,
      });

      const toleranceInfo = tolerance && tolerance.stoichiometryValid ? `, Goldschmidt t=${tolerance.factor} (${tolerance.prediction})` : "";
      emit("log", {
        phase: "phase-11",
        event: "Crystal structure predicted (MP data)",
        detail: `${formula}: ${sg} (${cs}), hull=${eAboveHull.toFixed(3)} eV/atom, dHf=${formE.toFixed(3)} eV/atom${toleranceInfo}`,
        dataSource: "Materials Project",
      });

      return result;
    }
  } catch {
  }

  try {
    const miedemaFormE = computeMiedemaFormationEnergy(formula);
    const miedemaDecomp = estimateDecompositionEnergy(formula);

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a computational crystallographer predicting crystal structures for materials.
Given a chemical formula${protoMatch ? ` (likely ${protoMatch.prototype}-type structure)` : ""}, predict:
1. Most likely space group and crystal system
2. Approximate lattice parameters (in Angstroms and degrees)
3. Structure prototype it most resembles from ICSD
4. Dimensionality (3D bulk, quasi-2D layered, 2D, 1D chain)

The Miedema model estimates formation energy = ${miedemaFormE.toFixed(3)} eV/atom and decomposition energy = ${miedemaDecomp.toFixed(3)} eV/atom for this compound.

Return JSON with fields: spaceGroup, crystalSystem, latticeA, latticeB, latticeC, alpha, beta, gamma, prototype, dimensionality`,
        },
        {
          role: "user",
          content: `Predict crystal structure for: ${formula}`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return null;

    const parsed = JSON.parse(content);

    const toNum = (v: unknown, fallback: number) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    };

    const isStable = miedemaDecomp <= 0.005;
    const isMetastable = !isStable && miedemaDecomp <= 0.15;
    const tolerance = computeGoldschmidtTolerance(formula, comp);
    const synth = estimateSynthesizability(miedemaDecomp, miedemaFormE, elements, formula, comp);

    const rawSg = (typeof parsed.spaceGroup === "string" && parsed.spaceGroup) || protoMatch?.spaceGroup || "P1";
    const miedemaSg = normalizeSpaceGroup(rawSg);
    const miedemaCs = (typeof parsed.crystalSystem === "string" && parsed.crystalSystem) || protoMatch?.crystalSystem || "triclinic";
    const miedemaProto = (typeof parsed.prototype === "string" && parsed.prototype) || protoMatch?.prototype || "unknown";
    const miedemaDim = (typeof parsed.dimensionality === "string" && parsed.dimensionality) || protoMatch?.dimensionality || "3D";
    const hints = buildStructuralHints(tolerance, miedemaCs, miedemaDim, miedemaProto, elements);

    const rawA = toNum(parsed.latticeA, 4.0);
    const rawC = toNum(parsed.latticeC, rawA);
    const csLower = miedemaCs.toLowerCase();
    let llmA = rawA;
    let llmB = toNum(parsed.latticeB, rawA);
    let llmC = rawC;
    if (csLower === "cubic") {
      llmB = llmA;
      llmC = llmA;
    } else if (csLower === "tetragonal") {
      llmB = llmA;
    } else if (csLower === "hexagonal" || csLower === "trigonal") {
      llmB = llmA;
    }

    const result: StructurePrediction = {
      spaceGroup: miedemaSg,
      crystalSystem: miedemaCs,
      latticeParams: {
        a: llmA,
        b: llmB,
        c: llmC,
        alpha: toNum(parsed.alpha, 90),
        beta: toNum(parsed.beta, 90),
        gamma: toNum(parsed.gamma, 90),
      },
      prototype: miedemaProto,
      dimensionality: miedemaDim,
      isStable,
      isMetastable,
      decompositionEnergy: miedemaDecomp,
      convexHullDistance: miedemaDecomp,
      synthesizability: synth.score,
      synthesisNotes: synth.notes,
      goldschmidtTolerance: tolerance,
      structuralHints: hints,
    };

    const id = `cs-${Date.now()}-${formula.replace(/[^a-zA-Z0-9]/g, "").slice(0, 8)}`;
    await storage.insertCrystalStructure({
      id,
      formula,
      spaceGroup: result.spaceGroup,
      crystalSystem: result.crystalSystem,
      latticeParams: result.latticeParams,
      atomicPositions: null,
      prototype: result.prototype,
      dimensionality: result.dimensionality,
      isStable: result.isStable,
      isMetastable: result.isMetastable,
      decompositionEnergy: result.decompositionEnergy,
      convexHullDistance: result.convexHullDistance,
      synthesizability: result.synthesizability,
      synthesisNotes: result.synthesisNotes,
      source: "Miedema + Structure Predictor",
      isGroundTruth: false,
    });

    const toleranceInfo2 = tolerance && tolerance.stoichiometryValid ? `, Goldschmidt t=${tolerance.factor} (${tolerance.prediction})` : "";
    emit("log", {
      phase: "phase-11",
      event: "Crystal structure predicted (Miedema)",
      detail: `${formula}: ${result.spaceGroup} (${result.crystalSystem}), Miedema dHf=${miedemaFormE.toFixed(3)} eV/atom, decomp=${miedemaDecomp.toFixed(3)} eV/atom${toleranceInfo2}`,
      dataSource: "Miedema model",
    });

    return result;
  } catch (err: any) {
    emit("log", {
      phase: "phase-11",
      event: "Structure prediction error",
      detail: `${formula}: ${err.message?.slice(0, 150)}`,
      dataSource: "Structure Predictor",
    });
    return null;
  }
}

export function assessDimensionality(structure: StructurePrediction): {
  dim: string;
  scImplication: string;
} {
  const dim = structure.dimensionality;
  let scImplication = "";

  switch (dim) {
    case "3D":
      scImplication = "Isotropic superconductivity expected; conventional BCS more likely; moderate Hc2";
      break;
    case "quasi-2D":
      scImplication = "Anisotropic superconductivity; layered structure may support higher Tc via 2D confinement; high upper critical field possible along c-axis";
      break;
    case "2D":
      scImplication = "Strongly 2D character; potential for enhanced fluctuations; Berezinskii-Kosterlitz-Thouless transition; very high anisotropy";
      break;
    case "1D":
      scImplication = "Quasi-1D chains risk Peierls instability and structural phase transitions; 1D character enhances nesting-driven CDW competition; may serve as coupling bridges in higher-dimensional host (e.g. Cu-O chains in YBCO) but bulk 1D Tc suppressed by fluctuations";
      break;
    default:
      scImplication = "Dimensionality unclear; standard analysis applies";
  }

  return { dim, scImplication };
}

export async function runStructurePredictionBatch(
  emit: EventEmitter,
  formulas: string[],
  batchSize?: number
): Promise<number> {
  let predicted = 0;
  const limit = batchSize ?? formulas.length;

  for (const formula of formulas.slice(0, limit)) {
    const result = await predictCrystalStructure(emit, formula);
    if (result) predicted++;
    await new Promise(r => setTimeout(r, 500));
  }

  return predicted;
}

const CHEMICAL_SUBSTITUTION_MAP: Record<string, string[]> = {
  "La": ["Y", "Ce", "Pr", "Nd", "Gd"],
  "Y": ["La", "Sc", "Lu", "Ho"],
  "Ba": ["Sr", "Ca", "Mg"],
  "Sr": ["Ba", "Ca"],
  "Cu": ["Ni", "Co", "Fe", "Zn"],
  "Fe": ["Co", "Mn", "Ni", "Ru"],
  "Ti": ["Zr", "Hf", "V"],
  "Nb": ["Ta", "V", "Mo"],
  "As": ["P", "Sb"],
  "Se": ["S", "Te"],
  "O": ["S", "Se", "F"],
  "H": ["F", "B"],
  "B": ["C", "N", "Al"],
  "N": ["P", "C", "B"],
};

const UNUSUAL_TOPOLOGIES = [
  { name: "Kagome", spaceGroup: "P6/mmm", crystalSystem: "hexagonal", dimensionality: "quasi-2D", description: "Corner-sharing triangular lattice with geometric frustration" },
  { name: "Pyrochlore", spaceGroup: "Fd-3m", crystalSystem: "cubic", dimensionality: "3D", description: "Corner-sharing tetrahedra with geometric frustration" },
  { name: "Mixed-2D-1D", spaceGroup: "Cmcm", crystalSystem: "orthorhombic", dimensionality: "mixed", description: "2D conducting sheets connected by 1D chain bridges" },
  { name: "Breathing-kagome", spaceGroup: "P-3m1", crystalSystem: "hexagonal", dimensionality: "quasi-2D", description: "Alternating large/small triangles with flat bands" },
  { name: "Honeycomb", spaceGroup: "P6_3/mmc", crystalSystem: "hexagonal", dimensionality: "2D", description: "Graphene-like honeycomb connectivity" },
];

const INTERCALANTS = ["H", "Li", "Na", "K"];

export interface GeneratedStructureVariant {
  formula: string;
  parentFormula: string;
  variationType: string;
  structuralNovelty: number;
  topology: string;
  spaceGroup: string;
  crystalSystem: string;
  dimensionality: string;
  description: string;
  suggestedElements?: string[];
}

const SC_ACTIVE_ELEMENTS = new Set([
  "Cu", "Fe", "Ni", "Co", "Mn", "Ru", "Ir", "Nb", "V", "Ti",
]);

function countsToFormula(cts: Record<string, number>): string {
  return Object.entries(cts)
    .filter(([, n]) => typeof n === "number" && Number.isFinite(n) && n > 0)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([el, n]) => {
      const rounded = Math.round(n);
      return rounded === 1 ? el : `${el}${rounded}`;
    })
    .join("");
}

function generateSubstitutionVariant(formula: string, comp?: ParsedComposition): GeneratedStructureVariant | null {
  const { elements, counts } = comp || parseComposition(formula);

  const scActiveInFormula = elements.filter(e => SC_ACTIVE_ELEMENTS.has(e));
  const substitutable = elements.filter(e =>
    CHEMICAL_SUBSTITUTION_MAP[e] &&
    CHEMICAL_SUBSTITUTION_MAP[e].length > 0 &&
    !(SC_ACTIVE_ELEMENTS.has(e) && scActiveInFormula.length <= 1)
  );
  if (substitutable.length === 0) return null;

  const targetEl = substitutable[Math.floor(Math.random() * substitutable.length)];
  const subs = CHEMICAL_SUBSTITUTION_MAP[targetEl];
  const availableSubs = subs.filter(s => !elements.includes(s));
  if (availableSubs.length === 0) return null;

  const newEl = availableSubs[Math.floor(Math.random() * availableSubs.length)];
  const newCounts: Record<string, number> = { ...counts };
  newCounts[newEl] = (newCounts[newEl] || 0) + (newCounts[targetEl] || 1);
  delete newCounts[targetEl];

  const newFormula = countsToFormula(newCounts);
  if (!newFormula) return null;

  const parentProto = matchPrototype(formula, comp);
  const proto = parentProto || { spaceGroup: "P1", crystalSystem: "triclinic", dimensionality: "3D", prototype: "unknown" };

  return {
    formula: newFormula,
    parentFormula: formula,
    variationType: "substitution",
    structuralNovelty: 0.3 + Math.random() * 0.3,
    topology: `${targetEl}->${newEl} substitution`,
    spaceGroup: proto.spaceGroup,
    crystalSystem: proto.crystalSystem,
    dimensionality: proto.dimensionality,
    description: `Chemical substitution: ${targetEl} replaced by ${newEl} in ${formula}`,
  };
}

function generateIntercalationVariant(formula: string, comp?: ParsedComposition): GeneratedStructureVariant | null {
  const resolved = comp || parseComposition(formula);
  const { elements, counts } = resolved;

  const availableIntercalants = INTERCALANTS.filter(i => !elements.includes(i) || (counts[i] || 0) < 4);
  if (availableIntercalants.length === 0) return null;

  const intercalant = availableIntercalants[Math.floor(Math.random() * availableIntercalants.length)];
  const amount = 1 + Math.floor(Math.random() * 3);

  const newCounts: Record<string, number> = { ...counts };
  newCounts[intercalant] = (newCounts[intercalant] || 0) + amount;

  const newFormula = countsToFormula(newCounts);
  if (!newFormula) return null;

  const hostProto = matchPrototype(formula, resolved);
  const hostDim = hostProto?.dimensionality || "quasi-2D";
  const hostSg = hostProto?.spaceGroup || "P6_3/mmc";
  const hostCs = hostProto?.crystalSystem || "hexagonal";

  return {
    formula: newFormula,
    parentFormula: formula,
    variationType: "intercalation",
    structuralNovelty: 0.4 + Math.random() * 0.3,
    topology: `${intercalant} intercalation into ${hostDim === "3D" ? "cage" : "layered"} host`,
    spaceGroup: hostSg,
    crystalSystem: hostCs,
    dimensionality: hostDim,
    description: `Intercalation of ${amount} ${intercalant} atoms into ${formula}`,
  };
}

function generateTopologyVariant(formula: string, _comp?: ParsedComposition): GeneratedStructureVariant | null {
  const topology = UNUSUAL_TOPOLOGIES[Math.floor(Math.random() * UNUSUAL_TOPOLOGIES.length)];

  return {
    formula,
    parentFormula: formula,
    variationType: "topology-mapping",
    structuralNovelty: 0.6 + Math.random() * 0.3,
    topology: topology.name,
    spaceGroup: topology.spaceGroup,
    crystalSystem: topology.crystalSystem,
    dimensionality: topology.dimensionality,
    description: topology.description,
  };
}

function canonicalizeFormula(f: string): string {
  const cts = parseFormulaCounts(f);
  const sorted = Object.entries(cts)
    .filter(([, n]) => typeof n === "number" && Number.isFinite(n) && n > 0)
    .sort(([a], [b]) => a.localeCompare(b));
  return sorted.map(([el, n]) => (n === 1 ? el : `${el}${n}`)).join("");
}

export function generateStructuralVariants(
  formula: string,
  maxVariants: number = 3
): GeneratedStructureVariant[] {
  const comp = parseComposition(formula);
  const variants: GeneratedStructureVariant[] = [];
  const seenFormulas = new Set<string>([canonicalizeFormula(formula)]);

  const generators = [generateSubstitutionVariant, generateIntercalationVariant, generateTopologyVariant];

  for (const gen of generators) {
    if (variants.length >= maxVariants) break;
    const variant = gen(formula, comp);
    if (!variant) continue;
    const canon = canonicalizeFormula(variant.formula);
    if (!seenFormulas.has(canon)) {
      seenFormulas.add(canon);
      variants.push(variant);
    }
  }

  return variants;
}

let totalStructuralVariantsGenerated = 0;
let totalNovelPrototypesGenerated = 0;

export function getStructuralVariantCount(): number {
  return totalStructuralVariantsGenerated;
}

export function getNovelPrototypeCount(): number {
  return totalNovelPrototypesGenerated;
}

export const DESIGN_PRINCIPLES = [
  "Alternating flat-band and dispersive-band layers",
  "Kagome planes separated by rattler-atom spacers",
  "Cage structure with hydrogen channels along one crystallographic axis",
  "Breathing pyrochlore with alternating large/small tetrahedra",
  "Mixed-valence ladder structure with spin-liquid ground state",
  "Honeycomb lattice with strong spin-orbit coupling and orbital degeneracy",
  "Layered structure with van Hove singularity near Fermi level",
  "Skutterudite-like cage with guest-atom rattling modes for enhanced phonon coupling",
  "Bipartite lattice with sublattice-selective orbital ordering",
  "Intercalated graphite-like sheets with heavy-metal spacer layers",
];

const VALID_SPACE_GROUPS = new Set([
  "P1", "P-1", "P2", "P21", "C2", "Pm", "Pc", "Cm", "Cc", "P2/m", "P21/m", "C2/m", "P2/c", "P21/c", "C2/c",
  "P222", "P2221", "P21212", "P212121", "C2221", "C222", "F222", "I222", "I212121",
  "Pmm2", "Pmc21", "Pcc2", "Pma2", "Pca21", "Pnc2", "Pmn21", "Pba2", "Pna21", "Pnn2",
  "Cmm2", "Cmc21", "Ccc2", "Amm2", "Abm2", "Ama2", "Aba2", "Fmm2", "Fdd2", "Imm2", "Iba2", "Ima2",
  "Pmmm", "Pnnn", "Pccm", "Pban", "Pmma", "Pnna", "Pmna", "Pcca", "Pbam", "Pccn", "Pbcm", "Pnnm",
  "Pmmn", "Pbcn", "Pbca", "Pnma", "Cmcm", "Cmca", "Cmmm", "Cccm", "Cmma", "Ccca",
  "Fmmm", "Fddd", "Immm", "Ibam", "Ibca", "Imma",
  "P4", "P41", "P42", "P43", "I4", "I41", "P-4", "I-4", "P4/m", "P42/m", "P4/n", "P42/n", "I4/m", "I41/a",
  "P422", "P4212", "P4122", "P41212", "P4222", "P42212", "P4322", "P43212", "I422", "I4122",
  "P4mm", "P4bm", "P42cm", "P42nm", "P4cc", "P4nc", "P42mc", "P42bc", "I4mm", "I4cm", "I41md", "I41cd",
  "P-42m", "P-42c", "P-421m", "P-421c", "P-4m2", "P-4c2", "P-4b2", "P-4n2", "I-4m2", "I-4c2", "I-42m", "I-42d",
  "P4/mmm", "P4/mcc", "P4/nbm", "P4/nnc", "P4/mbm", "P4/mnc", "P4/nmm", "P4/ncc",
  "P42/mmc", "P42/mcm", "P42/nbc", "P42/nnm", "P42/mbc", "P42/mnm", "P42/nmc", "P42/ncm",
  "I4/mmm", "I4/mcm", "I41/amd", "I41/acd",
  "P3", "P31", "P32", "R3", "P-3", "R-3",
  "P312", "P321", "P3112", "P3121", "P3212", "P3221", "R32",
  "P3m1", "P31m", "P3c1", "P31c", "R3m", "R3c",
  "P-31m", "P-3m1", "P-31c", "P-3c1", "R-3m", "R-3c",
  "P6", "P61", "P65", "P62", "P64", "P63", "P-6", "P6/m", "P63/m",
  "P622", "P6122", "P6522", "P6222", "P6422", "P6322",
  "P6mm", "P6cc", "P63cm", "P63mc",
  "P-6m2", "P-6c2", "P-62m", "P-62c",
  "P6/mmm", "P6/mcc", "P63/mcm", "P63/mmc",
  "P23", "F23", "I23", "P213", "I213",
  "Pm-3", "Pn-3", "Fm-3", "Fd-3", "Im-3", "Pa-3", "Ia-3",
  "P432", "P4232", "F432", "F4132", "I432", "P4332", "P4132", "I4132",
  "P-43m", "F-43m", "I-43m", "P-43n", "F-43c", "I-43d",
  "Pm-3m", "Pn-3n", "Pm-3n", "Pn-3m", "Fm-3m", "Fm-3c", "Fd-3m", "Fd-3c", "Im-3m", "Ia-3d",
]);

function isValidSpaceGroup(sg: string): boolean {
  return VALID_SPACE_GROUPS.has(sg);
}

export interface WyckoffSiteCoords {
  label: string;
  fractionalCoords: [number, number, number];
  speculative: boolean;
}

export interface StructuralEnrichment {
  spaceGroup: string;
  crystalSystem: string;
  latticeRatios: { a: number; b: number; c: number };
  wyckoffSites: Record<string, WyckoffSiteCoords>;
  coordinationNumbers?: Record<string, number>;
  needsExternalRelaxation: boolean;
}

const WYCKOFF_APPROX_COORDS: Record<string, [number, number, number]> = {
  "1a": [0, 0, 0],
  "1b": [0.5, 0.5, 0.5],
  "2a": [0, 0, 0],
  "2b": [0.5, 0.5, 0.5],
  "2c": [0, 0.5, 0],
  "2d": [0.5, 0, 0.5],
  "4a": [0, 0, 0],
  "4b": [0.5, 0.5, 0.5],
  "4c": [0.25, 0.25, 0.25],
  "4d": [0.75, 0.75, 0.75],
  "4e": [0, 0, 0.25],
  "6b": [0, 0.5, 0.5],
  "8c": [0.25, 0.25, 0.25],
  "8d": [0.375, 0.375, 0.375],
  "3a": [0, 0, 0],
  "3b": [0, 0, 0.5],
  "6c": [0.333, 0.667, 0.25],
};

export interface NovelPrototype {
  name: string;
  spaceGroup: string;
  crystalSystem: string;
  dimensionality: string;
  designPrinciple: string;
  wyckoffPositions: Record<string, string>;
  latticeRatios: { a: number; b: number; c: number };
  physicsRationale: string;
  noveltyScore: number;
  suggestedElements: string[];
  structuralEnrichment?: StructuralEnrichment;
}

function computePrototypeNoveltyScore(
  spaceGroup: string,
  crystalSystem: string,
  dimensionality: string,
  latticeRatios: { a: number; b: number; c: number }
): number {
  const knownSGs = new Set(Object.values(KNOWN_PROTOTYPES).map(p => p.spaceGroup));
  const knownCS = new Set(Object.values(KNOWN_PROTOTYPES).map(p => p.crystalSystem));
  const knownDims = new Set(Object.values(KNOWN_PROTOTYPES).map(p => p.dimensionality));

  let score = 0;

  const normalizedSG = normalizeSpaceGroup(spaceGroup);
  if (!knownSGs.has(normalizedSG)) {
    score += 0.4;
  } else {
    score += 0.1;
  }

  if (!knownCS.has(crystalSystem)) {
    score += 0.2;
  }

  if (!knownDims.has(dimensionality)) {
    score += 0.2;
  }

  const knownCARatios = Object.values(PROTOTYPE_CA_RATIOS);
  const ca = latticeRatios.c / latticeRatios.a;
  const minDist = Math.min(...knownCARatios.map(r => Math.abs(ca - r)));
  score += Math.min(0.2, minDist * 0.1);

  return Math.min(1.0, Math.max(0.1, score));
}

export async function generateNovelPrototype(
  emit: EventEmitter,
  designPrinciple?: string
): Promise<NovelPrototype | null> {
  const principle = designPrinciple || DESIGN_PRINCIPLES[Math.floor(Math.random() * DESIGN_PRINCIPLES.length)];

  const knownList = Object.entries(KNOWN_PROTOTYPES)
    .map(([name, p]) => `${name}: ${p.spaceGroup} (${p.crystalSystem}, ${p.dimensionality})`)
    .join("\n");

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are an expert computational materials scientist specializing in crystal structure design for superconductors.

Your task: propose a NOVEL crystal structure prototype that does NOT match any of these known prototypes:
${knownList}

Design principle to exploit: "${principle}"

Requirements:
- space_group: Must be a valid International Tables for Crystallography (ITC) Hermann-Mauguin symbol
- lattice_a, lattice_b, lattice_c: Lattice parameter ratios (normalized so a=1.0). Must be physically reasonable (ratios between 0.5 and 10.0)
- crystal_system: One of cubic, tetragonal, hexagonal, orthorhombic, monoclinic, triclinic, trigonal, rhombohedral
- dimensionality: One of 3D, quasi-2D, 2D, 1D, mixed
- wyckoff_positions: Object mapping element roles (e.g. "metal_A", "metal_B", "anion") to Wyckoff site labels (e.g. "4a", "8c", "2b")
- name: A short descriptive name for this prototype
- physics_rationale: ONE sentence explaining why this structure might support superconductivity
- suggested_elements: Array of 3-5 element symbols that would be good candidates for this structure
- coordination_numbers: Object mapping element roles to their coordination numbers (must be chemically valid, 2-12)

Return JSON with these fields. Keep physics_rationale to exactly one sentence to ensure all structural fields fit within the response.`,
        },
        {
          role: "user",
          content: `Design a novel crystal structure prototype exploiting: "${principle}"`,
        },
      ],
      response_format: { type: "json_object" },
      max_completion_tokens: 600,
    });

    const content = response.choices[0]?.message?.content;
    if (!content) return null;

    const parsed = JSON.parse(content);

    const spaceGroup = typeof parsed.space_group === "string" ? parsed.space_group : "P1";
    const crystalSystem = typeof parsed.crystal_system === "string" ? parsed.crystal_system.toLowerCase() : "triclinic";
    const dimensionality = typeof parsed.dimensionality === "string" ? parsed.dimensionality : "3D";
    const name = typeof parsed.name === "string" ? parsed.name : `novel-${Date.now()}`;
    const physicsRationale = typeof parsed.physics_rationale === "string" ? parsed.physics_rationale : "";
    const suggestedElements = Array.isArray(parsed.suggested_elements) ? parsed.suggested_elements.filter((e: unknown) => typeof e === "string") : [];

    const toNum = (v: unknown, fallback: number) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    };

    let latticeA = toNum(parsed.lattice_a, 1.0);
    let latticeB = toNum(parsed.lattice_b, 1.0);
    let latticeC = toNum(parsed.lattice_c, 1.0);

    latticeA = Math.max(0.5, Math.min(10.0, latticeA));
    latticeB = Math.max(0.5, Math.min(10.0, latticeB));
    latticeC = Math.max(0.5, Math.min(10.0, latticeC));

    const csLower = crystalSystem.toLowerCase();
    if (csLower === "cubic") {
      latticeB = latticeA;
      latticeC = latticeA;
    } else if (csLower === "tetragonal" || csLower === "hexagonal" || csLower === "trigonal") {
      latticeB = latticeA;
    } else if (csLower === "rhombohedral") {
      latticeB = latticeA;
      latticeC = latticeA;
    }

    const validSG = isValidSpaceGroup(spaceGroup) ? spaceGroup : "P1";

    const WYCKOFF_PATTERN = /^(\d+)([a-zA-Z])$/;
    const LOW_SYMMETRY_MAX_MULT: Record<string, number> = {
      "P1": 1, "P-1": 2,
    };
    const maxMult = LOW_SYMMETRY_MAX_MULT[validSG] ?? 192;

    const wyckoff: Record<string, string> = {};
    let hasSpeculativeWyckoff = false;
    if (parsed.wyckoff_positions && typeof parsed.wyckoff_positions === "object") {
      for (const [role, site] of Object.entries(parsed.wyckoff_positions)) {
        if (typeof site === "string") {
          const wMatch = site.match(WYCKOFF_PATTERN);
          if (wMatch) {
            const mult = parseInt(wMatch[1], 10);
            if (mult > maxMult) {
              wyckoff[role] = `${site} [speculative]`;
              hasSpeculativeWyckoff = true;
            } else {
              wyckoff[role] = site;
            }
          } else {
            wyckoff[role] = `${site} [speculative]`;
            hasSpeculativeWyckoff = true;
          }
        }
      }
    }
    if (hasSpeculativeWyckoff) {
      emit("log", {
        phase: "phase-11",
        event: "Wyckoff positions flagged as speculative",
        detail: `${name}: some Wyckoff labels may be incompatible with ${validSG}, flagged [speculative]`,
        dataSource: "Novel Prototype Generator",
      });
    }

    const coordNums = parsed.coordination_numbers;
    if (coordNums && typeof coordNums === "object") {
      for (const [role, cn] of Object.entries(coordNums)) {
        const cnNum = Number(cn);
        if (!Number.isFinite(cnNum) || cnNum < 2 || cnNum > 12) {
          emit("log", {
            phase: "phase-11",
            event: "Novel prototype coordination warning",
            detail: `Role ${role} has invalid coordination number ${cn}, structure may not be chemically valid`,
            dataSource: "Novel Prototype Generator",
          });
        }
      }
    }

    const latticeRatios = { a: latticeA, b: latticeB, c: latticeC };
    const noveltyScore = computePrototypeNoveltyScore(validSG, crystalSystem, dimensionality, latticeRatios);

    const normalizedValidSG = normalizeSpaceGroup(validSG);
    const isKnownDuplicate = Object.values(KNOWN_PROTOTYPES).some(
      p => p.spaceGroup === normalizedValidSG && p.crystalSystem === crystalSystem && p.dimensionality === dimensionality
    );

    if (isKnownDuplicate) {
      emit("log", {
        phase: "phase-11",
        event: "Novel prototype rejected (matches known)",
        detail: `${name}: ${validSG} (${crystalSystem}, ${dimensionality}) matches an existing known prototype`,
        dataSource: "Novel Prototype Generator",
      });
      return null;
    }

    const wyckoffSites: Record<string, WyckoffSiteCoords> = {};
    let needsRelax = false;
    for (const [role, siteLabel] of Object.entries(wyckoff)) {
      const cleanLabel = siteLabel.replace(/\s*\[speculative\]/, "");
      const isSpec = siteLabel.includes("[speculative]");
      const coords = WYCKOFF_APPROX_COORDS[cleanLabel];
      if (coords) {
        wyckoffSites[role] = { label: cleanLabel, fractionalCoords: coords, speculative: isSpec };
      } else {
        wyckoffSites[role] = { label: cleanLabel, fractionalCoords: [0, 0, 0], speculative: true };
        needsRelax = true;
      }
    }

    const parsedCoordNums: Record<string, number> = {};
    if (coordNums && typeof coordNums === "object") {
      for (const [role, cn] of Object.entries(coordNums)) {
        const cnNum = Number(cn);
        if (Number.isFinite(cnNum) && cnNum >= 2 && cnNum <= 12) {
          parsedCoordNums[role] = cnNum;
        }
      }
    }

    const enrichment: StructuralEnrichment = {
      spaceGroup: validSG,
      crystalSystem,
      latticeRatios,
      wyckoffSites,
      coordinationNumbers: Object.keys(parsedCoordNums).length > 0 ? parsedCoordNums : undefined,
      needsExternalRelaxation: needsRelax || hasSpeculativeWyckoff,
    };

    const prototype: NovelPrototype = {
      name,
      spaceGroup: validSG,
      crystalSystem,
      dimensionality,
      designPrinciple: principle,
      wyckoffPositions: wyckoff,
      latticeRatios,
      physicsRationale,
      noveltyScore,
      suggestedElements,
      structuralEnrichment: enrichment,
    };

    totalNovelPrototypesGenerated++;

    emit("log", {
      phase: "phase-11",
      event: "Novel crystal prototype generated",
      detail: `${name}: ${validSG} (${crystalSystem}, ${dimensionality}), principle="${principle}", novelty=${noveltyScore.toFixed(2)}, elements=[${suggestedElements.join(",")}]`,
      dataSource: "Novel Prototype Generator",
    });

    return prototype;
  } catch (err: any) {
    emit("log", {
      phase: "phase-11",
      event: "Novel prototype generation error",
      detail: err.message?.slice(0, 150) || "unknown error",
      dataSource: "Novel Prototype Generator",
    });
    return null;
  }
}

export async function runNovelPrototypeGeneration(
  emit: EventEmitter,
): Promise<GeneratedStructureVariant[]> {
  const results: GeneratedStructureVariant[] = [];

  const principleIdx1 = Math.floor(Math.random() * DESIGN_PRINCIPLES.length);
  let principleIdx2 = (principleIdx1 + 1 + Math.floor(Math.random() * (DESIGN_PRINCIPLES.length - 1))) % DESIGN_PRINCIPLES.length;

  const principles = [DESIGN_PRINCIPLES[principleIdx1], DESIGN_PRINCIPLES[principleIdx2]];

  const protos = await Promise.all(principles.map(p => generateNovelPrototype(emit, p)));

  for (const proto of protos) {
    if (!proto || proto.suggestedElements.length < 2) continue;

    const wyckoffRoleCount = Object.keys(proto.wyckoffPositions).length;
    const requiredElements = Math.max(2, Math.min(proto.suggestedElements.length, wyckoffRoleCount || 4));
    const elements = proto.suggestedElements.slice(0, requiredElements);
    const STOICH_TEMPLATES = [
      [1, 1, 3],    // ABO3 perovskite
      [2, 1, 4],    // A2BO4 K2NiF4
      [1, 2, 3],    // AB2X3
      [1, 3, 1],    // AB3X
      [1, 1, 2],    // ABX2
      [2, 2, 7],    // A2B2X7
      [3, 1, 1],    // A3BX
    ];
    const template = STOICH_TEMPLATES[Math.floor(Math.random() * STOICH_TEMPLATES.length)];
    const formulaParts = elements.map((el, i) => {
      const count = template[i % template.length];
      return count === 1 ? el : `${el}${count}`;
    });
    const formula = formulaParts.join("");

    const canonFormula = canonicalizeFormula(formula);
    const existing = await storage.getSuperconductorByFormula(canonFormula);
    if (existing) continue;

    results.push({
      formula: canonFormula,
      parentFormula: "novel-prototype",
      variationType: "novel-prototype",
      structuralNovelty: proto.noveltyScore,
      topology: proto.name,
      spaceGroup: proto.spaceGroup,
      crystalSystem: proto.crystalSystem,
      dimensionality: proto.dimensionality,
      description: `Novel prototype "${proto.name}": ${proto.physicsRationale}. Design principle: ${proto.designPrinciple}`,
      suggestedElements: proto.suggestedElements,
    });
  }

  return results;
}

export async function runGenerativeStructureDiscovery(
  emit: EventEmitter,
  topCandidates: { formula: string; predictedTc: number; ensembleScore: number }[]
): Promise<GeneratedStructureVariant[]> {
  const allVariants: GeneratedStructureVariant[] = [];

  let selected: typeof topCandidates;
  if (Math.random() < 0.1 && topCandidates.length > 3) {
    const shuffled = [...topCandidates].sort(() => Math.random() - 0.5);
    selected = shuffled.slice(0, 3);
  } else {
    selected = [...topCandidates]
      .sort((a, b) => (b.ensembleScore ?? 0) - (a.ensembleScore ?? 0))
      .slice(0, 3);
  }

  for (const candidate of selected) {
    const variants = generateStructuralVariants(candidate.formula, 2);

    for (const variant of variants) {
      variant.formula = canonicalizeFormula(variant.formula);
      const existing = await storage.getSuperconductorByFormula(variant.formula);
      if (existing) continue;

      allVariants.push(variant);
      totalStructuralVariantsGenerated++;

      emit("log", {
        phase: "phase-11",
        event: "Structural variant generated",
        detail: `${variant.formula} from ${variant.parentFormula} via ${variant.variationType} (${variant.topology}, novelty=${variant.structuralNovelty.toFixed(2)})`,
        dataSource: "Structure Generator",
      });
    }
  }

  return allVariants;
}

let mutationIntensityLevel = 1;

const ATOM_SWAP_MAP_LEVEL1: Record<string, string[]> = {
  Fe: ["Co", "Ni", "Mn", "Cr"],
  Cu: ["Ni", "Ag", "Zn"],
  O: ["S", "Se", "Te"],
  As: ["P", "Sb", "Bi"],
  Ba: ["Sr", "Ca", "La"],
};

const ATOM_SWAP_MAP_LEVEL2: Record<string, string[]> = {
  Fe: ["Co", "Ni", "Mn", "Cr", "V", "Ti", "Ru", "Os"],
  Cu: ["Ni", "Ag", "Zn", "Pd", "Pt", "Au"],
  O: ["S", "Se", "Te", "N", "F"],
  As: ["P", "Sb", "Bi", "Sn", "Ge"],
  Ba: ["Sr", "Ca", "La", "K", "Rb", "Cs"],
  Nb: ["Ta", "V", "Mo", "W", "Ti", "Zr"],
  Ti: ["Zr", "Hf", "V", "Nb", "Sc"],
  Si: ["Ge", "Sn", "Al", "Ga"],
  B: ["C", "N", "Al", "Si"],
};

const ATOM_SWAP_MAP_LEVEL3: Record<string, string[]> = {
  Fe: ["Co", "Ni", "Mn", "Cr", "V", "Ti", "Ru", "Os", "Ir", "Rh", "Re", "W"],
  Cu: ["Ni", "Ag", "Zn", "Pd", "Pt", "Au", "Hg", "Tl", "In"],
  O: ["S", "Se", "Te", "N", "F", "Cl"],
  As: ["P", "Sb", "Bi", "Sn", "Ge", "Pb", "In", "Tl"],
  Ba: ["Sr", "Ca", "La", "K", "Rb", "Cs", "Eu", "Yb"],
  Nb: ["Ta", "V", "Mo", "W", "Ti", "Zr", "Hf", "Re"],
  Ti: ["Zr", "Hf", "V", "Nb", "Sc", "Y", "La"],
  Si: ["Ge", "Sn", "Al", "Ga", "In", "Pb"],
  B: ["C", "N", "Al", "Si", "P", "Ga"],
  La: ["Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Y", "Sc"],
  C: ["N", "B", "Si", "P"],
  N: ["C", "O", "P", "As"],
};

function getAtomSwapMap(): Record<string, string[]> {
  if (mutationIntensityLevel >= 3) return ATOM_SWAP_MAP_LEVEL3;
  if (mutationIntensityLevel >= 2) return ATOM_SWAP_MAP_LEVEL2;
  return ATOM_SWAP_MAP_LEVEL1;
}

export function setMutationIntensity(level: number): void {
  mutationIntensityLevel = Math.max(1, Math.min(3, level));
}

export function getMutationIntensity(): number {
  return mutationIntensityLevel;
}

const ATOM_SWAP_MAP: Record<string, string[]> = {
  Fe: ["Co", "Ni", "Mn", "Cr"],
  Cu: ["Ni", "Ag", "Zn"],
  O: ["S", "Se", "Te"],
  As: ["P", "Sb", "Bi"],
  Ba: ["Sr", "Ca", "La"],
  Y: ["La", "Sc", "Gd"],
  Se: ["S", "Te"],
  Nb: ["Ta", "V", "Mo"],
  B: ["C", "N"],
};

const EVO_INTERCALANTS = ["Li", "Na", "K", "H"];

function reconstructFormula(elements: string[], counts: Record<string, number>): string {
  const unique = Array.from(new Set(elements)).sort((a, b) => a.localeCompare(b));
  return unique
    .filter(e => counts[e] > 0)
    .map(e => counts[e] > 1 ? e + counts[e] : e)
    .join("");
}

const INTERCALATION_ALLOWED_DIM = new Set(["quasi-2D", "2D", "3D"]);

export async function runEvolutionaryStructureSearch(
  candidates: Array<{ formula: string; predictedTc: number; ensembleScore: number }>,
  emit: EventEmitter
): Promise<string[]> {
  let parents = [...candidates]
    .sort((a, b) => b.ensembleScore - a.ensembleScore)
    .slice(0, 5);

  for (let gen = 0; gen < 3; gen++) {
    const mutants: Array<{ formula: string; score: number; parentTc: number }> = [];

    for (const parent of parents) {
      const parentElements = parseFormulaElements(parent.formula);
      const parentCounts = parseFormulaCounts(parent.formula);
      const parentProto = matchPrototype(parent.formula);
      const parentDim = parentProto?.dimensionality || "3D";
      const intercalationAllowed = INTERCALATION_ALLOWED_DIM.has(parentDim);

      for (let m = 0; m < 4; m++) {
        const elements = [...parentElements];
        const counts: Record<string, number> = { ...parentCounts };

        if (m === 0) {
          const idx = Math.floor(Math.random() * elements.length);
          const el = elements[idx];
          const distortionRange = mutationIntensityLevel >= 3 ? 0.5 : mutationIntensityLevel >= 2 ? 0.4 : 0.3;
          const scale = (1 - distortionRange / 2) + Math.random() * distortionRange;
          counts[el] = Math.max(1, Math.round((counts[el] || 1) * scale));
        } else if (m === 1) {
          const activeSwapMap = getAtomSwapMap();
          const swappable = elements.filter(e => e !== "H" && activeSwapMap[e]);
          if (swappable.length > 0) {
            const el = swappable[Math.floor(Math.random() * swappable.length)];
            const replacements = activeSwapMap[el];
            if (el === "B" && Math.random() < 0.5) {
            } else {
              const replacement = replacements[Math.floor(Math.random() * replacements.length)];
              const elIdx = elements.indexOf(el);
              if (elIdx !== -1 && !elements.includes(replacement)) {
                elements[elIdx] = replacement;
                counts[replacement] = counts[el];
                delete counts[el];
              }
            }
          }
        } else if (m === 2) {
          if (!intercalationAllowed) continue;
          const available = EVO_INTERCALANTS.filter(i => !elements.includes(i));
          if (available.length > 0) {
            const intercalant = available[Math.floor(Math.random() * available.length)];
            elements.push(intercalant);
            counts[intercalant] = 1 + Math.floor(Math.random() * 2);
          }
        } else {
          if (!intercalationAllowed) {
            const idx = Math.floor(Math.random() * elements.length);
            const el = elements[idx];
            const scale = 0.85 + Math.random() * 0.3;
            counts[el] = Math.max(1, Math.round((counts[el] || 1) * scale));
          } else {
            const idx = Math.floor(Math.random() * elements.length);
            const el = elements[idx];
            const scale = 0.85 + Math.random() * 0.3;
            counts[el] = Math.max(1, Math.round((counts[el] || 1) * scale));

            const elSet = new Set(elements);
            const available = EVO_INTERCALANTS.filter(i => !elSet.has(i));
            if (available.length > 0) {
              const intercalant = available[Math.floor(Math.random() * available.length)];
              if (!elSet.has(intercalant)) {
                elements.push(intercalant);
              }
              counts[intercalant] = (counts[intercalant] || 0) + 1;
            }
          }
        }

        const formula = reconstructFormula(elements, counts);

        if (!passesValenceFilter(formula)) continue;

        const dimensionalityScore = computeDimensionalityScore(formula);
        const motifResult = detectStructuralMotifs(formula);

        const mutantFeatures = extractFeatures(formula);
        const lambda = mutantFeatures.electronPhononLambda ?? 0.5;
        const logOmega = mutantFeatures.logPhononFreq ?? 2.0;
        const muStar = 0.13;
        const mcMillanTc = lambda > muStar
          ? (Math.pow(10, logOmega) / 1.2) * Math.exp(-1.04 * (1 + lambda) / (lambda - muStar * (1 + 0.62 * lambda)))
          : 0;
        const predictedTc = Math.max(0, Math.min(300, mcMillanTc));

        const mutantScore = 0.4 * motifResult.motifScore + 0.25 * dimensionalityScore + 0.35 * (predictedTc / 400);

        mutants.push({ formula, score: mutantScore, parentTc: predictedTc });
      }

      await new Promise(resolve => setImmediate(resolve));
    }

    const sorted = mutants.sort((a, b) => b.score - a.score).slice(0, 5);

    const bestScore = sorted[0]?.score ?? 0;
    emit("log", {
      phase: "phase-11",
      event: `Evolutionary generation ${gen + 1}`,
      detail: `Best score: ${bestScore.toFixed(4)}, top formula: ${sorted[0]?.formula ?? "N/A"}`,
      dataSource: "Structure Evolution",
    });

    parents = sorted.map(s => ({
      formula: s.formula,
      predictedTc: s.parentTc,
      ensembleScore: s.score,
    }));
  }

  const uniqueFormulas = Array.from(new Set(parents.map(p => p.formula)));
  return uniqueFormulas;
}
