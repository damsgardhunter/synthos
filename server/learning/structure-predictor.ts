import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

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
}

const KNOWN_PROTOTYPES: Record<string, { spaceGroup: string; crystalSystem: string; prototype: string; dimensionality: string }> = {
  "perovskite": { spaceGroup: "Pm-3m", crystalSystem: "cubic", prototype: "CaTiO3", dimensionality: "3D" },
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
  "graphite": { spaceGroup: "P63/mmc", crystalSystem: "hexagonal", prototype: "C", dimensionality: "2D" },
  "bismuth-selenide": { spaceGroup: "R-3m", crystalSystem: "rhombohedral", prototype: "Bi2Se3", dimensionality: "2D" },
};

function parseFormulaElements(formula: string): string[] {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? [...new Set(matches)] : [];
}

function matchPrototype(formula: string): typeof KNOWN_PROTOTYPES[string] | null {
  const elements = parseFormulaElements(formula);

  if (elements.includes("Cu") && elements.includes("O") && elements.length >= 3) {
    if (elements.includes("Y") || elements.includes("Ba")) return KNOWN_PROTOTYPES["YBCO"];
    return KNOWN_PROTOTYPES["cuprate"];
  }
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("P") || elements.includes("Se"))) {
    return KNOWN_PROTOTYPES["iron-pnictide"];
  }
  if (elements.includes("H") && elements.length === 2) {
    return KNOWN_PROTOTYPES["clathrate"];
  }
  if (elements.includes("Mg") && elements.includes("B")) {
    return KNOWN_PROTOTYPES["MgB2"];
  }
  if (elements.includes("Nb") && elements.includes("Sn")) {
    return KNOWN_PROTOTYPES["A15"];
  }
  if (elements.includes("O") && elements.length === 3) {
    return KNOWN_PROTOTYPES["perovskite"];
  }

  return null;
}

export async function predictCrystalStructure(
  emit: EventEmitter,
  formula: string
): Promise<StructurePrediction | null> {
  const protoMatch = matchPrototype(formula);

  try {
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
5. Thermodynamic stability (energy above convex hull in eV/atom, 0 = stable)
6. Whether it could be metastable (kinetically trapped)
7. Synthesizability score (0-1) and brief synthesis notes

Return JSON with fields: spaceGroup, crystalSystem, latticeA, latticeB, latticeC, alpha, beta, gamma, prototype, dimensionality, decompositionEnergy (eV/atom above hull), isStable (boolean), isMetastable (boolean), convexHullDistance (eV/atom), synthesizability (0-1), synthesisNotes (string under 150 chars)`,
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

    const result: StructurePrediction = {
      spaceGroup: parsed.spaceGroup ?? protoMatch?.spaceGroup ?? "P1",
      crystalSystem: parsed.crystalSystem ?? protoMatch?.crystalSystem ?? "triclinic",
      latticeParams: {
        a: parsed.latticeA ?? 4.0,
        b: parsed.latticeB ?? 4.0,
        c: parsed.latticeC ?? 4.0,
        alpha: parsed.alpha ?? 90,
        beta: parsed.beta ?? 90,
        gamma: parsed.gamma ?? 90,
      },
      prototype: parsed.prototype ?? protoMatch?.prototype ?? "unknown",
      dimensionality: parsed.dimensionality ?? protoMatch?.dimensionality ?? "3D",
      isStable: parsed.isStable ?? false,
      isMetastable: parsed.isMetastable ?? false,
      decompositionEnergy: parsed.decompositionEnergy ?? 0.1,
      convexHullDistance: parsed.convexHullDistance ?? 0.05,
      synthesizability: parsed.synthesizability ?? 0.5,
      synthesisNotes: parsed.synthesisNotes ?? "Standard synthesis conditions expected",
    };

    const id = `cs-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
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
      source: "Structure Predictor",
    });

    emit("log", {
      phase: "phase-11",
      event: "Crystal structure predicted",
      detail: `${formula}: ${result.spaceGroup} (${result.crystalSystem}), ${result.dimensionality}, prototype: ${result.prototype}, hull: ${result.convexHullDistance} eV/atom`,
      dataSource: "Structure Predictor",
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

export function evaluateConvexHullStability(decompositionEnergy: number): {
  isStable: boolean;
  isMetastable: boolean;
  verdict: string;
} {
  if (decompositionEnergy <= 0.005) {
    return { isStable: true, isMetastable: false, verdict: "Thermodynamically stable (on convex hull)" };
  }
  if (decompositionEnergy <= 0.05) {
    return { isStable: false, isMetastable: true, verdict: "Near hull - likely synthesizable as metastable phase" };
  }
  if (decompositionEnergy <= 0.15) {
    return { isStable: false, isMetastable: true, verdict: "Moderately above hull - may be kinetically trapped" };
  }
  return { isStable: false, isMetastable: false, verdict: "Far above hull - unlikely to be synthesized" };
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
      scImplication = "Quasi-1D superconductivity prone to fluctuations; Peierls instability competition; unlikely for high Tc";
      break;
    default:
      scImplication = "Dimensionality unclear; standard analysis applies";
  }

  return { dim, scImplication };
}

export async function runStructurePredictionBatch(
  emit: EventEmitter,
  formulas: string[]
): Promise<number> {
  let predicted = 0;

  for (const formula of formulas.slice(0, 3)) {
    const result = await predictCrystalStructure(emit, formula);
    if (result) predicted++;
    await new Promise(r => setTimeout(r, 500));
  }

  return predicted;
}
