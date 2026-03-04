import OpenAI from "openai";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import { ELEMENTAL_DATA } from "./elemental-data";
import { fetchSummary, fetchElasticity } from "./materials-project-client";

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
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
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

function computeMiedemaFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

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
      const nwsA = dA.miedemaNws13;
      const nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvg = (nwsA + nwsB) / 2;

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

      const vAvg = (vA * fractions[elA] + vB * fractions[elB]) / (fractions[elA] + fractions[elB]);

      const interfaceEnergy = -P * deltaPhi * deltaPhi + Q * deltaNws * deltaNws - R_P;
      const contribution = fAB * vAvg * interfaceEnergy / (nwsAvg * nwsAvg);

      deltaH += contribution;
    }
  }

  return deltaH / totalAtoms;
}

function estimateDecompositionEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);

  if (elements.length < 2) return 0;

  const compoundEnergy = computeMiedemaFormationEnergy(formula);

  if (elements.length === 2) {
    return Math.max(0, compoundEnergy);
  }

  let bestBinarySum = 0;
  let binaryCount = 0;

  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const elA = elements[i];
      const elB = elements[j];
      const binaryFormula = `${elA}${elB}`;
      const binaryEnergy = computeMiedemaFormationEnergy(binaryFormula);
      if (binaryEnergy < 0) {
        bestBinarySum += binaryEnergy;
        binaryCount++;
      }
    }
  }

  if (binaryCount === 0) {
    return Math.max(0, compoundEnergy);
  }

  const avgBinaryEnergy = bestBinarySum / binaryCount;
  const decomp = compoundEnergy - avgBinaryEnergy;

  return Math.max(0, decomp);
}

export async function evaluateConvexHullStability(
  decompositionEnergy: number,
  formula?: string
): Promise<{
  isStable: boolean;
  isMetastable: boolean;
  verdict: string;
  formationEnergy: number;
  hullDistance: number;
  source: string;
}> {
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
        };
      }
    } catch {
    }

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
      verdict,
      formationEnergy: miedemaFormE,
      hullDistance: effectiveDecomp,
      source: "Miedema model",
    };
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
  };
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

function estimateLatticeFromVolume(volume: number, nsites: number, crystalSystem: string): { a: number; b: number; c: number } {
  const volPerAtom = volume / Math.max(nsites, 1);
  const cubeRoot = Math.pow(volPerAtom, 1 / 3);

  switch (crystalSystem.toLowerCase()) {
    case "cubic":
      return { a: cubeRoot, b: cubeRoot, c: cubeRoot };
    case "tetragonal":
      return { a: cubeRoot, b: cubeRoot, c: cubeRoot * 1.2 };
    case "hexagonal":
    case "trigonal":
      return { a: cubeRoot * 1.05, b: cubeRoot * 1.05, c: cubeRoot * 1.6 };
    case "orthorhombic":
      return { a: cubeRoot * 0.95, b: cubeRoot * 1.0, c: cubeRoot * 1.1 };
    case "monoclinic":
      return { a: cubeRoot * 0.9, b: cubeRoot * 1.0, c: cubeRoot * 1.15 };
    default:
      return { a: cubeRoot, b: cubeRoot, c: cubeRoot };
  }
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

function estimateSynthesizability(hullDistance: number, formationEnergy: number, elements: string[]): { score: number; notes: string } {
  let score = 1.0;
  const notes: string[] = [];

  if (hullDistance > 0.3) {
    score -= 0.5;
    notes.push("far above convex hull");
  } else if (hullDistance > 0.1) {
    score -= 0.25;
    notes.push("moderately above hull");
  } else if (hullDistance > 0.05) {
    score -= 0.1;
    notes.push("slightly above hull");
  }

  if (formationEnergy > 0) {
    score -= 0.2;
    notes.push("positive formation energy");
  }

  const toxic = ["Tl", "Pb", "Cd", "Hg", "Be", "As"];
  const rare = ["Re", "Os", "Ir", "Ru", "Rh", "Pd", "Pt", "Au"];
  const radioactive = ["Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"];

  for (const el of elements) {
    if (radioactive.includes(el)) { score -= 0.3; notes.push(`radioactive element ${el}`); break; }
    if (rare.includes(el)) { score -= 0.1; notes.push(`rare/expensive element ${el}`); break; }
    if (toxic.includes(el)) { score -= 0.05; notes.push(`toxic element ${el}`); break; }
  }

  const hasH = elements.includes("H");
  if (hasH) {
    score -= 0.15;
    notes.push("hydride may require high pressure");
  }

  score = Math.max(0.05, Math.min(1.0, score));
  const noteStr = notes.length > 0 ? notes.join("; ") : "Standard synthesis conditions expected";

  return { score, notes: noteStr };
}

export async function predictCrystalStructure(
  emit: EventEmitter,
  formula: string
): Promise<StructurePrediction | null> {
  const protoMatch = matchPrototype(formula);
  const elements = parseFormulaElements(formula);

  try {
    const mpData = await fetchSummary(formula);

    if (mpData) {
      const cs = mpData.crystalSystem || protoMatch?.crystalSystem || "cubic";
      const sg = mpData.spaceGroup || protoMatch?.spaceGroup || "P1";

      const lattice = estimateLatticeFromVolume(mpData.volume, mpData.nsites, cs);
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

      const synth = estimateSynthesizability(eAboveHull, formE, elements);

      const proto = protoMatch?.prototype || `${cs} (${sg})`;

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
      });

      emit("log", {
        phase: "phase-11",
        event: "Crystal structure predicted (MP data)",
        detail: `${formula}: ${sg} (${cs}), hull=${eAboveHull.toFixed(3)} eV/atom, dHf=${formE.toFixed(3)} eV/atom`,
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
    const synth = estimateSynthesizability(miedemaDecomp, miedemaFormE, elements);

    const result: StructurePrediction = {
      spaceGroup: (typeof parsed.spaceGroup === "string" && parsed.spaceGroup) || protoMatch?.spaceGroup || "P1",
      crystalSystem: (typeof parsed.crystalSystem === "string" && parsed.crystalSystem) || protoMatch?.crystalSystem || "triclinic",
      latticeParams: {
        a: toNum(parsed.latticeA, 4.0),
        b: toNum(parsed.latticeB, 4.0),
        c: toNum(parsed.latticeC, 4.0),
        alpha: toNum(parsed.alpha, 90),
        beta: toNum(parsed.beta, 90),
        gamma: toNum(parsed.gamma, 90),
      },
      prototype: (typeof parsed.prototype === "string" && parsed.prototype) || protoMatch?.prototype || "unknown",
      dimensionality: (typeof parsed.dimensionality === "string" && parsed.dimensionality) || protoMatch?.dimensionality || "3D",
      isStable,
      isMetastable,
      decompositionEnergy: miedemaDecomp,
      convexHullDistance: miedemaDecomp,
      synthesizability: synth.score,
      synthesisNotes: synth.notes,
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
    });

    emit("log", {
      phase: "phase-11",
      event: "Crystal structure predicted (Miedema)",
      detail: `${formula}: ${result.spaceGroup} (${result.crystalSystem}), Miedema dHf=${miedemaFormE.toFixed(3)} eV/atom, decomp=${miedemaDecomp.toFixed(3)} eV/atom`,
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
