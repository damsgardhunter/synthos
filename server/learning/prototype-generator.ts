import { getElementData } from "./elemental-data";

function gcdFloat(a: number, b: number): number {
  const eps = 1e-9;
  a = Math.abs(a);
  b = Math.abs(b);
  while (b > eps) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function radiusOf(el: string): number {
  return getElementData(el)?.atomicRadius ?? 130;
}

function sizeCompatible(candidate: string, reference: string, tolerance: number = 0.15): boolean {
  const rCand = radiusOf(candidate);
  const rRef = radiusOf(reference);
  return Math.abs(rCand - rRef) / rRef <= tolerance;
}

export interface PrototypeCandidate {
  formula: string;
  prototype: string;
  spaceGroup: string;
  crystalSystem: string;
  dimensionality: string;
  siteAssignment: Record<string, string[]>;
}

export interface PrototypeInfo {
  name: string;
  spaceGroup: string;
  crystalSystem: string;
  dimensionality: string;
  stoichiometry: string;
  sites: string[];
}

const PROTOTYPE_REGISTRY: Record<string, PrototypeInfo> = {
  AlB2: {
    name: "AlB2-type",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    dimensionality: "quasi-2D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  Perovskite: {
    name: "Perovskite",
    spaceGroup: "Pm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "ABO3",
    sites: ["A", "B", "O"],
  },
  A15: {
    name: "A15-type",
    spaceGroup: "Pm-3n",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A3B",
    sites: ["A", "B"],
  },
  Clathrate: {
    name: "Clathrate",
    spaceGroup: "Im-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "MHx",
    sites: ["M", "H"],
  },
  ThCr2Si2: {
    name: "ThCr2Si2-type",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    dimensionality: "quasi-2D",
    stoichiometry: "AB2C2",
    sites: ["A", "B", "C"],
  },
  Spinel: {
    name: "Spinel",
    spaceGroup: "Fd-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB2O4",
    sites: ["A", "B", "O"],
  },
  MAX: {
    name: "MAX-phase",
    spaceGroup: "P63/mmc",
    crystalSystem: "hexagonal",
    dimensionality: "quasi-2D",
    stoichiometry: "Mn+1AXn",
    sites: ["M", "A", "X"],
  },
  LayeredNitride: {
    name: "Layered nitride",
    spaceGroup: "P-3m1",
    crystalSystem: "hexagonal",
    dimensionality: "2D",
    stoichiometry: "AxMNX",
    sites: ["A", "M", "N", "X"],
  },
  Laves: {
    name: "Laves",
    spaceGroup: "Fd-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  Heusler: {
    name: "Heusler",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A2BC",
    sites: ["A", "B", "C"],
  },
  RockSalt: {
    name: "Rock-salt",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  Fluorite: {
    name: "Fluorite",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  Cuprate1212: {
    name: "Cuprate-1212",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    dimensionality: "2D",
    stoichiometry: "ABC2O7",
    sites: ["A", "B", "C", "O"],
  },
  Pychlore: {
    name: "Pyrochlore",
    spaceGroup: "Fd-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A2B2O7",
    sites: ["A", "B", "O"],
  },
  Delafossite: {
    name: "Delafossite",
    spaceGroup: "R-3m",
    crystalSystem: "trigonal",
    dimensionality: "2D",
    stoichiometry: "ABO2",
    sites: ["A", "B", "O"],
  },
  RuddlesdenPopper: {
    name: "Ruddlesden-Popper",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    dimensionality: "quasi-2D",
    stoichiometry: "A2BO4",
    sites: ["A", "B", "O"],
  },
  AntiPerovskite: {
    name: "Anti-perovskite",
    spaceGroup: "Pm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A3BC",
    sites: ["A", "B", "C"],
  },
  ZincBlende: {
    name: "Zinc-blende",
    spaceGroup: "F-43m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  Wurtzite: {
    name: "Wurtzite",
    spaceGroup: "P63mc",
    crystalSystem: "hexagonal",
    dimensionality: "3D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  NickelArsenide: {
    name: "NiAs-type",
    spaceGroup: "P63/mmc",
    crystalSystem: "hexagonal",
    dimensionality: "3D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  CadmiumIodide: {
    name: "CdI2-type",
    spaceGroup: "P-3m1",
    crystalSystem: "hexagonal",
    dimensionality: "2D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  C11b: {
    name: "MoSi2-type",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  Fulleride: {
    name: "Fulleride-K3C60",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A3C60",
    sites: ["A", "C"],
  },
  Chevrel: {
    name: "Chevrel-phase",
    spaceGroup: "R-3",
    crystalSystem: "trigonal",
    dimensionality: "3D",
    stoichiometry: "AMo6X8",
    sites: ["A", "Mo", "X"],
  },
  HalfHeusler: {
    name: "Half-Heusler",
    spaceGroup: "F-43m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "ABC",
    sites: ["A", "B", "C"],
  },
  FeSeType: {
    name: "FeSe-type",
    spaceGroup: "P4/nmm",
    crystalSystem: "tetragonal",
    dimensionality: "2D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  Skutterudite: {
    name: "Skutterudite",
    spaceGroup: "Im-3",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB3",
    sites: ["A", "B"],
  },
  KagomeMetal: {
    name: "AV3Sb5-type",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    dimensionality: "2D",
    stoichiometry: "A3BC5",
    sites: ["A", "B", "C"],
  },
  H3SType: {
    name: "H3S-type",
    spaceGroup: "Im-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AH3",
    sites: ["A", "H"],
  },
  LaH10Type: {
    name: "LaH10-clathrate",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AH10",
    sites: ["A", "H"],
  },
  Cu3Au: {
    name: "Cu3Au-type",
    spaceGroup: "Pm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB3",
    sites: ["A", "B"],
  },
  BetaTungsten: {
    name: "A15 (Cr3Si-type)",
    spaceGroup: "Pm-3n",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A3B",
    sites: ["A", "B"],
  },
  ZintlPhase: {
    name: "BaAl4-type",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "AB4",
    sites: ["A", "B"],
  },
  Rutile: {
    name: "Rutile",
    spaceGroup: "P42/mnm",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  Corundum: {
    name: "Corundum",
    spaceGroup: "R-3c",
    crystalSystem: "trigonal",
    dimensionality: "3D",
    stoichiometry: "A2O3",
    sites: ["A", "O"],
  },
  Ilmenite: {
    name: "Ilmenite",
    spaceGroup: "R-3",
    crystalSystem: "trigonal",
    dimensionality: "3D",
    stoichiometry: "ABO3",
    sites: ["A", "B", "O"],
  },
  Scheelite: {
    name: "Scheelite",
    spaceGroup: "I41/a",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "ABO4",
    sites: ["A", "B", "O"],
  },
  Olivine: {
    name: "Olivine",
    spaceGroup: "Pnma",
    crystalSystem: "orthorhombic",
    dimensionality: "3D",
    stoichiometry: "A2BO4",
    sites: ["A", "B", "O"],
  },
  Hausmannite: {
    name: "Tetragonal Spinel",
    spaceGroup: "I41/amd",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "AB2O4",
    sites: ["A", "B", "O"],
  },
  Hauyne: {
    name: "Sodalite-type",
    spaceGroup: "P-43n",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A4B3C12",
    sites: ["A", "B", "C"],
  },
  Chalcopyrite: {
    name: "Chalcopyrite",
    spaceGroup: "I-42d",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "ABC2",
    sites: ["A", "B", "C"],
  },
  CubicBoronNitride: {
    name: "Zincblende-BN",
    spaceGroup: "F-43m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  Graphite: {
    name: "Graphite-type",
    spaceGroup: "P63/mmc",
    crystalSystem: "hexagonal",
    dimensionality: "2D",
    stoichiometry: "A",
    sites: ["A"],
  },
  L1_0: {
    name: "CuAu-I type",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    dimensionality: "3D",
    stoichiometry: "AB",
    sites: ["A", "B"],
  },
  Garnet: {
    name: "Garnet",
    spaceGroup: "Ia-3d",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A3B2C3O12",
    sites: ["A", "B", "C", "O"],
  },
  MoS2: {
    name: "Molybdenite (2H)",
    spaceGroup: "P63/mmc",
    crystalSystem: "hexagonal",
    dimensionality: "2D",
    stoichiometry: "AB2",
    sites: ["A", "B"],
  },
  Antifluorite: {
    name: "Antifluorite",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    dimensionality: "3D",
    stoichiometry: "A2B",
    sites: ["A", "B"],
  },
};

function generateAlB2(): PrototypeCandidate[] {
  const A = ["Mg", "Ca", "Sr", "Ba", "Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta"];
  const B = ["B"];
  const info = PROTOTYPE_REGISTRY.AlB2;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    for (const b of B) {
      candidates.push({
        formula: `${a}${b}2`,
        prototype: info.name,
        spaceGroup: info.spaceGroup,
        crystalSystem: info.crystalSystem,
        dimensionality: info.dimensionality,
        siteAssignment: { A: [a], B: [b] },
      });
    }
  }

  return candidates;
}

function generatePerovskite(): PrototypeCandidate[] {
  const A = ["La", "Sr", "Ba", "Ca", "Y", "Bi"];
  const B = ["Ti", "Mn", "Fe", "Co", "Ni", "Cu", "Nb"];
  const info = PROTOTYPE_REGISTRY.Perovskite;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    for (const b of B) {
      candidates.push({
        formula: `${a}${b}O3`,
        prototype: info.name,
        spaceGroup: info.spaceGroup,
        crystalSystem: info.crystalSystem,
        dimensionality: info.dimensionality,
        siteAssignment: { A: [a], B: [b], O: ["O"] },
      });
    }
  }

  return candidates;
}

function generateA15(): PrototypeCandidate[] {
  const A = ["Nb", "V", "Ti", "Zr", "Mo"];
  const B = ["Sn", "Ge", "Si", "Ga", "Al"];
  const info = PROTOTYPE_REGISTRY.A15;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    for (const b of B) {
      candidates.push({
        formula: `${a}3${b}`,
        prototype: info.name,
        spaceGroup: info.spaceGroup,
        crystalSystem: info.crystalSystem,
        dimensionality: info.dimensionality,
        siteAssignment: { A: [a], B: [b] },
      });
    }
  }

  return candidates;
}

function generateClathrate(): PrototypeCandidate[] {
  const M = ["La", "Y", "Ca", "Sc", "Th", "Ce", "Ac"];
  const stoichs = [4, 6, 9, 10, 12];
  const info = PROTOTYPE_REGISTRY.Clathrate;
  const candidates: PrototypeCandidate[] = [];

  for (const m of M) {
    for (const s of stoichs) {
      const label = s >= 10 ? "Sodalite" : "Clathrate";
      candidates.push({
        formula: `${m}H${s}`,
        prototype: label,
        spaceGroup: info.spaceGroup,
        crystalSystem: info.crystalSystem,
        dimensionality: info.dimensionality,
        siteAssignment: { M: [m], H: ["H"], stoichiometry: [String(s)] },
      });
    }
  }

  return candidates;
}

function generateThCr2Si2(): PrototypeCandidate[] {
  const A = ["La", "Ba", "Sr", "Ca", "K"];
  const B = ["Fe", "Co", "Ni", "Mn"];
  const C = ["As", "P", "Se", "S"];
  const info = PROTOTYPE_REGISTRY.ThCr2Si2;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    if (!sizeCompatible(a, "Th")) continue;
    for (const b of B) {
      if (!sizeCompatible(b, "Cr")) continue;
      for (const c of C) {
        if (!sizeCompatible(c, "Si")) continue;
        candidates.push({
          formula: `${a}${b}2${c}2`,
          prototype: info.name,
          spaceGroup: info.spaceGroup,
          crystalSystem: info.crystalSystem,
          dimensionality: info.dimensionality,
          siteAssignment: { A: [a], B: [b], C: [c] },
        });
      }
    }
  }

  return candidates;
}

function generateSpinel(): PrototypeCandidate[] {
  const A = ["Mg", "Fe", "Mn", "Zn", "Co"];
  const B = ["Al", "Cr", "Fe", "V", "Ti"];
  const info = PROTOTYPE_REGISTRY.Spinel;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    if (!sizeCompatible(a, "Mg")) continue;
    for (const b of B) {
      if (a === b) continue;
      if (!sizeCompatible(b, "Al")) continue;
      candidates.push({
        formula: `${a}${b}2O4`,
        prototype: info.name,
        spaceGroup: info.spaceGroup,
        crystalSystem: info.crystalSystem,
        dimensionality: info.dimensionality,
        siteAssignment: { A: [a], B: [b], O: ["O"] },
      });
    }
  }

  return candidates;
}

function generateMAX(): PrototypeCandidate[] {
  const M = ["Ti", "V", "Cr", "Nb", "Mo", "Zr", "Hf", "Ta", "W"];
  const A = ["Al", "Si", "Ga", "Ge", "Sn", "In"];
  const X = ["C", "N"];
  const nValues = [1, 2, 3];
  const info = PROTOTYPE_REGISTRY.MAX;
  const candidates: PrototypeCandidate[] = [];
  const seen = new Set<string>();

  for (const m of M) {
    for (const a of A) {
      for (const x of X) {
        for (const n of nValues) {
          const mCount = n + 1;
          const xCount = n;
          let formula = mCount === 1 ? m : `${m}${mCount}`;
          formula += a;
          if (xCount > 0) {
            formula += xCount === 1 ? x : `${x}${xCount}`;
          }
          if (seen.has(formula)) continue;
          seen.add(formula);
          candidates.push({
            formula,
            prototype: info.name,
            spaceGroup: info.spaceGroup,
            crystalSystem: info.crystalSystem,
            dimensionality: info.dimensionality,
            siteAssignment: { M: [m], A: [a], X: [x], n: [String(n)] },
          });
        }
      }
    }
  }

  return candidates;
}

function generateLayeredNitride(): PrototypeCandidate[] {
  const M = ["Zr", "Hf", "Ti"];
  const X = ["Cl", "Br", "I"];
  const A = ["Li", "Na", "K"];
  const xValues = [0.2, 0.4, 0.6, 0.8, 1.0];
  const info = PROTOTYPE_REGISTRY.LayeredNitride;
  const candidates: PrototypeCandidate[] = [];
  const seen = new Set<string>();

  for (const a of A) {
    for (const xv of xValues) {
      for (const m of M) {
        for (const halide of X) {
          const multiplier = xv <= 0 ? 5 : Math.round(1 / gcdFloat(xv, 1.0));
          const aCount = Math.round(xv * multiplier);
          const mCount = multiplier;
          const nCount = multiplier;
          const halideCount = multiplier;

          const aStr = aCount === 1 ? a : `${a}${aCount}`;
          const mStr = mCount === 1 ? m : `${m}${mCount}`;
          const nStr = nCount === 1 ? "N" : `N${nCount}`;
          const hStr = halideCount === 1 ? halide : `${halide}${halideCount}`;
          const formula = `${aStr}${mStr}${nStr}${hStr}`;

          if (seen.has(formula)) continue;
          seen.add(formula);
          candidates.push({
            formula,
            prototype: info.name,
            spaceGroup: info.spaceGroup,
            crystalSystem: info.crystalSystem,
            dimensionality: info.dimensionality,
            siteAssignment: { A: [a], M: [m], N: ["N"], X: [halide], x: [String(xv)] },
          });
        }
      }
    }
  }

  return candidates;
}

function generateLaves(): PrototypeCandidate[] {
  const A = ["Y", "Zr", "Hf", "Nb", "Ta", "La"];
  const B = ["Fe", "Co", "Ni", "Mn", "Ru", "Os"];
  const info = PROTOTYPE_REGISTRY.Laves;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    for (const b of B) {
      candidates.push({
        formula: `${a}${b}2`,
        prototype: info.name,
        spaceGroup: info.spaceGroup,
        crystalSystem: info.crystalSystem,
        dimensionality: info.dimensionality,
        siteAssignment: { A: [a], B: [b] },
      });
    }
  }

  return candidates;
}

function generateHeusler(): PrototypeCandidate[] {
  const A = ["Co", "Fe", "Ni", "Mn", "Cu"];
  const B = ["Ti", "V", "Mn", "Fe", "Cr"];
  const C = ["Si", "Ge", "Sn", "Al", "Ga"];
  const info = PROTOTYPE_REGISTRY.Heusler;
  const candidates: PrototypeCandidate[] = [];

  for (const a of A) {
    for (const b of B) {
      if (a === b) continue;
      if (!sizeCompatible(b, "Ti", 0.20)) continue;
      for (const c of C) {
        if (!sizeCompatible(c, "Si", 0.20)) continue;
        candidates.push({
          formula: `${a}2${b}${c}`,
          prototype: info.name,
          spaceGroup: info.spaceGroup,
          crystalSystem: info.crystalSystem,
          dimensionality: info.dimensionality,
          siteAssignment: { A: [a], B: [b], C: [c] },
        });
      }
    }
  }

  return candidates;
}

export function runPrototypeGeneration(): PrototypeCandidate[] {
  const all: PrototypeCandidate[] = [
    ...generateAlB2(),
    ...generatePerovskite(),
    ...generateA15(),
    ...generateClathrate(),
    ...generateThCr2Si2(),
    ...generateSpinel(),
    ...generateMAX(),
    ...generateLayeredNitride(),
    ...generateLaves(),
    ...generateHeusler(),
  ];

  const seen = new Set<string>();
  const deduped: PrototypeCandidate[] = [];
  for (const c of all) {
    const key = `${c.formula}|${c.prototype}`;
    if (!seen.has(key)) {
      seen.add(key);
      deduped.push(c);
    }
  }

  return deduped;
}

const PROTOTYPE_ALIASES: Record<string, string> = {
  "spinel": "Spinel",
  "tetragonal spinel": "Spinel",
  "spinel-type": "Spinel",
  "sodalite": "Hauyne",
  "molybdenite": "MoS2",
  "mos2": "MoS2",
  "zincblende": "CubicBoronNitride",
  "cuau": "L1_0",
  "ruddlesden popper": "RuddlesdenPopper",
  "ruddlesden-popper": "RuddlesdenPopper",
  "pyrochlore": "Pychlore",
  "cuprate": "Cuprate1212",
  "rock salt": "RockSalt",
  "rocksalt": "RockSalt",
  "rock-salt": "RockSalt",
  "graphite": "Graphite",
  "fluorite": "Fluorite",
  "antifluorite": "Antifluorite",
  "chalcopyrite": "Chalcopyrite",
  "garnet": "Garnet",
  "max phase": "MAX",
  "max-phase": "MAX",
  "max": "MAX",
  "mn+1axn": "MAX",
  "heusler": "Heusler",
  "heusler alloy": "Heusler",
  "full heusler": "Heusler",
  "half heusler": "Heusler",
  "laves": "Laves",
  "laves phase": "Laves",
  "c15": "Laves",
  "c14": "Laves",
  "perovskite": "Perovskite",
  "abo3": "Perovskite",
  "a15": "A15",
  "a15-type": "A15",
  "a-15": "A15",
  "cr3si": "A15",
  "nb3sn": "A15",
  "clathrate": "Clathrate",
  "cage clathrate": "Clathrate",
  "cage-clathrate": "Clathrate",
  "sodalite cage": "Clathrate",
  "alb2": "AlB2",
  "alb2-type": "AlB2",
  "mgb2": "AlB2",
  "mgb2-type": "AlB2",
  "delafossite": "Delafossite",
  "layered nitride": "LayeredNitride",
  "layered-nitride": "LayeredNitride",
  "thcr2si2": "ThCr2Si2",
  "thcr2si2-type": "ThCr2Si2",
  "122": "ThCr2Si2",
  "122-type": "ThCr2Si2",
  "1111": "ThCr2Si2",
};

export function getPrototypeInfo(name: string): PrototypeInfo | null {
  if (PROTOTYPE_REGISTRY[name]) {
    return PROTOTYPE_REGISTRY[name];
  }

  const lower = name.toLowerCase().trim();

  const aliasKey = PROTOTYPE_ALIASES[lower];
  if (aliasKey && PROTOTYPE_REGISTRY[aliasKey]) {
    return PROTOTYPE_REGISTRY[aliasKey];
  }

  for (const [, info] of Object.entries(PROTOTYPE_REGISTRY)) {
    if (info.name.toLowerCase() === lower) {
      return info;
    }
  }

  for (const [, info] of Object.entries(PROTOTYPE_REGISTRY)) {
    if (info.name.toLowerCase().includes(lower) || lower.includes(info.name.toLowerCase())) {
      return info;
    }
  }

  return null;
}
