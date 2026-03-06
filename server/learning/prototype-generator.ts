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
  const stoichs = [6, 10];
  const info = PROTOTYPE_REGISTRY.Clathrate;
  const candidates: PrototypeCandidate[] = [];

  for (const m of M) {
    for (const s of stoichs) {
      const label = s === 10 ? "Sodalite" : "Clathrate";
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
    for (const b of B) {
      for (const c of C) {
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
    for (const b of B) {
      if (a === b) continue;
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

  for (const m of M) {
    for (const a of A) {
      for (const x of X) {
        for (const n of nValues) {
          const mCount = n + 1;
          const xCount = n;
          let formula: string;
          if (mCount === 1) {
            formula = `${m}${a}`;
          } else {
            formula = `${m}${mCount}${a}`;
          }
          if (xCount === 1) {
            formula += x;
          } else {
            formula += `${x}${xCount}`;
          }
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

  for (const a of A) {
    for (const xv of xValues) {
      for (const m of M) {
        for (const halide of X) {
          const xStr = xv === 1.0 ? "" : `${xv}`;
          const aComponent = xStr ? `${a}${xStr}` : a;
          candidates.push({
            formula: `${aComponent}${m}N${halide}`,
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
      for (const c of C) {
        if (a === b) continue;
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

export function getPrototypeInfo(name: string): PrototypeInfo | null {
  if (PROTOTYPE_REGISTRY[name]) {
    return PROTOTYPE_REGISTRY[name];
  }
  for (const [, info] of Object.entries(PROTOTYPE_REGISTRY)) {
    if (info.name.toLowerCase() === name.toLowerCase()) {
      return info;
    }
  }
  return null;
}
