export interface DistortionVector {
  axis: "x" | "y" | "z" | "xy" | "xz" | "yz" | "xyz";
  magnitude: number;
  type: "tetragonal" | "orthorhombic" | "monoclinic" | "trigonal" | "shear" | "breathing";
}

export interface SubgroupRelation {
  parent: string;
  child: string;
  parentNumber: number;
  childNumber: number;
  index: number;
  distortionVectors: DistortionVector[];
  transitionType: "displacive" | "order-disorder" | "reconstructive";
  irrepLabel: string;
}

export interface WyckoffSiteInfo {
  spaceGroup: string;
  letter: string;
  multiplicity: number;
  siteSymmetryOrder: number;
  siteSymmetryLabel: string;
  positions: [number, number, number][];
}

export interface SymmetryEmbedding {
  wyckoffMultiplicity: number;
  siteSymmetryOrder: number;
  subgroupIndex: number;
  irrepEncoding: number;
  distortionMagnitude: number;
  parentGroupOrder: number;
}

const GROUP_SUBGROUP_HIERARCHY: SubgroupRelation[] = [
  {
    parent: "Pm-3m", child: "P4/mmm", parentNumber: 221, childNumber: 123,
    index: 3, transitionType: "displacive", irrepLabel: "Γ4-",
    distortionVectors: [
      { axis: "z", magnitude: 0.05, type: "tetragonal" },
    ],
  },
  {
    parent: "Pm-3m", child: "R-3m", parentNumber: 221, childNumber: 166,
    index: 4, transitionType: "displacive", irrepLabel: "Γ5-",
    distortionVectors: [
      { axis: "xyz", magnitude: 0.03, type: "trigonal" },
    ],
  },
  {
    parent: "Pm-3m", child: "Pnma", parentNumber: 221, childNumber: 62,
    index: 6, transitionType: "displacive", irrepLabel: "M3+",
    distortionVectors: [
      { axis: "x", magnitude: 0.08, type: "orthorhombic" },
      { axis: "y", magnitude: 0.06, type: "orthorhombic" },
    ],
  },
  {
    parent: "Fm-3m", child: "I4/mmm", parentNumber: 225, childNumber: 139,
    index: 3, transitionType: "displacive", irrepLabel: "Γ3+",
    distortionVectors: [
      { axis: "z", magnitude: 0.04, type: "tetragonal" },
    ],
  },
  {
    parent: "Fm-3m", child: "R-3m", parentNumber: 225, childNumber: 166,
    index: 4, transitionType: "displacive", irrepLabel: "Γ5+",
    distortionVectors: [
      { axis: "xyz", magnitude: 0.03, type: "trigonal" },
    ],
  },
  {
    parent: "Fm-3m", child: "Pnma", parentNumber: 225, childNumber: 62,
    index: 8, transitionType: "reconstructive", irrepLabel: "X5-",
    distortionVectors: [
      { axis: "x", magnitude: 0.10, type: "orthorhombic" },
      { axis: "y", magnitude: 0.08, type: "orthorhombic" },
      { axis: "z", magnitude: 0.06, type: "orthorhombic" },
    ],
  },
  {
    parent: "Im-3m", child: "P4/mmm", parentNumber: 229, childNumber: 123,
    index: 3, transitionType: "displacive", irrepLabel: "Γ4-",
    distortionVectors: [
      { axis: "z", magnitude: 0.05, type: "tetragonal" },
    ],
  },
  {
    parent: "Im-3m", child: "Pnma", parentNumber: 229, childNumber: 62,
    index: 6, transitionType: "displacive", irrepLabel: "H5-",
    distortionVectors: [
      { axis: "x", magnitude: 0.07, type: "orthorhombic" },
      { axis: "y", magnitude: 0.05, type: "orthorhombic" },
    ],
  },
  {
    parent: "P6/mmm", child: "P4/mmm", parentNumber: 191, childNumber: 123,
    index: 3, transitionType: "reconstructive", irrepLabel: "K1",
    distortionVectors: [
      { axis: "xy", magnitude: 0.10, type: "shear" },
      { axis: "z", magnitude: 0.04, type: "tetragonal" },
    ],
  },
  {
    parent: "P4/mmm", child: "Pnma", parentNumber: 123, childNumber: 62,
    index: 2, transitionType: "displacive", irrepLabel: "M5-",
    distortionVectors: [
      { axis: "x", magnitude: 0.06, type: "orthorhombic" },
    ],
  },
  {
    parent: "I4/mmm", child: "Pnma", parentNumber: 139, childNumber: 62,
    index: 2, transitionType: "displacive", irrepLabel: "X3+",
    distortionVectors: [
      { axis: "x", magnitude: 0.05, type: "orthorhombic" },
      { axis: "y", magnitude: 0.04, type: "orthorhombic" },
    ],
  },
  {
    parent: "R-3m", child: "Pnma", parentNumber: 166, childNumber: 62,
    index: 3, transitionType: "displacive", irrepLabel: "L1+",
    distortionVectors: [
      { axis: "x", magnitude: 0.06, type: "orthorhombic" },
      { axis: "yz", magnitude: 0.04, type: "shear" },
    ],
  },
  {
    parent: "Pm-3m", child: "Fm-3m", parentNumber: 221, childNumber: 225,
    index: 4, transitionType: "order-disorder", irrepLabel: "R1+",
    distortionVectors: [
      { axis: "xyz", magnitude: 0.02, type: "breathing" },
    ],
  },
  {
    parent: "Pm-3m", child: "Im-3m", parentNumber: 221, childNumber: 229,
    index: 2, transitionType: "order-disorder", irrepLabel: "Γ1+",
    distortionVectors: [
      { axis: "xyz", magnitude: 0.01, type: "breathing" },
    ],
  },
];

const WYCKOFF_DATABASE: WyckoffSiteInfo[] = [
  { spaceGroup: "Pm-3m", letter: "a", multiplicity: 1, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", positions: [[0, 0, 0]] },
  { spaceGroup: "Pm-3m", letter: "b", multiplicity: 1, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", positions: [[0.5, 0.5, 0.5]] },
  { spaceGroup: "Pm-3m", letter: "c", multiplicity: 3, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] },
  { spaceGroup: "Pm-3m", letter: "d", multiplicity: 3, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]] },
  { spaceGroup: "Fm-3m", letter: "a", multiplicity: 4, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", positions: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]] },
  { spaceGroup: "Fm-3m", letter: "b", multiplicity: 4, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", positions: [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]] },
  { spaceGroup: "Im-3m", letter: "a", multiplicity: 2, siteSymmetryOrder: 48, siteSymmetryLabel: "m-3m", positions: [[0, 0, 0], [0.5, 0.5, 0.5]] },
  { spaceGroup: "Im-3m", letter: "b", multiplicity: 6, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]] },
  { spaceGroup: "P6/mmm", letter: "a", multiplicity: 1, siteSymmetryOrder: 24, siteSymmetryLabel: "6/mmm", positions: [[0, 0, 0]] },
  { spaceGroup: "P6/mmm", letter: "c", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", positions: [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0]] },
  { spaceGroup: "P6/mmm", letter: "d", multiplicity: 2, siteSymmetryOrder: 12, siteSymmetryLabel: "-6m2", positions: [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]] },
  { spaceGroup: "P4/mmm", letter: "a", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0, 0, 0]] },
  { spaceGroup: "P4/mmm", letter: "b", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0, 0, 0.5]] },
  { spaceGroup: "P4/mmm", letter: "c", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0.5, 0.5, 0]] },
  { spaceGroup: "P4/mmm", letter: "d", multiplicity: 1, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0.5, 0.5, 0.5]] },
  { spaceGroup: "I4/mmm", letter: "a", multiplicity: 2, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0, 0, 0], [0.5, 0.5, 0.5]] },
  { spaceGroup: "I4/mmm", letter: "b", multiplicity: 2, siteSymmetryOrder: 16, siteSymmetryLabel: "4/mmm", positions: [[0, 0, 0.5], [0.5, 0.5, 0]] },
  { spaceGroup: "R-3m", letter: "a", multiplicity: 3, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m", positions: [[0, 0, 0], [1 / 3, 2 / 3, 2 / 3], [2 / 3, 1 / 3, 1 / 3]] },
  { spaceGroup: "R-3m", letter: "b", multiplicity: 3, siteSymmetryOrder: 12, siteSymmetryLabel: "-3m", positions: [[0, 0, 0.5], [1 / 3, 2 / 3, 1 / 6], [2 / 3, 1 / 3, 5 / 6]] },
  { spaceGroup: "Pnma", letter: "a", multiplicity: 4, siteSymmetryOrder: 4, siteSymmetryLabel: "-1", positions: [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0], [0.5, 0.5, 0.5]] },
  { spaceGroup: "Pnma", letter: "c", multiplicity: 4, siteSymmetryOrder: 2, siteSymmetryLabel: "m", positions: [[0.25, 0.25, 0], [0.75, 0.75, 0], [0.75, 0.25, 0.5], [0.25, 0.75, 0.5]] },
];

const SPACE_GROUP_ORDER: Record<string, number> = {
  "Pm-3m": 48, "Fm-3m": 192, "Im-3m": 96,
  "P6/mmm": 24, "P4/mmm": 16, "I4/mmm": 32,
  "R-3m": 36, "Pnma": 8,
};

const IRREP_ENCODING: Record<string, number> = {
  "Γ1+": 0.0, "Γ3+": 0.1, "Γ4-": 0.15, "Γ5+": 0.2, "Γ5-": 0.25,
  "R1+": 0.3, "M3+": 0.4, "M5-": 0.45, "X3+": 0.5, "X5-": 0.55,
  "H5-": 0.6, "K1": 0.7, "L1+": 0.8,
};

export function getSubgroups(spaceGroupName: string): SubgroupRelation[] {
  return GROUP_SUBGROUP_HIERARCHY.filter(r => r.parent === spaceGroupName);
}

export function getSupergroups(spaceGroupName: string): SubgroupRelation[] {
  return GROUP_SUBGROUP_HIERARCHY.filter(r => r.child === spaceGroupName);
}

export function getSubgroupChain(from: string, to: string): SubgroupRelation[] {
  const chain: SubgroupRelation[] = [];
  const visited = new Set<string>();

  function dfs(current: string): boolean {
    if (current === to) return true;
    if (visited.has(current)) return false;
    visited.add(current);
    const children = getSubgroups(current);
    for (const rel of children) {
      if (dfs(rel.child)) {
        chain.unshift(rel);
        return true;
      }
    }
    return false;
  }

  dfs(from);
  return chain;
}

export function getWyckoffSites(spaceGroupName: string): WyckoffSiteInfo[] {
  return WYCKOFF_DATABASE.filter(w => w.spaceGroup === spaceGroupName);
}

export function getWyckoffForPosition(
  spaceGroupName: string,
  fracPosition: [number, number, number],
  tolerance: number = 0.1
): WyckoffSiteInfo | null {
  const sites = getWyckoffSites(spaceGroupName);

  for (const site of sites) {
    for (const pos of site.positions) {
      const dx = Math.abs(fracPosition[0] - pos[0]) % 1.0;
      const dy = Math.abs(fracPosition[1] - pos[1]) % 1.0;
      const dz = Math.abs(fracPosition[2] - pos[2]) % 1.0;
      const d = Math.sqrt(
        Math.min(dx, 1 - dx) ** 2 +
        Math.min(dy, 1 - dy) ** 2 +
        Math.min(dz, 1 - dz) ** 2
      );
      if (d < tolerance) return site;
    }
  }

  return null;
}

export function computeSymmetryEmbedding(
  spaceGroupName: string,
  fracPosition?: [number, number, number]
): SymmetryEmbedding {
  const parentOrder = SPACE_GROUP_ORDER[spaceGroupName] ?? 8;
  const supergroups = getSupergroups(spaceGroupName);

  let subgroupIndex = 1;
  let irrepEncoding = 0;
  let distortionMagnitude = 0;

  if (supergroups.length > 0) {
    const rel = supergroups[0];
    subgroupIndex = rel.index;
    irrepEncoding = IRREP_ENCODING[rel.irrepLabel] ?? 0;
    distortionMagnitude = rel.distortionVectors.reduce((s, d) => s + d.magnitude, 0) / rel.distortionVectors.length;
  }

  let wyckoffMultiplicity = 1;
  let siteSymmetryOrder = parentOrder;

  if (fracPosition) {
    const wyckoff = getWyckoffForPosition(spaceGroupName, fracPosition);
    if (wyckoff) {
      wyckoffMultiplicity = wyckoff.multiplicity;
      siteSymmetryOrder = wyckoff.siteSymmetryOrder;
    }
  } else {
    const sites = getWyckoffSites(spaceGroupName);
    if (sites.length > 0) {
      wyckoffMultiplicity = sites[0].multiplicity;
      siteSymmetryOrder = sites[0].siteSymmetryOrder;
    }
  }

  return {
    wyckoffMultiplicity,
    siteSymmetryOrder,
    subgroupIndex,
    irrepEncoding,
    distortionMagnitude,
    parentGroupOrder: parentOrder,
  };
}

export function computeSymmetryFeatureVector(embedding: SymmetryEmbedding): number[] {
  return [
    embedding.wyckoffMultiplicity / 8.0,
    embedding.siteSymmetryOrder / 48.0,
    embedding.subgroupIndex / 8.0,
    embedding.irrepEncoding,
    embedding.distortionMagnitude / 0.1,
    embedding.parentGroupOrder / 192.0,
  ];
}

export interface BrokenSymmetryVariant {
  parentSpaceGroup: string;
  childSpaceGroup: string;
  latticeDistortion: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  atomicDisplacements: { dx: number; dy: number; dz: number }[];
  distortionAmplitude: number;
  relation: SubgroupRelation;
}

export function generateBrokenSymmetryVariants(
  spaceGroupName: string,
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  atomCount: number,
  amplitudeScale: number = 1.0
): BrokenSymmetryVariant[] {
  const subgroups = getSubgroups(spaceGroupName);
  const variants: BrokenSymmetryVariant[] = [];

  for (const rel of subgroups) {
    const newLattice = { ...lattice };
    const displacements: { dx: number; dy: number; dz: number }[] = [];

    for (const dv of rel.distortionVectors) {
      const amp = dv.magnitude * amplitudeScale;

      switch (dv.type) {
        case "tetragonal":
          newLattice.c *= (1 + amp);
          newLattice.a *= (1 - amp * 0.3);
          newLattice.b *= (1 - amp * 0.3);
          break;
        case "orthorhombic":
          if (dv.axis === "x") newLattice.a *= (1 + amp);
          if (dv.axis === "y") newLattice.b *= (1 + amp * 0.7);
          if (dv.axis === "z") newLattice.c *= (1 - amp * 0.5);
          break;
        case "trigonal":
          newLattice.alpha = 90 - amp * 180 / Math.PI * 2;
          newLattice.beta = 90 - amp * 180 / Math.PI * 2;
          newLattice.gamma = 90 - amp * 180 / Math.PI * 2;
          break;
        case "shear":
          if (dv.axis.includes("x") && dv.axis.includes("y")) {
            newLattice.gamma = 90 + amp * 180 / Math.PI;
          }
          if (dv.axis.includes("y") && dv.axis.includes("z")) {
            newLattice.alpha = 90 + amp * 180 / Math.PI * 0.5;
          }
          break;
        case "breathing":
          newLattice.a *= (1 + amp * 0.5);
          newLattice.b *= (1 + amp * 0.5);
          newLattice.c *= (1 + amp * 0.5);
          break;
      }
    }

    for (let i = 0; i < atomCount; i++) {
      let dx = 0, dy = 0, dz = 0;
      for (const dv of rel.distortionVectors) {
        const amp = dv.magnitude * amplitudeScale;
        const phase = (i / Math.max(1, atomCount - 1)) * Math.PI;
        if (dv.axis.includes("x")) dx += amp * Math.sin(phase) * 0.1;
        if (dv.axis.includes("y")) dy += amp * Math.cos(phase) * 0.1;
        if (dv.axis.includes("z")) dz += amp * Math.sin(phase + Math.PI / 4) * 0.1;
      }
      displacements.push({ dx, dy, dz });
    }

    const totalDistortion = rel.distortionVectors.reduce((s, d) => s + d.magnitude, 0) * amplitudeScale;

    variants.push({
      parentSpaceGroup: spaceGroupName,
      childSpaceGroup: rel.child,
      latticeDistortion: newLattice,
      atomicDisplacements: displacements,
      distortionAmplitude: totalDistortion,
      relation: rel,
    });
  }

  return variants;
}

export function applyBrokenSymmetry(
  atoms: { symbol: string; x: number; y: number; z: number }[],
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number },
  variant: BrokenSymmetryVariant
): {
  atoms: { symbol: string; x: number; y: number; z: number }[];
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
} {
  const newAtoms = atoms.map((atom, i) => {
    const disp = variant.atomicDisplacements[i] ?? { dx: 0, dy: 0, dz: 0 };
    return {
      symbol: atom.symbol,
      x: atom.x + disp.dx * variant.latticeDistortion.a,
      y: atom.y + disp.dy * variant.latticeDistortion.b,
      z: atom.z + disp.dz * variant.latticeDistortion.c,
    };
  });

  return {
    atoms: newAtoms,
    lattice: { ...variant.latticeDistortion },
  };
}

export function getSymmetrySubgroupStats(): {
  totalRelations: number;
  spaceGroupsCovered: number;
  wyckoffSitesCovered: number;
  maxChainDepth: number;
  avgSubgroupIndex: number;
} {
  const parentGroups = Array.from(new Set(GROUP_SUBGROUP_HIERARCHY.map(r => r.parent)));
  const childGroups = Array.from(new Set(GROUP_SUBGROUP_HIERARCHY.map(r => r.child)));
  const allGroups = Array.from(new Set([...parentGroups, ...childGroups]));

  const avgIndex = GROUP_SUBGROUP_HIERARCHY.reduce((s, r) => s + r.index, 0) / Math.max(1, GROUP_SUBGROUP_HIERARCHY.length);

  let maxDepth = 0;
  for (const sg of parentGroups) {
    for (const target of childGroups) {
      const chain = getSubgroupChain(sg, target);
      if (chain.length > maxDepth) maxDepth = chain.length;
    }
  }

  return {
    totalRelations: GROUP_SUBGROUP_HIERARCHY.length,
    spaceGroupsCovered: allGroups.length,
    wyckoffSitesCovered: WYCKOFF_DATABASE.length,
    maxChainDepth: maxDepth,
    avgSubgroupIndex: Math.round(avgIndex * 100) / 100,
  };
}
