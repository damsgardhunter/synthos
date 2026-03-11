import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  classifyHydrogenBonding,
  parseFormulaElements,
  type HydrogenBondingType,
} from "../learning/physics-engine";
import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";

export interface HHDistanceDistribution {
  estimatedMeanHH: number;
  estimatedMinHH: number;
  estimatedMaxHH: number;
  shortBondFraction: number;
  metallicBondFraction: number;
  distributionType: string;
}

export interface CageTopology {
  cageType: string;
  cageSymmetry: string;
  verticesPerCage: number;
  facesPerCage: number;
  cageVolume: number;
  cageCompleteness: number;
  sodaliteCharacter: number;
  clathrateCharacter: number;
}

export interface HydrogenNetworkAnalysis {
  formula: string;
  isHydride: boolean;
  hydrogenCount: number;
  hydrogenFraction: number;
  hydrogenToMetalRatio: number;
  bondingType: HydrogenBondingType;

  hydrogenNetworkDim: number;
  hydrogenCageScore: number;
  Hcoordination: number;
  hydrogenConnectivity: number;
  hydrogenPhononCouplingScore: number;

  hhDistribution: HHDistanceDistribution;
  cageTopology: CageTopology;
  hydrogenDensity: number;
  networkPercolation: number;
  phononContribution: {
    hydrogenPhononFreq: number;
    hydrogenPhononLambda: number;
    anharmonicCorrection: number;
  };

  percolates: boolean;
  geometricPercolationUsed: boolean;
  percolationDetail?: HydrogenPercolationResult;
  compositeSCScore: number;
  networkClass: string;
  insights: string[];
}

export interface PercolationAtom {
  symbol: string;
  x: number;
  y: number;
  z: number;
}

export interface PercolationLattice {
  a: number;
  b: number;
  c: number;
  alpha?: number;
  beta?: number;
  gamma?: number;
}

export interface HydrogenPercolationResult {
  hAtomCount: number;
  clusterCount: number;
  largestClusterSize: number;
  largestClusterFraction: number;
  percolates3D: boolean;
  percolationConfidence: number;
  cutoffAngstrom: number;
}

const H_PERCOLATION_CUTOFF_AMBIENT = 1.9;
const H_PERCOLATION_CUTOFF_100GPA = 1.5;
const H_PERCOLATION_CUTOFF_200GPA = 1.25;

function pressureAdaptedHCutoff(pressureGPa: number, hydrogenFraction: number = 0.5): number {
  let baseCutoff: number;
  if (pressureGPa <= 0) {
    baseCutoff = H_PERCOLATION_CUTOFF_AMBIENT;
  } else if (pressureGPa <= 100) {
    const t = pressureGPa / 100;
    baseCutoff = H_PERCOLATION_CUTOFF_AMBIENT + (H_PERCOLATION_CUTOFF_100GPA - H_PERCOLATION_CUTOFF_AMBIENT) * t;
  } else if (pressureGPa <= 200) {
    const t = (pressureGPa - 100) / 100;
    baseCutoff = H_PERCOLATION_CUTOFF_100GPA + (H_PERCOLATION_CUTOFF_200GPA - H_PERCOLATION_CUTOFF_100GPA) * t;
  } else {
    baseCutoff = H_PERCOLATION_CUTOFF_200GPA - 0.001 * (pressureGPa - 200);
    baseCutoff = Math.max(1.1, baseCutoff);
  }

  if (hydrogenFraction > 0.6) {
    const densityBonus = Math.min(0.15, (hydrogenFraction - 0.6) * 0.375);
    baseCutoff += densityBonus;
  }

  return baseCutoff;
}

function toFractional(atom: PercolationAtom, lat: PercolationLattice): PercolationAtom {
  if (Math.abs(atom.x) <= 1.01 && Math.abs(atom.y) <= 1.01 && Math.abs(atom.z) <= 1.01) {
    return atom;
  }

  const alpha = ((lat.alpha ?? 90) * Math.PI) / 180;
  const beta = ((lat.beta ?? 90) * Math.PI) / 180;
  const gamma = ((lat.gamma ?? 90) * Math.PI) / 180;

  const cosAlpha = Math.cos(alpha);
  const cosBeta = Math.cos(beta);
  const cosGamma = Math.cos(gamma);
  const sinGamma = Math.sin(gamma);
  const term = Math.sqrt(Math.max(0, 1 - cosAlpha * cosAlpha - cosBeta * cosBeta - cosGamma * cosGamma + 2 * cosAlpha * cosBeta * cosGamma));
  const vol = lat.a * lat.b * lat.c * term;

  if (vol < 1e-10) return atom;

  const fx = atom.x / lat.a - (cosGamma / sinGamma) * atom.y / lat.a
    + (cosAlpha * cosGamma - cosBeta) / (sinGamma * term) * atom.z / lat.a;
  const fy = atom.y / (lat.b * sinGamma)
    - (cosAlpha - cosBeta * cosGamma) / (sinGamma * term) * atom.z / lat.b;
  const fz = atom.z * sinGamma / (lat.c * term);

  return { symbol: atom.symbol, x: fx, y: fy, z: fz };
}

function periodicDistance(
  a: PercolationAtom,
  b: PercolationAtom,
  lat: PercolationLattice,
): number {
  const alpha = ((lat.alpha ?? 90) * Math.PI) / 180;
  const beta = ((lat.beta ?? 90) * Math.PI) / 180;
  const gamma = ((lat.gamma ?? 90) * Math.PI) / 180;

  let dx = a.x - b.x;
  let dy = a.y - b.y;
  let dz = a.z - b.z;
  dx -= Math.round(dx);
  dy -= Math.round(dy);
  dz -= Math.round(dz);

  const cosAlpha = Math.cos(alpha);
  const cosBeta = Math.cos(beta);
  const cosGamma = Math.cos(gamma);
  const sinGamma = Math.sin(gamma);

  const cx = lat.a * dx + lat.b * cosGamma * dy + lat.c * cosBeta * dz;
  const cy = lat.b * sinGamma * dy + lat.c * ((cosAlpha - cosBeta * cosGamma) / sinGamma) * dz;
  const term = 1 - cosAlpha * cosAlpha - cosBeta * cosBeta - cosGamma * cosGamma + 2 * cosAlpha * cosBeta * cosGamma;
  const cz = lat.c * Math.sqrt(Math.max(0, term)) / sinGamma * dz;

  return Math.sqrt(cx * cx + cy * cy + cz * cz);
}

function unionFind(n: number): { parent: Int32Array; rank: Int32Array; find: (x: number) => number; union: (x: number, y: number) => void } {
  const parent = new Int32Array(n);
  const rank = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;

  function find(x: number): number {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  function union(a: number, b: number): void {
    const ra = find(a), rb = find(b);
    if (ra === rb) return;
    if (rank[ra] < rank[rb]) { parent[ra] = rb; }
    else if (rank[ra] > rank[rb]) { parent[rb] = ra; }
    else { parent[rb] = ra; rank[ra]++; }
  }

  return { parent, rank, find, union };
}

export function checkHydrogenPercolation(
  atoms: PercolationAtom[],
  lattice: PercolationLattice,
  cutoffOrPressure?: number,
  pressureGPa?: number,
): HydrogenPercolationResult {
  const fracAtoms = atoms.map(a => toFractional(a, lattice));

  const hFraction = atoms.length > 0
    ? atoms.filter(a => a.symbol === "H").length / atoms.length
    : 0;

  let cutoff: number;
  if (pressureGPa !== undefined) {
    cutoff = pressureAdaptedHCutoff(pressureGPa, hFraction);
  } else if (cutoffOrPressure !== undefined) {
    cutoff = cutoffOrPressure;
  } else {
    cutoff = H_PERCOLATION_CUTOFF_AMBIENT;
  }
  const hAtoms = fracAtoms.filter(a => a.symbol === "H");
  const hCount = hAtoms.length;

  if (hCount === 0) {
    return { hAtomCount: 0, clusterCount: 0, largestClusterSize: 0, largestClusterFraction: 0, percolates3D: false, percolationConfidence: 0, cutoffAngstrom: cutoff };
  }
  if (hCount === 1) {
    return { hAtomCount: 1, clusterCount: 1, largestClusterSize: 1, largestClusterFraction: 1, percolates3D: false, percolationConfidence: 0, cutoffAngstrom: cutoff };
  }

  const uf = unionFind(hCount);

  const clusterSpans: Map<number, [boolean, boolean, boolean]> = new Map();

  const maxLatticeParam = Math.max(lattice.a, lattice.b, lattice.c);
  const fractionalCutoff = cutoff / Math.max(maxLatticeParam, 1);
  const nCells = Math.max(1, Math.floor(1.0 / fractionalCutoff));
  const cellSize = 1.0 / nCells;

  const cellMap = new Map<string, number[]>();
  const cellKey = (cx: number, cy: number, cz: number) => `${cx},${cy},${cz}`;

  for (let i = 0; i < hCount; i++) {
    let fx = hAtoms[i].x % 1.0; if (fx < 0) fx += 1.0;
    let fy = hAtoms[i].y % 1.0; if (fy < 0) fy += 1.0;
    let fz = hAtoms[i].z % 1.0; if (fz < 0) fz += 1.0;

    const cx = Math.min(nCells - 1, Math.floor(fx / cellSize));
    const cy = Math.min(nCells - 1, Math.floor(fy / cellSize));
    const cz = Math.min(nCells - 1, Math.floor(fz / cellSize));
    const key = cellKey(cx, cy, cz);
    const bucket = cellMap.get(key);
    if (bucket) bucket.push(i);
    else cellMap.set(key, [i]);
  }

  const markSpan = (root: number, axis: number) => {
    let spans = clusterSpans.get(root);
    if (!spans) { spans = [false, false, false]; clusterSpans.set(root, spans); }
    spans[axis] = true;
  };

  for (let cx = 0; cx < nCells; cx++) {
    for (let cy = 0; cy < nCells; cy++) {
      for (let cz = 0; cz < nCells; cz++) {
        const bucket = cellMap.get(cellKey(cx, cy, cz));
        if (!bucket) continue;

        for (let dcx = -1; dcx <= 1; dcx++) {
          for (let dcy = -1; dcy <= 1; dcy++) {
            for (let dcz = -1; dcz <= 1; dcz++) {
              const nx = ((cx + dcx) % nCells + nCells) % nCells;
              const ny = ((cy + dcy) % nCells + nCells) % nCells;
              const nz = ((cz + dcz) % nCells + nCells) % nCells;
              const neighborBucket = cellMap.get(cellKey(nx, ny, nz));
              if (!neighborBucket) continue;

              const isSelf = (dcx === 0 && dcy === 0 && dcz === 0);

              for (const i of bucket) {
                for (const j of neighborBucket) {
                  if (isSelf && j <= i) continue;
                  if (!isSelf && j < i) continue;

                  const dist = periodicDistance(hAtoms[i], hAtoms[j], lattice);
                  if (dist <= cutoff) {
                    uf.union(i, j);

                    const dx = hAtoms[i].x - hAtoms[j].x;
                    const dy = hAtoms[i].y - hAtoms[j].y;
                    const dz = hAtoms[i].z - hAtoms[j].z;
                    const root = uf.find(i);
                    if (Math.abs(Math.round(dx)) >= 1) markSpan(root, 0);
                    if (Math.abs(Math.round(dy)) >= 1) markSpan(root, 1);
                    if (Math.abs(Math.round(dz)) >= 1) markSpan(root, 2);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  const clusterSizes: Record<number, number> = {};
  const finalSpans: Record<number, [boolean, boolean, boolean]> = {};
  for (let i = 0; i < hCount; i++) {
    const root = uf.find(i);
    clusterSizes[root] = (clusterSizes[root] || 0) + 1;
    if (!finalSpans[root]) finalSpans[root] = [false, false, false];
  }

  clusterSpans.forEach((spans, oldRoot) => {
    const root = uf.find(oldRoot);
    if (!finalSpans[root]) finalSpans[root] = [false, false, false];
    if (spans[0]) finalSpans[root][0] = true;
    if (spans[1]) finalSpans[root][1] = true;
    if (spans[2]) finalSpans[root][2] = true;
  });

  const roots = Object.keys(clusterSizes).map(Number);
  const clusterCount = roots.length;
  const largestClusterSize = Math.max(...Object.values(clusterSizes));
  const largestClusterFraction = largestClusterSize / hCount;

  let percolates3D = false;
  let axesSpanned = 0;
  for (const root of roots) {
    const spans = finalSpans[root];
    if (!spans) continue;
    let rootAxes = 0;
    if (spans[0]) rootAxes++;
    if (spans[1]) rootAxes++;
    if (spans[2]) rootAxes++;
    if (rootAxes > axesSpanned) axesSpanned = rootAxes;
    if (rootAxes === 3) {
      percolates3D = true;
      break;
    }
  }

  let percolationConfidence: number;
  if (percolates3D) {
    percolationConfidence = Math.min(1.0, 0.7 + 0.3 * largestClusterFraction);
  } else if (axesSpanned === 2) {
    percolationConfidence = Math.min(0.6, 0.3 + 0.3 * largestClusterFraction);
  } else if (largestClusterFraction >= 0.9 && hCount >= 4) {
    percolationConfidence = 0.4 * largestClusterFraction;
  } else {
    percolationConfidence = 0.1 * largestClusterFraction;
  }

  return {
    hAtomCount: hCount,
    clusterCount,
    largestClusterSize,
    largestClusterFraction: Number(largestClusterFraction.toFixed(4)),
    percolates3D,
    percolationConfidence: Number(percolationConfidence.toFixed(4)),
    cutoffAngstrom: cutoff,
  };
}

const SODALITE_CAGE_ELEMENTS = ["La", "Y", "Ce", "Th", "Ac", "Ca", "Sr", "Ba"];
const CLATHRATE_CAGE_ELEMENTS = ["La", "Y", "Ca", "Ba", "Sr", "Sc", "Ce"];

const CAGE_TEMPLATES: Record<string, { vertices: number; faces: number; symmetry: string; baseVolume: number }> = {
  "H24-sodalite": { vertices: 24, faces: 14, symmetry: "Oh", baseVolume: 58.0 },
  "H32-clathrate-I": { vertices: 32, faces: 18, symmetry: "Pm-3n", baseVolume: 72.0 },
  "H29-clathrate-II": { vertices: 29, faces: 16, symmetry: "Fd-3m", baseVolume: 65.0 },
  "H20-dodecahedron": { vertices: 20, faces: 12, symmetry: "Ih", baseVolume: 45.0 },
  "H16-truncated-tetrahedron": { vertices: 16, faces: 8, symmetry: "Td", baseVolume: 35.0 },
  "H12-icosahedron": { vertices: 12, faces: 20, symmetry: "Ih", baseVolume: 25.0 },
  "H8-cube": { vertices: 8, faces: 6, symmetry: "Oh", baseVolume: 15.0 },
  "H6-octahedron": { vertices: 6, faces: 8, symmetry: "Oh", baseVolume: 10.0 },
};

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = regex.exec(cleaned)) !== null) {
    const el = m[1];
    const num = m[2] ? parseFloat(m[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function estimateHHDistances(
  hCount: number,
  metalElements: string[],
  metalAtomCount: number,
  hRatio: number,
  bondingType: HydrogenBondingType,
): HHDistanceDistribution {
  let meanHH = 2.0;
  let minHH = 1.6;
  let maxHH = 3.0;

  if (bondingType === "metallic-network" || bondingType === "cage-clathrate") {
    const hasHeavyMetal = metalElements.some(e => {
      const d = getElementData(e);
      return d && d.atomicMass > 88;
    });

    if (hRatio >= 10) {
      meanHH = 1.05;
      minHH = 0.85;
      maxHH = 1.30;
    } else if (hRatio >= 6) {
      meanHH = 1.15;
      minHH = 0.95;
      maxHH = 1.45;
    } else if (hRatio >= 4) {
      meanHH = 1.30;
      minHH = 1.05;
      maxHH = 1.65;
    } else {
      meanHH = 1.55;
      minHH = 1.20;
      maxHH = 2.00;
    }

    if (hasHeavyMetal) {
      meanHH *= 0.95;
      minHH *= 0.93;
    }
  } else if (bondingType === "interstitial") {
    meanHH = 2.10;
    minHH = 1.80;
    maxHH = 2.80;
  } else if (bondingType === "covalent-molecular") {
    meanHH = 0.74;
    minHH = 0.70;
    maxHH = 1.00;
  }

  const shortBondFraction = minHH < 1.2 ? Math.min(1.0, (1.2 - minHH) / 0.5) : 0;
  const metallicBondFraction = meanHH < 1.5 && bondingType !== "covalent-molecular"
    ? Math.min(1.0, (1.5 - meanHH) / 0.6)
    : 0;

  let distributionType = "sparse";
  if (metallicBondFraction > 0.7) distributionType = "metallic-condensed";
  else if (metallicBondFraction > 0.3) distributionType = "intermediate";
  else if (shortBondFraction > 0.5) distributionType = "molecular";

  return {
    estimatedMeanHH: Number(meanHH.toFixed(3)),
    estimatedMinHH: Number(minHH.toFixed(3)),
    estimatedMaxHH: Number(maxHH.toFixed(3)),
    shortBondFraction: Number(shortBondFraction.toFixed(4)),
    metallicBondFraction: Number(metallicBondFraction.toFixed(4)),
    distributionType,
  };
}

function classifyCageTopology(
  hCount: number,
  hRatio: number,
  metalElements: string[],
  bondingType: HydrogenBondingType,
): CageTopology {
  const hasSodaliteFormer = metalElements.some(e => SODALITE_CAGE_ELEMENTS.includes(e));
  const hasClathFormer = metalElements.some(e => CLATHRATE_CAGE_ELEMENTS.includes(e));

  if (hCount === 0 || hRatio < 2 || (bondingType !== "cage-clathrate" && bondingType !== "metallic-network")) {
    return {
      cageType: "none",
      cageSymmetry: "N/A",
      verticesPerCage: 0,
      facesPerCage: 0,
      cageVolume: 0,
      cageCompleteness: 0,
      sodaliteCharacter: 0,
      clathrateCharacter: 0,
    };
  }

  let bestTemplate = "H8-cube";
  let sodaliteChar = 0;
  let clathrateChar = 0;

  if (hRatio >= 8 && hasSodaliteFormer) {
    bestTemplate = "H24-sodalite";
    sodaliteChar = 0.95;
    clathrateChar = 0.3;
  } else if (hRatio >= 6 && hasClathFormer && !hasSodaliteFormer) {
    bestTemplate = "H32-clathrate-I";
    sodaliteChar = 0.2;
    clathrateChar = 0.95;
  } else if (hRatio >= 6 && hasClathFormer) {
    bestTemplate = "H20-dodecahedron";
    sodaliteChar = 0.6;
    clathrateChar = 0.7;
  } else if (hRatio >= 6) {
    bestTemplate = "H16-truncated-tetrahedron";
    sodaliteChar = 0.3;
    clathrateChar = 0.5;
  } else if (hRatio >= 4) {
    bestTemplate = "H12-icosahedron";
    sodaliteChar = 0.2;
    clathrateChar = 0.4;
  } else if (hRatio >= 3) {
    bestTemplate = "H8-cube";
    sodaliteChar = 0.1;
    clathrateChar = 0.2;
  } else {
    bestTemplate = "H6-octahedron";
    sodaliteChar = 0.05;
    clathrateChar = 0.1;
  }

  const template = CAGE_TEMPLATES[bestTemplate];
  const completeness = Math.min(1.0, hCount / template.vertices);

  const metalRadiiSum = metalElements.reduce((s, e) => {
    const d = getElementData(e);
    return s + (d ? d.atomicRadius : 150);
  }, 0);
  const avgMetalRadius = metalElements.length > 0 ? metalRadiiSum / metalElements.length : 150;
  const volumeScale = Math.pow(avgMetalRadius / 150, 3);

  return {
    cageType: bestTemplate,
    cageSymmetry: template.symmetry,
    verticesPerCage: template.vertices,
    facesPerCage: template.faces,
    cageVolume: Number((template.baseVolume * volumeScale).toFixed(2)),
    cageCompleteness: Number(completeness.toFixed(4)),
    sodaliteCharacter: Number(sodaliteChar.toFixed(4)),
    clathrateCharacter: Number(clathrateChar.toFixed(4)),
  };
}

function computeNetworkDimensionality(
  hRatio: number,
  bondingType: HydrogenBondingType,
  hhDist: HHDistanceDistribution,
): number {
  if (bondingType === "none") return 0;
  if (bondingType === "covalent-molecular") return 0.5;
  if (bondingType === "interstitial") return 1.0 + Math.min(1.0, hRatio * 0.3);

  let dim = 1.0;

  if (hhDist.metallicBondFraction > 0.7) {
    dim = 3.0;
  } else if (hhDist.metallicBondFraction > 0.3) {
    dim = 2.0 + hhDist.metallicBondFraction;
  } else if (hRatio >= 6) {
    dim = 2.5 + Math.min(0.5, (hRatio - 6) * 0.1);
  } else if (hRatio >= 4) {
    dim = 2.0 + (hRatio - 4) * 0.25;
  } else if (hRatio >= 2) {
    dim = 1.5 + (hRatio - 2) * 0.25;
  }

  return Number(Math.min(3.0, dim).toFixed(2));
}

function computeCoordinationNumber(
  hRatio: number,
  bondingType: HydrogenBondingType,
  cageTopology: CageTopology,
): number {
  if (bondingType === "none") return 0;
  if (bondingType === "covalent-molecular") return 1;
  if (bondingType === "interstitial") return Math.min(6, Math.round(2 + hRatio * 0.5));

  let coord = 2;

  if (cageTopology.cageType.includes("sodalite") || cageTopology.cageType.includes("clathrate")) {
    const vertices = cageTopology.verticesPerCage;
    if (vertices >= 24) coord = 4;
    else if (vertices >= 16) coord = 3;
    else coord = 3;

    if (cageTopology.cageCompleteness > 0.8) coord += 1;
  } else if (hRatio >= 6) {
    coord = 4;
    if (hRatio >= 10) coord = 5;
  } else if (hRatio >= 4) {
    coord = 3;
  }

  const maxCoord = (hRatio >= 6) ? 9 : 6;
  return Math.min(maxCoord, coord);
}

function computeHydrogenConnectivity(
  networkDim: number,
  coordination: number,
  hhDist: HHDistanceDistribution,
  cageTopology: CageTopology,
): number {
  const dimContrib = networkDim / 3.0;
  const coordContrib = coordination / 6.0;
  const bondContrib = hhDist.metallicBondFraction;
  const cageContrib = Math.max(cageTopology.sodaliteCharacter, cageTopology.clathrateCharacter);

  const connectivity = 0.30 * dimContrib + 0.25 * coordContrib + 0.25 * bondContrib + 0.20 * cageContrib;
  return Number(Math.min(1.0, connectivity).toFixed(4));
}

function computeHydrogenPhononCouplingScore(
  formula: string,
  hRatio: number,
  hhDist: HHDistanceDistribution,
  bondingType: HydrogenBondingType,
): number {
  if (bondingType === "none") return 0;
  if (bondingType === "covalent-molecular") return 0.05;

  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    const lambda = coupling.lambda;
    const maxPhonon = phonon.maxPhononFrequency;

    let hPhononFraction = 0;
    if (hRatio >= 6 && maxPhonon > 1000) {
      hPhononFraction = 0.7 + Math.min(0.25, (hRatio - 6) * 0.05);
    } else if (hRatio >= 4 && maxPhonon > 800) {
      hPhononFraction = 0.5 + Math.min(0.2, (hRatio - 4) * 0.1);
    } else if (hRatio >= 2) {
      hPhononFraction = 0.3 + Math.min(0.2, (hRatio - 2) * 0.05);
    } else {
      hPhononFraction = 0.1 + hRatio * 0.1;
    }

    if (hhDist.metallicBondFraction > 0.5) {
      hPhononFraction = Math.min(1.0, hPhononFraction * 1.2);
    }

    const hLambda = lambda * hPhononFraction;
    const score = Math.min(1.0, hLambda / 2.0);

    const anharmonicBonus = phonon.anharmonicityIndex > 0.3
      ? Math.min(0.15, phonon.anharmonicityIndex * 0.2)
      : 0;

    return Number(Math.min(1.0, score + anharmonicBonus).toFixed(4));
  } catch {
    let baseScore = 0.1;
    if (bondingType === "metallic-network") baseScore = 0.6;
    else if (bondingType === "cage-clathrate") baseScore = 0.5;
    else if (bondingType === "interstitial") baseScore = 0.2;
    return Number(Math.min(1.0, baseScore + hRatio * 0.03).toFixed(4));
  }
}

function computeHydrogenDensity(
  hCount: number,
  metalElements: string[],
  counts: Record<string, number>,
): number {
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0);
  if (totalAtoms === 0) return 0;

  let totalVolume = 0;
  for (const el of Object.keys(counts)) {
    const d = getElementData(el);
    const radius = d ? d.atomicRadius / 100 : 1.5;
    totalVolume += (4 / 3) * Math.PI * Math.pow(radius, 3) * counts[el];
  }

  if (totalVolume === 0) return 0;
  return Number((hCount / totalVolume).toFixed(4));
}

function computeCompositeSCScore(
  networkDim: number,
  cageScore: number,
  coordination: number,
  connectivity: number,
  phononCoupling: number,
  hhDist: HHDistanceDistribution,
  anharmonicCorrection: number = 1.0,
): number {
  const dimContrib = networkDim / 3.0;
  const cageContrib = cageScore;
  const coordContrib = coordination / 9.0;
  const connContrib = connectivity;
  const phononContrib = phononCoupling;
  const bondContrib = hhDist.metallicBondFraction;
  const anharmonicContrib = Math.min(1.0, Math.max(0, anharmonicCorrection - 0.8) / 0.4);

  const composite =
    0.13 * dimContrib +
    0.18 * cageContrib +
    0.08 * coordContrib +
    0.13 * connContrib +
    0.23 * phononContrib +
    0.13 * bondContrib +
    0.12 * anharmonicContrib;

  return Number(Math.min(1.0, composite).toFixed(4));
}

function classifyNetworkClass(
  bondingType: HydrogenBondingType,
  networkDim: number,
  cageTopology: CageTopology,
  hRatio: number,
): string {
  if (bondingType === "none") return "non-hydride";
  if (bondingType === "covalent-molecular") return "molecular-hydrogen";

  if (cageTopology.sodaliteCharacter > 0.7) return "sodalite-cage";
  if (cageTopology.clathrateCharacter > 0.7) return "clathrate-cage";
  if (cageTopology.clathrateCharacter > 0.3 && cageTopology.sodaliteCharacter > 0.3) return "mixed-cage";

  if (networkDim >= 2.5 && hRatio >= 6) return "3D-metallic-network";
  if (networkDim >= 2.0) return "2D-layered-network";
  if (networkDim >= 1.5) return "1D-chain-network";

  if (bondingType === "interstitial") return "interstitial-hydride";
  return "disordered-hydrogen";
}

function generateInsights(analysis: Omit<HydrogenNetworkAnalysis, "insights">): string[] {
  const insights: string[] = [];

  if (!analysis.isHydride) {
    insights.push("Non-hydride compound - hydrogen network analysis not applicable");
    return insights;
  }

  if (analysis.hydrogenToMetalRatio >= 10) {
    insights.push(`Extreme hydrogen ratio (H:M = ${analysis.hydrogenToMetalRatio.toFixed(1)}) suggests dense metallic hydrogen sublattice`);
  } else if (analysis.hydrogenToMetalRatio >= 6) {
    insights.push(`High hydrogen ratio (H:M = ${analysis.hydrogenToMetalRatio.toFixed(1)}) consistent with clathrate-like cage structures`);
  }

  if (analysis.hhDistribution.metallicBondFraction > 0.7) {
    insights.push(`Metallic H-H bonds dominate (${(analysis.hhDistribution.metallicBondFraction * 100).toFixed(0)}%) - strong phonon-mediated pairing expected`);
  }

  if (analysis.hydrogenNetworkDim >= 2.5) {
    insights.push(`3D hydrogen network (dim=${analysis.hydrogenNetworkDim}) provides high DOS and strong electron-phonon coupling`);
  }

  if (analysis.cageTopology.sodaliteCharacter > 0.5) {
    insights.push(`Sodalite-like cage topology (score=${analysis.cageTopology.sodaliteCharacter.toFixed(2)}) - optimal for high-Tc superconductivity`);
  }
  if (analysis.cageTopology.clathrateCharacter > 0.5) {
    insights.push(`Clathrate-like cage topology (score=${analysis.cageTopology.clathrateCharacter.toFixed(2)}) - favorable for phonon-mediated pairing`);
  }

  if (analysis.hydrogenPhononCouplingScore > 0.6) {
    insights.push(`Strong hydrogen phonon coupling (score=${analysis.hydrogenPhononCouplingScore.toFixed(3)}) - dominant contribution to lambda`);
  }

  if (analysis.Hcoordination >= 4) {
    insights.push(`High H coordination number (${analysis.Hcoordination}) indicates well-connected hydrogen sublattice`);
  }

  if (analysis.hydrogenDensity > 0.5) {
    insights.push(`High hydrogen density (${analysis.hydrogenDensity.toFixed(3)}) enhances phonon frequencies and coupling`);
  }

  if (analysis.networkPercolation > 0.8) {
    insights.push("Hydrogen network fully percolated - continuous metallic H sublattice");
  }

  if (analysis.isHydride && !analysis.percolates) {
    if (analysis.geometricPercolationUsed && analysis.percolationDetail) {
      const pd = analysis.percolationDetail;
      insights.push(`Non-percolating H network (geometric: ${pd.clusterCount} clusters, largest=${pd.largestClusterSize}/${pd.hAtomCount} H atoms, cutoff=${pd.cutoffAngstrom} A) - isolated H clusters, SC score reduced by 50%`);
    } else {
      insights.push(`Non-percolating hydrogen network (coordination×H-fraction=${(analysis.Hcoordination * analysis.hydrogenFraction).toFixed(2)} < 2.0) - SC score reduced by 50%`);
    }
  }

  return insights;
}

export function analyzeHydrogenNetwork(
  formula: string,
  atomPositions?: PercolationAtom[],
  latticeParams?: PercolationLattice,
): HydrogenNetworkAnalysis {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const hCount = counts["H"] || 0;

  const nonmetals = ["H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalElements = elements.filter(e => !nonmetals.includes(e));
  const metalAtomCount = metalElements.reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtomCount > 0 ? hCount / metalAtomCount : 0;
  const hFraction = hCount / totalAtoms;

  const isHydride = hCount > 0;

  const estimatedPressure = hRatio >= 6 ? 150 : hRatio >= 4 ? 100 : hRatio >= 2 ? 50 : 0;
  const bondingType = classifyHydrogenBonding(formula, estimatedPressure);

  const hhDist = estimateHHDistances(hCount, metalElements, metalAtomCount, hRatio, bondingType);

  const cageTopology = classifyCageTopology(hCount, hRatio, metalElements, bondingType);

  const networkDim = computeNetworkDimensionality(hRatio, bondingType, hhDist);

  const cageScore = Math.max(cageTopology.sodaliteCharacter, cageTopology.clathrateCharacter);

  const coordination = computeCoordinationNumber(hRatio, bondingType, cageTopology);

  const connectivity = computeHydrogenConnectivity(networkDim, coordination, hhDist, cageTopology);

  const phononCoupling = computeHydrogenPhononCouplingScore(formula, hRatio, hhDist, bondingType);

  const hydrogenDensity = computeHydrogenDensity(hCount, metalElements, counts);

  let networkPercolation = 0;
  if (bondingType === "metallic-network") {
    networkPercolation = Math.min(1.0, 0.7 + hRatio * 0.03);
  } else if (bondingType === "cage-clathrate") {
    networkPercolation = Math.min(1.0, 0.5 + cageTopology.cageCompleteness * 0.4);
  } else if (bondingType === "interstitial") {
    networkPercolation = Math.min(0.6, hRatio * 0.15);
  }

  let percolates = false;
  let geometricPercolationUsed = false;
  let percolationDetail: HydrogenPercolationResult | undefined;
  if (isHydride && atomPositions && latticeParams && atomPositions.length > 0) {
    const percResult = checkHydrogenPercolation(atomPositions, latticeParams, undefined, estimatedPressure);
    percolates = percResult.percolates3D || percResult.percolationConfidence >= 0.5;
    geometricPercolationUsed = true;
    percolationDetail = percResult;
    if (percResult.percolates3D) {
      networkPercolation = Math.max(networkPercolation, percResult.largestClusterFraction);
    } else {
      networkPercolation = percResult.percolationConfidence * percResult.largestClusterFraction;
    }
  } else {
    const percolationMetric = coordination * hFraction;
    percolates = isHydride ? percolationMetric > 2.0 : false;
  }

  let anharmonicCorrectionValue = 1.0;
  try {
    const elec = computeElectronicStructure(formula, null);
    const phon = computePhononSpectrum(formula, elec);
    const coup = computeElectronPhononCoupling(elec, phon, formula, 0);
    anharmonicCorrectionValue = coup.anharmonicCorrectionFactor;
  } catch {}

  let compositeSCScore = computeCompositeSCScore(
    networkDim, cageScore, coordination, connectivity, phononCoupling, hhDist, anharmonicCorrectionValue
  );

  if (isHydride && !percolates) {
    compositeSCScore = Number((compositeSCScore * 0.5).toFixed(4));
  }

  const networkClass = classifyNetworkClass(bondingType, networkDim, cageTopology, hRatio);

  let hydrogenPhononFreq = 0;
  let hydrogenPhononLambda = 0;
  let anharmonicCorrection = 1.0;
  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    hydrogenPhononFreq = phonon.maxPhononFrequency * (hRatio >= 4 ? 1.0 : 0.5);
    hydrogenPhononLambda = coupling.lambda * (hRatio >= 6 ? 0.7 : hRatio >= 4 ? 0.5 : 0.3);
    anharmonicCorrection = coupling.anharmonicCorrectionFactor;
  } catch {}

  const partial: Omit<HydrogenNetworkAnalysis, "insights"> = {
    formula,
    isHydride,
    hydrogenCount: hCount,
    hydrogenFraction: Number(hFraction.toFixed(4)),
    hydrogenToMetalRatio: Number(hRatio.toFixed(2)),
    bondingType,
    hydrogenNetworkDim: networkDim,
    hydrogenCageScore: Number(cageScore.toFixed(4)),
    Hcoordination: coordination,
    hydrogenConnectivity: Number(connectivity.toFixed(4)),
    hydrogenPhononCouplingScore: phononCoupling,
    hhDistribution: hhDist,
    cageTopology,
    hydrogenDensity,
    networkPercolation: Number(networkPercolation.toFixed(4)),
    percolates,
    geometricPercolationUsed,
    percolationDetail,
    phononContribution: {
      hydrogenPhononFreq: Number(hydrogenPhononFreq.toFixed(2)),
      hydrogenPhononLambda: Number(hydrogenPhononLambda.toFixed(4)),
      anharmonicCorrection: Number(anharmonicCorrection.toFixed(4)),
    },
    compositeSCScore,
    networkClass,
  };

  const insights = generateInsights(partial);

  return { ...partial, insights };
}

export function extractHydrogenNetworkFeatures(formula: string): Record<string, number> {
  const analysis = analyzeHydrogenNetwork(formula);
  return {
    hydrogenNetworkDim: analysis.hydrogenNetworkDim,
    hydrogenCageScore: analysis.hydrogenCageScore,
    Hcoordination: analysis.Hcoordination,
    hydrogenConnectivity: analysis.hydrogenConnectivity,
    hydrogenPhononCouplingScore: analysis.hydrogenPhononCouplingScore,
  };
}

let totalAnalyzed = 0;
let totalHydrides = 0;
let networkClassCounts: Record<string, number> = {};
let avgCompositeSCScore = 0;

export function trackHydrogenNetworkResult(analysis: HydrogenNetworkAnalysis): void {
  totalAnalyzed++;
  if (analysis.isHydride) totalHydrides++;
  networkClassCounts[analysis.networkClass] = (networkClassCounts[analysis.networkClass] || 0) + 1;
  avgCompositeSCScore = (avgCompositeSCScore * (totalAnalyzed - 1) + analysis.compositeSCScore) / totalAnalyzed;
}

export function getHydrogenNetworkStats(): {
  totalAnalyzed: number;
  totalHydrides: number;
  networkClassDistribution: Record<string, number>;
  avgCompositeSCScore: number;
} {
  return {
    totalAnalyzed,
    totalHydrides,
    networkClassDistribution: { ...networkClassCounts },
    avgCompositeSCScore: Number(avgCompositeSCScore.toFixed(4)),
  };
}
