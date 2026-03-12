import {
  ELEMENTAL_DATA,
  getElementData,
  getCompositionWeightedProperty,
  getAverageMass,
  isTransitionMetal,
  isRareEarth,
  isActinide,
} from "../learning/elemental-data";
import {
  getEntryByFormula,
} from "../crystal/crystal-structure-dataset";

const HBAR_SQ_OVER_M = 7.62; // ℏ²/m in eV·Å²

interface SKPairParams {
  sss: number;
  sps: number;
  pps: number;
  ppp: number;
  sds: number;
  pds: number;
  pdp: number;
  dds: number;
  ddp: number;
  ddd: number;
  r0: number;
}

const skPairCache = new Map<string, SKPairParams>();

const COVERED_ELEMENTS = new Set([
  "H", "Li", "B", "C", "N", "O", "Mg", "Al", "Si", "Ca",
  "Ti", "V", "Cr", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Sr",
  "Y", "Zr", "Nb", "Mo", "Ru", "Pd", "In", "Sn", "Ba", "La",
  "Ce", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Pb", "Bi", "Th",
  "Be", "Na", "K", "Sc", "Mn", "Ge", "As", "Se", "Rb", "Ag", "Te",
]);

function getOrbitalCount(el: string): number {
  if (isTransitionMetal(el) || isRareEarth(el) || isActinide(el)) return 9;
  const data = getElementData(el);
  if (!data) return 4;
  if (data.atomicNumber <= 2) return 1;
  return 4;
}

function getEquilibriumDistance(el1: string, el2: string): number {
  const d1 = getElementData(el1);
  const d2 = getElementData(el2);
  const r1 = d1 ? d1.atomicRadius / 100 : 1.5;
  const r2 = d2 ? d2.atomicRadius / 100 : 1.5;
  return (r1 + r2) * 0.85;
}

function computeHarrisonSK(el1: string, el2: string): SKPairParams {
  const key = el1 < el2 ? `${el1}-${el2}` : `${el2}-${el1}`;
  const cached = skPairCache.get(key);
  if (cached) return cached;

  const r0 = getEquilibriumDistance(el1, el2);
  const d2 = r0 * r0;
  const invD2 = HBAR_SQ_OVER_M / d2;

  const hasDEl1 = isTransitionMetal(el1) || isRareEarth(el1) || isActinide(el1);
  const hasDEl2 = isTransitionMetal(el2) || isRareEarth(el2) || isActinide(el2);
  const hasDEither = hasDEl1 || hasDEl2;

  const d1 = getElementData(el1);
  const d2data = getElementData(el2);
  const ve1 = d1?.valenceElectrons ?? 4;
  const ve2 = d2data?.valenceElectrons ?? 4;
  const veAvg = (ve1 + ve2) / 2;

  const LATE_TM = new Set(["Cu", "Ag", "Au", "Zn", "Pd", "Pt", "Ni"]);
  const isLateTM1 = LATE_TM.has(el1);
  const isLateTM2 = LATE_TM.has(el2);
  const lateTMDamping = (isLateTM1 && isLateTM2) ? 0.4 : (isLateTM1 || isLateTM2) ? 0.65 : 1.0;
  const dScale = hasDEither ? (veAvg > 5 ? 1.3 : 1.0) * lateTMDamping : 0.05;

  const params: SKPairParams = {
    sss: -1.32 * invD2,
    sps: 1.42 * invD2,
    pps: 2.22 * invD2,
    ppp: -0.63 * invD2,
    sds: -1.08 * invD2 * dScale,
    pds: -1.36 * invD2 * dScale,
    pdp: 0.60 * invD2 * dScale,
    dds: -1.81 * invD2 * dScale,
    ddp: 1.10 * invD2 * dScale,
    ddd: -0.35 * invD2 * dScale,
    r0,
  };

  skPairCache.set(key, params);
  return params;
}

function getSKParams(el1: string, el2: string): SKPairParams {
  return computeHarrisonSK(el1, el2);
}

function getOnsiteEnergies(el: string): { es: number; ep: number; ed: number } {
  const data = getElementData(el);
  if (!data) return { es: -8.0, ep: -4.0, ed: 0.0 };
  const ie = data.firstIonizationEnergy;
  const ea = data.electronAffinity ?? 0;
  const en = data.paulingElectronegativity ?? 2.0;
  const es = -(ie * 0.5 + ea * 0.5) * 0.8;
  const ep = es + 3.0 + en * 0.5;
  const ed = (isTransitionMetal(el) || isRareEarth(el) || isActinide(el))
    ? es + 1.5 + (data.valenceElectrons - 2) * 0.3
    : es + 8.0;
  return { es, ep, ed };
}

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

interface TBGraphNode {
  element: string;
  orbitalCount: number;
  orbitalStart: number;
  position: [number, number, number];
}

interface TBGraphEdge {
  source: number;
  target: number;
  sk: SKPairParams;
  displacement: [number, number, number];
  distance: number;
}

interface TBGraph {
  nodes: TBGraphNode[];
  edges: TBGraphEdge[];
  totalOrbitals: number;
  latticeVectors: number[][];
  formula: string;
}

function guessLatticeType(elements: string[]): string {
  if (elements.length === 1) {
    const el = elements[0];
    if (["Fe", "Cr", "V", "Nb", "Mo", "W", "Ta", "Na", "K", "Li", "Ba"].includes(el)) return "bcc";
    if (["Cu", "Ag", "Au", "Al", "Ni", "Pd", "Pt", "Pb", "Ca", "Sr"].includes(el)) return "fcc";
    if (["Ti", "Zr", "Hf", "Co", "Zn", "Mg", "Be", "Y", "Sc"].includes(el)) return "hexagonal";
  }
  if (elements.includes("B") && elements.some(e => isTransitionMetal(e) || isRareEarth(e))) return "hexagonal";
  if (elements.length >= 3 && elements.includes("O")) return "cubic";
  return "cubic";
}

function getDefaultLatticeVectors(latticeType: string, a: number): number[][] {
  switch (latticeType) {
    case "bcc":
      return [
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
      ];
    case "fcc":
      return [
        [0, a / 2, a / 2],
        [a / 2, 0, a / 2],
        [a / 2, a / 2, 0],
      ];
    case "hexagonal":
      return [
        [a, 0, 0],
        [-a / 2, a * Math.sqrt(3) / 2, 0],
        [0, 0, a * 1.6],
      ];
    default:
      return [
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
      ];
  }
}

function getNeighborDisplacements(latticeType: string): number[][] {
  switch (latticeType) {
    case "bcc":
      return [
        [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
        [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5],
      ];
    case "fcc":
      return [
        [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0],
        [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 0, -0.5],
        [0, 0.5, 0.5], [0, 0.5, -0.5], [0, -0.5, 0.5], [0, -0.5, -0.5],
      ];
    case "hexagonal":
      return [
        [1, 0, 0], [-1, 0, 0],
        [0.5, Math.sqrt(3) / 2, 0], [-0.5, -Math.sqrt(3) / 2, 0],
        [-0.5, Math.sqrt(3) / 2, 0], [0.5, -Math.sqrt(3) / 2, 0],
        [0, 0, 0.5], [0, 0, -0.5],
        [0.5, Math.sqrt(3) / 6, 0.5], [-0.5, -Math.sqrt(3) / 6, -0.5],
        [-0.5, Math.sqrt(3) / 6, 0.5], [0.5, -Math.sqrt(3) / 6, -0.5],
      ];
    default:
      return [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
      ];
  }
}

function buildTBGraph(formula: string, latticeParam?: number, crystalSystem?: string, customLatticeVectors?: number[][]): TBGraph {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const latticeType = crystalSystem ?? guessLatticeType(elements);

  let a = latticeParam ?? 3.5;
  for (const el of elements) {
    const data = getElementData(el);
    if (data?.latticeConstant) {
      a = data.latticeConstant / 100;
      break;
    }
  }

  let entry: any = null;
  try {
    entry = getEntryByFormula(formula);
  } catch {}
  if (entry) {
    a = entry.lattice.a;
  }

  const latticeVectors = customLatticeVectors && customLatticeVectors.length === 3
    ? customLatticeVectors
    : getDefaultLatticeVectors(latticeType, a);

  const nodes: TBGraphNode[] = [];
  let totalOrbitals = 0;

  for (const el of elements) {
    const count = Math.round(counts[el] || 1);
    const orbCount = getOrbitalCount(el);
    for (let i = 0; i < Math.min(count, 4); i++) {
      const pos: [number, number, number] = [
        i * 0.25 + Math.random() * 0.01,
        i * 0.25 + Math.random() * 0.01,
        i * 0.25 + Math.random() * 0.01,
      ];
      if (entry && entry.atomicPositions) {
        const matchPos = entry.atomicPositions.filter((p: any) => p.element === el);
        if (matchPos[i]) {
          pos[0] = matchPos[i].x;
          pos[1] = matchPos[i].y;
          pos[2] = matchPos[i].z;
        }
      }
      nodes.push({ element: el, orbitalCount: orbCount, orbitalStart: totalOrbitals, position: pos });
      totalOrbitals += orbCount;
    }
  }

  if (totalOrbitals > 60) {
    const newNodes: TBGraphNode[] = [];
    let orb = 0;
    for (const node of nodes) {
      if (orb + node.orbitalCount > 60) break;
      newNodes.push({ ...node, orbitalStart: orb });
      orb += node.orbitalCount;
      if (orb >= 60) break;
    }
    nodes.length = 0;
    nodes.push(...newNodes);
    totalOrbitals = orb;
  }

  const edges: TBGraphEdge[] = [];
  const neighborDisp = getNeighborDisplacements(latticeType);

  for (let i = 0; i < nodes.length; i++) {
    for (let j = 0; j < nodes.length; j++) {
      if (i === j) continue;
      const sk = getSKParams(nodes[i].element, nodes[j].element);
      for (const disp of neighborDisp) {
        const dx = (nodes[j].position[0] - nodes[i].position[0]) + disp[0];
        const dy = (nodes[j].position[1] - nodes[i].position[1]) + disp[1];
        const dz = (nodes[j].position[2] - nodes[i].position[2]) + disp[2];
        const dist = Math.sqrt(
          (dx * latticeVectors[0][0] + dy * latticeVectors[1][0] + dz * latticeVectors[2][0]) ** 2 +
          (dx * latticeVectors[0][1] + dy * latticeVectors[1][1] + dz * latticeVectors[2][1]) ** 2 +
          (dx * latticeVectors[0][2] + dy * latticeVectors[1][2] + dz * latticeVectors[2][2]) ** 2
        );
        if (dist > 0.5 && dist < 6.0) {
          edges.push({
            source: i,
            target: j,
            sk,
            displacement: [disp[0], disp[1], disp[2]],
            distance: dist,
          });
        }
      }
    }
  }

  if (edges.length === 0 && nodes.length > 1) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const sk = getSKParams(nodes[i].element, nodes[j].element);
        edges.push({ source: i, target: j, sk, displacement: [1, 0, 0], distance: sk.r0 });
        edges.push({ source: j, target: i, sk, displacement: [-1, 0, 0], distance: sk.r0 });
      }
    }
  }

  return { nodes, edges, totalOrbitals, latticeVectors, formula };
}

function buildHamiltonian(graph: TBGraph, kpoint: number[]): number[][] {
  const n = graph.totalOrbitals;
  const H: number[][] = [];
  for (let i = 0; i < n; i++) H[i] = new Array(n).fill(0);

  for (const node of graph.nodes) {
    const onsite = getOnsiteEnergies(node.element);
    const o = node.orbitalStart;
    if (o < n) H[o][o] = onsite.es;
    if (node.orbitalCount >= 4) {
      for (let p = 0; p < 3; p++) {
        if (o + 1 + p < n) H[o + 1 + p][o + 1 + p] = onsite.ep;
      }
    }
    if (node.orbitalCount >= 9) {
      for (let d = 0; d < 5; d++) {
        if (o + 4 + d < n) H[o + 4 + d][o + 4 + d] = onsite.ed;
      }
    }
  }

  const nNeighbors = Math.max(1, graph.edges.length / Math.max(1, graph.nodes.length));

  for (const edge of graph.edges) {
    const ni = graph.nodes[edge.source];
    const nj = graph.nodes[edge.target];
    const sk = edge.sk;
    const disp = edge.displacement;
    const phase = Math.cos(2 * Math.PI * (kpoint[0] * disp[0] + kpoint[1] * disp[1] + kpoint[2] * disp[2]));
    const scaleFactor = phase / nNeighbors;
    const distDecay = Math.exp(-1.5 * (edge.distance / sk.r0 - 1.0));
    const sf = scaleFactor * distDecay;

    const oi = ni.orbitalStart;
    const oj = nj.orbitalStart;

    if (oi < n && oj < n) {
      H[oi][oj] += sk.sss * sf;
    }

    if (ni.orbitalCount >= 4 && nj.orbitalCount >= 4) {
      for (let p = 0; p < 3; p++) {
        if (oi < n && oj + 1 + p < n) {
          H[oi][oj + 1 + p] += sk.sps * sf * 0.577;
        }
        if (oj < n && oi + 1 + p < n) {
          H[oj][oi + 1 + p] += sk.sps * sf * 0.577;
        }
      }
      for (let p1 = 0; p1 < 3; p1++) {
        for (let p2 = 0; p2 < 3; p2++) {
          if (oi + 1 + p1 < n && oj + 1 + p2 < n) {
            const sigW = p1 === p2 ? 1.0 / 3.0 : 0;
            const piW = p1 === p2 ? 2.0 / 3.0 : (p1 !== p2 ? -1.0 / 3.0 : 0);
            H[oi + 1 + p1][oj + 1 + p2] += (sk.pps * sigW + sk.ppp * piW) * sf;
          }
        }
      }
    }

    if (ni.orbitalCount >= 9 && nj.orbitalCount >= 9) {
      for (let d1 = 0; d1 < 5; d1++) {
        for (let d2 = 0; d2 < 5; d2++) {
          if (oi + 4 + d1 < n && oj + 4 + d2 < n) {
            let v = 0;
            if (d1 === d2) {
              v = (sk.dds * 0.2 + sk.ddp * 0.5 + sk.ddd * 0.3) * sf;
            } else {
              v = (sk.ddp * 0.3 + sk.ddd * 0.1) * sf * 0.5;
            }
            H[oi + 4 + d1][oj + 4 + d2] += v;
          }
        }
      }
    }

    if (ni.orbitalCount >= 9 && nj.orbitalCount >= 4) {
      for (let d = 0; d < 5; d++) {
        if (oi + 4 + d < n && oj < n) {
          H[oi + 4 + d][oj] += sk.sds * sf * 0.447;
        }
      }
      for (let d = 0; d < 5; d++) {
        for (let p = 0; p < 3; p++) {
          if (oi + 4 + d < n && oj + 1 + p < n) {
            const w = (d < 3 && d === p) ? 0.5 : 0.2;
            H[oi + 4 + d][oj + 1 + p] += sk.pds * sf * w;
          }
        }
      }
    }

    if (nj.orbitalCount >= 9 && ni.orbitalCount >= 4) {
      for (let d = 0; d < 5; d++) {
        if (oj + 4 + d < n && oi < n) {
          H[oj + 4 + d][oi] += sk.sds * sf * 0.447;
        }
      }
      for (let d = 0; d < 5; d++) {
        for (let p = 0; p < 3; p++) {
          if (oj + 4 + d < n && oi + 1 + p < n) {
            const w = (d < 3 && d === p) ? 0.5 : 0.2;
            H[oj + 4 + d][oi + 1 + p] += sk.pds * sf * w;
          }
        }
      }
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = (H[i][j] + H[j][i]) / 2;
      H[i][j] = avg;
      H[j][i] = avg;
    }
  }

  return H;
}

function diagonalizeHamiltonian(H: number[][]): number[] {
  const n = H.length;
  if (n === 0) return [];
  if (n === 1) return [H[0][0]];

  const diag = new Array(n);
  const offDiag = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    diag[i] = H[i][i];
  }
  for (let i = 0; i < n - 1; i++) {
    let sumSq = 0;
    for (let j = i + 1; j < n; j++) {
      sumSq += H[j][i] * H[j][i];
    }
    offDiag[i] = Math.sqrt(sumSq);
  }

  let minVal = diag[0] - Math.abs(offDiag[0] || 0);
  let maxVal = diag[0] + Math.abs(offDiag[0] || 0);
  for (let i = 1; i < n; i++) {
    const lo = diag[i] - Math.abs(offDiag[i] || 0) - Math.abs(offDiag[i - 1] || 0);
    const hi = diag[i] + Math.abs(offDiag[i] || 0) + Math.abs(offDiag[i - 1] || 0);
    if (lo < minVal) minVal = lo;
    if (hi > maxVal) maxVal = hi;
  }

  const margin = (maxVal - minVal) * 0.01;
  minVal -= margin;
  maxVal += margin;

  function countBelow(x: number): number {
    let count = 0;
    let d = 1.0;
    const STURM_EPS = 1e-14;
    for (let i = 0; i < n; i++) {
      const offSq = i > 0 ? offDiag[i - 1] * offDiag[i - 1] : 0;
      if (Math.abs(d) < STURM_EPS) {
        d = (diag[i] - x) - offSq / (d >= 0 ? STURM_EPS : -STURM_EPS);
      } else {
        d = (diag[i] - x) - offSq / d;
      }
      if (d < 0) count++;
    }
    return count;
  }

  const eigenvalues: number[] = [];
  for (let idx = 0; idx < n; idx++) {
    let lo = minVal;
    let hi = maxVal;
    for (let iter = 0; iter < 60; iter++) {
      const mid = (lo + hi) / 2;
      if (countBelow(mid) <= idx) lo = mid;
      else hi = mid;
    }
    eigenvalues.push((lo + hi) / 2);
  }

  return eigenvalues.sort((a, b) => a - b);
}

function getHighSymmetryPath(crystalSystem: string): { points: number[][]; labels: string[] } {
  switch (crystalSystem) {
    case "hexagonal":
      return {
        points: [[0, 0, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0], [0, 0, 0]],
        labels: ["Γ", "K", "M", "Γ"],
      };
    case "bcc":
      return {
        points: [[0, 0, 0], [0.5, -0.5, 0.5], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0.5, 0]],
        labels: ["Γ", "H", "P", "Γ", "N"],
      };
    case "fcc":
      return {
        points: [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.25, 0.75], [0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]],
        labels: ["Γ", "X", "W", "K", "Γ", "L"],
      };
    case "tetragonal":
      return {
        points: [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]],
        labels: ["Γ", "X", "M", "Γ", "Z", "R"],
      };
    case "orthorhombic":
      return {
        points: [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0, 0, 0], [0, 0, 0.5]],
        labels: ["Γ", "X", "S", "Y", "Γ", "Z"],
      };
    default:
      return {
        points: [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        labels: ["Γ", "X", "M", "Γ", "R"],
      };
  }
}

export interface TBBandStructureResult {
  kpoints: number[][];
  energies: number[][];
  highSymmetryLabels: { index: number; label: string }[];
  fermiEnergy: number;
  nBands: number;
  formula: string;
}

export function computeBandStructure(formula: string, nPoints: number = 30, customLatticeVectors?: number[][]): TBBandStructureResult {
  const elements = parseFormulaElements(formula);
  const crystalSystem = guessLatticeType(elements);
  const graph = buildTBGraph(formula, undefined, crystalSystem, customLatticeVectors);
  const path = getHighSymmetryPath(crystalSystem);

  const kpoints: number[][] = [];
  const highSymmetryLabels: { index: number; label: string }[] = [];

  for (let seg = 0; seg < path.points.length - 1; seg++) {
    const start = path.points[seg];
    const end = path.points[seg + 1];
    if (seg === 0) highSymmetryLabels.push({ index: 0, label: path.labels[seg] });
    for (let i = 0; i < nPoints; i++) {
      const t = i / nPoints;
      kpoints.push([
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
      ]);
    }
    highSymmetryLabels.push({ index: kpoints.length, label: path.labels[seg + 1] });
  }
  kpoints.push(path.points[path.points.length - 1]);

  const energies: number[][] = [];
  for (const k of kpoints) {
    const H = buildHamiltonian(graph, k);
    const evals = diagonalizeHamiltonian(H);
    energies.push(evals);
  }

  const counts = parseFormulaCounts(formula);
  let totalElectrons = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (data) totalElectrons += data.valenceElectrons * (counts[el] || 1);
  }
  const occupiedStates = Math.floor(totalElectrons / 2);
  const allEvals: number[] = [];
  for (const ev of energies) {
    for (const e of ev) {
      if (Number.isFinite(e)) allEvals.push(e);
    }
  }
  allEvals.sort((a, b) => a - b);
  const targetIdx = Math.min(occupiedStates * kpoints.length - 1, allEvals.length - 1);
  const fermiEnergy = targetIdx >= 0 && allEvals.length > 0 ? allEvals[Math.max(0, targetIdx)] : 0;

  return {
    kpoints,
    energies,
    highSymmetryLabels,
    fermiEnergy,
    nBands: graph.totalOrbitals,
    formula,
  };
}

export interface TBDOSResult {
  energies: number[];
  dos: number[];
  totalStates: number;
}

export function computeDOS(formula: string, nKpoints: number = 12, nBins: number = 200, customLatticeVectors?: number[][]): TBDOSResult {
  const elements = parseFormulaElements(formula);
  const crystalSystem = guessLatticeType(elements);
  const graph = buildTBGraph(formula, undefined, crystalSystem, customLatticeVectors);

  const allEigenvalues: number[] = [];
  const nk = Math.min(nKpoints, 20);
  for (let ix = 0; ix < nk; ix++) {
    for (let iy = 0; iy < nk; iy++) {
      for (let iz = 0; iz < nk; iz++) {
        const k = [(ix + 0.5) / nk, (iy + 0.5) / nk, (iz + 0.5) / nk];
        const H = buildHamiltonian(graph, k);
        const evals = diagonalizeHamiltonian(H);
        allEigenvalues.push(...evals.filter(e => Number.isFinite(e)));
      }
    }
  }

  if (allEigenvalues.length === 0) {
    return { energies: [], dos: [], totalStates: 0 };
  }

  const eMin = Math.min(...allEigenvalues);
  const eMax = Math.max(...allEigenvalues);
  const range = eMax - eMin || 1;
  const binWidth = range / nBins;
  const sigma = binWidth * 2;

  const energies: number[] = [];
  const dos: number[] = [];

  for (let i = 0; i < nBins; i++) {
    const e = eMin + (i + 0.5) * binWidth;
    energies.push(e);
    let density = 0;
    for (const ev of allEigenvalues) {
      const x = (e - ev) / sigma;
      density += Math.exp(-0.5 * x * x) / (sigma * Math.sqrt(2 * Math.PI));
    }
    dos.push(density);
  }

  const totalStates = allEigenvalues.length;
  return { energies, dos, totalStates };
}

export interface FermiProperties {
  fermiEnergy: number;
  dosAtFermi: number;
  effectiveMass: number;
  bandDegeneracy: number;
  metallicity: number;
  bandwidth: number;
}

export function computeFermiProperties(formula: string, customLatticeVectors?: number[][]): FermiProperties {
  const bandResult = computeBandStructure(formula, 40, customLatticeVectors);
  const dosResult = computeDOS(formula, 10, 200, customLatticeVectors);

  const fermiE = bandResult.fermiEnergy;

  let dosAtFermi = 0;
  if (dosResult.energies.length > 0) {
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < dosResult.energies.length; i++) {
      const dist = Math.abs(dosResult.energies[i] - fermiE);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    dosAtFermi = dosResult.dos[bestIdx] ?? 0;

    if (bestIdx > 0 && bestIdx < dosResult.dos.length - 1) {
      const w0 = 1.0 / (1.0 + bestDist * 10);
      const d1 = Math.abs(dosResult.energies[bestIdx - 1] - fermiE);
      const d2 = Math.abs(dosResult.energies[bestIdx + 1] - fermiE);
      const w1 = 1.0 / (1.0 + d1 * 10);
      const w2 = 1.0 / (1.0 + d2 * 10);
      const wSum = w0 + w1 + w2;
      dosAtFermi = (dosResult.dos[bestIdx] * w0 + dosResult.dos[bestIdx - 1] * w1 + dosResult.dos[bestIdx + 1] * w2) / wSum;
    }
  }

  let effectiveMass = 1.0;
  let maxBandwidth = 0;
  let bandDegeneracy = 0;

  if (bandResult.energies.length > 2 && bandResult.energies[0].length > 0) {
    const nBands = bandResult.energies[0].length;
    const nK = bandResult.energies.length;

    for (let b = 0; b < nBands; b++) {
      const bandEnergies = bandResult.energies.map(ev => ev[b] ?? 0);
      const bw = Math.max(...bandEnergies) - Math.min(...bandEnergies);
      if (bw > maxBandwidth) maxBandwidth = bw;

      const crossesFermi = bandEnergies.some((e, i) =>
        i > 0 && ((bandEnergies[i - 1] <= fermiE && e >= fermiE) || (bandEnergies[i - 1] >= fermiE && e <= fermiE))
      );
      if (crossesFermi) bandDegeneracy++;

      if (crossesFermi && nK > 2) {
        for (let ki = 1; ki < nK - 1; ki++) {
          const d2E = bandEnergies[ki + 1] - 2 * bandEnergies[ki] + bandEnergies[ki - 1];
          if (Math.abs(d2E) > 0.001) {
            const dk = 1.0 / nK;
            const m = HBAR_SQ_OVER_M / (d2E / (dk * dk));
            if (Number.isFinite(m) && Math.abs(m) > 0.01 && Math.abs(m) < 100) {
              effectiveMass = Math.abs(m);
              break;
            }
          }
        }
      }
    }
  }

  const metallicity = dosAtFermi > 0.1 ? Math.min(1.0, dosAtFermi / 5.0) : 0.1;

  return {
    fermiEnergy: fermiE,
    dosAtFermi,
    effectiveMass,
    bandDegeneracy,
    metallicity,
    bandwidth: maxBandwidth,
  };
}

export interface FlatBandResult {
  flatBandCount: number;
  flatBandScore: number;
  vanHoveProximity: number;
  bandwidths: number[];
}

export function detectFlatBands(bandResult: TBBandStructureResult): FlatBandResult {
  const nBands = bandResult.nBands;
  const bandwidths: number[] = [];
  let flatBandCount = 0;
  let flatBandScore = 0;

  for (let b = 0; b < nBands; b++) {
    const energies = bandResult.energies.map(ev => ev[b] ?? 0);
    const bw = Math.max(...energies) - Math.min(...energies);
    bandwidths.push(bw);
    if (bw < 0.1) {
      flatBandCount++;
      flatBandScore += (0.1 - bw) / 0.1;
    }
  }

  let vanHoveProximity = 0;
  const fermiE = bandResult.fermiEnergy;
  for (let b = 0; b < nBands; b++) {
    const energies = bandResult.energies.map(ev => ev[b] ?? 0);
    for (let ki = 1; ki < energies.length - 1; ki++) {
      const d2E = energies[ki + 1] - 2 * energies[ki] + energies[ki - 1];
      if (Math.abs(d2E) < 0.005 && Math.abs(energies[ki] - fermiE) < 0.5) {
        vanHoveProximity = Math.max(vanHoveProximity, 1.0 - Math.abs(energies[ki] - fermiE) / 0.5);
      }
    }
  }

  return {
    flatBandCount,
    flatBandScore: flatBandCount > 0 ? flatBandScore / flatBandCount : 0,
    vanHoveProximity,
    bandwidths,
  };
}

export interface ElectronPhononProxies {
  hopfieldEta: number;
  lambdaProxy: number;
  bondStiffness: number;
  debyeTemperature: number;
  avgForceConstant: number;
}

export function computeElectronPhononProxies(formula: string, fermiProps: FermiProperties): ElectronPhononProxies {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const LATE_TM_SET = new Set(["Cu", "Ag", "Au", "Zn", "Pd", "Pt", "Ni"]);
  let avgI2 = 0;
  for (const el of elements) {
    const sk = getSKParams(el, el);
    const frac = (counts[el] || 1) / totalAtoms;
    const spContrib = Math.abs(sk.sss) + Math.abs(sk.pps);
    const dContrib = Math.abs(sk.dds);
    const dWeight = LATE_TM_SET.has(el) ? 0.25 : 1.0;
    avgI2 += (spContrib + dContrib * dWeight) * frac;
  }
  avgI2 = avgI2 * avgI2;

  const hopfieldEta = fermiProps.dosAtFermi * avgI2;

  const avgMass = getAverageMass(counts);
  const debyeTemp = getCompositionWeightedProperty(counts, "debyeTemperature") ?? 300;
  const omegaD = debyeTemp * 8.617e-5;
  const omegaD2 = omegaD * omegaD;

  const lambdaProxy = avgMass > 0 && omegaD2 > 0 ? hopfieldEta / (avgMass * omegaD2 * 0.01) : 0;

  let coordination = 0;
  let avgForceConstant = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const bulk = data?.bulkModulus ?? 50;
    const frac = (counts[el] || 1) / totalAtoms;
    coordination += (isTransitionMetal(el) ? 12 : 8) * frac;
    avgForceConstant += bulk * 0.1 * frac;
  }

  const bondStiffness = coordination * avgForceConstant;

  return {
    hopfieldEta,
    lambdaProxy: Math.min(3.0, Math.max(0, lambdaProxy)),
    bondStiffness,
    debyeTemperature: debyeTemp,
    avgForceConstant,
  };
}

function birchMurnaghanVolume(pressure: number, V0: number, B0: number, B0p: number): number {
  if (pressure <= 0 || B0 <= 0) return V0;
  const pOverB = pressure / B0;
  let eta = 1.0 - pOverB * 0.5;
  for (let iter = 0; iter < 20; iter++) {
    const f = (1.0 / Math.pow(eta, 7 / 3) - 1.0 / Math.pow(eta, 5 / 3));
    const P_calc = 1.5 * B0 * f * (1 + 0.75 * (B0p - 4) * (1.0 / Math.pow(eta, 2 / 3) - 1));
    const diff = P_calc - pressure;
    if (Math.abs(diff) < 0.01) break;
    eta -= diff / (B0 * 10);
    eta = Math.max(0.5, Math.min(1.0, eta));
  }
  return V0 * eta;
}

const SOFT_ELEMENTS = new Set(["H", "Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba"]);

function estimateB0Prime(elements: string[]): number {
  if (elements.length === 0) return 4.0;
  let totalWeight = 0;
  let weightedB0p = 0;
  for (const el of elements) {
    const data = getElementData(el);
    const bulk = data?.bulkModulus ?? 50;
    let b0p: number;
    if (SOFT_ELEMENTS.has(el)) {
      b0p = el === "H" ? 3.4 : 3.6;
    } else if (isTransitionMetal(el)) {
      b0p = bulk > 200 ? 4.2 : 4.0;
    } else {
      b0p = 4.0;
    }
    const w = 1.0 / Math.max(1, bulk);
    totalWeight += w;
    weightedB0p += b0p * w;
  }
  return totalWeight > 0 ? weightedB0p / totalWeight : 4.0;
}

function getHoppingExponent(el1: string, el2: string): number {
  const hasD1 = isTransitionMetal(el1) || isRareEarth(el1) || isActinide(el1);
  const hasD2 = isTransitionMetal(el2) || isRareEarth(el2) || isActinide(el2);
  if (hasD1 && hasD2) return 2.0;
  const isLight1 = (getElementData(el1)?.atomicNumber ?? 20) <= 10;
  const isLight2 = (getElementData(el2)?.atomicNumber ?? 20) <= 10;
  if (isLight1 && isLight2) return 2.8;
  if (isLight1 || isLight2) return 2.4;
  if (!hasD1 && !hasD2) return 2.5;
  return 2.2;
}

function scaleHoppingForPressure(t0: number, r0: number, pressure: number, bulkModulus: number, elements?: string[], el1?: string, el2?: string): number {
  if (pressure <= 0) return t0;
  const V0 = r0 * r0 * r0;
  const b0p = elements ? estimateB0Prime(elements) : 4.0;
  const V = birchMurnaghanVolume(pressure, V0, bulkModulus, b0p);
  const rRatio = Math.cbrt(V0 / V);
  const n = (el1 && el2) ? getHoppingExponent(el1, el2) : 2.0;
  return t0 * Math.pow(rRatio, n);
}

export interface TBProperties {
  dosAtEF: number;
  bandFlatness: number;
  vanHoveProximity: number;
  hopfieldEta: number;
  lambdaProxy: number;
  effectiveMass: number;
  bandwidth: number;
  bandDegeneracy: number;
  metallicity: number;
  bondStiffness: number;
  debyeTemperature: number;
  fermiEnergy: number;
  flatBandCount: number;
  flatBandScore: number;
  formula: string;
  pressure: number;
  computeTimeMs: number;
}

const tbPropertiesCache = new Map<string, { result: TBProperties; timestamp: number }>();
const TB_CACHE_MAX = 500;
const TB_CACHE_TTL = 30 * 60 * 1000;

const tbStats = {
  computations: 0,
  cacheHits: 0,
  totalTimeMs: 0,
};

export function computeTBProperties(formula: string, pressure: number = 0, customLatticeVectors?: number[][]): TBProperties {
  const cacheKey = customLatticeVectors
    ? `${formula}@${pressure}@custom`
    : `${formula}@${pressure}`;
  const cached = tbPropertiesCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < TB_CACHE_TTL) {
    tbStats.cacheHits++;
    return cached.result;
  }

  const start = Date.now();
  tbStats.computations++;

  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);

  if (pressure > 0) {
    const avgBulk = getCompositionWeightedProperty(counts, "bulkModulus") ?? 100;
    Array.from(skPairCache.keys()).forEach(key => {
      const [el1, el2] = key.split("-");
      if (elements.includes(el1) || elements.includes(el2)) {
        skPairCache.delete(key);
      }
    });

    for (let i = 0; i < elements.length; i++) {
      for (let j = i; j < elements.length; j++) {
        const e1 = elements[i], e2 = elements[j];
        const base = computeHarrisonSK(e1, e2);
        const scale = scaleHoppingForPressure(1.0, base.r0, pressure, avgBulk, elements, e1, e2);
        if (Math.abs(scale - 1.0) > 0.001) {
          const key = e1 < e2 ? `${e1}-${e2}` : `${e2}-${e1}`;
          skPairCache.set(key, {
            sss: base.sss * scale, sps: base.sps * scale,
            pps: base.pps * scale, ppp: base.ppp * scale,
            sds: base.sds * scale, pds: base.pds * scale,
            pdp: base.pdp * scale, dds: base.dds * scale,
            ddp: base.ddp * scale, ddd: base.ddd * scale,
            r0: base.r0 / Math.cbrt(scale),
          });
        }
      }
    }
  }

  const bandResult = computeBandStructure(formula, 20, customLatticeVectors);
  const dosResult = computeDOS(formula, 8, 150, customLatticeVectors);
  const fermiProps = computeFermiProperties(formula, customLatticeVectors);
  const flatBandResult = detectFlatBands(bandResult);
  const ephProxies = computeElectronPhononProxies(formula, fermiProps);

  let dosAtEF = fermiProps.dosAtFermi;
  let lambdaProxy = ephProxies.lambdaProxy;
  let hopfieldEta = ephProxies.hopfieldEta;

  if (pressure > 0) {
    const avgBulk = getCompositionWeightedProperty(counts, "bulkModulus") ?? 100;
    const compressionRatio = Math.cbrt(1.0 / (1.0 + pressure / (avgBulk * 3)));

    const hFrac = (counts["H"] || 0) / (Object.values(counts).reduce((s, n) => s + n, 0) || 1);
    const isHydrideRich = hFrac > 0.5;
    const hasVanHove = flatBandResult.vanHoveProximity > 0.3;

    if (isHydrideRich) {
      const vhBoost = hasVanHove ? 0.3 : 0.15;
      dosAtEF *= (1.0 + vhBoost * pressure / 100);
      lambdaProxy *= (1.0 + 0.5 * pressure / 100) * Math.pow(1.0 / compressionRatio, 2);
      hopfieldEta *= (1.0 + 0.2 * pressure / 100);
    } else {
      const bandBroadening = 1.0 / (1.0 + 0.15 * pressure / 100);
      dosAtEF *= bandBroadening;
      if (hasVanHove && pressure > 20) {
        dosAtEF *= (1.0 + 0.1 * flatBandResult.vanHoveProximity * pressure / 100);
      }
      lambdaProxy *= Math.pow(1.0 / compressionRatio, 2);
      hopfieldEta *= bandBroadening;
    }
  }

  const computeTimeMs = Date.now() - start;
  tbStats.totalTimeMs += computeTimeMs;

  const result: TBProperties = {
    dosAtEF,
    bandFlatness: flatBandResult.flatBandScore,
    vanHoveProximity: flatBandResult.vanHoveProximity,
    hopfieldEta,
    lambdaProxy,
    effectiveMass: fermiProps.effectiveMass,
    bandwidth: fermiProps.bandwidth,
    bandDegeneracy: fermiProps.bandDegeneracy,
    metallicity: fermiProps.metallicity,
    bondStiffness: ephProxies.bondStiffness,
    debyeTemperature: ephProxies.debyeTemperature,
    fermiEnergy: fermiProps.fermiEnergy,
    flatBandCount: flatBandResult.flatBandCount,
    flatBandScore: flatBandResult.flatBandScore,
    formula,
    pressure,
    computeTimeMs,
  };

  if (tbPropertiesCache.size >= TB_CACHE_MAX) {
    let oldest = "";
    let oldestTime = Infinity;
    tbPropertiesCache.forEach((v, k) => {
      if (v.timestamp < oldestTime) {
        oldestTime = v.timestamp;
        oldest = k;
      }
    });
    if (oldest) tbPropertiesCache.delete(oldest);
  }
  tbPropertiesCache.set(cacheKey, { result, timestamp: Date.now() });

  return result;
}

export function getTBEngineStats() {
  return {
    computations: tbStats.computations,
    cacheHits: tbStats.cacheHits,
    cacheSize: tbPropertiesCache.size,
    cacheMaxSize: TB_CACHE_MAX,
    cacheTTLMinutes: TB_CACHE_TTL / 60000,
    avgComputeTimeMs: tbStats.computations > 0 ? Math.round(tbStats.totalTimeMs / tbStats.computations) : 0,
    coveredElements: COVERED_ELEMENTS.size,
    skPairCacheSize: skPairCache.size,
  };
}

export {
  buildTBGraph,
  buildHamiltonian,
  diagonalizeHamiltonian,
  getSKParams,
  COVERED_ELEMENTS,
  type TBGraph,
  type TBGraphNode,
  type TBGraphEdge,
  type SKPairParams,
};
