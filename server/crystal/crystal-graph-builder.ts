import { ELEMENTAL_DATA, getElementData } from "../learning/elemental-data";
import { buildCrystalGraph, type CrystalGraph, type NodeFeature, type EdgeFeature, type ThreeBodyFeature } from "../learning/graph-neural-net";

export interface CrystalStructureEntry {
  formula: string;
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  atomicPositions: { element: string; x: number; y: number; z: number }[];
  spacegroup: number;
  spacegroupSymbol: string;
  crystalSystem: string;
  prototype: string;
  formationEnergy: number;
  volume: number;
  density: number;
  nsites: number;
  source: string;
}

export interface StructureNodeFeature extends NodeFeature {
  covalentRadius: number;
  period: number;
  group: number;
  block: string;
  ionizationEnergy: number;
  electronAffinity: number;
  mendeleevNumber: number;
  fractionalPosition: { x: number; y: number; z: number };
}

export interface StructureEdgeFeature extends EdgeFeature {
  gaussianExpansion: number[];
  bondType: "ionic" | "covalent" | "metallic";
  coordinationNumber: number;
}

export interface StructureGraph extends CrystalGraph {
  nodes: StructureNodeFeature[];
  edges: StructureEdgeFeature[];
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number };
  spacegroup?: number;
  crystalSystem?: string;
  hasRealPositions: boolean;
}

const N_GAUSSIAN_BASIS = 20;
const GAUSSIAN_START = 0.5;
const GAUSSIAN_END = 8.0;
const GAUSSIAN_WIDTH = 0.5;
const DISTANCE_CUTOFF = 6.0;
const NODE_DIM = 32;

function gaussianDistanceExpansion(distance: number): number[] {
  const step = (GAUSSIAN_END - GAUSSIAN_START) / (N_GAUSSIAN_BASIS - 1);
  const basis: number[] = [];
  for (let i = 0; i < N_GAUSSIAN_BASIS; i++) {
    const center = GAUSSIAN_START + i * step;
    const diff = distance - center;
    basis.push(Math.exp(-(diff * diff) / (2 * GAUSSIAN_WIDTH * GAUSSIAN_WIDTH)));
  }
  return basis;
}

function getPeriod(atomicNumber: number): number {
  if (atomicNumber <= 2) return 1;
  if (atomicNumber <= 10) return 2;
  if (atomicNumber <= 18) return 3;
  if (atomicNumber <= 36) return 4;
  if (atomicNumber <= 54) return 5;
  if (atomicNumber <= 86) return 6;
  return 7;
}

function getGroup(atomicNumber: number): number {
  const groupMap: Record<number, number> = {
    1: 1, 2: 18, 3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,
    11: 1, 12: 2, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10,
    29: 11, 30: 12, 31: 13, 32: 14, 33: 15, 34: 16, 35: 17, 36: 18,
    37: 1, 38: 2, 39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10,
    47: 11, 48: 12, 49: 13, 50: 14, 51: 15, 52: 16, 53: 17, 54: 18,
    55: 1, 56: 2, 72: 4, 73: 5, 74: 6, 75: 7, 76: 8, 77: 9, 78: 10,
    79: 11, 80: 12, 81: 13, 82: 14, 83: 15,
  };
  return groupMap[atomicNumber] ?? 0;
}

function getBlock(atomicNumber: number): string {
  if (atomicNumber <= 2) return "s";
  if ([3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88].includes(atomicNumber)) return "s";
  if (atomicNumber >= 57 && atomicNumber <= 71) return "f";
  if (atomicNumber >= 89 && atomicNumber <= 103) return "f";
  if ((atomicNumber >= 21 && atomicNumber <= 30) || (atomicNumber >= 39 && atomicNumber <= 48) || (atomicNumber >= 72 && atomicNumber <= 80)) return "d";
  return "p";
}

function getCovalentRadius(atomicNumber: number, atomicRadius: number): number {
  const covalentRadii: Record<number, number> = {
    1: 31, 2: 28, 3: 128, 4: 96, 5: 84, 6: 76, 7: 71, 8: 66, 9: 57, 10: 58,
    11: 166, 12: 141, 13: 121, 14: 111, 15: 107, 16: 105, 17: 102, 18: 106,
    19: 203, 20: 176, 21: 170, 22: 160, 23: 153, 24: 139, 25: 150, 26: 142,
    27: 138, 28: 124, 29: 132, 30: 122, 31: 122, 32: 120, 33: 119, 34: 120,
    35: 120, 36: 116, 37: 220, 38: 195, 39: 190, 40: 175, 41: 164, 42: 154,
    43: 147, 44: 146, 45: 142, 46: 139, 47: 145, 48: 144, 49: 142, 50: 139,
    51: 139, 52: 138, 53: 139, 54: 140, 55: 244, 56: 215, 57: 207, 72: 175,
    73: 170, 74: 162, 75: 151, 76: 144, 77: 141, 78: 136, 79: 136, 80: 132,
    81: 145, 82: 146, 83: 148, 90: 206, 92: 196,
  };
  return covalentRadii[atomicNumber] ?? (atomicRadius * 0.85);
}

function getMendeleevNumber(atomicNumber: number): number {
  const mendeleevMap: Record<number, number> = {
    1: 92, 2: 98, 3: 1, 4: 67, 5: 72, 6: 77, 7: 82, 8: 87, 9: 93, 10: 99,
    11: 2, 12: 68, 13: 73, 14: 78, 15: 83, 16: 88, 17: 94, 18: 100, 19: 3,
    20: 7, 21: 11, 22: 43, 23: 44, 24: 45, 25: 46, 26: 47, 27: 48, 28: 49,
    29: 50, 30: 69, 31: 74, 32: 79, 33: 84, 34: 89, 35: 95, 36: 101, 37: 4,
    38: 8, 39: 12, 40: 51, 41: 52, 42: 53, 43: 54, 44: 55, 45: 56, 46: 57,
    47: 58, 48: 70, 49: 75, 50: 80, 51: 85, 52: 90, 53: 96, 54: 102, 55: 5,
    56: 9, 57: 13, 72: 59, 73: 60, 74: 61, 75: 62, 76: 63, 77: 64, 78: 65,
    79: 66, 80: 71, 81: 76, 82: 81, 83: 86, 90: 16, 92: 17,
  };
  return mendeleevMap[atomicNumber] ?? atomicNumber;
}

function latticeVectorsFromParams(lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }): number[][] {
  const { a, b, c, alpha, beta, gamma } = lattice;
  const alphaRad = (alpha * Math.PI) / 180;
  const betaRad = (beta * Math.PI) / 180;
  const gammaRad = (gamma * Math.PI) / 180;

  const cosAlpha = Math.cos(alphaRad);
  const cosBeta = Math.cos(betaRad);
  const cosGamma = Math.cos(gammaRad);
  const sinGamma = Math.sin(gammaRad);

  const v1 = [a, 0, 0];
  const v2 = [b * cosGamma, b * sinGamma, 0];
  const cx = c * cosBeta;
  const cy = c * (cosAlpha - cosBeta * cosGamma) / sinGamma;
  const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
  const v3 = [cx, cy, cz];

  return [v1, v2, v3];
}

function fractionalToCartesian(frac: { x: number; y: number; z: number }, latticeVectors: number[][]): [number, number, number] {
  const [v1, v2, v3] = latticeVectors;
  const x = frac.x * v1[0] + frac.y * v2[0] + frac.z * v3[0];
  const y = frac.x * v1[1] + frac.y * v2[1] + frac.z * v3[1];
  const z = frac.x * v1[2] + frac.y * v2[2] + frac.z * v3[2];
  return [x, y, z];
}

function minimumImageDistance(
  pos1: { x: number; y: number; z: number },
  pos2: { x: number; y: number; z: number },
  latticeVectors: number[][]
): number {
  let dx = pos2.x - pos1.x;
  let dy = pos2.y - pos1.y;
  let dz = pos2.z - pos1.z;

  dx -= Math.round(dx);
  dy -= Math.round(dy);
  dz -= Math.round(dz);

  const [v1, v2, v3] = latticeVectors;
  const cartX = dx * v1[0] + dy * v2[0] + dz * v3[0];
  const cartY = dx * v1[1] + dy * v2[1] + dz * v3[1];
  const cartZ = dx * v1[2] + dy * v2[2] + dz * v3[2];

  return Math.sqrt(cartX * cartX + cartY * cartY + cartZ * cartZ);
}

function classifyBondType(enDiff: number): "ionic" | "covalent" | "metallic" {
  if (enDiff > 1.7) return "ionic";
  if (enDiff > 0.4) return "covalent";
  return "metallic";
}

function buildNodeFeature(
  element: string,
  position: { x: number; y: number; z: number }
): StructureNodeFeature {
  const data = getElementData(element);
  const atomicNumber = data?.atomicNumber ?? 1;
  const en = data?.paulingElectronegativity ?? 1.5;
  const radius = data?.atomicRadius ?? 130;
  const valence = data?.valenceElectrons ?? 1;
  const mass = data?.atomicMass ?? 1;
  const covalentR = getCovalentRadius(atomicNumber, radius);
  const period = getPeriod(atomicNumber);
  const group = getGroup(atomicNumber);
  const block = getBlock(atomicNumber);
  const ionizationEnergy = data?.firstIonizationEnergy ?? 7;
  const electronAff = data?.electronAffinity ?? 0;
  const mendeleev = getMendeleevNumber(atomicNumber);

  const embedding = [
    atomicNumber / 100,
    en / 4.0,
    radius / 250,
    valence / 8,
    mass / 250,
    (data?.debyeTemperature ?? 300) / 2000,
    (data?.bulkModulus ?? 50) / 500,
    ionizationEnergy / 25,
    mendeleev / 103,
    Math.max(0, electronAff) / 4.0,
    covalentR / 250,
    period / 7.0,
    group / 18.0,
    block === "s" ? 0 : block === "p" ? 0.25 : block === "d" ? 0.5 : 0.75,
    position.x,
    position.y,
    position.z,
    Math.min(1.0, (data?.meltingPoint ?? 1000) / 4000),
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ];

  while (embedding.length < NODE_DIM) embedding.push(0);

  return {
    element,
    atomicNumber,
    electronegativity: en,
    atomicRadius: radius,
    valenceElectrons: valence,
    mass,
    embedding: embedding.slice(0, NODE_DIM),
    covalentRadius: covalentR,
    period,
    group,
    block,
    ionizationEnergy,
    electronAffinity: electronAff,
    mendeleevNumber: mendeleev,
    fractionalPosition: { ...position },
  };
}

export function computeVoronoiNeighbors(
  positions: { element: string; x: number; y: number; z: number }[],
  lattice: { a: number; b: number; c: number; alpha: number; beta: number; gamma: number }
): { i: number; j: number; distance: number; weight: number }[] {
  const latticeVectors = latticeVectorsFromParams(lattice);
  const n = positions.length;
  const neighbors: { i: number; j: number; distance: number; weight: number }[] = [];

  for (let i = 0; i < n; i++) {
    const distances: { j: number; dist: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dist = minimumImageDistance(positions[i], positions[j], latticeVectors);
      if (dist < DISTANCE_CUTOFF && dist > 0.1) {
        distances.push({ j, dist });
      }
    }

    distances.sort((a, b) => a.dist - b.dist);

    if (distances.length === 0) continue;

    const minDist = distances[0].dist;
    const voronoiCutoff = minDist * 1.8;

    for (const { j, dist } of distances) {
      if (dist > voronoiCutoff) break;
      const weight = 1.0 / (dist * dist);
      neighbors.push({ i, j, distance: dist, weight });
    }
  }

  return neighbors;
}

export function buildGraphFromStructure(entry: CrystalStructureEntry): StructureGraph {
  const latticeVectors = latticeVectorsFromParams(entry.lattice);
  const positions = entry.atomicPositions;

  const nodes: StructureNodeFeature[] = positions.map(pos =>
    buildNodeFeature(pos.element, { x: pos.x, y: pos.y, z: pos.z })
  );

  if (nodes.length === 0) {
    const defaultNode = buildNodeFeature("X", { x: 0, y: 0, z: 0 });
    nodes.push(defaultNode);
  }

  const edges: StructureEdgeFeature[] = [];
  const adjacency: number[][] = nodes.map(() => []);
  const coordinationCounts: number[] = new Array(nodes.length).fill(0);

  const voronoiNeighbors = computeVoronoiNeighbors(positions, entry.lattice);

  const addedEdges = new Set<string>();

  for (const { i, j, distance } of voronoiNeighbors) {
    const key = `${Math.min(i, j)}-${Math.max(i, j)}`;
    if (addedEdges.has(key)) continue;
    addedEdges.add(key);

    const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
    const bondType = classifyBondType(enDiff);
    const gaussianExp = gaussianDistanceExpansion(distance);
    const bondOrderEst = enDiff > 1.5 ? 0.5 : enDiff > 0.5 ? 1.0 : 1.5;
    const ionicCharacter = Math.min(1.0, enDiff / 2.5);
    const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;

    const features = [
      ...gaussianExp,
      bondOrderEst / 2.0,
      enDiff / 3.0,
      ionicCharacter,
      radiusSum,
    ];

    coordinationCounts[i]++;
    coordinationCounts[j]++;

    const edgeI: StructureEdgeFeature = {
      source: i,
      target: j,
      distance,
      bondOrderEstimate: bondOrderEst,
      features,
      gaussianExpansion: gaussianExp,
      bondType,
      coordinationNumber: 0,
    };
    const edgeJ: StructureEdgeFeature = {
      source: j,
      target: i,
      distance,
      bondOrderEstimate: bondOrderEst,
      features,
      gaussianExpansion: gaussianExp,
      bondType,
      coordinationNumber: 0,
    };

    edges.push(edgeI);
    edges.push(edgeJ);
    adjacency[i].push(j);
    adjacency[j].push(i);
  }

  if (edges.length === 0 && nodes.length > 1) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dist = minimumImageDistance(
          positions[i],
          positions[j],
          latticeVectors
        );
        if (dist < DISTANCE_CUTOFF && dist > 0.1) {
          const enDiff = Math.abs(nodes[i].electronegativity - nodes[j].electronegativity);
          const bondType = classifyBondType(enDiff);
          const gaussianExp = gaussianDistanceExpansion(dist);
          const bondOrderEst = enDiff > 1.5 ? 0.5 : enDiff > 0.5 ? 1.0 : 1.5;
          const ionicCharacter = Math.min(1.0, enDiff / 2.5);
          const radiusSum = (nodes[i].atomicRadius + nodes[j].atomicRadius) / 500;

          const features = [...gaussianExp, bondOrderEst / 2.0, enDiff / 3.0, ionicCharacter, radiusSum];

          coordinationCounts[i]++;
          coordinationCounts[j]++;

          edges.push({
            source: i, target: j, distance: dist, bondOrderEstimate: bondOrderEst,
            features, gaussianExpansion: gaussianExp, bondType, coordinationNumber: 0,
          });
          edges.push({
            source: j, target: i, distance: dist, bondOrderEstimate: bondOrderEst,
            features, gaussianExpansion: gaussianExp, bondType, coordinationNumber: 0,
          });
          adjacency[i].push(j);
          adjacency[j].push(i);
        }
      }
    }
  }

  for (const edge of edges) {
    edge.coordinationNumber = coordinationCounts[edge.source];
  }

  const threeBodyFeatures = computeThreeBody(nodes, edges, adjacency);

  return {
    nodes,
    edges,
    threeBodyFeatures,
    adjacency,
    formula: entry.formula,
    prototype: entry.prototype,
    lattice: { ...entry.lattice },
    spacegroup: entry.spacegroup,
    crystalSystem: entry.crystalSystem,
    hasRealPositions: true,
  };
}

function computeThreeBody(
  nodes: StructureNodeFeature[],
  edges: StructureEdgeFeature[],
  adjacency: number[][]
): ThreeBodyFeature[] {
  const features: ThreeBodyFeature[] = [];
  const edgeMap = new Map<string, number>();

  for (const edge of edges) {
    edgeMap.set(`${edge.source}-${edge.target}`, edge.distance);
  }

  for (let center = 0; center < nodes.length; center++) {
    const neighbors = adjacency[center];
    if (neighbors.length < 2) continue;

    const maxPairs = Math.min(neighbors.length, 8);
    for (let a = 0; a < maxPairs; a++) {
      for (let b = a + 1; b < maxPairs; b++) {
        const n1 = neighbors[a];
        const n2 = neighbors[b];
        const d1 = edgeMap.get(`${center}-${n1}`) ?? edgeMap.get(`${n1}-${center}`) ?? 2.5;
        const d2 = edgeMap.get(`${center}-${n2}`) ?? edgeMap.get(`${n2}-${center}`) ?? 2.5;
        const d12 = edgeMap.get(`${n1}-${n2}`) ?? edgeMap.get(`${n2}-${n1}`) ?? Math.sqrt(d1 * d1 + d2 * d2);

        let cosAngle = (d1 * d1 + d2 * d2 - d12 * d12) / (2 * d1 * d2);
        cosAngle = Math.max(-1, Math.min(1, cosAngle));
        const angle = Math.acos(cosAngle);

        features.push({ center, neighbor1: n1, neighbor2: n2, angle, distance1: d1, distance2: d2 });
      }
    }
  }

  return features;
}

export function buildGraphFromFormula(formula: string, pressureGpa?: number): CrystalGraph {
  return buildCrystalGraph(formula, undefined, pressureGpa);
}

export function getGraphFeatureVector(graph: StructureGraph | CrystalGraph): number[] {
  const nodes = graph.nodes;
  const edges = graph.edges;
  const nNodes = nodes.length;
  const nEdges = edges.length;

  const atomicNumbers = nodes.map(n => n.atomicNumber);
  const electronegativities = nodes.map(n => n.electronegativity);
  const radii = nodes.map(n => n.atomicRadius);
  const valences = nodes.map(n => n.valenceElectrons);
  const masses = nodes.map(n => n.mass);

  const mean = (arr: number[]) => arr.length > 0 ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;
  const std = (arr: number[]) => {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) * (v - m), 0) / arr.length);
  };

  const nodeFeats = [
    mean(atomicNumbers) / 100,
    std(atomicNumbers) / 100,
    mean(electronegativities) / 4,
    std(electronegativities) / 4,
    mean(radii) / 250,
    std(radii) / 250,
    mean(valences) / 8,
    std(valences) / 8,
    mean(masses) / 250,
    std(masses) / 250,
  ];

  const distances = edges.map(e => e.distance);
  const bondOrders = edges.map(e => e.bondOrderEstimate);

  const edgeFeats = [
    mean(distances) / 6,
    std(distances) / 6,
    Math.min(...(distances.length > 0 ? distances : [0])) / 6,
    Math.max(...(distances.length > 0 ? distances : [0])) / 6,
    mean(bondOrders) / 2,
    std(bondOrders) / 2,
  ];

  const coordNumbers = graph.adjacency.map(adj => adj.length);
  const coordFeats = [
    mean(coordNumbers) / 12,
    std(coordNumbers) / 12,
    Math.min(...(coordNumbers.length > 0 ? coordNumbers : [0])) / 12,
    Math.max(...(coordNumbers.length > 0 ? coordNumbers : [0])) / 12,
  ];

  const structGraph = graph as StructureGraph;
  let structFeats: number[];

  if (structGraph.hasRealPositions && structGraph.lattice) {
    const { a, b, c, alpha, beta, gamma } = structGraph.lattice;
    const volume = a * b * c;
    structFeats = [
      a / 20,
      b / 20,
      c / 20,
      alpha / 180,
      beta / 180,
      gamma / 180,
      Math.cbrt(volume) / 20,
      nNodes / 50,
    ];
  } else {
    structFeats = [0, 0, 0, 0.5, 0.5, 0.5, 0, nNodes / 50];
  }

  const ionicCount = edges.filter(e => {
    const se = e as StructureEdgeFeature;
    return se.bondType === "ionic";
  }).length;
  const covalentCount = edges.filter(e => {
    const se = e as StructureEdgeFeature;
    return se.bondType === "covalent";
  }).length;
  const metallicCount = edges.filter(e => {
    const se = e as StructureEdgeFeature;
    return se.bondType === "metallic";
  }).length;
  const totalBonds = Math.max(1, ionicCount + covalentCount + metallicCount);

  const bondTypeFeats = [
    ionicCount / totalBonds,
    covalentCount / totalBonds,
    metallicCount / totalBonds,
  ];

  const tbFeatures = graph.threeBodyFeatures;
  const angles = tbFeatures.map(tb => tb.angle);
  const angleFeats = [
    mean(angles) / Math.PI,
    std(angles) / Math.PI,
  ];

  const uniqueElements = new Set(nodes.map(n => n.element)).size;
  const compositionFeats = [
    uniqueElements / 8,
    nEdges / Math.max(1, nNodes * nNodes),
  ];

  const featureVector = [
    ...nodeFeats,
    ...edgeFeats,
    ...coordFeats,
    ...structFeats,
    ...bondTypeFeats,
    ...angleFeats,
    ...compositionFeats,
  ];

  return featureVector;
}

let crystalDatasetModule: any = null;

async function tryLoadDataset(): Promise<any> {
  if (crystalDatasetModule) return crystalDatasetModule;
  try {
    crystalDatasetModule = await import("./crystal-structure-dataset");
    return crystalDatasetModule;
  } catch {
    return null;
  }
}

export async function buildBestGraph(formula: string, pressureGpa?: number): Promise<StructureGraph | CrystalGraph> {
  const dataset = await tryLoadDataset();

  if (dataset) {
    try {
      const entries: CrystalStructureEntry[] = dataset.getTrainingData?.() ?? [];
      const normalizedFormula = formula.replace(/\s/g, "");
      const match = entries.find(
        (e: CrystalStructureEntry) => e.formula.replace(/\s/g, "") === normalizedFormula
      );
      if (match && match.atomicPositions && match.atomicPositions.length > 0) {
        return buildGraphFromStructure(match);
      }
    } catch {
    }
  }

  return buildGraphFromFormula(formula, pressureGpa);
}
