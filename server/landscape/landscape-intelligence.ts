import { getEmbeddingDataset, getEmbeddingPoint, type EmbeddingPoint } from "./discovery-landscape";
import { detectDiscoveryZones } from "./zone-detector";
import { encodeGenome, euclideanDistance, interpolateGenomes } from "../physics/materials-genome";
import { classifyFamily } from "../learning/utils";

export interface FrontierRegion {
  id: string;
  direction: [number, number, number];
  tcGradient: number;
  nearestPoints: string[];
  distanceFromCenter: number;
  explorationScore: number;
  suggestedElements: string[];
  suggestedFamilies: string[];
}

export interface DiscoveryCorridor {
  id: string;
  startPoint: [number, number, number];
  endPoint: [number, number, number];
  direction: [number, number, number];
  tcSlope: number;
  length: number;
  materialsAlong: string[];
  confidence: number;
}

export interface FrontierAnalysis {
  frontierRegions: FrontierRegion[];
  discoveryCorridors: DiscoveryCorridor[];
  convexHullVertices: string[];
  exploredVolumeFraction: number;
  frontierSurfaceArea: number;
  bestFrontierDirection: [number, number, number] | null;
  totalEmbeddedPoints: number;
}

export interface NoveltyScore {
  formula: string;
  overallNovelty: number;
  nearestNeighborDistance: number;
  nearestNeighborFormula: string;
  localDensity: number;
  familyDissimilarity: number;
  embeddingNovelty: number;
  genomicNovelty: number;
  isInExploredRegion: boolean;
  noveltyBreakdown: {
    distanceComponent: number;
    densityComponent: number;
    familyComponent: number;
  };
}

export interface ZoneIntelligence {
  zoneId: string;
  center3D: [number, number, number];
  tcMean: number;
  tcVariance: number;
  uncertainty: number;
  acquisitionScore: number;
  materialCount: number;
  evolutionTrend: "improving" | "stable" | "declining" | "new";
  correlatedZones: string[];
  suggestedElements: string[];
  suggestedStructures: string[];
  suggestedStoichiometries: string[];
  explorationPriority: "high" | "medium" | "low";
}

export interface ExplorationStrategy {
  targetZones: ZoneIntelligence[];
  interpolationCandidates: {
    formulaA: string;
    formulaB: string;
    interpolatedVector: number[];
    estimatedTc: number;
    suggestedElements: string[];
  }[];
  bridgeCandidates: {
    clusterA: string[];
    clusterB: string[];
    bridgePoint: [number, number, number];
    suggestedElements: string[];
    potentialTc: number;
  }[];
  overallRecommendation: string;
  explorationBudgetAllocation: Record<string, number>;
}

interface ZoneHistoryEntry {
  zoneId: string;
  cycle: number;
  tcMean: number;
  materialCount: number;
  acquisitionScore: number;
  timestamp: number;
}

const zoneHistory: ZoneHistoryEntry[] = [];
let lastIntelligenceCycle = 0;
const UCB_BETA = 1.5;

function computeCenter(points: EmbeddingPoint[]): [number, number, number] {
  if (points.length === 0) return [0, 0, 0];
  const center: [number, number, number] = [0, 0, 0];
  for (const p of points) {
    for (let d = 0; d < 3; d++) center[d] += p.position3D[d];
  }
  for (let d = 0; d < 3; d++) center[d] /= points.length;
  return center;
}

function dist3D(a: [number, number, number], b: [number, number, number]): number {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

function extractElements(formula: string): string[] {
  return formula.match(/[A-Z][a-z]?/g) ?? [];
}

function isConvexHullVertex(point: EmbeddingPoint, allPoints: EmbeddingPoint[]): boolean {
  const pos = point.position3D;
  for (let d = 0; d < 3; d++) {
    let isExtreme = true;
    for (const other of allPoints) {
      if (other.formula === point.formula) continue;
      if (other.position3D[d] > pos[d]) { isExtreme = false; break; }
    }
    if (isExtreme) return true;

    isExtreme = true;
    for (const other of allPoints) {
      if (other.formula === point.formula) continue;
      if (other.position3D[d] < pos[d]) { isExtreme = false; break; }
    }
    if (isExtreme) return true;
  }
  return false;
}

export function analyzeFrontier(): FrontierAnalysis {
  const points = getEmbeddingDataset();
  if (points.length < 5) {
    return {
      frontierRegions: [],
      discoveryCorridors: [],
      convexHullVertices: [],
      exploredVolumeFraction: 0,
      frontierSurfaceArea: 0,
      bestFrontierDirection: null,
      totalEmbeddedPoints: points.length,
    };
  }

  const center = computeCenter(points);

  const hullVertices = points.filter(p => isConvexHullVertex(p, points));
  const hullFormulas = hullVertices.map(p => p.formula);

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let d = 0; d < 3; d++) {
      min[d] = Math.min(min[d], p.position3D[d]);
      max[d] = Math.max(max[d], p.position3D[d]);
    }
  }

  const totalVolume = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]);
  const gridRes = 6;
  const cellSize: [number, number, number] = [
    (max[0] - min[0]) / gridRes || 1,
    (max[1] - min[1]) / gridRes || 1,
    (max[2] - min[2]) / gridRes || 1,
  ];

  const occupiedCells = new Set<string>();
  for (const p of points) {
    const ix = Math.min(gridRes - 1, Math.max(0, Math.floor((p.position3D[0] - min[0]) / cellSize[0])));
    const iy = Math.min(gridRes - 1, Math.max(0, Math.floor((p.position3D[1] - min[1]) / cellSize[1])));
    const iz = Math.min(gridRes - 1, Math.max(0, Math.floor((p.position3D[2] - min[2]) / cellSize[2])));
    occupiedCells.add(`${ix}-${iy}-${iz}`);
  }

  const totalCells = gridRes ** 3;
  const exploredVolumeFraction = occupiedCells.size / totalCells;

  const nDirections = 12;
  const frontierRegions: FrontierRegion[] = [];

  for (let i = 0; i < nDirections; i++) {
    const phi = (i / nDirections) * 2 * Math.PI;
    const theta = Math.acos(1 - 2 * ((i * 0.618) % 1));
    const direction: [number, number, number] = [
      Math.sin(theta) * Math.cos(phi),
      Math.sin(theta) * Math.sin(phi),
      Math.cos(theta),
    ];

    const projections = points.map(p => {
      const proj = direction[0] * (p.position3D[0] - center[0]) +
                   direction[1] * (p.position3D[1] - center[1]) +
                   direction[2] * (p.position3D[2] - center[2]);
      return { point: p, proj };
    });
    projections.sort((a, b) => b.proj - a.proj);

    const outerPoints = projections.slice(0, Math.max(3, Math.floor(points.length * 0.1)));
    const innerPoints = projections.slice(Math.floor(points.length * 0.5));

    const outerAvgTc = outerPoints.reduce((s, p) => s + p.point.tc, 0) / outerPoints.length;
    const innerAvgTc = innerPoints.length > 0
      ? innerPoints.reduce((s, p) => s + p.point.tc, 0) / innerPoints.length : 0;
    const tcGradient = outerAvgTc - innerAvgTc;

    const outerDist = outerPoints.length > 0 ? outerPoints[0].proj : 0;

    const elements = new Map<string, number>();
    const families = new Map<string, number>();
    for (const op of outerPoints) {
      for (const el of extractElements(op.point.formula)) {
        elements.set(el, (elements.get(el) ?? 0) + 1);
      }
      families.set(op.point.family, (families.get(op.point.family) ?? 0) + 1);
    }

    const topElements = Array.from(elements.entries())
      .sort((a, b) => b[1] - a[1]).slice(0, 6).map(e => e[0]);
    const topFamilies = Array.from(families.entries())
      .sort((a, b) => b[1] - a[1]).slice(0, 3).map(f => f[0]);

    const explorationScore = Math.min(1,
      Math.max(0, tcGradient / 50) * 0.4 +
      Math.max(0, outerDist / 10) * 0.3 +
      (1 - exploredVolumeFraction) * 0.3
    );

    frontierRegions.push({
      id: `frontier-${i + 1}`,
      direction: direction.map(v => Math.round(v * 1000) / 1000) as [number, number, number],
      tcGradient: Math.round(tcGradient * 10) / 10,
      nearestPoints: outerPoints.slice(0, 5).map(p => p.point.formula),
      distanceFromCenter: Math.round(outerDist * 100) / 100,
      explorationScore: Math.round(explorationScore * 1000) / 1000,
      suggestedElements: topElements,
      suggestedFamilies: topFamilies,
    });
  }

  frontierRegions.sort((a, b) => b.explorationScore - a.explorationScore);

  const discoveryCorridors: DiscoveryCorridor[] = [];
  const highTcPoints = points.filter(p => p.tc > 30).sort((a, b) => b.tc - a.tc);

  for (let i = 0; i < Math.min(highTcPoints.length, 5); i++) {
    for (let j = i + 1; j < Math.min(highTcPoints.length, 8); j++) {
      const pA = highTcPoints[i];
      const pB = highTcPoints[j];
      const d = dist3D(pA.position3D, pB.position3D);
      if (d < 1 || d > 15) continue;

      const direction: [number, number, number] = [
        (pB.position3D[0] - pA.position3D[0]) / d,
        (pB.position3D[1] - pA.position3D[1]) / d,
        (pB.position3D[2] - pA.position3D[2]) / d,
      ];

      const materialsAlong: string[] = [];
      for (const p of points) {
        const toP: [number, number, number] = [
          p.position3D[0] - pA.position3D[0],
          p.position3D[1] - pA.position3D[1],
          p.position3D[2] - pA.position3D[2],
        ];
        const projLen = toP[0] * direction[0] + toP[1] * direction[1] + toP[2] * direction[2];
        if (projLen < 0 || projLen > d) continue;
        const perpDist = Math.sqrt(
          (toP[0] - projLen * direction[0]) ** 2 +
          (toP[1] - projLen * direction[1]) ** 2 +
          (toP[2] - projLen * direction[2]) ** 2
        );
        if (perpDist < 2) materialsAlong.push(p.formula);
      }

      const tcSlope = (pB.tc - pA.tc) / d;
      const confidence = Math.min(1, materialsAlong.length / 5) * Math.min(1, Math.abs(tcSlope) / 10);

      discoveryCorridors.push({
        id: `corridor-${discoveryCorridors.length + 1}`,
        startPoint: pA.position3D,
        endPoint: pB.position3D,
        direction: direction.map(v => Math.round(v * 1000) / 1000) as [number, number, number],
        tcSlope: Math.round(tcSlope * 100) / 100,
        length: Math.round(d * 100) / 100,
        materialsAlong: materialsAlong.slice(0, 10),
        confidence: Math.round(confidence * 1000) / 1000,
      });
    }
  }

  discoveryCorridors.sort((a, b) => b.confidence - a.confidence);

  const bestFrontier = frontierRegions.length > 0 ? frontierRegions[0].direction : null;

  const frontierSurfaceArea = hullVertices.length > 2
    ? hullVertices.length * (totalVolume / totalCells) * 6 : 0;

  return {
    frontierRegions: frontierRegions.slice(0, 10),
    discoveryCorridors: discoveryCorridors.slice(0, 8),
    convexHullVertices: hullFormulas,
    exploredVolumeFraction: Math.round(exploredVolumeFraction * 1000) / 1000,
    frontierSurfaceArea: Math.round(frontierSurfaceArea * 100) / 100,
    bestFrontierDirection: bestFrontier,
    totalEmbeddedPoints: points.length,
  };
}

export function computeNoveltyScore(formula: string): NoveltyScore {
  const points = getEmbeddingDataset();

  const defaultResult: NoveltyScore = {
    formula,
    overallNovelty: 1.0,
    nearestNeighborDistance: Infinity,
    nearestNeighborFormula: "",
    localDensity: 0,
    familyDissimilarity: 1.0,
    embeddingNovelty: 1.0,
    genomicNovelty: 1.0,
    isInExploredRegion: false,
    noveltyBreakdown: { distanceComponent: 1.0, densityComponent: 1.0, familyComponent: 1.0 },
  };

  if (points.length < 3) return defaultResult;

  let genome: { vector: number[] };
  try {
    genome = encodeGenome(formula);
    if (!genome || !genome.vector || genome.vector.length === 0) return defaultResult;
  } catch {
    return defaultResult;
  }

  let nearestDist = Infinity;
  let nearestFormula = "";
  const distances: number[] = [];

  for (const p of points) {
    const d = euclideanDistance(genome.vector, p.genomeVector);
    distances.push(d);
    if (d < nearestDist && p.formula !== formula) {
      nearestDist = d;
      nearestFormula = p.formula;
    }
  }

  const sortedDists = [...distances].sort((a, b) => a - b);
  const medianDist = sortedDists[Math.floor(sortedDists.length / 2)];
  const maxDist = sortedDists[sortedDists.length - 1] || 1;

  const distanceComponent = Math.min(1, nearestDist / (medianDist + 0.01));

  const nearbyCount = distances.filter(d => d < medianDist * 0.5).length;
  const localDensity = Math.min(1, nearbyCount / Math.max(1, points.length));
  const densityComponent = Math.max(0, 1 - localDensity);

  const candidateFamily = classifyFamily(formula);
  const familyCounts = new Map<string, number>();
  for (const p of points) {
    familyCounts.set(p.family, (familyCounts.get(p.family) ?? 0) + 1);
  }
  const sameFamilyCount = familyCounts.get(candidateFamily) ?? 0;
  const familyFraction = sameFamilyCount / points.length;
  const familyComponent = Math.max(0, 1 - familyFraction * 2);

  const existingPoint = getEmbeddingPoint(formula);
  const embeddingNovelty = existingPoint
    ? Math.min(1, nearestDist / (maxDist * 0.3))
    : 1.0;

  const genomicNovelty = Math.min(1, nearestDist / (maxDist * 0.25));

  const overallNovelty = distanceComponent * 0.4 + densityComponent * 0.35 + familyComponent * 0.25;
  const isInExploredRegion = nearestDist < medianDist * 0.3 && localDensity > 0.5;

  return {
    formula,
    overallNovelty: Math.round(overallNovelty * 1000) / 1000,
    nearestNeighborDistance: Math.round(nearestDist * 1000) / 1000,
    nearestNeighborFormula: nearestFormula,
    localDensity: Math.round(localDensity * 1000) / 1000,
    familyDissimilarity: Math.round(familyComponent * 1000) / 1000,
    embeddingNovelty: Math.round(embeddingNovelty * 1000) / 1000,
    genomicNovelty: Math.round(genomicNovelty * 1000) / 1000,
    isInExploredRegion,
    noveltyBreakdown: {
      distanceComponent: Math.round(distanceComponent * 1000) / 1000,
      densityComponent: Math.round(densityComponent * 1000) / 1000,
      familyComponent: Math.round(familyComponent * 1000) / 1000,
    },
  };
}

export function analyzeZoneIntelligence(): ZoneIntelligence[] {
  const zones = detectDiscoveryZones(20);
  const points = getEmbeddingDataset();
  if (zones.length === 0 || points.length < 5) return [];

  const zoneIntelligence: ZoneIntelligence[] = [];

  for (const zone of zones) {
    const nearbyPoints = points.filter(p =>
      dist3D(p.position3D, zone.center3D) < 4.0
    );

    const tcs = nearbyPoints.map(p => p.tc);
    const tcMean = tcs.length > 0 ? tcs.reduce((s, t) => s + t, 0) / tcs.length : 0;
    const tcVariance = tcs.length > 1
      ? tcs.reduce((s, t) => s + (t - tcMean) ** 2, 0) / (tcs.length - 1) : 0;
    const uncertainty = Math.sqrt(tcVariance);

    const acquisitionScore = tcMean + UCB_BETA * uncertainty;

    const historyForZone = zoneHistory.filter(h => h.zoneId === zone.id);
    let evolutionTrend: "improving" | "stable" | "declining" | "new" = "new";
    if (historyForZone.length >= 2) {
      const recent = historyForZone[historyForZone.length - 1];
      const older = historyForZone[historyForZone.length - 2];
      const tcDelta = recent.tcMean - older.tcMean;
      if (tcDelta > 2) evolutionTrend = "improving";
      else if (tcDelta < -2) evolutionTrend = "declining";
      else evolutionTrend = "stable";
    }

    const correlatedZones: string[] = [];
    for (const otherZone of zones) {
      if (otherZone.id === zone.id) continue;
      const zDist = dist3D(zone.center3D, otherZone.center3D);
      if (zDist < 6) {
        const sharedElements = zone.suggestedElements.filter(e =>
          otherZone.suggestedElements.includes(e)
        );
        if (sharedElements.length >= 2) {
          correlatedZones.push(otherZone.id);
        }
      }
    }

    const elementCounts = new Map<string, number>();
    const familyCounts = new Map<string, number>();
    for (const p of nearbyPoints) {
      for (const el of extractElements(p.formula)) {
        elementCounts.set(el, (elementCounts.get(el) ?? 0) + 1);
      }
      familyCounts.set(p.family, (familyCounts.get(p.family) ?? 0) + 1);
    }

    const topElements = Array.from(elementCounts.entries())
      .sort((a, b) => b[1] - a[1]).slice(0, 8).map(e => e[0]);

    const topFamilies = Array.from(familyCounts.entries())
      .sort((a, b) => b[1] - a[1]).slice(0, 3).map(f => f[0]);

    const STRUCTURE_SUGGESTIONS: Record<string, string[]> = {
      Hydrides: ["clathrate", "sodalite", "layered-H"],
      Cuprates: ["perovskite", "CuO2-plane", "infinite-layer"],
      Borides: ["AlB2-type", "hexagonal-layer", "MgB2-type"],
      Carbides: ["NaCl-rocksalt", "A15", "MAX-phase"],
      Nitrides: ["NaCl-rocksalt", "perovskite", "anti-perovskite"],
      Pnictides: ["ThCr2Si2", "FeAs-layer", "1111-type"],
      Intermetallics: ["A15", "Laves-C15", "BCC"],
      Kagome: ["kagome-flat", "breathing-kagome", "pyrochlore"],
    };

    const suggestedStructures: string[] = [];
    for (const fam of topFamilies) {
      const structs = STRUCTURE_SUGGESTIONS[fam];
      if (structs) suggestedStructures.push(...structs);
    }

    const suggestedStoichiometries: string[] = [];
    if (topElements.length >= 2) {
      const [a, b] = topElements;
      suggestedStoichiometries.push(`${a}${b}3`, `${a}2${b}`, `${a}3${b}`);
      if (topElements.length >= 3) {
        const c = topElements[2];
        suggestedStoichiometries.push(`${a}${b}2${c}2`, `${a}${b}${c}3`);
      }
    }

    const priority: "high" | "medium" | "low" =
      acquisitionScore > 60 ? "high" : acquisitionScore > 25 ? "medium" : "low";

    zoneIntelligence.push({
      zoneId: zone.id,
      center3D: zone.center3D,
      tcMean: Math.round(tcMean * 10) / 10,
      tcVariance: Math.round(tcVariance * 10) / 10,
      uncertainty: Math.round(uncertainty * 10) / 10,
      acquisitionScore: Math.round(acquisitionScore * 10) / 10,
      materialCount: nearbyPoints.length,
      evolutionTrend,
      correlatedZones,
      suggestedElements: topElements,
      suggestedStructures: Array.from(new Set(suggestedStructures)).slice(0, 5),
      suggestedStoichiometries: suggestedStoichiometries.slice(0, 5),
      explorationPriority: priority,
    });
  }

  zoneIntelligence.sort((a, b) => b.acquisitionScore - a.acquisitionScore);
  return zoneIntelligence;
}

export function generateExplorationStrategy(): ExplorationStrategy {
  const zoneIntel = analyzeZoneIntelligence();
  const points = getEmbeddingDataset();
  const targetZones = zoneIntel.filter(z => z.explorationPriority !== "low");

  const interpolationCandidates: ExplorationStrategy["interpolationCandidates"] = [];
  const highTcPoints = points.filter(p => p.tc > 30).sort((a, b) => b.tc - a.tc);

  for (let i = 0; i < Math.min(highTcPoints.length, 6); i++) {
    for (let j = i + 1; j < Math.min(highTcPoints.length, 8); j++) {
      const pA = highTcPoints[i];
      const pB = highTcPoints[j];

      if (pA.family === pB.family) continue;

      try {
        const interpolated = interpolateGenomes(pA.formula, pB.formula, 0.5);
        const estimatedTc = (pA.tc + pB.tc) / 2 * 1.1;

        const elementsA = extractElements(pA.formula);
        const elementsB = extractElements(pB.formula);
        const combinedElements = Array.from(new Set([...elementsA, ...elementsB])).slice(0, 6);

        const resolvedFormulas = interpolated.nearestFormulas.map(n => n.formula);

        interpolationCandidates.push({
          formulaA: pA.formula,
          formulaB: pB.formula,
          interpolatedVector: interpolated.vector.slice(0, 10),
          estimatedTc: Math.round(estimatedTc * 10) / 10,
          suggestedElements: combinedElements.length > 0 ? combinedElements : resolvedFormulas.slice(0, 3),
        });
      } catch {}
    }
  }

  interpolationCandidates.sort((a, b) => b.estimatedTc - a.estimatedTc);

  const bridgeCandidates: ExplorationStrategy["bridgeCandidates"] = [];
  const familyGroups = new Map<string, EmbeddingPoint[]>();
  for (const p of points) {
    if (!familyGroups.has(p.family)) familyGroups.set(p.family, []);
    familyGroups.get(p.family)!.push(p);
  }

  const familyKeys = Array.from(familyGroups.keys());
  for (let i = 0; i < familyKeys.length; i++) {
    for (let j = i + 1; j < familyKeys.length; j++) {
      const groupA = familyGroups.get(familyKeys[i])!;
      const groupB = familyGroups.get(familyKeys[j])!;

      const highTcA = groupA.filter(p => p.tc > 20).sort((a, b) => b.tc - a.tc);
      const highTcB = groupB.filter(p => p.tc > 20).sort((a, b) => b.tc - a.tc);

      if (highTcA.length === 0 || highTcB.length === 0) continue;

      const bestA = highTcA[0];
      const bestB = highTcB[0];

      const bridgePoint: [number, number, number] = [
        (bestA.position3D[0] + bestB.position3D[0]) / 2,
        (bestA.position3D[1] + bestB.position3D[1]) / 2,
        (bestA.position3D[2] + bestB.position3D[2]) / 2,
      ];

      const elementsA = extractElements(bestA.formula);
      const elementsB = extractElements(bestB.formula);
      const bridgeElements = Array.from(new Set([...elementsA, ...elementsB])).slice(0, 5);

      bridgeCandidates.push({
        clusterA: highTcA.slice(0, 3).map(p => p.formula),
        clusterB: highTcB.slice(0, 3).map(p => p.formula),
        bridgePoint: bridgePoint.map(v => Math.round(v * 100) / 100) as [number, number, number],
        suggestedElements: bridgeElements,
        potentialTc: Math.round((bestA.tc + bestB.tc) / 2 * 10) / 10,
      });
    }
  }

  bridgeCandidates.sort((a, b) => b.potentialTc - a.potentialTc);

  const budgetAllocation: Record<string, number> = {};
  const totalScore = targetZones.reduce((s, z) => s + z.acquisitionScore, 0) || 1;
  for (const z of targetZones) {
    budgetAllocation[z.zoneId] = Math.round((z.acquisitionScore / totalScore) * 100) / 100;
  }

  let recommendation = "Insufficient data for specific recommendations. Continue broad exploration.";
  if (targetZones.length > 0) {
    const best = targetZones[0];
    recommendation = `Focus on zone ${best.zoneId} (Tc_mean=${best.tcMean}K, uncertainty=${best.uncertainty}K). ` +
      `Suggested elements: ${best.suggestedElements.slice(0, 4).join(", ")}. ` +
      `Structures: ${best.suggestedStructures.slice(0, 3).join(", ")}. ` +
      `${targetZones.length} total high-priority zones. ` +
      (bridgeCandidates.length > 0
        ? `${bridgeCandidates.length} bridge candidates between clusters detected.`
        : "No inter-cluster bridges found yet.");
  }

  return {
    targetZones: targetZones.slice(0, 10),
    interpolationCandidates: interpolationCandidates.slice(0, 10),
    bridgeCandidates: bridgeCandidates.slice(0, 8),
    overallRecommendation: recommendation,
    explorationBudgetAllocation: budgetAllocation,
  };
}

export function updateZoneHistory(cycle: number): void {
  const zones = analyzeZoneIntelligence();
  const now = Date.now();

  for (const z of zones) {
    zoneHistory.push({
      zoneId: z.zoneId,
      cycle,
      tcMean: z.tcMean,
      materialCount: z.materialCount,
      acquisitionScore: z.acquisitionScore,
      timestamp: now,
    });
  }

  if (zoneHistory.length > 500) {
    zoneHistory.splice(0, zoneHistory.length - 500);
  }

  lastIntelligenceCycle = cycle;
}

export function getLandscapeIntelligenceStats(): {
  lastCycle: number;
  zoneHistoryLength: number;
  frontierRegionCount: number;
  highPriorityZoneCount: number;
  totalEmbeddedPoints: number;
} {
  const points = getEmbeddingDataset();
  const zones = analyzeZoneIntelligence();
  return {
    lastCycle: lastIntelligenceCycle,
    zoneHistoryLength: zoneHistory.length,
    frontierRegionCount: analyzeFrontier().frontierRegions.length,
    highPriorityZoneCount: zones.filter(z => z.explorationPriority === "high").length,
    totalEmbeddedPoints: points.length,
  };
}

export function getIntelligenceGeneratorBias(): {
  elementBias: Record<string, number>;
  familyBias: Record<string, number>;
  structureBias: Record<string, number>;
  explorationRatio: number;
} {
  const strategy = generateExplorationStrategy();
  const elementBias: Record<string, number> = {};
  const familyBias: Record<string, number> = {};
  const structureBias: Record<string, number> = {};

  for (const zone of strategy.targetZones) {
    const weight = zone.acquisitionScore / 100;
    for (const el of zone.suggestedElements) {
      elementBias[el] = Math.max(elementBias[el] ?? 0, weight);
    }
    for (const struct of zone.suggestedStructures) {
      structureBias[struct] = Math.max(structureBias[struct] ?? 0, weight);
    }
  }

  const zones = analyzeZoneIntelligence();
  for (const z of zones) {
    if (z.suggestedElements.length > 0) {
      const family = classifyFamily(z.suggestedElements.join(""));
      familyBias[family] = Math.max(familyBias[family] ?? 0, z.acquisitionScore / 100);
    }
  }

  const maxEl = Math.max(...Object.values(elementBias), 1);
  for (const key of Object.keys(elementBias)) elementBias[key] /= maxEl;

  const maxFam = Math.max(...Object.values(familyBias), 1);
  for (const key of Object.keys(familyBias)) familyBias[key] /= maxFam;

  const maxStruct = Math.max(...Object.values(structureBias), 1);
  for (const key of Object.keys(structureBias)) structureBias[key] /= maxStruct;

  const points = getEmbeddingDataset();
  const explorationRatio = points.length < 20 ? 0.8 : Math.max(0.2, 0.6 - strategy.targetZones.length * 0.04);

  return { elementBias, familyBias, structureBias, explorationRatio };
}
