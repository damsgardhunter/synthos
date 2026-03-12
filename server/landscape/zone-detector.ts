import { getEmbeddingDataset, type EmbeddingPoint } from "./discovery-landscape";
import { getVarianceSummary } from "../learning/uncertainty-tracker";

export interface DiscoveryZone {
  id: string;
  center3D: [number, number, number];
  zoneScore: number;
  avgTc: number;
  maxTc: number;
  density: number;
  materialCount: number;
  scFraction: number;
  gradientDirection: [number, number, number];
  gradientMagnitude: number;
  representativeFormulas: string[];
  suggestedElements: string[];
  suggestedFamilies: string[];
  explorationPriority: "high" | "medium" | "low";
}

export interface ZoneMap {
  zones: DiscoveryZone[];
  totalMaterials: number;
  gridResolution: number;
  coveragePercent: number;
  topZones: DiscoveryZone[];
  suggestions: string[];
}

interface Voxel {
  ix: number;
  iy: number;
  iz: number;
  center: [number, number, number];
  materials: EmbeddingPoint[];
  density: number;
  avgTc: number;
  maxTc: number;
  scFraction: number;
  tcGradient: [number, number, number];
  gradientMag: number;
}

const GRID_RESOLUTION = 8;
let scTcThreshold = 20;

export function setScTcThreshold(threshold: number): void {
  scTcThreshold = Math.max(0.1, threshold);
}

export function getScTcThreshold(): number {
  return scTcThreshold;
}

function buildVoxelGrid(
  points: EmbeddingPoint[],
  resolution: number,
): { voxels: Map<string, Voxel>; bounds: { min: [number, number, number]; max: [number, number, number] }; cellSize: [number, number, number] } {
  if (points.length === 0) {
    return {
      voxels: new Map(),
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      cellSize: [1, 1, 1],
    };
  }

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let d = 0; d < 3; d++) {
      min[d] = Math.min(min[d], p.position3D[d]);
      max[d] = Math.max(max[d], p.position3D[d]);
    }
  }

  const padding = 0.5;
  for (let d = 0; d < 3; d++) {
    min[d] -= padding;
    max[d] += padding;
    if (max[d] - min[d] < 1) { max[d] = min[d] + 1; }
  }

  const cellSize: [number, number, number] = [
    (max[0] - min[0]) / resolution,
    (max[1] - min[1]) / resolution,
    (max[2] - min[2]) / resolution,
  ];

  const voxels = new Map<string, Voxel>();

  for (const p of points) {
    const ix = Math.min(resolution - 1, Math.max(0, Math.floor((p.position3D[0] - min[0]) / cellSize[0])));
    const iy = Math.min(resolution - 1, Math.max(0, Math.floor((p.position3D[1] - min[1]) / cellSize[1])));
    const iz = Math.min(resolution - 1, Math.max(0, Math.floor((p.position3D[2] - min[2]) / cellSize[2])));

    const key = `${ix}-${iy}-${iz}`;
    if (!voxels.has(key)) {
      voxels.set(key, {
        ix, iy, iz,
        center: [
          min[0] + (ix + 0.5) * cellSize[0],
          min[1] + (iy + 0.5) * cellSize[1],
          min[2] + (iz + 0.5) * cellSize[2],
        ],
        materials: [],
        density: 0,
        avgTc: 0,
        maxTc: 0,
        scFraction: 0,
        tcGradient: [0, 0, 0],
        gradientMag: 0,
      });
    }
    voxels.get(key)!.materials.push(p);
  }

  for (const voxel of voxels.values()) {
    const mats = voxel.materials;
    voxel.density = mats.length;
    voxel.avgTc = mats.reduce((s, m) => s + m.tc, 0) / mats.length;
    voxel.maxTc = Math.max(...mats.map(m => m.tc));
    voxel.scFraction = mats.filter(m => m.tc >= scTcThreshold).length / mats.length;
  }

  computeTcGradients(voxels, resolution);

  return { voxels, bounds: { min, max }, cellSize };
}

function computeTcGradients(
  voxels: Map<string, Voxel>,
  resolution: number,
): void {
  for (const voxel of voxels.values()) {
    const { ix, iy, iz } = voxel;
    const gradient: [number, number, number] = [0, 0, 0];

    const dirs: [number, number, number][] = [
      [1, 0, 0], [-1, 0, 0],
      [0, 1, 0], [0, -1, 0],
      [0, 0, 1], [0, 0, -1],
    ];

    let geometricDirs = 0;
    let activeDirs = 0;
    for (const [dx, dy, dz] of dirs) {
      const nx = ix + dx, ny = iy + dy, nz = iz + dz;
      const inBounds = nx >= 0 && nx < resolution && ny >= 0 && ny < resolution && nz >= 0 && nz < resolution;

      if (!inBounds) continue;

      geometricDirs++;
      const neighborKey = `${nx}-${ny}-${nz}`;
      const neighbor = voxels.get(neighborKey);
      if (!neighbor) continue;

      activeDirs++;
      const tcDiff = neighbor.avgTc - voxel.avgTc;
      gradient[0] += tcDiff * dx;
      gradient[1] += tcDiff * dy;
      gradient[2] += tcDiff * dz;
    }

    if (geometricDirs > 0 && geometricDirs < 6) {
      const boundaryNorm = 6 / geometricDirs;
      gradient[0] *= boundaryNorm;
      gradient[1] *= boundaryNorm;
      gradient[2] *= boundaryNorm;
    }

    voxel.tcGradient = gradient;
    voxel.gradientMag = Math.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2);
  }
}

function extractElements(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  return matches ?? [];
}

export function detectDiscoveryZones(maxZones: number = 20): DiscoveryZone[] {
  const points = getEmbeddingDataset();
  if (points.length < 5) return [];

  const { voxels, bounds, cellSize } = buildVoxelGrid(points, GRID_RESOLUTION);
  if (voxels.size === 0) return [];

  const maxDensity = Math.max(...Array.from(voxels.values()).map(v => v.density));
  const maxTcGlobal = Math.max(...points.map(p => p.tc));

  const occupiedVoxels = Array.from(voxels.values());
  const allVoxelKeys = new Set<string>();
  for (let ix = 0; ix < GRID_RESOLUTION; ix++) {
    for (let iy = 0; iy < GRID_RESOLUTION; iy++) {
      for (let iz = 0; iz < GRID_RESOLUTION; iz++) {
        allVoxelKeys.add(`${ix}-${iy}-${iz}`);
      }
    }
  }

  const candidates: {
    center: [number, number, number];
    score: number;
    avgTc: number;
    maxTc: number;
    density: number;
    scFraction: number;
    gradient: [number, number, number];
    gradientMag: number;
    nearbyMaterials: EmbeddingPoint[];
    ix: number; iy: number; iz: number;
  }[] = [];

  const medianDensity = (() => {
    const densities = occupiedVoxels.map(v => v.density).sort((a, b) => a - b);
    return densities.length > 0 ? densities[Math.floor(densities.length / 2)] : 1;
  })();

  for (const voxel of occupiedVoxels) {
    const densityNorm = maxDensity > 0 ? voxel.density / maxDensity : 0;
    const sparsityRaw = 1 - densityNorm;
    const lowDensityScore = Math.min(0.7, sparsityRaw) * (voxel.density >= 2 ? 1.0 : 0.3);
    const tcProximity = maxTcGlobal > 0 ? voxel.avgTc / maxTcGlobal : 0;
    const gradientNorm = voxel.gradientMag / (maxTcGlobal + 1);

    const scFractionScore = voxel.scFraction;
    const densityConfidence = Math.max(0, 1 - Math.abs(voxel.density - medianDensity) / Math.max(medianDensity, 1));

    const elementSet = new Set<string>();
    for (const m of voxel.materials) {
      for (const el of extractElements(m.formula)) {
        elementSet.add(el);
      }
    }
    const uniqueElements = elementSet.size;
    const elementsPerMaterial = voxel.materials.length > 0 ? uniqueElements / voxel.materials.length : 0;
    const elementDiversityScore = Math.min(1.0, elementsPerMaterial / 3.0);

    const score = tcProximity * 0.35 + lowDensityScore * 0.10 + gradientNorm * 0.20 + scFractionScore * 0.20 + densityConfidence * 0.05 + elementDiversityScore * 0.10;

    candidates.push({
      center: voxel.center,
      score,
      avgTc: voxel.avgTc,
      maxTc: voxel.maxTc,
      density: voxel.density,
      scFraction: voxel.scFraction,
      gradient: voxel.tcGradient,
      gradientMag: voxel.gradientMag,
      nearbyMaterials: voxel.materials,
      ix: voxel.ix, iy: voxel.iy, iz: voxel.iz,
    });
  }

  for (const key of allVoxelKeys) {
    if (voxels.has(key)) continue;
    const [ixStr, iyStr, izStr] = key.split("-");
    const ix = parseInt(ixStr), iy = parseInt(iyStr), iz = parseInt(izStr);

    const center: [number, number, number] = [
      bounds.min[0] + (ix + 0.5) * cellSize[0],
      bounds.min[1] + (iy + 0.5) * cellSize[1],
      bounds.min[2] + (iz + 0.5) * cellSize[2],
    ];

    let bestGradientScore = 0;
    let bestGradientTc = 0;
    let bestGradientMaxTc = 0;
    let bestGradient: [number, number, number] = [0, 0, 0];
    const nearbyMats: EmbeddingPoint[] = [];

    for (const voxel of occupiedVoxels) {
      const dx = center[0] - voxel.center[0];
      const dy = center[1] - voxel.center[1];
      const dz = center[2] - voxel.center[2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (d < cellSize[0] * 2) {
        nearbyMats.push(...voxel.materials.slice(0, 3));
      }

      if (voxel.gradientMag > 0.01 && d > 0) {
        const dirToEmpty = [dx / d, dy / d, dz / d];
        const gradNorm = [
          voxel.tcGradient[0] / voxel.gradientMag,
          voxel.tcGradient[1] / voxel.gradientMag,
          voxel.tcGradient[2] / voxel.gradientMag,
        ];
        const alignment = dirToEmpty[0] * gradNorm[0] + dirToEmpty[1] * gradNorm[1] + dirToEmpty[2] * gradNorm[2];

        if (alignment > 0.3) {
          const distDecay = Math.exp(-d * d / (cellSize[0] * cellSize[0] * 4));
          const gradScore = alignment * distDecay * (voxel.avgTc / (maxTcGlobal + 1));
          if (gradScore > bestGradientScore) {
            bestGradientScore = gradScore;
            bestGradientTc = voxel.avgTc;
            bestGradientMaxTc = voxel.maxTc;
            bestGradient = [...voxel.tcGradient] as [number, number, number];
          }
        }
      }
    }

    if (nearbyMats.length === 0 && bestGradientScore < 0.01) continue;

    const nearbyBonus = nearbyMats.length > 0 ? 0.15 * Math.min(1.0, nearbyMats.length / 5) : 0;
    const score = bestGradientScore * 0.85 + nearbyBonus;

    candidates.push({
      center,
      score: Math.min(1, score),
      avgTc: bestGradientTc * 0.7,
      maxTc: bestGradientMaxTc,
      density: 0,
      scFraction: 0,
      gradient: bestGradient,
      gradientMag: Math.sqrt(bestGradient[0] ** 2 + bestGradient[1] ** 2 + bestGradient[2] ** 2),
      nearbyMaterials: nearbyMats,
      ix, iy, iz,
    });
  }

  candidates.sort((a, b) => b.score - a.score);
  const topCandidates = candidates.slice(0, maxZones);

  const zones: DiscoveryZone[] = topCandidates.map((c, idx) => {
    const elements = new Map<string, number>();
    const families = new Map<string, number>();

    for (const m of c.nearbyMaterials) {
      for (const el of extractElements(m.formula)) {
        elements.set(el, (elements.get(el) ?? 0) + 1);
      }
      families.set(m.family, (families.get(m.family) ?? 0) + 1);
    }

    const topElements = Array.from(elements.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(e => e[0]);

    const topFamilies = Array.from(families.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(f => f[0]);

    const priority: "high" | "medium" | "low" =
      c.score > 0.7 ? "high" : c.score > 0.4 ? "medium" : "low";

    return {
      id: `zone-${idx + 1}`,
      center3D: c.center.map(v => Math.round(v * 100) / 100) as [number, number, number],
      zoneScore: Math.round(c.score * 1000) / 1000,
      avgTc: Math.round(c.avgTc * 10) / 10,
      maxTc: Math.round(c.maxTc * 10) / 10,
      density: c.density,
      materialCount: c.nearbyMaterials.length,
      scFraction: Math.round(c.scFraction * 1000) / 1000,
      gradientDirection: c.gradient.map(v => Math.round(v * 100) / 100) as [number, number, number],
      gradientMagnitude: Math.round(c.gradientMag * 100) / 100,
      representativeFormulas: c.nearbyMaterials.slice(0, 5).map(m => m.formula),
      suggestedElements: topElements,
      suggestedFamilies: topFamilies,
      explorationPriority: priority,
    };
  });

  return zones;
}

export function getZoneMap(): ZoneMap {
  const points = getEmbeddingDataset();
  const zones = detectDiscoveryZones();

  const totalVoxels = GRID_RESOLUTION ** 3;
  let occupiedVoxelCount = 0;
  if (points.length > 0) {
    const { voxels } = buildVoxelGrid(points, GRID_RESOLUTION);
    occupiedVoxelCount = voxels.size;
  }

  const coveragePercent = Math.round((occupiedVoxelCount / totalVoxels) * 100);
  const topZones = zones.filter(z => z.explorationPriority === "high");

  const suggestions: string[] = [];
  if (topZones.length > 0) {
    const best = topZones[0];
    suggestions.push(`Explore zone ${best.id} near [${best.center3D.join(", ")}] — ${best.suggestedFamilies.join("/")} families with elements ${best.suggestedElements.slice(0, 4).join(", ")}`);
  }

  const lowDensityZones = zones.filter(z => z.density === 0 && z.zoneScore > 0.5);
  if (lowDensityZones.length > 0) {
    suggestions.push(`${lowDensityZones.length} unexplored regions near high-Tc clusters detected — prime targets for novel discovery`);
  }

  const highGradientZones = zones.filter(z => z.gradientMagnitude > 5);
  if (highGradientZones.length > 0) {
    suggestions.push(`${highGradientZones.length} zones with steep Tc gradients — potential for rapid optimization along gradient direction`);
  }

  if (suggestions.length === 0) {
    suggestions.push("Insufficient data for landscape guidance — continue broad exploration to populate the map");
  }

  return {
    zones,
    totalMaterials: points.length,
    gridResolution: GRID_RESOLUTION,
    coveragePercent,
    topZones,
    suggestions,
  };
}

export interface GradientNavTarget {
  zoneId: string;
  center3D: [number, number, number];
  gradientDirection: [number, number, number];
  gradientMagnitude: number;
  avgTc: number;
  maxTc: number;
  suggestedElements: string[];
}

export function getZoneSuggestions(): {
  targetZones: DiscoveryZone[];
  elementBias: Record<string, number>;
  familyBias: Record<string, number>;
  explorationVsExploitation: number;
  gradientNavTargets: GradientNavTarget[];
} {
  const zones = detectDiscoveryZones(10);
  const points = getEmbeddingDataset();

  const elementBias: Record<string, number> = {};
  const familyBias: Record<string, number> = {};

  for (const zone of zones) {
    const weight = zone.zoneScore;

    const elementCounts = new Map<string, number>();
    for (const formula of zone.representativeFormulas) {
      for (const el of extractElements(formula)) {
        elementCounts.set(el, (elementCounts.get(el) ?? 0) + 1);
      }
    }
    const totalElementOccurrences = Array.from(elementCounts.values()).reduce((s, c) => s + c, 0) || 1;

    for (const el of zone.suggestedElements) {
      const frequency = (elementCounts.get(el) ?? 0) / totalElementOccurrences;
      const stoichiometricWeight = weight * (0.3 + 0.7 * Math.min(1.0, frequency * zone.suggestedElements.length));
      elementBias[el] = (elementBias[el] ?? 0) + stoichiometricWeight;
    }
    for (const fam of zone.suggestedFamilies) {
      familyBias[fam] = (familyBias[fam] ?? 0) + weight;
    }
  }

  const maxElWeight = Math.max(...Object.values(elementBias), 1);
  for (const key of Object.keys(elementBias)) {
    elementBias[key] = Math.round((elementBias[key] / maxElWeight) * 1000) / 1000;
  }
  const maxFamWeight = Math.max(...Object.values(familyBias), 1);
  for (const key of Object.keys(familyBias)) {
    familyBias[key] = Math.round((familyBias[key] / maxFamWeight) * 1000) / 1000;
  }

  const highPriorityCount = zones.filter(z => z.explorationPriority === "high").length;
  let explorationVsExploitation = points.length < 20 ? 0.8 : Math.max(0.2, 0.6 - highPriorityCount * 0.05);

  try {
    const variance = getVarianceSummary();
    if (variance.highUncertaintyFraction > 0.4) {
      explorationVsExploitation = Math.max(explorationVsExploitation, 0.7);
    } else if (variance.highUncertaintyFraction > 0.2) {
      explorationVsExploitation = Math.max(explorationVsExploitation, 0.5);
    } else if (variance.meanVariance > 0.3) {
      explorationVsExploitation = Math.max(explorationVsExploitation, 0.45);
    }
  } catch {}

  const gradientNavTargets: GradientNavTarget[] = zones
    .filter(z => z.gradientMagnitude > 0.1 && z.explorationPriority !== "low")
    .sort((a, b) => b.gradientMagnitude * b.zoneScore - a.gradientMagnitude * a.zoneScore)
    .slice(0, 5)
    .map(z => ({
      zoneId: z.id,
      center3D: z.center3D,
      gradientDirection: z.gradientDirection,
      gradientMagnitude: z.gradientMagnitude,
      avgTc: z.avgTc,
      maxTc: z.maxTc,
      suggestedElements: z.suggestedElements,
    }));

  return {
    targetZones: zones.filter(z => z.explorationPriority !== "low"),
    elementBias,
    familyBias,
    explorationVsExploitation: Math.round(explorationVsExploitation * 1000) / 1000,
    gradientNavTargets,
  };
}
