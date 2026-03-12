import { getEmbeddingDataset, getEmbeddingPoint, getLandscapeStats } from "./discovery-landscape";
import { getZoneSuggestions, getZoneMap, detectDiscoveryZones, type DiscoveryZone } from "./zone-detector";
import { encodeGenome } from "../physics/materials-genome";
import { getElementData } from "../learning/elemental-data";

export interface LandscapeRLBias {
  elementGroupWeights: Record<string, number>;
  familyWeights: Record<string, number>;
  explorationRatio: number;
  activeZoneCount: number;
}

export interface LandscapeInverseSeed {
  formula: string;
  genomeVector: number[];
  tc: number;
  zoneId: string;
  zoneScore: number;
}

export interface LandscapeDiffusionGuidance {
  preferredMotifs: string[];
  preferredFamilies: string[];
  targetTcRange: { min: number; max: number };
  structuralBias: Record<string, number>;
}

export interface ZoneBonus {
  formula: string;
  inZone: boolean;
  zoneId: string | null;
  bonus: number;
  nearestZoneDistance: number;
  nearestZoneScore: number;
}

function resolveElementGroup(symbol: string): string {
  const mapped = ELEMENT_TO_GROUP[symbol];
  if (mapped) return mapped;

  const data = getElementData(symbol);
  if (!data) return "unknown";

  const z = data.atomicNumber;
  if (z >= 89 && z <= 103) return "actinide";
  if (z >= 57 && z <= 71) return "lanthanide";
  if ((z >= 21 && z <= 30)) return "3d-transition";
  if ((z >= 39 && z <= 48)) return "4d-transition";
  if ((z >= 72 && z <= 80)) return "5d-transition";
  if ((z >= 104 && z <= 112)) return "6d-transition";
  if (z === 1 || (z >= 6 && z <= 9) || (z >= 15 && z <= 17) || z === 34 || z === 35 || z === 53) return "nonmetal";
  if (z === 3 || z === 11 || z === 19 || z === 37 || z === 55 || z === 87) return "alkali";
  if (z === 4 || z === 12 || z === 20 || z === 38 || z === 56 || z === 88) return "alkaline-earth";
  if (z === 5 || z === 14 || z === 32 || z === 33 || z === 51 || z === 52) return "metalloid";
  if (z === 13 || z === 31 || z === 49 || z === 50 || z === 81 || z === 82 || z === 83 || z === 84) return "post-transition";
  if (z === 2 || z === 10 || z === 18 || z === 36 || z === 54 || z === 86 || z === 118) return "noble-gas";

  return "unknown";
}

interface EmbeddingPointLike {
  position3D: [number, number, number];
  genomeVector: number[];
}

function estimateFuzzyPosition(
  queryVector: number[],
  existingPoints: EmbeddingPointLike[],
  kNearest: number = 10,
): [number, number, number] {
  const dists = existingPoints.map((p, idx) => {
    let sum = 0;
    const len = Math.min(queryVector.length, p.genomeVector.length);
    for (let i = 0; i < len; i++) {
      const d = queryVector[i] - p.genomeVector[i];
      sum += d * d;
    }
    return { idx, dist: Math.sqrt(sum) };
  });
  dists.sort((a, b) => a.dist - b.dist);
  const neighbors = dists.slice(0, Math.min(kNearest, dists.length));
  if (neighbors.length === 0) return [0, 0, 0];

  const rho = neighbors[0].dist;
  let sigmaLo = 1e-10;
  let sigmaHi = Math.max(100, neighbors[neighbors.length - 1]?.dist ?? 100);
  const logTarget = Math.log2(Math.min(kNearest, 5));
  for (let iter = 0; iter < 64; iter++) {
    let sumW = 0;
    for (const n of neighbors) {
      sumW += Math.exp(-Math.max(0, n.dist - rho) / (sigmaHi + 1e-10));
    }
    if (Math.log2(Math.max(sumW, 1e-30)) >= logTarget) break;
    sigmaHi *= 2;
    if (sigmaHi > 1e8) break;
  }
  for (let iter = 0; iter < 64; iter++) {
    const sigmaMid = (sigmaLo + sigmaHi) / 2;
    let sumW = 0;
    for (const n of neighbors) {
      sumW += Math.exp(-Math.max(0, n.dist - rho) / (sigmaMid + 1e-10));
    }
    const logSumW = Math.log2(Math.max(sumW, 1e-30));
    if (Math.abs(logSumW - logTarget) < 1e-5) break;
    if (logSumW > logTarget) sigmaHi = sigmaMid;
    else sigmaLo = sigmaMid;
    if (Math.abs(sigmaHi - sigmaLo) < 1e-10) break;
  }
  const localSigma = (sigmaLo + sigmaHi) / 2;

  const position: [number, number, number] = [0, 0, 0];
  let totalW = 0;
  for (const n of neighbors) {
    const w = Math.exp(-Math.max(0, n.dist - rho) / (localSigma + 1e-10));
    totalW += w;
    for (let d = 0; d < 3; d++) {
      position[d] += w * existingPoints[n.idx].position3D[d];
    }
  }
  if (totalW < 1e-10) {
    return [...existingPoints[neighbors[0].idx].position3D] as [number, number, number];
  }
  for (let d = 0; d < 3; d++) position[d] /= totalW;
  return position;
}

const ELEMENT_TO_GROUP: Record<string, string> = {
  Sc: "3d-transition", Ti: "3d-transition", V: "3d-transition", Cr: "3d-transition",
  Mn: "3d-transition", Fe: "3d-transition", Co: "3d-transition", Ni: "3d-transition",
  Cu: "3d-transition", Zn: "3d-transition",
  Y: "4d-transition", Zr: "4d-transition", Nb: "4d-transition", Mo: "4d-transition",
  Ru: "4d-transition", Rh: "4d-transition", Pd: "4d-transition",
  Hf: "5d-transition", Ta: "5d-transition", W: "5d-transition", Re: "5d-transition",
  Os: "5d-transition", Ir: "5d-transition", Pt: "5d-transition",
  La: "lanthanide", Ce: "lanthanide", Pr: "lanthanide", Nd: "lanthanide",
  Sm: "lanthanide", Eu: "lanthanide", Gd: "lanthanide", Tb: "lanthanide",
  Dy: "lanthanide", Ho: "lanthanide", Er: "lanthanide", Yb: "lanthanide", Lu: "lanthanide",
  B: "metalloid", Si: "metalloid", Ge: "metalloid", As: "metalloid", Sb: "metalloid", Te: "metalloid",
  H: "nonmetal", C: "nonmetal", N: "nonmetal", O: "nonmetal", F: "nonmetal",
  P: "nonmetal", S: "nonmetal", Se: "nonmetal",
  Li: "alkali", Na: "alkali", K: "alkali", Rb: "alkali", Cs: "alkali",
  Be: "alkaline-earth", Mg: "alkaline-earth", Ca: "alkaline-earth", Sr: "alkaline-earth", Ba: "alkaline-earth",
  Al: "post-transition", Ga: "post-transition", In: "post-transition", Sn: "post-transition",
  Tl: "post-transition", Pb: "post-transition", Bi: "post-transition",
};

const FAMILY_TO_MOTIFS: Record<string, string[]> = {
  Hydrides: ["clathrate-cage", "H-channel", "layered-hydride"],
  Cuprates: ["CuO2-plane", "perovskite-3D"],
  Carbides: ["NaCl-rocksalt", "A15-chain"],
  Borides: ["AlB2", "hexagonal-layer"],
  Nitrides: ["NaCl-rocksalt", "perovskite-3D"],
  Pnictides: ["FeAs-layer", "ThCr2Si2"],
  Intermetallics: ["A15-chain", "Laves-MgZn2"],
  Kagome: ["kagome-flat", "breathing-kagome"],
};

export function getLandscapeRLBias(): LandscapeRLBias {
  const suggestions = getZoneSuggestions();
  const elementGroupWeights: Record<string, number> = {};

  for (const [element, weight] of Object.entries(suggestions.elementBias)) {
    const group = resolveElementGroup(element);
    elementGroupWeights[group] = Math.max(elementGroupWeights[group] ?? 0, weight);
  }

  return {
    elementGroupWeights,
    familyWeights: suggestions.familyBias,
    explorationRatio: suggestions.explorationVsExploitation,
    activeZoneCount: suggestions.targetZones.length,
  };
}

export function getLandscapeInverseSeeds(maxSeeds: number = 10): LandscapeInverseSeed[] {
  const zones = detectDiscoveryZones(maxSeeds);
  const seeds: LandscapeInverseSeed[] = [];

  for (const zone of zones) {
    if (zone.explorationPriority === "low") continue;

    for (const formula of zone.representativeFormulas.slice(0, 2)) {
      const point = getEmbeddingPoint(formula);
      if (point) {
        seeds.push({
          formula: point.formula,
          genomeVector: point.genomeVector,
          tc: point.tc,
          zoneId: zone.id,
          zoneScore: zone.zoneScore,
        });
      }
    }
  }

  seeds.sort((a, b) => b.zoneScore - a.zoneScore);
  return seeds.slice(0, maxSeeds);
}

const HYBRID_MOTIFS: Record<string, string[]> = {
  "Hydrides+Cuprates": ["layered-hydride-CuO2", "perovskite-H-channel"],
  "Hydrides+Intermetallics": ["clathrate-A15", "H-channel-Laves"],
  "Hydrides+Borides": ["clathrate-AlB2", "H-layer-hexagonal"],
  "Cuprates+Pnictides": ["CuO2-FeAs-hetero", "ThCr2Si2-perovskite"],
  "Cuprates+Nitrides": ["CuO2-rocksalt-hetero", "perovskite-nitride"],
  "Pnictides+Kagome": ["FeAs-kagome-flat", "ThCr2Si2-breathing"],
  "Borides+Carbides": ["AlB2-NaCl-hetero", "hexagonal-rocksalt"],
  "Intermetallics+Kagome": ["A15-kagome", "Laves-breathing-kagome"],
};

function getHybridMotifKey(familyA: string, familyB: string): string | null {
  const key1 = `${familyA}+${familyB}`;
  const key2 = `${familyB}+${familyA}`;
  if (HYBRID_MOTIFS[key1]) return key1;
  if (HYBRID_MOTIFS[key2]) return key2;
  return null;
}

const TC_PHYSICAL_CEILING: Record<string, number> = {
  Hydrides: 300,
  Cuprates: 180,
  Pnictides: 80,
  Borides: 50,
  Carbides: 40,
  Nitrides: 35,
  Intermetallics: 30,
  Kagome: 10,
};
const TC_ABSOLUTE_CEILING = 350;

export function getLandscapeDiffusionGuidance(): LandscapeDiffusionGuidance {
  const suggestions = getZoneSuggestions();
  const preferredMotifs: Set<string> = new Set();
  const preferredFamilies: Set<string> = new Set();
  const structuralBias: Record<string, number> = {};

  for (const zone of suggestions.targetZones) {
    for (const family of zone.suggestedFamilies) {
      preferredFamilies.add(family);
      const motifs = FAMILY_TO_MOTIFS[family];
      if (motifs) {
        for (const m of motifs) {
          preferredMotifs.add(m);
          structuralBias[m] = Math.max(structuralBias[m] ?? 0, zone.zoneScore);
        }
      }
    }

    if (zone.suggestedFamilies.length >= 2) {
      for (let i = 0; i < zone.suggestedFamilies.length; i++) {
        for (let j = i + 1; j < zone.suggestedFamilies.length; j++) {
          const hybridKey = getHybridMotifKey(zone.suggestedFamilies[i], zone.suggestedFamilies[j]);
          if (hybridKey) {
            for (const hm of HYBRID_MOTIFS[hybridKey]) {
              preferredMotifs.add(hm);
              structuralBias[hm] = Math.max(structuralBias[hm] ?? 0, zone.zoneScore * 1.1);
            }
          }
        }
      }
    }
  }

  const zones = suggestions.targetZones;
  for (let zi = 0; zi < zones.length; zi++) {
    for (let zj = zi + 1; zj < zones.length; zj++) {
      const dist = Math.sqrt(
        (zones[zi].center3D[0] - zones[zj].center3D[0]) ** 2 +
        (zones[zi].center3D[1] - zones[zj].center3D[1]) ** 2 +
        (zones[zi].center3D[2] - zones[zj].center3D[2]) ** 2,
      );
      if (dist < 4.0) {
        for (const famI of zones[zi].suggestedFamilies) {
          for (const famJ of zones[zj].suggestedFamilies) {
            if (famI === famJ) continue;
            const hybridKey = getHybridMotifKey(famI, famJ);
            if (hybridKey) {
              const overlapScore = Math.min(zones[zi].zoneScore, zones[zj].zoneScore);
              for (const hm of HYBRID_MOTIFS[hybridKey]) {
                preferredMotifs.add(hm);
                structuralBias[hm] = Math.max(structuralBias[hm] ?? 0, overlapScore * 1.2);
              }
            } else {
              const motifsI = FAMILY_TO_MOTIFS[famI] ?? [];
              const motifsJ = FAMILY_TO_MOTIFS[famJ] ?? [];
              if (motifsI.length > 0 && motifsJ.length > 0) {
                const syntheticMotif = `${motifsI[0]}+${motifsJ[0]}-hybrid`;
                preferredMotifs.add(syntheticMotif);
                const overlapScore = Math.min(zones[zi].zoneScore, zones[zj].zoneScore);
                structuralBias[syntheticMotif] = Math.max(structuralBias[syntheticMotif] ?? 0, overlapScore * 0.8);
              }
            }
          }
        }
      }
    }
  }

  const highTcZones = suggestions.targetZones.filter(z => z.avgTc > 30);
  const rawMin = highTcZones.length > 0 ? Math.min(...highTcZones.map(z => z.avgTc)) * 0.8 : 30;
  const rawMax = highTcZones.length > 0 ? Math.max(...highTcZones.map(z => z.maxTc)) * 1.2 : 300;

  const allZoneFamilies = Array.from(preferredFamilies);
  let familyCeiling = -1;
  for (const fam of allZoneFamilies) {
    const ceil = TC_PHYSICAL_CEILING[fam];
    if (ceil !== undefined) {
      familyCeiling = Math.max(familyCeiling, ceil);
    }
  }
  if (familyCeiling < 0) familyCeiling = TC_ABSOLUTE_CEILING;

  const targetMin = Math.max(1, Math.round(rawMin));
  const targetMax = Math.round(Math.min(rawMax, familyCeiling));

  return {
    preferredMotifs: Array.from(preferredMotifs),
    preferredFamilies: Array.from(preferredFamilies),
    targetTcRange: { min: targetMin, max: Math.max(targetMin, targetMax) },
    structuralBias,
  };
}

export function getZoneBonus(formula: string): ZoneBonus {
  const points = getEmbeddingDataset();
  if (points.length < 5) {
    return { formula, inZone: false, zoneId: null, bonus: 0, nearestZoneDistance: Infinity, nearestZoneScore: 0 };
  }

  let genome: { vector: number[] };
  try {
    genome = encodeGenome(formula);
    if (!genome || !genome.vector || genome.vector.length === 0) {
      return { formula, inZone: false, zoneId: null, bonus: 0, nearestZoneDistance: Infinity, nearestZoneScore: 0 };
    }
  } catch {
    return { formula, inZone: false, zoneId: null, bonus: 0, nearestZoneDistance: Infinity, nearestZoneScore: 0 };
  }

  const existingPoint = getEmbeddingPoint(formula);
  let position3D: [number, number, number];

  if (existingPoint) {
    position3D = existingPoint.position3D;
  } else {
    position3D = estimateFuzzyPosition(genome.vector, points, Math.min(10, points.length));
  }

  const zones = detectDiscoveryZones(15);

  let nearestZoneDist = Infinity;
  let nearestZone: DiscoveryZone | null = null;
  let nearestZoneRadius = 3.0;

  for (const zone of zones) {
    const dist = Math.sqrt(
      (position3D[0] - zone.center3D[0]) ** 2 +
      (position3D[1] - zone.center3D[1]) ** 2 +
      (position3D[2] - zone.center3D[2]) ** 2,
    );
    if (dist < nearestZoneDist) {
      nearestZoneDist = dist;
      nearestZone = zone;
    }
  }

  if (!nearestZone) {
    return { formula, inZone: false, zoneId: null, bonus: 0, nearestZoneDistance: Infinity, nearestZoneScore: 0 };
  }

  const repFormulas = new Set(nearestZone.representativeFormulas);
  const zoneMemberDists: number[] = [];
  for (const p of points) {
    if (repFormulas.has(p.formula) || p.family === (nearestZone.suggestedFamilies?.[0] ?? "")) {
      const d = Math.sqrt(
        (p.position3D[0] - nearestZone.center3D[0]) ** 2 +
        (p.position3D[1] - nearestZone.center3D[1]) ** 2 +
        (p.position3D[2] - nearestZone.center3D[2]) ** 2,
      );
      zoneMemberDists.push(d);
    }
  }

  if (zoneMemberDists.length < 5) {
    const allDists: { d: number }[] = [];
    for (const p of points) {
      const d = Math.sqrt(
        (p.position3D[0] - nearestZone.center3D[0]) ** 2 +
        (p.position3D[1] - nearestZone.center3D[1]) ** 2 +
        (p.position3D[2] - nearestZone.center3D[2]) ** 2,
      );
      allDists.push({ d });
    }
    allDists.sort((a, b) => a.d - b.d);
    zoneMemberDists.length = 0;
    const kLocal = Math.min(Math.max(10, nearestZone.materialCount), allDists.length);
    for (let i = 0; i < kLocal; i++) {
      zoneMemberDists.push(allDists[i].d);
    }
  }

  if (zoneMemberDists.length >= 3) {
    let meanDist = 0;
    for (const d of zoneMemberDists) meanDist += d;
    meanDist /= zoneMemberDists.length;
    let sumSqDev = 0;
    for (const d of zoneMemberDists) {
      const dev = d - meanDist;
      sumSqDev += dev * dev;
    }
    const stddev = Math.sqrt(sumSqDev / zoneMemberDists.length);
    nearestZoneRadius = Math.max(0.5, Math.min(8.0, meanDist + stddev * 1.5));
  } else {
    nearestZoneRadius = 3.0;
  }

  const zoneRadius = nearestZoneRadius;
  const inZone = nearestZoneDist < zoneRadius;
  const proximityFactor = Math.exp(-0.5 * (nearestZoneDist * nearestZoneDist) / (zoneRadius * zoneRadius));
  const bonus = Math.round(proximityFactor * nearestZone.zoneScore * 0.15 * 1000) / 1000;

  return {
    formula,
    inZone,
    zoneId: inZone ? nearestZone.id : null,
    bonus,
    nearestZoneDistance: Math.round(nearestZoneDist * 100) / 100,
    nearestZoneScore: nearestZone.zoneScore,
  };
}

export function getFullLandscapeGuidance(): {
  rlBias: LandscapeRLBias;
  inverseSeeds: LandscapeInverseSeed[];
  diffusionGuidance: LandscapeDiffusionGuidance;
  landscapeStats: ReturnType<typeof getLandscapeStats>;
  zoneMap: ReturnType<typeof getZoneMap>;
} {
  return {
    rlBias: getLandscapeRLBias(),
    inverseSeeds: getLandscapeInverseSeeds(),
    diffusionGuidance: getLandscapeDiffusionGuidance(),
    landscapeStats: getLandscapeStats(),
    zoneMap: getZoneMap(),
  };
}
