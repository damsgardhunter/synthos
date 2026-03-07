import { getEmbeddingDataset, getEmbeddingPoint, getLandscapeStats } from "./discovery-landscape";
import { getZoneSuggestions, getZoneMap, detectDiscoveryZones, type DiscoveryZone } from "./zone-detector";
import { encodeGenome } from "../physics/materials-genome";

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
    const group = ELEMENT_TO_GROUP[element] ?? "unknown";
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
  }

  const highTcZones = suggestions.targetZones.filter(z => z.avgTc > 30);
  const targetMin = highTcZones.length > 0 ? Math.min(...highTcZones.map(z => z.avgTc)) * 0.8 : 30;
  const targetMax = highTcZones.length > 0 ? Math.max(...highTcZones.map(z => z.maxTc)) * 1.2 : 300;

  return {
    preferredMotifs: Array.from(preferredMotifs),
    preferredFamilies: Array.from(preferredFamilies),
    targetTcRange: { min: Math.round(targetMin), max: Math.round(targetMax) },
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
    const kNearest = Math.min(5, points.length);
    const dists = points.map(p => {
      let sum = 0;
      for (let i = 0; i < Math.min(genome.vector.length, p.genomeVector.length); i++) {
        const d = genome.vector[i] - p.genomeVector[i];
        sum += d * d;
      }
      return { point: p, dist: Math.sqrt(sum) };
    });
    dists.sort((a, b) => a.dist - b.dist);
    const neighbors = dists.slice(0, kNearest);

    position3D = [0, 0, 0];
    let totalWeight = 0;
    for (const n of neighbors) {
      const w = 1 / (n.dist + 1e-6);
      totalWeight += w;
      for (let d = 0; d < 3; d++) {
        position3D[d] += w * n.point.position3D[d];
      }
    }
    for (let d = 0; d < 3; d++) {
      position3D[d] /= totalWeight;
    }
  }

  const zones = detectDiscoveryZones(15);

  let nearestZoneDist = Infinity;
  let nearestZone: DiscoveryZone | null = null;

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

  const zoneRadius = 3.0;
  const inZone = nearestZoneDist < zoneRadius;
  const proximityFactor = Math.max(0, 1 - nearestZoneDist / (zoneRadius * 2));
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
