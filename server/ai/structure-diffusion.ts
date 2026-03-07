import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
} from "../learning/physics-engine";
import { extractFeatures } from "../learning/ml-predictor";
import { gbPredict } from "../learning/gradient-boost";

interface StructuralEmbedding {
  coordination: number;
  avgBondAngle: number;
  layerSpacing: number;
  cOverA: number;
  symmetryOrder: number;
  electronCountPerSite: number;
  dimensionality: number;
  cageFraction: number;
  interstitialFraction: number;
  connectivityIndex: number;
  anisotropy: number;
  voidFraction: number;
}

interface StructuralMotif {
  name: string;
  spaceGroup: string;
  crystalSystem: string;
  embedding: StructuralEmbedding;
  siteRoles: SiteRole[];
  latticeRatios: { a: number; b: number; c: number };
  scAffinity: number;
  tcRange: [number, number];
  pairingMechanism: string;
}

interface SiteRole {
  label: string;
  multiplicity: number;
  position: [number, number, number];
  role: "framework" | "interstitial" | "plane" | "chain" | "cage" | "spacer" | "reservoir";
  preferredCategories: string[];
}

interface StructureCandidate {
  motif: string;
  formula: string;
  siteAssignment: Record<string, string>;
  structuralScore: number;
  predictedTc: number;
  lambda: number;
  embedding: StructuralEmbedding;
}

export interface StructureDiffusionResult {
  candidates: StructureCandidate[];
  motifUsed: string;
  elementsTriedPerSite: Record<string, string[]>;
  bestFormula: string;
  bestTc: number;
  totalEvaluated: number;
}

interface StructureDiffusionStats {
  totalMotifGenerated: number;
  totalCandidatesEvaluated: number;
  bestTcAchieved: number;
  bestFormula: string;
  bestMotif: string;
  motifSuccessRates: Record<string, { tried: number; passed: number; avgTc: number }>;
  recentResults: { formula: string; tc: number; motif: string }[];
  learnedWeights: Record<string, number>;
}

const ELEMENT_CATEGORIES: Record<string, string[]> = {
  highCouplingTM: ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf"],
  lightPhonon: ["H", "B", "C", "N"],
  anion: ["O", "N", "F", "S", "Se", "Te"],
  rareEarth: ["La", "Y", "Ce", "Gd", "Nd", "Sc"],
  alkalineEarth: ["Ca", "Sr", "Ba", "Mg"],
  pnictogen: ["As", "P", "Sb", "Bi"],
  chalcogen: ["S", "Se", "Te"],
  magneticTM: ["Fe", "Co", "Ni", "Cu", "Mn"],
  pBlockMetal: ["Al", "Ga", "In", "Sn", "Ge", "Pb"],
  hydrogen: ["H"],
};

const SC_MOTIF_LIBRARY: StructuralMotif[] = [
  {
    name: "CuO2-plane",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 90, layerSpacing: 6.5, cOverA: 3.2,
      symmetryOrder: 16, electronCountPerSite: 9, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.1, connectivityIndex: 0.8,
      anisotropy: 0.85, voidFraction: 0.15,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0.35], role: "reservoir", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 2, position: [0, 0, 0], role: "plane", preferredCategories: ["magneticTM"] },
      { label: "X", multiplicity: 4, position: [0.5, 0, 0], role: "plane", preferredCategories: ["anion"] },
      { label: "X2", multiplicity: 2, position: [0, 0, 0.15], role: "spacer", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.2 },
    scAffinity: 0.92,
    tcRange: [30, 160],
    pairingMechanism: "d-wave spin fluctuation",
  },
  {
    name: "FeAs-layer",
    spaceGroup: "I4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 4, avgBondAngle: 109.5, layerSpacing: 6.0, cOverA: 3.0,
      symmetryOrder: 16, electronCountPerSite: 6, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.05, connectivityIndex: 0.75,
      anisotropy: 0.8, voidFraction: 0.12,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "reservoir", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 2, position: [0.25, 0.25, 0.25], role: "plane", preferredCategories: ["magneticTM", "highCouplingTM"] },
      { label: "X", multiplicity: 2, position: [0, 0.5, 0.15], role: "plane", preferredCategories: ["pnictogen", "chalcogen"] },
      { label: "X2", multiplicity: 2, position: [0, 0, 0.35], role: "spacer", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 3.0 },
    scAffinity: 0.85,
    tcRange: [20, 80],
    pairingMechanism: "s+- spin fluctuation",
  },
  {
    name: "clathrate-cage",
    spaceGroup: "Im-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 12, avgBondAngle: 60, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 1, dimensionality: 3,
      cageFraction: 0.75, interstitialFraction: 0, connectivityIndex: 0.95,
      anisotropy: 0, voidFraction: 0.3,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "cage", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "H", multiplicity: 8, position: [0.25, 0.25, 0.25], role: "framework", preferredCategories: ["hydrogen", "lightPhonon"] },
      { label: "H2", multiplicity: 6, position: [0, 0.5, 0.5], role: "framework", preferredCategories: ["hydrogen", "lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.95,
    tcRange: [100, 300],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "A15-chain",
    spaceGroup: "Pm-3n",
    crystalSystem: "cubic",
    embedding: {
      coordination: 12, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 5, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0.15, connectivityIndex: 0.9,
      anisotropy: 0.1, voidFraction: 0.1,
    },
    siteRoles: [
      { label: "A", multiplicity: 2, position: [0, 0, 0], role: "framework", preferredCategories: ["pBlockMetal", "alkalineEarth"] },
      { label: "B", multiplicity: 6, position: [0.25, 0, 0.5], role: "chain", preferredCategories: ["highCouplingTM"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.88,
    tcRange: [10, 40],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "kagome-flat",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 4, avgBondAngle: 120, layerSpacing: 5.5, cOverA: 1.8,
      symmetryOrder: 24, electronCountPerSite: 4, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.2, connectivityIndex: 0.7,
      anisotropy: 0.65, voidFraction: 0.25,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "spacer", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B", multiplicity: 3, position: [0.5, 0, 0.5], role: "plane", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 2, position: [0.333, 0.667, 0], role: "framework", preferredCategories: ["pBlockMetal", "pnictogen"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.8 },
    scAffinity: 0.72,
    tcRange: [2, 30],
    pairingMechanism: "flat-band enhanced",
  },
  {
    name: "hexagonal-layer",
    spaceGroup: "P6/mmm",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 6, avgBondAngle: 120, layerSpacing: 3.5, cOverA: 1.15,
      symmetryOrder: 24, electronCountPerSite: 3, dimensionality: 2,
      cageFraction: 0, interstitialFraction: 0.1, connectivityIndex: 0.85,
      anisotropy: 0.5, voidFraction: 0.18,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "plane", preferredCategories: ["alkalineEarth", "highCouplingTM"] },
      { label: "B", multiplicity: 2, position: [0.333, 0.667, 0.5], role: "plane", preferredCategories: ["lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.15 },
    scAffinity: 0.82,
    tcRange: [20, 50],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "perovskite-3D",
    spaceGroup: "Pm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 3, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 0.95,
      anisotropy: 0, voidFraction: 0.05,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "reservoir", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 1, position: [0.5, 0.5, 0.5], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 3, position: [0.5, 0.5, 0], role: "framework", preferredCategories: ["anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.75,
    tcRange: [5, 30],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "layered-hydride",
    spaceGroup: "P6_3/mmc",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 8, avgBondAngle: 109.5, layerSpacing: 4.0, cOverA: 2.5,
      symmetryOrder: 24, electronCountPerSite: 2, dimensionality: 2,
      cageFraction: 0.3, interstitialFraction: 0.5, connectivityIndex: 0.8,
      anisotropy: 0.6, voidFraction: 0.35,
    },
    siteRoles: [
      { label: "M", multiplicity: 2, position: [0.333, 0.667, 0.25], role: "plane", preferredCategories: ["highCouplingTM", "rareEarth"] },
      { label: "H1", multiplicity: 4, position: [0.333, 0.667, 0.1], role: "interstitial", preferredCategories: ["hydrogen"] },
      { label: "H2", multiplicity: 2, position: [0, 0, 0], role: "interstitial", preferredCategories: ["hydrogen", "lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 2.5 },
    scAffinity: 0.90,
    tcRange: [50, 250],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "NaCl-rocksalt",
    spaceGroup: "Fm-3m",
    crystalSystem: "cubic",
    embedding: {
      coordination: 6, avgBondAngle: 90, layerSpacing: 0, cOverA: 1.0,
      symmetryOrder: 48, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0, interstitialFraction: 0, connectivityIndex: 1.0,
      anisotropy: 0, voidFraction: 0.08,
    },
    siteRoles: [
      { label: "M", multiplicity: 4, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM"] },
      { label: "X", multiplicity: 4, position: [0.5, 0.5, 0.5], role: "framework", preferredCategories: ["anion", "lightPhonon"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1 },
    scAffinity: 0.70,
    tcRange: [5, 25],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "H-channel",
    spaceGroup: "P4/mmm",
    crystalSystem: "tetragonal",
    embedding: {
      coordination: 8, avgBondAngle: 90, layerSpacing: 0, cOverA: 0.8,
      symmetryOrder: 16, electronCountPerSite: 1.5, dimensionality: 3,
      cageFraction: 0.4, interstitialFraction: 0.6, connectivityIndex: 0.7,
      anisotropy: 0.3, voidFraction: 0.4,
    },
    siteRoles: [
      { label: "M", multiplicity: 1, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM"] },
      { label: "H1", multiplicity: 2, position: [0.5, 0, 0], role: "chain", preferredCategories: ["hydrogen"] },
      { label: "H2", multiplicity: 2, position: [0, 0.5, 0.5], role: "chain", preferredCategories: ["hydrogen"] },
      { label: "X", multiplicity: 1, position: [0.5, 0.5, 0], role: "interstitial", preferredCategories: ["lightPhonon", "anion"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 0.8 },
    scAffinity: 0.88,
    tcRange: [30, 200],
    pairingMechanism: "conventional phonon-mediated",
  },
  {
    name: "breathing-kagome",
    spaceGroup: "P-3m1",
    crystalSystem: "trigonal",
    embedding: {
      coordination: 4, avgBondAngle: 120, layerSpacing: 7.0, cOverA: 2.2,
      symmetryOrder: 12, electronCountPerSite: 5, dimensionality: 2,
      cageFraction: 0.15, interstitialFraction: 0.1, connectivityIndex: 0.6,
      anisotropy: 0.75, voidFraction: 0.3,
    },
    siteRoles: [
      { label: "A", multiplicity: 1, position: [0, 0, 0], role: "spacer", preferredCategories: ["alkalineEarth", "rareEarth"] },
      { label: "B", multiplicity: 3, position: [0.5, 0, 0.33], role: "plane", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "X", multiplicity: 3, position: [0.5, 0.5, 0.17], role: "framework", preferredCategories: ["pnictogen", "chalcogen", "pBlockMetal"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 2.2 },
    scAffinity: 0.68,
    tcRange: [1, 20],
    pairingMechanism: "flat-band enhanced",
  },
  {
    name: "Laves-MgZn2",
    spaceGroup: "P6_3/mmc",
    crystalSystem: "hexagonal",
    embedding: {
      coordination: 12, avgBondAngle: 60, layerSpacing: 0, cOverA: 1.63,
      symmetryOrder: 24, electronCountPerSite: 4, dimensionality: 3,
      cageFraction: 0.3, interstitialFraction: 0, connectivityIndex: 0.9,
      anisotropy: 0.2, voidFraction: 0.12,
    },
    siteRoles: [
      { label: "A", multiplicity: 4, position: [0.333, 0.667, 0.0625], role: "framework", preferredCategories: ["rareEarth", "alkalineEarth"] },
      { label: "B", multiplicity: 2, position: [0, 0, 0], role: "framework", preferredCategories: ["highCouplingTM", "magneticTM"] },
      { label: "B2", multiplicity: 6, position: [0.833, 0.667, 0.25], role: "framework", preferredCategories: ["highCouplingTM", "pBlockMetal"] },
    ],
    latticeRatios: { a: 1, b: 1, c: 1.63 },
    scAffinity: 0.65,
    tcRange: [1, 15],
    pairingMechanism: "conventional phonon-mediated",
  },
];

const stats: StructureDiffusionStats = {
  totalMotifGenerated: 0,
  totalCandidatesEvaluated: 0,
  bestTcAchieved: 0,
  bestFormula: "",
  bestMotif: "",
  motifSuccessRates: {},
  recentResults: [],
  learnedWeights: {},
};

for (const m of SC_MOTIF_LIBRARY) {
  stats.learnedWeights[m.name] = m.scAffinity;
}

function computeStructuralScore(embedding: StructuralEmbedding, targetTc: number): number {
  let score = 0;

  if (targetTc > 100) {
    score += embedding.cageFraction * 0.3;
    score += (embedding.coordination > 8 ? 0.2 : 0);
    score += embedding.interstitialFraction * 0.25;
  }

  if (targetTc > 30 && targetTc <= 100) {
    score += (embedding.dimensionality <= 2 ? 0.3 : 0.1);
    score += (embedding.anisotropy > 0.5 ? 0.2 : 0);
    score += (embedding.connectivityIndex > 0.7 ? 0.15 : 0);
  }

  if (targetTc <= 30) {
    score += (embedding.coordination >= 6 ? 0.2 : 0);
    score += (embedding.symmetryOrder > 20 ? 0.15 : 0);
    score += (1 - embedding.voidFraction) * 0.1;
  }

  score += embedding.connectivityIndex * 0.15;
  score += Math.min(1, embedding.symmetryOrder / 48) * 0.1;

  return Math.min(1, score);
}

function selectElementsForSite(site: SiteRole, triedCombos: Set<string>): string[] {
  const candidates: string[] = [];
  for (const cat of site.preferredCategories) {
    const pool = ELEMENT_CATEGORIES[cat];
    if (pool) candidates.push(...pool);
  }

  const unique = [...new Set(candidates)];
  const shuffled = unique.sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(4, shuffled.length));
}

function buildFormula(assignment: Record<string, string>, motif: StructuralMotif): string {
  const counts: Record<string, number> = {};
  for (const site of motif.siteRoles) {
    const el = assignment[site.label];
    if (!el) continue;
    counts[el] = (counts[el] || 0) + site.multiplicity;
  }

  const gcd = Object.values(counts).reduce((a, b) => {
    while (b) { [a, b] = [b, a % b]; }
    return a;
  });

  const sorted = Object.entries(counts).sort((a, b) => {
    const aIsM = ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "Fe", "Co", "Ni", "Cu", "Mn", "La", "Y", "Ce", "Ca", "Sr", "Ba", "Sc"].includes(a[0]);
    const bIsM = ["Nb", "V", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "Fe", "Co", "Ni", "Cu", "Mn", "La", "Y", "Ce", "Ca", "Sr", "Ba", "Sc"].includes(b[0]);
    if (aIsM !== bIsM) return aIsM ? -1 : 1;
    return a[0].localeCompare(b[0]);
  });

  return sorted.map(([el, c]) => {
    const reduced = c / gcd;
    return reduced === 1 ? el : `${el}${reduced}`;
  }).join("");
}

function evaluateCandidate(formula: string): { tc: number; lambda: number; gbTc: number; score: number } | null {
  try {
    const electronic = computeElectronicStructure(formula, null);
    const phonon = computePhononSpectrum(formula, electronic);
    const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);

    const omegaLogK = coupling.omegaLog * 1.44;
    const denom = coupling.lambda - coupling.muStar * (1 + 0.62 * coupling.lambda);
    let tc = 0;
    if (Math.abs(denom) > 1e-6 && coupling.lambda > 0.2) {
      const exponent = -1.04 * (1 + coupling.lambda) / denom;
      tc = (omegaLogK / 1.2) * Math.exp(exponent);
      if (!Number.isFinite(tc) || tc < 0) tc = 0;
    }
    if (electronic.metallicity < 0.4) {
      tc = tc * Math.max(0.02, electronic.metallicity);
    }
    tc = Math.min(400, tc);

    let gbTc = 0;
    let gbScore = 0;
    try {
      const features = extractFeatures(formula);
      if (features) {
        const gb = gbPredict(features);
        gbTc = gb.tcPredicted;
        gbScore = gb.score;
      }
    } catch {}

    return {
      tc: Math.max(tc, gbTc * 0.3),
      lambda: coupling.lambda,
      gbTc,
      score: gbScore,
    };
  } catch {
    return null;
  }
}

function perturbEmbedding(base: StructuralEmbedding, intensity: number): StructuralEmbedding {
  const perturb = (val: number, range: number) => {
    return Math.max(0, val + (Math.random() - 0.5) * range * intensity);
  };

  return {
    coordination: Math.round(perturb(base.coordination, 4)),
    avgBondAngle: perturb(base.avgBondAngle, 30),
    layerSpacing: perturb(base.layerSpacing, 3),
    cOverA: Math.max(0.5, perturb(base.cOverA, 1)),
    symmetryOrder: base.symmetryOrder,
    electronCountPerSite: perturb(base.electronCountPerSite, 3),
    dimensionality: base.dimensionality,
    cageFraction: Math.min(1, Math.max(0, perturb(base.cageFraction, 0.3))),
    interstitialFraction: Math.min(1, Math.max(0, perturb(base.interstitialFraction, 0.3))),
    connectivityIndex: Math.min(1, Math.max(0, perturb(base.connectivityIndex, 0.2))),
    anisotropy: Math.min(1, Math.max(0, perturb(base.anisotropy, 0.3))),
    voidFraction: Math.min(1, Math.max(0, perturb(base.voidFraction, 0.2))),
  };
}

function selectMotifs(targetTc: number, count: number): StructuralMotif[] {
  const scored = SC_MOTIF_LIBRARY.map(m => {
    const tcFit = m.tcRange[0] <= targetTc && targetTc <= m.tcRange[1] * 1.5 ? 1.0 :
      Math.max(0, 1.0 - Math.abs(targetTc - (m.tcRange[0] + m.tcRange[1]) / 2) / (m.tcRange[1] - m.tcRange[0] + 50));
    const learned = stats.learnedWeights[m.name] ?? m.scAffinity;
    return { motif: m, score: tcFit * 0.5 + learned * 0.3 + m.scAffinity * 0.2 };
  });

  scored.sort((a, b) => b.score - a.score);

  const selected: StructuralMotif[] = [];
  for (let i = 0; i < Math.min(count, scored.length); i++) {
    selected.push(scored[i].motif);
  }

  if (selected.length < count) {
    const remaining = SC_MOTIF_LIBRARY.filter(m => !selected.includes(m));
    for (let i = 0; selected.length < count && i < remaining.length; i++) {
      selected.push(remaining[i]);
    }
  }

  return selected;
}

export function runStructureFirstDesign(
  targetTc: number,
  motifsToTry: number = 4,
  elementsPerSite: number = 3,
): StructureDiffusionResult[] {
  const motifs = selectMotifs(targetTc, motifsToTry);
  const results: StructureDiffusionResult[] = [];

  for (const motif of motifs) {
    stats.totalMotifGenerated++;
    const candidates: StructureCandidate[] = [];
    const triedCombos = new Set<string>();
    const elementsTriedPerSite: Record<string, string[]> = {};

    const siteLists: Record<string, string[]> = {};
    for (const site of motif.siteRoles) {
      siteLists[site.label] = selectElementsForSite(site, triedCombos);
      elementsTriedPerSite[site.label] = siteLists[site.label];
    }

    const assignments: Record<string, string>[] = [];
    generateAssignments(motif.siteRoles, siteLists, 0, {}, assignments, elementsPerSite * 8);

    for (const assignment of assignments) {
      const formula = buildFormula(assignment, motif);
      if (!formula || formula.length < 2) continue;
      if (triedCombos.has(formula)) continue;
      triedCombos.add(formula);

      const result = evaluateCandidate(formula);
      if (!result) continue;

      stats.totalCandidatesEvaluated++;

      const embedding = perturbEmbedding(motif.embedding, 0.1);
      const structuralScore = computeStructuralScore(embedding, targetTc);

      candidates.push({
        motif: motif.name,
        formula,
        siteAssignment: assignment,
        structuralScore,
        predictedTc: Math.round(result.tc * 10) / 10,
        lambda: Math.round(result.lambda * 1000) / 1000,
        embedding,
      });
    }

    candidates.sort((a, b) => b.predictedTc - a.predictedTc);
    const best = candidates[0];

    if (!stats.motifSuccessRates[motif.name]) {
      stats.motifSuccessRates[motif.name] = { tried: 0, passed: 0, avgTc: 0 };
    }
    const mStats = stats.motifSuccessRates[motif.name];
    mStats.tried += candidates.length;
    mStats.passed += candidates.filter(c => c.predictedTc > 5).length;
    const totalTc = candidates.reduce((s, c) => s + c.predictedTc, 0);
    mStats.avgTc = candidates.length > 0 ? totalTc / candidates.length : 0;

    if (best) {
      if (best.predictedTc > stats.bestTcAchieved) {
        stats.bestTcAchieved = best.predictedTc;
        stats.bestFormula = best.formula;
        stats.bestMotif = motif.name;
      }

      stats.recentResults.push({
        formula: best.formula,
        tc: best.predictedTc,
        motif: motif.name,
      });
      if (stats.recentResults.length > 30) stats.recentResults.shift();

      const reward = best.predictedTc / Math.max(1, targetTc);
      const currentW = stats.learnedWeights[motif.name] ?? motif.scAffinity;
      stats.learnedWeights[motif.name] = currentW * 0.9 + reward * 0.1;
    }

    results.push({
      candidates: candidates.slice(0, 10),
      motifUsed: motif.name,
      elementsTriedPerSite,
      bestFormula: best?.formula ?? "",
      bestTc: best?.predictedTc ?? 0,
      totalEvaluated: candidates.length,
    });
  }

  return results;
}

function generateAssignments(
  sites: SiteRole[],
  elementLists: Record<string, string[]>,
  index: number,
  current: Record<string, string>,
  results: Record<string, string>[],
  maxResults: number,
): void {
  if (results.length >= maxResults) return;
  if (index >= sites.length) {
    results.push({ ...current });
    return;
  }

  const site = sites[index];
  const elements = elementLists[site.label] || [];

  for (const el of elements) {
    const alreadyUsedForDiffRole = Object.entries(current).some(([label, assignedEl]) => {
      if (assignedEl !== el) return false;
      const otherSite = sites.find(s => s.label === label);
      return otherSite && otherSite.role !== site.role;
    });
    if (alreadyUsedForDiffRole && Math.random() > 0.3) continue;

    current[site.label] = el;
    generateAssignments(sites, elementLists, index + 1, current, results, maxResults);
  }
}

export function runStructureDiffusionCycle(
  targetTc: number = 200,
  motifCount: number = 3,
  elementsPerSite: number = 3,
): { formulas: string[]; bestFormula: string; bestTc: number; motifsUsed: string[] } {
  const results = runStructureFirstDesign(targetTc, motifCount, elementsPerSite);

  const allFormulas: string[] = [];
  let bestTc = 0;
  let bestFormula = "";
  const motifsUsed: string[] = [];

  for (const r of results) {
    motifsUsed.push(r.motifUsed);
    for (const c of r.candidates) {
      allFormulas.push(c.formula);
      if (c.predictedTc > bestTc) {
        bestTc = c.predictedTc;
        bestFormula = c.formula;
      }
    }
  }

  return { formulas: allFormulas, bestFormula, bestTc, motifsUsed };
}

export function getStructureDiffusionStats(): StructureDiffusionStats {
  return { ...stats, motifSuccessRates: { ...stats.motifSuccessRates } };
}

export function getMotifLibrarySummary(): { name: string; spaceGroup: string; tcRange: [number, number]; scAffinity: number; pairingMechanism: string; learnedWeight: number }[] {
  return SC_MOTIF_LIBRARY.map(m => ({
    name: m.name,
    spaceGroup: m.spaceGroup,
    tcRange: m.tcRange,
    scAffinity: m.scAffinity,
    pairingMechanism: m.pairingMechanism,
    learnedWeight: stats.learnedWeights[m.name] ?? m.scAffinity,
  }));
}
