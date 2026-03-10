import { getElementData } from "../learning/elemental-data";

export interface BondInfo {
  atom1Index: number;
  atom2Index: number;
  element1: string;
  element2: string;
  length: number;
}

export interface CoordinationInfo {
  atomIndex: number;
  element: string;
  coordination: number;
  isDefect: boolean;
}

export interface LocalStrainInfo {
  atomIndex: number;
  element: string;
  strain: number;
  displacement: number;
  isDefect: boolean;
}

export interface DisorderMetrics {
  bondLengths: number[];
  bondMean: number;
  bondVariance: number;
  bondStdDev: number;
  bondMin: number;
  bondMax: number;
  totalBonds: number;
  coordinationNumbers: number[];
  coordinationMean: number;
  coordinationVariance: number;
  coordinationMin: number;
  coordinationMax: number;
  idealCoordination: number;
  coordinationDeficit: number;
  localStrains: number[];
  localStrainMean: number;
  localStrainMax: number;
  defectNeighborStrain: number;
  disorderScore: number;
  disorderClass: "perfect" | "mild" | "moderate" | "strong" | "amorphous";
  vacancyFraction: number;
  substitutionFraction: number;
  interstitialFraction: number;
  siteMixingFraction: number;
  amorphousFraction: number;
}

export interface DisorderMetricsStats {
  totalAnalyzed: number;
  avgDisorderScore: number;
  maxDisorderScore: number;
  byClass: Record<string, number>;
  avgBondVariance: number;
  avgCoordinationVariance: number;
  avgLocalStrain: number;
  recentAnalyses: Array<{
    formula: string;
    disorderScore: number;
    disorderClass: string;
    bondVariance: number;
    coordinationVariance: number;
    totalAtoms: number;
  }>;
  topDisordered: Array<{
    formula: string;
    disorderScore: number;
    disorderClass: string;
    bondVariance: number;
    coordinationVariance: number;
  }>;
}

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11,
  P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06, K: 2.03, Ca: 1.76, Sc: 1.70,
  Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24,
  Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20,
  Kr: 1.16, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54,
  Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39,
  Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04,
  Nd: 2.01, Gd: 1.96, Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44,
  Ir: 1.41, Pt: 1.36, Au: 1.36, Pb: 1.46, Bi: 1.48, Th: 2.06, U: 1.96,
};

const IDEAL_COORDINATION: Record<string, number> = {
  H: 1, Li: 4, Be: 4, B: 3, C: 4, N: 3, O: 2, F: 1,
  Na: 6, Mg: 6, Al: 6, Si: 4, P: 4, S: 2, Cl: 1,
  K: 8, Ca: 8, Sc: 6, Ti: 6, V: 6, Cr: 6, Mn: 6,
  Fe: 6, Co: 6, Ni: 6, Cu: 4, Zn: 4, Ga: 4, Ge: 4,
  As: 3, Se: 2, Br: 1, Sr: 8, Y: 8, Zr: 8, Nb: 6,
  Mo: 6, Ru: 6, Pd: 4, Ag: 4, In: 4, Sn: 4, Sb: 3,
  Te: 2, Ba: 12, La: 12, Ce: 12, Nd: 12, Bi: 3, Pb: 4,
  Hf: 8, Ta: 6, W: 6,
};

function getCovalentRadius(element: string): number {
  if (COVALENT_RADII[element]) return COVALENT_RADII[element];
  const data = getElementData(element);
  return data ? (data.atomicRadius ?? 150) / 100 : 1.5;
}

function getIdealCoordination(element: string): number {
  return IDEAL_COORDINATION[element] ?? 6;
}

function distance3D(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number }
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function computeBonds(
  atoms: Array<{ element: string; x: number; y: number; z: number; isDefect?: boolean }>,
  bondCutoffFactor: number = 1.3
): BondInfo[] {
  const bonds: BondInfo[] = [];
  const n = atoms.length;
  const maxAtoms = Math.min(n, 300);

  for (let i = 0; i < maxAtoms; i++) {
    const ri = getCovalentRadius(atoms[i].element);
    for (let j = i + 1; j < maxAtoms; j++) {
      const rj = getCovalentRadius(atoms[j].element);
      const cutoff = (ri + rj) * bondCutoffFactor;
      const dist = distance3D(atoms[i], atoms[j]);
      if (dist > 0.3 && dist <= cutoff) {
        bonds.push({
          atom1Index: i,
          atom2Index: j,
          element1: atoms[i].element,
          element2: atoms[j].element,
          length: dist,
        });
      }
    }
  }

  return bonds;
}

export function computeCoordination(
  atoms: Array<{ element: string; x: number; y: number; z: number; isDefect?: boolean }>,
  bonds: BondInfo[]
): CoordinationInfo[] {
  const coordCount: Record<number, number> = {};
  for (const bond of bonds) {
    coordCount[bond.atom1Index] = (coordCount[bond.atom1Index] || 0) + 1;
    coordCount[bond.atom2Index] = (coordCount[bond.atom2Index] || 0) + 1;
  }

  const maxAtoms = Math.min(atoms.length, 300);
  return Array.from({ length: maxAtoms }, (_, i) => ({
    atomIndex: i,
    element: atoms[i].element,
    coordination: coordCount[i] || 0,
    isDefect: atoms[i].isDefect ?? false,
  }));
}

export function computeLocalStrain(
  atoms: Array<{ element: string; x: number; y: number; z: number; isDefect?: boolean }>,
  bonds: BondInfo[]
): LocalStrainInfo[] {
  const bondsByAtom: Record<number, number[]> = {};
  for (const bond of bonds) {
    if (!bondsByAtom[bond.atom1Index]) bondsByAtom[bond.atom1Index] = [];
    if (!bondsByAtom[bond.atom2Index]) bondsByAtom[bond.atom2Index] = [];
    bondsByAtom[bond.atom1Index].push(bond.length);
    bondsByAtom[bond.atom2Index].push(bond.length);
  }

  const maxAtoms = Math.min(atoms.length, 300);
  return Array.from({ length: maxAtoms }, (_, i) => {
    const atomBonds = bondsByAtom[i] || [];
    const el = atoms[i].element;
    const idealBondLength = getCovalentRadius(el) * 2;

    let strain = 0;
    let displacement = 0;
    if (atomBonds.length > 0) {
      const meanBond = atomBonds.reduce((s, b) => s + b, 0) / atomBonds.length;
      strain = Math.abs(meanBond - idealBondLength) / idealBondLength;
      const deviations = atomBonds.map(b => Math.abs(b - meanBond));
      displacement = deviations.reduce((s, d) => s + d, 0) / deviations.length;
    }

    return {
      atomIndex: i,
      element: el,
      strain,
      displacement,
      isDefect: atoms[i].isDefect ?? false,
    };
  });
}

function variance(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

export function computeDisorderMetrics(
  atoms: Array<{ element: string; x: number; y: number; z: number; isDefect?: boolean; defectType?: string }>,
  bondCutoffFactor: number = 1.3
): DisorderMetrics {
  const bonds = computeBonds(atoms, bondCutoffFactor);
  const coordination = computeCoordination(atoms, bonds);
  const localStrain = computeLocalStrain(atoms, bonds);

  const bondLengths = bonds.map(b => b.length);
  const bondVar = variance(bondLengths);
  const bondMean = mean(bondLengths);
  const bondStdDev = Math.sqrt(bondVar);

  const coordNumbers = coordination.map(c => c.coordination);
  const coordVar = variance(coordNumbers);
  const coordMean = mean(coordNumbers);

  const elements = [...new Set(atoms.map(a => a.element))];
  const idealCoord = mean(elements.map(e => getIdealCoordination(e)));
  const coordDeficit = Math.abs(coordMean - idealCoord) / Math.max(idealCoord, 1);

  const strains = localStrain.map(s => s.strain);
  const strainMean = mean(strains);
  const strainMax = strains.length > 0 ? Math.max(...strains) : 0;

  const defectStrains = localStrain.filter(s => s.isDefect).map(s => s.strain);
  const defectNeighborStrain = mean(defectStrains);

  const totalAtoms = atoms.length;
  const vacancyFraction = atoms.filter(a => a.defectType === "vacancy").length / Math.max(totalAtoms, 1);
  const subFraction = atoms.filter(a => a.defectType === "substitution").length / Math.max(totalAtoms, 1);
  const intFraction = atoms.filter(a => a.defectType === "interstitial").length / Math.max(totalAtoms, 1);
  const mixFraction = atoms.filter(a => a.defectType === "site-mixing").length / Math.max(totalAtoms, 1);
  const amorphFraction = atoms.filter(a => a.defectType === "amorphous").length / Math.max(totalAtoms, 1);

  const defectFraction = atoms.filter(a => a.isDefect).length / Math.max(totalAtoms, 1);

  const normalizedBondVar = Math.min(1.0, bondVar / Math.max(bondMean * bondMean * 0.1, 0.01));
  const normalizedCoordVar = Math.min(1.0, coordVar / Math.max(idealCoord * idealCoord * 0.2, 0.5));
  const normalizedStrain = Math.min(1.0, strainMean * 3.0);

  const disorderScore = Math.min(1.0,
    defectFraction * 0.3 +
    normalizedBondVar * 0.25 +
    normalizedCoordVar * 0.25 +
    normalizedStrain * 0.20
  );

  let disorderClass: DisorderMetrics["disorderClass"];
  if (disorderScore < 0.05) disorderClass = "perfect";
  else if (disorderScore < 0.15) disorderClass = "mild";
  else if (disorderScore < 0.35) disorderClass = "moderate";
  else if (disorderScore < 0.60) disorderClass = "strong";
  else disorderClass = "amorphous";

  return {
    bondLengths,
    bondMean,
    bondVariance: bondVar,
    bondStdDev,
    bondMin: bondLengths.length > 0 ? Math.min(...bondLengths) : 0,
    bondMax: bondLengths.length > 0 ? Math.max(...bondLengths) : 0,
    totalBonds: bonds.length,
    coordinationNumbers: coordNumbers,
    coordinationMean: coordMean,
    coordinationVariance: coordVar,
    coordinationMin: coordNumbers.length > 0 ? Math.min(...coordNumbers) : 0,
    coordinationMax: coordNumbers.length > 0 ? Math.max(...coordNumbers) : 0,
    idealCoordination: idealCoord,
    coordinationDeficit: coordDeficit,
    localStrains: strains,
    localStrainMean: strainMean,
    localStrainMax: strainMax,
    defectNeighborStrain,
    disorderScore,
    disorderClass,
    vacancyFraction,
    substitutionFraction: subFraction,
    interstitialFraction: intFraction,
    siteMixingFraction: mixFraction,
    amorphousFraction: amorphFraction,
  };
}

const metricsStats: DisorderMetricsStats = {
  totalAnalyzed: 0,
  avgDisorderScore: 0,
  maxDisorderScore: 0,
  byClass: { perfect: 0, mild: 0, moderate: 0, strong: 0, amorphous: 0 },
  avgBondVariance: 0,
  avgCoordinationVariance: 0,
  avgLocalStrain: 0,
  recentAnalyses: [],
  topDisordered: [],
};

let totalScoreSum = 0;
let totalBondVarSum = 0;
let totalCoordVarSum = 0;
let totalStrainSum = 0;

export function recordMetricsAnalysis(formula: string, metrics: DisorderMetrics, totalAtoms: number): void {
  metricsStats.totalAnalyzed++;
  totalScoreSum += metrics.disorderScore;
  totalBondVarSum += metrics.bondVariance;
  totalCoordVarSum += metrics.coordinationVariance;
  totalStrainSum += metrics.localStrainMean;

  metricsStats.avgDisorderScore = totalScoreSum / metricsStats.totalAnalyzed;
  metricsStats.avgBondVariance = totalBondVarSum / metricsStats.totalAnalyzed;
  metricsStats.avgCoordinationVariance = totalCoordVarSum / metricsStats.totalAnalyzed;
  metricsStats.avgLocalStrain = totalStrainSum / metricsStats.totalAnalyzed;

  if (metrics.disorderScore > metricsStats.maxDisorderScore) {
    metricsStats.maxDisorderScore = metrics.disorderScore;
  }

  metricsStats.byClass[metrics.disorderClass] = (metricsStats.byClass[metrics.disorderClass] || 0) + 1;

  const entry = {
    formula,
    disorderScore: metrics.disorderScore,
    disorderClass: metrics.disorderClass,
    bondVariance: metrics.bondVariance,
    coordinationVariance: metrics.coordinationVariance,
    totalAtoms,
  };

  metricsStats.recentAnalyses.unshift(entry);
  if (metricsStats.recentAnalyses.length > 20) metricsStats.recentAnalyses.length = 20;

  const topEntry = {
    formula,
    disorderScore: metrics.disorderScore,
    disorderClass: metrics.disorderClass,
    bondVariance: metrics.bondVariance,
    coordinationVariance: metrics.coordinationVariance,
  };
  metricsStats.topDisordered.push(topEntry);
  metricsStats.topDisordered.sort((a, b) => b.disorderScore - a.disorderScore);
  if (metricsStats.topDisordered.length > 15) metricsStats.topDisordered.length = 15;
}

export function getDisorderMetricsStats(): DisorderMetricsStats {
  return { ...metricsStats };
}

export function extractMLFeatures(metrics: DisorderMetrics): Record<string, number> {
  return {
    disorder_score: metrics.disorderScore,
    bond_variance: metrics.bondVariance,
    bond_std_dev: metrics.bondStdDev,
    bond_mean: metrics.bondMean,
    coordination_mean: metrics.coordinationMean,
    coordination_variance: metrics.coordinationVariance,
    coordination_deficit: metrics.coordinationDeficit,
    local_strain_mean: metrics.localStrainMean,
    local_strain_max: metrics.localStrainMax,
    defect_neighbor_strain: metrics.defectNeighborStrain,
    vacancy_fraction: metrics.vacancyFraction,
    substitution_fraction: metrics.substitutionFraction,
    interstitial_fraction: metrics.interstitialFraction,
    site_mixing_fraction: metrics.siteMixingFraction,
    amorphous_fraction: metrics.amorphousFraction,
  };
}
