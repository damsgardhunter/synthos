import { ELEMENTAL_DATA, type ElementalProperties } from "./elemental-data";

function parseFormulaCounts(formula: string): Record<string, number> {
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  return counts;
}

function weightedMean(values: number[], weights: number[]): number {
  const totalW = weights.reduce((s, w) => s + w, 0);
  if (totalW === 0) return 0;
  return values.reduce((s, v, i) => s + v * weights[i], 0) / totalW;
}

function weightedStd(values: number[], weights: number[]): number {
  const mean = weightedMean(values, weights);
  const totalW = weights.reduce((s, w) => s + w, 0);
  if (totalW === 0) return 0;
  const variance = values.reduce((s, v, i) => s + weights[i] * (v - mean) ** 2, 0) / totalW;
  return Math.sqrt(variance);
}

function geometricMean(values: number[], weights: number[]): number {
  const totalW = weights.reduce((s, w) => s + w, 0);
  if (totalW === 0 || values.some(v => v <= 0)) return 0;
  const logSum = values.reduce((s, v, i) => s + weights[i] * Math.log(v), 0);
  return Math.exp(logSum / totalW);
}

function harmonicMean(values: number[], weights: number[]): number {
  const totalW = weights.reduce((s, w) => s + w, 0);
  if (totalW === 0 || values.some(v => v === 0)) return 0;
  const recipSum = values.reduce((s, v, i) => s + weights[i] / v, 0);
  return totalW / recipSum;
}

export interface CompositionFeatures {
  enMean: number;
  enStd: number;
  enMin: number;
  enMax: number;
  enRange: number;
  enGeomMean: number;

  radiusMean: number;
  radiusStd: number;
  radiusMin: number;
  radiusMax: number;
  radiusRange: number;

  massMean: number;
  massStd: number;
  massVariance: number;
  massMin: number;
  massMax: number;

  vecMean: number;
  vecStd: number;

  ieMean: number;
  ieStd: number;
  ieMin: number;
  ieMax: number;

  eaMean: number;
  eaStd: number;

  debyeMean: number;
  debyeStd: number;
  debyeGeomMean: number;

  bulkModMean: number;
  bulkModStd: number;

  meltingMean: number;
  meltingStd: number;
  meltingMin: number;
  meltingMax: number;

  densityEstimate: number;
  volumePerAtom: number;
  coordNumberEstimate: number;

  avgStonerParam: number;
  avgHubbardU: number;
  avgHopfieldEta: number;
  avgGruneisen: number;

  ionicCharacter: number;
  covalentCharacter: number;
  metallicCharacter: number;

  dElectronFrac: number;
  fElectronFrac: number;
  pElectronFrac: number;
  sElectronFrac: number;

  atomicNumberMean: number;
  atomicNumberStd: number;

  pettiforMean: number;
  pettiforStd: number;
  pettiforRange: number;

  shannonEntropy: number;
  nAtoms: number;
  stoichVariance: number;

  gammaMean: number;
  gammaStd: number;

  latticeConstMean: number;
  latticeConstStd: number;
}

const D_BLOCK = new Set([
  "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
  "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
  "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
]);
const F_BLOCK = new Set([
  "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
  "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
]);
const P_BLOCK = new Set([
  "B","C","N","O","F","Ne","Al","Si","P","S","Cl","Ar",
  "Ga","Ge","As","Se","Br","Kr","In","Sn","Sb","Te","I","Xe",
  "Tl","Pb","Bi","Po","At","Rn",
]);
const S_BLOCK = new Set([
  "H","He","Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
]);

const compositionCache = new Map<string, { features: CompositionFeatures; ts: number }>();
const COMP_CACHE_MAX = 2000;
const COMP_CACHE_TTL = 30 * 60 * 1000;

export function computeCompositionFeatures(formula: string): CompositionFeatures {
  const cached = compositionCache.get(formula);
  if (cached && Date.now() - cached.ts < COMP_CACHE_TTL) return cached.features;

  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const fractions: number[] = [];
  const totalAtoms = Object.values(counts).reduce((s, c) => s + c, 0);

  const elData: (ElementalProperties | null)[] = [];
  for (const el of elements) {
    fractions.push(counts[el] / totalAtoms);
    elData.push(ELEMENTAL_DATA[el] ?? null);
  }

  const enVals: number[] = [];
  const enWeights: number[] = [];
  const radiusVals: number[] = [];
  const radiusWeights: number[] = [];
  const massVals: number[] = [];
  const massWeights: number[] = [];
  const vecVals: number[] = [];
  const vecWeights: number[] = [];
  const ieVals: number[] = [];
  const ieWeights: number[] = [];
  const eaVals: number[] = [];
  const eaWeights: number[] = [];
  const debyeVals: number[] = [];
  const debyeWeights: number[] = [];
  const bulkVals: number[] = [];
  const bulkWeights: number[] = [];
  const meltVals: number[] = [];
  const meltWeights: number[] = [];
  const stonerVals: number[] = [];
  const stonerWeights: number[] = [];
  const hubbardVals: number[] = [];
  const hubbardWeights: number[] = [];
  const hopfieldVals: number[] = [];
  const hopfieldWeights: number[] = [];
  const gruneisenVals: number[] = [];
  const gruneisenWeights: number[] = [];
  const anVals: number[] = [];
  const anWeights: number[] = [];
  const pettiVals: number[] = [];
  const pettiWeights: number[] = [];
  const gammaVals: number[] = [];
  const gammaWeights: number[] = [];
  const latticeVals: number[] = [];
  const latticeWeights: number[] = [];

  let dCount = 0, fCount = 0, pCount = 0, sCount = 0;

  for (let i = 0; i < elements.length; i++) {
    const d = elData[i];
    const w = fractions[i];
    if (!d) continue;

    if (d.paulingElectronegativity != null) { enVals.push(d.paulingElectronegativity); enWeights.push(w); }
    radiusVals.push(d.atomicRadius); radiusWeights.push(w);
    massVals.push(d.atomicMass); massWeights.push(w);
    vecVals.push(d.valenceElectrons); vecWeights.push(w);
    ieVals.push(d.firstIonizationEnergy); ieWeights.push(w);
    if (d.electronAffinity != null) { eaVals.push(d.electronAffinity); eaWeights.push(w); }
    if (d.debyeTemperature != null) { debyeVals.push(d.debyeTemperature); debyeWeights.push(w); }
    if (d.bulkModulus != null) { bulkVals.push(d.bulkModulus); bulkWeights.push(w); }
    if (d.meltingPoint != null) { meltVals.push(d.meltingPoint); meltWeights.push(w); }
    if (d.stonerParameter != null) { stonerVals.push(d.stonerParameter); stonerWeights.push(w); }
    if (d.hubbardU != null) { hubbardVals.push(d.hubbardU); hubbardWeights.push(w); }
    if (d.mcMillanHopfieldEta != null) { hopfieldVals.push(d.mcMillanHopfieldEta); hopfieldWeights.push(w); }
    if (d.gruneisenParameter != null) { gruneisenVals.push(d.gruneisenParameter); gruneisenWeights.push(w); }
    anVals.push(d.atomicNumber); anWeights.push(w);
    pettiVals.push(d.pettiforScale); pettiWeights.push(w);
    if (d.sommerfeldGamma != null) { gammaVals.push(d.sommerfeldGamma); gammaWeights.push(w); }
    if (d.latticeConstant != null) { latticeVals.push(d.latticeConstant); latticeWeights.push(w); }

    const el = elements[i];
    if (D_BLOCK.has(el)) dCount += w;
    if (F_BLOCK.has(el)) fCount += w;
    if (P_BLOCK.has(el)) pCount += w;
    if (S_BLOCK.has(el)) sCount += w;
  }

  const enMean = enVals.length > 0 ? weightedMean(enVals, enWeights) : 0;
  const enStd = enVals.length > 1 ? weightedStd(enVals, enWeights) : 0;
  const enMin = enVals.length > 0 ? Math.min(...enVals) : 0;
  const enMax = enVals.length > 0 ? Math.max(...enVals) : 0;
  const enGeomMean = enVals.length > 0 ? geometricMean(enVals, enWeights) : 0;

  const radiusMean = radiusVals.length > 0 ? weightedMean(radiusVals, radiusWeights) : 0;
  const radiusStd = radiusVals.length > 1 ? weightedStd(radiusVals, radiusWeights) : 0;
  const radiusMin = radiusVals.length > 0 ? Math.min(...radiusVals) : 0;
  const radiusMax = radiusVals.length > 0 ? Math.max(...radiusVals) : 0;

  const massMean = massVals.length > 0 ? weightedMean(massVals, massWeights) : 0;
  const massStd = massVals.length > 1 ? weightedStd(massVals, massWeights) : 0;
  const massMin = massVals.length > 0 ? Math.min(...massVals) : 0;
  const massMax = massVals.length > 0 ? Math.max(...massVals) : 0;

  const vecMean = vecVals.length > 0 ? weightedMean(vecVals, vecWeights) : 0;
  const vecStd = vecVals.length > 1 ? weightedStd(vecVals, vecWeights) : 0;

  const ieMean = ieVals.length > 0 ? weightedMean(ieVals, ieWeights) : 0;
  const ieStd = ieVals.length > 1 ? weightedStd(ieVals, ieWeights) : 0;
  const ieMin = ieVals.length > 0 ? Math.min(...ieVals) : 0;
  const ieMax = ieVals.length > 0 ? Math.max(...ieVals) : 0;

  const eaMean = eaVals.length > 0 ? weightedMean(eaVals, eaWeights) : 0;
  const eaStd = eaVals.length > 1 ? weightedStd(eaVals, eaWeights) : 0;

  const debyeMean = debyeVals.length > 0 ? weightedMean(debyeVals, debyeWeights) : 0;
  const debyeStd = debyeVals.length > 1 ? weightedStd(debyeVals, debyeWeights) : 0;
  const debyeGeomMean = debyeVals.length > 0 ? geometricMean(debyeVals, debyeWeights) : 0;

  const bulkModMean = bulkVals.length > 0 ? weightedMean(bulkVals, bulkWeights) : 0;
  const bulkModStd = bulkVals.length > 1 ? weightedStd(bulkVals, bulkWeights) : 0;

  const meltingMean = meltVals.length > 0 ? weightedMean(meltVals, meltWeights) : 0;
  const meltingStd = meltVals.length > 1 ? weightedStd(meltVals, meltWeights) : 0;
  const meltingMin = meltVals.length > 0 ? Math.min(...meltVals) : 0;
  const meltingMax = meltVals.length > 0 ? Math.max(...meltVals) : 0;

  const avgRadius_m = radiusMean * 1e-12;
  const volumePerAtom = avgRadius_m > 0 ? (4 / 3) * Math.PI * avgRadius_m ** 3 * 1e30 : 0;
  const densityEstimate = massMean > 0 && volumePerAtom > 0 ? massMean / (volumePerAtom * 6.022e23) * 1e24 : 0;

  const avgEN = enMean;
  const coordNumberEstimate = avgEN > 0 ? Math.min(12, Math.max(4, Math.round(12 * (1 - enStd / (avgEN + 0.01))))) : 6;

  const ionicCharacter = enVals.length >= 2 ? 1 - Math.exp(-0.25 * (enMax - enMin) ** 2) : 0;
  const covalentCharacter = 1 - ionicCharacter;
  const totalBlock = dCount + fCount + pCount + sCount || 1;
  const metallicCharacter = (dCount + fCount) / totalBlock;

  const shannonEntropy = fractions.reduce((s, f) => f > 0 ? s - f * Math.log(f) : s, 0);
  const avgFrac = 1 / Math.max(1, elements.length);
  const stoichVariance = fractions.reduce((s, f) => s + (f - avgFrac) ** 2, 0) / Math.max(1, elements.length);

  const gammaMean = gammaVals.length > 0 ? weightedMean(gammaVals, gammaWeights) : 0;
  const gammaStd = gammaVals.length > 1 ? weightedStd(gammaVals, gammaWeights) : 0;

  const latticeConstMean = latticeVals.length > 0 ? weightedMean(latticeVals, latticeWeights) : 0;
  const latticeConstStd = latticeVals.length > 1 ? weightedStd(latticeVals, latticeWeights) : 0;

  const pettiforMean = pettiVals.length > 0 ? weightedMean(pettiVals, pettiWeights) : 0;
  const pettiforStd = pettiVals.length > 1 ? weightedStd(pettiVals, pettiWeights) : 0;
  const pettiforRange = pettiVals.length > 0 ? Math.max(...pettiVals) - Math.min(...pettiVals) : 0;

  const result: CompositionFeatures = {
    enMean, enStd, enMin, enMax, enRange: enMax - enMin, enGeomMean,
    radiusMean, radiusStd, radiusMin, radiusMax, radiusRange: radiusMax - radiusMin,
    massMean, massStd, massVariance: massStd ** 2, massMin, massMax,
    vecMean, vecStd,
    ieMean, ieStd, ieMin, ieMax,
    eaMean, eaStd,
    debyeMean, debyeStd, debyeGeomMean,
    bulkModMean, bulkModStd,
    meltingMean, meltingStd, meltingMin, meltingMax,
    densityEstimate, volumePerAtom, coordNumberEstimate,
    avgStonerParam: stonerVals.length > 0 ? weightedMean(stonerVals, stonerWeights) : 0,
    avgHubbardU: hubbardVals.length > 0 ? weightedMean(hubbardVals, hubbardWeights) : 0,
    avgHopfieldEta: hopfieldVals.length > 0 ? weightedMean(hopfieldVals, hopfieldWeights) : 0,
    avgGruneisen: gruneisenVals.length > 0 ? weightedMean(gruneisenVals, gruneisenWeights) : 0,
    ionicCharacter, covalentCharacter, metallicCharacter,
    dElectronFrac: dCount, fElectronFrac: fCount, pElectronFrac: pCount, sElectronFrac: sCount,
    atomicNumberMean: anVals.length > 0 ? weightedMean(anVals, anWeights) : 0,
    atomicNumberStd: anVals.length > 1 ? weightedStd(anVals, anWeights) : 0,
    pettiforMean, pettiforStd, pettiforRange,
    shannonEntropy, nAtoms: totalAtoms, stoichVariance,
    gammaMean, gammaStd,
    latticeConstMean, latticeConstStd,
  };

  if (compositionCache.size >= COMP_CACHE_MAX) {
    const oldest = compositionCache.keys().next().value;
    if (oldest) compositionCache.delete(oldest);
  }
  compositionCache.set(formula, { features: result, ts: Date.now() });

  return result;
}

export function compositionFeatureVector(cf: CompositionFeatures): number[] {
  return [
    cf.enMean, cf.enStd, cf.enMin, cf.enMax, cf.enRange, cf.enGeomMean,
    cf.radiusMean, cf.radiusStd, cf.radiusMin, cf.radiusMax, cf.radiusRange,
    cf.massMean, cf.massStd, cf.massVariance, cf.massMin, cf.massMax,
    cf.vecMean, cf.vecStd,
    cf.ieMean, cf.ieStd, cf.ieMin, cf.ieMax,
    cf.eaMean, cf.eaStd,
    cf.debyeMean, cf.debyeStd, cf.debyeGeomMean,
    cf.bulkModMean, cf.bulkModStd,
    cf.meltingMean, cf.meltingStd, cf.meltingMin, cf.meltingMax,
    cf.densityEstimate, cf.volumePerAtom, cf.coordNumberEstimate,
    cf.avgStonerParam, cf.avgHubbardU, cf.avgHopfieldEta, cf.avgGruneisen,
    cf.ionicCharacter, cf.covalentCharacter, cf.metallicCharacter,
    cf.dElectronFrac, cf.fElectronFrac, cf.pElectronFrac, cf.sElectronFrac,
    cf.atomicNumberMean, cf.atomicNumberStd,
    cf.pettiforMean, cf.pettiforStd, cf.pettiforRange,
    cf.shannonEntropy, cf.nAtoms, cf.stoichVariance,
    cf.gammaMean, cf.gammaStd,
    cf.latticeConstMean, cf.latticeConstStd,
  ];
}

export const COMPOSITION_FEATURE_NAMES = [
  "comp_enMean", "comp_enStd", "comp_enMin", "comp_enMax", "comp_enRange", "comp_enGeom",
  "comp_radiusMean", "comp_radiusStd", "comp_radiusMin", "comp_radiusMax", "comp_radiusRange",
  "comp_massMean", "comp_massStd", "comp_massVariance", "comp_massMin", "comp_massMax",
  "comp_vecMean", "comp_vecStd",
  "comp_ieMean", "comp_ieStd", "comp_ieMin", "comp_ieMax",
  "comp_eaMean", "comp_eaStd",
  "comp_debyeMean", "comp_debyeStd", "comp_debyeGeom",
  "comp_bulkModMean", "comp_bulkModStd",
  "comp_meltMean", "comp_meltStd", "comp_meltMin", "comp_meltMax",
  "comp_density", "comp_volPerAtom", "comp_coordNum",
  "comp_stoner", "comp_hubbard", "comp_hopfield", "comp_gruneisen",
  "comp_ionic", "comp_covalent", "comp_metallic",
  "comp_dFrac", "comp_fFrac", "comp_pFrac", "comp_sFrac",
  "comp_atomNumMean", "comp_atomNumStd",
  "comp_pettiMean", "comp_pettiStd", "comp_pettiRange",
  "comp_entropy", "comp_nAtoms", "comp_stoichVar",
  "comp_gammaMean", "comp_gammaStd",
  "comp_latticeMean", "comp_latticeStd",
];
