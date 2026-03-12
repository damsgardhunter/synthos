import { ELEMENTAL_DATA, type ElementalProperties } from "./elemental-data";
import { parseFormulaCounts } from "./utils";

interface RunningStats {
  sum: number;
  sumSq: number;
  wSum: number;
  wSumSq: number;
  min: number;
  max: number;
  count: number;
  logSum: number;
  allPositive: boolean;
}

function initStats(): RunningStats {
  return { sum: 0, sumSq: 0, wSum: 0, wSumSq: 0, min: Infinity, max: -Infinity, count: 0, logSum: 0, allPositive: true };
}

function pushStat(s: RunningStats, value: number, weight: number): void {
  s.sum += value * weight;
  s.sumSq += weight * value * value;
  s.wSum += weight;
  s.wSumSq += weight * weight;
  if (value < s.min) s.min = value;
  if (value > s.max) s.max = value;
  s.count++;
  if (value > 0) {
    s.logSum += weight * Math.log(value);
  } else {
    s.allPositive = false;
  }
}

function statMean(s: RunningStats): number {
  return s.wSum > 0 ? s.sum / s.wSum : 0;
}

function statStd(s: RunningStats): number {
  if (s.count < 2 || s.wSum <= 0) return 0;
  const mean = s.sum / s.wSum;
  const popVar = (s.sumSq / s.wSum) - mean * mean;
  const effectiveN = (s.wSum * s.wSum) / s.wSumSq;
  if (effectiveN <= 1.001) return Math.sqrt(Math.max(0, popVar));
  const besselFactor = effectiveN / (effectiveN - 1);
  return Math.sqrt(Math.max(0, popVar * besselFactor));
}

function statMin(s: RunningStats): number {
  return s.count > 0 ? s.min : 0;
}

function statMax(s: RunningStats): number {
  return s.count > 0 ? s.max : 0;
}

function statGeomMean(s: RunningStats): number {
  if (s.count === 0 || s.wSum <= 0 || !s.allPositive) return 0;
  return Math.exp(s.logSum / s.wSum);
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

function evictCacheIfNeeded(): void {
  if (compositionCache.size < COMP_CACHE_MAX) return;
  const now = Date.now();
  let oldestKey: string | null = null;
  let oldestTs = Infinity;
  const keys = Array.from(compositionCache.keys());
  for (const key of keys) {
    const entry = compositionCache.get(key)!;
    if (now - entry.ts > COMP_CACHE_TTL) {
      compositionCache.delete(key);
      if (compositionCache.size < COMP_CACHE_MAX) return;
    } else if (entry.ts < oldestTs) {
      oldestTs = entry.ts;
      oldestKey = key;
    }
  }
  if (oldestKey && compositionCache.size >= COMP_CACHE_MAX) {
    compositionCache.delete(oldestKey);
  }
}

export function computeCompositionFeatures(formula: string): CompositionFeatures {
  const cached = compositionCache.get(formula);
  if (cached && Date.now() - cached.ts < COMP_CACHE_TTL) {
    cached.ts = Date.now();
    return cached.features;
  }

  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const fractions: number[] = [];
  const totalAtoms = Object.values(counts).reduce((s, c) => s + c, 0);

  for (const el of elements) {
    fractions.push(counts[el] / totalAtoms);
  }

  const en = initStats();
  const radius = initStats();
  const mass = initStats();
  const vec = initStats();
  const ie = initStats();
  const ea = initStats();
  const debye = initStats();
  const bulk = initStats();
  const melt = initStats();
  const stoner = initStats();
  const hubbard = initStats();
  const hopfield = initStats();
  const gruneisen = initStats();
  const an = initStats();
  const petti = initStats();
  const gamma = initStats();
  const lattice = initStats();

  let dCount = 0, fCount = 0, pCount = 0, sCount = 0;

  for (let i = 0; i < elements.length; i++) {
    const d = ELEMENTAL_DATA[elements[i]] ?? null;
    const w = fractions[i];
    if (!d) continue;

    if (d.paulingElectronegativity != null) pushStat(en, d.paulingElectronegativity, w);
    pushStat(radius, d.atomicRadius, w);
    pushStat(mass, d.atomicMass, w);
    pushStat(vec, d.valenceElectrons, w);
    pushStat(ie, d.firstIonizationEnergy, w);
    if (d.electronAffinity != null) pushStat(ea, d.electronAffinity, w);
    if (d.debyeTemperature != null) pushStat(debye, d.debyeTemperature, w);
    if (d.bulkModulus != null) pushStat(bulk, d.bulkModulus, w);
    if (d.meltingPoint != null) pushStat(melt, d.meltingPoint, w);
    if (d.stonerParameter != null) pushStat(stoner, d.stonerParameter, w);
    if (d.hubbardU != null) pushStat(hubbard, d.hubbardU, w);
    if (d.mcMillanHopfieldEta != null) pushStat(hopfield, d.mcMillanHopfieldEta, w);
    if (d.gruneisenParameter != null) pushStat(gruneisen, d.gruneisenParameter, w);
    pushStat(an, d.atomicNumber, w);
    pushStat(petti, d.pettiforScale, w);
    if (d.sommerfeldGamma != null) pushStat(gamma, d.sommerfeldGamma, w);
    if (d.latticeConstant != null) pushStat(lattice, d.latticeConstant, w);

    const el = elements[i];
    if (D_BLOCK.has(el)) dCount += w;
    if (F_BLOCK.has(el)) fCount += w;
    if (P_BLOCK.has(el)) pCount += w;
    if (S_BLOCK.has(el)) sCount += w;
  }

  const enMean = statMean(en);
  const enStdVal = statStd(en);
  const enMinVal = statMin(en);
  const enMaxVal = statMax(en);
  const enGeomMean = statGeomMean(en);

  const radiusMean = statMean(radius);
  const radiusStdVal = statStd(radius);
  const radiusMinVal = statMin(radius);
  const radiusMaxVal = statMax(radius);

  const massMean = statMean(mass);
  const massStdVal = statStd(mass);
  const massMinVal = statMin(mass);
  const massMaxVal = statMax(mass);

  const vecMean = statMean(vec);
  const vecStdVal = statStd(vec);

  const ieMean = statMean(ie);
  const ieStdVal = statStd(ie);
  const ieMinVal = statMin(ie);
  const ieMaxVal = statMax(ie);

  const eaMean = statMean(ea);
  const eaStdVal = statStd(ea);

  const debyeMean = statMean(debye);
  const debyeStdVal = statStd(debye);
  const debyeGeomMean = statGeomMean(debye);

  const bulkModMean = statMean(bulk);
  const bulkModStdVal = statStd(bulk);

  const meltingMean = statMean(melt);
  const meltingStdVal = statStd(melt);
  const meltingMinVal = statMin(melt);
  const meltingMaxVal = statMax(melt);

  const FCC_PACKING = 0.74;
  const avgRadius_m = radiusMean * 1e-12;
  const sphereVol = avgRadius_m > 0 ? (4 / 3) * Math.PI * avgRadius_m ** 3 * 1e30 : 0;
  const volumePerAtom = sphereVol > 0 ? sphereVol / FCC_PACKING : 0;
  const densityEstimate = massMean > 0 && volumePerAtom > 0 ? massMean / (volumePerAtom * 6.022e23) * 1e24 : 0;

  const ionicCharacter = en.count >= 2 ? 1 - Math.exp(-0.25 * (enMaxVal - enMinVal) ** 2) : 0;
  const covalentCharacter = 1 - ionicCharacter;
  const totalBlock = dCount + fCount + pCount + sCount || 1;
  const metallicCharacter = (dCount + fCount) / totalBlock;

  const enBasedCoord = enMean > 0 ? 12 * (1 - enStdVal / (enMean + 0.01)) : 6;
  const coordNumberEstimate = Math.min(12, Math.max(4, Math.round(
    metallicCharacter * Math.max(enBasedCoord, 10) + (1 - metallicCharacter) * enBasedCoord
  )));

  const shannonEntropy = fractions.reduce((s, f) => f > 0 ? s - f * Math.log(f) : s, 0);
  const avgFrac = 1 / Math.max(1, elements.length);
  const stoichVariance = fractions.reduce((s, f) => s + (f - avgFrac) ** 2, 0) / Math.max(1, elements.length);

  const gammaMean = statMean(gamma);
  const gammaStdVal = statStd(gamma);

  const latticeConstMean = statMean(lattice);
  const latticeConstStdVal = statStd(lattice);

  const pettiforMean = statMean(petti);
  const pettiforStdVal = statStd(petti);
  const pettiforRange = petti.count > 0 ? statMax(petti) - statMin(petti) : 0;

  const result: CompositionFeatures = {
    enMean, enStd: enStdVal, enMin: enMinVal, enMax: enMaxVal, enRange: enMaxVal - enMinVal, enGeomMean,
    radiusMean, radiusStd: radiusStdVal, radiusMin: radiusMinVal, radiusMax: radiusMaxVal, radiusRange: radiusMaxVal - radiusMinVal,
    massMean, massStd: massStdVal, massMin: massMinVal, massMax: massMaxVal,
    vecMean, vecStd: vecStdVal,
    ieMean, ieStd: ieStdVal, ieMin: ieMinVal, ieMax: ieMaxVal,
    eaMean, eaStd: eaStdVal,
    debyeMean, debyeStd: debyeStdVal, debyeGeomMean,
    bulkModMean, bulkModStd: bulkModStdVal,
    meltingMean, meltingStd: meltingStdVal, meltingMin: meltingMinVal, meltingMax: meltingMaxVal,
    densityEstimate, volumePerAtom, coordNumberEstimate,
    avgStonerParam: statMean(stoner),
    avgHubbardU: statMean(hubbard),
    avgHopfieldEta: statMean(hopfield),
    avgGruneisen: statMean(gruneisen),
    ionicCharacter, covalentCharacter, metallicCharacter,
    dElectronFrac: dCount, fElectronFrac: fCount, pElectronFrac: pCount, sElectronFrac: sCount,
    atomicNumberMean: statMean(an),
    atomicNumberStd: statStd(an),
    pettiforMean, pettiforStd: pettiforStdVal, pettiforRange,
    shannonEntropy, nAtoms: totalAtoms, stoichVariance,
    gammaMean, gammaStd: gammaStdVal,
    latticeConstMean, latticeConstStd: latticeConstStdVal,
  };

  evictCacheIfNeeded();
  compositionCache.set(formula, { features: result, ts: Date.now() });

  return result;
}

export function compositionFeatureVector(cf: CompositionFeatures): number[] {
  return [
    cf.enMean, cf.enStd, cf.enMin, cf.enMax, cf.enRange, cf.enGeomMean,
    cf.radiusMean, cf.radiusStd, cf.radiusMin, cf.radiusMax, cf.radiusRange,
    cf.massMean, cf.massStd, cf.massMin, cf.massMax,
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
  "comp_massMean", "comp_massStd", "comp_massMin", "comp_massMax",
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
