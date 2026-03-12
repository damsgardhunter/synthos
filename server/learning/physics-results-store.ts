export interface PhysicsResult {
  formula: string;
  pressure: number;
  lambda: number;
  omegaLog: number;
  tc: number;
  dosAtEF: number;
  phononStable: boolean;
  muStar: number;
  omega2: number;
  gapRatio: number;
  isStrongCoupling: boolean;
  isotopeAlpha: number;
  formationEnergy: number | null;
  bandGap: number | null;
  isMetallic: boolean;
  tier: "full-dft" | "xtb" | "surrogate";
  alpha2FPeak: number;
  alpha2FPeakFreq: number;
  modeResolvedLambda: Record<string, number>;
  timestamp: number;
  advancedConstraints?: AdvancedConstraintFeatures;
}

export interface AdvancedConstraintFeatures {
  compositeScore: number;
  compositeBoost: number;
  nestingScore: number;
  nestingStrength: string;
  hybridizationScore: number;
  hybridizationType: string;
  lifshitzProximity: number;
  qcpScore: number;
  qcpType: string;
  dimensionalityScore: number;
  dimensionClass: string;
  anisotropy: number;
  softModeScore: number;
  softModeStable: boolean;
  chargeTransferScore: number;
  chargeTransferDelta: number;
  chargeTransferType: string;
  polarizabilityScore: number;
  dielectricConstant: number;
  screeningStrength: string;
}

export interface PhysicsFeatures {
  verifiedLambda: number;
  verifiedOmegaLog: number;
  verifiedTc: number;
  verifiedDosEF: number;
  verifiedPhononStable: boolean;
  verifiedMuStar: number;
  verifiedGapRatio: number;
  verifiedIsStrongCoupling: boolean;
  verifiedFormationEnergy: number | null;
  verifiedBandGap: number | null;
  verifiedIsMetallic: boolean;
  lambdaProxy: number;
  alphaCouplingStrength: number;
  phononHardness: number;
  massEnhancement: number;
  couplingAsymmetry: number;
  tier: "full-dft" | "xtb" | "surrogate";
  hasVerifiedData: boolean;
  lifshitzProximity: number;
  chargeTransferDelta: number;
  chargeTransferScore: number;
  qcpScore: number;
  dimensionalityScore: number;
  anisotropy: number;
  softModeScore: number;
  polarizabilityScore: number;
  dielectricConstant: number;
  compositeBoost: number;
}

const TIER_PRIORITY: Record<string, number> = {
  "full-dft": 3,
  "xtb": 2,
  "surrogate": 1,
};

const MAX_STORE_SIZE = 5000;

const store = new Map<string, PhysicsResult>();

function touchAccess(formula: string) {
  const val = store.get(formula);
  if (val) {
    store.delete(formula);
    store.set(formula, val);
  }
}

function evictIfNeeded() {
  while (store.size > MAX_STORE_SIZE) {
    const oldest = store.keys().next().value;
    if (oldest === undefined) break;
    store.delete(oldest);
    derivedFeaturesCache.delete(oldest);
  }
}

export function recordPhysicsResult(result: PhysicsResult): void {
  const existing = store.get(result.formula);

  if (existing) {
    const existingPriority = TIER_PRIORITY[existing.tier] ?? 0;
    const newPriority = TIER_PRIORITY[result.tier] ?? 0;
    if (newPriority < existingPriority) {
      return;
    }
  }

  store.set(result.formula, { ...result });
  touchAccess(result.formula);
  evictIfNeeded();

  const derived = computeDerivedFeatures(result);
  derivedFeaturesCache.set(result.formula, derived);
  for (const cb of [...featureRecalcCallbacks]) {
    try { cb(result.formula, derived); } catch {}
  }
}

export function getPhysicsResult(formula: string): PhysicsResult | null {
  const result = store.get(formula);
  if (result) {
    touchAccess(formula);
    return result;
  }
  return null;
}

export function getPhysicsFeatures(formula: string): PhysicsFeatures | null {
  const result = store.get(formula);
  if (!result) return null;

  touchAccess(formula);

  const modeValues = Object.values(result.modeResolvedLambda);
  const maxMode = modeValues.length > 0 ? Math.max(...modeValues) : 0;
  const minMode = modeValues.length > 0 ? Math.min(...modeValues.filter(v => v > 0.001)) : 0.001;

  const rawLambdaProxy = result.dosAtEF > 0 && result.omegaLog > 0
    ? result.dosAtEF * result.alpha2FPeak / (result.omegaLog * 0.01)
    : 0;
  const lambdaProxy = Math.min(3.0, Math.max(0, rawLambdaProxy));

  const phononHardness = result.omega2 > 0 ? result.omegaLog / result.omega2 : 0;

  const massEnhancement = 1 + result.lambda;

  const couplingAsymmetry = minMode > 0.001 ? maxMode / minMode : maxMode > 0 ? 10.0 : 1.0;

  const ac = result.advancedConstraints;
  return {
    verifiedLambda: result.lambda,
    verifiedOmegaLog: result.omegaLog,
    verifiedTc: result.tc,
    verifiedDosEF: result.dosAtEF,
    verifiedPhononStable: result.phononStable,
    verifiedMuStar: result.muStar,
    verifiedGapRatio: result.gapRatio,
    verifiedIsStrongCoupling: result.isStrongCoupling,
    verifiedFormationEnergy: result.formationEnergy,
    verifiedBandGap: result.bandGap,
    verifiedIsMetallic: result.isMetallic,
    lambdaProxy,
    alphaCouplingStrength: result.alpha2FPeak,
    phononHardness,
    massEnhancement,
    couplingAsymmetry,
    tier: result.tier,
    hasVerifiedData: true,
    lifshitzProximity: ac?.lifshitzProximity ?? 0,
    chargeTransferDelta: ac?.chargeTransferDelta ?? 0,
    chargeTransferScore: ac?.chargeTransferScore ?? 0,
    qcpScore: ac?.qcpScore ?? 0,
    dimensionalityScore: ac?.dimensionalityScore ?? 0,
    anisotropy: ac?.anisotropy ?? 0,
    softModeScore: ac?.softModeScore ?? 0,
    polarizabilityScore: ac?.polarizabilityScore ?? 0,
    dielectricConstant: ac?.dielectricConstant ?? 0,
    compositeBoost: ac?.compositeBoost ?? 1.0,
  };
}

export interface DerivedFeatures {
  bandwidth: number;
  vanHoveDistance: number;
  bondStiffness: number;
  electronPhononCouplingDensity: number;
  spectralWeight: number;
  anharmonicRatio: number;
  couplingEfficiency: number;
}

export function computeDerivedFeatures(result: PhysicsResult): DerivedFeatures {
  const bandwidth = result.dosAtEF > 0
    ? Math.min(10, 1.0 / (result.dosAtEF * 0.1))
    : 5.0;

  const vanHoveDistance = result.dosAtEF > 2.0
    ? Math.max(0, 1.0 - (result.dosAtEF - 2.0) / 5.0)
    : 1.0;

  const bondStiffness = result.omega2 > 0
    ? result.omega2 * result.omega2 * 0.0001
    : result.omegaLog > 0 ? result.omegaLog * 0.01 : 0;

  const electronPhononCouplingDensity = result.lambda > 0 && result.dosAtEF > 0
    ? result.lambda * result.dosAtEF
    : 0;

  const spectralWeight = result.alpha2FPeak > 0 && result.alpha2FPeakFreq > 0
    ? result.alpha2FPeak * result.alpha2FPeakFreq * 0.01
    : 0;

  const anharmonicRatio = result.omega2 > 0 && result.omegaLog > 0
    ? result.omega2 / result.omegaLog
    : 1.0;

  let couplingEfficiency = 0;
  if (result.lambda > 0.01 && result.tc > 0 && result.omegaLog > 0) {
    const adDenom = result.lambda - result.muStar * (1 + 0.62 * result.lambda);
    if (adDenom > 0.05) {
      const expArg = -1.04 * (1 + result.lambda) / adDenom;
      if (expArg > -30 && expArg < 30) {
        const adVal = result.omegaLog * Math.exp(expArg);
        couplingEfficiency = adVal > 1e-4 ? result.tc / adVal : 0;
      }
    }
  }
  if (!Number.isFinite(couplingEfficiency)) couplingEfficiency = 0;

  return {
    bandwidth,
    vanHoveDistance,
    bondStiffness,
    electronPhononCouplingDensity,
    spectralWeight,
    anharmonicRatio,
    couplingEfficiency: Number.isFinite(couplingEfficiency) ? Math.min(2.0, Math.max(0, couplingEfficiency)) : 0,
  };
}

const derivedFeaturesCache = new Map<string, DerivedFeatures>();

export function recalculateDerivedFeatures(formula: string): DerivedFeatures | null {
  const result = store.get(formula);
  if (!result) return null;

  const derived = computeDerivedFeatures(result);
  derivedFeaturesCache.set(formula, derived);
  return derived;
}

export function recalculateAllDerivedFeatures(): number {
  let count = 0;
  for (const [formula, result] of store) {
    const derived = computeDerivedFeatures(result);
    derivedFeaturesCache.set(formula, derived);
    count++;
  }
  return count;
}

export function getDerivedFeatures(formula: string): DerivedFeatures | null {
  if (derivedFeaturesCache.has(formula)) {
    return derivedFeaturesCache.get(formula)!;
  }
  return recalculateDerivedFeatures(formula);
}

const featureRecalcCallbacks: ((formula: string, derived: DerivedFeatures) => void)[] = [];

export function onFeatureRecalculation(callback: (formula: string, derived: DerivedFeatures) => void): void {
  featureRecalcCallbacks.push(callback);
}

export function getAllPhysicsResults(): PhysicsResult[] {
  return Array.from(store.values());
}

export function getPhysicsStoreSize(): number {
  return store.size;
}

export function getPhysicsStoreStats() {
  const results = Array.from(store.values());
  const n = results.length || 1;

  const tierCounts = { "full-dft": 0, "xtb": 0, "surrogate": 0 };
  let lambdaSum = 0;
  let tcSum = 0;
  let dosSum = 0;
  let strongCoupling = 0;
  let phononStable = 0;

  for (const r of results) {
    tierCounts[r.tier]++;
    lambdaSum += r.lambda;
    tcSum += r.tc;
    dosSum += r.dosAtEF;
    if (r.isStrongCoupling) strongCoupling++;
    if (r.phononStable) phononStable++;
  }

  return {
    totalEntries: results.length,
    maxSize: MAX_STORE_SIZE,
    tierBreakdown: tierCounts,
    avgLambda: Number((lambdaSum / n).toFixed(4)),
    avgTc: Number((tcSum / n).toFixed(2)),
    avgDosAtEF: Number((dosSum / n).toFixed(3)),
    strongCouplingFraction: Number((strongCoupling / n).toFixed(3)),
    phononStableFraction: Number((phononStable / n).toFixed(3)),
  };
}
