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
}

const TIER_PRIORITY: Record<string, number> = {
  "full-dft": 3,
  "xtb": 2,
  "surrogate": 1,
};

const MAX_STORE_SIZE = 5000;

const store = new Map<string, PhysicsResult>();
const accessOrder: string[] = [];

function touchAccess(formula: string) {
  const idx = accessOrder.indexOf(formula);
  if (idx !== -1) accessOrder.splice(idx, 1);
  accessOrder.push(formula);
}

function evictIfNeeded() {
  while (store.size > MAX_STORE_SIZE && accessOrder.length > 0) {
    const oldest = accessOrder.shift()!;
    store.delete(oldest);
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

  const lambdaProxy = result.dosAtEF > 0 && result.omegaLog > 0
    ? result.dosAtEF * result.alpha2FPeak / (result.omegaLog * 0.01)
    : 0;

  const phononHardness = result.omega2 > 0 ? result.omegaLog / result.omega2 : 0;

  const massEnhancement = 1 + result.lambda;

  const couplingAsymmetry = minMode > 0.001 ? maxMode / minMode : maxMode > 0 ? 10.0 : 1.0;

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
  };
}

export function getAllPhysicsResults(): PhysicsResult[] {
  return Array.from(store.values());
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
