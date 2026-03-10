const CM1_TO_KELVIN = 1.4388;
const MEV_TO_KELVIN = 11.604;

export interface AllenDynesTcInput {
  lambda: number;
  omegaLog: number;
  muStar: number;
  omega2Avg?: number;
  isHydride?: boolean;
}

export type TcMethod = "allen-dynes" | "sisso-hydride" | "xie-strong-coupling";

export interface AllenDynesTcResult {
  tc: number;
  f1: number;
  f2: number;
  regime: "weak" | "intermediate" | "strong" | "very-strong";
  method: TcMethod;
}

export function allenDynesTcFull(input: AllenDynesTcInput): AllenDynesTcResult {
  const { lambda, omegaLog, muStar, omega2Avg, isHydride } = input;

  const omegaLogK = omegaLog * CM1_TO_KELVIN;

  let regime: "weak" | "intermediate" | "strong" | "very-strong" = "weak";
  if (lambda > 2.5) regime = "very-strong";
  else if (lambda > 1.5) regime = "strong";
  else if (lambda > 0.5) regime = "intermediate";

  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denominator) < 1e-6 || denominator <= 0 || omegaLogK <= 0) {
    return { tc: 0, f1: 1, f2: 1, regime, method: "allen-dynes" };
  }

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  let f2 = 1.0;
  if (omega2Avg && omega2Avg > 0 && omegaLog > 0) {
    const sqrtOmega2 = Math.sqrt(omega2Avg);
    const omegaRatio = sqrtOmega2 / omegaLog;
    const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
    f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
    f2 = Math.max(0.7, Math.min(1.5, f2));
  }

  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) return { tc: 0, f1: Number(f1.toFixed(4)), f2: Number(f2.toFixed(4)), regime, method: "allen-dynes" as const };
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  if (!Number.isFinite(tc) || tc < 0) tc = 0;

  let method: TcMethod = "allen-dynes";

  if (isHydride && lambda > 1.2 && omegaLogK > 0) {
    const tcSisso = sissoHydrideTc(lambda, omegaLogK, muStar);
    const tcXie = xieStrongCouplingTc(lambda, omegaLogK, muStar, omega2Avg ? Math.sqrt(omega2Avg) * CM1_TO_KELVIN : undefined);
    const bestAlt = tcXie > tcSisso ? { tc: tcXie, m: "xie-strong-coupling" as TcMethod } : { tc: tcSisso, m: "sisso-hydride" as TcMethod };
    if (bestAlt.tc > tc) {
      tc = bestAlt.tc;
      method = bestAlt.m;
    }
  }

  tc = Math.max(0, Math.min(500, tc));

  return {
    tc: Number(tc.toFixed(2)),
    f1: Number(f1.toFixed(4)),
    f2: Number(f2.toFixed(4)),
    regime,
    method,
  };
}

function sissoHydrideTc(lambda: number, omegaLogK: number, muStar: number): number {
  const lambdaEff = lambda - muStar * (1 + lambda);
  if (lambdaEff <= 0) return 0;

  let prefactor: number;
  let lambdaExp: number;
  let muStarDamping: number;

  if (lambda > 3.0) {
    prefactor = 0.22;
    lambdaExp = 0.48;
    muStarDamping = Math.pow(1 + 4.8 * muStar * Math.log(Math.max(1.01, lambda)), -0.32);
  } else if (lambda > 2.0) {
    prefactor = 0.20;
    lambdaExp = 0.52;
    muStarDamping = Math.pow(1 + 5.5 * muStar * Math.log(Math.max(1.01, lambda)), -0.30);
  } else {
    prefactor = 0.182;
    lambdaExp = 0.572;
    muStarDamping = Math.pow(1 + 6.5 * muStar * Math.log(Math.max(1.01, lambda)), -0.278);
  }

  let tc = prefactor * omegaLogK * Math.pow(lambdaEff, lambdaExp) * muStarDamping;

  if (lambda > 2.5) {
    const anharmonicCorr = 1 - 0.025 * Math.pow(lambda - 2.5, 1.1);
    tc *= Math.max(0.7, anharmonicCorr);
  }

  return Number.isFinite(tc) ? Math.max(0, tc) : 0;
}

function xieStrongCouplingTc(lambda: number, omegaLogK: number, muStar: number, omega2K?: number): number {
  if (lambda <= 1.0 || omegaLogK <= 0) return 0;

  const muStarEff = muStar * (1 + 0.5 * muStar);
  const lambdaNet = lambda - muStarEff * (1 + lambda);
  if (lambdaNet <= 0) return 0;

  let A: number, B: number;
  if (lambda > 3.0) {
    A = 0.14 * (1 + 4.0 / (lambda + 2.0));
    B = 0.92 * (1 + 0.30 * muStar);
  } else if (lambda > 2.0) {
    A = 0.13 * (1 + 4.5 / (lambda + 2.3));
    B = 0.98 * (1 + 0.34 * muStar);
  } else {
    A = 0.12 * (1 + 5.2 / (lambda + 2.6));
    B = 1.04 * (1 + 0.38 * muStar);
  }

  const C = lambda * (1 - 0.14 * muStar) - muStar * (1 + 0.62 * lambda);
  if (C <= 0) return 0;

  const prefactor = omegaLogK * A;
  const exponent = -B * (1 + lambda) / C;
  if (exponent < -50) return 0;

  let tc = prefactor * Math.exp(exponent);

  if (omega2K && omega2K > 0) {
    const ratio = omega2K / omegaLogK;
    if (lambda > 2.0) {
      const spectralCorr = 1 + 0.035 * Math.pow(ratio - 1, 1.8) * Math.sqrt(lambda);
      tc *= Math.min(1.4, Math.max(0.85, spectralCorr));
    } else {
      const spectralCorr = 1 + 0.0241 * Math.pow(ratio - 1, 2) * lambda;
      tc *= Math.min(1.3, Math.max(0.9, spectralCorr));
    }
  }

  if (lambda > 3.5) {
    const saturationDamping = 1 - 0.03 * Math.pow(lambda - 3.5, 1.0);
    tc *= Math.max(0.65, saturationDamping);
  } else if (lambda > 2.5) {
    const saturationDamping = 1 - 0.02 * Math.pow(lambda - 2.5, 1.2);
    tc *= Math.max(0.75, saturationDamping);
  }

  return Number.isFinite(tc) ? Math.max(0, tc) : 0;
}

export function allenDynesTcSimple(
  lambda: number,
  omegaLogK: number,
  muStar: number,
  omega2Avg?: number,
): number {
  return allenDynesTcFull({
    lambda,
    omegaLog: omegaLogK / CM1_TO_KELVIN,
    muStar,
    omega2Avg,
  }).tc;
}

export function allenDynesTcFromKelvin(
  lambda: number,
  omegaLogK: number,
  muStar: number,
  omega2Avg?: number,
  isHydride?: boolean,
): number {
  const denominator = lambda - muStar * (1 + 0.62 * lambda);
  if (Math.abs(denominator) < 1e-6 || denominator <= 0 || omegaLogK <= 0) {
    return 0;
  }

  const lambdaBar = 2.46 * (1 + 3.8 * muStar);
  const f1 = Math.pow(1 + Math.pow(lambda / lambdaBar, 3 / 2), 1 / 3);

  let f2 = 1.0;
  if (omega2Avg && omega2Avg > 0) {
    const omegaLogFromK = omegaLogK / CM1_TO_KELVIN;
    const sqrtOmega2 = Math.sqrt(omega2Avg);
    const omegaRatio = sqrtOmega2 / omegaLogFromK;
    const Lambda2 = 1.82 * (1 + 6.3 * muStar) * omegaRatio;
    f2 = 1 + (omegaRatio - 1) * lambda * lambda / (lambda * lambda + Lambda2 * Lambda2);
    f2 = Math.max(0.8, f2);
  }

  const exponent = -1.04 * (1 + lambda) / denominator;
  if (exponent < -50) return 0;
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  if (!Number.isFinite(tc) || tc < 0) tc = 0;

  if (isHydride && lambda > 1.2) {
    const tcSisso = sissoHydrideTc(lambda, omegaLogK, muStar);
    const tcXie = xieStrongCouplingTc(lambda, omegaLogK, muStar, omega2Avg ? Math.sqrt(omega2Avg) * CM1_TO_KELVIN : undefined);
    tc = Math.max(tc, tcSisso, tcXie);
  }

  return Math.max(0, Math.min(500, tc));
}

export function invertAllenDynesLambda(
  tc: number,
  thetaD: number,
  muStar: number,
): number {
  if (tc <= 0 || thetaD <= 0) return 0;
  const omegaLogK = thetaD * 0.60;
  let lambdaLow = 0.05;
  let lambdaHigh = 6.0;
  for (let i = 0; i < 80; i++) {
    const lambdaMid = (lambdaLow + lambdaHigh) / 2;
    const tcCalc = allenDynesTcFromKelvin(lambdaMid, omegaLogK, muStar);
    if (tcCalc < tc) lambdaLow = lambdaMid;
    else lambdaHigh = lambdaMid;
  }
  return (lambdaLow + lambdaHigh) / 2;
}

export function convertOmegaLogMeVToKelvin(omegaLogMeV: number): number {
  return omegaLogMeV * MEV_TO_KELVIN;
}

export function convertOmegaLogCm1ToKelvin(omegaLogCm1: number): number {
  return omegaLogCm1 * CM1_TO_KELVIN;
}

export function validateOmegaLog(
  surrogateOmegaLog: number,
  spectralOmegaLog: number | null,
  debyeTemp: number,
  maxPhononFreq: number,
): { validatedOmegaLog: number; corrected: boolean; warning: string | null } {
  let omega = surrogateOmegaLog;
  let corrected = false;
  let warning: string | null = null;

  if (spectralOmegaLog && spectralOmegaLog > 0) {
    const ratio = surrogateOmegaLog / spectralOmegaLog;
    if (ratio > 2.0 || ratio < 0.5) {
      omega = spectralOmegaLog;
      corrected = true;
      warning = `Surrogate omega_log=${surrogateOmegaLog.toFixed(1)} diverged from spectral=${spectralOmegaLog.toFixed(1)} (ratio=${ratio.toFixed(2)}), using spectral value`;
    } else if (ratio > 1.5 || ratio < 0.67) {
      omega = 0.3 * surrogateOmegaLog + 0.7 * spectralOmegaLog;
      corrected = true;
      warning = `Surrogate omega_log blended with spectral (ratio=${ratio.toFixed(2)})`;
    }
  }

  const debyeLimit = debyeTemp * 0.69 / CM1_TO_KELVIN;
  if (omega > debyeLimit * 1.5 && debyeLimit > 10) {
    omega = debyeLimit * 1.2;
    corrected = true;
    warning = (warning ? warning + "; " : "") + `omega_log capped to 1.2x Debye estimate (${debyeLimit.toFixed(0)} cm-1)`;
  }

  if (omega > maxPhononFreq * 0.8 && maxPhononFreq > 0) {
    omega = maxPhononFreq * 0.6;
    corrected = true;
    warning = (warning ? warning + "; " : "") + `omega_log capped below maxPhononFreq`;
  }

  omega = Math.max(5, omega);

  return {
    validatedOmegaLog: Number(omega.toFixed(1)),
    corrected,
    warning,
  };
}

export { CM1_TO_KELVIN, MEV_TO_KELVIN };
