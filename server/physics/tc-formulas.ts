const CM1_TO_KELVIN = 1.4388;
const MEV_TO_KELVIN = 11.604;

export interface AllenDynesTcInput {
  lambda: number;
  omegaLog: number;
  muStar: number;
  omega2Avg?: number;
  isHydride?: boolean;
}

export interface AllenDynesTcResult {
  tc: number;
  f1: number;
  f2: number;
  regime: "weak" | "intermediate" | "strong" | "very-strong";
  method: "allen-dynes" | "sisso-hydride";
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
    f2 = Math.max(0.8, f2);
  }

  const exponent = -1.04 * (1 + lambda) / denominator;
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  if (!Number.isFinite(tc) || tc < 0) tc = 0;

  let method: "allen-dynes" | "sisso-hydride" = "allen-dynes";

  if (isHydride && lambda > 1.5 && omegaLogK > 0) {
    const tcSisso = sissoHydrideTc(lambda, omegaLogK, muStar);
    if (tcSisso > tc) {
      tc = tcSisso;
      method = "sisso-hydride";
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
  const tc = 0.182 * omegaLogK * Math.pow(lambdaEff, 0.572) *
    Math.pow(1 + 6.5 * muStar * Math.log(lambda), -0.278);
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
  let tc = (omegaLogK / 1.2) * f1 * f2 * Math.exp(exponent);

  if (!Number.isFinite(tc) || tc < 0) tc = 0;

  if (isHydride && lambda > 1.5) {
    const tcSisso = sissoHydrideTc(lambda, omegaLogK, muStar);
    if (tcSisso > tc) tc = tcSisso;
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

export { CM1_TO_KELVIN, MEV_TO_KELVIN };
