import { predictPressureCurve, type PressureCurve, type PressurePoint } from "./pressure-aware-surrogate";
import { computeEnthalpyStability } from "./enthalpy-stability";

export interface PressureResponseProfile {
  formula: string;
  tcVsPressure: { pressure: number; tc: number }[];
  stabilityVsPressure: { pressure: number; stable: boolean; enthalpy: number }[];
  bandgapVsPressure: { pressure: number; bandgap: number }[];
  peakTc: number;
  peakTcPressure: number;
  ambientTc: number;
  lowPressureTc: number;
  updatedAt: number;
}

export interface InterpolationResult {
  pressure: number;
  tc: number;
  bandgap: number;
  enthalpy: number;
  enthalpyStable: boolean;
  interpolated: boolean;
}

export interface PressurePropertyMapStats {
  totalProfiles: number;
  avgPeakTc: number;
  avgPeakPressure: number;
  ambientHighTcCount: number;
  lowPressureHighTcCount: number;
  profilesByPeakRange: { range: string; count: number }[];
  recentProfiles: { formula: string; peakTc: number; peakPressure: number }[];
}

const profileCache = new Map<string, PressureResponseProfile>();
const MAX_PROFILES = 500;

function evictIfNeeded(): void {
  if (profileCache.size >= MAX_PROFILES) {
    const oldestKey = profileCache.keys().next().value;
    if (oldestKey) profileCache.delete(oldestKey);
  }
}

export function buildPressureResponseProfile(formula: string): PressureResponseProfile {
  const existing = profileCache.get(formula);
  if (existing && Date.now() - existing.updatedAt < 20 * 60 * 1000) {
    profileCache.delete(formula);
    profileCache.set(formula, existing);
    return existing;
  }

  const curve: PressureCurve = predictPressureCurve(formula);
  const points: PressurePoint[] = curve.points;

  const tcVsPressure: { pressure: number; tc: number }[] = [];
  const stabilityVsPressure: { pressure: number; stable: boolean; enthalpy: number }[] = [];
  const bandgapVsPressure: { pressure: number; bandgap: number }[] = [];

  for (const pt of points) {
    tcVsPressure.push({ pressure: pt.pressureGpa, tc: pt.tc });
    stabilityVsPressure.push({
      pressure: pt.pressureGpa,
      stable: pt.enthalpyStable,
      enthalpy: pt.enthalpy,
    });
    bandgapVsPressure.push({ pressure: pt.pressureGpa, bandgap: pt.bandgap });
  }

  let peakTc = 0;
  let peakTcPressure = 0;
  for (const pt of tcVsPressure) {
    if (pt.tc > peakTc) {
      peakTc = pt.tc;
      peakTcPressure = pt.pressure;
    }
  }

  const ambientPt = tcVsPressure.find(pt => pt.pressure === 0);
  const ambientTc = ambientPt ? ambientPt.tc : 0;

  const lowPPts = tcVsPressure.filter(pt => pt.pressure <= 50);
  const lowPressureTc = lowPPts.length > 0
    ? Math.max(...lowPPts.map(p => p.tc))
    : ambientTc;

  const profile: PressureResponseProfile = {
    formula,
    tcVsPressure,
    stabilityVsPressure,
    bandgapVsPressure,
    peakTc,
    peakTcPressure,
    ambientTc,
    lowPressureTc,
    updatedAt: Date.now(),
  };

  evictIfNeeded();
  profileCache.set(formula, profile);
  return profile;
}

export function interpolateAtPressure(formula: string, targetPressure: number): InterpolationResult {
  const profile = buildPressureResponseProfile(formula);
  const tcPts = profile.tcVsPressure;
  const bgPts = profile.bandgapVsPressure;
  const stabPts = profile.stabilityVsPressure;

  if (tcPts.length === 0) {
    return { pressure: targetPressure, tc: 0, bandgap: 0, enthalpy: 0, enthalpyStable: false, interpolated: false };
  }

  const exact = tcPts.find(p => p.pressure === targetPressure);
  if (exact) {
    const bg = bgPts.find(p => p.pressure === targetPressure);
    const st = stabPts.find(p => p.pressure === targetPressure);
    return {
      pressure: targetPressure,
      tc: exact.tc,
      bandgap: bg?.bandgap ?? 0,
      enthalpy: st?.enthalpy ?? 0,
      enthalpyStable: st?.stable ?? false,
      interpolated: false,
    };
  }

  let loIdx = 0;
  let hiIdx = tcPts.length - 1;
  for (let i = 0; i < tcPts.length - 1; i++) {
    if (tcPts[i].pressure <= targetPressure && tcPts[i + 1].pressure >= targetPressure) {
      loIdx = i;
      hiIdx = i + 1;
      break;
    }
  }

  if (targetPressure < tcPts[0].pressure) {
    loIdx = 0;
    hiIdx = Math.min(1, tcPts.length - 1);
  } else if (targetPressure > tcPts[tcPts.length - 1].pressure) {
    loIdx = Math.max(0, tcPts.length - 2);
    hiIdx = tcPts.length - 1;
  }

  const span = tcPts[hiIdx].pressure - tcPts[loIdx].pressure;
  const t = span > 0 ? (targetPressure - tcPts[loIdx].pressure) / span : 0.5;
  const clampT = Math.max(0, Math.min(1, t));

  const loBg = bgPts[loIdx]?.bandgap ?? 0;
  const hiBg = bgPts[hiIdx]?.bandgap ?? 0;
  const loH = stabPts[loIdx]?.enthalpy ?? 0;
  const hiH = stabPts[hiIdx]?.enthalpy ?? 0;

  const interpEnthalpy = loH + (hiH - loH) * clampT;
  const enthalpyStable = interpEnthalpy < 0.025;

  return {
    pressure: targetPressure,
    tc: tcPts[loIdx].tc + (tcPts[hiIdx].tc - tcPts[loIdx].tc) * clampT,
    bandgap: loBg + (hiBg - loBg) * clampT,
    enthalpy: interpEnthalpy,
    enthalpyStable,
    interpolated: true,
  };
}

export function interpolateRange(
  formula: string,
  pressures: number[]
): InterpolationResult[] {
  return pressures.map(p => interpolateAtPressure(formula, p));
}

export function getProfileCount(): number {
  return profileCache.size;
}

export function getAllProfiles(): PressureResponseProfile[] {
  return Array.from(profileCache.values());
}

export function getPressurePropertyMapStats(): PressurePropertyMapStats {
  const profiles = Array.from(profileCache.values());

  if (profiles.length === 0) {
    return {
      totalProfiles: 0,
      avgPeakTc: 0,
      avgPeakPressure: 0,
      ambientHighTcCount: 0,
      lowPressureHighTcCount: 0,
      profilesByPeakRange: [],
      recentProfiles: [],
    };
  }

  const avgPeakTc = profiles.reduce((s, p) => s + p.peakTc, 0) / profiles.length;
  const avgPeakPressure = profiles.reduce((s, p) => s + p.peakTcPressure, 0) / profiles.length;
  const ambientHighTcCount = profiles.filter(p => p.ambientTc > 77).length;
  const lowPressureHighTcCount = profiles.filter(p => p.lowPressureTc > 77).length;

  const ranges = [
    { range: "0-50 GPa", min: 0, max: 50 },
    { range: "50-100 GPa", min: 50, max: 100 },
    { range: "100-200 GPa", min: 100, max: 200 },
    { range: "200-350 GPa", min: 200, max: 350 },
  ];

  const profilesByPeakRange = ranges.map(r => ({
    range: r.range,
    count: profiles.filter(p => p.peakTcPressure >= r.min && p.peakTcPressure < r.max).length,
  }));

  const recentProfiles = profiles
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 15)
    .map(p => ({ formula: p.formula, peakTc: Math.round(p.peakTc * 10) / 10, peakPressure: p.peakTcPressure }));

  return {
    totalProfiles: profiles.length,
    avgPeakTc: Math.round(avgPeakTc * 10) / 10,
    avgPeakPressure: Math.round(avgPeakPressure),
    ambientHighTcCount,
    lowPressureHighTcCount,
    profilesByPeakRange,
    recentProfiles,
  };
}

export function clearPressurePropertyMapCache(): void {
  profileCache.clear();
}
