import { predictPressureCurve, type PressureCurve, type PressurePoint } from "./pressure-aware-surrogate";
import { computeEnthalpyStability } from "./enthalpy-stability";
import { normalizeFormula } from "./utils";

export interface PressureResponseProfile {
  formula: string;
  tcVsPressure: { pressure: number; tc: number }[];
  stabilityVsPressure: { pressure: number; stable: boolean; enthalpy: number }[];
  bandgapVsPressure: { pressure: number; bandgap: number }[];
  absolutePeakTc: number;
  absolutePeakTcPressure: number;
  stablePeakTc: number;
  stablePeakTcPressure: number;
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
  units: { tc: "K"; bandgap: "eV"; enthalpy: "eV/atom"; pressure: "GPa" };
}

export interface PressurePropertyMapStats {
  totalProfiles: number;
  avgPeakTc: number;
  avgPeakPressure: number;
  ambientHighTcCount: number;
  lowPressureHighTcCount: number;
  profilesByPeakRange: { range: string; count: number }[];
  recentProfiles: { formula: string; absolutePeakTc: number; stablePeakTc: number; peakPressure: number }[];
}

const INTERP_UNITS = { tc: "K" as const, bandgap: "eV" as const, enthalpy: "eV/atom" as const, pressure: "GPa" as const };

const profileCache = new Map<string, PressureResponseProfile>();
const MAX_PROFILES = 500;

function evictIfNeeded(): void {
  if (profileCache.size >= MAX_PROFILES) {
    const oldestKey = profileCache.keys().next().value;
    if (oldestKey) profileCache.delete(oldestKey);
  }
}

export async function buildPressureResponseProfile(rawFormula: string): Promise<PressureResponseProfile> {
  const formula = normalizeFormula(rawFormula);
  const existing = profileCache.get(formula);
  if (existing && Date.now() - existing.updatedAt < 20 * 60 * 1000) {
    profileCache.delete(formula);
    profileCache.set(formula, existing);
    return existing;
  }

  const curve: PressureCurve = await predictPressureCurve(formula);
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

  let absolutePeakTc = 0;
  let absolutePeakTcPressure = 0;
  let stablePeakTc = 0;
  let stablePeakTcPressure = 0;
  for (let i = 0; i < tcVsPressure.length; i++) {
    const tc = tcVsPressure[i].tc;
    const p = tcVsPressure[i].pressure;
    if (tc > absolutePeakTc) {
      absolutePeakTc = tc;
      absolutePeakTcPressure = p;
    }
    if (stabilityVsPressure[i]?.stable && tc > stablePeakTc) {
      stablePeakTc = tc;
      stablePeakTcPressure = p;
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
    absolutePeakTc,
    absolutePeakTcPressure,
    stablePeakTc,
    stablePeakTcPressure,
    ambientTc,
    lowPressureTc,
    updatedAt: Date.now(),
  };

  evictIfNeeded();
  profileCache.set(formula, profile);
  return profile;
}

export async function interpolateAtPressure(rawFormula: string, targetPressure: number): Promise<InterpolationResult> {
  const profile = await buildPressureResponseProfile(rawFormula);
  const tcPts = profile.tcVsPressure;
  const bgPts = profile.bandgapVsPressure;
  const stabPts = profile.stabilityVsPressure;

  if (tcPts.length === 0) {
    return { pressure: targetPressure, tc: 0, bandgap: 0, enthalpy: 0, enthalpyStable: false, interpolated: false, units: INTERP_UNITS };
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
      units: INTERP_UNITS,
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
    tc: Math.max(0, tcPts[loIdx].tc + (tcPts[hiIdx].tc - tcPts[loIdx].tc) * clampT),
    bandgap: Math.max(0, loBg + (hiBg - loBg) * clampT),
    enthalpy: interpEnthalpy,
    enthalpyStable,
    interpolated: true,
    units: INTERP_UNITS,
  };
}

export async function interpolateRange(
  rawFormula: string,
  pressures: number[]
): Promise<InterpolationResult[]> {
  const formula = normalizeFormula(rawFormula);
  const results: InterpolationResult[] = [];
  for (const p of pressures) {
    results.push(await interpolateAtPressure(formula, p));
  }
  return results;
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

  const avgPeakTc = profiles.reduce((s, p) => s + p.absolutePeakTc, 0) / profiles.length;
  const avgPeakPressure = profiles.reduce((s, p) => s + p.absolutePeakTcPressure, 0) / profiles.length;
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
    count: profiles.filter(p => p.absolutePeakTcPressure >= r.min && p.absolutePeakTcPressure < r.max).length,
  }));

  const recentProfiles = profiles
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, 15)
    .map(p => ({
      formula: p.formula,
      absolutePeakTc: Math.round(p.absolutePeakTc * 10) / 10,
      stablePeakTc: Math.round(p.stablePeakTc * 10) / 10,
      peakPressure: p.absolutePeakTcPressure,
    }));

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
