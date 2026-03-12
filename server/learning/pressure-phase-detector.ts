import { extractFeatures } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { gnnPredictWithUncertainty, type GNNPredictionWithUncertainty } from "./graph-neural-net";
import { computeEnthalpy } from "./enthalpy-stability";

export interface PressureDataPoint {
  pressureGpa: number;
  tc: number;
  formationEnergy: number;
  bandgap: number;
  stability: number;
  lambda: number;
  enthalpy: number;
}

export type TransitionType =
  | "metallization"
  | "structural"
  | "superconducting_onset"
  | "tc_maximum"
  | "bandgap_closing"
  | "bandgap_opening"
  | "stability_sign_change"
  | "enthalpy_discontinuity";

export interface PhaseTransition {
  formula: string;
  pressureStart: number;
  pressureEnd: number;
  type: TransitionType;
  confidence: number;
  magnitude: number;
  details: string;
  detectedAt: number;
}

export interface PhaseTransitionStats {
  totalTransitionsDetected: number;
  formulasAnalyzed: number;
  transitionsByType: Record<TransitionType, number>;
  avgConfidence: number;
  highConfidenceCount: number;
  pressureRangeDistribution: { low: number; mid: number; high: number };
  recentTransitions: PhaseTransition[];
}

const transitionCache = new Map<string, PhaseTransition[]>();
const analysisTimestamps = new Map<string, number>();

const PRESSURE_STEPS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350];

const DTC_DP_THRESHOLD = 0.5;
const BANDGAP_CLOSE_THRESHOLD = 0.1;
const BANDGAP_OPEN_THRESHOLD = 0.3;
const STABILITY_SIGN_THRESHOLD = 0.05;
const TC_ONSET_THRESHOLD = 2.0;

function computePressureCurve(formula: string): PressureDataPoint[] {
  const curve: PressureDataPoint[] = [];

  for (const p of PRESSURE_STEPS) {
    try {
      const matPartial = { pressureGpa: p } as any;
      const features = extractFeatures(formula, matPartial);
      const gb = gbPredict(features, formula);

      let gnnResult: GNNPredictionWithUncertainty | null = null;
      try {
        gnnResult = gnnPredictWithUncertainty(formula, undefined, p);
      } catch {}

      const tc = gnnResult
        ? gb.tcPredicted * 0.4 + gnnResult.tc * 0.6
        : gb.tcPredicted;

      const formationEnergy = gnnResult
        ? gnnResult.formationEnergy
        : features.formationEnergy ?? 0;

      const bandgap = gnnResult
        ? gnnResult.bandgap
        : features.bandGap ?? 0;

      const stability = gnnResult
        ? gnnResult.stabilityProbability
        : features.stability ?? 0.5;

      const lambda = gnnResult
        ? gnnResult.lambda
        : features.electronPhononLambda;

      let enthalpy = formationEnergy;
      try {
        const hPt = computeEnthalpy(formula, p);
        enthalpy = hPt.enthalpy;
      } catch {
        enthalpy = formationEnergy + p * 0.006;
      }

      curve.push({
        pressureGpa: p,
        tc: Math.max(0, tc),
        formationEnergy,
        bandgap: Math.max(0, bandgap),
        stability,
        lambda: Math.max(0, lambda),
        enthalpy,
      });
    } catch {
      continue;
    }
  }

  return curve;
}

function computeDerivative(curve: PressureDataPoint[], field: keyof Omit<PressureDataPoint, "pressureGpa">): { pressure: number; derivative: number }[] {
  const derivatives: { pressure: number; derivative: number }[] = [];
  for (let i = 1; i < curve.length; i++) {
    const dp = curve[i].pressureGpa - curve[i - 1].pressureGpa;
    if (dp <= 0) continue;
    const dVal = (curve[i][field] as number) - (curve[i - 1][field] as number);
    derivatives.push({
      pressure: (curve[i].pressureGpa + curve[i - 1].pressureGpa) / 2,
      derivative: dVal / dp,
    });
  }
  return derivatives;
}

export function classifyTransition(transition: Omit<PhaseTransition, "type">): TransitionType {
  const details = transition.details.toLowerCase();

  if (details.includes("bandgap closing") || details.includes("metallization")) {
    return "metallization";
  }
  if (details.includes("bandgap opening")) {
    return "bandgap_opening";
  }
  if (details.includes("bandgap close")) {
    return "bandgap_closing";
  }
  if (details.includes("tc maximum") || details.includes("peak tc")) {
    return "tc_maximum";
  }
  if (details.includes("superconducting onset") || details.includes("tc onset")) {
    return "superconducting_onset";
  }
  if (details.includes("stability sign") || details.includes("stability change")) {
    return "stability_sign_change";
  }
  if (details.includes("structural") || details.includes("large dtc/dp")) {
    return "structural";
  }

  if (transition.magnitude > 50) return "structural";
  if (transition.magnitude > 10) return "superconducting_onset";
  return "structural";
}

export function detectPhaseTransitions(formula: string): PhaseTransition[] {
  const cached = transitionCache.get(formula);
  const lastAnalysis = analysisTimestamps.get(formula) ?? 0;
  if (cached && Date.now() - lastAnalysis < 30 * 60 * 1000) {
    return cached;
  }

  const curve = computePressureCurve(formula);
  if (curve.length < 3) return [];

  const transitions: PhaseTransition[] = [];
  const now = Date.now();

  const dTcdP = computeDerivative(curve, "tc");
  for (let i = 0; i < dTcdP.length; i++) {
    const d = dTcdP[i];
    const pStart = curve[i].pressureGpa;
    const pEnd = curve[i + 1].pressureGpa;

    if (i < dTcdP.length - 1 && dTcdP[i].derivative > 0 && dTcdP[i + 1].derivative < 0) {
      const peakP = curve[i + 1].pressureGpa;
      const peakTc = curve[i + 1].tc;
      if (peakTc > TC_ONSET_THRESHOLD) {
        transitions.push({
          formula,
          pressureStart: curve[i].pressureGpa,
          pressureEnd: curve[i + 2] ? curve[i + 2].pressureGpa : pEnd,
          type: "tc_maximum",
          confidence: Math.min(1.0, peakTc / 50),
          magnitude: peakTc,
          details: `Tc dome peak: ${peakTc.toFixed(1)} K at ${peakP.toFixed(0)} GPa (dTc/dP sign change: ${dTcdP[i].derivative.toFixed(3)} -> ${dTcdP[i + 1].derivative.toFixed(3)})`,
          detectedAt: now,
        });
      }
    }

    if (Math.abs(d.derivative) > DTC_DP_THRESHOLD) {
      const isLargeDrop = d.derivative < -DTC_DP_THRESHOLD;
      const isLargeRise = d.derivative > DTC_DP_THRESHOLD;

      const magnitude = Math.abs(d.derivative) * (pEnd - pStart);
      const confidence = Math.min(1.0, Math.abs(d.derivative) / (DTC_DP_THRESHOLD * 5));

      const baseTransition = {
        formula,
        pressureStart: pStart,
        pressureEnd: pEnd,
        confidence,
        magnitude,
        detectedAt: now,
      };

      if (isLargeRise) {
        const type = classifyTransition({
          ...baseTransition,
          details: "large dTc/dP rise - possible superconducting onset or structural transition",
        });
        transitions.push({
          ...baseTransition,
          type,
          details: `dTc/dP = ${d.derivative.toFixed(3)} K/GPa at ${d.pressure.toFixed(0)} GPa`,
        });
      } else if (isLargeDrop) {
        const type = classifyTransition({
          ...baseTransition,
          details: "large dTc/dP drop - possible structural transition",
        });
        transitions.push({
          ...baseTransition,
          type,
          details: `dTc/dP = ${d.derivative.toFixed(3)} K/GPa at ${d.pressure.toFixed(0)} GPa (decline)`,
        });
      }
    }
  }

  for (let i = 1; i < curve.length; i++) {
    const prev = curve[i - 1];
    const curr = curve[i];

    if (prev.bandgap > BANDGAP_OPEN_THRESHOLD && curr.bandgap < BANDGAP_CLOSE_THRESHOLD) {
      const confidence = Math.min(1.0, prev.bandgap / 1.0);
      transitions.push({
        formula,
        pressureStart: prev.pressureGpa,
        pressureEnd: curr.pressureGpa,
        type: "metallization",
        confidence,
        magnitude: prev.bandgap - curr.bandgap,
        details: `Bandgap closing: ${prev.bandgap.toFixed(2)} -> ${curr.bandgap.toFixed(2)} eV (metallization)`,
        detectedAt: now,
      });
    }

    if (prev.bandgap < BANDGAP_CLOSE_THRESHOLD && curr.bandgap > BANDGAP_OPEN_THRESHOLD) {
      const confidence = Math.min(1.0, curr.bandgap / 1.0);
      transitions.push({
        formula,
        pressureStart: prev.pressureGpa,
        pressureEnd: curr.pressureGpa,
        type: "bandgap_opening",
        confidence,
        magnitude: curr.bandgap - prev.bandgap,
        details: `Bandgap opening: ${prev.bandgap.toFixed(2)} -> ${curr.bandgap.toFixed(2)} eV`,
        detectedAt: now,
      });
    }
  }

  for (let i = 1; i < curve.length; i++) {
    const prev = curve[i - 1];
    const curr = curve[i];

    if ((prev.stability - 0.5) * (curr.stability - 0.5) < 0 &&
        Math.abs(prev.stability - curr.stability) > STABILITY_SIGN_THRESHOLD) {
      transitions.push({
        formula,
        pressureStart: prev.pressureGpa,
        pressureEnd: curr.pressureGpa,
        type: "stability_sign_change",
        confidence: Math.min(1.0, Math.abs(prev.stability - curr.stability) * 2),
        magnitude: Math.abs(prev.stability - curr.stability),
        details: `Stability transition: ${prev.stability.toFixed(2)} -> ${curr.stability.toFixed(2)} at ${curr.pressureGpa} GPa`,
        detectedAt: now,
      });
    }
  }

  let tcMaxIdx = 0;
  let tcMax = curve[0]?.tc ?? 0;
  for (let i = 1; i < curve.length; i++) {
    if (curve[i].tc > tcMax) {
      tcMax = curve[i].tc;
      tcMaxIdx = i;
    }
  }

  if (tcMaxIdx > 0 && tcMaxIdx < curve.length - 1) {
    const leftTc = curve[tcMaxIdx - 1].tc;
    const rightTc = curve[tcMaxIdx + 1].tc;
    if (tcMax > leftTc + TC_ONSET_THRESHOLD && tcMax > rightTc + TC_ONSET_THRESHOLD) {
      transitions.push({
        formula,
        pressureStart: curve[tcMaxIdx - 1].pressureGpa,
        pressureEnd: curve[tcMaxIdx + 1].pressureGpa,
        type: "tc_maximum",
        confidence: Math.min(1.0, (tcMax - Math.max(leftTc, rightTc)) / 20),
        magnitude: tcMax,
        details: `Tc maximum: ${tcMax.toFixed(1)} K at ${curve[tcMaxIdx].pressureGpa} GPa`,
        detectedAt: now,
      });
    }
  }

  for (let i = 1; i < curve.length; i++) {
    const prev = curve[i - 1];
    const curr = curve[i];
    if (prev.tc < TC_ONSET_THRESHOLD && curr.tc >= TC_ONSET_THRESHOLD) {
      transitions.push({
        formula,
        pressureStart: prev.pressureGpa,
        pressureEnd: curr.pressureGpa,
        type: "superconducting_onset",
        confidence: Math.min(1.0, curr.tc / 20),
        magnitude: curr.tc - prev.tc,
        details: `Superconducting onset: Tc rises from ${prev.tc.toFixed(1)} to ${curr.tc.toFixed(1)} K at ${curr.pressureGpa} GPa`,
        detectedAt: now,
      });
      break;
    }
  }

  const dHdP = computeDerivative(curve, "enthalpy");
  for (let i = 0; i < dHdP.length; i++) {
    const d = dHdP[i];
    if (Math.abs(d.derivative) > 0.05) {
      transitions.push({
        formula,
        pressureStart: curve[i].pressureGpa,
        pressureEnd: curve[i + 1].pressureGpa,
        type: "enthalpy_discontinuity",
        confidence: Math.min(1.0, Math.abs(d.derivative) / 0.2),
        magnitude: Math.abs(d.derivative) * (curve[i + 1].pressureGpa - curve[i].pressureGpa),
        details: `Enthalpy discontinuity: dH/dP = ${d.derivative.toFixed(4)} eV/GPa at ${d.pressure.toFixed(0)} GPa (H: ${curve[i].enthalpy.toFixed(3)} -> ${curve[i + 1].enthalpy.toFixed(3)} eV)`,
        detectedAt: now,
      });
    }

    if (i > 0) {
      const d2HdP2 = Math.abs(dHdP[i].derivative - dHdP[i - 1].derivative);
      const avgDP = (curve[i + 1].pressureGpa - curve[i - 1].pressureGpa) / 2;
      if (avgDP > 0 && d2HdP2 / avgDP > 0.002) {
        transitions.push({
          formula,
          pressureStart: curve[i - 1].pressureGpa,
          pressureEnd: curve[i + 1].pressureGpa,
          type: "structural",
          confidence: Math.min(1.0, (d2HdP2 / avgDP) / 0.01),
          magnitude: d2HdP2,
          details: `Structural transition (d²H/dP² spike): ${(d2HdP2 / avgDP).toFixed(4)} eV/GPa² at ${d.pressure.toFixed(0)} GPa`,
          detectedAt: now,
        });
      }
    }
  }

  const deduplicated = deduplicateTransitions(transitions);

  transitionCache.set(formula, deduplicated);
  analysisTimestamps.set(formula, now);

  return deduplicated;
}

function deduplicateTransitions(transitions: PhaseTransition[]): PhaseTransition[] {
  if (transitions.length <= 1) return transitions;

  const result: PhaseTransition[] = [];
  const sorted = [...transitions].sort((a, b) => a.pressureStart - b.pressureStart);

  for (const t of sorted) {
    const overlap = result.find(
      r => r.type === t.type &&
        Math.abs(r.pressureStart - t.pressureStart) < 30 &&
        Math.abs(r.pressureEnd - t.pressureEnd) < 30
    );
    if (overlap) {
      if (t.confidence > overlap.confidence) {
        const idx = result.indexOf(overlap);
        result[idx] = t;
      }
    } else {
      result.push(t);
    }
  }

  return result;
}

export function getPhaseTransitionStats(): PhaseTransitionStats {
  const allTransitions: PhaseTransition[] = [];
  transitionCache.forEach((transitions) => {
    allTransitions.push(...transitions);
  });

  const transitionsByType: Record<TransitionType, number> = {
    metallization: 0,
    structural: 0,
    superconducting_onset: 0,
    tc_maximum: 0,
    bandgap_closing: 0,
    bandgap_opening: 0,
    stability_sign_change: 0,
    enthalpy_discontinuity: 0,
  };

  for (const t of allTransitions) {
    transitionsByType[t.type] = (transitionsByType[t.type] || 0) + 1;
  }

  const avgConfidence = allTransitions.length > 0
    ? allTransitions.reduce((s, t) => s + t.confidence, 0) / allTransitions.length
    : 0;

  const highConfidenceCount = allTransitions.filter(t => t.confidence > 0.7).length;

  let low = 0, mid = 0, high = 0;
  for (const t of allTransitions) {
    const midPressure = (t.pressureStart + t.pressureEnd) / 2;
    if (midPressure < 50) low++;
    else if (midPressure < 200) mid++;
    else high++;
  }

  const recentTransitions = [...allTransitions]
    .sort((a, b) => b.detectedAt - a.detectedAt)
    .slice(0, 20);

  return {
    totalTransitionsDetected: allTransitions.length,
    formulasAnalyzed: transitionCache.size,
    transitionsByType,
    avgConfidence: Math.round(avgConfidence * 1000) / 1000,
    highConfidenceCount,
    pressureRangeDistribution: { low, mid, high },
    recentTransitions,
  };
}

export function clearPhaseTransitionCache(): void {
  transitionCache.clear();
  analysisTimestamps.clear();
}
