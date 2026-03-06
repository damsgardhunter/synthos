import { extractFeatures, physicsPredictor } from "./ml-predictor";
import { gbPredict } from "./gradient-boost";
import { normalizeFormula } from "./utils";
import { ELEMENTAL_DATA } from "./elemental-data";
import { computeMiedemaFormationEnergy } from "./phase-diagram-engine";
import type { EventEmitter } from "./engine";

export interface PhaseGridPoint {
  coords: number[];
  formula: string;
  tc: number;
  stable: boolean;
  hullDistance: number;
  uncertainty: number;
}

export interface PhaseHotspot {
  coords: number[];
  formula: string;
  tc: number;
  uncertainty: number;
}

export interface PhaseMap {
  dimensions: string[];
  elementSet: string[];
  grid: PhaseGridPoint[];
  hotspots: PhaseHotspot[];
  scannedAt: string;
}

export interface OptimalRegionResult {
  formula: string;
  optimalPressure: number;
  maxOperatingTemp: number;
  predictedTc: number;
  explorationScore: number;
  hullDistance: number;
}

const COARSE_STEP = 0.1;
const FINE_STEP = 0.02;
const PRESSURE_STEPS_COARSE = [0, 10, 25, 50, 75, 100, 150, 200, 250, 300];
const TEMPERATURE_STEPS = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];


function parseFormulaToElements(formula: string): { elements: string[]; counts: Record<string, number> } {
  const counts: Record<string, number> = {};
  const cleaned = formula.replace(/[₀-₉]/g, (c) => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  return { elements: Object.keys(counts), counts };
}

function buildFormula(counts: Record<string, number>): string {
  const parts: string[] = [];
  const sorted = Object.entries(counts)
    .filter(([, c]) => c > 0)
    .sort(([a], [b]) => {
      const enA = ELEMENTAL_DATA[a]?.paulingElectronegativity ?? 2.0;
      const enB = ELEMENTAL_DATA[b]?.paulingElectronegativity ?? 2.0;
      return enA - enB;
    });
  for (const [el, c] of sorted) {
    const rounded = Math.round(c);
    if (rounded <= 0) continue;
    parts.push(rounded === 1 ? el : `${el}${rounded}`);
  }
  return parts.join("");
}

function fastTcPredict(formula: string): { tc: number; uncertainty: number; hullDistance: number; lambda: number } {
  try {
    const features = extractFeatures(formula);
    if (!features) return { tc: 0, uncertainty: 50, hullDistance: 1.0, lambda: 0 };

    const physicsPred = physicsPredictor.predict(features);
    const gbResult = gbPredict(features);

    const tc = Math.max(0, gbResult.tcPredicted);
    const hullDist = physicsPred.hullDistance;
    const avgUncertainty = (physicsPred.lambdaUncertainty + physicsPred.dosUncertainty +
      physicsPred.omegaUncertainty + physicsPred.hullUncertainty) / 4;

    return { tc, uncertainty: avgUncertainty, hullDistance: hullDist, lambda: physicsPred.lambda };
  } catch {
    return { tc: 0, uncertainty: 50, hullDistance: 1.0, lambda: 0 };
  }
}

function pressureAdjustTc(baseTc: number, lambda: number, pressureGpa: number, formula: string): number {
  if (pressureGpa <= 0) return baseTc;

  const { elements } = parseFormulaToElements(formula);
  const hasHydrogen = elements.includes("H");

  let pressureFactor = 1.0;

  if (hasHydrogen) {
    if (pressureGpa < 100) {
      pressureFactor = 1.0 + pressureGpa * 0.005;
    } else if (pressureGpa < 200) {
      pressureFactor = 1.5 + (pressureGpa - 100) * 0.008;
    } else {
      pressureFactor = 2.3 + (pressureGpa - 200) * 0.004;
    }
    pressureFactor = Math.min(pressureFactor, 3.5);
  } else {
    if (pressureGpa < 50) {
      pressureFactor = 1.0 + pressureGpa * 0.003;
    } else if (pressureGpa < 150) {
      pressureFactor = 1.15 + (pressureGpa - 50) * 0.002;
    } else {
      pressureFactor = 1.35 + (pressureGpa - 150) * 0.001;
    }
    pressureFactor = Math.min(pressureFactor, 1.6);
  }

  const lambdaBoost = lambda > 1.5 ? 1.0 + (lambda - 1.5) * 0.1 : 1.0;
  return Math.round(baseTc * pressureFactor * lambdaBoost);
}

function estimateDecompositionTemp(formula: string, formationEnergy: number): number {
  const { elements } = parseFormulaToElements(formula);
  const meltingPoints = elements
    .map(el => ELEMENTAL_DATA[el]?.meltingPoint ?? 1000)
    .filter(mp => mp > 0);
  const minMelt = Math.min(...meltingPoints);

  const stabFactor = formationEnergy < -1.0 ? 0.8 : formationEnergy < -0.3 ? 0.6 : formationEnergy < 0 ? 0.4 : 0.2;
  return Math.round(minMelt * stabFactor);
}

export function exploreCompositionSpace(
  elementSet: string[],
  resolution: "coarse" | "fine" = "coarse"
): PhaseMap {
  const step = resolution === "coarse" ? COARSE_STEP : FINE_STEP;
  const grid: PhaseGridPoint[] = [];
  const n = elementSet.length;

  if (n === 2) {
    for (let x = step; x < 1.0; x += step) {
      const totalAtoms = 4;
      const c1 = Math.round(x * totalAtoms);
      const c2 = totalAtoms - c1;
      if (c1 <= 0 || c2 <= 0) continue;

      const counts: Record<string, number> = {};
      counts[elementSet[0]] = c1;
      counts[elementSet[1]] = c2;
      const formula = buildFormula(counts);

      const pred = fastTcPredict(formula);
      grid.push({
        coords: [x],
        formula,
        tc: pred.tc,
        stable: pred.hullDistance < 0.1,
        hullDistance: pred.hullDistance,
        uncertainty: pred.uncertainty,
      });
    }
  } else if (n === 3) {
    for (let x = step; x < 1.0; x += step) {
      for (let y = step; y < 1.0 - x; y += step) {
        const z = 1.0 - x - y;
        if (z < step / 2) continue;

        const totalAtoms = 6;
        const c1 = Math.max(1, Math.round(x * totalAtoms));
        const c2 = Math.max(1, Math.round(y * totalAtoms));
        const c3 = Math.max(1, totalAtoms - c1 - c2);
        if (c3 <= 0) continue;

        const counts: Record<string, number> = {};
        counts[elementSet[0]] = c1;
        counts[elementSet[1]] = c2;
        counts[elementSet[2]] = c3;
        const formula = buildFormula(counts);

        const pred = fastTcPredict(formula);
        grid.push({
          coords: [x, y],
          formula,
          tc: pred.tc,
          stable: pred.hullDistance < 0.1,
          hullDistance: pred.hullDistance,
          uncertainty: pred.uncertainty,
        });
      }
    }
  } else if (n === 4) {
    const bigStep = Math.max(step, 0.15);
    for (let a = bigStep; a < 0.7; a += bigStep) {
      for (let b = bigStep; b < 0.7 - a; b += bigStep) {
        for (let c = bigStep; c < 0.7 - a - b; c += bigStep) {
          const d = 1.0 - a - b - c;
          if (d < bigStep / 2) continue;

          const totalAtoms = 8;
          const c1 = Math.max(1, Math.round(a * totalAtoms));
          const c2 = Math.max(1, Math.round(b * totalAtoms));
          const c3 = Math.max(1, Math.round(c * totalAtoms));
          const c4 = Math.max(1, totalAtoms - c1 - c2 - c3);
          if (c4 <= 0) continue;

          const counts: Record<string, number> = {};
          counts[elementSet[0]] = c1;
          counts[elementSet[1]] = c2;
          counts[elementSet[2]] = c3;
          counts[elementSet[3]] = c4;
          const formula = buildFormula(counts);

          const pred = fastTcPredict(formula);
          grid.push({
            coords: [a, b, c],
            formula,
            tc: pred.tc,
            stable: pred.hullDistance < 0.1,
            hullDistance: pred.hullDistance,
            uncertainty: pred.uncertainty,
          });
        }
      }
    }
  }

  const hotspots = identifyHotspots(grid);

  return {
    dimensions: n === 2 ? [`x(${elementSet[0]})`] :
                n === 3 ? [`x(${elementSet[0]})`, `y(${elementSet[1]})`] :
                [`x(${elementSet[0]})`, `y(${elementSet[1]})`, `z(${elementSet[2]})`],
    elementSet,
    grid,
    hotspots,
    scannedAt: new Date().toISOString(),
  };
}

function identifyHotspots(grid: PhaseGridPoint[]): PhaseHotspot[] {
  if (grid.length === 0) return [];

  const sorted = [...grid].sort((a, b) => {
    const scoreA = a.tc + Math.sqrt(a.uncertainty) * 10;
    const scoreB = b.tc + Math.sqrt(b.uncertainty) * 10;
    return scoreB - scoreA;
  });

  const hotspots: PhaseHotspot[] = [];
  const usedFormulas = new Set<string>();

  for (const point of sorted) {
    if (hotspots.length >= 10) break;
    if (point.tc < 5 && point.uncertainty < 20) continue;
    const norm = normalizeFormula(point.formula);
    if (usedFormulas.has(norm)) continue;
    usedFormulas.add(norm);

    hotspots.push({
      coords: point.coords,
      formula: point.formula,
      tc: point.tc,
      uncertainty: point.uncertainty,
    });
  }

  return hotspots;
}

export function explorePressureCompositionSpace(
  elementSet: string[],
  pressureRange: number[] = PRESSURE_STEPS_COARSE
): { map: PhaseMap; optimalPairs: { formula: string; pressure: number; tc: number }[] } {
  const compMap = exploreCompositionSpace(elementSet, "coarse");

  const pressureGrid: PhaseGridPoint[] = [];

  const formulasToScan = compMap.hotspots.length > 0
    ? compMap.hotspots.map(h => h.formula)
    : compMap.grid.sort((a, b) => b.tc - a.tc).slice(0, 10).map(g => g.formula);

  for (const formula of formulasToScan) {
    const basePred = fastTcPredict(formula);

    for (const pressure of pressureRange) {
      const adjustedTc = pressureAdjustTc(basePred.tc, basePred.lambda, pressure, formula);

      pressureGrid.push({
        coords: [formulasToScan.indexOf(formula), pressure],
        formula,
        tc: adjustedTc,
        stable: basePred.hullDistance < 0.15,
        hullDistance: basePred.hullDistance,
        uncertainty: basePred.uncertainty * (1 + pressure / 500),
      });
    }
  }

  const optimalPairs: { formula: string; pressure: number; tc: number }[] = [];
  const byFormula = new Map<string, PhaseGridPoint[]>();
  for (const pt of pressureGrid) {
    const arr = byFormula.get(pt.formula) || [];
    arr.push(pt);
    byFormula.set(pt.formula, arr);
  }
  for (const [formula, points] of byFormula) {
    const best = points.reduce((a, b) => a.tc > b.tc ? a : b);
    optimalPairs.push({ formula, pressure: best.coords[1], tc: best.tc });
  }
  optimalPairs.sort((a, b) => b.tc - a.tc);

  return {
    map: {
      dimensions: ["composition_index", "pressure_GPa"],
      elementSet,
      grid: pressureGrid,
      hotspots: optimalPairs.slice(0, 5).map(p => ({
        coords: [formulasToScan.indexOf(p.formula), p.pressure],
        formula: p.formula,
        tc: p.tc,
        uncertainty: 0,
      })),
      scannedAt: new Date().toISOString(),
    },
    optimalPairs: optimalPairs.slice(0, 10),
  };
}

export function exploreTemperatureStability(
  formula: string,
  tempRange: number[] = TEMPERATURE_STEPS
): { maxOperatingTemp: number; stabilityProfile: { temp: number; decompositionRisk: number; phononStable: boolean }[] } {
  const { elements, counts } = parseFormulaToElements(formula);
  const totalAtoms = Object.values(counts).reduce((s, c) => s + c, 0);

  let formationEnergy = 0;
  try {
    formationEnergy = computeMiedemaFormationEnergy(formula);
  } catch {
    formationEnergy = -0.5;
  }

  const decompositionTemp = estimateDecompositionTemp(formula, formationEnergy);

  const avgMass = elements.reduce((s, el) => s + (ELEMENTAL_DATA[el]?.atomicMass ?? 50), 0) / elements.length;
  const debyeTemps = elements
    .map(el => ELEMENTAL_DATA[el]?.debyeTemperature ?? 300)
    .filter(d => d > 0);
  const avgDebye = debyeTemps.length > 0 ? debyeTemps.reduce((s, d) => s + d, 0) / debyeTemps.length : 300;

  const stabilityProfile: { temp: number; decompositionRisk: number; phononStable: boolean }[] = [];
  let maxOperatingTemp = 0;

  for (const temp of tempRange) {
    const thermalFraction = temp / decompositionTemp;
    const decompositionRisk = thermalFraction > 1.0 ? 1.0 :
      thermalFraction > 0.7 ? 0.3 + (thermalFraction - 0.7) * (0.7 / 0.3) :
      thermalFraction * 0.3 / 0.7;

    const phononStable = temp < avgDebye * 0.8;

    const entropyPenalty = temp > 0 ? 0.0001 * temp * Math.log(elements.length) : 0;
    const effectiveStability = formationEnergy + entropyPenalty;
    const thermodynamicallyStable = effectiveStability < 0;

    stabilityProfile.push({ temp, decompositionRisk, phononStable });

    if (decompositionRisk < 0.5 && phononStable && thermodynamicallyStable) {
      maxOperatingTemp = temp;
    }
  }

  return { maxOperatingTemp, stabilityProfile };
}

export function findOptimalRegion(
  elementSet: string[],
  emit?: EventEmitter
): OptimalRegionResult[] {
  const compMap = exploreCompositionSpace(elementSet, "coarse");

  const topByExploration = [...compMap.grid]
    .map(pt => ({
      ...pt,
      explorationScore: pt.tc + Math.sqrt(pt.uncertainty) * 15,
    }))
    .sort((a, b) => b.explorationScore - a.explorationScore)
    .slice(0, 15);

  const uniqueFormulas = [...new Set(topByExploration.map(t => normalizeFormula(t.formula)))];

  const results: OptimalRegionResult[] = [];

  for (const formula of uniqueFormulas.slice(0, 10)) {
    const basePred = fastTcPredict(formula);

    let bestTc = basePred.tc;
    let bestPressure = 0;

    for (const pressure of PRESSURE_STEPS_COARSE) {
      const adjustedTc = pressureAdjustTc(basePred.tc, basePred.lambda, pressure, formula);
      if (adjustedTc > bestTc) {
        bestTc = adjustedTc;
        bestPressure = pressure;
      }
    }

    if (bestPressure > 0) {
      const refinePressures: number[] = [];
      const pStep = bestPressure > 50 ? 10 : 5;
      for (let p = Math.max(0, bestPressure - 30); p <= bestPressure + 30; p += pStep) {
        refinePressures.push(p);
      }
      for (const p of refinePressures) {
        const tc = pressureAdjustTc(basePred.tc, basePred.lambda, p, formula);
        if (tc > bestTc) {
          bestTc = tc;
          bestPressure = p;
        }
      }
    }

    const { maxOperatingTemp } = exploreTemperatureStability(formula);

    const explorationScore = bestTc + Math.sqrt(basePred.uncertainty) * 15;

    results.push({
      formula,
      optimalPressure: bestPressure,
      maxOperatingTemp,
      predictedTc: bestTc,
      explorationScore,
      hullDistance: basePred.hullDistance,
    });
  }

  results.sort((a, b) => b.explorationScore - a.explorationScore);

  if (emit && results.length > 0) {
    const hotspotSummary = results.slice(0, 3)
      .map(r => `${r.formula} Tc=${r.predictedTc}K@${r.optimalPressure}GPa`)
      .join(", ");
    emit("log", {
      phase: "engine",
      event: `Phase exploration: scanned ${compMap.grid.length} compositions x ${PRESSURE_STEPS_COARSE.length} pressures for {${elementSet.join(",")}}. Hot spots: [${hotspotSummary}]`,
      detail: `Adaptive multi-dimensional scan of ${elementSet.join("-")} system. ${results.length} viable regions found. Best: ${results[0].formula} (Tc=${results[0].predictedTc}K, P=${results[0].optimalPressure}GPa, maxT=${results[0].maxOperatingTemp}K, hull=${results[0].hullDistance.toFixed(3)} eV/atom). Exploration scores include uncertainty bonus.`,
      dataSource: "Phase Explorer",
    });
  }

  return results.slice(0, 10);
}

export function getPhaseExplorationSeedFormulas(elementSet: string[]): string[] {
  const results = findOptimalRegion(elementSet);
  return results
    .filter(r => r.predictedTc > 10 && r.hullDistance < 0.3)
    .map(r => r.formula);
}
