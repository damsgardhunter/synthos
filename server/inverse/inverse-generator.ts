import { TargetProperties, CompositionBias, InverseCandidate } from "./target-schema";
import { defaultSynthesisVector, mutateSynthesisVector, optimizeSynthesisPath } from "../physics/synthesis-simulator";
import { isValidFormula } from "../learning/utils";
import { passesElementCountCap } from "../learning/candidate-generator";

const LIGHT_PHONON_ELEMENTS = ["H", "B", "C", "N", "O"];
const HIGH_COUPLING_TM = ["Nb", "V", "Ti", "Ta", "Mo", "W", "Zr", "Hf"];
const COVALENT_NETWORK = ["B", "C", "N", "Si", "Ge"];
const HIGH_DOS_METALS = ["Nb", "V", "Ta", "Mo", "Mn", "Fe", "Co", "Ni", "Cu"];
const RARE_EARTH = ["La", "Ce", "Y", "Gd", "Nd", "Pr", "Sm", "Eu"];
const ALKALINE_EARTH = ["Ca", "Sr", "Ba", "Mg"];
const PNICTOGEN = ["As", "P", "Sb", "Bi"];
const CHALCOGEN = ["S", "Se", "Te"];

const PROTOTYPE_TC_AFFINITY: Record<string, { minTc: number; maxTc: number; preferredElements: string[][] }> = {
  A15: { minTc: 10, maxTc: 150, preferredElements: [HIGH_COUPLING_TM, COVALENT_NETWORK] },
  AlB2: { minTc: 20, maxTc: 120, preferredElements: [["Mg", "Ca", "Ti", "Zr"], ["B"]] },
  Perovskite: { minTc: 20, maxTc: 200, preferredElements: [RARE_EARTH.concat(ALKALINE_EARTH), HIGH_COUPLING_TM, ["O", "N", "F"]] },
  ThCr2Si2: { minTc: 15, maxTc: 100, preferredElements: [ALKALINE_EARTH.concat(RARE_EARTH), ["Fe", "Co", "Ni", "Ru"], PNICTOGEN.concat(CHALCOGEN)] },
  Heusler: { minTc: 5, maxTc: 80, preferredElements: [["Li", "Cu", "Ni"], ["Mn", "Ti", "V"], ["Al", "Ga", "Sn", "Sb"]] },
  Layered: { minTc: 30, maxTc: 250, preferredElements: [["Cu", "Ni", "Fe"], RARE_EARTH.concat(ALKALINE_EARTH), ["O", "Se", "Te"]] },
  Kagome: { minTc: 5, maxTc: 50, preferredElements: [["K", "Cs", "Rb"], ["V", "Ti", "Mn"], ["Sb", "Sn", "Bi"]] },
  NaCl: { minTc: 10, maxTc: 100, preferredElements: [HIGH_COUPLING_TM, ["N", "C", "O"]] },
  BCC: { minTc: 5, maxTc: 60, preferredElements: [["Nb", "V", "Ta", "Mo", "W"]] },
  FCC: { minTc: 5, maxTc: 40, preferredElements: [["Pb", "Sn", "In", "Al"]] },
  Clathrate: { minTc: 100, maxTc: 400, preferredElements: [RARE_EARTH, ["H"]] },
  Fluorite: { minTc: 5, maxTc: 60, preferredElements: [RARE_EARTH, ["H", "F"]] },
  Skutterudite: { minTc: 3, maxTc: 40, preferredElements: [["Co", "Rh", "Ir", "Fe"], ["Sb", "As", "P"]] },
  Chevrel: { minTc: 5, maxTc: 30, preferredElements: [["Mo"], ["S", "Se", "Te"]] },
  Borocarbide: { minTc: 5, maxTc: 25, preferredElements: [["Y", "Lu", "Tm", "Er"], ["Ni", "Pd"], ["B", "C"]] },
  Pyrochlore: { minTc: 2, maxTc: 15, preferredElements: [["Cd", "Os", "Tl"], ["Re", "Os"], ["O"]] },
  InfiniteLayer: { minTc: 30, maxTc: 120, preferredElements: [["Nd", "Pr", "La"], ["Ni", "Cu"], ["O"]] },
};

const STOICH_PATTERNS = [
  { name: "AB", slots: 2, ratios: [[1,1],[2,1],[3,1],[1,2],[1,3]], weight: 2.0 },
  { name: "AB2", slots: 2, ratios: [[1,2],[1,3],[1,4],[2,3]], weight: 2.0 },
  { name: "AB3", slots: 2, ratios: [[1,3],[1,4],[2,3],[1,5]], weight: 2.0 },
  { name: "ABH", slots: 3, ratios: [[1,1,4],[1,1,6],[1,1,8],[1,1,10],[1,2,6]], weight: 2.0 },
  { name: "ABC3", slots: 3, ratios: [[1,1,3],[1,1,2],[2,1,3]], weight: 2.0 },
  { name: "AB2C2", slots: 3, ratios: [[1,2,2],[1,2,3],[2,2,5]], weight: 2.0 },
  { name: "A2B3C", slots: 3, ratios: [[2,3,1],[3,2,1],[2,3,2]], weight: 2.0 },
  { name: "A2B2C", slots: 3, ratios: [[2,2,1],[2,2,3],[2,2,5]], weight: 2.0 },
  { name: "A3BC", slots: 3, ratios: [[3,1,1],[3,1,3],[3,2,1]], weight: 1.5 },
  { name: "A4B3C", slots: 3, ratios: [[4,3,1],[4,3,2],[4,3,3]], weight: 1.0 },
  { name: "AB3C4", slots: 3, ratios: [[1,3,4],[1,3,3],[2,3,4]], weight: 1.5 },
  { name: "ABCD", slots: 4, ratios: [[1,1,1,1],[2,1,1,1],[1,2,1,1]], weight: 0.5 },
  { name: "ABCD2", slots: 4, ratios: [[1,1,1,2],[1,1,1,3],[2,1,1,2]], weight: 0.5 },
];

function variantPassesChargeBalance(elements: string[], ratios: number[]): boolean {
  if (elements.length <= 1) return true;
  const allMetallic = elements.every(el => {
    const states = COMMON_OXIDATION_STATES[el];
    if (!states) return true;
    return states.every(s => s >= 0);
  });
  const hasAnion = elements.some(el => {
    const states = COMMON_OXIDATION_STATES[el];
    return states ? states.some(s => s < 0) : false;
  });
  if (allMetallic && !hasAnion) return true;

  let best = Infinity;
  const enumerate = (idx: number, charge: number): void => {
    if (best === 0) return;
    if (idx === elements.length) { best = Math.min(best, Math.abs(charge)); return; }
    const states = COMMON_OXIDATION_STATES[elements[idx]];
    if (!states) { enumerate(idx + 1, charge); return; }
    for (const ox of states) enumerate(idx + 1, charge + ox * ratios[idx]);
  };
  enumerate(0, 0);
  return best <= 1;
}

const COMMON_OXIDATION_STATES: Record<string, number[]> = {
  H: [1, -1], Li: [1], Na: [1], K: [1], Rb: [1], Cs: [1],
  Be: [2], Mg: [2], Ca: [2], Sr: [2], Ba: [2],
  Sc: [3], Y: [3], La: [3], Ce: [3, 4], Gd: [3], Nd: [3], Pr: [3, 4], Sm: [3], Eu: [2, 3],
  Ti: [2, 3, 4], Zr: [4], Hf: [4],
  V: [2, 3, 4, 5], Nb: [3, 4, 5], Ta: [5],
  Cr: [2, 3, 6], Mo: [4, 6], W: [4, 6],
  Mn: [2, 3, 4, 7], Fe: [2, 3], Co: [2, 3], Ni: [2, 3], Cu: [1, 2], Zn: [2],
  Ru: [3, 4], Rh: [3], Pd: [2, 4], Ag: [1], Ir: [3, 4], Pt: [2, 4], Au: [1, 3],
  Al: [3], Ga: [3], In: [3], Sn: [2, 4], Pb: [2, 4],
  B: [3], C: [4, -4], N: [-3, 3, 5], Si: [4, -4], Ge: [4],
  O: [-2], S: [-2, 4, 6], Se: [-2, 4], Te: [-2, 4],
  F: [-1], Cl: [-1], Br: [-1], I: [-1],
  P: [-3, 3, 5], As: [-3, 3, 5], Sb: [-3, 3, 5], Bi: [3, 5],
};

function hasPlausibleChargeBalance(formula: string): boolean {
  const elMap = parseFormulaElements(formula);
  if (elMap.size === 0) return false;
  if (elMap.size === 1) return true;

  const entries = Array.from(elMap.entries());
  const allMetallic = entries.every(([el]) => {
    const states = COMMON_OXIDATION_STATES[el];
    if (!states) return true;
    return states.every(s => s >= 0) || states.some(s => s > 0);
  });
  const hasAnion = entries.some(([el]) => {
    const states = COMMON_OXIDATION_STATES[el];
    return states ? states.some(s => s < 0) : false;
  });
  if (allMetallic && !hasAnion) return true;

  function canBalance(idx: number, runningSum: number): boolean {
    if (idx === entries.length) return runningSum === 0;
    const [el, count] = entries[idx];
    const states = COMMON_OXIDATION_STATES[el];
    if (!states) return canBalance(idx + 1, runningSum);
    for (const ox of states) {
      if (canBalance(idx + 1, runningSum + ox * count)) return true;
    }
    return false;
  }

  return canBalance(0, 0);
}

function parseFormulaElements(formula: string): Map<string, number> {
  const elMap = new Map<string, number>();
  const matches = formula.match(/([A-Z][a-z]?)(\d*)/g) || [];
  for (const m of matches) {
    const elMatch = m.match(/^([A-Z][a-z]?)(\d*)$/);
    if (elMatch) {
      const el = elMatch[1];
      const count = elMatch[2] ? parseInt(elMatch[2]) : 1;
      elMap.set(el, (elMap.get(el) || 0) + count);
    }
  }
  return elMap;
}

function estimateQuickTc(formula: string, prototype: string, target: TargetProperties): number {
  const protoInfo = PROTOTYPE_TC_AFFINITY[prototype];
  if (!protoInfo) return 10;
  const baseTc = (protoInfo.minTc + protoInfo.maxTc) / 2;
  const elMap = parseFormulaElements(formula);
  const elements = Array.from(elMap.keys());
  const hasH = elMap.has("H");
  const hCount = elMap.get("H") || 0;
  let tc = baseTc;
  if (hasH && hCount >= 10) tc *= 1.5;
  else if (hasH && hCount >= 6) tc *= 1.3;
  else if (hasH && hCount >= 3) tc *= 1.1;
  const hasTM = HIGH_COUPLING_TM.some(el => elements.includes(el));
  if (hasTM) tc *= 1.15;
  const hasRE = RARE_EARTH.some(el => elements.includes(el));
  if (hasRE) tc *= 1.1;
  if (prototype === "Clathrate" && hasH) tc *= 1.5;
  if (prototype === "A15" && hasTM) tc *= 1.2;
  if (elements.length >= 3) tc *= 1.05;
  if (elements.length >= 4) tc *= 1.05;
  return Math.min(400, Math.max(1, tc + (Math.random() - 0.5) * 10));
}

function selectPrototypes(target: TargetProperties): string[] {
  if (target.preferredPrototypes && target.preferredPrototypes.length > 0) {
    return target.preferredPrototypes;
  }

  const candidates: { proto: string; score: number }[] = [];
  for (const [proto, info] of Object.entries(PROTOTYPE_TC_AFFINITY)) {
    if (target.targetTc >= info.minTc && target.targetTc <= info.maxTc * 1.5) {
      const tcCenter = (info.minTc + info.maxTc) / 2;
      const tcFit = 1.0 - Math.abs(target.targetTc - tcCenter) / (info.maxTc - info.minTc + 1);
      candidates.push({ proto, score: Math.max(0, tcFit) });
    }
  }

  if (target.targetTc > 200) {
    candidates.push({ proto: "Clathrate", score: 0.9 });
    candidates.push({ proto: "Layered", score: 0.7 });
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates.slice(0, 5).map(c => c.proto);
}

const FORBIDDEN_ELEMENTS = new Set([
  "He", "Ne", "Ar", "Kr", "Xe", "Rn",
  "Po", "At", "Fr", "Ra", "Pm", "Tc",
]);

function selectElements(
  target: TargetProperties,
  prototype: string,
  bias: CompositionBias
): string[][] {
  const protoInfo = PROTOTYPE_TC_AFFINITY[prototype];
  const baseGroups = protoInfo ? protoInfo.preferredElements : [HIGH_COUPLING_TM, LIGHT_PHONON_ELEMENTS];

  const result: string[][] = [];
  for (const group of baseGroups) {
    let filtered = group.filter(el => {
      if (FORBIDDEN_ELEMENTS.has(el)) return false;
      if (target.excludeElements && target.excludeElements.includes(el)) return false;
      return true;
    });

    if (target.preferredElements && target.preferredElements.length > 0) {
      const preferred = filtered.filter(el => target.preferredElements!.includes(el));
      if (preferred.length > 0) filtered = preferred;
    }

    const weighted = filtered.map(el => ({
      el,
      w: (bias.elementWeights.get(el) ?? 1.0) + Math.random() * 0.3,
    }));
    weighted.sort((a, b) => b.w - a.w);
    result.push(weighted.slice(0, Math.min(4, weighted.length)).map(w => w.el));
  }

  return result;
}

function buildFormula(elements: string[], ratios: number[]): string {
  let formula = "";
  for (let i = 0; i < elements.length; i++) {
    const r = Math.min(ratios[i], 12);
    formula += elements[i];
    if (r > 1) formula += r;
  }
  return formula;
}

function generateStoichiometryVariants(
  baseElements: string[],
  baseRatios: number[],
  count: number
): { elements: string[]; ratios: number[] }[] {
  const variants: { elements: string[]; ratios: number[] }[] = [
    { elements: baseElements, ratios: baseRatios },
  ];

  for (let i = 0; i < count - 1 && variants.length < count; i++) {
    const newRatios = [...baseRatios];
    const idx = Math.floor(Math.random() * newRatios.length);
    const delta = Math.random() < 0.5 ? 1 : -1;
    newRatios[idx] = Math.max(1, Math.min(12, newRatios[idx] + delta));

    const key = newRatios.join("-");
    if (!variants.some(v => v.ratios.join("-") === key)) {
      if (variantPassesChargeBalance(baseElements, newRatios)) {
        variants.push({ elements: baseElements, ratios: newRatios });
      }
    }
  }

  return variants;
}

export function generateInverseCandidates(
  target: TargetProperties,
  campaignId: string,
  iteration: number,
  bias: CompositionBias,
  count: number = 40
): InverseCandidate[] {
  const prototypes = selectPrototypes(target);
  const candidates: InverseCandidate[] = [];
  const seen = new Set<string>();

  const candidatesPerProto = Math.ceil(count / Math.max(prototypes.length, 1));

  for (const proto of prototypes) {
    const elementGroups = selectElements(target, proto, bias);
    const protoWeight = bias.prototypeWeights.get(proto) ?? 1.0;

    const nSlots = elementGroups.length;
    const matchingPatterns = STOICH_PATTERNS.filter(p => p.slots === nSlots || p.slots <= nSlots);

    const weightedPatterns = matchingPatterns.map(p => {
      const biasEntry = bias.stoichiometryPatterns.find(sp => p.name.startsWith(sp.pattern) || sp.pattern === p.name);
      return { pattern: p, weight: (biasEntry?.weight ?? 1.0) * (p.weight ?? 1.0) };
    });

    for (let attempt = 0; attempt < candidatesPerProto * 4 && candidates.length < count; attempt++) {
      const totalPW = weightedPatterns.reduce((s, w) => s + w.weight, 0);
      let pR = Math.random() * totalPW;
      let selectedPattern = weightedPatterns[0];
      for (const wp of weightedPatterns) {
        pR -= wp.weight;
        if (pR <= 0) { selectedPattern = wp; break; }
      }
      const pattern = selectedPattern.pattern;
      if (!pattern) continue;

      const chosenElements: string[] = [];
      if (elementGroups.length === 0) continue;
      for (let s = 0; s < pattern.slots; s++) {
        const groupIdx = s % elementGroups.length;
        const group = elementGroups[groupIdx];
        if (!group || group.length === 0) continue;

        const elWeights = group.map(el => ({
          el,
          w: (bias.elementWeights.get(el) ?? 1.0) * protoWeight + Math.random() * 0.2,
        }));
        if (elWeights.length === 0) continue;
        const totalW = elWeights.reduce((s, e) => s + e.w, 0);
        let r = Math.random() * totalW;
        let chosen = elWeights[0].el;
        for (const ew of elWeights) {
          r -= ew.w;
          if (r <= 0) { chosen = ew.el; break; }
        }
        chosenElements.push(chosen);
      }

      if (new Set(chosenElements).size < chosenElements.length) continue;

      const ratioSet = pattern.ratios[Math.floor(Math.random() * pattern.ratios.length)];
      const variants = generateStoichiometryVariants(chosenElements, ratioSet, 3);

      for (const v of variants) {
        const formula = buildFormula(v.elements, v.ratios);
        if (seen.has(formula)) continue;
        seen.add(formula);

        const isHydride = formula.includes("H");
        const matClass = isHydride ? "hydride" : proto === "Perovskite" ? "cuprate" : "default";
        const sv = defaultSynthesisVector(matClass);
        const synthVec = Math.random() < 0.5 ? mutateSynthesisVector(sv) : sv;

        const estimatedTc = estimateQuickTc(formula, proto, target);
        candidates.push({
          formula,
          source: "inverse",
          campaignId,
          targetDistance: Math.max(0, 1 - estimatedTc / Math.max(1, target.targetTc)),
          iteration,
          prototype: proto,
          synthesisVector: {
            temperature: synthVec.temperature,
            pressure: synthVec.pressure,
            coolingRate: synthVec.coolingRate,
            annealTime: synthVec.annealTime,
            strain: synthVec.strain,
          },
        });

        if (candidates.length >= count) break;
      }
    }
  }

  if (target.targetTc > 200 && (target.maxPressure ?? 300) >= 50) {
    const hydrideMetals = target.preferredElements?.filter(el =>
      HIGH_COUPLING_TM.includes(el) || RARE_EARTH.includes(el)
    ) ?? ["La", "Y", "Ca", "Nb"];

    for (const metal of hydrideMetals.slice(0, 4)) {
      for (const hCount of [6, 8, 10, 12]) {
        const formula = `${metal}H${hCount}`;
        if (!seen.has(formula)) {
          seen.add(formula);
          const estTc = estimateQuickTc(formula, "Clathrate", target);
          candidates.push({
            formula,
            source: "inverse",
            campaignId,
            targetDistance: Math.max(0, 1 - estTc / Math.max(1, target.targetTc)),
            iteration,
            prototype: "Clathrate",
          });
        }
      }
    }
  }

  return candidates.filter(c => {
    if (!isValidFormula(c.formula)) return false;
    if (!hasPlausibleChargeBalance(c.formula)) return false;
    if (!passesElementCountCap(c.formula)) return false;
    return true;
  }).slice(0, count);
}

export function refineCandidate(
  base: InverseCandidate,
  target: TargetProperties,
  bias: CompositionBias
): InverseCandidate[] {
  const formula = base.formula;
  const elements: string[] = [];
  const counts: number[] = [];

  const matches = formula.match(/([A-Z][a-z]?)(\d*)/g) || [];
  for (const m of matches) {
    const elMatch = m.match(/^([A-Z][a-z]?)(\d*)$/);
    if (elMatch) {
      elements.push(elMatch[1]);
      counts.push(parseInt(elMatch[2] || "1"));
    }
  }

  const refined: InverseCandidate[] = [];
  const seen = new Set<string>([formula]);

  for (let i = 0; i < elements.length; i++) {
    for (const delta of [-1, 1, 2]) {
      const newCounts = [...counts];
      newCounts[i] = Math.max(1, newCounts[i] + delta);
      const newFormula = buildFormula(elements, newCounts);
      if (!seen.has(newFormula)) {
        seen.add(newFormula);
        const baseSynth = base.synthesisVector;
        const mutSv = baseSynth
          ? mutateSynthesisVector({ ...defaultSynthesisVector(), temperature: baseSynth.temperature, pressure: baseSynth.pressure, coolingRate: baseSynth.coolingRate, annealTime: baseSynth.annealTime, strain: baseSynth.strain, currentDensity: 0, magneticField: 0, thermalCycles: 1, oxygenPressure: 0 })
          : mutateSynthesisVector(defaultSynthesisVector());
        refined.push({
          ...base,
          formula: newFormula,
          targetDistance: 1.0,
          iteration: base.iteration + 1,
          synthesisVector: {
            temperature: mutSv.temperature,
            pressure: mutSv.pressure,
            coolingRate: mutSv.coolingRate,
            annealTime: mutSv.annealTime,
            strain: mutSv.strain,
          },
        });
      }
    }
  }

  const substitutions: Record<string, string[]> = {
    "La": ["Y", "Ce", "Gd"], "Y": ["La", "Sc", "Gd"],
    "Nb": ["V", "Ta", "Mo"], "V": ["Nb", "Ti", "Cr"],
    "Fe": ["Co", "Ni", "Mn"], "Cu": ["Ag", "Au"],
    "B": ["C", "N"], "C": ["B", "N", "Si"],
    "O": ["N", "F", "S"], "Se": ["Te", "S"],
    "As": ["P", "Sb"], "Ca": ["Sr", "Ba"],
    "Mg": ["Al"], "Al": ["Mg", "Ga"],
    "Li": ["Mg", "Na"], "Ba": ["Ca", "Sr"],
    "Sr": ["Ca", "Ba"], "Sc": ["Y", "La"],
    "Hf": ["Zr", "Ti"], "Zr": ["Hf", "Ti"],
    "Ta": ["Nb", "V"], "Mo": ["W", "Cr"],
    "W": ["Mo", "Cr"],
  };

  for (let i = 0; i < elements.length; i++) {
    const subs = substitutions[elements[i]];
    if (!subs) continue;
    for (const sub of subs) {
      if (target.excludeElements?.includes(sub)) continue;
      if (elements.includes(sub)) continue;
      const newElements = [...elements];
      newElements[i] = sub;
      const newFormula = buildFormula(newElements, counts);
      if (!seen.has(newFormula)) {
        seen.add(newFormula);
        refined.push({
          ...base,
          formula: newFormula,
          targetDistance: 1.0,
          iteration: base.iteration + 1,
        });
      }
    }
  }

  return refined.filter(c => hasPlausibleChargeBalance(c.formula) && passesElementCountCap(c.formula));
}

export function createInitialBias(target: TargetProperties): CompositionBias {
  const elementWeights = new Map<string, number>();
  const prototypeWeights = new Map<string, number>();

  if (target.targetTc > 200) {
    for (const el of ["H", "B"]) elementWeights.set(el, 2.0);
    for (const el of RARE_EARTH) elementWeights.set(el, 1.5);
    prototypeWeights.set("Clathrate", 2.0);
    prototypeWeights.set("Layered", 1.5);
  } else if (target.targetTc > 50) {
    for (const el of HIGH_COUPLING_TM) elementWeights.set(el, 1.5);
    for (const el of COVALENT_NETWORK) elementWeights.set(el, 1.3);
    prototypeWeights.set("A15", 1.5);
    prototypeWeights.set("Perovskite", 1.3);
    prototypeWeights.set("ThCr2Si2", 1.3);
  } else {
    for (const el of HIGH_COUPLING_TM) elementWeights.set(el, 1.2);
    prototypeWeights.set("BCC", 1.3);
    prototypeWeights.set("NaCl", 1.2);
  }

  if (target.maxPressure < 10) {
    elementWeights.set("H", (elementWeights.get("H") ?? 1.0) * 0.5);
    prototypeWeights.set("Clathrate", (prototypeWeights.get("Clathrate") ?? 1.0) * 0.3);
  }

  if (target.preferredElements) {
    for (const el of target.preferredElements) {
      elementWeights.set(el, (elementWeights.get(el) ?? 1.0) + 1.0);
    }
  }

  const stoichiometryPatterns = [
    { pattern: "AB", weight: 1.0 },
    { pattern: "AB2", weight: 1.0 },
    { pattern: "AB3", weight: 0.8 },
    { pattern: "ABC3", weight: 1.2 },
    { pattern: "AB2C2", weight: 1.3 },
    { pattern: "A2B2C", weight: 1.1 },
    { pattern: "A3BC", weight: 1.0 },
    { pattern: "ABH", weight: target.targetTc > 200 ? 2.0 : 0.5 },
    { pattern: "ABCD2", weight: 1.0 },
  ];

  return { elementWeights, prototypeWeights, stoichiometryPatterns };
}
