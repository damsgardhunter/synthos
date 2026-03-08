import { TargetProperties, CompositionBias, InverseCandidate } from "./target-schema";
import { defaultSynthesisVector, mutateSynthesisVector, optimizeSynthesisPath } from "../physics/synthesis-simulator";

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
};

const STOICH_PATTERNS = [
  { name: "AB", slots: 2, ratios: [[1,1],[2,1],[3,1],[1,2],[1,3]] },
  { name: "AB2", slots: 2, ratios: [[1,2],[1,3],[1,4],[2,3]] },
  { name: "ABH", slots: 3, ratios: [[1,1,4],[1,1,6],[1,1,8],[1,1,10],[1,2,6]] },
  { name: "ABC3", slots: 3, ratios: [[1,1,3],[1,1,2],[2,1,3]] },
  { name: "AB2C2", slots: 3, ratios: [[1,2,2],[1,2,3],[2,2,5]] },
  { name: "A2B3C", slots: 3, ratios: [[2,3,1],[3,2,1],[2,3,2]] },
  { name: "ABCD", slots: 4, ratios: [[1,1,1,1],[2,1,1,1],[1,2,1,1]] },
];

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
  const elements = [...elMap.keys()];
  const hasH = elMap.has("H");
  const hCount = elMap.get("H") || 0;
  let tc = baseTc;
  if (hasH && hCount >= 6) tc *= 1.3 + hCount * 0.05;
  if (hasH && hCount >= 10) tc *= 1.2;
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
    formula += elements[i];
    if (ratios[i] > 1) formula += ratios[i];
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
    newRatios[idx] = Math.max(1, newRatios[idx] + delta);

    const key = newRatios.join("-");
    if (!variants.some(v => v.ratios.join("-") === key)) {
      variants.push({ elements: baseElements, ratios: newRatios });
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
      return { pattern: p, weight: biasEntry?.weight ?? 1.0 };
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
      for (let s = 0; s < pattern.slots; s++) {
        const groupIdx = s % elementGroups.length;
        const group = elementGroups[groupIdx];
        if (group.length === 0) continue;

        const elWeights = group.map(el => ({
          el,
          w: (bias.elementWeights.get(el) ?? 1.0) * protoWeight + Math.random() * 0.2,
        }));
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

  return candidates.slice(0, count);
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

  return refined;
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
    { pattern: "ABH", weight: target.targetTc > 200 ? 2.0 : 0.5 },
  ];

  return { elementWeights, prototypeWeights, stoichiometryPatterns };
}
