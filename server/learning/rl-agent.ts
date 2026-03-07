const ELEMENT_GROUPS = [
  { name: "alkali", elements: ["Li", "Na", "K", "Rb", "Cs"] },
  { name: "alkaline-earth", elements: ["Be", "Mg", "Ca", "Sr", "Ba"] },
  { name: "3d-transition", elements: ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"] },
  { name: "4d-transition", elements: ["Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag"] },
  { name: "5d-transition", elements: ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"] },
  { name: "lanthanide", elements: ["La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy", "Er", "Yb", "Lu"] },
  { name: "p-block-metal", elements: ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"] },
  { name: "metalloid", elements: ["B", "Si", "Ge", "As", "Sb", "Te", "Se"] },
  { name: "nonmetal", elements: ["H", "C", "N", "O", "F", "P", "S", "Cl"] },
] as const;

const STOICH_TEMPLATES = [
  { name: "binary-metal-rich", pattern: "A3B", nElements: 2 },
  { name: "binary-balanced", pattern: "AB", nElements: 2 },
  { name: "binary-anion-rich", pattern: "AB3", nElements: 2 },
  { name: "ternary-122", pattern: "AB2C2", nElements: 3 },
  { name: "ternary-perovskite", pattern: "ABC3", nElements: 3 },
  { name: "ternary-balanced", pattern: "A2BC", nElements: 3 },
  { name: "quaternary", pattern: "ABCD", nElements: 4 },
  { name: "hydride-rich", pattern: "AH10", nElements: 2 },
  { name: "ternary-hydride", pattern: "ABH6", nElements: 3 },
  { name: "boride-carbide", pattern: "A2B3C", nElements: 3 },
] as const;

const STRUCTURE_TYPES = [
  "A15", "AlB2", "NaCl", "Perovskite", "ThCr2Si2",
  "Heusler", "BCC", "FCC", "Layered", "Kagome",
  "HexBoride", "MX2", "Anti-perovskite", "CsCl",
  "Cu2Mg-Laves", "Fluorite", "Cr3Si", "Ni3Sn", "Fe3C", "Spinel",
] as const;

interface RLState {
  bestTc: number;
  avgRecentTc: number;
  recentRewardTrend: number;
  familyDiversity: number;
  stagnationCycles: number;
  explorationBudgetUsed: number;
  elementSuccessEntropy: number;
  cycleNumber: number;
}

interface RLAction {
  elementGroup1: number;
  elementGroup2: number;
  stoichTemplate: number;
  structureType: number;
}

interface Experience {
  state: RLState;
  action: RLAction;
  reward: number;
  timestamp: number;
}

interface ElementPairPrior {
  el1: string;
  el2: string;
  bias: number;
  reason: string;
}

const KNOWN_PAIR_PRIORS: ElementPairPrior[] = [
  { el1: "La", el2: "H", bias: 0.6, reason: "LaH10 high-Tc hydride" },
  { el1: "Y", el2: "H", bias: 0.5, reason: "YH6/YH9 high-Tc hydride" },
  { el1: "Ca", el2: "H", bias: 0.4, reason: "CaH6 superconducting hydride" },
  { el1: "Ba", el2: "H", bias: 0.35, reason: "BaH hydride family" },
  { el1: "Fe", el2: "As", bias: 0.45, reason: "Iron pnictide superconductors" },
  { el1: "Fe", el2: "Se", bias: 0.4, reason: "FeSe superconductor family" },
  { el1: "Cu", el2: "O", bias: 0.5, reason: "Cuprate superconductors" },
  { el1: "Nb", el2: "B", bias: 0.35, reason: "MgB2-type boride superconductors" },
  { el1: "Nb", el2: "N", bias: 0.3, reason: "NbN conventional superconductor" },
  { el1: "Nb", el2: "Ge", bias: 0.3, reason: "Nb3Ge A15 superconductor" },
  { el1: "Nb", el2: "Sn", bias: 0.3, reason: "Nb3Sn A15 superconductor" },
  { el1: "Y", el2: "Ba", bias: 0.45, reason: "YBCO cuprate superconductor" },
  { el1: "Bi", el2: "Se", bias: 0.3, reason: "Topological superconductor candidate" },
  { el1: "Bi", el2: "Sr", bias: 0.35, reason: "BSCCO cuprate superconductor" },
  { el1: "Mg", el2: "B", bias: 0.4, reason: "MgB2 conventional superconductor" },
  { el1: "La", el2: "Cu", bias: 0.35, reason: "LSCO cuprate family" },
  { el1: "Ir", el2: "H", bias: 0.3, reason: "Iridium hydride candidate" },
  { el1: "Ce", el2: "H", bias: 0.35, reason: "Cerium hydride candidate" },
  { el1: "Th", el2: "H", bias: 0.3, reason: "Thorium hydride candidate" },
  { el1: "V", el2: "Si", bias: 0.25, reason: "V3Si A15 superconductor" },
];

interface PolicyWeights {
  elementGroup: number[];
  stoichTemplate: number[];
  structureType: number[];
  elementPairBias: number[][];
  elementPairSpecific: Map<string, number>;
}

function softmax(logits: number[], temperature: number = 1.0): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp((l - maxLogit) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function sampleFromDistribution(probs: number[]): number {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return probs.length - 1;
}

function stateToFeatures(state: RLState): number[] {
  return [
    state.bestTc / 400,
    state.avgRecentTc / 200,
    state.recentRewardTrend,
    state.familyDiversity / 20,
    Math.min(1, state.stagnationCycles / 30),
    state.explorationBudgetUsed,
    state.elementSuccessEntropy,
    Math.min(1, state.cycleNumber / 5000),
  ];
}

export class RLChemicalSpaceAgent {
  private policy: PolicyWeights;
  private replayBuffer: Experience[] = [];
  private maxReplaySize = 2000;
  private learningRate = 0.01;
  private gamma = 0.99;
  private epsilon = 0.15;
  private epsilonDecay = 0.9995;
  private minEpsilon = 0.05;
  private temperature = 1.0;
  private temperatureDecay = 0.999;
  private minTemperature = 0.3;
  private totalUpdates = 0;
  private elementSuccessRates: Map<string, { successes: number; total: number }> = new Map();
  private pairSuccessRates: Map<string, { successes: number; total: number; avgTc: number }> = new Map();
  private bestActionSequence: { action: RLAction; reward: number }[] = [];

  constructor() {
    this.policy = {
      elementGroup: new Array(ELEMENT_GROUPS.length).fill(0),
      stoichTemplate: new Array(STOICH_TEMPLATES.length).fill(0),
      structureType: new Array(STRUCTURE_TYPES.length).fill(0),
      elementPairBias: Array.from(
        { length: ELEMENT_GROUPS.length },
        () => new Array(ELEMENT_GROUPS.length).fill(0)
      ),
      elementPairSpecific: new Map<string, number>(),
    };

    this.initializePriors();
  }

  private makeElementPairKey(el1: string, el2: string): string {
    return [el1, el2].sort().join("-");
  }

  private initializePriors(): void {
    const tmBias = 0.3;
    this.policy.elementGroup[2] = tmBias;
    this.policy.elementGroup[3] = tmBias + 0.1;
    this.policy.elementGroup[5] = 0.2;
    this.policy.elementGroup[7] = 0.15;
    this.policy.elementGroup[8] = 0.1;

    this.policy.stoichTemplate[3] = 0.2;
    this.policy.stoichTemplate[4] = 0.15;
    this.policy.stoichTemplate[7] = 0.3;
    this.policy.stoichTemplate[8] = 0.2;

    this.policy.structureType[0] = 0.2;
    this.policy.structureType[3] = 0.15;
    this.policy.structureType[4] = 0.2;
    this.policy.structureType[8] = 0.1;

    for (const prior of KNOWN_PAIR_PRIORS) {
      const key = this.makeElementPairKey(prior.el1, prior.el2);
      this.policy.elementPairSpecific.set(key, prior.bias);
    }
  }

  selectAction(state: RLState): RLAction {
    const stateFeatures = stateToFeatures(state);
    const stagnationBoost = Math.min(0.5, state.stagnationCycles * 0.02);

    const effectiveEpsilon = Math.min(this.epsilon + stagnationBoost, 0.5);

    if (Math.random() < effectiveEpsilon) {
      return {
        elementGroup1: Math.floor(Math.random() * ELEMENT_GROUPS.length),
        elementGroup2: Math.floor(Math.random() * ELEMENT_GROUPS.length),
        stoichTemplate: Math.floor(Math.random() * STOICH_TEMPLATES.length),
        structureType: Math.floor(Math.random() * STRUCTURE_TYPES.length),
      };
    }

    const contextBias = this.computeContextBias(stateFeatures);

    const elLogits = this.policy.elementGroup.map((w, i) => w + contextBias.element[i]);
    const elProbs = softmax(elLogits, this.temperature);
    const elementGroup1 = sampleFromDistribution(elProbs);

    const pairLogits = this.policy.elementPairBias[elementGroup1].map(
      (w, i) => w + this.policy.elementGroup[i] + contextBias.element[i]
    );
    const pairProbs = softmax(pairLogits, this.temperature);
    const elementGroup2 = sampleFromDistribution(pairProbs);

    const stoichLogits = this.policy.stoichTemplate.map((w, i) => w + contextBias.stoich[i]);
    const stoichProbs = softmax(stoichLogits, this.temperature);
    const stoichTemplate = sampleFromDistribution(stoichProbs);

    const structLogits = this.policy.structureType.map((w, i) => w + contextBias.struct[i]);
    const structProbs = softmax(structLogits, this.temperature);
    const structureType = sampleFromDistribution(structProbs);

    return { elementGroup1, elementGroup2, stoichTemplate, structureType };
  }

  private computeContextBias(stateFeatures: number[]): {
    element: number[];
    stoich: number[];
    struct: number[];
  } {
    const stagnation = stateFeatures[4];
    const bestTcNorm = stateFeatures[0];

    const elementBias = new Array(ELEMENT_GROUPS.length).fill(0);
    if (stagnation > 0.3) {
      for (let i = 0; i < elementBias.length; i++) {
        elementBias[i] += (Math.random() - 0.5) * stagnation * 0.5;
      }
    }

    for (const [key, stats] of this.elementSuccessRates) {
      const idx = ELEMENT_GROUPS.findIndex(g => g.name === key);
      if (idx >= 0 && stats.total > 5) {
        const successRate = stats.successes / stats.total;
        elementBias[idx] += (successRate - 0.5) * 0.3;
      }
    }

    for (let gi = 0; gi < ELEMENT_GROUPS.length; gi++) {
      let groupPairBoost = 0;
      let pairCount = 0;
      for (const el of ELEMENT_GROUPS[gi].elements) {
        for (const [pairKey, bias] of this.policy.elementPairSpecific) {
          const pairParts = pairKey.split("-");
          if ((pairParts[0] === el || pairParts[1] === el) && Math.abs(bias) > 0.05) {
            groupPairBoost += bias;
            pairCount++;
          }
        }
      }
      if (pairCount > 0) {
        elementBias[gi] += (groupPairBoost / pairCount) * 0.4;
      }
    }

    const stoichBias = new Array(STOICH_TEMPLATES.length).fill(0);
    if (bestTcNorm > 0.3) {
      stoichBias[7] += 0.2;
      stoichBias[8] += 0.15;
    }

    const structBias = new Array(STRUCTURE_TYPES.length).fill(0);

    return { element: elementBias, stoich: stoichBias, struct: structBias };
  }

  updatePolicy(state: RLState, action: RLAction, reward: number): void {
    this.replayBuffer.push({
      state,
      action,
      reward,
      timestamp: Date.now(),
    });

    if (this.replayBuffer.length > this.maxReplaySize) {
      this.replayBuffer = this.replayBuffer.slice(-this.maxReplaySize);
    }

    if (reward > 0) {
      this.bestActionSequence.push({ action, reward });
      if (this.bestActionSequence.length > 50) {
        this.bestActionSequence.sort((a, b) => b.reward - a.reward);
        this.bestActionSequence = this.bestActionSequence.slice(0, 50);
      }
    }

    const lr = this.learningRate / (1 + this.totalUpdates * 0.0001);

    const advantageReward = reward - this.getBaselineReward();

    this.policy.elementGroup[action.elementGroup1] += lr * advantageReward;
    this.policy.elementGroup[action.elementGroup2] += lr * advantageReward * 0.7;
    this.policy.stoichTemplate[action.stoichTemplate] += lr * advantageReward;
    this.policy.structureType[action.structureType] += lr * advantageReward;

    this.policy.elementPairBias[action.elementGroup1][action.elementGroup2] += lr * advantageReward * 0.5;
    this.policy.elementPairBias[action.elementGroup2][action.elementGroup1] += lr * advantageReward * 0.5;

    if (this.totalUpdates % 20 === 0 && this.replayBuffer.length >= 32) {
      this.replayBatch(32);
    }

    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
    this.temperature = Math.max(this.minTemperature, this.temperature * this.temperatureDecay);
    this.totalUpdates++;
  }

  private getBaselineReward(): number {
    if (this.replayBuffer.length < 10) return 0;
    const recent = this.replayBuffer.slice(-50);
    return recent.reduce((s, e) => s + e.reward, 0) / recent.length;
  }

  private replayBatch(batchSize: number): void {
    const buffer = this.replayBuffer;
    const indices: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      indices.push(Math.floor(Math.random() * buffer.length));
    }

    const baseline = this.getBaselineReward();
    const lr = this.learningRate * 0.5 / (1 + this.totalUpdates * 0.0001);

    for (const idx of indices) {
      const exp = buffer[idx];
      const advantage = exp.reward - baseline;
      const decay = Math.pow(this.gamma, (buffer.length - idx) / buffer.length);

      this.policy.elementGroup[exp.action.elementGroup1] += lr * advantage * decay;
      this.policy.stoichTemplate[exp.action.stoichTemplate] += lr * advantage * decay;
      this.policy.structureType[exp.action.structureType] += lr * advantage * decay;
    }
  }

  recordElementOutcome(elements: string[], tc: number, passed: boolean): void {
    for (const el of elements) {
      const group = ELEMENT_GROUPS.find(g => (g.elements as readonly string[]).includes(el));
      if (!group) continue;

      const stats = this.elementSuccessRates.get(group.name) || { successes: 0, total: 0 };
      stats.total++;
      if (passed || tc > 20) stats.successes++;
      this.elementSuccessRates.set(group.name, stats);
    }

    if (elements.length >= 2) {
      for (let i = 0; i < elements.length; i++) {
        for (let j = i + 1; j < elements.length; j++) {
          const pair = this.makeElementPairKey(elements[i], elements[j]);
          const stats = this.pairSuccessRates.get(pair) || { successes: 0, total: 0, avgTc: 0 };
          stats.total++;
          if (tc > 20) stats.successes++;
          stats.avgTc = (stats.avgTc * (stats.total - 1) + tc) / stats.total;
          this.pairSuccessRates.set(pair, stats);

          const currentBias = this.policy.elementPairSpecific.get(pair) ?? 0;
          const lr = this.learningRate * 0.3;
          if (tc > 50) {
            const tcBonus = Math.min(0.5, (tc - 50) / 400);
            this.policy.elementPairSpecific.set(pair, currentBias + lr * tcBonus);
          } else if (tc < 5 && stats.total > 5) {
            this.policy.elementPairSpecific.set(pair, Math.max(-0.3, currentBias - lr * 0.1));
          }
        }
      }
    }
  }

  generateCandidatesFromAction(action: RLAction, count: number = 20): string[] {
    const group1 = ELEMENT_GROUPS[action.elementGroup1];
    const group2 = ELEMENT_GROUPS[action.elementGroup2];
    const template = STOICH_TEMPLATES[action.stoichTemplate];

    const candidates: string[] = [];
    const seen = new Set<string>();

    const pairWeightedElements1 = this.getWeightedElements(group1.elements, group2.elements);
    const pairWeightedElements2 = this.getWeightedElements(group2.elements, group1.elements);

    for (let attempt = 0; attempt < count * 5 && candidates.length < count; attempt++) {
      const el1 = this.sampleWeightedElement(pairWeightedElements1, group1.elements);
      const el2 = this.sampleWeightedElement(pairWeightedElements2, group2.elements);
      if (el1 === el2) continue;

      let formula: string;
      const pattern = template.pattern;

      if (template.nElements === 2) {
        formula = applyBinaryPattern(el1, el2, pattern);
      } else if (template.nElements === 3) {
        const thirdGroupIdx = Math.floor(Math.random() * ELEMENT_GROUPS.length);
        const group3 = ELEMENT_GROUPS[thirdGroupIdx];
        const el3 = group3.elements[Math.floor(Math.random() * group3.elements.length)];
        if (el3 === el1 || el3 === el2) continue;
        formula = applyTernaryPattern(el1, el2, el3, pattern);
      } else {
        const g3 = ELEMENT_GROUPS[Math.floor(Math.random() * ELEMENT_GROUPS.length)];
        const g4 = ELEMENT_GROUPS[Math.floor(Math.random() * ELEMENT_GROUPS.length)];
        const el3 = g3.elements[Math.floor(Math.random() * g3.elements.length)];
        const el4 = g4.elements[Math.floor(Math.random() * g4.elements.length)];
        if (new Set([el1, el2, el3, el4]).size < 4) continue;
        formula = `${el1}${el2}${el3}${el4}`;
      }

      if (!seen.has(formula)) {
        seen.add(formula);
        candidates.push(formula);
      }
    }

    return candidates;
  }

  private getWeightedElements(
    elements: readonly string[],
    partnerElements: readonly string[]
  ): Map<string, number> {
    const weights = new Map<string, number>();
    for (const el of elements) {
      let w = 1.0;
      for (const partner of partnerElements) {
        const key = this.makeElementPairKey(el, partner);
        const pairBias = this.policy.elementPairSpecific.get(key) ?? 0;
        w += pairBias;
      }
      const pairStats = this.pairSuccessRates;
      for (const partner of partnerElements) {
        const key = this.makeElementPairKey(el, partner);
        const stats = pairStats.get(key);
        if (stats && stats.total >= 3) {
          const successRate = stats.successes / stats.total;
          w += (successRate - 0.3) * 0.5;
        }
      }
      weights.set(el, Math.max(0.1, w));
    }
    return weights;
  }

  private sampleWeightedElement(
    weights: Map<string, number>,
    elements: readonly string[]
  ): string {
    const totalWeight = Array.from(weights.values()).reduce((a, b) => a + b, 0);
    let r = Math.random() * totalWeight;
    for (const el of elements) {
      const w = weights.get(el) ?? 1.0;
      r -= w;
      if (r <= 0) return el;
    }
    return elements[elements.length - 1];
  }

  computeReward(
    tcPredicted: number,
    bestTcBefore: number,
    pipelinePassed: boolean,
    stabilityScore: number,
    noveltyBonus: number = 0
  ): number {
    let reward = 0;

    const tcNorm = Math.min(1, tcPredicted / 400);
    reward += tcNorm * 2.0;

    if (tcPredicted > bestTcBefore) {
      const improvement = (tcPredicted - bestTcBefore) / Math.max(1, bestTcBefore);
      reward += improvement * 5.0;
    }

    if (pipelinePassed) {
      reward += 1.0;
    }

    reward += stabilityScore * 0.5;
    reward += noveltyBonus * 0.3;

    if (tcPredicted < 5) {
      reward -= 0.5;
    }

    return reward;
  }

  getActionDescription(action: RLAction): string {
    const g1 = ELEMENT_GROUPS[action.elementGroup1].name;
    const g2 = ELEMENT_GROUPS[action.elementGroup2].name;
    const st = STOICH_TEMPLATES[action.stoichTemplate].name;
    const str = STRUCTURE_TYPES[action.structureType];
    return `${g1}+${g2} / ${st} / ${str}`;
  }

  getStats(): {
    totalUpdates: number;
    epsilon: number;
    temperature: number;
    replayBufferSize: number;
    topElementGroups: { name: string; weight: number }[];
    topStoichTemplates: { name: string; weight: number }[];
    topStructureTypes: { name: string; weight: number }[];
    elementSuccessRates: { group: string; rate: number; total: number }[];
    topPairs: { pair: string; avgTc: number; count: number }[];
    recentAvgReward: number;
  } {
    const elWeights = this.policy.elementGroup.map((w, i) => ({
      name: ELEMENT_GROUPS[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    elWeights.sort((a, b) => b.weight - a.weight);

    const stWeights = this.policy.stoichTemplate.map((w, i) => ({
      name: STOICH_TEMPLATES[i].name,
      weight: Math.round(w * 1000) / 1000,
    }));
    stWeights.sort((a, b) => b.weight - a.weight);

    const strWeights = this.policy.structureType.map((w, i) => ({
      name: STRUCTURE_TYPES[i],
      weight: Math.round(w * 1000) / 1000,
    }));
    strWeights.sort((a, b) => b.weight - a.weight);

    const elSuccess: { group: string; rate: number; total: number }[] = [];
    for (const [group, stats] of this.elementSuccessRates) {
      elSuccess.push({
        group,
        rate: Math.round((stats.successes / Math.max(1, stats.total)) * 1000) / 1000,
        total: stats.total,
      });
    }
    elSuccess.sort((a, b) => b.rate - a.rate);

    const topPairs: { pair: string; avgTc: number; count: number }[] = [];
    for (const [pair, stats] of this.pairSuccessRates) {
      if (stats.total >= 3) {
        topPairs.push({
          pair,
          avgTc: Math.round(stats.avgTc * 10) / 10,
          count: stats.total,
        });
      }
    }
    topPairs.sort((a, b) => b.avgTc - a.avgTc);

    const recent = this.replayBuffer.slice(-50);
    const recentAvgReward = recent.length > 0
      ? recent.reduce((s, e) => s + e.reward, 0) / recent.length
      : 0;

    return {
      totalUpdates: this.totalUpdates,
      epsilon: Math.round(this.epsilon * 1000) / 1000,
      temperature: Math.round(this.temperature * 1000) / 1000,
      replayBufferSize: this.replayBuffer.length,
      topElementGroups: elWeights.slice(0, 5),
      topStoichTemplates: stWeights.slice(0, 5),
      topStructureTypes: strWeights.slice(0, 5),
      elementSuccessRates: elSuccess.slice(0, 5),
      topPairs: topPairs.slice(0, 10),
      recentAvgReward: Math.round(recentAvgReward * 1000) / 1000,
    };
  }

  getElementGroups(): typeof ELEMENT_GROUPS { return ELEMENT_GROUPS; }
  getStoichTemplates(): typeof STOICH_TEMPLATES { return STOICH_TEMPLATES; }
  getStructureTypes(): typeof STRUCTURE_TYPES { return STRUCTURE_TYPES; }
}

function applyBinaryPattern(el1: string, el2: string, pattern: string): string {
  switch (pattern) {
    case "A3B": return `${el1}3${el2}`;
    case "AB": return `${el1}${el2}`;
    case "AB3": return `${el1}${el2}3`;
    case "AH10": return `${el1}${el2}10`;
    default: return `${el1}${el2}`;
  }
}

function applyTernaryPattern(el1: string, el2: string, el3: string, pattern: string): string {
  switch (pattern) {
    case "AB2C2": return `${el1}${el2}2${el3}2`;
    case "ABC3": return `${el1}${el2}${el3}3`;
    case "A2BC": return `${el1}2${el2}${el3}`;
    case "ABH6": return `${el1}${el2}${el3}6`;
    case "A2B3C": return `${el1}2${el2}3${el3}`;
    default: return `${el1}${el2}${el3}`;
  }
}

export const rlAgent = new RLChemicalSpaceAgent();
