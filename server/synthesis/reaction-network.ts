import { ELEMENTAL_DATA, getMeltingPoint } from "../learning/elemental-data";
import { classifyFamily } from "../learning/utils";
import { findBestPrecursors, computePrecursorAvailabilityScore } from "./precursor-database";

export interface ReactionNode {
  id: string;
  species: string;
  type: "precursor" | "intermediate" | "target";
  availability: number;
  costTier: string;
}

export interface ReactionEdge {
  from: string;
  to: string;
  reactionType: string;
  temperature: number;
  pressure: number;
  gibbsFreeEnergy: number;
  activationEnergy: number;
  precursorAvailability: number;
  weight: number;
}

export interface ReactionGraphRoute {
  path: string[];
  edges: ReactionEdge[];
  totalCost: number;
  maxTemperature: number;
  maxPressure: number;
  stepCount: number;
  bottleneck: string | null;
  method: string;
}

export interface ReactionNetworkResult {
  formula: string;
  family: string;
  nodes: ReactionNode[];
  edges: ReactionEdge[];
  routes: ReactionGraphRoute[];
  bestRoute: ReactionGraphRoute | null;
  graphPathCost: number;
  summary: string;
}

export interface ReactionNetworkStats {
  totalNetworksBuilt: number;
  totalNodesCreated: number;
  totalEdgesCreated: number;
  avgPathCost: number;
  methodBreakdown: Record<string, number>;
  familyBreakdown: Record<string, number>;
}

const networkStats: ReactionNetworkStats = {
  totalNetworksBuilt: 0,
  totalNodesCreated: 0,
  totalEdgesCreated: 0,
  avgPathCost: 0,
  methodBreakdown: {},
  familyBreakdown: {},
};

let pathCostSum = 0;

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  return parseNestedFormula(cleaned);
}

function parseNestedFormula(s: string): Record<string, number> {
  const counts: Record<string, number> = {};
  let i = 0;
  while (i < s.length) {
    if (s[i] === '(') {
      let depth = 1;
      let j = i + 1;
      while (j < s.length && depth > 0) {
        if (s[j] === '(') depth++;
        else if (s[j] === ')') depth--;
        j++;
      }
      const inner = parseNestedFormula(s.substring(i + 1, j - 1));
      let numStr = '';
      while (j < s.length && (s[j] >= '0' && s[j] <= '9' || s[j] === '.')) {
        numStr += s[j]; j++;
      }
      const mult = numStr ? parseFloat(numStr) : 1;
      for (const [el, cnt] of Object.entries(inner)) {
        counts[el] = (counts[el] || 0) + cnt * mult;
      }
      i = j;
    } else if (s[i] >= 'A' && s[i] <= 'Z') {
      let el = s[i]; i++;
      while (i < s.length && s[i] >= 'a' && s[i] <= 'z') { el += s[i]; i++; }
      let numStr = '';
      while (i < s.length && (s[i] >= '0' && s[i] <= '9' || s[i] === '.')) { numStr += s[i]; i++; }
      const num = numStr ? parseFloat(numStr) : 1;
      counts[el] = (counts[el] || 0) + num;
    } else { i++; }
  }
  return counts;
}

const MIEDEMA_NONMETALS = new Set(["H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "I", "Xe", "Te", "As"]);

function computeMiedemaFormationEnergy(formula: string): number {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  if (elements.length < 2) return 0;

  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const fractions: Record<string, number> = {};
  for (const el of elements) {
    fractions[el] = counts[el] / totalAtoms;
  }

  let deltaH = 0;
  for (let i = 0; i < elements.length; i++) {
    for (let j = i + 1; j < elements.length; j++) {
      const dA = ELEMENTAL_DATA[elements[i]];
      const dB = ELEMENTAL_DATA[elements[j]];
      if (!dA || !dB) continue;

      const phiA = dA.miedemaPhiStar;
      const phiB = dB.miedemaPhiStar;
      const nwsA = dA.miedemaNws13;
      const nwsB = dB.miedemaNws13;
      const vA = dA.miedemaV23;
      const vB = dB.miedemaV23;

      if (phiA == null || phiB == null || nwsA == null || nwsB == null || vA == null || vB == null) continue;

      const deltaPhi = phiA - phiB;
      const deltaNws = nwsA - nwsB;
      const nwsAvgInv = 2 / (1 / nwsA + 1 / nwsB);
      const fAB = 2 * fractions[elements[i]] * fractions[elements[j]];
      const vAvg = (vA * fractions[elements[i]] + vB * fractions[elements[j]]) / (fractions[elements[i]] + fractions[elements[j]]);
      let interfaceEnergy = (-14.1 * deltaPhi * deltaPhi + 9.4 * deltaNws * deltaNws) / nwsAvgInv;

      const aIsNonmetal = MIEDEMA_NONMETALS.has(elements[i]);
      const bIsNonmetal = MIEDEMA_NONMETALS.has(elements[j]);
      if (aIsNonmetal !== bIsNonmetal) {
        interfaceEnergy -= (0.73 * deltaPhi * deltaPhi) / nwsAvgInv;
      }

      deltaH += fAB * vAvg * interfaceEnergy;
    }
  }

  return deltaH / totalAtoms;
}

function computeActivationEnergy(elements: string[], temperature: number): number {
  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > maxMp) maxMp = mp;
  }
  const tammTemp = maxMp > 0 ? 0.57 * maxMp : 1200;
  const barrier = Math.max(0.1, (tammTemp - temperature) / tammTemp);
  return Math.max(0, Math.min(3.0, barrier * 2.0));
}

const COMMON_INTERMEDIATES: Record<string, { formula: string; elements: string[]; stability: number }[]> = {
  Mg: [{ formula: "MgO", elements: ["Mg", "O"], stability: 0.95 }, { formula: "MgB2", elements: ["Mg", "B"], stability: 0.85 }],
  Ba: [{ formula: "BaO", elements: ["Ba", "O"], stability: 0.9 }, { formula: "BaCO3", elements: ["Ba", "O"], stability: 0.95 }, { formula: "BaCuO2", elements: ["Ba", "Cu", "O"], stability: 0.85 }],
  Y: [{ formula: "Y2O3", elements: ["Y", "O"], stability: 0.95 }, { formula: "Y2BaCuO5", elements: ["Y", "Ba", "Cu", "O"], stability: 0.8 }],
  La: [{ formula: "La2O3", elements: ["La", "O"], stability: 0.95 }],
  Cu: [{ formula: "CuO", elements: ["Cu", "O"], stability: 0.9 }, { formula: "Cu2O", elements: ["Cu", "O"], stability: 0.85 }],
  Fe: [{ formula: "Fe2O3", elements: ["Fe", "O"], stability: 0.9 }, { formula: "FeAs", elements: ["Fe", "As"], stability: 0.75 }, { formula: "FeSe", elements: ["Fe", "Se"], stability: 0.8 }],
  Nb: [{ formula: "Nb2O5", elements: ["Nb", "O"], stability: 0.9 }, { formula: "NbN", elements: ["Nb", "N"], stability: 0.85 }],
  Ti: [{ formula: "TiO2", elements: ["Ti", "O"], stability: 0.95 }],
  Sr: [{ formula: "SrO", elements: ["Sr", "O"], stability: 0.9 }, { formula: "SrCO3", elements: ["Sr", "O"], stability: 0.95 }],
  Ca: [{ formula: "CaO", elements: ["Ca", "O"], stability: 0.95 }, { formula: "CaH2", elements: ["Ca", "H"], stability: 0.8 }],
  Al: [{ formula: "Al2O3", elements: ["Al", "O"], stability: 0.95 }],
  Sn: [{ formula: "SnO2", elements: ["Sn", "O"], stability: 0.9 }],
  V: [{ formula: "V2O5", elements: ["V", "O"], stability: 0.9 }, { formula: "V3Si", elements: ["V", "Si"], stability: 0.85 }],
  Bi: [{ formula: "Bi2O3", elements: ["Bi", "O"], stability: 0.9 }],
  Pb: [{ formula: "PbO", elements: ["Pb", "O"], stability: 0.85 }],
  Si: [{ formula: "SiO2", elements: ["Si", "O"], stability: 0.95 }],
  Zr: [{ formula: "ZrO2", elements: ["Zr", "O"], stability: 0.95 }],
  B: [{ formula: "B2O3", elements: ["B", "O"], stability: 0.9 }],
};

const PREFERRED_METHODS: Record<string, string> = {
  "Hydride": "high-pressure",
  "Cuprate": "solid-state",
  "Pnictide": "solid-state",
  "A15": "arc-melting",
  "Boride": "arc-melting",
  "Chalcogenide": "solid-state",
  "Oxide": "solid-state",
  "Nitride": "solid-state",
  "Carbide": "arc-melting",
  "Silicide": "arc-melting",
  "Other": "solid-state",
};

function computeEdgeWeight(gibbs: number, activation: number, availability: number, temperature: number, pressure: number): number {
  const thermoWeight = gibbs < 0 ? 0.1 : gibbs < 0.3 ? 0.3 + gibbs : 0.5 + gibbs * 0.5;
  const kineticWeight = activation * 0.3;
  const availWeight = (1 - availability) * 0.2;
  const tempWeight = Math.min(1.0, temperature / 3000) * 0.15;
  const pressWeight = Math.min(1.0, pressure / 300) * 0.1;
  return Math.max(0.01, thermoWeight + kineticWeight + availWeight + tempWeight + pressWeight);
}

function buildReactionGraph(formula: string, elements: string[], counts: Record<string, number>, family: string): { nodes: ReactionNode[]; edges: ReactionEdge[] } {
  const nodes: ReactionNode[] = [];
  const edges: ReactionEdge[] = [];
  const nodeMap = new Map<string, ReactionNode>();

  const method = PREFERRED_METHODS[family] || "solid-state";
  const totalAtoms = Object.values(counts).reduce((a, b) => a + b, 0);
  const hasH = elements.includes("H");
  const hFrac = (counts["H"] || 0) / totalAtoms;

  const targetNode: ReactionNode = { id: `target:${formula}`, species: formula, type: "target", availability: 0, costTier: "n/a" };
  nodes.push(targetNode);
  nodeMap.set(targetNode.id, targetNode);

  const precursorSelections = findBestPrecursors(elements, method);
  const availResult = computePrecursorAvailabilityScore(precursorSelections);

  for (const sel of precursorSelections) {
    const nodeId = `precursor:${sel.precursor.formula}`;
    if (!nodeMap.has(nodeId)) {
      const node: ReactionNode = {
        id: nodeId,
        species: sel.precursor.formula,
        type: "precursor",
        availability: sel.precursor.availability,
        costTier: sel.precursor.costTier,
      };
      nodes.push(node);
      nodeMap.set(nodeId, node);
    }
  }

  for (const el of elements) {
    const nodeId = `precursor:${el}`;
    if (!nodeMap.has(nodeId)) {
      const data = ELEMENTAL_DATA[el];
      const node: ReactionNode = {
        id: nodeId,
        species: el,
        type: "precursor",
        availability: data ? 0.8 : 0.5,
        costTier: "medium",
      };
      nodes.push(node);
      nodeMap.set(nodeId, node);
    }
  }

  const intermediates: { formula: string; elements: string[]; stability: number }[] = [];
  for (const el of elements) {
    const inters = COMMON_INTERMEDIATES[el];
    if (!inters) continue;
    for (const inter of inters) {
      const allInTarget = inter.elements.every(ie => elements.includes(ie));
      if (!allInTarget) continue;
      if (inter.formula === formula) continue;
      intermediates.push(inter);
    }
  }

  for (const inter of intermediates) {
    const nodeId = `intermediate:${inter.formula}`;
    if (!nodeMap.has(nodeId)) {
      const node: ReactionNode = {
        id: nodeId,
        species: inter.formula,
        type: "intermediate",
        availability: inter.stability,
        costTier: "n/a",
      };
      nodes.push(node);
      nodeMap.set(nodeId, node);
    }
  }

  const precursorNodes = nodes.filter(n => n.type === "precursor");

  let maxMp = 0;
  for (const el of elements) {
    const mp = getMeltingPoint(el);
    if (mp !== null && mp > maxMp) maxMp = mp;
  }
  const baseTemp = Math.round(Math.max(800, 0.57 * (maxMp || 1500)));
  const basePressure = hasH && hFrac > 0.5 ? 150 : 0;

  for (const pn of precursorNodes) {
    const gibbs = computeMiedemaFormationEnergy(formula) * 0.5;
    const activation = computeActivationEnergy(elements, baseTemp);
    const weight = computeEdgeWeight(gibbs, activation, pn.availability, baseTemp, basePressure);

    edges.push({
      from: pn.id,
      to: targetNode.id,
      reactionType: method === "arc-melting" ? "arc-melt" : method === "high-pressure" ? "high-pressure synthesis" : "solid-state reaction",
      temperature: baseTemp,
      pressure: basePressure,
      gibbsFreeEnergy: Number(gibbs.toFixed(4)),
      activationEnergy: Number(activation.toFixed(4)),
      precursorAvailability: pn.availability,
      weight: Number(weight.toFixed(4)),
    });
  }

  for (const inter of intermediates) {
    const interNodeId = `intermediate:${inter.formula}`;
    const interEls = inter.elements;

    for (const pn of precursorNodes) {
      const pnCounts = parseFormulaCounts(pn.species);
      const pnElements = Object.keys(pnCounts);
      const relevant = pnElements.some(e => interEls.includes(e));
      if (!relevant) continue;

      const SPINEL_INTERMEDIATES = new Set(["MgAl2O4", "FeCr2O4", "NiFe2O4", "CoFe2O4", "ZnFe2O4", "MnFe2O4"]);
      const isHighTempIntermediate = SPINEL_INTERMEDIATES.has(inter.formula) || inter.formula.includes("2O4");
      const interTemp = isHighTempIntermediate
        ? Math.round(Math.min(baseTemp * 1.1, baseTemp + 200))
        : Math.round(baseTemp * 0.7);
      const gibbs = computeMiedemaFormationEnergy(inter.formula) * 0.3;
      const activation = computeActivationEnergy(interEls, interTemp);
      const weight = computeEdgeWeight(gibbs, activation, pn.availability, interTemp, 0);

      edges.push({
        from: pn.id,
        to: interNodeId,
        reactionType: "intermediate formation",
        temperature: interTemp,
        pressure: 0,
        gibbsFreeEnergy: Number(gibbs.toFixed(4)),
        activationEnergy: Number(activation.toFixed(4)),
        precursorAvailability: pn.availability,
        weight: Number(weight.toFixed(4)),
      });
    }

    const remainingEls = elements.filter(e => !interEls.includes(e));
    const interGibbs = computeMiedemaFormationEnergy(formula) * 0.4;
    const interActivation = computeActivationEnergy(elements, baseTemp);
    const interWeight = computeEdgeWeight(interGibbs, interActivation, inter.stability, baseTemp, basePressure);

    edges.push({
      from: interNodeId,
      to: targetNode.id,
      reactionType: remainingEls.length > 0 ? `reaction with ${remainingEls.join(", ")}` : "final sintering",
      temperature: baseTemp,
      pressure: basePressure,
      gibbsFreeEnergy: Number(interGibbs.toFixed(4)),
      activationEnergy: Number(interActivation.toFixed(4)),
      precursorAvailability: inter.stability,
      weight: Number(interWeight.toFixed(4)),
    });
  }

  return { nodes, edges };
}

function dijkstra(nodes: ReactionNode[], edges: ReactionEdge[], targetId: string): ReactionGraphRoute[] {
  const precursorIds = nodes.filter(n => n.type === "precursor").map(n => n.id);
  const routes: ReactionGraphRoute[] = [];

  const adjacency = new Map<string, ReactionEdge[]>();
  for (const e of edges) {
    if (!adjacency.has(e.from)) adjacency.set(e.from, []);
    adjacency.get(e.from)!.push(e);
  }

  for (const startId of precursorIds) {
    const dist = new Map<string, number>();
    const prev = new Map<string, { node: string; edge: ReactionEdge } | null>();
    const visited = new Set<string>();

    for (const n of nodes) {
      dist.set(n.id, Infinity);
      prev.set(n.id, null);
    }
    dist.set(startId, 0);

    const queue = [...nodes.map(n => n.id)];

    while (queue.length > 0) {
      let minDist = Infinity;
      let minIdx = 0;
      for (let i = 0; i < queue.length; i++) {
        const d = dist.get(queue[i]) ?? Infinity;
        if (d < minDist) { minDist = d; minIdx = i; }
      }

      const u = queue.splice(minIdx, 1)[0];
      if (visited.has(u)) continue;
      visited.add(u);

      const neighbors = adjacency.get(u) ?? [];
      for (const edge of neighbors) {
        const alt = (dist.get(u) ?? Infinity) + edge.weight;
        if (alt < (dist.get(edge.to) ?? Infinity)) {
          dist.set(edge.to, alt);
          prev.set(edge.to, { node: u, edge });
        }
      }
    }

    const targetDist = dist.get(targetId);
    if (targetDist === undefined || !isFinite(targetDist)) continue;

    const path: string[] = [];
    const pathEdges: ReactionEdge[] = [];
    let current: string | undefined = targetId;
    while (current) {
      const nodeData = nodes.find(n => n.id === current);
      if (nodeData) path.unshift(nodeData.species);
      const prevData = prev.get(current);
      if (prevData) {
        pathEdges.unshift(prevData.edge);
        current = prevData.node;
      } else {
        current = undefined;
      }
    }

    if (path.length < 2) continue;

    const maxTemp = Math.max(...pathEdges.map(e => e.temperature));
    const maxPressure = Math.max(...pathEdges.map(e => e.pressure));
    let bottleneck: string | null = null;
    let maxWeight = 0;
    for (const e of pathEdges) {
      if (e.weight > maxWeight) {
        maxWeight = e.weight;
        bottleneck = `${e.from.split(":")[1]} -> ${e.to.split(":")[1]}`;
      }
    }

    routes.push({
      path,
      edges: pathEdges,
      totalCost: Number(targetDist.toFixed(4)),
      maxTemperature: maxTemp,
      maxPressure: maxPressure,
      stepCount: pathEdges.length,
      bottleneck,
      method: pathEdges[pathEdges.length - 1]?.reactionType ?? "unknown",
    });
  }

  routes.sort((a, b) => a.totalCost - b.totalCost);

  const seen = new Set<string>();
  const unique: ReactionGraphRoute[] = [];
  for (const r of routes) {
    const key = r.path.join("->");
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(r);
  }

  return unique.slice(0, 10);
}

export function buildReactionNetwork(formula: string): ReactionNetworkResult {
  const counts = parseFormulaCounts(formula);
  const elements = Object.keys(counts);
  const family = classifyFamily(formula);

  if (elements.length < 2) {
    return {
      formula,
      family,
      nodes: [],
      edges: [],
      routes: [],
      bestRoute: null,
      graphPathCost: 0,
      summary: "Single-element formula - no reaction network needed",
    };
  }

  const { nodes, edges } = buildReactionGraph(formula, elements, counts, family);
  const targetId = `target:${formula}`;
  const routes = dijkstra(nodes, edges, targetId);
  const bestRoute = routes.length > 0 ? routes[0] : null;
  const graphPathCost = bestRoute?.totalCost ?? 0;

  networkStats.totalNetworksBuilt++;
  networkStats.totalNodesCreated += nodes.length;
  networkStats.totalEdgesCreated += edges.length;
  pathCostSum += graphPathCost;
  networkStats.avgPathCost = Number((pathCostSum / networkStats.totalNetworksBuilt).toFixed(4));

  const method = PREFERRED_METHODS[family] || "solid-state";
  networkStats.methodBreakdown[method] = (networkStats.methodBreakdown[method] || 0) + 1;
  networkStats.familyBreakdown[family] = (networkStats.familyBreakdown[family] || 0) + 1;

  const routeSummary = bestRoute
    ? `Best route: ${bestRoute.path.join(" -> ")} (cost: ${bestRoute.totalCost.toFixed(3)}, ${bestRoute.stepCount} steps, max ${bestRoute.maxTemperature}K)`
    : "No viable route found";

  return {
    formula,
    family,
    nodes,
    edges,
    routes,
    bestRoute,
    graphPathCost,
    summary: `Reaction network for ${formula} (${family}): ${nodes.length} nodes, ${edges.length} edges, ${routes.length} routes. ${routeSummary}`,
  };
}

export function getReactionNetworkStats(): ReactionNetworkStats {
  return { ...networkStats };
}
