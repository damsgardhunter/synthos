import {
  ELEMENTAL_DATA,
  getElementData,
  getCompositionWeightedProperty,
  isTransitionMetal,
  isRareEarth,
  isActinide,
  getDebyeTemperature,
} from "./elemental-data";
import { evaluateConvexHullStability } from "./structure-predictor";
import { storage } from "../storage";
import type { EventEmitter } from "./engine";
import type { SuperconductorCandidate } from "@shared/schema";

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const matches = formula.match(/[A-Z][a-z]*/g);
  return matches ? [...new Set(matches)] : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]*)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + count;
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  return Object.values(counts).reduce((a, b) => a + b, 0);
}

const STABLE_BINARY_PHASES: Record<string, Record<string, string>> = {
  O: {
    Ba: "BaO", Sr: "SrO", Ca: "CaO", Cu: "CuO", La: "La2O3", Y: "Y2O3",
    Ti: "TiO2", Fe: "Fe2O3", Al: "Al2O3", Mg: "MgO", Zn: "ZnO", Bi: "Bi2O3",
    Nb: "Nb2O5", Ta: "Ta2O5", Zr: "ZrO2", Hf: "HfO2", Ce: "CeO2", Mn: "MnO2",
    Ni: "NiO", Co: "CoO", Pb: "PbO", Sn: "SnO2", V: "V2O5", W: "WO3",
    Mo: "MoO3", Cr: "Cr2O3", Sc: "Sc2O3", Nd: "Nd2O3", Sm: "Sm2O3",
    Eu: "Eu2O3", Gd: "Gd2O3", Th: "ThO2", U: "UO2",
  },
  H: {
    La: "LaH3", Y: "YH3", Ca: "CaH2", Ba: "BaH2", Sr: "SrH2", Mg: "MgH2",
    Ti: "TiH2", Zr: "ZrH2", Nb: "NbH", V: "VH", Pd: "PdH", Ce: "CeH3",
    Th: "ThH2", U: "UH3", Li: "LiH", Na: "NaH", K: "KH",
  },
  S: {
    Ba: "BaS", Cu: "Cu2S", Fe: "FeS", La: "La2S3", Ca: "CaS", Sr: "SrS",
    Zn: "ZnS", Pb: "PbS", Bi: "Bi2S3", Ni: "NiS", Mo: "MoS2",
  },
  N: {
    Ti: "TiN", Zr: "ZrN", Nb: "NbN", Ta: "TaN", La: "LaN", Hf: "HfN",
    Al: "AlN", V: "VN", Cr: "CrN", Fe: "Fe4N", Th: "ThN",
  },
  F: {
    Ba: "BaF2", Ca: "CaF2", Sr: "SrF2", La: "LaF3", Y: "YF3", Li: "LiF",
    Na: "NaF", K: "KF", Bi: "BiF3", Al: "AlF3",
  },
  C: {
    Ti: "TiC", Zr: "ZrC", Nb: "NbC", Ta: "TaC", W: "WC", Mo: "Mo2C",
    V: "VC", Hf: "HfC", Fe: "Fe3C", Ca: "CaC2", Si: "SiC",
  },
};

function resolveStableBinary(el1: string, el2: string): string | null {
  if (STABLE_BINARY_PHASES[el2]?.[el1]) return STABLE_BINARY_PHASES[el2][el1];
  if (STABLE_BINARY_PHASES[el1]?.[el2]) return STABLE_BINARY_PHASES[el1][el2];
  return null;
}

export type StabilityClass = "thermodynamically-stable" | "metastable-accessible" | "metastable-difficult" | "likely-unstable";

export interface ThermodynamicLandscape {
  formula: string;
  stabilityClass: StabilityClass;
  energyAboveHull: number;
  formationEnergy: number;
  decompositionBarrier: number;
  estimatedQuenchRateKPerSec: number | null;
  decompositionProducts: string[];
  competingPhaseCount: number;
  hasMagneticCompetition: boolean;
  hasStructuralInstability: boolean;
  maxSynthesisTemp: number;
  meltingPointEstimate: number;
  bulkModulusEstimate: number;
  isHydride: boolean;
  isHEA: boolean;
  isCuprate: boolean;
  source: string;
}

export async function analyzeThermodynamicLandscape(
  formula: string,
  candidate?: SuperconductorCandidate | null
): Promise<ThermodynamicLandscape> {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const hullResult = await evaluateConvexHullStability(0, formula);

  let stabilityClass: StabilityClass;
  const eAboveHull = hullResult.hullDistance;
  if (eAboveHull <= 0.005) {
    stabilityClass = "thermodynamically-stable";
  } else if (eAboveHull <= 0.1) {
    stabilityClass = "metastable-accessible";
  } else if (eAboveHull <= 0.3) {
    stabilityClass = "metastable-difficult";
  } else {
    stabilityClass = "likely-unstable";
  }

  const meltingPoints: number[] = [];
  const bulkModuli: number[] = [];
  const elementDataCache: Record<string, ReturnType<typeof getElementData>> = {};
  for (const el of elements) {
    const data = getElementData(el);
    elementDataCache[el] = data;
    if (data?.meltingPoint) meltingPoints.push(data.meltingPoint);
    if (data?.bulkModulus) bulkModuli.push(data.bulkModulus * (counts[el] / totalAtoms));
  }
  const meltingPointEstimate = meltingPoints.length > 0
    ? Math.round(meltingPoints.reduce((a, b) => a + b, 0) / meltingPoints.length)
    : 1500;
  const bulkModulusEstimate = bulkModuli.length > 0
    ? Math.round(bulkModuli.reduce((a, b) => a + b, 0))
    : 100;

  const decompositionBarrier = eAboveHull > 0
    ? Math.max(0.01, eAboveHull * 0.6)
    : 0;

  let estimatedQuenchRateKPerSec: number | null = null;
  if (decompositionBarrier > 0) {
    const kB = 8.617e-5;
    const targetLifetime = 3600;
    const Tsynth = meltingPointEstimate * 0.8;
    const exponent = decompositionBarrier / (kB * Tsynth);
    if (exponent > 50) {
      estimatedQuenchRateKPerSec = 1;
    } else if (exponent > 2) {
      const expVal = Math.exp(-exponent);
      estimatedQuenchRateKPerSec = expVal > 0
        ? Math.round(Tsynth / (targetLifetime / expVal))
        : 1;
      estimatedQuenchRateKPerSec = Math.max(1, Math.min(1e8, Math.abs(estimatedQuenchRateKPerSec)));
      if (decompositionBarrier < 0.05) estimatedQuenchRateKPerSec = Math.max(1e5, estimatedQuenchRateKPerSec);
      else if (decompositionBarrier < 0.1) estimatedQuenchRateKPerSec = Math.max(1e3, estimatedQuenchRateKPerSec);
      else estimatedQuenchRateKPerSec = Math.max(10, estimatedQuenchRateKPerSec);
    } else {
      estimatedQuenchRateKPerSec = 1e6;
    }
  }

  const decompositionProducts: string[] = [];
  if (elements.length >= 2 && eAboveHull > 0.01) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const binary = resolveStableBinary(elements[i], elements[j]);
        if (binary) decompositionProducts.push(binary);
      }
    }
    if (decompositionProducts.length > 4) {
      decompositionProducts.length = 4;
    }
  }

  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e))
    .reduce((s, e) => s + (counts[e] || 0), 0);
  const isHydride = hCount > 0 && metalAtoms > 0 && hCount / metalAtoms >= 4;

  const metalEls = elements.filter(e => isTransitionMetal(e) || isRareEarth(e) || isActinide(e) ||
    ["Al", "Mg", "Ti", "Zn", "Ga", "Sn"].includes(e));
  const isHEA = metalEls.length >= 4;
  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;

  const competingPhases = candidate?.competingPhases;
  const competingPhaseCount = Array.isArray(competingPhases) ? competingPhases.length : 0;
  const hasMagneticCompetition = Array.isArray(competingPhases)
    ? competingPhases.some((p: any) => p.type === "magnetism" && p.suppressesSC)
    : false;
  const hasStructuralInstability = Array.isArray(competingPhases)
    ? competingPhases.some((p: any) => p.type === "structural")
    : false;

  const meltingCelsius = meltingPointEstimate > 300 ? meltingPointEstimate - 273 : meltingPointEstimate;
  const maxSynthesisTemp = Math.round(Math.min(1600, meltingCelsius * 0.65));

  return {
    formula,
    stabilityClass,
    energyAboveHull: eAboveHull,
    formationEnergy: hullResult.formationEnergy,
    decompositionBarrier,
    estimatedQuenchRateKPerSec,
    decompositionProducts,
    competingPhaseCount,
    hasMagneticCompetition,
    hasStructuralInstability,
    maxSynthesisTemp,
    meltingPointEstimate,
    bulkModulusEstimate,
    isHydride,
    isHEA,
    isCuprate,
    source: hullResult.source,
  };
}

export interface NovelSynthesisRoute {
  method: string;
  steps: string[];
  temperatureProfile: string;
  pressureProfile: string;
  estimatedCoolingRate: string;
  atmosphere: string;
  expectedYieldRange: string;
  noveltyScore: number;
  synthesisConfidence: "high" | "medium" | "low";
  physicsJustification: string;
  source: "physics-reasoned";
  keyInnovation: string;
}

const PROCESSING_TECHNIQUES = {
  rapidQuenching: {
    name: "Rapid Solidification / Melt Spinning",
    coolingRates: "1e5 - 1e6 K/s",
    applicability: "metastable alloys, amorphous phases",
    maxTemp: 3000,
  },
  mechanicalAlloying: {
    name: "High-Energy Ball Milling",
    coolingRates: "N/A (solid state)",
    applicability: "HEA, nanocrystalline, metastable intermetallics",
    maxTemp: 500,
  },
  arcMelting: {
    name: "Arc Melting with Splat Quenching",
    coolingRates: "1e4 - 1e5 K/s",
    applicability: "refractory alloys, HEA, intermetallics",
    maxTemp: 4000,
  },
  sputtering: {
    name: "Magnetron Sputtering (Thin Film)",
    coolingRates: "1e8 - 1e10 K/s effective",
    applicability: "metastable thin films, epitaxial phases",
    maxTemp: 1500,
  },
  highPressureSynthesis: {
    name: "Diamond Anvil Cell / Multi-Anvil Press",
    coolingRates: "Variable (1-1000 K/s)",
    applicability: "hydrides, high-pressure phases, dense polymorphs",
    maxPressure: 300,
  },
  solGel: {
    name: "Sol-Gel with Controlled Calcination",
    coolingRates: "1-10 K/min",
    applicability: "oxides, cuprates, perovskites",
    maxTemp: 1200,
  },
  laserAblation: {
    name: "Pulsed Laser Deposition (PLD)",
    coolingRates: "1e9 - 1e11 K/s effective",
    applicability: "epitaxial thin films, metastable oxides",
    maxTemp: 3500,
  },
  electrodeposition: {
    name: "Electrochemical Deposition",
    coolingRates: "N/A (room temperature)",
    applicability: "metallic alloys, HEA thin films, nanostructured metals",
    maxTemp: 100,
  },
};

export function proposeNovelSynthesisRoutes(
  formula: string,
  landscape: ThermodynamicLandscape
): NovelSynthesisRoute[] {
  const routes: NovelSynthesisRoute[] = [];
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const hasRefractoryMetal = elements.some(e => ["W", "Mo", "Ta", "Nb", "Re", "Hf", "Zr"].includes(e));
  const hasVolatileElement = elements.some(e => ["Hg", "Cd", "Zn", "As", "Se", "Te", "S"].includes(e));
  const hasLightElement = elements.some(e => ["B", "C", "N"].includes(e));
  const hasOxygen = elements.includes("O");

  if (landscape.stabilityClass === "thermodynamically-stable") {
    if (landscape.isHydride) {
      routes.push(createHydrideRoute(formula, landscape, elements, counts));
    }
    if (landscape.isHEA) {
      routes.push(createHEAOptimizedRoute(formula, landscape, elements));
    }
    if (landscape.isCuprate) {
      routes.push(createCuprateOptimizedRoute(formula, landscape, elements));
    }
    if (routes.length === 0) {
      routes.push(createConventionalOptimizedRoute(formula, landscape, elements));
    }
  }

  if (landscape.stabilityClass === "metastable-accessible" || landscape.stabilityClass === "metastable-difficult") {
    if (landscape.isHydride) {
      routes.push(createHydrideRoute(formula, landscape, elements, counts));
    }

    if (landscape.isHEA) {
      routes.push(createHEANonEquilibriumRoute(formula, landscape, elements));
    }

    const quenchRate = landscape.estimatedQuenchRateKPerSec ?? 1e5;
    if (quenchRate <= 1e6) {
      routes.push(createRapidQuenchRoute(formula, landscape, elements));
    }

    if (quenchRate > 1e6 || landscape.stabilityClass === "metastable-difficult") {
      routes.push(createThinFilmRoute(formula, landscape, elements));
    }

    if (hasLightElement && !landscape.isHydride) {
      routes.push(createReactiveMillingRoute(formula, landscape, elements));
    }
  }

  if (landscape.stabilityClass === "likely-unstable") {
    routes.push(createThinFilmRoute(formula, landscape, elements));
    if (landscape.isHydride) {
      routes.push(createHydrideRoute(formula, landscape, elements, counts));
    }
  }

  return routes.slice(0, 3);
}

function createHEAOptimizedRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  const tmElements = elements.filter(e => isTransitionMetal(e));
  const heaMeltC = landscape.meltingPointEstimate > 300 ? landscape.meltingPointEstimate - 273 : landscape.meltingPointEstimate;
  const sinterTemp = Math.round(Math.min(1400, heaMeltC * 0.6));

  return {
    method: "Entropy-Stabilized Synthesis via Mechanical Alloying + SPS",
    steps: [
      `Weigh elemental powders (${elements.join(", ")}) in equimolar ratios, 99.9%+ purity`,
      `High-energy ball mill under Ar atmosphere: 300 rpm, BPR 10:1, WC vial, 24-48 hours with 30-min pause cycles`,
      `Characterize powder by XRD — confirm single-phase solid solution (broad FCC/BCC peaks)`,
      `Consolidate by spark plasma sintering (SPS) at ${sinterTemp}C, 50 MPa, 5 min hold under vacuum`,
      `Rapid cool at 100 K/min to trap high-entropy phase`,
      `Anneal at ${Math.round(sinterTemp * 0.6)}C for 2h under Ar to relieve stress without phase decomposition`,
      `Characterize: XRD (single phase), SEM-EDS (compositional uniformity), 4-point probe (resistivity vs T)`,
    ],
    temperatureProfile: `RT -> ${sinterTemp}C (100C/min SPS) -> hold 5 min -> ${Math.round(sinterTemp * 0.6)}C (anneal 2h) -> RT`,
    pressureProfile: "Ambient (milling) -> 50 MPa (SPS) -> Ambient (anneal)",
    estimatedCoolingRate: "100 K/min (SPS chamber)",
    atmosphere: "Ar throughout; vacuum during SPS",
    expectedYieldRange: "60-80% (powder consolidation losses)",
    noveltyScore: 0.7,
    synthesisConfidence: "medium",
    physicsJustification: `High configurational entropy (${(Math.log(elements.length) * 8.314).toFixed(1)} J/mol/K for ${elements.length} components) stabilizes single-phase solid solution. SPS preserves metastable high-entropy phase by limiting diffusion time. The cocktail effect from mass disorder (${tmElements.join("/")} mix) enhances phonon scattering, potentially boosting electron-phonon coupling.`,
    source: "physics-reasoned",
    keyInnovation: `Entropy-driven phase stabilization with rapid SPS consolidation to preserve disordered phonon spectrum favorable for superconductivity`,
  };
}

function createHEANonEquilibriumRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  return {
    method: "Combinatorial Sputtering with Composition Gradient",
    steps: [
      `Prepare individual sputter targets of ${elements.join(", ")} (99.99% purity)`,
      `Co-sputter onto single-crystal substrate (sapphire or Si) at room temperature`,
      `Use power-gradient configuration to create composition spread across 2-inch wafer`,
      `Map composition by automated EDS at 200+ points`,
      `Screen entire wafer for superconductivity: scanning SQUID microscopy at 4K sweep to 300K`,
      `Identify optimal composition region from Tc map`,
      `Deposit optimized composition as uniform film, 200-500 nm thick`,
      `Measure transport: R(T) 2-300K, Hall effect, magnetoresistance`,
    ],
    temperatureProfile: "Room temperature deposition (substrate unheated)",
    pressureProfile: "3-5 mTorr Ar sputter gas; base pressure < 5e-8 Torr",
    estimatedCoolingRate: "N/A (room-temperature process, effective quench rate >1e9 K/s)",
    atmosphere: "Ultra-high vacuum with 99.999% Ar sputter gas",
    expectedYieldRange: "N/A (thin film; wafer-scale screening)",
    noveltyScore: 0.85,
    synthesisConfidence: "medium",
    physicsJustification: `Room-temperature sputtering produces far-from-equilibrium atomic arrangements impossible in bulk synthesis. The extreme effective quench rate (>1e9 K/s) can trap metastable phases ${landscape.energyAboveHull.toFixed(3)} eV/atom above the hull. Combinatorial screening maps the composition-Tc landscape in a single experiment rather than synthesizing hundreds of discrete samples.`,
    source: "physics-reasoned",
    keyInnovation: `Combinatorial thin-film approach maps the entire HEA composition space for superconductivity in a single deposition, bypassing the need for bulk synthesis of each composition`,
  };
}

function createCuprateOptimizedRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  const rareEarth = elements.find(e => isRareEarth(e)) || "Y";
  const calcineTemp = Math.min(950, landscape.maxSynthesisTemp);

  return {
    method: "Optimized Sol-Gel with Oxygen-Controlled Annealing",
    steps: [
      `Dissolve metal nitrates (${elements.filter(e => e !== "O").join(", ")}) in stoichiometric ratios in dilute HNO3`,
      `Add citric acid (2:1 molar ratio to total metal ions) and ethylene glycol as complexing agents`,
      `Heat to 80C with stirring until gel formation (4-6 hours)`,
      `Decompose gel at 400C in air (2h) to form precursor powder`,
      `Calcine at ${calcineTemp}C in flowing O2 for 24h with intermediate grindings at 8h intervals`,
      `Press into pellets (1 GPa) and sinter at ${calcineTemp}C in flowing O2 for 48h`,
      `Slow-cool in pure O2: 1C/min from ${calcineTemp}C to 400C, hold at 400C for 24h (oxygen loading)`,
      `Characterize: XRD (phase purity), oxygen content (iodometric titration), Tc by AC susceptibility`,
    ],
    temperatureProfile: `80C (gelation) -> 400C (decompose) -> ${calcineTemp}C (calcine 24h) -> ${calcineTemp}C (sinter 48h) -> 400C (O2 anneal 24h) -> RT`,
    pressureProfile: "Ambient; 1 GPa for pellet pressing",
    estimatedCoolingRate: "1 C/min (controlled for optimal oxygen stoichiometry)",
    atmosphere: "Air (decomposition) -> Flowing O2 (calcination, sintering, annealing)",
    expectedYieldRange: "70-85% (phase-pure material after optimization)",
    noveltyScore: 0.5,
    synthesisConfidence: "high",
    physicsJustification: `Cuprate superconductivity depends critically on the CuO2 plane hole doping level, controlled by oxygen stoichiometry. The extended oxygen anneal at 400C in flowing O2 maximizes oxygen content in the charge reservoir layers. Sol-gel provides atomic-level mixing impossible in solid-state synthesis, ensuring homogeneous ${rareEarth}-Ba-Cu cation distribution. The slow cool rate (1C/min) prevents oxygen loss that would reduce Tc.`,
    source: "physics-reasoned",
    keyInnovation: `Dual-stage oxygen loading protocol: high-temperature sintering for phase formation followed by extended low-temperature anneal for optimal hole doping`,
  };
}

function createConventionalOptimizedRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  const sinterTemp = Math.round(landscape.maxSynthesisTemp * 0.85);

  return {
    method: "Optimized Solid-State Reaction with Phase-Purity Verification",
    steps: [
      `Weigh precursor powders (${elements.map(e => `${e} oxide/carbonate`).join(", ")}) in stoichiometric ratios`,
      `Ball mill in ethanol: planetary mill 250 rpm, 6h, agate media`,
      `Dry and press into pellets at 300 MPa`,
      `First firing at ${Math.round(sinterTemp * 0.8)}C in air for 12h`,
      `Regrind, repress, fire at ${sinterTemp}C for 24h`,
      `XRD phase check — repeat grinding/firing if secondary phases detected`,
      `Final sinter at ${sinterTemp}C for 48h under optimized atmosphere`,
      `Characterize: XRD, SEM, transport measurements`,
    ],
    temperatureProfile: `RT -> ${Math.round(sinterTemp * 0.8)}C (12h) -> regrind -> ${sinterTemp}C (24h) -> ${sinterTemp}C (48h final) -> RT`,
    pressureProfile: "300 MPa (pellet pressing); ambient otherwise",
    estimatedCoolingRate: "5 C/min (furnace cool)",
    atmosphere: "Air or flowing Ar depending on composition",
    expectedYieldRange: "80-95%",
    noveltyScore: 0.3,
    synthesisConfidence: "high",
    physicsJustification: `Thermodynamically stable compound (hull distance ${landscape.energyAboveHull.toFixed(3)} eV/atom) — standard solid-state synthesis is thermodynamically favorable. Multi-step firing with intermediate grinding ensures complete reaction and phase homogeneity. Sintering temperature set at 65% of melting point (${landscape.meltingPointEstimate}K -> ${landscape.maxSynthesisTemp}C) for optimal densification without decomposition.`,
    source: "physics-reasoned",
    keyInnovation: `Physics-guided sintering temperature (${sinterTemp}C) based on compositionally-weighted melting point estimate with iterative phase-purity feedback`,
  };
}

function createHydrideRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[], counts: Record<string, number>): NovelSynthesisRoute {
  const hCount = counts["H"] || 0;
  const metalEls = elements.filter(e => e !== "H" && e !== "O" && e !== "N");
  const targetPressure = Math.max(50, Math.round(landscape.energyAboveHull * 500 + 100));

  return {
    method: "High-Pressure Hydrogenation with Laser Heating in DAC",
    steps: [
      `Load ${metalEls.join("/")} precursor (pre-alloyed or layered) into diamond anvil cell (DAC)`,
      `Add ammonia borane (NH3BH3) or paraffin as hydrogen source`,
      `Compress to ${targetPressure} GPa incrementally (5 GPa/step) with XRD monitoring at each step`,
      `Laser heat to ${Math.min(2500, landscape.meltingPointEstimate)}K at target pressure to initiate hydrogenation`,
      `Hold at pressure for 1-4 hours to equilibrate hydrogen content`,
      `Characterize in situ: XRD (phase identification), Raman (H2 vibrons), electrical resistance (4-probe in DAC)`,
      `Map R(T) from 300K to 4K at target pressure to identify superconducting transition`,
      `Attempt controlled decompression (1 GPa/hr) to determine metastability window — monitor XRD for phase retention`,
    ],
    temperatureProfile: `RT -> ${Math.min(2500, landscape.meltingPointEstimate)}K (laser pulse, ~1s) -> RT -> cool to 4K for R(T)`,
    pressureProfile: `0 -> ${targetPressure} GPa (5 GPa/step) -> hold -> decompression attempt`,
    estimatedCoolingRate: "1e6 K/s (laser off; thermal mass of DAC sample ~micrograms)",
    atmosphere: "H2-rich environment from chemical hydrogen source",
    expectedYieldRange: "N/A (microgram-scale DAC experiment)",
    noveltyScore: 0.65,
    synthesisConfidence: "medium",
    physicsJustification: `Hydrogen-rich composition (H:metal ratio = ${(hCount / Math.max(1, elements.filter(e => e !== "H").reduce((s, e) => s + (counts[e] || 0), 0))).toFixed(1)}) requires extreme pressure to overcome H2 molecular stability. Chemical H-source (ammonia borane) provides reactive atomic H at lower pressures than pure H2 loading. Laser heating overcomes kinetic barriers to form the target ${formula} stoichiometry. Controlled decompression tests whether the phase can be quench-recovered to lower pressures — bulk modulus ${landscape.bulkModulusEstimate} GPa suggests ${landscape.bulkModulusEstimate > 150 ? "reasonable" : "limited"} structural rigidity for pressure retention.`,
    source: "physics-reasoned",
    keyInnovation: `Chemical hydrogen source (ammonia borane) for enhanced hydrogenation kinetics + controlled decompression protocol to test metastable phase retention at reduced pressures`,
  };
}

function createRapidQuenchRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  const meltTemp = landscape.meltingPointEstimate;
  const quenchRate = landscape.estimatedQuenchRateKPerSec ?? 1e5;

  let quenchMethod: string;
  let coolingRateStr: string;
  if (quenchRate > 1e5) {
    quenchMethod = "melt spinning onto Cu wheel (40 m/s tangential velocity)";
    coolingRateStr = "~1e6 K/s";
  } else if (quenchRate > 1e3) {
    quenchMethod = "arc melting with Cu-hearth splat quenching";
    coolingRateStr = "~1e4-1e5 K/s";
  } else {
    quenchMethod = "induction melting with water quench";
    coolingRateStr = "~1e3 K/s";
  }

  return {
    method: `Rapid Solidification: ${quenchMethod.split("(")[0].trim()}`,
    steps: [
      `Pre-alloy ${elements.join("+")} by arc melting under Ar (flip and remelt 5x for homogeneity)`,
      `Verify composition by EDS and weight loss check`,
      `Remelt and apply ${quenchMethod}`,
      `Characterize ribbon/splat: XRD (check for target phase vs equilibrium phases)`,
      `If amorphous: controlled crystallization anneal at ${Math.round(meltTemp * 0.3)}C to nucleate target phase`,
      `If crystalline metastable: verify phase by electron diffraction (TEM)`,
      `Transport measurements: R(T) 2-300K, magnetization M(T) in 10 Oe field-cooled/zero-field-cooled`,
    ],
    temperatureProfile: `${meltTemp}C (melt) -> RT (quench at ${coolingRateStr}) -> ${Math.round(meltTemp * 0.3)}C (optional anneal) -> RT`,
    pressureProfile: "Ambient (Ar atmosphere)",
    estimatedCoolingRate: coolingRateStr,
    atmosphere: "High-purity Ar (O2 < 1 ppm) for arc melting; Ar or vacuum for quenching",
    expectedYieldRange: "40-70% (ribbon/splat recovery; some material lost as splash)",
    noveltyScore: 0.7,
    synthesisConfidence: landscape.stabilityClass === "metastable-accessible" ? "high" : "medium",
    physicsJustification: `Decomposition barrier of ${landscape.decompositionBarrier.toFixed(3)} eV/atom requires cooling rate >= ${quenchRate.toExponential(0)} K/s to kinetically trap the metastable phase. ${quenchMethod} achieves ${coolingRateStr}, ${quenchRate <= (landscape.estimatedQuenchRateKPerSec ?? 0) ? "exceeding" : "approaching"} the required rate. The target phase is ${landscape.energyAboveHull.toFixed(3)} eV/atom above the convex hull — similar to known metastable superconductors that have been successfully quench-synthesized (e.g., A15 Nb3Ge at 0.05 eV above hull).`,
    source: "physics-reasoned",
    keyInnovation: `Quench rate matched to decomposition barrier height — ${coolingRateStr} traps the metastable ${formula} phase before diffusive decomposition can occur`,
  };
}

function createThinFilmRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  const hasVolatile = elements.some(e => ["Hg", "Cd", "Zn", "As", "Se", "Te"].includes(e));
  const method = hasVolatile ? "Molecular Beam Epitaxy (MBE)" : "Pulsed Laser Deposition (PLD)";

  return {
    method: `Epitaxial Thin Film: ${method}`,
    steps: [
      hasVolatile
        ? `Prepare elemental effusion cells for ${elements.join(", ")} with individual flux calibration by RHEED oscillations`
        : `Prepare polycrystalline ${formula} target by solid-state synthesis (phase-pure, 95%+ dense)`,
      `Select substrate: SrTiO3 (001) for perovskites, MgO for cubic, sapphire for general use — lattice mismatch < 5%`,
      `Clean substrate: ultrasonic (acetone, IPA, DI water), O2 plasma 10 min, anneal at 800C in O2`,
      hasVolatile
        ? `Deposit by co-evaporation at substrate temp ${Math.round(landscape.meltingPointEstimate * 0.3)}C under 1e-6 Torr O2 (if oxide)`
        : `PLD: KrF excimer 248nm, 2 J/cm2, 5 Hz rep rate, substrate temp ${Math.round(landscape.meltingPointEstimate * 0.4)}C`,
      `Deposit 50-200 nm film thickness (monitored by RHEED oscillation counting)`,
      `Post-anneal in situ at deposition temperature for 30 min`,
      `Cool at 10 C/min under deposition atmosphere`,
      `Characterize: XRD (epitaxy, rocking curve), AFM (surface roughness), R(T) by van der Pauw`,
    ],
    temperatureProfile: `Substrate: ${Math.round(landscape.meltingPointEstimate * 0.35)}C (deposition) -> hold 30 min (anneal) -> RT (10 C/min)`,
    pressureProfile: hasVolatile ? "1e-9 Torr base; 1e-6 Torr O2 during growth" : "1e-6 Torr base; 50-200 mTorr O2 during PLD",
    estimatedCoolingRate: "10 C/min (controlled chamber cool)",
    atmosphere: hasVolatile ? "UHV with controlled O2 partial pressure" : "O2/Ar mixture optimized for stoichiometry",
    expectedYieldRange: "N/A (thin film on substrate)",
    noveltyScore: 0.8,
    synthesisConfidence: landscape.stabilityClass === "likely-unstable" ? "low" : "medium",
    physicsJustification: `Epitaxial thin-film growth enables synthesis of phases ${landscape.energyAboveHull.toFixed(3)} eV/atom above the bulk convex hull through substrate-imposed strain stabilization and kinetic trapping at the growth surface. Each deposited monolayer crystallizes before diffusion can drive decomposition — effective quench rate >1e9 K/s. Substrate epitaxy can also stabilize crystal structures not accessible in bulk (e.g., tetragonal phases on cubic substrates via compressive strain). Film thickness < 200 nm ensures coherent strain throughout.`,
    source: "physics-reasoned",
    keyInnovation: `Epitaxial strain engineering: substrate lattice mismatch imposes biaxial strain that can stabilize the target crystal structure and modify the electronic band structure to enhance superconductivity`,
  };
}

function createReactiveMillingRoute(formula: string, landscape: ThermodynamicLandscape, elements: string[]): NovelSynthesisRoute {
  const lightEls = elements.filter(e => ["B", "C", "N"].includes(e));

  return {
    method: "Reactive High-Energy Ball Milling under Controlled Atmosphere",
    steps: [
      `Load metal powders (${elements.filter(e => !["B", "C", "N", "O", "H"].includes(e)).join(", ")}) into WC vial with WC balls (BPR 20:1)`,
      `For ${lightEls.join("/")} incorporation: use ${lightEls.includes("N") ? "N2 gas atmosphere (5 atm)" : lightEls.includes("B") ? "amorphous B powder mixed with metals" : "graphite powder mixed with metals"}`,
      `Mill at 500 rpm for 50-100 hours with temperature monitoring (pause if vial exceeds 200C)`,
      `Sample at 10h intervals: XRD to track phase evolution, particle size by laser diffraction`,
      `When target phase appears in XRD: reduce milling speed to 200 rpm for 10h (strain relief without decomposition)`,
      `Extract powder under Ar; consolidate by hot isostatic pressing (HIP) at ${Math.round(landscape.maxSynthesisTemp * 0.5)}C, 200 MPa, 1h`,
      `Characterize: neutron diffraction (light element positions), TEM (nanostructure), R(T) on pressed pellet`,
    ],
    temperatureProfile: `RT (milling) -> ${Math.round(landscape.maxSynthesisTemp * 0.5)}C (HIP consolidation, 1h) -> RT`,
    pressureProfile: "Local impact pressure ~6 GPa (ball collisions); 200 MPa (HIP)",
    estimatedCoolingRate: "N/A (solid-state process; HIP furnace cool ~5 C/min)",
    atmosphere: lightEls.includes("N") ? "5 atm N2" : "High-purity Ar",
    expectedYieldRange: "50-75% (powder losses during handling)",
    noveltyScore: 0.75,
    synthesisConfidence: "medium",
    physicsJustification: `Reactive milling drives ${lightEls.join("/")} incorporation through repeated fracture-weld cycles at ball collision sites, where local temperatures briefly reach ~200-300C and pressures reach ~6 GPa — conditions sufficient to overcome formation barriers of ${landscape.decompositionBarrier.toFixed(3)} eV/atom. The nanocrystalline grain structure (typically 5-20 nm) created by milling introduces grain boundary phonon scattering that can enhance electron-phonon coupling. Extended milling times (50-100h) ensure complete reaction even for kinetically sluggish ${lightEls.join("/")} diffusion.`,
    source: "physics-reasoned",
    keyInnovation: `Mechanochemical synthesis pathway bypasses thermal decomposition by reacting ${lightEls.join("/")} into the metal matrix at room temperature through repeated high-energy impact events`,
  };
}

export async function learnFromReactionDatabase(formula: string): Promise<{
  similarRoutes: { formula: string; method: string; conditions: any; relevance: number }[];
  conditionTemplates: { tempRange: string; pressureRange: string; atmosphere: string; method: string }[];
}> {
  const elements = parseFormulaElements(formula);

  const allReactions = await storage.getChemicalReactions(100);
  const allSynthesis = await storage.getSynthesisProcesses(100);

  const similarRoutes: { formula: string; method: string; conditions: any; relevance: number }[] = [];

  for (const syn of allSynthesis) {
    if (!syn.formula) continue;
    const synElements = parseFormulaElements(syn.formula);
    const overlap = elements.filter(e => synElements.includes(e)).length;
    const relevance = overlap / Math.max(elements.length, synElements.length);
    if (relevance >= 0.4) {
      similarRoutes.push({
        formula: syn.formula,
        method: syn.method || "unknown",
        conditions: syn.conditions,
        relevance,
      });
    }
  }

  similarRoutes.sort((a, b) => b.relevance - a.relevance);

  const conditionTemplates: { tempRange: string; pressureRange: string; atmosphere: string; method: string }[] = [];
  for (const route of similarRoutes.slice(0, 5)) {
    const cond = route.conditions as any;
    if (cond) {
      conditionTemplates.push({
        tempRange: cond.temperature ? `${cond.temperature}C` : "unknown",
        pressureRange: cond.pressure ? `${cond.pressure} atm` : "ambient",
        atmosphere: cond.atmosphere || "not specified",
        method: route.method,
      });
    }
  }

  return { similarRoutes: similarRoutes.slice(0, 10), conditionTemplates };
}

export async function runSynthesisReasoning(
  emit: EventEmitter,
  candidate: SuperconductorCandidate
): Promise<NovelSynthesisRoute[] | null> {
  try {
    const landscape = await analyzeThermodynamicLandscape(candidate.formula, candidate);

    const routes = proposeNovelSynthesisRoutes(candidate.formula, landscape);
    if (routes.length === 0) return null;

    const dbKnowledge = await learnFromReactionDatabase(candidate.formula);

    for (const route of routes) {
      const matchingTemplates = dbKnowledge.conditionTemplates.filter(t =>
        t.method.toLowerCase().includes(route.method.split(":")[0].toLowerCase().trim().split(" ")[0])
      );
      if (matchingTemplates.length > 0) {
        route.synthesisConfidence = route.synthesisConfidence === "low" ? "medium" : "high";
      }
      if (dbKnowledge.similarRoutes.length > 3) {
        route.noveltyScore = Math.max(0.3, route.noveltyScore - 0.1);
      }
    }

    emit("log", {
      phase: "phase-13",
      event: "Novel synthesis path proposed",
      detail: `${candidate.formula} (${landscape.stabilityClass}, Ehull=${landscape.energyAboveHull.toFixed(3)} eV/atom): ${routes.length} physics-reasoned route(s) — ${routes.map(r => r.method.split(":")[0].trim()).join(", ")}`,
      dataSource: "Synthesis Reasoning",
    });

    return routes;
  } catch (err: any) {
    emit("log", {
      phase: "phase-13",
      event: "Synthesis reasoning error",
      detail: `${candidate.formula}: ${err.message?.slice(0, 100)}`,
      dataSource: "Synthesis Reasoning",
    });
    return null;
  }
}
