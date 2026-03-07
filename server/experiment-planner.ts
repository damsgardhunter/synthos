export interface ExperimentCandidate {
  formula: string;
  predictedTc: number;
  stability: number;
  synthesisFeasibility: number;
  novelty: number;
  uncertainty: number;
  materialClass: string;
  crystalStructure: string;
  pairingMechanism?: string;
  synthesisPath?: {
    steps: { method: string; temperature: number; pressure: number; duration: number; atmosphere: string; notes: string }[];
    totalDuration: number;
    feasibilityScore: number;
  };
}

export interface RankedCandidate {
  formula: string;
  experimentScore: number;
  breakdown: {
    tcScore: number;
    stabilityScore: number;
    feasibilityScore: number;
    noveltyScore: number;
    uncertaintyBonus: number;
  };
  rank: number;
}

export interface LabInstructions {
  material: string;
  synthesis: {
    pressure: number;
    temperature: number;
    cooling: string;
    anneal: string;
  };
  precursors: string[];
  equipment: string[];
  safetyNotes: string[];
  estimatedDuration: string;
}

export interface CharacterizationMethod {
  method: string;
  priority: number;
  reason: string;
  estimatedTime: string;
  requiredEquipment: string;
}

export interface ExperimentPlan {
  formula: string;
  predictedTc: number;
  ranking: RankedCandidate;
  labInstructions: LabInstructions;
  characterization: CharacterizationMethod[];
  timeline: string;
  riskAssessment: string;
}

interface PlannerStats {
  totalPlansGenerated: number;
  totalMethodsSuggested: number;
  methodFrequency: Record<string, number>;
  topCandidates: { formula: string; score: number; predictedTc: number }[];
  avgExperimentScore: number;
  totalScoreAccumulated: number;
}

const stats: PlannerStats = {
  totalPlansGenerated: 0,
  totalMethodsSuggested: 0,
  methodFrequency: {},
  topCandidates: [],
  avgExperimentScore: 0,
  totalScoreAccumulated: 0,
};

const ELEMENT_PRECURSORS: Record<string, string[]> = {
  Ba: ["BaCO3", "BaO"],
  Sr: ["SrCO3", "SrO"],
  Ca: ["CaCO3", "CaO"],
  La: ["La2O3"],
  Y: ["Y2O3"],
  Cu: ["CuO", "Cu metal powder"],
  Fe: ["Fe powder", "Fe2O3"],
  Nb: ["Nb powder", "Nb2O5"],
  Ti: ["TiO2", "Ti powder"],
  Mg: ["MgB2 precursor", "Mg powder"],
  B: ["B powder", "amorphous boron"],
  H: ["H2 gas"],
  N: ["N2 gas", "NH3"],
  O: ["O2 gas"],
  C: ["graphite", "carbon black"],
  Al: ["Al powder", "Al2O3"],
  Ir: ["Ir powder", "IrO2"],
  Pd: ["Pd powder"],
  Pt: ["Pt powder"],
  V: ["V powder", "V2O5"],
  Zr: ["Zr powder", "ZrO2"],
  Hf: ["Hf powder", "HfO2"],
  S: ["S powder"],
  Se: ["Se powder"],
  Te: ["Te powder"],
  As: ["As powder"],
  P: ["P red", "P black"],
  Bi: ["Bi2O3", "Bi powder"],
  Pb: ["PbO", "Pb powder"],
  Tl: ["Tl2O3"],
  Hg: ["HgO"],
  Li: ["Li metal", "Li2CO3"],
  K: ["K2CO3"],
  Cs: ["Cs2CO3"],
  Rb: ["Rb2CO3"],
};

function parseElements(formula: string): string[] {
  const matches = formula.match(/[A-Z][a-z]?/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function getPrecursors(formula: string): string[] {
  const elements = parseElements(formula);
  const precursors: string[] = [];
  for (const el of elements) {
    const options = ELEMENT_PRECURSORS[el];
    if (options && options.length > 0) {
      precursors.push(options[0]);
    } else {
      precursors.push(`${el} powder (high purity)`);
    }
  }
  return precursors;
}

function getEquipment(materialClass: string, pressure: number, temperature: number): string[] {
  const equipment: string[] = ["analytical balance", "mortar and pestle or ball mill"];
  const mc = materialClass.toLowerCase();

  if (pressure > 100) {
    equipment.push("diamond anvil cell (DAC)");
    equipment.push("laser heating system");
  } else if (pressure > 10) {
    equipment.push("multi-anvil press");
  } else if (pressure > 1) {
    equipment.push("piston-cylinder apparatus");
  }

  if (temperature > 1500) {
    equipment.push("high-temperature tube furnace (>1500K)");
  } else {
    equipment.push("tube furnace");
  }

  if (mc.includes("cuprate") || mc.includes("oxide")) {
    equipment.push("oxygen flow controller");
  }

  if (mc.includes("hydride")) {
    equipment.push("hydrogen gas handling system");
    equipment.push("gas pressure regulator");
  }

  if (mc.includes("pnictide") || mc.includes("iron")) {
    equipment.push("arc melting furnace");
    equipment.push("sealed quartz tubes");
  }

  equipment.push("glove box (argon atmosphere)");
  equipment.push("X-ray diffractometer");

  return equipment;
}

function getSafetyNotes(formula: string, materialClass: string, pressure: number): string[] {
  const notes: string[] = [];
  const mc = materialClass.toLowerCase();
  const elements = parseElements(formula);

  if (pressure > 50) {
    notes.push("EXTREME PRESSURE: Follow diamond anvil cell safety protocols. Risk of catastrophic failure.");
  } else if (pressure > 10) {
    notes.push("HIGH PRESSURE: Ensure all pressure vessels are certified and inspected.");
  }

  if (mc.includes("hydride") || elements.includes("H")) {
    notes.push("HYDROGEN: Explosive gas. Ensure proper ventilation, leak detection, and no ignition sources.");
  }

  if (elements.includes("Tl")) {
    notes.push("THALLIUM: Highly toxic. Use full PPE, fume hood, and follow institutional toxicity protocols.");
  }
  if (elements.includes("Hg")) {
    notes.push("MERCURY: Toxic heavy metal. Handle in fume hood with mercury spill kit available.");
  }
  if (elements.includes("Pb")) {
    notes.push("LEAD: Toxic. Avoid ingestion/inhalation. Use PPE and proper waste disposal.");
  }
  if (elements.includes("As")) {
    notes.push("ARSENIC: Highly toxic. Handle in fume hood with full PPE.");
  }
  if (elements.includes("Se")) {
    notes.push("SELENIUM: Toxic fumes at high temperature. Work in well-ventilated fume hood.");
  }
  if (elements.includes("F")) {
    notes.push("FLUORINE: Highly corrosive and toxic. Specialized handling required.");
  }

  notes.push("Wear appropriate PPE: lab coat, safety glasses, heat-resistant gloves.");
  notes.push("Follow institutional chemical hygiene plan.");

  return notes;
}

export function rankCandidatesForExperiment(candidates: ExperimentCandidate[]): RankedCandidate[] {
  const maxTc = Math.max(1, ...candidates.map(c => c.predictedTc));

  const ranked = candidates.map(c => {
    const tcScore = Math.min(1, c.predictedTc / maxTc);
    const stabilityScore = Math.min(1, Math.max(0, c.stability));
    const feasibilityScore = Math.min(1, Math.max(0, c.synthesisFeasibility));
    const noveltyScore = Math.min(1, Math.max(0, c.novelty));
    const uncertaintyBonus = Math.min(1, Math.max(0, c.uncertainty)) * 0.5;

    const experimentScore =
      0.3 * tcScore +
      0.25 * stabilityScore +
      0.2 * feasibilityScore +
      0.15 * noveltyScore +
      0.1 * uncertaintyBonus;

    return {
      formula: c.formula,
      experimentScore,
      breakdown: {
        tcScore,
        stabilityScore,
        feasibilityScore,
        noveltyScore,
        uncertaintyBonus,
      },
      rank: 0,
    };
  });

  ranked.sort((a, b) => b.experimentScore - a.experimentScore);
  ranked.forEach((r, i) => { r.rank = i + 1; });

  return ranked;
}

export function generateLabInstructions(
  formula: string,
  synthesisPath: ExperimentCandidate["synthesisPath"] | undefined,
  predictedTc: number,
  crystalStructure: string
): LabInstructions {
  const elements = parseElements(formula);
  const isHydride = elements.includes("H") || crystalStructure.toLowerCase().includes("clathrate");
  const isCuprate = elements.includes("Cu") && elements.includes("O");
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("P"));

  let materialClass = "intermetallic";
  if (isHydride) materialClass = "hydride";
  else if (isCuprate) materialClass = "cuprate";
  else if (isPnictide) materialClass = "iron-pnictide";
  else if (elements.includes("B")) materialClass = "boride";
  else if (elements.includes("N") && !elements.includes("O")) materialClass = "nitride";
  else if (elements.includes("C") && !elements.includes("O")) materialClass = "carbide";

  let pressure = 0;
  let temperature = 1000;
  let coolingDescription = "Cool at 10 K/s under argon";
  let annealDescription = "Anneal at 700K for 8 hours";

  if (synthesisPath && synthesisPath.steps.length > 0) {
    pressure = Math.max(...synthesisPath.steps.map(s => s.pressure));
    temperature = Math.max(...synthesisPath.steps.map(s => s.temperature));

    const quenchStep = synthesisPath.steps.find(s => s.method.toLowerCase().includes("quench"));
    if (quenchStep) {
      coolingDescription = quenchStep.notes || `Rapid quench from ${temperature}K`;
    }

    const annealStep = synthesisPath.steps.find(s => s.method.toLowerCase().includes("anneal"));
    if (annealStep) {
      annealDescription = annealStep.notes || `Anneal at ${annealStep.temperature}K for ${annealStep.duration}h`;
    }
  } else {
    if (isHydride) {
      pressure = Math.max(50, predictedTc / 2);
      temperature = 800;
      coolingDescription = "Rapid quench at 1000 K/s to preserve high-pressure phase";
      annealDescription = "Brief anneal at 300K for 1 hour under hydrogen";
    } else if (isCuprate) {
      temperature = 950;
      coolingDescription = "Slow cool at 1 K/s under flowing oxygen";
      annealDescription = "Oxygen anneal at 500K for 48 hours";
    } else if (isPnictide) {
      temperature = 1200;
      coolingDescription = "Quench from reaction temperature at 200 K/s";
      annealDescription = "Anneal in sealed quartz tube at 800K for 72 hours";
    }
  }

  const estimatedHours = (synthesisPath?.totalDuration ?? (pressure > 50 ? 72 : 24));
  let durationStr: string;
  if (estimatedHours < 24) durationStr = `${Math.round(estimatedHours)} hours`;
  else durationStr = `${Math.round(estimatedHours / 24)} days`;

  return {
    material: formula,
    synthesis: {
      pressure,
      temperature,
      cooling: coolingDescription,
      anneal: annealDescription,
    },
    precursors: getPrecursors(formula),
    equipment: getEquipment(materialClass, pressure, temperature),
    safetyNotes: getSafetyNotes(formula, materialClass, pressure),
    estimatedDuration: durationStr,
  };
}

export function suggestCharacterizationMethods(
  formula: string,
  predictedTc: number,
  crystalStructure: string,
  pairingMechanism?: string
): CharacterizationMethod[] {
  const methods: CharacterizationMethod[] = [];
  const pm = (pairingMechanism || "").toLowerCase();
  const cs = (crystalStructure || "").toLowerCase();

  methods.push({
    method: "X-ray Diffraction (XRD)",
    priority: 1,
    reason: "Confirm crystal structure and phase purity",
    estimatedTime: "2-4 hours",
    requiredEquipment: "Powder X-ray diffractometer",
  });

  methods.push({
    method: "Resistivity vs Temperature",
    priority: 2,
    reason: `Verify superconducting transition at predicted Tc = ${predictedTc.toFixed(1)}K`,
    estimatedTime: "4-8 hours",
    requiredEquipment: "Four-probe resistivity measurement system with cryostat",
  });

  methods.push({
    method: "Magnetic Susceptibility (SQUID)",
    priority: 3,
    reason: "Confirm Meissner effect and determine superconducting volume fraction",
    estimatedTime: "4-6 hours",
    requiredEquipment: "SQUID magnetometer",
  });

  methods.push({
    method: "Specific Heat Measurement",
    priority: 4,
    reason: "Measure jump in specific heat at Tc to confirm bulk superconductivity",
    estimatedTime: "8-12 hours",
    requiredEquipment: "Physical Property Measurement System (PPMS)",
  });

  if (pm.includes("phonon") || pm.includes("electron-phonon") || pm.includes("conventional")) {
    methods.push({
      method: "Isotope Effect Measurement",
      priority: 5,
      reason: "Verify phonon-mediated pairing via isotope substitution",
      estimatedTime: "1-2 weeks (requires isotope samples)",
      requiredEquipment: "Isotope-substituted samples + resistivity setup",
    });
  }

  if (pm.includes("spin") || pm.includes("magnetic") || pm.includes("unconventional")) {
    methods.push({
      method: "Inelastic Neutron Scattering",
      priority: 5,
      reason: "Probe spin fluctuation spectrum and magnetic excitations",
      estimatedTime: "2-5 days (beam time)",
      requiredEquipment: "Neutron source (reactor or spallation)",
    });
  }

  if (cs.includes("layered") || cs.includes("tetragonal") || pm.includes("d-wave")) {
    methods.push({
      method: "Angle-Resolved Photoemission (ARPES)",
      priority: 5,
      reason: "Map electronic band structure and superconducting gap symmetry",
      estimatedTime: "1-3 days (beam time)",
      requiredEquipment: "Synchrotron beamline with ARPES endstation",
    });
  }

  methods.push({
    method: "Scanning Tunneling Microscopy (STM)",
    priority: 6,
    reason: "Measure local density of states and gap magnitude",
    estimatedTime: "1-3 days",
    requiredEquipment: "Low-temperature STM",
  });

  if (predictedTc > 77) {
    methods.push({
      method: "Upper Critical Field (Hc2) Measurement",
      priority: 6,
      reason: "Determine Hc2(T) for high-Tc material to assess application potential",
      estimatedTime: "1-2 days",
      requiredEquipment: "High-field magnet system + transport measurement",
    });
  }

  methods.push({
    method: "Energy-Dispersive X-ray Spectroscopy (EDX)",
    priority: 7,
    reason: "Verify elemental composition and stoichiometry",
    estimatedTime: "1-2 hours",
    requiredEquipment: "SEM with EDX detector",
  });

  methods.push({
    method: "Scanning Electron Microscopy (SEM)",
    priority: 8,
    reason: "Examine grain morphology and microstructure",
    estimatedTime: "2-4 hours",
    requiredEquipment: "Scanning electron microscope",
  });

  if (predictedTc > 100 || pm.includes("unconventional")) {
    methods.push({
      method: "Muon Spin Rotation (μSR)",
      priority: 7,
      reason: "Probe magnetic penetration depth and pairing symmetry",
      estimatedTime: "2-5 days (beam time)",
      requiredEquipment: "Muon source facility",
    });
  }

  if (cs.includes("clathrate") || cs.includes("cage")) {
    methods.push({
      method: "Raman Spectroscopy",
      priority: 6,
      reason: "Probe phonon modes in cage/clathrate structure",
      estimatedTime: "2-4 hours",
      requiredEquipment: "Raman spectrometer",
    });
  }

  methods.sort((a, b) => a.priority - b.priority);

  stats.totalMethodsSuggested += methods.length;
  for (const m of methods) {
    stats.methodFrequency[m.method] = (stats.methodFrequency[m.method] || 0) + 1;
  }

  return methods;
}

export function generateExperimentPlan(candidate: ExperimentCandidate): ExperimentPlan {
  const ranked = rankCandidatesForExperiment([candidate]);
  const ranking = ranked[0];

  const labInstructions = generateLabInstructions(
    candidate.formula,
    candidate.synthesisPath,
    candidate.predictedTc,
    candidate.crystalStructure
  );

  const characterization = suggestCharacterizationMethods(
    candidate.formula,
    candidate.predictedTc,
    candidate.crystalStructure,
    candidate.pairingMechanism
  );

  const synthDays = candidate.synthesisPath
    ? Math.ceil(candidate.synthesisPath.totalDuration / 24)
    : (labInstructions.synthesis.pressure > 50 ? 5 : 2);
  const charDays = Math.ceil(characterization.length * 0.5);
  const totalWeeks = Math.ceil((synthDays + charDays + 3) / 7);
  const timeline = `Estimated ${totalWeeks} week(s): ${synthDays} day(s) synthesis + ${charDays} day(s) characterization + 3 day(s) analysis`;

  let riskLevel: string;
  if (candidate.stability > 0.7 && candidate.synthesisFeasibility > 0.6) {
    riskLevel = "LOW";
  } else if (candidate.stability > 0.4 && candidate.synthesisFeasibility > 0.3) {
    riskLevel = "MODERATE";
  } else {
    riskLevel = "HIGH";
  }

  const risks: string[] = [];
  if (candidate.stability < 0.5) risks.push("thermodynamic instability may prevent phase formation");
  if (candidate.synthesisFeasibility < 0.3) risks.push("synthesis conditions are extremely challenging");
  if (labInstructions.synthesis.pressure > 100) risks.push("requires diamond anvil cell at extreme pressures");
  if (candidate.predictedTc > 200) risks.push("predicted Tc is exceptionally high - verify prediction methodology");
  if (risks.length === 0) risks.push("standard laboratory risks apply");

  const riskAssessment = `Risk level: ${riskLevel}. ${risks.join("; ")}.`;

  stats.totalPlansGenerated++;
  stats.totalScoreAccumulated += ranking.experimentScore;
  stats.avgExperimentScore = stats.totalScoreAccumulated / stats.totalPlansGenerated;

  const topEntry = { formula: candidate.formula, score: ranking.experimentScore, predictedTc: candidate.predictedTc };
  stats.topCandidates.push(topEntry);
  stats.topCandidates.sort((a, b) => b.score - a.score);
  if (stats.topCandidates.length > 20) stats.topCandidates = stats.topCandidates.slice(0, 20);

  return {
    formula: candidate.formula,
    predictedTc: candidate.predictedTc,
    ranking,
    labInstructions,
    characterization,
    timeline,
    riskAssessment,
  };
}

export function getExperimentPlannerStats() {
  return {
    totalPlansGenerated: stats.totalPlansGenerated,
    totalMethodsSuggested: stats.totalMethodsSuggested,
    methodFrequency: { ...stats.methodFrequency },
    topCandidates: [...stats.topCandidates],
    avgExperimentScore: stats.avgExperimentScore,
  };
}
