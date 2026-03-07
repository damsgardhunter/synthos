import type { ElectronicStructure, PhononSpectrum, ElectronPhononCoupling, NestingFunctionData, DynamicSpinSusceptibility } from "../learning/physics-engine";

export interface AdvancedPhysicsConstraints {
  fermiSurfaceNesting: {
    score: number;
    nestingStrength: string;
    preferredTopology: string;
    pocketCount: number;
    quasi2D: boolean;
    penalty: number;
  };
  orbitalHybridization: {
    score: number;
    hybridizationType: string;
    dpOverlap: number;
    spOverlap: number;
    sigmaContribution: number;
    penalty: number;
  };
  lifshitzProximity: {
    score: number;
    bandEdgeDistance: number;
    dosSpike: boolean;
    pocketTransition: boolean;
    penalty: number;
  };
  quantumCriticalFluctuation: {
    score: number;
    qcpType: string;
    magneticEnergyDiff: number;
    spinSusceptibilityPeak: number;
    isNearQCP: boolean;
    penalty: number;
  };
  electronicDimensionality: {
    score: number;
    anisotropy: number;
    dimensionClass: string;
    penalty: number;
  };
  phononSoftMode: {
    score: number;
    softModeFrequency: number;
    isStable: boolean;
    enhancementFactor: number;
    penalty: number;
  };
  chargeTransferEnergy: {
    score: number;
    delta: number;
    isOptimal: boolean;
    chargeTransferType: string;
    penalty: number;
  };
  latticePolarizability: {
    score: number;
    dielectricConstant: number;
    screeningStrength: string;
    penalty: number;
  };
  compositeScore: number;
  compositeBoost: number;
  summary: string;
}

export function computeAdvancedConstraints(
  formula: string,
  electronic: ElectronicStructure,
  phonon: PhononSpectrum,
  coupling: ElectronPhononCoupling,
  nestingData: NestingFunctionData,
  spinSusceptibility: DynamicSpinSusceptibility,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number,
  dimensionality: string
): AdvancedPhysicsConstraints {
  const fermiSurfaceNesting = computeFermiSurfaceNestingConstraint(
    electronic, nestingData, elements, counts, totalAtoms
  );
  const orbitalHybridization = computeOrbitalHybridizationConstraint(
    electronic, elements, counts, totalAtoms
  );
  const lifshitzProximity = computeLifshitzProximityConstraint(
    electronic, elements, counts, totalAtoms
  );
  const quantumCriticalFluctuation = computeQuantumCriticalConstraint(
    electronic, spinSusceptibility, elements, counts
  );
  const electronicDimensionality = computeElectronicDimensionalityConstraint(
    electronic, dimensionality, elements, counts, totalAtoms
  );
  const phononSoftMode = computePhononSoftModeConstraint(
    phonon, electronic, coupling
  );
  const chargeTransferEnergy = computeChargeTransferConstraint(
    electronic, elements, counts, totalAtoms
  );
  const latticePolarizability = computeLatticePolarizabilityConstraint(
    elements, counts, totalAtoms, phonon
  );

  const weights = {
    nesting: 0.15,
    hybridization: 0.15,
    lifshitz: 0.10,
    qcp: 0.12,
    dimensionality: 0.10,
    softMode: 0.13,
    chargeTransfer: 0.10,
    polarizability: 0.15,
  };

  const compositeScore =
    fermiSurfaceNesting.score * weights.nesting +
    orbitalHybridization.score * weights.hybridization +
    lifshitzProximity.score * weights.lifshitz +
    quantumCriticalFluctuation.score * weights.qcp +
    electronicDimensionality.score * weights.dimensionality +
    phononSoftMode.score * weights.softMode +
    chargeTransferEnergy.score * weights.chargeTransfer +
    latticePolarizability.score * weights.polarizability;

  const compositePenalty =
    fermiSurfaceNesting.penalty * weights.nesting +
    orbitalHybridization.penalty * weights.hybridization +
    lifshitzProximity.penalty * weights.lifshitz +
    quantumCriticalFluctuation.penalty * weights.qcp +
    electronicDimensionality.penalty * weights.dimensionality +
    phononSoftMode.penalty * weights.softMode +
    chargeTransferEnergy.penalty * weights.chargeTransfer +
    latticePolarizability.penalty * weights.polarizability;

  const compositeBoost = 1.0 + (compositeScore - 0.5) * 0.3 - compositePenalty * 0.2;
  const clampedBoost = Math.max(0.7, Math.min(1.4, compositeBoost));

  const topScores = [
    { name: "nesting", val: fermiSurfaceNesting.score },
    { name: "hybridization", val: orbitalHybridization.score },
    { name: "lifshitz", val: lifshitzProximity.score },
    { name: "QCP", val: quantumCriticalFluctuation.score },
    { name: "dimensionality", val: electronicDimensionality.score },
    { name: "soft-mode", val: phononSoftMode.score },
    { name: "charge-transfer", val: chargeTransferEnergy.score },
    { name: "polarizability", val: latticePolarizability.score },
  ].sort((a, b) => b.val - a.val);

  const top3 = topScores.slice(0, 3).map(s => `${s.name}=${s.val.toFixed(2)}`).join(", ");
  const summary = `Composite=${compositeScore.toFixed(3)}, boost=${clampedBoost.toFixed(3)} | Top: ${top3}`;

  return {
    fermiSurfaceNesting,
    orbitalHybridization,
    lifshitzProximity,
    quantumCriticalFluctuation,
    electronicDimensionality,
    phononSoftMode,
    chargeTransferEnergy,
    latticePolarizability,
    compositeScore: Number(compositeScore.toFixed(4)),
    compositeBoost: Number(clampedBoost.toFixed(4)),
    summary,
  };
}

function computeFermiSurfaceNestingConstraint(
  electronic: ElectronicStructure,
  nestingData: NestingFunctionData,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number
): AdvancedPhysicsConstraints["fermiSurfaceNesting"] {
  const nestingScore = electronic.nestingScore;
  const peakNesting = nestingData.peakNestingValue;
  const anisotropy = nestingData.nestingAnisotropy;
  const N_EF = electronic.densityOfStatesAtFermi;

  const fsTopology = electronic.fermiSurfaceTopology;
  const quasi2D = fsTopology.includes("2D") || fsTopology.includes("cylindrical");

  let pocketCount = 1;
  if (fsTopology.includes("multi-sheet") || fsTopology.includes("multi-band")) pocketCount = 3;
  if (fsTopology.includes("electron and hole")) pocketCount = 4;
  if (fsTopology.includes("nested multi-sheet")) pocketCount = 5;

  let score = 0;

  const relNesting = N_EF > 0 ? peakNesting / N_EF : 0;
  if (relNesting > 2.0) score += 0.35;
  else if (relNesting > 1.5) score += 0.25;
  else if (relNesting > 1.0) score += 0.15;

  score += nestingScore * 0.3;

  if (quasi2D) score += 0.15;
  if (pocketCount >= 3) score += 0.1;
  if (pocketCount >= 5) score += 0.05;

  if (anisotropy > 0.5) score += 0.05;

  score = Math.max(0, Math.min(1.0, score));

  let nestingStrength = "weak";
  if (score > 0.7) nestingStrength = "strong";
  else if (score > 0.4) nestingStrength = "moderate";

  let preferredTopology = "isotropic 3D";
  if (quasi2D && pocketCount >= 3) preferredTopology = "quasi-2D multi-pocket";
  else if (quasi2D) preferredTopology = "quasi-2D";
  else if (pocketCount >= 3) preferredTopology = "multi-pocket 3D";

  const penalty = score < 0.2 ? (0.2 - score) * 0.5 : 0;

  return {
    score: Number(score.toFixed(3)),
    nestingStrength,
    preferredTopology,
    pocketCount,
    quasi2D,
    penalty: Number(penalty.toFixed(3)),
  };
}

function computeOrbitalHybridizationConstraint(
  electronic: ElectronicStructure,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number
): AdvancedPhysicsConstraints["orbitalHybridization"] {
  const orb = electronic.orbitalFractions;

  const dpOverlap = Math.min(1.0, 2 * Math.sqrt(orb.d * orb.p));
  const spOverlap = Math.min(1.0, 2 * Math.sqrt(orb.s * orb.p));

  let sigmaContribution = 0;
  const hasB = elements.includes("B");
  const hasC = elements.includes("C");
  const hasN = elements.includes("N");
  const hasO = elements.includes("O");

  if (hasB && orb.p > 0.2) sigmaContribution = Math.min(1.0, orb.p * 2.5);
  else if (hasC && orb.p > 0.15) sigmaContribution = Math.min(1.0, orb.p * 2.0);
  else if ((hasN || hasO) && orb.p > 0.1) sigmaContribution = Math.min(1.0, orb.p * 1.5);

  let score = 0;

  score += dpOverlap * 0.35;
  score += spOverlap * 0.15;
  score += sigmaContribution * 0.2;

  const hasCuO = elements.includes("Cu") && hasO;
  const hasFeAs = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  if (hasCuO) score += 0.25;
  else if (hasFeAs) score += 0.2;
  else if (hasB && orb.p > 0.3) score += 0.15;

  if (orb.d > 0.3 && orb.p > 0.2) score += 0.05;

  score = Math.max(0, Math.min(1.0, score));

  let hybridizationType = "weak";
  if (hasCuO) hybridizationType = "Cu-d/O-p charge-transfer";
  else if (hasFeAs) hybridizationType = "Fe-d multi-orbital";
  else if (dpOverlap > 0.5) hybridizationType = "d-p hybridized";
  else if (sigmaContribution > 0.4) hybridizationType = "sigma-bond (MgB2-like)";
  else if (spOverlap > 0.4) hybridizationType = "s-p hybridized";
  else if (dpOverlap > 0.2 || spOverlap > 0.2) hybridizationType = "moderate";

  const penalty = score < 0.15 ? (0.15 - score) * 0.4 : 0;

  return {
    score: Number(score.toFixed(3)),
    hybridizationType,
    dpOverlap: Number(dpOverlap.toFixed(3)),
    spOverlap: Number(spOverlap.toFixed(3)),
    sigmaContribution: Number(sigmaContribution.toFixed(3)),
    penalty: Number(penalty.toFixed(3)),
  };
}

function computeLifshitzProximityConstraint(
  electronic: ElectronicStructure,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number
): AdvancedPhysicsConstraints["lifshitzProximity"] {
  const N_EF = electronic.densityOfStatesAtFermi;
  const vanHoveProx = electronic.vanHoveProximity;
  const flatBand = electronic.flatBandIndicator;
  const metallicity = electronic.metallicity;

  let bandEdgeDistance = 1.0;
  if (vanHoveProx > 0.7) bandEdgeDistance = 0.02 + (1 - vanHoveProx) * 0.2;
  else if (vanHoveProx > 0.4) bandEdgeDistance = 0.1 + (0.7 - vanHoveProx) * 0.5;
  else bandEdgeDistance = 0.5 + (0.4 - vanHoveProx);
  bandEdgeDistance = Math.max(0.01, Math.min(2.0, bandEdgeDistance));

  const dosSpike = N_EF > 3.0 && vanHoveProx > 0.5;

  const pocketTransition = (metallicity > 0.3 && metallicity < 0.7 && N_EF > 2.0)
    || vanHoveProx > 0.6;

  let score = 0;

  if (bandEdgeDistance < 0.1) score += 0.5;
  else if (bandEdgeDistance < 0.3) score += 0.3;
  else if (bandEdgeDistance < 0.5) score += 0.15;

  if (dosSpike) score += 0.2;
  if (pocketTransition) score += 0.15;
  if (flatBand > 0.5) score += 0.15;

  const isFeSe = elements.includes("Fe") && elements.includes("Se");
  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => {
    const d = getElementDataSafe(e);
    return d && d.group >= 3 && d.group <= 12;
  }).reduce((s, e) => s + (counts[e] || 0), 0);
  const hRatio = metalAtoms > 0 ? hCount / metalAtoms : 0;

  if (isFeSe) score += 0.1;
  if (hasH && hRatio >= 6) score += 0.1;

  score = Math.max(0, Math.min(1.0, score));

  const penalty = bandEdgeDistance > 0.5 ? Math.min(0.15, (bandEdgeDistance - 0.5) * 0.15) : 0;

  return {
    score: Number(score.toFixed(3)),
    bandEdgeDistance: Number(bandEdgeDistance.toFixed(3)),
    dosSpike,
    pocketTransition,
    penalty: Number(penalty.toFixed(3)),
  };
}

function computeQuantumCriticalConstraint(
  electronic: ElectronicStructure,
  spinSusc: DynamicSpinSusceptibility,
  elements: string[],
  counts: Record<string, number>
): AdvancedPhysicsConstraints["quantumCriticalFluctuation"] {
  const corr = electronic.correlationStrength;
  const isNearQCP = spinSusc.isNearQCP;
  const stonerEnh = spinSusc.stonerEnhancement;
  const chiPeak = spinSusc.chiStaticPeak;
  const sfEnergy = spinSusc.spinFluctuationEnergy;

  const magneticEnergyDiff = sfEnergy > 0 ? sfEnergy * 0.01 : 0.5;

  let score = 0;

  if (isNearQCP) score += 0.35;

  if (stonerEnh > 10) score += 0.25;
  else if (stonerEnh > 5) score += 0.15;
  else if (stonerEnh > 2) score += 0.05;

  if (sfEnergy < 5 && sfEnergy > 0) score += 0.15;
  else if (sfEnergy < 15) score += 0.08;

  if (corr > 0.6) score += 0.1;
  else if (corr > 0.4) score += 0.05;

  const hasCu = elements.includes("Cu");
  const hasO = elements.includes("O");
  const hasFe = elements.includes("Fe");
  const heavyFermionEls = ["Ce", "Yb", "U", "Pu"];
  const hasHeavyFermion = elements.some(e => heavyFermionEls.includes(e));

  if (hasCu && hasO && elements.length >= 3) score += 0.1;
  if (hasFe && (elements.includes("As") || elements.includes("Se"))) score += 0.1;
  if (hasHeavyFermion) score += 0.15;

  score = Math.max(0, Math.min(1.0, score));

  let qcpType = "none";
  if (hasHeavyFermion) qcpType = "magnetic-HF";
  else if (hasCu && hasO) qcpType = "doping-driven";
  else if (hasFe) qcpType = "magnetic-pnictide";
  else if (isNearQCP) qcpType = "magnetic-itinerant";
  else if (stonerEnh > 3) qcpType = "near-magnetic";

  const penalty = (!isNearQCP && stonerEnh < 2 && corr < 0.3) ? 0.05 : 0;

  return {
    score: Number(score.toFixed(3)),
    qcpType,
    magneticEnergyDiff: Number(magneticEnergyDiff.toFixed(4)),
    spinSusceptibilityPeak: Number(chiPeak.toFixed(3)),
    isNearQCP,
    penalty: Number(penalty.toFixed(3)),
  };
}

function computeElectronicDimensionalityConstraint(
  electronic: ElectronicStructure,
  dimensionality: string,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number
): AdvancedPhysicsConstraints["electronicDimensionality"] {
  const fsTopology = electronic.fermiSurfaceTopology;
  const nestingScore = electronic.nestingScore;
  const corr = electronic.correlationStrength;

  let anisotropy = 1.0;
  if (dimensionality === "quasi-2D" || dimensionality === "2D" || dimensionality === "layered") {
    anisotropy = 8.0;
    if (fsTopology.includes("cylindrical")) anisotropy = 12.0;
    if (corr > 0.6) anisotropy *= 1.3;
  } else if (dimensionality === "3D-HEA") {
    anisotropy = 1.2;
  } else {
    anisotropy = 1.5;
    if (fsTopology.includes("2D")) anisotropy = 6.0;
    else if (fsTopology.includes("layered") || fsTopology.includes("nesting")) anisotropy = 4.0;
  }

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  if (isCuprate) anisotropy = Math.max(anisotropy, 15);
  else if (isPnictide) anisotropy = Math.max(anisotropy, 8);

  let dimensionClass = "3D-isotropic";
  if (anisotropy > 10) dimensionClass = "strongly-2D";
  else if (anisotropy > 5) dimensionClass = "quasi-2D";
  else if (anisotropy > 3) dimensionClass = "moderately-anisotropic";

  let score = 0;

  if (anisotropy > 10) score += 0.4;
  else if (anisotropy > 5) score += 0.3;
  else if (anisotropy > 3) score += 0.15;
  else score += 0.05;

  if (nestingScore > 0.6) score += 0.2;
  else if (nestingScore > 0.3) score += 0.1;

  if (corr > 0.5 && anisotropy > 3) score += 0.15;

  if (isCuprate || isPnictide) score += 0.15;

  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  const metalAtoms = elements.filter(e => {
    const d = getElementDataSafe(e);
    return d && d.group >= 3 && d.group <= 12;
  }).reduce((s, e) => s + (counts[e] || 0), 0);
  if (hasH && metalAtoms > 0 && hCount / metalAtoms >= 6) {
    score += 0.1;
  }

  score = Math.max(0, Math.min(1.0, score));

  const penalty = (anisotropy < 2 && !hasH) ? 0.03 : 0;

  return {
    score: Number(score.toFixed(3)),
    anisotropy: Number(anisotropy.toFixed(2)),
    dimensionClass,
    penalty: Number(penalty.toFixed(3)),
  };
}

function computePhononSoftModeConstraint(
  phonon: PhononSpectrum,
  electronic: ElectronicStructure,
  coupling: ElectronPhononCoupling
): AdvancedPhysicsConstraints["phononSoftMode"] {
  const sms = phonon.softModeScore;
  const logAvg = phonon.logAverageFrequency;
  const maxFreq = phonon.maxPhononFrequency;
  const hasImaginary = phonon.hasImaginaryModes;
  const anharmonicity = phonon.anharmonicityIndex;

  const minFreqEstimate = logAvg * 0.3;
  const isStable = !hasImaginary && minFreqEstimate > 0;

  let score = 0;

  if (sms > 0.7) score += 0.4;
  else if (sms > 0.4) score += 0.25;
  else if (sms > 0.2) score += 0.1;

  if (isStable && sms > 0.3) {
    score += 0.2;
  }

  if (anharmonicity > 0.4 && anharmonicity < 0.8) score += 0.15;
  else if (anharmonicity > 0.2) score += 0.08;

  if (logAvg < 200 && logAvg > 20 && isStable) score += 0.1;

  const lambda = coupling.lambda;
  if (lambda > 1.0 && sms > 0.3) score += 0.1;

  score = Math.max(0, Math.min(1.0, score));

  let enhancementFactor = 1.0;
  if (sms > 0.5 && isStable) {
    enhancementFactor = 1.0 + (sms - 0.5) * 0.4;
  }

  let penalty = 0;
  if (hasImaginary) {
    penalty = 0.15;
  } else if (sms < 0.1) {
    penalty = 0.03;
  }

  return {
    score: Number(score.toFixed(3)),
    softModeFrequency: Number(minFreqEstimate.toFixed(1)),
    isStable,
    enhancementFactor: Number(enhancementFactor.toFixed(3)),
    penalty: Number(penalty.toFixed(3)),
  };
}

function computeChargeTransferConstraint(
  electronic: ElectronicStructure,
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number
): AdvancedPhysicsConstraints["chargeTransferEnergy"] {
  const orb = electronic.orbitalFractions;

  let dBandCenter = 0;
  let pBandCenter = 0;
  let hasTM = false;
  let hasAnion = false;

  for (const el of elements) {
    const d = getElementDataSafe(el);
    if (!d) continue;
    const frac = (counts[el] || 1) / totalAtoms;

    if (d.group >= 3 && d.group <= 12) {
      hasTM = true;
      const ie = d.ionizationEnergy || 700;
      dBandCenter += (ie / 1000) * frac;
    }

    if (["O", "N", "S", "Se", "Te", "F", "Cl", "Br", "I", "As", "P"].includes(el)) {
      hasAnion = true;
      const ea = d.electronAffinity || 150;
      pBandCenter += (ea / 500) * frac;
    }
  }

  let delta = Math.abs(dBandCenter - pBandCenter) * 3;
  if (!hasTM || !hasAnion) delta = 5.0;

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3;
  if (isCuprate) delta = 1.8;

  const isPnictide = elements.includes("Fe") && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  if (isPnictide) delta = 2.5;

  const isOptimal = delta > 0 && delta < 3.0;

  let score = 0;
  if (delta >= 0 && delta <= 3.0) {
    if (delta >= 1.0 && delta <= 2.5) score = 0.8 + (1 - Math.abs(delta - 1.75) / 0.75) * 0.2;
    else if (delta < 1.0) score = 0.4 + delta * 0.4;
    else score = 0.4 + (3.0 - delta) * 0.4;
  } else {
    score = Math.max(0, 0.3 - (delta - 3.0) * 0.1);
  }

  score = Math.max(0, Math.min(1.0, score));

  let chargeTransferType = "none";
  if (isCuprate) chargeTransferType = "Mott-Hubbard/charge-transfer";
  else if (isPnictide) chargeTransferType = "pnictide-correlated";
  else if (isOptimal && hasTM) chargeTransferType = "d-p charge-transfer";
  else if (hasTM && hasAnion) chargeTransferType = "weak charge-transfer";

  const penalty = delta > 5.0 ? 0.05 : 0;

  return {
    score: Number(score.toFixed(3)),
    delta: Number(delta.toFixed(3)),
    isOptimal,
    chargeTransferType,
    penalty: Number(penalty.toFixed(3)),
  };
}

function computeLatticePolarizabilityConstraint(
  elements: string[],
  counts: Record<string, number>,
  totalAtoms: number,
  phonon: PhononSpectrum
): AdvancedPhysicsConstraints["latticePolarizability"] {
  let epsilon = 5.0;

  const highDielectricElements: Record<string, number> = {
    "Ba": 45, "Sr": 40, "Ti": 35, "Pb": 30, "Bi": 25,
    "La": 22, "K": 20, "Ca": 18, "Zr": 16, "Hf": 15,
    "Na": 12, "Ta": 14, "Nb": 12, "Y": 10, "Sc": 8,
  };

  const highDielectricAnions: Record<string, number> = {
    "O": 2.5, "F": 1.8, "S": 1.5, "Se": 1.3, "Te": 1.2,
  };

  let ionicContrib = 0;
  let anionContrib = 0;

  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    if (highDielectricElements[el]) {
      ionicContrib += highDielectricElements[el] * frac;
    }
    if (highDielectricAnions[el]) {
      anionContrib += highDielectricAnions[el] * frac;
    }
  }

  epsilon = 5.0 + ionicContrib * anionContrib;

  if (phonon.softModeScore > 0.5) {
    epsilon *= (1 + phonon.softModeScore * 0.5);
  }

  if (phonon.anharmonicityIndex > 0.3) {
    epsilon *= (1 + phonon.anharmonicityIndex * 0.3);
  }

  const hasH = elements.includes("H");
  const hCount = counts["H"] || 0;
  if (hasH && hCount / totalAtoms > 0.3) {
    epsilon += 8;
  }

  const isSrTiO3 = elements.includes("Sr") && elements.includes("Ti") && elements.includes("O");
  if (isSrTiO3) epsilon = Math.max(epsilon, 300);

  const isBaTiO3 = elements.includes("Ba") && elements.includes("Ti") && elements.includes("O");
  if (isBaTiO3) epsilon = Math.max(epsilon, 1000);

  epsilon = Math.max(1, Math.min(10000, epsilon));

  let score = 0;
  if (epsilon > 100) score = 0.9;
  else if (epsilon > 50) score = 0.7 + (epsilon - 50) / 250;
  else if (epsilon > 20) score = 0.5 + (epsilon - 20) / 150;
  else if (epsilon > 10) score = 0.2 + (epsilon - 10) / 33;
  else score = epsilon * 0.02;

  score = Math.max(0, Math.min(1.0, score));

  let screeningStrength = "weak";
  if (epsilon > 100) screeningStrength = "very strong";
  else if (epsilon > 50) screeningStrength = "strong";
  else if (epsilon > 20) screeningStrength = "moderate";

  const penalty = epsilon < 5 ? 0.05 : 0;

  return {
    score: Number(score.toFixed(3)),
    dielectricConstant: Number(epsilon.toFixed(1)),
    screeningStrength,
    penalty: Number(penalty.toFixed(3)),
  };
}

interface SafeElementData {
  group: number;
  ionizationEnergy?: number;
  electronAffinity?: number;
}

const ELEMENT_GROUPS: Record<string, number> = {
  "H": 1, "He": 18, "Li": 1, "Be": 2, "B": 13, "C": 14, "N": 15, "O": 16, "F": 17, "Ne": 18,
  "Na": 1, "Mg": 2, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
  "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
  "Cu": 11, "Zn": 12, "Ga": 13, "Ge": 14, "As": 15, "Se": 16, "Br": 17, "Kr": 18,
  "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10,
  "Ag": 11, "Cd": 12, "In": 13, "Sn": 14, "Sb": 15, "Te": 16, "I": 17, "Xe": 18,
  "Cs": 1, "Ba": 2, "La": 3, "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10,
  "Au": 11, "Hg": 12, "Tl": 13, "Pb": 14, "Bi": 15, "Po": 16, "At": 17, "Rn": 18,
  "Ce": 3, "Pr": 3, "Nd": 3, "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3,
  "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
  "Th": 3, "U": 3, "Pu": 3, "Np": 3,
};

const IONIZATION_ENERGIES: Record<string, number> = {
  "H": 1312, "Li": 520, "Be": 900, "B": 801, "C": 1086, "N": 1402, "O": 1314, "F": 1681,
  "Na": 496, "Mg": 738, "Al": 578, "Si": 786, "P": 1012, "S": 1000, "Cl": 1251,
  "K": 419, "Ca": 590, "Sc": 633, "Ti": 659, "V": 651, "Cr": 653, "Mn": 717,
  "Fe": 762, "Co": 760, "Ni": 737, "Cu": 745, "Zn": 906,
  "Ga": 579, "Ge": 762, "As": 947, "Se": 941, "Br": 1140,
  "Rb": 403, "Sr": 550, "Y": 600, "Zr": 640, "Nb": 652, "Mo": 684,
  "Ru": 710, "Rh": 720, "Pd": 804, "Ag": 731, "Cd": 868,
  "In": 558, "Sn": 709, "Sb": 834, "Te": 869, "I": 1008,
  "Cs": 376, "Ba": 503, "La": 538, "Hf": 659, "Ta": 761, "W": 770,
  "Re": 760, "Os": 840, "Ir": 880, "Pt": 870, "Au": 890, "Hg": 1007,
  "Tl": 589, "Pb": 716, "Bi": 703,
  "Ce": 534, "Pr": 527, "Nd": 533, "Yb": 603, "U": 598, "Th": 587, "Pu": 585,
};

const ELECTRON_AFFINITIES: Record<string, number> = {
  "O": 141, "F": 328, "S": 200, "Cl": 349, "Se": 195, "Br": 325, "Te": 190, "I": 295,
  "N": 7, "P": 72, "As": 78, "C": 122, "B": 27,
};

function getElementDataSafe(el: string): SafeElementData | null {
  const group = ELEMENT_GROUPS[el];
  if (group === undefined) return null;
  return {
    group,
    ionizationEnergy: IONIZATION_ENERGIES[el],
    electronAffinity: ELECTRON_AFFINITIES[el],
  };
}
