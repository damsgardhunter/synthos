import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  isActinide,
  getCompositionWeightedProperty,
  getDebyeTemperature,
} from "../learning/elemental-data";
import { estimateBandwidthW } from "../learning/physics-engine";

export interface AtomicFeatures {
  atomic_mass_avg: number;
  bond_length_distribution: number;
  coordination_number: number;
  charge_transfer: number;
  atomic_radius_variance: number;
  electronegativity_spread: number;
}

export interface ElectronicFeatures {
  DOS_EF: number;
  bandwidth: number;
  van_hove_distance: number;
  orbital_character: { s: number; p: number; d: number; f: number };
  fermi_surface_shape: number;
  nesting_score: number;
  band_flatness: number;
  mott_proximity: number;
}

export interface MesoscopicFeatures {
  layeredness: number;
  lattice_anisotropy: number;
  strain_sensitivity: number;
  defect_tolerance: number;
  interlayer_coupling: number;
  dimensionality: number;
}

export interface MultiScaleFeatures {
  formula: string;
  atomic: AtomicFeatures;
  electronic: ElectronicFeatures;
  mesoscopic: MesoscopicFeatures;
}

export interface CrossScaleCoupling {
  electron_phonon_mass_ratio: number;
  strain_band_shift: number;
  layer_coupling_strength: number;
  bond_stiffness_vs_phonon: number;
  orbital_phonon_coupling: number;
  charge_transfer_vs_nesting: number;
}

export interface SensitivityResult {
  formula: string;
  atomic_sensitivity: Record<string, number>;
  electronic_sensitivity: Record<string, number>;
  mesoscopic_sensitivity: Record<string, number>;
  dominant_scale: "atomic" | "electronic" | "mesoscopic";
  scale_importances: { atomic: number; electronic: number; mesoscopic: number };
}

function parseFormulaElements(formula: string): string[] {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const matches = cleaned.match(/[A-Z][a-z]*/g);
  return matches ? Array.from(new Set(matches)) : [];
}

function parseFormulaCounts(formula: string): Record<string, number> {
  if (typeof formula !== "string") formula = String(formula ?? "");
  const cleaned = formula.replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)));
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function getTotalAtoms(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  return total > 0 ? total : 1;
}

function round4(v: number): number {
  return Number(v.toFixed(4));
}

function computeAtomicFeatures(formula: string): AtomicFeatures {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  let massSum = 0;
  let radiusValues: number[] = [];
  let enValues: number[] = [];

  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    massSum += data.atomicMass * frac;
    const count = Math.round(counts[el] || 1);
    for (let i = 0; i < count; i++) {
      radiusValues.push(data.atomicRadius);
      if (data.paulingElectronegativity !== null) {
        enValues.push(data.paulingElectronegativity);
      }
    }
  }

  const radiusMean = radiusValues.length > 0
    ? radiusValues.reduce((s, r) => s + r, 0) / radiusValues.length
    : 150;
  const radiusVariance = radiusValues.length > 1
    ? radiusValues.reduce((s, r) => s + (r - radiusMean) ** 2, 0) / radiusValues.length
    : 0;

  const enSpread = enValues.length >= 2
    ? Math.max(...enValues) - Math.min(...enValues)
    : 0;

  let chargeTransfer = 0;
  if (enValues.length >= 2) {
    const enAvg = enValues.reduce((s, v) => s + v, 0) / enValues.length;
    chargeTransfer = enValues.reduce((s, v) => s + Math.abs(v - enAvg), 0) / enValues.length;
  }

  const avgBondLength = radiusMean * 2 * 0.85 / 100;
  const bondLengthDistribution = Math.sqrt(radiusVariance) / radiusMean;

  let coordNumber = 6;
  const hasO = elements.includes("O");
  const hasTM = elements.some(e => isTransitionMetal(e));
  if (hasO && hasTM) coordNumber = 6;
  else if (elements.length === 1) {
    const data = getElementData(elements[0]);
    if (data && data.latticeConstant) {
      coordNumber = data.atomicNumber <= 20 ? 8 : 12;
    }
  } else if (elements.length >= 4) {
    coordNumber = 4 + elements.length;
  }

  return {
    atomic_mass_avg: round4(massSum),
    bond_length_distribution: round4(bondLengthDistribution),
    coordination_number: round4(Math.min(12, coordNumber)),
    charge_transfer: round4(chargeTransfer),
    atomic_radius_variance: round4(radiusVariance),
    electronegativity_spread: round4(enSpread),
  };
}

function computeElectronicFeatures(formula: string): ElectronicFeatures {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const gammaAvg = getCompositionWeightedProperty(counts, "sommerfeldGamma");
  let dosEF = 1.0;
  if (gammaAvg !== null && gammaAvg > 0) {
    dosEF = gammaAvg / 2.359;
  } else {
    let totalVE = 0;
    for (const el of elements) {
      const data = getElementData(el);
      if (data) totalVE += data.valenceElectrons * (counts[el] || 1);
    }
    const vec = totalVE / totalAtoms;
    let wAvg = 0;
    for (const el of elements) {
      const frac = (counts[el] || 1) / totalAtoms;
      wAvg += estimateBandwidthW(el) * frac;
    }
    wAvg = Math.max(1.0, wAvg);
    dosEF = vec / (2 * wAvg);
  }
  dosEF = Math.max(0.1, Math.min(10, dosEF));

  let bandwidth = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    bandwidth += estimateBandwidthW(el) * frac;
  }
  bandwidth = Math.max(1.0, bandwidth);

  let sFrac = 0, pFrac = 0, dFrac = 0, fFrac = 0;
  let orbWeightTotal = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data) continue;
    const frac = (counts[el] || 1) / totalAtoms;
    const ve = data.valenceElectrons;
    if (isRareEarth(el) || isActinide(el)) {
      fFrac += frac * 0.5;
      dFrac += frac * 0.3;
      sFrac += frac * 0.2;
    } else if (isTransitionMetal(el)) {
      dFrac += frac * 0.7;
      sFrac += frac * 0.2;
      pFrac += frac * 0.1;
    } else if (ve <= 2) {
      sFrac += frac * 0.8;
      pFrac += frac * 0.2;
    } else {
      pFrac += frac * 0.6;
      sFrac += frac * 0.4;
    }
    orbWeightTotal += frac;
  }
  if (orbWeightTotal > 0) {
    sFrac /= orbWeightTotal;
    pFrac /= orbWeightTotal;
    dFrac /= orbWeightTotal;
    fFrac /= orbWeightTotal;
  }

  const vanHoveDistance = dFrac > 0.3
    ? Math.max(0.01, 0.5 - dFrac * 0.8)
    : 0.5 + (1 - dFrac) * 0.5;

  const bandFlatness = dFrac > 0.4
    ? Math.min(1.0, dFrac * 1.2 + fFrac * 0.5)
    : Math.min(0.5, dFrac * 0.8 + fFrac * 1.5);

  const nonmetals = ["H", "He", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Se", "Br", "Kr", "Te", "I", "Xe"];
  const metalFrac = elements.filter(e => !nonmetals.includes(e))
    .reduce((s, e) => s + (counts[e] || 0), 0) / totalAtoms;

  let fermiSurfaceShape = 3.0;
  if (dFrac > 0.4 && metalFrac > 0.3) fermiSurfaceShape = 2.0 + (1.0 - dFrac);
  if (elements.includes("Cu") && elements.includes("O")) fermiSurfaceShape = 2.0;

  let nestingScore = 0;
  if (dFrac > 0.3 && metalFrac > 0.3) {
    nestingScore = Math.min(1.0, dFrac * 0.6 + (dosEF > 2 ? 0.2 : 0));
  }
  if (elements.includes("Fe") && (elements.includes("As") || elements.includes("Se"))) {
    nestingScore = Math.max(nestingScore, 0.7);
  }

  let mottProximity = 0;
  for (const el of elements) {
    const data = getElementData(el);
    if (!data || !data.hubbardU) continue;
    const W = estimateBandwidthW(el);
    const uOverW = data.hubbardU / W;
    const frac = (counts[el] || 1) / totalAtoms;
    mottProximity += uOverW * frac;
  }
  mottProximity = Math.min(1.0, mottProximity);

  return {
    DOS_EF: round4(dosEF),
    bandwidth: round4(bandwidth),
    van_hove_distance: round4(vanHoveDistance),
    orbital_character: {
      s: round4(sFrac),
      p: round4(pFrac),
      d: round4(dFrac),
      f: round4(fFrac),
    },
    fermi_surface_shape: round4(fermiSurfaceShape),
    nesting_score: round4(nestingScore),
    band_flatness: round4(bandFlatness),
    mott_proximity: round4(mottProximity),
  };
}

function computeMesoscopicFeatures(formula: string): MesoscopicFeatures {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);

  const isCuprate = elements.includes("Cu") && elements.includes("O") && elements.length >= 3
    && elements.some(e => isRareEarth(e) || ["Ba", "Sr", "Ca", "Bi", "Tl", "Hg", "Y", "La"].includes(e));
  const isIronPnictide = elements.includes("Fe")
    && (elements.includes("As") || elements.includes("Se") || elements.includes("P"));
  const isTMD = elements.some(e => ["Mo", "W", "Nb", "Ta"].includes(e))
    && elements.some(e => ["S", "Se", "Te"].includes(e));

  let layeredness = 0;
  if (isCuprate) layeredness = 0.9;
  else if (isIronPnictide) layeredness = 0.8;
  else if (isTMD) layeredness = 0.95;
  else if (elements.includes("O") && elements.length >= 3) layeredness = 0.4;
  else layeredness = 0.1;

  let latticeAnisotropy = layeredness * 0.8;
  if (elements.length === 1) {
    const data = getElementData(elements[0]);
    if (data && data.latticeConstant) {
      latticeAnisotropy = 0.1;
    }
  }
  if (isCuprate || isIronPnictide) latticeAnisotropy = Math.max(latticeAnisotropy, 0.7);

  let bulkModValues: number[] = [];
  for (const el of elements) {
    const data = getElementData(el);
    if (data && data.bulkModulus) {
      bulkModValues.push(data.bulkModulus);
    }
  }
  const avgBulkMod = bulkModValues.length > 0
    ? bulkModValues.reduce((s, b) => s + b, 0) / bulkModValues.length
    : 100;
  const strainSensitivity = Math.min(1.0, 1.0 / (1.0 + avgBulkMod / 100));

  let defectTolerance = 0.5;
  if (elements.length >= 4) defectTolerance += 0.2;
  if (elements.includes("O")) defectTolerance += 0.1;
  if (isCuprate) defectTolerance = 0.7;
  defectTolerance = Math.min(1.0, defectTolerance);

  let interlayerCoupling = 1.0 - layeredness * 0.7;
  if (isCuprate) interlayerCoupling = 0.2;
  if (isIronPnictide) interlayerCoupling = 0.3;
  if (isTMD) interlayerCoupling = 0.15;

  let dimensionality = 3.0;
  if (layeredness > 0.7) dimensionality = 2.0;
  else if (layeredness > 0.4) dimensionality = 2.5;

  return {
    layeredness: round4(layeredness),
    lattice_anisotropy: round4(latticeAnisotropy),
    strain_sensitivity: round4(strainSensitivity),
    defect_tolerance: round4(defectTolerance),
    interlayer_coupling: round4(interlayerCoupling),
    dimensionality: round4(dimensionality),
  };
}

export function computeMultiScaleFeatures(formula: string): MultiScaleFeatures {
  return {
    formula,
    atomic: computeAtomicFeatures(formula),
    electronic: computeElectronicFeatures(formula),
    mesoscopic: computeMesoscopicFeatures(formula),
  };
}

export function computeCrossScaleCoupling(multiScale: MultiScaleFeatures): CrossScaleCoupling {
  const { atomic, electronic, mesoscopic } = multiScale;

  const electronPhononMassRatio = atomic.atomic_mass_avg > 0
    ? electronic.DOS_EF / (atomic.atomic_mass_avg * 0.01)
    : 0;

  const strainBandShift = mesoscopic.strain_sensitivity * electronic.bandwidth * 0.1;

  const layerCouplingStrength = mesoscopic.interlayer_coupling * electronic.DOS_EF
    * (1.0 - mesoscopic.lattice_anisotropy * 0.5);

  const avgDebye = getAverageDebye(multiScale.formula);
  const phononFreqEstimate = avgDebye * 8.617e-5;
  const bondStiffnessVsPhonon = atomic.coordination_number > 0
    ? phononFreqEstimate / (atomic.bond_length_distribution + 0.01) * 0.1
    : 0;

  const orbitalPhononCoupling = (electronic.orbital_character.d + electronic.orbital_character.f * 0.5)
    * (1.0 - electronic.mott_proximity * 0.3)
    * Math.max(0.1, 1.0 / (atomic.atomic_mass_avg * 0.01));

  const chargeTransferVsNesting = atomic.charge_transfer * electronic.nesting_score;

  return {
    electron_phonon_mass_ratio: round4(Math.min(10, electronPhononMassRatio)),
    strain_band_shift: round4(Math.min(5, strainBandShift)),
    layer_coupling_strength: round4(Math.min(5, layerCouplingStrength)),
    bond_stiffness_vs_phonon: round4(Math.min(10, bondStiffnessVsPhonon)),
    orbital_phonon_coupling: round4(Math.min(5, orbitalPhononCoupling)),
    charge_transfer_vs_nesting: round4(Math.min(1, chargeTransferVsNesting)),
  };
}

function getAverageDebye(formula: string): number {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = getTotalAtoms(counts);
  let sum = 0;
  let count = 0;
  for (const el of elements) {
    const td = getDebyeTemperature(el);
    if (td !== null) {
      sum += td * (counts[el] || 1) / totalAtoms;
      count++;
    }
  }
  return count > 0 ? sum : 300;
}

export function runSensitivityAnalysis(formula: string): SensitivityResult {
  const base = computeMultiScaleFeatures(formula);
  const baseCoupling = computeCrossScaleCoupling(base);
  const baseScore = couplingToScore(baseCoupling);

  const atomicSensitivity: Record<string, number> = {};
  const atomicKeys: (keyof AtomicFeatures)[] = [
    "atomic_mass_avg", "bond_length_distribution", "coordination_number",
    "charge_transfer", "atomic_radius_variance", "electronegativity_spread",
  ];
  for (const key of atomicKeys) {
    const perturbed = { ...base, atomic: { ...base.atomic, [key]: base.atomic[key] * 1.1 + 0.01 } };
    const perturbedCoupling = computeCrossScaleCoupling(perturbed);
    const perturbedScore = couplingToScore(perturbedCoupling);
    const diff = Math.abs(perturbedScore - baseScore);
    atomicSensitivity[key] = round4(diff / (Math.abs(baseScore) + 0.01));
  }

  const electronicSensitivity: Record<string, number> = {};
  const electronicKeys: (keyof Omit<ElectronicFeatures, "orbital_character">)[] = [
    "DOS_EF", "bandwidth", "van_hove_distance",
    "fermi_surface_shape", "nesting_score", "band_flatness", "mott_proximity",
  ];
  for (const key of electronicKeys) {
    const perturbed = { ...base, electronic: { ...base.electronic, [key]: (base.electronic[key] as number) * 1.1 + 0.01 } };
    const perturbedCoupling = computeCrossScaleCoupling(perturbed);
    const perturbedScore = couplingToScore(perturbedCoupling);
    const diff = Math.abs(perturbedScore - baseScore);
    electronicSensitivity[key] = round4(diff / (Math.abs(baseScore) + 0.01));
  }

  const mesoscopicSensitivity: Record<string, number> = {};
  const mesoscopicKeys: (keyof MesoscopicFeatures)[] = [
    "layeredness", "lattice_anisotropy", "strain_sensitivity",
    "defect_tolerance", "interlayer_coupling", "dimensionality",
  ];
  for (const key of mesoscopicKeys) {
    const perturbed = { ...base, mesoscopic: { ...base.mesoscopic, [key]: base.mesoscopic[key] * 1.1 + 0.01 } };
    const perturbedCoupling = computeCrossScaleCoupling(perturbed);
    const perturbedScore = couplingToScore(perturbedCoupling);
    const diff = Math.abs(perturbedScore - baseScore);
    mesoscopicSensitivity[key] = round4(diff / (Math.abs(baseScore) + 0.01));
  }

  const atomicTotal = Object.values(atomicSensitivity).reduce((s, v) => s + v, 0);
  const electronicTotal = Object.values(electronicSensitivity).reduce((s, v) => s + v, 0);
  const mesoscopicTotal = Object.values(mesoscopicSensitivity).reduce((s, v) => s + v, 0);
  const grandTotal = atomicTotal + electronicTotal + mesoscopicTotal + 1e-10;

  const scaleImportances = {
    atomic: round4(atomicTotal / grandTotal),
    electronic: round4(electronicTotal / grandTotal),
    mesoscopic: round4(mesoscopicTotal / grandTotal),
  };

  let dominantScale: "atomic" | "electronic" | "mesoscopic" = "electronic";
  if (scaleImportances.atomic >= scaleImportances.electronic && scaleImportances.atomic >= scaleImportances.mesoscopic) {
    dominantScale = "atomic";
  } else if (scaleImportances.mesoscopic >= scaleImportances.electronic) {
    dominantScale = "mesoscopic";
  }

  return {
    formula,
    atomic_sensitivity: atomicSensitivity,
    electronic_sensitivity: electronicSensitivity,
    mesoscopic_sensitivity: mesoscopicSensitivity,
    dominant_scale: dominantScale,
    scale_importances: scaleImportances,
  };
}

function couplingToScore(coupling: CrossScaleCoupling): number {
  return coupling.electron_phonon_mass_ratio * 0.25
    + coupling.strain_band_shift * 0.1
    + coupling.layer_coupling_strength * 0.15
    + coupling.bond_stiffness_vs_phonon * 0.15
    + coupling.orbital_phonon_coupling * 0.25
    + coupling.charge_transfer_vs_nesting * 0.1;
}
