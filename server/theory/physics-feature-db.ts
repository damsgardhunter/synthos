import {
  computeElectronicStructure,
  computePhononSpectrum,
  computeElectronPhononCoupling,
  assessCorrelationStrength,
  computeDimensionalityScore,
  parseFormulaElements,
  estimateBandwidthW,
} from "../learning/physics-engine";
import {
  getElementData,
  isTransitionMetal,
  isRareEarth,
  getStonerParameter,
} from "../learning/elemental-data";
import { detectQuantumCriticality } from "../physics/quantum-criticality";
import { computePairingProfile } from "../physics/pairing-mechanisms";

export interface FeatureRecord {
  materialId: string;
  formula: string;
  featureVector: PhysicsFeatureVector;
  tc: number | null;
  pairingStrength: number | null;
  stability: number | null;
  timestamp: number;
}

export interface PhysicsFeatureVector {
  DOS_EF: number;
  van_hove_distance: number;
  band_flatness: number;
  fermi_surface_dimensionality: number;
  phonon_log_frequency: number;
  electron_phonon_lambda: number;
  nesting_score: number;
  orbital_degeneracy: number;
  charge_transfer: number;
  lattice_anisotropy: number;
  mott_proximity: number;
  spin_fluctuation: number;
  cdw_proximity: number;
  quantum_critical_score: number;
  pairing_strength: number;
  hydrogen_density: number;
  correlation_strength: number;
  bandwidth: number;
  debye_temp: number;
  anharmonicity: number;
}

export const FEATURE_NAMES: (keyof PhysicsFeatureVector)[] = [
  "DOS_EF",
  "van_hove_distance",
  "band_flatness",
  "fermi_surface_dimensionality",
  "phonon_log_frequency",
  "electron_phonon_lambda",
  "nesting_score",
  "orbital_degeneracy",
  "charge_transfer",
  "lattice_anisotropy",
  "mott_proximity",
  "spin_fluctuation",
  "cdw_proximity",
  "quantum_critical_score",
  "pairing_strength",
  "hydrogen_density",
  "correlation_strength",
  "bandwidth",
  "debye_temp",
  "anharmonicity",
];

const MAX_CACHE_SIZE = 2000;

class PhysicsFeatureDB {
  private records: Map<string, FeatureRecord> = new Map();
  private accessOrder: string[] = [];

  private evictIfNeeded(): void {
    while (this.records.size >= MAX_CACHE_SIZE && this.accessOrder.length > 0) {
      const oldest = this.accessOrder.shift()!;
      this.records.delete(oldest);
    }
  }

  private touch(materialId: string): void {
    const idx = this.accessOrder.indexOf(materialId);
    if (idx !== -1) {
      this.accessOrder.splice(idx, 1);
    }
    this.accessOrder.push(materialId);
  }

  addFeatureRecord(record: FeatureRecord): void {
    this.evictIfNeeded();
    this.records.set(record.materialId, record);
    this.touch(record.materialId);
  }

  getFeatureRecord(materialId: string): FeatureRecord | null {
    const record = this.records.get(materialId);
    if (!record) return null;
    this.touch(materialId);
    return record;
  }

  getFeatureDataset(): FeatureRecord[] {
    return Array.from(this.records.values());
  }

  getDatasetSize(): number {
    return this.records.size;
  }
}

const featureDB = new PhysicsFeatureDB();

export function addFeatureRecord(record: FeatureRecord): void {
  featureDB.addFeatureRecord(record);
}

export function getFeatureRecord(materialId: string): FeatureRecord | null {
  return featureDB.getFeatureRecord(materialId);
}

export function getFeatureDataset(): FeatureRecord[] {
  return featureDB.getFeatureDataset();
}

export function getDatasetSize(): number {
  return featureDB.getDatasetSize();
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

export function computeFeatureVector(formula: string): PhysicsFeatureVector {
  const elements = parseFormulaElements(formula);
  const counts = parseFormulaCounts(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + n, 0) || 1;

  const electronic = computeElectronicStructure(formula, null);
  const phonon = computePhononSpectrum(formula, electronic);
  const coupling = computeElectronPhononCoupling(electronic, phonon, formula, 0);
  const correlation = assessCorrelationStrength(formula);

  const DOS_EF = electronic.densityOfStatesAtFermi;
  const van_hove_distance = electronic.vanHoveProximity ?? 0;
  const band_flatness = electronic.bandFlatness ?? 0;

  let fermi_surface_dimensionality = 3;
  const fsTopo = electronic.fermiSurfaceTopology;
  if (fsTopo.includes("2D")) fermi_surface_dimensionality = 2;
  else if (fsTopo.includes("quasi")) fermi_surface_dimensionality = 2.5;
  else if (fsTopo.includes("multi")) fermi_surface_dimensionality = 2.5;

  const phonon_log_frequency = coupling.omegaLog;
  const electron_phonon_lambda = coupling.lambda;
  const nesting_score = electronic.nestingScore ?? 0;

  const dFrac = electronic.orbitalFractions?.d ?? 0;
  const fFrac = electronic.orbitalFractions?.f ?? 0;
  let orbital_degeneracy = 0;
  for (const el of elements) {
    if (isTransitionMetal(el)) {
      const data = getElementData(el);
      if (data && data.valenceElectrons >= 2 && data.valenceElectrons <= 8) {
        orbital_degeneracy += (counts[el] || 1) / totalAtoms;
      }
    }
    if (isRareEarth(el)) {
      orbital_degeneracy += (counts[el] || 1) / totalAtoms * 0.7;
    }
  }
  orbital_degeneracy = Math.min(5, orbital_degeneracy * 3 + dFrac * 2 + fFrac);

  let charge_transfer = 0;
  if (elements.length > 1) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const en_i = getElementData(elements[i])?.paulingElectronegativity ?? 1.5;
        const en_j = getElementData(elements[j])?.paulingElectronegativity ?? 1.5;
        const frac_i = (counts[elements[i]] || 1) / totalAtoms;
        const frac_j = (counts[elements[j]] || 1) / totalAtoms;
        charge_transfer += Math.abs(en_i - en_j) * Math.sqrt(frac_i * frac_j);
      }
    }
  }

  const lattice_anisotropy = 1.0 - computeDimensionalityScore(formula);

  const mott_proximity = electronic.mottProximityScore ?? 0;

  let spin_fluctuation = 0;
  for (const el of elements) {
    const stonerI = getStonerParameter(el);
    if (stonerI !== null && stonerI > 0) {
      const frac = (counts[el] || 1) / totalAtoms;
      const stonerProduct = stonerI * DOS_EF;
      spin_fluctuation = Math.max(spin_fluctuation, stonerProduct * frac);
    }
  }
  spin_fluctuation = Math.min(1.0, spin_fluctuation);

  let cdw_proximity = 0;
  if (nesting_score > 0.3 && DOS_EF > 1.0) {
    cdw_proximity = Math.min(1.0, nesting_score * DOS_EF * 0.2);
  }
  if (phonon.softModePresent) {
    cdw_proximity = Math.max(cdw_proximity, phonon.softModeScore * 0.5);
  }

  let quantum_critical_score = 0;
  try {
    const qc = detectQuantumCriticality(formula, { electronic, phonon, coupling });
    quantum_critical_score = qc.quantumCriticalScore;
  } catch {
    quantum_critical_score = 0;
  }

  let pairing_strength = 0;
  try {
    const pairing = computePairingProfile(formula);
    pairing_strength = pairing.compositePairingStrength;
  } catch {
    pairing_strength = Math.min(1.0, electron_phonon_lambda / 3.0);
  }

  const hCount = counts["H"] || 0;
  const hydrogen_density = hCount / totalAtoms;

  const correlation_strength = correlation.ratio;

  let bandwidth = 0;
  for (const el of elements) {
    const frac = (counts[el] || 1) / totalAtoms;
    bandwidth += estimateBandwidthW(el) * frac;
  }
  bandwidth = Math.max(1.0, bandwidth);

  const debye_temp = phonon.debyeTemperature;
  const anharmonicity = phonon.anharmonicityIndex;

  return {
    DOS_EF: Number(DOS_EF.toFixed(4)),
    van_hove_distance: Number(van_hove_distance.toFixed(4)),
    band_flatness: Number(band_flatness.toFixed(4)),
    fermi_surface_dimensionality: Number(fermi_surface_dimensionality.toFixed(2)),
    phonon_log_frequency: Number(phonon_log_frequency.toFixed(2)),
    electron_phonon_lambda: Number(electron_phonon_lambda.toFixed(4)),
    nesting_score: Number(nesting_score.toFixed(4)),
    orbital_degeneracy: Number(orbital_degeneracy.toFixed(4)),
    charge_transfer: Number(charge_transfer.toFixed(4)),
    lattice_anisotropy: Number(lattice_anisotropy.toFixed(4)),
    mott_proximity: Number(mott_proximity.toFixed(4)),
    spin_fluctuation: Number(spin_fluctuation.toFixed(4)),
    cdw_proximity: Number(cdw_proximity.toFixed(4)),
    quantum_critical_score: Number(quantum_critical_score.toFixed(4)),
    pairing_strength: Number(pairing_strength.toFixed(4)),
    hydrogen_density: Number(hydrogen_density.toFixed(4)),
    correlation_strength: Number(correlation_strength.toFixed(4)),
    bandwidth: Number(bandwidth.toFixed(4)),
    debye_temp: Number(debye_temp.toFixed(2)),
    anharmonicity: Number(anharmonicity.toFixed(4)),
  };
}

export function buildAndStoreFeatureRecord(
  formula: string,
  tc: number | null = null,
  pairingStrength: number | null = null,
  stability: number | null = null,
): FeatureRecord {
  const materialId = formula.replace(/\s+/g, "");
  const featureVector = computeFeatureVector(formula);

  const record: FeatureRecord = {
    materialId,
    formula,
    featureVector,
    tc,
    pairingStrength: pairingStrength ?? featureVector.pairing_strength,
    stability,
    timestamp: Date.now(),
  };

  addFeatureRecord(record);
  return record;
}

export function featureVectorToArray(vec: PhysicsFeatureVector): number[] {
  return FEATURE_NAMES.map(name => vec[name]);
}

export function arrayToFeatureVector(arr: number[]): PhysicsFeatureVector {
  const vec: any = {};
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    vec[FEATURE_NAMES[i]] = arr[i] ?? 0;
  }
  return vec as PhysicsFeatureVector;
}
