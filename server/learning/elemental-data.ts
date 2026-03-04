export interface ElementalProperties {
  symbol: string;
  atomicNumber: number;
  atomicMass: number;
  debyeTemperature: number | null;
  bulkModulus: number | null;
  stonerParameter: number | null;
  hubbardU: number | null;
  mcMillanHopfieldEta: number | null;
  miedemaPhiStar: number | null;
  miedemaNws13: number | null;
  miedemaV23: number | null;
  sommerfeldGamma: number | null;
  gruneisenParameter: number | null;
  atomicRadius: number;
  pettiforScale: number;
  valenceElectrons: number;
  paulingElectronegativity: number | null;
  electronAffinity: number | null;
  firstIonizationEnergy: number;
  elementalTc: number | null;
}

export const ELEMENTAL_DATA: Record<string, ElementalProperties> = {
  H: {
    symbol: "H", atomicNumber: 1, atomicMass: 1.008,
    debyeTemperature: 122, bulkModulus: 0.2, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 5.2, miedemaNws13: 1.5, miedemaV23: 2.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 25, pettiforScale: 103, valenceElectrons: 1,
    paulingElectronegativity: 2.20, electronAffinity: 0.754, firstIonizationEnergy: 13.598,
    elementalTc: null
  },
  He: {
    symbol: "He", atomicNumber: 2, atomicMass: 4.003,
    debyeTemperature: 25, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: null, miedemaNws13: null, miedemaV23: null,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 31, pettiforScale: 1, valenceElectrons: 0,
    paulingElectronegativity: null, electronAffinity: 0.0, firstIonizationEnergy: 24.587,
    elementalTc: null
  },
  Li: {
    symbol: "Li", atomicNumber: 3, atomicMass: 6.941,
    debyeTemperature: 344, bulkModulus: 11.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.35, miedemaPhiStar: 2.85, miedemaNws13: 0.98, miedemaV23: 5.5,
    sommerfeldGamma: 1.63, gruneisenParameter: 1.13,
    atomicRadius: 152, pettiforScale: 12, valenceElectrons: 1,
    paulingElectronegativity: 0.98, electronAffinity: 0.618, firstIonizationEnergy: 5.392,
    elementalTc: null
  },
  Be: {
    symbol: "Be", atomicNumber: 4, atomicMass: 9.012,
    debyeTemperature: 1440, bulkModulus: 130.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.22, miedemaPhiStar: 5.05, miedemaNws13: 1.60, miedemaV23: 3.2,
    sommerfeldGamma: 0.17, gruneisenParameter: 1.16,
    atomicRadius: 112, pettiforScale: 77, valenceElectrons: 2,
    paulingElectronegativity: 1.57, electronAffinity: 0.0, firstIonizationEnergy: 9.323,
    elementalTc: 0.026
  },
  B: {
    symbol: "B", atomicNumber: 5, atomicMass: 10.81,
    debyeTemperature: 1250, bulkModulus: 185.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 8.5, miedemaPhiStar: 5.30, miedemaNws13: 1.55, miedemaV23: 3.3,
    sommerfeldGamma: null, gruneisenParameter: 1.0,
    atomicRadius: 87, pettiforScale: 86, valenceElectrons: 3,
    paulingElectronegativity: 2.04, electronAffinity: 0.277, firstIonizationEnergy: 8.298,
    elementalTc: null
  },
  C: {
    symbol: "C", atomicNumber: 6, atomicMass: 12.011,
    debyeTemperature: 2230, bulkModulus: 442.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 6.25, miedemaNws13: 1.90, miedemaV23: 2.6,
    sommerfeldGamma: null, gruneisenParameter: 1.0,
    atomicRadius: 77, pettiforScale: 95, valenceElectrons: 4,
    paulingElectronegativity: 2.55, electronAffinity: 1.263, firstIonizationEnergy: 11.260,
    elementalTc: null
  },
  N: {
    symbol: "N", atomicNumber: 7, atomicMass: 14.007,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 6.86, miedemaNws13: 1.92, miedemaV23: 2.2,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 75, pettiforScale: 100, valenceElectrons: 5,
    paulingElectronegativity: 3.04, electronAffinity: -0.07, firstIonizationEnergy: 14.534,
    elementalTc: null
  },
  O: {
    symbol: "O", atomicNumber: 8, atomicMass: 15.999,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 6.98, miedemaNws13: 1.70, miedemaV23: 2.2,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 73, pettiforScale: 101, valenceElectrons: 6,
    paulingElectronegativity: 3.44, electronAffinity: 1.461, firstIonizationEnergy: 13.618,
    elementalTc: null
  },
  F: {
    symbol: "F", atomicNumber: 9, atomicMass: 18.998,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 7.30, miedemaNws13: 1.80, miedemaV23: 2.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 72, pettiforScale: 102, valenceElectrons: 7,
    paulingElectronegativity: 3.98, electronAffinity: 3.401, firstIonizationEnergy: 17.422,
    elementalTc: null
  },
  Ne: {
    symbol: "Ne", atomicNumber: 10, atomicMass: 20.180,
    debyeTemperature: 75, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: null, miedemaNws13: null, miedemaV23: null,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 38, pettiforScale: 2, valenceElectrons: 0,
    paulingElectronegativity: null, electronAffinity: 0.0, firstIonizationEnergy: 21.565,
    elementalTc: null
  },
  Na: {
    symbol: "Na", atomicNumber: 11, atomicMass: 22.990,
    debyeTemperature: 158, bulkModulus: 6.3, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.12, miedemaPhiStar: 2.70, miedemaNws13: 0.82, miedemaV23: 8.3,
    sommerfeldGamma: 1.38, gruneisenParameter: 1.25,
    atomicRadius: 186, pettiforScale: 11, valenceElectrons: 1,
    paulingElectronegativity: 0.93, electronAffinity: 0.548, firstIonizationEnergy: 5.139,
    elementalTc: null
  },
  Mg: {
    symbol: "Mg", atomicNumber: 12, atomicMass: 24.305,
    debyeTemperature: 400, bulkModulus: 35.4, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.18, miedemaPhiStar: 3.45, miedemaNws13: 1.17, miedemaV23: 5.8,
    sommerfeldGamma: 1.3, gruneisenParameter: 1.52,
    atomicRadius: 160, pettiforScale: 73, valenceElectrons: 2,
    paulingElectronegativity: 1.31, electronAffinity: 0.0, firstIonizationEnergy: 7.646,
    elementalTc: null
  },
  Al: {
    symbol: "Al", atomicNumber: 13, atomicMass: 26.982,
    debyeTemperature: 428, bulkModulus: 76.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.56, miedemaPhiStar: 4.20, miedemaNws13: 1.39, miedemaV23: 4.6,
    sommerfeldGamma: 1.35, gruneisenParameter: 2.17,
    atomicRadius: 143, pettiforScale: 80, valenceElectrons: 3,
    paulingElectronegativity: 1.61, electronAffinity: 0.441, firstIonizationEnergy: 5.986,
    elementalTc: 1.18
  },
  Si: {
    symbol: "Si", atomicNumber: 14, atomicMass: 28.086,
    debyeTemperature: 645, bulkModulus: 97.8, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.70, miedemaNws13: 1.50, miedemaV23: 4.2,
    sommerfeldGamma: null, gruneisenParameter: 1.0,
    atomicRadius: 117, pettiforScale: 87, valenceElectrons: 4,
    paulingElectronegativity: 1.90, electronAffinity: 1.385, firstIonizationEnergy: 8.152,
    elementalTc: null
  },
  P: {
    symbol: "P", atomicNumber: 15, atomicMass: 30.974,
    debyeTemperature: null, bulkModulus: 11.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 5.55, miedemaNws13: 1.65, miedemaV23: 3.5,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 110, pettiforScale: 91, valenceElectrons: 5,
    paulingElectronegativity: 2.19, electronAffinity: 0.746, firstIonizationEnergy: 10.487,
    elementalTc: null
  },
  S: {
    symbol: "S", atomicNumber: 16, atomicMass: 32.065,
    debyeTemperature: null, bulkModulus: 7.7, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 5.60, miedemaNws13: 1.50, miedemaV23: 3.6,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 104, pettiforScale: 96, valenceElectrons: 6,
    paulingElectronegativity: 2.58, electronAffinity: 2.077, firstIonizationEnergy: 10.360,
    elementalTc: null
  },
  Cl: {
    symbol: "Cl", atomicNumber: 17, atomicMass: 35.453,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 6.10, miedemaNws13: 1.65, miedemaV23: 3.2,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 99, pettiforScale: 99, valenceElectrons: 7,
    paulingElectronegativity: 3.16, electronAffinity: 3.617, firstIonizationEnergy: 12.968,
    elementalTc: null
  },
  Ar: {
    symbol: "Ar", atomicNumber: 18, atomicMass: 39.948,
    debyeTemperature: 92, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: null, miedemaNws13: null, miedemaV23: null,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 71, pettiforScale: 3, valenceElectrons: 0,
    paulingElectronegativity: null, electronAffinity: 0.0, firstIonizationEnergy: 15.760,
    elementalTc: null
  },
  K: {
    symbol: "K", atomicNumber: 19, atomicMass: 39.098,
    debyeTemperature: 91, bulkModulus: 3.1, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.08, miedemaPhiStar: 2.25, miedemaNws13: 0.65, miedemaV23: 12.8,
    sommerfeldGamma: 2.08, gruneisenParameter: 1.34,
    atomicRadius: 227, pettiforScale: 10, valenceElectrons: 1,
    paulingElectronegativity: 0.82, electronAffinity: 0.501, firstIonizationEnergy: 4.341,
    elementalTc: null
  },
  Ca: {
    symbol: "Ca", atomicNumber: 20, atomicMass: 40.078,
    debyeTemperature: 230, bulkModulus: 17.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.20, miedemaPhiStar: 2.55, miedemaNws13: 0.91, miedemaV23: 8.8,
    sommerfeldGamma: 2.72, gruneisenParameter: 1.04,
    atomicRadius: 197, pettiforScale: 16, valenceElectrons: 2,
    paulingElectronegativity: 1.00, electronAffinity: 0.025, firstIonizationEnergy: 6.113,
    elementalTc: null
  },
  Sc: {
    symbol: "Sc", atomicNumber: 21, atomicMass: 44.956,
    debyeTemperature: 360, bulkModulus: 56.6, stonerParameter: 0.20, hubbardU: 2.0,
    mcMillanHopfieldEta: 1.1, miedemaPhiStar: 3.25, miedemaNws13: 1.27, miedemaV23: 6.1,
    sommerfeldGamma: 10.7, gruneisenParameter: 0.93,
    atomicRadius: 162, pettiforScale: 19, valenceElectrons: 3,
    paulingElectronegativity: 1.36, electronAffinity: 0.188, firstIonizationEnergy: 6.562,
    elementalTc: null
  },
  Ti: {
    symbol: "Ti", atomicNumber: 22, atomicMass: 47.867,
    debyeTemperature: 420, bulkModulus: 110.0, stonerParameter: 0.32, hubbardU: 3.0,
    mcMillanHopfieldEta: 1.8, miedemaPhiStar: 3.65, miedemaNws13: 1.47, miedemaV23: 4.8,
    sommerfeldGamma: 3.35, gruneisenParameter: 1.23,
    atomicRadius: 147, pettiforScale: 51, valenceElectrons: 4,
    paulingElectronegativity: 1.54, electronAffinity: 0.079, firstIonizationEnergy: 6.828,
    elementalTc: 0.40
  },
  V: {
    symbol: "V", atomicNumber: 23, atomicMass: 50.942,
    debyeTemperature: 380, bulkModulus: 162.0, stonerParameter: 0.42, hubbardU: 3.1,
    mcMillanHopfieldEta: 3.5, miedemaPhiStar: 4.25, miedemaNws13: 1.64, miedemaV23: 4.1,
    sommerfeldGamma: 9.26, gruneisenParameter: 1.34,
    atomicRadius: 134, pettiforScale: 52, valenceElectrons: 5,
    paulingElectronegativity: 1.63, electronAffinity: 0.525, firstIonizationEnergy: 6.746,
    elementalTc: 5.40
  },
  Cr: {
    symbol: "Cr", atomicNumber: 24, atomicMass: 51.996,
    debyeTemperature: 630, bulkModulus: 160.0, stonerParameter: 0.38, hubbardU: 3.0,
    mcMillanHopfieldEta: 0.5, miedemaPhiStar: 4.65, miedemaNws13: 1.73, miedemaV23: 3.7,
    sommerfeldGamma: 1.40, gruneisenParameter: 1.21,
    atomicRadius: 128, pettiforScale: 54, valenceElectrons: 6,
    paulingElectronegativity: 1.66, electronAffinity: 0.666, firstIonizationEnergy: 6.767,
    elementalTc: null
  },
  Mn: {
    symbol: "Mn", atomicNumber: 25, atomicMass: 54.938,
    debyeTemperature: 410, bulkModulus: 120.0, stonerParameter: 0.41, hubbardU: 3.5,
    mcMillanHopfieldEta: 0.6, miedemaPhiStar: 4.45, miedemaNws13: 1.61, miedemaV23: 3.8,
    sommerfeldGamma: 9.20, gruneisenParameter: 1.40,
    atomicRadius: 127, pettiforScale: 55, valenceElectrons: 7,
    paulingElectronegativity: 1.55, electronAffinity: 0.0, firstIonizationEnergy: 7.434,
    elementalTc: null
  },
  Fe: {
    symbol: "Fe", atomicNumber: 26, atomicMass: 55.845,
    debyeTemperature: 470, bulkModulus: 170.0, stonerParameter: 0.46, hubbardU: 4.0,
    mcMillanHopfieldEta: 1.3, miedemaPhiStar: 4.93, miedemaNws13: 1.77, miedemaV23: 3.7,
    sommerfeldGamma: 4.98, gruneisenParameter: 1.67,
    atomicRadius: 126, pettiforScale: 56, valenceElectrons: 8,
    paulingElectronegativity: 1.83, electronAffinity: 0.151, firstIonizationEnergy: 7.902,
    elementalTc: null
  },
  Co: {
    symbol: "Co", atomicNumber: 27, atomicMass: 58.933,
    debyeTemperature: 445, bulkModulus: 180.0, stonerParameter: 0.49, hubbardU: 4.0,
    mcMillanHopfieldEta: 1.6, miedemaPhiStar: 5.10, miedemaNws13: 1.75, miedemaV23: 3.5,
    sommerfeldGamma: 4.73, gruneisenParameter: 1.87,
    atomicRadius: 125, pettiforScale: 58, valenceElectrons: 9,
    paulingElectronegativity: 1.88, electronAffinity: 0.662, firstIonizationEnergy: 7.881,
    elementalTc: null
  },
  Ni: {
    symbol: "Ni", atomicNumber: 28, atomicMass: 58.693,
    debyeTemperature: 450, bulkModulus: 186.0, stonerParameter: 0.50, hubbardU: 4.0,
    mcMillanHopfieldEta: 2.7, miedemaPhiStar: 5.20, miedemaNws13: 1.75, miedemaV23: 3.5,
    sommerfeldGamma: 7.02, gruneisenParameter: 1.88,
    atomicRadius: 124, pettiforScale: 59, valenceElectrons: 10,
    paulingElectronegativity: 1.91, electronAffinity: 1.156, firstIonizationEnergy: 7.640,
    elementalTc: null
  },
  Cu: {
    symbol: "Cu", atomicNumber: 29, atomicMass: 63.546,
    debyeTemperature: 343, bulkModulus: 137.0, stonerParameter: 0.07, hubbardU: 4.5,
    mcMillanHopfieldEta: 0.4, miedemaPhiStar: 4.55, miedemaNws13: 1.47, miedemaV23: 3.7,
    sommerfeldGamma: 0.695, gruneisenParameter: 1.96,
    atomicRadius: 128, pettiforScale: 60, valenceElectrons: 11,
    paulingElectronegativity: 1.90, electronAffinity: 1.236, firstIonizationEnergy: 7.726,
    elementalTc: null
  },
  Zn: {
    symbol: "Zn", atomicNumber: 30, atomicMass: 65.38,
    debyeTemperature: 327, bulkModulus: 69.4, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.38, miedemaPhiStar: 4.10, miedemaNws13: 1.32, miedemaV23: 4.4,
    sommerfeldGamma: 0.64, gruneisenParameter: 2.01,
    atomicRadius: 134, pettiforScale: 75, valenceElectrons: 12,
    paulingElectronegativity: 1.65, electronAffinity: 0.0, firstIonizationEnergy: 9.394,
    elementalTc: 0.85
  },
  Ga: {
    symbol: "Ga", atomicNumber: 31, atomicMass: 69.723,
    debyeTemperature: 320, bulkModulus: 56.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.55, miedemaPhiStar: 4.10, miedemaNws13: 1.31, miedemaV23: 5.2,
    sommerfeldGamma: 0.596, gruneisenParameter: 2.01,
    atomicRadius: 135, pettiforScale: 81, valenceElectrons: 3,
    paulingElectronegativity: 1.81, electronAffinity: 0.43, firstIonizationEnergy: 5.999,
    elementalTc: 1.08
  },
  Ge: {
    symbol: "Ge", atomicNumber: 32, atomicMass: 72.630,
    debyeTemperature: 374, bulkModulus: 75.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.55, miedemaNws13: 1.37, miedemaV23: 4.6,
    sommerfeldGamma: null, gruneisenParameter: 1.1,
    atomicRadius: 122, pettiforScale: 88, valenceElectrons: 4,
    paulingElectronegativity: 2.01, electronAffinity: 1.233, firstIonizationEnergy: 7.900,
    elementalTc: null
  },
  As: {
    symbol: "As", atomicNumber: 33, atomicMass: 74.922,
    debyeTemperature: 282, bulkModulus: 22.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.80, miedemaNws13: 1.44, miedemaV23: 4.4,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 119, pettiforScale: 92, valenceElectrons: 5,
    paulingElectronegativity: 2.18, electronAffinity: 0.804, firstIonizationEnergy: 9.789,
    elementalTc: null
  },
  Se: {
    symbol: "Se", atomicNumber: 34, atomicMass: 78.960,
    debyeTemperature: 90, bulkModulus: 8.3, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 5.10, miedemaNws13: 1.40, miedemaV23: 4.3,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 120, pettiforScale: 97, valenceElectrons: 6,
    paulingElectronegativity: 2.55, electronAffinity: 2.021, firstIonizationEnergy: 9.752,
    elementalTc: null
  },
  Br: {
    symbol: "Br", atomicNumber: 35, atomicMass: 79.904,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 5.40, miedemaNws13: 1.50, miedemaV23: 4.1,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 120, pettiforScale: 98, valenceElectrons: 7,
    paulingElectronegativity: 2.96, electronAffinity: 3.364, firstIonizationEnergy: 11.814,
    elementalTc: null
  },
  Kr: {
    symbol: "Kr", atomicNumber: 36, atomicMass: 83.798,
    debyeTemperature: 72, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: null, miedemaNws13: null, miedemaV23: null,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 88, pettiforScale: 4, valenceElectrons: 0,
    paulingElectronegativity: 3.00, electronAffinity: 0.0, firstIonizationEnergy: 14.000,
    elementalTc: null
  },
  Rb: {
    symbol: "Rb", atomicNumber: 37, atomicMass: 85.468,
    debyeTemperature: 56, bulkModulus: 2.5, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.05, miedemaPhiStar: 2.10, miedemaNws13: 0.60, miedemaV23: 15.7,
    sommerfeldGamma: 2.41, gruneisenParameter: 1.48,
    atomicRadius: 248, pettiforScale: 9, valenceElectrons: 1,
    paulingElectronegativity: 0.82, electronAffinity: 0.486, firstIonizationEnergy: 4.177,
    elementalTc: null
  },
  Sr: {
    symbol: "Sr", atomicNumber: 38, atomicMass: 87.620,
    debyeTemperature: 147, bulkModulus: 11.6, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.15, miedemaPhiStar: 2.40, miedemaNws13: 0.84, miedemaV23: 11.3,
    sommerfeldGamma: 3.6, gruneisenParameter: 1.05,
    atomicRadius: 215, pettiforScale: 15, valenceElectrons: 2,
    paulingElectronegativity: 0.95, electronAffinity: 0.052, firstIonizationEnergy: 5.695,
    elementalTc: null
  },
  Y: {
    symbol: "Y", atomicNumber: 39, atomicMass: 88.906,
    debyeTemperature: 280, bulkModulus: 41.2, stonerParameter: 0.25, hubbardU: 2.5,
    mcMillanHopfieldEta: 1.3, miedemaPhiStar: 3.20, miedemaNws13: 1.21, miedemaV23: 7.3,
    sommerfeldGamma: 7.80, gruneisenParameter: 1.0,
    atomicRadius: 180, pettiforScale: 20, valenceElectrons: 3,
    paulingElectronegativity: 1.22, electronAffinity: 0.307, firstIonizationEnergy: 6.217,
    elementalTc: null
  },
  Zr: {
    symbol: "Zr", atomicNumber: 40, atomicMass: 91.224,
    debyeTemperature: 291, bulkModulus: 91.1, stonerParameter: 0.30, hubbardU: 2.8,
    mcMillanHopfieldEta: 2.2, miedemaPhiStar: 3.45, miedemaNws13: 1.39, miedemaV23: 5.8,
    sommerfeldGamma: 2.80, gruneisenParameter: 0.95,
    atomicRadius: 160, pettiforScale: 48, valenceElectrons: 4,
    paulingElectronegativity: 1.33, electronAffinity: 0.426, firstIonizationEnergy: 6.634,
    elementalTc: 0.61
  },
  Nb: {
    symbol: "Nb", atomicNumber: 41, atomicMass: 92.906,
    debyeTemperature: 275, bulkModulus: 170.0, stonerParameter: 0.35, hubbardU: 2.5,
    mcMillanHopfieldEta: 12.8, miedemaPhiStar: 4.05, miedemaNws13: 1.62, miedemaV23: 4.9,
    sommerfeldGamma: 7.79, gruneisenParameter: 1.63,
    atomicRadius: 146, pettiforScale: 49, valenceElectrons: 5,
    paulingElectronegativity: 1.60, electronAffinity: 0.893, firstIonizationEnergy: 6.759,
    elementalTc: 9.25
  },
  Mo: {
    symbol: "Mo", atomicNumber: 42, atomicMass: 95.950,
    debyeTemperature: 450, bulkModulus: 230.0, stonerParameter: 0.28, hubbardU: 2.5,
    mcMillanHopfieldEta: 3.0, miedemaPhiStar: 4.65, miedemaNws13: 1.77, miedemaV23: 4.4,
    sommerfeldGamma: 2.0, gruneisenParameter: 1.57,
    atomicRadius: 139, pettiforScale: 50, valenceElectrons: 6,
    paulingElectronegativity: 2.16, electronAffinity: 0.746, firstIonizationEnergy: 7.092,
    elementalTc: 0.915
  },
  Tc: {
    symbol: "Tc", atomicNumber: 43, atomicMass: 98.0,
    debyeTemperature: 453, bulkModulus: 297.0, stonerParameter: 0.32, hubbardU: 2.8,
    mcMillanHopfieldEta: 6.8, miedemaPhiStar: 5.30, miedemaNws13: 1.81, miedemaV23: 4.2,
    sommerfeldGamma: 6.28, gruneisenParameter: 1.5,
    atomicRadius: 136, pettiforScale: 53, valenceElectrons: 7,
    paulingElectronegativity: 1.90, electronAffinity: 0.55, firstIonizationEnergy: 7.28,
    elementalTc: 7.8
  },
  Ru: {
    symbol: "Ru", atomicNumber: 44, atomicMass: 101.07,
    debyeTemperature: 600, bulkModulus: 220.0, stonerParameter: 0.30, hubbardU: 3.0,
    mcMillanHopfieldEta: 2.1, miedemaPhiStar: 5.40, miedemaNws13: 1.83, miedemaV23: 4.1,
    sommerfeldGamma: 3.3, gruneisenParameter: 1.45,
    atomicRadius: 134, pettiforScale: 57, valenceElectrons: 8,
    paulingElectronegativity: 2.20, electronAffinity: 1.05, firstIonizationEnergy: 7.361,
    elementalTc: 0.49
  },
  Rh: {
    symbol: "Rh", atomicNumber: 45, atomicMass: 102.91,
    debyeTemperature: 480, bulkModulus: 275.0, stonerParameter: 0.35, hubbardU: 3.5,
    mcMillanHopfieldEta: 1.5, miedemaPhiStar: 5.10, miedemaNws13: 1.76, miedemaV23: 4.1,
    sommerfeldGamma: 4.9, gruneisenParameter: 2.2,
    atomicRadius: 134, pettiforScale: 61, valenceElectrons: 9,
    paulingElectronegativity: 2.28, electronAffinity: 1.137, firstIonizationEnergy: 7.459,
    elementalTc: null
  },
  Pd: {
    symbol: "Pd", atomicNumber: 46, atomicMass: 106.42,
    debyeTemperature: 274, bulkModulus: 180.0, stonerParameter: 0.40, hubbardU: 3.5,
    mcMillanHopfieldEta: 4.5, miedemaPhiStar: 5.45, miedemaNws13: 1.67, miedemaV23: 4.3,
    sommerfeldGamma: 9.42, gruneisenParameter: 2.23,
    atomicRadius: 137, pettiforScale: 62, valenceElectrons: 10,
    paulingElectronegativity: 2.20, electronAffinity: 0.562, firstIonizationEnergy: 8.337,
    elementalTc: null
  },
  Ag: {
    symbol: "Ag", atomicNumber: 47, atomicMass: 107.87,
    debyeTemperature: 225, bulkModulus: 100.0, stonerParameter: 0.05, hubbardU: null,
    mcMillanHopfieldEta: 0.25, miedemaPhiStar: 4.45, miedemaNws13: 1.36, miedemaV23: 4.7,
    sommerfeldGamma: 0.646, gruneisenParameter: 2.40,
    atomicRadius: 144, pettiforScale: 63, valenceElectrons: 11,
    paulingElectronegativity: 1.93, electronAffinity: 1.302, firstIonizationEnergy: 7.576,
    elementalTc: null
  },
  Cd: {
    symbol: "Cd", atomicNumber: 48, atomicMass: 112.41,
    debyeTemperature: 209, bulkModulus: 42.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.30, miedemaPhiStar: 4.05, miedemaNws13: 1.24, miedemaV23: 5.5,
    sommerfeldGamma: 0.688, gruneisenParameter: 2.14,
    atomicRadius: 151, pettiforScale: 76, valenceElectrons: 12,
    paulingElectronegativity: 1.69, electronAffinity: 0.0, firstIonizationEnergy: 8.994,
    elementalTc: 0.517
  },
  In: {
    symbol: "In", atomicNumber: 49, atomicMass: 114.82,
    debyeTemperature: 108, bulkModulus: 41.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.70, miedemaPhiStar: 3.90, miedemaNws13: 1.17, miedemaV23: 6.3,
    sommerfeldGamma: 1.69, gruneisenParameter: 2.56,
    atomicRadius: 167, pettiforScale: 82, valenceElectrons: 3,
    paulingElectronegativity: 1.78, electronAffinity: 0.404, firstIonizationEnergy: 5.786,
    elementalTc: 3.41
  },
  Sn: {
    symbol: "Sn", atomicNumber: 50, atomicMass: 118.71,
    debyeTemperature: 200, bulkModulus: 58.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.95, miedemaPhiStar: 4.15, miedemaNws13: 1.24, miedemaV23: 6.4,
    sommerfeldGamma: 1.78, gruneisenParameter: 2.14,
    atomicRadius: 140, pettiforScale: 89, valenceElectrons: 4,
    paulingElectronegativity: 1.96, electronAffinity: 1.112, firstIonizationEnergy: 7.344,
    elementalTc: 3.72
  },
  Sb: {
    symbol: "Sb", atomicNumber: 51, atomicMass: 121.76,
    debyeTemperature: 211, bulkModulus: 42.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.40, miedemaNws13: 1.26, miedemaV23: 6.6,
    sommerfeldGamma: null, gruneisenParameter: 1.09,
    atomicRadius: 141, pettiforScale: 93, valenceElectrons: 5,
    paulingElectronegativity: 2.05, electronAffinity: 1.047, firstIonizationEnergy: 8.608,
    elementalTc: null
  },
  Te: {
    symbol: "Te", atomicNumber: 52, atomicMass: 127.60,
    debyeTemperature: 153, bulkModulus: 65.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.72, miedemaNws13: 1.31, miedemaV23: 6.6,
    sommerfeldGamma: null, gruneisenParameter: 1.68,
    atomicRadius: 143, pettiforScale: 94, valenceElectrons: 6,
    paulingElectronegativity: 2.10, electronAffinity: 1.971, firstIonizationEnergy: 9.010,
    elementalTc: null
  },
  I: {
    symbol: "I", atomicNumber: 53, atomicMass: 126.90,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 5.00, miedemaNws13: 1.38, miedemaV23: 6.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 140, pettiforScale: 98.5, valenceElectrons: 7,
    paulingElectronegativity: 2.66, electronAffinity: 3.059, firstIonizationEnergy: 10.451,
    elementalTc: null
  },
  Xe: {
    symbol: "Xe", atomicNumber: 54, atomicMass: 131.29,
    debyeTemperature: 64, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: null, miedemaNws13: null, miedemaV23: null,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 108, pettiforScale: 5, valenceElectrons: 0,
    paulingElectronegativity: 2.60, electronAffinity: 0.0, firstIonizationEnergy: 12.130,
    elementalTc: null
  },
  Cs: {
    symbol: "Cs", atomicNumber: 55, atomicMass: 132.91,
    debyeTemperature: 38, bulkModulus: 1.6, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.04, miedemaPhiStar: 1.95, miedemaNws13: 0.55, miedemaV23: 19.1,
    sommerfeldGamma: 3.20, gruneisenParameter: 1.56,
    atomicRadius: 265, pettiforScale: 8, valenceElectrons: 1,
    paulingElectronegativity: 0.79, electronAffinity: 0.472, firstIonizationEnergy: 3.894,
    elementalTc: null
  },
  Ba: {
    symbol: "Ba", atomicNumber: 56, atomicMass: 137.33,
    debyeTemperature: 110, bulkModulus: 9.6, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.12, miedemaPhiStar: 2.32, miedemaNws13: 0.81, miedemaV23: 12.8,
    sommerfeldGamma: 2.7, gruneisenParameter: 0.73,
    atomicRadius: 222, pettiforScale: 14, valenceElectrons: 2,
    paulingElectronegativity: 0.89, electronAffinity: 0.145, firstIonizationEnergy: 5.212,
    elementalTc: null
  },
  La: {
    symbol: "La", atomicNumber: 57, atomicMass: 138.91,
    debyeTemperature: 142, bulkModulus: 27.9, stonerParameter: 0.22, hubbardU: 5.6,
    mcMillanHopfieldEta: 2.0, miedemaPhiStar: 3.17, miedemaNws13: 1.18, miedemaV23: 8.0,
    sommerfeldGamma: 10.0, gruneisenParameter: 0.80,
    atomicRadius: 187, pettiforScale: 33, valenceElectrons: 3,
    paulingElectronegativity: 1.10, electronAffinity: 0.47, firstIonizationEnergy: 5.577,
    elementalTc: 6.00
  },
  Ce: {
    symbol: "Ce", atomicNumber: 58, atomicMass: 140.12,
    debyeTemperature: 179, bulkModulus: 21.5, stonerParameter: 0.20, hubbardU: 6.0,
    mcMillanHopfieldEta: 1.5, miedemaPhiStar: 3.18, miedemaNws13: 1.19, miedemaV23: 7.9,
    sommerfeldGamma: 12.8, gruneisenParameter: 1.05,
    atomicRadius: 182, pettiforScale: 34, valenceElectrons: 4,
    paulingElectronegativity: 1.12, electronAffinity: 0.955, firstIonizationEnergy: 5.539,
    elementalTc: null
  },
  Pr: {
    symbol: "Pr", atomicNumber: 59, atomicMass: 140.91,
    debyeTemperature: 152, bulkModulus: 28.8, stonerParameter: 0.18, hubbardU: 6.0,
    mcMillanHopfieldEta: 1.2, miedemaPhiStar: 3.19, miedemaNws13: 1.20, miedemaV23: 7.8,
    sommerfeldGamma: 11.0, gruneisenParameter: 0.82,
    atomicRadius: 182, pettiforScale: 35, valenceElectrons: 5,
    paulingElectronegativity: 1.13, electronAffinity: 0.962, firstIonizationEnergy: 5.464,
    elementalTc: null
  },
  Nd: {
    symbol: "Nd", atomicNumber: 60, atomicMass: 144.24,
    debyeTemperature: 163, bulkModulus: 31.8, stonerParameter: 0.18, hubbardU: 6.2,
    mcMillanHopfieldEta: 1.0, miedemaPhiStar: 3.19, miedemaNws13: 1.20, miedemaV23: 7.7,
    sommerfeldGamma: 10.5, gruneisenParameter: 0.85,
    atomicRadius: 181, pettiforScale: 36, valenceElectrons: 6,
    paulingElectronegativity: 1.14, electronAffinity: 1.916, firstIonizationEnergy: 5.525,
    elementalTc: null
  },
  Pm: {
    symbol: "Pm", atomicNumber: 61, atomicMass: 145.0,
    debyeTemperature: 158, bulkModulus: 33.0, stonerParameter: 0.17, hubbardU: 6.3,
    mcMillanHopfieldEta: 0.9, miedemaPhiStar: 3.19, miedemaNws13: 1.20, miedemaV23: 7.6,
    sommerfeldGamma: 10.0, gruneisenParameter: 0.85,
    atomicRadius: 183, pettiforScale: 37, valenceElectrons: 7,
    paulingElectronegativity: 1.13, electronAffinity: 0.129, firstIonizationEnergy: 5.582,
    elementalTc: null
  },
  Sm: {
    symbol: "Sm", atomicNumber: 62, atomicMass: 150.36,
    debyeTemperature: 169, bulkModulus: 37.8, stonerParameter: 0.17, hubbardU: 6.5,
    mcMillanHopfieldEta: 0.8, miedemaPhiStar: 3.20, miedemaNws13: 1.21, miedemaV23: 7.5,
    sommerfeldGamma: 9.0, gruneisenParameter: 0.89,
    atomicRadius: 180, pettiforScale: 38, valenceElectrons: 8,
    paulingElectronegativity: 1.17, electronAffinity: 0.162, firstIonizationEnergy: 5.644,
    elementalTc: null
  },
  Eu: {
    symbol: "Eu", atomicNumber: 63, atomicMass: 151.96,
    debyeTemperature: 127, bulkModulus: 8.3, stonerParameter: 0.16, hubbardU: 7.0,
    mcMillanHopfieldEta: 0.5, miedemaPhiStar: 2.50, miedemaNws13: 0.87, miedemaV23: 10.3,
    sommerfeldGamma: 6.5, gruneisenParameter: 1.15,
    atomicRadius: 180, pettiforScale: 17, valenceElectrons: 9,
    paulingElectronegativity: 1.20, electronAffinity: 0.116, firstIonizationEnergy: 5.670,
    elementalTc: null
  },
  Gd: {
    symbol: "Gd", atomicNumber: 64, atomicMass: 157.25,
    debyeTemperature: 200, bulkModulus: 37.9, stonerParameter: 0.20, hubbardU: 6.7,
    mcMillanHopfieldEta: 1.0, miedemaPhiStar: 3.20, miedemaNws13: 1.21, miedemaV23: 7.6,
    sommerfeldGamma: 6.0, gruneisenParameter: 0.80,
    atomicRadius: 180, pettiforScale: 27, valenceElectrons: 10,
    paulingElectronegativity: 1.20, electronAffinity: 0.212, firstIonizationEnergy: 6.150,
    elementalTc: null
  },
  Tb: {
    symbol: "Tb", atomicNumber: 65, atomicMass: 158.93,
    debyeTemperature: 177, bulkModulus: 38.7, stonerParameter: 0.18, hubbardU: 6.5,
    mcMillanHopfieldEta: 0.9, miedemaPhiStar: 3.21, miedemaNws13: 1.22, miedemaV23: 7.5,
    sommerfeldGamma: 4.5, gruneisenParameter: 0.79,
    atomicRadius: 177, pettiforScale: 28, valenceElectrons: 11,
    paulingElectronegativity: 1.10, electronAffinity: 1.165, firstIonizationEnergy: 5.864,
    elementalTc: null
  },
  Dy: {
    symbol: "Dy", atomicNumber: 66, atomicMass: 162.50,
    debyeTemperature: 210, bulkModulus: 40.5, stonerParameter: 0.17, hubbardU: 6.5,
    mcMillanHopfieldEta: 0.8, miedemaPhiStar: 3.21, miedemaNws13: 1.22, miedemaV23: 7.4,
    sommerfeldGamma: 4.9, gruneisenParameter: 0.78,
    atomicRadius: 178, pettiforScale: 29, valenceElectrons: 12,
    paulingElectronegativity: 1.22, electronAffinity: 0.352, firstIonizationEnergy: 5.939,
    elementalTc: null
  },
  Ho: {
    symbol: "Ho", atomicNumber: 67, atomicMass: 164.93,
    debyeTemperature: 190, bulkModulus: 40.2, stonerParameter: 0.16, hubbardU: 6.5,
    mcMillanHopfieldEta: 0.8, miedemaPhiStar: 3.22, miedemaNws13: 1.22, miedemaV23: 7.3,
    sommerfeldGamma: 4.7, gruneisenParameter: 0.78,
    atomicRadius: 176, pettiforScale: 30, valenceElectrons: 13,
    paulingElectronegativity: 1.23, electronAffinity: 0.338, firstIonizationEnergy: 6.022,
    elementalTc: null
  },
  Er: {
    symbol: "Er", atomicNumber: 68, atomicMass: 167.26,
    debyeTemperature: 200, bulkModulus: 44.4, stonerParameter: 0.15, hubbardU: 6.5,
    mcMillanHopfieldEta: 0.7, miedemaPhiStar: 3.22, miedemaNws13: 1.23, miedemaV23: 7.2,
    sommerfeldGamma: 4.5, gruneisenParameter: 0.82,
    atomicRadius: 176, pettiforScale: 31, valenceElectrons: 14,
    paulingElectronegativity: 1.24, electronAffinity: 0.312, firstIonizationEnergy: 6.108,
    elementalTc: null
  },
  Tm: {
    symbol: "Tm", atomicNumber: 69, atomicMass: 168.93,
    debyeTemperature: 200, bulkModulus: 44.5, stonerParameter: 0.15, hubbardU: 6.5,
    mcMillanHopfieldEta: 0.6, miedemaPhiStar: 3.22, miedemaNws13: 1.23, miedemaV23: 7.1,
    sommerfeldGamma: 4.3, gruneisenParameter: 0.84,
    atomicRadius: 176, pettiforScale: 32, valenceElectrons: 15,
    paulingElectronegativity: 1.25, electronAffinity: 1.029, firstIonizationEnergy: 6.184,
    elementalTc: null
  },
  Yb: {
    symbol: "Yb", atomicNumber: 70, atomicMass: 173.04,
    debyeTemperature: 120, bulkModulus: 13.0, stonerParameter: 0.14, hubbardU: 7.0,
    mcMillanHopfieldEta: 0.4, miedemaPhiStar: 2.55, miedemaNws13: 0.88, miedemaV23: 10.0,
    sommerfeldGamma: 3.0, gruneisenParameter: 1.25,
    atomicRadius: 176, pettiforScale: 18, valenceElectrons: 16,
    paulingElectronegativity: 1.10, electronAffinity: -0.020, firstIonizationEnergy: 6.254,
    elementalTc: null
  },
  Lu: {
    symbol: "Lu", atomicNumber: 71, atomicMass: 174.97,
    debyeTemperature: 210, bulkModulus: 47.6, stonerParameter: 0.20, hubbardU: 5.5,
    mcMillanHopfieldEta: 1.5, miedemaPhiStar: 3.22, miedemaNws13: 1.23, miedemaV23: 7.0,
    sommerfeldGamma: 8.19, gruneisenParameter: 0.75,
    atomicRadius: 174, pettiforScale: 21, valenceElectrons: 3,
    paulingElectronegativity: 1.27, electronAffinity: 0.34, firstIonizationEnergy: 5.426,
    elementalTc: 0.1
  },
  Hf: {
    symbol: "Hf", atomicNumber: 72, atomicMass: 178.49,
    debyeTemperature: 252, bulkModulus: 110.0, stonerParameter: 0.28, hubbardU: 2.8,
    mcMillanHopfieldEta: 2.0, miedemaPhiStar: 3.55, miedemaNws13: 1.43, miedemaV23: 5.6,
    sommerfeldGamma: 2.16, gruneisenParameter: 1.06,
    atomicRadius: 159, pettiforScale: 47, valenceElectrons: 4,
    paulingElectronegativity: 1.30, electronAffinity: 0.178, firstIonizationEnergy: 6.825,
    elementalTc: 0.128
  },
  Ta: {
    symbol: "Ta", atomicNumber: 73, atomicMass: 180.95,
    debyeTemperature: 240, bulkModulus: 200.0, stonerParameter: 0.30, hubbardU: 2.5,
    mcMillanHopfieldEta: 9.2, miedemaPhiStar: 4.05, miedemaNws13: 1.63, miedemaV23: 4.9,
    sommerfeldGamma: 5.9, gruneisenParameter: 1.65,
    atomicRadius: 146, pettiforScale: 46, valenceElectrons: 5,
    paulingElectronegativity: 1.50, electronAffinity: 0.322, firstIonizationEnergy: 7.550,
    elementalTc: 4.47
  },
  W: {
    symbol: "W", atomicNumber: 74, atomicMass: 183.84,
    debyeTemperature: 400, bulkModulus: 310.0, stonerParameter: 0.25, hubbardU: 2.3,
    mcMillanHopfieldEta: 2.0, miedemaPhiStar: 4.80, miedemaNws13: 1.81, miedemaV23: 4.5,
    sommerfeldGamma: 1.01, gruneisenParameter: 1.62,
    atomicRadius: 139, pettiforScale: 45, valenceElectrons: 6,
    paulingElectronegativity: 2.36, electronAffinity: 0.816, firstIonizationEnergy: 7.864,
    elementalTc: 0.015
  },
  Re: {
    symbol: "Re", atomicNumber: 75, atomicMass: 186.21,
    debyeTemperature: 430, bulkModulus: 370.0, stonerParameter: 0.28, hubbardU: 2.5,
    mcMillanHopfieldEta: 2.4, miedemaPhiStar: 5.40, miedemaNws13: 1.86, miedemaV23: 4.3,
    sommerfeldGamma: 2.35, gruneisenParameter: 1.66,
    atomicRadius: 137, pettiforScale: 44, valenceElectrons: 7,
    paulingElectronegativity: 1.90, electronAffinity: 0.15, firstIonizationEnergy: 7.834,
    elementalTc: 1.70
  },
  Os: {
    symbol: "Os", atomicNumber: 76, atomicMass: 190.23,
    debyeTemperature: 500, bulkModulus: 462.0, stonerParameter: 0.26, hubbardU: 2.8,
    mcMillanHopfieldEta: 1.5, miedemaPhiStar: 5.40, miedemaNws13: 1.85, miedemaV23: 4.3,
    sommerfeldGamma: 2.35, gruneisenParameter: 1.60,
    atomicRadius: 135, pettiforScale: 43, valenceElectrons: 8,
    paulingElectronegativity: 2.20, electronAffinity: 1.10, firstIonizationEnergy: 8.438,
    elementalTc: 0.66
  },
  Ir: {
    symbol: "Ir", atomicNumber: 77, atomicMass: 192.22,
    debyeTemperature: 420, bulkModulus: 320.0, stonerParameter: 0.28, hubbardU: 3.0,
    mcMillanHopfieldEta: 1.8, miedemaPhiStar: 5.55, miedemaNws13: 1.83, miedemaV23: 4.2,
    sommerfeldGamma: 3.1, gruneisenParameter: 1.74,
    atomicRadius: 136, pettiforScale: 42, valenceElectrons: 9,
    paulingElectronegativity: 2.20, electronAffinity: 1.565, firstIonizationEnergy: 8.967,
    elementalTc: 0.11
  },
  Pt: {
    symbol: "Pt", atomicNumber: 78, atomicMass: 195.08,
    debyeTemperature: 240, bulkModulus: 230.0, stonerParameter: 0.32, hubbardU: 3.2,
    mcMillanHopfieldEta: 2.5, miedemaPhiStar: 5.65, miedemaNws13: 1.78, miedemaV23: 4.4,
    sommerfeldGamma: 6.8, gruneisenParameter: 2.54,
    atomicRadius: 139, pettiforScale: 64, valenceElectrons: 10,
    paulingElectronegativity: 2.28, electronAffinity: 2.128, firstIonizationEnergy: 8.959,
    elementalTc: null
  },
  Au: {
    symbol: "Au", atomicNumber: 79, atomicMass: 196.97,
    debyeTemperature: 165, bulkModulus: 220.0, stonerParameter: 0.04, hubbardU: null,
    mcMillanHopfieldEta: 0.1, miedemaPhiStar: 5.15, miedemaNws13: 1.57, miedemaV23: 4.7,
    sommerfeldGamma: 0.729, gruneisenParameter: 3.03,
    atomicRadius: 144, pettiforScale: 65, valenceElectrons: 11,
    paulingElectronegativity: 2.54, electronAffinity: 2.309, firstIonizationEnergy: 9.226,
    elementalTc: null
  },
  Hg: {
    symbol: "Hg", atomicNumber: 80, atomicMass: 200.59,
    debyeTemperature: 72, bulkModulus: 25.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.40, miedemaPhiStar: 4.20, miedemaNws13: 1.24, miedemaV23: 5.8,
    sommerfeldGamma: 1.79, gruneisenParameter: 2.43,
    atomicRadius: 151, pettiforScale: 74, valenceElectrons: 12,
    paulingElectronegativity: 2.00, electronAffinity: 0.0, firstIonizationEnergy: 10.438,
    elementalTc: 4.15
  },
  Tl: {
    symbol: "Tl", atomicNumber: 81, atomicMass: 204.38,
    debyeTemperature: 78, bulkModulus: 43.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 0.65, miedemaPhiStar: 3.90, miedemaNws13: 1.12, miedemaV23: 6.6,
    sommerfeldGamma: 1.47, gruneisenParameter: 2.84,
    atomicRadius: 170, pettiforScale: 83, valenceElectrons: 3,
    paulingElectronegativity: 1.62, electronAffinity: 0.377, firstIonizationEnergy: 6.108,
    elementalTc: 2.38
  },
  Pb: {
    symbol: "Pb", atomicNumber: 82, atomicMass: 207.20,
    debyeTemperature: 105, bulkModulus: 46.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: 1.20, miedemaPhiStar: 4.10, miedemaNws13: 1.15, miedemaV23: 6.9,
    sommerfeldGamma: 2.98, gruneisenParameter: 2.65,
    atomicRadius: 175, pettiforScale: 90, valenceElectrons: 4,
    paulingElectronegativity: 2.33, electronAffinity: 0.364, firstIonizationEnergy: 7.417,
    elementalTc: 7.20
  },
  Bi: {
    symbol: "Bi", atomicNumber: 83, atomicMass: 208.98,
    debyeTemperature: 119, bulkModulus: 31.0, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.15, miedemaNws13: 1.16, miedemaV23: 7.2,
    sommerfeldGamma: 0.008, gruneisenParameter: 1.20,
    atomicRadius: 156, pettiforScale: 85, valenceElectrons: 5,
    paulingElectronegativity: 2.02, electronAffinity: 0.942, firstIonizationEnergy: 7.286,
    elementalTc: null
  },
  Po: {
    symbol: "Po", atomicNumber: 84, atomicMass: 209.0,
    debyeTemperature: 100, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.40, miedemaNws13: 1.20, miedemaV23: 7.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 167, pettiforScale: 95, valenceElectrons: 6,
    paulingElectronegativity: 2.00, electronAffinity: 1.90, firstIonizationEnergy: 8.417,
    elementalTc: null
  },
  At: {
    symbol: "At", atomicNumber: 85, atomicMass: 210.0,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 4.80, miedemaNws13: 1.35, miedemaV23: 6.5,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 140, pettiforScale: 98.7, valenceElectrons: 7,
    paulingElectronegativity: 2.20, electronAffinity: 2.80, firstIonizationEnergy: 9.30,
    elementalTc: null
  },
  Rn: {
    symbol: "Rn", atomicNumber: 86, atomicMass: 222.0,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: null, miedemaNws13: null, miedemaV23: null,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 120, pettiforScale: 6, valenceElectrons: 0,
    paulingElectronegativity: null, electronAffinity: 0.0, firstIonizationEnergy: 10.749,
    elementalTc: null
  },
  Fr: {
    symbol: "Fr", atomicNumber: 87, atomicMass: 223.0,
    debyeTemperature: null, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 1.80, miedemaNws13: 0.50, miedemaV23: 22.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 270, pettiforScale: 7, valenceElectrons: 1,
    paulingElectronegativity: 0.70, electronAffinity: 0.486, firstIonizationEnergy: 4.073,
    elementalTc: null
  },
  Ra: {
    symbol: "Ra", atomicNumber: 88, atomicMass: 226.0,
    debyeTemperature: 89, bulkModulus: null, stonerParameter: null, hubbardU: null,
    mcMillanHopfieldEta: null, miedemaPhiStar: 2.25, miedemaNws13: 0.75, miedemaV23: 14.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 215, pettiforScale: 13, valenceElectrons: 2,
    paulingElectronegativity: 0.90, electronAffinity: 0.10, firstIonizationEnergy: 5.279,
    elementalTc: null
  },
  Ac: {
    symbol: "Ac", atomicNumber: 89, atomicMass: 227.0,
    debyeTemperature: 124, bulkModulus: null, stonerParameter: 0.20, hubbardU: 4.0,
    mcMillanHopfieldEta: null, miedemaPhiStar: 3.00, miedemaNws13: 1.10, miedemaV23: 9.0,
    sommerfeldGamma: null, gruneisenParameter: null,
    atomicRadius: 195, pettiforScale: 22, valenceElectrons: 3,
    paulingElectronegativity: 1.10, electronAffinity: 0.35, firstIonizationEnergy: 5.17,
    elementalTc: null
  },
  Th: {
    symbol: "Th", atomicNumber: 90, atomicMass: 232.04,
    debyeTemperature: 163, bulkModulus: 54.0, stonerParameter: 0.22, hubbardU: 4.5,
    mcMillanHopfieldEta: 2.0, miedemaPhiStar: 3.30, miedemaNws13: 1.28, miedemaV23: 7.3,
    sommerfeldGamma: 4.3, gruneisenParameter: 1.28,
    atomicRadius: 180, pettiforScale: 39, valenceElectrons: 4,
    paulingElectronegativity: 1.30, electronAffinity: 0.0, firstIonizationEnergy: 6.307,
    elementalTc: 1.38
  },
  Pa: {
    symbol: "Pa", atomicNumber: 91, atomicMass: 231.04,
    debyeTemperature: 185, bulkModulus: 118.0, stonerParameter: 0.24, hubbardU: 4.0,
    mcMillanHopfieldEta: 2.5, miedemaPhiStar: 3.40, miedemaNws13: 1.35, miedemaV23: 6.8,
    sommerfeldGamma: 5.0, gruneisenParameter: 1.2,
    atomicRadius: 163, pettiforScale: 40, valenceElectrons: 5,
    paulingElectronegativity: 1.50, electronAffinity: 0.0, firstIonizationEnergy: 5.89,
    elementalTc: 1.40
  },
  U: {
    symbol: "U", atomicNumber: 92, atomicMass: 238.03,
    debyeTemperature: 207, bulkModulus: 100.0, stonerParameter: 0.26, hubbardU: 4.0,
    mcMillanHopfieldEta: 2.8, miedemaPhiStar: 3.40, miedemaNws13: 1.40, miedemaV23: 6.3,
    sommerfeldGamma: 9.14, gruneisenParameter: 2.09,
    atomicRadius: 156, pettiforScale: 41, valenceElectrons: 6,
    paulingElectronegativity: 1.38, electronAffinity: 0.0, firstIonizationEnergy: 6.194,
    elementalTc: null
  },
  Np: {
    symbol: "Np", atomicNumber: 93, atomicMass: 237.0,
    debyeTemperature: 175, bulkModulus: 118.0, stonerParameter: 0.24, hubbardU: 4.0,
    mcMillanHopfieldEta: 2.2, miedemaPhiStar: 3.45, miedemaNws13: 1.42, miedemaV23: 6.1,
    sommerfeldGamma: 13.9, gruneisenParameter: 1.8,
    atomicRadius: 155, pettiforScale: 41.5, valenceElectrons: 7,
    paulingElectronegativity: 1.36, electronAffinity: 0.0, firstIonizationEnergy: 6.266,
    elementalTc: null
  },
  Pu: {
    symbol: "Pu", atomicNumber: 94, atomicMass: 244.0,
    debyeTemperature: 171, bulkModulus: 54.0, stonerParameter: 0.22, hubbardU: 4.5,
    mcMillanHopfieldEta: 2.0, miedemaPhiStar: 3.45, miedemaNws13: 1.42, miedemaV23: 6.2,
    sommerfeldGamma: 46.0, gruneisenParameter: 1.6,
    atomicRadius: 159, pettiforScale: 42, valenceElectrons: 8,
    paulingElectronegativity: 1.28, electronAffinity: 0.0, firstIonizationEnergy: 6.026,
    elementalTc: null
  },
  Am: {
    symbol: "Am", atomicNumber: 95, atomicMass: 243.0,
    debyeTemperature: 120, bulkModulus: 30.0, stonerParameter: 0.20, hubbardU: 5.0,
    mcMillanHopfieldEta: 1.0, miedemaPhiStar: 2.80, miedemaNws13: 1.05, miedemaV23: 8.5,
    sommerfeldGamma: 2.0, gruneisenParameter: 1.0,
    atomicRadius: 173, pettiforScale: 23, valenceElectrons: 9,
    paulingElectronegativity: 1.30, electronAffinity: 0.0, firstIonizationEnergy: 5.974,
    elementalTc: 0.6
  },
  Cm: {
    symbol: "Cm", atomicNumber: 96, atomicMass: 247.0,
    debyeTemperature: 128, bulkModulus: 36.0, stonerParameter: 0.20, hubbardU: 5.0,
    mcMillanHopfieldEta: 0.8, miedemaPhiStar: 2.90, miedemaNws13: 1.10, miedemaV23: 8.0,
    sommerfeldGamma: 7.0, gruneisenParameter: 1.0,
    atomicRadius: 174, pettiforScale: 24, valenceElectrons: 10,
    paulingElectronegativity: 1.30, electronAffinity: 0.0, firstIonizationEnergy: 5.991,
    elementalTc: null
  },
};

export function getElementData(symbol: string): ElementalProperties | null {
  return ELEMENTAL_DATA[symbol] ?? null;
}

export function getCompositionWeightedProperty(
  elementCounts: Record<string, number>,
  property: keyof ElementalProperties
): number | null {
  let totalWeight = 0;
  let weightedSum = 0;
  for (const [el, count] of Object.entries(elementCounts)) {
    const data = ELEMENTAL_DATA[el];
    if (!data) continue;
    const val = data[property];
    if (typeof val === "number" && val !== null) {
      weightedSum += val * count;
      totalWeight += count;
    }
  }
  return totalWeight > 0 ? weightedSum / totalWeight : null;
}

export function getAverageMass(elementCounts: Record<string, number>): number {
  let totalMass = 0;
  let totalAtoms = 0;
  for (const [el, count] of Object.entries(elementCounts)) {
    const data = ELEMENTAL_DATA[el];
    if (data) {
      totalMass += data.atomicMass * count;
      totalAtoms += count;
    }
  }
  return totalAtoms > 0 ? totalMass / totalAtoms : 50;
}

export function getLightestMass(elements: string[]): number {
  let lightest = Infinity;
  for (const el of elements) {
    const data = ELEMENTAL_DATA[el];
    if (data && data.atomicMass < lightest) {
      lightest = data.atomicMass;
    }
  }
  return lightest === Infinity ? 50 : lightest;
}

export function getMaxValenceElectrons(elements: string[]): number {
  let maxV = 0;
  for (const el of elements) {
    const data = ELEMENTAL_DATA[el];
    if (data && data.valenceElectrons > maxV) {
      maxV = data.valenceElectrons;
    }
  }
  return maxV;
}

export function isTransitionMetal(symbol: string): boolean {
  const data = ELEMENTAL_DATA[symbol];
  if (!data) return false;
  const z = data.atomicNumber;
  return (z >= 21 && z <= 30) || (z >= 39 && z <= 48) || (z >= 72 && z <= 80);
}

export function isRareEarth(symbol: string): boolean {
  const data = ELEMENTAL_DATA[symbol];
  if (!data) return false;
  const z = data.atomicNumber;
  return z >= 57 && z <= 71;
}

export function isActinide(symbol: string): boolean {
  const data = ELEMENTAL_DATA[symbol];
  if (!data) return false;
  const z = data.atomicNumber;
  return z >= 89 && z <= 96;
}

export function hasDOrFElectrons(symbol: string): boolean {
  return isTransitionMetal(symbol) || isRareEarth(symbol) || isActinide(symbol);
}

export function getHubbardU(symbol: string): number | null {
  const data = ELEMENTAL_DATA[symbol];
  return data?.hubbardU ?? null;
}

export function getStonerParameter(symbol: string): number | null {
  const data = ELEMENTAL_DATA[symbol];
  return data?.stonerParameter ?? null;
}

export function getMcMillanHopfieldEta(symbol: string): number | null {
  const data = ELEMENTAL_DATA[symbol];
  return data?.mcMillanHopfieldEta ?? null;
}

export function getDebyeTemperature(symbol: string): number | null {
  const data = ELEMENTAL_DATA[symbol];
  return data?.debyeTemperature ?? null;
}

export function getElementalTc(symbol: string): number | null {
  const data = ELEMENTAL_DATA[symbol];
  return data?.elementalTc ?? null;
}

export const KNOWN_ELEMENTAL_SUPERCONDUCTORS: Record<string, number> = {};
for (const [symbol, data] of Object.entries(ELEMENTAL_DATA)) {
  if (data.elementalTc !== null) {
    KNOWN_ELEMENTAL_SUPERCONDUCTORS[symbol] = data.elementalTc;
  }
}
