import { db } from "./db";
import { elements, materials, learningPhases, novelPredictions, researchLogs } from "@shared/schema";
import { count } from "drizzle-orm";

const ELEMENTS_DATA = [
  { id: 1, symbol: "H", name: "Hydrogen", atomicMass: 1.008, period: 1, group: 1, category: "nonmetal", electronegativity: 2.2, electronConfig: "1s¹", meltingPoint: 14.01, boilingPoint: 20.28, density: 0.0000899, discoveredYear: 1766, description: "The lightest and most abundant element in the universe. Forms water and most organic compounds." },
  { id: 2, symbol: "He", name: "Helium", atomicMass: 4.003, period: 1, group: 18, category: "noble gas", electronegativity: null, electronConfig: "1s²", meltingPoint: null, boilingPoint: 4.22, density: 0.0001786, discoveredYear: 1868, description: "Second most abundant element in the universe. Used in superconducting magnets and MRI machines." },
  { id: 3, symbol: "Li", name: "Lithium", atomicMass: 6.941, period: 2, group: 1, category: "alkali metal", electronegativity: 0.98, electronConfig: "[He] 2s¹", meltingPoint: 453.65, boilingPoint: 1615, density: 0.534, discoveredYear: 1817, description: "Lightest metal. Critical in rechargeable batteries and mood-stabilizing medications." },
  { id: 4, symbol: "Be", name: "Beryllium", atomicMass: 9.012, period: 2, group: 2, category: "alkaline earth metal", electronegativity: 1.57, electronConfig: "[He] 2s²", meltingPoint: 1560, boilingPoint: 2742, density: 1.85, discoveredYear: 1798, description: "Very light, strong metal used in aerospace and nuclear applications." },
  { id: 5, symbol: "B", name: "Boron", atomicMass: 10.811, period: 2, group: 13, category: "metalloid", electronegativity: 2.04, electronConfig: "[He] 2s² 2p¹", meltingPoint: 2349, boilingPoint: 4200, density: 2.34, discoveredYear: 1808, description: "Semiconductor used in glass, ceramics, and as a neutron absorber in nuclear reactors." },
  { id: 6, symbol: "C", name: "Carbon", atomicMass: 12.011, period: 2, group: 14, category: "nonmetal", electronegativity: 2.55, electronConfig: "[He] 2s² 2p²", meltingPoint: 3800, boilingPoint: 4300, density: 2.267, discoveredYear: null, description: "Basis of all organic life. Forms graphite, diamond, graphene, and carbon nanotubes." },
  { id: 7, symbol: "N", name: "Nitrogen", atomicMass: 14.007, period: 2, group: 15, category: "nonmetal", electronegativity: 3.04, electronConfig: "[He] 2s² 2p³", meltingPoint: 63.15, boilingPoint: 77.36, density: 0.001251, discoveredYear: 1772, description: "Makes up 78% of Earth's atmosphere. Essential for proteins and DNA." },
  { id: 8, symbol: "O", name: "Oxygen", atomicMass: 15.999, period: 2, group: 16, category: "nonmetal", electronegativity: 3.44, electronConfig: "[He] 2s² 2p⁴", meltingPoint: 54.36, boilingPoint: 90.19, density: 0.001429, discoveredYear: 1774, description: "Third most abundant element in the universe. Essential for combustion and aerobic life." },
  { id: 9, symbol: "F", name: "Fluorine", atomicMass: 18.998, period: 2, group: 17, category: "halogen", electronegativity: 3.98, electronConfig: "[He] 2s² 2p⁵", meltingPoint: 53.53, boilingPoint: 85.03, density: 0.001696, discoveredYear: 1886, description: "Most electronegative and reactive element. Used in Teflon and pharmaceuticals." },
  { id: 10, symbol: "Ne", name: "Neon", atomicMass: 20.18, period: 2, group: 18, category: "noble gas", electronegativity: null, electronConfig: "[He] 2s² 2p⁶", meltingPoint: 24.56, boilingPoint: 27.07, density: 0.0009, discoveredYear: 1898, description: "Used in lighting and lasers. Fifth most abundant element in the universe." },
  { id: 11, symbol: "Na", name: "Sodium", atomicMass: 22.99, period: 3, group: 1, category: "alkali metal", electronegativity: 0.93, electronConfig: "[Ne] 3s¹", meltingPoint: 370.95, boilingPoint: 1156, density: 0.968, discoveredYear: 1807, description: "Essential electrolyte. Reacts vigorously with water. Key component of salt (NaCl)." },
  { id: 12, symbol: "Mg", name: "Magnesium", atomicMass: 24.305, period: 3, group: 2, category: "alkaline earth metal", electronegativity: 1.31, electronConfig: "[Ne] 3s²", meltingPoint: 923, boilingPoint: 1363, density: 1.738, discoveredYear: 1755, description: "Lightweight structural metal. Essential for chlorophyll and ATP in biology." },
  { id: 13, symbol: "Al", name: "Aluminum", atomicMass: 26.982, period: 3, group: 13, category: "post-transition metal", electronegativity: 1.61, electronConfig: "[Ne] 3s² 3p¹", meltingPoint: 933.47, boilingPoint: 2792, density: 2.7, discoveredYear: 1825, description: "Most abundant metal in Earth's crust. Lightweight, corrosion-resistant, widely used in aerospace." },
  { id: 14, symbol: "Si", name: "Silicon", atomicMass: 28.086, period: 3, group: 14, category: "metalloid", electronegativity: 1.9, electronConfig: "[Ne] 3s² 3p²", meltingPoint: 1687, boilingPoint: 3538, density: 2.33, discoveredYear: 1824, description: "Foundation of modern electronics. Second most abundant element in Earth's crust." },
  { id: 15, symbol: "P", name: "Phosphorus", atomicMass: 30.974, period: 3, group: 15, category: "nonmetal", electronegativity: 2.19, electronConfig: "[Ne] 3s² 3p³", meltingPoint: 317.3, boilingPoint: 550, density: 1.82, discoveredYear: 1669, description: "Essential for DNA, RNA, and ATP. Key nutrient in fertilizers." },
  { id: 16, symbol: "S", name: "Sulfur", atomicMass: 32.06, period: 3, group: 16, category: "nonmetal", electronegativity: 2.58, electronConfig: "[Ne] 3s² 3p⁴", meltingPoint: 388.36, boilingPoint: 717.87, density: 2.067, discoveredYear: null, description: "Used in rubber vulcanization, fertilizers, and pharmaceuticals. Creates acid rain when burned." },
  { id: 17, symbol: "Cl", name: "Chlorine", atomicMass: 35.45, period: 3, group: 17, category: "halogen", electronegativity: 3.16, electronConfig: "[Ne] 3s² 3p⁵", meltingPoint: 171.6, boilingPoint: 239.11, density: 0.003214, discoveredYear: 1774, description: "Widely used disinfectant and bleaching agent. Forms table salt with sodium." },
  { id: 18, symbol: "Ar", name: "Argon", atomicMass: 39.948, period: 3, group: 18, category: "noble gas", electronegativity: null, electronConfig: "[Ne] 3s² 3p⁶", meltingPoint: 83.8, boilingPoint: 87.3, density: 0.001784, discoveredYear: 1894, description: "Third most abundant gas in Earth's atmosphere. Used in welding and lighting." },
  { id: 22, symbol: "Ti", name: "Titanium", atomicMass: 47.867, period: 4, group: 4, category: "transition metal", electronegativity: 1.54, electronConfig: "[Ar] 3d² 4s²", meltingPoint: 1941, boilingPoint: 3560, density: 4.507, discoveredYear: 1791, description: "Strong, lightweight, corrosion-resistant metal. Used in aerospace, medical implants, and pigments." },
  { id: 24, symbol: "Cr", name: "Chromium", atomicMass: 51.996, period: 4, group: 6, category: "transition metal", electronegativity: 1.66, electronConfig: "[Ar] 3d⁵ 4s¹", meltingPoint: 2180, boilingPoint: 2944, density: 7.19, discoveredYear: 1798, description: "Hard metal used in stainless steel and chrome plating. Essential trace element in humans." },
  { id: 26, symbol: "Fe", name: "Iron", atomicMass: 55.845, period: 4, group: 8, category: "transition metal", electronegativity: 1.83, electronConfig: "[Ar] 3d⁶ 4s²", meltingPoint: 1811, boilingPoint: 3134, density: 7.874, discoveredYear: null, description: "Most used metal in history. Core of Earth. Essential in hemoglobin for oxygen transport." },
  { id: 28, symbol: "Ni", name: "Nickel", atomicMass: 58.693, period: 4, group: 10, category: "transition metal", electronegativity: 1.91, electronConfig: "[Ar] 3d⁸ 4s²", meltingPoint: 1728, boilingPoint: 3186, density: 8.908, discoveredYear: 1751, description: "Used in stainless steel, batteries, and catalysts. Second most abundant element in Earth's core." },
  { id: 29, symbol: "Cu", name: "Copper", atomicMass: 63.546, period: 4, group: 11, category: "transition metal", electronegativity: 1.9, electronConfig: "[Ar] 3d¹⁰ 4s¹", meltingPoint: 1357.77, boilingPoint: 2835, density: 8.96, discoveredYear: null, description: "Excellent electrical conductor. Used in wiring, plumbing, and alloys like bronze and brass." },
  { id: 30, symbol: "Zn", name: "Zinc", atomicMass: 65.38, period: 4, group: 12, category: "transition metal", electronegativity: 1.65, electronConfig: "[Ar] 3d¹⁰ 4s²", meltingPoint: 692.88, boilingPoint: 1180, density: 7.134, discoveredYear: 1746, description: "Used in galvanizing steel and batteries. Essential trace element in hundreds of enzymes." },
  { id: 47, symbol: "Ag", name: "Silver", atomicMass: 107.868, period: 5, group: 11, category: "transition metal", electronegativity: 1.93, electronConfig: "[Kr] 4d¹⁰ 5s¹", meltingPoint: 1234.93, boilingPoint: 2435, density: 10.49, discoveredYear: null, description: "Best electrical and thermal conductor of all metals. Used in electronics, photography, and medicine." },
  { id: 50, symbol: "Sn", name: "Tin", atomicMass: 118.71, period: 5, group: 14, category: "post-transition metal", electronegativity: 1.96, electronConfig: "[Kr] 4d¹⁰ 5s² 5p²", meltingPoint: 505.08, boilingPoint: 2875, density: 7.287, discoveredYear: null, description: "Used in soldering, alloys, and tin plating. Important in the Bronze Age civilization." },
  { id: 56, symbol: "Ba", name: "Barium", atomicMass: 137.327, period: 6, group: 2, category: "alkaline earth metal", electronegativity: 0.89, electronConfig: "[Xe] 6s²", meltingPoint: 1000, boilingPoint: 2170, density: 3.594, discoveredYear: 1808, description: "Used in barium sulfate for medical imaging. Component of superconducting compounds." },
  { id: 57, symbol: "La", name: "Lanthanum", atomicMass: 138.905, period: 6, group: null, category: "lanthanide", electronegativity: 1.1, electronConfig: "[Xe] 5d¹ 6s²", meltingPoint: 1193, boilingPoint: 3737, density: 6.145, discoveredYear: 1839, description: "First lanthanide. Used in nickel-metal hydride batteries and camera lenses." },
  { id: 79, symbol: "Au", name: "Gold", atomicMass: 196.967, period: 6, group: 11, category: "transition metal", electronegativity: 2.54, electronConfig: "[Xe] 4f¹⁴ 5d¹⁰ 6s¹", meltingPoint: 1337.33, boilingPoint: 3129, density: 19.3, discoveredYear: null, description: "Noble metal resistant to oxidation. Used in electronics, jewelry, and as monetary standard." },
  { id: 82, symbol: "Pb", name: "Lead", atomicMass: 207.2, period: 6, group: 14, category: "post-transition metal", electronegativity: 2.33, electronConfig: "[Xe] 4f¹⁴ 5d¹⁰ 6s² 6p²", meltingPoint: 600.61, boilingPoint: 2022, density: 11.34, discoveredYear: null, description: "Dense, malleable metal. Used in radiation shielding and batteries. Highly toxic." },
  { id: 92, symbol: "U", name: "Uranium", atomicMass: 238.029, period: 7, group: null, category: "actinide", electronegativity: 1.38, electronConfig: "[Rn] 5f³ 6d¹ 7s²", meltingPoint: 1405.3, boilingPoint: 4404, density: 19.1, discoveredYear: 1789, description: "Radioactive heavy metal. Primary fuel for nuclear reactors. Basis of nuclear weapons." },
];

const LEARNING_PHASES = [
  {
    id: 1,
    name: "Subatomic Particle Mastery",
    description: "Learning the fundamental building blocks: protons, neutrons, electrons, quarks, and the forces that govern atomic structure.",
    status: "completed",
    progress: 100,
    itemsLearned: 47,
    totalItems: 47,
    insights: [
      "Quarks combine in triplets to form hadrons via the strong nuclear force",
      "Electron orbitals follow quantum mechanical probability distributions",
      "Nuclear binding energy peaks at iron-56, explaining stellar nucleosynthesis",
      "The Pauli exclusion principle governs electron configuration in atoms"
    ]
  },
  {
    id: 2,
    name: "Periodic Table & Elemental Properties",
    description: "Systematic study of all 118 elements, their properties, electron configurations, and periodic trends.",
    status: "completed",
    progress: 100,
    itemsLearned: 118,
    totalItems: 118,
    insights: [
      "Electronegativity increases across periods and decreases down groups",
      "Atomic radius follows inverse trend to ionization energy",
      "Noble gases maintain full outer shells, predicting reactivity",
      "Transition metals exhibit variable oxidation states due to d-orbital electrons"
    ]
  },
  {
    id: 3,
    name: "Chemical Bonding & Molecular Structures",
    description: "Analyzing ionic, covalent, metallic, and van der Waals bonds, VSEPR theory, and molecular geometry.",
    status: "active",
    progress: 78,
    itemsLearned: 39,
    totalItems: 50,
    insights: [
      "Hybridization (sp, sp², sp³) determines molecular geometry and properties",
      "Hydrogen bonding creates anomalous properties in water and biological molecules",
      "Crystal field theory explains color and magnetic properties of transition metal complexes",
      "Resonance structures distribute electron density across multiple bonds"
    ]
  },
  {
    id: 4,
    name: "Known Materials & Crystal Structures",
    description: "Indexing all known materials from NIST, Materials Project, OQMD, and AFLOW databases with full property characterization.",
    status: "active",
    progress: 42,
    itemsLearned: 210,
    totalItems: 500,
    insights: [
      "BCC and FCC crystal structures exhibit different slip systems and mechanical properties",
      "Perovskite structures (ABO₃) host remarkable ferroelectric and superconducting properties",
      "Graphene's 2D hexagonal lattice creates exceptional electronic and mechanical behavior",
      "Metallic glasses lack long-range order, exhibiting unique mechanical properties"
    ]
  },
  {
    id: 5,
    name: "Property Prediction & Modeling",
    description: "Developing quantum mechanical models using DFT, molecular dynamics, and machine learning to predict material properties.",
    status: "pending",
    progress: 8,
    itemsLearned: 4,
    totalItems: 50,
    insights: [
      "DFT calculations accurately predict band gaps within 10-15% of experimental values",
      "Neural network potentials can approximate ab initio forces at classical MD speed"
    ]
  },
  {
    id: 6,
    name: "Novel Material Discovery",
    description: "Generating and evaluating new chemical compositions for targeted properties: superconductors, topological insulators, ultra-hard materials, and more.",
    status: "pending",
    progress: 2,
    itemsLearned: 4,
    totalItems: 200,
    insights: [
      "Hydrogen-rich compounds under pressure show promise for room-temperature superconductivity"
    ]
  },
  {
    id: 7,
    name: "Superconductor Research (XGBoost+NN)",
    description: "Hybrid ML ensemble: XGBoost feature extraction + neural network refinement targeting room-temperature superconductors with Meissner effect, zero resistance, Cooper pair formation, and quantum coherence.",
    status: "pending",
    progress: 0,
    itemsLearned: 0,
    totalItems: 500,
    insights: []
  },
  {
    id: 8,
    name: "Synthesis Process Mapping",
    description: "Learning how every material is created: precursors, conditions, equipment, step-by-step procedures. Understanding creation processes like diamond formation under pressure or water from hydrogen and oxygen.",
    status: "pending",
    progress: 0,
    itemsLearned: 0,
    totalItems: 300,
    insights: []
  },
  {
    id: 9,
    name: "Chemical Reaction Knowledge",
    description: "Cataloguing every chemical reaction and lab process relevant to superconductor creation: oxide formation, high-pressure synthesis, hydrogenation, doping, crystal growth, and thin film deposition.",
    status: "pending",
    progress: 0,
    itemsLearned: 0,
    totalItems: 300,
    insights: []
  },
  {
    id: 10,
    name: "Computational Physics",
    description: "DFT-informed electronic structure, phonon spectra, electron-phonon coupling (lambda), Eliashberg Tc prediction, competing phase analysis, critical field computation, and correlation strength assessment.",
    status: "pending",
    progress: 0,
    itemsLearned: 0,
    totalItems: 200,
    insights: []
  },
  {
    id: 11,
    name: "Crystal Structure Prediction",
    description: "Predicting crystal structures from composition: space groups, lattice parameters, prototype matching, convex hull stability, metastability assessment, dimensionality classification, and synthesizability scoring.",
    status: "pending",
    progress: 0,
    itemsLearned: 0,
    totalItems: 150,
    insights: []
  },
  {
    id: 12,
    name: "Multi-Fidelity Screening",
    description: "5-stage pipeline: ML filter -> electronic structure -> phonon/e-ph coupling -> Tc prediction (Eliashberg/unconventional) -> synthesis feasibility. Each stage includes uncertainty quantification and negative result logging.",
    status: "pending",
    progress: 0,
    itemsLearned: 0,
    totalItems: 300,
    insights: []
  }
];

const MATERIALS_DATA = [
  { id: "nist-001", name: "Silicon Dioxide (Quartz)", formula: "SiO₂", spacegroup: "P3₂21", bandGap: 8.9, formationEnergy: -9.18, stability: 0.0, source: "NIST", properties: { hardness: 7.0, density: 2.65, dielectric: 3.9, structure: "trigonal" } },
  { id: "mp-002", name: "Titanium Dioxide (Rutile)", formula: "TiO₂", spacegroup: "P4₂/mnm", bandGap: 3.0, formationEnergy: -9.63, stability: 0.0, source: "Materials Project", properties: { hardness: 6.5, density: 4.23, dielectric: 86, structure: "tetragonal" } },
  { id: "mp-003", name: "Iron Pyrite", formula: "FeS₂", spacegroup: "Pa3̄", bandGap: 0.95, formationEnergy: -1.61, stability: 0.0, source: "Materials Project", properties: { hardness: 6.5, density: 5.01, structure: "cubic", magnetism: "diamagnetic" } },
  { id: "oqmd-004", name: "Yttrium Barium Copper Oxide", formula: "YBa₂Cu₃O₇", spacegroup: "Pmmm", bandGap: 0.0, formationEnergy: -14.6, stability: 0.002, source: "OQMD", properties: { criticalTemp: 92, structure: "orthorhombic", type: "high-Tc superconductor", discoveredYear: 1987 } },
  { id: "mp-005", name: "Gallium Arsenide", formula: "GaAs", spacegroup: "F4̄3m", bandGap: 1.42, formationEnergy: -0.74, stability: 0.0, source: "Materials Project", properties: { density: 5.32, electronMobility: 8500, structure: "zinc blende", application: "solar cells, LEDs" } },
  { id: "aflow-006", name: "Graphene", formula: "C", spacegroup: "P6/mmm", bandGap: 0.0, formationEnergy: 0.0, stability: 0.0, source: "AFLOW", properties: { youngsModulus: 1000, thermalConductivity: 5000, electronMobility: 200000, structure: "2D hexagonal" } },
  { id: "nist-007", name: "Alumina (Corundum)", formula: "Al₂O₃", spacegroup: "R3̄c", bandGap: 8.8, formationEnergy: -16.46, stability: 0.0, source: "NIST", properties: { hardness: 9.0, density: 3.99, meltingPoint: 2345, structure: "rhombohedral" } },
  { id: "mp-008", name: "Lithium Iron Phosphate", formula: "LiFePO₄", spacegroup: "Pnma", bandGap: 3.7, formationEnergy: -20.1, stability: 0.0, source: "Materials Project", properties: { capacity: 170, voltage: 3.4, structure: "olivine", application: "Li-ion batteries" } },
  { id: "oqmd-009", name: "Bismuth Telluride", formula: "Bi₂Te₃", spacegroup: "R3̄m", bandGap: 0.16, formationEnergy: -1.49, stability: 0.0, source: "OQMD", properties: { ZT: 1.0, seebeck: -287, structure: "rhombohedral", type: "topological insulator" } },
  { id: "aflow-010", name: "Tungsten Carbide", formula: "WC", spacegroup: "P6̄m2", bandGap: 0.0, formationEnergy: -0.41, stability: 0.0, source: "AFLOW", properties: { hardness: 9.5, density: 15.63, youngsModulus: 696, application: "cutting tools" } },
  { id: "nist-011", name: "Sodium Chloride (Halite)", formula: "NaCl", spacegroup: "Fm3̄m", bandGap: 8.5, formationEnergy: -3.38, stability: 0.0, source: "NIST", properties: { density: 2.165, meltingPoint: 1074, structure: "rock salt", solubility: 359 } },
  { id: "mp-012", name: "Boron Nitride (Cubic)", formula: "BN", spacegroup: "F4̄3m", bandGap: 6.0, formationEnergy: -2.88, stability: 0.0, source: "Materials Project", properties: { hardness: 9.5, density: 3.48, thermalConductivity: 740, structure: "zinc blende" } },
  { id: "oqmd-013", name: "Magnesium Diboride", formula: "MgB₂", spacegroup: "P6/mmm", bandGap: 0.0, formationEnergy: -1.07, stability: 0.0, source: "OQMD", properties: { criticalTemp: 39, structure: "hexagonal", type: "conventional superconductor", discoveredYear: 2001 } },
  { id: "aflow-014", name: "Hafnium Dioxide", formula: "HfO₂", spacegroup: "P2₁/c", bandGap: 5.7, formationEnergy: -11.63, stability: 0.0, source: "AFLOW", properties: { density: 9.68, dielectric: 25, structure: "monoclinic", application: "gate dielectric in transistors" } },
  { id: "nist-015", name: "Calcium Fluoride (Fluorite)", formula: "CaF₂", spacegroup: "Fm3̄m", bandGap: 11.8, formationEnergy: -8.13, stability: 0.0, source: "NIST", properties: { density: 3.18, hardness: 4.0, opticalWindow: "UV-IR", structure: "fluorite" } },
];

const NOVEL_PREDICTIONS = [
  {
    id: "pred-001",
    name: "Hydrogen-Rich Lanthanum Superhydride",
    formula: "LaH₁₀",
    predictedProperties: { criticalTemp: 250, pressure: 190, type: "phonon-mediated superconductor", confidence_tc: 0.78 },
    confidence: 0.87,
    targetApplication: "Near room-temperature superconductor for power transmission",
    status: "literature-reported",
    notes: "Published experimental result (Drozdov et al. 2019): superconductivity observed at ~250K under ~190 GPa. Requires extreme pressure — not ambient-pressure viable. Included as reference from literature, not a platform prediction."
  },
  {
    id: "pred-002",
    name: "Carbonaceous Sulfur Hydride",
    formula: "C-S-H",
    predictedProperties: { criticalTemp: 288, pressure: 267, type: "phonon-mediated", estimated_tc_error: "±5K" },
    confidence: 0.72,
    targetApplication: "Room-temperature superconductor at extreme pressures",
    status: "predicted",
    notes: "Theoretical prediction based on high-throughput DFT screening of H-rich ternary systems."
  },
  {
    id: "pred-003",
    name: "Boron-Carbon Nitride Superalloy",
    formula: "B₂C₃N₄",
    predictedProperties: { hardness: 95, youngsModulus: 1100, bulkModulus: 450, density: 3.42 },
    confidence: 0.81,
    targetApplication: "Ultra-hard coating material harder than diamond for industrial cutting",
    status: "under_review",
    notes: "Novel ternary nitride predicted to exceed diamond hardness through bonding topology optimization."
  },
  {
    id: "pred-004",
    name: "Niobium-Titanium-Nitrogen Perovskite",
    formula: "NbTi₂N₃",
    predictedProperties: { criticalTemp: 45, structure: "perovskite-like", bandGap: 0.0, formation_energy: -2.1 },
    confidence: 0.65,
    targetApplication: "High-temperature superconducting wire for fusion reactor magnets",
    status: "predicted",
    notes: "Computational screening of nitrogen-based perovskites identifies this compound as highly stable."
  },
  {
    id: "pred-005",
    name: "Bismuth-Antimony Topological Superconductor",
    formula: "Bi₀.₅Sb₁.₅Te₃",
    predictedProperties: { bandGap: 0.0, ZT: 1.8, majoranaFermions: true, criticalTemp: 3.8 },
    confidence: 0.73,
    targetApplication: "Topological quantum computing qubit substrate",
    status: "under_review",
    notes: "Predicted to host Majorana zero modes at interfaces, enabling fault-tolerant quantum computation."
  },
  {
    id: "pred-006",
    name: "Copper-Oxygen-Fluorine Compound",
    formula: "Cu₂OF",
    predictedProperties: { criticalTemp: 175, structure: "layered", pressure: 0, type: "cuprate-like" },
    confidence: 0.58,
    targetApplication: "Ambient-pressure high-Tc superconductor for lossless power grids",
    status: "predicted",
    notes: "Novel fluorine-doped cuprate structure predicted to exceed known cuprate Tc values at ambient pressure."
  }
];

const RESEARCH_LOGS = [
  { phase: "phase-6", event: "Literature reference indexed", detail: "LaH₁₀ literature-reported Tc ~250K under ~190 GPa (Drozdov et al. 2019)", dataSource: "Published Literature" },
  { phase: "phase-4", event: "AFLOW sync completed", detail: "Indexed 2,847 new binary alloy structures from AFLOW library", dataSource: "AFLOW" },
  { phase: "phase-4", event: "Materials Project fetch", detail: "Retrieved 1,203 perovskite compounds with DFT-computed band gaps", dataSource: "Materials Project" },
  { phase: "phase-3", event: "Bonding analysis complete", detail: "Classified 892 hydrogen-bond networks in metal-organic frameworks", dataSource: "Internal" },
  { phase: "phase-5", event: "DFT batch completed", detail: "Computed formation energies for 340 ternary sulfide compounds", dataSource: "DFT Engine" },
  { phase: "phase-4", event: "OQMD integration", detail: "Fetched stability data for 15,420 oxide compounds", dataSource: "OQMD" },
  { phase: "phase-2", event: "Element database completed", detail: "All 118 elements catalogued with full spectroscopic data", dataSource: "NIST WebBook" },
  { phase: "phase-6", event: "Topology screening", detail: "Screened 4,200 candidate topological insulators using symmetry indicators", dataSource: "Internal ML Model" },
  { phase: "phase-5", event: "Neural potential trained", detail: "Graph neural network potential trained on 180K DFT trajectories with 98.2% accuracy", dataSource: "DFT Engine" },
  { phase: "phase-3", event: "Crystal structure prediction", detail: "Minima basin-hopping found 23 new polymorphs for TiO₂", dataSource: "Internal" },
  { phase: "phase-4", event: "NIST WebBook sync", detail: "Updated thermodynamic data for 28,000 compounds", dataSource: "NIST" },
  { phase: "phase-6", event: "High-throughput screening", detail: "Evaluated 50,000 ternary hydrides for superconducting Tc", dataSource: "Internal ML Model" },
];

export async function seedDatabase() {
  const [res] = await db.select({ total: count() }).from(elements);
  if (Number(res?.total ?? 0) > 0) {
    console.log("Database already seeded, skipping.");
    return;
  }

  console.log("Seeding database...");

  for (const el of ELEMENTS_DATA) {
    await db.insert(elements).values(el).onConflictDoNothing();
  }

  for (const phase of LEARNING_PHASES) {
    await db.insert(learningPhases).values(phase as any).onConflictDoUpdate({
      target: learningPhases.id,
      set: { progress: phase.progress, itemsLearned: phase.itemsLearned, status: phase.status }
    });
  }

  for (const mat of MATERIALS_DATA) {
    await db.insert(materials).values(mat as any).onConflictDoNothing();
  }

  for (const pred of NOVEL_PREDICTIONS) {
    await db.insert(novelPredictions).values(pred as any).onConflictDoNothing();
  }

  for (const log of RESEARCH_LOGS) {
    await db.insert(researchLogs).values(log as any);
  }

  console.log("Database seeded successfully.");
}
