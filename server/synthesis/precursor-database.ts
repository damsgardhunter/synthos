export interface Precursor {
  name: string;
  formula: string;
  element: string;
  availability: number;
  purity: string;
  costTier: "low" | "medium" | "high" | "very-high";
  supplierAvailability: number;
  safetyNotes: string;
  preferredMethods: string[];
}

export interface PrecursorSelection {
  element: string;
  precursor: Precursor;
  alternates: Precursor[];
}

export interface PrecursorAvailabilityResult {
  overallScore: number;
  selections: PrecursorSelection[];
  costEstimate: string;
  bottleneckElement: string | null;
}

const PRECURSOR_DB: Precursor[] = [
  { name: "Yttrium oxide", formula: "Y2O3", element: "Y", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.95, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Yttrium chloride", formula: "YCl3", element: "Y", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD", "sol-gel"] },
  { name: "Yttrium metal", formula: "Y", element: "Y", availability: 0.7, purity: "99.9%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Flammable as powder", preferredMethods: ["arc-melting"] },

  { name: "Barium carbonate", formula: "BaCO3", element: "Ba", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Toxic if ingested", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Barium oxide", formula: "BaO", element: "Ba", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Reacts with water/CO2", preferredMethods: ["solid-state"] },
  { name: "Barium nitrate", formula: "Ba(NO3)2", element: "Ba", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Oxidizer, toxic", preferredMethods: ["sol-gel"] },
  { name: "Barium metal", formula: "Ba", element: "Ba", availability: 0.5, purity: "99.5%", costTier: "high", supplierAvailability: 0.6, safetyNotes: "Highly reactive with air/water", preferredMethods: ["arc-melting"] },

  { name: "Copper(II) oxide", formula: "CuO", element: "Cu", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "ball-milling"] },
  { name: "Copper(I) oxide", formula: "Cu2O", element: "Cu", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Copper nitrate", formula: "Cu(NO3)2", element: "Cu", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Oxidizer", preferredMethods: ["sol-gel"] },
  { name: "Copper metal", formula: "Cu", element: "Cu", availability: 0.98, purity: "99.999%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Lanthanum oxide", formula: "La2O3", element: "La", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.95, safetyNotes: "Hygroscopic", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Lanthanum chloride", formula: "LaCl3", element: "La", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD", "sol-gel"] },
  { name: "Lanthanum metal", formula: "La", element: "La", availability: 0.7, purity: "99.9%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Pyrophoric as powder", preferredMethods: ["arc-melting"] },

  { name: "Strontium carbonate", formula: "SrCO3", element: "Sr", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Strontium oxide", formula: "SrO", element: "Sr", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Reacts with moisture", preferredMethods: ["solid-state"] },
  { name: "Strontium metal", formula: "Sr", element: "Sr", availability: 0.55, purity: "99.5%", costTier: "high", supplierAvailability: 0.6, safetyNotes: "Highly reactive", preferredMethods: ["arc-melting"] },

  { name: "Calcium carbonate", formula: "CaCO3", element: "Ca", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Calcium oxide", formula: "CaO", element: "Ca", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Reacts with moisture", preferredMethods: ["solid-state"] },
  { name: "Calcium metal", formula: "Ca", element: "Ca", availability: 0.7, purity: "99.5%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Reactive with water", preferredMethods: ["arc-melting"] },
  { name: "Calcium hydride", formula: "CaH2", element: "Ca", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Reacts vigorously with water", preferredMethods: ["high-pressure"] },

  { name: "Iron(III) oxide", formula: "Fe2O3", element: "Fe", availability: 0.98, purity: "99.9%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state", "ball-milling"] },
  { name: "Iron metal", formula: "Fe", element: "Fe", availability: 0.99, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },
  { name: "Iron(III) chloride", formula: "FeCl3", element: "Fe", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Corrosive", preferredMethods: ["CVD", "sol-gel"] },

  { name: "Arsenic trioxide", formula: "As2O3", element: "As", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Highly toxic, carcinogenic", preferredMethods: ["solid-state"] },
  { name: "Arsenic metal", formula: "As", element: "As", availability: 0.75, purity: "99.999%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Toxic", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Selenium powder", formula: "Se", element: "Se", availability: 0.85, purity: "99.999%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Toxic, handle in fume hood", preferredMethods: ["arc-melting", "solid-state", "sputtering"] },
  { name: "Selenium dioxide", formula: "SeO2", element: "Se", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Toxic", preferredMethods: ["sol-gel"] },

  { name: "Tellurium metal", formula: "Te", element: "Te", availability: 0.8, purity: "99.999%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Moderate toxicity", preferredMethods: ["arc-melting", "sputtering"] },
  { name: "Tellurium dioxide", formula: "TeO2", element: "Te", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Moderate toxicity", preferredMethods: ["solid-state"] },

  { name: "Niobium pentoxide", formula: "Nb2O5", element: "Nb", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Niobium metal", formula: "Nb", element: "Nb", availability: 0.85, purity: "99.95%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },
  { name: "Niobium chloride", formula: "NbCl5", element: "Nb", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Moisture sensitive, corrosive", preferredMethods: ["CVD"] },

  { name: "Titanium dioxide", formula: "TiO2", element: "Ti", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Titanium metal", formula: "Ti", element: "Ti", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Flammable as powder", preferredMethods: ["arc-melting", "sputtering"] },
  { name: "Titanium tetrachloride", formula: "TiCl4", element: "Ti", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Fumes in air, corrosive", preferredMethods: ["CVD"] },

  { name: "Magnesium oxide", formula: "MgO", element: "Mg", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state"] },
  { name: "Magnesium metal", formula: "Mg", element: "Mg", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Flammable", preferredMethods: ["arc-melting"] },
  { name: "Magnesium chloride", formula: "MgCl2", element: "Mg", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Hygroscopic", preferredMethods: ["CVD", "sol-gel"] },
  { name: "Magnesium diboride", formula: "MgB2", element: "Mg", availability: 0.7, purity: "99.9%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Safe", preferredMethods: ["ball-milling"] },

  { name: "Boron trioxide", formula: "B2O3", element: "B", availability: 0.9, purity: "99.99%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Boron powder", formula: "B", element: "B", availability: 0.85, purity: "99.99%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Flammable as fine powder", preferredMethods: ["arc-melting", "ball-milling"] },
  { name: "Boric acid", formula: "H3BO3", element: "B", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Low toxicity", preferredMethods: ["sol-gel"] },

  { name: "Tin dioxide", formula: "SnO2", element: "Sn", availability: 0.9, purity: "99.99%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Safe", preferredMethods: ["solid-state"] },
  { name: "Tin metal", formula: "Sn", element: "Sn", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Lead oxide", formula: "PbO", element: "Pb", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Toxic, handle with care", preferredMethods: ["solid-state"] },
  { name: "Lead nitrate", formula: "Pb(NO3)2", element: "Pb", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Toxic, oxidizer", preferredMethods: ["sol-gel"] },
  { name: "Lead metal", formula: "Pb", element: "Pb", availability: 0.9, purity: "99.99%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Toxic", preferredMethods: ["arc-melting"] },

  { name: "Bismuth trioxide", formula: "Bi2O3", element: "Bi", availability: 0.85, purity: "99.99%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Bismuth metal", formula: "Bi", element: "Bi", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Low toxicity", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Thallium trioxide", formula: "Tl2O3", element: "Tl", availability: 0.5, purity: "99.9%", costTier: "high", supplierAvailability: 0.55, safetyNotes: "Extremely toxic", preferredMethods: ["solid-state"] },
  { name: "Thallium nitrate", formula: "TlNO3", element: "Tl", availability: 0.45, purity: "99.9%", costTier: "high", supplierAvailability: 0.5, safetyNotes: "Extremely toxic", preferredMethods: ["sol-gel"] },

  { name: "Mercury oxide", formula: "HgO", element: "Hg", availability: 0.4, purity: "99.9%", costTier: "high", supplierAvailability: 0.45, safetyNotes: "Highly toxic, volatile", preferredMethods: ["solid-state"] },
  { name: "Mercury chloride", formula: "HgCl2", element: "Hg", availability: 0.35, purity: "99.9%", costTier: "high", supplierAvailability: 0.4, safetyNotes: "Highly toxic", preferredMethods: ["CVD"] },

  { name: "Zirconium dioxide", formula: "ZrO2", element: "Zr", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Safe", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Zirconium tetrachloride", formula: "ZrCl4", element: "Zr", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD"] },
  { name: "Zirconium metal", formula: "Zr", element: "Zr", availability: 0.85, purity: "99.95%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Flammable as powder", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Hafnium dioxide", formula: "HfO2", element: "Hf", availability: 0.8, purity: "99.99%", costTier: "high", supplierAvailability: 0.82, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "CVD"] },
  { name: "Hafnium tetrachloride", formula: "HfCl4", element: "Hf", availability: 0.7, purity: "99.9%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD"] },
  { name: "Hafnium metal", formula: "Hf", element: "Hf", availability: 0.65, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.7, safetyNotes: "Flammable as powder", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Vanadium pentoxide", formula: "V2O5", element: "V", availability: 0.9, purity: "99.99%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Toxic", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Vanadium metal", formula: "V", element: "V", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Chromium trioxide", formula: "Cr2O3", element: "Cr", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Avoid Cr(VI) compounds", preferredMethods: ["solid-state"] },
  { name: "Chromium metal", formula: "Cr", element: "Cr", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Manganese dioxide", formula: "MnO2", element: "Mn", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Oxidizer", preferredMethods: ["solid-state"] },
  { name: "Manganese carbonate", formula: "MnCO3", element: "Mn", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Manganese metal", formula: "Mn", element: "Mn", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Safe", preferredMethods: ["arc-melting"] },

  { name: "Cobalt oxide", formula: "CoO", element: "Co", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Possible carcinogen", preferredMethods: ["solid-state"] },
  { name: "Cobalt nitrate", formula: "Co(NO3)2", element: "Co", availability: 0.9, purity: "99.9%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Oxidizer, possible carcinogen", preferredMethods: ["sol-gel"] },
  { name: "Cobalt metal", formula: "Co", element: "Co", availability: 0.85, purity: "99.99%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Possible carcinogen", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Nickel oxide", formula: "NiO", element: "Ni", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Possible carcinogen as powder", preferredMethods: ["solid-state"] },
  { name: "Nickel nitrate", formula: "Ni(NO3)2", element: "Ni", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Oxidizer", preferredMethods: ["sol-gel"] },
  { name: "Nickel metal", formula: "Ni", element: "Ni", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Sensitizer", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Zinc oxide", formula: "ZnO", element: "Zn", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Zinc metal", formula: "Zn", element: "Zn", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["arc-melting"] },

  { name: "Aluminum oxide", formula: "Al2O3", element: "Al", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Aluminum metal", formula: "Al", element: "Al", availability: 0.99, purity: "99.999%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Flammable as powder", preferredMethods: ["arc-melting", "sputtering"] },
  { name: "Aluminum chloride", formula: "AlCl3", element: "Al", availability: 0.95, purity: "99.9%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Corrosive, moisture sensitive", preferredMethods: ["CVD"] },

  { name: "Gallium trioxide", formula: "Ga2O3", element: "Ga", availability: 0.8, purity: "99.99%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Gallium metal", formula: "Ga", element: "Ga", availability: 0.85, purity: "99.999%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Safe, low melting point", preferredMethods: ["arc-melting", "flux-growth"] },

  { name: "Indium trioxide", formula: "In2O3", element: "In", availability: 0.75, purity: "99.99%", costTier: "high", supplierAvailability: 0.8, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sputtering"] },
  { name: "Indium metal", formula: "In", element: "In", availability: 0.8, purity: "99.999%", costTier: "high", supplierAvailability: 0.82, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Germanium dioxide", formula: "GeO2", element: "Ge", availability: 0.8, purity: "99.99%", costTier: "high", supplierAvailability: 0.82, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Germanium metal", formula: "Ge", element: "Ge", availability: 0.75, purity: "99.999%", costTier: "high", supplierAvailability: 0.8, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Silicon dioxide", formula: "SiO2", element: "Si", availability: 0.99, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Inhalation hazard as fine powder", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Silicon metal", formula: "Si", element: "Si", availability: 0.99, purity: "99.999%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering", "CVD"] },

  { name: "Phosphorus pentoxide", formula: "P2O5", element: "P", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Corrosive, hygroscopic", preferredMethods: ["solid-state"] },
  { name: "Ammonium dihydrogen phosphate", formula: "NH4H2PO4", element: "P", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Safe", preferredMethods: ["sol-gel"] },

  { name: "Sulfur powder", formula: "S", element: "S", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Flammable", preferredMethods: ["solid-state", "arc-melting"] },
  { name: "Sodium sulfide", formula: "Na2S", element: "S", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Toxic, corrosive", preferredMethods: ["sol-gel"] },

  { name: "Nitrogen gas", formula: "N2", element: "N", availability: 0.99, purity: "99.999%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Asphyxiant in confined spaces", preferredMethods: ["sputtering", "CVD"] },
  { name: "Ammonia", formula: "NH3", element: "N", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Toxic, corrosive gas", preferredMethods: ["CVD"] },
  { name: "Lithium nitride", formula: "Li3N", element: "N", availability: 0.7, purity: "99.5%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Moisture sensitive", preferredMethods: ["solid-state", "high-pressure"] },

  { name: "Oxygen gas", formula: "O2", element: "O", availability: 0.99, purity: "99.999%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Supports combustion", preferredMethods: ["solid-state", "CVD", "sputtering"] },

  { name: "Hydrogen gas", formula: "H2", element: "H", availability: 0.95, purity: "99.999%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Highly flammable, explosive mixtures", preferredMethods: ["high-pressure", "CVD"] },
  { name: "Calcium hydride", formula: "CaH2", element: "H", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Reacts with water", preferredMethods: ["high-pressure", "solid-state"] },
  { name: "Sodium hydride", formula: "NaH", element: "H", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.9, safetyNotes: "Pyrophoric, water-reactive", preferredMethods: ["high-pressure"] },
  { name: "Lithium hydride", formula: "LiH", element: "H", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Pyrophoric, water-reactive", preferredMethods: ["high-pressure"] },
  { name: "Ammonia borane", formula: "NH3BH3", element: "H", availability: 0.7, purity: "99%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Hydrogen source, handle in glovebox", preferredMethods: ["high-pressure"] },

  { name: "Lithium carbonate", formula: "Li2CO3", element: "Li", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Lithium hydroxide", formula: "LiOH", element: "Li", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Corrosive", preferredMethods: ["sol-gel"] },
  { name: "Lithium metal", formula: "Li", element: "Li", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Highly reactive, fire hazard", preferredMethods: ["arc-melting"] },

  { name: "Sodium carbonate", formula: "Na2CO3", element: "Na", availability: 0.98, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Sodium hydroxide", formula: "NaOH", element: "Na", availability: 0.99, purity: "99.9%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Corrosive", preferredMethods: ["sol-gel"] },
  { name: "Sodium metal", formula: "Na", element: "Na", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Highly reactive with water", preferredMethods: ["arc-melting"] },

  { name: "Potassium carbonate", formula: "K2CO3", element: "K", availability: 0.95, purity: "99.99%", costTier: "low", supplierAvailability: 0.98, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Potassium hydroxide", formula: "KOH", element: "K", availability: 0.98, purity: "99.9%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Corrosive", preferredMethods: ["sol-gel"] },
  { name: "Potassium metal", formula: "K", element: "K", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Extremely reactive with water", preferredMethods: ["arc-melting"] },

  { name: "Cesium carbonate", formula: "Cs2CO3", element: "Cs", availability: 0.7, purity: "99.9%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Hygroscopic", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Cesium chloride", formula: "CsCl", element: "Cs", availability: 0.75, purity: "99.9%", costTier: "high", supplierAvailability: 0.78, safetyNotes: "Low toxicity", preferredMethods: ["flux-growth", "sol-gel"] },

  { name: "Rubidium carbonate", formula: "Rb2CO3", element: "Rb", availability: 0.6, purity: "99.9%", costTier: "high", supplierAvailability: 0.65, safetyNotes: "Hygroscopic", preferredMethods: ["solid-state"] },
  { name: "Rubidium chloride", formula: "RbCl", element: "Rb", availability: 0.65, purity: "99.9%", costTier: "high", supplierAvailability: 0.7, safetyNotes: "Low toxicity", preferredMethods: ["flux-growth"] },

  { name: "Tungsten trioxide", formula: "WO3", element: "W", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Tungsten metal", formula: "W", element: "W", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Molybdenum trioxide", formula: "MoO3", element: "Mo", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Molybdenum metal", formula: "Mo", element: "Mo", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Ruthenium dioxide", formula: "RuO2", element: "Ru", availability: 0.6, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.65, safetyNotes: "Toxic", preferredMethods: ["solid-state"] },
  { name: "Ruthenium chloride", formula: "RuCl3", element: "Ru", availability: 0.6, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.65, safetyNotes: "Toxic", preferredMethods: ["sol-gel", "CVD"] },

  { name: "Rhodium trioxide", formula: "Rh2O3", element: "Rh", availability: 0.5, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Expensive PGM", preferredMethods: ["solid-state"] },
  { name: "Rhodium chloride", formula: "RhCl3", element: "Rh", availability: 0.5, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Expensive PGM", preferredMethods: ["sol-gel"] },

  { name: "Palladium chloride", formula: "PdCl2", element: "Pd", availability: 0.6, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.65, safetyNotes: "Sensitizer", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Palladium metal", formula: "Pd", element: "Pd", availability: 0.65, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.7, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Iridium dioxide", formula: "IrO2", element: "Ir", availability: 0.45, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.5, safetyNotes: "Expensive PGM", preferredMethods: ["solid-state"] },
  { name: "Iridium chloride", formula: "IrCl3", element: "Ir", availability: 0.45, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.5, safetyNotes: "Expensive PGM", preferredMethods: ["sol-gel"] },
  { name: "Iridium metal", formula: "Ir", element: "Ir", availability: 0.4, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.45, safetyNotes: "Very expensive", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Platinum chloride", formula: "PtCl2", element: "Pt", availability: 0.55, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.6, safetyNotes: "Expensive", preferredMethods: ["sol-gel"] },
  { name: "Platinum metal", formula: "Pt", element: "Pt", availability: 0.6, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.65, safetyNotes: "Safe but expensive", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Chloroauric acid", formula: "HAuCl4", element: "Au", availability: 0.6, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.65, safetyNotes: "Corrosive", preferredMethods: ["sol-gel"] },
  { name: "Gold metal", formula: "Au", element: "Au", availability: 0.7, purity: "99.999%", costTier: "very-high", supplierAvailability: 0.75, safetyNotes: "Safe but expensive", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Silver nitrate", formula: "AgNO3", element: "Ag", availability: 0.9, purity: "99.99%", costTier: "medium", supplierAvailability: 0.92, safetyNotes: "Corrosive, stains skin", preferredMethods: ["sol-gel"] },
  { name: "Silver metal", formula: "Ag", element: "Ag", availability: 0.95, purity: "99.99%", costTier: "medium", supplierAvailability: 0.95, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Cadmium oxide", formula: "CdO", element: "Cd", availability: 0.7, purity: "99.9%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Toxic, carcinogenic", preferredMethods: ["solid-state"] },
  { name: "Cadmium chloride", formula: "CdCl2", element: "Cd", availability: 0.7, purity: "99.9%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Toxic, carcinogenic", preferredMethods: ["CVD"] },

  { name: "Rhenium heptoxide", formula: "Re2O7", element: "Re", availability: 0.5, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Hygroscopic, corrosive", preferredMethods: ["solid-state"] },
  { name: "Rhenium metal", formula: "Re", element: "Re", availability: 0.55, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.6, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Osmium tetroxide", formula: "OsO4", element: "Os", availability: 0.35, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.4, safetyNotes: "Extremely toxic and volatile", preferredMethods: ["solid-state"] },
  { name: "Osmium metal", formula: "Os", element: "Os", availability: 0.35, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.4, safetyNotes: "OsO4 fumes are lethal", preferredMethods: ["arc-melting"] },

  { name: "Tantalum pentoxide", formula: "Ta2O5", element: "Ta", availability: 0.85, purity: "99.99%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Tantalum metal", formula: "Ta", element: "Ta", availability: 0.85, purity: "99.95%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Safe", preferredMethods: ["arc-melting", "sputtering"] },

  { name: "Scandium trioxide", formula: "Sc2O3", element: "Sc", availability: 0.55, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.6, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Scandium metal", formula: "Sc", element: "Sc", availability: 0.4, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.45, safetyNotes: "Rare, expensive", preferredMethods: ["arc-melting"] },

  { name: "Thorium dioxide", formula: "ThO2", element: "Th", availability: 0.3, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.35, safetyNotes: "Radioactive, requires license", preferredMethods: ["solid-state"] },

  { name: "Uranium dioxide", formula: "UO2", element: "U", availability: 0.25, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.3, safetyNotes: "Radioactive, requires license", preferredMethods: ["solid-state"] },
  { name: "Triuranium octoxide", formula: "U3O8", element: "U", availability: 0.25, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.3, safetyNotes: "Radioactive, requires license", preferredMethods: ["solid-state"] },

  { name: "Cerium dioxide", formula: "CeO2", element: "Ce", availability: 0.9, purity: "99.99%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Cerium sesquioxide", formula: "Ce2O3", element: "Ce", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },

  { name: "Praseodymium oxide", formula: "Pr6O11", element: "Pr", availability: 0.8, purity: "99.99%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Praseodymium chloride", formula: "PrCl3", element: "Pr", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD", "sol-gel"] },

  { name: "Neodymium oxide", formula: "Nd2O3", element: "Nd", availability: 0.85, purity: "99.99%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Neodymium chloride", formula: "NdCl3", element: "Nd", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD"] },

  { name: "Samarium oxide", formula: "Sm2O3", element: "Sm", availability: 0.8, purity: "99.99%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Europium oxide", formula: "Eu2O3", element: "Eu", availability: 0.7, purity: "99.99%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Gadolinium oxide", formula: "Gd2O3", element: "Gd", availability: 0.8, purity: "99.99%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Terbium oxide", formula: "Tb4O7", element: "Tb", availability: 0.6, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.65, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Dysprosium oxide", formula: "Dy2O3", element: "Dy", availability: 0.7, purity: "99.99%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Holmium oxide", formula: "Ho2O3", element: "Ho", availability: 0.65, purity: "99.99%", costTier: "high", supplierAvailability: 0.7, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Erbium oxide", formula: "Er2O3", element: "Er", availability: 0.7, purity: "99.99%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Thulium oxide", formula: "Tm2O3", element: "Tm", availability: 0.5, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },
  { name: "Ytterbium oxide", formula: "Yb2O3", element: "Yb", availability: 0.7, purity: "99.99%", costTier: "high", supplierAvailability: 0.75, safetyNotes: "Low toxicity", preferredMethods: ["solid-state", "sol-gel"] },
  { name: "Lutetium oxide", formula: "Lu2O3", element: "Lu", availability: 0.5, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Low toxicity, expensive", preferredMethods: ["solid-state"] },

  { name: "Beryllium oxide", formula: "BeO", element: "Be", availability: 0.6, purity: "99.9%", costTier: "high", supplierAvailability: 0.65, safetyNotes: "Extremely toxic dust, carcinogenic", preferredMethods: ["solid-state"] },
  { name: "Beryllium metal", formula: "Be", element: "Be", availability: 0.5, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Extremely toxic dust", preferredMethods: ["arc-melting"] },

  { name: "Fluorine gas (dilute)", formula: "F2", element: "F", availability: 0.5, purity: "99.9%", costTier: "medium", supplierAvailability: 0.55, safetyNotes: "Extremely toxic, corrosive", preferredMethods: ["CVD"] },
  { name: "Lithium fluoride", formula: "LiF", element: "F", availability: 0.85, purity: "99.99%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Toxic", preferredMethods: ["solid-state", "flux-growth"] },
  { name: "Calcium fluoride", formula: "CaF2", element: "F", availability: 0.9, purity: "99.99%", costTier: "low", supplierAvailability: 0.95, safetyNotes: "Low toxicity", preferredMethods: ["solid-state"] },

  { name: "Chlorine gas", formula: "Cl2", element: "Cl", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.9, safetyNotes: "Toxic, corrosive gas", preferredMethods: ["CVD"] },
  { name: "Sodium chloride", formula: "NaCl", element: "Cl", availability: 0.99, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["flux-growth"] },

  { name: "Carbon powder", formula: "C", element: "C", availability: 0.99, purity: "99.99%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Combustible dust", preferredMethods: ["solid-state", "arc-melting"] },
  { name: "Graphite", formula: "C", element: "C", availability: 0.99, purity: "99.999%", costTier: "low", supplierAvailability: 0.99, safetyNotes: "Safe", preferredMethods: ["solid-state", "CVD"] },

  // ── Rare-earth hydrides ──────────────────────────────────────────────────────
  // Critical precursors for high-pressure hydride superconductors (LaH10, YH6/9,
  // CeH9, ScH12, LuH2±δ etc.).  These are typically synthesised in-situ from the
  // metal + H2, but pre-formed hydrides allow direct diamond-anvil-cell loading
  // without an intermediate hydrogenation step, improving run reproducibility.
  { name: "Lanthanum trihydride", formula: "LaH3", element: "La", availability: 0.55, purity: "99.5%", costTier: "high", supplierAvailability: 0.6, safetyNotes: "Pyrophoric, store in inert atmosphere", preferredMethods: ["high-pressure"] },
  { name: "Yttrium trihydride", formula: "YH3", element: "Y", availability: 0.55, purity: "99.5%", costTier: "high", supplierAvailability: 0.6, safetyNotes: "Pyrophoric, store in inert atmosphere", preferredMethods: ["high-pressure"] },
  { name: "Cerium dihydride", formula: "CeH2", element: "Ce", availability: 0.5, purity: "99.5%", costTier: "high", supplierAvailability: 0.55, safetyNotes: "Pyrophoric", preferredMethods: ["high-pressure"] },
  { name: "Praseodymium dihydride", formula: "PrH2", element: "Pr", availability: 0.45, purity: "99.5%", costTier: "high", supplierAvailability: 0.5, safetyNotes: "Pyrophoric", preferredMethods: ["high-pressure"] },
  { name: "Neodymium dihydride", formula: "NdH2", element: "Nd", availability: 0.45, purity: "99.5%", costTier: "high", supplierAvailability: 0.5, safetyNotes: "Pyrophoric", preferredMethods: ["high-pressure"] },
  { name: "Scandium dihydride", formula: "ScH2", element: "Sc", availability: 0.35, purity: "99%", costTier: "very-high", supplierAvailability: 0.4, safetyNotes: "Pyrophoric, rare", preferredMethods: ["high-pressure"] },
  { name: "Ytterbium dihydride", formula: "YbH2", element: "Yb", availability: 0.5, purity: "99.5%", costTier: "high", supplierAvailability: 0.55, safetyNotes: "Pyrophoric", preferredMethods: ["high-pressure"] },
  { name: "Lutetium dihydride", formula: "LuH2", element: "Lu", availability: 0.35, purity: "99%", costTier: "very-high", supplierAvailability: 0.4, safetyNotes: "Pyrophoric, expensive", preferredMethods: ["high-pressure"] },
  { name: "Thorium dihydride", formula: "ThH2", element: "Th", availability: 0.25, purity: "99%", costTier: "very-high", supplierAvailability: 0.3, safetyNotes: "Radioactive, requires license", preferredMethods: ["high-pressure"] },
  // Rare-earth hydrides also serve as hydrogen donors in DAC experiments
  { name: "Lanthanum trihydride (H-source)", formula: "LaH3", element: "H", availability: 0.5, purity: "99.5%", costTier: "high", supplierAvailability: 0.55, safetyNotes: "Pyrophoric; used as solid H-source in DAC", preferredMethods: ["high-pressure"] },
  { name: "Yttrium trihydride (H-source)", formula: "YH3", element: "H", availability: 0.5, purity: "99.5%", costTier: "high", supplierAvailability: 0.55, safetyNotes: "Pyrophoric; used as solid H-source in DAC", preferredMethods: ["high-pressure"] },

  // ── Additional arsenic precursors for pnictide SC routes ─────────────────────
  // As2O3 and elemental As are already in the DB; the following compounds are the
  // actual starting materials for FeAs-based (1111, 122, 111) and related families.
  { name: "Iron arsenide", formula: "FeAs", element: "As", availability: 0.65, purity: "99.9%", costTier: "medium", supplierAvailability: 0.7, safetyNotes: "Toxic; handle in fume hood", preferredMethods: ["solid-state", "arc-melting"] },
  { name: "Cobalt arsenide", formula: "CoAs", element: "As", availability: 0.55, purity: "99.9%", costTier: "medium", supplierAvailability: 0.6, safetyNotes: "Toxic", preferredMethods: ["solid-state"] },
  { name: "Nickel arsenide", formula: "NiAs", element: "As", availability: 0.55, purity: "99.9%", costTier: "medium", supplierAvailability: 0.6, safetyNotes: "Toxic", preferredMethods: ["solid-state", "arc-melting"] },
  { name: "Indium arsenide", formula: "InAs", element: "As", availability: 0.6, purity: "99.999%", costTier: "high", supplierAvailability: 0.65, safetyNotes: "Toxic, semiconductor-grade", preferredMethods: ["sputtering", "CVD"] },
  { name: "Gallium arsenide", formula: "GaAs", element: "As", availability: 0.7, purity: "99.999%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Toxic, semiconductor-grade", preferredMethods: ["sputtering", "CVD"] },
  { name: "Arsenic sulfide", formula: "As2S3", element: "As", availability: 0.6, purity: "99.9%", costTier: "medium", supplierAvailability: 0.65, safetyNotes: "Toxic", preferredMethods: ["solid-state"] },
  { name: "Arsine gas", formula: "AsH3", element: "As", availability: 0.4, purity: "99.999%", costTier: "high", supplierAvailability: 0.45, safetyNotes: "Extremely toxic gas, requires specialist handling", preferredMethods: ["CVD"] },
  // Iron arsenide also serves as iron precursor for pnictide routes
  { name: "Iron arsenide (Fe-source)", formula: "FeAs", element: "Fe", availability: 0.65, purity: "99.9%", costTier: "medium", supplierAvailability: 0.7, safetyNotes: "Toxic; handle in fume hood", preferredMethods: ["solid-state", "arc-melting"] },

  // ── Organometallic precursors ────────────────────────────────────────────────
  // Metal acetylacetonates (acac = C5H7O2⁻) are the workhorses of sol-gel and
  // MOD (metal-organic decomposition) routes to oxide superconductors and thin
  // films.  Metallocenes and metal alkoxides cover CVD/ALD and MOCVD routes.

  // Lanthanum / rare-earth acac (cuprate thin-films, sol-gel La2CuO4, LaFeAsO etc.)
  { name: "Lanthanum acetylacetonate", formula: "La(acac)3", element: "La", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Combustible", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Yttrium acetylacetonate", formula: "Y(acac)3", element: "Y", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Combustible", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Cerium acetylacetonate", formula: "Ce(acac)3", element: "Ce", availability: 0.7, purity: "99.9%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Combustible", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Neodymium acetylacetonate", formula: "Nd(acac)3", element: "Nd", availability: 0.65, purity: "99.9%", costTier: "medium", supplierAvailability: 0.7, safetyNotes: "Combustible", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Samarium acetylacetonate", formula: "Sm(acac)3", element: "Sm", availability: 0.6, purity: "99.9%", costTier: "medium", supplierAvailability: 0.65, safetyNotes: "Combustible", preferredMethods: ["sol-gel"] },
  { name: "Gadolinium acetylacetonate", formula: "Gd(acac)3", element: "Gd", availability: 0.6, purity: "99.9%", costTier: "medium", supplierAvailability: 0.65, safetyNotes: "Combustible", preferredMethods: ["sol-gel"] },

  // Transition metal acac (pnictides, cuprate building blocks, electrode films)
  { name: "Iron(III) acetylacetonate", formula: "Fe(acac)3", element: "Fe", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.92, safetyNotes: "Combustible, irritant", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Cobalt(II) acetylacetonate", formula: "Co(acac)2", element: "Co", availability: 0.85, purity: "99.9%", costTier: "medium", supplierAvailability: 0.88, safetyNotes: "Combustible, sensitizer", preferredMethods: ["sol-gel"] },
  { name: "Nickel(II) acetylacetonate", formula: "Ni(acac)2", element: "Ni", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.88, safetyNotes: "Sensitizer", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Copper(II) acetylacetonate", formula: "Cu(acac)2", element: "Cu", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.92, safetyNotes: "Combustible", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Barium acetylacetonate", formula: "Ba(acac)2", element: "Ba", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Toxic if ingested", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Strontium acetylacetonate", formula: "Sr(acac)2", element: "Sr", availability: 0.7, purity: "99.9%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Combustible", preferredMethods: ["sol-gel"] },
  { name: "Calcium acetylacetonate", formula: "Ca(acac)2", element: "Ca", availability: 0.8, purity: "99.9%", costTier: "low", supplierAvailability: 0.85, safetyNotes: "Combustible", preferredMethods: ["sol-gel"] },
  { name: "Niobium ethoxide", formula: "Nb(OEt)5", element: "Nb", availability: 0.7, purity: "99.9%", costTier: "medium", supplierAvailability: 0.75, safetyNotes: "Moisture sensitive, flammable", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Vanadium acetylacetonate", formula: "V(acac)3", element: "V", availability: 0.75, purity: "99.9%", costTier: "medium", supplierAvailability: 0.8, safetyNotes: "Irritant", preferredMethods: ["sol-gel"] },
  { name: "Manganese(II) acetylacetonate", formula: "Mn(acac)2", element: "Mn", availability: 0.8, purity: "99.9%", costTier: "low", supplierAvailability: 0.85, safetyNotes: "Irritant", preferredMethods: ["sol-gel"] },
  { name: "Zinc acetylacetonate", formula: "Zn(acac)2", element: "Zn", availability: 0.85, purity: "99.9%", costTier: "low", supplierAvailability: 0.88, safetyNotes: "Combustible", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Ruthenium acetylacetonate", formula: "Ru(acac)3", element: "Ru", availability: 0.5, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.55, safetyNotes: "Expensive PGM", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Iridium acetylacetonate", formula: "Ir(acac)3", element: "Ir", availability: 0.4, purity: "99.9%", costTier: "very-high", supplierAvailability: 0.45, safetyNotes: "Very expensive", preferredMethods: ["CVD"] },

  // Metal alkoxides — sol-gel and thin-film deposition
  { name: "Titanium isopropoxide", formula: "Ti(OiPr)4", element: "Ti", availability: 0.9, purity: "99.9%", costTier: "low", supplierAvailability: 0.92, safetyNotes: "Moisture sensitive, flammable", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Zirconium propoxide", formula: "Zr(OPr)4", element: "Zr", availability: 0.8, purity: "99.9%", costTier: "medium", supplierAvailability: 0.82, safetyNotes: "Moisture sensitive, flammable", preferredMethods: ["sol-gel", "CVD"] },
  { name: "Hafnium tetrakis(diethylamide)", formula: "Hf(NEt2)4", element: "Hf", availability: 0.55, purity: "99.99%", costTier: "very-high", supplierAvailability: 0.6, safetyNotes: "Moisture sensitive, pyrophoric", preferredMethods: ["CVD"] },
  { name: "Trimethylaluminum", formula: "Al(CH3)3", element: "Al", availability: 0.8, purity: "99.999%", costTier: "medium", supplierAvailability: 0.85, safetyNotes: "Pyrophoric, extremely flammable", preferredMethods: ["CVD"] },
  { name: "Tetrakis(dimethylamido)titanium", formula: "Ti(NMe2)4", element: "Ti", availability: 0.65, purity: "99.9%", costTier: "medium", supplierAvailability: 0.7, safetyNotes: "Moisture sensitive", preferredMethods: ["CVD"] },

  // Metallocenes — CVD and MOCVD sources
  { name: "Ferrocene", formula: "Fe(C5H5)2", element: "Fe", availability: 0.85, purity: "99%", costTier: "low", supplierAvailability: 0.88, safetyNotes: "Combustible", preferredMethods: ["CVD", "vapor-deposition"] },
  { name: "Nickelocene", formula: "Ni(C5H5)2", element: "Ni", availability: 0.65, purity: "99%", costTier: "medium", supplierAvailability: 0.7, safetyNotes: "Air/moisture sensitive", preferredMethods: ["CVD"] },
  { name: "Cobaltocene", formula: "Co(C5H5)2", element: "Co", availability: 0.6, purity: "99%", costTier: "medium", supplierAvailability: 0.65, safetyNotes: "Air sensitive", preferredMethods: ["CVD"] },
  { name: "Ruthenocene", formula: "Ru(C5H5)2", element: "Ru", availability: 0.45, purity: "99%", costTier: "very-high", supplierAvailability: 0.5, safetyNotes: "Expensive PGM", preferredMethods: ["CVD"] },
];

const precursorsByElement = new Map<string, Precursor[]>();

function buildIndex() {
  if (precursorsByElement.size > 0) return;
  for (const p of PRECURSOR_DB) {
    const list = precursorsByElement.get(p.element) || [];
    list.push(p);
    precursorsByElement.set(p.element, list);
  }
}

function getDefaultPrecursor(element: string): Precursor {
  return {
    name: `${element} (elemental, generic)`,
    formula: element,
    element,
    availability: 0.3,
    purity: "99%",
    costTier: "high",
    supplierAvailability: 0.35,
    safetyNotes: "Check MSDS before handling",
    preferredMethods: ["arc-melting"],
  };
}

export function findBestPrecursors(
  elements: string[],
  method: string
): PrecursorSelection[] {
  buildIndex();

  const selections: PrecursorSelection[] = [];

  for (const el of elements) {
    const candidates = precursorsByElement.get(el) || [];

    if (candidates.length === 0) {
      const fallback = getDefaultPrecursor(el);
      selections.push({ element: el, precursor: fallback, alternates: [] });
      continue;
    }

    const scored = candidates.map((p) => {
      let score = p.availability * 0.4 + p.supplierAvailability * 0.3;
      if (p.preferredMethods.includes(method)) {
        score += 0.3;
      } else {
        score += 0.05;
      }
      const costBonus =
        p.costTier === "low"
          ? 0.15
          : p.costTier === "medium"
            ? 0.1
            : p.costTier === "high"
              ? 0.05
              : 0;
      score += costBonus;
      return { precursor: p, score };
    });

    scored.sort((a, b) => b.score - a.score);

    selections.push({
      element: el,
      precursor: scored[0].precursor,
      alternates: scored.slice(1).map((s) => s.precursor),
    });
  }

  return selections;
}

export function computePrecursorAvailabilityScore(
  selections: PrecursorSelection[]
): PrecursorAvailabilityResult {
  if (selections.length === 0) {
    return {
      overallScore: 0,
      selections,
      costEstimate: "unknown",
      bottleneckElement: null,
    };
  }

  let minAvail = 1;
  let bottleneck: string | null = null;
  let totalAvail = 0;
  let maxCostTier = 0;

  for (const sel of selections) {
    const avail = sel.precursor.availability;
    totalAvail += avail;
    if (avail < minAvail) {
      minAvail = avail;
      bottleneck = sel.element;
    }
    const tierVal =
      sel.precursor.costTier === "very-high"
        ? 4
        : sel.precursor.costTier === "high"
          ? 3
          : sel.precursor.costTier === "medium"
            ? 2
            : 1;
    if (tierVal > maxCostTier) maxCostTier = tierVal;
  }

  const avgAvail = totalAvail / selections.length;
  const overallScore = avgAvail * 0.6 + minAvail * 0.4;

  const costEstimate =
    maxCostTier >= 4
      ? "very-high"
      : maxCostTier >= 3
        ? "high"
        : maxCostTier >= 2
          ? "medium"
          : "low";

  return {
    overallScore: Number(overallScore.toFixed(4)),
    selections,
    costEstimate,
    bottleneckElement: bottleneck,
  };
}

export function getPrecursorCount(): number {
  return PRECURSOR_DB.length;
}

export function getCoveredElements(): string[] {
  buildIndex();
  return Array.from(precursorsByElement.keys()).sort();
}
