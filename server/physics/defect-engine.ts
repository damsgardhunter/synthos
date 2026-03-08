import { ELEMENTAL_DATA } from "../learning/elemental-data";

export enum DefectType {
  Vacancy = "vacancy",
  Interstitial = "interstitial",
  Antisite = "antisite",
  Dopant = "dopant",
}

export interface DefectStructure {
  type: DefectType;
  site: string;
  element: string;
  concentration: number;
  formationEnergy: number;
  defectDensity: number;
  mutatedFormula: string;
}

const kB = 8.617e-5; // eV/K
const N_SITES = 1e22; // approximate atomic sites per cm³

const INTERSTITIAL_ELEMENTS = ["H", "O", "N", "Li"];

const NEIGHBOR_MAP: Record<string, string[]> = {
  Ti: ["V", "Zr", "Sc"],
  V: ["Ti", "Cr", "Nb"],
  Cr: ["V", "Mn", "Mo"],
  Mn: ["Cr", "Fe", "Tc"],
  Fe: ["Mn", "Co", "Ru"],
  Co: ["Fe", "Ni", "Rh"],
  Ni: ["Co", "Cu", "Pd"],
  Cu: ["Ni", "Zn", "Ag"],
  Zn: ["Cu", "Ga", "Cd"],
  Zr: ["Ti", "Nb", "Hf"],
  Nb: ["V", "Mo", "Zr"],
  Mo: ["Cr", "Nb", "W"],
  Ru: ["Fe", "Rh", "Os"],
  Rh: ["Co", "Pd", "Ir"],
  Pd: ["Ni", "Rh", "Pt"],
  Ag: ["Cu", "Pd", "Au"],
  Hf: ["Zr", "Ta", "Lu"],
  Ta: ["Nb", "W", "Hf"],
  W: ["Mo", "Ta", "Re"],
  Re: ["W", "Os", "Tc"],
  Os: ["Ru", "Ir", "Re"],
  Ir: ["Rh", "Pt", "Os"],
  Pt: ["Pd", "Ir", "Au"],
  Au: ["Ag", "Pt", "Hg"],
  Y: ["Sc", "Zr", "La"],
  La: ["Y", "Ba", "Ce"],
  Ba: ["La", "Sr", "Cs"],
  Sr: ["Ba", "Ca", "Rb"],
  Ca: ["Sr", "K", "Sc"],
  Bi: ["Pb", "Sb", "Te"],
  Pb: ["Bi", "Tl", "Sn"],
  Sn: ["Pb", "In", "Sb"],
  In: ["Sn", "Ga", "Tl"],
  Ga: ["In", "Zn", "Al"],
  Al: ["Ga", "Mg", "Si"],
  Mg: ["Al", "Ca", "Be"],
  Si: ["Al", "Ge", "P"],
  Ge: ["Si", "Sn", "Ga"],
  B: ["C", "Al", "Be"],
  C: ["B", "N", "Si"],
  N: ["C", "O", "P"],
  O: ["N", "S", "F"],
  S: ["O", "Se", "P"],
  Se: ["S", "Te", "Br"],
  Te: ["Se", "Bi", "I"],
  H: ["Li", "He", "B"],
  Li: ["H", "Na", "Be"],
  Na: ["Li", "K", "Mg"],
  K: ["Na", "Ca", "Rb"],
  Rb: ["K", "Sr", "Cs"],
  Cs: ["Rb", "Ba", "Fr"],
  Sc: ["Ti", "Y", "Ca"],
  Ce: ["La", "Pr", "Nd"],
  Pr: ["Ce", "Nd", "La"],
  Nd: ["Pr", "Pm", "Ce"],
  Sm: ["Nd", "Eu", "Pm"],
  Eu: ["Sm", "Gd", "Ba"],
  Gd: ["Eu", "Tb", "Sm"],
  Tb: ["Gd", "Dy", "Eu"],
  Dy: ["Tb", "Ho", "Gd"],
  Ho: ["Dy", "Er", "Tb"],
  Er: ["Ho", "Tm", "Dy"],
  Tm: ["Er", "Yb", "Ho"],
  Yb: ["Tm", "Lu", "Er"],
  Lu: ["Yb", "Hf", "Tm"],
  As: ["Se", "Ge", "P"],
  P: ["N", "As", "Si"],
  Sb: ["Bi", "Sn", "As"],
  Tl: ["Pb", "In", "Hg"],
  Hg: ["Au", "Tl", "Cd"],
  Cd: ["Zn", "Hg", "In"],
  Tc: ["Mo", "Ru", "Mn"],
  F: ["O", "Ne", "Cl"],
  Cl: ["F", "S", "Br"],
  Br: ["Cl", "Se", "I"],
  I: ["Br", "Te", "At"],
};

const defectStats = {
  totalDefectsGenerated: 0,
  typeBreakdown: {
    [DefectType.Vacancy]: 0,
    [DefectType.Interstitial]: 0,
    [DefectType.Antisite]: 0,
    [DefectType.Dopant]: 0,
  } as Record<string, number>,
  bestTcImprovement: 0,
  bestTcFormula: "",
  totalFormulasProcessed: 0,
};

function parseFormula(formula: string): Record<string, number> {
  const elements: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(formula)) !== null) {
    const el = match[1];
    const count = match[2] ? parseFloat(match[2]) : 1;
    elements[el] = (elements[el] || 0) + count;
  }
  return elements;
}

function formulaFromComposition(comp: Record<string, number>): string {
  return Object.entries(comp)
    .filter(([, v]) => v > 0.001)
    .map(([el, count]) => {
      if (Math.abs(count - 1) < 0.001) return el;
      if (Math.abs(count - Math.round(count)) < 0.01) return `${el}${Math.round(count)}`;
      return `${el}${count.toFixed(2)}`;
    })
    .join("");
}

function getElementEnergy(el: string): number {
  const data = ELEMENTAL_DATA[el];
  if (!data) return 3.0;
  const ie = data.firstIonizationEnergy || 7;
  const ea = data.electronAffinity || 0;
  return (ie - ea) * 0.5;
}

function getCoordinationFactor(el: string): number {
  const data = ELEMENTAL_DATA[el];
  if (!data) return 1.0;
  const ve = data.valenceElectrons;
  if (ve <= 2) return 0.8;
  if (ve <= 4) return 1.0;
  if (ve <= 6) return 1.1;
  return 1.2;
}

export function computeDefectFormationEnergy(
  formula: string,
  defectType: DefectType,
  element: string
): number {
  const comp = parseFormula(formula);
  const elements = Object.keys(comp);

  const hostEnergy = elements.reduce((sum, el) => {
    return sum + getElementEnergy(el) * (comp[el] || 0);
  }, 0) / Math.max(1, elements.length);

  const elEnergy = getElementEnergy(element);
  const coordFactor = getCoordinationFactor(element);

  let Ef: number;
  switch (defectType) {
    case DefectType.Vacancy:
      Ef = elEnergy * coordFactor * 0.6 + hostEnergy * 0.1;
      break;
    case DefectType.Interstitial:
      Ef = elEnergy * 0.3 + hostEnergy * coordFactor * 0.4;
      break;
    case DefectType.Antisite: {
      const otherEl = elements.find(e => e !== element) || element;
      const otherEnergy = getElementEnergy(otherEl);
      Ef = Math.abs(elEnergy - otherEnergy) * coordFactor * 0.8 + 0.5;
      break;
    }
    case DefectType.Dopant: {
      const neighbors = NEIGHBOR_MAP[element] || [];
      const dopant = neighbors[0] || element;
      const dopantEnergy = getElementEnergy(dopant);
      Ef = Math.abs(elEnergy - dopantEnergy) * 0.5 + hostEnergy * 0.15 + 0.3;
      break;
    }
    default:
      Ef = 2.0;
  }

  return Math.max(0.1, Math.min(8.0, Ef));
}

export function estimateDefectDensity(
  formationEnergy: number,
  temperature: number = 300
): number {
  const T = Math.max(1, temperature);
  const exponent = -formationEnergy / (kB * T);
  const density = N_SITES * Math.exp(exponent);
  return Math.max(1e6, Math.min(1e22, density));
}

export function generateDefectVariants(formula: string): DefectStructure[] {
  const comp = parseFormula(formula);
  const elements = Object.keys(comp);
  const variants: DefectStructure[] = [];

  defectStats.totalFormulasProcessed++;

  for (const el of elements) {
    if ((comp[el] || 0) < 0.01) continue;
    const Ef = computeDefectFormationEnergy(formula, DefectType.Vacancy, el);
    const density = estimateDefectDensity(Ef, 300);
    const vacConc = Math.min(0.10, Math.max(0.005, 0.05 * Math.exp(-Math.max(0, Ef - 1.0) / 0.5)));
    const newComp = { ...comp, [el]: comp[el] * (1 - vacConc) };

    variants.push({
      type: DefectType.Vacancy,
      site: el,
      element: el,
      concentration: vacConc,
      formationEnergy: Ef,
      defectDensity: density,
      mutatedFormula: formulaFromComposition(newComp),
    });
    defectStats.totalDefectsGenerated++;
    defectStats.typeBreakdown[DefectType.Vacancy]++;
  }

  for (const intEl of INTERSTITIAL_ELEMENTS) {
    if (elements.includes(intEl) && comp[intEl] > 1) continue;
    const Ef = computeDefectFormationEnergy(formula, DefectType.Interstitial, intEl);
    const density = estimateDefectDensity(Ef, 300);
    const intConc = Math.min(0.05, Math.max(0.002, 0.02 * Math.exp(-Math.max(0, Ef - 1.5) / 0.5)));

    const intComp = { ...comp, [intEl]: (comp[intEl] || 0) + intConc };
    variants.push({
      type: DefectType.Interstitial,
      site: "interstitial",
      element: intEl,
      concentration: intConc,
      formationEnergy: Ef,
      defectDensity: density,
      mutatedFormula: formulaFromComposition(intComp),
    });
    defectStats.totalDefectsGenerated++;
    defectStats.typeBreakdown[DefectType.Interstitial]++;
  }

  if (elements.length >= 2) {
    for (let i = 0; i < elements.length; i++) {
      for (let j = i + 1; j < elements.length; j++) {
        const elA = elements[i];
        const elB = elements[j];
        const Ef = computeDefectFormationEnergy(formula, DefectType.Antisite, elA);
        const density = estimateDefectDensity(Ef, 300);

        const asComp = { ...comp };
        const swapFrac = Math.min(0.08, Math.max(0.005, 0.03 * Math.exp(-Math.max(0, Ef - 1.0) / 0.5)));
        const originalA = asComp[elA] || 1;
        asComp[elA] = (asComp[elA] || 0) - swapFrac * originalA;
        asComp[elB] = (asComp[elB] || 0) + swapFrac * originalA;
        variants.push({
          type: DefectType.Antisite,
          site: `${elA}<->${elB}`,
          element: `${elA}/${elB}`,
          concentration: swapFrac,
          formationEnergy: Ef,
          defectDensity: density,
          mutatedFormula: formulaFromComposition(asComp),
        });
        defectStats.totalDefectsGenerated++;
        defectStats.typeBreakdown[DefectType.Antisite]++;
      }
    }
  }

  for (const el of elements) {
    const neighbors = NEIGHBOR_MAP[el];
    if (!neighbors || neighbors.length === 0) continue;
    const dopant = neighbors[0];
    if (elements.includes(dopant)) continue;
    const Ef = computeDefectFormationEnergy(formula, DefectType.Dopant, el);
    const density = estimateDefectDensity(Ef, 300);

    const dopFrac = Math.min(0.10, Math.max(0.005, 0.05 * Math.exp(-Math.max(0, Ef - 1.0) / 0.5)));
    const dopComp = { ...comp };
    dopComp[el] = (dopComp[el] || 1) * (1 - dopFrac);
    dopComp[dopant] = (dopComp[dopant] || 0) + (comp[el] || 1) * dopFrac;
    variants.push({
      type: DefectType.Dopant,
      site: el,
      element: dopant,
      concentration: dopFrac,
      formationEnergy: Ef,
      defectDensity: density,
      mutatedFormula: formulaFromComposition(dopComp),
    });
    defectStats.totalDefectsGenerated++;
    defectStats.typeBreakdown[DefectType.Dopant]++;
  }

  return variants;
}

export interface ElectronicAdjustment {
  adjustedDosAtEF: number;
  adjustedLambda: number;
  scatteringRate: number;
  tcModifier: number;
  notes: string;
}

export function adjustElectronicStructure(
  dosAtEF: number,
  lambda: number,
  defectDensity: number,
  defectType: DefectType,
  formula?: string
): ElectronicAdjustment {
  let dosModifier = 1.0;
  let lambdaModifier = 1.0;
  let scatteringRate = 0;
  const logDensity = Math.log10(Math.max(1, defectDensity));
  const normalizedDensity = Math.min(1, (logDensity - 14) / 8);

  switch (defectType) {
    case DefectType.Vacancy:
      dosModifier = 1.0 + normalizedDensity * 0.15;
      lambdaModifier = 1.0 - normalizedDensity * 0.08;
      scatteringRate = normalizedDensity * 0.2;
      break;
    case DefectType.Interstitial:
      dosModifier = 1.0 + normalizedDensity * 0.25;
      lambdaModifier = 1.0 + normalizedDensity * 0.12;
      scatteringRate = normalizedDensity * 0.1;
      break;
    case DefectType.Antisite:
      dosModifier = 1.0 + normalizedDensity * 0.05;
      lambdaModifier = 1.0 - normalizedDensity * 0.15;
      scatteringRate = normalizedDensity * 0.3;
      break;
    case DefectType.Dopant:
      dosModifier = 1.0 + normalizedDensity * 0.20;
      lambdaModifier = 1.0 + normalizedDensity * 0.08;
      scatteringRate = normalizedDensity * 0.05;
      break;
  }

  const adjustedDos = dosAtEF * dosModifier;
  const adjustedLambda = lambda * lambdaModifier;

  const pairBreaking = 1.0 - scatteringRate * 0.5;
  const dosBoost = dosModifier > 1.0 ? (dosModifier - 1.0) * 0.3 : 0;
  const lambdaEffect = lambdaModifier - 1.0;
  const tcModifier = 1.0 + dosBoost + lambdaEffect * 0.5 - scatteringRate * 0.2;

  const tcMod = Math.max(0.5, Math.min(1.5, tcModifier));

  const improvement = tcMod - 1.0;
  if (improvement > 0 && improvement > defectStats.bestTcImprovement) {
    defectStats.bestTcImprovement = improvement;
    if (formula) defectStats.bestTcFormula = formula;
  }

  let notes = "";
  if (defectType === DefectType.Vacancy) {
    notes = "Vacancies increase DOS via resonant states but reduce coupling through broken bonds";
  } else if (defectType === DefectType.Interstitial) {
    notes = "Interstitials can enhance both DOS and electron-phonon coupling through new bonding channels";
  } else if (defectType === DefectType.Antisite) {
    notes = "Antisite defects primarily act as pair-breaking scattering centers";
  } else {
    notes = "Dopants tune carrier concentration and can enhance pairing via optimized Fermi surface";
  }

  return {
    adjustedDosAtEF: adjustedDos,
    adjustedLambda: adjustedLambda,
    scatteringRate,
    tcModifier: tcMod,
    notes,
  };
}

export function getDefectEngineStats() {
  return {
    totalDefectsGenerated: defectStats.totalDefectsGenerated,
    typeBreakdown: { ...defectStats.typeBreakdown },
    bestTcImprovement: defectStats.bestTcImprovement,
    bestTcFormula: defectStats.bestTcFormula,
    totalFormulasProcessed: defectStats.totalFormulasProcessed,
  };
}
