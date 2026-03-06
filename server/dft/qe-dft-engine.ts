import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { getElementData } from "../learning/elemental-data";

const PROJECT_ROOT = path.resolve(process.cwd());
const XTB_BIN = path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb");
const XTB_HOME = path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = "/tmp/dft_calculations";
const TIMEOUT_MS = 120_000;

export interface DFTResult {
  formula: string;
  method: "GFN2-xTB";
  totalEnergy: number;
  totalEnergyPerAtom: number;
  homoLumoGap: number;
  isMetallic: boolean;
  homo: number | null;
  lumo: number | null;
  fermiLevel: number | null;
  dipoleMoment: number | null;
  charges: Record<string, number>;
  wibergBondOrders: { atom1: string; atom2: string; order: number }[];
  converged: boolean;
  wallTimeSeconds: number;
  atomCount: number;
  error: string | null;
}

export interface FormationEnergyResult {
  formula: string;
  formationEnergyPerAtom: number;
  formationEnergyTotal: number;
  elementalEnergies: Record<string, number>;
  compoundEnergy: number;
  stable: boolean;
}

interface AtomPosition {
  element: string;
  x: number;
  y: number;
  z: number;
}

const ELEMENTAL_REFERENCE_ENERGIES: Record<string, number> = {};

const COVALENT_RADII: Record<string, number> = {
  H: 0.31, He: 0.28, Li: 1.28, Be: 0.96, B: 0.84, C: 0.76, N: 0.71,
  O: 0.66, F: 0.57, Ne: 0.58, Na: 1.66, Mg: 1.41, Al: 1.21, Si: 1.11,
  P: 1.07, S: 1.05, Cl: 1.02, Ar: 1.06, K: 2.03, Ca: 1.76, Sc: 1.70,
  Ti: 1.60, V: 1.53, Cr: 1.39, Mn: 1.39, Fe: 1.32, Co: 1.26, Ni: 1.24,
  Cu: 1.32, Zn: 1.22, Ga: 1.22, Ge: 1.20, As: 1.19, Se: 1.20, Br: 1.20,
  Kr: 1.16, Rb: 2.20, Sr: 1.95, Y: 1.90, Zr: 1.75, Nb: 1.64, Mo: 1.54,
  Ru: 1.46, Rh: 1.42, Pd: 1.39, Ag: 1.45, Cd: 1.44, In: 1.42, Sn: 1.39,
  Sb: 1.39, Te: 1.38, I: 1.39, Cs: 2.44, Ba: 2.15, La: 2.07, Ce: 2.04,
  Hf: 1.75, Ta: 1.70, W: 1.62, Re: 1.51, Os: 1.44, Ir: 1.41, Pt: 1.36,
  Au: 1.36, Hg: 1.32, Tl: 1.45, Pb: 1.46, Bi: 1.48,
};

function parseFormula(formula: string): Record<string, number> {
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

function generateClusterStructure(formula: string): AtomPosition[] {
  const counts = parseFormula(formula);
  const atoms: AtomPosition[] = [];
  const elements = Object.keys(counts);

  const sortedElements = [...elements].sort((a, b) => {
    const elA = getElementData(a);
    const elB = getElementData(b);
    return (elA?.atomicRadius ?? 150) - (elB?.atomicRadius ?? 150);
  });

  let currentAtoms: AtomPosition[] = [];

  const centralElement = sortedElements[sortedElements.length - 1];
  const centralCount = Math.round(counts[centralElement]);

  for (let i = 0; i < centralCount; i++) {
    const angle = (2 * Math.PI * i) / centralCount;
    const r = centralCount === 1 ? 0 : 2.5;
    currentAtoms.push({
      element: centralElement,
      x: r * Math.cos(angle),
      y: r * Math.sin(angle),
      z: 0,
    });
  }

  for (const el of sortedElements) {
    if (el === centralElement) continue;
    const count = Math.round(counts[el]);
    const bondLength = (COVALENT_RADII[centralElement] || 1.5) + (COVALENT_RADII[el] || 1.2);

    for (let i = 0; i < count; i++) {
      const parentIdx = i % currentAtoms.length;
      const parent = currentAtoms[parentIdx];

      const theta = Math.PI * (0.3 + 0.4 * (i / count));
      const phi = (2 * Math.PI * i) / count + parentIdx * 0.5;
      const r = bondLength;

      currentAtoms.push({
        element: el,
        x: parent.x + r * Math.sin(theta) * Math.cos(phi),
        y: parent.y + r * Math.sin(theta) * Math.sin(phi),
        z: parent.z + r * Math.cos(theta),
      });
    }
  }

  return currentAtoms;
}

function writeXYZ(atoms: AtomPosition[], filepath: string, comment: string = ""): void {
  const lines = [
    String(atoms.length),
    comment || "Generated structure",
    ...atoms.map(a => `${a.element}  ${a.x.toFixed(6)}  ${a.y.toFixed(6)}  ${a.z.toFixed(6)}`),
  ];
  fs.writeFileSync(filepath, lines.join("\n") + "\n");
}

function parseXtbOutput(output: string): Partial<DFTResult> {
  const result: Partial<DFTResult> = {
    converged: false,
    charges: {},
    wibergBondOrders: [],
  };

  const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
  if (energyMatch) {
    result.totalEnergy = parseFloat(energyMatch[1]);
  }

  const gapMatch = output.match(/HOMO-LUMO GAP\s+([-\d.]+)\s+eV/);
  if (gapMatch) {
    result.homoLumoGap = parseFloat(gapMatch[1]);
    result.isMetallic = result.homoLumoGap < 0.5;
  }

  const homoMatch = output.match(/\(HOMO\)\s+([-\d.]+)\s+eV/);
  const lumoMatch = output.match(/\(LUMO\)\s+([-\d.]+)\s+eV/);
  if (homoMatch) result.homo = parseFloat(homoMatch[1]);
  if (lumoMatch) result.lumo = parseFloat(lumoMatch[1]);

  if (result.homo != null && result.lumo != null) {
    result.fermiLevel = (result.homo + result.lumo) / 2;
  }

  const dipoleMatch = output.match(/tot \(Debye\)\s*\n\s*full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)/);
  if (dipoleMatch) {
    result.dipoleMoment = parseFloat(dipoleMatch[1]);
  } else {
    const dipoleAlt = output.match(/molecular dipole:.*?tot \(Debye\).*?full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)/s);
    if (dipoleAlt) result.dipoleMoment = parseFloat(dipoleAlt[1]);
  }

  const chargeSection = output.match(/Mulliken Charges.*?\n([\s\S]*?)(?:\n\n|\nWiberg)/);
  if (chargeSection) {
    const lines = chargeSection[1].trim().split("\n");
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 3) {
        const el = parts[1];
        const charge = parseFloat(parts[parts.length - 1]);
        if (!isNaN(charge)) {
          result.charges![el] = (result.charges![el] || 0) + charge;
        }
      }
    }
  }

  const wibergSection = output.match(/Wiberg\/Mayer.*?bond orders.*?\n([\s\S]*?)(?:\n\n|molecular)/);
  if (wibergSection) {
    const lines = wibergSection[1].trim().split("\n");
    for (const line of lines) {
      const bondMatch = line.match(/(\w+)\s*--\s*(\w+)\s*:\s*([\d.]+)/);
      if (bondMatch) {
        result.wibergBondOrders!.push({
          atom1: bondMatch[1],
          atom2: bondMatch[2],
          order: parseFloat(bondMatch[3]),
        });
      }
    }
  }

  const wallMatch = output.match(/wall-time:\s+\d+ d,\s+\d+ h,\s+\d+ min,\s+([\d.]+) sec/);
  if (wallMatch) {
    result.wallTimeSeconds = parseFloat(wallMatch[1]);
  }

  if (output.includes("normal termination of xtb")) {
    result.converged = true;
  }

  return result;
}

async function computeElementalEnergy(element: string): Promise<number> {
  if (ELEMENTAL_REFERENCE_ENERGIES[element] !== undefined) {
    return ELEMENTAL_REFERENCE_ENERGIES[element];
  }

  const calcDir = path.join(WORK_DIR, `ref_${element}`);
  fs.mkdirSync(calcDir, { recursive: true });

  const atoms: AtomPosition[] = [{ element, x: 0, y: 0, z: 0 }];

  const elData = getElementData(element);
  if (elData && elData.valenceElectrons > 1) {
    const r = COVALENT_RADII[element] || 1.5;
    atoms.push({ element, x: r * 2, y: 0, z: 0 });
  }

  const xyzPath = path.join(calcDir, `${element}.xyz`);
  writeXYZ(atoms, xyzPath, `${element} reference`);

  try {
    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    const output = execSync(
      `cd ${calcDir} && ${XTB_BIN} ${xyzPath} --gfn 2 --sp --uhf 0 2>&1`,
      { timeout: 60000, env, maxBuffer: 10 * 1024 * 1024 }
    ).toString();

    const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
    if (energyMatch) {
      const energyPerAtom = parseFloat(energyMatch[1]) / atoms.length;
      ELEMENTAL_REFERENCE_ENERGIES[element] = energyPerAtom;
      return energyPerAtom;
    }
  } catch (err) {
    console.log(`[DFT-xTB] Failed to compute reference energy for ${element}`);
  }

  ELEMENTAL_REFERENCE_ENERGIES[element] = NaN;
  return NaN;
}

export async function runDFTCalculation(formula: string): Promise<DFTResult> {
  const startTime = Date.now();
  const calcId = `${formula.replace(/[^a-zA-Z0-9]/g, "_")}_${Date.now()}`;
  const calcDir = path.join(WORK_DIR, calcId);
  fs.mkdirSync(calcDir, { recursive: true });

  const atoms = generateClusterStructure(formula);
  const xyzPath = path.join(calcDir, "input.xyz");
  writeXYZ(atoms, xyzPath, formula);

  const result: DFTResult = {
    formula,
    method: "GFN2-xTB",
    totalEnergy: 0,
    totalEnergyPerAtom: 0,
    homoLumoGap: 0,
    isMetallic: false,
    homo: null,
    lumo: null,
    fermiLevel: null,
    dipoleMoment: null,
    charges: {},
    wibergBondOrders: [],
    converged: false,
    wallTimeSeconds: 0,
    atomCount: atoms.length,
    error: null,
  };

  try {
    const env = {
      ...process.env,
      XTBHOME: XTB_HOME,
      XTBPATH: XTB_PARAM,
      OMP_NUM_THREADS: "1",
      OMP_STACKSIZE: "512M",
    };

    const output = execSync(
      `cd ${calcDir} && ${XTB_BIN} ${xyzPath} --gfn 2 --sp 2>&1`,
      { timeout: TIMEOUT_MS, env, maxBuffer: 10 * 1024 * 1024 }
    ).toString();

    const parsed = parseXtbOutput(output);
    Object.assign(result, parsed);

    if (result.totalEnergy !== 0 && atoms.length > 0) {
      result.totalEnergyPerAtom = result.totalEnergy / atoms.length;
    }

    fs.writeFileSync(path.join(calcDir, "output.log"), output);
  } catch (err: any) {
    result.error = err.message?.slice(0, 200) || "DFT calculation failed";
    if (err.stdout) {
      const stdoutStr = err.stdout.toString();
      const parsed = parseXtbOutput(stdoutStr);
      if (parsed.totalEnergy) {
        Object.assign(result, parsed);
        result.converged = false;
        result.error = "Partial convergence: " + result.error;
      }
    }
  }

  result.wallTimeSeconds = (Date.now() - startTime) / 1000;

  try {
    fs.rmSync(calcDir, { recursive: true, force: true });
  } catch {}

  return result;
}

export async function computeFormationEnergy(formula: string, dftResult?: DFTResult): Promise<FormationEnergyResult> {
  const counts = parseFormula(formula);
  const totalAtoms = Object.values(counts).reduce((s, n) => s + Math.round(n), 0);

  const compoundResult = dftResult || await runDFTCalculation(formula);
  const compoundEnergy = compoundResult.totalEnergy;

  const elementalEnergies: Record<string, number> = {};
  let elementalTotal = 0;
  let hasInvalidReference = false;

  for (const [el, count] of Object.entries(counts)) {
    const n = Math.round(count);
    const refEnergy = await computeElementalEnergy(el);
    elementalEnergies[el] = refEnergy;
    if (isNaN(refEnergy)) {
      hasInvalidReference = true;
      console.log(`[DFT-xTB] Invalid reference energy for ${el}, formation energy unreliable`);
    } else {
      elementalTotal += refEnergy * n;
    }
  }

  if (hasInvalidReference) {
    return {
      formula,
      formationEnergyPerAtom: NaN,
      formationEnergyTotal: NaN,
      elementalEnergies,
      compoundEnergy,
      stable: false,
    };
  }

  const formationEnergyTotal = compoundEnergy - elementalTotal;
  const formationEnergyPerAtom = totalAtoms > 0 ? formationEnergyTotal / totalAtoms : 0;

  const HA_TO_EV = 27.2114;

  return {
    formula,
    formationEnergyPerAtom: formationEnergyPerAtom * HA_TO_EV,
    formationEnergyTotal: formationEnergyTotal * HA_TO_EV,
    elementalEnergies,
    compoundEnergy,
    stable: formationEnergyPerAtom < 0,
  };
}

export function isDFTAvailable(): boolean {
  try {
    return fs.existsSync(XTB_BIN);
  } catch {
    return false;
  }
}

export function getDFTMethodInfo(): { name: string; version: string; level: string } {
  return {
    name: "GFN2-xTB",
    version: "6.7.1",
    level: "Semi-empirical DFT (Density Functional Tight Binding)",
  };
}
