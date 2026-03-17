import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { binaryPath, getTempSubdir, IS_WINDOWS, toWslPath } from "../dft/platform-utils";
import { getElementData } from "../learning/elemental-data";
import type {
  HeterostructureResult,
  HeterostructureAtom,
  StackedStructure,
  BilayerStructure,
} from "./heterostructure-generator";
import {
  generateHeterostructure,
  generateBilayerCandidates,
  findBestSubstrates,
  getHeterostructureStats as getGenStats,
} from "./heterostructure-generator";

const PROJECT_ROOT = path.resolve(process.cwd());
// XTB_BIN: use env override (set XTB_BIN=/usr/bin/xtb on GCP where xtb-dist isn't deployed)
const XTB_BIN = binaryPath(process.env.XTB_BIN ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist/bin/xtb"));
const XTB_HOME = process.env.XTBHOME ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist");
const XTB_PARAM = process.env.XTBPATH ?? path.join(PROJECT_ROOT, "server/dft/xtb-dist/share/xtb");
const WORK_DIR = getTempSubdir("interface_relaxations");
const OPT_TIMEOUT_MS = 45_000;

// Cross-platform shell executor: routes through WSL2 on Windows so Linux xTB binary works.
function execXtbCmd(cmd: string, opts: { timeout: number; env: Record<string, string>; maxBuffer: number }): string {
  if (IS_WINDOWS) {
    // Native Windows binary (.exe) — run directly. Routing through WSL would translate
    // all paths to /mnt/c/... which a PE-format Windows xtb.exe cannot resolve.
    if (XTB_BIN.endsWith(".exe")) {
      return execSync(cmd, { timeout: opts.timeout, maxBuffer: opts.maxBuffer, env: opts.env }).toString();
    }
    // Linux ELF binary (e.g. from xtb-dist/) — route through WSL, translating paths
    const wslCmd = cmd.replace(/[A-Za-z]:\\[^ "&'|<>]*/g, (m) => toWslPath(m));
    const wslHome = opts.env.XTBHOME ? toWslPath(opts.env.XTBHOME) : "";
    const wslPath = opts.env.XTBPATH ? toWslPath(opts.env.XTBPATH) : "";
    const envPrefix = [
      wslHome ? `XTBHOME='${wslHome}'` : "",
      wslPath ? `XTBPATH='${wslPath}'` : "",
      opts.env.OMP_NUM_THREADS ? `OMP_NUM_THREADS=${opts.env.OMP_NUM_THREADS}` : "",
      opts.env.OMP_STACKSIZE ? `OMP_STACKSIZE=${opts.env.OMP_STACKSIZE}` : "",
    ].filter(Boolean).join(" ");
    const fullCmd = envPrefix ? `${envPrefix} ${wslCmd}` : wslCmd;
    return execSync(`wsl.exe -d Ubuntu -- bash -c "${fullCmd.replace(/"/g, '\\"')}"`, {
      timeout: opts.timeout,
      maxBuffer: opts.maxBuffer,
    }).toString();
  }
  return execSync(cmd, opts).toString();
}

export interface AtomCharge {
  element: string;
  index: number;
  charge: number;
  layer: number;
}

export interface InterfaceChargeTransfer {
  layer1AvgCharge: number;
  layer2AvgCharge: number;
  chargePerAtom: number;
  totalTransferred: number;
  direction: string;
  isSignificant: boolean;
  perAtomCharges: AtomCharge[];
}

export interface InterfaceStrain {
  initialInterlayerDistance: number;
  relaxedInterlayerDistance: number;
  strainPercent: number;
  isOptimalRange: boolean;
  volumeChange: number;
  maxAtomDisplacement: number;
  avgAtomDisplacement: number;
}

export interface InterfacePhononCoupling {
  massRatioFilmSub: number;
  isHeavyLightPair: boolean;
  substrateIsOxide: boolean;
  filmIsLight: boolean;
  couplingProxy: number;
  debyeTempRatio: number;
  phononOverlapScore: number;
}

export interface ElectronDensityOverlap {
  filmEN: number;
  subEN: number;
  enDifference: number;
  workFunctionDiff: number;
  bandAlignmentType: "type-I" | "type-II" | "type-III";
  overlapScore: number;
}

export interface InterfacePhysicsResult {
  film: string;
  substrate: string;
  chargeTransfer: InterfaceChargeTransfer;
  strain: InterfaceStrain;
  phononCoupling: InterfacePhononCoupling;
  electronDensity: ElectronDensityOverlap;
  compositeScore: number;
  scoreBreakdown: {
    latticeMatchScore: number;
    chargeTransferScore: number;
    strainScore: number;
    phononCouplingScore: number;
    electronDensityScore: number;
  };
  xtbConverged: boolean;
  xtbEnergy: number | null;
  wallTimeMs: number;
  relaxedAtoms: Array<{ element: string; x: number; y: number; z: number }> | null;
}

export interface InterfaceCandidate {
  film: string;
  substrate: string;
  heterostructure: HeterostructureResult;
  physics: InterfacePhysicsResult | null;
  rank: number;
  selectedForDFT: boolean;
}

export interface InterfaceDiscoveryStats {
  totalRelaxations: number;
  xtbSuccesses: number;
  xtbFailures: number;
  avgCompositeScore: number;
  avgChargeTransfer: number;
  avgStrain: number;
  significantChargeTransferCount: number;
  optimalStrainCount: number;
  topInterfaces: Array<{
    film: string;
    substrate: string;
    compositeScore: number;
    chargePerAtom: number;
    strainPct: number;
    phononCoupling: number;
  }>;
  recentRelaxations: Array<{
    film: string;
    substrate: string;
    compositeScore: number;
    xtbConverged: boolean;
    wallTimeMs: number;
  }>;
  activeLearningSelections: number;
}

let _xtbAvailableCache: boolean | null = null;

function isXtbAvailable(): boolean {
  if (_xtbAvailableCache !== null) return _xtbAvailableCache;
  try {
    if (!fs.existsSync(XTB_BIN)) {
      _xtbAvailableCache = false;
      return false;
    }
    if (IS_WINDOWS && XTB_BIN.endsWith(".exe")) {
      // Native Windows binary — test directly without WSL
      execSync(`"${XTB_BIN}" --version`, { timeout: 8000, stdio: "pipe" });
    } else if (IS_WINDOWS) {
      // Linux ELF binary — verify it runs inside WSL
      const wslBin = toWslPath(XTB_BIN);
      execSync(`wsl.exe -d Ubuntu -- bash -c "test -x '${wslBin}' && '${wslBin}' --version"`, {
        timeout: 8000,
        stdio: "pipe",
      });
    } else {
      execSync(`"${XTB_BIN}" --version`, { timeout: 8000, stdio: "pipe" });
    }
    _xtbAvailableCache = true;
    return true;
  } catch {
    _xtbAvailableCache = false;
    return false;
  }
}

function writeInterfaceXYZ(atoms: HeterostructureAtom[], filepath: string, comment: string): void {
  const lines = [
    String(atoms.length),
    comment,
    ...atoms.map(a => `${a.element}  ${a.x.toFixed(6)}  ${a.y.toFixed(6)}  ${a.z.toFixed(6)}`),
  ];
  fs.writeFileSync(filepath, lines.join("\n") + "\n");
}

function parseXtbCharges(output: string, atoms: HeterostructureAtom[]): AtomCharge[] {
  const charges: AtomCharge[] = [];

  const chargeBlock = output.match(/#\s+Z\s+covCN\s+q\s+C6AA\s+.*?\n([\s\S]*?)(?:\n\n|\nWiberg|\nmolecular)/);
  if (chargeBlock) {
    const lines = chargeBlock[1].trim().split("\n");
    for (let i = 0; i < lines.length && i < atoms.length; i++) {
      const parts = lines[i].trim().split(/\s+/);
      if (parts.length >= 4) {
        const charge = parseFloat(parts[3]);
        if (!isNaN(charge)) {
          charges.push({
            element: atoms[i].element,
            index: i,
            charge,
            layer: atoms[i].layer,
          });
        }
      }
    }
  }

  if (charges.length === 0) {
    for (let i = 0; i < atoms.length; i++) {
      charges.push({
        element: atoms[i].element,
        index: i,
        charge: 0,
        layer: atoms[i].layer,
      });
    }
  }

  return charges;
}

function parseOptimizedAtoms(calcDir: string, fallbackAtoms: HeterostructureAtom[]): Array<{ element: string; x: number; y: number; z: number }> | null {
  const optPath = path.join(calcDir, "xtbopt.xyz");
  if (!fs.existsSync(optPath)) return null;

  try {
    const content = fs.readFileSync(optPath, "utf-8").trim();
    const lines = content.split("\n");
    if (lines.length < 3) return null;

    const atomCount = parseInt(lines[0].trim(), 10);
    if (isNaN(atomCount) || atomCount < 2) return null;

    const atoms: Array<{ element: string; x: number; y: number; z: number }> = [];
    for (let i = 2; i < Math.min(lines.length, atomCount + 2); i++) {
      const parts = lines[i].trim().split(/\s+/);
      if (parts.length >= 4) {
        atoms.push({
          element: parts[0],
          x: parseFloat(parts[1]),
          y: parseFloat(parts[2]),
          z: parseFloat(parts[3]),
        });
      }
    }

    return atoms.length >= 2 ? atoms : null;
  } catch {
    return null;
  }
}

function computeChargeTransfer(charges: AtomCharge[], bilayer: BilayerStructure): InterfaceChargeTransfer {
  const layer1Charges = charges.filter(c => c.layer % 2 === 0);
  const layer2Charges = charges.filter(c => c.layer % 2 === 1);

  const layer1Avg = layer1Charges.length > 0
    ? layer1Charges.reduce((s, c) => s + c.charge, 0) / layer1Charges.length
    : 0;
  const layer2Avg = layer2Charges.length > 0
    ? layer2Charges.reduce((s, c) => s + c.charge, 0) / layer2Charges.length
    : 0;

  const totalTransferred = Math.abs(
    layer1Charges.reduce((s, c) => s + c.charge, 0)
  );
  const chargePerAtom = charges.length > 0 ? totalTransferred / charges.length : 0;

  let direction: string;
  if (layer1Avg > layer2Avg + 0.01) {
    direction = `${bilayer.layer2} -> ${bilayer.layer1} (electrons)`;
  } else if (layer2Avg > layer1Avg + 0.01) {
    direction = `${bilayer.layer1} -> ${bilayer.layer2} (electrons)`;
  } else {
    direction = "minimal transfer";
  }

  const isSignificant = chargePerAtom > 0.05;

  return {
    layer1AvgCharge: Number(layer1Avg.toFixed(5)),
    layer2AvgCharge: Number(layer2Avg.toFixed(5)),
    chargePerAtom: Number(chargePerAtom.toFixed(5)),
    totalTransferred: Number(totalTransferred.toFixed(5)),
    direction,
    isSignificant,
    perAtomCharges: charges,
  };
}

function computeInterfaceStrain(
  initialAtoms: HeterostructureAtom[],
  relaxedAtoms: Array<{ element: string; x: number; y: number; z: number }> | null,
  bilayer: BilayerStructure
): InterfaceStrain {
  const initialDistance = bilayer.interlayerDistance;

  if (!relaxedAtoms || relaxedAtoms.length !== initialAtoms.length) {
    return {
      initialInterlayerDistance: initialDistance,
      relaxedInterlayerDistance: initialDistance,
      strainPercent: Math.abs(bilayer.strain) * 100,
      isOptimalRange: Math.abs(bilayer.strain) >= 0.01 && Math.abs(bilayer.strain) <= 0.04,
      volumeChange: 0,
      maxAtomDisplacement: 0,
      avgAtomDisplacement: 0,
    };
  }

  let maxDisp = 0;
  let totalDisp = 0;
  for (let i = 0; i < initialAtoms.length && i < relaxedAtoms.length; i++) {
    const dx = relaxedAtoms[i].x - initialAtoms[i].x;
    const dy = relaxedAtoms[i].y - initialAtoms[i].y;
    const dz = relaxedAtoms[i].z - initialAtoms[i].z;
    const disp = Math.sqrt(dx * dx + dy * dy + dz * dz);
    totalDisp += disp;
    if (disp > maxDisp) maxDisp = disp;
  }
  const avgDisp = totalDisp / initialAtoms.length;

  const layer0InitZ = initialAtoms.filter(a => a.layer === 0).map(a => a.z);
  const layer1InitZ = initialAtoms.filter(a => a.layer === 1).map(a => a.z);
  const layer0RelZ = relaxedAtoms.filter((_, i) => i < initialAtoms.length && initialAtoms[i].layer === 0).map(a => a.z);
  const layer1RelZ = relaxedAtoms.filter((_, i) => i < initialAtoms.length && initialAtoms[i].layer === 1).map(a => a.z);

  const initCenter0 = layer0InitZ.length > 0 ? layer0InitZ.reduce((s, z) => s + z, 0) / layer0InitZ.length : 0;
  const initCenter1 = layer1InitZ.length > 0 ? layer1InitZ.reduce((s, z) => s + z, 0) / layer1InitZ.length : 0;
  const relCenter0 = layer0RelZ.length > 0 ? layer0RelZ.reduce((s, z) => s + z, 0) / layer0RelZ.length : 0;
  const relCenter1 = layer1RelZ.length > 0 ? layer1RelZ.reduce((s, z) => s + z, 0) / layer1RelZ.length : 0;

  const relaxedDistance = Math.abs(relCenter1 - relCenter0);
  const effectiveStrain = initialDistance > 0
    ? Math.abs(relaxedDistance - initialDistance) / initialDistance
    : Math.abs(bilayer.strain);

  const strainPct = effectiveStrain * 100;

  const initExtent = computeExtent(initialAtoms);
  const relExtent = computeExtent(relaxedAtoms as any);
  const volumeChange = initExtent > 0 ? (relExtent - initExtent) / initExtent : 0;

  return {
    initialInterlayerDistance: Number(initialDistance.toFixed(4)),
    relaxedInterlayerDistance: Number(relaxedDistance.toFixed(4)),
    strainPercent: Number(strainPct.toFixed(3)),
    isOptimalRange: strainPct >= 1.0 && strainPct <= 4.0,
    volumeChange: Number(volumeChange.toFixed(5)),
    maxAtomDisplacement: Number(maxDisp.toFixed(4)),
    avgAtomDisplacement: Number(avgDisp.toFixed(4)),
  };
}

function computeExtent(atoms: Array<{ x: number; y: number; z: number }>): number {
  if (atoms.length < 2) return 0;
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (const a of atoms) {
    if (a.x < minX) minX = a.x; if (a.x > maxX) maxX = a.x;
    if (a.y < minY) minY = a.y; if (a.y > maxY) maxY = a.y;
    if (a.z < minZ) minZ = a.z; if (a.z > maxZ) maxZ = a.z;
  }
  return (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1);
}

function computePhononCoupling(film: string, substrate: string): InterfacePhononCoupling {
  const filmCounts = parseFormulaCounts(film);
  const subCounts = parseFormulaCounts(substrate);

  const avgMass = (counts: Record<string, number>) => {
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    return Object.entries(counts).reduce((s, [el, n]) => {
      const data = getElementData(el);
      return s + (data?.atomicMass ?? 50) * (n / total);
    }, 0);
  };

  const avgDebye = (counts: Record<string, number>) => {
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    return Object.entries(counts).reduce((s, [el, n]) => {
      const data = getElementData(el);
      return s + (data?.debyeTemperature ?? 300) * (n / total);
    }, 0);
  };

  const filmMass = avgMass(filmCounts);
  const subMass = avgMass(subCounts);
  const filmDebye = avgDebye(filmCounts);
  const subDebye = avgDebye(subCounts);

  const massRatio = filmMass / subMass;
  const isHeavyLightPair = massRatio < 0.6 || massRatio > 1.7;
  const substrateIsOxide = Object.keys(subCounts).includes("O");
  const filmIsLight = filmMass < 40;

  const debyeRatio = Math.min(filmDebye, subDebye) / Math.max(filmDebye, subDebye, 1);

  const phononOverlap = 1.0 - Math.abs(debyeRatio - 0.7);

  let couplingProxy = 0;
  if (substrateIsOxide && filmIsLight) couplingProxy += 0.35;
  if (isHeavyLightPair) couplingProxy += 0.25;
  couplingProxy += (1.0 - debyeRatio) * 0.2;
  couplingProxy += phononOverlap * 0.2;
  couplingProxy = Math.min(1.0, couplingProxy);

  return {
    massRatioFilmSub: Number(massRatio.toFixed(4)),
    isHeavyLightPair,
    substrateIsOxide,
    filmIsLight,
    couplingProxy: Number(couplingProxy.toFixed(4)),
    debyeTempRatio: Number(debyeRatio.toFixed(4)),
    phononOverlapScore: Number(phononOverlap.toFixed(4)),
  };
}

function computeElectronDensityOverlap(film: string, substrate: string): ElectronDensityOverlap {
  const filmCounts = parseFormulaCounts(film);
  const subCounts = parseFormulaCounts(substrate);

  const avgEN = (counts: Record<string, number>) => {
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    return Object.entries(counts).reduce((s, [el, n]) => {
      const data = getElementData(el);
      return s + (data?.paulingElectronegativity ?? 1.5) * (n / total);
    }, 0);
  };

  const avgWF = (counts: Record<string, number>) => {
    const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
    return Object.entries(counts).reduce((s, [el, n]) => {
      const data = getElementData(el);
      const ie = data?.firstIonizationEnergy ?? 7;
      const ea = data?.electronAffinity ?? 0;
      return s + ((ie + Math.max(0, ea)) / 2) * (n / total);
    }, 0);
  };

  const filmEN = avgEN(filmCounts);
  const subEN = avgEN(subCounts);
  const filmWF = avgWF(filmCounts);
  const subWF = avgWF(subCounts);

  const enDiff = Math.abs(filmEN - subEN);
  const wfDiff = Math.abs(filmWF - subWF);

  let bandAlignmentType: "type-I" | "type-II" | "type-III";
  if (enDiff < 0.3 && wfDiff < 1.0) bandAlignmentType = "type-I";
  else if (enDiff > 0.8 || wfDiff > 2.5) bandAlignmentType = "type-III";
  else bandAlignmentType = "type-II";

  const overlapScore = Math.min(1.0,
    enDiff * 0.3 +
    wfDiff * 0.15 +
    (bandAlignmentType === "type-II" ? 0.3 : bandAlignmentType === "type-III" ? 0.15 : 0.1) +
    (enDiff > 0.5 ? 0.15 : 0) +
    (wfDiff > 1.5 ? 0.1 : 0)
  );

  return {
    filmEN: Number(filmEN.toFixed(4)),
    subEN: Number(subEN.toFixed(4)),
    enDifference: Number(enDiff.toFixed(4)),
    workFunctionDiff: Number(wfDiff.toFixed(4)),
    bandAlignmentType,
    overlapScore: Number(overlapScore.toFixed(4)),
  };
}

function parseFormulaCounts(formula: string): Record<string, number> {
  const cleaned = formula
    .replace(/[₀-₉]/g, c => String("₀₁₂₃₄₅₆₇₈₉".indexOf(c)))
    .replace(/\s+/g, "");
  const counts: Record<string, number> = {};
  const regex = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let match;
  while ((match = regex.exec(cleaned)) !== null) {
    const el = match[1];
    const num = match[2] ? parseFloat(match[2]) : 1;
    if (num > 0) counts[el] = (counts[el] || 0) + num;
  }
  return counts;
}

function computeCompositeScore(
  bilayer: BilayerStructure,
  chargeTransfer: InterfaceChargeTransfer,
  strain: InterfaceStrain,
  phononCoupling: InterfacePhononCoupling,
  electronDensity: ElectronDensityOverlap
): { compositeScore: number; breakdown: InterfacePhysicsResult["scoreBreakdown"] } {
  const mismatch = bilayer.latticeMismatch;
  let latticeMatchScore: number;
  if (mismatch < 0.01) latticeMatchScore = 1.0;
  else if (mismatch < 0.03) latticeMatchScore = 0.8;
  else if (mismatch < 0.05) latticeMatchScore = 0.5;
  else if (mismatch < 0.07) latticeMatchScore = 0.3;
  else latticeMatchScore = 0.1;

  let chargeTransferScore = 0;
  if (chargeTransfer.isSignificant) chargeTransferScore += 0.4;
  chargeTransferScore += Math.min(0.6, chargeTransfer.chargePerAtom * 4);

  let strainScore = 0;
  if (strain.isOptimalRange) strainScore = 0.8;
  else if (strain.strainPercent >= 0.5 && strain.strainPercent <= 6.0) strainScore = 0.5;
  else if (strain.strainPercent < 0.5) strainScore = 0.3;
  else strainScore = 0.1;
  if (strain.avgAtomDisplacement > 0.1 && strain.avgAtomDisplacement < 1.0) strainScore += 0.2;
  strainScore = Math.min(1.0, strainScore);

  const phononCouplingScore = phononCoupling.couplingProxy;

  const electronDensityScore = electronDensity.overlapScore;

  const compositeScore =
    latticeMatchScore * 0.25 +
    chargeTransferScore * 0.25 +
    strainScore * 0.20 +
    phononCouplingScore * 0.15 +
    electronDensityScore * 0.15;

  return {
    compositeScore: Number(Math.min(1.0, compositeScore).toFixed(4)),
    breakdown: {
      latticeMatchScore: Number(latticeMatchScore.toFixed(4)),
      chargeTransferScore: Number(chargeTransferScore.toFixed(4)),
      strainScore: Number(strainScore.toFixed(4)),
      phononCouplingScore: Number(phononCouplingScore.toFixed(4)),
      electronDensityScore: Number(electronDensityScore.toFixed(4)),
    },
  };
}

const relaxationHistory: InterfacePhysicsResult[] = [];
const MAX_RELAXATION_HISTORY = 300;
let activeLearningSelections = 0;

export async function relaxInterface(
  film: string,
  substrate: string
): Promise<InterfacePhysicsResult> {
  const startTime = Date.now();

  const hetero = generateHeterostructure(film, substrate);
  const bilayer = hetero.bilayer;
  const structure = hetero.structure;

  let xtbConverged = false;
  let xtbEnergy: number | null = null;
  let relaxedAtoms: Array<{ element: string; x: number; y: number; z: number }> | null = null;
  let charges: AtomCharge[] = [];

  // GFN2-xTB and GFN-FF in the bundled Windows xtb.exe lack reliable parameters
  // for lanthanides (La–Lu, Z=57–71) and actinides (Ac–Lr, Z=89–103).
  // On Linux (GCP) the system xTB is newer and handles these elements correctly.
  const interfaceElements = new Set(structure.atoms.map(a => a.element));
  const LANTHANIDES_ACTINIDES = new Set([
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
  ]);
  const hasUnsupportedElement = IS_WINDOWS && XTB_BIN.endsWith(".exe") &&
    [...interfaceElements].some(el => LANTHANIDES_ACTINIDES.has(el));

  if (isXtbAvailable() && !hasUnsupportedElement && structure.totalAtoms <= 60 && structure.totalAtoms >= 2) {
    try {
      fs.mkdirSync(WORK_DIR, { recursive: true });
      const calcId = `iface_${film}_${substrate}_${Date.now()}`.replace(/[^a-zA-Z0-9_]/g, "_");
      const calcDir = path.join(WORK_DIR, calcId);
      fs.mkdirSync(calcDir, { recursive: true });

      const xyzPath = path.join(calcDir, "interface.xyz");
      writeInterfaceXYZ(structure.atoms, xyzPath, `${film}/${substrate} interface`);

      const env: Record<string, string> = {
        ...process.env as Record<string, string>,
        XTBHOME: XTB_HOME,
        XTBPATH: XTB_PARAM,
        OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "6",
        OMP_STACKSIZE: "512M",
      };

      // Try GFN-FF first (force-field, handles lanthanides/actinides robustly).
      // Fall back to GFN2-xTB tight-binding if GFN-FF is not available or fails.
      let output = "";
      let gfnffOk = false;
      try {
        output = execXtbCmd(`cd ${calcDir} && ${XTB_BIN} interface.xyz --gfnff --opt crude 2>&1`, {
          timeout: OPT_TIMEOUT_MS,
          env,
          maxBuffer: 10 * 1024 * 1024,
        });
        gfnffOk = output.includes("normal termination of xtb");
      } catch { /* fall through to GFN2 */ }

      if (!gfnffOk) {
        output = execXtbCmd(`cd ${calcDir} && ${XTB_BIN} interface.xyz --gfn 2 --opt crude 2>&1`, {
          timeout: OPT_TIMEOUT_MS,
          env,
          maxBuffer: 10 * 1024 * 1024,
        });
      }

      if (output.includes("normal termination of xtb")) {
        xtbConverged = true;
      }

      const energyMatch = output.match(/TOTAL ENERGY\s+([-\d.]+)\s+Eh/);
      if (energyMatch) {
        xtbEnergy = parseFloat(energyMatch[1]);
      }

      charges = parseXtbCharges(output, structure.atoms);
      relaxedAtoms = parseOptimizedAtoms(calcDir, structure.atoms);

      try {
        fs.rmSync(calcDir, { recursive: true, force: true });
      } catch {}
    } catch (e: any) {
      const stdout = e?.stdout?.toString?.()?.slice(0, 400) ?? "";
      const stderr = e?.stderr?.toString?.()?.slice(0, 400) ?? "";
      const msg = e?.message?.slice(0, 200) ?? "";
      console.log(`[InterfaceRelax] xTB failed for ${film}/${substrate}: ${msg}${stdout ? `\n  stdout: ${stdout}` : ""}${stderr ? `\n  stderr: ${stderr}` : ""}`);
    }
  }

  if (charges.length === 0) {
    charges = structure.atoms.map((a, i) => ({
      element: a.element,
      index: i,
      charge: estimateCharge(a.element, film, substrate),
      layer: a.layer,
    }));
  }

  const chargeTransfer = computeChargeTransfer(charges, bilayer);
  const strain = computeInterfaceStrain(structure.atoms, relaxedAtoms, bilayer);
  const phononCoupling = computePhononCoupling(film, substrate);
  const electronDensity = computeElectronDensityOverlap(film, substrate);

  const { compositeScore, breakdown } = computeCompositeScore(
    bilayer, chargeTransfer, strain, phononCoupling, electronDensity
  );

  const result: InterfacePhysicsResult = {
    film,
    substrate,
    chargeTransfer,
    strain,
    phononCoupling,
    electronDensity,
    compositeScore,
    scoreBreakdown: breakdown,
    xtbConverged,
    xtbEnergy,
    wallTimeMs: Date.now() - startTime,
    relaxedAtoms,
  };

  relaxationHistory.push(result);
  if (relaxationHistory.length > MAX_RELAXATION_HISTORY) {
    relaxationHistory.splice(0, Math.floor(MAX_RELAXATION_HISTORY * 0.1));
  }

  return result;
}

function estimateCharge(element: string, film: string, substrate: string): number {
  const data = getElementData(element);
  if (!data) return 0;

  const en = data.paulingElectronegativity ?? 1.5;

  const filmCounts = parseFormulaCounts(film);
  const subCounts = parseFormulaCounts(substrate);
  const isInFilm = Object.keys(filmCounts).includes(element);

  const filmEN = Object.keys(filmCounts).reduce((s, el) => {
    const d = getElementData(el);
    return s + (d?.paulingElectronegativity ?? 1.5);
  }, 0) / Math.max(1, Object.keys(filmCounts).length);

  const subEN = Object.keys(subCounts).reduce((s, el) => {
    const d = getElementData(el);
    return s + (d?.paulingElectronegativity ?? 1.5);
  }, 0) / Math.max(1, Object.keys(subCounts).length);

  const avgEN = isInFilm ? filmEN : subEN;
  return (avgEN - en) * 0.15;
}

export function scoreInterfaceCandidatesForActiveLearning(
  filmFormulas: string[],
  topN: number = 10
): InterfaceCandidate[] {
  const candidates: InterfaceCandidate[] = [];

  for (const film of filmFormulas) {
    const bestSubs = findBestSubstrates(film, 3);

    for (const subMatch of bestSubs) {
      const hetero = generateHeterostructure(film, subMatch.substrate.formula);

      const phononCoupling = computePhononCoupling(film, subMatch.substrate.formula);
      const electronDensity = computeElectronDensityOverlap(film, subMatch.substrate.formula);

      const quickScore =
        (hetero.bilayer.mismatchQuality === "ideal" ? 0.4 : hetero.bilayer.mismatchQuality === "workable" ? 0.2 : 0.05) +
        hetero.chargeTransferPotential * 0.25 +
        phononCoupling.couplingProxy * 0.2 +
        electronDensity.overlapScore * 0.15;

      candidates.push({
        film,
        substrate: subMatch.substrate.formula,
        heterostructure: hetero,
        physics: null,
        rank: 0,
        selectedForDFT: false,
      });
    }
  }

  candidates.sort((a, b) => b.heterostructure.interfaceScore - a.heterostructure.interfaceScore);
  candidates.forEach((c, i) => c.rank = i + 1);

  return candidates.slice(0, topN);
}

export async function runInterfaceDiscoveryForActiveLearning(
  filmFormulas: string[],
  budget: number = 3
): Promise<InterfacePhysicsResult[]> {
  const candidates = scoreInterfaceCandidatesForActiveLearning(filmFormulas, budget * 2);

  const results: InterfacePhysicsResult[] = [];
  let relaxed = 0;

  for (const candidate of candidates) {
    if (relaxed >= budget) break;

    try {
      const physics = await relaxInterface(candidate.film, candidate.substrate);
      candidate.physics = physics;
      candidate.selectedForDFT = physics.compositeScore > 0.35;
      results.push(physics);
      relaxed++;

      if (candidate.selectedForDFT) {
        activeLearningSelections++;
      }
    } catch (e: any) {
      console.log(`[InterfaceDiscovery] Failed ${candidate.film}/${candidate.substrate}: ${e?.message?.slice(0, 150)}`);
    }
  }

  results.sort((a, b) => b.compositeScore - a.compositeScore);
  return results;
}

export function getInterfaceRelaxationStats(): InterfaceDiscoveryStats {
  const n = relaxationHistory.length;
  if (n === 0) {
    return {
      totalRelaxations: 0,
      xtbSuccesses: 0,
      xtbFailures: 0,
      avgCompositeScore: 0,
      avgChargeTransfer: 0,
      avgStrain: 0,
      significantChargeTransferCount: 0,
      optimalStrainCount: 0,
      topInterfaces: [],
      recentRelaxations: [],
      activeLearningSelections,
    };
  }

  const xtbSuccesses = relaxationHistory.filter(r => r.xtbConverged).length;

  let scoreSum = 0, ctSum = 0, strainSum = 0;
  let sigCT = 0, optStrain = 0;

  for (const r of relaxationHistory) {
    scoreSum += r.compositeScore;
    ctSum += r.chargeTransfer.chargePerAtom;
    strainSum += r.strain.strainPercent;
    if (r.chargeTransfer.isSignificant) sigCT++;
    if (r.strain.isOptimalRange) optStrain++;
  }

  const sorted = [...relaxationHistory].sort((a, b) => b.compositeScore - a.compositeScore);
  const topInterfaces = sorted.slice(0, 10).map(r => ({
    film: r.film,
    substrate: r.substrate,
    compositeScore: r.compositeScore,
    chargePerAtom: r.chargeTransfer.chargePerAtom,
    strainPct: r.strain.strainPercent,
    phononCoupling: r.phononCoupling.couplingProxy,
  }));

  const recentRelaxations = relaxationHistory.slice(-10).reverse().map(r => ({
    film: r.film,
    substrate: r.substrate,
    compositeScore: r.compositeScore,
    xtbConverged: r.xtbConverged,
    wallTimeMs: r.wallTimeMs,
  }));

  return {
    totalRelaxations: n,
    xtbSuccesses,
    xtbFailures: n - xtbSuccesses,
    avgCompositeScore: Number((scoreSum / n).toFixed(4)),
    avgChargeTransfer: Number((ctSum / n).toFixed(5)),
    avgStrain: Number((strainSum / n).toFixed(3)),
    significantChargeTransferCount: sigCT,
    optimalStrainCount: optStrain,
    topInterfaces,
    recentRelaxations,
    activeLearningSelections,
  };
}
