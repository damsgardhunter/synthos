import { computePhysicsTcUQ, computeElectronicStructure, computeElectronPhononCoupling, computePhononSpectrum } from "../server/learning/physics-engine";

const checks: [string, number][] = [
  ["HF", 0],
  ["C2F7H19Mo4N4", 0],
  ["BiHK2S2", 0],
  ["BaCaH2", 100],
  ["H4Li3N3Pr2", 100],
  ["Gd3H3Sr2", 150],
];

console.log("Formula".padEnd(16) + "metallicity".padStart(12) + "bandGap".padStart(9) + "lambda".padStart(8) + "Tc(UQ)".padStart(8));
console.log("-".repeat(53));

for (const [f, p] of checks) {
  const e = computeElectronicStructure(f);
  const ph = computePhononSpectrum(f, e, p);
  const c = computeElectronPhononCoupling(e, ph, f, p);
  const uq = computePhysicsTcUQ(f, p);
  console.log(
    f.padEnd(16) +
    e.metallicity.toFixed(2).padStart(12) +
    (e.bandGap ?? 0).toFixed(2).padStart(9) +
    c.lambda.toFixed(3).padStart(8) +
    (uq.mean.toFixed(1) + "K").padStart(8)
  );
}
