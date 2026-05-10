import { computePhysicsTcUQ, computeElectronicStructure, computePhononSpectrum, computeElectronPhononCoupling } from "../server/learning/physics-engine";

const formulas: [string, number][] = [
  ["B3H4Sc2", 150],
  ["B3H5LiSc2", 150],
  ["BeH4N", 100],
  ["B3H4Y2", 150],
  ["MgB2", 0],
  ["CaH6", 172],
  ["LaH10", 170],
  ["H3S", 155],
  ["ScH9", 200],
];

console.log("Formula".padEnd(14) + "lambda".padStart(8) + "omegaLog".padStart(10) + "muStar".padStart(8) + "Tc(UQ)".padStart(8));
console.log("-".repeat(48));

for (const [f, p] of formulas) {
  const e = computeElectronicStructure(f);
  const ph = computePhononSpectrum(f, e, p);
  const c = computeElectronPhononCoupling(e, ph, f, p);
  const uq = computePhysicsTcUQ(f, p);
  console.log(f.padEnd(14) + c.lambda.toFixed(3).padStart(8) + c.omegaLog.toFixed(0).padStart(10) + c.muStar.toFixed(3).padStart(8) + (uq.mean.toFixed(1) + "K").padStart(8));
}
