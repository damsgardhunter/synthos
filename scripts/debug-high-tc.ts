import { computePhysicsTcUQ, computeElectronPhononCoupling, computeElectronicStructure, computePhononSpectrum } from "../server/learning/physics-engine";

const tests: [string, number][] = [
  ["K3LaH11", 230],
  ["K2LaH8", 200],
  ["Ca2H5Nd3", 0],
  ["BaH5Sr", 200],
  ["H3S", 155],
  ["H3Sm", 170],
  ["EuH3", 0],
];

for (const [f, p] of tests) {
  const elec = computeElectronicStructure(f);
  const phon = computePhononSpectrum(f, elec, p);
  const coup = computeElectronPhononCoupling(elec, phon, f, p);
  const r = computePhysicsTcUQ(f, p);
  console.log(`${f} @${p}GPa: Tc=${r.mean}K | lambda=${coup.lambda.toFixed(2)} omegaLog=${coup.omegaLog.toFixed(0)} muStar=${coup.muStar.toFixed(3)} metal=${elec.metallicity.toFixed(2)}`);
}
process.exit(0);
