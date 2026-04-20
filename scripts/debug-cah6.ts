import { computePhysicsTcUQ, VERIFIED_COMPOUNDS, allenDynesTcUncalibrated } from "../server/learning/physics-engine";

const v = VERIFIED_COMPOUNDS["CaH6"];
console.log("CaH6 verified:", JSON.stringify(v));
const result = computePhysicsTcUQ("CaH6", v.pressureGpa);
console.log("CaH6 UQ mean:", result.mean, "analyticMean:", result.analyticMean, "mcMean:", result.mcMean);

const tcRaw = allenDynesTcUncalibrated(v.lambda, v.omegaLog, v.muStar, v.omega2Avg, true, "CaH6", v.pressureGpa);
console.log("CaH6 raw AD Tc:", tcRaw);

// Test with isHydride=false to see how much the hydride corrections contribute
const tcNoHydride = allenDynesTcUncalibrated(v.lambda, v.omegaLog, v.muStar, v.omega2Avg, false, "CaH6", v.pressureGpa);
console.log("CaH6 AD (isHydride=false):", tcNoHydride);

// H3S
const h = VERIFIED_COMPOUNDS["H3S"];
const h3sRaw = allenDynesTcUncalibrated(h.lambda, h.omegaLog, h.muStar, h.omega2Avg, true, "H3S", h.pressureGpa);
console.log("H3S raw AD Tc:", h3sRaw);
const h3sUQ = computePhysicsTcUQ("H3S", h.pressureGpa);
console.log("H3S UQ mean:", h3sUQ.mean);
