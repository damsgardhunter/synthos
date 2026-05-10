import { allenDynesTcRaw, isAllenDynesApplicable, allenDynesTcUncalibrated } from "../server/learning/physics-engine";

console.log("=== MoH6N at 140 GPa ===");
const adCheck = isAllenDynesApplicable("MoH6N", 1.5, 140);
console.log("isAllenDynesApplicable:", JSON.stringify(adCheck));
console.log("uncalibrated:", allenDynesTcUncalibrated(1.5, 1500, 0.10, undefined, true, "MoH6N").toFixed(1));
console.log("legacy (no formula):", allenDynesTcRaw(1.5, 1500, 0.10, undefined, true).toFixed(1));
console.log("gated (formula+P):", allenDynesTcRaw(1.5, 1500, 0.10, undefined, true, "MoH6N", 140).toFixed(1));

console.log("\n=== LaH10 at 170 GPa (sanity) ===");
console.log("LaH10 gated:", allenDynesTcRaw(3.0, 1500, 0.10, undefined, true, "LaH10", 170).toFixed(1));

console.log("\n=== BaH5Sr2 at 0 GPa (heavy-atom hydride) ===");
console.log("BaH5Sr2 gated:", allenDynesTcRaw(1.5, 1500, 0.10, undefined, true, "BaH5Sr2", 0).toFixed(1));

console.log("\n=== Y3H7 at 0 GPa (Y is clathrate-capable but ambient) ===");
console.log("Y3H7 gated:", allenDynesTcRaw(1.5, 1500, 0.10, undefined, true, "Y3H7", 0).toFixed(1));

console.log("\n=== H3S at 200 GPa (real superhydride) ===");
console.log("H3S gated:", allenDynesTcRaw(2.2, 1335, 0.10, undefined, true, "H3S", 200).toFixed(1));
