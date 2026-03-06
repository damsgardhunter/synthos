export interface FamilyGenerationResult {
  formulas: string[];
  family: string;
}

export function generateMAXPhase(): FamilyGenerationResult {
  const M = ["Ti", "V", "Cr", "Nb", "Mo"];
  const A = ["Al", "Si", "Ga", "Ge", "Sn"];
  const X = ["C", "N"];
  const nValues = [1, 2, 3];
  const formulas: string[] = [];

  for (const m of M) {
    for (const a of A) {
      for (const x of X) {
        for (const n of nValues) {
          const mCount = n + 1;
          const xCount = n;
          let formula: string;
          if (mCount === 1) {
            formula = `${m}${a}`;
          } else {
            formula = `${m}${mCount}${a}`;
          }
          if (xCount === 1) {
            formula += x;
          } else {
            formula += `${x}${xCount}`;
          }
          formulas.push(formula);
        }
      }
    }
  }

  return { formulas, family: "MAX-phase" };
}

export function generateBoride(): FamilyGenerationResult {
  const M = ["Mg", "Ca", "Sc", "Y", "Ti", "Zr", "Hf"];
  const stoichiometries = [
    { m: 1, b: 2 },
    { m: 1, b: 4 },
    { m: 1, b: 6 },
    { m: 1, b: 12 },
    { m: 2, b: 5 },
  ];
  const formulas: string[] = [];

  for (const m of M) {
    for (const s of stoichiometries) {
      let formula = s.m === 1 ? m : `${m}${s.m}`;
      formula += `B${s.b}`;
      formulas.push(formula);
    }
  }

  return { formulas, family: "Boride" };
}

export function generateHydride(): FamilyGenerationResult {
  const M = ["La", "Y", "Ca", "Sc", "Th"];
  const hCounts = [6, 9, 10, 12];
  const formulas: string[] = [];

  for (const m of M) {
    for (const h of hCounts) {
      formulas.push(`${m}H${h}`);
    }
  }

  return { formulas, family: "Hydride" };
}

export function generateIntercalatedNitride(): FamilyGenerationResult {
  const M = ["Zr", "Hf", "Ti"];
  const X = ["Cl", "Br", "I"];
  const A = ["Li", "Na", "K"];
  const xValues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const formulas: string[] = [];

  for (const a of A) {
    for (const x of xValues) {
      for (const m of M) {
        for (const halide of X) {
          const xStr = x === 1.0 ? "" : `${Math.round(x * 10) / 10}`;
          const aComponent = xStr ? `${a}${xStr}` : a;
          formulas.push(`${aComponent}${m}N${halide}`);
        }
      }
    }
  }

  return { formulas, family: "Intercalated-nitride" };
}

export function runFamilyAwareGeneration(): FamilyGenerationResult[] {
  return [
    generateMAXPhase(),
    generateBoride(),
    generateHydride(),
    generateIntercalatedNitride(),
  ];
}
