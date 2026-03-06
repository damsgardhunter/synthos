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

export function generateKagomeMetal(): FamilyGenerationResult {
  const A = ["K", "Rb", "Cs", "Ca", "Sr", "Ba"];
  const V_SITE = ["V", "Ti", "Cr", "Mn", "Fe", "Co", "Ni"];
  const X = ["Sb", "Bi", "Sn", "Ge", "As"];
  const formulas: string[] = [];

  for (const a of A) {
    for (const v of V_SITE) {
      for (const x of X) {
        formulas.push(`${a}${v}3${x}5`);
      }
    }
  }

  for (const a of A) {
    for (const v of V_SITE) {
      for (const x of X) {
        formulas.push(`${a}${v}3${x}4`);
        formulas.push(`${a}2${v}3${x}5`);
      }
    }
  }

  return { formulas, family: "Kagome" };
}

export function generateMixedMechanism(): FamilyGenerationResult {
  const formulas: string[] = [];

  const feAsSpacers = ["La", "Ce", "Pr", "Nd", "Sm", "Gd", "Ba", "Sr", "Ca"];
  for (const sp of feAsSpacers) {
    formulas.push(`${sp}FeAsO`);
    formulas.push(`${sp}FeAsF`);
    formulas.push(`${sp}Fe2As2`);
  }

  const feSeVariants = ["Li", "Na", "K", "Ca", "Sr", "Ba"];
  for (const sp of feSeVariants) {
    formulas.push(`${sp}FeSe2`);
    formulas.push(`FeSe`);
    formulas.push(`FeSe0.5Te0.5`);
  }
  formulas.push("FeSe");
  formulas.push("FeTe");
  formulas.push("Fe1.1Se");

  const niOSpacers = ["La", "Nd", "Pr", "Sm"];
  for (const sp of niOSpacers) {
    formulas.push(`${sp}NiO2`);
    formulas.push(`${sp}NiO3`);
    formulas.push(`${sp}4Ni3O8`);
  }

  const cuOSpacers = ["La", "Y", "Bi", "Tl", "Hg", "Ba", "Sr", "Ca", "Nd"];
  for (const sp of cuOSpacers) {
    formulas.push(`${sp}2CuO4`);
    formulas.push(`${sp}Ba2Cu3O7`);
  }

  const mixedTM = ["Fe", "Ni", "Co", "Cu"];
  const chalcogens = ["Se", "Te", "S"];
  const pnictogens = ["As", "P", "Sb"];
  const spacers = ["La", "Ba", "Sr", "Ca", "Y", "Ce"];
  for (const tm of mixedTM) {
    for (const ch of chalcogens) {
      formulas.push(`${tm}${ch}`);
      formulas.push(`${tm}${ch}2`);
      for (const sp of spacers.slice(0, 3)) {
        formulas.push(`${sp}${tm}2${ch}2`);
      }
    }
    for (const pn of pnictogens) {
      for (const sp of spacers.slice(0, 3)) {
        formulas.push(`${sp}${tm}${pn}O`);
        formulas.push(`${sp}${tm}2${pn}2`);
      }
    }
  }

  const unique = Array.from(new Set(formulas));
  return { formulas: unique, family: "Mixed-mechanism" };
}

export function generateLayeredChalcogenide(): FamilyGenerationResult {
  const M = ["Nb", "Ta", "Mo", "W", "Ti", "Zr", "Hf", "V", "Re"];
  const X = ["Se", "S", "Te"];
  const stoichiometries = [
    { m: 1, x: 2 },
    { m: 1, x: 3 },
    { m: 2, x: 3 },
  ];
  const formulas: string[] = [];

  for (const m of M) {
    for (const x of X) {
      for (const s of stoichiometries) {
        let formula = s.m === 1 ? m : `${m}${s.m}`;
        formula += s.x === 1 ? x : `${x}${s.x}`;
        formulas.push(formula);
      }
    }
  }

  const intercalants = ["Li", "Na", "K", "Ca", "Sr", "Cu"];
  for (const ic of intercalants) {
    for (const m of M.slice(0, 5)) {
      for (const x of X.slice(0, 2)) {
        formulas.push(`${ic}${m}${x}2`);
      }
    }
  }

  return { formulas, family: "Layered-chalcogenide" };
}

export function generateLayeredPnictide(): FamilyGenerationResult {
  const rareEarths = ["La", "Ce", "Pr", "Nd", "Sm", "Gd"];
  const alkalineEarths = ["Ba", "Sr", "Ca"];
  const transitionMetals = ["Fe", "Co", "Ni", "Mn", "Ru"];
  const pnictogens = ["As", "P", "Sb"];
  const formulas: string[] = [];

  for (const re of rareEarths) {
    for (const tm of transitionMetals) {
      for (const pn of pnictogens) {
        formulas.push(`${re}${tm}${pn}O`);
        formulas.push(`${re}${tm}2${pn}2`);
      }
    }
  }

  for (const ae of alkalineEarths) {
    for (const tm of transitionMetals) {
      for (const pn of pnictogens) {
        formulas.push(`${ae}${tm}2${pn}2`);
        formulas.push(`${ae}2${tm}2${pn}2O`);
      }
    }
  }

  for (const tm of transitionMetals.slice(0, 3)) {
    for (const pn of pnictogens) {
      formulas.push(`${tm}${pn}`);
      formulas.push(`${tm}2${pn}`);
    }
  }

  return { formulas, family: "Layered-pnictide" };
}

export function generateIntercalatedLayered(): FamilyGenerationResult {
  const intercalants = ["Li", "Na", "K", "Rb", "Cs", "Ca", "Sr", "Ba", "Eu", "Yb"];
  const hostMetals = ["Nb", "Ta", "Mo", "W", "Ti"];
  const hostAnions = ["Se", "S", "Te"];
  const formulas: string[] = [];

  for (const ic of intercalants) {
    for (const m of hostMetals) {
      for (const x of hostAnions) {
        formulas.push(`${ic}${m}${x}2`);
      }
    }
  }

  const graphiteIntercalants = ["Li", "K", "Rb", "Cs", "Ca", "Sr", "Ba", "Eu", "Yb"];
  for (const ic of graphiteIntercalants) {
    formulas.push(`${ic}C6`);
    formulas.push(`${ic}C8`);
    formulas.push(`${ic}C12`);
  }

  for (const ic of intercalants.slice(0, 5)) {
    for (const m of hostMetals.slice(0, 3)) {
      formulas.push(`${ic}${m}O2`);
    }
  }

  return { formulas, family: "Intercalated-layered" };
}

export function runFamilyAwareGeneration(): FamilyGenerationResult[] {
  return [
    generateMAXPhase(),
    generateBoride(),
    generateHydride(),
    generateIntercalatedNitride(),
    generateKagomeMetal(),
    generateMixedMechanism(),
    generateLayeredChalcogenide(),
    generateLayeredPnictide(),
    generateIntercalatedLayered(),
  ];
}
