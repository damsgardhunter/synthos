# Structure Template Builder — One Iteration

You are the crystal structure template builder for the **Quantum Alchemy Engine (QAE)**. Your job is to expand the prototype template library and known structures database so that ANY material a user enters gets proper Wyckoff atomic positions for DFT calculations.

Use `/loop 10m /structure-template-builder` to run on a schedule.

**CRITICAL RULE: Each iteration MUST add at least 1 new PROTOTYPE TEMPLATE to `server/learning/crystal-prototypes.ts`. Known structure entries in `server/learning/known-structures.ts` are secondary — templates are the priority because one template handles hundreds of compositions.**

---

## STEP 1 — Audit What Exists

Read the current state:

1. Read `server/learning/crystal-prototypes.ts` — count all entries in `PROTOTYPE_TEMPLATES` array. List their names.
2. Read `server/learning/known-structures.ts` — count all entries in `KNOWN_STRUCTURES`. List their formulas.
3. Check the TODO list below for which families still need templates.

## STEP 2 — Choose What to Add

Pick the HIGHEST PRIORITY missing template family from this list:

### Template families still needed (in priority order):
1. **122-type iron pnictide** (I4/mmm, ThCr2Si2-derived but specifically for AeFe2Pn2) — check if existing ThCr2Si2 already handles this correctly with chemistry rules
2. **111-type iron pnictide** (P4/nmm, LiFeAs-type) — AFeAs stoichiometry
3. **YBCO-123 cuprate** (Pmmm) — generic ABa2Cu3O7 template
4. **Bi-2212 cuprate** (I4/mmm) — layered Bi2Sr2CaCu2O8+δ 
5. **Hexaboride** (Pm-3m, CaB6-type) — 1 M + 6 B in octahedral cage
6. **Diamond cubic** (Fd-3m) — elemental Si/Ge/C type
7. **Garnet** (Ia-3d) — A3B2(SiO4)3
8. **1T dichalcogenide** (P-3m1, CdI2-type) — distinct from 2H MX2
9. **Inverse Heusler** (F-43m) — XA2B ordering
10. **M3AX2 MAX phase** (P63/mmc) — 312 variant
11. **M4AX3 MAX phase** (P63/mmc) — 413 variant
12. **Ruddlesden-Popper n=2** (I4/mmm) — A3B2O7
13. **Ruddlesden-Popper n=3** (I4/mmm) — A4B3O10
14. **Double perovskite** (Fm-3m) — A2BB'O6
15. **Pyrochlore** (Fd-3m) — A2B2O7 (check if existing handles it)
16. **Olivine** (Pnma) — A2BO4 (battery cathode material)
17. **Brownmillerite** (Ibm2) — A2B2O5 (oxygen-deficient perovskite)
18. **Delafossite** (R-3m) — ABO2 (CuFeO2-type)
19. **Chalcopyrite** (I-42d) — ABX2 (CuFeS2-type)
20. **Layered oxide** (R-3m) — AMO2 (LiCoO2-type, battery materials)

### For each template, you MUST:
- Use correct space group
- List ALL Wyckoff positions in the PRIMITIVE cell (not conventional)
- Set correct stoichiometryRatio matching the site counts
- Write chemistryRules that match the intended family WITHOUT matching unrelated compounds
- Add the template name to PACKING_FACTORS

## STEP 3 — Implement

1. Add the template to `PROTOTYPE_TEMPLATES` in `server/learning/crystal-prototypes.ts`
2. Add its packing factor
3. Optionally add 1-3 known structure entries for specific compounds in that family
4. Type-check: `npx tsc --noEmit server/learning/crystal-prototypes.ts server/learning/known-structures.ts`
5. Commit and push with a descriptive message

## STEP 4 — Verify

After adding, verify:
- The template site count matches stoichiometryRatio
- chemistryRules won't accidentally match unrelated compounds
- New templates don't shadow existing ones (check ordering in the array)

## STEP 5 — Report

Log what you added:
- Template name, space group, site count, stoichiometry
- Any known structures added
- What's next on the priority list

---

## Rules

- **ALWAYS add at least 1 template per iteration** — do NOT just add known structures
- Use literature Wyckoff positions (Bilbao Crystallographic Server, ICSD, Materials Project)
- Primitive cell positions, not conventional cell (unless ibrav=0 with conventional vectors)
- Test that stoichiometry ratios are correct before committing
- Don't break existing templates — only ADD new ones
