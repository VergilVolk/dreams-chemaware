# CALMS Enveda-to-GNPS query-formula overlap audit

Date: 2026-11-05  
Status: **POST-FREEZE QUERY-FORMULA OVERLAP AUDIT COMPLETE**  
Frozen method: **CALMS v1.0** (internal implementation Full23 v1.3.2)

## Question

Does the word `formula-disjoint` imply that the final GNPS panels contain no
query formulas seen anywhere in the complete Enveda data used to fit the
frozen CALMS parent?

No. The panel name describes the construction of that panel, not a global
zero-overlap guarantee against the complete, later Enveda refit union.

## Frozen inputs

| Role | File | SHA-256 |
|---|---|---|
| Final Enveda OOF formula axes | `data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/candidate_predictions_oof.npz` | `0d2f8a7a07c701b5482fd0f415cd608fcc71e878a0eb73630f674dcb81111d19` |
| GNPS identity panel used by final evaluation | `data/validation/full23_v131_direct_repairs_run_2434102/gnps_benchmark_view/panel_identity_disjoint.npz` | `3d53b8a17fcc8391eb5800c11db6e424577aeeffb3836da242934de50b96c007` |
| GNPS formula panel used by final evaluation | `data/validation/full23_v131_direct_repairs_run_2434102/gnps_benchmark_view/panel_formula_disjoint.npz` | `efbd24ca71c9570785bcc9c50c5cb2b79d15b6125b94bc3348ce331ee74a7fa1` |
| Frozen candidate-evidence parent | `data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/unified_candidate_integrator.pt` | `7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717` |

The audit stripped surrounding whitespace, formed unique sets from the two
Enveda `__formula` arrays and the two GNPS `query_formula` arrays, and computed
exact string intersections. It did not refit or evaluate any model.

## Result

| Query-formula set | Unique formulas |
|---|---:|
| Enveda identity-disjoint training panel | 8,454 |
| Enveda formula-disjoint training panel | 4,426 |
| Complete Enveda training union | 10,179 |
| GNPS identity panel | 4,204 |
| GNPS formula panel | 2,210 |

| Exact intersection | Shared formulas |
|---|---:|
| Enveda formula-only vs GNPS identity | 0 |
| Enveda formula-only vs GNPS formula | 0 |
| Complete Enveda union vs GNPS identity | **1,046** |
| Complete Enveda union vs GNPS formula | **339** |

All nonzero overlap came from the Enveda identity-disjoint training panel,
whose scientific design permits seen molecular formulas. The two counts cover
24.88% and 15.34% of the unique GNPS identity and formula query-formula sets,
respectively.

## Interpretation boundary

This result does **not** establish query-identity leakage, candidate-label
leakage or train/test spectrum duplication. It establishes a narrower and
important fact: the final GNPS `formula-disjoint` name must not be expanded
into a claim of formula disjointness from the complete Enveda training union.
Together with the documented use of GNPS for architecture diagnosis and the
final 0.9 residual multiplier, this supports treating GNPS strictly as consumed
development evidence rather than untouched external confirmation.

This audit covers the model's `query_formula` grouping axis. It does not claim
an exhaustive candidate-level `molecule_formula` overlap audit.
