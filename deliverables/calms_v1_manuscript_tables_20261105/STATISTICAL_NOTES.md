# Statistical notes for CALMS manuscript main tables

## Frozen method and evidence role

- CALMS v1.0 is the frozen released method (internal identifier `full23_v132`,
  internal version Full23 v1.3.2). No model was fitted and no bootstrap was
  rerun while producing these tables.
- GNPS Gold/Silver is **consumed development evidence**. It was used during
  CALMS development, including residual-amplitude selection. These tables are
  not unopened external confirmation and do not by themselves establish a
  field-wide SOTA claim.

## Two axes that must not be mixed

- Table 1 uses the strict 18-method common-availability axis: 10,983
  identity-disjoint queries (4,199 formula clusters) and 5,253
  formula-disjoint queries (2,207 formula clusters). This retains
  10,983/10,995 and 5,253/5,261 queries from the corresponding full panels.
- Table 2 uses each native full panel: 10,995 identity-disjoint queries (4,204
  formula clusters) and 5,261 formula-disjoint queries (2,210 formula
  clusters). Values can differ slightly from Table 1 because the axes differ.
- The two panels are not independent: 5,237 of 5,261 formula-panel query rows
  also occur in the identity panel. Query counts, corrected/introduced counts,
  and apparent discoveries must never be added across panels.

## Intervals and multiplicity

- Table 1 Recall@1 intervals use formula-cluster bootstrap resampling and are
  familywise-adjusted over 20 endpoints **within one CALMS-versus-one-
  comparator analysis**. They are not simultaneous over all 17 competitors,
  over both panels, or over prior CALMS development and scale selection.
- Table 2 global and near-subset intervals are descriptive 95% formula-cluster
  bootstrap intervals (10,000 resamples, seed 20261105) with no multiplicity
  adjustment. `corrected` and `introduced` are oriented from WSE to CALMS.
- The 90%-coverage interval is a paired functional bootstrap delta. Each
  method selects its own highest-confidence prefix at matched coverage under
  the same formula-cluster draw; the accepted query sets need not be identical.
- AURC relative reduction is descriptive. No bootstrap interval was computed
  for the AURC difference.

## Closed-set and mechanism boundaries

- Every query has its true candidate in the candidate block. The selective
  analysis is closed-set and does not evaluate no-match detection, open-set
  false acceptance, FDR control, calibrated match probability, or a deployable
  abstention threshold.
- Confidence is the frozen Top1-Top2 candidate-score gap. Margin correctness
  AUROC measures confidence ordering; it is distinct from the candidate-level
  micro AUROC in Table 1.
- The `near` subset uses truth/structure metadata. It is a post-freeze
  diagnostic subset, not an observable deployment router.
- All percentages in the CSVs are percentage values (for example, 88.4), and
  all `*_pp` fields are percentage-point differences.
- `common_domain_retention` is a fraction of the native full-panel query count,
  not a percentage.

## Table 3 is deliberately two-layered

- `parent_strict_refit` rows test the Full23 v1 **candidate-evidence parent**.
  Each reduced parent was strictly refitted with the same low-capacity Enveda
  formula-OOF protocol and dependent reliability summaries recomputed after
  feature removal. Full-minus-reduced differences estimate grouped evidence
  contributions under that fixed protocol; they do not isolate a single
  module's causal effect.
- The four parent comparisons are: full parent minus the DreaMS+WSE learned
  Additive baseline, minus the representation-group ablation, minus the
  ChemAware V2 ablation, and minus the P2b+MSG ablation. Enveda rows are
  consumed meta-development results; GNPS rows are transfer of the fixed
  Enveda-fitted models to already consumed development data.
- `final_embedding_increment` rows answer a different question: frozen final
  Full23 v1.3.2 minus its Full23 v1 parent on GNPS. The correction is
  `Full23_v1 + 0.9 * (Full23_v1.3.1 - Full23_v1)`. The 0.9 scale was selected
  **post-result after inspection of consumed GNPS development predictions**.
  Both formula-cluster intervals cross zero.
- Parent and final-increment rows must not be added. Table 3 is not a complete
  strict-refit ablation of final CALMS v1.0/v1.3.2, and it cannot assign the
  final model's total gain to individual modules.
