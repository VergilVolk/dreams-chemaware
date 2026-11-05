# CALMS v1.0 submission asset index

Date: 2026-11-05

This index separates executable release artifacts, manuscript evidence and
items that remain outside the frozen deliverable.

## Frozen release artifacts

| Artifact | Purpose | SHA-256 / verification |
|---|---|---|
| `deliverables/CALMS_v1.0_20261105.zip` | Portable frozen candidate scorer for Code Availability | `7305d38b74dad4edf74c1313e5a3989c72df1cfffd030781336a11dce2be6deb` |
| `deliverables/CALMS_v1.0_evidence_bundle_20261105.zip` | Compact reports, tables, figures and release audit for Supporting Information | `f6995e2acaa746d3a808d066e6129b71e1d692c8fe429c96b12bb2cc1a431648` |

The standalone archive contains 20 registered payloads plus its checksum
ledger. The evidence bundle contains 61 registered payloads plus its checksum
ledger. Both ledgers were independently recomputed without mismatch.

Rebuild commands:

```bash
python tasks/build_calms_v1_release_archive.py \
  --source deploy/calms_v1 \
  --output deliverables/CALMS_v1.0_20261105.zip

python tasks/build_calms_v1_evidence_bundle.py \
  --output deliverables/CALMS_v1.0_evidence_bundle_20261105.zip
```

## Manuscript tables

| Artifact | Recommended placement | SHA-256 |
|---|---|---|
| `deliverables/calms_v1_manuscript_tables_20261105/table1_primary_results.csv` | Main Table 1: common-axis Top-k, MRR, AUROC/AP and paired comparisons | `dd6f46f45282319a2499da41623f33f750966752de33e9b48d92992f613ee6a6` |
| `deliverables/calms_v1_manuscript_tables_20261105/table2_mechanism_selective.csv` | Main mechanism/reliability table or source for Fig. 2 | `90af5f0201b96244b81a27efc39115bc92fcf29ed95f87ebde18acfef71d9479` |
| `deliverables/calms_v1_manuscript_tables_20261105/table3_two_layer_ablation.csv` | Supplementary Table: parent refits and final embedding increment | `8b7fc3eb6be022b1d95d9a98c4b18d405e4c64acb5609fed592dd4dddee2c52f` |
| `deliverables/calms_v1_manuscript_tables_20261105/STATISTICAL_NOTES.md` | Supplementary Methods footnotes | `d2e6dc7368c15299eebffce3207dc3a76be181bf13998ceacc58a32b0ed345e2` |

Reproduce the tables from an extracted evidence bundle with:

```bash
python tasks/build_calms_manuscript_main_tables.py \
  --evidence-root . \
  --output reproduced_tables
```

## Figures

| Artifact | Recommended placement | Notes |
|---|---|---|
| `deliverables/calms_project_summary_20261103/calms_project_summary.pdf` | Graphical Abstract or overview figure | SHA-256 `e8adede2ec09c63665d9beec86ca5fa318d691597dce74741463e91c9e82fc5b`; biological-sample relationships are a conditional project extension, not part of the frozen scorer |
| `deliverables/calms_v1_mechanism_selective_20261105/calms_mechanism_selective.pdf` | Main mechanism/selective-ranking figure | SHA-256 `4672eb857feb1c1a924028dea474e796df372c0a49c48fc4cdd36b3b817c88d3`; source is included in the evidence bundle |
| `deliverables/full23_v132_manuscript_figures_20261102/` | Supplementary figure source | Must be relabelled from internal `Full23 v1.3.2` to public `CALMS v1.0` before submission |

The phrase “approximately 9 pp” is ambiguous across current figures. AUPRC
versus WSE and Recall@1 versus MS2DeepScore 2.x are different endpoints and
comparators; every caption must name both.

## Manuscript text and audit

| Artifact | Purpose |
|---|---|
| `docs/CALMS_MANUSCRIPT_METHODS_RESULTS_DRAFT_20261105.md` | Frozen Methods and Results draft |
| `docs/CALMS_MANUSCRIPT_ABSTRACT_DISCUSSION_DRAFT_20261105.md` | Abstract and Discussion draft |
| `docs/CALMS_REPRODUCIBILITY_CHECKLIST_20261105.md` | Supplementary reproducibility checklist |

## Submission allocation

### Main text

- CALMS architecture and explicit missing-evidence semantics;
- strict common-axis Table 1;
- mechanism/selective-ranking figure;
- one application result panel that distinguishes complete bidirectional,
  forward endpoint, candidate-pair and scored-query denominators;
- limitations stating consumed development, closed-set truth presence and no
  calibrated open-set claim.

### Supporting Information

- strict-refit parent and final-embedding table;
- full public-method ladder, Top-k, ROC and precision-recall figures;
- formula-cluster interval details and statistical notes;
- frozen release and reproducibility checklist;
- full application coverage and matched-coverage comparisons.

### Independent repository or DOI deposit

- the two frozen ZIP archives;
- four large per-query correction ledgers currently excluded from the compact
  bundle;
- complete environment lock and third-party license/citation inventory;
- full figure source files and submission manifest.

## Remaining work that does not require a new experiment

1. Relabel the complete figure suite as CALMS v1.0 and build a figure checksum
   archive.
2. Render one application figure from frozen retained/changed/gained/lost and
   response-ranking outputs.
3. Add journal-formatted Introduction, references, figure callouts, author
   contributions, acknowledgements and conflict statement.
4. Publish the large query ledgers separately if full bootstrap and stratified
   reanalysis from row-level predictions is promised.
5. Add a fully pinned environment lock; the current `numpy>=1.24` and
   `torch>=2.0` requirements are only lower bounds.

## Scientific boundary not fixable by packaging

There is no genuinely untouched external confirmation resource in the current
frozen evidence. The manuscript may claim reproducible superiority on consumed
development resources under a common closed-set axis. It must not claim
field-wide external SOTA, calibrated no-match performance or a raw-MGF-to-all-
evidence workflow.
