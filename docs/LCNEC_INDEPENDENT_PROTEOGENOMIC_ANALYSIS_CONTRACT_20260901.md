# LCNEC independent proteogenomic fixed-panel analysis contract

Frozen on 2026-09-01 after article-text context review but before opening the patient-level protein-expression matrix.
This contract operationalizes the already frozen 22-protein panel without changing its genes, strata, or claim boundary.

## Primary analysis

- Population: pure LCNEC patients with paired tumor and normal-adjacent proteome values.
- Unit: patient pair; no tumor and NAT sample is treated as independent.
- Effect: tumor minus NAT in the supplied log2 protein-abundance scale.
- Test: two-sided Wilcoxon signed-rank test for each available fixed-panel protein.
- Multiplicity: Benjamini-Hochberg across all 22 frozen proteins; missing proteins remain in the audit but do not receive substitute genes.
- Stable direction requires all of the following: nonzero mean and median effects with the same sign, unchanged mean sign in every leave-one-patient-out recomputation, and at least 60% of nonzero patient differences sharing that sign.
- Protein gate: measured in at least 20 pure-LCNEC pairs, stable direction, and fixed-panel BH q below 0.10.
- Axis gate: at least two proteins from the same frozen axis pass the protein gate with the same tumor-minus-NAT direction.

## Secondary analyses

- Combined LCNEC versus pure LCNEC uses tumor samples only and is reported separately from the primary paired endpoint.
- KEAP1-mutant versus KEAP1-wild is run only if an author-curated binary mutation label is present or exactly reproducible from the deposited analysis object. Raw mutation-table variants will not be silently relabeled to manufacture the paper's reported percentage.
- Secondary results cannot rescue a failed primary axis and receive their own multiplicity correction.

## Fixed boundaries

- Article-text mentions of IDO1, G6PD, PGD, TKT, and TALDO1 are context priors, not patient-level validation.
- Protein abundance is not enzyme activity or flux.
- This independent cohort does not validate metabolite identity or metabolite abundance.
- No genes, thresholds, strata, or directions may be replaced after the protein matrix is opened.
