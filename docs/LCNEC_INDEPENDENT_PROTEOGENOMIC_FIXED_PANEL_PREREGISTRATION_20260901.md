# LCNEC independent proteogenomic fixed-panel preregistration

Frozen: 2026-09-01, before inspecting patient-level abundance outcomes in the independent 107-patient proteogenomic cohort (PMID 42585338; PMCID PMC13464647).

## Purpose

Test whether an independent LCNEC proteogenomic cohort supplies non-circular protein-level context for the frozen metabolite hypotheses recovered from the 34-pair metabolomics atlas. This is not metabolite-level replication and cannot validate metabolite identity, abundance, flux or enzyme activity.

## Frozen panel

### Quinolinate / de novo NAD context

- `TDO2`, `IDO1`, `KYNU`, `HAAO`, `QPRT`, `NMNAT1`, `NMNAT2`, `NMNAT3`, `NADSYN1`

The most local discriminator is `QPRT`, which consumes quinolinate. Protein abundance alone cannot determine reaction flux. Upstream induction and downstream limitation are considered different compatible models, not interchangeable conclusions.

### ADP-ribose turnover context

- `PARP1`, `PARP2`, `CD38`, `ENPP1`, `NUDT5`

`NUDT5` is the most local downstream handling protein. Increased ADP-ribose-family abundance does not by itself imply increased PARP/CD38 activity or decreased NUDT5 activity.

### Ascorbate / redox context

- `SLC23A1`, `SLC23A2`, `GSR`, `TXNRD1`, `G6PD`, `PGD`, `TKT`, `TALDO1`

The pentose-phosphate proteins are included because the independent study pre-reported KEAP1-associated metabolic reprogramming. They are context for reducing-equivalent supply, not proof that ascorbate caused or responded to that program.

## Frozen strata and endpoints

1. Primary stratum: pure LCNEC paired tumor versus normal-adjacent tissue.
2. Secondary stratum: combined LCNEC versus pure LCNEC, analyzed separately.
3. Secondary modifier: KEAP1-mutant versus KEAP1-wild where sample-level genotype is available.
4. Primary protein endpoint: paired tumor-minus-NAT abundance effect for every available fixed-panel protein.
5. Multiple testing: Benjamini-Hochberg correction within the complete fixed panel; raw P values remain visible.
6. Missing proteins remain reported as missing. No replacement genes may be promoted after outcomes are opened.
7. Pathway summaries are descriptive unless at least two measured proteins from that frozen axis move coherently and the result remains visible after multiplicity correction.

## Decision rules

- A protein-level context result is positive only when the fixed-panel protein is measured, its paired direction is stable, and its reported or recomputed FDR is below 0.10.
- A pathway-context result is positive only when at least two fixed proteins in the same axis satisfy the protein rule with a coherent interpretation.
- A null or discordant result is retained and does not invalidate the metabolite abundance observation; it limits mechanism interpretation.
- Pure and combined LCNEC are never pooled to manufacture significance.

## Forbidden claims

- independent metabolite abundance replication;
- authentic-standard or MSI Level-1 identity;
- NAD, ADP-ribose or antioxidant flux;
- QPRT, NUDT5, PARP, CD38 or pentose-phosphate enzyme activity;
- causal tumor dependency or therapeutic vulnerability;
- patient-matched integration between the two public studies.

## Public source

- Article: https://pubmed.ncbi.nlm.nih.gov/42585338/
- Processed dataset: https://zenodo.org/records/18492443
- Code/data bundle: https://zenodo.org/records/20922299

