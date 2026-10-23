# Contrast Atlas open-response qualification result — 2026-10-23

## Decision

The 12-study open-response collection is a valid development resource for
testing preregistered **unsigned relevance ranking and abstention**.  It is not
yet qualified as the primary validation of signed biological-response
retrieval, open-world error control, or unresolved-entity discovery.

This is not a software-readiness judgment.  It follows from the biological
truth and opportunity structure of the collection itself.

## Fixed data boundary

The audited collection is
`contrast_atlas_open_response_collection_v8_plate3_covid_20261022`.

| Quantity | Denominator |
|---|---:|
| Studies | 12 |
| Source families | 12 |
| Biological contrasts | 27 |
| Response rows | 91,295 |
| Unique response entities | 38,247 |
| Source-disjoint directional query-candidate contrast pairs | 648 |

All inputs are development/consumed.

## The decisive truth audit

Only five canonical relations, represented by ten directional registry rows,
are preregistered.  They form two disconnected biological components:

1. high-fat-diet response: one mouse-urine contrast and female/male
   mouse-plasma contrasts;
2. IBD-versus-control response: pediatric VEO-IBD, adult mixed IBD and adult
   Crohn's active-or-remission contrasts.

Consequently:

| Registry property | Result |
|---|---:|
| Query contrasts with a registered target | 6/27 |
| Query contrasts without a registered target | 21/27 |
| Registered directional pairs | 10/648 |
| Unlabeled directional pairs | 638/648 |
| Confirmed negative pairs | 0 |
| Registered rows claiming signed truth | 0/10 |
| Registered rows with unspecified signed direction | 10/10 |

The 638 unregistered pairs are unknown relations, not negative truth.  A
standard accuracy, AUROC or FDR analysis that labels them negative would be
scientifically invalid.  The registry can test whether a method ranks the
prespecified related condition family highly; it cannot by itself establish
that metabolite-level directions should agree.

## Strict common-identity opportunities

Every study was repartitioned under the same frozen GNPS Gold/Silver policy
before measuring known-only overlap.  Results for the five canonical
relations are:

| Registered relation | Shared strict InChIKeys | Entity pairs | Both `|z| >= 1.96` | Same-sign pairs, descriptive |
|---|---:|---:|---:|---:|
| Mouse HFD vs female HFD | 0 | 0 | 0 | 0 |
| Mouse HFD vs male HFD | 0 | 0 | 0 | 0 |
| Pediatric VEO-IBD vs adult mixed IBD | 12 | 53 | 1 | 34 |
| Adult Crohn's vs adult mixed IBD | 1 | 3 | 0 | 2 |
| Adult Crohn's vs pediatric VEO-IBD | 18 | 28 | 1 | 15 |

In total, three of five canonical relations have any strict identity overlap;
two have none.  The 31 relation-specific InChIKeys produce 84 entity pairs,
but only two pairs are individually strong in both studies.  Entity-pair rows
are not independent when several features share an InChIKey, so `51/84`
same-sign pairs is descriptive and is not a binomial validation result.

The absence of strict HFD identity overlap explains why an unresolved chemical
relation layer could add coverage.  It does not prove that such added coverage
is correct.

## A second corrected boundary: known-only identity changed materially

The response collection originally inherited heterogeneous, source-specific
identity labels.  Reapplying one common identity policy changed 1,191 of
38,247 entity labels:

| Identity transition | Entities |
|---|---:|
| Original source-specific library-supported | 1,249 |
| Common-policy library-supported | 752 |
| Library-supported under both | 405 |
| Old library-supported, downgraded by common policy | 844 |
| Old unresolved, promoted by common policy | 347 |

Therefore every future `known_only` comparison must use the common frozen
identity partitions.  Mixing old per-source labels would change the evaluated
population and make the comparison nonreproducible.

## Unresolved-relation readiness

Official DreaMS embedding shards are complete for ten of the twelve studies.
The missing studies are `MSV000083077` and `MSV000097610`.  Five studies
participate in registered relations; four of those five are complete, with
`MSV000097610` missing.

Thus the unresolved chemical-relation track cannot yet be evaluated across the
complete registered IBD component.  The existing array job can be resumed for
indices 2 and 5; the ten completed shards are reusable and must not be
re-encoded.

## What is authorized next

1. Complete only the two missing official-embedding shards.
2. Run the frozen unsigned relevance retrieval on the current registry,
   reporting rank, coverage and abstention.  Do not call unregistered pairs
   negatives.
3. Compare `known_only`, all eligible entities and unresolved-only evidence
   under the harmonized identity policy.
4. Treat HFD and IBD as two biological relation clusters, not ten independent
   validations.
5. Keep this experiment as development evidence even if ranking improves.

In parallel, the primary biological benchmark must acquire additional
independent replicated response families or intervention truth.  Eligibility
requires a defensible expected relation, explicit direction semantics when a
signed claim is made, source-family separation, a no-match challenge, and
enough independent relation clusters to estimate uncertainty.  Disease-name
agreement alone is not signed truth.

## IBD application role

The corrected 509-edge IBD panel remains the algorithm/application handoff
panel.  Once B and the integrated score bundle arrive on the identical axis,
the biology side will report score exchanges separately for:

- 156 `[M+H]+`-domain candidate edges across 11 candidate-bearing queries,
  including three public-response opportunities;
- 353 cross-adduct candidate edges across 34 candidate-bearing queries,
  including seven public-response opportunities.

The IBD panel will then be frozen regardless of the result.  It is too small to
serve as the primary biological validation.

## Reproducibility

Implementation:
`tasks/audit_contrast_atlas_open_response_qualification.py`

Frozen output:
`data/validation/contrast_atlas_open_response_qualification_v2_20261023`

Unit tests:

```text
python -m pytest -q tasks/test_audit_contrast_atlas_open_response_qualification.py
2 passed
```

Report status:
`DEVELOPMENT_CONSUMED_QUALIFICATION_COMPLETE_PRIMARY_TRUTH_INSUFFICIENT`.

This result prevents a large but weak collection from being mistaken for a
validated biological-discovery benchmark.  It also identifies the cheapest
valid continuation: finish two missing embeddings, run the already frozen
unsigned task once, and direct new acquisition toward genuinely independent
response truth rather than additional unlabelled spectra.
