# BioAware B40 soft graph completion: executed result and decision

## Outcome

B40 tested whether the B37 catalogue prior could be upgraded into genuine
sample-context evidence by propagating visible seed identities for four hops on
a non-currency Rhea hypergraph projection.  The experiment was outcome-blind
until the fixed-action evaluation, used three component/weighted-degree-decile
matched seed permutations, retained structural zeros, and evaluated only an
outcome-blind primary subgroup in which at least two candidate molecules were
Rhea-mapped.

The result is negative.  B40 does not pass to external reconstruction or to a
context-conditioned representation.

## M0 identifiability preflight

- Rhea projection: 10,099 nodes, 16,705 undirected edges, 13,696 usable
  reactions and 651 connected components.
- 1,738 queries and 10,294 visible-seed contexts were reconstructed across six
  opened sources.
- 833 queries had at least two Rhea-mapped competing candidates; every source
  contributed at least 100 such queries.
- Visible-seed mapping was 59.39%; candidate mapping was 43.59%.
- Real diffusion exceeded the mean matched-seed null in 64.0% of contexts, but
  the median advantage was only 0.000128.
- Diffusion score remained strongly associated with candidate degree
  (Spearman 0.690).  This required candidate degree to remain a mandatory
  negative control.

M0 therefore established computational identifiability, not ranking utility.

## M1 internal fixed-action result

All actions started from the frozen B37 catalogue-topology decision and used
the same DreaMS-only low-margin risk layer.

| Action on 298 mapped-competition queries | Delta vs B37 topology | Corrected / introduced | Formula-cluster 95% CI |
|---|---:|---:|---:|
| Real four-hop diffusion | -3.69 pp | 23 / 34 | [-9.91, +2.95] pp |
| Real minus matched-seed-null mean | -8.05 pp | 19 / 43 | [-13.82, -2.12] pp |
| Candidate Rhea degree only (negative control) | +5.03 pp | 33 / 18 | [-1.95, +12.54] pp |

The real graph action did not beat the frozen topology prior or all structural
nulls.  Subtracting the matched-seed null made the result significantly worse,
not better.  The degree-only control had a positive point estimate but failed
the cluster interval, risk-net and source-consistency gates.

## M2 context-stratified fixed-action result

The identical frozen real-diffusion action was evaluated separately, without
refitting, in the three context semantics used by the six opened sources.

| Context stratum | Primary queries | Delta vs B37 topology | Delta vs DreaMS | Corrected / introduced |
|---|---:|---:|---:|---:|
| Synthetic rotation | 298 | -3.69 pp | +2.35 pp | 23 / 34 |
| Sample-local leave-one-seed-out (ST001154) | 128 | -13.28 pp | 0.00 pp | 14 / 31 |
| Hidden standards (KGMN-200STD) | 102 | -7.84 pp | -4.90 pp | 3 / 11 |

The result is not rescued by the most biologically relevant available
sample-local stratum.  In ST001154 the graph action exactly erased the B37
topology improvement relative to DreaMS and caused more than twice as many
harms as corrections.

## Scientific interpretation

1. The B37 gain is not evidence that visible sample seeds point toward the
   correct candidate through local Rhea paths.
2. A real seed set is not sufficient merely because it produces nonzero graph
   scores.  In the current panels, seed-to-candidate proximity often points to
   an incorrect member of the candidate set.
3. Candidate network membership and degree are database/catalogue attributes.
   They can calibrate candidate plausibility, but cannot be renamed reaction
   propagation or injected into a candidate-independent clean-spectrum encoder.
4. The failure is not an optimisation failure: B40-M1/M2 contain no learned
   graph model.  More hops, different decay, or a larger neural network would
   tune around a failed fixed-action mechanism rather than repair it.

## Frozen decision

- Stop the current Rhea diffusion/context-representation branch.
- Do not train a context adapter or shared encoder from B40 targets.
- Preserve B30/B37 as an opened-development catalogue-prior engineering module.
- Next test whether catalogue membership/degree is portable across independent
  knowledge resources (Rhea, strict KEGG reaction network, and MetDNA2 EMRN).
  This audit must keep database prior and sample-context evidence separate.
- A future reaction-context branch requires a prospective sample cohort in
  which query truths were not used to create the seed table, plus an action
  that beats topology and structural nulls before representation learning.

## Claim boundary

These are opened-data mechanism audits.  They are not blind SOTA evidence,
prospective unknown annotation, reaction causality, biological mechanism, or a
shared-embedding improvement.
