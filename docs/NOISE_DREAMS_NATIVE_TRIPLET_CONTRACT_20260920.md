# Noise fine-tuning reset: native DreaMS with Noise-only triplet content

> Superseded on 2026-09-22 by
> `NOISE_ACTION_HARD_NATIVE_CHEMAWARE_INTERFACE_20260922.md`.  The action-as-model-input
> design below is retained only as a failure record; it must not be submitted.

Date: 2026-09-20

## Decision

The training system is the published repository implementation.  Noise may
change only which spectra occupy the anchor, positive and negative roles.
There is no Noise loss, injector, teacher embedding, safety stream, optimizer
arbitration, gradient surgery or custom checkpoint format.

The production runtime is frozen to:

- `dreams.utils.data.ContrastiveSpectraDataset`, dynamically sampling one
  positive and one negative from precomputed membership lists;
- `dreams.models.heads.heads.ContrastiveHead`, including its biased
  1024-to-1024 linear projection;
- unmodified `SpectrumPreprocessor(DataFormatA, n_highest_peaks=100,
  prec_intens=1.1, precision=32)`, without subclassing or output repair;
- the native cosine hinge triplet loss with margin `0.1`;
- the native single Adam optimizer, learning rate `5e-6`, weight decay zero;
- batch size four, FP32, complete-backbone unfreeze at epoch zero and one
  registered complete-coverage training cycle (`max_epochs=1`);
- initialization from the complete official fine-tuned
  `embedding_model.ckpt` and native Lightning `.ckpt` output.

The published command's `max_epochs=301` is only a ceiling, and the published
checkpoint records epoch 17 rather than a completed 301-epoch trajectory.
Epoch count is not a transferable training dose when the custom triplet ledger
is much larger. Copying that ceiling here would create 5,213,019 optimizer
steps per arm. The registered Noise schedule is therefore one ordinary native
shuffled pass. Every action event appears exactly once. Only zero to three
deterministic action-free clean fillers may be repeated to make the frozen
ledger divisible by batch size four; exact event and step counts are derived
from that artifact rather than hard-coded. There is no query-balanced recycling or custom
batch ordering. Model, loss, optimizer, precision, batch size, shuffle,
drop-last and checkpoint format remain native DreaMS.

## Qualified Noise actions

The only action input is the formal, current-corrected-graph seven-source
ledger:

1. `N_mature`
2. `P_guided_original`
3. `E10B`
4. `E11`
5. `E12B`
6. `A4_exact`
7. `V4_gradient_path`

All 32,114 numerically robust, strict Top-1 corrective rows are retained in a
provenance ledger. Exact aliases of `(query, positive, negative, action tensor)`
collapse to one native optimizer unit so rediscovery by multiple sources cannot
become additional dose; every alias remains mapped to that canonical unit. No
one-best-per-query or non-identical-action compression is permitted. Historical held matrices, local
development ledgers, E13 training outputs, outcome/oracle tables and old R0
row indices are forbidden.

For action row `a`, the exact native relations are:

```text
clean boundary  = clean_query -> a.action_positive_row
                               / a.action_hard_negative_row
action boundary = action_spectra[a.action_tensor_index]
                  -> a.action_positive_row / a.action_hard_negative_row
```

The exact rows are validated against the same candidate graph.  The positive
must belong to a true molecule and the negative to a false molecule.
`reference_rows` is action-construction support and is never substituted for
the exact positive.

Some qualified attenuation actions retain real positive-m/z tokens while all
fragment intensities are zero. The official relative-intensity preprocessor
cannot represent these actions because it would divide by a zero base peak.
No sentinel, epsilon, NaN repair or preprocessor subclass is permitted. If
either targeted or matched-shuffled content is unrepresentable, both arms use
one exact clean-boundary-only relation. Such units remain in provenance but are
excluded from materialized-action coverage and the targeted-versus-shuffled
action-effect denominator. Clean-only units are canonicalized again by their
actual `(query, positive, negative)` triplet, so invisible action multiplicity
cannot multiply dose. All representable targeted and shuffled actions must
round-trip token-for-token through the unmodified native preprocessor.

## Large, diverse triplet construction

Every effective unit materializes exactly one native singleton triplet. A unit
whose targeted and shuffled views are both natively representable uses the
action spectrum as anchor; an unrepresentable unit uses the measured clean
query as anchor:

```text
clean boundary: clean query -> exact positive -> exact hard negative
action boundary: action spectrum -> exact positive -> exact hard negative
```

The action-boundary triplet teaches the effective action view directly on its
verified relation. The clean-only fallback preserves that relation without
fabricating an encoder input. The independent broad clean-query stream supplies
ordinary measured anchors, so clean, action, true-positive and hard-negative
roles train through native triplets without duplicating the same clean boundary
once per action or inventing a four-input loss. Independently sampling from enlarged positive and negative pools would
turn one verified boundary into as many as 1,024 mostly unverified pairings, so
all action memberships remain singleton. Every triplet is executed by the
unmodified native dataset and native hinge loss.

The native dataset table is materialized at event level, not merely at unique
spectrum level. Multiple clean-boundary events may reuse the same immutable
clean spectrum object, but each event owns an independent DataFrame anchor row
and its own exact positive/negative membership. This is required because the
native `ContrastiveSpectraDataset` reads memberships from the anchor row;
storing them once on a shared spectrum-registry row would make later actions
overwrite earlier actions and silently erase both the broad clean pool and
most exact action boundaries.

The optimization set also retains a clean anchor for every eligible outer-train
query.  Clean positives are measured same-identity spectra; clean negatives are
frozen official hard candidates plus every action-specific hard-negative
molecule.  Qualified action views are not silently inserted into the clean
positive pool.  This broad clean base prevents the
canonical action views from becoming a narrow 3,482-query-only training set while
remaining an ordinary identity-supervised triplet dataset.

The materialized dataset contains every effective native relation and preserves
all qualified action IDs in the alias ledger, but action-generation multiplicity
is not optimizer dose. The one-pass native loader gives every effective unit
exactly one triplet—an action-anchor triplet when representable, otherwise a
clean-anchor fallback—while the alias ledger covers all 32,114 qualified rows.
The ordinary native loader shuffles all events and drops only complete batches;
zero to three action-free clean fillers prevent any action event from falling into an
incomplete final batch. There is no requirement that a batch contain distinct
queries because native DreaMS computes each triplet independently and does not
use other batch members as negatives.

A formula-disjoint validation split is selected only from outer-train formulas
with no qualified action. Consequently every qualified action remains recorded
in provenance and every unique semantic unit remains in optimization;
validation remains formula-disjoint and the outer fold is not loaded as an
anchor.

The build must report and gate:

- exact 32,114 qualified-provenance-row coverage and all seven sources;
- a complete qualified-action-to-canonical-unit mapping, exactly one native
  triplet per effective unit, and zero duplicate effective triplets after collapse;
- at least 50,000 clean optimization anchors and 4,000 train formulas;
- at least 1,000 clean validation anchors;
- exact singleton action boundaries and clean positive/negative pool sizes;
- exact member coverage across the one registered complete-coverage cycle;
- identity-label correctness and exact candidate membership;
- formula-disjoint train/validation roles;
- a matched shuffled arm that changes only the action spectrum within the
  same supervision/source/family/exact-recipe stratum. For an exact alias
  group, one outcome-blind hash-selected provenance representative supplies
  that control stratum to both arms.

Large combination count and dynamic sampling reduce repetition; they do not
prove absence of overfitting.  That claim requires independent held retrieval,
near-subset and formula-cluster results.

## Entrypoints

- Triplet build: `tasks/build_noise_dreams_native_triplets.py`
- Native training: `tasks/train_noise_dreams_native.py`
- Local synthetic/runtime contracts: `tasks/test_noise_dreams_native.py`
- Slurm: `tasks/run_noise_dreams_native_2gpu.sbatch`

The Slurm job requests two GPUs, runs targeted and matched-shuffled native
continuations concurrently, and does not specify memory manually.  It must not
be submitted until its compute-node preflight sees the formal seven-source
ledger and both native official checkpoints.

Before optimization, every canonical exact action pair is replayed under the
actual official fine-tuned initialization (the action selector itself used the
mature E8 geometry). The replay gates are weighted by each canonical unit's
exact unit exposure in the frozen one-pass native schedule; they are not
ordinary row means. Targeted-versus-clean and targeted-versus-shuffled margins
are diagnostics rather than admissibility gates: a low current action margin is
exactly what a correct native triplet is meant to repair. The only signal gate
requires at least one real representable action to activate the native hinge
and the allocated-GPU role audit to reach the shared head and backbone. Per-source
active fractions are reported, not converted into arbitrary stop thresholds.
All seven qualified
sources are independently required in
the 32,114-row provenance ledger; a cross-source alias is not falsely credited
with the representative source's matched null. A real materialized four-example minibatch must also backpropagate through the
expected clean/action/positive/negative roles, projection head and backbone in
the same two sequential native steps used by production.  After both arms
finish, the job evaluates the
same 18,333-query formula-held corrected graph and writes Recall@1/2/3/5/10/20,
MRR, mean/median rank, macro and micro AUROC/AUPRC, margin, signed Top1-Top2,
corrected/introduced/risk-net, near-subset metrics, formula-cluster paired CIs,
and all-adduct plus `[M+H]+` MassSpecGym 10-ppm pooled AUROC/AUPRC.  Promotion
requires at least +4 pp Recall@1 versus both official and the registered
mature-E8 encoder, at least +0.5 pp for the targeted action view versus its
matched shuffled action view, positive paired CIs, positive risk nets and no
registered metric regression against official, shuffled and mature E8.  The
shared clean-boundary triplet is part of the Noise-mined relation; the separate
0.5-pp gate prevents that common supervision from being misreported as evidence
that the query-matched action spectrum itself helped.

The corrected graph's historical `official_embeddings.npz` is an action-mining
and candidate-membership provenance artifact produced by an older lightweight
inference path. It is never mixed with embeddings from the native
`ContrastiveHead`: before replay, the complete 87,848-row registry is re-encoded
by the exact native `embedding_model.ckpt` plus native preprocessor. Clean,
action, positive and negative margins therefore use one checkpoint geometry,
and the same native cache is the official comparator in final evaluation. Any
legacy-versus-native discrepancy is reported only as a labelled diagnostic and
is not a reason to substitute legacy positive/negative embeddings into a native
triplet.

The one-pass schedule controls multiplicity at the effective-triplet level.
Every broad clean query appears once, every representable action contributes
its action boundary once, and every unrepresentable action contributes its
deduplicated clean fallback once. Exact aliases discovered by multiple
sources do not multiply dose. Queries with several genuinely different actions
therefore contribute several different relations, but low-multiplicity queries
are never recycled merely to match the maximum. The minimal clean fillers are
action-free and are reported explicitly.
