# Noise direct-v3 loss-aware fine-tuning specification

Date: 2026-09-07

Status: implementation-complete candidate pending GPU numerical smoke and full held evaluation. This document does not claim a 4--5 pp gain before the registered result exists.

## 1. Scientific objective

Keep the mature E8 shared encoder initialization and directly fine-tune that encoder with the already discovered N, original positive-guided intensity, E10B, E11, E12B and A4 spectral actions. There is no teacher embedding target, representation distillation, reranker, P2b or P3 consumption. Inference remains one clean spectrum to one shared embedding.

The corrected graph contains 83,619 total queries. Fold 0 must keep 18,333 queries held, leaving 65,286 unique legal training queries; claiming 100,000 unique training queries would require leakage or a different dataset. Four complete epochs provide 261,144 clean-query exposures while every epoch still covers each legal training query exactly once.

The required outcome is at least +4.0 percentage points Recall@1 versus the exact E8 initialization on the same outer-formula-held graph, with broad metric improvement and strict paired formula-cluster evidence. This is a promotion criterion, not a guaranteed outcome.

## 2. What was actually wrong in v2

The failure was not that the action bank lacked useful actions. The information funnel erased or misused them at four separate boundaries.

1. `paired_margin_only` disabled the real action-ranking, counterfactual and safety branches. The action spectra were forwarded, but only a small set of scalar candidate-margin targets affected the clean encoder.
2. Harmful action identity, family, dose and tensor contents were never consumed. On the formal full outer-training graph, unioning harmful query IDs into an already complete clean-query set was a no-op.
3. Robustness-only actions were left in the audit ledger and never entered training.
4. Corrective batches were consumed first, followed by a long protect-only optimizer tail. The development-ledger audit gives 18 action batches versus 16,322 full-graph protect batches, or only 0.1103% action-active v2 steps. That percentage is a development-ledger diagnostic, not a formal-ledger forecast.
5. The old formal P router tried to retain 65,286 x 60 x 2 action/control embeddings at once. At 1,024 float32 dimensions that allocation alone is about 32 GB, before approximately 6.3 GB of action/control spectra, per-query candidate-vector copies, pandas tables and Python objects. A4 used the same monolithic pattern. Several hours without reaching training is therefore compatible with route construction, host-memory pressure or allocation failure; it is not evidence of a long-running fine-tuning epoch. The complete 66-cell inventory would make a monolithic implementation still larger, which is why the bounded chunk/frontier implementation is mandatory.
6. The query scopes named `official_errors` and `official_correct` were selected from official-DreaMS ranks although the actual initialization is E8. Those scopes are valid historical diagnostics but are the wrong boundary definition for an E8 successor and can omit E8-specific errors.
7. The first v3 scheduler implementation contained its own residual compression bug. It could place roughly nine full-graph protect microbatches behind one corrective optimizer step; the helper already knew how to recycle corrective batches, but the trainer neither passed the cap nor consumed the recycle scale. Thus it removed the protect-only tail while still collapsing most E8-style optimizer opportunities.
8. The first v3 candidate boundary still selected every loss edge from the clean query Top-k. An action that changed the hardest candidate molecule could therefore be forwarded while the new action-specific competitor was absent from the objective. This contradicted the earlier direct-boundary contract that candidate switches must be retained.
9. The calibration report claimed a 32-batch formula-diverse estimate, but the implementation ordered examples by identity only and did not include robust/harmful batch counts in the available-batch calculation. A sparse auxiliary panel could therefore be recycled during calibration and masquerade as 32 independent strata.
10. N, P and A4 controls had been reduced to one undifferentiated `control_margin`. N uses same-role matched-random paths, A4 uses strict/relaxed matched peaks (or a clean fallback), while P uses a wrong-identity direction. The common conservative minimum was mathematically safe because action-minus-clean remained a hard ceiling, but the report could not show whether signal was lost at the clean-improvement gate or at the source-specific control gate.
11. Equalizing N/P/A4 only inside each query did not equalize them over an epoch. If P produced corrective actions on many more queries, P-only queries could still dominate the outer query mean and recreate the historical P-over-N failure even though no individual query exceeded its action cap.

A tempting fifth change was explicitly rejected during implementation review: detaching all identity references would concentrate a local gradient, but the mature E8 factor experiment already showed that freezing official reference anchors loses about 0.22 pp versus the shared geometry. Direct-v3 therefore preserves shared query/reference rank updates. It repairs signal loss by activating the real action branch, balancing its optimizer duty cycle and calibrating/measuring its retained update, not by discarding a demonstrated E8 requirement.

Evidence report: `data/validation/noise_corrected_direct_v3_plan_audit_dev_v11_20260907/report.json`.

The formal v3 route producer fixes both scaling errors without changing the action definitions. It first evaluates the outer-training queries in current-E8 geometry, retains every E8 error, and adds an identity-diverse lowest-margin correct panel for harmful/robust supervision. P evaluates all 66 unique registered cells on that panel; A4 evaluates every eligible Top-50 proposal and four registered doses on the corresponding A4 panel. Neither selector reads held formulas or historical action outcomes. All 65,286 legal queries still enter the clean/protect objective once per epoch; the action panel only targets the expensive counterfactual forwards. N, P and A4 all encode query chunks with bounded resident arrays and retain a per-mechanism selector frontier that is mathematically sufficient for the later global N/P/A4 cap. This also removes the N router's former roughly 1.7-million-view monolithic encode.

The two-GPU producer balances wall-clock work as GPU0 `N -> A4` and GPU1 `P66`. The previous layout serialized P and A4 on the same device while GPU0 could become idle after N. The new layout changes no data, route, seed or action definition and still requests exactly two GPUs with no manual Slurm memory request.

The nine-arm trainer likewise uses two persistent per-GPU worker queues instead of a barrier after every pair of arms. Each worker receives three expensive routed/shuffled arms, and the three cheaper clean controls are split between them. This removes clean-versus-action idle barriers without changing any arm, seed, schedule or frozen input.

P66 construction also caches reference context without caching or approximating an action result. Per query it builds exactly ten reference profiles (five reference policies by positive/wrong-identity direction) and at most thirty recurrent-missing-peak contexts (five policies by three recurrence parameter sets by two directions), then materializes every one of the 66 action cells and all 66 paired direction controls from those exact contexts. The old loop recomputed the same profile and recurrence extraction for each recipe. Before the encoder, only action tensors that are byte-for-byte identical in dtype, shape and contents are encoded once; a lossless inverse index restores a score for every recipe/direction row, while the original per-action tensors remain available for frontier export. The route report records expected cache counts, raw versus unique encoder inputs and the exact inverse-index contract, so these wall-clock optimizations cannot silently skip or approximate an action. It also hashes the full action registry, E10 action executor, E11 reference selector, positive-guided profile builder, recurrent missing-peak builder, base router and action-panel selector; because the downstream combined ledger hashes the route report, any implementation change that determines a P tensor or selection changes formal provenance.

Formal N, P66 and A4 routing now fail closed on the registered scientific configuration. The mature-N builder is covered by the same rule. Each route report writes the effective thresholds, query-scope/boundary sampling settings where applicable, frontier caps, peak count and AMP mode, and marks the contract verified. Chunk and encoder batch sizes remain engineering controls, but every argument that changes which action is constructed or selected is explicit in the two-GPU SBATCH. The combined-ledger preflight rejects a route whose formal configuration contract is absent and requires every N/P/A4 route to carry the same registered formula-fold seed. The trainer independently freezes fold 0, formula seed, inner-holdout setting, evaluation batch size and bootstrap count, closing the route/train split-drift path that could otherwise leak a route-training formula into evaluation under a different split.

The trainer also registers all five development truncation controls as zero: corrective, harmful/risk, robust, clean-protection and outer-held evaluation query limits. A command that accidentally carries a smoke-test query cap can therefore no longer emit a formal report or encoder.

## 3. Frozen three-semantics ledger

The v3 ledger keeps three optimizer semantics separate:

| route | selected development rows | v3 use |
|---|---:|---|
| corrective | 848 | bounded conservative clean-margin transfer plus real action-view identity rank |
| harmful | 424 | action-content-specific damage weighting of the clean boundary; harmful view detached |
| robust | 185 | real action-view rank and clean-relative floor; no corrective reward |
| uncertain | 0 | audit only |

The original development-v3 ledger contained 1,457 actions over 94 queries and exposed five sources: N_mature, E10B, E11, E12B and A4_exact. A subsequent inventory audit found that the 60-cell late-P registry did not actually contain the entire earlier mature 12-cell P-intensity matrix: it omitted all four `prevalence_attenuation` doses and `matched_intensity_transport` at doses 0.25/0.75. The other six original intensity cells were already duplicated by E10B. The corrected registry adds only these six non-duplicates under their honest `P_guided_original` provenance, producing 66 unique P cells and six total action sources. These cells are not promoted from historical held outcomes: they are re-evaluated under current E8 on outer-train and receive corrective weight only through the same clean/control routing gate.

A flat lexicographic source/family round-robin could exhaust a per-query cap on numerous P families before reaching N. The corrected selector therefore round-robins mechanism blocks (N/P/A4) first. Inside each mechanism it equalizes source exposure first, then family exposure, and only then uses the current outer-train route score and stable action ID; this prevents late-sorting P sources from being systematically excluded while retaining the strongest recipe within a source/family leaf. The loss uses the stricter hierarchy query -> mechanism block (N/P/A4) -> action family -> source within that family -> action. Thus `recurrent_union_mix` being rediscovered in E10B, E11 and E12B does not receive three family votes, while the distinct reference policies and recipes from every source are still exposed and averaged inside that family. Action or experiment-generation multiplicity cannot become optimizer dose. The formal ledger is rebuilt with this corrected selector inside Slurm; historical development-v3 counts are diagnostic only.

Control semantics are now explicit action-ledger data rather than an inferred comment. N writes `matched_neutral` for a complete matched path and `clean_fallback` only when no complete path exists; A4 writes `matched_neutral` for strict/relaxed peak matches and `clean_fallback` otherwise; `P_guided_original`/E10B/E11/E12B must write `wrong_identity_direction`. The ledger rejects any source/semantic mismatch before it materializes training tensors and reports both semantic totals and source-by-semantic totals.

Artifact: `data/validation/noise_corrected_routed_npa4_ledger_e8_dev_v3_20260906`.

The corrected local selector replay is stored separately at `data/validation/noise_corrected_routed_npa4_ledger_e8_dev_v4_20260907`; it keeps the same 1,457-row bounded dose but changes two harmful selections so the mechanism hierarchy is respected. The bounded-memory v5 rebuild at `data/validation/noise_corrected_routed_npa4_ledger_e8_dev_v5_20260907` produces exactly the same training table, action IDs, action spectra and control spectra as v4 while loading spectra only after selection. No encoder was trained from any development ledger.

The v5 development ledger exposes why within-query equalization was insufficient. Its corrective panel contains 812 P rows on 58 queries, 28 N rows on 8 queries and 8 A4 rows on 4 queries. Harmful query coverage is P/N/A4 = 61/2/7 and robust coverage is 20/1/3. Thus the per-query objective would still be overwhelmingly P-weighted even though a mixed query itself was balanced. Under the new identity-aware inverse-incidence calculation, the illustrative development corrective coefficients are P 0.40, N 2.75 and A4 5.50, producing equal cumulative mechanism mass; the formal coefficients are recomputed only from the formal outer-training ledger.

A second schedule audit found that the first implementation contradicted its own sparse-auxiliary rule. It multiplied each robust/harmful batch by `optimizer_steps / auxiliary_batches`, up to the branch cap of 16, despite claiming those panels were query-local rather than globally equalized. That creates a large one-step safety gradient which can erase corrective signal through projection or clipping, and an even sparser formal panel would fail before training. The repaired scheduler consumes every robust/harmful query exactly once per epoch and spreads those batches over the complete action-step timeline without replication. A further implementation audit caught that merely permuting batch order still packed all sparse work into the first consecutive steps. The registered scheduler now alternates robust and harmful batches, assigns the combined sequence to uniformly spaced epoch positions, and guarantees zero robust/harmful overlap whenever their combined batch count fits within the optimizer-step count. Sparse panels remain at unit scale. A subsequent dense-panel audit adds the complementary boundary: if several auxiliary batches must share optimizer steps, each panel is scaled so its cumulative mass is no greater than one calibrated batch per step, with at most four combined auxiliary microbatches in memory. Only corrective supervision uses full-dose recycling. Calibration therefore sets the local robust/harmful ratio on steps where the constraint is present; it no longer fabricates sparse population mass, a front-loaded shock, or a dense batch-count multiplier.

The same audit removed two batch-boundary distortions. A partial final batch no longer receives the same optimizer mass as a full registered batch: its mean loss is multiplied by `actual_queries / registered_batch_queries` in both calibration and training. Protective microbatches no longer use `1 / microbatches_in_this_step`, which could make a short remainder step several times stronger than every other step. Every protective microbatch instead receives the same frozen epoch coefficient `optimizer_steps / total_protective_microbatches`, followed by the same partial-batch correction. Finally, the corrective action-to-risk scale is calculated from the dense corrective branches only (`transfer + payload + consistency`). Sparse robust/harmful constraints remain in worst-case pre-clip diagnostics but cannot depress the corrective scale by pretending to occur at every step.

A later gradient-order audit found one remaining unmeasured cancellation point: the implementation summed corrective, robust and harmful gradients before the reported full-graph risk projection. Robust/harmful constraints could therefore oppose and erase the main corrective direction before the old retention numerator was even defined. Direct-v3 now performs two explicit projections. First, separately inside the projection head and unfrozen backbone group, it removes only the component of the combined auxiliary gradient that opposes the dense corrective gradient, then adds the retained auxiliary component to the unchanged corrective gradient. Second, it projects that combined action gradient against the full-graph protection gradient. Calibration and every training step record the inner auxiliary retention and verify globally and per parameter group that the dot product with the original corrective direction never decreases. This is still one shared encoder and direct identity/boundary supervision; no teacher embedding or distilled target is introduced.

For formal-scale sources, P and A4 may discard non-frontier tensors only after all actions in a query have been executed and routed. For each mechanism and supervision kind, the retained prefix uses the same hierarchical ordering and is at least as wide as the global 16/8/8 caps. A composition test proves that global selection from the union of these frontiers returns exactly the action IDs selected from the unpruned union. Full route counts and per-query summaries remain auditable; the builder rejects a source whose declared frontier is narrower than the training caps.

C1 and E13 were re-audited as possible omissions. C1's reported +2.47 pp is produced by mixing a clean embedding with a same-identity prototype; it does not materialize a peak-space spectrum action, so treating that vector as a direct v3 action would silently reintroduce embedding distillation. E13 is not a new action bank: it is a failed shared-encoder training attempt using the E12-B recurrence action. Consequently neither C1 nor E13 is added as a fake extra direct-action source. In contrast, the six restored original intensity recipes do materialize valid peak-space spectra through the existing positive-guided executor and therefore belong in the direct-action inventory.

The development corrective conservative-gain proxy has median 0.05663. After the frozen 0.50 transfer fraction and 0.10 cap, the median target shift is 0.02832 and the maximum is 0.05; 272/848 actions hit the cap. This scalar audit is only a proxy. Formal training must log live per-edge active and capped counts.

## 4. Direct-v3 objectives

### 4.1 Corrective

For each query, the action and its paired control are scored against the same live shared-encoder molecule-max identity references. A transfer edge is legal only when both action-minus-clean and action-minus-control are positive. The target shift is:

`0.50 * min(relu(action-clean), relu(action-control), 0.10)`

The target value is detached so the model cannot manufacture a larger target, while the clean query and identity references keep the mature shared-encoder rank geometry. This is an action-conditioned direct boundary update, not embedding distillation.

In parallel, the real action spectrum and its shared identity references receive a small identity-ranking gradient and a one-sided safety floor whenever the live action margin falls more than 0.005 below the detached clean margin. The safety term repairs action drift without pulling the clean query toward an action or turning the action into an embedding teacher. This repairs the v2 case in which action tensors were forwarded but could receive no payload gradient, without reviving the rejected frozen-reference ablation.

The complete-candidate routers now retain, for every selected action and its chosen matched control, the exact E8 positive reference row, hardest-negative molecule index and hardest-negative spectrum row. The trainer adds every selected exact boundary row to the clean E8 reference set, with formal caps of 32 additional positive rows and 32 additional negative molecules (the theoretical maximum from 16 selected actions times action/control). If an action hard-negative molecule is already present in the clean Top-k, its distinct spectrum row is merged into that same molecule rather than skipped. Multiple routed argmax rows inside one molecule are never truncated by the ordinary clean-reference supplement limit. Each corrective action is optimized on the union of the clean Top-k, that action's Top-k and its control's Top-k. Thus a candidate switch is no longer erased merely because it was not difficult for the original clean view. All three causal arms receive the same true routed candidate frontier; the shuffled arm changes action content, not the frozen initial boundary.

Direct-v3 also restores the mature E8 `0.25` symmetric clean/action live-view consistency channel for corrective actions. Both views are produced by the same trainable encoder and both receive gradient; no action embedding is frozen or used as a teacher target. This preserves the full shared-embedding augmentation path in addition to the candidate-edge transfer path. It is calibrated and reported as a separate branch, and the pre-risk internal-combination gate rejects the run if consistency cancels the other action branches.

### 4.2 Robust

A robustness action receives action-view identity rank in the shared reference geometry and a one-sided floor relative to the detached clean margin. It cannot create a clean corrective target. It teaches the shared encoder that a verified safe perturbation must remain identifiable.

Its edge mask is the union of clean and action Top-k, so a safe view cannot hide a newly difficult candidate outside the clean-only boundary.

### 4.3 Harmful

The harmful spectrum is never optimized or imitated. Its detached margin determines which candidate edges suffered damage. Only those edge weights strengthen the corresponding clean query boundary and initialization floor. Unlike v2, the particular harmful content now changes the gradient.

The harmful-view Top-k is also unioned with the clean Top-k before detached damage weighting, so the exact candidate made dangerous by the harmful action enters the protection gradient.

### 4.4 Equalization and calibration

Within an N/P/A4 block, reductions are family-equal, then source-equal within a repeated family, then action-equal within that source/family cell. Across queries, frozen inverse-multiplicity weights have global mean one and make each IK14 identity's cumulative epoch mass equal inside each supervision panel. In the corrective branch only, a second frozen coefficient removes cross-query mechanism prevalence: if `E_m` is the identity-weighted number of corrective queries containing mechanism `m`, `Q` is total corrective query weight and `M` is the number of mechanisms present, every action in block `m` receives `Q / (M * E_m)`. The loss sums those weighted block means, so every present corrective mechanism contributes exactly `Q/M` cumulative epoch coefficient even when many queries contain only P. No action or query is dropped, and the corrective branch's constant-unit scale remains one before gradient calibration. A formal panel is rejected if inverse-incidence equalization needs a coefficient above the registered branch-scale cap of 16; one extremely rare mechanism is not allowed to acquire unbounded leverage merely to satisfy an algebraic equality.

Robust and harmful branches deliberately retain the older query-local mechanism mean. They still cover every selected action and consume each selected query batch exactly once per epoch; there is no sparse-panel global duty multiplier. A sparse panel keeps unit scale. If a panel is dense enough to place more than one of its batches per optimizer step, its per-batch scale becomes `steps / panel_batches`, capping its cumulative epoch mass at one calibrated batch per optimizer step. A single rare robust/harmful N or A4 query is therefore not promoted to one third of the entire auxiliary objective, while a dense N panel cannot multiply its calibrated constraint merely by contributing many microbatches. This distinction incorporates the E15 warning that sparse-source equal weighting is unsuitable as a direct training weight while still preventing high-coverage P or N auxiliary coverage from drowning the actual corrective transfer.

Identity-stratified training batches emit at most one view per identity until the other identities in that round are exhausted, reducing single-step multiplicity spikes and clipping. Calibration uses a separate formula-first, identity-safe, without-replacement order, then greedily orders those already formed batches so rare available mechanisms enter the 32-batch prefix before repeated common mechanisms; it does not rebatch, duplicate or remove an example. The formal gate requires every mechanism available in each corrective/robust/harmful panel to appear in its calibration prefix. All four branches must contribute at least 32 distinct full batches; robust or harmful batches may not be recycled to fake this count. This retains every spectrum/query while preventing repeated identities, frequent formulas or high-coverage P queries from biasing the frozen scale estimate.

The clean protection objective restores the mature E8 relative geometry weights: rank continuation 1, E8-margin floor 2 and E8-embedding preservation 5. It also restores E8's safety-selected gradient clip of 1.0; the rejected clip-2 variant is not smuggled back in. The base clean boundary again uses four same-identity positive references; all distinct exact action/control positive rows are retained, up to the formal maximum of 32 additional rows. The dense corrective branch is norm-matched to the protect branch; robust and harmful remain separately bounded auxiliaries, and the worst-case all-branch norm determines only the common pre-clip scale. Frozen pretraining calibration targets these gradient ratios:

- complete corrective payload (rank plus safety floor) / corrective transfer: 0.25;
- symmetric live clean/action consistency / corrective transfer: 0.25;
- robust / corrective transfer: 0.10;
- harmful boundary / full-graph protect: 0.25;
- dense corrective branch (`transfer + payload + consistency`) / protect branch: 1.0.

These are bounded starting ratios, not claims that the numerical values are universally optimal. A routed branch hitting its scale cap stops the run before training. A second frozen calibration pass measures the norm of the actually combined action gradient; it may not be replaced by the sum of component norms. The p10 ratio of combined norm to the component-norm upper bound must be at least 0.50, otherwise internal branch cancellation also stops the run. Calibration is performed before the first optimizer update and then frozen. The trainer itself contains the registered formal-v3 configuration and rejects any formal parameter drift before model/data loading; this is independent of the SBATCH text audit. Development smoke runs remain explicitly exempt so bounded fixtures can execute.

The cap rule applies to both active causal arms and to both calibration levels. If any requested payload/consistency/robust/harmful scale exceeds its branch cap, or if the requested total corrective-to-risk scale exceeds its global action cap, the job fails before the first optimizer update. The old behavior merely clipped the requested scale and continued, which could silently under-dose a weak action path or make routed and shuffled controls incomparable. Reports now preserve requested and effective scales, cap-truncation flags and an exact target-ratio check; the arm summary rejects any active arm with a truncated calibration.

## 5. Epoch-balanced scheduling

Corrective batches of at most four complete queries define a shuffled cycle, not a ceiling on optimizer steps. Every full-graph protect microbatch of eight queries is consumed once, with at most four protect microbatches behind one optimizer step. If the auxiliary/protective step requirement would make the natural corrective packing exceed the registered recycle cap, the identity-stratified corrective order is deterministically repartitioned into the smallest larger number of non-empty batches. Smaller batches are evenly dispersed, no query/action panel is split or removed, and partial-batch cardinality scaling preserves per-query mass. The complete cap-safe shuffled corrective cycle is exhausted before it is recycled. The formal route must keep both the weighted recycle factor and physical action exposure at or below 4.0.

The registered direct-v3 mode uses identity-equalized full-dose corrective recycling. Reciprocal repeat weights are multiplied by the mean recycle factor, so every cap-safe corrective batch has identical cumulative epoch mass even when the last cycle is incomplete, while the mean optimizer-step action scale remains one. This is an intentional repair of optimizer-level action starvation, not accidental dose from having explored more recipes. N/P/A4 and source/family reductions remain equal inside each query, all three causal arms use the identical schedule, and the report records the natural batch count, cap-safe batch count, unique actions, raw exposures and weighted dose separately. The same integer geometry is checked before model loading and again in every epoch. A normalized-replay implementation remains testable but is not the registered formal mode.

The trainer also recomputes the actually scheduled identity-weighted mechanism coefficient after corrective recycling. Corrective N, P and A4 masses must remain numerically equal in every epoch; a static pretraining coefficient report alone is insufficient. Robust/harmful masses are reported but are not forcibly equalized across queries. Any corrective schedule-time drift stops training.

Protect gradients are accumulated across the bounded one-to-four microbatches in a step with one fixed epoch coefficient per microbatch before PCGrad. Robust and harmful batches are spread across the same steps and appear once. Sparse panels retain unit scale; dense panels are scaled so each panel's cumulative epoch mass is at most one calibrated batch per optimizer step. Their microbatch count therefore preserves coverage without becoming an unregistered loss multiplier.

Corrective, robust and harmful forward graphs are backpropagated sequentially into separate corrective and auxiliary gradient buffers. The auxiliary buffer is first projected so it cannot oppose the corrective direction; their sum is then passed to the full-graph risk projection. The trainer never retains all three large spectrum graphs simultaneously. This is required after adding candidate-switch references: a worst-case corrective batch can contain roughly 350 spectrum views, and retaining auxiliary graphs at the same time would create an avoidable GPU-memory failure mode.

The full ledger must satisfy at most four protect microbatches and four combined robust/harmful microbatches per action step, plus at most 4x corrective-batch recycling. The optimizer-step count is expanded to meet both microbatch-cap constraints. Otherwise training stops before the first update. This prevents a query-limited action ledger from being disguised as a full-graph experiment while avoiding both the old many-protect-to-one-update compression and dense auxiliary-loss amplification.

An epoch refresh does not replace and forget the initialization-hard competitors. Negative references are the molecule-level union of the initial E8 top-8, the exact routed action/control rows and the epoch-current top-8. If several sources select the same molecule, their distinct spectrum rows are merged under that one chemical identity; exact routed rows and initialization-hard rows survive while current rows lead the ordering. The live loss still chooses the hardest eight molecule edges. This protects against candidate-switch reintroduction without double-counting one chemical identity or allowing per-step reference jitter.

Before the integration smoke, the Slurm job independently replays the hierarchical selector from `routing_ledger.csv.gz`, requires exact action-ID/supervision equality with the materialized `training_actions.csv.gz`, proves that no available N/P/A4 mechanism block was starved by a per-query cap, and requires the formal scheduling ratio to pass.

The plan audit also reports the fraction and percentage-point coverage of current-E8 training errors that have at least one selected corrective spectrum action. This is an executable-action headroom diagnostic only; it is never relabelled as a held encoder forecast or added to E8 performance.

The formal job requires this corrective-action coverage to be at least 80% of all outer-training current-E8 errors. Separately, selected actions that actually move a current-E8 error to action rank 1—not merely improve a still-negative margin—must cover at least 4.0 percentage points of all 65,286 legal training queries. Partial-margin actions remain valid lower-strength supervision, but they cannot inflate the Recall@1 capacity gate. The second quantity is an executable training-side ceiling, not a held forecast, and it still does not guarantee cross-formula transfer. Falling below either threshold stops before fine-tuning.

Required schedule invariants:

- every selected corrective action is exposed at least once before recycling, and every robust/harmful action is consumed exactly once per epoch; sparse auxiliary panels use unit scale, while dense panels are scaled to at most one calibrated batch of cumulative mass per optimizer step;
- corrective unique-action coverage, exposure multiplicity and weighted dose are reported separately;
- every outer-training protect query is consumed exactly once per epoch;
- no protect-only optimizer tail exists;
- every action-active step records raw action gradient, post-PCGrad gradient, clipping survival and actual AdamW update alignment.

## 6. Measuring the old 90% loss correctly

PCGrad-plus-clipping retention alone is insufficient. Direct-v3 reports:

1. action gradient norm before PCGrad;
2. action gradient norm after PCGrad;
3. PCGrad retention;
4. clip retention;
5. their action-specific product;
6. actual parameter update norm after `optimizer.step()`;
7. cosine alignment between the real AdamW descent update and the retained action gradient.
8. a same-state virtual AdamW `combined(action+risk)` update and `risk-only` update;
9. the norm fraction and action alignment of their difference, which is the local action-attributable optimizer update.

The frozen gates are action-retention p10 at least 0.50, clip-event fraction at most 0.10 and optimizer-action-alignment p10 at least 0.05. The virtual combined update must reproduce the real AdamW step to relative error at most `1e-3`. PCGrad and clip retention remain measured on every active step. AdamW update alignment and the combined-versus-risk counterfactual are both measured at 16 uniformly spaced optimizer positions per epoch, including both endpoints (at least 64 observations per formal active arm); full trainable-parameter snapshots are created only at those registered positions instead of at every step. The p10 norm of the action-attributable AdamW update relative to the combined update must be at least 0.10, and its p10 alignment with the retained action gradient must be at least 0.05. The 0.10 boundary is not a performance-tuned hyperparameter: it directly rejects the stated failure mode in which at least 90% of the action contribution disappears between gradient and parameter update. A common global multiplier is still logged but is not misreported as action-specific loss; causal routed-versus-shuffled/clean gates still decide whether the retained contribution is useful.

The same registered audit positions now split this accounting into the projection head and the unfrozen final backbone block. Each group must receive a nonzero action gradient on every sampled step, retain at least 10% through PCGrad plus clipping, retain at least 10% of its AdamW update as action-attributable, and meet the same 0.05 update-alignment boundaries. This does not add a loss or change optimizer dose; it prevents a healthy head from hiding a 90% or complete action断档 in the trainable encoder block.

Aggregate action norm is not allowed to stand in for the corrective direction. On every active step the trainer measures the projection of the post-risk-PCGrad, post-clip action gradient onto the pre-risk dense corrective gradient. Its p10 must be at least 0.10 globally and separately in the head and backbone. At the registered virtual-AdamW positions, the action-attributable parameter update must align with that same corrective gradient by cosine at least 0.05, again globally and in both parameter groups. This closes the loophole in which robust/harmful orthogonal components preserve a large total action norm while the actual error-correcting direction loses 90% or more.

Signal accounting starts before gradients: calibration and every active-arm epoch also report the fraction of eligible top-k edges with a live positive corrective transfer, the fraction clipped by the transfer cap, and the fraction activating the corrective action-view safety floor. The transfer ledger is additionally split by N/P/A4 mechanism, action family, individual source, source-by-family cell and control semantic. For every stratum it sums exact action-better-clean, action-better-control, jointly active, clean-limited, control-limited and capped edge counts over the whole calibration/epoch ledger, then calculates pooled fractions from those totals; values named counts are never averaged across minibatches. This locates a real semantic断档 before gradient retention is calculated, prevents a high P fraction from hiding a dead N or A4 path, and distinguishes an actually dead recipe family from dilution caused by the same family being rediscovered by multiple experiment generations. The clean control still skips these no-gradient action forwards.

The training loop keeps four independent gradient ledgers until arbitration: dense corrective, robustness-only, harmful-boundary and full-graph protection. For every optimizer step it records each raw norm; whenever a semantic branch is nonzero it records its raw cosine against protection, plus corrective-versus-robust and corrective-versus-harmful cosines when those panels are coactive. Calibration records the complete six-pair cosine matrix before inner projection and PCGrad. Only after this accounting are robust and harmful summed into the auxiliary constraint and projected against corrective. This does not add a loss or change dose, but prevents a healthy auxiliary total from hiding cancellation or a dead individual semantic branch.

For the routed arm, both the median calibration-batch aggregate active-transfer edge fraction and every present N/P/A4 mechanism's pooled active fraction must be at least 0.05. A nonzero aggregate transfer norm is insufficient if it comes from only a vanishing handful of edges or only one high-coverage mechanism. The shuffled control may correctly lose this semantic branch and is not forced to fabricate it. Its raw aggregate and per-mechanism fractions are still recorded, while both routed-only semantic gates are explicitly marked not applicable rather than silently reported as shuffled passes.

## 7. Causal controls

Every seed has three arms with identical initialization, ledger, schedule, learning rates, requested branch targets and full held graph.

1. `routed_direct`: chemically matched actions.
2. `shuffled_action_control`: action spectra reassigned across different queries within the same supervision kind, source, family and exact recipe, preserving dose, reference policy, N step/A4 cell and branch semantics while breaking query chemistry; a single-query exact cell uses its paired control. Family is an explicit key because the formal A4 `recipe_id` contains token and dose while A4 family independently contains peak role and gradient-rank bin. The older source/family-only development diagnostic reached 1,448/1,457 cross-query rows (99.38%), but that number is not reused because it did not prove exact-recipe matching. The formal source/family/exact-recipe control must independently retain at least 95% cross-query rows and no row may keep the same-query true action tensor.
3. `clean_control`: identical schedule and protect objective with the action gradient disabled.

Both active arms are independently calibrated to the same total action/protect gradient ratio. If shuffling correctly eliminates a semantic transfer or harmful branch, that branch is reported as zero and is never fabricated merely to imitate the routed arm; the remaining control gradients are total-norm matched. The routed arm must beat both controls with a strict formula-cluster Recall@1 CI. This distinguishes semantic action value from extra optimization or generic augmentation.

The clean-control arm keeps the exact optimizer-step, protect-query and risk schedule but skips action/control/reference encoder forwards because those branches are specified to contribute zero gradient. The report retains their frozen ledger counts and asserts the optimization contract. This removes pure compute waste from three formal control runs without changing any update.

Every arm report freezes the complete v3 objective/safety/calibration configuration and hashes the trainer, v3 objective core, v3 action router and full-graph evaluator. Reports must agree on these hashes before a causal comparison is computed. The transfer target detaches action/control outcomes, but its live rank loss intentionally updates the shared clean query and identity references; no report may describe that path as query-only.

The saved encoder is verified twice: by the serialized checkpoint-file hash and by a deterministic hash over sorted state-dict tensor names, dtypes, shapes and bytes. Each per-seed summary reloads the checkpoint and checks both declarations. The final three-seed gate requires three distinct routed model-state hashes; changing only a seed field, runtime or JSON file can no longer make a duplicated encoder look like an independent result.

The held per-query table labels the primary baseline columns `initial_E8_*`. It must not inherit the generic evaluator's `official_*` prefix when the supplied baseline is actually E8. The arm summarizer rejects missing/non-finite rows, invalid ranks, query-count disagreement, or any Recall@1 mismatch between the per-query table and `decision.json`; CI and full metrics therefore cannot come from two inconsistent result copies. Official DreaMS remains a separately reported secondary metric panel.

## 8. Evaluation and promotion

For each of three seeds, promotion requires all of the following on the exact outer-formula-held graph:

- Recall@1 gain versus E8 at least +4.0 pp;
- Recall@1 formula-cluster CI strictly positive;
- near-subset Recall@1 formula-cluster CI strictly positive;
- Recall@2/3/5/10/20 all strictly higher unless the exact E8 baseline is already 1.0, in which case exact non-regression is required;
- MRR strictly higher, mean rank strictly lower and median rank non-worse;
- macro-query AUROC and AUPRC strictly higher;
- micro-candidate AUROC and AUPRC strictly higher;
- positive-vs-best-negative margin and correctness-signed Top1--Top2 gap strictly higher;
- near-subset counterparts strictly higher except median rank may be equal;
- MassSpecGym 10-ppm pooled pairwise AUROC/AUPRC strictly higher;
- MassSpecGym `[M+H]+` 10-ppm pooled pairwise AUROC/AUPRC strictly higher;
- corrected exceeds introduced and `corrected - 2 * introduced > 0` both overall and independently on the near subset;
- routed beats shuffled and clean controls with strict formula-cluster CIs;
- all signal-retention, clipping and optimizer-alignment gates pass.

Raw Top1--Top2 gap is still reported, but it is not direction-gated because a confidently wrong prediction also has a large unsigned gap. The signed gap is positive for a correct Top-1 and negative for an incorrect Top-1, so increasing it has the intended monotone meaning.

The four primary per-seed paired formula-cluster intervals (candidate versus E8 overall, candidate versus E8 near, routed versus shuffled and routed versus clean) use a pre-registered Bonferroni family size of four. Unadjusted intervals may remain descriptive but cannot authorize promotion. The final gate requires each of the three registered training seeds to pass independently, verifies the seed stored inside each arm report, rejects a duplicated routed decision, and requires all seeds to share the identical frozen ledger and candidate graph.

The pooled `[M+H]+` value is explicitly named **MassSpecGym `[M+H]+` 10-ppm pooled pairwise AUROC**. It is not called an exact reproduction of the NIST20 paper value 0.85. Before training, the evaluator metadata manifest must match the graph report's frozen SHA256 and exactly reproduce the graph's query rows/identities/formulas, query and molecule pointers, molecule labels/identities/formulas and candidate spectrum rows; a same-length but reordered adduct vector is rejected.

Only if all conditions pass for all three fold-0 seeds may the result be called fold-0 promotion-authorized. It is still not a multifold or P3 result.

## 9. Files and execution boundary

- Core objectives and scheduling: `tasks/noise_corrected_direct_v3_core.py`
- Core numerical tests: `tasks/test_noise_corrected_direct_v3_core.py`
- Submission-bundle import/SBATCH contract test: `tasks/test_noise_corrected_direct_v3_bundle.py`
- Three-way routing: `tasks/noise_corrected_action_routing_v3.py`
- Training-only source/family/exact-recipe shuffled control: `tasks/noise_corrected_shuffled_control_v3.py`
- Current-E8 action panel and lossless frontier: `tasks/noise_corrected_action_panel.py`
- Ledger builder: `tasks/build_noise_corrected_routed_action_ledger.py`
- Integrated trainer: `tasks/train_noise_corrected_routed_direct.py`
- Strict arm summarizer: `tasks/summarize_noise_corrected_direct_v3_arms.py`
- Two-GPU bounded route producer: `tasks/run_noise_corrected_v3_routes_fold0.sbatch`
- Two-GPU successor trainer: `tasks/run_noise_corrected_direct_v3_from_formal_routes.sbatch`

### 9.1 Formal shuffled-control runtime correction

The completed formal route bundle `noise_corrected_npa4_formal_fold_0_run_2332161_v3routes` passed routing, ledger and training-headroom gates and remains reusable. Its formal plan observed 4,448 current-E8 errors, selected corrective actions for 4,186 (94.11%), and measured 5.2844 pp of actual action-rank-1 training-geometry headroom. This is a capacity gate, not a held-performance result.

The first chained training job then failed before formal optimization in `source_family_shuffled_action_bank`. The original implementation grouped shuffled donors by supervision kind, source and `recipe_id`, then asserted that each such group had one family. That assertion is false for formal A4: its `recipe_id` records token and dose, while peak role and gradient-rank bin are stored independently in `family`. The small synthetic tests used one family per recipe and therefore failed to exercise the formal schema collision.

The repaired shuffled control is isolated in `tasks/noise_corrected_shuffled_control_v3.py`, leaving the already-frozen route module and its hash unchanged. Donors are now grouped by supervision kind, source, family and exact recipe; a regression fixture reproduces the A4 collision and proves that donors neither cross families nor retain the same query. The trainer hashes this training-only module, all three arms must share that hash, and a dataframe-only formal preflight requires at least 95% cross-query coverage before model initialization. All formal arms must be rerun together after this correction; partial arms from the failed job cannot be mixed with the repaired run.

The already submitted old N+P+A4 job is immutable and is not modified or cancelled. Its server copy may still finish and remains useful as a routing-capacity diagnostic, but its frozen router version does not record exact action/control candidate-switch rows and therefore cannot silently enter the strengthened v3 trainer. The new bounded route producer is the formal replacement: one submission creates N/P/A4 routes with the required boundary provenance and, after all route contracts pass, automatically submits the v3 trainer with the frozen route directory. No Python command in this workflow is to be run on the login node. Both SBATCH files request exactly two GPUs and contain no manual memory directive.

The trainer runs the nine seed/arm jobs through a two-GPU work-conserving queue rather than leaving one GPU idle during each clean-control arm. Per-seed summaries are computed only after all nine immutable arm directories exist, so the scheduling optimization does not change causal pairing.

## 10. What remains empirical

This implementation removes the previously demonstrated compression, scheduling and gradient-accounting failures; it does not mathematically guarantee a +4 or +5 pp encoder. Three facts can only be established on the formal compute-node run:

1. The complete outer-train routes must retain corrective actions for at least 80% of current-E8 errors, selected actual Top-1-correcting actions must cover at least 4.0 percentage points of the 65,286 legal training queries, and the protective/action scheduling ratios must remain within their registered caps.
2. The integration smoke must show live routed transfer in every available N/P/A4 mechanism and must pass aggregate plus separate head/backbone end-to-end action-retention gates, including corrective-direction survival after risk PCGrad, clipping and AdamW. A route/action headroom number cannot substitute for this evidence.
3. Each of the three routed encoders must independently gain at least +4.0 pp on the current run's outer-held fold-0 formula panel, beat both causal controls with simultaneous positive formula-cluster intervals, improve the entire registered metric panel, and pass corrected/introduced risk-net gates both overall and on the near subset. This remains a historically used development fold, not an untouched confirmatory test.

If route construction or the integration smoke fails, the nine formal arms do not start. If signal transmission passes but held performance does not, the result identifies a generalization/objective problem rather than another hidden 90% injection断档; the first diagnosis must use the recorded mechanism/source/family/control-semantic transfer counts and head/backbone attribution, not an unregistered learning-rate or loss-weight sweep.
