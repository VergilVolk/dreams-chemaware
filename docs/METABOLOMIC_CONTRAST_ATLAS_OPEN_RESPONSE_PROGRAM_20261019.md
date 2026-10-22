# Metabolomic Contrast Atlas: open study-response program

Date: 2026-10-19  
Role: biology-method development; all currently assembled studies are development/consumed

## Scientific object

The primary object is not a spectrum, annotation, disease label, or sample-composition vector. It is a study-level quantitative response:

\[
R_{d,c}=\{z_i,\hat\beta_{dci},SE_{dci},q_i,e_i\}_{i=1}^{n_{dc}},
\]

where `d` is a study, `c` is a design-native contrast, `z_i` is phenotype-blind MS/MS evidence, `beta/SE` describe the within-study directed response, `q_i` is measurement support, and `e_i` records identity and ion-family evidence. Exact structure is optional; effect direction and uncertainty are not.

The central falsifiable hypothesis is:

> A study-level, directed and uncertainty-aware chemical response can be searched and accumulated across independent public LC-MS/MS studies, and unresolved MS/MS-linked entities can add correct, calibrated biological relations that a measurement-matched known-only analysis cannot recover.

This is the intended extension beyond a spectrum atlas. DreaMS Atlas organizes 201 million spectra through spectral embeddings and metadata; MASST/reverse metabolomics searches spectrum occurrence and context; StructureMASST searches structures and substructures; MEMO represents sample composition; MS2LDA/PALS organize motifs or metabolite sets; PAIRUP-MS matches features for meta-analysis. The required new contribution is quantitative response retrieval with direction, uncertainty, abstention, unknown-information attribution, and independent biological verification. A knowledge graph or web interface alone is not this contribution.

Relevant primary references:

- DreaMS and DreaMS Atlas: https://doi.org/10.1038/s41587-025-02663-3
- MASST: https://doi.org/10.1038/s41587-019-0375-9
- ReDU: https://doi.org/10.1038/s41592-020-01101-3
- reverse metabolomics: https://doi.org/10.1038/s41586-023-06906-8
- StructureMASST: https://doi.org/10.1038/s41587-026-03082-8
- MEMO: https://doi.org/10.3389/fbinf.2022.842964
- MS2LDA+: https://doi.org/10.1021/acs.analchem.7b01391
- PALS: https://doi.org/10.3390/metabo11010044
- PAIRUP-MS: https://doi.org/10.1371/journal.pcbi.1006733
- iDMET: https://doi.org/10.1186/s12859-022-05068-0
- public-repository metabolite/pathway meta-networks: https://doi.org/10.1007/s11306-026-02507-4

## Prior-art correction and defensible novelty boundary

The broad idea of comparing study-level differential metabolomic profiles is **not new**. iDMET already represented condition contrasts as differential metabolomic profiles, compared their concordant up/down named metabolites, and connected 27 public cancer studies; it also reported a cross-study drug-response hypothesis. A 2026 repository-integration framework further constructed metabolite- and pathway-level meta-networks from harmonised annotations and demonstrated a global study network across 1,181 Metabolomics Workbench studies. Therefore neither "a network of biological contrasts", "public-study integration", nor "searching similar metabolic responses" is a defensible standalone novelty claim.

The claim that remains open and worth testing is narrower and harder:

> Can phenotype-blind MS/MS evidence extend cross-study quantitative-response retrieval beyond harmonised named metabolites to unresolved chemical entities, while retaining effect uncertainty, open-set abstention, calibrated false-link risk, and a measurement-matched demonstration that the unresolved evidence changes a correct biological conclusion?

This distinction is mandatory. iDMET and current repository meta-networks depend on common metabolite names or harmonised identifiers and largely threshold differential calls or count their co-occurrence. The proposed contribution must instead succeed on unresolved MS/MS-linked entities and must beat both named-metabolite response integration and spectrum/composition matching on the same candidate graph. If it cannot do that, the project is an implementation variant of existing response-network work rather than a new biological discovery method.

### 2026-10-20 correction: identity transport and chemical-relation transport are different scientific objects

The original open DreaMS graph was built by unrestricted spectral-neighbour search. That graph is useful for finding chemical analogues, but it cannot be used to merge effects as if every edge represented one entity. Among 5,695 frozen `all`-world directional edges, only 2,379 are within 10 ppm; the median absolute precursor difference is 1.9822 Da. Transitive closure is worse: among 20 multi-member components with at least two strict known identities, only 14 were identity-pure, and only 28/54 calibratable known-member pairs belonged to pure components. Ordinary molecular-family clustering is therefore disallowed for claim-bearing response replication.

A new phenotype-blind identity graph was frozen before strict identity truth was opened. It first restricts each query to 10-ppm-compatible candidates in each independent target source, then selects the top official-DreaMS cosine above 0.70. Across 17,358 entities from five consumed studies it contains 2,516 `all`-world directional edges. On 124 strict planar-identity query-to-target-source groups, it answered 95 and recovered 91 correctly: 76.61% total coverage and 95.79% precision among answered. The matched `known` control answered 94 and recovered 93 (98.94% precision). These are posttruth development-calibration results, not evidence that unresolved links are correct and not external confirmation.

The method now has two non-interchangeable outputs:

1. **identity track** — intended to use precursor/neutral-mass and adduct-compatible, identity-calibrated links with a no-match option; only this track may accumulate beta/SE for the same unresolved entity across studies. The current implementation is narrower: same observed precursor within 10 ppm, without full neutral-mass/adduct resolution;
2. **relation track** — cross-mass or analogue spectral relations used to test whether distinct entities support a reproducible biological-response association. These relations cannot count as entity replication or share a same-molecule meta-analytic effect, but they are a claim-bearing scientific layer when independently calibrated and biologically verified, not merely a neighbour lookup.

This correction tightens the novelty boundary. PAIRUP-MS already matches unknown signals across studies and performs meta-analysis; iDMET+ already retrieves differential profiles; reverse metabolomics and StructureMASST already connect spectra or structures to public context. A defensible advance must jointly demonstrate MS/MS-conditioned unresolved **entity** recovery across heterogeneous acquisition, uncertainty-aware response accumulation, calibrated abstention, and a held-out biological conclusion unavailable to measurement-matched known-only analysis.

The 2026 repository study also exposes the precise empirical opening rather than merely a rhetorical gap. In its six-study COVID plasma example, only ten annotated metabolites recurred in at least three studies and none recurred in four or more. Mass-only harmonised re-annotation greatly increased putative ChEBI coverage, yet the authors reported weak or little validation concordance after that expansion and noted that retention time and fragmentation evidence were unavailable but desirable. Our hypothesis is therefore falsifiable against an especially relevant predecessor: MS/MS-linked unresolved evidence must improve correct cross-study response conclusions without reproducing the false-positive expansion caused by permissive mass annotation.

## Retrieval, not label matching

For a frozen spectral representation `phi`, a minimal response sketch is the signed spectral measure

\[
\mu^+_{dc}=\frac{\sum_i w_i^+\phi(z_i)}{\sum_iw_i^+},\qquad
\mu^-_{dc}=\frac{\sum_i w_i^-\phi(z_i)}{\sum_iw_i^-},
\]

with uncertainty-shrunk response weight

\[
w_i=\frac{\hat\beta_i}{\sqrt{\hat\beta_i^2+SE_i^2}}\sqrt{q_i},
\quad w_i^+=\max(w_i,0),\quad w_i^-=\max(-w_i,0).
\]

A query is compared with every source-family-disjoint candidate. Similar and opposite responses are returned separately. Missing chemical support, missing direction, or an interval crossing zero causes abstention. Candidate condition names may be opened only after the truth-blind ranking is frozen.

The current fixed-capacity partial-transport implementation is the claim-bearing estimator because its chemical graph and transport capacities are phenotype blind. Response-adaptive historical transport scores are descriptive only.

## First real open collection frozen on 2026-10-19

The machine contract is `tasks/contrast_atlas_open_collection_v1.json`; the builder is `tasks/build_contrast_atlas_open_response_collection.py`; the current artifact is `data/validation/contrast_atlas_open_response_collection_v2_20261019`.

It contains:

- 5 independent accessions/source families;
- 16 study-level contrasts;
- 221 biological units across sources;
- 42,447 representative MS/MS spectra;
- 24,549 identity-partitioned feature entities: 381 library-supported and 24,168 unresolved;
- 17,358 unique response-eligible MS/MS-linked feature entities;
- 59,789 eligible contrast-entity response rows: 960 library-supported and 58,829 unresolved;
- 12-15 source-disjoint candidate contrasts per query.

The five sources cover mouse urine HFD, human urine obesity/diabetes, human saliva obesity/depressive symptoms, mouse plasma HFD/sex, and human skin atopic dermatitis.

Only one relation family is registered as exact at the study-design level: `MSV000097426 mouse_hfd_vs_nfd` against the female- and male-stratified HFD contrasts in `MSV000088023`. It is an exact intervention family, not a guarantee that urine and plasma metabolite directions agree. The expected signed direction is deliberately unspecified. Cross-biofluid obesity is a consumed coarse-relation falsification that previously returned the wrong target; AD has no independent full-unknown counterpart in this collection. Every other pair is unadjudicated, never a negative.

This collection is already larger and scientifically more open than the AD two-query/two-candidate experiment, but it remains development/consumed and has no external-validation claim.

## First open retrieval result: transparent fragment baseline

The first truth-blind retrieval was completed on the frozen five-source collection with a deliberately transparent representation: 0.1-Da fragment bins, square-root intensity, per-spectrum L2 normalisation, and positive/negative response measures weighted by

\[
w_i=\hat\beta_i/\sqrt{\hat\beta_i^2+SE_i^2}.
\]

All 16 contrasts were queried against every source-family-disjoint contrast. The fixed universe contains 198 directed query-candidate rows per evidence world and 594 rows across `all`, `known`, and `unresolved`; every row was retained and scoreable. Registered relations were opened only after the complete score table and ranks were written.

This baseline did **not** recover the registered HFD study family. With mouse-urine HFD as query, the female- and male-stratified mouse-plasma HFD contrasts ranked 12/15 and 11/15 by signed-similar response in the `all` world, 12/15 and 13/15 by unsigned response, and 12/15 by composition. In the reverse direction, signed-similar ranks were 7/12 and 5/12, while unsigned and composition ranks were both 12/12. Because signed direction was not preregistered across urine and plasma, the more favorable signed-opposite ranks are not retroactively counted as success.

The signed response did change the retrieval problem rather than merely copy sample composition: only 1/16 `all`-world queries had the same Top-1 contrast under signed-similar and composition scoring (3/16 shared only the same target source). However, 15/16 `all` Top-1 answers were identical to `unresolved` Top-1 because unresolved entities dominate the collection numerically. This is not evidence of unknown-information gain. It makes a measurement-matched known control and a response-profile reassignment control mandatory before any unknown-increment claim.

The correct conclusion is negative and method-informative: a global fragment-composition response centroid is not sufficient to create a cross-study biological-response atlas. Source/biofluid chemistry and measurement domain remain stronger than the registered design relation. The result must not be rescued by threshold selection. The next comparison therefore moves to phenotype-blind local chemical alignment or frozen spectral embeddings on the identical response objects and candidate universe, with response reassignment and composition controls retained.

Artifacts: `tasks/run_contrast_atlas_sparse_fragment_measure_baseline.py` and `data/validation/contrast_atlas_sparse_fragment_measure_baseline_v1_20261019`.

## Seven-source extension with an independent IBD relation family

The first collection was extended without changing or overwriting its frozen artifact. The append-only contract is `tasks/contrast_atlas_open_collection_ibd_extension_v1.json`; the extended artifact is `data/validation/contrast_atlas_open_response_collection_v3_ibd_20261019`.

The extension adds two independent human fecal studies: pediatric VEO-IBD versus healthy (`MSV000097610`, 21 versus 27) and the merged adult IBD200 source family (`MSV000084775`, `MSV000084908`, and `MSV000086509` treated as one source; 158 CD/UC versus 19 healthy). Their complete ledgers contain 6,081 and 746 MS2-linked entities with finite beta and positive SE. The old pairwise program had retained only 86 reciprocal links; these complete ledgers prevent that correspondence filter from defining the response universe.

The resulting open collection contains:

- 7 independent source families and 18 study-level contrasts;
- 446 biological units;
- 54,142 nonempty representative MS/MS spectra recorded by the source contracts;
- 24,185 response-eligible feature entities and 66,616 contrast-entity response rows;
- 1,602 library-supported and 65,014 unresolved response rows under the source-frozen partitions;
- 14-17 source-family-disjoint candidate contrasts per query;
- two registered relation families (HFD and IBD), six directed relations after reversal, and zero registered signed-direction truths.

The IBD relation is only `prespecified_near_exact_condition_family`: species, fecal biospecimen, and IBD-versus-healthy design agree, while age, subtype, and treatment exposure differ. Its expected signed direction is unspecified. The previous IBD exact-target ranks (2 in all/known and 3 in unresolved) remain a frozen negative result. In addition, the IBD artifacts currently carry their legacy local library-hit partition; until a common identity policy is reapplied, they may expand the `all`-entity open retrieval but cannot support a cross-source unknown-increment claim.

The transparent fragment-measure baseline was rerun once on this enlarged, fixed universe. The IBD relation ranked 5/17 from pediatric to adult and 4/17 in reverse under `all` signed-similar scoring; it was not Top-1. Under the `known` composition control the two sources ranked first for one another, whereas their signed-similar ranks were 7/17 and 3/17. This separation is scientifically important: shared biospecimen chemistry can make two studies compositionally recognizable without making their directed biological responses the best match. The enlarged HFD ranks also remained weak (13-14/17 forward and 7-8/14 reverse). Thus a second relation family confirms, rather than rescues, the failure of global fragment centroids.

After the truth-blind candidate scores were frozen, a separate strict-identity calibration ledger was opened. It uses only exact `study_id + raw_entity_id` joins and valid planar InChIKeys from frozen library-supported annotations; compound-name, precursor-mass, retention-time, and score-based fallbacks are forbidden. Among 936 response-eligible library-supported entities, 630 have a strict planar key. Forty planar identities occur in at least two independent source families, yielding 157 member entities, nine source-family pairs, and 418 directed cross-source exact-identity pairs. This ledger can test whether a frozen spectral graph retrieves the same chemical connectivity across sources. It is not biological-relation truth, does not specify response direction, and cannot demonstrate value from unresolved entities. Artifact: `data/validation/contrast_atlas_exact_identity_calibration_v3_20261019`.

A strict-identity known-response baseline was then run as the relevant annotated-response comparator. It aggregates duplicate features within each `study x contrast x planar identity` without treating redundant ions as independent precision, aligns only exact identities, and retains every source-disjoint candidate even when no response score can be estimated. Of 264 directed candidate pairs, only 56 had at least three shared strict identities; 130 had none and 78 had fewer than three. The median overlap was one identity and the maximum was 15. The registered HFD pairs had zero strict-identity overlap and were therefore unavailable. The two IBD sources shared 12 identities: pediatric-to-adult ranked 2/4 by signed-similar response and 1/4 by unsigned response, while adult-to-pediatric ranked 1/1 by both. Because signed direction is not preregistered and the answerable candidate graph is sparse, these are descriptive ranks, not biological confirmation. The result establishes the correct opportunity and risk: unresolved MS/MS may increase answerable coverage, but it counts as progress only when those added answers are independently correct. Artifact: `data/validation/contrast_atlas_strict_identity_known_response_baseline_v1_20261019`.

The frozen local official-DreaMS graph was then audited **without tuning it** against that separately opened strict-identity ledger. Across the five-source graph, 124 query-to-target-source groups have an exact planar-identity answer (306 directed entity pairs). In the `known` world the frozen `k=1`, cosine-at-least-0.70 graph emitted an edge for 98/124 groups (79.03%) and selected a correct same-planar-identity target in 92/98 (93.88%). In the `all` world it emitted 100/124 edges (80.65%) but only 87/100 were correct (87.00%); adding unresolved distractors therefore reduced exact-identity Top-1 precision in this calibration. The unresolved world has no identity truth by construction. This resolves an important diagnostic ambiguity: the weak HFD response ranks are not simply because the frozen representation cannot connect known chemical entities across sources. They are still **not** biological evidence: the HFD signed direction is unspecified, and neither calibration result validates response retrieval or an unknown-information increment. Artifact: `data/validation/contrast_atlas_official_local_exact_identity_calibration_v1_20261019` and `tasks/evaluate_contrast_atlas_frozen_graph_exact_identity.py`.

## Repository-scale open candidate program

The frozen Pan-ReDU positive-mode design has 18 accessions, 20 panels, 18 preliminary source families, 37 real contrasts, 188 source-disjoint panel pairs, and 1,258 directed query-target comparisons. Each query has 30-36 candidates (median 35). This is the current systematic scale path.

The pairwise v4 estimator and finalizer preserve a fixed measurement universe, phenotype-blind ion-family capacities, response uncertainty, explicit abstention, matched-known controls, unresolved controls, and response-reassignment falsification. The truth-blind open view is `tasks/build_pan_redu_open_retrieval_view.py`; it returns Top similar, Top opposite, and abstention without reading comparison labels or truth.

The repository-scale matrix has not yet been executed locally. Its generic BiologicalSex/LifeStage/HealthStatus labels are candidate metadata, not biological truth. Existing design audit provides zero claim-bearing truth pairs. Therefore 1,258 planned comparisons are an execution denominator, not 1,258 results.

## Current validation bottleneck and the next admissible relation family

### 2026-10-19 update: from a blocked pair to a three-source replication experiment

The two-source COVID qualification below remains valid but is no longer the complete experimental design. An independent Austrian plasma study, `MSV000092887`, provides 39 COVID-control matched pairs and a complete 3,041-feature HILIC-positive mzTab matrix. A real paired response object has now been built: 2,598 features pass the abundance-response support rule, 804 pass the differential-detection rule, and 3,020 pass at least one channel. The publication reports 1,982 features with MS2 and 175 reference-spectrum hits; the 1,807 implied MS2-bearing features without a reference-spectrum match require phenotype-blind raw-mzXML linkage before they may be called unresolved MS2 entities.

The COVID program is now one consumed calibration family, not the decisive experiment for the whole method. Two independent sources may nominate identity-calibrated unresolved entities; linkage and response direction are frozen; the third source tests effect direction and uncertainty. Shared `COVID-19` naming is never truth, and analogue-family agreement is not entity replication. Generality must additionally be tested across other perturbation and disease families in the open collection and, ultimately, an unopened source. Full contract and current denominators: `docs/COVID_THREE_SOURCE_UNRESOLVED_RESPONSE_REPLICATION_20261019.md` and `tasks/contrast_atlas_covid_three_source_replication_v1.json`.

A repository-wide audit found **zero** currently processed local pairs that simultaneously provide independent source families, a prespecified signed biological relation, real beta/SE, and feature-linked MS/MS. Disease-name agreement, intervention-name agreement, or a favorable post-hoc direction is not enough to manufacture this truth. The HFD and IBD relations above remain useful design-relevance probes with deliberately unspecified signed direction.

The highest-value low-cost candidate is independent COVID plasma data from `MSV000085505` and `MTBLS1866`. The frozen Pan-ReDU snapshot records 33 COVID and 13 healthy positive-mode files for `MSV000085505`, with 112,077 MS2 scans in total, and 83 COVID plus 26 healthy files per polarity for `MTBLS1866`; its positive-mode files are DDA and contain about 1.11 million MS2 scans. They are separate accessions, repositories, and platforms. This makes them a much stronger candidate for a signed same-biospecimen disease relation than the existing cross-biofluid HFD pair.

They are **not yet admitted as truth**. `MSV000085505` is marked partial and lacks a resolved subject key; the locally visible `MTBLS1866` primary-group file count does not yet match the published analysis denominator; neither source currently has a frozen beta/SE response object joined to feature-level MS/MS in this repository. The next action is therefore a metadata and sample-crosswalk qualification, not another retrieval run. Failure to distinguish biological groups or repeated subjects, unexplained selective file absence, or inadequate feature-MS2 linkage is an immediate stop. Passing that audit would permit positive-mode response construction with the published annotated COVID response serving only as a hidden external anchor.

## Required experiments

### Gate 1: response information must matter

On the same frozen candidate universe compare:

1. composition-only spectrum evidence;
2. unsigned effect magnitude;
3. signed beta without uncertainty;
4. signed beta/SE response;
5. label-shuffled or biological-unit-permuted response.

The method fails if signed uncertainty-aware response does not recover more predeclared relations than composition/unsigned controls at the same error risk.

### Gate 2: unresolved information must add correct answers

Compare `all`, `known_only`, measurement-matched known, identity-field/all-entity controls, unresolved-only, and measurement-preserving unresolved-response reassignment on the identical query/candidate graph. Primary endpoints are correct-answer coverage at fixed false-link risk and source-family-clustered uncertainty. More answered queries without more correct relations is not a pass.

The `known_only` comparator must include a strict harmonised-identity response baseline, not merely the known subset passed through the new spectral scorer. It aligns valid cross-source planar identities, retains direction and uncertainty, and keeps candidates with no shared identities as explicit unavailable/abstain cases. This is the relevant control for iDMET- and annotation-network-style response integration. The proposed method earns a distinct contribution only if unresolved MS/MS evidence adds correct, independently adjudicated answers beyond that comparator.

### Gate 3: algorithm contribution must be attributable

Using identical response objects and candidate graph, compare mass-only, raw cosine, WSE, official DreaMS, Noise V1, and only then a frozen qualified unified model. Noise/ChemAware/P2b training and benchmark selection remain outside this biology-method task. A method is admitted here only as a frozen spectral-evidence provider.

### Gate 4: open discovery must survive independent biology

At least one top open relation not used for model or threshold selection must be validated by a new independent study, an orthogonal perturbation, targeted remeasurement, or an experimentally supported mechanism. Shared disease names are not validation; absent evidence is not a negative; a spectrum neighbor is not molecular identity.

### Gate 5: accumulation must update evidence

When a new study is added, a registered relation must be able to gain support, lose support, or become heterogeneous. The output must retain source-wise effects and uncertainty rather than collapse to a growing count of matching spectra.

## Result categories

Every open query must yield one of:

- `similar_response`: direction identified and positive;
- `opposite_response`: direction identified and negative;
- `indeterminate_response`: chemical evidence exists but response direction is not resolved;
- `abstain_insufficient_evidence`: the comparison cannot be answered;
- `unadjudicated_hypothesis`: a ranked relation without independent biological truth.

`No match` is permitted only after a calibrated query-level risk rule; the absence of a registered relation is not a true negative.

## Anti-drift boundary

- AD local/field retrieval remains an algorithm-attribution positive control, not the definition of the method.
- Kidney treatment-to-normal estimands remain study-specific analyses, not general retrieval rules.
- ChemProp drug matching, TT7 genetics, IBD, HFD, and obesity are shards or controls, not replacement project definitions.
- File volume, test count, schema count, and web-platform features are not scientific endpoints.
- The next claim-bearing milestone is a completed multi-source open response matrix plus at least one independently adjudicated relation for which unresolved MS/MS changes the correct scientific answer.

## 2026-10-20 科学目标冻结与七来源双层裁决

Metabolomic Contrast Atlas 的总目标冻结为**研究级、有方向、有幅度和不确定性的公共生物学响应检索**。身份层回答同一实体复现，化学关系层回答不同但相关实体的可验证响应关联；后者不能合并同一分子的 beta/SE，但不得被降级为普通谱图邻居查询。后续算子失败只修改算子，不再重定义课题。

现有 7 来源、18 对比的真实双层结果已经集中复算。严格身份层在 6 条登记关系中仅回答 2 条、弃权 4 条；关系层 `all` 与 `unresolved` 对 6 条登记关系的四类评分均为 0/6 Top-1。集合没有 signed-direction truth，因此正确/错误开放检索均不可裁决，unknown 增量尚未证明。完整数字与产物见 `docs/CONTRAST_ATLAS_DUAL_LAYER_REAL_RESULT_20261020.md`。所有结果均为 development / consumed。

同日新增的组成中心化响应核把 IBD 双向 signed-similar 排名从 `all` 5/4 改善到 3/1、从 `unresolved` 5/7 改善到 3/3；composition 排名并未同步改善，说明算子开始捕获响应特异信号。但 31 次分层响应重排尚未给出充分的 unresolved 证据，HFD 仍失败，因此该结果只晋级为关系层候选算子，不改变 unknown 增量未证明的中心裁决。

随后发现成人 IBD200 的 19 个健康对照全部位于 Plate3，原 158-vs-19 响应与板次完美混杂。Plate3 内 36-vs-19 重算后，成人→儿童 IBD 的 `all` alignment rank 从 1 降到 7，儿童→成人的 `known` rank 从 1 降到 6。原 IBD Top-1 已正式撤回；仅保留“提示性、对效应估计敏感”的开发信号。

来源内响应轴审计提供了更严格的局部正结果：mouse HFD 查询在同一目标来源的四个候选中，用 `all` 与 `unresolved` 均将两个 HFD-vs-normal 对比精确排为 Top-2，而两个 sex contrasts 排为 3/4；composition 在四个候选间相同。31 次响应重排中仅 3 次得到同等或更大分离边际。该 exact-intervention 正控排除了简单来源识别，但 known-only 因证据不足弃权，故仍不是 matched unknown increment。Post-hoc obesity 泛化为 all 0/4、known 0/4、unresolved 1/4 exact Top-2，表明该能力尚未普适化。

同一 centered operator 在冻结 official DreaMS embedding 上复现 HFD `all/unresolved` 精确 Top-2，重排边际同样仅 3/31 达到或超过观察值；official obesity 仍仅 all 1/4、known 0/4、unresolved 1/4。由此排除“仅粗 fragment bins 有效”，但不改变局部正控而非通用成功的边界。Noise V1 只允许通过固定服务器入口在同一实体、同一终点上比较。
