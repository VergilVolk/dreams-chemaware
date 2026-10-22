# B-anchored dual-encoder and multi-reranker integration

Date: 2026-10-22

## Objective and fixed boundary

This is the algorithm-agent implementation of the agreed integration plan.  It
keeps the selected WSE/Noise/P2b/ChemAware-V2 model B fixed and asks only
whether the remaining frozen assets add formula-OOF development value on the
same GNPS candidate axis.  GNPS and MassSpecGym are consumed development data;
Enveda is also consumed.  This job reads none of them as a new external claim
and cannot authorize external generalization.

## Asset fidelity

| Asset | Actual computation in this implementation | Qualification |
|---|---|---|
| WSE, Noise V1, frozen P2b | inherited from the exact B OOF score and B checkpoint | enabled; P2b enters once |
| ChemAware V2 continuous evidence | inherited from B; A is replayed as the no-V2 comparator | development-only, no hard action |
| ChemAware Stage-1 | independently encoded and scored on every frozen candidate edge | role-3 confirmed internally; GNPS transfer previously uncertain |
| ChemAware Phase-A | independently encoded and scored; never renamed Stage-1 | role-2 development-only |
| Noise MSG stack | frozen formula-OOF model, applied through its existing GNPS pair-score cache | provisional |
| Noise RRF 1.7 | exact historical fixed recipe exported to the common pair axis | retired/audit-only; explicit last-step ablation only |
| BioAware | not fabricated on library queries; candidate-keyed algorithm scores are exported for the biology agent's real-context comparison | unavailable until a real event ledger and qualified action exist |

Stage-1 and Phase-A are not parameter-merged.  Their candidate evidence is
represented separately relative to Noise V1.  MSG and RRF are represented
relative to the already-counted P2b composite.  Consequently, adding a family
does not insert another unconditional copy of the same absolute spectral
signal.

## Fixed comparison matrix

The entrypoint cross-fits the following arms by formula hash, using the parent
B epochs, hidden width, optimizer rate and trust penalty without a grid:

1. Stage-1 only versus B;
2. Phase-A only versus B;
3. both encoders versus B;
4. both encoders plus MSG versus both encoders;
5. an audit-only extension with historical RRF versus the MSG system;
6. the complete deployable spectral system versus a same-input linear additive control;
7. the complete deployable spectral system versus B and WSE;
8. the complete deployable spectral system versus an A-anchor replay using the identical learned
   incremental residual (V2 removal while holding new evidence fixed).

Every learned score in the comparison is formula-OOF.  Upstream MSG is allowed
only as the already-frozen formula-OOF parent model applied to GNPS without
GNPS labels.  Candidate and query IDs must exactly match the parent B ledger.

Promotion is narrow: the complete spectral arm (dual encoder plus MSG, without
the retired RRF) is fitted as a development candidate only if its Recall@1
improvement over B has a positive
formula-cluster bootstrap lower bound on both GNPS panels.  Failure retains B;
it does not trigger a new architecture or tuning loop.  Historical RRF status
does not change even if its point estimate happens to help.

## Executable artifacts

- `tasks/run_b_anchored_dual_encoder_multireranker_gnps.sbatch`: one-GPU job;
- `tasks/train_b_anchored_dual_encoder_multireranker.py`: cross-fit, metrics,
  per-candidate ledger, ablations and conditional development freeze;
- `tasks/export_gnps_stage1_rrf_pair_scores.py`: faithful RRF cache adapter;
- `tasks/validate_gnps_panel_candidate_axis.py`: semantic edge-axis equality
  check between the historical cache panel and the reconstructed B panel;
- `tasks/apply_b_anchored_dual_encoder_multireranker.py`: label-free inference
  from the B checkpoint plus the selected integrated residual;
- `tasks/test_b_anchored_dual_encoder_multireranker.py`: CPU and sbatch contracts.

The server command is:

```bash
sbatch tasks/run_b_anchored_dual_encoder_multireranker_gnps.sbatch
```

The ordinary command automatically finds the single directory containing the
frozen MSG `report.json`; files inside that directory are not counted as
separate caches.  If genuinely multiple historical MSG cache directories
exist, discovery stops rather than silently selecting one.  An explicit cache
can then be supplied where the cluster permits exported variables:

```bash
sbatch --export=ALL,MSG_FUSION_GNPS=data/validation/noise_msg_fusion_run_<job>/gnps_fusion_scores \
  tasks/run_b_anchored_dual_encoder_multireranker_gnps.sbatch
```

The result directory contains the extended score bundle, both aligned evidence
bundles, RRF cache, full report and `modeling/candidate_predictions.npz`.  The
latter preserves B, A, WSE and every new arm on the exact same candidate axis,
which is the handoff required for unchanged-condition biological evaluation.
