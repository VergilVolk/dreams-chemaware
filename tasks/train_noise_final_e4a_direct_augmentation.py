"""E4-A: direct peak-noise fine-tuning of the shared DreaMS embedding.

This stage deliberately has no P2b score or post-embedding reranker.  It has
two mutually explicit action-selection modes.  ``fixed`` uses two previously
frozen S3A policies exactly like image augmentations:

* candidate_gradient, attenuation 0.50, terminal step 6;
* role_confounder, attenuation 1.00, terminal step 5.

``outcome_mined`` uses one correction-producing raw-spectrum intervention per
query, selected only inside the non-held formula partition from the frozen
S3A+A4 action matrix.  Outcome fields select the training augmentation and are
then stripped; they never become a loss, sample weight, evaluation input or
inference input.

For a training query the clean and perturbed spectra are both encoded by the
same trainable DreaMS model.  Positive and negative reference spectra are also
encoded by that model.  The objective combines clean groupwise ranking,
perturbed-view groupwise ranking, clean/perturbed consistency, an official
margin floor and clean-embedding preservation.  At inference the saved model
receives only an ordinary clean spectrum and emits one new embedding.

Formula folds are fixed before training.  The held formula fold is evaluated
once after a fixed epoch count and is never used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything,
    sha256_file, stable_fold, strict_rank,
)
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, evaluate_embeddings, formula_bootstrap_delta,
    forward_embeddings, margins, parse_controls, parse_path, representatives,
)
from noise_v3_core import attenuate_sequence  # noqa: E402
from noise_final_e4_pmt_core import (  # noqa: E402
    coverage_first_identity_balanced_schedules, coverage_first_schedules,
    identity_family_balanced_exposure_weights,
    select_materialized_direct_action_panel,
)
from noise_final_candidate_boundary_core import (  # noqa: E402
    candidate_boundary_objective, candidate_safety_objective,
)
from noise_final_direct_boundary_v2_core import (  # noqa: E402
    direct_boundary_objective,
)
from noise_corrected_shuffled_control_v3 import (  # noqa: E402
    source_family_shuffled_action_bank,
)
from noise_corrected_action_routing_v3 import (  # noqa: E402
    score_candidate_boundary,
)
from noise_e4_action_injector_v1_bridge import (  # noqa: E402
    E4ActionInjectorV1Bridge,
    E4ActionInjectorV1Step,
    summarize_e4_action_injector_v1_steps,
)
from noise_e4_signal_preserving_hybrid_v2 import (  # noqa: E402
    E4SignalPreservingInjectorBridgeV2,
    SignalPreservingInjectionV2Step,
    bounded_semantic_microbatches,
    build_identity_equal_action_bag_plan,
    flatten_weighted_action_bags,
    query_local_e4_semantic_residual,
    summarize_signal_preserving_v2_steps,
)
from noise_e4_live_shared_hybrid_v3 import (  # noqa: E402
    QueryEqualActionPlanV3,
    SeparatedE4ActionInjectorV3,
    SeparatedE4ActionStepV3,
    build_query_equal_action_plan_v3,
    live_shared_e4_action_objective_v3,
    summarize_separated_e4_action_steps_v3,
)
from noise_corrected_fullgraph_evaluation import (  # noqa: E402
    full_metrics as corrected_full_metrics,
    official_scores as corrected_official_scores,
    paired_outcome_table as corrected_paired_outcome_table,
    score_embedding_query_subset as corrected_score_embedding_query_subset,
)
from evaluate_noise_e4_best_actions_injector_v1_final import (  # noqa: E402
    _registered_metric_strict_or_boundary_failures,
    _registered_metric_violations,
)
from audit_noise_final_positive_guided_matrix import (  # noqa: E402
    apply_action as apply_positive_intensity_action,
    reference_profile,
)
from audit_noise_final_positive_peak_transfer import (  # noqa: E402
    apply_transfer as apply_positive_peak_transfer,
    recurrent_missing_peaks,
)


FIXED_POLICY = {
    "candidate": (("candidate_gradient", 0.50, 6),),
    "confounder": (("role_confounder", 1.00, 5),),
    "combined": (
        ("candidate_gradient", 0.50, 6),
        ("role_confounder", 1.00, 5),
    ),
    # Every cell below already exists in the frozen, outcome-free R0 table.
    # This is a dose curriculum, not per-query post-outcome action selection.
    "curriculum": (
        ("candidate_gradient", 0.50, 3),
        ("candidate_gradient", 0.50, 4),
        ("candidate_gradient", 0.50, 5),
        ("candidate_gradient", 0.50, 6),
        ("role_confounder", 1.00, 1),
        ("role_confounder", 1.00, 2),
        ("role_confounder", 1.00, 3),
        ("role_confounder", 1.00, 4),
        ("role_confounder", 1.00, 5),
    ),
}


def load_frozen_r0_policy_actions(
    r0_dir: Path, policy: str,
) -> pd.DataFrame:
    """Load the immutable nine-cell R0 action supplier used by historical E4."""
    actions = pd.read_csv(r0_dir / "training_actions.csv.gz")
    forbidden_columns = {
        "corrected", "introduced", "target_rank", "target_margin", "random_margin"
    }
    leaked = forbidden_columns.intersection(actions.columns)
    if leaked:
        raise RuntimeError(
            "post-outcome columns leaked into frozen R0 training: "
            f"{sorted(leaked)}"
        )
    selected: list[pd.DataFrame] = []
    for selector, attenuation, step in FIXED_POLICY[policy]:
        block = actions.loc[
            actions["selector"].astype(str).eq(selector)
            & np.isclose(actions["attenuation"].astype(float), attenuation)
            & actions["step"].astype(int).eq(step)
        ].copy()
        if block.empty:
            raise RuntimeError(
                f"missing frozen R0 policy cell {selector}|{attenuation}|{step}"
            )
        selected.append(block)
    return pd.concat(selected, ignore_index=True)


def load_corrected_e4_base_actions(
    action_dir: Path,
    graph: CandidateGraph,
    *,
    outer_fold: int,
    formula_fold_seed: int,
    initial_checkpoint: Path,
    graph_path: Path,
    embedding_cache: Path,
    data_path: Path,
    official_checkpoint: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the outcome-free E4 mechanism bank built on the corrected graph.

    This is deliberately not the old R0 table: its row/query namespace belongs
    to an earlier graph.  The corrected bank re-mines the same nine registered
    E4 action cells in the exact E8 geometry used by the later seven-source
    action ledger.
    """
    report_path = action_dir / "report.json"
    actions_path = action_dir / "training_actions.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contracts = report.get("contracts", {})
    provenance = report.get("provenance", {})
    model_provenance = report.get("model_provenance", {})
    expected_registered_actions = {
        "candidate_gradient": {
            "attenuation": 0.5,
            "maximum_steps": 6,
            "publish_steps": [3, 4, 5, 6],
        },
        "role_confounder": {
            "attenuation": 1.0,
            "maximum_steps": 5,
            "publish_steps": [1, 2, 3, 4, 5],
        },
    }
    expected_counts = {
        "source_queries": 65286,
        "action_rows": 190324,
        "action_queries": 44623,
        "action_identities": 7624,
        "action_formulas": 4825,
    }
    expected_configuration = {
        "formula_fold_seed": 20260825,
        "n_highest_peaks": 100,
        "top_k_negatives": 5,
        "softmax_temperature": 0.1,
        "fragment_tolerance": 0.02,
        "control_repeats": 2,
        "seed": 20260906,
        "max_queries": 0,
        "amp": False,
    }
    if (
        report.get("status") != "noise_corrected_full_action_bank_complete"
        or report.get("formal") is not True
        or report.get("formal_training_authorized") is not True
        or int(report.get("outer_formula_fold", -1)) != outer_fold
        or report.get("development_query_scope") != "all"
        or report.get("configuration") != expected_configuration
        or report.get("registered_actions") != expected_registered_actions
        or any(int(report.get(key, -1)) != value for key, value in expected_counts.items())
        or contracts.get("outer_held_formulas_published") is not False
        or contracts.get(
            "registered_formal_action_bank_configuration_verified"
        ) is not True
        or contracts.get("current_geometry_remined_each_step") is not True
        or contracts.get("candidate_context_recomputed_each_step") is not True
        or contracts.get("target_action_eligibility_independent_of_matched_controls") is not True
        or contracts.get("available_matched_controls_complete_and_distinct") is not True
        or contracts.get("action_multiplicity_is_not_training_dose") is not True
        or contracts.get("action_outcomes_computed") is not False
        or contracts.get("teacher_embedding_or_margin_used") is not False
        or contracts.get("P2b") != "forbidden"
        or contracts.get("P3_consumed") is not False
        or provenance.get("script_sha256")
        != "dafffea08745e59a5af4f37324858bb0472283d9d2b9d30a8bf411d6ee6f7d1d"
    ):
        raise RuntimeError("corrected E4 base action-bank contract failed")
    expected_hashes = {
        "candidate_graph_sha256": sha256_file(graph_path),
        "embedding_cache_sha256": sha256_file(embedding_cache),
        "hdf5_sha256": sha256_file(data_path),
    }
    if any(provenance.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("corrected E4 base provenance does not match this graph/data")
    if (
        model_provenance.get("official_checkpoint_sha256")
        != sha256_file(official_checkpoint)
        or model_provenance.get("initial_student_checkpoint_sha256")
        != sha256_file(initial_checkpoint)
    ):
        raise RuntimeError("corrected E4 base was not mined in the requested E8 geometry")

    actions = pd.read_csv(actions_path, low_memory=False)
    required_columns = {
        "action_id", "query_index", "query_row", "query_ik14",
        "query_formula", "formula_fold", "selector", "attenuation",
        "step", "target_path", "hard_negative_row",
    }
    if missing := required_columns - set(actions.columns):
        raise RuntimeError(f"corrected E4 base lacks columns: {sorted(missing)}")
    forbidden = {
        "corrected", "introduced", "target_rank", "target_margin",
        "random_margin", "teacher_margin", "action_rank", "action_margin",
    }
    if forbidden & set(actions.columns) or actions["action_id"].duplicated().any():
        raise RuntimeError("corrected E4 base is duplicated or outcome-contaminated")
    if (
        len(actions) != int(report.get("action_rows", -1))
        or actions["query_index"].nunique() != int(report.get("action_queries", -1))
        or actions["query_ik14"].nunique() != int(report.get("action_identities", -1))
        or actions["query_formula"].nunique() != int(report.get("action_formulas", -1))
    ):
        raise RuntimeError("corrected E4 base counts disagree with its report")
    query = actions["query_index"].to_numpy(np.int64)
    if np.any(query < 0) or np.any(query >= graph.n_queries):
        raise RuntimeError("corrected E4 base query index is outside the graph")
    if (
        not np.array_equal(actions["query_row"].to_numpy(np.int64), graph.query_row[query])
        or not np.array_equal(
            actions["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query],
        )
        or not np.array_equal(
            actions["query_formula"].astype(str).to_numpy(), graph.query_formula[query],
        )
    ):
        raise RuntimeError("corrected E4 base row/identity/formula namespace drifted")
    expected_fold = actions["query_formula"].astype(str).map(
        lambda value: stable_fold(value, 5, formula_fold_seed)
    ).to_numpy(np.int8)
    if (
        not np.array_equal(expected_fold, actions["formula_fold"].to_numpy(np.int8))
        or np.any(expected_fold == outer_fold)
    ):
        raise RuntimeError("corrected E4 base leaked the held formula fold")
    registered = {
        ("candidate_gradient", 0.50, step) for step in (3, 4, 5, 6)
    } | {
        ("role_confounder", 1.00, step) for step in (1, 2, 3, 4, 5)
    }
    observed_cells = {
        (str(row.selector), round(float(row.attenuation), 2), int(row.step))
        for row in actions[["selector", "attenuation", "step"]].itertuples(index=False)
    }
    if observed_cells != registered:
        raise RuntimeError(
            f"corrected E4 base action cells drifted: {sorted(observed_cells)}"
        )
    return actions, report

# A stored strict rank can be a machine-precision tie.  Such an action has no
# robust corrective content to inject and can flip sign solely because the
# training microbatch changes fp32 reduction order.
MATERIALIZED_ACTION_REPLAY_MARGIN_FLOOR = 5e-6
# Job 2333052 clipped every optimizer step.  Its observed clip scale was
# approximately 0.155--0.173, so 0.16 is the preregistered static scale that
# restores loss magnitude before clipping without changing any branch ratio.
# This is deliberately not adapted from held performance or per-step gradients.
MATERIALIZED_MULTI_ACTION_PRECLIP_LOSS_SCALE = 0.16
OFFICIAL_ZERO_CHANGE_SCORE_ATOL = 5e-4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument(
        "--source-manifest", type=Path, default=None,
        help=(
            "Corrected-graph manifest.npz containing query_adduct. Required by "
            "materialized_routed so all-adduct and [M+H]+ pooled pairwise metrics "
            "are evaluated together with retrieval metrics."
        ),
    )
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument(
        "--positive-manifest-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_pn_positive_manifest",
        help="Frozen strict cross-condition P-arm manifest. Used only when positive-stream-weight > 0.",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--initial-student-checkpoint", type=Path, default=None,
        help=(
            "Optional mature E4-A checkpoint for residual continuation. Its outer fold "
            "must equal --outer-fold; preservation and incremental evaluation are then "
            "anchored to this checkpoint while official DreaMS remains the primary baseline."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4a_direct")
    parser.add_argument(
        "--causal-arm",
        choices=("legacy", "clean_duplicate", "matched_random", "targeted"),
        default="legacy",
        help=(
            "Strict E4-A attribution factor. clean_duplicate repeats the clean query as "
            "the action view; matched_random uses one of the two frozen R0 matched-control "
            "paths selected without outcomes; targeted uses the frozen R0 target path."
        ),
    )
    parser.add_argument("--policy", choices=tuple(FIXED_POLICY), default="candidate")
    parser.add_argument(
        "--action-selection",
        choices=("fixed", "outcome_mined", "materialized_routed"),
        default="fixed",
        help=(
            "fixed uses globally preregistered S3A cells; outcome_mined uses one "
            "train-fold correction-mined raw-spectrum action per query."
        ),
    )
    parser.add_argument(
        "--outcome-action-dir", type=Path, default=None,
        help="Directory containing report.json and corrective_teacher_actions.csv.gz for outcome_mined mode.",
    )
    parser.add_argument(
        "--materialized-action-dir", type=Path, default=None,
        help=(
            "Frozen corrected-graph routed ledger containing training_actions.csv.gz, "
            "action_spectra.npz and report.json. Only corrective rows enter the historical "
            "E4 direct objective; no action margin or embedding is used as a teacher target."
        ),
    )
    parser.add_argument(
        "--e4-base-action-dir", type=Path, default=None,
        help=(
            "Formal outcome-free full E4 action bank built on the same corrected "
            "graph and frozen E8 initialization. Required only by e4_base_semantic_v2."
        ),
    )
    parser.add_argument(
        "--materialized-action-arm",
        choices=("targeted", "shuffled", "clean_duplicate"),
        default="targeted",
        help=(
            "Action-view-only causal arm for materialized_routed training. shuffled keeps "
            "source/family/exact-recipe semantics but changes query; clean_duplicate replaces "
            "the action tensor with the ordinary clean query."
        ),
    )
    parser.add_argument(
        "--materialized-injection-mode",
        choices=(
            "one_best_e4",
            "multi_action_balanced",
            "complete_panel_historical_e4",
            "e4_base_semantic_v2",
            "e4_live_shared_v3",
        ),
        default="one_best_e4",
        help=(
            "one_best_e4 preserves the historical maximum-margin action per query. "
            "multi_action_balanced exposes every strict corrective action once, "
            "normalizes total dose by identity and source/family, and stops spending "
            "action-rank gradient after the action clears the E4 rank margin. "
            "complete_panel_historical_e4 keeps every strict action but restores the "
            "ungated, unit-weight historical E4 objective. e4_base_semantic_v2 keeps "
            "the corrected-graph full E4 mechanism stream intact and injects all later "
            "actions only as "
            "identity-equal, query-local semantic residuals. e4_live_shared_v3 "
            "restores the complete live E4 clean/action/positive/negative relation "
            "and uses query-equal, non-mixed action opportunities."
        ),
    )
    parser.add_argument(
        "--optimizer-boundary-mode",
        choices=(
            "ordinary_adamw", "action_injector_v1", "signal_preserving_v2",
            "separated_e4_action_v3",
        ),
        default="ordinary_adamw",
        help=(
            "Keep the historical ordinary AdamW step or pass the unchanged E4 "
            "action/safety gradient ledgers through the frozen ActionInjectorV1. "
            "signal_preserving_v2 uses intact historical E4 as the protective baseline "
            "and attributes only the query-local later-action semantic residual. "
            "separated_e4_action_v3 owns independent E4-only and later full-E4 "
            "action-stream AdamW states, then reaches the registered exact action "
            "fraction subject to E4 projection and total-norm safety bounds."
        ),
    )
    parser.add_argument(
        "--injector-target-attributable-fraction", type=float, default=0.25,
    )
    parser.add_argument(
        "--injector-minimum-protective-retention", type=float, default=0.90,
    )
    parser.add_argument(
        "--injector-maximum-update-norm-ratio", type=float, default=1.50,
    )
    parser.add_argument("--action-scope", choices=("errors", "all"), default="errors")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-actions", type=int, default=4)
    parser.add_argument("--views-per-identity", type=int, default=2)
    parser.add_argument(
        "--error-views-per-identity", type=int, default=0,
        help="Additional N-arm views per baseline-error identity; zero preserves the validated baseline sampler.",
    )
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-clean-rank", type=float, default=1.0)
    parser.add_argument("--lambda-aug-rank", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.25)
    parser.add_argument(
        "--direct-transfer-mode",
        choices=("symmetric", "student_action_stopgrad", "official_action"),
        default="symmetric",
        help=(
            "How the clean spectrum receives the action-view signal. symmetric preserves "
            "the historical E4-A objective; student_action_stopgrad prevents the clean "
            "target from being erased by the consistency gradient; official_action uses "
            "the frozen official encoder's action embedding as a fixed training-only target."
        ),
    )
    parser.add_argument(
        "--rank-reference-mode", choices=("shared", "official"), default="shared",
        help=(
            "shared preserves historical end-to-end rank gradients. official anchors rank "
            "losses to frozen official reference embeddings so the optimizer cannot solve "
            "a corrective query merely by moving its candidate references."
        ),
    )
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument(
        "--safety-ratio", type=float, default=1.0,
        help="Number of safety examples per action example; changes coverage, not loss magnitude.",
    )
    parser.add_argument(
        "--safety-stream-weight", type=float, default=1.0,
        help="Explicit multiplier on the mean safety loss. Use this, not safety-ratio, to strengthen safety gradients.",
    )
    parser.add_argument("--positive-stream-weight", type=float, default=0.0)
    parser.add_argument("--positive-ratio", type=float, default=1.0)
    parser.add_argument("--positive-views-per-identity", type=int, default=2)
    parser.add_argument("--lambda-positive-rank", type=float, default=1.0)
    parser.add_argument("--lambda-positive-margin-floor", type=float, default=2.0)
    parser.add_argument(
        "--guided-noise-policy", choices=("none", "intensity", "transfer", "both", "selected"),
        default="none",
        help="Fixed real-positive-guided peak-noise stream; independent of the legacy P pair stream.",
    )
    parser.add_argument(
        "--guided-query-scope", choices=("positive_deficit_errors", "all"),
        default="positive_deficit_errors",
        help=(
            "Queries receiving the fixed guided action. all is an outcome-free noise "
            "augmentation and requires a formal action-matrix authorization."
        ),
    )
    parser.add_argument(
        "--guided-intensity-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_positive_guided_matrix",
    )
    parser.add_argument(
        "--guided-transfer-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_positive_peak_transfer",
    )
    parser.add_argument(
        "--guided-action-authorization-dir", type=Path, default=None,
        help=(
            "Formal action-matrix artifact authorizing a non-historical guided recipe. "
            "E13 uses the frozen E12-B top3 recurrence result here."
        ),
    )
    parser.add_argument(
        "--guided-crossfit-root", type=Path, default=None,
        help=(
            "Root containing fold_0..fold_4 E14 selected_actions.csv.gz and report.json. "
            "Required only for guided-noise-policy=selected. The current outer fold is "
            "excluded fail-closed."
        ),
    )
    parser.add_argument(
        "--guided-reference-checkpoint", type=Path, default=None,
        help=(
            "Frozen shared encoder used only to choose top3 real same-identity reference "
            "spectra. Required by the E12-B-authorized recipe."
        ),
    )
    parser.add_argument(
        "--error-signatures", type=Path,
        default=ROOT / "data/validation/g8r_real_error_analysis/query_error_signatures.csv.gz",
    )
    parser.add_argument("--guided-noise-weight", type=float, default=1.0)
    parser.add_argument(
        "--guided-auto-balance", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Scale the selected P branch once at initialization so its gradient norm "
            "does not exceed the combined mature N+safety gradient norm."
        ),
    )
    parser.add_argument("--guided-noise-ratio", type=float, default=1.0)
    parser.add_argument("--guided-noise-views-per-identity", type=int, default=2)
    parser.add_argument(
        "--guided-risk-control-ratio", type=float, default=0.0,
        help=(
            "Action-specific mature-clean-correct controls per selected corrective "
            "example. Available only in E14 selected mode."
        ),
    )
    parser.add_argument("--lambda-guided-transfer", type=float, default=0.50)
    parser.add_argument(
        "--lambda-guided-teacher-margin", type=float, default=0.0,
        help="Weight of the frozen crossfit teacher-margin floor in selected E14 mode.",
    )
    parser.add_argument("--guided-teacher-margin-cap", type=float, default=0.20)
    parser.add_argument(
        "--guided-teacher-target-mode", choices=("absolute", "delta"),
        default="absolute",
        help=(
            "absolute replays the action margin; delta transfers only a conservative "
            "fraction of action-minus-clean margin."
        ),
    )
    parser.add_argument("--guided-teacher-delta-fraction", type=float, default=0.50)
    parser.add_argument("--guided-teacher-delta-cap", type=float, default=0.20)
    parser.add_argument(
        "--guided-transfer-mode", choices=("stopgrad", "symmetric"), default="stopgrad",
        help=(
            "Gradient convention for clean/action consistency in the guided P stream. "
            "symmetric updates both branches of the one shared encoder; stopgrad is the "
            "historical E7 mechanism control."
        ),
    )
    parser.add_argument(
        "--guided-recurrence-prevalence", type=float, default=0.67,
        help="Minimum positive-reference prevalence for a recurrent missing peak.",
    )
    parser.add_argument(
        "--guided-recurrence-max-peaks", type=int, default=5,
        help="Maximum recurrent positive peaks transferred into one action view.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--pmt-manifest-dir", type=Path, default=None,
        help="Strict N-only E4-PMT manifest. Activates identical action membership across PMT arms.",
    )
    parser.add_argument(
        "--pmt-arm", choices=("none", "clean_duplicate", "matched_random", "paired_target"),
        default="none",
    )
    parser.add_argument("--pmt-alpha", type=float, default=0.0)
    parser.add_argument("--pmt-advantage-cap", type=float, default=0.10)
    parser.add_argument("--pmt-preference-gap", type=float, default=0.01)
    parser.add_argument("--lambda-pmt-preference", type=float, default=1.0)
    parser.add_argument(
        "--candidate-boundary-loss", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Replace scalar PMT with direct positive-reference x negative-molecule "
            "edge training. Requires pmt-arm=paired_target."
        ),
    )
    parser.add_argument(
        "--candidate-boundary-version", choices=("v1_edges", "v2_molecule_max"),
        default="v1_edges",
        help=(
            "v2_molecule_max makes the clean evaluator boundary the primary "
            "corrective target and treats target/control only as a detached router."
        ),
    )
    parser.add_argument("--boundary-advantage-temperature", type=float, default=0.02)
    parser.add_argument("--boundary-hard-temperature", type=float, default=0.10)
    parser.add_argument("--boundary-topk-negatives", type=int, default=8)
    parser.add_argument("--lambda-boundary-clean", type=float, default=1.0)
    parser.add_argument("--lambda-boundary-target", type=float, default=1.0)
    parser.add_argument("--lambda-boundary-counterfactual", type=float, default=1.0)
    parser.add_argument("--lambda-boundary-full-clean", type=float, default=0.10)
    parser.add_argument(
        "--lambda-boundary-action-safety", type=float, default=0.25,
        help=(
            "One-sided safety weight for routed non-corrective raw actions. "
            "Their corrective weight remains exactly zero."
        ),
    )
    parser.add_argument(
        "--boundary-auto-balance", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "At initialization, estimate corrective and safety gradient norms on "
            "32 formula-stratified microbatches. v1 only boosts; v2 may downscale "
            "an oversized corrective branch, with scale=1 when safety is inactive."
        ),
    )
    parser.add_argument("--boundary-calibration-formulas", type=int, default=32)
    parser.add_argument("--boundary-calibration-examples", type=int, default=4)
    parser.add_argument("--boundary-action-to-safety-norm-ratio", type=float, default=1.0)
    parser.add_argument("--boundary-action-scale-cap", type=float, default=4.0)
    parser.add_argument(
        "--refresh-hard-negatives", action=argparse.BooleanOptionalAction, default=False,
        help="Refresh clean-geometry top-k negative molecules before every epoch.",
    )
    parser.add_argument(
        "--run-suffix", default="",
        help="Optional filesystem-safe suffix for preregistered optimizer scans.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class DirectExample:
    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    official_margin: float
    official_rank: int
    sample_weight: float
    policy: str = "clean_safety"
    target_path: tuple[int, ...] = ()
    control_path: tuple[int, ...] = ()
    teacher_advantage: float = 0.0
    forced_negative_row: int = -1
    forced_positive_row: int = -1
    attenuation: float = 0.0
    action_id: str = ""
    materialized_action_index: int = -1


@dataclass(frozen=True)
class GuidedNoiseExample:
    """Outcome-free, real-positive-guided training augmentation."""

    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    action_reference_rows: tuple[int, ...]
    official_margin: float
    official_rank: int
    sample_weight: float
    policy: str
    family: str
    dose: float
    auxiliary_dose: float = 0.0
    recurrence_prevalence: float = 0.67
    recurrence_max_peaks: int = 5
    support_weighted: bool = False
    teacher_margin: float = float("nan")
    teacher_margin_delta: float = float("nan")
    supervision_kind: str = "corrective"


def strict_bool(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{name} is not a strict boolean column")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def parse_reference_rows(value: object) -> tuple[int, ...]:
    rows = tuple(int(part) for part in str(value).split(";") if str(part).strip())
    if not rows or len(rows) != len(set(rows)):
        raise RuntimeError("guided action reference rows must be non-empty and unique")
    return rows


def official_rank_margin(graph: CandidateGraph) -> tuple[np.ndarray, np.ndarray]:
    score = graph.features[:, graph.dreams_column]
    molecule_score = np.maximum.reduceat(score, graph.molecule_ptr[:-1])
    rank = np.empty(graph.n_queries, dtype=np.int16)
    margin = np.empty(graph.n_queries, dtype=np.float32)
    for query in range(graph.n_queries):
        left, right = map(int, graph.query_ptr[query:query + 2])
        values = molecule_score[left:right]
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid frozen candidate scores for query {query}")
        rank[query] = strict_rank(values)
        margin[query] = float(values[0] - np.max(values[1:]))
    return rank, margin


def _stable_control_index(row: pd.Series) -> int:
    """Choose one of two frozen matched controls without using any outcome."""
    key = (
        f"{int(row['query_index'])}|{str(row['selector'])}|"
        f"{float(row['attenuation']):.8f}|{int(row['step'])}|e4a-causal-v1"
    )
    return int(hashlib.sha256(key.encode("utf-8")).digest()[0] & 1)


def materialize_causal_arm(
    actions: pd.DataFrame, arm: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Materialize the sole experimental difference in the attribution trial.

    The returned table retains every row, query, policy label, hard negative and
    sampling key.  Only ``target_path`` changes.  This guarantees that the
    identity-balanced sampler and all candidate references are arm-invariant.
    """
    if arm == "legacy":
        return actions, {"arm": arm, "rows": int(len(actions))}
    required = {
        "query_index", "selector", "attenuation", "step", "target_path",
        "matched_control_paths", "hard_negative_row", "query_ik14", "query_formula",
    }
    missing = required - set(actions.columns)
    if missing:
        raise RuntimeError(f"causal attribution actions lack columns: {sorted(missing)}")
    output = actions.copy()
    control_indices: list[int] = []
    selected_paths: list[str] = []
    target_paths: list[tuple[int, ...]] = []
    for _, row in output.iterrows():
        target = parse_path(row["target_path"])
        controls = parse_controls(row["matched_control_paths"])
        step = int(row["step"])
        if len(target) != step or any(len(path) != step for path in controls):
            raise RuntimeError("target/control path length does not match the frozen step")
        if target in controls or controls[0] == controls[1]:
            raise RuntimeError("matched-control paths are not distinct from the target and each other")
        target_paths.append(target)
        if arm == "targeted":
            control_indices.append(-1)
            selected_paths.append(",".join(map(str, target)))
        elif arm == "matched_random":
            index = _stable_control_index(row)
            control_indices.append(index)
            selected_paths.append(",".join(map(str, controls[index])))
        elif arm == "clean_duplicate":
            control_indices.append(-1)
            selected_paths.append("")
        else:  # pragma: no cover - argparse and tests protect this branch
            raise ValueError(f"unknown causal arm: {arm}")
    output["target_path"] = selected_paths
    output["causal_control_index"] = np.asarray(control_indices, dtype=np.int8)
    canonical = "\n".join(
        f"{int(row.query_index)}|{row.selector}|{float(row.attenuation):.8f}|"
        f"{int(row.step)}|{row.target_path}|{int(row.causal_control_index)}"
        for row in output[[
            "query_index", "selector", "attenuation", "step", "target_path",
            "causal_control_index",
        ]].itertuples(index=False)
    )
    audit: dict[str, object] = {
        "arm": arm,
        "rows": int(len(output)),
        "queries": int(output["query_index"].nunique()),
        "identities": int(output["query_ik14"].astype(str).nunique()),
        "formulas": int(output["query_formula"].astype(str).nunique()),
        "materialized_path_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "target_rows_preserved": bool(len(target_paths) == len(output)),
    }
    if arm == "matched_random":
        counts = output["causal_control_index"].value_counts().sort_index()
        audit["matched_control_index_counts"] = {
            str(int(index)): int(value) for index, value in counts.items()
        }
        if set(map(int, counts.index)) != {0, 1}:
            raise RuntimeError("deterministic matched-control assignment did not use both controls")
        imbalance = abs(int(counts.loc[0]) - int(counts.loc[1])) / max(len(output), 1)
        audit["matched_control_assignment_imbalance"] = float(imbalance)
        # Small smoke panels have ordinary binomial variation; the tolerance
        # contracts to 2% for the full R0 ledger without outcome-dependent retry.
        allowed_imbalance = max(0.02, 2.5 / math.sqrt(max(len(output), 1)))
        audit["matched_control_assignment_max_imbalance"] = float(allowed_imbalance)
        if imbalance > allowed_imbalance:
            raise RuntimeError(
                "matched-control assignment is unexpectedly imbalanced: "
                f"{imbalance:.4f} > {allowed_imbalance:.4f}"
            )
    return output, audit


def full_graph_query_details(
    graph: CandidateGraph, rows: np.ndarray, encoded: np.ndarray, queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return query details with the exact bounded scorer used by the router.

    Do not expand query embeddings to the full spectrum-edge ledger here.  On
    the corrected graph that temporary is several gigabytes, and two E4 arms
    can cross the Slurm cgroup limit even though both models fit on their GPUs.
    Per-query matrix-vector products also preserve the V4 router's fp32 tie
    geometry, which is part of the immutable materialized-action contract.
    """
    position = {int(row): index for index, row in enumerate(rows)}
    ranks: list[int] = []
    top_local: list[int] = []
    margins_out: list[float] = []
    for query_value in np.asarray(queries, dtype=np.int64):
        query = int(query_value)
        _, candidate_rows, local_ptr, _ = graph.query_block(query)
        try:
            candidate_vectors = encoded[
                [position[int(row)] for row in candidate_rows]
            ]
            query_vector = encoded[position[int(graph.query_row[query])]]
        except KeyError as error:
            raise RuntimeError(
                f"embedding rows do not cover query {query} candidate block"
            ) from error
        score = score_candidate_boundary(
            candidate_rows, local_ptr, candidate_vectors, query_vector,
        )
        ranks.append(int(score.rank))
        # Candidate molecule 0 is first, so np.argmax keeps it when the best
        # negative is exactly tied; strict rank still (deliberately) counts the
        # tie as an error.
        top_local.append(
            int(score.hard_negative_molecule_index) if score.margin < 0 else 0
        )
        margins_out.append(float(score.margin))
    return (
        np.asarray(ranks, dtype=np.int16),
        np.asarray(top_local, dtype=np.int32),
        np.asarray(margins_out, dtype=np.float32),
    )


def streaming_evaluate_embeddings(
    graph: CandidateGraph, rows: np.ndarray, encoded: np.ndarray, queries: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Evaluate strict retrieval without a full-edge embedding expansion."""
    rank, _, _ = full_graph_query_details(graph, rows, encoded, queries)
    near = graph.query_has_near[np.asarray(queries, dtype=np.int64)]
    return rank, {
        "n_queries": int(len(rank)),
        "recall1": float(np.mean(rank == 1)),
        "mrr": float(np.mean(1.0 / rank)),
        "errors": int(np.sum(rank != 1)),
        "near_n": int(np.sum(near)),
        "near_recall1": float(np.mean(rank[near] == 1)) if np.any(near) else float("nan"),
    }


def graph_multimetric_summary(
    graph: CandidateGraph, rows: np.ndarray, encoded: np.ndarray, queries: np.ndarray,
) -> dict[str, float | int]:
    """Report complete-candidate metrics with bounded per-query storage."""
    position = {int(row): index for index, row in enumerate(rows)}
    ranks: list[int] = []
    macro_auc: list[float] = []
    macro_ap: list[float] = []
    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    reciprocal_candidate_percentile: list[float] = []
    positive_margins: list[float] = []
    top2_gaps: list[float] = []
    top1_correct: list[bool] = []
    for query_value in np.asarray(queries, dtype=np.int64):
        query = int(query_value)
        _, candidate_rows, local_ptr, _ = graph.query_block(query)
        try:
            candidate_vectors = encoded[
                [position[int(row)] for row in candidate_rows]
            ]
            query_vector = encoded[position[int(graph.query_row[query])]]
        except KeyError as error:
            raise RuntimeError(
                f"embedding rows do not cover multimetric query {query}"
            ) from error
        row_scores = np.asarray(candidate_vectors @ query_vector, dtype=np.float32)
        values = np.asarray(
            np.maximum.reduceat(row_scores, local_ptr[:-1]), dtype=np.float64,
        )
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid multimetric candidate scores for query {query}")
        labels = np.zeros(len(values), dtype=np.int8)
        labels[0] = 1
        rank = strict_rank(values)
        ranks.append(rank)
        positive = values[0]
        negatives = values[1:]
        macro_auc.append(float(
            (np.sum(positive > negatives) + 0.5 * np.sum(positive == negatives))
            / len(negatives)
        ))
        macro_ap.append(float(average_precision_score(labels, values)))
        reciprocal_candidate_percentile.append(float(
            1.0 - (rank - 1) / max(len(values) - 1, 1)
        ))
        order = np.argsort(-values, kind="stable")
        positive_margins.append(float(positive - np.max(negatives)))
        top2_gaps.append(float(values[order[0]] - values[order[1]]))
        top1_correct.append(bool(rank == 1))
        pooled_scores.append(values)
        pooled_labels.append(labels)
    rank_array = np.asarray(ranks, dtype=np.int64)
    labels_all = np.concatenate(pooled_labels)
    scores_all = np.concatenate(pooled_scores)
    top2_gap_array = np.asarray(top2_gaps, dtype=np.float64)
    correct_array = np.asarray(top1_correct, dtype=bool)
    confidence_order = np.argsort(-top2_gap_array, kind="stable")
    cumulative_error = np.cumsum(~correct_array[confidence_order]) / np.arange(
        1, len(correct_array) + 1,
    )
    report: dict[str, float | int] = {
        "queries": int(len(rank_array)),
        "candidate_molecules": int(len(scores_all)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
        "median_rank": float(np.median(rank_array)),
        "macro_query_auc": float(np.mean(macro_auc)),
        "macro_query_auprc": float(np.mean(macro_ap)),
        "micro_candidate_auc": float(roc_auc_score(labels_all, scores_all)),
        "micro_candidate_auprc": float(average_precision_score(labels_all, scores_all)),
        "mean_reciprocal_candidate_percentile": float(
            np.mean(reciprocal_candidate_percentile)
        ),
        "mean_positive_vs_best_negative_margin": float(np.mean(positive_margins)),
        "mean_top1_top2_gap": float(np.mean(top2_gap_array)),
        "selective_one_minus_aurc": float(1.0 - np.mean(cumulative_error)),
    }
    for target_error in (0.01, 0.05, 0.10):
        eligible = np.flatnonzero(cumulative_error <= target_error)
        accepted = int(eligible[-1] + 1) if len(eligible) else 0
        key = int(round(100 * target_error))
        report[f"coverage_at_empirical_top1_error_{key}pct"] = float(
            accepted / len(correct_array)
        )
        report[f"confidence_threshold_at_empirical_top1_error_{key}pct"] = (
            float(top2_gap_array[confidence_order[accepted - 1]])
            if accepted else float("inf")
        )
    for cutoff in (1, 2, 3, 5, 10, 20):
        report[f"recall{cutoff}"] = float(np.mean(rank_array <= cutoff))
    return report


def nested_numeric_delta(candidate: dict, reference: dict) -> dict:
    """Subtract matching numeric leaves in a nested registered metric panel."""
    output: dict[str, object] = {}
    for key, value in candidate.items():
        if key not in reference:
            continue
        baseline = reference[key]
        if isinstance(value, dict) and isinstance(baseline, dict):
            output[key] = nested_numeric_delta(value, baseline)
        elif (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and isinstance(baseline, (int, float)) and not isinstance(baseline, bool)
        ):
            output[key] = float(value - baseline)
    return output


def live_shared_v3_single_arm_promotion(
    corrected_metric_panel: dict[str, object],
    final_summary: dict[str, object],
    incremental_ci: dict[str, float],
) -> tuple[dict[str, bool], dict[str, object]]:
    """Judge one V3 arm against E8 without treating it as causal evidence."""
    if not corrected_metric_panel:
        raise RuntimeError(
            "live-shared V3 requires the complete corrected registered metric panel"
        )
    candidate = corrected_metric_panel["student"]
    initialization = corrected_metric_panel["initialization"]
    paired = corrected_metric_panel["paired_top1_vs_initialization"]
    near = paired["near"]
    violations = _registered_metric_violations(candidate, initialization)
    strict_failures = _registered_metric_strict_or_boundary_failures(
        candidate, initialization,
    )
    official_recall1 = float(
        corrected_metric_panel["official"]["retrieval"]["recall@1"]
    )
    student_recall1 = float(candidate["retrieval"]["recall@1"])
    gates = {
        "clean_recall_positive_vs_initial_e8": bool(
            final_summary["incremental_delta_recall1"] > 0
        ),
        "formula_ci_positive_vs_initial_e8": bool(incremental_ci["ci_low"] > 0),
        "corrected_gt_introduced_vs_initial_e8": bool(
            paired["corrected"] > paired["introduced"]
        ),
        "risk_net_lambda2_positive_vs_initial_e8": bool(
            paired["risk_net_lambda2"] > 0
        ),
        "near_risk_net_lambda2_positive_vs_initial_e8": bool(
            near["risk_net_lambda2"] > 0
        ),
        "all_registered_metrics_nonnegative_vs_initial_e8": bool(not violations),
        "all_registered_metrics_strict_or_boundary_vs_initial_e8": bool(
            not strict_failures
        ),
        "strict_four_pp_recall1_gain_vs_official": bool(
            student_recall1 - official_recall1 >= 0.04
        ),
        "preservation_ge_0_995": bool(
            final_summary["preservation_vs_initialization_mean"] >= 0.995
        ),
    }
    diagnostics = {
        "reference": "initialization",
        "registered_metric_violations": violations,
        "registered_metric_strict_or_boundary_failures": strict_failures,
        "paired_top1": paired,
        "formula_cluster_delta_recall1": incremental_ci,
        "recall1_gain_vs_official_pp": 100.0 * (
            student_recall1 - official_recall1
        ),
        "recall1_gain_vs_initial_e8_pp": 100.0 * float(
            final_summary["incremental_delta_recall1"]
        ),
    }
    return gates, diagnostics


def unfreeze_last_blocks(model, blocks: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    encoder = model.backbone.transformer_encoder
    if blocks < 1 or blocks > int(encoder.n_layers):
        raise ValueError(f"unfreeze-blocks must be in 1..{int(encoder.n_layers)}")
    layers = list(range(int(encoder.n_layers) - blocks, int(encoder.n_layers)))
    count = 0
    for layer in layers:
        for module in (encoder.atts[layer], encoder.ffs[layer],
                       encoder.scales[2 * layer], encoder.scales[2 * layer + 1]):
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    parameter.requires_grad = True
                    count += parameter.numel()
    if getattr(encoder, "pre_norm", False):
        for parameter in encoder.scales[-1].parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
                count += parameter.numel()
    return {
        "transformer_layers": int(encoder.n_layers),
        "unfrozen_layers": layers,
        "unfrozen_backbone_parameters": int(count),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def make_examples(graph: CandidateGraph, frame: pd.DataFrame, rank: np.ndarray,
                  margin: np.ndarray, positives: int, negatives: int,
                  action: bool) -> list[DirectExample]:
    if frame.empty:
        return []
    output: list[DirectExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        forced_value = (
            getattr(row, "hard_negative_row", None)
            if action else None
        )
        if forced_value is None and action:
            forced_value = getattr(row, "action_hard_negative_row", None)
        forced = (
            int(forced_value)
            if forced_value is not None and not pd.isna(forced_value) else None
        )
        positive_rows, negative_rows = representatives(
            graph, query, positives, negatives, forced,
        )
        if action and hasattr(row, "action_positive_row"):
            forced_positive_value = getattr(row, "action_positive_row")
            if pd.isna(forced_positive_value):
                raise RuntimeError(
                    f"materialized action for query {query} lacks its winning positive row"
                )
            forced_positive = int(forced_positive_value)
            _, candidate_rows, molecule_ptr, _ = graph.query_block(query)
            positive_set = set(map(int, candidate_rows[: int(molecule_ptr[1])]))
            if forced_positive not in positive_set:
                raise RuntimeError(
                    f"materialized action positive row {forced_positive} is not positive "
                    f"for query {query}"
                )
            # Keep the clean-active positive reference and append the
            # action-active reference when they differ. Replacing one with the
            # other corrupts one of the two margins before training.
            if forced_positive not in set(map(int, positive_rows)):
                positive_rows = tuple(positive_rows) + (forced_positive,)
            if forced is None:
                raise RuntimeError(
                    f"materialized action for query {query} lacks its hard negative row"
                )
            negative_set = set(map(int, candidate_rows[int(molecule_ptr[1]):]))
            if forced not in negative_set:
                raise RuntimeError(
                    f"materialized action hard negative row {forced} is not negative "
                    f"for query {query}"
                )
        identity = str(row.query_ik14)
        if action and hasattr(row, "action_tensor_index"):
            source = str(getattr(row, "source", "materialized"))
            family = str(getattr(row, "family", "unknown"))
            recipe = str(getattr(row, "recipe_id", getattr(row, "action_id", "unknown")))
            policy = f"{source}|{family}|{recipe}"
        else:
            policy = (
                f"{row.selector}|step={int(row.step)}" if action
                else "clean_safety"
            )
        output.append(DirectExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=identity,
            formula=str(row.query_formula),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            official_margin=float(margin[query]),
            official_rank=int(rank[query]),
            # PMT-v2 uses the frozen outer-train route as a binary corrective
            # weight. Non-corrective actions remain in the raw-action bank but
            # can only enter the one-sided safety term.
            sample_weight=(
                float(row.corrective_weight)
                if action and hasattr(row, "corrective_weight") else 1.0
            ),
            policy=policy,
            target_path=(
                parse_path(row.target_path)
                if action and hasattr(row, "target_path") else ()
            ),
            control_path=(
                parse_path(row.control_path)
                if action and hasattr(row, "control_path") else ()
            ),
            teacher_advantage=(
                float(row.teacher_advantage)
                if action and hasattr(row, "teacher_advantage") else 0.0
            ),
            forced_negative_row=forced if forced is not None else -1,
            forced_positive_row=(
                int(row.action_positive_row)
                if action and hasattr(row, "action_positive_row") else -1
            ),
            attenuation=(
                float(row.attenuation)
                if action and hasattr(row, "attenuation") else 0.0
            ),
            action_id=(
                str(row.action_id)
                if action and hasattr(row, "action_id") else ""
            ),
            materialized_action_index=(
                int(row.action_tensor_index)
                if action and hasattr(row, "action_tensor_index") else -1
            ),
        ))
    return output


def align_candidate_references_by_query(
    graph: CandidateGraph, examples: list[DirectExample], positives: int, negatives: int,
) -> list[DirectExample]:
    """Give every action of one query the same evaluator-aligned candidate set."""
    selected: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    output: list[DirectExample] = []
    for example in examples:
        query = int(example.query_index)
        if query not in selected:
            selected[query] = representatives(
                graph, query, positives, negatives, None,
            )
        positive_rows, negative_rows = selected[query]
        output.append(replace(
            example, positive_rows=positive_rows, negative_rows=negative_rows,
        ))
    return output


def make_positive_examples(graph: CandidateGraph, frame: pd.DataFrame,
                           rank: np.ndarray, margin: np.ndarray,
                           negatives: int) -> list[DirectExample]:
    """Build P-arm examples with one explicit real cross-condition positive."""
    output: list[DirectExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        _, negative_rows = representatives(graph, query, 1, negatives, None)
        output.append(DirectExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=str(row.query_ik14),
            formula=str(row.query_formula),
            positive_rows=(int(row.positive_row),),
            negative_rows=negative_rows,
            official_margin=float(margin[query]),
            official_rank=int(rank[query]),
            sample_weight=1.0,
            policy=f"positive|{row.relation}",
        ))
    return output


def make_guided_noise_examples(
    graph: CandidateGraph, frame: pd.DataFrame, rank: np.ndarray, margin: np.ndarray,
    positives: int, negatives: int,
) -> list[GuidedNoiseExample]:
    output: list[GuidedNoiseExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        if hasattr(row, "teacher_positive_row") and hasattr(row, "teacher_hard_negative_row"):
            # E14 stores the exact spectra that defined its molecular margin.
            # Replaying a different candidate subset silently changes the target.
            positive_rows = (int(row.teacher_positive_row),)
            negative_rows = (int(row.teacher_hard_negative_row),)
        else:
            positive_rows, negative_rows = representatives(
                graph, query, positives, negatives, None,
            )
        baseline_margin = float(getattr(
            row, "teacher_pair_clean_margin",
            getattr(row, "crossfit_clean_margin", margin[query]),
        ))
        baseline_rank = int(getattr(row, "crossfit_clean_rank", rank[query]))
        output.append(GuidedNoiseExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=str(row.query_ik14),
            formula=str(row.query_formula),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            action_reference_rows=parse_reference_rows(row.positive_reference_rows),
            official_margin=baseline_margin,
            official_rank=baseline_rank,
            sample_weight=1.0,
            policy=str(row.guided_policy),
            family=str(row.guided_family),
            dose=float(row.guided_dose),
            auxiliary_dose=float(getattr(row, "guided_auxiliary_dose", 0.0)),
            recurrence_prevalence=float(getattr(
                row, "guided_recurrence_prevalence", 0.67,
            )),
            recurrence_max_peaks=int(getattr(
                row, "guided_recurrence_max_peaks", 5,
            )),
            support_weighted=bool(getattr(row, "guided_support_weighted", False)),
            teacher_margin=float(getattr(row, "teacher_margin", float("nan"))),
            teacher_margin_delta=float(getattr(
                row, "teacher_margin_delta", float("nan"),
            )),
            supervision_kind=str(getattr(row, "control_kind", "corrective")),
        ))
    return output


def guided_variant(
    store: SpectrumStore, example: GuidedNoiseExample,
    recurrence_prevalence: float = 0.67, recurrence_max_peaks: int = 5,
) -> torch.Tensor:
    clean = store.one(example.query_row)
    references = [store.one(row) for row in example.action_reference_rows]
    prevalence, target = reference_profile(clean, references, 0.02)
    recurrence_prevalence = float(example.recurrence_prevalence)
    recurrence_max_peaks = int(example.recurrence_max_peaks)
    if not np.isfinite(recurrence_prevalence) or not 0 < recurrence_prevalence <= 1:
        raise RuntimeError("guided recurrence prevalence is invalid")
    if recurrence_max_peaks < 1:
        raise RuntimeError("guided recurrence maximum is invalid")
    if example.family in {"consensus_projection", "matched_intensity_transport"}:
        return apply_positive_intensity_action(
            clean, prevalence, target, example.family, example.dose,
        )
    if example.family in {
        "recurrent_union_mix", "recurrent_peak_graft", "balanced_peak_exchange",
    }:
        missing = recurrent_missing_peaks(
            clean, references, 0.02,
            recurrence_prevalence, recurrence_max_peaks,
        )
        if example.support_weighted and len(missing):
            missing = np.asarray(missing, dtype=np.float32).copy()
            missing[:, 1] *= missing[:, 2]
        variant, _ = apply_positive_peak_transfer(
            clean, missing, prevalence, example.family, example.dose,
        )
        return variant
    if example.family in {"transport_then_union", "consensus_then_union"}:
        first_family = (
            "matched_intensity_transport"
            if example.family == "transport_then_union"
            else "consensus_projection"
        )
        intensity = apply_positive_intensity_action(
            clean, prevalence, target, first_family, example.dose,
        )
        missing = recurrent_missing_peaks(
            clean, references, 0.02,
            recurrence_prevalence, recurrence_max_peaks,
        )
        variant, _ = apply_positive_peak_transfer(
            intensity, missing, prevalence, "recurrent_union_mix",
            example.auxiliary_dose,
        )
        return variant
    raise RuntimeError(f"unregistered guided action family: {example.family}")


def flatten_guided(
    store: SpectrumStore, examples: list[GuidedNoiseExample],
    recurrence_prevalence: float = 0.67, recurrence_max_peaks: int = 5,
):
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        item: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(example.query_row))
        clean_rows.append(example.query_row)
        item["action"] = len(tensors)
        tensors.append(guided_variant(
            store, example, recurrence_prevalence, recurrence_max_peaks,
        ))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def guided_consistency_values(
    clean_z: torch.Tensor, action_z: torch.Tensor, mode: str,
) -> torch.Tensor:
    if mode not in {"stopgrad", "symmetric"}:
        raise ValueError(f"unknown guided transfer mode: {mode}")
    action_target = action_z if mode == "symmetric" else action_z.detach()
    return 1.0 - torch.sum(clean_z * action_target, dim=1)


def guided_noise_loss(
    model, store: SpectrumStore, examples: list[GuidedNoiseExample],
    official_by_row: dict[int, np.ndarray], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_guided(
        store, examples,
        args.guided_recurrence_prevalence,
        args.guided_recurrence_max_peaks,
    )
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_margin = margins(encoded, layout, "clean")
    action_margin = margins(encoded, layout, "action")
    clean_rank_each = F.softplus((args.rank_margin - clean_margin) / args.temperature)
    action_rank_each = F.softplus((args.rank_margin - action_margin) / args.temperature)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    action_z = torch.stack([encoded[int(item["action"])] for item in layout])
    # E8 established that symmetric clean/action training transfers ranking
    # information more effectively than a detached action branch.  E13 keeps
    # the old stop-gradient form only as an explicit mechanism control.
    consistency_each = guided_consistency_values(
        clean_z, action_z, args.guided_transfer_mode,
    )
    teacher_margin = torch.tensor(
        [example.teacher_margin for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    baseline_teacher_margin = torch.tensor(
        [example.official_margin for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    teacher_margin_delta = torch.tensor(
        [example.teacher_margin_delta for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    has_fixed_teacher = torch.isfinite(teacher_margin)
    if args.guided_teacher_target_mode == "absolute":
        teacher_target = torch.clamp(
            teacher_margin, min=0.0,
            max=args.guided_teacher_margin_cap,
        )
    elif args.guided_teacher_target_mode == "delta":
        observed_delta = torch.where(
            torch.isfinite(teacher_margin_delta),
            teacher_margin_delta,
            teacher_margin - baseline_teacher_margin,
        )
        conservative_delta = args.guided_teacher_delta_fraction * torch.clamp(
            observed_delta, min=0.0, max=args.guided_teacher_delta_cap,
        )
        teacher_target = baseline_teacher_margin + conservative_delta
    else:
        raise RuntimeError(f"unknown guided teacher target mode: {args.guided_teacher_target_mode}")
    teacher_target = torch.where(
        has_fixed_teacher, teacher_target, clean_margin.detach(),
    )
    self_transfer = F.relu(action_margin.detach() - clean_margin)
    fixed_transfer = F.relu(teacher_target - clean_margin)
    fixed_teacher_each = torch.where(
        has_fixed_teacher, fixed_transfer, torch.zeros_like(fixed_transfer),
    )
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor_each = F.relu(floors - clean_margin)
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    clean_rank = weighted_mean(clean_rank_each, examples)
    action_rank = weighted_mean(action_rank_each, examples)
    consistency = weighted_mean(consistency_each, examples)
    floor = weighted_mean(floor_each, examples)
    loss = (
        args.lambda_clean_rank * clean_rank
        + args.lambda_aug_rank * action_rank
        + args.lambda_consistency * consistency
        + args.lambda_guided_transfer * weighted_mean(self_transfer, examples)
        + args.lambda_guided_teacher_margin * weighted_mean(fixed_teacher_each, examples)
        + args.lambda_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "guided_clean_rank": float(clean_rank.detach()),
        "guided_action_rank": float(action_rank.detach()),
        "guided_consistency": float(consistency.detach()),
        "guided_transfer": float(weighted_mean(self_transfer, examples).detach()),
        "guided_fixed_teacher_margin": float(weighted_mean(fixed_teacher_each, examples).detach()),
        "guided_fixed_teacher_fraction": float(has_fixed_teacher.float().mean().detach()),
        "guided_fixed_teacher_target": float(torch.where(
            has_fixed_teacher, teacher_target, torch.zeros_like(teacher_target)
        ).sum().detach() / torch.clamp(has_fixed_teacher.float().sum(), min=1.0)),
        "guided_risk_control_fraction": float(np.mean([
            example.supervision_kind != "corrective" for example in examples
        ])),
        "guided_margin_floor": float(floor.detach()),
        "guided_preserve": float(preserve.detach()),
        "guided_clean_margin": float(clean_margin.mean().detach()),
        "guided_action_margin": float(action_margin.mean().detach()),
        "guided_action_advantage": float((action_margin - clean_margin).mean().detach()),
    }


@torch.no_grad()
def audit_guided_teacher_replay(
    model,
    store: SpectrumStore,
    examples: list[GuidedNoiseExample],
    device: torch.device,
    args,
) -> dict[str, float]:
    """Fail closed unless action construction reproduces frozen teacher margins."""
    clean_error: list[np.ndarray] = []
    action_error: list[np.ndarray] = []
    for left in range(0, len(examples), args.eval_batch_size):
        batch = examples[left:left + args.eval_batch_size]
        spectra, layout, _ = flatten_guided(
            store, batch,
            args.guided_recurrence_prevalence,
            args.guided_recurrence_max_peaks,
        )
        encoded = forward_embeddings(model, spectra.to(device), False)
        clean = margins(encoded, layout, "clean").float().cpu().numpy()
        action = margins(encoded, layout, "action").float().cpu().numpy()
        expected_clean = np.asarray([item.official_margin for item in batch], dtype=np.float32)
        expected_action = np.asarray([item.teacher_margin for item in batch], dtype=np.float32)
        clean_error.append(np.abs(clean - expected_clean))
        action_error.append(np.abs(action - expected_action))
    clean_error_array = np.concatenate(clean_error)
    action_error_array = np.concatenate(action_error)
    report = {
        "queries": int(len(examples)),
        "clean_margin_max_abs_error": float(np.max(clean_error_array)),
        "action_margin_max_abs_error": float(np.max(action_error_array)),
        "clean_margin_p99_abs_error": float(np.quantile(clean_error_array, 0.99)),
        "action_margin_p99_abs_error": float(np.quantile(action_error_array, 0.99)),
    }
    if report["clean_margin_max_abs_error"] > 2e-4 or report["action_margin_max_abs_error"] > 2e-4:
        raise RuntimeError(f"E14 teacher replay failed: {report}")
    return report


def flatten_direct(
    store: SpectrumStore,
    examples: list[DirectExample],
    action: bool,
    materialized_action_spectra: np.ndarray | None = None,
    materialized_action_arm: str = "targeted",
):
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        item: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(example.query_row))
        clean_rows.append(example.query_row)
        if action:
            item["action"] = len(tensors)
            if example.materialized_action_index >= 0:
                if materialized_action_arm == "clean_duplicate":
                    tensors.append(store.one(example.query_row))
                else:
                    if materialized_action_spectra is None:
                        raise RuntimeError(
                            "materialized direct example has no action spectrum bank"
                        )
                    index = int(example.materialized_action_index)
                    if not 0 <= index < len(materialized_action_spectra):
                        raise RuntimeError(
                            f"materialized action tensor index is out of bounds: {index}"
                        )
                    tensors.append(torch.from_numpy(
                        np.asarray(materialized_action_spectra[index], dtype=np.float32)
                    ))
            else:
                tensors.append(attenuate_sequence(
                    store.one(example.query_row), example.target_path, example.attenuation,
                ))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def flatten_pmt(store: SpectrumStore, examples: list[DirectExample]):
    """Encode clean, target and frozen matched control in one candidate block."""
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        if not example.target_path or not example.control_path:
            raise RuntimeError("paired PMT example lacks target or control path")
        if example.target_path == example.control_path:
            raise RuntimeError("paired PMT target and control paths are identical")
        item: dict[str, object] = {"clean": len(tensors)}
        clean = store.one(example.query_row)
        tensors.append(clean); clean_rows.append(example.query_row)
        item["target"] = len(tensors)
        tensors.append(attenuate_sequence(clean, example.target_path, example.attenuation))
        item["control"] = len(tensors)
        tensors.append(attenuate_sequence(clean, example.control_path, example.attenuation))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows)); clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows)); clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def anchor_margins(
    examples: list[DirectExample], anchor_by_row: dict[int, np.ndarray],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for example in examples:
        query = anchor_by_row[int(example.query_row)]
        positive = np.stack([anchor_by_row[int(row)] for row in example.positive_rows])
        negative = np.stack([anchor_by_row[int(row)] for row in example.negative_rows])
        values.append(float(np.max(positive @ query) - np.max(negative @ query)))
    return torch.as_tensor(values, device=device, dtype=dtype)


def edge_margin_matrices(
    encoded: torch.Tensor, layout: list[dict], view: str,
) -> list[torch.Tensor]:
    """Keep every positive-reference x negative-molecule margin."""
    output: list[torch.Tensor] = []
    for item in layout:
        query = encoded[int(item[view])]
        positive_score = encoded[item["positive"]] @ query
        negative_score = encoded[item["negative"]] @ query
        output.append(positive_score[:, None] - negative_score[None, :])
    return output


def molecule_margin_vectors(
    encoded: torch.Tensor, layout: list[dict], view: str,
) -> list[torch.Tensor]:
    """Match retrieval: max true-reference score minus each negative molecule.

    ``negative`` contains one live representative per sampled negative molecule;
    positive references are reduced by max exactly as in the official evaluator.
    """
    output: list[torch.Tensor] = []
    for item in layout:
        query = encoded[int(item[view])]
        positive_score = torch.max(encoded[item["positive"]] @ query)
        negative_score = encoded[item["negative"]] @ query
        output.append(positive_score - negative_score)
    return output


def anchor_edge_margin_matrices(
    examples: list[DirectExample], anchor_by_row: dict[int, np.ndarray],
    device: torch.device, dtype: torch.dtype,
) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for example in examples:
        query = anchor_by_row[int(example.query_row)]
        positive = np.stack([anchor_by_row[int(row)] for row in example.positive_rows])
        negative = np.stack([anchor_by_row[int(row)] for row in example.negative_rows])
        output.append(torch.as_tensor(
            positive @ query[:, None] - (negative @ query)[None, :],
            device=device, dtype=dtype,
        ))
    return output


def refresh_candidate_references(
    graph: CandidateGraph, examples: list[DirectExample], rows: np.ndarray,
    encoded: np.ndarray, positives: int, negatives: int,
    preserve_forced_negative: bool = True,
) -> list[DirectExample]:
    """Refresh current hard molecules without dropping the action switch edge."""
    if positives < 1 or negatives < 1:
        raise ValueError("positive/negative reference counts must be positive")
    position = {int(row): index for index, row in enumerate(rows)}
    selected: dict[tuple[int, int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    output: list[DirectExample] = []
    for example in examples:
        forced_key = int(example.forced_negative_row) if preserve_forced_negative else -1
        forced_positive_key = int(example.forced_positive_row)
        cache_key = (int(example.query_index), forced_key, forced_positive_key)
        if cache_key not in selected:
            query = int(example.query_index)
            query_z = encoded[position[int(example.query_row)]]
            left, right = map(int, graph.query_ptr[query:query + 2])
            pos_left, pos_right = map(int, graph.molecule_ptr[left:left + 2])
            pos_rows = graph.pair_candidate_row[pos_left:pos_right].astype(np.int64)
            pos_scores = encoded[[position[int(row)] for row in pos_rows]] @ query_z
            pos_order = np.argsort(-pos_scores, kind="stable")
            positive_rows = tuple(map(int, pos_rows[pos_order[:positives]]))
            if forced_positive_key >= 0:
                if forced_positive_key not in set(map(int, pos_rows)):
                    raise RuntimeError(
                        f"forced action positive {forced_positive_key} is not positive "
                        f"for query {query}"
                    )
                # Max over the union preserves both the clean-active and the
                # action-active positive boundary without changing either
                # score. Substitution silently destroys the clean boundary.
                if forced_positive_key not in set(map(int, positive_rows)):
                    positive_rows = tuple(positive_rows) + (forced_positive_key,)

            # Retain one clean-active spectrum for every top negative molecule.
            # The action-active row can be another spectrum of the same
            # molecule and is appended below: max over this row union preserves
            # both full-graph boundaries exactly.
            choices: list[tuple[float, int, int]] = []
            for molecule in range(left + 1, right):
                pair_left, pair_right = map(int, graph.molecule_ptr[molecule:molecule + 2])
                molecule_rows = graph.pair_candidate_row[pair_left:pair_right].astype(np.int64)
                molecule_scores = encoded[
                    [position[int(row)] for row in molecule_rows]
                ] @ query_z
                local = int(np.argmax(molecule_scores))
                choices.append((
                    float(molecule_scores[local]), int(molecule_rows[local]), int(molecule),
                ))
            choices.sort(key=lambda item: (-item[0], item[1], item[2]))
            selected_choices = list(choices[:negatives])
            forced = forced_key
            if forced >= 0:
                forced_molecule = None
                for molecule in range(left + 1, right):
                    pair_left, pair_right = map(
                        int, graph.molecule_ptr[molecule:molecule + 2]
                    )
                    if forced in set(map(
                        int, graph.pair_candidate_row[pair_left:pair_right]
                    )):
                        forced_molecule = int(molecule)
                        break
                if forced_molecule is None:
                    raise RuntimeError(
                        f"forced action negative {forced} is not a candidate for query {query}"
                    )
            negative_rows = [row for _, row, _ in selected_choices]
            negative_molecules = [molecule for _, _, molecule in selected_choices]
            if len(set(negative_molecules)) != len(negative_molecules):
                raise RuntimeError(f"dynamic negatives repeat a molecule for query {query}")
            if forced >= 0 and forced not in set(map(int, negative_rows)):
                negative_rows.append(forced)
            if len(set(negative_rows)) != len(negative_rows):
                raise RuntimeError(f"dynamic negatives repeat a spectrum row for query {query}")
            selected[cache_key] = (positive_rows, tuple(negative_rows))
        positive_rows, negative_rows = selected[cache_key]
        output.append(replace(
            example, positive_rows=positive_rows, negative_rows=negative_rows,
        ))
    return output


def candidate_boundary_action_loss(
    model, store: SpectrumStore, examples: list[DirectExample],
    anchor_by_row: dict[int, np.ndarray], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direct candidate-edge training; no scalar teacher or embedding target."""
    spectra, layout, clean_rows = flatten_pmt(store, examples)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    if args.candidate_boundary_version == "v2_molecule_max":
        clean_each = molecule_margin_vectors(encoded, layout, "clean")
        target_each = molecule_margin_vectors(encoded, layout, "target")
        control_each = molecule_margin_vectors(encoded, layout, "control")
        query_groups: dict[int, list[int]] = {}
        for index, example in enumerate(examples):
            query_groups.setdefault(int(example.query_index), []).append(index)
        clean: list[torch.Tensor] = []
        target: list[list[torch.Tensor]] = []
        control: list[list[torch.Tensor]] = []
        action_weights: list[torch.Tensor] = []
        for indices in query_groups.values():
            first = indices[0]
            reference_key = (
                examples[first].positive_rows, examples[first].negative_rows,
            )
            if any(
                (examples[index].positive_rows, examples[index].negative_rows)
                != reference_key for index in indices[1:]
            ):
                raise RuntimeError(
                    "candidate-boundary v2 query actions do not share candidate references"
                )
            clean.append(clean_each[first])
            target.append([target_each[index] for index in indices])
            control.append([control_each[index] for index in indices])
            action_weights.append(torch.as_tensor(
                [examples[index].sample_weight for index in indices],
                device=encoded.device, dtype=encoded.dtype,
            ))
        result = direct_boundary_objective(
            clean, target, control, action_weights,
            rank_margin=args.rank_margin,
            rank_temperature=args.temperature,
            advantage_temperature=args.boundary_advantage_temperature,
            topk_negatives=args.boundary_topk_negatives,
            action_safety_slack=args.margin_floor_slack,
            lambda_clean=args.lambda_boundary_full_clean,
            lambda_corrective_clean=args.lambda_boundary_clean,
            lambda_action_rank=args.lambda_boundary_target,
            lambda_counterfactual=args.lambda_boundary_counterfactual,
            lambda_action_safety=args.lambda_boundary_action_safety,
        )
        boundary_metrics = {
            "boundary_clean": float(result.corrective_clean_rank.detach()),
            "boundary_target": float(result.action_rank.detach()),
            "boundary_counterfactual": float(result.counterfactual.detach()),
            "boundary_full_clean": float(result.clean_rank.detach()),
            "boundary_active_edge_fraction": float(result.effective_queries / len(query_groups)),
            "boundary_mean_live_advantage": float("nan"),
            "boundary_effective_query_fraction": float(result.effective_queries / len(query_groups)),
            "boundary_routed_actions": float(len(examples)),
            "boundary_qualified_actions": float(result.qualified_actions),
            "boundary_unqualified_actions": float(result.unqualified_actions),
            "boundary_action_safety": float(result.action_safety.detach()),
        }
    else:
        clean = edge_margin_matrices(encoded, layout, "clean")
        target = edge_margin_matrices(encoded, layout, "target")
        control = edge_margin_matrices(encoded, layout, "control")
        result = candidate_boundary_objective(
            clean, target, control,
            rank_margin=args.rank_margin,
            rank_temperature=args.temperature,
            advantage_temperature=args.boundary_advantage_temperature,
            hard_temperature=args.boundary_hard_temperature,
            topk_negatives=args.boundary_topk_negatives,
            lambda_clean=args.lambda_boundary_clean,
            lambda_target=args.lambda_boundary_target,
            lambda_counterfactual=args.lambda_boundary_counterfactual,
            lambda_full_clean=args.lambda_boundary_full_clean,
        )
        boundary_metrics = {
            "boundary_clean": float(result.clean_boundary.detach()),
            "boundary_target": float(result.target_boundary.detach()),
            "boundary_counterfactual": float(result.counterfactual.detach()),
            "boundary_full_clean": float(result.full_clean_rank.detach()),
            "boundary_active_edge_fraction": float(result.active_edge_fraction.detach()),
            "boundary_mean_live_advantage": float(result.mean_advantage.detach()),
            "boundary_effective_query_fraction": float(result.effective_queries / len(examples)),
        }
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    if args.candidate_boundary_version == "v2_molecule_max":
        # Repeated actions for one query re-encode identical clean/reference
        # rows. Preserve each physical spectrum once so action multiplicity
        # cannot silently multiply the anchor dose.
        unique: dict[int, int] = {}
        for row, index in zip(clean_rows, clean_indices):
            unique.setdefault(int(row), int(index))
        preserve_indices = list(unique.values())
        preserve_rows = list(unique.keys())
    else:
        preserve_indices = clean_indices
        preserve_rows = clean_rows
    anchors = torch.from_numpy(np.stack([anchor_by_row[row] for row in preserve_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[preserve_indices] * anchors, dim=1)).mean()
    loss = result.loss + args.lambda_preserve * preserve
    return loss, {**boundary_metrics, "action_preserve": float(preserve.detach())}


def paired_pmt_loss(
    model, store: SpectrumStore, examples: list[DirectExample],
    anchor_by_row: dict[int, np.ndarray], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    """E4 loss with explicit target-control preference and clean inheritance.

    The control retains its own identity/rank floor, so preference cannot be
    satisfied by deliberately degrading the matched-random branch.
    """
    spectra, layout, clean_rows = flatten_pmt(store, examples)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    target_z = torch.stack([encoded[int(item["target"])] for item in layout])
    control_z = torch.stack([encoded[int(item["control"])] for item in layout])
    clean_margin = margins(encoded, layout, "clean")
    target_layout = [dict(item, action=item["target"]) for item in layout]
    control_layout = [dict(item, action=item["control"]) for item in layout]
    target_margin = margins(encoded, target_layout, "action")
    control_margin = margins(encoded, control_layout, "action")
    clean_rank_each = F.softplus((args.rank_margin - clean_margin) / args.temperature)
    target_rank_each = F.softplus((args.rank_margin - target_margin) / args.temperature)
    control_rank_each = F.softplus((args.rank_margin - control_margin) / args.temperature)
    preference_each = F.relu(args.pmt_preference_gap - (target_margin - control_margin))
    initial_margin = anchor_margins(examples, anchor_by_row, device, clean_margin.dtype)
    advantage = torch.as_tensor(
        [example.teacher_advantage for example in examples],
        device=device, dtype=clean_margin.dtype,
    ).clamp(min=0.0, max=args.pmt_advantage_cap)
    inherited_floor = initial_margin + args.pmt_alpha * advantage
    clean_floor_each = F.relu(inherited_floor - args.margin_floor_slack - clean_margin)
    control_floor_each = F.relu(initial_margin - args.margin_floor_slack - control_margin)
    consistency_each = 0.5 * (
        1.0 - torch.sum(clean_z * target_z, dim=1)
        + 1.0 - torch.sum(clean_z * control_z, dim=1)
    )
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"]); clean_indices.extend(item["negative"])
    anchors = torch.from_numpy(np.stack([anchor_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * anchors, dim=1)).mean()
    clean_rank = weighted_mean(clean_rank_each, examples)
    paired_rank = weighted_mean(0.5 * (target_rank_each + control_rank_each), examples)
    preference = weighted_mean(preference_each, examples)
    clean_floor = weighted_mean(clean_floor_each, examples)
    control_floor = weighted_mean(control_floor_each, examples)
    consistency = weighted_mean(consistency_each, examples)
    loss = (
        args.lambda_clean_rank * clean_rank
        + args.lambda_aug_rank * paired_rank
        + args.lambda_consistency * consistency
        + args.lambda_margin_floor * (clean_floor + control_floor)
        + args.lambda_pmt_preference * preference
        + args.lambda_preserve * preserve
    )
    return loss, {
        "action_clean_rank": float(clean_rank.detach()),
        "action_aug_rank": float(paired_rank.detach()),
        "action_consistency": float(consistency.detach()),
        "action_margin_floor": float(clean_floor.detach()),
        "pmt_control_floor": float(control_floor.detach()),
        "pmt_preference": float(preference.detach()),
        "action_preserve": float(preserve.detach()),
        "action_clean_margin": float(clean_margin.mean().detach()),
        "action_aug_margin": float(target_margin.mean().detach()),
        "pmt_control_margin": float(control_margin.mean().detach()),
        "pmt_current_advantage": float((target_margin - control_margin).mean().detach()),
        "pmt_teacher_advantage": float(advantage.mean().detach()),
        "pmt_inherited_margin_increment": float((args.pmt_alpha * advantage).mean().detach()),
    }


def weighted_mean(values: torch.Tensor, examples: list[DirectExample]) -> torch.Tensor:
    weights = torch.tensor(
        [example.sample_weight for example in examples],
        device=values.device, dtype=values.dtype,
    )
    return torch.sum(values * weights) / torch.sum(weights)


def materialized_direct_mean(
    values: torch.Tensor,
    examples: list[DirectExample],
    injection_mode: str,
) -> torch.Tensor:
    """Reduce a materialized branch without recreating action multiplicity.

    The multi-action weights have a fixed, run-level meaning: their sum is the
    historical 4x4 E4 dose per identity.  Renormalising them inside every
    four-row microbatch would undo that contract and let a duplicated family
    regain unit dose.  The historical one-best path remains bitwise equivalent
    to the old normalized mean because all of its weights are one.
    """
    if injection_mode == "one_best_e4":
        return weighted_mean(values, examples)
    if injection_mode != "multi_action_balanced":
        raise ValueError(f"unknown materialized injection mode: {injection_mode}")
    weights = torch.as_tensor(
        [example.sample_weight for example in examples],
        device=values.device, dtype=values.dtype,
    )
    return torch.mean(values * weights)


def gated_materialized_action_rank(
    action_margin: torch.Tensor,
    *,
    rank_margin: float,
    temperature: float,
    injection_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop rewarding already robust actions only in the repaired multi arm."""
    raw = F.softplus((float(rank_margin) - action_margin) / float(temperature))
    if injection_mode in {"one_best_e4", "complete_panel_historical_e4"}:
        active = torch.ones_like(action_margin, dtype=torch.bool)
        return raw, active
    if injection_mode != "multi_action_balanced":
        raise ValueError(f"unknown materialized injection mode: {injection_mode}")
    active = action_margin.detach() < float(rank_margin)
    return torch.where(active, raw, torch.zeros_like(raw)), active


def action_key(example: DirectExample) -> tuple[int, str, tuple[int, ...], float]:
    return (
        int(example.query_index), str(example.policy), tuple(example.target_path),
        float(example.attenuation),
    )


def frozen_reference_margins(
    query_vectors: torch.Tensor, examples: list[DirectExample],
    official_by_row: dict[int, np.ndarray],
) -> torch.Tensor:
    """Score trainable queries against fixed official candidate anchors.

    This keeps the retrieval geometry shared at inference while making the
    *training diagnostic* identifiable: a query-ranking improvement cannot be
    paid for by moving the few positive/negative reference spectra in its
    minibatch.  Reference spectra still receive the ordinary preservation
    loss and are encoded by the same saved model at evaluation.
    """
    output: list[torch.Tensor] = []
    for vector, example in zip(query_vectors, examples):
        positive = torch.as_tensor(
            np.stack([official_by_row[int(row)] for row in example.positive_rows]),
            device=vector.device, dtype=vector.dtype,
        )
        negative = torch.as_tensor(
            np.stack([official_by_row[int(row)] for row in example.negative_rows]),
            device=vector.device, dtype=vector.dtype,
        )
        output.append(torch.max(positive @ vector) - torch.max(negative @ vector))
    return torch.stack(output)


@torch.no_grad()
def encode_official_action_targets(
    model: torch.nn.Module, store: SpectrumStore, examples: list[DirectExample],
    device: torch.device, batch_size: int,
) -> dict[tuple[int, str, tuple[int, ...], float], np.ndarray]:
    """Freeze the official embedding of every distinct raw-spectrum action.

    Targets are generated once before the first optimizer step.  They are not
    labels, candidate scores, P2b outputs, or post-embedding modules: each is
    simply the official DreaMS encoding of the preregistered perturbed spectrum.
    """
    unique: dict[tuple[int, str, tuple[int, ...], float], DirectExample] = {}
    for example in examples:
        unique.setdefault(action_key(example), example)
    ordered = [unique[key] for key in sorted(unique)]
    targets: dict[tuple[int, str, tuple[int, ...], float], np.ndarray] = {}
    model.eval()
    for left in range(0, len(ordered), batch_size):
        block = ordered[left:left + batch_size]
        spectra = torch.stack([
            attenuate_sequence(
                store.one(example.query_row), example.target_path, example.attenuation,
            ) for example in block
        ]).to(device)
        vectors = forward_embeddings(model, spectra, False).float().cpu().numpy()
        if not np.all(np.isfinite(vectors)):
            raise RuntimeError("official action target encoding produced non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(np.abs(norms - 1.0) > 2e-3):
            raise RuntimeError("official action target encoding produced non-unit values")
        for example, vector in zip(block, vectors):
            targets[action_key(example)] = np.asarray(vector, dtype=np.float32)
        right = left + len(block)
        if right == len(ordered) or right % (batch_size * 20) == 0:
            print(f"[official-action-targets] {right:,}/{len(ordered):,}", flush=True)
    if len(targets) != len(unique):
        raise RuntimeError("official action target cache is incomplete")
    return targets


@torch.no_grad()
def audit_materialized_action_replay(
    model: torch.nn.Module,
    store: SpectrumStore,
    examples: list[DirectExample],
    action_spectra: np.ndarray,
    expected_margin_by_action: dict[str, float],
    expected_clean_margin_by_action: dict[str, float],
    reference_by_row: dict[int, np.ndarray],
    device: torch.device,
    batch_size: int,
    amp: bool,
    arm: str,
    enforce_expected_geometry: bool = True,
) -> dict[str, float | int | bool | str]:
    """Verify every action against cached, exact current-E8 candidates.

    Query and candidate embeddings were already encoded from the immutable
    initialization checkpoint. Re-encoding them for every action expanded 32k
    actions into roughly 450k spectra. The GPU now encodes each action once;
    stored winning rows are scored against the same current-E8 cache.
    """
    if arm != "targeted":
        return {
            "evaluated": False,
            "reason": "causal control deliberately changes the action view",
            "actions": int(len(examples)),
        }
    if batch_size < 1:
        raise ValueError("materialized replay action batch size must be positive")
    if not examples:
        raise RuntimeError("materialized replay has no examples")
    maximum_spectra_per_forward = int(batch_size)
    effective_batch_size = int(batch_size)
    print(
        "[materialized-E4-replay-batch] "
        + json.dumps({
            "requested_action_batch_size": int(batch_size),
            "effective_action_batch_size": int(effective_batch_size),
            "spectra_per_example": 1,
            "maximum_spectra_per_forward_contract": int(maximum_spectra_per_forward),
            "planned_maximum_spectra_per_forward": int(effective_batch_size),
        }),
        flush=True,
    )
    observed: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    observed_clean: list[np.ndarray] = []
    expected_clean: list[np.ndarray] = []
    observed_maximum_spectra = 0
    for left in range(0, len(examples), effective_batch_size):
        block = examples[left:left + effective_batch_size]
        spectra = torch.stack([
            torch.from_numpy(np.asarray(
                action_spectra[int(example.materialized_action_index)],
                dtype=np.float32,
            ))
            for example in block
        ])
        observed_maximum_spectra = max(observed_maximum_spectra, len(spectra))
        if len(spectra) > maximum_spectra_per_forward:
            raise RuntimeError(
                "materialized replay exceeded the bounded CUDA spectrum budget"
            )
        encoded = forward_embeddings(model, spectra.to(device), amp).float().cpu().numpy()

        def exact_reference_margin(
            vector: np.ndarray, example: DirectExample,
        ) -> float:
            positive = np.stack([
                reference_by_row[int(row)] for row in example.positive_rows
            ])
            negative = np.stack([
                reference_by_row[int(row)] for row in example.negative_rows
            ])
            return float(np.max(positive @ vector) - np.max(negative @ vector))

        observed.append(np.asarray([
            exact_reference_margin(vector, example)
            for vector, example in zip(encoded, block)
        ], dtype=np.float32))
        observed_clean.append(np.asarray([
            exact_reference_margin(
                reference_by_row[int(example.query_row)], example,
            )
            for example in block
        ], dtype=np.float32))
        expected.append(np.asarray([
            expected_margin_by_action[example.action_id] for example in block
        ], dtype=np.float32))
        expected_clean.append(np.asarray([
            expected_clean_margin_by_action[example.action_id] for example in block
        ], dtype=np.float32))
        right = left + len(block)
        if right == len(examples) or right % (effective_batch_size * 20) == 0:
            print(
                f"[materialized-E4-action-replay] {right:,}/{len(examples):,}",
                flush=True,
            )
    observed_array = np.concatenate(observed)
    expected_array = np.concatenate(expected)
    absolute_error = np.abs(observed_array - expected_array)
    observed_clean_array = np.concatenate(observed_clean)
    expected_clean_array = np.concatenate(expected_clean)
    clean_absolute_error = np.abs(observed_clean_array - expected_clean_array)
    report: dict[str, float | int | bool | str] = {
        "evaluated": True,
        "actions": int(len(examples)),
        "all_actions_encoded_exactly_once": True,
        "candidate_reference_embeddings_reused_from_initialization": True,
        "candidate_reference_geometry": (
            "current_E8" if enforce_expected_geometry else "official_DreaMS"
        ),
        "requested_action_batch_size": int(batch_size),
        "effective_action_batch_size": int(effective_batch_size),
        "spectra_per_example": 1,
        "maximum_spectra_per_forward_contract": int(maximum_spectra_per_forward),
        "planned_maximum_spectra_per_forward": int(effective_batch_size),
        "observed_maximum_spectra_per_forward": int(observed_maximum_spectra),
        "positive_margin_actions": int(np.sum(observed_array > 0)),
        "positive_margin_fraction": float(np.mean(observed_array > 0)),
        "margin_max_abs_error": float(np.max(absolute_error)),
        "margin_p99_abs_error": float(np.quantile(absolute_error, 0.99)),
        "clean_margin_max_abs_error": float(np.max(clean_absolute_error)),
        "clean_margin_p99_abs_error": float(np.quantile(clean_absolute_error, 0.99)),
        "all_strict_top1_actions_reproduced": bool(np.all(observed_array > 0)),
        "exact_winning_positive_and_negative_rows_preserved": True,
        "stored_selection_geometry_enforced": bool(enforce_expected_geometry),
    }
    if enforce_expected_geometry and not report["all_strict_top1_actions_reproduced"]:
        raise RuntimeError(f"materialized best-action replay lost Top-1: {report}")
    # The route was computed in batches and this replay can use a different
    # batch size, so tolerate only ordinary fp32 kernel drift.
    if enforce_expected_geometry and (
        report["margin_max_abs_error"] > 5e-4
        or report["clean_margin_max_abs_error"] > 5e-4
    ):
        raise RuntimeError(f"materialized best-action margin replay drifted: {report}")
    return report


def gradient_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Return the pre-clipping L2 norm without modifying gradients."""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2).item()
            squared += value * value
    return math.sqrt(squared)


def detached_loss_gradients(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter],
) -> tuple[float, list[torch.Tensor | None]]:
    """Return one branch gradient without touching optimizer ``.grad`` buffers."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=False, create_graph=False, allow_unused=True,
    )
    detached = [gradient.detach() if gradient is not None else None for gradient in gradients]
    squared = sum(
        float(torch.sum(gradient.float() * gradient.float()).detach())
        for gradient in detached if gradient is not None
    )
    return math.sqrt(squared), detached


def gradient_cosine(
    first: list[torch.Tensor | None], second: list[torch.Tensor | None],
) -> float:
    dot = 0.0
    first_squared = 0.0
    second_squared = 0.0
    for left, right in zip(first, second):
        if left is not None:
            first_squared += float(torch.sum(left.float() * left.float()).detach())
        if right is not None:
            second_squared += float(torch.sum(right.float() * right.float()).detach())
        if left is not None and right is not None:
            dot += float(torch.sum(left.float() * right.float()).detach())
    denominator = math.sqrt(first_squared * second_squared)
    return dot / denominator if denominator > 0 else float("nan")


def combined_gradient_norm(
    components: list[tuple[list[torch.Tensor | None], float]],
) -> float:
    squared = 0.0
    for parameter_index in range(len(components[0][0])):
        combined = None
        for gradients, weight in components:
            gradient = gradients[parameter_index]
            if gradient is None:
                continue
            contribution = gradient.float() * float(weight)
            combined = contribution if combined is None else combined + contribution
        if combined is not None:
            squared += float(torch.sum(combined * combined).detach())
    return math.sqrt(squared)


def audit_historical_e4_action_specific_alignment(
    model: torch.nn.Module,
    store: SpectrumStore,
    examples: list[DirectExample],
    materialized_action_spectra: np.ndarray,
    materialized_action_arm: str,
    head_parameters: list[torch.nn.Parameter],
    backbone_parameters: list[torch.nn.Parameter],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Compare action-only signal with an independently constructed clean objective."""
    positive_alignment_required = materialized_action_arm == "targeted"
    batches = formula_stratified_microbatches(
        examples, formulas=16, examples_per_formula=args.batch_actions,
        seed=args.seed + 3109,
    )
    parameters = head_parameters + backbone_parameters
    head_count = len(head_parameters)
    records = {"head": [], "backbone": []}
    clean_nonzero = {"head": [], "backbone": []}
    action_nonzero = {"head": [], "backbone": []}
    for batch in batches:
        action_spectra, action_layout, _ = flatten_direct(
            store, batch, True,
            materialized_action_spectra=materialized_action_spectra,
            materialized_action_arm=materialized_action_arm,
        )
        action_encoded = forward_embeddings(model, action_spectra.to(device), args.amp)
        action_margin = margins(action_encoded, action_layout, "action")
        action_rank = weighted_mean(
            F.softplus((args.rank_margin - action_margin) / args.temperature), batch,
        )
        clean_action_z = torch.stack([
            action_encoded[int(item["clean"])] for item in action_layout
        ])
        action_z = torch.stack([
            action_encoded[int(item["action"])] for item in action_layout
        ])
        action_consistency = weighted_mean(
            1.0 - torch.sum(clean_action_z * action_z, dim=1), batch,
        )
        action_specific_objective = (
            args.lambda_aug_rank * action_rank
            + args.lambda_consistency * action_consistency
        )
        _, action_gradient = detached_loss_gradients(
            action_specific_objective, parameters,
        )

        spectra, layout, _ = flatten_direct(store, batch, False)
        encoded = forward_embeddings(model, spectra.to(device), args.amp)
        clean_margin = margins(encoded, layout, "clean")
        clean_rank = weighted_mean(
            F.softplus((args.rank_margin - clean_margin) / args.temperature), batch,
        )
        floors = torch.tensor(
            [example.official_margin - args.margin_floor_slack for example in batch],
            device=device, dtype=clean_margin.dtype,
        )
        clean_floor = weighted_mean(F.relu(floors - clean_margin), batch)
        clean_objective = (
            args.lambda_clean_rank * clean_rank
            + args.lambda_margin_floor * clean_floor
        )
        _, clean_gradient = detached_loss_gradients(clean_objective, parameters)
        for group, left, right in (
            ("head", 0, head_count),
            ("backbone", head_count, len(parameters)),
        ):
            action_group = action_gradient[left:right]
            clean_group = clean_gradient[left:right]
            records[group].append(gradient_cosine(action_group, clean_group))
            action_nonzero[group].append(any(
                value is not None and bool(torch.any(value != 0))
                for value in action_group
            ))
            clean_nonzero[group].append(any(
                value is not None and bool(torch.any(value != 0))
                for value in clean_group
            ))
    report: dict[str, object] = {
        "audit_only_no_optimizer_steps": True,
        "formula_microbatches": int(len(batches)),
        "action_specific_definition": "aug_rank_plus_0.25_symmetric_consistency",
        "clean_corrective_definition": "clean_rank_plus_2.0_margin_floor",
        "preservation_excluded_as_separate_protective_component": True,
        "shared_loss_terms_between_compared_objectives": False,
        "definition": "cosine(action_specific_gradient, clean_corrective_gradient)",
        "positive_alignment_required": bool(positive_alignment_required),
        "parameter_groups": {},
    }
    gate = True
    for group in ("head", "backbone"):
        values = np.asarray(records[group], dtype=np.float64)
        finite = values[np.isfinite(values)]
        group_report = {
            "observations": int(len(values)),
            "action_gradient_nonzero_fraction": float(np.mean(action_nonzero[group])),
            "clean_corrective_gradient_nonzero_fraction": float(np.mean(clean_nonzero[group])),
            "alignment_p10": float(np.quantile(finite, 0.10)) if len(finite) else None,
            "alignment_median": float(np.median(finite)) if len(finite) else None,
        }
        group_pass = bool(
            len(finite) == len(values)
            and group_report["action_gradient_nonzero_fraction"] == 1.0
            and group_report["clean_corrective_gradient_nonzero_fraction"] == 1.0
            and (
                not positive_alignment_required
                or float(group_report["alignment_median"]) > 0
            )
        )
        group_report["gate_passed"] = group_pass
        report["parameter_groups"][group] = group_report
        gate = gate and group_pass
    report["gate_passed"] = bool(gate)
    if not gate:
        raise RuntimeError(f"historical E4 action-specific semantic gate failed: {report}")
    return report


def formula_stratified_microbatches(
    examples: list[DirectExample], *, formulas: int, examples_per_formula: int,
    seed: int, selected_formulas: list[str] | None = None,
) -> list[list[DirectExample]]:
    """Build deterministic formula-cluster microbatches for gradient calibration."""
    if formulas < 1 or examples_per_formula < 1:
        raise ValueError("formula calibration dimensions must be positive")
    grouped: dict[str, list[DirectExample]] = {}
    for example in examples:
        grouped.setdefault(str(example.formula), []).append(example)
    eligible = (
        [str(formula) for formula in selected_formulas]
        if selected_formulas is not None else
        [formula for formula, rows in grouped.items() if rows]
    )
    if any(formula not in grouped for formula in eligible):
        missing = sorted(set(eligible) - set(grouped))
        raise RuntimeError(f"formula calibration is missing requested strata: {missing[:5]}")
    if len(eligible) < formulas:
        raise RuntimeError(
            f"candidate-boundary calibration has {len(eligible)} formulas; need {formulas}"
        )
    rng = np.random.default_rng(seed)
    eligible = sorted(eligible)
    if selected_formulas is None:
        rng.shuffle(eligible)
    else:
        eligible = eligible[:formulas]
    batches: list[list[DirectExample]] = []
    for formula in eligible[:formulas]:
        rows = sorted(
            grouped[formula],
            key=lambda item: (item.query_index, item.policy, item.target_path),
        )
        if len(rows) > examples_per_formula:
            chosen = np.sort(rng.choice(
                len(rows), size=examples_per_formula, replace=False,
            ))
            rows = [rows[int(index)] for index in chosen]
        batches.append(rows)
    return batches


def calibrate_candidate_boundary_gradients(
    model, store: SpectrumStore, action_examples: list[DirectExample],
    safety_examples: list[DirectExample], anchor_by_row: dict[int, np.ndarray],
    device: torch.device, args,
) -> tuple[float, dict]:
    """Measure branch scales without updates and return a bounded action multiplier.

    The multiplier is one scalar for optimizer conditioning only.  It never
    replaces the positive-reference x negative-molecule edge matrix used by
    the objective, and it is not derived from held outcomes.
    """
    action_formulas = {str(example.formula) for example in action_examples}
    safety_formulas = {str(example.formula) for example in safety_examples}
    common_formulas = sorted(action_formulas & safety_formulas)
    if len(common_formulas) < args.boundary_calibration_formulas:
        raise RuntimeError(
            "candidate-boundary action/safety calibration has only "
            f"{len(common_formulas)} shared formulas; need "
            f"{args.boundary_calibration_formulas}"
        )
    formula_rng = np.random.default_rng(args.seed + 699983)
    formula_rng.shuffle(common_formulas)
    selected_formulas = common_formulas[:args.boundary_calibration_formulas]
    action_batches = formula_stratified_microbatches(
        action_examples, formulas=args.boundary_calibration_formulas,
        examples_per_formula=args.boundary_calibration_examples,
        seed=args.seed + 700001,
        selected_formulas=selected_formulas,
    )
    safety_batches = formula_stratified_microbatches(
        safety_examples, formulas=args.boundary_calibration_formulas,
        examples_per_formula=args.boundary_calibration_examples,
        seed=args.seed + 700003,
        selected_formulas=selected_formulas,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    action_norms: list[float] = []
    safety_norms: list[float] = []
    cosines: list[float] = []
    for index, (action_batch, safety_batch) in enumerate(
        zip(action_batches, safety_batches), start=1,
    ):
        action_loss, _ = candidate_boundary_action_loss(
            model, store, action_batch, anchor_by_row, device, args,
        )
        action_norm, action_gradient = detached_loss_gradients(action_loss, trainable)
        safety_loss_value, _ = candidate_boundary_safety_loss(
            model, store, safety_batch, anchor_by_row, device, args,
        )
        safety_norm, safety_gradient = detached_loss_gradients(safety_loss_value, trainable)
        if not (np.isfinite(action_norm) and np.isfinite(safety_norm)):
            raise RuntimeError("candidate-boundary calibration produced a non-finite norm")
        action_norms.append(action_norm)
        safety_norms.append(safety_norm)
        cosines.append(gradient_cosine(action_gradient, safety_gradient))
        del action_gradient, safety_gradient
        if index % 8 == 0 or index == len(action_batches):
            print(
                f"[boundary calibration] {index}/{len(action_batches)} formulas",
                flush=True,
            )
    action_median = float(np.median(action_norms))
    safety_median = float(np.median(safety_norms))
    if action_median <= 0:
        raise RuntimeError("candidate-boundary corrective gradient is identically zero")
    requested = (
        args.boundary_action_to_safety_norm_ratio
        * args.safety_stream_weight * safety_median / action_median
    )
    if args.boundary_auto_balance:
        if args.candidate_boundary_version == "v2_molecule_max" and safety_median == 0:
            scale = 1.0
        else:
            scale = min(
                args.boundary_action_scale_cap,
                max(0.0 if args.candidate_boundary_version == "v2_molecule_max" else 1.0,
                    requested),
            )
    else:
        scale = 1.0
    finite_cosines = np.asarray(
        [value for value in cosines if np.isfinite(value)], dtype=np.float64,
    )
    report = {
        "formula_microbatches": int(len(action_batches)),
        "action_safety_formula_strata_identical": True,
        "examples_per_formula_maximum": int(args.boundary_calibration_examples),
        "action_gradient_norm_median": action_median,
        "action_gradient_norm_p10": float(np.quantile(action_norms, 0.10)),
        "action_gradient_norm_p90": float(np.quantile(action_norms, 0.90)),
        "safety_gradient_norm_median": safety_median,
        "safety_gradient_norm_p10": float(np.quantile(safety_norms, 0.10)),
        "safety_gradient_norm_p90": float(np.quantile(safety_norms, 0.90)),
        "action_safety_cosine_mean": (
            float(np.mean(finite_cosines)) if len(finite_cosines) else None
        ),
        "action_safety_cosine_positive_fraction": (
            float(np.mean(finite_cosines > 0)) if len(finite_cosines) else None
        ),
        "safety_inactive_at_initialization": bool(safety_median == 0),
        "requested_action_scale": float(requested),
        "effective_action_scale": float(scale),
        "scale_was_capped": bool(requested > args.boundary_action_scale_cap),
        "held_outcomes_used": False,
        "optimizer_steps": 0,
    }
    return float(scale), report


def direct_action_loss(model, store: SpectrumStore, examples: list[DirectExample],
                       official_by_row: dict[int, np.ndarray], device: torch.device,
                       args, official_action_targets=None,
                       materialized_action_spectra: np.ndarray | None = None,
                       materialized_action_arm: str = "targeted",
                       ) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_direct(
        store, examples, True,
        materialized_action_spectra=materialized_action_spectra,
        materialized_action_arm=materialized_action_arm,
    )
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    aug_z = torch.stack([encoded[int(item["action"])] for item in layout])
    if args.rank_reference_mode == "official":
        clean_margin = frozen_reference_margins(clean_z, examples, official_by_row)
        aug_margin = frozen_reference_margins(aug_z, examples, official_by_row)
    else:
        clean_margin = margins(encoded, layout, "clean")
        aug_margin = margins(encoded, layout, "action")
    clean_rank_each = F.softplus((args.rank_margin - clean_margin) / args.temperature)
    if (
        args.action_selection == "materialized_routed"
        and args.materialized_injection_mode != "complete_panel_historical_e4"
    ):
        aug_rank_each, aug_rank_active = gated_materialized_action_rank(
            aug_margin,
            rank_margin=args.rank_margin,
            temperature=args.temperature,
            injection_mode=args.materialized_injection_mode,
        )
    else:
        aug_rank_each = F.softplus((args.rank_margin - aug_margin) / args.temperature)
        aug_rank_active = torch.ones_like(aug_margin, dtype=torch.bool)
    if args.direct_transfer_mode == "symmetric":
        transfer_target = aug_z
    elif args.direct_transfer_mode == "student_action_stopgrad":
        transfer_target = aug_z.detach()
    elif args.direct_transfer_mode == "official_action":
        if official_action_targets is None:
            raise RuntimeError("official_action transfer requires frozen action targets")
        transfer_target = torch.as_tensor(
            np.stack([official_action_targets[action_key(example)] for example in examples]),
            device=device, dtype=clean_z.dtype,
        )
    else:  # pragma: no cover - argparse protects this branch
        raise RuntimeError(f"unknown direct transfer mode: {args.direct_transfer_mode}")
    consistency_each = 1.0 - torch.sum(clean_z * transfer_target, dim=1)
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor_each = F.relu(floors - clean_margin)

    clean_indices: list[int] = []
    preserve_each: list[torch.Tensor] = []
    clean_cursor = 0
    for item in layout:
        indices = [int(item["clean"]), *item["positive"], *item["negative"]]
        clean_indices.extend(indices)
        row_count = 1 + len(item["positive"]) + len(item["negative"])
        reference_rows = clean_rows[clean_cursor:clean_cursor + row_count]
        reference = torch.from_numpy(
            np.stack([official_by_row[row] for row in reference_rows])
        ).to(device=device, dtype=encoded.dtype)
        preserve_each.append(
            (1.0 - torch.sum(encoded[indices] * reference, dim=1)).mean()
        )
        clean_cursor += row_count
    if clean_cursor != len(clean_rows):
        raise RuntimeError("direct preservation layout lost a clean reference row")
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    if len(official) != len(clean_indices):
        raise RuntimeError("direct preservation tensor and row ledger disagree")
    if (
        args.action_selection == "materialized_routed"
        and args.materialized_injection_mode != "complete_panel_historical_e4"
    ):
        reduce = lambda values: materialized_direct_mean(
            values, examples, args.materialized_injection_mode,
        )
        preserve = reduce(torch.stack(preserve_each))
    else:
        # Preserve the legacy per-spectrum reduction outside the repaired
        # materialized arm.  No older experiment silently changes semantics.
        preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
        reduce = lambda values: weighted_mean(values, examples)
    clean_rank = reduce(clean_rank_each)
    aug_rank = reduce(aug_rank_each)
    consistency = reduce(consistency_each)
    floor = reduce(floor_each)
    loss = (
        args.lambda_clean_rank * clean_rank
        + args.lambda_aug_rank * aug_rank
        + args.lambda_consistency * consistency
        + args.lambda_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "action_clean_rank": float(clean_rank.detach()),
        "action_aug_rank": float(aug_rank.detach()),
        "action_consistency": float(consistency.detach()),
        "action_margin_floor": float(floor.detach()),
        "action_preserve": float(preserve.detach()),
        "action_clean_margin": float(clean_margin.mean().detach()),
        "action_aug_margin": float(aug_margin.mean().detach()),
        "action_clean_margin_pass": float((clean_margin > 0).float().mean().detach()),
        "action_aug_margin_pass": float((aug_margin > 0).float().mean().detach()),
        "action_aug_rank_active_fraction": float(
            aug_rank_active.float().mean().detach()
        ),
        "action_effective_weight_sum": float(sum(
            example.sample_weight for example in examples
        )),
        "action_transfer_target_cosine": float(
            torch.sum(clean_z * transfer_target, dim=1).mean().detach()
        ),
    }


def materialized_query_local_semantic_loss(
    model,
    store: SpectrumStore,
    examples: list[DirectExample],
    action_weights: list[float],
    materialized_action_spectra: np.ndarray,
    materialized_action_arm: str,
    device: torch.device,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Later-action residual only; historical E4 remains a separate base stream.

    The forward values are the selected query's clean molecule-boundary rank,
    gated E4 action rank, and E4's symmetric clean/action consistency. Candidate
    references are detached only inside this residual; clean and action views
    remain live. Action rank remains live only until the action clears the E4
    margin.
    """
    if len(examples) != len(action_weights) or not examples:
        raise ValueError("semantic action examples and weights must align")
    spectra, layout, _ = flatten_direct(
        store,
        examples,
        True,
        materialized_action_spectra=materialized_action_spectra,
        materialized_action_arm=materialized_action_arm,
    )
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    return query_local_e4_semantic_residual(
        encoded,
        layout,
        action_weights,
        rank_margin=args.rank_margin,
        temperature=args.temperature,
        lambda_clean_rank=args.lambda_clean_rank,
        lambda_aug_rank=args.lambda_aug_rank,
        lambda_consistency=args.lambda_consistency,
    )


def materialized_live_shared_e4_loss_v3(
    model,
    store: SpectrumStore,
    examples: list[DirectExample],
    materialized_action_spectra: np.ndarray,
    materialized_action_arm: str,
    objective_reference_by_row: dict[int, np.ndarray],
    device: torch.device,
    args,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, list[dict]]:
    """Complete historical E4 relation with query-equal reduction.

    Clean, action, positive and negative spectra remain live in the same
    encoder graph.  Fixed vectors are used only by E4's preservation term.
    Returning the encoded tensor/layout lets the pre-injection audit inspect
    the four input-role gradients directly rather than trusting loss logs.
    """
    if not examples:
        raise ValueError("V3 live-shared action batch is empty")
    if len({int(example.query_index) for example in examples}) != len(examples):
        raise RuntimeError("V3 mixed two actions for one query in one optimizer step")
    spectra, layout, _ = flatten_direct(
        store,
        examples,
        True,
        materialized_action_spectra=materialized_action_spectra,
        materialized_action_arm=materialized_action_arm,
    )
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    anchors: list[torch.Tensor] = []
    for example in examples:
        rows = [
            int(example.query_row),
            *map(int, example.positive_rows),
            *map(int, example.negative_rows),
        ]
        anchors.append(torch.as_tensor(
            np.stack([objective_reference_by_row[row] for row in rows]),
            device=device,
            dtype=encoded.dtype,
        ))
    loss, report = live_shared_e4_action_objective_v3(
        encoded,
        layout,
        anchors,
        [float(example.official_margin) for example in examples],
        rank_margin=args.rank_margin,
        temperature=args.temperature,
        margin_floor_slack=args.margin_floor_slack,
        lambda_clean_rank=args.lambda_clean_rank,
        lambda_aug_rank=args.lambda_aug_rank,
        lambda_consistency=args.lambda_consistency,
        lambda_margin_floor=args.lambda_margin_floor,
        lambda_preserve=args.lambda_preserve,
    )
    return loss, report, encoded, layout


def _clone_parameter_gradients(
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _gradient_ledger_norm(
    gradients: list[torch.Tensor | None], positions: list[int],
) -> float:
    return math.sqrt(sum(
        float(torch.sum(gradients[index].detach().float() ** 2).double())
        for index in positions if gradients[index] is not None
    ))


def audit_live_shared_v3_preinjection_gradients(
    model,
    store: SpectrumStore,
    examples: list[DirectExample],
    targeted_action_spectra: np.ndarray,
    shuffled_action_spectra: np.ndarray,
    objective_reference_by_row: dict[int, np.ndarray],
    parameter_groups: dict[str, list[torch.nn.Parameter]],
    device: torch.device,
    args,
    maximum_queries: int = 32,
) -> dict[str, object]:
    """Compare targeted/shuffled gradients before any optimizer injection."""
    selected: list[DirectExample] = []
    seen_queries: set[int] = set()
    for example in examples:
        query = int(example.query_index)
        if query in seen_queries:
            continue
        seen_queries.add(query)
        selected.append(example)
        if len(selected) >= maximum_queries:
            break
    if len(selected) < 4:
        raise RuntimeError("V3 pre-injection audit needs at least four distinct queries")
    group_positions = {
        name: {id(parameter) for parameter in parameters}
        for name, parameters in parameter_groups.items()
    }
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    def one_arm(spectra_bank: np.ndarray) -> tuple[
        list[torch.Tensor | None],
        dict[str, float], dict[str, float],
        dict[str, float], dict[str, float],
        dict[str, list[int]], dict[str, bool],
    ]:
        model.zero_grad(set_to_none=True)
        role_norm_sq = {role: 0.0 for role in ("clean", "action", "positive", "negative")}
        role_nonzero = {role: 0 for role in role_norm_sq}
        rank_role_norm_sq = {role: 0.0 for role in role_norm_sq}
        rank_role_nonzero = {role: 0 for role in role_norm_sq}
        rank_zero_queries = {role: [] for role in role_norm_sq}
        rank_anchor_expected_nonzero: dict[tuple[int, str], bool] = {}
        batches = list(batched(selected, args.batch_actions))
        for batch in batches:
            loss, _, encoded, layout = materialized_live_shared_e4_loss_v3(
                model,
                store,
                batch,
                spectra_bank,
                "targeted",
                objective_reference_by_row,
                device,
                args,
            )
            encoded.retain_grad()
            rank_probe_terms: list[torch.Tensor] = []
            for example, item in zip(batch, layout):
                clean = encoded[int(item["clean"])]
                action = encoded[int(item["action"])]
                positives = encoded[list(map(int, item["positive"]))]
                negatives = encoded[list(map(int, item["negative"]))]
                clean_positive_scores = positives @ clean
                clean_negative_scores = negatives @ clean
                action_positive_scores = positives @ action
                action_negative_scores = negatives @ action
                clean_positive_index = int(torch.argmax(clean_positive_scores))
                clean_negative_index = int(torch.argmax(clean_negative_scores))
                action_positive_index = int(torch.argmax(action_positive_scores))
                action_negative_index = int(torch.argmax(action_negative_scores))
                clean_margin = (
                    clean_positive_scores[clean_positive_index]
                    - clean_negative_scores[clean_negative_index]
                )
                action_margin = (
                    action_positive_scores[action_positive_index]
                    - action_negative_scores[action_negative_index]
                )
                # A rank-anchor derivative is mathematically zero when the
                # winning positive and negative embeddings are identical.  A
                # duplicated/indistinguishable spectrum can therefore make one
                # clean anchor rank-degenerate without disconnecting that input
                # from the complete E4 loss.  Record this explicitly instead of
                # demanding an impossible non-zero derivative.
                rank_anchor_expected_nonzero[(int(example.query_index), "clean")] = bool(
                    torch.any(
                        positives[clean_positive_index].detach()
                        != negatives[clean_negative_index].detach()
                    ).item()
                )
                rank_anchor_expected_nonzero[(int(example.query_index), "action")] = bool(
                    torch.any(
                        positives[action_positive_index].detach()
                        != negatives[action_negative_index].detach()
                    ).item()
                )
                rank_probe_terms.extend((
                    F.softplus((args.rank_margin - clean_margin) / args.temperature),
                    F.softplus((args.rank_margin - action_margin) / args.temperature),
                ))
            rank_probe = torch.stack(rank_probe_terms).mean()
            rank_encoded_gradient = torch.autograd.grad(
                rank_probe, encoded, retain_graph=True,
            )[0]
            (loss / len(batches)).backward()
            if encoded.grad is None:
                raise RuntimeError("V3 encoded-role gradient audit did not retain gradients")
            for example, item in zip(batch, layout):
                role_indices = {
                    "clean": [int(item["clean"])],
                    "action": [int(item["action"])],
                    "positive": list(map(int, item["positive"])),
                    "negative": list(map(int, item["negative"])),
                }
                for role, indices in role_indices.items():
                    value = encoded.grad[indices].detach().float()
                    squared = float(torch.sum(value * value).double())
                    role_norm_sq[role] += squared
                    role_nonzero[role] += int(squared > 0)
                    rank_value = rank_encoded_gradient[indices].detach().float()
                    rank_squared = float(torch.sum(rank_value * rank_value).double())
                    rank_role_norm_sq[role] += rank_squared
                    rank_role_nonzero[role] += int(rank_squared > 0)
                    if rank_squared == 0:
                        rank_zero_queries[role].append(int(example.query_index))
        gradients = _clone_parameter_gradients(trainable)
        role_norms = {role: math.sqrt(value) for role, value in role_norm_sq.items()}
        role_fraction = {
            role: role_nonzero[role] / len(selected) for role in role_nonzero
        }
        rank_role_norms = {
            role: math.sqrt(value) for role, value in rank_role_norm_sq.items()
        }
        rank_role_fraction = {
            role: rank_role_nonzero[role] / len(selected)
            for role in rank_role_nonzero
        }
        anchor_expectation_matched = {
            role: all(
                rank_anchor_expected_nonzero[(int(example.query_index), role)]
                == (int(example.query_index) not in set(rank_zero_queries[role]))
                for example in selected
            )
            for role in ("clean", "action")
        }
        return (
            gradients, role_norms, role_fraction,
            rank_role_norms, rank_role_fraction,
            rank_zero_queries, anchor_expectation_matched,
        )

    (
        targeted, targeted_roles, targeted_nonzero,
        targeted_rank_roles, targeted_rank_nonzero,
        targeted_rank_zero_queries, targeted_anchor_expectation_matched,
    ) = one_arm(targeted_action_spectra)
    (
        shuffled, shuffled_roles, shuffled_nonzero,
        shuffled_rank_roles, shuffled_rank_nonzero,
        shuffled_rank_zero_queries, shuffled_anchor_expectation_matched,
    ) = one_arm(shuffled_action_spectra)
    group_report: dict[str, dict[str, float]] = {}
    for name, identifiers in group_positions.items():
        positions = [
            index for index, parameter in enumerate(trainable)
            if id(parameter) in identifiers
        ]
        targeted_norm = _gradient_ledger_norm(targeted, positions)
        shuffled_norm = _gradient_ledger_norm(shuffled, positions)
        difference = [
            None if targeted[index] is None and shuffled[index] is None else (
                (torch.zeros_like(shuffled[index]) if targeted[index] is None else targeted[index])
                - (torch.zeros_like(targeted[index]) if shuffled[index] is None else shuffled[index])
            )
            for index in range(len(trainable))
        ]
        difference_norm = _gradient_ledger_norm(difference, positions)
        dot = sum(
            float(torch.sum(targeted[index].float() * shuffled[index].float()).double())
            for index in positions
            if targeted[index] is not None and shuffled[index] is not None
        )
        cosine = dot / max(targeted_norm * shuffled_norm, 1e-30)
        group_report[name] = {
            "targeted_gradient_norm": targeted_norm,
            "shuffled_gradient_norm": shuffled_norm,
            "targeted_shuffled_cosine": cosine,
            "targeted_shuffled_difference_norm": difference_norm,
            "relative_difference": difference_norm / max(
                targeted_norm, shuffled_norm, 1e-30,
            ),
        }
    model.zero_grad(set_to_none=True)
    roles_live = all(
        value > 0
        for report in (targeted_roles, shuffled_roles)
        for value in report.values()
    )
    every_query_role_live = all(
        math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for report in (targeted_nonzero, shuffled_nonzero)
        for value in report.values()
    )
    ranking_roles_live = all(
        value > 0
        for report in (targeted_rank_roles, shuffled_rank_roles)
        for value in report.values()
    )
    every_query_non_degenerate_ranking_role_live = bool(
        all(targeted_anchor_expectation_matched.values())
        and all(shuffled_anchor_expectation_matched.values())
        and all(
            math.isclose(report[role], 1.0, rel_tol=0.0, abs_tol=1e-12)
            for report in (targeted_rank_nonzero, shuffled_rank_nonzero)
            for role in ("action", "positive", "negative")
        )
    )
    arms_distinct = all(
        report["relative_difference"] > 1e-7
        for report in group_report.values()
    )
    output = {
        "queries": len(selected),
        "targeted_role_gradient_norms": targeted_roles,
        "shuffled_role_gradient_norms": shuffled_roles,
        "targeted_role_nonzero_query_fraction": targeted_nonzero,
        "shuffled_role_nonzero_query_fraction": shuffled_nonzero,
        "targeted_ranking_path_role_gradient_norms": targeted_rank_roles,
        "shuffled_ranking_path_role_gradient_norms": shuffled_rank_roles,
        "targeted_ranking_path_role_nonzero_query_fraction": (
            targeted_rank_nonzero
        ),
        "shuffled_ranking_path_role_nonzero_query_fraction": (
            shuffled_rank_nonzero
        ),
        "targeted_ranking_path_zero_query_indices": targeted_rank_zero_queries,
        "shuffled_ranking_path_zero_query_indices": shuffled_rank_zero_queries,
        "targeted_rank_anchor_analytic_expectation_matched": (
            targeted_anchor_expectation_matched
        ),
        "shuffled_rank_anchor_analytic_expectation_matched": (
            shuffled_anchor_expectation_matched
        ),
        "parameter_groups": group_report,
        "clean_action_positive_negative_all_live": roles_live,
        "clean_action_positive_negative_live_for_every_audited_query": (
            every_query_role_live
        ),
        "clean_action_positive_negative_ranking_paths_all_live": (
            ranking_roles_live
        ),
        "ranking_paths_live_for_every_non_degenerate_audited_query": (
            every_query_non_degenerate_ranking_role_live
        ),
        "targeted_and_shuffled_distinct_before_injector": arms_distinct,
        "optimizer_steps": 0,
        "gate_passed": bool(
            roles_live
            and every_query_role_live
            and ranking_roles_live
            and every_query_non_degenerate_ranking_role_live
            and arms_distinct
        ),
    }
    if not output["gate_passed"]:
        raise RuntimeError(f"V3 pre-injection gradient audit failed: {output}")
    return output


def safety_loss(model, store: SpectrumStore, examples: list[DirectExample],
                official_by_row: dict[int, np.ndarray], device: torch.device,
                args) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_direct(store, examples, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    clean_margin = (
        frozen_reference_margins(clean_z, examples, official_by_row)
        if args.rank_reference_mode == "official"
        else margins(encoded, layout, "clean")
    )
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor = weighted_mean(F.relu(floors - clean_margin), examples)
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    loss = args.lambda_margin_floor * floor + args.lambda_preserve * preserve
    return loss, {
        "safety_margin_floor": float(floor.detach()),
        "safety_preserve": float(preserve.detach()),
        "safety_margin": float(clean_margin.mean().detach()),
    }


def candidate_boundary_safety_loss(
    model, store: SpectrumStore, examples: list[DirectExample],
    anchor_by_row: dict[int, np.ndarray], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Protect all current top-k molecule edges rather than one old maximum."""
    spectra, layout, clean_rows = flatten_direct(store, examples, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    if args.candidate_boundary_version == "v2_molecule_max":
        current = [value.unsqueeze(0) for value in molecule_margin_vectors(
            encoded, layout, "clean",
        )]
        initial_edges = anchor_edge_margin_matrices(
            examples, anchor_by_row, device, encoded.dtype,
        )
        initial = [value.max(dim=0, keepdim=True).values for value in initial_edges]
    else:
        current = edge_margin_matrices(encoded, layout, "clean")
        initial = anchor_edge_margin_matrices(
            examples, anchor_by_row, device, encoded.dtype,
        )
    constraint, metrics = candidate_safety_objective(
        current, initial, slack=args.margin_floor_slack,
        topk_negatives=args.boundary_topk_negatives,
    )
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    anchors = torch.from_numpy(np.stack([anchor_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * anchors, dim=1)).mean()
    loss = args.lambda_margin_floor * constraint + args.lambda_preserve * preserve
    return loss, {
        **metrics,
        "safety_preserve": float(preserve.detach()),
    }


def positive_arm_loss(model, store: SpectrumStore, examples: list[DirectExample],
                      official_by_row: dict[int, np.ndarray], device: torch.device,
                      args) -> tuple[torch.Tensor, dict[str, float]]:
    """Rank an explicit cross-condition positive over the same query negatives.

    The official model supplies only a per-example safety floor and embedding
    preservation target.  It is not a teacher action and cannot choose pairs.
    """
    spectra, layout, clean_rows = flatten_direct(store, examples, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    current_margin = margins(encoded, layout, "clean")
    rank_each = F.softplus((args.rank_margin - current_margin) / args.temperature)

    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    official_margins: list[torch.Tensor] = []
    for item in layout:
        q = official[int(item["clean"])]
        positive_score = torch.max(official[item["positive"]] @ q)
        negative_score = torch.max(official[item["negative"]] @ q)
        official_margins.append(positive_score - negative_score)
    official_margin = torch.stack(official_margins)
    floor_each = F.relu(official_margin - args.margin_floor_slack - current_margin)

    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    rank = weighted_mean(rank_each, examples)
    floor = weighted_mean(floor_each, examples)
    loss = (
        args.lambda_positive_rank * rank
        + args.lambda_positive_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "positive_rank": float(rank.detach()),
        "positive_margin_floor": float(floor.detach()),
        "positive_preserve": float(preserve.detach()),
        "positive_margin": float(current_margin.mean().detach()),
        "positive_margin_pass": float((current_margin > 0).float().mean().detach()),
        "positive_official_margin": float(official_margin.mean().detach()),
    }


def batched(values: list[DirectExample], size: int):
    for left in range(0, len(values), size):
        yield values[left:left + size]


def query_complete_action_batches(
    values: list[DirectExample], maximum_actions: int,
) -> list[list[DirectExample]]:
    """Return one complete query action set per query-weighted optimizer step."""
    if maximum_actions < 1:
        raise ValueError("maximum actions must be positive")
    grouped: dict[int, list[DirectExample]] = {}
    order: list[int] = []
    for value in values:
        query = int(value.query_index)
        if query not in grouped:
            grouped[query] = []
            order.append(query)
        grouped[query].append(value)
    batches: list[list[DirectExample]] = []
    for query in order:
        group = grouped[query]
        if len(group) > maximum_actions:
            raise RuntimeError(
                f"query {query} has {len(group)} actions; cap is {maximum_actions}"
            )
        batches.append(group)
    if sum(map(len, batches)) != len(values):
        raise RuntimeError("query-complete batching lost an action")
    for batch in batches:
        closed: set[int] = set()
        previous: int | None = None
        for value in batch:
            query = int(value.query_index)
            if previous is not None and query != previous:
                closed.add(previous)
            if query in closed:
                raise RuntimeError("query action set is not contiguous")
            previous = query
    return batches


def coverage_first_query_schedules(
    examples: list[DirectExample], epochs: int, seed: int,
) -> list[list[int]]:
    """Expose every routed action once while keeping each query set indivisible."""
    if not examples or epochs < 1:
        raise ValueError("query coverage needs examples and positive epochs")
    grouped: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(int(example.query_index), []).append(index)
    rng = np.random.default_rng(seed)
    queries = np.asarray(sorted(grouped), dtype=np.int64)
    queries = queries[rng.permutation(len(queries))]
    schedules: list[list[int]] = [[] for _ in range(epochs)]
    loads = np.zeros(epochs, dtype=np.int64)
    for query in queries:
        epoch = int(np.argmin(loads))
        group = grouped[int(query)]
        schedules[epoch].extend(group)
        loads[epoch] += len(group)
    flat = [index for schedule in schedules for index in schedule]
    if len(flat) != len(examples) or set(flat) != set(range(len(examples))):
        raise RuntimeError("query coverage did not expose each routed action exactly once")
    index_epoch = {
        index: epoch for epoch, schedule in enumerate(schedules) for index in schedule
    }
    for query, indices in grouped.items():
        containing = {index_epoch[index] for index in indices}
        if len(containing) != 1:
            raise RuntimeError(f"query {query} was split across epochs")
    return schedules


def identity_balanced_epoch(examples: list[DirectExample], rng: np.random.Generator,
                            views_per_identity: int) -> list[DirectExample]:
    """Draw exactly K views per identity and round-robin available policies."""
    if views_per_identity < 1:
        raise ValueError("views-per-identity must be positive")
    groups: dict[str, list[DirectExample]] = {}
    for example in examples:
        groups.setdefault(example.identity, []).append(example)
    output: list[DirectExample] = []
    for identity in sorted(groups):
        values = groups[identity]
        by_policy: dict[str, list[DirectExample]] = {}
        for value in values:
            by_policy.setdefault(value.policy, []).append(value)
        policies = sorted(by_policy)
        policy_order = np.asarray(policies, dtype=object)[rng.permutation(len(policies))]
        local_orders = {
            policy: rng.permutation(len(by_policy[policy])) for policy in policies
        }
        for offset in range(views_per_identity):
            policy = str(policy_order[offset % len(policy_order)])
            order = local_orders[policy]
            local = int(order[(offset // len(policy_order)) % len(order)])
            output.append(by_policy[policy][local])
    rng.shuffle(output)
    return output


def sampling_schedule_sha256(examples: list[DirectExample]) -> str:
    """Hash arm-invariant sampler keys; deliberately exclude the action path."""
    canonical = "\n".join(
        f"{item.query_index}|{item.query_row}|{item.identity}|{item.formula}|"
        f"{item.policy}|{','.join(map(str, item.positive_rows))}|"
        f"{','.join(map(str, item.negative_rows))}"
        for item in examples
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def action_exposure_schedule_sha256(examples: list[DirectExample]) -> str:
    """Hash arm-invariant action exposure order.

    The causal arms deliberately replace ``target_path`` with a clean or
    matched-random path.  Including that payload would make identical sampling
    schedules appear unequal, so this hash contains only the frozen action
    identity, query identity and dose.
    """
    canonical = "\n".join(
        f"{item.query_index}|{item.query_row}|{item.identity}|{item.formula}|"
        f"{item.policy}|{item.action_id}|{item.attenuation:.8f}"
        for item in examples
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_official_reencoding_zero_change(
    graph: CandidateGraph,
    rows: np.ndarray,
    official_embeddings: np.ndarray,
    reencoded_embeddings: np.ndarray,
    queries: np.ndarray,
    *,
    score_atol: float = OFFICIAL_ZERO_CHANGE_SCORE_ATOL,
) -> dict[str, object]:
    """Distinguish harmless strict-tie flips from real initialization drift.

    The frozen cache and a fresh CUDA forward are two fp32 serialization and
    reduction paths for the same hash-pinned official encoder.  Rank is
    discontinuous at a tie, so a raw mismatch count is not a valid identity
    test.  We instead require every held edge score to remain within the
    preregistered tolerance and inspect every negative-vs-positive comparison
    whose Boolean contribution to strict rank changed.  A rank flip is allowed
    only when each changed comparison is inside twice that tolerance in at
    least one of the two equivalent score paths.  Looking only at the best
    negative margin is insufficient: a stable rank-2 query can become rank 3
    when a different negative is tied with the positive.
    """
    rows = np.asarray(rows, dtype=np.int64)
    official_embeddings = np.asarray(official_embeddings, dtype=np.float32)
    reencoded_embeddings = np.asarray(reencoded_embeddings, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.int64)
    if (
        rows.ndim != 1
        or len(np.unique(rows)) != len(rows)
        or official_embeddings.shape != reencoded_embeddings.shape
        or official_embeddings.ndim != 2
        or len(official_embeddings) != len(rows)
        or queries.ndim != 1
        or not len(queries)
        or not np.isfinite(score_atol)
        or score_atol <= 0
    ):
        raise ValueError("official zero-change audit inputs are malformed")
    position = {int(row): index for index, row in enumerate(rows)}
    cosine = np.einsum(
        "ij,ij->i", official_embeddings, reencoded_embeddings,
    )
    maximum_pair_error = 0.0
    query_max_errors: list[float] = []
    mismatches: list[dict[str, object]] = []
    rank_mismatch_count = 0
    nonboundary = 0
    for raw_query in queries:
        query = int(raw_query)
        molecule_left, molecule_right = map(
            int, graph.query_ptr[query:query + 2],
        )
        pair_left = int(graph.molecule_ptr[molecule_left])
        pair_right = int(graph.molecule_ptr[molecule_right])
        try:
            query_position = position[int(graph.query_row[query])]
            candidate_positions = np.asarray([
                position[int(row)]
                for row in graph.pair_candidate_row[pair_left:pair_right]
            ], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(
                "official zero-change audit misses a held graph row"
            ) from error
        official_pair = np.asarray(
            official_embeddings[candidate_positions]
            @ official_embeddings[query_position],
            dtype=np.float32,
        )
        reencoded_pair = np.asarray(
            reencoded_embeddings[candidate_positions]
            @ reencoded_embeddings[query_position],
            dtype=np.float32,
        )
        local_max_error = float(np.max(np.abs(
            official_pair - reencoded_pair,
        )))
        maximum_pair_error = max(maximum_pair_error, local_max_error)
        query_max_errors.append(local_max_error)
        local_ptr = (
            graph.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        )
        official_molecule = np.maximum.reduceat(official_pair, local_ptr[:-1])
        reencoded_molecule = np.maximum.reduceat(reencoded_pair, local_ptr[:-1])
        official_rank = strict_rank(official_molecule)
        reencoded_rank = strict_rank(reencoded_molecule)
        if official_rank != reencoded_rank:
            rank_mismatch_count += 1
            official_rank_gaps = np.asarray(
                official_molecule[1:] - official_molecule[0], dtype=np.float64,
            )
            reencoded_rank_gaps = np.asarray(
                reencoded_molecule[1:] - reencoded_molecule[0], dtype=np.float64,
            )
            switched = (
                (official_rank_gaps >= 0.0) != (reencoded_rank_gaps >= 0.0)
            )
            switched_count = int(np.sum(switched))
            if switched_count <= 0:
                raise RuntimeError(
                    "strict rank changed without a changed candidate comparison"
                )
            switched_min_abs_gap = np.minimum(
                np.abs(official_rank_gaps[switched]),
                np.abs(reencoded_rank_gaps[switched]),
            )
            boundary = bool(np.all(
                switched_min_abs_gap <= 2.0 * float(score_atol)
            ))
            official_margin = float(
                official_molecule[0] - np.max(official_molecule[1:])
            )
            reencoded_margin = float(
                reencoded_molecule[0] - np.max(reencoded_molecule[1:])
            )
            nonboundary += int(not boundary)
            if len(mismatches) < 64:
                mismatches.append({
                    "query_index": query,
                    "query_row": int(graph.query_row[query]),
                    "official_cache_rank": int(official_rank),
                    "fresh_official_rank": int(reencoded_rank),
                    "official_cache_margin": official_margin,
                    "fresh_official_margin": reencoded_margin,
                    "maximum_pair_score_abs_error": local_max_error,
                    "switched_candidate_comparisons": switched_count,
                    "maximum_switched_boundary_min_abs_gap": float(np.max(
                        switched_min_abs_gap,
                    )),
                    "boundary_within_tolerance": bool(boundary),
                })
    mean_cosine = float(np.mean(cosine))
    report: dict[str, object] = {
        "verification": "official_checkpoint_edge_scores_plus_candidate_comparison_tie_aware_ranks",
        "score_tolerance": float(score_atol),
        "preservation_mean": mean_cosine,
        "preservation_minimum": float(np.min(cosine)),
        "rank_mismatches": int(rank_mismatch_count),
        "boundary_rank_mismatches": int(rank_mismatch_count - nonboundary),
        "nonboundary_rank_mismatches": int(nonboundary),
        "maximum_pair_score_abs_error": float(maximum_pair_error),
        "query_p99_maximum_pair_score_abs_error": float(np.quantile(
            np.asarray(query_max_errors, dtype=np.float64), 0.99,
        )),
        "mismatch_detail_first_64": mismatches,
    }
    report["gate_passed"] = bool(
        np.isfinite(cosine).all()
        and mean_cosine >= 0.9999
        and maximum_pair_error <= float(score_atol)
        and nonboundary == 0
    )
    return report


def validate_causal_configuration(args: argparse.Namespace) -> None:
    """Freeze every non-arm degree of freedom in the first attribution trial."""
    if args.causal_arm == "legacy":
        return
    exact = {
        "action_selection": "fixed",
        "policy": "curriculum",
        "action_scope": "all",
        "outer_fold": 0,
        "formula_fold_seed": 20260825,
        "epochs": 4,
        "batch_actions": 4,
        "views_per_identity": 4,
        "error_views_per_identity": 0,
        "positive_spectra": 4,
        "negative_molecules": 8,
        "unfreeze_blocks": 1,
        "direct_transfer_mode": "symmetric",
        "rank_reference_mode": "shared",
        "guided_noise_policy": "none",
    }
    for name, expected in exact.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(
                f"causal attribution freezes --{name.replace('_', '-')}={expected!r}; "
                f"observed {observed!r}"
            )
    floats = {
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "rank_margin": 0.05,
        "temperature": 0.10,
        "lambda_clean_rank": 1.0,
        "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25,
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0,
        "margin_floor_slack": 0.005,
        "safety_ratio": 1.0,
        "safety_stream_weight": 1.0,
        "positive_stream_weight": 0.0,
        "grad_clip": 1.0,
    }
    for name, expected in floats.items():
        observed = float(getattr(args, name))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"causal attribution freezes --{name.replace('_', '-')}={expected}; "
                f"observed {observed}"
            )
    if args.initial_student_checkpoint is not None:
        raise ValueError("causal attribution must start from the official initialization")
    if args.amp:
        raise ValueError("causal attribution freezes full-fp32 training (--no-amp)")
    if not args.run_suffix:
        raise ValueError("causal attribution requires a unique --run-suffix")


def validate_materialized_e4_configuration(args: argparse.Namespace) -> None:
    """Freeze the historical E4 optimization kernel around a new action tensor bank."""
    if args.action_selection != "materialized_routed":
        if args.e4_base_action_dir is not None:
            raise ValueError(
                "--e4-base-action-dir is legal only with materialized_routed "
                "e4_base_semantic_v2"
            )
        if args.materialized_action_dir is not None:
            raise ValueError(
                "--materialized-action-dir is legal only with materialized_routed selection"
            )
        if args.materialized_action_arm != "targeted":
            raise ValueError(
                "--materialized-action-arm is legal only with materialized_routed selection"
            )
        if args.materialized_injection_mode != "one_best_e4":
            raise ValueError(
                "--materialized-injection-mode is legal only with materialized_routed selection"
            )
        if args.optimizer_boundary_mode == "action_injector_v1":
            validate_fixed_e4_action_injector_configuration(args)
        elif args.optimizer_boundary_mode != "ordinary_adamw":
            raise ValueError(f"unknown E4 optimizer boundary: {args.optimizer_boundary_mode}")
        return
    exact = {
        "policy": "curriculum",
        "action_scope": "all",
        "epochs": 4,
        "batch_actions": 4,
        "views_per_identity": 4,
        "error_views_per_identity": 0,
        "positive_spectra": 4,
        "negative_molecules": 8,
        "unfreeze_blocks": 1,
        "direct_transfer_mode": "symmetric",
        "rank_reference_mode": "shared",
        "guided_noise_policy": "none",
        "pmt_arm": "none",
        "candidate_boundary_loss": False,
        "causal_arm": "legacy",
    }
    for name, expected in exact.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(
                f"materialized E4 freezes --{name.replace('_', '-')}={expected!r}; "
                f"observed {observed!r}"
            )
    floats = {
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "rank_margin": 0.05,
        "temperature": 0.10,
        "lambda_clean_rank": 1.0,
        "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25,
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0,
        "margin_floor_slack": 0.005,
        "safety_ratio": 1.0,
        "safety_stream_weight": 1.0,
        "positive_stream_weight": 0.0,
        "grad_clip": 1.0,
    }
    for name, expected in floats.items():
        observed = float(getattr(args, name))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"materialized E4 freezes --{name.replace('_', '-')}={expected}; "
                f"observed {observed}"
            )
    complete_historical_hybrid = (
        args.materialized_injection_mode == "complete_panel_historical_e4"
    )
    signal_preserving_hybrid = (
        args.materialized_injection_mode == "e4_base_semantic_v2"
    )
    live_shared_v3_hybrid = (
        args.materialized_injection_mode == "e4_live_shared_v3"
    )
    official_start_hybrid = complete_historical_hybrid
    if official_start_hybrid:
        if args.initial_student_checkpoint is not None:
            raise ValueError(
                "E4 best-action hybrid must start from official DreaMS, not E8"
            )
    elif args.initial_student_checkpoint is None:
        raise ValueError("materialized E4 requires the routed current-geometry checkpoint")
    if args.amp:
        raise ValueError("materialized E4 freezes full-fp32 training (--no-amp)")
    if not args.run_suffix:
        raise ValueError("materialized E4 requires a unique --run-suffix")
    if signal_preserving_hybrid or live_shared_v3_hybrid:
        if args.e4_base_action_dir is None:
            raise ValueError(
                f"{args.materialized_injection_mode} requires --e4-base-action-dir from the "
                "same corrected graph and E8 geometry"
            )
        required_boundary = (
            "separated_e4_action_v3"
            if live_shared_v3_hybrid else "signal_preserving_v2"
        )
        if args.optimizer_boundary_mode != required_boundary:
            raise ValueError(
                f"{args.materialized_injection_mode} requires "
                f"--optimizer-boundary-mode={required_boundary}"
            )
        injector_floats = {
            "injector_target_attributable_fraction": 0.25,
            "injector_minimum_protective_retention": 0.90,
            "injector_maximum_update_norm_ratio": 1.50,
        }
        for name, expected in injector_floats.items():
            observed = float(getattr(args, name))
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"E4 hybrid freezes --{name.replace('_', '-')}="
                    f"{expected}; observed {observed}"
                )
        if (
            args.outcome_action_dir is not None
            or args.pmt_manifest_dir is not None
            or args.guided_action_authorization_dir is not None
            or args.guided_crossfit_root is not None
        ):
            raise ValueError(
                "E4 hybrid forbids outcome/PMT/guided teacher assets"
            )
    elif args.optimizer_boundary_mode == "action_injector_v1":
        if args.e4_base_action_dir is not None:
            raise ValueError(
                "--e4-base-action-dir is legal only with e4_base_semantic_v2"
            )
        if args.materialized_injection_mode not in {
            "multi_action_balanced", "complete_panel_historical_e4",
        }:
            raise ValueError(
                "E4 ActionInjectorV1 requires the complete multi-action panel; "
                "one-best compression is forbidden"
            )
        injector_floats = {
            "injector_target_attributable_fraction": 0.25,
            "injector_minimum_protective_retention": 0.90,
            "injector_maximum_update_norm_ratio": 1.50,
        }
        for name, expected in injector_floats.items():
            observed = float(getattr(args, name))
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"E4 ActionInjectorV1 freezes --{name.replace('_', '-')}="
                    f"{expected}; observed {observed}"
                )
        if complete_historical_hybrid and (
            args.outcome_action_dir is not None
            or args.pmt_manifest_dir is not None
            or args.guided_action_authorization_dir is not None
            or args.guided_crossfit_root is not None
        ):
            raise ValueError(
                "complete-panel historical E4 forbids outcome/PMT/guided teacher assets"
            )
    elif args.optimizer_boundary_mode != "ordinary_adamw":
        raise ValueError(
            f"unknown E4 optimizer boundary: {args.optimizer_boundary_mode}"
        )
    elif args.e4_base_action_dir is not None:
        raise ValueError(
            "--e4-base-action-dir is legal only with e4_base_semantic_v2 "
            "or e4_live_shared_v3"
        )


def validate_fixed_e4_action_injector_configuration(
    args: argparse.Namespace,
) -> None:
    """Fail closed around the one approved E4-R0 plus Injector V1 hybrid.

    This route keeps the historical E4 action bank, sampler, loss, official
    initialization and trainable capacity.  The only intervention is the
    registered optimizer boundary after the action and safety backwards.
    """
    exact = {
        "action_selection": "fixed",
        "policy": "curriculum",
        "action_scope": "all",
        "outer_fold": 0,
        "formula_fold_seed": 20260825,
        "epochs": 4,
        "batch_actions": 4,
        "views_per_identity": 4,
        "error_views_per_identity": 0,
        "positive_spectra": 4,
        "negative_molecules": 8,
        "unfreeze_blocks": 1,
        "direct_transfer_mode": "symmetric",
        "rank_reference_mode": "shared",
        "guided_noise_policy": "none",
        "pmt_arm": "none",
        "candidate_boundary_loss": False,
        "refresh_hard_negatives": False,
        "causal_arm": "legacy",
    }
    for name, expected in exact.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(
                f"E4-R0 Injector V1 freezes --{name.replace('_', '-')}={expected!r}; "
                f"observed {observed!r}"
            )
    floats = {
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "rank_margin": 0.05,
        "temperature": 0.10,
        "lambda_clean_rank": 1.0,
        "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25,
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0,
        "margin_floor_slack": 0.005,
        "safety_ratio": 1.0,
        "safety_stream_weight": 1.0,
        "positive_stream_weight": 0.0,
        "grad_clip": 1.0,
        "injector_target_attributable_fraction": 0.25,
        "injector_minimum_protective_retention": 0.90,
        "injector_maximum_update_norm_ratio": 1.50,
    }
    for name, expected in floats.items():
        observed = float(getattr(args, name))
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"E4-R0 Injector V1 freezes --{name.replace('_', '-')}={expected}; "
                f"observed {observed}"
            )
    if args.initial_student_checkpoint is not None:
        raise ValueError("E4-R0 Injector V1 must start from official DreaMS, not E8")
    if args.outcome_action_dir is not None or args.materialized_action_dir is not None:
        raise ValueError("E4-R0 Injector V1 forbids outcome-mined/materialized action assets")
    if args.pmt_manifest_dir is not None:
        raise ValueError("E4-R0 Injector V1 forbids PMT assets")
    if args.guided_action_authorization_dir is not None or args.guided_crossfit_root is not None:
        raise ValueError("E4-R0 Injector V1 forbids guided/teacher action assets")
    if args.amp:
        raise ValueError("E4-R0 Injector V1 freezes full-fp32 training (--no-amp)")
    if not args.run_suffix:
        raise ValueError("E4-R0 Injector V1 requires a unique --run-suffix")


def train_e4_base_semantic_v2_epochs(
    *,
    model,
    optimizer: torch.optim.AdamW,
    injector: E4SignalPreservingInjectorBridgeV2,
    store: SpectrumStore,
    e4_base_examples: list[DirectExample],
    semantic_examples: list[DirectExample],
    safety_examples: list[DirectExample],
    materialized_action_spectra: np.ndarray,
    objective_reference_by_row: dict[int, np.ndarray],
    head_parameters: list[torch.nn.Parameter],
    backbone_parameters: list[torch.nn.Parameter],
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
) -> tuple[
    list[dict[str, object]],
    list[SignalPreservingInjectionV2Step],
    dict[str, object],
]:
    """Train corrected E4 directly and add later actions at one audited boundary."""
    if args.amp or args.smoke:
        raise RuntimeError("signal-preserving V2 requires formal full-fp32 execution")
    if epochs != 4 or args.batch_actions != 4 or args.views_per_identity != 4:
        raise RuntimeError("signal-preserving V2 changed the frozen four-epoch E4 schedule")
    if not e4_base_examples or not semantic_examples or not safety_examples:
        raise RuntimeError("signal-preserving V2 received an empty training stream")
    if len({item.action_id for item in semantic_examples}) != len(semantic_examples):
        raise RuntimeError("later semantic action identifiers are empty or duplicated")
    if any(item.materialized_action_index < 0 for item in semantic_examples):
        raise RuntimeError("later semantic stream lost a materialized action tensor")
    if any(item.materialized_action_index >= 0 for item in e4_base_examples):
        raise RuntimeError("corrected E4 base was silently replaced by materialized outcomes")

    # Reuse the complete historical E4 objective without letting the outer
    # materialized mode activate its later-action gates or reductions.
    e4_args = copy.copy(args)
    e4_args.action_selection = "fixed"
    e4_args.materialized_injection_mode = "one_best_e4"
    e4_args.optimizer_boundary_mode = "ordinary_adamw"

    base_rng = np.random.default_rng(args.seed)
    base_epochs: list[list[DirectExample]] = []
    safety_epochs: list[list[DirectExample]] = []
    expected_base_identities = len({item.identity for item in e4_base_examples})
    expected_steps = expected_base_identities
    for _ in range(epochs):
        base_epoch = identity_balanced_epoch(
            e4_base_examples, base_rng, args.views_per_identity,
        )
        if len(base_epoch) != expected_base_identities * args.views_per_identity:
            raise RuntimeError("corrected E4 base identity dose drifted")
        base_epochs.append(base_epoch)
        safety_epochs.append(identity_balanced_epoch(safety_examples, base_rng, 1))

    semantic_source_families: list[str] = []
    for item in semantic_examples:
        policy_parts = item.policy.split("|", 2)
        if len(policy_parts) != 3 or not all(policy_parts[:2]):
            raise RuntimeError(
                "later semantic action lost its explicit source/family namespace"
            )
        semantic_source_families.append("|".join(policy_parts[:2]))
    semantic_plan = build_identity_equal_action_bag_plan(
        [item.action_id for item in semantic_examples],
        [item.identity for item in semantic_examples],
        semantic_source_families,
        epochs=epochs,
        bags_per_identity_per_epoch=args.views_per_identity,
        optimizer_steps_per_epoch=expected_steps,
        seed=args.seed + 314159,
    )
    spectra_per_action = [
        2 + len(item.positive_rows) + len(item.negative_rows)
        for item in semantic_examples
    ]
    if max(spectra_per_action) > 64:
        raise RuntimeError("one later semantic action exceeds the 64-spectrum forward cap")

    history: list[dict[str, object]] = []
    receipts: list[SignalPreservingInjectionV2Step] = []
    semantic_schedule_payload: list[str] = []
    for epoch in range(1, epochs + 1):
        model.eval()
        base_epoch = base_epochs[epoch - 1]
        safety_epoch = safety_epochs[epoch - 1]
        action_batches = list(batched(base_epoch, args.batch_actions))
        if len(action_batches) != expected_steps:
            raise RuntimeError("corrected E4 base optimizer-step count drifted")
        semantic_steps = semantic_plan.epoch_steps[epoch - 1]
        if len(semantic_steps) != len(action_batches):
            raise RuntimeError("later semantic plan is not aligned to E4 base steps")
        totals: dict[str, float] = {}
        started = time.time()
        safety_cursor = 0
        for step, (base_batch, semantic_bags) in enumerate(
            zip(action_batches, semantic_steps), start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            semantic_log: dict[str, float] = {}
            semantic_loss_value = 0.0
            semantic_action_count = 0
            semantic_microbatch_count = 0
            if semantic_bags:
                semantic_indices, semantic_weights = flatten_weighted_action_bags(
                    semantic_bags
                )
                microbatches = bounded_semantic_microbatches(
                    semantic_indices,
                    semantic_weights,
                    spectra_per_action,
                    maximum_spectra_per_forward=64,
                )
                for micro_indices, micro_weights in microbatches:
                    micro_examples = [semantic_examples[index] for index in micro_indices]
                    semantic_loss, micro_log = materialized_query_local_semantic_loss(
                        model,
                        store,
                        micro_examples,
                        micro_weights,
                        materialized_action_spectra,
                        args.materialized_action_arm,
                        device,
                        args,
                    )
                    semantic_loss.backward()
                    semantic_loss_value += float(semantic_loss.detach())
                    semantic_action_count += len(micro_examples)
                    semantic_microbatch_count += 1
                    for key, value in micro_log.items():
                        if isinstance(value, bool):
                            continue
                        semantic_log[key] = semantic_log.get(key, 0.0) + float(value)
                injector.capture_semantic_corrective_()
                semantic_schedule_payload.append(
                    f"{epoch}|{step}|" + ",".join(
                        f"{semantic_examples[index].action_id}:{weight:.17g}"
                        for index, weight in zip(semantic_indices, semantic_weights)
                    )
                )
            else:
                injector.capture_semantic_corrective_(allow_zero=True)
                semantic_schedule_payload.append(f"{epoch}|{step}|ZERO")

            # Clear only the live .grad fields.  The injector owns a detached
            # copy of the later semantic residual and receives no spectra or
            # action metadata.
            optimizer.zero_grad(set_to_none=True)
            base_loss, base_log = direct_action_loss(
                model,
                store,
                base_batch,
                objective_reference_by_row,
                device,
                e4_args,
                official_action_targets=None,
                materialized_action_spectra=None,
                materialized_action_arm="targeted",
            )
            base_loss.backward()

            safety_size = len(base_batch)
            if safety_cursor + safety_size > len(safety_epoch):
                base_rng.shuffle(safety_epoch)
                safety_cursor = 0
            safe_batch = safety_epoch[safety_cursor:safety_cursor + safety_size]
            safety_cursor += safety_size
            if len(safe_batch) != safety_size:
                raise RuntimeError("corrected E4 safety sampler returned a short batch")
            safe_loss, safe_log = safety_loss(
                model,
                store,
                safe_batch,
                objective_reference_by_row,
                device,
                e4_args,
            )
            (args.safety_stream_weight * safe_loss).backward()

            head_grad_norm = gradient_l2_norm(head_parameters)
            backbone_grad_norm = gradient_l2_norm(backbone_parameters)
            receipt = injector.step_and_inject_(maximum_gradient_norm=args.grad_clip)
            receipts.append(receipt)
            step_log = {
                "loss": float(base_loss.detach())
                + args.safety_stream_weight * float(safe_loss.detach())
                + semantic_loss_value,
                "historical_e4_loss": float(base_loss.detach()),
                "safety_loss": float(safe_loss.detach()),
                "later_semantic_loss": semantic_loss_value,
                "later_semantic_active": float(receipt.semantic_active),
                "later_semantic_action_count": float(semantic_action_count),
                "later_semantic_microbatches": float(semantic_microbatch_count),
                "head_gradient_norm_before_e4_clip": head_grad_norm,
                "backbone_gradient_norm_before_e4_clip": backbone_grad_norm,
                "historical_e4_gradient_norm": (
                    receipt.historical_e4_gradient_norm_before_clip
                ),
                "semantic_gradient_norm": receipt.semantic_gradient_norm_before_clip,
                "historical_e4_clip_retention": receipt.historical_e4_clip_retention,
                "combined_clip_retention": receipt.combined_clip_retention,
                "final_to_historical_e4_update_norm_ratio": (
                    receipt.final_to_historical_e4_update_norm_ratio
                ),
                "maximum_optimizer_action_fraction_error": (
                    receipt.maximum_fraction_abs_error
                ),
                **base_log,
                **safe_log,
                **semantic_log,
            }
            for key, value in step_log.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if step % 100 == 0 or step == expected_steps:
                print(
                    f"[E4+semantic-v2 epoch={epoch}] {step}/{expected_steps} "
                    f"base={totals['historical_e4_loss']/step:.5f} "
                    f"semantic-active={int(totals['later_semantic_active'])}",
                    flush=True,
                )
        record: dict[str, object] = {
            key: value / expected_steps for key, value in totals.items()
        }
        record.update({
            "epoch": epoch,
            "steps": expected_steps,
            "seconds": time.time() - started,
            "e4_base_sampling_schedule_sha256": sampling_schedule_sha256(base_epoch),
            "e4_base_action_exposure_sha256": action_exposure_schedule_sha256(base_epoch),
            "safety_sampling_schedule_sha256": sampling_schedule_sha256(safety_epoch),
            "later_semantic_active_steps": int(sum(bool(value) for value in semantic_steps)),
            "later_semantic_zero_steps": int(sum(not value for value in semantic_steps)),
        })
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    schedule_report = semantic_plan.audit_manifest()
    e4_base_exposed_action_ids = {
        str(item.action_id) for epoch_rows in base_epochs for item in epoch_rows
    }
    e4_base_exposed_query_indices = {
        int(item.query_index) for epoch_rows in base_epochs for item in epoch_rows
    }
    e4_base_total_action_rows = int(len(e4_base_examples))
    e4_base_unique_action_rows_exposed = int(len(e4_base_exposed_action_ids))
    schedule_report.update({
        "e4_base_action_rows": e4_base_total_action_rows,
        "e4_base_identities": int(expected_base_identities),
        "e4_base_views_per_identity_per_epoch": int(args.views_per_identity),
        "e4_base_optimizer_steps_per_epoch": int(expected_steps),
        "e4_base_physical_action_exposures": int(sum(map(len, base_epochs))),
        "e4_base_unique_action_rows_exposed": e4_base_unique_action_rows_exposed,
        "e4_base_unique_action_row_coverage_fraction": float(
            e4_base_unique_action_rows_exposed / e4_base_total_action_rows
        ),
        "e4_base_action_rows_not_exposed": int(
            e4_base_total_action_rows - e4_base_unique_action_rows_exposed
        ),
        "e4_base_unique_queries_exposed": int(len(e4_base_exposed_query_indices)),
        "e4_base_bank_is_validated_supplier_not_full_row_coverage": True,
        "e4_base_historical_identity_balanced_sampler_preserved": True,
        "later_action_schedule_sha256": hashlib.sha256(
            "\n".join(semantic_schedule_payload).encode("utf-8")
        ).hexdigest(),
        "maximum_spectra_per_semantic_forward": 64,
        "later_actions_share_encoder_and_are_not_teacher_targets": True,
        "each_later_identity_has_four_distinct_optimizer_opportunities_per_epoch": True,
        "later_source_family_equal_effective_dose_within_identity": True,
    })
    return history, receipts, schedule_report


def train_e4_live_shared_v3_epochs(
    *,
    model,
    optimizer: torch.optim.AdamW,
    injector: SeparatedE4ActionInjectorV3,
    store: SpectrumStore,
    e4_base_examples: list[DirectExample],
    action_examples: list[DirectExample],
    safety_examples: list[DirectExample],
    materialized_action_spectra: np.ndarray,
    objective_reference_by_row: dict[int, np.ndarray],
    head_parameters: list[torch.nn.Parameter],
    backbone_parameters: list[torch.nn.Parameter],
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
) -> tuple[
    list[dict[str, object]],
    list[SeparatedE4ActionStepV3],
    dict[str, object],
]:
    """Train the intact E4 base plus full live-shared later-action E4 steps."""
    if args.amp or args.smoke:
        raise RuntimeError("live-shared V3 requires formal full-fp32 execution")
    if epochs != 4 or args.batch_actions != 4 or args.views_per_identity != 4:
        raise RuntimeError("live-shared V3 changed the frozen four-epoch E4 schedule")
    if not e4_base_examples or not action_examples or not safety_examples:
        raise RuntimeError("live-shared V3 received an empty training stream")
    if len({item.action_id for item in action_examples}) != len(action_examples):
        raise RuntimeError("live-shared V3 action identifiers are empty or duplicated")
    if any(item.materialized_action_index < 0 for item in action_examples):
        raise RuntimeError("live-shared V3 lost a materialized action tensor")

    e4_args = copy.copy(args)
    e4_args.action_selection = "fixed"
    e4_args.materialized_injection_mode = "one_best_e4"
    e4_args.optimizer_boundary_mode = "ordinary_adamw"
    rng = np.random.default_rng(args.seed)
    expected_identities = len({item.identity for item in e4_base_examples})
    expected_steps = expected_identities
    base_epochs: list[list[DirectExample]] = []
    safety_epochs: list[list[DirectExample]] = []
    for _ in range(epochs):
        base_epoch = identity_balanced_epoch(
            e4_base_examples, rng, args.views_per_identity,
        )
        if len(base_epoch) != expected_identities * args.views_per_identity:
            raise RuntimeError("live-shared V3 E4 base identity dose drifted")
        base_epochs.append(base_epoch)
        safety_epochs.append(identity_balanced_epoch(safety_examples, rng, 1))

    source_families: list[str] = []
    for item in action_examples:
        parts = str(item.policy).split("|", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise RuntimeError("live-shared V3 action lost source/family provenance")
        source_families.append("|".join(parts[:2]))
    plan: QueryEqualActionPlanV3 = build_query_equal_action_plan_v3(
        [item.action_id for item in action_examples],
        [int(item.query_index) for item in action_examples],
        source_families,
        epochs=epochs,
        views_per_query_per_epoch=args.views_per_identity,
        actions_per_step=args.batch_actions,
        optimizer_steps_per_epoch=expected_steps,
        seed=args.seed + 271828,
    )
    if plan.maximum_actions_per_query > epochs * args.views_per_identity:
        raise RuntimeError(
            "live-shared V3 cannot expose every action without unequal query dose"
        )
    if plan.unique_actions_exposed != plan.action_count:
        raise RuntimeError("live-shared V3 schedule did not expose every best action")

    history: list[dict[str, object]] = []
    receipts: list[SeparatedE4ActionStepV3] = []
    action_schedule_payload: list[str] = []
    for epoch in range(1, epochs + 1):
        model.eval()
        base_batches = list(batched(base_epochs[epoch - 1], args.batch_actions))
        if len(base_batches) != expected_steps:
            raise RuntimeError("live-shared V3 changed the E4 optimizer-step count")
        action_steps = plan.epoch_steps[epoch - 1]
        safety_epoch = safety_epochs[epoch - 1]
        safety_cursor = 0
        totals: dict[str, float] = {}
        started = time.time()
        for step_index, (base_batch, action_indices) in enumerate(
            zip(base_batches, action_steps), start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            action_log: dict[str, float] = {}
            action_loss_value = 0.0
            if action_indices:
                action_batch = [action_examples[index] for index in action_indices]
                spectra_count = sum(
                    2 + len(item.positive_rows) + len(item.negative_rows)
                    for item in action_batch
                )
                if spectra_count > 64:
                    raise RuntimeError(
                        f"live-shared V3 forward exceeds 64 spectra: {spectra_count}"
                    )
                action_loss, action_log, _, _ = materialized_live_shared_e4_loss_v3(
                    model,
                    store,
                    action_batch,
                    materialized_action_spectra,
                    args.materialized_action_arm,
                    objective_reference_by_row,
                    device,
                    args,
                )
                action_loss.backward()
                action_loss_value = float(action_loss.detach())
                injector.capture_action_gradient_()
                action_schedule_payload.append(
                    f"{epoch}|{step_index}|" + ",".join(
                        action_examples[index].action_id for index in action_indices
                    )
                )
            else:
                (sum(parameter.sum() * 0.0 for parameter in head_parameters)).backward()
                injector.capture_action_gradient_(allow_zero=True)
                action_schedule_payload.append(f"{epoch}|{step_index}|ZERO")

            optimizer.zero_grad(set_to_none=True)
            base_loss, base_log = direct_action_loss(
                model,
                store,
                base_batch,
                objective_reference_by_row,
                device,
                e4_args,
                official_action_targets=None,
                materialized_action_spectra=None,
                materialized_action_arm="targeted",
            )
            base_loss.backward()
            safety_size = len(base_batch)
            if safety_cursor + safety_size > len(safety_epoch):
                rng.shuffle(safety_epoch)
                safety_cursor = 0
            safe_batch = safety_epoch[safety_cursor:safety_cursor + safety_size]
            safety_cursor += safety_size
            if len(safe_batch) != safety_size:
                raise RuntimeError("live-shared V3 safety sampler returned a short batch")
            safe_loss, safe_log = safety_loss(
                model,
                store,
                safe_batch,
                objective_reference_by_row,
                device,
                e4_args,
            )
            (args.safety_stream_weight * safe_loss).backward()
            head_grad_norm = gradient_l2_norm(head_parameters)
            backbone_grad_norm = gradient_l2_norm(backbone_parameters)
            receipt = injector.step_(maximum_gradient_norm=args.grad_clip)
            receipts.append(receipt)
            step_log = {
                "loss": (
                    float(base_loss.detach())
                    + args.safety_stream_weight * float(safe_loss.detach())
                    + action_loss_value
                ),
                "historical_e4_loss": float(base_loss.detach()),
                "safety_loss": float(safe_loss.detach()),
                "later_full_e4_action_loss": action_loss_value,
                "later_action_active": float(receipt.action_active),
                "later_actions_in_step": float(len(action_indices)),
                "head_gradient_norm_before_e4_clip": head_grad_norm,
                "backbone_gradient_norm_before_e4_clip": backbone_grad_norm,
                "e4_gradient_norm": receipt.e4_gradient_norm_before_clip,
                "action_gradient_norm": receipt.action_gradient_norm_before_clip,
                "e4_clip_retention": receipt.e4_clip_retention,
                "action_clip_retention": receipt.action_clip_retention,
                "action_rejected_as_opposed": float(
                    receipt.action_rejected_as_opposed
                ),
                "final_action_fraction_max": max(
                    receipt.final_action_fraction_by_group.values()
                ),
                **base_log,
                **safe_log,
                **action_log,
            }
            for key, value in step_log.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if step_index % 100 == 0 or step_index == expected_steps:
                print(
                    f"[E4+live-shared-v3 epoch={epoch}] "
                    f"{step_index}/{expected_steps} "
                    f"base={totals['historical_e4_loss']/step_index:.5f} "
                    f"action-active={int(totals['later_action_active'])}",
                    flush=True,
                )
        record: dict[str, object] = {
            key: value / expected_steps for key, value in totals.items()
        }
        record.update({
            "epoch": epoch,
            "steps": expected_steps,
            "seconds": time.time() - started,
            "e4_base_sampling_schedule_sha256": sampling_schedule_sha256(
                base_epochs[epoch - 1]
            ),
            "safety_sampling_schedule_sha256": sampling_schedule_sha256(safety_epoch),
            "later_action_active_steps": sum(bool(value) for value in action_steps),
            "later_action_zero_steps": sum(not value for value in action_steps),
        })
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    schedule_report = plan.audit_manifest()
    exposed_base_action_ids = {
        str(item.action_id) for values in base_epochs for item in values
    }
    exposed_base_query_indices = {
        int(item.query_index) for values in base_epochs for item in values
    }
    schedule_report.update({
        "later_action_schedule_sha256": hashlib.sha256(
            "\n".join(action_schedule_payload).encode("utf-8")
        ).hexdigest(),
        "complete_live_shared_e4_loss": True,
        "clean_action_positive_negative_all_trainable": True,
        "action_rank_hard_gate": False,
        "same_query_actions_mixed_before_optimizer": False,
        "teacher_embedding_or_margin_target": False,
        "e4_base_action_rows": len(e4_base_examples),
        "e4_base_identities": expected_identities,
        "e4_base_views_per_identity_per_epoch": args.views_per_identity,
        "e4_base_optimizer_steps_per_epoch": expected_steps,
        "e4_base_physical_action_exposures": sum(map(len, base_epochs)),
        "e4_base_unique_action_rows_exposed": len(exposed_base_action_ids),
        "e4_base_unique_action_row_coverage_fraction": (
            len(exposed_base_action_ids) / len(e4_base_examples)
        ),
        "e4_base_action_rows_not_exposed": (
            len(e4_base_examples) - len(exposed_base_action_ids)
        ),
        "e4_base_unique_queries_exposed": len(exposed_base_query_indices),
        "e4_base_bank_is_validated_supplier_not_full_row_coverage": True,
        "e4_base_historical_identity_balanced_sampler_preserved": True,
    })
    return history, receipts, schedule_report


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E4-A direct augmentation requires CUDA")
    if args.head_lr < args.backbone_lr or args.backbone_lr <= 0:
        raise ValueError("require head-lr >= backbone-lr > 0")
    validate_causal_configuration(args)
    validate_materialized_e4_configuration(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    required = [
        args.graph, args.data, args.embedding_cache, args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    signal_preserving_v2 = bool(
        args.action_selection == "materialized_routed"
        and args.materialized_injection_mode == "e4_base_semantic_v2"
    )
    live_shared_v3 = bool(
        args.action_selection == "materialized_routed"
        and args.materialized_injection_mode == "e4_live_shared_v3"
    )
    if args.action_selection != "materialized_routed":
        required.extend([
            args.r0_dir / "report.json",
            args.r0_dir / "training_actions.csv.gz",
        ])
    else:
        if args.materialized_action_dir is None:
            raise ValueError(
                "materialized_routed action selection requires --materialized-action-dir"
            )
        required.extend([
            args.materialized_action_dir / "report.json",
            args.materialized_action_dir / "training_actions.csv.gz",
            args.materialized_action_dir / "action_spectra.npz",
        ])
        if signal_preserving_v2 or live_shared_v3:
            if args.e4_base_action_dir is None:
                raise RuntimeError("E4 hybrid base action directory disappeared")
            required.extend([
                args.e4_base_action_dir / "report.json",
                args.e4_base_action_dir / "training_actions.csv.gz",
            ])
        if args.source_manifest is None:
            raise ValueError("materialized_routed requires --source-manifest")
        required.append(args.source_manifest)
    if args.pmt_arm != "none":
        if args.pmt_manifest_dir is None:
            raise ValueError("PMT arm requires --pmt-manifest-dir")
        required.extend([
            args.pmt_manifest_dir / "report.json",
            args.pmt_manifest_dir / "corrective_actions.csv.gz",
            args.pmt_manifest_dir / "all_routed_actions.csv.gz",
        ])
    if args.action_selection == "outcome_mined":
        if args.outcome_action_dir is None:
            raise ValueError("outcome_mined action selection requires --outcome-action-dir")
        required.extend([
            args.outcome_action_dir / "report.json",
            args.outcome_action_dir / "corrective_teacher_actions.csv.gz",
        ])
    if args.positive_stream_weight > 0:
        required.extend([
            args.positive_manifest_dir / "report.json",
            args.positive_manifest_dir / "positive_pairs.csv.gz",
        ])
    initial_decision_path: Path | None = None
    initial_ledger_path: Path | None = None
    if args.initial_student_checkpoint is not None:
        initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
        initial_ledger_path = args.initial_student_checkpoint.parent / "held_per_query.csv.gz"
        required.extend([args.initial_student_checkpoint, initial_decision_path])
    if args.guided_noise_policy == "selected":
        if args.guided_crossfit_root is None:
            raise ValueError("selected guided noise requires --guided-crossfit-root")
        if args.initial_student_checkpoint is None:
            raise ValueError("selected guided noise requires --initial-student-checkpoint")
    elif args.guided_noise_policy != "none":
        required.extend([
            args.guided_intensity_dir / "report.json",
            args.guided_intensity_dir / "action_manifest.csv.gz",
            args.guided_transfer_dir / "report.json",
            args.guided_transfer_dir / "action_manifest.csv.gz",
            args.error_signatures,
        ])
        if args.guided_action_authorization_dir is not None:
            required.append(args.guided_action_authorization_dir / "report.json")
            if args.guided_noise_policy in {"transfer", "both"}:
                if args.guided_reference_checkpoint is None:
                    raise ValueError(
                        "E12-B-authorized transfer requires --guided-reference-checkpoint"
                    )
                required.append(args.guided_reference_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    r0: dict = {}
    if args.action_selection != "materialized_routed":
        r0 = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
        if not r0.get("formal") or r0.get("contracts", {}).get("P2b") != "forbidden":
            raise RuntimeError("E4-A requires the formal, P2b-free R0 manifest")
        if not r0.get("contracts", {}).get("action_outcomes_absent_from_training_manifest"):
            raise RuntimeError("R0 does not certify outcome-free training actions")
    if args.positive_stream_weight < 0 or args.positive_ratio < 0:
        raise ValueError("positive stream weight/ratio must be nonnegative")
    if args.safety_ratio <= 0 or args.safety_stream_weight <= 0:
        raise ValueError("safety-ratio and safety-stream-weight must be positive")
    if (
        args.guided_noise_weight < 0
        or args.guided_noise_ratio < 0
        or args.guided_risk_control_ratio < 0
    ):
        raise ValueError("guided noise weight/ratio must be nonnegative")
    if args.guided_noise_policy != "none" and args.guided_noise_weight <= 0:
        raise ValueError("guided noise policy requires a positive guided-noise-weight")
    if args.lambda_guided_transfer < 0 or args.lambda_guided_teacher_margin < 0:
        raise ValueError("guided transfer weights must be nonnegative")
    if args.guided_teacher_margin_cap <= 0:
        raise ValueError("guided teacher margin cap must be positive")
    if not 0 < args.guided_teacher_delta_fraction <= 1:
        raise ValueError("guided teacher delta fraction must be in (0, 1]")
    if args.guided_teacher_delta_cap <= 0:
        raise ValueError("guided teacher delta cap must be positive")
    if args.guided_risk_control_ratio > 0 and args.guided_noise_policy != "selected":
        raise ValueError("guided risk controls are available only in selected E14 mode")
    if args.guided_auto_balance and args.guided_noise_policy != "selected":
        raise ValueError("guided auto-balance is available only in selected E14 mode")
    if args.guided_noise_views_per_identity < 1:
        raise ValueError("guided-noise-views-per-identity must be positive")
    if not 0 < args.guided_recurrence_prevalence <= 1:
        raise ValueError("guided-recurrence-prevalence must be in (0, 1]")
    if args.guided_recurrence_max_peaks < 1:
        raise ValueError("guided-recurrence-max-peaks must be positive")
    nonhistorical_guided_recipe = (
        not math.isclose(args.guided_recurrence_prevalence, 0.67, abs_tol=1e-12)
        or args.guided_recurrence_max_peaks != 5
        or args.guided_transfer_mode == "symmetric"
        or args.guided_query_scope == "all"
    )
    if (
        args.guided_noise_policy in {"transfer", "both"}
        and nonhistorical_guided_recipe
        and args.guided_action_authorization_dir is None
    ):
        raise ValueError(
            "non-historical guided transfer requires --guided-action-authorization-dir"
        )
    if args.error_views_per_identity < 0:
        raise ValueError("error-views-per-identity must be nonnegative")
    if args.pmt_arm == "none" and args.pmt_alpha != 0:
        raise ValueError("pmt-alpha requires an active PMT arm")
    if args.pmt_arm != "paired_target" and args.pmt_alpha != 0:
        raise ValueError("only paired_target may use a nonzero pmt-alpha")
    if (
        args.pmt_arm == "paired_target" and not args.candidate_boundary_loss
        and not (0 < args.pmt_alpha <= 1)
    ):
        raise ValueError("scalar paired_target requires 0 < pmt-alpha <= 1")
    if args.pmt_advantage_cap <= 0 or args.pmt_preference_gap <= 0:
        raise ValueError("PMT advantage cap and preference gap must be positive")
    if args.candidate_boundary_loss and args.pmt_arm != "paired_target":
        raise ValueError("candidate-boundary-loss requires pmt-arm=paired_target")
    if args.candidate_boundary_loss and args.pmt_alpha != 0:
        raise ValueError("candidate-boundary loss forbids scalar pmt-alpha inheritance")
    if args.boundary_topk_negatives < 1 or args.boundary_topk_negatives > args.negative_molecules:
        raise ValueError("boundary-topk-negatives must be within sampled negative molecules")
    if min(
        args.boundary_advantage_temperature, args.boundary_hard_temperature,
        args.lambda_boundary_clean, args.lambda_boundary_target,
        args.lambda_boundary_counterfactual, args.lambda_boundary_full_clean,
    ) <= 0:
        raise ValueError("candidate-boundary temperatures and weights must be positive")
    if (
        args.boundary_calibration_formulas < 32
        or args.boundary_calibration_examples < 1
        or args.boundary_action_to_safety_norm_ratio <= 0
        or args.boundary_action_scale_cap < 1
    ):
        raise ValueError("candidate-boundary gradient calibration is underspecified")
    if args.pmt_arm != "none" and (
        args.action_selection != "fixed" or args.policy != "curriculum"
        or args.positive_stream_weight != 0 or args.guided_noise_policy != "none"
        or args.causal_arm != "legacy"
    ):
        raise ValueError("PMT is N-only fixed curriculum and cannot mix legacy causal/P branches")

    if args.run_suffix and not all(
        character.isalnum() or character in "-_" for character in args.run_suffix
    ):
        raise ValueError("run-suffix may contain only letters, digits, '-' and '_'")
    tag = (
        f"{args.policy}_{args.action_scope}_views{args.views_per_identity}_blocks{args.unfreeze_blocks}_"
        f"blr_{args.backbone_lr:.0e}_hlr_{args.head_lr:.0e}"
    )
    if args.action_selection != "fixed":
        tag += f"_as_{args.action_selection}"
    if args.action_selection == "materialized_routed":
        tag += f"_marm_{args.materialized_action_arm}"
        if args.materialized_injection_mode != "one_best_e4":
            tag += f"_minj_{args.materialized_injection_mode}"
        if args.optimizer_boundary_mode != "ordinary_adamw":
            tag += f"_ob_{args.optimizer_boundary_mode}"
    elif args.optimizer_boundary_mode != "ordinary_adamw":
        tag += f"_ob_{args.optimizer_boundary_mode}"
    if args.initial_student_checkpoint is not None:
        tag += "_warm"
    if args.positive_stream_weight > 0:
        tag += f"_pnw_{args.positive_stream_weight:g}_pv{args.positive_views_per_identity}"
    if args.guided_noise_policy != "none":
        tag += (
            f"_gpn_{args.guided_noise_policy}_gw{args.guided_noise_weight:g}"
            f"_gv{args.guided_noise_views_per_identity}"
            f"_gtm_{args.guided_transfer_mode}"
            f"_grp{args.guided_recurrence_prevalence:g}"
            f"_gmax{args.guided_recurrence_max_peaks}"
            f"_gscope_{args.guided_query_scope}"
        )
        if args.guided_noise_policy == "selected":
            tag += (
                f"_gtmargin{args.lambda_guided_teacher_margin:g}"
                f"_gttarget{args.guided_teacher_target_mode}"
                f"_gtdf{args.guided_teacher_delta_fraction:g}"
                f"_grisk{args.guided_risk_control_ratio:g}"
                f"_gbal{int(args.guided_auto_balance)}"
            )
    if args.safety_stream_weight != 1.0:
        tag += f"_sw{args.safety_stream_weight:g}"
    if args.error_views_per_identity > 0:
        tag += f"_ev{args.error_views_per_identity}"
    if args.run_suffix:
        tag += f"_{args.run_suffix}"
    if args.causal_arm != "legacy":
        tag += f"_causal_{args.causal_arm}"
    if args.pmt_arm != "none":
        tag += f"_pmt_{args.pmt_arm}_alpha{args.pmt_alpha:g}"
    if args.candidate_boundary_loss:
        tag += f"_candidate_boundary_{args.candidate_boundary_version}"
    output = args.output_root / tag / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite E4-A result: {output}")

    graph = CandidateGraph(args.graph)
    corrected_query_adduct: np.ndarray | None = None
    if args.source_manifest is not None:
        with np.load(args.source_manifest, allow_pickle=False) as source:
            required_source = {"query_row", "query_ik14", "query_formula", "query_adduct"}
            if missing_source := required_source - set(source.files):
                raise RuntimeError(
                    f"corrected source manifest lacks fields: {sorted(missing_source)}"
                )
            if (
                not np.array_equal(np.asarray(source["query_row"]), graph.query_row)
                or not np.array_equal(np.asarray(source["query_ik14"], dtype=str), graph.query_ik14)
                or not np.array_equal(
                    np.asarray(source["query_formula"], dtype=str), graph.query_formula
                )
            ):
                raise RuntimeError("corrected source manifest does not align to the graph")
            corrected_query_adduct = np.asarray(source["query_adduct"], dtype=str)
        if corrected_query_adduct.shape != (graph.n_queries,):
            raise RuntimeError("corrected query adducts do not align to the graph")
    official_rank, official_margin = official_rank_margin(graph)
    outcome_report: dict = {}
    materialized_action_report: dict = {}
    e4_base_action_report: dict[str, object] = {}
    materialized_action_control_report: dict[str, object] = {}
    materialized_action_spectra: np.ndarray | None = None
    v3_targeted_action_spectra: np.ndarray | None = None
    v3_shuffled_action_spectra: np.ndarray | None = None
    materialized_expected_margin_by_action: dict[str, float] = {}
    materialized_expected_clean_margin_by_action: dict[str, float] = {}
    pmt_manifest_corrective_query_scope: str | None = None
    if args.pmt_arm != "none":
        pmt_report = json.loads(
            (args.pmt_manifest_dir / "report.json").read_text(encoding="utf-8")
        )
        pmt_contract = pmt_report.get("contracts", {})
        pmt_manifest_corrective_query_scope = str(
            pmt_report.get("corrective_query_scope", "all")
        )
        if (
            pmt_report.get("status") != "noise_final_e4_pmt_manifest_complete"
            or pmt_report.get("formal") is not True
            or int(pmt_report.get("outer_formula_fold", -1)) != args.outer_fold
            or pmt_contract.get("noncorrective_target_weight_exact_zero") is not True
            or pmt_contract.get("M2_predictions_used") is not False
            or pmt_contract.get("P_actions_used") is not False
        ):
            raise RuntimeError("PMT manifest violates minimal-repair contract")
        action_name = (
            "all_routed_actions.csv.gz"
            if args.candidate_boundary_loss
            and args.candidate_boundary_version == "v2_molecule_max"
            else "corrective_actions.csv.gz"
        )
        actions = pd.read_csv(args.pmt_manifest_dir / action_name, low_memory=False)
        required_pmt = {
            "action_id", "route", "corrective_weight", "teacher_advantage",
            "target_path", "control_path", "cell_id",
        }
        if missing := required_pmt - set(actions.columns):
            raise RuntimeError(f"PMT corrective manifest lacks columns: {sorted(missing)}")
        if actions["action_id"].duplicated().any():
            raise RuntimeError("PMT action identifiers are not unique")
        if action_name == "all_routed_actions.csv.gz":
            valid_routes = {"corrective", "robustness_only", "harmful", "uncertain"}
            if (
                not set(actions["route"].astype(str)).issubset(valid_routes)
                or not actions["corrective_weight"].isin([0.0, 1.0]).all()
                or not actions.loc[
                    ~actions["route"].eq("corrective"), "corrective_weight"
                ].eq(0).all()
                or not actions.loc[
                    actions["route"].eq("corrective"), "corrective_weight"
                ].eq(1).all()
            ):
                raise RuntimeError("PMT routed membership or exact-zero weights are invalid")
        elif (
            not actions["route"].eq("corrective").all()
            or not actions["corrective_weight"].eq(1).all()
            or actions["teacher_advantage"].le(0).any()
        ):
            raise RuntimeError("PMT corrective membership or weights are invalid")
        if args.pmt_arm == "clean_duplicate":
            actions["target_path"] = ""
        elif args.pmt_arm == "matched_random":
            actions["target_path"] = actions["control_path"]
    elif args.action_selection == "materialized_routed":
        if args.materialized_action_dir is None:  # guarded above; narrows the type
            raise RuntimeError("materialized action directory was not resolved")
        report_path = args.materialized_action_dir / "report.json"
        action_path = args.materialized_action_dir / "training_actions.csv.gz"
        spectra_path = args.materialized_action_dir / "action_spectra.npz"
        materialized_action_report = json.loads(report_path.read_text(encoding="utf-8"))
        contracts = materialized_action_report.get("contracts", {})
        if (
            materialized_action_report.get("status")
            != "noise_corrected_routed_action_ledger_complete"
            or int(materialized_action_report.get("outer_formula_fold", -1))
            != args.outer_fold
            or contracts.get("three_semantics_are_separate") is not True
            or contracts.get("control_semantics_explicit_and_source_validated") is not True
            or contracts.get("all_route_formula_fold_seeds_match") is not True
            or contracts.get("all_route_clean_ranks_match") is not True
            or contracts.get("outer_held_formula_consumed") is not False
            or contracts.get("teacher_embedding_target_used") is not False
            or contracts.get("P3_consumed") is not False
        ):
            raise RuntimeError("materialized routed action ledger violates the E4-native contract")
        for key, path in (
            ("training_actions_sha256", action_path),
            ("action_spectra_sha256", spectra_path),
        ):
            if materialized_action_report.get("provenance", {}).get(key) != sha256_file(path):
                raise RuntimeError(f"materialized routed action provenance drifted: {key}")
        all_actions = pd.read_csv(action_path, low_memory=False)
        required_materialized = {
            "action_id", "query_index", "query_row", "query_ik14", "query_formula",
            "formula_fold", "source", "family", "recipe_id", "supervision_kind",
            "action_tensor_index", "control_semantic", "action_hard_negative_row",
            "action_positive_row", "clean_rank", "clean_margin", "action_rank",
            "action_margin",
        }
        if missing_columns := required_materialized - set(all_actions.columns):
            raise RuntimeError(
                "materialized routed actions lack columns: "
                f"{sorted(missing_columns)}"
            )
        with np.load(spectra_path, allow_pickle=False) as body:
            if set(body.files) != {"action_ids", "action_spectra", "control_spectra"}:
                raise RuntimeError("materialized action tensor schema failed")
            action_ids = np.asarray(body["action_ids"], dtype=str)
            true_action_spectra = np.asarray(body["action_spectra"], dtype=np.float32)
            control_spectra = np.asarray(body["control_spectra"], dtype=np.float32)
        tensor_index = all_actions["action_tensor_index"].to_numpy(np.int64)
        if (
            true_action_spectra.shape != control_spectra.shape
            or true_action_spectra.shape[1:] != (args.n_highest_peaks + 1, 2)
            or not np.array_equal(tensor_index, np.arange(len(all_actions), dtype=np.int64))
            or not np.array_equal(all_actions["action_id"].astype(str).to_numpy(), action_ids)
            or not np.isfinite(true_action_spectra).all()
            or not np.isfinite(control_spectra).all()
        ):
            raise RuntimeError("materialized action tensor/table alignment failed")
        # Action-space evaluation needs only one maximum-margin winner to count
        # a corrected query.  Direct encoder training is different: collapsing
        # 32k successful source/family/path views to one winner per query throws
        # away the invariances that must be learned.  Keep the historical arm
        # available, but let the repaired arm retain every robust strict action.
        actions, strict_top1_actions = select_materialized_direct_action_panel(
            all_actions,
            mode=(
                "multi_action_balanced"
                if args.materialized_injection_mode in {
                    "complete_panel_historical_e4", "e4_base_semantic_v2",
                    "e4_live_shared_v3",
                }
                else args.materialized_injection_mode
            ),
            margin_floor=MATERIALIZED_ACTION_REPLAY_MARGIN_FLOOR,
        )
        selected_rows_before_floor = int(
            strict_top1_actions["query_index"].nunique()
            if args.materialized_injection_mode == "one_best_e4"
            else len(strict_top1_actions)
        )
        numerical_boundary_actions_excluded = int(
            selected_rows_before_floor - len(actions)
        )
        if (
            actions.empty
            or (
                args.materialized_injection_mode == "one_best_e4"
                and actions["query_index"].duplicated().any()
            )
        ):
            raise RuntimeError("numerically robust materialized direct action panel failed")
        expected_best_action_sources = {
            "N_mature", "P_guided_original", "E10B", "E11", "E12B",
            "A4_exact", "V4_gradient_path",
        }
        qualifying_sources = set(strict_top1_actions["source"].astype(str))
        if not expected_best_action_sources.issubset(qualifying_sources):
            raise RuntimeError(
                "materialized best-action union is missing a qualified action source: "
                f"{sorted(expected_best_action_sources - qualifying_sources)}"
            )
        # Subset and reindex the tensor bank *after* the best-action union is
        # frozen.  The shuffled arm must draw donors only from the exact same
        # selected rows, otherwise it changes action quality as well as query
        # matching and ceases to be a one-variable causal control.
        original_tensor_index = actions["action_tensor_index"].to_numpy(np.int64)
        selected_true_action_spectra = true_action_spectra[original_tensor_index]
        selected_control_spectra = control_spectra[original_tensor_index]
        actions = actions.reset_index(drop=True)
        actions["action_tensor_index"] = np.arange(len(actions), dtype=np.int64)
        # The historical arm gives each selected augmentation unit weight.  In
        # multi-action mode, physical rows all remain present but each identity
        # receives the same total 4x4 E4 dose, split equally across its observed
        # source/family groups and then their actions.  No margin or outcome
        # magnitude enters this weight.
        # Final multi-action weights depend on coverage-schedule multiplicity
        # and are solved after the schedule is frozen below. Keep a positive
        # placeholder through tensor replay; one-best remains unit weighted.
        actions["corrective_weight"] = np.float32(1.0)
        actions["teacher_advantage"] = np.float32(0.0)
        if args.materialized_injection_mode == "e4_live_shared_v3":
            v3_targeted_action_spectra = selected_true_action_spectra
            (
                v3_shuffled_action_spectra,
                v3_shuffled_control_report,
            ) = source_family_shuffled_action_bank(
                actions,
                selected_true_action_spectra,
                selected_control_spectra,
                seed=args.seed + 1701,
            )
        else:
            v3_shuffled_control_report = {}
        if args.materialized_action_arm == "shuffled":
            if v3_shuffled_action_spectra is not None:
                materialized_action_spectra = v3_shuffled_action_spectra
                materialized_action_control_report = v3_shuffled_control_report
            else:
                materialized_action_spectra, materialized_action_control_report = (
                    source_family_shuffled_action_bank(
                        actions, selected_true_action_spectra, selected_control_spectra,
                        seed=args.seed + 1701,
                    )
                )
        else:
            materialized_action_spectra = selected_true_action_spectra
            materialized_action_control_report = {
                "strategy": (
                    "true_query_matched_action_spectrum"
                    if args.materialized_action_arm == "targeted"
                    else "clean_query_duplicate_action_view"
                ),
                "rows": int(len(actions)),
            }
        if actions[["action_positive_row", "action_hard_negative_row"]].isna().any().any():
            raise RuntimeError(
                "materialized E4-native actions lack an exact winning candidate boundary"
            )
        materialized_expected_margin_by_action = {
            str(row.action_id): float(row.action_margin)
            for row in actions[["action_id", "action_margin"]].itertuples(index=False)
        }
        materialized_expected_clean_margin_by_action = {
            str(row.action_id): float(row.clean_margin)
            for row in actions[["action_id", "clean_margin"]].itertuples(index=False)
        }
        if (
            len(materialized_expected_margin_by_action) != len(actions)
            or not np.isfinite(list(materialized_expected_margin_by_action.values())).all()
            or len(materialized_expected_clean_margin_by_action) != len(actions)
            or not np.isfinite(
                list(materialized_expected_clean_margin_by_action.values())
            ).all()
        ):
            raise RuntimeError("materialized E4 action margins are missing or non-finite")
        expected_initial_hash = materialized_action_report.get(
            "model_provenance", {}
        ).get("initial_student_checkpoint_sha256")
        if args.materialized_injection_mode == "complete_panel_historical_e4":
            if not isinstance(expected_initial_hash, str) or len(expected_initial_hash) != 64:
                raise RuntimeError(
                    "complete-panel actions lack their frozen E8 selection provenance"
                )
        elif (
            args.initial_student_checkpoint is None
            or expected_initial_hash != sha256_file(args.initial_student_checkpoint)
        ):
            raise RuntimeError(
                "materialized actions were not routed in the requested initial encoder geometry"
            )
        materialized_action_control_report.update({
            "qualifying_strict_top1_corrective_rows": int(len(strict_top1_actions)),
            "selected_best_action_union_rows_before_numerical_floor": int(
                selected_rows_before_floor
            ),
            "selected_best_action_union_rows": int(len(actions)),
            "numerical_boundary_actions_excluded": int(
                numerical_boundary_actions_excluded
            ),
            "strict_replay_action_margin_floor": float(
                MATERIALIZED_ACTION_REPLAY_MARGIN_FLOOR
            ),
            "selected_strict_top1_corrective_queries": int(
                actions["query_index"].nunique()
            ),
            "all_ledger_rows": int(len(all_actions)),
            "qualifying_sources": sorted(qualifying_sources),
            "sources": sorted(set(actions["source"].astype(str))),
            "families": int(actions[["source", "family"]].drop_duplicates().shape[0]),
            "selected_action_rows_preserved": True,
            "clean_and_action_active_candidate_row_union_preserved": True,
            "exact_actions_preserved": bool(
                args.materialized_action_arm == "targeted"
            ),
            "one_maximum_margin_action_per_query": bool(
                args.materialized_injection_mode == "one_best_e4"
            ),
            "multi_action_panel_all_strict": bool(
                args.materialized_injection_mode in {
                    "multi_action_balanced", "complete_panel_historical_e4",
                    "e4_base_semantic_v2", "e4_live_shared_v3",
                }
            ),
            "causal_control_donors_restricted_to_selected_union": True,
            "selected_actions_equal_unit_weight": bool(
                args.materialized_injection_mode in {
                    "one_best_e4", "complete_panel_historical_e4",
                }
            ),
            "selected_in_e8_geometry_but_trained_from_official": bool(
                args.materialized_injection_mode == "complete_panel_historical_e4"
            ),
            "identity_family_effective_dose_normalized": False,
            "effective_action_weight_sum": None,
            "effective_action_weight_per_identity": (
                None
                if args.materialized_injection_mode in {
                    "complete_panel_historical_e4", "e4_base_semantic_v2",
                    "e4_live_shared_v3",
                }
                else float(args.epochs * args.views_per_identity)
            ),
            "routing_scores_not_used_as_loss_targets": True,
            "strict_clean_wrong_action_top1_only": True,
            "teacher_embedding_target_used": False,
        })
        del (
            all_actions, strict_top1_actions, true_action_spectra,
            control_spectra, selected_control_spectra,
        )
        if args.materialized_injection_mode != "e4_live_shared_v3":
            del selected_true_action_spectra
    elif args.action_selection == "fixed":
        actions = load_frozen_r0_policy_actions(args.r0_dir, args.policy)
    else:
        report_path = args.outcome_action_dir / "report.json"
        action_path = args.outcome_action_dir / "corrective_teacher_actions.csv.gz"
        outcome_report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            outcome_report.get("status") != "noise_final_r1_privileged_teacher_complete"
            or not outcome_report.get("formal")
            or outcome_report.get("contracts", {}).get("P2b") != "forbidden"
            or int(outcome_report.get("locally_materialised_union_recoverable", -1)) != 882
        ):
            raise RuntimeError("outcome-mined action artifact is not formal and P2b-free")
        actions = pd.read_csv(action_path)
        if len(actions) != 882:
            raise RuntimeError(f"outcome-mined action count drifted: {len(actions)} != 882")
        required = {
            "query_index", "query_row", "query_ik14", "query_formula", "formula_fold",
            "baseline_rank", "teacher_rank", "teacher_margin", "selector", "attenuation",
            "step", "target_path", "teacher_hard_negative_row",
        }
        if required - set(actions.columns):
            raise RuntimeError(
                f"outcome-mined action table is missing columns: {sorted(required - set(actions.columns))}"
            )
        if actions["query_index"].duplicated().any():
            raise RuntimeError("outcome-mined action table must contain one selected action per query")
        if actions["baseline_rank"].astype(int).le(1).any() or actions["teacher_rank"].astype(int).ne(1).any():
            raise RuntimeError("outcome-mined action table contains a non-correction")
        for row in actions[["query_index", "teacher_hard_negative_row"]].itertuples(index=False):
            query = int(row.query_index)
            forced = int(row.teacher_hard_negative_row)
            _, candidate_rows, molecule_ptr, _ = graph.query_block(query)
            positive_rows = set(map(int, candidate_rows[: int(molecule_ptr[1])]))
            negative_rows = set(map(int, candidate_rows[int(molecule_ptr[1]):]))
            if forced in positive_rows or forced not in negative_rows:
                raise RuntimeError(
                    f"outcome-mined hard negative row {forced} is not a negative candidate for query {query}"
                )
        # Outcome fields are legal for train-fold action mining, but must never
        # enter a loss or sample weight.  Strip them before example creation.
        actions = actions.rename(columns={"teacher_hard_negative_row": "hard_negative_row"})
        actions = actions.drop(columns=[
            column for column in ("teacher_rank", "teacher_margin") if column in actions
        ])
    actions, causal_action_audit = materialize_causal_arm(actions, args.causal_arm)
    if args.pmt_arm != "none":
        causal_action_audit.update({
            "pmt_arm": args.pmt_arm,
            "pmt_alpha": float(args.pmt_alpha),
            "strict_corrective_membership": bool(
                not args.candidate_boundary_loss
                or args.candidate_boundary_version != "v2_molecule_max"
            ),
            "all_routed_actions_loaded": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "M2_predictions_used": False,
        })
    actions["baseline_rank"] = official_rank[actions["query_index"].to_numpy(np.int64)]
    if args.action_scope == "errors":
        actions = actions.loc[actions["baseline_rank"].astype(int).ne(1)].copy()
    train_actions = actions.loc[actions["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    e4_base_train_actions = pd.DataFrame()
    if (
        args.action_selection == "materialized_routed"
        and args.materialized_injection_mode in {
            "e4_base_semantic_v2", "e4_live_shared_v3",
        }
    ):
        if args.e4_base_action_dir is None or args.initial_student_checkpoint is None:
            raise RuntimeError("E4 hybrid base/E8 path was not resolved")
        e4_base_train_actions, e4_base_action_report = load_corrected_e4_base_actions(
            args.e4_base_action_dir,
            graph,
            outer_fold=args.outer_fold,
            formula_fold_seed=args.formula_fold_seed,
            initial_checkpoint=args.initial_student_checkpoint,
            graph_path=args.graph,
            embedding_cache=args.embedding_cache,
            data_path=args.data,
            official_checkpoint=args.official_checkpoint,
        )
        if e4_base_train_actions.empty:
            raise RuntimeError("signal-preserving V2 lost the corrected E4 base stream")
    # Outcome-mined held actions are deliberately not used even for subgroup
    # evaluation.  The primary evaluation is the complete clean held fold.
    held_action = (
        actions.loc[actions["formula_fold"].astype(int).eq(args.outer_fold)].copy()
        if args.action_selection == "fixed" else actions.iloc[0:0].copy()
    )
    held_formulas = {
        str(formula) for formula in graph.query_formula
        if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
    }
    if train_actions["query_formula"].astype(str).isin(held_formulas).any():
        raise RuntimeError("formula isolation failed in action manifest")

    positive_pairs = pd.DataFrame()
    train_positive = pd.DataFrame()
    held_positive = pd.DataFrame()
    positive_report: dict = {}
    if args.positive_stream_weight > 0:
        positive_report = json.loads(
            (args.positive_manifest_dir / "report.json").read_text(encoding="utf-8")
        )
        contracts = positive_report.get("contracts", {})
        if not positive_report.get("formal") or not positive_report.get("pass_to_pn_training"):
            raise RuntimeError("P-arm manifest is not a passing formal artifact")
        if contracts.get("teacher") != "forbidden" or contracts.get("P2b") != "forbidden":
            raise RuntimeError("P-arm manifest violates teacher/P2b boundary")
        positive_pairs = pd.read_csv(args.positive_manifest_dir / "positive_pairs.csv.gz")
        forbidden_positive = {
            "corrected", "introduced", "target_rank", "target_margin",
            "teacher_score", "teacher_rows", "p2b_score",
        }
        leaked_positive = forbidden_positive.intersection(positive_pairs.columns)
        if leaked_positive:
            raise RuntimeError(f"outcome/teacher columns leaked into P-arm: {sorted(leaked_positive)}")
        required_positive = {
            "query_index", "query_row", "positive_row", "query_ik14",
            "query_formula", "formula_fold", "relation",
        }
        if required_positive - set(positive_pairs.columns):
            raise RuntimeError("P-arm manifest is missing required pair columns")
        positive_query = positive_pairs["query_index"].to_numpy(np.int64)
        if np.any((positive_query < 0) | (positive_query >= graph.n_queries)):
            raise RuntimeError("P-arm query index is out of graph range")
        if not np.array_equal(
            positive_pairs["query_row"].to_numpy(np.int64), graph.query_row[positive_query]
        ):
            raise RuntimeError("P-arm query rows do not reproduce frozen candidate graph")
        if not np.array_equal(
            positive_pairs["query_ik14"].astype(str).to_numpy(), graph.query_ik14[positive_query]
        ):
            raise RuntimeError("P-arm identity does not reproduce frozen candidate graph")
        if not np.array_equal(
            positive_pairs["query_formula"].astype(str).to_numpy(), graph.query_formula[positive_query]
        ):
            raise RuntimeError("P-arm formula does not reproduce frozen candidate graph")
        for query, positive_row in zip(
            positive_query, positive_pairs["positive_row"].to_numpy(np.int64)
        ):
            pair_slice, candidate_rows, local_ptr, _ = graph.query_block(int(query))
            del pair_slice
            if int(positive_row) not in set(map(int, candidate_rows[: int(local_ptr[1])])):
                raise RuntimeError(f"P-arm row {positive_row} is not a positive for query {query}")
        observed = positive_pairs["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).to_numpy(np.int8)
        if not np.array_equal(observed, positive_pairs["formula_fold"].to_numpy(np.int8)):
            raise RuntimeError("P-arm formula folds do not reproduce locally")
        train_positive = positive_pairs.loc[
            positive_pairs["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        held_positive = positive_pairs.loc[
            positive_pairs["formula_fold"].astype(int).eq(args.outer_fold)
        ].copy()
        if train_positive["query_formula"].astype(str).isin(
            set(held_positive["query_formula"].astype(str))
        ).any():
            raise RuntimeError("formula isolation failed in P-arm manifest")

    guided_frame = pd.DataFrame()
    train_guided = pd.DataFrame()
    held_guided = pd.DataFrame()
    train_guided_risk = pd.DataFrame()
    guided_intensity_report: dict = {}
    guided_transfer_report: dict = {}
    guided_authorization_report: dict = {}
    guided_crossfit_reports: dict[str, dict] = {}
    if args.guided_noise_policy == "selected":
        if args.guided_crossfit_root is None:
            raise RuntimeError("selected guided noise requires --guided-crossfit-root")
        graph_hash = sha256_file(args.graph)
        fold_dir = args.guided_crossfit_root / f"fold_{args.outer_fold}"
        report_path = fold_dir / "report.json"
        manifest_path = fold_dir / "selected_actions.csv.gz"
        risk_path = fold_dir / "risk_controls.csv.gz"
        outcome_path = fold_dir / "action_outcomes.npz"
        amendment_path = fold_dir / "capacity_amendment.json"
        if not report_path.is_file() or not manifest_path.is_file() or not risk_path.is_file():
            raise FileNotFoundError(f"E14 teacher is incomplete: {fold_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        capacity_authorized = bool(report.get("pass_to_shared_encoder_transfer"))
        amendment: dict = {}
        if not capacity_authorized and amendment_path.is_file():
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            capacity_authorized = bool(
                outcome_path.is_file()
                and amendment.get("status")
                == "noise_final_e14_capacity_amendment_complete"
                and amendment.get("formal")
                and amendment.get("posthoc_amendment")
                and amendment.get("original_report_unchanged")
                and amendment.get("pass_to_shared_encoder_transfer")
                and amendment.get("provenance", {}).get("report_sha256")
                == sha256_file(report_path)
                and amendment.get("provenance", {}).get("selected_actions_sha256")
                == sha256_file(manifest_path)
                and amendment.get("provenance", {}).get("action_outcomes_sha256")
                == sha256_file(outcome_path)
                and amendment.get("provenance", {}).get("graph_sha256") == graph_hash
            )
        if (
            report.get("status") != "noise_final_e14_crossfit_p_teacher_complete"
            or not report.get("formal")
            or not capacity_authorized
            or int(report.get("outer_formula_fold", -1)) != args.outer_fold
            or report.get("provenance", {}).get("graph_sha256") != graph_hash
            or args.initial_student_checkpoint is None
            or report.get("provenance", {}).get("student_checkpoint_sha256")
            != sha256_file(args.initial_student_checkpoint)
            or not report.get("contracts", {}).get(
                "teacher_checkpoint_excludes_student_outer_formula_fold"
            )
            or not report.get("contracts", {}).get(
                "all_selected_queries_exclude_student_outer_formula_fold"
            )
            or not report.get("contracts", {}).get(
                "prior_fixed_cell_safety_filter_applied"
            )
            or not report.get("contracts", {}).get(
                "outer_train_multifold_action_safety_filter_applied"
            )
            or not report.get("contracts", {}).get(
                "action_specific_risk_controls_materialized"
            )
        ):
            raise RuntimeError(f"invalid E14 teacher for outer fold {args.outer_fold}")
        guided_crossfit_reports[str(args.outer_fold)] = report | {
            "capacity_authorization": (
                "posthoc_clustered_amendment" if amendment else "original_gate"
            ),
            "capacity_amendment": amendment,
        }
        guided_frame = pd.read_csv(manifest_path)
        required_selected = {
            "query_index", "query_row", "query_ik14", "query_formula",
            "formula_fold", "official_rank", "crossfit_clean_rank",
            "teacher_rank", "teacher_margin", "teacher_margin_delta",
            "positive_reference_rows",
            "teacher_positive_row", "teacher_hard_negative_row",
            "teacher_pair_clean_margin",
            "action_id", "guided_family", "guided_dose",
            "guided_auxiliary_dose", "guided_recurrence_prevalence",
            "guided_recurrence_max_peaks", "guided_support_weighted",
        }
        if required_selected - set(guided_frame.columns):
            raise RuntimeError(
                "selected E14 manifest is missing columns: "
                f"{sorted(required_selected - set(guided_frame.columns))}"
            )
        if guided_frame["query_index"].duplicated().any():
            raise RuntimeError("selected E14 teacher repeats a query")
        query = guided_frame["query_index"].to_numpy(np.int64)
        if np.any((query < 0) | (query >= graph.n_queries)):
            raise RuntimeError("selected E14 query is outside the graph")
        if not np.array_equal(
            guided_frame["query_row"].to_numpy(np.int64), graph.query_row[query]
        ):
            raise RuntimeError("selected E14 query rows drifted")
        if not np.array_equal(
            guided_frame["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query]
        ) or not np.array_equal(
            guided_frame["query_formula"].astype(str).to_numpy(), graph.query_formula[query]
        ):
            raise RuntimeError("selected E14 identity/formula drifted")
        observed_fold = guided_frame["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).to_numpy(np.int8)
        if not np.array_equal(
            observed_fold, guided_frame["formula_fold"].to_numpy(np.int8)
        ):
            raise RuntimeError("selected E14 formula fold does not reproduce")
        if np.any(observed_fold == args.outer_fold):
            raise RuntimeError("held formula actions leaked into selected E14 training")
        if not (
            (guided_frame["official_rank"].to_numpy(int) != 1).all()
            and (guided_frame["crossfit_clean_rank"].to_numpy(int) != 1).all()
            and (guided_frame["teacher_rank"].to_numpy(int) == 1).all()
        ):
            raise RuntimeError("selected E14 rows are not strict clean-wrong/action-correct cases")
        guided_frame["guided_policy"] = guided_frame["action_id"].astype(str)
        train_guided = guided_frame.copy()
        held_guided = guided_frame.iloc[0:0].copy()
        train_guided_risk = pd.read_csv(risk_path)
        required_risk = required_selected | {"control_kind"}
        if required_risk - set(train_guided_risk.columns):
            raise RuntimeError(
                "selected E14 risk controls are missing columns: "
                f"{sorted(required_risk - set(train_guided_risk.columns))}"
            )
        if train_guided_risk[["query_index", "action_id"]].duplicated().any():
            raise RuntimeError("selected E14 risk controls repeat a query/action pair")
        risk_query = train_guided_risk["query_index"].to_numpy(np.int64)
        if np.any((risk_query < 0) | (risk_query >= graph.n_queries)):
            raise RuntimeError("selected E14 risk-control query is outside the graph")
        risk_observed_fold = train_guided_risk["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).to_numpy(np.int8)
        if not np.array_equal(
            risk_observed_fold,
            train_guided_risk["formula_fold"].to_numpy(np.int8),
        ) or np.any(risk_observed_fold == args.outer_fold):
            raise RuntimeError("selected E14 risk controls violate formula isolation")
        if not (
            train_guided_risk["crossfit_clean_rank"].to_numpy(int) == 1
        ).all() or not train_guided_risk["control_kind"].astype(str).isin(
            {"introduced", "protected_boundary"}
        ).all():
            raise RuntimeError("selected E14 risk controls are not mature-clean-correct")
        train_guided_risk["guided_policy"] = train_guided_risk["action_id"].astype(str)
        held_formulas = {
            str(formula) for formula in graph.query_formula
            if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
        }
        if train_guided["query_formula"].astype(str).isin(held_formulas).any():
            raise RuntimeError("selected E14 teacher violates held-formula isolation")
        for row in train_guided[[
            "query_index", "positive_reference_rows", "teacher_positive_row",
            "teacher_hard_negative_row",
        ]].itertuples(index=False):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
            positive_set = set(map(int, candidate_rows[: int(local_ptr[1])]))
            negative_set = set(map(int, candidate_rows[int(local_ptr[1]):]))
            if not set(parse_reference_rows(row.positive_reference_rows)) <= positive_set:
                raise RuntimeError(
                    f"selected E14 reference is not positive for query {row.query_index}"
                )
            if int(row.teacher_positive_row) not in positive_set:
                raise RuntimeError("selected E14 teacher-positive row is not positive")
            if int(row.teacher_hard_negative_row) not in negative_set:
                raise RuntimeError("selected E14 teacher-negative row is not negative")
        for row in train_guided_risk[[
            "query_index", "positive_reference_rows", "teacher_positive_row",
            "teacher_hard_negative_row",
        ]].itertuples(index=False):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
            positive_set = set(map(int, candidate_rows[: int(local_ptr[1])]))
            negative_set = set(map(int, candidate_rows[int(local_ptr[1]):]))
            if not set(parse_reference_rows(row.positive_reference_rows)) <= positive_set:
                raise RuntimeError("selected E14 risk reference is not positive")
            if int(row.teacher_positive_row) not in positive_set:
                raise RuntimeError("selected E14 risk positive row is not positive")
            if int(row.teacher_hard_negative_row) not in negative_set:
                raise RuntimeError("selected E14 risk negative row is not negative")
    elif args.guided_noise_policy != "none":
        guided_intensity_report = json.loads(
            (args.guided_intensity_dir / "report.json").read_text(encoding="utf-8")
        )
        guided_transfer_report = json.loads(
            (args.guided_transfer_dir / "report.json").read_text(encoding="utf-8")
        )
        if guided_intensity_report.get("status") != "noise_final_positive_guided_matrix_complete":
            raise RuntimeError("guided intensity matrix is not a formal completed artifact")
        if guided_transfer_report.get("status") != "noise_final_positive_peak_transfer_complete":
            raise RuntimeError("guided transfer matrix is not a formal completed artifact")
        if not guided_intensity_report.get("formal") or not guided_transfer_report.get("formal"):
            raise RuntimeError("guided noise artifacts must be formal")
        graph_hash = sha256_file(args.graph)
        if guided_intensity_report.get("provenance", {}).get("graph_sha256") != graph_hash:
            raise RuntimeError("guided intensity matrix graph mismatch")
        if guided_transfer_report.get("provenance", {}).get("graph_sha256") != graph_hash:
            raise RuntimeError("guided transfer matrix graph mismatch")
        required_cells = set()
        if args.guided_noise_policy in {"intensity", "both"}:
            required_cells.add("consensus_projection|dose=0.75")
        if args.guided_noise_policy in {"transfer", "both"}:
            required_cells.add("recurrent_union_mix|dose=0.50")
        available_cells = set(guided_intensity_report.get("passing_cells", [])) | set(
            guided_transfer_report.get("passing_cells", [])
        )
        if not required_cells <= available_cells:
            raise RuntimeError(f"guided policy uses cells that did not pass: {sorted(required_cells - available_cells)}")

        if args.guided_action_authorization_dir is not None:
            guided_authorization_report = json.loads(
                (args.guided_action_authorization_dir / "report.json").read_text(encoding="utf-8")
            )
            if (
                guided_authorization_report.get("status")
                != "noise_final_e12b_relaxed_recurrence_complete"
                or not guided_authorization_report.get("formal")
            ):
                raise RuntimeError("guided action authorization is not the formal E12-B artifact")
            if guided_authorization_report.get("provenance", {}).get("graph_sha256") != graph_hash:
                raise RuntimeError("guided action authorization graph mismatch")
            if (
                args.guided_reference_checkpoint is not None
                and guided_authorization_report.get("provenance", {}).get("student_checkpoint_sha256")
                != sha256_file(args.guided_reference_checkpoint)
            ):
                raise RuntimeError("guided reference checkpoint differs from E12-B provenance")
            expected_cell = (
                "top3|standard|"
                f"max={args.guided_recurrence_max_peaks}|dose=0.50"
            )
            best_cell = guided_authorization_report.get("best_fixed_cell", {})
            if (
                args.guided_noise_policy in {"transfer", "both"}
                and (
                    not math.isclose(args.guided_recurrence_prevalence, 0.50, abs_tol=1e-12)
                    or best_cell.get("cell_id") != expected_cell
                    or expected_cell not in set(guided_authorization_report.get("passing_fixed_cells", []))
                )
            ):
                raise RuntimeError(
                    "E12-B does not authorize the requested relaxed recurrence recipe"
                )

        base = pd.read_csv(
            args.guided_intensity_dir / "action_manifest.csv.gz",
            usecols=[
                "query_index", "query_row", "query_ik14", "query_formula",
                "positive_reference_rows",
            ],
        )
        transfer = pd.read_csv(
            args.guided_transfer_dir / "action_manifest.csv.gz",
            usecols=["query_index", "positive_missing_peak_count"],
        )
        signature = pd.read_csv(
            args.error_signatures,
            usecols=["query_index", "positive_deficit"],
        )
        if any(frame["query_index"].duplicated().any() for frame in (base, transfer, signature)):
            raise RuntimeError("guided noise inputs must be one row per query")
        guided_frame = base.merge(transfer, on="query_index", validate="one_to_one").merge(
            signature, on="query_index", validate="one_to_one",
        ).sort_values("query_index", kind="stable").reset_index(drop=True)
        if len(guided_frame) != graph.n_queries or not np.array_equal(
            guided_frame["query_index"].to_numpy(np.int64), np.arange(graph.n_queries)
        ):
            raise RuntimeError("guided action ledgers do not cover the candidate graph one-to-one")
        query = guided_frame["query_index"].to_numpy(np.int64)
        if not np.array_equal(guided_frame["query_row"].to_numpy(np.int64), graph.query_row[query]):
            raise RuntimeError("guided query rows do not reproduce graph")
        if not np.array_equal(guided_frame["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query]):
            raise RuntimeError("guided identities do not reproduce graph")
        if not np.array_equal(guided_frame["query_formula"].astype(str).to_numpy(), graph.query_formula[query]):
            raise RuntimeError("guided formulas do not reproduce graph")
        positive_deficit = strict_bool(guided_frame["positive_deficit"], "positive_deficit")
        if args.guided_query_scope == "all":
            eligible = np.ones(len(guided_frame), dtype=bool)
        else:
            eligible = positive_deficit & (official_rank[query] != 1)
        guided_frame = guided_frame.loc[eligible].copy()
        configured: list[pd.DataFrame] = []
        if args.guided_noise_policy in {"intensity", "both"}:
            block = guided_frame.copy()
            block["guided_policy"] = "positive_intensity_consensus"
            block["guided_family"] = "consensus_projection"
            block["guided_dose"] = 0.75
            configured.append(block)
        if args.guided_noise_policy in {"transfer", "both"}:
            # The historical count was computed with prevalence 0.67/max 5.
            # It must not filter the relaxed E12-B recipe (0.50/max 10), whose
            # expanded eligibility is precisely the intervention being tested.
            if guided_authorization_report:
                block = guided_frame.copy()
            else:
                block = guided_frame.loc[
                    guided_frame["positive_missing_peak_count"].astype(int) > 0
                ].copy()
            block["guided_policy"] = "positive_recurrent_peak_transfer"
            block["guided_family"] = "recurrent_union_mix"
            block["guided_dose"] = 0.50
            configured.append(block)
        guided_frame = pd.concat(configured, ignore_index=True)
        guided_frame["formula_fold"] = guided_frame["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, args.formula_fold_seed)
        ).astype(np.int8)
        # Reference rows are checked against the positive molecule.  This is
        # the only place identity labels enter action construction.
        for row in guided_frame[["query_index", "positive_reference_rows"]].drop_duplicates().itertuples(index=False):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
            positive_set = set(map(int, candidate_rows[: int(local_ptr[1])]))
            if not set(parse_reference_rows(row.positive_reference_rows)) <= positive_set:
                raise RuntimeError(f"guided reference is not positive for query {row.query_index}")
        train_guided = guided_frame.loc[
            guided_frame["formula_fold"].astype(int).ne(args.outer_fold)
        ].copy()
        held_guided = guided_frame.loc[
            guided_frame["formula_fold"].astype(int).eq(args.outer_fold)
        ].copy()
        if train_guided["query_formula"].astype(str).isin(
            set(held_guided["query_formula"].astype(str))
        ).any():
            raise RuntimeError("formula isolation failed in guided noise stream")

    # Full clean ledger is reconstructed from frozen graph labels/scores.  It
    # is not a teacher: it only provides ground-truth ranking and safety replay.
    formula_fold_by_query = {
        query: stable_fold(str(formula), 5, args.formula_fold_seed)
        for query, formula in enumerate(graph.query_formula)
    }
    # The independently recomputed split must reproduce every R0 action row.
    observed_fold = actions["query_index"].astype(int).map(formula_fold_by_query).to_numpy(np.int8)
    if not np.array_equal(observed_fold, actions["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("formula-fold reconstruction does not reproduce frozen R0")
    if not e4_base_train_actions.empty:
        historical_fold = e4_base_train_actions["query_index"].astype(int).map(
            formula_fold_by_query
        ).to_numpy(np.int8)
        if not np.array_equal(
            historical_fold,
            e4_base_train_actions["formula_fold"].to_numpy(np.int8),
        ):
            raise RuntimeError(
                "formula-fold reconstruction does not reproduce the corrected V2 E4 base"
            )
    held_queries = np.asarray(
        [query for query, fold in formula_fold_by_query.items() if fold == args.outer_fold],
        dtype=np.int64,
    )
    safety_queries = np.asarray([
        query for query, fold in formula_fold_by_query.items()
        if fold != args.outer_fold and official_rank[query] == 1
    ], dtype=np.int64)
    safety_frame = pd.DataFrame({
        "query_index": safety_queries,
        "query_row": graph.query_row[safety_queries],
        "query_ik14": graph.query_ik14[safety_queries],
        "query_formula": graph.query_formula[safety_queries],
    })

    action_examples = make_examples(
        graph, train_actions, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules, True,
    )
    e4_base_action_examples: list[DirectExample] = []
    if args.candidate_boundary_loss and args.candidate_boundary_version == "v2_molecule_max":
        action_examples = align_candidate_references_by_query(
            graph, action_examples, args.positive_spectra, args.negative_molecules,
        )
    safety_examples = make_examples(
        graph, safety_frame, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules, False,
    )
    positive_examples = make_positive_examples(
        graph, train_positive, official_rank, official_margin,
        args.negative_molecules,
    ) if args.positive_stream_weight > 0 else []
    guided_examples: list[GuidedNoiseExample] = []
    error_action_examples = [example for example in action_examples if example.official_rank != 1]
    if len(action_examples) < 100 or len(set(x.identity for x in action_examples)) < 100:
        raise RuntimeError("direct action training pool is unexpectedly small")
    if args.positive_stream_weight > 0 and (
        len(positive_examples) < 500
        or len(set(x.identity for x in positive_examples)) < 250
    ):
        raise RuntimeError("strict cross-condition P-arm training pool is unexpectedly small")
    reachable_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    store = SpectrumStore(args.data, reachable_rows, args.n_highest_peaks)
    _, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    if set(map(int, reachable_rows)) - set(cache_index):
        raise RuntimeError("official embedding cache does not cover candidate graph")
    # Materialize only the reachable cache once.  The former dict-of-views plus
    # np.stack retained both the full cache and a second 87,848 x 1024 array.
    official_encoded = np.stack([
        cache_embeddings[cache_index[int(row)]] for row in reachable_rows
    ]).astype(np.float32, copy=False)
    del cache_embeddings, cache_index
    official_by_row = {
        int(row): official_encoded[index]
        for index, row in enumerate(reachable_rows)
    }

    # E12-B selected top3 real same-identity references in the mature E8
    # geometry, not in the original official geometry. Recompute those rows
    # from the frozen, provenance-checked E8 checkpoint before any optimizer
    # exists. They construct training noise only and are never used at inference.
    if (
        args.guided_noise_policy in {"transfer", "both"}
        and args.guided_action_authorization_dir is not None
    ):
        reference_model, reference_initialization = load_base_model(
            args.official_checkpoint, args.architecture_checkpoint,
            device, args.n_highest_peaks,
        )
        if reference_initialization not in {"official_embedding", "official_embedding_slim"}:
            raise RuntimeError("guided reference encoder has unexpected initialization")
        reference_package = torch_load_compat(args.guided_reference_checkpoint, map_location="cpu")
        if (
            reference_package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or not reference_package.get("inference_clean_only")
            or reference_package.get("P2b_used")
        ):
            raise RuntimeError("guided reference checkpoint violates shared-embedding contract")
        reference_model.load_state_dict(reference_package["model_state"], strict=True)
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference_model.eval()
        reference_encoded = encode_rows(
            reference_model, store, reachable_rows, device,
            args.eval_batch_size, False, "E13-reference-fp32",
        )
        reference_index = {int(row): index for index, row in enumerate(reachable_rows)}
        selected_rows: dict[int, str] = {}
        for query in np.unique(train_guided["query_index"].to_numpy(np.int64)):
            _, candidate_rows, local_ptr, _ = graph.query_block(int(query))
            positive_rows = np.asarray(candidate_rows[: int(local_ptr[1])], dtype=np.int64)
            query_vector = reference_encoded[reference_index[int(graph.query_row[int(query)])]]
            positive_vectors = reference_encoded[
                [reference_index[int(row)] for row in positive_rows]
            ]
            order = np.argsort(-(positive_vectors @ query_vector), kind="stable")[:3]
            chosen = tuple(map(int, positive_rows[order]))
            if not chosen:
                raise RuntimeError(f"guided query {query} lacks a positive reference")
            selected_rows[int(query)] = ";".join(map(str, chosen))
        train_guided = train_guided.copy()
        train_guided["positive_reference_rows"] = train_guided["query_index"].astype(int).map(
            selected_rows
        )
        if train_guided["positive_reference_rows"].isna().any():
            raise RuntimeError("failed to assign mature-E8 top3 references")
        del reference_model, reference_encoded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    guided_examples = make_guided_noise_examples(
        graph, train_guided, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    ) if args.guided_noise_policy != "none" else []
    guided_risk_examples = make_guided_noise_examples(
        graph, train_guided_risk, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules,
    ) if args.guided_noise_policy == "selected" else []
    if args.guided_noise_policy != "none" and (
        len(guided_examples) < 500
        or len(set(example.identity for example in guided_examples)) < 250
    ):
        raise RuntimeError("guided positive-noise training pool is unexpectedly small")
    if args.guided_risk_control_ratio > 0 and (
        len(guided_risk_examples) < 100
        or len(set(example.identity for example in guided_risk_examples)) < 50
    ):
        raise RuntimeError("guided action-specific risk-control pool is unexpectedly small")
    guided_teacher_replay: dict[str, float] = {}
    branch_gradient_audit: dict = {}
    effective_guided_noise_weight = float(args.guided_noise_weight)
    effective_boundary_action_weight = 1.0

    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    initial_package: dict = {}
    initial_decision: dict = {}
    if args.initial_student_checkpoint is not None:
        initial_package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
        if initial_decision_path is None:
            raise RuntimeError("initial decision path was not resolved")
        initial_decision = json.loads(initial_decision_path.read_text(encoding="utf-8"))
        initial_configuration = initial_decision.get("configuration", {})
        if (
            initial_package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or int(initial_package.get("outer_fold", -1)) != args.outer_fold
            or not initial_package.get("inference_clean_only")
            or initial_package.get("P2b_used")
            or initial_decision.get("status")
            != "noise_final_e4a_direct_augmentation_complete"
            or not initial_decision.get("formal")
            or int(initial_configuration.get("outer_fold", -1)) != args.outer_fold
            or int(initial_configuration.get("formula_fold_seed", -1))
            != args.formula_fold_seed
            or int(initial_configuration.get("seed", -1))
            != int(initial_package.get("seed", -2))
        ):
            raise RuntimeError("initial student checkpoint violates the E14 outer-fold contract")
        model.load_state_dict(initial_package["model_state"], strict=True)
        # load_state_dict copied tensors into the CUDA model; retaining the CPU
        # checkpoint duplicates roughly 117M parameters for the whole run.
        del initial_package
        initialization = "mature_e4a_continuation"
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    # Gradients stay on, stochastic dropout stays off.
    model.eval()
    materialized_clean_replay: dict[str, object] = {}
    materialized_action_replay: dict[str, object] = {}
    official_action_targets = (
        encode_official_action_targets(
            model, store, action_examples, device, args.eval_batch_size,
        )
        if args.direct_transfer_mode == "official_action" else None
    )
    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer_groups = [
        {"params": head_parameters, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": 0.0},
    ]
    if args.optimizer_boundary_mode in {
        "action_injector_v1", "signal_preserving_v2", "separated_e4_action_v3",
    }:
        optimizer_groups[0]["group_name"] = "head"
        optimizer_groups[1]["group_name"] = "backbone"
    optimizer = torch.optim.AdamW(optimizer_groups)
    trainable_parameters = head_parameters + backbone_parameters
    if (
        len({id(parameter) for parameter in trainable_parameters})
        != len(trainable_parameters)
        or {id(parameter) for parameter in trainable_parameters}
        != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    ):
        raise RuntimeError("E4 optimizer parameter partition is incomplete")
    e4_action_injector = (
        E4ActionInjectorV1Bridge(
            optimizer,
            trainable_parameters,
            target_attributable_fraction=(
                args.injector_target_attributable_fraction
            ),
            minimum_protective_component_retention=(
                args.injector_minimum_protective_retention
            ),
            maximum_update_norm_ratio_to_original=(
                args.injector_maximum_update_norm_ratio
            ),
        )
        if args.optimizer_boundary_mode == "action_injector_v1" else None
    )
    e4_action_injector_steps: list[E4ActionInjectorV1Step] = []
    signal_preserving_injector = (
        E4SignalPreservingInjectorBridgeV2(
            optimizer,
            trainable_parameters,
            target_attributable_fraction=(
                args.injector_target_attributable_fraction
            ),
            minimum_historical_e4_retention=(
                args.injector_minimum_protective_retention
            ),
            maximum_historical_e4_update_norm_ratio=(
                args.injector_maximum_update_norm_ratio
            ),
        )
        if args.optimizer_boundary_mode == "signal_preserving_v2" else None
    )
    signal_preserving_steps: list[SignalPreservingInjectionV2Step] = []
    live_shared_v3_injector = (
        SeparatedE4ActionInjectorV3(
            optimizer,
            trainable_parameters,
            target_action_fraction=args.injector_target_attributable_fraction,
            minimum_e4_projection_retention=(
                args.injector_minimum_protective_retention
            ),
            maximum_update_norm_ratio=args.injector_maximum_update_norm_ratio,
        )
        if args.optimizer_boundary_mode == "separated_e4_action_v3" else None
    )
    live_shared_v3_steps: list[SeparatedE4ActionStepV3] = []
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    initial_encoded = encode_rows(
        model, store, reachable_rows, device, args.eval_batch_size, False, "E4A-init-fp32",
    )
    training_anchor_by_row = {
        int(row): initial_encoded[index] for index, row in enumerate(reachable_rows)
    }
    # Historical E4 used the frozen official cache for its margin floor and
    # preservation targets.  For the approved fixed-R0 Injector route this is
    # deliberately not replaced by a newly encoded or warm-start anchor.  The
    # shared rank references remain live because rank-reference-mode=shared.
    objective_reference_by_row = (
        official_by_row
        if args.optimizer_boundary_mode in {
            "action_injector_v1", "signal_preserving_v2",
            "separated_e4_action_v3",
        }
        and (
            args.action_selection == "fixed"
            or (
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode
                == "complete_panel_historical_e4"
            )
        )
        else training_anchor_by_row
    )
    if (
        (signal_preserving_v2 or live_shared_v3)
        and objective_reference_by_row is not training_anchor_by_row
    ):
        raise RuntimeError(
            "E4 continuation hybrid must use the frozen E8 initialization as "
            "its floor and preservation anchor"
        )
    initial_cosine = np.einsum("ij,ij->i", initial_encoded, official_encoded)
    baseline_rank, baseline_summary = streaming_evaluate_embeddings(
        graph, reachable_rows, official_encoded, held_queries,
    )
    initial_rank, initial_summary = streaming_evaluate_embeddings(
        graph, reachable_rows, initial_encoded, held_queries,
    )
    zero_change_report: dict[str, object]
    if args.initial_student_checkpoint is None:
        zero_change_report = audit_official_reencoding_zero_change(
            graph, reachable_rows, official_encoded, initial_encoded,
            held_queries,
        )
        initial_mismatches = int(zero_change_report["rank_mismatches"])
        direct_rank_mismatches = int(np.sum(initial_rank != baseline_rank))
        if initial_mismatches != direct_rank_mismatches:
            raise RuntimeError(
                "zero-change audit disagrees with the primary bounded scorer: "
                f"audit={initial_mismatches} primary={direct_rank_mismatches}"
            )
        if zero_change_report.get("gate_passed") is not True:
            raise RuntimeError(
                "zero-change gate failed: "
                f"{json.dumps(zero_change_report, sort_keys=True)}"
            )
    else:
        if initial_ledger_path is None:
            raise RuntimeError("initial held-ledger path was not resolved")
        if args.action_selection == "materialized_routed":
            # E8 was trained on the earlier graph, whereas the routed action
            # ledger is built and evaluated on the corrected 83k-query graph.
            # Replaying E8's old held table here would compare different query
            # universes. The ledger instead binds the exact E8 checkpoint hash
            # before any action is loaded, and this forward pass establishes its
            # clean baseline on the current graph.
            initial_mismatches = None
            initial_replay_verification = (
                "checkpoint+materialized_action_ledger_hash+current_graph_forward"
            )
        elif initial_ledger_path.is_file():
            initial_ledger = pd.read_csv(initial_ledger_path).sort_values(
                "query_index", kind="stable"
            )
            if not np.array_equal(
                initial_ledger["query_index"].to_numpy(np.int64), held_queries
            ):
                raise RuntimeError("initial checkpoint held ledger does not match outer fold")
            initial_mismatches: int | None = int(np.sum(
                initial_rank != initial_ledger["final_rank"].to_numpy(np.int16)
            ))
            if initial_mismatches:
                raise RuntimeError(
                    f"mature initialization replay changed {initial_mismatches} held ranks"
                )
            initial_replay_verification = "checkpoint+decision+held_ledger+graph"
        else:
            held_summary = initial_decision.get("held_clean", {})
            reproduced = {
                "n_queries": int(len(initial_rank)),
                "errors": int(np.sum(initial_rank != 1)),
                "corrected": int(np.sum((baseline_rank != 1) & (initial_rank == 1))),
                "introduced": int(np.sum((baseline_rank == 1) & (initial_rank != 1))),
            }
            expected = {key: int(held_summary.get(key, -1)) for key in reproduced}
            if reproduced["n_queries"] != expected["n_queries"]:
                raise RuntimeError(
                    "mature initialization query count disagrees with decision: "
                    f"reproduced={reproduced} expected={expected}"
                )
            # A freshly loaded CUDA model can change a handful of strict ranks
            # at near-exact score ties.  The missing per-query ledger cannot be
            # reconstructed after the fact, so compare the immutable decision
            # aggregates under the same 0.1% numerical-replay tolerance used by
            # the official baseline gate.  This is not a performance tolerance:
            # larger drift still fails closed before optimization.
            replay_count_tolerance = max(1, int(math.ceil(0.001 * len(initial_rank))))
            aggregate_delta = {
                key: int(reproduced[key] - expected[key])
                for key in ("errors", "corrected", "introduced")
            }
            if any(abs(value) > replay_count_tolerance for value in aggregate_delta.values()):
                raise RuntimeError(
                    "mature initialization aggregate replay disagrees with decision: "
                    f"reproduced={reproduced} expected={expected} "
                    f"tolerance={replay_count_tolerance}"
                )
            expected_recall = float(held_summary.get("recall1", float("nan")))
            if not np.isfinite(expected_recall) or not np.isclose(
                float(np.mean(initial_rank == 1)), expected_recall,
                rtol=0.0,
                atol=replay_count_tolerance / max(len(initial_rank), 1) + 1e-12,
            ):
                raise RuntimeError(
                    "mature initialization recall replay disagrees with decision"
                )
            initial_mismatches = None
            initial_replay_verification = "checkpoint+decision+graph_aggregate"
        zero_change_report = {
            "verification": initial_replay_verification,
            "preservation_mean": float(np.mean(initial_cosine)),
            "preservation_minimum": float(np.min(initial_cosine)),
            "rank_mismatches": initial_mismatches,
            "gate_passed": True,
        }
    if args.action_selection == "materialized_routed":
        complete_panel_historical_e4 = (
            args.materialized_injection_mode == "complete_panel_historical_e4"
        )
        signal_preserving_hybrid = (
            args.materialized_injection_mode in {
                "e4_base_semantic_v2", "e4_live_shared_v3",
            }
        )
        official_start_materialized = complete_panel_historical_e4
        all_queries = np.arange(graph.n_queries, dtype=np.int64)
        initial_all_rank, _, initial_all_margin = full_graph_query_details(
            graph, reachable_rows, initial_encoded, all_queries,
        )
        if not np.array_equal(initial_all_rank[held_queries], initial_rank):
            raise RuntimeError(
                "current-graph E8 rank reconstruction differs from held initialization"
            )
        train_actions = train_actions.copy()
        train_query = train_actions["query_index"].to_numpy(np.int64)
        stored_clean_rank = train_actions["clean_rank"].to_numpy(np.int64)
        stored_clean_margin = train_actions["clean_margin"].to_numpy(np.float32)
        replay_rank = initial_all_rank[train_query]
        replay_margin = initial_all_margin[train_query]
        rank_mismatch = replay_rank != stored_clean_rank
        clean_margin_error = np.abs(replay_margin - stored_clean_margin)
        # A strict rank can flip only at an effectively exact zero margin.  We
        # record such numerical tie flips separately; any substantive mismatch
        # remains a hard provenance failure before the optimizer exists.
        near_zero_tie = (
            np.abs(replay_margin) <= 5e-5
        ) & (np.abs(stored_clean_margin) <= 5e-5)
        unexplained_rank_mismatch = rank_mismatch & ~near_zero_tie
        materialized_clean_replay = {
            "exact_router_scoring_used": True,
            "selection_geometry": "frozen_E8",
            "training_initialization_geometry": (
                "official_DreaMS" if official_start_materialized else "frozen_E8"
            ),
            "queries": int(len(train_query)),
            "rank_mismatches": int(np.sum(rank_mismatch)),
            "rank_mismatches_explained_by_near_zero_tie": int(
                np.sum(rank_mismatch & near_zero_tie)
            ),
            "unexplained_rank_mismatches": int(np.sum(unexplained_rank_mismatch)),
            "margin_max_abs_error": float(np.max(clean_margin_error)),
            "margin_p99_abs_error": float(np.quantile(clean_margin_error, 0.99)),
        }
        if not official_start_materialized and np.any(unexplained_rank_mismatch):
            bad = np.flatnonzero(unexplained_rank_mismatch)[:20]
            raise RuntimeError(
                "materialized action clean ranks disagree outside numerical ties: "
                f"report={materialized_clean_replay}; "
                f"examples={[{'query_index': int(train_query[i]), 'replayed_rank': int(replay_rank[i]), 'stored_rank': int(stored_clean_rank[i]), 'replayed_margin': float(replay_margin[i]), 'stored_margin': float(stored_clean_margin[i])} for i in bad]}"
            )
        if (
            not official_start_materialized
            and float(np.max(clean_margin_error)) > 5e-4
        ):
            raise RuntimeError(
                "materialized action clean margins do not replay in current E8 geometry: "
                f"{materialized_clean_replay}"
            )
        # Preserve the frozen route's strict error membership at exact ties,
        # while all actual margins and candidate references come from this
        # checkpoint's exact current-geometry replay.
        action_reference_rank = initial_all_rank.copy()
        action_reference_margin = initial_all_margin.copy()
        if not official_start_materialized:
            action_reference_rank[train_query] = stored_clean_rank
            action_reference_margin[train_query] = replay_margin
            train_actions["baseline_rank"] = stored_clean_rank
        else:
            # The action *selection* is frozen from E8, but E4's scientific
            # logic starts from official DreaMS and derives every floor,
            # safety query and candidate reference in that actual geometry.
            train_actions["baseline_rank"] = replay_rank
        action_examples = make_examples(
            graph, train_actions, action_reference_rank, action_reference_margin,
            args.positive_spectra, args.negative_molecules, True,
        )
        safety_queries = np.asarray([
            query for query, fold in formula_fold_by_query.items()
            if fold != args.outer_fold and initial_all_rank[query] == 1
        ], dtype=np.int64)
        safety_frame = pd.DataFrame({
            "query_index": safety_queries,
            "query_row": graph.query_row[safety_queries],
            "query_ik14": graph.query_ik14[safety_queries],
            "query_formula": graph.query_formula[safety_queries],
        })
        safety_examples = make_examples(
            graph, safety_frame, initial_all_rank, initial_all_margin,
            args.positive_spectra, args.negative_molecules, False,
        )
        action_examples = refresh_candidate_references(
            graph, action_examples, reachable_rows, initial_encoded,
            args.positive_spectra, args.negative_molecules,
            preserve_forced_negative=True,
        )
        safety_examples = refresh_candidate_references(
            graph, safety_examples, reachable_rows, initial_encoded,
            args.positive_spectra, args.negative_molecules,
            preserve_forced_negative=True,
        )
        if signal_preserving_hybrid:
            e4_base_train_actions = e4_base_train_actions.copy()
            base_query = e4_base_train_actions["query_index"].to_numpy(np.int64)
            e4_base_train_actions["baseline_rank"] = initial_all_rank[base_query]
            e4_base_action_examples = make_examples(
                graph,
                e4_base_train_actions,
                initial_all_rank,
                initial_all_margin,
                args.positive_spectra,
                args.negative_molecules,
                True,
            )
            e4_base_action_examples = refresh_candidate_references(
                graph,
                e4_base_action_examples,
                reachable_rows,
                initial_encoded,
                args.positive_spectra,
                args.negative_molecules,
                preserve_forced_negative=True,
            )
            expected_base_identities = int(
                e4_base_action_report.get("action_identities", -1)
            )
            if (
                len(e4_base_action_examples)
                != int(e4_base_action_report.get("action_rows", -1))
                or len({item.identity for item in e4_base_action_examples})
                != expected_base_identities
            ):
                raise RuntimeError(
                    "corrected E4 base examples lost rows or identities before training"
                )
        error_action_examples = [
            example for example in action_examples if example.official_rank != 1
        ]
        if args.materialized_injection_mode == "one_best_e4":
            if any(
                not math.isclose(example.sample_weight, 1.0)
                for example in action_examples
            ):
                raise RuntimeError(
                    "materialized one-best union contains a non-unit loss weight"
                )
        else:
            if any(
                not math.isfinite(example.sample_weight)
                or example.sample_weight <= 0
                for example in action_examples
            ):
                raise RuntimeError(
                    "materialized multi-action panel contains a non-positive loss weight"
                )
        if (
            not complete_panel_historical_e4
            and len(error_action_examples) != len(action_examples)
        ):
            raise RuntimeError(
                "strict materialized action set contains a current-E8-correct query"
            )
        if materialized_action_spectra is None:
            raise RuntimeError("materialized action spectrum bank disappeared before replay")
        materialized_action_replay = audit_materialized_action_replay(
            model,
            store,
            action_examples,
            materialized_action_spectra,
            materialized_expected_margin_by_action,
            materialized_expected_clean_margin_by_action,
            training_anchor_by_row,
            device,
            args.eval_batch_size,
            args.amp,
            args.materialized_action_arm,
            enforce_expected_geometry=not official_start_materialized,
        )
        materialized_action_replay.update({
            "safety_baseline": (
                "official_DreaMS_on_corrected_graph"
                if official_start_materialized
                else "current_E8_on_corrected_graph"
            ),
            "initialization_correct_outer_train_queries": int(len(safety_examples)),
            "initialization_wrong_action_queries": int(
                len({example.query_index for example in error_action_examples})
            ),
        })
        print(
            f"[materialized E4 clean/action replay] "
            f"{json.dumps({'clean': materialized_clean_replay, 'action': materialized_action_replay})}",
            flush=True,
        )
        if complete_panel_historical_e4:
            branch_gradient_audit = audit_historical_e4_action_specific_alignment(
                model, store, action_examples,
                materialized_action_spectra, args.materialized_action_arm,
                head_parameters, backbone_parameters, device, args,
            )
            print(
                "[historical-E4 action-specific/clean-corrective alignment] "
                f"{json.dumps(branch_gradient_audit)}",
                flush=True,
            )
            model.zero_grad(set_to_none=True)
    # The array is immutable during optimization; keep a second name, not a
    # second 360 MB allocation.  Per-row anchors are views into the same owner.
    initial_reference_encoded = initial_encoded
    if args.candidate_boundary_loss:
        effective_boundary_action_weight, branch_gradient_audit = (
            calibrate_candidate_boundary_gradients(
                model, store, action_examples, safety_examples,
                training_anchor_by_row, device, args,
            )
        )
        print(
            f"[candidate-boundary gradient calibration] "
            f"{json.dumps(branch_gradient_audit)}",
            flush=True,
        )
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.guided_noise_policy == "selected":
        guided_teacher_replay = audit_guided_teacher_replay(
            model, store, guided_examples, device, args,
        )
        print(f"[E14 teacher replay] {json.dumps(guided_teacher_replay)}", flush=True)
        # Measure each branch at the identical mature initialization before any
        # update.  Historical E5 mixed branches before clipping, hiding whether
        # P was diluted or dominated the validated N/safety operating regime.
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient_records: dict[str, list[torch.Tensor | None]] = {}
        branch_losses: dict[str, float] = {}
        branch_norms: dict[str, float] = {}
        audit_batches = {
            "n_action": action_examples[: min(4, len(action_examples))],
            "safety": safety_examples[: min(4, len(safety_examples))],
            "p_corrective": guided_examples[: min(4, len(guided_examples))],
            "p_risk": guided_risk_examples[: min(4, len(guided_risk_examples))],
        }
        for branch, batch in audit_batches.items():
            if not batch:
                continue
            if branch == "n_action":
                if args.candidate_boundary_loss:
                    branch_loss, _ = candidate_boundary_action_loss(
                        model, store, batch, training_anchor_by_row, device, args,
                    )
                elif args.pmt_arm == "paired_target":
                    branch_loss, _ = paired_pmt_loss(
                        model, store, batch, training_anchor_by_row, device, args,
                    )
                else:
                    branch_loss, _ = direct_action_loss(
                        model, store, batch, training_anchor_by_row, device, args,
                        official_action_targets,
                        materialized_action_spectra,
                        args.materialized_action_arm,
                    )
            elif branch == "safety":
                if args.candidate_boundary_loss:
                    branch_loss, _ = candidate_boundary_safety_loss(
                        model, store, batch, training_anchor_by_row, device, args,
                    )
                else:
                    branch_loss, _ = safety_loss(
                        model, store, batch, training_anchor_by_row, device, args,
                    )
            else:
                branch_loss, _ = guided_noise_loss(
                    model, store, batch, training_anchor_by_row, device, args,
                )
            norm, gradients = detached_loss_gradients(
                branch_loss, trainable_parameters,
            )
            branch_losses[branch] = float(branch_loss.detach())
            branch_norms[branch] = float(norm)
            gradient_records[branch] = gradients
        pairwise_cosine: dict[str, float] = {}
        names = sorted(gradient_records)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                pairwise_cosine[f"{left}__{right}"] = gradient_cosine(
                    gradient_records[left], gradient_records[right]
                )
        branch_gradient_audit = {
            "examples_per_branch": {
                key: int(len(value)) for key, value in audit_batches.items()
            },
            "loss": branch_losses,
            "gradient_l2_norm": branch_norms,
            "gradient_cosine": pairwise_cosine,
            "p_corrective_to_n_norm_ratio": float(
                branch_norms.get("p_corrective", float("nan"))
                / max(branch_norms.get("n_action", 0.0), 1e-12)
            ),
            "p_corrective_to_safety_norm_ratio": float(
                branch_norms.get("p_corrective", float("nan"))
                / max(branch_norms.get("safety", 0.0), 1e-12)
            ),
        }
        if args.guided_auto_balance:
            base_components = [
                (gradient_records["n_action"], 1.0),
                (gradient_records["safety"], args.safety_stream_weight),
            ]
            risk_ratio = (
                args.guided_risk_control_ratio
                if "p_risk" in gradient_records else 0.0
            )
            normalizer = 1.0 + risk_ratio
            guided_components = [(
                gradient_records["p_corrective"], 1.0 / normalizer,
            )]
            if "p_risk" in gradient_records and args.guided_risk_control_ratio > 0:
                guided_components.append((
                    gradient_records["p_risk"], risk_ratio / normalizer,
                ))
            base_norm = combined_gradient_norm(base_components)
            guided_norm = combined_gradient_norm(guided_components)
            balance_scale = min(1.0, max(0.05, base_norm / max(guided_norm, 1e-12)))
            effective_guided_noise_weight = float(
                args.guided_noise_weight * balance_scale
            )
            branch_gradient_audit.update({
                "combined_n_safety_gradient_norm": float(base_norm),
                "combined_p_gradient_norm": float(guided_norm),
                "auto_balance_scale": float(balance_scale),
                "effective_guided_noise_weight": effective_guided_noise_weight,
            })
        print(f"[E14 branch gradients] {json.dumps(branch_gradient_audit)}", flush=True)
        del gradient_records
        model.zero_grad(set_to_none=True)
    live_shared_v3_preinjection_audit: dict[str, object] = {}
    if live_shared_v3:
        if (
            v3_targeted_action_spectra is None
            or v3_shuffled_action_spectra is None
        ):
            raise RuntimeError(
                "live-shared V3 lost targeted or shuffled spectra before gradient audit"
            )
        live_shared_v3_preinjection_audit = (
            audit_live_shared_v3_preinjection_gradients(
                model,
                store,
                action_examples,
                v3_targeted_action_spectra,
                v3_shuffled_action_spectra,
                objective_reference_by_row,
                {
                    "head": head_parameters,
                    "backbone": backbone_parameters,
                },
                device,
                args,
            )
        )
        print(
            "[V3 pre-injection gradients] "
            + json.dumps(live_shared_v3_preinjection_audit),
            flush=True,
        )
    del initial_encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    # Keep the already-validated N/safety sampling stream bitwise independent
    # of whether P-arm is enabled.  This makes the P-weight scan a true paired
    # intervention rather than a hidden resampling experiment.
    positive_rng = np.random.default_rng(args.seed + 104729)
    guided_rng = np.random.default_rng(args.seed + 209759)
    history = []
    epochs = 1 if args.smoke else args.epochs
    signal_preserving_schedule_report: dict[str, object] = {}
    live_shared_v3_schedule_report: dict[str, object] = {}
    if signal_preserving_v2:
        if signal_preserving_injector is None or materialized_action_spectra is None:
            raise RuntimeError("signal-preserving V2 injector/action spectra disappeared")
        history, signal_preserving_steps, signal_preserving_schedule_report = (
            train_e4_base_semantic_v2_epochs(
                model=model,
                optimizer=optimizer,
                injector=signal_preserving_injector,
                store=store,
                e4_base_examples=e4_base_action_examples,
                semantic_examples=action_examples,
                safety_examples=safety_examples,
                materialized_action_spectra=materialized_action_spectra,
                objective_reference_by_row=objective_reference_by_row,
                head_parameters=head_parameters,
                backbone_parameters=backbone_parameters,
                device=device,
                args=args,
                epochs=epochs,
            )
        )
    elif live_shared_v3:
        if live_shared_v3_injector is None or materialized_action_spectra is None:
            raise RuntimeError("live-shared V3 injector/action spectra disappeared")
        (
            history,
            live_shared_v3_steps,
            live_shared_v3_schedule_report,
        ) = train_e4_live_shared_v3_epochs(
            model=model,
            optimizer=optimizer,
            injector=live_shared_v3_injector,
            store=store,
            e4_base_examples=e4_base_action_examples,
            action_examples=action_examples,
            safety_examples=safety_examples,
            materialized_action_spectra=materialized_action_spectra,
            objective_reference_by_row=objective_reference_by_row,
            head_parameters=head_parameters,
            backbone_parameters=backbone_parameters,
            device=device,
            args=args,
            epochs=epochs,
        )
    pmt_schedules = None
    materialized_schedules = None
    if args.pmt_arm != "none":
        if args.candidate_boundary_loss and args.candidate_boundary_version == "v2_molecule_max":
            pmt_schedules = coverage_first_query_schedules(
                action_examples, epochs, args.seed,
            )
        else:
            identifiers = [f"{item.query_index}|{item.policy}" for item in action_examples]
            pmt_schedules = coverage_first_schedules(
                identifiers,
                [item.identity for item in action_examples],
                [item.policy for item in action_examples],
                epochs, args.views_per_identity, args.seed,
            )
        flat_indices = [index for schedule in pmt_schedules for index in schedule]
        first_exposure = flat_indices[:len(action_examples)]
        if len(set(first_exposure)) != len(action_examples):
            raise RuntimeError("PMT recycled an action before complete first exposure")
    elif (
        args.action_selection == "materialized_routed"
        and not signal_preserving_v2
        and not live_shared_v3
    ):
        action_ids = [item.action_id for item in action_examples]
        if not all(action_ids) or len(set(action_ids)) != len(action_ids):
            raise RuntimeError("materialized E4 action identifiers are empty or duplicated")
        materialized_schedules = coverage_first_identity_balanced_schedules(
            action_ids,
            [item.identity for item in action_examples],
            [item.policy for item in action_examples],
            epochs,
            args.views_per_identity,
            args.seed,
        )
        flat_indices = [index for schedule in materialized_schedules for index in schedule]
        first_exposure = flat_indices[:len(action_examples)]
        if len(set(first_exposure)) != len(action_examples):
            raise RuntimeError(
                "materialized E4 recycled an action before complete first exposure"
            )
        identity_unique = pd.Series(
            [item.identity for item in action_examples], dtype=str,
        ).value_counts().astype(int).to_dict()
        identity_exposure = pd.Series([
            action_examples[index].identity
            for schedule in materialized_schedules for index in schedule
        ], dtype=str).value_counts().astype(int).to_dict()
        if args.materialized_injection_mode == "multi_action_balanced":
            expected_identity_exposure = {
                identity: max(count, epochs * args.views_per_identity)
                for identity, count in identity_unique.items()
            }
            if identity_exposure != expected_identity_exposure:
                raise RuntimeError("materialized multi-action physical coverage drifted")
            if (
                len(train_actions) != len(action_examples)
                or train_actions["action_id"].astype(str).tolist() != action_ids
            ):
                raise RuntimeError(
                    "materialized training table and direct examples lost row alignment"
                )
            scheduled_weights = identity_family_balanced_exposure_weights(
                train_actions,
                flat_indices,
                total_weight_per_identity=float(epochs * args.views_per_identity),
            )
            action_examples = [
                replace(example, sample_weight=float(scheduled_weights[index]))
                for index, example in enumerate(action_examples)
            ]
            train_actions = train_actions.copy()
            train_actions["corrective_weight"] = scheduled_weights
            effective_by_identity: dict[str, float] = {}
            for index in flat_indices:
                example = action_examples[index]
                effective_by_identity[example.identity] = (
                    effective_by_identity.get(example.identity, 0.0)
                    + float(example.sample_weight)
                )
            expected_effective = float(epochs * args.views_per_identity)
            if not all(
                math.isclose(value, expected_effective, abs_tol=2e-5)
                for value in effective_by_identity.values()
            ):
                raise RuntimeError(
                    "materialized multi-action scheduled effective dose drifted"
                )
            materialized_action_control_report.update({
                "historical_identity_exposure_budget_restored": True,
                "identity_effective_dose_restored": True,
                "identity_family_effective_dose_normalized": True,
                "effective_action_weight_sum": float(sum(
                    action_examples[index].sample_weight for index in flat_indices
                )),
                "all_strict_actions_exposed_before_recycling": True,
                "all_strict_actions_exposed_exactly_once": bool(
                    len(flat_indices) == len(action_examples)
                ),
                "physical_action_recycling_used": bool(
                    len(flat_indices) > len(action_examples)
                ),
                "identity_exposure_target_per_identity": int(
                    epochs * args.views_per_identity
                ),
                "identities_exceeding_historical_budget": int(sum(
                    count > epochs * args.views_per_identity
                    for count in identity_unique.values()
                )),
            })
        else:
            expected_identity_exposure = {
                identity: max(count, epochs * args.views_per_identity)
                for identity, count in identity_unique.items()
            }
            if identity_exposure != expected_identity_exposure:
                raise RuntimeError("materialized E4 identity exposure budget drifted")
            materialized_action_control_report.update({
                "historical_identity_exposure_budget_restored": True,
                "identity_effective_dose_restored": bool(
                    args.materialized_injection_mode == "one_best_e4"
                ),
                "historical_minimum_identity_exposure_budget_preserved": True,
                "identity_effective_dose_equalized": bool(
                    args.materialized_injection_mode == "one_best_e4"
                ),
                "all_strict_actions_exposed_before_recycling": True,
                "all_strict_actions_exposed_exactly_once": bool(
                    len(flat_indices) == len(action_examples)
                ),
                "physical_action_recycling_used": bool(
                    len(flat_indices) > len(action_examples)
                ),
                "identity_exposure_target_per_identity": int(
                    epochs * args.views_per_identity
                ),
                "identity_exposure_minimum": int(min(identity_exposure.values())),
                "identity_exposure_median": float(np.median(list(
                    identity_exposure.values()
                ))),
                "identity_exposure_maximum": int(max(identity_exposure.values())),
                "identities_exceeding_historical_budget": int(sum(
                    count > epochs * args.views_per_identity
                    for count in identity_unique.values()
                )),
            })
    materialized_preclip_loss_scale = (
        MATERIALIZED_MULTI_ACTION_PRECLIP_LOSS_SCALE
        if args.action_selection == "materialized_routed"
        and args.materialized_injection_mode == "multi_action_balanced"
        else 1.0
    )
    if args.action_selection == "materialized_routed":
        materialized_action_control_report.update({
            "static_preclip_loss_scale": float(materialized_preclip_loss_scale),
            "static_preclip_scale_source": (
                "job2333052_observed_training_clip_scale_not_held_performance"
                if args.materialized_injection_mode == "multi_action_balanced"
                else "historical_unscaled_E4"
            ),
        })
    for epoch in (
        () if (signal_preserving_v2 or live_shared_v3)
        else range(1, epochs + 1)
    ):
        model.eval()
        if args.candidate_boundary_loss and args.refresh_hard_negatives:
            refresh_encoded = encode_rows(
                model, store, reachable_rows, device, args.eval_batch_size, False,
                f"E4-boundary-refresh-e{epoch}",
            )
            action_examples = refresh_candidate_references(
                graph, action_examples, reachable_rows, refresh_encoded,
                args.positive_spectra, args.negative_molecules,
                preserve_forced_negative=(
                    args.candidate_boundary_version != "v2_molecule_max"
                ),
            )
            safety_examples = refresh_candidate_references(
                graph, safety_examples, reachable_rows, refresh_encoded,
                args.positive_spectra, args.negative_molecules,
            )
            error_action_examples = [
                example for example in action_examples if example.official_rank != 1
            ]
            del refresh_encoded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        epoch_actions = (
            [action_examples[index] for index in pmt_schedules[epoch - 1]]
            if pmt_schedules is not None else
            (
                [action_examples[index] for index in materialized_schedules[epoch - 1]]
                if materialized_schedules is not None else
                identity_balanced_epoch(action_examples, rng, args.views_per_identity)
            )
        )
        if args.error_views_per_identity > 0:
            extra_errors = identity_balanced_epoch(
                error_action_examples, rng, args.error_views_per_identity,
            )
            epoch_actions.extend(extra_errors)
            rng.shuffle(epoch_actions)
        safety_epoch = identity_balanced_epoch(safety_examples, rng, 1)
        positive_epoch = (
            identity_balanced_epoch(
                positive_examples, positive_rng, args.positive_views_per_identity,
            ) if positive_examples else []
        )
        guided_epoch = (
            identity_balanced_epoch(
                guided_examples, guided_rng, args.guided_noise_views_per_identity,
            ) if guided_examples else []
        )
        guided_risk_epoch = (
            identity_balanced_epoch(guided_risk_examples, guided_rng, 1)
            if guided_risk_examples else []
        )
        if args.smoke:
            epoch_actions = (
                query_complete_action_batches(epoch_actions, 16)[0]
                if args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
                else epoch_actions[:16]
            )
            safety_epoch = safety_epoch[:32]
            positive_epoch = positive_epoch[:32]
            guided_epoch = guided_epoch[:32]
            guided_risk_epoch = guided_risk_epoch[:32]
        action_schedule_sha256 = sampling_schedule_sha256(epoch_actions)
        action_exposure_sha256 = action_exposure_schedule_sha256(epoch_actions)
        safety_schedule_sha256 = sampling_schedule_sha256(safety_epoch)
        action_batches = (
            query_complete_action_batches(epoch_actions, args.batch_actions)
            if args.candidate_boundary_loss
            and args.candidate_boundary_version == "v2_molecule_max"
            else list(batched(epoch_actions, args.batch_actions))
        )
        steps = len(action_batches)
        totals: dict[str, float] = {}
        started = time.time()
        safety_cursor = 0
        positive_cursor = 0
        guided_cursor = 0
        guided_risk_cursor = 0
        for step, action_batch in enumerate(action_batches, start=1):
            action_dose = (
                len({int(item.query_index) for item in action_batch})
                if args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
                else (
                    sum(float(item.sample_weight) for item in action_batch)
                    if args.action_selection == "materialized_routed"
                    and args.materialized_injection_mode == "multi_action_balanced"
                    else len(action_batch)
                )
            )
            # Keep the physical safety forward bounded by the four-action E4
            # microbatch.  A singleton identity can legitimately give one
            # action weight 16; turning that weight into 16 physical safety
            # examples would recreate the Graphormer OOM.  Effective safety
            # dose is restored as a scalar below instead.
            safety_size = max(1, int(round(len(action_batch) * args.safety_ratio)))
            safety_dose_multiplier = (
                float(action_dose) / max(float(len(action_batch)), 1.0)
                if args.action_selection == "materialized_routed"
                and args.materialized_injection_mode == "multi_action_balanced"
                else 1.0
            )
            if safety_cursor + safety_size > len(safety_epoch):
                rng.shuffle(safety_epoch)
                safety_cursor = 0
            safe_batch = safety_epoch[safety_cursor:safety_cursor + safety_size]
            safety_cursor += safety_size
            optimizer.zero_grad(set_to_none=True)
            if args.candidate_boundary_loss:
                action_loss, action_log = candidate_boundary_action_loss(
                    model, store, action_batch, training_anchor_by_row, device, args,
                )
            elif args.pmt_arm == "paired_target":
                action_loss, action_log = paired_pmt_loss(
                    model, store, action_batch, training_anchor_by_row, device, args,
                )
            else:
                action_loss, action_log = direct_action_loss(
                    model, store, action_batch, objective_reference_by_row, device, args,
                    official_action_targets,
                    materialized_action_spectra,
                    args.materialized_action_arm,
                )
            scaler.scale(
                materialized_preclip_loss_scale
                * effective_boundary_action_weight * action_loss
            ).backward()
            if e4_action_injector is not None:
                # Snapshot the complete, unchanged historical E4 branch before
                # the matched safety backward accumulates into parameter.grad.
                # ActionInjectorV1 remains blind to spectra/actions/loss terms.
                e4_action_injector.capture_corrective_()
            positive_log: dict[str, float] = {}
            positive_loss_value = 0.0
            if positive_epoch and args.positive_stream_weight > 0:
                positive_size = max(1, int(round(len(action_batch) * args.positive_ratio)))
                if positive_cursor + positive_size > len(positive_epoch):
                    positive_rng.shuffle(positive_epoch)
                    positive_cursor = 0
                positive_batch = positive_epoch[positive_cursor:positive_cursor + positive_size]
                positive_cursor += positive_size
                positive_loss, positive_log = positive_arm_loss(
                    model, store, positive_batch, training_anchor_by_row, device, args,
                )
                positive_loss_value = float(positive_loss.detach())
                scaler.scale(args.positive_stream_weight * positive_loss).backward()
            guided_log: dict[str, float] = {}
            guided_loss_value = 0.0
            if guided_epoch and args.guided_noise_policy != "none":
                guided_size = max(1, int(round(len(action_batch) * args.guided_noise_ratio)))
                if guided_cursor + guided_size > len(guided_epoch):
                    guided_rng.shuffle(guided_epoch)
                    guided_cursor = 0
                guided_batch = guided_epoch[guided_cursor:guided_cursor + guided_size]
                guided_cursor += guided_size
                if guided_risk_epoch and args.guided_risk_control_ratio > 0:
                    risk_size = max(
                        1, int(round(guided_size * args.guided_risk_control_ratio))
                    )
                    if guided_risk_cursor + risk_size > len(guided_risk_epoch):
                        guided_rng.shuffle(guided_risk_epoch)
                        guided_risk_cursor = 0
                    guided_batch = guided_batch + guided_risk_epoch[
                        guided_risk_cursor:guided_risk_cursor + risk_size
                    ]
                    guided_risk_cursor += risk_size
                guided_loss, guided_log = guided_noise_loss(
                    model, store, guided_batch, training_anchor_by_row, device, args,
                )
                guided_loss_value = float(guided_loss.detach())
                scaler.scale(effective_guided_noise_weight * guided_loss).backward()
            if args.candidate_boundary_loss:
                safe_loss, safe_log = candidate_boundary_safety_loss(
                    model, store, safe_batch, training_anchor_by_row, device, args,
                )
            else:
                safe_loss, safe_log = safety_loss(
                    model, store, safe_batch, objective_reference_by_row, device, args,
                )
            scaler.scale(
                materialized_preclip_loss_scale
                * args.safety_stream_weight * safety_dose_multiplier * safe_loss
            ).backward()
            scaler.unscale_(optimizer)
            head_grad_norm = gradient_l2_norm(head_parameters)
            backbone_grad_norm = gradient_l2_norm(backbone_parameters)
            if e4_action_injector is not None:
                injection_step = e4_action_injector.step_and_inject_(
                    maximum_gradient_norm=args.grad_clip,
                )
                e4_action_injector_steps.append(injection_step)
                grad_norm_value = injection_step.gradient_norm_before_clip
                clip_scale = injection_step.global_clip_retention
                clip_applied = float(clip_scale < 1.0)
                # Materialized E4 is full fp32 by contract, so the scaler is
                # disabled.  Updating it is an explicit no-op and no second
                # optimizer step is allowed here.
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.grad_clip,
                )
                grad_norm_value = float(grad_norm)
                clip_applied = float(grad_norm_value > args.grad_clip)
                clip_scale = min(
                    1.0, args.grad_clip / max(grad_norm_value, 1e-12),
                )
                scaler.step(optimizer)
                scaler.update()
            log = {
                "loss": (
                    materialized_preclip_loss_scale * (
                        effective_boundary_action_weight * float(action_loss.detach())
                        + args.safety_stream_weight * safety_dose_multiplier
                        * float(safe_loss.detach())
                    )
                    + args.positive_stream_weight * positive_loss_value
                    + effective_guided_noise_weight * guided_loss_value
                ),
                "gradient_norm": grad_norm_value,
                "head_gradient_norm": head_grad_norm,
                "backbone_gradient_norm": backbone_grad_norm,
                "gradient_clip_applied": clip_applied,
                "gradient_clip_scale": clip_scale,
                "materialized_preclip_loss_scale": float(
                    materialized_preclip_loss_scale
                ),
                "effective_action_dose": float(action_dose),
                "safety_dose_multiplier": float(safety_dose_multiplier),
                "physical_safety_batch_size": int(len(safe_batch)),
                **action_log, **positive_log, **guided_log, **safe_log,
            }
            if e4_action_injector is not None:
                log.update({
                    "injector_v1_fraction_error": (
                        injection_step.maximum_fraction_abs_error
                    ),
                    "injector_v1_minimum_protective_retention": min(
                        injection_step.protective_component_retention_by_group.values()
                    ),
                    "injector_v1_update_norm_ratio": (
                        injection_step.maximum_update_norm_ratio
                    ),
                    "injector_v1_virtual_adamw_error": (
                        injection_step.virtual_adamw_relative_error
                    ),
                    "injector_v1_first_moment_error": (
                        injection_step.first_moment_reconstruction_relative_error
                    ),
                    "injector_v1_fp32_replay_error": (
                        injection_step.fp32_parameter_replay_relative_error
                    ),
                })
            for key, value in log.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if step % 25 == 0 or step == steps:
                print(
                    f"[E4A epoch={epoch}] {step}/{steps} "
                    f"loss={totals['loss']/step:.5f} grad={totals['gradient_norm']/step:.4f}",
                    flush=True,
                )
        record = {key: value / steps for key, value in totals.items()}
        record.update({
            "epoch": epoch,
            "steps": steps,
            "seconds": time.time() - started,
            "action_sampling_schedule_sha256": action_schedule_sha256,
            "action_exposure_schedule_sha256": action_exposure_sha256,
            "safety_sampling_schedule_sha256": safety_schedule_sha256,
        })
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    e4_action_injector_report = summarize_e4_action_injector_v1_steps(
        e4_action_injector_steps,
    )
    if e4_action_injector is not None:
        e4_action_injector_report["contract"] = (
            e4_action_injector.audit_manifest()
        )
        if int(e4_action_injector_report["steps"]) != sum(
            int(record["steps"]) for record in history
        ):
            raise RuntimeError(
                "E4 ActionInjectorV1 did not audit every optimizer step"
            )
        if e4_action_injector_report.get("gate_passed") is not True:
            raise RuntimeError(
                "E4 ActionInjectorV1 final signal gate failed: "
                f"{e4_action_injector_report}"
            )
    signal_preserving_injector_report = summarize_signal_preserving_v2_steps(
        signal_preserving_steps,
    )
    if signal_preserving_injector is not None:
        signal_preserving_injector_report["contract"] = (
            signal_preserving_injector.audit_manifest()
        )
        if int(signal_preserving_injector_report.get("steps", -1)) != sum(
            int(record["steps"]) for record in history
        ):
            raise RuntimeError(
                "signal-preserving V2 did not audit every optimizer step"
            )
        if signal_preserving_injector_report.get("gate_passed") is not True:
            raise RuntimeError(
                "signal-preserving V2 final gate failed: "
                f"{signal_preserving_injector_report}"
            )
    live_shared_v3_injector_report = summarize_separated_e4_action_steps_v3(
        live_shared_v3_steps,
        target_action_fraction=args.injector_target_attributable_fraction,
        minimum_e4_projection_retention=(
            args.injector_minimum_protective_retention
        ),
        maximum_update_norm_ratio=args.injector_maximum_update_norm_ratio,
    )
    if live_shared_v3_injector is not None:
        live_shared_v3_injector_report["contract"] = (
            live_shared_v3_injector.audit_manifest()
        )
        if int(live_shared_v3_injector_report.get("steps", -1)) != sum(
            int(record["steps"]) for record in history
        ):
            raise RuntimeError("live-shared V3 did not audit every optimizer step")
        if live_shared_v3_injector_report.get("gate_passed") is not True:
            raise RuntimeError(
                "live-shared V3 optimizer contract failed: "
                f"{live_shared_v3_injector_report}"
            )

    final_encoded = encode_rows(
        model, store, reachable_rows, device, args.eval_batch_size, False, "E4A-final-fp32",
    )
    final_rank, final_summary = streaming_evaluate_embeddings(
        graph, reachable_rows, final_encoded, held_queries,
    )
    baseline_detail_rank, baseline_top_molecule, baseline_full_margin = full_graph_query_details(
        graph, reachable_rows, official_encoded, held_queries,
    )
    initial_detail_rank, initial_top_molecule, initial_full_margin = full_graph_query_details(
        graph, reachable_rows, initial_reference_encoded, held_queries,
    )
    final_detail_rank, final_top_molecule, final_full_margin = full_graph_query_details(
        graph, reachable_rows, final_encoded, held_queries,
    )
    if not np.array_equal(baseline_detail_rank, baseline_rank):
        raise RuntimeError("full-graph baseline detail ranks disagree with primary evaluation")
    if not np.array_equal(initial_detail_rank, initial_rank):
        raise RuntimeError("full-graph initialization detail ranks disagree with primary evaluation")
    if not np.array_equal(final_detail_rank, final_rank):
        raise RuntimeError("full-graph final detail ranks disagree with primary evaluation")
    baseline_multimetric = graph_multimetric_summary(
        graph, reachable_rows, official_encoded, held_queries,
    )
    initialization_multimetric = graph_multimetric_summary(
        graph, reachable_rows, initial_reference_encoded, held_queries,
    )
    final_multimetric = graph_multimetric_summary(
        graph, reachable_rows, final_encoded, held_queries,
    )
    corrected_metric_panel: dict[str, object] = {}
    if corrected_query_adduct is not None:
        official_corrected_metrics, official_corrected_query = corrected_full_metrics(
            graph, corrected_official_scores(graph),
            query_adduct=corrected_query_adduct, queries=held_queries,
        )
        initial_corrected_metrics, initial_corrected_query = corrected_full_metrics(
            graph,
            corrected_score_embedding_query_subset(
                graph, reachable_rows, initial_reference_encoded, held_queries,
            ),
            query_adduct=corrected_query_adduct, queries=held_queries,
        )
        final_corrected_metrics, final_corrected_query = corrected_full_metrics(
            graph,
            corrected_score_embedding_query_subset(
                graph, reachable_rows, final_encoded, held_queries,
            ),
            query_adduct=corrected_query_adduct, queries=held_queries,
        )
        paired_vs_official = corrected_paired_outcome_table(
            official_corrected_query, final_corrected_query,
        )
        paired_vs_initial = corrected_paired_outcome_table(
            initial_corrected_query, final_corrected_query,
        )

        def paired_counts(frame: pd.DataFrame) -> dict[str, object]:
            near = frame["near"].to_numpy(bool)
            corrected = frame["corrected"].to_numpy(bool)
            introduced = frame["introduced"].to_numpy(bool)
            return {
                "corrected": int(np.sum(corrected)),
                "introduced": int(np.sum(introduced)),
                "risk_net_lambda2": int(np.sum(corrected) - 2 * np.sum(introduced)),
                "near": {
                    "queries": int(np.sum(near)),
                    "corrected": int(np.sum(corrected & near)),
                    "introduced": int(np.sum(introduced & near)),
                    "risk_net_lambda2": int(
                        np.sum(corrected & near) - 2 * np.sum(introduced & near)
                    ),
                },
            }

        corrected_metric_panel = {
            "official": official_corrected_metrics,
            "initialization": initial_corrected_metrics,
            "student": final_corrected_metrics,
            "student_minus_official": nested_numeric_delta(
                final_corrected_metrics, official_corrected_metrics,
            ),
            "student_minus_initialization": nested_numeric_delta(
                final_corrected_metrics, initial_corrected_metrics,
            ),
            "paired_top1_vs_official": paired_counts(paired_vs_official),
            "paired_top1_vs_initialization": paired_counts(paired_vs_initial),
        }
    final_preservation = np.einsum("ij,ij->i", final_encoded, initial_reference_encoded)
    final_official_cosine = np.einsum("ij,ij->i", final_encoded, official_encoded)
    old_correct, new_correct = baseline_rank == 1, final_rank == 1
    initial_correct = initial_rank == 1
    delta_ci = formula_bootstrap_delta(
        baseline_rank, final_rank, graph.query_formula[held_queries],
        args.bootstrap_resamples, args.seed,
    )
    incremental_ci = formula_bootstrap_delta(
        initial_rank, final_rank, graph.query_formula[held_queries],
        args.bootstrap_resamples, args.seed + 1,
    )
    final_summary.update({
        "baseline_recall1": baseline_summary["recall1"],
        "delta_recall1": float(final_summary["recall1"] - baseline_summary["recall1"]),
        "baseline_mrr": baseline_summary["mrr"],
        "delta_mrr": float(final_summary["mrr"] - baseline_summary["mrr"]),
        "baseline_near_recall1": baseline_summary["near_recall1"],
        "delta_near_recall1": float(final_summary["near_recall1"] - baseline_summary["near_recall1"]),
        "corrected": int(np.sum(~old_correct & new_correct)),
        "introduced": int(np.sum(old_correct & ~new_correct)),
        "top_molecule_switches": int(np.sum(final_top_molecule != baseline_top_molecule)),
        "wrong_to_different_wrong": int(np.sum(
            (~old_correct) & (~new_correct)
            & (final_top_molecule != baseline_top_molecule)
        )),
        "risk_net": int(np.sum(~old_correct & new_correct) - 2 * np.sum(old_correct & ~new_correct)),
        "preservation_mean": float(np.mean(final_preservation)),
        "preservation_p01": float(np.quantile(final_preservation, 0.01)),
        "formula_cluster_delta_recall1": delta_ci,
        "initialization_recall1": initial_summary["recall1"],
        "incremental_delta_recall1": float(
            final_summary["recall1"] - initial_summary["recall1"]
        ),
        "initialization_mrr": initial_summary["mrr"],
        "incremental_delta_mrr": float(final_summary["mrr"] - initial_summary["mrr"]),
        "initialization_near_recall1": initial_summary["near_recall1"],
        "incremental_delta_near_recall1": float(
            final_summary["near_recall1"] - initial_summary["near_recall1"]
        ),
        "incremental_corrected": int(np.sum(~initial_correct & new_correct)),
        "incremental_introduced": int(np.sum(initial_correct & ~new_correct)),
        "incremental_top_molecule_switches": int(np.sum(
            final_top_molecule != initial_top_molecule
        )),
        "incremental_wrong_to_different_wrong": int(np.sum(
            (~initial_correct) & (~new_correct)
            & (final_top_molecule != initial_top_molecule)
        )),
        "incremental_risk_net": int(
            np.sum(~initial_correct & new_correct) - 2 * np.sum(initial_correct & ~new_correct)
        ),
        "initialization_formula_cluster_delta_recall1": incremental_ci,
        "preservation_vs_initialization_mean": float(np.mean(final_preservation)),
        "preservation_vs_initialization_p01": float(np.quantile(final_preservation, 0.01)),
        "cosine_vs_official_mean": float(np.mean(final_official_cosine)),
        "mean_full_margin_delta_vs_official": float(
            np.mean(final_full_margin - baseline_full_margin)
        ),
        "mean_full_margin_delta_vs_initialization": float(
            np.mean(final_full_margin - initial_full_margin)
        ),
        "top_molecule_changed_vs_official": int(np.sum(
            final_top_molecule != baseline_top_molecule
        )),
        "wrong_to_different_wrong": int(np.sum(
            (baseline_rank != 1) & (final_rank != 1)
            & (final_top_molecule != baseline_top_molecule)
        )),
        "complete_candidate_metrics": {
            "official": baseline_multimetric,
            "initialization": initialization_multimetric,
            "student": final_multimetric,
            "student_minus_official": {
                key: float(final_multimetric[key] - baseline_multimetric[key])
                for key in final_multimetric
                if isinstance(final_multimetric[key], float)
            },
            "student_minus_initialization": {
                key: float(final_multimetric[key] - initialization_multimetric[key])
                for key in final_multimetric
                if isinstance(final_multimetric[key], float)
            },
        },
        "corrected_graph_registered_metrics": corrected_metric_panel,
    })
    if args.positive_stream_weight > 0 and not held_positive.empty:
        row_to_final = {int(row): final_encoded[index] for index, row in enumerate(reachable_rows)}
        row_to_official = {int(row): official_encoded[index] for index, row in enumerate(reachable_rows)}
        pair_official: list[float] = []
        pair_final: list[float] = []
        for row in held_positive.itertuples(index=False):
            q, p = int(row.query_row), int(row.positive_row)
            pair_official.append(float(np.dot(row_to_official[q], row_to_official[p])))
            pair_final.append(float(np.dot(row_to_final[q], row_to_final[p])))
        pair_official_array = np.asarray(pair_official, dtype=np.float64)
        pair_final_array = np.asarray(pair_final, dtype=np.float64)
        identity_delta = pd.DataFrame({
            "identity": held_positive["query_ik14"].astype(str).to_numpy(),
            "delta": pair_final_array - pair_official_array,
        }).groupby("identity", sort=True)["delta"].mean().to_numpy()
        final_summary["held_cross_condition_positive"] = {
            "pairs": int(len(held_positive)),
            "identities": int(held_positive["query_ik14"].nunique()),
            "baseline_cosine": float(np.mean(pair_official_array)),
            "student_cosine": float(np.mean(pair_final_array)),
            "delta_cosine": float(np.mean(pair_final_array - pair_official_array)),
            "identity_mean_delta_cosine": float(np.mean(identity_delta)),
            "fraction_pairs_improved": float(np.mean(pair_final_array > pair_official_array)),
        }
    held_action_query = np.unique(held_action["query_index"].to_numpy(np.int64))
    held_action_mask = np.isin(held_queries, held_action_query)
    if np.any(held_action_mask):
        final_summary["held_action_clean"] = {
            "queries": int(np.sum(held_action_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_action_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_action_mask] == 1)),
            "corrected": int(np.sum((baseline_rank[held_action_mask] != 1) & (final_rank[held_action_mask] == 1))),
            "introduced": int(np.sum((baseline_rank[held_action_mask] == 1) & (final_rank[held_action_mask] != 1))),
        }
    if args.positive_stream_weight > 0 and not held_positive.empty:
        held_positive_query = np.unique(held_positive["query_index"].to_numpy(np.int64))
        held_positive_mask = np.isin(held_queries, held_positive_query)
        final_summary["held_positive_clean"] = {
            "queries": int(np.sum(held_positive_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_positive_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_positive_mask] == 1)),
            "delta_accuracy": float(
                np.mean(final_rank[held_positive_mask] == 1)
                - np.mean(baseline_rank[held_positive_mask] == 1)
            ),
            "corrected": int(np.sum((baseline_rank[held_positive_mask] != 1) & (final_rank[held_positive_mask] == 1))),
            "introduced": int(np.sum((baseline_rank[held_positive_mask] == 1) & (final_rank[held_positive_mask] != 1))),
        }
    if args.guided_noise_policy != "none" and not held_guided.empty:
        held_guided_query = np.unique(held_guided["query_index"].to_numpy(np.int64))
        held_guided_mask = np.isin(held_queries, held_guided_query)
        guided_scope_summary = {
            "queries": int(np.sum(held_guided_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_guided_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_guided_mask] == 1)),
            "delta_accuracy": float(
                np.mean(final_rank[held_guided_mask] == 1)
                - np.mean(baseline_rank[held_guided_mask] == 1)
            ),
            "corrected": int(np.sum(
                (baseline_rank[held_guided_mask] != 1) & (final_rank[held_guided_mask] == 1)
            )),
            "introduced": int(np.sum(
                (baseline_rank[held_guided_mask] == 1) & (final_rank[held_guided_mask] != 1)
            )),
        }
        final_summary["held_guided_action_scope"] = guided_scope_summary
        if args.guided_query_scope == "positive_deficit_errors":
            final_summary["held_guided_positive_deficit"] = guided_scope_summary

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({
        "query_index": held_queries,
        "query_row": graph.query_row[held_queries],
        "query_ik14": graph.query_ik14[held_queries],
        "query_formula": graph.query_formula[held_queries],
        "has_near": graph.query_has_near[held_queries],
        "baseline_rank": baseline_rank,
        "initialization_rank": initial_rank,
        "final_rank": final_rank,
        "baseline_top_molecule_local": baseline_top_molecule,
        "initialization_top_molecule_local": initial_top_molecule,
        "final_top_molecule_local": final_top_molecule,
        "baseline_full_margin": baseline_full_margin,
        "initialization_full_margin": initial_full_margin,
        "final_full_margin": final_full_margin,
        "corrected": (baseline_rank != 1) & (final_rank == 1),
        "introduced": (baseline_rank == 1) & (final_rank != 1),
    }).to_csv(output / "held_per_query.csv.gz", index=False, compression="gzip")
    checkpoint = {
        "status": "noise_final_e4a_direct_shared_dreams_encoder",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initialization": initialization,
        "policy": args.policy,
        "action_scope": args.action_scope,
        "seed": args.seed,
        "outer_fold": args.outer_fold,
        "causal_arm": args.causal_arm,
        "action_selection": args.action_selection,
        "materialized_action_arm": args.materialized_action_arm,
        "materialized_injection_mode": args.materialized_injection_mode,
        "optimizer_boundary_mode": args.optimizer_boundary_mode,
        "pmt_arm": args.pmt_arm,
        "pmt_alpha": args.pmt_alpha,
        "candidate_boundary_loss": args.candidate_boundary_loss,
        "refresh_hard_negatives": args.refresh_hard_negatives,
        "capacity": capacity,
        "initial_student_checkpoint_sha256": (
            sha256_file(args.initial_student_checkpoint)
            if args.initial_student_checkpoint is not None else None
        ),
        "inference_clean_only": True,
        "P2b_used": False,
        "teacher_used": bool(
            args.direct_transfer_mode == "official_action"
            or args.guided_noise_policy == "selected"
            or (args.pmt_arm == "paired_target" and not args.candidate_boundary_loss)
        ),
        "teacher_kind": (
            "frozen_official_raw_action_embedding"
            if args.direct_transfer_mode == "official_action"
            else (
                "outer_fold_isolated_privileged_action_margin"
                if args.guided_noise_policy == "selected" else (
                    "outer_train_current_geometry_target_control_advantage"
                    if args.pmt_arm == "paired_target" and not args.candidate_boundary_loss
                    else "none"
                )
            )
        ),
        "positive_arm_used": bool(args.positive_stream_weight > 0),
        "guided_noise_policy": args.guided_noise_policy,
        "guided_query_scope": args.guided_query_scope,
        "guided_noise_used": bool(args.guided_noise_policy != "none"),
        "guided_transfer_mode": args.guided_transfer_mode,
        "guided_recurrence_prevalence": args.guided_recurrence_prevalence,
        "guided_recurrence_max_peaks": args.guided_recurrence_max_peaks,
        "guided_teacher_target_mode": args.guided_teacher_target_mode,
        "guided_teacher_delta_fraction": args.guided_teacher_delta_fraction,
        "guided_risk_control_ratio": args.guided_risk_control_ratio,
    }
    final_checkpoint_path = output / "final_shared_encoder.pt"
    torch.save(checkpoint, final_checkpoint_path)
    gates = {
        "clean_recall_positive": bool(final_summary["delta_recall1"] > 0),
        "formula_ci_positive": bool(delta_ci["ci_low"] > 0),
        "corrected_gt_introduced": bool(final_summary["corrected"] > final_summary["introduced"]),
        "risk_net_positive": bool(final_summary["risk_net"] > 0),
        "near_nonnegative": bool(final_summary["delta_near_recall1"] >= 0),
        "mrr_nonnegative": bool(final_summary["delta_mrr"] >= 0),
        "preservation_ge_0_995": bool(final_summary["preservation_mean"] >= 0.995),
    }
    promotion_reference = "official"
    promotion_diagnostics: dict[str, object] = {}
    if live_shared_v3_injector is not None:
        promotion_reference = "initialization"
        gates, promotion_diagnostics = live_shared_v3_single_arm_promotion(
            corrected_metric_panel, final_summary, incremental_ci,
        )
    if e4_action_injector is not None:
        gates["action_injector_v1_signal_and_safety"] = bool(
            e4_action_injector_report.get("gate_passed") is True
        )
    if signal_preserving_injector is not None:
        gates["signal_preserving_v2_update_contract"] = bool(
            signal_preserving_injector_report.get("gate_passed") is True
        )
    if live_shared_v3_injector is not None:
        gates["live_shared_v3_update_contract"] = bool(
            live_shared_v3_injector_report.get("gate_passed") is True
        )
        gates["live_shared_v3_preinjection_gradient_contract"] = bool(
            live_shared_v3_preinjection_audit.get("gate_passed") is True
        )
    if args.positive_stream_weight > 0:
        gates.update({
            "cross_condition_pair_cosine_positive": bool(
                final_summary["held_cross_condition_positive"]["delta_cosine"] > 0
            ),
            "cross_condition_query_recall_nonnegative": bool(
                final_summary["held_positive_clean"]["delta_accuracy"] >= 0
            ),
        })
    if args.guided_noise_policy not in {"none", "selected"}:
        gates.update({
            "guided_action_scope_corrections_positive": bool(
                final_summary["held_guided_action_scope"]["corrected"] > 0
            ),
        })
    elif args.guided_noise_policy == "selected":
        gates.update({
            "crossfit_teacher_outer_fold_excluded": True,
            "teacher_action_replay_exact": bool(
                guided_teacher_replay.get("action_margin_max_abs_error", 1.0) <= 2e-4
            ),
            "incremental_formula_ci_nonnegative": bool(
                incremental_ci["ci_low"] >= 0
            ),
            "incremental_corrected_gt_introduced": bool(
                final_summary["incremental_corrected"]
                > final_summary["incremental_introduced"]
            ),
        })
    decision = {
        "status": "noise_final_e4a_direct_augmentation_complete",
        "formal": not args.smoke,
        "configuration": vars(args) | {
            "r0_fixed_cells": FIXED_POLICY[args.policy] if args.action_selection == "fixed" else [],
            "pmt_manifest_corrective_query_scope": pmt_manifest_corrective_query_scope,
        },
        "causal_action_audit": causal_action_audit,
        "capacity": capacity,
        "data": {
            "train_action_rows": len(train_actions),
            "train_action_identities": int(train_actions["query_ik14"].nunique()),
            "train_action_formulas": int(train_actions["query_formula"].nunique()),
            "train_action_baseline_errors": int(np.sum(train_actions["baseline_rank"].astype(int).ne(1))),
            "train_action_baseline_correct": int(np.sum(train_actions["baseline_rank"].astype(int).eq(1))),
            "train_action_cells": (
                train_actions.groupby(
                    ["source", "family", "recipe_id"]
                    if args.action_selection == "materialized_routed"
                    else ["selector", "attenuation", "step"]
                ).size().rename("rows").reset_index().to_dict("records")
            ),
            "held_action_rows": len(held_action),
            "train_positive_pairs": len(train_positive),
            "train_positive_identities": int(train_positive["query_ik14"].nunique()) if not train_positive.empty else 0,
            "train_positive_formulas": int(train_positive["query_formula"].nunique()) if not train_positive.empty else 0,
            "held_positive_pairs": len(held_positive),
            "train_guided_rows": len(train_guided),
            "train_guided_identities": int(train_guided["query_ik14"].nunique()) if not train_guided.empty else 0,
            "train_guided_formulas": int(train_guided["query_formula"].nunique()) if not train_guided.empty else 0,
            "train_guided_policies": (
                train_guided.groupby(["guided_family", "guided_dose"])
                .size().rename("rows").reset_index().to_dict("records")
                if not train_guided.empty else []
            ),
            "train_guided_risk_rows": len(train_guided_risk),
            "train_guided_risk_identities": int(
                train_guided_risk["query_ik14"].nunique()
            ) if not train_guided_risk.empty else 0,
            "train_guided_risk_formulas": int(
                train_guided_risk["query_formula"].nunique()
            ) if not train_guided_risk.empty else 0,
            "train_guided_risk_kinds": (
                train_guided_risk["control_kind"].value_counts().astype(int).to_dict()
                if not train_guided_risk.empty else {}
            ),
            "held_guided_rows": len(held_guided),
            "held_queries": len(held_queries),
            "safety_queries": len(safety_examples),
            "pmt_unique_corrective_actions": (
                int(sum(example.sample_weight > 0 for example in action_examples))
                if args.pmt_arm != "none" else 0
            ),
            "pmt_unique_routed_actions": (
                len(action_examples) if args.pmt_arm != "none" else 0
            ),
            "pmt_unique_noncorrective_actions": (
                int(sum(example.sample_weight == 0 for example in action_examples))
                if args.pmt_arm != "none" else 0
            ),
            "pmt_total_action_exposures": (
                sum(len(values) for values in pmt_schedules)
                if pmt_schedules is not None else 0
            ),
            "pmt_action_exposures_by_policy": (
                pd.Series([
                    action_examples[index].policy
                    for schedule in pmt_schedules for index in schedule
                ]).value_counts().sort_index().astype(int).to_dict()
                if pmt_schedules is not None else {}
            ),
            "materialized_unique_corrective_actions": (
                len(action_examples)
                if args.action_selection == "materialized_routed" else 0
            ),
            "materialized_total_action_exposures": (
                int(signal_preserving_schedule_report.get("physical_action_exposures", 0))
                if signal_preserving_v2 else (
                    int(live_shared_v3_schedule_report.get(
                        "physical_action_exposures", 0,
                    )) if live_shared_v3 else (
                        sum(len(values) for values in materialized_schedules)
                        if materialized_schedules is not None else 0
                    )
                )
            ),
            "materialized_all_actions_seen_before_recycling": bool(
                materialized_schedules is not None
                or signal_preserving_schedule_report.get(
                    "all_unique_actions_exposed"
                ) is True
                or (
                    live_shared_v3
                    and live_shared_v3_schedule_report.get(
                        "unique_actions_exposed"
                    ) == live_shared_v3_schedule_report.get("action_count")
                )
            ),
            "materialized_action_sources": (
                sorted(set(train_actions["source"].astype(str)))
                if args.action_selection == "materialized_routed" else []
            ),
            "corrected_e4_base_action_rows": int(len(e4_base_action_examples)),
            "corrected_e4_base_identities": int(
                len({item.identity for item in e4_base_action_examples})
            ),
        },
        "zero_change_gate": zero_change_report,
        "guided_teacher_replay": guided_teacher_replay,
        "materialized_action_control": materialized_action_control_report,
        "materialized_clean_replay": materialized_clean_replay,
        "materialized_action_replay": materialized_action_replay,
        "branch_gradient_audit": branch_gradient_audit,
        "live_shared_v3_preinjection_gradient_audit": (
            live_shared_v3_preinjection_audit
        ),
        "optimizer_boundary_injection": (
            live_shared_v3_injector_report
            if live_shared_v3_injector is not None else (
                signal_preserving_injector_report
                if signal_preserving_injector is not None
                else e4_action_injector_report
            )
        ),
        "signal_preserving_v2_schedule": signal_preserving_schedule_report,
        "live_shared_v3_schedule": live_shared_v3_schedule_report,
        "corrected_e4_base_action_bank": e4_base_action_report,
        "effective_boundary_action_weight": effective_boundary_action_weight,
        "effective_guided_noise_weight": effective_guided_noise_weight,
        "held_clean": final_summary,
        "promotion_reference": promotion_reference,
        "promotion_diagnostics": promotion_diagnostics,
        "gates": gates,
        "pass_to_multifold": bool(all(gates.values())),
        "history": history,
        "contracts": {
            "shared_query_reference_encoder": True,
            "model_weights_changed": True,
            "last_transformer_blocks_and_official_head_trainable": True,
            "clean_and_augmented_raw_spectra_train_same_encoder": True,
            "direct_transfer_mode": args.direct_transfer_mode,
            "rank_reference_mode": args.rank_reference_mode,
            "official_action_targets_frozen_before_optimizer": bool(
                args.direct_transfer_mode == "official_action"
            ),
            "official_reference_anchors_training_only": bool(
                args.rank_reference_mode == "official"
            ),
            "real_cross_condition_positive_pairs_train_same_encoder": bool(args.positive_stream_weight > 0),
            "real_positive_guided_peak_noise_trains_same_encoder": bool(args.guided_noise_policy != "none"),
            "guided_queries_are_positive_deficit_official_errors": bool(
                args.guided_noise_policy not in {"none", "selected"}
                and args.guided_query_scope == "positive_deficit_errors"
            ),
            "guided_query_scope": args.guided_query_scope,
            "guided_action_cells_fixed_globally_before_training": bool(
                args.guided_noise_policy not in {"none", "selected"}
            ),
            "guided_transfer_mode": args.guided_transfer_mode,
            "guided_recurrence_recipe": {
                "reference_policy": "top3",
                "minimum_reference_prevalence": args.guided_recurrence_prevalence,
                "maximum_transferred_peaks": args.guided_recurrence_max_peaks,
                "dose": 0.50,
            },
            "guided_nonhistorical_recipe_formally_authorized": bool(
                guided_authorization_report or guided_crossfit_reports
            ),
            "guided_reference_policy": (
                "formula_outer_fold_excluded_per_query_selected_references"
                if args.guided_noise_policy == "selected"
                else (
                    "top3_by_frozen_mature_e8_embedding"
                    if guided_authorization_report else "historical_manifest"
                )
            ),
            "guided_action_outcomes_used_for_per_query_selection": bool(
                args.guided_noise_policy == "selected"
            ),
            "guided_action_specific_risk_controls_used": bool(
                args.guided_noise_policy == "selected"
                and args.guided_risk_control_ratio > 0
            ),
            "guided_teacher_target_mode": args.guided_teacher_target_mode,
            "positive_pair_selection_uses_model_outcome": False,
            "action_recipe_fixed_before_this_training_run": True,
            "causal_attribution_arm": args.causal_arm,
            "causal_arm_changes_only_action_view": bool(args.causal_arm != "legacy"),
            "matched_control_selection_uses_outcome": False,
            "causal_sampler_keys_arm_invariant": bool(args.causal_arm != "legacy"),
            "causal_candidate_references_arm_invariant": bool(args.causal_arm != "legacy"),
            "action_selection": args.action_selection,
            "training_only_outcome_mined_actions": bool(args.action_selection == "outcome_mined"),
            "action_outcomes_used_for_training_action_selection": bool(
                args.action_selection in {"outcome_mined", "materialized_routed"}
                or args.pmt_arm != "none"
            ),
            "materialized_raw_action_spectra_used": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_action_arm != "clean_duplicate"
            ),
            "materialized_action_arm": args.materialized_action_arm,
            "materialized_injection_mode": args.materialized_injection_mode,
            "materialized_corrective_actions_only": bool(
                args.action_selection == "materialized_routed"
            ),
            "materialized_strict_clean_wrong_action_top1_only": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode
                != "complete_panel_historical_e4"
            ),
            "materialized_selected_as_strict_corrective_in_frozen_e8_geometry": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode in {
                    "complete_panel_historical_e4", "e4_base_semantic_v2",
                    "e4_live_shared_v3",
                }
            ),
            "materialized_all_actions_exposed_before_recycling": bool(
                args.action_selection == "materialized_routed"
                and (
                    not live_shared_v3
                    or live_shared_v3_schedule_report.get("unique_actions_exposed")
                    == live_shared_v3_schedule_report.get("action_count")
                )
            ),
            "materialized_action_loss_is_historical_e4": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode in {
                    "one_best_e4", "complete_panel_historical_e4",
                    "e4_live_shared_v3",
                }
            ),
            "materialized_action_loss_is_balanced_multi_action_direct": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode == "multi_action_balanced"
            ),
            "materialized_all_strict_actions_preserved": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode in {
                    "multi_action_balanced", "complete_panel_historical_e4",
                    "e4_base_semantic_v2", "e4_live_shared_v3",
                }
            ),
            "materialized_satisfied_action_rank_gradient_gated": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode in {
                    "multi_action_balanced", "e4_base_semantic_v2",
                }
            ),
            "materialized_identity_family_equal_effective_dose": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode in {
                    "multi_action_balanced", "e4_base_semantic_v2",
                }
            ),
            "materialized_static_preclip_loss_scale": float(
                materialized_preclip_loss_scale
            ),
            "materialized_action_control_changes_only_action_view": bool(
                args.action_selection == "materialized_routed"
            ),
            "materialized_action_embedding_target_used": False,
            "materialized_action_margin_target_used": False,
            "optimizer_boundary_mode": args.optimizer_boundary_mode,
            "action_injector_v1_used": bool(e4_action_injector is not None),
            "signal_preserving_v2_used": bool(
                signal_preserving_injector is not None
            ),
            "live_shared_v3_used": bool(live_shared_v3_injector is not None),
            "live_shared_v3_clean_action_positive_negative_all_live": bool(
                live_shared_v3
                and live_shared_v3_preinjection_audit.get(
                    "clean_action_positive_negative_all_live"
                ) is True
            ),
            "live_shared_v3_every_audited_query_all_four_roles_live": bool(
                live_shared_v3
                and live_shared_v3_preinjection_audit.get(
                    "clean_action_positive_negative_live_for_every_audited_query"
                ) is True
            ),
            "live_shared_v3_all_four_roles_live_through_ranking_paths": bool(
                live_shared_v3
                and live_shared_v3_preinjection_audit.get(
                    "clean_action_positive_negative_ranking_paths_all_live"
                ) is True
                and live_shared_v3_preinjection_audit.get(
                    "ranking_paths_live_for_every_non_degenerate_audited_query"
                ) is True
            ),
            "live_shared_v3_targeted_shuffled_compared_before_injector": bool(
                live_shared_v3
                and live_shared_v3_preinjection_audit.get(
                    "targeted_and_shuffled_distinct_before_injector"
                ) is True
            ),
            "live_shared_v3_action_rank_not_gated": bool(live_shared_v3),
            "live_shared_v3_complete_e4_terms_preserved": bool(live_shared_v3),
            "live_shared_v3_query_equal_dose": bool(
                live_shared_v3
                and live_shared_v3_schedule_report.get(
                    "query_dose_equal_within_every_epoch"
                ) is True
            ),
            "live_shared_v3_same_query_actions_not_mixed": bool(
                live_shared_v3
                and live_shared_v3_schedule_report.get(
                    "same_query_actions_never_share_an_optimizer_step"
                ) is True
            ),
            "live_shared_v3_independent_optimizer_states": bool(
                live_shared_v3_injector is not None
            ),
            "live_shared_v3_exact_action_fraction_reached": bool(
                live_shared_v3_injector is not None
                and live_shared_v3_injector_report.get(
                    "exact_action_fraction_reached"
                ) is True
            ),
            "live_shared_v3_action_transmission_nonzero": bool(
                live_shared_v3_injector is not None
                and live_shared_v3_injector_report.get(
                    "action_transmission_nonzero"
                ) is True
            ),
            "historical_e4_fixed_r0_nine_cell_curriculum_used": bool(
                args.action_selection == "fixed"
                and args.policy == "curriculum"
                and tuple(FIXED_POLICY[args.policy])
                == tuple(FIXED_POLICY["curriculum"])
            ),
            "historical_e4_official_initialization_used": bool(
                (
                    args.action_selection == "fixed"
                    or (
                        args.action_selection == "materialized_routed"
                        and args.materialized_injection_mode
                        == "complete_panel_historical_e4"
                    )
                )
                and args.initial_student_checkpoint is None
            ),
            "historical_e4_official_cache_is_floor_and_preservation_target": bool(
                objective_reference_by_row is official_by_row
            ),
            "historical_e4_identity_balanced_four_views_used": bool(
                (
                    args.action_selection == "fixed"
                    or signal_preserving_v2
                    or live_shared_v3
                )
                and args.views_per_identity == 4
            ),
            "historical_e4_symmetric_shared_direct_loss_unmodified": bool(
                (
                    args.action_selection == "fixed"
                    or (
                        args.action_selection == "materialized_routed"
                        and args.materialized_injection_mode
                        in {
                            "complete_panel_historical_e4",
                            "e4_base_semantic_v2",
                            "e4_live_shared_v3",
                        }
                    )
                )
                and args.direct_transfer_mode == "symmetric"
                and args.rank_reference_mode == "shared"
                and math.isclose(materialized_preclip_loss_scale, 1.0)
            ),
            "injector_is_only_optimizer_boundary_change": bool(
                args.action_selection == "fixed"
                and e4_action_injector is not None
                and args.initial_student_checkpoint is None
                and args.positive_stream_weight == 0
                and args.guided_noise_policy == "none"
                and args.pmt_arm == "none"
                and not args.candidate_boundary_loss
            ),
            "hybrid_changes_only_action_supplier_and_optimizer_boundary": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode
                == "complete_panel_historical_e4"
                and e4_action_injector is not None
                and args.initial_student_checkpoint is None
                and args.positive_stream_weight == 0
                and args.guided_noise_policy == "none"
                and args.pmt_arm == "none"
                and not args.candidate_boundary_loss
                and objective_reference_by_row is official_by_row
                and math.isclose(materialized_preclip_loss_scale, 1.0)
            ),
            "complete_seven_source_action_panel_used_without_one_best_compression": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode in {
                    "complete_panel_historical_e4", "e4_base_semantic_v2",
                    "e4_live_shared_v3",
                }
                and materialized_action_control_report.get(
                    "selected_best_action_union_rows"
                ) == 32114
                and materialized_action_control_report.get(
                    "selected_strict_top1_corrective_queries"
                ) == 3482
            ),
            "v2_starts_from_frozen_e8_not_official_restart": bool(
                signal_preserving_v2 and args.initial_student_checkpoint is not None
            ),
            "v2_e8_initialization_is_floor_and_preservation_target": bool(
                signal_preserving_v2
                and objective_reference_by_row is training_anchor_by_row
            ),
            "v2_corrected_graph_e4_base_is_outcome_free": bool(
                signal_preserving_v2
                and e4_base_action_report.get("contracts", {}).get(
                    "action_outcomes_computed"
                ) is False
                and e4_base_action_report.get("contracts", {}).get(
                    "teacher_embedding_or_margin_used"
                ) is False
            ),
            "v2_e4_base_and_later_actions_share_frozen_e8_geometry": bool(
                signal_preserving_v2
                and args.initial_student_checkpoint is not None
                and e4_base_action_report.get("model_provenance", {}).get(
                    "initial_student_checkpoint_sha256"
                ) == sha256_file(args.initial_student_checkpoint)
                and materialized_action_report.get("model_provenance", {}).get(
                    "initial_student_checkpoint_sha256"
                ) == sha256_file(args.initial_student_checkpoint)
            ),
            "v2_later_actions_are_query_local_semantic_residual_only": bool(
                signal_preserving_v2
            ),
            "v2_later_semantic_clean_boundary_rank_active": bool(
                signal_preserving_v2
            ),
            "v2_later_satisfied_action_rank_gradient_gated": bool(
                signal_preserving_v2
            ),
            "v2_later_symmetric_clean_action_consistency_preserved": bool(
                signal_preserving_v2
            ),
            "v2_every_later_identity_has_four_distinct_optimizer_steps_per_epoch": bool(
                signal_preserving_v2
                and signal_preserving_schedule_report.get(
                    "each_later_identity_has_four_distinct_optimizer_opportunities_per_epoch"
                ) is True
            ),
            "v2_all_later_actions_exposed_with_family_local_coverage_first": bool(
                signal_preserving_v2
                and signal_preserving_schedule_report.get(
                    "all_unique_actions_exposed"
                ) is True
                and signal_preserving_schedule_report.get(
                    "recycling_only_after_per_identity_source_family_complete_coverage"
                ) is True
            ),
            "v2_later_source_family_equal_effective_dose": bool(
                signal_preserving_v2
                and signal_preserving_schedule_report.get(
                    "source_family_semantic_dose_equal_within_identity"
                ) is True
                and signal_preserving_schedule_report.get(
                    "every_source_family_present_on_every_identity_optimizer_step"
                ) is True
            ),
            "v2_every_optimizer_step_audited": bool(
                signal_preserving_injector is not None
                and int(signal_preserving_injector_report.get("steps", 0))
                == sum(int(record["steps"]) for record in history)
            ),
            "v2_signal_and_e4_retention_gate_passed": bool(
                signal_preserving_injector is not None
                and signal_preserving_injector_report.get("gate_passed") is True
            ),
            "complete_panel_unit_action_weights_without_identity_or_source_reweighting": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode
                == "complete_panel_historical_e4"
                and materialized_action_control_report.get(
                    "selected_actions_equal_unit_weight"
                ) is True
                and materialized_action_control_report.get(
                    "identity_family_effective_dose_normalized"
                ) is False
                and materialized_action_control_report.get(
                    "identity_effective_dose_equalized"
                ) is False
            ),
            "action_specific_to_clean_corrective_gradient_gate_passed": bool(
                args.action_selection == "materialized_routed"
                and args.materialized_injection_mode
                == "complete_panel_historical_e4"
                and branch_gradient_audit.get("gate_passed") is True
            ),
            "pure_e4_control_retrained": False,
            "action_injector_v1_receives_complete_historical_e4_gradient": bool(
                e4_action_injector is not None
            ),
            "action_injector_v1_action_selection_or_tensor_mutation": False,
            "action_injector_v1_historical_e4_loss_mutation": False,
            "action_injector_v1_every_optimizer_step_audited": bool(
                e4_action_injector is not None
                and int(e4_action_injector_report.get("steps", 0))
                == sum(int(record["steps"]) for record in history)
            ),
            "action_injector_v1_signal_and_safety_gate_passed": bool(
                e4_action_injector is not None
                and e4_action_injector_report.get("gate_passed") is True
            ),
            "corrected_metrics_use_exact_query_matvec": bool(
                corrected_query_adduct is not None
            ),
            "pmt_routes_mined_only_on_outer_train": bool(args.pmt_arm != "none"),
            "action_outcomes_used_in_loss_or_sample_weight": bool(
                args.guided_noise_policy == "selected"
                or args.pmt_arm == "paired_target"
            ),
            "pmt_arm": args.pmt_arm,
            "pmt_target_control_same_batch": bool(args.pmt_arm == "paired_target"),
            "pmt_harmful_target_weight_exact_zero": bool(args.pmt_arm != "none"),
            "pmt_all_corrective_actions_exposed_before_recycling": bool(
                args.pmt_arm != "none"
            ),
            "pmt_all_routed_actions_exposed_before_recycling": bool(
                args.pmt_arm != "none"
                and args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "pmt_control_has_identity_and_margin_floor": bool(
                args.pmt_arm == "paired_target" and not args.candidate_boundary_loss
            ),
            "pmt_M2_predictions_used": False,
            "pmt_P_actions_used": False,
            "candidate_boundary_matrix_preserved": bool(args.candidate_boundary_loss),
            "candidate_boundary_version": args.candidate_boundary_version,
            "candidate_boundary_uses_molecule_max": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "candidate_boundary_clean_primary_corrective_dose": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "candidate_boundary_query_normalized_multi_action_dose": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "candidate_boundary_query_shared_candidate_references": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "candidate_boundary_noncorrective_action_safety_only": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
                and args.lambda_boundary_action_safety > 0
            ),
            "candidate_boundary_scalar_teacher_used": False,
            "candidate_boundary_control_symmetrically_ranked": False,
            "candidate_boundary_control_gradient_stopped": bool(
                args.candidate_boundary_loss
            ),
            "candidate_boundary_live_topk_refresh": bool(
                args.candidate_boundary_loss and args.refresh_hard_negatives
            ),
            "candidate_boundary_formula_stratified_gradient_calibration": bool(
                args.candidate_boundary_loss
                and branch_gradient_audit.get("formula_microbatches", 0)
                >= args.boundary_calibration_formulas
            ),
            "candidate_boundary_action_not_weakened_by_calibration": bool(
                not args.candidate_boundary_loss
                or effective_boundary_action_weight >= 1.0
            ),
            "candidate_boundary_v2_calibration_may_downscale": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "identity_equal_action_weighting": bool(
                not args.candidate_boundary_loss
                or args.candidate_boundary_version != "v2_molecule_max"
            ),
            "query_equal_action_weighting": bool(
                args.candidate_boundary_loss
                and args.candidate_boundary_version == "v2_molecule_max"
            ),
            "action_views_per_identity_per_epoch": args.views_per_identity,
            "additional_error_views_per_identity_per_epoch": args.error_views_per_identity,
            "formula_held_out": True,
            "dropout_disabled_during_gradient_training": True,
            "inference_clean_spectrum_only": True,
            "teacher": (
                "frozen_official_raw_action_embedding"
                if args.direct_transfer_mode == "official_action"
                else (
                    "training_only_action_mining"
                    if args.action_selection == "outcome_mined"
                    else (
                        "outer_train_action_routing_only_no_teacher_target"
                        if args.action_selection == "materialized_routed" else (
                            "outer_fold_isolated_privileged_action_margin"
                            if args.guided_noise_policy == "selected" else (
                                "outer_train_current_geometry_target_control_advantage"
                                if args.pmt_arm == "paired_target" and not args.candidate_boundary_loss
                                else "forbidden"
                            )
                        )
                    )
                )
            ),
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "r0_report_sha256": (
                sha256_file(args.r0_dir / "report.json")
                if args.action_selection != "materialized_routed" else None
            ),
            "r0_actions_sha256": (
                sha256_file(args.r0_dir / "training_actions.csv.gz")
                if args.action_selection != "materialized_routed" else None
            ),
            "materialized_action_report_sha256": (
                sha256_file(args.materialized_action_dir / "report.json")
                if args.action_selection == "materialized_routed" else None
            ),
            "materialized_training_actions_sha256": (
                sha256_file(args.materialized_action_dir / "training_actions.csv.gz")
                if args.action_selection == "materialized_routed" else None
            ),
            "materialized_action_spectra_sha256": (
                sha256_file(args.materialized_action_dir / "action_spectra.npz")
                if args.action_selection == "materialized_routed" else None
            ),
            "corrected_e4_base_report_sha256": (
                sha256_file(args.e4_base_action_dir / "report.json")
                if (signal_preserving_v2 or live_shared_v3) else None
            ),
            "corrected_e4_base_actions_sha256": (
                sha256_file(args.e4_base_action_dir / "training_actions.csv.gz")
                if (signal_preserving_v2 or live_shared_v3) else None
            ),
            "outcome_action_report_sha256": (
                sha256_file(args.outcome_action_dir / "report.json")
                if args.action_selection == "outcome_mined" else None
            ),
            "outcome_action_manifest_sha256": (
                sha256_file(args.outcome_action_dir / "corrective_teacher_actions.csv.gz")
                if args.action_selection == "outcome_mined" else None
            ),
            "positive_report_sha256": (
                sha256_file(args.positive_manifest_dir / "report.json")
                if args.positive_stream_weight > 0 else None
            ),
            "positive_pairs_sha256": (
                sha256_file(args.positive_manifest_dir / "positive_pairs.csv.gz")
                if args.positive_stream_weight > 0 else None
            ),
            "guided_intensity_report_sha256": (
                sha256_file(args.guided_intensity_dir / "report.json")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_intensity_manifest_sha256": (
                sha256_file(args.guided_intensity_dir / "action_manifest.csv.gz")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_transfer_report_sha256": (
                sha256_file(args.guided_transfer_dir / "report.json")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_transfer_manifest_sha256": (
                sha256_file(args.guided_transfer_dir / "action_manifest.csv.gz")
                if args.guided_noise_policy not in {"none", "selected"} else None
            ),
            "guided_action_authorization_report_sha256": (
                sha256_file(args.guided_action_authorization_dir / "report.json")
                if args.guided_action_authorization_dir is not None else None
            ),
            "guided_reference_checkpoint_sha256": (
                sha256_file(args.guided_reference_checkpoint)
                if args.guided_reference_checkpoint is not None else None
            ),
            "initial_student_checkpoint_sha256": (
                sha256_file(args.initial_student_checkpoint)
                if args.initial_student_checkpoint is not None else None
            ),
            "pmt_manifest_report_sha256": (
                sha256_file(args.pmt_manifest_dir / "report.json")
                if args.pmt_arm != "none" else None
            ),
            "pmt_corrective_actions_sha256": (
                sha256_file(args.pmt_manifest_dir / "corrective_actions.csv.gz")
                if args.pmt_arm != "none" else None
            ),
            "guided_crossfit_report_sha256": (
                sha256_file(
                    args.guided_crossfit_root / f"fold_{args.outer_fold}" / "report.json"
                )
                if args.guided_noise_policy == "selected" else None
            ),
            "guided_crossfit_manifest_sha256": (
                sha256_file(
                    args.guided_crossfit_root / f"fold_{args.outer_fold}" / "selected_actions.csv.gz"
                )
                if args.guided_noise_policy == "selected" else None
            ),
            "guided_crossfit_risk_controls_sha256": (
                sha256_file(
                    args.guided_crossfit_root / f"fold_{args.outer_fold}" / "risk_controls.csv.gz"
                )
                if args.guided_noise_policy == "selected" else None
            ),
            "guided_crossfit_capacity_amendment_sha256": (
                sha256_file(
                    args.guided_crossfit_root
                    / f"fold_{args.outer_fold}" / "capacity_amendment.json"
                )
                if (
                    args.guided_noise_policy == "selected"
                    and (
                        args.guided_crossfit_root
                        / f"fold_{args.outer_fold}" / "capacity_amendment.json"
                    ).is_file()
                ) else None
            ),
            "graph_sha256": sha256_file(args.graph),
            "data_sha256": sha256_file(args.data),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "architecture_checkpoint_sha256": sha256_file(
                args.architecture_checkpoint
            ),
            "source_manifest_sha256": (
                sha256_file(args.source_manifest)
                if args.source_manifest is not None else None
            ),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "final_shared_encoder_sha256": sha256_file(final_checkpoint_path),
            "script_sha256": sha256_file(Path(__file__)),
            "action_injector_v1_sha256": (
                sha256_file(ROOT / "tasks/noise_action_injector_v1.py")
                if e4_action_injector is not None else None
            ),
            "e4_action_injector_v1_bridge_sha256": (
                sha256_file(ROOT / "tasks/noise_e4_action_injector_v1_bridge.py")
                if e4_action_injector is not None else None
            ),
            "e4_signal_preserving_hybrid_v2_sha256": (
                sha256_file(ROOT / "tasks/noise_e4_signal_preserving_hybrid_v2.py")
                if signal_preserving_injector is not None else None
            ),
            "e4_live_shared_hybrid_v3_sha256": (
                sha256_file(ROOT / "tasks/noise_e4_live_shared_hybrid_v3.py")
                if live_shared_v3_injector is not None else None
            ),
        },
        "claim_limit": (
            "held-formula development result for a directly fine-tuned shared embedding. "
            "Historical 4.93 pp and current 5.33 pp are best-action-union headroom, "
            "not promised clean-encoder weight gains."
        ),
    }
    # Path objects are not JSON serialisable.
    decision["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in decision["configuration"].items()
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
