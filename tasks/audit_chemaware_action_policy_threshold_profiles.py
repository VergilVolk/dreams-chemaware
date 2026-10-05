"""Evaluate validation-defined ChemAware action-policy operating profiles."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_conservative_action_policy import build_action_table
from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir", type=Path,
        default=ROOT / "data/validation/chemaware_conservative_action_policy_v1",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_action_policy_threshold_profiles_v1",
    )
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def select_validation_profiles(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    nonempty = [row for row in rows if int(row["selected"]) > 0]
    safety = max(
        rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), float(row["threshold"]),
        ),
    )
    recall_eligible = [
        row for row in nonempty
        if int(row["corrected_at_1"]) > 2 * int(row["introduced_at_1"])
    ]
    recall = max(
        recall_eligible,
        key=lambda row: (
            float(row["delta_recall1"]), int(row["risk_utility_at_1"]),
            -int(row["introduced_at_1"]), float(row["threshold"]),
        ),
    )
    precision_eligible = [row for row in nonempty if int(row["introduced_at_1"]) <= 1]
    precision = max(
        precision_eligible,
        key=lambda row: (
            int(row["corrected_at_1"]), int(row["risk_utility_at_1"]),
            float(row["threshold"]),
        ),
    )
    return {"safety": safety, "recall": recall, "precision": precision}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    source_report = json.loads((args.policy_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(args.policy_dir / "inner_policy.npz", allow_pickle=False) as loaded:
        policy = {key: np.asarray(loaded[key]) for key in loaded.files}
    actions = [
        (float(item["mass_beta"]), float(item["rule_beta"]))
        for item in source_report["method"]["actions"]
    ]
    profiles = select_validation_profiles(source_report["validation"]["primary_threshold_grid"])
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    cache = KernelCache(SimpleNamespace(**vars(args)), row_position, variants=("mass", "rule_response"))
    scored = score_queries(
        policy["query"], body, official, row_position, cache, ("mass", "rule_response"),
    )
    table = build_action_table(scored, actions)
    utility = (
        np.asarray(policy["predicted_benefit"], dtype=np.float64)
        - float(source_report["method"]["risk_penalty"])
        * np.asarray(policy["predicted_harm"], dtype=np.float64)
    )
    best_action = np.argmax(utility, axis=1)
    best_value = utility[np.arange(len(utility)), best_action]
    baseline = np.asarray(scored["old_rank"], dtype=np.int16)
    formula = np.asarray(policy["formula"]).astype(str)
    results = {}
    ranks = {}
    for offset, (name, validation_row) in enumerate(profiles.items()):
        threshold = float(validation_row["threshold"])
        active = best_value >= threshold
        rank = baseline.copy()
        rank[active] = table["rank"][np.arange(len(rank))[active], best_action[active]]
        ranks[name] = rank
        results[name] = {
            "selection_rule": {
                "chosen_on": "formula fold 2 validation only",
                "validation": validation_row,
            },
            "inner": retrieval(baseline, rank),
            "inner_selected": int(active.sum()),
            "inner_selected_formulas": int(len(np.unique(formula[active]))),
            "inner_formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
                formula, baseline, rank, args.bootstrap_draws, args.seed + 100 * offset,
            ),
        }
    report = {
        "status": "CHEMAWARE_ACTION_POLICY_THRESHOLD_PROFILES_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "validation-defined profiles evaluated on already-used inner fold; outer remains sealed",
        "claim_limit": (
            "Operating-profile sensitivity analysis of an existing development policy.  "
            "Profiles are selected from validation outcomes, never from inner outcomes."
        ),
        "profiles": results,
        "contracts": {
            "policy_predictions_reused_without_refit": True,
            "profile_rules_use_validation_only": True,
            "inner_outcomes_not_used_to_choose_threshold": True,
            "explicit_no_op_below_threshold": True,
            "outer_fold_untouched": True,
        },
        "provenance": {
            "policy_report_sha256": sha256(args.policy_dir / "report.json"),
            "policy_inner_sha256": sha256(args.policy_dir / "inner_policy.npz"),
            "manifest_sha256": sha256(args.manifest),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_policy_profiles_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_profile_ranks.npz", query=policy["query"], formula=formula,
            baseline_rank=baseline, **{f"{name}_rank": rank for name, rank in ranks.items()},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "profiles": results,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
