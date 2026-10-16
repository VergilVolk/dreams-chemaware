#!/usr/bin/env python
"""Export aligned continuous ChemAware V2/null utilities on frozen GNPS panels.

This is a label-blind adapter around the already frozen V2 policy and GNPS
kernel cache.  It does not export the thresholded action.  For every candidate
challenger it preserves the continuous utility under correct chemistry, zero
contrast, reversed contrast, and truth-blind candidate mismatch.  Downstream
code forms chemical differences, which cancels the shared nuisance term.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries_truthblind
from chemaware_truthblind_candidate_core import predict_truthblind_policy_arms
from export_chemaware_v2_grand_fusion_scores import (
    _body_from_panel,
    _kernel_args,
    _load_policy,
    _token_state,
)
from gnps_pair_score_cache import sha256_file


ARM_FIELDS = {
    "correct": "utility_correct",
    "zero_contrast": "utility_zero",
    "reversed_contrast": "utility_reversed",
    "candidate_rotated_truthblind": "utility_candidate_mismatch",
}


def parse_panel(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("panel must be NAME=PATH")
    name, path = value.split("=", 1)
    if name not in ("identity_disjoint", "formula_disjoint"):
        raise argparse.ArgumentTypeError(f"unsupported panel: {name}")
    return name, Path(path)


def export_panel(
    *, name: str, path: Path, policy: dict, replay: dict, policy_hash: str,
    token_dir: Path, rule_library: Path, output: Path, batch_queries: int,
) -> dict:
    body = _body_from_panel(path)
    with np.load(path, allow_pickle=False) as panel:
        query_ids = np.asarray(panel["query_ik14"]).astype("U")
        candidate_ids = np.asarray(panel["molecule_ik14"]).astype("U")
        query_formula = np.asarray(panel["query_formula"]).astype("U")
        near_query = np.asarray(panel["near_query"], dtype=bool)
    rows, official, row_position = _token_state(token_dir)
    del rows
    variants = tuple(dict.fromkeys((
        "mass", str(policy["rule_key"]), *map(str, policy["control_rule_keys"]),
    )))
    cache = KernelCache(_kernel_args(token_dir, rule_library, replay), row_position, variants)
    molecule_count = int(body["query_ptr"][-1])
    utilities = {
        field: np.zeros(molecule_count, dtype=np.float32)
        for field in ARM_FIELDS.values()
    }
    availability = np.zeros(molecule_count, dtype=bool)
    maximum_nuisance_error = 0.0

    for start in range(0, len(body["query_row"]), batch_queries):
        stop = min(len(body["query_row"]), start + batch_queries)
        query_index = np.arange(start, stop, dtype=np.int64)
        scored = score_queries_truthblind(
            query_index, body, official, row_position, cache, variants,
        )
        predictions = predict_truthblind_policy_arms(
            policy, scored, [scored for _ in policy["control_rule_keys"]],
        )
        primary = predictions["correct"]
        for arm_name, arm in predictions.items():
            if not (
                np.array_equal(arm["valid"], primary["valid"])
                and np.array_equal(arm["proposed_candidate"], primary["proposed_candidate"])
                and np.array_equal(arm["baseline_candidate"], primary["baseline_candidate"])
            ):
                raise RuntimeError(f"candidate alignment drifted in {arm_name}")
            valid = np.asarray(primary["valid"], dtype=bool)
            for nuisance_name in ("benefit_base", "harm_base"):
                delta = np.abs(
                    np.asarray(arm[nuisance_name], dtype=np.float64)[valid]
                    - np.asarray(primary[nuisance_name], dtype=np.float64)[valid]
                )
                if len(delta):
                    maximum_nuisance_error = max(maximum_nuisance_error, float(np.max(delta)))

        for local, query in enumerate(range(start, stop)):
            left = int(body["query_ptr"][query])
            right = int(body["query_ptr"][query + 1])
            row_valid = np.asarray(primary["valid"][local], dtype=bool)
            proposed = np.asarray(primary["proposed_candidate"][local], dtype=np.int64)
            if int(row_valid.sum()) != right - left - 1:
                raise RuntimeError(f"V2 challenger coverage drifted at query {query}")
            for slot in np.flatnonzero(row_valid):
                candidate = int(proposed[slot])
                destination = left + candidate
                if not (left <= destination < right) or availability[destination]:
                    raise RuntimeError(f"candidate destination drifted at query {query}")
                availability[destination] = True
                for arm_name, field in ARM_FIELDS.items():
                    value = float(np.asarray(predictions[arm_name]["utility"])[local, slot])
                    if not np.isfinite(value):
                        raise RuntimeError(f"non-finite {arm_name} utility at query {query}")
                    utilities[field][destination] = np.float32(value)
        print(f"[continuous ChemAware] {name}: {stop:,}/{len(body['query_row']):,}", flush=True)

    expected = molecule_count - len(body["query_row"])
    if int(availability.sum()) != expected:
        raise RuntimeError(f"candidate availability drifted: {availability.sum()} != {expected}")
    if maximum_nuisance_error > 1e-7:
        raise RuntimeError(
            f"V2 null arms do not share nuisance predictions: {maximum_nuisance_error}"
        )
    np.savez_compressed(
        output,
        schema=np.asarray("gnps_chemaware_continuous_null_evidence_v1"),
        panel=np.asarray(name),
        query_ptr=np.asarray(body["query_ptr"], dtype=np.int64),
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        query_formula=query_formula,
        near_query=near_query,
        availability=availability,
        **utilities,
        policy_sha256=np.asarray(policy_hash),
        panel_sha256=np.asarray(sha256_file(path)),
    )
    return {
        "panel": name,
        "queries": int(len(body["query_row"])),
        "candidates": molecule_count,
        "available_challengers": int(availability.sum()),
        "unavailable_baseline_candidates": int((~availability).sum()),
        "maximum_null_nuisance_error": maximum_nuisance_error,
        "panel_sha256": sha256_file(path),
        "output_sha256": sha256_file(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--rule-library", type=Path, required=True)
    parser.add_argument("--panel", action="append", type=parse_panel, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-queries", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.batch_queries < 1:
        raise ValueError("output must be new and batch size positive")
    panel_map = dict(args.panel)
    if set(panel_map) != {"identity_disjoint", "formula_disjoint"}:
        raise ValueError("both primary GNPS panels must be exported together")
    policy_args = argparse.Namespace(
        policy_dir=args.policy_dir,
        rule_library=args.rule_library,
        expected_policy_sha256="",
    )
    policy, replay, policy_hash = _load_policy(policy_args)
    if tuple(policy.get("control_rule_keys", ())) != (
        "rule_response_content_permuted",
        "rule_response_content_permuted_b",
        "rule_response_content_permuted_c",
    ):
        raise RuntimeError("continuous export requires the frozen three-null V2 policy")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gnps_chem_continuous.", dir=args.output.parent))
    try:
        panels = {}
        for name, path in panel_map.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            destination = staging / f"continuous_chem_{name}.npz"
            panels[name] = export_panel(
                name=name, path=path, policy=policy, replay=replay,
                policy_hash=policy_hash, token_dir=args.token_dir,
                rule_library=args.rule_library, output=destination,
                batch_queries=args.batch_queries,
            )
        report = {
            "status": "GNPS_CHEMAWARE_CONTINUOUS_NULL_EVIDENCE_COMPLETE",
            "dataset_role": "development_consumed",
            "labels_opened": False,
            "thresholded_action_exported": False,
            "utility_difference_semantics": (
                "correct-minus-null cancels the shared V2 nuisance prediction; "
                "unavailable official-baseline candidates remain explicit no-ops"
            ),
            "policy_sha256": policy_hash,
            "token_cache": str(args.token_dir),
            "panels": panels,
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
