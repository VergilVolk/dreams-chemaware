#!/usr/bin/env python
"""Export candidate-level ChemAware utilities for conditional-null learning.

Unlike the legacy grand-fusion export, this file does not convert a policy
decision into a promoted pair score.  It preserves the continuous utility of
every candidate challenger under the aligned chemistry and under the three
deployment-safe counterfactual arms.  Labels are never opened.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries_truthblind
from chemaware_truthblind_candidate_core import predict_truthblind_policy_arms
from export_chemaware_v2_grand_fusion_scores import (
    _kernel_args, _load_policy, _token_state,
)
from conditional_null_energy_core import deterministic_keyed_derangements
from noise_final_core import CandidateGraph, sha256_file


POLICY_ARMS = {
    "correct": "candidate_utility_correct",
    "zero_contrast": "candidate_utility_zero",
    "reversed_contrast": "candidate_utility_reversed",
}


def _verify_policy_inputs(args: argparse.Namespace) -> dict[str, str]:
    """Bind the frozen policy to the exact inputs used for this replay."""
    report_path = args.policy_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    declared = report.get("provenance")
    if not isinstance(declared, dict):
        raise RuntimeError("ChemAware policy report has no input provenance")
    observed = {
        "manifest_sha256": sha256_file(args.manifest),
        "official_embeddings_sha256": sha256_file(
            args.token_dir / "official_embeddings_f32.npy"
        ),
        "rule_library_sha256": sha256_file(args.rule_library),
    }
    mismatch = {
        key: {"declared": declared.get(key), "observed": value}
        for key, value in observed.items()
        if declared.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "ChemAware policy inputs differ from the frozen replay provenance: "
            + json.dumps(mismatch, sort_keys=True)
        )
    return observed


def _embedding_norms(official: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.einsum(
        "ij,ij->i", official, official, dtype=np.float64, optimize=True,
    ))
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("ChemAware official token embeddings have invalid norms")
    return norms


def _normalized_global_for_query(
    raw_global: np.ndarray,
    body: dict[str, np.ndarray],
    query: int,
    row_position: dict[int, int],
    official_norms: np.ndarray,
) -> np.ndarray:
    """Normalize token-cache dot products for a diagnostic comparison only."""
    molecule_left, molecule_right = map(int, body["query_ptr"][query:query + 2])
    edge_left = int(body["molecule_ptr"][molecule_left])
    edge_right = int(body["molecule_ptr"][molecule_right])
    reference_rows = body["pair_candidate_row"][edge_left:edge_right]
    reference_positions = np.fromiter(
        (row_position[int(row)] for row in reference_rows),
        dtype=np.int64,
        count=len(reference_rows),
    )
    query_position = row_position[int(body["query_row"][query])]
    denominator = official_norms[reference_positions] * official_norms[query_position]
    raw = np.asarray(raw_global, dtype=np.float64)
    if raw.shape != denominator.shape:
        raise RuntimeError(f"official score shape drifted at query {query}")
    return (raw / denominator).astype(np.float32)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--rule-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-queries", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists() or args.batch_queries < 1:
        raise ValueError("output must be new and batch size positive")
    policy_args = argparse.Namespace(
        policy_dir=args.policy_dir,
        rule_library=args.rule_library,
        expected_policy_sha256="",
    )
    policy, replay, policy_hash = _load_policy(policy_args)
    policy_input_provenance = _verify_policy_inputs(args)
    if tuple(policy.get("control_rule_keys", ())) != (
        "rule_response_content_permuted",
        "rule_response_content_permuted_b",
        "rule_response_content_permuted_c",
    ):
        raise RuntimeError("conditional-null export requires the frozen three-null V2 policy")

    evidence_file = args.evidence / "evidence.npz"
    evidence_report = json.loads((args.evidence / "report.json").read_text(encoding="utf-8"))
    if evidence_report.get("status") != "NOISE_MSG_PAIR_EVIDENCE_COMPLETE":
        raise RuntimeError("input pair evidence is not frozen")
    with np.load(evidence_file, allow_pickle=False) as evidence:
        evidence_query_ptr = np.asarray(evidence["query_ptr"], dtype=np.int64)
        evidence_molecule_ptr = np.asarray(evidence["molecule_ptr"], dtype=np.int64)
        evidence_formula = np.asarray(evidence["query_formula"], dtype=str)
        expected_official = np.asarray(evidence["official_cosine"], dtype=np.float32)
    with np.load(args.manifest, allow_pickle=False) as manifest:
        body = {
            "query_row": np.asarray(manifest["query_row"], dtype=np.int64),
            "query_ptr": np.asarray(manifest["query_ptr"], dtype=np.int64),
            "molecule_ptr": np.asarray(manifest["molecule_ptr"], dtype=np.int64),
            "pair_candidate_row": np.asarray(manifest["pair_candidate_row"], dtype=np.int64),
            "molecule_ik14": np.asarray(manifest["molecule_ik14"], dtype=str),
            "molecule_label": np.asarray(manifest["molecule_label"], dtype=np.int8),
        }
        manifest_formula = np.asarray(manifest["query_formula"], dtype=str)
    graph = CandidateGraph(args.graph)
    if evidence_report.get("provenance", {}).get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("pair evidence was not built from the supplied candidate graph")
    if not (
        np.array_equal(evidence_query_ptr, body["query_ptr"])
        and np.array_equal(evidence_molecule_ptr, body["molecule_ptr"])
        and np.array_equal(evidence_formula, manifest_formula)
        and len(expected_official) == len(body["pair_candidate_row"])
        and np.array_equal(body["query_row"], graph.query_row)
        and np.array_equal(body["query_ptr"], graph.query_ptr)
        and np.array_equal(body["molecule_ptr"], graph.molecule_ptr)
        and np.array_equal(body["pair_candidate_row"], graph.pair_candidate_row)
        and np.array_equal(body["molecule_ik14"], graph.molecule_ik14)
        and np.array_equal(body["molecule_label"], graph.molecule_label)
        and np.array_equal(manifest_formula, graph.query_formula)
    ):
        raise RuntimeError("ChemAware manifest does not align with frozen evidence")

    rows, official, row_position = _token_state(args.token_dir)
    official_norms = _embedding_norms(official)
    variants = tuple(dict.fromkeys((
        "mass", str(policy["rule_key"]), *map(str, policy["control_rule_keys"]),
    )))
    cache = KernelCache(_kernel_args(args.token_dir, args.rule_library, replay), row_position, variants)
    molecule_count = int(body["query_ptr"][-1])
    utility = {name: np.zeros(molecule_count, dtype=np.float32) for name in POLICY_ARMS.values()}
    valid = np.zeros(molecule_count, dtype=bool)

    query_count = len(body["query_row"])
    maximum_raw_official_error = 0.0
    maximum_normalized_official_error = 0.0
    official_raw_numeric_mismatch_queries = 0
    official_numeric_mismatch_queries = 0
    official_molecule_top1_agreement_queries = 0
    for start in range(0, query_count, args.batch_queries):
        stop = min(query_count, start + args.batch_queries)
        query_ids = np.arange(start, stop, dtype=np.int64)
        scored = score_queries_truthblind(
            query_ids, body, official, row_position, cache, variants,
        )
        predictions = predict_truthblind_policy_arms(
            policy, scored, [scored for _ in policy["control_rule_keys"]],
        )
        for local, query in enumerate(range(start, stop)):
            molecule_left = int(body["query_ptr"][query])
            molecule_right = int(body["query_ptr"][query + 1])
            edge_left = int(body["molecule_ptr"][molecule_left])
            edge_right = int(body["molecule_ptr"][molecule_right])
            observed = np.asarray(scored["global"][local], dtype=np.float32)
            expected_block = expected_official[edge_left:edge_right]
            maximum_raw_official_error = max(
                maximum_raw_official_error,
                float(np.max(np.abs(observed - expected_block), initial=0.0)),
            )
            if not np.allclose(observed, expected_block, rtol=2e-5, atol=2e-6):
                official_raw_numeric_mismatch_queries += 1
            alignment_values = _normalized_global_for_query(
                observed, body, query, row_position, official_norms,
            )
            maximum_normalized_official_error = max(
                maximum_normalized_official_error,
                float(np.max(np.abs(alignment_values - expected_block), initial=0.0)),
            )
            if not np.allclose(
                alignment_values, expected_block, rtol=2e-5, atol=2e-6,
            ):
                official_numeric_mismatch_queries += 1
            local_pointer = (
                body["molecule_ptr"][molecule_left:molecule_right + 1] - edge_left
            ).astype(np.int64)
            observed_molecule = np.maximum.reduceat(
                alignment_values, local_pointer[:-1],
            )
            expected_molecule = np.maximum.reduceat(
                expected_block, local_pointer[:-1],
            )
            official_molecule_top1_agreement_queries += int(
                np.argmax(observed_molecule) == np.argmax(expected_molecule)
            )
            correct = predictions["correct"]
            row_valid = np.asarray(correct["valid"][local], dtype=bool)
            proposed = np.asarray(correct["proposed_candidate"][local], dtype=np.int64)
            for slot in np.flatnonzero(row_valid):
                candidate = int(proposed[slot])
                if not (0 <= candidate < molecule_right - molecule_left):
                    raise RuntimeError(f"candidate slot drifted at query {query}")
                destination = molecule_left + candidate
                if valid[destination]:
                    raise RuntimeError(f"duplicate candidate utility at query {query}")
                valid[destination] = True
                for arm, field in POLICY_ARMS.items():
                    arm_prediction = predictions[arm]
                    if not bool(np.asarray(arm_prediction["valid"])[local, slot]):
                        raise RuntimeError(f"null-arm validity drifted at query {query}")
                    value = float(np.asarray(arm_prediction["utility"])[local, slot])
                    if not np.isfinite(value):
                        raise RuntimeError(f"non-finite utility at query {query}")
                    utility[field][destination] = value
        print(f"[conditional-null ChemAware] {stop:,}/{query_count:,} queries", flush=True)

    # The baseline candidate has no challenger action by construction and is
    # therefore an exact chemical no-op.  Every non-baseline candidate should
    # carry one valid utility row.
    expected_valid = molecule_count - query_count
    if int(valid.sum()) != expected_valid:
        raise RuntimeError(f"candidate utility coverage drifted: {valid.sum()} != {expected_valid}")
    utility["candidate_utility_rotated"] = deterministic_keyed_derangements(
        utility["candidate_utility_correct"],
        body["molecule_ik14"],
        body["query_row"].astype(str),
        body["query_ptr"],
        seeds=(20261007,),
    )[0].astype(np.float32)
    arrays = {
        **utility,
        "chem_valid": valid,
        "query_key": body["query_row"].astype(str),
        "candidate_key": body["molecule_ik14"],
        "candidate_reference_count": np.diff(body["molecule_ptr"]).astype(np.int32),
        "query_candidate_count": np.diff(body["query_ptr"]).astype(np.int32),
        "query_has_near": graph.query_has_near,
        "evidence_sha256": np.asarray(sha256_file(evidence_file)),
        "graph_sha256": np.asarray(sha256_file(args.graph)),
        "manifest_sha256": np.asarray(sha256_file(args.manifest)),
        "policy_sha256": np.asarray(policy_hash),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".npz", dir=args.output.parent)
    os.close(fd)
    temporary = Path(raw)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "CHEMAWARE_CONDITIONAL_NULL_EVIDENCE_COMPLETE",
        "labels_opened": False,
        "queries": query_count,
        "molecules": molecule_count,
        "valid_challengers": int(valid.sum()),
        "null_arms": ["zero_contrast", "reversed_contrast", "candidate_id_deranged"],
        "candidate_order_invariant": True,
        "official_alignment_semantics": (
            "candidate axis is established by exact graph hash and row/pointer arrays; "
            "token raw-dot policy space is preserved; normalized historical official "
            "scores are diagnostic only and cannot invalidate an already exact axis"
        ),
        "maximum_raw_official_error": maximum_raw_official_error,
        "maximum_normalized_official_error": maximum_normalized_official_error,
        "official_raw_numeric_mismatch_queries": (
            official_raw_numeric_mismatch_queries
        ),
        "official_numeric_mismatch_queries": official_numeric_mismatch_queries,
        "official_molecule_top1_agreement_queries": (
            official_molecule_top1_agreement_queries
        ),
        "official_molecule_top1_agreement_rate": (
            official_molecule_top1_agreement_queries / query_count
        ),
        "baseline_candidate_semantics": "chemical no-op utility zero and chem_valid false",
        "evidence_sha256": sha256_file(evidence_file),
        "policy_sha256": policy_hash,
        "verified_policy_input_provenance": policy_input_provenance,
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
